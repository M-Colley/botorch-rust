//! Approximate GP models (phase 10c): `SingleTaskVariationalGP` with its
//! variational ELBO, and the pivoted-Cholesky inducing point allocation of
//! `botorch.models.utils.inducing_point_allocators`.
//!
//! The Python layer applies the input and outcome transforms to the
//! training data (as BoTorch does at construction), allocates the inducing
//! points and passes everything in the transformed spaces; the model keeps
//! the fitted input transform (eval mode) and the `Standardize` statistics
//! for its posteriors. Posteriors and acquisition functions use the generic
//! model handle ([`PyModel`]).

use std::sync::{Arc, RwLock};

use botorch_core::acquisition::{ScalarizedModel, ScalarizedPosteriorTransform};
use botorch_core::fit::{FitOptions, FitStatus, Fittable};
use botorch_core::models::approximate_gp::{
    ElboFit, SingleTaskVariationalGP, SingleTaskVariationalGPOptions, VariationalLikelihood,
    VariationalMean,
};
use botorch_core::models::gp::ConstantMeanSpec;
use botorch_core::models::inducing_point_allocators::{
    expected_improvement_quality, pivoted_cholesky_init,
};
use botorch_core::optim::fit::fit_gpytorch_mll_torch;
use botorch_core::optim::minimize::TorchMinimizeOptions;
use botorch_core::optim::{AdamOptions, ExpMaStoppingCriterion};
use botorch_core::transforms::input::ChainedInputTransform;
use botorch_core::transforms::Standardize;
use botorch_core::{Matrix, Model, Vector};
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayOut};
use crate::errors::{dimension_error, value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::models::PyModel;
use crate::specs::{PyConstraint, PyLikelihoodSpec, PyPrior};
use crate::threads;

/// `(loss, iterations, evaluations, status, message, attempts)`.
type FitSummary = (f64, usize, usize, &'static str, String, usize);

type PriorInfo = (usize, f64, u8, Option<f64>, Option<String>);

fn status_name(s: FitStatus) -> &'static str {
    match s {
        FitStatus::Success => "SUCCESS",
        FitStatus::Stopped => "STOPPED",
        FitStatus::Failure => "FAILURE",
    }
}

/// BoTorch's `SingleTaskVariationalGP` (see
/// `botorch_core::models::approximate_gp`).
#[pyclass(
    frozen,
    name = "SingleTaskVariationalGP",
    module = "botorch_rs._botorch_rs"
)]
pub struct PySingleTaskVariationalGP {
    inner: RwLock<Arc<SingleTaskVariationalGP>>,
}

impl PySingleTaskVariationalGP {
    fn get(&self) -> Arc<SingleTaskVariationalGP> {
        Arc::clone(&self.inner.read().unwrap_or_else(|e| e.into_inner()))
    }

    fn set(&self, model: SingleTaskVariationalGP) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(model);
    }

    /// The offset and size of the raw parameter `name`.
    fn param_range(&self, name: &str) -> Option<std::ops::Range<usize>> {
        let mut off = 0;
        for (n, size) in self.get().parameter_names() {
            if n == name {
                return Some(off..off + size);
            }
            off += size;
        }
        None
    }

    fn param(&self, name: &str) -> Option<Vec<f64>> {
        let r = self.param_range(name)?;
        Some(self.get().raw_parameters()[r].to_vec())
    }

    fn set_param(&self, name: &str, values: &[f64]) -> PyResult<()> {
        let r = self
            .param_range(name)
            .ok_or_else(|| value_error(format!("the model has no parameter {name:?}")))?;
        if r.len() != values.len() {
            return Err(dimension_error(format!(
                "{name} has {} values, got {}",
                r.len(),
                values.len()
            )));
        }
        let mut m = (*self.get()).clone();
        let mut raw = m.raw_parameters().to_vec();
        raw[r].copy_from_slice(values);
        m.set_raw_parameters(&raw).into_py_result()?;
        self.set(m);
        Ok(())
    }
}

#[pymethods]
impl PySingleTaskVariationalGP {
    /// A model from the transformed training inputs `train_x` (`n x d'`),
    /// the standardized targets `train_y` (`n x m`, optional), the inducing
    /// points `inducing` (`M x d'`), the kernel tree `covar_tree`, the mean
    /// (`mean="constant"` with `mean_prior` / `mean_constraint`, or
    /// `"zero"`), the likelihood (`likelihood="gaussian"` or `"multitask"`
    /// with the noise spec `noise`), the fitted input chain (eval mode) and
    /// the `Standardize` statistics, and the flat raw parameters `raw`
    /// (default: the initial values).
    #[new]
    #[pyo3(signature = (
        train_x,
        train_y,
        inducing,
        num_outputs,
        dim,
        covar_tree,
        covar_batched,
        noise,
        likelihood="gaussian",
        mean="constant",
        mean_prior=None,
        mean_constraint=None,
        mean_batched=false,
        learn_inducing_points=true,
        jitter=1e-6,
        input_chain=None,
        input_chain_training=None,
        input_chain_prefixes=None,
        raw_train_x=None,
        outcome_means=None,
        outcome_stdvs=None,
        outcome_min_stdv=1e-8,
        raw=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        train_x: Array,
        train_y: Option<Array>,
        inducing: Array,
        num_outputs: usize,
        dim: usize,
        covar_tree: PyRef<'_, PyKernelNode>,
        covar_batched: bool,
        noise: PyRef<'_, PyLikelihoodSpec>,
        likelihood: &str,
        mean: &str,
        mean_prior: Option<PyRef<'_, PyPrior>>,
        mean_constraint: Option<PyRef<'_, PyConstraint>>,
        mean_batched: bool,
        learn_inducing_points: bool,
        jitter: f64,
        input_chain: Option<Vec<Bound<'_, PyAny>>>,
        input_chain_training: Option<Vec<bool>>,
        input_chain_prefixes: Option<Vec<String>>,
        raw_train_x: Option<Array>,
        outcome_means: Option<Vec<f64>>,
        outcome_stdvs: Option<Vec<f64>>,
        outcome_min_stdv: f64,
        raw: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let x = train_x.matrix("train_X")?;
        let y = train_y.map(|y| y.matrix("train_Y")).transpose()?;
        let z = inducing.matrix("inducing_points")?;
        let likelihood = match likelihood {
            "gaussian" => VariationalLikelihood::Gaussian(noise.inner),
            "multitask" => VariationalLikelihood::Multitask(noise.inner),
            other => return Err(value_error(format!("unknown likelihood {other:?}"))),
        };
        let mean_module = match mean {
            "constant" => VariationalMean::Constant(ConstantMeanSpec {
                prior: mean_prior.map(|p| p.inner),
                constraint: mean_constraint.map(|c| c.inner),
            }),
            "zero" => VariationalMean::Zero,
            other => return Err(value_error(format!("unknown mean module {other:?}"))),
        };
        // The chain is fitted to the raw training inputs (learning e.g. the
        // bounds of a `Normalize`, as BoTorch's `transform_inputs` in train
        // mode does at construction) and then used in eval mode.
        let input_transform: Option<ChainedInputTransform> = match input_chain {
            Some(steps) => {
                let mut chain = crate::input_perturbation::chain_of(
                    &steps,
                    input_chain_training,
                    input_chain_prefixes,
                )?;
                let raw_x = raw_train_x
                    .ok_or_else(|| value_error("an input chain needs `raw_train_x`"))?
                    .matrix("train_X")?;
                chain.fit(&raw_x).into_py_result()?;
                Some(chain)
            }
            None => None,
        };
        let outcome_transform = match (outcome_means, outcome_stdvs) {
            (Some(m), Some(s)) => {
                Some(Standardize::fitted(m, s, outcome_min_stdv).into_py_result()?)
            }
            (None, None) => None,
            _ => return Err(value_error("outcome_means and outcome_stdvs go together")),
        };
        let options = SingleTaskVariationalGPOptions {
            covar_module: covar_tree.inner.clone(),
            covar_batched,
            mean_module,
            mean_batched,
            likelihood,
            learn_inducing_points,
            jitter,
            input_transform,
            outcome_transform,
        };
        let model = SingleTaskVariationalGP::new(x, y, z, num_outputs, dim, raw, options)
            .into_py_result()?;
        Ok(Self {
            inner: RwLock::new(Arc::new(model)),
        })
    }

    /// The model handle for posteriors and acquisition functions, seen
    /// through the posterior transform `(weights, offset)` if given.
    #[pyo3(signature = (transform=None))]
    fn model_handle(&self, transform: Option<(Vec<f64>, f64)>) -> PyResult<PyModel> {
        let model: Arc<dyn Model> = self.get();
        let inner = match transform {
            Some((w, offset)) => {
                let t = ScalarizedPosteriorTransform::new(Vector::from_vec(w), offset)
                    .into_py_result()?;
                ScalarizedModel::wrap(model, Some(t)).into_py_result()?
            }
            None => model,
        };
        Ok(PyModel { inner })
    }

    /// The flat raw parameters.
    fn raw_parameters(&self) -> Vec<f64> {
        self.get().raw_parameters().to_vec()
    }

    /// Sets the flat raw parameters.
    fn set_raw_parameters(&self, values: Vec<f64>) -> PyResult<()> {
        let mut m = (*self.get()).clone();
        m.set_raw_parameters(&values).into_py_result()?;
        self.set(m);
        Ok(())
    }

    /// `(name, size)` of every raw parameter (BoTorch's names in the ELBO).
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.get().parameter_names()
    }

    /// The optimizer bounds of the raw parameters.
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.get().raw_parameter_bounds()
    }

    /// The current inducing points (`M x d'`).
    fn inducing_points(&self) -> ArrayOut {
        matrix_out(&self.get().inducing_points())
    }

    /// Sets the inducing points (`M x d'`).
    fn set_inducing_points(&self, z: Array) -> PyResult<()> {
        let z = z.matrix("inducing_points")?;
        let mut m = (*self.get()).clone();
        m.set_inducing_points(&z).into_py_result()?;
        self.set(m);
        Ok(())
    }

    /// The likelihood's noise values (`[noise]` or `[task_noises..., noise]`).
    fn likelihood_values(&self) -> Vec<f64> {
        self.get().likelihood_values()
    }

    /// The kernel matrix of output `output` at the points `x` (`n x d'`,
    /// transformed space).
    fn kernel_matrix(&self, py: Python<'_>, x: Array, output: usize) -> PyResult<ArrayOut> {
        let x = x.matrix("X")?;
        let model = self.get();
        threads::run(py, || -> botorch_core::Result<ArrayOut> {
            let k = model.kernel(output)?;
            Ok(matrix_out(&k.covariance_symmetric(&x)?))
        })
        .into_py_result()
    }

    /// The latent posterior `q(f)` at transformed points `x` (`q x d'`):
    /// the means (`q x m`), the covariance of every output (`m x q x q`) and
    /// the marginal variances (`q x m`, GPyTorch's `variance` with the lazy
    /// `diag=True` kernel values), in the transformed outcome space.
    fn latent(&self, py: Python<'_>, x: Array) -> PyResult<(ArrayOut, ArrayOut, ArrayOut)> {
        let x = x.matrix("X")?;
        let model = self.get();
        let (parts, vars) = threads::run(py, || -> botorch_core::Result<_> {
            Ok((model.latent_posterior(&x)?, model.latent_variances(&x)?))
        })
        .into_py_result()?;
        let q = x.nrows();
        let m = parts.len();
        let mut mean = vec![0.0; q * m];
        let mut var = vec![0.0; q * m];
        let mut cov = Vec::with_capacity(m * q * q);
        for (t, (mu, sigma)) in parts.iter().enumerate() {
            for i in 0..q {
                mean[i * m + t] = mu[i];
                var[i * m + t] = vars[t][i];
            }
            cov.extend(matrix_out(sigma).0);
        }
        Ok(((mean, vec![q, m]), (cov, vec![m, q, q]), (var, vec![q, m])))
    }

    /// The noise of a Gaussian likelihood (`[noise]`), or the global noise
    /// of a multitask likelihood.
    #[getter]
    fn noise(&self) -> Vec<f64> {
        let v = self.get().likelihood_values();
        vec![*v.last().expect("a likelihood has a noise")]
    }

    /// Sets the noise of a Gaussian likelihood (a single value).
    fn set_noise(&self, value: Array) -> PyResult<()> {
        let model = self.get();
        let VariationalLikelihood::Gaussian(spec) = model.options().likelihood else {
            return Err(value_error("set the task noises of a multitask likelihood"));
        };
        if value.0.len() != 1 {
            return Err(dimension_error(format!(
                "the noise has shape (1,), got a value of shape {:?}",
                value.1
            )));
        }
        self.set_param(
            "likelihood.noise_covar.raw_noise",
            &[spec.noise_constraint.inverse_transform(value.0[0])],
        )
    }

    /// `(task_noises, noise)` of a multitask likelihood.
    fn noises(&self) -> (Vec<f64>, f64) {
        let mut v = self.get().likelihood_values();
        let global = v.pop().expect("a likelihood has a noise");
        (v, global)
    }

    /// The raw constants of the mean (one per output, or one shared).
    #[getter]
    fn constant(&self) -> Vec<f64> {
        self.param("model.mean_module.raw_constant")
            .unwrap_or_default()
    }

    /// Sets the constants from a value broadcastable to their shape.
    fn set_constant(&self, value: Array) -> PyResult<()> {
        let current = self
            .param("model.mean_module.raw_constant")
            .ok_or_else(|| value_error("the model has a ZeroMean"))?;
        let k = current.len();
        let new: Vec<f64> = match value.0.len() {
            1 => vec![value.0[0]; k],
            n if n == k => value.0.clone(),
            _ => {
                return Err(dimension_error(format!(
                    "raw_constant has {k} values, got a value of shape {:?}",
                    value.1
                )))
            }
        };
        self.set_param("model.mean_module.raw_constant", &new)
    }

    /// The batch shape of the model's modules (`[m]` for several outputs).
    #[getter]
    fn aug_batch_shape(&self) -> Vec<usize> {
        let m = botorch_core::Model::num_outputs(&*self.get());
        if m > 1 {
            vec![m]
        } else {
            Vec::new()
        }
    }

    /// The ELBO terms `(log_likelihood, kl_divergence, log_prior)` at the
    /// transformed inputs `x` (`n x d'`) and standardized targets `y` (`n x
    /// m`), at the raw parameters `raw` (the current ones if `None`).
    #[pyo3(signature = (x, y, num_data, beta=1.0, raw=None))]
    fn elbo(
        &self,
        py: Python<'_>,
        x: Array,
        y: Array,
        num_data: f64,
        beta: f64,
        raw: Option<Vec<f64>>,
    ) -> PyResult<(f64, f64, f64)> {
        let x = x.matrix("X")?;
        let y = y.matrix("target")?;
        let model = self.get();
        let t = threads::run(py, || {
            model.elbo_terms(raw.as_deref(), &x, &y, num_data, beta)
        })
        .into_py_result()?;
        Ok((t.log_likelihood, t.kl_divergence, t.log_prior))
    }

    /// The ELBO on the training data and its gradient with respect to the
    /// raw parameters (the current ones if `raw` is `None`).
    #[pyo3(signature = (num_data, beta=1.0, raw=None))]
    fn elbo_value_and_grad(
        &self,
        py: Python<'_>,
        num_data: f64,
        beta: f64,
        raw: Option<Vec<f64>>,
    ) -> PyResult<(f64, Vec<f64>)> {
        let model = self.get();
        let raw = raw.unwrap_or_else(|| model.raw_parameters().to_vec());
        let (t, g) = threads::run(py, || model.elbo_value_and_grad(&raw, num_data, beta))
            .into_py_result()?;
        Ok((t.value(), g))
    }

    /// `fit_gpytorch_mll` on the ELBO with L-BFGS-B and BoTorch's retries
    /// (prior samples of the kernel and mean parameters).
    #[pyo3(signature = (num_data, beta=1.0, maxcor=None, pgtol=None, ftol=None, maxiter=None, maxfun=None, maxls=None, max_attempts=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        num_data: f64,
        beta: f64,
        maxcor: Option<usize>,
        pgtol: Option<f64>,
        ftol: Option<f64>,
        maxiter: Option<usize>,
        maxfun: Option<usize>,
        maxls: Option<usize>,
        max_attempts: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<FitSummary> {
        let d = FitOptions::default();
        let options = FitOptions {
            maxcor: maxcor.unwrap_or(d.maxcor),
            pgtol: pgtol.unwrap_or(d.pgtol),
            ftol: ftol.unwrap_or(d.ftol),
            maxiter: maxiter.unwrap_or(d.maxiter),
            maxfun: maxfun.unwrap_or(d.maxfun),
            maxls: maxls.unwrap_or(d.maxls),
            max_attempts: max_attempts.unwrap_or(d.max_attempts),
            seed: seed.unwrap_or(d.seed),
        };
        let mut model = ElboFit {
            model: (*self.get()).clone(),
            num_data,
            beta,
        };
        let r = threads::run(py, || model.fit_mll(options)).into_py_result()?;
        self.set(model.model);
        Ok((
            r.loss,
            r.iterations,
            r.evaluations,
            status_name(r.status),
            r.message,
            r.attempts,
        ))
    }

    /// `fit_gpytorch_mll_torch` on the ELBO (Adam with BoTorch's
    /// `ExpMAStoppingCriterion`), one attempt; returns
    /// `(step, fval, status, runtime, message)`.
    #[pyo3(signature = (num_data, beta=1.0, mask=None, bounds=None, step_limit=None, stopping=None, timeout_sec=None, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=false, maximize=false))]
    #[allow(clippy::too_many_arguments)]
    fn fit_torch(
        &self,
        py: Python<'_>,
        num_data: f64,
        beta: f64,
        mask: Option<Vec<bool>>,
        bounds: Option<Vec<(f64, f64)>>,
        step_limit: Option<usize>,
        stopping: Option<(usize, bool, usize, f64, f64)>,
        timeout_sec: Option<f64>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
        maximize: bool,
    ) -> PyResult<PriorInfo> {
        let options = TorchMinimizeOptions {
            adam: AdamOptions {
                lr,
                betas,
                eps,
                weight_decay,
                amsgrad,
                maximize,
            },
            step_limit,
            timeout_sec,
            stopping_criterion: stopping
                .map(|s| ExpMaStoppingCriterion::new(s.0, s.1, s.2, s.3, s.4)),
        };
        let mut model = ElboFit {
            model: (*self.get()).clone(),
            num_data,
            beta,
        };
        let res = threads::run(py, || {
            fit_gpytorch_mll_torch(
                &mut model,
                mask.as_deref(),
                bounds.as_deref(),
                &options,
                None,
            )
        })
        .into_py_result()?;
        self.set(model.model);
        Ok((
            res.step,
            res.fval,
            res.status.value(),
            res.runtime,
            res.message.clone(),
        ))
    }

    /// `sample_all_priors(mll.model)` (kernel and mean priors), in place.
    #[pyo3(signature = (seed=None))]
    fn sample_all_priors(&self, seed: Option<u64>) {
        use rand::SeedableRng;
        let model = (*self.get()).clone();
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            seed.unwrap_or_else(botorch_core::utils::sampling::random_seed),
        );
        let raw = model.sample_all_priors(model.raw_parameters(), &mut rng);
        let mut m = model;
        if m.set_raw_parameters(&raw).is_ok() {
            self.set(m);
        }
    }

    /// The number of outputs.
    #[getter]
    fn num_outputs(&self) -> usize {
        botorch_core::Model::num_outputs(&*self.get())
    }

    /// The raw input dimension.
    #[getter]
    fn dim(&self) -> usize {
        botorch_core::Model::dim(&*self.get())
    }

    /// The number of inducing points.
    #[getter]
    fn num_inducing(&self) -> usize {
        self.get().num_inducing()
    }

    /// The transformed training inputs.
    fn train_inputs(&self) -> ArrayOut {
        matrix_out(self.get().train_inputs())
    }

    /// The standardized training targets (`None` without them).
    fn train_targets(&self) -> Option<ArrayOut> {
        self.get().train_targets().map(matrix_out)
    }

    /// An independent copy (for `deepcopy`).
    fn copy(&self) -> Self {
        Self {
            inner: RwLock::new(self.get()),
        }
    }
}

/// BoTorch's `_pivoted_cholesky_init`: the indices of the selected rows for
/// the kernel matrix `kernel_matrix` (`n x n`) and the quality scores.
#[pyfunction]
#[pyo3(signature = (kernel_matrix, max_length, quality_scores, epsilon=1e-6))]
pub fn pivoted_cholesky_init_py(
    kernel_matrix: Array,
    max_length: usize,
    quality_scores: Vec<f64>,
    epsilon: f64,
) -> PyResult<Vec<usize>> {
    let k: Matrix = kernel_matrix.matrix("kernel_matrix")?;
    if !k.is_square() {
        return Err(dimension_error("kernel_matrix must be square"));
    }
    pivoted_cholesky_init(&k, max_length, &quality_scores, epsilon).into_py_result()
}

/// The quality scores of BoTorch's `ExpectedImprovementQualityFunction`
/// from posterior means and variances.
#[pyfunction]
pub fn expected_improvement_quality_py(
    mean: Vec<f64>,
    variance: Vec<f64>,
    maximize: bool,
) -> PyResult<Vec<f64>> {
    expected_improvement_quality(&mean, &variance, maximize).into_py_result()
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySingleTaskVariationalGP>()?;
    m.add_function(wrap_pyfunction!(pivoted_cholesky_init_py, m)?)?;
    m.add_function(wrap_pyfunction!(expected_improvement_quality_py, m)?)?;
    Ok(())
}
