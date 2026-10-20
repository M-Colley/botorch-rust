//! `HigherOrderGP` (phase 10c): the Kronecker-structured exact GP of
//! BoTorch's `botorch.models.higher_order_gp`, its MLL, posteriors and
//! Matheron samples (see `botorch_core::models::higher_order_gp`).
//!
//! The Python layer applies the outcome transform (`FlattenedStandardize`)
//! and draws the initial latent coordinates (torch's generator); targets
//! arrive standardized and flattened (`n x S`).

use std::sync::{Arc, RwLock};

use botorch_core::fit::{FitOptions, FitStatus, Fittable};
use botorch_core::models::higher_order_gp::{
    HigherOrderGP, HigherOrderGPOptions, HogpFit, LatentPrior,
};
use botorch_core::optim::fit::fit_gpytorch_mll_torch;
use botorch_core::optim::minimize::TorchMinimizeOptions;
use botorch_core::optim::{AdamOptions, ExpMaStoppingCriterion};
use botorch_core::transforms::input::ChainedInputTransform;
use botorch_core::Model;
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{dimension_error, value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::models::PyModel;
use crate::specs::PyLikelihoodSpec;
use crate::threads;

type PriorInfo = (usize, f64, u8, Option<f64>, Option<String>);

fn status_name(s: FitStatus) -> &'static str {
    match s {
        FitStatus::Success => "SUCCESS",
        FitStatus::Stopped => "STOPPED",
        FitStatus::Failure => "FAILURE",
    }
}

/// BoTorch's `HigherOrderGP` (unbatched).
#[pyclass(frozen, name = "HigherOrderGP", module = "botorch_rs._botorch_rs")]
pub struct PyHigherOrderGP {
    inner: RwLock<Arc<HigherOrderGP>>,
}

impl PyHigherOrderGP {
    fn get(&self) -> Arc<HigherOrderGP> {
        Arc::clone(&self.inner.read().unwrap_or_else(|e| e.into_inner()))
    }

    fn set(&self, model: HigherOrderGP) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(model);
    }

    fn wrap(model: HigherOrderGP) -> Self {
        Self {
            inner: RwLock::new(Arc::new(model)),
        }
    }
}

#[pymethods]
impl PyHigherOrderGP {
    /// A model from the transformed training inputs `train_x` (`n x d'`),
    /// the standardized flattened targets `train_y` (`n x S`), the target
    /// shape, the numbers of latent coordinates, the kernel trees (on the
    /// inputs, then on the latents of every output dimension), the noise
    /// spec, the latent priors' covariances (`latent_init="gp"`), the
    /// fitted input chain and the `FlattenedStandardize` statistics, and the
    /// flat raw parameters.
    #[new]
    #[pyo3(signature = (
        train_x,
        train_y,
        target_shape,
        latent_dims,
        dim,
        covar_trees,
        noise,
        raw,
        latent_priors=None,
        learn_latent_pars=true,
        input_chain=None,
        input_chain_training=None,
        input_chain_prefixes=None,
        raw_train_x=None,
        outcome=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        train_x: Array,
        train_y: Array,
        target_shape: Vec<usize>,
        latent_dims: Vec<usize>,
        dim: usize,
        covar_trees: Vec<PyRef<'_, PyKernelNode>>,
        noise: PyRef<'_, PyLikelihoodSpec>,
        raw: Vec<f64>,
        latent_priors: Option<Vec<Option<Array>>>,
        learn_latent_pars: bool,
        input_chain: Option<Vec<Bound<'_, PyAny>>>,
        input_chain_training: Option<Vec<bool>>,
        input_chain_prefixes: Option<Vec<String>>,
        raw_train_x: Option<Array>,
        outcome: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let x = train_x.matrix("train_X")?;
        let y = train_y.matrix("train_Y")?;
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
        let latent_priors = match latent_priors {
            Some(ps) => ps
                .into_iter()
                .map(|p| {
                    p.map(|a| {
                        a.matrix("covariance")
                            .map(|c| LatentPrior { covariance: c })
                    })
                    .transpose()
                })
                .collect::<PyResult<Vec<_>>>()?,
            None => Vec::new(),
        };
        let options = HigherOrderGPOptions {
            covar_modules: covar_trees.iter().map(|t| t.inner.clone()).collect(),
            noise: noise.inner,
            latent_priors,
            learn_latent_pars,
            input_transform,
            outcome,
        };
        let model = HigherOrderGP::new(x, y, target_shape, latent_dims, dim, Some(raw), options)
            .into_py_result()?;
        Ok(Self::wrap(model))
    }

    /// The model handle for posteriors and acquisition functions (outputs:
    /// the flattened target tensor).
    fn model_handle(&self) -> PyModel {
        PyModel { inner: self.get() }
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

    /// `(name, size)` of every raw parameter.
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.get().parameter_names()
    }

    /// The optimizer bounds of the raw parameters.
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.get().raw_parameter_bounds()
    }

    /// The batch shape of the modules (unbatched).
    #[getter]
    fn aug_batch_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    /// The noise variance (`[noise]`).
    #[getter]
    fn noise(&self) -> Vec<f64> {
        vec![self.get().noise()]
    }

    /// Sets the noise variance.
    fn set_noise(&self, value: Array) -> PyResult<()> {
        if value.0.len() != 1 {
            return Err(dimension_error("the noise has shape (1,)"));
        }
        let mut m = (*self.get()).clone();
        let mut raw = m.raw_parameters().to_vec();
        raw[0] = m
            .options()
            .noise
            .noise_constraint
            .inverse_transform(value.0[0]);
        m.set_raw_parameters(&raw).into_py_result()?;
        self.set(m);
        Ok(())
    }

    /// The MLL at the current (or given) raw parameters.
    #[pyo3(signature = (raw=None))]
    fn mll(&self, py: Python<'_>, raw: Option<Vec<f64>>) -> PyResult<ArrayOut> {
        let m = self.get();
        let raw = raw.unwrap_or_else(|| m.raw_parameters().to_vec());
        threads::run(py, || m.mll_value_and_grad(&raw, false))
            .into_py_result()
            .map(|r| (vec![r.0], Vec::new()))
    }

    /// The MLL and its gradient at the current (or given) raw parameters.
    #[pyo3(signature = (raw=None))]
    fn mll_value_and_grad(
        &self,
        py: Python<'_>,
        raw: Option<Vec<f64>>,
    ) -> PyResult<(ArrayOut, Vec<f64>)> {
        let m = self.get();
        let raw = raw.unwrap_or_else(|| m.raw_parameters().to_vec());
        threads::run(py, || m.mll_value_and_grad(&raw, true))
            .into_py_result()
            .map(|(value, gradient)| ((vec![value], Vec::new()), gradient))
    }

    /// `fit_gpytorch_mll` (L-BFGS-B with retries); returns `(loss,
    /// iterations, evaluations, status, message, attempts)`.
    #[pyo3(signature = (maxcor=None, pgtol=None, ftol=None, maxiter=None, maxfun=None, maxls=None, max_attempts=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        maxcor: Option<usize>,
        pgtol: Option<f64>,
        ftol: Option<f64>,
        maxiter: Option<usize>,
        maxfun: Option<usize>,
        maxls: Option<usize>,
        max_attempts: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<(f64, usize, usize, &'static str, String, usize)> {
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
        let mut f = HogpFit {
            model: (*self.get()).clone(),
        };
        let r = threads::run(py, || f.fit_mll(options)).into_py_result()?;
        self.set(f.model);
        Ok((
            r.loss,
            r.iterations,
            r.evaluations,
            status_name(r.status),
            r.message,
            r.attempts,
        ))
    }

    /// `fit_gpytorch_mll_torch` (Adam); returns `(step, fval, status,
    /// runtime, message)`.
    #[pyo3(signature = (mask=None, bounds=None, step_limit=None, stopping=None, timeout_sec=None, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=false, maximize=false))]
    #[allow(clippy::too_many_arguments)]
    fn fit_torch(
        &self,
        py: Python<'_>,
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
        let mut f = HogpFit {
            model: (*self.get()).clone(),
        };
        let m = &mut f;
        let res = threads::run(py, || {
            fit_gpytorch_mll_torch(m, mask.as_deref(), bounds.as_deref(), &options, None)
        })
        .into_py_result()?;
        self.set(f.model);
        Ok((
            res.step,
            res.fval,
            res.status.value(),
            res.runtime,
            res.message.clone(),
        ))
    }

    /// `sample_all_priors(model)` (in place).
    #[pyo3(signature = (seed=None))]
    fn sample_all_priors(&self, seed: Option<u64>) {
        use rand::SeedableRng;
        let mut m = (*self.get()).clone();
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            seed.unwrap_or_else(botorch_core::utils::sampling::random_seed),
        );
        let raw = m.sample_all_priors(m.raw_parameters(), &mut rng);
        if m.set_raw_parameters(&raw).is_ok() {
            self.set(m);
        }
    }

    /// Posterior means and variances (`q x S` each) at the raw-space points
    /// `x` (`q x d`), in the original (`original=True`) or standardized
    /// outcome space.
    #[pyo3(signature = (x, observation_noise=false, original=true))]
    fn mean_variance(
        &self,
        py: Python<'_>,
        x: Array,
        observation_noise: bool,
        original: bool,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let x = x.matrix("X")?;
        let m = self.get();
        let (mean, var) = threads::run(py, || {
            m.mean_and_variance_in(&x, observation_noise, original)
        })
        .into_py_result()?;
        Ok((matrix_out(&mean).into(), matrix_out(&var).into()))
    }

    /// Matheron samples (`S_mc x q·S`) at `x` (`q x d`) from base samples
    /// (`S_mc x ((n + q)·S + n·S)`).
    #[pyo3(signature = (x, base_samples, original=true))]
    fn samples(
        &self,
        py: Python<'_>,
        x: Array,
        base_samples: Array,
        original: bool,
    ) -> PyResult<ArrayBuf> {
        let x = x.matrix("X")?;
        let b = base_samples.matrix("base_samples")?;
        let m = self.get();
        let s = threads::run(py, || m.rsample_in(&x, &b, original)).into_py_result()?;
        Ok(matrix_out(&s).into())
    }

    #[pyo3(signature=(x, mean_adjoint, variance_adjoint, original=true))]
    fn moment_vjp(
        &self,
        py: Python<'_>,
        x: Array,
        mean_adjoint: Array,
        variance_adjoint: Array,
        original: bool,
    ) -> PyResult<ArrayBuf> {
        let x = x.matrix("X")?;
        let mean = mean_adjoint.matrix("mean_adjoint")?;
        let variance = variance_adjoint.matrix("variance_adjoint")?;
        let model = self.get();
        threads::run(py, || {
            model.mean_variance_vjp_in(&x, &mean, &variance, original)
        })
        .map(|g| matrix_out(&g).into())
        .into_py_result()
    }

    #[pyo3(signature=(x, base_samples, adjoint, original=true))]
    fn samples_vjp(
        &self,
        py: Python<'_>,
        x: Array,
        base_samples: Array,
        adjoint: Array,
        original: bool,
    ) -> PyResult<ArrayBuf> {
        let x = x.matrix("X")?;
        let base = base_samples.matrix("base_samples")?;
        let adjoint = adjoint.matrix("adjoint")?;
        let model = self.get();
        threads::run(py, || model.rsample_vjp_in(&x, &base, &adjoint, original))
            .map(|g| matrix_out(&g).into())
            .into_py_result()
    }

    /// The model conditioned on transformed inputs `x_tf` (`n' x d'`) and
    /// standardized targets `y_tf` (`n' x S`).
    fn condition_on_observations(&self, x_tf: Array, y_tf: Array) -> PyResult<Self> {
        let x = x_tf.matrix("X")?;
        let y = y_tf.matrix("Y")?;
        let m = self
            .get()
            .condition_on_observations(&x, &y)
            .into_py_result()?;
        Ok(Self::wrap(m))
    }

    /// The transformed training inputs.
    fn train_inputs(&self) -> ArrayOut {
        matrix_out(self.get().train_inputs())
    }

    /// The standardized training targets (`n x S`).
    fn train_targets(&self) -> ArrayOut {
        matrix_out(self.get().train_targets())
    }

    /// The raw input dimension.
    #[getter]
    fn dim(&self) -> usize {
        Model::dim(&*self.get())
    }

    /// The number of training points.
    #[getter]
    fn num_train(&self) -> usize {
        self.get().train_inputs().nrows()
    }

    /// An independent copy (for `deepcopy`).
    fn copy(&self) -> Self {
        Self {
            inner: RwLock::new(self.get()),
        }
    }
}

/// Registers the classes of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHigherOrderGP>()?;
    m.add_function(wrap_pyfunction!(higher_order_latent_sample_py, m)?)?;
    Ok(())
}

/// Apply the GP latent prior's Cholesky factor to supplied normal draws.
/// Drawing the random values remains in Python so the caller's seed is respected.
#[pyfunction]
fn higher_order_latent_sample_py(covariance: Array, normal: Array) -> PyResult<ArrayOut> {
    let covariance = covariance.matrix("covariance")?;
    let normal = normal.matrix("normal")?;
    if covariance.nrows() != covariance.ncols() || covariance.nrows() != normal.nrows() {
        return Err(dimension_error(
            "the latent covariance must be square and match the normal draws",
        ));
    }
    let root = botorch_core::linalg::psd_safe_cholesky(&covariance).into_py_result()?;
    // GPyTorch samples r vectors of length s and then reshapes them to s x r.
    let samples = root.l() * normal;
    // Explicitly flatten the transpose in row-major order.
    let data = (0..samples.ncols())
        .flat_map(|j| (0..samples.nrows()).map(move |i| (i, j)))
        .map(|(i, j)| samples[(i, j)])
        .collect::<Vec<_>>();
    Ok((data, vec![samples.nrows(), samples.ncols()]))
}
