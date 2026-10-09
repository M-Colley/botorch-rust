//! `SingleTaskGP` (`botorch.models.SingleTaskGP`): construction, posteriors,
//! the exact marginal log likelihood, fitting and hyperparameter access.
//!
//! The model is stored as an `Arc` snapshot behind a lock: acquisition
//! functions share the snapshot that is current when they are created, while
//! fitting or setting hyperparameters installs a new snapshot (copy on write),
//! so the GIL can be released during heavy work without data races.

use std::sync::{Arc, RwLock};

use botorch_core::linalg::CovarianceRoot;
use botorch_core::posterior::MIN_VARIANCE;
use botorch_core::{
    fit_gpytorch_mll, ExactMarginalLogLikelihood, FitOptions, FitStatus, Likelihood, Matrix,
    OutcomeTransformChoice, SingleTaskGP, SingleTaskGPOptions,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::arrays::{matrix_out, stack_out, vector_out, Array, ArrayOut};
use crate::errors::{dimension_error, unsupported, value_error, IntoPyResult};
use crate::specs::{PyKernelSpec, PyLikelihoodSpec};
use crate::threads;
use crate::transforms::{PyNormalize, PyStandardize};

/// Rows per parallel work item of the marginal posterior.
const MARGINAL_CHUNK: usize = 256;

/// The fit summary returned to Python: `(loss, iterations, evaluations,
/// status, message, attempts)`.
type FitSummary = (f64, usize, usize, &'static str, String, usize);

/// A single-output exact GP with BoTorch's `SingleTaskGP` defaults.
#[pyclass(frozen, name = "SingleTaskGP", module = "botorch_rs._botorch_rs")]
pub struct PySingleTaskGP {
    inner: RwLock<Arc<SingleTaskGP>>,
}

impl PySingleTaskGP {
    /// The current model.
    pub fn snapshot(&self) -> Arc<SingleTaskGP> {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn replace(&self, model: SingleTaskGP) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(model);
    }

    /// Applies `f` to a copy of the model and installs the copy on success.
    fn update<F>(&self, f: F) -> PyResult<()>
    where
        F: FnOnce(&mut SingleTaskGP) -> botorch_core::Result<()>,
    {
        let mut model = (*self.snapshot()).clone();
        f(&mut model).into_py_result()?;
        self.replace(model);
        Ok(())
    }
}

/// Marginal posterior means and variances of the rows of `x` (each row its
/// own `q = 1` posterior), optionally including the observation noise.
fn marginals(
    model: &SingleTaskGP,
    x: &Matrix,
    observation_noise: bool,
) -> botorch_core::Result<(Vec<f64>, Vec<f64>)> {
    let n = x.nrows();
    let starts: Vec<usize> = (0..n).step_by(MARGINAL_CHUNK).collect();
    let parts = starts
        .par_iter()
        .map(|&s| {
            let len = MARGINAL_CHUNK.min(n - s);
            model.posterior_marginals(&x.rows(s, len).into_owned())
        })
        .collect::<botorch_core::Result<Vec<_>>>()?;
    let mut mean = Vec::with_capacity(n);
    let mut var = Vec::with_capacity(n);
    for (m, v) in parts {
        mean.extend(m.iter());
        var.extend(v.iter());
    }
    if observation_noise {
        let noise = model.observation_noise();
        for (i, v) in var.iter_mut().enumerate() {
            if *v > MIN_VARIANCE {
                *v += noise;
            } else {
                // The latent variance was clamped: recompute the noisy variance
                // exactly (BoTorch clamps after adding the noise).
                *v = model
                    .posterior(&x.rows(i, 1).into_owned(), true)?
                    .variance()[0];
            }
        }
    }
    Ok((mean, var))
}

#[pymethods]
impl PySingleTaskGP {
    /// Builds a model from `train_X` (`n x d`), `train_Y` (`n x 1`) and
    /// optional observed noise variances `train_Yvar` (`n x 1`).
    ///
    /// `outcome_transform=None` with `standardize_by_default=True` is BoTorch's
    /// `DEFAULT` (a fresh `Standardize`); with `False` no outcome transform is
    /// used.
    #[new]
    #[pyo3(signature = (
        train_x,
        train_y,
        train_yvar=None,
        input_transform=None,
        outcome_transform=None,
        standardize_by_default=true,
        covar_module=None,
        likelihood=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        train_x: Array,
        train_y: Array,
        train_yvar: Option<Array>,
        input_transform: Option<PyRef<'_, PyNormalize>>,
        outcome_transform: Option<PyRef<'_, PyStandardize>>,
        standardize_by_default: bool,
        covar_module: Option<PyRef<'_, PyKernelSpec>>,
        likelihood: Option<PyRef<'_, PyLikelihoodSpec>>,
    ) -> PyResult<Self> {
        let train_x = train_x.matrix("train_X")?;
        let train_y = train_y.matrix("train_Y")?;
        let options = SingleTaskGPOptions {
            train_yvar: train_yvar.map(|v| v.matrix("train_Yvar")).transpose()?,
            input_transform: input_transform.map(|t| t.for_model()),
            outcome_transform: match (outcome_transform, standardize_by_default) {
                (Some(st), _) => OutcomeTransformChoice::Standardize(st.inner.clone()),
                (None, true) => OutcomeTransformChoice::Default,
                (None, false) => OutcomeTransformChoice::None,
            },
            covar_module: covar_module.map(|c| c.inner),
            likelihood: likelihood.map(|l| l.inner),
            ..Default::default()
        };
        let model = SingleTaskGP::new(train_x, train_y, options).into_py_result()?;
        Ok(Self {
            inner: RwLock::new(Arc::new(model)),
        })
    }

    /// A deep copy of the model.
    fn copy(&self) -> Self {
        Self {
            inner: RwLock::new(Arc::new((*self.snapshot()).clone())),
        }
    }

    // ------------------------------------------------------------------
    // Data and configuration
    // ------------------------------------------------------------------

    /// Input dimension `d`.
    #[getter]
    fn dim(&self) -> usize {
        self.snapshot().dim()
    }

    /// Number of training points `n`.
    #[getter]
    fn num_train(&self) -> usize {
        self.snapshot().num_train()
    }

    /// Whether the observation noise is fixed (`train_Yvar` given).
    #[getter]
    fn is_fixed_noise(&self) -> bool {
        matches!(self.snapshot().likelihood(), Likelihood::FixedNoise { .. })
    }

    /// Training inputs in the original space (`n x d`).
    fn train_inputs(&self) -> ArrayOut {
        matrix_out(self.snapshot().train_inputs())
    }

    /// Training targets in the original space (`n x 1`).
    fn train_targets(&self) -> ArrayOut {
        matrix_out(self.snapshot().train_targets())
    }

    /// Observed noise variances in the original space (`n x 1`), if given.
    fn train_yvar(&self) -> Option<ArrayOut> {
        self.snapshot().train_yvar().map(matrix_out)
    }

    /// Training inputs after the input transform (`n x d`).
    fn transformed_train_inputs(&self) -> ArrayOut {
        matrix_out(self.snapshot().transformed_train_inputs())
    }

    /// Training targets after the outcome transform (`n`), i.e. BoTorch's
    /// `model.train_targets`.
    fn transformed_train_targets(&self) -> ArrayOut {
        vector_out(self.snapshot().transformed_train_targets())
    }

    /// The fitted input transform, if any.
    fn input_transform(&self) -> Option<PyNormalize> {
        self.snapshot()
            .input_transform()
            .map(|tf| PyNormalize::fitted(tf.clone(), tf.learn_bounds()))
    }

    /// The fitted outcome transform, if any.
    fn outcome_transform(&self) -> Option<PyStandardize> {
        self.snapshot()
            .outcome_transform()
            .map(|st| PyStandardize { inner: st.clone() })
    }

    /// The covariance-module configuration.
    fn covar_module(&self) -> PyKernelSpec {
        PyKernelSpec {
            inner: *self.snapshot().covar_module(),
        }
    }

    /// The homoskedastic likelihood configuration (`None` for fixed noise).
    fn likelihood_spec(&self) -> Option<PyLikelihoodSpec> {
        match self.snapshot().likelihood() {
            Likelihood::Gaussian(spec) => Some(PyLikelihoodSpec { inner: *spec }),
            Likelihood::FixedNoise { .. } => None,
        }
    }

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------

    /// BoTorch parameter names and sizes in flat order.
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.snapshot().parameter_names()
    }

    /// The raw hyperparameters in flat order (BoTorch's `named_parameters()`).
    fn raw_parameters(&self) -> Vec<f64> {
        self.snapshot().raw_parameters().as_slice().to_vec()
    }

    /// Sets the raw hyperparameters from a flat vector.
    fn set_raw_parameters(&self, values: Vec<f64>) -> PyResult<()> {
        self.update(|m| m.set_raw_parameters(&values))
    }

    /// Optimizer bounds of the raw hyperparameters in flat order.
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.snapshot().raw_parameter_bounds()
    }

    /// The lengthscales (length `d`).
    #[getter]
    fn lengthscale(&self) -> Vec<f64> {
        self.snapshot().lengthscale().as_slice().to_vec()
    }

    /// The outputscale (`None` without `ScaleKernel`).
    #[getter]
    fn outputscale(&self) -> Option<f64> {
        self.snapshot().outputscale()
    }

    /// The likelihood noise variance in the transformed outcome space
    /// (GPyTorch's `likelihood.noise`): one value, or one per training point
    /// for fixed noise.
    #[getter]
    fn noise(&self) -> Vec<f64> {
        self.snapshot().likelihood_noise().as_slice().to_vec()
    }

    /// The constant mean (transformed outcome space).
    #[getter]
    fn constant(&self) -> f64 {
        self.snapshot().constant()
    }

    /// Sets the lengthscales (one value for all dimensions, or `d` values).
    fn set_lengthscale(&self, values: Vec<f64>) -> PyResult<()> {
        let model = self.snapshot();
        let d = model.dim();
        let values = match values.len() {
            1 => vec![values[0]; d],
            n if n == d => values,
            n => {
                return Err(dimension_error(format!(
                    "expected 1 or {d} lengthscales, got {n}"
                )))
            }
        };
        let constraint = model.covar_module().lengthscale_constraint;
        self.update(|m| {
            let mut h = m.hyperparameters().clone();
            for (r, v) in h.raw_lengthscale.iter_mut().zip(&values) {
                *r = constraint.inverse_transform(*v);
            }
            m.set_hyperparameters(h)
        })
    }

    /// Sets the outputscale of the `ScaleKernel`.
    fn set_outputscale(&self, value: f64) -> PyResult<()> {
        let Some(scale) = self.snapshot().covar_module().scale else {
            return Err(unsupported("The covariance module has no outputscale."));
        };
        self.update(|m| {
            let mut h = m.hyperparameters().clone();
            h.raw_outputscale = Some(scale.outputscale_constraint.inverse_transform(value));
            m.set_hyperparameters(h)
        })
    }

    /// Sets the (homoskedastic) noise variance in the transformed space.
    fn set_noise(&self, value: f64) -> PyResult<()> {
        let Likelihood::Gaussian(spec) = *self.snapshot().likelihood() else {
            return Err(unsupported(
                "The noise of a fixed-noise likelihood (train_Yvar) cannot be set.",
            ));
        };
        self.update(|m| {
            let mut h = m.hyperparameters().clone();
            h.raw_noise = Some(spec.noise_constraint.inverse_transform(value));
            m.set_hyperparameters(h)
        })
    }

    /// Sets the constant mean (transformed outcome space).
    fn set_constant(&self, value: f64) -> PyResult<()> {
        self.update(|m| {
            let mut h = m.hyperparameters().clone();
            h.raw_constant = value;
            m.set_hyperparameters(h)
        })
    }

    /// The observation-noise variance (original space) that
    /// `posterior(X, observation_noise=True)` adds.
    fn observation_noise(&self) -> f64 {
        self.snapshot().observation_noise()
    }

    // ------------------------------------------------------------------
    // Posterior
    // ------------------------------------------------------------------

    /// Marginal posterior means and variances at the points `X` (`... x d`):
    /// two arrays of shape `...` (variances clamped at 1e-10 like
    /// `posterior.variance`).
    #[pyo3(signature = (x, observation_noise=false))]
    fn posterior_marginals(
        &self,
        py: Python<'_>,
        x: Array,
        observation_noise: bool,
    ) -> PyResult<(ArrayOut, ArrayOut)> {
        let model = self.snapshot();
        let (batch, points) = x.points("X")?;
        if points.ncols() != model.dim() {
            return Err(dimension_error(format!(
                "Expected X with {} columns (d), got {}",
                model.dim(),
                points.ncols()
            )));
        }
        let (mean, var) =
            threads::run(py, || marginals(&model, &points, observation_noise)).into_py_result()?;
        Ok(((mean, batch.clone()), (var, batch)))
    }

    /// Joint posterior of every `q x d` batch of `X` (`... x q x d`): means
    /// (`... x q`) and covariance matrices (`... x q x q`).
    #[pyo3(signature = (x, observation_noise=false))]
    fn posterior_joint(
        &self,
        py: Python<'_>,
        x: Array,
        observation_noise: bool,
    ) -> PyResult<(ArrayOut, ArrayOut)> {
        let model = self.snapshot();
        let (batch, mats) = x.batches("X")?;
        let q = x.1[x.1.len() - 2];
        let posts = threads::run(py, || {
            mats.par_iter()
                .map(|m| model.posterior(m, observation_noise))
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        let mut mean = Vec::with_capacity(posts.len() * q);
        for p in &posts {
            mean.extend(p.mean.iter());
        }
        let covs: Vec<Matrix> = posts.into_iter().map(|p| p.covariance).collect();
        let mut mean_shape = batch.clone();
        mean_shape.push(q);
        Ok(((mean, mean_shape), stack_out(&covs, &batch, q, q)))
    }

    /// Posterior samples `mean + L z` at every `q x d` batch of `X`
    /// (`... x q x d`) for base samples `z` of shape `S x q` (shared by all
    /// batches) or `S x B x q` (one set per batch, `B` the number of
    /// batches); returns an array of shape `S x ... x q`.
    #[pyo3(signature = (x, base_samples, observation_noise=false))]
    fn posterior_samples(
        &self,
        py: Python<'_>,
        x: Array,
        base_samples: Array,
        observation_noise: bool,
    ) -> PyResult<ArrayOut> {
        let model = self.snapshot();
        let (batch, mats) = x.batches("X")?;
        let q = x.1[x.1.len() - 2];
        let nb = mats.len();
        let z = &base_samples;
        let per_batch = match z.1.as_slice() {
            &[_, zq] if zq == q => false,
            &[_, zb, zq] if zb == nb && zq == q => true,
            s => {
                return Err(dimension_error(format!(
                    "base samples of shape {s:?} do not match {nb} batches of q = {q} points"
                )))
            }
        };
        if z.0.len() != z.1.iter().product::<usize>() {
            return Err(dimension_error("base samples do not match their shape"));
        }
        let s = z.1[0];
        let samples = threads::run(py, || {
            mats.par_iter()
                .enumerate()
                .map(|(b, xb)| {
                    let post = model.posterior(xb, observation_noise)?;
                    let root = CovarianceRoot::new(&post.covariance)?.to_matrix();
                    let zb = Matrix::from_fn(s, q, |i, j| {
                        if per_batch {
                            z.0[(i * nb + b) * q + j]
                        } else {
                            z.0[i * q + j]
                        }
                    });
                    let mut y = root * zb.transpose(); // q x S
                    for j in 0..q {
                        let m = post.mean[j];
                        y.row_mut(j).iter_mut().for_each(|v| *v += m);
                    }
                    Ok(y)
                })
                .collect::<botorch_core::Result<Vec<Matrix>>>()
        })
        .into_py_result()?;
        let mut data = Vec::with_capacity(s * nb * q);
        for i in 0..s {
            for y in &samples {
                data.extend((0..q).map(|j| y[(j, i)]));
            }
        }
        let mut shape = vec![s];
        shape.extend(batch);
        shape.push(q);
        Ok((data, shape))
    }

    // ------------------------------------------------------------------
    // Marginal log likelihood and fitting
    // ------------------------------------------------------------------

    /// The exact MLL (divided by `n`, priors included, as in GPyTorch) at the
    /// current or the given raw hyperparameters.
    #[pyo3(signature = (raw_parameters=None))]
    fn mll(&self, py: Python<'_>, raw_parameters: Option<Vec<f64>>) -> PyResult<f64> {
        let model = self.snapshot();
        threads::run(py, || {
            let mll = ExactMarginalLogLikelihood::new(&model);
            match &raw_parameters {
                Some(p) => mll.value_at(p),
                None => mll.value(),
            }
        })
        .into_py_result()
    }

    /// The exact MLL and its gradient with respect to the raw hyperparameters.
    #[pyo3(signature = (raw_parameters=None))]
    fn mll_value_and_grad(
        &self,
        py: Python<'_>,
        raw_parameters: Option<Vec<f64>>,
    ) -> PyResult<(f64, Vec<f64>)> {
        let model = self.snapshot();
        let params = raw_parameters.unwrap_or_else(|| model.raw_parameters().as_slice().to_vec());
        let (v, g) = threads::run(py, || {
            ExactMarginalLogLikelihood::new(&model).value_and_grad(&params)
        })
        .into_py_result()?;
        Ok((v, g.as_slice().to_vec()))
    }

    /// Fits the hyperparameters by maximizing the MLL with L-BFGS-B
    /// (`fit_gpytorch_mll`); returns `(loss, iterations, evaluations, status,
    /// message, attempts)`. On failure the model is left unchanged.
    #[pyo3(signature = (
        maxcor=None,
        pgtol=None,
        ftol=None,
        maxiter=None,
        maxfun=None,
        maxls=None,
        max_attempts=None,
        seed=None,
    ))]
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
        if !(options.pgtol >= 0.0 && options.ftol >= 0.0) {
            return Err(value_error("gtol and ftol must be non-negative"));
        }
        let mut model = (*self.snapshot()).clone();
        let result = threads::run(py, || fit_gpytorch_mll(&mut model, options)).into_py_result()?;
        self.replace(model);
        let status = match result.status {
            FitStatus::Success => "SUCCESS",
            FitStatus::Stopped => "STOPPED",
            FitStatus::Failure => "FAILURE",
        };
        Ok((
            result.loss,
            result.iterations,
            result.evaluations,
            status,
            result.message,
            result.attempts,
        ))
    }
}
