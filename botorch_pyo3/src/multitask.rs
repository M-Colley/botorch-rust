//! `MultiTaskGP` and `KroneckerMultiTaskGP` (`botorch.models.multitask`):
//! the multi-task GPs with an ICM kernel on long-format data and for the
//! block design (see `botorch_core::models::multitask`).
//!
//! Like `SingleTaskGP`, the model is an `Arc` snapshot behind a lock. It
//! holds a batch of models: one for a model built from data, `n_f · b` for
//! the fantasy models of `fantasize` (BoTorch's fantasy model with batch
//! shape `[n_f] + batch_shape`), which share their hyperparameters.
//! Posteriors are evaluated through the `Model` handles of
//! `multitask_model` (over output tasks) and `multitask_task_inputs_model`
//! (inputs with a task column).

use std::sync::{Arc, RwLock};

use botorch_core::acquisition::{ScalarizedModel, ScalarizedPosteriorTransform};
use botorch_core::models::multitask::{
    KroneckerMultiTaskGP, KroneckerMultiTaskGPOptions, ModelBatch, MultiTaskGP, MultiTaskGPOptions,
    MultiTaskMean, MultiTaskNoise, MultiTaskTaskInputs, TaskCovarPrior,
};
use botorch_core::models::Model;
use botorch_core::{
    fit_gpytorch_mll, BatchedInputTransform, FitOptions, FitStatus, Matrix, OutcomeTransformChoice,
    Vector,
};
use pyo3::prelude::*;

use crate::arrays::{broadcast_indices, broadcast_shapes, matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{dimension_error, unsupported, value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::models::PyModel;
use crate::sampling::PySobolQMCNormalSampler;
use crate::specs::PyLikelihoodSpec;
use crate::threads;
use crate::transforms::{PyNormalize, PyStandardize};

/// `(loss, iterations, evaluations, status, message, attempts)`.
type ElementFit = (f64, usize, usize, &'static str, String, usize);
/// The fit summary of `SingleTaskGP.fit` with one element.
type FitSummary = (
    f64,
    usize,
    usize,
    &'static str,
    String,
    usize,
    Vec<ElementFit>,
);

/// A posterior transform received from Python: `(weights, offset)`.
type TransformArg = (Vec<f64>, f64);

fn status_name(status: FitStatus) -> &'static str {
    match status {
        FitStatus::Success => "SUCCESS",
        FitStatus::Stopped => "STOPPED",
        FitStatus::Failure => "FAILURE",
    }
}

/// The models of a `MultiTaskGP` handle (one unless it holds fantasies).
#[derive(Clone)]
struct MtBatch {
    batch_shape: Vec<usize>,
    models: Vec<Arc<MultiTaskGP>>,
}

/// BoTorch's `MultiTaskGP` (see `botorch_core::models::multitask::gp`).
#[pyclass(frozen, name = "MultiTaskGP", module = "botorch_rs._botorch_rs")]
pub struct PyMultiTaskGP {
    inner: RwLock<Arc<MtBatch>>,
}

impl PyMultiTaskGP {
    fn from_batch(batch: MtBatch) -> Self {
        Self {
            inner: RwLock::new(Arc::new(batch)),
        }
    }

    fn state(&self) -> Arc<MtBatch> {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The first model (the hyperparameters of all of them).
    fn first(&self) -> Arc<MultiTaskGP> {
        Arc::clone(&self.state().models[0])
    }

    /// The model, which must not be a batch of fantasies.
    fn single(&self, what: &str) -> PyResult<Arc<MultiTaskGP>> {
        let s = self.state();
        if s.models.len() != 1 || !s.batch_shape.is_empty() {
            return Err(unsupported(format!(
                "{what} is not supported for batched (fantasy) MultiTaskGPs in botorch_rs."
            )));
        }
        Ok(Arc::clone(&s.models[0]))
    }

    /// Applies `f` to copies of all models and installs them on success.
    fn update<F>(&self, f: F) -> PyResult<()>
    where
        F: Fn(&mut MultiTaskGP) -> botorch_core::Result<()>,
    {
        let s = self.state();
        let models = s
            .models
            .iter()
            .map(|m| {
                let mut c = (**m).clone();
                f(&mut c).map(|()| Arc::new(c))
            })
            .collect::<botorch_core::Result<Vec<_>>>()
            .into_py_result()?;
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(MtBatch {
            batch_shape: s.batch_shape.clone(),
            models,
        });
        Ok(())
    }

    fn set_named(&self, name: &str, values: &[f64]) -> PyResult<()> {
        let model = self.first();
        let mut raw = model.raw_parameters().clone();
        let mut offset = 0;
        for (n, size) in model.parameter_names() {
            if n == name {
                if values.len() != size {
                    return Err(dimension_error(format!(
                        "{name} has {size} values, got {}",
                        values.len()
                    )));
                }
                raw.rows_mut(offset, size).copy_from_slice(values);
                return self.update(|m| m.set_raw_parameters(raw.as_slice()));
            }
            offset += size;
        }
        Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "The model has no parameter {name:?}."
        )))
    }
}

fn with_transform(model: Arc<dyn Model>, transform: Option<TransformArg>) -> PyResult<PyModel> {
    let t = transform
        .map(|(w, offset)| ScalarizedPosteriorTransform::new(Vector::from_vec(w), offset))
        .transpose()
        .into_py_result()?;
    Ok(PyModel {
        inner: ScalarizedModel::wrap(model, t).into_py_result()?,
    })
}

/// The `Model` handle of a `MultiTaskGP` over its output tasks (or the tasks
/// `output_tasks`), seen through the posterior transform if given; a batch of
/// models for fantasy models.
#[pyfunction]
#[pyo3(signature = (model, output_tasks=None, transform=None))]
pub fn multitask_model(
    model: PyRef<'_, PyMultiTaskGP>,
    output_tasks: Option<Vec<i64>>,
    transform: Option<TransformArg>,
) -> PyResult<PyModel> {
    let s = model.state();
    let models = s
        .models
        .iter()
        .map(|m| -> PyResult<Arc<dyn Model>> {
            Ok(match &output_tasks {
                Some(t) => Arc::new(m.with_output_tasks(t).into_py_result()?),
                None => Arc::clone(m) as Arc<dyn Model>,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let handle: Arc<dyn Model> = if s.batch_shape.is_empty() {
        Arc::clone(&models[0])
    } else {
        Arc::new(ModelBatch::new(s.batch_shape.clone(), models).into_py_result()?)
    };
    with_transform(handle, transform)
}

/// The `Model` handle of a `MultiTaskGP` with inputs that include the task
/// column (`d + 1` columns, one output).
#[pyfunction]
pub fn multitask_task_inputs_model(model: PyRef<'_, PyMultiTaskGP>) -> PyResult<PyModel> {
    let s = model.state();
    let models: Vec<Arc<dyn Model>> = s
        .models
        .iter()
        .map(|m| Arc::new(MultiTaskTaskInputs::new(Arc::clone(m))) as Arc<dyn Model>)
        .collect();
    let inner: Arc<dyn Model> = if s.batch_shape.is_empty() {
        Arc::clone(&models[0])
    } else {
        Arc::new(ModelBatch::new(s.batch_shape.clone(), models).into_py_result()?)
    };
    Ok(PyModel { inner })
}

#[pymethods]
impl PyMultiTaskGP {
    /// Builds the model from `train_x` (`n x (d + 1)` with the task column
    /// `task_feature`) and `train_y` (`n x 1`); see
    /// `botorch_core::models::multitask::MultiTaskGPOptions`.
    /// `outcome_transform=None` with `standardize_by_default=True` is
    /// BoTorch's `DEFAULT` (`Standardize(m=1)`). `covar_module` is the data
    /// kernel (with `covar_module_active_dims` for a composite kernel),
    /// `task_covar_module` the task kernel (`None`: BoTorch's
    /// `PositiveIndexKernel` with `rank` and the default `BetaPrior` unless
    /// `default_task_prior=False`). `likelihood_kind` is `"hadamard"`
    /// (per-task noise) or `"gaussian"` (shared noise) for a learned noise;
    /// `mean_kind` is `"multitask"` or `"constant"`.
    #[new]
    #[pyo3(signature = (
        train_x,
        train_y,
        task_feature,
        train_yvar=None,
        covar_module=None,
        covar_module_active_dims=None,
        task_covar_module=None,
        default_task_prior=true,
        likelihood=None,
        likelihood_kind="hadamard",
        mean_kind="multitask",
        output_tasks=None,
        rank=None,
        all_tasks=None,
        outcome_transform=None,
        standardize_by_default=true,
        input_transform=None,
        input_transform_indices=None,
        validate_task_values=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        train_x: Array,
        train_y: Array,
        task_feature: isize,
        train_yvar: Option<Array>,
        covar_module: Option<PyRef<'_, PyKernelNode>>,
        covar_module_active_dims: Option<Vec<usize>>,
        task_covar_module: Option<PyRef<'_, PyKernelNode>>,
        default_task_prior: bool,
        likelihood: Option<PyRef<'_, PyLikelihoodSpec>>,
        likelihood_kind: &str,
        mean_kind: &str,
        output_tasks: Option<Vec<i64>>,
        rank: Option<usize>,
        all_tasks: Option<Vec<i64>>,
        outcome_transform: Option<PyRef<'_, PyStandardize>>,
        standardize_by_default: bool,
        input_transform: Option<PyRef<'_, PyNormalize>>,
        input_transform_indices: Option<Vec<usize>>,
        validate_task_values: bool,
    ) -> PyResult<Self> {
        let x = train_x.matrix("train_X")?;
        let y = train_y.matrix("train_Y")?;
        let yvar = train_yvar.map(|v| v.matrix("train_Yvar")).transpose()?;
        let outcome = match (outcome_transform, standardize_by_default) {
            (Some(st), _) => OutcomeTransformChoice::Standardize(
                botorch_core::Standardize::new().with_min_stdv(st.inner[0].min_stdv()),
            ),
            (None, true) => OutcomeTransformChoice::Default,
            (None, false) => OutcomeTransformChoice::None,
        };
        let likelihood = match (likelihood, likelihood_kind) {
            (None, _) => None,
            (Some(l), "hadamard") => Some(MultiTaskNoise::PerTask(l.inner)),
            (Some(l), "gaussian") => Some(MultiTaskNoise::Shared(l.inner)),
            (_, other) => return Err(value_error(format!("unknown likelihood kind {other:?}"))),
        };
        let mean = match mean_kind {
            "multitask" => MultiTaskMean::PerTask,
            "constant" => MultiTaskMean::Shared,
            other => return Err(value_error(format!("unknown mean kind {other:?}"))),
        };
        let input_transform = match input_transform {
            None => None,
            Some(tf) => match tf.for_model(&[])? {
                BatchedInputTransform::Shared(t) => Some(t),
                BatchedInputTransform::PerBatch(_) => {
                    return Err(unsupported(
                        "A Normalize with a batch shape is not supported by MultiTaskGP.",
                    ))
                }
            },
        };
        let options = MultiTaskGPOptions {
            train_yvar: yvar,
            mean,
            covar_module: covar_module.map(|c| c.inner.clone()),
            covar_module_active_dims,
            likelihood,
            task_covar_prior: if default_task_prior {
                TaskCovarPrior::Default
            } else {
                TaskCovarPrior::None
            },
            task_covar_module: task_covar_module.map(|c| c.inner.clone()),
            output_tasks,
            rank,
            all_tasks,
            outcome_transform: outcome,
            input_transform,
            input_transform_indices,
            validate_task_values,
        };
        let model =
            threads::run(py, || MultiTaskGP::new(x, y, task_feature, options)).into_py_result()?;
        Ok(Self::from_batch(MtBatch {
            batch_shape: Vec::new(),
            models: vec![Arc::new(model)],
        }))
    }

    /// A deep copy of the model.
    fn copy(&self) -> Self {
        Self::from_batch((*self.state()).clone())
    }

    // ------------------------------------------------------------------
    // Data and configuration
    // ------------------------------------------------------------------

    /// The number of non-task features `d`.
    #[getter]
    fn dim(&self) -> usize {
        self.first().dim()
    }

    /// Number of training points `n`.
    #[getter]
    fn num_train(&self) -> usize {
        self.first().num_train()
    }

    /// The number of tasks.
    #[getter]
    fn num_tasks(&self) -> usize {
        self.first().num_tasks()
    }

    /// The number of outputs (output tasks).
    #[getter]
    fn num_outputs(&self) -> usize {
        self.first().task_layout().output_tasks().len()
    }

    /// The (non-negative) column of the task feature.
    #[getter]
    fn task_feature(&self) -> usize {
        self.first().task_layout().task_feature()
    }

    /// The sorted task values.
    #[getter]
    fn all_tasks(&self) -> Vec<i64> {
        self.first().task_layout().all_tasks().to_vec()
    }

    /// The output tasks (task values).
    #[getter]
    fn output_tasks(&self) -> Vec<i64> {
        self.first().task_layout().output_tasks().to_vec()
    }

    /// The task indices with training data.
    #[getter]
    fn observed_task_indices(&self) -> Vec<usize> {
        self.first().task_layout().observed_task_indices()
    }

    /// Whether task values are task indices (BoTorch's task mapper is None).
    #[getter]
    fn tasks_are_indices(&self) -> bool {
        self.first().task_layout().is_identity()
    }

    /// The batch shape (non-empty for fantasy models).
    #[getter]
    fn batch_shape(&self) -> Vec<usize> {
        self.state().batch_shape.clone()
    }

    /// The batch shape of the hyperparameters (always empty).
    #[getter]
    fn aug_batch_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    /// The rank of the task covariance.
    #[getter]
    fn rank(&self) -> usize {
        self.first().rank()
    }

    /// The position of the task kernel among the sub-kernels of
    /// `covar_module`.
    #[getter]
    fn task_kernel_index(&self) -> usize {
        self.first().task_kernel_index()
    }

    /// Whether the noise is fixed (`train_Yvar`).
    #[getter]
    fn is_fixed_noise(&self) -> bool {
        self.first().likelihood().is_none()
    }

    /// The covariance module (the product kernel tree).
    fn covar_tree(&self) -> PyKernelNode {
        PyKernelNode {
            inner: self.first().covar_module().clone(),
        }
    }

    /// Training inputs including the task column (`batch x n x (d + 1)`).
    fn train_inputs(&self) -> ArrayBuf {
        let s = self.state();
        stacked(&s, |m| m.train_inputs().clone())
    }

    /// Training targets in the original space (`batch x n x 1`).
    fn train_targets(&self) -> ArrayBuf {
        let s = self.state();
        stacked(&s, |m| m.train_targets().clone())
    }

    /// Training targets after the outcome transform (`batch x n`).
    fn transformed_train_targets(&self) -> ArrayBuf {
        let s = self.state();
        let mut data = Vec::new();
        for m in &s.models {
            data.extend(m.transformed_train_targets().iter());
        }
        let mut shape = s.batch_shape.clone();
        shape.push(s.models[0].num_train());
        ArrayBuf(data, shape)
    }

    /// Training inputs after the input transform (`batch x n x (d + 1)`).
    fn transformed_train_inputs(&self) -> PyResult<ArrayBuf> {
        let s = self.state();
        let mats = s
            .models
            .iter()
            .map(|m| m.transformed_train_inputs())
            .collect::<botorch_core::Result<Vec<_>>>()
            .into_py_result()?;
        Ok(stack(&s.batch_shape, &mats))
    }

    /// The fitted outcome transform, if any.
    fn outcome_transform(&self) -> PyResult<Option<PyStandardize>> {
        let m = self.first();
        let Some(st) = m.outcome_transform() else {
            return Ok(None);
        };
        PyStandardize::from_stats(
            vec![st.means().to_vec()],
            vec![st.stdvs().to_vec()],
            st.min_stdv(),
            Vec::new(),
        )
        .map(Some)
    }

    /// The fitted input transform (`learn_bounds` as given), if any.
    fn input_transform(&self, learn_bounds: bool) -> Option<PyNormalize> {
        let m = self.first();
        let (tf, _) = m.input_transform()?;
        Some(PyNormalize::fitted(
            &BatchedInputTransform::Shared(tf.clone()),
            learn_bounds,
            &[],
        ))
    }

    /// The learned-noise configuration (prior and constraint), `None` for
    /// fixed noise.
    fn likelihood_spec(&self) -> Option<PyLikelihoodSpec> {
        self.first().likelihood().map(|l| PyLikelihoodSpec {
            inner: match l {
                MultiTaskNoise::PerTask(s) | MultiTaskNoise::Shared(s) => *s,
            },
        })
    }

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------

    /// BoTorch parameter names and sizes in flat order.
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.first().parameter_names()
    }

    /// BoTorch parameter names and shapes.
    fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        self.first().parameter_shapes()
    }

    /// The raw hyperparameters in flat order (BoTorch's `named_parameters()`).
    fn raw_parameters(&self) -> Vec<f64> {
        self.first().raw_parameters().as_slice().to_vec()
    }

    /// Sets the raw hyperparameters from a flat vector.
    fn set_raw_parameters(&self, values: Vec<f64>) -> PyResult<()> {
        self.update(|m| m.set_raw_parameters(&values))
    }

    /// Sets one raw parameter by its BoTorch name (flat values).
    fn set_raw_parameter(&self, name: &str, values: Vec<f64>) -> PyResult<()> {
        self.set_named(name, &values)
    }

    /// Optimizer bounds of the raw hyperparameters in flat order.
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.first().raw_parameter_bounds()
    }

    /// The noise variances (transformed space): per task, shared, or per
    /// training row (fixed noise; `batch x n` for fantasy models).
    #[getter]
    fn noise(&self) -> Vec<f64> {
        let s = self.state();
        let mut out = Vec::new();
        let fixed = s.models[0].likelihood().is_none();
        for m in if fixed { &s.models[..] } else { &s.models[..1] } {
            out.extend(m.likelihood_noise().iter());
        }
        out
    }

    /// Sets the learned noise variances (actual values, broadcast from one
    /// value): a list of values or a `(data, shape)` array.
    fn set_noise(&self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let values: Vec<f64> = match values.extract::<Array>() {
            Ok(a) => a.0,
            Err(_) => values.extract()?,
        };
        let m = self.first();
        let Some(l) = m.likelihood() else {
            return Err(unsupported(
                "The noise of a fixed-noise model cannot be set.",
            ));
        };
        let (spec, k) = match l {
            MultiTaskNoise::PerTask(s) => (*s, m.num_tasks()),
            MultiTaskNoise::Shared(s) => (*s, 1),
        };
        let values = match values.len() {
            1 => vec![values[0]; k],
            n if n == k => values,
            n => {
                return Err(dimension_error(format!(
                    "noise has {k} values, got {n} (a value of this shape cannot be broadcast)"
                )))
            }
        };
        let raw: Vec<f64> = values
            .iter()
            .map(|&v| spec.noise_constraint.inverse_transform(v))
            .collect();
        self.set_named("likelihood.noise_covar.raw_noise", &raw)
    }

    /// The constant means (transformed space).
    #[getter]
    fn constant(&self) -> Vec<f64> {
        self.first().constants().as_slice().to_vec()
    }

    /// Sets the constant of a shared `ConstantMean` (broadcast from one value).
    fn set_constant(&self, value: Array) -> PyResult<()> {
        let m = self.first();
        let names = m.parameter_names();
        if !names.iter().any(|(n, _)| n == "mean_module.raw_constant") {
            return Err(unsupported(
                "The model has one constant per task (mean_module.base_means[t].constant).",
            ));
        }
        if value.0.len() != 1 {
            return Err(dimension_error(format!(
                "The constant is a scalar, got a value of shape {:?}.",
                value.1
            )));
        }
        self.set_named("mean_module.raw_constant", &value.0)
    }

    /// BoTorch's `eval()` for tasks without data: the rows of the task
    /// factor of unobserved tasks become the mean of the observed rows.
    fn eval_unobserved(&self) -> PyResult<()> {
        let m = self.first();
        if m.task_layout().unobserved_task_indices().is_empty() {
            return Ok(());
        }
        self.update(|m| {
            m.eval_unobserved_task_factors();
            Ok(())
        })
    }

    /// The task covariance matrix of the task kernel (`K x K`).
    fn task_covariance(&self, py: Python<'_>) -> PyResult<ArrayOut> {
        let m = self.first();
        let c = threads::run(py, || m.task_covariance()).into_py_result()?;
        Ok(matrix_out(&c))
    }

    // ------------------------------------------------------------------
    // Marginal log likelihood and fitting
    // ------------------------------------------------------------------

    /// The exact MLL (divided by `n`, priors included) at the current or the
    /// given flat raw hyperparameters (shape `()`).
    #[pyo3(signature = (raw_parameters=None))]
    fn mll(&self, py: Python<'_>, raw_parameters: Option<Vec<f64>>) -> PyResult<ArrayOut> {
        let model = self.single("The marginal log likelihood")?;
        let value = threads::run(py, || match &raw_parameters {
            Some(p) => model.mll_value_and_grad(p).map(|(v, _)| v),
            None => model.mll_value(),
        })
        .into_py_result()?;
        Ok((vec![value], Vec::new()))
    }

    /// The exact MLL and its gradient with respect to the flat raw
    /// hyperparameters.
    #[pyo3(signature = (raw_parameters=None))]
    fn mll_value_and_grad(
        &self,
        py: Python<'_>,
        raw_parameters: Option<Vec<f64>>,
    ) -> PyResult<(ArrayOut, Vec<f64>)> {
        let model = self.single("The marginal log likelihood")?;
        let params = raw_parameters.unwrap_or_else(|| model.raw_parameters().as_slice().to_vec());
        let (v, g) = threads::run(py, || model.mll_value_and_grad(&params)).into_py_result()?;
        Ok(((vec![v], Vec::new()), g.as_slice().to_vec()))
    }

    /// Fits the hyperparameters with L-BFGS-B (`fit_gpytorch_mll`) and applies
    /// BoTorch's `eval()`. On failure the model is left unchanged.
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
        let mut model = (*self.single("Fitting")?).clone();
        let r = threads::run(py, || fit_gpytorch_mll(&mut model, options)).into_py_result()?;
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(MtBatch {
            batch_shape: Vec::new(),
            models: vec![Arc::new(model)],
        });
        let e: ElementFit = (
            r.loss,
            r.iterations,
            r.evaluations,
            status_name(r.status),
            r.message.clone(),
            r.attempts,
        );
        Ok((
            r.loss,
            r.iterations,
            r.evaluations,
            status_name(r.status),
            r.message,
            r.attempts,
            vec![e],
        ))
    }

    // ------------------------------------------------------------------
    // Conditioning and fantasies
    // ------------------------------------------------------------------

    /// BoTorch's `condition_on_observations(X, Y, noise=noise)`: `X`
    /// (`batch_x x n' x (d + 1)`), `Y` (`batch_y x n' x 1`, original space)
    /// and optional `noise` (shape of `Y`, transformed space); the result has
    /// the broadcast batch shape.
    #[pyo3(signature = (x, y, noise=None))]
    fn condition_on_observations(
        &self,
        py: Python<'_>,
        x: Array,
        y: Array,
        noise: Option<Array>,
    ) -> PyResult<Self> {
        let model = self.single("condition_on_observations")?;
        let (xb, xs) = x.batches("X")?;
        let (yb, ys) = y.batches("Y")?;
        let nz = noise.map(|v| v.batches("noise")).transpose()?;
        let empty: Vec<usize> = Vec::new();
        let nb = nz.as_ref().map_or(&empty, |(b, _)| b);
        let out = broadcast_shapes(&xb, &yb)
            .and_then(|s| broadcast_shapes(&s, nb))
            .ok_or_else(|| {
                dimension_error(format!(
                    "The batch shapes of X ({xb:?}), Y ({yb:?}) and the noise ({nb:?}) are not \
                     broadcastable."
                ))
            })?;
        let ix = broadcast_indices(&xb, &out);
        let iy = broadcast_indices(&yb, &out);
        let inz = broadcast_indices(nb, &out);
        let models = threads::run(py, || -> botorch_core::Result<Vec<Arc<MultiTaskGP>>> {
            let mut slots: Vec<Option<Arc<MultiTaskGP>>> = vec![None; ix.len()];
            // Elements with the same inputs (and noise) share the update.
            let mut groups: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
                std::collections::BTreeMap::new();
            for r in 0..ix.len() {
                groups.entry((ix[r], inz[r])).or_default().push(r);
            }
            for ((a, b), members) in groups {
                let noise_vec: Option<Vec<f64>> = nz
                    .as_ref()
                    .map(|(_, mats)| mats[b].column(0).iter().copied().collect());
                let yy: Vec<Matrix> = members.iter().map(|&r| ys[iy[r]].clone()).collect();
                let conditioned =
                    model.condition_on_observations_many(&xs[a], &yy, noise_vec.as_deref())?;
                for (&r, c) in members.iter().zip(conditioned) {
                    slots[r] = Some(Arc::new(c));
                }
            }
            Ok(slots.into_iter().map(|s| s.expect("conditioned")).collect())
        })
        .into_py_result()?;
        Ok(Self::from_batch(MtBatch {
            batch_shape: out,
            models,
        }))
    }

    /// BoTorch's `fantasize(X, sampler)`: `X` (`batch x n' x (d + 1)`); the
    /// fantasy model has the batch shape `[n_f] + batch`.
    #[pyo3(signature = (x, sampler, observation_noise=None))]
    fn fantasize(
        &self,
        py: Python<'_>,
        x: Array,
        sampler: PyRef<'_, PySobolQMCNormalSampler>,
        observation_noise: Option<Array>,
    ) -> PyResult<Self> {
        if observation_noise.is_some() {
            return Err(unsupported(
                "Passing a tensor of observations is not supported by MultiTaskGP.",
            ));
        }
        let model = self.single("fantasize")?;
        let (xb, xs) = x.batches("X")?;
        let k = x.1[x.1.len() - 2];
        let s = Arc::clone(&sampler.inner);
        let n_f = s.num_samples();
        let models = threads::run(py, || -> botorch_core::Result<Vec<Arc<MultiTaskGP>>> {
            let z = if k == 0 {
                Arc::new(Matrix::zeros(n_f, 0))
            } else {
                s.base_samples(k)?
            };
            let per_batch = xs
                .iter()
                .map(|xm| model.fantasize_with_base_samples(xm, &z))
                .collect::<botorch_core::Result<Vec<_>>>()?;
            let nb = per_batch.len();
            let mut out = Vec::with_capacity(n_f * nb);
            for i in 0..n_f {
                for fantasies in &per_batch {
                    out.push(Arc::new(fantasies[i].clone()));
                }
            }
            Ok(out)
        })
        .into_py_result()?;
        let mut batch_shape = vec![n_f];
        batch_shape.extend(xb);
        Ok(Self::from_batch(MtBatch {
            batch_shape,
            models,
        }))
    }
}

/// `f(model)` of every model, stacked to `batch x r x c`.
fn stacked<F>(s: &MtBatch, f: F) -> ArrayBuf
where
    F: Fn(&MultiTaskGP) -> Matrix,
{
    let mats: Vec<Matrix> = s.models.iter().map(|m| f(m)).collect();
    stack(&s.batch_shape, &mats)
}

fn stack(batch: &[usize], mats: &[Matrix]) -> ArrayBuf {
    let (r, c) = mats[0].shape();
    let mut data = Vec::with_capacity(mats.len() * r * c);
    for m in mats {
        data.extend(matrix_out(m).0);
    }
    let mut shape = batch.to_vec();
    shape.extend([r, c]);
    ArrayBuf(data, shape)
}

// ----------------------------------------------------------------------
// KroneckerMultiTaskGP
// ----------------------------------------------------------------------

/// BoTorch's `KroneckerMultiTaskGP` (see
/// `botorch_core::models::multitask::kronecker`).
#[pyclass(
    frozen,
    name = "KroneckerMultiTaskGP",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyKroneckerMultiTaskGP {
    inner: RwLock<Arc<KroneckerMultiTaskGP>>,
}

impl PyKroneckerMultiTaskGP {
    fn snapshot(&self) -> Arc<KroneckerMultiTaskGP> {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn replace(&self, model: KroneckerMultiTaskGP) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(model);
    }

    fn set_named(&self, name: &str, values: &[f64]) -> PyResult<()> {
        let model = self.snapshot();
        let mut raw = model.raw_parameters().clone();
        let mut offset = 0;
        for (n, size) in model.parameter_names() {
            if n == name {
                if values.len() != size {
                    return Err(dimension_error(format!(
                        "{name} has {size} values, got {}",
                        values.len()
                    )));
                }
                raw.rows_mut(offset, size).copy_from_slice(values);
                let mut m = (*model).clone();
                m.set_raw_parameters(raw.as_slice()).into_py_result()?;
                self.replace(m);
                return Ok(());
            }
            offset += size;
        }
        Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "The model has no parameter {name:?}."
        )))
    }
}

/// The `Model` handle of a `KroneckerMultiTaskGP` (its exact joint
/// posterior over all tasks).
#[pyfunction]
pub fn kronecker_model(model: PyRef<'_, PyKroneckerMultiTaskGP>) -> PyModel {
    let m: Arc<dyn Model> = model.snapshot();
    PyModel { inner: m }
}

#[pymethods]
impl PyKroneckerMultiTaskGP {
    /// Builds the model from `train_x` (`n x d`) and `train_y` (`n x m`).
    /// `data_covar_module` and `task_covar_module` are kernel trees (the
    /// task kernel an `IndexKernel` with its prior), `likelihood` the noise
    /// prior and constraint of the `MultitaskGaussianLikelihood`.
    #[new]
    #[pyo3(signature = (
        train_x,
        train_y,
        data_covar_module=None,
        task_covar_module=None,
        likelihood=None,
        likelihood_rank=0,
        rank=None,
        outcome_transform=None,
        input_transform=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        train_x: Array,
        train_y: Array,
        data_covar_module: Option<PyRef<'_, PyKernelNode>>,
        task_covar_module: Option<PyRef<'_, PyKernelNode>>,
        likelihood: Option<PyRef<'_, PyLikelihoodSpec>>,
        likelihood_rank: usize,
        rank: Option<usize>,
        outcome_transform: Option<PyRef<'_, PyStandardize>>,
        input_transform: Option<PyRef<'_, PyNormalize>>,
    ) -> PyResult<Self> {
        let x = train_x.matrix("train_X")?;
        let y = train_y.matrix("train_Y")?;
        let input_transform =
            match input_transform {
                None => None,
                Some(tf) => match tf.for_model(&[])? {
                    BatchedInputTransform::Shared(t) => Some(t),
                    BatchedInputTransform::PerBatch(_) => return Err(unsupported(
                        "A Normalize with a batch shape is not supported by KroneckerMultiTaskGP.",
                    )),
                },
            };
        let options = KroneckerMultiTaskGPOptions {
            data_covar_module: data_covar_module.map(|c| c.inner.clone()),
            task_covar_module: task_covar_module.map(|c| c.inner.clone()),
            likelihood: likelihood.map(|l| l.inner),
            likelihood_rank,
            rank,
            outcome_transform: outcome_transform
                .map(|st| botorch_core::Standardize::new().with_min_stdv(st.inner[0].min_stdv())),
            input_transform,
            ..Default::default()
        };
        let model =
            threads::run(py, || KroneckerMultiTaskGP::new(x, y, options)).into_py_result()?;
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

    /// Number of tasks (outputs) `m`.
    #[getter]
    fn num_tasks(&self) -> usize {
        self.snapshot().num_tasks()
    }

    /// The batch shape of the hyperparameters (always empty).
    #[getter]
    fn aug_batch_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Training inputs in the original space (`n x d`).
    fn train_inputs(&self) -> ArrayBuf {
        matrix_out(self.snapshot().train_inputs()).into()
    }

    /// Training targets after the outcome transform (`n x m`).
    fn transformed_train_targets(&self) -> ArrayBuf {
        matrix_out(self.snapshot().transformed_train_targets()).into()
    }

    /// Cached noiseless training covariance, interleaved point then task.
    fn train_full_covar(&self, py: Python<'_>) -> PyResult<ArrayBuf> {
        let model = self.snapshot();
        let value = threads::run(py, || model.train_full_covar()).into_py_result()?;
        Ok(matrix_out(value.as_ref()).into())
    }

    /// Cached transformed-space predictive mean solve (interleaved).
    fn predictive_mean_cache(&self, py: Python<'_>) -> PyResult<ArrayBuf> {
        let model = self.snapshot();
        let value = threads::run(py, || model.predictive_mean_cache()).into_py_result()?;
        Ok(ArrayBuf(value.as_slice().to_vec(), vec![value.len()]))
    }

    /// The fitted outcome transform, if any.
    fn outcome_transform(&self) -> PyResult<Option<PyStandardize>> {
        let m = self.snapshot();
        let Some(st) = m.outcome_transform() else {
            return Ok(None);
        };
        PyStandardize::from_stats(
            vec![st.means().to_vec()],
            vec![st.stdvs().to_vec()],
            st.min_stdv(),
            Vec::new(),
        )
        .map(Some)
    }

    /// The fitted input transform (`learn_bounds` as given), if any.
    fn input_transform(&self, learn_bounds: bool) -> Option<PyNormalize> {
        let m = self.snapshot();
        let tf = m.input_transform()?;
        Some(PyNormalize::fitted(
            &BatchedInputTransform::Shared(tf.clone()),
            learn_bounds,
            &[],
        ))
    }

    /// The noise prior and constraint.
    fn likelihood_spec(&self) -> PyLikelihoodSpec {
        PyLikelihoodSpec {
            inner: *self.snapshot().likelihood(),
        }
    }

    /// BoTorch parameter names and sizes in flat order.
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.snapshot().parameter_names()
    }

    /// BoTorch parameter names and shapes.
    fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        self.snapshot().parameter_shapes()
    }

    /// The raw hyperparameters in flat order.
    fn raw_parameters(&self) -> Vec<f64> {
        self.snapshot().raw_parameters().as_slice().to_vec()
    }

    /// Sets the raw hyperparameters from a flat vector.
    fn set_raw_parameters(&self, values: Vec<f64>) -> PyResult<()> {
        let mut m = (*self.snapshot()).clone();
        m.set_raw_parameters(&values).into_py_result()?;
        self.replace(m);
        Ok(())
    }

    /// Sets one raw parameter by its BoTorch name (flat values).
    fn set_raw_parameter(&self, name: &str, values: Vec<f64>) -> PyResult<()> {
        self.set_named(name, &values)
    }

    /// Optimizer bounds of the raw hyperparameters in flat order.
    fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.snapshot().raw_parameter_bounds()
    }

    /// `(task_noises or None, noise)`.
    fn noises(&self) -> (Option<Vec<f64>>, f64) {
        let (t, s) = self.snapshot().noises();
        (t.map(|v| v.as_slice().to_vec()), s)
    }

    /// The task noise covariance `N` (`m x m`).
    fn task_noise_covariance(&self) -> ArrayOut {
        matrix_out(&self.snapshot().task_noise_covariance())
    }

    /// The constant means (transformed space).
    #[getter]
    fn constant(&self) -> Vec<f64> {
        self.snapshot().constants().as_slice().to_vec()
    }

    /// The task covariance matrix (`m x m`).
    fn task_covariance(&self, py: Python<'_>) -> PyResult<ArrayOut> {
        let m = self.snapshot();
        let c = threads::run(py, || m.task_covariance()).into_py_result()?;
        Ok(matrix_out(&c))
    }

    /// The exact MLL (divided by `n m`, priors included).
    #[pyo3(signature = (raw_parameters=None))]
    fn mll(&self, py: Python<'_>, raw_parameters: Option<Vec<f64>>) -> PyResult<ArrayOut> {
        let model = self.snapshot();
        let value = threads::run(py, || match &raw_parameters {
            Some(p) => model.mll_value_and_grad(p).map(|(v, _)| v),
            None => model.mll_value(),
        })
        .into_py_result()?;
        Ok((vec![value], Vec::new()))
    }

    /// The exact MLL and its gradient with respect to the raw
    /// hyperparameters.
    #[pyo3(signature = (raw_parameters=None))]
    fn mll_value_and_grad(
        &self,
        py: Python<'_>,
        raw_parameters: Option<Vec<f64>>,
    ) -> PyResult<(ArrayOut, Vec<f64>)> {
        let model = self.snapshot();
        let params = raw_parameters.unwrap_or_else(|| model.raw_parameters().as_slice().to_vec());
        let (v, g) = threads::run(py, || model.mll_value_and_grad(&params)).into_py_result()?;
        Ok(((vec![v], Vec::new()), g.as_slice().to_vec()))
    }

    /// Fits the hyperparameters with L-BFGS-B (`fit_gpytorch_mll`). On
    /// failure the model is left unchanged.
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
        let r = threads::run(py, || fit_gpytorch_mll(&mut model, options)).into_py_result()?;
        self.replace(model);
        let e: ElementFit = (
            r.loss,
            r.iterations,
            r.evaluations,
            status_name(r.status),
            r.message.clone(),
            r.attempts,
        );
        Ok((
            r.loss,
            r.iterations,
            r.evaluations,
            status_name(r.status),
            r.message,
            r.attempts,
            vec![e],
        ))
    }

    /// Matheron samples (`S x batch x (q m)`, interleaved) at `x` (`batch x
    /// q x d`) from base samples `z` (`S x total` shared by the batches, or
    /// `S x batch x total`).
    #[pyo3(signature=(x, mean_adjoint, variance_adjoint, observation_noise=false))]
    fn marginal_vjp(
        &self,
        py: Python<'_>,
        x: Array,
        mean_adjoint: Array,
        variance_adjoint: Array,
        observation_noise: bool,
    ) -> PyResult<ArrayBuf> {
        let x = x.matrix("X")?;
        let mean = mean_adjoint.matrix("mean_adjoint")?;
        let variance = variance_adjoint.matrix("variance_adjoint")?;
        let model = self.snapshot();
        threads::run(py, || {
            model.marginal_vjp(&x, &mean, &variance, observation_noise)
        })
        .map(|g| matrix_out(&g).into())
        .into_py_result()
    }

    #[pyo3(signature=(x, z, adjoint, observation_noise=false))]
    fn matheron_samples_vjp(
        &self,
        py: Python<'_>,
        x: Array,
        z: Array,
        adjoint: Array,
        observation_noise: bool,
    ) -> PyResult<ArrayBuf> {
        let x = x.matrix("X")?;
        let z = z.matrix("base_samples")?;
        let adjoint = adjoint.matrix("adjoint")?;
        let model = self.snapshot();
        threads::run(py, || {
            model.matheron_samples_vjp(&x, &z, &adjoint, observation_noise)
        })
        .map(|g| matrix_out(&g).into())
        .into_py_result()
    }

    #[pyo3(signature = (x, z, observation_noise=false))]
    fn matheron_samples(
        &self,
        py: Python<'_>,
        x: Array,
        z: Array,
        observation_noise: bool,
    ) -> PyResult<ArrayBuf> {
        let model = self.snapshot();
        let (xb, xs) = x.batches("X")?;
        let nb = xs.len();
        let (s, total, per_batch) = match z.1.as_slice() {
            &[s, t] => (s, t, false),
            &[s, b, t] if b == nb => (s, t, true),
            other => {
                return Err(dimension_error(format!(
                    "base samples of shape {other:?} do not match {nb} batches"
                )))
            }
        };
        let q = x.1[x.1.len() - 2];
        let m = model.num_tasks();
        let outs = threads::run(py, || {
            xs.iter()
                .enumerate()
                .map(|(b, xm)| {
                    let zb = Matrix::from_fn(s, total, |i, k| {
                        if per_batch {
                            z.0[(i * nb + b) * total + k]
                        } else {
                            z.0[i * total + k]
                        }
                    });
                    model.matheron_samples(xm, &zb, observation_noise)
                })
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        let mut data = Vec::with_capacity(s * nb * q * m);
        for i in 0..s {
            for o in &outs {
                data.extend(o.row(i).iter());
            }
        }
        let mut shape = vec![s];
        shape.extend(xb);
        shape.extend([q, m]);
        Ok(ArrayBuf(data, shape))
    }
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiTaskGP>()?;
    m.add_class::<PyKroneckerMultiTaskGP>()?;
    m.add_function(wrap_pyfunction!(multitask_model, m)?)?;
    m.add_function(wrap_pyfunction!(multitask_task_inputs_model, m)?)?;
    m.add_function(wrap_pyfunction!(kronecker_model, m)?)?;
    Ok(())
}
