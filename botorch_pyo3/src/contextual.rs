//! `LCEMGP` (`botorch.models.contextual_multioutput`): the latent context
//! embedding multi-output GP with the task handling of `MultiTaskGP`.
//!
//! Like `SingleTaskGP`, the model is an `Arc` snapshot behind a lock:
//! acquisition functions share the snapshot that is current when they are
//! created, and fitting or setting hyperparameters installs a new one.
//! Posteriors are evaluated through the `Model` handles of
//! `PyModel::from_lcemgp` (over the output tasks or other tasks) and
//! `PyModel::from_lcemgp_task_inputs` (inputs with a task column).

use std::sync::{Arc, RwLock};

use botorch_core::models::contextual_multioutput::{LCEMGPOptions, LCEMGP};
use botorch_core::{fit_gpytorch_mll, FitOptions, FitStatus, Matrix, OutcomeTransformChoice};
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{dimension_error, IntoPyResult};
use crate::specs::{PyKernelSpec, PyLikelihoodSpec};
use crate::threads;
use crate::transforms::PyStandardize;

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

fn status_name(status: FitStatus) -> &'static str {
    match status {
        FitStatus::Success => "SUCCESS",
        FitStatus::Stopped => "STOPPED",
        FitStatus::Failure => "FAILURE",
    }
}

/// BoTorch's `LCEMGP` (see `botorch_core::models::contextual_multioutput`).
#[pyclass(frozen, name = "LCEMGP", module = "botorch_rs._botorch_rs")]
pub struct PyLCEMGP {
    inner: RwLock<Arc<LCEMGP>>,
}

impl PyLCEMGP {
    /// The current model.
    pub fn snapshot(&self) -> Arc<LCEMGP> {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn replace(&self, model: LCEMGP) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(model);
    }

    /// Applies `f` to a copy of the model and installs the copy on success.
    fn update<F>(&self, f: F) -> PyResult<()>
    where
        F: FnOnce(&mut LCEMGP) -> botorch_core::Result<()>,
    {
        let mut model = (*self.snapshot()).clone();
        f(&mut model).into_py_result()?;
        self.replace(model);
        Ok(())
    }

    /// Sets the raw parameter named `name` (flat values).
    fn set_named(&self, name: &str, values: &[f64]) -> PyResult<()> {
        let model = self.snapshot();
        let mut raw = model.raw_parameters();
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

fn matrices(arrays: Vec<Array>, name: &str) -> PyResult<Vec<Matrix>> {
    arrays.iter().map(|a| a.matrix(name)).collect()
}

fn observation_noise(noise: Option<Array>) -> PyResult<Option<Vec<f64>>> {
    noise
        .map(|array| {
            let matrix = array.matrix("noise")?;
            if matrix.ncols() != 1 {
                return Err(dimension_error("noise must have shape (n, 1)."));
            }
            Ok(matrix.column(0).iter().copied().collect())
        })
        .transpose()
}

#[pymethods]
impl PyLCEMGP {
    /// Builds the model from `train_x` (`n x (d + 1)` with the task column
    /// `task_feature`) and `train_y` (`n x 1`). `outcome_transform=None`
    /// with `standardize_by_default=True` is BoTorch's `DEFAULT`
    /// (`Standardize(m=1)`); with `False` no outcome transform is used.
    #[new]
    #[pyo3(signature = (
        train_x,
        train_y,
        task_feature,
        output_tasks=None,
        all_tasks=None,
        context_cat_feature=None,
        context_emb_feature=None,
        embs_dim_list=None,
        outcome_transform=None,
        standardize_by_default=true,
        covar_module=None,
        embedding_weights=None,
        seed=None,
        provided_embedding=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        train_x: Array,
        train_y: Array,
        task_feature: isize,
        output_tasks: Option<Vec<i64>>,
        all_tasks: Option<Vec<i64>>,
        context_cat_feature: Option<Array>,
        context_emb_feature: Option<Array>,
        embs_dim_list: Option<Vec<usize>>,
        outcome_transform: Option<PyRef<'_, PyStandardize>>,
        standardize_by_default: bool,
        covar_module: Option<PyRef<'_, PyKernelSpec>>,
        embedding_weights: Option<Vec<Array>>,
        seed: Option<u64>,
        provided_embedding: bool,
    ) -> PyResult<Self> {
        let x = train_x.matrix("train_X")?;
        let y = train_y.matrix("train_Y")?;
        let outcome = match (outcome_transform, standardize_by_default) {
            (Some(st), _) => OutcomeTransformChoice::Standardize(
                botorch_core::Standardize::new().with_min_stdv(st.inner[0].min_stdv()),
            ),
            (None, true) => OutcomeTransformChoice::Default,
            (None, false) => OutcomeTransformChoice::None,
        };
        let options = LCEMGPOptions {
            output_tasks,
            all_tasks,
            context_cat_feature: context_cat_feature
                .map(|a| a.matrix("context_cat_feature"))
                .transpose()?,
            context_emb_feature: context_emb_feature
                .map(|a| a.matrix("context_emb_feature"))
                .transpose()?,
            embs_dim_list,
            outcome_transform: outcome,
            covar_module: covar_module.map(|c| c.inner),
            task_covar_module: None,
            embedding_weights: embedding_weights
                .map(|w| matrices(w, "embedding_weights"))
                .transpose()?,
            seed,
            provided_embedding,
        };
        let model =
            threads::run(py, || LCEMGP::new(x, y, task_feature, options)).into_py_result()?;
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

    /// Condition on original-space targets and optional transformed-space noise.
    #[pyo3(signature = (x, y, noise=None))]
    fn condition_on_observations(
        &self,
        py: Python<'_>,
        x: Array,
        y: Array,
        noise: Option<Array>,
    ) -> PyResult<Self> {
        let (x, y) = (x.matrix("X")?, y.matrix("Y")?);
        let values = observation_noise(noise)?;
        let model = self.snapshot();
        let conditioned = threads::run(py, || {
            model.condition_on_observations(&x, &y, values.as_deref())
        })
        .into_py_result()?;
        Ok(Self {
            inner: RwLock::new(Arc::new(conditioned)),
        })
    }

    /// Multiple targets at shared inputs reuse the native covariance factor.
    #[pyo3(signature = (x, ys, noise=None))]
    fn condition_on_observations_many(
        &self,
        py: Python<'_>,
        x: Array,
        ys: Vec<Array>,
        noise: Option<Array>,
    ) -> PyResult<Vec<Self>> {
        let x = x.matrix("X")?;
        let ys = matrices(ys, "Y")?;
        let values = observation_noise(noise)?;
        let model = self.snapshot();
        let conditioned = threads::run(py, || {
            model.condition_on_observations_many(&x, &ys, values.as_deref())
        })
        .into_py_result()?;
        Ok(conditioned
            .into_iter()
            .map(|m| Self {
                inner: RwLock::new(Arc::new(m)),
            })
            .collect())
    }

    // ------------------------------------------------------------------
    // Data and configuration
    // ------------------------------------------------------------------

    /// The number of non-task features `d`.
    #[getter]
    fn dim(&self) -> usize {
        self.snapshot().dim()
    }

    /// Number of training points `n`.
    #[getter]
    fn num_train(&self) -> usize {
        self.snapshot().num_train()
    }

    /// The number of tasks.
    #[getter]
    fn num_tasks(&self) -> usize {
        self.snapshot().num_tasks()
    }

    /// The number of outputs (output tasks).
    #[getter]
    fn num_outputs(&self) -> usize {
        self.snapshot().output_task_indices().len()
    }

    /// The (non-negative) column of the task feature.
    #[getter]
    fn task_feature(&self) -> usize {
        self.snapshot().task_layout().task_feature()
    }

    /// The sorted task values.
    #[getter]
    fn all_tasks(&self) -> Vec<i64> {
        self.snapshot().task_layout().all_tasks().to_vec()
    }

    /// The output tasks (task values).
    #[getter]
    fn output_tasks(&self) -> Vec<i64> {
        self.snapshot().task_layout().output_tasks().to_vec()
    }

    /// The task indices with training data.
    #[getter]
    fn observed_task_indices(&self) -> Vec<usize> {
        let m = self.snapshot();
        (0..m.num_tasks())
            .filter(|&k| m.task_layout().is_observed(k))
            .collect()
    }

    /// Whether task values are task indices (BoTorch's task mapper is None).
    #[getter]
    fn tasks_are_indices(&self) -> bool {
        self.snapshot().task_layout().is_identity()
    }

    /// `(num_embeddings, embedding_dim)` of every embedding layer.
    #[getter]
    fn embedding_shapes(&self) -> Vec<(usize, usize)> {
        self.snapshot().embedding_shapes().to_vec()
    }

    /// The dimension of the task embeddings (learned plus fixed).
    #[getter]
    fn embedding_dim(&self) -> usize {
        self.snapshot().embedding_dim()
    }

    /// The embedding index of every categorical feature per context row.
    fn context_cat_feature(&self) -> Vec<Vec<usize>> {
        self.snapshot().context_cat_feature().to_vec()
    }

    /// The fixed context embeddings, if any.
    fn context_emb_feature(&self) -> Option<ArrayOut> {
        self.snapshot().context_emb_feature().map(matrix_out)
    }

    /// Training inputs including the task column (`n x (d + 1)`).
    fn train_inputs(&self) -> ArrayBuf {
        matrix_out(self.snapshot().train_inputs()).into()
    }

    /// Training targets in the original space (`n x 1`).
    fn train_targets(&self) -> ArrayBuf {
        matrix_out(self.snapshot().train_targets()).into()
    }

    /// Training targets after the outcome transform (`n`).
    fn transformed_train_targets(&self) -> ArrayBuf {
        let m = self.snapshot();
        let t = m.transformed_train_targets();
        ArrayBuf(t.as_slice().to_vec(), vec![t.len()])
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

    /// The data kernel configuration.
    fn covar_module(&self) -> PyKernelSpec {
        PyKernelSpec {
            inner: *self.snapshot().covar_module(),
        }
    }

    /// The task kernel configuration.
    fn task_covar_module(&self) -> PyKernelSpec {
        PyKernelSpec {
            inner: *self.snapshot().task_covar_module(),
        }
    }

    /// Replaces the task kernel (its hyperparameters start at the kernel's
    /// initial values).
    fn set_task_covar_module(&self, spec: PyRef<'_, PyKernelSpec>) -> PyResult<()> {
        let spec = spec.inner;
        self.update(|m| m.set_task_covar_module(spec))
    }

    /// The per-task noise configuration (prior and constraint).
    fn likelihood_spec(&self) -> PyLikelihoodSpec {
        PyLikelihoodSpec {
            inner: *self.snapshot().likelihood(),
        }
    }

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------

    /// BoTorch parameter names and sizes in flat order.
    fn parameter_names(&self) -> Vec<(String, usize)> {
        self.snapshot().parameter_names()
    }

    /// BoTorch parameter names and shapes.
    fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        self.snapshot().parameter_shapes()
    }

    /// The raw hyperparameters in flat order (BoTorch's `named_parameters()`).
    fn raw_parameters(&self) -> Vec<f64> {
        self.snapshot().raw_parameters().as_slice().to_vec()
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
        self.snapshot().raw_parameter_bounds()
    }

    /// Whether this is the provided-embedding variant.
    #[getter]
    fn is_provided_embedding(&self) -> bool {
        self.snapshot().is_provided_embedding()
    }

    /// The context-specific weight of the provided-embedding variant.
    #[getter]
    fn context_specific_weight(&self) -> Option<f64> {
        self.snapshot().context_specific_weight()
    }

    /// The Unity project's `eval()`: the constant means of tasks without data
    /// become the average of the observed tasks' constants.
    fn eval_unobserved_means(&self) {
        let model = self.snapshot();
        let mut updated = (*model).clone();
        updated.eval_unobserved_means();
        if updated.constant() != model.constant() {
            self.replace(updated);
        }
    }

    /// The per-task noise variances (transformed space).
    #[getter]
    fn noise(&self) -> Vec<f64> {
        self.snapshot().noise().as_slice().to_vec()
    }

    /// The per-task constant means (transformed space).
    #[getter]
    fn constant(&self) -> Vec<f64> {
        self.snapshot().constant().as_slice().to_vec()
    }

    /// The lengthscales of the data kernel (`"data"`) or the task kernel
    /// (`"task"`).
    fn lengthscale_of(&self, which: &str) -> PyResult<Vec<f64>> {
        let m = self.snapshot();
        Ok(match which_kernel(which)? {
            Which::Data => m.lengthscale(),
            Which::Task => m.task_lengthscale(),
        }
        .as_slice()
        .to_vec())
    }

    /// The outputscale of the data or task kernel (`None` without
    /// `ScaleKernel`).
    fn outputscale_of(&self, which: &str) -> PyResult<Option<f64>> {
        let m = self.snapshot();
        Ok(match which_kernel(which)? {
            Which::Data => m.outputscale(),
            Which::Task => m.task_outputscale(),
        })
    }

    /// Sets the lengthscales of the data or task kernel (actual values,
    /// broadcast from one value or one per dimension).
    fn set_lengthscale_of(&self, which: &str, values: Vec<f64>) -> PyResult<()> {
        let m = self.snapshot();
        let (spec, name) = match which_kernel(which)? {
            Which::Data => (*m.covar_module(), "covar_module"),
            Which::Task => (*m.task_covar_module(), "task_covar_module_base"),
        };
        let d = spec.ard_num_dims;
        let values = broadcast(&values, d, "lengthscale")?;
        let raw: Vec<f64> = values
            .iter()
            .map(|&v| spec.lengthscale_constraint.inverse_transform(v))
            .collect();
        let full = if spec.scale.is_some() {
            format!("{name}.base_kernel.raw_lengthscale")
        } else {
            format!("{name}.raw_lengthscale")
        };
        self.set_named(&full, &raw)
    }

    /// Sets the outputscale of the data or task kernel.
    fn set_outputscale_of(&self, which: &str, value: f64) -> PyResult<()> {
        let m = self.snapshot();
        let (spec, name) = match which_kernel(which)? {
            Which::Data => (*m.covar_module(), "covar_module"),
            Which::Task => (*m.task_covar_module(), "task_covar_module_base"),
        };
        let Some(scale) = spec.scale else {
            return Err(crate::errors::unsupported(
                "The covariance module has no outputscale.",
            ));
        };
        let raw = scale.outputscale_constraint.inverse_transform(value);
        self.set_named(&format!("{name}.raw_outputscale"), &[raw])
    }

    /// Sets the per-task noise variances (actual values, broadcast from one
    /// value or one per task).
    fn set_noise(&self, values: Vec<f64>) -> PyResult<()> {
        let m = self.snapshot();
        let values = broadcast(&values, m.num_tasks(), "noise")?;
        let c = m.likelihood().noise_constraint;
        let raw: Vec<f64> = values.iter().map(|&v| c.inverse_transform(v)).collect();
        self.set_named("likelihood.noise_covar.raw_noise", &raw)
    }

    /// The stored embedding weights (`num_embeddings x dim` each).
    fn embedding_weights(&self) -> Vec<ArrayOut> {
        self.snapshot()
            .embedding_weights()
            .iter()
            .map(matrix_out)
            .collect()
    }

    /// Applies the in-place renormalization of BoTorch's forward pass
    /// (`torch.nn.Embedding(max_norm=1.0)`) to the stored embedding weights.
    fn renorm_embeddings(&self) {
        let model = self.snapshot();
        let mut renormed = (*model).clone();
        renormed.renorm_embeddings();
        if renormed.embedding_weights() != model.embedding_weights() {
            self.replace(renormed);
        }
    }

    /// The embeddings of every context row (`_task_embeddings()`).
    fn task_embeddings(&self) -> ArrayOut {
        matrix_out(&self.snapshot().task_embeddings())
    }

    /// The task covariance over the context rows (`_eval_context_covar()`).
    fn task_covariance(&self, py: Python<'_>) -> PyResult<ArrayOut> {
        let m = self.snapshot();
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
        let model = self.snapshot();
        let value = threads::run(py, || match &raw_parameters {
            Some(p) => model.mll_value_and_grad(p).map(|(v, _)| v),
            None => model.mll_value(),
        })
        .into_py_result()?;
        Ok((vec![value], Vec::new()))
    }

    /// The exact MLL and its gradient with respect to the flat raw
    /// hyperparameters (torch's gradient for the renormalized embedding
    /// weights).
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
            return Err(crate::errors::value_error(
                "gtol and ftol must be non-negative",
            ));
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
}

enum Which {
    Data,
    Task,
}

fn which_kernel(which: &str) -> PyResult<Which> {
    match which {
        "data" => Ok(Which::Data),
        "task" => Ok(Which::Task),
        other => Err(crate::errors::value_error(format!(
            "unknown kernel {other:?} (\"data\" or \"task\")"
        ))),
    }
}

/// One value repeated `n` times, or `n` values.
fn broadcast(values: &[f64], n: usize, name: &str) -> PyResult<Vec<f64>> {
    match values.len() {
        1 => Ok(vec![values[0]; n]),
        k if k == n => Ok(values.to_vec()),
        k => Err(dimension_error(format!(
            "{name} has {n} values, got {k} (a value of this shape cannot be broadcast)"
        ))),
    }
}
