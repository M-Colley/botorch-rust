//! Acquisition functions (`botorch.acquisition`) and `prune_inferior_points`.
//!
//! One Python class wraps every acquisition function of `botorch_core`; the
//! `botorch_rs.acquisition` package provides the BoTorch-named classes on top
//! of it. Acquisition functions are built on a [`PyModel`] handle: the model
//! snapshot that is current at construction (see [`crate::models`]), seen
//! through the posterior transform if one is given.
//!
//! MC objectives and outcome constraints arrive as small specifications
//! built by the Python layer: `("identity",)`, `("linear", weights)`,
//! `("callable", objective)` (BoTorch's `objective(samples, X=X)`, evaluated
//! through [`crate::callbacks`]), `("constrained", objective, constraints,
//! infeasible_cost, eta)` or `("native", obj)` for an objective evaluated in
//! Rust (a risk measure, [`crate::risk_measures::PyRiskMeasure`]); a list of
//! callables with one temperature each (and optionally BoTorch's
//! per-constraint `fat` list) for outcome constraints.

use std::sync::{Arc, Mutex};

use botorch_core::acquisition::{
    prune_inferior_points, AcquisitionFunction, ConstrainedExpectedImprovement,
    ConstrainedMcObjective, ExpectedImprovement, IdentityMcObjective, LinearMcObjective,
    LogConstrainedExpectedImprovement, LogExpectedImprovement, LogProbabilityOfImprovement,
    McAcquisitionOptions, McObjective, OutputConstraint, PosteriorMean, PosteriorStandardDeviation,
    ProbabilityOfImprovement, PruneOptions, QExpectedImprovement, QLogExpectedImprovement,
    QLogExpectedImprovementOptions, QLogNoisyExpectedImprovement,
    QLogNoisyExpectedImprovementOptions, QLogProbabilityOfFeasibility,
    QLogProbabilityOfFeasibilityOptions, QLowerConfidenceBound, QNoisyExpectedImprovement,
    QPosteriorStandardDeviation, QProbabilityOfImprovement, QSimpleRegret, QUpperConfidenceBound,
    UpperConfidenceBound,
};
use botorch_core::utils::objective::{McConstraint, OutcomeConstraints};
use botorch_core::{Matrix, Model, Vector};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::arrays::{broadcast_shapes, matrix_out, Array, ArrayBuf, ArrayOut};
use crate::callbacks::PyCallback;
use crate::errors::{dimension_error, unsupported, value_error, IntoPyResult};
use crate::models::PyModel;
use crate::sampling::PySobolQMCNormalSampler;
use crate::threads;

/// An acquisition function over candidate sets `X` (`batch x q x d`).
#[pyclass(
    frozen,
    name = "AcquisitionFunction",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyAcquisitionFunction {
    /// The acquisition function.
    pub inner: Arc<dyn AcquisitionFunction>,
    /// The same object when it is a qLogNEI (for its baseline accessors).
    nei: Option<Arc<QLogNoisyExpectedImprovement>>,
    linear_nei: Option<Arc<QNoisyExpectedImprovement>>,
    last_incumbents: Mutex<Option<ArrayBuf>>,
    name: &'static str,
    objective_batch: Vec<usize>,
}

/// Immutable reverse tape from one parallel acquisition evaluation.
#[pyclass(
    frozen,
    name = "AcquisitionParameterTape",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyAcquisitionParameterTape {
    records: Vec<botorch_core::acquisition::adjoints::ParameterAdjoints>,
    batch_shape: Vec<usize>,
}

#[pymethods]
impl PyAcquisitionParameterTape {
    fn parameter_vjp(
        &self,
        py: Python<'_>,
        weights: Array,
        parameter_count: usize,
    ) -> PyResult<ArrayBuf> {
        if weights.1 != self.batch_shape || weights.0.len() != self.records.len() {
            return Err(crate::errors::dimension_error(
                "Acquisition adjoints must match candidate batch shape",
            ));
        }
        let gradient = threads::run(py, || {
            let mut gradient = Vector::zeros(parameter_count);
            for (record, &weight) in self.records.iter().zip(&weights.0) {
                if weight == 0.0 {
                    continue;
                }
                let part = record.parameter_vjp(weight)?;
                if part.len() != parameter_count {
                    return Err(botorch_core::BotorchError::DimensionMismatch(
                        "Acquisition tape parameter layout differs from its captured leaves".into(),
                    ));
                }
                gradient += part;
            }
            Ok(gradient)
        })
        .into_py_result()?;
        Ok(ArrayBuf(
            gradient.as_slice().to_vec(),
            vec![parameter_count],
        ))
    }
}

impl PyAcquisitionFunction {
    pub(crate) fn new_plain(inner: Arc<dyn AcquisitionFunction>, name: &'static str) -> Self {
        Self {
            inner,
            nei: None,
            linear_nei: None,
            last_incumbents: Mutex::new(None),
            name,
            objective_batch: Vec::new(),
        }
    }

    /// An acquisition function built on a qLogNEI (whose baseline accessors
    /// it exposes), e.g. qLogNParEGO.
    pub(crate) fn with_nei(
        inner: Arc<dyn AcquisitionFunction>,
        nei: Arc<QLogNoisyExpectedImprovement>,
        name: &'static str,
    ) -> Self {
        Self {
            inner,
            nei: Some(nei),
            linear_nei: None,
            last_incumbents: Mutex::new(None),
            name,
            objective_batch: Vec::new(),
        }
    }

    pub(crate) fn with_linear_nei(nei: Arc<QNoisyExpectedImprovement>) -> Self {
        Self {
            inner: nei.clone(),
            nei: None,
            linear_nei: Some(nei),
            last_incumbents: Mutex::new(None),
            name: "qNoisyExpectedImprovement",
            objective_batch: Vec::new(),
        }
    }

    pub(crate) fn with_objective_batch(mut self, shape: Option<Vec<usize>>) -> Self {
        self.objective_batch = shape.unwrap_or_default();
        self
    }

    fn check_objective_batch(&self, batch: &[usize]) -> PyResult<()> {
        if !self.objective_batch.is_empty()
            && broadcast_shapes(batch, &self.objective_batch).as_deref() != Some(batch)
        {
            return Err(dimension_error(format!(
                "candidate batch {batch:?} must include the learned preference batch {:?}; expand X explicitly",
                self.objective_batch
            )));
        }
        Ok(())
    }

    fn in_objective_batch<T>(&self, batch: &[usize], index: usize, f: impl FnOnce() -> T) -> T {
        if self.objective_batch.is_empty() {
            f()
        } else {
            crate::callbacks::context::with_context(batch, Some(index), f)
        }
    }

    fn captured_evaluation(
        &self,
        x: &Matrix,
        with_grad: bool,
    ) -> botorch_core::Result<(f64, Option<Matrix>, Vec<f64>)> {
        match (&self.nei, &self.linear_nei) {
            (Some(nei), _) => nei.forward_with_incumbents(x, with_grad),
            (_, Some(nei)) => nei.forward_with_incumbents(x, with_grad),
            _ => unreachable!("capture is only requested for NEI acquisition functions"),
        }
    }

    fn store_incumbents(&self, best: &[Vec<f64>], batch: &[usize]) {
        let samples = best.first().map_or_else(
            || {
                self.nei.as_ref().map_or_else(
                    || {
                        self.linear_nei
                            .as_ref()
                            .map_or(0, |n| n.sampler().num_samples())
                    },
                    |n| n.sampler().num_samples(),
                )
            },
            Vec::len,
        );
        let mut values = Vec::with_capacity(samples * best.len());
        for sample in 0..samples {
            for candidate in best {
                values.push(candidate[sample]);
            }
        }
        let mut shape = vec![samples];
        shape.extend_from_slice(batch);
        *self
            .last_incumbents
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(ArrayBuf(values, shape));
    }

    fn nei(&self) -> PyResult<&Arc<QLogNoisyExpectedImprovement>> {
        self.nei.as_ref().ok_or_else(|| {
            unsupported(format!(
                "{} has no baseline (only qLogNoisyExpectedImprovement does).",
                self.name
            ))
        })
    }
}

pub(crate) fn model_of(model: &PyModel) -> Arc<dyn Model> {
    model.inner.clone()
}

fn require(value: Option<f64>, name: &str, acqf: &str) -> PyResult<f64> {
    value.ok_or_else(|| value_error(format!("{acqf} requires `{name}`.")))
}

pub(crate) fn sampler_of(
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
) -> Option<Arc<botorch_core::SobolQMCNormalSampler>> {
    sampler.map(|s| Arc::clone(&s.inner))
}

/// Splits `X` (`... x q x d`) into candidate sets.
fn candidate_sets(x: &Array) -> PyResult<(Vec<usize>, Vec<Matrix>)> {
    x.batches("X")
}

/// An MC objective from its specification (see the module documentation).
pub(crate) fn objective_of(
    spec: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<dyn McObjective>>> {
    let Some(spec) = spec.filter(|s| !s.is_none()) else {
        return Ok(None);
    };
    let kind: String = spec.get_item(0)?.extract()?;
    let objective: Arc<dyn McObjective> = match kind.as_str() {
        "identity" => Arc::new(IdentityMcObjective),
        "linear" => {
            let w: Vec<f64> = spec.get_item(1)?.extract()?;
            Arc::new(LinearMcObjective::new(Vector::from_vec(w)).into_py_result()?)
        }
        "callable" => Arc::new(PyCallback::objective(spec.get_item(1)?.unbind(), true)),
        "constrained" => {
            let inner: Arc<dyn McObjective> =
                Arc::new(PyCallback::objective(spec.get_item(1)?.unbind(), false));
            let constraints: Vec<Py<PyAny>> = spec.get_item(2)?.extract()?;
            let infeasible_cost: f64 = spec.get_item(3)?.extract()?;
            let eta: Vec<f64> = spec.get_item(4)?.extract()?;
            Arc::new(
                ConstrainedMcObjective::new(
                    inner,
                    constraint_callbacks(constraints),
                    infeasible_cost,
                    eta,
                )
                .into_py_result()?,
            )
        }
        "penalized" => crate::so_wrappers::penalized_objective(spec)?,
        "native" => {
            let obj: PyRef<'_, crate::risk_measures::PyRiskMeasure> =
                spec.get_item(1)?.extract()?;
            Arc::clone(&obj.inner)
        }
        other => return Err(unsupported(format!("Unknown MC objective {other:?}."))),
    };
    Ok(Some(objective))
}

pub(crate) fn constraint_callbacks(constraints: Vec<Py<PyAny>>) -> Vec<Arc<dyn McConstraint>> {
    constraints
        .into_iter()
        .map(|c| Arc::new(PyCallback::constraint(c)) as Arc<dyn McConstraint>)
        .collect()
}

/// Outcome constraints from callables and one temperature per constraint.
pub(crate) fn constraints_of(
    constraints: Option<Vec<Py<PyAny>>>,
    eta: Option<Vec<f64>>,
) -> PyResult<Option<OutcomeConstraints>> {
    constraints
        .map(|cs| {
            let k = cs.len();
            let eta = eta.unwrap_or_else(|| vec![1e-3; k]);
            OutcomeConstraints::new(constraint_callbacks(cs), eta).into_py_result()
        })
        .transpose()
}

/// The analytic acquisition functions (BoTorch class names).
const ANALYTIC: [&str; 7] = [
    "ExpectedImprovement",
    "LogExpectedImprovement",
    "ProbabilityOfImprovement",
    "LogProbabilityOfImprovement",
    "UpperConfidenceBound",
    "PosteriorMean",
    "PosteriorStandardDeviation",
];

/// The sample-reducing Monte-Carlo acquisition functions.
const MONTE_CARLO: [&str; 7] = [
    "qExpectedImprovement",
    "qProbabilityOfImprovement",
    "qLogExpectedImprovement",
    "qUpperConfidenceBound",
    "qSimpleRegret",
    "qLowerConfidenceBound",
    "qPosteriorStandardDeviation",
];

/// The `'static` name in `names` equal to `name`.
fn static_name(name: &str, names: &[&'static str]) -> PyResult<&'static str> {
    names
        .iter()
        .find(|n| **n == name)
        .copied()
        .ok_or_else(|| unsupported(format!("Unknown acquisition function {name:?}.")))
}

#[pymethods]
impl PyAcquisitionFunction {
    /// An analytic acquisition function (`q = 1`): `"ExpectedImprovement"`,
    /// `"LogExpectedImprovement"`, `"ProbabilityOfImprovement"`,
    /// `"LogProbabilityOfImprovement"` (with `best_f`),
    /// `"UpperConfidenceBound"` (with `beta`), `"PosteriorMean"` or
    /// `"PosteriorStandardDeviation"`.
    #[staticmethod]
    #[pyo3(signature = (name, model, best_f=None, beta=None, maximize=true))]
    fn analytic(
        name: &str,
        model: PyRef<'_, PyModel>,
        best_f: Option<f64>,
        beta: Option<f64>,
        maximize: bool,
    ) -> PyResult<Self> {
        let name = static_name(name, &ANALYTIC)?;
        let m = model_of(&model);
        let best_f = || require(best_f, "best_f", name);
        let inner: Arc<dyn AcquisitionFunction> = match name {
            "ExpectedImprovement" => {
                Arc::new(ExpectedImprovement::new(m, best_f()?, maximize).into_py_result()?)
            }
            "LogExpectedImprovement" => {
                Arc::new(LogExpectedImprovement::new(m, best_f()?, maximize).into_py_result()?)
            }
            "ProbabilityOfImprovement" => {
                Arc::new(ProbabilityOfImprovement::new(m, best_f()?, maximize).into_py_result()?)
            }
            "LogProbabilityOfImprovement" => {
                Arc::new(LogProbabilityOfImprovement::new(m, best_f()?, maximize).into_py_result()?)
            }
            "UpperConfidenceBound" => Arc::new(
                UpperConfidenceBound::new(m, require(beta, "beta", name)?, maximize)
                    .into_py_result()?,
            ),
            "PosteriorMean" => Arc::new(PosteriorMean::new(m, maximize).into_py_result()?),
            _ => Arc::new(PosteriorStandardDeviation::new(m, maximize).into_py_result()?),
        };
        Ok(Self::new_plain(inner, name))
    }

    /// A Monte-Carlo acquisition function: `"qExpectedImprovement"`,
    /// `"qProbabilityOfImprovement"` (with `best_f`, `tau`),
    /// `"qLogExpectedImprovement"` (with `best_f`, `fat`, `tau_max`,
    /// `tau_relu`), `"qUpperConfidenceBound"` (with `beta`) or
    /// `"qSimpleRegret"`, with an optional MC `objective` and outcome
    /// `constraints` (callables, one temperature `eta` each; `fat_list` is
    /// BoTorch's per-constraint `fat` of qLogEI).
    #[staticmethod]
    #[pyo3(signature = (
        name,
        model,
        sampler=None,
        x_pending=None,
        best_f=None,
        beta=None,
        tau=1e-3,
        fat=true,
        tau_max=1e-2,
        tau_relu=1e-6,
        objective=None,
        constraints=None,
        eta=None,
        fat_list=None,
        objective_batch_shape=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn monte_carlo(
        name: &str,
        model: PyRef<'_, PyModel>,
        sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
        x_pending: Option<Array>,
        best_f: Option<f64>,
        beta: Option<f64>,
        tau: f64,
        fat: bool,
        tau_max: f64,
        tau_relu: f64,
        objective: Option<Bound<'_, PyAny>>,
        constraints: Option<Vec<Py<PyAny>>>,
        eta: Option<Vec<f64>>,
        fat_list: Option<Vec<Option<bool>>>,
        objective_batch_shape: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let name = static_name(name, &MONTE_CARLO)?;
        let m = model_of(&model);
        let sampler = sampler_of(sampler);
        let x_pending = x_pending.map(|p| p.matrix("X_pending")).transpose()?;
        let objective = objective_of(objective.as_ref())?;
        let constraints = constraints_of(constraints, eta)?;
        let options = McAcquisitionOptions {
            sampler: sampler.clone(),
            x_pending: x_pending.clone(),
            objective: objective.clone(),
            constraints: constraints.clone(),
        };
        let best_f = || require(best_f, "best_f", name);
        let inner: Arc<dyn AcquisitionFunction> = match name {
            "qExpectedImprovement" => {
                Arc::new(QExpectedImprovement::new(m, best_f()?, options).into_py_result()?)
            }
            "qProbabilityOfImprovement" => Arc::new(
                QProbabilityOfImprovement::new(m, best_f()?, tau, options).into_py_result()?,
            ),
            "qLogExpectedImprovement" => Arc::new(
                QLogExpectedImprovement::new(
                    m,
                    best_f()?,
                    QLogExpectedImprovementOptions {
                        sampler,
                        x_pending,
                        fat,
                        tau_max,
                        tau_relu,
                        objective,
                        constraints,
                        constraint_fat: fat_list,
                    },
                )
                .into_py_result()?,
            ),
            "qUpperConfidenceBound" => Arc::new(
                QUpperConfidenceBound::new(m, require(beta, "beta", name)?, options)
                    .into_py_result()?,
            ),
            "qLowerConfidenceBound" => Arc::new(
                QLowerConfidenceBound::new(m, require(beta, "beta", name)?, options)
                    .into_py_result()?,
            ),
            "qPosteriorStandardDeviation" => {
                Arc::new(QPosteriorStandardDeviation::new(m, options).into_py_result()?)
            }
            _ => Arc::new(QSimpleRegret::new(m, options).into_py_result()?),
        };
        Ok(Self::new_plain(inner, name).with_objective_batch(objective_batch_shape))
    }

    /// `qLogProbabilityOfFeasibility(model, constraints, ...)` with outcome
    /// constraints (callables, one temperature `eta` each).
    #[staticmethod]
    #[pyo3(signature = (
        model,
        constraints,
        eta,
        sampler=None,
        x_pending=None,
        fat=true,
        tau_max=1e-2,
        fat_list=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn q_log_probability_of_feasibility(
        model: PyRef<'_, PyModel>,
        constraints: Vec<Py<PyAny>>,
        eta: Vec<f64>,
        sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
        x_pending: Option<Array>,
        fat: bool,
        tau_max: f64,
        fat_list: Option<Vec<Option<bool>>>,
    ) -> PyResult<Self> {
        let m = model_of(&model);
        let constraints =
            constraints_of(Some(constraints), Some(eta))?.expect("constraints are given");
        let acqf = QLogProbabilityOfFeasibility::new(
            m,
            constraints,
            QLogProbabilityOfFeasibilityOptions {
                sampler: sampler_of(sampler),
                x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
                fat,
                tau_max,
                constraint_fat: fat_list,
            },
        )
        .into_py_result()?;
        Ok(Self::new_plain(
            Arc::new(acqf),
            "qLogProbabilityOfFeasibility",
        ))
    }

    /// `"LogConstrainedExpectedImprovement"` or
    /// `"ConstrainedExpectedImprovement"` (BoTorch's analytic constrained EI)
    /// with the objective output `objective_index` and constraints
    /// `(output index, lower, upper)` (`None`: unbounded).
    #[staticmethod]
    #[pyo3(signature = (name, model, best_f, objective_index, constraints, maximize=true))]
    fn constrained(
        name: &str,
        model: PyRef<'_, PyModel>,
        best_f: f64,
        objective_index: usize,
        constraints: Vec<OutputConstraint>,
        maximize: bool,
    ) -> PyResult<Self> {
        let m = model_of(&model);
        let (inner, name): (Arc<dyn AcquisitionFunction>, &'static str) = match name {
            "LogConstrainedExpectedImprovement" => (
                Arc::new(
                    LogConstrainedExpectedImprovement::new(
                        m,
                        best_f,
                        objective_index,
                        constraints,
                        maximize,
                    )
                    .into_py_result()?,
                ),
                "LogConstrainedExpectedImprovement",
            ),
            "ConstrainedExpectedImprovement" => (
                Arc::new(
                    ConstrainedExpectedImprovement::new(
                        m,
                        best_f,
                        objective_index,
                        constraints,
                        maximize,
                    )
                    .into_py_result()?,
                ),
                "ConstrainedExpectedImprovement",
            ),
            other => {
                return Err(unsupported(format!(
                    "Unknown constrained acquisition function {other:?}."
                )))
            }
        };
        Ok(Self::new_plain(inner, name))
    }

    /// `qLogNoisyExpectedImprovement(model, X_baseline, ...)`; the baseline is
    /// pruned with `prune_inferior_points` (`prune_num_samples` samples,
    /// `prune_max_frac`, seeded by `prune_seed`, under the objective and the
    /// constraints) if `prune_baseline`.
    #[staticmethod]
    #[pyo3(signature = (
        model,
        x_baseline,
        sampler=None,
        x_pending=None,
        fat=true,
        prune_baseline=true,
        cache_root=true,
        prune_num_samples=2048,
        prune_max_frac=1.0,
        prune_seed=None,
        tau_max=1e-2,
        tau_relu=1e-6,
        incremental=true,
        objective=None,
        constraints=None,
        eta=None,
        fat_list=None,
        lower_bound_points=None,
        objective_batch_shape=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn q_log_noisy_expected_improvement(
        py: Python<'_>,
        model: PyRef<'_, PyModel>,
        x_baseline: Array,
        sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
        x_pending: Option<Array>,
        fat: bool,
        prune_baseline: bool,
        cache_root: bool,
        prune_num_samples: usize,
        prune_max_frac: f64,
        prune_seed: Option<u64>,
        tau_max: f64,
        tau_relu: f64,
        incremental: bool,
        objective: Option<Bound<'_, PyAny>>,
        constraints: Option<Vec<Py<PyAny>>>,
        eta: Option<Vec<f64>>,
        fat_list: Option<Vec<Option<bool>>>,
        lower_bound_points: Option<Array>,
        objective_batch_shape: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let m = model_of(&model);
        let x_baseline = x_baseline.matrix("X_baseline")?;
        let options = QLogNoisyExpectedImprovementOptions {
            sampler: sampler_of(sampler),
            x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
            fat,
            prune_baseline,
            cache_root,
            prune_options: PruneOptions {
                num_samples: prune_num_samples,
                max_frac: prune_max_frac,
                seed: prune_seed,
                ..Default::default()
            },
            tau_max,
            tau_relu,
            incremental,
            objective: objective_of(objective.as_ref())?,
            constraints: constraints_of(constraints, eta)?,
            constraint_fat: fat_list,
            lower_bound_points: lower_bound_points
                .map(|p| p.matrix("lower_bound_points"))
                .transpose()?,
        };
        let acqf = threads::run(py, || {
            crate::callbacks::context::with_context(&[], None, || {
                QLogNoisyExpectedImprovement::new(m, &x_baseline, options)
            })
        })
        .into_py_result()?;
        let acqf = Arc::new(acqf);
        Ok(
            Self::with_nei(acqf.clone(), acqf, "qLogNoisyExpectedImprovement")
                .with_objective_batch(objective_batch_shape),
        )
    }

    /// The BoTorch class name.
    #[getter]
    fn name(&self) -> &'static str {
        self.name
    }

    /// Input dimension `d`.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// The required `q` (1 for analytic acquisition functions), or `None`.
    #[getter]
    fn expected_q(&self) -> Option<usize> {
        self.inner.expected_q()
    }

    /// Whether the values are non-negative (`initialize_q_batch_nonneg`).
    #[getter]
    fn is_nonnegative(&self) -> bool {
        self.inner.is_nonnegative()
    }

    /// Whether the values are on a log scale.
    #[getter]
    fn is_log(&self) -> bool {
        self.inner.is_log()
    }

    /// Values at the candidate sets `X` (`... x q x d`); shape `...`.
    fn forward(&self, py: Python<'_>, x: Array) -> PyResult<ArrayBuf> {
        let (batch, sets) = candidate_sets(&x)?;
        self.check_objective_batch(&batch)?;
        if self.nei.is_some() || self.linear_nei.is_some() {
            let results = threads::run(py, || {
                sets.par_iter()
                    .enumerate()
                    .map(|(i, s)| {
                        self.in_objective_batch(&batch, i, || self.captured_evaluation(s, false))
                    })
                    .collect::<botorch_core::Result<Vec<_>>>()
            })
            .into_py_result()?;
            self.store_incumbents(
                &results.iter().map(|r| r.2.clone()).collect::<Vec<_>>(),
                &batch,
            );
            return Ok(ArrayBuf(results.into_iter().map(|r| r.0).collect(), batch));
        }
        let values = threads::run(py, || {
            if self.objective_batch.is_empty() {
                self.inner.forward_many(&sets)
            } else {
                sets.par_iter()
                    .enumerate()
                    .map(|(i, s)| self.in_objective_batch(&batch, i, || self.inner.forward(s)))
                    .collect()
            }
        })
        .into_py_result()?;
        Ok(ArrayBuf(values, batch))
    }

    /// Values and exact native input gradients; NEI state is captured in
    /// candidate order after parallel evaluation, never in a shared worker slot.
    fn forward_with_grad(&self, py: Python<'_>, x: Array) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let (batch, sets) = candidate_sets(&x)?;
        self.check_objective_batch(&batch)?;
        let capture = self.nei.is_some() || self.linear_nei.is_some();
        let results = threads::run(py, || {
            sets.par_iter()
                .enumerate()
                .map(|(i, s)| {
                    self.in_objective_batch(&batch, i, || {
                        if capture {
                            self.captured_evaluation(s, true)
                        } else {
                            self.inner
                                .forward_with_grad(s)
                                .map(|(v, g)| (v, Some(g), Vec::new()))
                        }
                    })
                })
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        if capture {
            self.store_incumbents(
                &results.iter().map(|r| r.2.clone()).collect::<Vec<_>>(),
                &batch,
            );
        }
        let mut values = Vec::with_capacity(results.len());
        let mut grads = Vec::with_capacity(x.0.len());
        for (v, g, _) in results {
            values.push(v);
            let (data, _) = matrix_out(&g.expect("gradient requested"));
            grads.extend(data);
        }
        Ok((ArrayBuf(values, batch), ArrayBuf(grads, x.1.clone())))
    }

    #[getter]
    fn supports_parameter_gradients(&self) -> bool {
        self.inner.supports_parameter_gradients()
    }

    #[pyo3(signature = (x, query_gradient=false))]
    fn forward_with_adjoint_tape(
        &self,
        py: Python<'_>,
        x: Array,
        query_gradient: bool,
    ) -> PyResult<(ArrayBuf, Option<ArrayBuf>, PyAcquisitionParameterTape)> {
        let (batch, sets) = candidate_sets(&x)?;
        self.check_objective_batch(&batch)?;
        let capture = self.nei.is_some() || self.linear_nei.is_some();
        let results = threads::run(py, || {
            sets.par_iter()
                .enumerate()
                .map(|(index, points)| {
                    self.in_objective_batch(&batch, index, || match (&self.nei, &self.linear_nei) {
                        (Some(nei), _) => {
                            nei.forward_with_adjoints_and_incumbents(points, query_gradient)
                        }
                        (_, Some(nei)) => {
                            nei.forward_with_adjoints_and_incumbents(points, query_gradient)
                        }
                        _ => self
                            .inner
                            .forward_with_adjoints(points, query_gradient)
                            .map(|result| (result, Vec::new())),
                    })
                })
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        if capture {
            self.store_incumbents(
                &results
                    .iter()
                    .map(|(_, best)| best.clone())
                    .collect::<Vec<_>>(),
                &batch,
            );
        }
        let mut values = Vec::with_capacity(results.len());
        let mut gradients = Vec::new();
        let mut records = Vec::with_capacity(results.len());
        for (result, _) in results {
            values.push(result.value);
            if let Some(gradient) = result.query_gradient {
                gradients.extend(matrix_out(&gradient).0);
            }
            records.push(result.posterior);
        }
        let tape = PyAcquisitionParameterTape {
            records,
            batch_shape: batch.clone(),
        };
        Ok((
            ArrayBuf(values, batch),
            query_gradient.then_some(ArrayBuf(gradients, x.1)),
            tape,
        ))
    }

    /// Cached-baseline incumbents, available before evaluation when enabled.
    fn baseline_best_f(&self) -> Option<Vec<f64>> {
        self.nei
            .as_ref()
            .and_then(|n| n.baseline_best_f())
            .or_else(|| {
                self.linear_nei
                    .as_ref()
                    .and_then(|n| n.baseline_best_f().map(<[f64]>::to_vec))
            })
    }

    /// Exact incumbents from the last completed forward, shaped sample x batch.
    fn last_baseline_best_f(&self) -> Option<ArrayBuf> {
        self.last_incumbents
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|v| ArrayBuf(v.0.clone(), v.1.clone()))
    }

    #[getter]
    fn cache_root(&self) -> bool {
        self.nei.as_ref().is_some_and(|n| n.cache_root())
            || self.linear_nei.as_ref().is_some_and(|n| n.cache_root())
    }

    /// The pending points (`m x d`), if any.
    fn x_pending(&self) -> Option<ArrayOut> {
        self.inner.x_pending().as_ref().map(matrix_out)
    }

    /// Sets (or with `None` clears) the pending points (`m x d`).
    #[pyo3(signature = (x_pending=None))]
    fn set_x_pending(&self, py: Python<'_>, x_pending: Option<Array>) -> PyResult<()> {
        let x_pending = x_pending.map(|p| p.matrix("X_pending")).transpose()?;
        let acqf = &self.inner;
        threads::run(py, || acqf.set_x_pending(x_pending.as_ref())).into_py_result()?;
        *self
            .last_incumbents
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        Ok(())
    }

    /// The (pruned) baseline of a qLogNEI (`n x d`).
    fn x_baseline(&self) -> PyResult<ArrayOut> {
        Ok(matrix_out(&self.nei()?.x_baseline()))
    }

    /// The incumbent points of a qLogNEI: the baseline and, in incremental
    /// mode, the pending points (BoTorch's `X_baseline` property).
    fn full_x_baseline(&self) -> PyResult<ArrayOut> {
        Ok(matrix_out(&self.nei()?.full_x_baseline()))
    }

    /// The points at which a qLogNEI estimates the lower bound of the
    /// objective for infeasible baseline points.
    fn lower_bound_points(&self) -> PyResult<ArrayOut> {
        Ok(matrix_out(&self.nei()?.lower_bound_points()))
    }

    /// The lower bound of the objective that a qLogNEI used for infeasible
    /// baseline points, if an evaluation needed it (`None` otherwise and for
    /// other acquisition functions).
    fn infeasible_lower_bound(&self) -> Option<f64> {
        self.nei.as_ref().and_then(|n| n.infeasible_lower_bound())
    }
}

/// `prune_inferior_points(model, X, num_samples, max_frac)`: the points of
/// `X` (`n x d`) with a non-zero estimated probability of being the best
/// under the MC `objective` and outcome `constraints` (callables).
#[pyfunction]
#[pyo3(
    name = "prune_inferior_points",
    signature = (
        model,
        x,
        num_samples=2048,
        max_frac=1.0,
        seed=None,
        base_samples=None,
        objective=None,
        constraints=None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn prune_inferior_points_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    x: Array,
    num_samples: usize,
    max_frac: f64,
    seed: Option<u64>,
    base_samples: Option<Array>,
    objective: Option<Bound<'_, PyAny>>,
    constraints: Option<Vec<Py<PyAny>>>,
) -> PyResult<ArrayOut> {
    let m = model.inner.clone();
    let x = x.matrix("X")?;
    let options = PruneOptions {
        num_samples,
        max_frac,
        seed,
        base_samples: base_samples.map(|z| z.matrix("base_samples")).transpose()?,
        objective: objective_of(objective.as_ref())?,
        constraints: constraints.map(constraint_callbacks),
    };
    let pruned = threads::run(py, || {
        crate::callbacks::context::with_context(&[], None, || {
            prune_inferior_points(m.as_ref(), &x, &options)
        })
    })
    .into_py_result()?;
    Ok(matrix_out(&pruned))
}
