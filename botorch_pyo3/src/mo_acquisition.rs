//! Multi-objective acquisition functions (`botorch.acquisition.multi_objective`):
//! qEHVI / qLogEHVI, qNEHVI / qLogNEHVI, the analytic EHVI, qLogNParEGO and
//! `prune_inferior_points_multi_objective`.
//!
//! The acquisition functions are returned as [`PyAcquisitionFunction`]s, so
//! that evaluation, gradients and `optimize_acqf` work as for the other
//! acquisition functions; the noisy ones also return a [`PyNehvi`] handle on
//! their baseline state. Box decompositions arrive as their hypercell bounds
//! (`2 x num_cells x m`), which the Python layer reads from botorch_rs or
//! BoTorch partitionings.
//!
//! Multi-output objectives arrive as specifications: `("identity",
//! outcomes)`, `("weighted", weights, outcomes)` (evaluated in Rust),
//! `("callable", objective)` (BoTorch's `objective(samples, X=X)` on torch
//! tensors, through [`crate::callbacks`]) or `("native", obj)` for a
//! [`PyMultiOutputObjective`] (e.g. a feasibility-weighted objective).

use std::sync::Arc;

use botorch_core::acquisition::multi_objective::{
    prune_inferior_points_multi_objective, ExpectedHypervolumeImprovement,
    FeasibilityWeightedMcMultiOutputObjective, HypercellBounds, IdentityMcMultiOutputObjective,
    McMultiOutputObjective, PruneMultiObjectiveOptions, QExpectedHypervolumeImprovement,
    QExpectedHypervolumeImprovementOptions, QLogExpectedHypervolumeImprovement,
    QLogExpectedHypervolumeImprovementOptions, QLogNParEGO, QLogNParEGOOptions,
    QLogNoisyExpectedHypervolumeImprovement, QLogNoisyExpectedHypervolumeImprovementOptions,
    QNoisyExpectedHypervolumeImprovement, QNoisyExpectedHypervolumeImprovementOptions,
    WeightedMcMultiOutputObjective,
};
use botorch_core::acquisition::{AcquisitionFunction, PruneOptions};
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;

use crate::acquisition::{
    constraint_callbacks, constraints_of, model_of, sampler_of, PyAcquisitionFunction,
};
use crate::arrays::{matrix_out, Array, ArrayBuf};
use crate::callbacks::PyCallback;
use crate::errors::{dimension_error, unsupported, IntoPyResult};
use crate::models::PyModel;
use crate::sampling::PySobolQMCNormalSampler;
use crate::threads;

/// A multi-output MC objective evaluated in Rust (BoTorch's
/// `FeasibilityWeightedMCMultiOutputObjective`, or any objective built from
/// a specification).
#[pyclass(
    frozen,
    name = "MultiOutputObjective",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyMultiOutputObjective {
    inner: Arc<dyn McMultiOutputObjective>,
    infeasible_cost: Vec<f64>,
    objective_idcs: Vec<usize>,
    constraint_idcs: Vec<usize>,
}

impl PyMultiOutputObjective {
    /// A multi-output objective evaluated in Rust (e.g. a multi-output risk
    /// measure of [`crate::risk_measures`]).
    pub(crate) fn from_objective(inner: Arc<dyn McMultiOutputObjective>) -> Self {
        Self {
            inner,
            infeasible_cost: Vec::new(),
            objective_idcs: Vec::new(),
            constraint_idcs: Vec::new(),
        }
    }
}

#[pymethods]
impl PyMultiOutputObjective {
    /// `FeasibilityWeightedMCMultiOutputObjective(model, X_baseline,
    /// constraint_idcs, objective)` on the current snapshot of `model`.
    #[staticmethod]
    #[pyo3(signature = (model, x_baseline, constraint_idcs, objective=None))]
    fn feasibility_weighted(
        py: Python<'_>,
        model: PyRef<'_, PyModel>,
        x_baseline: Array,
        constraint_idcs: Vec<i64>,
        objective: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let m = model_of(&model);
        let x = x_baseline.matrix("X_baseline")?;
        let objective = mo_objective_of(objective.as_ref())?;
        let obj = threads::run(py, || {
            FeasibilityWeightedMcMultiOutputObjective::new(
                m.as_ref(),
                &x,
                &constraint_idcs,
                objective,
            )
        })
        .into_py_result()?;
        Ok(Self {
            infeasible_cost: obj.infeasible_cost().to_vec(),
            objective_idcs: obj.objective_idcs().to_vec(),
            constraint_idcs: obj.constraint_idcs().to_vec(),
            inner: Arc::new(obj),
        })
    }

    /// The infeasible cost of every objective output (empty without
    /// constraints).
    #[getter]
    fn infeasible_cost(&self) -> Vec<f64> {
        self.infeasible_cost.clone()
    }

    /// The indices of the objective outputs.
    #[getter]
    fn objective_idcs(&self) -> Vec<usize> {
        self.objective_idcs.clone()
    }

    /// The (non-negative) constraint indices.
    #[getter]
    fn constraint_idcs(&self) -> Vec<usize> {
        self.constraint_idcs.clone()
    }
}

/// A multi-output objective from its specification (see the module
/// documentation).
pub(crate) fn mo_objective_of(
    spec: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<dyn McMultiOutputObjective>>> {
    let Some(spec) = spec.filter(|s| !s.is_none()) else {
        return Ok(None);
    };
    let kind: String = spec.get_item(0)?.extract()?;
    let objective: Arc<dyn McMultiOutputObjective> = match kind.as_str() {
        "identity" => {
            let outcomes: Option<Vec<i64>> = spec.get_item(1)?.extract()?;
            Arc::new(
                IdentityMcMultiOutputObjective::new(outcomes.as_deref(), None).into_py_result()?,
            )
        }
        "weighted" => {
            let weights: Vec<f64> = spec.get_item(1)?.extract()?;
            let outcomes: Option<Vec<i64>> = spec.get_item(2)?.extract()?;
            Arc::new(
                WeightedMcMultiOutputObjective::new(
                    Vector::from_vec(weights),
                    outcomes.as_deref(),
                    None,
                )
                .into_py_result()?,
            )
        }
        "callable" => Arc::new(PyCallback::mo_objective(spec.get_item(1)?.unbind())),
        "native" => {
            let obj: PyRef<'_, PyMultiOutputObjective> = spec.get_item(1)?.extract()?;
            Arc::clone(&obj.inner)
        }
        other => {
            return Err(unsupported(format!(
                "Unknown multi-output objective {other:?}."
            )))
        }
    };
    Ok(Some(objective))
}

/// Hypercell bounds of shape `2 x num_cells x m`.
fn cells_of(cells: &Array) -> PyResult<HypercellBounds> {
    let (nc, m) = match cells.1.as_slice() {
        [2, nc, m] => (*nc, *m),
        s => {
            return Err(dimension_error(format!(
                "Expected hypercell bounds of shape `2 x num_cells x m`, got {s:?}."
            )))
        }
    };
    let half = nc * m;
    HypercellBounds::new(1, nc, m, cells.0[..half].to_vec(), cells.0[half..].to_vec())
        .into_py_result()
}

/// `qExpectedHypervolumeImprovement` (`log=False`) or
/// `qLogExpectedHypervolumeImprovement` (`log=True`) with the hypercell
/// bounds of a partitioning (`2 x num_cells x m`).
#[pyfunction]
#[pyo3(signature = (
    log,
    model,
    ref_point,
    cells,
    sampler=None,
    x_pending=None,
    objective=None,
    constraints=None,
    eta=None,
    fat=false,
    tau_relu=1e-6,
    tau_max=1e-2,
))]
#[allow(clippy::too_many_arguments)]
pub fn q_expected_hypervolume_improvement(
    log: bool,
    model: PyRef<'_, PyModel>,
    ref_point: Vec<f64>,
    cells: Array,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    x_pending: Option<Array>,
    objective: Option<Bound<'_, PyAny>>,
    constraints: Option<Vec<Py<PyAny>>>,
    eta: Option<Vec<f64>>,
    fat: bool,
    tau_relu: f64,
    tau_max: f64,
) -> PyResult<PyAcquisitionFunction> {
    let m = model_of(&model);
    let cells = cells_of(&cells)?;
    let sampler = sampler_of(sampler);
    let x_pending = x_pending.map(|p| p.matrix("X_pending")).transpose()?;
    let objective = mo_objective_of(objective.as_ref())?;
    let constraints = constraints_of(constraints, eta)?;
    if log {
        let a = QLogExpectedHypervolumeImprovement::from_cell_bounds(
            m,
            &ref_point,
            cells,
            QLogExpectedHypervolumeImprovementOptions {
                sampler,
                objective,
                constraints,
                x_pending,
                fat,
                tau_relu,
                tau_max,
            },
        )
        .into_py_result()?;
        Ok(PyAcquisitionFunction::new_plain(
            Arc::new(a),
            "qLogExpectedHypervolumeImprovement",
        ))
    } else {
        let a = QExpectedHypervolumeImprovement::from_cell_bounds(
            m,
            &ref_point,
            cells,
            QExpectedHypervolumeImprovementOptions {
                sampler,
                objective,
                constraints,
                x_pending,
                fat,
            },
        )
        .into_py_result()?;
        Ok(PyAcquisitionFunction::new_plain(
            Arc::new(a),
            "qExpectedHypervolumeImprovement",
        ))
    }
}

/// The baseline state of a qNEHVI or qLogNEHVI.
enum Nehvi {
    Linear(Arc<QNoisyExpectedHypervolumeImprovement>),
    Log(Arc<QLogNoisyExpectedHypervolumeImprovement>),
}

macro_rules! nehvi_call {
    ($self:expr, $a:ident => $e:expr) => {
        match &$self.inner {
            Nehvi::Linear($a) => $e,
            Nehvi::Log($a) => $e,
        }
    };
}

/// The baseline state of a qNEHVI / qLogNEHVI (BoTorch's
/// `NoisyExpectedHypervolumeMixin` attributes).
#[pyclass(frozen, name = "NehviState", module = "botorch_rs._botorch_rs")]
pub struct PyNehvi {
    inner: Nehvi,
}

#[pymethods]
impl PyNehvi {
    /// The observed (pruned) baseline (`_X_baseline`).
    fn x_baseline(&self) -> ArrayBuf {
        matrix_out(&nehvi_call!(self, a => a.x_baseline())).into()
    }

    /// The baseline and the cached pending points (`X_baseline`).
    fn x_baseline_and_pending(&self) -> ArrayBuf {
        matrix_out(&nehvi_call!(self, a => a.x_baseline_and_pending())).into()
    }

    /// The hypercell bounds under every MC sample (`2 x S x num_cells x m`).
    fn cell_bounds(&self) -> ArrayBuf {
        let c = nehvi_call!(self, a => a.cell_bounds());
        let mut data = c.lower().to_vec();
        data.extend_from_slice(c.upper());
        ArrayBuf(data, vec![2, c.batches(), c.num_cells(), c.num_outcomes()])
    }

    /// `_prev_nehvi`.
    fn prev_nehvi(&self) -> f64 {
        nehvi_call!(self, a => a.prev_nehvi())
    }

    /// `_hypervolumes` (one per MC sample).
    fn hypervolumes(&self) -> ArrayBuf {
        let hvs = nehvi_call!(self, a => a.hypervolumes());
        let n = hvs.len();
        ArrayBuf(hvs, vec![n])
    }

    /// `_baseline_hvs` (non-incremental NEHVI), else `None`.
    fn baseline_hvs(&self) -> Option<ArrayBuf> {
        nehvi_call!(self, a => a.baseline_hvs()).map(|hvs| {
            let n = hvs.len();
            ArrayBuf(hvs, vec![n])
        })
    }

    /// The base samples of the cached points (`S x N m`), if any.
    fn base_samples(&self) -> Option<ArrayBuf> {
        nehvi_call!(self, a => a.base_samples())
            .as_ref()
            .map(|z| matrix_out(z).into())
    }
}

/// `qNoisyExpectedHypervolumeImprovement` (`log=False`) or
/// `qLogNoisyExpectedHypervolumeImprovement` (`log=True`); returns the
/// acquisition function and its baseline state.
#[pyfunction]
#[pyo3(signature = (
    log,
    model,
    ref_point,
    x_baseline,
    sampler=None,
    x_pending=None,
    objective=None,
    constraints=None,
    eta=None,
    fat=false,
    prune_baseline=false,
    prune_num_samples=2048,
    prune_seed=None,
    alpha=0.0,
    cache_pending=true,
    max_iep=0,
    incremental_nehvi=true,
    cache_root=true,
    tau_relu=1e-6,
    tau_max=1e-3,
))]
#[allow(clippy::too_many_arguments)]
pub fn q_noisy_expected_hypervolume_improvement(
    py: Python<'_>,
    log: bool,
    model: PyRef<'_, PyModel>,
    ref_point: Vec<f64>,
    x_baseline: Array,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    x_pending: Option<Array>,
    objective: Option<Bound<'_, PyAny>>,
    constraints: Option<Vec<Py<PyAny>>>,
    eta: Option<Vec<f64>>,
    fat: bool,
    prune_baseline: bool,
    prune_num_samples: usize,
    prune_seed: Option<u64>,
    alpha: f64,
    cache_pending: bool,
    max_iep: usize,
    incremental_nehvi: bool,
    cache_root: bool,
    tau_relu: f64,
    tau_max: f64,
) -> PyResult<(PyAcquisitionFunction, PyNehvi)> {
    let m = model_of(&model);
    let x_baseline = x_baseline.matrix("X_baseline")?;
    let sampler = sampler_of(sampler);
    let x_pending = x_pending.map(|p| p.matrix("X_pending")).transpose()?;
    let objective = mo_objective_of(objective.as_ref())?;
    let constraints = constraints_of(constraints, eta)?;
    let prune_options = PruneMultiObjectiveOptions {
        num_samples: prune_num_samples,
        seed: prune_seed,
        ..Default::default()
    };
    if log {
        let options = QLogNoisyExpectedHypervolumeImprovementOptions {
            sampler,
            objective,
            constraints,
            x_pending,
            prune_baseline,
            prune_options,
            alpha,
            cache_pending,
            max_iep,
            incremental_nehvi,
            cache_root,
            tau_relu,
            tau_max,
            fat,
        };
        let a = threads::run(py, || {
            QLogNoisyExpectedHypervolumeImprovement::new(m, &ref_point, &x_baseline, options)
        })
        .into_py_result()?;
        let a = Arc::new(a);
        Ok((
            PyAcquisitionFunction::new_plain(
                a.clone() as Arc<dyn AcquisitionFunction>,
                "qLogNoisyExpectedHypervolumeImprovement",
            ),
            PyNehvi {
                inner: Nehvi::Log(a),
            },
        ))
    } else {
        let options = QNoisyExpectedHypervolumeImprovementOptions {
            sampler,
            objective,
            constraints,
            x_pending,
            fat,
            prune_baseline,
            prune_options,
            alpha,
            cache_pending,
            max_iep,
            incremental_nehvi,
            cache_root,
        };
        let a = threads::run(py, || {
            QNoisyExpectedHypervolumeImprovement::new(m, &ref_point, &x_baseline, options)
        })
        .into_py_result()?;
        let a = Arc::new(a);
        Ok((
            PyAcquisitionFunction::new_plain(
                a.clone() as Arc<dyn AcquisitionFunction>,
                "qNoisyExpectedHypervolumeImprovement",
            ),
            PyNehvi {
                inner: Nehvi::Linear(a),
            },
        ))
    }
}

/// `ExpectedHypervolumeImprovement(model, ref_point, partitioning)` with the
/// hypercell bounds (`2 x num_cells x m`) and Pareto front (`n x m`) of the
/// partitioning.
#[pyfunction]
pub fn expected_hypervolume_improvement(
    model: PyRef<'_, PyModel>,
    ref_point: Vec<f64>,
    cells: Array,
    pareto_y: Array,
) -> PyResult<PyAcquisitionFunction> {
    let m = model_of(&model);
    let cells = cells_of(&cells)?;
    let pareto = if pareto_y.0.is_empty() {
        Matrix::zeros(0, cells.num_outcomes())
    } else {
        pareto_y.matrix("pareto_Y")?
    };
    let a = ExpectedHypervolumeImprovement::from_cell_bounds(m, &ref_point, cells, &pareto)
        .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "ExpectedHypervolumeImprovement",
    ))
}

/// `qLogNParEGO(model, X_baseline, scalarization_weights, ...)`; returns the
/// acquisition function, the scalarization weights and `Y_baseline`.
#[pyfunction]
#[pyo3(signature = (
    model,
    x_baseline,
    scalarization_weights=None,
    weights_seed=None,
    sampler=None,
    objective=None,
    constraints=None,
    eta=None,
    x_pending=None,
    fat=true,
    prune_baseline=false,
    prune_num_samples=2048,
    prune_seed=None,
    tau_relu=1e-6,
    tau_max=1e-2,
    incremental=true,
))]
#[allow(clippy::too_many_arguments)]
pub fn q_log_nparego(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    x_baseline: Array,
    scalarization_weights: Option<Vec<f64>>,
    weights_seed: Option<u64>,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    objective: Option<Bound<'_, PyAny>>,
    constraints: Option<Vec<Py<PyAny>>>,
    eta: Option<Vec<f64>>,
    x_pending: Option<Array>,
    fat: bool,
    prune_baseline: bool,
    prune_num_samples: usize,
    prune_seed: Option<u64>,
    tau_relu: f64,
    tau_max: f64,
    incremental: bool,
) -> PyResult<(PyAcquisitionFunction, Vec<f64>, ArrayBuf)> {
    let m = model_of(&model);
    let x_baseline = x_baseline.matrix("X_baseline")?;
    let options = QLogNParEGOOptions {
        scalarization_weights,
        weights_seed,
        sampler: sampler_of(sampler),
        objective: mo_objective_of(objective.as_ref())?,
        constraints: constraints_of(constraints, eta)?,
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        fat,
        prune_baseline,
        prune_options: PruneOptions {
            num_samples: prune_num_samples,
            seed: prune_seed,
            ..Default::default()
        },
        tau_relu,
        tau_max,
        incremental,
    };
    let a = threads::run(py, || QLogNParEGO::new(m, &x_baseline, options)).into_py_result()?;
    let weights = a.scalarization_weights().to_vec();
    let y_baseline = ArrayBuf::from(matrix_out(a.y_baseline()));
    let nei = Arc::clone(a.log_nei());
    Ok((
        PyAcquisitionFunction::with_nei(Arc::new(a), nei, "qLogNParEGO"),
        weights,
        y_baseline,
    ))
}

/// `prune_inferior_points_multi_objective(model, X, ref_point, objective,
/// constraints, num_samples, max_frac)`: the points of `X` (`n x d`) with a
/// non-zero estimated probability of being Pareto optimal.
#[pyfunction]
#[pyo3(
    name = "prune_inferior_points_multi_objective",
    signature = (
        model,
        x,
        ref_point,
        objective=None,
        constraints=None,
        num_samples=2048,
        max_frac=1.0,
        seed=None,
        base_samples=None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn prune_inferior_points_multi_objective_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    x: Array,
    ref_point: Vec<f64>,
    objective: Option<Bound<'_, PyAny>>,
    constraints: Option<Vec<Py<PyAny>>>,
    num_samples: usize,
    max_frac: f64,
    seed: Option<u64>,
    base_samples: Option<Array>,
) -> PyResult<ArrayBuf> {
    let m = model_of(&model);
    let x = x.matrix("X")?;
    let options = PruneMultiObjectiveOptions {
        num_samples,
        max_frac,
        seed,
        base_samples: base_samples.map(|z| z.matrix("base_samples")).transpose()?,
        objective: mo_objective_of(objective.as_ref())?,
        constraints: constraints.map(constraint_callbacks),
    };
    let pruned = threads::run(py, || {
        prune_inferior_points_multi_objective(m.as_ref(), &x, &ref_point, &options)
    })
    .into_py_result()?;
    Ok(matrix_out(&pruned).into())
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiOutputObjective>()?;
    m.add_class::<PyNehvi>()?;
    m.add_function(wrap_pyfunction!(q_expected_hypervolume_improvement, m)?)?;
    m.add_function(wrap_pyfunction!(
        q_noisy_expected_hypervolume_improvement,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(expected_hypervolume_improvement, m)?)?;
    m.add_function(wrap_pyfunction!(q_log_nparego, m)?)?;
    m.add_function(wrap_pyfunction!(
        prune_inferior_points_multi_objective_py,
        m
    )?)?;
    Ok(())
}
