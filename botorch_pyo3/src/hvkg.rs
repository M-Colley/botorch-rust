//! Cost models (`botorch.models.cost`), cost-aware utilities
//! (`botorch.acquisition.cost_aware`), the multi-fidelity helpers of
//! `botorch.acquisition.utils`, and the hypervolume knowledge gradient
//! (`botorch.acquisition.multi_objective.hypervolume_knowledge_gradient`)
//! with its one-shot initial conditions
//! (`botorch.optim.initializers.gen_one_shot_hvkg_initial_conditions`).
//!
//! Projections arrive as specifications: `None` (the identity),
//! `("target", target_fidelities)` (`project_to_target_fidelity`, evaluated in
//! Rust) or `("callable", project)` (a callable on torch tensors, through
//! [`crate::callbacks`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use botorch_core::acquisition::cost_aware::{CostAwareUtility, InverseCostWeightedUtility};
use botorch_core::acquisition::fidelity::{
    expand_trace_observations as expand_core, project_to_target_fidelity as project_core,
    InputProjection, TargetFidelityProjection,
};
use botorch_core::acquisition::multi_objective::{
    HypervolumeValueFunction, QHypervolumeKnowledgeGradient, QHypervolumeKnowledgeGradientOptions,
    QMultiFidelityHypervolumeKnowledgeGradient, QMultiFidelityHypervolumeKnowledgeGradientOptions,
};
use botorch_core::acquisition::one_shot::OneShotAcquisitionFunction;
use botorch_core::acquisition::AcquisitionFunction;
use botorch_core::models::cost::{
    AffineFidelityCostModel, CostModel, FixedCostModel, PosteriorMeanCostModel,
};
use botorch_core::models::BoolMatrix;
use botorch_core::optim::OptimizeAcqfOptions;
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use crate::acquisition::{model_of, sampler_of, PyAcquisitionFunction};
use crate::arrays::{stack_out, Array, ArrayBuf, BoolBuf};
use crate::callbacks::{
    generic_cost_utility, PyCallback, PyCostModel, PyProjection, PyTensorFunction,
};
use crate::errors::{dimension_error, unsupported, value_error, IntoPyResult};
use crate::mo_acquisition::mo_objective_of;
use crate::models::PyModel;
use crate::optim::{linear_constraints, LinearArg};
use crate::sampling::PySobolQMCNormalSampler;
use crate::threads;

/// A boolean matrix from `(data, shape)`.
fn bool_matrix(mask: &(Vec<bool>, Vec<usize>), name: &str) -> PyResult<BoolMatrix> {
    match mask.1.as_slice() {
        &[r, c] if r * c == mask.0.len() => Ok(BoolMatrix::from_row_slice(r, c, &mask.0)),
        s => Err(dimension_error(format!(
            "Expected {name} to be `q x m`, but got shape {s:?}."
        ))),
    }
}

fn bool_out(mask: &BoolMatrix) -> BoolBuf {
    let (r, c) = mask.shape();
    let mut data = Vec::with_capacity(r * c);
    for i in 0..r {
        data.extend(mask.row(i).iter().copied());
    }
    BoolBuf(data, vec![r, c])
}

/// A cost model (`botorch.models.cost`, or the posterior mean of a model, or
/// a callable).
#[pyclass(frozen, name = "CostModel", module = "botorch_rs._botorch_rs")]
pub struct PyCostModelHandle {
    /// The cost model.
    pub inner: Arc<dyn CostModel>,
}

#[pymethods]
impl PyCostModelHandle {
    /// `AffineFidelityCostModel(fidelity_weights, fixed_cost)`.
    #[staticmethod]
    #[pyo3(signature = (fidelity_weights=None, fixed_cost=0.01))]
    fn affine(fidelity_weights: Option<Vec<(isize, f64)>>, fixed_cost: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                AffineFidelityCostModel::new(fidelity_weights.as_deref(), fixed_cost)
                    .into_py_result()?,
            ),
        })
    }

    /// `FixedCostModel(fixed_cost)`.
    #[staticmethod]
    fn fixed(fixed_cost: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(FixedCostModel::new(Vector::from_vec(fixed_cost)).into_py_result()?),
        })
    }

    /// The posterior mean of a model.
    #[staticmethod]
    fn posterior_mean(model: PyRef<'_, PyModel>) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(PosteriorMeanCostModel::new(model_of(&model)).into_py_result()?),
        })
    }

    /// A callable of `1 x q x d` torch tensors returning `1 x q x m` costs.
    #[staticmethod]
    fn callable(f: Py<PyAny>, num_outputs: usize) -> Self {
        Self {
            inner: Arc::new(PyCostModel::new(f, num_outputs)),
        }
    }

    /// Number of outputs.
    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    /// The costs of the points of `X` (`... x q x d`): `... x q x m`.
    fn forward(&self, py: Python<'_>, x: Array) -> PyResult<ArrayBuf> {
        let (batch, sets) = x.batches("X")?;
        let inner = Arc::clone(&self.inner);
        let costs = threads::run(py, || {
            sets.iter()
                .map(|s| inner.forward(s))
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        let q = x.1[x.1.len() - 2];
        Ok(stack_out(&costs, &batch, q, self.inner.num_outputs()).into())
    }

    /// The gradient with respect to `X` of `Σ cost_bar ⊙ costs`.
    fn backward(&self, py: Python<'_>, x: Array, cost_bar: Array) -> PyResult<ArrayBuf> {
        let (batch, sets) = x.batches("X")?;
        let (_, bars) = cost_bar.batches("cost_bar")?;
        if bars.len() != sets.len() {
            return Err(dimension_error("cost_bar does not match X"));
        }
        let inner = Arc::clone(&self.inner);
        let grads = threads::run(py, || {
            sets.iter()
                .zip(&bars)
                .map(|(s, b)| inner.backward(s, b))
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        let (q, d) = (x.1[x.1.len() - 2], x.1[x.1.len() - 1]);
        Ok(stack_out(&grads, &batch, q, d).into())
    }
}

/// A cost-aware utility (`InverseCostWeightedUtility`,
/// `GenericCostAwareUtility`).
#[pyclass(frozen, name = "CostAwareUtility", module = "botorch_rs._botorch_rs")]
pub struct PyCostAwareUtility {
    /// The utility.
    pub inner: Arc<dyn CostAwareUtility>,
}

#[pymethods]
impl PyCostAwareUtility {
    /// `InverseCostWeightedUtility(cost_model, use_mean, cost_objective,
    /// log)`; `cost_objective` is a callable `objective(samples, X=None)`.
    #[staticmethod]
    #[pyo3(signature = (cost_model, use_mean=true, cost_objective=None, log=false))]
    fn inverse_cost_weighted(
        cost_model: PyRef<'_, PyCostModelHandle>,
        use_mean: bool,
        cost_objective: Option<Py<PyAny>>,
        log: bool,
    ) -> PyResult<Self> {
        let objective = cost_objective.map(|f| {
            Arc::new(PyCallback::objective(f, false))
                as Arc<dyn botorch_core::acquisition::McObjective>
        });
        Ok(Self {
            inner: Arc::new(
                InverseCostWeightedUtility::new(
                    Arc::clone(&cost_model.inner),
                    use_mean,
                    objective,
                    log,
                )
                .into_py_result()?,
            ),
        })
    }

    /// `GenericCostAwareUtility(cost)` with `cost(X, deltas)` on torch
    /// tensors.
    #[staticmethod]
    fn generic(cost: Py<PyAny>) -> Self {
        Self {
            inner: Arc::new(generic_cost_utility(cost)),
        }
    }

    /// BoTorch's `_log` attribute (`None` if the utility has none).
    #[getter]
    fn log(&self) -> Option<bool> {
        self.inner.log()
    }

    /// The utilities of the improvements `deltas` (`n_f x b`) at the
    /// candidate sets `X` (`b x q x d`): `n_f x b`; with an `adjoint` of the
    /// utilities (`n_f x b`), also the adjoints of `deltas` and `X`.
    #[pyo3(signature = (x, deltas, evaluation_mask=None, adjoint=None))]
    fn forward(
        &self,
        py: Python<'_>,
        x: Array,
        deltas: Array,
        evaluation_mask: Option<(Vec<bool>, Vec<usize>)>,
        adjoint: Option<Array>,
    ) -> PyResult<(ArrayBuf, Option<(ArrayBuf, ArrayBuf)>)> {
        let (batch, sets) = x.batches("X")?;
        let b = sets.len();
        let nd = deltas.0.len();
        if b == 0 || !nd.is_multiple_of(b) {
            return Err(dimension_error(format!(
                "deltas of shape {:?} do not match {b} candidate sets",
                deltas.1
            )));
        }
        if let Some(a) = &adjoint {
            if a.0.len() != nd {
                return Err(dimension_error("the adjoint must have the shape of deltas"));
            }
        }
        let n_f = nd / b;
        let mask = evaluation_mask
            .as_ref()
            .map(|m| bool_matrix(m, "X_evaluation_mask"))
            .transpose()?;
        let inner = Arc::clone(&self.inner);
        let with_grad = adjoint.is_some();
        let results = threads::run(py, || {
            sets.iter()
                .enumerate()
                .map(|(j, s)| {
                    let dj: Vec<f64> = (0..n_f).map(|i| deltas.0[i * b + j]).collect();
                    let out = inner.evaluate(s, &dj, mask.as_ref(), with_grad)?;
                    let grads = match (out.pullback, &adjoint) {
                        (Some(pb), Some(a)) => {
                            let aj: Vec<f64> = (0..n_f).map(|i| a.0[i * b + j]).collect();
                            Some(pb(&aj)?)
                        }
                        _ => None,
                    };
                    Ok((out.values, grads))
                })
                .collect::<botorch_core::Result<Vec<_>>>()
        })
        .into_py_result()?;
        let mut values = vec![0.0; nd];
        let mut dbar = vec![0.0; nd];
        let mut xbar = Vec::with_capacity(b);
        for (j, (v, g)) in results.into_iter().enumerate() {
            for i in 0..n_f {
                values[i * b + j] = v[i];
            }
            if let Some((dg, xg)) = g {
                for i in 0..n_f {
                    dbar[i * b + j] = dg[i];
                }
                xbar.push(xg);
            }
        }
        let shape = deltas.1.clone();
        let grads = with_grad.then(|| {
            let (q, d) = (x.1[x.1.len() - 2], x.1[x.1.len() - 1]);
            (
                ArrayBuf(dbar, shape.clone()),
                ArrayBuf::from(stack_out(&xbar, &batch, q, d)),
            )
        });
        Ok((ArrayBuf(values, shape), grads))
    }
}

/// A projection from its specification (see the module documentation).
fn projection_of(
    spec: Option<&Bound<'_, PyAny>>,
    d: usize,
) -> PyResult<Option<Arc<dyn InputProjection>>> {
    let Some(spec) = spec.filter(|s| !s.is_none()) else {
        return Ok(None);
    };
    let kind: String = spec.get_item(0)?.extract()?;
    match kind.as_str() {
        "target" => {
            let tf: Vec<(isize, f64)> = spec.get_item(1)?.extract()?;
            Ok(Some(Arc::new(
                TargetFidelityProjection::new(tf, d).into_py_result()?,
            )))
        }
        "callable" => Ok(Some(Arc::new(PyProjection(PyTensorFunction::new(
            spec.get_item(1)?.unbind(),
        ))))),
        other => Err(unsupported(format!("Unknown projection {other:?}."))),
    }
}

/// The knowledge gradient behind a Python handle.
enum Hvkg {
    Base(Arc<QHypervolumeKnowledgeGradient>),
    MultiFidelity(Arc<QMultiFidelityHypervolumeKnowledgeGradient>),
}

/// The state of a qHVKG / qMFHVKG (evaluation masks, pending points and the
/// one-shot initial conditions).
#[pyclass(frozen, name = "HvkgState", module = "botorch_rs._botorch_rs")]
pub struct PyHvkg {
    inner: Hvkg,
}

impl PyHvkg {
    fn hvkg(&self) -> &QHypervolumeKnowledgeGradient {
        match &self.inner {
            Hvkg::Base(a) => a,
            Hvkg::MultiFidelity(a) => a.as_hvkg(),
        }
    }
}

#[pymethods]
impl PyHvkg {
    /// BoTorch's `set_X_pending(X_pending, X_pending_evaluation_mask)`.
    #[pyo3(signature = (x_pending=None, mask=None))]
    fn set_x_pending(
        &self,
        x_pending: Option<Array>,
        mask: Option<(Vec<bool>, Vec<usize>)>,
    ) -> PyResult<()> {
        let x = x_pending.map(|p| p.matrix("X_pending")).transpose()?;
        let mask = mask
            .as_ref()
            .map(|m| bool_matrix(m, "X_pending_evaluation_mask"))
            .transpose()?;
        self.hvkg()
            .set_x_pending_with_mask(x.as_ref(), mask)
            .into_py_result()
    }

    /// The evaluation mask of the pending points, if any.
    fn x_pending_evaluation_mask(&self) -> Option<BoolBuf> {
        self.hvkg()
            .x_pending_evaluation_mask()
            .map(|m| bool_out(&m))
    }

    /// BoTorch's `construct_evaluation_mask` for `q` candidates.
    fn construct_evaluation_mask(&self, q: usize) -> PyResult<Option<BoolBuf>> {
        Ok(self
            .hvkg()
            .construct_evaluation_mask(q)
            .into_py_result()?
            .map(|m| bool_out(&m)))
    }

    /// `gen_one_shot_hvkg_initial_conditions`: `num_restarts x (q +
    /// n_f num_pareto) x d` initial conditions and BoTorch's warnings.
    #[pyo3(signature = (
        bounds,
        q,
        num_restarts,
        raw_samples,
        *,
        fixed_features=None,
        inequality_constraints=None,
        equality_constraints=None,
        seed=None,
        eta=None,
        alpha=None,
        batch_limit=None,
        init_batch_limit=None,
        maxiter=None,
        frac_random=None,
        num_inner_restarts=None,
        raw_inner_samples=None,
        one_shot_eta=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn gen_one_shot_initial_conditions(
        &self,
        py: Python<'_>,
        bounds: Array,
        q: usize,
        num_restarts: usize,
        raw_samples: usize,
        fixed_features: Option<BTreeMap<usize, f64>>,
        inequality_constraints: Option<Vec<LinearArg<'_>>>,
        equality_constraints: Option<Vec<LinearArg<'_>>>,
        seed: Option<u64>,
        eta: Option<f64>,
        alpha: Option<f64>,
        batch_limit: Option<usize>,
        init_batch_limit: Option<usize>,
        maxiter: Option<usize>,
        frac_random: Option<f64>,
        num_inner_restarts: Option<usize>,
        raw_inner_samples: Option<usize>,
        one_shot_eta: Option<f64>,
    ) -> PyResult<(ArrayBuf, Vec<String>)> {
        let bounds = bounds.matrix("bounds")?;
        let d = bounds.ncols();
        let mut options = OptimizeAcqfOptions {
            seed,
            batch_limit,
            init_batch_limit,
            fixed_features: fixed_features.unwrap_or_default(),
            ..Default::default()
        };
        if let Some(e) = eta {
            options.eta = e;
        }
        if let Some(a) = alpha {
            options.alpha = a;
        }
        if let Some(it) = maxiter {
            options.lbfgsb.maxiter = it;
            options.slsqp.maxiter = it;
        }
        options.constraints.inequality = linear_constraints(inequality_constraints)?;
        options.constraints.equality = linear_constraints(equality_constraints)?;
        if let Some(f) = frac_random {
            options.one_shot.frac_random = f;
        }
        if let Some(n) = num_inner_restarts {
            options.one_shot.num_inner_restarts = n;
        }
        if let Some(n) = raw_inner_samples {
            options.one_shot.raw_inner_samples = n;
        }
        options.one_shot.eta = one_shot_eta;
        let acqf = self.hvkg();
        let q_aug = acqf.get_augmented_q_batch_size(q);
        let (ics, warnings) = threads::run(py, || {
            let seed = options
                .seed
                .unwrap_or_else(botorch_core::utils::sampling::random_seed);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x6f6e_655f_7368_6f74);
            acqf.gen_one_shot_initial_conditions(
                &bounds,
                q,
                num_restarts,
                raw_samples,
                &options,
                &mut rng,
            )
        })
        .into_py_result()?;
        Ok((stack_out(&ics, &[ics.len()], q_aug, d).into(), warnings))
    }
}

/// `qHypervolumeKnowledgeGradient` (or, with `target_fidelities`,
/// `qMultiFidelityHypervolumeKnowledgeGradient`); returns the acquisition
/// function and its state.
#[pyfunction]
#[pyo3(signature = (
    model,
    ref_point,
    num_fantasies=8,
    num_pareto=10,
    samplers=None,
    objective=None,
    inner_sampler=None,
    x_evaluation_mask=None,
    x_pending=None,
    x_pending_evaluation_mask=None,
    current_value=None,
    use_posterior_mean=true,
    cost_aware_utility=None,
    log=false,
    target_fidelities=None,
    project=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn q_hypervolume_knowledge_gradient(
    model: PyRef<'_, PyModel>,
    ref_point: Vec<f64>,
    num_fantasies: usize,
    num_pareto: usize,
    samplers: Option<Vec<PyRef<'_, PySobolQMCNormalSampler>>>,
    objective: Option<Bound<'_, PyAny>>,
    inner_sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    x_evaluation_mask: Option<(Vec<bool>, Vec<usize>)>,
    x_pending: Option<Array>,
    x_pending_evaluation_mask: Option<(Vec<bool>, Vec<usize>)>,
    current_value: Option<f64>,
    use_posterior_mean: bool,
    cost_aware_utility: Option<PyRef<'_, PyCostAwareUtility>>,
    log: bool,
    target_fidelities: Option<Vec<(isize, f64)>>,
    project: Option<Bound<'_, PyAny>>,
) -> PyResult<(PyAcquisitionFunction, PyHvkg)> {
    let m = model_of(&model);
    let d = m.dim();
    let options = QHypervolumeKnowledgeGradientOptions {
        num_fantasies,
        num_pareto,
        samplers: samplers.map(|s| s.iter().map(|v| Arc::clone(&v.inner)).collect()),
        objective: mo_objective_of(objective.as_ref())?,
        inner_sampler: sampler_of(inner_sampler),
        x_evaluation_mask: x_evaluation_mask
            .as_ref()
            .map(|v| bool_matrix(v, "X_evaluation_mask"))
            .transpose()?,
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        x_pending_evaluation_mask: x_pending_evaluation_mask
            .as_ref()
            .map(|v| bool_matrix(v, "X_pending_evaluation_mask"))
            .transpose()?,
        current_value,
        use_posterior_mean,
        cost_aware_utility: cost_aware_utility.map(|u| Arc::clone(&u.inner)),
        log,
    };
    let project = projection_of(project.as_ref(), d)?;
    match target_fidelities {
        None => {
            if project.is_some() {
                return Err(value_error("`project` requires target fidelities"));
            }
            let a = Arc::new(
                QHypervolumeKnowledgeGradient::new(m, &ref_point, options).into_py_result()?,
            );
            Ok((
                PyAcquisitionFunction::new_plain(
                    a.clone() as Arc<dyn AcquisitionFunction>,
                    "qHypervolumeKnowledgeGradient",
                ),
                PyHvkg {
                    inner: Hvkg::Base(a),
                },
            ))
        }
        Some(tf) => {
            let a = Arc::new(
                QMultiFidelityHypervolumeKnowledgeGradient::new(
                    m,
                    &ref_point,
                    tf,
                    QMultiFidelityHypervolumeKnowledgeGradientOptions {
                        hvkg: options,
                        project,
                    },
                )
                .into_py_result()?,
            );
            Ok((
                PyAcquisitionFunction::new_plain(
                    a.clone() as Arc<dyn AcquisitionFunction>,
                    "qMultiFidelityHypervolumeKnowledgeGradient",
                ),
                PyHvkg {
                    inner: Hvkg::MultiFidelity(a),
                },
            ))
        }
    }
}

/// BoTorch's `_get_hv_value_function(model, ref_point, objective, sampler,
/// project, use_posterior_mean, log)`.
#[pyfunction]
#[pyo3(signature = (model, ref_point, objective=None, sampler=None, use_posterior_mean=false, log=false, project=None))]
pub fn hv_value_function(
    model: PyRef<'_, PyModel>,
    ref_point: Vec<f64>,
    objective: Option<Bound<'_, PyAny>>,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    use_posterior_mean: bool,
    log: bool,
    project: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAcquisitionFunction> {
    let m = model_of(&model);
    let d = m.dim();
    let f = HypervolumeValueFunction::new(
        m,
        &ref_point,
        mo_objective_of(objective.as_ref())?,
        sampler_of(sampler),
        use_posterior_mean,
        log,
        projection_of(project.as_ref(), d)?,
    )
    .into_py_result()?;
    let name = if log {
        "qLogExpectedHypervolumeImprovement"
    } else {
        "qExpectedHypervolumeImprovement"
    };
    Ok(PyAcquisitionFunction::new_plain(Arc::new(f), name))
}

/// BoTorch's `project_to_target_fidelity(X, target_fidelities, d)` for `X`
/// of shape `... x q x (d or d - d_f)`.
#[pyfunction]
pub fn project_to_target_fidelity(
    x: Array,
    target_fidelities: Vec<(isize, f64)>,
    d: usize,
) -> PyResult<ArrayBuf> {
    let (batch, sets) = x.batches("X")?;
    let out = sets
        .iter()
        .map(|s| project_core(s, &target_fidelities, d))
        .collect::<botorch_core::Result<Vec<Matrix>>>()
        .into_py_result()?;
    let q = x.1[x.1.len() - 2];
    Ok(stack_out(&out, &batch, q, d).into())
}

/// BoTorch's `expand_trace_observations(X, fidelity_dims, num_trace_obs)`
/// for `X` of shape `... x q x d`.
#[pyfunction]
#[pyo3(signature = (x, fidelity_dims=None, num_trace_obs=0))]
pub fn expand_trace_observations(
    x: Array,
    fidelity_dims: Option<Vec<isize>>,
    num_trace_obs: usize,
) -> PyResult<ArrayBuf> {
    let (batch, sets) = x.batches("X")?;
    let out = sets
        .iter()
        .map(|s| expand_core(s, fidelity_dims.as_deref(), num_trace_obs))
        .collect::<botorch_core::Result<Vec<Matrix>>>()
        .into_py_result()?;
    let (q, d) = (x.1[x.1.len() - 2], x.1[x.1.len() - 1]);
    Ok(stack_out(&out, &batch, q * (1 + num_trace_obs), d).into())
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCostModelHandle>()?;
    m.add_class::<PyCostAwareUtility>()?;
    m.add_class::<PyHvkg>()?;
    m.add_function(wrap_pyfunction!(q_hypervolume_knowledge_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(hv_value_function, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_target_fidelity, m)?)?;
    m.add_function(wrap_pyfunction!(expand_trace_observations, m)?)?;
    Ok(())
}
