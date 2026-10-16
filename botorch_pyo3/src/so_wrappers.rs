//! Wrappers of acquisition functions of phase 8
//! (`botorch.acquisition.{penalized, proximal, fixed_feature,
//! prior_guided}`): penalties of candidate sets and of points, the
//! penalized MC objective, and the proximal, fixed-feature and prior-guided
//! acquisition functions.
//!
//! Penalties arrive as specifications: `("L2", init_point)`, `("L1",
//! init_point)`, `("Gaussian", init_point, sigma)`, `("GroupLasso",
//! init_point, groups)`, `("L0", target_point, a)` (BoTorch's classes,
//! evaluated in Rust) or `("callable", f)` (a module of torch tensors with
//! gradients from `torch.autograd`). Penalties of points (of
//! `PenalizedMCObjective`) are `("L1", init_point)`, `("L0", target_point,
//! a)` or `("callable", f)`.

use std::sync::Arc;

use botorch_core::acquisition::penalized::{
    FnPenalty, FnPointPenalty, GaussianPenalty, GroupLassoPenalty, L0PenaltyApprox,
    L0PenaltyApproxObjective, L1Penalty, L1PenaltyObjective, L2Penalty,
    PenalizedAcquisitionFunction, PenalizedMcObjective, Penalty, PointPenalty, PointPullback,
};
use botorch_core::acquisition::wrappers::{
    FixedFeatureAcquisitionFunction, PriorGuidedAcquisitionFunction, PriorModule,
    ProximalAcquisitionFunction,
};
use botorch_core::acquisition::McObjective;
use botorch_core::{BotorchError, Matrix, Result, Vector};
use pyo3::prelude::*;

use crate::acquisition::{model_of, objective_of, PyAcquisitionFunction};
use crate::arrays::Array;
use crate::callbacks::{rows, PyTensorFunction};
use crate::errors::{unsupported, value_error, IntoPyResult};
use crate::models::PyModel;

/// A reference point from a list of floats.
fn point_of(spec: &Bound<'_, PyAny>, i: usize) -> PyResult<Vector> {
    let p: Vec<f64> = spec.get_item(i)?.extract()?;
    Ok(Vector::from_vec(p))
}

/// The values of a callable of `1 x q x d` inputs with `q` (or one) values.
fn call_points(
    f: &PyTensorFunction,
    x: &Matrix,
    with_grad: bool,
    expected: usize,
    what: &str,
) -> Result<(Vec<f64>, Option<Py<PyAny>>)> {
    let (q, d) = x.shape();
    let (values, shape, handle) = f.call(&[(rows(x), vec![1, q, d])], with_grad)?;
    if values.len() != expected {
        return Err(BotorchError::DimensionMismatch(format!(
            "the {what} returned a tensor of shape {shape:?} for a candidate set of {q} \
             points, expected {expected} values"
        )));
    }
    Ok((values, handle))
}

/// A penalty of candidate sets from its specification.
fn penalty_of(spec: &Bound<'_, PyAny>) -> PyResult<Arc<dyn Penalty>> {
    let kind: String = spec.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "L2" => Arc::new(L2Penalty::new(point_of(spec, 1)?)),
        "L1" => Arc::new(L1Penalty::new(point_of(spec, 1)?)),
        "Gaussian" => Arc::new(GaussianPenalty::new(
            point_of(spec, 1)?,
            spec.get_item(2)?.extract()?,
        )),
        "GroupLasso" => {
            let p = point_of(spec, 1)?;
            let d = p.len() as isize;
            let groups: Vec<Vec<isize>> = spec.get_item(2)?.extract()?;
            let groups = groups
                .into_iter()
                .map(|g| {
                    g.into_iter()
                        .map(|k| {
                            let k = if k < 0 { k + d } else { k };
                            usize::try_from(k)
                                .ok()
                                .filter(|&k| k < p.len())
                                .ok_or_else(|| {
                                    value_error(format!("group index out of range for d = {d}"))
                                })
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
                .collect::<PyResult<Vec<_>>>()?;
            Arc::new(GroupLassoPenalty::new(p, groups))
        }
        "L0" => Arc::new(L0PenaltyApprox::new(
            point_of(spec, 1)?,
            spec.get_item(2)?.extract()?,
        )),
        "callable" => {
            let f = PyTensorFunction::new(spec.get_item(1)?.unbind());
            Arc::new(FnPenalty::new(move |x: &Matrix, with_grad: bool| {
                let (values, handle) = call_points(&f, x, with_grad, 1, "penalty")?;
                let grad = match handle {
                    Some(h) => {
                        let g = PyTensorFunction::vjp(&h, &[1.0])?;
                        Some(Matrix::from_row_slice(x.nrows(), x.ncols(), &g[0]))
                    }
                    None => None,
                };
                Ok((values[0], grad))
            }))
        }
        other => return Err(unsupported(format!("Unknown penalty {other:?}."))),
    })
}

/// The pullback of per-point values computed by a callable.
fn callable_pullback(h: Py<PyAny>, q: usize, d: usize) -> PointPullback {
    Box::new(move |bar: &Vector| {
        let g = PyTensorFunction::vjp(&h, bar.as_slice())?;
        Ok(Matrix::from_row_slice(q, d, &g[0]))
    })
}

/// A penalty of points from its specification.
fn point_penalty_of(spec: &Bound<'_, PyAny>) -> PyResult<Arc<dyn PointPenalty>> {
    let kind: String = spec.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "L1" => Arc::new(L1PenaltyObjective::new(point_of(spec, 1)?)),
        "L0" => Arc::new(L0PenaltyApproxObjective::new(
            point_of(spec, 1)?,
            spec.get_item(2)?.extract()?,
        )),
        "callable" => {
            let f = PyTensorFunction::new(spec.get_item(1)?.unbind());
            Arc::new(FnPointPenalty::new(move |x: &Matrix, with_grad: bool| {
                let (q, d) = x.shape();
                let (values, handle) = call_points(&f, x, with_grad, q, "penalty objective")?;
                Ok((
                    Vector::from_vec(values),
                    handle.map(|h| callable_pullback(h, q, d)),
                ))
            }))
        }
        other => return Err(unsupported(format!("Unknown penalty objective {other:?}."))),
    })
}

/// `PenalizedMCObjective` from `("penalized", objective, penalty,
/// regularization_parameter)` (the MC objective specification of
/// `objective_of`).
pub(crate) fn penalized_objective(spec: &Bound<'_, PyAny>) -> PyResult<Arc<dyn McObjective>> {
    let inner = objective_of(Some(&spec.get_item(1)?))?
        .ok_or_else(|| value_error("PenalizedMCObjective requires an objective."))?;
    let penalty = point_penalty_of(&spec.get_item(2)?)?;
    let lam: f64 = spec.get_item(3)?.extract()?;
    Ok(Arc::new(PenalizedMcObjective::new(inner, penalty, lam)))
}

/// `PenalizedAcquisitionFunction(raw_acqf, penalty_func,
/// regularization_parameter)`.
#[pyfunction]
pub fn penalized_acquisition_function(
    raw_acqf: PyRef<'_, PyAcquisitionFunction>,
    penalty: Bound<'_, PyAny>,
    regularization_parameter: f64,
) -> PyResult<PyAcquisitionFunction> {
    let p = penalty_of(&penalty)?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(PenalizedAcquisitionFunction::new(
            Arc::clone(&raw_acqf.inner),
            p,
            regularization_parameter,
        )),
        "PenalizedAcquisitionFunction",
    ))
}

/// `ProximalAcquisitionFunction(acq_function, proximal_weights,
/// transformed_weighting, beta)` around the last training point of `model`
/// (the first output GP's raw inputs and input transform).
#[pyfunction]
#[pyo3(signature = (acq_function, model, proximal_weights, transformed_weighting=true, beta=None))]
pub fn proximal_acquisition_function(
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    model: PyRef<'_, PyModel>,
    proximal_weights: Vec<f64>,
    transformed_weighting: bool,
    beta: Option<f64>,
) -> PyResult<PyAcquisitionFunction> {
    let m = model_of(&model);
    let gps = m
        .output_gps()
        .ok_or_else(|| unsupported("Acquisition function model must have `train_inputs`."))?;
    let gp = gps[0];
    if gp.input_chain().is_some() {
        return Err(unsupported(
            "ProximalAcquisitionFunction supports Normalize input transforms only.",
        ));
    }
    let x = gp.train_inputs();
    let last = Vector::from_fn(x.ncols(), |k, _| x[(x.nrows() - 1, k)]);
    let a = ProximalAcquisitionFunction::new(
        Arc::clone(&acq_function.inner),
        Vector::from_vec(proximal_weights),
        last,
        gp.input_transform().cloned(),
        transformed_weighting,
        beta,
    )
    .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "ProximalAcquisitionFunction",
    ))
}

/// `FixedFeatureAcquisitionFunction(acq_function, d, columns, values)` with
/// the values as a `1 x d_f` or `q x d_f` array (in the order of the
/// sorted columns).
#[pyfunction]
pub fn fixed_feature_acquisition_function(
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    d: usize,
    columns: Vec<usize>,
    values: Array,
) -> PyResult<PyAcquisitionFunction> {
    let v = values.matrix("values")?;
    let a = FixedFeatureAcquisitionFunction::new(Arc::clone(&acq_function.inner), d, columns, v)
        .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "FixedFeatureAcquisitionFunction",
    ))
}

/// `PriorGuidedAcquisitionFunction(acq_function, prior_module, log,
/// prior_exponent, X_pending)` with the prior a callable of `1 x n x d`
/// tensors returning the `n` (log) densities.
#[pyfunction]
#[pyo3(signature = (acq_function, prior_module, log=false, prior_exponent=1.0, x_pending=None))]
pub fn prior_guided_acquisition_function(
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    prior_module: Py<PyAny>,
    log: bool,
    prior_exponent: f64,
    x_pending: Option<Array>,
) -> PyResult<PyAcquisitionFunction> {
    let f = PyTensorFunction::new(prior_module);
    let prior = PriorModule::new(move |x: &Matrix, with_grad: bool| {
        let (n, d) = x.shape();
        let (values, handle) = call_points(&f, x, with_grad, n, "prior module")?;
        Ok((
            Vector::from_vec(values),
            handle.map(|h| callable_pullback(h, n, d)),
        ))
    });
    let a = PriorGuidedAcquisitionFunction::new(
        Arc::clone(&acq_function.inner),
        prior,
        log,
        prior_exponent,
        x_pending.map(|p| p.matrix("X_pending")).transpose()?,
    )
    .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "PriorGuidedAcquisitionFunction",
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(penalized_acquisition_function, m)?)?;
    m.add_function(wrap_pyfunction!(proximal_acquisition_function, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_feature_acquisition_function, m)?)?;
    m.add_function(wrap_pyfunction!(prior_guided_acquisition_function, m)?)?;
    Ok(())
}
