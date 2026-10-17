//! Models of phase 7b: the deterministic `AffineDeterministicModel` and
//! `FixedSingleSampleModel` (as model handles of acquisition functions), the
//! Gaussian moments of an `EnsembleModel`'s posterior (for analytic
//! acquisition functions), and the (weighted) average of the acquisition
//! functions of the members of an ensemble model
//! (`EnsembleMapSaasSingleTaskGP`, `EnsembleModel`).

use std::sync::Arc;

use botorch_core::acquisition::EnsembleAverage;
use botorch_core::models::{
    AffineDeterministicModel, EnsembleMomentsModel, FixedSingleSampleModel,
};
use botorch_core::Vector;
use pyo3::prelude::*;

use crate::acquisition::PyAcquisitionFunction;
use crate::arrays::Array;
use crate::errors::IntoPyResult;
use crate::models::PyModel;

/// `AffineDeterministicModel(a, b)`: `b + X a` with `a` of shape `d x m`
/// and `b` of length `m`.
#[pyfunction]
pub fn affine_deterministic_model(a: Array, b: Vec<f64>) -> PyResult<PyModel> {
    let a = a.matrix("a")?;
    let model = AffineDeterministicModel::new(a, Vector::from_vec(b)).into_py_result()?;
    Ok(PyModel {
        inner: Arc::new(model),
    })
}

/// `FixedSingleSampleModel(model, w, jitter)`: `mean + sqrt(variance +
/// jitter) * w` of the posterior of a model handle.
#[pyfunction]
pub fn fixed_single_sample_model(
    model: PyRef<'_, PyModel>,
    w: Vec<f64>,
    jitter: f64,
) -> PyResult<PyModel> {
    let m = FixedSingleSampleModel::new(model.inner.clone(), Vector::from_vec(w), jitter)
        .into_py_result()?;
    Ok(PyModel { inner: Arc::new(m) })
}

/// The mean and (unbiased weighted) variance of the `size` members of an
/// ensemble whose values are the `size * m` outputs of `members` (member
/// `k`, output `t` at output `k * m + t`), weighted by `weights` (uniform if
/// `None`): BoTorch's `EnsemblePosterior.mean` / `.variance`.
#[pyfunction]
#[pyo3(signature = (members, size, weights=None))]
pub fn ensemble_moments_model(
    members: PyRef<'_, PyModel>,
    size: usize,
    weights: Option<Vec<f64>>,
) -> PyResult<PyModel> {
    let m = EnsembleMomentsModel::new(members.inner.clone(), size, weights).into_py_result()?;
    Ok(PyModel { inner: Arc::new(m) })
}

/// The average of the acquisition functions of the members of an ensemble
/// (`logmeanexp` for log-scale members), weighted by `weights` if given.
#[pyfunction]
#[pyo3(signature = (members, weights=None))]
pub fn ensemble_average(
    members: Vec<PyRef<'_, PyAcquisitionFunction>>,
    weights: Option<Vec<f64>>,
) -> PyResult<PyAcquisitionFunction> {
    let members: Vec<_> = members.iter().map(|m| m.inner.clone()).collect();
    let a = match weights {
        Some(w) => EnsembleAverage::with_weights(members, w),
        None => EnsembleAverage::new(members),
    }
    .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(
        Arc::new(a),
        "EnsembleAverage",
    ))
}

/// Registers the functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(affine_deterministic_model, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_single_sample_model, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_moments_model, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_average, m)?)?;
    Ok(())
}
