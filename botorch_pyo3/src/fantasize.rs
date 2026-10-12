//! Conditioning on observations and fantasy models (BoTorch's
//! `condition_on_observations` and `fantasize` of `SingleTaskGP` and
//! `ModelListGP`): every function returns new model handles; the Python layer
//! wraps them in models that share the original's modules.

use std::sync::Arc;

use botorch_core::models::{BatchedMatrix, BoolMatrix, FantasySampler};
use botorch_core::{BatchedMultiOutputGP, ModelListGP};
use pyo3::prelude::*;

use crate::arrays::Array;
use crate::errors::{dimension_error, IntoPyResult};
use crate::models::PySingleTaskGP;
use crate::sampling::PySobolQMCNormalSampler;
use crate::threads;

/// A `batch_shape x r x c` array as a batch of matrices.
fn batched(a: &Array, name: &str) -> PyResult<BatchedMatrix> {
    let (batch, mats) = a.batches(name)?;
    BatchedMatrix::new(batch, mats).into_py_result()
}

/// An evaluation mask (`n x m`) from `(data, shape)`.
fn bool_matrix(mask: &(Vec<bool>, Vec<usize>)) -> PyResult<BoolMatrix> {
    match mask.1.as_slice() {
        &[r, c] if r * c == mask.0.len() => Ok(BoolMatrix::from_row_slice(r, c, &mask.0)),
        s => Err(dimension_error(format!(
            "Expected a 2-dimensional evaluation mask, got shape {s:?}."
        ))),
    }
}

/// BoTorch's `SingleTaskGP.condition_on_observations(X, Y, noise=noise)`:
/// `X` (`batch_x x n' x d`), `Y` (`batch_y x n' x m`, original space) and
/// optional `noise` (shape of `Y`, transformed space); returns the
/// conditioned model.
#[pyfunction]
#[pyo3(signature = (model, x, y, noise=None))]
pub fn condition_on_observations(
    py: Python<'_>,
    model: PyRef<'_, PySingleTaskGP>,
    x: Array,
    y: Array,
    noise: Option<Array>,
) -> PyResult<PySingleTaskGP> {
    let gp = model.snapshot();
    let x = batched(&x, "X")?;
    let y = batched(&y, "Y")?;
    let noise = noise.map(|v| batched(&v, "noise")).transpose()?;
    let out = threads::run(py, || gp.condition_on_observations(&x, &y, noise.as_ref()))
        .into_py_result()?;
    Ok(PySingleTaskGP::from_model(out))
}

/// BoTorch's `SingleTaskGP.fantasize(X, sampler, observation_noise)`:
/// returns the fantasy model (batch shape `[n_f] + broadcast(batch_x,
/// model batch shape)`).
#[pyfunction]
#[pyo3(signature = (model, x, sampler, observation_noise=None))]
pub fn fantasize(
    py: Python<'_>,
    model: PyRef<'_, PySingleTaskGP>,
    x: Array,
    sampler: PyRef<'_, PySobolQMCNormalSampler>,
    observation_noise: Option<Array>,
) -> PyResult<PySingleTaskGP> {
    let gp = model.snapshot();
    let x = batched(&x, "X")?;
    let noise = observation_noise
        .map(|v| batched(&v, "observation_noise"))
        .transpose()?;
    let s = Arc::clone(&sampler.inner);
    let out = threads::run(py, || gp.fantasize(&x, &s, noise.as_ref())).into_py_result()?;
    Ok(PySingleTaskGP::from_model(out))
}

fn list_of(models: &[PyRef<'_, PySingleTaskGP>]) -> PyResult<ModelListGP> {
    ModelListGP::from_shared(models.iter().map(|m| m.snapshot()).collect()).into_py_result()
}

fn handles(list: ModelListGP) -> Vec<PySingleTaskGP> {
    list.models()
        .iter()
        .map(|m| PySingleTaskGP::from_model(BatchedMultiOutputGP::clone(m)))
        .collect()
}

/// BoTorch's `ModelListGP.condition_on_observations(X, Y, noise=noise)` (with
/// meta-pytorch/botorch#3399): `xs` holds the inputs of every output, `noise`
/// is in the original space; returns the conditioned sub-models.
#[pyfunction]
#[pyo3(signature = (models, xs, y, noise=None))]
pub fn model_list_condition_on_observations(
    py: Python<'_>,
    models: Vec<PyRef<'_, PySingleTaskGP>>,
    xs: Vec<Array>,
    y: Array,
    noise: Option<Array>,
) -> PyResult<Vec<PySingleTaskGP>> {
    let list = list_of(&models)?;
    let xs = xs
        .iter()
        .map(|x| batched(x, "X"))
        .collect::<PyResult<Vec<_>>>()?;
    let y = batched(&y, "Y")?;
    let noise = noise.map(|v| batched(&v, "noise")).transpose()?;
    let out = threads::run(py, || {
        list.condition_on_observations(&xs, &y, noise.as_ref())
    })
    .into_py_result()?;
    Ok(handles(out))
}

/// BoTorch's `ModelListGP.fantasize(X, sampler, observation_noise,
/// evaluation_mask)`: one sampler per output (`shared=False`, a
/// `ListSampler`) or one for all; returns the fantasy sub-models.
#[pyfunction]
#[pyo3(signature = (models, x, samplers, shared, observation_noise=None, evaluation_mask=None))]
pub fn model_list_fantasize(
    py: Python<'_>,
    models: Vec<PyRef<'_, PySingleTaskGP>>,
    x: Array,
    samplers: Vec<PyRef<'_, PySobolQMCNormalSampler>>,
    shared: bool,
    observation_noise: Option<Array>,
    evaluation_mask: Option<(Vec<bool>, Vec<usize>)>,
) -> PyResult<Vec<PySingleTaskGP>> {
    let list = list_of(&models)?;
    let x = batched(&x, "X")?;
    let noise = observation_noise
        .map(|v| batched(&v, "observation_noise"))
        .transpose()?;
    let mask = evaluation_mask.as_ref().map(bool_matrix).transpose()?;
    let samplers: Vec<Arc<botorch_core::SobolQMCNormalSampler>> =
        samplers.iter().map(|s| Arc::clone(&s.inner)).collect();
    let out = threads::run(py, || {
        let sampler = if shared {
            FantasySampler::Shared(&samplers[0])
        } else {
            FantasySampler::PerOutput(&samplers)
        };
        list.fantasize(&x, sampler, noise.as_ref(), mask.as_ref())
    })
    .into_py_result()?;
    Ok(handles(out))
}

/// Registers the functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(condition_on_observations, m)?)?;
    m.add_function(wrap_pyfunction!(fantasize, m)?)?;
    m.add_function(wrap_pyfunction!(model_list_condition_on_observations, m)?)?;
    m.add_function(wrap_pyfunction!(model_list_fantasize, m)?)?;
    Ok(())
}
