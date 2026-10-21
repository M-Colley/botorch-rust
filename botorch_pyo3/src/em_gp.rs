//! Exact EM prior estimation and fixed-state interpolation bindings.

use crate::arrays::{matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::specs::PyPrior;
use crate::threads;
use botorch_core::kernels::TreeKernel;
use botorch_core::models::em_gp::{pretrain_em_prior, EMDataset, EMOptions};
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;

type EMOutput = (
    ArrayOut,
    ArrayBuf,
    ArrayBuf,
    ArrayOut,
    Vec<(ArrayOut, ArrayBuf)>,
    usize,
);

pub(crate) fn matrix(a: Array, name: &str) -> PyResult<Matrix> {
    if a.1.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(a.0.len()) {
        return Err(value_error(format!(
            "{name} length differs from declared shape"
        )));
    }
    a.matrix(name)
}
pub(crate) fn vector(a: Array, name: &str) -> PyResult<Vector> {
    if !(a.1.len() == 1 || a.1.len() == 2 && a.1[1] == 1)
        || a.1.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(a.0.len())
    {
        return Err(value_error(format!(
            "{name} requires a vector or n x 1 array"
        )));
    }
    Ok(Vector::from_column_slice(&a.0))
}
fn vout(v: Vector) -> ArrayOut {
    (v.as_slice().to_vec(), vec![v.len()])
}

#[pyfunction]
#[pyo3(signature=(xs,ys,inducing,kernel,raw,constant,indices,noise,iterations,mean_prior,covariance_prior,iw_nu,tolerance,naive_init,shrinkage,record_history))]
#[allow(clippy::too_many_arguments)]
fn em_pretrain(
    py: Python<'_>,
    xs: Vec<Array>,
    ys: Vec<Array>,
    inducing: Array,
    kernel: PyRef<'_, PyKernelNode>,
    raw: Vec<f64>,
    constant: f64,
    indices: Option<Vec<Vec<usize>>>,
    noise: f64,
    iterations: usize,
    mean_prior: bool,
    covariance_prior: bool,
    iw_nu: Option<f64>,
    tolerance: Option<f64>,
    naive_init: bool,
    shrinkage: f64,
    record_history: bool,
) -> PyResult<EMOutput> {
    if xs.len() != ys.len() {
        return Err(value_error("EM dataset input and target counts differ"));
    }
    let data = xs
        .into_iter()
        .zip(ys)
        .map(|(x, y)| {
            Ok(EMDataset {
                x: matrix(x, "dataset X")?,
                y: vector(y, "dataset Y")?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let inducing = matrix(inducing, "inducing points")?;
    let kernel = TreeKernel::build(&kernel.inner, &raw, inducing.ncols()).into_py_result()?;
    let options = EMOptions {
        noise,
        iterations,
        mean_prior,
        covariance_prior,
        iw_nu,
        tolerance,
        naive_init,
        shrinkage,
        record_history,
    };
    let result = threads::run(py, || {
        pretrain_em_prior(
            &data,
            kernel,
            inducing,
            constant,
            indices.as_deref(),
            &options,
        )
    })
    .into_py_result()?;
    Ok((
        vout(result.mean),
        matrix_out(&result.covariance).into(),
        matrix_out(&result.kernel_cholesky).into(),
        vout(result.delta_mean),
        result
            .history
            .into_iter()
            .map(|(m, c)| (vout(m), matrix_out(&c).into()))
            .collect(),
        result.iterations,
    ))
}

/// Lossless auxiliary metadata needed to serialize native prior descriptors.
#[pyfunction]
fn em_prior_state(prior: PyRef<'_, PyPrior>) -> (&'static str, Vec<f64>, bool, Option<f64>) {
    use botorch_core::priors::Prior;
    let (float32, torch_log) = match prior.inner {
        Prior::LogNormal {
            float32, torch_log, ..
        }
        | Prior::HalfNormal {
            float32, torch_log, ..
        }
        | Prior::Uniform {
            float32, torch_log, ..
        } => (float32, torch_log),
        _ => (false, None),
    };
    (
        prior.inner.name(),
        prior.inner.parameters(),
        float32,
        torch_log,
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(em_pretrain, m)?)?;
    m.add_function(wrap_pyfunction!(em_prior_state, m)?)
}
