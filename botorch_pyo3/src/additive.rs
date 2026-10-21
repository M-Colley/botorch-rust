//! Orthogonal additive covariance inspection and fitted component models.

use crate::arrays::{matrix_out, Array, ArrayBuf};
use crate::errors::{value_error, IntoPyResult};
use crate::kernel_trees::PyKernelNode;
use crate::models::{PyModel, PySingleTaskGP};
use botorch_core::kernels::TreeKernel;
use botorch_core::models::gp::additive::AdditiveComponentModel;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyfunction]
#[pyo3(signature = (model, extra_batch_dims=0))]
pub fn additive_component_model(
    py: Python<'_>,
    model: PyRef<'_, PySingleTaskGP>,
    extra_batch_dims: usize,
) -> PyResult<PyModel> {
    let snapshot = model.snapshot();
    crate::threads::run(py, || {
        AdditiveComponentModel::new(&snapshot, extra_batch_dims)
    })
    .into_py_result()
    .map(|m| PyModel { inner: Arc::new(m) })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (node, raw, input_dim, x1=None, x2=None, mode="base", diag=false, eps=1e-6))]
pub fn additive_kernel_parts(
    py: Python<'_>,
    node: PyRef<'_, PyKernelNode>,
    raw: Vec<f64>,
    input_dim: usize,
    x1: Option<Array>,
    x2: Option<Array>,
    mode: &str,
    diag: bool,
    eps: f64,
) -> PyResult<ArrayBuf> {
    let a = x1.map(|x| x.matrix("x1")).transpose()?;
    let b = x2.map(|x| x.matrix("x2")).transpose()?;
    let tree = &node.inner;
    let output = crate::threads::run(py, || -> botorch_core::Result<ArrayBuf> {
        let tree = TreeKernel::build(tree, &raw, input_dim)?;
        if mode == "normalizer" {
            let values = tree.additive_normalizers(eps)?;
            let d = values.len();
            return Ok(ArrayBuf(values, vec![d, 1, 1]));
        }
        let a = a
            .as_ref()
            .ok_or_else(|| botorch_core::BotorchError::InvalidArgument("x1 is required".into()))?;
        let b = b.as_ref().unwrap_or(a);
        let matrices = match mode {
            "base" => tree.additive_base_covariances(a, b)?,
            "components" => tree.additive_components(&tree.prepare(a)?, &tree.prepare(b)?)?,
            _ => {
                return Err(botorch_core::BotorchError::InvalidArgument(
                    "unknown additive kernel mode".into(),
                ))
            }
        };
        let count = matrices.len();
        let mut values = Vec::new();
        for matrix in matrices {
            if diag {
                if matrix.nrows() != matrix.ncols() {
                    return Err(botorch_core::BotorchError::DimensionMismatch(
                        "diag=True requires equal point counts".into(),
                    ));
                }
                values.extend(matrix.diagonal().iter().copied());
            } else {
                values.extend(matrix_out(&matrix).0);
            }
        }
        let mut shape = vec![count, a.nrows()];
        if !diag {
            shape.push(b.nrows());
        }
        Ok(ArrayBuf(values, shape))
    });
    if !matches!(mode, "base" | "components" | "normalizer") {
        return Err(value_error("unknown additive kernel mode"));
    }
    output.into_py_result()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(additive_component_model, m)?)?;
    m.add_function(wrap_pyfunction!(additive_kernel_parts, m)?)?;
    Ok(())
}
