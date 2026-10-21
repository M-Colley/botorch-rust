//! Heterogeneous target-task acquisition input embedding.

use std::sync::Arc;

use botorch_core::models::heterogeneous::TargetTaskModel;
use pyo3::prelude::*;

use crate::errors::IntoPyResult;
use crate::models::PyModel;

/// A full-feature native model viewed through target-task inputs.
#[pyfunction]
pub fn target_task_model(
    model: PyRef<'_, PyModel>,
    feature_indices: Vec<usize>,
    imputation: Vec<f64>,
) -> PyResult<PyModel> {
    Ok(PyModel {
        inner: Arc::new(
            TargetTaskModel::new(model.inner.clone(), feature_indices, imputation)
                .into_py_result()?,
        ),
    })
}
