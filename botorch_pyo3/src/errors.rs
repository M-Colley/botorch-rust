//! Mapping of [`BotorchError`] onto Python exceptions.
//!
//! The exception classes live in `botorch_rs.exceptions.errors` (pure Python,
//! so that they can inherit from both `BotorchRsError` and the matching
//! built-in exception, e.g. `BotorchTensorDimensionError(BotorchRsError,
//! ValueError)`). If that module cannot be imported (the extension used on its
//! own), the built-in base exception is raised instead.

use botorch_core::BotorchError;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

/// Module that defines the exception hierarchy.
const ERRORS_MODULE: &str = "botorch_rs.exceptions.errors";

type Fallback = fn(String) -> PyErr;

/// Creates an instance of `botorch_rs.exceptions.errors.<class>` with `msg`.
fn raise(class: &str, fallback: Fallback, msg: String) -> PyErr {
    Python::attach(|py| {
        let ty = py
            .import(ERRORS_MODULE)
            .and_then(|m| m.getattr(class))
            .and_then(|t| t.cast_into::<PyType>().map_err(PyErr::from));
        match ty {
            Ok(ty) => PyErr::from_type(ty, msg),
            Err(_) => fallback(msg),
        }
    })
}

/// Converts a core error into the corresponding Python exception.
pub fn to_pyerr(e: BotorchError) -> PyErr {
    match e {
        BotorchError::DimensionMismatch(msg) => dimension_error(msg),
        BotorchError::InputData(msg) => raise("InputDataError", PyValueError::new_err, msg),
        BotorchError::InvalidArgument(msg) => value_error(msg),
        BotorchError::Unsupported(msg) => unsupported(msg),
        e @ BotorchError::NotPsd { .. } => {
            raise("NotPSDError", PyRuntimeError::new_err, e.to_string())
        }
        e @ BotorchError::Nan { .. } => raise("NanError", PyRuntimeError::new_err, e.to_string()),
        BotorchError::Numerical(msg) => raise("NumericalError", PyRuntimeError::new_err, msg),
        BotorchError::ModelFitting(msg) => raise("ModelFittingError", PyRuntimeError::new_err, msg),
    }
}

/// `BotorchTensorDimensionError` (a `ValueError`).
pub fn dimension_error(msg: impl Into<String>) -> PyErr {
    raise(
        "BotorchTensorDimensionError",
        PyValueError::new_err,
        msg.into(),
    )
}

/// `InvalidArgumentError` (a `ValueError`).
pub fn value_error(msg: impl Into<String>) -> PyErr {
    raise("InvalidArgumentError", PyValueError::new_err, msg.into())
}

/// `UnsupportedError` (a `NotImplementedError`).
pub fn unsupported(msg: impl Into<String>) -> PyErr {
    raise(
        "UnsupportedError",
        PyNotImplementedError::new_err,
        msg.into(),
    )
}

/// Conversion of `botorch_core` results into `PyResult`s.
pub trait IntoPyResult<T> {
    /// Maps the error with [`to_pyerr`].
    fn into_py_result(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for botorch_core::Result<T> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(to_pyerr)
    }
}
