//! Error type shared by all `botorch_core` APIs.
//!
//! The variants mirror the exceptions raised by BoTorch / GPyTorch /
//! linear_operator in the corresponding situations, so that later bindings can
//! map them onto the familiar Python exception types.

use thiserror::Error;

/// Errors produced by `botorch_core`.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum BotorchError {
    /// Input shapes are inconsistent (BoTorch `BotorchTensorDimensionError`).
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    /// Invalid input data such as NaN values or negative noise variances
    /// (BoTorch `InputDataError`).
    #[error("invalid input data: {0}")]
    InputData(String),
    /// An invalid argument was supplied (e.g. a non-positive prior scale).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// The requested feature is not supported (BoTorch `UnsupportedError`).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A matrix could not be Cholesky-factorized even after adding jitter
    /// (linear_operator `NotPSDError`).
    #[error("matrix not positive definite after repeatedly adding jitter up to {jitter:.1e}")]
    NotPsd {
        /// The largest jitter that was added to the diagonal.
        jitter: f64,
    },
    /// A matrix to be factorized contains NaN values (linear_operator `NanError`).
    #[error("cholesky: {count} of {total} elements of the matrix are NaN")]
    Nan {
        /// Number of NaN entries.
        count: usize,
        /// Total number of entries.
        total: usize,
    },
    /// A numerical problem such as non-finite objective values.
    #[error("numerical error: {0}")]
    Numerical(String),
    /// All model fitting attempts failed (BoTorch `ModelFittingError`).
    #[error("model fitting failed: {0}")]
    ModelFitting(String),
}

/// Convenience alias for `Result<T, BotorchError>`.
pub type Result<T> = std::result::Result<T, BotorchError>;
