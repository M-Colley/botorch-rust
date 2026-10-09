//! Acquisition functions (`botorch.acquisition`).
//!
//! An [`AcquisitionFunction`] maps a candidate set — a `q × d` matrix of
//! raw-space points, one per row — to a scalar to be maximized. Values are
//! exactly those of BoTorch's `acqf(X)` for `X` of shape `1 × q × d`, and
//! [`AcquisitionFunction::forward_with_grad`] returns the exact gradient with
//! respect to `X` (what `torch.autograd` computes for the same expression).
//!
//! * [`analytic`] — `ExpectedImprovement`, `LogExpectedImprovement`,
//!   `ProbabilityOfImprovement`, `LogProbabilityOfImprovement`,
//!   `UpperConfidenceBound`, `PosteriorMean`, `PosteriorStandardDeviation`
//!   (`q = 1`).
//! * [`monte_carlo`] — `qExpectedImprovement`, `qProbabilityOfImprovement`,
//!   `qUpperConfidenceBound`, `qSimpleRegret` (any `q`, pending points).
//! * [`logei`] — `qLogExpectedImprovement`, `qLogNoisyExpectedImprovement`.
//! * [`utils`] — `prune_inferior_points`.
//!
//! Acquisition functions hold their model as an `Arc<dyn Model>` and are
//! `Send + Sync`; pending points are stored behind a lock so that
//! [`AcquisitionFunction::set_x_pending`] takes `&self` (like BoTorch's
//! mutable `set_X_pending`).

use std::fmt::Debug;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::Matrix;

pub mod analytic;
pub mod logei;
pub mod monte_carlo;
pub mod utils;

pub use analytic::{
    ExpectedImprovement, LogExpectedImprovement, LogProbabilityOfImprovement, PosteriorMean,
    PosteriorStandardDeviation, ProbabilityOfImprovement, UpperConfidenceBound,
};
pub use logei::{
    QLogExpectedImprovement, QLogExpectedImprovementOptions, QLogNoisyExpectedImprovement,
    QLogNoisyExpectedImprovementOptions,
};
pub use monte_carlo::{
    McAcquisitionOptions, QExpectedImprovement, QProbabilityOfImprovement, QSimpleRegret,
    QUpperConfidenceBound,
};
pub use utils::{prune_inferior_points, prune_inferior_points_indices, PruneOptions};

/// An acquisition function `α(X)` over candidate sets `X` (`q × d`, raw input
/// space), to be maximized (BoTorch's `AcquisitionFunction`).
pub trait AcquisitionFunction: Send + Sync + Debug {
    /// The model the acquisition function is defined on.
    fn model(&self) -> &Arc<dyn Model>;

    /// Input dimension `d`.
    fn dim(&self) -> usize {
        self.model().dim()
    }

    /// The required number of points per candidate set (`Some(1)` for analytic
    /// acquisition functions), or `None` if any `q` is supported.
    fn expected_q(&self) -> Option<usize> {
        None
    }

    /// The acquisition value of the candidate set `x` (`q × d`), i.e.
    /// BoTorch's `acqf(X)` for `X` of shape `1 × q × d`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns (or
    /// of rows when [`AcquisitionFunction::expected_q`] is set), and model
    /// errors.
    fn forward(&self, x: &Matrix) -> Result<f64>;

    /// The acquisition value and its gradient with respect to `x` (`q × d`).
    ///
    /// # Errors
    /// See [`AcquisitionFunction::forward`].
    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)>;

    /// Values for a batch of candidate sets (BoTorch's `acqf(X)` for `X` of
    /// shape `b × q × d`), evaluated in parallel.
    ///
    /// # Errors
    /// The first error of [`AcquisitionFunction::forward`].
    fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
        xs.par_iter().map(|x| self.forward(x)).collect()
    }

    /// The pending points (`m × d`) that are concatenated to every candidate
    /// set, if any.
    fn x_pending(&self) -> Option<Matrix> {
        None
    }

    /// Informs the acquisition function about pending points (BoTorch's
    /// `set_X_pending`); `None` clears them.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for acquisition functions that do not
    /// account for pending points (the analytic ones, as in BoTorch) and
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns.
    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        let _ = x_pending;
        Err(BotorchError::Unsupported(
            "Analytic acquisition functions do not account for X_pending yet.".into(),
        ))
    }

    /// Whether the acquisition function is known to be non-negative (BoTorch's
    /// `is_nonnegative`), which selects `initialize_q_batch_nonneg` in
    /// [`crate::optim::optimize_acqf`].
    fn is_nonnegative(&self) -> bool {
        false
    }

    /// Whether the values are on a log scale (BoTorch's `_log` attribute).
    fn is_log(&self) -> bool {
        false
    }
}

/// Checks the shape of a candidate set.
pub(crate) fn check_candidates(x: &Matrix, d: usize, expected_q: Option<usize>) -> Result<()> {
    if x.ncols() != d {
        return Err(BotorchError::DimensionMismatch(format!(
            "Expected X with {d} columns (d), got {}",
            x.ncols()
        )));
    }
    if x.nrows() == 0 {
        return Err(BotorchError::DimensionMismatch(
            "X must contain at least one point".into(),
        ));
    }
    if let Some(q) = expected_q {
        if x.nrows() != q {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X to be `batch_shape x q={q} x d`, but got X with {} points.",
                x.nrows()
            )));
        }
    }
    Ok(())
}

/// Checks the shape of pending points.
pub(crate) fn check_pending(x_pending: &Matrix, d: usize) -> Result<()> {
    if x_pending.ncols() != d {
        return Err(BotorchError::DimensionMismatch(format!(
            "Expected X_pending with {d} columns, got {}",
            x_pending.ncols()
        )));
    }
    Ok(())
}
