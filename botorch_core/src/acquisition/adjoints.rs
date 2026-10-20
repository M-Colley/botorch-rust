//! Immutable posterior bars recorded by a single acquisition reverse pass.

use std::sync::Arc;

use crate::{Matrix, Model, Result, Vector};

/// One Gaussian law and its unit acquisition-value adjoints.
#[derive(Debug)]
pub struct PosteriorAdjoints {
    /// Immutable native model used for the acquisition evaluation.
    pub model: Arc<dyn Model>,
    /// Full points, including pending points when present.
    pub inputs: Matrix,
    /// Gaussian-base mean adjoints, output-major.
    pub mean: Vector,
    /// Gaussian-base covariance adjoints, output-major.
    pub covariance: Matrix,
}

impl PosteriorAdjoints {
    /// Contract this tape into raw parameters with an upstream scalar weight.
    ///
    /// # Errors
    /// The model's analytic posterior parameter pullback errors.
    pub fn parameter_vjp(&self, weight: f64) -> Result<Vector> {
        self.model.posterior_parameter_vjp(
            &self.inputs,
            &(&self.mean * weight),
            &(&self.covariance * weight),
            false,
        )
    }
}

/// Acquisition value, optional query gradient, and retained posterior tape.
#[derive(Debug)]
pub struct AcquisitionAdjoints {
    /// Scalar acquisition value for one candidate set.
    pub value: f64,
    /// Exact query-input derivative, when requested.
    pub query_gradient: Option<Matrix>,
    /// Model-parameter reverse tape; no acquisition reevaluation is needed.
    pub posterior: PosteriorAdjoints,
}
