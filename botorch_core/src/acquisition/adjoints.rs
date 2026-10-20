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

/// Compact posterior bars from cached-root noisy sampling. The cached baseline
/// factor and incumbent are constants, matching the reference graph.
#[derive(Debug)]
pub struct CachedPosteriorAdjoints {
    /// Immutable model snapshot.
    pub model: Arc<dyn Model>,
    /// Baseline followed by candidate and pending raw inputs.
    pub inputs: Matrix,
    /// Number of posterior baseline points per output.
    pub baseline_points: usize,
    /// Candidate/pending mean bars, one vector per independent output.
    pub mean: Vec<Vector>,
    /// Candidate-to-baseline covariance bars, one block per output.
    pub cross: Vec<Matrix>,
    /// Candidate/pending covariance bars, one block per output.
    pub covariance: Vec<Matrix>,
}

impl CachedPosteriorAdjoints {
    fn parameter_vjp(&self, weight: f64) -> Result<Vector> {
        let n = self.inputs.nrows() * self.model.points_per_input();
        let nb = self.baseline_points;
        let m = self.model.num_outputs();
        if nb > n || self.mean.len() != m || self.cross.len() != m || self.covariance.len() != m {
            return Err(crate::BotorchError::DimensionMismatch(
                "Cached parameter tape output/baseline layout is inconsistent".into(),
            ));
        }
        let nq = n - nb;
        if (0..m).any(|t| {
            self.mean[t].len() != nq
                || self.cross[t].shape() != (nq, nb)
                || self.covariance[t].shape() != (nq, nq)
        }) {
            return Err(crate::BotorchError::DimensionMismatch(
                "Cached parameter tape mean/covariance block shapes are inconsistent".into(),
            ));
        }
        let mut mean = Vector::zeros(n * m);
        let mut covariance = Matrix::zeros(n * m, n * m);
        for t in 0..m {
            mean.rows_mut(t * n + nb, nq).copy_from(&self.mean[t]);
            covariance
                .view_mut((t * n + nb, t * n), (nq, nb))
                .copy_from(&self.cross[t]);
            covariance
                .view_mut((t * n + nb, t * n + nb), (nq, nq))
                .copy_from(&self.covariance[t]);
        }
        // Baseline covariance bars stay zero: cached L_B is detached. The
        // ordinary model VJP still differentiates the live cross blocks and
        // the full training solve, including baseline input transformations.
        self.model.posterior_parameter_vjp(
            &self.inputs,
            &(mean * weight),
            &(covariance * weight),
            false,
        )
    }
}

/// Actual-path parameter tape. Cached blocks stay compact until parameter
/// backward; query-only cached evaluations do not allocate a dense baseline block.
#[derive(Debug)]
pub enum ParameterAdjoints {
    /// Live joint covariance-root path, including cached-root failure fallback.
    Joint(PosteriorAdjoints),
    /// Successful low-rank update with the baseline root held constant.
    Cached(CachedPosteriorAdjoints),
}

impl ParameterAdjoints {
    /// Contract the saved actual-path bars without reevaluating the acquisition.
    ///
    /// # Errors
    /// Errors from the model's native parameter pullback.
    pub fn parameter_vjp(&self, weight: f64) -> Result<Vector> {
        match self {
            Self::Joint(tape) => tape.parameter_vjp(weight),
            Self::Cached(tape) => tape.parameter_vjp(weight),
        }
    }
}

impl From<PosteriorAdjoints> for ParameterAdjoints {
    fn from(tape: PosteriorAdjoints) -> Self {
        Self::Joint(tape)
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
    pub posterior: ParameterAdjoints,
}
