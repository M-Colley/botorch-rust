//! Gaussian posterior containers returned by [`crate::SingleTaskGP`].

use crate::{Matrix, Vector};

/// Minimum variance returned by [`Posterior::variance`]
/// (`gpytorch.settings.min_variance` for double precision).
pub const MIN_VARIANCE: f64 = 1e-10;

/// A joint Gaussian posterior over `q` points, in the original outcome space.
///
/// This corresponds to BoTorch's `GPyTorchPosterior` for a single-output
/// model: `mean` is `posterior.mean.squeeze(-1)` and `covariance` is
/// `posterior.distribution.covariance_matrix`.
#[derive(Clone, Debug, PartialEq)]
pub struct Posterior {
    /// Posterior mean (length `q`).
    pub mean: Vector,
    /// Posterior covariance (`q × q`).
    pub covariance: Matrix,
}

impl Posterior {
    /// Number of points `q`.
    pub fn len(&self) -> usize {
        self.mean.len()
    }

    /// Whether the posterior is over zero points.
    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    /// Marginal variances, clamped from below at [`MIN_VARIANCE`] exactly like
    /// GPyTorch's `MultivariateNormal.variance` (which BoTorch's
    /// `posterior.variance` returns).
    pub fn variance(&self) -> Vector {
        self.covariance.diagonal().map(|v| v.max(MIN_VARIANCE))
    }

    /// Unclamped diagonal of the covariance matrix.
    pub fn raw_variance(&self) -> Vector {
        self.covariance.diagonal()
    }
}

/// Joint posterior over `q` points together with derivatives with respect to
/// every input coordinate, all in the original (untransformed) input and
/// outcome spaces.
///
/// Derivatives are indexed by `j = p·d + k`, the `k`-th coordinate of the
/// `p`-th point: `dmean[j][a] = ∂mean[a]/∂x[p,k]` and
/// `dcov[j][(a, b)] = ∂cov[a,b]/∂x[p,k]`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointPosteriorGrad {
    /// Posterior mean (length `q`).
    pub mean: Vector,
    /// Posterior covariance (`q × q`, without observation noise).
    pub covariance: Matrix,
    /// Mean derivatives (`q·d` vectors of length `q`).
    pub dmean: Vec<Vector>,
    /// Covariance derivatives (`q·d` matrices of size `q × q`).
    pub dcov: Vec<Matrix>,
}
