//! The [`Model`] trait: the interface between probabilistic models and
//! acquisition functions (BoTorch's `botorch.models.model.Model` API, restricted
//! to what acquisition functions need).
//!
//! All inputs are in the *raw* (original) input space and all outputs in the
//! *original* outcome space, exactly like BoTorch's `model.posterior(X)`.
//! Implementations must be `Send + Sync` so that acquisition functions can
//! share them across threads (`Arc<dyn Model>`).

use std::fmt::Debug;
use std::ops::Range;

use crate::error::{BotorchError, Result};
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::{Matrix, Vector};

/// A single-output probabilistic model with Gaussian posteriors, as consumed
/// by the acquisition functions of this crate.
///
/// Point sets are `n × d` matrices whose rows are points. Derivatives are taken
/// with respect to the raw-space inputs.
pub trait Model: Send + Sync + Debug {
    /// Input dimension `d`.
    fn dim(&self) -> usize;

    /// Number of outputs `m` (BoTorch's `model.num_outputs`).
    fn num_outputs(&self) -> usize;

    /// Training inputs in the original input space (`n × d`).
    fn train_inputs(&self) -> &Matrix;

    /// Training targets in the original outcome space (`n × m`).
    fn train_targets(&self) -> &Matrix;

    /// Joint posterior at the `q × d` inputs `x` (BoTorch's
    /// `model.posterior(X, observation_noise=...)`).
    ///
    /// # Errors
    /// Implementation-specific (dimension mismatches, numerical failures).
    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior>;

    /// Marginal posterior means and variances of every row of `x` evaluated
    /// independently, i.e. the posterior of a `b × 1 × d` batch in BoTorch.
    /// Variances are clamped from below like `posterior.variance`.
    ///
    /// The default implementation calls [`Model::posterior`] row by row.
    ///
    /// # Errors
    /// See [`Model::posterior`].
    fn posterior_marginals(&self, x: &Matrix) -> Result<(Vector, Vector)> {
        let b = x.nrows();
        let mut mean = Vector::zeros(b);
        let mut var = Vector::zeros(b);
        for i in 0..b {
            let p = self.posterior(&x.rows(i, 1).into_owned(), false)?;
            mean[i] = p.mean[0];
            var[i] = p.variance()[0];
        }
        Ok((mean, var))
    }

    /// Posterior mean and (clamped) variance at a single point (length `d`)
    /// together with their gradients with respect to the point:
    /// `(mean, variance, ∂mean/∂x, ∂variance/∂x)`.
    ///
    /// # Errors
    /// See [`Model::posterior`].
    fn posterior_q1_with_grad(&self, x: &[f64]) -> Result<(f64, f64, Vector, Vector)>;

    /// Joint posterior (without observation noise) at `x` with the
    /// derivatives of the mean and covariance with respect to every input
    /// coordinate (forward mode, see [`JointPosteriorGrad`]).
    ///
    /// # Errors
    /// See [`Model::posterior`].
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad>;

    /// Reverse-mode derivative (vector-Jacobian product) of the joint
    /// posterior without observation noise: for the scalar
    /// `s = ⟨mean_adjoint, μ(x)⟩ + Σ_ab covariance_adjoint[a,b] Σ(x)[a,b]`,
    /// returns `∂s/∂x[p, k]` for the rows `p ∈ rows` (a `rows.len() × d`
    /// matrix). Only the symmetric part of `covariance_adjoint` matters.
    ///
    /// This is what acquisition functions need to backpropagate through
    /// posterior samples. The default implementation contracts
    /// [`Model::joint_posterior_with_grad`]; models should override it with a
    /// cheaper direct computation.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes, otherwise
    /// see [`Model::posterior`].
    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        check_vjp_shapes(x, &rows, mean_adjoint, covariance_adjoint)?;
        let d = x.ncols();
        let g = self.joint_posterior_with_grad(x)?;
        let mut out = Matrix::zeros(rows.len(), d);
        for (r, p) in rows.enumerate() {
            for k in 0..d {
                let j = p * d + k;
                out[(r, k)] = mean_adjoint.dot(&g.dmean[j])
                    + covariance_adjoint.component_mul(&g.dcov[j]).sum();
            }
        }
        Ok(out)
    }
}

/// Validates the arguments of [`Model::joint_posterior_vjp`].
pub(crate) fn check_vjp_shapes(
    x: &Matrix,
    rows: &Range<usize>,
    mean_adjoint: &Vector,
    covariance_adjoint: &Matrix,
) -> Result<()> {
    let n = x.nrows();
    if rows.start > rows.end || rows.end > n {
        return Err(BotorchError::DimensionMismatch(format!(
            "rows {rows:?} out of range for {n} points"
        )));
    }
    if mean_adjoint.len() != n || covariance_adjoint.shape() != (n, n) {
        return Err(BotorchError::DimensionMismatch(format!(
            "adjoints must have shapes {n} and {n}x{n}, got {} and {}x{}",
            mean_adjoint.len(),
            covariance_adjoint.nrows(),
            covariance_adjoint.ncols()
        )));
    }
    Ok(())
}
