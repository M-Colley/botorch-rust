//! The [`Model`] trait: the interface between probabilistic models and
//! acquisition functions (BoTorch's `botorch.models.model.Model` API, restricted
//! to what acquisition functions need).
//!
//! All inputs are in the *raw* (original) input space and all outputs in the
//! *original* outcome space, exactly like BoTorch's `model.posterior(X)`.
//! Implementations must be `Send + Sync` so that acquisition functions can
//! share them across threads (`Arc<dyn Model>`).
//!
//! A model has `m = num_outputs()` outputs. Joint posteriors over `q` points
//! are flattened in BoTorch's output-major order (index `t·q + p` for output
//! `t` at point `p`, see [`crate::posterior`]), which for single-output
//! models is the plain `q`-dimensional posterior. The methods come in three
//! groups:
//!
//! * the joint posterior ([`Model::posterior`]) with forward-mode
//!   ([`Model::joint_posterior_with_grad`]) and reverse-mode
//!   ([`Model::joint_posterior_vjp`]) derivatives with respect to the inputs,
//!   used by Monte-Carlo acquisition functions;
//! * posteriors of individual points over all outputs, with unclamped output
//!   covariances ([`Model::point_posteriors`],
//!   [`Model::point_posterior_with_grad`]), used by analytic acquisition
//!   functions of multi-output models and by posterior transforms;
//! * single-output conveniences with BoTorch's variance clamping
//!   ([`Model::posterior_marginals`], [`Model::posterior_q1_with_grad`]).
//!
//! Only the first group has to be implemented; the others have default
//! implementations in terms of it, which models override with faster (and
//! mutually bitwise-consistent) computations.

use std::fmt::Debug;
use std::ops::Range;

use crate::error::{BotorchError, Result};
use crate::models::SingleTaskGP;
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior, MIN_VARIANCE};
use crate::{Matrix, Vector};

/// A probabilistic model with Gaussian posteriors, as consumed by the
/// acquisition functions of this crate.
///
/// Point sets are `n × d` matrices whose rows are points. Derivatives are taken
/// with respect to the raw-space inputs.
pub trait Model: Send + Sync + Debug {
    /// Input dimension `d`.
    fn dim(&self) -> usize;

    /// Number of outputs `m` (BoTorch's `model.num_outputs`).
    fn num_outputs(&self) -> usize;

    /// The batch shape of the model (BoTorch's `model.batch_shape`); empty for
    /// unbatched models. Acquisition functions require unbatched models.
    fn batch_shape(&self) -> &[usize] {
        &[]
    }

    /// The independent single-output exact GPs that produce the outputs of
    /// an unbatched model, in output order (`None` otherwise, the default).
    /// Acquisition functions use them to compute joint posteriors with a
    /// fixed point set from cached quantities (BoTorch's `cache_root`, see
    /// [`SingleTaskGP::cross_posterior`]); without them they fall back to
    /// [`Model::posterior`].
    fn output_gps(&self) -> Option<Vec<&SingleTaskGP>> {
        None
    }

    /// The number of posterior points per input point: `n_w` for a model
    /// with a one-to-many input transform (BoTorch's `InputPerturbation`,
    /// see [`crate::transforms::input`]), 1 otherwise (the default).
    ///
    /// The posterior of `q` points is then over `q·n_w` points, point-major
    /// (point `i·n_w + w` is perturbation `w` of input `i`); the adjoints of
    /// [`Model::joint_posterior_vjp`] have that size while `rows` and the
    /// result refer to the `q` inputs. Methods that evaluate every input on
    /// its own ([`Model::point_posteriors`] and the single-point methods)
    /// require one point per input.
    fn points_per_input(&self) -> usize {
        1
    }

    /// Whether the model is deterministic (BoTorch's `DeterministicModel`,
    /// e.g. [`crate::models::PosteriorMeanModel`]): its posterior is a point
    /// mass with zero covariance, so Monte-Carlo samples equal the mean for
    /// any base samples, and its variances are exactly zero (BoTorch's
    /// `EnsemblePosterior` of one member does not clamp them). `false` by
    /// default.
    fn is_deterministic(&self) -> bool {
        false
    }

    /// Joint posterior at the `q × d` inputs `x` over all outputs (BoTorch's
    /// `model.posterior(X, observation_noise=...)`), flattened in output-major
    /// order.
    ///
    /// # Errors
    /// Implementation-specific (dimension mismatches, numerical failures).
    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior>;

    /// Joint posterior (without observation noise) at `x` with the
    /// derivatives of the mean and covariance with respect to every input
    /// coordinate (forward mode, see [`JointPosteriorGrad`]).
    ///
    /// # Errors
    /// See [`Model::posterior`].
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad>;

    /// Reverse-mode derivative (vector-Jacobian product) of the joint
    /// posterior without observation noise: for the scalar
    /// `s = ⟨mean_adjoint, μ(x)⟩ + Σ_ab covariance_adjoint[a,b] Σ(x)[a,b]`
    /// (adjoints of length `n·m` and size `n·m × n·m` in the output-major
    /// order of [`Model::posterior`], `n = x.nrows()`), returns `∂s/∂x[p, k]`
    /// for the rows `p ∈ rows` (a `rows.len() × d` matrix). Only the symmetric
    /// part of `covariance_adjoint` matters.
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
        check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            self.num_outputs(),
            self.points_per_input(),
        )?;
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

    /// Posteriors of every row of `x`, each evaluated on its own (a
    /// `b × 1 × d` batch in BoTorch), over all outputs, with *unclamped*
    /// output covariances and without observation noise.
    ///
    /// The default implementation calls [`Model::posterior`] row by row.
    ///
    /// # Errors
    /// See [`Model::posterior`].
    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        let b = x.nrows();
        let m = self.num_outputs();
        let mut mean = Matrix::zeros(b, m);
        let mut covariance = Matrix::zeros(b, m * m);
        for i in 0..b {
            let p = self.posterior(&x.rows(i, 1).into_owned(), false)?;
            check_point_posterior(&p.mean, m)?;
            for a in 0..m {
                mean[(i, a)] = p.mean[a];
                for c in 0..m {
                    covariance[(i, a * m + c)] = p.covariance[(a, c)];
                }
            }
        }
        Ok(PointPosteriors { mean, covariance })
    }

    /// The posterior at a single point (length `d`) over all outputs — mean
    /// (length `m`) and *unclamped* output covariance (`m × m`) — together
    /// with their derivatives with respect to the point (a
    /// [`JointPosteriorGrad`] with `q = 1`).
    ///
    /// The default implementation calls [`Model::joint_posterior_with_grad`].
    ///
    /// # Errors
    /// See [`Model::posterior`].
    fn point_posterior_with_grad(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        let g = self.joint_posterior_with_grad(&Matrix::from_row_slice(1, x.len(), x))?;
        check_point_posterior(&g.mean, self.num_outputs())?;
        Ok(g)
    }

    /// Marginal posterior means and variances of every row of `x` evaluated
    /// independently (the posterior of a `b × 1 × d` batch in BoTorch) for a
    /// *single-output* model. Variances are clamped from below like
    /// `posterior.variance` (except for deterministic models, see
    /// [`Model::is_deterministic`]).
    ///
    /// The default implementation uses [`Model::point_posteriors`].
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models, otherwise see
    /// [`Model::posterior`].
    fn posterior_marginals(&self, x: &Matrix) -> Result<(Vector, Vector)> {
        require_single_output(self.num_outputs())?;
        let pp = self.point_posteriors(x)?;
        let b = pp.len();
        let deterministic = self.is_deterministic();
        Ok((
            Vector::from_fn(b, |i, _| pp.mean[(i, 0)]),
            Vector::from_fn(b, |i, _| {
                let v = pp.covariance[(i, 0)];
                if deterministic {
                    v
                } else {
                    clamp_variance(v)
                }
            }),
        ))
    }

    /// Posterior mean and (clamped) variance of a *single-output* model at a
    /// single point (length `d`) together with their gradients with respect
    /// to the point: `(mean, variance, ∂mean/∂x, ∂variance/∂x)`. The gradient
    /// of a clamped variance is zero; deterministic models are not clamped.
    ///
    /// The default implementation uses [`Model::point_posterior_with_grad`].
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models, otherwise see
    /// [`Model::posterior`].
    fn posterior_q1_with_grad(&self, x: &[f64]) -> Result<(f64, f64, Vector, Vector)> {
        require_single_output(self.num_outputs())?;
        let g = self.point_posterior_with_grad(x)?;
        let d = x.len();
        let mean = g.mean[0];
        let var = g.covariance[(0, 0)];
        let dmean = Vector::from_fn(d, |k, _| g.dmean[k][0]);
        if var < MIN_VARIANCE && !self.is_deterministic() {
            return Ok((mean, MIN_VARIANCE, dmean, Vector::zeros(d)));
        }
        let dvar = Vector::from_fn(d, |k, _| g.dcov[k][(0, 0)]);
        Ok((mean, var, dmean, dvar))
    }
}

/// BoTorch's variance clamp (`gpytorch.settings.min_variance`).
#[inline]
pub(crate) fn clamp_variance(v: f64) -> f64 {
    if v < MIN_VARIANCE {
        MIN_VARIANCE
    } else {
        v
    }
}

fn require_single_output(m: usize) -> Result<()> {
    if m != 1 {
        return Err(BotorchError::Unsupported(format!(
            "this method requires a single-output model, got {m} outputs"
        )));
    }
    Ok(())
}

fn check_point_posterior(mean: &Vector, m: usize) -> Result<()> {
    if mean.len() != m {
        return Err(BotorchError::DimensionMismatch(format!(
            "expected a posterior over {m} outputs at one point, got {} values",
            mean.len()
        )));
    }
    Ok(())
}

/// Validates the arguments of [`Model::joint_posterior_vjp`] for a model with
/// `num_outputs` outputs and one posterior point per input (the models of
/// this crate use [`check_vjp_shapes_expanded`]).
#[allow(dead_code)]
pub(crate) fn check_vjp_shapes(
    x: &Matrix,
    rows: &Range<usize>,
    mean_adjoint: &Vector,
    covariance_adjoint: &Matrix,
    num_outputs: usize,
) -> Result<()> {
    check_vjp_shapes_expanded(x, rows, mean_adjoint, covariance_adjoint, num_outputs, 1)
}

/// [`check_vjp_shapes`] for a model with `points_per_input` posterior points
/// per input (see [`Model::points_per_input`]).
pub(crate) fn check_vjp_shapes_expanded(
    x: &Matrix,
    rows: &Range<usize>,
    mean_adjoint: &Vector,
    covariance_adjoint: &Matrix,
    num_outputs: usize,
    points_per_input: usize,
) -> Result<()> {
    let n = x.nrows();
    if rows.start > rows.end || rows.end > n {
        return Err(BotorchError::DimensionMismatch(format!(
            "rows {rows:?} out of range for {n} points"
        )));
    }
    let len = n * points_per_input * num_outputs;
    if mean_adjoint.len() != len || covariance_adjoint.shape() != (len, len) {
        return Err(BotorchError::DimensionMismatch(format!(
            "adjoints must have shapes {len} and {len}x{len}, got {} and {}x{}",
            mean_adjoint.len(),
            covariance_adjoint.nrows(),
            covariance_adjoint.ncols()
        )));
    }
    Ok(())
}
