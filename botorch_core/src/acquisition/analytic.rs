//! Analytic (`q = 1`) acquisition functions mirroring
//! `botorch.acquisition.analytic`.
//!
//! All of them evaluate the marginal posterior `N(μ(x), σ²(x))` of a
//! single-output model at a single point, with
//! `σ = sqrt(max(variance, 1e-12))` (BoTorch's `_mean_and_sigma`, where the
//! variance is already clamped at `1e-10` by the model), and the scaled
//! improvement `u = (μ - best_f) / σ` (negated when `maximize` is false).
//! Gradients are exact (chain rule through
//! [`Model::posterior_q1_with_grad`]).
//!
//! Multi-output models are used through a posterior transform
//! ([`crate::acquisition::ScalarizedModel`]). The constrained acquisition
//! functions [`LogConstrainedExpectedImprovement`] and
//! [`ConstrainedExpectedImprovement`] read the per-output marginals of a
//! multi-output model directly.

use std::sync::Arc;

use rayon::prelude::*;

use crate::acquisition::{check_candidates, check_model, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::utils::probability::{
    erfcx, log_ndtr, log_ndtr_grad, log_phi, log_prob_normal_in, log_prob_normal_in_with_grad,
    ndtr, phi,
};
use crate::utils::safe_math::log1mexp;
use crate::{Matrix, Vector};

/// Lower clamp of the variance in BoTorch's `_mean_and_sigma`.
pub const MIN_VAR: f64 = 1e-12;

/// `-(2 ** -0.5)` (`_neg_inv_sqrt2` in `botorch.acquisition.analytic`; the same
/// double as `-FRAC_1_SQRT_2`).
const NEG_INV_SQRT2: f64 = -std::f64::consts::FRAC_1_SQRT_2;
/// `math.log(math.pi / 2) / 2` (`_log_sqrt_pi_div_2`).
const LOG_SQRT_PI_DIV_2: f64 = 0.2257913526447274;
/// Branch point of [`log_ei_helper`].
const LOG_EI_BOUND: f64 = -1.0;
/// `-1 / sqrt(eps)` for double precision, the second branch point of
/// [`log_ei_helper`].
const NEG_INV_SQRT_EPS: f64 = -1e6;

/// Points evaluated per parallel task in [`AcquisitionFunction::forward_many`].
const MARGINAL_CHUNK: usize = 64;

/// `φ(u) + u Φ(u)`, the expected improvement of a standard normal over `-u`
/// (BoTorch's `_ei_helper`).
#[inline]
pub fn ei_helper(u: f64) -> f64 {
    phi(u) + u * ndtr(u)
}

/// `log(|u| Φ(u) / φ(u))` for `u < 0`, computed as
/// `log(erfcx(-u/√2) |u|) + log(√(π/2))` (BoTorch's `_log_abs_u_Phi_div_phi`).
#[inline]
fn log_abs_u_phi_div_phi(u: f64) -> f64 {
    (erfcx(NEG_INV_SQRT2 * u) * u.abs()).ln() + LOG_SQRT_PI_DIV_2
}

/// `D(t) = 1 / (√π erfcx(t)) - t` for `t > 0` without cancellation: directly
/// for `t ≤ 4` (relative error below 5e-15) and otherwise by the continued
/// fraction `D = (1/2) / (t + 1 / (t + (3/2) / (t + 2 / (t + ...))))` truncated
/// after 40 terms (Laplace's continued fraction of `erfc`), which is accurate
/// to machine precision there.
#[inline]
fn erfcx_mills_gap(t: f64) -> f64 {
    const SQRT_PI: f64 = 1.772_453_850_905_516;
    if t <= 4.0 {
        1.0 / (SQRT_PI * erfcx(t)) - t
    } else {
        let mut k = 0.0;
        for j in (1..=40).rev() {
            k = (f64::from(j) / 2.0) / (t + k);
        }
        k
    }
}

/// `log(φ(u) + u Φ(u))`, accurate for `u ∈ [-1e100, 1e100]` (BoTorch's
/// `_log_ei_helper`): the naive formula for `u > -1`, otherwise
/// `log φ(u) + log1mexp(log(|u| Φ(u) / φ(u)))`, and the leading-order
/// asymptotics `log φ(u) - 2 log|u|` for `u < -1e6`.
pub fn log_ei_helper(u: f64) -> f64 {
    if u > LOG_EI_BOUND {
        return ei_helper(u).ln();
    }
    if u > NEG_INV_SQRT_EPS {
        log_phi(u) + log1mexp(log_abs_u_phi_div_phi(u))
    } else {
        log_phi(u) + -2.0 * u.abs().ln()
    }
}

/// Derivative of [`log_ei_helper`], `Φ(u) / (φ(u) + u Φ(u))`, accurate to a
/// few ULPs for all `u`.
///
/// For `u ≤ -1` it is evaluated as `1 / (√2 D(t))` with `t = -u/√2` and
/// `D(t) = 1/(√π erfcx(t)) - t`, which is evaluated without cancellation by a
/// continued fraction for large `t`; for `u < -1e6` it is
/// the derivative `-u - 2/u` of the asymptotic expression used for the value.
/// PyTorch's autograd of BoTorch's expression suffers from cancellation in the
/// derivative of `log1mexp(log(|u| Φ(u) / φ(u)))` and has a relative error of
/// roughly `ε u²` (1.5e-10 at `u = -10³`, 4e-7 at `u = -10⁵`); this function is
/// exact to machine precision there.
pub fn log_ei_helper_grad(u: f64) -> f64 {
    if u > LOG_EI_BOUND {
        return ndtr(u) / ei_helper(u);
    }
    if u > NEG_INV_SQRT_EPS {
        1.0 / (std::f64::consts::SQRT_2 * erfcx_mills_gap(NEG_INV_SQRT2 * u))
    } else {
        -u + -2.0 * (1.0 / u.abs()) * u.signum()
    }
}

/// Posterior mean and standard deviation at a single point.
fn mean_sigma(model: &dyn Model, x: &Matrix) -> Result<(f64, f64)> {
    let (mean, var) = model.posterior_marginals(x)?;
    Ok((mean[0], var[0].max(MIN_VAR).sqrt()))
}

/// Posterior mean and standard deviation with their gradients.
fn mean_sigma_with_grad(model: &dyn Model, x: &Matrix) -> Result<(f64, f64, Vector, Vector)> {
    let point: Vec<f64> = x.row(0).iter().copied().collect();
    let (mean, var, dmean, dvar) = model.posterior_q1_with_grad(&point)?;
    let sigma = var.max(MIN_VAR).sqrt();
    let dsigma = if var >= MIN_VAR {
        dvar / (2.0 * sigma)
    } else {
        Vector::zeros(point.len())
    };
    Ok((mean, sigma, dmean, dsigma))
}

/// Scaled improvement `u = (μ - best_f) / σ` (negated for minimization) and
/// its gradient.
fn scaled_improvement(
    mean: f64,
    sigma: f64,
    dmean: &Vector,
    dsigma: &Vector,
    best_f: f64,
    maximize: bool,
) -> (f64, Vector) {
    let diff = mean - best_f;
    let u = diff / sigma;
    let du = dmean / sigma - dsigma * (diff / (sigma * sigma));
    if maximize {
        (u, du)
    } else {
        (-u, -du)
    }
}

fn row_matrix(g: Vector) -> Matrix {
    Matrix::from_row_slice(1, g.len(), g.as_slice())
}

fn check_single_output(model: &dyn Model) -> Result<()> {
    check_model(
        model,
        "Must specify a posterior transform when using a multi-output model.",
    )
}

fn check_best_f(best_f: f64) -> Result<()> {
    if best_f.is_nan() {
        return Err(BotorchError::InvalidArgument(
            "best_f must not be NaN".into(),
        ));
    }
    Ok(())
}

/// Evaluates `value(mean, sigma)` for a batch of `1 × d` candidate sets in
/// parallel chunks through the model's batched marginals.
fn marginal_batch<F>(model: &dyn Model, xs: &[Matrix], value: F) -> Result<Vec<f64>>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let d = model.dim();
    for x in xs {
        check_candidates(x, d, Some(1))?;
    }
    let chunks: Vec<Result<Vec<f64>>> = xs
        .par_chunks(MARGINAL_CHUNK)
        .map(|chunk| {
            let stacked = Matrix::from_fn(chunk.len(), d, |i, j| chunk[i][(0, j)]);
            let (mean, var) = model.posterior_marginals(&stacked)?;
            Ok(mean
                .iter()
                .zip(var.iter())
                .map(|(&m, &v)| value(m, v.max(MIN_VAR).sqrt()))
                .collect())
        })
        .collect();
    let mut out = Vec::with_capacity(xs.len());
    for c in chunks {
        out.extend(c?);
    }
    Ok(out)
}

macro_rules! analytic_common {
    () => {
        fn model(&self) -> &Arc<dyn Model> {
            &self.model
        }

        fn expected_q(&self) -> Option<usize> {
            Some(1)
        }

        fn forward(&self, x: &Matrix) -> Result<f64> {
            check_candidates(x, self.model.dim(), Some(1))?;
            let (mean, sigma) = mean_sigma(self.model.as_ref(), x)?;
            Ok(self.value(mean, sigma))
        }

        fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
            marginal_batch(self.model.as_ref(), xs, |m, s| self.value(m, s))
        }
    };
}

/// Single-outcome analytic Expected Improvement,
/// `EI(x) = E[max(f(x) - best_f, 0)] = σ (φ(u) + u Φ(u))`
/// (BoTorch's `ExpectedImprovement`). BoTorch recommends
/// [`LogExpectedImprovement`] for better numerics.
#[derive(Debug, Clone)]
pub struct ExpectedImprovement {
    model: Arc<dyn Model>,
    best_f: f64,
    maximize: bool,
}

impl ExpectedImprovement {
    /// Creates `ExpectedImprovement(model, best_f, maximize=maximize)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models and
    /// [`BotorchError::InvalidArgument`] for a NaN `best_f`.
    pub fn new(model: Arc<dyn Model>, best_f: f64, maximize: bool) -> Result<Self> {
        check_single_output(model.as_ref())?;
        check_best_f(best_f)?;
        Ok(Self {
            model,
            best_f,
            maximize,
        })
    }

    /// The incumbent value `best_f`.
    pub fn best_f(&self) -> f64 {
        self.best_f
    }

    /// Whether the problem is a maximization problem.
    pub fn maximize(&self) -> bool {
        self.maximize
    }

    fn value(&self, mean: f64, sigma: f64) -> f64 {
        let u = (mean - self.best_f) / sigma;
        let u = if self.maximize { u } else { -u };
        sigma * ei_helper(u)
    }
}

impl AcquisitionFunction for ExpectedImprovement {
    analytic_common!();

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        check_candidates(x, self.model.dim(), Some(1))?;
        let (mean, sigma, dmean, dsigma) = mean_sigma_with_grad(self.model.as_ref(), x)?;
        let (u, du) = scaled_improvement(mean, sigma, &dmean, &dsigma, self.best_f, self.maximize);
        let h = ei_helper(u);
        let grad = &dsigma * h + du * (sigma * ndtr(u));
        Ok((sigma * h, row_matrix(grad)))
    }

    fn is_nonnegative(&self) -> bool {
        true
    }
}

/// Single-outcome analytic Log Expected Improvement,
/// `log EI(x) = log(φ(u) + u Φ(u)) + log σ`, computed with BoTorch's
/// numerically robust `_log_ei_helper` (BoTorch's `LogExpectedImprovement`,
/// Ament et al. 2023).
#[derive(Debug, Clone)]
pub struct LogExpectedImprovement {
    model: Arc<dyn Model>,
    best_f: f64,
    maximize: bool,
}

impl LogExpectedImprovement {
    /// Creates `LogExpectedImprovement(model, best_f, maximize=maximize)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models and
    /// [`BotorchError::InvalidArgument`] for a NaN `best_f`.
    pub fn new(model: Arc<dyn Model>, best_f: f64, maximize: bool) -> Result<Self> {
        check_single_output(model.as_ref())?;
        check_best_f(best_f)?;
        Ok(Self {
            model,
            best_f,
            maximize,
        })
    }

    /// The incumbent value `best_f`.
    pub fn best_f(&self) -> f64 {
        self.best_f
    }

    /// Whether the problem is a maximization problem.
    pub fn maximize(&self) -> bool {
        self.maximize
    }

    fn value(&self, mean: f64, sigma: f64) -> f64 {
        let u = (mean - self.best_f) / sigma;
        let u = if self.maximize { u } else { -u };
        log_ei_helper(u) + sigma.ln()
    }
}

impl AcquisitionFunction for LogExpectedImprovement {
    analytic_common!();

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        check_candidates(x, self.model.dim(), Some(1))?;
        let (mean, sigma, dmean, dsigma) = mean_sigma_with_grad(self.model.as_ref(), x)?;
        let (u, du) = scaled_improvement(mean, sigma, &dmean, &dsigma, self.best_f, self.maximize);
        let grad = du * log_ei_helper_grad(u) + dsigma / sigma;
        Ok((log_ei_helper(u) + sigma.ln(), row_matrix(grad)))
    }

    fn is_log(&self) -> bool {
        true
    }
}

/// Single-outcome analytic Probability of Improvement, `PI(x) = Φ(u)`
/// (BoTorch's `ProbabilityOfImprovement`).
#[derive(Debug, Clone)]
pub struct ProbabilityOfImprovement {
    model: Arc<dyn Model>,
    best_f: f64,
    maximize: bool,
}

impl ProbabilityOfImprovement {
    /// Creates `ProbabilityOfImprovement(model, best_f, maximize=maximize)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models and
    /// [`BotorchError::InvalidArgument`] for a NaN `best_f`.
    pub fn new(model: Arc<dyn Model>, best_f: f64, maximize: bool) -> Result<Self> {
        check_single_output(model.as_ref())?;
        check_best_f(best_f)?;
        Ok(Self {
            model,
            best_f,
            maximize,
        })
    }

    fn value(&self, mean: f64, sigma: f64) -> f64 {
        let u = (mean - self.best_f) / sigma;
        ndtr(if self.maximize { u } else { -u })
    }
}

impl AcquisitionFunction for ProbabilityOfImprovement {
    analytic_common!();

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        check_candidates(x, self.model.dim(), Some(1))?;
        let (mean, sigma, dmean, dsigma) = mean_sigma_with_grad(self.model.as_ref(), x)?;
        let (u, du) = scaled_improvement(mean, sigma, &dmean, &dsigma, self.best_f, self.maximize);
        Ok((ndtr(u), row_matrix(du * phi(u))))
    }

    fn is_nonnegative(&self) -> bool {
        true
    }
}

/// Single-outcome analytic Log Probability of Improvement, `log Φ(u)`
/// computed with BoTorch's stable `log_ndtr` (BoTorch's
/// `LogProbabilityOfImprovement`).
#[derive(Debug, Clone)]
pub struct LogProbabilityOfImprovement {
    model: Arc<dyn Model>,
    best_f: f64,
    maximize: bool,
}

impl LogProbabilityOfImprovement {
    /// Creates `LogProbabilityOfImprovement(model, best_f, maximize=maximize)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models and
    /// [`BotorchError::InvalidArgument`] for a NaN `best_f`.
    pub fn new(model: Arc<dyn Model>, best_f: f64, maximize: bool) -> Result<Self> {
        check_single_output(model.as_ref())?;
        check_best_f(best_f)?;
        Ok(Self {
            model,
            best_f,
            maximize,
        })
    }

    fn value(&self, mean: f64, sigma: f64) -> f64 {
        let u = (mean - self.best_f) / sigma;
        log_ndtr(if self.maximize { u } else { -u })
    }
}

impl AcquisitionFunction for LogProbabilityOfImprovement {
    analytic_common!();

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        check_candidates(x, self.model.dim(), Some(1))?;
        let (mean, sigma, dmean, dsigma) = mean_sigma_with_grad(self.model.as_ref(), x)?;
        let (u, du) = scaled_improvement(mean, sigma, &dmean, &dsigma, self.best_f, self.maximize);
        Ok((log_ndtr(u), row_matrix(du * log_ndtr_grad(u))))
    }

    fn is_log(&self) -> bool {
        true
    }
}

/// Single-outcome Upper Confidence Bound, `UCB(x) = μ(x) + √β σ(x)` (with
/// `-μ` when minimizing) (BoTorch's `UpperConfidenceBound`).
#[derive(Debug, Clone)]
pub struct UpperConfidenceBound {
    model: Arc<dyn Model>,
    beta: f64,
    maximize: bool,
}

impl UpperConfidenceBound {
    /// Creates `UpperConfidenceBound(model, beta, maximize=maximize)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models and
    /// [`BotorchError::InvalidArgument`] unless `beta` is finite and
    /// non-negative (BoTorch would silently produce NaN values).
    pub fn new(model: Arc<dyn Model>, beta: f64, maximize: bool) -> Result<Self> {
        check_single_output(model.as_ref())?;
        if !(beta >= 0.0 && beta.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "beta must be finite and non-negative, got {beta}"
            )));
        }
        Ok(Self {
            model,
            beta,
            maximize,
        })
    }

    /// The trade-off parameter `beta`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    fn value(&self, mean: f64, sigma: f64) -> f64 {
        (if self.maximize { mean } else { -mean }) + self.beta.sqrt() * sigma
    }
}

impl AcquisitionFunction for UpperConfidenceBound {
    analytic_common!();

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        check_candidates(x, self.model.dim(), Some(1))?;
        let (mean, sigma, dmean, dsigma) = mean_sigma_with_grad(self.model.as_ref(), x)?;
        let sb = self.beta.sqrt();
        let dm = if self.maximize { dmean } else { -dmean };
        Ok((self.value(mean, sigma), row_matrix(dm + dsigma * sb)))
    }
}

/// Single-outcome posterior mean (negated when minimizing) (BoTorch's
/// `PosteriorMean`).
#[derive(Debug, Clone)]
pub struct PosteriorMean {
    model: Arc<dyn Model>,
    maximize: bool,
}

impl PosteriorMean {
    /// Creates `PosteriorMean(model, maximize=maximize)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models.
    pub fn new(model: Arc<dyn Model>, maximize: bool) -> Result<Self> {
        check_single_output(model.as_ref())?;
        Ok(Self { model, maximize })
    }

    fn value(&self, mean: f64, _sigma: f64) -> f64 {
        if self.maximize {
            mean
        } else {
            -mean
        }
    }
}

impl AcquisitionFunction for PosteriorMean {
    analytic_common!();

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        check_candidates(x, self.model.dim(), Some(1))?;
        let (mean, sigma, dmean, _) = mean_sigma_with_grad(self.model.as_ref(), x)?;
        let dm = if self.maximize { dmean } else { -dmean };
        Ok((self.value(mean, sigma), row_matrix(dm)))
    }
}

/// Single-outcome posterior standard deviation (negated when minimizing), an
/// acquisition function for pure exploration (BoTorch's
/// `PosteriorStandardDeviation`).
#[derive(Debug, Clone)]
pub struct PosteriorStandardDeviation {
    model: Arc<dyn Model>,
    maximize: bool,
}

impl PosteriorStandardDeviation {
    /// Creates `PosteriorStandardDeviation(model, maximize=maximize)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models.
    pub fn new(model: Arc<dyn Model>, maximize: bool) -> Result<Self> {
        check_single_output(model.as_ref())?;
        Ok(Self { model, maximize })
    }

    fn value(&self, _mean: f64, sigma: f64) -> f64 {
        if self.maximize {
            sigma
        } else {
            -sigma
        }
    }
}

impl AcquisitionFunction for PosteriorStandardDeviation {
    analytic_common!();

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        check_candidates(x, self.model.dim(), Some(1))?;
        let (mean, sigma, _, dsigma) = mean_sigma_with_grad(self.model.as_ref(), x)?;
        let ds = if self.maximize { dsigma } else { -dsigma };
        Ok((self.value(mean, sigma), row_matrix(ds)))
    }
}

/// An output constraint `lower <= y_index <= upper` of the constrained
/// analytic acquisition functions (one entry `{index: (lower, upper)}` of
/// BoTorch's `constraints` dictionary; `None` is an unbounded side).
pub type OutputConstraint = (usize, Option<f64>, Option<f64>);

/// The preprocessed constraint bounds (BoTorch's
/// `_preprocess_constraint_bounds`): constraints with only a lower bound, only
/// an upper bound and both bounds, each in insertion order.
#[derive(Clone, Debug, PartialEq)]
struct ConstraintBounds {
    lower: Vec<(usize, f64)>,
    upper: Vec<(usize, f64)>,
    both: Vec<(usize, f64, f64)>,
}

impl ConstraintBounds {
    fn new(
        constraints: &[OutputConstraint],
        objective_index: Option<usize>,
        num_outputs: usize,
    ) -> Result<Self> {
        if constraints.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "There must be at least one constraint.".into(),
            ));
        }
        let mut out = Self {
            lower: Vec::new(),
            upper: Vec::new(),
            both: Vec::new(),
        };
        let mut seen = Vec::with_capacity(constraints.len());
        for &(k, lo, hi) in constraints {
            if Some(k) == objective_index {
                return Err(BotorchError::InvalidArgument(
                    "Output corresponding to objective should not be a constraint.".into(),
                ));
            }
            if k >= num_outputs {
                return Err(BotorchError::InvalidArgument(format!(
                    "constraint index {k} is out of range for a model with {num_outputs} outputs"
                )));
            }
            if seen.contains(&k) {
                return Err(BotorchError::InvalidArgument(format!(
                    "output {k} is constrained more than once"
                )));
            }
            seen.push(k);
            if lo.is_some_and(f64::is_nan) || hi.is_some_and(f64::is_nan) {
                return Err(BotorchError::InvalidArgument(format!(
                    "the bounds of output {k} must not be NaN"
                )));
            }
            match (lo, hi) {
                (Some(l), Some(u)) => {
                    if u <= l {
                        return Err(BotorchError::InvalidArgument(
                            "Upper bound is less than the lower bound.".into(),
                        ));
                    }
                    out.both.push((k, l, u));
                }
                (Some(l), None) => out.lower.push((k, l)),
                (None, Some(u)) => out.upper.push((k, u)),
                (None, None) => {}
            }
        }
        Ok(out)
    }

    /// `Σ_i log P(lower_i <= y_i <= upper_i)` for independent Gaussian
    /// outputs (BoTorch's `compute_log_prob_feas_from_bounds`).
    fn log_prob_feas(&self, means: &[f64], sigmas: &[f64]) -> f64 {
        let mut total = 0.0;
        if !self.lower.is_empty() {
            let mut s = 0.0;
            for &(k, l) in &self.lower {
                s += log_ndtr(-((l - means[k]) / sigmas[k]));
            }
            total += s;
        }
        if !self.upper.is_empty() {
            let mut s = 0.0;
            for &(k, u) in &self.upper {
                s += log_ndtr((u - means[k]) / sigmas[k]);
            }
            total += s;
        }
        if !self.both.is_empty() {
            let mut s = 0.0;
            for &(k, l, u) in &self.both {
                let a = (l - means[k]) / sigmas[k];
                let b = (u - means[k]) / sigmas[k];
                s += log_prob_normal_in(a, b).unwrap_or(f64::NAN);
            }
            total += s;
        }
        total
    }

    /// [`ConstraintBounds::log_prob_feas`] and its gradient, given the
    /// gradients of the means and standard deviations.
    fn log_prob_feas_with_grad(&self, mo: &OutputMoments) -> (f64, Vector) {
        let d = mo.dmean.first().map_or(0, |v| v.len());
        let mut grad = Vector::zeros(d);
        // ∂z/∂x for z = (c - μ)/σ: -(∂μ + z ∂σ)/σ.
        let dz = |k: usize, z: f64| -(&mo.dmean[k] + &mo.dsigma[k] * z) / mo.sigma[k];
        let mut total = 0.0;
        if !self.lower.is_empty() {
            let mut s = 0.0;
            for &(k, l) in &self.lower {
                let z = (l - mo.mean[k]) / mo.sigma[k];
                s += log_ndtr(-z);
                grad -= dz(k, z) * log_ndtr_grad(-z);
            }
            total += s;
        }
        if !self.upper.is_empty() {
            let mut s = 0.0;
            for &(k, u) in &self.upper {
                let z = (u - mo.mean[k]) / mo.sigma[k];
                s += log_ndtr(z);
                grad += dz(k, z) * log_ndtr_grad(z);
            }
            total += s;
        }
        if !self.both.is_empty() {
            let mut s = 0.0;
            for &(k, l, u) in &self.both {
                let a = (l - mo.mean[k]) / mo.sigma[k];
                let b = (u - mo.mean[k]) / mo.sigma[k];
                match log_prob_normal_in_with_grad(a, b) {
                    Ok((v, ga, gb)) => {
                        s += v;
                        grad += dz(k, a) * ga + dz(k, b) * gb;
                    }
                    Err(_) => {
                        s += f64::NAN;
                        grad.fill(f64::NAN);
                    }
                }
            }
            total += s;
        }
        (total, grad)
    }
}

/// Per-output posterior means and standard deviations at a single point
/// (BoTorch's `_mean_and_sigma` of a multi-output posterior: every variance
/// clamped at `1e-10`, then at `1e-12`), with gradients.
struct OutputMoments {
    mean: Vec<f64>,
    sigma: Vec<f64>,
    dmean: Vec<Vector>,
    dsigma: Vec<Vector>,
}

/// The lower clamp of the model's variances before `_mean_and_sigma`'s
/// (GPyTorch's `min_variance`; none for deterministic models, whose
/// `EnsemblePosterior` returns zero variances).
fn variance_floor(model: &dyn Model) -> f64 {
    if model.is_deterministic() {
        0.0
    } else {
        crate::posterior::MIN_VARIANCE
    }
}

/// The standard deviation of a variance, clamped like `_mean_and_sigma`
/// (after the model's clamp at `floor`).
#[inline]
fn clamped_sigma(var: f64, floor: f64) -> f64 {
    let v = if var < floor { floor } else { var };
    v.max(MIN_VAR).sqrt()
}

fn output_moments(model: &dyn Model, x: &Matrix) -> Result<OutputMoments> {
    let point: Vec<f64> = x.row(0).iter().copied().collect();
    let g = model.point_posterior_with_grad(&point)?;
    let m = model.num_outputs();
    let d = point.len();
    let floor = variance_floor(model);
    let mut out = OutputMoments {
        mean: Vec::with_capacity(m),
        sigma: Vec::with_capacity(m),
        dmean: Vec::with_capacity(m),
        dsigma: Vec::with_capacity(m),
    };
    for t in 0..m {
        let var = g.covariance[(t, t)];
        let sigma = clamped_sigma(var, floor);
        out.mean.push(g.mean[t]);
        out.sigma.push(sigma);
        out.dmean.push(Vector::from_fn(d, |k, _| g.dmean[k][t]));
        out.dsigma.push(if var >= floor.max(MIN_VAR) {
            Vector::from_fn(d, |k, _| g.dcov[k][(t, t)] / (2.0 * sigma))
        } else {
            Vector::zeros(d)
        });
    }
    Ok(out)
}

/// Means and clamped standard deviations of every output at every row of
/// `xs` (each a `1 × d` candidate set), evaluated in parallel chunks.
fn output_moments_batch<F>(model: &dyn Model, xs: &[Matrix], value: F) -> Result<Vec<f64>>
where
    F: Fn(&[f64], &[f64]) -> f64 + Sync,
{
    let d = model.dim();
    for x in xs {
        check_candidates(x, d, Some(1))?;
    }
    let m = model.num_outputs();
    let floor = variance_floor(model);
    let chunks: Vec<Result<Vec<f64>>> = xs
        .par_chunks(MARGINAL_CHUNK)
        .map(|chunk| {
            let stacked = Matrix::from_fn(chunk.len(), d, |i, j| chunk[i][(0, j)]);
            let pp = model.point_posteriors(&stacked)?;
            Ok((0..chunk.len())
                .map(|i| {
                    let mean: Vec<f64> = (0..m).map(|t| pp.mean[(i, t)]).collect();
                    let sigma: Vec<f64> = (0..m)
                        .map(|t| clamped_sigma(pp.variance(i, t), floor))
                        .collect();
                    value(&mean, &sigma)
                })
                .collect())
        })
        .collect();
    let mut out = Vec::with_capacity(xs.len());
    for c in chunks {
        out.extend(c?);
    }
    Ok(out)
}

/// Shared state of the constrained analytic acquisition functions.
#[derive(Debug, Clone)]
struct ConstrainedCore {
    model: Arc<dyn Model>,
    best_f: f64,
    objective_index: usize,
    bounds: ConstraintBounds,
    constraints: Vec<OutputConstraint>,
    maximize: bool,
}

impl ConstrainedCore {
    fn new(
        model: Arc<dyn Model>,
        best_f: f64,
        objective_index: usize,
        constraints: Vec<OutputConstraint>,
        maximize: bool,
    ) -> Result<Self> {
        crate::acquisition::check_unbatched(model.as_ref())?;
        check_best_f(best_f)?;
        let m = model.num_outputs();
        let bounds = ConstraintBounds::new(&constraints, Some(objective_index), m)?;
        if objective_index >= m {
            return Err(BotorchError::InvalidArgument(format!(
                "objective_index {objective_index} is out of range for a model with {m} outputs"
            )));
        }
        Ok(Self {
            model,
            best_f,
            objective_index,
            bounds,
            constraints,
            maximize,
        })
    }

    /// The scaled improvement of the objective.
    fn u(&self, mean: f64, sigma: f64) -> f64 {
        let u = (mean - self.best_f) / sigma;
        if self.maximize {
            u
        } else {
            -u
        }
    }
}

macro_rules! constrained_common {
    () => {
        /// The incumbent value `best_f`.
        pub fn best_f(&self) -> f64 {
            self.core.best_f
        }

        /// The index of the objective output.
        pub fn objective_index(&self) -> usize {
            self.core.objective_index
        }

        /// The output constraints `(index, lower, upper)` in the given order.
        pub fn constraints(&self) -> &[OutputConstraint] {
            &self.core.constraints
        }

        /// Whether the problem is a maximization problem.
        pub fn maximize(&self) -> bool {
            self.core.maximize
        }
    };
}

/// Log Constrained Expected Improvement (BoTorch's
/// `LogConstrainedExpectedImprovement`, Ament et al. 2023):
/// `log EI(x) + Σ_i log P(lower_i <= y_i(x) <= upper_i)` for the objective
/// output `objective_index` and independent Gaussian constraint outputs,
/// with BoTorch's numerically robust `_log_ei_helper` and `log_ndtr` /
/// `log_prob_normal_in`.
#[derive(Debug, Clone)]
pub struct LogConstrainedExpectedImprovement {
    core: ConstrainedCore,
}

impl LogConstrainedExpectedImprovement {
    /// Creates `LogConstrainedExpectedImprovement(model, best_f,
    /// objective_index, constraints, maximize)`; `constraints` lists
    /// `(output index, lower, upper)` with `None` for an unbounded side.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for no constraints, a constrained
    /// objective, an upper bound not above the lower bound, indices out of
    /// range, duplicates or a NaN `best_f`, and [`BotorchError::Unsupported`]
    /// for batched models.
    pub fn new(
        model: Arc<dyn Model>,
        best_f: f64,
        objective_index: usize,
        constraints: Vec<OutputConstraint>,
        maximize: bool,
    ) -> Result<Self> {
        Ok(Self {
            core: ConstrainedCore::new(model, best_f, objective_index, constraints, maximize)?,
        })
    }

    constrained_common!();

    fn value(&self, mean: &[f64], sigma: &[f64]) -> f64 {
        let c = &self.core;
        let k = c.objective_index;
        let u = c.u(mean[k], sigma[k]);
        log_ei_helper(u) + sigma[k].ln() + c.bounds.log_prob_feas(mean, sigma)
    }
}

impl AcquisitionFunction for LogConstrainedExpectedImprovement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.core.model
    }

    fn expected_q(&self) -> Option<usize> {
        Some(1)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.forward_many(std::slice::from_ref(x))?[0])
    }

    fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
        output_moments_batch(self.core.model.as_ref(), xs, |m, s| self.value(m, s))
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let c = &self.core;
        check_candidates(x, c.model.dim(), Some(1))?;
        let mo = output_moments(c.model.as_ref(), x)?;
        let k = c.objective_index;
        let (u, du) = scaled_improvement(
            mo.mean[k],
            mo.sigma[k],
            &mo.dmean[k],
            &mo.dsigma[k],
            c.best_f,
            c.maximize,
        );
        let (log_pf, dlog_pf) = c.bounds.log_prob_feas_with_grad(&mo);
        let value = log_ei_helper(u) + mo.sigma[k].ln() + log_pf;
        let grad = du * log_ei_helper_grad(u) + &mo.dsigma[k] / mo.sigma[k] + dlog_pf;
        Ok((value, row_matrix(grad)))
    }

    fn is_log(&self) -> bool {
        true
    }
}

/// Constrained Expected Improvement (BoTorch's
/// `ConstrainedExpectedImprovement`): `EI(x) · Π_i P(lower_i <= y_i(x) <=
/// upper_i)` for the objective output `objective_index` and independent
/// Gaussian constraint outputs. BoTorch recommends
/// [`LogConstrainedExpectedImprovement`] for better numerics.
#[derive(Debug, Clone)]
pub struct ConstrainedExpectedImprovement {
    core: ConstrainedCore,
}

impl ConstrainedExpectedImprovement {
    /// Creates `ConstrainedExpectedImprovement(model, best_f,
    /// objective_index, constraints, maximize)` (see
    /// [`LogConstrainedExpectedImprovement::new`]).
    ///
    /// # Errors
    /// See [`LogConstrainedExpectedImprovement::new`].
    pub fn new(
        model: Arc<dyn Model>,
        best_f: f64,
        objective_index: usize,
        constraints: Vec<OutputConstraint>,
        maximize: bool,
    ) -> Result<Self> {
        Ok(Self {
            core: ConstrainedCore::new(model, best_f, objective_index, constraints, maximize)?,
        })
    }

    constrained_common!();

    fn value(&self, mean: &[f64], sigma: &[f64]) -> f64 {
        let c = &self.core;
        let k = c.objective_index;
        let u = c.u(mean[k], sigma[k]);
        let ei = sigma[k] * ei_helper(u);
        ei * c.bounds.log_prob_feas(mean, sigma).exp()
    }
}

impl AcquisitionFunction for ConstrainedExpectedImprovement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.core.model
    }

    fn expected_q(&self) -> Option<usize> {
        Some(1)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.forward_many(std::slice::from_ref(x))?[0])
    }

    fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
        output_moments_batch(self.core.model.as_ref(), xs, |m, s| self.value(m, s))
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let c = &self.core;
        check_candidates(x, c.model.dim(), Some(1))?;
        let mo = output_moments(c.model.as_ref(), x)?;
        let k = c.objective_index;
        let (u, du) = scaled_improvement(
            mo.mean[k],
            mo.sigma[k],
            &mo.dmean[k],
            &mo.dsigma[k],
            c.best_f,
            c.maximize,
        );
        let h = ei_helper(u);
        let ei = mo.sigma[k] * h;
        let dei = &mo.dsigma[k] * h + du * (mo.sigma[k] * ndtr(u));
        let (log_pf, dlog_pf) = c.bounds.log_prob_feas_with_grad(&mo);
        let pf = log_pf.exp();
        let value = ei * pf;
        let grad = dei * pf + dlog_pf * (ei * pf);
        Ok((value, row_matrix(grad)))
    }

    fn is_nonnegative(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_ei_helper_matches_naive_formula_and_asymptotics() {
        for &u in &[-0.99, -0.5, 0.0, 1.0, 4.0] {
            assert!((log_ei_helper(u) - ei_helper(u).ln()).abs() < 1e-14);
        }
        // Lower branch agrees with the naive formula where the latter is accurate.
        for &u in &[-1.0, -2.0, -5.0] {
            let naive = ei_helper(u).ln();
            assert!(
                (log_ei_helper(u) - naive).abs() < 1e-12 * naive.abs(),
                "{u}"
            );
        }
        // Continuity at the branch points.
        let a = log_ei_helper(-1.0);
        let b = log_ei_helper(-1.0 + 1e-12);
        assert!((a - b).abs() < 1e-10);
        let a = log_ei_helper(-1e6);
        let b = log_ei_helper(-1e6 + 1e-6);
        assert!((a - b).abs() < 1e-10 * a.abs());
        // Asymptotics: log EI(u) ≈ log φ(u) - 2 log|u| for u → -∞.
        let u = -1e8;
        assert!((log_ei_helper(u) - (log_phi(u) - 2.0 * u.abs().ln())).abs() < 1e-6);
        assert!(log_ei_helper(-1e100).is_finite());
    }

    #[test]
    fn log_ei_helper_gradient() {
        for &u in &[-1e4, -40.0, -3.0, -1.0, -0.999, 0.0, 2.5, 30.0] {
            let h = 1e-6 * f64::max(1.0, f64::abs(u));
            let fd = (log_ei_helper(u + h) - log_ei_helper(u - h)) / (2.0 * h);
            let g = log_ei_helper_grad(u);
            assert!(
                (g - fd).abs() <= 1e-6 * g.abs().max(1.0),
                "{u}: {g} vs {fd}"
            );
            if u > -30.0 {
                assert!((g - ndtr(u) / ei_helper(u)).abs() <= 1e-10 * g.abs(), "{u}");
            }
        }
        // Far tail: gradient ≈ |u| + 2/|u|.
        let u = -1e7;
        assert!((log_ei_helper_grad(u) - (-u - 2.0 / u)).abs() < 1e-6);
        // Exact values Φ(u) / (φ(u) + u Φ(u)) (mpmath, 80 digits).
        let exact = [
            (-1.0, 1.9042712333296918),
            (-2.0, 2.679416883955586),
            (-6.0, 6.309_840_773_209_832),
            (-30.0, 30.066_446_154_162_42),
            (-150.0, 150.0133315561084),
            (-1000.0, 1000.001999994),
            (-3000.0, 3000.0006666664444),
            (-1e5, 100000.00002),
            (-999999.0, 999999.000002),
        ];
        for (u, g) in exact {
            let rel = (log_ei_helper_grad(u) - g).abs() / g;
            assert!(rel < 4e-15, "u = {u}: relative error {rel:e}");
        }
    }
}
