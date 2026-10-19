//! Preference acquisition functions (`botorch.acquisition.preference` and
//! `PairwiseMCPosteriorVariance` of `botorch.acquisition.active_learning`):
//!
//! * [`AnalyticExpectedUtilityOfBestOption`] (analytic EUBO, Lin et al. 2022)
//!   on a pairwise preference model, optionally composed with a
//!   deterministic outcome model and against a previous winner;
//! * qEUBO (Astudillo et al. 2023): `E[max_i f(Y(x_i))]`, BoTorch's
//!   `qExpectedUtilityOfBestOption`, is `qSimpleRegret` on the preference
//!   model seen through the outcome model ([`ComposedModel`]);
//! * [`PairwiseBayesianActiveLearningByDisagreement`] (BALD for pairs,
//!   Houlsby et al. 2011), with explicit standard normal draws (BoTorch draws
//!   them from torch's global generator at every evaluation);
//! * [`PairwiseMcPosteriorVariance`]: the posterior sample variance of an
//!   objective (a link function) of the differences of pairs of points.
//!
//! Values are BoTorch's expressions; gradients with respect to the
//! candidates are exact (reverse mode through the preference model's
//! posterior and the outcome model). Clamps follow torch ≥ 2.14 (no
//! gradient exactly at the bound).

use std::fmt;
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::acquisition::mc_objective::{McObjective, McSamples};
use crate::acquisition::monte_carlo::JointSamples;
use crate::acquisition::{check_candidates, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::sampling::SobolQMCNormalSampler;
use crate::{Matrix, Vector};

/// BoTorch's `SIGMA_JITTER`: the lower clamp of the variance of the
/// difference of two utilities.
pub const SIGMA_JITTER: f64 = 1e-8;
/// `math.log(math.sqrt(2 * math.pi))` (torch's `Normal.log_prob`).
const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// torch's `Normal(0, 1).cdf(u)`.
fn ncdf(u: f64) -> f64 {
    crate::models::likelihoods::pairwise::normal_cdf(u)
}

/// torch's `exp(Normal(0, 1).log_prob(u))`.
fn npdf(u: f64) -> f64 {
    (-(u * u) / 2.0 - LOG_SQRT_2PI).exp()
}

// ---------------------------------------------------------------------------
// The preference model seen through an outcome model
// ---------------------------------------------------------------------------

/// A preference model evaluated at the outputs of a deterministic outcome
/// model: the posterior at `X` is the preference model's posterior at
/// `Y = outcome_model(X)` (BoTorch's `pref_model.posterior(outcome_model(X))`
/// in the preference acquisition functions), with the chain rule for the
/// derivatives.
pub struct ComposedModel {
    pref: Arc<dyn Model>,
    outcome: Arc<dyn Model>,
}

impl fmt::Debug for ComposedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComposedModel").finish_non_exhaustive()
    }
}

impl ComposedModel {
    /// `pref ∘ outcome`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] unless the outcome model is
    /// deterministic, unbatched and has as many outputs as the preference
    /// model has inputs.
    pub fn new(pref: Arc<dyn Model>, outcome: Arc<dyn Model>) -> Result<Self> {
        if !outcome.is_deterministic() {
            return Err(BotorchError::Unsupported(
                "the outcome model of a preference acquisition function must be deterministic \
                 (e.g. a PosteriorMeanModel, FixedSingleSampleModel or \
                 GenericDeterministicModel)"
                    .into(),
            ));
        }
        if !outcome.batch_shape().is_empty() || !pref.batch_shape().is_empty() {
            return Err(BotorchError::Unsupported(
                "preference acquisition functions require unbatched models".into(),
            ));
        }
        if outcome.num_outputs() != pref.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "the outcome model has {} outputs, the preference model {} inputs",
                outcome.num_outputs(),
                pref.dim()
            )));
        }
        Ok(Self { pref, outcome })
    }
}

/// The outcomes `Y(x)` (`q × m`) of a deterministic outcome model and, with
/// `grad`, their derivatives (`dY[j]`, `q·m` output-major, for every input
/// coordinate `j = p·d + k`).
fn outcomes(outcome: &dyn Model, x: &Matrix, grad: bool) -> Result<(Matrix, Option<Vec<Vector>>)> {
    if grad {
        let g = outcome.joint_posterior_with_grad(x)?;
        let q = x.nrows();
        let m = g.num_outputs;
        let y = Matrix::from_fn(q, m, |p, t| g.mean[t * q + p]);
        Ok((y, Some(g.dmean)))
    } else {
        let p = outcome.posterior(x, false)?;
        Ok((p.mean_matrix(), None))
    }
}

/// The adjoint of `x` (`rows × d`) from the adjoint of the outcomes of its
/// first `rows` points (`ybar`, `rows × m`).
fn outcomes_backward(ybar: &Matrix, dy: &[Vector], q: usize, d: usize) -> Matrix {
    let m = ybar.ncols();
    let rows = ybar.nrows();
    Matrix::from_fn(q, d, |p, k| {
        let col = &dy[p * d + k];
        let mut s = 0.0;
        for pp in 0..rows {
            for t in 0..m {
                s += ybar[(pp, t)] * col[t * q + pp];
            }
        }
        s
    })
}

impl Model for ComposedModel {
    fn dim(&self) -> usize {
        self.outcome.dim()
    }

    fn num_outputs(&self) -> usize {
        self.pref.num_outputs()
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let (y, _) = outcomes(self.outcome.as_ref(), x, false)?;
        self.pref.posterior(&y, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let (q, d) = x.shape();
        let (y, dy) = outcomes(self.outcome.as_ref(), x, true)?;
        let dy = dy.expect("derivatives requested");
        let m = y.ncols();
        let g = self.pref.joint_posterior_with_grad(&y)?;
        let len = g.mean.len();
        let mut dmean = vec![Vector::zeros(len); q * d];
        let mut dcov = vec![Matrix::zeros(len, len); q * d];
        for p in 0..q {
            for k in 0..d {
                let j = p * d + k;
                for pp in 0..q {
                    for t in 0..m {
                        let f = dy[j][t * q + pp];
                        if f != 0.0 {
                            let jj = pp * m + t;
                            dmean[j].axpy(f, &g.dmean[jj], 1.0);
                            dcov[j] += &g.dcov[jj] * f;
                        }
                    }
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean: g.mean,
            covariance: g.covariance,
            dmean,
            dcov,
            num_outputs: g.num_outputs,
        })
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: std::ops::Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        let (q, d) = x.shape();
        let (y, dy) = outcomes(self.outcome.as_ref(), x, true)?;
        let ybar = self
            .pref
            .joint_posterior_vjp(&y, 0..q, mean_adjoint, covariance_adjoint)?;
        let g = outcomes_backward(&ybar, &dy.expect("derivatives requested"), q, d);
        Ok(g.rows(rows.start, rows.len()).into_owned())
    }
}

/// The preference model, composed with the outcome model if given.
///
/// # Errors
/// See [`ComposedModel::new`].
pub fn compose(pref: Arc<dyn Model>, outcome: Option<Arc<dyn Model>>) -> Result<Arc<dyn Model>> {
    match outcome {
        None => Ok(pref),
        Some(o) => Ok(Arc::new(ComposedModel::new(pref, o)?)),
    }
}

// ---------------------------------------------------------------------------
// The posterior of a pair
// ---------------------------------------------------------------------------

/// The inputs of the preference model for a candidate set `x`: `Y(x)`, and
/// the previous winner appended.
struct Pair {
    y: Matrix,
    dy: Option<Vec<Vector>>,
    post: Posterior,
}

fn pair_posterior(
    pref: &dyn Model,
    outcome: Option<&dyn Model>,
    winner: Option<&Vector>,
    x: &Matrix,
    grad: bool,
) -> Result<Pair> {
    let (mut y, dy) = match outcome {
        Some(o) => outcomes(o, x, grad)?,
        None => (x.clone(), None),
    };
    if let Some(w) = winner {
        let m = y.ncols();
        if w.len() != m {
            return Err(BotorchError::DimensionMismatch(format!(
                "the previous winner has {} values, the preference model {m} inputs",
                w.len()
            )));
        }
        let mut full = Matrix::zeros(y.nrows() + 1, m);
        full.rows_mut(0, y.nrows()).copy_from(&y);
        full.row_mut(y.nrows()).copy_from(&w.transpose());
        y = full;
    }
    let post = pref.posterior(&y, false)?;
    if post.num_points() != 2 || post.num_outputs() != 1 {
        return Err(BotorchError::Unsupported(
            "preference acquisition functions require a single-output preference model".into(),
        ));
    }
    Ok(Pair { y, dy, post })
}

/// The gradient with respect to `x` (`q × d`) of a function of the pair's
/// posterior with adjoints `mean_bar` (2) and `cov_bar` (2 × 2).
fn pair_backward(
    pref: &dyn Model,
    pair: &Pair,
    x: &Matrix,
    mean_bar: &Vector,
    cov_bar: &Matrix,
) -> Result<Matrix> {
    let (q, d) = x.shape();
    let ybar = pref.joint_posterior_vjp(&pair.y, 0..q, mean_bar, cov_bar)?;
    match &pair.dy {
        None => Ok(ybar),
        Some(dy) => Ok(outcomes_backward(&ybar, dy, q, d)),
    }
}

/// `w Σ w` for `w = [1, -1]` in torch's order (`(w @ Σ) @ w`).
fn diff_variance(cov: &Matrix) -> f64 {
    let a = cov[(0, 0)] - cov[(1, 0)];
    let b = cov[(0, 1)] - cov[(1, 1)];
    a - b
}

/// The adjoint of `Σ` from the adjoint of `w Σ w`.
fn diff_variance_bar(vbar: f64) -> Matrix {
    Matrix::from_row_slice(2, 2, &[vbar, -vbar, -vbar, vbar])
}

// ---------------------------------------------------------------------------
// Analytic EUBO
// ---------------------------------------------------------------------------

/// BoTorch's `AnalyticExpectedUtilityOfBestOption`: for the utilities
/// `(f₁, f₂)` of a pair (the candidate and the previous winner if given),
/// `σ (φ(u) + u Φ(u)) [+ μ₂]` with `δ = μ₁ - μ₂`, `σ = sqrt(max(Var(f₁ -
/// f₂), 1e-8))`, `u = δ / σ` (`+ μ₂` without a previous winner, i.e.
/// `E[max(f₁, f₂)]`).
#[derive(Debug)]
pub struct AnalyticExpectedUtilityOfBestOption {
    pref: Arc<dyn Model>,
    outcome: Option<Arc<dyn Model>>,
    previous_winner: Option<Vector>,
}

impl AnalyticExpectedUtilityOfBestOption {
    /// The acquisition function of the preference model `pref`, the
    /// deterministic `outcome` model (`None`: the candidates are the
    /// preference model's inputs) and the previous winner (in the outcome
    /// space).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for a batched or multi-output preference
    /// model or a stochastic outcome model; dimension mismatches.
    pub fn new(
        pref: Arc<dyn Model>,
        outcome: Option<Arc<dyn Model>>,
        previous_winner: Option<Vector>,
    ) -> Result<Self> {
        crate::acquisition::check_model(
            pref.as_ref(),
            "Must specify a posterior transform when using a multi-output model.",
        )?;
        if let Some(o) = &outcome {
            ComposedModel::new(Arc::clone(&pref), Arc::clone(o))?;
        }
        Ok(Self {
            pref,
            outcome,
            previous_winner,
        })
    }

    /// The previous winner.
    pub fn previous_winner(&self) -> Option<&Vector> {
        self.previous_winner.as_ref()
    }

    fn expected_q(&self) -> usize {
        if self.previous_winner.is_some() {
            1
        } else {
            2
        }
    }

    fn evaluate(&self, x: &Matrix, grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.dim();
        if x.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X with {d} columns (d), got {}",
                x.ncols()
            )));
        }
        if x.nrows() != self.expected_q() {
            return Err(BotorchError::Unsupported(
                "AnalyticExpectedUtilityOfBestOption only supports q=2 (no previous winner \
                 specified) or q=1 (previous winner specified)."
                    .into(),
            ));
        }
        let pair = pair_posterior(
            self.pref.as_ref(),
            self.outcome.as_deref(),
            self.previous_winner.as_ref(),
            x,
            grad,
        )?;
        let mean = &pair.post.mean;
        let cov = &pair.post.covariance;
        let delta = mean[0] - mean[1];
        let var = diff_variance(cov);
        let sigma = var.max(SIGMA_JITTER).sqrt();
        let u = delta / sigma;
        let ucdf = ncdf(u);
        let updf = npdf(u);
        let mut value = sigma * (updf + u * ucdf);
        if self.previous_winner.is_none() {
            value += mean[1];
        }
        if !grad {
            return Ok((value, None));
        }
        // d/dδ = Φ(u), d/dσ = φ(u).
        let mut mean_bar = Vector::from_vec(vec![ucdf, -ucdf]);
        if self.previous_winner.is_none() {
            mean_bar[1] += 1.0;
        }
        let sigma_bar = updf;
        let var_bar = if var > SIGMA_JITTER {
            sigma_bar / (2.0 * sigma)
        } else {
            0.0
        };
        let cov_bar = diff_variance_bar(var_bar);
        let g = pair_backward(self.pref.as_ref(), &pair, x, &mean_bar, &cov_bar)?;
        Ok((value, Some(g)))
    }
}

impl AcquisitionFunction for AnalyticExpectedUtilityOfBestOption {
    fn model(&self) -> &Arc<dyn Model> {
        &self.pref
    }

    fn dim(&self) -> usize {
        match &self.outcome {
            Some(o) => o.dim(),
            None => self.pref.dim(),
        }
    }

    fn expected_q(&self) -> Option<usize> {
        Some(AnalyticExpectedUtilityOfBestOption::expected_q(self))
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }
}

// ---------------------------------------------------------------------------
// Pairwise BALD
// ---------------------------------------------------------------------------

/// The entropy of `Bernoulli(probs=p)` as torch computes it
/// (`binary_cross_entropy_with_logits(logits(clamp(p, eps, 1 - eps)), p)`)
/// and its derivative with respect to `p`.
fn bernoulli_entropy(p: f64) -> (f64, f64) {
    let eps = f64::EPSILON;
    let pc = p.clamp(eps, 1.0 - eps);
    let l = pc.ln() - (-pc).ln_1p();
    let h = l.max(0.0) - l * p + (-l.abs()).exp().ln_1p();
    // ∂/∂l = sigmoid(l) - p; ∂/∂p (target) = -l; ∂l/∂p through the clamp.
    let dl = if p > eps && p < 1.0 - eps {
        1.0 / pc + 1.0 / (1.0 - pc)
    } else {
        0.0
    };
    let sig = 1.0 / (1.0 + (-l).exp());
    (h, (sig - p) * dl - l)
}

/// BoTorch's `PairwiseBayesianActiveLearningByDisagreement` (MC BALD) for a
/// pair of candidates: `H[Φ(μ / sqrt(v + 1))] - mean_i H[Φ(μ + σ ε_i)]` with
/// `μ` the mean of `f₁ - f₂`, `v = 2·std_noise + Var(f₁ - f₂)`,
/// `σ = sqrt(max(v, 1e-8))`, `H` the Bernoulli entropy and `ε_i` standard
/// normal draws (`num_samples` of them per evaluation: given explicitly, or
/// drawn from the function's own generator).
pub struct PairwiseBayesianActiveLearningByDisagreement {
    pref: Arc<dyn Model>,
    outcome: Option<Arc<dyn Model>>,
    num_samples: usize,
    std_noise: f64,
    rng: Mutex<StdRng>,
}

impl fmt::Debug for PairwiseBayesianActiveLearningByDisagreement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PairwiseBayesianActiveLearningByDisagreement")
            .field("num_samples", &self.num_samples)
            .field("std_noise", &self.std_noise)
            .finish_non_exhaustive()
    }
}

impl PairwiseBayesianActiveLearningByDisagreement {
    /// The acquisition function (BoTorch's defaults: `num_samples = 1024`,
    /// `std_noise = 0`); `seed` seeds the draws of evaluations without
    /// explicit draws.
    ///
    /// # Errors
    /// See [`AnalyticExpectedUtilityOfBestOption::new`].
    pub fn new(
        pref: Arc<dyn Model>,
        outcome: Option<Arc<dyn Model>>,
        num_samples: usize,
        std_noise: f64,
        seed: u64,
    ) -> Result<Self> {
        crate::acquisition::check_model(
            pref.as_ref(),
            "Must specify a posterior transform when using a multi-output model.",
        )?;
        if let Some(o) = &outcome {
            ComposedModel::new(Arc::clone(&pref), Arc::clone(o))?;
        }
        Ok(Self {
            pref,
            outcome,
            num_samples,
            std_noise,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        })
    }

    /// The number of draws per evaluation.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn draws(&self) -> Vec<f64> {
        let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
        (0..self.num_samples)
            .map(|_| StandardNormal.sample(&mut *rng))
            .collect()
    }

    /// The value (and gradient) at the pair `x` (`2 × d`) with the standard
    /// normal draws `eps` (`num_samples` values).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong shape of `x` or
    /// `eps`; model errors; [`BotorchError::InvalidArgument`] for NaN
    /// probabilities (torch's `Bernoulli` validation).
    pub fn evaluate_with_draws(
        &self,
        x: &Matrix,
        eps: &[f64],
        grad: bool,
    ) -> Result<(f64, Option<Matrix>)> {
        check_candidates(x, self.dim(), Some(2))?;
        if eps.len() != self.num_samples {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} draws, got {}",
                self.num_samples,
                eps.len()
            )));
        }
        let pair = pair_posterior(self.pref.as_ref(), self.outcome.as_deref(), None, x, grad)?;
        let mean = &pair.post.mean;
        let mu = mean[0] - mean[1];
        let var = 2.0 * self.std_noise + diff_variance(&pair.post.covariance);
        let sigma = var.max(SIGMA_JITTER).sqrt();
        let a = (var + 1.0).sqrt();
        let p0 = ncdf(mu / a);
        if p0.is_nan() {
            return Err(BotorchError::InvalidArgument(
                "Expected parameter probs of distribution Bernoulli to satisfy the constraint \
                 Interval(lower_bound=0.0, upper_bound=1.0), but found invalid values"
                    .into(),
            ));
        }
        let (h0, dh0) = bernoulli_entropy(p0);
        let n = eps.len() as f64;
        let mut cond = 0.0;
        let mut mu_bar = 0.0;
        let mut sigma_bar = 0.0;
        for &e in eps {
            let z = mu + e * sigma;
            let p = ncdf(z);
            let (h, dh) = bernoulli_entropy(p);
            cond += h;
            if grad {
                // d/dz Φ(z) = φ(z); the conditional entropy enters with -1/N.
                let g = -dh * npdf(z) / n;
                mu_bar += g;
                sigma_bar += g * e;
            }
        }
        let value = h0 - cond / n;
        if !grad {
            return Ok((value, None));
        }
        // The posterior entropy: Φ(μ / sqrt(v + 1)).
        let g0 = dh0 * npdf(mu / a);
        mu_bar += g0 / a;
        let mut var_bar = g0 * (-mu / (a * a)) * (0.5 / a);
        if var > SIGMA_JITTER {
            var_bar += sigma_bar / (2.0 * sigma);
        }
        let mean_bar = Vector::from_vec(vec![mu_bar, -mu_bar]);
        let cov_bar = diff_variance_bar(var_bar);
        let g = pair_backward(self.pref.as_ref(), &pair, x, &mean_bar, &cov_bar)?;
        Ok((value, Some(g)))
    }
}

impl AcquisitionFunction for PairwiseBayesianActiveLearningByDisagreement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.pref
    }

    fn dim(&self) -> usize {
        match &self.outcome {
            Some(o) => o.dim(),
            None => self.pref.dim(),
        }
    }

    fn expected_q(&self) -> Option<usize> {
        Some(2)
    }

    /// The value with fresh draws from the function's own generator (BoTorch
    /// draws new samples at every evaluation).
    fn forward(&self, x: &Matrix) -> Result<f64> {
        let eps = self.draws();
        Ok(self.evaluate_with_draws(x, &eps, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let eps = self.draws();
        let (v, g) = self.evaluate_with_draws(x, &eps, true)?;
        Ok((v, g.expect("gradient requested")))
    }
}

// ---------------------------------------------------------------------------
// PairwiseMCPosteriorVariance
// ---------------------------------------------------------------------------

/// BoTorch's `PairwiseMCPosteriorVariance`: for `q = 2k` points, the mean
/// over the `k` pairs `(x_{2i}, x_{2i+1})` of the sample variance (over the
/// MC samples, unbiased) of `objective(y_{2i} - y_{2i+1})` (a link function
/// such as the probit, evaluated without `X`).
#[derive(Debug)]
pub struct PairwiseMcPosteriorVariance {
    model: Arc<dyn Model>,
    objective: Arc<dyn McObjective>,
    sampler: Arc<SobolQMCNormalSampler>,
}

impl PairwiseMcPosteriorVariance {
    /// The acquisition function of a single-output `model`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for a batched or multi-output model.
    pub fn new(
        model: Arc<dyn Model>,
        objective: Arc<dyn McObjective>,
        sampler: Arc<SobolQMCNormalSampler>,
    ) -> Result<Self> {
        crate::acquisition::check_model(
            model.as_ref(),
            "PairwiseMCPosteriorVariance requires a single-output model.",
        )?;
        Ok(Self {
            model,
            objective,
            sampler,
        })
    }

    fn evaluate(&self, x: &Matrix, grad: bool) -> Result<(f64, Option<Matrix>)> {
        let q = x.nrows();
        if q == 0 || !q.is_multiple_of(2) {
            return Err(BotorchError::Runtime(
                "q must be a multiple of 2 for PairwiseMCPosteriorVariance".into(),
            ));
        }
        check_candidates(x, self.model.dim(), None)?;
        let js = JointSamples::new(self.model.as_ref(), x.clone(), &self.sampler)?;
        let samples = js.samples(0..q);
        let s = samples.num_samples();
        let k = q / 2;
        let data = samples.data();
        let diff = Matrix::from_fn(k, s, |i, c| data[(2 * i, c)] - data[(2 * i + 1, c)]);
        let diffs = McSamples::new(diff, k, 1)?;
        let v = self.objective.evaluate(&diffs, None, grad)?;
        if v.values.shape() != (k, s) {
            return Err(BotorchError::DimensionMismatch(format!(
                "the objective returned {} x {} values for {k} pairs and {s} samples",
                v.values.nrows(),
                v.values.ncols()
            )));
        }
        let sf = s as f64;
        let mut value = 0.0;
        let mut obj_bar = Matrix::zeros(k, s);
        for i in 0..k {
            let row = v.values.row(i);
            let m = row.sum() / sf;
            let var = row.iter().map(|o| (o - m) * (o - m)).sum::<f64>() / (sf - 1.0);
            value += var;
            if grad {
                for c in 0..s {
                    obj_bar[(i, c)] = 2.0 * (row[c] - m) / (sf - 1.0) / k as f64;
                }
            }
        }
        value /= k as f64;
        if !grad {
            return Ok((value, None));
        }
        let pullback = v.pullback.expect("pullback requested");
        let (diff_bar, _) = pullback(&obj_bar)?;
        let mut ybar = Matrix::zeros(q, s);
        for i in 0..k {
            for c in 0..s {
                ybar[(2 * i, c)] += diff_bar[(i, c)];
                ybar[(2 * i + 1, c)] -= diff_bar[(i, c)];
            }
        }
        let g = js.backward(self.model.as_ref(), 0..q, &ybar, 0..q)?;
        Ok((value, Some(g)))
    }
}

impl AcquisitionFunction for PairwiseMcPosteriorVariance {
    fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }
}

// ---------------------------------------------------------------------------
// LearnedObjective
// ---------------------------------------------------------------------------

/// The posterior samples `μ(x) + R(x) z` of a single-output model at the
/// `q` points `x` for the base samples `base` (`N × q`), as `N × q`, and,
/// with the adjoint `ybar` (`N × q`) of the samples, the adjoint of `x`
/// (`q × d`): the computation of BoTorch's `LearnedObjective` for one set
/// of outcome samples (the preference model's posterior at the samples,
/// sampled with its `IIDNormalSampler`'s base samples).
///
/// # Errors
/// Model errors; [`BotorchError::DimensionMismatch`] for inconsistent
/// shapes.
pub fn posterior_samples_vjp(
    model: &dyn Model,
    x: &Matrix,
    base: &Matrix,
    ybar: Option<&Matrix>,
) -> Result<(Matrix, Option<Matrix>)> {
    let q = x.nrows();
    let single_output = |s: &McSamples| {
        if s.num_outputs() == 1 {
            Ok(())
        } else {
            Err(BotorchError::Unsupported(
                "LearnedObjective requires a single-output preference model".into(),
            ))
        }
    };
    let Some(ybar) = ybar else {
        let js = JointSamples::with_base_samples(model, x.clone(), Arc::new(base.clone()))?;
        let samples = js.samples(0..q);
        single_output(&samples)?;
        return Ok((samples.into_data().transpose(), None));
    };
    if ybar.shape() != base.shape() {
        return Err(BotorchError::DimensionMismatch(format!(
            "the adjoint has shape {:?}, the base samples {:?}",
            ybar.shape(),
            base.shape()
        )));
    }
    // With one output the adjoint has the layout of the base samples.
    let (samples, g) = super::monte_carlo::posterior_samples_with_vjp(model, x, base, ybar)?;
    single_output(&samples)?;
    Ok((samples.into_data().transpose(), Some(g)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_entropy_derivative() {
        for p in [0.1, 0.5, 0.73, 0.999] {
            let (_, d) = bernoulli_entropy(p);
            let h = 1e-7;
            let fd = (bernoulli_entropy(p + h).0 - bernoulli_entropy(p - h).0) / (2.0 * h);
            assert!((fd - d).abs() < 1e-6, "{p}: {fd} vs {d}");
        }
    }
}
