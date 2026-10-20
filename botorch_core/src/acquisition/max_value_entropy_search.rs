//! Max-value entropy search (`botorch.acquisition.max_value_entropy_search`;
//! Wang and Jegelka, *Max-value Entropy Search for Efficient Bayesian
//! Optimization*, ICML 2017; Moss et al., *GIBBON: General-purpose
//! Information-Based Bayesian Optimisation*, JMLR 2021; Takeno et al.,
//! *Multi-fidelity Bayesian Optimization with Max-value Entropy Search and
//! its Parallelization*, ICML 2020).
//!
//! * [`QMaxValueEntropy`] (qMES): the mutual information between the noisy
//!   observation at a candidate and the maximum value, estimated with
//!   `num_mv_samples` samples of the maximum over a discrete candidate set
//!   ([`sample_max_value_gumbel`], [`sample_max_value_thompson`]) and
//!   `num_y_samples` quasi-MC samples of the observation, with BoTorch's
//!   regression-adjusted estimate of the conditional entropy. Pending points
//!   are fantasized (`num_fantasies` fantasies) and the information gain is
//!   averaged over the fantasy models.
//! * [`QLowerBoundMaxValueEntropy`] (GIBBON): the cheap lower bound of the
//!   information gain, with the repulsion term of the pending points for
//!   batches.
//! * [`QMultiFidelityMaxValueEntropy`] and
//!   [`QMultiFidelityLowerBoundMaxValueEntropy`]: the information gain about
//!   the maximum at the target fidelity (`project`) of observations at a
//!   lower fidelity (with trace observations, `expand`), weighted by a
//!   cost-aware utility.
//!
//! The information gains follow BoTorch's computation step by step (the same
//! clamps at `1e-8`, `torch.distributions.Normal`'s CDF, the multivariate
//! normal log densities and entropies); their exact gradients are obtained
//! by evaluating the formulas with dual numbers ([`super::dual`]) seeded on
//! the posterior quantities, which are then propagated through the model.
//!
//! **Differences to BoTorch:**
//!
//! * **Training inputs of the candidate set.** BoTorch appends
//!   `model.train_inputs[0]` to the candidate set of the max-value samples,
//!   which for a model with an input transform are the *transformed*
//!   inputs, which the posterior transforms again; botorch_rs appends the
//!   raw training inputs (both agree without input transforms or with an
//!   explicit `train_inputs`).
//! * **The repulsion term of GIBBON** uses the posterior cross-covariance of
//!   the candidate and the pending points. BoTorch computes it with
//!   `model(X)`, i.e. without the input transform and in the
//!   outcome-transformed space, while the other terms are in the original
//!   space; both agree for models without transforms.
//! * **Random numbers.** The uniform numbers of the Gumbel sampling and the
//!   normal numbers of the Thompson sampling come from a generator seeded by
//!   `seed` (BoTorch: torch's global generator); the max-value samples can be
//!   set explicitly (`set_posterior_max_values`).

use std::sync::{Arc, RwLock};

use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::acquisition::cost_aware::CostAwareUtility;
use crate::acquisition::dual::{Dual, Scalar};
use crate::acquisition::fidelity::InputProjection;
use crate::acquisition::monte_carlo::{read_lock, vstack, write_lock};
use crate::acquisition::objective::ScalarizedPosteriorTransform;
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::gp::{FantasyGp, FantasyPosterior};
use crate::models::{Model, SingleTaskGP};
use crate::posterior::MIN_VARIANCE;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::probability::log_ndtr;
use crate::utils::sampling::random_seed;
use crate::{Matrix, Vector};

/// BoTorch's `CLAMP_LB` of the max-value entropy search.
pub const CLAMP_LB: f64 = 1e-8;
/// BoTorch's default number of max-value samples.
pub const DEFAULT_NUM_MV_SAMPLES: usize = 10;
/// BoTorch's default number of fantasies of the pending points.
pub const DEFAULT_NUM_FANTASIES: usize = 16;
/// BoTorch's default number of samples of the observation.
pub const DEFAULT_NUM_Y_SAMPLES: usize = 128;

/// `log(2π)`.
const LOG_2PI: f64 = 1.837_877_066_409_345_3;

/// SciPy's default absolute tolerance of `brentq`.
const BRENTQ_XTOL: f64 = 2e-12;
/// SciPy's default relative tolerance of `brentq` (`4 eps`).
const BRENTQ_RTOL: f64 = 8.881_784_197_001_252e-16;
/// SciPy's default maximal number of iterations of `brentq`.
const BRENTQ_MAXITER: usize = 100;

/// SciPy's `brentq(f, xa, xb, xtol, rtol, maxiter)` (a port of
/// `scipy/optimize/Zeros/brentq.c`): a root of `f` in `[xa, xb]`, where
/// `f(xa)` and `f(xb)` have different signs.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `f(xa)` and `f(xb)` have the same
/// sign (SciPy's message) and [`BotorchError::Numerical`] if the iteration
/// does not converge.
pub(crate) fn brentq<F: FnMut(f64) -> f64>(
    mut f: F,
    xa: f64,
    xb: f64,
    xtol: f64,
    rtol: f64,
    maxiter: usize,
) -> Result<f64> {
    let (mut xpre, mut xcur) = (xa, xb);
    let (mut xblk, mut fblk, mut spre, mut scur) = (0.0, 0.0, 0.0, 0.0);
    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }
    if fpre.is_sign_negative() == fcur.is_sign_negative() {
        return Err(BotorchError::InvalidArgument(
            "f(a) and f(b) must have different signs".into(),
        ));
    }
    for _ in 0..maxiter {
        if fpre != 0.0 && fcur != 0.0 && fpre.is_sign_negative() != fcur.is_sign_negative() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }
        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // interpolate
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                // good short step
                spre = scur;
                scur = stry;
            } else {
                // bisect
                spre = sbis;
                scur = sbis;
            }
        } else {
            // bisect
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }
    Err(BotorchError::Numerical("brentq failed to converge".into()))
}

/// The weighted posterior means and standard deviations
/// (`sqrt(clamp(variance, 1e-8))`) of the points of `candidate_set`.
fn candidate_marginals(
    model: &dyn Model,
    candidate_set: &Matrix,
    maximize: bool,
) -> Result<(Vector, Vector)> {
    let (mean, var) = model.posterior_marginals(candidate_set)?;
    let w = if maximize { 1.0 } else { -1.0 };
    Ok((mean.map(|m| w * m), var.map(|v| v.max(1e-8).sqrt())))
}

/// The location `a` and scale `b` of the Gumbel distribution that BoTorch's
/// `_sample_max_value_Gumbel` fits to the maximum of independent normal
/// variables with the posterior marginals at `candidate_set` (weighted by
/// `±1` for `maximize`): the quartiles of `Π_i Φ((y - μ_i) / σ_i)` are found
/// with `brentq` and matched to the Gumbel quartiles.
///
/// # Errors
/// [`BotorchError::Unsupported`] for multi-output models, errors of the
/// posterior and of `brentq`.
pub fn gumbel_parameters(
    model: &dyn Model,
    candidate_set: &Matrix,
    maximize: bool,
) -> Result<(f64, f64)> {
    let (mu, sigma) = candidate_marginals(model, candidate_set, maximize)?;
    let lo = mu
        .iter()
        .zip(sigma.iter())
        .map(|(m, s)| m - 3.0 * s)
        .fold(f64::INFINITY, f64::min);
    let hi = mu
        .iter()
        .zip(sigma.iter())
        .map(|(m, s)| m + 5.0 * s)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut q = [0.0; 3];
    for (qi, p) in q.iter_mut().zip([0.25, 0.5, 0.75]) {
        *qi = brentq(
            |y| {
                let s: f64 = mu
                    .iter()
                    .zip(sigma.iter())
                    .map(|(m, s)| log_ndtr((y - m) / s))
                    .sum();
                s.exp() - p
            },
            lo,
            hi,
            BRENTQ_XTOL,
            BRENTQ_RTOL,
            BRENTQ_MAXITER,
        )?;
    }
    let b = (q[0] - q[2]) / ((4.0f64 / 3.0).ln().ln() - 4.0f64.ln().ln());
    let a = q[1] + b * 2.0f64.ln().ln();
    Ok((a, b))
}

/// BoTorch's `_sample_max_value_Gumbel`: `num_samples` samples
/// `a - b log(-log(u))` of the Gumbel approximation
/// ([`gumbel_parameters`]) of the maximum of the posterior over
/// `candidate_set`, for the uniform numbers `uniforms` (BoTorch's
/// `torch.rand(num_samples, 1)`).
///
/// # Errors
/// See [`gumbel_parameters`].
pub fn sample_max_value_gumbel(
    model: &dyn Model,
    candidate_set: &Matrix,
    maximize: bool,
    uniforms: &[f64],
) -> Result<Vec<f64>> {
    let (a, b) = gumbel_parameters(model, candidate_set, maximize)?;
    Ok(uniforms.iter().map(|u| a - b * (-u.ln()).ln()).collect())
}

/// BoTorch's `_sample_max_value_Thompson`: the maxima of joint posterior
/// samples (weighted by `±1` for `maximize`) over `candidate_set`, one per
/// row of the standard normal `base_samples` (`num_samples × n`; BoTorch
/// draws them with `torch.randn`).
///
/// # Errors
/// Shape errors and errors of the posterior.
pub fn sample_max_value_thompson(
    model: &dyn Model,
    candidate_set: &Matrix,
    maximize: bool,
    base_samples: &Matrix,
) -> Result<Vec<f64>> {
    let post = model.posterior(candidate_set, false)?;
    let samples = post.rsample_from_base_samples(base_samples)?;
    let w = if maximize { 1.0 } else { -1.0 };
    let data = samples.data();
    Ok((0..data.ncols())
        .map(|c| {
            data.column(c)
                .iter()
                .map(|v| w * v)
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect())
}

/// The max-value samples of [`MaxValueBase`] for the candidate set (with
/// the training inputs, pending points and projection applied), drawn with
/// a generator seeded by `seed`.
fn draw_max_values(
    model: &dyn Model,
    candidate_set: &Matrix,
    num_samples: usize,
    maximize: bool,
    use_gumbel: bool,
    seed: u64,
) -> Result<Vec<f64>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    if use_gumbel {
        let uniforms: Vec<f64> = (0..num_samples).map(|_| rng.random::<f64>()).collect();
        sample_max_value_gumbel(model, candidate_set, maximize, &uniforms)
    } else {
        let n = candidate_set.nrows();
        let base = Matrix::from_fn(num_samples, n, |_, _| StandardNormal.sample(&mut rng));
        sample_max_value_thompson(model, candidate_set, maximize, &base)
    }
}

// ---------------------------------------------------------------------------
// Information gains, generic over dual numbers
// ---------------------------------------------------------------------------

/// The lower-triangular Cholesky factor of a small symmetric matrix of
/// scalars (row-major), with GPyTorch's `psd_safe_cholesky` jitter
/// (`1e-8`, `1e-7`, `1e-6` on the diagonal after a failure).
pub(crate) fn cholesky_scalar<S: Scalar>(a: &[Vec<S>]) -> Result<Vec<Vec<S>>> {
    let n = a.len();
    let zero = a[0][0].lift(0.0);
    'outer: for attempt in 0..4 {
        let jitter = if attempt == 0 {
            0.0
        } else {
            crate::linalg::CHOLESKY_JITTER * 10f64.powi(attempt - 1)
        };
        let mut l = vec![vec![zero; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = a[i][j];
                if i == j {
                    s = s + jitter;
                }
                for (lik, ljk) in l[i].iter().zip(&l[j]).take(j) {
                    s = s - *lik * *ljk;
                }
                if i == j {
                    if s.value().is_nan() || s.value() <= 0.0 {
                        continue 'outer;
                    }
                    l[i][i] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        return Ok(l);
    }
    Err(BotorchError::NotPsd {
        jitter: crate::linalg::CHOLESKY_JITTER * 100.0,
    })
}

/// Solves `L u = b` for a lower-triangular `L`.
fn solve_lower<S: Scalar>(l: &[Vec<S>], b: &[S]) -> Vec<S> {
    let n = l.len();
    let mut u: Vec<S> = b.to_vec();
    for i in 0..n {
        let mut s = u[i];
        for k in 0..i {
            s = s - l[i][k] * u[k];
        }
        u[i] = s / l[i][i];
    }
    u
}

/// Solves `Lᵀ u = b` for a lower-triangular `L`.
fn solve_upper<S: Scalar>(l: &[Vec<S>], b: &[S]) -> Vec<S> {
    let n = l.len();
    let mut u: Vec<S> = b.to_vec();
    for i in (0..n).rev() {
        let mut s = u[i];
        for k in i + 1..n {
            s = s - l[k][i] * u[k];
        }
        u[i] = s / l[i][i];
    }
    u
}

/// The quantities of one candidate (and fantasy) that the information gain
/// of [`QMaxValueEntropy`] depends on.
#[derive(Clone, Debug)]
struct MesInputs<S> {
    /// The weighted posterior mean at the target point (`weight · μ_M`).
    mean_target: S,
    /// The clamped noiseless variance at the target point (`variance_M`).
    var_target: S,
    /// The noiseless covariances of the (expanded) candidate points with the
    /// target point (`covar_mM`, `1 + K`).
    cov_mt: Vec<S>,
    /// The (unweighted) noisy posterior mean at the expanded points.
    mean_m: Vec<S>,
    /// The noisy posterior covariance at the expanded points (row-major).
    cov_m: Vec<Vec<S>>,
}

/// qMES's information gain (BoTorch's
/// `qMaxValueEntropy._compute_information_gain` for one candidate and one
/// fantasy) with the max values `max_values` and the base samples `z` of the
/// observation (`s_m × (1 + K)`).
fn mes_information_gain<S: Scalar>(
    inp: &MesInputs<S>,
    weight: f64,
    max_values: &[f64],
    z: &Matrix,
) -> Result<S> {
    let k1 = inp.mean_m.len();
    let s_m = z.nrows();
    let s_big = max_values.len();
    let l = cholesky_scalar(&inp.cov_m)?;
    // temp = Σ_m⁻¹ c (cholesky_solve).
    let t = solve_upper(&l, &solve_lower(&l, &inp.cov_mt));
    // var_new = var_M - tᵀ c.
    let mut tc = inp.var_target.lift(0.0);
    for (ti, ci) in t.iter().zip(&inp.cov_mt) {
        tc = tc + *ti * *ci;
    }
    let var_new = inp.var_target - tc;
    let stdv_new = var_new.clamp_min(CLAMP_LB).sqrt();
    let stdv_m = inp.var_target.sqrt();
    let mut logdet_half = inp.var_target.lift(0.0);
    for (i, li) in l.iter().enumerate() {
        logdet_half = logdet_half + li[i].ln();
    }
    // H0 = H(y_m | x, D).
    let h0_const = 0.5 * k1 as f64 * (1.0 + LOG_2PI);
    let h0 = logdet_half + h0_const;
    // Samples, the means of the conditional max-value distribution and the
    // log densities of the samples.
    let mut mean_new = Vec::with_capacity(s_m);
    let mut log_pdf = Vec::with_capacity(s_m);
    for k in 0..s_m {
        // y = μ_m + L z_k (unweighted), samples_m = w y.
        let mut y = Vec::with_capacity(k1);
        for (i, li) in l.iter().enumerate() {
            let mut v = inp.mean_m[i];
            for (j, lij) in li.iter().enumerate().take(i + 1) {
                v = v + *lij * z[(k, j)];
            }
            y.push(v);
        }
        let mut mp = inp.var_target.lift(0.0);
        for i in 0..k1 {
            mp = mp + t[i] * (y[i] * weight - inp.mean_m[i] * weight);
        }
        mean_new.push(mp + inp.mean_target);
        let diff: Vec<S> = (0..k1).map(|i| y[i] - inp.mean_m[i]).collect();
        let u = solve_lower(&l, &diff);
        let mut quad = inp.var_target.lift(0.0);
        for ui in &u {
            quad = quad + *ui * *ui;
        }
        log_pdf.push(-(quad * 0.5) - logdet_half - 0.5 * k1 as f64 * LOG_2PI);
    }
    let cdf_mvs: Vec<S> = max_values
        .iter()
        .map(|&mv| {
            ((-inp.mean_target + mv) / stdv_m)
                .ncdf()
                .clamp_min(CLAMP_LB)
        })
        .collect();
    let n_all = (s_big * s_m) as f64;
    let mut h1 = Vec::with_capacity(s_big * s_m);
    let mut h1_sum = inp.var_target.lift(0.0);
    for (j, &mv) in max_values.iter().enumerate() {
        for k in 0..s_m {
            let cdf_new = ((-mean_new[k] + mv) / stdv_new).ncdf().clamp_min(CLAMP_LB);
            let zz = cdf_new / cdf_mvs[j];
            let v = -(zz * zz.ln()) - zz * log_pdf[k];
            if v.value().is_nan() {
                return Err(BotorchError::InputData(
                    "Encountered NaN values in the information gain (check_no_nans).".into(),
                ));
            }
            h1_sum = h1_sum + v;
            h1.push(v);
        }
    }
    let h1_bar = h1_sum / n_all;
    // The unbiased variance of one observation is undefined. Without a
    // control variate the estimator remains H0 - mean(h1).
    if s_m <= 1 || n_all <= 1.0 {
        return Ok(h0 - h1_bar);
    }
    let mut h0_sum = inp.var_target.lift(0.0);
    for lp in &log_pdf {
        // Center before summing so a constant control variate has exactly
        // zero variance, rather than a rounding residual from its mean.
        h0_sum = h0_sum + (log_pdf[0] - *lp);
    }
    let h0_bar = -log_pdf[0] + h0_sum / s_m as f64;
    let mut cov = inp.var_target.lift(0.0);
    let mut var_h1 = inp.var_target.lift(0.0);
    for j in 0..s_big {
        for k in 0..s_m {
            let d1 = h1[j * s_m + k] - h1_bar;
            let d0 = -log_pdf[k] - h0_bar;
            cov = cov + d1 * d0;
            var_h1 = var_h1 + d1 * d1;
        }
    }
    let cov = cov / n_all;
    let var_h1 = var_h1 / (n_all - 1.0);
    let mut var_h0 = inp.var_target.lift(0.0);
    for lp in &log_pdf {
        let d0 = -*lp - h0_bar;
        var_h0 = var_h0 + d0 * d0;
    }
    let var_h0 = var_h0 / (s_m as f64 - 1.0);
    let product = var_h0 * var_h1;
    let beta = if product.value() > 0.0 {
        cov / product.sqrt()
    } else {
        inp.var_target.lift(0.0)
    };
    let h1_hat = h1_bar - beta * (h0_bar - h0);
    Ok(h0 - h1_hat)
}

/// GIBBON's information gain (BoTorch's
/// `qLowerBoundMaxValueEntropy._compute_information_gain` without the
/// repulsion term) for the weighted mean `mean_target` and clamped variance
/// `var_target` at the target point, the covariance `cov_mt` with the
/// candidate and the clamped noisy variance `var_m` at the candidate.
fn gibbon_information_gain<S: Scalar>(
    mean_target: S,
    var_target: S,
    cov_mt: S,
    var_m: S,
    max_values: &[f64],
) -> S {
    let stdv = var_target.sqrt();
    let rho2 = cov_mt * cov_mt / (var_m * var_target);
    let mut acq = mean_target.lift(0.0);
    for &mv in max_values {
        let z = (-mean_target + mv) / stdv;
        let cdf = z.ncdf().clamp_min(CLAMP_LB);
        let lp = -(z * z) / 2.0 - LOG_SQRT_2PI;
        let pdf = lp.exp();
        let ratio = pdf / cdf;
        let inner = -(rho2 * ratio * (z + ratio)) + 1.0;
        acq = acq + inner.clamp_min(CLAMP_LB).ln() * -0.5;
    }
    acq / max_values.len() as f64
}

/// `log(sqrt(2π))` of `Normal.log_prob`.
const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

// ---------------------------------------------------------------------------
// Posterior sources
// ---------------------------------------------------------------------------

/// Where the posteriors at the candidates come from: the model, or fantasy
/// `i` of the fantasy models of the pending points.
enum Source<'a> {
    Model(&'a dyn Model),
    Fantasy(&'a FantasyGp<'a>, usize),
}

/// A noiseless single-output posterior from a [`Source`] with what its
/// reverse pass needs.
struct SourcePosterior {
    mean: Vector,
    cov: Matrix,
    handle: Option<FantasyPosterior>,
}

impl Source<'_> {
    fn posterior(&self, x: &Matrix) -> Result<SourcePosterior> {
        match self {
            Source::Model(m) => {
                let p = m.posterior(x, false)?;
                Ok(SourcePosterior {
                    mean: p.mean,
                    cov: p.covariance,
                    handle: None,
                })
            }
            Source::Fantasy(f, i) => {
                let p = f.posterior(*i, x, true)?;
                Ok(SourcePosterior {
                    mean: p.mean.clone(),
                    cov: p.covariance.clone().expect("covariance requested"),
                    handle: Some(p),
                })
            }
        }
    }

    fn vjp(
        &self,
        x: &Matrix,
        sp: &SourcePosterior,
        mean_bar: &Vector,
        cov_bar: &Matrix,
    ) -> Result<Matrix> {
        match self {
            Source::Model(m) => m.joint_posterior_vjp(x, 0..x.nrows(), mean_bar, cov_bar),
            Source::Fantasy(f, _) => {
                let mut acc = f.adjoint();
                Ok(f.posterior_backward(
                    sp.handle.as_ref().expect("fantasy posterior"),
                    mean_bar,
                    Some(cov_bar),
                    &mut acc,
                ))
            }
        }
    }
}

/// A posterior transform of a single-output posterior (`w μ + o`, `w² Σ`).
fn transformed(
    t: Option<&ScalarizedPosteriorTransform>,
    mean: &Vector,
    cov: &Matrix,
) -> (Vector, Matrix) {
    match t {
        Some(t) => {
            let w = t.weights()[0];
            (mean.map(|m| t.offset() + w * m), cov.map(|c| w * w * c))
        }
        None => (mean.clone(), cov.clone()),
    }
}

/// The adjoints of an untransformed posterior from those of the transformed
/// one.
fn pulled_back(
    t: Option<&ScalarizedPosteriorTransform>,
    mean_bar: Vector,
    cov_bar: Matrix,
) -> (Vector, Matrix) {
    match t {
        Some(t) => {
            let w = t.weights()[0];
            (mean_bar * w, cov_bar * (w * w))
        }
        None => (mean_bar, cov_bar),
    }
}

// ---------------------------------------------------------------------------
// The shared state
// ---------------------------------------------------------------------------

/// Options of the max-value entropy search acquisition functions (BoTorch's
/// keyword arguments of `MaxValueBase` and its subclasses).
#[derive(Clone, Debug)]
pub struct MaxValueEntropyOptions {
    /// The number of max-value samples (default 10).
    pub num_mv_samples: usize,
    /// A posterior transform of the single-output model (e.g. a negation).
    pub posterior_transform: Option<ScalarizedPosteriorTransform>,
    /// Sample the max values with the Gumbel approximation (default `true`)
    /// instead of discrete Thompson sampling.
    pub use_gumbel: bool,
    /// Whether larger values are better (default `true`).
    pub maximize: bool,
    /// Pending points (`m × d`): appended to the candidate set of the max
    /// values, fantasized (qMES) or repulsing the candidates (GIBBON).
    pub x_pending: Option<Matrix>,
    /// The training inputs appended to the candidate set (default: the raw
    /// training inputs of the model).
    pub train_inputs: Option<Matrix>,
    /// The number of fantasies of the pending points (qMES; default 16).
    pub num_fantasies: usize,
    /// The number of samples of the observation (qMES; default 128).
    pub num_y_samples: usize,
    /// The sampler of the observation samples (qMES; default:
    /// `num_y_samples` Sobol samples with a random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// The sampler of the fantasies of the pending points (qMES; default:
    /// `num_fantasies` Sobol samples with a random seed).
    pub fantasies_sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// The seed of the uniform (Gumbel) or normal (Thompson) numbers of the
    /// max-value samples (default: random).
    pub seed: Option<u64>,
}

impl Default for MaxValueEntropyOptions {
    fn default() -> Self {
        Self {
            num_mv_samples: DEFAULT_NUM_MV_SAMPLES,
            posterior_transform: None,
            use_gumbel: true,
            maximize: true,
            x_pending: None,
            train_inputs: None,
            num_fantasies: DEFAULT_NUM_FANTASIES,
            num_y_samples: DEFAULT_NUM_Y_SAMPLES,
            sampler: None,
            fantasies_sampler: None,
            seed: None,
        }
    }
}

/// The multi-fidelity configuration of the max-value entropy search.
#[derive(Clone, Debug)]
struct Fidelity {
    cost_aware_utility: Arc<dyn CostAwareUtility>,
    project: Arc<dyn InputProjection>,
    expand: Option<Arc<dyn InputProjection>>,
}

/// The state shared by the max-value entropy search acquisition functions
/// (BoTorch's `MaxValueBase`).
#[derive(Debug)]
struct MaxValueBase {
    model: Arc<dyn Model>,
    transform: Option<ScalarizedPosteriorTransform>,
    candidate_set: Matrix,
    num_mv_samples: usize,
    use_gumbel: bool,
    maximize: bool,
    weight: f64,
    max_values: RwLock<Vec<f64>>,
    x_pending: RwLock<Option<Matrix>>,
    seed: u64,
}

/// The model as a single-output model through the transform.
fn scalarized(
    model: &Arc<dyn Model>,
    transform: Option<&ScalarizedPosteriorTransform>,
) -> Result<Arc<dyn Model>> {
    crate::acquisition::ScalarizedModel::wrap(Arc::clone(model), transform.cloned())
}

impl MaxValueBase {
    fn new(
        model: Arc<dyn Model>,
        candidate_set: &Matrix,
        o: &MaxValueEntropyOptions,
        name: &str,
        project: Option<&Arc<dyn InputProjection>>,
    ) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        if model.num_outputs() > 1 {
            return Err(BotorchError::Unsupported(format!(
                "Multi-output models are not supported by {name}."
            )));
        }
        crate::acquisition::require_gaussian_model(model.as_ref(), name)?;
        if let Some(t) = &o.posterior_transform {
            if t.num_outputs() != 1 {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Output shape not equal to that of weights. Output shape is 1 and weights \
                     are [{}]",
                    t.num_outputs()
                )));
            }
        }
        let d = model.dim();
        if candidate_set.ncols() != d || candidate_set.nrows() == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected a non-empty `n x {d}` candidate set, got {}x{}",
                candidate_set.nrows(),
                candidate_set.ncols()
            )));
        }
        if o.num_mv_samples == 0 {
            return Err(BotorchError::InvalidArgument(
                "num_mv_samples must be positive".into(),
            ));
        }
        let train = match &o.train_inputs {
            Some(t) => Some(t.clone()),
            None => model
                .output_gps()
                .and_then(|g| g.first().map(|g| g.train_inputs().clone())),
        };
        let candidate_set = match &train {
            Some(t) => {
                if t.ncols() != d {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "train_inputs must have {d} columns, got {}",
                        t.ncols()
                    )));
                }
                vstack(&[candidate_set, t], d)
            }
            None => candidate_set.clone(),
        };
        if let Some(p) = &o.x_pending {
            check_pending(p, d)?;
        }
        let base = Self {
            transform: o.posterior_transform.clone(),
            candidate_set,
            num_mv_samples: o.num_mv_samples,
            use_gumbel: o.use_gumbel,
            maximize: o.maximize,
            weight: if o.maximize { 1.0 } else { -1.0 },
            max_values: RwLock::new(Vec::new()),
            x_pending: RwLock::new(o.x_pending.clone()),
            seed: o.seed.unwrap_or_else(random_seed),
            model,
        };
        let mv = base.sample_max_values(o.x_pending.as_ref(), project)?;
        *write_lock(&base.max_values) = mv;
        Ok(base)
    }

    /// BoTorch's `_sample_max_values`: the max values over the candidate set,
    /// the pending points and the projection.
    fn sample_max_values(
        &self,
        x_pending: Option<&Matrix>,
        project: Option<&Arc<dyn InputProjection>>,
    ) -> Result<Vec<f64>> {
        let d = self.model.dim();
        let mut cs = match x_pending {
            Some(p) => vstack(&[&self.candidate_set, p], d),
            None => self.candidate_set.clone(),
        };
        if let Some(p) = project {
            cs = p.project(&cs)?;
        }
        let model = scalarized(&self.model, self.transform.as_ref())?;
        draw_max_values(
            model.as_ref(),
            &cs,
            self.num_mv_samples,
            self.maximize,
            self.use_gumbel,
            self.seed,
        )
    }

    fn max_values(&self) -> Vec<f64> {
        read_lock(&self.max_values).clone()
    }

    fn set_max_values(&self, values: Vec<f64>) -> Result<()> {
        if values.is_empty() || values.iter().any(|v| v.is_nan()) {
            return Err(BotorchError::InvalidArgument(
                "posterior_max_values must be a non-empty vector without NaN values".into(),
            ));
        }
        *write_lock(&self.max_values) = values;
        Ok(())
    }

    /// The observation noise variance added by `posterior(x,
    /// observation_noise=True)` at the points `x` (untransformed).
    fn noise(&self, x: &Matrix) -> Result<Vector> {
        let a = self.model.posterior(x, true)?;
        let b = self.model.posterior(x, false)?;
        let n = x.nrows();
        Ok(Vector::from_fn(n, |i, _| {
            a.covariance[(i, i)] - b.covariance[(i, i)]
        }))
    }
}

// ---------------------------------------------------------------------------
// qMES and qMFMES
// ---------------------------------------------------------------------------

/// The fantasy models of the pending points of qMES.
#[derive(Debug)]
struct Fantasies {
    x_pending: Matrix,
    base: Arc<Matrix>,
    num_fantasies: usize,
}

/// The value of one fantasy (or of the model) at one candidate and its
/// gradient with respect to the candidate.
type FantasyIg = (f64, Option<Matrix>);

/// BoTorch's `qMaxValueEntropy`: see the [module documentation](self).
#[derive(Debug)]
pub struct QMaxValueEntropy {
    base: MaxValueBase,
    sampler: Arc<SobolQMCNormalSampler>,
    fantasies_sampler: Arc<SobolQMCNormalSampler>,
    num_fantasies: usize,
    fantasies: RwLock<Option<Fantasies>>,
    fidelity: Option<Fidelity>,
}

impl QMaxValueEntropy {
    /// `qMaxValueEntropy(model, candidate_set, num_fantasies,
    /// num_mv_samples, num_y_samples, posterior_transform, use_gumbel,
    /// maximize, X_pending, train_inputs)` for a single-output model.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output or batched models
    /// (BoTorch's message), [`BotorchError::DimensionMismatch`] for inputs of
    /// the wrong dimension, and errors of the max-value sampling.
    pub fn new(
        model: Arc<dyn Model>,
        candidate_set: &Matrix,
        options: MaxValueEntropyOptions,
    ) -> Result<Self> {
        Self::build(model, candidate_set, options, None, "qMaxValueEntropy")
    }

    fn build(
        model: Arc<dyn Model>,
        candidate_set: &Matrix,
        options: MaxValueEntropyOptions,
        fidelity: Option<Fidelity>,
        name: &str,
    ) -> Result<Self> {
        let base = MaxValueBase::new(
            model,
            candidate_set,
            &options,
            name,
            fidelity.as_ref().map(|f| &f.project),
        )?;
        let sampler = match options.sampler {
            Some(s) => s,
            None => Arc::new(SobolQMCNormalSampler::new(options.num_y_samples, None)?),
        };
        if sampler.num_samples() < 2 {
            return Err(BotorchError::InvalidArgument(
                "the information gain needs at least two samples of the observation".into(),
            ));
        }
        let fantasies_sampler = match options.fantasies_sampler {
            Some(s) => s,
            None => Arc::new(SobolQMCNormalSampler::new(options.num_fantasies, None)?),
        };
        let acqf = Self {
            base,
            sampler,
            num_fantasies: fantasies_sampler.num_samples(),
            fantasies_sampler,
            fantasies: RwLock::new(None),
            fidelity,
        };
        acqf.set_x_pending(options.x_pending.as_ref())?;
        Ok(acqf)
    }

    /// The max-value samples (BoTorch's `posterior_max_values`, weighted by
    /// `±1` for `maximize`).
    pub fn posterior_max_values(&self) -> Vec<f64> {
        self.base.max_values()
    }

    /// Sets the max-value samples.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for no or NaN values.
    pub fn set_posterior_max_values(&self, values: Vec<f64>) -> Result<()> {
        self.base.set_max_values(values)
    }

    /// The candidate set (with the training inputs).
    pub fn candidate_set(&self) -> &Matrix {
        &self.base.candidate_set
    }

    /// The sampler of the observation samples.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.sampler
    }

    /// The sampler of the fantasies of the pending points.
    pub fn fantasies_sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.fantasies_sampler
    }

    /// The number of fantasies of the pending points.
    pub fn num_fantasies(&self) -> usize {
        self.num_fantasies
    }

    /// The single output GP of the model (required for fantasies).
    fn gp(&self) -> Result<&SingleTaskGP> {
        match self.base.model.output_gps() {
            Some(g) if g.len() == 1 => Ok(g[0]),
            _ => Err(BotorchError::Unsupported(
                "pending points require a model with one exact GP output (fantasize)".into(),
            )),
        }
    }

    /// The points of the information gain: the expanded candidate and the
    /// target point.
    fn points(&self, x: &Matrix) -> Result<(Matrix, Matrix)> {
        match &self.fidelity {
            None => Ok((x.clone(), x.clone())),
            Some(f) => {
                let xe = match &f.expand {
                    Some(e) => e.project(x)?,
                    None => x.clone(),
                };
                Ok((xe, f.project.project(x)?))
            }
        }
    }

    /// The information gain of `source` at the candidate `x` (`1 × d`) and,
    /// if requested, its gradient with respect to `x`.
    fn source_ig(
        &self,
        source: &Source<'_>,
        noise: &Vector,
        x: &Matrix,
        max_values: &[f64],
        z: &Matrix,
        with_grad: bool,
    ) -> Result<FantasyIg> {
        let t = self.base.transform.as_ref();
        let w = self.base.weight;
        let (xe, xt) = self.points(x)?;
        let k1 = xe.nrows();
        let d = x.ncols();
        if self.fidelity.is_none() {
            // Single fidelity: tangents (μ, v) of the posterior at x.
            let sp = source.posterior(x)?;
            let (m, c) = transformed(t, &sp.mean, &sp.cov);
            let w2 = t.map_or(1.0, |t| t.weights()[0].powi(2));
            let nz = w2 * noise[0];
            let ig = |mu: f64, v: f64| -> Result<FantasyIg> {
                if !with_grad {
                    let inp = sf_inputs(mu, v, nz, w);
                    return Ok((mes_information_gain(&inp, w, max_values, z)?, None));
                }
                let mu_d = Dual::<2>::variable(mu, 0, 2);
                let v_d = Dual::<2>::variable(v, 1, 2);
                let inp = sf_inputs(mu_d, v_d, nz, w);
                let r = mes_information_gain(&inp, w, max_values, z)?;
                let (mb, cb) = pulled_back(
                    t,
                    Vector::from_element(1, r.g[0]),
                    Matrix::from_element(1, 1, r.g[1]),
                );
                Ok((r.v, Some(source.vjp(x, &sp, &mb, &cb)?)))
            };
            return ig(m[0], c[(0, 0)]);
        }
        // Multi-fidelity: the noiseless joint posterior of [xe; xt] and the
        // noisy posterior of xe.
        let x_all = vstack(&[&xe, &xt], d);
        let sp_all = source.posterior(&x_all)?;
        let sp_m = source.posterior(&xe)?;
        let (m_all, c_all) = transformed(t, &sp_all.mean, &sp_all.cov);
        let (m_m, mut c_m) = transformed(t, &sp_m.mean, &sp_m.cov);
        let w2 = t.map_or(1.0, |t| t.weights()[0].powi(2));
        for i in 0..k1 {
            c_m[(i, i)] += w2 * noise[i];
        }
        let last = k1;
        let ntan = 2 + 2 * k1 + k1 * (k1 + 1) / 2;
        macro_rules! with_dual {
            ($n:expr) => {{
                let mut idx = 0usize;
                let mut var = |v: f64| {
                    let r = Dual::<$n>::variable(v, idx, ntan);
                    idx += 1;
                    r
                };
                let mean_t = var(m_all[last]);
                let var_t_raw = var(c_all[(last, last)]);
                let cov_mt: Vec<Dual<$n>> = (0..k1).map(|i| var(c_all[(i, last)])).collect();
                let mean_m: Vec<Dual<$n>> = (0..k1).map(|i| var(m_m[i])).collect();
                let mut cov_m = vec![vec![mean_t.lift(0.0); k1]; k1];
                for i in 0..k1 {
                    for j in i..k1 {
                        let v = var(c_m[(i, j)]);
                        cov_m[i][j] = v;
                        cov_m[j][i] = v;
                    }
                }
                let inp = MesInputs {
                    mean_target: mean_t * w,
                    var_target: var_t_raw.clamp_min(MIN_VARIANCE).clamp_min(CLAMP_LB),
                    cov_mt,
                    mean_m,
                    cov_m,
                };
                let r = mes_information_gain(&inp, w, max_values, z)?;
                let g = r.grad();
                // Adjoints of the joint noiseless posterior and of the noisy
                // posterior of the expanded points.
                let n_all = k1 + 1;
                let mut mb_all = Vector::zeros(n_all);
                let mut cb_all = Matrix::zeros(n_all, n_all);
                mb_all[last] = g[0];
                cb_all[(last, last)] = g[1];
                for i in 0..k1 {
                    cb_all[(i, last)] = g[2 + i];
                }
                let mut mb_m = Vector::zeros(k1);
                let mut cb_m = Matrix::zeros(k1, k1);
                for i in 0..k1 {
                    mb_m[i] = g[2 + k1 + i];
                }
                let mut p = 2 + 2 * k1;
                for i in 0..k1 {
                    for j in i..k1 {
                        cb_m[(i, j)] = g[p];
                        p += 1;
                    }
                }
                (r.v, mb_all, cb_all, mb_m, cb_m)
            }};
        }
        if !with_grad {
            let mut cov_m = vec![vec![0.0; k1]; k1];
            for i in 0..k1 {
                for j in 0..k1 {
                    cov_m[i][j] = c_m[(i, j)];
                }
            }
            let inp = MesInputs {
                mean_target: m_all[last] * w,
                var_target: c_all[(last, last)].max(MIN_VARIANCE).max(CLAMP_LB),
                cov_mt: (0..k1).map(|i| c_all[(i, last)]).collect(),
                mean_m: (0..k1).map(|i| m_m[i]).collect(),
                cov_m,
            };
            return Ok((mes_information_gain(&inp, w, max_values, z)?, None));
        }
        let (v, mb_all, cb_all, mb_m, cb_m) = if ntan <= 6 {
            with_dual!(6)
        } else if ntan <= 16 {
            with_dual!(16)
        } else if ntan <= 40 {
            with_dual!(40)
        } else {
            return Err(BotorchError::Unsupported(
                "too many trace observations for the gradient of the multi-fidelity max-value \
                 entropy"
                    .into(),
            ));
        };
        let (mb_all, cb_all) = pulled_back(t, mb_all, cb_all);
        let (mb_m, cb_m) = pulled_back(t, mb_m, cb_m);
        let g_all = source.vjp(&x_all, &sp_all, &mb_all, &cb_all)?;
        let g_m = source.vjp(&xe, &sp_m, &mb_m, &cb_m)?;
        let f = self.fidelity.as_ref().expect("multi-fidelity");
        let mut gx = f.project.backward(x, &g_all.rows(k1, 1).into_owned())?;
        let ge = g_all.rows(0, k1).into_owned() + g_m;
        gx += match &f.expand {
            Some(e) => e.backward(x, &ge)?,
            None => ge,
        };
        Ok((v, Some(gx)))
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        check_candidates(x, self.base.model.dim(), Some(1))?;
        let max_values = self.base.max_values();
        let (xe, _) = self.points(x)?;
        let z = self.sampler.base_samples(xe.nrows())?;
        let fantasies = read_lock(&self.fantasies);
        let (values, grads): (Vec<f64>, Vec<Option<Matrix>>) = match fantasies.as_ref() {
            None => {
                let noise = self.base.noise(&xe)?;
                let (v, g) = self.source_ig(
                    &Source::Model(self.base.model.as_ref()),
                    &noise,
                    x,
                    &max_values,
                    &z,
                    with_grad,
                )?;
                (vec![v], vec![g])
            }
            Some(fz) => {
                let gp = self.gp()?;
                let fg = FantasyGp::new(gp, &fz.x_pending, Arc::clone(&fz.base), fz.num_fantasies)?;
                let noise = Vector::from_element(xe.nrows(), gp.observation_noise());
                let parts = (0..fz.num_fantasies)
                    .into_par_iter()
                    .map(|i| {
                        self.source_ig(
                            &Source::Fantasy(&fg, i),
                            &noise,
                            x,
                            &max_values,
                            &z,
                            with_grad,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                parts.into_iter().unzip()
            }
        };
        // Cost-aware utility (multi-fidelity) and the mean over fantasies.
        let (utilities, pullback) = match &self.fidelity {
            Some(f) => {
                let out = f.cost_aware_utility.evaluate(x, &values, None, with_grad)?;
                (out.values, out.pullback)
            }
            None => (values, None),
        };
        let n = utilities.len() as f64;
        let value = utilities.iter().sum::<f64>() / n;
        if !with_grad {
            return Ok((value, None));
        }
        let weights = vec![1.0 / n; utilities.len()];
        let (deltas_bar, xbar) = match pullback {
            Some(pb) => {
                let (d, xb) = pb(&weights)?;
                (d, Some(xb))
            }
            None => (weights, None),
        };
        let mut grad = xbar.unwrap_or_else(|| Matrix::zeros(1, x.ncols()));
        for (g, s) in grads.iter().zip(&deltas_bar) {
            grad += g.as_ref().expect("gradient requested") * *s;
        }
        Ok((value, Some(grad)))
    }

    /// BoTorch's `set_X_pending`: fantasizes the model at the pending points
    /// (the max-value samples are not redrawn, as in BoTorch).
    fn set_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.base.model.dim())?;
        }
        let new = match x_pending {
            Some(p) if p.nrows() > 0 => {
                self.gp()?;
                Some(Fantasies {
                    x_pending: p.clone(),
                    base: self.fantasies_sampler.base_samples(p.nrows())?,
                    num_fantasies: self.num_fantasies,
                })
            }
            _ => None,
        };
        *write_lock(&self.fantasies) = new;
        *write_lock(&self.base.x_pending) = x_pending.cloned();
        Ok(())
    }
}

/// The inputs of the single-fidelity information gain from the
/// (transformed) posterior mean `mu` and noiseless variance `v` at the
/// candidate and the (transformed) noise variance `noise`.
fn sf_inputs<S: Scalar>(mu: S, v: S, noise: f64, w: f64) -> MesInputs<S> {
    let var_target = v.clamp_min(MIN_VARIANCE).clamp_min(CLAMP_LB);
    MesInputs {
        mean_target: mu * w,
        var_target,
        cov_mt: vec![var_target],
        mean_m: vec![mu],
        cov_m: vec![vec![v + noise]],
    }
}

impl AcquisitionFunction for QMaxValueEntropy {
    fn model(&self) -> &Arc<dyn Model> {
        &self.base.model
    }

    fn expected_q(&self) -> Option<usize> {
        Some(1)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        read_lock(&self.base.x_pending).clone()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.set_pending(x_pending)
    }
}

/// Options of the multi-fidelity max-value entropy search (BoTorch's keyword
/// arguments in addition to [`MaxValueEntropyOptions`]).
#[derive(Clone, Debug, Default)]
pub struct MultiFidelityMaxValueEntropyOptions {
    /// The options shared with [`QMaxValueEntropy`].
    pub mes: MaxValueEntropyOptions,
    /// The cost-aware utility (default: `InverseCostWeightedUtility` of an
    /// `AffineFidelityCostModel(fidelity_weights={-1: 1.0})`).
    pub cost_aware_utility: Option<Arc<dyn CostAwareUtility>>,
    /// The projection onto the target fidelity (default: the identity), also
    /// applied to the candidate set.
    pub project: Option<Arc<dyn InputProjection>>,
    /// The expansion of a candidate by trace observations (default: none).
    pub expand: Option<Arc<dyn InputProjection>>,
}

/// BoTorch's default cost-aware utility of the multi-fidelity max-value
/// entropy search.
fn default_cost_utility(d: usize) -> Result<Arc<dyn CostAwareUtility>> {
    let cost = crate::models::cost::AffineFidelityCostModel::new(Some(&[(-1, 1.0)]), 0.01)?;
    let _ = d;
    Ok(Arc::new(
        crate::acquisition::cost_aware::InverseCostWeightedUtility::new(
            Arc::new(cost),
            true,
            None,
            false,
        )?,
    ))
}

fn fidelity_of(
    d: usize,
    cost_aware_utility: Option<Arc<dyn CostAwareUtility>>,
    project: Option<Arc<dyn InputProjection>>,
    expand: Option<Arc<dyn InputProjection>>,
) -> Result<Fidelity> {
    Ok(Fidelity {
        cost_aware_utility: match cost_aware_utility {
            Some(u) => u,
            None => default_cost_utility(d)?,
        },
        project: project
            .unwrap_or_else(|| Arc::new(crate::acquisition::fidelity::IdentityProjection)),
        expand,
    })
}

/// BoTorch's `qMultiFidelityMaxValueEntropy`: the max-value entropy search
/// about the maximum at the target fidelity (`project`) of observations at
/// the candidate and its trace observations (`expand`), weighted by a
/// cost-aware utility (see the [module documentation](self)).
#[derive(Debug)]
pub struct QMultiFidelityMaxValueEntropy {
    inner: QMaxValueEntropy,
}

impl QMultiFidelityMaxValueEntropy {
    /// `qMultiFidelityMaxValueEntropy(model, candidate_set, num_fantasies,
    /// num_mv_samples, num_y_samples, posterior_transform, use_gumbel,
    /// maximize, X_pending, cost_aware_utility, project, expand)`; the
    /// candidate set is projected first (it may omit the fidelity columns if
    /// the projection inserts them).
    ///
    /// # Errors
    /// See [`QMaxValueEntropy::new`].
    pub fn new(
        model: Arc<dyn Model>,
        candidate_set: &Matrix,
        options: MultiFidelityMaxValueEntropyOptions,
    ) -> Result<Self> {
        let d = model.dim();
        let fidelity = fidelity_of(
            d,
            options.cost_aware_utility,
            options.project,
            options.expand,
        )?;
        let cs = fidelity.project.project(candidate_set)?;
        Ok(Self {
            inner: QMaxValueEntropy::build(
                model,
                &cs,
                options.mes,
                Some(fidelity),
                "qMultiFidelityMaxValueEntropy",
            )?,
        })
    }

    /// The underlying max-value entropy search (its accessors).
    pub fn as_mes(&self) -> &QMaxValueEntropy {
        &self.inner
    }
}

impl AcquisitionFunction for QMultiFidelityMaxValueEntropy {
    fn model(&self) -> &Arc<dyn Model> {
        self.inner.model()
    }

    fn expected_q(&self) -> Option<usize> {
        Some(1)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        self.inner.forward(x)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        self.inner.forward_with_grad(x)
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.inner.x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.inner.set_x_pending(x_pending)
    }
}

// ---------------------------------------------------------------------------
// GIBBON
// ---------------------------------------------------------------------------

/// The pending points of GIBBON with the noisy covariance of their
/// observations (BoTorch's `B`) and its factor.
#[derive(Debug)]
struct Repulsion {
    x_pending: Matrix,
    chol: crate::linalg::CholeskyFactor,
}

/// BoTorch's `qLowerBoundMaxValueEntropy` (GIBBON): see the [module
/// documentation](self).
#[derive(Debug)]
pub struct QLowerBoundMaxValueEntropy {
    base: MaxValueBase,
    repulsion: RwLock<Option<Repulsion>>,
    fidelity: Option<Fidelity>,
}

impl QLowerBoundMaxValueEntropy {
    /// `qLowerBoundMaxValueEntropy(model, candidate_set, num_mv_samples,
    /// posterior_transform, use_gumbel, maximize, X_pending, train_inputs)`.
    ///
    /// # Errors
    /// See [`QMaxValueEntropy::new`].
    pub fn new(
        model: Arc<dyn Model>,
        candidate_set: &Matrix,
        options: MaxValueEntropyOptions,
    ) -> Result<Self> {
        let base = MaxValueBase::new(
            model,
            candidate_set,
            &options,
            "qLowerBoundMaxValueEntropy",
            None,
        )?;
        let acqf = Self {
            base,
            repulsion: RwLock::new(None),
            fidelity: None,
        };
        acqf.set_pending(options.x_pending.as_ref())?;
        Ok(acqf)
    }

    /// The max-value samples (BoTorch's `posterior_max_values`).
    pub fn posterior_max_values(&self) -> Vec<f64> {
        self.base.max_values()
    }

    /// Sets the max-value samples.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for no or NaN values.
    pub fn set_posterior_max_values(&self, values: Vec<f64>) -> Result<()> {
        self.base.set_max_values(values)
    }

    /// The candidate set (with the training inputs).
    pub fn candidate_set(&self) -> &Matrix {
        &self.base.candidate_set
    }

    fn set_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.base.model.dim())?;
        }
        let rep = match x_pending {
            Some(p) if p.nrows() > 0 => {
                let b = self.base.model.posterior(p, true)?;
                Some(Repulsion {
                    x_pending: p.clone(),
                    chol: crate::linalg::psd_safe_cholesky(&b.covariance)?,
                })
            }
            _ => None,
        };
        *write_lock(&self.repulsion) = rep;
        *write_lock(&self.base.x_pending) = x_pending.cloned();
        Ok(())
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let model = self.base.model.as_ref();
        check_candidates(x, model.dim(), Some(1))?;
        let t = self.base.transform.as_ref();
        let w = self.base.weight;
        let max_values = self.base.max_values();
        let w2 = t.map_or(1.0, |t| t.weights()[0].powi(2));
        let rep = read_lock(&self.repulsion);
        if rep.is_some() && t.is_some() {
            return Err(BotorchError::Unsupported(
                "qLowerBoundMaxValueEntropy does not support PosteriorTransformswhen X_pending \
                 is not None."
                    .into(),
            ));
        }
        let d = x.ncols();
        // The target point (the projection) and the candidate.
        let xt = match &self.fidelity {
            Some(f) => f.project.project(x)?,
            None => x.clone(),
        };
        let mf = self.fidelity.is_some();
        // Points of the joint noiseless posterior: the candidate, the target
        // point (multi-fidelity) and the pending points (repulsion).
        let mut parts: Vec<&Matrix> = vec![x];
        if mf {
            parts.push(&xt);
        }
        if let Some(r) = rep.as_ref() {
            parts.push(&r.x_pending);
        }
        let pts = vstack(&parts, d);
        let post = model.posterior(&pts, false)?;
        let (mean, cov) = transformed(t, &post.mean, &post.covariance);
        let noise = w2 * self.base.noise(x)?[0];
        let np = pts.nrows();
        let first_pending = if mf { 2 } else { 1 };
        // Tangents: the mean and variance at x, the target mean, variance and
        // covariance (multi-fidelity) and the cross-covariances with the
        // pending points (repulsion).
        let ntan = 2 + if mf { 3 } else { 0 } + (np - first_pending);
        let value_with = |vals: &[f64]| -> Result<f64> {
            self.gibbon(vals, mf, rep.as_ref(), noise, w, &max_values)
        };
        let mut vals = vec![mean[0], cov[(0, 0)]];
        if mf {
            vals.extend([mean[1], cov[(1, 1)], cov[(0, 1)]]);
        }
        vals.extend((first_pending..np).map(|j| cov[(0, j)]));
        let ig = value_with(&vals)?;
        // Cost-aware utility (multi-fidelity).
        let (value, pullback) = match &self.fidelity {
            Some(f) => {
                let out = f.cost_aware_utility.evaluate(x, &[ig], None, with_grad)?;
                (out.values[0], out.pullback)
            }
            None => (ig, None),
        };
        if !with_grad {
            return Ok((value, None));
        }
        macro_rules! grad_with {
            ($n:expr) => {{
                let duals: Vec<Dual<$n>> = vals
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| Dual::<$n>::variable(v, i, ntan))
                    .collect();
                self.gibbon(&duals, mf, rep.as_ref(), noise, w, &max_values)?
            }};
        }
        let g: Vec<f64> = if ntan <= 6 {
            grad_with!(6).grad().to_vec()
        } else if ntan <= 16 {
            grad_with!(16).grad().to_vec()
        } else if ntan <= 64 {
            grad_with!(64).grad().to_vec()
        } else {
            return Err(BotorchError::Unsupported(
                "too many pending points for the gradient of GIBBON".into(),
            ));
        };
        let (scale, xbar) = match pullback {
            Some(pb) => {
                let (d_bar, xb) = pb(&[1.0])?;
                (d_bar[0], Some(xb))
            }
            None => (1.0, None),
        };
        let mut mb = Vector::zeros(np);
        let mut cb = Matrix::zeros(np, np);
        mb[0] = g[0] * scale;
        cb[(0, 0)] = g[1] * scale;
        let mut gi = 2;
        if mf {
            mb[1] = g[2] * scale;
            cb[(1, 1)] = g[3] * scale;
            cb[(0, 1)] = g[4] * scale;
            gi = 5;
        }
        for j in first_pending..np {
            cb[(0, j)] = g[gi] * scale;
            gi += 1;
        }
        let (mb, cb) = pulled_back(t, mb, cb);
        let gp = model.joint_posterior_vjp(&pts, 0..np, &mb, &cb)?;
        let mut grad = gp.rows(0, 1).into_owned();
        if let Some(f) = &self.fidelity {
            grad += f.project.backward(x, &gp.rows(1, 1).into_owned())?;
        }
        if let Some(xb) = xbar {
            grad += xb;
        }
        Ok((value, Some(grad)))
    }

    /// GIBBON's value from the posterior quantities `vals` (see
    /// [`QLowerBoundMaxValueEntropy::evaluate`]).
    fn gibbon<S: Scalar>(
        &self,
        vals: &[S],
        mf: bool,
        rep: Option<&Repulsion>,
        noise: f64,
        w: f64,
        max_values: &[f64],
    ) -> Result<S> {
        let (mu, v) = (vals[0], vals[1]);
        // The noisy variance at the candidate (clamped by the posterior and
        // at CLAMP_LB).
        let var_m = (v + noise).clamp_min(MIN_VARIANCE).clamp_min(CLAMP_LB);
        let (mean_t, var_t, cov_mt) = if mf {
            let vt = vals[3].clamp_min(MIN_VARIANCE).clamp_min(CLAMP_LB);
            (vals[2] * w, vt, vals[4])
        } else {
            let vt = v.clamp_min(MIN_VARIANCE).clamp_min(CLAMP_LB);
            (mu * w, vt, vt)
        };
        let acq = gibbon_information_gain(mean_t, var_t, cov_mt, var_m, max_values);
        let Some(r) = rep else {
            return Ok(acq);
        };
        // The repulsion term 0.5 log(1 - A B⁻¹ Aᵀ / var_m).
        let a: Vec<S> = vals[if mf { 5 } else { 2 }..].to_vec();
        let l = r.chol.l();
        let m = a.len();
        let lower: Vec<Vec<S>> = (0..m)
            .map(|i| (0..m).map(|j| mu.lift(l[(i, j)])).collect())
            .collect();
        let u = solve_lower(&lower, &a);
        let mut inv_quad = mu.lift(0.0);
        for ui in &u {
            inv_quad = inv_quad + *ui * *ui;
        }
        let eps = 1e-12;
        let v_det = var_m.clamp_min_var(inv_quad * (1.0 + eps)) - inv_quad;
        Ok(acq + (v_det.ln() - var_m.ln()) * 0.5)
    }
}

impl AcquisitionFunction for QLowerBoundMaxValueEntropy {
    fn model(&self) -> &Arc<dyn Model> {
        &self.base.model
    }

    fn expected_q(&self) -> Option<usize> {
        Some(1)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        read_lock(&self.base.x_pending).clone()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.set_pending(x_pending)
    }
}

/// BoTorch's `qMultiFidelityLowerBoundMaxValueEntropy`: GIBBON about the
/// maximum at the target fidelity (`project`), weighted by a cost-aware
/// utility; trace observations are not supported (as in BoTorch). Pending
/// points repulse the candidates as in GIBBON; unlike BoTorch main, the
/// `X_pending` of the constructor is used as well (BoTorch drops it: its
/// constructor does not pass it on), both for the max-value samples and the
/// repulsion.
#[derive(Debug)]
pub struct QMultiFidelityLowerBoundMaxValueEntropy {
    inner: QLowerBoundMaxValueEntropy,
}

impl QMultiFidelityLowerBoundMaxValueEntropy {
    /// `qMultiFidelityLowerBoundMaxValueEntropy(model, candidate_set,
    /// num_fantasies, num_mv_samples, num_y_samples, X_pending,
    /// posterior_transform, use_gumbel, maximize, cost_aware_utility,
    /// project)`.
    ///
    /// # Errors
    /// See [`QMaxValueEntropy::new`]; [`BotorchError::Unsupported`] for an
    /// expansion (BoTorch's message).
    pub fn new(
        model: Arc<dyn Model>,
        candidate_set: &Matrix,
        options: MultiFidelityMaxValueEntropyOptions,
    ) -> Result<Self> {
        if options.expand.is_some() {
            return Err(BotorchError::Unsupported(
                "qMultiFidelityLowerBoundMaxValueEntropy does not support trace observations. "
                    .into(),
            ));
        }
        let d = model.dim();
        let fidelity = fidelity_of(d, options.cost_aware_utility, options.project, None)?;
        let cs = fidelity.project.project(candidate_set)?;
        let base = MaxValueBase::new(
            model,
            &cs,
            &options.mes,
            "qMultiFidelityLowerBoundMaxValueEntropy",
            Some(&fidelity.project),
        )?;
        let inner = QLowerBoundMaxValueEntropy {
            base,
            repulsion: RwLock::new(None),
            fidelity: Some(fidelity),
        };
        inner.set_pending(options.mes.x_pending.as_ref())?;
        Ok(Self { inner })
    }

    /// The underlying GIBBON (its accessors).
    pub fn as_gibbon(&self) -> &QLowerBoundMaxValueEntropy {
        &self.inner
    }
}

impl AcquisitionFunction for QMultiFidelityLowerBoundMaxValueEntropy {
    fn model(&self) -> &Arc<dyn Model> {
        self.inner.model()
    }

    fn expected_q(&self) -> Option<usize> {
        Some(1)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        self.inner.forward(x)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        self.inner.forward_with_grad(x)
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.inner.x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.inner.set_x_pending(x_pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_control_variance_and_single_observation_remain_finite() {
        // A distant threshold makes the truncation ratio exactly one. At
        // zero normal draws H0 - (-log p(mean)) = 1/2, independently of
        // Gaussian mean/variance. No control variate is defined or needed.
        for observations in [1, 8] {
            let normals = Matrix::zeros(observations, 1);
            let value =
                mes_information_gain(&sf_inputs(0.0, 0.3, 0.7, 1.0), 1.0, &[100.0; 4], &normals)
                    .unwrap();
            assert!((value - 0.5).abs() < 1e-14);
            let derivative = mes_information_gain(
                &sf_inputs(
                    Dual::<2>::variable(0.0, 0, 2),
                    Dual::<2>::variable(0.3, 1, 2),
                    0.7,
                    1.0,
                ),
                1.0,
                &[100.0; 4],
                &normals,
            )
            .unwrap();
            assert!((derivative.v - value).abs() < 1e-14);
            assert!(derivative.grad().iter().all(|g| g.abs() < 1e-14));
        }
    }

    #[test]
    fn brentq_finds_roots() {
        let r = brentq(|x| x * x - 2.0, 0.0, 2.0, BRENTQ_XTOL, BRENTQ_RTOL, 100).unwrap();
        assert!((r - 2f64.sqrt()).abs() < 1e-11);
        // SciPy fails to converge on this triple root too.
        assert!(brentq(
            |x| (x - 0.3).powi(3),
            -1.0,
            1.0,
            BRENTQ_XTOL,
            BRENTQ_RTOL,
            100
        )
        .is_err());
        let r = brentq(|x| x.exp() - 2.0, -3.0, 4.0, BRENTQ_XTOL, BRENTQ_RTOL, 100).unwrap();
        assert!((r - 2f64.ln()).abs() < 1e-11);
        assert!(brentq(|x| x * x + 1.0, -1.0, 1.0, BRENTQ_XTOL, BRENTQ_RTOL, 100).is_err());
    }

    #[test]
    fn information_gain_gradients_match_finite_differences() {
        let z = Matrix::from_fn(16, 1, |k, _| ((k as f64 + 0.5) / 16.0 - 0.5) * 3.0);
        let mvs = [1.2, 1.5, 0.9, 2.0];
        let f = |mu: f64, v: f64| {
            mes_information_gain(&sf_inputs(mu, v, 0.01, 1.0), 1.0, &mvs, &z).unwrap()
        };
        let (mu, v) = (0.8, 0.3);
        let d = mes_information_gain(
            &sf_inputs(
                Dual::<2>::variable(mu, 0, 2),
                Dual::<2>::variable(v, 1, 2),
                0.01,
                1.0,
            ),
            1.0,
            &mvs,
            &z,
        )
        .unwrap();
        let h = 1e-6;
        let gm = (f(mu + h, v) - f(mu - h, v)) / (2.0 * h);
        let gv = (f(mu, v + h) - f(mu, v - h)) / (2.0 * h);
        assert!((d.v - f(mu, v)).abs() < 1e-14);
        assert!(
            (d.g[0] - gm).abs() < 1e-7 * gm.abs().max(1.0),
            "{} {gm}",
            d.g[0]
        );
        assert!(
            (d.g[1] - gv).abs() < 1e-7 * gv.abs().max(1.0),
            "{} {gv}",
            d.g[1]
        );
        let g = |mu: f64, v: f64| {
            let vt = v.max(CLAMP_LB);
            gibbon_information_gain(mu, vt, vt, v + 0.01, &mvs)
        };
        let dg = {
            let mu_d = Dual::<2>::variable(mu, 0, 2);
            let v_d = Dual::<2>::variable(v, 1, 2);
            let vt = v_d.clamp_min(CLAMP_LB);
            gibbon_information_gain(mu_d, vt, vt, v_d + 0.01, &mvs)
        };
        let gm = (g(mu + h, v) - g(mu - h, v)) / (2.0 * h);
        let gv = (g(mu, v + h) - g(mu, v - h)) / (2.0 * h);
        assert!((dg.g[0] - gm).abs() < 1e-7 * gm.abs().max(1.0));
        assert!((dg.g[1] - gv).abs() < 1e-7 * gv.abs().max(1.0));
    }
}
