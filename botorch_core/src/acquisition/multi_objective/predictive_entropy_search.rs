//! Predictive entropy search for multi-objective Bayesian optimization
//! (`botorch.acquisition.multi_objective.predictive_entropy_search`;
//! Garrido-Merchán & Hernández-Lobato, *Predictive Entropy Search for
//! Multi-objective Bayesian Optimization with Constraints*, Neurocomputing
//! 2019, supplementary material).
//!
//! [`QMultiObjectivePredictiveEntropySearch`] approximates the mutual
//! information between the observations at the candidates and the sampled
//! Pareto optimal inputs by expectation propagation (EP): the condition that
//! no point dominates the Pareto points is a product of factors
//! `ω(f(x_n), f(x_p))`, approximated by Gaussian factors in natural
//! parameters. EP runs once at construction over the training inputs and the
//! Pareto sets; every evaluation adds the candidates, performs one EP update
//! of their factors, reuses the cached factors of the Pareto points and
//! returns half the difference of the log determinants of the predictive
//! covariances at the candidates before and after.
//!
//! The computation follows BoTorch's step by step: the same jitters, the
//! successive halving of the damping factor until the damped natural
//! covariance has a Cholesky factorization (with BoTorch's test of the
//! factorization's `info` codes), the convergence test on the relative
//! changes of the means and covariances, the fallback to the previous
//! factors when an update produces NaNs or a singular 2 × 2 matrix, and the
//! clamps of `log_cdf_robust`. Gradients are the exact derivatives of the
//! value with the damping factors held fixed, as `torch.autograd` computes
//! them; where BoTorch's autograd differentiates through a clamped logarithm
//! of zero and returns NaN, botorch_core returns the derivative of the
//! clamped function (zero).
//!
//! Differences from BoTorch main:
//!
//! * EP uses the raw training inputs. BoTorch reads `model.train_inputs`,
//!   which models in eval mode (e.g. after fitting) store transformed, and
//!   transforms them again in `model.posterior`, so with an input transform
//!   its EP runs at the wrong inputs (a bug without a pull request yet; in
//!   train mode BoTorch uses the raw inputs, as here).
//! * Every candidate set is evaluated on its own. BoTorch evaluates a batch
//!   of candidate sets jointly, and a NaN in the update of one candidate set
//!   (or the failure of a Cholesky factorization of one set) changes the
//!   fallback and damping decisions of all others.
//!
//! As in BoTorch, the observation noise at the candidates enters the
//! predictive covariance of the factors (BoTorch computes both predictive
//! covariances with `observation_noise=True`, so its separately computed
//! observation noise is zero), and the training EP works with the noiseless
//! posterior.

use std::sync::{Arc, RwLock};

use rayon::prelude::*;

use crate::acquisition::multi_objective::entropy_linalg::{cholesky_ex, cholesky_inverse, Lu};
use crate::acquisition::multi_objective::joint_entropy_search::{
    normal_cdf, normal_cdf_grad, normal_log_prob, CLAMP_LB,
};
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::{Model, SingleTaskGP};
use crate::{Matrix, Vector};

/// `torch.finfo(torch.float64).min`, BoTorch's `NEG_INF` of PES.
const NEG_INF: f64 = f64::MIN;

/// The lower bound of the damping factor below which it is set to zero.
const MIN_DAMPING: f64 = 1e-3;

/// BoTorch's error when the previous natural covariance of EP has no
/// Cholesky factorization.
const NOT_PSD_MESSAGE: &str = "The previous covariance is not positive semi-definite. This \
                               usually happens if the predictive covariance is ill-conditioned \
                               and the added jitter is insufficient.";

/// BoTorch's `check_no_nans` error.
const NAN_MESSAGE: &str = "Input data contains NaN values.";

/// The logarithm of the standard normal CDF, computed robustly (BoTorch's
/// `log_cdf_robust`): `log Φ(x)` for `x < 5` and `-Φ(-x)` (`log(1 - z) ≈ -z`)
/// otherwise, clamped to `[f64::MIN, -eps]`.
pub fn log_cdf_robust(x: f64) -> f64 {
    let v = if x < 5.0 {
        normal_cdf(x).ln()
    } else {
        -normal_cdf(-x)
    };
    v.clamp(NEG_INF, -CLAMP_LB)
}

/// [`log_cdf_robust`] and its derivative (zero where clamped).
fn log_cdf_robust_with_grad(x: f64) -> (f64, f64) {
    let (raw, grad) = if x < 5.0 {
        let c = normal_cdf(x);
        (c.ln(), normal_cdf_grad(x) / c)
    } else {
        (-normal_cdf(-x), normal_cdf_grad(-x))
    };
    let v = raw.clamp(NEG_INF, -CLAMP_LB);
    // Scalar bounds: no gradient at either bound (torch ≥ 2.14).
    let active = NEG_INF < raw && raw < -CLAMP_LB;
    (v, if active { grad } else { 0.0 })
}

// ---------------------------------------------------------------------------
// 2 × 2 matrices (row-major [a00, a01, a10, a11])
// ---------------------------------------------------------------------------

type M2 = [f64; 4];
const I2: M2 = [1.0, 0.0, 0.0, 1.0];

/// `torch.linalg.inv` of a 2 × 2 matrix: LU with partial pivoting and the
/// solves of the identity; `None` for an exactly zero pivot (torch raises).
fn inv2(a: &M2) -> Option<M2> {
    // Row pivoting (the first row of largest magnitude, `idamax`).
    let swap = a[2].abs() > a[0].abs() || (a[0].is_nan() && !a[2].is_nan());
    let (r0, r1, perm) = if swap {
        ([a[2], a[3]], [a[0], a[1]], [1usize, 0])
    } else {
        ([a[0], a[1]], [a[2], a[3]], [0usize, 1])
    };
    let u00 = r0[0];
    let u01 = r0[1];
    let (l10, u11) = if u00 == 0.0 {
        (r1[0], r1[1])
    } else {
        let l = r1[0] / u00;
        (l, r1[1] - l * u01)
    };
    if u00 == 0.0 || u11 == 0.0 {
        return None;
    }
    let mut out = [0.0; 4];
    for c in 0..2 {
        // x = P e_c, forward substitution with the unit lower factor, then
        // backward substitution with U.
        let b = |i: usize| if perm[i] == c { 1.0 } else { 0.0 };
        let x0 = b(0);
        let mut x1 = b(1) - l10 * x0;
        x1 /= u11;
        let x0 = (x0 - u01 * x1) / u00;
        out[c] = x0;
        out[2 + c] = x1;
    }
    Some(out)
}

#[inline]
fn mv2(a: &M2, v: &[f64; 2]) -> [f64; 2] {
    [a[0] * v[0] + a[1] * v[1], a[2] * v[0] + a[3] * v[1]]
}

#[inline]
fn mm2(a: &M2, b: &M2) -> M2 {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

#[inline]
fn tr2(a: &M2) -> M2 {
    [a[0], a[2], a[1], a[3]]
}

#[inline]
fn add_jitter(a: &M2, j: f64) -> M2 {
    [a[0] + j, a[1], a[2], a[3] + j]
}

/// Adjoint of `X = A⁻¹` given `X̄`: `-Xᵀ X̄ Xᵀ`.
#[inline]
fn inv2_backward(x: &M2, x_bar: &M2) -> M2 {
    let xt = tr2(x);
    mm2(&mm2(&xt, x_bar), &xt).map(|v| -v)
}

// ---------------------------------------------------------------------------
// The EP factor update (`_update_omega`)
// ---------------------------------------------------------------------------

/// The inputs of the factor update of one (row, Pareto point) pair for one
/// output: the means of `f(x_r)` and `f(x_p)`, their 2 × 2 covariance (both
/// off-diagonal entries are `Σ[r, p]`), the old factors and whether the pair
/// is on the Pareto diagonal (`x_r = x_p`).
#[derive(Clone, Copy)]
struct PairInput {
    m: [f64; 2],
    c: M2,
    om: [f64; 2],
    oc: M2,
    diag: bool,
}

/// The forward quantities of the factor update of one pair and output (for
/// the backward pass).
#[derive(Clone, Copy, Default)]
struct PairTerms {
    m: [f64; 2],
    n2: M2,
    cnc: M2,
    cnm: [f64; 2],
    cc: M2,
    v: f64,
    v_active: bool,
    sd: f64,
    sd_active: bool,
    alpha: f64,
    lc_grad: f64,
    lp_active: bool,
    rho: f64,
    rho_active: bool,
    h: f64,
    unc: M2,
    om_new: [f64; 2],
    oc_new: M2,
}

/// `_update_omega` for one pair over all outputs (`inputs[t]`): the new
/// factors of every output, or `None` if a 2 × 2 matrix is singular
/// (`torch.linalg.inv` raises, and `_safe_update_omega` keeps the old
/// factors).
fn update_pair(inputs: &[PairInput], maximize: bool, jitter: f64) -> Option<Vec<PairTerms>> {
    let w = if maximize { 1.0 } else { -1.0 };
    let mut terms = Vec::with_capacity(inputs.len());
    let mut lcs = Vec::with_capacity(inputs.len());
    let mut lps = Vec::with_capacity(inputs.len());
    for inp in inputs {
        let c2 = if inp.diag {
            I2
        } else {
            add_jitter(&inp.c, jitter)
        };
        let n2 = inv2(&c2)?;
        let nm2 = mv2(&n2, &inp.m);
        let cnm = [nm2[0] - inp.om[0], nm2[1] - inp.om[1]];
        let cnc = if inp.diag {
            I2
        } else {
            [
                n2[0] - inp.oc[0],
                n2[1] - inp.oc[1],
                n2[2] - inp.oc[2],
                n2[3] - inp.oc[3],
            ]
        };
        let cc = inv2(&add_jitter(&cnc, jitter))?;
        let cm = mv2(&cc, &cnm);
        let raw_v = cc[0] + cc[3] - 2.0 * cc[1];
        let (v, v_active) = clamp_min_active(raw_v, CLAMP_LB);
        let (sd, sd_active) = clamp_min_active(v.sqrt(), CLAMP_LB);
        let alpha = w * (cm[0] - cm[1]) / sd;
        let (lc, lc_grad) = log_cdf_robust_with_grad(alpha);
        let (lp, lp_active) = clamp_min_active(normal_log_prob(alpha), NEG_INF);
        lcs.push(lc);
        lps.push(lp);
        terms.push(PairTerms {
            m: inp.m,
            n2,
            cnc,
            cnm,
            cc,
            v,
            v_active,
            sd,
            sd_active,
            alpha,
            lc_grad,
            lp_active,
            ..Default::default()
        });
    }
    let (s, _) = clamp_min_active(lcs.iter().sum(), NEG_INF);
    let (lz, _) = clamp_min_active((-s.exp()).ln_1p(), NEG_INF);
    for (t, tm) in terms.iter_mut().enumerate() {
        let lr = -lz + s - lcs[t] + lps[t];
        let e = lr.exp();
        let rho_active = NEG_INF < e && e < f64::MAX;
        let rho = -e.clamp(NEG_INF, f64::MAX);
        let g1 = w * (rho / tm.sd);
        let dm = [g1, -g1];
        let h = -rho * (rho + tm.alpha) / tm.v;
        let d2 = [h, -h, -h, h];
        let cm = mv2(&tm.cc, &tm.cnm);
        let step = mv2(&tm.cc, &dm);
        let um = [cm[0] + step[0], cm[1] + step[1]];
        let corr = mm2(&mm2(&tm.cc, &d2), &tm.cc);
        let uc = if inputs[t].diag {
            I2
        } else {
            [
                tm.cc[0] + corr[0],
                tm.cc[1] + corr[1],
                tm.cc[2] + corr[2],
                tm.cc[3] + corr[3],
            ]
        };
        let unc = inv2(&add_jitter(&uc, jitter))?;
        let unm = mv2(&unc, &um);
        tm.rho = rho;
        tm.rho_active = rho_active;
        tm.h = h;
        tm.unc = unc;
        tm.om_new = [unm[0] - tm.cnm[0], unm[1] - tm.cnm[1]];
        tm.oc_new = [
            unc[0] - tm.cnc[0],
            unc[1] - tm.cnc[1],
            unc[2] - tm.cnc[2],
            unc[3] - tm.cnc[3],
        ];
    }
    Some(terms)
}

/// The adjoints of the means (`[m_r, m_p]`) and of the covariance entries
/// `[Σ_rr, Σ_rp, Σ_pp]` of every output of one pair, given the adjoints of
/// the new natural covariance factors (`oc_bar[t]`). The new mean factors
/// do not enter the value of PES and are not differentiated.
fn update_pair_backward(
    terms: &[PairTerms],
    oc_bar: &[M2],
    maximize: bool,
) -> Vec<([f64; 2], [f64; 3])> {
    let w = if maximize { 1.0 } else { -1.0 };
    let m = terms.len();
    let s_raw: f64 = terms.iter().map(|t| log_cdf_robust(t.alpha)).sum();
    let (s, s_active) = clamp_min_active(s_raw, NEG_INF);
    let lz_raw = (-s.exp()).ln_1p();
    let lz_active = lz_raw > NEG_INF;
    let mut alpha_bar = vec![0.0; m];
    let mut v_bar = vec![0.0; m];
    let mut cc_bar = vec![[0.0; 4]; m];
    let mut cnc_bar = vec![[0.0; 4]; m];
    let mut lr_bar = vec![0.0; m];
    for t in 0..m {
        let tm = &terms[t];
        // oc_new = unc - cnc; unc = (uc + j I)⁻¹.
        let unc_bar = oc_bar[t];
        for k in 0..4 {
            cnc_bar[t][k] -= oc_bar[t][k];
        }
        let uc_bar = inv2_backward(&tm.unc, &unc_bar);
        // uc = cc + cc d2 cc (identity on the Pareto diagonal, not
        // differentiated: those factors are zeroed).
        let d2 = [tm.h, -tm.h, -tm.h, tm.h];
        let d2cc = mm2(&d2, &tm.cc);
        let ccd2 = mm2(&tm.cc, &d2);
        let a = mm2(&uc_bar, &tr2(&d2cc));
        let b = mm2(&tr2(&ccd2), &uc_bar);
        for k in 0..4 {
            cc_bar[t][k] += uc_bar[k] + a[k] + b[k];
        }
        let d2_bar = mm2(&mm2(&tr2(&tm.cc), &uc_bar), &tr2(&tm.cc));
        let h_bar = d2_bar[0] - d2_bar[1] - d2_bar[2] + d2_bar[3];
        // h = -ρ (ρ + α) / v.
        let rho_bar = h_bar * (-(tm.rho + tm.alpha) - tm.rho) / tm.v;
        alpha_bar[t] += h_bar * -tm.rho / tm.v;
        v_bar[t] -= h_bar * tm.h / tm.v;
        // ρ = -clamp(exp(lr)).
        if tm.rho_active {
            lr_bar[t] = rho_bar * tm.rho;
        }
    }
    // lr_t = -lz + S - lc_t + lp_t, S = Σ_t lc_t, lz = log1p(-exp(S)).
    let mut lz_bar = 0.0;
    let mut s_bar = 0.0;
    let mut lc_bar = vec![0.0; m];
    for t in 0..m {
        lz_bar -= lr_bar[t];
        s_bar += lr_bar[t];
        lc_bar[t] -= lr_bar[t];
        if terms[t].lp_active {
            alpha_bar[t] += lr_bar[t] * -terms[t].alpha;
        }
    }
    if lz_active {
        s_bar += lz_bar * (1.0 / (1.0 + -s.exp())) * -s.exp();
    }
    if s_active {
        for v in lc_bar.iter_mut() {
            *v += s_bar;
        }
    }
    let mut out = Vec::with_capacity(m);
    for t in 0..m {
        let tm = &terms[t];
        alpha_bar[t] += lc_bar[t] * tm.lc_grad;
        // α = md / sd.
        let md = tm.alpha * tm.sd;
        let md_bar = alpha_bar[t] / tm.sd;
        let sd_bar = -alpha_bar[t] * (md / tm.sd) / tm.sd;
        if tm.sd_active {
            v_bar[t] += sd_bar / (2.0 * tm.v.sqrt());
        }
        let mut ccb = cc_bar[t];
        if tm.v_active {
            ccb[0] += v_bar[t];
            ccb[3] += v_bar[t];
            ccb[1] -= 2.0 * v_bar[t];
        }
        // md = w (cm0 - cm1), cm = cc cnm.
        let cm_bar = [w * md_bar, -w * md_bar];
        ccb[0] += cm_bar[0] * tm.cnm[0];
        ccb[1] += cm_bar[0] * tm.cnm[1];
        ccb[2] += cm_bar[1] * tm.cnm[0];
        ccb[3] += cm_bar[1] * tm.cnm[1];
        let cnm_bar = mv2(&tr2(&tm.cc), &cm_bar);
        // cc = (cnc + j I)⁻¹; cnc = n2 (no old factors at test time).
        let e_bar = inv2_backward(&tm.cc, &ccb);
        let mut n2_bar = [0.0; 4];
        for k in 0..4 {
            n2_bar[k] = cnc_bar[t][k] + e_bar[k];
        }
        // cnm = nm2 = n2 m.
        n2_bar[0] += cnm_bar[0] * tm.m[0];
        n2_bar[1] += cnm_bar[0] * tm.m[1];
        n2_bar[2] += cnm_bar[1] * tm.m[0];
        n2_bar[3] += cnm_bar[1] * tm.m[1];
        let m_bar = mv2(&tr2(&tm.n2), &cnm_bar);
        // n2 = (c2 + j I)⁻¹ with c2 = [[Σ_rr, Σ_rp], [Σ_rp, Σ_pp]].
        let c_bar = inv2_backward(&tm.n2, &n2_bar);
        out.push((m_bar, [c_bar[0], c_bar[1] + c_bar[2], c_bar[3]]));
    }
    out
}

/// `clamp_min(v, lo)` and whether the gradient passes (`v > lo`: a scalar
/// bound has a zero subgradient at equality in torch ≥ 2.14; none for NaN).
#[inline]
fn clamp_min_active(v: f64, lo: f64) -> (f64, bool) {
    if v < lo {
        (lo, false)
    } else {
        (v, v > lo)
    }
}

// ---------------------------------------------------------------------------
// EP elements
// ---------------------------------------------------------------------------

/// The predictive distribution of one output at `R` points with jitter and
/// its natural parameters (BoTorch's `_initialize_predictive_matrices` for
/// one output and batch element).
#[derive(Clone, Debug, PartialEq)]
pub struct PredictiveMatrices {
    /// The predictive mean (`R`).
    pub mean: Vector,
    /// The predictive covariance plus `jitter · I` (`R × R`).
    pub cov: Matrix,
    /// The natural mean `cov⁻¹ mean`.
    pub nat_mean: Vector,
    /// The natural covariance `cov⁻¹` (from its Cholesky factor).
    pub nat_cov: Matrix,
}

/// BoTorch's `_initialize_predictive_matrices` for one output: the posterior
/// of `gp` at the rows of `x` (with the observation noise if
/// `observation_noise`), its covariance plus `jitter · I` and the natural
/// parameters.
///
/// # Errors
/// [`BotorchError::Numerical`] if the jittered covariance has no Cholesky
/// factorization (BoTorch ignores the failure of `cholesky_ex` there and
/// continues with an invalid factor), and errors of the posterior.
pub fn initialize_predictive_matrices(
    gp: &SingleTaskGP,
    x: &Matrix,
    observation_noise: bool,
    jitter: f64,
) -> Result<PredictiveMatrices> {
    let post = gp.posterior(x, observation_noise)?;
    let mut cov = post.covariance;
    for i in 0..cov.nrows() {
        cov[(i, i)] += jitter;
    }
    let (l, info) = cholesky_ex(&cov);
    if info != 0 {
        return Err(BotorchError::Numerical(format!(
            "the predictive covariance of PES (with jitter {jitter}) has no Cholesky \
             factorization (leading minor {info})"
        )));
    }
    let nat_cov = cholesky_inverse(&l);
    let nat_mean = &nat_cov * &post.mean;
    Ok(PredictiveMatrices {
        mean: post.mean,
        cov,
        nat_mean,
        nat_cov,
    })
}

/// The EP factors of one Pareto sample and output over `R = N + P` rows and
/// `P` Pareto points: natural means (`R × P × 2`) and covariances
/// (`R × P × 2 × 2`), row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct EpFactors {
    rows: usize,
    p: usize,
    mean: Vec<f64>,
    cov: Vec<f64>,
}

impl EpFactors {
    /// Zero factors (BoTorch's initialization).
    pub fn zeros(rows: usize, p: usize) -> Self {
        Self {
            rows,
            p,
            mean: vec![0.0; rows * p * 2],
            cov: vec![0.0; rows * p * 4],
        }
    }

    /// The number of rows `R`.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of Pareto points `P`.
    pub fn num_pareto_points(&self) -> usize {
        self.p
    }

    /// The natural mean factor of row `r` and Pareto point `p` (2 values).
    pub fn mean(&self, r: usize, p: usize) -> [f64; 2] {
        let i = (r * self.p + p) * 2;
        [self.mean[i], self.mean[i + 1]]
    }

    /// The natural covariance factor of row `r` and Pareto point `p`
    /// (2 × 2, row-major).
    pub fn cov(&self, r: usize, p: usize) -> [f64; 4] {
        let i = (r * self.p + p) * 4;
        [
            self.cov[i],
            self.cov[i + 1],
            self.cov[i + 2],
            self.cov[i + 3],
        ]
    }

    /// Factors from their row-major values: natural means `R × P × 2` and
    /// covariances `R × P × 2 × 2`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for values of the wrong length.
    pub fn from_values(rows: usize, p: usize, mean: Vec<f64>, cov: Vec<f64>) -> Result<Self> {
        if mean.len() != rows * p * 2 || cov.len() != rows * p * 4 {
            return Err(BotorchError::DimensionMismatch(format!(
                "EP factors of {rows} rows and {p} Pareto points need {} mean and {} covariance \
                 values, got {} and {}",
                rows * p * 2,
                rows * p * 4,
                mean.len(),
                cov.len()
            )));
        }
        Ok(Self { rows, p, mean, cov })
    }

    /// The natural mean factors, row-major `R × P × 2`.
    pub fn means(&self) -> &[f64] {
        &self.mean
    }

    /// The natural covariance factors, row-major `R × P × 2 × 2`.
    pub fn covs(&self) -> &[f64] {
        &self.cov
    }

    /// Sets the factors of row `r` and Pareto point `p`.
    pub fn set(&mut self, r: usize, p: usize, mean: [f64; 2], cov: [f64; 4]) {
        let i = (r * self.p + p) * 2;
        self.mean[i..i + 2].copy_from_slice(&mean);
        let j = (r * self.p + p) * 4;
        self.cov[j..j + 4].copy_from_slice(&cov);
    }

    /// The factors of the rows `start..` (the Pareto rows).
    fn tail(&self, start: usize) -> Self {
        Self {
            rows: self.rows - start,
            p: self.p,
            mean: self.mean[start * self.p * 2..].to_vec(),
            cov: self.cov[start * self.p * 4..].to_vec(),
        }
    }

    fn has_nan(&self) -> bool {
        self.mean.iter().chain(&self.cov).any(|v| v.is_nan())
    }

    /// BoTorch's `_damped_update`: `df · new + (1 - df) · self`.
    pub fn damped(&self, new: &Self, df: f64) -> Self {
        Self {
            rows: self.rows,
            p: self.p,
            mean: damp_vec(&self.mean, &new.mean, df),
            cov: damp_vec(&self.cov, &new.cov, df),
        }
    }
}

fn damp_vec(old: &[f64], new: &[f64], df: f64) -> Vec<f64> {
    old.iter()
        .zip(new)
        .map(|(o, n)| df * n + (1.0 - df) * o)
        .collect()
}

fn damp_mat(old: &Matrix, new: &Matrix, df: f64) -> Matrix {
    Matrix::from_fn(old.nrows(), old.ncols(), |i, j| {
        df * new[(i, j)] + (1.0 - df) * old[(i, j)]
    })
}

/// BoTorch's `_update_omega` for one Pareto sample: the new EP factors of
/// every output (`means[t]`, `covs[t]` and `omegas[t]` over `R = n + p` rows,
/// the last `p` of them the Pareto points), found by matching the moments of
/// the tilted distributions of all (row, Pareto point) pairs. `None` if a
/// 2 × 2 matrix is singular (`torch.linalg.inv` raises there).
pub fn update_omega(
    means: &[&Vector],
    covs: &[&Matrix],
    omegas: &[&EpFactors],
    n: usize,
    p: usize,
    maximize: bool,
    jitter: f64,
) -> Option<Vec<EpFactors>> {
    update_omega_with_terms(means, covs, omegas, n, p, maximize, jitter, 0).map(|(f, _)| f)
}

/// The inputs of the EP factor update of one Pareto sample: the means,
/// covariances and old factors of every output.
pub struct OmegaInputs<'a> {
    /// The means of every output (`R`).
    pub means: Vec<&'a Vector>,
    /// The covariances of every output (`R × R`).
    pub covs: Vec<&'a Matrix>,
    /// The old factors of every output.
    pub omegas: Vec<&'a EpFactors>,
}

/// BoTorch's `_safe_update_omega` over a batch of Pareto samples: the new
/// factors of every sample, or the old ones of every sample if any update
/// is `None` or contains a NaN; the second value tells whether the update
/// fell back.
pub fn safe_update_omega(
    inputs: &[OmegaInputs<'_>],
    n: usize,
    p: usize,
    maximize: bool,
    jitter: f64,
) -> (Vec<Vec<EpFactors>>, bool) {
    let updates: Vec<Option<Vec<EpFactors>>> = inputs
        .par_iter()
        .map(|i| update_omega(&i.means, &i.covs, &i.omegas, n, p, maximize, jitter))
        .collect();
    let failed = updates
        .iter()
        .any(|u| u.as_ref().is_none_or(|f| f.iter().any(EpFactors::has_nan)));
    if failed {
        let old = inputs
            .iter()
            .map(|i| i.omegas.iter().map(|f| (*f).clone()).collect())
            .collect();
        return (old, true);
    }
    (
        updates.into_iter().map(|u| u.expect("checked")).collect(),
        false,
    )
}

/// The new factors of every output and the forward quantities of the kept
/// rows (`[r][p][t]`).
type FactorUpdate = (Vec<EpFactors>, Vec<Vec<Vec<PairTerms>>>);

/// [`update_omega`] that also returns the forward quantities of the rows
/// `0..keep_rows` for the backward pass (indexed `[r][p][t]`).
#[allow(clippy::too_many_arguments)]
fn update_omega_with_terms(
    means: &[&Vector],
    covs: &[&Matrix],
    omegas: &[&EpFactors],
    n: usize,
    p: usize,
    maximize: bool,
    jitter: f64,
    keep_rows: usize,
) -> Option<FactorUpdate> {
    let m = means.len();
    let rows = n + p;
    let mut out: Vec<EpFactors> = (0..m).map(|_| EpFactors::zeros(rows, p)).collect();
    let mut kept = Vec::with_capacity(keep_rows);
    let mut inputs = Vec::with_capacity(m);
    for r in 0..rows {
        let mut kept_row = Vec::new();
        for k in 0..p {
            let col = n + k;
            inputs.clear();
            for t in 0..m {
                let (mean, cov) = (means[t], covs[t]);
                let c = cov[(r, col)];
                inputs.push(PairInput {
                    m: [mean[r], mean[col]],
                    c: [cov[(r, r)], c, c, cov[(col, col)]],
                    om: omegas[t].mean(r, k),
                    oc: omegas[t].cov(r, k),
                    diag: r == col,
                });
            }
            let terms = update_pair(&inputs, maximize, jitter)?;
            for (t, tm) in terms.iter().enumerate() {
                out[t].set(r, k, tm.om_new, tm.oc_new);
            }
            if r < keep_rows {
                kept_row.push(terms);
            }
        }
        if r < keep_rows {
            kept.push(kept_row);
        }
    }
    Some((out, kept))
}

/// BoTorch's `_update_marginals`: the natural parameters of the marginal
/// (the prior natural parameters plus the factors, without the Pareto
/// diagonal) over `R = n + p` rows. Zeroes the Pareto diagonal of `omega`
/// (as BoTorch does in place).
pub fn update_marginals(
    pred_nat_mean: &Vector,
    pred_nat_cov: &Matrix,
    omega: &mut EpFactors,
    n: usize,
    p: usize,
) -> (Vector, Matrix) {
    for k in 0..p {
        omega.set(n + k, k, [0.0; 2], [0.0; 4]);
    }
    let rows = n + p;
    let mut nat_mean = pred_nat_mean.clone();
    let mut nat_cov = pred_nat_cov.clone();
    for r in 0..rows {
        let s: f64 = (0..p).map(|k| omega.mean(r, k)[0]).sum();
        nat_mean[r] += s;
    }
    for k in 0..p {
        let s: f64 = (0..rows).map(|r| omega.mean(r, k)[1]).sum();
        nat_mean[n + k] += s;
    }
    for r in 0..rows {
        let s: f64 = (0..p).map(|k| omega.cov(r, k)[0]).sum();
        nat_cov[(r, r)] += s;
    }
    for k in 0..p {
        let s: f64 = (0..rows).map(|r| omega.cov(r, k)[3]).sum();
        nat_cov[(n + k, n + k)] += s;
    }
    for k in 0..p {
        for r in 0..rows {
            nat_cov[(r, n + k)] += omega.cov(r, k)[1];
        }
        for r in 0..rows {
            nat_cov[(n + k, r)] += omega.cov(r, k)[2];
        }
    }
    (nat_mean, nat_cov)
}

/// BoTorch's `_update_damping` over all elements: the new damping factors
/// (halved, and set to zero below 1e-3, where the damped natural covariance
/// has no Cholesky factorization) and the Cholesky factors of the damped
/// natural covariances (of the jittered ones after a retry), with BoTorch's
/// test of the sum of the `info` codes of all elements.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] with BoTorch's message if the previous
/// natural covariances are not positive definite even with jitter.
pub fn update_damping(
    nat_cov: &[&Matrix],
    nat_cov_new: &[&Matrix],
    damping: &[f64],
    jitter: f64,
) -> Result<(Vec<f64>, Vec<Matrix>)> {
    let with_jitter = |a: &Matrix| {
        let mut b = a.clone();
        for i in 0..b.nrows() {
            b[(i, i)] += jitter;
        }
        b
    };
    let infos: usize = nat_cov
        .par_iter()
        .map(|c| cholesky_ex(&with_jitter(c)).1)
        .sum();
    if infos > 1 {
        return Err(BotorchError::InvalidArgument(NOT_PSD_MESSAGE.into()));
    }
    let mut df = damping.to_vec();
    let factor = |df: &[f64], jit: bool| -> Vec<(Matrix, usize)> {
        nat_cov
            .par_iter()
            .zip(nat_cov_new.par_iter())
            .zip(df.par_iter())
            .map(|((old, new), &d)| {
                let damped = damp_mat(old, new, d);
                cholesky_ex(&if jit { with_jitter(&damped) } else { damped })
            })
            .collect()
    };
    let mut res = factor(&df, false);
    loop {
        let info_sum: usize = res.iter().map(|(_, i)| i).sum();
        let nans = res.iter().any(|(l, _)| l.iter().any(|v| v.is_nan()));
        if info_sum <= 1 && !nans {
            break;
        }
        df = df
            .iter()
            .zip(&res)
            .map(|(&d, (_, info))| {
                if *info == 0 {
                    d
                } else {
                    let alt = 0.5 * d;
                    if alt > MIN_DAMPING {
                        alt
                    } else {
                        0.0
                    }
                }
            })
            .collect();
        res = factor(&df, true);
    }
    Ok((df, res.into_iter().map(|(l, _)| l).collect()))
}

/// BoTorch's `_augment_factors_with_cached_factors` for one element:
/// replaces the factors of the Pareto rows (rows `q..` of `omega`) by the
/// factors that EP computed for them at construction (`cached`, the Pareto
/// rows only).
///
/// # Panics
/// If `cached` does not have the Pareto rows of `omega`.
pub fn augment_factors_with_cached_factors(q: usize, omega: &mut EpFactors, cached: &EpFactors) {
    let p = omega.num_pareto_points();
    assert!(
        cached.rows() == omega.rows() - q && cached.num_pareto_points() == p,
        "the cached factors must have the Pareto rows of the factors"
    );
    for r in 0..cached.rows() {
        for k in 0..p {
            omega.set(q + r, k, cached.mean(r, k), cached.cov(r, k));
        }
    }
}

/// BoTorch's `_compute_log_determinant`: `Σ_t log det(Σ[:q, :q])` over the
/// outputs, averaged over the Pareto samples (`covs[s·M + t]` for `M =
/// num_outputs` outputs).
///
/// # Errors
/// [`BotorchError::InputData`] (BoTorch's `check_no_nans`) for a NaN log
/// determinant, e.g. of a block with a negative determinant.
pub fn compute_log_determinant(covs: &[&Matrix], q: usize, num_outputs: usize) -> Result<f64> {
    let ns = covs.len() / num_outputs;
    let mut total = 0.0;
    for s in 0..ns {
        let mut sum = 0.0;
        for t in 0..num_outputs {
            let ld = Lu::new(&covs[s * num_outputs + t].view((0, 0), (q, q)).into_owned()).logdet();
            if ld.is_nan() {
                return Err(BotorchError::InputData(NAN_MESSAGE.into()));
            }
            sum += ld;
        }
        total += sum;
    }
    Ok(total / ns as f64)
}

/// BoTorch's `_update_damping_when_converged` for one element: the damping
/// factor, set to zero after the third iteration once the largest changes
/// of the mean and the covariance are below `threshold` times their largest
/// entries.
pub fn update_damping_when_converged(
    mean_old: &Vector,
    mean_new: &Vector,
    cov_old: &Matrix,
    cov_new: &Matrix,
    damping: f64,
    iteration: usize,
    threshold: f64,
) -> f64 {
    if iteration > 2 && converged(mean_old, mean_new, cov_old, cov_new, threshold) {
        0.0
    } else {
        damping
    }
}

fn converged(
    mean_old: &Vector,
    mean_new: &Vector,
    cov_old: &Matrix,
    cov_new: &Matrix,
    threshold: f64,
) -> bool {
    let amax = |it: &mut dyn Iterator<Item = f64>| it.fold(0.0f64, |a, v| a.max(v.abs()));
    let am = amax(&mut mean_old.iter().copied());
    let ac = amax(&mut cov_old.iter().copied());
    let dm = amax(&mut mean_new.iter().zip(mean_old).map(|(a, b)| a - b));
    let dc = amax(&mut cov_new.iter().zip(cov_old).map(|(a, b)| a - b));
    dm < threshold * am && dc < threshold * ac
}

// ---------------------------------------------------------------------------
// The acquisition function
// ---------------------------------------------------------------------------

/// Options of [`QMultiObjectivePredictiveEntropySearch`] (BoTorch's keyword
/// arguments and defaults).
#[derive(Clone, Debug)]
pub struct QMultiObjectivePredictiveEntropySearchOptions {
    /// Maximization (default) or minimization.
    pub maximize: bool,
    /// Pending points (`m × d`), appended to every candidate set.
    pub x_pending: Option<Matrix>,
    /// The maximum number of EP iterations at construction (250).
    pub max_ep_iterations: usize,
    /// The jitter of the matrix inverses of EP at construction (1e-4).
    pub ep_jitter: f64,
    /// The jitter of the matrix inverses at evaluation (1e-4).
    pub test_jitter: f64,
    /// The relative change of the means and covariances below which EP has
    /// converged (1e-2).
    pub threshold: f64,
}

impl Default for QMultiObjectivePredictiveEntropySearchOptions {
    fn default() -> Self {
        Self {
            maximize: true,
            x_pending: None,
            max_ep_iterations: 250,
            ep_jitter: 1e-4,
            test_jitter: 1e-4,
            threshold: 1e-2,
        }
    }
}

/// The result of EP at construction.
#[derive(Clone, Debug, PartialEq)]
pub struct EpSummary {
    /// The number of EP iterations.
    pub iterations: usize,
    /// The final damping factor of every Pareto sample and output (`s·M + t`);
    /// zero once converged.
    pub damping: Vec<f64>,
}

/// Multi-objective predictive entropy search (BoTorch's
/// `qMultiObjectivePredictiveEntropySearch`; see the [module
/// documentation](self)). The value can be negative; BoTorch recommends
/// finite differences because EP is not always differentiable (the damping
/// factors change with the candidates).
#[derive(Debug)]
pub struct QMultiObjectivePredictiveEntropySearch {
    model: Arc<dyn Model>,
    pareto_sets: Vec<Matrix>,
    options: QMultiObjectivePredictiveEntropySearchOptions,
    x_pending: RwLock<Option<Matrix>>,
    /// The cached factors of the Pareto rows, element `s·M + t`.
    cached: Vec<EpFactors>,
    summary: EpSummary,
}

impl QMultiObjectivePredictiveEntropySearch {
    /// PES with the sampled Pareto sets (`S` matrices of `P × d`); runs EP.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for batched models and models without
    /// independent exact GP outputs, [`BotorchError::DimensionMismatch`] for
    /// Pareto sets of the wrong shape, [`BotorchError::InvalidArgument`] with
    /// BoTorch's messages when EP fails (a natural covariance without a
    /// Cholesky factorization) and [`BotorchError::InputData`] for NaNs.
    pub fn new(
        model: Arc<dyn Model>,
        pareto_sets: Vec<Matrix>,
        options: QMultiObjectivePredictiveEntropySearchOptions,
    ) -> Result<Self> {
        if !model.batch_shape().is_empty() {
            return Err(BotorchError::Unsupported(
                "Batch GP models (e.g. fantasized models) are not supported.".into(),
            ));
        }
        check_unbatched(model.as_ref())?;
        let d = model.dim();
        let p = pareto_sets.first().map_or(0, |s| s.nrows());
        if pareto_sets.is_empty() || p == 0 || pareto_sets.iter().any(|s| s.shape() != (p, d)) {
            return Err(BotorchError::Unsupported(
                "The Pareto set should have a shape of `num_pareto_samples x num_pareto_points x \
                 input_dim`."
                    .into(),
            ));
        }
        if let Some(xp) = &options.x_pending {
            check_pending(xp, d)?;
        }
        let gps = model.output_gps().ok_or_else(|| {
            BotorchError::Unsupported(
                "PES requires a model of independent exact GPs (a SingleTaskGP or a ModelListGP)"
                    .into(),
            )
        })?;
        // The raw training inputs of the first output (BoTorch:
        // `model.train_inputs[0][0]`).
        let train_x = gps[0].train_inputs().clone();
        let (cached, summary) = expectation_propagation(&gps, &train_x, &pareto_sets, &options)?;
        let x_pending = RwLock::new(options.x_pending.clone());
        Ok(Self {
            model,
            pareto_sets,
            options,
            x_pending,
            cached,
            summary,
        })
    }

    /// The sampled Pareto sets.
    pub fn pareto_sets(&self) -> &[Matrix] {
        &self.pareto_sets
    }

    /// The result of EP at construction.
    pub fn ep_summary(&self) -> &EpSummary {
        &self.summary
    }

    /// The cached EP factors of the Pareto rows (BoTorch's
    /// `_omega_f_nat_mean[..., N:, :, :]` and `_omega_f_nat_cov`), element
    /// `s·M + t` for Pareto sample `s` and output `t`.
    pub fn cached_factors(&self) -> &[EpFactors] {
        &self.cached
    }

    fn pending(&self) -> Option<Matrix> {
        self.x_pending
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The value and (with `with_grad`) its gradient at the candidates.
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let q_cand = x.nrows();
        let full = match self.pending() {
            Some(p) if p.nrows() > 0 => {
                let mut f = Matrix::zeros(q_cand + p.nrows(), d);
                f.rows_mut(0, q_cand).copy_from(x);
                f.rows_mut(q_cand, p.nrows()).copy_from(&p);
                f
            }
            _ => x.clone(),
        };
        let q = full.nrows();
        let gps = self.model.output_gps().expect("checked at construction");
        let m = gps.len();
        let ns = self.pareto_sets.len();
        let p = self.pareto_sets[0].nrows();
        let jitter = self.options.test_jitter;
        let maximize = self.options.maximize;
        // The predictive distributions at [X; X*_s] with noise and jitter.
        let aug: Vec<Matrix> = self
            .pareto_sets
            .iter()
            .map(|ps| {
                let mut a = Matrix::zeros(q + p, d);
                a.rows_mut(0, q).copy_from(&full);
                a.rows_mut(q, p).copy_from(ps);
                a
            })
            .collect();
        let preds: Vec<PredictiveMatrices> = (0..ns * m)
            .map(|e| initialize_predictive_matrices(gps[e % m], &aug[e / m], true, jitter))
            .collect::<Result<Vec<_>>>()?;
        // One EP update of the factors of the candidate rows (all samples;
        // the update falls back to zero factors if any of them fails).
        let zeros = EpFactors::zeros(q + p, p);
        let updates: Vec<Option<FactorUpdate>> = (0..ns)
            .map(|s| {
                let means: Vec<&Vector> = (0..m).map(|t| &preds[s * m + t].mean).collect();
                let covs: Vec<&Matrix> = (0..m).map(|t| &preds[s * m + t].cov).collect();
                let omegas: Vec<&EpFactors> = vec![&zeros; m];
                update_omega_with_terms(
                    &means,
                    &covs,
                    &omegas,
                    q,
                    p,
                    maximize,
                    jitter,
                    if with_grad { q_cand } else { 0 },
                )
            })
            .collect();
        let failed = updates.iter().any(|u| {
            u.as_ref()
                .is_none_or(|(f, _)| f.iter().any(EpFactors::has_nan))
        });
        let mut omegas: Vec<EpFactors> = Vec::with_capacity(ns * m);
        for (s, u) in updates.iter().enumerate() {
            for t in 0..m {
                let mut om = match (failed, u) {
                    (false, Some((f, _))) => f[t].clone(),
                    _ => zeros.clone(),
                };
                augment_factors_with_cached_factors(q, &mut om, &self.cached[s * m + t]);
                omegas.push(om);
            }
        }
        let mut nat_covs = Vec::with_capacity(ns * m);
        for (e, om) in omegas.iter_mut().enumerate() {
            let pr = &preds[e];
            let (_, nc) = update_marginals(&pr.nat_mean, &pr.nat_cov, om, q, p);
            nat_covs.push(nc);
        }
        let pred_nat: Vec<&Matrix> = preds.iter().map(|pr| &pr.nat_cov).collect();
        let new_nat: Vec<&Matrix> = nat_covs.iter().collect();
        let (df, chols) = update_damping(&pred_nat, &new_nat, &vec![1.0; ns * m], jitter)?;
        let covs_f: Vec<Matrix> = chols.iter().map(cholesky_inverse).collect();
        if covs_f.iter().any(|c| c.iter().any(|v| v.is_nan())) {
            return Err(BotorchError::InputData(NAN_MESSAGE.into()));
        }
        // The log determinants of the candidate blocks.
        let pred_covs: Vec<&Matrix> = preds.iter().map(|pr| &pr.cov).collect();
        let post_covs: Vec<&Matrix> = covs_f.iter().collect();
        let ld_pred = compute_log_determinant(&pred_covs, q, m)?;
        let ld_post = compute_log_determinant(&post_covs, q, m)?;
        let value = 0.5 * (ld_pred - ld_post);
        if value.is_nan() {
            return Err(BotorchError::InputData(NAN_MESSAGE.into()));
        }
        if !with_grad {
            return Ok((value, None));
        }
        // Reverse mode: the adjoints of the predictive means and covariances
        // of every element, pulled back through the posteriors.
        let mut grad = Matrix::zeros(q_cand, d);
        let scale = 0.5 / ns as f64;
        let r = q + p;
        for s in 0..ns {
            let mut cov_bar = vec![Matrix::zeros(r, r); m];
            let mut mean_bar = vec![Vector::zeros(r); m];
            let mut nf_bar = Vec::with_capacity(m);
            for (t, cb) in cov_bar.iter_mut().enumerate() {
                let e = s * m + t;
                // + 0.5/S log det(Σ_pred[:q, :q]).
                let la = Lu::new(&preds[e].cov.view((0, 0), (q, q)).into_owned());
                let ia = la.inverse().transpose() * scale;
                let mut block = cb.view_mut((0, 0), (q, q));
                block += &ia;
                // - 0.5/S log det(Σ_f[:q, :q]) with Σ_f = D⁻¹ and the damped
                // natural covariance D = df N_f + (1 - df) N_pred (+ jitter).
                let mut cf_bar = Matrix::zeros(r, r);
                let lb = Lu::new(&covs_f[e].view((0, 0), (q, q)).into_owned());
                cf_bar
                    .view_mut((0, 0), (q, q))
                    .copy_from(&(lb.inverse().transpose() * -scale));
                let cf = &covs_f[e];
                let d_bar = -(cf * &cf_bar * cf);
                let nfb = &d_bar * df[e];
                // N_f = N_pred + the factors, so N_pred collects both terms.
                let np_bar = &d_bar * (1.0 - df[e]) + &nfb;
                let npc = &preds[e].nat_cov;
                *cb -= npc * &np_bar * npc;
                nf_bar.push(nfb);
            }
            // The factors of the candidate rows (constant if the update fell
            // back to zero factors; the pending rows are constant).
            if !failed {
                let kept = &updates[s].as_ref().expect("the update succeeded").1;
                for (rr, row) in kept.iter().enumerate() {
                    for (k, terms) in row.iter().enumerate() {
                        let col = q + k;
                        let ocb: Vec<M2> = nf_bar
                            .iter()
                            .map(|nb| [nb[(rr, rr)], nb[(rr, col)], nb[(col, rr)], nb[(col, col)]])
                            .collect();
                        for (t, (mb, cb)) in update_pair_backward(terms, &ocb, maximize)
                            .into_iter()
                            .enumerate()
                        {
                            mean_bar[t][rr] += mb[0];
                            mean_bar[t][col] += mb[1];
                            cov_bar[t][(rr, rr)] += cb[0];
                            cov_bar[t][(rr, col)] += cb[1];
                            cov_bar[t][(col, col)] += cb[2];
                        }
                    }
                }
            }
            for t in 0..m {
                grad +=
                    gps[t].joint_posterior_vjp(&aug[s], 0..q_cand, &mean_bar[t], &cov_bar[t])?;
            }
        }
        Ok((value, Some(grad)))
    }
}

/// EP over the training inputs and the Pareto sets (BoTorch's
/// `_expectation_propagation`): the cached factors of the Pareto rows and
/// the summary.
fn expectation_propagation(
    gps: &[&SingleTaskGP],
    train_x: &Matrix,
    pareto_sets: &[Matrix],
    options: &QMultiObjectivePredictiveEntropySearchOptions,
) -> Result<(Vec<EpFactors>, EpSummary)> {
    let m = gps.len();
    let ns = pareto_sets.len();
    let (n, d) = train_x.shape();
    let p = pareto_sets[0].nrows();
    let rows = n + p;
    let jitter = options.ep_jitter;
    let aug: Vec<Matrix> = pareto_sets
        .iter()
        .map(|ps| {
            let mut a = Matrix::zeros(rows, d);
            a.rows_mut(0, n).copy_from(train_x);
            a.rows_mut(n, p).copy_from(ps);
            a
        })
        .collect();
    let preds: Vec<PredictiveMatrices> = (0..ns * m)
        .into_par_iter()
        .map(|e| initialize_predictive_matrices(gps[e % m], &aug[e / m], false, jitter))
        .collect::<Result<Vec<_>>>()?;
    let mut mean_f: Vec<Vector> = preds.iter().map(|pr| pr.mean.clone()).collect();
    let mut cov_f: Vec<Matrix> = preds.iter().map(|pr| pr.cov.clone()).collect();
    let mut nat_mean_f: Vec<Vector> = preds.iter().map(|pr| pr.nat_mean.clone()).collect();
    let mut nat_cov_f: Vec<Matrix> = preds.iter().map(|pr| pr.nat_cov.clone()).collect();
    let mut omega: Vec<EpFactors> = (0..ns * m).map(|_| EpFactors::zeros(rows, p)).collect();
    let mut damping = vec![1.0; ns * m];
    let mut iteration = 0;
    while damping.iter().sum::<f64>() > 0.0 && iteration < options.max_ep_iterations {
        // The factor update (`_safe_update_omega`).
        let inputs: Vec<OmegaInputs<'_>> = (0..ns)
            .map(|s| OmegaInputs {
                means: (0..m).map(|t| &mean_f[s * m + t]).collect(),
                covs: (0..m).map(|t| &cov_f[s * m + t]).collect(),
                omegas: (0..m).map(|t| &omega[s * m + t]).collect(),
            })
            .collect();
        let (updates, _) = safe_update_omega(&inputs, n, p, options.maximize, jitter);
        drop(inputs);
        let mut omega_new: Vec<EpFactors> = updates.into_iter().flatten().collect();
        // The marginal update.
        let marginals: Vec<(Vector, Matrix)> = omega_new
            .par_iter_mut()
            .zip(preds.par_iter())
            .map(|(om, pr)| update_marginals(&pr.nat_mean, &pr.nat_cov, om, n, p))
            .collect();
        // The damping update.
        let olds: Vec<&Matrix> = nat_cov_f.iter().collect();
        let news: Vec<&Matrix> = marginals.iter().map(|(_, c)| c).collect();
        let (df, chols) = update_damping(&olds, &news, &damping, jitter)?;
        if chols.iter().any(|l| l.iter().any(|v| v.is_nan())) {
            return Err(BotorchError::InputData(NAN_MESSAGE.into()));
        }
        damping = df;
        // The damped updates.
        let results: Vec<(EpFactors, Vector, Matrix, Matrix, Vector)> = (0..ns * m)
            .into_par_iter()
            .map(|e| {
                let dfe = damping[e];
                let om = omega[e].damped(&omega_new[e], dfe);
                let nm = Vector::from_iterator(
                    rows,
                    nat_mean_f[e]
                        .iter()
                        .zip(marginals[e].0.iter())
                        .map(|(o, nw)| dfe * nw + (1.0 - dfe) * o),
                );
                let nc = damp_mat(&nat_cov_f[e], &marginals[e].1, dfe);
                let cov_new = cholesky_inverse(&chols[e]);
                let mean_new = &cov_new * &nm;
                (om, nm, nc, cov_new, mean_new)
            })
            .collect();
        if results.iter().any(|r| r.3.iter().any(|v| v.is_nan())) {
            return Err(BotorchError::InputData(NAN_MESSAGE.into()));
        }
        for (e, (om, nm, nc, cov_new, mean_new)) in results.into_iter().enumerate() {
            damping[e] = update_damping_when_converged(
                &mean_f[e],
                &mean_new,
                &cov_f[e],
                &cov_new,
                damping[e],
                iteration,
                options.threshold,
            );
            omega[e] = om;
            nat_mean_f[e] = nm;
            nat_cov_f[e] = nc;
            cov_f[e] = cov_new;
            mean_f[e] = mean_new;
        }
        iteration += 1;
    }
    if omega.iter().any(EpFactors::has_nan) {
        return Err(BotorchError::InputData(NAN_MESSAGE.into()));
    }
    let cached = omega.iter().map(|o| o.tail(n)).collect();
    Ok((
        cached,
        EpSummary {
            iterations: iteration,
            damping,
        },
    ))
}

impl AcquisitionFunction for QMultiObjectivePredictiveEntropySearch {
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

    fn x_pending(&self) -> Option<Matrix> {
        self.pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.model.dim())?;
        }
        *self.x_pending.write().unwrap_or_else(|e| e.into_inner()) = x_pending.cloned();
        Ok(())
    }
}
