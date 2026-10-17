//! Lower-bound joint entropy search for multi-objective Bayesian optimization
//! (`botorch.acquisition.multi_objective.joint_entropy_search`; Tu et al.,
//! *Joint Entropy Search for Multi-Objective Bayesian Optimization*, NeurIPS
//! 2022).
//!
//! * [`QLowerBoundMultiObjectiveJointEntropySearch`]: the mutual information
//!   between the observations at the candidates and the sampled Pareto
//!   optimal input–output pairs, with the conditional entropy estimated per
//!   candidate (the lower bound for `q > 1`).
//! * The entropy estimators shared with
//!   [`QLowerBoundMultiObjectiveMaxValueEntropySearch`](super::max_value_entropy_search::QLowerBoundMultiObjectiveMaxValueEntropySearch):
//!   [`compute_entropy_noiseless`] (`"0"`), [`compute_entropy_upper_bound`]
//!   (`"LB"`, and `"LB2"` with only the diagonal of the covariance) and
//!   [`compute_entropy_monte_carlo`] (`"MC"`), BoTorch's private
//!   `_compute_entropy_*` functions for one candidate point.
//! * [`ConditionedOutput`]: the posterior of an output GP conditioned on
//!   noisy observations at the sampled Pareto points (BoTorch's
//!   `condition_on_observations` as JES uses it: a rank update of the
//!   posterior with the likelihood's noise).
//!
//! Values and gradients follow BoTorch's computation step by step (the same
//! clamps, `torch.distributions.Normal`'s `0.5 (1 + erf(x / √2))` CDF and
//! `torch.logdet`); gradients are the exact reverse-mode derivatives of that
//! computation, as `torch.autograd` computes them. The conditional entropy
//! of every candidate is averaged over the Pareto samples and its gradient
//! flows through the per-point posteriors (the Pareto samples, cells and MC
//! base samples are constants).
//!
//! Deviations from BoTorch main (see the README):
//!
//! * The Pareto sets are input-transformed once when the model is
//!   conditioned on them (meta-pytorch/botorch#3399): BoTorch main passes
//!   `model.transform_inputs(pareto_sets)` to `condition_on_observations`,
//!   which transforms them again, so with an input transform (e.g.
//!   `Normalize`) it conditions on the Pareto fronts at the wrong inputs.
//!   Without input transforms both agree.
//! * The outputs of the sub-models of a `ModelListGP` are each conditioned on
//!   their own columns of the Pareto fronts, also for multi-output
//!   sub-models (#3399; BoTorch main mixes them up).

use std::f64::consts::{FRAC_2_SQRT_PI, PI, SQRT_2};
use std::sync::{Arc, RwLock};

use errorfunctions::RealErrorFunctions;
use rayon::prelude::*;

use crate::acquisition::multi_objective::entropy_linalg::Lu;
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::models::gp::BaselineCache;
use crate::models::{Likelihood, Model, SingleTaskGP};
use crate::posterior::MIN_VARIANCE;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::multi_objective::box_decompositions::CellBounds;
use crate::{Matrix, Vector};

/// `torch.finfo(torch.float64).eps`, BoTorch's `CLAMP_LB`.
pub(crate) const CLAMP_LB: f64 = f64::EPSILON;

/// `math.log(math.sqrt(2 * math.pi))` of `Normal.log_prob`.
const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// `math.log(2 * math.pi)` of `MultivariateNormal.log_prob`.
const LOG_2PI: f64 = 1.837_877_066_409_345_3;

/// The jitter added to the moment-matched covariance of the `"LB"` estimate.
const LB_JITTER: f64 = 1e-6;

/// BoTorch's default number of Monte-Carlo samples of the `"MC"` estimate.
pub const DEFAULT_ENTROPY_MC_SAMPLES: usize = 64;

/// The estimate of the conditional entropy (BoTorch's `estimation_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EstimationType {
    /// `"0"`: the entropy of the truncated noiseless posterior plus the
    /// noise ([`compute_entropy_noiseless`]).
    Noiseless,
    /// `"LB"` (default): the entropy of a Gaussian with the moments of the
    /// truncated noisy posterior, an upper bound of the conditional entropy
    /// and hence a lower bound of the information gain
    /// ([`compute_entropy_upper_bound`]).
    #[default]
    LowerBound,
    /// `"LB2"`: `"LB"` with only the variances (no correlations between the
    /// outputs).
    LowerBound2,
    /// `"MC"`: a Monte-Carlo estimate with samples of the noisy posterior
    /// ([`compute_entropy_monte_carlo`]).
    MonteCarlo,
}

impl EstimationType {
    /// The estimation type of BoTorch's name (`"0"`, `"LB"`, `"LB2"`,
    /// `"MC"`).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] (BoTorch's `NotImplementedError`) for
    /// other names.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "0" => Ok(Self::Noiseless),
            "LB" => Ok(Self::LowerBound),
            "LB2" => Ok(Self::LowerBound2),
            "MC" => Ok(Self::MonteCarlo),
            _ => Err(BotorchError::Unsupported(
                "Currently the only supported estimation type are: \"0\", \"LB\", \"LB2\", \"MC\"."
                    .into(),
            )),
        }
    }

    /// BoTorch's name of the estimation type.
    pub fn name(self) -> &'static str {
        match self {
            Self::Noiseless => "0",
            Self::LowerBound => "LB",
            Self::LowerBound2 => "LB2",
            Self::MonteCarlo => "MC",
        }
    }
}

// ---------------------------------------------------------------------------
// Normal distribution as torch.distributions.Normal computes it
// ---------------------------------------------------------------------------

/// `Normal(0, 1).cdf(x) = 0.5 (1 + erf(x / √2))`.
#[inline]
pub(crate) fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + RealErrorFunctions::erf(x / SQRT_2))
}

/// The derivative of [`normal_cdf`] as autograd computes it.
#[inline]
pub(crate) fn normal_cdf_grad(x: f64) -> f64 {
    let u = x / SQRT_2;
    0.5 * (FRAC_2_SQRT_PI * (-(u * u)).exp()) / SQRT_2
}

/// `Normal(0, 1).log_prob(x) = -x²/2 - log √(2π)`.
#[inline]
pub(crate) fn normal_log_prob(x: f64) -> f64 {
    -(x * x) / 2.0 - LOG_SQRT_2PI
}

/// `0.5 M (1 + log 2π)`, the constant of the entropy of an `M`-dimensional
/// Gaussian (as BoTorch computes it, also for the joint entropy of `q > 1`
/// points).
pub(crate) fn entropy_constant(m: usize) -> f64 {
    0.5 * m as f64 * (1.0 + (2.0 * PI).ln())
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
// Box quantities
// ---------------------------------------------------------------------------

/// The hypercells of one Pareto sample: `num_cells × m` lower and upper
/// vertices (row-major).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cells<'a> {
    lower: &'a [f64],
    upper: &'a [f64],
    num_cells: usize,
    m: usize,
}

impl<'a> Cells<'a> {
    pub(crate) fn of(bounds: &'a CellBounds, p: usize) -> Self {
        Self {
            lower: bounds.lower_of(p),
            upper: bounds.upper_of(p),
            num_cells: bounds.num_cells(),
            m: bounds.num_outcomes(),
        }
    }

    fn bound(&self, k: usize) -> &'a [f64] {
        if k == 0 {
            self.lower
        } else {
            self.upper
        }
    }
}

/// The normal quantities of the standardized cell bounds shared by the
/// estimators: `g = (b - μ) / √v` at both vertices, its CDF and density,
/// `W_jm = clamp_min(Φ(g¹) - Φ(g⁰), eps)`, `W_j = exp(Σ_m log W_jm)` and
/// `W = clamp_max(Σ_j W_j, 1)`.
struct Boxes {
    j: usize,
    m: usize,
    s: Vec<f64>,
    g: [Vec<f64>; 2],
    pdf: [Vec<f64>; 2],
    wm: Vec<f64>,
    wm_active: Vec<bool>,
    wj: Vec<f64>,
    w: f64,
    w_active: bool,
}

/// Adjoints of the box quantities (see [`Boxes::backward`]).
struct BoxAdjoints {
    wm: Vec<f64>,
    wj: Vec<f64>,
    w: f64,
    /// Of `V_jm = g¹ φ(g¹) - g⁰ φ(g⁰)`.
    v: Vec<f64>,
    /// Of `G_jm = φ(g¹) - φ(g⁰)`.
    gdiff: Vec<f64>,
    /// Of `g` directly.
    g: [Vec<f64>; 2],
}

impl BoxAdjoints {
    fn zeros(j: usize, m: usize) -> Self {
        Self {
            wm: vec![0.0; j * m],
            wj: vec![0.0; j],
            w: 0.0,
            v: vec![0.0; j * m],
            gdiff: vec![0.0; j * m],
            g: [vec![0.0; j * m], vec![0.0; j * m]],
        }
    }
}

impl Boxes {
    fn new(cells: &Cells<'_>, mean: &[f64], variance: &[f64]) -> Self {
        let (j, m) = (cells.num_cells, cells.m);
        let s: Vec<f64> = variance.iter().map(|v| v.sqrt()).collect();
        let mut g = [vec![0.0; j * m], vec![0.0; j * m]];
        let mut cdf = [vec![0.0; j * m], vec![0.0; j * m]];
        let mut pdf = [vec![0.0; j * m], vec![0.0; j * m]];
        for k in 0..2 {
            let b = cells.bound(k);
            for idx in 0..j * m {
                let t = idx % m;
                let gk = (b[idx] - mean[t]) / s[t];
                g[k][idx] = gk;
                cdf[k][idx] = normal_cdf(gk);
                pdf[k][idx] = normal_log_prob(gk).exp();
            }
        }
        let mut wm = vec![0.0; j * m];
        let mut wm_active = vec![false; j * m];
        for idx in 0..j * m {
            (wm[idx], wm_active[idx]) = clamp_min_active(cdf[1][idx] - cdf[0][idx], CLAMP_LB);
        }
        let wj: Vec<f64> = (0..j)
            .map(|jj| {
                wm[jj * m..(jj + 1) * m]
                    .iter()
                    .map(|v| v.ln())
                    .sum::<f64>()
                    .exp()
            })
            .collect();
        let total: f64 = wj.iter().sum();
        // W.clamp_max(1.0): no gradient at W == 1 either (torch ≥ 2.14).
        let (w, w_active) = if total > 1.0 {
            (1.0, false)
        } else {
            (total, total < 1.0)
        };
        Self {
            j,
            m,
            s,
            g,
            pdf,
            wm,
            wm_active,
            wj,
            w,
            w_active,
        }
    }

    /// `V_jm = g¹ φ(g¹) - g⁰ φ(g⁰)`.
    #[inline]
    fn v(&self, idx: usize) -> f64 {
        self.g[1][idx] * self.pdf[1][idx] - self.g[0][idx] * self.pdf[0][idx]
    }

    /// `G_jm = φ(g¹) - φ(g⁰)`.
    #[inline]
    fn gdiff(&self, idx: usize) -> f64 {
        self.pdf[1][idx] - self.pdf[0][idx]
    }

    /// Accumulates the adjoints of the mean and the variance.
    fn backward(&self, mut a: BoxAdjoints, mean_bar: &mut [f64], var_bar: &mut [f64]) {
        let (j, m) = (self.j, self.m);
        if self.w_active {
            for wj in a.wj.iter_mut() {
                *wj += a.w;
            }
        }
        for jj in 0..j {
            for t in 0..m {
                let idx = jj * m + t;
                a.wm[idx] += a.wj[jj] * self.wj[jj] * (1.0 / self.wm[idx]);
            }
        }
        let mut s_bar = vec![0.0; m];
        for idx in 0..j * m {
            let t = idx % m;
            let wm_bar = if self.wm_active[idx] { a.wm[idx] } else { 0.0 };
            for k in 0..2 {
                let sign = if k == 1 { 1.0 } else { -1.0 };
                let (gk, pk) = (self.g[k][idx], self.pdf[k][idx]);
                // d(g φ(g))/dg = φ - g² φ; dφ/dg = -g φ (autograd of exp(log_prob)).
                let mut g_bar = a.g[k][idx] + sign * wm_bar * normal_cdf_grad(gk);
                g_bar += sign * a.v[idx] * (pk + gk * (pk * -gk));
                g_bar += sign * a.gdiff[idx] * (pk * -gk);
                mean_bar[t] -= g_bar / self.s[t];
                s_bar[t] -= g_bar * gk / self.s[t];
            }
        }
        for t in 0..m {
            var_bar[t] += s_bar[t] / (2.0 * self.s[t]);
        }
    }
}

// ---------------------------------------------------------------------------
// Entropy estimators (one point, one Pareto sample)
// ---------------------------------------------------------------------------

/// Adjoints of the statistics of one point under one Pareto sample.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct StatsAdjoint {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub noise: Vec<f64>,
}

impl StatsAdjoint {
    fn zeros(m: usize) -> Self {
        Self {
            mean: vec![0.0; m],
            variance: vec![0.0; m],
            noise: vec![0.0; m],
        }
    }
}

/// `_compute_entropy_noiseless` for one point and one Pareto sample.
fn entropy_noiseless(
    cells: &Cells<'_>,
    mean: &[f64],
    variance: &[f64],
    noise: &[f64],
    adj: Option<&mut StatsAdjoint>,
) -> f64 {
    let (j, m) = (cells.num_cells, cells.m);
    let bx = Boxes::new(cells, mean, variance);
    let mut ratio_term = 0.0;
    for jj in 0..j {
        for t in 0..m {
            let idx = jj * m + t;
            ratio_term += 0.5 * (bx.wj[jj] * (bx.v(idx) / bx.wm[idx])) / bx.w;
        }
    }
    let vpn: Vec<f64> = variance.iter().zip(noise).map(|(v, o)| v + o).collect();
    let log_sum: f64 = vpn.iter().map(|v| v.ln()).sum();
    let log_term = 0.5 * log_sum + bx.w.ln();
    let h = entropy_constant(m) + (log_term - ratio_term);
    if let Some(a) = adj {
        for (t, v) in vpn.iter().enumerate() {
            let d = 0.5 / v;
            a.variance[t] += d;
            a.noise[t] += d;
        }
        let mut ba = BoxAdjoints::zeros(j, m);
        ba.w = 1.0 / bx.w;
        for jj in 0..j {
            for t in 0..m {
                let idx = jj * m + t;
                let r = bx.v(idx) / bx.wm[idx];
                let ratio = 0.5 * (bx.wj[jj] * r) / bx.w;
                // h -= 0.5 (W_j r) / W.
                ba.wj[jj] -= 0.5 * r / bx.w;
                let r_bar = -0.5 * bx.wj[jj] / bx.w;
                ba.v[idx] += r_bar / bx.wm[idx];
                ba.wm[idx] -= r_bar * r / bx.wm[idx];
                ba.w += ratio / bx.w;
            }
        }
        bx.backward(ba, &mut a.mean, &mut a.variance);
    }
    h
}

/// `_compute_entropy_upper_bound` for one point and one Pareto sample
/// (`only_diagonal`: `"LB2"`).
fn entropy_upper_bound(
    cells: &Cells<'_>,
    mean: &[f64],
    variance: &[f64],
    noise: &[f64],
    only_diagonal: bool,
    adj: Option<&mut StatsAdjoint>,
) -> f64 {
    let (j, m) = (cells.num_cells, cells.m);
    let bx = Boxes::new(cells, mean, variance);
    let s = &bx.s;
    let vpn: Vec<f64> = variance.iter().zip(noise).map(|(v, o)| v + o).collect();
    let mut c = vec![0.0; j * m];
    let mut r = vec![0.0; j * m];
    for jj in 0..j {
        for t in 0..m {
            let idx = jj * m + t;
            c[idx] = bx.gdiff(idx) / bx.wm[idx];
            r[idx] = c[idx] * bx.wj[jj] / bx.w;
        }
    }
    let sum_r: Vec<f64> = (0..m)
        .map(|t| (0..j).map(|jj| r[jj * m + t]).sum())
        .collect();
    let mom1: Vec<f64> = (0..m).map(|t| mean[t] - s[t] * sum_r[t]).collect();
    let dws_term = |jj: usize, t: usize| {
        let idx = jj * m + t;
        bx.wj[jj] * variance[t] * bx.v(idx) / bx.wm[idx] / bx.w
    };
    let dws: Vec<f64> = (0..m)
        .map(|t| (0..j).map(|jj| dws_term(jj, t)).sum())
        .collect();
    let constant = entropy_constant(m);

    if only_diagonal {
        let cross: Vec<f64> = (0..m)
            .map(|t| -2.0 * (0..j).map(|jj| mean[t] * s[t] * r[jj * m + t]).sum::<f64>())
            .collect();
        let mut var = vec![0.0; m];
        let mut var_active = vec![false; m];
        for t in 0..m {
            let mom2 = vpn[t] - dws[t] + cross[t] + mean[t].powi(2);
            (var[t], var_active[t]) = clamp_min_active(mom2 - mom1[t].powi(2), CLAMP_LB);
        }
        let h = constant + 0.5 * var.iter().map(|v| v.ln()).sum::<f64>();
        if let Some(a) = adj {
            let mut ba = BoxAdjoints::zeros(j, m);
            let mut r_bar = vec![0.0; j * m];
            let mut s_bar = vec![0.0; m];
            for t in 0..m {
                let var_bar = if var_active[t] { 0.5 / var[t] } else { 0.0 };
                let mom2_bar = var_bar;
                let mom1_bar = -2.0 * mom1[t] * var_bar;
                a.variance[t] += mom2_bar;
                a.noise[t] += mom2_bar;
                a.mean[t] += 2.0 * mean[t] * mom2_bar;
                // cross = -2 Σ_j (μ s) R_j.
                let ms_bar = -2.0 * mom2_bar * sum_r[t];
                for jj in 0..j {
                    r_bar[jj * m + t] += -2.0 * mom2_bar * mean[t] * s[t];
                }
                a.mean[t] += ms_bar * s[t];
                s_bar[t] += ms_bar * mean[t];
                // mom1 = μ - s Σ_j R_j.
                a.mean[t] += mom1_bar;
                s_bar[t] -= mom1_bar * sum_r[t];
                for jj in 0..j {
                    r_bar[jj * m + t] -= mom1_bar * s[t];
                }
                // dws = Σ_j W_j v V / W_jm / W.
                dws_backward(&bx, variance, t, -mom2_bar, &mut ba, &mut a.variance);
            }
            moments_backward(&bx, &c, &r, &r_bar, &mut ba);
            for t in 0..m {
                a.variance[t] += s_bar[t] / (2.0 * s[t]);
            }
            bx.backward(ba, &mut a.mean, &mut a.variance);
        }
        return h;
    }

    // Full covariance of the moment-matched Gaussian.
    let sc = |jj: usize, t: usize| s[t] * c[jj * m + t];
    let mut mom2 = Matrix::zeros(m, m);
    let mut offs = Matrix::zeros(m, m);
    for jj in 0..j {
        for a in 0..m {
            for b in 0..m {
                let cc = -(mean[a] * sc(jj, b)) + -(mean[b] * sc(jj, a));
                mom2[(a, b)] += cc * bx.wj[jj] / bx.w;
                offs[(a, b)] += sc(jj, a) * sc(jj, b) * bx.wj[jj] / bx.w;
            }
        }
    }
    let mut var = Matrix::zeros(m, m);
    for a in 0..m {
        for b in 0..m {
            let mut v = mom2[(a, b)] + mean[a] * mean[b] + offs[(a, b)];
            if a == b {
                v += vpn[a] - dws[a] - offs[(a, a)];
            }
            var[(a, b)] = v - mom1[a] * mom1[b];
        }
    }
    let jittered = &var + Matrix::identity(m, m) * LB_JITTER;
    let lu = Lu::new(&jittered);
    let h = constant + 0.5 * lu.logdet();
    if let Some(adj) = adj {
        let var_bar = lu.inverse().transpose() * 0.5;
        let mut ba = BoxAdjoints::zeros(j, m);
        let mut r_bar = vec![0.0; j * m];
        let mut c_bar = vec![0.0; j * m];
        let mut s_bar = vec![0.0; m];
        let mut offs_bar = var_bar.clone();
        for t in 0..m {
            // var = mom2 - mom1 mom1ᵀ.
            let mom1_bar: f64 = -(0..m)
                .map(|b| (var_bar[(t, b)] + var_bar[(b, t)]) * mom1[b])
                .sum::<f64>();
            adj.mean[t] += mom1_bar;
            s_bar[t] -= mom1_bar * sum_r[t];
            for jj in 0..j {
                r_bar[jj * m + t] -= mom1_bar * s[t];
            }
            // The diagonal term vpn - dws - init_diag.
            let d_bar = var_bar[(t, t)];
            adj.variance[t] += d_bar;
            adj.noise[t] += d_bar;
            dws_backward(&bx, variance, t, -d_bar, &mut ba, &mut adj.variance);
            offs_bar[(t, t)] -= d_bar;
            // The squared mean.
            adj.mean[t] += (0..m)
                .map(|b| (var_bar[(t, b)] + var_bar[(b, t)]) * mean[b])
                .sum::<f64>();
        }
        for jj in 0..j {
            let mut wj_bar = 0.0;
            for a in 0..m {
                for b in 0..m {
                    let cc = -(mean[a] * sc(jj, b)) + -(mean[b] * sc(jj, a));
                    let off = sc(jj, a) * sc(jj, b);
                    let cc_bar = var_bar[(a, b)] * bx.wj[jj] / bx.w;
                    let off_bar = offs_bar[(a, b)] * bx.wj[jj] / bx.w;
                    wj_bar += (var_bar[(a, b)] * cc + offs_bar[(a, b)] * off) / bx.w;
                    ba.w -=
                        (var_bar[(a, b)] * cc + offs_bar[(a, b)] * off) * bx.wj[jj] / (bx.w * bx.w);
                    // cc = -μ_a sC_b - μ_b sC_a.
                    adj.mean[a] -= cc_bar * sc(jj, b);
                    adj.mean[b] -= cc_bar * sc(jj, a);
                    let (ia, ib) = (jj * m + a, jj * m + b);
                    let sc_bar_a = -cc_bar * mean[b] + off_bar * sc(jj, b);
                    let sc_bar_b = -cc_bar * mean[a] + off_bar * sc(jj, a);
                    // sC = s C.
                    s_bar[a] += sc_bar_a * c[ia];
                    c_bar[ia] += sc_bar_a * s[a];
                    s_bar[b] += sc_bar_b * c[ib];
                    c_bar[ib] += sc_bar_b * s[b];
                }
            }
            ba.wj[jj] += wj_bar;
        }
        // C also enters R; R and C go back to G, W_jm, W_j and W.
        for idx in 0..j * m {
            let jj = idx / m;
            c_bar[idx] += r_bar[idx] * bx.wj[jj] / bx.w;
        }
        for idx in 0..j * m {
            let jj = idx / m;
            ba.wj[jj] += r_bar[idx] * c[idx] / bx.w;
            ba.w -= r_bar[idx] * r[idx] / bx.w;
            ba.gdiff[idx] += c_bar[idx] / bx.wm[idx];
            ba.wm[idx] -= c_bar[idx] * c[idx] / bx.wm[idx];
        }
        for t in 0..m {
            adj.variance[t] += s_bar[t] / (2.0 * s[t]);
        }
        bx.backward(ba, &mut adj.mean, &mut adj.variance);
    }
    h
}

/// Backward of `dws_t = Σ_j W_j v_t V_jt / W_jt / W` with adjoint `dws_bar`.
fn dws_backward(
    bx: &Boxes,
    variance: &[f64],
    t: usize,
    dws_bar: f64,
    ba: &mut BoxAdjoints,
    var_bar: &mut [f64],
) {
    let m = bx.m;
    for jj in 0..bx.j {
        let idx = jj * m + t;
        let term = bx.wj[jj] * variance[t] * bx.v(idx) / bx.wm[idx] / bx.w;
        let common = dws_bar / (bx.wm[idx] * bx.w);
        ba.wj[jj] += common * variance[t] * bx.v(idx);
        var_bar[t] += common * bx.wj[jj] * bx.v(idx);
        ba.v[idx] += common * bx.wj[jj] * variance[t];
        ba.wm[idx] -= dws_bar * term / bx.wm[idx];
        ba.w -= dws_bar * term / bx.w;
    }
}

/// Backward of `R_jt = C_jt W_j / W` and `C_jt = G_jt / W_jt` (only_diagonal).
fn moments_backward(bx: &Boxes, c: &[f64], r: &[f64], r_bar: &[f64], ba: &mut BoxAdjoints) {
    let m = bx.m;
    for idx in 0..bx.j * m {
        let jj = idx / m;
        let c_bar = r_bar[idx] * bx.wj[jj] / bx.w;
        ba.wj[jj] += r_bar[idx] * c[idx] / bx.w;
        ba.w -= r_bar[idx] * r[idx] / bx.w;
        ba.gdiff[idx] += c_bar / bx.wm[idx];
        ba.wm[idx] -= c_bar * c[idx] / bx.wm[idx];
    }
}

/// Adjoints of the Monte-Carlo inputs of one point and Pareto sample.
pub(crate) struct McAdjoint {
    /// Of the samples (`S × M`).
    pub samples: Matrix,
    /// Of the log-densities (`S`).
    pub log_prob: Vec<f64>,
}

/// `_compute_entropy_monte_carlo` for one point and one Pareto sample, with
/// the noisy posterior samples (`S × M`) and their log-densities (`S`).
fn entropy_monte_carlo(
    cells: &Cells<'_>,
    mean: &[f64],
    variance: &[f64],
    noise: &[f64],
    samples: &Matrix,
    log_prob: &[f64],
    adj: Option<(&mut StatsAdjoint, &mut McAdjoint)>,
) -> f64 {
    let (j, m) = (cells.num_cells, cells.m);
    let ns = samples.nrows();
    let bx = Boxes::new(cells, mean, variance);
    let vpn: Vec<f64> = variance.iter().zip(noise).map(|(v, o)| v + o).collect();
    let sqrt_vpn: Vec<f64> = vpn.iter().map(|v| v.sqrt()).collect();
    let rho: Vec<f64> = (0..m).map(|t| (variance[t] / vpn[t]).sqrt()).collect();
    let mut denom = vec![0.0; m];
    let mut denom_active = vec![false; m];
    for t in 0..m {
        let (c, active) = clamp_min_active(1.0 - rho[t] * rho[t], CLAMP_LB);
        denom[t] = c.sqrt();
        denom_active[t] = active;
    }
    let log_w = bx.w.ln();
    let mut total = 0.0;
    // Per sample: z, g', Φ(g'), W'_jm, W'_j, W', W ratio, log term.
    struct SampleTerms {
        z: Vec<f64>,
        gn: [Vec<f64>; 2],
        wm: Vec<f64>,
        wm_active: Vec<bool>,
        wj: Vec<f64>,
        w: f64,
        w_active: bool,
        ratio: f64,
        log_term: f64,
    }
    let mut terms: Vec<SampleTerms> = Vec::with_capacity(if adj.is_some() { ns } else { 0 });
    for si in 0..ns {
        let z: Vec<f64> = (0..m)
            .map(|t| (samples[(si, t)] - mean[t]) / sqrt_vpn[t])
            .collect();
        let mut gn = [vec![0.0; j * m], vec![0.0; j * m]];
        let mut wm = vec![0.0; j * m];
        let mut wm_active = vec![false; j * m];
        for idx in 0..j * m {
            let t = idx % m;
            for (k, gk) in gn.iter_mut().enumerate() {
                gk[idx] = (bx.g[k][idx] - rho[t] * z[t]) / denom[t];
            }
            (wm[idx], wm_active[idx]) =
                clamp_min_active(normal_cdf(gn[1][idx]) - normal_cdf(gn[0][idx]), CLAMP_LB);
        }
        let wj: Vec<f64> = (0..j)
            .map(|jj| {
                wm[jj * m..(jj + 1) * m]
                    .iter()
                    .map(|v| v.ln())
                    .sum::<f64>()
                    .exp()
            })
            .collect();
        let sum: f64 = wj.iter().sum();
        // W.clamp_max(1.0): no gradient at W == 1 either (torch ≥ 2.14).
        let (w, w_active) = if sum > 1.0 {
            (1.0, false)
        } else {
            (sum, sum < 1.0)
        };
        let ratio = (w.ln() - log_w).exp();
        let log_term = w.ln() + log_prob[si];
        total += ratio * log_term;
        if adj.is_some() {
            terms.push(SampleTerms {
                z,
                gn,
                wm,
                wm_active,
                wj,
                w,
                w_active,
                ratio,
                log_term,
            });
        }
    }
    let mc = -(total / ns as f64);
    let h = mc + log_w;
    if let Some((a, mca)) = adj {
        let mut ba = BoxAdjoints::zeros(j, m);
        let mut log_w_bar = 1.0;
        let mut rho_bar = vec![0.0; m];
        let mut denom_bar = vec![0.0; m];
        let mut vpn_bar = vec![0.0; m];
        let term_bar = -1.0 / ns as f64;
        for (si, st) in terms.iter().enumerate() {
            let ratio_bar = term_bar * st.log_term;
            let log_term_bar = term_bar * st.ratio;
            mca.log_prob[si] += log_term_bar;
            let log_wn_bar = log_term_bar + ratio_bar * st.ratio;
            log_w_bar -= ratio_bar * st.ratio;
            let wn_bar = log_wn_bar / st.w;
            let mut wj_bar = vec![0.0; j];
            if st.w_active {
                wj_bar.iter_mut().for_each(|v| *v = wn_bar);
            }
            let mut z_bar = vec![0.0; m];
            for idx in 0..j * m {
                let (jj, t) = (idx / m, idx % m);
                let wm_bar = if st.wm_active[idx] {
                    wj_bar[jj] * st.wj[jj] * (1.0 / st.wm[idx])
                } else {
                    0.0
                };
                for k in 0..2 {
                    let sign = if k == 1 { 1.0 } else { -1.0 };
                    let gn = st.gn[k][idx];
                    let gn_bar = sign * wm_bar * normal_cdf_grad(gn);
                    // g' = (g - ρ z) / d.
                    ba.g[k][idx] += gn_bar / denom[t];
                    rho_bar[t] -= gn_bar * st.z[t] / denom[t];
                    z_bar[t] -= gn_bar * rho[t] / denom[t];
                    denom_bar[t] -= gn_bar * gn / denom[t];
                }
            }
            // z = (y - μ) / √vpn.
            for t in 0..m {
                mca.samples[(si, t)] += z_bar[t] / sqrt_vpn[t];
                a.mean[t] -= z_bar[t] / sqrt_vpn[t];
                vpn_bar[t] -= z_bar[t] * st.z[t] / sqrt_vpn[t] / (2.0 * sqrt_vpn[t]);
            }
        }
        for t in 0..m {
            // d = √clamp_min(1 - ρ², eps).
            if denom_active[t] {
                let c_bar = denom_bar[t] / (2.0 * denom[t]);
                rho_bar[t] -= c_bar * 2.0 * rho[t];
            }
            // ρ = √(v / vpn).
            let q_bar = rho_bar[t] / (2.0 * rho[t]);
            a.variance[t] += q_bar / vpn[t];
            vpn_bar[t] -= q_bar * variance[t] / (vpn[t] * vpn[t]);
            a.variance[t] += vpn_bar[t];
            a.noise[t] += vpn_bar[t];
        }
        ba.w = log_w_bar / bx.w;
        bx.backward(ba, &mut a.mean, &mut a.variance);
    }
    h
}

// ---------------------------------------------------------------------------
// Public estimators (BoTorch's `_compute_entropy_*` for one point)
// ---------------------------------------------------------------------------

/// The posterior statistics of one candidate point under every Pareto
/// sample (`P × M` each, BoTorch's `num_pareto_samples x 1 x M` slices of
/// the `batch_shape x num_pareto_samples x q x 1 x M` tensors).
#[derive(Clone, Debug, PartialEq)]
pub struct EntropyStatistics {
    /// The posterior mean.
    pub mean: Matrix,
    /// The posterior variance without observation noise.
    pub variance: Matrix,
    /// The observation noise.
    pub observation_noise: Matrix,
}

fn check_statistics(cells: &CellBounds, stats: &EntropyStatistics) -> Result<(usize, usize)> {
    let p = cells.batch_size();
    let m = cells.num_outcomes();
    if cells.batch_shape().len() != 1 {
        return Err(BotorchError::Unsupported(
            "The hypercell_bounds should have a shape of `num_pareto_samples x 2 x num_boxes x \
             num_objectives`."
                .into(),
        ));
    }
    for s in [&stats.mean, &stats.variance, &stats.observation_noise] {
        if s.shape() != (p, m) {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected statistics of shape {p}x{m} (num_pareto_samples x M), got {}x{}",
                s.nrows(),
                s.ncols()
            )));
        }
    }
    Ok((p, m))
}

fn row(m: &Matrix, i: usize) -> Vec<f64> {
    m.row(i).iter().copied().collect()
}

/// BoTorch's `_compute_entropy_noiseless` for one point: the entropy of the
/// noiseless posterior truncated to the dominated region of each Pareto
/// sample, plus the noise, averaged over the Pareto samples (`"0"`).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for statistics of the wrong shape.
pub fn compute_entropy_noiseless(cells: &CellBounds, stats: &EntropyStatistics) -> Result<f64> {
    let (p, _) = check_statistics(cells, stats)?;
    let total: f64 = (0..p)
        .map(|k| {
            entropy_noiseless(
                &Cells::of(cells, k),
                &row(&stats.mean, k),
                &row(&stats.variance, k),
                &row(&stats.observation_noise, k),
                None,
            )
        })
        .sum();
    Ok(total / p as f64)
}

/// BoTorch's `_compute_entropy_upper_bound` for one point: the entropy of a
/// Gaussian with the moments of the truncated noisy posterior (`"LB"`; with
/// `only_diagonal`, of its marginals, `"LB2"`), averaged over the Pareto
/// samples.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for statistics of the wrong shape.
pub fn compute_entropy_upper_bound(
    cells: &CellBounds,
    stats: &EntropyStatistics,
    only_diagonal: bool,
) -> Result<f64> {
    let (p, _) = check_statistics(cells, stats)?;
    let total: f64 = (0..p)
        .map(|k| {
            entropy_upper_bound(
                &Cells::of(cells, k),
                &row(&stats.mean, k),
                &row(&stats.variance, k),
                &row(&stats.observation_noise, k),
                only_diagonal,
                None,
            )
        })
        .sum();
    Ok(total / p as f64)
}

/// BoTorch's `_compute_entropy_monte_carlo` for one point: a Monte-Carlo
/// estimate with `S` noisy posterior samples per Pareto sample
/// (`samples[k]` is `S × M`) and their log-densities (`samples_log_prob`,
/// `S × P`), averaged over the Pareto samples (`"MC"`).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inputs of the wrong shape.
pub fn compute_entropy_monte_carlo(
    cells: &CellBounds,
    stats: &EntropyStatistics,
    samples: &[Matrix],
    samples_log_prob: &Matrix,
) -> Result<f64> {
    let (p, m) = check_statistics(cells, stats)?;
    let ns = samples_log_prob.nrows();
    if samples.len() != p
        || samples.iter().any(|s| s.shape() != (ns, m))
        || samples_log_prob.ncols() != p
        || ns == 0
    {
        return Err(BotorchError::DimensionMismatch(format!(
            "expected {p} sample matrices of shape S x {m} and log-densities of shape S x {p}"
        )));
    }
    let total: f64 = (0..p)
        .map(|k| {
            let lp: Vec<f64> = samples_log_prob.column(k).iter().copied().collect();
            entropy_monte_carlo(
                &Cells::of(cells, k),
                &row(&stats.mean, k),
                &row(&stats.variance, k),
                &row(&stats.observation_noise, k),
                &samples[k],
                &lp,
                None,
            )
        })
        .sum();
    Ok(total / p as f64)
}

/// The derivatives of an entropy estimate of one point (see
/// [`compute_entropy_with_grad`]).
#[derive(Clone, Debug, PartialEq)]
pub struct EntropyGradient {
    /// With respect to the mean (`P × M`).
    pub mean: Matrix,
    /// With respect to the variance (`P × M`).
    pub variance: Matrix,
    /// With respect to the observation noise (`P × M`).
    pub observation_noise: Matrix,
    /// With respect to the MC samples (`P` matrices of `S × M`), for `"MC"`.
    pub samples: Option<Vec<Matrix>>,
    /// With respect to the MC log-densities (`S × P`), for `"MC"`.
    pub samples_log_prob: Option<Matrix>,
}

/// An entropy estimate of one point (the function of
/// [`compute_entropy_noiseless`], [`compute_entropy_upper_bound`] or
/// [`compute_entropy_monte_carlo`] that `estimation_type` selects; `mc` holds
/// the samples and log-densities of `"MC"`) and its exact derivatives with
/// respect to all inputs.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inputs of the wrong shape and
/// [`BotorchError::InvalidArgument`] if `"MC"` lacks its samples.
pub fn compute_entropy_with_grad(
    estimation_type: EstimationType,
    cells: &CellBounds,
    stats: &EntropyStatistics,
    mc: Option<(&[Matrix], &Matrix)>,
) -> Result<(f64, EntropyGradient)> {
    let (p, m) = check_statistics(cells, stats)?;
    let mut grad = EntropyGradient {
        mean: Matrix::zeros(p, m),
        variance: Matrix::zeros(p, m),
        observation_noise: Matrix::zeros(p, m),
        samples: None,
        samples_log_prob: None,
    };
    let mut total = 0.0;
    if estimation_type == EstimationType::MonteCarlo {
        let Some((samples, lp)) = mc else {
            return Err(BotorchError::InvalidArgument(
                "the \"MC\" estimate requires samples and their log-densities".into(),
            ));
        };
        // Checks the shapes.
        compute_entropy_monte_carlo(cells, stats, samples, lp)?;
        let ns = lp.nrows();
        grad.samples = Some(vec![Matrix::zeros(ns, m); p]);
        grad.samples_log_prob = Some(Matrix::zeros(ns, p));
    }
    for k in 0..p {
        let (mean, var, noise) = (
            row(&stats.mean, k),
            row(&stats.variance, k),
            row(&stats.observation_noise, k),
        );
        let cells_k = Cells::of(cells, k);
        let mut a = StatsAdjoint::zeros(m);
        let h = match (estimation_type, mc) {
            (EstimationType::Noiseless, _) => {
                entropy_noiseless(&cells_k, &mean, &var, &noise, Some(&mut a))
            }
            (EstimationType::LowerBound, _) => {
                entropy_upper_bound(&cells_k, &mean, &var, &noise, false, Some(&mut a))
            }
            (EstimationType::LowerBound2, _) => {
                entropy_upper_bound(&cells_k, &mean, &var, &noise, true, Some(&mut a))
            }
            (EstimationType::MonteCarlo, Some((samples, lp))) => {
                let lpk: Vec<f64> = lp.column(k).iter().copied().collect();
                let mut mca = McAdjoint {
                    samples: Matrix::zeros(lp.nrows(), m),
                    log_prob: vec![0.0; lp.nrows()],
                };
                let h = entropy_monte_carlo(
                    &cells_k,
                    &mean,
                    &var,
                    &noise,
                    &samples[k],
                    &lpk,
                    Some((&mut a, &mut mca)),
                );
                let scale = 1.0 / p as f64;
                if let Some(gs) = grad.samples.as_mut() {
                    gs[k] = mca.samples * scale;
                }
                if let Some(gl) = grad.samples_log_prob.as_mut() {
                    for (si, v) in mca.log_prob.iter().enumerate() {
                        gl[(si, k)] = v * scale;
                    }
                }
                h
            }
            (EstimationType::MonteCarlo, None) => unreachable!("checked above"),
        };
        total += h;
        for t in 0..m {
            grad.mean[(k, t)] = a.mean[t] / p as f64;
            grad.variance[(k, t)] = a.variance[t] / p as f64;
            grad.observation_noise[(k, t)] = a.noise[t] / p as f64;
        }
    }
    Ok((total / p as f64, grad))
}

// ---------------------------------------------------------------------------
// The conditional entropy of one candidate point
// ---------------------------------------------------------------------------

/// The posterior of one candidate point under one Pareto sample, in the
/// form the posteriors of BoTorch provide it: the mean, the unclamped
/// variance without noise and the observation noise of every output.
pub(crate) struct RawPointPosterior<'a> {
    pub mean: &'a [f64],
    pub variance: &'a [f64],
    pub noise: &'a [f64],
}

/// The adjoints of a [`RawPointPosterior`] (mean and unclamped variance).
pub(crate) struct RawPointAdjoint {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
}

/// GPyTorch's `variance.clamp_min(1e-10)` followed by BoTorch's
/// `clamp_min(eps)`, and whether the gradient passes.
#[inline]
fn clamp_variance(v: f64) -> (f64, bool) {
    let (v1, a1) = clamp_min_active(v, MIN_VARIANCE);
    let (v2, a2) = clamp_min_active(v1, CLAMP_LB);
    (v2, a1 && a2)
}

/// The conditional-entropy settings shared by MES and JES.
pub(crate) struct ConditionalEntropy<'a> {
    pub cells: &'a CellBounds,
    pub estimation_type: EstimationType,
    /// JES clamps the observation noise (`(vpn - v).clamp_min(eps)`), MES
    /// does not.
    pub clamp_noise: bool,
    /// The MC base samples (`S × M`) for [`EstimationType::MonteCarlo`].
    pub base_samples: Option<&'a Matrix>,
}

impl ConditionalEntropy<'_> {
    /// The entropy of one point under Pareto sample `p` (BoTorch's
    /// per-sample entropy before the mean over the samples) and, with
    /// `with_grad`, the adjoints of its raw posterior.
    pub fn sample_entropy(
        &self,
        p: usize,
        post: &RawPointPosterior<'_>,
        with_grad: bool,
    ) -> (f64, Option<RawPointAdjoint>) {
        let m = post.mean.len();
        let cells = Cells::of(self.cells, p);
        let mut vpn_raw = vec![0.0; m];
        let mut vpn = vec![0.0; m];
        let mut vpn_active = vec![false; m];
        let mut var = vec![0.0; m];
        let mut var_active = vec![false; m];
        let mut noise = vec![0.0; m];
        let mut noise_active = vec![true; m];
        for t in 0..m {
            vpn_raw[t] = post.variance[t] + post.noise[t];
            (vpn[t], vpn_active[t]) = clamp_variance(vpn_raw[t]);
            (var[t], var_active[t]) = clamp_variance(post.variance[t]);
            let diff = vpn[t] - var[t];
            (noise[t], noise_active[t]) = if self.clamp_noise {
                clamp_min_active(diff, CLAMP_LB)
            } else {
                (diff, true)
            };
        }
        let mut adj = with_grad.then(|| StatsAdjoint::zeros(m));
        let mut mc_adj = None;
        let h = match self.estimation_type {
            EstimationType::Noiseless => {
                entropy_noiseless(&cells, post.mean, &var, &noise, adj.as_mut())
            }
            EstimationType::LowerBound => {
                entropy_upper_bound(&cells, post.mean, &var, &noise, false, adj.as_mut())
            }
            EstimationType::LowerBound2 => {
                entropy_upper_bound(&cells, post.mean, &var, &noise, true, adj.as_mut())
            }
            EstimationType::MonteCarlo => {
                let z = self
                    .base_samples
                    .expect("the MC estimate requires base samples");
                let ns = z.nrows();
                // Samples of N(μ, diag(v + noise)) and their log-densities
                // (`rsample` with a Cholesky root, `MultivariateNormal.log_prob`).
                let sd: Vec<f64> = vpn_raw.iter().map(|v| v.sqrt()).collect();
                let samples = Matrix::from_fn(ns, m, |si, t| post.mean[t] + sd[t] * z[(si, t)]);
                let log_prob: Vec<f64> = (0..ns)
                    .map(|si| {
                        let mut inv_quad = 0.0;
                        let mut logdet = 0.0;
                        for t in 0..m {
                            let u = (samples[(si, t)] - post.mean[t]) / sd[t];
                            inv_quad += u * u;
                            logdet += 2.0 * sd[t].ln();
                        }
                        -0.5 * (inv_quad + logdet + m as f64 * LOG_2PI)
                    })
                    .collect();
                let mut mca = with_grad.then(|| McAdjoint {
                    samples: Matrix::zeros(ns, m),
                    log_prob: vec![0.0; ns],
                });
                let h = entropy_monte_carlo(
                    &cells,
                    post.mean,
                    &var,
                    &noise,
                    &samples,
                    &log_prob,
                    adj.as_mut().zip(mca.as_mut()),
                );
                mc_adj = mca.map(|a| (a, z, sd));
                h
            }
        };
        let Some(adj) = adj else {
            return (h, None);
        };
        let mut mean_bar = adj.mean;
        let mut vraw_bar = vec![0.0; m];
        let mut vpn_raw_bar = vec![0.0; m];
        for t in 0..m {
            let noise_bar = if noise_active[t] { adj.noise[t] } else { 0.0 };
            // noise = vpn - var.
            if vpn_active[t] {
                vpn_raw_bar[t] += noise_bar;
            }
            if var_active[t] {
                vraw_bar[t] += adj.variance[t] - noise_bar;
            }
        }
        if let Some((mca, z, sd)) = mc_adj {
            for si in 0..z.nrows() {
                for t in 0..m {
                    // log p = -0.5 (Σ u² + Σ log sd² + M log 2π), u = (y - μ) / sd.
                    let u = (post.mean[t] + sd[t] * z[(si, t)] - post.mean[t]) / sd[t];
                    let lp_bar = mca.log_prob[si];
                    let y_bar = mca.samples[(si, t)] - lp_bar * u / sd[t];
                    mean_bar[t] += lp_bar * u / sd[t];
                    let sd_bar_lp = lp_bar * (u * u / sd[t] - 1.0 / sd[t]);
                    // y = μ + sd z.
                    mean_bar[t] += y_bar;
                    let sd_bar = sd_bar_lp + y_bar * z[(si, t)];
                    vpn_raw_bar[t] += sd_bar / (2.0 * sd[t]);
                }
            }
        }
        for t in 0..m {
            vraw_bar[t] += vpn_raw_bar[t];
        }
        (
            h,
            Some(RawPointAdjoint {
                mean: mean_bar,
                variance: vraw_bar,
            }),
        )
    }
}

/// `0.5 log det Σ` of the joint posterior with noise over all outputs at the
/// candidates (BoTorch's initial entropy without its constant) and, with
/// `grad`, its gradient with respect to the first `q` rows of `x`.
pub(crate) fn initial_entropy(
    gps: &[&SingleTaskGP],
    x: &Matrix,
    q: usize,
    grad: Option<&mut Matrix>,
) -> Result<f64> {
    let mut logdet = 0.0;
    let mut grads = Vec::new();
    for gp in gps {
        let post = gp.posterior(x, true)?;
        let lu = Lu::new(&post.covariance);
        logdet += lu.logdet();
        if grad.is_some() {
            let adj = lu.inverse().transpose() * 0.5;
            let mean_adj = Vector::zeros(x.nrows());
            grads.push(gp.joint_posterior_vjp(x, 0..q, &mean_adj, &adj)?);
        }
    }
    if let Some(g) = grad {
        for gi in grads {
            *g += gi;
        }
    }
    Ok(0.5 * logdet)
}

// ---------------------------------------------------------------------------
// Conditioning on the Pareto points
// ---------------------------------------------------------------------------

/// The posterior of a single-output GP conditioned, separately, on each of
/// `b` sets of `k` noisy observations: BoTorch's `condition_on_observations`
/// of a `SingleTaskGP` with a batch of observations and the likelihood's
/// noise, as qLowerBoundMultiObjectiveJointEntropySearch conditions on its
/// sampled Pareto sets and fronts.
///
/// With the posterior `μ, Σ` of the GP (original outcome space), the
/// observations `y_i` at the raw inputs `Z_i` of set `i` and the noise
/// variance `σ²`, the conditional posterior is the rank-`k` update
///
/// ```text
/// μ_i(x) = μ(x) + Σ(x, Z_i) A_i⁻¹ (y_i - μ(Z_i)),
/// σ²_i(x) = Σ(x, x) - Σ(x, Z_i) A_i⁻¹ Σ(Z_i, x),     A_i = Σ(Z_i, Z_i) + σ² I,
/// ```
///
/// which is the posterior of the GP with the observations appended to its
/// training data. The cross-covariances of all sets are computed together
/// from cached quantities of the GP ([`SingleTaskGP::baseline_cache`]).
#[derive(Clone, Debug)]
pub struct ConditionedOutput {
    baseline: BaselineCache,
    chol: Vec<CholeskyFactor>,
    alpha: Vec<Vector>,
    noise: f64,
    k: usize,
    d: usize,
    n_train: usize,
}

impl ConditionedOutput {
    /// Conditions `gp` on the observations `values[i]` (length `k`, original
    /// outcome space) at the raw inputs `sets[i]` (`k × d`) with the noise
    /// variance `noise` (original outcome space).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
    /// [`BotorchError::InvalidArgument`] for no sets or a negative noise,
    /// and errors of the factorizations.
    pub fn new(
        gp: &SingleTaskGP,
        sets: &[Matrix],
        values: &[Vec<f64>],
        noise: f64,
    ) -> Result<Self> {
        let Some(first) = sets.first() else {
            return Err(BotorchError::InvalidArgument(
                "at least one set of observations is required".into(),
            ));
        };
        let (k, d) = first.shape();
        if d != gp.dim() || k == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "the observation inputs must be non-empty `k x {}` matrices, got {k}x{d}",
                gp.dim()
            )));
        }
        if values.len() != sets.len()
            || sets.iter().any(|s| s.shape() != (k, d))
            || values.iter().any(|v| v.len() != k)
        {
            return Err(BotorchError::DimensionMismatch(
                "every set of observations needs the same number of inputs and values".into(),
            ));
        }
        if noise.is_nan() || noise < 0.0 {
            return Err(BotorchError::InvalidArgument(format!(
                "the observation noise must be non-negative, got {noise}"
            )));
        }
        let all = Matrix::from_fn(sets.len() * k, d, |r, c| sets[r / k][(r % k, c)]);
        let baseline = gp.baseline_cache(&all)?;
        let mut chol = Vec::with_capacity(sets.len());
        let mut alpha = Vec::with_capacity(sets.len());
        for (z, y) in sets.iter().zip(values) {
            let post = gp.posterior(z, false)?;
            let mut a = post.covariance;
            for i in 0..k {
                a[(i, i)] += noise;
            }
            let l = psd_safe_cholesky(&a)?;
            let resid =
                Vector::from_iterator(k, y.iter().zip(post.mean.iter()).map(|(y, m)| y - m));
            alpha.push(l.solve_vec(&resid));
            chol.push(l);
        }
        Ok(Self {
            baseline,
            chol,
            alpha,
            noise,
            k,
            d,
            n_train: gp.num_train(),
        })
    }

    /// The number of sets of observations.
    pub fn num_sets(&self) -> usize {
        self.chol.len()
    }

    /// The noise variance of the observations.
    pub fn noise(&self) -> f64 {
        self.noise
    }

    fn check_gp(&self, gp: &SingleTaskGP) -> Result<()> {
        if gp.dim() != self.d || gp.num_train() != self.n_train {
            return Err(BotorchError::DimensionMismatch(
                "the GP differs from the one that was conditioned".into(),
            ));
        }
        Ok(())
    }

    /// The conditional means and (unclamped, noiseless) variances of every
    /// row of `x` (`n × d`, each point on its own) under every set:
    /// `n × b` matrices.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a different GP or a column
    /// mismatch, and errors of the GP.
    pub fn marginals(&self, gp: &SingleTaskGP, x: &Matrix) -> Result<(Matrix, Matrix)> {
        self.check_gp(gp)?;
        let cp = gp.cross_posterior(&self.baseline, x)?;
        let (n, b, k) = (x.nrows(), self.num_sets(), self.k);
        let mut mean = Matrix::zeros(n, b);
        let mut var = Matrix::zeros(n, b);
        for i in 0..n {
            for p in 0..b {
                let c = Vector::from_iterator(k, (0..k).map(|r| cp.cross[(i, p * k + r)]));
                mean[(i, p)] = cp.mean[i] + c.dot(&self.alpha[p]);
                let w = self.chol[p].solve_lower_vec(&c);
                var[(i, p)] = cp.covariance[(i, i)] - w.dot(&w);
            }
        }
        Ok((mean, var))
    }

    /// The gradient with respect to `x` of `Σ_ip mean_adjoint[i, p] μ_p(x_i)
    /// + var_adjoint[i, p] σ²_p(x_i)` (see [`ConditionedOutput::marginals`]),
    /// for the first `q` rows of `x` (`q × d`).
    ///
    /// # Errors
    /// See [`ConditionedOutput::marginals`].
    pub fn marginals_vjp(
        &self,
        gp: &SingleTaskGP,
        x: &Matrix,
        q: usize,
        mean_adjoint: &Matrix,
        var_adjoint: &Matrix,
    ) -> Result<Matrix> {
        self.check_gp(gp)?;
        let (n, b, k) = (x.nrows(), self.num_sets(), self.k);
        if mean_adjoint.shape() != (n, b) || var_adjoint.shape() != (n, b) || q > n {
            return Err(BotorchError::DimensionMismatch(format!(
                "the adjoints of {n} points under {b} sets must be {n}x{b}"
            )));
        }
        let cp = gp.cross_posterior(&self.baseline, x)?;
        let mut mbar = Vector::zeros(n);
        let mut cross_bar = Matrix::zeros(n, b * k);
        let mut cov_bar = Matrix::zeros(n, n);
        for i in 0..q {
            for p in 0..b {
                let c = Vector::from_iterator(k, (0..k).map(|r| cp.cross[(i, p * k + r)]));
                let (ma, va) = (mean_adjoint[(i, p)], var_adjoint[(i, p)]);
                mbar[i] += ma;
                cov_bar[(i, i)] += va;
                let w = self.chol[p].solve_lower_vec(&c);
                let u = self.chol[p].solve_lower_transpose_vec(&w);
                for r in 0..k {
                    cross_bar[(i, p * k + r)] += ma * self.alpha[p][r] - 2.0 * va * u[r];
                }
            }
        }
        let g = gp.cross_posterior_vjp(&self.baseline, x, &mbar, &cross_bar, &cov_bar)?;
        Ok(g.rows(0, q).into_owned())
    }
}

// ---------------------------------------------------------------------------
// The acquisition functions
// ---------------------------------------------------------------------------

/// Options of the lower-bound entropy search acquisition functions
/// (BoTorch's keyword arguments).
#[derive(Clone, Debug, Default)]
pub struct LowerBoundMultiObjectiveEntropySearchOptions {
    /// Pending points (`m × d`), appended to every candidate set.
    pub x_pending: Option<Matrix>,
    /// The entropy estimate (default `"LB"`).
    pub estimation_type: EstimationType,
    /// The sampler of the `"MC"` estimate (default: 64 Sobol samples with a
    /// random seed, BoTorch's `SobolQMCNormalSampler(sample_shape=[64])`);
    /// the base samples have one column per output.
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
}

/// The state shared by the lower-bound entropy search acquisition functions
/// (BoTorch's `LowerBoundMultiObjectiveEntropySearch`): the model, the
/// hypercell bounds of the sampled Pareto fronts (`P × J × M`), the entropy
/// estimate, the MC sampler and the pending points.
#[derive(Debug)]
pub struct LowerBoundMultiObjectiveEntropySearch {
    model: Arc<dyn Model>,
    hypercell_bounds: CellBounds,
    estimation_type: EstimationType,
    sampler: Arc<SobolQMCNormalSampler>,
    x_pending: RwLock<Option<Matrix>>,
}

impl LowerBoundMultiObjectiveEntropySearch {
    /// # Errors
    /// [`BotorchError::Unsupported`] for batched models, models without
    /// independent exact GP outputs and hypercell bounds that are not
    /// batched over the Pareto samples, [`BotorchError::DimensionMismatch`]
    /// if the cells do not have one dimension per output.
    pub(crate) fn new(
        model: Arc<dyn Model>,
        hypercell_bounds: CellBounds,
        options: LowerBoundMultiObjectiveEntropySearchOptions,
    ) -> Result<Self> {
        if !model.batch_shape().is_empty() {
            return Err(BotorchError::Unsupported(
                "Batch GP models (e.g. fantasized models) are not supported.".into(),
            ));
        }
        check_unbatched(model.as_ref())?;
        if model.output_gps().is_none() {
            return Err(BotorchError::Unsupported(
                "the entropy search acquisition functions require a model of independent exact \
                 GPs (a SingleTaskGP or a ModelListGP)"
                    .into(),
            ));
        }
        if hypercell_bounds.batch_shape().len() != 1 {
            return Err(BotorchError::Unsupported(
                "The hypercell_bounds should have a shape of `num_pareto_samples x 2 x num_boxes \
                 x num_objectives`."
                    .into(),
            ));
        }
        if hypercell_bounds.num_outcomes() != model.num_outputs() {
            return Err(BotorchError::DimensionMismatch(format!(
                "the hypercell bounds have {} outcomes but the model has {} outputs",
                hypercell_bounds.num_outcomes(),
                model.num_outputs()
            )));
        }
        if let Some(p) = &options.x_pending {
            check_pending(p, model.dim())?;
        }
        let sampler = match options.sampler {
            Some(s) => s,
            None => Arc::new(SobolQMCNormalSampler::new(
                DEFAULT_ENTROPY_MC_SAMPLES,
                None,
            )?),
        };
        Ok(Self {
            model,
            hypercell_bounds,
            estimation_type: options.estimation_type,
            sampler,
            x_pending: RwLock::new(options.x_pending),
        })
    }

    /// The model.
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// The hypercell bounds (batch shape `[num_pareto_samples]`).
    pub fn hypercell_bounds(&self) -> &CellBounds {
        &self.hypercell_bounds
    }

    /// The number of Pareto samples.
    pub fn num_pareto_samples(&self) -> usize {
        self.hypercell_bounds.batch_size()
    }

    /// The entropy estimate.
    pub fn estimation_type(&self) -> EstimationType {
        self.estimation_type
    }

    /// The MC sampler.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.sampler
    }

    pub(crate) fn pending(&self) -> Option<Matrix> {
        self.x_pending
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub(crate) fn set_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.model.dim())?;
        }
        *self.x_pending.write().unwrap_or_else(|e| e.into_inner()) = x_pending.cloned();
        Ok(())
    }

    /// The candidates with the pending points appended.
    pub(crate) fn full_candidates(&self, x: &Matrix) -> Result<Matrix> {
        check_candidates(x, self.model.dim(), None)?;
        Ok(match self.pending() {
            Some(p) if p.nrows() > 0 => {
                let mut full = Matrix::zeros(x.nrows() + p.nrows(), x.ncols());
                full.rows_mut(0, x.nrows()).copy_from(x);
                full.rows_mut(x.nrows(), p.nrows()).copy_from(&p);
                full
            }
            _ => x.clone(),
        })
    }

    /// The MC base samples (`S × M`) of the `"MC"` estimate.
    pub(crate) fn mc_base_samples(&self) -> Result<Option<Arc<Matrix>>> {
        if self.estimation_type == EstimationType::MonteCarlo {
            Ok(Some(self.sampler.base_samples(self.model.num_outputs())?))
        } else {
            Ok(None)
        }
    }

    /// The settings of the conditional entropy (`base_samples` from
    /// [`Self::mc_base_samples`]).
    pub(crate) fn conditional_entropy<'a>(
        &'a self,
        clamp_noise: bool,
        base_samples: Option<&'a Matrix>,
    ) -> ConditionalEntropy<'a> {
        ConditionalEntropy {
            cells: &self.hypercell_bounds,
            estimation_type: self.estimation_type,
            clamp_noise,
            base_samples,
        }
    }
}

/// Options of [`QLowerBoundMultiObjectiveJointEntropySearch`].
pub type QLowerBoundMultiObjectiveJointEntropySearchOptions =
    LowerBoundMultiObjectiveEntropySearchOptions;

/// The lower-bound multi-objective joint entropy search acquisition function
/// (BoTorch's `qLowerBoundMultiObjectiveJointEntropySearch`):
///
/// ```text
/// α(X) = H[p(y_X | D)] - Σ_{x ∈ X} (1/P) Σ_p H[p(y_x | D, (X*_p, Y*_p))],
/// ```
///
/// where the first term is the entropy of the joint posterior with noise at
/// the candidates (and pending points) and the second one sums the entropy
/// estimates of every point under the model conditioned on each sampled
/// Pareto set `X*_p` and front `Y*_p` (with the likelihood's noise),
/// truncated to the dominated region of the front (the hypercells). The
/// value can be negative, and adding points to `X` does not necessarily
/// increase it.
#[derive(Debug)]
pub struct QLowerBoundMultiObjectiveJointEntropySearch {
    base: LowerBoundMultiObjectiveEntropySearch,
    pareto_sets: Vec<Matrix>,
    pareto_fronts: Vec<Matrix>,
    conditioned: Vec<ConditionedOutput>,
}

impl QLowerBoundMultiObjectiveJointEntropySearch {
    /// JES with the sampled Pareto sets (`P` matrices of `N × d`), Pareto
    /// fronts (`P` matrices of `N × M`) and the hypercell bounds of the
    /// fronts (batch shape `[P]`, e.g. from
    /// [`compute_sample_box_decomposition`](super::pareto_samples::compute_sample_box_decomposition)).
    ///
    /// # Errors
    /// [`BotorchError::Runtime`] for fixed-noise models (BoTorch: "FixedNoiseGaussianLikelihood.fantasize
    /// requires a `noise` kwarg"), [`BotorchError::DimensionMismatch`] for
    /// inconsistent shapes, and the errors of
    /// [`LowerBoundMultiObjectiveEntropySearch`] and of the conditioning.
    pub fn new(
        model: Arc<dyn Model>,
        pareto_sets: Vec<Matrix>,
        pareto_fronts: Vec<Matrix>,
        hypercell_bounds: CellBounds,
        options: QLowerBoundMultiObjectiveJointEntropySearchOptions,
    ) -> Result<Self> {
        let base = LowerBoundMultiObjectiveEntropySearch::new(model, hypercell_bounds, options)?;
        let p = base.num_pareto_samples();
        let (d, m) = (base.model.dim(), base.model.num_outputs());
        if pareto_sets.len() != p || pareto_fronts.len() != p {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {p} Pareto sets and fronts (num_pareto_samples of the hypercell \
                 bounds), got {} and {}",
                pareto_sets.len(),
                pareto_fronts.len()
            )));
        }
        let n = pareto_sets[0].nrows();
        if pareto_sets.iter().any(|s| s.shape() != (n, d))
            || pareto_fronts.iter().any(|f| f.shape() != (n, m))
            || n == 0
        {
            return Err(BotorchError::DimensionMismatch(
                "The Pareto set and front should have a shape of `num_pareto_samples x \
                 num_pareto_points x input_dim` and `num_pareto_samples x num_pareto_points x \
                 num_objectives`, respectively"
                    .into(),
            ));
        }
        let gps = base
            .model
            .output_gps()
            .expect("checked by LowerBoundMultiObjectiveEntropySearch");
        let conditioned = gps
            .par_iter()
            .enumerate()
            .map(|(t, gp)| {
                if matches!(gp.likelihood(), Likelihood::FixedNoise { .. }) {
                    return Err(BotorchError::Runtime(
                        "FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg".into(),
                    ));
                }
                let values: Vec<Vec<f64>> = pareto_fronts
                    .iter()
                    .map(|f| f.column(t).iter().copied().collect())
                    .collect();
                ConditionedOutput::new(gp, &pareto_sets, &values, gp.observation_noise())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            base,
            pareto_sets,
            pareto_fronts,
            conditioned,
        })
    }

    /// The shared state (cells, estimate, sampler).
    pub fn base(&self) -> &LowerBoundMultiObjectiveEntropySearch {
        &self.base
    }

    /// The sampled Pareto sets.
    pub fn pareto_sets(&self) -> &[Matrix] {
        &self.pareto_sets
    }

    /// The sampled Pareto fronts.
    pub fn pareto_fronts(&self) -> &[Matrix] {
        &self.pareto_fronts
    }

    /// The conditioned outputs (one per model output).
    pub fn conditioned_outputs(&self) -> &[ConditionedOutput] {
        &self.conditioned
    }

    /// The value and, with `with_grad`, the gradient at the candidates `x`.
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let base = &self.base;
        let full = base.full_candidates(x)?;
        let (n, q, d) = (full.nrows(), x.nrows(), x.ncols());
        let gps = base.model.output_gps().expect("checked at construction");
        let m = gps.len();
        let p = base.num_pareto_samples();
        let mut grad = with_grad.then(|| Matrix::zeros(q, d));
        let h0 = entropy_constant(m) + initial_entropy(&gps, &full, q, grad.as_mut())?;
        // Conditional means and variances: [output] n × P.
        let marginals = gps
            .iter()
            .zip(&self.conditioned)
            .map(|(gp, c)| c.marginals(gp, &full))
            .collect::<Result<Vec<_>>>()?;
        let noise: Vec<f64> = self.conditioned.iter().map(|c| c.noise()).collect();
        let z = base.mc_base_samples()?;
        let ce = base.conditional_entropy(true, z.as_deref());
        let mut cond_total = 0.0;
        let mut mean_adj = vec![Matrix::zeros(n, p); m];
        let mut var_adj = vec![Matrix::zeros(n, p); m];
        for i in 0..n {
            let mut hi = 0.0;
            for k in 0..p {
                let mean: Vec<f64> = marginals.iter().map(|(mu, _)| mu[(i, k)]).collect();
                let var: Vec<f64> = marginals.iter().map(|(_, v)| v[(i, k)]).collect();
                let post = RawPointPosterior {
                    mean: &mean,
                    variance: &var,
                    noise: &noise,
                };
                let (h, adj) = ce.sample_entropy(k, &post, with_grad && i < q);
                hi += h;
                if let Some(a) = adj {
                    for t in 0..m {
                        mean_adj[t][(i, k)] = -a.mean[t] / p as f64;
                        var_adj[t][(i, k)] = -a.variance[t] / p as f64;
                    }
                }
            }
            cond_total += hi / p as f64;
        }
        if let Some(g) = grad.as_mut() {
            for (t, (gp, c)) in gps.iter().zip(&self.conditioned).enumerate() {
                *g += c.marginals_vjp(gp, &full, q, &mean_adj[t], &var_adj[t])?;
            }
        }
        Ok((h0 - cond_total, grad))
    }
}

impl AcquisitionFunction for QLowerBoundMultiObjectiveJointEntropySearch {
    fn model(&self) -> &Arc<dyn Model> {
        &self.base.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.base.pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.base.set_pending(x_pending)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Random cells, statistics and MC inputs of one point.
    fn inputs(
        m: usize,
        j: usize,
        seed: u64,
    ) -> (CellBounds, EntropyStatistics, Vec<Matrix>, Matrix) {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let p = 2;
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for _ in 0..p {
            for jj in 0..j {
                for _ in 0..m {
                    if jj == j - 1 {
                        // A padded (empty) cell.
                        lower.push(0.0);
                        upper.push(0.0);
                    } else if jj == 0 {
                        lower.push(-1e10);
                        upper.push(next() * 2.0 - 0.5);
                    } else {
                        let l = next() * 2.0 - 1.5;
                        lower.push(l);
                        upper.push(l + next());
                    }
                }
            }
        }
        let cells = CellBounds::new(vec![p], j, m, lower, upper).unwrap();
        let stats = EntropyStatistics {
            mean: Matrix::from_fn(p, m, |_, _| next() - 0.5),
            variance: Matrix::from_fn(p, m, |_, _| 0.2 + next()),
            observation_noise: Matrix::from_fn(p, m, |_, _| 0.05 + 0.2 * next()),
        };
        let ns = 6;
        let samples: Vec<Matrix> = (0..p)
            .map(|_| Matrix::from_fn(ns, m, |_, _| 2.0 * next() - 1.0))
            .collect();
        let lp = Matrix::from_fn(ns, p, |_, _| -1.0 - next());
        (cells, stats, samples, lp)
    }

    /// The gradients of the estimators agree with central finite
    /// differences.
    #[test]
    fn estimator_gradients_match_finite_differences() {
        for m in [1, 2, 3] {
            let (cells, stats, samples, lp) = inputs(m, 4, 11 + m as u64);
            let c = Cells::of(&cells, 1);
            let (mean, var, noise) = (
                row(&stats.mean, 1),
                row(&stats.variance, 1),
                row(&stats.observation_noise, 1),
            );
            let lp1: Vec<f64> = lp.column(1).iter().copied().collect();
            for kind in 0..4 {
                let f = |mu: &[f64],
                         v: &[f64],
                         o: &[f64],
                         y: &Matrix,
                         l: &[f64],
                         adj: Option<(&mut StatsAdjoint, &mut McAdjoint)>|
                 -> f64 {
                    match kind {
                        0 => entropy_noiseless(&c, mu, v, o, adj.map(|a| a.0)),
                        1 => entropy_upper_bound(&c, mu, v, o, false, adj.map(|a| a.0)),
                        2 => entropy_upper_bound(&c, mu, v, o, true, adj.map(|a| a.0)),
                        _ => entropy_monte_carlo(&c, mu, v, o, y, l, adj),
                    }
                };
                let mut a = StatsAdjoint::zeros(m);
                let mut mca = McAdjoint {
                    samples: Matrix::zeros(6, m),
                    log_prob: vec![0.0; 6],
                };
                let h = f(
                    &mean,
                    &var,
                    &noise,
                    &samples[1],
                    &lp1,
                    Some((&mut a, &mut mca)),
                );
                assert_eq!(h, f(&mean, &var, &noise, &samples[1], &lp1, None));
                let eps = 1e-6;
                let check = |fd: f64, an: f64, what: &str| {
                    assert!(
                        (fd - an).abs() < 1e-6 * an.abs().max(1.0),
                        "kind {kind} m {m} {what}: fd {fd} analytic {an}"
                    );
                };
                for t in 0..m {
                    for which in 0..3 {
                        let mut plus = [mean.clone(), var.clone(), noise.clone()];
                        let mut minus = plus.clone();
                        plus[which][t] += eps;
                        minus[which][t] -= eps;
                        let fd = (f(&plus[0], &plus[1], &plus[2], &samples[1], &lp1, None)
                            - f(&minus[0], &minus[1], &minus[2], &samples[1], &lp1, None))
                            / (2.0 * eps);
                        let an = [&a.mean, &a.variance, &a.noise][which][t];
                        check(fd, an, ["mean", "variance", "noise"][which]);
                    }
                }
                if kind == 3 {
                    for si in 0..6 {
                        for t in 0..m {
                            let mut yp = samples[1].clone();
                            yp[(si, t)] += eps;
                            let mut ym = samples[1].clone();
                            ym[(si, t)] -= eps;
                            let fd = (f(&mean, &var, &noise, &yp, &lp1, None)
                                - f(&mean, &var, &noise, &ym, &lp1, None))
                                / (2.0 * eps);
                            check(fd, mca.samples[(si, t)], "samples");
                        }
                        let mut lpp = lp1.clone();
                        lpp[si] += eps;
                        let mut lpm = lp1.clone();
                        lpm[si] -= eps;
                        let fd = (f(&mean, &var, &noise, &samples[1], &lpp, None)
                            - f(&mean, &var, &noise, &samples[1], &lpm, None))
                            / (2.0 * eps);
                        check(fd, mca.log_prob[si], "log_prob");
                    }
                }
            }
            // The public functions average over the Pareto samples.
            let h = compute_entropy_upper_bound(&cells, &stats, false).unwrap();
            let h0 = entropy_upper_bound(
                &Cells::of(&cells, 0),
                &row(&stats.mean, 0),
                &row(&stats.variance, 0),
                &row(&stats.observation_noise, 0),
                false,
                None,
            );
            let h1 = entropy_upper_bound(&c, &mean, &var, &noise, false, None);
            assert!((h - (h0 + h1) / 2.0).abs() < 1e-15);
            assert!(compute_entropy_noiseless(&cells, &stats)
                .unwrap()
                .is_finite());
            assert!(compute_entropy_monte_carlo(&cells, &stats, &samples, &lp)
                .unwrap()
                .is_finite());
        }
    }

    #[test]
    fn estimation_type_names() {
        for name in ["0", "LB", "LB2", "MC"] {
            assert_eq!(EstimationType::from_name(name).unwrap().name(), name);
        }
        assert!(matches!(
            EstimationType::from_name("lb"),
            Err(BotorchError::Unsupported(_))
        ));
        assert_eq!(EstimationType::default(), EstimationType::LowerBound);
    }
}
