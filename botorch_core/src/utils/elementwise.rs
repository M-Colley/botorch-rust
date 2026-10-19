//! Element-wise special functions and reductions over flat arrays, with
//! their derivatives: the tensor functions of `botorch.utils.safe_math`,
//! `botorch.utils.probability.utils`, `botorch.utils.probability.bvn` and
//! `botorch.utils.rounding.approximate_round`, for the Python bindings
//! (whose autograd support uses the derivatives).
//!
//! [`elementwise`] evaluates a function of one or more inputs of equal
//! length (the caller broadcasts them) and returns one or more outputs and,
//! on request, the partial derivative of every output with respect to every
//! input at every element. [`reduce`] evaluates a reduction over consecutive
//! rows of `n` values (the caller moves the reduced dimensions to the end)
//! and returns one value per row and, on request, the derivative with
//! respect to every element.
//!
//! The formulas are BoTorch's, operation by operation, so that values agree
//! to a few ULPs; the derivatives are those of PyTorch's autograd of the same
//! expressions (one branch per element for `torch.where`, the conventions of
//! PyTorch ≥ 2.14 at clamp bounds: a scalar bound gets no gradient at a tie,
//! a tensor bound half of it; see `docs/clamp-subgradients.md`).
//! `single_precision` selects the float32 constants of BoTorch's
//! dtype-dependent functions (`exp`, `log`, `log_softplus`); the arithmetic
//! is in double precision.

use crate::acquisition::dual::{Dual, Scalar};
use crate::error::{BotorchError, Result};
use crate::utils::mvnxpb::{bvn_generic, bvnmom_generic, bvnu};
use crate::utils::probability as prob;
use crate::utils::safe_math as sm;

/// `math.log(torch.finfo(torch.float64).max) - 1e-4` (BoTorch's `exp`).
const MAX_EXP_F64: f64 = 709.782_712_893_384 - 1e-4;
/// The same bound for float32, rounded to float32 (`get_constants_like`).
const MAX_EXP_F32: f64 = (88.722_839_052_068_35_f64 - 1e-4) as f32 as f64;
/// `torch.finfo(torch.float32).tiny`.
const TINY_F32: f64 = f32::MIN_POSITIVE as f64;
/// Weight of the Cauchy term of `fatplus`.
const FATPLUS_ALPHA: f64 = 0.1;
/// `math.sqrt(1 / 3)`, the inflection point of `fatmoid`.
const INV_SQRT_3: f64 = 0.5773502691896257;
/// `2 / sqrt(pi)`.
const TWO_INV_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// The result of [`elementwise`]: `values[o][i]` is output `o` at element
/// `i`; `grads[o][k][i]` its derivative with respect to input `k` (if
/// requested).
#[derive(Clone, Debug, Default)]
pub struct Elementwise {
    /// One vector per output.
    pub values: Vec<Vec<f64>>,
    /// Per output, per input, per element (if requested).
    pub grads: Option<Vec<Vec<Vec<f64>>>>,
}

/// The number of inputs and outputs of an element-wise function, or `None`
/// for an unknown name.
pub fn arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "ndtr" | "phi" | "log_phi" | "log_ndtr" | "log_erfc" | "log_erfcx"
        | "standard_normal_log_hazard" | "exp" | "log" | "log1mexp" | "log1pexp" | "logexpit"
        | "sigmoid" | "log_sigmoid" | "fat_sigmoid" | "log_fat_sigmoid" | "cauchy" => (1, 1),
        "add" | "sub" | "mul" | "div" | "logplusexp" | "logdiffexp" | "log_prob_normal_in"
        | "fatmaximum" | "fatminimum" | "fatplus" | "log_fatplus" | "log_softplus" | "fatmoid"
        | "log_fatmoid" | "approximate_round" => (2, 1),
        "bvnu" => (3, 1),
        "bvn" => (5, 1),
        "bvnmom" => (6, 2),
        _ => return None,
    })
}

/// BoTorch's `softplus(y)` (beta 1, threshold 20) and its derivative.
#[inline]
fn softplus_with_grad(y: f64) -> (f64, f64) {
    if y > 20.0 {
        (y, 1.0)
    } else {
        let z = y.exp();
        (z.ln_1p(), z / (z + 1.0))
    }
}

#[inline]
fn cauchy_with_grad(x: f64) -> (f64, f64) {
    let t = 1.0 + x * x;
    (1.0 / t, -(1.0 / (t * t)) * (2.0 * x))
}

/// `log(erfcx(x))` and its derivative (BoTorch's `log_erfcx`).
#[inline]
fn log_erfcx_with_grad(x: f64) -> (f64, f64) {
    if x > 0.0 {
        let e = prob::erfcx(x);
        (e.ln(), (2.0 * x * e - TWO_INV_SQRT_PI) / e)
    } else {
        let e = prob::erfc(x);
        (
            e.ln() + x * x,
            -TWO_INV_SQRT_PI * (-(x * x)).exp() / e + 2.0 * x,
        )
    }
}

/// BoTorch's `log_softplus(x, tau)` with the cutoffs of the dtype, and its
/// derivative with respect to `x`.
#[inline]
fn log_softplus_with_grad(x: f64, tau: f64, single: bool) -> (f64, f64) {
    let (upper, lower) = if single { (16.0, -15.0) } else { (32.0, -35.0) };
    if x / tau > lower {
        // torch's softplus(x, beta=1/tau, threshold=upper).
        let beta = 1.0 / tau;
        let xb = x * beta;
        if xb > upper {
            (x.ln(), 1.0 / x)
        } else {
            let z = xb.exp();
            let sp = z.ln_1p() / beta;
            (sp.ln(), (z / (z + 1.0)) / sp)
        }
    } else {
        (x / tau + tau.ln(), 1.0 / tau)
    }
}

/// `fatplus(x, tau)` and its derivative with respect to `x`.
#[inline]
fn fatplus_with_grad(x: f64, tau: f64) -> (f64, f64) {
    let y = x / tau;
    let (sp, spg) = softplus_with_grad(y);
    let (c, cg) = cauchy_with_grad(y);
    (
        tau * (sp + FATPLUS_ALPHA * c),
        tau * (spg + FATPLUS_ALPHA * cg) / tau,
    )
}

/// `fatmoid(x, tau)` and its derivative with respect to `x`.
#[inline]
fn fatmoid_with_grad(x: f64, tau: f64) -> (f64, f64) {
    let y = x / tau;
    if y < 0.0 {
        let (c, cg) = cauchy_with_grad(y - INV_SQRT_3);
        (2.0 / 3.0 * c, 2.0 / 3.0 * cg / tau)
    } else {
        let (c, cg) = cauchy_with_grad(y + INV_SQRT_3);
        (1.0 - 2.0 / 3.0 * c, -(2.0 / 3.0 * cg) / tau)
    }
}

/// One-input functions: `(value, derivative)`.
fn unary(name: &str, x: f64, single: bool) -> (f64, f64) {
    match name {
        "ndtr" => (prob::ndtr(x), prob::phi(x)),
        "phi" => {
            let p = prob::phi(x);
            (p, -x * p)
        }
        "log_phi" => (prob::log_phi(x), -x),
        "log_ndtr" => (prob::log_ndtr(x), prob::log_ndtr_grad(x)),
        "log_erfc" => (prob::log_erfc(x), prob::log_erfc_grad(x)),
        "log_erfcx" => log_erfcx_with_grad(x),
        "standard_normal_log_hazard" => {
            let (v, g) = log_erfcx_with_grad(prob::INV_SQRT_2 * x);
            (prob::LOG_TWO_INV_SQRT_2PI - v, -g * prob::INV_SQRT_2)
        }
        "exp" => {
            let bound = if single { MAX_EXP_F32 } else { MAX_EXP_F64 };
            if x > bound {
                (bound.exp(), 0.0)
            } else {
                let e = x.exp();
                // A tensor bound: half of the gradient at a tie.
                (e, if x == bound { 0.5 * e } else { e })
            }
        }
        "log" => {
            let tiny = if single { TINY_F32 } else { f64::MIN_POSITIVE };
            if x <= tiny {
                // A scalar bound: no gradient at or below it.
                (x.max(tiny).ln(), 0.0)
            } else {
                (x.ln(), 1.0 / x)
            }
        }
        "log1mexp" => (sm::log1mexp(x), sm::log1mexp_grad(x)),
        "log1pexp" => (sm::log1pexp(x), sm::log1pexp_grad(x)),
        "logexpit" | "log_sigmoid" => (sm::logexpit(x), sm::logexpit_grad(x)),
        "sigmoid" => {
            let e = sm::logexpit(x).exp();
            (e, e * sm::logexpit_grad(x))
        }
        "log_fat_sigmoid" => sm::log_fatmoid_with_grad(x, 1.0),
        "fat_sigmoid" => {
            let (v, g) = sm::log_fatmoid_with_grad(x, 1.0);
            let e = v.exp();
            (e, e * g)
        }
        "cauchy" => cauchy_with_grad(x),
        _ => unreachable!("unknown unary function {name}"),
    }
}

/// BoTorch's `_inf_max_helper` of `fatmax` / `logsumexp` for two values.
fn fatmax2(a: f64, b: f64, tau: f64, alpha: f64) -> (f64, f64, f64) {
    let (v, g) = sm::fatmax_with_grad(&[a, b], tau, alpha);
    (v, g[0], g[1])
}

/// Two-input functions: `(value, d/da, d/db)` (`b` is `tau` for the
/// smoothed functions, whose derivative with respect to `tau` is not
/// computed).
fn binary(name: &str, a: f64, b: f64, params: &[f64], single: bool) -> Result<(f64, f64, f64)> {
    Ok(match name {
        "add" => {
            if a.is_infinite() && b.is_infinite() && a != b {
                (0.0, 0.0, 0.0)
            } else {
                (a + b, 1.0, 1.0)
            }
        }
        "sub" => {
            if a.is_infinite() && b.is_infinite() && a == b {
                (0.0, 0.0, 0.0)
            } else {
                (a - b, 1.0, -1.0)
            }
        }
        "mul" => {
            if (a.is_infinite() && b == 0.0) || (b.is_infinite() && a == 0.0) {
                (0.0, 0.0, 0.0)
            } else {
                (a * b, b, a)
            }
        }
        "div" => {
            if (a == 0.0 && b == 0.0) || (a.is_infinite() && b.is_infinite()) {
                (if a != b { -1.0 } else { 1.0 }, 0.0, 0.0)
            } else {
                let q = a / b;
                (q, 1.0 / b, -q / b)
            }
        }
        "logplusexp" => {
            let (v, g) = sm::logsumexp_with_grad(&[a, b]);
            (v, g[0], g[1])
        }
        "logdiffexp" => {
            let is_inf = b == f64::NEG_INFINITY;
            let z = a - if is_inf { 0.0 } else { b };
            let g = sm::log1mexp_grad(z);
            (
                b + sm::log1mexp(z),
                g,
                if is_inf { 1.0 } else { 1.0 - g },
            )
        }
        "log_prob_normal_in" => prob::log_prob_normal_in_with_grad(a, b)?,
        "fatmaximum" => fatmax2(a, b, params[0], params[1]),
        "fatminimum" => {
            let (v, ga, gb) = fatmax2(-a, -b, params[0], params[1]);
            (-v, ga, gb)
        }
        "fatplus" => {
            let (v, g) = fatplus_with_grad(a, b);
            (v, g, 0.0)
        }
        "log_fatplus" => {
            let (v, g) = fatplus_with_grad(a, b);
            (v.ln(), g / v, 0.0)
        }
        "log_softplus" => {
            let (v, g) = log_softplus_with_grad(a, b, single);
            (v, g, 0.0)
        }
        "fatmoid" => {
            let (v, g) = fatmoid_with_grad(a, b);
            (v, g, 0.0)
        }
        "log_fatmoid" => {
            let (v, g) = fatmoid_with_grad(a, b);
            (v.ln(), g / v, 0.0)
        }
        "approximate_round" => {
            let offset = a.floor();
            let t = ((a - offset - 0.5) / b).tanh();
            (offset + (t + 1.0) / 2.0, (1.0 - t * t) / b / 2.0, 0.0)
        }
        _ => unreachable!("unknown binary function {name}"),
    })
}

/// Evaluates the element-wise function `name` of `inputs` (equal lengths).
///
/// `params` are the scalar parameters of `fatmaximum` / `fatminimum` (`tau`,
/// `alpha`). With `with_grad`, the derivatives of every output with respect
/// to every input are returned too.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for an unknown function, a wrong number
/// of inputs or parameters, inputs of different lengths, and for
/// `log_prob_normal_in` unless `a < b` everywhere.
pub fn elementwise(
    name: &str,
    inputs: &[&[f64]],
    params: &[f64],
    single_precision: bool,
    with_grad: bool,
) -> Result<Elementwise> {
    let (nin, nout) = arity(name)
        .ok_or_else(|| BotorchError::InvalidArgument(format!("unknown function {name:?}")))?;
    if inputs.len() != nin {
        return Err(BotorchError::InvalidArgument(format!(
            "{name} takes {nin} inputs, got {}",
            inputs.len()
        )));
    }
    let len = inputs[0].len();
    if inputs.iter().any(|x| x.len() != len) {
        return Err(BotorchError::InvalidArgument(
            "the inputs must have the same number of elements".into(),
        ));
    }
    if matches!(name, "fatmaximum" | "fatminimum") && params.len() != 2 {
        return Err(BotorchError::InvalidArgument(format!(
            "{name} takes the parameters (tau, alpha)"
        )));
    }
    if name == "log_prob_normal_in"
        && !inputs[0]
            .iter()
            .zip(inputs[1])
            .all(|(a, b)| a < b)
    {
        return Err(BotorchError::InvalidArgument(
            "Received input tensors a, b for which not all a < b.".into(),
        ));
    }
    let mut values = vec![vec![0.0; len]; nout];
    let mut grads = with_grad.then(|| vec![vec![vec![0.0; len]; nin]; nout]);
    for i in 0..len {
        match nin {
            1 => {
                let (v, g) = unary(name, inputs[0][i], single_precision);
                values[0][i] = v;
                if let Some(gr) = grads.as_mut() {
                    gr[0][0][i] = g;
                }
            }
            2 => {
                let (v, ga, gb) =
                    binary(name, inputs[0][i], inputs[1][i], params, single_precision)?;
                values[0][i] = v;
                if let Some(gr) = grads.as_mut() {
                    gr[0][0][i] = ga;
                    gr[0][1][i] = gb;
                }
            }
            _ => {
                let x: Vec<f64> = inputs.iter().map(|v| v[i]).collect();
                if let Some(gr) = grads.as_mut() {
                    let d: Vec<Dual<6>> = x
                        .iter()
                        .enumerate()
                        .map(|(k, &v)| Dual::variable(v, k, nin))
                        .collect();
                    let outs = bvn_family(name, &d);
                    for (o, out) in outs.iter().enumerate() {
                        values[o][i] = out.v;
                        for k in 0..nin {
                            gr[o][k][i] = out.g[k];
                        }
                    }
                } else {
                    let outs = bvn_family(name, &x);
                    for (o, out) in outs.iter().enumerate() {
                        values[o][i] = *out;
                    }
                }
            }
        }
    }
    Ok(Elementwise { values, grads })
}

/// `bvnu`, `bvn` and `bvnmom` (with an explicit probability `p`).
fn bvn_family<S: Scalar>(name: &str, x: &[S]) -> Vec<S> {
    match name {
        "bvnu" => vec![bvnu(x[0], x[1], x[2])],
        "bvn" => vec![bvn_generic(x[0], x[1], x[2], x[3], x[4])],
        "bvnmom" => {
            let (a, b) = bvnmom_generic(x[0], x[1], x[2], x[3], x[4], x[5]);
            vec![a, b]
        }
        _ => unreachable!("unknown function {name}"),
    }
}

/// The names of the reductions of [`reduce`] and their number of parameters.
pub fn reduction_params(name: &str) -> Option<usize> {
    Some(match name {
        "logsumexp" | "logmeanexp" => 0,
        "smooth_amax" | "smooth_amin" => 1,
        "fatmax" | "fatmin" => 2,
        _ => return None,
    })
}

/// Evaluates the reduction `name` over the consecutive rows of `n` values of
/// `x` (BoTorch's `logsumexp`, `logmeanexp`, `smooth_amax(tau)`,
/// `smooth_amin(tau)`, `fatmax(tau, alpha)`, `fatmin(tau, alpha)` over the
/// reduced dimensions, with BoTorch's treatment of infinities; empty rows
/// give `-inf`, `+inf` for the minima). Returns the values and, with
/// `with_grad`, the derivative with respect to every element of `x`.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for an unknown reduction, wrong
/// parameters, `n` not dividing the length of `x`, and for `fatmax` /
/// `fatmin` with a negative `tau` (BoTorch's "Argument `x` must be
/// non-negative.").
pub fn reduce(
    name: &str,
    x: &[f64],
    n: usize,
    params: &[f64],
    with_grad: bool,
) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    let np = reduction_params(name)
        .ok_or_else(|| BotorchError::InvalidArgument(format!("unknown reduction {name:?}")))?;
    if params.len() != np {
        return Err(BotorchError::InvalidArgument(format!(
            "{name} takes {np} parameters, got {}",
            params.len()
        )));
    }
    if (n == 0 && !x.is_empty()) || (n > 0 && !x.len().is_multiple_of(n)) {
        return Err(BotorchError::InvalidArgument(format!(
            "{} values are not rows of {n}",
            x.len()
        )));
    }
    if matches!(name, "fatmax" | "fatmin") && params[0] < 0.0 {
        // `_pareto(-y / tau)` with `y = x - max(x) <= 0`: negative for a
        // negative tau unless every finite element is a maximum.
        let neg = if n == 0 {
            false
        } else {
            x.chunks(n).any(|row| {
                let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                m.is_finite() && row.iter().any(|&v| v < m)
            })
        };
        if neg {
            return Err(BotorchError::InvalidArgument(
                "Argument `x` must be non-negative.".into(),
            ));
        }
    }
    if n == 0 {
        return Ok((Vec::new(), with_grad.then(Vec::new)));
    }
    let mut values = Vec::with_capacity(x.len() / n);
    let mut grads = with_grad.then(|| vec![0.0; x.len()]);
    for (r, row) in x.chunks(n).enumerate() {
        let (v, g): (f64, Option<Vec<f64>>) = match name {
            "logsumexp" => {
                if with_grad {
                    let (v, g) = sm::logsumexp_with_grad(row);
                    (v, Some(g))
                } else {
                    (sm::logsumexp(row), None)
                }
            }
            "logmeanexp" => {
                if with_grad {
                    let (v, g) = sm::logmeanexp_with_grad(row);
                    (v, Some(g))
                } else {
                    (sm::logmeanexp(row), None)
                }
            }
            "smooth_amax" | "smooth_amin" => {
                let sign = if name == "smooth_amin" { -1.0 } else { 1.0 };
                let tau = params[0];
                let y: Vec<f64> = row.iter().map(|v| sign * v).collect();
                let (v, g) = sm::smooth_amax_with_grad(&y, tau);
                (sign * v, with_grad.then_some(g))
            }
            _ => {
                let sign = if name == "fatmin" { -1.0 } else { 1.0 };
                let y: Vec<f64> = row.iter().map(|v| sign * v).collect();
                let (v, g) = sm::fatmax_with_grad(&y, params[0], params[1]);
                (sign * v, with_grad.then_some(g))
            }
        };
        values.push(v);
        if let (Some(out), Some(g)) = (grads.as_mut(), g) {
            out[r * n..(r + 1) * n].copy_from_slice(&g);
        }
    }
    Ok((values, grads))
}

/// The groups of outcome constraints of BoTorch's
/// `compute_log_prob_feas_from_bounds`: one-sided lower bounds, one-sided
/// upper bounds and two-sided bounds, each with the indices of the outcomes
/// they apply to.
#[derive(Clone, Debug, Default)]
pub struct OutcomeBounds {
    /// `(index, lower bound)` pairs.
    pub lower: Vec<(usize, f64)>,
    /// `(index, upper bound)` pairs.
    pub upper: Vec<(usize, f64)>,
    /// `(index, lower bound, upper bound)` triples.
    pub both: Vec<(usize, f64, f64)>,
}

/// BoTorch's `compute_log_prob_feas_from_bounds`: the log probability that
/// independent normal outcomes (`means`, `sigmas`: rows of `m` values)
/// satisfy the bounds, one value per row, with (on request) the derivatives
/// with respect to the means and the standard deviations.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for an index out of range, rows of
/// different lengths and two-sided bounds whose standardized lower bound is
/// not below the upper one (BoTorch's `log_prob_normal_in` error).
#[allow(clippy::type_complexity)]
pub fn log_prob_feas_from_bounds(
    bounds: &OutcomeBounds,
    means: &[f64],
    sigmas: &[f64],
    m: usize,
    with_grad: bool,
) -> Result<(Vec<f64>, Option<(Vec<f64>, Vec<f64>)>)> {
    if means.len() != sigmas.len() || m == 0 || !means.len().is_multiple_of(m) {
        return Err(BotorchError::InvalidArgument(
            "means and sigmas must have the same shape `(b) x m`".into(),
        ));
    }
    let max_index = bounds
        .lower
        .iter()
        .map(|t| t.0)
        .chain(bounds.upper.iter().map(|t| t.0))
        .chain(bounds.both.iter().map(|t| t.0))
        .max();
    if max_index.is_some_and(|i| i >= m) {
        return Err(BotorchError::InvalidArgument(format!(
            "constraint index out of range for {m} outcomes"
        )));
    }
    let rows = means.len() / m;
    let mut out = vec![0.0; rows];
    let mut grads = with_grad.then(|| (vec![0.0; means.len()], vec![0.0; means.len()]));
    for r in 0..rows {
        let mu = &means[r * m..(r + 1) * m];
        let sd = &sigmas[r * m..(r + 1) * m];
        let mut lp = 0.0;
        if !bounds.lower.is_empty() {
            let mut s = 0.0;
            for &(i, c) in &bounds.lower {
                let dl = (c - mu[i]) / sd[i];
                s += prob::log_ndtr(-dl);
                if let Some((gm, gs)) = grads.as_mut() {
                    let g = prob::log_ndtr_grad(-dl);
                    gm[r * m + i] += g / sd[i];
                    gs[r * m + i] += g * dl / sd[i];
                }
            }
            lp += s;
        }
        if !bounds.upper.is_empty() {
            let mut s = 0.0;
            for &(i, c) in &bounds.upper {
                let du = (c - mu[i]) / sd[i];
                s += prob::log_ndtr(du);
                if let Some((gm, gs)) = grads.as_mut() {
                    let g = prob::log_ndtr_grad(du);
                    gm[r * m + i] -= g / sd[i];
                    gs[r * m + i] -= g * du / sd[i];
                }
            }
            lp += s;
        }
        if !bounds.both.is_empty() {
            let mut s = 0.0;
            for &(i, lo, hi) in &bounds.both {
                let dl = (lo - mu[i]) / sd[i];
                let du = (hi - mu[i]) / sd[i];
                let (v, ga, gb) = prob::log_prob_normal_in_with_grad(dl, du).map_err(|_| {
                    BotorchError::InvalidArgument(
                        "Received input tensors a, b for which not all a < b.".into(),
                    )
                })?;
                s += v;
                if let Some((gm, gs)) = grads.as_mut() {
                    gm[r * m + i] -= (ga + gb) / sd[i];
                    gs[r * m + i] -= (ga * dl + gb * du) / sd[i];
                }
            }
            lp += s;
        }
        out[r] = lp;
    }
    Ok((out, grads))
}

/// BoTorch's `percentile_of_score` (SciPy's `percentileofscore` with
/// `kind="rank"`): for every row of `n` values of `data` and its score, the
/// percentile rank `(#below + #below_or_equal + [#below < #below_or_equal]) ·
/// 50 / n`, NaN if the score or a value of the row is NaN. With
/// `single_precision`, the product is computed in float32 like PyTorch's
/// integer-by-float multiplication with the default dtype.
pub fn percentile_of_score(data: &[f64], scores: &[f64], n: usize, single_precision: bool) -> Vec<f64> {
    if n == 0 {
        return vec![f64::NAN; scores.len()];
    }
    data.chunks(n)
        .zip(scores)
        .map(|(row, &score)| {
            if score.is_nan() || row.iter().any(|v| v.is_nan()) {
                return f64::NAN;
            }
            let left = row.iter().filter(|&&v| v < score).count();
            let right = row.iter().filter(|&&v| v <= score).count();
            let total = left + right + usize::from(left < right);
            if single_precision {
                f64::from(total as f32 * (50.0 / n as f64) as f32)
            } else {
                total as f64 * (50.0 / n as f64)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd1(name: &str, x: f64, h: f64) -> f64 {
        let up = elementwise(name, &[&[x + h]], &[], false, false).unwrap();
        let dn = elementwise(name, &[&[x - h]], &[], false, false).unwrap();
        (up.values[0][0] - dn.values[0][0]) / (2.0 * h)
    }

    #[test]
    fn unary_derivatives_match_finite_differences() {
        for name in [
            "ndtr",
            "phi",
            "log_phi",
            "log_ndtr",
            "log_erfc",
            "log_erfcx",
            "standard_normal_log_hazard",
            "exp",
            "log",
            "log1pexp",
            "logexpit",
            "sigmoid",
            "fat_sigmoid",
            "log_fat_sigmoid",
            "cauchy",
        ] {
            for &x in &[-7.0, -1.3, -0.2, 0.4, 2.5, 9.0] {
                if name == "log" && x <= 0.0 {
                    continue;
                }
                let e = elementwise(name, &[&[x]], &[], false, true).unwrap();
                let g = e.grads.unwrap()[0][0][0];
                let f = fd1(name, x, 1e-6);
                assert!(
                    (g - f).abs() <= 1e-6 * f.abs().max(1e-3),
                    "{name}'({x}): {g} vs {f}"
                );
            }
        }
        // log1mexp on x < 0.
        for &x in &[-5.0, -0.5, -1e-3] {
            let e = elementwise("log1mexp", &[&[x]], &[], false, true).unwrap();
            let f = fd1("log1mexp", x, 1e-8);
            let g = e.grads.unwrap()[0][0][0];
            assert!((g - f).abs() <= 1e-5 * f.abs(), "{g} vs {f}");
        }
    }

    #[test]
    fn safe_functions_follow_botorch() {
        let e = elementwise("exp", &[&[1e4, MAX_EXP_F64]], &[], false, true).unwrap();
        assert_eq!(e.values[0][0], MAX_EXP_F64.exp());
        assert_eq!(e.grads.as_ref().unwrap()[0][0][0], 0.0);
        assert_eq!(e.grads.unwrap()[0][0][1], 0.5 * MAX_EXP_F64.exp());
        let e = elementwise("exp", &[&[100.0]], &[], true, false).unwrap();
        assert_eq!(e.values[0][0], MAX_EXP_F32.exp());
        let e = elementwise("log", &[&[0.0, -1.0]], &[], false, true).unwrap();
        assert_eq!(e.values[0][0], f64::MIN_POSITIVE.ln());
        assert_eq!(e.grads.unwrap()[0][0], vec![0.0, 0.0]);
        let inf = f64::INFINITY;
        let a = [inf, inf, 0.0, 0.0, 2.0];
        let b = [-inf, inf, 0.0, inf, 4.0];
        let add = elementwise("add", &[&a, &b], &[], false, false).unwrap();
        assert_eq!(add.values[0][..2], [0.0, inf]);
        let sub = elementwise("sub", &[&a, &b], &[], false, false).unwrap();
        assert_eq!(sub.values[0][..2], [inf, 0.0]);
        let mul = elementwise("mul", &[&a, &b], &[], false, false).unwrap();
        assert_eq!(mul.values[0][2..], [0.0, 0.0, 8.0]);
        let div = elementwise("div", &[&a, &b], &[], false, false).unwrap();
        assert_eq!(div.values[0], vec![-1.0, 1.0, 1.0, 0.0, 0.5]);
        assert!(elementwise("log_prob_normal_in", &[&[1.0], &[0.0]], &[], false, false).is_err());
    }

    #[test]
    fn reductions() {
        let x = [1.0, 2.0, 3.0, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let (v, g) = reduce("logsumexp", &x, 3, &[], true).unwrap();
        assert!((v[0] - (1f64.exp() + 2f64.exp() + 3f64.exp()).ln()).abs() < 1e-15);
        assert_eq!(v[1], f64::NEG_INFINITY);
        assert_eq!(g.unwrap()[3..], [1.0, 1.0, 1.0]);
        let (v, _) = reduce("smooth_amin", &[1.0, 2.0], 2, &[1e-3], false).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-9);
        assert!(reduce("fatmax", &[1.0, 2.0], 2, &[-1.0, 2.0], false).is_err());
        let (v, _) = reduce("fatmin", &[1.0, 2.0], 2, &[1e-3, 2.0], false).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-5);
        let (v, _) = reduce("logsumexp", &[], 0, &[], false).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn bvn_family_gradients() {
        let r = [0.3, 0.95, -0.6];
        let xl = [-1.0, -0.5, -2.0];
        let yl = [-0.4, -1e7, 0.1];
        let xu = [0.7, 0.8, 1.5];
        let yu = [1.2, 0.3, f64::INFINITY];
        let e = elementwise("bvn", &[&r, &xl, &yl, &xu, &yu], &[], false, true).unwrap();
        let g = e.grads.unwrap();
        for i in 0..3 {
            let h = 1e-6;
            let mut up = [r[i], xl[i], yl[i], xu[i], yu[i]];
            let mut dn = up;
            up[3] += h;
            dn[3] -= h;
            let f = |v: [f64; 5]| crate::utils::mvnxpb::bvn(v[0], v[1], v[2], v[3], v[4]);
            let fdv = (f(up) - f(dn)) / (2.0 * h);
            assert!((g[0][3][i] - fdv).abs() < 1e-7, "{i}");
            assert_eq!(e.values[0][i], f([r[i], xl[i], yl[i], xu[i], yu[i]]));
        }
        let p = e.values[0].clone();
        let m = elementwise("bvnmom", &[&r, &xl, &yl, &xu, &yu, &p], &[], false, false).unwrap();
        assert_eq!(m.values.len(), 2);
    }
}
