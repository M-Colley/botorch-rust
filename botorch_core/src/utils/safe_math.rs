//! Numerically careful special functions mirroring `botorch.utils.safe_math`:
//! `log1mexp`, `log1pexp`, `logsumexp` / `logmeanexp` (with BoTorch's
//! treatment of infinities), the smooth and fat-tailed approximations to
//! `ReLU` and `max` used by the LogEI acquisition functions (`log_softplus`,
//! `fatplus`, `log_fatplus`, `fatmax`, `smooth_amax`), and sigmoids.
//!
//! Reductions take a slice and return the value; the `*_with_grad` variants
//! also return the gradient with respect to every element, computed with the
//! same conventions as PyTorch's autograd (e.g. `amax` distributes its
//! gradient evenly between tied maxima).
//!
//! References: M. Mächler, *Accurately Computing log(1 - exp(-|a|))* (2012);
//! S. Ament et al., *Unexpected Improvements to Expected Improvement for
//! Bayesian Optimization*, NeurIPS 2023.

use crate::utils::probability::LOG_2;

/// Default temperature of the smooth approximations (`safe_math.TAU`).
pub const TAU: f64 = 1.0;
/// Default exponent of the power-law decay of [`fatmax`] (`safe_math.ALPHA`).
pub const ALPHA: f64 = 2.0;
/// Weight of the Cauchy term in [`fatplus`].
const FATPLUS_ALPHA: f64 = 0.1;
/// `math.sqrt(1 / 3)`, the inflection point of [`fatmoid`].
const INV_SQRT_3: f64 = 0.5773502691896257;
/// Below this argument `exp` rounds to zero (`exp(x) < 2^-1075`).
const EXP_UNDERFLOW: f64 = -746.0;

/// `torch.nn.functional.softplus(x)` with `beta = 1`, `threshold = 20`.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < EXP_UNDERFLOW {
        // exp(x) rounds to 0, so log1p(exp(x)) = 0 exactly.
        0.0
    } else {
        x.exp().ln_1p()
    }
}

/// `log(1 - exp(x))` for `x < 0`, accurate in both regimes (Mächler 2012).
#[inline]
pub fn log1mexp(x: f64) -> f64 {
    if -LOG_2 < x {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

/// Derivative of [`log1mexp`], `-exp(x) / (1 - exp(x))`, evaluated per branch
/// like PyTorch's autograd.
#[inline]
pub fn log1mexp_grad(x: f64) -> f64 {
    if -LOG_2 < x {
        let em1 = x.exp_m1();
        -(1.0 / -em1) * (em1 + 1.0)
    } else {
        let ex = x.exp();
        -(1.0 / (-ex + 1.0)) * ex
    }
}

/// `log(1 + exp(x))`, accurate for all `x`.
#[inline]
pub fn log1pexp(x: f64) -> f64 {
    if x <= 18.0 {
        x.exp().ln_1p()
    } else {
        x + (-x).exp()
    }
}

/// Derivative of [`log1pexp`] (the logistic sigmoid).
#[inline]
pub fn log1pexp_grad(x: f64) -> f64 {
    if x <= 18.0 {
        let ex = x.exp();
        ex / (ex + 1.0)
    } else {
        1.0 - (-x).exp()
    }
}

/// Logarithm of the logistic sigmoid, `-log1pexp(-x)`.
#[inline]
pub fn logexpit(x: f64) -> f64 {
    -log1pexp(-x)
}

/// Derivative of [`logexpit`], `σ(-x)`, evaluated per branch of
/// [`log1pexp`] like PyTorch's autograd.
#[inline]
pub fn logexpit_grad(x: f64) -> f64 {
    log1pexp_grad(-x)
}

/// The logistic sigmoid `1 / (1 + exp(-x))` (`torch.sigmoid`).
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of [`sigmoid`], `σ(x)(1 - σ(x))`.
#[inline]
pub fn sigmoid_grad(x: f64) -> f64 {
    let s = sigmoid(x);
    (1.0 - s) * s
}

/// The Lorentzian `1 / (1 + x²)` (an un-normalized Cauchy density).
#[inline]
pub fn cauchy(x: f64) -> f64 {
    1.0 / (1.0 + x * x)
}

/// Derivative of [`cauchy`].
#[inline]
fn cauchy_grad(x: f64) -> f64 {
    let t = 1.0 + x * x;
    -(1.0 / (t * t)) * (2.0 * x)
}

/// The rational function of [`fatmax`]: decreasing on `x ≥ 0`, equal to 1 at
/// 0 with first and second derivatives of magnitude 1 there, decaying as
/// `x^{-alpha}` (BoTorch's `_pareto`).
#[inline]
pub fn pareto(x: f64, alpha: f64) -> f64 {
    let a = alpha / 2.0;
    let beta_1 = 2.0 * a;
    let beta_0 = a * beta_1;
    let base = beta_0 / (beta_0 + beta_1 * x + x * x);
    // `pow(x, 1)` is `x` (torch copies the base for an exponent of 1).
    if a == 1.0 {
        base
    } else {
        base.powf(a)
    }
}

/// Derivative of [`pareto`] with respect to `x`.
#[inline]
fn pareto_grad(x: f64, alpha: f64) -> f64 {
    if x == f64::INFINITY {
        // The limit (an element at -inf below a finite maximum); autograd
        // gives NaN here (inf / inf).
        return 0.0;
    }
    let a = alpha / 2.0;
    let beta_1 = 2.0 * a;
    let beta_0 = a * beta_1;
    let den = beta_0 + beta_1 * x + x * x;
    let base = beta_0 / den;
    let pow = if a == 1.0 { 1.0 } else { base.powf(a - 1.0) };
    a * pow * (-beta_0 / (den * den)) * (beta_1 + 2.0 * x)
}

/// Fat-tailed smooth approximation of `ReLU(x) = max(x, 0)`:
/// `tau · (softplus(x/tau) + 0.1 · cauchy(x/tau))` (BoTorch's `fatplus`).
#[inline]
pub fn fatplus(x: f64, tau: f64) -> f64 {
    let y = x / tau;
    tau * (softplus(y) + FATPLUS_ALPHA * cauchy(y))
}

/// `log(fatplus(x, tau))` (BoTorch's `log_fatplus`).
#[inline]
pub fn log_fatplus(x: f64, tau: f64) -> f64 {
    fatplus(x, tau).ln()
}

/// [`log_fatplus`] and its derivative with respect to `x`.
#[inline]
pub fn log_fatplus_with_grad(x: f64, tau: f64) -> (f64, f64) {
    let y = x / tau;
    // softplus (as in `softplus`) and its derivative (PyTorch's backward
    // formula `z / (z + 1)`) share `z = exp(y)`.
    let (sp, spg) = if y > 20.0 {
        (y, 1.0)
    } else if y < EXP_UNDERFLOW {
        (0.0, 0.0)
    } else {
        let z = y.exp();
        (z.ln_1p(), z / (z + 1.0))
    };
    let fp = tau * (sp + FATPLUS_ALPHA * cauchy(y));
    let inner = spg + FATPLUS_ALPHA * cauchy_grad(y);
    (fp.ln(), (tau / fp) * inner / tau)
}

/// `log(softplus(x, beta = 1/tau))`, accurate for very negative `x/tau`
/// (BoTorch's `log_softplus` in double precision).
#[inline]
pub fn log_softplus(x: f64, tau: f64) -> f64 {
    log_softplus_with_grad(x, tau).0
}

/// [`log_softplus`] and its derivative with respect to `x`.
#[inline]
pub fn log_softplus_with_grad(x: f64, tau: f64) -> (f64, f64) {
    const UPPER: f64 = 32.0;
    const LOWER: f64 = -35.0;
    let y = x / tau;
    if y > LOWER {
        if y > UPPER {
            // torch's softplus returns x itself above the threshold.
            (x.ln(), 1.0 / x)
        } else {
            let z = y.exp();
            let sp = tau * z.ln_1p();
            (sp.ln(), (z / (z + 1.0)) / sp)
        }
    } else {
        (y + tau.ln(), 1.0 / tau)
    }
}

/// The maximum of `x` (`-inf` for an empty slice).
#[inline]
fn max_of(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Gradient of `y_inf.sum()` in BoTorch's `_inf_max_helper`: ones on the
/// elements equal to an infinite maximum.
fn inf_max_grad(x: &[f64], m: f64) -> Vec<f64> {
    x.iter().map(|&v| if v == m { 1.0 } else { 0.0 }).collect()
}

/// Sum of the elements equal to an infinite maximum (BoTorch's value in that
/// case: `+inf`, or `-inf` if every element is `-inf`).
fn inf_max_value(x: &[f64], m: f64) -> f64 {
    x.iter().filter(|&&v| v == m).sum()
}

/// Number of elements tied at the maximum (`amax` splits its gradient evenly).
fn tie_count(x: &[f64], m: f64) -> f64 {
    x.iter().filter(|&&v| v == m).count() as f64
}

/// `log(Σ exp(x_i))` with BoTorch's handling of infinities: `+inf` if any
/// element is `+inf`, `-inf` if all are `-inf` (or `x` is empty).
pub fn logsumexp(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = max_of(x);
    if m.is_infinite() {
        return inf_max_value(x, m);
    }
    m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

/// [`logsumexp`] and its gradient (the softmax of `x`; for infinite maxima
/// BoTorch's convention of a unit gradient on the infinite elements).
pub fn logsumexp_with_grad(x: &[f64]) -> (f64, Vec<f64>) {
    if x.is_empty() {
        return (f64::NEG_INFINITY, Vec::new());
    }
    let m = max_of(x);
    if m.is_infinite() {
        return (inf_max_value(x, m), inf_max_grad(x, m));
    }
    let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    let value = m + s.ln();
    (value, e.iter().map(|v| v / s).collect())
}

/// `log(mean(exp(x)))` (BoTorch's `logmeanexp`).
pub fn logmeanexp(x: &[f64]) -> f64 {
    logsumexp(x) - (x.len() as f64).ln()
}

/// [`logmeanexp`] and its gradient.
pub fn logmeanexp_with_grad(x: &[f64]) -> (f64, Vec<f64>) {
    let (v, g) = logsumexp_with_grad(x);
    (v - (x.len() as f64).ln(), g)
}

/// `log(exp(a) + exp(b))`.
pub fn logplusexp(a: f64, b: f64) -> f64 {
    logsumexp(&[a, b])
}

/// `log(exp(log_b) - exp(log_a))` for `log_b > log_a` (BoTorch's
/// `logdiffexp`).
pub fn logdiffexp(log_a: f64, log_b: f64) -> f64 {
    let shift = if log_b == f64::NEG_INFINITY {
        0.0
    } else {
        log_b
    };
    log_b + log1mexp(log_a - shift)
}

/// Smooth approximation of `max(x)`: `tau · logsumexp(x / tau)` (BoTorch's
/// `smooth_amax`).
pub fn smooth_amax(x: &[f64], tau: f64) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = x.iter().map(|v| v / tau).fold(f64::NEG_INFINITY, f64::max);
    if m.is_infinite() {
        return x.iter().map(|v| v / tau).filter(|&v| v == m).sum::<f64>() * tau;
    }
    (m + x.iter().map(|v| (v / tau - m).exp()).sum::<f64>().ln()) * tau
}

/// [`smooth_amax`] and its gradient (the softmax of `x / tau`).
pub fn smooth_amax_with_grad(x: &[f64], tau: f64) -> (f64, Vec<f64>) {
    let scaled: Vec<f64> = x.iter().map(|v| v / tau).collect();
    let (v, g) = logsumexp_with_grad(&scaled);
    (v * tau, g)
}

/// Smooth approximation of `min(x)`: `-smooth_amax(-x)`.
pub fn smooth_amin(x: &[f64], tau: f64) -> f64 {
    let neg: Vec<f64> = x.iter().map(|v| -v).collect();
    -smooth_amax(&neg, tau)
}

/// Fat-tailed smooth approximation of `max(x)` (BoTorch's `fatmax`):
/// `M + tau · log Σ_i pareto((M - x_i) / tau, alpha)` with `M = max(x)`.
pub fn fatmax(x: &[f64], tau: f64, alpha: f64) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = max_of(x);
    if m.is_infinite() {
        return inf_max_value(x, m);
    }
    let p: f64 = x.iter().map(|v| pareto(-(v - m) / tau, alpha)).sum();
    m + tau * p.ln()
}

/// [`fatmax`] and its gradient, including the dependence on `M = max(x)`
/// (which autograd propagates through `amax`, split evenly between ties).
pub fn fatmax_with_grad(x: &[f64], tau: f64, alpha: f64) -> (f64, Vec<f64>) {
    let mut grad = vec![0.0; x.len()];
    let value = fatmax_grad_into(x, tau, alpha, &mut grad);
    (value, grad)
}

/// [`fatmax_with_grad`] writing the gradient into `grad` (no allocation).
pub(crate) fn fatmax_grad_into(x: &[f64], tau: f64, alpha: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(x.len(), grad.len());
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = max_of(x);
    if m.is_infinite() {
        for (g, &v) in grad.iter_mut().zip(x) {
            *g = if v == m { 1.0 } else { 0.0 };
        }
        return inf_max_value(x, m);
    }
    // First pass: P = Σ pareto(z_i) and p'(z_i) (stored in `grad`).
    let mut p = 0.0;
    for (g, &v) in grad.iter_mut().zip(x) {
        let zi = -(v - m) / tau;
        p += pareto(zi, alpha);
        *g = pareto_grad(zi, alpha);
    }
    let value = m + tau * p.ln();
    // d/dx_i through y_i = x_i - M: -p'(z_i) / P; d/dM: 1 + Σ_j p'(z_j) / P.
    let dm = 1.0 + grad.iter().sum::<f64>() / p;
    let ties = tie_count(x, m);
    for (g, &v) in grad.iter_mut().zip(x) {
        let mut gi = -*g / p;
        if v == m {
            gi += dm / ties;
        }
        *g = gi;
    }
    value
}

/// Fat-tailed smooth approximation of `min(x)`: `-fatmax(-x)`.
pub fn fatmin(x: &[f64], tau: f64, alpha: f64) -> f64 {
    let neg: Vec<f64> = x.iter().map(|v| -v).collect();
    -fatmax(&neg, tau, alpha)
}

/// Twice continuously differentiable, fat-tailed approximation of the
/// Heaviside step function (BoTorch's `fatmoid`).
pub fn fatmoid(x: f64, tau: f64) -> f64 {
    let x = x / tau;
    if x < 0.0 {
        2.0 / 3.0 * cauchy(x - INV_SQRT_3)
    } else {
        1.0 - 2.0 / 3.0 * cauchy(x + INV_SQRT_3)
    }
}

/// `log(fatmoid(x, tau))`.
pub fn log_fatmoid(x: f64, tau: f64) -> f64 {
    fatmoid(x, tau).ln()
}

/// [`log_fatmoid`] and its derivative with respect to `x`.
pub fn log_fatmoid_with_grad(x: f64, tau: f64) -> (f64, f64) {
    let y = x / tau;
    let (f, df) = if y < 0.0 {
        let z = y - INV_SQRT_3;
        (2.0 / 3.0 * cauchy(z), 2.0 / 3.0 * cauchy_grad(z))
    } else {
        let z = y + INV_SQRT_3;
        (1.0 - 2.0 / 3.0 * cauchy(z), -(2.0 / 3.0 * cauchy_grad(z)))
    };
    (f.ln(), df / f / tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central difference with a step relative to `scale`.
    fn fd<F: Fn(f64) -> f64>(f: F, x: f64, scale: f64) -> f64 {
        let h = 1e-6 * scale;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    fn close(a: f64, b: f64, rtol: f64) -> bool {
        (a - b).abs() <= rtol * a.abs().max(b.abs()).max(1e-12)
    }

    #[test]
    fn log1mexp_and_log1pexp() {
        for &x in &[-1e-12f64, -1e-3, -0.5, -0.7, -5.0, -40.0] {
            // exp(log1mexp(x)) + exp(x) = 1.
            assert!((log1mexp(x).exp() + x.exp() - 1.0).abs() < 1e-15, "{x}");
            assert!(
                close(log1mexp_grad(x), fd(log1mexp, x, x.abs()), 1e-6),
                "{x}"
            );
        }
        assert!(close(log1mexp(-1e-12), (1e-12f64).ln(), 1e-9));
        for &x in &[-50.0f64, -3.0, 0.0, 2.0, 17.9, 18.1, 60.0] {
            assert!(close(log1pexp(x), x.exp().ln_1p().max(x), 1e-14));
            assert!(close(log1pexp_grad(x), fd(log1pexp, x, 1.0), 1e-6));
            if x <= 2.0 {
                assert!(close(logexpit(x), sigmoid(x).ln(), 1e-12));
            } else {
                // log σ(x) = -log(1 + e^-x) ≈ -e^-x for large x.
                assert!(close(logexpit(x), -(-x).exp(), 1e-6));
            }
        }
        assert!(close(sigmoid_grad(0.3), fd(sigmoid, 0.3, 1.0), 1e-8));
    }

    #[test]
    fn logsumexp_handles_infinities() {
        let (v, g) = logsumexp_with_grad(&[1.0, 2.0, 3.0]);
        let e: f64 = [1.0f64, 2.0, 3.0].iter().map(|v| v.exp()).sum();
        assert!(close(v, e.ln(), 1e-15));
        assert!(close(g.iter().sum::<f64>(), 1.0, 1e-15));
        assert_eq!(logsumexp(&[f64::INFINITY, 1.0]), f64::INFINITY);
        let (v, g) = logsumexp_with_grad(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
        assert_eq!(v, f64::NEG_INFINITY);
        assert_eq!(g, vec![1.0, 1.0]);
        assert_eq!(logsumexp(&[]), f64::NEG_INFINITY);
        assert!(close(logmeanexp(&[0.5, 0.5]), 0.5, 1e-15));
        assert!(close(logplusexp(0.0, 0.0), LOG_2, 1e-15));
        assert!(close(logdiffexp(0.0, 1.0), (1f64.exp() - 1.0).ln(), 1e-15));
        assert!(close(smooth_amin(&[1.0, 2.0], 1e-3), 1.0, 1e-9));
        assert!(close(fatmin(&[1.0, 2.0], 1e-3, ALPHA), 1.0, 1e-5));
    }

    #[test]
    fn fat_functions_and_gradients() {
        for &tau in &[1e-6, 1e-2, 1.0] {
            for &x in &[-3.0, -1e-4, 0.0, 2e-6, 0.4, 25.0] {
                let (v, g) = log_fatplus_with_grad(x, tau);
                assert_eq!(v, log_fatplus(x, tau));
                let scale = x.abs().max(tau);
                assert!(
                    close(g, fd(|t| log_fatplus(t, tau), x, scale), 1e-5),
                    "{x} {tau}"
                );
                let (v, g) = log_softplus_with_grad(x, tau);
                assert_eq!(v, log_softplus(x, tau));
                assert!(
                    close(g, fd(|t| log_softplus(t, tau), x, scale), 1e-5),
                    "{x} {tau}"
                );
            }
        }
        // The underflow shortcut of softplus is exact.
        assert_eq!((-746.0f64).exp(), 0.0);
        assert_eq!(log_fatplus(-1.0, 1e-6), (1e-6 * (0.1 * cauchy(-1e6))).ln());
        // fatplus approaches ReLU for small tau.
        assert!((fatplus(0.3, 1e-6) - 0.3).abs() < 1e-12);
        assert!(fatplus(-0.3, 1e-6) > 0.0);
        // pareto: value 1 and slope -1 at 0.
        assert_eq!(pareto(0.0, ALPHA), 1.0);
        assert!(close(pareto_grad(0.0, ALPHA), -1.0, 1e-15));
        assert!(close(
            pareto_grad(0.7, 3.0),
            fd(|t| pareto(t, 3.0), 0.7, 1.0),
            1e-7
        ));
        // fatmax: close to max for small tau, gradient sums to one.
        let x = [0.2, -0.1, 0.19, 0.2];
        for &tau in &[1e-2, 0.3] {
            let (v, g) = fatmax_with_grad(&x, tau, ALPHA);
            assert!(v >= 0.2);
            assert!(close(g.iter().sum::<f64>(), 1.0, 1e-12));
            for i in [1usize, 2] {
                let h = 1e-7;
                let mut xp = x;
                xp[i] += h;
                let mut xm = x;
                xm[i] -= h;
                let fdi = (fatmax(&xp, tau, ALPHA) - fatmax(&xm, tau, ALPHA)) / (2.0 * h);
                assert!(close(g[i], fdi, 1e-5), "fatmax grad {i}: {} vs {fdi}", g[i]);
            }
            let (_, gs) = smooth_amax_with_grad(&x, tau);
            assert!(close(gs.iter().sum::<f64>(), 1.0, 1e-12));
        }
        assert!(close(fatmoid(0.0, 1.0), 0.5, 1e-15));
        assert!(fatmoid(-1e3, 1.0) > 0.0 && fatmoid(1e3, 1.0) < 1.0);
        assert!(close(log_fatmoid(2.0, 1.0), fatmoid(2.0, 1.0).ln(), 1e-15));
        for &x in &[-40.0, -2.0, -1e-3, 0.0, 0.7, 30.0] {
            for &tau in &[1.0, 1e-3] {
                let (v, g) = log_fatmoid_with_grad(x * tau, tau);
                assert_eq!(v, log_fatmoid(x * tau, tau));
                let fdg = fd(|t| log_fatmoid(t, tau), x * tau, tau);
                assert!(close(g, fdg, 1e-6), "log_fatmoid' at {x}: {g} vs {fdg}");
            }
            assert!(close(logexpit_grad(x), fd(logexpit, x, 1.0), 1e-6), "{x}");
        }
    }
}
