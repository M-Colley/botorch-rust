//! Standard-normal utilities mirroring `botorch.utils.probability.utils`
//! (`ndtr`, `phi`, `log_phi`, `log_ndtr`, `log_erfc`, `log_erfcx`, ...), with
//! derivatives.
//!
//! The error functions come from the `errorfunctions` crate, whose `erfcx` is
//! the Faddeeva-package algorithm also used by PyTorch (`torch.special.erfcx`)
//! and whose `erfc` is accurate to a few ULPs. The constants are the exact
//! double-precision values of BoTorch's Python constants (e.g.
//! `1 / math.sqrt(2)`), so that the same floating-point operations are
//! performed. Derivatives (`*_grad`) follow the formulas of PyTorch's autograd
//! for the same expressions where that matters numerically (for example the
//! `erfcx` branch of `log_erfc`).

use errorfunctions::RealErrorFunctions;

use crate::error::{BotorchError, Result};
use crate::utils::safe_math::logdiffexp;

/// `math.log(2)`.
pub const LOG_2: f64 = std::f64::consts::LN_2;
/// `1 / math.sqrt(2 * math.pi)`.
pub const INV_SQRT_2PI: f64 = 0.3989422804014327;
/// `1 / math.sqrt(2)` (BoTorch's `_inv_sqrt_2`).
pub const INV_SQRT_2: f64 = 0.7071067811865475;
/// `-1 / math.sqrt(2)` (BoTorch's `_neg_inv_sqrt_2`).
pub const NEG_INV_SQRT_2: f64 = -0.7071067811865475;
/// `math.log(2 * math.pi) / 2`.
pub const LOG_SQRT_2PI: f64 = 0.9189385332046727;
/// `math.log(2) - math.log(2 * math.pi) / 2`, i.e. `log(2 / sqrt(2π))`.
pub const LOG_TWO_INV_SQRT_2PI: f64 = -0.22579135264472738;
/// `2 / sqrt(π)` as used in PyTorch's `erfc`/`erfcx` derivatives.
pub const TWO_INV_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// `exp(-x²)` without the rounding error of `x²`: `x = x_h + x_l` with `x_h`
/// holding the leading 26 bits, so that `x_h²` is exact and
/// `x² = x_h² + x_l (x + x_h)`.
#[inline]
fn exp_neg_square(x: f64) -> f64 {
    let xh = f64::from_bits(x.to_bits() & 0xffff_ffff_f800_0000);
    let xl = x - xh;
    (-(xh * xh)).exp() * (-(xl * (x + xh))).exp()
}

/// The complementary error function `erfc(x)`, accurate to a few ULPs
/// including the far right tail: `exp(-x²) erfcx(x)` for `x > 0` with
/// `exp(-x²)` evaluated without the rounding error of `x²` (whose relative
/// effect, `x² ε`, would otherwise reach 1e-13 near the underflow threshold),
/// and `2 - erfc(-x)` for `x < 0`.
#[inline]
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x >= 0.0 {
        if x > 27.3 {
            0.0
        } else {
            exp_neg_square(x) * erfcx(x)
        }
    } else if x < -6.0 {
        2.0
    } else {
        2.0 - exp_neg_square(x) * erfcx(-x)
    }
}

/// The scaled complementary error function `erfcx(x) = exp(x²) erfc(x)`
/// (`torch.special.erfcx`).
#[inline]
pub fn erfcx(x: f64) -> f64 {
    RealErrorFunctions::erfcx(x)
}

/// Derivative of [`erfcx`] (PyTorch's formula `2 x erfcx(x) - 2/√π`).
#[inline]
pub fn erfcx_grad(x: f64) -> f64 {
    2.0 * x * erfcx(x) - TWO_INV_SQRT_PI
}

/// Standard normal CDF `Φ(x) = erfc(-x/√2) / 2` (BoTorch's `ndtr`).
#[inline]
pub fn ndtr(x: f64) -> f64 {
    0.5 * erfc(NEG_INV_SQRT_2 * x)
}

/// Standard normal PDF `φ(x)` (BoTorch's `phi`).
#[inline]
pub fn phi(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * (x * x)).exp()
}

/// Logarithm of the standard normal PDF (BoTorch's `log_phi`).
#[inline]
pub fn log_phi(x: f64) -> f64 {
    -0.5 * (x * x) - LOG_SQRT_2PI
}

/// `log(erfc(x))`, accurate for large positive `x` (BoTorch's `log_erfc`):
/// `log(erfcx(x)) - x²` for `x > 0` and `log(erfc(x))` otherwise.
#[inline]
pub fn log_erfc(x: f64) -> f64 {
    if x > 0.0 {
        erfcx(x).ln() - x * x
    } else {
        erfc(x).ln()
    }
}

/// Derivative of [`log_erfc`], evaluated like PyTorch's autograd of the same
/// expression.
#[inline]
pub fn log_erfc_grad(x: f64) -> f64 {
    if x > 0.0 {
        let e = erfcx(x);
        (2.0 * x * e - TWO_INV_SQRT_PI) / e - 2.0 * x
    } else {
        -TWO_INV_SQRT_PI * (-(x * x)).exp() / erfc(x)
    }
}

/// `log(erfcx(x))`, accurate for all `x` (BoTorch's `log_erfcx`).
#[inline]
pub fn log_erfcx(x: f64) -> f64 {
    if x > 0.0 {
        erfcx(x).ln()
    } else {
        erfc(x).ln() + x * x
    }
}

/// `log Φ(x)`, accurate deep in the left tail (BoTorch's `log_ndtr`).
#[inline]
pub fn log_ndtr(x: f64) -> f64 {
    log_erfc(NEG_INV_SQRT_2 * x) - LOG_2
}

/// Derivative of [`log_ndtr`], `φ(x) / Φ(x)` (evaluated stably).
#[inline]
pub fn log_ndtr_grad(x: f64) -> f64 {
    log_erfc_grad(NEG_INV_SQRT_2 * x) * NEG_INV_SQRT_2
}

/// Logarithm of the standard normal hazard function `log(φ(x) / Φ(-x))`
/// (BoTorch's `standard_normal_log_hazard`).
#[inline]
pub fn standard_normal_log_hazard(x: f64) -> f64 {
    LOG_TWO_INV_SQRT_2PI - log_erfcx(INV_SQRT_2 * x)
}

/// `log(Φ(b) - Φ(a))` for `a < b`, accurate in both tails (BoTorch's
/// `log_prob_normal_in`).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] unless `a < b`.
pub fn log_prob_normal_in(a: f64, b: f64) -> Result<f64> {
    if a.is_nan() || b.is_nan() || a >= b {
        return Err(BotorchError::InvalidArgument(format!(
            "log_prob_normal_in requires a < b, got a = {a}, b = {b}"
        )));
    }
    // Use the left tail, where log_ndtr is well behaved.
    let (a, b) = if b.abs() > a.abs() { (-b, -a) } else { (a, b) };
    Ok(logdiffexp(log_ndtr(a), log_ndtr(b)))
}

/// [`log_prob_normal_in`] with its partial derivatives
/// `(value, ∂/∂a, ∂/∂b)`, evaluated like PyTorch's autograd of BoTorch's
/// expression (`logdiffexp(log_ndtr(a), log_ndtr(b))` after reflecting the
/// interval into the left tail).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] unless `a < b`.
pub fn log_prob_normal_in_with_grad(a: f64, b: f64) -> Result<(f64, f64, f64)> {
    if a.is_nan() || b.is_nan() || a >= b {
        return Err(BotorchError::InvalidArgument(format!(
            "log_prob_normal_in requires a < b, got a = {a}, b = {b}"
        )));
    }
    let reversed = b.abs() > a.abs();
    let (lo, hi) = if reversed { (-b, -a) } else { (a, b) };
    let (la, lb) = (log_ndtr(lo), log_ndtr(hi));
    let value = logdiffexp(la, lb);
    let shift = if lb == f64::NEG_INFINITY { 0.0 } else { lb };
    let g = crate::utils::safe_math::log1mexp_grad(la - shift);
    let d_lo = g * log_ndtr_grad(lo);
    let d_hi = (1.0 - g) * log_ndtr_grad(hi);
    if reversed {
        Ok((value, -d_hi, -d_lo))
    } else {
        Ok((value, d_lo, d_hi))
    }
}

/// Inverse of the standard normal CDF, `Φ⁻¹(p) = -√2 erfc⁻¹(2p)`, accurate in
/// both tails (statrs' `erfc_inv`). Returns `-inf` / `+inf` at `p = 0` / `1`
/// and NaN outside `[0, 1]`.
#[inline]
pub fn ndtri(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    -std::f64::consts::SQRT_2 * statrs::function::erf::erfc_inv(2.0 * p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    // Reference values computed with mpmath (50 digits).
    #[test]
    fn values_match_high_precision_references() {
        let log_ndtr_ref = [
            (-1e6, -500000000014.73445),
            (-40.0, -804.6084420137538),
            (-10.0, -53.23128515051247),
            (-1.0, -1.8410216450092635),
            (0.0, -std::f64::consts::LN_2),
            (1.5, -0.06914345561223398),
            (5.0, -2.866516129637636e-7),
            (10.0, -7.619853024160526e-24),
        ];
        // BoTorch's formula, log(erfc(-x/√2)) - log(2), cancels in the right
        // tail, so log-space values are compared in absolute terms there.
        for (x, v) in log_ndtr_ref {
            let err = (log_ndtr(x) - v).abs() / v.abs().max(1.0);
            assert!(err < 1e-14, "log_ndtr({x}): {err:e}");
        }
        assert!(rel(ndtr(-3.0), 0.0013498980316300946) < 1e-14);
        assert!(rel(ndtr(2.0), 0.9772498680518208) < 1e-15);
        assert!(rel(phi(1.3), 0.17136859204780736) < 1e-15);
        assert!(rel(erfcx(30.0), 0.018795888861416754) < 1e-15);
        // Far right tail of erfc (mpmath): the relative error stays at a few ULPs.
        assert!(rel(erfc(21.0), 8.032_453_871_022_456e-194) < 1e-15);
        assert!(rel(erfc(26.5), 2.2109076642637343e-307) < 2e-15);
        assert!(rel(erfc(5.0), 1.537_459_794_428_035e-12) < 1e-15);
        assert!(rel(erfc(-0.3), 1.3286267594591274) < 1e-15);
        assert_eq!(erfc(0.0), 1.0);
        assert_eq!(erfc(30.0), 0.0);
        assert_eq!(erfc(-7.0), 2.0);
        assert!(rel(ndtri(1e-20), -9.262340089798407) < 1e-14);
        assert!(rel(ndtri(0.3), -0.5244005127080408) < 1e-14);
        assert!(rel(ndtri(1e-10), -6.361340902404056) < 1e-14);
        assert_eq!(ndtri(0.5), 0.0);
        assert_eq!(ndtri(0.0), f64::NEG_INFINITY);
        assert!(ndtri(1.5).is_nan());
        // log hazard: log(phi(x) / Phi(-x)) at x = 2.
        assert!(
            rel(
                standard_normal_log_hazard(2.0),
                (phi(2.0) / ndtr(-2.0)).ln()
            ) < 1e-14
        );
        // log P(a < Z < b).
        let p = log_prob_normal_in(-1.0, 2.0).unwrap();
        assert!(rel(p, (ndtr(2.0) - ndtr(-1.0)).ln()) < 1e-14);
        let tail = log_prob_normal_in(30.0, 31.0).unwrap();
        assert!(rel(tail, -454.32124395634325) < 1e-13);
        assert!(log_prob_normal_in(1.0, 1.0).is_err());
    }

    #[test]
    fn derivatives_match_finite_differences() {
        for &x in &[-1e3, -35.0, -8.0, -1.0, -0.2, 0.0, 0.7, 3.0, 12.0] {
            let h = 1e-6 * f64::max(1.0, f64::abs(x));
            let fd = (log_ndtr(x + h) - log_ndtr(x - h)) / (2.0 * h);
            let g = log_ndtr_grad(x);
            assert!(
                (g - fd).abs() <= 1e-6 * g.abs().max(1e-10),
                "log_ndtr'({x}): {g} vs {fd}"
            );
            // phi / Phi, stably.
            if x > -30.0 {
                assert!(rel(g, phi(x) / ndtr(x)) < 1e-12, "{x}");
            }
        }
        for &x in &[-5.0f64, -0.3, 0.0, 0.4, 3.0, 49.0, 51.0, 1e3] {
            let h = 1e-6 * x.abs().max(1.0);
            let fd = (erfcx(x + h) - erfcx(x - h)) / (2.0 * h);
            assert!(
                (erfcx_grad(x) - fd).abs() <= 1e-6 * fd.abs().max(1e-10),
                "{x}"
            );
        }
        // log P(a < Z < b) and its partial derivatives, in both tails and
        // across the reflection.
        for &(a, b) in &[
            (-1.0, 2.0),
            (-3.0, 0.5),
            (0.2, 0.4),
            (-0.4, -0.2),
            (30.0, 31.0),
            (-12.0, -11.5),
            (-50.0, 40.0),
        ] {
            let (v, ga, gb) = log_prob_normal_in_with_grad(a, b).unwrap();
            assert_eq!(v, log_prob_normal_in(a, b).unwrap());
            let h = 1e-6;
            let fa = (log_prob_normal_in(a + h, b).unwrap()
                - log_prob_normal_in(a - h, b).unwrap())
                / (2.0 * h);
            let fb = (log_prob_normal_in(a, b + h).unwrap()
                - log_prob_normal_in(a, b - h).unwrap())
                / (2.0 * h);
            assert!(
                (ga - fa).abs() <= 1e-6 * fa.abs().max(1.0),
                "({a}, {b}): {ga} vs {fa}"
            );
            assert!(
                (gb - fb).abs() <= 1e-6 * fb.abs().max(1.0),
                "({a}, {b}): {gb} vs {fb}"
            );
        }
        assert!(log_prob_normal_in_with_grad(1.0, 1.0).is_err());
    }
}
