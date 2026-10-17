//! Forward-mode dual numbers for the information-gain formulas of the
//! max-value entropy search acquisition functions
//! ([`super::max_value_entropy_search`]).
//!
//! The information gain of a candidate is a scalar function of a handful of
//! posterior quantities (means, variances and covariances at the candidate,
//! its trace observations and the target fidelity). Evaluating the formula
//! generically over [`Scalar`] gives the value (`f64`) or, with
//! [`Dual`] numbers seeded on those quantities, its exact gradient with
//! respect to them in the same pass; the gradient is then propagated to the
//! inputs by the model's reverse-mode derivatives.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::acquisition::multi_objective::joint_entropy_search::{normal_cdf, normal_cdf_grad};

/// A dual number `v + Σ_i g_i ε_i` with up to `N` tangents (`n` in use).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Dual<const N: usize> {
    pub v: f64,
    pub g: [f64; N],
    pub n: usize,
}

impl<const N: usize> Dual<N> {
    /// A constant with `n` tangents.
    pub fn constant(v: f64, n: usize) -> Self {
        Self { v, g: [0.0; N], n }
    }

    /// The `i`-th of `n` independent variables with value `v`.
    pub fn variable(v: f64, i: usize, n: usize) -> Self {
        let mut d = Self::constant(v, n);
        d.g[i] = 1.0;
        d
    }

    /// The tangents.
    pub fn grad(&self) -> &[f64] {
        &self.g[..self.n]
    }

    #[inline]
    fn chain(&self, v: f64, dv: f64) -> Self {
        let mut out = Self::constant(v, self.n);
        for i in 0..self.n {
            out.g[i] = dv * self.g[i];
        }
        out
    }
}

/// The scalar operations of the information-gain formulas.
pub(crate) trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// A constant like `self` (same number of tangents).
    fn lift(&self, v: f64) -> Self;
    /// The value.
    fn value(&self) -> f64;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    /// `torch.distributions.Normal(0, 1).cdf`.
    fn ncdf(self) -> Self;
    /// `x.clamp_min(lo)` with a scalar bound: zero gradient where clamped,
    /// including at `x == lo` (torch ≥ 2.14: a scalar bound is a fixed
    /// parameter, so the minimum-norm subgradient `0` applies at equality).
    fn clamp_min(self, lo: f64) -> Self;
    /// `x.clamp(lo)` with a tensor lower bound: the gradient of the active
    /// argument, split evenly between `x` and `lo` at a tie (torch ≥ 2.14).
    fn clamp_min_var(self, lo: Self) -> Self;
    /// The value of `self` with the gradient `(∂self + ∂other) / 2`: torch
    /// ≥ 2.14's even split between an input and a tensor bound (or the two
    /// arguments of `torch.maximum` / `torch.minimum`) that tie.
    fn tie(self, other: Self) -> Self;
    /// BoTorch's `ndtr` (`erfc(-x / √2) / 2`, accurate in the left tail).
    fn ndtr(self) -> Self;
    fn sin(self) -> Self;
    fn asin(self) -> Self;
    /// `|x|` (gradient `sign(x)`, zero at zero, as autograd).
    fn abs(self) -> Self;
}

impl Scalar for f64 {
    fn lift(&self, v: f64) -> Self {
        v
    }
    fn value(&self) -> f64 {
        *self
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn ncdf(self) -> Self {
        normal_cdf(self)
    }
    fn clamp_min(self, lo: f64) -> Self {
        if self < lo {
            lo
        } else {
            self
        }
    }
    fn clamp_min_var(self, lo: Self) -> Self {
        if self < lo {
            lo
        } else {
            self
        }
    }
    fn tie(self, _other: Self) -> Self {
        self
    }
    fn ndtr(self) -> Self {
        crate::utils::probability::ndtr(self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn asin(self) -> Self {
        f64::asin(self)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

impl<const N: usize> Scalar for Dual<N> {
    fn lift(&self, v: f64) -> Self {
        Self::constant(v, self.n)
    }
    fn value(&self) -> f64 {
        self.v
    }
    fn ln(self) -> Self {
        self.chain(self.v.ln(), 1.0 / self.v)
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e)
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s)
    }
    fn ncdf(self) -> Self {
        self.chain(normal_cdf(self.v), normal_cdf_grad(self.v))
    }
    fn clamp_min(self, lo: f64) -> Self {
        if self.v < lo {
            Self::constant(lo, self.n)
        } else if self.v == lo {
            // Torch ≥ 2.14: zero gradient at equality (the value is kept, so
            // that a signed zero survives).
            Self::constant(self.v, self.n)
        } else {
            self
        }
    }
    fn clamp_min_var(self, lo: Self) -> Self {
        if self.v < lo.v {
            lo
        } else if self.v == lo.v {
            self.tie(lo)
        } else {
            self
        }
    }
    fn tie(mut self, other: Self) -> Self {
        for i in 0..self.n {
            self.g[i] = 0.5 * (self.g[i] + other.g[i]);
        }
        self
    }
    fn ndtr(self) -> Self {
        self.chain(
            crate::utils::probability::ndtr(self.v),
            crate::utils::probability::phi(self.v),
        )
    }
    fn sin(self) -> Self {
        self.chain(self.v.sin(), self.v.cos())
    }
    fn asin(self) -> Self {
        self.chain(self.v.asin(), 1.0 / (1.0 - self.v * self.v).sqrt())
    }
    fn abs(self) -> Self {
        let s = if self.v > 0.0 {
            1.0
        } else if self.v < 0.0 {
            -1.0
        } else {
            0.0
        };
        self.chain(self.v.abs(), s)
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Dual<N>;
    #[inline]
    fn add(mut self, o: Dual<N>) -> Dual<N> {
        self.v += o.v;
        for i in 0..self.n {
            self.g[i] += o.g[i];
        }
        self
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Dual<N>;
    #[inline]
    fn sub(mut self, o: Dual<N>) -> Dual<N> {
        self.v -= o.v;
        for i in 0..self.n {
            self.g[i] -= o.g[i];
        }
        self
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Dual<N>;
    #[inline]
    fn mul(self, o: Dual<N>) -> Dual<N> {
        let mut out = Self::constant(self.v * o.v, self.n);
        for i in 0..self.n {
            out.g[i] = self.g[i] * o.v + self.v * o.g[i];
        }
        out
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Dual<N>;
    #[inline]
    fn div(self, o: Dual<N>) -> Dual<N> {
        let q = self.v / o.v;
        let mut out = Self::constant(q, self.n);
        for i in 0..self.n {
            out.g[i] = (self.g[i] - q * o.g[i]) / o.v;
        }
        out
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Dual<N>;
    #[inline]
    fn neg(mut self) -> Dual<N> {
        self.v = -self.v;
        for i in 0..self.n {
            self.g[i] = -self.g[i];
        }
        self
    }
}

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Dual<N>;
    #[inline]
    fn add(mut self, o: f64) -> Dual<N> {
        self.v += o;
        self
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Dual<N>;
    #[inline]
    fn sub(mut self, o: f64) -> Dual<N> {
        self.v -= o;
        self
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Dual<N>;
    #[inline]
    fn mul(mut self, o: f64) -> Dual<N> {
        self.v *= o;
        for i in 0..self.n {
            self.g[i] *= o;
        }
        self
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Dual<N>;
    #[inline]
    fn div(mut self, o: f64) -> Dual<N> {
        self.v /= o;
        for i in 0..self.n {
            self.g[i] /= o;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f<S: Scalar>(x: S, y: S) -> S {
        ((x * y).exp() + x / y).ln().sqrt() * (x - y * 2.0).ncdf()
    }

    #[test]
    fn dual_gradients_match_finite_differences() {
        let (x, y) = (0.3, 0.7);
        let d = f(Dual::<2>::variable(x, 0, 2), Dual::<2>::variable(y, 1, 2));
        let h = 1e-6;
        let gx = (f(x + h, y) - f(x - h, y)) / (2.0 * h);
        let gy = (f(x, y + h) - f(x, y - h)) / (2.0 * h);
        assert!((d.v - f(x, y)).abs() < 1e-15);
        assert!((d.grad()[0] - gx).abs() < 1e-8);
        assert!((d.grad()[1] - gy).abs() < 1e-8);
    }

    #[test]
    fn clamp_subgradients_follow_torch_2_14() {
        let x = |v: f64| Dual::<2>::variable(v, 0, 2);
        // Scalar bound: gradient 1 above the bound, 0 at and below it.
        assert_eq!(x(0.5).clamp_min(0.5).grad(), &[0.0, 0.0]);
        assert_eq!(x(0.6).clamp_min(0.5).grad(), &[1.0, 0.0]);
        assert_eq!(x(0.4).clamp_min(0.5).grad(), &[0.0, 0.0]);
        // The value is kept at a tie (a signed zero survives).
        assert!(x(-0.0).clamp_min(0.0).v.is_sign_negative());
        // Tensor bound: the gradient is split evenly at a tie.
        let lo = Dual::<2>::variable(0.5, 1, 2);
        assert_eq!(x(0.5).clamp_min_var(lo).grad(), &[0.5, 0.5]);
        assert_eq!(x(0.6).clamp_min_var(lo).grad(), &[1.0, 0.0]);
        assert_eq!(x(0.4).clamp_min_var(lo).grad(), &[0.0, 1.0]);
        assert_eq!(0.5_f64.clamp_min_var(0.5), 0.5);
    }
}
