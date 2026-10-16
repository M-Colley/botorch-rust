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
    /// `x.clamp_min(lo)` (zero gradient where clamped, as autograd).
    fn clamp_min(self, lo: f64) -> Self;
    /// `x.clamp(lo)` with a tensor lower bound (gradient of the active
    /// argument; autograd's `clamp` with tensor bounds passes the gradient
    /// to the bound where it is active).
    fn clamp_min_var(self, lo: Self) -> Self;
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
        } else {
            self
        }
    }
    fn clamp_min_var(self, lo: Self) -> Self {
        if self.v < lo.v {
            lo
        } else {
            self
        }
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
}
