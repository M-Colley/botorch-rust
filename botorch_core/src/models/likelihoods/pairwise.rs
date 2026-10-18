//! Pairwise likelihoods for preference models (`botorch.models.likelihoods.pairwise`):
//! [`PairwiseLikelihood::Probit`] (`PairwiseProbitLikelihood`) and
//! [`PairwiseLikelihood::Logit`] (`PairwiseLogitLikelihood`).
//!
//! A set of `m` comparisons of `n` items is BoTorch's matrix `D` (`m × n`):
//! row `k` is zero except for `D[k, w] = 1` at the preferred item `w` and
//! `D[k, l] = -1` at the other item `l`; [`Comparisons`] stores these rows
//! sparsely. A comparison of an item with itself (which consolidation
//! creates from a comparison of two near-duplicate datapoints) is a zero row
//! (`e_w - e_l = 0`: probability 1/2 for every utility). BoTorch fills `D`
//! with two `scatter_` calls, so that its row keeps `-1` at the item (a
//! probit likelihood `Φ(-f_i/√2)` that depends on the utility, and a crash
//! of the logit likelihood): a BoTorch bug (see `docs/drafts/10b-preference.md`).
//!
//! The values are BoTorch's expressions evaluated in the same order:
//!
//! * probit: `z = clamp(D (u / √2), -3, 3)`, `p = Φ(z)` (torch's
//!   `Normal.cdf`, `0.5 (1 + erf(z / √2))`), `log p = log(p)`, the hazard
//!   `h = exp(standard_normal_log_hazard(-z)) = φ(z) / Φ(z)`, the negative
//!   log-likelihood gradient `-(Dᵀ h) / √2` and Hessian
//!   `Dᵀ diag(h (h + z) / 2) D`;
//! * logit: `log p = logsigmoid(clamp(D u, -8, 8))`, gradient
//!   `-Dᵀ (e^{u_l} / (e^{u_w} + e^{u_l}))` and Hessian `Dᵀ diag(t - t²) D`
//!   with `t = sigmoid(-D u)` (both unclamped, as in BoTorch).
//!
//! The derivatives that autograd takes through these expressions (needed
//! for the gradient of the Laplace marginal log likelihood, which BoTorch
//! differentiates through two Newton steps) are provided as well: the
//! Jacobian of the gradient ([`PairwiseLikelihood::gradient_jacobian`]), and
//! the vector-Jacobian products of the log likelihood and of the Hessian.
//! Derivatives of `clamp` follow torch ≥ 2.14 (zero exactly at the bounds,
//! see `docs/clamp-subgradients.md`).

use crate::error::{BotorchError, Result};
use crate::utils::probability::{standard_normal_log_hazard, INV_SQRT_2PI};
use crate::utils::safe_math::sigmoid;
use crate::{Matrix, Vector};

/// `math.sqrt(2)`.
const SQRT_2: f64 = std::f64::consts::SQRT_2;
/// The probit likelihood's clamp of the z-scores (`_zlim`).
pub const PROBIT_ZLIM: f64 = 3.0;
/// The logit likelihood's clamp of the logits (`_logit_lim`).
pub const LOGIT_LIM: f64 = 8.0;

/// The comparisons of a pairwise model as the rows of BoTorch's matrix `D`.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparisons {
    n: usize,
    pairs: Vec<(usize, usize)>,
    /// The nonzero entries `(item, value)` of every row of `D`.
    rows: Vec<Vec<(usize, f64)>>,
}

impl Comparisons {
    /// The comparisons `pairs[k] = (preferred item, other item)` of `n`
    /// items.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an item index out of range
    /// (torch's `scatter_` error).
    pub fn new(pairs: Vec<(usize, usize)>, n: usize) -> Result<Self> {
        for &(w, l) in &pairs {
            for i in [w, l] {
                if i >= n {
                    return Err(BotorchError::InvalidArgument(format!(
                        "index {i} is out of bounds for dimension 1 with size {n}"
                    )));
                }
            }
        }
        let rows = pairs
            .iter()
            .map(|&(w, l)| {
                if w == l {
                    Vec::new()
                } else {
                    vec![(w, 1.0), (l, -1.0)]
                }
            })
            .collect();
        Ok(Self { n, pairs, rows })
    }

    /// The number of items `n`.
    pub fn num_items(&self) -> usize {
        self.n
    }

    /// The number of comparisons `m`.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether there are no comparisons.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The comparisons `(preferred, other)`.
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// BoTorch's dense `D` (`m × n`).
    pub fn d_matrix(&self) -> Matrix {
        let mut d = Matrix::zeros(self.len(), self.n);
        for (k, row) in self.rows.iter().enumerate() {
            for &(i, v) in row {
                d[(k, i)] = v;
            }
        }
        d
    }

    /// `D v` (length `m`).
    pub fn apply(&self, v: &Vector) -> Vector {
        Vector::from_iterator(
            self.len(),
            self.rows
                .iter()
                .map(|row| row.iter().fold(0.0, |s, &(i, d)| s + d * v[i])),
        )
    }

    /// `Dᵀ c` (length `n`), accumulated in the order of the comparisons.
    pub fn apply_t(&self, c: &Vector) -> Vector {
        let mut out = Vector::zeros(self.n);
        for (k, row) in self.rows.iter().enumerate() {
            for &(i, d) in row {
                out[i] += d * c[k];
            }
        }
        out
    }

    /// `Dᵀ diag(w) D` (`n × n`).
    pub fn weighted_gram(&self, w: &Vector) -> Matrix {
        let mut h = Matrix::zeros(self.n, self.n);
        for (k, row) in self.rows.iter().enumerate() {
            for &(i, di) in row {
                for &(j, dj) in row {
                    h[(i, j)] += di * w[k] * dj;
                }
            }
        }
        h
    }

    /// `diag(D A Dᵀ)` (length `m`) for an `n × n` matrix `A`.
    pub fn quad_diag(&self, a: &Matrix) -> Vector {
        Vector::from_iterator(
            self.len(),
            self.rows.iter().map(|row| {
                let mut s = 0.0;
                for &(i, di) in row {
                    for &(j, dj) in row {
                        s += di * a[(i, j)] * dj;
                    }
                }
                s
            }),
        )
    }

    /// The column sums of `D` (each item's wins minus losses).
    pub fn win_count(&self) -> Vector {
        self.apply_t(&Vector::from_element(self.len(), 1.0))
    }

    /// The winner and loser of every comparison as BoTorch's logit
    /// likelihood reads them (`(D == 1).nonzero()` and `(D == -1).nonzero()`
    /// reshaped to one per comparison).
    ///
    /// # Errors
    /// None (a comparison of an item with itself has a zero row; BoTorch's
    /// reshape fails on it).
    fn winners_losers(&self) -> Result<Vec<(usize, usize)>> {
        Ok(self.pairs.clone())
    }
}

/// A pairwise likelihood (BoTorch's `PairwiseLikelihood` subclasses).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PairwiseLikelihood {
    /// `PairwiseProbitLikelihood` (the default of `PairwiseGP`).
    #[default]
    Probit,
    /// `PairwiseLogitLikelihood`.
    Logit,
}

/// The per-comparison quantities of the probit likelihood at a utility.
struct ProbitTerms {
    /// The clamped z-scores.
    z: Vector,
    /// Whether the z-score lies strictly inside the clamp (derivative 1).
    inside: Vec<bool>,
    /// The hazard `h = φ(z) / Φ(z)`.
    h: Vector,
}

impl PairwiseLikelihood {
    /// The name of BoTorch's class.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Probit => "PairwiseProbitLikelihood",
            Self::Logit => "PairwiseLogitLikelihood",
        }
    }

    fn probit_terms(u: &Vector, d: &Comparisons) -> ProbitTerms {
        let scaled = u / SQRT_2;
        let s = d.apply(&scaled);
        let z = s.map(|v| v.clamp(-PROBIT_ZLIM, PROBIT_ZLIM));
        let inside = s
            .iter()
            .map(|&v| v > -PROBIT_ZLIM && v < PROBIT_ZLIM)
            .collect();
        let h = z.map(|v| standard_normal_log_hazard(-v).exp());
        ProbitTerms { z, inside, h }
    }

    /// The logits `D u` (unclamped).
    fn logits(u: &Vector, d: &Comparisons) -> Vector {
        d.apply(u)
    }

    /// The probability of every comparison (BoTorch's `p`, length `m`).
    pub fn p(&self, u: &Vector, d: &Comparisons) -> Vector {
        match self {
            Self::Probit => {
                let t = Self::probit_terms(u, d);
                t.z.map(normal_cdf)
            }
            Self::Logit => Self::logits(u, d).map(|l| sigmoid(l.clamp(-LOGIT_LIM, LOGIT_LIM))),
        }
    }

    /// The log probability of every comparison (BoTorch's `log_p`).
    pub fn log_p(&self, u: &Vector, d: &Comparisons) -> Vector {
        match self {
            Self::Probit => self.p(u, d).map(f64::ln),
            Self::Logit => Self::logits(u, d).map(|l| logsigmoid(l.clamp(-LOGIT_LIM, LOGIT_LIM))),
        }
    }

    /// The gradient of `Σ_k log_p_k` with respect to the utility, as
    /// autograd computes it (zero through clamped comparisons).
    pub fn log_p_sum_grad(&self, u: &Vector, d: &Comparisons) -> Vector {
        match self {
            Self::Probit => {
                let t = Self::probit_terms(u, d);
                // d log Φ(z)/dz through torch's erf: φ(z) / Φ(z).
                let c = Vector::from_iterator(
                    d.len(),
                    (0..d.len()).map(|k| {
                        if t.inside[k] {
                            let z = t.z[k];
                            INV_SQRT_2PI * (-0.5 * z * z).exp() / normal_cdf(z) / SQRT_2
                        } else {
                            0.0
                        }
                    }),
                );
                d.apply_t(&c)
            }
            Self::Logit => {
                let l = Self::logits(u, d);
                let c = l.map(|v| {
                    if v > -LOGIT_LIM && v < LOGIT_LIM {
                        sigmoid(-v)
                    } else {
                        0.0
                    }
                });
                d.apply_t(&c)
            }
        }
    }

    /// The sum over comparisons of the gradient of the negative log
    /// likelihood with respect to the utility (BoTorch's
    /// `negative_log_gradient_sum`, length `n`).
    ///
    /// # Errors
    /// The logit likelihood's error for a comparison of an item with itself.
    pub fn negative_log_gradient_sum(&self, u: &Vector, d: &Comparisons) -> Result<Vector> {
        match self {
            Self::Probit => {
                let t = Self::probit_terms(u, d);
                let hf = t.h / SQRT_2;
                Ok(-d.apply_t(&hf))
            }
            Self::Logit => {
                let wl = d.winners_losers()?;
                let g = Vector::from_iterator(
                    wl.len(),
                    wl.iter().map(|&(w, l)| {
                        let ex = u[w].exp();
                        let ey = u[l].exp();
                        if ex.is_finite() && ey.is_finite() {
                            ey / (ex + ey)
                        } else {
                            // BoTorch's `ey / (ex + ey)` is `inf / inf = NaN`
                            // for utilities above ~709 (a bug); the value is
                            // `sigmoid(u_l - u_w)`.
                            sigmoid(u[l] - u[w])
                        }
                    }),
                );
                Ok(-d.apply_t(&g))
            }
        }
    }

    /// The per-comparison weights `w` of the Hessian `Dᵀ diag(w) D`.
    fn hessian_weights(&self, u: &Vector, d: &Comparisons) -> Vector {
        match self {
            Self::Probit => {
                let t = Self::probit_terms(u, d);
                Vector::from_iterator(
                    d.len(),
                    (0..d.len()).map(|k| t.h[k] * (t.h[k] + t.z[k]) / 2.0),
                )
            }
            Self::Logit => Self::logits(u, d).map(|l| {
                let t = sigmoid(-l);
                t - t * t
            }),
        }
    }

    /// The sum over comparisons of the Hessian of the negative log
    /// likelihood with respect to the utility (BoTorch's
    /// `negative_log_hessian_sum`, `n × n`).
    pub fn negative_log_hessian_sum(&self, u: &Vector, d: &Comparisons) -> Matrix {
        d.weighted_gram(&self.hessian_weights(u, d))
    }

    /// The Jacobian of [`PairwiseLikelihood::negative_log_gradient_sum`]
    /// as autograd computes it: the Hessian, except that clamped probit
    /// comparisons contribute nothing (the gradient is constant there).
    ///
    /// # Errors
    /// See [`PairwiseLikelihood::negative_log_gradient_sum`].
    pub fn gradient_jacobian(&self, u: &Vector, d: &Comparisons) -> Result<Matrix> {
        match self {
            Self::Probit => {
                let t = Self::probit_terms(u, d);
                let w = Vector::from_iterator(
                    d.len(),
                    (0..d.len()).map(|k| {
                        if t.inside[k] {
                            t.h[k] * (t.h[k] + t.z[k]) / 2.0
                        } else {
                            0.0
                        }
                    }),
                );
                Ok(d.weighted_gram(&w))
            }
            Self::Logit => {
                d.winners_losers()?;
                Ok(self.negative_log_hessian_sum(u, d))
            }
        }
    }

    /// The vector-Jacobian product of the Hessian: the gradient of
    /// `Σ_ij A_ij H(u)_ij` with respect to `u` for an `n × n` matrix `A`.
    pub fn hessian_vjp(&self, u: &Vector, d: &Comparisons, a: &Matrix) -> Vector {
        let q = d.quad_diag(a);
        match self {
            Self::Probit => {
                let t = Self::probit_terms(u, d);
                let c = Vector::from_iterator(
                    d.len(),
                    (0..d.len()).map(|k| {
                        if !t.inside[k] {
                            return 0.0;
                        }
                        let (h, z) = (t.h[k], t.z[k]);
                        let dh = -h * (h + z);
                        let dw = (dh * (h + z) + h * (dh + 1.0)) / 2.0;
                        q[k] * dw / SQRT_2
                    }),
                );
                d.apply_t(&c)
            }
            Self::Logit => {
                let l = Self::logits(u, d);
                let c = Vector::from_iterator(
                    d.len(),
                    (0..d.len()).map(|k| {
                        let t = sigmoid(-l[k]);
                        let dt = -t * (1.0 - t);
                        q[k] * (1.0 - 2.0 * t) * dt
                    }),
                );
                d.apply_t(&c)
            }
        }
    }
}

/// torch's `Normal(0, 1).cdf(z)`: `0.5 (1 + erf(z / √2))`.
pub(crate) fn normal_cdf(z: f64) -> f64 {
    use errorfunctions::RealErrorFunctions;
    0.5 * (1.0 + RealErrorFunctions::erf(z / SQRT_2))
}

/// torch's `logsigmoid`: `min(x, 0) - log1p(exp(-|x|))`.
fn logsigmoid(x: f64) -> f64 {
    x.min(0.0) - (-x.abs()).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comps() -> Comparisons {
        Comparisons::new(vec![(0, 1), (2, 1), (3, 0), (1, 3)], 4).unwrap()
    }

    fn fd<F: Fn(&Vector) -> f64>(f: F, u: &Vector) -> Vector {
        let h = 1e-6;
        Vector::from_iterator(
            u.len(),
            (0..u.len()).map(|i| {
                let mut a = u.clone();
                let mut b = u.clone();
                a[i] += h;
                b[i] -= h;
                (f(&a) - f(&b)) / (2.0 * h)
            }),
        )
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let d = comps();
        let u = Vector::from_vec(vec![0.3, -0.8, 1.1, 0.2]);
        for lik in [PairwiseLikelihood::Probit, PairwiseLikelihood::Logit] {
            let g = lik.log_p_sum_grad(&u, &d);
            let f = fd(|v| lik.log_p(v, &d).sum(), &u);
            assert!((g - f).amax() < 1e-8, "{lik:?} log_p grad");
            // The negative log gradient is the derivative of -Σ log p
            // (inside the clamps).
            let b = lik.negative_log_gradient_sum(&u, &d).unwrap();
            assert!((b + lik.log_p_sum_grad(&u, &d)).amax() < 1e-12, "{lik:?} b");
            let j = lik.gradient_jacobian(&u, &d).unwrap();
            for i in 0..4 {
                let col = fd(|v| lik.negative_log_gradient_sum(v, &d).unwrap()[i], &u);
                assert!((j.row(i).transpose() - col).amax() < 1e-7, "{lik:?} jac");
            }
            let a = Matrix::from_fn(4, 4, |i, j| (i as f64 + 1.0) * 0.3 - j as f64 * 0.2);
            let v = lik.hessian_vjp(&u, &d, &a);
            let f = fd(
                |x| lik.negative_log_hessian_sum(x, &d).component_mul(&a).sum(),
                &u,
            );
            assert!((v - f).amax() < 1e-7, "{lik:?} hessian vjp");
        }
    }

    #[test]
    fn logit_gradient_does_not_overflow() {
        let d = Comparisons::new(vec![(0, 1)], 2).unwrap();
        let lik = PairwiseLikelihood::Logit;
        let big = lik
            .negative_log_gradient_sum(&Vector::from_vec(vec![800.0, 800.0]), &d)
            .unwrap();
        let small = lik
            .negative_log_gradient_sum(&Vector::from_vec(vec![1.0, 1.0]), &d)
            .unwrap();
        assert_eq!(big, small);
        assert_eq!(big[0], -0.5);
    }

    #[test]
    fn self_comparison_is_uninformative() {
        let d = Comparisons::new(vec![(1, 1), (0, 1)], 2).unwrap();
        let dm = d.d_matrix();
        assert_eq!(dm.row(0).amax(), 0.0);
        let u = Vector::from_vec(vec![0.4, -1.3]);
        for lik in [PairwiseLikelihood::Probit, PairwiseLikelihood::Logit] {
            let p = lik.p(&u, &d);
            assert!((p[0] - 0.5).abs() < 1e-15, "{lik:?}");
            let only = Comparisons::new(vec![(0, 1)], 2).unwrap();
            let b = lik.negative_log_gradient_sum(&u, &d).unwrap();
            let b1 = lik.negative_log_gradient_sum(&u, &only).unwrap();
            assert!((b - b1).amax() < 1e-15, "{lik:?}");
        }
        assert!(Comparisons::new(vec![(0, 2)], 2).is_err());
    }
}
