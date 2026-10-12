//! Scalarizations (`botorch.utils.multi_objective.scalarization`): the
//! augmented Chebyshev scalarization of ParEGO ([`ChebyshevScalarization`],
//! [`get_chebyshev_scalarization`]).
//!
//! Reference: J. Knowles. *ParEGO: a hybrid algorithm with on-line landscape
//! approximation for expensive multiobjective optimization problems.* IEEE
//! Transactions on Evolutionary Computation 10(1), 2006.

use crate::error::{BotorchError, Result};
use crate::{Matrix, Vector};

/// The augmented Chebyshev scalarization returned by BoTorch's
/// `get_chebyshev_scalarization(weights, Y, alpha)`:
///
/// `s(y) = -(max_j w_j ŷ_j + alpha Σ_j w_j ŷ_j)`, with `ŷ = (-y - l) / (u - l)`
/// for the bounds `[l, u]` of the negated observed outcomes (an upper bound
/// equal to its lower bound is replaced by `l + 1`, as in BoTorch's
/// `normalize`), minus 1 for the outcomes with negative weights (to be
/// minimized); without observations, `ŷ = -y`. Larger values are better.
#[derive(Clone, Debug, PartialEq)]
pub struct ChebyshevScalarization {
    weights: Vec<f64>,
    alpha: f64,
    /// `Y_bounds` of BoTorch (the negated outcomes' minima and maxima),
    /// `None` without observations.
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    /// The divisors of `normalize`: `u - l`, or `(l + 1) - l` for constant
    /// outcomes (`_update_constant_bounds`).
    ranges: Vec<f64>,
}

/// BoTorch's `get_chebyshev_scalarization(weights, Y, alpha)`: the
/// augmented Chebyshev scalarization with `m` `weights` (positive for
/// maximization, negative for minimization) normalized with the observed
/// outcomes `y` (`n × m`; may have no rows).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] if `weights` does not have `m`
/// values; [`BotorchError::Unsupported`] for negative weights without
/// observations (BoTorch's messages).
pub fn get_chebyshev_scalarization(
    weights: &[f64],
    y: &Matrix,
    alpha: f64,
) -> Result<ChebyshevScalarization> {
    ChebyshevScalarization::new(weights, y, alpha)
}

impl ChebyshevScalarization {
    /// See [`get_chebyshev_scalarization`].
    ///
    /// # Errors
    /// As [`get_chebyshev_scalarization`].
    pub fn new(weights: &[f64], y: &Matrix, alpha: f64) -> Result<Self> {
        let (n, m) = y.shape();
        if weights.len() != m {
            return Err(BotorchError::DimensionMismatch(format!(
                "weights must be an `m`-dim tensor where Y is `... x m`.Got shapes \
                 torch.Size([{}]) and torch.Size([{n}, {m}]).",
                weights.len()
            )));
        }
        if n == 0 {
            if weights.iter().any(|&w| w < 0.0) {
                return Err(BotorchError::Unsupported(
                    "negative weights (for minimization) are only supported if Y is provided."
                        .into(),
                ));
            }
            return Ok(Self {
                weights: weights.to_vec(),
                alpha,
                bounds: None,
                ranges: Vec::new(),
            });
        }
        // The bounds of the negated outcomes: [min(-Y), max(-Y)].
        let mut lower = vec![0.0; m];
        let mut upper = vec![0.0; m];
        let mut ranges = vec![0.0; m];
        for j in 0..m {
            let col = (0..n).map(|i| -y[(i, j)]);
            lower[j] = col.clone().fold(f64::INFINITY, super::nan_min);
            upper[j] = col.fold(f64::NEG_INFINITY, super::nan_max);
            // `normalize` with `_update_constant_bounds`: a constant outcome
            // is only shifted.
            let u = if upper[j] == lower[j] {
                lower[j] + 1.0
            } else {
                upper[j]
            };
            ranges[j] = u - lower[j];
        }
        Ok(Self {
            weights: weights.to_vec(),
            alpha,
            bounds: Some((lower, upper)),
            ranges,
        })
    }

    /// The weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The weight `alpha` of the sum term.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The number of outcomes.
    pub fn num_outcomes(&self) -> usize {
        self.weights.len()
    }

    /// The normalization bounds of the negated outcomes (lower, upper;
    /// BoTorch's `Y_bounds`), or `None` if there were no observations.
    pub fn bounds(&self) -> Option<(&[f64], &[f64])> {
        self.bounds
            .as_ref()
            .map(|(l, u)| (l.as_slice(), u.as_slice()))
    }

    /// The scaled weighted outcome `w_j ŷ_j` and its derivative with
    /// respect to `y_j`.
    #[inline]
    fn product(&self, j: usize, y: f64) -> (f64, f64) {
        let w = self.weights[j];
        match &self.bounds {
            None => (w * -y, -w),
            Some((l, _)) => {
                let range = self.ranges[j];
                let mut v = (-y - l[j]) / range;
                if w < 0.0 {
                    v -= 1.0;
                }
                (w * v, -w / range)
            }
        }
    }

    /// The scalarization of one outcome vector (`m` values).
    ///
    /// # Panics
    /// If `y` does not have `m` values.
    pub fn evaluate(&self, y: &[f64]) -> f64 {
        assert_eq!(
            y.len(),
            self.weights.len(),
            "expected {} outcomes",
            self.weights.len()
        );
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        for (j, &v) in y.iter().enumerate() {
            let (p, _) = self.product(j, v);
            if p > max || p.is_nan() && !max.is_nan() {
                max = p;
            }
            sum += p;
        }
        -(max + self.alpha * sum)
    }

    /// The scalarization of one outcome vector and its gradient with respect
    /// to `y` (written to `grad`). The maximum is differentiated at the first
    /// maximal term, like torch's `max` backward.
    ///
    /// # Panics
    /// If `y` or `grad` do not have `m` values.
    pub fn evaluate_with_grad(&self, y: &[f64], grad: &mut [f64]) -> f64 {
        let m = self.weights.len();
        assert!(y.len() == m && grad.len() == m, "expected {m} outcomes");
        let mut max = f64::NEG_INFINITY;
        let mut arg = 0;
        let mut sum = 0.0;
        for (j, &v) in y.iter().enumerate() {
            let (p, dp) = self.product(j, v);
            if j == 0 || p > max || p.is_nan() && !max.is_nan() {
                max = p;
                arg = j;
            }
            sum += p;
            grad[j] = -self.alpha * dp;
        }
        grad[arg] -= self.product(arg, y[arg]).1;
        -(max + self.alpha * sum)
    }

    /// The scalarization of `n` outcome vectors stored row-major (`n·m`
    /// values).
    ///
    /// # Panics
    /// If `y.len()` is not a multiple of `m`.
    pub fn evaluate_rows(&self, y: &[f64]) -> Vec<f64> {
        let m = self.weights.len().max(1);
        assert_eq!(y.len() % m, 0, "expected rows of {m} outcomes");
        y.chunks_exact(m).map(|row| self.evaluate(row)).collect()
    }

    /// The scalarization of the rows of `y` (`n × m`).
    ///
    /// # Panics
    /// If `y` does not have `m` columns.
    pub fn evaluate_matrix(&self, y: &Matrix) -> Vector {
        assert_eq!(
            y.ncols(),
            self.weights.len(),
            "expected {} outcomes",
            self.weights.len()
        );
        Vector::from_vec(self.evaluate_rows(&super::row_major(y)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_gradients() {
        let y = Matrix::from_row_slice(3, 2, &[0.0, 1.0, 1.0, 3.0, 0.5, 2.0]);
        let s = get_chebyshev_scalarization(&[0.3, -0.7], &y, 0.05).unwrap();
        let (l, u) = s.bounds().unwrap();
        assert_eq!((l, u), (&[-1.0, -3.0][..], &[0.0, -1.0][..]));
        let x = [0.25, 1.5];
        // ŷ = ((-0.25 + 1) / 1, (-1.5 + 3) / 2 - 1) = (0.75, -0.25)
        let p: [f64; 2] = [0.3 * 0.75, -0.7 * -0.25];
        let expected = -(p[0].max(p[1]) + 0.05 * (p[0] + p[1]));
        assert_eq!(s.evaluate(&x), expected);
        let mut g = [0.0; 2];
        assert_eq!(s.evaluate_with_grad(&x, &mut g), expected);
        let h = 1e-6;
        for j in 0..2 {
            let (mut a, mut b) = (x, x);
            a[j] += h;
            b[j] -= h;
            let fd = (s.evaluate(&a) - s.evaluate(&b)) / (2.0 * h);
            assert!((fd - g[j]).abs() < 1e-8, "{j}: {fd} vs {}", g[j]);
        }
        assert_eq!(
            s.evaluate_matrix(&Matrix::from_row_slice(1, 2, &x))[0],
            expected
        );
        // No observations: the outcomes are not normalized.
        let s0 = get_chebyshev_scalarization(&[0.5, 0.5], &Matrix::zeros(0, 2), 0.0).unwrap();
        // -max(0.5 * -1, 0.5 * -3)
        assert_eq!(s0.evaluate(&[1.0, 3.0]), 0.5);
        assert!(get_chebyshev_scalarization(&[0.5, -0.5], &Matrix::zeros(0, 2), 0.0).is_err());
        assert!(get_chebyshev_scalarization(&[0.5], &y, 0.0).is_err());
        // Constant outcomes are shifted, not scaled.
        let c = Matrix::from_row_slice(2, 1, &[2.0, 2.0]);
        let s1 = get_chebyshev_scalarization(&[1.0], &c, 0.0).unwrap();
        assert_eq!(s1.evaluate(&[1.0]), -(-1.0 + 2.0));
    }
}
