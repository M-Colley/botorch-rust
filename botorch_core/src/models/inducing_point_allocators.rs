//! Allocation of the inducing points of sparse GP models (BoTorch's
//! `botorch.models.utils.inducing_point_allocators`).
//!
//! The allocators select inducing points among the training inputs by a
//! greedy approximation of the MAP estimate of a determinantal point process
//! (DPP) defined through its diversity-quality decomposition: the kernel
//! matrix of the training inputs (diversity) and per-point quality scores.
//! [`pivoted_cholesky_init`] is BoTorch's `_pivoted_cholesky_init` (Chen et
//! al. 2018, "Fast greedy MAP inference for determinantal point process"),
//! which is numerically a pivoted Cholesky factorization that records the
//! pivots; the quality functions are BoTorch's `UnitQualityFunction`
//! ([`unit_quality`], the greedy variance reduction of Burt et al. 2020) and
//! `ExpectedImprovementQualityFunction` ([`expected_improvement_quality`],
//! Moss et al. 2023).

use crate::error::{BotorchError, Result};
use crate::utils::probability::{ndtr, phi};
use crate::Matrix;

/// The default numerical jitter of [`pivoted_cholesky_init`] (BoTorch's
/// `epsilon = 1e-6`): the selection stops once the largest remaining
/// conditional variance is smaller.
pub const PIVOTED_CHOLESKY_EPSILON: f64 = 1e-6;

/// `torch.argmax` of a vector: the index of the first maximal value, where a
/// NaN counts as the maximum (the first NaN wins), as torch propagates NaN
/// through its reductions.
fn torch_argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            return i;
        }
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// BoTorch's `_pivoted_cholesky_init`: the indices of the training inputs
/// chosen as inducing points, in the order of selection.
///
/// Starting from the diagonal `d_i = K_ii` of the kernel matrix (the prior
/// variances), the point with the largest score `d_i q_i²` is selected,
/// the conditional variances are updated with the new row of the pivoted
/// Cholesky factor, and selection continues until `max_length` points are
/// chosen or the conditional variance of the best remaining point drops
/// below `epsilon` (then fewer points are returned, as in BoTorch). Ties
/// are broken by the lowest index and NaN scores count as maximal (torch's
/// `argmax`).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] (BoTorch's `ValueError`) if `quality`
/// does not have one score per row of the square `kernel_matrix`, and
/// [`BotorchError::DimensionMismatch`] for a matrix that is not square.
pub fn pivoted_cholesky_init(
    kernel_matrix: &Matrix,
    max_length: usize,
    quality: &[f64],
    epsilon: f64,
) -> Result<Vec<usize>> {
    let n = kernel_matrix.nrows();
    if quality.len() != n {
        return Err(BotorchError::InvalidArgument(
            "_pivoted_cholesky_init requires a quality score for each of train_inputs".into(),
        ));
    }
    if kernel_matrix.ncols() != n {
        return Err(BotorchError::DimensionMismatch(format!(
            "the kernel matrix must be square, got {}x{}",
            n,
            kernel_matrix.ncols()
        )));
    }
    if n == 0 {
        return Err(BotorchError::InvalidArgument(
            "argmax(): Expected reduction dim to be specified for input.numel() == 0.".into(),
        ));
    }
    let q2: Vec<f64> = quality.iter().map(|q| q * q).collect();
    // Rows of the pivoted Cholesky factor (`cis` in BoTorch), `max_length x n`.
    let mut cis: Vec<Vec<f64>> = Vec::with_capacity(max_length);
    let mut di2s: Vec<f64> = (0..n).map(|i| kernel_matrix[(i, i)]).collect();
    let scores: Vec<f64> = di2s.iter().zip(&q2).map(|(d, q)| d * q).collect();
    let mut selected = torch_argmax(&scores);
    let mut items = vec![selected];
    while items.len() < max_length {
        let k = items.len() - 1;
        let di_optimal = di2s[selected].sqrt();
        // elements - ci_optimal @ cis[:k, :], with ci_optimal = cis[:k, selected].
        let mut eis: Vec<f64> = (0..n).map(|j| kernel_matrix[(selected, j)]).collect();
        let mut proj = vec![0.0; n];
        for row in cis.iter().take(k) {
            let c = row[selected];
            for (p, r) in proj.iter_mut().zip(row) {
                *p += c * r;
            }
        }
        for (e, p) in eis.iter_mut().zip(&proj) {
            *e = (*e - p) / di_optimal;
        }
        for (d, e) in di2s.iter_mut().zip(&eis) {
            *d -= e * e;
        }
        cis.push(eis);
        di2s[selected] = f64::NEG_INFINITY;
        let scores: Vec<f64> = di2s.iter().zip(&q2).map(|(d, q)| d * q).collect();
        selected = torch_argmax(&scores);
        // `di2s[selected] < epsilon` is false for NaN, as in torch.
        if di2s[selected] < epsilon {
            break;
        }
        items.push(selected);
    }
    items.truncate(max_length);
    Ok(items)
}

/// BoTorch's `UnitQualityFunction`: a quality of one for every point.
pub fn unit_quality(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// BoTorch's `ExpectedImprovementQualityFunction` from the posterior means
/// and variances of the previous model at the points: the expected
/// improvement `σ (φ(u) + u Φ(u))` over the best posterior mean (the
/// largest if `maximize`, else the smallest) with `σ = sqrt(max(v, 1e-12))`
/// and `u = ±(μ - best) / σ`.
///
/// The variances are those BoTorch reads from `posterior.variance`
/// (already clamped at GPyTorch's `1e-10` for Gaussian posteriors).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] if `mean` and `variance` differ in
/// length.
pub fn expected_improvement_quality(
    mean: &[f64],
    variance: &[f64],
    maximize: bool,
) -> Result<Vec<f64>> {
    if mean.len() != variance.len() {
        return Err(BotorchError::DimensionMismatch(format!(
            "{} posterior means but {} variances",
            mean.len(),
            variance.len()
        )));
    }
    if mean.is_empty() {
        return Ok(Vec::new());
    }
    // torch.max / torch.min propagate NaN.
    let best_f = if mean.iter().any(|m| m.is_nan()) {
        f64::NAN
    } else if maximize {
        mean.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    } else {
        mean.iter().copied().fold(f64::INFINITY, f64::min)
    };
    Ok(mean
        .iter()
        .zip(variance)
        .map(|(&m, &v)| {
            let sigma = if v < 1e-12 { 1e-12 } else { v }.sqrt();
            let u = if maximize {
                (m - best_f) / sigma
            } else {
                -(m - best_f) / sigma
            };
            sigma * (phi(u) + u * ndtr(u))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rbf(x: &[f64], ls: f64) -> Matrix {
        let n = x.len();
        Matrix::from_fn(n, n, |i, j| {
            let d = (x[i] - x[j]) / ls;
            (-0.5 * d * d).exp()
        })
    }

    #[test]
    fn argmax_follows_torch() {
        assert_eq!(torch_argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(torch_argmax(&[1.0, f64::NAN, 5.0, f64::NAN]), 1);
        assert_eq!(torch_argmax(&[f64::NEG_INFINITY, -1.0]), 1);
    }

    #[test]
    fn greedy_selection_maximizes_conditional_variance() {
        let x = [0.0, 0.1, 0.5, 0.9, 1.0, 0.45];
        let k = rbf(&x, 0.3);
        let sel = pivoted_cholesky_init(&k, 3, &unit_quality(6), 1e-6).unwrap();
        // The first pivot is the first point (equal prior variances).
        assert_eq!(sel[0], 0);
        // Every later pivot has the largest conditional variance given the
        // previous ones (checked against an explicit Schur complement).
        for t in 1..sel.len() {
            let prev = &sel[..t];
            let kpp = Matrix::from_fn(t, t, |a, b| k[(prev[a], prev[b])]);
            let inv = kpp.try_inverse().unwrap();
            let cond = |i: usize| {
                let kp = Matrix::from_fn(t, 1, |a, _| k[(prev[a], i)]);
                k[(i, i)] - (kp.transpose() * &inv * &kp)[(0, 0)]
            };
            let best = (0..6)
                .filter(|i| !prev.contains(i))
                .map(cond)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!((cond(sel[t]) - best).abs() < 1e-12);
        }
    }

    #[test]
    fn duplicates_stop_the_selection_early() {
        let x = [0.2, 0.2, 0.7, 0.7];
        let k = rbf(&x, 0.3);
        let sel = pivoted_cholesky_init(&k, 4, &unit_quality(4), 1e-6).unwrap();
        assert_eq!(sel, vec![0, 2]);
        assert!(pivoted_cholesky_init(&k, 4, &[1.0; 3], 1e-6).is_err());
        assert_eq!(
            pivoted_cholesky_init(&k, 0, &[1.0; 4], 1e-6).unwrap(),
            vec![]
        );
    }

    #[test]
    fn ei_quality_matches_closed_form() {
        let q = expected_improvement_quality(&[0.0, 1.0], &[1.0, 0.0], true).unwrap();
        let u = -1.0;
        assert!((q[0] - (phi(u) + u * ndtr(u))).abs() < 1e-15);
        // A point at the best mean with (clamped) zero variance.
        assert!((q[1] - 1e-6 * phi(0.0)).abs() < 1e-20);
        let qm = expected_improvement_quality(&[0.0, 1.0], &[1.0, 1.0], false).unwrap();
        assert!((qm[0] - phi(0.0)).abs() < 1e-15);
    }
}
