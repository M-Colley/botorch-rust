//! Pareto dominance (`botorch.utils.multi_objective.pareto`).
//!
//! [`is_non_dominated`] returns BoTorch's mask: point `i` is kept unless some
//! point is at least as good in every outcome and strictly better in one,
//! and with `deduplicate` only the **first** of several identical points is
//! kept (`==` comparisons, so `-0.0` and `0.0` are identical).
//!
//! BoTorch computes the mask with one of two algorithms: an all-pairs
//! comparison for small inputs, and a loop over the points
//! (`_is_non_dominated_loop`) when `n > 1000` or the pairwise comparison
//! would exceed 5 MB (`n² · batch · 8` bytes in float64, 4 in float32). Both
//! return the same mask: in the loop, every point `i` that is still
//! efficient eliminates the efficient points `j ≠ i` that are not strictly
//! better than `i` in any outcome, i.e. those weakly dominated by `i`. A
//! dominated point is weakly dominated by a non-dominated point that is never
//! eliminated before it is visited (or by an earlier duplicate of it), so it
//! is eliminated; identical points eliminate each other in index order, so
//! the first one survives, which is what the all-pairs version's
//! `argmax`-based deduplication keeps. This module implements the loop (it is
//! `O(n · n_kept · m)` and allocation-free) for every input size; with
//! `deduplicate = false` the duplicates of the surviving points are added
//! back, as in BoTorch.
//!
//! NaN: a point with a NaN in any outcome is dominated (`false`) and does not
//! eliminate any other point. This is the documented behaviour, which
//! BoTorch implements since meta-pytorch/botorch#3392; BoTorch main only
//! masks out points whose outcomes are *all* NaN, so partially NaN points are
//! non-dominated there (except, with `deduplicate`, when they are not the
//! first point), and in the loop variant a NaN point can eliminate genuinely
//! non-dominated points.

use rayon::prelude::*;

use super::{row_major, PointBatch};
use crate::Matrix;

/// Minimal work (`batch · n² · m`, about 1 ns per unit) before batch
/// elements are processed in parallel: below about a millisecond, waking up
/// the worker threads costs more than it saves.
const PARALLEL_WORK: usize = 1 << 20;

/// BoTorch's `is_non_dominated(Y, maximize, deduplicate)` for one `n × m`
/// point set (rows are points): `mask[i]` tells whether point `i` is
/// non-dominated.
pub fn is_non_dominated(y: &Matrix, maximize: bool, deduplicate: bool) -> Vec<bool> {
    let values = row_major(y);
    is_non_dominated_rows(&values, y.nrows(), y.ncols(), maximize, deduplicate)
}

/// [`is_non_dominated`] for `n` points of `m` outcomes stored row-major in
/// `y` (`y.len() == n·m`).
///
/// # Panics
/// If `y.len() != n·m`.
pub fn is_non_dominated_rows(
    y: &[f64],
    n: usize,
    m: usize,
    maximize: bool,
    deduplicate: bool,
) -> Vec<bool> {
    let mut mask = vec![false; n];
    non_dominated_into(y, n, m, maximize, deduplicate, &mut mask);
    mask
}

/// [`is_non_dominated`] for every element of a batch (BoTorch's
/// `(batch_shape) x n x m` input): the flat `(batch_shape) x n` mask
/// (row-major). Batch elements are processed in parallel.
pub fn is_non_dominated_batch(y: &PointBatch, maximize: bool, deduplicate: bool) -> Vec<bool> {
    let (n, m, b) = (y.n(), y.m(), y.batch_size());
    let mut mask = vec![false; b * n];
    if n == 0 {
        return mask;
    }
    let work = b.saturating_mul(n).saturating_mul(n).saturating_mul(m);
    if b > 1 && m > 0 && work >= PARALLEL_WORK {
        mask.par_chunks_mut(n)
            .zip(y.values().par_chunks(n * m))
            .for_each(|(out, values)| {
                non_dominated_into(values, n, m, maximize, deduplicate, out);
            });
    } else {
        for (k, out) in mask.chunks_mut(n).enumerate() {
            non_dominated_into(y.element(k), n, m, maximize, deduplicate, out);
        }
    }
    mask
}

/// Writes the mask of the `n` points of `y` into `out`.
fn non_dominated_into(
    y: &[f64],
    n: usize,
    m: usize,
    maximize: bool,
    deduplicate: bool,
    out: &mut [bool],
) {
    assert_eq!(y.len(), n * m, "expected {n} x {m} values");
    let row = |i: usize| &y[i * m..(i + 1) * m];
    // Points with NaNs are dominated and never used to eliminate others.
    for (i, o) in out.iter_mut().enumerate() {
        *o = !row(i).iter().any(|v| v.is_nan());
    }
    for i in 0..n {
        if !out[i] {
            continue;
        }
        let yi = row(i);
        for (j, o) in out.iter_mut().enumerate() {
            if j == i || !*o {
                continue;
            }
            // `j` survives only if it is strictly better than `i` somewhere.
            let better = if maximize {
                row(j).iter().zip(yi).any(|(a, b)| a > b)
            } else {
                row(j).iter().zip(yi).any(|(a, b)| a < b)
            };
            if !better {
                *o = false;
            }
        }
    }
    if !deduplicate {
        // Add back the duplicates of the surviving points.
        let kept: Vec<usize> = (0..n).filter(|&i| out[i]).collect();
        for &i in &kept {
            let yi = row(i);
            for (j, o) in out.iter_mut().enumerate() {
                if !*o && row(j).iter().zip(yi).all(|(a, b)| a == b) {
                    *o = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The all-pairs definition (BoTorch's vectorized path, with NaN points
    /// masked out).
    fn reference(y: &[f64], n: usize, m: usize, maximize: bool, dedup: bool) -> Vec<bool> {
        let row = |i: usize| &y[i * m..(i + 1) * m];
        let dominates = |a: &[f64], b: &[f64]| {
            if maximize {
                a.iter().zip(b).all(|(x, y)| x >= y) && a.iter().zip(b).any(|(x, y)| x > y)
            } else {
                a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
            }
        };
        (0..n)
            .map(|i| {
                let nan = row(i).iter().any(|v| v.is_nan());
                let dominated = (0..n).any(|j| dominates(row(j), row(i)));
                let first = (0..i).all(|j| !row(j).iter().zip(row(i)).all(|(a, b)| a == b));
                !nan && !dominated && (!dedup || first)
            })
            .collect()
    }

    #[test]
    fn matches_the_pairwise_definition() {
        // Discrete values produce many ties and duplicates.
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % 4
        };
        for &(n, m) in &[(0usize, 2usize), (1, 3), (7, 2), (30, 3), (60, 4), (25, 1)] {
            let mut y: Vec<f64> = (0..n * m).map(|_| next() as f64).collect();
            if n > 5 {
                y[3 * m] = f64::NAN;
                y[m + m - 1] = f64::INFINITY;
            }
            for maximize in [true, false] {
                for dedup in [true, false] {
                    assert_eq!(
                        is_non_dominated_rows(&y, n, m, maximize, dedup),
                        reference(&y, n, m, maximize, dedup),
                        "n={n} m={m} maximize={maximize} dedup={dedup}"
                    );
                }
            }
        }
    }

    #[test]
    fn duplicates_nan_and_signed_zero() {
        let y = Matrix::from_row_slice(
            5,
            2,
            &[0.0, 1.0, 1.0, 0.0, -0.0, 1.0, f64::NAN, 5.0, 0.5, 0.5],
        );
        assert_eq!(
            is_non_dominated(&y, true, true),
            vec![true, true, false, false, true]
        );
        assert_eq!(
            is_non_dominated(&y, true, false),
            vec![true, true, true, false, true]
        );
        // A NaN in the first point does not keep it.
        let y = Matrix::from_row_slice(2, 2, &[f64::NAN, 0.0, 1.0, 1.0]);
        assert_eq!(is_non_dominated(&y, true, true), vec![false, true]);
        // m = 0: every point equals the first one.
        let y = Matrix::zeros(3, 0);
        assert_eq!(is_non_dominated(&y, true, true), vec![true, false, false]);
    }

    #[test]
    fn batches() {
        let a = Matrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
        let b = Matrix::from_row_slice(3, 2, &[1.0, 1.0, 0.0, 1.0, 0.5, 0.5]);
        let pb = PointBatch::from_matrices(&[a.clone(), b.clone()]).unwrap();
        let mut expected = is_non_dominated(&a, true, true);
        expected.extend(is_non_dominated(&b, true, true));
        assert_eq!(is_non_dominated_batch(&pb, true, true), expected);
        // Large enough to run in parallel.
        let big: Vec<Matrix> = (0..64)
            .map(|k| Matrix::from_fn(80, 3, |i, j| ((i * 7 + j * 13 + k * 5) % 11) as f64))
            .collect();
        const { assert!(64 * 80 * 80 * 3 >= PARALLEL_WORK) };
        let pb = PointBatch::from_matrices(&big).unwrap();
        let expected: Vec<bool> = big
            .iter()
            .flat_map(|y| is_non_dominated(y, false, false))
            .collect();
        assert_eq!(is_non_dominated_batch(&pb, false, false), expected);
    }
}
