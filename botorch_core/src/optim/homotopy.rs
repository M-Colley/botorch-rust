//! `prune_candidates` of `botorch.optim.optimize_homotopy` (the homotopy
//! loop itself lives in the Python layer, where the homotopy parameters
//! are).

use crate::error::{BotorchError, Result};
use crate::Matrix;

/// Prunes candidates (`n × d`) that are within `prune_tolerance` (Euclidean
/// distance) of a better one (BoTorch's `prune_candidates`): the candidates
/// are sorted by descending value and each is kept if its distance to all
/// kept candidates exceeds the tolerance. Ties keep the index order; NaN
/// values sort first, like torch's `argsort(descending=True)`.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for a value count that differs from
/// the number of candidates or a negative tolerance.
pub fn prune_candidates(
    candidates: &Matrix,
    acq_values: &[f64],
    prune_tolerance: f64,
) -> Result<Matrix> {
    let n = candidates.nrows();
    if acq_values.len() != n {
        return Err(BotorchError::InvalidArgument(
            "`acq_values` must be of size `n`.".into(),
        ));
    }
    if prune_tolerance < 0.0 || prune_tolerance.is_nan() {
        return Err(BotorchError::InvalidArgument(
            "`prune_tolerance` must be >= 0.".into(),
        ));
    }
    let d = candidates.ncols();
    if n == 0 {
        return Ok(Matrix::zeros(0, d));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let (va, vb) = (acq_values[a], acq_values[b]);
        match (va.is_nan(), vb.is_nan()) {
            (true, true) => a.cmp(&b),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => vb
                .partial_cmp(&va)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b)),
        }
    });
    let mut kept: Vec<usize> = vec![order[0]];
    for &i in &order[1..] {
        let min_dist = kept
            .iter()
            .map(|&k| {
                (0..d)
                    .map(|j| (candidates[(i, j)] - candidates[(k, j)]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        if min_dist > prune_tolerance {
            kept.push(i);
        }
    }
    Ok(Matrix::from_fn(kept.len(), d, |r, c| {
        candidates[(kept[r], c)]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prunes_close_candidates() {
        let c = Matrix::from_row_slice(4, 1, &[0.0, 1e-5, 0.5, 0.50001]);
        let out = prune_candidates(&c, &[1.0, 2.0, 0.5, 0.7], 1e-4).unwrap();
        assert_eq!(out.as_slice(), &[1e-5, 0.50001]);
        assert!(prune_candidates(&c, &[1.0], 1e-4).is_err());
        assert!(prune_candidates(&c, &[1.0; 4], -1.0).is_err());
    }
}
