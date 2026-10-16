//! The index computations of the sampling strategies of
//! `botorch.generation.sampling` (`MaxPosteriorSampling`,
//! `ConstrainedMaxPosteriorSampling`, `BoltzmannSampling`); the posterior
//! samples and acquisition values come from the models and acquisition
//! functions.

use crate::error::{BotorchError, Result};
use crate::generation::gen::torch_argmax;

/// Indices of the `k` largest values in descending order with torch's `topk`
/// semantics (NaN is the largest value; ties keep the index order).
fn topk_indices(values: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        let (va, vb) = (values[a], values[b]);
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
    idx.truncate(k);
    idx
}

/// BoTorch's `_flip_sub_unique`: the first `k` distinct elements of `x`,
/// traversing `x` from the back.
fn flip_sub_unique(x: &[usize], k: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(k);
    for &v in x.iter().rev() {
        if !out.contains(&v) {
            out.push(v);
        }
        if out.len() >= k {
            break;
        }
    }
    out
}

/// The selected indices of `MaxPosteriorSampling.maximize_samples` for
/// objective values `obj` of shape `num_samples × batch × n` (row-major):
/// with replacement, the maximizer of every sample; without replacement,
/// BoTorch's deduplication of the top `num_samples` maximizers (the lower
/// triangle of the top-k index matrix traversed from the back). Returns
/// `batch × num_samples` indices (row-major) into the `n` points.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for inconsistent sizes or (without
/// replacement) `num_samples > n`.
pub fn maximize_samples(
    obj: &[f64],
    num_samples: usize,
    batch: usize,
    n: usize,
    replacement: bool,
) -> Result<Vec<usize>> {
    if obj.len() != num_samples * batch * n || n == 0 {
        return Err(BotorchError::InvalidArgument(format!(
            "expected {num_samples} x {batch} x {n} objective values, got {}",
            obj.len()
        )));
    }
    let row = |s: usize, b: usize| &obj[(s * batch + b) * n..(s * batch + b + 1) * n];
    let mut out = vec![0usize; batch * num_samples];
    if replacement {
        for s in 0..num_samples {
            for b in 0..batch {
                out[b * num_samples + s] = torch_argmax(row(s, b));
            }
        }
        return Ok(out);
    }
    if num_samples > n {
        return Err(BotorchError::InvalidArgument(format!(
            "selected index k out of range: cannot draw {num_samples} distinct points from {n}"
        )));
    }
    for b in 0..batch {
        let top: Vec<Vec<usize>> = (0..num_samples)
            .map(|s| topk_indices(row(s, b), num_samples))
            .collect();
        // torch.tril_indices: row-major over the lower triangle.
        let mut sub = Vec::with_capacity(num_samples * (num_samples + 1) / 2);
        for (r, t) in top.iter().enumerate() {
            sub.extend_from_slice(&t[..=r]);
        }
        let unique = flip_sub_unique(&sub, num_samples);
        if unique.len() != num_samples {
            return Err(BotorchError::Runtime(
                "could not find enough distinct maximizers".into(),
            ));
        }
        out[b * num_samples..(b + 1) * num_samples].copy_from_slice(&unique);
    }
    Ok(out)
}

/// The scores of `ConstrainedMaxPosteriorSampling` (BoTorch's
/// `_convert_samples_to_scores`) for objective samples `y` (`s × b × n`)
/// and constraint samples `c` (`s × b × n × k`, feasible if `c <= 0`): the
/// objective value of feasible points and `-inf` for infeasible ones, or,
/// for realizations without a feasible point, the negative total violation.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for inconsistent sizes.
pub fn constrained_scores(
    y: &[f64],
    c: &[f64],
    rows: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f64>> {
    if y.len() != rows * n || c.len() != rows * n * k {
        return Err(BotorchError::InvalidArgument(
            "inconsistent sizes of the objective and constraint samples".into(),
        ));
    }
    let mut scores = y.to_vec();
    for r in 0..rows {
        let feasible: Vec<bool> = (0..n)
            .map(|i| (0..k).all(|j| c[(r * n + i) * k + j] <= 0.0))
            .collect();
        if feasible.iter().any(|&f| f) {
            for i in 0..n {
                if !feasible[i] {
                    scores[r * n + i] = f64::NEG_INFINITY;
                }
            }
        } else {
            for i in 0..n {
                let violation: f64 = (0..k).map(|j| c[(r * n + i) * k + j].max(0.0)).sum();
                scores[r * n + i] = -violation;
            }
        }
    }
    Ok(scores)
}

/// Boltzmann weights `exp(eta · standardize(values))` (BoTorch's
/// `boltzmann_sample` before the multinomial draw): the values are
/// standardized with the unbiased standard deviation (replaced by 1 below
/// `1e-9`) and `eta` is multiplied by `temp_decrease` while a weight is
/// infinite. Returns the weights and the final `eta`.
pub fn boltzmann_weights(values: &[f64], eta: f64, temp_decrease: f64) -> (Vec<f64>, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let std = (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt();
    let std = if std >= 1e-9 { std } else { 1.0 };
    let z: Vec<f64> = values.iter().map(|v| (v - mean) / std).collect();
    let mut eta = eta;
    let mut w: Vec<f64> = z.iter().map(|v| (eta * v).exp()).collect();
    while w.iter().any(|v| v.is_infinite()) {
        eta *= temp_decrease;
        w = z.iter().map(|v| (eta * v).exp()).collect();
    }
    (w, eta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_sub_unique_example() {
        let x = [1, 6, 4, 3, 6, 3];
        assert_eq!(flip_sub_unique(&x, 3), vec![3, 6, 4]);
        assert_eq!(flip_sub_unique(&x, 4), vec![3, 6, 4, 1]);
        assert_eq!(flip_sub_unique(&x, 10), vec![3, 6, 4, 1]);
    }

    #[test]
    fn maximize_with_and_without_replacement() {
        // 3 samples over 4 points; sample 0 and 1 share the maximizer.
        let obj = [
            0.0, 3.0, 1.0, 2.0, //
            0.0, 5.0, 4.0, 1.0, //
            9.0, 1.0, 2.0, 3.0,
        ];
        assert_eq!(
            maximize_samples(&obj, 3, 1, 4, true).unwrap(),
            vec![1, 1, 0]
        );
        let idx = maximize_samples(&obj, 3, 1, 4, false).unwrap();
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
        assert!(maximize_samples(&obj, 5, 1, 4, false).is_err());
    }

    #[test]
    fn constrained_scores_cases() {
        let y = [1.0, 2.0, 3.0, 4.0];
        // Row 0: point 1 feasible; row 1: none feasible.
        let c = [1.0, -1.0, 0.5, 2.0];
        let s = constrained_scores(&y, &c, 2, 2, 1).unwrap();
        assert_eq!(s, vec![f64::NEG_INFINITY, 2.0, -0.5, -2.0]);
    }

    #[test]
    fn boltzmann_weights_standardize() {
        let (w, eta) = boltzmann_weights(&[0.0, 1.0, 2.0], 1.0, 0.5);
        assert_eq!(eta, 1.0);
        assert!((w[1] - 1.0).abs() < 1e-15);
        let (w, _) = boltzmann_weights(&[5.0, 5.0], 1.0, 0.5);
        assert_eq!(w, vec![1.0, 1.0]);
    }
}
