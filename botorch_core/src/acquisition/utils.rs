//! Utilities for acquisition functions (`botorch.acquisition.utils`):
//! [`prune_inferior_points`].

use crate::error::{BotorchError, Result};
use crate::linalg::CovarianceRoot;
use crate::models::Model;
use crate::utils::sampling::draw_sobol_normal_samples;
use crate::Matrix;

/// Options of [`prune_inferior_points`].
#[derive(Clone, Debug, PartialEq)]
pub struct PruneOptions {
    /// Number of posterior samples used to estimate the probabilities of being
    /// the best point (BoTorch default 2048).
    pub num_samples: usize,
    /// Maximum fraction of points to retain, in `(0, 1]` (default 1).
    pub max_frac: f64,
    /// Seed of the Sobol normal base samples; `None` draws a random seed (as
    /// BoTorch does).
    pub seed: Option<u64>,
    /// Explicit base samples (`num_samples × n`), overriding `seed`.
    pub base_samples: Option<Matrix>,
}

impl Default for PruneOptions {
    fn default() -> Self {
        Self {
            num_samples: 2048,
            max_frac: 1.0,
            seed: None,
            base_samples: None,
        }
    }
}

/// Indices (into the rows of `x`) of the points that have a non-zero empirical
/// probability of being the best point under the model's joint posterior,
/// in increasing order (BoTorch's `prune_inferior_points`).
///
/// For every posterior sample the maximizing point (the first one on ties,
/// like `torch.argmax`) is recorded. If more than `ceil(max_frac · n)` points
/// are maximizers, the most frequent ones are kept (in order of decreasing
/// frequency, ties in increasing index order). BoTorch's implementation
/// indexes `X` with positions into its array of unique maximizers in that
/// case, which selects wrong points; this function implements the documented
/// behavior. With the default `max_frac = 1` both coincide.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for an empty `x` or `max_frac` outside
/// `(0, 1]`, [`BotorchError::Unsupported`] for multi-output models,
/// [`BotorchError::DimensionMismatch`] for base samples of the wrong shape,
/// and model errors.
pub fn prune_inferior_points_indices(
    model: &dyn Model,
    x: &Matrix,
    options: &PruneOptions,
) -> Result<Vec<usize>> {
    let n = x.nrows();
    if n == 0 {
        return Err(BotorchError::InvalidArgument(
            "X must have at least one point.".into(),
        ));
    }
    if !(options.max_frac > 0.0 && options.max_frac <= 1.0) {
        return Err(BotorchError::InvalidArgument(format!(
            "max_frac must take values in (0, 1], is {}",
            options.max_frac
        )));
    }
    if model.num_outputs() != 1 {
        return Err(BotorchError::Unsupported(
            "prune_inferior_points requires a single-output model or an objective".into(),
        ));
    }
    let max_points = (options.max_frac * n as f64).ceil() as usize;
    let post = model.posterior(x, false)?;
    let root = CovarianceRoot::new(&post.covariance)?.to_matrix();
    let base = match &options.base_samples {
        Some(z) => {
            if z.ncols() != n || z.nrows() == 0 {
                return Err(BotorchError::DimensionMismatch(format!(
                    "base samples must have {n} columns, got {}x{}",
                    z.nrows(),
                    z.ncols()
                )));
            }
            z.clone()
        }
        None => draw_sobol_normal_samples(n, options.num_samples, options.seed)?,
    };
    // Samples: points × MC samples.
    let mut y = root * base.transpose();
    for i in 0..n {
        let m = post.mean[i];
        y.row_mut(i).iter_mut().for_each(|v| *v += m);
    }
    let mut counts = vec![0usize; n];
    for c in 0..y.ncols() {
        let mut best = 0;
        for i in 1..n {
            let (v, b) = (y[(i, c)], y[(best, c)]);
            // torch.argmax treats NaN as the maximum.
            if !b.is_nan() && (v > b || v.is_nan()) {
                best = i;
            }
        }
        counts[best] += 1;
    }
    let mut idcs: Vec<usize> = (0..n).filter(|&i| counts[i] > 0).collect();
    if idcs.len() > max_points {
        idcs.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
        idcs.truncate(max_points);
    }
    Ok(idcs)
}

/// The points of `x` that have a non-zero empirical probability of being the
/// best point (see [`prune_inferior_points_indices`]).
///
/// # Errors
/// See [`prune_inferior_points_indices`].
pub fn prune_inferior_points(
    model: &dyn Model,
    x: &Matrix,
    options: &PruneOptions,
) -> Result<Matrix> {
    let idcs = prune_inferior_points_indices(model, x, options)?;
    Ok(Matrix::from_fn(idcs.len(), x.ncols(), |i, j| {
        x[(idcs[i], j)]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SingleTaskGP;
    use crate::SingleTaskGPOptions;

    fn model_on_a_line() -> (SingleTaskGP, Matrix) {
        let x = Matrix::from_fn(8, 1, |i, _| i as f64 / 7.0);
        let y = Matrix::from_fn(8, 1, |i, _| (3.0 * x[(i, 0)]).sin());
        let mut gp = SingleTaskGP::new(x.clone(), y, SingleTaskGPOptions::default()).unwrap();
        gp.set_raw_parameters(&[0.05, 0.0, 0.3]).unwrap();
        (gp, x)
    }

    /// Counts how often each point is the maximizer of a posterior sample.
    fn counts(gp: &SingleTaskGP, x: &Matrix, z: &Matrix) -> Vec<usize> {
        let post = gp.posterior(x, false).unwrap();
        let l = CovarianceRoot::new(&post.covariance).unwrap().to_matrix();
        let mut c = vec![0usize; x.nrows()];
        for s in 0..z.nrows() {
            let ys = &post.mean + &l * z.row(s).transpose();
            c[ys.argmax().0] += 1;
        }
        c
    }

    #[test]
    fn pruning_keeps_the_maximizers() {
        let (gp, x) = model_on_a_line();
        let z = draw_sobol_normal_samples(8, 512, Some(3)).unwrap();
        let c = counts(&gp, &x, &z);
        let options = PruneOptions {
            num_samples: 512,
            base_samples: Some(z.clone()),
            ..Default::default()
        };
        let all = prune_inferior_points_indices(&gp, &x, &options).unwrap();
        let nonzero: Vec<usize> = (0..8).filter(|&i| c[i] > 0).collect();
        assert_eq!(all, nonzero);
        assert!(all.len() > 2, "the test needs several maximizers");
        // max_frac = 0.25 keeps ceil(0.25 * 8) = 2 points: the most frequent ones.
        let options = PruneOptions {
            max_frac: 0.25,
            ..options
        };
        let top = prune_inferior_points_indices(&gp, &x, &options).unwrap();
        let mut by_count = nonzero.clone();
        by_count.sort_by(|&a, &b| c[b].cmp(&c[a]));
        assert_eq!(top, by_count[..2].to_vec());
        let pruned = prune_inferior_points(&gp, &x, &options).unwrap();
        assert_eq!(pruned.nrows(), 2);
        assert_eq!(pruned[(0, 0)], x[(top[0], 0)]);
        // Invalid arguments.
        let bad = PruneOptions {
            max_frac: 0.0,
            ..Default::default()
        };
        assert!(prune_inferior_points_indices(&gp, &x, &bad).is_err());
        assert!(
            prune_inferior_points_indices(&gp, &Matrix::zeros(0, 1), &PruneOptions::default())
                .is_err()
        );
    }
}
