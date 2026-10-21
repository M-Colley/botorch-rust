//! Utilities for acquisition functions (`botorch.acquisition.utils`):
//! [`prune_inferior_points`] (with MC objectives and outcome constraints),
//! [`compute_best_feasible_objective`] and [`get_infeasible_cost`].

use std::sync::Arc;

use crate::acquisition::check_unbatched;
use crate::acquisition::mc_objective::{McObjective, McSamples};
use crate::acquisition::monte_carlo::JointSamples;
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::utils::objective::{
    compute_feasibility_indicator, compute_smoothed_feasibility_indicator, McConstraint,
    OutcomeConstraints,
};
use crate::utils::sampling::draw_sobol_normal_samples;
use crate::Matrix;

/// Options of [`prune_inferior_points`].
#[derive(Clone, Debug)]
pub struct PruneOptions {
    /// Number of posterior samples used to estimate the probabilities of being
    /// the best point (BoTorch default 2048).
    pub num_samples: usize,
    /// Maximum fraction of points to retain, in `(0, 1]` (default 1).
    pub max_frac: f64,
    /// Seed of the Sobol normal base samples; `None` draws a random seed (as
    /// BoTorch does).
    pub seed: Option<u64>,
    /// Explicit base samples (`num_samples × n·m` for `n` points and `m`
    /// model outputs, BoTorch's layout), overriding `seed`.
    pub base_samples: Option<Matrix>,
    /// The MC objective under which the points are compared (required for
    /// multi-output models without a posterior transform).
    pub objective: Option<Arc<dyn McObjective>>,
    /// Outcome constraints: infeasible points are ranked below all feasible
    /// ones.
    pub constraints: Option<Vec<Arc<dyn McConstraint>>>,
}

impl Default for PruneOptions {
    fn default() -> Self {
        Self {
            num_samples: 2048,
            max_frac: 1.0,
            seed: None,
            base_samples: None,
            objective: None,
            constraints: None,
        }
    }
}

/// Salt of the seed of the second draw of base samples when no point is
/// feasible (BoTorch draws new samples there).
const REDRAW_SALT: u64 = 0x7072_756e_655f_6662;

/// Indices (into the rows of `x`) of the points that have a non-zero empirical
/// probability of being the best point under the model's joint posterior and
/// the objective, in increasing order (BoTorch's `prune_inferior_points`).
///
/// For every posterior sample the maximizing point (the first one on ties,
/// like `torch.argmax`) is recorded. With outcome constraints, points that
/// are infeasible in a sample get a value below the smallest objective value
/// of all samples; if no point is feasible in any sample, the points are
/// ranked by their smoothed log probability of feasibility (`eta = 1e-3`)
/// under a new set of samples (the same base samples when they were given
/// explicitly, as with BoTorch's `sampler`). If more than
/// `ceil(max_frac · n)` points are maximizers, the most frequent ones are
/// kept (in order of decreasing frequency, ties in increasing index order).
/// BoTorch's implementation indexes `X` with positions into its array of
/// unique maximizers in that case, which selects wrong points; this function
/// implements the documented behavior. With the default `max_frac = 1` both
/// coincide.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for an empty `x` or `max_frac` outside
/// `(0, 1]`, [`BotorchError::Unsupported`] for multi-output models without an
/// objective, [`BotorchError::DimensionMismatch`] for base samples of the
/// wrong shape, and errors of the model, the objective and the constraints.
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
    check_unbatched(model)?;
    let m = model.num_outputs();
    if options.objective.is_none() && m != 1 {
        return Err(BotorchError::Unsupported(
            "Models with multiple outputs require an objective or posterior_transform.".into(),
        ));
    }
    let max_points = (options.max_frac * n as f64).ceil() as usize;
    // The posterior points (`n·n_w` with a one-to-many input transform).
    let np = n * model.points_per_input();
    if options.constraints.as_ref().is_some_and(|c| !c.is_empty()) && np != n {
        return Err(BotorchError::Unsupported(
            "Outcome constraints are not supported with one-to-many input transforms.".into(),
        ));
    }
    let base = match &options.base_samples {
        Some(z) => {
            if z.ncols() != np * m || z.nrows() == 0 {
                return Err(BotorchError::DimensionMismatch(format!(
                    "base samples must have {} columns ({np} points x {m} outputs), got {}x{}",
                    np * m,
                    z.nrows(),
                    z.ncols()
                )));
            }
            z.clone()
        }
        None => draw_sobol_normal_samples(np * m, options.num_samples, options.seed)?,
    };
    let js = JointSamples::with_base_samples(model, x.clone(), Arc::new(base))?;
    let samples = js.samples(0..np);
    let mut obj = match &options.objective {
        Some(o) => {
            let v = o.evaluate(&samples, Some(x), false)?.values;
            crate::acquisition::mc_objective::check_objective_shape(&v, n, samples.num_samples())?;
            v
        }
        None => {
            if np != n {
                return Err(crate::acquisition::mc_objective::q_batch_mismatch(np, n));
            }
            samples.data().clone()
        }
    };
    if let Some(constraints) = options.constraints.as_ref().filter(|c| !c.is_empty()) {
        let feasible = compute_feasibility_indicator(constraints, &samples)?;
        if !feasible.iter().any(|&f| f) {
            // No point is feasible: rank by the probability of feasibility.
            let resampled = if options.base_samples.is_some() {
                samples
            } else {
                let seed = options.seed.map(|s| s ^ REDRAW_SALT);
                let z = draw_sobol_normal_samples(n * m, options.num_samples, seed)?;
                let js = JointSamples::with_base_samples(model, x.clone(), Arc::new(z))?;
                js.samples(0..n)
            };
            let oc = OutcomeConstraints::uniform(constraints.clone(), 1e-3)?;
            let fat = vec![Some(false); constraints.len()];
            obj =
                compute_smoothed_feasibility_indicator(&oc, &fat, &resampled, true, false)?.values;
        } else if feasible.iter().any(|&f| !f) {
            // torch.min propagates NaN.
            let min = obj.iter().copied().fold(f64::INFINITY, |a, v| {
                if v.is_nan() || a.is_nan() {
                    f64::NAN
                } else {
                    a.min(v)
                }
            });
            for c in 0..obj.ncols() {
                for p in 0..n {
                    if !feasible[(p, c % feasible.ncols())] {
                        obj[(p, c)] = min - 1.0;
                    }
                }
            }
        }
    }
    let mut counts = vec![0usize; n];
    for c in 0..obj.ncols() {
        let mut best = 0;
        for i in 1..n {
            let (v, b) = (obj[(i, c)], obj[(best, c)]);
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

/// NaN-propagating maximum (`torch.amax`).
fn amax(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(f64::NEG_INFINITY, |m, v| {
        if v.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.max(v)
        }
    })
}

/// The best feasible objective value of every sample (BoTorch's
/// `compute_best_feasible_objective`): for objective values `obj` (`n × S`)
/// of samples of `n` points, the largest value of a point that satisfies all
/// `constraints` in that sample (the largest value overall without
/// constraints). If some sample has no feasible point, the values of
/// infeasible points are replaced by `infeasible_value()` (BoTorch: a lower
/// bound of the objective, see [`estimate_objective_lower_bound`]).
///
/// # Errors
/// Errors of the constraints and of `infeasible_value`.
pub fn compute_best_feasible_objective<F>(
    samples: &McSamples,
    obj: &Matrix,
    constraints: Option<&[Arc<dyn McConstraint>]>,
    infeasible_value: F,
) -> Result<Vec<f64>>
where
    F: FnOnce() -> Result<f64>,
{
    let (n, s) = obj.shape();
    let Some(constraints) = constraints else {
        return Ok((0..s)
            .map(|c| amax(obj.column(c).iter().copied()))
            .collect());
    };
    let feasible = compute_feasibility_indicator(constraints, samples)?;
    crate::acquisition::mc_objective::check_objective_shape(
        obj,
        feasible.nrows(),
        feasible.ncols(),
    )?;
    let all_have_feasible = (0..s).all(|c| (0..n).any(|p| feasible[(p, c % feasible.ncols())]));
    let fill = if all_have_feasible {
        f64::NEG_INFINITY
    } else {
        infeasible_value()?
    };
    Ok((0..s)
        .map(|c| {
            amax((0..n).map(|p| {
                if feasible[(p, c % feasible.ncols())] {
                    obj[(p, c)]
                } else {
                    fill
                }
            }))
        })
        .collect())
}

/// The infeasible cost `M ≥ 0` of a model and objective at the points `x`
/// (BoTorch's `get_infeasible_cost`): `-min(0, min_x min(obj(μ - 6σ),
/// obj(μ + 6σ)))` over the marginal posteriors at `x`, so that `-M` is below
/// the objective almost everywhere.
///
/// # Errors
/// Errors of the model and the objective.
pub fn get_infeasible_cost(
    model: &dyn Model,
    objective: Option<&dyn McObjective>,
    x: &Matrix,
) -> Result<f64> {
    let n = x.nrows();
    let m = model.num_outputs();
    let pp = model.point_posteriors(x)?;
    let shifted = |sign: f64| -> Result<Matrix> {
        let data = Matrix::from_fn(m * n, 1, |r, _| {
            let (t, p) = (r / n, r % n);
            let var = pp
                .variance(p, t)
                .max(crate::posterior::MIN_VARIANCE)
                .max(0.0);
            pp.mean[(p, t)] + sign * (6.0 * var.sqrt())
        });
        let y = McSamples::new(data, n, m)?;
        match objective {
            Some(o) => Ok(o.evaluate(&y, Some(x), false)?.values),
            None => {
                if m != 1 {
                    return Err(BotorchError::Unsupported(
                        crate::acquisition::monte_carlo::MULTI_OUTPUT_MESSAGE.into(),
                    ));
                }
                Ok(y.into_data())
            }
        }
    };
    let lo = shifted(-1.0)?;
    let hi = shifted(1.0)?;
    let mut lb = f64::INFINITY;
    for (a, b) in lo.iter().zip(hi.iter()) {
        // torch.stack(...).min(dim=0) then min over the points (NaN propagates).
        let v = if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(*b)
        };
        lb = if v.is_nan() || lb.is_nan() {
            f64::NAN
        } else {
            lb.min(v)
        };
    }
    Ok(-(lb.min(0.0)))
}

/// A lower bound of the objective (BoTorch's `_estimate_objective_lower_bound`
/// evaluated at the given points): `-get_infeasible_cost(model, objective,
/// x)`. BoTorch draws 32 uniformly random points in the bounding box of the
/// baseline points, expanded by 10% in each direction.
///
/// # Errors
/// See [`get_infeasible_cost`].
pub fn estimate_objective_lower_bound(
    model: &dyn Model,
    objective: Option<&dyn McObjective>,
    x: &Matrix,
) -> Result<f64> {
    Ok(-get_infeasible_cost(model, objective, x)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::CovarianceRoot;
    use crate::models::SingleTaskGP;
    use crate::utils::objective::LinearOutcomeConstraint;
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

    #[test]
    fn constraints_rank_infeasible_points_last() {
        let (gp, x) = model_on_a_line();
        let z = draw_sobol_normal_samples(8, 256, Some(5)).unwrap();
        // Feasible iff y <= 0.6: the maximizers are restricted to points
        // whose samples stay below 0.6.
        let cons: Vec<Arc<dyn McConstraint>> = vec![Arc::new(
            LinearOutcomeConstraint::new(crate::Vector::from_vec(vec![1.0]), -0.6).unwrap(),
        )];
        let options = PruneOptions {
            num_samples: 256,
            base_samples: Some(z.clone()),
            constraints: Some(cons.clone()),
            ..Default::default()
        };
        let idx = prune_inferior_points_indices(&gp, &x, &options).unwrap();
        let post = gp.posterior(&x, false).unwrap();
        let l = CovarianceRoot::new(&post.covariance).unwrap().to_matrix();
        let mut expected = [0usize; 8];
        for s in 0..z.nrows() {
            let ys = &post.mean + &l * z.row(s).transpose();
            let min = ys.min();
            let masked = ys.map(|v| if v <= 0.6 { v } else { min - 1.0 });
            expected[masked.argmax().0] += 1;
        }
        let expected: Vec<usize> = (0..8).filter(|&i| expected[i] > 0).collect();
        assert_eq!(idx, expected);
        // Best feasible objective per sample.
        let js = JointSamples::with_base_samples(&gp, x.clone(), Arc::new(z)).unwrap();
        let samples = js.samples(0..8);
        let best =
            compute_best_feasible_objective(&samples, samples.data(), Some(&cons), || Ok(-7.0))
                .unwrap();
        let data = samples.data();
        let some_all_infeasible =
            (0..data.ncols()).any(|c| data.column(c).iter().all(|&v| v > 0.6));
        let fill = if some_all_infeasible {
            -7.0
        } else {
            f64::NEG_INFINITY
        };
        for (c, b) in best.iter().enumerate() {
            let e = data
                .column(c)
                .iter()
                .map(|&v| if v <= 0.6 { v } else { fill })
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(*b, e, "sample {c}");
        }
        // The infeasible cost is non-negative and bounds the objective.
        let m_cost = get_infeasible_cost(&gp, None, &x).unwrap();
        assert!(m_cost >= 0.0);
        assert_eq!(
            estimate_objective_lower_bound(&gp, None, &x).unwrap(),
            -m_cost
        );
    }
}
