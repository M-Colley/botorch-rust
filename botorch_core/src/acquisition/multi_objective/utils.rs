//! Utilities of the multi-objective acquisition functions
//! (`botorch.acquisition.multi_objective.utils`):
//! [`get_default_partitioning_alpha`] and
//! [`prune_inferior_points_multi_objective`].
//!
//! `compute_sample_box_decomposition`, `random_search_optimizer` and
//! `sample_optimal_points` (used by the information-theoretic acquisition
//! functions) need pathwise posterior samples, which are not ported.

use std::sync::Arc;

use crate::acquisition::check_unbatched;
use crate::acquisition::mc_objective::McSamples;
use crate::acquisition::monte_carlo::JointSamples;
use crate::acquisition::multi_objective::objective::{check_values, McMultiOutputObjective};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::utils::multi_objective::pareto::is_non_dominated_batch;
use crate::utils::multi_objective::PointBatch;
use crate::utils::objective::{compute_feasibility_indicator, McConstraint};
use crate::utils::sampling::draw_sobol_normal_samples;
use crate::Matrix;

/// BoTorch's warning of [`get_default_partitioning_alpha`] for more than six
/// objectives.
pub const MANY_OBJECTIVES_WARNING: &str = "EHVI works best for less than 7 objectives.";

/// The approximation level `alpha` of the box decompositions for
/// `num_objectives` objectives (BoTorch's `get_default_partitioning_alpha`):
/// `0` (exact, `FastNondominatedPartitioning`) for up to four objectives,
/// `1e-3` for five, `1e-2` for six or more. The second value is BoTorch's
/// warning (a `BotorchWarning`) for more than six objectives.
pub fn get_default_partitioning_alpha(num_objectives: usize) -> (f64, Option<&'static str>) {
    if num_objectives <= 4 {
        return (0.0, None);
    }
    let warning = (num_objectives > 6).then_some(MANY_OBJECTIVES_WARNING);
    (if num_objectives >= 6 { 1e-2 } else { 1e-3 }, warning)
}

/// Options of [`prune_inferior_points_multi_objective`].
#[derive(Clone, Debug)]
pub struct PruneMultiObjectiveOptions {
    /// Number of posterior samples (BoTorch default 2048).
    pub num_samples: usize,
    /// Maximum fraction of points to retain, in `(0, 1]` (default 1).
    pub max_frac: f64,
    /// Seed of the Sobol normal base samples; `None` draws a random seed (as
    /// BoTorch does).
    pub seed: Option<u64>,
    /// Explicit base samples (`num_samples × n·m`, BoTorch's layout),
    /// overriding `seed`.
    pub base_samples: Option<Matrix>,
    /// The multi-output objective (default: all model outputs).
    pub objective: Option<Arc<dyn McMultiOutputObjective>>,
    /// Outcome constraints (feasible iff `<= 0`): infeasible points of a
    /// sample are set to the reference point.
    pub constraints: Option<Vec<Arc<dyn McConstraint>>>,
}

impl Default for PruneMultiObjectiveOptions {
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

/// Indices (into the rows of `x`, increasing) of the points that have a
/// non-zero empirical probability of being Pareto optimal, feasible and
/// better than `ref_point` (BoTorch's `prune_inferior_points_multi_objective`).
///
/// For every posterior sample, the objective values of infeasible points are
/// set to the reference point; a point counts in a sample if it is not
/// dominated (`deduplicate=False`) and better than the reference point in
/// every objective. If more than `ceil(max_frac · n)` points have a non-zero
/// probability, the most probable ones are kept (ties in index order).
///
/// With a one-to-many input transform and a risk-measure objective, the
/// objective may return several points per input (e.g. `MVaR`); as in
/// BoTorch, the probabilities are computed per objective point and an input
/// is kept if one of its objective points is (`idcs // effective_n_w`).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for an empty `x` or `max_frac` outside
/// `(0, 1]`, [`BotorchError::Unsupported`] for batched models,
/// [`BotorchError::DimensionMismatch`] for base samples of the wrong shape or
/// a reference point whose length differs from the number of objectives, and
/// errors of the model, the objective and the constraints.
pub fn prune_inferior_points_multi_objective_indices(
    model: &dyn Model,
    x: &Matrix,
    ref_point: &[f64],
    options: &PruneMultiObjectiveOptions,
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
    let obj: McSamples = match &options.objective {
        Some(o) => {
            let v = o.evaluate(&samples, Some(x), false)?.values;
            check_values(&v, &samples, Some(x), o.verify_output_shape())?;
            v
        }
        None => samples.clone(),
    };
    // The objective points (several per input for set-valued objectives).
    let no = obj.num_points();
    let mo = obj.num_outputs();
    if ref_point.len() != mo {
        return Err(BotorchError::DimensionMismatch(format!(
            "The size of tensor a ({mo}) must match the size of tensor b ({}) at non-singleton \
             dimension 1",
            ref_point.len()
        )));
    }
    let s = obj.num_samples();
    let feasible = match options.constraints.as_ref().filter(|c| !c.is_empty()) {
        Some(c) => Some(compute_feasibility_indicator(c, &samples)?),
        None => None,
    };
    // Row-major S x n x m' values, infeasible points set to the reference point.
    let mut values = Vec::with_capacity(s * no * mo);
    for c in 0..s {
        for p in 0..no {
            if feasible.as_ref().is_some_and(|f| !f[(p, c)]) {
                values.extend_from_slice(ref_point);
            } else {
                values.extend((0..mo).map(|t| obj.value(c, p, t)));
            }
        }
    }
    let batch = PointBatch::new(vec![s], no, mo, values)?;
    let mask = is_non_dominated_batch(&batch, true, false);
    let mut counts = vec![0usize; no];
    for c in 0..s {
        for (p, count) in counts.iter_mut().enumerate() {
            if mask[c * no + p] && batch.point(c, p).iter().zip(ref_point).all(|(v, r)| v > r) {
                *count += 1;
            }
        }
    }
    let mut idcs: Vec<usize> = (0..no).filter(|&i| counts[i] > 0).collect();
    if idcs.len() > max_points {
        // torch.sort(probs, stable=True, descending=True), then the first
        // max_points indices.
        let mut order: Vec<usize> = (0..no).collect();
        order.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
        idcs = order[..max_points].to_vec();
    }
    // `(idcs / effective_n_w).long().unique()`: the inputs, in increasing
    // order.
    let effective_n_w = (no / n).max(1);
    let mut idcs: Vec<usize> = idcs.into_iter().map(|i| i / effective_n_w).collect();
    idcs.sort_unstable();
    idcs.dedup();
    Ok(idcs)
}

/// The points of `x` with a non-zero empirical probability of being Pareto
/// optimal (see [`prune_inferior_points_multi_objective_indices`]).
///
/// # Errors
/// See [`prune_inferior_points_multi_objective_indices`].
pub fn prune_inferior_points_multi_objective(
    model: &dyn Model,
    x: &Matrix,
    ref_point: &[f64],
    options: &PruneMultiObjectiveOptions,
) -> Result<Matrix> {
    let idcs = prune_inferior_points_multi_objective_indices(model, x, ref_point, options)?;
    Ok(Matrix::from_fn(idcs.len(), x.ncols(), |i, j| {
        x[(idcs[i], j)]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BatchedMultiOutputGP, SingleTaskGPOptions};
    use crate::utils::objective::LinearOutcomeConstraint;

    #[test]
    fn default_alpha() {
        assert_eq!(get_default_partitioning_alpha(2), (0.0, None));
        assert_eq!(get_default_partitioning_alpha(4), (0.0, None));
        assert_eq!(get_default_partitioning_alpha(5), (1e-3, None));
        assert_eq!(get_default_partitioning_alpha(6), (1e-2, None));
        assert_eq!(
            get_default_partitioning_alpha(7),
            (1e-2, Some(MANY_OBJECTIVES_WARNING))
        );
    }

    #[test]
    fn pruning_keeps_pareto_candidates() {
        let n = 8;
        let x = Matrix::from_fn(n, 1, |i, _| i as f64 / 7.0);
        let y = Matrix::from_fn(n, 2, |i, t| {
            let v = x[(i, 0)];
            if t == 0 {
                (3.0 * v).sin()
            } else {
                (2.0 * v).cos() - 0.3 * v
            }
        });
        let mut gp =
            BatchedMultiOutputGP::multi_output(x.clone(), y, SingleTaskGPOptions::default())
                .unwrap();
        gp.set_raw_parameters(&[0.01, 0.01, 0.0, 0.0, 0.3, 0.3])
            .unwrap();
        let z = draw_sobol_normal_samples(2 * n, 256, Some(1)).unwrap();
        let options = PruneMultiObjectiveOptions {
            base_samples: Some(z.clone()),
            num_samples: 256,
            ..Default::default()
        };
        let all = prune_inferior_points_multi_objective_indices(&gp, &x, &[-2.0, -2.0], &options)
            .unwrap();
        assert!(!all.is_empty() && all.len() < n, "{all:?}");
        assert!(all.windows(2).all(|w| w[0] < w[1]));
        // A reference point above every sample prunes everything.
        let none = prune_inferior_points_multi_objective_indices(&gp, &x, &[10.0, 10.0], &options)
            .unwrap();
        assert!(none.is_empty());
        // max_frac keeps the most probable points.
        let few = prune_inferior_points_multi_objective_indices(
            &gp,
            &x,
            &[-2.0, -2.0],
            &PruneMultiObjectiveOptions {
                max_frac: 0.25,
                ..options.clone()
            },
        )
        .unwrap();
        assert_eq!(few.len(), 2.min(all.len()));
        assert!(few.iter().all(|i| all.contains(i)));
        // An unsatisfiable constraint makes every point infeasible.
        let cons: Vec<Arc<dyn McConstraint>> = vec![Arc::new(
            LinearOutcomeConstraint::new(crate::Vector::from_vec(vec![0.0, 0.0]), 1.0).unwrap(),
        )];
        let infeasible = prune_inferior_points_multi_objective_indices(
            &gp,
            &x,
            &[-2.0, -2.0],
            &PruneMultiObjectiveOptions {
                constraints: Some(cons),
                ..options.clone()
            },
        )
        .unwrap();
        assert!(infeasible.is_empty());
        assert!(prune_inferior_points_multi_objective_indices(&gp, &x, &[0.0], &options).is_err());
    }
}
