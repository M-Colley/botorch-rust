//! Optimization over mixed discrete and continuous domains:
//! [`optimize_acqf_mixed`] (BoTorch's `botorch.optim.optimize.optimize_acqf_mixed`,
//! an enumeration of fixed-feature combinations) and
//! [`optimize_acqf_mixed_alternating`] (`botorch.optim.optimize_mixed`, which
//! alternates discrete local search and continuous L-BFGS-B steps from many
//! starting points), with the building blocks of `optimize_mixed`
//! ([`round_discrete_dims`], [`get_nearest_neighbors`],
//! [`get_categorical_neighbors`], [`get_spray_points`],
//! [`sample_feasible_points`], [`generate_starting_points`],
//! [`discrete_step`], [`continuous_step`]).

use std::collections::BTreeMap;

use rand::rngs::Xoshiro256PlusPlus;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::generation::gen::torch_argmax;
use crate::lbfgsb::LbfgsbStatus;
use crate::optim::discrete::lex_cmp;
use crate::optim::initializers::{apply_fixed_features, initialize_q_batch};
use crate::optim::optimize::{
    default_lbfgsb_options, gen_candidates_lbfgsb, optimize_acqf_detailed, OptimizeAcqfOptions,
    OptimizerStatus, PostProcessing,
};
use crate::optim::parameter_constraints::{
    evaluate_feasibility, ParameterConstraints, CONSTRAINT_TOLERANCE,
};
use crate::optim::slsqp::{gen_candidates_slsqp, project_to_feasible_space, SlsqpOptions};
use crate::utils::polytope::{sparse_to_dense_constraints, HitAndRunPolytopeSampler};
use crate::utils::sampling::{check_bounds, draw_sobol_samples, random_seed, SobolEngine};
use crate::Matrix;

/// Above this number of discrete combinations, `optimize_acqf_homotopy` (and
/// Ax) use [`optimize_acqf_mixed_alternating`] instead of
/// [`optimize_acqf_mixed`] (BoTorch's `ALTERNATING_OPTIMIZER_THRESHOLD`).
pub const ALTERNATING_OPTIMIZER_THRESHOLD: usize = 10;

/// Discrete dimensions (index → allowed values).
pub type DiscreteDims = BTreeMap<usize, Vec<f64>>;

/// Whether to use [`optimize_acqf_mixed_alternating`] for a mixed search
/// space (BoTorch's `should_use_mixed_alternating_optimizer`): if the
/// discrete and categorical dimensions have more than
/// [`ALTERNATING_OPTIMIZER_THRESHOLD`] combinations.
pub fn should_use_mixed_alternating_optimizer(
    discrete_dims: Option<&DiscreteDims>,
    cat_dims: Option<&DiscreteDims>,
) -> bool {
    if discrete_dims.is_none() && cat_dims.is_none() {
        return false;
    }
    let mut n_combos: usize = 1;
    for values in discrete_dims
        .into_iter()
        .chain(cat_dims)
        .flat_map(|m| m.values())
    {
        n_combos = n_combos.saturating_mul(values.len());
    }
    n_combos > ALTERNATING_OPTIMIZER_THRESHOLD
}

// ---------------------------------------------------------------------------
// optimize_acqf_mixed
// ---------------------------------------------------------------------------

/// The result of [`optimize_acqf_mixed`].
#[derive(Clone, Debug, PartialEq)]
pub struct MixedResult {
    /// The candidates (`q × d`).
    pub candidates: Matrix,
    /// The acquisition value: of the best restart for `q = 1`, the joint
    /// value of the candidates for `q > 1`.
    pub acq_value: f64,
    /// For `q = 1`: all restarts (`1 × d` candidates and values) of the best
    /// fixed features (BoTorch's `return_best_only=False` output).
    pub restarts: Vec<(Matrix, f64)>,
    /// Messages of BoTorch's warnings.
    pub warnings: Vec<String>,
}

/// Optimizes the acquisition function over a list of fixed-feature
/// combinations and returns the best candidates (BoTorch's
/// `optimize_acqf_mixed`): for `q = 1`, [`optimize_acqf_detailed`] runs for
/// every element of `fixed_features_list` (in parallel; combinations whose
/// candidate generation fails with a `CandidateGenerationError` are skipped
/// with BoTorch's warning) and the best restart over all combinations is
/// returned; for `q > 1` the candidates are generated sequentially, each
/// conditioned on the previous ones as pending points, and the joint value
/// is returned.
///
/// `options.fixed_features` is ignored (the elements of
/// `fixed_features_list` take its place). Unlike BoTorch, restarts that are
/// infeasible after the projection onto the constraints are never selected.
///
/// # Errors
/// [`BotorchError::Unsupported`] for inter-point constraints, one-shot
/// acquisition functions with `q > 1` and `return_best_only = false` with
/// `q > 1`; [`BotorchError::InvalidArgument`] for an empty list;
/// [`BotorchError::CandidateGeneration`] if all combinations fail; and the
/// errors of [`optimize_acqf_detailed`].
#[allow(clippy::too_many_arguments)]
pub fn optimize_acqf_mixed(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    q: usize,
    num_restarts: usize,
    fixed_features_list: &[BTreeMap<usize, f64>],
    raw_samples: usize,
    options: &OptimizeAcqfOptions,
    return_best_only: bool,
) -> Result<MixedResult> {
    options.constraints.validate_sequential()?;
    if !return_best_only && q > 1 {
        return Err(BotorchError::Unsupported(
            "`return_best_only=False` is only supported for q=1.".into(),
        ));
    }
    if fixed_features_list.is_empty() {
        return Err(BotorchError::InvalidArgument(
            "fixed_features_list must be non-empty.".into(),
        ));
    }
    if acq_function.as_one_shot().is_some() && q > 1 {
        return Err(BotorchError::Unsupported(
            "`OneShotAcquisitionFunction`s that do not implement `evaluate` are currently not \
             supported when `q > 1`. This is needed to compute the joint acquisition value."
                .into(),
        ));
    }
    let d = check_bounds(bounds)?;
    if q == 1 {
        let n_ff = fixed_features_list.len();
        let timeout = options.timeout_sec.map(|t| t / n_ff as f64);
        let results: Vec<Result<_>> = fixed_features_list
            .par_iter()
            .map(|ff| {
                let opts = OptimizeAcqfOptions {
                    fixed_features: ff.iter().map(|(&k, &v)| (k % d.max(1), v)).collect(),
                    timeout_sec: timeout,
                    ..options.clone()
                };
                optimize_acqf_detailed(acq_function, bounds, 1, num_restarts, raw_samples, &opts)
            })
            .collect();
        let mut ok = Vec::new();
        let mut failures = 0;
        for r in results {
            match r {
                Ok(res) => ok.push(res),
                Err(BotorchError::CandidateGeneration(_)) => failures += 1,
                Err(e) => return Err(e),
            }
        }
        if ok.is_empty() {
            return Err(BotorchError::CandidateGeneration(
                "Candidate generation failed for all `fixed_features`.".into(),
            ));
        }
        let mut warnings: Vec<String> = ok.iter().flat_map(|r| r.warnings.clone()).collect();
        if failures > 0 {
            warnings.push(format!(
                "Candidate generation failed for {failures} combinations of `fixed_features`. \
                 To suppress this warning, make sure all equality/inequality constraints can be \
                 satisfied by all `fixed_features` in `fixed_features_list`."
            ));
        }
        // The maximum over the restarts of every combination, then the best
        // combination (torch.max / torch.argmax semantics).
        let restart_values = |res: &crate::optim::OptimizeAcqfResult| -> Vec<f64> {
            if res.restarts.is_empty() {
                vec![res.acq_value]
            } else {
                res.restarts
                    .iter()
                    .map(|r| {
                        if r.feasible {
                            r.acq_value
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect()
            }
        };
        let maxima: Vec<(usize, f64)> = ok
            .iter()
            .map(|res| {
                let v = restart_values(res);
                let i = torch_argmax(&v);
                (i, v[i])
            })
            .collect();
        let best_ff = torch_argmax(&maxima.iter().map(|m| m.1).collect::<Vec<_>>());
        let res = &ok[best_ff];
        let restarts: Vec<(Matrix, f64)> = if res.restarts.is_empty() {
            vec![(res.candidates.clone(), res.acq_value)]
        } else {
            res.restarts
                .iter()
                .map(|r| (r.candidates.clone(), r.acq_value))
                .collect()
        };
        let (candidates, acq_value) = restarts[maxima[best_ff].0].clone();
        return Ok(MixedResult {
            candidates,
            acq_value,
            restarts: if return_best_only {
                Vec::new()
            } else {
                restarts
            },
            warnings,
        });
    }
    // Sequential greedy optimization for q > 1.
    let base = acq_function.x_pending();
    let step_options = OptimizeAcqfOptions {
        timeout_sec: options.timeout_sec.map(|t| t / q as f64),
        ..options.clone()
    };
    let mut rows: Vec<Matrix> = Vec::with_capacity(q);
    let mut warnings = Vec::new();
    let result = (|| -> Result<()> {
        for _ in 0..q {
            let step = optimize_acqf_mixed(
                acq_function,
                bounds,
                1,
                num_restarts,
                fixed_features_list,
                raw_samples,
                &step_options,
                true,
            )?;
            warnings.extend(step.warnings);
            rows.push(step.candidates);
            let parts: Vec<&Matrix> = base.iter().chain(rows.iter()).collect();
            acq_function
                .set_x_pending(Some(&crate::acquisition::monte_carlo::vstack(&parts, d)))?;
        }
        Ok(())
    })();
    acq_function.set_x_pending(base.as_ref())?;
    result?;
    let parts: Vec<&Matrix> = rows.iter().collect();
    let candidates = crate::acquisition::monte_carlo::vstack(&parts, d);
    let acq_value = acq_function.forward(&candidates)?;
    Ok(MixedResult {
        candidates,
        acq_value,
        restarts: Vec::new(),
        warnings,
    })
}

// ---------------------------------------------------------------------------
// Building blocks of optimize_acqf_mixed_alternating
// ---------------------------------------------------------------------------

/// Rounds the discrete dimensions of every row of `x` to the nearest allowed
/// value (BoTorch's `round_discrete_dims`; the first value on ties).
pub fn round_discrete_dims(x: &mut Matrix, discrete_dims: &DiscreteDims) {
    for (&dim, values) in discrete_dims {
        if values.is_empty() || dim >= x.ncols() {
            continue;
        }
        for i in 0..x.nrows() {
            let v = x[(i, dim)];
            let mut best = 0;
            let mut best_diff = (v - values[0]).abs();
            for (j, &a) in values.iter().enumerate().skip(1) {
                let diff = (v - a).abs();
                if diff < best_diff {
                    best = j;
                    best_diff = diff;
                }
            }
            x[(i, dim)] = values[best];
        }
    }
}

/// The index of the value closest to `v` (the first on ties).
fn closest_index(values: &[f64], v: f64) -> usize {
    let mut best = 0;
    for (i, &a) in values.iter().enumerate() {
        if (a - v).abs() < (values[best] - v).abs() {
            best = i;
        }
    }
    best
}

/// All points that differ from `x` by one step up or down in one discrete
/// dimension (BoTorch's `get_nearest_neighbors`: 1-Manhattan neighbors in
/// the index space of the allowed values), deduplicated and sorted like
/// `torch.unique(dim=0)` in index space, without `x` itself.
pub fn get_nearest_neighbors(
    x: &[f64],
    bounds: &Matrix,
    discrete_dims: &DiscreteDims,
) -> Vec<Vec<f64>> {
    let d = x.len();
    let mut x_int = x.to_vec();
    for (&dim, values) in discrete_dims {
        x_int[dim] = closest_index(values, x[dim]) as f64;
    }
    let mut lower: Vec<f64> = (0..d).map(|j| bounds[(0, j)]).collect();
    let mut upper: Vec<f64> = (0..d).map(|j| bounds[(1, j)]).collect();
    for (&dim, values) in discrete_dims {
        lower[dim] = 0.0;
        upper[dim] = (values.len() - 1) as f64;
    }
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(2 * discrete_dims.len());
    for &dim in discrete_dims.keys() {
        let mut minus = x_int.clone();
        minus[dim] -= 1.0;
        for j in 0..d {
            minus[j] = minus[j].max(lower[j]);
        }
        rows.push(minus);
    }
    for &dim in discrete_dims.keys() {
        let mut plus = x_int.clone();
        plus[dim] += 1.0;
        for j in 0..d {
            plus[j] = plus[j].min(upper[j]);
        }
        rows.push(plus);
    }
    rows.sort_by(|a, b| lex_cmp(a, b));
    rows.dedup();
    rows.retain(|r| r != &x_int);
    for r in &mut rows {
        for (&dim, values) in discrete_dims {
            r[dim] = values[r[dim] as usize];
        }
    }
    rows
}

/// All points that differ from `x` in the value of one categorical
/// dimension (BoTorch's `get_categorical_neighbors`), deduplicated and
/// sorted like `torch.unique(dim=0)`, without `x`. Dimensions with more than
/// `max_num_cat_values` values contribute a random sample of that many other
/// values.
pub fn get_categorical_neighbors<R: Rng + ?Sized>(
    x: &[f64],
    cat_dims: &DiscreteDims,
    max_num_cat_values: usize,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (&dim, values) in cat_dims {
        let chosen: Vec<f64> = if values.len() <= max_num_cat_values {
            values.clone()
        } else {
            let others: Vec<f64> = values.iter().copied().filter(|&v| v != x[dim]).collect();
            let k = max_num_cat_values.min(others.len());
            rand::seq::index::sample(rng, others.len(), k)
                .into_iter()
                .map(|i| others[i])
                .collect()
        };
        for v in chosen {
            let mut r = x.to_vec();
            r[dim] = v;
            rows.push(r);
        }
    }
    rows.sort_by(|a, b| lex_cmp(a, b));
    rows.dedup();
    rows.retain(|r| r.as_slice() != x);
    rows
}

/// Spray points around the baseline points (BoTorch's `get_spray_points`):
/// for every point of `x_baseline`, `num_spray_points` points with random
/// discrete and categorical neighbors and Gaussian perturbations (standard
/// deviation `std_cont_perturbation`, clamped to the bounds) of the
/// continuous dimensions. Other dimensions are zero.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if a baseline point has no discrete or
/// categorical neighbors.
#[allow(clippy::too_many_arguments)]
pub fn get_spray_points<R: Rng + ?Sized>(
    x_baseline: &Matrix,
    cont_dims: &[usize],
    discrete_dims: &DiscreteDims,
    cat_dims: &DiscreteDims,
    bounds: &Matrix,
    num_spray_points: usize,
    std_cont_perturbation: f64,
    rng: &mut R,
) -> Result<Matrix> {
    let d = bounds.ncols();
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(x_baseline.nrows() * num_spray_points);
    for b in 0..x_baseline.nrows() {
        let x: Vec<f64> = x_baseline.row(b).iter().copied().collect();
        let pick = |pts: &[Vec<f64>], rng: &mut R| -> Result<Vec<Vec<f64>>> {
            if pts.is_empty() {
                return Err(BotorchError::InvalidArgument(
                    "a baseline point has no discrete or categorical neighbors".into(),
                ));
            }
            Ok((0..num_spray_points)
                .map(|_| pts[rng.random_range(0..pts.len())].clone())
                .collect())
        };
        let disc = if discrete_dims.is_empty() {
            None
        } else {
            Some(pick(
                &get_nearest_neighbors(&x, bounds, discrete_dims),
                rng,
            )?)
        };
        let cat = if cat_dims.is_empty() {
            None
        } else {
            let n = get_categorical_neighbors(&x, cat_dims, MAX_DISCRETE_VALUES, rng);
            Some(pick(&n, rng)?)
        };
        let noise: Vec<f64> = (0..num_spray_points * cont_dims.len())
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        for s in 0..num_spray_points {
            let mut p = vec![0.0; d];
            if let Some(dp) = &disc {
                for &k in discrete_dims.keys() {
                    p[k] = dp[s][k];
                }
            }
            if let Some(cp) = &cat {
                for &k in cat_dims.keys() {
                    p[k] = cp[s][k];
                }
            }
            for (j, &k) in cont_dims.iter().enumerate() {
                let v = x[k] + std_cont_perturbation * noise[s * cont_dims.len() + j];
                p[k] = v.max(bounds[(0, k)]).min(bounds[(1, k)]);
            }
            out.push(p);
        }
    }
    Ok(Matrix::from_fn(out.len(), d, |i, j| out[i][j]))
}

/// BoTorch's default maximum number of values of a discrete dimension for
/// the discrete steps (`MAX_DISCRETE_VALUES`), above which a dimension is
/// optimized by continuous relaxation (discrete dimensions) or by random
/// subsets of values (categorical dimensions).
pub const MAX_DISCRETE_VALUES: usize = 20;

/// The initialization of the starting points of
/// [`optimize_acqf_mixed_alternating`] (`options["initialization_strategy"]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitializationStrategy {
    /// Optimize the continuous relaxation from raw samples (default unless
    /// all discrete dimensions are binary).
    ContinuousRelaxation,
    /// Sobol points with a random number of binary dimensions set to their
    /// upper bound (default if there are at least two binary and no other
    /// discrete dimensions).
    EquallySpaced,
    /// Sobol points.
    Random,
}

/// Options of [`optimize_acqf_mixed_alternating`] (BoTorch's `options`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixedAlternatingOptions {
    /// `initialization_strategy` (`None`: BoTorch's default rule).
    pub initialization_strategy: Option<InitializationStrategy>,
    /// Convergence tolerance of the improvement per alternating step (1e-8).
    pub tol: f64,
    /// Maximum number of alternating steps (64).
    pub maxiter_alternating: usize,
    /// Maximum number of iterations of a discrete step (4).
    pub maxiter_discrete: usize,
    /// Maximum number of L-BFGS-B iterations of a continuous step (8).
    pub maxiter_continuous: usize,
    /// Maximum number of L-BFGS-B iterations of the continuous relaxation in
    /// the initialization (100).
    pub maxiter_init: usize,
    /// Discrete dimensions with more values are relaxed to continuous ones and
    /// rounded (20).
    pub max_discrete_values: usize,
    /// Spray points per baseline point (`None`: 20 if all discrete dimensions
    /// are binary, else 0).
    pub num_spray_points: Option<usize>,
    /// Standard deviation of the continuous perturbations of the spray points
    /// (0.1).
    pub std_cont_perturbation: f64,
    /// Restarts per chunk of the continuous optimizations (2048).
    pub batch_limit: usize,
    /// Points per evaluation chunk in the initialization (`batch_limit`).
    pub init_batch_limit: Option<usize>,
}

impl Default for MixedAlternatingOptions {
    fn default() -> Self {
        Self {
            initialization_strategy: None,
            tol: 1e-8,
            maxiter_alternating: 64,
            maxiter_discrete: 4,
            maxiter_continuous: 8,
            maxiter_init: 100,
            max_discrete_values: MAX_DISCRETE_VALUES,
            num_spray_points: None,
            std_cont_perturbation: 0.1,
            batch_limit: 2048,
            init_batch_limit: None,
        }
    }
}

/// The search space and settings of a mixed alternating optimization (what
/// BoTorch passes around as `opt_inputs` plus the dimension maps).
#[derive(Clone, Debug)]
pub struct MixedProblem<'a> {
    /// The acquisition function.
    pub acq_function: &'a dyn AcquisitionFunction,
    /// The bounds (`2 × d`).
    pub bounds: &'a Matrix,
    /// The discrete dimensions optimized by the discrete steps (sorted
    /// values; relaxed and fixed dimensions removed).
    pub discrete_dims: DiscreteDims,
    /// The categorical dimensions (sorted values; fixed dimensions removed).
    pub cat_dims: DiscreteDims,
    /// The continuous dimensions (without fixed features).
    pub cont_dims: Vec<usize>,
    /// The user's fixed features.
    pub fixed_features: BTreeMap<usize, f64>,
    /// Linear inequality and equality constraints (intra-point).
    pub constraints: ParameterConstraints,
    /// The options.
    pub options: MixedAlternatingOptions,
    /// Number of starting points.
    pub num_restarts: usize,
    /// Number of raw samples of the initialization.
    pub raw_samples: usize,
    /// Post-processing of the continuous steps' candidates (rounding of the
    /// relaxed dimensions and the user's `post_processing_func`).
    pub post_processing: Option<PostProcessing>,
    /// Retry the relaxation's optimization on optimization warnings.
    pub retry_on_optimization_warning: bool,
    /// The baseline points for the spray points (BoTorch's `get_X_baseline`).
    pub x_baseline: Option<Matrix>,
}

impl MixedProblem<'_> {
    fn init_batch_limit(&self) -> usize {
        self.options
            .init_batch_limit
            .unwrap_or(self.options.batch_limit)
            .max(1)
    }

    fn feasible(&self, x: &[f64]) -> Result<bool> {
        if self.constraints.is_empty() {
            return Ok(true);
        }
        let m = Matrix::from_row_slice(1, x.len(), x);
        evaluate_feasibility(&m, &self.constraints, CONSTRAINT_TOLERANCE)
    }

    fn filter_infeasible(&self, rows: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            if self.feasible(&r)? {
                out.push(r);
            }
        }
        Ok(out)
    }

    fn values(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>> {
        let d = self.bounds.ncols();
        let sets: Vec<Matrix> = rows
            .iter()
            .map(|r| Matrix::from_row_slice(1, d, r))
            .collect();
        let mut out = Vec::with_capacity(sets.len());
        for c in sets.chunks(self.init_batch_limit()) {
            out.extend(self.acq_function.forward_many(c)?);
        }
        Ok(out)
    }
}

fn matrix_rows(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| m.row(i).iter().copied().collect())
        .collect()
}

fn rows_matrix(rows: &[Vec<f64>], d: usize) -> Matrix {
    Matrix::from_fn(rows.len(), d, |i, j| rows[i][j])
}

/// Feasible points of the domain (BoTorch's `sample_feasible_points`): Sobol
/// points (or, with linear constraints, hit-and-run samples of the
/// constraint polytope) with the discrete and categorical dimensions rounded
/// and the fixed features set; infeasible points are dropped. Up to 10 tries
/// of `2 × remaining` points.
///
/// # Errors
/// [`BotorchError::CandidateGeneration`] if no feasible point is found.
pub fn sample_feasible_points<R: Rng + ?Sized>(
    problem: &MixedProblem<'_>,
    num_points: usize,
    rng: &mut R,
) -> Result<Matrix> {
    let bounds = problem.bounds;
    let d = bounds.ncols();
    let linear = &problem.constraints;
    let mut all: Vec<Vec<f64>> = Vec::new();
    let unconstrained = linear.inequality.is_empty() && linear.equality.is_empty();
    let engine = if unconstrained {
        Some(SobolEngine::new(d, Some(rng.random::<u64>()))?)
    } else {
        None
    };
    let mut sampler = if unconstrained {
        None
    } else {
        let ineq = if linear.inequality.is_empty() {
            None
        } else {
            let (a, b) = sparse_to_dense_constraints(d, &linear.inequality)?;
            Some((-a, -b))
        };
        let eq = if linear.equality.is_empty() {
            None
        } else {
            Some(sparse_to_dense_constraints(d, &linear.equality)?)
        };
        Some(HitAndRunPolytopeSampler::new(
            ineq,
            eq,
            Some(bounds),
            None,
            200,
            20,
            Some(rng.random::<u64>()),
        )?)
    };
    let mut drawn = 0;
    for _ in 0..10 {
        let remaining = num_points.saturating_sub(all.len());
        if remaining == 0 {
            break;
        }
        let n = remaining * 2;
        let mut base = match (&engine, sampler.as_mut()) {
            (Some(e), _) => {
                let u = e.draw(n, drawn)?;
                drawn += n;
                Matrix::from_fn(n, d, |i, j| {
                    u[(i, j)] * (bounds[(1, j)] - bounds[(0, j)]) + bounds[(0, j)]
                })
            }
            (None, Some(s)) => s.draw(n)?,
            _ => unreachable!("a Sobol engine or a polytope sampler"),
        };
        round_discrete_dims(&mut base, &problem.discrete_dims);
        round_discrete_dims(&mut base, &problem.cat_dims);
        apply_fixed_features(&mut base, &problem.fixed_features);
        all.extend(problem.filter_infeasible(matrix_rows(&base))?);
    }
    if all.is_empty() {
        return Err(BotorchError::CandidateGeneration(
            "Could not generate any feasible starting points for mixed optimizer.".into(),
        ));
    }
    all.truncate(num_points);
    Ok(rows_matrix(&all, d))
}

/// The starting points of the alternating optimization and their values
/// (BoTorch's `generate_starting_points`): candidates from the initialization
/// strategy (the continuous relaxation optimized by `optimize_acqf`, or Sobol
/// points, binary dimensions equally spaced) plus spray points around the
/// baseline points, rounded, with the fixed features set and infeasible ones
/// removed, completed by [`sample_feasible_points`]; `num_restarts` of them
/// are selected by Boltzmann sampling ([`initialize_q_batch`]).
///
/// Returns the points (`b × d`, usually `b = num_restarts`), their values and
/// BoTorch's warnings.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for equally spaced initialization with
/// non-binary discrete dimensions, and the errors of
/// [`sample_feasible_points`] and the acquisition function.
pub fn generate_starting_points<R: Rng + ?Sized>(
    problem: &MixedProblem<'_>,
    rng: &mut R,
) -> Result<(Matrix, Vec<f64>, Vec<String>)> {
    let bounds = problem.bounds;
    let d = bounds.ncols();
    let opts = &problem.options;
    let binary_dims: Vec<usize> = problem
        .discrete_dims
        .iter()
        .filter(|(_, v)| v.len() == 2)
        .map(|(&k, _)| k)
        .collect();
    let num_binary = binary_dims.len();
    let num_discrete = problem.discrete_dims.len() - num_binary;
    let strategy =
        opts.initialization_strategy
            .unwrap_or(if num_discrete == 0 && num_binary >= 2 {
                InitializationStrategy::EquallySpaced
            } else {
                InitializationStrategy::ContinuousRelaxation
            });
    let mut warnings = Vec::new();
    let mut x_init: Vec<Vec<f64>> = Vec::new();
    if strategy == InitializationStrategy::ContinuousRelaxation {
        let relax_options = OptimizeAcqfOptions {
            seed: Some(rng.random::<u64>()),
            lbfgsb: crate::lbfgsb::LbfgsbOptions {
                maxiter: opts.maxiter_init,
                ..default_lbfgsb_options()
            },
            batch_limit: Some(opts.batch_limit),
            init_batch_limit: Some(problem.init_batch_limit()),
            fixed_features: problem.fixed_features.clone(),
            constraints: problem.constraints.clone(),
            retry_on_optimization_warning: problem.retry_on_optimization_warning,
            ..OptimizeAcqfOptions::default()
        };
        match optimize_acqf_detailed(
            problem.acq_function,
            bounds,
            1,
            problem.num_restarts,
            problem.raw_samples,
            &relax_options,
        ) {
            Ok(res) => {
                warnings.extend(res.warnings);
                x_init = if res.restarts.is_empty() {
                    matrix_rows(&res.candidates)
                } else {
                    res.restarts
                        .iter()
                        .map(|r| r.candidates.row(0).iter().copied().collect())
                        .collect()
                };
            }
            Err(e) => warnings.push(format!(
                "Failed to initialize using continuous relaxation. Using \
                 `sample_feasible_points` for initialization. Original error message: {e}"
            )),
        }
    }
    if x_init.is_empty() {
        let sobol = draw_sobol_samples(bounds, problem.raw_samples, 1, Some(rng.random::<u64>()))?;
        x_init = sobol
            .iter()
            .map(|m| m.row(0).iter().copied().collect())
            .collect();
    }
    if strategy == InitializationStrategy::EquallySpaced {
        if num_discrete > 0 {
            return Err(BotorchError::InvalidArgument(
                "Equally spaced initialization is not supported with non-binary discrete \
                 variables."
                    .into(),
            ));
        }
        if num_binary == 0 {
            return Err(BotorchError::InvalidArgument(
                "Equally spaced initialization requires binary dimensions.".into(),
            ));
        }
        let ks: Vec<usize> = (0..problem.raw_samples)
            .map(|_| rng.random_range(0..num_binary))
            .collect();
        for (i, x) in x_init.iter_mut().enumerate() {
            for &k in &binary_dims {
                x[k] = bounds[(0, k)];
            }
            let perm = rand::seq::index::sample(rng, num_binary, num_binary).into_vec();
            for &p in perm.iter().take(ks.get(i).copied().unwrap_or(0)) {
                let k = binary_dims[p];
                x[k] = bounds[(1, k)];
            }
        }
    }
    let num_spray = opts
        .num_spray_points
        .unwrap_or(if num_discrete == 0 { 20 } else { 0 });
    if num_spray > 0 {
        if let Some(xb) = &problem.x_baseline {
            let spray = get_spray_points(
                xb,
                &problem.cont_dims,
                &problem.discrete_dims,
                &problem.cat_dims,
                bounds,
                num_spray,
                opts.std_cont_perturbation,
                rng,
            )?;
            x_init.extend(matrix_rows(&spray));
        }
    }
    let mut m = rows_matrix(&x_init, d);
    round_discrete_dims(&mut m, &problem.discrete_dims);
    round_discrete_dims(&mut m, &problem.cat_dims);
    apply_fixed_features(&mut m, &problem.fixed_features);
    let mut points = problem.filter_infeasible(matrix_rows(&m))?;
    if points.len() < problem.num_restarts {
        let extra = sample_feasible_points(problem, problem.num_restarts - points.len(), rng)?;
        points.extend(matrix_rows(&extra));
    }
    let mut values = problem.values(&points)?;
    if points.len() > problem.num_restarts {
        let (idx, fallback) = initialize_q_batch(&values, problem.num_restarts, 1.0, rng)?;
        if fallback {
            warnings.push(
                "All acquisition values for raw samples points are the same or not finite for \
                 at least one batch. Choosing initial conditions at random."
                    .into(),
            );
        }
        points = idx.iter().map(|&i| points[i].clone()).collect();
        values = idx.iter().map(|&i| values[i]).collect();
    }
    Ok((rows_matrix(&points, d), values, warnings))
}

/// Discrete nearest-neighbor search from every row of `current_x`
/// (BoTorch's `discrete_step`): up to `maxiter_discrete` times, every point
/// that is not done moves to its best feasible discrete or categorical
/// neighbor if that improves its value; a point is done when the improvement
/// is at most `tol` or it has no neighbors. Updates `current_x` in place and
/// returns the values.
///
/// # Errors
/// Errors of the acquisition function and the constraints.
pub fn discrete_step<R: Rng + ?Sized>(
    problem: &MixedProblem<'_>,
    current_x: &mut Matrix,
    rng: &mut R,
) -> Result<Vec<f64>> {
    let b = current_x.nrows();
    let mut rows = matrix_rows(current_x);
    let mut values = problem.values(&rows)?;
    let mut done = vec![false; b];
    for _ in 0..problem.options.maxiter_discrete {
        let mut lists: Vec<Option<Vec<Vec<f64>>>> = vec![None; b];
        for i in 0..b {
            if done[i] {
                continue;
            }
            let mut nb = Vec::new();
            if !problem.discrete_dims.is_empty() {
                let n = get_nearest_neighbors(&rows[i], problem.bounds, &problem.discrete_dims);
                nb.extend(problem.filter_infeasible(n)?);
            }
            if !problem.cat_dims.is_empty() {
                let n = get_categorical_neighbors(
                    &rows[i],
                    &problem.cat_dims,
                    MAX_DISCRETE_VALUES,
                    rng,
                );
                nb.extend(problem.filter_infeasible(n)?);
            }
            if nb.is_empty() {
                done[i] = true;
            }
            lists[i] = Some(nb);
        }
        if done.iter().all(|&x| x) {
            break;
        }
        let all: Vec<Vec<f64>> = lists.iter().flatten().flatten().cloned().collect();
        let acq = problem.values(&all)?;
        let mut offset = 0;
        for i in 0..b {
            if done[i] {
                continue;
            }
            let list = lists[i].as_ref().expect("neighbors of an active point");
            let width = list.len();
            let vals = &acq[offset..offset + width];
            let j = torch_argmax(vals);
            let improvement = vals[j] - values[i];
            if improvement > 0.0 {
                values[i] = vals[j];
                rows[i] = list[j].clone();
            }
            if improvement <= problem.options.tol {
                done[i] = true;
            }
            offset += width;
        }
    }
    *current_x = rows_matrix(&rows, current_x.ncols());
    Ok(values)
}

/// Continuous optimization of every row of `current_x` with the discrete and
/// categorical dimensions fixed (BoTorch's `continuous_step`, i.e.
/// `optimize_acqf` from the given points with `maxiter_continuous`
/// L-BFGS-B iterations and `gtol = tol`, or SLSQP with linear constraints),
/// followed by the post-processing and the projection of infeasible results.
/// Returns the optimized points, their values and BoTorch's warnings.
///
/// # Errors
/// [`BotorchError::CandidateGeneration`] if all results are infeasible, and
/// errors of the optimizers and the acquisition function.
pub fn continuous_step(
    problem: &MixedProblem<'_>,
    current_x: &Matrix,
) -> Result<(Matrix, Vec<f64>, Vec<String>)> {
    let d = current_x.ncols();
    let b = current_x.nrows();
    let non_cont: Vec<usize> = problem
        .discrete_dims
        .keys()
        .chain(problem.cat_dims.keys())
        .copied()
        .collect();
    let rows = matrix_rows(current_x);
    if non_cont.len() + problem.fixed_features.len() >= d {
        let values = problem.values(&rows)?;
        return Ok((current_x.clone(), values, Vec::new()));
    }
    let fixed_of = |i: usize| -> BTreeMap<usize, f64> {
        let mut f: BTreeMap<usize, f64> = non_cont.iter().map(|&k| (k, rows[i][k])).collect();
        f.extend(problem.fixed_features.iter().map(|(&k, &v)| (k, v)));
        f
    };
    let constrained = !problem.constraints.is_empty();
    let lbfgsb = crate::lbfgsb::LbfgsbOptions {
        maxiter: problem.options.maxiter_continuous,
        gtol: problem.options.tol,
        ..default_lbfgsb_options()
    };
    let slsqp = SlsqpOptions {
        maxiter: problem.options.maxiter_continuous,
        ..SlsqpOptions::default()
    };
    let results = (0..b)
        .into_par_iter()
        .map(|i| -> Result<(Matrix, f64, bool)> {
            let ic = Matrix::from_row_slice(1, d, &rows[i]);
            let fixed = fixed_of(i);
            if constrained {
                let r = gen_candidates_slsqp(
                    &[ic],
                    problem.acq_function,
                    problem.bounds,
                    &fixed,
                    &problem.constraints,
                    &slsqp,
                )?;
                Ok((
                    r.candidates[0].clone(),
                    r.acq_values[0],
                    !r.warnings.is_empty(),
                ))
            } else {
                let r = gen_candidates_lbfgsb(
                    &[ic],
                    problem.acq_function,
                    problem.bounds,
                    &fixed,
                    &lbfgsb,
                )?;
                let failed =
                    r[0].status == OptimizerStatus::Lbfgsb(LbfgsbStatus::AbnormalLineSearch);
                Ok((r[0].candidates.clone(), r[0].acq_value, failed))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let mut warnings = Vec::new();
    if results.iter().any(|r| r.2) {
        let tail = if b < problem.num_restarts {
            "Trying again with a new set of initial conditions."
        } else {
            "Because you specified `batch_initial_conditions` larger than required \
             `num_restarts`, optimization will not be retried with new initial conditions and \
             will proceed with the current solution. Suggested remediation: Try again with \
             different `batch_initial_conditions`, don't provide `batch_initial_conditions`, \
             or increase `num_restarts`."
        };
        warnings.push(format!(
            "Optimization failed in `gen_candidates_scipy` with the following warning(s):\n\
             [abnormal termination in the line search]\n{tail}"
        ));
    }
    let mut sets: Vec<Matrix> = results.iter().map(|r| r.0.clone()).collect();
    let mut values: Vec<f64> = results.iter().map(|r| r.1).collect();
    if let Some(pp) = &problem.post_processing {
        sets = pp.apply(&sets)?;
        values = problem.acq_function.forward_many(&sets)?;
    }
    if constrained {
        let mut feasible = sets
            .iter()
            .map(|s| evaluate_feasibility(s, &problem.constraints, CONSTRAINT_TOLERANCE))
            .collect::<Result<Vec<_>>>()?;
        for i in 0..b {
            if feasible[i] {
                continue;
            }
            let mut projected = project_to_feasible_space(
                &sets[i..=i],
                problem.bounds,
                &problem.constraints,
                &fixed_of(i),
            )?;
            if let Some(pp) = &problem.post_processing {
                projected = pp.apply(&projected)?;
            }
            values[i] = problem.acq_function.forward(&projected[0])?;
            sets[i] = projected.remove(0);
            feasible[i] =
                evaluate_feasibility(&sets[i], &problem.constraints, CONSTRAINT_TOLERANCE)?;
        }
        if feasible.iter().all(|f| !f) {
            return Err(BotorchError::CandidateGeneration(format!(
                "The optimizer produced infeasible candidates. {b} out of {b} batches of \
                 candidates were infeasible. Please make sure the constraints are satisfiable \
                 and relax them if needed. "
            )));
        }
    }
    let out = Matrix::from_fn(b, d, |i, j| sets[i][(0, j)]);
    Ok((out, values, warnings))
}

/// The function type of a replacement of [`generate_starting_points`]
/// (a hook for reproducing given starting points): returns the points
/// (`b × d`) and their values.
pub type StartingPointsFn<'a> = dyn Fn(&MixedProblem<'_>) -> Result<(Matrix, Vec<f64>)> + Sync + 'a;

/// The result of [`optimize_acqf_mixed_alternating`].
#[derive(Clone, Debug, PartialEq)]
pub struct MixedAlternatingResult {
    /// The candidates (`q × d`), post-processed.
    pub candidates: Matrix,
    /// Their joint acquisition value.
    pub acq_value: f64,
    /// Messages of BoTorch's warnings.
    pub warnings: Vec<String>,
}

/// Validates the dimension maps and builds the problem of
/// [`optimize_acqf_mixed_alternating`] (sorting values, relaxing discrete
/// dimensions with more than `max_discrete_values` values that do not appear
/// in constraints, removing fixed dimensions). Returns the problem and the
/// relaxed dimensions.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] with BoTorch's messages for discrete
/// dimensions that do not start/end at the bounds, equality constraints on
/// discrete dimensions and invalid dimension indices;
/// [`BotorchError::Unsupported`] for inter-point or nonlinear constraints.
#[allow(clippy::too_many_arguments)]
pub fn mixed_problem<'a>(
    acq_function: &'a dyn AcquisitionFunction,
    bounds: &'a Matrix,
    discrete_dims: &DiscreteDims,
    cat_dims: &DiscreteDims,
    options: &MixedAlternatingOptions,
    num_restarts: usize,
    raw_samples: usize,
    fixed_features: &BTreeMap<usize, f64>,
    constraints: &ParameterConstraints,
    retry_on_optimization_warning: bool,
    x_baseline: Option<Matrix>,
) -> Result<(MixedProblem<'a>, DiscreteDims)> {
    let d = check_bounds(bounds)?;
    if !constraints.nonlinear.is_empty() {
        return Err(BotorchError::Unsupported(
            "`optimize_acqf_mixed_alternating` does not support nonlinear inequality constraints."
                .into(),
        ));
    }
    let sort = |m: &DiscreteDims| -> DiscreteDims {
        m.iter()
            .map(|(&k, v)| {
                let mut v = v.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                (k, v)
            })
            .collect()
    };
    let discrete = sort(discrete_dims);
    let cat = sort(cat_dims);
    for (&dim, values) in &discrete {
        if dim >= d || values.is_empty() {
            continue;
        }
        let (lo, hi) = (bounds[(0, dim)], bounds[(1, dim)]);
        if values[0] != lo {
            return Err(BotorchError::InvalidArgument(format!(
                "Discrete dimension {dim} must start at the lower bound {lo:?} but starts at {:?}.",
                values[0]
            )));
        }
        if values[values.len() - 1] != hi {
            return Err(BotorchError::InvalidArgument(format!(
                "Discrete dimension {dim} must end at the upper bound {hi:?} but end at {:?}.",
                values[values.len() - 1]
            )));
        }
    }
    let constrained_dims: Vec<usize> = constraints
        .inequality
        .iter()
        .chain(&constraints.equality)
        .flat_map(|c| match c.indices() {
            crate::optim::ConstraintIndices::IntraPoint(idx) => idx.clone(),
            crate::optim::ConstraintIndices::InterPoint(p) => p.iter().map(|x| x.1).collect(),
        })
        .collect();
    for c in &constraints.equality {
        if let crate::optim::ConstraintIndices::IntraPoint(idx) = c.indices() {
            if idx
                .iter()
                .any(|k| discrete.contains_key(k) || cat.contains_key(k))
            {
                return Err(BotorchError::InvalidArgument(
                    "Equality constraints can only be used with continuous degrees of freedom."
                        .into(),
                ));
            }
        }
    }
    // Continuous relaxation of large discrete dimensions not in constraints.
    let mut kept = DiscreteDims::new();
    let mut relaxed = DiscreteDims::new();
    for (k, v) in discrete {
        if !constrained_dims.contains(&k) && v.len() > options.max_discrete_values {
            relaxed.insert(k, v);
        } else {
            kept.insert(k, v);
        }
    }
    constraints.validate_sequential()?;
    constraints.validate(1, d)?;
    let kept: DiscreteDims = kept
        .into_iter()
        .filter(|(k, _)| !fixed_features.contains_key(k))
        .collect();
    let cat: DiscreteDims = cat
        .into_iter()
        .filter(|(k, _)| !fixed_features.contains_key(k))
        .collect();
    let non_cont: Vec<usize> = kept.keys().chain(cat.keys()).copied().collect();
    let mut unique = non_cont.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != non_cont.len() || non_cont.iter().any(|&k| k >= d) {
        return Err(BotorchError::InvalidArgument(
            "`discrete_dims` and `cat_dims` must be dictionaries with unique, disjoint integers \
             as keys between 0 and num_dims - 1."
                .into(),
        ));
    }
    let cont_dims: Vec<usize> = (0..d)
        .filter(|k| !non_cont.contains(k) && !fixed_features.contains_key(k))
        .collect();
    let problem = MixedProblem {
        acq_function,
        bounds,
        discrete_dims: kept,
        cat_dims: cat,
        cont_dims,
        fixed_features: fixed_features.clone(),
        constraints: constraints.clone(),
        options: *options,
        num_restarts,
        raw_samples,
        post_processing: None,
        retry_on_optimization_warning,
        x_baseline,
    };
    Ok((problem, relaxed))
}

/// The post-processing of [`optimize_acqf_mixed_alternating`]: rounding of
/// the relaxed dimensions followed by the user's function.
pub fn relaxation_post_processing(
    relaxed: DiscreteDims,
    user: Option<PostProcessing>,
) -> Option<PostProcessing> {
    if relaxed.is_empty() {
        return user;
    }
    Some(PostProcessing::new(move |sets: &[Matrix]| {
        let rounded: Vec<Matrix> = sets
            .iter()
            .map(|s| {
                let mut s = s.clone();
                round_discrete_dims(&mut s, &relaxed);
                s
            })
            .collect();
        match &user {
            Some(u) => u.apply(&rounded),
            None => Ok(rounded),
        }
    }))
}

/// Optimizes the acquisition function over mixed integer, categorical and
/// continuous inputs (BoTorch's `optimize_acqf_mixed_alternating`), for the
/// problem built by [`mixed_problem`]: for every one of the `q` candidates
/// (generated sequentially, conditioning on the previous ones as pending
/// points), starting points are generated ([`generate_starting_points`], or
/// `starting_points`) and alternate [`discrete_step`]s and
/// [`continuous_step`]s until the improvement of every point is below `tol`
/// or `maxiter_alternating` steps; the best point is the candidate. The
/// candidates are post-processed and their joint value is returned.
///
/// Without discrete and categorical dimensions the caller runs
/// `optimize_acqf` instead (as BoTorch does).
///
/// # Errors
/// The errors of the building blocks and of the acquisition function.
pub fn optimize_acqf_mixed_alternating(
    problem: &MixedProblem<'_>,
    q: usize,
    seed: Option<u64>,
    starting_points: Option<&StartingPointsFn<'_>>,
    return_acq_values: bool,
) -> Result<MixedAlternatingResult> {
    let acq_function = problem.acq_function;
    let d = problem.bounds.ncols();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.unwrap_or_else(random_seed));
    let base = if q > 1 {
        acq_function.x_pending()
    } else {
        None
    };
    let mut candidates: Vec<Matrix> = Vec::with_capacity(q);
    let mut warnings = Vec::new();
    let result = (|| -> Result<()> {
        for _ in 0..q {
            let (mut best_x, mut best_vals) = match starting_points {
                Some(f) => f(problem)?,
                None => {
                    let (x, v, w) = generate_starting_points(problem, &mut rng)?;
                    warnings.extend(w);
                    (x, v)
                }
            };
            if best_x.nrows() != best_vals.len() || best_x.ncols() != d || best_x.nrows() == 0 {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the starting points must be b x {d} with b values, got {}x{} and {} values",
                    best_x.nrows(),
                    best_x.ncols(),
                    best_vals.len()
                )));
            }
            let b = best_x.nrows();
            let mut done = vec![false; b];
            for _ in 0..problem.options.maxiter_alternating {
                let starting = best_vals.clone();
                let active: Vec<usize> = (0..b).filter(|&i| !done[i]).collect();
                let mut sub = Matrix::from_fn(active.len(), d, |r, c| best_x[(active[r], c)]);
                let vals = discrete_step(problem, &mut sub, &mut rng)?;
                for (r, &i) in active.iter().enumerate() {
                    best_x.row_mut(i).copy_from(&sub.row(r));
                    best_vals[i] = vals[r];
                }
                let (cx, cv, w) = continuous_step(problem, &sub)?;
                warnings.extend(w);
                for (r, &i) in active.iter().enumerate() {
                    best_x.row_mut(i).copy_from(&cx.row(r));
                    best_vals[i] = cv[r];
                }
                for i in 0..b {
                    if best_vals[i] - starting[i] < problem.options.tol {
                        done[i] = true;
                    }
                }
                if done.iter().all(|&x| x) {
                    break;
                }
            }
            let best = torch_argmax(&best_vals);
            candidates.push(best_x.rows(best, 1).into_owned());
            if q > 1 {
                let parts: Vec<&Matrix> = base.iter().chain(candidates.iter()).collect();
                acq_function
                    .set_x_pending(Some(&crate::acquisition::monte_carlo::vstack(&parts, d)))?;
            }
        }
        Ok(())
    })();
    if q > 1 {
        acq_function.set_x_pending(base.as_ref())?;
    }
    result?;
    let parts: Vec<&Matrix> = candidates.iter().collect();
    let mut cands = crate::acquisition::monte_carlo::vstack(&parts, d);
    if let Some(pp) = &problem.post_processing {
        cands = pp.apply(&[cands])?.remove(0);
    }
    let acq_value = if return_acq_values {
        acq_function.forward(&cands)?
    } else {
        f64::NAN
    };
    Ok(MixedAlternatingResult {
        candidates: cands,
        acq_value,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_and_neighbors() {
        let dims: DiscreteDims = [(0usize, vec![0.0, 1.0, 2.0, 5.0])].into_iter().collect();
        let mut x = Matrix::from_row_slice(2, 2, &[1.4, 0.3, 3.6, 0.2]);
        round_discrete_dims(&mut x, &dims);
        assert_eq!(x[(0, 0)], 1.0);
        assert_eq!(x[(1, 0)], 5.0);
        let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 5.0, 1.0]);
        let n = get_nearest_neighbors(&[2.0, 0.3], &bounds, &dims);
        assert_eq!(n, vec![vec![1.0, 0.3], vec![5.0, 0.3]]);
        let n = get_nearest_neighbors(&[0.0, 0.3], &bounds, &dims);
        assert_eq!(n, vec![vec![1.0, 0.3]]);
        let cats: DiscreteDims = [(1usize, vec![0.0, 1.0, 2.0])].into_iter().collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let n = get_categorical_neighbors(&[0.5, 1.0], &cats, 20, &mut rng);
        assert_eq!(n, vec![vec![0.5, 0.0], vec![0.5, 2.0]]);
        assert!(should_use_mixed_alternating_optimizer(
            Some(&dims),
            Some(&cats)
        ));
        assert!(!should_use_mixed_alternating_optimizer(Some(&cats), None));
        assert!(!should_use_mixed_alternating_optimizer(None, None));
    }
}
