//! Initial conditions for multi-start acquisition optimization, mirroring
//! `botorch.optim.initializers`: [`gen_batch_initial_conditions`] draws
//! `raw_samples` scrambled Sobol candidate sets (or, with linear constraints,
//! hit-and-run samples from the constraint polytope,
//! [`sample_q_batches_from_polytope`]), evaluates the acquisition function on
//! all of them and selects `num_restarts` starting points with
//! [`initialize_q_batch`] (Boltzmann sampling) or
//! [`initialize_q_batch_nonneg`] (for non-negative acquisition functions).

use std::collections::BTreeMap;

use rand::{Rng, RngExt};

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::optim::parameter_constraints::{ConstraintIndices, LinearConstraint};
use crate::utils::polytope::get_polytope_samples;
use crate::utils::sampling::{check_bounds, draw_sobol_samples};
use crate::Matrix;

/// Default number of burn-in steps of the polytope sampler of the initial
/// conditions (`options.get("n_burnin", 10000)`).
pub const DEFAULT_N_BURNIN: usize = 10_000;
/// Default thinning of the polytope sampler (`options.get("n_thinning", 32)`).
pub const DEFAULT_N_THINNING: usize = 32;

/// Default Boltzmann temperature of [`initialize_q_batch`] and
/// [`initialize_q_batch_nonneg`] (BoTorch's `eta=1.0`).
pub const DEFAULT_ETA: f64 = 1.0;
/// Default threshold fraction of [`initialize_q_batch_nonneg`]
/// (BoTorch's `alpha=1e-4`).
pub const DEFAULT_ALPHA: f64 = 1e-4;

/// Indices drawn without replacement with probabilities proportional to
/// `weights` (in order of drawing, like `torch.multinomial(weights, n)`).
fn weighted_sample_without_replacement<R: Rng + ?Sized>(
    weights: &[f64],
    n: usize,
    rng: &mut R,
) -> Vec<usize> {
    let mut w = weights.to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let total: f64 = w.iter().sum();
        let mut pick = None;
        if total > 0.0 && total.is_finite() {
            let mut u = rng.random::<f64>() * total;
            for (i, &wi) in w.iter().enumerate() {
                if wi > 0.0 {
                    pick = Some(i);
                    if u < wi {
                        break;
                    }
                    u -= wi;
                }
            }
        }
        // All remaining weights are zero: fall back to a uniform draw.
        let i = pick.unwrap_or_else(|| {
            let remaining: Vec<usize> = (0..w.len()).filter(|i| !out.contains(i)).collect();
            remaining[rng.random_range(0..remaining.len())]
        });
        out.push(i);
        w[i] = 0.0;
    }
    out
}

/// The first `n` elements of a uniformly random permutation of `0..len`.
fn random_subset<R: Rng + ?Sized>(len: usize, n: usize, rng: &mut R) -> Vec<usize> {
    rand::seq::index::sample(rng, len, n).into_vec()
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Unbiased standard deviation (`torch.std`).
fn std_unbiased(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// Boltzmann selection of initial conditions (BoTorch's `initialize_q_batch`):
/// draws `n` of the candidates without replacement with probabilities
/// proportional to `exp(eta · z)`, where `z` are the standardized acquisition
/// values (`eta` is halved while weights overflow), and makes sure the best
/// candidate is included (replacing the last draw if necessary).
///
/// Returns the selected indices into `acq_values` and whether BoTorch's
/// `BadInitialCandidatesWarning` fallback was used (all values equal or
/// non-finite: a uniformly random subset).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `n` exceeds the number of candidates.
pub fn initialize_q_batch<R: Rng + ?Sized>(
    acq_values: &[f64],
    n: usize,
    eta: f64,
    rng: &mut R,
) -> Result<(Vec<usize>, bool)> {
    let n_samples = acq_values.len();
    if n > n_samples {
        return Err(BotorchError::InvalidArgument(format!(
            "n ({n}) cannot be larger than the number of provided samples ({n_samples})"
        )));
    }
    if n == n_samples {
        return Ok(((0..n).collect(), false));
    }
    let ystd = std_unbiased(acq_values);
    if ystd == 0.0 || !ystd.is_finite() {
        return Ok((random_subset(n_samples, n, rng), true));
    }
    let max_idx = argmax(acq_values);
    // botorch.utils.transforms.standardize + boltzmann_sample.
    let mean = acq_values.iter().sum::<f64>() / n_samples as f64;
    let std = if ystd >= 1e-9 { ystd } else { 1.0 };
    let z: Vec<f64> = acq_values.iter().map(|v| (v - mean) / std).collect();
    let mut eta = eta;
    let mut weights: Vec<f64> = z.iter().map(|v| (eta * v).exp()).collect();
    while weights.iter().any(|w| w.is_infinite()) {
        eta *= 0.5;
        weights = z.iter().map(|v| (eta * v).exp()).collect();
    }
    let mut idcs = weighted_sample_without_replacement(&weights, n, rng);
    if !idcs.contains(&max_idx) {
        idcs[n - 1] = max_idx;
    }
    Ok((idcs, false))
}

/// Selection of initial conditions for non-negative acquisition functions
/// (BoTorch's `initialize_q_batch_nonneg`): candidates with values below
/// `alpha · max` are ignored (`alpha` is divided by 10 until at least `n`
/// remain) and the rest are drawn with weights `exp(eta (v / max - 1))`; if
/// fewer than `n` values are positive, all positive candidates are taken and
/// the rest are filled randomly.
///
/// Returns the selected indices and whether BoTorch's
/// `BadInitialCandidatesWarning` fallback was used (no positive value: a
/// uniformly random subset).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `n` exceeds the number of candidates.
pub fn initialize_q_batch_nonneg<R: Rng + ?Sized>(
    acq_values: &[f64],
    n: usize,
    eta: f64,
    alpha: f64,
    rng: &mut R,
) -> Result<(Vec<usize>, bool)> {
    let n_samples = acq_values.len();
    if n > n_samples {
        return Err(BotorchError::InvalidArgument(
            "n cannot be larger than the number of provided samples".into(),
        ));
    }
    if n == n_samples {
        return Ok(((0..n).collect(), false));
    }
    let max_idx = argmax(acq_values);
    let max_val = acq_values[max_idx];
    if max_val.is_nan() || max_val <= 0.0 {
        return Ok((random_subset(n_samples, n, rng), true));
    }
    let pos: Vec<usize> = (0..n_samples).filter(|&i| acq_values[i] > 0.0).collect();
    if pos.len() < n {
        let rest: Vec<usize> = (0..n_samples)
            .filter(|&i| acq_values[i].is_nan() || acq_values[i] <= 0.0)
            .collect();
        let fill = random_subset(rest.len(), n - pos.len(), rng);
        let mut chosen = vec![false; n_samples];
        for &i in &pos {
            chosen[i] = true;
        }
        for &j in &fill {
            chosen[rest[j]] = true;
        }
        // BoTorch indexes X with a boolean mask: original order.
        return Ok(((0..n_samples).filter(|&i| chosen[i]).collect(), false));
    }
    let mut alpha = alpha;
    let mut alpha_pos: Vec<usize>;
    loop {
        alpha_pos = (0..n_samples)
            .filter(|&i| acq_values[i] >= alpha * max_val)
            .collect();
        if alpha_pos.len() >= n {
            break;
        }
        alpha *= 0.1;
    }
    let weights: Vec<f64> = alpha_pos
        .iter()
        .map(|&i| (eta * (acq_values[i] / max_val - 1.0)).exp())
        .collect();
    let mut idcs: Vec<usize> = weighted_sample_without_replacement(&weights, n, rng)
        .into_iter()
        .map(|j| alpha_pos[j])
        .collect();
    if !idcs.contains(&max_idx) {
        idcs[n - 1] = max_idx;
    }
    Ok((idcs, false))
}

/// Sets the fixed features of a candidate set in place.
pub(crate) fn apply_fixed_features(x: &mut Matrix, fixed: &BTreeMap<usize, f64>) {
    for (&k, &v) in fixed {
        for i in 0..x.nrows() {
            x[(i, k)] = v;
        }
    }
}

/// Settings of [`gen_batch_initial_conditions`] (the initialization-related
/// entries of BoTorch's `options` dictionary).
#[derive(Clone, Debug, PartialEq)]
pub struct InitialConditionOptions {
    /// Seed of the Sobol raw samples (`options["seed"]`); `None` draws a
    /// random seed.
    pub seed: Option<u64>,
    /// Boltzmann temperature (`options["eta"]`, default 1).
    pub eta: f64,
    /// Threshold fraction of the non-negative heuristic (`options["alpha"]`,
    /// default `1e-4`).
    pub alpha: f64,
    /// Force [`initialize_q_batch_nonneg`] (`options["nonnegative"]`); it is
    /// also used whenever the acquisition function reports
    /// [`AcquisitionFunction::is_nonnegative`].
    pub nonnegative: bool,
    /// Number of raw candidate sets evaluated per chunk
    /// (`options["init_batch_limit"]`, falling back to `batch_limit`); `None`
    /// evaluates all at once. Chunking does not change the result.
    pub batch_limit: Option<usize>,
    /// Features fixed to given values (`fixed_features`).
    pub fixed_features: BTreeMap<usize, f64>,
    /// Linear inequality constraints (`Σ c x ≥ rhs`): the raw samples are
    /// drawn from the constraint polytope.
    pub inequality_constraints: Vec<LinearConstraint>,
    /// Linear equality constraints (`Σ c x = rhs`).
    pub equality_constraints: Vec<LinearConstraint>,
    /// Burn-in steps of the polytope sampler (`options["n_burnin"]`).
    pub n_burnin: usize,
    /// Thinning of the polytope sampler (`options["n_thinning"]`).
    pub n_thinning: usize,
    /// Select the `num_restarts` raw samples with the top values
    /// ([`initialize_q_batch_topn`], `options["topn"]`) instead of the
    /// Boltzmann heuristics.
    pub topn: Option<TopnOptions>,
    /// Add `raw_samples` sets of points sampled around the best points to
    /// the raw samples (`options["sample_around_best"]`).
    pub sample_around_best: Option<SampleAroundBest>,
}

impl Default for InitialConditionOptions {
    fn default() -> Self {
        Self {
            seed: None,
            eta: DEFAULT_ETA,
            alpha: DEFAULT_ALPHA,
            nonnegative: false,
            batch_limit: None,
            fixed_features: BTreeMap::new(),
            inequality_constraints: Vec::new(),
            equality_constraints: Vec::new(),
            n_burnin: DEFAULT_N_BURNIN,
            n_thinning: DEFAULT_N_THINNING,
            topn: None,
            sample_around_best: None,
        }
    }
}

/// Settings of [`initialize_q_batch_topn`] (`options["largest"]` and
/// `options["sorted"]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopnOptions {
    /// Take the largest values (default: the acquisition function's
    /// `maximize`, else true).
    pub largest: bool,
    /// Return the selection sorted by value (default true).
    pub sorted: bool,
}

impl Default for TopnOptions {
    fn default() -> Self {
        Self {
            largest: true,
            sorted: true,
        }
    }
}

/// Sampling around the best points (BoTorch's `sample_around_best` options):
/// the best points (`best_pct` percent of the baseline points by their
/// predicted objective, or the Pareto set; found by the caller like BoTorch's
/// `sample_points_around_best`) and the perturbation settings.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleAroundBest {
    /// The best points (`n_best × d`).
    pub best_x: Matrix,
    /// Standard deviation of the truncated normal perturbations of all
    /// dimensions (`options["sample_around_best_sigma"]`, default 1e-3).
    pub sigma: f64,
    /// Standard deviation of the perturbations of a subset of the dimensions
    /// (`options["sample_around_best_subset_sigma"]`, default 0.1).
    pub subset_sigma: f64,
    /// Probability of perturbing a dimension in the subset perturbations
    /// (`options["sample_around_best_prob_perturb"]`; default `min(20 / d,
    /// 1)`). Given, or with `d >= 20`, half of the points come from subset
    /// perturbations.
    pub prob_perturb: Option<f64>,
}

impl SampleAroundBest {
    /// The best points with BoTorch's default perturbation settings.
    pub fn new(best_x: Matrix) -> Self {
        Self {
            best_x,
            sigma: 1e-3,
            subset_sigma: 1e-1,
            prob_perturb: None,
        }
    }
}

/// The `n` largest (or smallest) values' indices with torch's `topk`
/// semantics (NaN is the largest value, ties keep the index order); sorted by
/// value if `sorted`, else in index order.
pub(crate) fn topk_indices(values: &[f64], n: usize, largest: bool, sorted: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    let key = |i: usize| if largest { values[i] } else { -values[i] };
    idx.sort_by(|&a, &b| {
        let (va, vb) = (key(a), key(b));
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
    idx.truncate(n);
    if !sorted {
        idx.sort_unstable();
    }
    idx
}

/// Selection of the initial conditions with the top values (BoTorch's
/// `initialize_q_batch_topn`): the `n` largest values (smallest if not
/// `largest`). Returns the indices and whether BoTorch's
/// `BadInitialCandidatesWarning` fallback was used (all values equal: a
/// uniformly random subset).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `n` exceeds the number of candidates.
pub fn initialize_q_batch_topn<R: Rng + ?Sized>(
    acq_values: &[f64],
    n: usize,
    options: TopnOptions,
    rng: &mut R,
) -> Result<(Vec<usize>, bool)> {
    let n_samples = acq_values.len();
    if n > n_samples {
        return Err(BotorchError::InvalidArgument(format!(
            "n ({n}) cannot be larger than the number of provided samples ({n_samples})"
        )));
    }
    if n == n_samples {
        return Ok(((0..n).collect(), false));
    }
    if std_unbiased(acq_values) == 0.0 {
        return Ok((random_subset(n_samples, n, rng), true));
    }
    Ok((
        topk_indices(acq_values, n, options.largest, options.sorted),
        false,
    ))
}

/// `torch.distributions.Normal(0, 1).cdf`: `0.5 (1 + erf(x / √2))`.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

/// `torch.distributions.Normal(0, 1).icdf`: `erfinv(2 p - 1) √2`.
fn normal_icdf(p: f64) -> f64 {
    statrs::function::erf::erf_inv(2.0 * p - 1.0) * std::f64::consts::SQRT_2
}

/// The column ranges of `bounds` with BoTorch's handling of constant bounds
/// (`_update_constant_bounds`: an upper bound equal to the lower bound is
/// replaced by `lower + 1`).
fn ranges(bounds: &Matrix) -> Vec<(f64, f64)> {
    (0..bounds.ncols())
        .map(|j| {
            let (lo, hi) = (bounds[(0, j)], bounds[(1, j)]);
            (lo, if hi == lo { lo + 1.0 } else { hi })
        })
        .collect()
}

/// `n` uniform points (`n × d`): a scrambled Sobol sequence (seeded from
/// `rng`) if `qmc`, else i.i.d.
fn uniform_points<R: Rng + ?Sized>(n: usize, d: usize, qmc: bool, rng: &mut R) -> Result<Matrix> {
    if qmc {
        let unit = Matrix::from_fn(2, d, |i, _| i as f64);
        let seed = rng.random::<u64>();
        let sets = draw_sobol_samples(&unit, n, 1, Some(seed))?;
        Ok(Matrix::from_fn(n, d, |i, j| sets[i][(0, j)]))
    } else {
        Ok(Matrix::from_fn(n, d, |_, _| rng.random::<f64>()))
    }
}

/// The deterministic part of [`sample_truncated_normal_perturbations`]: the
/// points `centers` (`n × d`, raw space) perturbed by truncated normal draws
/// obtained by the inverse transform of the uniform draws `u` (`n × d`).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inconsistent shapes.
pub fn truncated_normal_perturbations_from_uniform(
    centers: &Matrix,
    u: &Matrix,
    sigma: f64,
    bounds: &Matrix,
) -> Result<Matrix> {
    let d = bounds.ncols();
    if centers.ncols() != d || u.shape() != centers.shape() {
        return Err(BotorchError::DimensionMismatch(format!(
            "centers ({}x{}) and uniform draws ({}x{}) must be n x {d}",
            centers.nrows(),
            centers.ncols(),
            u.nrows(),
            u.ncols()
        )));
    }
    let r = ranges(bounds);
    Ok(Matrix::from_fn(centers.nrows(), d, |i, j| {
        let (lo, hi) = r[j];
        let x = (centers[(i, j)] - lo) / (hi - lo);
        let alpha = -x / sigma;
        let beta = (1.0 - x) / sigma;
        let cdf_alpha = normal_cdf(alpha);
        let p = cdf_alpha + u[(i, j)] * (normal_cdf(beta) - cdf_alpha);
        let pert = normal_icdf(p) * sigma;
        let v = (x + pert).clamp(0.0, 1.0);
        v * (hi - lo) + lo
    }))
}

/// `n_discrete_points` points around the points `x` (`n × d`): a randomly
/// chosen point of `x` (the only one if `x` has one row) perturbed by
/// `N(0, sigma² I)` noise (relative to the bounds) truncated to stay within
/// `bounds` (BoTorch's `sample_truncated_normal_perturbations`; quasi-random
/// uniform draws if `qmc`).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inconsistent shapes and
/// [`BotorchError::InvalidArgument`] for an empty `x` or too many points.
pub fn sample_truncated_normal_perturbations<R: Rng + ?Sized>(
    x: &Matrix,
    n_discrete_points: usize,
    sigma: f64,
    bounds: &Matrix,
    qmc: bool,
    rng: &mut R,
) -> Result<Matrix> {
    let d = check_bounds(bounds)?;
    if x.ncols() != d || x.nrows() == 0 {
        return Err(BotorchError::DimensionMismatch(format!(
            "X must be a non-empty n x {d} matrix, got {}x{}",
            x.nrows(),
            x.ncols()
        )));
    }
    let rows: Vec<usize> = if x.nrows() > 1 {
        (0..n_discrete_points)
            .map(|_| rng.random_range(0..x.nrows()))
            .collect()
    } else {
        vec![0; n_discrete_points]
    };
    let centers = Matrix::from_fn(n_discrete_points, d, |i, j| x[(rows[i], j)]);
    let u = uniform_points(n_discrete_points, d, qmc, rng)?;
    truncated_normal_perturbations_from_uniform(&centers, &u, sigma, bounds)
}

/// `n_discrete_points` points around the points `x` (`n × d`, raw space)
/// that perturb only a subset of the dimensions (BoTorch's
/// `sample_perturbed_subset_dims`): every dimension of a randomly chosen
/// point is replaced by a truncated normal perturbation of itself with
/// probability `prob_perturb` (default `min(20 / d, 1)`); points without a
/// perturbed dimension get `ceil(d · prob_perturb)` random dimensions
/// perturbed.
///
/// Unlike BoTorch, every point is perturbed around itself: BoTorch draws the
/// perturbations around a second, independently chosen point of `x` when
/// `x` has several rows, which mixes the coordinates of two different points
/// (see the documentation of the port).
///
/// # Errors
/// See [`sample_truncated_normal_perturbations`].
pub fn sample_perturbed_subset_dims<R: Rng + ?Sized>(
    x: &Matrix,
    bounds: &Matrix,
    n_discrete_points: usize,
    sigma: f64,
    qmc: bool,
    prob_perturb: Option<f64>,
    rng: &mut R,
) -> Result<Matrix> {
    let d = check_bounds(bounds)?;
    if x.ncols() != d || x.nrows() == 0 {
        return Err(BotorchError::DimensionMismatch(format!(
            "X must be a non-empty n x {d} matrix, got {}x{}",
            x.nrows(),
            x.ncols()
        )));
    }
    let prob = prob_perturb.unwrap_or_else(|| (20.0 / d as f64).min(1.0));
    let rows: Vec<usize> = if x.nrows() > 1 {
        (0..n_discrete_points)
            .map(|_| rng.random_range(0..x.nrows()))
            .collect()
    } else {
        vec![0; n_discrete_points]
    };
    let mut cand = Matrix::from_fn(n_discrete_points, d, |i, j| x[(rows[i], j)]);
    let u = uniform_points(n_discrete_points, d, qmc, rng)?;
    let pert = truncated_normal_perturbations_from_uniform(&cand, &u, sigma, bounds)?;
    let mut mask: Vec<Vec<bool>> = (0..n_discrete_points)
        .map(|_| (0..d).map(|_| rng.random::<f64>() <= prob).collect())
        .collect();
    let n_perturb = (d as f64 * prob).ceil() as usize;
    for row in &mut mask {
        if row.iter().all(|&m| !m) {
            let perm = random_subset(d, d, rng);
            for (pos, &k) in perm.iter().enumerate() {
                row[k] = pos < n_perturb;
            }
        }
    }
    for (i, row) in mask.iter().enumerate() {
        for (j, &m) in row.iter().enumerate() {
            if m {
                cand[(i, j)] = pert[(i, j)];
            }
        }
    }
    Ok(cand)
}

/// `n_discrete_points` points around the best points (the second half of
/// BoTorch's `sample_points_around_best`, after the best points are found):
/// truncated normal perturbations of all dimensions with `sigma`, and, if
/// `prob_perturb` is given or `d >= 20`, half of the points from
/// [`sample_perturbed_subset_dims`] with `subset_sigma`, shuffled.
///
/// BoTorch uses `sigma` for the subset perturbations as well (its
/// `subset_sigma` argument is unused); this port uses `subset_sigma`, as
/// documented by BoTorch.
///
/// # Errors
/// See [`sample_truncated_normal_perturbations`].
pub fn sample_points_around<R: Rng + ?Sized>(
    best: &SampleAroundBest,
    n_discrete_points: usize,
    bounds: &Matrix,
    rng: &mut R,
) -> Result<Matrix> {
    let d = bounds.ncols();
    let use_perturbed = best.best_x.ncols() >= 20 || best.prob_perturb.is_some();
    let n_trunc = if use_perturbed {
        n_discrete_points / 2
    } else {
        n_discrete_points
    };
    let trunc = sample_truncated_normal_perturbations(
        &best.best_x,
        n_trunc,
        best.sigma,
        bounds,
        true,
        rng,
    )?;
    if !use_perturbed {
        return Ok(trunc);
    }
    let subset = sample_perturbed_subset_dims(
        &best.best_x,
        bounds,
        n_discrete_points - n_trunc,
        best.subset_sigma,
        true,
        best.prob_perturb,
        rng,
    )?;
    let all: Vec<usize> = random_subset(n_discrete_points, n_discrete_points, rng);
    Ok(Matrix::from_fn(n_discrete_points, d, |i, j| {
        let k = all[i];
        if k < n_trunc {
            trunc[(k, j)]
        } else {
            subset[(k - n_trunc, j)]
        }
    }))
}

/// Constraints on the `q·d` flattened entries of a q-batch (BoTorch's
/// `transform_constraints`): an intra-point constraint becomes `q`
/// constraints, an inter-point constraint one.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for feature indices out of range.
pub fn transform_constraints(
    constraints: &[LinearConstraint],
    q: usize,
    d: usize,
) -> Result<Vec<LinearConstraint>> {
    let mut out = Vec::new();
    for c in constraints {
        match c.indices() {
            ConstraintIndices::IntraPoint(idx) => {
                if idx.iter().any(|&k| k >= d) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "Constraint indices cannot exceed the problem dimension d={d}."
                    )));
                }
                for i in 0..q {
                    out.push(LinearConstraint::intra_point(
                        idx.iter().map(|&j| i * d + j).collect(),
                        c.coefficients().to_vec(),
                        c.rhs(),
                    )?);
                }
            }
            ConstraintIndices::InterPoint(idx) => {
                if idx.iter().any(|&(_, k)| k >= d) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "Constraint indices cannot exceed the problem dimension d={d}."
                    )));
                }
                out.push(LinearConstraint::intra_point(
                    idx.iter().map(|&(p, k)| p * d + k).collect(),
                    c.coefficients().to_vec(),
                    c.rhs(),
                )?);
            }
        }
    }
    Ok(out)
}

/// `n` q-batches (`q × d` each) sampled from the polytope given by `bounds`
/// and linear constraints (BoTorch's `sample_q_batches_from_polytope`):
/// `n·q` points of one hit-and-run chain, or with inter-point constraints `n`
/// points of a chain on the `q·d`-dimensional space (thinning `n_thinning ·
/// q`).
///
/// # Errors
/// See [`get_polytope_samples`].
#[allow(clippy::too_many_arguments)]
pub fn sample_q_batches_from_polytope(
    n: usize,
    q: usize,
    bounds: &Matrix,
    n_burnin: usize,
    n_thinning: usize,
    seed: Option<u64>,
    inequality_constraints: &[LinearConstraint],
    equality_constraints: &[LinearConstraint],
) -> Result<Vec<Matrix>> {
    let d = bounds.ncols();
    let inter_point = inequality_constraints
        .iter()
        .chain(equality_constraints)
        .any(|c| c.is_inter_point());
    let samples = if inter_point {
        let wide = Matrix::from_fn(2, q * d, |i, j| bounds[(i, j % d)]);
        get_polytope_samples(
            n,
            &wide,
            &transform_constraints(inequality_constraints, q, d)?,
            &transform_constraints(equality_constraints, q, d)?,
            seed,
            n_burnin,
            n_thinning * q,
        )?
    } else {
        get_polytope_samples(
            n * q,
            bounds,
            inequality_constraints,
            equality_constraints,
            seed,
            n_burnin,
            n_thinning,
        )?
    };
    Ok((0..n)
        .map(|i| {
            Matrix::from_fn(q, d, |p, k| {
                let flat = (i * q + p) * d + k;
                samples[(flat / samples.ncols(), flat % samples.ncols())]
            })
        })
        .collect())
}

/// Generates `num_restarts` initial candidate sets (`q × d`) for multi-start
/// optimization (BoTorch's `gen_batch_initial_conditions`):
///
/// 1. draw `n = raw_samples` scrambled Sobol candidate sets in `bounds`
///    ([`draw_sobol_samples`]) -- or, with linear constraints, hit-and-run
///    samples from the constraint polytope
///    ([`sample_q_batches_from_polytope`]) -- and set the fixed features;
/// 2. evaluate the acquisition function on all of them (in parallel);
/// 3. select `num_restarts` of them with [`initialize_q_batch_nonneg`] if the
///    acquisition function is non-negative (or `options.nonnegative`), and
///    [`initialize_q_batch`] otherwise.
///
/// If the selection falls back to random choices (all values equal,
/// non-finite or non-positive), the procedure is repeated with `2, 3, 4 ×
/// raw_samples` points and the seed incremented, as in BoTorch. `rng` drives
/// the random selection (BoTorch uses torch's global RNG).
///
/// Returns the initial conditions and whether the last attempt still used the
/// random fallback.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] / [`BotorchError::InvalidArgument`] for
/// invalid bounds or sizes, and errors of the acquisition function.
#[allow(clippy::too_many_arguments)]
pub fn gen_batch_initial_conditions<R: Rng + ?Sized>(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    options: &InitialConditionOptions,
    rng: &mut R,
) -> Result<(Vec<Matrix>, bool)> {
    gen_batch_initial_conditions_with_fantasies(
        acq_function,
        bounds,
        q,
        num_restarts,
        raw_samples,
        options,
        None,
        rng,
    )
}

/// [`gen_batch_initial_conditions`] with BoTorch's `fixed_X_fantasies`: the
/// fixed points (`k × d`, e.g. the fantasy designs of a one-shot acquisition
/// function) are appended to every raw candidate set before it is evaluated,
/// so the initial conditions are `(q + k) × d`.
///
/// # Errors
/// See [`gen_batch_initial_conditions`]; [`BotorchError::DimensionMismatch`]
/// if the fixed points do not have `d` columns (BoTorch's message).
#[allow(clippy::too_many_arguments)]
pub fn gen_batch_initial_conditions_with_fantasies<R: Rng + ?Sized>(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    options: &InitialConditionOptions,
    fixed_x_fantasies: Option<&Matrix>,
    rng: &mut R,
) -> Result<(Vec<Matrix>, bool)> {
    let d = check_bounds(bounds)?;
    if let Some(f) = fixed_x_fantasies {
        if f.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "`fixed_X_fantasies` and `bounds` must both have the same trailing \
                 dimension `d`, but have {} and {d}, respectively.",
                f.ncols()
            )));
        }
    }
    if d != acq_function.dim() {
        return Err(BotorchError::DimensionMismatch(format!(
            "bounds have {d} columns but the acquisition function has d = {}",
            acq_function.dim()
        )));
    }
    if num_restarts == 0 || raw_samples == 0 || q == 0 {
        return Err(BotorchError::InvalidArgument(
            "q, num_restarts and raw_samples must be positive".into(),
        ));
    }
    if let Some(&k) = options.fixed_features.keys().find(|&&k| k >= d) {
        return Err(BotorchError::InvalidArgument(format!(
            "fixed feature index {k} out of range for d = {d}"
        )));
    }
    if options.sample_around_best.is_some() && !options.equality_constraints.is_empty() {
        return Err(BotorchError::Unsupported(
            "Option 'sample_around_best' is not supported when equalityconstraints are present."
                .into(),
        ));
    }
    if let Some(sab) = &options.sample_around_best {
        if sab.best_x.ncols() != d || sab.best_x.nrows() == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "the best points must be a non-empty n x {d} matrix, got {}x{}",
                sab.best_x.nrows(),
                sab.best_x.ncols()
            )));
        }
    }
    let nonneg = options.nonnegative || acq_function.is_nonnegative();
    let mut seed = options.seed;
    let (factor_max, mut factor) = (5usize, 1usize);
    let mut last = None;
    while factor < factor_max {
        let n = raw_samples * factor;
        let mut x_rnd = if options.inequality_constraints.is_empty()
            && options.equality_constraints.is_empty()
        {
            draw_sobol_samples(bounds, n, q, seed)?
        } else {
            sample_q_batches_from_polytope(
                n,
                q,
                bounds,
                options.n_burnin,
                options.n_thinning,
                seed,
                &options.inequality_constraints,
                &options.equality_constraints,
            )?
        };
        if let Some(sab) = &options.sample_around_best {
            let around = sample_points_around(sab, n * q, bounds, rng)?;
            for i in 0..n {
                x_rnd.push(around.rows(i * q, q).into_owned());
            }
        }
        for x in &mut x_rnd {
            apply_fixed_features(x, &options.fixed_features);
            if let Some(f) = fixed_x_fantasies {
                let mut full = Matrix::zeros(q + f.nrows(), d);
                full.rows_mut(0, q).copy_from(x);
                full.rows_mut(q, f.nrows()).copy_from(f);
                *x = full;
            }
        }
        let chunk = options.batch_limit.unwrap_or(x_rnd.len()).max(1);
        let mut acq_vals = Vec::with_capacity(x_rnd.len());
        for c in x_rnd.chunks(chunk) {
            acq_vals.extend(acq_function.forward_many(c)?);
        }
        let (idcs, fallback) = if let Some(topn) = options.topn {
            initialize_q_batch_topn(&acq_vals, num_restarts, topn, rng)?
        } else if nonneg {
            initialize_q_batch_nonneg(&acq_vals, num_restarts, options.eta, options.alpha, rng)?
        } else {
            initialize_q_batch(&acq_vals, num_restarts, options.eta, rng)?
        };
        let ics: Vec<Matrix> = idcs.iter().map(|&i| x_rnd[i].clone()).collect();
        if !fallback {
            return Ok((ics, false));
        }
        last = Some(ics);
        factor += 1;
        seed = seed.map(|s| s.wrapping_add(1));
    }
    Ok((last.expect("at least one attempt"), true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::Xoshiro256PlusPlus;
    use rand::SeedableRng;

    #[test]
    fn boltzmann_selection_prefers_large_values_and_keeps_the_best() {
        let vals: Vec<f64> = (0..200).map(|i| (i as f64 / 20.0).sin()).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let mut hits = vec![0usize; 200];
        for _ in 0..200 {
            let (idx, fallback) = initialize_q_batch(&vals, 10, 1.0, &mut rng).unwrap();
            assert!(!fallback);
            assert_eq!(idx.len(), 10);
            let mut s = idx.clone();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), 10, "without replacement");
            assert!(idx.contains(&argmax(&vals)));
            for i in idx {
                hits[i] += 1;
            }
        }
        // Points with high values are selected more often than low ones.
        let high: usize = (0..200).filter(|&i| vals[i] > 0.5).map(|i| hits[i]).sum();
        let low: usize = (0..200).filter(|&i| vals[i] < -0.5).map(|i| hits[i]).sum();
        assert!(high > 3 * low, "high {high} low {low}");
        let (idx, fallback) = initialize_q_batch(&[1.0; 5], 2, 1.0, &mut rng).unwrap();
        assert!(fallback && idx.len() == 2);
        assert!(initialize_q_batch(&vals, 300, 1.0, &mut rng).is_err());
        // A large eta makes exp(eta * z) overflow; eta is halved until it does not.
        let mut outlier = vec![0.0; 100];
        outlier[17] = 1e3;
        let (idx, fallback) = initialize_q_batch(&outlier, 4, 1e3, &mut rng).unwrap();
        assert!(!fallback && idx.contains(&17) && idx.len() == 4);
        assert_eq!(
            initialize_q_batch(&vals[..3], 3, 1.0, &mut rng).unwrap().0,
            vec![0, 1, 2]
        );
    }

    #[test]
    fn nonnegative_selection() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        // Fewer positive values than restarts: all positives are taken.
        let vals = [0.0, 0.5, 0.0, 0.2, 0.0, 0.0];
        let (idx, fallback) = initialize_q_batch_nonneg(&vals, 3, 1.0, 1e-4, &mut rng).unwrap();
        assert!(!fallback);
        assert!(idx.contains(&1) && idx.contains(&3) && idx.len() == 3);
        // All non-positive: random fallback.
        let (_, fallback) = initialize_q_batch_nonneg(&[0.0; 4], 2, 1.0, 1e-4, &mut rng).unwrap();
        assert!(fallback);
        // Values below alpha * max are ignored.
        let vals: Vec<f64> = (0..50)
            .map(|i| if i < 10 { 1.0 + i as f64 } else { 1e-9 })
            .collect();
        for _ in 0..20 {
            let (idx, _) = initialize_q_batch_nonneg(&vals, 5, 1.0, 1e-4, &mut rng).unwrap();
            assert!(idx.iter().all(|&i| i < 10));
            assert!(idx.contains(&9));
        }
    }
}
