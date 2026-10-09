//! Initial conditions for multi-start acquisition optimization, mirroring
//! `botorch.optim.initializers`: [`gen_batch_initial_conditions`] draws
//! `raw_samples` scrambled Sobol candidate sets, evaluates the acquisition
//! function on all of them and selects `num_restarts` starting points with
//! [`initialize_q_batch`] (Boltzmann sampling) or
//! [`initialize_q_batch_nonneg`] (for non-negative acquisition functions).

use std::collections::BTreeMap;

use rand::{Rng, RngExt};

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::utils::sampling::{check_bounds, draw_sobol_samples};
use crate::Matrix;

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
        }
    }
}

/// Generates `num_restarts` initial candidate sets (`q × d`) for multi-start
/// optimization (BoTorch's `gen_batch_initial_conditions` without constraints):
///
/// 1. draw `n = raw_samples` scrambled Sobol candidate sets in `bounds`
///    ([`draw_sobol_samples`]) and set the fixed features;
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
    let d = check_bounds(bounds)?;
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
    let nonneg = options.nonnegative || acq_function.is_nonnegative();
    let mut seed = options.seed;
    let (factor_max, mut factor) = (5usize, 1usize);
    let mut last = None;
    while factor < factor_max {
        let n = raw_samples * factor;
        let mut x_rnd = draw_sobol_samples(bounds, n, q, seed)?;
        for x in &mut x_rnd {
            apply_fixed_features(x, &options.fixed_features);
        }
        let chunk = options.batch_limit.unwrap_or(n).max(1);
        let mut acq_vals = Vec::with_capacity(n);
        for c in x_rnd.chunks(chunk) {
            acq_vals.extend(acq_function.forward_many(c)?);
        }
        let (idcs, fallback) = if nonneg {
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
