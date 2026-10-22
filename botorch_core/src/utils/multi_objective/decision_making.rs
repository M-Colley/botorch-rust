//! Preference-aware a posteriori selection of Schmitt et al. (2022),
//! <https://doi.org/10.3390/inventions7030046>, equations (8)–(13).
//!
//! Objective values follow BoTorch's maximization convention by default at
//! the caller; `maximize = false` accepts the paper's costs. Dominated rows
//! and duplicate rows (after the first) are removed before normalization.
//! The signed distance is essential: a point can lie above the distance
//! plane, so even `alpha = 0` need not include the entire Pareto front.

use super::pareto::is_non_dominated;
use super::PointBatch;
use crate::{BotorchError, Matrix, Result};

/// Geometry of a knee region. Indices always refer to the original rows.
#[derive(Clone, Debug, PartialEq)]
pub struct KneeRegion {
    /// First-occurrence, non-dominated indices.
    pub pareto_indices: Vec<usize>,
    /// Normalized minimization costs of the Pareto rows (flat axes are zero).
    pub normalized_costs: Matrix,
    /// Original index defining the distance plane (first maximum norm).
    pub distance_plane_index: usize,
    /// Signed distances for the Pareto rows, in `pareto_indices` order.
    pub signed_distances: Vec<f64>,
    /// Inclusion mask over the original input rows.
    pub mask: Vec<bool>,
}

/// Normalize Pareto costs and determine the knee region (equations 8–11).
/// `alpha` must be finite and lie in `[0, 1]`. A larger value narrows the
/// region toward the maximum signed distance. All ties keep input order.
pub fn analyze_knee_region(y: &Matrix, alpha: f64, maximize: bool) -> Result<KneeRegion> {
    if y.nrows() == 0 || y.ncols() == 0 {
        return Err(BotorchError::InvalidArgument(
            "Y must have at least one row and one objective".into(),
        ));
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(BotorchError::InvalidArgument(
            "alpha must be finite and in [0, 1]".into(),
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(BotorchError::InputData(
            "Y must contain only finite values".into(),
        ));
    }
    let pareto_indices: Vec<_> = is_non_dominated(y, maximize, true)
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| keep.then_some(i))
        .collect();
    let m = y.ncols();
    let cost = |i: usize, j: usize| if maximize { -y[(i, j)] } else { y[(i, j)] };
    let mut z = Matrix::zeros(pareto_indices.len(), m);
    for j in 0..m {
        let lo = pareto_indices
            .iter()
            .map(|&i| cost(i, j))
            .fold(f64::INFINITY, f64::min);
        let hi = pareto_indices
            .iter()
            .map(|&i| cost(i, j))
            .fold(f64::NEG_INFINITY, f64::max);
        if hi > lo {
            // Scaling first also handles finite inputs whose range overflows.
            let scale = lo.abs().max(hi.abs()).max(1.0);
            let a = lo / scale;
            let range = hi / scale - a;
            for (k, &i) in pareto_indices.iter().enumerate() {
                z[(k, j)] = ((cost(i, j) / scale - a) / range).clamp(0.0, 1.0);
            }
        }
    }
    let mut q = 0;
    let mut largest = -1.0;
    for k in 0..z.nrows() {
        let norm_squared: f64 = z.row(k).iter().map(|v| v * v).sum();
        if norm_squared > largest {
            largest = norm_squared;
            q = k;
        }
    }
    let plane_sum: f64 = z.row(q).iter().sum();
    let signed_distances: Vec<_> = (0..z.nrows())
        .map(|k| (plane_sum - z.row(k).iter().sum::<f64>()) / (m as f64).sqrt())
        .collect();
    let threshold = alpha * signed_distances.iter().copied().fold(0.0, f64::max);
    let mut mask = vec![false; y.nrows()];
    for (&i, &delta) in pareto_indices.iter().zip(&signed_distances) {
        mask[i] = delta >= threshold;
    }
    Ok(KneeRegion {
        distance_plane_index: pareto_indices[q],
        pareto_indices,
        normalized_costs: z,
        signed_distances,
        mask,
    })
}

/// Knee-region mask over the original input rows.
pub fn knee_region(y: &Matrix, alpha: f64, maximize: bool) -> Result<Vec<bool>> {
    Ok(analyze_knee_region(y, alpha, maximize)?.mask)
}

/// Select the first minimum of `p · normalized_costs` in the knee region
/// (the supporting preference plane in equations 12–13). Preferences must
/// be finite, nonnegative, match the objective count and have positive sum.
pub fn select_preferred(
    y: &Matrix,
    preferences: &[f64],
    alpha: f64,
    maximize: bool,
) -> Result<usize> {
    if preferences.len() != y.ncols() {
        return Err(BotorchError::DimensionMismatch(
            "preferences must match the number of objectives".into(),
        ));
    }
    if preferences.iter().any(|v| !v.is_finite() || *v < 0.0)
        || !preferences.iter().any(|v| *v > 0.0)
    {
        return Err(BotorchError::InvalidArgument(
            "preferences must be finite, nonnegative and not all zero".into(),
        ));
    }
    let region = analyze_knee_region(y, alpha, maximize)?;
    let scale = preferences.iter().copied().fold(0.0, f64::max);
    let mut best = None;
    let mut best_score = f64::INFINITY;
    for (k, &i) in region.pareto_indices.iter().enumerate() {
        if region.mask[i] {
            let score: f64 = preferences
                .iter()
                .enumerate()
                .map(|(j, p)| (p / scale) * region.normalized_costs[(k, j)])
                .sum();
            if score < best_score {
                best = Some(i);
                best_score = score;
            }
        }
    }
    best.ok_or_else(|| BotorchError::Numerical("knee region is empty".into()))
}

/// Independent knee regions for a `batch_shape × n × m` point batch.
pub fn knee_region_batch(y: &PointBatch, alpha: f64, maximize: bool) -> Result<Vec<bool>> {
    // Validate even an empty batch, rather than silently accepting bad options.
    if y.batch_size() == 0 {
        return Err(BotorchError::InvalidArgument(
            "Y must contain at least one point set".into(),
        ));
    }
    let mut masks = Vec::with_capacity(y.batch_size() * y.n());
    for b in 0..y.batch_size() {
        masks.extend(knee_region(&y.matrix(b), alpha, maximize)?);
    }
    Ok(masks)
}

/// Independent preferred indices for a point batch and shared preferences.
pub fn select_preferred_batch(
    y: &PointBatch,
    preferences: &[f64],
    alpha: f64,
    maximize: bool,
) -> Result<Vec<usize>> {
    if y.batch_size() == 0 {
        return Err(BotorchError::InvalidArgument(
            "Y must contain at least one point set".into(),
        ));
    }
    (0..y.batch_size())
        .map(|b| select_preferred(&y.matrix(b), preferences, alpha, maximize))
        .collect()
}
