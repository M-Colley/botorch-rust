//! Hypervolume utilities (`botorch.utils.multi_objective.hypervolume`):
//! [`Hypervolume`] (the dimension-sweep algorithm of Fonseca et al.),
//! [`infer_reference_point`] and [`get_hypervolume_maximizing_subset`].
//!
//! References:
//!
//! * C. M. Fonseca, L. Paquete, and M. López-Ibáñez. *An improved
//!   dimension-sweep algorithm for the hypervolume indicator.* IEEE Congress
//!   on Evolutionary Computation, pp. 1157–1163, 2006.
//! * H. Ishibuchi, N. Akedo, and Y. Nojima. *A many-objective test problem
//!   for visually examining diversity maintenance behavior in a decision
//!   space.* GECCO 2011 (the 0.1 scale of [`infer_reference_point`]).
//!
//! The NEHVI mixins of BoTorch's module belong to the acquisition functions:
//! `compute_subset_indices` and `SubsetIndexCachingMixin` are re-exported
//! here from [`crate::acquisition::multi_objective::hypervolume`];
//! `NoisyExpectedHypervolumeMixin` belongs to the noisy hypervolume
//! acquisition functions.

pub use crate::acquisition::multi_objective::hypervolume::{
    compute_subset_indices, SubsetIndexCache,
};

use rayon::prelude::*;

use super::box_decompositions::{BoxDecomposition, DominatedPartitioning};
use super::{clamp_min, nan_max, nan_min, row_major, PointBatch};
use crate::error::{BotorchError, Result};
use crate::Matrix;

/// BoTorch's `MIN_Y_RANGE`: the smallest objective range used by
/// [`infer_reference_point`].
pub const MIN_Y_RANGE: f64 = 1e-7;

/// BoTorch's `infer_reference_point(pareto_Y, max_ref_point, scale,
/// scale_max_ref_point)` (maximization): `nadir - scale · range` unless some
/// Pareto point is better than `max_ref_point`, in which case
/// `max_ref_point` (or `max_ref_point - scale · range` with
/// `scale_max_ref_point`). Elements of `max_ref_point` may be NaN (treated as
/// not given) unless `pareto_y` is empty.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `pareto_y` is empty and
/// `max_ref_point` is not given or contains NaN (BoTorch's `BotorchError`),
/// [`BotorchError::DimensionMismatch`] if the lengths disagree.
pub fn infer_reference_point(
    pareto_y: &Matrix,
    max_ref_point: Option<&[f64]>,
    scale: f64,
    scale_max_ref_point: bool,
) -> Result<Vec<f64>> {
    let (n, m) = pareto_y.shape();
    if let Some(r) = max_ref_point {
        if r.len() != m && n > 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "max_ref_point has {} elements, expected {m}",
                r.len()
            )));
        }
    }
    if n == 0 {
        let Some(max_ref) = max_ref_point else {
            return Err(BotorchError::InvalidArgument(
                "Empty pareto set and no max ref point provided".into(),
            ));
        };
        if max_ref.iter().any(|v| v.is_nan()) {
            return Err(BotorchError::InvalidArgument(
                "Empty pareto set and max ref point includes NaN.".into(),
            ));
        }
        if scale_max_ref_point {
            return Ok(max_ref.iter().map(|&r| r - scale * r.abs()).collect());
        }
        return Ok(max_ref.to_vec());
    }
    let non_nan: Vec<bool> = match max_ref_point {
        Some(r) => r.iter().map(|v| !v.is_nan()).collect(),
        None => vec![true; m],
    };
    let better: Vec<bool> = match max_ref_point {
        // Count all points exceeding the non-NaN reference values as better.
        Some(r) => (0..n)
            .map(|i| (0..m).all(|j| !non_nan[j] || pareto_y[(i, j)] > r[j]))
            .collect(),
        None => vec![true; n],
    };
    let any_better = better.iter().any(|&b| b);
    let all_non_nan = non_nan.iter().all(|&b| b);
    if let (Some(max_ref), true, true) = (max_ref_point, any_better, all_non_nan) {
        if scale_max_ref_point {
            return Ok((0..m)
                .map(|j| {
                    let best = (0..n)
                        .filter(|&i| better[i])
                        .map(|i| pareto_y[(i, j)])
                        .fold(f64::NEG_INFINITY, nan_max);
                    max_ref[j] - scale * (best - max_ref[j])
                })
                .collect());
        }
        return Ok(max_ref.to_vec());
    }
    let (mut ref_point, y_range): (Vec<f64>, Vec<f64>) = if n == 1 {
        // A single observation: subtract MIN_Y_RANGE to handle a singleton
        // Pareto set with objective values of 0.
        let y_range: Vec<f64> = (0..m)
            .map(|j| clamp_min(pareto_y[(0, j)].abs(), MIN_Y_RANGE))
            .collect();
        let r = (0..m)
            .map(|j| pareto_y[(0, j)] - scale * y_range[j])
            .collect();
        (r, y_range)
    } else {
        // Make sure that every element of the nadir point is no greater than
        // the corresponding element of max_ref_point.
        let column = |j: usize| (0..n).map(move |i| pareto_y[(i, j)]);
        let mut nadir: Vec<f64> = (0..m)
            .map(|j| column(j).fold(f64::INFINITY, nan_min))
            .collect();
        if let Some(max_ref) = max_ref_point {
            for j in 0..m {
                if non_nan[j] {
                    nadir[j] = nan_min(nadir[j], max_ref[j]);
                }
            }
        }
        let ideal: Vec<f64> = (0..m)
            .map(|j| column(j).fold(f64::NEG_INFINITY, nan_max))
            .collect();
        // Handle the case where all values of an objective are the same.
        let y_range: Vec<f64> = (0..m)
            .map(|j| clamp_min(ideal[j] - nadir[j], MIN_Y_RANGE))
            .collect();
        let r = (0..m).map(|j| nadir[j] - scale * y_range[j]).collect();
        (r, y_range)
    };
    // Set the non-NaN elements, if any, to max_ref_point.
    if let (Some(max_ref), true) = (max_ref_point, any_better) {
        if non_nan.iter().any(|&b| b) && !all_non_nan {
            for j in 0..m {
                if non_nan[j] {
                    ref_point[j] = if scale_max_ref_point {
                        max_ref[j] - scale * y_range[j]
                    } else {
                        max_ref[j]
                    };
                }
            }
        }
    }
    Ok(ref_point)
}

/// The hypervolume dominated by a set of points with respect to a reference
/// point (maximization): BoTorch's `Hypervolume`, the dimension-sweep
/// algorithm of Fonseca et al. (2006) in the variant of Simon Wessing's
/// implementation (variant 3, version 1.2) that BoTorch adapted from pymoo.
///
/// The algorithm sweeps the last objective and recurses over the remaining
/// ones, keeping the points in `m` doubly linked lists sorted by each
/// objective (`MultiList`) and caching partial areas and volumes per node;
/// two objectives end the recursion with a direct sum. Its run time is
/// `O(n^(m-2) log n)`. This is a line-by-line port of BoTorch's version
/// (including its updates of the `bounds` vector and of the cached areas),
/// so the floating-point operations, and therefore the results, are the same
/// as BoTorch's in float64.
#[derive(Clone, Debug, PartialEq)]
pub struct Hypervolume {
    ref_point: Vec<f64>,
}

impl Hypervolume {
    /// `Hypervolume(ref_point)` for maximization.
    pub fn new(ref_point: &[f64]) -> Self {
        Self {
            ref_point: ref_point.to_vec(),
        }
    }

    /// The reference point (maximization).
    pub fn ref_point(&self) -> &[f64] {
        &self.ref_point
    }

    /// Replaces the reference point.
    pub fn set_ref_point(&mut self, ref_point: &[f64]) {
        self.ref_point = ref_point.to_vec();
    }

    /// The hypervolume dominated by the rows of `pareto_y` (`n × m`); points
    /// that do not weakly dominate the reference point are ignored (as are
    /// points with NaN outcomes). `pareto_y` need not be a Pareto set.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `pareto_y` does not have `m`
    /// columns.
    pub fn compute(&self, pareto_y: &Matrix) -> Result<f64> {
        let m = self.ref_point.len();
        if pareto_y.ncols() != m {
            return Err(BotorchError::DimensionMismatch(format!(
                "pareto_Y must have the same number of objectives as ref_point. Got {}, \
                 expected {m}.",
                pareto_y.ncols()
            )));
        }
        let values = row_major(pareto_y);
        Ok(self.compute_rows(&values, pareto_y.nrows()))
    }

    /// [`Hypervolume::compute`] for `n` points stored row-major.
    ///
    /// # Panics
    /// If `pareto_y.len() != n·m`.
    pub fn compute_rows(&self, pareto_y: &[f64], n: usize) -> f64 {
        let m = self.ref_point.len();
        assert_eq!(pareto_y.len(), n * m, "expected {n} x {m} values");
        // BoTorch flips the signs and computes the hypervolume of the
        // minimization problem, with the points shifted so that the
        // reference point is the origin.
        let neg_ref: Vec<f64> = self.ref_point.iter().map(|r| -r).collect();
        let mut data = Vec::with_capacity(n * m);
        for point in pareto_y.chunks_exact(m.max(1)).take(n) {
            if point.iter().zip(&neg_ref).all(|(y, r)| -y <= *r) {
                data.extend(point.iter().zip(&neg_ref).map(|(y, r)| -y - r));
            }
        }
        if m == 0 {
            return 0.0;
        }
        let n_pareto = data.len() / m;
        let mut list = MultiList::new(data, m);
        let mut bounds = vec![f64::NEG_INFINITY; m];
        list.hv_recursive(m - 1, n_pareto, &mut bounds)
    }
}

/// BoTorch's `MultiList` and `Node`s: `m` doubly linked lists over the same
/// nodes (one list per objective, sorted by that objective), stored as
/// arrays indexed by node; node `n` is the sentinel.
struct MultiList {
    m: usize,
    sentinel: usize,
    /// `n × m` point coordinates (the sentinel has none).
    data: Vec<f64>,
    /// `(n + 1) × m` successors / predecessors per list.
    next: Vec<usize>,
    prev: Vec<usize>,
    ignore: Vec<usize>,
    /// `(n + 1) × m` cached areas and volumes.
    area: Vec<f64>,
    volume: Vec<f64>,
}

impl MultiList {
    /// `_initialize_multilist`: the nodes are sorted by each objective in
    /// turn (stably, starting from the order of the previous objective, as
    /// `sort_by_dimension` does) and appended to that objective's list.
    fn new(data: Vec<f64>, m: usize) -> Self {
        let n = data.len() / m;
        let sentinel = n;
        let mut list = Self {
            m,
            sentinel,
            data,
            next: vec![sentinel; (n + 1) * m],
            prev: vec![sentinel; (n + 1) * m],
            ignore: vec![0; n + 1],
            area: vec![0.0; (n + 1) * m],
            volume: vec![0.0; (n + 1) * m],
        };
        let mut order: Vec<usize> = (0..n).collect();
        for i in 0..m {
            // `sort_by_dimension`: Python's stable sort of (value, index, node).
            order.sort_by(|&a, &b| {
                list.data[a * m + i]
                    .partial_cmp(&list.data[b * m + i])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &node in &order {
                list.append(node, i);
            }
        }
        list
    }

    #[inline]
    fn d(&self, node: usize, i: usize) -> f64 {
        self.data[node * self.m + i]
    }

    #[inline]
    fn next(&self, node: usize, i: usize) -> usize {
        self.next[node * self.m + i]
    }

    #[inline]
    fn prev(&self, node: usize, i: usize) -> usize {
        self.prev[node * self.m + i]
    }

    fn append(&mut self, node: usize, i: usize) {
        let m = self.m;
        let last = self.prev[self.sentinel * m + i];
        self.next[node * m + i] = self.sentinel;
        self.prev[node * m + i] = last;
        self.prev[self.sentinel * m + i] = node;
        self.next[last * m + i] = node;
    }

    /// `bounds = min(bounds, node.data)` (all objectives).
    fn update_bounds(&self, node: usize, bounds: &mut [f64]) {
        for (k, b) in bounds.iter_mut().enumerate() {
            *b = nan_min(*b, self.d(node, k));
        }
    }

    /// Removes `node` from the lists `0..index`.
    fn remove(&mut self, node: usize, index: usize, bounds: &mut [f64]) {
        let m = self.m;
        for k in 0..index {
            let predecessor = self.prev[node * m + k];
            let successor = self.next[node * m + k];
            self.next[predecessor * m + k] = successor;
            self.prev[successor * m + k] = predecessor;
        }
        self.update_bounds(node, bounds);
    }

    /// Re-inserts `node` at its original position in the lists `0..index`.
    fn reinsert(&mut self, node: usize, index: usize, bounds: &mut [f64]) {
        let m = self.m;
        for k in 0..index {
            let predecessor = self.prev[node * m + k];
            let successor = self.next[node * m + k];
            self.next[predecessor * m + k] = node;
            self.prev[successor * m + k] = node;
        }
        self.update_bounds(node, bounds);
    }

    /// `_hv_recursive(i, n_pareto, bounds)` (minimization, reference point
    /// at the origin).
    fn hv_recursive(&mut self, i: usize, mut n_pareto: usize, bounds: &mut [f64]) -> f64 {
        let m = self.m;
        let sentinel = self.sentinel;
        if n_pareto == 0 {
            return 0.0;
        }
        if i == 0 {
            // One dimension.
            return -self.d(self.next(sentinel, 0), 0);
        }
        if i == 1 {
            // Two dimensions end the recursion.
            let mut hvol = 0.0;
            let mut q = self.next(sentinel, 1);
            let mut h = self.d(q, 0);
            let mut p = self.next(q, 1);
            while p != sentinel {
                hvol += h * (self.d(q, 1) - self.d(p, 1));
                if self.d(p, 0) < h {
                    h = self.d(p, 0);
                }
                q = p;
                p = self.next(q, 1);
            }
            hvol += h * self.d(q, 1);
            return hvol;
        }
        let mut p = sentinel;
        let mut q = self.prev(p, i);
        while q != sentinel {
            if self.ignore[q] < i {
                self.ignore[q] = 0;
            }
            q = self.prev(q, i);
        }
        q = self.prev(p, i);
        while n_pareto > 1 && (self.d(q, i) > bounds[i] || self.d(self.prev(q, i), i) >= bounds[i])
        {
            p = q;
            self.remove(p, i, bounds);
            q = self.prev(p, i);
            n_pareto -= 1;
        }
        let q_prev = self.prev(q, i);
        let mut hvol;
        if n_pareto > 1 {
            hvol = self.volume[q_prev * m + i]
                + self.area[q_prev * m + i] * (self.d(q, i) - self.d(q_prev, i));
        } else {
            hvol = 0.0;
            self.area[q * m] = 1.0;
            // `q.area[1:i+1] = q.area[:i] * -(q.data[:i])`: the right-hand
            // side uses the areas before the assignment.
            let old: Vec<f64> = self.area[q * m..q * m + i].to_vec();
            for (k, a) in old.iter().enumerate() {
                self.area[q * m + k + 1] = a * -self.d(q, k);
            }
        }
        self.volume[q * m + i] = hvol;
        if self.ignore[q] >= i {
            self.area[q * m + i] = self.area[q_prev * m + i];
        } else {
            self.area[q * m + i] = self.hv_recursive(i - 1, n_pareto, bounds);
            if self.area[q * m + i] <= self.area[q_prev * m + i] {
                self.ignore[q] = i;
            }
        }
        while p != sentinel {
            let p_data = self.d(p, i);
            hvol += self.area[q * m + i] * (p_data - self.d(q, i));
            bounds[i] = p_data;
            self.reinsert(p, i, bounds);
            n_pareto += 1;
            q = p;
            p = self.next(p, i);
            self.volume[q * m + i] = hvol;
            let q_prev = self.prev(q, i);
            if self.ignore[q] >= i {
                self.area[q * m + i] = self.area[q_prev * m + i];
            } else {
                self.area[q * m + i] = self.hv_recursive(i - 1, n_pareto, bounds);
                if self.area[q * m + i] <= self.area[q_prev * m + i] {
                    self.ignore[q] = i;
                }
            }
        }
        hvol -= self.area[q * m + i] * self.d(q, i);
        hvol
    }
}

/// BoTorch's `get_hypervolume_maximizing_subset(n, Y, ref_point)`: greedily
/// selects `n` rows of `y` (`n' × m`), each maximizing the hypervolume of the
/// selected set (computed with [`DominatedPartitioning`], as in BoTorch).
/// Returns the selected rows (in selection order) and their indices.
///
/// Candidates are scanned in increasing index order and the first one that
/// improves the best hypervolume so far is kept, as in BoTorch; the
/// hypervolumes of the candidates of a step are computed in parallel. When no
/// candidate improves the hypervolume, BoTorch draws one at random with
/// Python's `random.choice` over the remaining indices; here `choose` is
/// called with the remaining indices (in increasing order) and returns the
/// selected one (e.g. `|idx| idx[rng.random_range(0..idx.len())]`; the Python
/// bindings call `random.choice`).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `y` has fewer than `n` rows or
/// `choose` returns an index that is not remaining,
/// [`BotorchError::DimensionMismatch`] if `ref_point` does not have `m`
/// elements.
pub fn get_hypervolume_maximizing_subset<F>(
    n: usize,
    y: &Matrix,
    ref_point: &[f64],
    mut choose: F,
) -> Result<(Matrix, Vec<usize>)>
where
    F: FnMut(&[usize]) -> usize,
{
    let (total, m) = y.shape();
    if total < n {
        return Err(BotorchError::InvalidArgument(format!(
            "Y has fewer points ({total}) than the requested subset size ({n})."
        )));
    }
    if ref_point.len() != m {
        return Err(BotorchError::DimensionMismatch(format!(
            "ref_point has {} elements, expected {m}",
            ref_point.len()
        )));
    }
    let rows = row_major(y);
    let mut subset: Vec<f64> = Vec::with_capacity(n * m);
    let mut selected = Vec::with_capacity(n);
    let mut remaining: Vec<usize> = (0..total).collect();
    let mut best_hv = 0.0;
    for _ in 0..n {
        let k = selected.len();
        let hvs: Vec<Result<f64>> = remaining
            .par_iter()
            .map(|&i| {
                let mut values = Vec::with_capacity((k + 1) * m);
                values.extend_from_slice(&subset);
                values.extend_from_slice(&rows[i * m..(i + 1) * m]);
                let points = PointBatch::from_rows(k + 1, m, values)?;
                let partitioning = DominatedPartitioning::from_points(ref_point, Some(&points))?;
                Ok(partitioning.compute_hypervolume()[0])
            })
            .collect();
        let mut best_idx = None;
        for (&i, hv) in remaining.iter().zip(hvs) {
            let hv = hv?;
            if hv > best_hv {
                best_idx = Some(i);
                best_hv = hv;
            }
        }
        let best = match best_idx {
            Some(i) => i,
            // No point improves the hypervolume (this only happens if Y is
            // not a Pareto frontier of points better than the reference
            // point): select a random one.
            None => {
                let i = choose(&remaining);
                if !remaining.contains(&i) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "the chosen index {i} is not one of the remaining indices"
                    )));
                }
                i
            }
        };
        remaining.retain(|&i| i != best);
        selected.push(best);
        subset.extend_from_slice(&rows[best * m..(best + 1) * m]);
    }
    Ok((Matrix::from_row_slice(n, m, &subset), selected))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypervolume_small_cases() {
        let hv = Hypervolume::new(&[0.0, 0.0]);
        let y = Matrix::from_row_slice(3, 2, &[1.0, 3.0, 2.0, 2.0, 3.0, 1.0]);
        assert_eq!(hv.compute(&y).unwrap(), 6.0);
        // Points that do not dominate the reference point are ignored.
        let y = Matrix::from_row_slice(2, 2, &[1.0, 3.0, -1.0, 5.0]);
        assert_eq!(hv.compute(&y).unwrap(), 3.0);
        assert_eq!(hv.compute(&Matrix::zeros(0, 2)).unwrap(), 0.0);
        assert!(hv.compute(&Matrix::zeros(1, 3)).is_err());
        // One objective.
        let hv1 = Hypervolume::new(&[1.0]);
        assert_eq!(
            hv1.compute(&Matrix::from_row_slice(2, 1, &[3.0, 2.0]))
                .unwrap(),
            2.0
        );
        // BoTorch's test case (m = 3).
        let hv3 = Hypervolume::new(&[-1.0, -2.0, -3.0]);
        let y = Matrix::from_row_slice(3, 3, &[1.0, 6.0, 8.0, 2.0, 4.0, 10.0, 3.0, 5.0, 7.0]);
        assert_eq!(hv3.compute(&y).unwrap(), 358.0);
    }

    #[test]
    fn hypervolume_agrees_with_inclusion_exclusion() {
        // Exact hypervolume by inclusion-exclusion over all subsets.
        fn exact(points: &[Vec<f64>], r: &[f64]) -> f64 {
            let n = points.len();
            let mut total = 0.0;
            for mask in 1u32..(1 << n) {
                let mut corner = vec![f64::INFINITY; r.len()];
                for (i, p) in points.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        for (c, v) in corner.iter_mut().zip(p) {
                            *c = c.min(*v);
                        }
                    }
                }
                let vol: f64 = corner
                    .iter()
                    .zip(r)
                    .map(|(c, r)| (c - r).max(0.0))
                    .product();
                total += if mask.count_ones() % 2 == 1 {
                    vol
                } else {
                    -vol
                };
            }
            total
        }
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for m in 2..=5 {
            for n in [1usize, 2, 5, 9] {
                let points: Vec<Vec<f64>> =
                    (0..n).map(|_| (0..m).map(|_| next()).collect()).collect();
                let r = vec![0.1; m];
                let y = Matrix::from_fn(n, m, |i, j| points[i][j]);
                let hv = Hypervolume::new(&r).compute(&y).unwrap();
                let e = exact(&points, &r);
                assert!(
                    (hv - e).abs() <= 1e-12 * e.abs().max(1e-3),
                    "m={m} n={n}: {hv} vs {e}"
                );
            }
        }
    }

    #[test]
    fn reference_points() {
        let y = Matrix::from_row_slice(3, 2, &[1.0, 3.0, 2.0, 2.0, 3.0, 1.0]);
        // nadir - 0.1 * range
        let r = infer_reference_point(&y, None, 0.1, false).unwrap();
        assert_eq!(r, vec![1.0 - 0.1 * 2.0, 1.0 - 0.1 * 2.0]);
        // A point is better than max_ref_point.
        let r = infer_reference_point(&y, Some(&[0.5, 0.5]), 0.1, false).unwrap();
        assert_eq!(r, vec![0.5, 0.5]);
        let r = infer_reference_point(&y, Some(&[0.5, 0.5]), 0.1, true).unwrap();
        assert_eq!(r, vec![0.5 - 0.1 * 2.5, 0.5 - 0.1 * 2.5]);
        // No point is better: the nadir is capped by max_ref_point.
        let r = infer_reference_point(&y, Some(&[5.0, 0.0]), 0.1, false).unwrap();
        assert_eq!(r, vec![1.0 - 0.1 * 2.0, 0.0 - 0.1 * 3.0]);
        // A single point (with a zero objective).
        let y1 = Matrix::from_row_slice(1, 2, &[0.0, 2.0]);
        let r = infer_reference_point(&y1, None, 0.1, false).unwrap();
        assert_eq!(r, vec![0.0 - 0.1 * MIN_Y_RANGE, 2.0 - 0.1 * 2.0]);
        // NaN elements of max_ref_point are not used.
        let r = infer_reference_point(&y, Some(&[f64::NAN, 0.5]), 0.1, false).unwrap();
        assert_eq!(r, vec![1.0 - 0.1 * 2.0, 0.5]);
        // Empty Pareto sets.
        let e = Matrix::zeros(0, 2);
        assert!(infer_reference_point(&e, None, 0.1, false).is_err());
        assert!(infer_reference_point(&e, Some(&[f64::NAN, 1.0]), 0.1, false).is_err());
        assert_eq!(
            infer_reference_point(&e, Some(&[-2.0, 1.0]), 0.1, true).unwrap(),
            vec![-2.0 - 0.2, 1.0 - 0.1]
        );
    }

    #[test]
    fn maximizing_subsets() {
        let y = Matrix::from_row_slice(4, 2, &[1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 0.5, 0.5]);
        let (sub, idx) =
            get_hypervolume_maximizing_subset(2, &y, &[0.0, 0.0], |_| unreachable!()).unwrap();
        // (2, 2) first (hypervolume 4), then (1, 3) (+1, the first of two ties).
        assert_eq!(idx, vec![1, 0]);
        assert_eq!(sub, Matrix::from_row_slice(2, 2, &[2.0, 2.0, 1.0, 3.0]));
        // A dominated point adds no hypervolume and is drawn at random.
        let (_, idx) =
            get_hypervolume_maximizing_subset(4, &y, &[0.0, 0.0], |r| *r.last().unwrap()).unwrap();
        assert_eq!(idx, vec![1, 0, 2, 3]);
        assert!(get_hypervolume_maximizing_subset(5, &y, &[0.0, 0.0], |r| r[0]).is_err());
        assert!(get_hypervolume_maximizing_subset(4, &y, &[0.0, 0.0, 0.0], |r| r[0]).is_err());
    }
}
