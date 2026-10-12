//! Multi-objective utilities mirroring `botorch.utils.multi_objective`:
//! Pareto dominance ([`pareto`]), hypervolumes ([`hypervolume`]), box
//! decompositions of the dominated and non-dominated space
//! ([`box_decompositions`]) and the augmented Chebyshev scalarization
//! ([`scalarization`]). `botorch.utils.multi_objective.optimize` (NSGA-II via
//! pymoo) is not ported.
//!
//! | BoTorch | botorch_core |
//! |---|---|
//! | `is_non_dominated(Y, maximize, deduplicate)` | [`pareto::is_non_dominated`] (`n × m` matrix), [`pareto::is_non_dominated_rows`] (row-major slice), [`pareto::is_non_dominated_batch`] (a [`PointBatch`], parallel over the batch) |
//! | `Hypervolume(ref_point).compute(pareto_Y)` | [`hypervolume::Hypervolume::new`]`(&ref_point)`, [`hypervolume::Hypervolume::compute`]`(&pareto_y)` |
//! | `infer_reference_point(pareto_Y, max_ref_point, scale, scale_max_ref_point)` | [`hypervolume::infer_reference_point`] |
//! | `get_hypervolume_maximizing_subset(n, Y, ref_point)` | [`hypervolume::get_hypervolume_maximizing_subset`]`(n, &y, &ref_point, choose)` (`choose` replaces `random.choice`) |
//! | `NondominatedPartitioning(ref_point, Y, alpha)` | [`box_decompositions::NondominatedPartitioning::new`] / [`new_batched`](box_decompositions::NondominatedPartitioning::new_batched) / [`from_points`](box_decompositions::NondominatedPartitioning::from_points) |
//! | `FastNondominatedPartitioning(ref_point, Y)`, `DominatedPartitioning(ref_point, Y)` | [`box_decompositions::FastNondominatedPartitioning`], [`box_decompositions::DominatedPartitioning`] (same constructors) |
//! | `bd.pareto_Y`, `bd.Y`, `bd.ref_point`, `bd.num_outcomes`, `bd.batch_shape`, `bd.update(Y)`, `bd.reset()`, `bd.partition_space()`, `bd.compute_hypervolume()`, `bd.get_hypercell_bounds()` | the [`box_decompositions::BoxDecomposition`] trait; bounds as contiguous [`box_decompositions::CellBounds`] |
//! | `BoxDecompositionList(*decompositions)` | [`box_decompositions::BoxDecompositionList`] (parallel construction with [`box_decompositions::BoxDecompositionList::par_build`] and parallel updates) |
//! | `box_decompositions.utils.*` | [`box_decompositions::utils`]: [`compute_local_upper_bounds`](box_decompositions::utils::compute_local_upper_bounds), [`get_partition_bounds`](box_decompositions::utils::get_partition_bounds), [`update_local_upper_bounds_incremental`](box_decompositions::utils::update_local_upper_bounds_incremental), [`compute_non_dominated_hypercell_bounds_2d`](box_decompositions::utils::compute_non_dominated_hypercell_bounds_2d), [`compute_dominated_hypercell_bounds_2d`](box_decompositions::utils::compute_dominated_hypercell_bounds_2d), [`expand_ref_point`](box_decompositions::utils::expand_ref_point) (`_expand_ref_point`), [`pad_batch_pareto_frontier`](box_decompositions::utils::pad_batch_pareto_frontier) (`_pad_batch_pareto_frontier`) |
//! | `get_chebyshev_scalarization(weights, Y, alpha)` | [`scalarization::get_chebyshev_scalarization`] -> [`scalarization::ChebyshevScalarization`] (values and gradients) |
//!
//! ## Conventions
//!
//! * Like BoTorch, everything assumes **maximization** unless an argument says
//!   otherwise; box decompositions store the negated outcomes internally and
//!   run the published algorithms for minimization, exactly as BoTorch does.
//! * A point set is an `n × m` [`Matrix`] (rows are points) or, for batches
//!   of equally sized point sets (BoTorch's `batch_shape x n x m` tensors), a
//!   [`PointBatch`] stored row-major. Hypercell bounds are returned as
//!   contiguous row-major [`box_decompositions::CellBounds`].
//! * All computations are in `f64`. Results agree with BoTorch's float64
//!   results to rounding, and the algorithms reproduce BoTorch's output
//!   *order* (Pareto points, hypercells, local upper bounds), not only the
//!   sets.
//! * All types are `Send + Sync`; independent batch elements (Pareto masks of
//!   a batch, decompositions of a [`box_decompositions::BoxDecompositionList`])
//!   are processed in parallel on the current rayon pool.
//!
//! ## Differences from BoTorch
//!
//! * Points with a NaN in any coordinate are dominated in
//!   [`pareto::is_non_dominated`] and never affect other points (the fix of
//!   meta-pytorch/botorch#3392; BoTorch main only masks out all-NaN points).
//! * `FastPartitioning.update` skips the incremental update when no new point
//!   is better than the reference point (also from #3392; the results are
//!   identical, BoTorch main re-inserts the whole frontier).
//! * `FastNondominatedPartitioning` and `DominatedPartitioning` with a batched
//!   reference point and no Pareto points return one single cell per batch
//!   element; BoTorch fails to broadcast the reference point there.
//!   Similarly, [`box_decompositions::utils::pad_batch_pareto_frontier`]
//!   accepts one reference point for all batch elements (BoTorch requires one
//!   per batch element when a front is empty).

pub mod box_decompositions;
pub mod hypervolume;
pub mod pareto;
pub mod scalarization;

use crate::error::{BotorchError, Result};
use crate::Matrix;

/// A batch of equally sized point sets: BoTorch's `(batch_shape) x n x m`
/// tensors, stored row-major (point `i` of batch element `b` occupies
/// `values[(b·n + i)·m .. (b·n + i + 1)·m]`). An empty `batch_shape` is a
/// single `n × m` point set.
#[derive(Clone, Debug, PartialEq)]
pub struct PointBatch {
    batch_shape: Vec<usize>,
    n: usize,
    m: usize,
    values: Vec<f64>,
}

impl PointBatch {
    /// A batch of shape `batch_shape x n x m` holding `values` (row-major).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `values` does not have
    /// `prod(batch_shape)·n·m` elements.
    pub fn new(batch_shape: Vec<usize>, n: usize, m: usize, values: Vec<f64>) -> Result<Self> {
        let expected = batch_shape.iter().product::<usize>() * n * m;
        if values.len() != expected {
            return Err(BotorchError::DimensionMismatch(format!(
                "{} values do not match the shape {:?}",
                values.len(),
                [batch_shape.as_slice(), &[n, m]].concat()
            )));
        }
        Ok(Self {
            batch_shape,
            n,
            m,
            values,
        })
    }

    /// An unbatched `n × m` point set from row-major values.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `values.len() != n·m`.
    pub fn from_rows(n: usize, m: usize, values: Vec<f64>) -> Result<Self> {
        Self::new(Vec::new(), n, m, values)
    }

    /// The rows of `y` as an unbatched point set.
    pub fn from_matrix(y: &Matrix) -> Self {
        Self {
            batch_shape: Vec::new(),
            n: y.nrows(),
            m: y.ncols(),
            values: row_major(y),
        }
    }

    /// A batch of `ys.len()` point sets (batch shape `[ys.len()]`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `ys` is empty or the matrices
    /// have different shapes.
    pub fn from_matrices(ys: &[Matrix]) -> Result<Self> {
        let Some(first) = ys.first() else {
            return Err(BotorchError::DimensionMismatch(
                "a batch needs at least one point set".into(),
            ));
        };
        let (n, m) = first.shape();
        let mut values = Vec::with_capacity(ys.len() * n * m);
        for y in ys {
            if y.shape() != (n, m) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "all point sets of a batch must have the same shape, got {n}x{m} and {}x{}",
                    y.nrows(),
                    y.ncols()
                )));
            }
            values.extend(row_major(y));
        }
        Ok(Self {
            batch_shape: vec![ys.len()],
            n,
            m,
            values,
        })
    }

    /// The batch shape (empty for a single point set).
    pub fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    /// The number of batch elements (1 for a single point set).
    pub fn batch_size(&self) -> usize {
        self.batch_shape.iter().product()
    }

    /// The number of points per batch element.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of outcomes.
    pub fn m(&self) -> usize {
        self.m
    }

    /// The full shape `[*batch_shape, n, m]`.
    pub fn shape(&self) -> Vec<usize> {
        let mut s = self.batch_shape.clone();
        s.extend([self.n, self.m]);
        s
    }

    /// All values, row-major.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The values, row-major.
    pub fn into_values(self) -> Vec<f64> {
        self.values
    }

    /// The `n × m` block (row-major) of batch element `b`.
    pub fn element(&self, b: usize) -> &[f64] {
        let len = self.n * self.m;
        &self.values[b * len..(b + 1) * len]
    }

    /// Point `i` of batch element `b`.
    pub fn point(&self, b: usize, i: usize) -> &[f64] {
        let start = (b * self.n + i) * self.m;
        &self.values[start..start + self.m]
    }

    /// Batch element `b` as an `n × m` matrix.
    pub fn matrix(&self, b: usize) -> Matrix {
        Matrix::from_row_slice(self.n, self.m, self.element(b))
    }

    /// All batch elements as `n × m` matrices.
    pub fn to_matrices(&self) -> Vec<Matrix> {
        (0..self.batch_size()).map(|b| self.matrix(b)).collect()
    }
}

/// The entries of `y` in row-major order.
pub(crate) fn row_major(y: &Matrix) -> Vec<f64> {
    let (n, m) = y.shape();
    let mut out = Vec::with_capacity(n * m);
    for i in 0..n {
        out.extend(y.row(i).iter());
    }
    out
}

/// `torch.min` of two values: NaN if either is NaN (unlike `f64::min`).
#[inline]
pub(crate) fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b < a {
        b
    } else {
        a
    }
}

/// `torch.max` of two values: NaN if either is NaN (unlike `f64::max`).
#[inline]
pub(crate) fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b > a {
        b
    } else {
        a
    }
}

/// `clamp_min(v, lo)` with torch's NaN propagation.
#[inline]
pub(crate) fn clamp_min(v: f64, lo: f64) -> f64 {
    if v < lo {
        lo
    } else {
        v
    }
}

/// `torch.Size([...])`, as BoTorch prints shapes in its messages.
pub(crate) fn torch_size(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("torch.Size([{}])", dims.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_batch_layout() {
        let a = Matrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.map(|v| v + 10.0);
        let pb = PointBatch::from_matrices(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(pb.shape(), vec![2, 2, 3]);
        assert_eq!(pb.point(1, 0), &[11.0, 12.0, 13.0]);
        assert_eq!(pb.matrix(0), a);
        assert_eq!(pb.to_matrices(), vec![a.clone(), b]);
        assert_eq!(
            PointBatch::from_matrix(&a).element(0),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert!(PointBatch::new(vec![2], 2, 3, vec![0.0; 11]).is_err());
        assert!(PointBatch::from_matrices(&[]).is_err());
        assert!(PointBatch::from_matrices(&[a, Matrix::zeros(1, 3)]).is_err());
        assert!(nan_min(1.0, f64::NAN).is_nan() && nan_max(f64::NAN, 1.0).is_nan());
        assert!(clamp_min(f64::NAN, 0.0).is_nan());
        assert_eq!(clamp_min(-1.0, 0.0), 0.0);
        assert_eq!(torch_size(&[3, 2]), "torch.Size([3, 2])");
    }
}
