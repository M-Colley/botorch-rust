//! Multi-objective utilities mirroring `botorch.utils.multi_objective`:
//! Pareto dominance ([`pareto`]).
//!
//! | BoTorch | botorch_core |
//! |---|---|
//! | `is_non_dominated(Y, maximize, deduplicate)` | [`pareto::is_non_dominated`] (`n × m` matrix), [`pareto::is_non_dominated_rows`] (row-major slice), [`pareto::is_non_dominated_batch`] (a [`PointBatch`], parallel over the batch) |
//!
//! ## Conventions
//!
//! * Like BoTorch, everything assumes **maximization** unless an argument says
//!   otherwise.
//! * A point set is an `n × m` [`Matrix`] (rows are points) or, for batches
//!   of equally sized point sets (BoTorch's `batch_shape x n x m` tensors), a
//!   [`PointBatch`] stored row-major.
//! * All types are `Send + Sync`; independent batch elements are processed in
//!   parallel on the current rayon pool.
//!
//! ## Differences from BoTorch
//!
//! * Points with a NaN in any coordinate are dominated in
//!   [`pareto::is_non_dominated`] and never affect other points (the fix of
//!   meta-pytorch/botorch#3392; BoTorch main only masks out all-NaN points).

pub mod pareto;

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
    }
}
