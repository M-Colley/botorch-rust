//! Dense float64 arrays exchanged with the Python layer.
//!
//! The `botorch_rs` package converts torch tensors, NumPy arrays and nested
//! lists into a flat row-major `list[float]` together with its shape, i.e. a
//! `(data, shape)` tuple, and converts the `(data, shape)` results of this
//! module back into the caller's array type. Keeping the extension free of
//! NumPy / torch types is what lets the package work without either library.

use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;

use crate::errors::dimension_error;

/// An array received from Python: `(data, shape)` with `data` in row-major
/// (C) order.
#[derive(FromPyObject)]
pub struct Array(pub Vec<f64>, pub Vec<usize>);

/// An array returned to Python: `(data, shape)` in row-major order.
pub type ArrayOut = (Vec<f64>, Vec<usize>);

impl Array {
    fn validate(&self, name: &str) -> PyResult<()> {
        let n: usize = self.1.iter().product();
        if n != self.0.len() {
            return Err(dimension_error(format!(
                "{name}: {} values do not match the shape {:?}",
                self.0.len(),
                self.1
            )));
        }
        Ok(())
    }

    /// The array as an `r × c` matrix (it must be two-dimensional).
    pub fn matrix(&self, name: &str) -> PyResult<Matrix> {
        self.validate(name)?;
        match self.1.as_slice() {
            &[r, c] => Ok(Matrix::from_row_slice(r, c, &self.0)),
            s => Err(dimension_error(format!(
                "Expected `{name}` to be a 2-dimensional tensor, got shape {s:?}."
            ))),
        }
    }

    /// The array as a matrix whose rows are the points of a `(..., d)` array
    /// (all leading dimensions flattened); returns the leading shape too.
    pub fn points(&self, name: &str) -> PyResult<(Vec<usize>, Matrix)> {
        self.validate(name)?;
        let Some((&d, batch)) = self.1.split_last() else {
            return Err(dimension_error(format!(
                "Expected `{name}` to have at least one dimension (`... x d`)."
            )));
        };
        let n: usize = batch.iter().product();
        Ok((batch.to_vec(), Matrix::from_row_slice(n, d, &self.0)))
    }

    /// The array as a batch of `q × d` matrices (shape `(*batch, q, d)` with
    /// at least two dimensions); returns the batch shape too.
    pub fn batches(&self, name: &str) -> PyResult<(Vec<usize>, Vec<Matrix>)> {
        self.validate(name)?;
        if self.1.len() < 2 {
            return Err(dimension_error(format!(
                "`{name}` must have at least 2 dimensions (`batch_shape x q x d`), got shape {:?}.",
                self.1
            )));
        }
        let (q, d) = (self.1[self.1.len() - 2], self.1[self.1.len() - 1]);
        let batch = self.1[..self.1.len() - 2].to_vec();
        let b: usize = batch.iter().product();
        let mats = (0..b)
            .map(|i| Matrix::from_row_slice(q, d, &self.0[i * q * d..(i + 1) * q * d]))
            .collect();
        Ok((batch, mats))
    }
}

/// A matrix in row-major order with its shape.
pub fn matrix_out(m: &Matrix) -> ArrayOut {
    let (r, c) = m.shape();
    let mut data = Vec::with_capacity(r * c);
    for i in 0..r {
        data.extend(m.row(i).iter());
    }
    (data, vec![r, c])
}

/// A vector as a one-dimensional array.
pub fn vector_out(v: &Vector) -> ArrayOut {
    (v.as_slice().to_vec(), vec![v.len()])
}

/// Matrices of identical shape `r × c` stacked into an array of shape
/// `(*batch, r, c)`.
pub fn stack_out(mats: &[Matrix], batch: &[usize], r: usize, c: usize) -> ArrayOut {
    let mut data = Vec::with_capacity(mats.len() * r * c);
    for m in mats {
        for i in 0..r {
            data.extend(m.row(i).iter());
        }
    }
    let mut shape = batch.to_vec();
    shape.extend([r, c]);
    (data, shape)
}
