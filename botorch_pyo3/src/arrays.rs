//! Dense float64 arrays exchanged with the Python layer.
//!
//! Arrays cross the boundary as `(data, shape)` tuples, with the values in
//! row-major (C) order. The `botorch_rs` package passes the values of torch
//! tensors and NumPy arrays as an `array.array('d')`, a float64 buffer of the
//! standard library that [`Array`] reads with one memcpy through the buffer
//! protocol, and the values of nested lists and of arrays with a few values
//! as a `list[float]`, which is cheaper for a few values. Results whose size
//! grows with the inputs ([`ArrayBuf`]) come back as `array.array('d')` too
//! (as lists up to [`LIST_MAX`] values), small results of fixed size
//! ([`ArrayOut`]: bounds, hyperparameters, ...) as lists. The Python layer
//! converts both forms back into the caller's array type. Keeping the
//! extension free of NumPy / torch types is what lets the package work without
//! either library.

use botorch_core::Matrix;
use pyo3::buffer::{PyBuffer, PyUntypedBuffer};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyList, PyTuple};

use crate::errors::dimension_error;

/// An array received from Python: `(data, shape)` with `data` in row-major
/// (C) order, a float64 buffer or a sequence of numbers (see
/// [`float_data`]).
#[derive(FromPyObject)]
pub struct Array(
    #[pyo3(from_py_with = float_data)] pub Vec<f64>,
    pub Vec<usize>,
);

/// The values of a Python array argument: a one-dimensional buffer of native
/// float64 values (an `array.array('d')` or a NumPy float64 array, copied
/// with one memcpy), or else any sequence of numbers (a list of floats,
/// extracted value by value).
pub fn float_data(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if !obj.is_instance_of::<PyList>() && !obj.is_instance_of::<PyTuple>() {
        if let Ok(buf) = PyUntypedBuffer::get(obj) {
            // Only formats in the native byte order (`PyBuffer` itself accepts
            // `>d` on little-endian machines), and not the unaligned dummy
            // pointer of an empty `array.array` (read as a sequence below).
            if buf.dimensions() == 1 && matches!(buf.format().to_bytes(), b"d" | b"@d" | b"=d") {
                if let Ok(buf) = buf.into_typed::<f64>() {
                    return buf.to_vec(obj.py());
                }
            }
        }
    }
    obj.extract()
}

/// An array returned to Python: `(data, shape)` in row-major order, with
/// `data` a list of floats (for small results; see [`ArrayBuf`]).
pub type ArrayOut = (Vec<f64>, Vec<usize>);

/// An array returned to Python like [`ArrayOut`], but with `data` an
/// `array.array('d')` if it has more than [`LIST_MAX`] values: they are
/// copied with one memcpy instead of being converted into a Python float
/// each. For results whose size grows with the inputs (`(values,
/// shape).into()` converts an [`ArrayOut`]).
pub struct ArrayBuf(pub Vec<f64>, pub Vec<usize>);

/// Results with at most this many values are returned as lists, which are
/// cheaper than a buffer for a few values.
pub const LIST_MAX: usize = 16;

impl From<ArrayOut> for ArrayBuf {
    fn from((data, shape): ArrayOut) -> Self {
        Self(data, shape)
    }
}

impl<'py> IntoPyObject<'py> for ArrayBuf {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        (float_values(py, &self.0)?, self.1).into_pyobject(py)
    }
}

/// `values` for Python: a list of floats for at most [`LIST_MAX`] values,
/// else an `array.array('d')` ([`float_buffer`]).
pub fn float_values<'py>(py: Python<'py>, values: &[f64]) -> PyResult<Bound<'py, PyAny>> {
    if values.len() <= LIST_MAX {
        Ok(PyList::new(py, values)?.into_any())
    } else {
        float_buffer(py, values)
    }
}

/// `values` as a new `array.array('d')`, filled with one memcpy through the
/// buffer protocol.
pub fn float_buffer<'py>(py: Python<'py>, values: &[f64]) -> PyResult<Bound<'py, PyAny>> {
    static ZERO: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let zero = ZERO.get_or_try_init(py, || -> PyResult<_> {
        let array = py.import("array")?.getattr("array")?;
        Ok(array.call1(("d", (0.0,)))?.unbind())
    })?;
    // `array('d', [0.0]) * n` allocates the array without a Python float per
    // value; the prototype itself is never modified.
    let out = zero.bind(py).mul(values.len())?;
    // (An empty array exports an unaligned dummy pointer.)
    if !values.is_empty() {
        PyBuffer::<f64>::get(&out)?.copy_from_slice(py, values)?;
    }
    Ok(out)
}

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

/// The numpy / torch broadcast of two shapes (right-aligned; dimensions of
/// size 1 broadcast), or `None` if they are incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i < n - a.len() {
            1
        } else {
            a[i - (n - a.len())]
        };
        let db = if i < n - b.len() {
            1
        } else {
            b[i - (n - b.len())]
        };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

/// For every element (row-major) of the broadcast shape `out`, the flat
/// row-major index of the corresponding element of an array of shape
/// `shape` (which must broadcast to `out`).
pub fn broadcast_indices(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let total: usize = out.iter().product();
    let offset = out.len() - shape.len();
    let mut strides = vec![0; shape.len()];
    let mut s = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = if shape[i] == 1 { 0 } else { s };
        s *= shape[i];
    }
    let mut result = Vec::with_capacity(total);
    let mut idx = vec![0usize; out.len()];
    for _ in 0..total {
        let mut flat = 0;
        for (i, &st) in strides.iter().enumerate() {
            flat += idx[offset + i] * st;
        }
        result.push(flat);
        // Increment the multi-index (row-major).
        for k in (0..out.len()).rev() {
            idx[k] += 1;
            if idx[k] < out[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    result
}

/// `data` (of shape `shape`) broadcast to `target`, or `None` if the shapes
/// are incompatible.
pub fn broadcast_to(data: &[f64], shape: &[usize], target: &[usize]) -> Option<Vec<f64>> {
    let out = broadcast_shapes(shape, target)?;
    if out != target {
        return None;
    }
    Some(
        broadcast_indices(shape, target)
            .into_iter()
            .map(|i| data[i])
            .collect(),
    )
}
