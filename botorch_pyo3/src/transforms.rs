//! `Normalize` and `Standardize` (`botorch.models.transforms`), including
//! their batched forms (`batch_shape`).
//!
//! The classes are immutable: fitting returns a new, fitted object (the Python
//! wrappers swap it in), which keeps them trivially thread-safe. A transform
//! with a batch shape holds one transform per batch element (row-major).

use botorch_core::{BatchedInputTransform, Matrix, Normalize, Standardize};
use pyo3::prelude::*;

use crate::arrays::{broadcast_indices, broadcast_shapes, matrix_out, Array, ArrayBuf, ArrayOut};
use crate::errors::{dimension_error, value_error, IntoPyResult};

/// The row-major product of a shape (1 for the empty shape).
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Splits an array of shape `(*batch, r, c)` into its batch shape and the
/// `r × c` matrices.
fn split_batches(x: &Array, name: &str) -> PyResult<(Vec<usize>, Vec<Matrix>)> {
    x.batches(name)
}

/// Input normalization to the unit cube (BoTorch's `Normalize`).
#[pyclass(frozen, name = "Normalize", module = "botorch_rs._botorch_rs")]
pub struct PyNormalize {
    /// The transforms applied by `transform` / `untransform` (fitted or with
    /// explicit bounds), one per batch element.
    pub inner: Vec<Normalize>,
    /// The batch shape (empty for a single transform).
    pub batch_shape: Vec<usize>,
    /// Whether the bounds are (re-)learned from training data.
    pub learn_bounds: bool,
}

impl PyNormalize {
    fn element_for_model(&self, tf: &Normalize) -> Normalize {
        if self.learn_bounds {
            Normalize::new(tf.dim()).with_min_range(tf.min_range())
        } else {
            tf.clone()
        }
    }

    /// The transform handed to a model with batch shape `batch`: fresh
    /// learnable transforms if the bounds are learned (the model fits them to
    /// its training inputs), else copies with the fixed bounds.
    pub fn for_model(&self, batch: &[usize]) -> PyResult<BatchedInputTransform> {
        if self.batch_shape.is_empty() {
            return Ok(BatchedInputTransform::Shared(
                self.element_for_model(&self.inner[0]),
            ));
        }
        if self.batch_shape != batch {
            return Err(dimension_error(format!(
                "The input transform has batch shape {:?} but the training data have batch \
                 shape {batch:?}.",
                self.batch_shape
            )));
        }
        Ok(BatchedInputTransform::PerBatch(
            self.inner
                .iter()
                .map(|tf| self.element_for_model(tf))
                .collect(),
        ))
    }

    /// Wraps the transform fitted by a model.
    pub fn fitted(tf: &BatchedInputTransform, learn_bounds: bool, batch: &[usize]) -> Self {
        match tf {
            BatchedInputTransform::Shared(t) => Self {
                inner: vec![t.clone()],
                batch_shape: Vec::new(),
                learn_bounds,
            },
            BatchedInputTransform::PerBatch(ts) => {
                if batch.is_empty() {
                    Self {
                        inner: vec![ts[0].clone()],
                        batch_shape: Vec::new(),
                        learn_bounds,
                    }
                } else {
                    Self {
                        inner: ts.clone(),
                        batch_shape: batch.to_vec(),
                        learn_bounds,
                    }
                }
            }
        }
    }

    fn d(&self) -> usize {
        self.inner[0].dim()
    }

    /// Per-element values as an array of shape `(*batch, rows, d)`.
    fn stacked<F>(&self, rows: usize, f: F) -> ArrayOut
    where
        F: Fn(&Normalize) -> Matrix,
    {
        let mut data = Vec::new();
        for tf in &self.inner {
            data.extend(matrix_out(&f(tf)).0);
        }
        let mut shape = self.batch_shape.clone();
        shape.extend([rows, self.d()]);
        (data, shape)
    }

    fn apply(&self, x: &Array, inverse: bool) -> PyResult<ArrayBuf> {
        let (xbatch, mats) = split_batches(x, "X")
            .map_err(|_| dimension_error("`X` must have at least 2 dimensions (`... x n x d`)."))?;
        let out_batch = broadcast_shapes(&xbatch, &self.batch_shape).ok_or_else(|| {
            dimension_error(format!(
                "X with batch shape {xbatch:?} cannot be broadcast with the transform's batch \
                 shape {:?}.",
                self.batch_shape
            ))
        })?;
        let xi = broadcast_indices(&xbatch, &out_batch);
        let ti = broadcast_indices(&self.batch_shape, &out_batch);
        let n = x.1[x.1.len() - 2];
        let mut data = Vec::with_capacity(numel(&out_batch) * n * self.d());
        for (a, b) in xi.iter().zip(&ti) {
            let tf = &self.inner[*b];
            let out = if inverse {
                tf.untransform(&mats[*a])
            } else {
                tf.transform(&mats[*a])
            }
            .into_py_result()?;
            data.extend(matrix_out(&out).0);
        }
        let mut shape = out_batch;
        shape.extend([n, self.d()]);
        Ok(ArrayBuf(data, shape))
    }
}

#[pymethods]
impl PyNormalize {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// `Normalize(d, bounds, learn_bounds, min_range, batch_shape)`; without
    /// `bounds` and without learning, the transform is the identity. Bounds of
    /// shape `batch_shape x 2 x d` give one transform per batch element.
    #[new]
    #[pyo3(signature = (d, bounds=None, learn_bounds=true, min_range=1e-8, batch_shape=Vec::new()))]
    fn new(
        d: usize,
        bounds: Option<Array>,
        learn_bounds: bool,
        min_range: f64,
        batch_shape: Vec<usize>,
    ) -> PyResult<Self> {
        if d == 0 {
            return Err(value_error("Normalize requires d >= 1."));
        }
        if !(min_range >= 0.0 && min_range.is_finite()) {
            return Err(value_error(format!(
                "min_range must be finite and non-negative, got {min_range}"
            )));
        }
        let (batch_shape, mats) = match bounds {
            Some(b) => {
                let (bb, mats) = split_batches(&b, "bounds")?;
                if mats.iter().any(|m| m.nrows() != 2 || m.ncols() != d) {
                    return Err(dimension_error(format!(
                        "Dimensions of provided `bounds` are incompatible with transform_dimension = {d}!"
                    )));
                }
                (bb, mats)
            }
            None => (
                batch_shape.clone(),
                vec![Matrix::from_fn(2, d, |i, _| i as f64); numel(&batch_shape)],
            ),
        };
        let inner = mats
            .iter()
            .map(|b| Normalize::with_bounds(b).map(|t| t.with_min_range(min_range)))
            .collect::<botorch_core::Result<Vec<_>>>()
            .into_py_result()?;
        if inner.is_empty() {
            return Err(value_error("the batch shape must not contain zeros"));
        }
        Ok(Self {
            inner,
            batch_shape,
            learn_bounds,
        })
    }

    /// Input dimension.
    #[getter(d)]
    fn get_d(&self) -> usize {
        self.d()
    }

    /// The batch shape (empty for a single transform).
    #[getter(batch_shape)]
    fn get_batch_shape(&self) -> Vec<usize> {
        self.batch_shape.clone()
    }

    /// Whether the bounds are learned from training data.
    #[getter(learn_bounds)]
    fn get_learn_bounds(&self) -> bool {
        self.learn_bounds
    }

    /// The `min_range` setting.
    #[getter]
    fn min_range(&self) -> f64 {
        self.inner[0].min_range()
    }

    /// Offsets (lower bounds), shape `batch_shape x 1 x d`.
    #[getter]
    fn offset(&self) -> ArrayOut {
        self.stacked(1, |tf| {
            Matrix::from_row_slice(1, tf.dim(), tf.offset().as_slice())
        })
    }

    /// Coefficients (ranges), shape `batch_shape x 1 x d`.
    #[getter]
    fn coefficient(&self) -> ArrayOut {
        self.stacked(1, |tf| {
            Matrix::from_row_slice(1, tf.dim(), tf.coefficient().as_slice())
        })
    }

    /// The bounds, shape `batch_shape x 2 x d`.
    #[getter]
    fn bounds(&self) -> ArrayOut {
        self.stacked(2, Normalize::bounds)
    }

    /// A copy whose bounds are learned from the points `X` (`... x n x d`;
    /// with a batch shape, `... x batch_shape x n x d`) if `learn_bounds` is
    /// set (BoTorch's train-mode forward pass: minima and ranges over all
    /// dimensions but the batch dimensions), else an unchanged copy.
    fn fit(&self, x: Array) -> PyResult<Self> {
        let fresh: Vec<Normalize> = self
            .inner
            .iter()
            .map(|tf| self.element_for_model(tf))
            .collect();
        if !self.learn_bounds {
            return Ok(Self {
                inner: self.inner.clone(),
                batch_shape: self.batch_shape.clone(),
                learn_bounds: false,
            });
        }
        let (xbatch, mats) =
            split_batches(&x, "X").or_else(|_| x.points("X").map(|(b, m)| (b, vec![m])))?;
        let k = self.batch_shape.len();
        if xbatch.len() < k || xbatch[xbatch.len() - k..] != self.batch_shape[..] {
            return Err(dimension_error(format!(
                "`X` must have the batch shape {:?} of the transform as trailing batch \
                 dimensions, got {xbatch:?}.",
                self.batch_shape
            )));
        }
        let nb = numel(&self.batch_shape);
        let mut inner = Vec::with_capacity(nb);
        for (b, mut tf) in fresh.into_iter().enumerate() {
            // Every point set of this batch element (extra leading dimensions included).
            let parts: Vec<&Matrix> = mats.iter().skip(b).step_by(nb).collect();
            let rows: usize = parts.iter().map(|m| m.nrows()).sum();
            let d = tf.dim();
            let mut all = Matrix::zeros(rows, d);
            let mut r = 0;
            for m in parts {
                all.rows_mut(r, m.nrows()).copy_from(m);
                r += m.nrows();
            }
            tf.fit(&all).into_py_result()?;
            inner.push(tf);
        }
        Ok(Self {
            inner,
            batch_shape: self.batch_shape.clone(),
            learn_bounds: true,
        })
    }

    /// Normalizes the points `X` (`... x n x d`) with the current bounds.
    fn transform(&self, x: Array) -> PyResult<ArrayBuf> {
        self.apply(&x, false)
    }

    /// Inverts the normalization of `X` (`... x n x d`).
    fn untransform(&self, x: Array) -> PyResult<ArrayBuf> {
        self.apply(&x, true)
    }
}

/// Outcome standardization (BoTorch's `Standardize` for `m` outputs and an
/// optional batch shape).
#[pyclass(frozen, name = "Standardize", module = "botorch_rs._botorch_rs")]
pub struct PyStandardize {
    /// The (possibly fitted) transforms, one per batch element.
    pub inner: Vec<Standardize>,
    /// The batch shape.
    pub batch_shape: Vec<usize>,
}

impl PyStandardize {
    /// A fitted transform built from per-element statistics
    /// (`means[b][t]`, `stdvs[b][t]`).
    pub fn from_stats(
        means: Vec<Vec<f64>>,
        stdvs: Vec<Vec<f64>>,
        min_stdv: f64,
        batch_shape: Vec<usize>,
    ) -> PyResult<Self> {
        let inner = means
            .into_iter()
            .zip(stdvs)
            .map(|(m, s)| Standardize::fitted(m, s, min_stdv))
            .collect::<botorch_core::Result<Vec<_>>>()
            .into_py_result()?;
        Ok(Self { inner, batch_shape })
    }

    fn min_stdv_value(&self) -> f64 {
        self.inner[0].min_stdv()
    }

    fn stats(&self, f: fn(&Standardize) -> &[f64]) -> ArrayOut {
        let m = self.inner.first().map_or(0, |s| f(s).len());
        let mut data = Vec::new();
        for s in &self.inner {
            data.extend_from_slice(f(s));
        }
        let mut shape = self.batch_shape.clone();
        shape.extend([1, m]);
        (data, shape)
    }

    /// Applies `f` to every batch element of `y` (and `yvar`), broadcasting
    /// an unbatched `y` over the batch shape.
    fn map<F>(
        &self,
        y: &Array,
        yvar: Option<&Array>,
        f: F,
    ) -> PyResult<(ArrayBuf, Option<ArrayBuf>)>
    where
        F: Fn(
            &Standardize,
            &Matrix,
            Option<&Matrix>,
        ) -> botorch_core::Result<(Matrix, Option<Matrix>)>,
    {
        let (ybatch, ys) = split_batches(y, "Y")?;
        let yvs = match yvar {
            Some(v) => {
                let (vb, vs) = split_batches(v, "Yvar")?;
                if vb != ybatch {
                    return Err(dimension_error("Yvar must have the shape of Y."));
                }
                Some(vs)
            }
            None => None,
        };
        let out_batch = broadcast_shapes(&ybatch, &self.batch_shape).ok_or_else(|| {
            dimension_error(format!(
                "Y with batch shape {ybatch:?} cannot be broadcast with the transform's batch \
                 shape {:?}.",
                self.batch_shape
            ))
        })?;
        let yi = broadcast_indices(&ybatch, &out_batch);
        let ti = broadcast_indices(&self.batch_shape, &out_batch);
        let (n, m) = (y.1[y.1.len() - 2], y.1[y.1.len() - 1]);
        let mut data = Vec::new();
        let mut vdata = Vec::new();
        for (a, b) in yi.iter().zip(&ti) {
            let (yt, vt) =
                f(&self.inner[*b], &ys[*a], yvs.as_ref().map(|v| &v[*a])).into_py_result()?;
            data.extend(matrix_out(&yt).0);
            if let Some(vt) = vt {
                vdata.extend(matrix_out(&vt).0);
            }
        }
        let mut shape = out_batch;
        shape.extend([n, m]);
        let vout = yvs.map(|_| ArrayBuf(vdata, shape.clone()));
        Ok((ArrayBuf(data, shape), vout))
    }
}

#[pymethods]
impl PyStandardize {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// An untrained `Standardize(min_stdv=..., batch_shape=...)`.
    #[new]
    #[pyo3(signature = (min_stdv=1e-8, batch_shape=Vec::new()))]
    fn new(min_stdv: f64, batch_shape: Vec<usize>) -> PyResult<Self> {
        if !(min_stdv >= 0.0 && min_stdv.is_finite()) {
            return Err(value_error(format!(
                "min_stdv must be finite and non-negative, got {min_stdv}"
            )));
        }
        let n = numel(&batch_shape);
        if n == 0 {
            return Err(value_error("the batch shape must not contain zeros"));
        }
        Ok(Self {
            inner: vec![Standardize::new().with_min_stdv(min_stdv); n],
            batch_shape,
        })
    }

    /// The `min_stdv` setting.
    #[getter]
    fn min_stdv(&self) -> f64 {
        self.min_stdv_value()
    }

    /// The batch shape.
    #[getter(batch_shape)]
    fn get_batch_shape(&self) -> Vec<usize> {
        self.batch_shape.clone()
    }

    /// Whether the transform has been fitted.
    #[getter]
    fn is_trained(&self) -> bool {
        self.inner.iter().all(Standardize::is_trained)
    }

    /// Per-output means, shape `batch_shape x 1 x m` (`m = 0` until fitted).
    #[getter]
    fn means(&self) -> ArrayOut {
        self.stats(Standardize::means)
    }

    /// Per-output standard deviations, shape `batch_shape x 1 x m`.
    #[getter]
    fn stdvs(&self) -> ArrayOut {
        self.stats(Standardize::stdvs)
    }

    /// Fits the statistics to `Y` (`batch_shape x n x m`) and returns the
    /// fitted transform with the standardized `Y` and `Yvar` (BoTorch's
    /// train-mode forward).
    #[pyo3(signature = (y, yvar=None))]
    #[allow(clippy::type_complexity)]
    fn fit_transform(
        &self,
        y: Array,
        yvar: Option<Array>,
    ) -> PyResult<(Self, ArrayBuf, Option<ArrayBuf>)> {
        let (ybatch, ys) = split_batches(&y, "Y")?;
        if ybatch != self.batch_shape {
            return Err(dimension_error(format!(
                "Expected Y.shape[:-2] to be {:?}, matching the `batch_shape` argument to \
                 `Standardize`, but got Y.shape[:-2]={ybatch:?}.",
                self.batch_shape
            )));
        }
        let yvs = match &yvar {
            Some(v) => Some(split_batches(v, "Yvar")?.1),
            None => None,
        };
        let mut inner = Vec::with_capacity(ys.len());
        let mut data = Vec::new();
        let mut vdata = Vec::new();
        for (b, yb) in ys.iter().enumerate() {
            let mut st = Standardize::new().with_min_stdv(self.min_stdv_value());
            let (yt, vt) = st
                .fit_transform(yb, yvs.as_ref().map(|v| &v[b]))
                .into_py_result()?;
            data.extend(matrix_out(&yt).0);
            if let Some(vt) = vt {
                vdata.extend(matrix_out(&vt).0);
            }
            inner.push(st);
        }
        let shape = y.1.clone();
        let vout = yvar.map(|_| ArrayBuf(vdata, shape.clone()));
        Ok((
            Self {
                inner,
                batch_shape: self.batch_shape.clone(),
            },
            ArrayBuf(data, shape),
            vout,
        ))
    }

    /// Standardizes `Y` (`... x n x m`) and `Yvar` with the fitted statistics.
    #[pyo3(signature = (y, yvar=None))]
    fn transform(&self, y: Array, yvar: Option<Array>) -> PyResult<(ArrayBuf, Option<ArrayBuf>)> {
        self.map(&y, yvar.as_ref(), |s, y, v| s.transform(y, v))
    }

    /// Un-standardizes `Y` (`... x n x m`) and `Yvar`.
    #[pyo3(signature = (y, yvar=None))]
    fn untransform(&self, y: Array, yvar: Option<Array>) -> PyResult<(ArrayBuf, Option<ArrayBuf>)> {
        self.map(&y, yvar.as_ref(), |s, y, v| s.untransform(y, v))
    }
}
