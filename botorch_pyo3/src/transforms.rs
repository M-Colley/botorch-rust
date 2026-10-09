//! `Normalize` and `Standardize` (`botorch.models.transforms`).
//!
//! The classes are immutable: fitting returns a new, fitted object (the Python
//! wrappers swap it in), which keeps them trivially thread-safe.

use botorch_core::{Matrix, Normalize, Standardize};
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayOut};
use crate::errors::{dimension_error, value_error, IntoPyResult};

/// Input normalization to the unit cube (BoTorch's `Normalize`).
#[pyclass(frozen, name = "Normalize", module = "botorch_rs._botorch_rs")]
pub struct PyNormalize {
    /// The transform applied by `transform` / `untransform` (fitted or with
    /// explicit bounds).
    pub inner: Normalize,
    /// Whether the bounds are (re-)learned from training data.
    pub learn_bounds: bool,
}

impl PyNormalize {
    /// The transform handed to a model: a fresh learnable `Normalize` if the
    /// bounds are learned (the model fits it to its training inputs), else a
    /// copy with the fixed bounds.
    pub fn for_model(&self) -> Normalize {
        if self.learn_bounds {
            Normalize::new(self.inner.dim()).with_min_range(self.inner.min_range())
        } else {
            self.inner.clone()
        }
    }

    /// Wraps a transform fitted by a model.
    pub fn fitted(inner: Normalize, learn_bounds: bool) -> Self {
        Self {
            inner,
            learn_bounds,
        }
    }

    fn apply(&self, x: &Array, inverse: bool) -> PyResult<ArrayOut> {
        let (batch, points) = x.points("X")?;
        let out = if inverse {
            self.inner.untransform(&points)
        } else {
            self.inner.transform(&points)
        }
        .into_py_result()?;
        let (data, _) = matrix_out(&out);
        let mut shape = batch;
        shape.push(self.inner.dim());
        Ok((data, shape))
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

    /// `Normalize(d, bounds, learn_bounds, min_range)`; without `bounds` and
    /// without learning, the transform is the identity.
    #[new]
    #[pyo3(signature = (d, bounds=None, learn_bounds=true, min_range=1e-8))]
    fn new(d: usize, bounds: Option<Array>, learn_bounds: bool, min_range: f64) -> PyResult<Self> {
        if d == 0 {
            return Err(value_error("Normalize requires d >= 1."));
        }
        if !(min_range >= 0.0 && min_range.is_finite()) {
            return Err(value_error(format!(
                "min_range must be finite and non-negative, got {min_range}"
            )));
        }
        let bounds = match bounds {
            Some(b) => {
                let b = b.matrix("bounds")?;
                if b.nrows() != 2 || b.ncols() != d {
                    return Err(dimension_error(format!(
                        "Dimensions of provided `bounds` are incompatible with transform_dimension = {d}!"
                    )));
                }
                b
            }
            None => Matrix::from_fn(2, d, |i, _| i as f64),
        };
        let inner = Normalize::with_bounds(&bounds)
            .into_py_result()?
            .with_min_range(min_range);
        Ok(Self {
            inner,
            learn_bounds,
        })
    }

    /// Input dimension.
    #[getter]
    fn d(&self) -> usize {
        self.inner.dim()
    }

    /// Whether the bounds are learned from training data.
    #[getter]
    fn learn_bounds(&self) -> bool {
        self.learn_bounds
    }

    /// The `min_range` setting.
    #[getter]
    fn min_range(&self) -> f64 {
        self.inner.min_range()
    }

    /// Per-dimension offsets (lower bounds), length `d`.
    #[getter]
    fn offset(&self) -> Vec<f64> {
        self.inner.offset().as_slice().to_vec()
    }

    /// Per-dimension coefficients (ranges), length `d`.
    #[getter]
    fn coefficient(&self) -> Vec<f64> {
        self.inner.coefficient().as_slice().to_vec()
    }

    /// The `2 x d` bounds.
    #[getter]
    fn bounds(&self) -> ArrayOut {
        matrix_out(&self.inner.bounds())
    }

    /// A copy whose bounds are learned from the points `X` (`... x d`) if
    /// `learn_bounds` is set (BoTorch's train-mode forward pass), else an
    /// unchanged copy.
    fn fit(&self, x: Array) -> PyResult<Self> {
        if !self.learn_bounds {
            return Ok(Self::fitted(self.inner.clone(), false));
        }
        let (_, points) = x.points("X")?;
        let mut tf = self.for_model();
        tf.fit(&points).into_py_result()?;
        Ok(Self::fitted(tf, true))
    }

    /// Normalizes the points `X` (`... x d`) with the current bounds.
    fn transform(&self, x: Array) -> PyResult<ArrayOut> {
        self.apply(&x, false)
    }

    /// Inverts the normalization of `X` (`... x d`).
    fn untransform(&self, x: Array) -> PyResult<ArrayOut> {
        self.apply(&x, true)
    }
}

/// Outcome standardization (BoTorch's `Standardize` for `m` outputs).
#[pyclass(frozen, name = "Standardize", module = "botorch_rs._botorch_rs")]
pub struct PyStandardize {
    /// The (possibly fitted) transform.
    pub inner: Standardize,
}

fn yvar_matrix(yvar: Option<Array>) -> PyResult<Option<Matrix>> {
    yvar.map(|v| v.matrix("Yvar")).transpose()
}

fn out_pair(y: Matrix, yvar: Option<Matrix>) -> (ArrayOut, Option<ArrayOut>) {
    (matrix_out(&y), yvar.as_ref().map(matrix_out))
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

    /// An untrained `Standardize(min_stdv=...)`.
    #[new]
    #[pyo3(signature = (min_stdv=1e-8))]
    fn new(min_stdv: f64) -> PyResult<Self> {
        if !(min_stdv >= 0.0 && min_stdv.is_finite()) {
            return Err(value_error(format!(
                "min_stdv must be finite and non-negative, got {min_stdv}"
            )));
        }
        Ok(Self {
            inner: Standardize::new().with_min_stdv(min_stdv),
        })
    }

    /// The `min_stdv` setting.
    #[getter]
    fn min_stdv(&self) -> f64 {
        self.inner.min_stdv()
    }

    /// Whether the transform has been fitted.
    #[getter]
    fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }

    /// Per-output means (empty until fitted).
    #[getter]
    fn means(&self) -> Vec<f64> {
        self.inner.means().to_vec()
    }

    /// Per-output standard deviations (empty until fitted).
    #[getter]
    fn stdvs(&self) -> Vec<f64> {
        self.inner.stdvs().to_vec()
    }

    /// Fits the statistics to `Y` (`n x m`) and returns the fitted transform
    /// with the standardized `Y` and `Yvar` (BoTorch's train-mode forward).
    #[pyo3(signature = (y, yvar=None))]
    #[allow(clippy::type_complexity)]
    fn fit_transform(
        &self,
        y: Array,
        yvar: Option<Array>,
    ) -> PyResult<(Self, ArrayOut, Option<ArrayOut>)> {
        let y = y.matrix("Y")?;
        let yvar = yvar_matrix(yvar)?;
        let mut st = Standardize::new().with_min_stdv(self.inner.min_stdv());
        let (yt, vt) = st.fit_transform(&y, yvar.as_ref()).into_py_result()?;
        let (yt, vt) = out_pair(yt, vt);
        Ok((Self { inner: st }, yt, vt))
    }

    /// Standardizes `Y` (`n x m`) and `Yvar` with the fitted statistics.
    #[pyo3(signature = (y, yvar=None))]
    fn transform(&self, y: Array, yvar: Option<Array>) -> PyResult<(ArrayOut, Option<ArrayOut>)> {
        let y = y.matrix("Y")?;
        let yvar = yvar_matrix(yvar)?;
        let (yt, vt) = self.inner.transform(&y, yvar.as_ref()).into_py_result()?;
        Ok(out_pair(yt, vt))
    }

    /// Un-standardizes `Y` (`n x m`) and `Yvar`.
    #[pyo3(signature = (y, yvar=None))]
    fn untransform(&self, y: Array, yvar: Option<Array>) -> PyResult<(ArrayOut, Option<ArrayOut>)> {
        let y = y.matrix("Y")?;
        let yvar = yvar_matrix(yvar)?;
        let (yt, vt) = self.inner.untransform(&y, yvar.as_ref()).into_py_result()?;
        Ok(out_pair(yt, vt))
    }
}
