//! The input and outcome transforms of phase 6b
//! (`botorch.models.transforms.input` / `.outcome` / `.utils`): the steps of
//! a model's chain of input transforms ([`PyInputStep`], wrapping
//! `botorch_core::transforms::input_steps::NewStep`), the per-output steps of
//! an outcome transform ([`PyOutcomeStep`]), the Python callables of
//! `AppendFeatures` and `NumericToCategoricalEncoding` (evaluated through
//! `botorch_rs._transforms_bridge`), and the helpers of
//! `botorch.models.transforms.utils`.

use std::sync::Arc;

use botorch_core::transforms::input_steps::{
    AffineKind, AffineStep, AppendStep, CategoricalEncoder, FeatureFunction, FeatureSource,
    FilterStep, ImputationStep, Log10Step, NewStep, NumericToCategoricalStep, OneHotEncoder,
    OneHotToNumericStep, RoundStep, TransformFlags, WarpStep,
};
use botorch_core::transforms::outcome::{OutcomeStep, StratifiedStandardize};
use botorch_core::transforms::utils as tf_utils;
use botorch_core::transforms::Standardize;
use botorch_core::{BotorchError, Matrix, Vector};
use pyo3::prelude::*;

use crate::arrays::{float_data, float_values, Array, ArrayBuf, ArrayOut};
use crate::callbacks::callback_error;
use crate::errors::{dimension_error, value_error, IntoPyResult};
use crate::input_perturbation::row_major;
use crate::specs::PyPrior;
use crate::threads;

/// The Python module that evaluates feature callables and encoders.
const TRANSFORMS_BRIDGE: &str = "botorch_rs._transforms_bridge";

/// A feature callable of `AppendFeatures` (with its `fkwargs`).
#[derive(Debug)]
struct PyFeatureFunction {
    callable: Py<PyAny>,
    fkwargs: Option<Py<PyAny>>,
}

impl PyFeatureFunction {
    fn kwargs<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.fkwargs {
            Some(k) => k.bind(py).clone(),
            None => py.None().into_bound(py),
        }
    }
}

impl FeatureFunction for PyFeatureFunction {
    fn features(&self, x: &Matrix) -> botorch_core::Result<Matrix> {
        let (n, k) = x.shape();
        let data = row_major(x);
        let (values, nf, df) = Python::attach(|py| -> PyResult<(Vec<f64>, usize, usize)> {
            let bridge = py.import(TRANSFORMS_BRIDGE)?;
            let res = bridge.call_method1(
                "features",
                (
                    self.callable.bind(py),
                    float_values(py, &data)?,
                    n,
                    k,
                    self.kwargs(py),
                ),
            )?;
            let (flat, nf, df): (Bound<'_, PyAny>, usize, usize) = res.extract()?;
            Ok((float_data(&flat)?, nf, df))
        })
        .map_err(callback_error)?;
        if values.len() != n * nf * df {
            return Err(BotorchError::DimensionMismatch(format!(
                "the feature callable returned {} values, expected {}",
                values.len(),
                n * nf * df
            )));
        }
        Ok(Matrix::from_row_slice(n * nf, df, &values))
    }

    fn vjp(&self, x: &Matrix, adjoint: &Matrix) -> botorch_core::Result<Matrix> {
        let (n, k) = x.shape();
        let data = row_major(x);
        let bar = row_major(adjoint);
        let values = Python::attach(|py| -> PyResult<Vec<f64>> {
            let bridge = py.import(TRANSFORMS_BRIDGE)?;
            let res = bridge.call_method1(
                "features_vjp",
                (
                    self.callable.bind(py),
                    float_values(py, &data)?,
                    n,
                    k,
                    float_values(py, &bar)?,
                    self.kwargs(py),
                ),
            )?;
            float_data(&res)
        })
        .map_err(callback_error)?;
        if values.len() != n * k {
            return Err(BotorchError::DimensionMismatch(format!(
                "the gradient of the feature callable has {} values, expected {}",
                values.len(),
                n * k
            )));
        }
        Ok(Matrix::from_row_slice(n, k, &values))
    }
}

/// An encoder of `NumericToCategoricalEncoding` given by a Python callable.
#[derive(Debug)]
struct PyEncoder {
    callable: Py<PyAny>,
}

impl CategoricalEncoder for PyEncoder {
    fn encode(&self, values: &[i64]) -> botorch_core::Result<Matrix> {
        let (data, n, w) = Python::attach(|py| -> PyResult<(Vec<f64>, usize, usize)> {
            let bridge = py.import(TRANSFORMS_BRIDGE)?;
            let res = bridge.call_method1("encode", (self.callable.bind(py), values.to_vec()))?;
            let (flat, n, w): (Bound<'_, PyAny>, usize, usize) = res.extract()?;
            Ok((float_data(&flat)?, n, w))
        })
        .map_err(callback_error)?;
        Ok(Matrix::from_row_slice(n, w, &data))
    }
}

fn flags(f: (bool, bool, bool)) -> TransformFlags {
    TransformFlags {
        train: f.0,
        eval: f.1,
        fantasize: f.2,
    }
}

/// Stacks the batches of `x` into one matrix (all points of all batches).
fn stacked(x: &Array, name: &str) -> PyResult<Matrix> {
    let (_, mats) = x.batches(name)?;
    let d = mats.first().map_or(0, Matrix::ncols);
    let n: usize = mats.iter().map(Matrix::nrows).sum();
    let mut out = Matrix::zeros(n, d);
    let mut r = 0;
    for m in &mats {
        out.rows_mut(r, m.nrows()).copy_from(m);
        r += m.nrows();
    }
    Ok(out)
}

/// Applies `f` to every batch matrix of `x` and stacks the results with the
/// batch shape of `x`.
fn per_batch<F>(py: Python<'_>, x: &Array, f: F) -> PyResult<ArrayBuf>
where
    F: Fn(&Matrix) -> botorch_core::Result<Matrix> + Sync,
{
    let (batch, mats) = x.batches("X")?;
    let outs = threads::run(py, || {
        mats.iter()
            .map(&f)
            .collect::<botorch_core::Result<Vec<_>>>()
    })
    .into_py_result()?;
    let (rows, cols) = outs.first().map_or((0, 0), Matrix::shape);
    let mut data = Vec::with_capacity(outs.len() * rows * cols);
    for o in &outs {
        data.extend(row_major(o));
    }
    let mut shape = batch;
    shape.extend([rows, cols]);
    Ok(ArrayBuf(data, shape))
}

/// A step of a chain of input transforms (any input transform of BoTorch
/// other than a plain `Normalize` and `InputPerturbation`; immutable).
#[pyclass(frozen, name = "InputStep", module = "botorch_rs._botorch_rs")]
pub struct PyInputStep {
    /// The step.
    pub inner: NewStep,
}

#[pymethods]
impl PyInputStep {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// `AffineInputTransform`, `Normalize` or `InputStandardize` (`kind`:
    /// `"fixed"`, `"normalize"` or `"standardize"`) of the dimensions
    /// `indices` (all if `None`).
    #[staticmethod]
    #[pyo3(signature = (
        d, indices, offset, coefficient, kind, learn, reverse, flags,
        min_range=1e-8, center=0.5, min_std=1e-8,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn affine(
        d: usize,
        indices: Option<Vec<usize>>,
        offset: Vec<f64>,
        coefficient: Vec<f64>,
        kind: &str,
        learn: bool,
        reverse: bool,
        flags: (bool, bool, bool),
        min_range: f64,
        center: f64,
        min_std: f64,
    ) -> PyResult<Self> {
        let kind = match kind {
            "normalize" => AffineKind::Normalize { min_range, center },
            "standardize" => AffineKind::Standardize { min_std },
            "fixed" => AffineKind::Fixed,
            other => {
                return Err(value_error(format!(
                    "unknown affine transform kind {other:?}"
                )))
            }
        };
        let step = AffineStep::new(d, indices, offset, coefficient, kind, learn)
            .into_py_result()?
            .with_reverse(reverse)
            .with_flags(self::flags(flags));
        Ok(Self {
            inner: NewStep::Affine(step),
        })
    }

    /// `Warp(d, indices, eps, bounds, concentration priors, reverse)`.
    #[staticmethod]
    #[pyo3(signature = (
        d, indices, eps, bounds, c0_prior, c1_prior, reverse, flags, batched=false,
        concentrations=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn warp(
        d: usize,
        indices: Vec<usize>,
        eps: f64,
        bounds: Option<(Vec<f64>, Vec<f64>)>,
        c0_prior: Option<PyRef<'_, PyPrior>>,
        c1_prior: Option<PyRef<'_, PyPrior>>,
        reverse: bool,
        flags: (bool, bool, bool),
        batched: bool,
        concentrations: Option<(Vec<f64>, Vec<f64>)>,
    ) -> PyResult<Self> {
        let mut step = WarpStep::new(d, indices, eps, bounds)
            .into_py_result()?
            .with_priors(c0_prior.map(|p| p.inner), c1_prior.map(|p| p.inner))
            .with_reverse(reverse)
            .with_flags(self::flags(flags))
            .with_batched(batched);
        if let Some((c0, c1)) = concentrations {
            step = step.with_concentrations(c0, c1).into_py_result()?;
        }
        Ok(Self {
            inner: NewStep::Warp(step),
        })
    }

    /// `Round(integer_indices, categorical_features, approximate, tau)`
    /// (`categorical`: `(start, cardinality)` pairs).
    #[staticmethod]
    fn round(
        integer_indices: Vec<usize>,
        categorical: Vec<(usize, usize)>,
        approximate: bool,
        tau: f64,
        flags: (bool, bool, bool),
    ) -> PyResult<Self> {
        let step = RoundStep::new(integer_indices, categorical, approximate, tau)
            .into_py_result()?
            .with_flags(self::flags(flags));
        Ok(Self {
            inner: NewStep::Round(step),
        })
    }

    /// `Log10(indices, reverse)`.
    #[staticmethod]
    fn log10(indices: Vec<usize>, reverse: bool, flags: (bool, bool, bool)) -> Self {
        Self {
            inner: NewStep::Log10(
                Log10Step::new(indices)
                    .with_reverse(reverse)
                    .with_flags(self::flags(flags)),
            ),
        }
    }

    /// `AppendFeatures(feature_set=...)` or `AppendFeatures(f=...)` (exactly
    /// one of `feature_set` and `f`).
    #[staticmethod]
    #[pyo3(signature = (feature_set, f, indices, fkwargs, skip_expand, flags))]
    fn append(
        feature_set: Option<Array>,
        f: Option<Py<PyAny>>,
        indices: Option<Vec<usize>>,
        fkwargs: Option<Py<PyAny>>,
        skip_expand: bool,
        flags: (bool, bool, bool),
    ) -> PyResult<Self> {
        let source = match (feature_set, f) {
            (Some(fs), None) => {
                if fs.1.len() != 2 {
                    return Err(value_error(
                        "`feature_set` must be an `n_f x d_f`-dim tensor!",
                    ));
                }
                FeatureSource::Fixed(fs.matrix("feature_set")?)
            }
            (None, Some(f)) => FeatureSource::Function(
                Arc::new(PyFeatureFunction {
                    callable: f,
                    fkwargs,
                }),
                indices,
            ),
            (None, None) => {
                return Err(value_error(
                    "Either a `feature_set` or a callable `f` has to be provided.",
                ))
            }
            (Some(_), Some(_)) => {
                return Err(value_error(
                    "Only one can be used: either `feature_set` or callable `f`.",
                ))
            }
        };
        let step = AppendStep::new(source, skip_expand)
            .into_py_result()?
            .with_flags(self::flags(flags));
        Ok(Self {
            inner: NewStep::Append(step),
        })
    }

    /// `InteractionFeatures(indices)`.
    #[staticmethod]
    #[pyo3(signature = (indices=None))]
    fn interaction(indices: Option<Vec<usize>>) -> PyResult<Self> {
        Ok(Self {
            inner: NewStep::Append(AppendStep::interaction(indices).into_py_result()?),
        })
    }

    /// `FilterFeatures(feature_indices)`.
    #[staticmethod]
    fn filter(feature_indices: Vec<usize>, flags: (bool, bool, bool)) -> PyResult<Self> {
        Ok(Self {
            inner: NewStep::Filter(
                FilterStep::new(feature_indices)
                    .into_py_result()?
                    .with_flags(self::flags(flags)),
            ),
        })
    }

    /// `OneHotToNumeric(dim, categorical_features)`.
    #[staticmethod]
    fn one_hot_to_numeric(
        dim: usize,
        categorical: Vec<(usize, usize)>,
        flags: (bool, bool, bool),
    ) -> PyResult<Self> {
        Ok(Self {
            inner: NewStep::OneHotToNumeric(
                OneHotToNumericStep::new(dim, categorical)
                    .into_py_result()?
                    .with_flags(self::flags(flags)),
            ),
        })
    }

    /// `NumericToCategoricalEncoding(dim, categorical_features, encoders)`:
    /// one encoder per categorical feature (`None`: one-hot encoding with
    /// the feature's cardinality, evaluated natively).
    #[staticmethod]
    fn numeric_to_categorical(
        dim: usize,
        categorical: Vec<(usize, usize)>,
        encoders: Vec<Option<Py<PyAny>>>,
        flags: (bool, bool, bool),
    ) -> PyResult<Self> {
        if encoders.len() != categorical.len() {
            return Err(value_error(
                "The keys of `encoders` must match the keys of `categorical_features`.",
            ));
        }
        let encs: Vec<Arc<dyn CategoricalEncoder>> = encoders
            .into_iter()
            .zip(&categorical)
            .map(|(e, &(_, card))| -> Arc<dyn CategoricalEncoder> {
                match e {
                    Some(c) => Arc::new(PyEncoder { callable: c }),
                    None => Arc::new(OneHotEncoder { num_classes: card }),
                }
            })
            .collect();
        Ok(Self {
            inner: NewStep::NumericToCategorical(
                NumericToCategoricalStep::new(dim, categorical, encs)
                    .into_py_result()?
                    .with_flags(self::flags(flags)),
            ),
        })
    }

    /// `LearnedFeatureImputation(feature_indices, d, target_task, bounds)`
    /// (`feature_indices`: `(task value, observed columns)` pairs).
    #[staticmethod]
    #[pyo3(signature = (feature_indices, d, target_task, bounds, flags, raw=None))]
    fn imputation(
        feature_indices: Vec<(i64, Vec<usize>)>,
        d: usize,
        target_task: Option<i64>,
        bounds: Option<(Vec<f64>, Vec<f64>)>,
        flags: (bool, bool, bool),
        raw: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mut step = ImputationStep::new(feature_indices, d, target_task, bounds)
            .into_py_result()?
            .with_flags(self::flags(flags));
        if let Some(r) = raw {
            step = step.with_raw(r).into_py_result()?;
        }
        Ok(Self {
            inner: NewStep::Imputation(step),
        })
    }

    /// BoTorch's class name of the step.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// `(transform_on_train, transform_on_eval, transform_on_fantasize)`.
    #[getter]
    fn flags(&self) -> (bool, bool, bool) {
        let f = self.inner.flags();
        (f.train, f.eval, f.fantasize)
    }

    /// The learned parameters (Warp: `concentration0` then
    /// `concentration1`; imputation: the raw values).
    #[getter]
    fn params(&self) -> Vec<f64> {
        self.inner.params()
    }

    /// The names and shapes of the learned parameters.
    #[getter]
    fn param_shapes(&self) -> Vec<(String, Vec<usize>)> {
        self.inner.param_shapes()
    }

    /// The step with the learned parameters `values`.
    fn with_params(&self, values: Vec<f64>) -> PyResult<Self> {
        let mut inner = self.inner.clone();
        inner.set_params(&values).into_py_result()?;
        Ok(Self { inner })
    }

    /// The fitted coefficients of an affine step (or of the normalization
    /// of a `Warp`): `(offset, coefficient)`.
    #[getter]
    fn affine_state(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        let a = match &self.inner {
            NewStep::Affine(a) => a,
            NewStep::Warp(w) => w.normalize(),
            _ => return None,
        };
        Some((a.offset().to_vec(), a.coefficient().to_vec()))
    }

    /// The number of features per point of `AppendFeatures` (once known).
    #[getter]
    fn n_f(&self) -> Option<usize> {
        match &self.inner {
            NewStep::Append(a) => a.n_f(),
            _ => None,
        }
    }

    /// The imputation values of `LearnedFeatureImputation` (`num_tasks · (d
    /// + 1)`, constrained).
    #[getter]
    fn imputation_values(&self) -> Option<Vec<f64>> {
        match &self.inner {
            NewStep::Imputation(s) => Some(s.imputation_values().0),
            _ => None,
        }
    }

    /// The step with its current coefficients, never refitted (e.g. the
    /// normalization of a batched `Warp`, learned from all batch elements).
    fn frozen(&self) -> Self {
        Self {
            inner: self.inner.frozen(),
        }
    }

    /// The output dimension for `d` input dimensions.
    fn out_dim(&self, d: usize) -> PyResult<usize> {
        self.inner.out_dim(d).into_py_result()
    }

    /// The step with its coefficients learned from `X` (all batch elements
    /// together; BoTorch's train-mode `_update_coefficients`), and the sizes
    /// of a feature callable determined.
    fn fit(&self, py: Python<'_>, x: Array) -> PyResult<Self> {
        let all = stacked(&x, "X")?;
        let mut inner = self.inner.clone();
        threads::run(py, || {
            inner.probe(&all)?;
            inner.fit(&all)
        })
        .into_py_result()?;
        Ok(Self { inner })
    }

    /// BoTorch's `transform(X)` (`batch_shape x n x d`; regardless of the
    /// flags): `batch_shape x n·k x d'`.
    fn transform(&self, py: Python<'_>, x: Array) -> PyResult<ArrayBuf> {
        let mut inner = self.inner.clone();
        if let Ok(all) = stacked(&x, "X") {
            inner.probe(&all).into_py_result()?;
        }
        per_batch(py, &x, |m| inner.transform(m))
    }

    /// BoTorch's `untransform(X)`.
    fn untransform(&self, py: Python<'_>, x: Array) -> PyResult<ArrayBuf> {
        per_batch(py, &x, |m| self.inner.untransform(m))
    }
}

/// A step of the outcome transform of one output (immutable).
#[pyclass(frozen, name = "OutcomeStep", module = "botorch_rs._botorch_rs")]
pub struct PyOutcomeStep {
    /// The step.
    pub inner: OutcomeStep,
}

#[pymethods]
impl PyOutcomeStep {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// An untrained `Standardize` of one output.
    #[staticmethod]
    fn standardize(min_stdv: f64) -> Self {
        Self {
            inner: OutcomeStep::Standardize(Standardize::new().with_min_stdv(min_stdv)),
        }
    }

    /// A fitted `Standardize` of one output.
    #[staticmethod]
    fn standardize_fitted(mean: f64, stdv: f64, min_stdv: f64) -> PyResult<Self> {
        Ok(Self {
            inner: OutcomeStep::Standardize(
                Standardize::fitted(vec![mean], vec![stdv], min_stdv).into_py_result()?,
            ),
        })
    }

    /// `Log()`.
    #[staticmethod]
    fn log() -> Self {
        Self {
            inner: OutcomeStep::Log,
        }
    }

    /// `Power(power)`.
    #[staticmethod]
    fn power(power: f64) -> Self {
        Self {
            inner: OutcomeStep::Power(power),
        }
    }

    /// `Bilog()`.
    #[staticmethod]
    fn bilog() -> Self {
        Self {
            inner: OutcomeStep::Bilog,
        }
    }

    /// `StratifiedStandardize(stratification_idx, all_task_values,
    /// min_stdv)`, untrained or with fitted statistics.
    #[staticmethod]
    #[pyo3(signature = (stratification_idx, task_values, min_stdv, stats=None))]
    fn stratified(
        stratification_idx: usize,
        task_values: Vec<i64>,
        min_stdv: f64,
        stats: Option<(Vec<f64>, Vec<f64>)>,
    ) -> PyResult<Self> {
        let mut st = StratifiedStandardize::new(stratification_idx, &task_values, min_stdv)
            .into_py_result()?;
        if let Some((m, s)) = stats {
            st = st.with_stats(m, s).into_py_result()?;
        }
        Ok(Self {
            inner: OutcomeStep::StratifiedStandardize(st),
        })
    }

    /// BoTorch's class name.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// The fitted statistics `(means, stdvs)` of a standardization (one value
    /// for `Standardize`, one per stratum for `StratifiedStandardize`).
    #[getter]
    fn stats(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        match &self.inner {
            OutcomeStep::Standardize(st) if st.is_trained() => {
                Some((st.means().to_vec(), st.stdvs().to_vec()))
            }
            OutcomeStep::StratifiedStandardize(st) => {
                Some((st.means().to_vec(), st.stdvs().to_vec()))
            }
            _ => None,
        }
    }

    /// Whether a standardization has been fitted.
    #[getter]
    fn is_trained(&self) -> bool {
        match &self.inner {
            OutcomeStep::Standardize(st) => st.is_trained(),
            OutcomeStep::StratifiedStandardize(st) => st.is_trained(),
            _ => true,
        }
    }

    /// The step in train mode on `y` (`n`) with noise `yvar` and inputs `x`
    /// (`n x d`, for `StratifiedStandardize`): the fitted step and the
    /// transformed values.
    #[pyo3(signature = (y, yvar=None, x=None))]
    fn fit_transform(
        &self,
        y: Vec<f64>,
        yvar: Option<Vec<f64>>,
        x: Option<Array>,
    ) -> PyResult<(Self, Vec<f64>, Option<Vec<f64>>)> {
        let n = y.len();
        let ym = Matrix::from_column_slice(n, 1, &y);
        let vm = yvar.map(|v| Matrix::from_column_slice(v.len(), 1, &v));
        let xm = match x {
            Some(a) => a.matrix("X")?,
            None => Matrix::zeros(n, 0),
        };
        let mut inner = self.inner.clone();
        let (yt, vt) = inner
            .fit_transform(&ym, vm.as_ref(), &xm)
            .into_py_result()?;
        Ok((
            Self { inner },
            yt.iter().copied().collect(),
            vt.map(|v| v.iter().copied().collect()),
        ))
    }

    /// The step in eval mode (or its inverse with `inverse`).
    #[pyo3(signature = (y, yvar=None, x=None, inverse=false))]
    fn apply(
        &self,
        y: Vec<f64>,
        yvar: Option<Vec<f64>>,
        x: Option<Array>,
        inverse: bool,
    ) -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
        let n = y.len();
        let ym = Matrix::from_column_slice(n, 1, &y);
        let vm = yvar.map(|v| Matrix::from_column_slice(v.len(), 1, &v));
        let xm = match x {
            Some(a) => a.matrix("X")?,
            None => Matrix::zeros(n, 0),
        };
        let (yt, vt) = if inverse {
            self.inner.untransform(&ym, vm.as_ref(), &xm)
        } else {
            self.inner.transform(&ym, vm.as_ref(), &xm)
        }
        .into_py_result()?;
        Ok((
            yt.iter().copied().collect(),
            vt.map(|v| v.iter().copied().collect()),
        ))
    }
}

fn check_same(a: &Array, b: &Array, what: &str) -> PyResult<()> {
    if a.0.len() != b.0.len() {
        return Err(dimension_error(format!(
            "{what}: the arguments must have the same shape, got {:?} and {:?}",
            a.1, b.1
        )));
    }
    Ok(())
}

/// `norm_to_lognorm_mean(mu, var)` (elementwise; same shapes).
#[pyfunction]
fn norm_to_lognorm_mean_py(mu: Array, var: Array) -> PyResult<ArrayOut> {
    check_same(&mu, &var, "norm_to_lognorm_mean")?;
    let out =
        mu.0.iter()
            .zip(&var.0)
            .map(|(&m, &v)| tf_utils::norm_to_lognorm_mean(m, v))
            .collect();
    Ok((out, mu.1))
}

/// `norm_to_lognorm_variance(mu, var)` (elementwise; same shapes).
#[pyfunction]
fn norm_to_lognorm_variance_py(mu: Array, var: Array) -> PyResult<ArrayOut> {
    check_same(&mu, &var, "norm_to_lognorm_variance")?;
    let out =
        mu.0.iter()
            .zip(&var.0)
            .map(|(&m, &v)| tf_utils::norm_to_lognorm_variance(m, v))
            .collect();
    Ok((out, mu.1))
}

/// `norm_to_lognorm(mu, Cov)` / `lognorm_to_norm(mu, Cov)` for
/// `batch x n` means and `batch x n x n` covariances.
#[pyfunction]
fn lognormal_moments_py(mu: Array, cov: Array, inverse: bool) -> PyResult<(ArrayOut, ArrayOut)> {
    let n = *mu
        .1
        .last()
        .ok_or_else(|| dimension_error("mu must have at least one dimension"))?;
    if cov.1.len() != mu.1.len() + 1
        || cov.1[cov.1.len() - 1] != n
        || cov.1[cov.1.len() - 2] != n
        || cov.0.len() != mu.0.len() * n
    {
        return Err(dimension_error(format!(
            "Cov must have shape {:?} x {n}, got {:?}",
            mu.1, cov.1
        )));
    }
    let b = mu.0.len().checked_div(n).unwrap_or(0);
    let mut mo = Vec::with_capacity(mu.0.len());
    let mut co = Vec::with_capacity(cov.0.len());
    for i in 0..b {
        let m = Vector::from_column_slice(&mu.0[i * n..(i + 1) * n]);
        let c = Matrix::from_row_slice(n, n, &cov.0[i * n * n..(i + 1) * n * n]);
        let (m2, c2) = if inverse {
            tf_utils::lognorm_to_norm(&m, &c)
        } else {
            tf_utils::norm_to_lognorm(&m, &c)
        };
        mo.extend(m2.iter());
        co.extend(row_major(&c2));
    }
    Ok(((mo, mu.1), (co, cov.1)))
}

/// `kumaraswamy_warp(X, c0, c1, eps)` / `inv_kumaraswamy_warp` with
/// concentrations per last dimension.
#[pyfunction]
fn kumaraswamy_warp_py(
    x: Array,
    c0: Vec<f64>,
    c1: Vec<f64>,
    eps: f64,
    inverse: bool,
) -> PyResult<ArrayOut> {
    let d =
        *x.1.last()
            .ok_or_else(|| dimension_error("X must have at least one dimension"))?;
    let bc = |c: &[f64], j: usize| if c.len() == 1 { c[0] } else { c[j] };
    if !(c0.len() == 1 || c0.len() == d) || !(c1.len() == 1 || c1.len() == d) {
        return Err(dimension_error(format!(
            "the concentrations must have 1 or {d} entries, got {} and {}",
            c0.len(),
            c1.len()
        )));
    }
    let out =
        x.0.iter()
            .enumerate()
            .map(|(i, &v)| {
                let j = i % d.max(1);
                if inverse {
                    tf_utils::inv_kumaraswamy_warp(v, bc(&c0, j), bc(&c1, j), eps)
                } else {
                    tf_utils::kumaraswamy_warp(v, bc(&c0, j), bc(&c1, j), eps)
                }
            })
            .collect();
    Ok((out, x.1))
}

/// `interaction_features(X)`: `batch x q x 1 x d(d-1)/2`.
#[pyfunction]
fn interaction_features_py(x: Array) -> PyResult<ArrayOut> {
    let d =
        *x.1.last()
            .ok_or_else(|| dimension_error("X must have at least one dimension"))?;
    let rows = x.0.len().checked_div(d).unwrap_or(0);
    let k = d * d.saturating_sub(1) / 2;
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        out.extend(tf_utils::interaction_features(&x.0[r * d..(r + 1) * d]));
    }
    let mut shape = x.1[..x.1.len() - 1].to_vec();
    shape.extend([1, k]);
    Ok((out, shape))
}

/// `nanstd(X, dim)` over the (non-negative) dimension `dim`.
#[pyfunction]
fn nanstd_py(x: Array, dim: usize, keepdim: bool) -> PyResult<ArrayOut> {
    if dim >= x.1.len() {
        return Err(dimension_error(format!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {dim})",
            -(x.1.len() as i64),
            x.1.len() as i64 - 1
        )));
    }
    let size = x.1[dim];
    let inner: usize = x.1[dim + 1..].iter().product();
    let outer: usize = x.1[..dim].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let col: Vec<f64> = (0..size).map(|s| x.0[(o * size + s) * inner + i]).collect();
            out.push(tf_utils::nanstd(&col));
        }
    }
    let mut shape = x.1.clone();
    if keepdim {
        shape[dim] = 1;
    } else {
        shape.remove(dim);
    }
    Ok((out, shape))
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInputStep>()?;
    m.add_class::<PyOutcomeStep>()?;
    m.add_function(wrap_pyfunction!(norm_to_lognorm_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(norm_to_lognorm_variance_py, m)?)?;
    m.add_function(wrap_pyfunction!(lognormal_moments_py, m)?)?;
    m.add_function(wrap_pyfunction!(kumaraswamy_warp_py, m)?)?;
    m.add_function(wrap_pyfunction!(interaction_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(nanstd_py, m)?)?;
    Ok(())
}
