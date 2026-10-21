//! Risk measures (`botorch.acquisition.risk_measures`) and multi-output
//! risk measures (`botorch.acquisition.multi_objective.multi_output_risk_measures`)
//! evaluated in Rust: [`PyRiskMeasure`] (`CVaR`, `VaR`, `WorstCase`,
//! `Expectation`, `MARS`) is a single-output MC objective,
//! [`PyMultiOutputRiskMeasure`] (`MVaR`, `IndependentCVaR`, `IndependentVaR`,
//! `MultiOutputWorstCase`, `MultiOutputExpectation`) a multi-output one.
//!
//! The objects are immutable; the Python layer builds new ones when MARS's
//! weights or baseline change. Acquisition functions receive them as
//! `("native", obj)` objective specifications. Called directly, they map
//! posterior samples `... x (q * n_w) x m` to the risk measure (`... x q`, or
//! `... x q' x m'` for the multi-output ones), with vector-Jacobian products
//! for torch's autograd.
//!
//! Preprocessing functions arrive as objective specifications too
//! ([`crate::acquisition`], [`crate::mo_acquisition`]); a `("callable", f)`
//! preprocessing function is called as `f(samples)`, as in BoTorch.

use std::sync::Arc;

use botorch_core::acquisition::multi_objective::multi_output_risk_measures::{
    mvar_make_differentiable, y_normalization_bounds, IndependentCVaR, IndependentVaR, MVaR,
    MVaROptions, Mars, MultiOutputExpectation, MultiOutputWorstCase,
};
use botorch_core::acquisition::multi_objective::McMultiOutputObjective;
use botorch_core::acquisition::risk_measures::{CVaR, Expectation, VaR, WorstCase};
use botorch_core::acquisition::{McObjective, McSamples};
use botorch_core::Matrix;
use pyo3::prelude::*;

use crate::acquisition::objective_of;
use crate::arrays::{matrix_out, Array, ArrayBuf, ArrayOut};
use crate::callbacks::PyCallback;
use crate::errors::{dimension_error, IntoPyResult};
use crate::mo_acquisition::{mo_objective_of, PyMultiOutputObjective};
use crate::models::PyModel;
use crate::threads;

/// The preprocessing function of a single-output risk measure.
fn preprocessing_of(spec: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Arc<dyn McObjective>>> {
    let Some(spec) = spec.filter(|s| !s.is_none()) else {
        return Ok(None);
    };
    let kind: String = spec.get_item(0)?.extract()?;
    if kind == "callable" {
        // `preprocessing_function(samples)` (no `X`).
        return Ok(Some(Arc::new(PyCallback::constraint(
            spec.get_item(1)?.unbind(),
        ))));
    }
    objective_of(Some(spec))
}

/// The preprocessing function of a multi-output risk measure.
fn mo_preprocessing_of(
    spec: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<dyn McMultiOutputObjective>>> {
    let Some(spec) = spec.filter(|s| !s.is_none()) else {
        return Ok(None);
    };
    let kind: String = spec.get_item(0)?.extract()?;
    if kind == "callable" {
        return Ok(Some(Arc::new(PyCallback::mo_preprocessing(
            spec.get_item(1)?.unbind(),
        ))));
    }
    mo_objective_of(Some(spec))
}

/// Posterior samples `... x n x m` as MC samples with all leading dimensions
/// flattened into the sample dimension; returns the leading shape too.
fn samples_of(samples: &Array) -> PyResult<(Vec<usize>, McSamples)> {
    let shape = &samples.1;
    if shape.len() < 2 || samples.0.len() != shape.iter().product::<usize>() {
        return Err(dimension_error(format!(
            "Expected samples of shape `sample_shape x batch_shape x q x m`, got {shape:?}."
        )));
    }
    let (n, m) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let lead = shape[..shape.len() - 2].to_vec();
    let s: usize = lead.iter().product();
    Ok((
        lead,
        McSamples::from_row_major(&samples.0, s, n, m).into_py_result()?,
    ))
}

/// A single-output risk measure (or MARS) evaluated in Rust.
#[pyclass(frozen, name = "RiskMeasure", module = "botorch_rs._botorch_rs")]
pub struct PyRiskMeasure {
    /// The objective.
    pub inner: Arc<dyn McObjective>,
    mars: Option<Arc<Mars>>,
}

impl PyRiskMeasure {
    fn plain(inner: Arc<dyn McObjective>) -> Self {
        Self { inner, mars: None }
    }

    fn of_mars(mars: Mars) -> Self {
        let mars = Arc::new(mars);
        Self {
            inner: mars.clone(),
            mars: Some(mars),
        }
    }

    fn as_mars(&self) -> PyResult<&Arc<Mars>> {
        self.mars
            .as_ref()
            .ok_or_else(|| crate::errors::unsupported("This risk measure is not MARS."))
    }
}

#[pymethods]
impl PyRiskMeasure {
    /// `CVaR(alpha, n_w, preprocessing_function)`.
    #[staticmethod]
    #[pyo3(signature = (alpha, n_w, preprocessing=None))]
    fn cvar(alpha: f64, n_w: usize, preprocessing: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let p = preprocessing_of(preprocessing.as_ref())?;
        Ok(Self::plain(Arc::new(
            CVaR::new(alpha, n_w, p).into_py_result()?,
        )))
    }

    /// `VaR(alpha, n_w, preprocessing_function)`.
    #[staticmethod]
    #[pyo3(signature = (alpha, n_w, preprocessing=None))]
    fn var(alpha: f64, n_w: usize, preprocessing: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let p = preprocessing_of(preprocessing.as_ref())?;
        Ok(Self::plain(Arc::new(
            VaR::new(alpha, n_w, p).into_py_result()?,
        )))
    }

    /// `WorstCase(n_w, preprocessing_function)`.
    #[staticmethod]
    #[pyo3(signature = (n_w, preprocessing=None))]
    fn worst_case(n_w: usize, preprocessing: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let p = preprocessing_of(preprocessing.as_ref())?;
        Ok(Self::plain(Arc::new(
            WorstCase::new(n_w, p).into_py_result()?,
        )))
    }

    /// `Expectation(n_w, preprocessing_function)`.
    #[staticmethod]
    #[pyo3(signature = (n_w, preprocessing=None))]
    fn expectation(n_w: usize, preprocessing: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let p = preprocessing_of(preprocessing.as_ref())?;
        Ok(Self::plain(Arc::new(
            Expectation::new(n_w, p).into_py_result()?,
        )))
    }

    /// `MARS(alpha, n_w, chebyshev_weights, baseline_Y, ref_point,
    /// preprocessing_function)` (the preprocessing function is a
    /// multi-output objective specification).
    #[staticmethod]
    #[pyo3(signature = (alpha, n_w, chebyshev_weights, baseline_y=None, ref_point=None, preprocessing=None))]
    fn mars(
        alpha: f64,
        n_w: usize,
        chebyshev_weights: Vec<f64>,
        baseline_y: Option<Array>,
        ref_point: Option<Vec<f64>>,
        preprocessing: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let p = mo_preprocessing_of(preprocessing.as_ref())?;
        let y = baseline_y.map(|y| y.matrix("baseline_Y")).transpose()?;
        Ok(Self::of_mars(
            Mars::new(alpha, n_w, chebyshev_weights, y, ref_point, p).into_py_result()?,
        ))
    }

    /// MARS with new Chebyshev weights (BoTorch's `chebyshev_weights`
    /// setter).
    fn with_chebyshev_weights(&self, weights: Vec<f64>) -> PyResult<Self> {
        let mut m = (**self.as_mars()?).clone();
        m.set_chebyshev_weights(weights).into_py_result()?;
        Ok(Self::of_mars(m))
    }

    /// MARS with the baseline outcomes `baseline_Y` (`n x m`).
    fn with_baseline_y(&self, baseline_y: Array) -> PyResult<Self> {
        let mut m = (**self.as_mars()?).clone();
        m.set_baseline_y(baseline_y.matrix("baseline_Y")?)
            .into_py_result()?;
        Ok(Self::of_mars(m))
    }

    /// MARS with the baseline of BoTorch's `set_baseline_Y(None, None,
    /// Y_samples)`: the non-dominated MVaR points of the predictions
    /// `Y_samples` (`(n * n_w) x m`).
    fn with_baseline_from_samples(&self, py: Python<'_>, y_samples: Array) -> PyResult<Self> {
        let mut m = (**self.as_mars()?).clone();
        let y = y_samples.matrix("Y_samples")?;
        threads::run(py, || m.set_baseline_y_from_samples(&y)).into_py_result()?;
        Ok(Self::of_mars(m))
    }

    /// MARS with the baseline of BoTorch's `set_baseline_Y(model,
    /// X_baseline)`: the posterior means of the `n_w` points of every
    /// baseline point.
    fn with_baseline_from_model(
        &self,
        py: Python<'_>,
        model: PyRef<'_, PyModel>,
        x_baseline: Array,
    ) -> PyResult<Self> {
        let mut m = (**self.as_mars()?).clone();
        let x = x_baseline.matrix("X_baseline")?;
        let model = Arc::clone(&model.inner);
        threads::run(py, || m.set_baseline_y_from_model(model.as_ref(), &x)).into_py_result()?;
        Ok(Self::of_mars(m))
    }

    /// MARS's Chebyshev weights.
    #[getter]
    fn chebyshev_weights(&self) -> PyResult<Vec<f64>> {
        Ok(self.as_mars()?.chebyshev_weights().to_vec())
    }

    /// MARS's baseline outcomes, if set.
    #[getter]
    fn baseline_y(&self) -> PyResult<Option<ArrayOut>> {
        Ok(self.as_mars()?.baseline_y().map(matrix_out))
    }

    /// MARS's normalization bounds (`2 x m`), once the baseline is set.
    #[getter]
    fn normalization_bounds(&self) -> PyResult<Option<ArrayOut>> {
        Ok(self.as_mars()?.normalization_bounds().map(matrix_out))
    }

    /// The risk measure of `samples` (`... x n x m`): `... x n'`.
    fn forward(&self, py: Python<'_>, samples: Array) -> PyResult<ArrayBuf> {
        let (lead, s) = samples_of(&samples)?;
        let obj = &self.inner;
        let v = threads::run(py, || obj.evaluate(&s, None, false)).into_py_result()?;
        let (n, ns) = v.values.shape();
        let mut data = Vec::with_capacity(n * ns);
        for c in 0..ns {
            data.extend(v.values.column(c).iter());
        }
        let mut shape = lead;
        shape.push(n);
        Ok(ArrayBuf(data, shape))
    }

    /// The vector-Jacobian product of [`PyRiskMeasure::forward`]: the adjoint
    /// of `samples` for the adjoint of the values.
    fn vjp(&self, py: Python<'_>, samples: Array, adjoint: Array) -> PyResult<ArrayBuf> {
        let (_, s) = samples_of(&samples)?;
        let obj = &self.inner;
        let (n, m, ns) = (s.num_points(), s.num_outputs(), s.num_samples());
        let sbar = threads::run(py, || -> botorch_core::Result<Vec<f64>> {
            let v = obj.evaluate(&s, None, true)?;
            let nv = v.values.nrows();
            if adjoint.0.len() != ns * nv {
                return Err(botorch_core::BotorchError::DimensionMismatch(format!(
                    "the adjoint has {} values, expected {}",
                    adjoint.0.len(),
                    ns * nv
                )));
            }
            // Values `... x n'` (row-major) as `n' × S`.
            let bar = Matrix::from_fn(nv, ns, |p, c| adjoint.0[c * nv + p]);
            let pullback = v.pullback.ok_or_else(|| {
                botorch_core::BotorchError::Runtime("the risk measure has no gradient".into())
            })?;
            let (sbar, _) = pullback(&bar)?;
            Ok(McSamples::new(sbar, n, m)?.to_row_major())
        })
        .into_py_result()?;
        Ok(ArrayBuf(sbar, samples.1.clone()))
    }
}

/// A multi-output risk measure evaluated in Rust.
#[pyclass(
    frozen,
    name = "MultiOutputRiskMeasure",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyMultiOutputRiskMeasure {
    inner: Arc<dyn McMultiOutputObjective>,
    mvar: Option<MVaR>,
}

impl PyMultiOutputRiskMeasure {
    fn plain(inner: Arc<dyn McMultiOutputObjective>) -> Self {
        Self { inner, mvar: None }
    }
}

#[pymethods]
impl PyMultiOutputRiskMeasure {
    /// `MVaR(n_w, alpha, expectation, preprocessing_function, pad_to_n_w,
    /// filter_dominated, use_counting)`.
    #[staticmethod]
    #[pyo3(signature = (
        n_w,
        alpha,
        expectation=false,
        preprocessing=None,
        pad_to_n_w=false,
        filter_dominated=true,
        use_counting=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn mvar(
        n_w: usize,
        alpha: f64,
        expectation: bool,
        preprocessing: Option<Bound<'_, PyAny>>,
        pad_to_n_w: bool,
        filter_dominated: bool,
        use_counting: bool,
    ) -> PyResult<Self> {
        let p = mo_preprocessing_of(preprocessing.as_ref())?;
        let mvar = MVaR::new(
            n_w,
            alpha,
            p,
            MVaROptions {
                expectation,
                pad_to_n_w,
                filter_dominated,
                use_counting,
            },
        )
        .into_py_result()?;
        Ok(Self {
            inner: Arc::new(mvar.clone()),
            mvar: Some(mvar),
        })
    }

    /// `IndependentCVaR(alpha, n_w, preprocessing_function)`.
    #[staticmethod]
    #[pyo3(signature = (alpha, n_w, preprocessing=None))]
    fn independent_cvar(
        alpha: f64,
        n_w: usize,
        preprocessing: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let p = mo_preprocessing_of(preprocessing.as_ref())?;
        Ok(Self::plain(Arc::new(
            IndependentCVaR::new(alpha, n_w, p).into_py_result()?,
        )))
    }

    /// `IndependentVaR(alpha, n_w, preprocessing_function)`.
    #[staticmethod]
    #[pyo3(signature = (alpha, n_w, preprocessing=None))]
    fn independent_var(
        alpha: f64,
        n_w: usize,
        preprocessing: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let p = mo_preprocessing_of(preprocessing.as_ref())?;
        Ok(Self::plain(Arc::new(
            IndependentVaR::new(alpha, n_w, p).into_py_result()?,
        )))
    }

    /// `MultiOutputWorstCase(n_w, preprocessing_function)`.
    #[staticmethod]
    #[pyo3(signature = (n_w, preprocessing=None))]
    fn worst_case(n_w: usize, preprocessing: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let p = mo_preprocessing_of(preprocessing.as_ref())?;
        Ok(Self::plain(Arc::new(
            MultiOutputWorstCase::new(n_w, p).into_py_result()?,
        )))
    }

    /// `MultiOutputExpectation(n_w, preprocessing_function)`.
    #[staticmethod]
    #[pyo3(signature = (n_w, preprocessing=None))]
    fn expectation(n_w: usize, preprocessing: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let p = mo_preprocessing_of(preprocessing.as_ref())?;
        Ok(Self::plain(Arc::new(
            MultiOutputExpectation::new(n_w, p).into_py_result()?,
        )))
    }

    /// The objective handle for the multi-objective acquisition functions.
    fn objective(&self) -> PyMultiOutputObjective {
        PyMultiOutputObjective::from_objective(Arc::clone(&self.inner))
    }

    /// The MVaR sets of `Y` (`batch x n_w x m`, the samples of one
    /// candidate per batch element): one `k x m` array per batch element
    /// (BoTorch's `get_mvar_set_vectorized`, or `get_mvar_set_via_counting`
    /// with `use_counting`).
    fn mvar_set(&self, y: Array, use_counting: bool) -> PyResult<Vec<ArrayOut>> {
        let base = self
            .mvar
            .as_ref()
            .ok_or_else(|| crate::errors::unsupported("This risk measure is not MVaR."))?;
        let mvar = MVaR::new(
            base.n_w(),
            base.alpha(),
            None,
            MVaROptions {
                use_counting,
                ..*base.options()
            },
        )
        .into_py_result()?;
        let (_, mats) = y.batches("Y")?;
        mats.iter()
            .map(|m| {
                let set = mvar.mvar_set(m).into_py_result()?;
                let k = set.len();
                let cols = m.ncols();
                Ok((set.into_iter().flatten().collect(), vec![k, cols]))
            })
            .collect()
    }

    /// The risk measure of `samples` (`... x n x m`): `... x n' x m'`.
    fn forward(&self, py: Python<'_>, samples: Array) -> PyResult<ArrayBuf> {
        let (lead, s) = samples_of(&samples)?;
        let obj = &self.inner;
        let v = threads::run(py, || obj.evaluate(&s, None, false)).into_py_result()?;
        let mut shape = lead;
        shape.extend([v.values.num_points(), v.values.num_outputs()]);
        Ok(ArrayBuf(v.values.to_row_major(), shape))
    }

    /// The vector-Jacobian product of [`PyMultiOutputRiskMeasure::forward`].
    fn vjp(&self, py: Python<'_>, samples: Array, adjoint: Array) -> PyResult<ArrayBuf> {
        let (_, s) = samples_of(&samples)?;
        let obj = &self.inner;
        let (n, m, ns) = (s.num_points(), s.num_outputs(), s.num_samples());
        let sbar = threads::run(py, || -> botorch_core::Result<Vec<f64>> {
            let v = obj.evaluate(&s, None, true)?;
            let (nv, mv) = (v.values.num_points(), v.values.num_outputs());
            if adjoint.0.len() != ns * nv * mv {
                return Err(botorch_core::BotorchError::DimensionMismatch(format!(
                    "the adjoint has {} values, expected {}",
                    adjoint.0.len(),
                    ns * nv * mv
                )));
            }
            let bar = McSamples::from_row_major(&adjoint.0, ns, nv, mv)?.into_data();
            let pullback = v.pullback.ok_or_else(|| {
                botorch_core::BotorchError::Runtime("the risk measure has no gradient".into())
            })?;
            let (sbar, _) = pullback(&bar)?;
            Ok(McSamples::new(sbar, n, m)?.to_row_major())
        })
        .into_py_result()?;
        Ok(ArrayBuf(sbar, samples.1.clone()))
    }
}

/// MARS's normalization bounds (`2 x m`) of the outcomes `Y` (`n x m`)
/// relative to `ref_point` (BoTorch's `MARS._get_Y_normalization_bounds`).
#[pyfunction]
#[pyo3(signature = (y, ref_point=None))]
pub fn mars_y_normalization_bounds(y: Array, ref_point: Option<Vec<f64>>) -> PyResult<ArrayOut> {
    if y.1.len() != 2 {
        return Err(crate::errors::unsupported("Batched Y is not supported."));
    }
    let y = y.matrix("Y")?;
    Ok(matrix_out(
        &y_normalization_bounds(&y, ref_point.as_deref()).into_py_result()?,
    ))
}

/// Registers the classes of this module.
/// Public MVaR component mapping and its VJP, in original row-major order.
#[pyfunction(name = "mvar_make_differentiable")]
#[pyo3(signature = (prepared_samples, mvars, adjoint=None))]
fn mvar_make_differentiable_py(
    py: Python<'_>,
    prepared_samples: Array,
    mvars: Array,
    adjoint: Option<Array>,
) -> PyResult<ArrayBuf> {
    let ps = &prepared_samples.1;
    let ms = &mvars.1;
    if ps.len() != 3
        || ms.len() != 3
        || ps[0] != ms[0]
        || ps[2] != ms[2]
        || adjoint.as_ref().is_some_and(|bar| bar.1 != *ms)
    {
        return Err(dimension_error("Expected prepared_samples [batch,n_w,m], mvars [batch,k,m], and optional matching adjoint"));
    }
    let length = |shape: &[usize]| shape.iter().try_fold(1usize, |n, &v| n.checked_mul(v));
    if length(ps) != Some(prepared_samples.0.len())
        || length(ms) != Some(mvars.0.len())
        || adjoint
            .as_ref()
            .is_some_and(|bar| length(&bar.1) != Some(bar.0.len()))
    {
        return Err(dimension_error("array data do not match their shapes"));
    }
    let (batch, nw, m, k) = (ps[0], ps[1], ps[2], ms[1]);
    let values = threads::run(py, || -> botorch_core::Result<Vec<f64>> {
        let mut result = Vec::new();
        for b in 0..batch {
            let prepared =
                Matrix::from_row_slice(nw, m, &prepared_samples.0[b * nw * m..(b + 1) * nw * m]);
            let points = Matrix::from_row_slice(k, m, &mvars.0[b * k * m..(b + 1) * k * m]);
            let bar = adjoint
                .as_ref()
                .map(|a| Matrix::from_row_slice(k, m, &a.0[b * k * m..(b + 1) * k * m]));
            let mapped = mvar_make_differentiable(&prepared, &points, bar.as_ref())?;
            for i in 0..mapped.nrows() {
                for t in 0..m {
                    result.push(mapped[(i, t)]);
                }
            }
        }
        Ok(result)
    })
    .into_py_result()?;
    Ok(ArrayBuf(
        values,
        if adjoint.is_some() {
            ps.clone()
        } else {
            ms.clone()
        },
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRiskMeasure>()?;
    m.add_class::<PyMultiOutputRiskMeasure>()?;
    m.add_function(wrap_pyfunction!(mars_y_normalization_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(mvar_make_differentiable_py, m)?)?;
    Ok(())
}
