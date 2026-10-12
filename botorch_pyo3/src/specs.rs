//! Configurations of GP modules: priors, constraints, covariance modules and
//! homoskedastic likelihoods (the GPyTorch objects that BoTorch's
//! `SingleTaskGP` accepts as `covar_module=` / `likelihood=`).
//!
//! GPyTorch stores prior parameters and constraint bounds created from Python
//! floats in torch's default dtype (float32) before the model is cast to
//! double precision; `float32=True` (the default) reproduces that rounding so
//! that models agree with BoTorch to double precision.

use botorch_core::constraints::{Constraint, Interval, ParameterTransform};
use botorch_core::kernels::{
    get_covar_module_with_dim_scaled_prior, get_matern_kernel_with_gamma_prior,
};
use botorch_core::models::{
    get_gaussian_likelihood_with_gamma_prior, get_gaussian_likelihood_with_lognormal_prior,
    MIN_INFERRED_NOISE_LEVEL,
};
use botorch_core::{
    GaussianLikelihoodSpec, GreaterThan, KernelKind, KernelSpec, Prior, ScaleKernelSpec,
};
use pyo3::prelude::*;

use crate::errors::{unsupported, value_error, IntoPyResult};

/// A hyperparameter prior (`gpytorch.priors.LogNormalPrior` / `GammaPrior`).
#[pyclass(frozen, name = "Prior", module = "botorch_rs._botorch_rs")]
pub struct PyPrior {
    /// The prior.
    pub inner: Prior,
}

#[pymethods]
impl PyPrior {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// `LogNormalPrior(loc, scale)`.
    #[staticmethod]
    #[pyo3(signature = (loc, scale, float32=true))]
    fn log_normal(loc: f64, scale: f64, float32: bool) -> PyResult<Self> {
        let p = Prior::log_normal(loc, scale).into_py_result()?;
        Ok(Self {
            inner: if float32 { p.rounded_to_f32() } else { p },
        })
    }

    /// `GammaPrior(concentration, rate)`.
    #[staticmethod]
    #[pyo3(signature = (concentration, rate, float32=true))]
    fn gamma(concentration: f64, rate: f64, float32: bool) -> PyResult<Self> {
        let p = Prior::gamma(concentration, rate).into_py_result()?;
        Ok(Self {
            inner: if float32 { p.rounded_to_f32() } else { p },
        })
    }

    /// `"LogNormal"` or `"Gamma"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            Prior::LogNormal { .. } => "LogNormal",
            Prior::Gamma { .. } => "Gamma",
        }
    }

    /// `(loc, scale)` or `(concentration, rate)` (as stored).
    #[getter]
    fn parameters(&self) -> (f64, f64) {
        match self.inner {
            Prior::LogNormal { loc, scale, .. } => (loc, scale),
            Prior::Gamma {
                concentration,
                rate,
            } => (concentration, rate),
        }
    }

    /// The log density at `x`.
    fn log_prob(&self, x: f64) -> f64 {
        self.inner.log_prob(x)
    }

    /// The mode.
    #[getter]
    fn mode(&self) -> f64 {
        self.inner.mode()
    }
}

fn transform_from_name(name: Option<&str>) -> PyResult<Option<ParameterTransform>> {
    match name {
        None => Ok(None),
        Some("softplus") => Ok(Some(ParameterTransform::Softplus)),
        Some("sigmoid") => Ok(Some(ParameterTransform::Sigmoid)),
        Some(other) => Err(unsupported(format!(
            "Only the softplus and sigmoid transforms (or None) are supported, got {other:?}."
        ))),
    }
}

/// A parameter constraint: `gpytorch.constraints.GreaterThan` / `Positive`
/// or `Interval`.
#[pyclass(frozen, name = "Constraint", module = "botorch_rs._botorch_rs")]
pub struct PyConstraint {
    /// The constraint.
    pub inner: Constraint,
}

#[pymethods]
impl PyConstraint {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// `GreaterThan(lower_bound, transform, initial_value)`; `transform` is
    /// `"softplus"` (GPyTorch's default) or `None` (not enforced: the bound is
    /// imposed by the optimizer, as in BoTorch's default models).
    #[staticmethod]
    #[pyo3(signature = (lower_bound, transform=Some("softplus"), initial_value=None, float32=true))]
    fn greater_than(
        lower_bound: f64,
        transform: Option<&str>,
        initial_value: Option<f64>,
        float32: bool,
    ) -> PyResult<Self> {
        let transform = match transform_from_name(transform)? {
            Some(ParameterTransform::Sigmoid) => {
                return Err(unsupported(
                    "GreaterThan supports the softplus transform (or None) in botorch_rs.",
                ))
            }
            t => t,
        };
        if lower_bound.is_nan() || lower_bound == f64::INFINITY {
            return Err(value_error(format!("invalid lower bound {lower_bound}")));
        }
        if let Some(v) = initial_value {
            if !(v.is_finite() && v >= lower_bound) {
                return Err(value_error(format!(
                    "initial_value {v} violates the lower bound {lower_bound}"
                )));
            }
        }
        let c = GreaterThan::new(lower_bound, transform, initial_value);
        Ok(Self {
            inner: if float32 { c.rounded_to_f32() } else { c }.into(),
        })
    }

    /// `Interval(lower_bound, upper_bound, transform, inv_transform,
    /// initial_value)`; `transform` is `"sigmoid"` (GPyTorch's default),
    /// `"softplus"` or `None` (not enforced: box bounds in the optimizer),
    /// `inv_transform` the transform whose inverse maps values to raw values
    /// (`None`: GPyTorch's default `inv_sigmoid`).
    #[staticmethod]
    #[pyo3(signature = (
        lower_bound,
        upper_bound,
        transform=Some("sigmoid"),
        inv_transform=None,
        initial_value=None,
        float32=true,
    ))]
    fn interval(
        lower_bound: f64,
        upper_bound: f64,
        transform: Option<&str>,
        inv_transform: Option<&str>,
        initial_value: Option<f64>,
        float32: bool,
    ) -> PyResult<Self> {
        let transform = transform_from_name(transform)?;
        let inverse = transform_from_name(inv_transform)?;
        if float32 {
            // GPyTorch validates the bounds after storing them in float32.
            Interval::new(
                lower_bound as f32 as f64,
                upper_bound as f32 as f64,
                transform,
                inverse,
                None,
            )
            .into_py_result()?;
        }
        let c = Interval::new(lower_bound, upper_bound, transform, inverse, initial_value)
            .into_py_result()?;
        Ok(Self {
            inner: if float32 { c.rounded_to_f32() } else { c }.into(),
        })
    }

    /// `"GreaterThan"` or `"Interval"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            Constraint::GreaterThan(_) => "GreaterThan",
            Constraint::Interval(_) => "Interval",
        }
    }

    /// The lower bound.
    #[getter]
    fn lower_bound(&self) -> f64 {
        self.inner.lower_bound()
    }

    /// The upper bound (`inf` for `GreaterThan`).
    #[getter]
    fn upper_bound(&self) -> f64 {
        self.inner.upper_bound()
    }

    /// Whether the constraint is enforced through a transform.
    #[getter]
    fn enforced(&self) -> bool {
        self.inner.enforced()
    }

    /// The name of the transform (`None` if not enforced).
    #[getter]
    fn transform_name(&self) -> Option<&'static str> {
        self.inner.transform_kind().map(ParameterTransform::name)
    }

    /// The initial value (actual space), if any.
    #[getter]
    fn initial_value(&self) -> Option<f64> {
        self.inner.initial_value()
    }

    /// The raw initial value (GPyTorch's `constraint.initial_value`), if any.
    #[getter]
    fn initial_raw_value(&self) -> Option<f64> {
        self.inner.initial_raw_value()
    }

    /// Optimizer bounds of the raw parameter (`get_parameters_and_bounds`).
    #[getter]
    fn raw_bounds(&self) -> (f64, f64) {
        self.inner.raw_bounds()
    }

    /// Maps a raw value to the actual value.
    fn transform(&self, raw: f64) -> f64 {
        self.inner.transform(raw)
    }

    /// Maps an actual value to the raw value.
    fn inverse_transform(&self, value: f64) -> f64 {
        self.inner.inverse_transform(value)
    }

    /// The derivative of the transform at a raw value.
    fn transform_derivative(&self, raw: f64) -> f64 {
        self.inner.transform_derivative(raw)
    }
}

/// A covariance module: an ARD RBF or Matérn-5/2 kernel with optional prior
/// and constraint on the lengthscales, optionally wrapped in a `ScaleKernel`.
#[pyclass(frozen, name = "KernelSpec", module = "botorch_rs._botorch_rs")]
pub struct PyKernelSpec {
    /// The configuration.
    pub inner: KernelSpec,
}

fn kind_name(kind: KernelKind) -> &'static str {
    match kind {
        KernelKind::Rbf => "rbf",
        KernelKind::Matern52 => "matern52",
    }
}

#[pymethods]
impl PyKernelSpec {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// A base kernel (`"rbf"` or `"matern52"`) with GPyTorch's defaults for
    /// omitted arguments (no prior, `Positive()` constraint).
    #[new]
    #[pyo3(signature = (kind, ard_num_dims, lengthscale_prior=None, lengthscale_constraint=None))]
    fn new(
        kind: &str,
        ard_num_dims: usize,
        lengthscale_prior: Option<PyRef<'_, PyPrior>>,
        lengthscale_constraint: Option<PyRef<'_, PyConstraint>>,
    ) -> PyResult<Self> {
        let kind = match kind {
            "rbf" => KernelKind::Rbf,
            "matern52" => KernelKind::Matern52,
            other => {
                return Err(unsupported(format!(
                    "Unsupported kernel {other:?} (RBF and Matern-5/2 are available)."
                )))
            }
        };
        if ard_num_dims == 0 {
            return Err(value_error("ard_num_dims must be positive"));
        }
        let mut spec = KernelSpec::new(kind, ard_num_dims);
        if let Some(p) = lengthscale_prior {
            spec.lengthscale_prior = Some(p.inner);
        }
        if let Some(c) = lengthscale_constraint {
            spec.lengthscale_constraint = c.inner;
        }
        Ok(Self { inner: spec })
    }

    /// BoTorch's `get_covar_module_with_dim_scaled_prior(ard_num_dims,
    /// use_rbf_kernel)`.
    #[staticmethod]
    #[pyo3(signature = (ard_num_dims, use_rbf_kernel=true))]
    fn dim_scaled(ard_num_dims: usize, use_rbf_kernel: bool) -> PyResult<Self> {
        if ard_num_dims == 0 {
            return Err(value_error("ard_num_dims must be positive"));
        }
        Ok(Self {
            inner: get_covar_module_with_dim_scaled_prior(ard_num_dims, use_rbf_kernel),
        })
    }

    /// BoTorch's legacy `get_matern_kernel_with_gamma_prior(ard_num_dims)`.
    #[staticmethod]
    fn matern_gamma(ard_num_dims: usize) -> PyResult<Self> {
        if ard_num_dims == 0 {
            return Err(value_error("ard_num_dims must be positive"));
        }
        Ok(Self {
            inner: get_matern_kernel_with_gamma_prior(ard_num_dims),
        })
    }

    /// This kernel wrapped in a `ScaleKernel` (GPyTorch defaults: no prior,
    /// `Positive()` constraint).
    #[pyo3(signature = (outputscale_prior=None, outputscale_constraint=None))]
    fn with_scale_kernel(
        &self,
        outputscale_prior: Option<PyRef<'_, PyPrior>>,
        outputscale_constraint: Option<PyRef<'_, PyConstraint>>,
    ) -> PyResult<Self> {
        if self.inner.scale.is_some() {
            return Err(unsupported(
                "Nested ScaleKernels are not supported.".to_string(),
            ));
        }
        let mut scale = ScaleKernelSpec::new(outputscale_prior.map(|p| p.inner));
        if let Some(c) = outputscale_constraint {
            scale.outputscale_constraint = c.inner;
        }
        Ok(Self {
            inner: self.inner.with_scale_kernel(scale),
        })
    }

    /// `"rbf"` or `"matern52"`.
    #[getter]
    fn kind(&self) -> &'static str {
        kind_name(self.inner.kind)
    }

    /// Number of lengthscales.
    #[getter]
    fn ard_num_dims(&self) -> usize {
        self.inner.ard_num_dims
    }

    /// Whether a `ScaleKernel` wraps the base kernel.
    #[getter]
    fn has_scale_kernel(&self) -> bool {
        self.inner.scale.is_some()
    }

    /// The lengthscale prior.
    #[getter]
    fn lengthscale_prior(&self) -> Option<PyPrior> {
        self.inner.lengthscale_prior.map(|inner| PyPrior { inner })
    }

    /// The lengthscale constraint.
    #[getter]
    fn lengthscale_constraint(&self) -> PyConstraint {
        PyConstraint {
            inner: self.inner.lengthscale_constraint,
        }
    }

    /// The outputscale prior (`None` without prior or `ScaleKernel`).
    #[getter]
    fn outputscale_prior(&self) -> Option<PyPrior> {
        self.inner
            .scale
            .and_then(|s| s.outputscale_prior)
            .map(|inner| PyPrior { inner })
    }

    /// The outputscale constraint (`None` without `ScaleKernel`).
    #[getter]
    fn outputscale_constraint(&self) -> Option<PyConstraint> {
        self.inner.scale.map(|s| PyConstraint {
            inner: s.outputscale_constraint,
        })
    }
}

/// A homoskedastic Gaussian likelihood (`gpytorch.likelihoods.GaussianLikelihood`).
#[pyclass(
    frozen,
    name = "GaussianLikelihoodSpec",
    module = "botorch_rs._botorch_rs"
)]
pub struct PyLikelihoodSpec {
    /// The configuration.
    pub inner: GaussianLikelihoodSpec,
}

#[pymethods]
impl PyLikelihoodSpec {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// `GaussianLikelihood(noise_prior, noise_constraint)`; the default
    /// constraint is GPyTorch's `GreaterThan(1e-4)` (softplus).
    #[new]
    #[pyo3(signature = (noise_prior=None, noise_constraint=None))]
    fn new(
        noise_prior: Option<PyRef<'_, PyPrior>>,
        noise_constraint: Option<PyRef<'_, PyConstraint>>,
    ) -> Self {
        Self {
            inner: GaussianLikelihoodSpec {
                noise_prior: noise_prior.map(|p| p.inner),
                noise_constraint: match noise_constraint {
                    Some(c) => c.inner,
                    None => GreaterThan::new(
                        MIN_INFERRED_NOISE_LEVEL,
                        Some(ParameterTransform::Softplus),
                        None,
                    )
                    .rounded_to_f32()
                    .into(),
                },
            },
        }
    }

    /// BoTorch's `get_gaussian_likelihood_with_lognormal_prior()`.
    #[staticmethod]
    fn lognormal() -> Self {
        Self {
            inner: get_gaussian_likelihood_with_lognormal_prior(),
        }
    }

    /// BoTorch's `get_gaussian_likelihood_with_gamma_prior()`.
    #[staticmethod]
    fn gamma() -> Self {
        Self {
            inner: get_gaussian_likelihood_with_gamma_prior(),
        }
    }

    /// The noise prior.
    #[getter]
    fn noise_prior(&self) -> Option<PyPrior> {
        self.inner.noise_prior.map(|inner| PyPrior { inner })
    }

    /// The noise constraint.
    #[getter]
    fn noise_constraint(&self) -> PyConstraint {
        PyConstraint {
            inner: self.inner.noise_constraint,
        }
    }
}
