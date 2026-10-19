//! Covariance modules as kernel trees (`botorch_core::kernels::KernelNode`):
//! the configuration of GPyTorch's and BoTorch's kernel classes that
//! `botorch_rs.kernels` and `botorch_rs.models.kernels` build and pass to a
//! `SingleTaskGP` as its `covar_module`.

use botorch_core::kernels::{KernelNode, MatrixPrior, TreeKernel};
use botorch_core::Matrix;
use pyo3::prelude::*;

use crate::arrays::{Array, ArrayBuf};
use crate::errors::{dimension_error, IntoPyResult};
use crate::specs::{kind_from_name, kind_name, PyConstraint, PyKernelSpec, PyPrior};

/// A kernel (tree) configuration: the kernel class, its parameters with
/// constraints and priors, `active_dims` and sub-kernels.
#[pyclass(frozen, name = "KernelNode", module = "botorch_rs._botorch_rs")]
pub struct PyKernelNode {
    /// The configuration.
    pub inner: KernelNode,
}

impl PyKernelNode {
    fn wrap(inner: KernelNode) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyKernelNode {
    /// Immutable: copies are the object itself.
    fn __copy__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Immutable: copies are the object itself.
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: &Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }

    /// `RBFKernel` / `MaternKernel` (`kind`: `"rbf"`, `"matern12"`,
    /// `"matern32"`, `"matern52"`).
    #[staticmethod]
    #[pyo3(signature = (kind, ard_num_dims=None))]
    fn stationary(kind: &str, ard_num_dims: Option<usize>) -> PyResult<Self> {
        KernelNode::stationary(kind_from_name(kind)?, ard_num_dims)
            .into_py_result()
            .map(Self::wrap)
    }

    /// `RQKernel`.
    #[staticmethod]
    #[pyo3(signature = (ard_num_dims=None))]
    fn rational_quadratic(ard_num_dims: Option<usize>) -> PyResult<Self> {
        KernelNode::rational_quadratic(ard_num_dims)
            .into_py_result()
            .map(Self::wrap)
    }

    /// `PeriodicKernel`.
    #[staticmethod]
    #[pyo3(signature = (ard_num_dims=None))]
    fn periodic(ard_num_dims: Option<usize>) -> PyResult<Self> {
        KernelNode::periodic(ard_num_dims)
            .into_py_result()
            .map(Self::wrap)
    }

    /// `LinearKernel`.
    #[staticmethod]
    #[pyo3(signature = (ard_num_dims=None))]
    fn linear(ard_num_dims: Option<usize>) -> PyResult<Self> {
        KernelNode::linear(ard_num_dims)
            .into_py_result()
            .map(Self::wrap)
    }

    /// `PolynomialKernel(power)`.
    #[staticmethod]
    fn polynomial(power: f64) -> PyResult<Self> {
        KernelNode::polynomial(power)
            .into_py_result()
            .map(Self::wrap)
    }

    /// `CosineKernel`.
    #[staticmethod]
    fn cosine() -> Self {
        Self::wrap(KernelNode::cosine())
    }

    /// `IndexKernel(num_tasks, rank)`.
    #[staticmethod]
    fn index(num_tasks: usize, rank: usize) -> PyResult<Self> {
        KernelNode::index(num_tasks, rank)
            .into_py_result()
            .map(Self::wrap)
    }

    /// BoTorch's `PositiveIndexKernel`.
    #[staticmethod]
    fn positive_index(
        num_tasks: usize,
        rank: usize,
        target_task_index: usize,
        unit_scale_for_target: bool,
    ) -> PyResult<Self> {
        KernelNode::positive_index(num_tasks, rank, target_task_index, unit_scale_for_target)
            .into_py_result()
            .map(Self::wrap)
    }

    /// BoTorch's `CategoricalKernel`.
    #[staticmethod]
    #[pyo3(signature = (ard_num_dims=None))]
    fn categorical(ard_num_dims: Option<usize>) -> PyResult<Self> {
        KernelNode::categorical(ard_num_dims)
            .into_py_result()
            .map(Self::wrap)
    }

    /// BoTorch's `DownsamplingKernel`.
    #[staticmethod]
    fn downsampling() -> Self {
        Self::wrap(KernelNode::downsampling())
    }

    /// Integrated white noise on scalar time inputs (no parameters).
    #[staticmethod]
    #[pyo3(signature = (order=1))]
    fn integrated_white_noise(order: usize) -> PyResult<Self> {
        KernelNode::integrated_white_noise(order)
            .map(Self::wrap)
            .into_py_result()
    }

    /// BoTorch's `ExponentialDecayKernel`.
    #[staticmethod]
    #[pyo3(signature = (ard_num_dims=None))]
    fn exponential_decay(ard_num_dims: Option<usize>) -> PyResult<Self> {
        KernelNode::exponential_decay(ard_num_dims)
            .into_py_result()
            .map(Self::wrap)
    }

    /// BoTorch's `InfiniteWidthBNNKernel`.
    #[staticmethod]
    fn infinite_width_bnn(depth: usize, acos_eps: f64) -> PyResult<Self> {
        KernelNode::infinite_width_bnn(depth, acos_eps)
            .into_py_result()
            .map(Self::wrap)
    }

    /// BoTorch's `LinearTruncatedFidelityKernel`.
    #[staticmethod]
    fn linear_truncated_fidelity(
        fidelity_dims: Vec<usize>,
        unbiased: PyRef<'_, PyKernelNode>,
        biased: PyRef<'_, PyKernelNode>,
    ) -> PyResult<Self> {
        KernelNode::linear_truncated_fidelity(
            fidelity_dims,
            unbiased.inner.clone(),
            biased.inner.clone(),
        )
        .into_py_result()
        .map(Self::wrap)
    }

    /// BoTorch's `OrthogonalAdditiveKernel` with an RBF or Matérn base kernel
    /// (a `KernelNode.stationary` without ARD).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn orthogonal_additive(
        dim: usize,
        base: PyRef<'_, PyKernelNode>,
        per_dim_lengthscales: bool,
        quad_deg: usize,
        second_order: bool,
        check_hypercube: bool,
        coeff_constraint: PyRef<'_, PyConstraint>,
    ) -> PyResult<Self> {
        KernelNode::orthogonal_additive(
            dim,
            base.inner.clone(),
            per_dim_lengthscales,
            quad_deg,
            second_order,
            check_hypercube,
            coeff_constraint.inner,
        )
        .into_py_result()
        .map(Self::wrap)
    }

    /// `ScaleKernel(base)`.
    #[staticmethod]
    fn scale(base: PyRef<'_, PyKernelNode>) -> Self {
        Self::wrap(KernelNode::scale(base.inner.clone()))
    }

    /// Structural additive contextual kernel with a shared RBF base.
    #[staticmethod]
    fn contextual_sac(
        decomposition: Vec<(String, Vec<usize>)>,
        base: PyRef<'_, PyKernelNode>,
    ) -> PyResult<Self> {
        KernelNode::contextual_sac(decomposition, base.inner.clone())
            .into_py_result()
            .map(Self::wrap)
    }

    /// `AdditiveKernel(*kernels)`.
    #[staticmethod]
    fn additive(kernels: Vec<PyRef<'_, PyKernelNode>>) -> PyResult<Self> {
        KernelNode::additive(kernels.iter().map(|k| k.inner.clone()).collect())
            .into_py_result()
            .map(Self::wrap)
    }

    /// `ProductKernel(*kernels)`.
    #[staticmethod]
    fn product(kernels: Vec<PyRef<'_, PyKernelNode>>) -> PyResult<Self> {
        KernelNode::product(kernels.iter().map(|k| k.inner.clone()).collect())
            .into_py_result()
            .map(Self::wrap)
    }

    /// This kernel with `active_dims`.
    fn with_active_dims(&self, active_dims: Vec<usize>) -> PyResult<Self> {
        self.inner
            .clone()
            .with_active_dims(active_dims)
            .into_py_result()
            .map(Self::wrap)
    }

    /// This kernel with the constraint, prior and raw initial value of its
    /// parameter `name` replaced (where given).
    #[pyo3(signature = (name, constraint=None, prior=None, default_raw=None))]
    fn with_param(
        &self,
        name: &str,
        constraint: Option<PyRef<'_, PyConstraint>>,
        prior: Option<PyRef<'_, PyPrior>>,
        default_raw: Option<f64>,
    ) -> PyResult<Self> {
        let mut node = self.inner.clone();
        if let Some(c) = constraint {
            node = node.with_constraint(name, c.inner).into_py_result()?;
        }
        if let Some(p) = prior {
            node = node.with_prior(name, p.inner).into_py_result()?;
        }
        if let Some(v) = default_raw {
            node = node.with_default_raw(name, v).into_py_result()?;
        }
        Ok(Self::wrap(node))
    }

    /// This kernel with a univariate task prior (`IndexKernel`:
    /// on every entry of the task covariance; `PositiveIndexKernel`: on the
    /// lower-triangle task correlations).
    fn with_task_prior(&self, prior: PyRef<'_, PyPrior>) -> PyResult<Self> {
        self.inner
            .clone()
            .with_task_prior(MatrixPrior::Elementwise(prior.inner))
            .into_py_result()
            .map(Self::wrap)
    }

    /// This kernel with GPyTorch's `LKJCovariancePrior(n, eta, sd_prior)` on
    /// the task covariance (`IndexKernel`).
    fn with_task_prior_lkj(
        &self,
        n: usize,
        eta: f64,
        sd_prior: PyRef<'_, PyPrior>,
    ) -> PyResult<Self> {
        self.inner
            .clone()
            .with_task_prior(MatrixPrior::LkjCovariance {
                n,
                eta,
                sd_prior: sd_prior.inner,
            })
            .into_py_result()
            .map(Self::wrap)
    }

    /// This kernel with BoTorch's SAAS prior (`add_saas_prior`): `prior`
    /// on `tau / lengthscale**2`, with the fixed `tau` or (`tau=None`) the
    /// learned parameter `raw_tau` with `tau_constraint` and `tau_prior`.
    #[pyo3(signature = (prior, tau=None, tau_constraint=None, tau_prior=None))]
    fn with_saas_prior(
        &self,
        prior: PyRef<'_, PyPrior>,
        tau: Option<f64>,
        tau_constraint: Option<PyRef<'_, PyConstraint>>,
        tau_prior: Option<PyRef<'_, PyPrior>>,
    ) -> PyResult<Self> {
        let tau_param = match (tau_constraint, tau_prior) {
            (Some(c), Some(p)) => Some((c.inner, p.inner)),
            _ => None,
        };
        self.inner
            .clone()
            .with_saas_prior(prior.inner, tau, tau_param)
            .into_py_result()
            .map(Self::wrap)
    }

    /// The fixed `tau`s of the SAAS priors in the tree (pre-order).
    fn saas_taus(&self) -> Vec<f64> {
        self.inner.saas_taus()
    }

    /// `(name, element shape, raw initial value)` of the kernel's own
    /// parameters (without those of its sub-kernels).
    fn own_params(&self) -> Vec<(String, Vec<usize>, f64)> {
        self.inner
            .params()
            .iter()
            .map(|p| (p.name.clone(), p.shape.clone(), p.initial_raw()))
            .collect()
    }

    /// `(full name, element shape)` of all parameters with the name prefix
    /// `prefix`, in `named_parameters()` order.
    #[pyo3(signature = (prefix=""))]
    fn param_shapes(&self, prefix: &str) -> Vec<(String, Vec<usize>)> {
        self.inner.param_shapes(prefix)
    }

    /// The number of raw values (per model element).
    #[getter]
    fn num_raw(&self) -> usize {
        self.inner.num_raw()
    }

    /// The raw initial values of all parameters (flat).
    fn initial_raw(&self) -> Vec<f64> {
        self.inner.initial_raw()
    }

    /// Checks the tree against `input_dim` inputs.
    fn validate(&self, input_dim: usize) -> PyResult<()> {
        self.inner.validate(input_dim).into_py_result()
    }

    /// The `KernelSpec` this tree is equivalent to for `input_dim` inputs
    /// (an RBF or Matérn kernel with ARD lengthscales, optionally in a
    /// `ScaleKernel`, without `active_dims`), if any.
    fn as_kernel_spec(&self, input_dim: usize) -> Option<PyKernelSpec> {
        self.inner
            .as_kernel_spec(input_dim)
            .map(|inner| PyKernelSpec { inner })
    }

    /// The kernel class name of the root (e.g. `"Stationary(Rbf)"`).
    #[getter]
    fn kind(&self) -> String {
        match self.inner.kind() {
            botorch_core::kernels::KernelType::Stationary(k) => kind_name(*k).to_string(),
            other => format!("{other:?}"),
        }
    }

    /// The kernel matrix `K(x1, x2)` (`n1 × n2`), or the `diag=True` values
    /// `k(x1_i, x1_i)` if `diag`, with the raw parameter values `raw` (flat,
    /// one model element) for inputs with `x1.shape[-1]` columns.
    #[pyo3(signature = (raw, x1, x2=None, diag=false))]
    fn evaluate(
        &self,
        py: Python<'_>,
        raw: Vec<f64>,
        x1: Array,
        x2: Option<Array>,
        diag: bool,
    ) -> PyResult<ArrayBuf> {
        let a = x1.matrix("x1")?;
        let b = match &x2 {
            Some(x) => Some(x.matrix("x2")?),
            None => None,
        };
        let d = a.ncols();
        if let Some(b) = &b {
            if b.ncols() != d {
                return Err(dimension_error(
                    "x1_ and x2_ must have the same number of dimensions!",
                ));
            }
        }
        let node = &self.inner;
        let out = crate::threads::run(py, || -> botorch_core::Result<ArrayBuf> {
            let kern = TreeKernel::build(node, &raw, d)?;
            let pa = kern.prepare(&a)?;
            let n = a.nrows();
            let other = match &b {
                Some(b) if b != &a => Some(kern.prepare(b)?),
                _ => None,
            };
            if diag {
                let values = match &other {
                    Some(pb) => {
                        if pb.len() != n {
                            return Err(botorch_core::BotorchError::DimensionMismatch(
                                "diag=True requires x1 and x2 with the same number of points"
                                    .into(),
                            ));
                        }
                        let k = kern.cross(&pa, pb);
                        (0..n).map(|i| k[(i, i)]).collect()
                    }
                    None => (0..n).map(|i| kern.diag_mode(&pa, i)).collect(),
                };
                return Ok(ArrayBuf(values, vec![n]));
            }
            let k: Matrix = match &other {
                Some(pb) => kern.cross(&pa, pb),
                None => kern.symmetric(&pa),
            };
            let (n1, n2) = k.shape();
            let mut data = Vec::with_capacity(n1 * n2);
            for i in 0..n1 {
                for j in 0..n2 {
                    data.push(k[(i, j)]);
                }
            }
            Ok(ArrayBuf(data, vec![n1, n2]))
        });
        out.into_py_result()
    }

    /// Input adjoints of `K(x1, x2)` without allocating its dense Jacobian.
    fn evaluate_vjp(
        &self,
        py: Python<'_>,
        raw: Vec<f64>,
        x1: Array,
        x2: Array,
        adjoint: Array,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let a = x1.matrix("x1")?;
        let b = x2.matrix("x2")?;
        let g = adjoint.matrix("adjoint")?;
        if a.ncols() != b.ncols() || g.shape() != (a.nrows(), b.nrows()) {
            return Err(dimension_error(
                "Kernel adjoint and input dimensions do not match",
            ));
        }
        crate::threads::run(py, || -> botorch_core::Result<_> {
            let kern = TreeKernel::build(&self.inner, &raw, a.ncols())?;
            let pa = kern.prepare(&a)?;
            let pb = kern.prepare(&b)?;
            let mut ga = Matrix::zeros(a.nrows(), a.ncols());
            let mut gb = Matrix::zeros(b.nrows(), b.ncols());
            for i in 0..a.nrows() {
                let (_, dg) = kern.cross_with_grad(&pa, i, &pb);
                for j in 0..b.nrows() {
                    for d in 0..a.ncols() {
                        ga[(i, d)] += g[(i, j)] * dg[(j, d)];
                    }
                }
            }
            for j in 0..b.nrows() {
                let (_, dg) = kern.cross_with_grad(&pb, j, &pa);
                for i in 0..a.nrows() {
                    for d in 0..b.ncols() {
                        gb[(j, d)] += g[(i, j)] * dg[(i, d)];
                    }
                }
            }
            Ok((
                crate::arrays::matrix_out(&ga).into(),
                crate::arrays::matrix_out(&gb).into(),
            ))
        })
        .into_py_result()
    }

    /// The log density of the priors and its gradient with respect to the raw
    /// values `raw` (for inputs with `input_dim` columns).
    fn log_prior(&self, raw: Vec<f64>, input_dim: usize) -> PyResult<(f64, Vec<f64>)> {
        let kern = TreeKernel::build(&self.inner, &raw, input_dim).into_py_result()?;
        let (v, g) = kern.log_prior_and_grad().into_py_result()?;
        let derivs = self.inner.transform_derivatives(&raw);
        Ok((v, g.iter().zip(derivs).map(|(g, d)| g * d).collect()))
    }

    fn __repr__(&self) -> String {
        format!("KernelNode({:?})", self.inner.kind())
    }
}

/// Gauss-Legendre nodes and weights of degree `deg` on `[-1, 1]`
/// (`numpy.polynomial.legendre.leggauss`).
#[pyfunction]
#[pyo3(name = "leggauss")]
pub fn leggauss_py(deg: usize) -> (Vec<f64>, Vec<f64>) {
    botorch_core::kernels::oak::leggauss(deg)
}

/// The log density of GPyTorch's `LKJCovariancePrior(n, eta, sd_prior)` at
/// `matrix`: the LKJ log density of the correlations plus the log densities
/// of the standard deviations.
#[pyfunction]
#[pyo3(name = "lkj_covariance_log_prob")]
pub fn lkj_covariance_log_prob_py(
    matrix: Array,
    eta: f64,
    sd_prior: PyRef<'_, PyPrior>,
) -> PyResult<f64> {
    let m = matrix.matrix("X")?;
    if m.nrows() != m.ncols() {
        return Err(dimension_error("Correlation matrix is not square"));
    }
    botorch_core::kernels::eval::lkj_covariance_log_prob(&m, eta, &sd_prior.inner)
        .map(|(v, _)| v)
        .into_py_result()
}
