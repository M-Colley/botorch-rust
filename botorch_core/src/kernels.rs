//! Stationary ARD kernels (RBF and Matérn) with an optional `ScaleKernel`
//! wrapper, following GPyTorch's parameterization exactly:
//!
//! * inputs are divided by the lengthscales first (`x / ℓ`), then distances are
//!   computed on the scaled inputs;
//! * RBF: `k(x, y) = exp(-r² / 2)` with `r² = Σⱼ ((xⱼ - yⱼ) / ℓⱼ)²`;
//! * Matérn-ν (ν = 1/2, 3/2, 5/2): `exp(-r)`, `(1 + √3 r) exp(-√3 r)` and
//!   `(1 + √5 r + 5/3 r²) exp(-√5 r)`, where the distance `r` is clamped from
//!   below at `1e-15` (as in `gpytorch.kernels.kernel.dist`);
//! * `ScaleKernel`: `k_s(x, y) = s · k(x, y)` with outputscale `s`.
//!
//! [`KernelSpec`] describes a covariance *module* (kernel type, priors,
//! constraints, initialization) like a GPyTorch kernel object, whereas
//! [`Kernel`] is a kernel with concrete hyperparameter values that can be
//! evaluated, together with its derivatives.
//!
//! General covariance modules -- trees of GPyTorch and BoTorch kernels with
//! `active_dims`, sums, products and `ScaleKernel`s -- are described by
//! [`KernelNode`] (see [`tree`]); [`CovarModule`] is either kind.

use crate::constraints::{Constraint, GreaterThan};
use crate::error::{BotorchError, Result};
use crate::priors::Prior;
use crate::{Matrix, Vector};

pub mod contextual;
mod covar;
pub mod empirical;
pub mod eval;
pub mod nonfinite;
pub mod oak;
pub mod tree;

pub use covar::{BuiltCovar, CovarModule, Prepared};
pub use eval::TreeKernel;
pub use nonfinite::TrainingNan;
pub use tree::{KernelNode, KernelType, MatrixPrior, ParamSpec, SaasPrior};

/// Lower clamp of the Matérn distance (`gpytorch.kernels.kernel.dist`).
pub const MATERN_MIN_DISTANCE: f64 = 1e-15;

const SQRT3: f64 = 1.732_050_807_568_877_2;
const SQRT5: f64 = 2.236_067_977_499_79;

/// The base kernel family of a stationary kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum KernelKind {
    /// Squared-exponential kernel (`gpytorch.kernels.RBFKernel`); BoTorch's
    /// default since version 0.12.
    #[default]
    Rbf,
    /// Matérn kernel with smoothness ν = 5/2 (`gpytorch.kernels.MaternKernel(nu=2.5)`).
    Matern52,
    /// Matérn kernel with smoothness ν = 1/2 (`MaternKernel(nu=0.5)`, the
    /// exponential kernel).
    Matern12,
    /// Matérn kernel with smoothness ν = 3/2 (`MaternKernel(nu=1.5)`).
    Matern32,
}

impl KernelKind {
    /// The Matérn kernel with smoothness `nu` (0.5, 1.5 or 2.5).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] with GPyTorch's message for other
    /// values of `nu`.
    pub fn matern(nu: f64) -> Result<Self> {
        if nu == 0.5 {
            Ok(KernelKind::Matern12)
        } else if nu == 1.5 {
            Ok(KernelKind::Matern32)
        } else if nu == 2.5 {
            Ok(KernelKind::Matern52)
        } else {
            Err(BotorchError::InvalidArgument(
                "nu expected to be 0.5, 1.5, or 2.5".into(),
            ))
        }
    }

    /// The smoothness ν of a Matérn kernel (`None` for RBF).
    pub fn nu(self) -> Option<f64> {
        match self {
            KernelKind::Rbf => None,
            KernelKind::Matern12 => Some(0.5),
            KernelKind::Matern32 => Some(1.5),
            KernelKind::Matern52 => Some(2.5),
        }
    }

    /// The kernel value as a function of the squared scaled distance `r²`
    /// (the Matérn distance clamped at [`MATERN_MIN_DISTANCE`]).
    #[inline]
    pub fn value(self, r2: f64) -> f64 {
        match self {
            KernelKind::Rbf => (r2 / -2.0).exp(),
            KernelKind::Matern52 => {
                let r = r2.sqrt().max(MATERN_MIN_DISTANCE);
                (SQRT5 * r + 1.0 + 5.0 / 3.0 * (r * r)) * (-SQRT5 * r).exp()
            }
            KernelKind::Matern12 => {
                let r = r2.sqrt().max(MATERN_MIN_DISTANCE);
                (-r).exp()
            }
            KernelKind::Matern32 => {
                let r = r2.sqrt().max(MATERN_MIN_DISTANCE);
                (SQRT3 * r + 1.0) * (-SQRT3 * r).exp()
            }
        }
    }

    /// The kernel value in GPyTorch's `diag=True` mode at a point and
    /// itself: the distance is exactly zero (not clamped), so every kernel
    /// is 1.
    #[inline]
    pub fn diag_value(self) -> f64 {
        1.0
    }

    /// The kernel value `k` and the derivative factor `g` such that
    /// `∂k/∂uⱼ = -g·uⱼ` where `uⱼ` is the scaled coordinate difference.
    #[inline]
    pub fn value_and_factor(self, r2: f64) -> (f64, f64) {
        match self {
            KernelKind::Rbf => {
                let k = (r2 / -2.0).exp();
                (k, k)
            }
            KernelKind::Matern52 => {
                let r_true = r2.sqrt();
                let r = r_true.max(MATERN_MIN_DISTANCE);
                let e = (-SQRT5 * r).exp();
                let k = (SQRT5 * r + 1.0 + 5.0 / 3.0 * (r * r)) * e;
                // The distance is clamped in GPyTorch, so the gradient vanishes there
                // (also at equality: a scalar bound, torch ≥ 2.14).
                let g = if r_true <= MATERN_MIN_DISTANCE {
                    0.0
                } else {
                    5.0 / 3.0 * (1.0 + SQRT5 * r) * e
                };
                (k, g)
            }
            KernelKind::Matern12 => {
                let r_true = r2.sqrt();
                let r = r_true.max(MATERN_MIN_DISTANCE);
                let k = (-r).exp();
                let g = if r_true <= MATERN_MIN_DISTANCE {
                    0.0
                } else {
                    k / r
                };
                (k, g)
            }
            KernelKind::Matern32 => {
                let r_true = r2.sqrt();
                let r = r_true.max(MATERN_MIN_DISTANCE);
                let e = (-SQRT3 * r).exp();
                let k = (SQRT3 * r + 1.0) * e;
                let g = if r_true <= MATERN_MIN_DISTANCE {
                    0.0
                } else {
                    3.0 * e
                };
                (k, g)
            }
        }
    }
}

/// Configuration of a `ScaleKernel` wrapper (`gpytorch.kernels.ScaleKernel`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleKernelSpec {
    /// Prior on the outputscale (evaluated on the actual outputscale).
    pub outputscale_prior: Option<Prior>,
    /// Constraint on the outputscale (GPyTorch default: `Positive()`).
    pub outputscale_constraint: Constraint,
}

impl ScaleKernelSpec {
    /// GPyTorch's `ScaleKernel` defaults (no prior, `Positive()` constraint,
    /// raw outputscale initialized to zero).
    pub fn new(outputscale_prior: Option<Prior>) -> Self {
        Self {
            outputscale_prior,
            outputscale_constraint: Constraint::positive(),
        }
    }

    /// The `ScaleKernel` of BoTorch's legacy `get_matern_kernel_with_gamma_prior`:
    /// `GammaPrior(2.0, 0.15)` on the outputscale and a `Positive()` constraint.
    pub fn legacy_gamma() -> Self {
        Self::new(Some(
            Prior::Gamma {
                concentration: 2.0,
                rate: 0.15,
            }
            .rounded_to_f32(),
        ))
    }

    /// Initial raw outputscale (GPyTorch initializes `raw_outputscale` to 0
    /// unless the constraint carries an initial value).
    pub fn initial_raw_outputscale(&self) -> f64 {
        self.outputscale_constraint
            .initial_raw_value()
            .unwrap_or(0.0)
    }
}

/// Configuration of a covariance module: kernel family, ARD dimension,
/// lengthscale prior and constraint, and an optional `ScaleKernel` wrapper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelSpec {
    /// The base kernel family.
    pub kind: KernelKind,
    /// Number of input dimensions (one lengthscale per dimension).
    pub ard_num_dims: usize,
    /// Prior on each lengthscale (evaluated on the actual lengthscales).
    pub lengthscale_prior: Option<Prior>,
    /// Constraint on the lengthscales.
    pub lengthscale_constraint: Constraint,
    /// Optional `ScaleKernel` wrapper.
    pub scale: Option<ScaleKernelSpec>,
}

impl KernelSpec {
    /// A GPyTorch kernel with library defaults: no prior, `Positive()`
    /// lengthscale constraint (softplus), raw lengthscales initialized to zero.
    pub fn new(kind: KernelKind, ard_num_dims: usize) -> Self {
        Self {
            kind,
            ard_num_dims,
            lengthscale_prior: None,
            lengthscale_constraint: Constraint::positive(),
            scale: None,
        }
    }

    /// Wraps the kernel in a `ScaleKernel`.
    pub fn with_scale_kernel(mut self, scale: ScaleKernelSpec) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Initial raw lengthscale (shared by all dimensions): the constraint's
    /// initial value if present, otherwise zero (GPyTorch's default).
    pub fn initial_raw_lengthscale(&self) -> f64 {
        self.lengthscale_constraint
            .initial_raw_value()
            .unwrap_or(0.0)
    }

    /// Initial raw outputscale if a `ScaleKernel` is used.
    pub fn initial_raw_outputscale(&self) -> Option<f64> {
        self.scale.map(|s| s.initial_raw_outputscale())
    }

    /// Builds the concrete kernel for the given raw hyperparameters.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `raw_lengthscale.len() != ard_num_dims`,
    /// [`BotorchError::InvalidArgument`] if the scale-kernel configuration and
    /// `raw_outputscale` disagree or a value is not positive.
    pub fn build(&self, raw_lengthscale: &[f64], raw_outputscale: Option<f64>) -> Result<Kernel> {
        if raw_lengthscale.len() != self.ard_num_dims {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw lengthscales, got {}",
                self.ard_num_dims,
                raw_lengthscale.len()
            )));
        }
        let lengthscale = Vector::from_iterator(
            raw_lengthscale.len(),
            raw_lengthscale
                .iter()
                .map(|&r| self.lengthscale_constraint.transform(r)),
        );
        let outputscale = match (self.scale, raw_outputscale) {
            (Some(s), Some(raw)) => Some(s.outputscale_constraint.transform(raw)),
            (None, None) => None,
            _ => {
                return Err(BotorchError::InvalidArgument(
                    "raw outputscale must be given if and only if a ScaleKernel is used".into(),
                ))
            }
        };
        Kernel::new(self.kind, lengthscale, outputscale)
    }

    /// [`KernelSpec::build`] without the checks that the lengthscales and
    /// the outputscale are positive and finite, for the hyperparameters that
    /// raw parameters produce during a fit: softplus underflows to zero for
    /// raw values below about -745, and GPyTorch evaluates the kernel
    /// anyway (see [`Kernel::training_nan`]).
    ///
    /// # Errors
    /// The dimension and structure errors of [`KernelSpec::build`].
    pub(crate) fn build_unchecked(
        &self,
        raw_lengthscale: &[f64],
        raw_outputscale: Option<f64>,
    ) -> Result<Kernel> {
        if raw_lengthscale.len() != self.ard_num_dims || raw_lengthscale.is_empty() {
            return self.build(raw_lengthscale, raw_outputscale);
        }
        let lengthscale = Vector::from_iterator(
            raw_lengthscale.len(),
            raw_lengthscale
                .iter()
                .map(|&r| self.lengthscale_constraint.transform(r)),
        );
        let outputscale = match (self.scale, raw_outputscale) {
            (Some(s), Some(raw)) => Some(s.outputscale_constraint.transform(raw)),
            (None, None) => None,
            _ => return self.build(raw_lengthscale, raw_outputscale),
        };
        Ok(Kernel {
            kind: self.kind,
            lengthscale,
            outputscale,
        })
    }
}

/// BoTorch's `get_covar_module_with_dim_scaled_prior` (the `SingleTaskGP`
/// default): an RBF (or Matérn-5/2 if `use_rbf_kernel` is false) ARD kernel
/// with lengthscale prior `LogNormal(√2 + log(d)/2, √3)` and constraint
/// `GreaterThan(2.5e-2, transform=None, initial_value=prior.mode)`.
///
/// The prior parameters, bound and initial value are rounded to float32
/// exactly as in BoTorch (see [`Prior::rounded_to_f32`]).
pub fn get_covar_module_with_dim_scaled_prior(
    ard_num_dims: usize,
    use_rbf_kernel: bool,
) -> KernelSpec {
    let d = ard_num_dims as f64;
    let prior = Prior::LogNormal {
        loc: std::f64::consts::SQRT_2 + d.ln() * 0.5,
        scale: 3f64.sqrt(),
        float32: false,
        torch_log: None,
    }
    .rounded_to_f32();
    let constraint: Constraint = GreaterThan::new(2.5e-2, None, Some(prior.mode_f32()))
        .rounded_to_f32()
        .into();
    KernelSpec {
        kind: if use_rbf_kernel {
            KernelKind::Rbf
        } else {
            KernelKind::Matern52
        },
        ard_num_dims,
        lengthscale_prior: Some(prior),
        lengthscale_constraint: constraint,
        scale: None,
    }
}

/// BoTorch's legacy `get_matern_kernel_with_gamma_prior`:
/// `ScaleKernel(MaternKernel(nu=2.5, lengthscale_prior=GammaPrior(3.0, 6.0)),
/// outputscale_prior=GammaPrior(2.0, 0.15))` with GPyTorch's default
/// `Positive()` (softplus) constraints.
pub fn get_matern_kernel_with_gamma_prior(ard_num_dims: usize) -> KernelSpec {
    KernelSpec {
        kind: KernelKind::Matern52,
        ard_num_dims,
        lengthscale_prior: Some(
            Prior::Gamma {
                concentration: 3.0,
                rate: 6.0,
            }
            .rounded_to_f32(),
        ),
        lengthscale_constraint: Constraint::positive(),
        scale: Some(ScaleKernelSpec::legacy_gamma()),
    }
}

/// Derivatives of `K(X, X)` with respect to the actual hyperparameters.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelGradients {
    /// The covariance matrix `K(X, X)`.
    pub covariance: Matrix,
    /// `∂K/∂ℓⱼ` for every lengthscale `j`.
    pub lengthscale: Vec<Matrix>,
    /// `∂K/∂s` if a `ScaleKernel` is used.
    pub outputscale: Option<Matrix>,
}

/// Row-major copy of a point set with every coordinate divided by its
/// lengthscale (GPyTorch's `x.div(lengthscale)`).
#[derive(Clone, Debug)]
pub struct ScaledPoints {
    data: Vec<f64>,
    n: usize,
    d: usize,
}

impl ScaledPoints {
    /// The number of points.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether there are no points.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    #[inline]
    pub(crate) fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
}

/// A stationary ARD kernel with concrete hyperparameter values.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    kind: KernelKind,
    lengthscale: Vector,
    outputscale: Option<f64>,
}

impl Kernel {
    /// Creates a kernel with actual (positive) lengthscales and an optional
    /// outputscale (`ScaleKernel`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if there are no lengthscales or a value
    /// is not positive and finite.
    pub fn new(kind: KernelKind, lengthscale: Vector, outputscale: Option<f64>) -> Result<Self> {
        if lengthscale.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "at least one lengthscale is required".into(),
            ));
        }
        if lengthscale.iter().any(|&l| !(l > 0.0 && l.is_finite())) {
            return Err(BotorchError::InvalidArgument(format!(
                "lengthscales must be positive and finite, got {:?}",
                lengthscale.as_slice()
            )));
        }
        if let Some(s) = outputscale {
            if !(s > 0.0 && s.is_finite()) {
                return Err(BotorchError::InvalidArgument(format!(
                    "outputscale must be positive and finite, got {s}"
                )));
            }
        }
        Ok(Self {
            kind,
            lengthscale,
            outputscale,
        })
    }

    /// GPyTorch's NaN entries of the training covariance `K(X, X)` (`X` the
    /// `n × d` training inputs) and the lengthscales whose gradient is NaN
    /// (see [`nonfinite`]). `inputs_require_grad`: the inputs are learned
    /// (e.g. LCEMGP's embeddings), which selects GPyTorch's autograd path
    /// also for a single lengthscale.
    pub fn training_nan(&self, x: &Matrix, inputs_require_grad: bool) -> TrainingNan {
        let (n, d) = x.shape();
        if n == 0 || d != self.dim() {
            return TrainingNan::default();
        }
        let data: Vec<f64> = (0..n)
            .flat_map(|i| (0..d).map(move |j| x[(i, j)]))
            .collect();
        let function_path = d == 1 && !inputs_require_grad;
        // A `ScaleKernel` multiplies by a finite outputscale: the base
        // kernel values are bounded, so its NaN entries are the base's.
        nonfinite::stationary(
            self.kind,
            &data,
            n,
            d,
            self.lengthscale.as_slice(),
            function_path,
        )
    }

    /// The kernel family.
    pub fn kind(&self) -> KernelKind {
        self.kind
    }

    /// The (actual) lengthscales.
    pub fn lengthscale(&self) -> &Vector {
        &self.lengthscale
    }

    /// The (actual) outputscale, if a `ScaleKernel` is used.
    pub fn outputscale(&self) -> Option<f64> {
        self.outputscale
    }

    /// Input dimension.
    pub fn dim(&self) -> usize {
        self.lengthscale.len()
    }

    #[inline]
    fn scale_factor(&self) -> f64 {
        self.outputscale.unwrap_or(1.0)
    }

    fn check_dim(&self, x: &Matrix, what: &str) -> Result<()> {
        if x.ncols() != self.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "{what} has {} columns but the kernel has {} lengthscales",
                x.ncols(),
                self.dim()
            )));
        }
        Ok(())
    }

    /// Scales a point set by the lengthscales into a row-major buffer.
    pub(crate) fn scale_points(&self, x: &Matrix) -> ScaledPoints {
        let (n, d) = x.shape();
        debug_assert_eq!(d, self.dim());
        let mut data = vec![0.0; n * d];
        for j in 0..d {
            let l = self.lengthscale[j];
            for i in 0..n {
                data[i * d + j] = x[(i, j)] / l;
            }
        }
        ScaledPoints { data, n, d }
    }

    /// Scales a single point by the lengthscales.
    pub(crate) fn scale_point(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.lengthscale.iter())
            .map(|(v, l)| v / l)
            .collect()
    }

    /// Base kernel value as a function of the squared scaled distance.
    #[inline]
    fn base_value(&self, r2: f64) -> f64 {
        self.kind.value(r2)
    }

    /// Base kernel value `k` and the derivative factor `g` such that
    /// `∂k/∂uⱼ = -g·uⱼ` where `uⱼ` is the scaled coordinate difference.
    #[inline]
    fn base_value_and_factor(&self, r2: f64) -> (f64, f64) {
        self.kind.value_and_factor(r2)
    }

    #[inline]
    fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Cross-covariance on pre-scaled points.
    pub(crate) fn cross_scaled(&self, a: &ScaledPoints, b: &ScaledPoints) -> Matrix {
        let s = self.scale_factor();
        Matrix::from_fn(a.n, b.n, |i, j| {
            s * self.base_value(Self::sq_dist(a.row(i), b.row(j)))
        })
    }

    /// Symmetric covariance on pre-scaled points.
    pub(crate) fn symmetric_scaled(&self, a: &ScaledPoints) -> Matrix {
        let s = self.scale_factor();
        let n = a.n;
        let mut k = Matrix::zeros(n, n);
        let diag = s * self.base_value(0.0);
        for i in 0..n {
            k[(i, i)] = diag;
            for j in (i + 1)..n {
                let v = s * self.base_value(Self::sq_dist(a.row(i), a.row(j)));
                k[(i, j)] = v;
                k[(j, i)] = v;
            }
        }
        k
    }

    /// Covariance matrix `K(X1, X2)` (`n1 × n2`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the column counts do not match `dim()`.
    pub fn covariance(&self, x1: &Matrix, x2: &Matrix) -> Result<Matrix> {
        self.check_dim(x1, "x1")?;
        self.check_dim(x2, "x2")?;
        Ok(self.cross_scaled(&self.scale_points(x1), &self.scale_points(x2)))
    }

    /// Symmetric covariance matrix `K(X, X)` (`n × n`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the column count does not match `dim()`.
    pub fn covariance_symmetric(&self, x: &Matrix) -> Result<Matrix> {
        self.check_dim(x, "x")?;
        Ok(self.symmetric_scaled(&self.scale_points(x)))
    }

    /// Diagonal of `K(X, X)` (the prior variance, constant for stationary kernels).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the column count does not match `dim()`.
    pub fn diag(&self, x: &Matrix) -> Result<Vector> {
        self.check_dim(x, "x")?;
        Ok(Vector::from_element(x.nrows(), self.prior_variance()))
    }

    /// The prior variance `k(x, x)` (the diagonal of the kernel matrix, which
    /// BoTorch's posterior variances use for up to
    /// `gpytorch.settings.max_eager_kernel_size` points).
    pub fn prior_variance(&self) -> f64 {
        self.scale_factor() * self.base_value(0.0)
    }

    /// `K(X, X)` and its derivatives with respect to every lengthscale and the
    /// outputscale (in actual, not raw, parameter space).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the column count does not match `dim()`.
    pub fn hyperparameter_gradients(&self, x: &Matrix) -> Result<KernelGradients> {
        self.check_dim(x, "x")?;
        let n = x.nrows();
        let d = self.dim();
        let s = self.scale_factor();
        let xs = self.scale_points(x);
        let mut cov = Matrix::zeros(n, n);
        let mut dl = vec![Matrix::zeros(n, n); d];
        let mut ds = self.outputscale.map(|_| Matrix::zeros(n, n));
        let k0 = self.base_value(0.0);
        for i in 0..n {
            cov[(i, i)] = s * k0;
            if let Some(ds) = ds.as_mut() {
                ds[(i, i)] = k0;
            }
            for j in (i + 1)..n {
                let (a, b) = (xs.row(i), xs.row(j));
                let (k, g) = self.base_value_and_factor(Self::sq_dist(a, b));
                cov[(i, j)] = s * k;
                cov[(j, i)] = s * k;
                if let Some(ds) = ds.as_mut() {
                    ds[(i, j)] = k;
                    ds[(j, i)] = k;
                }
                for (m, dlm) in dl.iter_mut().enumerate() {
                    let u = a[m] - b[m];
                    let v = s * g * u * u / self.lengthscale[m];
                    dlm[(i, j)] = v;
                    dlm[(j, i)] = v;
                }
            }
        }
        Ok(KernelGradients {
            covariance: cov,
            lengthscale: dl,
            outputscale: ds,
        })
    }

    /// Contracts the hyperparameter derivatives with a symmetric weight matrix:
    /// returns `(Σᵢⱼ Wᵢⱼ ∂Kᵢⱼ/∂ℓₘ)ₘ` and `Σᵢⱼ Wᵢⱼ ∂Kᵢⱼ/∂s` (if scaled) without
    /// materializing the derivative matrices.
    pub(crate) fn contract_hyper_grad(
        &self,
        xs: &ScaledPoints,
        w: &Matrix,
    ) -> (Vector, Option<f64>) {
        let n = xs.n;
        let d = self.dim();
        let s = self.scale_factor();
        let mut gl = vec![0.0; d];
        let mut gs = 0.0;
        let k0 = self.base_value(0.0);
        for i in 0..n {
            gs += w[(i, i)] * k0;
            let a = xs.row(i);
            for j in (i + 1)..n {
                let b = xs.row(j);
                let (k, g) = self.base_value_and_factor(Self::sq_dist(a, b));
                let wij = w[(i, j)] + w[(j, i)];
                gs += wij * k;
                let c = wij * g;
                for m in 0..d {
                    let u = a[m] - b[m];
                    gl[m] += c * u * u;
                }
            }
        }
        let gl = Vector::from_iterator(
            d,
            gl.iter()
                .zip(self.lengthscale.iter())
                .map(|(g, l)| s * g / l),
        );
        (gl, self.outputscale.map(|_| gs))
    }

    /// Covariances `k(x, X2)` of a single point with a point set and their
    /// derivatives with respect to `x`, both on pre-scaled inputs. Row `i` of
    /// the returned `n2 × d` matrix holds `∂k(x, x2ᵢ)/∂x` (derivative with
    /// respect to the *unscaled* coordinates of `x`).
    pub(crate) fn cross_with_grad_scaled(&self, x: &[f64], xs: &ScaledPoints) -> (Vector, Matrix) {
        let d = self.dim();
        let s = self.scale_factor();
        let mut k = Vector::zeros(xs.n);
        let mut dk = Matrix::zeros(xs.n, d);
        for i in 0..xs.n {
            let b = xs.row(i);
            let (kv, g) = self.base_value_and_factor(Self::sq_dist(x, b));
            k[i] = s * kv;
            for m in 0..d {
                dk[(i, m)] = -s * g * (x[m] - b[m]) / self.lengthscale[m];
            }
        }
        (k, dk)
    }

    /// Covariances `k(x, X2)` (length `n2`) of a single point `x` with the rows
    /// of `x2`, and the `n2 × d` matrix of derivatives `∂k(x, x2ᵢ)/∂x`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] on inconsistent dimensions.
    pub fn covariance_with_grad_x(&self, x: &[f64], x2: &Matrix) -> Result<(Vector, Matrix)> {
        if x.len() != self.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "x has length {} but the kernel has {} lengthscales",
                x.len(),
                self.dim()
            )));
        }
        self.check_dim(x2, "x2")?;
        let xs = self.scale_point(x);
        Ok(self.cross_with_grad_scaled(&xs, &self.scale_points(x2)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points() -> Matrix {
        Matrix::from_row_slice(4, 2, &[0.1, 0.2, 0.5, 0.9, 0.3, 0.3, 0.95, 0.05])
    }

    #[test]
    fn rbf_matches_closed_form() {
        let k = Kernel::new(KernelKind::Rbf, Vector::from_vec(vec![0.3, 0.7]), None).unwrap();
        let x = points();
        let km = k.covariance_symmetric(&x).unwrap();
        let (a, b) = (x.row(0), x.row(1));
        let r2 = ((a[0] - b[0]) / 0.3).powi(2) + ((a[1] - b[1]) / 0.7).powi(2);
        assert!((km[(0, 1)] - (-0.5 * r2).exp()).abs() < 1e-15);
        assert_eq!(km[(2, 2)], 1.0);
        let cross = k.covariance(&x, &x).unwrap();
        assert!((cross - &km).norm() < 1e-15);
    }

    #[test]
    fn matern_matches_closed_form() {
        let k = Kernel::new(
            KernelKind::Matern52,
            Vector::from_vec(vec![0.3, 0.7]),
            Some(1.7),
        )
        .unwrap();
        let x = points();
        let km = k.covariance_symmetric(&x).unwrap();
        let (a, b) = (x.row(1), x.row(3));
        let r = (((a[0] - b[0]) / 0.3).powi(2) + ((a[1] - b[1]) / 0.7).powi(2)).sqrt();
        let expected =
            1.7 * (1.0 + 5f64.sqrt() * r + 5.0 / 3.0 * r * r) * (-(5f64.sqrt()) * r).exp();
        assert!((km[(1, 3)] - expected).abs() < 1e-15);
        assert!((km[(0, 0)] - 1.7).abs() < 1e-15);
    }

    #[test]
    fn hyper_gradients_match_finite_differences() {
        let x = points();
        for kind in [KernelKind::Rbf, KernelKind::Matern52] {
            let ls = vec![0.4, 0.25];
            let k = Kernel::new(kind, Vector::from_vec(ls.clone()), Some(1.3)).unwrap();
            let grads = k.hyperparameter_gradients(&x).unwrap();
            for m in 0..2 {
                let h = 1e-6;
                let mut lp = ls.clone();
                lp[m] += h;
                let mut lm = ls.clone();
                lm[m] -= h;
                let kp = Kernel::new(kind, Vector::from_vec(lp), Some(1.3))
                    .unwrap()
                    .covariance_symmetric(&x)
                    .unwrap();
                let km = Kernel::new(kind, Vector::from_vec(lm), Some(1.3))
                    .unwrap()
                    .covariance_symmetric(&x)
                    .unwrap();
                let fd = (kp - km) / (2.0 * h);
                assert!((fd - &grads.lengthscale[m]).amax() < 1e-7);
            }
            let ds = grads.outputscale.unwrap();
            assert!((ds * 1.3 - &grads.covariance).amax() < 1e-14);
        }
    }

    #[test]
    fn x_gradients_match_finite_differences() {
        let x2 = points();
        for kind in [KernelKind::Rbf, KernelKind::Matern52] {
            let k = Kernel::new(kind, Vector::from_vec(vec![0.4, 0.25]), Some(0.8)).unwrap();
            let x = [0.42, 0.61];
            let (_, dk) = k.covariance_with_grad_x(&x, &x2).unwrap();
            for m in 0..2 {
                let h = 1e-6;
                let mut xp = x;
                xp[m] += h;
                let mut xm = x;
                xm[m] -= h;
                let kp = k.covariance_with_grad_x(&xp, &x2).unwrap().0;
                let km = k.covariance_with_grad_x(&xm, &x2).unwrap().0;
                let fd = (kp - km) / (2.0 * h);
                assert!((fd - dk.column(m)).amax() < 1e-8);
            }
        }
    }

    #[test]
    fn dim_scaled_preset_matches_botorch() {
        // Values recorded from BoTorch (float32 prior parameters and initial values).
        let spec = get_covar_module_with_dim_scaled_prior(5, true);
        match spec.lengthscale_prior.unwrap() {
            Prior::LogNormal { loc, scale, .. } => {
                assert_eq!(loc, 2.218932628631592);
                assert_eq!(scale, 1.7320507764816284);
            }
            _ => panic!("expected LogNormal"),
        }
        assert_eq!(
            spec.lengthscale_constraint.lower_bound(),
            0.02500000037252903
        );
        assert_eq!(spec.initial_raw_lengthscale(), 0.4579169750213623);
        assert_eq!(
            get_covar_module_with_dim_scaled_prior(1, true).initial_raw_lengthscale(),
            0.20478667318820953
        );
        assert_eq!(
            get_covar_module_with_dim_scaled_prior(2, true).initial_raw_lengthscale(),
            0.2896120846271515
        );
    }
}
