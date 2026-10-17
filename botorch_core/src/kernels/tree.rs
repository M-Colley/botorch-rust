//! Covariance modules as trees of kernels (GPyTorch's and BoTorch's kernel
//! classes): the structure, the parameters with their names, shapes,
//! constraints, priors and initial values, and the input dimensions
//! (`active_dims`).
//!
//! A [`KernelNode`] mirrors a GPyTorch kernel object: its own parameters in
//! GPyTorch's registration order, followed by those of its sub-kernels
//! (`ScaleKernel.base_kernel`, `AdditiveKernel.kernels.{i}`, ...), which is
//! the order of `named_parameters()`. Parameters are described per model
//! element: a model with a batch shape (BoTorch's `_aug_batch_shape`) has one
//! value set per element, and the shapes here exclude those batch
//! dimensions (e.g. `raw_lengthscale` has shape `[1, ard_num_dims]`).
//!
//! Evaluation with concrete hyperparameter values is in
//! [`crate::kernels::eval`].
//!
//! Like GPyTorch, a `ScaleKernel` takes over the `active_dims` of its base
//! kernel (it calls the base kernel's `forward` directly, so the base
//! kernel's own `active_dims` are not applied a second time);
//! [`KernelNode::scale`] moves them up.

use crate::constraints::{Constraint, GreaterThan};
use crate::error::{BotorchError, Result};
use crate::kernels::KernelKind;
use crate::priors::Prior;

/// One parameter of a kernel (per model element).
#[derive(Clone, Debug, PartialEq)]
pub struct ParamSpec {
    /// GPyTorch's parameter name, e.g. `"raw_lengthscale"` or
    /// `"covar_factor"`.
    pub name: String,
    /// The shape without batch dimensions, e.g. `[1, d]` for ARD lengthscales.
    pub shape: Vec<usize>,
    /// The constraint (`None` for unconstrained parameters such as
    /// `IndexKernel.covar_factor`: the actual value is the raw value).
    pub constraint: Option<Constraint>,
    /// Whether the constraint is registered with GPyTorch's
    /// `register_constraint` (then it gives the optimizer bounds and its
    /// initial value). `OrthogonalAdditiveKernel`'s `coeff_constraint` only
    /// transforms its coefficients.
    pub registered: bool,
    /// A univariate prior on every entry of the actual (constrained) value.
    pub prior: Option<Prior>,
    /// The raw initial value of every entry unless a registered constraint
    /// has an initial value (GPyTorch initializes most parameters to zero).
    pub default_raw: f64,
}

impl ParamSpec {
    /// A parameter with a registered constraint, no prior and raw initial
    /// value 0.
    pub fn new(name: &str, shape: Vec<usize>, constraint: Option<Constraint>) -> Self {
        Self {
            name: name.to_string(),
            shape,
            constraint,
            registered: true,
            prior: None,
            default_raw: 0.0,
        }
    }

    /// The number of values per model element.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The raw initial value (the registered constraint's initial value if
    /// it has one, as GPyTorch's `register_constraint` applies it).
    pub fn initial_raw(&self) -> f64 {
        match (&self.constraint, self.registered) {
            (Some(c), true) => c.initial_raw_value().unwrap_or(self.default_raw),
            _ => self.default_raw,
        }
    }

    /// The actual value of a raw value.
    #[inline]
    pub fn transform(&self, raw: f64) -> f64 {
        match &self.constraint {
            Some(c) => c.transform(raw),
            None => raw,
        }
    }

    /// `d actual / d raw`.
    #[inline]
    pub fn transform_derivative(&self, raw: f64) -> f64 {
        match &self.constraint {
            Some(c) => c.transform_derivative(raw),
            None => 1.0,
        }
    }

    /// The raw value of an actual value.
    #[inline]
    pub fn inverse_transform(&self, value: f64) -> f64 {
        match &self.constraint {
            Some(c) => c.inverse_transform(value),
            None => value,
        }
    }

    /// The optimizer bounds of every raw entry
    /// (`botorch.optim.utils.get_parameters_and_bounds`; unbounded without
    /// a registered constraint).
    pub fn raw_bounds(&self) -> (f64, f64) {
        match (&self.constraint, self.registered) {
            (Some(c), true) => c.raw_bounds(),
            _ => (f64::NEG_INFINITY, f64::INFINITY),
        }
    }
}

/// A prior on a matrix-valued function of a kernel's parameters: the task
/// covariance matrix of an `IndexKernel` (GPyTorch's `IndexKernelPrior`), or
/// the lower-triangle correlations of a `PositiveIndexKernel`.
#[derive(Clone, Debug, PartialEq)]
pub enum MatrixPrior {
    /// A univariate prior on every entry.
    Elementwise(Prior),
    /// GPyTorch's `LKJCovariancePrior(n, eta, sd_prior)`: an LKJ prior with
    /// concentration `eta` on the correlation matrix and `sd_prior` on every
    /// marginal standard deviation of an `n × n` covariance matrix.
    LkjCovariance {
        /// The matrix size.
        n: usize,
        /// The LKJ concentration (> 0).
        eta: f64,
        /// The prior of the marginal standard deviations.
        sd_prior: Prior,
    },
}

/// BoTorch's SAAS prior (`botorch.models.map_saas.add_saas_prior`) on a
/// kernel with lengthscales: a prior (BoTorch: `HalfCauchyPrior(1.0)`) on
/// `tau / ℓⱼ²` for every lengthscale `ℓⱼ`, with the global shrinkage `tau`
/// either fixed or learned as the kernel's parameter `raw_tau` (which has
/// its own prior, BoTorch: `HalfCauchyPrior(0.1)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaasPrior {
    /// The prior on every `tau / ℓⱼ²`.
    pub prior: Prior,
    /// The fixed global shrinkage, or `None` if it is the parameter
    /// `raw_tau`.
    pub tau: Option<f64>,
}

/// BoTorch's clamping margin (`botorch.models.map_saas.EPS`) of the
/// lengthscales and `tau` set from prior samples.
pub const SAAS_EPS: f64 = 1e-8;

/// The kernel classes of a [`KernelNode`].
#[derive(Clone, Debug, PartialEq)]
pub enum KernelType {
    /// `RBFKernel` or `MaternKernel(nu)` (parameter `raw_lengthscale`).
    Stationary(KernelKind),
    /// `RQKernel`: `(1 + r²/(2α))^(-α)` (`raw_lengthscale`, `raw_alpha`).
    RationalQuadratic,
    /// `PeriodicKernel`: `exp(-2 Σⱼ sin²(π |xⱼ - yⱼ| / pⱼ) / ℓⱼ)`
    /// (`raw_lengthscale`, `raw_period_length`).
    Periodic,
    /// `LinearKernel`: `Σⱼ vⱼ xⱼ yⱼ` (`raw_variance`).
    Linear,
    /// `PolynomialKernel`: `(xᵀy + c)^power` (`raw_offset`).
    Polynomial {
        /// The power.
        power: f64,
    },
    /// `CosineKernel`: `cos(π ‖x - y‖ / p)` (`raw_period_length`).
    Cosine,
    /// `IndexKernel`: `B[i, j]` for task indices `i, j` with
    /// `B = F Fᵀ + diag(v)` (`covar_factor`, `raw_var`).
    Index {
        /// The number of tasks.
        num_tasks: usize,
        /// The rank of `F`.
        rank: usize,
    },
    /// BoTorch's `PositiveIndexKernel`: like `IndexKernel` with a positive
    /// factor and the task covariance normalized by its target diagonal
    /// entry (`raw_var`, `raw_covar_factor`).
    PositiveIndex {
        /// The number of tasks.
        num_tasks: usize,
        /// The rank of the factor.
        rank: usize,
        /// The task whose variance is normalized to 1.
        target_task_index: usize,
        /// Whether to normalize by the target task's variance.
        unit_scale_for_target: bool,
    },
    /// BoTorch's `CategoricalKernel`: `exp(-mean_j([xⱼ ≠ yⱼ] / ℓⱼ))`
    /// (`raw_lengthscale`).
    Categorical,
    /// BoTorch's `DownsamplingKernel`: `c + Σⱼ (1 - xⱼ)^(1+δ) (1 - yⱼ)^(1+δ)`
    /// (`raw_power`, `raw_offset`).
    Downsampling,
    /// BoTorch's `ExponentialDecayKernel`: `w + (1 + ‖(x + y)/ℓ‖)^(-α)`
    /// (`raw_lengthscale`, `raw_power`, `raw_offset`).
    ExponentialDecay,
    /// BoTorch's `InfiniteWidthBNNKernel` (`raw_weight_var`, `raw_bias_var`).
    InfiniteWidthBnn {
        /// The depth of the network.
        depth: usize,
        /// The clamping margin of the arc-cosine argument.
        acos_eps: f64,
    },
    /// BoTorch's `LinearTruncatedFidelityKernel` with sub-kernels
    /// `covar_module_unbiased` and `covar_module_biased` on the
    /// non-fidelity inputs (`raw_power`).
    LinearTruncatedFidelity {
        /// The one or two fidelity dimensions (of the kernel's inputs).
        fidelity_dims: Vec<usize>,
    },
    /// BoTorch's `OrthogonalAdditiveKernel` with a stationary one-dimensional
    /// base kernel (`raw_offset`, `raw_coeffs_1`, `raw_coeffs_2`).
    OrthogonalAdditive {
        /// The input dimension.
        dim: usize,
        /// The number of Gauss-Legendre nodes.
        quad_deg: usize,
        /// Whether second-order interactions are modeled.
        second_order: bool,
        /// Whether inputs outside `[0, 1]^d` are rejected.
        check_hypercube: bool,
        /// Whether the base kernel has one lengthscale per input dimension
        /// (`batch_shape = [dim]`) instead of a shared one.
        per_dim_lengthscales: bool,
    },
    /// `ScaleKernel` (`raw_outputscale`, sub-kernel `base_kernel`).
    Scale,
    /// `AdditiveKernel` (sub-kernels `kernels.{i}`).
    Additive,
    /// `ProductKernel` (sub-kernels `kernels.{i}`).
    Product,
}

/// A covariance module: a kernel with its parameters, optional
/// `active_dims` and sub-kernels (see the [module documentation](self)).
#[derive(Clone, Debug, PartialEq)]
pub struct KernelNode {
    kind: KernelType,
    active_dims: Option<Vec<usize>>,
    params: Vec<ParamSpec>,
    task_prior: Option<MatrixPrior>,
    saas: Option<SaasPrior>,
    children: Vec<KernelNode>,
}

fn positive() -> Option<Constraint> {
    Some(Constraint::positive())
}

fn lengthscale(ard_num_dims: Option<usize>) -> ParamSpec {
    ParamSpec::new(
        "raw_lengthscale",
        vec![1, ard_num_dims.unwrap_or(1)],
        positive(),
    )
}

fn check_ard(ard_num_dims: Option<usize>) -> Result<()> {
    if ard_num_dims == Some(0) {
        return Err(BotorchError::InvalidArgument(
            "ard_num_dims must be positive".into(),
        ));
    }
    Ok(())
}

impl KernelNode {
    fn leaf(kind: KernelType, params: Vec<ParamSpec>) -> Self {
        Self {
            kind,
            active_dims: None,
            params,
            task_prior: None,
            saas: None,
            children: Vec::new(),
        }
    }

    /// `RBFKernel(ard_num_dims)` / `MaternKernel(nu, ard_num_dims)`: one
    /// lengthscale per input dimension, or one shared lengthscale if
    /// `ard_num_dims` is `None`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `ard_num_dims = Some(0)`.
    pub fn stationary(kind: KernelKind, ard_num_dims: Option<usize>) -> Result<Self> {
        check_ard(ard_num_dims)?;
        Ok(Self::leaf(
            KernelType::Stationary(kind),
            vec![lengthscale(ard_num_dims)],
        ))
    }

    /// `RQKernel(ard_num_dims)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `ard_num_dims = Some(0)`.
    pub fn rational_quadratic(ard_num_dims: Option<usize>) -> Result<Self> {
        check_ard(ard_num_dims)?;
        Ok(Self::leaf(
            KernelType::RationalQuadratic,
            vec![
                lengthscale(ard_num_dims),
                ParamSpec::new("raw_alpha", vec![1], positive()),
            ],
        ))
    }

    /// `PeriodicKernel(ard_num_dims)`: lengthscales and period lengths of
    /// shape `[1, ard_num_dims]` (`[1, 1]`, shared, without).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `ard_num_dims = Some(0)`.
    pub fn periodic(ard_num_dims: Option<usize>) -> Result<Self> {
        check_ard(ard_num_dims)?;
        Ok(Self::leaf(
            KernelType::Periodic,
            vec![
                lengthscale(ard_num_dims),
                ParamSpec::new(
                    "raw_period_length",
                    vec![1, ard_num_dims.unwrap_or(1)],
                    positive(),
                ),
            ],
        ))
    }

    /// `LinearKernel(ard_num_dims)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `ard_num_dims = Some(0)`.
    pub fn linear(ard_num_dims: Option<usize>) -> Result<Self> {
        check_ard(ard_num_dims)?;
        Ok(Self::leaf(
            KernelType::Linear,
            vec![ParamSpec::new(
                "raw_variance",
                vec![1, ard_num_dims.unwrap_or(1)],
                positive(),
            )],
        ))
    }

    /// `PolynomialKernel(power)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for a non-finite power.
    pub fn polynomial(power: f64) -> Result<Self> {
        if !power.is_finite() {
            return Err(BotorchError::InvalidArgument(format!(
                "the power of a PolynomialKernel must be finite, got {power}"
            )));
        }
        Ok(Self::leaf(
            KernelType::Polynomial { power },
            vec![ParamSpec::new("raw_offset", vec![1], positive())],
        ))
    }

    /// `CosineKernel()`.
    pub fn cosine() -> Self {
        Self::leaf(
            KernelType::Cosine,
            vec![ParamSpec::new("raw_period_length", vec![1, 1], positive())],
        )
    }

    /// `IndexKernel(num_tasks, rank)` (initialized to zero here; GPyTorch
    /// draws `covar_factor` and `raw_var` from a standard normal).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] with GPyTorch's message if
    /// `rank > num_tasks`, or for zero tasks or rank.
    pub fn index(num_tasks: usize, rank: usize) -> Result<Self> {
        check_tasks(num_tasks, rank)?;
        Ok(Self::leaf(
            KernelType::Index { num_tasks, rank },
            vec![
                ParamSpec::new("covar_factor", vec![num_tasks, rank], None),
                ParamSpec::new("raw_var", vec![num_tasks], positive()),
            ],
        ))
    }

    /// BoTorch's `PositiveIndexKernel(num_tasks, rank, target_task_index,
    /// unit_scale_for_target)` (initialized to zero here).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] with BoTorch's messages for
    /// `rank > num_tasks` or an invalid target task.
    pub fn positive_index(
        num_tasks: usize,
        rank: usize,
        target_task_index: usize,
        unit_scale_for_target: bool,
    ) -> Result<Self> {
        check_tasks(num_tasks, rank)?;
        if target_task_index >= num_tasks {
            return Err(BotorchError::InvalidArgument(format!(
                "target_task_index must be between 0 and {}, got {target_task_index}",
                num_tasks - 1
            )));
        }
        Ok(Self::leaf(
            KernelType::PositiveIndex {
                num_tasks,
                rank,
                target_task_index,
                unit_scale_for_target,
            },
            vec![
                ParamSpec::new("raw_var", vec![num_tasks], positive()),
                ParamSpec::new(
                    "raw_covar_factor",
                    vec![num_tasks, rank],
                    Some(
                        GreaterThan::new(0.0, Some(crate::ParameterTransform::Softplus), None)
                            .into(),
                    ),
                ),
            ],
        ))
    }

    /// BoTorch's `CategoricalKernel(ard_num_dims)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `ard_num_dims = Some(0)`.
    pub fn categorical(ard_num_dims: Option<usize>) -> Result<Self> {
        check_ard(ard_num_dims)?;
        Ok(Self::leaf(
            KernelType::Categorical,
            vec![lengthscale(ard_num_dims)],
        ))
    }

    /// BoTorch's `DownsamplingKernel()`.
    pub fn downsampling() -> Self {
        Self::leaf(
            KernelType::Downsampling,
            vec![
                ParamSpec::new("raw_power", vec![1], positive()),
                ParamSpec::new("raw_offset", vec![1], positive()),
            ],
        )
    }

    /// BoTorch's `ExponentialDecayKernel(ard_num_dims)`.
    ///
    /// BoTorch registers the *offset* constraint for `raw_power` (its
    /// `power_constraint` argument is ignored); here `raw_power` has its own
    /// constraint (set it with [`KernelNode::with_constraint`]).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `ard_num_dims = Some(0)`.
    pub fn exponential_decay(ard_num_dims: Option<usize>) -> Result<Self> {
        check_ard(ard_num_dims)?;
        Ok(Self::leaf(
            KernelType::ExponentialDecay,
            vec![
                lengthscale(ard_num_dims),
                ParamSpec::new("raw_power", vec![1], positive()),
                ParamSpec::new("raw_offset", vec![1], positive()),
            ],
        ))
    }

    /// BoTorch's `InfiniteWidthBNNKernel(depth, acos_eps)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an `acos_eps` outside `[0, 1)`.
    pub fn infinite_width_bnn(depth: usize, acos_eps: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&acos_eps) {
            return Err(BotorchError::InvalidArgument(format!(
                "acos_eps must be in [0, 1), got {acos_eps}"
            )));
        }
        Ok(Self::leaf(
            KernelType::InfiniteWidthBnn { depth, acos_eps },
            vec![
                ParamSpec::new("raw_weight_var", vec![1, 1], positive()),
                ParamSpec::new("raw_bias_var", vec![1, 1], positive()),
            ],
        ))
    }

    /// BoTorch's `LinearTruncatedFidelityKernel(fidelity_dims, ...)` with
    /// the given sub-kernels on the non-fidelity inputs (BoTorch's defaults
    /// are Matérn kernels with `GammaPrior(3, 6)` and `GammaPrior(6, 2)`
    /// lengthscale priors).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] with BoTorch's messages for repeated
    /// fidelity dimensions and [`BotorchError::Unsupported`] for other than
    /// one or two of them.
    pub fn linear_truncated_fidelity(
        fidelity_dims: Vec<usize>,
        unbiased: KernelNode,
        biased: KernelNode,
    ) -> Result<Self> {
        let n = fidelity_dims.len();
        let mut sorted = fidelity_dims.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != n {
            return Err(BotorchError::InvalidArgument(
                "fidelity_dims must not have repeated elements".into(),
            ));
        }
        if !(1..=2).contains(&n) {
            return Err(BotorchError::Unsupported(
                "LinearTruncatedFidelityKernel accepts either one or twofidelity parameters."
                    .into(),
            ));
        }
        Ok(Self {
            kind: KernelType::LinearTruncatedFidelity { fidelity_dims },
            active_dims: None,
            params: vec![ParamSpec::new("raw_power", vec![1], positive())],
            task_prior: None,
            saas: None,
            children: vec![unbiased, biased],
        })
    }

    /// BoTorch's `OrthogonalAdditiveKernel(dim, base_kernel, ...)` with a
    /// stationary base kernel `base` (an RBF or Matérn kernel without ARD and
    /// `active_dims`, e.g. `KernelNode::stationary(KernelKind::Rbf, None)`,
    /// possibly with a lengthscale prior and constraint) that has one
    /// lengthscale per input dimension if `per_dim_lengthscales` (BoTorch's
    /// `batch_shape = [dim]`), else a shared one, and the coefficient
    /// constraint `coeff_constraint` (GPyTorch's `Positive()` by default; it
    /// transforms the coefficients but is not registered, so it gives no
    /// optimizer bounds and no initial values).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `dim = 0` or `quad_deg = 0` and
    /// [`BotorchError::Unsupported`] for other base kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn orthogonal_additive(
        dim: usize,
        base: KernelNode,
        per_dim_lengthscales: bool,
        quad_deg: usize,
        second_order: bool,
        check_hypercube: bool,
        coeff_constraint: Constraint,
    ) -> Result<Self> {
        if dim == 0 || quad_deg == 0 {
            return Err(BotorchError::InvalidArgument(
                "OrthogonalAdditiveKernel needs dim > 0 and quad_deg > 0".into(),
            ));
        }
        let log_d = (dim as f64).ln();
        let coeff = |name: &str, shape: Vec<usize>, init: f64| ParamSpec {
            name: name.to_string(),
            shape,
            constraint: Some(coeff_constraint),
            registered: false,
            prior: None,
            default_raw: init,
        };
        let mut params = vec![
            coeff("raw_offset", vec![], 0.0),
            coeff("raw_coeffs_1", vec![dim], -log_d),
        ];
        if second_order {
            params.push(coeff(
                "raw_coeffs_2",
                vec![dim * (dim - 1) / 2],
                -2.0 * log_d,
            ));
        }
        let mut base = base;
        if !matches!(base.kind, KernelType::Stationary(_))
            || base.active_dims.is_some()
            || base.params[0].numel() != 1
        {
            return Err(BotorchError::Unsupported(
                "the base kernel of an OrthogonalAdditiveKernel must be an RBFKernel or a                  MaternKernel with a single lengthscale and without active_dims"
                    .into(),
            ));
        }
        base.params[0].shape = if per_dim_lengthscales {
            vec![dim, 1, 1]
        } else {
            vec![1, 1]
        };
        Ok(Self {
            kind: KernelType::OrthogonalAdditive {
                dim,
                quad_deg,
                second_order,
                check_hypercube,
                per_dim_lengthscales,
            },
            active_dims: None,
            params,
            task_prior: None,
            saas: None,
            children: vec![base],
        })
    }

    /// `ScaleKernel(base)`, taking over the `active_dims` of `base`.
    pub fn scale(mut base: KernelNode) -> Self {
        let active_dims = base.active_dims.take();
        Self {
            kind: KernelType::Scale,
            active_dims,
            params: vec![ParamSpec::new("raw_outputscale", vec![], positive())],
            task_prior: None,
            saas: None,
            children: vec![base],
        }
    }

    /// `AdditiveKernel(*kernels)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] without kernels.
    pub fn additive(kernels: Vec<KernelNode>) -> Result<Self> {
        Self::composite(KernelType::Additive, kernels)
    }

    /// `ProductKernel(*kernels)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] without kernels.
    pub fn product(kernels: Vec<KernelNode>) -> Result<Self> {
        Self::composite(KernelType::Product, kernels)
    }

    fn composite(kind: KernelType, kernels: Vec<KernelNode>) -> Result<Self> {
        if kernels.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "AdditiveKernel and ProductKernel need at least one kernel".into(),
            ));
        }
        Ok(Self {
            kind,
            active_dims: None,
            params: Vec::new(),
            task_prior: None,
            saas: None,
            children: kernels,
        })
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    /// Sets `active_dims` (the input columns the kernel acts on). A
    /// `ScaleKernel` keeps the `active_dims` taken over from its base kernel.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for empty `active_dims` and
    /// [`BotorchError::Unsupported`] for `active_dims` of an
    /// `AdditiveKernel` / `ProductKernel` (GPyTorch has none) or a second set
    /// for a `ScaleKernel`.
    pub fn with_active_dims(mut self, active_dims: Vec<usize>) -> Result<Self> {
        if active_dims.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "active_dims must not be empty".into(),
            ));
        }
        match self.kind {
            KernelType::Additive | KernelType::Product => {
                return Err(BotorchError::Unsupported(
                    "AdditiveKernel and ProductKernel have no active_dims".into(),
                ))
            }
            KernelType::Scale if self.active_dims.is_some() => {
                // GPyTorch replaces them with the base kernel's.
                return Ok(self);
            }
            _ => {}
        }
        self.active_dims = Some(active_dims);
        Ok(self)
    }

    fn param_mut(&mut self, name: &str) -> Result<&mut ParamSpec> {
        let kind = format!("{:?}", self.kind);
        self.params
            .iter_mut()
            .find(|p| p.name == name)
            .ok_or_else(|| {
                BotorchError::InvalidArgument(format!("{kind} has no parameter {name:?}"))
            })
    }

    /// Sets the constraint of the parameter `name` (e.g. `"raw_lengthscale"`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an unknown or unconstrained
    /// parameter.
    pub fn with_constraint(mut self, name: &str, constraint: Constraint) -> Result<Self> {
        let p = self.param_mut(name)?;
        if p.constraint.is_none() {
            return Err(BotorchError::InvalidArgument(format!(
                "parameter {name:?} has no constraint"
            )));
        }
        p.constraint = Some(constraint);
        Ok(self)
    }

    /// Sets the prior on the actual value of the parameter `name`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an unknown parameter.
    pub fn with_prior(mut self, name: &str, prior: Prior) -> Result<Self> {
        self.param_mut(name)?.prior = Some(prior);
        Ok(self)
    }

    /// Sets the raw initial value of the parameter `name` (used unless a
    /// registered constraint has an initial value).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an unknown parameter or a
    /// non-finite value.
    pub fn with_default_raw(mut self, name: &str, value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(BotorchError::InvalidArgument(format!(
                "the initial value of {name:?} must be finite"
            )));
        }
        self.param_mut(name)?.default_raw = value;
        Ok(self)
    }

    /// Sets the prior on the task covariance (`IndexKernel`, GPyTorch's
    /// `IndexKernelPrior` on `F Fᵀ + diag(v)`) or on the lower-triangle task
    /// correlations (`PositiveIndexKernel`, elementwise priors only).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for other kernels and LKJ priors on a
    /// `PositiveIndexKernel`; [`BotorchError::InvalidArgument`] for an LKJ
    /// prior of the wrong size.
    pub fn with_task_prior(mut self, prior: MatrixPrior) -> Result<Self> {
        match (&self.kind, &prior) {
            (KernelType::Index { num_tasks, .. }, MatrixPrior::LkjCovariance { n, .. }) => {
                if n != num_tasks {
                    return Err(BotorchError::InvalidArgument(format!(
                        "an LKJ prior of size {n} cannot be a prior on {num_tasks} tasks"
                    )));
                }
            }
            (KernelType::Index { .. }, MatrixPrior::Elementwise(_)) => {}
            (KernelType::PositiveIndex { .. }, MatrixPrior::Elementwise(_)) => {}
            (KernelType::PositiveIndex { .. }, MatrixPrior::LkjCovariance { .. }) => {
                return Err(BotorchError::Unsupported(
                    "the prior of a PositiveIndexKernel acts on the vector of lower-triangle \
                     correlations; use a univariate prior such as BetaPrior"
                        .into(),
                ))
            }
            _ => {
                return Err(BotorchError::Unsupported(
                    "only IndexKernel and PositiveIndexKernel have a task prior".into(),
                ))
            }
        }
        self.task_prior = Some(prior);
        Ok(self)
    }

    /// Adds BoTorch's SAAS prior (`add_saas_prior`): `prior` on
    /// `tau / ℓⱼ²` for every lengthscale, with the fixed global shrinkage
    /// `tau`, or (`tau = None`) a learned `tau`: the parameter `raw_tau`
    /// (one value per model element, registered after the lengthscale) with
    /// the constraint and prior `tau_param`. The lengthscale constraint is
    /// set separately (BoTorch: `LogTransformedInterval(0.01, 1e4,
    /// initial_value=1)`).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] with BoTorch's messages for a kernel
    /// without lengthscales or with a lengthscale prior (botorch_rs supports
    /// the SAAS prior on `RBFKernel` and `MaternKernel`);
    /// [`BotorchError::InvalidArgument`] for a learned `tau` without
    /// `tau_param`, a second learned `tau`, or a non-finite fixed `tau`.
    pub fn with_saas_prior(
        mut self,
        prior: Prior,
        tau: Option<f64>,
        tau_param: Option<(Constraint, Prior)>,
    ) -> Result<Self> {
        let has_lengthscale = matches!(
            self.kind,
            KernelType::Stationary(_)
                | KernelType::RationalQuadratic
                | KernelType::Periodic
                | KernelType::Categorical
                | KernelType::ExponentialDecay
        );
        if !has_lengthscale {
            return Err(BotorchError::Unsupported(
                "base_kernel must have lengthscale(s)".into(),
            ));
        }
        if !matches!(self.kind, KernelType::Stationary(_)) {
            return Err(BotorchError::Unsupported(
                "botorch_rs supports the SAAS prior on RBFKernel and MaternKernel".into(),
            ));
        }
        if self.params[0].prior.is_some() {
            return Err(BotorchError::Unsupported(
                "base_kernel must not specify a lengthscale prior".into(),
            ));
        }
        match tau {
            Some(t) if !t.is_finite() => {
                return Err(BotorchError::InvalidArgument(format!(
                    "the global shrinkage tau must be finite, got {t}"
                )))
            }
            Some(_) => {}
            None => {
                if self.params.iter().any(|p| p.name == "raw_tau") {
                    return Err(BotorchError::InvalidArgument(
                        "attribute 'raw_tau' already exists".into(),
                    ));
                }
                let (constraint, tau_prior) = tau_param.ok_or_else(|| {
                    BotorchError::InvalidArgument(
                        "a learned tau needs its constraint and prior".into(),
                    )
                })?;
                let mut p = ParamSpec::new("raw_tau", vec![], Some(constraint));
                p.prior = Some(tau_prior);
                self.params.push(p);
            }
        }
        self.saas = Some(SaasPrior { prior, tau });
        Ok(self)
    }

    /// Sets the fixed `tau` of every SAAS prior with a fixed `tau` in the
    /// tree (pre-order), e.g. one value per kernel of an
    /// `AdditiveKernel` of SAAS kernels.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `taus` does not have one value
    /// per fixed `tau`, [`BotorchError::InvalidArgument`] for non-finite
    /// values.
    pub fn set_saas_taus(&mut self, taus: &[f64]) -> Result<()> {
        let mut i = 0;
        self.set_saas_taus_from(taus, &mut i)?;
        if i != taus.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {i} fixed SAAS taus, got {}",
                taus.len()
            )));
        }
        Ok(())
    }

    fn set_saas_taus_from(&mut self, taus: &[f64], i: &mut usize) -> Result<()> {
        if let Some(saas) = self.saas.as_mut() {
            if saas.tau.is_some() {
                let t = *taus.get(*i).ok_or_else(|| {
                    BotorchError::DimensionMismatch(format!(
                        "not enough fixed SAAS taus ({} given)",
                        taus.len()
                    ))
                })?;
                if !t.is_finite() {
                    return Err(BotorchError::InvalidArgument(format!(
                        "the global shrinkage tau must be finite, got {t}"
                    )));
                }
                saas.tau = Some(t);
                *i += 1;
            }
        }
        for child in &mut self.children {
            child.set_saas_taus_from(taus, i)?;
        }
        Ok(())
    }

    /// The fixed `tau`s of the SAAS priors in the tree (pre-order).
    pub fn saas_taus(&self) -> Vec<f64> {
        let mut out = Vec::new();
        if let Some(t) = self.saas.and_then(|s| s.tau) {
            out.push(t);
        }
        for child in &self.children {
            out.extend(child.saas_taus());
        }
        out
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// The kernel class.
    pub fn kind(&self) -> &KernelType {
        &self.kind
    }

    /// The `active_dims`, if any.
    pub fn active_dims(&self) -> Option<&[usize]> {
        self.active_dims.as_deref()
    }

    /// The node's own parameters in GPyTorch's order.
    pub fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    /// The task prior (`IndexKernel`, `PositiveIndexKernel`).
    pub fn task_prior(&self) -> Option<&MatrixPrior> {
        self.task_prior.as_ref()
    }

    /// The SAAS prior on the lengthscales, if any.
    pub fn saas(&self) -> Option<&SaasPrior> {
        self.saas.as_ref()
    }

    /// The sub-kernels.
    pub fn children(&self) -> &[KernelNode] {
        &self.children
    }

    /// The attribute names of the sub-kernels (`"base_kernel"`,
    /// `"kernels.0"`, ...).
    pub fn child_names(&self) -> Vec<String> {
        match self.kind {
            KernelType::Scale | KernelType::OrthogonalAdditive { .. } => {
                vec!["base_kernel".to_string()]
            }
            KernelType::LinearTruncatedFidelity { .. } => vec![
                "covar_module_unbiased".to_string(),
                "covar_module_biased".to_string(),
            ],
            _ => (0..self.children.len())
                .map(|i| format!("kernels.{i}"))
                .collect(),
        }
    }

    /// Visits every parameter in `named_parameters()` order with its full
    /// name (`prefix` + path).
    pub fn visit_params<'a, F: FnMut(String, &'a ParamSpec)>(&'a self, prefix: &str, f: &mut F) {
        for p in &self.params {
            f(format!("{prefix}{}", p.name), p);
        }
        for (child, name) in self.children.iter().zip(self.child_names()) {
            child.visit_params(&format!("{prefix}{name}."), f);
        }
    }

    /// All parameters in `named_parameters()` order.
    pub fn all_params(&self) -> Vec<&ParamSpec> {
        let mut out = Vec::new();
        self.visit_params("", &mut |_, p| out.push(p));
        out
    }

    /// The number of raw values per model element.
    pub fn num_raw(&self) -> usize {
        self.all_params().iter().map(|p| p.numel()).sum()
    }

    /// `(name, shape)` of every parameter with the given name prefix (e.g.
    /// `"covar_module."`).
    pub fn param_shapes(&self, prefix: &str) -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::new();
        self.visit_params(prefix, &mut |name, p| out.push((name, p.shape.clone())));
        out
    }

    /// The raw initial values of all parameters (flat).
    pub fn initial_raw(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.num_raw());
        for p in self.all_params() {
            out.extend(std::iter::repeat_n(p.initial_raw(), p.numel()));
        }
        out
    }

    /// The optimizer bounds of every raw value (flat).
    pub fn raw_bounds(&self) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(self.num_raw());
        for p in self.all_params() {
            out.extend(std::iter::repeat_n(p.raw_bounds(), p.numel()));
        }
        out
    }

    /// `d actual / d raw` of every raw value (flat).
    pub fn transform_derivatives(&self, raw: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(raw.len());
        let mut off = 0;
        for p in self.all_params() {
            for &r in &raw[off..off + p.numel()] {
                out.push(p.transform_derivative(r));
            }
            off += p.numel();
        }
        out
    }

    /// The actual values of every raw value (flat).
    pub fn transform_all(&self, raw: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(raw.len());
        let mut off = 0;
        for p in self.all_params() {
            for &r in &raw[off..off + p.numel()] {
                out.push(p.transform(r));
            }
            off += p.numel();
        }
        out
    }

    /// Whether the node is a `ScaleKernel` around a single stationary
    /// kernel or a stationary kernel, without `active_dims` and with one
    /// lengthscale per input dimension, i.e. expressible as a
    /// [`crate::KernelSpec`] for `input_dim` inputs.
    pub fn as_kernel_spec(&self, input_dim: usize) -> Option<crate::KernelSpec> {
        let stationary = |n: &KernelNode| -> Option<crate::KernelSpec> {
            if n.active_dims.is_some() || !n.children.is_empty() || n.saas.is_some() {
                return None;
            }
            let KernelType::Stationary(kind) = n.kind else {
                return None;
            };
            let p = &n.params[0];
            if p.shape != [1, input_dim] || !p.registered {
                return None;
            }
            Some(crate::KernelSpec {
                kind,
                ard_num_dims: input_dim,
                lengthscale_prior: p.prior,
                lengthscale_constraint: p.constraint?,
                scale: None,
            })
        };
        match self.kind {
            KernelType::Stationary(_) => stationary(self),
            KernelType::Scale if self.active_dims.is_none() => {
                let mut spec = stationary(&self.children[0])?;
                let p = &self.params[0];
                spec.scale = Some(crate::ScaleKernelSpec {
                    outputscale_prior: p.prior,
                    outputscale_constraint: p.constraint?,
                });
                Some(spec)
            }
            _ => None,
        }
    }

    /// Checks the tree against `input_dim` model inputs: `active_dims` in
    /// range, lengthscale (and other per-dimension parameter) sizes equal to
    /// the number of inputs of each kernel, one input for index kernels,
    /// valid fidelity dimensions and the dimension of an
    /// `OrthogonalAdditiveKernel`.
    ///
    /// # Errors
    /// [`BotorchError::Runtime`] (GPyTorch and torch raise `RuntimeError`)
    /// for inconsistent dimensions and out-of-range `active_dims`;
    /// [`BotorchError::InvalidArgument`] for out-of-range fidelity dimensions.
    pub fn validate(&self, input_dim: usize) -> Result<()> {
        let k = match &self.active_dims {
            Some(dims) => {
                if let Some(&bad) = dims.iter().find(|&&j| j >= input_dim) {
                    return Err(BotorchError::Runtime(format!(
                        "active_dims {dims:?} contain index {bad}, but the kernel's inputs have \
                         only {input_dim} dimensions (index out of range)"
                    )));
                }
                dims.len()
            }
            None => input_dim,
        };
        let check_ard = |p: &ParamSpec, what: &str| -> Result<()> {
            let size = *p.shape.last().unwrap_or(&1);
            if size != 1 && size != k {
                return Err(BotorchError::Runtime(format!(
                    "The kernel's {what} has {size} entries (ard_num_dims={size}) but its \
                     inputs have d={k} dimensions."
                )));
            }
            Ok(())
        };
        match &self.kind {
            KernelType::Stationary(_)
            | KernelType::RationalQuadratic
            | KernelType::Categorical
            | KernelType::ExponentialDecay => check_ard(&self.params[0], "lengthscale")?,
            KernelType::Periodic => {
                check_ard(&self.params[0], "lengthscale")?;
                check_ard(&self.params[1], "period_length")?;
            }
            KernelType::Linear => check_ard(&self.params[0], "variance")?,
            KernelType::Index { .. } | KernelType::PositiveIndex { .. } => {
                if k != 1 {
                    return Err(BotorchError::Runtime(format!(
                        "An index kernel needs exactly one input dimension (the task \
                         feature, e.g. active_dims=[t]), got {k}."
                    )));
                }
            }
            KernelType::LinearTruncatedFidelity { fidelity_dims } => {
                if let Some(&bad) = fidelity_dims.iter().find(|&&j| j >= k) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "fidelity dimension {bad} is out of range for {k} inputs (index out of \
                         range)"
                    )));
                }
                if k <= fidelity_dims.len() {
                    return Err(BotorchError::Runtime(
                        "Input to LinearTruncatedFidelityKernel must have at least one \
                         non-fidelity dimension."
                            .into(),
                    ));
                }
                for child in &self.children {
                    child.validate(k - fidelity_dims.len())?;
                }
                return Ok(());
            }
            KernelType::OrthogonalAdditive { dim, .. } => {
                if *dim != k {
                    return Err(BotorchError::Runtime(format!(
                        "The OrthogonalAdditiveKernel has dim={dim} but its inputs have \
                         d={k} dimensions."
                    )));
                }
                return Ok(());
            }
            _ => {}
        }
        for child in &self.children {
            child.validate(k)?;
        }
        Ok(())
    }

    /// Whether the node or one of its sub-kernels has a prior.
    pub fn has_priors(&self) -> bool {
        self.task_prior.is_some()
            || self.saas.is_some()
            || self.params.iter().any(|p| p.prior.is_some())
            || self.children.iter().any(KernelNode::has_priors)
    }
}

/// BoTorch's `sample_all_priors` for a kernel tree: every entry of a
/// parameter with a univariate prior is drawn from the prior (its raw value
/// set through the constraint's inverse), and the factor of a
/// `PositiveIndexKernel` with a task prior is set from correlations drawn
/// from the prior (BoTorch's `_set_lower_triangle_corr`: the correlation
/// matrix projected to the nearest positive-definite one, then the
/// Cholesky factor's first `rank` columns clamped at `1e-6`). An
/// `IndexKernel` task prior has no setting closure in GPyTorch; its
/// parameters are kept.
pub fn sample_tree_priors<R: rand::Rng + ?Sized>(node: &KernelNode, raw: &mut [f64], rng: &mut R) {
    let mut off = 0;
    sample_node(node, raw, &mut off, rng);
}

fn sample_node<R: rand::Rng + ?Sized>(
    node: &KernelNode,
    raw: &mut [f64],
    off: &mut usize,
    rng: &mut R,
) {
    let start = *off;
    for p in &node.params {
        if let Some(prior) = p.prior {
            let saas_tau = node.saas.is_some() && p.name == "raw_tau";
            for v in &mut raw[*off..*off + p.numel()] {
                let mut value = prior.sample(rng);
                if saas_tau {
                    // BoTorch's `tau_prior_setting_closure`.
                    if let Some(c) = &p.constraint {
                        value = value.clamp(c.lower_bound() + SAAS_EPS, c.upper_bound() - SAAS_EPS);
                    }
                }
                *v = p.inverse_transform(value);
            }
        }
        *off += p.numel();
    }
    if let Some(saas) = &node.saas {
        // BoTorch's `inv_lengthscale_prior_setting_closure`: lengthscales
        // `sqrt(tau / v)` for samples `v` of the prior, clamped to the
        // lengthscale constraint (after `tau` was set).
        let ls = &node.params[0];
        let tau = saas.tau.unwrap_or_else(|| {
            let mut t_off = start;
            let mut t = f64::NAN;
            for p in &node.params {
                if p.name == "raw_tau" {
                    t = p.transform(raw[t_off]);
                }
                t_off += p.numel();
            }
            t
        });
        let (lb, ub) = ls
            .constraint
            .as_ref()
            .map_or((0.0, f64::INFINITY), |c| (c.lower_bound(), c.upper_bound()));
        for v in &mut raw[start..start + ls.numel()] {
            let sample = saas.prior.sample(rng);
            let l = (tau / sample).sqrt().clamp(lb + SAAS_EPS, ub - SAAS_EPS);
            *v = ls.inverse_transform(l);
        }
    }
    if let (
        KernelType::PositiveIndex {
            num_tasks, rank, ..
        },
        Some(MatrixPrior::Elementwise(prior)),
    ) = (&node.kind, &node.task_prior)
    {
        let (t, r) = (*num_tasks, *rank);
        let eps = 1e-6;
        let mut corr = crate::Matrix::identity(t, t);
        for i in 1..t {
            for j in 0..i {
                let v = prior.sample(rng).clamp(0.0, 1.0);
                corr[(i, j)] = v;
                corr[(j, i)] = v;
            }
        }
        let eig = corr.symmetric_eigen();
        let vals = eig.eigenvalues.map(|v| v.max(eps));
        let mut c =
            &eig.eigenvectors * crate::Matrix::from_diagonal(&vals) * eig.eigenvectors.transpose();
        let d: Vec<f64> = (0..t).map(|i| c[(i, i)].sqrt()).collect();
        for i in 0..t {
            for j in 0..t {
                c[(i, j)] /= d[i] * d[j];
            }
        }
        if let Some(chol) = c.cholesky() {
            let l = chol.l();
            // raw_var comes first, then raw_covar_factor (`t × r`).
            let fp = &node.params[1];
            let f_off = start + node.params[0].numel();
            for i in 0..t {
                for k in 0..r {
                    raw[f_off + i * r + k] = fp.inverse_transform(l[(i, k)].max(eps));
                }
            }
        }
    }
    for child in &node.children {
        sample_node(child, raw, off, rng);
    }
}

fn check_tasks(num_tasks: usize, rank: usize) -> Result<()> {
    if num_tasks == 0 || rank == 0 {
        return Err(BotorchError::InvalidArgument(
            "num_tasks and rank must be positive".into(),
        ));
    }
    if rank > num_tasks {
        return Err(BotorchError::InvalidArgument(
            "Cannot create a task covariance matrix larger than the number of tasks".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_named_parameters_order() {
        let a = KernelNode::stationary(KernelKind::Rbf, Some(2))
            .unwrap()
            .with_active_dims(vec![0, 2])
            .unwrap();
        let b = KernelNode::periodic(None).unwrap();
        let tree = KernelNode::scale(KernelNode::product(vec![a, b]).unwrap());
        let names: Vec<_> = tree.param_shapes("covar_module.").into_iter().collect();
        assert_eq!(
            names,
            vec![
                ("covar_module.raw_outputscale".to_string(), vec![]),
                (
                    "covar_module.base_kernel.kernels.0.raw_lengthscale".to_string(),
                    vec![1, 2]
                ),
                (
                    "covar_module.base_kernel.kernels.1.raw_lengthscale".to_string(),
                    vec![1, 1]
                ),
                (
                    "covar_module.base_kernel.kernels.1.raw_period_length".to_string(),
                    vec![1, 1]
                ),
            ]
        );
        assert_eq!(tree.num_raw(), 5);
        tree.validate(3).unwrap();
        assert!(tree.validate(2).is_err());
    }

    #[test]
    fn scale_takes_over_active_dims() {
        let base = KernelNode::stationary(KernelKind::Matern52, Some(1))
            .unwrap()
            .with_active_dims(vec![1])
            .unwrap();
        let s = KernelNode::scale(base);
        assert_eq!(s.active_dims(), Some(&[1usize][..]));
        assert_eq!(s.children()[0].active_dims(), None);
        assert!(s.as_kernel_spec(1).is_none());
        let plain = KernelNode::scale(KernelNode::stationary(KernelKind::Rbf, Some(3)).unwrap());
        let spec = plain.as_kernel_spec(3).unwrap();
        assert!(spec.scale.is_some());
        assert!(plain.as_kernel_spec(2).is_none());
    }

    #[test]
    fn constructor_errors() {
        assert!(KernelNode::index(2, 3).is_err());
        assert!(KernelNode::positive_index(3, 1, 3, true).is_err());
        let m = || KernelNode::stationary(KernelKind::Matern52, None).unwrap();
        assert!(KernelNode::linear_truncated_fidelity(vec![0, 0], m(), m()).is_err());
        assert!(KernelNode::linear_truncated_fidelity(vec![0, 1, 2], m(), m()).is_err());
        assert!(KernelNode::additive(vec![]).is_err());
        assert!(KernelNode::stationary(KernelKind::Rbf, Some(0)).is_err());
        assert!(m()
            .with_prior("raw_alpha", Prior::gamma(1.0, 1.0).unwrap())
            .is_err());
        let idx = KernelNode::index(3, 1).unwrap();
        assert!(idx
            .clone()
            .with_task_prior(MatrixPrior::LkjCovariance {
                n: 2,
                eta: 1.0,
                sd_prior: Prior::gamma(1.0, 1.0).unwrap()
            })
            .is_err());
        assert!(idx.validate(2).is_err());
    }
}
