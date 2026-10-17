//! `SingleTaskGP`: exact single-output GP regression with BoTorch's defaults.
//!
//! The model mirrors `botorch.models.SingleTaskGP` for a single output
//! (`m = 1`):
//!
//! * optional input transform (BoTorch default: none): a `Normalize`, or a
//!   chain of any of BoTorch's input transforms ([`ChainedInputTransform`],
//!   applied in train, eval or fantasize mode by their flags, with learned
//!   parameters such as `Warp`'s concentrations fitted with the GP), and
//!   outcome transform (BoTorch default: `Standardize`; or a chain of
//!   `Standardize`, `StratifiedStandardize`, `Log`, `Power` and `Bilog`,
//!   whose nonlinear steps make the posterior a `TransformedPosterior`, see
//!   [`crate::transforms::outcome`]);
//! * `ConstantMean` (initialized to zero, no prior);
//! * covariance module from [`get_covar_module_with_dim_scaled_prior`]
//!   (RBF, dimension-scaled LogNormal lengthscale prior) unless overridden,
//!   either by another [`KernelSpec`] or by a general kernel tree
//!   ([`KernelNode`]: sums, products and `ScaleKernel`s of GPyTorch's and
//!   BoTorch's kernels with `active_dims`);
//! * homoskedastic `GaussianLikelihood` with a `LogNormal(-4, 1)` noise prior
//!   and `GreaterThan(1e-4)` constraint, or a fixed-noise likelihood when
//!   observation variances `train_yvar` are given.
//!
//! Training data are stored both in the original space and transformed; the
//! hyperparameters are stored in BoTorch's raw parameterization. A Cholesky
//! factorization of the training covariance and the representer weights are
//! cached lazily and invalidated whenever the hyperparameters change.

use std::ops::Range;
use std::sync::OnceLock;

use crate::constraints::{Constraint, GreaterThan};
use crate::error::{BotorchError, Result};
use crate::kernels::{
    get_covar_module_with_dim_scaled_prior, BuiltCovar, CovarModule, Kernel, KernelKind,
    KernelNode, KernelSpec, Prepared, ScaleKernelSpec, TreeKernel,
};
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::models::model::clamp_variance;
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior, MIN_VARIANCE};
use crate::priors::Prior;
use crate::transforms::input::{ChainedInputTransform, InputTape, TransformMode};
use crate::transforms::outcome::{OutcomeChain, PointAffine, SampleStep, SampleTransform};
use crate::transforms::{Normalize, OutcomeTransformChoice, Standardize};
use crate::{Matrix, Vector};

mod conditioning;
mod noiseless_fantasy;

pub(crate) use conditioning::{FantasyAdjoint, FantasyGp, FantasyPosterior};

/// Lower bound on the inferred noise variance of BoTorch's default likelihoods.
pub const MIN_INFERRED_NOISE_LEVEL: f64 = 1e-4;

/// Minimum fixed noise variance (`gpytorch.settings.min_fixed_noise` for double
/// precision): smaller (transformed) observation variances are clamped to it,
/// exactly like GPyTorch's `FixedGaussianNoise`.
pub const MIN_FIXED_NOISE: f64 = 1e-6;

/// Configuration of a homoskedastic `gpytorch.likelihoods.GaussianLikelihood`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianLikelihoodSpec {
    /// Prior on the noise variance (evaluated on the actual noise).
    pub noise_prior: Option<Prior>,
    /// Constraint on the noise variance.
    pub noise_constraint: Constraint,
}

impl GaussianLikelihoodSpec {
    /// Initial raw noise: the constraint's initial value if present, otherwise
    /// zero (GPyTorch's default).
    pub fn initial_raw_noise(&self) -> f64 {
        self.noise_constraint.initial_raw_value().unwrap_or(0.0)
    }
}

/// BoTorch's `get_gaussian_likelihood_with_lognormal_prior` (the
/// `SingleTaskGP` default): noise prior `LogNormal(-4, 1)` and constraint
/// `GreaterThan(1e-4, transform=None, initial_value=prior.mode)`, with the
/// float32 rounding BoTorch applies to these constants.
pub fn get_gaussian_likelihood_with_lognormal_prior() -> GaussianLikelihoodSpec {
    let prior = Prior::LogNormal {
        loc: -4.0,
        scale: 1.0,
        float32: false,
        torch_log: None,
    }
    .rounded_to_f32();
    GaussianLikelihoodSpec {
        noise_prior: Some(prior),
        noise_constraint: GreaterThan::new(MIN_INFERRED_NOISE_LEVEL, None, Some(prior.mode_f32()))
            .rounded_to_f32()
            .into(),
    }
}

/// BoTorch's legacy `get_gaussian_likelihood_with_gamma_prior`: noise prior
/// `Gamma(1.1, 0.05)` and constraint `GreaterThan(1e-4, transform=None,
/// initial_value=(1.1 - 1) / 0.05)`.
pub fn get_gaussian_likelihood_with_gamma_prior() -> GaussianLikelihoodSpec {
    let prior = Prior::Gamma {
        concentration: 1.1,
        rate: 0.05,
    }
    .rounded_to_f32();
    GaussianLikelihoodSpec {
        noise_prior: Some(prior),
        noise_constraint: GreaterThan::new(MIN_INFERRED_NOISE_LEVEL, None, Some(prior.mode_f32()))
            .rounded_to_f32()
            .into(),
    }
}

/// The likelihood of a [`SingleTaskGP`].
#[derive(Clone, Debug, PartialEq)]
pub enum Likelihood {
    /// Homoskedastic Gaussian noise with a learned noise variance.
    Gaussian(GaussianLikelihoodSpec),
    /// Fixed, known noise (`FixedNoiseGaussianLikelihood`).
    FixedNoise {
        /// Noise variances of the training points in the *transformed*
        /// outcome space (i.e. `train_yvar / std²` under `Standardize`),
        /// clamped from below at [`MIN_FIXED_NOISE`].
        noise: Vector,
    },
}

/// Layout of the flat raw-parameter vector, which follows the order of
/// `model.named_parameters()` in BoTorch:
/// `likelihood.noise_covar.raw_noise` (if the noise is learned),
/// `mean_module.raw_constant`, then for a [`KernelSpec`]
/// `covar_module.raw_outputscale` (if a `ScaleKernel` is used) and the `d`
/// raw lengthscales, or for a kernel tree its `kernel` raw values in
/// `named_parameters()` order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterLayout {
    /// Whether a raw noise parameter is present.
    pub has_noise: bool,
    /// Whether a raw outputscale parameter is present (standard kernels).
    pub has_outputscale: bool,
    /// Number of lengthscales (input dimension; 0 for a kernel tree).
    pub dim: usize,
    /// Number of raw values of a kernel tree (0 for a [`KernelSpec`]).
    pub kernel: usize,
    /// Number of learned parameters of the input transform (e.g. `Warp`'s
    /// concentrations), last in BoTorch's order.
    pub input: usize,
}

impl ParameterLayout {
    /// Total number of raw parameters.
    pub fn len(&self) -> usize {
        usize::from(self.has_noise)
            + 1
            + usize::from(self.has_outputscale)
            + self.dim
            + self.kernel
            + self.input
    }

    /// Whether the layout is empty (never true for a valid model).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Index of the raw noise, if present.
    pub fn noise_index(&self) -> Option<usize> {
        self.has_noise.then_some(0)
    }

    /// Index of the raw constant mean.
    pub fn constant_index(&self) -> usize {
        usize::from(self.has_noise)
    }

    /// Index of the raw outputscale, if present.
    pub fn outputscale_index(&self) -> Option<usize> {
        self.has_outputscale.then_some(self.constant_index() + 1)
    }

    /// Index range of the raw lengthscales.
    pub fn lengthscale_range(&self) -> Range<usize> {
        let start = self.constant_index() + 1 + usize::from(self.has_outputscale);
        start..start + self.dim
    }

    /// Index range of the raw values of a kernel tree.
    pub fn kernel_range(&self) -> Range<usize> {
        let start = self.lengthscale_range().end;
        start..start + self.kernel
    }

    /// Index range of the learned parameters of the input transform.
    pub fn input_range(&self) -> Range<usize> {
        let start = self.kernel_range().end;
        start..start + self.input
    }
}

/// Hyperparameters of a [`SingleTaskGP`] in BoTorch's raw parameterization.
///
/// With BoTorch's default (non-enforced) constraints the raw values equal the
/// actual values; with GPyTorch's default `Positive()` constraints the actual
/// value is `softplus(raw)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Hyperparameters {
    /// `likelihood.noise_covar.raw_noise` (`None` for fixed noise).
    pub raw_noise: Option<f64>,
    /// `mean_module.raw_constant` (equal to the constant mean).
    pub raw_constant: f64,
    /// `covar_module.raw_outputscale` (`None` without `ScaleKernel`).
    pub raw_outputscale: Option<f64>,
    /// `covar_module(.base_kernel).raw_lengthscale` (length `d`; empty for a
    /// kernel tree).
    pub raw_lengthscale: Vector,
    /// The raw values of a kernel tree in `named_parameters()` order (empty
    /// for a [`KernelSpec`]).
    pub raw_kernel: Vector,
    /// The learned parameters of the input transform (e.g. `Warp`'s
    /// concentrations, which are not transformed by their constraint), in
    /// `named_parameters()` order; empty without such parameters.
    pub raw_input: Vector,
}

impl Hyperparameters {
    /// The layout of [`Hyperparameters::to_vector`].
    pub fn layout(&self) -> ParameterLayout {
        ParameterLayout {
            has_noise: self.raw_noise.is_some(),
            has_outputscale: self.raw_outputscale.is_some(),
            dim: self.raw_lengthscale.len(),
            kernel: self.raw_kernel.len(),
            input: self.raw_input.len(),
        }
    }

    /// Flattens the parameters in BoTorch's `named_parameters()` order.
    pub fn to_vector(&self) -> Vector {
        let mut v = Vec::with_capacity(self.layout().len());
        if let Some(n) = self.raw_noise {
            v.push(n);
        }
        v.push(self.raw_constant);
        if let Some(s) = self.raw_outputscale {
            v.push(s);
        }
        v.extend(self.raw_lengthscale.iter());
        v.extend(self.raw_kernel.iter());
        v.extend(self.raw_input.iter());
        Vector::from_vec(v)
    }

    /// Inverse of [`Hyperparameters::to_vector`].
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `values.len() != layout.len()`.
    pub fn from_slice(layout: ParameterLayout, values: &[f64]) -> Result<Self> {
        if values.len() != layout.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                layout.len(),
                values.len()
            )));
        }
        Ok(Self {
            raw_noise: layout.noise_index().map(|i| values[i]),
            raw_constant: values[layout.constant_index()],
            raw_outputscale: layout.outputscale_index().map(|i| values[i]),
            raw_lengthscale: Vector::from_column_slice(&values[layout.lengthscale_range()]),
            raw_kernel: Vector::from_column_slice(&values[layout.kernel_range()]),
            raw_input: Vector::from_column_slice(&values[layout.input_range()]),
        })
    }
}

/// Options of [`SingleTaskGP::new`], mirroring the keyword arguments of
/// BoTorch's `SingleTaskGP`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SingleTaskGPOptions {
    /// Observed noise variances (`n × 1`); if given, a fixed-noise likelihood
    /// is used instead of learning a homoskedastic noise level.
    pub train_yvar: Option<Matrix>,
    /// Optional input transform (BoTorch default: none). A `Normalize` with
    /// learned bounds is fitted to the training inputs.
    pub input_transform: Option<Normalize>,
    /// A general input transform (BoTorch's `ChainedInputTransform` or any
    /// single input transform, see [`crate::transforms::input`]) instead of
    /// `input_transform`: fitted and applied to the training inputs in train
    /// mode, applied to the points of every posterior in eval mode (possibly
    /// one-to-many, possibly changing the dimension).
    pub input_chain: Option<ChainedInputTransform>,
    /// Outcome transform (BoTorch default: `Standardize`).
    pub outcome_transform: OutcomeTransformChoice,
    /// Kernel family of the default covariance module (ignored if
    /// `covar_module` is given).
    pub kernel: KernelKind,
    /// Wrap the default covariance module in a `ScaleKernel` with BoTorch's
    /// legacy `GammaPrior(2.0, 0.15)` outputscale prior (ignored if
    /// `covar_module` is given).
    pub use_scale_kernel: bool,
    /// Full covariance-module override (BoTorch's `covar_module` argument),
    /// e.g. [`crate::kernels::get_matern_kernel_with_gamma_prior`].
    pub covar_module: Option<KernelSpec>,
    /// A general covariance module (a kernel tree) instead of
    /// `covar_module` (BoTorch's `covar_module` argument with any GPyTorch or
    /// BoTorch kernel).
    pub covar_tree: Option<KernelNode>,
    /// Override of the homoskedastic likelihood (BoTorch's `likelihood`
    /// argument), e.g. [`get_gaussian_likelihood_with_gamma_prior`]. Must be
    /// `None` when `train_yvar` is given.
    pub likelihood: Option<GaussianLikelihoodSpec>,
}

/// Quantities derived from the hyperparameters and cached for prediction.
#[derive(Clone, Debug)]
struct PredictionCache {
    kernel: BuiltCovar,
    train_scaled: Prepared,
    chol: CholeskyFactor,
    /// `K⁻¹ (y - c)` in the transformed outcome space.
    alpha: Vector,
}

/// Exact GP regression model with BoTorch `SingleTaskGP` semantics (single
/// output, double precision).
#[derive(Clone, Debug)]
pub struct SingleTaskGP {
    train_x: Matrix,
    train_y: Matrix,
    train_yvar: Option<Matrix>,
    train_x_tf: Matrix,
    train_y_tf: Vector,
    input_transform: Option<Normalize>,
    input_chain: Option<ChainedInputTransform>,
    /// The training inputs as transformed when the model was built, if they
    /// differ from `train_x_tf` (steps that were in eval mode then).
    construction_x_tf: Option<Matrix>,
    outcome_transform: Option<Standardize>,
    /// A general outcome transform (exclusive with `outcome_transform`).
    outcome_chain: Option<OutcomeChain>,
    covar_module: CovarModule,
    likelihood: Likelihood,
    hyperparameters: Hyperparameters,
    /// Noise variances (transformed space) of training points that were
    /// added by conditioning a homoskedastic model on observations with
    /// explicit `noise`; `None` entries (and all points if absent) use the
    /// likelihood noise. Like BoTorch's fantasy models, the explicit noise
    /// only enters the predictions, not the marginal log likelihood.
    point_noise: Option<Vec<Option<f64>>>,
    cache: OnceLock<std::result::Result<PredictionCache, BotorchError>>,
}

fn check_finite(m: &Matrix, what: &str) -> Result<()> {
    if m.iter().any(|v| v.is_nan()) {
        return Err(BotorchError::InputData(format!(
            "{what} contains NaN values."
        )));
    }
    Ok(())
}

impl SingleTaskGP {
    /// Builds a model from training inputs `train_x` (`n × d`, rows are
    /// points) and targets `train_y` (`n × 1`).
    ///
    /// # Errors
    /// * [`BotorchError::Unsupported`] if `train_y` has more than one column;
    /// * [`BotorchError::DimensionMismatch`] for inconsistent shapes;
    /// * [`BotorchError::InputData`] for empty data, NaN values or negative
    ///   noise variances;
    /// * [`BotorchError::InvalidArgument`] for inconsistent options.
    pub fn new(train_x: Matrix, train_y: Matrix, options: SingleTaskGPOptions) -> Result<Self> {
        let (n, d) = train_x.shape();
        if train_y.ncols() > 1 {
            return Err(BotorchError::Unsupported(format!(
                "SingleTaskGP in botorch_core currently supports a single output (m = 1), got m = {}",
                train_y.ncols()
            )));
        }
        if train_y.ncols() == 0 {
            return Err(BotorchError::DimensionMismatch(
                "train_y must have one column".into(),
            ));
        }
        if n == 0 || d == 0 {
            return Err(BotorchError::InputData(format!(
                "train_x must have at least one row and one column, got {n}x{d}"
            )));
        }
        if train_y.nrows() != n {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X and Y to have the same number of rows (got X with {n} and Y with {}).",
                train_y.nrows()
            )));
        }
        if let Some(yvar) = &options.train_yvar {
            if yvar.shape() != train_y.shape() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected Yvar with the same shape as Y {:?}, got {:?}.",
                    train_y.shape(),
                    yvar.shape()
                )));
            }
            check_finite(yvar, "train_yvar")?;
            if yvar.iter().any(|&v| v < 0.0) {
                return Err(BotorchError::InputData(
                    "Input data contains negative variances.".into(),
                ));
            }
            if options.likelihood.is_some() {
                return Err(BotorchError::InvalidArgument(
                    "a custom homoskedastic likelihood cannot be combined with train_yvar".into(),
                ));
            }
        }
        check_finite(&train_x, "train_x")?;
        check_finite(&train_y, "train_y")?;

        // Input transform (fitted on the training inputs, as in train mode).
        let input_transform = match options.input_transform {
            Some(mut tf) => {
                if tf.dim() != d {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "input transform has dimension {} but train_x has {d} columns",
                        tf.dim()
                    )));
                }
                tf.fit(&train_x)?;
                Some(tf)
            }
            None => None,
        };
        let mut construction_x_tf = None;
        let input_chain = match options.input_chain {
            Some(mut chain) => {
                if input_transform.is_some() {
                    return Err(BotorchError::InvalidArgument(
                        "input_transform and input_chain cannot both be given".into(),
                    ));
                }
                construction_x_tf = Some(chain.fit(&train_x)?);
                Some(chain)
            }
            None => None,
        };
        let train_x_tf = match (&input_transform, &input_chain) {
            (Some(tf), _) => tf.transform(&train_x)?,
            (None, Some(chain)) => chain.preprocess(&train_x)?,
            (None, None) => train_x.clone(),
        };
        // BoTorch transforms the training inputs at construction in the
        // transforms' modes; keep that version only if it differs.
        let construction_x_tf = construction_x_tf.filter(|c| c != &train_x_tf);
        check_finite(&train_x_tf, "transformed train_x")?;
        let x_for_outcome = construction_x_tf.as_ref().unwrap_or(&train_x_tf);

        // Outcome transform.
        let mut outcome_chain = None;
        let mut outcome_transform = match options.outcome_transform {
            OutcomeTransformChoice::Default => Some(Standardize::new()),
            OutcomeTransformChoice::None => None,
            OutcomeTransformChoice::Standardize(s) => Some(s),
            OutcomeTransformChoice::Chains(mut chains) => {
                if chains.len() != 1 {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "a single-output model needs one outcome chain, got {}",
                        chains.len()
                    )));
                }
                outcome_chain = chains.pop();
                None
            }
        };
        let (y_tf, yvar_tf) = match (outcome_transform.as_mut(), outcome_chain.as_mut()) {
            (Some(st), _) => st.fit_transform(&train_y, options.train_yvar.as_ref())?,
            (None, Some(chain)) => {
                chain.fit_transform(&train_y, options.train_yvar.as_ref(), x_for_outcome)?
            }
            (None, None) => (train_y.clone(), options.train_yvar.clone()),
        };
        check_finite(&y_tf, "transformed train_y")?;
        if let Some(v) = &yvar_tf {
            check_finite(v, "transformed train_yvar")?;
        }

        let likelihood = match &yvar_tf {
            Some(v) => Likelihood::FixedNoise {
                noise: Vector::from_iterator(
                    n,
                    v.column(0).iter().map(|&s| s.max(MIN_FIXED_NOISE)),
                ),
            },
            None => Likelihood::Gaussian(
                options
                    .likelihood
                    .unwrap_or_else(get_gaussian_likelihood_with_lognormal_prior),
            ),
        };

        // The kernel sees the transformed inputs (whose dimension some input
        // transforms change).
        let d_tf = train_x_tf.ncols();
        let covar_module = match (options.covar_module, options.covar_tree) {
            (Some(_), Some(_)) => {
                return Err(BotorchError::InvalidArgument(
                    "covar_module and covar_tree cannot both be given".into(),
                ))
            }
            (Some(spec), None) => CovarModule::Standard(spec),
            (None, Some(tree)) => {
                tree.validate(d_tf)?;
                CovarModule::Tree(tree)
            }
            (None, None) => {
                let base =
                    get_covar_module_with_dim_scaled_prior(d_tf, options.kernel == KernelKind::Rbf);
                CovarModule::Standard(if options.use_scale_kernel {
                    base.with_scale_kernel(ScaleKernelSpec::legacy_gamma())
                } else {
                    base
                })
            }
        };
        if let CovarModule::Standard(spec) = &covar_module {
            if spec.ard_num_dims != d_tf {
                return Err(BotorchError::DimensionMismatch(format!(
                    "covar_module has ard_num_dims = {} but the transformed train_x has {d_tf}                      columns",
                    spec.ard_num_dims
                )));
            }
        }

        let raw_noise = match &likelihood {
            Likelihood::Gaussian(spec) => Some(spec.initial_raw_noise()),
            Likelihood::FixedNoise { .. } => None,
        };
        let raw_input = Vector::from_vec(
            input_chain
                .as_ref()
                .map(ChainedInputTransform::params)
                .unwrap_or_default(),
        );
        let hyperparameters = match &covar_module {
            CovarModule::Standard(spec) => Hyperparameters {
                raw_noise,
                raw_constant: 0.0,
                raw_outputscale: spec.initial_raw_outputscale(),
                raw_lengthscale: Vector::from_element(d_tf, spec.initial_raw_lengthscale()),
                raw_kernel: Vector::zeros(0),
                raw_input,
            },
            CovarModule::Tree(tree) => Hyperparameters {
                raw_noise,
                raw_constant: 0.0,
                raw_outputscale: None,
                raw_lengthscale: Vector::zeros(0),
                raw_kernel: Vector::from_vec(tree.initial_raw()),
                raw_input,
            },
        };

        Ok(Self {
            train_x,
            train_y,
            train_yvar: options.train_yvar,
            train_x_tf,
            train_y_tf: Vector::from_column_slice(y_tf.column(0).as_slice()),
            input_transform,
            input_chain,
            construction_x_tf,
            outcome_transform,
            outcome_chain,
            covar_module,
            likelihood,
            hyperparameters,
            point_noise: None,
            cache: OnceLock::new(),
        })
    }

    // ------------------------------------------------------------------
    // Data and configuration accessors
    // ------------------------------------------------------------------

    /// Training inputs in the original input space (`n × d`).
    pub fn train_inputs(&self) -> &Matrix {
        &self.train_x
    }

    /// Training targets in the original outcome space (`n × 1`).
    pub fn train_targets(&self) -> &Matrix {
        &self.train_y
    }

    /// Observed noise variances in the original outcome space, if given.
    pub fn train_yvar(&self) -> Option<&Matrix> {
        self.train_yvar.as_ref()
    }

    /// Training inputs after the input transform (`n × d`).
    pub fn transformed_train_inputs(&self) -> &Matrix {
        &self.train_x_tf
    }

    /// Training targets after the outcome transform (BoTorch's
    /// `model.train_targets`, length `n`).
    pub fn transformed_train_targets(&self) -> &Vector {
        &self.train_y_tf
    }

    /// Number of training points `n`.
    pub fn num_train(&self) -> usize {
        self.train_x.nrows()
    }

    /// Input dimension `d` (of the points of a posterior: that of the
    /// training inputs unless the input transform maps both to the kernel's
    /// dimensions differently, e.g. `AppendFeatures` with training inputs
    /// that include the features).
    pub fn dim(&self) -> usize {
        let d = self.train_x.ncols();
        self.input_chain.as_ref().map_or(d, |c| c.input_dim(d))
    }

    /// The dimension of the transformed inputs (what the kernel sees; `d`
    /// unless the input transform changes it, e.g. `AppendFeatures`).
    pub fn transformed_dim(&self) -> usize {
        self.train_x_tf.ncols()
    }

    /// The training inputs as transformed when the model was built (BoTorch's
    /// `transformed_X` in the constructor, used for its input-scaling
    /// warnings): the transformed training inputs unless some step of the
    /// input transform was in eval mode then.
    pub fn construction_train_inputs(&self) -> &Matrix {
        self.construction_x_tf.as_ref().unwrap_or(&self.train_x_tf)
    }

    /// Number of outputs (always 1 in this version).
    pub fn num_outputs(&self) -> usize {
        1
    }

    /// The fitted input transform, if any.
    pub fn input_transform(&self) -> Option<&Normalize> {
        self.input_transform.as_ref()
    }

    /// The fitted general input transform (`ChainedInputTransform` /
    /// `InputPerturbation`), if any.
    pub fn input_chain(&self) -> Option<&ChainedInputTransform> {
        self.input_chain.as_ref()
    }

    /// The number of posterior points per input point (`n_w` of a one-to-many
    /// input transform, else 1; see [`Model::points_per_input`]).
    pub fn points_per_input(&self) -> usize {
        self.input_chain
            .as_ref()
            .map_or(1, ChainedInputTransform::points_per_input)
    }

    /// Errors for a one-to-many input transform: conditioning on
    /// observations and fantasizing (`what`) need one model input per
    /// observation. BoTorch fantasizes in "fantasize mode", which skips
    /// perturbations with `transform_on_fantasize=False`; that mode is not
    /// ported (see [`crate::transforms::input`]).
    pub(crate) fn require_no_perturbation(&self, what: &str) -> Result<()> {
        let k = self.points_per_input();
        if k != 1 {
            return Err(BotorchError::Unsupported(format!(
                "{what} is not supported for models with a one-to-many input transform \
                 (InputPerturbation with n_w = {k}): the perturbed inputs would not match the \
                 observations"
            )));
        }
        Ok(())
    }

    /// Errors unless every input maps to a single posterior point.
    fn require_one_point_per_input(&self) -> Result<()> {
        let k = self.points_per_input();
        if k != 1 {
            return Err(BotorchError::Unsupported(format!(
                "this evaluates every input point on its own, which requires one posterior \
                 point per input; the model's one-to-many input transform (InputPerturbation) \
                 gives {k}"
            )));
        }
        Ok(())
    }

    /// The fitted outcome transform if it is a plain `Standardize` (also as
    /// the only step of a chain).
    pub fn outcome_transform(&self) -> Option<&Standardize> {
        self.outcome_transform.as_ref().or_else(|| {
            self.outcome_chain
                .as_ref()
                .and_then(OutcomeChain::as_standardize)
        })
    }

    /// The fitted general outcome transform (see
    /// [`crate::transforms::outcome`]), if the model has one.
    pub fn outcome_chain(&self) -> Option<&OutcomeChain> {
        self.outcome_chain.as_ref()
    }

    /// Whether the posterior is a BoTorch `TransformedPosterior` (a
    /// nonlinear outcome transform: `Log`, `Power`, `Bilog`).
    pub fn has_transformed_posterior(&self) -> bool {
        self.outcome_chain.as_ref().is_some_and(|c| !c.is_linear())
    }

    /// Whether the outcome transform is a constant affine map (no transform,
    /// `Standardize` or chains of them): the posterior is Gaussian and its
    /// un-standardization does not depend on the inputs.
    pub fn has_plain_outcome(&self) -> bool {
        self.outcome_chain
            .as_ref()
            .is_none_or(|c| c.is_linear() && !c.depends_on_inputs())
    }

    /// Errors unless the outcome transform is a constant affine map (see
    /// [`SingleTaskGP::has_plain_outcome`]), for computations (`what`) that
    /// support only those.
    pub(crate) fn require_plain_outcome(&self, what: &str) -> Result<()> {
        if !self.has_plain_outcome() {
            return Err(BotorchError::Unsupported(format!(
                "{what} is not supported for models with a nonlinear or input-dependent outcome \
                 transform (Log, Power, Bilog, StratifiedStandardize)"
            )));
        }
        Ok(())
    }

    /// Errors unless the input transform is a `Normalize` or a chain of
    /// `Normalize`s and affine transforms (constant, diagonal Jacobians), for
    /// computations (`what`) that support only those.
    pub(crate) fn require_affine_inputs(&self, what: &str) -> Result<()> {
        if self.input_jacobian().is_none() {
            return Err(BotorchError::Unsupported(format!(
                "{what} is not supported for models with this input transform (only Normalize \
                 and other affine input transforms)"
            )));
        }
        Ok(())
    }

    /// The covariance-module configuration.
    pub fn covar_module(&self) -> &CovarModule {
        &self.covar_module
    }

    /// The likelihood.
    pub fn likelihood(&self) -> &Likelihood {
        &self.likelihood
    }

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------

    /// The current raw hyperparameters.
    pub fn hyperparameters(&self) -> &Hyperparameters {
        &self.hyperparameters
    }

    /// Sets the raw hyperparameters (and invalidates cached factorizations).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the structure does not match the
    /// model (noise / outputscale presence, number of lengthscales) and
    /// [`BotorchError::InvalidArgument`] for non-finite values or values that
    /// map to non-positive lengthscales or outputscales or to a negative noise
    /// variance (a zero noise variance is allowed, as in GPyTorch; the
    /// optimizer bounds keep fitted noise levels at or above `1e-4`).
    pub fn set_hyperparameters(&mut self, hyperparameters: Hyperparameters) -> Result<()> {
        if hyperparameters.layout() != self.parameter_layout() {
            return Err(BotorchError::DimensionMismatch(format!(
                "hyperparameter layout {:?} does not match the model layout {:?}",
                hyperparameters.layout(),
                self.parameter_layout()
            )));
        }
        let values = hyperparameters.to_vector();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "hyperparameters must be finite".into(),
            ));
        }
        self.build_covar(&hyperparameters)?;
        // Learned input-transform parameters change the transformed inputs.
        let input_update = match &self.input_chain {
            Some(chain)
                if hyperparameters.raw_input != self.hyperparameters.raw_input
                    && chain.num_params() > 0 =>
            {
                let chain = chain.with_params(hyperparameters.raw_input.as_slice())?;
                let x_tf = chain.preprocess(&self.train_x)?;
                check_finite(&x_tf, "transformed train_x")?;
                Some((chain, x_tf))
            }
            _ => None,
        };
        if let (Likelihood::Gaussian(spec), Some(raw)) =
            (&self.likelihood, hyperparameters.raw_noise)
        {
            let noise = spec.noise_constraint.transform(raw);
            if noise.is_nan() || noise < 0.0 {
                return Err(BotorchError::InvalidArgument(format!(
                    "noise must be non-negative, got {noise}"
                )));
            }
        }
        if let Some((chain, x_tf)) = input_update {
            self.input_chain = Some(chain);
            self.train_x_tf = x_tf;
        }
        self.hyperparameters = hyperparameters;
        self.cache = OnceLock::new();
        Ok(())
    }

    /// The layout of the flat raw-parameter vector.
    pub fn parameter_layout(&self) -> ParameterLayout {
        let has_noise = matches!(self.likelihood, Likelihood::Gaussian(_));
        let input = self
            .input_chain
            .as_ref()
            .map_or(0, ChainedInputTransform::num_params);
        match &self.covar_module {
            CovarModule::Standard(spec) => ParameterLayout {
                has_noise,
                has_outputscale: spec.scale.is_some(),
                dim: self.transformed_dim(),
                kernel: 0,
                input,
            },
            CovarModule::Tree(tree) => ParameterLayout {
                has_noise,
                has_outputscale: false,
                dim: 0,
                kernel: tree.num_raw(),
                input,
            },
        }
    }

    /// The raw hyperparameters flattened in BoTorch's `named_parameters()` order.
    pub fn raw_parameters(&self) -> Vector {
        self.hyperparameters.to_vector()
    }

    /// Sets the raw hyperparameters from a flat vector (see [`ParameterLayout`]).
    ///
    /// # Errors
    /// See [`SingleTaskGP::set_hyperparameters`].
    pub fn set_raw_parameters(&mut self, values: &[f64]) -> Result<()> {
        let h = Hyperparameters::from_slice(self.parameter_layout(), values)?;
        self.set_hyperparameters(h)
    }

    /// BoTorch parameter names and sizes in flat-vector order, e.g.
    /// `("covar_module.raw_lengthscale", d)`.
    pub fn parameter_names(&self) -> Vec<(String, usize)> {
        self.parameter_element_shapes()
            .into_iter()
            .map(|(name, shape)| (name, shape.iter().product()))
            .collect()
    }

    /// BoTorch parameter names and shapes in flat-vector order (the shapes of
    /// an unbatched model, e.g. `("covar_module.raw_lengthscale", [1, d])`).
    pub fn parameter_element_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let layout = self.parameter_layout();
        let mut names = Vec::new();
        if layout.has_noise {
            names.push(("likelihood.noise_covar.raw_noise".to_string(), vec![1]));
        }
        names.push(("mean_module.raw_constant".to_string(), vec![]));
        match &self.covar_module {
            CovarModule::Standard(_) => {
                if layout.has_outputscale {
                    names.push(("covar_module.raw_outputscale".to_string(), vec![]));
                    names.push((
                        "covar_module.base_kernel.raw_lengthscale".to_string(),
                        vec![1, layout.dim],
                    ));
                } else {
                    names.push((
                        "covar_module.raw_lengthscale".to_string(),
                        vec![1, layout.dim],
                    ));
                }
            }
            CovarModule::Tree(tree) => names.extend(tree.param_shapes("covar_module.")),
        }
        if let Some(chain) = &self.input_chain {
            names.extend(chain.param_shapes());
        }
        names
    }

    /// Optimizer bounds for every raw parameter in flat-vector order, as
    /// produced by `botorch.optim.utils.get_parameters_and_bounds`
    /// (unconstrained parameters get `(-inf, inf)`).
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let layout = self.parameter_layout();
        let mut bounds = Vec::with_capacity(layout.len());
        if let Likelihood::Gaussian(spec) = &self.likelihood {
            bounds.push(spec.noise_constraint.raw_bounds());
        }
        bounds.push((f64::NEG_INFINITY, f64::INFINITY));
        match &self.covar_module {
            CovarModule::Standard(spec) => {
                if let Some(scale) = &spec.scale {
                    bounds.push(scale.outputscale_constraint.raw_bounds());
                }
                let ls = spec.lengthscale_constraint.raw_bounds();
                bounds.extend(std::iter::repeat_n(ls, layout.dim));
            }
            CovarModule::Tree(tree) => bounds.extend(tree.raw_bounds()),
        }
        if let Some(chain) = &self.input_chain {
            bounds.extend(chain.param_bounds());
        }
        bounds
    }

    /// The constant mean (in the transformed outcome space).
    pub fn constant(&self) -> f64 {
        self.hyperparameters.raw_constant
    }

    /// The actual lengthscales of a standard kernel (empty for a kernel
    /// tree).
    pub fn lengthscale(&self) -> Vector {
        match &self.covar_module {
            CovarModule::Standard(spec) => self
                .hyperparameters
                .raw_lengthscale
                .map(|r| spec.lengthscale_constraint.transform(r)),
            CovarModule::Tree(_) => Vector::zeros(0),
        }
    }

    /// The actual outputscale, if a standard kernel with a `ScaleKernel` is
    /// used.
    pub fn outputscale(&self) -> Option<f64> {
        let CovarModule::Standard(spec) = &self.covar_module else {
            return None;
        };
        match (&spec.scale, self.hyperparameters.raw_outputscale) {
            (Some(s), Some(raw)) => Some(s.outputscale_constraint.transform(raw)),
            _ => None,
        }
    }

    /// The likelihood noise variance in the transformed outcome space, like
    /// GPyTorch's `likelihood.noise`: a single value for a homoskedastic
    /// likelihood and one value per training point for fixed noise.
    pub fn likelihood_noise(&self) -> Vector {
        match &self.likelihood {
            Likelihood::Gaussian(spec) => Vector::from_element(
                1,
                spec.noise_constraint.transform(
                    self.hyperparameters
                        .raw_noise
                        .expect("homoskedastic models have a noise parameter"),
                ),
            ),
            Likelihood::FixedNoise { noise } => noise.clone(),
        }
    }

    /// The standard kernel with the current hyperparameters.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if the hyperparameters are invalid
    /// and [`BotorchError::Unsupported`] for a kernel tree (see
    /// [`SingleTaskGP::covar`]).
    pub fn kernel(&self) -> Result<Kernel> {
        match &self.covar_module {
            CovarModule::Standard(spec) => spec.build(
                self.hyperparameters.raw_lengthscale.as_slice(),
                self.hyperparameters.raw_outputscale,
            ),
            CovarModule::Tree(_) => Err(BotorchError::Unsupported(
                "the covariance module is a kernel tree, not an RBF or Matern kernel".into(),
            )),
        }
    }

    /// The covariance module with the current hyperparameters.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if the hyperparameters are invalid.
    pub fn covar(&self) -> Result<BuiltCovar> {
        self.build_covar(&self.hyperparameters)
    }

    /// The covariance module with the given hyperparameters.
    pub(crate) fn build_covar(&self, h: &Hyperparameters) -> Result<BuiltCovar> {
        Ok(match &self.covar_module {
            CovarModule::Standard(spec) => {
                BuiltCovar::Standard(spec.build(h.raw_lengthscale.as_slice(), h.raw_outputscale)?)
            }
            CovarModule::Tree(tree) => BuiltCovar::Tree(TreeKernel::build(
                tree,
                h.raw_kernel.as_slice(),
                self.transformed_dim(),
            )?),
        })
    }

    /// Mean and standard deviation of a plain outcome transform (`(0, 1)`
    /// without one): the constant affine un-transform of the posterior.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for a nonlinear or input-dependent
    /// outcome transform (see [`SingleTaskGP::has_plain_outcome`]).
    pub(crate) fn outcome_affine(&self) -> Result<(f64, f64)> {
        match (&self.outcome_transform, &self.outcome_chain) {
            (Some(st), _) => Ok((st.means()[0], st.stdvs()[0])),
            (None, Some(chain)) => {
                self.require_plain_outcome("this computation")?;
                match chain
                    .posterior_parts(&Matrix::zeros(0, self.transformed_dim()))?
                    .0
                {
                    PointAffine::Constant(o, s) => Ok((o, s)),
                    PointAffine::PerPoint(_) => unreachable!("plain outcomes are constant"),
                }
            }
            (None, None) => Ok((0.0, 1.0)),
        }
    }

    /// The un-transformation of the posterior at the transformed points
    /// `x_tf`: the affine map of every point and the sample steps of a
    /// `TransformedPosterior` (see [`OutcomeChain::posterior_parts`]).
    pub(crate) fn outcome_parts(&self, x_tf: &Matrix) -> Result<(PointAffine, Vec<SampleStep>)> {
        match (&self.outcome_transform, &self.outcome_chain) {
            (Some(st), _) => Ok((
                PointAffine::Constant(st.means()[0], st.stdvs()[0]),
                Vec::new(),
            )),
            (None, Some(chain)) => chain.posterior_parts(x_tf),
            (None, None) => Ok((PointAffine::IDENTITY, Vec::new())),
        }
    }

    /// The observation-noise variance (transformed outcome space) added by
    /// `posterior(.., observation_noise = true)`: the learned noise, or the
    /// mean of the fixed training noise (BoTorch's choice for fixed-noise
    /// models).
    fn observation_noise_transformed(&self) -> f64 {
        let noise = self.likelihood_noise();
        noise.sum() / noise.len() as f64
    }

    /// The observation-noise variance in the *original* outcome space that
    /// `posterior(x, true)` adds to the diagonal of the posterior covariance.
    pub fn observation_noise(&self) -> f64 {
        let sd = self.outcome_affine().map_or(f64::NAN, |a| a.1);
        sd * sd * self.observation_noise_transformed()
    }

    // ------------------------------------------------------------------
    // Prediction
    // ------------------------------------------------------------------

    fn compute_cache(&self) -> Result<PredictionCache> {
        let kernel = self.covar()?;
        let train_scaled = kernel.prepare(&self.train_x_tf)?;
        let mut k = kernel.symmetric(&train_scaled);
        let noise = self.likelihood_noise();
        let n = self.num_train();
        for i in 0..n {
            k[(i, i)] += self.training_noise(&noise, i);
        }
        let chol = psd_safe_cholesky(&k)?;
        let resid = self.train_y_tf.add_scalar(-self.constant());
        let alpha = chol.solve_vec(&resid);
        Ok(PredictionCache {
            kernel,
            train_scaled,
            chol,
            alpha,
        })
    }

    fn cache(&self) -> Result<&PredictionCache> {
        self.cache
            .get_or_init(|| self.compute_cache())
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The Cholesky factor of the (jittered) training covariance
    /// `K(X, X) + noise` in the transformed space (computed lazily).
    ///
    /// # Errors
    /// [`BotorchError::NotPsd`] if the factorization fails even with jitter.
    pub fn train_cholesky(&self) -> Result<&CholeskyFactor> {
        Ok(&self.cache()?.chol)
    }

    fn check_columns(&self, x: &Matrix, mode: TransformMode) -> Result<()> {
        if x.ncols() != self.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected inputs with {} columns, got {}",
                self.dim(),
                x.ncols()
            )));
        }
        if let Some(chain) = &self.input_chain {
            chain.check_mode_dim(x.ncols(), self.transformed_dim(), mode)?;
        }
        Ok(())
    }

    /// The points of a posterior at the raw-space inputs `x`, transformed by
    /// the input transform in eval mode.
    pub(crate) fn transform_inputs(&self, x: &Matrix) -> Result<Matrix> {
        self.transform_inputs_mode(x, TransformMode::Eval)
    }

    /// The raw-space inputs `x` transformed by the input transform in `mode`.
    pub(crate) fn transform_inputs_mode(&self, x: &Matrix, mode: TransformMode) -> Result<Matrix> {
        self.check_columns(x, mode)?;
        if let Some(chain) = &self.input_chain {
            return chain.transform_mode(x, mode);
        }
        match &self.input_transform {
            Some(tf) => tf.transform(x),
            None => Ok(x.clone()),
        }
    }

    /// The transformed inputs of a general input transform with the record
    /// of their derivatives (`None` without one).
    fn chain_tape(&self, x: &Matrix) -> Result<Option<(Matrix, InputTape)>> {
        let Some(chain) = &self.input_chain else {
            return Ok(None);
        };
        self.check_columns(x, TransformMode::Eval)?;
        chain.transform_with_tape(x).map(Some)
    }

    /// The reverse pass of a tape of the model's input chain.
    fn tape_backward(
        &self,
        tape: &InputTape,
        rows: Range<usize>,
        adjoint: &Matrix,
    ) -> Result<Matrix> {
        let chain = self
            .input_chain
            .as_ref()
            .expect("tapes are recorded by the input chain");
        tape.backward_through(chain, rows, adjoint)
    }

    /// Chain-rule factors `∂x̃ⱼ/∂xⱼ` of an input transform with a constant
    /// diagonal Jacobian (none, `Normalize`, or a chain of affine steps that
    /// keeps the dimension); `None` for other input transforms.
    pub(crate) fn input_jacobian(&self) -> Option<Vector> {
        if let Some(chain) = &self.input_chain {
            return chain
                .constant_diag_jacobian(TransformMode::Eval, self.dim())
                .map(Vector::from_vec);
        }
        Some(self.legacy_jacobian())
    }

    /// The Jacobian of the plain `Normalize` input transform (ones without).
    fn legacy_jacobian(&self) -> Vector {
        match &self.input_transform {
            Some(tf) => tf.coefficient().map(|c| 1.0 / c),
            None => Vector::from_element(self.dim(), 1.0),
        }
    }

    /// Joint posterior at the `q × d` raw-space inputs `x`, in the original
    /// outcome space. With `observation_noise`, the likelihood noise is added
    /// to the diagonal (the mean training noise for fixed-noise models).
    ///
    /// For a nonlinear outcome transform the result is BoTorch's
    /// `TransformedPosterior`: the Gaussian base posterior with its
    /// [`Posterior::sample_transform`].
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a column mismatch and
    /// [`BotorchError::NotPsd`] if the training covariance cannot be factorized.
    pub fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let noise = if observation_noise {
            ObservationNoise::Likelihood
        } else {
            ObservationNoise::None
        };
        self.posterior_impl(x, noise, TransformMode::Eval)
    }

    /// [`SingleTaskGP::posterior`] with the input transform in `mode` and
    /// the given observation noise.
    pub(crate) fn posterior_impl(
        &self,
        x: &Matrix,
        noise: ObservationNoise<'_>,
        mode: TransformMode,
    ) -> Result<Posterior> {
        let xt = self.transform_inputs_mode(x, mode)?;
        let cache = self.cache()?;
        let q = xt.nrows();
        let xs = cache.kernel.prepare(&xt)?;
        let kxs = cache.kernel.cross(&xs, &cache.train_scaled);
        let mut cov = cache.kernel.symmetric(&xs);
        let mean_tf = (&kxs * &cache.alpha).add_scalar(self.constant());
        let mut v = kxs.transpose();
        cache.chol.solve_lower_mut(&mut v);
        cov.gemm_tr(-1.0, &v, &v, 1.0);
        if let ObservationNoise::Likelihood = noise {
            let noise = self.observation_noise_transformed();
            for i in 0..q {
                cov[(i, i)] += noise;
            }
        }
        let (aff, samples) = self.outcome_parts(&xt)?;
        let (mean, mut cov) = match aff {
            PointAffine::Constant(mu, sd) => {
                (mean_tf.map(|m| mu + sd * m), cov.map(|c| sd * c * sd))
            }
            PointAffine::PerPoint(_) => (
                Vector::from_fn(q, |i, _| {
                    let (o, s) = aff.at(i);
                    o + s * mean_tf[i]
                }),
                Matrix::from_fn(q, q, |i, j| aff.scale(i) * cov[(i, j)] * aff.scale(j)),
            ),
        };
        if let ObservationNoise::Explicit(v) = noise {
            if v.len() != q {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected {q} observation-noise values, got {}",
                    v.len()
                )));
            }
            for (i, &n) in v.iter().enumerate() {
                let s = aff.scale(i);
                cov[(i, i)] += s * s * n;
            }
        }
        Ok(Posterior::single_output(mean, cov)
            .with_sample_transform(SampleTransform::new(vec![samples], q)))
    }

    /// Posterior mean and variance at a single raw-space point `x` (length
    /// `d`) together with their gradients with respect to `x`, all in the
    /// original spaces: `(mean, variance, ∂mean/∂x, ∂variance/∂x)`.
    ///
    /// The variance is clamped at [`MIN_VARIANCE`] like BoTorch's
    /// `posterior.variance` (with a zero gradient when clamped). For a
    /// nonlinear outcome transform these are the moments of the
    /// `TransformedPosterior` (where BoTorch defines them), which BoTorch does
    /// not clamp.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `x.len() != d`,
    /// [`BotorchError::NotPsd`] if the training covariance cannot be
    /// factorized and [`BotorchError::Unsupported`] for a
    /// `TransformedPosterior` without moments (`Power`, `Bilog`).
    pub fn posterior_q1_with_grad(&self, x: &[f64]) -> Result<(f64, f64, Vector, Vector)> {
        let (mean, var, dmean, dvar) = self.posterior_q1_with_grad_unclamped(x)?;
        if var < MIN_VARIANCE && !self.has_transformed_posterior() {
            return Ok((mean, MIN_VARIANCE, dmean, Vector::zeros(self.dim())));
        }
        Ok((mean, var, dmean, dvar))
    }

    /// [`SingleTaskGP::posterior_q1_with_grad`] without clamping the
    /// variance. Posterior transforms and multi-output models build on this:
    /// BoTorch clamps the variance of the *transformed* posterior.
    ///
    /// # Errors
    /// See [`SingleTaskGP::posterior_q1_with_grad`].
    pub fn posterior_q1_with_grad_unclamped(
        &self,
        x: &[f64],
    ) -> Result<(f64, f64, Vector, Vector)> {
        if x.len() != self.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected a point with {} coordinates, got {}",
                self.dim(),
                x.len()
            )));
        }
        let (gaussian, samples) = if let Some((xt, tape)) =
            self.chain_tape(&Matrix::from_row_slice(1, x.len(), x))?
        {
            self.require_one_point_per_input()?;
            let (aff, samples) = self.outcome_parts(&xt)?;
            let xt: Vec<f64> = xt.row(0).iter().copied().collect();
            let (mean, var, dmean, dvar) = self.q1_transformed(&xt, None, aff.at(0))?;
            let pull = |g: &Vector| -> Result<Vector> {
                let raw = self.tape_backward(
                    &tape,
                    0..1,
                    &Matrix::from_row_slice(1, g.len(), g.as_slice()),
                )?;
                Ok(raw.row(0).transpose())
            };
            ((mean, var, pull(&dmean)?, pull(&dvar)?), samples)
        } else {
            let xt = match &self.input_transform {
                Some(tf) => tf.transform_point(x)?,
                None => x.to_vec(),
            };
            let (aff, samples) = self.outcome_parts(&Matrix::from_row_slice(1, xt.len(), &xt))?;
            let jac = self.legacy_jacobian();
            (self.q1_transformed(&xt, Some(&jac), aff.at(0))?, samples)
        };
        let Some(st) = SampleTransform::new(vec![samples], 1) else {
            return Ok(gaussian);
        };
        let (mean, var, dmean, dvar) = gaussian;
        let (m, v, a, b, c, e) = st.moments(0, 0, mean, var)?;
        Ok((m, v, &dmean * a + &dvar * b, &dmean * c + &dvar * e))
    }

    /// The posterior and its gradient at a transformed point `xt` (with the
    /// affine outcome un-transform `(mu, sd)` of the point); the gradients
    /// are with respect to `xt`, scaled by `jac` per dimension if given (the
    /// chain rule of `Normalize`).
    fn q1_transformed(
        &self,
        xt: &[f64],
        jac: Option<&Vector>,
        (mu, sd): (f64, f64),
    ) -> Result<(f64, f64, Vector, Vector)> {
        let cache = self.cache()?;
        let xs = cache
            .kernel
            .prepare(&Matrix::from_row_slice(1, xt.len(), xt))?;
        let (k, dk) = cache.kernel.cross_with_grad(&xs, 0, &cache.train_scaled);
        let mean_tf = self.constant() + k.dot(&cache.alpha);
        let dmean_tf = dk.tr_mul(&cache.alpha);
        let v = cache.chol.solve_lower_vec(&k);
        let (kdiag, dkdiag) = cache.kernel.diag_with_grad(&xs, 0);
        let var_tf = kdiag - v.dot(&v);
        let w = cache.chol.solve_lower_transpose_vec(&v);
        let mut dvar_tf = dk.tr_mul(&w) * -2.0;
        if let Some(g) = dkdiag {
            dvar_tf += g;
        }
        let mean = mu + sd * mean_tf;
        let var = sd * sd * var_tf;
        let (dmean, dvar) = match jac {
            Some(jac) => (
                dmean_tf.component_mul(jac) * sd,
                dvar_tf.component_mul(jac) * (sd * sd),
            ),
            None => (dmean_tf * sd, dvar_tf * (sd * sd)),
        };
        Ok((mean, var, dmean, dvar))
    }

    /// Joint posterior (without observation noise) at the `q × d` raw-space
    /// inputs `x` with the derivatives of the mean and covariance with respect
    /// to all `q·d` input coordinates (see [`JointPosteriorGrad`]). For a
    /// nonlinear outcome transform, these are the moments of the Gaussian
    /// base posterior (see [`SingleTaskGP::posterior`]).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a column mismatch and
    /// [`BotorchError::NotPsd`] if the training covariance cannot be factorized.
    pub fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        if let Some((xt, tape)) = self.chain_tape(x)? {
            // Derivatives with respect to the transformed points, then the
            // chain rule through the Jacobians of the transform.
            let (aff, _) = self.outcome_parts(&xt)?;
            let g = self.joint_with_grad_transformed(&xt, None, &aff)?;
            let chain = self.input_chain.as_ref().expect("chain present");
            let jacs = tape.jacobians_through(chain)?;
            let (q, d) = x.shape();
            let dt = xt.ncols();
            let k = tape.points_per_input();
            let len = g.mean.len();
            let mut dmean = vec![Vector::zeros(len); q * d];
            let mut dcov = vec![Matrix::zeros(len, len); q * d];
            for i in 0..q {
                for w in 0..k {
                    let r = i * k + w;
                    for l in 0..dt {
                        for j in 0..d {
                            let f = jacs[r][(l, j)];
                            if f != 0.0 {
                                dmean[i * d + j].axpy(f, &g.dmean[r * dt + l], 1.0);
                                dcov[i * d + j] += &g.dcov[r * dt + l] * f;
                            }
                        }
                    }
                }
            }
            return Ok(JointPosteriorGrad {
                mean: g.mean,
                covariance: g.covariance,
                dmean,
                dcov,
                num_outputs: 1,
            });
        }
        let xt = self.transform_inputs(x)?;
        let (aff, _) = self.outcome_parts(&xt)?;
        let jac = self.legacy_jacobian();
        self.joint_with_grad_transformed(&xt, Some(&jac), &aff)
    }

    /// The joint posterior at transformed points `xt` with derivatives with
    /// respect to `xt`, scaled by `jac` per dimension if given.
    fn joint_with_grad_transformed(
        &self,
        xt: &Matrix,
        jac: Option<&Vector>,
        aff: &PointAffine,
    ) -> Result<JointPosteriorGrad> {
        let cache = self.cache()?;
        let (q, d) = xt.shape();
        let n = self.num_train();
        let kernel = &cache.kernel;
        let xs = kernel.prepare(xt)?;

        // Test-train covariances and their derivatives for every test point.
        let mut kxs = Matrix::zeros(q, n);
        let mut dks = Vec::with_capacity(q);
        for p in 0..q {
            let (k, dk) = kernel.cross_with_grad(&xs, p, &cache.train_scaled);
            kxs.set_row(p, &k.transpose());
            dks.push(dk);
        }
        let mean_tf = (&kxs * &cache.alpha).add_scalar(self.constant());
        let mut v = kxs.transpose(); // n × q
        cache.chol.solve_lower_mut(&mut v);
        let mut cov_tf = kernel.symmetric(&xs);
        cov_tf.gemm_tr(-1.0, &v, &v, 1.0);
        let mut w = v; // becomes K⁻¹ K(X, x*) (n × q)
        cache.chol.solve_lower_transpose_mut(&mut w);

        let mut dmean = Vec::with_capacity(q * d);
        let mut dcov = Vec::with_capacity(q * d);
        for (p, dk) in dks.iter().enumerate() {
            // Derivatives of k(x_p, x_b) with respect to x_p for all test points b.
            let (_, dkss) = kernel.cross_with_grad(&xs, p, &xs);
            // dk_wᵀ: (d × q) = dkᵀ W
            let dk_w = dk.tr_mul(&w);
            let dk_alpha = dk.tr_mul(&cache.alpha);
            let sp = aff.scale(p);
            for m in 0..d {
                let scale = jac.map_or(1.0, |j| j[m]);
                let mut dm = Vector::zeros(q);
                dm[p] = sp * dk_alpha[m] * scale;
                dmean.push(dm);
                let mut dc = Matrix::zeros(q, q);
                for b in 0..q {
                    let g = (dkss[(b, m)] - dk_w[(m, b)]) * scale * sp * aff.scale(b);
                    if b == p {
                        dc[(p, p)] = 2.0 * g;
                    } else {
                        dc[(p, b)] = g;
                        dc[(b, p)] = g;
                    }
                }
                dcov.push(dc);
            }
        }
        let (mean, covariance) = match *aff {
            PointAffine::Constant(mu, sd) => {
                (mean_tf.map(|v| mu + sd * v), cov_tf.map(|c| sd * c * sd))
            }
            PointAffine::PerPoint(_) => (
                Vector::from_fn(q, |i, _| aff.offset(i) + aff.scale(i) * mean_tf[i]),
                Matrix::from_fn(q, q, |i, j| aff.scale(i) * cov_tf[(i, j)] * aff.scale(j)),
            ),
        };
        Ok(JointPosteriorGrad {
            mean,
            covariance,
            dmean,
            dcov,
            num_outputs: 1,
        })
    }

    /// Marginal posterior means and variances of the rows of `x` (each row
    /// evaluated as its own `q = 1` posterior, like a `b × 1 × d` batch in
    /// BoTorch), in the original spaces. Variances are clamped at
    /// [`MIN_VARIANCE`] (except the moments of a `TransformedPosterior`,
    /// which BoTorch does not clamp). The values are bitwise identical to
    /// those of [`SingleTaskGP::posterior_q1_with_grad`].
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a column mismatch,
    /// [`BotorchError::NotPsd`] if the training covariance cannot be
    /// factorized and [`BotorchError::Unsupported`] for a
    /// `TransformedPosterior` without moments.
    pub fn posterior_marginals(&self, x: &Matrix) -> Result<(Vector, Vector)> {
        let (mean, var) = self.posterior_marginals_unclamped(x)?;
        if self.has_transformed_posterior() {
            return Ok((mean, var));
        }
        Ok((mean, var.map(clamp_variance)))
    }

    /// [`SingleTaskGP::posterior_marginals`] without clamping the variances
    /// (bitwise identical to [`SingleTaskGP::posterior_q1_with_grad_unclamped`]).
    ///
    /// # Errors
    /// See [`SingleTaskGP::posterior_marginals`].
    pub fn posterior_marginals_unclamped(&self, x: &Matrix) -> Result<(Vector, Vector)> {
        self.marginals_impl(x, false)
    }

    /// The marginal means and variances of the posterior points of the rows
    /// of `x` as BoTorch's `posterior(X, observation_noise).mean` and
    /// `.variance` report them, without the clamp of Gaussian variances:
    /// with `observation_noise`, the likelihood noise is added in the
    /// model's outcome space (before a nonlinear un-transform).
    ///
    /// # Errors
    /// See [`SingleTaskGP::posterior_marginals`].
    pub fn marginals_with_noise(
        &self,
        x: &Matrix,
        observation_noise: bool,
    ) -> Result<(Vector, Vector)> {
        self.marginals_impl(x, observation_noise)
    }

    fn marginals_impl(&self, x: &Matrix, observation_noise: bool) -> Result<(Vector, Vector)> {
        // With a one-to-many input transform: the marginals of all `b·n_w`
        // transformed points.
        let xt = self.transform_inputs(x)?;
        let cache = self.cache()?;
        let b = xt.nrows();
        let xs = cache.kernel.prepare(&xt)?;
        let kxs = cache.kernel.cross(&xs, &cache.train_scaled);
        let (aff, samples) = self.outcome_parts(&xt)?;
        let mut mean = Vector::zeros(b);
        let mut var = Vector::zeros(b);
        for i in 0..b {
            let (mu, sd) = aff.at(i);
            let k = kxs.row(i).transpose();
            mean[i] = mu + sd * (self.constant() + k.dot(&cache.alpha));
            let v = cache.chol.solve_lower_vec(&k);
            var[i] = sd * sd * (cache.kernel.diag(&xs, i) - v.dot(&v));
        }
        if observation_noise {
            let noise_tf = self.observation_noise_transformed();
            for i in 0..b {
                let sd = aff.scale(i);
                var[i] += sd * sd * noise_tf;
            }
        }
        if let Some(st) = SampleTransform::new(vec![samples], b) {
            for i in 0..b {
                let (m, v, ..) = st.moments(0, i, mean[i], var[i])?;
                mean[i] = m;
                var[i] = v;
            }
        }
        Ok((mean, var))
    }

    /// Reverse-mode derivative of the joint posterior (without observation
    /// noise) at the raw-space inputs `x` (`N × d`): returns
    /// `∂s/∂x[p, ·]` for `p ∈ rows`, where
    /// `s = ⟨mean_adjoint, μ⟩ + Σ_ab covariance_adjoint[a,b] Σ[a,b]`
    /// (only the symmetric part of `covariance_adjoint` matters). For a
    /// nonlinear outcome transform, `μ` and `Σ` are those of the Gaussian
    /// base posterior (see [`SingleTaskGP::posterior`]).
    ///
    /// This costs one posterior evaluation plus `O(|rows|·(n + N)·d)` and is
    /// what Monte-Carlo acquisition functions use to backpropagate through
    /// posterior samples.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes and
    /// [`BotorchError::NotPsd`] if the training covariance cannot be factorized.
    pub fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        super::model::check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            1,
            self.points_per_input(),
        )?;
        if let Some((xt, tape)) = self.chain_tape(x)? {
            let k = tape.points_per_input();
            let rows_tf = rows.start * k..rows.end * k;
            let (aff, _) = self.outcome_parts(&xt)?;
            let g =
                self.vjp_transformed(&xt, rows_tf, mean_adjoint, covariance_adjoint, None, &aff)?;
            return self.tape_backward(&tape, rows, &g);
        }
        let xt = self.transform_inputs(x)?;
        let (aff, _) = self.outcome_parts(&xt)?;
        let jac = self.legacy_jacobian();
        self.vjp_transformed(
            &xt,
            rows,
            mean_adjoint,
            covariance_adjoint,
            Some(&jac),
            &aff,
        )
    }

    /// [`SingleTaskGP::joint_posterior_vjp`] at transformed points `xt`, with
    /// derivatives with respect to `xt` scaled by `jac` per dimension if given.
    fn vjp_transformed(
        &self,
        xt: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
        jac: Option<&Vector>,
        aff: &PointAffine,
    ) -> Result<Matrix> {
        let cache = self.cache()?;
        let d = xt.ncols();
        let kernel = &cache.kernel;
        let xs = kernel.prepare(xt)?;
        // Adjoints in the transformed outcome space (symmetrized covariance adjoint).
        let (mbar, abar) = match *aff {
            PointAffine::Constant(_, sd) => (
                mean_adjoint * sd,
                (covariance_adjoint + covariance_adjoint.transpose()) * (0.5 * sd * sd),
            ),
            PointAffine::PerPoint(_) => {
                let n = mean_adjoint.len();
                (
                    Vector::from_fn(n, |i, _| mean_adjoint[i] * aff.scale(i)),
                    Matrix::from_fn(n, n, |i, j| {
                        (covariance_adjoint[(i, j)] + covariance_adjoint[(j, i)])
                            * (0.5 * aff.scale(i) * aff.scale(j))
                    }),
                )
            }
        };
        // W = K⁻¹ K(X_train, x) (n × N).
        let kxs = kernel.cross(&xs, &cache.train_scaled);
        let mut w = kxs.transpose();
        cache.chol.solve_lower_mut(&mut w);
        cache.chol.solve_lower_transpose_mut(&mut w);
        let mut out = Matrix::zeros(rows.len(), d);
        for (r, p) in rows.enumerate() {
            let a_p = abar.column(p).into_owned();
            // ∂/∂x_p of ⟨mbar, μ⟩ + ⟨abar, Σ⟩ =
            //   Dk_pᵀ (mbar_p α - 2 W abar[:, p]) + 2 Σ_b abar[p, b] ∂₁k(x_p, x_b).
            let v = &cache.alpha * mbar[p] - (&w * &a_p) * 2.0;
            let (_, dk) = kernel.cross_with_grad(&xs, p, &cache.train_scaled);
            let mut g = dk.tr_mul(&v);
            let (_, dkk) = kernel.cross_with_grad(&xs, p, &xs);
            g += dkk.tr_mul(&a_p) * 2.0;
            for k in 0..d {
                out[(r, k)] = match jac {
                    Some(jac) => g[k] * jac[k],
                    None => g[k],
                };
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Posteriors jointly with a fixed point set
    // ------------------------------------------------------------------

    /// Caches what the posteriors of new points jointly with the fixed raw-
    /// space points `xb` (`n_b × d`, e.g. the baseline of a noisy acquisition
    /// function) need: see [`SingleTaskGP::cross_posterior`].
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a column mismatch,
    /// [`BotorchError::NotPsd`] if the training covariance cannot be
    /// factorized and [`BotorchError::Unsupported`] for a nonlinear or
    /// input-dependent outcome transform.
    pub fn baseline_cache(&self, xb: &Matrix) -> Result<BaselineCache> {
        self.require_plain_outcome("posteriors jointly with cached points")?;
        let xt = self.transform_inputs(xb)?;
        let cache = self.cache()?;
        let scaled = cache.kernel.prepare(&xt)?;
        let mut v = cache.kernel.cross(&scaled, &cache.train_scaled).transpose();
        cache.chol.solve_lower_mut(&mut v);
        let mut w = v.clone();
        cache.chol.solve_lower_transpose_mut(&mut w);
        Ok(BaselineCache {
            scaled,
            v,
            w,
            n: xt.nrows(),
        })
    }

    /// The blocks of the joint posterior (without observation noise) of the
    /// fixed points of `baseline` and the raw-space points `x` (`q × d`) that
    /// involve `x`: the mean at `x`, the cross-covariance `Σ(x, B)` and the
    /// covariance `Σ(x, x)` (original outcome space). These are the blocks
    /// of [`SingleTaskGP::posterior`] over `[B; x]`, computed in
    /// `O(n² q)` instead of `O(n² (n_b + q))` for `n` training points.
    ///
    /// # Errors
    /// See [`SingleTaskGP::baseline_cache`].
    pub fn cross_posterior(&self, baseline: &BaselineCache, x: &Matrix) -> Result<CrossPosterior> {
        let (mu, sd) = self.outcome_affine()?;
        let xt = self.transform_inputs(x)?;
        let cache = self.cache()?;
        let xs = cache.kernel.prepare(&xt)?;
        let kxs = cache.kernel.cross(&xs, &cache.train_scaled);
        let mean_tf = (&kxs * &cache.alpha).add_scalar(self.constant());
        let mut v = kxs.transpose();
        cache.chol.solve_lower_mut(&mut v);
        let mut cross = cache.kernel.cross(&xs, &baseline.scaled);
        cross.gemm_tr(-1.0, &v, &baseline.v, 1.0);
        let mut cov = cache.kernel.symmetric(&xs);
        cov.gemm_tr(-1.0, &v, &v, 1.0);
        Ok(CrossPosterior {
            mean: mean_tf.map(|m| mu + sd * m),
            cross: cross.map(|c| sd * c * sd),
            covariance: cov.map(|c| sd * c * sd),
        })
    }

    /// Reverse-mode derivative of [`SingleTaskGP::cross_posterior`]: for the
    /// scalar `s = ⟨mean_adjoint, μ(x)⟩ + ⟨cross_adjoint, Σ(x, B)⟩ +
    /// ⟨covariance_adjoint, Σ(x, x)⟩` (only the symmetric part of
    /// `covariance_adjoint` matters), returns `∂s/∂x` (`q × d`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes; see also
    /// [`SingleTaskGP::baseline_cache`].
    pub fn cross_posterior_vjp(
        &self,
        baseline: &BaselineCache,
        x: &Matrix,
        mean_adjoint: &Vector,
        cross_adjoint: &Matrix,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        let q = x.nrows() * self.points_per_input();
        if mean_adjoint.len() != q
            || cross_adjoint.shape() != (q, baseline.n)
            || covariance_adjoint.shape() != (q, q)
        {
            return Err(BotorchError::DimensionMismatch(format!(
                "adjoints of {q} points and a baseline of {} points must have shapes {q}, \
                 {q}x{} and {q}x{q}",
                baseline.n, baseline.n
            )));
        }
        if let Some((xt, tape)) = self.chain_tape(x)? {
            let g = self.cross_vjp_transformed(
                baseline,
                &xt,
                mean_adjoint,
                cross_adjoint,
                covariance_adjoint,
                None,
            )?;
            return self.tape_backward(&tape, 0..x.nrows(), &g);
        }
        let xt = self.transform_inputs(x)?;
        let jac = self.legacy_jacobian();
        self.cross_vjp_transformed(
            baseline,
            &xt,
            mean_adjoint,
            cross_adjoint,
            covariance_adjoint,
            Some(&jac),
        )
    }

    /// [`SingleTaskGP::cross_posterior_vjp`] at transformed points `xt`, with
    /// derivatives with respect to `xt` scaled by `jac` if given.
    fn cross_vjp_transformed(
        &self,
        baseline: &BaselineCache,
        xt: &Matrix,
        mean_adjoint: &Vector,
        cross_adjoint: &Matrix,
        covariance_adjoint: &Matrix,
        jac: Option<&Vector>,
    ) -> Result<Matrix> {
        let q = xt.nrows();
        let cache = self.cache()?;
        let d = xt.ncols();
        let kernel = &cache.kernel;
        let xs = kernel.prepare(xt)?;
        let (_, sd) = self.outcome_affine()?;
        let mbar = mean_adjoint * sd;
        let cbar = cross_adjoint * (sd * sd);
        let dbar = (covariance_adjoint + covariance_adjoint.transpose()) * (0.5 * sd * sd);
        // W = K⁻¹ K(X_train, x) (n × q).
        let mut w = kernel.cross(&xs, &cache.train_scaled).transpose();
        cache.chol.solve_lower_mut(&mut w);
        cache.chol.solve_lower_transpose_mut(&mut w);
        let mut out = Matrix::zeros(q, d);
        for p in 0..q {
            let d_p = dbar.column(p).into_owned();
            let c_p = cbar.row(p).transpose();
            // ∂/∂x_p = Dk_pᵀ (mbar_p α - 2 W dbar[:, p] - W_B cbar[p, :]ᵀ)
            //        + 2 Σ_b dbar[p, b] ∂₁k(x_p, x_b) + Σ_b cbar[p, b] ∂₁k(x_p, B_b).
            let v = &cache.alpha * mbar[p] - (&w * &d_p) * 2.0 - &baseline.w * &c_p;
            let (_, dk) = kernel.cross_with_grad(&xs, p, &cache.train_scaled);
            let mut g = dk.tr_mul(&v);
            let (_, dkk) = kernel.cross_with_grad(&xs, p, &xs);
            g += dkk.tr_mul(&d_p) * 2.0;
            let (_, dkb) = kernel.cross_with_grad(&xs, p, &baseline.scaled);
            g += dkb.tr_mul(&c_p);
            for k in 0..d {
                out[(p, k)] = match jac {
                    Some(jac) => g[k] * jac[k],
                    None => g[k],
                };
            }
        }
        Ok(out)
    }
}

/// The observation noise of [`SingleTaskGP::posterior_impl`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum ObservationNoise<'a> {
    /// None.
    None,
    /// The likelihood noise (the mean training noise for fixed noise), added
    /// in the transformed outcome space.
    Likelihood,
    /// Given variances per point (transformed outcome space).
    Explicit(&'a [f64]),
}

/// Quantities of a fixed point set `B` cached by
/// [`SingleTaskGP::baseline_cache`]: its scaled transformed points,
/// `V_B = L⁻¹ K(X_train, B)` and `W_B = K⁻¹ K(X_train, B)`.
#[derive(Clone, Debug)]
pub struct BaselineCache {
    scaled: Prepared,
    v: Matrix,
    w: Matrix,
    n: usize,
}

impl BaselineCache {
    /// Number of fixed points `n_b`.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the fixed point set is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// The blocks of a joint posterior over `[B; x]` that involve the new points
/// `x` (see [`SingleTaskGP::cross_posterior`]).
#[derive(Clone, Debug, PartialEq)]
pub struct CrossPosterior {
    /// Posterior mean at `x` (`q`).
    pub mean: Vector,
    /// Cross-covariance `Σ(x, B)` (`q × n_b`).
    pub cross: Matrix,
    /// Covariance `Σ(x, x)` (`q × q`).
    pub covariance: Matrix,
}

impl Model for SingleTaskGP {
    fn dim(&self) -> usize {
        SingleTaskGP::dim(self)
    }

    fn num_outputs(&self) -> usize {
        SingleTaskGP::num_outputs(self)
    }

    fn output_gps(&self) -> Option<Vec<&SingleTaskGP>> {
        // Consumers of the output GPs compute Gaussian posteriors with a
        // constant un-standardization.
        self.has_plain_outcome().then(|| vec![self])
    }

    fn points_per_input(&self) -> usize {
        SingleTaskGP::points_per_input(self)
    }

    fn transformed_outputs(&self) -> Vec<bool> {
        vec![self.has_transformed_posterior()]
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        SingleTaskGP::posterior(self, x, observation_noise)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.require_one_point_per_input()?;
        let (mean, var) = self.posterior_marginals_unclamped(x)?;
        Ok(PointPosteriors::single_output(&mean, &var))
    }

    fn point_posterior_with_grad(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        let (mean, var, dmean, dvar) = self.posterior_q1_with_grad_unclamped(x)?;
        Ok(JointPosteriorGrad {
            mean: Vector::from_element(1, mean),
            covariance: Matrix::from_element(1, 1, var),
            dmean: dmean.iter().map(|&g| Vector::from_element(1, g)).collect(),
            dcov: dvar
                .iter()
                .map(|&g| Matrix::from_element(1, 1, g))
                .collect(),
            num_outputs: 1,
        })
    }

    fn posterior_marginals(&self, x: &Matrix) -> Result<(Vector, Vector)> {
        SingleTaskGP::posterior_marginals(self, x)
    }

    fn posterior_q1_with_grad(&self, x: &[f64]) -> Result<(f64, f64, Vector, Vector)> {
        SingleTaskGP::posterior_q1_with_grad(self, x)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        SingleTaskGP::joint_posterior_with_grad(self, x)
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        SingleTaskGP::joint_posterior_vjp(self, x, rows, mean_adjoint, covariance_adjoint)
    }
}
