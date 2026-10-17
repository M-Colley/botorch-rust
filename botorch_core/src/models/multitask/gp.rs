//! [`MultiTaskGP`]: BoTorch's `MultiTaskGP` (`botorch.models.multitask`), a
//! multi-task exact GP with an ICM (intrinsic coregionalization model) kernel
//! on long-format data (Bonilla et al., 2007; Swersky et al., 2013).
//!
//! # The model
//!
//! The training inputs `n × (d + 1)` contain a task column (see
//! [`TaskLayout`]); the other `d` columns are the data features. For `K`
//! tasks with indices `t`:
//!
//! * the covariance is BoTorch's `covar_module = data_covar_module *
//!   task_covar_module`: a data kernel (any kernel tree, by default BoTorch's
//!   dimension-scaled RBF kernel, on the non-task columns: `active_dims =
//!   _base_idxr` unless the kernel has its own) times a
//!   `PositiveIndexKernel(num_tasks=K, rank)` on the task column with the
//!   task prior `task_covar_prior` (default `BetaPrior(2.5, 1.5)` on the task
//!   correlations). As GPyTorch's `k1 * k2` flattens a `ProductKernel` data
//!   kernel, its sub-kernels become `covar_module.kernels.{0..}` and the task
//!   kernel the last one;
//! * the mean is a `MultitaskMean` of per-task `ConstantMean`s
//!   (`mean_module.base_means.{t}.raw_constant`), or one shared `ConstantMean`;
//! * the noise is per task (`HadamardGaussianLikelihood` with BoTorch's
//!   `LogNormal(-4, 1)` prior and `GreaterThan(1e-4)` constraint,
//!   `likelihood.noise_covar.raw_noise` of shape `K`), shared by all tasks
//!   (a `GaussianLikelihood`), or fixed (`train_Yvar`,
//!   `FixedNoiseGaussianLikelihood`);
//! * a `Standardize` outcome transform over all training rows (BoTorch's
//!   default) and an optional `Normalize` input transform (on all columns or
//!   on the columns `indices`), applied to the full inputs before the task
//!   values are read.
//!
//! Raw task values are mapped to task indices (the positions in the sorted
//! `all_tasks`) after the input transform, with BoTorch's validation
//! (`validate_task_values`).
//!
//! The posterior at `X` without the task column (`q × d`) is over the output
//! tasks (BoTorch's `_make_X_full`: the rows of `X` with every output task, in
//! BoTorch's output-major layout), with correlated outputs. With the task
//! column (`q × (d + 1)`, [`MultiTaskTaskInputs`]) it is single-output over
//! the given tasks. Observation noise adds the noise level of each point's
//! task (for fixed noise, the mean training noise of the task, as BoTorch).
//!
//! # Tasks without data
//!
//! As in BoTorch, [`MultiTaskGP::eval_unobserved_task_factors`] (BoTorch's
//! `eval()`, called by every posterior) sets the rows of the task kernel's
//! `covar_factor` that belong to tasks without training data to the mean of
//! the observed tasks' rows.
//!
//! # Differences from BoTorch
//!
//! * Per-task noise levels are looked up by task index, not raw task value
//!   (BoTorch's `HadamardGaussianLikelihood` reads the raw task column, so
//!   tasks whose value is not their index get another task's noise or none,
//!   meta-pytorch/botorch#3396); identical for task values `0..K-1`.
//! * `active_dims` of a `ProductKernel` data kernel apply to its sub-kernels
//!   (BoTorch sets them on the product, which GPyTorch's `*` then flattens
//!   away, so the sub-kernels also see the task column), and `eval()` finds
//!   the task kernel as the last sub-kernel (BoTorch uses `kernels[1]`).
//! * An `LKJCovariancePrior` task prior acts on the task covariance matrix
//!   (BoTorch evaluates it on the vector of correlations and fails).
//! * For fixed noise, the observation noise of a task without training data
//!   is the mean training noise of all tasks (BoTorch: NaN).

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use rand::rngs::StdRng;

use crate::constraints::Constraint;
use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::kernels::eval::TreePrep;
use crate::kernels::tree::sample_tree_priors;
use crate::kernels::{
    get_covar_module_with_dim_scaled_prior, KernelNode, KernelSpec, KernelType, MatrixPrior,
    TreeKernel,
};
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::models::gp::{
    get_gaussian_likelihood_with_lognormal_prior, GaussianLikelihoodSpec, MIN_FIXED_NOISE,
};
use crate::models::model::check_vjp_shapes_expanded;
use crate::models::multitask::TaskLayout;
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
use crate::priors::Prior;
use crate::transforms::{Normalize, OutcomeTransformChoice, Standardize};
use crate::{Matrix, Vector};

/// `log(2π)`.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// The prior of the task covariance (BoTorch's `task_covar_prior`).
#[derive(Clone, Debug, PartialEq, Default)]
pub enum TaskCovarPrior {
    /// BoTorch's default: `BetaPrior(2.5, 1.5)` on the task correlations.
    #[default]
    Default,
    /// No prior (`task_covar_prior=None`).
    None,
    /// A univariate prior on the task correlations or an LKJ covariance prior
    /// on the task covariance matrix.
    Prior(MatrixPrior),
}

/// The mean module of a [`MultiTaskGP`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MultiTaskMean {
    /// BoTorch's default `MultitaskMean(ConstantMean(), num_tasks)`: one
    /// constant per task.
    #[default]
    PerTask,
    /// A `ConstantMean` shared by all tasks (`mean_module=ConstantMean()`).
    Shared,
}

/// A likelihood with learned noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MultiTaskNoise {
    /// `HadamardGaussianLikelihood`: one noise level per task (BoTorch's
    /// default without `train_Yvar`).
    PerTask(GaussianLikelihoodSpec),
    /// `GaussianLikelihood`: one noise level for all tasks.
    Shared(GaussianLikelihoodSpec),
}

impl MultiTaskNoise {
    fn spec(&self) -> &GaussianLikelihoodSpec {
        match self {
            MultiTaskNoise::PerTask(s) | MultiTaskNoise::Shared(s) => s,
        }
    }
}

/// Options of [`MultiTaskGP::new`] (BoTorch's keyword arguments).
#[derive(Clone, Debug, PartialEq)]
pub struct MultiTaskGPOptions {
    /// Observed noise variances (`n × 1`): a fixed-noise likelihood.
    pub train_yvar: Option<Matrix>,
    /// The mean module.
    pub mean: MultiTaskMean,
    /// The data kernel (BoTorch's `covar_module`; default: the
    /// dimension-scaled RBF kernel). Without `active_dims` of its own, it acts
    /// on the non-task columns.
    pub covar_module: Option<KernelNode>,
    /// The `active_dims` of a composite (`AdditiveKernel` / `ProductKernel`)
    /// data kernel, which a [`KernelNode`] cannot hold (GPyTorch sets them on
    /// the composite object); `None`: the non-task columns.
    pub covar_module_active_dims: Option<Vec<usize>>,
    /// The likelihood (`None`: per-task noise, or fixed noise with
    /// `train_yvar`).
    pub likelihood: Option<MultiTaskNoise>,
    /// The task prior.
    pub task_covar_prior: TaskCovarPrior,
    /// The task kernel itself (an `IndexKernel` or `PositiveIndexKernel` on
    /// the `K` tasks with `active_dims = [task_feature]`), instead of the
    /// default `PositiveIndexKernel(K, rank)` with `task_covar_prior` (e.g.
    /// with initial values drawn as GPyTorch draws them).
    pub task_covar_module: Option<KernelNode>,
    /// The output tasks (task values; `None`: all tasks).
    pub output_tasks: Option<Vec<i64>>,
    /// The rank of the task covariance (`None`: the number of tasks).
    pub rank: Option<usize>,
    /// All tasks, including tasks without data (`None`: the tasks of the data).
    pub all_tasks: Option<Vec<i64>>,
    /// The outcome transform (BoTorch default: `Standardize`).
    pub outcome_transform: OutcomeTransformChoice,
    /// A `Normalize` input transform of the columns `input_transform_indices`
    /// (all `d + 1` columns if `None`); its dimension is the number of
    /// transformed columns.
    pub input_transform: Option<Normalize>,
    /// The columns of the full inputs that `input_transform` normalizes.
    pub input_transform_indices: Option<Vec<usize>>,
    /// BoTorch's `validate_task_values`.
    pub validate_task_values: bool,
}

impl Default for MultiTaskGPOptions {
    fn default() -> Self {
        Self {
            train_yvar: None,
            mean: MultiTaskMean::PerTask,
            covar_module: None,
            covar_module_active_dims: None,
            likelihood: None,
            task_covar_prior: TaskCovarPrior::Default,
            task_covar_module: None,
            output_tasks: None,
            rank: None,
            all_tasks: None,
            outcome_transform: OutcomeTransformChoice::Default,
            input_transform: None,
            input_transform_indices: None,
            validate_task_values: true,
        }
    }
}

/// The noise model of a [`MultiTaskGP`].
#[derive(Clone, Debug, PartialEq)]
enum Noise {
    Learned(MultiTaskNoise),
    /// Noise variances of the training rows (transformed space, clamped at
    /// [`MIN_FIXED_NOISE`]).
    Fixed(Vector),
}

/// BoTorch's `get_covar_module_with_dim_scaled_prior` (or another
/// [`KernelSpec`]) as a kernel tree.
pub fn kernel_spec_tree(spec: &KernelSpec) -> Result<KernelNode> {
    let mut base = KernelNode::stationary(spec.kind, Some(spec.ard_num_dims))?
        .with_constraint("raw_lengthscale", spec.lengthscale_constraint)?;
    if let Some(p) = spec.lengthscale_prior {
        base = base.with_prior("raw_lengthscale", p)?;
    }
    let Some(scale) = spec.scale else {
        return Ok(base);
    };
    let mut node =
        KernelNode::scale(base).with_constraint("raw_outputscale", scale.outputscale_constraint)?;
    if let Some(p) = scale.outputscale_prior {
        node = node.with_prior("raw_outputscale", p)?;
    }
    Ok(node)
}

/// BoTorch's default task prior `BetaPrior(2.5, 1.5)` (float32 parameters).
pub fn default_task_covar_prior() -> MatrixPrior {
    MatrixPrior::Elementwise(
        Prior::beta(2.5, 1.5)
            .expect("valid Beta parameters")
            .rounded_to_f32(),
    )
}

/// BoTorch's `data_covar_module * task_covar_module`: the product of the data
/// kernel (with its `active_dims`, see [`MultiTaskGPOptions`]) and the task
/// kernel, a `ProductKernel` data kernel flattened like GPyTorch's `*`.
/// Returns the tree and the position of the task kernel among its children.
fn assemble_covar(
    data: KernelNode,
    data_dims: Option<Vec<usize>>,
    base_idxr: &[usize],
    task: KernelNode,
) -> Result<(KernelNode, usize)> {
    let composite = matches!(data.kind(), KernelType::Additive | KernelType::Product);
    let data = if composite {
        let dims = data_dims.unwrap_or_else(|| base_idxr.to_vec());
        data.with_parent_active_dims(&dims)?
    } else if data.active_dims().is_none() {
        data.with_active_dims(data_dims.unwrap_or_else(|| base_idxr.to_vec()))?
    } else {
        data
    };
    let mut kids = match data.kind() {
        KernelType::Product => data.children().to_vec(),
        _ => vec![data],
    };
    let index = kids.len();
    kids.push(task);
    Ok((KernelNode::product(kids)?, index))
}

/// Quantities derived from the hyperparameters and cached for prediction.
#[derive(Debug)]
struct MtCache {
    kernel: TreeKernel,
    train_prep: TreePrep,
    chol: Arc<CholeskyFactor>,
    /// `K⁻¹ (y - μ)` (transformed space).
    alpha: Vector,
}

/// Points of the joint posterior: the kernel inputs (`N × (d + 1)`,
/// transformed, with task indices in the task column), their task indices
/// and the rows of the user's inputs they come from.
#[derive(Clone, Debug)]
struct AugPoints {
    kx: Matrix,
    tasks: Vec<usize>,
    src: Vec<usize>,
    /// For every column of the user's inputs: the kernel column and the
    /// chain-rule factor of the input transform (`None`: the task column).
    cols: Vec<Option<(usize, f64)>>,
    /// Number of user input rows.
    num_inputs: usize,
}

/// BoTorch's `MultiTaskGP` (see the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct MultiTaskGP {
    /// Training inputs in the original space (`n × (d + 1)`).
    train_x: Matrix,
    /// Training targets in the original space (`n × 1`).
    train_y: Matrix,
    train_yvar: Option<Matrix>,
    layout: TaskLayout,
    validate: bool,
    input_transform: Option<Normalize>,
    input_indices: Option<Vec<usize>>,
    outcome_transform: Option<Standardize>,
    /// Kernel inputs of the training rows.
    train_kx: Matrix,
    train_tasks: Vec<usize>,
    y_tf: Vector,
    mean: MultiTaskMean,
    noise: Noise,
    covar: KernelNode,
    task_kernel: usize,
    /// The number of rows added by conditioning with explicit noise of a
    /// model with learned noise: their noise variances (transformed space).
    point_noise: Vec<Option<f64>>,
    raw: Vector,
    cache: OnceLock<std::result::Result<Arc<MtCache>, BotorchError>>,
}

fn check_no_nan(m: &Matrix, what: &str) -> Result<()> {
    if m.iter().any(|v| v.is_nan()) {
        return Err(BotorchError::InputData(format!(
            "{what} contains NaN values."
        )));
    }
    Ok(())
}

impl MultiTaskGP {
    /// Builds the model from `train_x` (`n × (d + 1)`, with the task values in
    /// column `task_feature`) and `train_y` (`n × 1`); see
    /// [`MultiTaskGPOptions`].
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
    /// [`BotorchError::InputData`] for NaN values or negative noise
    /// variances, the errors of [`TaskLayout::new`] and
    /// [`BotorchError::InvalidArgument`] for an invalid rank or kernel.
    pub fn new(
        train_x: Matrix,
        train_y: Matrix,
        task_feature: isize,
        options: MultiTaskGPOptions,
    ) -> Result<Self> {
        let n = train_x.nrows();
        if n == 0 {
            return Err(BotorchError::InputData(
                "train_X must have at least one row".into(),
            ));
        }
        if train_y.ncols() != 1 {
            return Err(BotorchError::DimensionMismatch(format!(
                "MultiTaskGP requires a single-output train_Y (n x 1), got {} columns",
                train_y.ncols()
            )));
        }
        if train_y.nrows() != n {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X and Y to have the same number of rows (got X with {n} and Y with {}).",
                train_y.nrows()
            )));
        }
        check_no_nan(&train_x, "train_X")?;
        check_no_nan(&train_y, "train_Y")?;
        if let Some(yvar) = &options.train_yvar {
            if yvar.shape() != train_y.shape() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected Yvar with the same shape as Y {:?}, got {:?}.",
                    train_y.shape(),
                    yvar.shape()
                )));
            }
            check_no_nan(yvar, "train_Yvar")?;
            if yvar.iter().any(|&v| v < 0.0) {
                return Err(BotorchError::InputData(
                    "Input data contains negative variances.".into(),
                ));
            }
            if options.likelihood.is_some() {
                return Err(BotorchError::InvalidArgument(
                    "a likelihood with learned noise cannot be combined with train_Yvar".into(),
                ));
            }
        }

        // Input transform (fitted on the training inputs, train mode).
        let (input_transform, input_indices) = fit_input_transform(
            options.input_transform,
            options.input_transform_indices,
            &train_x,
        )?;
        let train_x_tf =
            apply_input_transform(input_transform.as_ref(), input_indices.as_deref(), &train_x)?;

        let layout = TaskLayout::new(
            &train_x_tf,
            task_feature,
            options.output_tasks.as_deref(),
            options.all_tasks.as_deref(),
        )?;
        let d = layout.num_non_task_features();
        let k_tasks = layout.num_tasks();
        let tf = layout.task_feature();

        // Outcome transform.
        let mut outcome_transform = match options.outcome_transform {
            OutcomeTransformChoice::Default => Some(Standardize::new()),
            OutcomeTransformChoice::None => None,
            OutcomeTransformChoice::Standardize(s) => Some(s),
            OutcomeTransformChoice::Chains(_) => {
                return Err(BotorchError::Unsupported(
                    "MultiTaskGP supports the Standardize outcome transform only".into(),
                ))
            }
        };
        let (y_tf, yvar_tf) = match outcome_transform.as_mut() {
            Some(st) => st.fit_transform(&train_y, options.train_yvar.as_ref())?,
            None => (train_y.clone(), options.train_yvar.clone()),
        };
        check_no_nan(&y_tf, "transformed train_Y")?;
        let noise = match yvar_tf {
            Some(v) => Noise::Fixed(Vector::from_iterator(
                n,
                v.column(0).iter().map(|&s| s.max(MIN_FIXED_NOISE)),
            )),
            None => Noise::Learned(options.likelihood.unwrap_or_else(|| {
                MultiTaskNoise::PerTask(get_gaussian_likelihood_with_lognormal_prior())
            })),
        };

        // Kernel inputs of the training rows.
        let (train_kx, train_tasks) =
            map_task_column(&layout, &train_x_tf, options.validate_task_values)?;

        // Covariance module.
        let task = match options.task_covar_module {
            Some(task) => {
                let ok = match task.kind() {
                    KernelType::PositiveIndex { num_tasks, .. }
                    | KernelType::Index { num_tasks, .. } => *num_tasks == k_tasks,
                    _ => false,
                };
                if !ok || task.active_dims() != Some(&[tf][..]) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "the task kernel must be an index kernel on the {k_tasks} tasks with \
                         active_dims = [{tf}]"
                    )));
                }
                task
            }
            None => {
                let rank = options.rank.unwrap_or(k_tasks);
                let mut task = KernelNode::positive_index(k_tasks, rank, 0, true)?;
                match options.task_covar_prior {
                    TaskCovarPrior::Default => {
                        task = task.with_task_prior(default_task_covar_prior())?
                    }
                    TaskCovarPrior::None => {}
                    TaskCovarPrior::Prior(p) => task = task.with_task_prior(p)?,
                }
                task.with_active_dims(vec![tf])?
            }
        };
        let base_idxr: Vec<usize> = (0..=d).filter(|&c| c != tf).collect();
        let data = match options.covar_module {
            Some(node) => node,
            None => kernel_spec_tree(&get_covar_module_with_dim_scaled_prior(d, true))?,
        };
        let (covar, task_kernel) =
            assemble_covar(data, options.covar_module_active_dims, &base_idxr, task)?;
        covar.validate(d + 1)?;

        let n_noise = match &noise {
            Noise::Learned(MultiTaskNoise::PerTask(_)) => k_tasks,
            Noise::Learned(MultiTaskNoise::Shared(_)) => 1,
            Noise::Fixed(_) => 0,
        };
        let n_mean = match options.mean {
            MultiTaskMean::PerTask => k_tasks,
            MultiTaskMean::Shared => 1,
        };
        let mut raw = Vec::with_capacity(n_noise + n_mean + covar.num_raw());
        if let Noise::Learned(l) = &noise {
            raw.extend(std::iter::repeat_n(l.spec().initial_raw_noise(), n_noise));
        }
        raw.extend(std::iter::repeat_n(0.0, n_mean));
        raw.extend(covar.initial_raw());
        Ok(Self {
            train_x,
            train_y,
            train_yvar: options.train_yvar,
            layout,
            validate: options.validate_task_values,
            input_transform,
            input_indices,
            outcome_transform,
            train_kx,
            train_tasks,
            y_tf: Vector::from_column_slice(y_tf.column(0).as_slice()),
            mean: options.mean,
            noise,
            covar,
            task_kernel,
            point_noise: Vec::new(),
            raw: Vector::from_vec(raw),
            cache: OnceLock::new(),
        })
    }

    // ------------------------------------------------------------------
    // Data and configuration
    // ------------------------------------------------------------------

    /// Training inputs including the task column, in the original space
    /// (`n × (d + 1)`).
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

    /// The training inputs after the input transform (BoTorch's
    /// `transform_inputs(train_X)`).
    ///
    /// # Errors
    /// See [`Normalize::transform`].
    pub fn transformed_train_inputs(&self) -> Result<Matrix> {
        self.transform_full(&self.train_x)
    }

    /// The training targets after the outcome transform (`model.train_targets`).
    pub fn transformed_train_targets(&self) -> &Vector {
        &self.y_tf
    }

    /// The task indices of the training rows.
    pub fn train_task_indices(&self) -> &[usize] {
        &self.train_tasks
    }

    /// The task structure.
    pub fn task_layout(&self) -> &TaskLayout {
        &self.layout
    }

    /// The number of non-task features `d`.
    pub fn dim(&self) -> usize {
        self.layout.num_non_task_features()
    }

    /// Number of training points `n`.
    pub fn num_train(&self) -> usize {
        self.train_x.nrows()
    }

    /// The number of tasks `K`.
    pub fn num_tasks(&self) -> usize {
        self.layout.num_tasks()
    }

    /// Whether task values are validated (BoTorch's `validate_task_values`).
    pub fn validate_task_values(&self) -> bool {
        self.validate
    }

    /// The full covariance module (`covar_module`, a `ProductKernel`).
    pub fn covar_module(&self) -> &KernelNode {
        &self.covar
    }

    /// The position of the task kernel among the sub-kernels of
    /// `covar_module`.
    pub fn task_kernel_index(&self) -> usize {
        self.task_kernel
    }

    /// The task kernel (`PositiveIndexKernel`).
    pub fn task_covar_module(&self) -> &KernelNode {
        &self.covar.children()[self.task_kernel]
    }

    /// The rank of the task covariance.
    pub fn rank(&self) -> usize {
        match self.task_covar_module().kind() {
            KernelType::PositiveIndex { rank, .. } | KernelType::Index { rank, .. } => *rank,
            _ => unreachable!("the task kernel is an index kernel"),
        }
    }

    /// The mean module.
    pub fn mean_module(&self) -> MultiTaskMean {
        self.mean
    }

    /// The likelihood with learned noise, `None` for fixed noise.
    pub fn likelihood(&self) -> Option<&MultiTaskNoise> {
        match &self.noise {
            Noise::Learned(l) => Some(l),
            Noise::Fixed(_) => None,
        }
    }

    /// The fitted input transform, if any, and the columns it normalizes
    /// (`None`: all columns).
    pub fn input_transform(&self) -> Option<(&Normalize, Option<&[usize]>)> {
        self.input_transform
            .as_ref()
            .map(|t| (t, self.input_indices.as_deref()))
    }

    /// The fitted outcome transform, if any.
    pub fn outcome_transform(&self) -> Option<&Standardize> {
        self.outcome_transform.as_ref()
    }

    /// A copy whose posterior is over other output tasks (task values).
    ///
    /// # Errors
    /// [`BotorchError::Runtime`] if a value is not a task.
    pub fn with_output_tasks(&self, output_tasks: &[i64]) -> Result<Self> {
        let layout = self.layout.with_output_tasks(output_tasks)?;
        Ok(Self {
            layout,
            ..self.clone()
        })
    }

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------

    fn num_noise_params(&self) -> usize {
        match &self.noise {
            Noise::Learned(MultiTaskNoise::PerTask(_)) => self.num_tasks(),
            Noise::Learned(MultiTaskNoise::Shared(_)) => 1,
            Noise::Fixed(_) => 0,
        }
    }

    fn num_mean_params(&self) -> usize {
        match self.mean {
            MultiTaskMean::PerTask => self.num_tasks(),
            MultiTaskMean::Shared => 1,
        }
    }

    fn kernel_range(&self) -> Range<usize> {
        let start = self.num_noise_params() + self.num_mean_params();
        start..start + self.covar.num_raw()
    }

    /// The raw hyperparameters in BoTorch's `named_parameters()` order (also
    /// the order of BoTorch's optimizer).
    pub fn raw_parameters(&self) -> &Vector {
        &self.raw
    }

    /// Sets the raw hyperparameters from a flat vector.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of values and
    /// [`BotorchError::InvalidArgument`] for non-finite values or negative
    /// noise variances.
    pub fn set_raw_parameters(&mut self, values: &[f64]) -> Result<()> {
        self.check_raw(values)?;
        self.raw = Vector::from_column_slice(values);
        self.cache = OnceLock::new();
        Ok(())
    }

    fn check_raw(&self, values: &[f64]) -> Result<()> {
        if values.len() != self.raw.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                self.raw.len(),
                values.len()
            )));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "hyperparameters must be finite".into(),
            ));
        }
        if let Noise::Learned(l) = &self.noise {
            for &r in &values[..self.num_noise_params()] {
                let v = l.spec().noise_constraint.transform(r);
                if v.is_nan() || v < 0.0 {
                    return Err(BotorchError::InvalidArgument(format!(
                        "noise must be non-negative, got {v}"
                    )));
                }
            }
        }
        TreeKernel::build(&self.covar, &values[self.kernel_range()], self.dim() + 1)?;
        Ok(())
    }

    /// BoTorch parameter names and shapes in flat-vector order.
    pub fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::new();
        match &self.noise {
            Noise::Learned(MultiTaskNoise::PerTask(_)) => out.push((
                "likelihood.noise_covar.raw_noise".to_string(),
                vec![self.num_tasks()],
            )),
            Noise::Learned(MultiTaskNoise::Shared(_)) => {
                out.push(("likelihood.noise_covar.raw_noise".to_string(), vec![1]));
            }
            Noise::Fixed(_) => {}
        }
        match self.mean {
            MultiTaskMean::PerTask => {
                for t in 0..self.num_tasks() {
                    out.push((format!("mean_module.base_means.{t}.raw_constant"), vec![]));
                }
            }
            MultiTaskMean::Shared => out.push(("mean_module.raw_constant".to_string(), vec![])),
        }
        out.extend(self.covar.param_shapes("covar_module."));
        out
    }

    /// BoTorch parameter names and sizes in flat-vector order.
    pub fn parameter_names(&self) -> Vec<(String, usize)> {
        self.parameter_shapes()
            .into_iter()
            .map(|(name, shape)| (name, shape.iter().product()))
            .collect()
    }

    /// Optimizer bounds of every raw parameter in flat-vector order
    /// (`get_parameters_and_bounds`).
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let mut b = Vec::with_capacity(self.raw.len());
        if let Noise::Learned(l) = &self.noise {
            b.extend(std::iter::repeat_n(
                l.spec().noise_constraint.raw_bounds(),
                self.num_noise_params(),
            ));
        }
        b.extend(std::iter::repeat_n(free, self.num_mean_params()));
        b.extend(self.covar.raw_bounds());
        b
    }

    /// The noise variances in the transformed outcome space: one per task
    /// (`HadamardGaussianLikelihood`), one shared value, or one per training
    /// row (fixed noise), like GPyTorch's `likelihood.noise`.
    pub fn likelihood_noise(&self) -> Vector {
        self.noise_of(self.raw.as_slice())
    }

    fn noise_of(&self, raw: &[f64]) -> Vector {
        match &self.noise {
            Noise::Learned(l) => Vector::from_iterator(
                self.num_noise_params(),
                raw[..self.num_noise_params()]
                    .iter()
                    .map(|&r| l.spec().noise_constraint.transform(r)),
            ),
            Noise::Fixed(v) => v.clone(),
        }
    }

    /// The constant means (transformed space): one per task or one shared.
    pub fn constants(&self) -> Vector {
        let off = self.num_noise_params();
        Vector::from_column_slice(&self.raw.as_slice()[off..off + self.num_mean_params()])
    }

    fn mean_of(&self, raw: &[f64], task: usize) -> f64 {
        let off = self.num_noise_params();
        match self.mean {
            MultiTaskMean::PerTask => raw[off + task],
            MultiTaskMean::Shared => raw[off],
        }
    }

    /// The raw values of the covariance module.
    pub fn raw_kernel(&self) -> &[f64] {
        &self.raw.as_slice()[self.kernel_range()]
    }

    /// The covariance module with the current hyperparameters.
    ///
    /// # Errors
    /// See [`TreeKernel::build`].
    pub fn kernel(&self) -> Result<TreeKernel> {
        TreeKernel::build(&self.covar, self.raw_kernel(), self.dim() + 1)
    }

    /// The task covariance matrix of the task kernel (`covar_matrix`, `K × K`).
    ///
    /// # Errors
    /// See [`TreeKernel::build`].
    pub fn task_covariance(&self) -> Result<Matrix> {
        let k = self.num_tasks();
        let tf = self.layout.task_feature();
        let range = self.task_kernel_raw_range();
        let t = TreeKernel::build(
            self.task_covar_module(),
            &self.raw_kernel()[range],
            self.dim() + 1,
        )?;
        let tasks = Matrix::from_fn(
            k,
            self.dim() + 1,
            |i, j| if j == tf { i as f64 } else { 0.0 },
        );
        t.covariance_symmetric(&tasks)
    }

    /// The range of the task kernel's raw values within the kernel's.
    fn task_kernel_raw_range(&self) -> Range<usize> {
        let mut off = 0;
        for (i, child) in self.covar.children().iter().enumerate() {
            let len = child.num_raw();
            if i == self.task_kernel {
                return off..off + len;
            }
            off += len;
        }
        unreachable!("the task kernel is a child of the covariance module")
    }

    /// BoTorch's `MultiTaskGP.eval()` for tasks without training data: sets
    /// the rows of the task kernel's `covar_factor` that belong to
    /// unobserved tasks to the mean of the observed tasks' rows (through the
    /// constraint's inverse transform, which GPyTorch applies to all rows).
    /// A no-op if every task has data.
    pub fn eval_unobserved_task_factors(&mut self) {
        let unobserved = self.layout.unobserved_task_indices();
        if unobserved.is_empty() {
            return;
        }
        let observed = self.layout.observed_task_indices();
        let node = self.task_covar_module().clone();
        let Some(fp) = node.params().iter().find(|p| p.name == "raw_covar_factor") else {
            return;
        };
        let var_len: usize = node
            .params()
            .iter()
            .take_while(|p| p.name != "raw_covar_factor")
            .map(|p| p.numel())
            .sum();
        let r = self.rank();
        let kr = self.kernel_range();
        let start = kr.start + self.task_kernel_raw_range().start + var_len;
        let raw = &mut self.raw.as_mut_slice()[start..start + fp.numel()];
        let actual: Vec<f64> = raw.iter().map(|&v| fp.transform(v)).collect();
        let mut mean = vec![0.0; r];
        for &t in &observed {
            for (c, m) in mean.iter_mut().enumerate() {
                *m += actual[t * r + c];
            }
        }
        for m in &mut mean {
            *m /= observed.len() as f64;
        }
        let mut new = actual;
        for &t in &unobserved {
            new[t * r..(t + 1) * r].copy_from_slice(&mean);
        }
        for (dst, v) in raw.iter_mut().zip(new) {
            *dst = fp.inverse_transform(v);
        }
        self.cache = OnceLock::new();
    }

    // ------------------------------------------------------------------
    // Transforms and task mapping
    // ------------------------------------------------------------------

    /// The input transform applied to full inputs (`... × (d + 1)`).
    fn transform_full(&self, x: &Matrix) -> Result<Matrix> {
        apply_input_transform(
            self.input_transform.as_ref(),
            self.input_indices.as_deref(),
            x,
        )
    }

    /// `∂x̃_c/∂x_c` of the input transform for every column `c` of the full
    /// inputs.
    fn column_factors(&self) -> Vec<f64> {
        let mut f = vec![1.0; self.dim() + 1];
        if let Some(tf) = &self.input_transform {
            let coef = tf.coefficient();
            match &self.input_indices {
                Some(idx) => {
                    for (j, &c) in idx.iter().enumerate() {
                        f[c] = 1.0 / coef[j];
                    }
                }
                None => {
                    for (c, v) in f.iter_mut().enumerate() {
                        *v = 1.0 / coef[c];
                    }
                }
            }
        }
        f
    }

    /// The points of the joint posterior at `x`: `q × d` inputs with the
    /// tasks `tasks` (task values, BoTorch's `_make_X_full`, output-major), or
    /// `q × (d + 1)` inputs with their own task column (`tasks = None`).
    fn augment(&self, x: &Matrix, tasks: Option<&[i64]>) -> Result<AugPoints> {
        let d = self.dim();
        let tf = self.layout.task_feature();
        let factors = self.column_factors();
        let (full, src, cols) = match tasks {
            Some(tasks) => {
                if x.ncols() != d {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "Expected inputs with {d} columns (the non-task features), got {}",
                        x.ncols()
                    )));
                }
                for &t in tasks {
                    self.layout.map_task(t as f64, self.validate)?;
                }
                let q = x.nrows();
                let mut full = Matrix::zeros(q * tasks.len(), d + 1);
                for (o, &t) in tasks.iter().enumerate() {
                    full.rows_mut(o * q, q)
                        .copy_from(&self.layout.with_task_column(x, t as f64));
                }
                let src = (0..tasks.len()).flat_map(|_| 0..q).collect();
                let cols = (0..d)
                    .map(|k| {
                        let c = if k < tf { k } else { k + 1 };
                        Some((c, factors[c]))
                    })
                    .collect();
                (full, src, cols)
            }
            None => {
                if x.ncols() != d + 1 {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "Expected inputs with {} columns (including the task feature), got {}",
                        d + 1,
                        x.ncols()
                    )));
                }
                let cols = (0..=d)
                    .map(|c| (c != tf).then_some((c, factors[c])))
                    .collect();
                (x.clone(), (0..x.nrows()).collect(), cols)
            }
        };
        let xt = self.transform_full(&full)?;
        let (kx, tasks) = map_task_column(&self.layout, &xt, self.validate)?;
        Ok(AugPoints {
            kx,
            tasks,
            src,
            cols,
            num_inputs: x.nrows(),
        })
    }

    // ------------------------------------------------------------------
    // Marginal log likelihood and fitting
    // ------------------------------------------------------------------

    /// The noise variance of training row `i` (transformed space).
    fn train_noise(&self, noise: &Vector, i: usize) -> f64 {
        let n0 = self.num_train() - self.point_noise.len();
        if i >= n0 {
            if let Some(v) = self.point_noise[i - n0] {
                return v;
            }
        }
        match &self.noise {
            Noise::Learned(MultiTaskNoise::PerTask(_)) => noise[self.train_tasks[i]],
            Noise::Learned(MultiTaskNoise::Shared(_)) => noise[0],
            Noise::Fixed(v) => v[i],
        }
    }

    /// The exact marginal log likelihood (divided by `n`, priors included) at
    /// the current hyperparameters.
    ///
    /// # Errors
    /// [`BotorchError::NotPsd`] / [`BotorchError::Nan`] if the covariance
    /// cannot be factorized, and [`BotorchError::InvalidArgument`] for a
    /// hyperparameter outside the support of its prior.
    pub fn mll_value(&self) -> Result<f64> {
        self.evaluate_mll(self.raw.as_slice(), false)
            .map(|(v, _)| v)
    }

    /// The exact MLL and its gradient with respect to the flat raw
    /// hyperparameters at `raw`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of parameters,
    /// [`BotorchError::InvalidArgument`] for invalid values and
    /// [`BotorchError::NotPsd`] / [`BotorchError::Nan`] if the covariance
    /// cannot be factorized.
    pub fn mll_value_and_grad(&self, raw: &[f64]) -> Result<(f64, Vector)> {
        if raw.len() != self.raw.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                self.raw.len(),
                raw.len()
            )));
        }
        if raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "non-finite raw hyperparameters: {raw:?}"
            )));
        }
        self.evaluate_mll(raw, true)
            .map(|(v, g)| (v, g.expect("gradient requested")))
    }

    fn evaluate_mll(&self, raw: &[f64], with_grad: bool) -> Result<(f64, Option<Vector>)> {
        let n = self.num_train();
        let nf = n as f64;
        let kr = self.kernel_range();
        let tree = TreeKernel::build(&self.covar, &raw[kr.clone()], self.dim() + 1)?;
        let prep = tree.prepare(&self.train_kx)?;
        let mut k = tree.symmetric(&prep);
        let noise = self.noise_of(raw);
        for i in 0..n {
            k[(i, i)] += self.train_noise(&noise, i);
        }
        let chol = psd_safe_cholesky(&k)?;
        let resid = Vector::from_fn(n, |i, _| {
            self.y_tf[i] - self.mean_of(raw, self.train_tasks[i])
        });
        let v = chol.solve_lower_vec(&resid);
        let mut res = -0.5 * (v.dot(&v) + chol.logdet() + nf * LN_2PI);

        // Prior terms in GPyTorch's `named_priors()` order.
        let noise_prior = match &self.noise {
            Noise::Learned(l) => l.spec().noise_prior,
            Noise::Fixed(_) => None,
        };
        if let Some(p) = noise_prior {
            for &s in noise.iter() {
                res += p.log_prob(s);
            }
        }
        let tree_prior_grad = if with_grad {
            let (lp, g) = tree.log_prior_and_grad()?;
            res += lp;
            Some(g)
        } else {
            res += tree.log_prior()?;
            None
        };
        let value = res / nf;
        if !with_grad {
            return Ok((value, None));
        }

        // dmll/dθ = [½ tr(W ∂K/∂θ) + ∂ log p/∂θ] / n with W = ααᵀ - K⁻¹.
        let alpha = chol.solve_lower_transpose_vec(&v);
        let mut w = chol.inverse();
        w.ger(1.0, &alpha, &alpha, -1.0);
        let mut grad = Vector::zeros(raw.len());
        // Noise.
        if let Noise::Learned(l) = &self.noise {
            let spec = l.spec();
            let mut g = vec![0.0; self.num_noise_params()];
            let n0 = n - self.point_noise.len();
            for i in 0..n {
                if i >= n0 && self.point_noise[i - n0].is_some() {
                    continue;
                }
                let slot = match l {
                    MultiTaskNoise::PerTask(_) => self.train_tasks[i],
                    MultiTaskNoise::Shared(_) => 0,
                };
                g[slot] += 0.5 * w[(i, i)];
            }
            for (j, gj) in g.iter().enumerate() {
                let mut gj = *gj;
                if let Some(p) = spec.noise_prior {
                    gj += p.grad_log_prob(noise[j]);
                }
                grad[j] = gj * spec.noise_constraint.transform_derivative(raw[j]) / nf;
            }
        }
        // Means.
        let off = self.num_noise_params();
        for i in 0..n {
            let slot = match self.mean {
                MultiTaskMean::PerTask => self.train_tasks[i],
                MultiTaskMean::Shared => 0,
            };
            grad[off + slot] += alpha[i] / nf;
        }
        // Kernel.
        let gk = tree.contract(&prep, &w);
        let gp = tree_prior_grad.expect("prior gradient computed");
        let derivs = self.covar.transform_derivatives(&raw[kr.clone()]);
        for (j, idx) in kr.enumerate() {
            grad[idx] = (0.5 * gk[j] + gp[j]) * derivs[j] / nf;
        }
        Ok((value, Some(grad)))
    }

    /// BoTorch's `sample_all_priors`: the raw parameters `initial` with every
    /// hyperparameter that has a (univariate) prior drawn from it.
    fn sample_all_priors(&self, initial: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let mut p = initial.to_vec();
        if let Noise::Learned(l) = &self.noise {
            let spec = l.spec();
            if let Some(prior) = spec.noise_prior {
                for v in &mut p[..self.num_noise_params()] {
                    *v = spec.noise_constraint.inverse_transform(prior.sample(rng));
                }
            }
        }
        let kr = self.kernel_range();
        sample_tree_priors(&self.covar, &mut p[kr], rng);
        p
    }
}

impl Fittable for MultiTaskGP {
    type Output = FitResult;

    /// BoTorch's `fit_gpytorch_mll` (`fit_gpytorch_mll_scipy`: one L-BFGS-B
    /// problem over all hyperparameters, with retries from prior samples),
    /// followed by BoTorch's `eval()` (see
    /// [`MultiTaskGP::eval_unobserved_task_factors`]).
    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        let initial = self.raw.as_slice().to_vec();
        let bounds = self.raw_parameter_bounds();
        let (result, x) = {
            let model: &MultiTaskGP = self;
            minimize_with_retries(
                &initial,
                &bounds,
                options,
                true,
                |x| {
                    model
                        .mll_value_and_grad(x)
                        .map(|(v, g)| (v, g.as_slice().to_vec()))
                },
                |rng| model.sample_all_priors(&initial, rng),
            )?
        };
        self.set_raw_parameters(&x)?;
        self.eval_unobserved_task_factors();
        Ok(result)
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// Fits a `Normalize` of the columns `indices` (all columns if `None`) to
/// the training inputs.
fn fit_input_transform(
    tf: Option<Normalize>,
    indices: Option<Vec<usize>>,
    train_x: &Matrix,
) -> Result<(Option<Normalize>, Option<Vec<usize>>)> {
    let Some(mut tf) = tf else {
        return Ok((None, None));
    };
    let cols = train_x.ncols();
    let sel = match &indices {
        Some(idx) => {
            let mut sorted = idx.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if idx.is_empty() || sorted.len() != idx.len() || idx.iter().any(|&c| c >= cols) {
                return Err(BotorchError::InvalidArgument(format!(
                    "invalid input transform indices {idx:?} for inputs with {cols} columns"
                )));
            }
            if tf.dim() != idx.len() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the input transform has dimension {} but {} indices",
                    tf.dim(),
                    idx.len()
                )));
            }
            select_columns(train_x, idx)
        }
        None => {
            if tf.dim() != cols {
                return Err(BotorchError::DimensionMismatch(format!(
                    "input transform has dimension {} but train_X has {cols} columns",
                    tf.dim()
                )));
            }
            train_x.clone()
        }
    };
    tf.fit(&sel)?;
    Ok((Some(tf), indices))
}

fn select_columns(x: &Matrix, idx: &[usize]) -> Matrix {
    Matrix::from_fn(x.nrows(), idx.len(), |i, j| x[(i, idx[j])])
}

/// Applies a fitted `Normalize` to the columns `indices` of `x`.
fn apply_input_transform(
    tf: Option<&Normalize>,
    indices: Option<&[usize]>,
    x: &Matrix,
) -> Result<Matrix> {
    let Some(tf) = tf else {
        return Ok(x.clone());
    };
    match indices {
        None => tf.transform(x),
        Some(idx) => {
            if idx.iter().any(|&c| c >= x.ncols()) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "inputs with {} columns do not have the transformed columns {idx:?}",
                    x.ncols()
                )));
            }
            let t = tf.transform(&select_columns(x, idx))?;
            let mut out = x.clone();
            for (j, &c) in idx.iter().enumerate() {
                out.set_column(c, &t.column(j));
            }
            Ok(out)
        }
    }
}

/// Replaces the task column of transformed full inputs by the task indices.
fn map_task_column(
    layout: &TaskLayout,
    xt: &Matrix,
    validate: bool,
) -> Result<(Matrix, Vec<usize>)> {
    let tf = layout.task_feature();
    let tasks = xt
        .column(tf)
        .iter()
        .map(|&v| layout.map_task(v, validate))
        .collect::<Result<Vec<_>>>()?;
    let mut kx = xt.clone();
    for (i, &t) in tasks.iter().enumerate() {
        kx[(i, tf)] = t as f64;
    }
    Ok((kx, tasks))
}

/// The constraint of a learned noise level (for the Python bindings).
pub fn noise_constraint(noise: &MultiTaskNoise) -> Constraint {
    noise.spec().noise_constraint
}

// ----------------------------------------------------------------------
// Prediction
// ----------------------------------------------------------------------

impl MultiTaskGP {
    fn compute_cache(&self) -> Result<MtCache> {
        let kernel = self.kernel()?;
        let train_prep = kernel.prepare(&self.train_kx)?;
        let mut k = kernel.symmetric(&train_prep);
        let noise = self.likelihood_noise();
        let n = self.num_train();
        for i in 0..n {
            k[(i, i)] += self.train_noise(&noise, i);
        }
        let chol = psd_safe_cholesky(&k)?;
        let alpha = chol.solve_vec(&self.residuals());
        Ok(MtCache {
            kernel,
            train_prep,
            chol: Arc::new(chol),
            alpha,
        })
    }

    /// `y - μ` of the training rows (transformed space).
    fn residuals(&self) -> Vector {
        let raw = self.raw.as_slice();
        Vector::from_fn(self.num_train(), |i, _| {
            self.y_tf[i] - self.mean_of(raw, self.train_tasks[i])
        })
    }

    fn cache(&self) -> Result<Arc<MtCache>> {
        self.cache
            .get_or_init(|| self.compute_cache().map(Arc::new))
            .clone()
    }

    fn outcome_affine(&self) -> (f64, f64) {
        match &self.outcome_transform {
            Some(st) => (st.means()[0], st.stdvs()[0]),
            None => (0.0, 1.0),
        }
    }

    /// The observation noise of every task (transformed space): the noise
    /// level of the task, or for fixed noise the mean training noise of the
    /// task's rows (of all rows for a task without data).
    fn observation_noise_by_task(&self) -> Vec<f64> {
        let k = self.num_tasks();
        let noise = self.likelihood_noise();
        match &self.noise {
            Noise::Learned(MultiTaskNoise::PerTask(_)) => noise.iter().copied().collect(),
            Noise::Learned(MultiTaskNoise::Shared(_)) => vec![noise[0]; k],
            Noise::Fixed(v) => {
                let all = v.sum() / v.len() as f64;
                (0..k)
                    .map(|t| {
                        let rows: Vec<f64> = (0..v.len())
                            .filter(|&i| self.train_tasks[i] == t)
                            .map(|i| v[i])
                            .collect();
                        if rows.is_empty() {
                            all
                        } else {
                            rows.iter().sum::<f64>() / rows.len() as f64
                        }
                    })
                    .collect()
            }
        }
    }

    /// The joint posterior at the points `aug` in the original outcome space.
    fn aug_joint(&self, aug: &AugPoints, observation_noise: bool) -> Result<(Vector, Matrix)> {
        let c = self.cache()?;
        let prep = c.kernel.prepare(&aug.kx)?;
        let kxs = c.kernel.cross(&prep, &c.train_prep);
        let raw = self.raw.as_slice();
        let mean_tf = Vector::from_fn(aug.kx.nrows(), |a, _| {
            self.mean_of(raw, aug.tasks[a]) + kxs.row(a).dot(&c.alpha.transpose())
        });
        let mut cov = c.kernel.symmetric(&prep);
        let mut v = kxs.transpose();
        c.chol.solve_lower_mut(&mut v);
        cov.gemm_tr(-1.0, &v, &v, 1.0);
        if observation_noise {
            let by_task = self.observation_noise_by_task();
            for (a, &t) in aug.tasks.iter().enumerate() {
                cov[(a, a)] += by_task[t];
            }
        }
        let (mu, sd) = self.outcome_affine();
        Ok((mean_tf.map(|m| mu + sd * m), cov.map(|v| sd * v * sd)))
    }

    /// [`MultiTaskGP::aug_joint`] (without observation noise) with
    /// derivatives with respect to every coordinate of the user's inputs
    /// (index `p·d_in + k`).
    fn aug_with_grad(&self, aug: &AugPoints, num_outputs: usize) -> Result<JointPosteriorGrad> {
        let c = self.cache()?;
        let nn = aug.kx.nrows();
        let din = aug.cols.len();
        let prep = c.kernel.prepare(&aug.kx)?;
        let n = self.num_train();
        let mut kxs = Matrix::zeros(nn, n);
        let mut dks = Vec::with_capacity(nn);
        for a in 0..nn {
            let (k, dk) = c.kernel.cross_with_grad(&prep, a, &c.train_prep);
            kxs.set_row(a, &k.transpose());
            dks.push(dk);
        }
        let raw = self.raw.as_slice();
        let mean_tf = Vector::from_fn(nn, |a, _| {
            self.mean_of(raw, aug.tasks[a]) + kxs.row(a).dot(&c.alpha.transpose())
        });
        let mut cov = c.kernel.symmetric(&prep);
        let mut v = kxs.transpose();
        c.chol.solve_lower_mut(&mut v);
        cov.gemm_tr(-1.0, &v, &v, 1.0);
        let mut w = v;
        c.chol.solve_lower_transpose_mut(&mut w); // K⁻¹ K(X, A) (n × N)
        let (mu, sd) = self.outcome_affine();
        let mut dmean = vec![Vector::zeros(nn); aug.num_inputs * din];
        let mut dcov = vec![Matrix::zeros(nn, nn); aug.num_inputs * din];
        for a in 0..nn {
            let p = aug.src[a];
            let (_, dkaa) = c.kernel.cross_with_grad(&prep, a, &prep);
            let dk_alpha = dks[a].tr_mul(&c.alpha);
            let dk_w = dks[a].tr_mul(&w);
            for (kcol, entry) in aug.cols.iter().enumerate() {
                let Some((cc, f)) = *entry else { continue };
                let j = p * din + kcol;
                dmean[j][a] += sd * dk_alpha[cc] * f;
                for b in 0..nn {
                    let g = (dkaa[(b, cc)] - dk_w[(cc, b)]) * f * sd * sd;
                    dcov[j][(a, b)] += g;
                    dcov[j][(b, a)] += g;
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean: mean_tf.map(|m| mu + sd * m),
            covariance: cov.map(|v| sd * v * sd),
            dmean,
            dcov,
            num_outputs,
        })
    }

    /// Reverse-mode derivative of [`MultiTaskGP::aug_joint`] (without
    /// observation noise) for the user's input rows `rows`.
    fn aug_vjp(
        &self,
        aug: &AugPoints,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        let c = self.cache()?;
        let nn = aug.kx.nrows();
        let prep = c.kernel.prepare(&aug.kx)?;
        let kxs = c.kernel.cross(&prep, &c.train_prep);
        let (_, sd) = self.outcome_affine();
        let mbar = mean_adjoint * sd;
        let abar = (covariance_adjoint + covariance_adjoint.transpose()) * (0.5 * sd * sd);
        let mut w = kxs.transpose();
        c.chol.solve_lower_mut(&mut w);
        c.chol.solve_lower_transpose_mut(&mut w); // K⁻¹ K(X, A) (n × N)
        let mut out = Matrix::zeros(rows.len(), aug.cols.len());
        for a in 0..nn {
            let p = aug.src[a];
            if !rows.contains(&p) {
                continue;
            }
            let a_col = abar.column(a).into_owned();
            // ∂/∂A_a: Dk_aᵀ (mbar_a α - 2 W abar[:, a]) + 2 Σ_b abar[a, b] ∂₁k(A_a, A_b).
            let vvec = &c.alpha * mbar[a] - (&w * &a_col) * 2.0;
            let (_, dk) = c.kernel.cross_with_grad(&prep, a, &c.train_prep);
            let mut g = dk.tr_mul(&vvec);
            let (_, dkk) = c.kernel.cross_with_grad(&prep, a, &prep);
            g += dkk.tr_mul(&a_col) * 2.0;
            for (kcol, entry) in aug.cols.iter().enumerate() {
                if let Some((cc, f)) = *entry {
                    out[(p - rows.start, kcol)] += g[cc] * f;
                }
            }
        }
        Ok(out)
    }

    /// The joint posterior at the non-task features `x` (`q × d`) over the
    /// output tasks (output-major), BoTorch's `model.posterior(X)`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a column mismatch and
    /// [`BotorchError::NotPsd`] if the training covariance cannot be
    /// factorized.
    pub fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let tasks = self.layout.output_tasks().to_vec();
        let aug = self.augment(x, Some(&tasks))?;
        let (mean, cov) = self.aug_joint(&aug, observation_noise)?;
        Ok(Posterior::multi_output(mean, cov, tasks.len()))
    }

    /// The joint posterior at full inputs `x_full` (`q × (d + 1)`, with task
    /// values in the task column): a single-output posterior over the `q`
    /// points, BoTorch's `model.posterior(X)` for `X` that includes the task
    /// feature.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a column mismatch, invalid
    /// task values (see [`TaskLayout::map_task`]) and
    /// [`BotorchError::NotPsd`] if the training covariance cannot be
    /// factorized.
    pub fn posterior_with_tasks(
        &self,
        x_full: &Matrix,
        observation_noise: bool,
    ) -> Result<Posterior> {
        let aug = self.augment(x_full, None)?;
        let (mean, cov) = self.aug_joint(&aug, observation_noise)?;
        Ok(Posterior::single_output(mean, cov))
    }

    /// Point posteriors of the rows of `x` (each on its own) over the tasks
    /// `tasks` (`x` without the task column) or the rows' own tasks.
    fn aug_point_posteriors(&self, x: &Matrix, tasks: Option<&[i64]>) -> Result<PointPosteriors> {
        let aug = self.augment(x, tasks)?;
        let c = self.cache()?;
        let b = x.nrows();
        let m = tasks.map_or(1, <[i64]>::len);
        let prep = c.kernel.prepare(&aug.kx)?;
        let kxs = c.kernel.cross(&prep, &c.train_prep);
        let mut v = kxs.transpose();
        c.chol.solve_lower_mut(&mut v);
        let raw = self.raw.as_slice();
        let (mu, sd) = self.outcome_affine();
        let mut mean = Matrix::zeros(b, m);
        let mut covariance = Matrix::zeros(b, m * m);
        let d1 = self.dim() + 1;
        for p in 0..b {
            let rows: Vec<usize> = (0..m).map(|o| o * b + p).collect();
            for (o, &a) in rows.iter().enumerate() {
                mean[(p, o)] = mu
                    + sd * (self.mean_of(raw, aug.tasks[a]) + kxs.row(a).dot(&c.alpha.transpose()));
            }
            let prior = if m == 1 {
                Matrix::from_element(1, 1, c.kernel.diag(&prep, p))
            } else {
                let pts = Matrix::from_fn(m, d1, |o, j| aug.kx[(rows[o], j)]);
                c.kernel.symmetric(&c.kernel.prepare(&pts)?)
            };
            for o in 0..m {
                for u in 0..m {
                    let cov = prior[(o, u)] - v.column(rows[o]).dot(&v.column(rows[u]));
                    covariance[(p, o * m + u)] = sd * cov * sd;
                }
            }
        }
        Ok(PointPosteriors { mean, covariance })
    }

    // ------------------------------------------------------------------
    // Conditioning and fantasies
    // ------------------------------------------------------------------

    /// BoTorch's `condition_on_observations(X, Y, noise=noise)` for every
    /// set of observations `ys` at the same inputs: new full inputs `x`
    /// (`n' × (d + 1)`, original space), observations `ys` (`n' × 1` each,
    /// original space) and optional noise variances (`n'`, outcome-transformed
    /// space, as BoTorch assumes). The models keep the hyperparameters and the
    /// fitted transforms (applied in eval mode) and share the update of the
    /// Cholesky factor. Explicit noise of a model with learned noise enters
    /// the posterior mean and covariance (BoTorch main uses it for the
    /// covariance only, meta-pytorch/botorch#3396); a fixed-noise model
    /// requires it.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes, invalid
    /// task values (see [`TaskLayout::map_task`]), [`BotorchError::Runtime`]
    /// without noise for a fixed-noise model and [`BotorchError::NotPsd`] if
    /// the covariance cannot be factorized.
    pub fn condition_on_observations_many(
        &self,
        x: &Matrix,
        ys: &[Matrix],
        noise: Option<&[f64]>,
    ) -> Result<Vec<MultiTaskGP>> {
        let k = x.nrows();
        let aug = self.augment(x, None)?;
        for y in ys {
            if y.shape() != (k, 1) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected Y of shape ({k}, 1), got {:?}",
                    y.shape()
                )));
            }
        }
        if let Some(nz) = noise {
            if nz.len() != k {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected {k} noise variances, got {}",
                    nz.len()
                )));
            }
        }
        let n = self.num_train();
        let mut base = self.clone();
        base.cache = OnceLock::new();
        base.train_x = vstack(&self.train_x, x);
        base.train_kx = vstack(&self.train_kx, &aug.kx);
        base.train_tasks.extend(aug.tasks.iter().copied());
        // The original-space variances of the new rows are not known.
        base.train_yvar = None;
        // The noise of the new rows (transformed space).
        let new_noise: Vec<f64> = match &self.noise {
            Noise::Fixed(v) => {
                let nz = noise.ok_or_else(|| {
                    BotorchError::Runtime(
                        "FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg".into(),
                    )
                })?;
                let clamped: Vec<f64> = nz.iter().map(|&s| s.max(MIN_FIXED_NOISE)).collect();
                let mut ext = v.as_slice().to_vec();
                ext.extend(&clamped);
                base.noise = Noise::Fixed(Vector::from_vec(ext));
                clamped
            }
            Noise::Learned(l) => {
                let lik = self.likelihood_noise();
                (0..k)
                    .map(|i| match noise {
                        Some(nz) => {
                            base.point_noise.push(Some(nz[i]));
                            nz[i]
                        }
                        None => {
                            base.point_noise.push(None);
                            match l {
                                MultiTaskNoise::PerTask(_) => lik[aug.tasks[i]],
                                MultiTaskNoise::Shared(_) => lik[0],
                            }
                        }
                    })
                    .collect()
            }
        };
        // The shared update of the Cholesky factor.
        let c = self.cache()?;
        let prep = c.kernel.prepare(&aug.kx)?;
        let cross = c.kernel.cross(&prep, &c.train_prep);
        let mut corner = c.kernel.symmetric(&prep);
        for (i, nz) in new_noise.iter().enumerate() {
            corner[(i, i)] += nz;
        }
        let chol = match c.chol.bordered(&cross, &corner) {
            Some(ch) => ch,
            None => {
                let mut full = Matrix::zeros(n + k, n + k);
                full.view_mut((0, 0), (n, n))
                    .copy_from(&c.kernel.symmetric(&c.train_prep));
                let noise_old = self.likelihood_noise();
                for i in 0..n {
                    full[(i, i)] += self.train_noise(&noise_old, i);
                }
                full.view_mut((n, 0), (k, n)).copy_from(&cross);
                full.view_mut((0, n), (n, k)).copy_from(&cross.transpose());
                full.view_mut((n, n), (k, k)).copy_from(&corner);
                psd_safe_cholesky(&full)?
            }
        };
        let chol = Arc::new(chol);
        let train_prep = c.kernel.prepare(&base.train_kx)?;
        ys.iter()
            .map(|y| {
                let y_tf = match &self.outcome_transform {
                    Some(st) => st.transform(y, None)?.0,
                    None => y.clone(),
                };
                let mut m = base.clone();
                m.train_y = vstack(&self.train_y, y);
                let mut yt = self.y_tf.as_slice().to_vec();
                yt.extend(y_tf.column(0).iter());
                m.y_tf = Vector::from_vec(yt);
                let alpha = chol.solve_vec(&m.residuals());
                let cache = MtCache {
                    kernel: c.kernel.clone(),
                    train_prep: train_prep.clone(),
                    chol: Arc::clone(&chol),
                    alpha,
                };
                m.cache = OnceLock::from(Ok(Arc::new(cache)));
                Ok(m)
            })
            .collect()
    }

    /// [`MultiTaskGP::condition_on_observations_many`] for one set of
    /// observations.
    ///
    /// # Errors
    /// See [`MultiTaskGP::condition_on_observations_many`].
    pub fn condition_on_observations(
        &self,
        x: &Matrix,
        y: &Matrix,
        noise: Option<&[f64]>,
    ) -> Result<MultiTaskGP> {
        Ok(self
            .condition_on_observations_many(x, std::slice::from_ref(y), noise)?
            .pop()
            .expect("one model"))
    }

    /// BoTorch's `fantasize(X, sampler)`: samples `n_f` sets of observations
    /// at the full inputs `x` (`n' × (d + 1)`) from the posterior with
    /// observation noise, with the base samples `base` (`n_f × n'`), and
    /// conditions the model on each of them (without explicit noise, as
    /// BoTorch).
    ///
    /// As in BoTorch, fixed-noise models (whose default fantasy noise is a
    /// tensor) cannot be fantasized: `NotImplementedError` ("Passing a tensor
    /// of observations is not supported by MultiTaskGP.").
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for fixed noise; see
    /// [`MultiTaskGP::condition_on_observations_many`].
    pub fn fantasize_with_base_samples(
        &self,
        x: &Matrix,
        base: &Matrix,
    ) -> Result<Vec<MultiTaskGP>> {
        if matches!(self.noise, Noise::Fixed(_)) {
            return Err(BotorchError::Unsupported(
                "Passing a tensor of observations is not supported by MultiTaskGP.".into(),
            ));
        }
        let k = x.nrows();
        let ys: Vec<Matrix> = if k == 0 {
            vec![Matrix::zeros(0, 1); base.nrows()]
        } else {
            let post = self.posterior_with_tasks(x, true)?;
            let samples = post.rsample_from_base_samples(base)?;
            (0..base.nrows())
                .map(|i| Matrix::from_fn(k, 1, |j, _| samples.value(i, j, 0)))
                .collect()
        };
        self.condition_on_observations_many(x, &ys, None)
    }
}

fn vstack(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.nrows() + b.nrows(), a.ncols());
    out.rows_mut(0, a.nrows()).copy_from(a);
    out.rows_mut(a.nrows(), b.nrows()).copy_from(b);
    out
}

impl Model for MultiTaskGP {
    fn dim(&self) -> usize {
        MultiTaskGP::dim(self)
    }

    fn num_outputs(&self) -> usize {
        self.layout.output_tasks().len()
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        MultiTaskGP::posterior(self, x, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let tasks = self.layout.output_tasks().to_vec();
        let aug = self.augment(x, Some(&tasks))?;
        self.aug_with_grad(&aug, tasks.len())
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        let tasks = self.layout.output_tasks().to_vec();
        check_vjp_shapes_expanded(x, &rows, mean_adjoint, covariance_adjoint, tasks.len(), 1)?;
        let aug = self.augment(x, Some(&tasks))?;
        self.aug_vjp(&aug, rows, mean_adjoint, covariance_adjoint)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        let tasks = self.layout.output_tasks().to_vec();
        self.aug_point_posteriors(x, Some(&tasks))
    }
}

/// The view of a [`MultiTaskGP`] whose inputs include the task column
/// (`d + 1` columns, a single output): BoTorch's `model.posterior(X)` for `X`
/// with task values. Derivatives with respect to the task column are zero.
#[derive(Clone, Debug)]
pub struct MultiTaskTaskInputs {
    model: Arc<MultiTaskGP>,
}

impl MultiTaskTaskInputs {
    /// The view of `model`.
    pub fn new(model: Arc<MultiTaskGP>) -> Self {
        Self { model }
    }

    /// The model.
    pub fn model(&self) -> &Arc<MultiTaskGP> {
        &self.model
    }
}

impl Model for MultiTaskTaskInputs {
    fn dim(&self) -> usize {
        self.model.dim() + 1
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        self.model.posterior_with_tasks(x, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let aug = self.model.augment(x, None)?;
        self.model.aug_with_grad(&aug, 1)
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        check_vjp_shapes_expanded(x, &rows, mean_adjoint, covariance_adjoint, 1, 1)?;
        let aug = self.model.augment(x, None)?;
        self.model
            .aug_vjp(&aug, rows, mean_adjoint, covariance_adjoint)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.model.aug_point_posteriors(x, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, d: usize, tasks: &[f64], tf: usize) -> (Matrix, Matrix) {
        let x = Matrix::from_fn(n, d + 1, |i, j| {
            if j == tf {
                tasks[i % tasks.len()]
            } else {
                ((i * 7 + j * 3) % 11) as f64 / 11.0 + 0.05 * j as f64
            }
        });
        let y = Matrix::from_fn(n, 1, |i, _| {
            let s: f64 = (0..=d).filter(|&j| j != tf).map(|j| x[(i, j)]).sum();
            (3.0 * s).sin() + 0.3 * x[(i, tf)]
        });
        (x, y)
    }

    fn random_raw(m: &MultiTaskGP, seed: u64) -> Vec<f64> {
        let mut state = seed;
        m.raw_parameter_bounds()
            .iter()
            .map(|&(lo, _)| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                if lo.is_finite() {
                    lo + 0.05 + 0.5 * u
                } else {
                    u - 0.5
                }
            })
            .collect()
    }

    fn models() -> Vec<MultiTaskGP> {
        let (x, y) = data(9, 2, &[0.0, 1.0, 2.0], 2);
        let (x2, y2) = data(8, 2, &[3.0, 7.0], 0);
        let mut out = vec![
            MultiTaskGP::new(x.clone(), y.clone(), -1, MultiTaskGPOptions::default()).unwrap(),
            MultiTaskGP::new(
                x2,
                y2,
                0,
                MultiTaskGPOptions {
                    rank: Some(1),
                    output_tasks: Some(vec![7]),
                    all_tasks: Some(vec![3, 5, 7]),
                    ..Default::default()
                },
            )
            .unwrap(),
            MultiTaskGP::new(
                x,
                y,
                2,
                MultiTaskGPOptions {
                    train_yvar: Some(Matrix::from_element(9, 1, 0.02)),
                    mean: MultiTaskMean::Shared,
                    input_transform: Some(Normalize::new(2)),
                    input_transform_indices: Some(vec![0, 1]),
                    ..Default::default()
                },
            )
            .unwrap(),
        ];
        for (i, m) in out.iter_mut().enumerate() {
            let raw = random_raw(m, 11 + i as u64);
            m.set_raw_parameters(&raw).unwrap();
        }
        out
    }

    #[test]
    fn names_and_layout() {
        let m = &models()[0];
        let names: Vec<String> = m.parameter_shapes().into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names,
            vec![
                "likelihood.noise_covar.raw_noise",
                "mean_module.base_means.0.raw_constant",
                "mean_module.base_means.1.raw_constant",
                "mean_module.base_means.2.raw_constant",
                "covar_module.kernels.0.raw_lengthscale",
                "covar_module.kernels.1.raw_var",
                "covar_module.kernels.1.raw_covar_factor",
            ]
        );
        assert_eq!(
            m.covar_module().children()[0].active_dims(),
            Some(&[0, 1][..])
        );
        assert_eq!(m.covar_module().children()[1].active_dims(), Some(&[2][..]));
    }

    #[test]
    fn mll_gradient_matches_finite_differences() {
        for m in models() {
            let raw = m.raw_parameters().as_slice().to_vec();
            let (_, g) = m.mll_value_and_grad(&raw).unwrap();
            for j in 0..raw.len() {
                let h = 1e-6;
                let mut p = raw.clone();
                p[j] += h;
                let fp = m.mll_value_and_grad(&p).unwrap().0;
                p[j] -= 2.0 * h;
                let fm = m.mll_value_and_grad(&p).unwrap().0;
                let fd = (fp - fm) / (2.0 * h);
                assert!(
                    (fd - g[j]).abs() < 1e-6 * (1.0 + fd.abs()),
                    "{j}: {fd} vs {}",
                    g[j]
                );
            }
        }
    }

    #[test]
    fn posterior_gradients_match_finite_differences() {
        for m in models() {
            let d = m.dim();
            let x = Matrix::from_fn(2, d, |i, j| 0.2 + 0.3 * i as f64 + 0.1 * j as f64);
            let mo = Model::num_outputs(&m);
            let g = m.joint_posterior_with_grad(&x).unwrap();
            let n = g.mean.len();
            let ma = Vector::from_fn(n, |i, _| 0.3 + 0.1 * i as f64);
            let ca = Matrix::from_fn(n, n, |i, j| 0.05 * (i + 2 * j) as f64 - 0.1);
            let vjp = m.joint_posterior_vjp(&x, 0..2, &ma, &ca).unwrap();
            for p in 0..2 {
                for k in 0..d {
                    let h = 1e-6;
                    let mut xp = x.clone();
                    xp[(p, k)] += h;
                    let a = Model::posterior(&m, &xp, false).unwrap();
                    xp[(p, k)] -= 2.0 * h;
                    let b = Model::posterior(&m, &xp, false).unwrap();
                    let dm = (&a.mean - &b.mean) / (2.0 * h);
                    let dc = (&a.covariance - &b.covariance) / (2.0 * h);
                    let j = p * d + k;
                    assert!((&dm - &g.dmean[j]).amax() < 1e-6, "dmean {p} {k}");
                    assert!((&dc - &g.dcov[j]).amax() < 1e-6, "dcov {p} {k}");
                    let s = ma.dot(&dm) + ca.component_mul(&dc).sum();
                    assert!(
                        (s - vjp[(p, k)]).abs() < 1e-6,
                        "vjp {p} {k}: {s} vs {}",
                        vjp[(p, k)]
                    );
                }
            }
            assert_eq!(g.mean.len(), 2 * mo);
            // Point posteriors agree with the joint posteriors of single points.
            let pp = m.point_posteriors(&x).unwrap();
            for p in 0..2 {
                let post = Model::posterior(&m, &x.rows(p, 1).into_owned(), false).unwrap();
                for o in 0..mo {
                    assert!((pp.mean[(p, o)] - post.mean[o]).abs() < 1e-12);
                    for u in 0..mo {
                        assert!(
                            (pp.covariance[(p, o * mo + u)] - post.covariance[(o, u)]).abs()
                                < 1e-12
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn task_inputs_view_matches_output_tasks() {
        let m = Arc::new(models().remove(0));
        let x = Matrix::from_fn(2, 2, |i, j| 0.3 + 0.2 * (i + j) as f64);
        let all = m.posterior(&x, true).unwrap();
        let view = MultiTaskTaskInputs::new(Arc::clone(&m));
        let full = m.task_layout().with_task_column(&x, 2.0);
        let one = view.posterior(&full, true).unwrap();
        for p in 0..2 {
            assert!((one.mean[p] - all.mean[4 + p]).abs() < 1e-12);
            assert!((one.covariance[(p, p)] - all.covariance[(4 + p, 4 + p)]).abs() < 1e-12);
        }
        let g = view.joint_posterior_with_grad(&full).unwrap();
        assert!(g.dmean[2].amax() == 0.0 && g.dcov[2].amax() == 0.0);
    }

    #[test]
    fn conditioning_matches_a_model_with_the_data() {
        let m = models().remove(0);
        let xn = Matrix::from_row_slice(2, 3, &[0.2, 0.7, 1.0, 0.9, 0.1, 2.0]);
        let yn = Matrix::from_row_slice(2, 1, &[0.4, -0.3]);
        let c = m.condition_on_observations(&xn, &yn, None).unwrap();
        // The same model with the data, without the cached update.
        let mut fresh = c.clone();
        fresh.cache = OnceLock::new();
        let x = Matrix::from_fn(3, 2, |i, j| 0.1 + 0.25 * (i + j) as f64);
        let a = c.posterior(&x, false).unwrap();
        let b = fresh.posterior(&x, false).unwrap();
        assert!((&a.mean - &b.mean).amax() < 1e-10);
        assert!((&a.covariance - &b.covariance).amax() < 1e-10);
        assert_eq!(c.num_train(), m.num_train() + 2);
    }

    #[test]
    fn unobserved_task_factors_are_the_observed_mean() {
        let mut m = models().remove(1);
        m.eval_unobserved_task_factors();
        let node = m.task_covar_module().clone();
        let r = m.rank();
        let start =
            m.kernel_range().start + m.task_kernel_raw_range().start + node.params()[0].numel();
        let fp = &node.params()[1];
        let f: Vec<f64> = m.raw.as_slice()[start..start + 3 * r]
            .iter()
            .map(|&v| fp.transform(v))
            .collect();
        // Task 5 (index 1) has no data.
        assert!((f[1] - 0.5 * (f[0] + f[2])).abs() < 1e-12);
    }
}
