//! [`LCEMGP`]: BoTorch's latent context embedding multi-output GP
//! (`botorch.models.contextual_multioutput.LCEMGP`, Feng et al., NeurIPS
//! 2020), a `MultiTaskGP` whose task covariance is a kernel on context
//! embeddings.
//!
//! # The model
//!
//! The training inputs `n × (d + 1)` contain a task (context) column; the
//! other `d` columns are the data features (see [`TaskLayout`]). For `K`
//! tasks with indices `t`:
//!
//! * per-task constant means `c_t` (GPyTorch's `MultitaskMean` of
//!   `ConstantMean`s, parameters `mean_module.base_means.{t}.raw_constant`);
//! * per-task noise variances `σ²_t` (`HadamardGaussianLikelihood` with a
//!   `LogNormal(-4, 1)` prior and `GreaterThan(1e-4)`, parameter
//!   `likelihood.noise_covar.raw_noise` of shape `K`);
//! * the covariance `k((x, t), (x', t')) = k_x(x, x') · k_e(E_t, E_t')`: the
//!   data kernel `k_x` (BoTorch's dimension-scaled RBF kernel,
//!   `covar_module`) times a kernel `k_e` on the context embeddings
//!   (`task_covar_module_base`, an RBF kernel with lengthscales constrained to
//!   `Interval(0, 2)` without transform, i.e. box bounds, and no prior);
//! * the embedding `E_t` of a task is the concatenation of learned embeddings
//!   (one `torch.nn.Embedding(num_embeddings, dim, max_norm=1.0)` per
//!   categorical context feature, looked up with `context_cat_feature`) and
//!   the fixed embedding `context_emb_feature[t]`;
//! * a `Standardize` outcome transform over all training rows (BoTorch's
//!   default).
//!
//! The posterior over the data features `X` (`q × d`) is over the output
//! tasks (`output_tasks`, in BoTorch's output-major layout); tasks without
//! training data (`all_tasks`) are predicted through the embedding kernel.
//!
//! # Embedding renormalization
//!
//! `torch.nn.Embedding(max_norm=1.0)` renormalizes the looked-up rows of its
//! weight *in place* on every forward pass (under `no_grad`, before the
//! lookup): a row with norm `‖w‖ > 1` is multiplied by `1 / (‖w‖ + 1e-7)`.
//! The model therefore always uses the renormalized weights
//! ([`embedding_renorm`]), and the gradient of the marginal log likelihood
//! with respect to the weights is the gradient at the renormalized weights,
//! *without* the derivative of the renormalization. During fitting, BoTorch's
//! L-BFGS-B state is the raw weight vector `x`, while every evaluation sets
//! the parameter to `x`, renormalizes it and returns the loss and gradient
//! at `renorm(x)`; the fitted parameter is the final `x` (renormalized by the
//! next forward pass). [`LCEMGP::mll_value_and_grad`] and the
//! [`Fittable`] implementation reproduce exactly this. The stored weights are
//! BoTorch's parameter values; [`LCEMGP::renorm_embeddings`] applies the
//! in-place renormalization of a forward pass.
//!
//! # The Unity project's models
//!
//! The Unity project (BOforUnity) runs two variants, both supported here:
//!
//! 1. **Learned embeddings** (BoTorch's `LCEMGP` with `context_cat_feature`
//!    and `embs_dim_list=[1]`, no fixed embeddings).
//! 2. **Provided embeddings** (`provided_embedding`, the Unity project's
//!    `_ProvidedEmbeddingLCEMGP`, not part of BoTorch): no learned
//!    embeddings; the task kernel sees only `context_emb_feature`, and the
//!    context covariance is `(1 - w) · k_e(E, E) + w · I` with a learned
//!    context-specific weight `w` (`raw_context_specific_weight`, constraint
//!    `Interval(0, 1, transform=None, initial_value=0)`, i.e. box bounds), so
//!    that contexts with identical embeddings can still be fitted apart.
//!
//! # Differences from BoTorch
//!
//! The model is BoTorch's with the behavior the Unity project gives it (its
//! `_UnobservedContextLCEMGP`), which fixes two problems of BoTorch 0.18.1
//! and main:
//!
//! 1. `MultiTaskGP.eval()` initializes the `IndexKernel` factors of tasks
//!    without data, which `LCEMGP` does not have (an `AttributeError` with
//!    unobserved tasks). Instead, [`LCEMGP::eval_unobserved_means`] (called
//!    by the Python binding's `eval()`, i.e. by every posterior) sets the
//!    constant mean of every task without data to the average of the
//!    observed tasks' constants; an unobserved constant never leaves its
//!    initial value otherwise, which offsets the predictions for a new
//!    context. Observed tasks are unaffected.
//! 2. BoTorch sizes the default task kernel for the learned embeddings only
//!    (`ard_num_dims = sum(embs_dim_list)`) but evaluates it on the learned
//!    and the fixed embeddings, which fails with a `context_emb_feature`.
//!    Here the default task kernel has one lengthscale per embedding
//!    dimension (`sum(embs_dim_list) + m`).
//!
//! Raw task values are mapped to task indices before the per-task noise
//! levels are looked up (meta-pytorch/botorch#3396; identical to BoTorch for
//! task values `0..K-1`). The learned embeddings are initialized from
//! `N(0, 1)` with botorch-rust's seeded generator (BoTorch: torch's global
//! generator); explicit initial weights reproduce BoTorch.
//!
//! # Fitting
//!
//! The flat raw parameters follow the order of BoTorch's optimizer
//! (`get_parameters_and_bounds` of the marginal log likelihood: the
//! likelihood's noise first, then the model's own `raw_context_specific_weight`
//! and its modules). As in BoTorch, an evaluation whose task covariance is
//! NaN (a task lengthscale at the lower bound 0 of `Interval(0, 2)`) gives
//! NaN values, from which L-BFGS-B backtracks.

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use rand::rngs::{StdRng, Xoshiro256PlusPlus};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

use crate::constraints::Interval;
use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::kernels::{
    get_covar_module_with_dim_scaled_prior, Kernel, KernelKind, KernelSpec, ScaledPoints,
};
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::models::gp::{get_gaussian_likelihood_with_lognormal_prior, GaussianLikelihoodSpec};
use crate::models::model::check_vjp_shapes_expanded;
use crate::models::multitask::TaskLayout;
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
use crate::transforms::{OutcomeTransformChoice, Standardize};
use crate::utils::sampling::random_seed;
use crate::{Matrix, Vector};

/// `max_norm` of LCEMGP's embedding layers.
pub const EMBEDDING_MAX_NORM: f64 = 1.0;

/// `log(2π)`.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// torch's `embedding_renorm_`: every row `r` of `rows` (duplicates are
/// renormalized once) whose Euclidean norm exceeds `max_norm` is multiplied
/// by `max_norm / (norm + 1e-7)`.
pub fn embedding_renorm(weight: &mut Matrix, rows: &[usize], max_norm: f64) {
    let mut sorted = rows.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    for r in sorted {
        let norm = weight.row(r).iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > max_norm {
            let scale = max_norm / (norm + 1e-7);
            for v in weight.row_mut(r).iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// BoTorch's default task kernel of `LCEMGP` (with the corrected size, see
/// the [module documentation](self)): an RBF kernel on `ard_num_dims`
/// embedding dimensions with the lengthscale constraint `Interval(0.0, 2.0,
/// transform=None, initial_value=1.0)` and no prior.
pub fn default_task_covar_module(ard_num_dims: usize) -> KernelSpec {
    let constraint = Interval::new(0.0, 2.0, None, None, Some(1.0))
        .expect("valid interval")
        .rounded_to_f32();
    KernelSpec {
        lengthscale_constraint: constraint.into(),
        ..KernelSpec::new(KernelKind::Rbf, ard_num_dims)
    }
}

/// Options of [`LCEMGP::new`] (BoTorch's keyword arguments).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LCEMGPOptions {
    /// The output tasks (task values; `None`: all tasks, sorted).
    pub output_tasks: Option<Vec<i64>>,
    /// All task values, including tasks without data (`None`: the tasks of
    /// the training data).
    pub all_tasks: Option<Vec<i64>>,
    /// The categorical context features (`K × k`, integer valued; row `t`
    /// belongs to task index `t`). `None`: the task values (`k = 1`), in the
    /// order of `all_tasks`.
    pub context_cat_feature: Option<Matrix>,
    /// The fixed context embeddings (`K × m`), concatenated to the learned
    /// embeddings.
    pub context_emb_feature: Option<Matrix>,
    /// The dimension of the learned embedding of every categorical feature
    /// (`None`: 1 each).
    pub embs_dim_list: Option<Vec<usize>>,
    /// Outcome transform (BoTorch default: `Standardize` over all rows).
    pub outcome_transform: OutcomeTransformChoice,
    /// The data kernel (`covar_module`; default: BoTorch's dimension-scaled
    /// RBF kernel).
    pub covar_module: Option<KernelSpec>,
    /// The task kernel on the embeddings (`task_covar_module_base`; default:
    /// [`default_task_covar_module`]).
    pub task_covar_module: Option<KernelSpec>,
    /// Initial weights of the embedding layers (`num_embeddings × dim`
    /// each); default: `N(0, 1)` draws from a generator seeded with `seed`.
    pub embedding_weights: Option<Vec<Matrix>>,
    /// Seed of the default embedding initialization (`None`: random).
    pub seed: Option<u64>,
    /// The provided-embedding variant (see the [module documentation](self)):
    /// no learned embeddings, the task kernel on `context_emb_feature` only
    /// (required) and a learned context-specific weight.
    pub provided_embedding: bool,
}

/// The hyperparameters of an [`LCEMGP`] in BoTorch's raw parameterization.
#[derive(Clone, Debug, PartialEq)]
pub struct LcemParameters {
    /// `likelihood.noise_covar.raw_noise` (one per task).
    pub raw_noise: Vector,
    /// `raw_context_specific_weight` (the provided-embedding variant).
    pub raw_context_specific_weight: Option<f64>,
    /// `mean_module.base_means.{t}.raw_constant` (one per task).
    pub raw_constant: Vector,
    /// `covar_module.raw_outputscale` (with a `ScaleKernel`).
    pub raw_outputscale: Option<f64>,
    /// `covar_module(.base_kernel).raw_lengthscale` (`d`).
    pub raw_lengthscale: Vector,
    /// `emb_layers.{i}.weight` (`num_embeddings × dim` each).
    pub embedding_weights: Vec<Matrix>,
    /// `task_covar_module_base.raw_outputscale` (with a `ScaleKernel`).
    pub task_raw_outputscale: Option<f64>,
    /// `task_covar_module_base(.base_kernel).raw_lengthscale`.
    pub task_raw_lengthscale: Vector,
}

impl LcemParameters {
    /// Flattens the parameters in the order of BoTorch's optimizer (see the
    /// [module documentation](self)).
    pub fn to_vector(&self) -> Vector {
        let mut v = Vec::new();
        v.extend(self.raw_noise.iter());
        v.extend(self.raw_context_specific_weight);
        v.extend(self.raw_constant.iter());
        v.extend(self.raw_outputscale);
        v.extend(self.raw_lengthscale.iter());
        for w in &self.embedding_weights {
            for i in 0..w.nrows() {
                v.extend(w.row(i).iter());
            }
        }
        v.extend(self.task_raw_outputscale);
        v.extend(self.task_raw_lengthscale.iter());
        Vector::from_vec(v)
    }

    /// The parameters of the same structure as `self` from a flat vector.
    fn with_values(&self, values: &[f64]) -> Result<Self> {
        let expected = self.to_vector().len();
        if values.len() != expected {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {expected} raw parameters, got {}",
                values.len()
            )));
        }
        let mut it = values.iter().copied();
        let mut take = |k: usize| Vector::from_iterator(k, it.by_ref().take(k));
        let raw_noise = take(self.raw_noise.len());
        let raw_context_specific_weight = self.raw_context_specific_weight.map(|_| take(1)[0]);
        let raw_constant = take(self.raw_constant.len());
        let raw_outputscale = self.raw_outputscale.map(|_| take(1)[0]);
        let raw_lengthscale = take(self.raw_lengthscale.len());
        let embedding_weights = self
            .embedding_weights
            .iter()
            .map(|w| {
                let v = take(w.len());
                Matrix::from_row_slice(w.nrows(), w.ncols(), v.as_slice())
            })
            .collect();
        let task_raw_outputscale = self.task_raw_outputscale.map(|_| take(1)[0]);
        let task_raw_lengthscale = take(self.task_raw_lengthscale.len());
        Ok(Self {
            raw_noise,
            raw_context_specific_weight,
            raw_constant,
            raw_outputscale,
            raw_lengthscale,
            embedding_weights,
            task_raw_outputscale,
            task_raw_lengthscale,
        })
    }
}

/// Quantities derived from the hyperparameters and cached for prediction.
#[derive(Clone, Debug)]
struct LcemCache {
    kernel: Kernel,
    train_scaled: ScaledPoints,
    /// The task covariance over the context rows (`K' × K'`).
    task_cov: Matrix,
    noise: Vector,
    chol: Arc<CholeskyFactor>,
    alpha: Vector,
}

/// BoTorch's `LCEMGP` (see the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct LCEMGP {
    train_x: Matrix,
    train_y: Matrix,
    layout: TaskLayout,
    x_data: Matrix,
    train_tasks: Vec<usize>,
    y_tf: Vector,
    /// Explicit transformed-space noise for conditioned rows, otherwise the
    /// current learned noise of the row's task.
    point_noise: Vec<Option<f64>>,
    outcome_transform: Option<Standardize>,
    likelihood: GaussianLikelihoodSpec,
    covar_module: KernelSpec,
    task_covar_module: KernelSpec,
    /// Per context row, the embedding index of every categorical feature.
    context_cat: Vec<Vec<usize>>,
    /// `(num_embeddings, dim)` of every embedding layer.
    embedding_shapes: Vec<(usize, usize)>,
    context_emb: Option<Matrix>,
    output_tasks: Vec<usize>,
    provided_embedding: bool,
    params: LcemParameters,
    cache: OnceLock<std::result::Result<Arc<LcemCache>, BotorchError>>,
}

fn check_no_nan(m: &Matrix, what: &str) -> Result<()> {
    if m.iter().any(|v| v.is_nan()) {
        return Err(BotorchError::InputData(format!(
            "{what} contains NaN values."
        )));
    }
    Ok(())
}

/// Reads an integer-valued feature like torch's `.to(torch.long)`.
fn index_value(v: f64, what: &str) -> Result<i64> {
    if !v.is_finite() {
        return Err(BotorchError::InputData(format!(
            "{what} must be finite, got {v}"
        )));
    }
    Ok(v.trunc() as i64)
}

impl LCEMGP {
    /// Builds the model from training inputs `train_x` (`n × (d + 1)`, with
    /// the task values in column `task_feature`) and targets `train_y`
    /// (`n × 1`); see [`LCEMGPOptions`].
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
    /// [`BotorchError::InputData`] for NaN values or invalid context
    /// features, and the errors of [`TaskLayout::new`].
    pub fn new(
        train_x: Matrix,
        train_y: Matrix,
        task_feature: isize,
        options: LCEMGPOptions,
    ) -> Result<Self> {
        let n = train_x.nrows();
        if n == 0 {
            return Err(BotorchError::InputData(
                "train_X must have at least one row".into(),
            ));
        }
        if train_y.ncols() != 1 {
            return Err(BotorchError::DimensionMismatch(format!(
                "LCEMGP requires a single-output train_Y (n x 1), got {} columns",
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
        let layout = TaskLayout::new(
            &train_x,
            task_feature,
            options.output_tasks.as_deref(),
            options.all_tasks.as_deref(),
        )?;
        let (x_data, train_tasks) = layout.split_inputs(&train_x)?;
        let d = layout.num_non_task_features();
        let k_tasks = layout.num_tasks();

        let mut outcome_transform = match options.outcome_transform {
            OutcomeTransformChoice::Default => Some(Standardize::new()),
            OutcomeTransformChoice::None => None,
            OutcomeTransformChoice::Standardize(s) => Some(s),
            OutcomeTransformChoice::Chains(_) => {
                return Err(BotorchError::Unsupported(
                    "LCEMGP supports the Standardize outcome transform only".into(),
                ))
            }
        };
        let y_tf = match outcome_transform.as_mut() {
            Some(st) => st.fit_transform(&train_y, None)?.0,
            None => train_y.clone(),
        };
        check_no_nan(&y_tf, "transformed train_Y")?;

        let (context_cat, embedding_shapes, context_emb, emb_dim) = if options.provided_embedding {
            let f = match options.context_emb_feature {
                Some(f) if f.ncols() > 0 => f,
                _ => {
                    return Err(BotorchError::InvalidArgument(
                        "the provided-embedding LCEMGP requires a context_emb_feature with at \
                         least one column"
                            .into(),
                    ))
                }
            };
            check_no_nan(&f, "context_emb_feature")?;
            if f.nrows() < k_tasks {
                return Err(BotorchError::DimensionMismatch(format!(
                    "context_emb_feature has {} rows but the model has {k_tasks} tasks (one row \
                     per task index)",
                    f.nrows()
                )));
            }
            let m = f.ncols();
            (vec![Vec::new(); f.nrows()], Vec::new(), Some(f), m)
        } else {
            learned_context_features(
                &layout,
                options.context_cat_feature,
                options.context_emb_feature,
                options.embs_dim_list,
                options.all_tasks.as_deref(),
            )?
        };

        let covar_module = options
            .covar_module
            .unwrap_or_else(|| get_covar_module_with_dim_scaled_prior(d, true));
        if covar_module.ard_num_dims != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "covar_module has ard_num_dims = {} but the inputs have {d} non-task features",
                covar_module.ard_num_dims
            )));
        }
        let task_covar_module = options
            .task_covar_module
            .unwrap_or_else(|| default_task_covar_module(emb_dim));
        check_task_kernel(&task_covar_module, emb_dim)?;

        let embedding_weights = match options.embedding_weights {
            Some(ws) => {
                check_embedding_weights(&ws, &embedding_shapes)?;
                ws
            }
            None => {
                // N(0, 1) like `torch.nn.Embedding`, drawn row by row.
                let mut rng =
                    Xoshiro256PlusPlus::seed_from_u64(options.seed.unwrap_or_else(random_seed));
                embedding_shapes
                    .iter()
                    .map(|&(r, c)| {
                        let draws: Vec<f64> =
                            (0..r * c).map(|_| rng.sample(StandardNormal)).collect();
                        Matrix::from_row_slice(r, c, &draws)
                    })
                    .collect()
            }
        };
        let likelihood = get_gaussian_likelihood_with_lognormal_prior();
        let params = LcemParameters {
            raw_noise: Vector::from_element(k_tasks, likelihood.initial_raw_noise()),
            raw_context_specific_weight: options.provided_embedding.then(|| {
                context_specific_weight_constraint()
                    .initial_raw_value()
                    .expect("initial value")
            }),
            raw_constant: Vector::zeros(k_tasks),
            raw_outputscale: covar_module.initial_raw_outputscale(),
            raw_lengthscale: Vector::from_element(d, covar_module.initial_raw_lengthscale()),
            embedding_weights,
            task_raw_outputscale: task_covar_module.initial_raw_outputscale(),
            task_raw_lengthscale: Vector::from_element(
                emb_dim,
                task_covar_module.initial_raw_lengthscale(),
            ),
        };
        let output_tasks = layout.output_task_indices();
        Ok(Self {
            train_x,
            train_y,
            layout,
            x_data,
            train_tasks,
            y_tf: Vector::from_column_slice(y_tf.column(0).as_slice()),
            point_noise: vec![None; n],
            outcome_transform,
            likelihood,
            covar_module,
            task_covar_module,
            context_cat,
            embedding_shapes,
            context_emb,
            output_tasks,
            provided_embedding: options.provided_embedding,
            params,
            cache: OnceLock::new(),
        })
    }

    // ------------------------------------------------------------------
    // Data and configuration
    // ------------------------------------------------------------------

    /// Training inputs including the task column (`n × (d + 1)`).
    pub fn train_inputs(&self) -> &Matrix {
        &self.train_x
    }

    /// Training targets in the original outcome space (`n × 1`).
    pub fn train_targets(&self) -> &Matrix {
        &self.train_y
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

    /// The number of non-task features `d` (the input dimension of the
    /// posterior over the output tasks).
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

    /// The output tasks (task indices).
    pub fn output_task_indices(&self) -> &[usize] {
        &self.output_tasks
    }

    /// The fitted outcome transform, if any.
    pub fn outcome_transform(&self) -> Option<&Standardize> {
        self.outcome_transform.as_ref()
    }

    /// The per-task noise configuration (prior and constraint).
    pub fn likelihood(&self) -> &GaussianLikelihoodSpec {
        &self.likelihood
    }

    /// The data kernel configuration (`covar_module`).
    pub fn covar_module(&self) -> &KernelSpec {
        &self.covar_module
    }

    /// The task kernel configuration (`task_covar_module_base`).
    pub fn task_covar_module(&self) -> &KernelSpec {
        &self.task_covar_module
    }

    /// The embedding index of every categorical feature for every context
    /// row (`context_cat_feature` as integers).
    pub fn context_cat_feature(&self) -> &[Vec<usize>] {
        &self.context_cat
    }

    /// The fixed embeddings (`context_emb_feature`), if any.
    pub fn context_emb_feature(&self) -> Option<&Matrix> {
        self.context_emb.as_ref()
    }

    /// `(num_embeddings, embedding_dim)` of every embedding layer (BoTorch's
    /// `emb_dims`).
    pub fn embedding_shapes(&self) -> &[(usize, usize)] {
        &self.embedding_shapes
    }

    /// The dimension of the task embeddings (learned plus fixed).
    pub fn embedding_dim(&self) -> usize {
        self.task_covar_module.ard_num_dims
    }

    /// A copy whose posterior is over other output tasks (task values).
    ///
    /// # Errors
    /// [`BotorchError::Runtime`] if a value is not a task.
    pub fn with_output_tasks(&self, output_tasks: &[i64]) -> Result<Self> {
        let layout = self.layout.with_output_tasks(output_tasks)?;
        let output_tasks = layout.output_task_indices();
        Ok(Self {
            layout,
            output_tasks,
            ..self.clone()
        })
    }

    /// Replaces the task kernel (BoTorch's `model.task_covar_module_base =
    /// kernel`): its hyperparameters start at the kernel's initial values.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the kernel's `ard_num_dims`
    /// differs from the embedding dimension.
    pub fn set_task_covar_module(&mut self, spec: KernelSpec) -> Result<()> {
        check_task_kernel(&spec, self.embedding_dim())?;
        self.params.task_raw_outputscale = spec.initial_raw_outputscale();
        self.params.task_raw_lengthscale =
            Vector::from_element(spec.ard_num_dims, spec.initial_raw_lengthscale());
        self.task_covar_module = spec;
        self.cache = OnceLock::new();
        Ok(())
    }

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------

    /// The current raw hyperparameters.
    pub fn parameters(&self) -> &LcemParameters {
        &self.params
    }

    /// The raw hyperparameters flattened in BoTorch's `named_parameters()`
    /// order.
    pub fn raw_parameters(&self) -> Vector {
        self.params.to_vector()
    }

    /// Sets the raw hyperparameters from a flat vector.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of values and
    /// [`BotorchError::InvalidArgument`] for non-finite values, kernels with
    /// non-positive lengthscales or outputscales, or negative noise.
    pub fn set_raw_parameters(&mut self, values: &[f64]) -> Result<()> {
        let p = self.params.with_values(values)?;
        self.set_parameters(p)
    }

    /// Sets the raw hyperparameters.
    ///
    /// # Errors
    /// See [`LCEMGP::set_raw_parameters`].
    pub fn set_parameters(&mut self, params: LcemParameters) -> Result<()> {
        if params.to_vector().len() != self.params.to_vector().len()
            || params.raw_noise.len() != self.num_tasks()
            || params.raw_context_specific_weight.is_some() != self.provided_embedding
            || params.raw_lengthscale.len() != self.dim()
            || params.raw_outputscale.is_some() != self.params.raw_outputscale.is_some()
            || params.task_raw_outputscale.is_some() != self.params.task_raw_outputscale.is_some()
            || params.task_raw_lengthscale.len() != self.embedding_dim()
        {
            return Err(BotorchError::DimensionMismatch(
                "the hyperparameters do not match the structure of the model".into(),
            ));
        }
        check_embedding_weights(&params.embedding_weights, &self.embedding_shapes)?;
        self.validate(&params)?;
        self.params = params;
        self.cache = OnceLock::new();
        Ok(())
    }

    fn validate(&self, p: &LcemParameters) -> Result<()> {
        if p.to_vector().iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "hyperparameters must be finite".into(),
            ));
        }
        self.covar_module
            .build(p.raw_lengthscale.as_slice(), p.raw_outputscale)?;
        self.task_covar_module
            .build(p.task_raw_lengthscale.as_slice(), p.task_raw_outputscale)?;
        for &raw in p.raw_noise.iter() {
            let noise = self.likelihood.noise_constraint.transform(raw);
            if noise.is_nan() || noise < 0.0 {
                return Err(BotorchError::InvalidArgument(format!(
                    "noise must be non-negative, got {noise}"
                )));
            }
        }
        Ok(())
    }

    /// BoTorch parameter names and shapes in flat-vector order (the order of
    /// BoTorch's optimizer: `model.named_parameters()` yields the model's own
    /// `raw_context_specific_weight` first).
    pub fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let k = self.num_tasks();
        let mut out = vec![("likelihood.noise_covar.raw_noise".to_string(), vec![k])];
        if self.provided_embedding {
            out.push(("raw_context_specific_weight".to_string(), vec![1]));
        }
        for t in 0..k {
            out.push((format!("mean_module.base_means.{t}.raw_constant"), vec![]));
        }
        kernel_shapes(&mut out, "covar_module", &self.covar_module);
        for (i, &(r, c)) in self.embedding_shapes.iter().enumerate() {
            out.push((format!("emb_layers.{i}.weight"), vec![r, c]));
        }
        kernel_shapes(&mut out, "task_covar_module_base", &self.task_covar_module);
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
    /// (`get_parameters_and_bounds`): the noise and lengthscale constraints'
    /// bounds (`Interval(0, 2)` box bounds for the task lengthscales), and
    /// `(-inf, inf)` for the means and the embedding weights.
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let mut b = Vec::new();
        let noise = self.likelihood.noise_constraint.raw_bounds();
        b.extend(std::iter::repeat_n(noise, self.num_tasks()));
        if self.provided_embedding {
            b.push(context_specific_weight_constraint().raw_bounds());
        }
        b.extend(std::iter::repeat_n(free, self.num_tasks()));
        kernel_bounds(&mut b, &self.covar_module);
        for &(r, c) in &self.embedding_shapes {
            b.extend(std::iter::repeat_n(free, r * c));
        }
        kernel_bounds(&mut b, &self.task_covar_module);
        b
    }

    /// The per-task noise variances (transformed outcome space).
    pub fn noise(&self) -> Vector {
        self.params
            .raw_noise
            .map(|r| self.likelihood.noise_constraint.transform(r))
    }

    /// The per-task constant means (transformed outcome space).
    pub fn constant(&self) -> &Vector {
        &self.params.raw_constant
    }

    /// Whether this is the provided-embedding variant.
    pub fn is_provided_embedding(&self) -> bool {
        self.provided_embedding
    }

    /// The context-specific weight `w` of the provided-embedding variant.
    pub fn context_specific_weight(&self) -> Option<f64> {
        self.params
            .raw_context_specific_weight
            .map(|r| context_specific_weight_constraint().transform(r))
    }

    /// The Unity project's `eval()` for tasks without data: sets their
    /// constant means to the average of the observed tasks' constants (an
    /// unobserved constant has no gradient and never leaves its initial
    /// value otherwise). Observed tasks, the marginal log likelihood and
    /// every other hyperparameter are unchanged.
    pub fn eval_unobserved_means(&mut self) {
        let k = self.num_tasks();
        let observed: Vec<usize> = (0..k).filter(|&t| self.layout.is_observed(t)).collect();
        if observed.len() == k || observed.is_empty() {
            return;
        }
        let c = &self.params.raw_constant;
        let level = observed.iter().map(|&t| c[t]).sum::<f64>() / observed.len() as f64;
        let mut changed = false;
        for t in 0..k {
            if !self.layout.is_observed(t) && self.params.raw_constant[t] != level {
                self.params.raw_constant[t] = level;
                changed = true;
            }
        }
        if changed {
            self.cache = OnceLock::new();
        }
    }

    /// The data kernel's lengthscales.
    pub fn lengthscale(&self) -> Vector {
        self.params
            .raw_lengthscale
            .map(|r| self.covar_module.lengthscale_constraint.transform(r))
    }

    /// The data kernel's outputscale (with a `ScaleKernel`).
    pub fn outputscale(&self) -> Option<f64> {
        let s = self.covar_module.scale?;
        Some(
            s.outputscale_constraint
                .transform(self.params.raw_outputscale?),
        )
    }

    /// The task kernel's lengthscales.
    pub fn task_lengthscale(&self) -> Vector {
        self.params
            .task_raw_lengthscale
            .map(|r| self.task_covar_module.lengthscale_constraint.transform(r))
    }

    /// The task kernel's outputscale (with a `ScaleKernel`).
    pub fn task_outputscale(&self) -> Option<f64> {
        let s = self.task_covar_module.scale?;
        Some(
            s.outputscale_constraint
                .transform(self.params.task_raw_outputscale?),
        )
    }

    /// The stored embedding weights (BoTorch's `emb_layers.{i}.weight`,
    /// possibly not yet renormalized).
    pub fn embedding_weights(&self) -> &[Matrix] {
        &self.params.embedding_weights
    }

    /// The rows of every embedding layer that are looked up.
    fn looked_up_rows(&self, layer: usize) -> Vec<usize> {
        self.context_cat.iter().map(|r| r[layer]).collect()
    }

    /// The embedding weights after the renormalization of a forward pass.
    fn renormed_weights(&self, weights: &[Matrix]) -> Vec<Matrix> {
        weights
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let mut w = w.clone();
                embedding_renorm(&mut w, &self.looked_up_rows(i), EMBEDDING_MAX_NORM);
                w
            })
            .collect()
    }

    /// Applies the in-place renormalization of BoTorch's forward pass to the
    /// stored embedding weights (the model's predictions do not change).
    pub fn renorm_embeddings(&mut self) {
        let renormed = self.renormed_weights(&self.params.embedding_weights);
        if renormed != self.params.embedding_weights {
            // The cache depends on the effective (renormalized) weights only.
            self.params.embedding_weights = renormed;
        }
    }

    /// The embeddings of every context row (`K' × D`: the renormalized
    /// learned embeddings followed by the fixed embeddings), BoTorch's
    /// `_task_embeddings()`.
    pub fn task_embeddings(&self) -> Matrix {
        self.embeddings_from(&self.renormed_weights(&self.params.embedding_weights))
    }

    fn embeddings_from(&self, renormed: &[Matrix]) -> Matrix {
        let rows = self.context_cat.len();
        let dim = self.embedding_dim();
        let mut e = Matrix::zeros(rows, dim);
        for (a, idx) in self.context_cat.iter().enumerate() {
            let mut col = 0;
            for (w, &r) in renormed.iter().zip(idx) {
                for j in 0..w.ncols() {
                    e[(a, col + j)] = w[(r, j)];
                }
                col += w.ncols();
            }
            if let Some(f) = &self.context_emb {
                for j in 0..f.ncols() {
                    e[(a, col + j)] = f[(a, j)];
                }
            }
        }
        e
    }

    /// The task kernel with the task hyperparameters of `p`.
    fn task_kernel(&self, p: &LcemParameters) -> Result<Kernel> {
        self.task_covar_module
            .build(p.task_raw_lengthscale.as_slice(), p.task_raw_outputscale)
    }

    /// The task covariance over the context rows (`K' × K'`), BoTorch's
    /// `_eval_context_covar()`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for invalid task-kernel
    /// hyperparameters.
    pub fn task_covariance(&self) -> Result<Matrix> {
        Ok(self.cache()?.task_cov.clone())
    }

    // ------------------------------------------------------------------
    // Marginal log likelihood and fitting
    // ------------------------------------------------------------------

    /// The exact marginal log likelihood (divided by `n`, priors included) at
    /// the current hyperparameters.
    ///
    /// # Errors
    /// [`BotorchError::NotPsd`] / [`BotorchError::Nan`] if the covariance
    /// cannot be factorized.
    pub fn mll_value(&self) -> Result<f64> {
        self.evaluate_mll(&self.params, false).map(|(v, _)| v)
    }

    /// The exact MLL and its gradient with respect to the flat raw
    /// hyperparameters at `raw` (the gradient with respect to the embedding
    /// weights is torch's: at the renormalized weights, see the [module
    /// documentation](self)).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of parameters,
    /// [`BotorchError::InvalidArgument`] for invalid values and
    /// [`BotorchError::NotPsd`] / [`BotorchError::Nan`] if the covariance
    /// cannot be factorized.
    pub fn mll_value_and_grad(&self, raw: &[f64]) -> Result<(f64, Vector)> {
        let p = self.params.with_values(raw)?;
        if raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "non-finite raw hyperparameters: {raw:?}"
            )));
        }
        self.evaluate_mll(&p, true)
            .map(|(v, g)| (v, g.expect("gradient requested")))
    }

    fn evaluate_mll(&self, p: &LcemParameters, with_grad: bool) -> Result<(f64, Option<Vector>)> {
        let n = self.num_train();
        let nf = n as f64;
        let k_rows = self.context_cat.len();
        let lc = self.task_covar_module.lengthscale_constraint;
        if p.task_raw_lengthscale
            .iter()
            .any(|&r| lc.transform(r) == 0.0)
        {
            // GPyTorch divides the embeddings by the lengthscales: a zero
            // lengthscale (the lower bound of `Interval(0, 2)`) makes the task
            // covariance NaN, and BoTorch's closure returns NaN.
            return Err(BotorchError::Nan {
                count: k_rows * k_rows,
                total: k_rows * k_rows,
            });
        }
        // Hyperparameters from raw values may underflow to zero; GPyTorch's
        // NaN entries are injected below (`training_nan`).
        let kt = self
            .task_covar_module
            .build_unchecked(p.task_raw_lengthscale.as_slice(), p.task_raw_outputscale)?;
        let kd = self
            .covar_module
            .build_unchecked(p.raw_lengthscale.as_slice(), p.raw_outputscale)?;
        let renormed = self.renormed_weights(&p.embedding_weights);
        let e = self.embeddings_from(&renormed);
        let es = kt.scale_points(&e);
        let mut kernel_cov = kt.symmetric_scaled(&es);
        // The embeddings are learned (GPyTorch's autograd path).
        kt.training_nan(&e, true).inject(&mut kernel_cov);
        let w_ctx = p.raw_context_specific_weight;
        let task_cov = mix_context_covariance(kernel_cov.clone(), w_ctx);
        let xs = kd.scale_points(&self.x_data);
        let mut kdata = kd.symmetric_scaled(&xs);
        let data_nan = kd.training_nan(&self.x_data, false);
        data_nan.inject(&mut kdata);
        let noise = p
            .raw_noise
            .map(|r| self.likelihood.noise_constraint.transform(r));
        let tasks = &self.train_tasks;
        let mut k = Matrix::from_fn(n, n, |i, j| kdata[(i, j)] * task_cov[(tasks[i], tasks[j])]);
        for i in 0..n {
            k[(i, i)] += self.point_noise[i].unwrap_or(noise[tasks[i]]);
        }
        let chol = psd_safe_cholesky(&k)?;
        let resid = Vector::from_fn(n, |i, _| self.y_tf[i] - p.raw_constant[tasks[i]]);
        let v = chol.solve_lower_vec(&resid);
        let mut res = -0.5 * (v.dot(&v) + chol.logdet() + nf * LN_2PI);

        // Prior terms in GPyTorch's `named_priors()` order.
        if let Some(prior) = self.likelihood.noise_prior {
            res += noise.iter().map(|&s| prior.log_prob(s)).sum::<f64>();
        }
        res += kernel_log_prior(&self.covar_module, &kd);
        res += kernel_log_prior(&self.task_covar_module, &kt);
        let value = res / nf;
        if !with_grad {
            return Ok((value, None));
        }

        // dmll/dθ = [½ tr(W ∂K/∂θ) + ∂ log p/∂θ] / n with W = ααᵀ - K⁻¹.
        let alpha = chol.solve_lower_transpose_vec(&v);
        let mut w = chol.inverse();
        w.ger(1.0, &alpha, &alpha, -1.0);
        let kt_count = self.context_cat.len();
        let mut grad = Vec::with_capacity(self.params.to_vector().len());
        // Noise.
        let mut gnoise = vec![0.0; self.num_tasks()];
        let mut gconst = vec![0.0; self.num_tasks()];
        for i in 0..n {
            if self.point_noise[i].is_none() {
                gnoise[tasks[i]] += 0.5 * w[(i, i)];
            }
            gconst[tasks[i]] += alpha[i];
        }
        for (t, g) in gnoise.iter().enumerate() {
            let mut g = *g;
            if let Some(prior) = self.likelihood.noise_prior {
                g += prior.grad_log_prob(noise[t]);
            }
            grad.push(
                g * self
                    .likelihood
                    .noise_constraint
                    .transform_derivative(p.raw_noise[t])
                    / nf,
            );
        }
        // dmll/dT_ab = ½ G_ab with G_ab = Σ_{t_i = a, t_j = b} W_ij K_x,ij.
        let mut g = Matrix::zeros(kt_count, kt_count);
        for i in 0..n {
            for j in 0..n {
                g[(tasks[i], tasks[j])] += w[(i, j)] * kdata[(i, j)];
            }
        }
        // The context-specific weight: ∂T/∂w = I - K_e.
        if let Some(wv) = w_ctx {
            let mut gw = 0.0;
            for a in 0..kt_count {
                for b in 0..kt_count {
                    let delta = if a == b { 1.0 } else { 0.0 };
                    gw += g[(a, b)] * (delta - kernel_cov[(a, b)]);
                }
            }
            let c = context_specific_weight_constraint();
            grad.push(0.5 * gw * c.transform_derivative(wv) / nf);
            // The task kernel enters as (1 - w) · K_e.
            g *= 1.0 - wv;
        }
        grad.extend(gconst.iter().map(|g| g / nf));
        // Data kernel: tr(W ∂(K_x ∘ T)/∂θ) = tr((W ∘ T) ∂K_x/∂θ).
        let wt = Matrix::from_fn(n, n, |i, j| w[(i, j)] * task_cov[(tasks[i], tasks[j])]);
        let (gl, gs) = kd.contract_hyper_grad(&xs, &wt);
        push_kernel_grad(
            &mut grad,
            &self.covar_module,
            &kd,
            &gl,
            gs,
            p.raw_outputscale,
            &p.raw_lengthscale,
            nf,
        );
        let ls_start = grad.len() - p.raw_lengthscale.len();
        for &j in data_nan.nan_gradients() {
            grad[ls_start + j] = f64::NAN;
        }
        // Embeddings: dmll/dE_a = ½ Σ_b (G_ab + G_ba) ∂₁k_e(E_a, E_b), at the
        // renormalized weights (torch's gradient of the looked-up rows).
        let mut ge = Matrix::zeros(kt_count, self.embedding_dim());
        for a in 0..kt_count {
            let (_, dk) = kt.cross_with_grad_scaled(es.row(a), &es);
            for b in 0..kt_count {
                let c = 0.5 * (g[(a, b)] + g[(b, a)]);
                if c != 0.0 {
                    for m in 0..self.embedding_dim() {
                        ge[(a, m)] += c * dk[(b, m)];
                    }
                }
            }
        }
        let mut col = 0;
        for (layer, wgt) in p.embedding_weights.iter().enumerate() {
            let mut gw = Matrix::zeros(wgt.nrows(), wgt.ncols());
            for (a, idx) in self.context_cat.iter().enumerate() {
                for j in 0..wgt.ncols() {
                    gw[(idx[layer], j)] += ge[(a, col + j)];
                }
            }
            for r in 0..gw.nrows() {
                grad.extend(gw.row(r).iter().map(|v| v / nf));
            }
            col += wgt.ncols();
        }
        // Task kernel hyperparameters.
        let (gtl, gts) = kt.contract_hyper_grad(&es, &g);
        push_kernel_grad(
            &mut grad,
            &self.task_covar_module,
            &kt,
            &gtl,
            gts,
            p.task_raw_outputscale,
            &p.task_raw_lengthscale,
            nf,
        );
        Ok((value, Some(Vector::from_vec(grad))))
    }

    /// BoTorch's `sample_all_priors`: the current raw parameters with every
    /// hyperparameter that has a prior drawn from it.
    fn sample_all_priors(&self, initial: &LcemParameters, rng: &mut StdRng) -> Vec<f64> {
        let mut p = initial.clone();
        if let Some(prior) = self.likelihood.noise_prior {
            for v in p.raw_noise.iter_mut() {
                *v = self
                    .likelihood
                    .noise_constraint
                    .inverse_transform(prior.sample(rng));
            }
        }
        sample_kernel_priors(
            &self.covar_module,
            &mut p.raw_outputscale,
            &mut p.raw_lengthscale,
            rng,
        );
        sample_kernel_priors(
            &self.task_covar_module,
            &mut p.task_raw_outputscale,
            &mut p.task_raw_lengthscale,
            rng,
        );
        p.to_vector().as_slice().to_vec()
    }

    // ------------------------------------------------------------------
    // Prediction
    // ------------------------------------------------------------------

    fn compute_cache(&self) -> Result<LcemCache> {
        let p = &self.params;
        let kernel = self
            .covar_module
            .build(p.raw_lengthscale.as_slice(), p.raw_outputscale)?;
        let kt = self.task_kernel(p)?;
        let e = self.task_embeddings();
        let task_cov = mix_context_covariance(
            kt.symmetric_scaled(&kt.scale_points(&e)),
            p.raw_context_specific_weight,
        );
        let train_scaled = kernel.scale_points(&self.x_data);
        let kdata = kernel.symmetric_scaled(&train_scaled);
        let n = self.num_train();
        let tasks = &self.train_tasks;
        let noise = self.noise();
        let mut k = Matrix::from_fn(n, n, |i, j| kdata[(i, j)] * task_cov[(tasks[i], tasks[j])]);
        for i in 0..n {
            k[(i, i)] += self.point_noise[i].unwrap_or(noise[tasks[i]]);
        }
        let chol = psd_safe_cholesky(&k)?;
        let resid = Vector::from_fn(n, |i, _| self.y_tf[i] - p.raw_constant[tasks[i]]);
        let alpha = chol.solve_vec(&resid);
        Ok(LcemCache {
            kernel,
            train_scaled,
            task_cov,
            noise,
            chol: Arc::new(chol),
            alpha,
        })
    }

    fn cache(&self) -> Result<Arc<LcemCache>> {
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

    fn check_x(&self, x: &Matrix) -> Result<()> {
        if x.ncols() != self.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected inputs with {} columns (the non-task features), got {}",
                self.dim(),
                x.ncols()
            )));
        }
        Ok(())
    }

    /// The output-major evaluation pairs `(row, task)` of `q` rows and the
    /// output tasks.
    fn output_pairs(&self, q: usize) -> Vec<(usize, usize)> {
        self.output_tasks
            .iter()
            .flat_map(|&t| (0..q).map(move |p| (p, t)))
            .collect()
    }

    /// Test-train covariances of the pairs (`N × n`).
    fn pair_cross(&self, c: &LcemCache, kxs: &Matrix, pairs: &[(usize, usize)]) -> Matrix {
        let tasks = &self.train_tasks;
        Matrix::from_fn(pairs.len(), self.num_train(), |a, i| {
            let (r, t) = pairs[a];
            kxs[(r, i)] * c.task_cov[(t, tasks[i])]
        })
    }

    /// The joint posterior over evaluation pairs `(row of x, task index)` in
    /// the original outcome space.
    fn joint_pairs(
        &self,
        x: &Matrix,
        pairs: &[(usize, usize)],
        observation_noise: bool,
    ) -> Result<(Vector, Matrix)> {
        let c = self.cache()?;
        let xs = c.kernel.scale_points(x);
        let kxs = c.kernel.cross_scaled(&xs, &c.train_scaled);
        let kxx = c.kernel.symmetric_scaled(&xs);
        let kstar = self.pair_cross(&c, &kxs, pairs);
        let mean_tf = Vector::from_fn(pairs.len(), |a, _| {
            self.params.raw_constant[pairs[a].1] + kstar.row(a).dot(&c.alpha.transpose())
        });
        let mut cov = Matrix::from_fn(pairs.len(), pairs.len(), |a, b| {
            let ((ra, ta), (rb, tb)) = (pairs[a], pairs[b]);
            kxx[(ra, rb)] * c.task_cov[(ta, tb)]
        });
        let mut v = kstar.transpose();
        c.chol.solve_lower_mut(&mut v);
        cov.gemm_tr(-1.0, &v, &v, 1.0);
        if observation_noise {
            for (a, &(_, t)) in pairs.iter().enumerate() {
                cov[(a, a)] += c.noise[t];
            }
        }
        let (mu, sd) = self.outcome_affine();
        Ok((mean_tf.map(|m| mu + sd * m), cov.map(|v| sd * v * sd)))
    }

    /// Per row of `x`, the derivatives `∂k_x(x_r, X_train)/∂x_r` (`n × d`).
    fn row_cross_grads(&self, c: &LcemCache, xs: &ScaledPoints, rows: usize) -> Vec<Matrix> {
        (0..rows)
            .map(|r| {
                c.kernel
                    .cross_with_grad_scaled(xs.row(r), &c.train_scaled)
                    .1
            })
            .collect()
    }

    /// [`LCEMGP::joint_pairs`] (without observation noise) with derivatives
    /// with respect to every coordinate of the rows of `x` (index
    /// `r·d + k`).
    fn joint_pairs_with_grad(
        &self,
        x: &Matrix,
        pairs: &[(usize, usize)],
        num_outputs: usize,
    ) -> Result<JointPosteriorGrad> {
        let c = self.cache()?;
        let (q, d) = x.shape();
        let nn = pairs.len();
        let tasks = &self.train_tasks;
        let xs = c.kernel.scale_points(x);
        let kxs = c.kernel.cross_scaled(&xs, &c.train_scaled);
        let kxx = c.kernel.symmetric_scaled(&xs);
        let kstar = self.pair_cross(&c, &kxs, pairs);
        let mean_tf = Vector::from_fn(nn, |a, _| {
            self.params.raw_constant[pairs[a].1] + kstar.row(a).dot(&c.alpha.transpose())
        });
        let mut cov = Matrix::from_fn(nn, nn, |a, b| {
            let ((ra, ta), (rb, tb)) = (pairs[a], pairs[b]);
            kxx[(ra, rb)] * c.task_cov[(ta, tb)]
        });
        let mut v = kstar.transpose();
        c.chol.solve_lower_mut(&mut v);
        cov.gemm_tr(-1.0, &v, &v, 1.0);
        let mut w = v;
        c.chol.solve_lower_transpose_mut(&mut w); // K⁻¹ kstarᵀ (n × N)
        let (mu, sd) = self.outcome_affine();
        let dks = self.row_cross_grads(&c, &xs, q);
        let dkxx: Vec<Matrix> = (0..q)
            .map(|r| c.kernel.cross_with_grad_scaled(xs.row(r), &xs).1)
            .collect();
        let mut dmean = vec![Vector::zeros(nn); q * d];
        let mut dcov = vec![Matrix::zeros(nn, nn); q * d];
        for (a, &(r, t)) in pairs.iter().enumerate() {
            // Dk_a (n × d): rows T[t, t_i] ∂k_x(x_r, x_i)/∂x_r.
            let dk = Matrix::from_fn(self.num_train(), d, |i, m| {
                c.task_cov[(t, tasks[i])] * dks[r][(i, m)]
            });
            let dk_alpha = dk.tr_mul(&c.alpha);
            let dk_w = dk.tr_mul(&w); // d × N
            for m in 0..d {
                let j = r * d + m;
                dmean[j][a] += sd * dk_alpha[m];
                for (b, &(rb, tb)) in pairs.iter().enumerate() {
                    let g = (c.task_cov[(t, tb)] * dkxx[r][(rb, m)] - dk_w[(m, b)]) * sd * sd;
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

    /// Reverse-mode derivative of [`LCEMGP::joint_pairs`] (without
    /// observation noise) for the rows `rows` of `x`.
    fn joint_pairs_vjp(
        &self,
        x: &Matrix,
        pairs: &[(usize, usize)],
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        let c = self.cache()?;
        let d = x.ncols();
        let tasks = &self.train_tasks;
        let xs = c.kernel.scale_points(x);
        let kxs = c.kernel.cross_scaled(&xs, &c.train_scaled);
        let kstar = self.pair_cross(&c, &kxs, pairs);
        let (_, sd) = self.outcome_affine();
        let mbar = mean_adjoint * sd;
        let abar = (covariance_adjoint + covariance_adjoint.transpose()) * (0.5 * sd * sd);
        let mut w = kstar.transpose();
        c.chol.solve_lower_mut(&mut w);
        c.chol.solve_lower_transpose_mut(&mut w); // K⁻¹ kstarᵀ (n × N)
        let mut out = Matrix::zeros(rows.len(), d);
        for (row_out, r) in rows.clone().enumerate() {
            let (_, dkr) = c.kernel.cross_with_grad_scaled(xs.row(r), &c.train_scaled);
            let (_, dkk) = c.kernel.cross_with_grad_scaled(xs.row(r), &xs);
            for (a, &(ra, t)) in pairs.iter().enumerate() {
                if ra != r {
                    continue;
                }
                let a_col = abar.column(a).into_owned();
                // ∂/∂x_r: Dk_aᵀ (mbar_a α - 2 W abar[:, a])
                //       + 2 Σ_b abar[a, b] T[t, t_b] ∂₁k_x(x_r, x_{r_b}).
                let vvec = &c.alpha * mbar[a] - (&w * &a_col) * 2.0;
                for m in 0..d {
                    let mut g = 0.0;
                    for i in 0..self.num_train() {
                        g += c.task_cov[(t, tasks[i])] * dkr[(i, m)] * vvec[i];
                    }
                    for (b, &(rb, tb)) in pairs.iter().enumerate() {
                        g += 2.0 * a_col[b] * c.task_cov[(t, tb)] * dkk[(rb, m)];
                    }
                    out[(row_out, m)] += g;
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
        self.check_x(x)?;
        let pairs = self.output_pairs(x.nrows());
        let (mean, cov) = self.joint_pairs(x, &pairs, observation_noise)?;
        Ok(Posterior::multi_output(mean, cov, self.output_tasks.len()))
    }

    /// The joint posterior at full inputs `x_full` (`q × (d + 1)`, with task
    /// values in the task column): a single-output posterior over the `q`
    /// points, BoTorch's `model.posterior(X)` for `X` that includes the task
    /// feature.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a column mismatch, invalid
    /// task values (see [`TaskLayout::task_index`]) and
    /// [`BotorchError::NotPsd`] if the training covariance cannot be
    /// factorized.
    pub fn posterior_with_tasks(
        &self,
        x_full: &Matrix,
        observation_noise: bool,
    ) -> Result<Posterior> {
        let (x, tasks) = self.layout.split_inputs(x_full)?;
        let pairs: Vec<_> = tasks.into_iter().enumerate().collect();
        let (mean, cov) = self.joint_pairs(&x, &pairs, observation_noise)?;
        Ok(Posterior::single_output(mean, cov))
    }

    /// Condition on full inputs and original-space targets, retaining the
    /// fitted outcome transform, context embeddings and hyperparameters.
    /// Explicit variances are in transformed outcome space, matching the
    /// multitask conditioning protocol. All returned models share one factor.
    ///
    /// # Errors
    /// Returns shape/task validation errors, an input error for nonfinite
    /// observations or negative noise, and factorization errors for invalid
    /// covariance matrices.
    pub fn condition_on_observations_many(
        &self,
        x: &Matrix,
        ys: &[Matrix],
        noise: Option<&[f64]>,
    ) -> Result<Vec<Self>> {
        let q = x.nrows();
        let (data, tasks) = self.layout.split_inputs(x)?;
        if x.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InputData("X must be finite.".into()));
        }
        for y in ys {
            if y.shape() != (q, 1) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected Y of shape ({q}, 1), got {:?}",
                    y.shape()
                )));
            }
            if y.iter().any(|v| !v.is_finite()) {
                return Err(BotorchError::InputData("Y must be finite.".into()));
            }
        }
        if let Some(values) = noise {
            if values.len() != q {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected {q} noise variances, got {}",
                    values.len()
                )));
            }
            if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(BotorchError::InputData(
                    "Observation noise must be finite and nonnegative.".into(),
                ));
            }
        }
        if ys.is_empty() {
            return Ok(Vec::new());
        }
        let append = |a: &Matrix, b: &Matrix| {
            let mut result = Matrix::zeros(a.nrows() + b.nrows(), a.ncols());
            result.rows_mut(0, a.nrows()).copy_from(a);
            result.rows_mut(a.nrows(), b.nrows()).copy_from(b);
            result
        };
        let mut base = self.clone();
        base.train_x = append(&self.train_x, x);
        base.x_data = append(&self.x_data, &data);
        base.train_tasks.extend(tasks);
        base.point_noise
            .extend((0..q).map(|i| noise.map(|values| values[i])));
        base.cache = OnceLock::new();
        let mut shared: Option<LcemCache> = None;
        ys.iter()
            .map(|y| {
                let mut model = base.clone();
                model.train_y = append(&self.train_y, y);
                let transformed = match &self.outcome_transform {
                    Some(transform) => transform.transform(y, None)?.0,
                    None => y.clone(),
                };
                let mut target = self.y_tf.as_slice().to_vec();
                target.extend(transformed.column(0).iter().copied());
                model.y_tf = Vector::from_vec(target);
                let cache = match &shared {
                    Some(cache) => {
                        let residual = Vector::from_fn(model.num_train(), |i, _| {
                            model.y_tf[i] - model.params.raw_constant[model.train_tasks[i]]
                        });
                        LcemCache {
                            alpha: cache.chol.solve_vec(&residual),
                            ..cache.clone()
                        }
                    }
                    None => {
                        let cache = model.compute_cache()?;
                        shared = Some(cache.clone());
                        cache
                    }
                };
                model.cache = OnceLock::from(Ok(Arc::new(cache)));
                Ok(model)
            })
            .collect()
    }

    /// Single observation-set form of [`Self::condition_on_observations_many`].
    ///
    /// # Errors
    /// See [`Self::condition_on_observations_many`].
    pub fn condition_on_observations(
        &self,
        x: &Matrix,
        y: &Matrix,
        noise: Option<&[f64]>,
    ) -> Result<Self> {
        Ok(self
            .condition_on_observations_many(x, std::slice::from_ref(y), noise)?
            .pop()
            .expect("one observation set"))
    }
}

/// The view of an [`LCEMGP`] whose inputs include the task column (`d + 1`
/// columns, a single output): BoTorch's `model.posterior(X)` for `X` with
/// task values. Derivatives with respect to the task column are zero.
#[derive(Clone, Debug)]
pub struct LcemTaskInputs {
    model: Arc<LCEMGP>,
}

impl LcemTaskInputs {
    /// The view of `model`.
    pub fn new(model: Arc<LCEMGP>) -> Self {
        Self { model }
    }

    fn split(&self, x: &Matrix) -> Result<(Matrix, Vec<(usize, usize)>)> {
        let (xd, tasks) = self.model.layout.split_inputs(x)?;
        Ok((xd, tasks.into_iter().enumerate().collect()))
    }

    /// Maps derivatives with respect to the data features (`r × d`) to the
    /// full inputs (`r × (d + 1)`, zero for the task column).
    fn expand(&self, g: &Matrix) -> Matrix {
        let tf = self.model.layout.task_feature();
        Matrix::from_fn(g.nrows(), g.ncols() + 1, |i, j| match j.cmp(&tf) {
            std::cmp::Ordering::Less => g[(i, j)],
            std::cmp::Ordering::Equal => 0.0,
            std::cmp::Ordering::Greater => g[(i, j - 1)],
        })
    }
}

impl Model for LcemTaskInputs {
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
        let (xd, pairs) = self.split(x)?;
        let g = self.model.joint_pairs_with_grad(&xd, &pairs, 1)?;
        let (q, d) = xd.shape();
        let tf = self.model.layout.task_feature();
        let n = g.mean.len();
        let mut dmean = Vec::with_capacity(q * (d + 1));
        let mut dcov = Vec::with_capacity(q * (d + 1));
        for r in 0..q {
            for j in 0..=d {
                if j == tf {
                    dmean.push(Vector::zeros(n));
                    dcov.push(Matrix::zeros(n, n));
                } else {
                    let k = if j < tf { j } else { j - 1 };
                    dmean.push(g.dmean[r * d + k].clone());
                    dcov.push(g.dcov[r * d + k].clone());
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean: g.mean,
            covariance: g.covariance,
            dmean,
            dcov,
            num_outputs: 1,
        })
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        check_vjp_shapes_expanded(x, &rows, mean_adjoint, covariance_adjoint, 1, 1)?;
        let (xd, pairs) = self.split(x)?;
        let g = self
            .model
            .joint_pairs_vjp(&xd, &pairs, rows, mean_adjoint, covariance_adjoint)?;
        Ok(self.expand(&g))
    }
}

impl Model for LCEMGP {
    fn dim(&self) -> usize {
        LCEMGP::dim(self)
    }

    fn num_outputs(&self) -> usize {
        self.output_tasks.len()
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        LCEMGP::posterior(self, x, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.check_x(x)?;
        let pairs = self.output_pairs(x.nrows());
        self.joint_pairs_with_grad(x, &pairs, self.output_tasks.len())
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        self.check_x(x)?;
        check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            self.output_tasks.len(),
            1,
        )?;
        let pairs = self.output_pairs(x.nrows());
        self.joint_pairs_vjp(x, &pairs, rows, mean_adjoint, covariance_adjoint)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.check_x(x)?;
        let c = self.cache()?;
        let b = x.nrows();
        let m = self.output_tasks.len();
        let xs = c.kernel.scale_points(x);
        let kxs = c.kernel.cross_scaled(&xs, &c.train_scaled);
        let prior = c.kernel.prior_variance();
        let tasks = &self.train_tasks;
        let (mu, sd) = self.outcome_affine();
        let mut mean = Matrix::zeros(b, m);
        let mut vs = Vec::with_capacity(m);
        for (o, &t) in self.output_tasks.iter().enumerate() {
            let kt = Matrix::from_fn(b, self.num_train(), |p, i| {
                kxs[(p, i)] * c.task_cov[(t, tasks[i])]
            });
            for p in 0..b {
                mean[(p, o)] =
                    mu + sd * (self.params.raw_constant[t] + kt.row(p).dot(&c.alpha.transpose()));
            }
            let mut v = kt.transpose();
            c.chol.solve_lower_mut(&mut v);
            vs.push(v);
        }
        let mut covariance = Matrix::zeros(b, m * m);
        for p in 0..b {
            for (o, &t) in self.output_tasks.iter().enumerate() {
                for (u, &tu) in self.output_tasks.iter().enumerate() {
                    let cov = prior * c.task_cov[(t, tu)] - vs[o].column(p).dot(&vs[u].column(p));
                    covariance[(p, o * m + u)] = sd * cov * sd;
                }
            }
        }
        Ok(PointPosteriors { mean, covariance })
    }
}

impl Fittable for LCEMGP {
    type Output = FitResult;

    /// BoTorch's `fit_gpytorch_mll` (`fit_gpytorch_mll_scipy`, one L-BFGS-B
    /// problem over all hyperparameters, with retries from prior samples);
    /// the fitted embedding weights are L-BFGS-B's final iterate, as in
    /// BoTorch (see the [module documentation](self)).
    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        let initial = self.params.clone();
        let x_init = initial.to_vector().as_slice().to_vec();
        let bounds = self.raw_parameter_bounds();
        let (result, x) = {
            let model: &LCEMGP = self;
            minimize_with_retries(
                &x_init,
                &bounds,
                options,
                |x| {
                    model
                        .mll_value_and_grad(x)
                        .map(|(v, g)| (v, g.as_slice().to_vec()))
                },
                |rng| model.sample_all_priors(&initial, rng),
            )?
        };
        self.set_raw_parameters(&x)?;
        Ok(result)
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// The constraint of the context-specific weight of the provided-embedding
/// variant: `Interval(0.0, 1.0, transform=None, initial_value=0.0)`.
pub fn context_specific_weight_constraint() -> Interval {
    Interval::new(0.0, 1.0, None, None, Some(0.0))
        .expect("valid interval")
        .rounded_to_f32()
}

/// The context covariance `(1 - w) · K + w · I` (`K` without a weight).
fn mix_context_covariance(k: Matrix, w: Option<f64>) -> Matrix {
    match w {
        None => k,
        Some(w) => {
            let n = k.nrows();
            Matrix::from_fn(n, n, |a, b| {
                (1.0 - w) * k[(a, b)] + w * if a == b { 1.0 } else { 0.0 }
            })
        }
    }
}

/// The learned context features of BoTorch's `LCEMGP`: the embedding index
/// of every categorical feature per context row, the shapes of the embedding
/// layers, the fixed embeddings and the total embedding dimension.
type ContextFeatures = (Vec<Vec<usize>>, Vec<(usize, usize)>, Option<Matrix>, usize);

fn learned_context_features(
    layout: &TaskLayout,
    context_cat_feature: Option<Matrix>,
    context_emb_feature: Option<Matrix>,
    embs_dim_list: Option<Vec<usize>>,
    all_tasks: Option<&[i64]>,
) -> Result<ContextFeatures> {
    let k_tasks = layout.num_tasks();
    // Default: the task values, in the order of `all_tasks` as given (BoTorch).
    let ccf = match context_cat_feature {
        Some(c) => c,
        None => {
            let values: Vec<f64> = match all_tasks {
                Some(a) if !a.is_empty() => a.iter().map(|&t| t as f64).collect(),
                _ => layout.all_tasks().iter().map(|&t| t as f64).collect(),
            };
            Matrix::from_column_slice(values.len(), 1, &values)
        }
    };
    let rows = ccf.nrows();
    let k = ccf.ncols();
    if k == 0 {
        return Err(BotorchError::DimensionMismatch(
            "context_cat_feature must have at least one column".into(),
        ));
    }
    if rows < k_tasks {
        return Err(BotorchError::DimensionMismatch(format!(
            "context_cat_feature has {rows} rows but the model has {k_tasks} tasks (one row per \
             task index)"
        )));
    }
    let embs_dim_list = embs_dim_list.unwrap_or_else(|| vec![1; k]);
    if embs_dim_list.len() != k || embs_dim_list.contains(&0) {
        return Err(BotorchError::InvalidArgument(format!(
            "embs_dim_list must have one positive embedding dimension per categorical feature \
             ({k}), got {embs_dim_list:?}"
        )));
    }
    let mut context_cat = vec![Vec::with_capacity(k); rows];
    let mut embedding_shapes = Vec::with_capacity(k);
    for (i, &dim) in embs_dim_list.iter().enumerate() {
        let values = ccf
            .column(i)
            .iter()
            .map(|&v| index_value(v, "context_cat_feature"))
            .collect::<Result<Vec<_>>>()?;
        let mut unique = values.clone();
        unique.sort_unstable();
        unique.dedup();
        let num = unique.len();
        for (r, &v) in values.iter().enumerate() {
            if v < 0 || v as usize >= num {
                return Err(BotorchError::InputData(format!(
                    "context_cat_feature[:, {i}] contains the index {v}, but its embedding has \
                     {num} rows (one per distinct value; indices must be in 0..{num})"
                )));
            }
            context_cat[r].push(v as usize);
        }
        embedding_shapes.push((num, dim));
    }
    let context_emb = match context_emb_feature {
        Some(f) => {
            if f.nrows() != rows {
                return Err(BotorchError::DimensionMismatch(format!(
                    "context_emb_feature has {} rows, but context_cat_feature has {rows}",
                    f.nrows()
                )));
            }
            check_no_nan(&f, "context_emb_feature")?;
            (f.ncols() > 0).then_some(f)
        }
        None => None,
    };
    let n_embs: usize = embs_dim_list.iter().sum();
    let emb_dim = n_embs + context_emb.as_ref().map_or(0, |f| f.ncols());
    Ok((context_cat, embedding_shapes, context_emb, emb_dim))
}

fn check_task_kernel(spec: &KernelSpec, emb_dim: usize) -> Result<()> {
    if spec.ard_num_dims != emb_dim {
        return Err(BotorchError::DimensionMismatch(format!(
            "The task kernel has ard_num_dims = {} but the context embeddings have {emb_dim} \
             dimensions (learned plus fixed)",
            spec.ard_num_dims
        )));
    }
    Ok(())
}

fn check_embedding_weights(ws: &[Matrix], shapes: &[(usize, usize)]) -> Result<()> {
    if ws.len() != shapes.len() || ws.iter().zip(shapes).any(|(w, &s)| w.shape() != s) {
        return Err(BotorchError::DimensionMismatch(format!(
            "expected embedding weights of shapes {shapes:?}, got {:?}",
            ws.iter().map(|w| w.shape()).collect::<Vec<_>>()
        )));
    }
    Ok(())
}

fn kernel_shapes(out: &mut Vec<(String, Vec<usize>)>, prefix: &str, spec: &KernelSpec) {
    if spec.scale.is_some() {
        out.push((format!("{prefix}.raw_outputscale"), vec![]));
        out.push((
            format!("{prefix}.base_kernel.raw_lengthscale"),
            vec![1, spec.ard_num_dims],
        ));
    } else {
        out.push((
            format!("{prefix}.raw_lengthscale"),
            vec![1, spec.ard_num_dims],
        ));
    }
}

fn kernel_bounds(out: &mut Vec<(f64, f64)>, spec: &KernelSpec) {
    if let Some(s) = &spec.scale {
        out.push(s.outputscale_constraint.raw_bounds());
    }
    let ls = spec.lengthscale_constraint.raw_bounds();
    out.extend(std::iter::repeat_n(ls, spec.ard_num_dims));
}

/// The log priors of a kernel's outputscale and lengthscales (GPyTorch's
/// `named_priors()` order).
fn kernel_log_prior(spec: &KernelSpec, kernel: &Kernel) -> f64 {
    let mut res = 0.0;
    if let (Some(scale), Some(s)) = (spec.scale, kernel.outputscale()) {
        if let Some(prior) = scale.outputscale_prior {
            res += prior.log_prob(s);
        }
    }
    if let Some(prior) = spec.lengthscale_prior {
        res += kernel
            .lengthscale()
            .iter()
            .map(|&l| prior.log_prob(l))
            .sum::<f64>();
    }
    res
}

/// Appends the gradient of a kernel's raw outputscale and lengthscales from
/// the contracted derivatives `gl` / `gs` (of `½ tr(W ∂K/∂θ)` up to the
/// factor ½) and the log priors, divided by `n`.
#[allow(clippy::too_many_arguments)]
fn push_kernel_grad(
    grad: &mut Vec<f64>,
    spec: &KernelSpec,
    kernel: &Kernel,
    gl: &Vector,
    gs: Option<f64>,
    raw_outputscale: Option<f64>,
    raw_lengthscale: &Vector,
    nf: f64,
) {
    if let (Some(scale), Some(s), Some(gs), Some(raw)) =
        (spec.scale, kernel.outputscale(), gs, raw_outputscale)
    {
        let mut g = 0.5 * gs;
        if let Some(prior) = scale.outputscale_prior {
            g += prior.grad_log_prob(s);
        }
        grad.push(g * scale.outputscale_constraint.transform_derivative(raw) / nf);
    }
    let lengthscale = kernel.lengthscale();
    for j in 0..gl.len() {
        let mut g = 0.5 * gl[j];
        if let Some(prior) = spec.lengthscale_prior {
            g += prior.grad_log_prob(lengthscale[j]);
        }
        grad.push(
            g * spec
                .lengthscale_constraint
                .transform_derivative(raw_lengthscale[j])
                / nf,
        );
    }
}

fn sample_kernel_priors(
    spec: &KernelSpec,
    raw_outputscale: &mut Option<f64>,
    raw_lengthscale: &mut Vector,
    rng: &mut StdRng,
) {
    if let (Some(scale), Some(raw)) = (spec.scale, raw_outputscale.as_mut()) {
        if let Some(prior) = scale.outputscale_prior {
            *raw = scale
                .outputscale_constraint
                .inverse_transform(prior.sample(rng));
        }
    }
    if let Some(prior) = spec.lengthscale_prior {
        for v in raw_lengthscale.iter_mut() {
            *v = spec
                .lengthscale_constraint
                .inverse_transform(prior.sample(rng));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two contexts with data, one without; a 3-dimensional fixed embedding.
    fn model(m: usize, seed: u64) -> LCEMGP {
        let n = 9;
        let x = Matrix::from_fn(n, 3, |i, j| {
            if j == 2 {
                (i % 2) as f64
            } else {
                ((i * 7 + j * 3) % 11) as f64 / 10.0
            }
        });
        let y = Matrix::from_fn(n, 1, |i, _| {
            (3.0 * x[(i, 0)]).sin() + x[(i, 1)] * x[(i, 1)] + 0.3 * x[(i, 2)]
        });
        let emb = (m > 0).then(|| Matrix::from_fn(3, m, |a, j| 0.2 * a as f64 - 0.1 * j as f64));
        let options = LCEMGPOptions {
            output_tasks: Some(vec![2]),
            all_tasks: Some(vec![0, 1, 2]),
            context_cat_feature: Some(Matrix::from_column_slice(3, 1, &[0.0, 1.0, 2.0])),
            context_emb_feature: emb,
            embs_dim_list: Some(vec![1]),
            embedding_weights: Some(vec![Matrix::from_column_slice(3, 1, &[1.7, -0.4, 0.9])]),
            seed: Some(seed),
            ..Default::default()
        };
        let mut gp = LCEMGP::new(x, y, -1, options).unwrap();
        let mut raw = gp.raw_parameters();
        let k = gp.num_tasks();
        for t in 0..k {
            raw[t] = 0.02 + 0.01 * t as f64;
            raw[k + t] = 0.1 * t as f64 - 0.05;
        }
        let len = raw.len();
        raw[len - 1] = 0.8;
        gp.set_raw_parameters(raw.as_slice()).unwrap();
        gp
    }

    #[test]
    fn structure_follows_botorch() {
        let gp = model(3, 0);
        let names: Vec<String> = gp.parameter_names().into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names,
            vec![
                "likelihood.noise_covar.raw_noise",
                "mean_module.base_means.0.raw_constant",
                "mean_module.base_means.1.raw_constant",
                "mean_module.base_means.2.raw_constant",
                "covar_module.raw_lengthscale",
                "emb_layers.0.weight",
                "task_covar_module_base.raw_lengthscale",
            ]
        );
        assert_eq!(gp.embedding_dim(), 4);
        let b = gp.raw_parameter_bounds();
        assert_eq!(b[b.len() - 1], (0.0, 2.0));
        assert_eq!(gp.num_outputs(), 1);
        // Renormalization: |1.7| > 1 is scaled to 1.7 / (1.7 + 1e-7).
        let e = gp.task_embeddings();
        assert_eq!(e[(0, 0)], 1.7 * (1.0 / (1.7 + 1e-7)));
        assert_eq!(e[(1, 0)], -0.4);
    }

    #[test]
    fn mll_gradient_matches_finite_differences() {
        for m in [0, 3] {
            let gp = model(m, 1);
            let raw = gp.raw_parameters();
            let (_, g) = gp.mll_value_and_grad(raw.as_slice()).unwrap();
            for j in 0..raw.len() {
                let h = 1e-6;
                let mut rp = raw.clone();
                rp[j] += h;
                let mut rm = raw.clone();
                rm[j] -= h;
                let fp = gp.mll_value_and_grad(rp.as_slice()).unwrap().0;
                let fm = gp.mll_value_and_grad(rm.as_slice()).unwrap().0;
                let fd = (fp - fm) / (2.0 * h);
                // The renormalized weight (index of row 0) has torch's
                // gradient, which ignores the renormalization.
                let names = gp.parameter_names();
                let emb_start: usize = names
                    .iter()
                    .take_while(|(n, _)| !n.starts_with("emb_layers"))
                    .map(|(_, s)| s)
                    .sum();
                if j == emb_start {
                    continue;
                }
                assert!(
                    (fd - g[j]).abs() < 1e-6 * (1.0 + g[j].abs()),
                    "m={m} j={j} fd={fd} g={}",
                    g[j]
                );
            }
        }
    }

    #[test]
    fn posterior_gradients_match_finite_differences() {
        let gp = model(2, 2);
        let x = Matrix::from_row_slice(2, 2, &[0.3, 0.8, 0.55, 0.1]);
        let g = Model::joint_posterior_with_grad(&gp, &x).unwrap();
        let p = Model::posterior(&gp, &x, false).unwrap();
        assert!((&g.mean - &p.mean).amax() < 1e-14);
        assert!((&g.covariance - &p.covariance).amax() < 1e-14);
        let ma = Vector::from_vec(vec![0.7, -1.3]);
        let ca = Matrix::from_row_slice(2, 2, &[0.4, -0.2, 0.9, 1.1]);
        let vjp = gp.joint_posterior_vjp(&x, 0..2, &ma, &ca).unwrap();
        for r in 0..2 {
            for k in 0..2 {
                let h = 1e-6;
                let mut xp = x.clone();
                xp[(r, k)] += h;
                let mut xm = x.clone();
                xm[(r, k)] -= h;
                let pp = Model::posterior(&gp, &xp, false).unwrap();
                let pm = Model::posterior(&gp, &xm, false).unwrap();
                let fd_mean = (&pp.mean - &pm.mean) / (2.0 * h);
                let fd_cov = (&pp.covariance - &pm.covariance) / (2.0 * h);
                let j = r * 2 + k;
                assert!((&fd_mean - &g.dmean[j]).amax() < 1e-7);
                assert!((&fd_cov - &g.dcov[j]).amax() < 1e-7);
                let s = ma.dot(&fd_mean) + ca.component_mul(&fd_cov).sum();
                assert!((s - vjp[(r, k)]).abs() < 1e-6, "{s} {}", vjp[(r, k)]);
            }
        }
        let pp = Model::point_posteriors(&gp, &x).unwrap();
        for p_ in 0..2 {
            let single = Model::posterior(&gp, &x.rows(p_, 1).into_owned(), false).unwrap();
            assert!((pp.mean[(p_, 0)] - single.mean[0]).abs() < 1e-12);
            assert!((pp.covariance[(p_, 0)] - single.covariance[(0, 0)]).abs() < 1e-12);
        }
    }

    #[test]
    fn task_inputs_view_matches_output_task_posterior() {
        let gp = Arc::new(model(1, 3));
        let x = Matrix::from_row_slice(2, 2, &[0.3, 0.8, 0.55, 0.1]);
        let xf = Matrix::from_fn(2, 3, |i, j| if j == 2 { 2.0 } else { x[(i, j)] });
        let a = Model::posterior(gp.as_ref(), &x, true).unwrap();
        let view = LcemTaskInputs::new(Arc::clone(&gp));
        let b = view.posterior(&xf, true).unwrap();
        assert!((&a.mean - &b.mean).amax() < 1e-14);
        assert!((&a.covariance - &b.covariance).amax() < 1e-14);
        let ma = Vector::from_vec(vec![0.7, -1.3]);
        let ca = Matrix::from_row_slice(2, 2, &[0.4, -0.2, 0.9, 1.1]);
        let va = gp.joint_posterior_vjp(&x, 0..2, &ma, &ca).unwrap();
        let vb = view.joint_posterior_vjp(&xf, 0..2, &ma, &ca).unwrap();
        assert_eq!(vb[(0, 2)], 0.0);
        assert!((va[(1, 1)] - vb[(1, 1)]).abs() < 1e-14);
    }

    #[test]
    fn fit_improves_the_likelihood() {
        let mut gp = model(3, 4);
        let before = gp.mll_value().unwrap();
        let r = crate::fit_gpytorch_mll(&mut gp, FitOptions::default()).unwrap();
        assert!(-r.loss >= before);
        assert!((gp.mll_value().unwrap() + r.loss).abs() < 1e-9);
    }

    /// The provided-embedding variant with a context-specific weight.
    fn provided(w: f64) -> LCEMGP {
        let base = model(0, 5);
        let options = LCEMGPOptions {
            output_tasks: Some(vec![2]),
            all_tasks: Some(vec![0, 1, 2]),
            context_emb_feature: Some(Matrix::from_row_slice(
                3,
                2,
                &[0.9, 0.1, 0.85, 0.15, 0.85, 0.15],
            )),
            provided_embedding: true,
            ..Default::default()
        };
        let mut gp = LCEMGP::new(
            base.train_inputs().clone(),
            base.train_targets().clone(),
            -1,
            options,
        )
        .unwrap();
        let names = gp.parameter_names();
        assert_eq!(names[1].0, "raw_context_specific_weight");
        assert!(names.iter().all(|(n, _)| !n.starts_with("emb_layers")));
        let mut raw = gp.raw_parameters();
        raw[3] = w;
        raw[4] = 0.3;
        let len = raw.len();
        raw[len - 2] = 0.4;
        raw[len - 1] = 0.7;
        gp.set_raw_parameters(raw.as_slice()).unwrap();
        gp
    }

    #[test]
    fn provided_embedding_gradient_and_covariance() {
        let gp = provided(0.35);
        assert_eq!(gp.raw_parameter_bounds()[3], (0.0, 1.0));
        let t = gp.task_covariance().unwrap();
        // Contexts 1 and 2 have identical embeddings: correlation 1 - w.
        assert!((t[(1, 2)] - 0.65).abs() < 1e-15);
        assert_eq!(t[(2, 2)], 1.0);
        let raw = gp.raw_parameters();
        let (_, g) = gp.mll_value_and_grad(raw.as_slice()).unwrap();
        for j in 0..raw.len() {
            let h = 1e-6;
            let mut rp = raw.clone();
            rp[j] += h;
            let mut rm = raw.clone();
            rm[j] -= h;
            let fd = (gp.mll_value_and_grad(rp.as_slice()).unwrap().0
                - gp.mll_value_and_grad(rm.as_slice()).unwrap().0)
                / (2.0 * h);
            assert!(
                (fd - g[j]).abs() < 1e-6 * (1.0 + g[j].abs()),
                "j={j} fd={fd} g={}",
                g[j]
            );
        }
    }

    #[test]
    fn unobserved_means_take_the_observed_average() {
        let mut gp = provided(0.0);
        let before = gp.mll_value().unwrap();
        gp.eval_unobserved_means();
        let c = gp.constant().clone();
        assert_eq!(c[2], (c[0] + c[1]) / 2.0);
        assert_eq!(gp.mll_value().unwrap(), before);
        // A task kernel lengthscale at its lower bound 0 gives NaN.
        let mut raw = gp.raw_parameters();
        let len = raw.len();
        raw[len - 1] = 0.0;
        assert!(matches!(
            gp.mll_value_and_grad(raw.as_slice()),
            Err(BotorchError::Nan { .. })
        ));
    }

    #[test]
    fn conditioned_lcem_matches_dense_gaussian_and_keeps_fitted_state() {
        for explicit in [false, true] {
            let gp = model(2, 14);
            let x = Matrix::from_row_slice(2, 3, &[0.18, 0.37, 0.0, 0.74, 0.22, 2.0]);
            let test = Matrix::from_row_slice(2, 3, &[0.32, 0.58, 1.0, 0.61, 0.19, 2.0]);
            let y = Matrix::from_column_slice(2, 1, &[0.8, -0.4]);
            let noise = [0.015, 0.08];
            let mut joint_x = Matrix::zeros(4, 3);
            joint_x.rows_mut(0, 2).copy_from(&x);
            joint_x.rows_mut(2, 2).copy_from(&test);
            let joint = gp.posterior_with_tasks(&joint_x, false).unwrap();
            let sd = gp.outcome_affine().1;
            let task_noise = gp.noise();
            let mut corner = joint.covariance.view((0, 0), (2, 2)).into_owned();
            for i in 0..2 {
                corner[(i, i)] += sd
                    * sd
                    * if explicit {
                        noise[i]
                    } else {
                        task_noise[gp.layout.task_index(x[(i, 2)]).unwrap()]
                    };
            }
            let root = psd_safe_cholesky(&corner).unwrap();
            let cross = joint.covariance.view((2, 0), (2, 2)).into_owned();
            let delta = Vector::from_fn(2, |i, _| y[(i, 0)] - joint.mean[i]);
            let expected_mean =
                joint.mean.rows(2, 2).into_owned() + &cross * root.solve_vec(&delta);
            let expected_covariance = joint.covariance.view((2, 2), (2, 2)).into_owned()
                - &cross * root.solve(&cross.transpose());
            let models = gp
                .condition_on_observations_many(
                    &x,
                    &[y.clone(), &y + Matrix::from_element(2, 1, 0.2)],
                    explicit.then_some(noise.as_slice()),
                )
                .unwrap();
            let actual = models[0].posterior_with_tasks(&test, false).unwrap();
            assert!((actual.mean - expected_mean).amax() < 3e-12);
            assert!((actual.covariance - expected_covariance).amax() < 3e-12);
            assert_eq!(gp.parameters(), models[0].parameters());
            assert_eq!(gp.outcome_transform(), models[0].outcome_transform());
            assert_eq!(gp.num_train() + 2, models[0].num_train());
            assert!(Arc::ptr_eq(
                &models[0].cache().unwrap().chol,
                &models[1].cache().unwrap().chol
            ));
            assert_eq!(gp.num_train(), 9);
        }
    }

    #[test]
    fn explicit_conditioning_noise_survives_parameter_changes_and_mll_gradients() {
        let mut gp = model(1, 8);
        let mut parameters = gp.parameters().clone();
        // Stay inside max_norm: its intentional no-grad renormalization is
        // not the finite-difference law of the embedding lookup.
        for weight in &mut parameters.embedding_weights {
            *weight *= 0.4;
        }
        gp.set_parameters(parameters).unwrap();
        let x = Matrix::from_row_slice(2, 3, &[0.18, 0.37, 0.0, 0.74, 0.22, 1.0]);
        let y = Matrix::from_column_slice(2, 1, &[0.8, -0.4]);
        let conditioned = gp
            .condition_on_observations(&x, &y, Some(&[0.015, 0.08]))
            .unwrap();
        let raw = conditioned.raw_parameters();
        let (_, gradient) = conditioned.mll_value_and_grad(raw.as_slice()).unwrap();
        for j in 0..raw.len() {
            let mut plus = raw.clone();
            let mut minus = raw.clone();
            plus[j] += 1e-6;
            minus[j] -= 1e-6;
            let fd = (conditioned.mll_value_and_grad(plus.as_slice()).unwrap().0
                - conditioned.mll_value_and_grad(minus.as_slice()).unwrap().0)
                / 2e-6;
            assert!(
                (fd - gradient[j]).abs() < 2e-6,
                "parameter {j}: {fd} vs {}",
                gradient[j]
            );
        }
        assert!(gp
            .condition_on_observations(&x, &y, Some(&[-1.0, 0.08]))
            .is_err());
        assert!(gp
            .condition_on_observations(&x, &y, Some(&[f64::NAN, 0.08]))
            .is_err());
        assert!(gp.condition_on_observations(&x, &y, Some(&[0.08])).is_err());
        assert!(gp
            .condition_on_observations(&x, &Matrix::zeros(3, 1), None)
            .is_err());
    }
}
