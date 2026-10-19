//! Approximate (variational) GP models (BoTorch's
//! `botorch.models.approximate_gp`): [`SingleTaskVariationalGP`], the
//! stochastic variational GP (SVGP) of Hensman et al. 2013 with GPyTorch's
//! whitened `VariationalStrategy` and `CholeskyVariationalDistribution`, its
//! variational ELBO (`gpytorch.mlls.VariationalELBO`) with analytic
//! gradients, posteriors with input derivatives (the [`Model`] trait, so
//! every acquisition function works on it) and fitting
//! ([`crate::fit::fit_gpytorch_mll`], `fit_gpytorch_mll_torch`).
//!
//! # The model
//!
//! For each output `t` (BoTorch models `m` outputs as a batch of
//! independent SVGPs, GPyTorch's `IndependentMultitaskVariationalStrategy`)
//! with kernel `k_t`, constant mean `c_t`, inducing points `Z` (`M × d`,
//! shared by the outputs) and variational parameters `m_t` (`M`) and
//! `L_t` (the lower triangle of `chol_variational_covar`, `S_t = L_t L_tᵀ`):
//!
//! ```text
//! L    = chol(K(Z, Z) + j I)           (psd_safe_cholesky, max_tries = 4)
//! A    = L⁻¹ K(Z, X)
//! μ(X) = c_t + Aᵀ m_t
//! Σ(X) = K(X, X) + j I + Aᵀ (S_t - I) A
//! ```
//!
//! with GPyTorch's variational jitter `j = 1e-6` (double precision). The
//! variational ELBO of targets `y` at `n` points is
//!
//! ```text
//! ELBO = Σ_t Σ_i E_q[log N(y_ti | f, σ²_t)] / n - β Σ_t KL(q(u_t) ‖ N(0, I)) / N + log p(θ) / N
//! E_q[...] = -½ [((y - μ)² + v) / σ² + log σ² + log 2π]
//! KL       = ½ [-Σ_k log L_kk² + ‖L‖²_F + ‖m‖² - M]
//! ```
//!
//! (`v` the marginal variance clamped at `1e-10` like GPyTorch's
//! `MultivariateNormal.variance`, `N = num_data`, `σ²_t` the Gaussian noise or,
//! with a `MultitaskGaussianLikelihood` of rank 0, the task noise plus the
//! global noise).
//!
//! # Parameters
//!
//! The flat raw parameters follow `VariationalELBO(likelihood,
//! model.model, ...).named_parameters()`: the likelihood's raw noise
//! (`raw_task_noises` and `raw_noise` for a multitask likelihood), the
//! inducing points (if learned), the variational means (`m × M`) and
//! Cholesky factors (`m × M × M`, upper triangles unused), the raw
//! constant of the mean (one per output or shared), and the kernel's raw
//! parameters (BoTorch's batched layout: every parameter for all outputs,
//! then the next parameter; or one shared set).

use std::ops::Range;
use std::sync::OnceLock;

use rand::rngs::StdRng;

use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::kernels::eval::TreePrep;
use crate::kernels::tree::{sample_tree_priors, KernelNode};
use crate::kernels::TreeKernel;
use crate::linalg::{cholesky_backward, psd_safe_cholesky_with, CholeskyFactor, CHOLESKY_JITTER};
use crate::models::gp::{ConstantMeanSpec, GaussianLikelihoodSpec};
use crate::models::model::check_vjp_shapes_expanded;
use crate::models::Model;
use crate::optim::fit::MllModel;
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior, MIN_VARIANCE};
use crate::transforms::input::{ChainedInputTransform, TransformMode};
use crate::transforms::Standardize;
use crate::{Matrix, Vector};

/// GPyTorch's `variational_cholesky_jitter` for double precision: added to
/// `K(Z, Z)` and to the prior covariance of the data.
pub const VARIATIONAL_JITTER: f64 = 1e-6;

/// The number of jitter attempts of the variational strategy's Cholesky
/// factorization (`psd_safe_cholesky(..., max_tries=4)` in GPyTorch's
/// `VariationalStrategy._cholesky_factor`).
pub const VARIATIONAL_CHOLESKY_MAX_TRIES: usize = 4;

/// `log(2π)`.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// The likelihood of a [`SingleTaskVariationalGP`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VariationalLikelihood {
    /// GPyTorch's `GaussianLikelihood` (one output): `raw_noise`.
    Gaussian(GaussianLikelihoodSpec),
    /// GPyTorch's `MultitaskGaussianLikelihood` with `rank=0` (several
    /// outputs): `raw_task_noises` (one per output) and the global
    /// `raw_noise`, both with this constraint and prior; the noise of output
    /// `t` is `task_noises[t] + noise`.
    Multitask(GaussianLikelihoodSpec),
}

impl VariationalLikelihood {
    fn spec(&self) -> &GaussianLikelihoodSpec {
        match self {
            Self::Gaussian(s) | Self::Multitask(s) => s,
        }
    }
}

/// The mean module of a [`SingleTaskVariationalGP`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VariationalMean {
    /// GPyTorch's `ConstantMean` (with an optional prior and a non-enforced
    /// constraint).
    Constant(ConstantMeanSpec),
    /// GPyTorch's `ZeroMean` (no parameters).
    Zero,
}

/// Options of [`SingleTaskVariationalGP::new`].
#[derive(Clone, Debug)]
pub struct SingleTaskVariationalGPOptions {
    /// The kernel tree (in the transformed input space).
    pub covar_module: KernelNode,
    /// Whether every output has its own kernel parameters (a kernel with
    /// `batch_shape = [m]`, BoTorch's default for `m > 1`); otherwise the
    /// outputs share them.
    pub covar_batched: bool,
    /// The mean module.
    pub mean_module: VariationalMean,
    /// Whether every output has its own constant (`batch_shape = [m]`).
    pub mean_batched: bool,
    /// The likelihood.
    pub likelihood: VariationalLikelihood,
    /// Whether the inducing points are parameters (BoTorch's
    /// `learn_inducing_points`, default `true`).
    pub learn_inducing_points: bool,
    /// The variational jitter (default [`VARIATIONAL_JITTER`]).
    pub jitter: f64,
    /// The input transform applied to the points of posteriors (fitted on
    /// the training inputs, used in eval mode). The training and inducing
    /// inputs given to the model are already transformed.
    pub input_transform: Option<ChainedInputTransform>,
    /// A fitted `Standardize` outcome transform (the targets given to the
    /// model are already standardized).
    pub outcome_transform: Option<Standardize>,
}

/// Offsets of the parameter groups in the flat raw vector.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    likelihood: Range<usize>,
    inducing: Option<Range<usize>>,
    var_mean: Range<usize>,
    var_chol: Range<usize>,
    constant: Range<usize>,
    kernel: Range<usize>,
}

/// The posterior quantities of one output at the current parameters.
#[derive(Clone, Debug)]
struct OutputCache {
    kernel: TreeKernel,
    z: TreePrep,
    chol: CholeskyFactor,
    /// The variational mean `m_t`.
    mean: Vector,
    /// The lower-triangular `L_t`.
    lv: Matrix,
    /// `L⁻ᵀ m_t`.
    alpha: Vector,
    /// `L⁻ᵀ (S_t - I) L⁻¹`.
    w: Matrix,
    constant: f64,
    /// The noise variance (transformed outcome space).
    noise: f64,
}

/// BoTorch's `SingleTaskVariationalGP` (see the [module
/// documentation](self)).
#[derive(Clone, Debug)]
pub struct SingleTaskVariationalGP {
    /// Training inputs in the transformed space (`n × d'`).
    train_x: Matrix,
    /// Standardized training targets (`n × m`), if given.
    train_y: Option<Matrix>,
    /// Raw input dimension `d`.
    dim: usize,
    num_outputs: usize,
    num_inducing: usize,
    /// The inducing points when they are not learned (a buffer).
    fixed_inducing: Matrix,
    options: SingleTaskVariationalGPOptions,
    layout: Layout,
    raw: Vec<f64>,
    cache: OnceLock<std::result::Result<Vec<OutputCache>, BotorchError>>,
}

/// The terms of the ELBO (`combine_terms=False` in GPyTorch): the expected
/// log likelihood divided by the number of points, the KL divergence times
/// `β / N`, and the log prior divided by `N`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElboTerms {
    /// `Σ E_q[log p(y | f)] / n`.
    pub log_likelihood: f64,
    /// `β KL(q(u) ‖ p(u)) / N`.
    pub kl_divergence: f64,
    /// `log p(θ) / N`.
    pub log_prior: f64,
}

impl ElboTerms {
    /// The ELBO, `log_likelihood - kl_divergence + log_prior`.
    pub fn value(&self) -> f64 {
        self.log_likelihood - self.kl_divergence + self.log_prior
    }
}

fn kernel_param_numels(node: &KernelNode) -> Vec<usize> {
    node.all_params().iter().map(|p| p.numel()).collect()
}

impl SingleTaskVariationalGP {
    /// Builds a model from the transformed training inputs `train_x` (`n ×
    /// d'`), the standardized training targets `train_y` (`n × m`, or
    /// `None` for a model without targets), the inducing points `inducing`
    /// (`M × d'`, in the transformed space) and the flat raw parameters
    /// `raw` (in the order of the [module documentation](self); `None` for
    /// the defaults: the constraints' initial values, a zero variational
    /// mean and identity Cholesky factors). `dim` is the raw input
    /// dimension (that of `train_x` without an input transform).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
    /// [`BotorchError::InputData`] for NaN inputs, and
    /// [`BotorchError::Unsupported`] for a multitask likelihood with one
    /// output, a Gaussian likelihood with several outputs, or a one-to-many
    /// input transform.
    pub fn new(
        train_x: Matrix,
        train_y: Option<Matrix>,
        inducing: Matrix,
        num_outputs: usize,
        dim: usize,
        raw: Option<Vec<f64>>,
        options: SingleTaskVariationalGPOptions,
    ) -> Result<Self> {
        let dt = train_x.ncols();
        if num_outputs == 0 {
            return Err(BotorchError::InvalidArgument(
                "num_outputs must be at least 1".into(),
            ));
        }
        if inducing.ncols() != dt {
            return Err(BotorchError::DimensionMismatch(format!(
                "the inducing points have {} columns, the (transformed) training inputs {dt}",
                inducing.ncols()
            )));
        }
        if let Some(y) = &train_y {
            if y.nrows() != train_x.nrows() || y.ncols() != num_outputs {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected {} x {num_outputs} training targets, got {} x {}",
                    train_x.nrows(),
                    y.nrows(),
                    y.ncols()
                )));
            }
        }
        if train_x.iter().any(|v| v.is_nan()) || inducing.iter().any(|v| v.is_nan()) {
            return Err(BotorchError::InputData(
                "the training inputs or inducing points contain NaN values.".into(),
            ));
        }
        match (&options.likelihood, num_outputs) {
            (VariationalLikelihood::Gaussian(_), m) if m > 1 => {
                return Err(BotorchError::Unsupported(
                    "a GaussianLikelihood with several outputs (GPyTorch's variational ELBO \
                     fails on it); use a MultitaskGaussianLikelihood"
                        .into(),
                ))
            }
            (VariationalLikelihood::Multitask(_), 1) => {
                return Err(BotorchError::Unsupported(
                    "a MultitaskGaussianLikelihood with one output".into(),
                ))
            }
            _ => {}
        }
        if let Some(chain) = &options.input_transform {
            if chain.points_per_input() != 1 {
                return Err(BotorchError::Unsupported(
                    "one-to-many input transforms (e.g. InputPerturbation) in a \
                     SingleTaskVariationalGP"
                        .into(),
                ));
            }
            chain.check_mode_dim(dim, dt, TransformMode::Eval)?;
        } else if dim != dt {
            return Err(BotorchError::DimensionMismatch(format!(
                "input dimension {dim} differs from the training inputs' {dt}"
            )));
        }
        if let Some(st) = &options.outcome_transform {
            if !st.is_trained() || st.means().len() != num_outputs {
                return Err(BotorchError::InvalidArgument(format!(
                    "the outcome transform must be a fitted Standardize over {num_outputs} \
                     outputs"
                )));
            }
        }
        options.covar_module.validate(dt)?;
        let m_ind = inducing.nrows();
        let m = num_outputs;
        let nl = match options.likelihood {
            VariationalLikelihood::Gaussian(_) => 1,
            VariationalLikelihood::Multitask(_) => m + 1,
        };
        let mut off = 0;
        let mut take = |len: usize| {
            let r = off..off + len;
            off += len;
            r
        };
        let likelihood = take(nl);
        let inducing_range = options.learn_inducing_points.then(|| take(m_ind * dt));
        let var_mean = take(m * m_ind);
        let var_chol = take(m * m_ind * m_ind);
        let n_const = match options.mean_module {
            VariationalMean::Zero => 0,
            VariationalMean::Constant(_) => {
                if options.mean_batched {
                    m
                } else {
                    1
                }
            }
        };
        let constant = take(n_const);
        let nk = options.covar_module.num_raw() * if options.covar_batched { m } else { 1 };
        let kernel = take(nk);
        let total = off;
        let layout = Layout {
            likelihood,
            inducing: inducing_range,
            var_mean,
            var_chol,
            constant,
            kernel,
        };
        let raw = match raw {
            Some(r) => {
                if r.len() != total {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "expected {total} raw parameters, got {}",
                        r.len()
                    )));
                }
                r
            }
            None => {
                let mut r = vec![0.0; total];
                let noise0 = options.likelihood.spec().initial_raw_noise();
                for v in &mut r[layout.likelihood.clone()] {
                    *v = noise0;
                }
                if let Some(ir) = &layout.inducing {
                    for (k, v) in r[ir.clone()].iter_mut().enumerate() {
                        *v = inducing[(k / dt, k % dt)];
                    }
                }
                for t in 0..m {
                    for k in 0..m_ind {
                        r[layout.var_chol.start + t * m_ind * m_ind + k * m_ind + k] = 1.0;
                    }
                }
                if let VariationalMean::Constant(spec) = options.mean_module {
                    for v in &mut r[layout.constant.clone()] {
                        *v = spec.initial_raw();
                    }
                }
                let init = options.covar_module.initial_raw();
                let reps = if options.covar_batched { m } else { 1 };
                let numels = kernel_param_numels(&options.covar_module);
                let mut pos = layout.kernel.start;
                let mut src = 0;
                for ne in numels {
                    for _ in 0..reps {
                        r[pos..pos + ne].copy_from_slice(&init[src..src + ne]);
                        pos += ne;
                    }
                    src += ne;
                }
                r
            }
        };
        if raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "the raw parameters must be finite".into(),
            ));
        }
        Ok(Self {
            train_x,
            train_y,
            dim,
            num_outputs,
            num_inducing: m_ind,
            fixed_inducing: inducing,
            options,
            layout,
            raw,
            cache: OnceLock::new(),
        })
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// The transformed training inputs (`n × d'`).
    pub fn train_inputs(&self) -> &Matrix {
        &self.train_x
    }

    /// The standardized training targets (`n × m`), if any.
    pub fn train_targets(&self) -> Option<&Matrix> {
        self.train_y.as_ref()
    }

    /// The number of inducing points `M`.
    pub fn num_inducing(&self) -> usize {
        self.num_inducing
    }

    /// The transformed input dimension `d'`.
    pub fn transformed_dim(&self) -> usize {
        self.train_x.ncols()
    }

    /// The options the model was built with.
    pub fn options(&self) -> &SingleTaskVariationalGPOptions {
        &self.options
    }

    /// The flat raw parameters.
    pub fn raw_parameters(&self) -> &[f64] {
        &self.raw
    }

    /// Sets the flat raw parameters.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong length and
    /// [`BotorchError::InvalidArgument`] for non-finite values.
    pub fn set_raw_parameters(&mut self, raw: &[f64]) -> Result<()> {
        if raw.len() != self.raw.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                self.raw.len(),
                raw.len()
            )));
        }
        if raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "the raw parameters must be finite".into(),
            ));
        }
        self.raw = raw.to_vec();
        self.cache = OnceLock::new();
        Ok(())
    }

    /// Sets the inducing points (`M × d'`; the number of points must stay
    /// the same), as `init_inducing_points` does.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong shape.
    pub fn set_inducing_points(&mut self, z: &Matrix) -> Result<()> {
        let dt = self.transformed_dim();
        if z.shape() != (self.num_inducing, dt) {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} x {dt} inducing points, got {} x {}",
                self.num_inducing,
                z.nrows(),
                z.ncols()
            )));
        }
        match self.layout.inducing.clone() {
            Some(r) => {
                for (k, v) in self.raw[r].iter_mut().enumerate() {
                    *v = z[(k / dt, k % dt)];
                }
            }
            None => self.fixed_inducing = z.clone(),
        }
        self.cache = OnceLock::new();
        Ok(())
    }

    /// The names and sizes of the raw parameters (BoTorch's names in
    /// `VariationalELBO(likelihood, model.model, ...).named_parameters()`).
    pub fn parameter_names(&self) -> Vec<(String, usize)> {
        let m = self.num_outputs;
        let multi = m > 1;
        let vs = if multi {
            "model.variational_strategy.base_variational_strategy."
        } else {
            "model.variational_strategy."
        };
        let mut out = Vec::new();
        match self.options.likelihood {
            VariationalLikelihood::Gaussian(_) => {
                out.push(("likelihood.noise_covar.raw_noise".to_string(), 1));
            }
            VariationalLikelihood::Multitask(_) => {
                out.push(("likelihood.raw_task_noises".to_string(), m));
                out.push(("likelihood.raw_noise".to_string(), 1));
            }
        }
        if let Some(r) = &self.layout.inducing {
            out.push((format!("{vs}inducing_points"), r.len()));
        }
        out.push((
            format!("{vs}_variational_distribution.variational_mean"),
            self.layout.var_mean.len(),
        ));
        out.push((
            format!("{vs}_variational_distribution.chol_variational_covar"),
            self.layout.var_chol.len(),
        ));
        if !self.layout.constant.is_empty() {
            out.push((
                "model.mean_module.raw_constant".to_string(),
                self.layout.constant.len(),
            ));
        }
        let reps = if self.options.covar_batched { m } else { 1 };
        let mut names = Vec::new();
        self.options
            .covar_module
            .visit_params("model.covar_module.", &mut |name, p| {
                names.push((name, p.numel() * reps));
            });
        out.extend(names);
        out
    }

    /// The optimizer bounds of the raw parameters (from the non-enforced
    /// constraints; unbounded otherwise).
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let mut b = vec![free; self.raw.len()];
        let lb = self.options.likelihood.spec().noise_constraint.raw_bounds();
        for v in &mut b[self.layout.likelihood.clone()] {
            *v = lb;
        }
        if let VariationalMean::Constant(spec) = self.options.mean_module {
            for v in &mut b[self.layout.constant.clone()] {
                *v = spec.raw_bounds();
            }
        }
        let kb = self.options.covar_module.raw_bounds();
        let reps = if self.options.covar_batched {
            self.num_outputs
        } else {
            1
        };
        let mut pos = self.layout.kernel.start;
        let mut src = 0;
        for ne in kernel_param_numels(&self.options.covar_module) {
            for _ in 0..reps {
                b[pos..pos + ne].copy_from_slice(&kb[src..src + ne]);
                pos += ne;
            }
            src += ne;
        }
        b
    }

    /// The inducing points at the raw parameters `raw`.
    fn inducing_at(&self, raw: &[f64]) -> Matrix {
        let dt = self.transformed_dim();
        match &self.layout.inducing {
            Some(r) => Matrix::from_row_slice(self.num_inducing, dt, &raw[r.clone()]),
            None => self.fixed_inducing.clone(),
        }
    }

    /// The current inducing points (`M × d'`).
    pub fn inducing_points(&self) -> Matrix {
        self.inducing_at(&self.raw)
    }

    /// Indices into the flat raw vector of output `t`'s kernel parameters
    /// (in the order of the kernel tree's raw parameters).
    fn kernel_indices(&self, t: usize) -> Vec<usize> {
        let batched = self.options.covar_batched;
        let reps = if batched { self.num_outputs } else { 1 };
        let mut out = Vec::with_capacity(self.options.covar_module.num_raw());
        let mut pos = self.layout.kernel.start;
        for ne in kernel_param_numels(&self.options.covar_module) {
            let start = pos + if batched { t * ne } else { 0 };
            out.extend(start..start + ne);
            pos += ne * reps;
        }
        out
    }

    fn constant_index(&self, t: usize) -> Option<usize> {
        match self.options.mean_module {
            VariationalMean::Zero => None,
            VariationalMean::Constant(_) => {
                Some(self.layout.constant.start + if self.options.mean_batched { t } else { 0 })
            }
        }
    }

    /// The noise variance of every output (transformed outcome space) at
    /// `raw`.
    fn noises_at(&self, raw: &[f64]) -> Vec<f64> {
        let spec = self.options.likelihood.spec();
        let c = spec.noise_constraint;
        let r = &raw[self.layout.likelihood.clone()];
        match self.options.likelihood {
            VariationalLikelihood::Gaussian(_) => vec![c.transform(r[0])],
            VariationalLikelihood::Multitask(_) => {
                let m = self.num_outputs;
                let global = c.transform(r[m]);
                (0..m).map(|t| c.transform(r[t]) + global).collect()
            }
        }
    }

    /// The likelihood's noise values (transformed space): `[noise]` for a
    /// Gaussian likelihood, `[task_noises..., noise]` for a multitask one.
    pub fn likelihood_values(&self) -> Vec<f64> {
        let c = self.options.likelihood.spec().noise_constraint;
        self.raw[self.layout.likelihood.clone()]
            .iter()
            .map(|&r| c.transform(r))
            .collect()
    }

    /// The kernel of output `t` at `raw`.
    fn kernel_at(&self, raw: &[f64], t: usize) -> Result<TreeKernel> {
        let kraw: Vec<f64> = self.kernel_indices(t).iter().map(|&i| raw[i]).collect();
        TreeKernel::build(&self.options.covar_module, &kraw, self.transformed_dim())
    }

    /// The kernel of output `t` at the current parameters (for evaluating
    /// it, e.g. in the inducing point allocators).
    ///
    /// # Errors
    /// See [`TreeKernel::build`].
    pub fn kernel(&self, t: usize) -> Result<TreeKernel> {
        self.kernel_at(&self.raw, t)
    }

    fn variational_at(&self, raw: &[f64], t: usize) -> (Vector, Matrix) {
        let mi = self.num_inducing;
        let ms = self.layout.var_mean.start + t * mi;
        let mean = Vector::from_column_slice(&raw[ms..ms + mi]);
        let cs = self.layout.var_chol.start + t * mi * mi;
        let lv = Matrix::from_fn(
            mi,
            mi,
            |i, j| {
                if j <= i {
                    raw[cs + i * mi + j]
                } else {
                    0.0
                }
            },
        );
        (mean, lv)
    }

    fn constant_at(&self, raw: &[f64], t: usize) -> f64 {
        self.constant_index(t).map_or(0.0, |i| raw[i])
    }

    /// `psd_safe_cholesky(K(Z, Z) + j I, max_tries=4)`.
    fn inducing_cholesky(
        &self,
        kernel: &TreeKernel,
        z: &TreePrep,
    ) -> Result<(Matrix, CholeskyFactor)> {
        let mut kzz = kernel.symmetric(z);
        for i in 0..self.num_inducing {
            kzz[(i, i)] += self.options.jitter;
        }
        let chol = psd_safe_cholesky_with(&kzz, CHOLESKY_JITTER, VARIATIONAL_CHOLESKY_MAX_TRIES)?;
        Ok((kzz, chol))
    }

    fn compute_cache(&self) -> Result<Vec<OutputCache>> {
        let z = self.inducing_points();
        let noises = self.noises_at(&self.raw);
        let mi = self.num_inducing;
        let mut out = Vec::with_capacity(self.num_outputs);
        for t in 0..self.num_outputs {
            let kernel = self.kernel_at(&self.raw, t)?;
            let zp = kernel.prepare(&z)?;
            let (_, chol) = self.inducing_cholesky(&kernel, &zp)?;
            let (mean, lv) = self.variational_at(&self.raw, t);
            let alpha = chol.l().tr_solve_lower_triangular(&mean).ok_or_else(|| {
                BotorchError::Numerical("singular inducing-point Cholesky factor".into())
            })?;
            let mut s_minus_i = &lv * lv.transpose();
            for k in 0..mi {
                s_minus_i[(k, k)] -= 1.0;
            }
            // W = L⁻ᵀ (S - I) L⁻¹.
            let linv = chol.inverse_factor();
            let w = linv.transpose() * &s_minus_i * &linv;
            out.push(OutputCache {
                kernel,
                z: zp,
                chol,
                mean,
                lv,
                alpha,
                w,
                constant: self.constant_at(&self.raw, t),
                noise: noises[if noises.len() == 1 { 0 } else { t }],
            });
        }
        Ok(out)
    }

    fn cache(&self) -> Result<&Vec<OutputCache>> {
        self.cache
            .get_or_init(|| self.compute_cache())
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The outcome transform's mean and standard deviation of output `t`.
    fn affine(&self, t: usize) -> (f64, f64) {
        match &self.options.outcome_transform {
            Some(st) => (st.means()[t], st.stdvs()[t]),
            None => (0.0, 1.0),
        }
    }

    // ------------------------------------------------------------------
    // Input transform
    // ------------------------------------------------------------------

    fn check_columns(&self, x: &Matrix) -> Result<()> {
        if x.ncols() != self.dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected inputs with {} columns, got {}",
                self.dim,
                x.ncols()
            )));
        }
        Ok(())
    }

    /// The raw-space points `x` transformed by the input transform (eval
    /// mode).
    ///
    /// # Errors
    /// For a column mismatch.
    pub fn transform_inputs(&self, x: &Matrix) -> Result<Matrix> {
        self.check_columns(x)?;
        match &self.options.input_transform {
            Some(chain) => chain.transform_mode(x, TransformMode::Eval),
            None => Ok(x.clone()),
        }
    }

    // ------------------------------------------------------------------
    // Posterior
    // ------------------------------------------------------------------

    /// The latent posterior `q(f)` of output `t` at transformed points `xt`
    /// in the transformed outcome space (GPyTorch's `VariationalStrategy`
    /// in eval mode): mean and covariance (with the variational jitter).
    fn latent(&self, c: &OutputCache, xt: &Matrix) -> Result<(Vector, Matrix)> {
        let xp = c.kernel.prepare(xt)?;
        let mut a = c.kernel.cross(&c.z, &xp);
        c.chol.solve_lower_mut(&mut a);
        let mean = (a.tr_mul(&c.mean)).add_scalar(c.constant);
        // (S - I) A = L_v (L_vᵀ A) - A, then Aᵀ of that.
        let sa = &c.lv * c.lv.tr_mul(&a) - &a;
        let mut cov = c.kernel.symmetric(&xp);
        cov += a.tr_mul(&sa);
        for i in 0..xt.nrows() {
            cov[(i, i)] += self.options.jitter;
        }
        Ok((mean, cov))
    }

    /// The marginal variances of `q(f)` at transformed points `xt` as
    /// GPyTorch's `variance` computes them (the lazy `diag=True` kernel
    /// values), in the transformed outcome space, with their derivatives
    /// with respect to the points if `with_grad`.
    fn latent_marginals(
        &self,
        c: &OutputCache,
        xt: &Matrix,
        with_grad: bool,
    ) -> Result<(Vector, Vector, Vec<Matrix>)> {
        let q = xt.nrows();
        let dt = xt.ncols();
        let xp = c.kernel.prepare(xt)?;
        let mut a = c.kernel.cross(&c.z, &xp);
        c.chol.solve_lower_mut(&mut a);
        let mean = (a.tr_mul(&c.mean)).add_scalar(c.constant);
        let sa = &c.lv * c.lv.tr_mul(&a) - &a;
        let mut var = Vector::zeros(q);
        let mut grads = Vec::new();
        let kzx = if with_grad {
            c.kernel.cross(&c.z, &xp)
        } else {
            Matrix::zeros(0, 0)
        };
        for i in 0..q {
            let update: f64 = a.column(i).dot(&sa.column(i));
            if with_grad {
                let (kd, dkd) = c.kernel.diag_mode_with_grad(&xp, i);
                var[i] = kd + self.options.jitter + update;
                let (_, dc) = c.kernel.cross_with_grad(&xp, i, &c.z); // M x dt
                let wc = &c.w * kzx.column(i);
                let dmean = dc.tr_mul(&c.alpha);
                let dvar = dkd + dc.tr_mul(&wc) * 2.0;
                let mut g = Matrix::zeros(2, dt);
                g.row_mut(0).copy_from(&dmean.transpose());
                g.row_mut(1).copy_from(&dvar.transpose());
                grads.push(g);
            } else {
                var[i] = c.kernel.diag_mode(&xp, i) + self.options.jitter + update;
            }
        }
        Ok((mean, var, grads))
    }

    /// The marginal variances of `q(f)` (GPyTorch's `variance`) of every
    /// output at transformed points `xt` (`m` vectors, transformed outcome
    /// space).
    ///
    /// # Errors
    /// See [`SingleTaskVariationalGP::latent_posterior`].
    pub fn latent_variances(&self, xt: &Matrix) -> Result<Vec<Vector>> {
        self.cache()?
            .iter()
            .map(|c| self.latent_marginals(c, xt, false).map(|r| r.1))
            .collect()
    }

    /// The latent variational posterior `q(f)` at transformed points `xt`
    /// (`q × d'`) in the transformed outcome space (GPyTorch's output of
    /// `model.model(X)`): the mean and covariance of every output.
    ///
    /// # Errors
    /// For a column mismatch or a failing Cholesky factorization of
    /// `K(Z, Z)`.
    pub fn latent_posterior(&self, xt: &Matrix) -> Result<Vec<(Vector, Matrix)>> {
        if xt.ncols() != self.transformed_dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected inputs with {} columns, got {}",
                self.transformed_dim(),
                xt.ncols()
            )));
        }
        self.cache()?.iter().map(|c| self.latent(c, xt)).collect()
    }

    /// The posterior at transformed points `xt` over all outputs
    /// (output-major), in the original outcome space.
    fn posterior_transformed(&self, xt: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let caches = self.cache()?;
        let q = xt.nrows();
        let m = self.num_outputs;
        let mut mean = Vector::zeros(q * m);
        let mut cov = Matrix::zeros(q * m, q * m);
        for (t, c) in caches.iter().enumerate() {
            let (mu, mut sigma) = self.latent(c, xt)?;
            if observation_noise {
                for i in 0..q {
                    sigma[(i, i)] += c.noise;
                }
            }
            let (om, os) = self.affine(t);
            for i in 0..q {
                mean[t * q + i] = om + os * mu[i];
                for j in 0..q {
                    cov[(t * q + i, t * q + j)] = os * sigma[(i, j)] * os;
                }
            }
        }
        Ok(Posterior::multi_output(mean, cov, m))
    }

    /// The joint posterior at raw-space points `x` (BoTorch's
    /// `model.posterior(X, observation_noise=...)`).
    ///
    /// # Errors
    /// For a column mismatch or a failing Cholesky factorization of
    /// `K(Z, Z)`.
    pub fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let xt = self.transform_inputs(x)?;
        self.posterior_transformed(&xt, observation_noise)
    }

    /// The joint posterior and its derivatives with respect to the
    /// transformed points (forward mode).
    fn with_grad_transformed(&self, xt: &Matrix) -> Result<JointPosteriorGrad> {
        let caches = self.cache()?;
        let (q, dt) = xt.shape();
        let m = self.num_outputs;
        let qm = q * m;
        let mut mean = Vector::zeros(qm);
        let mut cov = Matrix::zeros(qm, qm);
        let mut dmean = vec![Vector::zeros(qm); q * dt];
        let mut dcov = vec![Matrix::zeros(qm, qm); q * dt];
        for (t, c) in caches.iter().enumerate() {
            let (mu, sigma) = self.latent(c, xt)?;
            let (om, os) = self.affine(t);
            let s2 = os * os;
            for i in 0..q {
                mean[t * q + i] = om + os * mu[i];
                for j in 0..q {
                    cov[(t * q + i, t * q + j)] = s2 * sigma[(i, j)];
                }
            }
            let xp = c.kernel.prepare(xt)?;
            let kzx = c.kernel.cross(&c.z, &xp);
            let wc = &c.w * &kzx; // M x q
            for p in 0..q {
                let (_, dc) = c.kernel.cross_with_grad(&xp, p, &c.z); // M x dt
                let (_, dk) = c.kernel.cross_with_grad(&xp, p, &xp); // q x dt
                let (_, dkpp) = c.kernel.diag_with_grad(&xp, p);
                for l in 0..dt {
                    let j = p * dt + l;
                    let dcl = dc.column(l);
                    dmean[j][t * q + p] = os * dcl.dot(&c.alpha);
                    let row = wc.tr_mul(&dcl); // q: (W C)ᵀ dC
                    let block = &mut dcov[j];
                    for jj in 0..q {
                        if jj == p {
                            continue;
                        }
                        let v = s2 * (dk[(jj, l)] + row[jj]);
                        block[(t * q + p, t * q + jj)] = v;
                        block[(t * q + jj, t * q + p)] = v;
                    }
                    block[(t * q + p, t * q + p)] = s2 * (dkpp[l] + 2.0 * row[p]);
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean,
            covariance: cov,
            dmean,
            dcov,
            num_outputs: m,
        })
    }

    /// The joint posterior (without observation noise) at raw-space points
    /// `x` with the derivatives of its mean and covariance with respect to
    /// every input coordinate (see [`JointPosteriorGrad`]).
    ///
    /// # Errors
    /// See [`SingleTaskVariationalGP::posterior`].
    pub fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.check_columns(x)?;
        let Some(chain) = &self.options.input_transform else {
            return self.with_grad_transformed(x);
        };
        let (xt, tape) = chain.transform_with_tape(x)?;
        let g = self.with_grad_transformed(&xt)?;
        let jacs = tape.jacobians_through(chain)?;
        let (q, d) = x.shape();
        let dt = xt.ncols();
        let len = g.mean.len();
        let mut dmean = vec![Vector::zeros(len); q * d];
        let mut dcov = vec![Matrix::zeros(len, len); q * d];
        for i in 0..q {
            for l in 0..dt {
                for j in 0..d {
                    let f = jacs[i][(l, j)];
                    if f != 0.0 {
                        dmean[i * d + j].axpy(f, &g.dmean[i * dt + l], 1.0);
                        dcov[i * d + j] += &g.dcov[i * dt + l] * f;
                    }
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean: g.mean,
            covariance: g.covariance,
            dmean,
            dcov,
            num_outputs: g.num_outputs,
        })
    }

    /// Reverse mode at transformed points: the adjoint of `xt[rows]`.
    fn vjp_transformed(
        &self,
        xt: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        let caches = self.cache()?;
        let (q, dt) = xt.shape();
        let mut out = Matrix::zeros(rows.len(), dt);
        for (t, c) in caches.iter().enumerate() {
            let (_, os) = self.affine(t);
            let s2 = os * os;
            let abar = Vector::from_fn(q, |i, _| os * mean_adjoint[t * q + i]);
            let sbar = Matrix::from_fn(q, q, |i, j| {
                0.5 * s2
                    * (covariance_adjoint[(t * q + i, t * q + j)]
                        + covariance_adjoint[(t * q + j, t * q + i)])
            });
            if abar.iter().all(|v| *v == 0.0) && sbar.iter().all(|v| *v == 0.0) {
                continue;
            }
            let xp = c.kernel.prepare(xt)?;
            let kzx = c.kernel.cross(&c.z, &xp);
            // W C Σ̄ (M x q).
            let wcs = &c.w * &kzx * &sbar;
            for (r, p) in rows.clone().enumerate() {
                let (_, dc) = c.kernel.cross_with_grad(&xp, p, &c.z); // M x dt
                let (_, dk) = c.kernel.cross_with_grad(&xp, p, &xp); // q x dt
                let coef = &c.alpha * abar[p] + wcs.column(p) * 2.0; // M
                let mut g = dc.tr_mul(&coef); // dt
                let srow = sbar.row(p).transpose();
                g += dk.tr_mul(&srow) * 2.0;
                for l in 0..dt {
                    out[(r, l)] += g[l];
                }
            }
        }
        Ok(out)
    }

    /// Reverse-mode derivative of the joint posterior (see
    /// [`Model::joint_posterior_vjp`]).
    ///
    /// # Errors
    /// See [`SingleTaskVariationalGP::posterior`].
    pub fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            self.num_outputs,
            1,
        )?;
        self.check_columns(x)?;
        let Some(chain) = &self.options.input_transform else {
            return self.vjp_transformed(x, rows, mean_adjoint, covariance_adjoint);
        };
        let (xt, tape) = chain.transform_with_tape(x)?;
        let g = self.vjp_transformed(&xt, rows.clone(), mean_adjoint, covariance_adjoint)?;
        tape.backward_through(chain, rows, &g)
    }

    // ------------------------------------------------------------------
    // Variational ELBO
    // ------------------------------------------------------------------

    /// The terms of the variational ELBO (GPyTorch's
    /// `VariationalELBO(likelihood, model.model, num_data, beta)` evaluated
    /// on `model.model(x)` and the targets `y`) at the raw parameters
    /// `raw` (the current ones if `None`), for transformed inputs `x` (`n ×
    /// d'`) and standardized targets `y` (`n × m`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
    /// [`BotorchError::NotPsd`] / [`BotorchError::Nan`] if `K(Z, Z)` cannot
    /// be factorized, [`BotorchError::InvalidArgument`] for parameters
    /// outside the support of their priors.
    pub fn elbo_terms(
        &self,
        raw: Option<&[f64]>,
        x: &Matrix,
        y: &Matrix,
        num_data: f64,
        beta: f64,
    ) -> Result<ElboTerms> {
        self.elbo_impl(raw.unwrap_or(&self.raw), x, y, num_data, beta, false)
            .map(|(t, _)| t)
    }

    /// The ELBO on the training data and its gradient with respect to the
    /// raw parameters `raw` (BoTorch's training closure, `mll(model.model(
    /// *train_inputs), train_targets)`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for a model without training
    /// targets, otherwise see [`SingleTaskVariationalGP::elbo_terms`].
    pub fn elbo_value_and_grad(
        &self,
        raw: &[f64],
        num_data: f64,
        beta: f64,
    ) -> Result<(ElboTerms, Vec<f64>)> {
        let y = self.train_y.as_ref().ok_or_else(|| {
            BotorchError::InvalidArgument(
                "the model has no training targets (`train_Y` was not given)".into(),
            )
        })?;
        let (terms, g) = self.elbo_impl(raw, &self.train_x, y, num_data, beta, true)?;
        Ok((terms, g.expect("gradient requested")))
    }

    #[allow(clippy::too_many_lines)]
    fn elbo_impl(
        &self,
        raw: &[f64],
        x: &Matrix,
        y: &Matrix,
        num_data: f64,
        beta: f64,
        with_grad: bool,
    ) -> Result<(ElboTerms, Option<Vec<f64>>)> {
        if raw.len() != self.raw.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                self.raw.len(),
                raw.len()
            )));
        }
        let (n, dt) = x.shape();
        let m = self.num_outputs;
        if dt != self.transformed_dim() || y.nrows() != n || y.ncols() != m {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected inputs with {} columns and {n} x {m} targets, got {} columns and \
                 {} x {} targets",
                self.transformed_dim(),
                dt,
                y.nrows(),
                y.ncols()
            )));
        }
        let mi = self.num_inducing;
        let jitter = self.options.jitter;
        let z = self.inducing_at(raw);
        let lik_spec = *self.options.likelihood.spec();
        let nc = lik_spec.noise_constraint;
        let lraw = &raw[self.layout.likelihood.clone()];
        let lvals: Vec<f64> = lraw.iter().map(|&r| nc.transform(r)).collect();
        let noises = self.noises_at(raw);
        let nf = n as f64;
        let mut grad = vec![0.0; raw.len()];
        let mut ll_total = 0.0;
        let mut kl_total = 0.0;
        // dELBO/d noise of every output.
        let mut g_noise = vec![0.0; m];
        // Per-output latent quantities, needed to decide GPyTorch's variance
        // clamp, which applies to all outputs at once.
        struct Part {
            kernel: TreeKernel,
            zx: TreePrep,
            chol: CholeskyFactor,
            a: Matrix,
            mean: Vector,
            lv: Matrix,
            mu: Vector,
            var: Vector,
            kidx: Vec<usize>,
        }
        let mut stacked = Matrix::zeros(mi + n, dt);
        for i in 0..mi {
            stacked.row_mut(i).copy_from(&z.row(i));
        }
        for i in 0..n {
            stacked.row_mut(mi + i).copy_from(&x.row(i));
        }
        let mut parts = Vec::with_capacity(m);
        for t in 0..m {
            let kidx = self.kernel_indices(t);
            let kraw: Vec<f64> = kidx.iter().map(|&i| raw[i]).collect();
            let kernel = TreeKernel::build(&self.options.covar_module, &kraw, dt)?;
            let zx = kernel.prepare(&stacked)?;
            let zp = kernel.prepare(&z)?;
            let xp = kernel.prepare(x)?;
            let (_, chol) = self.inducing_cholesky(&kernel, &zp)?;
            let mut a = kernel.cross(&zp, &xp);
            chol.solve_lower_mut(&mut a);
            let (mean, lv) = self.variational_at(raw, t);
            let c = self.constant_at(raw, t);
            let mu = a.tr_mul(&mean).add_scalar(c);
            // var_i = k(x_i, x_i) + j + Σ_k (Aᵀ L_v)_ik² - Σ_k A_ki².
            let b = a.tr_mul(&lv);
            let var = Vector::from_fn(n, |i, _| {
                let kd = kernel.diag_mode(&xp, i);
                let sb: f64 = b.row(i).iter().map(|v| v * v).sum();
                let sa: f64 = a.column(i).iter().map(|v| v * v).sum();
                kd + jitter + (sb - sa)
            });
            parts.push(Part {
                kernel,
                zx,
                chol,
                a,
                mean,
                lv,
                mu,
                var,
                kidx,
            });
        }
        let clamp = parts
            .iter()
            .any(|p| p.var.iter().any(|&v| v < MIN_VARIANCE));
        for (t, p) in parts.iter_mut().enumerate() {
            let s2 = noises[if noises.len() == 1 { 0 } else { t }];
            let ln_s2 = s2.ln();
            let mut ll = 0.0;
            let mut gmu = Vector::zeros(n);
            let mut gvar = Vector::zeros(n);
            let mut gs2 = 0.0;
            for i in 0..n {
                let mut v = p.var[i];
                let mut dv = 1.0;
                if clamp && v <= MIN_VARIANCE {
                    // `clamp_min`'s gradient at and below the bound is zero
                    // (torch >= 2.14 at the bound).
                    v = MIN_VARIANCE;
                    dv = 0.0;
                }
                let r = y[(i, t)] - p.mu[i];
                let q = r * r + v;
                ll += -0.5 * (q / s2 + ln_s2 + LN_2PI);
                gmu[i] = r / s2 / nf;
                gvar[i] = -0.5 / s2 / nf * dv;
                gs2 += 0.5 * (q / (s2 * s2) - 1.0 / s2) / nf;
            }
            ll_total += ll;
            g_noise[t] = gs2;
            // KL(q(u) ‖ N(0, I)).
            let mut kl = -(mi as f64);
            for k in 0..mi {
                let lkk = p.lv[(k, k)];
                kl -= (lkk * lkk).ln();
            }
            kl += p.lv.iter().map(|v| v * v).sum::<f64>();
            kl += p.mean.dot(&p.mean);
            kl *= 0.5;
            kl_total += kl;
            if !with_grad {
                continue;
            }
            let kl_scale = beta / num_data;
            // Variational mean and Cholesky factor.
            let gm = &p.a * &gmu - &p.mean * kl_scale;
            let ms = self.layout.var_mean.start + t * mi;
            for k in 0..mi {
                grad[ms + k] += gm[k];
            }
            // d/dL_v: 2 A D Aᵀ L_v - β/N (L_v - diag(1/L_kk)).
            let ad = Matrix::from_fn(mi, n, |k, i| p.a[(k, i)] * gvar[i]);
            let glv = (&ad * p.a.transpose() * &p.lv) * 2.0;
            let cs = self.layout.var_chol.start + t * mi * mi;
            for i in 0..mi {
                for j in 0..=i {
                    let mut g = glv[(i, j)] - kl_scale * p.lv[(i, j)];
                    if i == j {
                        g += kl_scale / p.lv[(i, i)];
                    }
                    grad[cs + i * mi + j] += g;
                }
            }
            // Constant mean.
            if let Some(ci) = self.constant_index(t) {
                grad[ci] += gmu.sum();
            }
            // G_A = m gμᵀ + 2 (S - I) A D.
            let mut s_minus_i = &p.lv * p.lv.transpose();
            for k in 0..mi {
                s_minus_i[(k, k)] -= 1.0;
            }
            let ga = &p.mean * gmu.transpose() + (&s_minus_i * &ad) * 2.0;
            // A = L⁻¹ C: Ḡ_C = L⁻ᵀ G_A, Ḡ_L = -Ḡ_C Aᵀ (lower).
            let mut gc = ga;
            p.chol.solve_lower_transpose_mut(&mut gc);
            let gl = -(&gc * p.a.transpose());
            let gkzz = cholesky_backward(p.chol.l(), &gl);
            // Kernel hyperparameters: contraction on the stacked [Z; X].
            let mut wbig = Matrix::zeros(mi + n, mi + n);
            for i in 0..mi {
                for j in 0..mi {
                    wbig[(i, j)] = gkzz[(i, j)];
                }
                for j in 0..n {
                    wbig[(i, mi + j)] = 0.5 * gc[(i, j)];
                    wbig[(mi + j, i)] = 0.5 * gc[(i, j)];
                }
            }
            // The marginal variances use GPyTorch's lazy `diag=True` values.
            let mut gvals = p.kernel.contract(&p.zx, &wbig);
            let xp = p.kernel.prepare(x)?;
            let gdiag = p.kernel.contract_diag_mode(&xp, gvar.as_slice());
            for (g, d) in gvals.iter_mut().zip(&gdiag) {
                *g += d;
            }
            let kraw: Vec<f64> = p.kidx.iter().map(|&i| raw[i]).collect();
            let deriv = self.options.covar_module.transform_derivatives(&kraw);
            for (k, &gi) in p.kidx.iter().enumerate() {
                grad[gi] += gvals[k] * deriv[k];
            }
            // Inducing points.
            if let Some(ir) = &self.layout.inducing {
                for i in 0..mi {
                    let (_, dzx) = p.kernel.cross_with_grad(&p.zx, i, &p.zx); // (mi + n) x dt
                    for l in 0..dt {
                        let mut g = 0.0;
                        for j in 0..mi {
                            g += 2.0 * gkzz[(i, j)] * dzx[(j, l)];
                        }
                        for j in 0..n {
                            g += gc[(i, j)] * dzx[(mi + j, l)];
                        }
                        grad[ir.start + i * dt + l] += g;
                    }
                }
            }
        }
        let mut log_prior = 0.0;
        // Likelihood priors (on every noise value).
        if let Some(prior) = lik_spec.noise_prior {
            for &v in &lvals {
                log_prior += prior.log_prob(v);
            }
        }
        // Mean priors.
        if let VariationalMean::Constant(spec) = self.options.mean_module {
            if let Some(prior) = spec.prior {
                for &c in &raw[self.layout.constant.clone()] {
                    log_prior += prior.log_prob(c);
                    if with_grad {
                        // The constant's raw value is its value.
                    }
                }
            }
        }
        // Kernel priors.
        let kernel_reps = if self.options.covar_batched { m } else { 1 };
        for part in parts.iter().take(kernel_reps) {
            let kernel = &part.kernel;
            let lp = if with_grad {
                let (lp, g) = kernel.log_prior_and_grad()?;
                let kraw: Vec<f64> = part.kidx.iter().map(|&i| raw[i]).collect();
                let deriv = self.options.covar_module.transform_derivatives(&kraw);
                for (k, &gi) in part.kidx.iter().enumerate() {
                    grad[gi] += g[k] * deriv[k] / num_data;
                }
                lp
            } else {
                kernel.log_prior()?
            };
            log_prior += lp;
        }
        if with_grad {
            // Likelihood raw noise gradients (data term and priors).
            let lr = self.layout.likelihood.clone();
            let prior = lik_spec.noise_prior;
            match self.options.likelihood {
                VariationalLikelihood::Gaussian(_) => {
                    let v = lvals[0];
                    let mut g = g_noise[0];
                    if let Some(p) = prior {
                        g += p.grad_log_prob(v) / num_data;
                    }
                    grad[lr.start] += g * nc.transform_derivative(lraw[0]);
                }
                VariationalLikelihood::Multitask(_) => {
                    let mut g_global = 0.0;
                    for t in 0..m {
                        let mut g = g_noise[t];
                        g_global += g_noise[t];
                        if let Some(p) = prior {
                            g += p.grad_log_prob(lvals[t]) / num_data;
                        }
                        grad[lr.start + t] += g * nc.transform_derivative(lraw[t]);
                    }
                    if let Some(p) = prior {
                        g_global += p.grad_log_prob(lvals[m]) / num_data;
                    }
                    grad[lr.start + m] += g_global * nc.transform_derivative(lraw[m]);
                }
            }
            if let VariationalMean::Constant(spec) = self.options.mean_module {
                if let Some(prior) = spec.prior {
                    for i in self.layout.constant.clone() {
                        grad[i] += prior.grad_log_prob(raw[i]) / num_data;
                    }
                }
            }
        }
        let terms = ElboTerms {
            log_likelihood: ll_total / nf,
            kl_divergence: kl_total * beta / num_data,
            log_prior: log_prior / num_data,
        };
        Ok((terms, with_grad.then_some(grad)))
    }

    // ------------------------------------------------------------------
    // Fitting helpers
    // ------------------------------------------------------------------

    /// BoTorch's `sample_all_priors(mll.model)` for the approximate model:
    /// every kernel and mean parameter with a prior is drawn from it (the
    /// likelihood, which is not part of `mll.model`, keeps its values).
    pub fn sample_all_priors(&self, raw: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let mut out = raw.to_vec();
        if let VariationalMean::Constant(spec) = self.options.mean_module {
            if let Some(prior) = spec.prior {
                for i in self.layout.constant.clone() {
                    out[i] = prior.sample(rng);
                }
            }
        }
        let reps = if self.options.covar_batched {
            self.num_outputs
        } else {
            1
        };
        for t in 0..reps {
            let idx = self.kernel_indices(t);
            let mut kraw: Vec<f64> = idx.iter().map(|&i| out[i]).collect();
            sample_tree_priors(&self.options.covar_module, &mut kraw, rng);
            for (k, &i) in idx.iter().enumerate() {
                out[i] = kraw[k];
            }
        }
        out
    }
}

/// The variational ELBO of a [`SingleTaskVariationalGP`] (with GPyTorch's
/// `num_data` and `beta`) as the objective of
/// [`crate::fit::fit_gpytorch_mll`] (BoTorch's `_fit_fallback_approximate`
/// with `fit_gpytorch_mll_scipy`: L-BFGS-B over all raw parameters, retried
/// from prior samples of the kernel and mean parameters) and of
/// [`crate::optim::fit::fit_gpytorch_mll_torch`].
#[derive(Clone, Debug)]
pub struct ElboFit {
    /// The model (holds the fitted parameters afterwards).
    pub model: SingleTaskVariationalGP,
    /// GPyTorch's `num_data`.
    pub num_data: f64,
    /// GPyTorch's `beta`.
    pub beta: f64,
}

impl Fittable for ElboFit {
    type Output = FitResult;

    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        let x0 = self.model.raw.clone();
        let bounds = self.model.raw_parameter_bounds();
        let (result, x) = {
            let m = &self.model;
            let (nd, beta) = (self.num_data, self.beta);
            minimize_with_retries(
                &x0,
                &bounds,
                options,
                |x| {
                    m.elbo_value_and_grad(x, nd, beta)
                        .map(|(t, g)| (t.value(), g))
                },
                |rng| m.sample_all_priors(&x0, rng),
            )?
        };
        self.model.set_raw_parameters(&x)?;
        Ok(result)
    }
}

impl MllModel for ElboFit {
    fn raw_parameter_vector(&self) -> Vec<f64> {
        self.model.raw.clone()
    }

    fn set_raw_parameter_vector(&mut self, values: &[f64]) -> Result<()> {
        self.model.set_raw_parameters(values)
    }

    fn raw_bounds(&self) -> Vec<(f64, f64)> {
        self.model.raw_parameter_bounds()
    }

    fn mll_values_and_gradient(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let (t, g) = self
            .model
            .elbo_value_and_grad(x, self.num_data, self.beta)?;
        Ok((vec![t.value()], g))
    }
}

impl SingleTaskVariationalGP {
    /// The posterior of every row of `x` on its own (GPyTorch's marginal
    /// variances, i.e. the lazy `diag=True` kernel values), over all
    /// outputs: means (`b × m`) and diagonal output covariances (`b × m²`,
    /// independent outputs).
    fn point_posteriors_impl(
        &self,
        x: &Matrix,
        observation_noise: bool,
    ) -> Result<PointPosteriors> {
        let xt = self.transform_inputs(x)?;
        let caches = self.cache()?;
        let (b, m) = (xt.nrows(), self.num_outputs);
        let mut mean = Matrix::zeros(b, m);
        let mut covariance = Matrix::zeros(b, m * m);
        for (t, c) in caches.iter().enumerate() {
            let (mu, var, _) = self.latent_marginals(c, &xt, false)?;
            let (om, os) = self.affine(t);
            let noise = if observation_noise { c.noise } else { 0.0 };
            for i in 0..b {
                mean[(i, t)] = om + os * mu[i];
                covariance[(i, t * m + t)] = os * (var[i] + noise) * os;
            }
        }
        Ok(PointPosteriors { mean, covariance })
    }

    /// The posterior at one point with GPyTorch's marginal variance and the
    /// derivatives with respect to the point.
    fn point_posterior_with_grad_impl(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        let xr = Matrix::from_row_slice(1, x.len(), x);
        self.check_columns(&xr)?;
        let (xt, jac) = match &self.options.input_transform {
            Some(chain) => {
                let (xt, tape) = chain.transform_with_tape(&xr)?;
                let jacs = tape.jacobians_through(chain)?;
                (xt, Some(jacs.into_iter().next().expect("one point")))
            }
            None => (xr.clone(), None),
        };
        let caches = self.cache()?;
        let m = self.num_outputs;
        let d = x.len();
        let dt = xt.ncols();
        let mut mean = Vector::zeros(m);
        let mut cov = Matrix::zeros(m, m);
        let mut dmean = vec![Vector::zeros(m); d];
        let mut dcov = vec![Matrix::zeros(m, m); d];
        for (t, c) in caches.iter().enumerate() {
            let (mu, var, grads) = self.latent_marginals(c, &xt, true)?;
            let (om, os) = self.affine(t);
            mean[t] = om + os * mu[0];
            cov[(t, t)] = os * var[0] * os;
            let g = &grads[0];
            for j in 0..d {
                let (mut gm, mut gv) = (0.0, 0.0);
                for l in 0..dt {
                    let f = match &jac {
                        Some(jm) => jm[(l, j)],
                        None => {
                            if l == j {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };
                    gm += f * g[(0, l)];
                    gv += f * g[(1, l)];
                }
                dmean[j][t] = os * gm;
                dcov[j][(t, t)] = os * os * gv;
            }
        }
        Ok(JointPosteriorGrad {
            mean,
            covariance: cov,
            dmean,
            dcov,
            num_outputs: m,
        })
    }
}

impl Model for SingleTaskVariationalGP {
    fn dim(&self) -> usize {
        self.dim
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.point_posteriors_impl(x, false)
    }

    fn point_posteriors_with_noise(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.point_posteriors_impl(x, true)
    }

    fn point_posteriors_joint(&self, x: &Matrix) -> Result<PointPosteriors> {
        let (b, m) = (x.nrows(), self.num_outputs);
        let mut mean = Matrix::zeros(b, m);
        let mut covariance = Matrix::zeros(b, m * m);
        for i in 0..b {
            let p = self.posterior(&x.rows(i, 1).into_owned(), false)?;
            for a in 0..m {
                mean[(i, a)] = p.mean[a];
                for c in 0..m {
                    covariance[(i, a * m + c)] = p.covariance[(a, c)];
                }
            }
        }
        Ok(PointPosteriors { mean, covariance })
    }

    fn point_posterior_joint_with_grad(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        self.joint_posterior_with_grad(&Matrix::from_row_slice(1, x.len(), x))
    }

    fn point_posterior_with_grad(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        self.point_posterior_with_grad_impl(x)
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        SingleTaskVariationalGP::posterior(self, x, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        SingleTaskVariationalGP::joint_posterior_with_grad(self, x)
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        SingleTaskVariationalGP::joint_posterior_vjp(
            self,
            x,
            rows,
            mean_adjoint,
            covariance_adjoint,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::GreaterThan;
    use crate::kernels::KernelKind;
    use crate::models::gp::get_gaussian_likelihood_with_lognormal_prior;
    use crate::priors::Prior;

    fn rbf_tree(d: usize) -> KernelNode {
        KernelNode::stationary(KernelKind::Rbf, Some(d))
            .unwrap()
            .with_prior(
                "raw_lengthscale",
                Prior::LogNormal {
                    loc: 1.0,
                    scale: 1.2,
                    float32: false,
                    torch_log: None,
                },
            )
            .unwrap()
    }

    fn scale_tree(d: usize) -> KernelNode {
        KernelNode::scale(KernelNode::stationary(KernelKind::Matern52, Some(d)).unwrap())
    }

    fn data(n: usize, d: usize, m: usize, seed: u64) -> (Matrix, Matrix) {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let x = Matrix::from_fn(n, d, |_, _| next());
        let y = Matrix::from_fn(n, m, |i, t| {
            (3.0 * x[(i, 0)] + t as f64).sin() + 0.3 * x[(i, d - 1)]
        });
        (x, y)
    }

    fn model(m: usize, learn: bool, tree: KernelNode, batched: bool) -> SingleTaskVariationalGP {
        let (x, y) = data(9, 2, m, 3);
        let z = Matrix::from_fn(4, 2, |i, j| x[(2 * i, j)]);
        let likelihood = if m == 1 {
            VariationalLikelihood::Gaussian(get_gaussian_likelihood_with_lognormal_prior())
        } else {
            VariationalLikelihood::Multitask(GaussianLikelihoodSpec {
                noise_prior: None,
                noise_constraint: GreaterThan::new(
                    1e-4,
                    Some(crate::constraints::ParameterTransform::Softplus),
                    None,
                )
                .into(),
            })
        };
        let options = SingleTaskVariationalGPOptions {
            covar_module: tree,
            covar_batched: batched,
            mean_module: VariationalMean::Constant(ConstantMeanSpec {
                prior: Some(Prior::Normal {
                    loc: 0.1,
                    scale: 2.0,
                }),
                constraint: None,
            }),
            mean_batched: batched,
            likelihood,
            learn_inducing_points: learn,
            jitter: VARIATIONAL_JITTER,
            input_transform: None,
            outcome_transform: None,
        };
        let mut mdl = SingleTaskVariationalGP::new(x, Some(y), z, m, 2, None, options).unwrap();
        // Random variational parameters and hyperparameters.
        let mut raw = mdl.raw_parameters().to_vec();
        for (k, v) in raw.iter_mut().enumerate() {
            let s = ((k as f64) * 0.7311).sin();
            *v += 0.3 * s;
        }
        mdl.set_raw_parameters(&raw).unwrap();
        mdl
    }

    fn check_gradient(mdl: &SingleTaskVariationalGP) {
        let raw = mdl.raw_parameters().to_vec();
        let (t0, g) = mdl.elbo_value_and_grad(&raw, 12.0, 0.7).unwrap();
        let mut worst: f64 = 0.0;
        let gmax = g.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        for k in 0..raw.len() {
            let h = 1e-6 * raw[k].abs().max(1.0);
            let mut rp = raw.clone();
            rp[k] += h;
            let mut rm = raw.clone();
            rm[k] -= h;
            let fp = mdl.elbo_value_and_grad(&rp, 12.0, 0.7).unwrap().0.value();
            let fm = mdl.elbo_value_and_grad(&rm, 12.0, 0.7).unwrap().0.value();
            let fd = (fp - fm) / (2.0 * h);
            worst = worst.max((fd - g[k]).abs() / gmax.max(1e-12));
        }
        assert!(t0.value().is_finite());
        assert!(worst < 1e-6, "worst relative gradient error {worst}");
    }

    #[test]
    fn elbo_gradient_matches_finite_differences() {
        check_gradient(&model(1, true, rbf_tree(2), false));
        check_gradient(&model(1, false, scale_tree(2), false));
        check_gradient(&model(2, true, scale_tree(2), true));
        check_gradient(&model(3, true, rbf_tree(2), false));
        // The arc-cosine kernel, whose `diag=True` values differ from the
        // diagonal of its full matrix.
        let bnn = KernelNode::scale(KernelNode::infinite_width_bnn(2, 1e-7).unwrap());
        check_gradient(&model(1, true, bnn, false));
    }

    #[test]
    fn point_posterior_gradients_match_finite_differences() {
        let bnn = KernelNode::scale(KernelNode::infinite_width_bnn(2, 1e-7).unwrap());
        for mdl in [
            model(1, true, bnn, false),
            model(2, true, scale_tree(2), true),
        ] {
            let x = [0.3, 0.6];
            let g = Model::point_posterior_with_grad(&mdl, &x).unwrap();
            let pp = Model::point_posteriors(&mdl, &Matrix::from_row_slice(1, 2, &x)).unwrap();
            let m = mdl.num_outputs;
            for t in 0..m {
                assert!((pp.mean[(0, t)] - g.mean[t]).abs() < 1e-14);
                assert!((pp.covariance[(0, t * m + t)] - g.covariance[(t, t)]).abs() < 1e-14);
            }
            for k in 0..2 {
                let h = 1e-6;
                let mut xp = x;
                xp[k] += h;
                let mut xm = x;
                xm[k] -= h;
                let pp = Model::point_posteriors(&mdl, &Matrix::from_row_slice(1, 2, &xp)).unwrap();
                let pm = Model::point_posteriors(&mdl, &Matrix::from_row_slice(1, 2, &xm)).unwrap();
                for t in 0..m {
                    let dm = (pp.mean[(0, t)] - pm.mean[(0, t)]) / (2.0 * h);
                    let dv =
                        (pp.covariance[(0, t * m + t)] - pm.covariance[(0, t * m + t)]) / (2.0 * h);
                    assert!((dm - g.dmean[k][t]).abs() < 1e-6, "{dm} {}", g.dmean[k][t]);
                    assert!(
                        (dv - g.dcov[k][(t, t)]).abs() < 1e-6,
                        "{dv} {}",
                        g.dcov[k][(t, t)]
                    );
                }
            }
        }
    }

    #[test]
    fn posterior_gradients_match_finite_differences() {
        for mdl in [
            model(1, true, rbf_tree(2), false),
            model(2, true, scale_tree(2), true),
        ] {
            let x = Matrix::from_row_slice(3, 2, &[0.1, 0.7, 0.4, 0.2, 0.9, 0.5]);
            let g = mdl.joint_posterior_with_grad(&x).unwrap();
            let len = g.mean.len();
            let ma = Vector::from_fn(len, |i, _| 0.3 + i as f64);
            let ca = Matrix::from_fn(len, len, |i, j| ((i * 3 + j) as f64 * 0.37).cos());
            let vjp = mdl.joint_posterior_vjp(&x, 0..3, &ma, &ca).unwrap();
            for p in 0..3 {
                for k in 0..2 {
                    let h = 1e-6;
                    let mut xp = x.clone();
                    xp[(p, k)] += h;
                    let mut xm = x.clone();
                    xm[(p, k)] -= h;
                    let pp = mdl.posterior(&xp, false).unwrap();
                    let pm = mdl.posterior(&xm, false).unwrap();
                    let dm = (&pp.mean - &pm.mean) / (2.0 * h);
                    let dc = (&pp.covariance - &pm.covariance) / (2.0 * h);
                    let j = p * 2 + k;
                    assert!((&dm - &g.dmean[j]).amax() < 1e-6);
                    assert!((&dc - &g.dcov[j]).amax() < 1e-6);
                    let s = ma.dot(&dm) + ca.component_mul(&dc).sum();
                    assert!((s - vjp[(p, k)]).abs() < 1e-5 * s.abs().max(1.0));
                }
            }
        }
    }
}
