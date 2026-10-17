//! [`KroneckerMultiTaskGP`]: BoTorch's `KroneckerMultiTaskGP`
//! (`botorch.models.multitask`), a multi-task GP with an ICM kernel for the
//! "block design" (every task observed at every training point).
//!
//! # The model
//!
//! For training inputs `X` (`n × d`) and targets `Y` (`n × m`):
//!
//! * the covariance of the observations of tasks `s, t` at points `x, x'` is
//!   `k_x(x, x') · B[s, t]` (GPyTorch's `MultitaskKernel`: a data kernel
//!   `covar_module.data_covar_module`, by default BoTorch's dimension-scaled
//!   RBF kernel, and an `IndexKernel` `covar_module.task_covar_module` with
//!   `B = F Fᵀ + diag(v)` of rank `rank`, by default with the prior
//!   `LKJCovariancePrior(m, eta=1.5, SmoothedBoxPrior(e⁻⁶, e^1.25, 0.05))`);
//! * the noise is `I_n ⊗ N` (`MultitaskGaussianLikelihood`): with
//!   `likelihood_rank = 0`, `N = diag(task_noises) + σ² I`
//!   (`likelihood.raw_task_noises`, `likelihood.raw_noise`, both with
//!   BoTorch's `LogNormal(-4, 1)` prior and `GreaterThan(1e-4)` constraint);
//!   with `likelihood_rank = r > 0`, `N = G Gᵀ + σ² I`
//!   (`likelihood.task_noise_covar_factor`, `m × r`);
//! * one constant mean per task (`mean_module.base_means.{t}.raw_constant`);
//! * optional `Standardize(m)` outcome and `Normalize` input transforms
//!   (BoTorch's default: none).
//!
//! The observations are flattened in GPyTorch's interleaved order (index
//! `i·m + t` for task `t` at point `i`); the marginal log likelihood is
//! divided by `n·m`. The joint posteriors of this crate's [`Model`] trait
//! are exact and in output-major order like all multi-output posteriors.
//!
//! # BoTorch's posterior
//!
//! BoTorch's `KroneckerMultiTaskGP.posterior` returns a
//! `MultitaskGPPosterior`: the mean and the marginal variances (a diagonal
//! `MultitaskMultivariateNormal`, interleaved), and samples drawn with
//! Matheron's rule from base samples for the joint prior of training and
//! test points, the training noise and (with observation noise) the test
//! noise ([`KroneckerMultiTaskGP::matheron_samples`]). Its variances are
//! wrong for task-dependent noise (the default likelihood after fitting):
//! the closed form it uses holds for homoskedastic noise only (fixed in
//! M-Colley/botorch `fix/models-posteriors`). The variances here are the
//! exact posterior variances, which are also those of BoTorch's samples.

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use rand::rngs::StdRng;

use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::kernels::eval::TreePrep;
use crate::kernels::tree::sample_tree_priors;
use crate::kernels::{
    get_covar_module_with_dim_scaled_prior, KernelNode, KernelType, MatrixPrior, TreeKernel,
};
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::models::gp::{get_gaussian_likelihood_with_lognormal_prior, GaussianLikelihoodSpec};
use crate::models::model::check_vjp_shapes_expanded;
use crate::models::multitask::gp::kernel_spec_tree;
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::priors::Prior;
use crate::transforms::{Normalize, Standardize};
use crate::{Matrix, Vector};

/// `log(2π)`.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// BoTorch's default `sd_prior` of the task covariance prior:
/// `SmoothedBoxPrior(exp(-6), exp(1.25), 0.05)` (float32 parameters).
pub fn default_sd_prior() -> Prior {
    Prior::smoothed_box((-6.0f64).exp(), 1.25f64.exp(), 0.05)
        .expect("valid SmoothedBox parameters")
        .rounded_to_f32()
}

/// Options of [`KroneckerMultiTaskGP::new`] (BoTorch's keyword arguments).
#[derive(Clone, Debug, PartialEq)]
pub struct KroneckerMultiTaskGPOptions {
    /// The data kernel (`data_covar_module`; default: BoTorch's
    /// dimension-scaled RBF kernel).
    pub data_covar_module: Option<KernelNode>,
    /// The task prior (`task_covar_prior`; default: an
    /// `LKJCovariancePrior(m, eta, sd_prior)`).
    pub task_covar_prior: Option<MatrixPrior>,
    /// `eta` of the default task prior (BoTorch's keyword `eta`, 1.5).
    pub eta: f64,
    /// `sd_prior` of the default task prior (default:
    /// [`default_sd_prior`]).
    pub sd_prior: Option<Prior>,
    /// The task kernel itself (an `IndexKernel` on the `m` tasks, without
    /// `active_dims`), instead of `IndexKernel(m, rank)` with the task prior
    /// (e.g. with initial values drawn as GPyTorch draws them).
    pub task_covar_module: Option<KernelNode>,
    /// The rank of the task covariance (default `m`).
    pub rank: Option<usize>,
    /// The noise prior and constraint of the `MultitaskGaussianLikelihood`
    /// (default: BoTorch's `LogNormal(-4, 1)` and `GreaterThan(1e-4)`).
    pub likelihood: Option<GaussianLikelihoodSpec>,
    /// The rank of the task noise covariance (`likelihood_rank`, default 0:
    /// diagonal).
    pub likelihood_rank: usize,
    /// The outcome transform (BoTorch's default: none).
    pub outcome_transform: Option<Standardize>,
    /// The input transform.
    pub input_transform: Option<Normalize>,
}

impl Default for KroneckerMultiTaskGPOptions {
    fn default() -> Self {
        Self {
            data_covar_module: None,
            task_covar_prior: None,
            eta: 1.5,
            sd_prior: None,
            task_covar_module: None,
            rank: None,
            likelihood: None,
            likelihood_rank: 0,
            outcome_transform: None,
            input_transform: None,
        }
    }
}

/// Quantities derived from the hyperparameters and cached for prediction.
#[derive(Debug)]
struct KronCache {
    data: TreeKernel,
    train_prep: TreePrep,
    /// The task covariance `B` (`m × m`).
    b: Matrix,
    /// The task noise covariance `N` (`m × m`).
    noise: Matrix,
    chol: Arc<CholeskyFactor>,
    /// `K⁻¹ (y - μ)` (interleaved, transformed space).
    alpha: Vector,
}

/// BoTorch's `KroneckerMultiTaskGP` (see the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct KroneckerMultiTaskGP {
    train_x: Matrix,
    train_y: Matrix,
    x_tf: Matrix,
    y_tf: Matrix,
    input_transform: Option<Normalize>,
    outcome_transform: Option<Standardize>,
    data: KernelNode,
    task: KernelNode,
    noise_spec: GaussianLikelihoodSpec,
    likelihood_rank: usize,
    raw: Vector,
    cache: OnceLock<std::result::Result<Arc<KronCache>, BotorchError>>,
}

fn check_no_nan(m: &Matrix, what: &str) -> Result<()> {
    if m.iter().any(|v| v.is_nan()) {
        return Err(BotorchError::InputData(format!(
            "{what} contains NaN values."
        )));
    }
    Ok(())
}

impl KroneckerMultiTaskGP {
    /// Builds the model from `train_x` (`n × d`) and `train_y` (`n × m`); see
    /// [`KroneckerMultiTaskGPOptions`].
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
    /// [`BotorchError::InputData`] for NaN values and
    /// [`BotorchError::InvalidArgument`] for invalid ranks or kernels.
    pub fn new(
        train_x: Matrix,
        train_y: Matrix,
        options: KroneckerMultiTaskGPOptions,
    ) -> Result<Self> {
        let (n, d) = train_x.shape();
        let m = train_y.ncols();
        if n == 0 || d == 0 || m == 0 {
            return Err(BotorchError::InputData(format!(
                "KroneckerMultiTaskGP needs training data, got X of shape {n}x{d} and Y with {m} \
                 columns"
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
        let input_transform = match options.input_transform {
            Some(mut tf) => {
                if tf.dim() != d {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "input transform has dimension {} but train_X has {d} columns",
                        tf.dim()
                    )));
                }
                tf.fit(&train_x)?;
                Some(tf)
            }
            None => None,
        };
        let x_tf = match &input_transform {
            Some(tf) => tf.transform(&train_x)?,
            None => train_x.clone(),
        };
        let mut outcome_transform = options.outcome_transform;
        let y_tf = match outcome_transform.as_mut() {
            Some(st) => st.fit_transform(&train_y, None)?.0,
            None => train_y.clone(),
        };
        check_no_nan(&y_tf, "transformed train_Y")?;
        let rank = options.rank.unwrap_or(m);
        let task = match options.task_covar_module {
            Some(task) => {
                let ok =
                    matches!(task.kind(), KernelType::Index { num_tasks, .. } if *num_tasks == m);
                if !ok || task.active_dims().is_some() {
                    return Err(BotorchError::InvalidArgument(format!(
                        "the task kernel must be an IndexKernel on the {m} tasks without \
                         active_dims"
                    )));
                }
                task
            }
            None => {
                let prior = options
                    .task_covar_prior
                    .unwrap_or(MatrixPrior::LkjCovariance {
                        n: m,
                        eta: options.eta,
                        sd_prior: options.sd_prior.unwrap_or_else(default_sd_prior),
                    });
                KernelNode::index(m, rank)?.with_task_prior(prior)?
            }
        };
        let data = match options.data_covar_module {
            Some(node) => node,
            None => kernel_spec_tree(&get_covar_module_with_dim_scaled_prior(d, true))?,
        };
        data.validate(d)?;
        task.validate(1)?;
        let lr = options.likelihood_rank;
        if lr > m {
            return Err(BotorchError::InvalidArgument(format!(
                "Cannot have rank ({lr}) greater than num_tasks ({m})"
            )));
        }
        let noise_spec = options
            .likelihood
            .unwrap_or_else(get_gaussian_likelihood_with_lognormal_prior);
        let init = noise_spec.initial_raw_noise();
        let mut raw = Vec::new();
        if lr == 0 {
            raw.extend(std::iter::repeat_n(init, m));
        } else {
            raw.extend(std::iter::repeat_n(0.0, m * lr));
        }
        raw.push(init);
        raw.extend(std::iter::repeat_n(0.0, m));
        raw.extend(task.initial_raw());
        raw.extend(data.initial_raw());
        Ok(Self {
            train_x,
            train_y,
            x_tf,
            y_tf,
            input_transform,
            outcome_transform,
            data,
            task,
            noise_spec,
            likelihood_rank: lr,
            raw: Vector::from_vec(raw),
            cache: OnceLock::new(),
        })
    }

    // ------------------------------------------------------------------
    // Data and configuration
    // ------------------------------------------------------------------

    /// Training inputs in the original space (`n × d`).
    pub fn train_inputs(&self) -> &Matrix {
        &self.train_x
    }

    /// Training targets in the original space (`n × m`).
    pub fn train_targets(&self) -> &Matrix {
        &self.train_y
    }

    /// Training targets after the outcome transform (`n × m`).
    pub fn transformed_train_targets(&self) -> &Matrix {
        &self.y_tf
    }

    /// Input dimension `d`.
    pub fn dim(&self) -> usize {
        self.train_x.ncols()
    }

    /// Number of training points `n`.
    pub fn num_train(&self) -> usize {
        self.train_x.nrows()
    }

    /// Number of tasks (outputs) `m`.
    pub fn num_tasks(&self) -> usize {
        self.train_y.ncols()
    }

    /// The rank of the task noise covariance (0: diagonal).
    pub fn likelihood_rank(&self) -> usize {
        self.likelihood_rank
    }

    /// The noise prior and constraint.
    pub fn likelihood(&self) -> &GaussianLikelihoodSpec {
        &self.noise_spec
    }

    /// The data kernel.
    pub fn data_covar_module(&self) -> &KernelNode {
        &self.data
    }

    /// The task kernel (`IndexKernel`).
    pub fn task_covar_module(&self) -> &KernelNode {
        &self.task
    }

    /// The fitted input transform, if any.
    pub fn input_transform(&self) -> Option<&Normalize> {
        self.input_transform.as_ref()
    }

    /// The fitted outcome transform, if any.
    pub fn outcome_transform(&self) -> Option<&Standardize> {
        self.outcome_transform.as_ref()
    }

    // ------------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------------

    fn num_noise_params(&self) -> usize {
        let m = self.num_tasks();
        if self.likelihood_rank == 0 {
            m + 1
        } else {
            m * self.likelihood_rank + 1
        }
    }

    fn mean_range(&self) -> Range<usize> {
        let s = self.num_noise_params();
        s..s + self.num_tasks()
    }

    fn task_range(&self) -> Range<usize> {
        let s = self.mean_range().end;
        s..s + self.task.num_raw()
    }

    fn data_range(&self) -> Range<usize> {
        let s = self.task_range().end;
        s..s + self.data.num_raw()
    }

    /// The raw hyperparameters in BoTorch's `named_parameters()` order.
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
        let noise = self.noise_matrix(values);
        if noise.diagonal().iter().any(|v| v.is_nan() || *v < 0.0) {
            return Err(BotorchError::InvalidArgument(
                "noise must be non-negative".into(),
            ));
        }
        TreeKernel::build(&self.task, &values[self.task_range()], 1)?;
        TreeKernel::build(&self.data, &values[self.data_range()], self.dim())?;
        self.raw = Vector::from_column_slice(values);
        self.cache = OnceLock::new();
        Ok(())
    }

    /// BoTorch parameter names and shapes in flat-vector order.
    pub fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let m = self.num_tasks();
        let mut out = Vec::new();
        if self.likelihood_rank == 0 {
            out.push(("likelihood.raw_task_noises".to_string(), vec![m]));
        } else {
            out.push((
                "likelihood.task_noise_covar_factor".to_string(),
                vec![m, self.likelihood_rank],
            ));
        }
        out.push(("likelihood.raw_noise".to_string(), vec![1]));
        for t in 0..m {
            out.push((format!("mean_module.base_means.{t}.raw_constant"), vec![]));
        }
        out.extend(self.task.param_shapes("covar_module.task_covar_module."));
        out.extend(self.data.param_shapes("covar_module.data_covar_module."));
        out
    }

    /// BoTorch parameter names and sizes in flat-vector order.
    pub fn parameter_names(&self) -> Vec<(String, usize)> {
        self.parameter_shapes()
            .into_iter()
            .map(|(name, shape)| (name, shape.iter().product()))
            .collect()
    }

    /// Optimizer bounds of every raw parameter in flat-vector order.
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let nb = self.noise_spec.noise_constraint.raw_bounds();
        let m = self.num_tasks();
        let mut b = Vec::with_capacity(self.raw.len());
        if self.likelihood_rank == 0 {
            b.extend(std::iter::repeat_n(nb, m));
        } else {
            b.extend(std::iter::repeat_n(free, m * self.likelihood_rank));
        }
        b.push(nb);
        b.extend(std::iter::repeat_n(free, m));
        b.extend(self.task.raw_bounds());
        b.extend(self.data.raw_bounds());
        b
    }

    /// The task noise covariance `N` (`m × m`) of raw parameters `raw`.
    fn noise_matrix(&self, raw: &[f64]) -> Matrix {
        let m = self.num_tasks();
        let c = self.noise_spec.noise_constraint;
        let sigma2 = c.transform(raw[self.num_noise_params() - 1]);
        let mut n = if self.likelihood_rank == 0 {
            Matrix::from_diagonal(&Vector::from_fn(m, |t, _| c.transform(raw[t])))
        } else {
            let g =
                Matrix::from_row_slice(m, self.likelihood_rank, &raw[..m * self.likelihood_rank]);
            &g * g.transpose()
        };
        for t in 0..m {
            n[(t, t)] += sigma2;
        }
        n
    }

    /// The task noise covariance `N` at the current hyperparameters.
    pub fn task_noise_covariance(&self) -> Matrix {
        self.noise_matrix(self.raw.as_slice())
    }

    /// The task noises (`likelihood.task_noises`, rank 0) and the global
    /// noise (`likelihood.noise`).
    pub fn noises(&self) -> (Option<Vector>, f64) {
        let c = self.noise_spec.noise_constraint;
        let m = self.num_tasks();
        let raw = self.raw.as_slice();
        let tn =
            (self.likelihood_rank == 0).then(|| Vector::from_fn(m, |t, _| c.transform(raw[t])));
        (tn, c.transform(raw[self.num_noise_params() - 1]))
    }

    /// The constant means (transformed space).
    pub fn constants(&self) -> Vector {
        Vector::from_column_slice(&self.raw.as_slice()[self.mean_range()])
    }

    /// The task covariance matrix `B = F Fᵀ + diag(v)` (`covar_matrix`).
    ///
    /// # Errors
    /// See [`TreeKernel::build`].
    pub fn task_covariance(&self) -> Result<Matrix> {
        self.task_matrix(self.raw.as_slice())
    }

    fn task_points(&self) -> Matrix {
        Matrix::from_fn(self.num_tasks(), 1, |t, _| t as f64)
    }

    fn task_matrix(&self, raw: &[f64]) -> Result<Matrix> {
        let tree = TreeKernel::build(&self.task, &raw[self.task_range()], 1)?;
        tree.covariance_symmetric(&self.task_points())
    }

    // ------------------------------------------------------------------
    // Marginal log likelihood and fitting
    // ------------------------------------------------------------------

    /// The full training covariance (`nm × nm`, interleaved) and its parts.
    fn train_covariance(&self, kx: &Matrix, b: &Matrix, noise: &Matrix) -> Matrix {
        let (n, m) = (self.num_train(), self.num_tasks());
        Matrix::from_fn(n * m, n * m, |a, c| {
            let (i, s) = (a / m, a % m);
            let (j, t) = (c / m, c % m);
            kx[(i, j)] * b[(s, t)] + if i == j { noise[(s, t)] } else { 0.0 }
        })
    }

    fn residuals(&self, raw: &[f64]) -> Vector {
        let (n, m) = (self.num_train(), self.num_tasks());
        let c = &raw[self.mean_range()];
        Vector::from_fn(n * m, |a, _| self.y_tf[(a / m, a % m)] - c[a % m])
    }

    /// The exact marginal log likelihood (divided by `n·m`, priors included).
    ///
    /// # Errors
    /// [`BotorchError::NotPsd`] / [`BotorchError::Nan`] if the covariance
    /// cannot be factorized.
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
        let (n, m) = (self.num_train(), self.num_tasks());
        let nm = (n * m) as f64;
        let data = TreeKernel::build(&self.data, &raw[self.data_range()], self.dim())?;
        let task = TreeKernel::build(&self.task, &raw[self.task_range()], 1)?;
        let xp = data.prepare(&self.x_tf)?;
        let tp = task.prepare(&self.task_points())?;
        let kx = data.symmetric(&xp);
        let b = task.symmetric(&tp);
        let noise = self.noise_matrix(raw);
        let k = self.train_covariance(&kx, &b, &noise);
        let chol = psd_safe_cholesky(&k)?;
        let resid = self.residuals(raw);
        let v = chol.solve_lower_vec(&resid);
        let mut res = -0.5 * (v.dot(&v) + chol.logdet() + nm * LN_2PI);
        // Priors in GPyTorch's `named_priors()` order.
        let c = self.noise_spec.noise_constraint;
        let sigma_idx = self.num_noise_params() - 1;
        if let Some(p) = self.noise_spec.noise_prior {
            if self.likelihood_rank == 0 {
                for &r in &raw[..m] {
                    res += p.log_prob(c.transform(r));
                }
            }
            res += p.log_prob(c.transform(raw[sigma_idx]));
        }
        let (tg, dg) = if with_grad {
            let (lt, gt) = task.log_prior_and_grad()?;
            let (ld, gd) = data.log_prior_and_grad()?;
            res += lt + ld;
            (Some(gt), Some(gd))
        } else {
            res += task.log_prior()? + data.log_prior()?;
            (None, None)
        };
        let value = res / nm;
        if !with_grad {
            return Ok((value, None));
        }
        let alpha = chol.solve_lower_transpose_vec(&v);
        let mut w = chol.inverse();
        w.ger(1.0, &alpha, &alpha, -1.0); // W = ααᵀ - K⁻¹
        let mut grad = Vector::zeros(raw.len());
        // Noise: dmll/dN[s, t] = ½ Σ_i W[(i, s), (i, t)].
        let mut wn = Matrix::zeros(m, m);
        for i in 0..n {
            for s in 0..m {
                for t in 0..m {
                    wn[(s, t)] += w[(i * m + s, i * m + t)];
                }
            }
        }
        let prior = self.noise_spec.noise_prior;
        if self.likelihood_rank == 0 {
            for s in 0..m {
                let val = c.transform(raw[s]);
                let mut g = 0.5 * wn[(s, s)];
                if let Some(p) = prior {
                    g += p.grad_log_prob(val);
                }
                grad[s] = g * c.transform_derivative(raw[s]) / nm;
            }
        } else {
            let r = self.likelihood_rank;
            let gmat = Matrix::from_row_slice(m, r, &raw[..m * r]);
            let ws = (&wn + wn.transpose()) * 0.5;
            let gg = &ws * &gmat; // ½ (Wn + Wnᵀ) G · 2 · ½
            for s in 0..m {
                for k in 0..r {
                    grad[s * r + k] = gg[(s, k)] / nm;
                }
            }
        }
        {
            let val = c.transform(raw[sigma_idx]);
            let mut g = 0.5 * wn.trace();
            if let Some(p) = prior {
                g += p.grad_log_prob(val);
            }
            grad[sigma_idx] = g * c.transform_derivative(raw[sigma_idx]) / nm;
        }
        // Means: dmll/dc_t = Σ_i α[(i, t)].
        let mr = self.mean_range();
        for a in 0..n * m {
            grad[mr.start + a % m] += alpha[a] / nm;
        }
        // Task kernel: W_t[s, t] = Σ_ij W[(i, s), (j, t)] Kx[i, j].
        let mut wt = Matrix::zeros(m, m);
        let mut wx = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let kij = kx[(i, j)];
                let mut acc = 0.0;
                for s in 0..m {
                    for t in 0..m {
                        let wv = w[(i * m + s, j * m + t)];
                        wt[(s, t)] += wv * kij;
                        acc += wv * b[(s, t)];
                    }
                }
                wx[(i, j)] = acc;
            }
        }
        let gt = task.contract(&tp, &wt);
        let tg = tg.expect("prior gradient");
        let tr = self.task_range();
        let td = self.task.transform_derivatives(&raw[tr.clone()]);
        for (j, idx) in tr.enumerate() {
            grad[idx] = (0.5 * gt[j] + tg[j]) * td[j] / nm;
        }
        let gd = data.contract(&xp, &wx);
        let dg = dg.expect("prior gradient");
        let dr = self.data_range();
        let dd = self.data.transform_derivatives(&raw[dr.clone()]);
        for (j, idx) in dr.enumerate() {
            grad[idx] = (0.5 * gd[j] + dg[j]) * dd[j] / nm;
        }
        Ok((value, Some(grad)))
    }

    /// BoTorch's `sample_all_priors` (the noise levels and the data kernel's
    /// univariate priors; the LKJ task prior has no setting closure).
    fn sample_all_priors(&self, initial: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let mut p = initial.to_vec();
        if let Some(prior) = self.noise_spec.noise_prior {
            let c = self.noise_spec.noise_constraint;
            let m = self.num_tasks();
            if self.likelihood_rank == 0 {
                for v in &mut p[..m] {
                    *v = c.inverse_transform(prior.sample(rng));
                }
            }
            let s = self.num_noise_params() - 1;
            p[s] = c.inverse_transform(prior.sample(rng));
        }
        let dr = self.data_range();
        sample_tree_priors(&self.data, &mut p[dr], rng);
        let tr = self.task_range();
        sample_tree_priors(&self.task, &mut p[tr], rng);
        p
    }

    // ------------------------------------------------------------------
    // Prediction
    // ------------------------------------------------------------------

    fn compute_cache(&self) -> Result<KronCache> {
        let raw = self.raw.as_slice();
        let data = TreeKernel::build(&self.data, &raw[self.data_range()], self.dim())?;
        let train_prep = data.prepare(&self.x_tf)?;
        let kx = data.symmetric(&train_prep);
        let b = self.task_matrix(raw)?;
        let noise = self.noise_matrix(raw);
        let k = self.train_covariance(&kx, &b, &noise);
        let chol = psd_safe_cholesky(&k)?;
        let alpha = chol.solve_vec(&self.residuals(raw));
        Ok(KronCache {
            data,
            train_prep,
            b,
            noise,
            chol: Arc::new(chol),
            alpha,
        })
    }

    fn cache(&self) -> Result<Arc<KronCache>> {
        self.cache
            .get_or_init(|| self.compute_cache().map(Arc::new))
            .clone()
    }

    /// `(means, stdvs)` of the outcome transform (zeros and ones without).
    fn outcome_affine(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.num_tasks();
        match &self.outcome_transform {
            Some(st) => (st.means().to_vec(), st.stdvs().to_vec()),
            None => (vec![0.0; m], vec![1.0; m]),
        }
    }

    fn transform_x(&self, x: &Matrix) -> Result<(Matrix, Vec<f64>)> {
        let d = self.dim();
        if x.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected inputs with {d} columns, got {}",
                x.ncols()
            )));
        }
        Ok(match &self.input_transform {
            Some(tf) => (
                tf.transform(x)?,
                tf.coefficient().iter().map(|c| 1.0 / c).collect(),
            ),
            None => (x.clone(), vec![1.0; d]),
        })
    }

    /// The cross-covariance of the output-major test outputs (`(s, p)` at
    /// `s·q + p`) with the interleaved training observations (`(i, t)` at
    /// `i·m + t`): `kx(x_p, X_i) B[s, t]`.
    fn cross(&self, kxs: &Matrix, b: &Matrix) -> Matrix {
        let (q, n) = kxs.shape();
        let m = self.num_tasks();
        Matrix::from_fn(q * m, n * m, |a, c| {
            let (s, p) = (a / q, a % q);
            let (i, t) = (c / m, c % m);
            kxs[(p, i)] * b[(s, t)]
        })
    }

    /// The joint posterior over the `q` rows of `x` and all tasks
    /// (output-major), in the original outcome space.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a column mismatch and
    /// [`BotorchError::NotPsd`] if the training covariance cannot be
    /// factorized.
    pub fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let c = self.cache()?;
        let (xt, _) = self.transform_x(x)?;
        let q = xt.nrows();
        let m = self.num_tasks();
        let prep = c.data.prepare(&xt)?;
        let kxs = c.data.cross(&prep, &c.train_prep);
        let kxx = c.data.symmetric(&prep);
        let ks = self.cross(&kxs, &c.b);
        let consts = self.constants();
        let mean_tf = &ks * &c.alpha + Vector::from_fn(q * m, |a, _| consts[a / q]);
        let mut cov = Matrix::from_fn(q * m, q * m, |a, b| {
            kxx[(a % q, b % q)] * c.b[(a / q, b / q)]
        });
        let mut v = ks.transpose();
        c.chol.solve_lower_mut(&mut v);
        cov.gemm_tr(-1.0, &v, &v, 1.0);
        if observation_noise {
            for s in 0..m {
                for t in 0..m {
                    for p in 0..q {
                        cov[(s * q + p, t * q + p)] += c.noise[(s, t)];
                    }
                }
            }
        }
        let (mu, sd) = self.outcome_affine();
        let mean = Vector::from_fn(q * m, |a, _| mu[a / q] + sd[a / q] * mean_tf[a]);
        let cov = Matrix::from_fn(q * m, q * m, |a, b| sd[a / q] * cov[(a, b)] * sd[b / q]);
        Ok(Posterior::multi_output(mean, cov, m))
    }

    fn joint_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let c = self.cache()?;
        let (xt, jac) = self.transform_x(x)?;
        let (q, d) = xt.shape();
        let (n, m) = (self.num_train(), self.num_tasks());
        let nn = q * m;
        let prep = c.data.prepare(&xt)?;
        let mut kxs = Matrix::zeros(q, n);
        let mut dks = Vec::with_capacity(q);
        let mut dkk = Vec::with_capacity(q);
        for p in 0..q {
            let (k, dk) = c.data.cross_with_grad(&prep, p, &c.train_prep);
            kxs.set_row(p, &k.transpose());
            dks.push(dk);
            dkk.push(c.data.cross_with_grad(&prep, p, &prep).1);
        }
        let kxx = c.data.symmetric(&prep);
        let ks = self.cross(&kxs, &c.b);
        let consts = self.constants();
        let mean_tf = &ks * &c.alpha + Vector::from_fn(nn, |a, _| consts[a / q]);
        let mut cov = Matrix::from_fn(nn, nn, |a, b| kxx[(a % q, b % q)] * c.b[(a / q, b / q)]);
        let mut v = ks.transpose();
        c.chol.solve_lower_mut(&mut v);
        cov.gemm_tr(-1.0, &v, &v, 1.0);
        let mut w = v;
        c.chol.solve_lower_transpose_mut(&mut w); // K⁻¹ K*ᵀ (nm × qm)
        let (mu, sd) = self.outcome_affine();
        let mut dmean = vec![Vector::zeros(nn); q * d];
        let mut dcov = vec![Matrix::zeros(nn, nn); q * d];
        for a in 0..nn {
            let (s, p) = (a / q, a % q);
            for k in 0..d {
                let f = jac[k];
                // ∂K*[a, (i, t)] / ∂x_p[k] = dkxs_p[i, k] B[s, t].
                let mut da = Vector::zeros(n * m);
                for i in 0..n {
                    for t in 0..m {
                        da[i * m + t] = dks[p][(i, k)] * c.b[(s, t)];
                    }
                }
                let j = p * d + k;
                dmean[j][a] += sd[s] * da.dot(&c.alpha) * f;
                let dw = w.tr_mul(&da); // (qm) : Σ_c W[c, b] da[c]
                for bcol in 0..nn {
                    let (t, p2) = (bcol / q, bcol % q);
                    let g = (dkk[p][(p2, k)] * c.b[(s, t)] - dw[bcol]) * f * sd[s] * sd[t];
                    dcov[j][(a, bcol)] += g;
                    dcov[j][(bcol, a)] += g;
                }
            }
        }
        let mean = Vector::from_fn(nn, |a, _| mu[a / q] + sd[a / q] * mean_tf[a]);
        let cov = Matrix::from_fn(nn, nn, |a, b| sd[a / q] * cov[(a, b)] * sd[b / q]);
        Ok(JointPosteriorGrad {
            mean,
            covariance: cov,
            dmean,
            dcov,
            num_outputs: m,
        })
    }

    fn vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        let c = self.cache()?;
        let (xt, jac) = self.transform_x(x)?;
        let (q, d) = xt.shape();
        let (n, m) = (self.num_train(), self.num_tasks());
        let nn = q * m;
        let prep = c.data.prepare(&xt)?;
        let kxs = c.data.cross(&prep, &c.train_prep);
        let ks = self.cross(&kxs, &c.b);
        let (_, sd) = self.outcome_affine();
        // Adjoints in the transformed outcome space.
        let mbar = Vector::from_fn(nn, |a, _| mean_adjoint[a] * sd[a / q]);
        let abar = Matrix::from_fn(nn, nn, |a, b| {
            0.5 * (covariance_adjoint[(a, b)] + covariance_adjoint[(b, a)]) * sd[a / q] * sd[b / q]
        });
        let mut w = ks.transpose();
        c.chol.solve_lower_mut(&mut w);
        c.chol.solve_lower_transpose_mut(&mut w); // K⁻¹ K*ᵀ (nm × qm)
        let mut out = Matrix::zeros(rows.len(), d);
        for p in rows.clone() {
            let (_, dk) = c.data.cross_with_grad(&prep, p, &c.train_prep);
            let (_, dkk) = c.data.cross_with_grad(&prep, p, &prep);
            for s in 0..m {
                let a = s * q + p;
                let a_col = abar.column(a).into_owned();
                // Dk_aᵀ (mbar_a α - 2 W abar[:, a]) + 2 Σ_b abar[a, b] ∂₁k(A_a, A_b).
                let vvec = &c.alpha * mbar[a] - (&w * &a_col) * 2.0;
                for k in 0..d {
                    let mut g = 0.0;
                    for i in 0..n {
                        let mut acc = 0.0;
                        for t in 0..m {
                            acc += c.b[(s, t)] * vvec[i * m + t];
                        }
                        g += dk[(i, k)] * acc;
                    }
                    for bcol in 0..nn {
                        let (t, p2) = (bcol / q, bcol % q);
                        g += 2.0 * a_col[bcol] * dkk[(p2, k)] * c.b[(s, t)];
                    }
                    out[(p - rows.start, k)] += g * jac[k];
                }
            }
        }
        Ok(out)
    }

    /// BoTorch's `MultitaskGPPosterior.rsample_from_base_samples` at the
    /// `q` rows of `x`: samples (`S × (q·m)`, interleaved `p·m + t`, original
    /// outcome space) by Matheron's rule from base samples `z` (`S × total`,
    /// `total = (n + q)·m + n·m`, plus `q·m` with observation noise): the
    /// joint prior of the training and test outputs (`(n + q)·m`, the
    /// interleaved training observations first), the training noise (`n·m`)
    /// and the test noise. The roots are Cholesky factors (BoTorch:
    /// eigendecomposition-based roots), so the samples have the same
    /// distribution as BoTorch's but not the same values for given base
    /// samples.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for base samples of the wrong
    /// width, and errors of the factorizations.
    pub fn matheron_samples(
        &self,
        x: &Matrix,
        z: &Matrix,
        observation_noise: bool,
    ) -> Result<Matrix> {
        let c = self.cache()?;
        let (xt, _) = self.transform_x(x)?;
        let q = xt.nrows();
        let (n, m) = (self.num_train(), self.num_tasks());
        let total = (n + q) * m + n * m + if observation_noise { q * m } else { 0 };
        if z.ncols() != total {
            return Err(BotorchError::DimensionMismatch(format!(
                "base samples must have {total} columns, got {}",
                z.ncols()
            )));
        }
        let prep = c.data.prepare(&xt)?;
        let joint_x = {
            let all = Matrix::from_fn(n + q, xt.ncols(), |i, j| {
                if i < n {
                    self.x_tf[(i, j)]
                } else {
                    xt[(i - n, j)]
                }
            });
            c.data.covariance_symmetric(&all)?
        };
        let lx = psd_safe_cholesky(&joint_x)?;
        let lb = psd_safe_cholesky(&c.b)?;
        let ln = psd_safe_cholesky(&c.noise)?;
        let kxs = c.data.cross(&prep, &c.train_prep); // q × n
        let s_count = z.nrows();
        let mut out = Matrix::zeros(s_count, q * m);
        let (mu, sd) = self.outcome_affine();
        let consts = self.constants();
        let resid = self.residuals(self.raw.as_slice());
        let jl = (n + q) * m;
        for sidx in 0..s_count {
            // Joint prior sample (L_x ⊗ L_B) z₁ (interleaved over n + q points).
            let z1 = Matrix::from_fn(n + q, m, |i, t| z[(sidx, i * m + t)]);
            let f = lx.l() * z1 * lb.l().transpose(); // (n+q) × m
                                                      // Training noise (I ⊗ L_N) z₂.
            let z2 = Matrix::from_fn(n, m, |i, t| z[(sidx, jl + i * m + t)]);
            let eps = z2 * ln.l().transpose();
            let obs = Vector::from_fn(n * m, |a, _| f[(a / m, a % m)] + eps[(a / m, a % m)]);
            let diff = &resid - obs;
            let sol = c.chol.solve_vec(&diff); // interleaved n·m
            for p in 0..q {
                for s in 0..m {
                    let mut upd = 0.0;
                    for i in 0..n {
                        let mut acc = 0.0;
                        for t in 0..m {
                            acc += c.b[(s, t)] * sol[i * m + t];
                        }
                        upd += kxs[(p, i)] * acc;
                    }
                    let mut val = f[(n + p, s)] + upd;
                    if observation_noise {
                        let off = jl + n * m;
                        let mut e = 0.0;
                        for t in 0..=s {
                            e += ln.l()[(s, t)] * z[(sidx, off + p * m + t)];
                        }
                        val += e;
                    }
                    out[(sidx, p * m + s)] = mu[s] + sd[s] * (val + consts[s]);
                }
            }
        }
        Ok(out)
    }
}

impl Fittable for KroneckerMultiTaskGP {
    type Output = FitResult;

    /// BoTorch's `fit_gpytorch_mll` (one L-BFGS-B problem over all
    /// hyperparameters, with retries from prior samples).
    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        let initial = self.raw.as_slice().to_vec();
        let bounds = self.raw_parameter_bounds();
        let (result, x) = {
            let model: &KroneckerMultiTaskGP = self;
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
        Ok(result)
    }
}

impl Model for KroneckerMultiTaskGP {
    fn dim(&self) -> usize {
        KroneckerMultiTaskGP::dim(self)
    }

    fn num_outputs(&self) -> usize {
        self.num_tasks()
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        KroneckerMultiTaskGP::posterior(self, x, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.joint_with_grad(x)
    }

    fn joint_posterior_vjp(
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
            self.num_tasks(),
            1,
        )?;
        self.vjp(x, rows, mean_adjoint, covariance_adjoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(lr: usize, standardize: bool) -> KroneckerMultiTaskGP {
        let n = 7;
        let x = Matrix::from_fn(n, 2, |i, j| ((i * 5 + j * 3) % 11) as f64 / 11.0);
        let y = Matrix::from_fn(n, 3, |i, t| {
            (3.0 * (x[(i, 0)] + x[(i, 1)])).sin() * (1.0 + 0.3 * t as f64) + 0.2 * t as f64
        });
        let mut m = KroneckerMultiTaskGP::new(
            x,
            y,
            KroneckerMultiTaskGPOptions {
                rank: Some(2),
                likelihood_rank: lr,
                outcome_transform: standardize.then(Standardize::new),
                input_transform: Some(Normalize::new(2)),
                ..Default::default()
            },
        )
        .unwrap();
        let bounds = m.raw_parameter_bounds();
        let raw: Vec<f64> = bounds
            .iter()
            .enumerate()
            .map(|(i, &(lo, _))| {
                let u = ((i * 37 + 11) % 17) as f64 / 17.0;
                if lo.is_finite() {
                    lo + 0.05 + 0.4 * u
                } else {
                    u - 0.4
                }
            })
            .collect();
        m.set_raw_parameters(&raw).unwrap();
        m
    }

    #[test]
    fn mll_gradient_matches_finite_differences() {
        for lr in [0, 2] {
            let m = model(lr, true);
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
                    "{lr} {j}: {fd} vs {}",
                    g[j]
                );
            }
        }
    }

    #[test]
    fn posterior_gradients_match_finite_differences() {
        let m = model(0, true);
        let x = Matrix::from_fn(2, 2, |i, j| 0.3 + 0.25 * i as f64 + 0.1 * j as f64);
        let g = m.joint_posterior_with_grad(&x).unwrap();
        let nn = g.mean.len();
        let ma = Vector::from_fn(nn, |i, _| 0.2 + 0.1 * i as f64);
        let ca = Matrix::from_fn(nn, nn, |i, j| 0.03 * (i + 2 * j) as f64 - 0.1);
        let vjp = m.joint_posterior_vjp(&x, 0..2, &ma, &ca).unwrap();
        for p in 0..2 {
            for k in 0..2 {
                let h = 1e-6;
                let mut xp = x.clone();
                xp[(p, k)] += h;
                let a = m.posterior(&xp, false).unwrap();
                xp[(p, k)] -= 2.0 * h;
                let b = m.posterior(&xp, false).unwrap();
                let dm = (&a.mean - &b.mean) / (2.0 * h);
                let dc = (&a.covariance - &b.covariance) / (2.0 * h);
                let j = p * 2 + k;
                assert!((&dm - &g.dmean[j]).amax() < 1e-6);
                assert!((&dc - &g.dcov[j]).amax() < 1e-6);
                let s = ma.dot(&dm) + ca.component_mul(&dc).sum();
                assert!((s - vjp[(p, k)]).abs() < 1e-6, "{s} vs {}", vjp[(p, k)]);
            }
        }
    }

    #[test]
    fn matheron_samples_have_the_posterior_distribution() {
        // The linear map from base samples to samples reproduces the joint
        // posterior covariance (with observation noise).
        for (lr, noise) in [(0, false), (2, true)] {
            let m = model(lr, true);
            let x = Matrix::from_fn(2, 2, |i, j| 0.2 + 0.3 * i as f64 + 0.15 * j as f64);
            let (n, t) = (m.num_train(), m.num_tasks());
            let total = (n + 2) * t + n * t + if noise { 2 * t } else { 0 };
            let mut z = Matrix::zeros(total + 1, total);
            for i in 0..total {
                z[(i + 1, i)] = 1.0;
            }
            let s = m.matheron_samples(&x, &z, noise).unwrap();
            let mean = s.row(0).transpose();
            let a = Matrix::from_fn(total, 2 * t, |i, j| s[(i + 1, j)] - mean[j]);
            let cov = a.transpose() * &a;
            let post = m.posterior(&x, noise).unwrap();
            for i in 0..2 * t {
                // Interleaved (p·m + s) vs output-major (s·q + p).
                let pi = (i % t) * 2 + i / t;
                assert!((mean[i] - post.mean[pi]).abs() < 1e-9);
                for j in 0..2 * t {
                    let pj = (j % t) * 2 + j / t;
                    assert!(
                        (cov[(i, j)] - post.covariance[(pi, pj)]).abs() < 1e-7,
                        "{i} {j}: {} vs {}",
                        cov[(i, j)],
                        post.covariance[(pi, pj)]
                    );
                }
            }
        }
    }
}
