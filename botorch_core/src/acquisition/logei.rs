//! Monte-Carlo LogEI acquisition functions mirroring
//! `botorch.acquisition.logei` (Ament et al., *Unexpected Improvements to
//! Expected Improvement for Bayesian Optimization*, NeurIPS 2023).
//!
//! For posterior samples `y_{js}` of the candidate points, both functions
//! compute
//!
//! ```text
//! α(X) = logmeanexp_s( fatmax_j( log_fatplus(y_js - best_f_s, τ_relu), τ_max ) )
//! ```
//!
//! (with `log_softplus` and `smooth_amax` instead when `fat = false`), i.e. a
//! smooth, numerically robust logarithm of qEI. qLogNEI replaces the fixed
//! `best_f` by the per-sample maximum of the joint posterior samples at the
//! (pruned) baseline points.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::acquisition::monte_carlo::{
    read_lock, sampler_or_default, vstack, write_lock, JointSamples, McAcquisitionOptions, McCore,
    Reduction,
};
use crate::acquisition::utils::{prune_inferior_points, PruneOptions};
use crate::acquisition::{check_candidates, check_pending, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::linalg::CovarianceRoot;
use crate::models::Model;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::safe_math::{
    fatmax, fatmax_grad_into, log_fatplus, log_fatplus_with_grad, log_softplus,
    log_softplus_with_grad, logmeanexp, logmeanexp_with_grad, smooth_amax, smooth_amax_with_grad,
    ALPHA,
};
use crate::Matrix;

/// Default temperature of the smooth ReLU (`logei.TAU_RELU`).
pub const TAU_RELU: f64 = 1e-6;
/// Default temperature of the smooth maximum over the `q` points
/// (`logei.TAU_MAX`).
pub const TAU_MAX: f64 = 1e-2;

fn check_tau(tau: f64, name: &str) -> Result<f64> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(BotorchError::InvalidArgument(format!(
            "{name} must be positive and finite, got {tau}"
        )));
    }
    Ok(tau)
}

/// The log-improvement reduction of [`QLogExpectedImprovement`] and
/// [`QLogNoisyExpectedImprovement`] for samples `y` (`points × S`) and
/// per-sample incumbents `best_f(s)`.
fn log_improvement_reduce<B>(
    y: &Matrix,
    best_f: B,
    fat: bool,
    tau_relu: f64,
    tau_max: f64,
    with_grad: bool,
) -> Reduction
where
    B: Fn(usize) -> f64,
{
    let (n, s) = y.shape();
    let mut r = vec![0.0; s];
    let mut l = vec![0.0; n];
    if !with_grad {
        for (c, rc) in r.iter_mut().enumerate() {
            let b = best_f(c);
            for (i, li) in l.iter_mut().enumerate() {
                let z = y[(i, c)] - b;
                *li = if fat {
                    log_fatplus(z, tau_relu)
                } else {
                    log_softplus(z, tau_relu)
                };
            }
            *rc = if fat {
                fatmax(&l, tau_max, ALPHA)
            } else {
                smooth_amax(&l, tau_max)
            };
        }
        return (logmeanexp(&r), None);
    }
    let mut g = Matrix::zeros(n, s);
    let mut dl = vec![0.0; n];
    let mut rg = vec![0.0; n];
    for (c, rc) in r.iter_mut().enumerate() {
        let b = best_f(c);
        for i in 0..n {
            let z = y[(i, c)] - b;
            let (v, d) = if fat {
                log_fatplus_with_grad(z, tau_relu)
            } else {
                log_softplus_with_grad(z, tau_relu)
            };
            l[i] = v;
            dl[i] = d;
        }
        *rc = if fat {
            fatmax_grad_into(&l, tau_max, ALPHA, &mut rg)
        } else {
            let (v, gr) = smooth_amax_with_grad(&l, tau_max);
            rg.copy_from_slice(&gr);
            v
        };
        for i in 0..n {
            g[(i, c)] = rg[i] * dl[i];
        }
    }
    let (value, w) = logmeanexp_with_grad(&r);
    for c in 0..s {
        for i in 0..n {
            g[(i, c)] *= w[c];
        }
    }
    (value, Some(g))
}

/// Options of [`QLogExpectedImprovement`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QLogExpectedImprovementOptions {
    /// The sampler of base samples (default: 512 Sobol samples, random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Pending points concatenated to every candidate set.
    pub x_pending: Option<Matrix>,
    /// Use the fat-tailed `fatplus` / `fatmax` approximations (default `true`)
    /// instead of `softplus` / `logsumexp`.
    pub fat: bool,
    /// Temperature of the smooth maximum over the `q` points (default `1e-2`).
    pub tau_max: f64,
    /// Temperature of the smooth ReLU (default `1e-6`).
    pub tau_relu: f64,
}

impl Default for QLogExpectedImprovementOptions {
    fn default() -> Self {
        Self {
            sampler: None,
            x_pending: None,
            fat: true,
            tau_max: TAU_MAX,
            tau_relu: TAU_RELU,
        }
    }
}

/// MC-based batch Log Expected Improvement, a smooth approximation of
/// `log qEI(X) = log E[max(max_j y_j - best_f, 0)]` (BoTorch's
/// `qLogExpectedImprovement`).
#[derive(Debug)]
pub struct QLogExpectedImprovement {
    core: McCore,
    best_f: f64,
    fat: bool,
    tau_max: f64,
    tau_relu: f64,
}

impl QLogExpectedImprovement {
    /// Creates `qLogExpectedImprovement(model, best_f, ...)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models,
    /// [`BotorchError::InvalidArgument`] for a NaN `best_f` or non-positive
    /// temperatures and [`BotorchError::DimensionMismatch`] for pending points
    /// of the wrong dimension.
    pub fn new(
        model: Arc<dyn Model>,
        best_f: f64,
        options: QLogExpectedImprovementOptions,
    ) -> Result<Self> {
        if best_f.is_nan() {
            return Err(BotorchError::InvalidArgument(
                "best_f must not be NaN".into(),
            ));
        }
        let tau_max = check_tau(options.tau_max, "tau_max")?;
        let tau_relu = check_tau(options.tau_relu, "tau_relu")?;
        let core = McCore::new(
            model,
            McAcquisitionOptions {
                sampler: options.sampler,
                x_pending: options.x_pending,
            },
        )?;
        Ok(Self {
            core,
            best_f,
            fat: options.fat,
            tau_max,
            tau_relu,
        })
    }

    /// The incumbent value `best_f`.
    pub fn best_f(&self) -> f64 {
        self.best_f
    }

    /// The sampler.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.core.sampler
    }

    fn reduce(&self, y: &Matrix, with_grad: bool) -> Reduction {
        log_improvement_reduce(
            y,
            |_| self.best_f,
            self.fat,
            self.tau_relu,
            self.tau_max,
            with_grad,
        )
    }
}

impl AcquisitionFunction for QLogExpectedImprovement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.core.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.core.evaluate(x, false, |y, g| self.reduce(y, g))?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.core.evaluate(x, true, |y, g| self.reduce(y, g))?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.core.x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.core.set_x_pending(x_pending)
    }

    fn is_log(&self) -> bool {
        true
    }
}

/// Options of [`QLogNoisyExpectedImprovement`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QLogNoisyExpectedImprovementOptions {
    /// The sampler of base samples (default: 512 Sobol samples, random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Pending points.
    pub x_pending: Option<Matrix>,
    /// Use the fat-tailed approximations (default `true`).
    pub fat: bool,
    /// Remove baseline points that are very unlikely to be the best
    /// (`prune_inferior_points`, default `true`).
    pub prune_baseline: bool,
    /// Settings of the pruning (2048 samples by default).
    pub prune_options: PruneOptions,
    /// Temperature of the smooth maximum over the `q` points (default `1e-2`).
    pub tau_max: f64,
    /// Temperature of the smooth ReLU (default `1e-6`).
    pub tau_relu: f64,
    /// Treat pending points as part of the baseline and compute the
    /// incremental improvement of the candidates (BoTorch's default `true`);
    /// with `false`, pending points are appended to every candidate set.
    pub incremental: bool,
}

impl Default for QLogNoisyExpectedImprovementOptions {
    fn default() -> Self {
        Self {
            sampler: None,
            x_pending: None,
            fat: true,
            prune_baseline: true,
            prune_options: PruneOptions::default(),
            tau_max: TAU_MAX,
            tau_relu: TAU_RELU,
            incremental: true,
        }
    }
}

/// Per-sample incumbents keyed by (number of jointly sampled points, jitter bits).
type IncumbentCache = Mutex<HashMap<(usize, u64), Arc<Vec<f64>>>>;

/// Baseline-dependent state of [`QLogNoisyExpectedImprovement`].
#[derive(Clone, Debug)]
struct NeiState {
    /// The (pruned) baseline points (BoTorch's `_X_baseline`).
    x_baseline: Matrix,
    /// Pending points.
    x_pending: Option<Matrix>,
    /// The incumbent points: baseline and, if incremental, pending points
    /// (BoTorch's `X_baseline` property).
    full_baseline: Matrix,
}

/// MC-based batch Log Noisy Expected Improvement (BoTorch's
/// `qLogNoisyExpectedImprovement`): a smooth approximation of
/// `log E[max(max_j y_j - max_i y_i^baseline, 0)]` over joint posterior
/// samples of the candidates and the previously evaluated (baseline) points.
///
/// This is the plain computation of BoTorch's `cache_root=False`: the joint
/// posterior over `[X_baseline; X_pending; X]` (incremental) or
/// `[X_baseline; X; X_pending]` is sampled with base samples for all points
/// and the incumbent of each sample is the maximum over the baseline columns
/// (without gradient, as in BoTorch). Since the baseline block of the joint
/// Cholesky factor does not depend on `X`, the incumbents are cached per
/// jitter level. BoTorch's default `cache_root=True` computes the same
/// quantity with a low-rank update of a cached baseline factor; it differs
/// only in which base samples are used for the baseline points (an
/// independent Sobol draw for the baseline alone) and in where jitter is
/// added when the joint covariance is singular.
#[derive(Debug)]
pub struct QLogNoisyExpectedImprovement {
    model: Arc<dyn Model>,
    sampler: Arc<SobolQMCNormalSampler>,
    fat: bool,
    tau_max: f64,
    tau_relu: f64,
    prune_baseline: bool,
    prune_options: PruneOptions,
    incremental: bool,
    state: RwLock<NeiState>,
    /// Incumbent samples per (number of joint points, jitter bits).
    best_f_cache: IncumbentCache,
}

impl QLogNoisyExpectedImprovement {
    /// Creates `qLogNoisyExpectedImprovement(model, X_baseline, ...)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models,
    /// [`BotorchError::InvalidArgument`] for an empty baseline or invalid
    /// temperatures, [`BotorchError::DimensionMismatch`] for inputs of the
    /// wrong dimension, and model errors from the pruning.
    pub fn new(
        model: Arc<dyn Model>,
        x_baseline: &Matrix,
        options: QLogNoisyExpectedImprovementOptions,
    ) -> Result<Self> {
        if model.num_outputs() != 1 {
            return Err(BotorchError::Unsupported(
                "Must specify an objective or a posterior transform when using a multi-output model."
                    .into(),
            ));
        }
        let d = model.dim();
        if x_baseline.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "X_baseline must have {d} columns, got {}",
                x_baseline.ncols()
            )));
        }
        if x_baseline.nrows() == 0 {
            return Err(BotorchError::InvalidArgument(
                "X_baseline must have at least one point.".into(),
            ));
        }
        if let Some(p) = &options.x_pending {
            check_pending(p, d)?;
        }
        let tau_max = check_tau(options.tau_max, "tau_max")?;
        let tau_relu = check_tau(options.tau_relu, "tau_relu")?;
        let sampler = sampler_or_default(options.sampler)?;
        let acqf = Self {
            model,
            sampler,
            fat: options.fat,
            tau_max,
            tau_relu,
            prune_baseline: options.prune_baseline,
            prune_options: options.prune_options,
            incremental: options.incremental,
            state: RwLock::new(NeiState {
                x_baseline: x_baseline.clone(),
                x_pending: None,
                full_baseline: x_baseline.clone(),
            }),
            best_f_cache: Mutex::new(HashMap::new()),
        };
        let state = acqf.init_baseline(x_baseline, options.x_pending)?;
        *write_lock(&acqf.state) = state;
        Ok(acqf)
    }

    /// BoTorch's `_init_baseline`: prunes the baseline (if enabled) and
    /// assembles the incumbent set.
    fn init_baseline(&self, x_baseline: &Matrix, x_pending: Option<Matrix>) -> Result<NeiState> {
        let pruned = if self.prune_baseline {
            prune_inferior_points(self.model.as_ref(), x_baseline, &self.prune_options)?
        } else {
            x_baseline.clone()
        };
        let full_baseline = match (&x_pending, self.incremental) {
            (Some(p), true) => vstack(&[&pruned, p], self.model.dim()),
            _ => pruned.clone(),
        };
        Ok(NeiState {
            x_baseline: pruned,
            x_pending,
            full_baseline,
        })
    }

    /// The (pruned) baseline points.
    pub fn x_baseline(&self) -> Matrix {
        read_lock(&self.state).x_baseline.clone()
    }

    /// The incumbent points: the baseline and, in incremental mode, the
    /// pending points (BoTorch's `X_baseline` property).
    pub fn full_x_baseline(&self) -> Matrix {
        read_lock(&self.state).full_baseline.clone()
    }

    /// The sampler.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.sampler
    }

    /// Per-sample incumbents `max_i y_i^baseline` (no gradient).
    fn incumbents(&self, js: &JointSamples, n_baseline: usize) -> Arc<Vec<f64>> {
        let key = match &js.root {
            CovarianceRoot::Cholesky(f) => Some((js.x.nrows(), f.jitter().to_bits())),
            _ => None,
        };
        let mut cache = self.best_f_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(k) = key {
            if let Some(v) = cache.get(&k) {
                return Arc::clone(v);
            }
        }
        let yb = js.samples(0..n_baseline);
        let best: Vec<f64> = (0..yb.ncols())
            .map(|c| {
                yb.column(c)
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, |m, v| {
                        if v.is_nan() || m.is_nan() {
                            f64::NAN
                        } else {
                            m.max(v)
                        }
                    })
            })
            .collect();
        let best = Arc::new(best);
        if let Some(k) = key {
            cache.insert(k, Arc::clone(&best));
        }
        best
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let state = read_lock(&self.state);
        let q = x.nrows();
        let n_bl = state.full_baseline.nrows();
        let x_full = match (&state.x_pending, self.incremental) {
            (Some(p), false) => vstack(&[&state.full_baseline, x, p], d),
            _ => vstack(&[&state.full_baseline, x], d),
        };
        let n = x_full.nrows();
        let js = JointSamples::new(self.model.as_ref(), x_full, &self.sampler)?;
        let best_f = self.incumbents(&js, n_bl);
        let y = js.samples(n_bl..n);
        let (value, ybar) = log_improvement_reduce(
            &y,
            |s| best_f[s],
            self.fat,
            self.tau_relu,
            self.tau_max,
            with_grad,
        );
        match ybar {
            Some(ybar) if with_grad => {
                let grad = js.backward(self.model.as_ref(), n_bl..n, &ybar, n_bl..n_bl + q)?;
                Ok((value, Some(grad)))
            }
            _ => Ok((value, None)),
        }
    }
}

impl AcquisitionFunction for QLogNoisyExpectedImprovement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        read_lock(&self.state).x_pending.clone()
    }

    /// Sets the pending points. In incremental mode they become part of the
    /// incumbent set and, as in BoTorch, the stored (already pruned) baseline
    /// is pruned again.
    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.model.dim())?;
        }
        let mut state = write_lock(&self.state);
        let new_state = if self.incremental {
            let base = state.x_baseline.clone();
            self.init_baseline(&base, x_pending.cloned())?
        } else {
            NeiState {
                x_pending: x_pending.cloned(),
                ..state.clone()
            }
        };
        *state = new_state;
        self.best_f_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        Ok(())
    }

    fn is_log(&self) -> bool {
        true
    }
}
