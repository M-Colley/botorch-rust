//! Monte-Carlo LogEI acquisition functions mirroring
//! `botorch.acquisition.logei` (Ament et al., *Unexpected Improvements to
//! Expected Improvement for Bayesian Optimization*, NeurIPS 2023).
//!
//! For objective values `y_{js}` of the candidate points (posterior samples
//! for single-output models), qLogEI and qLogNEI compute
//!
//! ```text
//! α(X) = logmeanexp_s( fatmax_j( log_fatplus(y_js - best_f_s, τ_relu) + log ind_js, τ_max ) )
//! ```
//!
//! (with `log_softplus` and `smooth_amax` instead when `fat = false`), i.e. a
//! smooth, numerically robust logarithm of qEI, where `ind` is the smoothed
//! feasibility indicator of outcome constraints (if any). qLogNEI replaces
//! the fixed `best_f` by the per-sample best feasible objective value at the
//! (pruned) baseline points. [`QLogProbabilityOfFeasibility`] reduces the log
//! feasibility indicator alone.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};

use crate::acquisition::mc_objective::{McObjective, McSamples};
use crate::acquisition::monte_carlo::{
    check_constraints_one_to_one, check_mc_model, read_lock, sampler_or_default, vstack,
    write_lock, JointSamples, McAcquisitionOptions, McCore, Reduced, SampledObjective,
};
use crate::acquisition::utils::{
    compute_best_feasible_objective, estimate_objective_lower_bound, prune_inferior_points,
    PruneOptions,
};
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::objective::{
    check_fat, compute_smoothed_feasibility_indicator, OutcomeConstraints,
};
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

/// Number of uniformly random points at which BoTorch estimates a lower bound
/// of the objective when no baseline point is feasible
/// (`_estimate_objective_lower_bound`).
const LOWER_BOUND_POINTS: usize = 32;
/// Salt of the seed of those points (derived from the sampler's seed).
const LOWER_BOUND_SALT: u64 = 0x6c6f_7765_725f_6264;

fn check_tau(tau: f64, name: &str) -> Result<f64> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(BotorchError::InvalidArgument(format!(
            "{name} must be positive and finite, got {tau}"
        )));
    }
    Ok(tau)
}

/// BoTorch's error for a [`crate::acquisition::ConstrainedMcObjective`] in a
/// LogEI acquisition function.
fn check_not_constrained_objective(objective: Option<&dyn McObjective>) -> Result<()> {
    if objective.is_some_and(|o| o.is_constrained()) {
        return Err(BotorchError::InvalidArgument(
            "Log-Improvement should not be used with `ConstrainedMCObjective`.Please pass the \
             `constraints` directly to the constructor of the acquisition function."
                .into(),
        ));
    }
    Ok(())
}

/// The smoothing kinds of the constraints of a LogEI acquisition function:
/// `constraint_fat` if given (BoTorch's `fat` as a list), else `fat` for
/// every constraint.
fn constraint_fat(
    constraints: Option<&OutcomeConstraints>,
    fat: bool,
    constraint_fat: Option<Vec<Option<bool>>>,
) -> Result<Vec<Option<bool>>> {
    let k = constraints.map_or(0, |c| c.len());
    match (constraints, constraint_fat) {
        (Some(c), Some(f)) => {
            check_fat(c, &f)?;
            Ok(f)
        }
        (None, Some(f)) if !f.is_empty() => Err(BotorchError::InvalidArgument(
            "Number of provided constraints and number of provided fats do not match.".into(),
        )),
        _ => Ok(vec![Some(fat); k]),
    }
}

/// The smooth log improvement `log_fatplus(y - b, τ)` (or `log_softplus`) of
/// every point and sample of `obj` (`n × S`) and its derivative.
fn log_improvement<B>(
    obj: &Matrix,
    best_f: B,
    fat: bool,
    tau: f64,
    with_grad: bool,
) -> (Matrix, Matrix)
where
    B: Fn(usize) -> f64,
{
    let (n, s) = obj.shape();
    let mut l = Matrix::zeros(n, s);
    let mut dl = Matrix::zeros(if with_grad { n } else { 0 }, if with_grad { s } else { 0 });
    for c in 0..s {
        let b = best_f(c);
        for i in 0..n {
            let z = obj[(i, c)] - b;
            if with_grad {
                let (v, d) = if fat {
                    log_fatplus_with_grad(z, tau)
                } else {
                    log_softplus_with_grad(z, tau)
                };
                l[(i, c)] = v;
                dl[(i, c)] = d;
            } else {
                l[(i, c)] = if fat {
                    log_fatplus(z, tau)
                } else {
                    log_softplus(z, tau)
                };
            }
        }
    }
    (l, dl)
}

/// `logmeanexp_s( fatmax_j(l_js, τ_max) )` (or `smooth_amax`) and the
/// adjoint of `l` (`n × S`).
fn log_reduce(l: &Matrix, fat: bool, tau_max: f64, with_grad: bool) -> (f64, Option<Matrix>) {
    let (n, s) = l.shape();
    let mut r = vec![0.0; s];
    let mut col = vec![0.0; n];
    if !with_grad {
        for (c, rc) in r.iter_mut().enumerate() {
            for (i, v) in col.iter_mut().enumerate() {
                *v = l[(i, c)];
            }
            *rc = if fat {
                fatmax(&col, tau_max, ALPHA)
            } else {
                smooth_amax(&col, tau_max)
            };
        }
        return (logmeanexp(&r), None);
    }
    let mut g = Matrix::zeros(n, s);
    let mut rg = vec![0.0; n];
    for (c, rc) in r.iter_mut().enumerate() {
        for (i, v) in col.iter_mut().enumerate() {
            *v = l[(i, c)];
        }
        *rc = if fat {
            fatmax_grad_into(&col, tau_max, ALPHA, &mut rg)
        } else {
            let (v, gr) = smooth_amax_with_grad(&col, tau_max);
            rg.copy_from_slice(&gr);
            v
        };
        for i in 0..n {
            g[(i, c)] = rg[i];
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

/// The log-improvement reduction of [`QLogExpectedImprovement`] and
/// [`QLogNoisyExpectedImprovement`] for objective values `obj` (`n × S`),
/// per-sample incumbents `best_f(s)` and an optional log feasibility
/// indicator.
fn log_improvement_reduce<B>(
    obj: &Matrix,
    best_f: B,
    log_ind: Option<&Matrix>,
    fat: bool,
    tau_relu: f64,
    tau_max: f64,
    with_grad: bool,
) -> Reduced
where
    B: Fn(usize) -> f64,
{
    let (mut l, dl) = log_improvement(obj, best_f, fat, tau_relu, with_grad);
    if let Some(li) = log_ind {
        l += li;
    }
    let (value, l_bar) = log_reduce(&l, fat, tau_max, with_grad);
    match l_bar {
        None => Reduced {
            value,
            obj_bar: None,
            ind_bar: None,
        },
        Some(lb) => Reduced {
            value,
            obj_bar: Some(lb.component_mul(&dl)),
            ind_bar: log_ind.map(|_| lb),
        },
    }
}

/// Options of [`QLogExpectedImprovement`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QLogExpectedImprovementOptions {
    /// The sampler of base samples (default: 512 Sobol samples, random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Pending points concatenated to every candidate set.
    pub x_pending: Option<Matrix>,
    /// Use the fat-tailed `fatplus` / `fatmax` approximations (default `true`)
    /// instead of `softplus` / `logsumexp`; also the default smoothing of the
    /// feasibility indicator (`fatmoid` / logistic sigmoid).
    pub fat: bool,
    /// Temperature of the smooth maximum over the `q` points (default `1e-2`).
    pub tau_max: f64,
    /// Temperature of the smooth ReLU (default `1e-6`).
    pub tau_relu: f64,
    /// The MC objective (required for multi-output models without a
    /// posterior transform).
    pub objective: Option<Arc<dyn McObjective>>,
    /// Outcome constraints (BoTorch's `constraints` and `eta`).
    pub constraints: Option<OutcomeConstraints>,
    /// Per-constraint smoothing (BoTorch's `fat` given as a list: `Some(true)`
    /// fatmoid, `Some(false)` sigmoid, `None` for constraints that return
    /// probabilities); `None` uses `fat` for every constraint.
    pub constraint_fat: Option<Vec<Option<bool>>>,
}

impl Default for QLogExpectedImprovementOptions {
    fn default() -> Self {
        Self {
            sampler: None,
            x_pending: None,
            fat: true,
            tau_max: TAU_MAX,
            tau_relu: TAU_RELU,
            objective: None,
            constraints: None,
            constraint_fat: None,
        }
    }
}

/// MC-based batch Log Expected Improvement, a smooth approximation of
/// `log qEI(X) = log E[max_j (max(obj_j - best_f, 0) · ind_j)]` (BoTorch's
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
    /// [`BotorchError::Unsupported`] for multi-output models without an
    /// objective, [`BotorchError::InvalidArgument`] for a NaN `best_f`,
    /// non-positive temperatures, a [`crate::acquisition::ConstrainedMcObjective`]
    /// or invalid constraint settings, and [`BotorchError::DimensionMismatch`]
    /// for pending points of the wrong dimension.
    pub fn new(
        model: Arc<dyn Model>,
        best_f: f64,
        options: QLogExpectedImprovementOptions,
    ) -> Result<Self> {
        check_not_constrained_objective(options.objective.as_deref())?;
        if best_f.is_nan() {
            return Err(BotorchError::InvalidArgument(
                "best_f must not be NaN".into(),
            ));
        }
        let tau_max = check_tau(options.tau_max, "tau_max")?;
        let tau_relu = check_tau(options.tau_relu, "tau_relu")?;
        let fat_list = constraint_fat(
            options.constraints.as_ref(),
            options.fat,
            options.constraint_fat,
        )?;
        let core = McCore::with_fat(
            model,
            McAcquisitionOptions {
                sampler: options.sampler,
                x_pending: options.x_pending,
                objective: options.objective,
                constraints: options.constraints,
            },
            fat_list,
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

    fn reduce(&self, y: &Matrix, log_ind: Option<&Matrix>, with_grad: bool) -> Reduced {
        log_improvement_reduce(
            y,
            |_| self.best_f,
            log_ind,
            self.fat,
            self.tau_relu,
            self.tau_max,
            with_grad,
        )
    }
}

impl crate::acquisition::SampleReducing for QLogExpectedImprovement {
    fn evaluate_weighted(
        &self,
        x: &Matrix,
        weights: &[f64],
        with_grad: bool,
    ) -> Result<crate::acquisition::WeightedEvaluation> {
        self.core
            .evaluate_weighted(x, Some(weights), with_grad, true, |y, w, g| {
                self.reduce(y, w, g)
            })
    }

    fn log_utilities(&self) -> bool {
        true
    }
}

impl AcquisitionFunction for QLogExpectedImprovement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.core.model
    }

    fn as_sample_reducing(&self) -> Option<&dyn crate::acquisition::SampleReducing> {
        Some(self)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self
            .core
            .evaluate(x, false, true, |y, w, g| self.reduce(y, w, g))?
            .0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self
            .core
            .evaluate(x, true, true, |y, w, g| self.reduce(y, w, g))?;
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
    /// Use the fat-tailed approximations (default `true`); also the default
    /// smoothing of the feasibility indicator.
    pub fat: bool,
    /// Remove baseline points that are very unlikely to be the best
    /// (`prune_inferior_points`, default `true`). The objective and the
    /// constraints are passed on to the pruning.
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
    /// The MC objective (required for multi-output models without a
    /// posterior transform).
    pub objective: Option<Arc<dyn McObjective>>,
    /// Outcome constraints (BoTorch's `constraints` and `eta`); the incumbent
    /// of every sample is the best *feasible* baseline value.
    pub constraints: Option<OutcomeConstraints>,
    /// Per-constraint smoothing (BoTorch's `fat` given as a list); `None`
    /// uses `fat` for every constraint.
    pub constraint_fat: Option<Vec<Option<bool>>>,
    /// The points (`k × d`) at which the lower bound of the objective is
    /// estimated when a sample has no feasible baseline point. `None` (the
    /// default) draws 32 uniformly random points in the expanded bounding box
    /// of the incumbent points with a generator seeded by the sampler's seed
    /// (BoTorch draws them from torch's global generator).
    pub lower_bound_points: Option<Matrix>,
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
            objective: None,
            constraints: None,
            constraint_fat: None,
            lower_bound_points: None,
        }
    }
}

/// Per-sample incumbents keyed by (number of jointly sampled points, jitter
/// bits of the covariance roots).
type IncumbentCache = Mutex<HashMap<(usize, Vec<u64>), Arc<Vec<f64>>>>;

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
/// `log E[max_j (max(obj_j - best_i obj_i^baseline, 0) · ind_j)]` over joint
/// posterior samples of the candidates and the previously evaluated
/// (baseline) points.
///
/// This is the plain computation of BoTorch's `cache_root=False`: the joint
/// posterior over `[X_baseline; X_pending; X]` (incremental) or
/// `[X_baseline; X; X_pending]` is sampled with base samples for all points,
/// the objective is evaluated on all of them, and the incumbent of each
/// sample is the best feasible objective value at the baseline points
/// (without gradient, as in BoTorch). If a sample has no feasible baseline
/// point, BoTorch replaces the values of infeasible points by a lower bound
/// of the objective estimated at 32 uniformly random points around the
/// baseline; here these points come from a generator seeded by the sampler's
/// seed and the bound is computed once. For pointwise objectives and
/// constraints, the baseline block of the joint Cholesky factor (and hence
/// the incumbents) does not depend on `X`, so the incumbents are cached per
/// jitter level. BoTorch's default `cache_root=True` computes the same
/// quantity with a low-rank update of a cached baseline factor; it differs
/// only in which base samples are used for the baseline points (an
/// independent Sobol draw for the baseline alone) and in where jitter is
/// added when the joint covariance is singular.
#[derive(Debug)]
pub struct QLogNoisyExpectedImprovement {
    model: Arc<dyn Model>,
    sampler: Arc<SobolQMCNormalSampler>,
    objective: Option<Arc<dyn McObjective>>,
    constraints: Option<OutcomeConstraints>,
    constraint_fat: Vec<Option<bool>>,
    fat: bool,
    tau_max: f64,
    tau_relu: f64,
    prune_baseline: bool,
    prune_options: PruneOptions,
    incremental: bool,
    state: RwLock<NeiState>,
    /// Incumbent samples per (number of joint points, jitter bits).
    best_f_cache: IncumbentCache,
    /// The lower bound used for infeasible baseline points (once computed).
    lower_bound: Mutex<Option<f64>>,
    /// Explicit points of the lower-bound estimate.
    lower_bound_points: Option<Matrix>,
}

impl QLogNoisyExpectedImprovement {
    /// Creates `qLogNoisyExpectedImprovement(model, X_baseline, ...)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models without an
    /// objective, [`BotorchError::InvalidArgument`] for an empty baseline,
    /// invalid temperatures, a [`crate::acquisition::ConstrainedMcObjective`]
    /// or invalid constraint settings, [`BotorchError::DimensionMismatch`] for
    /// inputs of the wrong dimension, and model errors from the pruning.
    pub fn new(
        model: Arc<dyn Model>,
        x_baseline: &Matrix,
        options: QLogNoisyExpectedImprovementOptions,
    ) -> Result<Self> {
        check_not_constrained_objective(options.objective.as_deref())?;
        check_mc_model(model.as_ref(), options.objective.as_deref())?;
        if options.constraints.is_some() {
            check_constraints_one_to_one(model.as_ref())?;
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
        if let Some(p) = &options.lower_bound_points {
            if p.ncols() != d || p.nrows() == 0 {
                return Err(BotorchError::DimensionMismatch(format!(
                    "lower_bound_points must be k x {d} with k > 0, got {}x{}",
                    p.nrows(),
                    p.ncols()
                )));
            }
        }
        let tau_max = check_tau(options.tau_max, "tau_max")?;
        let tau_relu = check_tau(options.tau_relu, "tau_relu")?;
        let cfat = constraint_fat(
            options.constraints.as_ref(),
            options.fat,
            options.constraint_fat,
        )?;
        let sampler = sampler_or_default(options.sampler)?;
        let mut prune_options = options.prune_options;
        prune_options.objective = options.objective.clone();
        prune_options.constraints = options
            .constraints
            .as_ref()
            .map(|c| c.constraints().to_vec());
        let acqf = Self {
            model,
            sampler,
            objective: options.objective,
            constraints: options.constraints,
            constraint_fat: cfat,
            fat: options.fat,
            tau_max,
            tau_relu,
            prune_baseline: options.prune_baseline,
            prune_options,
            incremental: options.incremental,
            state: RwLock::new(NeiState {
                x_baseline: x_baseline.clone(),
                x_pending: None,
                full_baseline: x_baseline.clone(),
            }),
            best_f_cache: Mutex::new(HashMap::new()),
            lower_bound: Mutex::new(None),
            lower_bound_points: options.lower_bound_points,
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

    /// The points (`32 × d` by default) at which the lower bound of the
    /// objective is estimated when a sample has no feasible baseline point:
    /// the explicit `lower_bound_points`, or uniformly random points in the
    /// bounding box of the incumbent points expanded by 10% on each side
    /// (BoTorch's `_estimate_objective_lower_bound`), with uniform numbers
    /// from a generator seeded by the sampler's seed.
    pub fn lower_bound_points(&self) -> Matrix {
        if let Some(p) = &self.lower_bound_points {
            return p.clone();
        }
        let xb = self.full_x_baseline();
        let d = xb.ncols();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.sampler.seed() ^ LOWER_BOUND_SALT);
        let u = Matrix::from_fn(LOWER_BOUND_POINTS, d, |_, _| rng.random::<f64>());
        Matrix::from_fn(LOWER_BOUND_POINTS, d, |i, k| {
            let col = xb.column(k);
            let lo = col.min();
            let hi = col.max();
            let range = hi - lo;
            let padding = 0.1 * range;
            lo - padding + u[(i, k)] * (range + 2.0 * padding)
        })
    }

    /// The lower bound of the objective used for infeasible baseline points,
    /// if an evaluation needed it (some sample had no feasible baseline
    /// point; BoTorch then warns that qLogProbabilityOfFeasibility is the
    /// better choice).
    pub fn infeasible_lower_bound(&self) -> Option<f64> {
        *self.lower_bound.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The lower bound of the objective for infeasible baseline points
    /// (computed once).
    fn infeasible_value(&self) -> Result<f64> {
        let mut lb = self.lower_bound.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = *lb {
            return Ok(v);
        }
        let v = estimate_objective_lower_bound(
            self.model.as_ref(),
            self.objective.as_deref(),
            &self.lower_bound_points(),
        )?;
        *lb = Some(v);
        Ok(v)
    }

    /// Per-sample incumbents: the best feasible objective value at the
    /// baseline points (no gradient).
    fn incumbents(&self, so: &SampledObjective, n_baseline: usize) -> Result<Arc<Vec<f64>>> {
        let pointwise = self.objective.as_ref().is_none_or(|o| o.is_pointwise())
            && self.constraints.as_ref().is_none_or(|c| c.is_pointwise());
        let key = if pointwise {
            so.js
                .roots
                .jitter_bits()
                .map(|bits| (so.js.num_points(), bits))
        } else {
            None
        };
        if let Some(k) = &key {
            let cache = self.best_f_cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(v) = cache.get(k) {
                return Ok(Arc::clone(v));
            }
        }
        let k = so.js.points_per_input();
        let bl_samples = so.samples.points(0..n_baseline * k);
        let bl_obj = so.obj.rows(0, n_baseline).into_owned();
        let best = compute_best_feasible_objective(
            &bl_samples,
            &bl_obj,
            self.constraints.as_ref().map(|c| c.constraints()),
            || self.infeasible_value(),
        )?;
        let best = Arc::new(best);
        if let Some(k) = key {
            self.best_f_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(k, Arc::clone(&best));
        }
        Ok(best)
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let (v, g) = self.evaluate_weighted(x, None, with_grad)?;
        Ok((v, g.map(|(g, _)| g)))
    }

    /// The value with the log utilities of the candidate points (and
    /// non-incremental pending points) shifted by `weights` (BoTorch's
    /// prior-guided weighting) and the gradients with respect to `x` and the
    /// weights.
    fn evaluate_weighted(
        &self,
        x: &Matrix,
        weights: Option<&[f64]>,
        with_grad: bool,
    ) -> Result<crate::acquisition::WeightedEvaluation> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let state = read_lock(&self.state);
        let q = x.nrows();
        let n_bl = state.full_baseline.nrows();
        let x_full = match (&state.x_pending, self.incremental) {
            (Some(p), false) => vstack(&[&state.full_baseline, x, p], d),
            _ => vstack(&[&state.full_baseline, x], d),
        };
        drop(state);
        let n = x_full.nrows();
        let so = SampledObjective::new(
            self.model.as_ref(),
            x_full,
            &self.sampler,
            self.objective.as_deref(),
            with_grad,
        )?;
        let best_f = self.incumbents(&so, n_bl)?;
        let k = so.js.points_per_input();
        let cand = n_bl * k..n * k;
        let ind = match &self.constraints {
            Some(c) => Some(compute_smoothed_feasibility_indicator(
                c,
                &self.constraint_fat,
                &so.samples.points(cand.clone()),
                true,
                with_grad,
            )?),
            None => None,
        };
        let obj = so.obj.rows(n_bl, n - n_bl).into_owned();
        if let Some(w) = weights {
            if w.len() != obj.nrows() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected {} point weights, got {}",
                    obj.nrows(),
                    w.len()
                )));
            }
        }
        let combined = crate::acquisition::monte_carlo::weighted_indicator(
            ind.as_ref().map(|i| &i.values),
            weights,
            true,
            obj.nrows(),
            obj.ncols(),
        );
        if crate::acquisition::monte_carlo::has_nan(&so.obj) || best_f.iter().any(|b| b.is_nan()) {
            return Ok(crate::acquisition::monte_carlo::nan_evaluation(
                q,
                d,
                weights.map_or(0, <[f64]>::len),
                with_grad,
            ));
        }
        let r = log_improvement_reduce(
            &obj,
            |s| best_f[s],
            combined.as_ref(),
            self.fat,
            self.tau_relu,
            self.tau_max,
            with_grad,
        );
        if !with_grad {
            return Ok((r.value, None));
        }
        let mut obj_bar = Matrix::zeros(n, obj.ncols());
        obj_bar
            .rows_mut(n_bl, n - n_bl)
            .copy_from(&r.obj_bar.expect("gradient requested"));
        let (ind_bar, w_bar) = crate::acquisition::monte_carlo::split_indicator_adjoint(
            ind.as_ref().map(|i| &i.values),
            weights,
            true,
            r.ind_bar,
        );
        let samples_bar = match (ind, ind_bar) {
            (Some(ind), Some(ib)) => {
                let cand_bar = ind.backward(&ib)?;
                Some(so.samples.embed_points_adjoint(cand, &cand_bar))
            }
            _ => None,
        };
        let grad = so.backward(self.model.as_ref(), &obj_bar, samples_bar, n_bl..n_bl + q)?;
        Ok((r.value, Some((grad, w_bar))))
    }
}

impl crate::acquisition::SampleReducing for QLogNoisyExpectedImprovement {
    fn evaluate_weighted(
        &self,
        x: &Matrix,
        weights: &[f64],
        with_grad: bool,
    ) -> Result<crate::acquisition::WeightedEvaluation> {
        QLogNoisyExpectedImprovement::evaluate_weighted(self, x, Some(weights), with_grad)
    }

    fn log_utilities(&self) -> bool {
        true
    }
}

impl AcquisitionFunction for QLogNoisyExpectedImprovement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    fn as_sample_reducing(&self) -> Option<&dyn crate::acquisition::SampleReducing> {
        Some(self)
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
        *self.lower_bound.lock().unwrap_or_else(|e| e.into_inner()) = None;
        Ok(())
    }

    fn is_log(&self) -> bool {
        true
    }
}

/// Options of [`QLogProbabilityOfFeasibility`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QLogProbabilityOfFeasibilityOptions {
    /// The sampler of base samples (default: 512 Sobol samples, random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Pending points concatenated to every candidate set.
    pub x_pending: Option<Matrix>,
    /// Use the fat-tailed `fatmoid` and `fatmax` (default `true`) instead of
    /// the logistic sigmoid and `logsumexp`.
    pub fat: bool,
    /// Temperature of the smooth maximum over the `q` points (default `1e-2`).
    pub tau_max: f64,
    /// Per-constraint smoothing (BoTorch's `fat` given as a list); `None`
    /// uses `fat` for every constraint.
    pub constraint_fat: Option<Vec<Option<bool>>>,
}

impl Default for QLogProbabilityOfFeasibilityOptions {
    fn default() -> Self {
        Self {
            sampler: None,
            x_pending: None,
            fat: true,
            tau_max: TAU_MAX,
            constraint_fat: None,
        }
    }
}

/// MC-based batch log probability of feasibility (BoTorch's
/// `qLogProbabilityOfFeasibility`):
/// `logmeanexp_s( fatmax_j( log ind_js, τ_max ) )` for the smoothed
/// feasibility indicator `ind` of outcome constraints on the posterior
/// samples (all outputs; no objective is applied).
#[derive(Debug)]
pub struct QLogProbabilityOfFeasibility {
    model: Arc<dyn Model>,
    sampler: Arc<SobolQMCNormalSampler>,
    x_pending: RwLock<Option<Matrix>>,
    constraints: OutcomeConstraints,
    constraint_fat: Vec<Option<bool>>,
    fat: bool,
    tau_max: f64,
}

impl QLogProbabilityOfFeasibility {
    /// Creates `qLogProbabilityOfFeasibility(model, constraints, ...)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for batched models,
    /// [`BotorchError::InvalidArgument`] for invalid temperatures or
    /// constraint settings and [`BotorchError::DimensionMismatch`] for pending
    /// points of the wrong dimension.
    pub fn new(
        model: Arc<dyn Model>,
        constraints: OutcomeConstraints,
        options: QLogProbabilityOfFeasibilityOptions,
    ) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        let tau_max = check_tau(options.tau_max, "tau_max")?;
        let cfat = constraint_fat(Some(&constraints), options.fat, options.constraint_fat)?;
        if let Some(p) = &options.x_pending {
            check_pending(p, model.dim())?;
        }
        Ok(Self {
            sampler: sampler_or_default(options.sampler)?,
            x_pending: RwLock::new(options.x_pending),
            model,
            constraints,
            constraint_fat: cfat,
            fat: options.fat,
            tau_max,
        })
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let q = x.nrows();
        let x_full = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let js = JointSamples::new(self.model.as_ref(), x_full, &self.sampler)?;
        // All posterior points (with a one-to-many input transform, every
        // perturbation of every point, as in BoTorch).
        let n = js.num_points();
        let samples: McSamples = js.samples(0..n);
        let ind = compute_smoothed_feasibility_indicator(
            &self.constraints,
            &self.constraint_fat,
            &samples,
            true,
            with_grad,
        )?;
        let (value, l_bar) = log_reduce(&ind.values, self.fat, self.tau_max, with_grad);
        match l_bar {
            None => Ok((value, None)),
            Some(lb) => {
                let sbar = ind.backward(&lb)?;
                let grad = js.backward(self.model.as_ref(), 0..n, &sbar, 0..q)?;
                Ok((value, Some(grad)))
            }
        }
    }
}

impl AcquisitionFunction for QLogProbabilityOfFeasibility {
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
        read_lock(&self.x_pending).clone()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.model.dim())?;
        }
        *write_lock(&self.x_pending) = x_pending.cloned();
        Ok(())
    }

    fn is_log(&self) -> bool {
        true
    }
}
