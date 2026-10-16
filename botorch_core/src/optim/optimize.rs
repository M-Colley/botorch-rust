//! Acquisition-function optimization mirroring `botorch.optim.optimize_acqf`
//! with the default `gen_candidates_scipy`: L-BFGS-B for box constraints and
//! SLSQP when linear or nonlinear parameter constraints are present.
//!
//! * Initial conditions come from [`gen_batch_initial_conditions`]: `raw_samples`
//!   scrambled Sobol candidate sets (hit-and-run samples from the constraint
//!   polytope with linear constraints), Boltzmann (or non-negative) selection
//!   of `num_restarts` starting points; or from explicit
//!   `batch_initial_conditions` / an `ic_generator` (required for nonlinear
//!   constraints, as in BoTorch).
//! * Without constraints, every restart is optimized by an independent
//!   L-BFGS-B run ([`crate::lbfgsb`], a port of L-BFGS-B 3.0 reproducing
//!   scipy's iterates) on the negated acquisition value with exact gradients.
//!   This is exactly BoTorch's default "parallel" path (`fmin_l_bfgs_b_batched`,
//!   used with SciPy 1.13–1.18), where each restart is a separate problem; the
//!   restarts run in parallel with rayon. The default options are those of
//!   `gen_candidates_scipy` on BoTorch's main branch: `maxiter = 2000`, no
//!   limit on function evaluations (the batched driver does not check
//!   `maxfun`) and SciPy's defaults otherwise (`maxcor = 10`,
//!   `ftol = 2.2e-9`, `gtol = 1e-5`, `maxls = 20`).
//! * With constraints, chunks of `batch_limit` restarts (by default all
//!   restarts with linear constraints and a single restart with nonlinear
//!   constraints) are optimized as one SLSQP problem each, exactly like
//!   BoTorch ([`gen_candidates_slsqp`]); infeasible results are projected onto
//!   the feasible set ([`project_to_feasible_space`]) and only feasible
//!   restarts are returned as the best candidates.
//! * The final candidates are clamped to the bounds and re-evaluated; the best
//!   restart is returned. If a restart terminates abnormally (BoTorch's
//!   `OptimizationWarning`), the optimization is repeated once with new
//!   initial conditions, as with BoTorch's `retry_on_optimization_warning`.
//! * `sequential = true` generates `q` candidates greedily, conditioning on the
//!   previous ones as pending points.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::generation::gen::{gen_candidates_torch, TorchGenOptions};
use crate::lbfgsb::{self, LbfgsbError, LbfgsbOptions, LbfgsbStatus};
use crate::optim::initializers::{
    apply_fixed_features, gen_batch_initial_conditions, InitialConditionOptions, SampleAroundBest,
    TopnOptions, DEFAULT_ALPHA, DEFAULT_ETA, DEFAULT_N_BURNIN, DEFAULT_N_THINNING,
};
use crate::optim::one_shot::OneShotInitialConditionOptions;
use crate::optim::parameter_constraints::{
    evaluate_feasibility, ParameterConstraints, CONSTRAINT_TOLERANCE,
};
use crate::optim::slsqp::{
    clamp_to_bounds, gen_candidates_slsqp, project_to_feasible_space, SlsqpOptions,
};
use crate::utils::sampling::{check_bounds, random_seed};
use crate::Matrix;

/// Default maximum number of L-BFGS-B iterations per restart
/// (`gen_candidates_scipy`'s `options.get("maxiter", 2000)`).
pub const DEFAULT_MAXITER: usize = 2000;

/// The L-BFGS-B options used by default for acquisition optimization.
pub fn default_lbfgsb_options() -> LbfgsbOptions {
    LbfgsbOptions {
        maxiter: DEFAULT_MAXITER,
        maxfun: usize::MAX,
        ..LbfgsbOptions::default()
    }
}

/// What an initial-condition generator is asked for (the arguments BoTorch
/// passes to `ic_generator`).
pub struct IcRequest<'a> {
    /// The acquisition function.
    pub acq_function: &'a dyn AcquisitionFunction,
    /// The bounds (`2 × d`).
    pub bounds: &'a Matrix,
    /// Points per candidate set.
    pub q: usize,
    /// Number of initial conditions to generate.
    pub num_restarts: usize,
    /// Number of raw samples.
    pub raw_samples: usize,
    /// The options of the optimization (fixed features, constraints, seed).
    pub options: &'a OptimizeAcqfOptions,
}

/// The function type of an [`IcGenerator`].
pub type IcGeneratorFn = dyn Fn(&IcRequest<'_>) -> Result<Vec<Matrix>> + Send + Sync;

/// A generator of initial conditions (BoTorch's `ic_generator`), returning
/// `num_restarts` candidate sets (`q × d`). Required with nonlinear
/// constraints unless explicit initial conditions are given.
#[derive(Clone)]
pub struct IcGenerator(pub Arc<IcGeneratorFn>);

impl Debug for IcGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("IcGenerator")
    }
}

impl IcGenerator {
    /// A generator from a closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&IcRequest<'_>) -> Result<Vec<Matrix>> + Send + Sync + 'static,
    {
        Self(Arc::new(f))
    }
}

/// The optimizer of [`optimize_acqf`] (BoTorch's `options["method"]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizationMethod {
    /// SLSQP if there are parameter constraints, else L-BFGS-B (BoTorch's
    /// default).
    #[default]
    Auto,
    /// SLSQP (`"SLSQP"`): chunks of `batch_limit` restarts are optimized
    /// jointly within the bounds and the constraints.
    Slsqp,
    /// L-BFGS-B (`"L-BFGS-B"`): independent restarts within the bounds; not
    /// available with parameter constraints (BoTorch raises "Method L-BFGS-B
    /// cannot handle constraints.").
    Lbfgsb,
}

/// The function type of a [`PostProcessing`].
pub type PostProcessingFn = dyn Fn(&[Matrix]) -> Result<Vec<Matrix>> + Send + Sync;

/// A post-processing function applied to the optimized candidate sets of all
/// restarts (BoTorch's `post_processing_func` on the `b × q × d` candidates,
/// e.g. rounding); the acquisition values are recomputed afterwards.
#[derive(Clone)]
pub struct PostProcessing(pub Arc<PostProcessingFn>);

impl Debug for PostProcessing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PostProcessing")
    }
}

impl PostProcessing {
    /// A post-processing function from a closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&[Matrix]) -> Result<Vec<Matrix>> + Send + Sync + 'static,
    {
        Self(Arc::new(f))
    }

    /// Applies the function and checks the shapes of the result.
    ///
    /// # Errors
    /// Errors of the function, and [`BotorchError::DimensionMismatch`] if
    /// it changes the number or the shapes of the candidate sets.
    pub fn apply(&self, sets: &[Matrix]) -> Result<Vec<Matrix>> {
        let out = (self.0)(sets)?;
        if out.len() != sets.len() || out.iter().zip(sets).any(|(a, b)| a.shape() != b.shape()) {
            return Err(BotorchError::DimensionMismatch(
                "post_processing_func must return candidates of the shape of its input".into(),
            ));
        }
        Ok(out)
    }
}

/// What a custom candidate generator is asked for (the arguments BoTorch
/// passes to `gen_candidates`).
pub struct GenRequest<'a> {
    /// The initial conditions of the chunk (`q × d` each).
    pub initial_conditions: &'a [Matrix],
    /// The acquisition function.
    pub acq_function: &'a dyn AcquisitionFunction,
    /// The bounds (`2 × d`).
    pub bounds: &'a Matrix,
    /// The fixed features.
    pub fixed_features: &'a BTreeMap<usize, f64>,
    /// The timeout of the chunk.
    pub timeout_sec: Option<f64>,
}

/// The function type of a [`CustomGenerator`]: returns the candidate sets and
/// their acquisition values.
pub type CustomGeneratorFn =
    dyn Fn(&GenRequest<'_>) -> Result<(Vec<Matrix>, Vec<f64>)> + Send + Sync;

/// A user-provided candidate generator (BoTorch's `gen_candidates` callable).
#[derive(Clone)]
pub struct CustomGenerator(pub Arc<CustomGeneratorFn>);

impl Debug for CustomGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CustomGenerator")
    }
}

impl CustomGenerator {
    /// A generator from a closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&GenRequest<'_>) -> Result<(Vec<Matrix>, Vec<f64>)> + Send + Sync + 'static,
    {
        Self(Arc::new(f))
    }
}

/// The candidate generator of [`optimize_acqf`] (BoTorch's `gen_candidates`).
#[derive(Clone, Debug, Default)]
pub enum CandidateGenerator {
    /// `gen_candidates_scipy` (default): L-BFGS-B, or SLSQP with constraints.
    #[default]
    Scipy,
    /// `gen_candidates_torch`: Adam with an exponential-moving-average
    /// stopping criterion on every chunk of `batch_limit` restarts.
    Torch(TorchGenOptions),
    /// A user-provided generator, called for every chunk of `batch_limit`
    /// restarts.
    Custom(CustomGenerator),
}

/// Options of [`optimize_acqf`] (BoTorch's keyword arguments and `options`
/// dictionary entries).
#[derive(Clone, Debug)]
pub struct OptimizeAcqfOptions {
    /// Seed of the raw Sobol samples (`options["seed"]`); it also seeds the
    /// random selection of initial conditions (which BoTorch draws from
    /// torch's global RNG) and the polytope sampler. `None` draws a random
    /// seed.
    pub seed: Option<u64>,
    /// L-BFGS-B settings of each restart (default [`default_lbfgsb_options`]).
    pub lbfgsb: LbfgsbOptions,
    /// SLSQP settings with constraints (`options["maxiter"]`,
    /// `options["ftol"]`; defaults 2000 and SciPy's `1e-6`).
    pub slsqp: SlsqpOptions,
    /// The optimizer (BoTorch's `options["method"]`): by default SLSQP with
    /// parameter constraints and L-BFGS-B without.
    pub method: OptimizationMethod,
    /// Number of restarts optimized per chunk (`options["batch_limit"]`).
    /// Without constraints restarts are independent problems, so this only
    /// bounds the work in flight; with constraints each chunk is one SLSQP
    /// problem (default: all restarts, or 1 with nonlinear constraints, which
    /// require 1).
    pub batch_limit: Option<usize>,
    /// Number of raw samples evaluated per chunk (`options["init_batch_limit"]`,
    /// falling back to `batch_limit`); does not change the result.
    pub init_batch_limit: Option<usize>,
    /// Boltzmann temperature of the initialization (`options["eta"]`).
    pub eta: f64,
    /// Threshold fraction of the non-negative initialization
    /// (`options["alpha"]`).
    pub alpha: f64,
    /// Use the non-negative initialization heuristic (`options["nonnegative"]`);
    /// it is also used for acquisition functions that report
    /// [`AcquisitionFunction::is_nonnegative`].
    pub nonnegative: bool,
    /// Features fixed to given values (`fixed_features`): they are excluded
    /// from the optimization and set in every candidate.
    pub fixed_features: BTreeMap<usize, f64>,
    /// Explicit initial conditions (`batch_initial_conditions`, each `q × d`);
    /// `num_restarts - len` further ones are generated from raw samples.
    pub batch_initial_conditions: Option<Vec<Matrix>>,
    /// Generate the `q` candidates one at a time, each conditioned on the
    /// previous ones as pending points (`sequential=True`).
    pub sequential: bool,
    /// Repeat the optimization once with new initial conditions if a restart
    /// terminates abnormally (`retry_on_optimization_warning`, default true).
    pub retry_on_optimization_warning: bool,
    /// Linear and nonlinear parameter constraints (`inequality_constraints`,
    /// `equality_constraints`, `nonlinear_inequality_constraints`).
    pub constraints: ParameterConstraints,
    /// Burn-in steps of the polytope sampler of the initial conditions
    /// (`options["n_burnin"]`, default 10000).
    pub n_burnin: usize,
    /// Thinning of the polytope sampler (`options["n_thinning"]`, default 32).
    pub n_thinning: usize,
    /// Generator of initial conditions (BoTorch's `ic_generator`).
    pub ic_generator: Option<IcGenerator>,
    /// Settings of the initial conditions of one-shot acquisition functions
    /// (`options["frac_random"]`, `["num_inner_restarts"]`,
    /// `["raw_inner_samples"]`, `["eta"]`).
    pub one_shot: OneShotInitialConditionOptions,
    /// Post-processing of the optimized candidates of all restarts
    /// (`post_processing_func`); the acquisition values are recomputed.
    pub post_processing: Option<PostProcessing>,
    /// Time limit in seconds (`timeout_sec`), split evenly over the
    /// sequential steps, the chunks of `batch_limit` restarts and the
    /// L-BFGS-B problems of a chunk; the optimizers stop after the
    /// iteration that exceeds it (not enforced for SLSQP).
    pub timeout_sec: Option<f64>,
    /// L-BFGS-B without constraints: `None` optimizes every restart as an
    /// independent problem (BoTorch's parallel `fmin_l_bfgs_b_batched` path,
    /// the default); `Some(k)` optimizes groups of `k` restarts of a chunk as
    /// one problem whose objective is the sum of their values (BoTorch's
    /// serial `scipy.optimize.minimize` path, taken with `timeout_sec`, with
    /// options other than `maxiter`, `ftol`, `pgtol`, `factr`, `tol`,
    /// `max_cor` and `maxls`, and with
    /// `options["max_optimization_problem_aggregation_size"] = k`; use
    /// `usize::MAX` for one problem per chunk). In that mode scipy's
    /// `maxfun` default of 15000 applies, which the caller sets in
    /// [`OptimizeAcqfOptions::lbfgsb`].
    pub lbfgsb_aggregation: Option<usize>,
    /// The candidate generator (`gen_candidates`).
    pub gen_candidates: CandidateGenerator,
    /// Select the initial conditions by their top acquisition values
    /// (`options["topn"]` with `["largest"]` / `["sorted"]`).
    pub topn: Option<TopnOptions>,
    /// Add points sampled around the best baseline points to the raw samples
    /// (`options["sample_around_best"]`; the Python layer finds the best
    /// points like BoTorch's `sample_points_around_best`).
    pub sample_around_best: Option<SampleAroundBest>,
}

impl OptimizeAcqfOptions {
    /// The settings of [`gen_batch_initial_conditions`] corresponding to
    /// these options.
    pub fn initial_condition_options(&self) -> InitialConditionOptions {
        InitialConditionOptions {
            seed: self.seed,
            eta: self.eta,
            alpha: self.alpha,
            nonnegative: self.nonnegative,
            batch_limit: self.init_batch_limit.or(self.batch_limit),
            fixed_features: self.fixed_features.clone(),
            inequality_constraints: self.constraints.inequality.clone(),
            equality_constraints: self.constraints.equality.clone(),
            n_burnin: self.n_burnin,
            n_thinning: self.n_thinning,
            topn: self.topn,
            sample_around_best: self.sample_around_best.clone(),
        }
    }
}

impl Default for OptimizeAcqfOptions {
    fn default() -> Self {
        Self {
            seed: None,
            lbfgsb: default_lbfgsb_options(),
            slsqp: SlsqpOptions::default(),
            method: OptimizationMethod::Auto,
            batch_limit: None,
            init_batch_limit: None,
            eta: DEFAULT_ETA,
            alpha: DEFAULT_ALPHA,
            nonnegative: false,
            fixed_features: BTreeMap::new(),
            batch_initial_conditions: None,
            sequential: false,
            retry_on_optimization_warning: true,
            constraints: ParameterConstraints::default(),
            n_burnin: DEFAULT_N_BURNIN,
            n_thinning: DEFAULT_N_THINNING,
            ic_generator: None,
            one_shot: OneShotInitialConditionOptions::default(),
            post_processing: None,
            timeout_sec: None,
            lbfgsb_aggregation: None,
            gen_candidates: CandidateGenerator::Scipy,
            topn: None,
            sample_around_best: None,
        }
    }
}

/// Termination of the optimizer of a restart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerStatus {
    /// L-BFGS-B (box constraints).
    Lbfgsb(LbfgsbStatus),
    /// SLSQP (parameter constraints) with SciPy's exit mode of the chunk's
    /// joint problem (0: success, 9: iteration limit).
    Slsqp(i32),
    /// Adam (`gen_candidates_torch`); `true` if it stopped on the timeout.
    Torch(bool),
    /// A custom candidate generator.
    Custom,
}

/// The result of one restart of [`gen_candidates_lbfgsb`] or of the SLSQP
/// path.
#[derive(Clone, Debug, PartialEq)]
pub struct RestartResult {
    /// The optimized candidate set (`q × d`), clamped to the bounds.
    pub candidates: Matrix,
    /// The acquisition value at `candidates`.
    pub acq_value: f64,
    /// Iterations (of the chunk's joint problem for SLSQP).
    pub iterations: usize,
    /// Acquisition evaluations (with gradient; for SLSQP evaluations of the
    /// chunk's joint objective).
    pub evaluations: usize,
    /// Termination reason.
    pub status: OptimizerStatus,
    /// Whether the candidates satisfy the parameter constraints (always true
    /// without constraints).
    pub feasible: bool,
}

/// Detailed result of [`optimize_acqf_detailed`].
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizeAcqfResult {
    /// The candidates (`q × d`).
    pub candidates: Matrix,
    /// The acquisition value of the joint candidate set, or for sequential
    /// optimization the value of the last greedy step.
    pub acq_value: f64,
    /// BoTorch's returned acquisition values: one joint value, or `q` values
    /// for sequential optimization (each conditional on the previous
    /// candidates).
    pub acq_values: Vec<f64>,
    /// All restarts of the (last) joint optimization, in restart order
    /// (BoTorch's `return_best_only=False` output).
    pub restarts: Vec<RestartResult>,
    /// Messages corresponding to BoTorch's warnings (random initial
    /// conditions, retried optimizations).
    pub warnings: Vec<String>,
}

/// Optimizes `acq_function` over the box `bounds` (`2 × d`) for `q` points
/// with `num_restarts` restarts initialized from `raw_samples` raw samples,
/// and returns the best `q × d` candidate set and its acquisition value
/// (BoTorch's `optimize_acqf(acq_function, bounds, q, num_restarts,
/// raw_samples, options, ...)`). See [`optimize_acqf_detailed`].
///
/// # Errors
/// See [`optimize_acqf_detailed`].
pub fn optimize_acqf(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    options: &OptimizeAcqfOptions,
) -> Result<(Matrix, f64)> {
    optimize_acqf_detailed(acq_function, bounds, q, num_restarts, raw_samples, options)
        .map(|r| (r.candidates, r.acq_value))
}

/// [`optimize_acqf`] returning all restarts, the per-step values of
/// sequential optimization and diagnostic messages. `raw_samples = 0` means
/// BoTorch's `raw_samples=None` (only explicit or generated initial
/// conditions).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] / [`BotorchError::InvalidArgument`] for
/// invalid bounds, sizes, fixed features, constraints or initial conditions;
/// [`BotorchError::Unsupported`] for sequential optimization of acquisition
/// functions without pending-point support or with inter-point constraints;
/// [`BotorchError::Runtime`] for nonlinear constraints without initial
/// conditions or generator; [`BotorchError::CandidateGeneration`] if no
/// feasible candidates were found; [`BotorchError::Numerical`] for NaN
/// gradients; and errors of the acquisition function and the constraints.
pub fn optimize_acqf_detailed(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    options: &OptimizeAcqfOptions,
) -> Result<OptimizeAcqfResult> {
    let d = check_bounds(bounds)?;
    if d != acq_function.dim() {
        return Err(BotorchError::DimensionMismatch(format!(
            "bounds have {d} columns but the acquisition function has d = {}",
            acq_function.dim()
        )));
    }
    if q == 0 || num_restarts == 0 {
        return Err(BotorchError::InvalidArgument(
            "q and num_restarts must be positive".into(),
        ));
    }
    if let Some(eq) = acq_function.expected_q() {
        if (options.sequential && eq != 1) || (!options.sequential && eq != q) {
            return Err(BotorchError::DimensionMismatch(format!(
                "the acquisition function requires q = {eq}, got q = {q}"
            )));
        }
    }
    check_fixed_features(&options.fixed_features, bounds)?;
    // BoTorch's order of checks: initial conditions, sequential optimization,
    // then the constraints themselves.
    if !options.constraints.nonlinear.is_empty()
        && options.batch_initial_conditions.is_none()
        && options.ic_generator.is_none()
    {
        return Err(BotorchError::Runtime(
            "`ic_generator` must be given if there are non-linear inequality constraints.".into(),
        ));
    }
    if options.batch_initial_conditions.is_none()
        && options.ic_generator.is_none()
        && raw_samples == 0
    {
        return Err(BotorchError::InvalidArgument(
            "Must specify `raw_samples` when `batch_initial_conditions` is None`.".into(),
        ));
    }
    let sequential = options.sequential && q > 1;
    if sequential {
        options.constraints.validate_sequential()?;
    }
    // One-shot acquisition functions optimize augmented candidate sets.
    let q_opt = acq_function
        .as_one_shot()
        .map_or(q, |os| os.get_augmented_q_batch_size(q));
    options
        .constraints
        .validate(if sequential { 1 } else { q_opt }, d)?;
    if options.method == OptimizationMethod::Lbfgsb && !options.constraints.is_empty() {
        return Err(BotorchError::InvalidArgument(
            "Method L-BFGS-B cannot handle constraints.".into(),
        ));
    }
    let seed = options.seed.unwrap_or_else(random_seed);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x6f70_7469_6d69_7a65);
    let joint_options = OptimizeAcqfOptions {
        seed: Some(seed),
        ..options.clone()
    };

    // All features fixed: nothing to optimize.
    if options.fixed_features.len() == d {
        let mut x = Matrix::zeros(q_opt, d);
        apply_fixed_features(&mut x, &options.fixed_features);
        let v = acq_function.forward(&x)?;
        let x = match acq_function.as_one_shot() {
            Some(os) => os.extract_candidates(&x)?,
            None => x,
        };
        return Ok(OptimizeAcqfResult {
            candidates: x,
            acq_value: v,
            acq_values: vec![v],
            restarts: Vec::new(),
            warnings: Vec::new(),
        });
    }

    if !sequential {
        return optimize_batch(
            acq_function,
            bounds,
            q,
            num_restarts,
            raw_samples,
            &joint_options,
            &mut rng,
        );
    }

    // Sequential greedy optimization via pending points.
    if options.batch_initial_conditions.is_some() {
        return Err(BotorchError::Unsupported(
            "`batch_initial_conditions` is not supported for sequential optimization. Either \
             avoid specifying `batch_initial_conditions` to use the custom initializer or use \
             the `ic_generator` kwarg to generate initial conditions for the case of nonlinear \
             inequality constraints."
                .into(),
        ));
    }
    if acq_function.as_one_shot().is_some() {
        return Err(BotorchError::Unsupported(
            "sequential optimization currently not supported for one-shot acquisition \
             functions. Must have `sequential=False`."
                .into(),
        ));
    }
    // The total timeout is split evenly over the sequential steps.
    let step_options = OptimizeAcqfOptions {
        timeout_sec: options.timeout_sec.map(|t| t / q as f64),
        ..joint_options.clone()
    };
    let joint_options = step_options;
    let base_pending = acq_function.x_pending();
    let mut candidates: Vec<Matrix> = Vec::with_capacity(q);
    let mut values = Vec::with_capacity(q);
    let mut warnings = Vec::new();
    let mut last = None;
    for i in 0..q {
        if i > 0 {
            let mut parts: Vec<&Matrix> = Vec::new();
            if let Some(p) = &base_pending {
                parts.push(p);
            }
            parts.extend(candidates.iter());
            let pending = crate::acquisition::monte_carlo::vstack(&parts, d);
            acq_function.set_x_pending(Some(&pending))?;
        }
        let step = optimize_batch(
            acq_function,
            bounds,
            1,
            num_restarts,
            raw_samples,
            &joint_options,
            &mut rng,
        );
        let step = match step {
            Ok(s) => s,
            Err(e) => {
                let _ = acq_function.set_x_pending(base_pending.as_ref());
                return Err(e);
            }
        };
        candidates.push(step.candidates.clone());
        values.push(step.acq_value);
        warnings.extend(step.warnings.iter().cloned());
        last = Some(step);
    }
    acq_function.set_x_pending(base_pending.as_ref())?;
    let parts: Vec<&Matrix> = candidates.iter().collect();
    let last = last.expect("q > 1 steps");
    Ok(OptimizeAcqfResult {
        candidates: crate::acquisition::monte_carlo::vstack(&parts, d),
        acq_value: *values.last().expect("q > 1"),
        acq_values: values,
        restarts: last.restarts,
        warnings,
    })
}

fn check_fixed_features(fixed: &BTreeMap<usize, f64>, bounds: &Matrix) -> Result<()> {
    let d = bounds.ncols();
    for (&k, &v) in fixed {
        if k >= d {
            return Err(BotorchError::InvalidArgument(format!(
                "fixed feature index {k} out of range for d = {d}"
            )));
        }
        if !v.is_finite() {
            return Err(BotorchError::InvalidArgument(format!(
                "fixed feature {k} must be finite, got {v}"
            )));
        }
    }
    Ok(())
}

/// Joint optimization of `q` points (BoTorch's `_optimize_acqf_batch`).
fn optimize_batch(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    options: &OptimizeAcqfOptions,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<OptimizeAcqfResult> {
    let d = bounds.ncols();
    let constrained = !options.constraints.is_empty();
    let use_slsqp = match options.method {
        OptimizationMethod::Auto => constrained,
        OptimizationMethod::Slsqp => true,
        OptimizationMethod::Lbfgsb => false,
    };
    let one_shot = acq_function.as_one_shot();
    // The number of points of the optimized sets (augmented for one-shot
    // acquisition functions).
    let q_opt = one_shot.map_or(q, |os| os.get_augmented_q_batch_size(q));
    let provided = options.batch_initial_conditions.clone().unwrap_or_default();
    for ic in &provided {
        if ic.shape() != (q_opt, d) {
            return Err(BotorchError::DimensionMismatch(format!(
                "batch_initial_conditions must be {q_opt} x {d}, got {}x{}",
                ic.nrows(),
                ic.ncols()
            )));
        }
    }
    let required = num_restarts.saturating_sub(provided.len());
    let ic_options = options.initial_condition_options();
    let generate = raw_samples > 0 && required > 0;
    let mut warnings = Vec::new();
    let make_ics =
        |rng: &mut Xoshiro256PlusPlus, warnings: &mut Vec<String>| -> Result<Vec<Matrix>> {
            let mut ics = provided.clone();
            if generate {
                let generated = match &options.ic_generator {
                    Some(generator) => {
                        let request = IcRequest {
                            acq_function,
                            bounds,
                            q,
                            num_restarts: required,
                            raw_samples,
                            options,
                        };
                        let ics = (generator.0)(&request)?;
                        for ic in &ics {
                            if ic.shape() != (q_opt, d) {
                                return Err(BotorchError::DimensionMismatch(format!(
                                    "the ic_generator returned a {}x{} initial condition, \
                                     expected {q_opt} x {d}",
                                    ic.nrows(),
                                    ic.ncols()
                                )));
                            }
                        }
                        ics
                    }
                    None if one_shot.is_some() => {
                        let os = one_shot.expect("one-shot acquisition function");
                        let (ics, ws) = os.gen_one_shot_initial_conditions(
                            bounds,
                            q,
                            required,
                            raw_samples,
                            options,
                            &mut *rng,
                        )?;
                        warnings.extend(ws);
                        ics
                    }
                    None => {
                        let (generated, fallback) = gen_batch_initial_conditions(
                            acq_function,
                            bounds,
                            q,
                            required,
                            raw_samples,
                            &ic_options,
                            rng,
                        )?;
                        if fallback {
                            warnings.push(
                                "Unable to find non-zero acquisition function values - initial \
                                 conditions are being selected randomly."
                                    .into(),
                            );
                        }
                        generated
                    }
                };
                ics.extend(generated);
            }
            if ics.is_empty() {
                return Err(BotorchError::InvalidArgument(
                    "Either `batch_initial_conditions` or `raw_samples` must be set.".into(),
                ));
            }
            Ok(ics)
        };
    let run = |ics: &[Matrix]| -> Result<(Vec<RestartResult>, Vec<String>)> {
        generate_candidates(acq_function, bounds, ics, num_restarts, use_slsqp, options)
    };
    let ics = make_ics(rng, &mut warnings)?;
    let (mut restarts, opt_warnings) = run(&ics)?;
    if !opt_warnings.is_empty() && options.retry_on_optimization_warning {
        if generate {
            warnings.push(if use_slsqp {
                format!(
                    "Optimization failed in `gen_candidates_scipy` with the following \
                     warning(s):\n{opt_warnings:?}\nTrying again with a new set of initial \
                     conditions."
                )
            } else {
                "Optimization failed in L-BFGS-B (abnormal termination in the line search); \
                 trying again with a new set of initial conditions."
                    .into()
            });
            let ics = make_ics(rng, &mut warnings)?;
            let (rs, ws) = run(&ics)?;
            restarts = rs;
            if !ws.is_empty() {
                warnings.push(
                    "Optimization failed on the second try, after generating a new set of \
                     initial conditions."
                        .into(),
                );
            }
        } else if provided.is_empty() || required > 0 {
            // BoTorch announces a retry but has no raw samples to retry with.
            warnings.push(format!(
                "Optimization failed in `gen_candidates_scipy` with the following \
                 warning(s):\n{opt_warnings:?}\nTrying again with a new set of initial \
                 conditions."
            ));
        } else {
            warnings.push(if use_slsqp {
                format!(
                    "Optimization failed in `gen_candidates_scipy` with the following \
                     warning(s):\n{opt_warnings:?}\nBecause you specified \
                     `batch_initial_conditions` larger than required `num_restarts`, \
                     optimization will not be retried with new initial conditions and will \
                     proceed with the current solution. Suggested remediation: Try again with \
                     different `batch_initial_conditions`, don't provide \
                     `batch_initial_conditions`, or increase `num_restarts`."
                )
            } else {
                "Optimization failed in L-BFGS-B (abnormal termination in the line search); \
                 not retried because all initial conditions were provided."
                    .into()
            });
        }
    }
    if let Some(pp) = &options.post_processing {
        post_process(acq_function, &mut restarts, pp)?;
    }
    if constrained {
        enforce_feasibility(acq_function, bounds, &mut restarts, options)?;
    }
    let best = best_restart(&restarts)?;
    let r = &restarts[best];
    let candidates = match one_shot {
        Some(os) => os.extract_candidates(&r.candidates)?,
        None => r.candidates.clone(),
    };
    Ok(OptimizeAcqfResult {
        candidates,
        acq_value: r.acq_value,
        acq_values: vec![r.acq_value],
        restarts,
        warnings,
    })
}

/// The number of restarts per chunk (BoTorch's `batch_limit`, by default
/// `num_restarts`, or 1 with nonlinear constraints).
fn chunk_size(options: &OptimizeAcqfOptions, num_restarts: usize) -> usize {
    let nonlinear = !options.constraints.nonlinear.is_empty();
    options
        .batch_limit
        .unwrap_or(if nonlinear { 1 } else { num_restarts })
        .max(1)
}

/// BoTorch's `_optimize_batch_candidates`: runs the candidate generator on
/// the initial conditions in chunks of `batch_limit` restarts. Returns the
/// restarts (in order) and the messages of BoTorch's `OptimizationWarning`s,
/// which trigger the retry with new initial conditions.
fn generate_candidates(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    ics: &[Matrix],
    num_restarts: usize,
    use_slsqp: bool,
    options: &OptimizeAcqfOptions,
) -> Result<(Vec<RestartResult>, Vec<String>)> {
    let batch_limit = chunk_size(options, num_restarts);
    let n_chunks = ics.len().div_ceil(batch_limit).max(1);
    let chunk_timeout = options.timeout_sec.map(|t| t / n_chunks as f64);
    match &options.gen_candidates {
        CandidateGenerator::Scipy if use_slsqp => {
            run_slsqp(acq_function, bounds, ics, options, batch_limit)
        }
        CandidateGenerator::Scipy => {
            let rs = match options.lbfgsb_aggregation {
                None => run_restarts(acq_function, bounds, ics, options, batch_limit)?,
                Some(k) => {
                    let mut out = Vec::with_capacity(ics.len());
                    for chunk in ics.chunks(batch_limit) {
                        out.extend(gen_candidates_lbfgsb_joint(
                            chunk,
                            acq_function,
                            bounds,
                            &options.fixed_features,
                            &options.lbfgsb,
                            k,
                            chunk_timeout,
                        )?);
                    }
                    out
                }
            };
            let failed = rs
                .iter()
                .any(|r| r.status == OptimizerStatus::Lbfgsb(LbfgsbStatus::AbnormalLineSearch));
            let ws = if failed {
                vec!["abnormal termination in the line search".to_string()]
            } else {
                Vec::new()
            };
            Ok((rs, ws))
        }
        CandidateGenerator::Torch(torch_options) => {
            if !options.constraints.is_empty() {
                return Err(BotorchError::Unsupported(
                    "gen_candidates_torch() got an unexpected keyword argument for the \
                     parameter constraints: `gen_candidates_torch` does not support \
                     parameter constraints."
                        .into(),
                ));
            }
            let mut out = Vec::with_capacity(ics.len());
            for chunk in ics.chunks(batch_limit) {
                let opts = TorchGenOptions {
                    timeout_sec: chunk_timeout,
                    ..*torch_options
                };
                let res = gen_candidates_torch(
                    chunk,
                    acq_function,
                    bounds,
                    &options.fixed_features,
                    &opts,
                    None,
                )?;
                for (candidates, acq_value) in res.candidates.into_iter().zip(res.acq_values) {
                    out.push(RestartResult {
                        candidates,
                        acq_value,
                        iterations: res.steps,
                        evaluations: res.steps,
                        status: OptimizerStatus::Torch(res.timed_out),
                        feasible: true,
                    });
                }
            }
            Ok((out, Vec::new()))
        }
        CandidateGenerator::Custom(generator) => {
            let mut out = Vec::with_capacity(ics.len());
            for chunk in ics.chunks(batch_limit) {
                let request = GenRequest {
                    initial_conditions: chunk,
                    acq_function,
                    bounds,
                    fixed_features: &options.fixed_features,
                    timeout_sec: chunk_timeout,
                };
                let (cands, values) = (generator.0)(&request)?;
                if cands.len() != chunk.len()
                    || values.len() != chunk.len()
                    || cands.iter().zip(chunk).any(|(a, b)| a.shape() != b.shape())
                {
                    return Err(BotorchError::DimensionMismatch(
                        "gen_candidates must return candidates of the shape of the initial \
                         conditions and one value per candidate set"
                            .into(),
                    ));
                }
                for (candidates, acq_value) in cands.into_iter().zip(values) {
                    out.push(RestartResult {
                        candidates,
                        acq_value,
                        iterations: 0,
                        evaluations: 0,
                        status: OptimizerStatus::Custom,
                        feasible: true,
                    });
                }
            }
            Ok((out, Vec::new()))
        }
    }
}

/// Applies the post-processing function to the candidates of all restarts and
/// recomputes their acquisition values (BoTorch's `post_processing_func`).
fn post_process(
    acq_function: &dyn AcquisitionFunction,
    restarts: &mut [RestartResult],
    pp: &PostProcessing,
) -> Result<()> {
    if restarts.is_empty() {
        return Ok(());
    }
    let sets: Vec<Matrix> = restarts.iter().map(|r| r.candidates.clone()).collect();
    let processed = pp.apply(&sets)?;
    let values = acq_function.forward_many(&processed)?;
    for ((r, x), v) in restarts.iter_mut().zip(processed).zip(values) {
        r.candidates = x;
        r.acq_value = v;
    }
    Ok(())
}

/// SLSQP over chunks of `batch_limit` restarts (BoTorch's constrained
/// `_optimize_batch_candidates`); the chunks are independent problems and run
/// in parallel. Returns the restarts and the optimization warnings.
fn run_slsqp(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    ics: &[Matrix],
    options: &OptimizeAcqfOptions,
    batch_limit: usize,
) -> Result<(Vec<RestartResult>, Vec<String>)> {
    let nonlinear = !options.constraints.nonlinear.is_empty();
    if nonlinear && batch_limit != 1 {
        return Err(BotorchError::InvalidArgument(
            "`batch_limit` must be 1 when non-linear inequality constraints are given.".into(),
        ));
    }
    let chunks: Vec<&[Matrix]> = ics.chunks(batch_limit).collect();
    let results = chunks
        .par_iter()
        .map(|c| {
            gen_candidates_slsqp(
                c,
                acq_function,
                bounds,
                &options.fixed_features,
                &options.constraints,
                &options.slsqp,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut restarts = Vec::with_capacity(ics.len());
    let mut warnings = Vec::new();
    for r in results {
        warnings.extend(r.warnings.iter().cloned());
        for (candidates, acq_value) in r.candidates.into_iter().zip(r.acq_values) {
            restarts.push(RestartResult {
                candidates,
                acq_value,
                iterations: r.iterations,
                evaluations: r.evaluations,
                status: OptimizerStatus::Slsqp(r.mode),
                feasible: true,
            });
        }
    }
    Ok((restarts, warnings))
}

/// BoTorch's feasibility post-processing: infeasible restarts are projected
/// onto the feasible set (jointly) and re-evaluated; an error is raised if no
/// restart is feasible afterwards.
fn enforce_feasibility(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    restarts: &mut [RestartResult],
    options: &OptimizeAcqfOptions,
) -> Result<()> {
    let constraints = &options.constraints;
    for r in restarts.iter_mut() {
        r.feasible = evaluate_feasibility(&r.candidates, constraints, CONSTRAINT_TOLERANCE)?;
    }
    let infeasible: Vec<usize> = (0..restarts.len())
        .filter(|&i| !restarts[i].feasible)
        .collect();
    if !infeasible.is_empty() {
        let sets: Vec<Matrix> = infeasible
            .iter()
            .map(|&i| restarts[i].candidates.clone())
            .collect();
        let mut projected =
            project_to_feasible_space(&sets, bounds, constraints, &options.fixed_features)?;
        if let Some(pp) = &options.post_processing {
            projected = pp.apply(&projected)?;
        }
        let values = acq_function.forward_many(&projected)?;
        for ((&i, x), v) in infeasible.iter().zip(projected).zip(values) {
            restarts[i].feasible = evaluate_feasibility(&x, constraints, CONSTRAINT_TOLERANCE)?;
            restarts[i].candidates = x;
            restarts[i].acq_value = v;
        }
    }
    let n_infeasible = restarts.iter().filter(|r| !r.feasible).count();
    if n_infeasible == restarts.len() {
        return Err(BotorchError::CandidateGeneration(format!(
            "The optimizer produced infeasible candidates. {n_infeasible} out of {} batches of \
             candidates were infeasible. Please make sure the constraints are satisfiable and \
             relax them if needed. ",
            restarts.len()
        )));
    }
    Ok(())
}

/// Index of the feasible restart with the largest (non-NaN) value.
fn best_restart(restarts: &[RestartResult]) -> Result<usize> {
    let mut best: Option<usize> = None;
    for (i, r) in restarts.iter().enumerate() {
        if r.acq_value.is_nan() || !r.feasible {
            continue;
        }
        if best.is_none_or(|b| r.acq_value > restarts[b].acq_value) {
            best = Some(i);
        }
    }
    best.ok_or_else(|| {
        BotorchError::Numerical("all restarts produced NaN acquisition values".into())
    })
}

fn run_restarts(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    ics: &[Matrix],
    options: &OptimizeAcqfOptions,
    chunk: usize,
) -> Result<Vec<RestartResult>> {
    let mut out = Vec::with_capacity(ics.len());
    for c in ics.chunks(chunk.max(1)) {
        out.extend(gen_candidates_lbfgsb(
            c,
            acq_function,
            bounds,
            &options.fixed_features,
            &options.lbfgsb,
        )?);
    }
    Ok(out)
}

/// Optimizes groups of `aggregation` initial conditions (`q × d` each) as one
/// L-BFGS-B problem whose objective is the negated sum of their acquisition
/// values (BoTorch's `gen_candidates_scipy` in its serial mode,
/// `scipy.optimize.minimize` with `max_optimization_problem_aggregation_size`),
/// keeping `fixed_features` fixed; the groups run in parallel. With a
/// `timeout_sec`, every group gets `timeout_sec / n_groups` seconds and stops
/// after the iteration that exceeds it (BoTorch's `minimize_with_timeout`).
///
/// # Errors
/// See [`gen_candidates_lbfgsb`].
pub fn gen_candidates_lbfgsb_joint(
    initial_conditions: &[Matrix],
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    fixed_features: &BTreeMap<usize, f64>,
    options: &LbfgsbOptions,
    aggregation: usize,
    timeout_sec: Option<f64>,
) -> Result<Vec<RestartResult>> {
    let d = check_bounds(bounds)?;
    check_fixed_features(fixed_features, bounds)?;
    let free: Vec<usize> = (0..d).filter(|k| !fixed_features.contains_key(k)).collect();
    let groups: Vec<&[Matrix]> = initial_conditions.chunks(aggregation.max(1)).collect();
    let group_timeout = timeout_sec.map(|t| t / groups.len().max(1) as f64);
    let results = groups
        .par_iter()
        .map(|g| {
            optimize_group(
                g,
                acq_function,
                bounds,
                &free,
                fixed_features,
                options,
                group_timeout,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(results.into_iter().flatten().collect())
}

/// One L-BFGS-B problem over several restarts (see
/// [`gen_candidates_lbfgsb_joint`]).
fn optimize_group(
    ics: &[Matrix],
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    free: &[usize],
    fixed: &BTreeMap<usize, f64>,
    options: &LbfgsbOptions,
    timeout_sec: Option<f64>,
) -> Result<Vec<RestartResult>> {
    let start = Instant::now();
    let d = bounds.ncols();
    let b = ics.len();
    if b == 0 {
        return Ok(Vec::new());
    }
    let q = ics[0].nrows();
    for ic in ics {
        if ic.shape() != (q, d) || q == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "initial conditions must be {q} x {d}, got {}x{}",
                ic.nrows(),
                ic.ncols()
            )));
        }
    }
    let nf = free.len();
    let per = q * nf;
    let mut x0 = Vec::with_capacity(b * per);
    let mut lower = Vec::with_capacity(b * per);
    let mut upper = Vec::with_capacity(b * per);
    for ic in ics {
        let mut c = ic.clone();
        clamp_to_bounds(&mut c, bounds, true)?;
        for i in 0..q {
            for &k in free {
                x0.push(c[(i, k)]);
                lower.push(bounds[(0, k)]);
                upper.push(bounds[(1, k)]);
            }
        }
    }
    let assemble = |x: &[f64], r: usize| -> Matrix {
        let mut m = Matrix::zeros(q, d);
        for i in 0..q {
            for (j, &k) in free.iter().enumerate() {
                m[(i, k)] = x[r * per + i * nf + j];
            }
        }
        apply_fixed_features(&mut m, fixed);
        m
    };
    let objective = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
        if x.iter().any(|v| v.is_nan()) {
            return Err(BotorchError::Numerical(format!(
                "{} elements of the {} element array `x` are NaN.",
                x.iter().filter(|v| v.is_nan()).count(),
                x.len()
            )));
        }
        let sets: Vec<Matrix> = (0..b).map(|r| assemble(x, r)).collect();
        let evals = sets
            .par_iter()
            .map(|s| acq_function.forward_with_grad(s))
            .collect::<Result<Vec<_>>>()?;
        let mut total = 0.0;
        let mut grad = Vec::with_capacity(b * per);
        for (v, g) in &evals {
            total += -v;
            for i in 0..q {
                for &k in free {
                    grad.push(-g[(i, k)]);
                }
            }
        }
        if grad.iter().any(|v| v.is_nan()) {
            return Err(BotorchError::Numerical(format!(
                "{} elements of the {} element gradient array `gradf` are NaN. This often \
                 indicates numerical issues.",
                grad.iter().filter(|v| v.is_nan()).count(),
                grad.len()
            )));
        }
        Ok((total, grad))
    };
    let callback = |_: usize, _: &[f64], _: f64| -> bool {
        timeout_sec.is_some_and(|t| start.elapsed().as_secs_f64() > t)
    };
    let res =
        match lbfgsb::minimize_with_callback(objective, &x0, &lower, &upper, options, callback) {
            Ok(r) => r,
            Err(LbfgsbError::Objective(e)) => return Err(e),
            Err(LbfgsbError::InvalidInput(msg)) => return Err(BotorchError::InvalidArgument(msg)),
        };
    let mut sets: Vec<Matrix> = (0..b).map(|r| assemble(&res.x, r)).collect();
    for s in &mut sets {
        clamp_to_bounds(s, bounds, true)?;
    }
    let values = acq_function.forward_many(&sets)?;
    Ok(sets
        .into_iter()
        .zip(values)
        .map(|(candidates, acq_value)| RestartResult {
            candidates,
            acq_value,
            iterations: res.iterations,
            evaluations: res.evaluations,
            status: OptimizerStatus::Lbfgsb(res.status),
            feasible: true,
        })
        .collect())
}

/// Optimizes the acquisition function from each of the initial conditions
/// (`q × d` each) with an independent L-BFGS-B run within `bounds`, keeping
/// `fixed_features` fixed (BoTorch's `gen_candidates_scipy` in its parallel
/// mode). The restarts run in parallel; results are returned in order.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
/// [`BotorchError::InvalidArgument`] for initial conditions outside the bounds
/// (beyond round-off) or invalid L-BFGS-B options, [`BotorchError::Numerical`]
/// for NaN gradients, and errors of the acquisition function.
pub fn gen_candidates_lbfgsb(
    initial_conditions: &[Matrix],
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    fixed_features: &BTreeMap<usize, f64>,
    options: &LbfgsbOptions,
) -> Result<Vec<RestartResult>> {
    let d = check_bounds(bounds)?;
    check_fixed_features(fixed_features, bounds)?;
    let free: Vec<usize> = (0..d).filter(|k| !fixed_features.contains_key(k)).collect();
    initial_conditions
        .par_iter()
        .map(|ic| optimize_one(ic, acq_function, bounds, &free, fixed_features, options))
        .collect()
}

fn optimize_one(
    ic: &Matrix,
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    free: &[usize],
    fixed: &BTreeMap<usize, f64>,
    options: &LbfgsbOptions,
) -> Result<RestartResult> {
    let d = bounds.ncols();
    if ic.ncols() != d || ic.nrows() == 0 {
        return Err(BotorchError::DimensionMismatch(format!(
            "initial conditions must be q x {d}, got {}x{}",
            ic.nrows(),
            ic.ncols()
        )));
    }
    let q = ic.nrows();
    let mut x0m = ic.clone();
    clamp_to_bounds(&mut x0m, bounds, true)?;
    let nf = free.len();
    let mut x0 = Vec::with_capacity(q * nf);
    let mut lower = Vec::with_capacity(q * nf);
    let mut upper = Vec::with_capacity(q * nf);
    for i in 0..q {
        for &k in free {
            x0.push(x0m[(i, k)]);
            lower.push(bounds[(0, k)]);
            upper.push(bounds[(1, k)]);
        }
    }
    let assemble = |x: &[f64]| -> Matrix {
        let mut m = Matrix::zeros(q, d);
        for i in 0..q {
            for (j, &k) in free.iter().enumerate() {
                m[(i, k)] = x[i * nf + j];
            }
        }
        apply_fixed_features(&mut m, fixed);
        m
    };
    let objective = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
        if x.iter().any(|v| v.is_nan()) {
            return Err(BotorchError::Numerical(
                "NaN in the optimization variables".into(),
            ));
        }
        let xm = assemble(x);
        let (v, g) = acq_function.forward_with_grad(&xm)?;
        let mut grad = Vec::with_capacity(q * nf);
        for i in 0..q {
            for &k in free {
                grad.push(-g[(i, k)]);
            }
        }
        if grad.iter().any(|v| v.is_nan()) {
            return Err(BotorchError::Numerical(format!(
                "{} elements of the {} element gradient are NaN; this often indicates \
                 numerical issues",
                grad.iter().filter(|v| v.is_nan()).count(),
                grad.len()
            )));
        }
        Ok((-v, grad))
    };
    let res = match lbfgsb::minimize(objective, &x0, &lower, &upper, options) {
        Ok(r) => r,
        Err(LbfgsbError::Objective(e)) => return Err(e),
        Err(LbfgsbError::InvalidInput(msg)) => return Err(BotorchError::InvalidArgument(msg)),
    };
    let mut candidates = assemble(&res.x);
    clamp_to_bounds(&mut candidates, bounds, true)?;
    let acq_value = acq_function.forward(&candidates)?;
    Ok(RestartResult {
        candidates,
        acq_value,
        iterations: res.iterations,
        evaluations: res.evaluations,
        status: OptimizerStatus::Lbfgsb(res.status),
        feasible: true,
    })
}
