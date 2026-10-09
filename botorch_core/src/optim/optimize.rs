//! Acquisition-function optimization mirroring `botorch.optim.optimize_acqf`
//! with the default `gen_candidates_scipy` (box bounds, no linear or nonlinear
//! constraints).
//!
//! * Initial conditions come from [`gen_batch_initial_conditions`]: `raw_samples`
//!   scrambled Sobol candidate sets, Boltzmann (or non-negative) selection of
//!   `num_restarts` starting points.
//! * Every restart is optimized by an independent L-BFGS-B run
//!   ([`crate::lbfgsb`], a port of L-BFGS-B 3.0 reproducing scipy's
//!   iterates) on the negated acquisition value with exact gradients. This is
//!   exactly BoTorch's default "parallel" path (`fmin_l_bfgs_b_batched`, used
//!   with SciPy 1.13–1.18), where each restart is a separate problem; the
//!   restarts run in parallel with rayon. The default options are those of
//!   `gen_candidates_scipy` on BoTorch's main branch: `maxiter = 2000`, no
//!   limit on function evaluations (the batched driver does not check
//!   `maxfun`) and SciPy's defaults otherwise (`maxcor = 10`,
//!   `ftol = 2.2e-9`, `gtol = 1e-5`, `maxls = 20`).
//! * The final candidates are clamped to the bounds and re-evaluated; the best
//!   restart is returned. If a restart terminates abnormally (BoTorch's
//!   `OptimizationWarning`), the optimization is repeated once with new
//!   initial conditions, as with BoTorch's `retry_on_optimization_warning`.
//! * `sequential = true` generates `q` candidates greedily, conditioning on the
//!   previous ones as pending points.

use std::collections::BTreeMap;

use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::lbfgsb::{self, LbfgsbError, LbfgsbOptions, LbfgsbStatus};
use crate::optim::initializers::{
    apply_fixed_features, gen_batch_initial_conditions, InitialConditionOptions, DEFAULT_ALPHA,
    DEFAULT_ETA,
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

/// Options of [`optimize_acqf`] (BoTorch's keyword arguments and `options`
/// dictionary entries that apply without constraints).
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizeAcqfOptions {
    /// Seed of the raw Sobol samples (`options["seed"]`); it also seeds the
    /// random selection of initial conditions, which BoTorch draws from
    /// torch's global RNG. `None` draws a random seed.
    pub seed: Option<u64>,
    /// L-BFGS-B settings of each restart (default [`default_lbfgsb_options`]).
    pub lbfgsb: LbfgsbOptions,
    /// Number of restarts optimized per chunk (`options["batch_limit"]`).
    /// Restarts are independent problems, so this only bounds the work in
    /// flight and does not change the result.
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
}

impl Default for OptimizeAcqfOptions {
    fn default() -> Self {
        Self {
            seed: None,
            lbfgsb: default_lbfgsb_options(),
            batch_limit: None,
            init_batch_limit: None,
            eta: DEFAULT_ETA,
            alpha: DEFAULT_ALPHA,
            nonnegative: false,
            fixed_features: BTreeMap::new(),
            batch_initial_conditions: None,
            sequential: false,
            retry_on_optimization_warning: true,
        }
    }
}

/// The result of one restart of [`gen_candidates_lbfgsb`].
#[derive(Clone, Debug, PartialEq)]
pub struct RestartResult {
    /// The optimized candidate set (`q × d`), clamped to the bounds.
    pub candidates: Matrix,
    /// The acquisition value at `candidates`.
    pub acq_value: f64,
    /// L-BFGS-B iterations.
    pub iterations: usize,
    /// Acquisition evaluations (with gradient).
    pub evaluations: usize,
    /// L-BFGS-B termination reason.
    pub status: LbfgsbStatus,
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
/// with `num_restarts` L-BFGS-B restarts initialized from `raw_samples` raw
/// samples, and returns the best `q × d` candidate set and its acquisition
/// value (BoTorch's `optimize_acqf(acq_function, bounds, q, num_restarts,
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
/// sequential optimization and diagnostic messages.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] / [`BotorchError::InvalidArgument`] for
/// invalid bounds, sizes, fixed features or initial conditions;
/// [`BotorchError::Unsupported`] for sequential optimization of acquisition
/// functions without pending-point support; [`BotorchError::Numerical`] for NaN
/// gradients; and errors of the acquisition function.
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
    let seed = options.seed.unwrap_or_else(random_seed);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x6f70_7469_6d69_7a65);
    let joint_options = OptimizeAcqfOptions {
        seed: Some(seed),
        ..options.clone()
    };

    // All features fixed: nothing to optimize.
    if options.fixed_features.len() == d {
        let mut x = Matrix::zeros(q, d);
        apply_fixed_features(&mut x, &options.fixed_features);
        let v = acq_function.forward(&x)?;
        return Ok(OptimizeAcqfResult {
            candidates: x,
            acq_value: v,
            acq_values: vec![v],
            restarts: Vec::new(),
            warnings: Vec::new(),
        });
    }

    if !(options.sequential && q > 1) {
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
            "`batch_initial_conditions` is not supported for sequential optimization".into(),
        ));
    }
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
    let provided = options.batch_initial_conditions.clone().unwrap_or_default();
    for ic in &provided {
        if ic.shape() != (q, d) {
            return Err(BotorchError::DimensionMismatch(format!(
                "batch_initial_conditions must be {q} x {d}, got {}x{}",
                ic.nrows(),
                ic.ncols()
            )));
        }
    }
    let required = num_restarts.saturating_sub(provided.len());
    let ic_options = InitialConditionOptions {
        seed: options.seed,
        eta: options.eta,
        alpha: options.alpha,
        nonnegative: options.nonnegative,
        batch_limit: options.init_batch_limit.or(options.batch_limit),
        fixed_features: options.fixed_features.clone(),
    };
    let mut warnings = Vec::new();
    let make_ics =
        |rng: &mut Xoshiro256PlusPlus, warnings: &mut Vec<String>| -> Result<Vec<Matrix>> {
            let mut ics = provided.clone();
            if required > 0 {
                if raw_samples == 0 {
                    return Err(BotorchError::InvalidArgument(
                        "Must specify `raw_samples` when `batch_initial_conditions` is None".into(),
                    ));
                }
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
                        "Unable to find non-zero acquisition function values - initial conditions \
                     are being selected randomly."
                            .into(),
                    );
                }
                ics.extend(generated);
            }
            Ok(ics)
        };
    let ics = make_ics(rng, &mut warnings)?;
    let mut restarts = run_restarts(acq_function, bounds, &ics, options)?;
    let abnormal = |rs: &[RestartResult]| {
        rs.iter()
            .any(|r| r.status == LbfgsbStatus::AbnormalLineSearch)
    };
    if abnormal(&restarts) && options.retry_on_optimization_warning {
        if required > 0 {
            warnings.push(
                "Optimization failed in L-BFGS-B (abnormal termination in the line search); \
                 trying again with a new set of initial conditions."
                    .into(),
            );
            let ics = make_ics(rng, &mut warnings)?;
            restarts = run_restarts(acq_function, bounds, &ics, options)?;
            if abnormal(&restarts) {
                warnings.push(
                    "Optimization failed on the second try, after generating a new set of \
                     initial conditions."
                        .into(),
                );
            }
        } else {
            warnings.push(
                "Optimization failed in L-BFGS-B (abnormal termination in the line search); \
                 not retried because all initial conditions were provided."
                    .into(),
            );
        }
    }
    let best = best_restart(&restarts)?;
    let r = &restarts[best];
    Ok(OptimizeAcqfResult {
        candidates: r.candidates.clone(),
        acq_value: r.acq_value,
        acq_values: vec![r.acq_value],
        restarts,
        warnings,
    })
}

/// Index of the restart with the largest (non-NaN) value.
fn best_restart(restarts: &[RestartResult]) -> Result<usize> {
    let mut best: Option<usize> = None;
    for (i, r) in restarts.iter().enumerate() {
        if r.acq_value.is_nan() {
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
) -> Result<Vec<RestartResult>> {
    let chunk = options.batch_limit.unwrap_or(ics.len()).max(1);
    let mut out = Vec::with_capacity(ics.len());
    for c in ics.chunks(chunk) {
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

/// `columnwise_clamp` with BoTorch's `raise_on_violation` check
/// (`torch.allclose` tolerances).
fn clamp_to_bounds(x: &mut Matrix, bounds: &Matrix) -> Result<()> {
    for i in 0..x.nrows() {
        for k in 0..x.ncols() {
            let (lo, hi) = (bounds[(0, k)], bounds[(1, k)]);
            let v = x[(i, k)];
            let c = v.max(lo).min(hi);
            if (v - c).abs() > 1e-8 + 1e-5 * c.abs() || v.is_nan() {
                return Err(BotorchError::InvalidArgument(format!(
                    "Original value(s) are out of bounds: x[{i}, {k}] = {v} not in [{lo}, {hi}]"
                )));
            }
            x[(i, k)] = c;
        }
    }
    Ok(())
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
    clamp_to_bounds(&mut x0m, bounds)?;
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
    clamp_to_bounds(&mut candidates, bounds)?;
    let acq_value = acq_function.forward(&candidates)?;
    Ok(RestartResult {
        candidates,
        acq_value,
        iterations: res.iterations,
        evaluations: res.evaluations,
        status: res.status,
    })
}
