//! `optimize_acqf` (`botorch.optim.optimize_acqf`) with multi-start L-BFGS-B
//! or, with parameter constraints, SLSQP; `gen_batch_initial_conditions`,
//! `sample_q_batches_from_polytope` and `evaluate_feasibility`.
//!
//! Linear constraints arrive as BoTorch's `(indices, coefficients, rhs)`
//! tuples with the indices converted to (nested) lists: one-dimensional for
//! intra-point constraints, `(point, feature)` pairs for inter-point
//! constraints. Nonlinear constraints are `(callable, is_intrapoint)` pairs
//! evaluated through [`crate::callbacks`].

use std::collections::BTreeMap;
use std::sync::Arc;

use botorch_core::optim::{
    default_lbfgsb_options, evaluate_feasibility, gen_batch_initial_conditions,
    optimize_acqf_detailed, sample_q_batches_from_polytope, IcGenerator, InitialConditionOptions,
    LinearConstraint, NonlinearInequalityConstraint, OptimizationMethod, OptimizeAcqfOptions,
    ParameterConstraints, SlsqpOptions, CONSTRAINT_TOLERANCE,
};
use botorch_core::Matrix;
use pyo3::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use crate::acquisition::PyAcquisitionFunction;
use crate::arrays::{matrix_out, stack_out, Array, ArrayBuf, ArrayOut};
use crate::callbacks::PyNonlinearConstraint;
use crate::errors::{dimension_error, value_error, IntoPyResult};
use crate::threads;

/// `(candidates (q x d), acquisition values, all restart candidates
/// (R x q x d), all restart values, warnings as (category, message))`.
type OptimizeOut = (
    ArrayOut,
    Vec<f64>,
    ArrayOut,
    Vec<f64>,
    Vec<(&'static str, String)>,
);

/// A linear constraint received from Python: `(indices, coefficients, rhs)`.
pub type LinearArg<'py> = (Bound<'py, PyAny>, Vec<f64>, f64);

/// The Python warning category of a message of `optimize_acqf_detailed`.
fn warning_category(message: &str) -> &'static str {
    if message.contains("initial conditions are being selected randomly") {
        "BadInitialCandidatesWarning"
    } else {
        "RuntimeWarning"
    }
}

/// A linear constraint from BoTorch's tuple (indices as nested lists).
pub fn linear_constraint(c: &LinearArg<'_>) -> PyResult<LinearConstraint> {
    let (indices, coefficients, rhs) = c;
    let constraint = if let Ok(idx) = indices.extract::<Vec<usize>>() {
        LinearConstraint::intra_point(idx, coefficients.clone(), *rhs)
    } else if let Ok(pairs) = indices.extract::<Vec<(usize, usize)>>() {
        LinearConstraint::inter_point(pairs, coefficients.clone(), *rhs)
    } else if let Ok(rows) = indices.extract::<Vec<Vec<usize>>>() {
        if rows.iter().any(|r| r.len() != 2) {
            return Err(dimension_error(
                "Linear constraints supported only on individual candidates and across \
                 q-batches, not across general batch shapes.",
            ));
        }
        LinearConstraint::inter_point(
            rows.iter().map(|r| (r[0], r[1])).collect(),
            coefficients.clone(),
            *rhs,
        )
    } else {
        return Err(value_error(
            "constraint indices must be a 1-d or 2-d tensor of non-negative integers",
        ));
    };
    constraint.into_py_result()
}

/// Linear constraints from BoTorch's tuples.
pub fn linear_constraints(cs: Option<Vec<LinearArg<'_>>>) -> PyResult<Vec<LinearConstraint>> {
    cs.unwrap_or_default()
        .iter()
        .map(linear_constraint)
        .collect()
}

/// The parameter constraints of an optimization.
fn parameter_constraints(
    inequality: Option<Vec<LinearArg<'_>>>,
    equality: Option<Vec<LinearArg<'_>>>,
    nonlinear: Option<Vec<(Py<PyAny>, bool)>>,
    d: usize,
) -> PyResult<ParameterConstraints> {
    Ok(ParameterConstraints {
        inequality: linear_constraints(inequality)?,
        equality: linear_constraints(equality)?,
        nonlinear: nonlinear
            .unwrap_or_default()
            .into_iter()
            .map(|(f, intra)| {
                Arc::new(PyNonlinearConstraint::new(f, intra, d))
                    as Arc<dyn NonlinearInequalityConstraint>
            })
            .collect(),
    })
}

/// BoTorch's `gen_batch_initial_conditions` as an initial-condition
/// generator (what `ic_generator=gen_batch_initial_conditions` selects).
fn native_ic_generator() -> IcGenerator {
    IcGenerator::new(|req| {
        let o = req.options;
        let options = InitialConditionOptions {
            seed: o.seed,
            eta: o.eta,
            alpha: o.alpha,
            nonnegative: o.nonnegative,
            batch_limit: o.init_batch_limit.or(o.batch_limit),
            fixed_features: o.fixed_features.clone(),
            inequality_constraints: o.constraints.inequality.clone(),
            equality_constraints: o.constraints.equality.clone(),
            n_burnin: o.n_burnin,
            n_thinning: o.n_thinning,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(o.seed.unwrap_or(0) ^ 0x6963_5f67_656e);
        Ok(gen_batch_initial_conditions(
            req.acq_function,
            req.bounds,
            req.q,
            req.num_restarts,
            req.raw_samples,
            &options,
            &mut rng,
        )?
        .0)
    })
}

/// Optimizes `acq_function` over the box `bounds` (`2 x d`) for `q` points
/// with `num_restarts` restarts started from the best of `raw_samples` raw
/// samples (Boltzmann selection), with L-BFGS-B or, with parameter
/// constraints, SLSQP (`method`: `"L-BFGS-B"` or `"SLSQP"` overrides the
/// choice).
#[pyfunction]
#[pyo3(signature = (
    acq_function,
    bounds,
    q,
    num_restarts,
    raw_samples=None,
    *,
    seed=None,
    maxiter=None,
    maxfun=None,
    ftol=None,
    gtol=None,
    maxcor=None,
    maxls=None,
    batch_limit=None,
    init_batch_limit=None,
    eta=None,
    alpha=None,
    nonnegative=false,
    fixed_features=None,
    batch_initial_conditions=None,
    sequential=false,
    retry_on_optimization_warning=true,
    inequality_constraints=None,
    equality_constraints=None,
    nonlinear_inequality_constraints=None,
    ic_generator=None,
    native_ic_generator=false,
    method=None,
    n_burnin=None,
    n_thinning=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn optimize_acqf(
    py: Python<'_>,
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    bounds: Array,
    q: usize,
    num_restarts: usize,
    raw_samples: Option<usize>,
    seed: Option<u64>,
    maxiter: Option<usize>,
    maxfun: Option<usize>,
    ftol: Option<f64>,
    gtol: Option<f64>,
    maxcor: Option<usize>,
    maxls: Option<usize>,
    batch_limit: Option<usize>,
    init_batch_limit: Option<usize>,
    eta: Option<f64>,
    alpha: Option<f64>,
    nonnegative: bool,
    fixed_features: Option<BTreeMap<usize, f64>>,
    batch_initial_conditions: Option<Array>,
    sequential: bool,
    retry_on_optimization_warning: bool,
    inequality_constraints: Option<Vec<LinearArg<'_>>>,
    equality_constraints: Option<Vec<LinearArg<'_>>>,
    nonlinear_inequality_constraints: Option<Vec<(Py<PyAny>, bool)>>,
    ic_generator: Option<Py<PyAny>>,
    native_ic_generator: bool,
    method: Option<&str>,
    n_burnin: Option<usize>,
    n_thinning: Option<usize>,
) -> PyResult<OptimizeOut> {
    let bounds = bounds.matrix("bounds")?;
    let d = bounds.ncols();
    let mut lbfgsb = default_lbfgsb_options();
    lbfgsb.maxiter = maxiter.unwrap_or(lbfgsb.maxiter);
    lbfgsb.maxfun = maxfun.unwrap_or(lbfgsb.maxfun);
    lbfgsb.ftol = ftol.unwrap_or(lbfgsb.ftol);
    lbfgsb.gtol = gtol.unwrap_or(lbfgsb.gtol);
    lbfgsb.maxcor = maxcor.unwrap_or(lbfgsb.maxcor);
    lbfgsb.maxls = maxls.unwrap_or(lbfgsb.maxls);
    let slsqp_defaults = SlsqpOptions::default();
    let slsqp_options = SlsqpOptions {
        maxiter: maxiter.unwrap_or(slsqp_defaults.maxiter),
        ftol: ftol.unwrap_or(slsqp_defaults.ftol),
    };
    if batch_limit == Some(0) || init_batch_limit == Some(0) {
        return Err(value_error("batch_limit must be positive"));
    }
    let method = match method {
        None => OptimizationMethod::Auto,
        Some("SLSQP") => OptimizationMethod::Slsqp,
        Some("L-BFGS-B") => OptimizationMethod::Lbfgsb,
        Some(other) => {
            return Err(crate::errors::unsupported(format!(
                "Only method='L-BFGS-B' and method='SLSQP' are supported, got {other:?}."
            )))
        }
    };
    let defaults = OptimizeAcqfOptions::default();
    let ics: Option<Vec<Matrix>> = batch_initial_conditions
        .map(|ic| ic.batches("batch_initial_conditions").map(|(_, mats)| mats))
        .transpose()?;
    let constraints = parameter_constraints(
        inequality_constraints,
        equality_constraints,
        nonlinear_inequality_constraints,
        d,
    )?;
    let ic_generator = match (ic_generator, native_ic_generator) {
        (_, true) => Some(self::native_ic_generator()),
        (Some(f), false) => Some(crate::callbacks::ic_generator(f)),
        (None, false) => None,
    };
    let options = OptimizeAcqfOptions {
        seed,
        lbfgsb,
        slsqp: slsqp_options,
        method,
        batch_limit,
        init_batch_limit,
        eta: eta.unwrap_or(defaults.eta),
        alpha: alpha.unwrap_or(defaults.alpha),
        nonnegative,
        fixed_features: fixed_features.unwrap_or_default(),
        batch_initial_conditions: ics,
        sequential,
        retry_on_optimization_warning,
        constraints,
        n_burnin: n_burnin.unwrap_or(defaults.n_burnin),
        n_thinning: n_thinning.unwrap_or(defaults.n_thinning),
        ic_generator,
    };
    let acqf = &acq_function.inner;
    let result = threads::run(py, || {
        optimize_acqf_detailed(
            acqf.as_ref(),
            &bounds,
            q,
            num_restarts,
            raw_samples.unwrap_or(0),
            &options,
        )
    })
    .into_py_result()?;
    let rq = result.restarts.first().map_or(q, |r| r.candidates.nrows());
    let restart_mats: Vec<Matrix> = result
        .restarts
        .iter()
        .map(|r| r.candidates.clone())
        .collect();
    let restart_values = result.restarts.iter().map(|r| r.acq_value).collect();
    let warnings = result
        .warnings
        .into_iter()
        .map(|w| (warning_category(&w), w))
        .collect();
    Ok((
        matrix_out(&result.candidates),
        result.acq_values,
        stack_out(&restart_mats, &[restart_mats.len()], rq, d),
        restart_values,
        warnings,
    ))
}

/// BoTorch's `gen_batch_initial_conditions`: `num_restarts` initial
/// candidate sets (`num_restarts x q x d`) selected from `raw_samples` Sobol
/// (or, with linear constraints, polytope) samples.
#[pyfunction]
#[pyo3(name = "gen_batch_initial_conditions", signature = (
    acq_function,
    bounds,
    q,
    num_restarts,
    raw_samples,
    *,
    fixed_features=None,
    seed=None,
    eta=None,
    alpha=None,
    nonnegative=false,
    batch_limit=None,
    inequality_constraints=None,
    equality_constraints=None,
    n_burnin=None,
    n_thinning=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn gen_batch_initial_conditions_py(
    py: Python<'_>,
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    bounds: Array,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    fixed_features: Option<BTreeMap<usize, f64>>,
    seed: Option<u64>,
    eta: Option<f64>,
    alpha: Option<f64>,
    nonnegative: bool,
    batch_limit: Option<usize>,
    inequality_constraints: Option<Vec<LinearArg<'_>>>,
    equality_constraints: Option<Vec<LinearArg<'_>>>,
    n_burnin: Option<usize>,
    n_thinning: Option<usize>,
) -> PyResult<(ArrayOut, bool)> {
    let bounds = bounds.matrix("bounds")?;
    let d = bounds.ncols();
    let defaults = InitialConditionOptions::default();
    let options = InitialConditionOptions {
        seed,
        eta: eta.unwrap_or(defaults.eta),
        alpha: alpha.unwrap_or(defaults.alpha),
        nonnegative,
        batch_limit,
        fixed_features: fixed_features.unwrap_or_default(),
        inequality_constraints: linear_constraints(inequality_constraints)?,
        equality_constraints: linear_constraints(equality_constraints)?,
        n_burnin: n_burnin.unwrap_or(defaults.n_burnin),
        n_thinning: n_thinning.unwrap_or(defaults.n_thinning),
    };
    let acqf = &acq_function.inner;
    let (ics, fallback) = threads::run(py, || {
        let seed = options
            .seed
            .unwrap_or_else(botorch_core::utils::sampling::random_seed);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x6f70_7469_6d69_7a65);
        gen_batch_initial_conditions(
            acqf.as_ref(),
            &bounds,
            q,
            num_restarts,
            raw_samples,
            &options,
            &mut rng,
        )
    })
    .into_py_result()?;
    Ok((stack_out(&ics, &[ics.len()], q, d), fallback))
}

/// BoTorch's `sample_q_batches_from_polytope`: `n x q x d` samples of the
/// polytope given by `bounds` and linear constraints.
#[pyfunction]
#[pyo3(name = "sample_q_batches_from_polytope", signature = (
    n,
    q,
    bounds,
    n_burnin,
    n_thinning,
    seed=None,
    inequality_constraints=None,
    equality_constraints=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn sample_q_batches_from_polytope_py(
    py: Python<'_>,
    n: usize,
    q: usize,
    bounds: Array,
    n_burnin: usize,
    n_thinning: usize,
    seed: Option<u64>,
    inequality_constraints: Option<Vec<LinearArg<'_>>>,
    equality_constraints: Option<Vec<LinearArg<'_>>>,
) -> PyResult<ArrayBuf> {
    let bounds = bounds.matrix("bounds")?;
    let d = bounds.ncols();
    let ineq = linear_constraints(inequality_constraints)?;
    let eq = linear_constraints(equality_constraints)?;
    let batches = threads::run(py, || {
        sample_q_batches_from_polytope(n, q, &bounds, n_burnin, n_thinning, seed, &ineq, &eq)
    })
    .into_py_result()?;
    Ok(stack_out(&batches, &[n], q, d).into())
}

/// BoTorch's `evaluate_feasibility`: whether each candidate set of `X`
/// (`batch x q x d`) satisfies the constraints within `tolerance`.
#[pyfunction]
#[pyo3(name = "evaluate_feasibility", signature = (
    x,
    inequality_constraints=None,
    equality_constraints=None,
    nonlinear_inequality_constraints=None,
    tolerance=None,
))]
pub fn evaluate_feasibility_py(
    py: Python<'_>,
    x: Array,
    inequality_constraints: Option<Vec<LinearArg<'_>>>,
    equality_constraints: Option<Vec<LinearArg<'_>>>,
    nonlinear_inequality_constraints: Option<Vec<(Py<PyAny>, bool)>>,
    tolerance: Option<f64>,
) -> PyResult<(Vec<bool>, Vec<usize>)> {
    let (batch, sets) = x.batches("X")?;
    let d = x.1.last().copied().unwrap_or(0);
    let constraints = parameter_constraints(
        inequality_constraints,
        equality_constraints,
        nonlinear_inequality_constraints,
        d,
    )?;
    let tol = tolerance.unwrap_or(CONSTRAINT_TOLERANCE);
    let feasible = threads::run(py, || {
        sets.iter()
            .map(|s| evaluate_feasibility(s, &constraints, tol))
            .collect::<botorch_core::Result<Vec<bool>>>()
    })
    .into_py_result()?;
    Ok((feasible, batch))
}
