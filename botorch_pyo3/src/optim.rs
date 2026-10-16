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

use botorch_core::generation::TorchGenOptions;
use botorch_core::optim::{
    default_lbfgsb_options, evaluate_feasibility, gen_batch_initial_conditions,
    gen_batch_initial_conditions_with_fantasies, optimize_acqf_detailed,
    sample_q_batches_from_polytope, AdamOptions, CandidateGenerator, IcGenerator,
    InitialConditionOptions, LinearConstraint, NonlinearInequalityConstraint,
    OneShotInitialConditionOptions, OptimizationMethod, OptimizeAcqfOptions, OptimizeAcqfResult,
    ParameterConstraints, SampleAroundBest, SlsqpOptions, TopnOptions, CONSTRAINT_TOLERANCE,
};
use botorch_core::Matrix;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use crate::acquisition::PyAcquisitionFunction;
use crate::arrays::{matrix_out, stack_out, Array, ArrayBuf, ArrayOut, BoolBuf};
use crate::callbacks::PyNonlinearConstraint;
use crate::errors::{dimension_error, value_error, IntoPyResult};
use crate::threads;

/// `(candidates (q x d), acquisition values, all restart candidates
/// (R x q x d), all restart values, warnings as (category, message))`.
pub type OptimizeOut = (
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
pub fn parameter_constraints(
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
        let options = o.initial_condition_options();
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

/// Reads an optional keyword argument (`None` values count as missing).
pub fn kw<'py, T>(kwargs: Option<&Bound<'py, PyDict>>, key: &str) -> PyResult<Option<T>>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    match kwargs.map(|k| k.get_item(key)).transpose()?.flatten() {
        Some(v) if !v.is_none() => {
            Ok(Some(v.extract().map_err(|e| {
                value_error(format!("invalid value for `{key}`: {e}"))
            })?))
        }
        _ => Ok(None),
    }
}

/// Reads an optional keyword argument holding a Python object.
pub fn kw_obj(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<Py<PyAny>>> {
    match kwargs.map(|k| k.get_item(key)).transpose()?.flatten() {
        Some(v) if !v.is_none() => Ok(Some(v.unbind())),
        _ => Ok(None),
    }
}

/// The keyword arguments understood by [`acqf_options`].
const OPTION_KEYS: &[&str] = &[
    "seed",
    "maxiter",
    "maxfun",
    "ftol",
    "gtol",
    "maxcor",
    "maxls",
    "batch_limit",
    "init_batch_limit",
    "eta",
    "alpha",
    "nonnegative",
    "fixed_features",
    "batch_initial_conditions",
    "sequential",
    "retry_on_optimization_warning",
    "inequality_constraints",
    "equality_constraints",
    "nonlinear_inequality_constraints",
    "ic_generator",
    "native_ic_generator",
    "method",
    "n_burnin",
    "n_thinning",
    "frac_random",
    "num_inner_restarts",
    "raw_inner_samples",
    "one_shot_eta",
    "post_processing",
    "timeout_sec",
    "lbfgsb_aggregation",
    "gen_candidates",
    "lr",
    "betas",
    "eps",
    "weight_decay",
    "amsgrad",
    "maximize",
    "stop_maxiter",
    "stop_minimize",
    "n_window",
    "stop_eta",
    "rel_tol",
    "topn",
    "largest",
    "sorted",
    "sample_around_best",
    "sample_around_best_sigma",
    "sample_around_best_subset_sigma",
    "sample_around_best_prob_perturb",
];

/// The options of `optimize_acqf` from keyword arguments (BoTorch's
/// `options` entries and arguments, prepared by the Python layer).
pub fn acqf_options(d: usize, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<OptimizeAcqfOptions> {
    if let Some(k) = kwargs {
        for key in k.keys() {
            let key: String = key.extract()?;
            if !OPTION_KEYS.contains(&key.as_str()) {
                return Err(value_error(format!("unknown optimize_acqf option `{key}`")));
            }
        }
    }
    let maxiter: Option<usize> = kw(kwargs, "maxiter")?;
    let ftol: Option<f64> = kw(kwargs, "ftol")?;
    let mut lbfgsb = default_lbfgsb_options();
    lbfgsb.maxiter = maxiter.unwrap_or(lbfgsb.maxiter);
    lbfgsb.maxfun = kw(kwargs, "maxfun")?.unwrap_or(lbfgsb.maxfun);
    lbfgsb.ftol = ftol.unwrap_or(lbfgsb.ftol);
    lbfgsb.gtol = kw(kwargs, "gtol")?.unwrap_or(lbfgsb.gtol);
    lbfgsb.maxcor = kw(kwargs, "maxcor")?.unwrap_or(lbfgsb.maxcor);
    lbfgsb.maxls = kw(kwargs, "maxls")?.unwrap_or(lbfgsb.maxls);
    let slsqp_defaults = SlsqpOptions::default();
    let slsqp = SlsqpOptions {
        maxiter: maxiter.unwrap_or(slsqp_defaults.maxiter),
        ftol: ftol.unwrap_or(slsqp_defaults.ftol),
    };
    let batch_limit: Option<usize> = kw(kwargs, "batch_limit")?;
    let init_batch_limit: Option<usize> = kw(kwargs, "init_batch_limit")?;
    if batch_limit == Some(0) || init_batch_limit == Some(0) {
        return Err(value_error("batch_limit must be positive"));
    }
    let method: Option<String> = kw(kwargs, "method")?;
    let method = match method.as_deref() {
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
    let ics: Option<Array> = kw(kwargs, "batch_initial_conditions")?;
    let ics: Option<Vec<Matrix>> = ics
        .map(|ic| ic.batches("batch_initial_conditions").map(|(_, mats)| mats))
        .transpose()?;
    let constraints = parameter_constraints(
        kw(kwargs, "inequality_constraints")?,
        kw(kwargs, "equality_constraints")?,
        kw(kwargs, "nonlinear_inequality_constraints")?,
        d,
    )?;
    let ic_gen = kw_obj(kwargs, "ic_generator")?;
    let native: bool = kw(kwargs, "native_ic_generator")?.unwrap_or(false);
    let ic_generator = match (ic_gen, native) {
        (_, true) => Some(self::native_ic_generator()),
        (Some(f), false) => Some(crate::callbacks::ic_generator(f)),
        (None, false) => None,
    };
    let post_processing = kw_obj(kwargs, "post_processing")?;
    let generator = kwargs
        .map(|k| k.get_item("gen_candidates"))
        .transpose()?
        .flatten();
    let gen_candidates = match generator {
        None => CandidateGenerator::Scipy,
        Some(v) if v.is_none() => CandidateGenerator::Scipy,
        Some(v) => {
            if let Ok(name) = v.extract::<String>() {
                match name.as_str() {
                    "scipy" => CandidateGenerator::Scipy,
                    "torch" => CandidateGenerator::Torch(torch_gen_options(kwargs)?),
                    other => {
                        return Err(value_error(format!(
                            "unknown candidate generator {other:?}"
                        )))
                    }
                }
            } else {
                CandidateGenerator::Custom(crate::optim_ext::custom_generator(v.unbind()))
            }
        }
    };
    let topn = if kw(kwargs, "topn")?.unwrap_or(false) {
        Some(TopnOptions {
            largest: kw(kwargs, "largest")?.unwrap_or(true),
            sorted: kw(kwargs, "sorted")?.unwrap_or(true),
        })
    } else {
        None
    };
    let best: Option<Array> = kw(kwargs, "sample_around_best")?;
    let sample_around_best = match best {
        Some(b) => {
            let mut s = SampleAroundBest::new(b.matrix("best_X")?);
            s.sigma = kw(kwargs, "sample_around_best_sigma")?.unwrap_or(s.sigma);
            s.subset_sigma =
                kw(kwargs, "sample_around_best_subset_sigma")?.unwrap_or(s.subset_sigma);
            s.prob_perturb = kw(kwargs, "sample_around_best_prob_perturb")?;
            Some(s)
        }
        None => None,
    };
    Ok(OptimizeAcqfOptions {
        seed: kw(kwargs, "seed")?,
        lbfgsb,
        slsqp,
        method,
        batch_limit,
        init_batch_limit,
        eta: kw(kwargs, "eta")?.unwrap_or(defaults.eta),
        alpha: kw(kwargs, "alpha")?.unwrap_or(defaults.alpha),
        nonnegative: kw(kwargs, "nonnegative")?.unwrap_or(false),
        fixed_features: kw::<BTreeMap<usize, f64>>(kwargs, "fixed_features")?.unwrap_or_default(),
        batch_initial_conditions: ics,
        sequential: kw(kwargs, "sequential")?.unwrap_or(false),
        retry_on_optimization_warning: kw(kwargs, "retry_on_optimization_warning")?.unwrap_or(true),
        constraints,
        n_burnin: kw(kwargs, "n_burnin")?.unwrap_or(defaults.n_burnin),
        n_thinning: kw(kwargs, "n_thinning")?.unwrap_or(defaults.n_thinning),
        ic_generator,
        one_shot: OneShotInitialConditionOptions {
            frac_random: kw(kwargs, "frac_random")?.unwrap_or(defaults.one_shot.frac_random),
            num_inner_restarts: kw(kwargs, "num_inner_restarts")?
                .unwrap_or(defaults.one_shot.num_inner_restarts),
            raw_inner_samples: kw(kwargs, "raw_inner_samples")?
                .unwrap_or(defaults.one_shot.raw_inner_samples),
            eta: kw(kwargs, "one_shot_eta")?,
        },
        post_processing: post_processing.map(crate::optim_ext::py_post_processing),
        timeout_sec: kw(kwargs, "timeout_sec")?,
        lbfgsb_aggregation: kw(kwargs, "lbfgsb_aggregation")?,
        gen_candidates,
        topn,
        sample_around_best,
    })
}

/// The Adam and stopping settings of `gen_candidates_torch` from keyword
/// arguments (`lr`, `betas`, `eps`, `weight_decay`, `amsgrad`, `maximize`,
/// `stop_maxiter`, `stop_minimize`, `n_window`, `stop_eta`, `rel_tol`,
/// `timeout_sec`).
///
/// # Errors
/// For values of the wrong type.
pub fn torch_gen_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<TorchGenOptions> {
    let d = TorchGenOptions::default();
    Ok(TorchGenOptions {
        adam: AdamOptions {
            lr: kw(kwargs, "lr")?.unwrap_or(d.adam.lr),
            betas: kw(kwargs, "betas")?.unwrap_or(d.adam.betas),
            eps: kw(kwargs, "eps")?.unwrap_or(d.adam.eps),
            weight_decay: kw(kwargs, "weight_decay")?.unwrap_or(d.adam.weight_decay),
            amsgrad: kw(kwargs, "amsgrad")?.unwrap_or(d.adam.amsgrad),
            maximize: kw(kwargs, "maximize")?.unwrap_or(d.adam.maximize),
        },
        maxiter: kw(kwargs, "stop_maxiter")?.unwrap_or(d.maxiter),
        minimize: kw(kwargs, "stop_minimize")?.unwrap_or(d.minimize),
        n_window: kw(kwargs, "n_window")?.unwrap_or(d.n_window),
        eta: kw(kwargs, "stop_eta")?.unwrap_or(d.eta),
        rel_tol: kw(kwargs, "rel_tol")?.unwrap_or(d.rel_tol),
        timeout_sec: kw(kwargs, "timeout_sec")?,
    })
}

/// The output of an optimization: candidates, values, all restarts and
/// warnings.
pub fn optimize_out(result: OptimizeAcqfResult, q: usize, d: usize) -> OptimizeOut {
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
    (
        matrix_out(&result.candidates),
        result.acq_values,
        stack_out(&restart_mats, &[restart_mats.len()], rq, d),
        restart_values,
        warnings,
    )
}

/// Optimizes `acq_function` over the box `bounds` (`2 x d`) for `q` points
/// with `num_restarts` restarts started from the best of `raw_samples` raw
/// samples (Boltzmann selection), with L-BFGS-B or, with parameter
/// constraints, SLSQP (`method`: `"L-BFGS-B"` or `"SLSQP"` overrides the
/// choice); see [`acqf_options`] for the keyword arguments.
#[pyfunction]
#[pyo3(signature = (acq_function, bounds, q, num_restarts, raw_samples=None, **kwargs))]
pub fn optimize_acqf(
    py: Python<'_>,
    acq_function: PyRef<'_, PyAcquisitionFunction>,
    bounds: Array,
    q: usize,
    num_restarts: usize,
    raw_samples: Option<usize>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<OptimizeOut> {
    let bounds = bounds.matrix("bounds")?;
    let d = bounds.ncols();
    let options = acqf_options(d, kwargs)?;
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
    Ok(optimize_out(result, q, d))
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
    fixed_x_fantasies=None,
    topn=false,
    largest=true,
    sorted=true,
    sample_around_best=None,
    sample_around_best_sigma=None,
    sample_around_best_subset_sigma=None,
    sample_around_best_prob_perturb=None,
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
    fixed_x_fantasies: Option<Array>,
    topn: bool,
    largest: bool,
    sorted: bool,
    sample_around_best: Option<Array>,
    sample_around_best_sigma: Option<f64>,
    sample_around_best_subset_sigma: Option<f64>,
    sample_around_best_prob_perturb: Option<f64>,
) -> PyResult<(ArrayOut, bool)> {
    let bounds = bounds.matrix("bounds")?;
    let d = bounds.ncols();
    let fantasies = fixed_x_fantasies
        .map(|f| f.matrix("fixed_X_fantasies"))
        .transpose()?;
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
        topn: topn.then_some(TopnOptions { largest, sorted }),
        sample_around_best: match sample_around_best {
            Some(b) => {
                let mut s = SampleAroundBest::new(b.matrix("best_X")?);
                s.sigma = sample_around_best_sigma.unwrap_or(s.sigma);
                s.subset_sigma = sample_around_best_subset_sigma.unwrap_or(s.subset_sigma);
                s.prob_perturb = sample_around_best_prob_perturb;
                Some(s)
            }
            None => None,
        },
    };
    let acqf = &acq_function.inner;
    let (ics, fallback) = threads::run(py, || {
        let seed = options
            .seed
            .unwrap_or_else(botorch_core::utils::sampling::random_seed);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x6f70_7469_6d69_7a65);
        gen_batch_initial_conditions_with_fantasies(
            acqf.as_ref(),
            &bounds,
            q,
            num_restarts,
            raw_samples,
            &options,
            fantasies.as_ref(),
            &mut rng,
        )
    })
    .into_py_result()?;
    let rows = ics.first().map_or(q, |m| m.nrows());
    Ok((stack_out(&ics, &[ics.len()], rows, d), fallback))
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
) -> PyResult<BoolBuf> {
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
    Ok(BoolBuf(feasible, batch))
}
