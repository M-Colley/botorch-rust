//! `optimize_acqf` (`botorch.optim.optimize_acqf`) with multi-start L-BFGS-B.

use std::collections::BTreeMap;

use botorch_core::optim::{default_lbfgsb_options, optimize_acqf_detailed, OptimizeAcqfOptions};
use botorch_core::Matrix;
use pyo3::prelude::*;

use crate::acquisition::PyAcquisitionFunction;
use crate::arrays::{matrix_out, stack_out, Array, ArrayOut};
use crate::errors::{value_error, IntoPyResult};
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

/// The Python warning category of a message of `optimize_acqf_detailed`.
fn warning_category(message: &str) -> &'static str {
    if message.contains("initial conditions are being selected randomly") {
        "BadInitialCandidatesWarning"
    } else {
        "RuntimeWarning"
    }
}

/// Optimizes `acq_function` over the box `bounds` (`2 x d`) for `q` points
/// with `num_restarts` L-BFGS-B restarts started from the best of
/// `raw_samples` Sobol samples (Boltzmann selection).
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
    if batch_limit == Some(0) || init_batch_limit == Some(0) {
        return Err(value_error("batch_limit must be positive"));
    }
    let defaults = OptimizeAcqfOptions::default();
    let ics: Option<Vec<Matrix>> = batch_initial_conditions
        .map(|ic| ic.batches("batch_initial_conditions").map(|(_, mats)| mats))
        .transpose()?;
    let options = OptimizeAcqfOptions {
        seed,
        lbfgsb,
        batch_limit,
        init_batch_limit,
        eta: eta.unwrap_or(defaults.eta),
        alpha: alpha.unwrap_or(defaults.alpha),
        nonnegative,
        fixed_features: fixed_features.unwrap_or_default(),
        batch_initial_conditions: ics,
        sequential,
        retry_on_optimization_warning,
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
