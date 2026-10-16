//! Cyclic optimization of `q` candidates (BoTorch's `optimize_acqf_cyclic`).

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::optim::optimize::{optimize_acqf_detailed, OptimizeAcqfOptions};
use crate::optim::stopping::{ExpMaStoppingCriterion, StoppingCriterion};
use crate::Matrix;

/// The result of [`optimize_acqf_cyclic`].
#[derive(Clone, Debug, PartialEq)]
pub struct CyclicResult {
    /// The candidates (`q × d`).
    pub candidates: Matrix,
    /// The acquisition value of every candidate conditional on all the
    /// others (for `q = 1`, its value).
    pub acq_values: Vec<f64>,
    /// The number of completed cycles after the initial sequential pass.
    pub cycles: usize,
    /// Messages of BoTorch's warnings.
    pub warnings: Vec<String>,
}

/// The warning BoTorch emits for every re-optimization of a candidate (a
/// 2-dim `batch_initial_conditions` with `raw_samples` set).
pub const TWO_DIM_IC_WARNING: &str = "If using a 2-dim `batch_initial_conditions` botorch will \
     default to old behavior of ignoring `num_restarts` and just use the given \
     `batch_initial_conditions` by setting `raw_samples` to None.";

/// Generates `q` candidates by cyclic optimization (BoTorch's
/// `optimize_acqf_cyclic`): the candidates are first generated sequentially
/// (`optimize_acqf` with `sequential = true`); then, until `stopping`
/// (BoTorch's `ExpMAStoppingCriterion(**cyclic_options)`, evaluated on the
/// `q` values after every cycle) holds, every candidate in turn is
/// re-optimized from its current value with all other candidates (and the
/// original pending points) as pending points.
///
/// `options.batch_initial_conditions` must not be set for `q > 1` (as in
/// BoTorch, sequential optimization does not support them).
///
/// # Errors
/// The errors of [`optimize_acqf_detailed`].
#[allow(clippy::needless_range_loop)]
pub fn optimize_acqf_cyclic(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    options: &OptimizeAcqfOptions,
    stopping: &ExpMaStoppingCriterion,
) -> Result<CyclicResult> {
    if !options.constraints.nonlinear.is_empty() {
        return Err(BotorchError::Unsupported(
            "`optimize_acqf_cyclic` does not support nonlinear inequality constraints.".into(),
        ));
    }
    let first_options = OptimizeAcqfOptions {
        sequential: true,
        ..options.clone()
    };
    let first = optimize_acqf_detailed(
        acq_function,
        bounds,
        q,
        num_restarts,
        raw_samples,
        &first_options,
    )?;
    let mut candidates = first.candidates;
    let mut values = first.acq_values;
    let mut warnings = first.warnings;
    let mut cycles = 0;
    if q > 1 {
        let d = bounds.ncols();
        let mut criterion = stopping.clone();
        criterion.reset();
        let mut stop = criterion.evaluate(&values);
        let base = acq_function.x_pending();
        let result = (|| -> Result<()> {
            while !stop {
                for i in 0..q {
                    let n0 = base.as_ref().map_or(0, |b| b.nrows());
                    let others: Vec<usize> = (0..q).filter(|&j| j != i).collect();
                    let pending = Matrix::from_fn(n0 + q - 1, d, |r, c| {
                        if r < n0 {
                            base.as_ref().expect("base")[(r, c)]
                        } else {
                            candidates[(others[r - n0], c)]
                        }
                    });
                    acq_function.set_x_pending(Some(&pending))?;
                    let step_options = OptimizeAcqfOptions {
                        sequential: false,
                        batch_initial_conditions: Some(vec![candidates.rows(i, 1).into_owned()]),
                        ..options.clone()
                    };
                    warnings.push(TWO_DIM_IC_WARNING.to_string());
                    let step = optimize_acqf_detailed(
                        acq_function,
                        bounds,
                        1,
                        num_restarts,
                        0,
                        &step_options,
                    )?;
                    candidates.row_mut(i).copy_from(&step.candidates.row(0));
                    values[i] = step.acq_value;
                    warnings.extend(step.warnings);
                }
                cycles += 1;
                stop = criterion.evaluate(&values);
            }
            Ok(())
        })();
        acq_function.set_x_pending(base.as_ref())?;
        result?;
    }
    Ok(CyclicResult {
        candidates,
        acq_values: values,
        cycles,
        warnings,
    })
}
