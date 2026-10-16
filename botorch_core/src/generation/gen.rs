//! `gen_candidates_torch` and `get_best_candidates`
//! (`botorch.generation.gen`).

use std::collections::BTreeMap;
use std::time::Instant;

use rayon::prelude::*;

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::optim::adam::{Adam, AdamOptions};
use crate::optim::stopping::{ExpMaStoppingCriterion, StoppingCriterion};
use crate::Matrix;

/// BoTorch's default learning rate of `gen_candidates_torch`
/// (`optimizer_options.setdefault("lr", 0.025)`).
pub const DEFAULT_TORCH_LR: f64 = 0.025;

/// Settings of [`gen_candidates_torch`]: the Adam settings
/// (`options["optimizer_options"]`), the stopping criterion
/// (`options["stopping_criterion_options"]`, BoTorch's
/// `ExpMAStoppingCriterion` arguments) and the timeout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorchGenOptions {
    /// Adam settings (learning rate 0.025 by default).
    pub adam: AdamOptions,
    /// Maximum number of iterations of the stopping criterion (10000).
    pub maxiter: usize,
    /// Whether the stopping criterion assumes minimization (true; the loss is
    /// the negated sum of the acquisition values).
    pub minimize: bool,
    /// Window of the moving average (10).
    pub n_window: usize,
    /// Decay of the moving-average weights (1).
    pub eta: f64,
    /// Relative tolerance (1e-5).
    pub rel_tol: f64,
    /// Stop after this many seconds (`timeout_sec`).
    pub timeout_sec: Option<f64>,
}

impl Default for TorchGenOptions {
    fn default() -> Self {
        let c = ExpMaStoppingCriterion::default();
        Self {
            adam: AdamOptions {
                lr: DEFAULT_TORCH_LR,
                ..AdamOptions::default()
            },
            maxiter: c.maxiter,
            minimize: c.minimize,
            n_window: c.n_window,
            eta: 1.0,
            rel_tol: c.rel_tol,
            timeout_sec: None,
        }
    }
}

impl TorchGenOptions {
    /// The stopping criterion of these settings.
    pub fn stopping_criterion(&self) -> ExpMaStoppingCriterion {
        ExpMaStoppingCriterion::new(
            self.maxiter,
            self.minimize,
            self.n_window,
            self.eta,
            self.rel_tol,
        )
    }
}

/// The result of [`gen_candidates_torch`].
#[derive(Clone, Debug, PartialEq)]
pub struct TorchGenResult {
    /// The optimized candidate sets (`q × d` each), clamped to the bounds,
    /// with the fixed features set.
    pub candidates: Vec<Matrix>,
    /// Their acquisition values.
    pub acq_values: Vec<f64>,
    /// The number of optimizer steps.
    pub steps: usize,
    /// Whether the optimization stopped because of the timeout.
    pub timed_out: bool,
}

/// A callback invoked after every evaluation with the iteration (from 1),
/// the loss (the negated sum of the acquisition values) and the gradient of
/// the loss with respect to the free features of every candidate set
/// (BoTorch's `callback(i, loss, grad)`).
pub type TorchCallback<'a> = &'a mut dyn FnMut(usize, f64, &[Matrix]) -> Result<()>;

/// Index of the largest value with torch's `argmax` semantics: the first NaN
/// if there is one (NaN compares as the largest value), else the first
/// maximum. Returns 0 for an empty slice.
pub fn torch_argmax(values: &[f64]) -> usize {
    if let Some(i) = values.iter().position(|v| v.is_nan()) {
        return i;
    }
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// The candidate set with the largest value (BoTorch's
/// `get_best_candidates`; `torch.argmax` semantics, see [`torch_argmax`]).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if the lengths differ or are zero.
pub fn get_best_candidates(batch_candidates: &[Matrix], batch_values: &[f64]) -> Result<Matrix> {
    if batch_candidates.is_empty() || batch_candidates.len() != batch_values.len() {
        return Err(BotorchError::InvalidArgument(format!(
            "expected one value per candidate set, got {} candidate sets and {} values",
            batch_candidates.len(),
            batch_values.len()
        )));
    }
    Ok(batch_candidates[torch_argmax(batch_values)].clone())
}

/// Clamps `x` column-wise to `[lower, upper]` (`±inf` entries do not clamp).
fn clamp_columns(x: &Matrix, lower: &[f64], upper: &[f64]) -> Matrix {
    Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
        let v = x[(i, j)];
        // torch.clamp: max(min(v, upper), lower) with NaN propagation.
        if v.is_nan() {
            v
        } else {
            v.max(lower[j]).min(upper[j])
        }
    })
}

/// Optimizes the acquisition function from the initial conditions (`q × d`
/// each) with Adam, BoTorch's `gen_candidates_torch`:
///
/// * the candidates are clamped to `bounds` (`2 × d`, `±inf` for no bound)
///   and the fixed features removed from the optimization;
/// * every iteration evaluates `loss = -Σ_r acq(clamp(X_r))` (the restarts
///   in parallel) and its gradient with respect to the clamped candidates,
///   takes an Adam step on the unclamped variables and checks the
///   [`ExpMaStoppingCriterion`] on the loss (all restarts stop together, as
///   in BoTorch) and the timeout;
/// * the final candidates are clamped, the fixed features set, and the
///   acquisition function is evaluated.
///
/// Unlike BoTorch (meta-pytorch/botorch, see the documentation), fixed
/// features combined with bounds work: the free features are clamped to their
/// own bounds (BoTorch fails to broadcast the bounds of all `d` features).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
/// [`BotorchError::InvalidArgument`] for invalid bounds, fixed features or
/// Adam settings, [`BotorchError::Numerical`] for NaN gradients, and errors of
/// the acquisition function and the callback.
pub fn gen_candidates_torch(
    initial_conditions: &[Matrix],
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    fixed_features: &BTreeMap<usize, f64>,
    options: &TorchGenOptions,
    mut callback: Option<TorchCallback<'_>>,
) -> Result<TorchGenResult> {
    let start = Instant::now();
    let d = bounds.ncols();
    if bounds.nrows() != 2 {
        return Err(BotorchError::DimensionMismatch(format!(
            "bounds must be 2 x d, got {}x{}",
            bounds.nrows(),
            d
        )));
    }
    if initial_conditions.is_empty() {
        return Err(BotorchError::InvalidArgument(
            "at least one initial condition is required".into(),
        ));
    }
    let q = initial_conditions[0].nrows();
    for ic in initial_conditions {
        if ic.shape() != (q, d) || q == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "initial conditions must be {q} x {d}, got {}x{}",
                ic.nrows(),
                ic.ncols()
            )));
        }
    }
    for j in 0..d {
        if bounds[(0, j)] > bounds[(1, j)] {
            return Err(BotorchError::InvalidArgument(
                "Lower bounds cannot exceed upper bounds.".into(),
            ));
        }
    }
    if let Some(&k) = fixed_features.keys().find(|&&k| k >= d) {
        return Err(BotorchError::InvalidArgument(format!(
            "fixed feature index {k} out of range for d = {d}"
        )));
    }
    let free: Vec<usize> = (0..d).filter(|k| !fixed_features.contains_key(k)).collect();
    let nf = free.len();
    let lower: Vec<f64> = free.iter().map(|&k| bounds[(0, k)]).collect();
    let upper: Vec<f64> = free.iter().map(|&k| bounds[(1, k)]).collect();
    let full_lower: Vec<f64> = (0..d).map(|k| bounds[(0, k)]).collect();
    let full_upper: Vec<f64> = (0..d).map(|k| bounds[(1, k)]).collect();
    let b = initial_conditions.len();
    // The optimization variables: clamped initial conditions without the
    // fixed features, flattened restart by restart, point by point.
    let mut params = Vec::with_capacity(b * q * nf);
    for ic in initial_conditions {
        let c = clamp_columns(ic, &full_lower, &full_upper);
        for p in 0..q {
            for &k in &free {
                params.push(c[(p, k)]);
            }
        }
    }
    let assemble = |x: &[f64], r: usize| -> Matrix {
        let mut m = Matrix::zeros(q, d);
        for p in 0..q {
            for (j, &k) in free.iter().enumerate() {
                let v = x[(r * q + p) * nf + j];
                m[(p, k)] = if v.is_nan() {
                    v
                } else {
                    v.max(lower[j]).min(upper[j])
                };
            }
        }
        for (&k, &v) in fixed_features {
            for p in 0..q {
                m[(p, k)] = v;
            }
        }
        m
    };
    let mut adam = Adam::new(params.len(), options.adam)?;
    let mut criterion = options.stopping_criterion();
    let mut steps = 0;
    let mut timed_out = false;
    loop {
        steps += 1;
        let sets: Vec<Matrix> = (0..b).map(|r| assemble(&params, r)).collect();
        let evals = sets
            .par_iter()
            .map(|x| acq_function.forward_with_grad(x))
            .collect::<Result<Vec<_>>>()?;
        let mut total = 0.0;
        for (v, _) in &evals {
            total += v;
        }
        let loss = -total;
        let mut grad = Vec::with_capacity(params.len());
        for (_, g) in &evals {
            for p in 0..q {
                for &k in &free {
                    grad.push(-g[(p, k)]);
                }
            }
        }
        if grad.iter().any(|g| g.is_nan()) {
            return Err(BotorchError::Numerical(format!(
                "{} elements of the {} element gradient are NaN; this often indicates \
                 numerical issues",
                grad.iter().filter(|g| g.is_nan()).count(),
                grad.len()
            )));
        }
        if let Some(cb) = callback.as_mut() {
            let gm: Vec<Matrix> = (0..b)
                .map(|r| Matrix::from_row_slice(q, nf, &grad[r * q * nf..(r + 1) * q * nf]))
                .collect();
            cb(steps, loss, &gm)?;
        }
        adam.step(&mut params, &grad);
        let mut stop = criterion.evaluate(&[loss]);
        if let Some(t) = options.timeout_sec {
            if start.elapsed().as_secs_f64() > t {
                stop = true;
                timed_out = true;
            }
        }
        if stop {
            break;
        }
    }
    let candidates: Vec<Matrix> = (0..b).map(|r| assemble(&params, r)).collect();
    let acq_values = acq_function.forward_many(&candidates)?;
    Ok(TorchGenResult {
        candidates,
        acq_values,
        steps,
        timed_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_semantics() {
        assert_eq!(torch_argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(torch_argmax(&[1.0, f64::NAN, 3.0, f64::NAN]), 1);
        assert_eq!(torch_argmax(&[f64::NEG_INFINITY, f64::NEG_INFINITY]), 0);
        let c = vec![Matrix::zeros(1, 2), Matrix::from_element(1, 2, 1.0)];
        assert_eq!(get_best_candidates(&c, &[0.0, 1.0]).unwrap()[(0, 0)], 1.0);
        assert!(get_best_candidates(&c, &[0.0]).is_err());
    }
}
