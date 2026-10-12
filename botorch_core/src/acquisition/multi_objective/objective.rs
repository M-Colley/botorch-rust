//! Multi-output MC objectives (`botorch.acquisition.multi_objective.objective`):
//! [`IdentityMcMultiOutputObjective`], [`WeightedMcMultiOutputObjective`],
//! [`GenericMcMultiOutputObjective`] and
//! [`FeasibilityWeightedMcMultiOutputObjective`].
//!
//! A multi-output objective maps the posterior samples of `n` points and `m`
//! model outputs ([`McSamples`]) to `m'` objective values per point and
//! sample (again [`McSamples`], with `m'` outputs), all to be maximized. Like
//! the single-output objectives of [`crate::acquisition::mc_objective`], they
//! provide exact reverse-mode derivatives through a [`SamplePullback`] that
//! maps the adjoint of the objective values (`(m'·n) × S`, the layout of
//! [`McSamples`]) to the adjoint of the samples and, for objectives that
//! depend on `X`, of the inputs.

use std::fmt::Debug;
use std::sync::Arc;

use crate::acquisition::mc_objective::{q_batch_mismatch, McSamples, SamplePullback};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::posterior::MIN_VARIANCE;
use crate::utils::multi_objective::{clamp_min, nan_min};
use crate::utils::objective::{
    compute_smoothed_feasibility_indicator, McConstraint, OutcomeConstraints,
};
use crate::{Matrix, Vector};

/// The values of a multi-output objective (`m'` outputs per point and
/// sample), with a pullback if derivatives were requested.
pub struct MultiOutputValues {
    /// The objective values (`(m'·n) × S`, output-major).
    pub values: McSamples,
    /// Maps the adjoint of `values` to the adjoint of the samples (and of the
    /// inputs, for objectives that depend on `X`).
    pub pullback: Option<SamplePullback>,
}

impl Debug for MultiOutputValues {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiOutputValues")
            .field("values", &self.values)
            .field("pullback", &self.pullback.is_some())
            .finish()
    }
}

/// A multi-output MC objective (BoTorch's `MCMultiOutputObjective`): maps
/// posterior samples of `n` points and `m` outputs to `m'` objectives per
/// point and sample, all to be maximized.
pub trait McMultiOutputObjective: Send + Sync + Debug {
    /// The objective values of `samples` at the points `x` (`n × d`; `None`
    /// where BoTorch passes `X=None`); with `with_grad`, also the pullback.
    ///
    /// # Errors
    /// Implementation-specific (e.g. an outcome index out of range).
    fn evaluate(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues>;

    /// Whether the values at a point depend only on the samples at that point
    /// (and not on `X`). Defaults to `false`.
    fn is_pointwise(&self) -> bool {
        false
    }
}

/// Checks that an objective returned values for the `n` points of the
/// samples (BoTorch's q-batch check of `MCAcquisitionObjective.__call__`).
pub(crate) fn check_values(values: &McSamples, samples: &McSamples) -> Result<()> {
    if values.num_points() != samples.num_points() || values.num_samples() != samples.num_samples()
    {
        return Err(q_batch_mismatch(values.num_points(), samples.num_points()));
    }
    Ok(())
}

/// Normalizes possibly negative outcome indices (BoTorch's
/// `normalize_indices`).
fn normalize_indices(indices: &[i64], d: usize) -> Result<Vec<usize>> {
    indices
        .iter()
        .map(|&i| {
            let j = if i < 0 { i + d as i64 } else { i };
            if j < 0 || j > d as i64 - 1 {
                return Err(BotorchError::InvalidArgument(format!(
                    "Index {j} out of bounds for tensor or length {d}."
                )));
            }
            Ok(j as usize)
        })
        .collect()
}

/// Selects the outputs `outcomes` of `samples` (all outputs for `None`) and
/// scales them by `weights` (if given).
fn select_and_weight(
    samples: &McSamples,
    outcomes: Option<&[usize]>,
    weights: Option<&Vector>,
) -> Result<McSamples> {
    let (n, m) = (samples.num_points(), samples.num_outputs());
    let idx: Vec<usize> = match outcomes {
        Some(o) => {
            if let Some(&bad) = o.iter().find(|&&t| t >= m) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "index out of range in self: outcome {bad} of samples with {m} outputs"
                )));
            }
            o.to_vec()
        }
        None => (0..m).collect(),
    };
    if let Some(w) = weights {
        if w.len() != idx.len() {
            return Err(BotorchError::Runtime(format!(
                "The size of tensor a ({}) must match the size of tensor b ({}) at non-singleton \
                 dimension {}",
                idx.len(),
                w.len(),
                3
            )));
        }
    }
    let data = samples.data();
    let s = samples.num_samples();
    let mut out = Matrix::zeros(idx.len() * n, s);
    for (k, &t) in idx.iter().enumerate() {
        let scale = weights.map(|w| w[k]);
        for p in 0..n {
            for c in 0..s {
                let v = data[(t * n + p, c)];
                out[(k * n + p, c)] = match scale {
                    Some(w) => v * w,
                    None => v,
                };
            }
        }
    }
    McSamples::new(out, n, idx.len())
}

/// The pullback of [`select_and_weight`]: scatters the adjoint of the
/// selected (weighted) outputs back to all `m` outputs.
fn select_pullback(
    n: usize,
    m: usize,
    outcomes: Option<Vec<usize>>,
    weights: Option<Vector>,
) -> SamplePullback {
    Box::new(move |bar: &Matrix| {
        let s = bar.ncols();
        let idx: Vec<usize> = outcomes.clone().unwrap_or_else(|| (0..m).collect());
        let mut out = Matrix::zeros(m * n, s);
        for (k, &t) in idx.iter().enumerate() {
            let w = weights.as_ref().map_or(1.0, |w| w[k]);
            for p in 0..n {
                for c in 0..s {
                    out[(t * n + p, c)] += bar[(k * n + p, c)] * w;
                }
            }
        }
        Ok((out, None))
    })
}

/// The trivial objective (BoTorch's `IdentityMCMultiOutputObjective`): the
/// samples themselves, or the outputs `outcomes`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IdentityMcMultiOutputObjective {
    outcomes: Option<Vec<usize>>,
}

impl IdentityMcMultiOutputObjective {
    /// `IdentityMCMultiOutputObjective(outcomes, num_outcomes)`: `outcomes`
    /// (at least two; negative indices count from the end and need
    /// `num_outcomes`, the number of model outputs).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for fewer than two outcomes and
    /// [`BotorchError::InvalidArgument`] for negative indices without
    /// `num_outcomes` or indices out of range, with BoTorch's messages.
    pub fn new(outcomes: Option<&[i64]>, num_outcomes: Option<usize>) -> Result<Self> {
        let Some(outcomes) = outcomes else {
            return Ok(Self { outcomes: None });
        };
        if outcomes.len() < 2 {
            return Err(BotorchError::DimensionMismatch(
                "Must specify at least two outcomes for MOO.".into(),
            ));
        }
        let outcomes = if outcomes.iter().any(|&i| i < 0) {
            let Some(d) = num_outcomes else {
                return Err(BotorchError::InvalidArgument(
                    "num_outcomes is required if any outcomes are less than 0.".into(),
                ));
            };
            normalize_indices(outcomes, d)?
        } else {
            outcomes.iter().map(|&i| i as usize).collect()
        };
        Ok(Self {
            outcomes: Some(outcomes),
        })
    }

    /// The selected outcomes (`None`: all).
    pub fn outcomes(&self) -> Option<&[usize]> {
        self.outcomes.as_deref()
    }
}

impl McMultiOutputObjective for IdentityMcMultiOutputObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        let values = match &self.outcomes {
            None => samples.clone(),
            Some(o) => select_and_weight(samples, Some(o), None)?,
        };
        let pullback = with_grad.then(|| {
            select_pullback(
                samples.num_points(),
                samples.num_outputs(),
                self.outcomes.clone(),
                None,
            )
        });
        Ok(MultiOutputValues { values, pullback })
    }

    fn is_pointwise(&self) -> bool {
        true
    }
}

/// An objective that reweights the (selected) outputs, `samples[..., outcomes]
/// * weights` (BoTorch's `WeightedMCMultiOutputObjective`), e.g. weights of
/// `-1` to minimize outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMcMultiOutputObjective {
    identity: IdentityMcMultiOutputObjective,
    weights: Vector,
}

impl WeightedMcMultiOutputObjective {
    /// `WeightedMCMultiOutputObjective(weights, outcomes, num_outcomes)`.
    ///
    /// # Errors
    /// As [`IdentityMcMultiOutputObjective::new`], and
    /// [`BotorchError::DimensionMismatch`] if `outcomes` and `weights` have
    /// different lengths (BoTorch's message).
    pub fn new(
        weights: Vector,
        outcomes: Option<&[i64]>,
        num_outcomes: Option<usize>,
    ) -> Result<Self> {
        let identity = IdentityMcMultiOutputObjective::new(outcomes, num_outcomes)?;
        if let Some(o) = outcomes {
            if weights.len() != o.len() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "weights must contain the same number of elements as outcomes, but got {} \
                     weights and {} outcomes.",
                    weights.len(),
                    o.len()
                )));
            }
        }
        Ok(Self { identity, weights })
    }

    /// The weights.
    pub fn weights(&self) -> &Vector {
        &self.weights
    }

    /// The selected outcomes (`None`: all).
    pub fn outcomes(&self) -> Option<&[usize]> {
        self.identity.outcomes()
    }
}

impl McMultiOutputObjective for WeightedMcMultiOutputObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        let outcomes = self.identity.outcomes();
        let values = select_and_weight(samples, outcomes, Some(&self.weights))?;
        let pullback = with_grad.then(|| {
            select_pullback(
                samples.num_points(),
                samples.num_outputs(),
                outcomes.map(<[usize]>::to_vec),
                Some(self.weights.clone()),
            )
        });
        Ok(MultiOutputValues { values, pullback })
    }

    fn is_pointwise(&self) -> bool {
        true
    }
}

/// The values of a pointwise multi-output function at one point and sample
/// (`m'` values) and their Jacobian with respect to the `m` outputs
/// (row-major `m' × m`).
pub type MultiOutputFn = dyn Fn(&[f64]) -> (Vec<f64>, Vec<f64>) + Send + Sync;

/// A multi-output objective given by a function of the outputs at one point,
/// applied to every point and MC sample (BoTorch's
/// `GenericMCMultiOutputObjective` for callables that act pointwise, e.g.
/// `lambda Y, X: torch.stack([Y[..., 0], -Y[..., 1] ** 2], -1)`).
///
/// The closure returns the `m'` values and their Jacobian (row-major
/// `m' × m`). Objectives that couple points or depend on `X` implement
/// [`McMultiOutputObjective`] instead.
#[derive(Clone)]
pub struct GenericMcMultiOutputObjective {
    f: Arc<MultiOutputFn>,
}

impl Debug for GenericMcMultiOutputObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GenericMcMultiOutputObjective")
    }
}

impl GenericMcMultiOutputObjective {
    /// The objective `f(y)` with Jacobian (`f` returns `(values, jacobian)`).
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&[f64]) -> (Vec<f64>, Vec<f64>) + Send + Sync + 'static,
    {
        Self { f: Arc::new(f) }
    }
}

impl McMultiOutputObjective for GenericMcMultiOutputObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        let (n, m, s) = (
            samples.num_points(),
            samples.num_outputs(),
            samples.num_samples(),
        );
        let data = samples.data();
        let mut y = vec![0.0; m];
        let mut values: Option<Matrix> = None;
        let mut jac: Vec<f64> = Vec::new();
        let mut mp = 0;
        for c in 0..s {
            for p in 0..n {
                for (t, yt) in y.iter_mut().enumerate() {
                    *yt = data[(t * n + p, c)];
                }
                let (v, g) = (self.f)(&y);
                let out = values.get_or_insert_with(|| {
                    mp = v.len();
                    if with_grad {
                        jac = vec![0.0; mp * m * n * s];
                    }
                    Matrix::zeros(mp * n, s)
                });
                if v.len() != mp || (with_grad && g.len() != mp * m) {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "a multi-output objective of {m} outputs returned {} values and a \
                         Jacobian of {} entries (expected {mp} and {})",
                        v.len(),
                        g.len(),
                        mp * m
                    )));
                }
                for (k, vk) in v.iter().enumerate() {
                    out[(k * n + p, c)] = *vk;
                }
                if with_grad {
                    let start = (c * n + p) * mp * m;
                    jac[start..start + mp * m].copy_from_slice(&g);
                }
            }
        }
        let values = values.unwrap_or_else(|| Matrix::zeros(0, s));
        let values = McSamples::new(values, n, mp.max(1))?;
        let pullback: Option<SamplePullback> = with_grad.then(|| {
            Box::new(move |bar: &Matrix| {
                let mut out = Matrix::zeros(m * n, s);
                for c in 0..s {
                    for p in 0..n {
                        let start = (c * n + p) * mp * m;
                        for k in 0..mp {
                            let b = bar[(k * n + p, c)];
                            if b == 0.0 {
                                continue;
                            }
                            for t in 0..m {
                                out[(t * n + p, c)] += b * jac[start + k * m + t];
                            }
                        }
                    }
                }
                Ok((out, None))
            }) as SamplePullback
        });
        Ok(MultiOutputValues { values, pullback })
    }

    fn is_pointwise(&self) -> bool {
        true
    }
}

/// The temperature of the sigmoid of
/// [`FeasibilityWeightedMcMultiOutputObjective`]: BoTorch's default `eta =
/// 1e-3` of `apply_constraints`, which `compute_smoothed_feasibility_indicator`
/// materializes in float32 (`torch.full`), i.e. `1e-3` rounded to float32.
pub const FEASIBILITY_WEIGHTING_ETA: f64 = 1e-3_f32 as f64;

/// The constraint `-Y[..., index] <= 0` (feasible iff the output is
/// non-negative).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NonNegativeOutput(usize);

impl McConstraint for NonNegativeOutput {
    fn evaluate(
        &self,
        samples: &McSamples,
        with_grad: bool,
    ) -> Result<crate::acquisition::mc_objective::SampleValues> {
        let (n, m) = (samples.num_points(), samples.num_outputs());
        let t = self.0;
        if t >= m {
            return Err(BotorchError::DimensionMismatch(format!(
                "index {t} is out of bounds for dimension with size {m}"
            )));
        }
        let values = -samples.data().rows(t * n, n).into_owned();
        let pullback: Option<SamplePullback> = with_grad.then(|| {
            Box::new(move |bar: &Matrix| {
                let mut out = Matrix::zeros(m * n, bar.ncols());
                out.rows_mut(t * n, n).copy_from(&(-bar));
                Ok((out, None))
            }) as SamplePullback
        });
        Ok(crate::acquisition::mc_objective::SampleValues { values, pullback })
    }

    fn is_pointwise(&self) -> bool {
        true
    }
}

/// The infeasible cost of every output of a model at the points `x`
/// (BoTorch's `get_infeasible_cost(X, model, objective=lambda y, X: y)`):
/// `M_t = -min(0, min_x min(μ_t - 6σ_t, μ_t + 6σ_t))` over the marginal
/// posteriors.
///
/// # Errors
/// Errors of the model.
pub fn get_infeasible_cost_per_output(model: &dyn Model, x: &Matrix) -> Result<Vec<f64>> {
    let m = model.num_outputs();
    let pp = model.point_posteriors(x)?;
    let mut lb = vec![f64::INFINITY; m];
    for (t, l) in lb.iter_mut().enumerate() {
        for p in 0..x.nrows() {
            let var = pp.variance(p, t).max(MIN_VARIANCE).max(0.0);
            let six = 6.0 * var.sqrt();
            let mu = pp.mean[(p, t)];
            // torch.stack(...).min(dim=0) and min over the points (NaN propagates).
            *l = nan_min(*l, nan_min(mu - six, mu + six));
        }
    }
    Ok(lb.into_iter().map(|v| -(nan_min(v, 0.0))).collect())
}

/// Feasibility-weighted objective (BoTorch's
/// `FeasibilityWeightedMCMultiOutputObjective`): outputs `constraint_idcs`
/// of the model are constraints (feasible iff non-negative); the other
/// outputs are weighted by the sigmoid approximation of the feasibility of
/// the sample, with BoTorch's `apply_constraints`,
///
/// ```text
/// (max(Y[..., objective_idcs] + M, 0) · Π_i sigmoid(Y[..., i] / eta)) - M,
/// ```
///
/// where `M` is the infeasible cost of each objective output at
/// `X_baseline` and `eta` is [`FEASIBILITY_WEIGHTING_ETA`], before the
/// optional `objective` is applied. Without constraints the samples are
/// passed to the objective unchanged.
///
/// Every constraint uses its own output (the fix of a late-binding bug in
/// BoTorch, meta-pytorch/botorch#3392: BoTorch main evaluates the last
/// constraint index for every constraint).
#[derive(Clone, Debug)]
pub struct FeasibilityWeightedMcMultiOutputObjective {
    constraint_idcs: Vec<usize>,
    objective_idcs: Vec<usize>,
    infeasible_cost: Vec<f64>,
    constraints: Option<OutcomeConstraints>,
    objective: Option<Arc<dyn McMultiOutputObjective>>,
}

impl FeasibilityWeightedMcMultiOutputObjective {
    /// `FeasibilityWeightedMCMultiOutputObjective(model, X_baseline,
    /// constraint_idcs, objective)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for duplicate or out-of-range
    /// constraint indices (BoTorch's message), and errors of the model.
    pub fn new(
        model: &dyn Model,
        x_baseline: &Matrix,
        constraint_idcs: &[i64],
        objective: Option<Arc<dyn McMultiOutputObjective>>,
    ) -> Result<Self> {
        let num_outputs = model.num_outputs();
        let idcs: Vec<i64> = constraint_idcs
            .iter()
            .map(|&i| if i < 0 { num_outputs as i64 + i } else { i })
            .collect();
        let mut sorted = idcs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != idcs.len() {
            return Err(BotorchError::InvalidArgument(
                "Received duplicate entries for `constraint_idcs`.".into(),
            ));
        }
        if let Some(&bad) = idcs.iter().find(|&&i| i < 0 || i >= num_outputs as i64) {
            return Err(BotorchError::DimensionMismatch(format!(
                "index {bad} is out of bounds for dimension with size {num_outputs}"
            )));
        }
        let constraint_idcs: Vec<usize> = idcs.iter().map(|&i| i as usize).collect();
        let objective_idcs: Vec<usize> = (0..num_outputs)
            .filter(|i| !constraint_idcs.contains(i))
            .collect();
        let (infeasible_cost, constraints) = if constraint_idcs.is_empty() {
            (Vec::new(), None)
        } else {
            let all = get_infeasible_cost_per_output(model, x_baseline)?;
            let cost = objective_idcs.iter().map(|&i| all[i]).collect();
            let cons: Vec<Arc<dyn McConstraint>> = constraint_idcs
                .iter()
                .map(|&i| Arc::new(NonNegativeOutput(i)) as Arc<dyn McConstraint>)
                .collect();
            (
                cost,
                Some(OutcomeConstraints::uniform(
                    cons,
                    FEASIBILITY_WEIGHTING_ETA,
                )?),
            )
        };
        Ok(Self {
            constraint_idcs,
            objective_idcs,
            infeasible_cost,
            constraints,
            objective,
        })
    }

    /// The (non-negative) constraint indices.
    pub fn constraint_idcs(&self) -> &[usize] {
        &self.constraint_idcs
    }

    /// The indices of the objective outputs.
    pub fn objective_idcs(&self) -> &[usize] {
        &self.objective_idcs
    }

    /// The infeasible cost `M` of every objective output (empty without
    /// constraints).
    pub fn infeasible_cost(&self) -> &[f64] {
        &self.infeasible_cost
    }

    /// The feasibility-weighted objective outputs (BoTorch's
    /// `apply_feasibility_weights`).
    fn weighted(&self, samples: &McSamples, with_grad: bool) -> Result<MultiOutputValues> {
        let Some(constraints) = &self.constraints else {
            let pullback: Option<SamplePullback> =
                with_grad.then(|| Box::new(|bar: &Matrix| Ok((bar.clone(), None))) as _);
            return Ok(MultiOutputValues {
                values: samples.clone(),
                pullback,
            });
        };
        let (n, m, s) = (
            samples.num_points(),
            samples.num_outputs(),
            samples.num_samples(),
        );
        let fat = vec![Some(false); constraints.len()];
        let ind =
            compute_smoothed_feasibility_indicator(constraints, &fat, samples, false, with_grad)?;
        let k = self.objective_idcs.len();
        let data = samples.data();
        // obj.add(M).clamp_min(0).mul(w).add(-M), per objective output.
        let mut shifted = Matrix::zeros(k * n, s);
        let mut values = Matrix::zeros(k * n, s);
        for (a, &t) in self.objective_idcs.iter().enumerate() {
            let cost = self.infeasible_cost[a];
            for p in 0..n {
                for c in 0..s {
                    let sh = data[(t * n + p, c)] + cost;
                    shifted[(a * n + p, c)] = sh;
                    values[(a * n + p, c)] = clamp_min(sh, 0.0) * ind.values[(p, c)] + -cost;
                }
            }
        }
        let values = McSamples::new(values, n, k)?;
        let pullback: Option<SamplePullback> = if with_grad {
            let objective_idcs = self.objective_idcs.clone();
            let w = ind.values.clone();
            Some(Box::new(move |bar: &Matrix| {
                let mut out = Matrix::zeros(m * n, s);
                let mut w_bar = Matrix::zeros(n, s);
                for (a, &t) in objective_idcs.iter().enumerate() {
                    for p in 0..n {
                        for c in 0..s {
                            let b = bar[(a * n + p, c)];
                            let sh = shifted[(a * n + p, c)];
                            // clamp_min passes the gradient where sh >= 0.
                            if sh >= 0.0 {
                                out[(t * n + p, c)] += b * w[(p, c)];
                            }
                            w_bar[(p, c)] += b * clamp_min(sh, 0.0);
                        }
                    }
                }
                out += ind.backward(&w_bar)?;
                Ok((out, None))
            }))
        } else {
            None
        };
        Ok(MultiOutputValues { values, pullback })
    }
}

impl McMultiOutputObjective for FeasibilityWeightedMcMultiOutputObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<MultiOutputValues> {
        let weighted = self.weighted(samples, with_grad)?;
        let Some(objective) = &self.objective else {
            return Ok(weighted);
        };
        let out = objective.evaluate(&weighted.values, x, with_grad)?;
        check_values(&out.values, samples)?;
        let pullback: Option<SamplePullback> = match (weighted.pullback, out.pullback) {
            (Some(inner), Some(outer)) => Some(Box::new(move |bar: &Matrix| {
                let (mid, xbar) = outer(bar)?;
                let (sbar, _) = inner(&mid)?;
                Ok((sbar, xbar))
            })),
            _ => None,
        };
        Ok(MultiOutputValues {
            values: out.values,
            pullback,
        })
    }

    fn is_pointwise(&self) -> bool {
        self.objective.as_ref().is_none_or(|o| o.is_pointwise())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(n: usize, m: usize, s: usize) -> McSamples {
        let row_major: Vec<f64> = (0..s * n * m)
            .map(|i| ((i * 7) % 13) as f64 / 6.0 - 1.0)
            .collect();
        McSamples::from_row_major(&row_major, s, n, m).unwrap()
    }

    /// Central differences of `sum(bar ⊙ objective(samples))`.
    fn check_pullback(obj: &dyn McMultiOutputObjective, s: &McSamples) {
        let out = obj.evaluate(s, None, true).unwrap();
        let (k, n, ns) = (out.values.num_outputs(), s.num_points(), s.num_samples());
        let bar = Matrix::from_fn(k * n, ns, |r, c| 0.3 + 0.1 * r as f64 - 0.07 * c as f64);
        let (adj, _) = (out.pullback.unwrap())(&bar).unwrap();
        let rm = s.to_row_major();
        let h = 1e-6;
        let m = s.num_outputs();
        for i in 0..rm.len() {
            let f = |delta: f64| {
                let mut v = rm.clone();
                v[i] += delta;
                let sp = McSamples::from_row_major(&v, ns, n, m).unwrap();
                obj.evaluate(&sp, None, false)
                    .unwrap()
                    .values
                    .data()
                    .component_mul(&bar)
                    .sum()
            };
            let fd = (f(h) - f(-h)) / (2.0 * h);
            let (c, rest) = (i / (n * m), i % (n * m));
            let (p, t) = (rest / m, rest % m);
            let an = adj[(t * n + p, c)];
            assert!(
                (fd - an).abs() < 1e-6 * fd.abs().max(1.0),
                "{i}: {an} vs {fd}"
            );
        }
    }

    #[test]
    fn identity_and_weighted_objectives() {
        let s = samples(2, 3, 4);
        let id = IdentityMcMultiOutputObjective::new(None, None).unwrap();
        assert_eq!(id.evaluate(&s, None, false).unwrap().values, s);
        let sel = IdentityMcMultiOutputObjective::new(Some(&[2, -3]), Some(3)).unwrap();
        assert_eq!(sel.outcomes(), Some(&[2usize, 0][..]));
        let v = sel.evaluate(&s, None, false).unwrap().values;
        assert_eq!(v.value(1, 1, 0), s.value(1, 1, 2));
        assert_eq!(v.value(3, 0, 1), s.value(3, 0, 0));
        check_pullback(&sel, &s);
        let w = WeightedMcMultiOutputObjective::new(
            Vector::from_vec(vec![1.5, -2.0]),
            Some(&[1, 2]),
            None,
        )
        .unwrap();
        let v = w.evaluate(&s, None, false).unwrap().values;
        assert_eq!(v.value(2, 1, 1), -2.0 * s.value(2, 1, 2));
        check_pullback(&w, &s);
        // Errors with BoTorch's messages.
        let e = IdentityMcMultiOutputObjective::new(Some(&[0]), None).unwrap_err();
        assert!(e.to_string().contains("at least two outcomes"));
        let e = IdentityMcMultiOutputObjective::new(Some(&[0, -1]), None).unwrap_err();
        assert!(e.to_string().contains("num_outcomes is required"));
        let e = IdentityMcMultiOutputObjective::new(Some(&[0, -4]), Some(3)).unwrap_err();
        assert!(e.to_string().contains("out of bounds"));
        let e =
            WeightedMcMultiOutputObjective::new(Vector::from_vec(vec![1.0]), Some(&[0, 1]), None)
                .unwrap_err();
        assert!(e.to_string().contains("same number of elements"));
        let bad =
            WeightedMcMultiOutputObjective::new(Vector::from_vec(vec![1.0]), None, None).unwrap();
        assert!(bad.evaluate(&s, None, false).is_err());
    }

    #[test]
    fn generic_objective_jacobian() {
        let s = samples(3, 2, 5);
        let g = GenericMcMultiOutputObjective::new(|y| {
            (
                vec![y[0] * y[1], -y[1] * y[1], y[0]],
                vec![y[1], y[0], 0.0, -2.0 * y[1], 1.0, 0.0],
            )
        });
        let v = g.evaluate(&s, None, false).unwrap().values;
        assert_eq!(v.num_outputs(), 3);
        assert_eq!(v.value(4, 2, 0), s.value(4, 2, 0) * s.value(4, 2, 1));
        check_pullback(&g, &s);
    }
}
