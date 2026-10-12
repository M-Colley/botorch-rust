//! Cost-aware utilities (`botorch.acquisition.cost_aware`): the trade-off
//! between the improvement in utility of a candidate set (e.g. the samples
//! of the knowledge-gradient over its fantasies) and the cost of evaluating
//! it, used by cost-aware and multi-fidelity acquisition functions.
//!
//! * [`InverseCostWeightedUtility`] divides positive improvements by the
//!   cost of a [`CostModel`] (multiplies negative ones), or subtracts the log
//!   cost from log improvements;
//! * [`GenericCostAwareUtility`] wraps a function of the candidates and the
//!   improvements.
//!
//! Utilities are evaluated for one t-batch: a candidate set `x` (`q × d`)
//! and the `n_f` improvements (`deltas`) of its fantasies, and return `n_f`
//! utilities with a reverse pass to the improvements and the candidates.

use std::fmt::Debug;
use std::sync::Arc;

use crate::acquisition::mc_objective::{McObjective, McSamples};
use crate::error::{BotorchError, Result};
use crate::models::cost::CostModel;
use crate::models::BoolMatrix;
use crate::Matrix;

/// The reverse pass of a utility: maps the adjoint of the utilities (`n_f`)
/// to the adjoints of the improvements (`n_f`) and of the candidates
/// (`q × d`).
pub type UtilityPullback = Box<dyn FnOnce(&[f64]) -> Result<(Vec<f64>, Matrix)> + Send>;

/// The utilities of the improvements of a candidate set.
pub struct UtilityValues {
    /// One utility per improvement.
    pub values: Vec<f64>,
    /// The reverse pass, if requested.
    pub pullback: Option<UtilityPullback>,
}

/// A cost-aware utility (BoTorch's `CostAwareUtility`).
pub trait CostAwareUtility: Send + Sync + Debug {
    /// The cost-transformed utilities of the improvements `deltas` (`n_f`
    /// samples of the improvement at the candidate set `x`, `q × d`),
    /// BoTorch's `utility(X, deltas, sampler, X_evaluation_mask)` for one
    /// t-batch; `x_evaluation_mask` (`q × m`) says which outputs are
    /// evaluated at which candidate (decoupled evaluations).
    ///
    /// # Errors
    /// Implementation-specific.
    fn evaluate(
        &self,
        x: &Matrix,
        deltas: &[f64],
        x_evaluation_mask: Option<&BoolMatrix>,
        with_grad: bool,
    ) -> Result<UtilityValues>;

    /// BoTorch's `_log` attribute: `Some(true)` if the utility takes and
    /// returns log values, `None` if it has no such attribute
    /// (`GenericCostAwareUtility`).
    fn log(&self) -> Option<bool> {
        None
    }
}

/// BoTorch's `InverseCostWeightedUtility(cost_model, use_mean, cost_objective,
/// log)`: with the total cost `c` of the candidates (the costs of the points
/// summed over the `q`-batch, after the cost objective), a positive
/// improvement `δ` becomes `δ / c` and a non-positive one `δ · c`; with
/// `log`, the log improvement becomes `δ - log c`.
///
/// The costs are the cost model's values (the posterior mean, BoTorch's
/// `use_mean=True`). With `use_mean = false`, BoTorch samples the cost
/// model's posterior; this is supported for deterministic cost models,
/// whose samples equal their values.
#[derive(Debug, Clone)]
pub struct InverseCostWeightedUtility {
    cost_model: Arc<dyn CostModel>,
    use_mean: bool,
    cost_objective: Option<Arc<dyn McObjective>>,
    log: bool,
}

impl InverseCostWeightedUtility {
    /// `InverseCostWeightedUtility(cost_model, use_mean, cost_objective,
    /// log)`. Without `cost_objective`, the costs of a single-output cost
    /// model are used directly and those of a multi-output model are summed
    /// over the outputs (BoTorch's defaults).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for `use_mean = false` with a cost
    /// model that is not deterministic.
    pub fn new(
        cost_model: Arc<dyn CostModel>,
        use_mean: bool,
        cost_objective: Option<Arc<dyn McObjective>>,
        log: bool,
    ) -> Result<Self> {
        if !use_mean && !cost_model.is_deterministic() {
            return Err(BotorchError::Unsupported(
                "use_mean=False (posterior samples of the cost) is only supported for \
                 deterministic cost models"
                    .into(),
            ));
        }
        Ok(Self {
            cost_model,
            use_mean,
            cost_objective,
            log,
        })
    }

    /// The cost model.
    pub fn cost_model(&self) -> &Arc<dyn CostModel> {
        &self.cost_model
    }

    /// Whether the posterior mean of the cost is used.
    pub fn use_mean(&self) -> bool {
        self.use_mean
    }

    /// The total cost of the candidate set `x` and, if `with_grad`, its
    /// gradient with respect to `x`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for evaluation masks whose rows differ
    /// (BoTorch's `NotImplementedError`), [`BotorchError::InvalidArgument`]
    /// for non-positive costs, and errors of the cost model and objective.
    pub fn total_cost(
        &self,
        x: &Matrix,
        x_evaluation_mask: Option<&BoolMatrix>,
        with_grad: bool,
    ) -> Result<(f64, Option<Matrix>)> {
        let q = x.nrows();
        let full = self.cost_model.forward(x)?;
        let outputs: Vec<usize> = match x_evaluation_mask {
            Some(mask) => {
                if (1..mask.nrows()).any(|i| mask.row(i) != mask.row(0)) {
                    return Err(BotorchError::Unsupported(
                        "Currently, all candidates must be evaluated on the same outputs.".into(),
                    ));
                }
                (0..mask.ncols()).filter(|&t| mask[(0, t)]).collect()
            }
            None => (0..full.ncols()).collect(),
        };
        if let Some(&t) = outputs.iter().find(|&&t| t >= full.ncols()) {
            return Err(BotorchError::DimensionMismatch(format!(
                "index {t} is out of bounds for the {} outputs of the cost model",
                full.ncols()
            )));
        }
        let k = outputs.len();
        if k == 0 {
            return Err(BotorchError::InvalidArgument(
                "the evaluation mask selects no output of the cost model".into(),
            ));
        }
        let costs = Matrix::from_fn(q, k, |p, j| full[(p, outputs[j])]);
        // The cost objective: one value per point.
        let (per_point, objective_pullback) = match &self.cost_objective {
            Some(obj) => {
                let samples = McSamples::from_row_major(&row_major(&costs), 1, q, k)?;
                let v = obj.evaluate(&samples, None, with_grad)?;
                let values: Vec<f64> = (0..q).map(|p| v.values[(p, 0)]).collect();
                (values, v.pullback.map(|pb| (pb, k)))
            }
            None => {
                let values: Vec<f64> = if self.cost_model.num_outputs() == 1 {
                    (0..q).map(|p| costs[(p, 0)]).collect()
                } else {
                    (0..q).map(|p| costs.row(p).iter().sum()).collect()
                };
                (values, None)
            }
        };
        if !per_point.iter().all(|&c| c > 0.0) {
            return Err(BotorchError::InvalidArgument(
                "Costs must be strictly positive. Consider clamping cost_objective.".into(),
            ));
        }
        let total: f64 = per_point.iter().sum();
        if !with_grad {
            return Ok((total, None));
        }
        // Adjoint of the selected costs (q × k) for an adjoint of one.
        let costs_bar = match objective_pullback {
            Some((pb, _)) => {
                let (sbar, _) = pb(&Matrix::from_element(q, 1, 1.0))?;
                let s = McSamples::new(sbar, q, k)?;
                Matrix::from_fn(q, k, |p, j| s.value(0, p, j))
            }
            // The identity or the sum over the outputs.
            None => Matrix::from_element(q, k, 1.0),
        };
        let mut full_bar = Matrix::zeros(q, full.ncols());
        for (j, &t) in outputs.iter().enumerate() {
            for p in 0..q {
                full_bar[(p, t)] += costs_bar[(p, j)];
            }
        }
        Ok((total, Some(self.cost_model.backward(x, &full_bar)?)))
    }
}

/// Row-major data of a matrix.
fn row_major(m: &Matrix) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.len());
    for i in 0..m.nrows() {
        out.extend(m.row(i).iter());
    }
    out
}

impl CostAwareUtility for InverseCostWeightedUtility {
    fn evaluate(
        &self,
        x: &Matrix,
        deltas: &[f64],
        x_evaluation_mask: Option<&BoolMatrix>,
        with_grad: bool,
    ) -> Result<UtilityValues> {
        let (cost, cost_grad) = self.total_cost(x, x_evaluation_mask, with_grad)?;
        let log = self.log;
        let values: Vec<f64> = deltas
            .iter()
            .map(|&dl| {
                if log {
                    dl - cost.ln()
                } else if dl > 0.0 {
                    dl / cost
                } else {
                    dl * cost
                }
            })
            .collect();
        let pullback: Option<UtilityPullback> = match cost_grad {
            None => None,
            Some(unit) => {
                let deltas = deltas.to_vec();
                Some(Box::new(move |vbar: &[f64]| {
                    let mut dbar = Vec::with_capacity(deltas.len());
                    let mut cbar = 0.0;
                    for (&dl, &vb) in deltas.iter().zip(vbar) {
                        if log {
                            dbar.push(vb);
                            cbar -= vb / cost;
                        } else if dl > 0.0 {
                            dbar.push(vb / cost);
                            cbar -= vb * dl / (cost * cost);
                        } else {
                            dbar.push(vb * cost);
                            cbar += vb * dl;
                        }
                    }
                    Ok((dbar, unit * cbar))
                }))
            }
        };
        Ok(UtilityValues { values, pullback })
    }

    fn log(&self) -> Option<bool> {
        Some(self.log)
    }
}

/// The function of a [`GenericCostAwareUtility`]: maps the candidates
/// (`q × d`) and the improvements (`n_f`) to the utilities, with the reverse
/// pass if requested.
pub type CostUtilityFn = dyn Fn(&Matrix, &[f64], bool) -> Result<UtilityValues> + Send + Sync;

/// BoTorch's `GenericCostAwareUtility(cost)`: a callable of the candidates
/// and the improvements. It has no `_log` attribute (a log HVKG rejects it,
/// as in BoTorch).
#[derive(Clone)]
pub struct GenericCostAwareUtility {
    f: Arc<CostUtilityFn>,
}

impl Debug for GenericCostAwareUtility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GenericCostAwareUtility")
    }
}

impl GenericCostAwareUtility {
    /// A utility from a function of the candidates and the improvements.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Matrix, &[f64], bool) -> Result<UtilityValues> + Send + Sync + 'static,
    {
        Self { f: Arc::new(f) }
    }
}

impl CostAwareUtility for GenericCostAwareUtility {
    fn evaluate(
        &self,
        x: &Matrix,
        deltas: &[f64],
        _x_evaluation_mask: Option<&BoolMatrix>,
        with_grad: bool,
    ) -> Result<UtilityValues> {
        let out = (self.f)(x, deltas, with_grad)?;
        if out.values.len() != deltas.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "the cost-aware utility returned {} values for {} improvements",
                out.values.len(),
                deltas.len()
            )));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::cost::{AffineFidelityCostModel, FixedCostModel};
    use crate::Vector;

    #[test]
    fn inverse_cost_weighting_and_gradients() {
        let x = Matrix::from_row_slice(2, 2, &[0.1, 0.5, 0.3, 0.25]);
        let model = Arc::new(AffineFidelityCostModel::new(None, 0.2).unwrap());
        let u = InverseCostWeightedUtility::new(model, true, None, false).unwrap();
        let deltas = [0.3, -0.4];
        let out = u.evaluate(&x, &deltas, None, true).unwrap();
        let cost: f64 = 0.7 + 0.45;
        assert!((out.values[0] - 0.3 / cost).abs() < 1e-15);
        assert!((out.values[1] + 0.4 * cost).abs() < 1e-15);
        let vbar = [1.0, 2.0];
        let (dbar, xbar) = (out.pullback.unwrap())(&vbar).unwrap();
        assert!((dbar[0] - 1.0 / cost).abs() < 1e-15 && (dbar[1] - 2.0 * cost).abs() < 1e-15);
        // d/dc: -0.3/c² + 2·(-0.4); the cost depends on the last column.
        let dc = -0.3 / (cost * cost) - 0.8;
        assert!((xbar[(0, 1)] - dc).abs() < 1e-14 && xbar[(1, 0)] == 0.0);
        // Log utilities and a fixed cost per output with an evaluation mask.
        let fixed = Arc::new(FixedCostModel::new(Vector::from_vec(vec![1.0, 3.0])).unwrap());
        let u = InverseCostWeightedUtility::new(fixed.clone(), true, None, true).unwrap();
        let mask = BoolMatrix::from_row_slice(2, 2, &[false, true, false, true]);
        let out = u.evaluate(&x, &[0.5], Some(&mask), false).unwrap();
        assert!((out.values[0] - (0.5 - 6f64.ln())).abs() < 1e-15);
        let bad = BoolMatrix::from_row_slice(2, 2, &[true, true, false, true]);
        assert!(u.evaluate(&x, &[0.5], Some(&bad), false).is_err());
        let neg = Arc::new(FixedCostModel::new(Vector::from_vec(vec![-1.0])).unwrap());
        let u = InverseCostWeightedUtility::new(neg, false, None, false).unwrap();
        assert!(u.evaluate(&x, &[0.5], None, false).is_err());
    }
}
