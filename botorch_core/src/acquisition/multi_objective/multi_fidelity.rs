//! Multi-objective multi-fidelity optimization
//! (`botorch.acquisition.multi_objective.multi_fidelity`): [`Momf`] (Irshad,
//! Karsch and Döpp, *Expected hypervolume improvement for simultaneous
//! multi-objective and multi-fidelity optimization*, 2021).
//!
//! MOMF is qEHVI over the objectives and a "trust" objective (the fidelity,
//! the last output of the model's training data), weighted by the inverse
//! cost of the candidates (BoTorch's `InverseCostWeightedUtility` with
//! `use_mean=True`): `HVI / C` for a positive improvement and `HVI · C`
//! otherwise, where `C` is the total cost of the candidates and the pending
//! points. The cost comes from a [`CostCall`]; BoTorch's default is an
//! `AffineFidelityCostModel` with `C(s) = 1 + s` for the fidelity `s` in the
//! last input, provided here by [`affine_fidelity_cost`].

use std::fmt::Debug;
use std::sync::Arc;

use crate::acquisition::multi_objective::hypervolume::HypercellBounds;
use crate::acquisition::multi_objective::monte_carlo::{cells_of, EhviCore};
use crate::acquisition::multi_objective::objective::McMultiOutputObjective;
use crate::acquisition::multi_objective::Reduction;
use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::multi_objective::box_decompositions::BoxDecomposition;
use crate::utils::objective::OutcomeConstraints;
use crate::Matrix;

/// The cost of evaluating points (BoTorch's `cost_call` of MOMF, a
/// deterministic cost model): one cost per point.
pub trait CostCall: Send + Sync + Debug {
    /// The costs of the rows of `x` (`n × d`) and, with `with_grad`, the
    /// gradient of every cost with respect to its point (`n × d`).
    ///
    /// # Errors
    /// Implementation-specific.
    fn cost(&self, x: &Matrix, with_grad: bool) -> Result<(Vec<f64>, Option<Matrix>)>;
}

/// The affine fidelity cost `fixed_cost + Σ_j weights[j] · x[fidelity_dims[j]]`
/// per point: what BoTorch's `AffineFidelityCostModel` computes.
#[derive(Clone, Debug, PartialEq)]
struct AffineFidelityCost {
    /// `(input index, weight)` sorted by index (negative indices count from
    /// the end, as BoTorch's `sorted(fidelity_weights)`).
    terms: Vec<(i64, f64)>,
    fixed_cost: f64,
}

impl CostCall for AffineFidelityCost {
    fn cost(&self, x: &Matrix, with_grad: bool) -> Result<(Vec<f64>, Option<Matrix>)> {
        let (n, d) = x.shape();
        let cols = self
            .terms
            .iter()
            .map(|&(i, w)| {
                let j = if i < 0 { i + d as i64 } else { i };
                if j < 0 || j >= d as i64 {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "index {i} is out of bounds for dimension with size {d}"
                    )));
                }
                Ok((j as usize, w))
            })
            .collect::<Result<Vec<_>>>()?;
        let costs = (0..n)
            .map(|p| {
                let lin: f64 = cols.iter().map(|&(j, w)| x[(p, j)] * w).sum();
                self.fixed_cost + lin
            })
            .collect();
        let grad = with_grad.then(|| {
            let mut g = Matrix::zeros(n, d);
            for p in 0..n {
                for &(j, w) in &cols {
                    g[(p, j)] += w;
                }
            }
            g
        });
        Ok((costs, grad))
    }
}

/// The affine fidelity cost of BoTorch's `AffineFidelityCostModel(
/// fidelity_weights, fixed_cost)`: `fixed_cost + Σ weight · x[index]` for the
/// `(input index, weight)` pairs (negative indices count from the end).
/// MOMF's default is `affine_fidelity_cost(&[(-1, 1.0)], 1.0)`.
///
/// This stands in for BoTorch's cost models, which are ported separately;
/// the weights are used as exact `f64` values (BoTorch stores them in
/// float32).
pub fn affine_fidelity_cost(fidelity_weights: &[(i64, f64)], fixed_cost: f64) -> Arc<dyn CostCall> {
    let mut terms = fidelity_weights.to_vec();
    terms.sort_by_key(|&(i, _)| i);
    Arc::new(AffineFidelityCost { terms, fixed_cost })
}

/// Options of [`Momf`] (BoTorch's keyword arguments).
#[derive(Clone, Debug, Default)]
pub struct MomfOptions {
    /// The sampler (default: 128 Sobol samples with a random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// The multi-output objective (default: the identity on all outputs).
    pub objective: Option<Arc<dyn McMultiOutputObjective>>,
    /// Outcome constraints (feasible iff `<= 0`) with their temperatures.
    pub constraints: Option<OutcomeConstraints>,
    /// Pending points concatenated to every candidate set (their cost
    /// counts too).
    pub x_pending: Option<Matrix>,
    /// The cost of the candidates (BoTorch's `cost_call`); default: the
    /// affine cost `1 + s` of the fidelity `s` in the last input.
    pub cost_call: Option<Arc<dyn CostCall>>,
}

/// BoTorch's `MOMF`: the cost-weighted expected hypervolume improvement
/// (see the [module documentation](self)).
#[derive(Debug)]
pub struct Momf {
    core: EhviCore,
    cost: Arc<dyn CostCall>,
}

impl Momf {
    /// `MOMF(model, ref_point, partitioning, sampler, objective, constraints,
    /// eta, X_pending, cost_call)`; the reference point and the partitioning
    /// include the trust objective.
    ///
    /// # Errors
    /// As [`crate::acquisition::multi_objective::QExpectedHypervolumeImprovement::new`].
    pub fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        partitioning: &dyn BoxDecomposition,
        options: MomfOptions,
    ) -> Result<Self> {
        Self::from_cell_bounds(model, ref_point, cells_of(partitioning)?, options)
    }

    /// Like [`Momf::new`] with the hypercell bounds of the decomposition.
    ///
    /// # Errors
    /// See [`Momf::new`].
    pub fn from_cell_bounds(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        cells: HypercellBounds,
        options: MomfOptions,
    ) -> Result<Self> {
        let core = EhviCore::new(
            model,
            ref_point,
            cells,
            options.sampler,
            options.objective,
            options.constraints,
            options.x_pending,
            false,
            false,
        )?;
        Ok(Self {
            core,
            cost: options
                .cost_call
                .unwrap_or_else(|| affine_fidelity_cost(&[(-1, 1.0)], 1.0)),
        })
    }

    /// The reference point.
    pub fn ref_point(&self) -> &[f64] {
        &self.core.ref_point
    }

    /// The hypercell bounds.
    pub fn cell_bounds(&self) -> &HypercellBounds {
        &self.core.cells
    }

    /// The sampler.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.core.mc.sampler
    }

    /// The cost of the candidates.
    pub fn cost_call(&self) -> &Arc<dyn CostCall> {
        &self.cost
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let (hv, hv_grad) = self.core.evaluate(x, Reduction::Linear, with_grad)?;
        // The cost of the candidates and the pending points
        // (`concatenate_pending_points` before `cost_aware_utility`).
        let x_full = match self.core.x_pending() {
            Some(p) => crate::acquisition::monte_carlo::vstack(&[x, &p], x.ncols()),
            None => x.clone(),
        };
        let (costs, cost_grad) = self.cost.cost(&x_full, with_grad)?;
        if costs.len() != x_full.nrows() {
            return Err(BotorchError::DimensionMismatch(format!(
                "the cost function returned {} costs for {} points",
                costs.len(),
                x_full.nrows()
            )));
        }
        if !costs.iter().all(|&c| c > 0.0) {
            return Err(BotorchError::InvalidArgument(
                "Costs must be strictly positive. Consider clamping cost_objective.".into(),
            ));
        }
        let c: f64 = costs.iter().sum();
        let positive = hv > 0.0;
        let value = if positive { hv / c } else { hv * c };
        let Some(hv_grad) = hv_grad else {
            return Ok((value, None));
        };
        let q = x.nrows();
        let cg = cost_grad.ok_or_else(|| {
            BotorchError::Runtime("the cost function returned no gradient".into())
        })?;
        if cg.shape() != x_full.shape() {
            return Err(BotorchError::DimensionMismatch(format!(
                "the cost gradient has shape {}x{}, expected {}x{}",
                cg.nrows(),
                cg.ncols(),
                x_full.nrows(),
                x_full.ncols()
            )));
        }
        let cg = cg.rows(0, q);
        let grad = if positive {
            hv_grad / c - cg * (hv / (c * c))
        } else {
            hv_grad * c + cg * hv
        };
        Ok((value, Some(grad)))
    }
}

impl AcquisitionFunction for Momf {
    fn model(&self) -> &Arc<dyn Model> {
        &self.core.mc.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.core.x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.core.set_x_pending(x_pending)
    }

    fn is_nonnegative(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_cost() {
        let c = affine_fidelity_cost(&[(-1, 2.0), (0, 0.5)], 1.0);
        let x = Matrix::from_row_slice(2, 3, &[0.2, 0.0, 0.5, 1.0, 0.0, 0.25]);
        let (costs, g) = c.cost(&x, true).unwrap();
        assert_eq!(costs, vec![1.0 + 0.2 * 0.5 + 0.5 * 2.0, 1.0 + 0.5 + 0.5]);
        let g = g.unwrap();
        assert_eq!(
            g.row(0).iter().copied().collect::<Vec<_>>(),
            vec![0.5, 0.0, 2.0]
        );
        assert!(affine_fidelity_cost(&[(3, 1.0)], 1.0)
            .cost(&x, false)
            .is_err());
    }
}
