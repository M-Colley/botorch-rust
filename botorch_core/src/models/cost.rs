//! Cost models for cost-aware and multi-fidelity optimization
//! (`botorch.models.cost`): [`AffineFidelityCostModel`] and
//! [`FixedCostModel`], deterministic models of the cost of evaluating a
//! candidate, and [`PosteriorMeanCostModel`], the posterior mean of a fitted
//! model (what `InverseCostWeightedUtility(cost_model, use_mean=True)` uses
//! for a GP cost model).
//!
//! A [`CostModel`] maps the points of a candidate set `x` (`q × d`, one per
//! row) to their costs (`q × m`), BoTorch's `cost_model(X)` (and
//! `cost_model.posterior(X).mean`) for `X` of shape `1 × q × d`, with the
//! reverse-mode derivative with respect to `x`.

use std::fmt::Debug;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::{Matrix, Vector};

/// A model of the cost of evaluating the points of a candidate set.
pub trait CostModel: Send + Sync + Debug {
    /// Number of outputs (cost components) `m`.
    fn num_outputs(&self) -> usize;

    /// The costs of the points of `x` (`q × d`): a `q × m` matrix (BoTorch's
    /// `forward`, or the posterior mean).
    ///
    /// # Errors
    /// Implementation-specific (dimension mismatches).
    fn forward(&self, x: &Matrix) -> Result<Matrix>;

    /// Reverse mode: the gradient with respect to `x` of `Σ cost_bar ⊙ costs`
    /// (`cost_bar` of shape `q × m`).
    ///
    /// # Errors
    /// See [`CostModel::forward`].
    fn backward(&self, x: &Matrix, cost_bar: &Matrix) -> Result<Matrix>;

    /// Whether the costs are deterministic, i.e. posterior samples equal
    /// [`CostModel::forward`] (BoTorch's `DeterministicModel`).
    fn is_deterministic(&self) -> bool {
        true
    }
}

/// Resolves a (possibly negative, Python-style) column index for `d` columns.
fn column(index: isize, d: usize) -> Result<usize> {
    let k = if index < 0 { d as isize + index } else { index };
    if k < 0 || k as usize >= d {
        return Err(BotorchError::DimensionMismatch(format!(
            "index {index} is out of bounds for dimension with size {d}"
        )));
    }
    Ok(k as usize)
}

/// BoTorch's `AffineFidelityCostModel`: the cost of a point is
/// `fixed_cost + Σⱼ weights[j] · x[fidelity_dims[j]]` (a single output).
///
/// The fidelity dimensions are the sorted keys of `fidelity_weights`
/// (negative indices count from the end, as in BoTorch); by default the last
/// column with weight 1. BoTorch stores the weights as a tensor of torch's
/// default dtype (float32 unless changed); botorch-rust uses them as exact
/// `f64` values, like the other Python floats (see the crate documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct AffineFidelityCostModel {
    fidelity_dims: Vec<isize>,
    weights: Vec<f64>,
    fixed_cost: f64,
}

impl AffineFidelityCostModel {
    /// `AffineFidelityCostModel(fidelity_weights, fixed_cost)`; `None`
    /// weights mean `{-1: 1.0}` (BoTorch's default `fixed_cost` is `0.01`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for non-finite weights or cost.
    pub fn new(fidelity_weights: Option<&[(isize, f64)]>, fixed_cost: f64) -> Result<Self> {
        let mut pairs: Vec<(isize, f64)> = fidelity_weights.map_or(vec![(-1, 1.0)], <[_]>::to_vec);
        pairs.sort_by_key(|&(k, _)| k);
        pairs.dedup_by_key(|&mut (k, _)| k);
        if !fixed_cost.is_finite() || pairs.iter().any(|&(_, w)| !w.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "the weights and the fixed cost must be finite".into(),
            ));
        }
        Ok(Self {
            fidelity_dims: pairs.iter().map(|&(k, _)| k).collect(),
            weights: pairs.iter().map(|&(_, w)| w).collect(),
            fixed_cost,
        })
    }

    /// The fidelity dimensions (sorted, possibly negative).
    pub fn fidelity_dims(&self) -> &[isize] {
        &self.fidelity_dims
    }

    /// The weights of the fidelity dimensions.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The fixed cost of a point.
    pub fn fixed_cost(&self) -> f64 {
        self.fixed_cost
    }

    fn columns(&self, d: usize) -> Result<Vec<usize>> {
        self.fidelity_dims.iter().map(|&k| column(k, d)).collect()
    }
}

impl CostModel for AffineFidelityCostModel {
    fn num_outputs(&self) -> usize {
        1
    }

    fn forward(&self, x: &Matrix) -> Result<Matrix> {
        let cols = self.columns(x.ncols())?;
        Ok(Matrix::from_fn(x.nrows(), 1, |p, _| {
            // torch.einsum("...f,f", X[..., dims], weights) + fixed_cost.
            let lin: f64 = cols
                .iter()
                .zip(&self.weights)
                .map(|(&k, &w)| x[(p, k)] * w)
                .sum();
            self.fixed_cost + lin
        }))
    }

    fn backward(&self, x: &Matrix, cost_bar: &Matrix) -> Result<Matrix> {
        let cols = self.columns(x.ncols())?;
        let mut g = Matrix::zeros(x.nrows(), x.ncols());
        for p in 0..x.nrows() {
            for (&k, &w) in cols.iter().zip(&self.weights) {
                g[(p, k)] += cost_bar[(p, 0)] * w;
            }
        }
        Ok(g)
    }
}

/// BoTorch's `FixedCostModel`: a fixed cost per output (e.g. per objective
/// of a decoupled multi-objective problem) for every point.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedCostModel {
    fixed_cost: Vector,
}

impl FixedCostModel {
    /// `FixedCostModel(fixed_cost)` with one cost per output.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty or non-finite cost.
    pub fn new(fixed_cost: Vector) -> Result<Self> {
        if fixed_cost.is_empty() || fixed_cost.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "the fixed cost must be a non-empty vector of finite values".into(),
            ));
        }
        Ok(Self { fixed_cost })
    }

    /// The fixed cost of every output.
    pub fn fixed_cost(&self) -> &Vector {
        &self.fixed_cost
    }
}

impl CostModel for FixedCostModel {
    fn num_outputs(&self) -> usize {
        self.fixed_cost.len()
    }

    fn forward(&self, x: &Matrix) -> Result<Matrix> {
        Ok(Matrix::from_fn(x.nrows(), self.fixed_cost.len(), |_, t| {
            self.fixed_cost[t]
        }))
    }

    fn backward(&self, x: &Matrix, _cost_bar: &Matrix) -> Result<Matrix> {
        Ok(Matrix::zeros(x.nrows(), x.ncols()))
    }
}

/// The posterior mean of a model as a cost model (BoTorch's
/// `cost_model.posterior(X).mean` of a GP cost model, e.g. one fitted on
/// observed costs). Not deterministic: posterior samples of the cost are not
/// supported.
#[derive(Clone, Debug)]
pub struct PosteriorMeanCostModel {
    model: Arc<dyn Model>,
}

impl PosteriorMeanCostModel {
    /// The posterior mean of `model` (an unbatched model).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for batched models.
    pub fn new(model: Arc<dyn Model>) -> Result<Self> {
        crate::acquisition::check_unbatched(model.as_ref())?;
        Ok(Self { model })
    }
}

impl CostModel for PosteriorMeanCostModel {
    fn num_outputs(&self) -> usize {
        self.model.num_outputs()
    }

    fn forward(&self, x: &Matrix) -> Result<Matrix> {
        Ok(self.model.posterior(x, false)?.mean_matrix())
    }

    fn backward(&self, x: &Matrix, cost_bar: &Matrix) -> Result<Matrix> {
        let (q, m) = (x.nrows(), self.model.num_outputs());
        // Output-major adjoint of the joint posterior mean.
        let mean_bar = Vector::from_fn(q * m, |i, _| cost_bar[(i % q, i / q)]);
        self.model
            .joint_posterior_vjp(x, 0..q, &mean_bar, &Matrix::zeros(q * m, q * m))
    }

    fn is_deterministic(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_and_fixed_costs() {
        let x = Matrix::from_row_slice(2, 3, &[0.1, 0.2, 0.5, 0.3, 0.4, 1.0]);
        let m = AffineFidelityCostModel::new(None, 0.01).unwrap();
        assert_eq!(m.forward(&x).unwrap().as_slice(), &[0.51, 1.01]);
        let m = AffineFidelityCostModel::new(Some(&[(2, 2.0), (0, 0.5)]), 1.0).unwrap();
        assert_eq!(m.fidelity_dims(), &[0, 2]);
        let c = m.forward(&x).unwrap();
        assert!((c[(1, 0)] - (1.0 + 0.15 + 2.0)).abs() < 1e-15);
        let g = m
            .backward(&x, &Matrix::from_column_slice(2, 1, &[1.0, -2.0]))
            .unwrap();
        assert_eq!(
            g.row(1).iter().copied().collect::<Vec<_>>(),
            vec![-1.0, 0.0, -4.0]
        );
        assert!(AffineFidelityCostModel::new(Some(&[(3, 1.0)]), 0.0)
            .unwrap()
            .forward(&x)
            .is_err());
        let f = FixedCostModel::new(Vector::from_vec(vec![1.0, 3.0])).unwrap();
        assert_eq!(
            f.forward(&x)
                .unwrap()
                .row(1)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![1.0, 3.0]
        );
        assert_eq!(f.num_outputs(), 2);
    }
}
