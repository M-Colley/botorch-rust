//! Empirical covariance from aligned, piecewise-linear historical curves.

use crate::error::{BotorchError, Result};
use crate::{Matrix, Vector};

/// Fixed basis data. Rows in every table refer to the same historical curve.
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalBasisSpec {
    /// Sorted progression coordinates for each output or task.
    pub knots: Vec<Vec<f64>>,
    /// Basis curves, one row per effective curve.
    pub tables: Vec<Matrix>,
    /// Fixed weights, or initial weights when `learn_weights` is true.
    pub weights: Vec<f64>,
    /// Register a direct, nonnegative weight parameter for each curve.
    pub learn_weights: bool,
    /// Original historical curve count minus the covariance correction.
    pub denominator: f64,
    /// Optional column of integer output/task indices (progression is column 0).
    pub group_column: Option<usize>,
    /// Suppress covariance between different output groups.
    pub independent: bool,
}

impl EmpiricalBasisSpec {
    /// Validate fixed data and the selected input dimension.
    pub fn validate(&self, dim: usize) -> Result<()> {
        if self.tables.is_empty()
            || self.knots.len() != self.tables.len()
            || self.weights.is_empty()
            || self.weights.iter().any(|v| !v.is_finite())
            || !self.denominator.is_finite()
            || self.denominator <= 0.0
            || self.group_column.is_some_and(|c| c == 0 || c >= dim)
            || (self.group_column.is_none() && self.tables.len() != 1)
            || dim == 0
        {
            return Err(BotorchError::InvalidArgument(
                "inconsistent empirical basis specification".into(),
            ));
        }
        for (knots, table) in self.knots.iter().zip(&self.tables) {
            if knots.len() < 2
                || table.shape() != (self.weights.len(), knots.len())
                || knots.iter().any(|v| !v.is_finite())
                || knots.windows(2).any(|v| v[0] >= v[1])
                || table.iter().any(|v| !v.is_finite())
            {
                return Err(BotorchError::InvalidArgument(
                    "empirical basis requires finite curves on strictly increasing knots".into(),
                ));
            }
        }
        Ok(())
    }
    fn group(&self, x: &[f64]) -> Result<usize> {
        let value = self.group_column.map_or(0.0, |c| x[c]);
        if !value.is_finite()
            || value < 0.0
            || value.fract() != 0.0
            || value >= self.tables.len() as f64
        {
            return Err(BotorchError::InvalidArgument(format!(
                "empirical task index {value} is out of range"
            )));
        }
        Ok(value as usize)
    }
    /// Validate a query before infallible tree evaluation.
    pub fn validate_point(&self, x: &[f64]) -> Result<()> {
        let group = self.group(x)?;
        let knots = &self.knots[group];
        if !x[0].is_finite() || x[0] < knots[0] || x[0] > knots[knots.len() - 1] {
            return Err(BotorchError::InvalidArgument(format!(
                "empirical progression {} is outside [{}, {}]",
                x[0],
                knots[0],
                knots[knots.len() - 1]
            )));
        }
        Ok(())
    }
    /// Basis values and progression slopes, using the left interval at knots.
    pub fn basis(&self, x: &[f64]) -> (usize, Vector, Vector) {
        let group = self.group_column.map_or(0, |c| x[c] as usize);
        let knots = &self.knots[group];
        let table = &self.tables[group];
        let i = knots
            .partition_point(|&v| v < x[0])
            .saturating_sub(1)
            .min(knots.len() - 2);
        let delta = knots[i + 1] - knots[i];
        let w = (x[0] - knots[i]) / delta;
        (
            group,
            Vector::from_fn(table.nrows(), |r, _| {
                table[(r, i)] * (1.0 - w) + table[(r, i + 1)] * w
            }),
            Vector::from_fn(table.nrows(), |r, _| {
                (table[(r, i + 1)] - table[(r, i)]) / delta
            }),
        )
    }
}

/// An empirical covariance with the current constrained curve weights.
#[derive(Clone, Debug)]
pub struct EmpiricalBasisState {
    /// Fixed historical data and registration metadata.
    pub spec: EmpiricalBasisSpec,
    weights: Vec<f64>,
    offset: usize,
}
impl EmpiricalBasisState {
    /// Instantiate actual parameter values at a kernel-tree offset.
    pub fn new(spec: &EmpiricalBasisSpec, values: &[f64], offset: usize) -> Result<Self> {
        let weights = if spec.learn_weights {
            values.to_vec()
        } else {
            spec.weights.clone()
        };
        if weights.len() != spec.weights.len() || weights.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "inconsistent empirical curve weights".into(),
            ));
        }
        Ok(Self {
            spec: spec.clone(),
            weights,
            offset,
        })
    }
    /// Covariance and derivative with respect to the first progression coordinate.
    pub fn pair(&self, x: &[f64], y: &[f64]) -> (f64, Vector) {
        let (a, u, du) = self.spec.basis(x);
        let (b, v, _) = self.spec.basis(y);
        let mut value = 0.0;
        let mut grad = Vector::zeros(x.len());
        if !self.spec.independent || a == b {
            for r in 0..self.weights.len() {
                let scale = self.weights[r] * self.weights[r] / self.spec.denominator;
                value += scale * u[r] * v[r];
                grad[0] += scale * du[r] * v[r];
            }
        }
        (value, grad)
    }
    /// Accumulate derivatives with respect to actual curve weights.
    pub fn accumulate_parameter_gradient(
        &self,
        x: &[f64],
        y: &[f64],
        weight: f64,
        out: &mut [f64],
    ) {
        if !self.spec.learn_weights {
            return;
        }
        let (a, u, _) = self.spec.basis(x);
        let (b, v, _) = self.spec.basis(y);
        if !self.spec.independent || a == b {
            for r in 0..self.weights.len() {
                out[self.offset + r] +=
                    weight * 2.0 * self.weights[r] * u[r] * v[r] / self.spec.denominator;
            }
        }
    }
}
