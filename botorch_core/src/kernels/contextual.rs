//! Native latent-context additive covariance and its exact derivatives.
//!
//! Spatial and embedding RBF lengthscales are shared across all context
//! pairs. Renormalization uses LCEMGP's torch-compatible embedding utility.

use crate::error::{BotorchError, Result};
use crate::models::contextual_multioutput::embedding_renorm;
use crate::Matrix;

/// Fixed structure of a latent-context additive kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct LceaSpec {
    /// Input columns for every context, in spatial feature order.
    pub contexts: Vec<Vec<usize>>,
    /// Embedding-table indices of each context's categorical features.
    pub categorical: Vec<Vec<usize>>,
    /// `(number of categories, embedding dimension)` of each learned table.
    pub embedding_shapes: Vec<(usize, usize)>,
    /// Additional fixed embeddings, one row per context.
    pub fixed_embeddings: Matrix,
    /// Optional known context weights multiplying one shared amplitude.
    pub context_weights: Option<Vec<f64>>,
}

impl LceaSpec {
    /// Number of spatial features per context.
    pub fn spatial_dim(&self) -> usize {
        self.contexts.first().map_or(0, Vec::len)
    }

    /// Total learned and fixed embedding feature dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_shapes.iter().map(|s| s.1).sum::<usize>() + self.fixed_embeddings.ncols()
    }

    /// Number of amplitude parameters.
    pub fn scale_count(&self) -> usize {
        if self.context_weights.is_some() {
            1
        } else {
            self.contexts.len()
        }
    }

    /// Number of learned table entries.
    pub fn weight_count(&self) -> usize {
        self.embedding_shapes.iter().map(|(n, d)| n * d).sum()
    }

    /// Number of parameters preceding the spatial lengthscales.
    pub fn context_parameter_count(&self) -> usize {
        self.scale_count() + self.weight_count() + self.embedding_dim()
    }

    /// Persist the max-norm update of looked-up embedding rows.
    pub fn normalize_parameters(&self, parameters: &mut [f64]) -> Result<()> {
        if parameters.len() != self.context_parameter_count() + self.spatial_dim() {
            return Err(BotorchError::InvalidArgument(
                "inconsistent LCEA parameter values".into(),
            ));
        }
        let mut start = self.scale_count();
        for (layer, &(rows, dim)) in self.embedding_shapes.iter().enumerate() {
            let mut weight =
                Matrix::from_row_slice(rows, dim, &parameters[start..start + rows * dim]);
            let indices: Vec<usize> = self.categorical.iter().map(|r| r[layer]).collect();
            embedding_renorm(&mut weight, &indices, 1.0);
            for r in 0..rows {
                for j in 0..dim {
                    parameters[start + r * dim + j] = weight[(r, j)];
                }
            }
            start += rows * dim;
        }
        Ok(())
    }

    /// Validate dimensions before indexing.
    pub fn validate(&self, input_dim: usize) -> Result<()> {
        let contexts = self.contexts.len();
        let d = self.spatial_dim();
        if contexts == 0 || d == 0 || self.contexts.iter().any(|v| v.len() != d) {
            return Err(BotorchError::InvalidArgument(
                "LCEAKernel needs nonempty equally-sized context decompositions".into(),
            ));
        }
        if self.contexts.iter().flatten().any(|&j| j >= input_dim) {
            return Err(BotorchError::Runtime(
                "LCEAKernel context index is out of range".into(),
            ));
        }
        if self.fixed_embeddings.nrows() != contexts
            || self.embedding_dim() == 0
            || self.fixed_embeddings.iter().any(|v| !v.is_finite())
            || self.categorical.len() != contexts
            || self
                .categorical
                .iter()
                .any(|row| row.len() != self.embedding_shapes.len())
            || self.embedding_shapes.iter().any(|&(n, d)| n == 0 || d == 0)
        {
            return Err(BotorchError::InvalidArgument(
                "inconsistent LCEA context embeddings".into(),
            ));
        }
        for row in &self.categorical {
            for (&r, &(n, _)) in row.iter().zip(&self.embedding_shapes) {
                if r >= n {
                    return Err(BotorchError::InvalidArgument(
                        "LCEA categorical index is out of range".into(),
                    ));
                }
            }
        }
        if self
            .context_weights
            .as_ref()
            .is_some_and(|w| w.len() != contexts || w.iter().any(|v| !v.is_finite()))
        {
            return Err(BotorchError::InvalidArgument(
                "inconsistent LCEA context weights".into(),
            ));
        }
        Ok(())
    }

    /// Effective embeddings from amplitudes, learned tables, task
    /// lengthscales and spatial lengthscales, in that parameter order.
    pub fn embeddings(&self, parameters: &[f64]) -> Matrix {
        let mut embeddings = Matrix::zeros(self.contexts.len(), self.embedding_dim());
        let mut start = self.scale_count();
        let mut column = 0;
        for (layer, &(rows, dim)) in self.embedding_shapes.iter().enumerate() {
            let mut weight =
                Matrix::from_row_slice(rows, dim, &parameters[start..start + rows * dim]);
            let indices: Vec<usize> = self.categorical.iter().map(|r| r[layer]).collect();
            embedding_renorm(&mut weight, &indices, 1.0);
            for c in 0..self.contexts.len() {
                for j in 0..dim {
                    embeddings[(c, column + j)] = weight[(indices[c], j)];
                }
            }
            start += rows * dim;
            column += dim;
        }
        embeddings
            .columns_mut(column, self.fixed_embeddings.ncols())
            .copy_from(&self.fixed_embeddings);
        embeddings
    }
}

/// Context covariance, its parameter derivatives, and shared spatial RBF.
#[derive(Clone, Debug)]
pub struct LceaState {
    spec: LceaSpec,
    spatial_ls: Vec<f64>,
    context_covariance: Matrix,
    context_derivatives: Vec<Matrix>,
    offset: usize,
}

impl LceaState {
    /// Build from actual constrained parameters, at a global tree offset.
    pub fn new(spec: &LceaSpec, parameters: &[f64], offset: usize) -> Result<Self> {
        spec.validate(usize::MAX)?;
        let d = spec.spatial_dim();
        let count = spec.context_parameter_count();
        if parameters.len() != count + d || parameters.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "inconsistent LCEA parameter values".into(),
            ));
        }
        let contexts = spec.contexts.len();
        let embeddings = spec.embeddings(parameters);
        let task_off = spec.scale_count() + spec.weight_count();
        let task_ls = &parameters[task_off..count];
        let scales: Vec<f64> = (0..contexts)
            .map(|c| match &spec.context_weights {
                Some(w) => parameters[0] * w[c],
                None => parameters[c],
            })
            .collect();
        let mut cov = Matrix::zeros(contexts, contexts);
        let mut derivatives = vec![Matrix::zeros(contexts, contexts); count];
        for c in 0..contexts {
            for e in 0..contexts {
                let r2: f64 = (0..task_ls.len())
                    .map(|j| ((embeddings[(c, j)] - embeddings[(e, j)]) / task_ls[j]).powi(2))
                    .sum();
                let base = (-0.5 * r2).exp();
                let value = scales[c] * scales[e] * base;
                cov[(c, e)] = value;
                match &spec.context_weights {
                    Some(w) => {
                        derivatives[0][(c, e)] = (w[c] * scales[e] + scales[c] * w[e]) * base
                    }
                    None => {
                        derivatives[c][(c, e)] += scales[e] * base;
                        derivatives[e][(c, e)] += scales[c] * base;
                    }
                }
                for j in 0..task_ls.len() {
                    let diff = embeddings[(c, j)] - embeddings[(e, j)];
                    derivatives[task_off + j][(c, e)] =
                        value * (diff / task_ls[j]).powi(2) / task_ls[j];
                }
                let mut start = spec.scale_count();
                let mut column = 0;
                for (layer, &(rows, dim)) in spec.embedding_shapes.iter().enumerate() {
                    for j in 0..dim {
                        let col = column + j;
                        let diff = embeddings[(c, col)] - embeddings[(e, col)];
                        let grad = -value * diff / task_ls[col].powi(2);
                        derivatives[start + spec.categorical[c][layer] * dim + j][(c, e)] += grad;
                        derivatives[start + spec.categorical[e][layer] * dim + j][(c, e)] -= grad;
                    }
                    start += rows * dim;
                    column += dim;
                }
            }
        }
        Ok(Self {
            spec: spec.clone(),
            spatial_ls: parameters[count..].to_vec(),
            context_covariance: cov,
            context_derivatives: derivatives,
            offset,
        })
    }

    /// Context covariance including the amplitude outer product.
    pub fn context_covariance(&self) -> &Matrix {
        &self.context_covariance
    }

    fn spatial(&self, x: &[f64], y: &[f64], c: usize, e: usize) -> f64 {
        let r2: f64 = (0..self.spatial_ls.len())
            .map(|j| {
                ((x[self.spec.contexts[c][j]] - y[self.spec.contexts[e][j]]) / self.spatial_ls[j])
                    .powi(2)
            })
            .sum();
        (-0.5 * r2).exp()
    }

    /// Covariance summed over every ordered context pair.
    pub fn value(&self, x: &[f64], y: &[f64]) -> f64 {
        (0..self.spec.contexts.len())
            .map(|c| {
                (0..self.spec.contexts.len())
                    .map(|e| self.context_covariance[(c, e)] * self.spatial(x, y, c, e))
                    .sum::<f64>()
            })
            .sum()
    }

    /// Covariance and derivative in the first input's coordinates.
    pub fn input_gradient(&self, x: &[f64], y: &[f64]) -> (f64, Vec<f64>) {
        let mut value = 0.0;
        let mut gradient = vec![0.0; x.len()];
        for c in 0..self.spec.contexts.len() {
            for e in 0..self.spec.contexts.len() {
                let v = self.context_covariance[(c, e)] * self.spatial(x, y, c, e);
                value += v;
                for j in 0..self.spatial_ls.len() {
                    let col = self.spec.contexts[c][j];
                    gradient[col] -=
                        v * (x[col] - y[self.spec.contexts[e][j]]) / self.spatial_ls[j].powi(2);
                }
            }
        }
        (value, gradient)
    }

    /// Add a weighted pair's derivatives into a full tree parameter vector.
    pub fn accumulate_parameter_gradient(
        &self,
        x: &[f64],
        y: &[f64],
        weight: f64,
        out: &mut [f64],
    ) {
        let spatial_off = self.offset + self.context_derivatives.len();
        for c in 0..self.spec.contexts.len() {
            for e in 0..self.spec.contexts.len() {
                let v = self.spatial(x, y, c, e) * weight;
                for (p, derivative) in self.context_derivatives.iter().enumerate() {
                    out[self.offset + p] += v * derivative[(c, e)];
                }
                for j in 0..self.spatial_ls.len() {
                    let diff = x[self.spec.contexts[c][j]] - y[self.spec.contexts[e][j]];
                    out[spatial_off + j] +=
                        v * self.context_covariance[(c, e)] * (diff / self.spatial_ls[j]).powi(2)
                            / self.spatial_ls[j];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(weighted: bool) -> LceaSpec {
        LceaSpec {
            contexts: vec![vec![0, 2], vec![3, 1]],
            categorical: vec![vec![0], vec![1]],
            embedding_shapes: vec![(2, 2)],
            fixed_embeddings: Matrix::from_row_slice(2, 1, &[0.1, 0.7]),
            context_weights: weighted.then(|| vec![0.3, 0.7]),
        }
    }

    #[test]
    fn contextual_values_and_derivatives() {
        for weighted in [false, true] {
            let spec = spec(weighted);
            spec.validate(4).unwrap();
            let mut p = if weighted { vec![0.8] } else { vec![0.8, 0.4] };
            p.extend([0.2, 0.3, -0.1, 0.4, 0.7, 0.9, 0.8, 0.3, 0.6]);
            let state = LceaState::new(&spec, &p, 2).unwrap();
            let (x, y) = ([0.2, 0.8, 0.3, 0.5], [0.9, 0.4, 0.7, 0.1]);
            let (v, g) = state.input_gradient(&x, &y);
            assert!((v - state.value(&y, &x)).abs() < 1e-14);
            for j in 0..4 {
                let mut xp = x;
                xp[j] += 1e-6;
                let mut xm = x;
                xm[j] -= 1e-6;
                let fd = (state.value(&xp, &y) - state.value(&xm, &y)) / 2e-6;
                assert!((fd - g[j]).abs() < 1e-8, "input {j}: {fd} != {}", g[j]);
            }
            let mut gp = vec![0.0; p.len() + 2];
            state.accumulate_parameter_gradient(&x, &y, 1.0, &mut gp);
            for j in 0..p.len() {
                let mut pp = p.clone();
                pp[j] += 1e-6;
                let mut pm = p.clone();
                pm[j] -= 1e-6;
                let fd = (LceaState::new(&spec, &pp, 2).unwrap().value(&x, &y)
                    - LceaState::new(&spec, &pm, 2).unwrap().value(&x, &y))
                    / 2e-6;
                assert!(
                    (fd - gp[j + 2]).abs() < 1e-8,
                    "param {j}: {fd} != {}",
                    gp[j + 2]
                );
            }
        }
    }
}
