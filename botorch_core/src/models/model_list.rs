//! [`ModelListGP`]: independent GPs for the outputs, possibly with different
//! training data (BoTorch's `botorch.models.ModelListGP`).
//!
//! The outputs of a model list are the outputs of its sub-models in order
//! (a sub-model may itself have several outputs). Its joint posterior is the
//! block-diagonal Gaussian of the sub-model posteriors in BoTorch's
//! output-major layout — BoTorch's `ModelListGPyTorchModel.posterior`, which
//! returns a `MultitaskMultivariateNormal` with `interleaved=False` (with a
//! single output, the sub-model's own posterior).

use std::ops::Range;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::batched::{block_diagonal, block_diagonal_grad};
use crate::models::model::check_vjp_shapes;
use crate::models::{BatchedMultiOutputGP, Model};
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
use crate::{Matrix, Vector};

/// A list of independent GP models (BoTorch's `ModelListGP(*models)`).
///
/// Sub-models are shared through `Arc`s, so cloning a list is cheap; fitting
/// replaces the fitted sub-models (copy on write). All sub-models must have
/// the same input dimension and batch shape.
#[derive(Clone, Debug)]
pub struct ModelListGP {
    models: Vec<Arc<BatchedMultiOutputGP>>,
    num_outputs: usize,
}

impl ModelListGP {
    /// A model list of the given sub-models (BoTorch's `ModelListGP(*models)`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty list and
    /// [`BotorchError::DimensionMismatch`] if the input dimensions or batch
    /// shapes differ.
    pub fn new(models: Vec<BatchedMultiOutputGP>) -> Result<Self> {
        Self::from_shared(models.into_iter().map(Arc::new).collect())
    }

    /// A model list sharing the given sub-models.
    ///
    /// # Errors
    /// See [`ModelListGP::new`].
    pub fn from_shared(models: Vec<Arc<BatchedMultiOutputGP>>) -> Result<Self> {
        let Some(first) = models.first() else {
            return Err(BotorchError::InvalidArgument(
                "ModelListGP requires at least one model".into(),
            ));
        };
        let (d, batch) = (first.dim(), first.batch_shape().to_vec());
        for (i, m) in models.iter().enumerate() {
            if m.dim() != d {
                return Err(BotorchError::DimensionMismatch(format!(
                    "all sub-models must have the same input dimension: model 0 has d = {d}, \
                     model {i} has d = {}",
                    m.dim()
                )));
            }
            if m.batch_shape() != batch.as_slice() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "all sub-models must have the same batch shape: model 0 has {batch:?}, \
                     model {i} has {:?}",
                    m.batch_shape()
                )));
            }
        }
        let num_outputs = models.iter().map(|m| m.num_outputs()).sum();
        Ok(Self {
            models,
            num_outputs,
        })
    }

    /// The sub-models.
    pub fn models(&self) -> &[Arc<BatchedMultiOutputGP>] {
        &self.models
    }

    /// Replaces sub-model `index` (e.g. by a refitted copy).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the new model's input dimension,
    /// batch shape or number of outputs differs, or the index is out of range.
    pub fn replace_model(&mut self, index: usize, model: BatchedMultiOutputGP) -> Result<()> {
        let old = self.models.get(index).ok_or_else(|| {
            BotorchError::DimensionMismatch(format!("model index {index} out of range"))
        })?;
        if old.dim() != model.dim()
            || old.batch_shape() != model.batch_shape()
            || old.num_outputs() != model.num_outputs()
        {
            return Err(BotorchError::DimensionMismatch(
                "the replacement must have the same input dimension, batch shape and number \
                 of outputs"
                    .into(),
            ));
        }
        self.models[index] = Arc::new(model);
        Ok(())
    }

    /// Number of sub-models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the list is empty (never true for a constructed list).
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Total number of outputs (BoTorch's `num_outputs`).
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Input dimension `d`.
    pub fn dim(&self) -> usize {
        self.models[0].dim()
    }

    /// The common batch shape of the sub-models.
    pub fn batch_shape(&self) -> &[usize] {
        self.models[0].batch_shape()
    }

    /// The range of list outputs that belong to sub-model `index`.
    pub fn output_range(&self, index: usize) -> Range<usize> {
        let start: usize = self.models[..index].iter().map(|m| m.num_outputs()).sum();
        start..start + self.models[index].num_outputs()
    }

    /// The marginal log likelihoods of the sub-models (each over the elements
    /// of its `_aug_batch_shape`). GPyTorch's `SumMarginalLogLikelihood` is
    /// their (broadcast) sum divided by the number of sub-models.
    ///
    /// # Errors
    /// See [`BatchedMultiOutputGP::mll_values`].
    pub fn mll_values(&self) -> Result<Vec<Vec<f64>>> {
        self.models.iter().map(|m| m.mll_values()).collect()
    }

    /// The joint posterior of batch element `batch` at the `q × d` inputs `x`
    /// over all outputs (block diagonal, output-major).
    ///
    /// # Errors
    /// See [`BatchedMultiOutputGP::posterior_at`].
    pub fn posterior_at(
        &self,
        batch: usize,
        x: &Matrix,
        observation_noise: bool,
    ) -> Result<Posterior> {
        if self.models.len() == 1 {
            return self.models[0].posterior_at(batch, x, observation_noise);
        }
        let parts = self
            .models
            .iter()
            .map(|m| m.posterior_at(batch, x, observation_noise))
            .collect::<Result<Vec<_>>>()?;
        Ok(block_diagonal(&parts))
    }

    /// Posteriors of every row of `x` on its own (`q = 1`) for batch element
    /// `batch` over all outputs (unclamped).
    ///
    /// # Errors
    /// See [`BatchedMultiOutputGP::posterior_at`].
    pub fn point_posteriors_at(
        &self,
        batch: usize,
        x: &Matrix,
        observation_noise: bool,
    ) -> Result<PointPosteriors> {
        let parts = self
            .models
            .iter()
            .map(|m| m.point_posteriors_at(batch, x, observation_noise))
            .collect::<Result<Vec<_>>>()?;
        Ok(concat_point_posteriors(&parts))
    }

    fn require_unbatched(&self) -> Result<()> {
        if !self.batch_shape().is_empty() {
            return Err(BotorchError::Unsupported(format!(
                "batched models (batch_shape = {:?}) cannot be used through the Model trait \
                 (e.g. by acquisition functions); evaluate them per batch element with \
                 posterior_at",
                self.batch_shape()
            )));
        }
        Ok(())
    }
}

/// Point posteriors of independent output groups, concatenated along the outputs.
fn concat_point_posteriors(parts: &[PointPosteriors]) -> PointPosteriors {
    if parts.len() == 1 {
        return parts[0].clone();
    }
    let b = parts[0].len();
    let m: usize = parts.iter().map(|p| p.num_outputs()).sum();
    let mut mean = Matrix::zeros(b, m);
    let mut cov = Matrix::zeros(b, m * m);
    let mut o = 0;
    for p in parts {
        let k = p.num_outputs();
        for i in 0..b {
            for a in 0..k {
                mean[(i, o + a)] = p.mean[(i, a)];
                for c in 0..k {
                    cov[(i, (o + a) * m + o + c)] = p.covariance[(i, a * k + c)];
                }
            }
        }
        o += k;
    }
    PointPosteriors {
        mean,
        covariance: cov,
    }
}

impl Model for ModelListGP {
    fn dim(&self) -> usize {
        ModelListGP::dim(self)
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn batch_shape(&self) -> &[usize] {
        ModelListGP::batch_shape(self)
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        self.require_unbatched()?;
        self.posterior_at(0, x, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.require_unbatched()?;
        let parts = self
            .models
            .iter()
            .map(|m| m.joint_posterior_with_grad(x))
            .collect::<Result<Vec<_>>>()?;
        Ok(block_diagonal_grad(&parts))
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        self.require_unbatched()?;
        check_vjp_shapes(x, &rows, mean_adjoint, covariance_adjoint, self.num_outputs)?;
        let q = x.nrows();
        let mut out = Matrix::zeros(rows.len(), x.ncols());
        let mut o = 0;
        for m in &self.models {
            let k = m.num_outputs() * q;
            let ma = mean_adjoint.rows(o, k).into_owned();
            let ca = covariance_adjoint.view((o, o), (k, k)).into_owned();
            out += m.joint_posterior_vjp(x, rows.clone(), &ma, &ca)?;
            o += k;
        }
        Ok(out)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.require_unbatched()?;
        self.point_posteriors_at(0, x, false)
    }

    fn point_posterior_with_grad(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        self.require_unbatched()?;
        let parts = self
            .models
            .iter()
            .map(|m| m.point_posterior_with_grad(x))
            .collect::<Result<Vec<_>>>()?;
        Ok(block_diagonal_grad(&parts))
    }

    fn posterior_marginals(&self, x: &Matrix) -> Result<(Vector, Vector)> {
        self.require_unbatched()?;
        if self.num_outputs == 1 {
            return self.models[0].posterior_marginals(x);
        }
        Err(BotorchError::Unsupported(format!(
            "this method requires a single-output model, got {} outputs",
            self.num_outputs
        )))
    }

    fn posterior_q1_with_grad(&self, x: &[f64]) -> Result<(f64, f64, Vector, Vector)> {
        self.require_unbatched()?;
        if self.num_outputs == 1 {
            return self.models[0].posterior_q1_with_grad(x);
        }
        Err(BotorchError::Unsupported(format!(
            "this method requires a single-output model, got {} outputs",
            self.num_outputs
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SingleTaskGP, SingleTaskGPOptions};

    fn gp(n: usize, shift: f64, m: usize) -> BatchedMultiOutputGP {
        let x = Matrix::from_fn(n, 2, |i, j| ((i * 5 + j * 2) % n) as f64 / n as f64);
        let y = Matrix::from_fn(n, m, |i, t| {
            (x[(i, 0)] * 4.0 + shift * (t + 1) as f64).sin() + x[(i, 1)]
        });
        BatchedMultiOutputGP::multi_output(x, y, SingleTaskGPOptions::default()).unwrap()
    }

    #[test]
    fn posterior_is_block_diagonal_concatenation() {
        let a = gp(8, 0.0, 1);
        let b = gp(6, 0.7, 2);
        let list = ModelListGP::new(vec![a.clone(), b.clone()]).unwrap();
        assert_eq!(list.num_outputs(), 3);
        assert_eq!(list.output_range(1), 1..3);
        let x = Matrix::from_row_slice(2, 2, &[0.1, 0.2, 0.5, 0.9]);
        let p = Model::posterior(&list, &x, false).unwrap();
        let pa = Model::posterior(&a, &x, false).unwrap();
        let pb = Model::posterior(&b, &x, false).unwrap();
        assert_eq!(p.select_outputs(&[0]).unwrap().mean, pa.mean);
        assert_eq!(p.select_outputs(&[1, 2]).unwrap().covariance, pb.covariance);
        assert_eq!(p.covariance[(0, 2)], 0.0);
        let pp = Model::point_posteriors(&list, &x).unwrap();
        assert_eq!(pp.num_outputs(), 3);
        assert_eq!(pp.covariance_of(1)[(0, 1)], 0.0);
        // VJP equals the contraction of the forward-mode derivatives.
        let g = Model::joint_posterior_with_grad(&list, &x).unwrap();
        let ma = Vector::from_fn(6, |i, _| 0.2 * i as f64 - 0.4);
        let ca = Matrix::from_fn(6, 6, |i, j| ((i + 2 * j) % 4) as f64 / 4.0);
        let vjp = list.joint_posterior_vjp(&x, 0..2, &ma, &ca).unwrap();
        for j in 0..4 {
            let expected = ma.dot(&g.dmean[j]) + ca.component_mul(&g.dcov[j]).sum();
            assert!((vjp[(j / 2, j % 2)] - expected).abs() < 1e-12 * expected.abs().max(1.0));
        }
        assert!(ModelListGP::new(Vec::new()).is_err());
        let single: BatchedMultiOutputGP = SingleTaskGP::new(
            Matrix::from_row_slice(2, 1, &[0.0, 1.0]),
            Matrix::from_row_slice(2, 1, &[0.0, 1.0]),
            SingleTaskGPOptions::default(),
        )
        .unwrap()
        .into();
        assert!(ModelListGP::new(vec![a, single]).is_err());
    }
}
