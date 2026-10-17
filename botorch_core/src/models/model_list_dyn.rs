//! [`ModelList`]: a multi-output model of independent sub-models of any kind
//! (BoTorch's `botorch.models.model.ModelList`, the base of `ModelListGP`),
//! e.g. a `ModelListGP` of [`crate::models::LCEMGP`]s.
//!
//! Its outputs are the outputs of the sub-models in order, and its joint
//! posterior is the block-diagonal concatenation of the sub-model posteriors
//! in BoTorch's output-major layout, exactly like [`ModelListGP`] (which is
//! specialized to `SingleTaskGP` sub-models and adds batching, fitting and
//! fantasies).
//!
//! [`ModelListGP`]: crate::models::ModelListGP

use std::ops::Range;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::batched::{block_diagonal, block_diagonal_grad};
use crate::models::model::check_vjp_shapes_expanded;
use crate::models::model_list::concat_point_posteriors;
use crate::models::{Model, SingleTaskGP};
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
use crate::{Matrix, Vector};

/// A list of independent unbatched models (see the [module
/// documentation](self)).
#[derive(Clone, Debug)]
pub struct ModelList {
    models: Vec<Arc<dyn Model>>,
    num_outputs: usize,
}

impl ModelList {
    /// A list of the given sub-models.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty list,
    /// [`BotorchError::DimensionMismatch`] if the input dimensions or the
    /// numbers of posterior points per input differ, and
    /// [`BotorchError::Unsupported`] for batched sub-models.
    pub fn new(models: Vec<Arc<dyn Model>>) -> Result<Self> {
        let Some(first) = models.first() else {
            return Err(BotorchError::InvalidArgument(
                "ModelList requires at least one model".into(),
            ));
        };
        let (d, k) = (first.dim(), first.points_per_input());
        for (i, m) in models.iter().enumerate() {
            if m.dim() != d {
                return Err(BotorchError::DimensionMismatch(format!(
                    "all sub-models must have the same input dimension: model 0 has d = {d}, \
                     model {i} has d = {}",
                    m.dim()
                )));
            }
            if m.points_per_input() != k {
                return Err(BotorchError::DimensionMismatch(format!(
                    "all sub-models must map an input to the same number of posterior points: \
                     model 0 has {k}, model {i} has {}",
                    m.points_per_input()
                )));
            }
            if !m.batch_shape().is_empty() {
                return Err(BotorchError::Unsupported(format!(
                    "ModelList requires unbatched sub-models, model {i} has batch shape {:?}",
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
    pub fn models(&self) -> &[Arc<dyn Model>] {
        &self.models
    }
}

impl Model for ModelList {
    fn dim(&self) -> usize {
        self.models[0].dim()
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn points_per_input(&self) -> usize {
        self.models[0].points_per_input()
    }

    fn output_gps(&self) -> Option<Vec<&SingleTaskGP>> {
        let mut out = Vec::with_capacity(self.num_outputs);
        for m in &self.models {
            out.extend(m.output_gps()?);
        }
        Some(out)
    }

    fn list_output_sizes(&self) -> Option<Vec<usize>> {
        Some(self.models.iter().map(|m| m.num_outputs()).collect())
    }

    fn transformed_outputs(&self) -> Vec<bool> {
        self.models
            .iter()
            .flat_map(|m| m.transformed_outputs())
            .collect()
    }

    fn fantasy_output_groups(&self) -> Option<Vec<usize>> {
        let mut out = Vec::with_capacity(self.models.len());
        for m in &self.models {
            out.extend(m.fantasy_output_groups()?);
        }
        Some(out)
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        if self.models.len() == 1 {
            return self.models[0].posterior(x, observation_noise);
        }
        let parts = self
            .models
            .iter()
            .map(|m| m.posterior(x, observation_noise))
            .collect::<Result<Vec<_>>>()?;
        Ok(block_diagonal(&parts))
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
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
        let k = self.points_per_input();
        check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            self.num_outputs,
            k,
        )?;
        let q = x.nrows() * k;
        let mut out = Matrix::zeros(rows.len(), x.ncols());
        let mut o = 0;
        for m in &self.models {
            let len = m.num_outputs() * q;
            let ma = mean_adjoint.rows(o, len).into_owned();
            let ca = covariance_adjoint.view((o, o), (len, len)).into_owned();
            out += m.joint_posterior_vjp(x, rows.clone(), &ma, &ca)?;
            o += len;
        }
        Ok(out)
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        let parts = self
            .models
            .iter()
            .map(|m| m.point_posteriors(x))
            .collect::<Result<Vec<_>>>()?;
        Ok(concat_point_posteriors(&parts))
    }

    fn point_posterior_with_grad(&self, x: &[f64]) -> Result<JointPosteriorGrad> {
        let parts = self
            .models
            .iter()
            .map(|m| m.point_posterior_with_grad(x))
            .collect::<Result<Vec<_>>>()?;
        Ok(block_diagonal_grad(&parts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BatchedMultiOutputGP, ModelListGP, SingleTaskGPOptions};

    fn gp(n: usize, shift: f64) -> BatchedMultiOutputGP {
        let x = Matrix::from_fn(n, 2, |i, j| ((i * 5 + j * 2) % n) as f64 / n as f64);
        let y = Matrix::from_fn(n, 1, |i, _| (x[(i, 0)] * 4.0 + shift).sin() + x[(i, 1)]);
        BatchedMultiOutputGP::multi_output(x, y, SingleTaskGPOptions::default()).unwrap()
    }

    #[test]
    fn equals_model_list_gp() {
        let (a, b) = (gp(8, 0.0), gp(6, 0.7));
        let typed = ModelListGP::new(vec![a.clone(), b.clone()]).unwrap();
        let list = ModelList::new(vec![Arc::new(a), Arc::new(b)]).unwrap();
        let x = Matrix::from_row_slice(2, 2, &[0.1, 0.2, 0.5, 0.9]);
        let p = list.posterior(&x, true).unwrap();
        let e = Model::posterior(&typed, &x, true).unwrap();
        assert_eq!(p, e);
        let ma = Vector::from_fn(4, |i, _| 0.2 * i as f64 - 0.4);
        let ca = Matrix::from_fn(4, 4, |i, j| ((i + 2 * j) % 4) as f64 / 4.0);
        assert_eq!(
            list.joint_posterior_vjp(&x, 0..2, &ma, &ca).unwrap(),
            typed.joint_posterior_vjp(&x, 0..2, &ma, &ca).unwrap()
        );
        assert_eq!(
            list.point_posteriors(&x).unwrap(),
            Model::point_posteriors(&typed, &x).unwrap()
        );
        assert!(list.output_gps().is_some());
        assert!(ModelList::new(Vec::new()).is_err());
    }
}
