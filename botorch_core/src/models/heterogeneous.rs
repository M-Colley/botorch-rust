//! Target-task input embedding for heterogeneous multitask acquisition models.

use std::ops::Range;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, PointPosteriors, Posterior};
use crate::{Matrix, Vector};

/// Embed target-task features into the full feature space of a native GP.
///
/// Public inputs contain the target features in their declared order followed
/// by a task indicator fixed at zero. The wrapped model already selects task
/// zero and accepts full features without an explicit task indicator.
#[derive(Clone, Debug)]
pub struct TargetTaskModel {
    model: Arc<dyn Model>,
    feature_indices: Vec<usize>,
    imputation: Vec<f64>,
}

impl TargetTaskModel {
    /// Build an immutable native target-task model view.
    pub fn new(
        model: Arc<dyn Model>,
        feature_indices: Vec<usize>,
        imputation: Vec<f64>,
    ) -> Result<Self> {
        let unique: std::collections::HashSet<_> = feature_indices.iter().collect();
        if feature_indices.is_empty()
            || unique.len() != feature_indices.len()
            || feature_indices.iter().any(|&i| i >= model.dim())
            || imputation.len() != model.dim()
            || imputation.iter().any(|v| !v.is_finite())
        {
            return Err(BotorchError::InvalidArgument(
                "Target-task feature indices and finite imputation values must match the full model dimension".into(),
            ));
        }
        Ok(Self {
            model,
            feature_indices,
            imputation,
        })
    }

    /// Map public target inputs to the wrapped full feature space.
    fn map(&self, x: &Matrix) -> Result<Matrix> {
        if x.ncols() != self.dim() {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected {} target features plus task column, got {} columns",
                self.dim() - 1,
                x.ncols()
            )));
        }
        if x.column(x.ncols() - 1).iter().any(|&task| task != 0.0) {
            return Err(BotorchError::Unsupported(
                "Posterior can only be called for the target task".into(),
            ));
        }
        let mut mapped = Matrix::from_fn(x.nrows(), self.model.dim(), |_, j| self.imputation[j]);
        for (j, &full) in self.feature_indices.iter().enumerate() {
            mapped.column_mut(full).copy_from(&x.column(j));
        }
        Ok(mapped)
    }
}

impl Model for TargetTaskModel {
    fn dim(&self) -> usize {
        self.feature_indices.len() + 1
    }
    fn num_outputs(&self) -> usize {
        self.model.num_outputs()
    }
    fn batch_shape(&self) -> &[usize] {
        self.model.batch_shape()
    }
    fn points_per_input(&self) -> usize {
        self.model.points_per_input()
    }
    fn is_deterministic(&self) -> bool {
        self.model.is_deterministic()
    }
    fn transformed_outputs(&self) -> Vec<bool> {
        self.model.transformed_outputs()
    }
    fn clamps_variance(&self) -> bool {
        self.model.clamps_variance()
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        self.model.posterior(&self.map(x)?, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let source = self.model.joint_posterior_with_grad(&self.map(x)?)?;
        let size = source.mean.len();
        let mut dmean = Vec::with_capacity(x.nrows() * self.dim());
        let mut dcov = Vec::with_capacity(x.nrows() * self.dim());
        for row in 0..x.nrows() {
            for &full in &self.feature_indices {
                let index = row * self.model.dim() + full;
                dmean.push(source.dmean[index].clone());
                dcov.push(source.dcov[index].clone());
            }
            dmean.push(Vector::zeros(size));
            dcov.push(Matrix::zeros(size, size));
        }
        Ok(JointPosteriorGrad {
            dmean,
            dcov,
            ..source
        })
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        let source = self.model.joint_posterior_vjp(
            &self.map(x)?,
            rows,
            mean_adjoint,
            covariance_adjoint,
        )?;
        Ok(Matrix::from_fn(source.nrows(), self.dim(), |i, j| {
            self.feature_indices
                .get(j)
                .map_or(0.0, |&full| source[(i, full)])
        }))
    }

    fn point_posteriors(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.model.point_posteriors(&self.map(x)?)
    }

    fn point_posteriors_with_noise(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.model.point_posteriors_with_noise(&self.map(x)?)
    }

    fn point_posteriors_joint(&self, x: &Matrix) -> Result<PointPosteriors> {
        self.model.point_posteriors_joint(&self.map(x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SingleTaskGP, SingleTaskGPOptions};

    #[test]
    fn target_embedding_projects_forward_and_reverse_posterior_derivatives() {
        let train_x = Matrix::from_row_slice(
            4,
            3,
            &[0.1, 0.4, 0.7, 0.3, 0.8, 0.1, 0.5, 0.2, 0.4, 0.9, 0.6, 0.2],
        );
        let gp = Arc::new(
            SingleTaskGP::new(
                train_x,
                Matrix::from_column_slice(4, 1, &[0.2, 0.5, -0.1, 0.9]),
                SingleTaskGPOptions::default(),
            )
            .unwrap(),
        );
        let model = TargetTaskModel::new(gp, vec![2, 0], vec![0.2, 0.6, 0.4]).unwrap();
        let x = Matrix::from_row_slice(2, 3, &[0.3, 0.7, 0.0, 0.6, 0.2, 0.0]);
        let mean_adjoint = Vector::from_vec(vec![0.3, -0.7]);
        let covariance_adjoint = Matrix::from_row_slice(2, 2, &[0.2, 0.4, 0.4, -0.3]);
        let grad = model.joint_posterior_with_grad(&x).unwrap();
        let reverse = model
            .joint_posterior_vjp(&x, 0..2, &mean_adjoint, &covariance_adjoint)
            .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let mut plus = x.clone();
                plus[(i, j)] += 1e-6;
                let mut minus = x.clone();
                minus[(i, j)] -= 1e-6;
                let value = |points: &Matrix| {
                    let p = model.posterior(points, false).unwrap();
                    mean_adjoint.dot(&p.mean)
                        + covariance_adjoint.component_mul(&p.covariance).sum()
                };
                let finite_difference = (value(&plus) - value(&minus)) / 2e-6;
                let forward = mean_adjoint.dot(&grad.dmean[i * 3 + j])
                    + covariance_adjoint
                        .component_mul(&grad.dcov[i * 3 + j])
                        .sum();
                assert!((finite_difference - reverse[(i, j)]).abs() < 2e-7);
                assert!((finite_difference - forward).abs() < 2e-7);
            }
            assert_eq!(reverse[(i, 2)], 0.0);
            assert_eq!(grad.dmean[i * 3 + 2], Vector::zeros(2));
        }
        let mut other_task = x.clone();
        other_task[(0, 2)] = 1.0;
        assert!(model.posterior(&other_task, false).is_err());
    }
}
