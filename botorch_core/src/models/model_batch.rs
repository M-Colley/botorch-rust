//! Immutable model snapshots arranged along model batch dimensions.

use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::Matrix;

/// A batch of independent model snapshots with common posterior metadata.
#[derive(Clone, Debug)]
pub struct ModelBatch {
    models: Vec<Arc<dyn Model>>,
    shape: Vec<usize>,
}

impl ModelBatch {
    /// Construct a row-major batch of unbatched models.
    ///
    /// # Errors
    /// Rejects empty/mismatched batches and differing model dimensions,
    /// output layouts or transform metadata.
    pub fn new(models: Vec<Arc<dyn Model>>, shape: Vec<usize>) -> Result<Self> {
        let count = shape.iter().try_fold(1usize, |n, &s| n.checked_mul(s));
        let Some(first) = models.first() else {
            return Err(BotorchError::InvalidArgument("model batch is empty".into()));
        };
        if count != Some(models.len())
            || models.iter().any(|model| {
                !model.batch_shape().is_empty()
                    || model.dim() != first.dim()
                    || model.num_outputs() != first.num_outputs()
                    || model.points_per_input() != first.points_per_input()
                    || model.transformed_outputs() != first.transformed_outputs()
                    || model.list_output_sizes() != first.list_output_sizes()
                    || model.is_deterministic() != first.is_deterministic()
                    || model.clamps_variance() != first.clamps_variance()
            })
        {
            return Err(BotorchError::DimensionMismatch(
                "model batch shape and posterior metadata must match all snapshots".into(),
            ));
        }
        Ok(Self { models, shape })
    }
}

impl Model for ModelBatch {
    fn dim(&self) -> usize {
        self.models[0].dim()
    }
    fn num_outputs(&self) -> usize {
        self.models[0].num_outputs()
    }
    fn points_per_input(&self) -> usize {
        self.models[0].points_per_input()
    }
    fn batch_shape(&self) -> &[usize] {
        &self.shape
    }
    fn batch_element(&self, b: usize) -> Option<&dyn Model> {
        self.models.get(b).map(AsRef::as_ref)
    }
    fn transformed_outputs(&self) -> Vec<bool> {
        self.models[0].transformed_outputs()
    }
    fn list_output_sizes(&self) -> Option<Vec<usize>> {
        self.models[0].list_output_sizes()
    }
    fn is_deterministic(&self) -> bool {
        self.models[0].is_deterministic()
    }
    fn clamps_variance(&self) -> bool {
        self.models[0].clamps_variance()
    }
    fn posterior(&self, x: &Matrix, noisy: bool) -> Result<Posterior> {
        if self.shape.is_empty() {
            return self.models[0].posterior(x, noisy);
        }
        Err(BotorchError::Unsupported(
            "select a model batch element before prediction".into(),
        ))
    }
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        if self.shape.is_empty() {
            return self.models[0].joint_posterior_with_grad(x);
        }
        Err(BotorchError::Unsupported(
            "select a model batch element before differentiation".into(),
        ))
    }
}
