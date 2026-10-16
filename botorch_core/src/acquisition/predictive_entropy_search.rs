//! Single-objective predictive entropy search
//! (`botorch.acquisition.predictive_entropy_search`; Hernández-Lobato et
//! al., *Predictive Entropy Search for Efficient Global Optimization of
//! Black-box Functions*, NeurIPS 2014).
//!
//! [`QPredictiveEntropySearch`] is BoTorch's `qPredictiveEntropySearch`:
//! [`QMultiObjectivePredictiveEntropySearch`] with one optimal input per
//! sample (the "Pareto sets" `optimal_inputs[i]` of one point). Expectation
//! propagation runs at construction over the training inputs and the
//! optimal inputs; every evaluation performs one EP update at the
//! candidates with the cached factors of the optimal inputs. As in BoTorch
//! (meta-pytorch/botorch#3429), the factors of the training inputs are not
//! used at the candidates.

use std::sync::Arc;

use crate::acquisition::multi_objective::{
    QMultiObjectivePredictiveEntropySearch, QMultiObjectivePredictiveEntropySearchOptions,
};
use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::Matrix;

/// BoTorch's `qPredictiveEntropySearch`: see the [module
/// documentation](self).
#[derive(Debug)]
pub struct QPredictiveEntropySearch {
    inner: QMultiObjectivePredictiveEntropySearch,
    optimal_inputs: Matrix,
}

impl QPredictiveEntropySearch {
    /// `qPredictiveEntropySearch(model, optimal_inputs, maximize, X_pending,
    /// max_ep_iterations, ep_jitter, test_jitter, threshold)` for a
    /// single-output model and the optimal inputs (`num_samples × d`).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models; see
    /// [`QMultiObjectivePredictiveEntropySearch::new`].
    pub fn new(
        model: Arc<dyn Model>,
        optimal_inputs: &Matrix,
        options: QMultiObjectivePredictiveEntropySearchOptions,
    ) -> Result<Self> {
        if model.num_outputs() != 1 {
            return Err(BotorchError::Unsupported(
                "qPredictiveEntropySearch requires a single-output model".into(),
            ));
        }
        let sets = (0..optimal_inputs.nrows())
            .map(|i| optimal_inputs.rows(i, 1).into_owned())
            .collect();
        Ok(Self {
            inner: QMultiObjectivePredictiveEntropySearch::new(model, sets, options)?,
            optimal_inputs: optimal_inputs.clone(),
        })
    }

    /// The optimal inputs.
    pub fn optimal_inputs(&self) -> &Matrix {
        &self.optimal_inputs
    }

    /// The underlying multi-objective PES (its EP summary and factors).
    pub fn as_mo(&self) -> &QMultiObjectivePredictiveEntropySearch {
        &self.inner
    }
}

impl AcquisitionFunction for QPredictiveEntropySearch {
    fn model(&self) -> &Arc<dyn Model> {
        self.inner.model()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        self.inner.forward(x)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        self.inner.forward_with_grad(x)
    }

    fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
        self.inner.forward_many(xs)
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.inner.x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.inner.set_x_pending(x_pending)
    }
}
