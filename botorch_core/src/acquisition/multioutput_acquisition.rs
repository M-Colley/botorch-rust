//! Multi-output acquisition functions (`botorch.acquisition.multioutput_acquisition`):
//! acquisition functions with one value per output, to be optimized with a
//! multi-objective optimizer such as NSGA-II
//! ([`crate::utils::multi_objective::optimize::optimize_with_nsgaii`]):
//! [`MultiOutputPosteriorMean`] and [`MultiOutputAcquisitionFunctionWrapper`].

use std::fmt::Debug;
use std::sync::Arc;

use crate::acquisition::{check_candidates, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::Matrix;

/// A multi-output acquisition function (BoTorch's
/// `MultiOutputAcquisitionFunction`): `m` values per candidate set, all to be
/// maximized. Pending points are not supported (as in BoTorch).
pub trait MultiOutputAcquisitionFunction: Send + Sync + Debug {
    /// The number of outputs `m`.
    fn num_outputs(&self) -> usize;

    /// The input dimension `d`.
    fn dim(&self) -> usize;

    /// The values (`b × m`) of a batch of candidate sets (BoTorch's
    /// `acqf(X)` for `X` of shape `b x q x d`), evaluated in parallel where
    /// possible.
    ///
    /// # Errors
    /// Shape mismatches and errors of the model.
    fn forward_many(&self, xs: &[Matrix]) -> Result<Matrix>;

    /// The `m` values of a candidate set (`q × d`).
    ///
    /// # Errors
    /// See [`MultiOutputAcquisitionFunction::forward_many`].
    fn forward(&self, x: &Matrix) -> Result<Vec<f64>> {
        let v = self.forward_many(std::slice::from_ref(x))?;
        Ok(v.row(0).iter().copied().collect())
    }
}

/// The posterior mean of every output at single points (BoTorch's
/// `MultiOutputPosteriorMean`), optionally multiplied by `weights` (negative
/// weights minimize an output).
#[derive(Debug)]
pub struct MultiOutputPosteriorMean {
    model: Arc<dyn Model>,
    weights: Option<Vec<f64>>,
}

impl MultiOutputPosteriorMean {
    /// `MultiOutputPosteriorMean(model, weights)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for a single-output or batched model
    /// (BoTorch's `NotImplementedError`) and [`BotorchError::InvalidArgument`]
    /// for a wrong number of weights (BoTorch's messages).
    pub fn new(model: Arc<dyn Model>, weights: Option<Vec<f64>>) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        let m = model.num_outputs();
        if m < 2 {
            return Err(BotorchError::Unsupported(
                "MultiPosteriorMean only supports multi-output models.".into(),
            ));
        }
        if let Some(w) = &weights {
            if w.len() != m {
                return Err(BotorchError::InvalidArgument(format!(
                    "weights must have {m} elements, but got {}.",
                    w.len()
                )));
            }
        }
        Ok(Self { model, weights })
    }

    /// The model.
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// The weights, if any.
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }
}

impl MultiOutputAcquisitionFunction for MultiOutputPosteriorMean {
    fn num_outputs(&self) -> usize {
        self.model.num_outputs()
    }

    fn dim(&self) -> usize {
        self.model.dim()
    }

    fn forward_many(&self, xs: &[Matrix]) -> Result<Matrix> {
        let d = self.model.dim();
        for x in xs {
            check_candidates(x, d, Some(1))?;
        }
        let m = self.model.num_outputs();
        let stacked = Matrix::from_fn(xs.len(), d, |i, j| xs[i][(0, j)]);
        let pp = self.model.point_posteriors(&stacked)?;
        Ok(Matrix::from_fn(xs.len(), m, |i, t| {
            let v = pp.mean[(i, t)];
            match &self.weights {
                Some(w) => v * w[t],
                None => v,
            }
        }))
    }
}

/// Single-output acquisition functions as the outputs of a multi-output
/// acquisition function (BoTorch's `MultiOutputAcquisitionFunctionWrapper`).
#[derive(Debug)]
pub struct MultiOutputAcquisitionFunctionWrapper {
    acqfs: Vec<Arc<dyn AcquisitionFunction>>,
}

impl MultiOutputAcquisitionFunctionWrapper {
    /// `MultiOutputAcquisitionFunctionWrapper(acqfs)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty list and
    /// [`BotorchError::DimensionMismatch`] for different input dimensions.
    pub fn new(acqfs: Vec<Arc<dyn AcquisitionFunction>>) -> Result<Self> {
        let Some(first) = acqfs.first() else {
            return Err(BotorchError::InvalidArgument(
                "MultiOutputAcquisitionFunctionWrapper needs at least one acquisition function"
                    .into(),
            ));
        };
        let d = first.dim();
        if acqfs.iter().any(|a| a.dim() != d) {
            return Err(BotorchError::DimensionMismatch(
                "all acquisition functions must have the same input dimension".into(),
            ));
        }
        Ok(Self { acqfs })
    }

    /// The acquisition functions.
    pub fn acqfs(&self) -> &[Arc<dyn AcquisitionFunction>] {
        &self.acqfs
    }
}

impl MultiOutputAcquisitionFunction for MultiOutputAcquisitionFunctionWrapper {
    fn num_outputs(&self) -> usize {
        self.acqfs.len()
    }

    fn dim(&self) -> usize {
        self.acqfs[0].dim()
    }

    fn forward_many(&self, xs: &[Matrix]) -> Result<Matrix> {
        let cols = self
            .acqfs
            .iter()
            .map(|a| a.forward_many(xs))
            .collect::<Result<Vec<_>>>()?;
        Ok(Matrix::from_fn(xs.len(), cols.len(), |i, t| cols[t][i]))
    }
}
