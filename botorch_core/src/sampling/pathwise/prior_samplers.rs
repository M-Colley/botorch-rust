//! Prior paths from random Fourier features
//! (`botorch.sampling.pathwise.prior_samplers`).

use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::{Model, SingleTaskGP};
use crate::sampling::pathwise::features::KernelFeatureMap;
use crate::sampling::pathwise::paths::{GeneralizedLinearPath, PathList, SamplePath};
use crate::sampling::pathwise::posterior_samplers::{output_draws, MatheronPathOptions};
use crate::sampling::pathwise::utils::OutcomeUntransform;
use crate::Matrix;

/// The prior paths of one single-output GP: a Bayesian linear model on the
/// random Fourier features of its kernel with the given `frequencies`
/// (`L/2 × d`) and `weights` (`num_samples × L`), the model's constant mean
/// as bias and its input transform; with `untransform`, the values are
/// untransformed with the model's outcome transform (BoTorch's
/// `_draw_kernel_feature_paths_ExactGP`; `draw_matheron_paths` removes the
/// outcome transform from the prior paths).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for draws of the wrong shape and
/// errors of the kernel.
pub fn kernel_feature_path(
    gp: &SingleTaskGP,
    frequencies: Matrix,
    weights: Matrix,
    untransform: bool,
) -> Result<GeneralizedLinearPath> {
    let feature_map = KernelFeatureMap::new(gp.kernel()?, frequencies)?;
    GeneralizedLinearPath::new(
        Arc::new(feature_map),
        weights,
        Some(gp.constant()),
        gp.input_transform().cloned(),
        if untransform {
            OutcomeUntransform::of(gp)
        } else {
            None
        },
    )
}

/// Draws `num_samples` functions from the random-Fourier-feature
/// approximation of the GP prior of every output of `model` (BoTorch's
/// `draw_kernel_feature_paths(model, sample_shape=[num_samples])`): one
/// [`GeneralizedLinearPath`] per output (in output order, the sub-models of a
/// `ModelListGP` flattened), with the model's constant mean, input transform
/// and outcome untransform.
///
/// The frequencies and weights are those of
/// [`MatheronPathDraws`](super::utils::MatheronPathDraws) (explicit in
/// `options.draws`, else drawn from `options.seed`), so that the prior paths
/// of [`draw_matheron_paths`](super::posterior_samplers::draw_matheron_paths)
/// with the same options are these paths.
///
/// # Errors
/// [`BotorchError::Unsupported`] for models without independent exact GP
/// outputs (batched models), [`BotorchError::InvalidArgument`] for
/// `num_samples = 0` and errors of the draws.
pub fn draw_kernel_feature_paths(
    model: &dyn Model,
    num_samples: usize,
    options: &MatheronPathOptions,
) -> Result<PathList> {
    let gps = model_gps(model)?;
    let draws = output_draws(&gps, num_samples, options)?;
    let paths = gps
        .iter()
        .zip(draws)
        .map(|(gp, d)| {
            kernel_feature_path(gp, d.frequencies, d.weights, true)
                .map(|p| Arc::new(p) as Arc<dyn SamplePath>)
        })
        .collect::<Result<Vec<_>>>()?;
    PathList::new(paths)
}

/// The independent single-output GPs of an unbatched model.
pub(crate) fn model_gps(model: &dyn Model) -> Result<Vec<&SingleTaskGP>> {
    if !model.batch_shape().is_empty() {
        return Err(BotorchError::Unsupported(format!(
            "pathwise sampling of batched models (batch shape {:?}) is not supported",
            model.batch_shape()
        )));
    }
    model.output_gps().ok_or_else(|| {
        BotorchError::Unsupported(
            "pathwise sampling requires a model of independent exact GPs (a SingleTaskGP or a \
             ModelListGP)"
                .into(),
        )
    })
}
