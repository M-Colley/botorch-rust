//! Pathwise updates (`botorch.sampling.pathwise.update_strategies`).

use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::{Likelihood, Model, SingleTaskGP};
use crate::sampling::pathwise::features::KernelEvaluationMap;
use crate::sampling::pathwise::paths::GeneralizedLinearPath;
use crate::sampling::pathwise::prior_samplers::model_gps;
use crate::sampling::pathwise::utils::{derive_seed, noise_draws};
use crate::Matrix;

/// The Gaussian pathwise update of an exact GP (BoTorch's `gaussian_update`
/// for an `ExactGP` with a Gaussian likelihood):
///
/// ```text
/// update_s(·) = k(·, X) (K + Σ)⁻¹ (y - f_s(X) - ε_s),   ε_s = Σ^{1/2} z_s,
/// ```
///
/// where `X` and `y` are the model's transformed training inputs and
/// targets, `K + Σ` its training covariance with the likelihood noise `Σ`
/// (factorized like the model's own posterior, with the same jitter),
/// `f_s(X)` the values of prior paths at `X` (`sample_values`,
/// `num_samples × n`) and `z_s` standard normal draws (`noise`,
/// `num_samples × n`, BoTorch's `torch.randn_like`). The returned path has
/// the kernel evaluated at `X` as features, the update weights
/// `(K + Σ)⁻¹ (y - f_s(X) - ε_s)` (`num_samples × n`) and the model's input
/// transform, and returns values in the transformed outcome space.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for draws of the wrong shape and
/// errors of the model's factorization.
pub fn gaussian_update(
    gp: &SingleTaskGP,
    sample_values: &Matrix,
    noise: &Matrix,
) -> Result<GeneralizedLinearPath> {
    let n = gp.num_train();
    let s = sample_values.nrows();
    if sample_values.ncols() != n || noise.shape() != (s, n) {
        return Err(BotorchError::DimensionMismatch(format!(
            "the sample values and noise of an update on {n} training points must be \
             `num_samples x {n}`, got {}x{} and {}x{}",
            sample_values.nrows(),
            sample_values.ncols(),
            noise.nrows(),
            noise.ncols()
        )));
    }
    // ε = sqrt(noise) z (`DiagLinearOperator.cholesky() @ z`).
    let noise_sd: Vec<f64> = match gp.likelihood() {
        Likelihood::Gaussian(_) => vec![gp.likelihood_noise()[0].sqrt(); n],
        Likelihood::FixedNoise { noise } => noise.iter().map(|v| v.sqrt()).collect(),
    };
    let targets = gp.transformed_train_targets();
    // errors = y - (f(X) + ε), one column per sample.
    let mut errors = Matrix::zeros(n, s);
    for k in 0..s {
        for i in 0..n {
            let noisy = sample_values[(k, i)] + noise_sd[i] * noise[(k, i)];
            errors[(i, k)] = targets[i] - noisy;
        }
    }
    let weights = gp.train_cholesky()?.solve(&errors).transpose();
    let map = KernelEvaluationMap::new(gp.kernel()?, gp.transformed_train_inputs().clone())?;
    GeneralizedLinearPath::new(
        Arc::new(map),
        weights,
        None,
        gp.input_transform().cloned(),
        None,
    )
}

/// The standard normal noise draws of the Gaussian updates of every output
/// of `model` (`num_samples × n` each, `n` the output's number of training
/// points), from the streams of `seed` that
/// [`crate::sampling::pathwise::matheron_path_draws`] uses: for the same
/// seed they are the `noise` of its draws, so that the [`gaussian_update`]s
/// of the prior paths of [`crate::sampling::pathwise::draw_kernel_feature_paths`]
/// with the same seed give the paths of
/// [`crate::sampling::pathwise::draw_matheron_paths`].
///
/// # Errors
/// [`BotorchError::Unsupported`] for batched models or models without
/// independent exact GP outputs.
pub fn draw_update_noise(model: &dyn Model, num_samples: usize, seed: u64) -> Result<Vec<Matrix>> {
    Ok(model_gps(model)?
        .iter()
        .enumerate()
        .map(|(t, gp)| noise_draws(num_samples, gp.num_train(), derive_seed(seed, t as u64)))
        .collect())
}
