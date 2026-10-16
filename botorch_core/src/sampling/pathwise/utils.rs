//! Utilities of the pathwise samplers (`botorch.sampling.pathwise.utils`):
//! the outcome untransform of a path ([`OutcomeUntransform`], BoTorch's
//! `OutcomeUntransformer`) and the random numbers of a Matheron path
//! ([`MatheronPathDraws`]).
//!
//! The input transform of a path is the model's [`Normalize`](crate::Normalize),
//! applied with eval-mode semantics. BoTorch's `InverseLengthscaleTransform`,
//! `SineCosineTransform` and `OutputscaleTransform` are part of
//! [`KernelFeatureMap`](super::features::KernelFeatureMap).

use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

use crate::error::{BotorchError, Result};
use crate::kernels::KernelKind;
use crate::models::SingleTaskGP;
use crate::utils::sampling::draw_sobol_normal_samples;
use crate::Matrix;

/// The outcome untransform of a path: BoTorch's `OutcomeUntransformer` of a
/// `Standardize`, `y = mean + std · f`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutcomeUntransform {
    /// The mean of the standardization.
    pub mean: f64,
    /// The standard deviation of the standardization.
    pub std: f64,
}

impl OutcomeUntransform {
    /// The untransform of the outcome transform of `gp`, if it has one.
    pub fn of(gp: &SingleTaskGP) -> Option<Self> {
        gp.outcome_transform().map(|st| Self {
            mean: st.means()[0],
            std: st.stdvs()[0],
        })
    }

    /// `mean + std · f` (BoTorch's `Standardize.untransform`).
    #[inline]
    pub fn apply(&self, f: f64) -> f64 {
        self.mean + self.std * f
    }
}

/// A seed for stream `stream` derived from `seed` (SplitMix64), so that the
/// random numbers of different outputs, samples and purposes are independent
/// and do not depend on the order in which they are drawn.
pub(crate) fn derive_seed(seed: u64, stream: u64) -> u64 {
    let mut z = seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The standard normal noise draws (`num_samples × n`) of the paths of one
/// output whose draws come from `seed` (stream 3, i.i.d. from a seeded
/// Xoshiro256++ generator).
pub(crate) fn noise_draws(num_samples: usize, n: usize, seed: u64) -> Matrix {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(derive_seed(seed, 3));
    Matrix::from_fn(num_samples, n, |_, _| rng.sample(StandardNormal))
}

/// Checks that the number of Fourier features is even (BoTorch's message).
pub(crate) fn check_num_features(num_features: usize) -> Result<()> {
    if num_features == 0 || num_features % 2 == 1 {
        return Err(BotorchError::Unsupported(format!(
            "Expected an even number of output features, but received \
             num_outputs={num_features}."
        )));
    }
    Ok(())
}

/// The frequencies of the random Fourier features of a stationary kernel
/// (`num_features / 2 × num_inputs`, BoTorch's `KernelFeatureMap.weight`):
/// quasi-random standard normal vectors (the spectral density of the RBF
/// kernel), divided for a Matérn-ν kernel (ν = 1/2, 3/2, 5/2) by the square
/// root of an independent `Gamma(ν, ν)` variable per vector (a multivariate
/// Student-t with `2ν` degrees of freedom), as in BoTorch's
/// `_gen_kernel_features_matern`.
///
/// # Errors
/// [`BotorchError::Unsupported`] for an odd number of features and the
/// errors of [`draw_sobol_normal_samples`].
pub fn draw_fourier_frequencies(
    kind: KernelKind,
    num_inputs: usize,
    num_features: usize,
    seed: u64,
) -> Result<Matrix> {
    check_num_features(num_features)?;
    let half = num_features / 2;
    let mut w = draw_sobol_normal_samples(num_inputs, half, Some(derive_seed(seed, 0)))?;
    if let Some(nu) = kind.nu() {
        let gamma = rand_distr::Gamma::new(nu, 1.0 / nu)
            .map_err(|e| BotorchError::InvalidArgument(e.to_string()))?;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(derive_seed(seed, 1));
        for mut row in w.row_iter_mut() {
            let g: f64 = rng.sample(gamma);
            row *= 1.0 / g.sqrt();
        }
    }
    Ok(w)
}

/// The random numbers of the Matheron paths of one single-output GP:
/// `num_samples` paths with `num_features` Fourier features on `n` training
/// points in `d` dimensions.
///
/// Read from BoTorch's objects, a draw reproduces BoTorch's paths: the
/// frequencies are `paths["prior_paths"].feature_map.weight`, the weights
/// `paths["prior_paths"].weight` and the noise the standard normal values
/// that `gaussian_update` draws with `torch.randn_like` (before scaling by
/// the noise standard deviation).
#[derive(Clone, Debug, PartialEq)]
pub struct MatheronPathDraws {
    /// The frequencies of the Fourier features (`num_features / 2 × d`), in
    /// units of the inverse lengthscales.
    pub frequencies: Matrix,
    /// The weights of the prior paths (`num_samples × num_features`).
    pub weights: Matrix,
    /// Standard normal values of the observation noise of the update
    /// (`num_samples × n`).
    pub noise: Matrix,
}

impl MatheronPathDraws {
    /// Draws the random numbers of `num_samples` paths of a GP with kernel
    /// family `kind` on `n` training points in `d` dimensions from
    /// botorch_core's generators: the frequencies with
    /// [`draw_fourier_frequencies`], the weights as quasi-random standard
    /// normal vectors and the noise i.i.d. standard normal (seeded
    /// Xoshiro256++), each from its own stream of `seed`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `num_samples = 0` and the errors
    /// of [`draw_fourier_frequencies`].
    pub fn sample(
        kind: KernelKind,
        d: usize,
        n: usize,
        num_samples: usize,
        num_features: usize,
        seed: u64,
    ) -> Result<Self> {
        if num_samples == 0 {
            return Err(BotorchError::InvalidArgument(
                "at least one sample path is required".into(),
            ));
        }
        let frequencies = draw_fourier_frequencies(kind, d, num_features, seed)?;
        let weights =
            draw_sobol_normal_samples(num_features, num_samples, Some(derive_seed(seed, 2)))?;
        let noise = noise_draws(num_samples, n, seed);
        Ok(Self {
            frequencies,
            weights,
            noise,
        })
    }

    /// The number of sample paths.
    pub fn num_samples(&self) -> usize {
        self.weights.nrows()
    }

    /// The number of Fourier features.
    pub fn num_features(&self) -> usize {
        self.weights.ncols()
    }

    /// Checks the shapes against a GP on `n` points in `d` dimensions.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes and
    /// [`BotorchError::Unsupported`] for an odd number of features.
    pub fn check(&self, d: usize, n: usize) -> Result<()> {
        check_num_features(self.num_features())?;
        let (s, l) = self.weights.shape();
        if self.frequencies.shape() != (l / 2, d) || self.noise.shape() != (s, n) || s == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "Matheron path draws with weights of shape {s}x{l} need frequencies of shape \
                 {}x{d} and noise of shape {s}x{n}, got {}x{} and {}x{}",
                l / 2,
                self.frequencies.nrows(),
                self.frequencies.ncols(),
                self.noise.nrows(),
                self.noise.ncols()
            )));
        }
        if self
            .frequencies
            .iter()
            .chain(self.weights.iter())
            .chain(self.noise.iter())
            .any(|v| !v.is_finite())
        {
            return Err(BotorchError::InvalidArgument(
                "Matheron path draws must be finite".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_are_seeded_and_shaped() {
        let a = MatheronPathDraws::sample(KernelKind::Rbf, 3, 7, 4, 16, 11).unwrap();
        assert_eq!(a.frequencies.shape(), (8, 3));
        assert_eq!(a.weights.shape(), (4, 16));
        assert_eq!(a.noise.shape(), (4, 7));
        a.check(3, 7).unwrap();
        assert!(a.check(2, 7).is_err());
        let b = MatheronPathDraws::sample(KernelKind::Rbf, 3, 7, 4, 16, 11).unwrap();
        assert_eq!(a, b);
        let c = MatheronPathDraws::sample(KernelKind::Rbf, 3, 7, 4, 16, 12).unwrap();
        assert_ne!(a.weights, c.weights);
        assert!(draw_fourier_frequencies(KernelKind::Rbf, 2, 7, 0).is_err());
        assert!(MatheronPathDraws::sample(KernelKind::Rbf, 3, 7, 0, 16, 1).is_err());
        // The Matérn frequencies are the RBF frequencies scaled per row.
        let m = draw_fourier_frequencies(KernelKind::Matern52, 3, 16, 5).unwrap();
        let r = draw_fourier_frequencies(KernelKind::Rbf, 3, 16, 5).unwrap();
        for i in 0..8 {
            let ratio = m[(i, 0)] / r[(i, 0)];
            assert!(ratio > 0.0);
            for j in 1..3 {
                assert!((m[(i, j)] / r[(i, j)] - ratio).abs() < 1e-12 * ratio);
            }
        }
    }

    #[test]
    fn matern_frequencies_have_student_t_moments() {
        // The second moment of a Student-t with 2ν = 5 degrees of freedom is
        // 5 / 3 per coordinate (that of the RBF frequencies is 1).
        let w = draw_fourier_frequencies(KernelKind::Matern52, 2, 2 * 8192, 3).unwrap();
        let m2 = w.iter().map(|v| v * v).sum::<f64>() / w.len() as f64;
        assert!((m2 - 5.0 / 3.0).abs() < 0.1, "{m2}");
        let r = draw_fourier_frequencies(KernelKind::Rbf, 2, 2 * 8192, 3).unwrap();
        let r2 = r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64;
        assert!((r2 - 1.0).abs() < 0.01, "{r2}");
    }
}
