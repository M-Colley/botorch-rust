//! Monte-Carlo base-sample generation (`botorch.sampling`).
//!
//! [`SobolQMCNormalSampler`] produces the `N(0, I)` base samples `Z` used to
//! draw posterior samples `μ + L zᵀ` by the reparameterization trick. As in
//! BoTorch, the base samples are a deterministic function of the sampler's
//! seed and of the number of jointly sampled points (`q + #pending`, plus the
//! baseline for noisy acquisition functions), and are reused across
//! evaluations: acquisition values are a smooth, deterministic sample-average
//! approximation (SAA) that can be optimized with L-BFGS-B.
//!
//! [`pathwise`] draws functions from GP posteriors (Matheron paths on random
//! Fourier features, `botorch.sampling.pathwise`).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::{BotorchError, Result};
use crate::utils::sampling::{draw_sobol_normal_samples, random_seed, SOBOL_MAX_SAMPLES};
use crate::utils::torch_rng::TorchGenerator;
use crate::Matrix;

pub mod pathwise;

/// Default number of MC samples of BoTorch's `MCAcquisitionFunction`
/// (`MCSamplerMixin._default_sample_shape = torch.Size([512])`).
pub const DEFAULT_MC_SAMPLES: usize = 512;

/// Quasi-MC sampler of standard normal base samples from a scrambled Sobol
/// sequence (BoTorch's `SobolQMCNormalSampler(sample_shape=[num_samples],
/// seed=seed)`).
///
/// Base samples for `dim` jointly sampled points form a `num_samples × dim`
/// matrix (row `s` is MC sample `s`, column `j` belongs to point `j`), drawn
/// with [`draw_sobol_normal_samples`] and cached per `dim`. Explicit base
/// samples can be installed with [`SobolQMCNormalSampler::set_base_samples`]
/// (e.g. to reproduce BoTorch's values exactly).
///
/// With a [`BaseSampleSource`] ([`SobolQMCNormalSampler::with_source`]) the
/// base samples come from the source instead of the Sobol sequence, e.g.
/// i.i.d. normal draws ([`IidNormalSource`], BoTorch's `IIDNormalSampler`,
/// or torch's draws passed in by the Python bindings); they are cached per
/// `dim` in the same way.
#[derive(Debug)]
pub struct SobolQMCNormalSampler {
    num_samples: usize,
    seed: u64,
    cache: RwLock<HashMap<usize, Arc<Matrix>>>,
    source: Option<Arc<dyn BaseSampleSource>>,
}

impl Clone for SobolQMCNormalSampler {
    fn clone(&self) -> Self {
        Self {
            num_samples: self.num_samples,
            seed: self.seed,
            cache: RwLock::new(read_lock(&self.cache).clone()),
            source: self.source.clone(),
        }
    }
}

/// A source of `N(0, 1)` base samples for [`SobolQMCNormalSampler`]: the
/// `num_samples × dim` base samples for `dim` jointly sampled points (as a
/// deterministic function of the source's seed, so that acquisition values
/// are a sample-average approximation).
pub trait BaseSampleSource: Send + Sync + std::fmt::Debug {
    /// The base samples (`num_samples × dim`).
    ///
    /// # Errors
    /// Implementation-specific (e.g. a failing Python callable).
    fn draw(&self, num_samples: usize, dim: usize) -> Result<Matrix>;
}

/// I.i.d. standard normal base samples as BoTorch's `IIDNormalSampler`
/// draws them in double precision: `torch.randn(num_samples x dim)` after
/// `torch.manual_seed(seed)` (PyTorch's CPU generator, reproduced by
/// [`TorchGenerator`]; row-major: all `dim` values of sample 0, then sample
/// 1, ...).
#[derive(Clone, Copy, Debug)]
pub struct IidNormalSource {
    /// The seed of the generator.
    pub seed: u64,
}

impl BaseSampleSource for IidNormalSource {
    fn draw(&self, num_samples: usize, dim: usize) -> Result<Matrix> {
        let data = TorchGenerator::new(self.seed).randn(num_samples * dim);
        Ok(Matrix::from_row_slice(num_samples, dim, &data))
    }
}

fn read_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn write_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

impl SobolQMCNormalSampler {
    /// A sampler drawing `num_samples` samples; `None` draws a random seed
    /// (like BoTorch, which seeds samplers from torch's global RNG).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `1 <= num_samples <= 2¹⁶`.
    pub fn new(num_samples: usize, seed: Option<u64>) -> Result<Self> {
        if num_samples == 0 || num_samples > SOBOL_MAX_SAMPLES {
            return Err(BotorchError::InvalidArgument(format!(
                "num_samples must be in 1..={SOBOL_MAX_SAMPLES}, got {num_samples}"
            )));
        }
        Ok(Self {
            num_samples,
            seed: seed.unwrap_or_else(random_seed),
            cache: RwLock::new(HashMap::new()),
            source: None,
        })
    }

    /// A sampler drawing `num_samples` base samples from `source` (any
    /// positive number of samples; `seed` is reported by
    /// [`SobolQMCNormalSampler::seed`]).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `num_samples == 0`.
    pub fn with_source(
        num_samples: usize,
        seed: Option<u64>,
        source: Arc<dyn BaseSampleSource>,
    ) -> Result<Self> {
        if num_samples == 0 {
            return Err(BotorchError::InvalidArgument(
                "num_samples must be positive".into(),
            ));
        }
        Ok(Self {
            num_samples,
            seed: seed.unwrap_or_else(random_seed),
            cache: RwLock::new(HashMap::new()),
            source: Some(source),
        })
    }

    /// I.i.d. normal base samples ([`IidNormalSource`] seeded with `seed`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `num_samples == 0`.
    pub fn iid(num_samples: usize, seed: Option<u64>) -> Result<Self> {
        let seed = seed.unwrap_or_else(random_seed);
        Self::with_source(num_samples, Some(seed), Arc::new(IidNormalSource { seed }))
    }

    /// Whether the base samples come from a [`BaseSampleSource`] (not the
    /// Sobol sequence).
    pub fn has_source(&self) -> bool {
        self.source.is_some()
    }

    /// A sampler with explicit base samples (`num_samples × dim`) for `dim`
    /// jointly sampled points; other dimensions are drawn from the Sobol
    /// sequence with `seed`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty matrix, non-finite values
    /// or more than `2¹⁶` rows.
    pub fn from_base_samples(base_samples: Matrix, seed: Option<u64>) -> Result<Self> {
        let sampler = Self::new(base_samples.nrows(), seed)?;
        sampler.set_base_samples(base_samples)?;
        Ok(sampler)
    }

    /// Number of MC samples.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// The seed of the Sobol sequence.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The base samples (`num_samples × dim`) for `dim` jointly sampled points.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn base_samples(&self, dim: usize) -> Result<Arc<Matrix>> {
        if let Some(z) = read_lock(&self.cache).get(&dim) {
            return Ok(Arc::clone(z));
        }
        if dim == 0 {
            return Err(BotorchError::InvalidArgument("dim must be positive".into()));
        }
        let z = match &self.source {
            Some(source) => {
                let z = source.draw(self.num_samples, dim)?;
                if z.shape() != (self.num_samples, dim) {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "the base sample source returned {}x{} samples, expected {}x{dim}",
                        z.nrows(),
                        z.ncols(),
                        self.num_samples
                    )));
                }
                Arc::new(z)
            }
            None => Arc::new(draw_sobol_normal_samples(
                dim,
                self.num_samples,
                Some(self.seed),
            )?),
        };
        Ok(Arc::clone(write_lock(&self.cache).entry(dim).or_insert(z)))
    }

    /// Installs explicit base samples for `base_samples.ncols()` points
    /// (BoTorch: assigning `sampler.base_samples`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the number of rows differs from
    /// [`SobolQMCNormalSampler::num_samples`], [`BotorchError::InvalidArgument`]
    /// for an empty matrix or non-finite values.
    pub fn set_base_samples(&self, base_samples: Matrix) -> Result<()> {
        if base_samples.nrows() != self.num_samples {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} base samples, got {}",
                self.num_samples,
                base_samples.nrows()
            )));
        }
        if base_samples.ncols() == 0 || base_samples.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "base samples must be a non-empty matrix of finite values".into(),
            ));
        }
        write_lock(&self.cache).insert(base_samples.ncols(), Arc::new(base_samples));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_samples_are_cached_and_deterministic() {
        let s = SobolQMCNormalSampler::new(64, Some(5)).unwrap();
        let a = s.base_samples(3).unwrap();
        let b = s.base_samples(3).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.shape(), (64, 3));
        let t = SobolQMCNormalSampler::new(64, Some(5)).unwrap();
        assert_eq!(*t.base_samples(3).unwrap(), *a);
        let c = s.clone();
        assert!(Arc::ptr_eq(&c.base_samples(3).unwrap(), &a));
        let explicit = Matrix::from_element(64, 2, 0.25);
        let e = SobolQMCNormalSampler::from_base_samples(explicit.clone(), Some(5)).unwrap();
        assert_eq!(*e.base_samples(2).unwrap(), explicit);
        assert_eq!(*e.base_samples(3).unwrap(), *a);
        assert!(e.set_base_samples(Matrix::zeros(3, 2)).is_err());
        assert!(SobolQMCNormalSampler::new(0, None).is_err());
        assert!(e.base_samples(0).is_err());
    }

    #[test]
    fn iid_source() {
        let s = SobolQMCNormalSampler::iid(100_000, Some(3)).unwrap();
        assert!(s.has_source());
        let z = s.base_samples(2).unwrap();
        assert_eq!(z.shape(), (100_000, 2));
        let mean = z.column(0).mean();
        let var = z.column(0).iter().map(|v| v * v).sum::<f64>() / 1e5;
        assert!(mean.abs() < 0.02 && (var - 1.0).abs() < 0.02);
        let t = SobolQMCNormalSampler::iid(100_000, Some(3)).unwrap();
        assert_eq!(*t.base_samples(2).unwrap(), *z);
        assert!(SobolQMCNormalSampler::iid(0, None).is_err());
    }
}
