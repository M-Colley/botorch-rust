//! Quasi-random sampling mirroring `botorch.utils.sampling`:
//! [`draw_sobol_samples`] (scrambled Sobol points in a box) and
//! [`draw_sobol_normal_samples`] (quasi-random standard normal samples).
//!
//! The Sobol points come from [`SobolEngine`], an Owen-scrambled Sobol
//! sequence built from the components of the `sobol_burley` crate (Burley's
//! hash-based Owen scrambling with Joe–Kuo direction numbers). It reproduces
//! `sobol_burley::sample` exactly but keeps the full 32-bit scrambled integer,
//! mapping it to the midpoint of its cell, `(k + 1/2) / 2³²`, so that samples
//! lie strictly inside `(0, 1)` with 32 bits of resolution (torch's
//! `SobolEngine` has 30). Up to `2¹⁶` points per sequence are supported;
//! dimensions beyond the 256 of the direction-number table are "padded" with
//! independently seeded (shuffled and scrambled) copies of the first 256
//! dimensions, as recommended by Burley (2020).
//!
//! The values cannot match torch's `SobolEngine` (a different scrambling
//! scheme and random number generator), but they have the same distribution:
//! a randomized (t, m, s)-net when `n = 2^m`.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};

use sobol_burley::parts::{hash, owen_scramble_rev, sobol_rev};
use sobol_burley::NUM_DIMENSIONS;

use crate::error::{BotorchError, Result};
use crate::utils::probability::ndtri;
use crate::Matrix;

/// Maximum number of points of a [`SobolEngine`] sequence (`2¹⁶`).
pub const SOBOL_MAX_SAMPLES: usize = 1 << 16;

/// A fresh random seed (from the standard library's randomly keyed hasher),
/// used when no seed is given, like BoTorch's `torch.randint` seed draw.
pub fn random_seed() -> u64 {
    let mut h = RandomState::new().build_hasher();
    h.write_u64(0x5eed_5eed_5eed_5eed);
    h.finish()
}

/// SplitMix64 finalizer, used to derive 32-bit seeds.
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// One coordinate of an Owen-scrambled Sobol point as a 32-bit integer; this
/// is `sobol_burley::sample` without the final conversion to `f32`.
#[inline]
fn sobol_owen_u32(sample_index: u32, dimension: u32, seed: u32) -> u32 {
    let shuffled_rev_index =
        owen_scramble_rev(sample_index.reverse_bits(), hash(seed ^ 0x79c6_8e4a));
    let sobol = sobol_rev(shuffled_rev_index, dimension);
    let scramble = {
        let seed = seed.wrapping_mul(0x9c8f_2d3b);
        let ds = dimension >> 2;
        ds ^ seed ^ [0x912f_69ba, 0x174f_18ab, 0x691e_72ca, 0xb40c_c1b8][dimension as usize & 0b11]
    };
    owen_scramble_rev(sobol, hash(scramble)).reverse_bits()
}

/// A seeded, Owen-scrambled Sobol sequence in `[0, 1)^dim`
/// (the counterpart of `torch.quasirandom.SobolEngine(dim, scramble=True,
/// seed=seed)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SobolEngine {
    dim: usize,
    seed: u64,
}

impl SobolEngine {
    /// Creates an engine of dimension `dim`; `None` draws a random seed.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn new(dim: usize, seed: Option<u64>) -> Result<Self> {
        if dim == 0 {
            return Err(BotorchError::InvalidArgument(
                "the Sobol dimension must be positive".into(),
            ));
        }
        Ok(Self {
            dim,
            seed: seed.unwrap_or_else(random_seed),
        })
    }

    /// The dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Coordinate `dimension` of point `index`, in `(0, 1)`.
    ///
    /// # Panics
    /// If `index >= 2¹⁶` or `dimension >= dim`.
    pub fn sample(&self, index: usize, dimension: usize) -> f64 {
        assert!(index < SOBOL_MAX_SAMPLES, "Sobol index out of range");
        assert!(dimension < self.dim, "Sobol dimension out of range");
        let nd = NUM_DIMENSIONS as usize;
        let block = (dimension / nd) as u64;
        let seed = (mix64(self.seed ^ mix64(block)) >> 32) as u32;
        let k = sobol_owen_u32(index as u32, (dimension % nd) as u32, seed);
        (f64::from(k) + 0.5) * (1.0 / 4_294_967_296.0)
    }

    /// Points `start, ..., start + n - 1` as an `n × dim` matrix.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `start + n > 2¹⁶`.
    pub fn draw(&self, n: usize, start: usize) -> Result<Matrix> {
        if start + n > SOBOL_MAX_SAMPLES {
            return Err(BotorchError::InvalidArgument(format!(
                "at most {SOBOL_MAX_SAMPLES} Sobol points are supported, requested {}",
                start + n
            )));
        }
        Ok(Matrix::from_fn(n, self.dim, |i, j| {
            self.sample(start + i, j)
        }))
    }
}

/// `n` quasi-random `q`-batches of points in the box `bounds` (`2 × d`, lower
/// bounds in row 0): BoTorch's `draw_sobol_samples(bounds, n, q, seed=seed)`.
/// Element `[i]` is the `q × d` batch `i`; its point `j`, coordinate `k` is
/// `lower_k + u_{i, j·d + k} (upper_k - lower_k)` for a `q·d`-dimensional
/// scrambled Sobol sequence `u`.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] if `bounds` is not `2 × d`,
/// [`BotorchError::InvalidArgument`] for non-finite bounds, `q = 0` or
/// `n > 2¹⁶`.
pub fn draw_sobol_samples(
    bounds: &Matrix,
    n: usize,
    q: usize,
    seed: Option<u64>,
) -> Result<Vec<Matrix>> {
    let d = check_bounds(bounds)?;
    if q == 0 {
        return Err(BotorchError::InvalidArgument("q must be positive".into()));
    }
    let engine = SobolEngine::new(q * d, seed)?;
    let raw = engine.draw(n, 0)?;
    Ok((0..n)
        .map(|i| {
            Matrix::from_fn(q, d, |j, k| {
                raw[(i, j * d + k)] * (bounds[(1, k)] - bounds[(0, k)]) + bounds[(0, k)]
            })
        })
        .collect())
}

/// `n × d` quasi-random samples from `N(0, I_d)` obtained by the inverse
/// transform of a scrambled Sobol sequence (BoTorch's
/// `draw_sobol_normal_samples`, which uses `NormalQMCEngine(inv_transform=True)`).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `d == 0` or `n > 2¹⁶`.
pub fn draw_sobol_normal_samples(d: usize, n: usize, seed: Option<u64>) -> Result<Matrix> {
    let engine = SobolEngine::new(d, seed)?;
    Ok(engine.draw(n, 0)?.map(ndtri))
}

/// Validates `2 × d` box bounds and returns `d`.
pub(crate) fn check_bounds(bounds: &Matrix) -> Result<usize> {
    if bounds.nrows() != 2 || bounds.ncols() == 0 {
        return Err(BotorchError::DimensionMismatch(format!(
            "bounds should be a `2 x d` matrix, got {}x{}",
            bounds.nrows(),
            bounds.ncols()
        )));
    }
    if bounds.iter().any(|v| !v.is_finite()) {
        return Err(BotorchError::InvalidArgument(
            "Currently only finite values in `bounds` are supported.".into(),
        ));
    }
    if (0..bounds.ncols()).any(|k| bounds[(0, k)] > bounds[(1, k)]) {
        return Err(BotorchError::InvalidArgument(
            "Lower bounds cannot exceed upper bounds.".into(),
        ));
    }
    Ok(bounds.ncols())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_reproduces_sobol_burley() {
        for seed in [0u32, 7, 123_456] {
            for dim in [0u32, 1, 5, 255] {
                for i in [0u32, 1, 2, 77, 1023, 65535] {
                    let k = sobol_owen_u32(i, dim, seed);
                    let f = f32::from_bits((k >> 9) | 0x3f80_0000) - 1.0;
                    assert_eq!(f, sobol_burley::sample(i, dim, seed));
                }
            }
        }
    }

    #[test]
    fn sobol_points_are_stratified() {
        // The first 2^m points of each coordinate hit every interval [k/2^m, (k+1)/2^m) once.
        let engine = SobolEngine::new(300, Some(42)).unwrap();
        let x = engine.draw(256, 0).unwrap();
        for j in [0usize, 3, 100, 255, 256, 299] {
            let mut hit = vec![false; 256];
            for i in 0..256 {
                let v = x[(i, j)];
                assert!(v > 0.0 && v < 1.0);
                hit[(v * 256.0) as usize] = true;
            }
            assert!(hit.iter().all(|&h| h), "dimension {j}");
        }
        // The first two coordinates form a (0, m, 2)-net: every 1/16 x 1/16 cell
        // contains exactly one of the 256 points.
        let mut cells = vec![0usize; 256];
        for i in 0..256 {
            let a = (x[(i, 0)] * 16.0) as usize;
            let b = (x[(i, 1)] * 16.0) as usize;
            cells[a * 16 + b] += 1;
        }
        assert!(cells.iter().all(|&c| c == 1));
        // Deterministic given the seed, different across seeds.
        assert_eq!(
            engine.draw(4, 0).unwrap(),
            SobolEngine::new(300, Some(42)).unwrap().draw(4, 0).unwrap()
        );
        assert_ne!(
            engine.draw(4, 0).unwrap(),
            SobolEngine::new(300, Some(43)).unwrap().draw(4, 0).unwrap()
        );
        assert!(engine.draw(2, SOBOL_MAX_SAMPLES - 1).is_err());
        assert!(SobolEngine::new(0, None).is_err());
    }

    #[test]
    fn box_and_normal_samples() {
        let bounds = Matrix::from_row_slice(2, 3, &[-1.0, 0.0, 10.0, 1.0, 0.5, 12.0]);
        let xs = draw_sobol_samples(&bounds, 64, 2, Some(3)).unwrap();
        assert_eq!(xs.len(), 64);
        for x in &xs {
            assert_eq!(x.shape(), (2, 3));
            for j in 0..2 {
                for k in 0..3 {
                    assert!(x[(j, k)] >= bounds[(0, k)] && x[(j, k)] <= bounds[(1, k)]);
                }
            }
        }
        // Coordinate means are close to the box centers (QMC: error ~ 1/n).
        for k in 0..3 {
            let mean: f64 = xs.iter().map(|x| x[(1, k)]).sum::<f64>() / 64.0;
            let center = 0.5 * (bounds[(0, k)] + bounds[(1, k)]);
            assert!((mean - center).abs() < 0.05 * (bounds[(1, k)] - bounds[(0, k)]));
        }
        let z = draw_sobol_normal_samples(4, 1024, Some(11)).unwrap();
        for j in 0..4 {
            let col = z.column(j);
            let m = col.mean();
            let v = col.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / 1023.0;
            assert!(m.abs() < 1e-2, "mean {m}");
            assert!((v - 1.0).abs() < 2e-2, "variance {v}");
        }
        assert!(draw_sobol_samples(&Matrix::zeros(3, 2), 4, 1, None).is_err());
        assert!(draw_sobol_samples(&bounds, 4, 0, None).is_err());
    }
}
