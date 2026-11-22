//! Quasi-random sampling mirroring `botorch.utils.sampling`:
//! [`draw_sobol_samples`] (scrambled Sobol points in a box) and
//! [`draw_sobol_normal_samples`] (quasi-random standard normal samples);
//! uniform samples from the simplex and the unit sphere
//! ([`sample_simplex`], [`sample_hypersphere`]).
//!
//! [`SobolEngine`] reproduces PyTorch's `torch.quasirandom.SobolEngine(...,
//! scramble=True, seed=seed)` bit for bit up to `SOBOL_MAX_DIM = 21_201`
//! dimensions: 30-bit Joe–Kuo D(6) direction numbers scrambled by Matoušek's
//! linear matrix scramble and digital shift drawn from [`TorchGenerator`].
//! Dimensions beyond `21_201` wrap in blocks of `21_201` with per-block seeds.

use std::collections::hash_map::RandomState;
use std::f64::consts::SQRT_2;
use std::hash::{BuildHasher, Hasher};

use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use statrs::function::erf::erf_inv;

use crate::error::{BotorchError, Result};
use crate::utils::torch_rng::TorchGenerator;
use crate::Matrix;

/// Maximum number of points of a [`SobolEngine`] sequence (`2¹⁶`).
pub const SOBOL_MAX_SAMPLES: usize = 1 << 16;

/// Number of Joe–Kuo D(6) dimensions matching `torch.quasirandom.SobolEngine.MAXDIM`.
pub const SOBOL_MAX_DIM: usize = 21_201;

const MAXBIT: usize = 30;
const SCALE_F32: f32 = 1.0 / ((1u32 << MAXBIT) as f32);
const SCALE_F64: f64 = 1.0 / ((1u64 << MAXBIT) as f64);

/// Packed Joe–Kuo D(6) polynomials and initial direction numbers for dimensions `1..21_201`
/// (3-clause BSD, S. Joe & F. Y. Kuo, ACM TOMS 34(2), 2008).
static SOBOL_JOE_KUO: &[u8] = include_bytes!("sobol_joe_kuo.bin");

/// A fresh random seed (from the standard library's randomly keyed hasher),
/// used when no seed is given, like BoTorch's `torch.randint` seed draw.
pub fn random_seed() -> u64 {
    let mut h = RandomState::new().build_hasher();
    h.write_u64(0x5eed_5eed_5eed_5eed);
    h.finish()
}

/// Unscrambled 30-bit Joe–Kuo D(6) direction numbers for `dimension` (`0..21_201`).
fn joe_kuo_direction_numbers(dimension: usize) -> [u32; MAXBIT] {
    debug_assert!(dimension < SOBOL_MAX_DIM);
    if dimension == 0 {
        let mut v = [0u32; MAXBIT];
        for (k, vk) in v.iter_mut().enumerate() {
            *vk = 1u32 << (MAXBIT - 1 - k);
        }
        return v;
    }
    let header_pos = (dimension - 1) * 4;
    let entry = u32::from_le_bytes([
        SOBOL_JOE_KUO[header_pos],
        SOBOL_JOE_KUO[header_pos + 1],
        SOBOL_JOE_KUO[header_pos + 2],
        SOBOL_JOE_KUO[header_pos + 3],
    ]);
    let deg = (entry >> 27) as usize;
    let mut pos = (entry & 0x07ff_ffff) as usize;
    let a = if deg == 1 {
        0u32
    } else if deg <= 9 {
        let val = u32::from(SOBOL_JOE_KUO[pos]);
        pos += 1;
        val
    } else if deg <= 17 {
        let val = u32::from(u16::from_le_bytes([
            SOBOL_JOE_KUO[pos],
            SOBOL_JOE_KUO[pos + 1],
        ]));
        pos += 2;
        val
    } else {
        let val = u32::from_le_bytes([
            SOBOL_JOE_KUO[pos],
            SOBOL_JOE_KUO[pos + 1],
            SOBOL_JOE_KUO[pos + 2],
            0,
        ]);
        pos += 3;
        val
    };

    let mut m = [0u32; MAXBIT];
    m[0] = 1;
    for k in 2..=deg {
        let half = if k <= 9 {
            let val = u32::from(SOBOL_JOE_KUO[pos]);
            pos += 1;
            val
        } else if k <= 17 {
            let val = u32::from(u16::from_le_bytes([
                SOBOL_JOE_KUO[pos],
                SOBOL_JOE_KUO[pos + 1],
            ]));
            pos += 2;
            val
        } else {
            let val = u32::from_le_bytes([
                SOBOL_JOE_KUO[pos],
                SOBOL_JOE_KUO[pos + 1],
                SOBOL_JOE_KUO[pos + 2],
                0,
            ]);
            pos += 3;
            val
        };
        m[k - 1] = (half << 1) | 1;
    }
    for k in deg..MAXBIT {
        let mut val = (m[k - deg] << deg) ^ m[k - deg];
        for i in 1..deg {
            if ((a >> (deg - 1 - i)) & 1) != 0 {
                val ^= m[k - i] << i;
            }
        }
        m[k] = val;
    }
    let mut v = [0u32; MAXBIT];
    for k in 0..MAXBIT {
        v[k] = m[k] << (MAXBIT - 1 - k);
    }
    v
}

#[inline]
fn quasi_to_f64(quasi: u32, index: usize) -> f64 {
    if index == 0 {
        // PyTorch computes `_first_point = self.quasi / 2**30` in default
        // float32 precision and casts it to float64 on `draw(dtype=float64)`.
        f64::from((quasi as f32) * SCALE_F32)
    } else {
        f64::from(quasi) * SCALE_F64
    }
}

/// A seeded, Matoušek-scrambled Sobol sequence in `[0, 1)^dim` reproducing
/// `torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SobolEngine {
    dim: usize,
    seed: u64,
    shift: Vec<u32>,
    state: Vec<[u32; MAXBIT]>,
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
        let seed = seed.unwrap_or_else(random_seed);
        let mut shift = Vec::with_capacity(dim);
        let mut state = Vec::with_capacity(dim);

        let mut offset = 0usize;
        let mut block = 0u64;
        while offset < dim {
            let chunk = (dim - offset).min(SOBOL_MAX_DIM);
            let mut g = TorchGenerator::new(seed.wrapping_add(block));
            for _ in 0..chunk {
                let mut s = 0u32;
                for b in 0..MAXBIT {
                    s |= (g.random() & 1) << b;
                }
                shift.push(s);
            }
            for j in 0..chunk {
                let mut ltm_rows = [0u32; MAXBIT];
                for (r, row_mask) in ltm_rows.iter_mut().enumerate() {
                    let mut mask = 1u32 << (MAXBIT - 1 - r);
                    for c in 0..MAXBIT {
                        let bit = g.random() & 1;
                        if c < r && bit != 0 {
                            mask |= 1u32 << (MAXBIT - 1 - c);
                        }
                    }
                    *row_mask = mask;
                }
                let raw_v = joe_kuo_direction_numbers(j);
                let mut scram = [0u32; MAXBIT];
                for (k, &vk) in raw_v.iter().enumerate() {
                    let mut acc = 0u32;
                    for (r, &mask) in ltm_rows.iter().enumerate() {
                        let parity = (vk & mask).count_ones() & 1;
                        acc |= parity << (MAXBIT - 1 - r);
                    }
                    scram[k] = acc;
                }
                state.push(scram);
            }
            offset += chunk;
            block += 1;
        }

        Ok(Self {
            dim,
            seed,
            shift,
            state,
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

    /// Coordinate `dimension` of point `index`, in `[0, 1)`.
    ///
    /// # Panics
    /// If `index >= 2¹⁶` or `dimension >= dim`.
    pub fn sample(&self, index: usize, dimension: usize) -> f64 {
        assert!(index < SOBOL_MAX_SAMPLES, "Sobol index out of range");
        assert!(dimension < self.dim, "Sobol dimension out of range");
        let mut q = self.shift[dimension];
        let sd = &self.state[dimension];
        let mut gray = (index ^ (index >> 1)) as u32;
        while gray != 0 {
            let b = gray.trailing_zeros() as usize;
            q ^= sd[b];
            gray &= gray - 1;
        }
        quasi_to_f64(q, index)
    }

    /// Points `start, ..., start + n - 1` as an `n × dim` matrix.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `start + n > 2¹⁶`.
    pub fn draw(&self, n: usize, start: usize) -> Result<Matrix> {
        if start > SOBOL_MAX_SAMPLES || n > SOBOL_MAX_SAMPLES - start {
            return Err(BotorchError::InvalidArgument(format!(
                "at most {SOBOL_MAX_SAMPLES} Sobol points are supported, requested {n} starting at {start}"
            )));
        }
        if n == 0 {
            return Ok(Matrix::zeros(0, self.dim));
        }
        let mut quasi = self.shift.clone();
        let mut gray = (start ^ (start >> 1)) as u32;
        while gray != 0 {
            let b = gray.trailing_zeros() as usize;
            for (qd, sd) in quasi.iter_mut().zip(&self.state) {
                *qd ^= sd[b];
            }
            gray &= gray - 1;
        }
        let mut out = Matrix::zeros(n, self.dim);
        for d in 0..self.dim {
            out[(0, d)] = quasi_to_f64(quasi[d], start);
        }
        for row in 1..n {
            let prev = (start + row - 1) as u32;
            let j = (!prev).trailing_zeros() as usize;
            for (d, (qd, sd)) in quasi.iter_mut().zip(&self.state).enumerate() {
                *qd ^= sd[j];
                out[(row, d)] = f64::from(*qd) * SCALE_F64;
            }
        }
        Ok(out)
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
    Ok(engine.draw(n, 0)?.map(|u| {
        let v = 0.5 + (1.0 - f64::EPSILON) * (u - 0.5);
        erf_inv(2.0 * v - 1.0) * SQRT_2
    }))
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

/// `n` points drawn uniformly from the probability simplex `{x ∈ [0, 1]^d :
/// Σ_j x_j = 1}` (BoTorch's `sample_simplex(d, n, qmc, seed)`): the spacings
/// of `d - 1` sorted uniform random numbers ([`simplex_from_uniform`]),
/// which are i.i.d. from a seeded Xoshiro256++ generator, or the points of a
/// scrambled Sobol sequence with `qmc` ([`SobolEngine`]). For `d = 1` every
/// point is `1`.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `d == 0` or, with `qmc`, `n > 2¹⁶`.
pub fn sample_simplex(d: usize, n: usize, qmc: bool, seed: Option<u64>) -> Result<Matrix> {
    if d == 0 {
        return Err(BotorchError::InvalidArgument(
            "the dimension of the simplex must be positive".into(),
        ));
    }
    if d == 1 {
        return Ok(Matrix::from_element(n, 1, 1.0));
    }
    let u = if qmc {
        SobolEngine::new(d - 1, seed)?.draw(n, 0)?
    } else {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.unwrap_or_else(random_seed));
        Matrix::from_fn(n, d - 1, |_, _| rng.random::<f64>())
    };
    Ok(simplex_from_uniform(&u))
}

/// The simplex points of `sample_simplex` from `n × (d - 1)` uniform random
/// numbers: each row is sorted, framed by 0 and 1, and differenced, giving
/// `n × d` points that sum to 1.
pub fn simplex_from_uniform(u: &Matrix) -> Matrix {
    let (n, k) = u.shape();
    let mut out = Matrix::zeros(n, k + 1);
    let mut row: Vec<f64> = Vec::with_capacity(k);
    for i in 0..n {
        row.clear();
        row.extend(u.row(i).iter());
        // torch.sort: NaNs last.
        row.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
        });
        let mut prev = 0.0;
        for (j, &v) in row.iter().enumerate() {
            out[(i, j)] = v - prev;
            prev = v;
        }
        out[(i, k)] = 1.0 - prev;
    }
    out
}

/// `n` points drawn uniformly from the unit sphere in `d` dimensions
/// (BoTorch's `sample_hypersphere(d, n, qmc, seed)`): standard normal
/// vectors normalized to unit length ([`hypersphere_from_normal`]), i.i.d.
/// from a seeded Xoshiro256++ generator or quasi-random with `qmc`
/// ([`draw_sobol_normal_samples`]). For `d = 1` the points are `±1` with equal
/// probability (also with `qmc`, as in BoTorch).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `d == 0` or, with `qmc`, `n > 2¹⁶`.
pub fn sample_hypersphere(d: usize, n: usize, qmc: bool, seed: Option<u64>) -> Result<Matrix> {
    if d == 0 {
        return Err(BotorchError::InvalidArgument(
            "the dimension of the hypersphere must be positive".into(),
        ));
    }
    let seed = seed.unwrap_or_else(random_seed);
    if d == 1 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        return Ok(Matrix::from_fn(n, 1, |_, _| {
            if rng.random::<bool>() {
                1.0
            } else {
                -1.0
            }
        }));
    }
    let z = if qmc {
        draw_sobol_normal_samples(d, n, Some(seed))?
    } else {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        Matrix::from_fn(n, d, |_, _| rng.sample(StandardNormal))
    };
    Ok(hypersphere_from_normal(&z))
}

/// The sphere points of `sample_hypersphere` from `n × d` normal samples:
/// every row divided by its Euclidean norm.
pub fn hypersphere_from_normal(z: &Matrix) -> Matrix {
    let mut out = z.clone();
    for mut row in out.row_iter_mut() {
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        row /= norm;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_matches_pytorch_sobol_reference() {
        // Exact bit-for-bit reference outputs from `torch.quasirandom.SobolEngine(4, scramble=True, seed=42).draw(2, dtype=torch.float64)`.
        let engine = SobolEngine::new(4, Some(42)).unwrap();
        assert_eq!(
            engine.shift,
            vec![1_071_071_778, 112_062_556, 883_666_784, 450_361_855]
        );
        assert_eq!(
            &engine.state[0][..6],
            &[
                590_237_441,
                286_241_853,
                190_587_803,
                95_940_033,
                46_732_757,
                22_205_704
            ]
        );
        let pts = engine.draw(4, 0).unwrap();
        let expected_r0 = [
            0.997_513_353_824_615_5,
            0.104_366_391_897_201_54,
            0.822_978_854_179_382_3,
            0.419_432_163_238_525_4,
        ];
        let expected_r1 = [
            0.452_700_885_944_068_43,
            0.838_067_288_510_501_4,
            0.073_270_472_697_913_65,
            0.616_169_951_856_136_3,
        ];
        for (j, (&e0, &e1)) in expected_r0.iter().zip(&expected_r1).enumerate() {
            assert_eq!(pts[(0, j)], e0);
            assert_eq!(pts[(1, j)], e1);
            assert_eq!(engine.sample(0, j), e0);
            assert_eq!(engine.sample(1, j), e1);
            assert_eq!(engine.sample(3, j), pts[(3, j)]);
        }
        // Chunked draws equal a single contiguous draw.
        let p_a = engine.draw(2, 0).unwrap();
        let p_b = engine.draw(2, 2).unwrap();
        assert_eq!(pts.rows(0, 2), p_a);
        assert_eq!(pts.rows(2, 2), p_b);
    }

    #[test]
    fn max_dimension_and_normal_qmc_match_pytorch_reference() {
        // Exact bit-for-bit reference at the highest Joe-Kuo D(6) dimension (21_201, index 21_200)
        // from `torch.quasirandom.SobolEngine(21201, scramble=True, seed=99).draw(3, dtype=float64)`.
        let engine = SobolEngine::new(SOBOL_MAX_DIM, Some(99)).unwrap();
        let pts = engine.draw(3, 0).unwrap();
        let expected_last_col = [
            0.381_288_975_477_218_63,
            0.711_793_801_747_262_5,
            0.971_440_812_572_836_9,
        ];
        for (r, &expected) in expected_last_col.iter().enumerate() {
            assert_eq!(pts[(r, SOBOL_MAX_DIM - 1)], expected);
            assert_eq!(engine.sample(r, SOBOL_MAX_DIM - 1), expected);
        }

        // Dimensions beyond SOBOL_MAX_DIM wrap cleanly with per-block seeds.
        let wrapped = SobolEngine::new(SOBOL_MAX_DIM + 4, Some(99)).unwrap();
        let wpts = wrapped.draw(2, 0).unwrap();
        assert_eq!(wpts.shape(), (2, SOBOL_MAX_DIM + 4));
        assert_eq!(wpts[(0, SOBOL_MAX_DIM - 1)], expected_last_col[0]);
        assert!(wpts.row(1).iter().all(|&v| (0.0..1.0).contains(&v)));

        // `draw_sobol_normal_samples` matches BoTorch's `NormalQMCEngine(inv_transform=True)`.
        let zn = draw_sobol_normal_samples(3, 4, Some(77)).unwrap();
        let expected_zn = [
            [
                2.044_215_144_495_239_7,
                0.468_129_882_974_111_7,
                0.994_212_832_947_359_5,
            ],
            [
                -0.768_221_066_248_545_4,
                -0.059_962_609_053_906_275,
                -0.429_420_403_491_839_13,
            ],
        ];
        for (r, row) in expected_zn.iter().enumerate() {
            for (c, &expected) in row.iter().enumerate() {
                assert!((zn[(r, c)] - expected).abs() <= 1e-14);
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
