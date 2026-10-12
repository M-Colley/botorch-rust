//! Pareto fronts (BoTorch's `gen_pareto_front`) in closed form: evenly spaced
//! points of the ZDT fronts, and quasi-random points on the simplex (DTLZ1)
//! and on the positive orthant of the unit sphere (DTLZ2-4, C2DTLZ2).
//!
//! The quasi-random points come from botorch-rust's scrambled Sobol sequence
//! (`utils::sampling`), transformed like BoTorch's `sample_simplex` and
//! `sample_hypersphere` with `qmc=True`. BoTorch scrambles torch's
//! `SobolEngine` with torch's global random number generator, so the points
//! differ from BoTorch's but have the same distribution; pass a seed for
//! reproducible fronts.

use std::f64::consts::PI;

use crate::error::{BotorchError, Result};
use crate::utils::probability::ndtri;
use crate::utils::sampling::{random_seed, SobolEngine, SOBOL_MAX_SAMPLES};
use crate::Matrix;

use super::dtlz::c2dtlz2_slack_at_objectives;

/// The intervals of `f_0` that make up the disconnected Pareto front of ZDT3.
const ZDT3_PARTS: [(f64, f64); 5] = [
    (0.0, 0.0830015349),
    (0.1822287280, 0.2577623634),
    (0.4093136748, 0.4538821041),
    (0.6183967944, 0.6525117038),
    (0.8233317983, 0.8518328654),
];
/// BoTorch's nugget that keeps the points of ZDT3's front inside the intervals.
const ZDT3_EPS: f64 = 1e-6;

/// `torch.linspace(start, end, steps)` as computed by torch on the CPU: the
/// first half of the points from `start`, the second half from `end`.
pub(super) fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    match steps {
        0 => vec![],
        1 => vec![start],
        _ => {
            let step = (end - start) / (steps - 1) as f64;
            let halfway = steps / 2;
            (0..steps)
                .map(|i| {
                    if i < halfway {
                        start + step * i as f64
                    } else {
                        end - step * (steps - i - 1) as f64
                    }
                })
                .collect()
        }
    }
}

/// Uniform samples from the simplex `{w >= 0: sum w = 1}` in `R^d` from
/// uniform samples `u` (`n × (d - 1)`) like BoTorch's `sample_simplex`: the
/// differences of `(0, sorted(u_i), 1)`.
pub fn simplex_from_uniform(u: &Matrix) -> Matrix {
    let d = u.ncols() + 1;
    let mut out = Matrix::zeros(u.nrows(), d);
    let mut sorted = vec![0.0; d + 1];
    for i in 0..u.nrows() {
        sorted[0] = 0.0;
        for k in 0..d - 1 {
            sorted[k + 1] = u[(i, k)];
        }
        sorted[1..d].sort_by(f64::total_cmp);
        sorted[d] = 1.0;
        for k in 0..d {
            out[(i, k)] = sorted[k + 1] - sorted[k];
        }
    }
    out
}

/// Points on the unit sphere from normal samples `z` (`n × d`) like BoTorch's
/// `sample_hypersphere`: each row divided by its Euclidean norm.
pub fn hypersphere_from_normal(z: &Matrix) -> Matrix {
    let mut out = z.clone();
    for mut row in out.row_iter_mut() {
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        row.iter_mut().for_each(|v| *v /= norm);
    }
    out
}

/// `n` points of a scrambled Sobol sequence in `[0, 1)^dim` (independent
/// sequences of at most `2^16` points, seeded from `seed`, for larger `n`).
fn sobol_uniform(dim: usize, n: usize, seed: u64) -> Result<Matrix> {
    let mut out = Matrix::zeros(n, dim);
    for (b, start) in (0..n).step_by(SOBOL_MAX_SAMPLES).enumerate() {
        let len = (n - start).min(SOBOL_MAX_SAMPLES);
        let engine = SobolEngine::new(dim, Some(seed.wrapping_add(b as u64)))?;
        out.rows_mut(start, len).copy_from(&engine.draw(len, 0)?);
    }
    Ok(out)
}

/// `n` quasi-random points on the positive orthant of the unit sphere in
/// `R^m` (BoTorch's `sample_hypersphere(m, n, qmc=True).abs()`).
fn positive_sphere(m: usize, n: usize, seed: u64) -> Result<Matrix> {
    let z = sobol_uniform(m, n, seed)?.map(ndtri);
    Ok(hypersphere_from_normal(&z).abs())
}

/// The Pareto front of ZDT1 (`zdt = 1`), ZDT2 or ZDT3 with `n` points.
pub(super) fn zdt_front(zdt: usize, n: usize) -> Matrix {
    let f0: Vec<f64> = if zdt == 3 {
        let parts = ZDT3_PARTS.len();
        (0..parts)
            .flat_map(|i| {
                let size = n / parts + usize::from(i < n % parts);
                let (left, right) = ZDT3_PARTS[i];
                linspace(left + ZDT3_EPS, right - ZDT3_EPS, size)
            })
            .collect()
    } else {
        linspace(0.0, 1.0, n)
    };
    Matrix::from_fn(f0.len(), 2, |i, j| {
        let f = f0[i];
        match (j, zdt) {
            (0, _) => f,
            (_, 1) => 1.0 - f.sqrt(),
            (_, 2) => 1.0 - f * f,
            _ => 1.0 - f.sqrt() - f * (10.0 * PI * f).sin(),
        }
    })
}

/// The Pareto front of DTLZ1: `n` points of the simplex `sum f_i = 0.5`.
pub(super) fn dtlz1_front(m: usize, n: usize, seed: Option<u64>) -> Result<Matrix> {
    if m == 1 {
        return Ok(Matrix::from_element(n, 1, 0.5));
    }
    let u = sobol_uniform(m - 1, n, seed.unwrap_or_else(random_seed))?;
    Ok(simplex_from_uniform(&u) * 0.5)
}

/// The Pareto front of DTLZ2-4: `n` points of the unit sphere in the
/// positive orthant.
pub(super) fn sphere_front(m: usize, n: usize, seed: Option<u64>) -> Result<Matrix> {
    positive_sphere(m, n, seed.unwrap_or_else(random_seed))
}

/// The most quasi-random points drawn for the feasible front of C2DTLZ2.
const C2DTLZ2_MAX_DRAWS: usize = 1 << 26;

/// The Pareto front of C2DTLZ2: `n` points of the unit sphere in the positive
/// orthant that satisfy the constraint, i.e. the feasible part of DTLZ2's
/// front (drawn by rejection from quasi-random points on the sphere). BoTorch
/// returns points of the whole sphere, which are partly infeasible.
pub(super) fn c2dtlz2_front(m: usize, n: usize, seed: Option<u64>) -> Result<Matrix> {
    let seed = seed.unwrap_or_else(random_seed);
    let batch = (2 * n).clamp(64, SOBOL_MAX_SAMPLES).next_power_of_two();
    let mut data = Vec::with_capacity(n * m);
    let mut drawn = 0;
    let mut stream = 0u64;
    while data.len() < n * m {
        if drawn >= C2DTLZ2_MAX_DRAWS {
            return Err(BotorchError::Numerical(format!(
                "found only {} of {n} feasible points of the C2DTLZ2 front among {drawn} points",
                data.len() / m
            )));
        }
        // Independent scrambles for every batch (seeds spaced like the
        // streams of `sobol_uniform`).
        let s = positive_sphere(m, batch, seed.wrapping_add(stream << 32))?;
        for row in s.row_iter() {
            let f: Vec<f64> = row.iter().copied().collect();
            if c2dtlz2_slack_at_objectives(&f) >= 0.0 && data.len() < n * m {
                data.extend_from_slice(&f);
            }
        }
        drawn += batch;
        stream += 1;
    }
    Ok(Matrix::from_row_slice(n, m, &data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_torch() {
        assert_eq!(linspace(0.0, 1.0, 0), Vec::<f64>::new());
        assert_eq!(linspace(0.5, 1.0, 1), vec![0.5]);
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        // Both end points are exact.
        let v = linspace(1e-6, 0.0830015349 - 1e-6, 7);
        assert_eq!((v[0], v[6]), (1e-6, 0.0830015349 - 1e-6));
    }

    #[test]
    fn samplers() {
        let u = Matrix::from_row_slice(2, 2, &[0.7, 0.2, 0.1, 0.4]);
        let s = simplex_from_uniform(&u);
        assert_eq!(
            s.row(0).iter().copied().collect::<Vec<_>>(),
            [0.2, 0.7 - 0.2, 1.0 - 0.7]
        );
        assert!(s.row_iter().all(|r| (r.sum() - 1.0).abs() < 1e-15));
        let z = Matrix::from_row_slice(1, 2, &[3.0, -4.0]);
        assert_eq!(hypersphere_from_normal(&z)[(0, 1)], -0.8);
    }

    #[test]
    fn fronts() {
        let f = zdt_front(3, 12);
        assert_eq!(f.nrows(), 12);
        assert_eq!(zdt_front(3, 3).nrows(), 3);
        let s = sphere_front(3, 100, Some(1)).unwrap();
        assert!(s.row_iter().all(|r| (r.norm() - 1.0).abs() < 1e-14));
        assert!(s.iter().all(|&v| v >= 0.0));
        assert_eq!(s, sphere_front(3, 100, Some(1)).unwrap());
        let d = dtlz1_front(4, 50, Some(2)).unwrap();
        assert!(d.row_iter().all(|r| (r.sum() - 0.5).abs() < 1e-15));
        for m in 1..=6 {
            let c = c2dtlz2_front(m, 300, Some(3)).unwrap();
            assert_eq!(c.shape(), (300, m));
            for r in c.row_iter() {
                let f: Vec<f64> = r.iter().copied().collect();
                assert!(c2dtlz2_slack_at_objectives(&f) >= 0.0);
            }
        }
        // More points than one Sobol sequence holds.
        let big = sobol_uniform(2, SOBOL_MAX_SAMPLES + 10, 0).unwrap();
        assert_ne!(big.row(0), big.row(SOBOL_MAX_SAMPLES));
    }
}
