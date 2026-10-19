//! Quasi-Monte-Carlo sampling of normal distributions
//! (`botorch.sampling.qmc`): the Box–Muller and inverse transforms of
//! `NormalQMCEngine` and the root of the covariance of
//! `MultivariateNormalQMCEngine` (Pagès, *Numerical Probability*, 2018).
//!
//! The uniform points are passed in (torch's scrambled `SobolEngine` when
//! the Python layer has torch, so that the samples equal BoTorch's).

use std::f64::consts::{PI, SQRT_2};

use statrs::function::erf::erf_inv;

use crate::error::{BotorchError, Result};
use crate::Matrix;

/// `NormalQMCEngine.draw`'s transform of `n` uniform points with
/// `sobol_dim` coordinates (row-major) into `n × d` standard normal
/// samples: the inverse transform `erfinv(2 v - 1) √2` of `v = 0.5 + (1 -
/// eps) (u - 0.5)` (`eps` of float64, or of float32 with
/// `single_precision`), or the Box–Muller transform of consecutive pairs of
/// coordinates (`(R cos θ, R sin θ)` with `R = √(-2 log u_{2k})`, `θ = 2π
/// u_{2k+1}`), keeping the first `d` coordinates.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for inconsistent sizes.
pub fn normal_qmc_transform(
    uniform: &[f64],
    n: usize,
    sobol_dim: usize,
    d: usize,
    inv_transform: bool,
    single_precision: bool,
) -> Result<Vec<f64>> {
    if uniform.len() != n * sobol_dim
        || d > sobol_dim
        || (!inv_transform && !sobol_dim.is_multiple_of(2))
    {
        return Err(BotorchError::InvalidArgument(format!(
            "expected {n} x {sobol_dim} uniform points for {d} normal coordinates"
        )));
    }
    let mut out = Vec::with_capacity(n * d);
    for row in uniform.chunks(sobol_dim.max(1)).take(n) {
        if inv_transform {
            let eps = if single_precision {
                f64::from(f32::EPSILON)
            } else {
                f64::EPSILON
            };
            out.extend(row[..d].iter().map(|&u| {
                let v = 0.5 + (1.0 - eps) * (u - 0.5);
                erf_inv(2.0 * v - 1.0) * SQRT_2
            }));
        } else {
            let mut tf = Vec::with_capacity(sobol_dim);
            for pair in row.chunks(2) {
                let r = (-2.0 * pair[0].ln()).sqrt();
                let theta = 2.0 * PI * pair[1];
                tf.push(r * theta.cos());
                tf.push(r * theta.sin());
            }
            out.extend_from_slice(&tf[..d]);
        }
    }
    Ok(out)
}

/// `MultivariateNormalQMCEngine`'s root `R` of the covariance (`Σ = Rᵀ R`):
/// the transposed Cholesky factor, or, if the factorization fails, `(V
/// √λ₊)ᵀ` from the eigendecomposition with eigenvalues below `tol` (1e-8, or
/// 1e-6 with `single_precision`) set to zero.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] with BoTorch's messages for a
/// non-square or non-symmetric covariance (`torch.allclose` tolerances) and
/// eigenvalues below `-tol` ("Covariance matrix not PSD.").
pub fn mvn_qmc_root(cov: &Matrix, single_precision: bool) -> Result<Matrix> {
    if !cov.is_square() {
        return Err(BotorchError::InvalidArgument(
            "Covariance matrix is not square.".into(),
        ));
    }
    let symmetric = cov
        .iter()
        .zip(cov.transpose().iter())
        .all(|(a, b)| (a - b).abs() <= 1e-8 + 1e-5 * b.abs());
    if !symmetric {
        return Err(BotorchError::InvalidArgument(
            "Covariance matrix is not symmetric.".into(),
        ));
    }
    if let Some(ch) = cov.clone().cholesky() {
        return Ok(ch.l().transpose());
    }
    let eig = nalgebra::SymmetricEigen::new(cov.clone());
    let tol = if single_precision { 1e-6 } else { 1e-8 };
    if eig.eigenvalues.iter().any(|&v| v < -tol) {
        return Err(BotorchError::InvalidArgument(
            "Covariance matrix not PSD.".into(),
        ));
    }
    let n = cov.nrows();
    // Ascending eigenvalues like torch.linalg.eigh.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let mut root = Matrix::zeros(n, n);
    for (j, &k) in order.iter().enumerate() {
        let lam = eig.eigenvalues[k];
        let s = if lam > tol { lam.sqrt() } else { 0.0 };
        for r in 0..n {
            // (eigvec * root).T: row j of the result is eigenvector j scaled.
            root[(j, r)] = eig.eigenvectors[(r, k)] * s;
        }
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms() {
        let u = [0.25, 0.5, 0.75, 0.125];
        let bm = normal_qmc_transform(&u, 1, 4, 3, false, false).unwrap();
        let r0 = (-2.0 * 0.25f64.ln()).sqrt();
        assert!((bm[0] - r0 * (PI).cos()).abs() < 1e-15);
        assert!((bm[1] - r0 * (PI).sin()).abs() < 1e-15);
        let inv = normal_qmc_transform(&u, 1, 4, 4, true, false).unwrap();
        assert!(inv[1].abs() < 1e-15);
        assert!((inv[0] - crate::utils::probability::ndtri(0.25)).abs() < 1e-12);
        assert!(normal_qmc_transform(&u, 1, 3, 3, false, false).is_err());
    }

    #[test]
    fn roots() {
        let cov = Matrix::from_row_slice(2, 2, &[1.0, 0.25, 0.25, 2.0]);
        let r = mvn_qmc_root(&cov, false).unwrap();
        assert!((r.transpose() * &r - &cov).abs().max() < 1e-15);
        let singular = Matrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let r = mvn_qmc_root(&singular, false).unwrap();
        assert!((r.transpose() * &r - &singular).abs().max() < 1e-12);
        let bad = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(mvn_qmc_root(&bad, false).is_err());
        let asym = Matrix::from_row_slice(2, 2, &[1.0, 0.5, 0.1, 1.0]);
        assert!(mvn_qmc_root(&asym, false).is_err());
    }
}
