//! Matrix decompositions and interpolation for empirical Gaussian processes.

use nalgebra::{linalg::SVD, SymmetricEigen};

use crate::error::{BotorchError, Result};
use crate::{Matrix, Vector};

fn finite(a: &Matrix) -> Result<()> {
    if a.nrows() == 0 || a.ncols() == 0 || a.iter().any(|v| !v.is_finite()) {
        return Err(BotorchError::InvalidArgument(
            "expected a nonempty finite matrix".into(),
        ));
    }
    Ok(())
}

fn eigen(a: &Matrix) -> Result<SymmetricEigen<f64, nalgebra::Dyn>> {
    finite(a)?;
    if !a.is_square() {
        return Err(BotorchError::DimensionMismatch(
            "expected a square matrix".into(),
        ));
    }
    Ok(SymmetricEigen::new((a + a.transpose()) * 0.5))
}

/// Economy basis B with BᵀB = AᵀA, by SVD or Gram eigendecomposition.
pub fn orthogonal_basis(a: &Matrix, method: &str) -> Result<Matrix> {
    finite(a)?;
    let r = a.nrows().min(a.ncols());
    match method {
        "svd" => {
            let svd = SVD::try_new(a.clone(), false, true, f64::EPSILON, 10000)
                .ok_or_else(|| BotorchError::Numerical("SVD failed to converge".into()))?;
            let vt = svd.v_t.unwrap();
            Ok(Matrix::from_fn(r, a.ncols(), |i, j| {
                svd.singular_values[i] * vt[(i, j)]
            }))
        }
        "eigh" => {
            let eig = eigen(&(a.transpose() * a))?;
            let mut order: Vec<_> = (0..a.ncols()).collect();
            order.sort_by(|&i, &j| eig.eigenvalues[j].total_cmp(&eig.eigenvalues[i]));
            Ok(Matrix::from_fn(r, a.ncols(), |i, j| {
                eig.eigenvalues[order[i]].max(0.0).sqrt() * eig.eigenvectors[(j, order[i])]
            }))
        }
        _ => Err(BotorchError::InvalidArgument(format!(
            "Unknown method: {method:?}. Must be 'svd' or 'eigh'."
        ))),
    }
}

/// U1ᵀU2 divided by the number of curves minus the degrees-of-freedom correction.
pub fn sample_covariance(
    u1: &Matrix,
    u2: &Matrix,
    num_curves: usize,
    correction: i64,
) -> Result<Matrix> {
    if u1.nrows() != u2.nrows() {
        return Err(BotorchError::DimensionMismatch(
            "basis matrices must have the same curve dimension".into(),
        ));
    }
    let denominator = num_curves as f64 - correction as f64;
    if denominator <= 0.0 {
        return Err(BotorchError::InvalidArgument(format!(
            "num_curves ({num_curves}) must be greater than correction ({correction})."
        )));
    }
    Ok(u1.transpose() * u2 / denominator)
}

/// Exact first-order vector-Jacobian product of the sample covariance.
pub fn sample_covariance_vjp(
    u1: &Matrix,
    u2: &Matrix,
    gradient: &Matrix,
    denominator: f64,
) -> Result<(Matrix, Matrix)> {
    if gradient.shape() != (u1.ncols(), u2.ncols())
        || u1.nrows() != u2.nrows()
        || denominator <= 0.0
    {
        return Err(BotorchError::DimensionMismatch(
            "invalid covariance derivative dimensions or denominator".into(),
        ));
    }
    Ok((
        u2 * gradient.transpose() / denominator,
        u1 * gradient / denominator,
    ))
}

/// Minimum-Frobenius projection of a symmetric matrix onto eigenvalues >= floor.
pub fn project_psd(a: &Matrix, floor: f64) -> Result<Matrix> {
    if !floor.is_finite() {
        return Err(BotorchError::InvalidArgument(
            "min_eigval must be finite".into(),
        ));
    }
    let eig = eigen(a)?;
    let scaled = Matrix::from_fn(a.nrows(), a.ncols(), |i, j| {
        eig.eigenvectors[(i, j)] * eig.eigenvalues[j].max(floor)
    });
    let out = scaled * eig.eigenvectors.transpose();
    Ok((&out + out.transpose()) * 0.5)
}

/// Spectral divided-difference derivative of the PSD projection.
pub fn project_psd_vjp(a: &Matrix, floor: f64, gradient: &Matrix) -> Result<Matrix> {
    let eig = eigen(a)?;
    if gradient.shape() != a.shape() {
        return Err(BotorchError::DimensionMismatch(
            "PSD gradient shape differs from matrix".into(),
        ));
    }
    let sym = (gradient + gradient.transpose()) * 0.5;
    let mut local = eig.eigenvectors.transpose() * sym * &eig.eigenvectors;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let x = eig.eigenvalues[i];
            let y = eig.eigenvalues[j];
            let divided = if (x - y).abs() > 1e-12 * (1.0 + x.abs().max(y.abs())) {
                (x.max(floor) - y.max(floor)) / (x - y)
            } else if x > floor {
                1.0
            } else {
                0.0
            };
            local[(i, j)] *= divided;
        }
    }
    Ok(&eig.eigenvectors * local * eig.eigenvectors.transpose())
}

/// Blend a covariance toward a target scaled to the covariance's trace.
pub fn trace_matched_shrinkage(cov: &Matrix, target: &Matrix, alpha: f64) -> Result<Matrix> {
    if !cov.is_square() || cov.shape() != target.shape() || !(0.0..=1.0).contains(&alpha) {
        return Err(BotorchError::InvalidArgument(
            "covariance/target must be equal square matrices and alpha in [0, 1]".into(),
        ));
    }
    Ok(cov * (1.0 - alpha) + target * (alpha * cov.trace() / target.trace().max(1e-12)))
}

/// Exact gradients with respect to covariance, target and scalar shrinkage.
pub fn trace_matched_shrinkage_vjp(
    cov: &Matrix,
    target: &Matrix,
    alpha: f64,
    gradient: &Matrix,
) -> Result<(Matrix, Matrix, f64)> {
    trace_matched_shrinkage(cov, target, alpha)?;
    if gradient.shape() != cov.shape() {
        return Err(BotorchError::DimensionMismatch(
            "shrinkage derivative shape differs".into(),
        ));
    }
    let denominator = target.trace().max(1e-12);
    let ratio = cov.trace() / denominator;
    let inner: f64 = gradient.iter().zip(target.iter()).map(|(a, b)| a * b).sum();
    let gc = gradient * (1.0 - alpha)
        + Matrix::identity(cov.nrows(), cov.ncols()) * (alpha * inner / denominator);
    let mut gt = gradient * (alpha * ratio);
    if target.trace() > 1e-12 {
        gt -= Matrix::identity(cov.nrows(), cov.ncols()) * (alpha * ratio * inner / denominator);
    }
    let ga = gradient
        .iter()
        .zip(target.iter().zip(cov.iter()))
        .map(|(g, (t, c))| g * (ratio * t - c))
        .sum();
    Ok((gc, gt, ga))
}

/// Synthetic curves with the requested mean and covariance, using a Helmert simplex.
pub fn em_prior_to_basis_curves(
    mu: &Vector,
    sigma: &Matrix,
    num_modes: Option<usize>,
    correction: i64,
    tolerance: f64,
) -> Result<Matrix> {
    if mu.len() != sigma.nrows() || num_modes == Some(0) {
        return Err(BotorchError::InvalidArgument(
            "expected matching mean/covariance and num_modes >= 1".into(),
        ));
    }
    let eig = eigen(sigma)?;
    let mut order: Vec<_> = (0..sigma.nrows()).collect();
    order.sort_by(|&i, &j| eig.eigenvalues[j].total_cmp(&eig.eigenvalues[i]));
    order.truncate(num_modes.unwrap_or(order.len()));
    order.retain(|&i| eig.eigenvalues[i] > tolerance);
    let r = order.len();
    if r == 0 {
        return Err(BotorchError::InvalidArgument(
            "Sigma has no eigenvalues above eigenvalue_tol; prior is degenerate".into(),
        ));
    }
    let count = r + 1;
    if correction as f64 >= count as f64 {
        return Err(BotorchError::InvalidArgument(
            "correction must be less than the synthesized curve count".into(),
        ));
    }
    let scale = (count as f64 - correction as f64).sqrt();
    // Orthonormal Helmert contrasts: row k sums to zero and has norm one.
    let simplex = Matrix::from_fn(r, count, |k, j| {
        let den = (((k + 1) * (k + 2)) as f64).sqrt();
        if j <= k {
            1.0 / den
        } else if j == k + 1 {
            -(k as f64 + 1.0) / den
        } else {
            0.0
        }
    });
    let modes = Matrix::from_fn(sigma.nrows(), r, |i, j| {
        eig.eigenvectors[(i, order[j])] * eig.eigenvalues[order[j]].max(0.0).sqrt()
    });
    let deviations = modes * simplex * scale;
    Ok(Matrix::from_fn(count, mu.len(), |i, j| {
        mu[j] + deviations[(j, i)]
    }))
}

/// Piecewise-linear interpolation with sorted, strictly increasing finite knots.
#[derive(Clone, Debug)]
pub struct LinearInterpolation {
    knots: Vec<f64>,
    values: Matrix,
}

impl LinearInterpolation {
    /// Each row of values is a curve on the shared knots.
    pub fn new(knots: Vec<f64>, values: Matrix) -> Result<Self> {
        if knots.len() < 2
            || values.ncols() != knots.len()
            || knots.iter().any(|v| !v.is_finite())
            || knots.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(BotorchError::InvalidArgument("interpolation requires at least two strictly increasing finite knots and matching values".into()));
        }
        Ok(Self { knots, values })
    }
    fn interval(&self, query: f64, bounds_error: bool) -> Result<Option<(usize, f64)>> {
        if query < self.knots[0] || query > self.knots[self.knots.len() - 1] {
            if bounds_error {
                return Err(BotorchError::InvalidArgument(format!(
                    "A value ({query}) in x_new is outside the interpolation range [{}, {}].",
                    self.knots[0],
                    self.knots[self.knots.len() - 1]
                )));
            }
            return Ok(None);
        }
        if query.is_nan() {
            return Ok(None);
        }
        let index = self
            .knots
            .partition_point(|&v| v < query)
            .saturating_sub(1)
            .min(self.knots.len() - 2);
        Ok(Some((
            index,
            (query - self.knots[index]) / (self.knots[index + 1] - self.knots[index]),
        )))
    }
    /// Evaluate every curve at every query (outer, rather than paired, batching).
    pub fn evaluate(&self, query: &[f64], bounds_error: bool, fill_value: f64) -> Result<Matrix> {
        let intervals = query
            .iter()
            .map(|&q| self.interval(q, bounds_error))
            .collect::<Result<Vec<_>>>()?;
        Ok(Matrix::from_fn(
            self.values.nrows(),
            query.len(),
            |i, j| match intervals[j] {
                Some((k, w)) => self.values[(i, k)] * (1.0 - w) + self.values[(i, k + 1)] * w,
                None => fill_value,
            },
        ))
    }
    /// First derivatives with respect to knots, curve values and query locations.
    pub fn vjp(
        &self,
        query: &[f64],
        gradient: &Matrix,
        bounds_error: bool,
    ) -> Result<(Vec<f64>, Matrix, Vec<f64>)> {
        if gradient.shape() != (self.values.nrows(), query.len()) {
            return Err(BotorchError::DimensionMismatch(
                "interpolation derivative shape differs".into(),
            ));
        }
        let mut gx = vec![0.0; self.knots.len()];
        let mut gy = Matrix::zeros(self.values.nrows(), self.knots.len());
        let mut gq = vec![0.0; query.len()];
        for (j, &q) in query.iter().enumerate() {
            if let Some((k, w)) = self.interval(q, bounds_error)? {
                for i in 0..self.values.nrows() {
                    let g = gradient[(i, j)];
                    let slope = (self.values[(i, k + 1)] - self.values[(i, k)])
                        / (self.knots[k + 1] - self.knots[k]);
                    gy[(i, k)] += g * (1.0 - w);
                    gy[(i, k + 1)] += g * w;
                    gq[j] += g * slope;
                    gx[k] += g * slope * (w - 1.0);
                    gx[k + 1] -= g * slope * w;
                }
            }
        }
        Ok((gx, gy, gq))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_basis_covariance_psd_and_em_invariants() {
        for (rows, cols) in [(8, 3), (3, 8), (3, 3)] {
            let a = Matrix::from_fn(rows, cols, |i, j| ((i * 3 + j + 1) as f64).sin());
            for method in ["svd", "eigh"] {
                let b = orthogonal_basis(&a, method).unwrap();
                assert_eq!(b.shape(), (rows.min(cols), cols));
                assert!((b.transpose() * b - a.transpose() * &a).amax() < 1e-11);
            }
        }
        let sigma = Matrix::from_row_slice(3, 3, &[1.0, 0.3, 0.1, 0.3, 0.7, -0.1, 0.1, -0.1, 0.4]);
        let mu = Vector::from_vec(vec![2.0, -1.0, 0.3]);
        for correction in [0, 1, -1] {
            let curves = em_prior_to_basis_curves(&mu, &sigma, None, correction, 1e-12).unwrap();
            let mean = Vector::from_fn(3, |j, _| curves.column(j).sum() / curves.nrows() as f64);
            assert!((&mean - &mu).amax() < 1e-12);
            let centered = Matrix::from_fn(curves.nrows(), 3, |i, j| curves[(i, j)] - mean[j]);
            let cov = sample_covariance(&centered, &centered, curves.nrows(), correction).unwrap();
            assert!((cov - &sigma).amax() < 1e-12);
        }
        let indefinite = Matrix::from_diagonal(&Vector::from_vec(vec![-1.0, 2.0]));
        assert!(
            (project_psd(&indefinite, 0.1).unwrap()
                - Matrix::from_diagonal(&Vector::from_vec(vec![0.1, 2.0])))
            .amax()
                < 1e-12
        );
    }

    #[test]
    fn empirical_interpolation_derivatives_match_finite_differences() {
        let x = vec![0.0, 0.3, 1.0];
        let y = Matrix::from_row_slice(2, 3, &[1.0, 2.0, 0.2, -1.0, 0.3, 1.2]);
        let q = vec![0.1, 0.6, 0.9];
        let g = Matrix::from_fn(2, 3, |i, j| (i + j + 1) as f64 * 0.2);
        let interp = LinearInterpolation::new(x.clone(), y.clone()).unwrap();
        let (gx, gy, gq) = interp.vjp(&q, &g, true).unwrap();
        let objective = |x: Vec<f64>, y: Matrix, q: Vec<f64>| {
            LinearInterpolation::new(x, y)
                .unwrap()
                .evaluate(&q, true, 0.0)
                .unwrap()
                .component_mul(&g)
                .sum()
        };
        for i in 0..x.len() {
            let mut p = x.clone();
            p[i] += 1e-6;
            let mut m = x.clone();
            m[i] -= 1e-6;
            let fd =
                (objective(p, y.clone(), q.clone()) - objective(m, y.clone(), q.clone())) / 2e-6;
            assert!((fd - gx[i]).abs() < 1e-8);
        }
        for i in 0..y.nrows() {
            for j in 0..y.ncols() {
                let mut p = y.clone();
                p[(i, j)] += 1e-6;
                let mut m = y.clone();
                m[(i, j)] -= 1e-6;
                let fd = (objective(x.clone(), p, q.clone()) - objective(x.clone(), m, q.clone()))
                    / 2e-6;
                assert!((fd - gy[(i, j)]).abs() < 1e-8);
            }
        }
        for i in 0..q.len() {
            let mut p = q.clone();
            p[i] += 1e-6;
            let mut m = q.clone();
            m[i] -= 1e-6;
            let fd =
                (objective(x.clone(), y.clone(), p) - objective(x.clone(), y.clone(), m)) / 2e-6;
            assert!((fd - gq[i]).abs() < 1e-8);
        }
        assert!(interp.evaluate(&[-1.0], true, 0.0).is_err());
        assert_eq!(interp.evaluate(&[-1.0], false, 7.0).unwrap()[(0, 0)], 7.0);
    }

    #[test]
    fn empirical_projection_and_shrinkage_derivatives_match_finite_differences() {
        let a = Matrix::from_row_slice(2, 2, &[-0.3, 0.4, 0.4, 1.2]);
        let target = Matrix::from_row_slice(2, 2, &[1.0, 0.2, 0.2, 2.0]);
        let gradient = Matrix::from_row_slice(2, 2, &[0.3, -0.1, 0.2, 0.7]);
        let psd = project_psd_vjp(&a, 0.0, &gradient).unwrap();
        let (gc, gt, ga) = trace_matched_shrinkage_vjp(&a, &target, 0.3, &gradient).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let mut p = a.clone();
                p[(i, j)] += 1e-6;
                let mut m = a.clone();
                m[(i, j)] -= 1e-6;
                let fd = (project_psd(&p, 0.0).unwrap() - project_psd(&m, 0.0).unwrap())
                    .component_mul(&gradient)
                    .sum()
                    / 2e-6;
                assert!((fd - psd[(i, j)]).abs() < 1e-8);
                let fd = (trace_matched_shrinkage(&p, &target, 0.3).unwrap()
                    - trace_matched_shrinkage(&m, &target, 0.3).unwrap())
                .component_mul(&gradient)
                .sum()
                    / 2e-6;
                assert!((fd - gc[(i, j)]).abs() < 1e-8);
                let mut p = target.clone();
                p[(i, j)] += 1e-6;
                let mut m = target.clone();
                m[(i, j)] -= 1e-6;
                let fd = (trace_matched_shrinkage(&a, &p, 0.3).unwrap()
                    - trace_matched_shrinkage(&a, &m, 0.3).unwrap())
                .component_mul(&gradient)
                .sum()
                    / 2e-6;
                assert!((fd - gt[(i, j)]).abs() < 1e-8);
            }
        }
        let fd = (trace_matched_shrinkage(&a, &target, 0.300001).unwrap()
            - trace_matched_shrinkage(&a, &target, 0.299999).unwrap())
        .component_mul(&gradient)
        .sum()
            / 2e-6;
        assert!((fd - ga).abs() < 1e-8);
    }
}
