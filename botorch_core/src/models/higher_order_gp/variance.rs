//! Marginal conditional variances from explicitly supplied Kronecker factors.

use super::{kron_vec, Tensor};
use crate::error::{BotorchError, Result};
use crate::linalg::psd_safe_cholesky;
use crate::{Matrix, Vector};
use nalgebra::SymmetricEigen;

fn validate(factors: &[Matrix], train: usize, noise: f64) -> Result<()> {
    if factors.is_empty() || train == 0 || factors[0].nrows() < train {
        return Err(BotorchError::DimensionMismatch(
            "Kronecker factors need a data factor containing the positive training size".into(),
        ));
    }
    if factors
        .iter()
        .any(|f| f.nrows() == 0 || f.nrows() != f.ncols())
    {
        return Err(BotorchError::DimensionMismatch(
            "Kronecker factors must be nonempty square matrices".into(),
        ));
    }
    if !noise.is_finite() || noise < 0.0 || factors.iter().any(|f| f.iter().any(|v| !v.is_finite()))
    {
        return Err(BotorchError::InvalidArgument(
            "Kronecker factors must be finite and noise finite/nonnegative".into(),
        ));
    }
    Ok(())
}

fn parts(factors: &[Matrix], train: usize) -> (Vec<Matrix>, Vec<Matrix>, Vec<Vector>) {
    let q = factors[0].nrows() - train;
    let mut training = Vec::with_capacity(factors.len());
    let mut cross = Vec::with_capacity(factors.len());
    let mut diagonals = Vec::with_capacity(factors.len());
    for (i, f) in factors.iter().enumerate() {
        let a = if i == 0 {
            f.view((0, 0), (train, train)).into_owned()
        } else {
            f.clone()
        };
        training.push((&a + a.transpose()) * 0.5);
        cross.push(if i == 0 {
            f.view((train, 0), (q, train)).into_owned()
        } else {
            f.clone()
        });
        diagonals.push(if i == 0 {
            Vector::from_fn(q, |j, _| f[(train + j, train + j)])
        } else {
            f.diagonal()
        });
    }
    (training, cross, diagonals)
}

/// Flattened point-major marginal conditional variances for a data factor
/// over training+test inputs and the remaining output-mode factors.
/// Training factors are interpreted symmetrically. No model kernel is rebuilt.
///
/// # Errors
/// Returns dimension/finite-value errors or a nonpositive noisy training spectrum.
pub fn posterior_variances_from_factors(
    factors: &[Matrix],
    train: usize,
    noise: f64,
) -> Result<Vector> {
    validate(factors, train, noise)?;
    let (training, cross, diagonals) = parts(factors, train);
    let eigen: Vec<_> = training
        .iter()
        .map(|f| SymmetricEigen::new(f.clone()))
        .collect();
    let values: Vec<_> = eigen.iter().map(|e| e.eigenvalues.clone()).collect();
    let spectrum = kron_vec(&values);
    let denominator: Vec<_> = spectrum.iter().map(|v| v + noise).collect();
    if denominator.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        return Err(BotorchError::InvalidArgument(
            "Noisy Kronecker training covariance must be positive definite".into(),
        ));
    }
    let mut reduction = Tensor::new(
        training.iter().map(Matrix::nrows).collect(),
        denominator.iter().map(|v| 1.0 / v).collect(),
    );
    for (i, (factor, eigen)) in cross.iter().zip(eigen.iter()).enumerate() {
        reduction = reduction.mode_mul(i, &(factor * &eigen.eigenvectors).map(|v| v * v));
    }
    Ok(Vector::from_iterator(
        reduction.data.len(),
        kron_vec(&diagonals)
            .iter()
            .zip(&reduction.data)
            .map(|(a, b)| a - b),
    ))
}

fn kron(factors: &[Matrix]) -> Matrix {
    factors
        .iter()
        .fold(Matrix::from_element(1, 1, 1.0), |a, b| a.kronecker(b))
}

fn indices(mut index: usize, sizes: &[usize]) -> Vec<usize> {
    let mut out = vec![0; sizes.len()];
    for (i, size) in sizes.iter().enumerate().rev() {
        out[i] = index % size;
        index /= size;
    }
    out
}

fn kron_pullback(factors: &[Matrix], adjoint: &Matrix) -> Vec<Matrix> {
    let rows: Vec<_> = factors.iter().map(Matrix::nrows).collect();
    let cols: Vec<_> = factors.iter().map(Matrix::ncols).collect();
    let mut gradients: Vec<_> = factors
        .iter()
        .map(|f| Matrix::zeros(f.nrows(), f.ncols()))
        .collect();
    for i in 0..adjoint.nrows() {
        let ri = indices(i, &rows);
        for j in 0..adjoint.ncols() {
            let ci = indices(j, &cols);
            for (k, gradient) in gradients.iter_mut().enumerate() {
                let mut value = adjoint[(i, j)];
                for (l, f) in factors.iter().enumerate() {
                    if l != k {
                        value *= f[(ri[l], ci[l])];
                    }
                }
                gradient[(ri[k], ci[k])] += value;
            }
        }
    }
    gradients
}

/// Exact first-order adjoints to the supplied factors and scalar noise.
/// The forward variance computation is factorized; this optional pullback
/// materializes the training solve and cross-covariance contractions.
///
/// # Errors
/// Returns input errors, an adjoint size mismatch, or training-factorization errors.
pub fn posterior_variances_factors_vjp(
    factors: &[Matrix],
    train: usize,
    noise: f64,
    adjoint: &Vector,
) -> Result<(Vec<Matrix>, f64)> {
    validate(factors, train, noise)?;
    let (training, cross, diagonals) = parts(factors, train);
    let prior = kron_vec(&diagonals);
    if adjoint.len() != prior.len() {
        return Err(BotorchError::DimensionMismatch(
            "Variance adjoint must have one value per test output".into(),
        ));
    }
    let mut covariance = kron(&training);
    for i in 0..covariance.nrows() {
        covariance[(i, i)] += noise;
    }
    let full_cross = kron(&cross);
    let solve = psd_safe_cholesky(&covariance)?.solve(&full_cross.transpose());
    let weighted = Matrix::from_fn(solve.nrows(), solve.ncols(), |i, j| {
        solve[(i, j)] * adjoint[j]
    });
    let train_adjoint = &weighted * solve.transpose();
    let cross_adjoint = weighted.transpose() * -2.0;
    let train_gradients = kron_pullback(&training, &train_adjoint);
    let cross_gradients = kron_pullback(&cross, &cross_adjoint);
    let mut out: Vec<_> = factors
        .iter()
        .map(|f| Matrix::zeros(f.nrows(), f.ncols()))
        .collect();
    for (i, (a, b)) in train_gradients.iter().zip(&cross_gradients).enumerate() {
        let symmetric = (a + a.transpose()) * 0.5;
        if i == 0 {
            out[i]
                .view_mut((0, 0), (train, train))
                .copy_from(&symmetric);
            out[i].view_mut((train, 0), b.shape()).copy_from(b);
        } else {
            out[i] += symmetric + b;
        }
    }
    let sizes: Vec<_> = diagonals.iter().map(Vector::len).collect();
    for p in 0..prior.len() {
        let idx = indices(p, &sizes);
        for i in 0..factors.len() {
            let mut value = adjoint[p];
            for (j, diagonal) in diagonals.iter().enumerate() {
                if j != i {
                    value *= diagonal[idx[j]];
                }
            }
            let position = idx[i] + if i == 0 { train } else { 0 };
            out[i][(position, position)] += value;
        }
    }
    Ok((out, train_adjoint.trace()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supplied_factor_variances_and_pullbacks_match_dense_and_finite_differences() {
        let factors: Vec<_> = [5, 2, 3]
            .into_iter()
            .map(|n| {
                let a = Matrix::from_fn(n, n, |i, j| ((i * 11 + j * 7 + 3) % 17) as f64 / 19.0);
                &a * a.transpose() + Matrix::identity(n, n) * 0.3
            })
            .collect();
        let (train, noise) = (3, 0.13);
        let (training, cross, diagonals) = parts(&factors, train);
        let a = kron(&training);
        let c = kron(&cross);
        let noisy = &a + Matrix::identity(a.nrows(), a.ncols()) * noise;
        let dense = kron_vec(&diagonals)
            .iter()
            .zip(
                (c.clone() * psd_safe_cholesky(&noisy).unwrap().solve(&c.transpose()))
                    .diagonal()
                    .iter(),
            )
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let actual = posterior_variances_from_factors(&factors, train, noise).unwrap();
        assert!((actual - Vector::from_vec(dense)).amax() < 1e-11);
        let weights = Vector::from_fn(c.nrows(), |i, _| 0.2 + i as f64 * 0.03);
        let (gradient, ng) =
            posterior_variances_factors_vjp(&factors, train, noise, &weights).unwrap();
        let value = |f: &[Matrix], n: f64| {
            posterior_variances_from_factors(f, train, n)
                .unwrap()
                .dot(&weights)
        };
        let h = 1e-6;
        for (k, f) in factors.iter().enumerate() {
            for i in 0..f.nrows() {
                for j in 0..f.ncols() {
                    let mut changed = factors.clone();
                    changed[k][(i, j)] += h;
                    let plus = value(&changed, noise);
                    changed[k][(i, j)] -= 2.0 * h;
                    let fd = (plus - value(&changed, noise)) / (2.0 * h);
                    assert!(
                        (fd - gradient[k][(i, j)]).abs() < 2e-7,
                        "factor{k} ({i},{j}): {fd} {}",
                        gradient[k][(i, j)]
                    );
                }
            }
        }
        let fd = (value(&factors, noise + h) - value(&factors, noise - h)) / (2.0 * h);
        assert!((fd - ng).abs() < 2e-7);
    }
}
