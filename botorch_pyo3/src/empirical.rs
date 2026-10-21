//! Native numerical helpers for empirical Gaussian processes.

use botorch_core::error::{BotorchError, Result};
use botorch_core::utils::empirical as core;
use botorch_core::{Matrix, Vector};
use pyo3::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayBuf};
use crate::errors::{value_error, IntoPyResult};
use crate::threads;

fn validate_data(array: &Array) -> Result<()> {
    let size = array
        .1
        .iter()
        .try_fold(1usize, |size, &dim| size.checked_mul(dim));
    if size != Some(array.0.len()) {
        return Err(BotorchError::DimensionMismatch(
            "array values do not match the declared shape".into(),
        ));
    }
    Ok(())
}
fn matrices(array: &Array) -> Result<(usize, usize, usize)> {
    validate_data(array)?;
    let shape = &array.1;
    if shape.len() < 2 {
        return Err(BotorchError::DimensionMismatch(
            "expected batch x rows x cols matrix data".into(),
        ));
    }
    Ok((
        shape[..shape.len() - 2].iter().product(),
        shape[shape.len() - 2],
        shape[shape.len() - 1],
    ))
}
fn matrix(array: &Array, batch: usize, rows: usize, cols: usize) -> Matrix {
    Matrix::from_row_slice(
        rows,
        cols,
        &array.0[batch * rows * cols..(batch + 1) * rows * cols],
    )
}
fn append(out: &mut Vec<f64>, matrix: &Matrix) {
    out.extend(matrix_out(matrix).0);
}

#[pyfunction]
#[pyo3(signature=(a,method="svd"))]
fn empirical_orthogonal_basis(py: Python<'_>, a: Array, method: &str) -> PyResult<ArrayBuf> {
    threads::run(py, || {
        let (batch, rows, cols) = matrices(&a)?;
        let mut out = Vec::new();
        for b in 0..batch {
            append(
                &mut out,
                &core::orthogonal_basis(&matrix(&a, b, rows, cols), method)?,
            );
        }
        let mut shape = a.1.clone();
        let axis = shape.len() - 2;
        shape[axis] = rows.min(cols);
        Ok(ArrayBuf(out, shape))
    })
    .into_py_result()
}

#[pyfunction]
#[pyo3(signature=(u1,u2,num_curves,correction=0,diag=false))]
fn empirical_covariance(
    py: Python<'_>,
    u1: Array,
    u2: Array,
    num_curves: usize,
    correction: i64,
    diag: bool,
) -> PyResult<ArrayBuf> {
    threads::run(py, || {
        let (batch, rows, n1) = matrices(&u1)?;
        let (b2, r2, n2) = matrices(&u2)?;
        if (batch, rows) != (b2, r2)
            || u1.1[..u1.1.len() - 2] != u2.1[..u2.1.len() - 2]
            || (diag && n1 != n2)
        {
            return Err(BotorchError::DimensionMismatch(
                "basis dimensions differ".into(),
            ));
        }
        let mut out = Vec::new();
        for b in 0..batch {
            let cov = core::sample_covariance(
                &matrix(&u1, b, rows, n1),
                &matrix(&u2, b, rows, n2),
                num_curves,
                correction,
            )?;
            if diag {
                out.extend((0..n1).map(|i| cov[(i, i)]));
            } else {
                append(&mut out, &cov);
            }
        }
        let mut shape = u1.1[..u1.1.len() - 2].to_vec();
        shape.push(n1);
        if !diag {
            shape.push(n2);
        }
        Ok(ArrayBuf(out, shape))
    })
    .into_py_result()
}

#[pyfunction]
#[pyo3(signature=(u1,u2,gradient,num_curves,correction=0,diag=false))]
fn empirical_covariance_vjp(
    py: Python<'_>,
    u1: Array,
    u2: Array,
    gradient: Array,
    num_curves: usize,
    correction: i64,
    diag: bool,
) -> PyResult<(ArrayBuf, ArrayBuf)> {
    threads::run(py, || {
        let (batch, rows, n1) = matrices(&u1)?;
        let (b2, r2, n2) = matrices(&u2)?;
        validate_data(&gradient)?;
        let width = if diag { Some(n1) } else { n1.checked_mul(n2) }.ok_or_else(|| {
            BotorchError::DimensionMismatch("covariance derivative shape is too large".into())
        })?;
        let mut expected = u1.1[..u1.1.len() - 2].to_vec();
        expected.push(n1);
        if !diag {
            expected.push(n2);
        }
        if (batch, rows) != (b2, r2)
            || u1.1[..u1.1.len() - 2] != u2.1[..u2.1.len() - 2]
            || gradient.1 != expected
            || gradient.0.len() != batch.saturating_mul(width)
            || (diag && n1 != n2)
        {
            return Err(BotorchError::DimensionMismatch(
                "covariance derivative dimensions differ".into(),
            ));
        }
        let (mut g1, mut g2) = (Vec::new(), Vec::new());
        for b in 0..batch {
            let g = if diag {
                Matrix::from_diagonal(&Vector::from_column_slice(
                    &gradient.0[b * width..(b + 1) * width],
                ))
            } else {
                matrix(&gradient, b, n1, n2)
            };
            let (a, z) = core::sample_covariance_vjp(
                &matrix(&u1, b, rows, n1),
                &matrix(&u2, b, rows, n2),
                &g,
                num_curves as f64 - correction as f64,
            )?;
            append(&mut g1, &a);
            append(&mut g2, &z);
        }
        Ok((ArrayBuf(g1, u1.1), ArrayBuf(g2, u2.1)))
    })
    .into_py_result()
}

#[pyfunction]
#[pyo3(signature=(a,min_eigval=0.0))]
fn empirical_project_psd(py: Python<'_>, a: Array, min_eigval: f64) -> PyResult<ArrayBuf> {
    threads::run(py, || {
        let (batch, rows, cols) = matrices(&a)?;
        let mut out = Vec::new();
        for b in 0..batch {
            append(
                &mut out,
                &core::project_psd(&matrix(&a, b, rows, cols), min_eigval)?,
            );
        }
        Ok(ArrayBuf(out, a.1))
    })
    .into_py_result()
}

#[pyfunction]
#[pyo3(signature=(a,gradient,min_eigval=0.0))]
fn empirical_project_psd_vjp(
    py: Python<'_>,
    a: Array,
    gradient: Array,
    min_eigval: f64,
) -> PyResult<ArrayBuf> {
    threads::run(py, || {
        let (batch, rows, cols) = matrices(&a)?;
        validate_data(&gradient)?;
        if gradient.1 != a.1 {
            return Err(BotorchError::DimensionMismatch(
                "PSD gradient shape differs".into(),
            ));
        }
        let mut out = Vec::new();
        for b in 0..batch {
            append(
                &mut out,
                &core::project_psd_vjp(
                    &matrix(&a, b, rows, cols),
                    min_eigval,
                    &matrix(&gradient, b, rows, cols),
                )?,
            );
        }
        Ok(ArrayBuf(out, a.1))
    })
    .into_py_result()
}

#[pyfunction]
fn empirical_shrinkage(
    py: Python<'_>,
    cov: Array,
    target: Array,
    alpha: f64,
) -> PyResult<ArrayBuf> {
    let c = cov.matrix("cov")?;
    let t = target.matrix("target")?;
    threads::run(py, || core::trace_matched_shrinkage(&c, &t, alpha))
        .map(|m| matrix_out(&m).into())
        .into_py_result()
}

#[pyfunction]
fn empirical_shrinkage_vjp(
    py: Python<'_>,
    cov: Array,
    target: Array,
    alpha: f64,
    gradient: Array,
) -> PyResult<(ArrayBuf, ArrayBuf, f64)> {
    let c = cov.matrix("cov")?;
    let t = target.matrix("target")?;
    let g = gradient.matrix("gradient")?;
    threads::run(py, || core::trace_matched_shrinkage_vjp(&c, &t, alpha, &g))
        .map(|(c, t, a)| (matrix_out(&c).into(), matrix_out(&t).into(), a))
        .into_py_result()
}

#[pyfunction]
#[pyo3(signature=(mu,sigma,num_modes=None,correction=0,eigenvalue_tol=0.0))]
fn empirical_em_basis(
    py: Python<'_>,
    mu: Array,
    sigma: Array,
    num_modes: Option<usize>,
    correction: i64,
    eigenvalue_tol: f64,
) -> PyResult<ArrayBuf> {
    validate_data(&mu).into_py_result()?;
    if mu.1.len() != 1 {
        return Err(value_error("expected mu of shape (n,)"));
    }
    let s = sigma.matrix("Sigma")?;
    threads::run(py, || {
        core::em_prior_to_basis_curves(
            &Vector::from_vec(mu.0),
            &s,
            num_modes,
            correction,
            eigenvalue_tol,
        )
    })
    .map(|m| {
        let (v, mut shape) = matrix_out(&m);
        shape.push(1);
        ArrayBuf(v, shape)
    })
    .into_py_result()
}

fn interpolant(x: Array, y: &Array) -> PyResult<core::LinearInterpolation> {
    validate_data(&x).into_py_result()?;
    validate_data(y).into_py_result()?;
    if x.1.len() != 1 || y.1.is_empty() || y.1[y.1.len() - 1] != x.0.len() {
        return Err(value_error(
            "expected 1D knots and curve values with matching final dimension",
        ));
    }
    let n = x.0.len();
    if n == 0 {
        return Err(value_error("interpolation requires at least two knots"));
    }
    core::LinearInterpolation::new(x.0, Matrix::from_row_slice(y.0.len() / n, n, &y.0))
        .into_py_result()
}

#[pyfunction]
#[pyo3(signature=(x,y,x_new,bounds_error=true,fill_value=f64::NAN))]
fn empirical_interp(
    py: Python<'_>,
    x: Array,
    y: Array,
    x_new: Array,
    bounds_error: bool,
    fill_value: f64,
) -> PyResult<ArrayBuf> {
    validate_data(&x_new).into_py_result()?;
    let interp = interpolant(x, &y)?;
    let mut shape = y.1[..y.1.len() - 1].to_vec();
    shape.extend_from_slice(&x_new.1);
    threads::run(py, || interp.evaluate(&x_new.0, bounds_error, fill_value))
        .map(|m| ArrayBuf(matrix_out(&m).0, shape))
        .into_py_result()
}

#[pyfunction]
#[pyo3(signature=(x,y,x_new,gradient,bounds_error=true))]
fn empirical_interp_vjp(
    py: Python<'_>,
    x: Array,
    y: Array,
    x_new: Array,
    gradient: Array,
    bounds_error: bool,
) -> PyResult<(ArrayBuf, ArrayBuf, ArrayBuf)> {
    validate_data(&x_new).into_py_result()?;
    validate_data(&gradient).into_py_result()?;
    let xs = x.1.clone();
    let interp = interpolant(x, &y)?;
    let rows = y.1[..y.1.len() - 1].iter().product();
    let cols = x_new.0.len();
    let mut expected = y.1[..y.1.len() - 1].to_vec();
    expected.extend_from_slice(&x_new.1);
    if gradient.1 != expected || gradient.0.len() != rows * cols {
        return Err(value_error("interpolation gradient dimensions differ"));
    }
    let g = Matrix::from_row_slice(rows, cols, &gradient.0);
    threads::run(py, || interp.vjp(&x_new.0, &g, bounds_error))
        .map(|(xg, yg, qg)| {
            (
                ArrayBuf(xg, xs),
                ArrayBuf(matrix_out(&yg).0, y.1),
                ArrayBuf(qg, x_new.1),
            )
        })
        .into_py_result()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(empirical_orthogonal_basis, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_covariance_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_project_psd, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_project_psd_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_shrinkage, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_shrinkage_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_em_basis, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_interp, m)?)?;
    m.add_function(wrap_pyfunction!(empirical_interp_vjp, m)?)?;
    Ok(())
}
