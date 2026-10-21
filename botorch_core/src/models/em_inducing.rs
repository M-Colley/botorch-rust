//! Analytic kernel input derivatives for learned EM inducing locations.

use crate::error::{BotorchError, Result};
use crate::kernels::TreeKernel;
use crate::Matrix;

/// One entry of a kernel matrix derivative when either point set moves.
pub(crate) fn kernel_input_direction(
    kernel: &TreeKernel,
    x: &Matrix,
    y: &Matrix,
    left: Option<(usize, usize)>,
    right: Option<(usize, usize)>,
) -> Result<Matrix> {
    if x.ncols() != y.ncols()
        || left.is_some_and(|(r, c)| r >= x.nrows() || c >= x.ncols())
        || right.is_some_and(|(r, c)| r >= y.nrows() || c >= y.ncols())
    {
        return Err(BotorchError::DimensionMismatch(
            "invalid EM inducing derivative coordinate".into(),
        ));
    }
    let px = kernel.prepare(x)?;
    let py = kernel.prepare(y)?;
    let mut result = Matrix::zeros(x.nrows(), y.nrows());
    if let Some((row, column)) = left {
        let gradient = kernel.cross_with_grad(&px, row, &py).1;
        for j in 0..y.nrows() {
            result[(row, j)] += gradient[(j, column)];
        }
    }
    if let Some((row, column)) = right {
        let gradient = kernel.cross_with_grad(&py, row, &px).1;
        for i in 0..x.nrows() {
            result[(i, row)] += gradient[(i, column)];
        }
    }
    Ok(result)
}

/// Contract a full joint kernel adjoint with the selected inducing rows.
pub(crate) fn inducing_input_vjp(
    kernel: &TreeKernel,
    joint: &Matrix,
    adjoint: &Matrix,
    start: usize,
    count: usize,
) -> Result<Matrix> {
    if adjoint.shape() != (joint.nrows(), joint.nrows())
        || start
            .checked_add(count)
            .is_none_or(|end| end > joint.nrows())
    {
        return Err(BotorchError::DimensionMismatch(
            "invalid EM inducing input adjoint".into(),
        ));
    }
    let prepared = kernel.prepare(joint)?;
    let mut result = Matrix::zeros(count, joint.ncols());
    for row in 0..count {
        let index = start + row;
        let gradient = kernel.cross_with_grad(&prepared, index, &prepared).1;
        for c in 0..joint.ncols() {
            for j in 0..joint.nrows() {
                result[(row, c)] += (adjoint[(index, j)] + adjoint[(j, index)]) * gradient[(j, c)];
            }
        }
    }
    Ok(result)
}
