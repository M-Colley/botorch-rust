//! Shared query pullback for train-first data Cholesky Matheron samplers.

use crate::kernels::TreeKernel;
use crate::linalg::cholesky_backward;
use crate::{Matrix, Result};

pub(crate) fn data_query_vjp(
    kernel: &TreeKernel,
    all: &Matrix,
    train: usize,
    root: &Matrix,
    root_adjoint: &Matrix,
    cross_adjoint: &Matrix,
) -> Result<Matrix> {
    let q = all.nrows() - train;
    let d = all.ncols();
    let joint_adjoint = cholesky_backward(root, root_adjoint);
    let prep = kernel.prepare(all)?;
    let xp = kernel.prepare(&all.rows(train, q).into_owned())?;
    let tp = kernel.prepare(&all.rows(0, train).into_owned())?;
    let mut gradient = Matrix::zeros(q, d);
    for p in 0..q {
        let row = train + p;
        let (_, joint_derivative) = kernel.cross_with_grad(&prep, row, &prep);
        let (_, diagonal) = kernel.diag_with_grad(&prep, row);
        let (_, cross_derivative) = kernel.cross_with_grad(&xp, p, &tp);
        for j in 0..d {
            gradient[(p, j)] = joint_adjoint[(row, row)] * diagonal[j];
            for r in 0..all.nrows() {
                if r != row {
                    gradient[(p, j)] += (joint_adjoint[(row, r)] + joint_adjoint[(r, row)])
                        * joint_derivative[(r, j)];
                }
            }
            for r in 0..train {
                gradient[(p, j)] += cross_adjoint[(p, r)] * cross_derivative[(r, j)];
            }
        }
    }
    Ok(gradient)
}

pub(crate) fn kronecker_vjp(left: &Matrix, right: &Matrix, adjoint: &Matrix) -> (Matrix, Matrix) {
    let mut lb = Matrix::zeros(left.nrows(), left.ncols());
    let mut rb = Matrix::zeros(right.nrows(), right.ncols());
    for i in 0..left.nrows() {
        for j in 0..left.ncols() {
            for a in 0..right.nrows() {
                for b in 0..right.ncols() {
                    let value = adjoint[(i * right.nrows() + a, j * right.ncols() + b)];
                    lb[(i, j)] += value * right[(a, b)];
                    rb[(a, b)] += value * left[(i, j)];
                }
            }
        }
    }
    (lb, rb)
}

pub(crate) fn cross_self_query_vjp(
    kernel: &TreeKernel,
    train: &Matrix,
    query: &Matrix,
    cross_adjoint: &Matrix,
    self_adjoint: &Matrix,
) -> Result<Matrix> {
    let tp = kernel.prepare(train)?;
    let qp = kernel.prepare(query)?;
    let mut gradient = Matrix::zeros(query.nrows(), query.ncols());
    for p in 0..query.nrows() {
        let (_, dcross) = kernel.cross_with_grad(&qp, p, &tp);
        let (_, dself) = kernel.cross_with_grad(&qp, p, &qp);
        let (_, diagonal) = kernel.diag_with_grad(&qp, p);
        for j in 0..query.ncols() {
            gradient[(p, j)] = self_adjoint[(p, p)] * diagonal[j];
            for i in 0..train.nrows() {
                gradient[(p, j)] += cross_adjoint[(i, p)] * dcross[(i, j)];
            }
            for r in 0..query.nrows() {
                if r != p {
                    gradient[(p, j)] +=
                        (self_adjoint[(p, r)] + self_adjoint[(r, p)]) * dself[(r, j)];
                }
            }
        }
    }
    Ok(gradient)
}
