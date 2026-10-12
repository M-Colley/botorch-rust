//! Small dense matrix routines with the semantics of the torch functions
//! that the entropy-search acquisition functions use: `torch.logdet` and
//! `torch.linalg.inv` (LU with partial pivoting).

use crate::Matrix;

/// An LU factorization with partial (row) pivoting, `P A = L U` (LAPACK's
/// `getrf`): `lu` holds `U` in its upper triangle and the multipliers of
/// `L` below the diagonal; `perm[i]` is the original row of row `i`.
pub(crate) struct Lu {
    lu: Matrix,
    perm: Vec<usize>,
    swaps: usize,
}

impl Lu {
    /// Factorizes the square matrix `a`. Exactly zero pivots are kept (the
    /// matrix is singular); `getrf` continues past them as well.
    pub fn new(a: &Matrix) -> Self {
        let n = a.nrows();
        let mut lu = a.clone();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut swaps = 0;
        for k in 0..n {
            // The first row with the largest magnitude (BLAS `idamax`).
            let mut p = k;
            let mut best = lu[(k, k)].abs();
            for i in (k + 1)..n {
                let v = lu[(i, k)].abs();
                if v > best || (best.is_nan() && !v.is_nan()) {
                    best = v;
                    p = i;
                }
            }
            if p != k {
                lu.swap_rows(p, k);
                perm.swap(p, k);
                swaps += 1;
            }
            let pivot = lu[(k, k)];
            if pivot == 0.0 {
                continue;
            }
            for i in (k + 1)..n {
                let l = lu[(i, k)] / pivot;
                lu[(i, k)] = l;
                for j in (k + 1)..n {
                    let u = lu[(k, j)];
                    lu[(i, j)] -= l * u;
                }
            }
        }
        Self { lu, perm, swaps }
    }

    /// `torch.logdet`: `log|det A|`, NaN for a negative determinant and
    /// `-inf` for a singular matrix.
    pub fn logdet(&self) -> f64 {
        let n = self.lu.nrows();
        let mut negative = self.swaps % 2 == 1;
        let mut log_abs = 0.0;
        for i in 0..n {
            let u = self.lu[(i, i)];
            if u.is_nan() {
                return f64::NAN;
            }
            if u < 0.0 {
                negative = !negative;
            }
            log_abs += u.abs().ln();
        }
        if negative && log_abs > f64::NEG_INFINITY {
            f64::NAN
        } else {
            log_abs
        }
    }

    /// Solves `A X = B` (LAPACK's `getrs`).
    pub fn solve(&self, b: &Matrix) -> Matrix {
        let n = self.lu.nrows();
        let mut x = Matrix::from_fn(n, b.ncols(), |i, j| b[(self.perm[i], j)]);
        for c in 0..x.ncols() {
            for i in 0..n {
                let mut s = x[(i, c)];
                for k in 0..i {
                    s -= self.lu[(i, k)] * x[(k, c)];
                }
                x[(i, c)] = s;
            }
            for i in (0..n).rev() {
                let mut s = x[(i, c)];
                for k in (i + 1)..n {
                    s -= self.lu[(i, k)] * x[(k, c)];
                }
                x[(i, c)] = s / self.lu[(i, i)];
            }
        }
        x
    }

    /// `A⁻¹` (`torch.linalg.inv`, which solves `A X = I`).
    pub fn inverse(&self) -> Matrix {
        let n = self.lu.nrows();
        self.solve(&Matrix::identity(n, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize) -> Matrix {
        let b = Matrix::from_fn(n, n, |i, j| {
            ((i * 5 + j * 3) % 7) as f64 / 7.0 + if i == j { 1.0 } else { 0.0 }
        });
        &b * b.transpose()
    }

    #[test]
    fn lu_logdet_and_inverse() {
        let a = spd(5);
        let det = a.clone().determinant();
        let lu = Lu::new(&a);
        assert!((lu.logdet() - det.ln()).abs() < 1e-12);
        assert!((&a * lu.inverse() - Matrix::identity(5, 5)).amax() < 1e-12);
        // A negative determinant gives NaN, a singular matrix -inf.
        let neg = Matrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        assert!(Lu::new(&neg).logdet().is_nan());
        let sing = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert_eq!(Lu::new(&sing).logdet(), f64::NEG_INFINITY);
    }
}
