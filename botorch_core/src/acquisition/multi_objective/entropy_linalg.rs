//! Small dense matrix routines with the semantics of the torch functions
//! that the entropy-search acquisition functions use: `torch.logdet` and
//! `torch.linalg.inv` (LU with partial pivoting), `torch.linalg.cholesky_ex`
//! (a Cholesky factorization that reports the failing leading minor instead
//! of raising, like LAPACK's `potrf`) and `torch.cholesky_inverse`.

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

/// `torch.linalg.cholesky_ex(a)`: the lower Cholesky factor and LAPACK's
/// `info` (0 on success, else the order of the first leading minor that is
/// not positive definite). On failure at column `j`, the factor holds the
/// finished columns, the failing pivot `a_jj - Σ l_jk²` and the remaining
/// lower entries of `a` (the unblocked `potf2` state); the strict upper
/// triangle is zero, as torch returns it.
pub(crate) fn cholesky_ex(a: &Matrix) -> (Matrix, usize) {
    let n = a.nrows();
    let mut l = a.lower_triangle();
    for j in 0..n {
        let mut ajj = l[(j, j)];
        for k in 0..j {
            ajj -= l[(j, k)] * l[(j, k)];
        }
        if ajj <= 0.0 || ajj.is_nan() {
            l[(j, j)] = ajj;
            return (l, j + 1);
        }
        let ljj = ajj.sqrt();
        l[(j, j)] = ljj;
        for i in (j + 1)..n {
            let mut s = l[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = s / ljj;
        }
    }
    (l, 0)
}

/// `torch.cholesky_inverse(l)`: `(L Lᵀ)⁻¹` from a lower factor, computed as
/// `L⁻ᵀ L⁻¹` (LAPACK's `potri`) and symmetric by construction.
pub(crate) fn cholesky_inverse(l: &Matrix) -> Matrix {
    let n = l.nrows();
    // L⁻¹ by forward substitution.
    let mut inv = Matrix::zeros(n, n);
    for j in 0..n {
        inv[(j, j)] = 1.0 / l[(j, j)];
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in j..i {
                s += l[(i, k)] * inv[(k, j)];
            }
            inv[(i, j)] = -s / l[(i, i)];
        }
    }
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            // (L⁻ᵀ L⁻¹)_ij = Σ_k L⁻¹_ki L⁻¹_kj over k ≥ max(i, j) = i.
            let mut s = 0.0;
            for k in i..n {
                s += inv[(k, i)] * inv[(k, j)];
            }
            out[(i, j)] = s;
            out[(j, i)] = s;
        }
    }
    out
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
    fn lu_logdet_inverse_and_cholesky() {
        let a = spd(5);
        let det = a.clone().determinant();
        assert!((Lu::new(&a).logdet() - det.ln()).abs() < 1e-12);
        let inv = Lu::new(&a).inverse();
        assert!((&a * &inv - Matrix::identity(5, 5)).amax() < 1e-12);
        let (l, info) = cholesky_ex(&a);
        assert_eq!(info, 0);
        assert!((&l * l.transpose() - &a).amax() < 1e-12);
        assert!((cholesky_inverse(&l) - &inv).amax() < 1e-10);
        // A negative determinant gives NaN, a singular matrix -inf / None.
        let neg = Matrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        assert!(Lu::new(&neg).logdet().is_nan());
        let sing = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert_eq!(Lu::new(&sing).logdet(), f64::NEG_INFINITY);
        // Cholesky failure at the second leading minor.
        let bad = Matrix::from_row_slice(3, 3, &[1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let (lb, info) = cholesky_ex(&bad);
        assert_eq!(info, 2);
        assert_eq!(lb[(1, 1)], 1.0 - 4.0);
        assert_eq!(lb[(2, 2)], 1.0);
    }
}
