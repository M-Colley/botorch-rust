//! Dense linear algebra helpers: a jittered ("PSD-safe") Cholesky factorization
//! mirroring `linear_operator.utils.cholesky.psd_safe_cholesky`, plus the
//! triangular solves, inverses and log-determinants built on top of it.

use nalgebra::Cholesky;

use crate::error::{BotorchError, Result};
use crate::{Matrix, Vector};

/// Initial jitter added to the diagonal when a Cholesky factorization fails
/// (`linear_operator.settings.cholesky_jitter` for double precision).
pub const CHOLESKY_JITTER: f64 = 1e-8;

/// Number of jitter attempts, each with ten times the previous jitter
/// (`linear_operator.settings.cholesky_max_tries`).
pub const CHOLESKY_MAX_TRIES: usize = 3;

/// A lower-triangular Cholesky factor `L` with `L Lᵀ = A + jitter·I`.
#[derive(Clone, Debug, PartialEq)]
pub struct CholeskyFactor {
    l: Matrix,
    jitter: f64,
}

impl CholeskyFactor {
    /// The lower-triangular factor (the strict upper triangle is zero).
    pub fn l(&self) -> &Matrix {
        &self.l
    }

    /// The jitter that had to be added to the diagonal (0 if none).
    pub fn jitter(&self) -> f64 {
        self.jitter
    }

    /// The dimension `n` of the factorized `n × n` matrix.
    pub fn dim(&self) -> usize {
        self.l.nrows()
    }

    /// Overwrites `b` with `L⁻¹ b`.
    pub fn solve_lower_mut(&self, b: &mut Matrix) {
        assert_eq!(
            b.nrows(),
            self.dim(),
            "dimension mismatch in triangular solve"
        );
        self.l.solve_lower_triangular_unchecked_mut(b);
    }

    /// Overwrites `b` with `L⁻ᵀ b`.
    pub fn solve_lower_transpose_mut(&self, b: &mut Matrix) {
        assert_eq!(
            b.nrows(),
            self.dim(),
            "dimension mismatch in triangular solve"
        );
        self.l.tr_solve_lower_triangular_unchecked_mut(b);
    }

    /// Returns `L⁻¹ b` for a vector `b`.
    pub fn solve_lower_vec(&self, b: &Vector) -> Vector {
        assert_eq!(
            b.len(),
            self.dim(),
            "dimension mismatch in triangular solve"
        );
        let mut out = b.clone();
        self.l.solve_lower_triangular_unchecked_mut(&mut out);
        out
    }

    /// Returns `L⁻ᵀ b` for a vector `b`.
    pub fn solve_lower_transpose_vec(&self, b: &Vector) -> Vector {
        assert_eq!(
            b.len(),
            self.dim(),
            "dimension mismatch in triangular solve"
        );
        let mut out = b.clone();
        self.l.tr_solve_lower_triangular_unchecked_mut(&mut out);
        out
    }

    /// Returns `(L Lᵀ)⁻¹ b`.
    pub fn solve(&self, b: &Matrix) -> Matrix {
        let mut out = b.clone();
        self.solve_lower_mut(&mut out);
        self.solve_lower_transpose_mut(&mut out);
        out
    }

    /// Returns `(L Lᵀ)⁻¹ b` for a vector `b`.
    pub fn solve_vec(&self, b: &Vector) -> Vector {
        assert_eq!(b.len(), self.dim(), "dimension mismatch in Cholesky solve");
        let mut out = b.clone();
        self.l.solve_lower_triangular_unchecked_mut(&mut out);
        self.l.tr_solve_lower_triangular_unchecked_mut(&mut out);
        out
    }

    /// Returns `L⁻¹` (lower triangular), computed by forward substitution that
    /// exploits the triangular structure of the identity right-hand side.
    pub fn inverse_factor(&self) -> Matrix {
        let n = self.dim();
        let l = &self.l;
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
        inv
    }

    /// Returns `(L Lᵀ)⁻¹`, i.e. the inverse of the (jittered) factorized matrix.
    pub fn inverse(&self) -> Matrix {
        let linv = self.inverse_factor();
        let mut out = linv.transpose() * &linv;
        // Symmetrize to remove round-off asymmetry.
        let n = out.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                let v = 0.5 * (out[(i, j)] + out[(j, i)]);
                out[(i, j)] = v;
                out[(j, i)] = v;
            }
        }
        out
    }

    /// Returns `log det(L Lᵀ) = 2 Σᵢ log Lᵢᵢ`.
    pub fn logdet(&self) -> f64 {
        2.0 * self.l.diagonal().iter().map(|v| v.ln()).sum::<f64>()
    }
}

/// Attempts a plain Cholesky factorization, returning `None` if `a` is not
/// numerically positive definite.
fn try_cholesky(a: &Matrix) -> Option<Matrix> {
    Cholesky::new(a.clone()).map(|c| c.unpack())
}

/// Cholesky factorization that adds jitter to the diagonal on failure,
/// mirroring `linear_operator.utils.cholesky.psd_safe_cholesky` with the
/// double-precision defaults: jitter `1e-8`, `1e-7`, `1e-6` is tried in turn
/// (the jitter values are cumulative, i.e. the `i`-th retry factorizes
/// `A + 1e-8·10ⁱ·I`).
///
/// # Errors
/// [`BotorchError::Nan`] if `a` contains NaN values and
/// [`BotorchError::NotPsd`] if all attempts fail.
pub fn psd_safe_cholesky(a: &Matrix) -> Result<CholeskyFactor> {
    psd_safe_cholesky_with(a, CHOLESKY_JITTER, CHOLESKY_MAX_TRIES)
}

/// [`psd_safe_cholesky`] with explicit initial `jitter` and `max_tries`.
///
/// # Errors
/// See [`psd_safe_cholesky`].
pub fn psd_safe_cholesky_with(a: &Matrix, jitter: f64, max_tries: usize) -> Result<CholeskyFactor> {
    if !a.is_square() {
        return Err(BotorchError::DimensionMismatch(format!(
            "Cholesky factorization requires a square matrix, got {}x{}",
            a.nrows(),
            a.ncols()
        )));
    }
    if let Some(l) = try_cholesky(a) {
        return Ok(CholeskyFactor { l, jitter: 0.0 });
    }
    let nan_count = a.iter().filter(|v| v.is_nan()).count();
    if nan_count > 0 {
        return Err(BotorchError::Nan {
            count: nan_count,
            total: a.len(),
        });
    }
    let mut aprime = a.clone();
    let mut jitter_prev = 0.0;
    let mut jitter_new = jitter;
    for i in 0..max_tries {
        jitter_new = jitter * 10f64.powi(i as i32);
        let delta = jitter_new - jitter_prev;
        for k in 0..aprime.nrows() {
            aprime[(k, k)] += delta;
        }
        jitter_prev = jitter_new;
        if let Some(l) = try_cholesky(&aprime) {
            return Ok(CholeskyFactor {
                l,
                jitter: jitter_new,
            });
        }
    }
    Err(BotorchError::NotPsd { jitter: jitter_new })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize) -> Matrix {
        let b = Matrix::from_fn(n, n, |i, j| {
            ((i * 7 + j * 3) % 5) as f64 + if i == j { 1.0 } else { 0.0 }
        });
        &b * b.transpose() + Matrix::identity(n, n)
    }

    #[test]
    fn solves_and_inverse_are_consistent() {
        let a = spd(6);
        let f = psd_safe_cholesky(&a).unwrap();
        assert_eq!(f.jitter(), 0.0);
        let inv = f.inverse();
        let eye = &a * &inv;
        for i in 0..6 {
            for j in 0..6 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((eye[(i, j)] - target).abs() < 1e-10);
            }
        }
        let b = Vector::from_fn(6, |i, _| i as f64 - 2.0);
        let x = f.solve_vec(&b);
        assert!((&a * &x - &b).norm() < 1e-10);
        let det = a.clone().determinant();
        assert!((f.logdet() - det.ln()).abs() < 1e-10);
        let linv = f.inverse_factor();
        assert!((&linv * f.l() - Matrix::identity(6, 6)).norm() < 1e-12);
    }

    #[test]
    fn jitter_is_added_for_singular_matrices() {
        let v = Matrix::from_column_slice(3, 1, &[1.0, 2.0, 3.0]);
        let a = &v * v.transpose();
        let f = psd_safe_cholesky(&a).unwrap();
        assert!(f.jitter() > 0.0);
        let rec = f.l() * f.l().transpose();
        let expected = &a + Matrix::identity(3, 3) * f.jitter();
        assert!((rec - expected).norm() < 1e-12);
    }

    #[test]
    fn errors_on_nan_and_indefinite() {
        let mut a = Matrix::identity(2, 2);
        a[(0, 1)] = f64::NAN;
        a[(1, 0)] = f64::NAN;
        assert!(matches!(
            psd_safe_cholesky(&a),
            Err(BotorchError::Nan { .. })
        ));
        let b = -Matrix::identity(2, 2);
        assert!(matches!(
            psd_safe_cholesky(&b),
            Err(BotorchError::NotPsd { .. })
        ));
    }
}
