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

    /// The Cholesky factor of the bordered matrix `[[A, Bᵀ], [B, C]]` from
    /// this factor of `A` (`n × n`), the new rows `cross = B` (`k × n`) and
    /// the new corner `corner = C` (`k × k`): an exact rank-`k` update,
    ///
    /// ```text
    /// [[L, 0], [B L⁻ᵀ, chol(C - B A⁻¹ Bᵀ)]],
    /// ```
    ///
    /// in `O(n² k)` instead of `O((n + k)³)`. This is what a Cholesky
    /// factorization of the whole matrix computes (block by block), and what
    /// GPyTorch's fantasy models do (`LinearOperator.cat_rows`).
    ///
    /// Returns `None` if this factor needed jitter (the update would keep
    /// the jitter of the old block only) or if the Schur complement is not
    /// numerically positive definite; callers then factorize the whole
    /// matrix with [`psd_safe_cholesky`].
    ///
    /// # Panics
    /// If the shapes are inconsistent.
    pub fn bordered(&self, cross: &Matrix, corner: &Matrix) -> Option<CholeskyFactor> {
        let n = self.dim();
        let k = cross.nrows();
        assert!(
            cross.ncols() == n && corner.shape() == (k, k),
            "bordered: inconsistent shapes"
        );
        if self.jitter != 0.0 {
            return None;
        }
        // L21ᵀ = L⁻¹ Bᵀ (n × k).
        let mut l21t = cross.transpose();
        self.solve_lower_mut(&mut l21t);
        let mut schur = corner.clone();
        schur.gemm_tr(-1.0, &l21t, &l21t, 1.0);
        let l22 = try_cholesky(&schur)?;
        let mut l = Matrix::zeros(n + k, n + k);
        l.view_mut((0, 0), (n, n)).copy_from(&self.l);
        l.view_mut((n, 0), (k, n)).copy_from(&l21t.transpose());
        l.view_mut((n, n), (k, k)).copy_from(&l22);
        Some(CholeskyFactor { l, jitter: 0.0 })
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

/// Reverse-mode derivative of the Cholesky factorization.
///
/// Given the lower-triangular factor `L` of `A = L Lᵀ` (possibly jittered)
/// and the adjoint `L̄ = ∂f/∂L` of a scalar `f` (only its lower triangle is
/// used), returns the symmetric adjoint `Ā` such that `df = Σ_ab Ā_ab dA_ab`
/// for every symmetric perturbation `dA`:
/// `Ā = sym(L⁻ᵀ Φ(Lᵀ L̄) L⁻¹)`, where `Φ` takes the lower triangle and halves
/// the diagonal (Murray 2016, "Differentiation of the Cholesky decomposition").
///
/// # Panics
/// If the shapes of `l` and `l_bar` differ or are not square.
pub fn cholesky_backward(l: &Matrix, l_bar: &Matrix) -> Matrix {
    let n = l.nrows();
    assert!(
        l.is_square() && l_bar.shape() == (n, n),
        "cholesky_backward: shape mismatch"
    );
    let lbar_lower = l_bar.lower_triangle();
    let mut phi = l.tr_mul(&lbar_lower);
    for j in 0..n {
        phi[(j, j)] *= 0.5;
        for i in 0..j {
            phi[(i, j)] = 0.0;
        }
    }
    // G = L⁻ᵀ Φ L⁻¹: Y = L⁻ᵀ Φ, then Gᵀ = L⁻ᵀ Yᵀ.
    l.tr_solve_lower_triangular_unchecked_mut(&mut phi);
    let mut gt = phi.transpose();
    l.tr_solve_lower_triangular_unchecked_mut(&mut gt);
    (&gt + gt.transpose()) * 0.5
}

/// A square root `R` of a posterior covariance (`R Rᵀ = Σ`) used to map base
/// samples to posterior samples, chosen like gpytorch's
/// `MultivariateNormal.rsample(base_samples=...)` with
/// `root_decomposition()`: `sqrt` for a single point, the jittered
/// [`psd_safe_cholesky`] otherwise, and an eigendecomposition with clamped
/// eigenvalues if the Cholesky factorization fails even with jitter.
#[derive(Clone, Debug, PartialEq)]
pub enum CovarianceRoot {
    /// A `1 × 1` covariance `v`: `R = sqrt(max(v, 0))`.
    Scalar {
        /// The variance `v` (unclamped).
        variance: f64,
        /// `sqrt(max(v, 0))`.
        root: f64,
    },
    /// Lower Cholesky factor of `Σ + jitter·I`.
    Cholesky(CholeskyFactor),
    /// `V diag(sqrt(max(λ, 0)))` from the eigendecomposition `Σ = V Λ Vᵀ`
    /// (gpytorch's `symeig` fallback). Not differentiable.
    Eigen(Matrix),
    /// The zero root of an exactly zero `n × n` covariance (e.g. the block
    /// of a deterministic sub-model of a model list, whose samples are its
    /// values, as BoTorch samples a `PosteriorList` posterior by posterior).
    Zero(usize),
}

impl CovarianceRoot {
    /// Computes the root of the symmetric matrix `cov`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a non-square or empty matrix
    /// and [`BotorchError::Nan`] for NaN entries.
    pub fn new(cov: &Matrix) -> Result<Self> {
        let n = cov.nrows();
        if !cov.is_square() || n == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "covariance root requires a non-empty square matrix, got {}x{}",
                cov.nrows(),
                cov.ncols()
            )));
        }
        if cov.iter().all(|&v| v == 0.0) && n > 1 {
            return Ok(Self::Zero(n));
        }
        if n == 1 {
            let variance = cov[(0, 0)];
            if variance.is_nan() {
                return Err(BotorchError::Nan { count: 1, total: 1 });
            }
            return Ok(Self::Scalar {
                variance,
                root: variance.max(0.0).sqrt(),
            });
        }
        match psd_safe_cholesky(cov) {
            Ok(f) => Ok(Self::Cholesky(f)),
            Err(BotorchError::NotPsd { .. }) => {
                let eig = nalgebra::SymmetricEigen::new(cov.clone());
                let mut root = eig.eigenvectors;
                for (j, &lam) in eig.eigenvalues.iter().enumerate() {
                    let s = lam.max(0.0).sqrt();
                    root.column_mut(j).scale_mut(s);
                }
                Ok(Self::Eigen(root))
            }
            Err(e) => Err(e),
        }
    }

    /// The dimension `n` of the covariance.
    pub fn dim(&self) -> usize {
        match self {
            Self::Scalar { .. } => 1,
            Self::Cholesky(f) => f.dim(),
            Self::Eigen(r) => r.nrows(),
            Self::Zero(n) => *n,
        }
    }

    /// The jitter added to the diagonal (0 unless the Cholesky factorization
    /// needed it).
    pub fn jitter(&self) -> f64 {
        match self {
            Self::Cholesky(f) => f.jitter(),
            _ => 0.0,
        }
    }

    /// The root as a dense `n × n` matrix.
    pub fn to_matrix(&self) -> Matrix {
        match self {
            Self::Scalar { root, .. } => Matrix::from_element(1, 1, *root),
            Self::Cholesky(f) => f.l().clone(),
            Self::Eigen(r) => r.clone(),
            Self::Zero(n) => Matrix::zeros(*n, *n),
        }
    }

    /// The rows `rows` of the root (`rows.len() × n`).
    pub fn rows(&self, rows: std::ops::Range<usize>) -> Matrix {
        match self {
            Self::Scalar { root, .. } => Matrix::from_element(rows.len(), 1, *root),
            Self::Cholesky(f) => f.l().rows(rows.start, rows.len()).into_owned(),
            Self::Eigen(r) => r.rows(rows.start, rows.len()).into_owned(),
            Self::Zero(n) => Matrix::zeros(rows.len(), *n),
        }
    }

    /// Reverse-mode derivative: maps the adjoint of the root (`n × n`, only
    /// the entries of the root's sparsity pattern are used) to the symmetric
    /// adjoint of the covariance.
    ///
    /// # Errors
    /// [`BotorchError::Numerical`] for the (non-differentiable)
    /// eigendecomposition fallback.
    pub fn backward(&self, root_adjoint: &Matrix) -> Result<Matrix> {
        match self {
            Self::Scalar { variance, root } => {
                let g = if *variance > 0.0 {
                    root_adjoint[(0, 0)] / (2.0 * root)
                } else {
                    0.0
                };
                Ok(Matrix::from_element(1, 1, g))
            }
            Self::Cholesky(f) => Ok(cholesky_backward(f.l(), root_adjoint)),
            // A constant (zero) covariance: no adjoint.
            Self::Zero(n) => Ok(Matrix::zeros(*n, *n)),
            Self::Eigen(_) => Err(BotorchError::Numerical(
                "the posterior covariance is not positive definite even with jitter; \
                 its eigendecomposition root is not differentiable"
                    .into(),
            )),
        }
    }
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

    #[test]
    fn bordered_factor_equals_the_full_factorization() {
        let a = spd(7);
        let full = psd_safe_cholesky(&a).unwrap();
        for n in [1usize, 4, 6] {
            let k = 7 - n;
            let head = psd_safe_cholesky(&a.view((0, 0), (n, n)).into_owned()).unwrap();
            let cross = a.view((n, 0), (k, n)).into_owned();
            let corner = a.view((n, n), (k, k)).into_owned();
            let b = head.bordered(&cross, &corner).unwrap();
            assert!((b.l() - full.l()).amax() < 1e-12 * full.l().amax());
        }
        // A Schur complement that is not positive definite (C - B A⁻¹ Bᵀ < 0)
        // and a factor with jitter.
        let head = psd_safe_cholesky(&a.view((0, 0), (6, 6)).into_owned()).unwrap();
        let cross = a.view((6, 0), (1, 6)).into_owned();
        assert!(head.bordered(&cross, &Matrix::zeros(1, 1)).is_none());
        let v = Matrix::from_column_slice(3, 1, &[1.0, 2.0, 3.0]);
        let jittered = psd_safe_cholesky(&(&v * v.transpose())).unwrap();
        assert!(jittered.jitter() > 0.0);
        assert!(jittered
            .bordered(&Matrix::zeros(1, 3), &Matrix::identity(1, 1))
            .is_none());
    }

    #[test]
    fn cholesky_backward_matches_finite_differences() {
        let n = 5;
        let a = spd(n);
        let l = psd_safe_cholesky(&a).unwrap();
        let w = Matrix::from_fn(n, n, |i, j| ((i * 3 + j * 5) % 7) as f64 / 7.0 - 0.4);
        let f = |m: &Matrix| psd_safe_cholesky(m).unwrap().l().component_mul(&w).sum();
        let abar = cholesky_backward(l.l(), &w);
        assert!((&abar - abar.transpose()).amax() < 1e-14);
        for (i, j) in [(0usize, 0usize), (1, 0), (3, 2), (4, 4), (4, 1)] {
            let h = 1e-6;
            let mut e = Matrix::zeros(n, n);
            e[(i, j)] = 1.0;
            e[(j, i)] = 1.0;
            let fd = (f(&(&a + &e * h)) - f(&(&a - &e * h))) / (2.0 * h);
            let an = abar.component_mul(&e).sum();
            assert!(
                (fd - an).abs() < 1e-7 * an.abs().max(1.0),
                "({i},{j}): {an} vs {fd}"
            );
        }
    }

    #[test]
    fn covariance_roots() {
        let a = spd(4);
        let r = CovarianceRoot::new(&a).unwrap();
        assert!(matches!(r, CovarianceRoot::Cholesky(_)));
        let m = r.to_matrix();
        assert!((&m * m.transpose() - &a).amax() < 1e-10);
        assert_eq!(r.rows(1..3), m.rows(1, 2).into_owned());
        let s = CovarianceRoot::new(&Matrix::from_element(1, 1, 4.0)).unwrap();
        assert_eq!(s.to_matrix()[(0, 0)], 2.0);
        assert_eq!(
            s.backward(&Matrix::from_element(1, 1, 1.0)).unwrap()[(0, 0)],
            0.25
        );
        let neg = CovarianceRoot::new(&Matrix::from_element(1, 1, -1e-18)).unwrap();
        assert_eq!(neg.to_matrix()[(0, 0)], 0.0);
        // Indefinite beyond the jitter: eigendecomposition fallback.
        let b = Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1e-3]);
        let e = CovarianceRoot::new(&b).unwrap();
        assert!(matches!(e, CovarianceRoot::Eigen(_)));
        let z = CovarianceRoot::new(&Matrix::zeros(3, 3)).unwrap();
        assert!(matches!(z, CovarianceRoot::Zero(3)));
        assert_eq!(z.rows(1..3), Matrix::zeros(2, 3));
        assert_eq!(
            z.backward(&Matrix::identity(3, 3)).unwrap(),
            Matrix::zeros(3, 3)
        );
        let em = e.to_matrix();
        let rec = &em * em.transpose();
        assert!((rec[(0, 0)] - 1.0).abs() < 1e-14 && rec[(1, 1)].abs() < 1e-14);
        assert!(e.backward(&Matrix::zeros(2, 2)).is_err());
        assert!(CovarianceRoot::new(&Matrix::zeros(2, 3)).is_err());
    }
}
