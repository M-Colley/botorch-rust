//! MINPACK's `hybrj`: Powell's hybrid (dogleg trust-region) method for
//! systems of nonlinear equations with a user-supplied Jacobian, the solver
//! of `scipy.optimize.fsolve(func, x0, fprime=...)`.
//!
//! This is a direct Rust port of MINPACK 1 (`hybrj`, `dogleg`, `qrfac`
//! without pivoting, `qform`, `r1updt`, `r1mpyq`, `enorm`) as scipy runs it
//! (`_minpack._hybrj` with `mode = 1`, `nprint = 0`). The structure, the
//! floating-point operations and the 1-based indexing of the Fortran code are
//! kept so that the port can be checked line by line; on functions whose
//! values both libraries compute bitwise identically, the evaluation points,
//! the result and the counters are scipy's bitwise
//! (`botorch_core/tests/preference_10b_parity.rs`). No nonlinear solver is
//! available among the offline crates, so the reference algorithm is ported.
//!
//! [`fsolve`] adds scipy's defaults: `xtol = 1.49012e-8`, `maxfev = 100·(n +
//! 1)` when 0, `factor = 100`, automatic variable scaling (`diag = None`).
//!
//! License of the original code (retained as required): MINPACK is
//! distributed under the MINPACK license (University of Chicago, Argonne
//! National Laboratory; Jorge J. Moré, Burton S. Garbow, Kenneth E.
//! Hillstrom, 1980). Redistribution and use in source and binary forms, with
//! or without modification, are permitted provided that the copyright
//! notice, the acknowledgment "This product includes software developed by
//! the University of Chicago, as Operator of Argonne National Laboratory"
//! and the disclaimer are retained. THE SOFTWARE IS PROVIDED "AS IS",
//! WITHOUT WARRANTY OF ANY KIND.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::explicit_counter_loop
)]

use std::ops::{Index, IndexMut};

use crate::error::{BotorchError, Result};
use crate::Matrix;

/// Machine precision (`dpmpar(1)`).
const EPSMCH: f64 = f64::EPSILON;
/// The largest magnitude (`dpmpar(3)`).
const GIANT: f64 = f64::MAX;

/// A 1-based vector (Fortran indexing).
#[derive(Clone, Debug)]
struct V(Vec<f64>);

impl V {
    fn zeros(n: usize) -> Self {
        V(vec![0.0; n])
    }
}

impl Index<usize> for V {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        &self.0[i - 1]
    }
}

impl IndexMut<usize> for V {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i - 1]
    }
}

/// A 1-based, column-major `n × n` matrix (Fortran `fjac(ldfjac, n)`).
#[derive(Clone, Debug)]
struct M {
    n: usize,
    a: Vec<f64>,
}

impl Index<(usize, usize)> for M {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.a[(i - 1) + (j - 1) * self.n]
    }
}

impl IndexMut<(usize, usize)> for M {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.a[(i - 1) + (j - 1) * self.n]
    }
}

/// MINPACK's `enorm`: the Euclidean norm of `x[start..start + n]` (1-based
/// `start`), computed with scaling to avoid destructive underflow and
/// overflow.
fn enorm(n: usize, x: &[f64]) -> f64 {
    const RDWARF: f64 = 3.834e-20;
    const RGIANT: f64 = 1.304e19;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut x1max = 0.0;
    let mut x3max = 0.0;
    let floatn = n as f64;
    let agiant = RGIANT / floatn;
    for &xi in &x[..n] {
        let xabs = xi.abs();
        if xabs > RDWARF && xabs < agiant {
            // Sum for intermediate components.
            s2 += xabs * xabs;
        } else if xabs <= RDWARF {
            // Sum for small components.
            if xabs > x3max {
                let r = x3max / xabs;
                s3 = 1.0 + s3 * (r * r);
                x3max = xabs;
            } else if xabs != 0.0 {
                let r = xabs / x3max;
                s3 += r * r;
            }
        } else {
            // Sum for large components.
            if xabs > x1max {
                let r = x1max / xabs;
                s1 = 1.0 + s1 * (r * r);
                x1max = xabs;
            } else {
                let r = xabs / x1max;
                s1 += r * r;
            }
        }
    }
    if s1 != 0.0 {
        x1max * (s1 + (s2 / x1max) / x1max).sqrt()
    } else if s2 != 0.0 {
        if s2 >= x3max {
            (s2 * (1.0 + (x3max / s2) * (x3max * s3))).sqrt()
        } else {
            (x3max * ((s2 / x3max) + (x3max * s3))).sqrt()
        }
    } else {
        x3max * s3.sqrt()
    }
}

/// `enorm` of a 1-based vector.
fn enorm_v(n: usize, x: &V) -> f64 {
    enorm(n, &x.0)
}

/// `enorm` of the 1-based column segment `a(i0..=m, j)`.
fn enorm_col(a: &M, i0: usize, m: usize, j: usize) -> f64 {
    let start = (i0 - 1) + (j - 1) * a.n;
    enorm(m + 1 - i0, &a.a[start..])
}

/// MINPACK's `qrfac` without column pivoting on the `m × n` matrix `a`
/// (here `m = n`): Householder QR, `rdiag` the diagonal of `R`, `acnorm` the
/// norms of the columns of `a` (`wa` is workspace).
fn qrfac(m: usize, n: usize, a: &mut M, rdiag: &mut V, acnorm: &mut V, wa: &mut V) {
    // Compute the initial column norms and initialize several arrays.
    for j in 1..=n {
        acnorm[j] = enorm_col(a, 1, m, j);
        rdiag[j] = acnorm[j];
        wa[j] = rdiag[j];
    }
    // Reduce a to r with Householder transformations.
    let minmn = m.min(n);
    for j in 1..=minmn {
        // Compute the Householder transformation to reduce the j-th column
        // of a to a multiple of the j-th unit vector.
        let mut ajnorm = enorm_col(a, j, m, j);
        if ajnorm != 0.0 {
            if a[(j, j)] < 0.0 {
                ajnorm = -ajnorm;
            }
            for i in j..=m {
                a[(i, j)] /= ajnorm;
            }
            a[(j, j)] += 1.0;
            // Apply the transformation to the remaining columns (no pivoting:
            // the norms are not updated).
            let jp1 = j + 1;
            if n >= jp1 {
                for k in jp1..=n {
                    let mut sum = 0.0;
                    for i in j..=m {
                        sum += a[(i, j)] * a[(i, k)];
                    }
                    let temp = sum / a[(j, j)];
                    for i in j..=m {
                        a[(i, k)] -= temp * a[(i, j)];
                    }
                }
            }
        }
        rdiag[j] = -ajnorm;
    }
}

/// MINPACK's `qform`: accumulates the orthogonal matrix `Q` of the factored
/// form left in `q` by [`qrfac`] (`m = n` here; `wa` is workspace).
fn qform(m: usize, n: usize, q: &mut M, wa: &mut V) {
    // Zero out the upper triangle of q in the first min(m, n) columns.
    let minmn = m.min(n);
    if minmn >= 2 {
        for j in 2..=minmn {
            for i in 1..j {
                q[(i, j)] = 0.0;
            }
        }
    }
    // Initialize the remaining columns to those of the identity matrix.
    let np1 = n + 1;
    if m >= np1 {
        for j in np1..=m {
            for i in 1..=m {
                q[(i, j)] = 0.0;
            }
            q[(j, j)] = 1.0;
        }
    }
    // Accumulate q from its factored form.
    for l in 1..=minmn {
        let k = minmn - l + 1;
        for i in k..=m {
            wa[i] = q[(i, k)];
            q[(i, k)] = 0.0;
        }
        q[(k, k)] = 1.0;
        if wa[k] != 0.0 {
            for j in k..=m {
                let mut sum = 0.0;
                for i in k..=m {
                    sum += q[(i, j)] * wa[i];
                }
                let temp = sum / wa[k];
                for i in k..=m {
                    q[(i, j)] -= temp * wa[i];
                }
            }
        }
    }
}

/// MINPACK's `dogleg`: the convex combination of the Gauss-Newton and the
/// scaled gradient directions that minimizes the model within the trust
/// region of radius `delta` (`r` the packed upper triangle by rows, `qtb =
/// Qᵀ b`; the result in `x`, `wa1` and `wa2` workspace).
fn dogleg(n: usize, r: &V, diag: &V, qtb: &V, delta: f64, x: &mut V, wa1: &mut V, wa2: &mut V) {
    // First, calculate the Gauss-Newton direction.
    let mut jj = (n * (n + 1)) / 2 + 1;
    for k in 1..=n {
        let j = n - k + 1;
        let jp1 = j + 1;
        jj -= k;
        let mut l = jj + 1;
        let mut sum = 0.0;
        if n >= jp1 {
            for i in jp1..=n {
                sum += r[l] * x[i];
                l += 1;
            }
        }
        let mut temp = r[jj];
        if temp == 0.0 {
            let mut l = j;
            for i in 1..=j {
                temp = temp.max(r[l].abs());
                l = l + n - i;
            }
            temp *= EPSMCH;
            if temp == 0.0 {
                temp = EPSMCH;
            }
        }
        x[j] = (qtb[j] - sum) / temp;
    }
    // Test whether the Gauss-Newton direction is acceptable.
    for j in 1..=n {
        wa1[j] = 0.0;
        wa2[j] = diag[j] * x[j];
    }
    let qnorm = enorm_v(n, wa2);
    if qnorm <= delta {
        return;
    }
    // The Gauss-Newton direction is not acceptable. Next, calculate the
    // scaled gradient direction.
    let mut l = 1;
    for j in 1..=n {
        let temp = qtb[j];
        for i in j..=n {
            wa1[i] += r[l] * temp;
            l += 1;
        }
        wa1[j] /= diag[j];
    }
    // Calculate the norm of the scaled gradient and test for the special
    // case in which the scaled gradient is zero.
    let gnorm = enorm_v(n, wa1);
    let mut sgnorm = 0.0;
    let mut alpha = delta / qnorm;
    if gnorm != 0.0 {
        // Calculate the point along the scaled gradient at which the
        // quadratic is minimized.
        for j in 1..=n {
            wa1[j] = (wa1[j] / gnorm) / diag[j];
        }
        let mut l = 1;
        for j in 1..=n {
            let mut sum = 0.0;
            for i in j..=n {
                sum += r[l] * wa1[i];
                l += 1;
            }
            wa2[j] = sum;
        }
        let temp = enorm_v(n, wa2);
        sgnorm = (gnorm / temp) / temp;
        // Test whether the scaled gradient direction is acceptable.
        alpha = 0.0;
        if sgnorm < delta {
            // The scaled gradient direction is not acceptable. Finally,
            // calculate the point along the dogleg at which the quadratic
            // is minimized.
            let bnorm = enorm_v(n, qtb);
            let mut temp = (bnorm / gnorm) * (bnorm / qnorm) * (sgnorm / delta);
            let dq = delta / qnorm;
            let sd = sgnorm / delta;
            temp = temp - dq * (sd * sd)
                + ((temp - dq) * (temp - dq) + (1.0 - dq * dq) * (1.0 - sd * sd)).sqrt();
            alpha = (dq * (1.0 - sd * sd)) / temp;
        }
    }
    // Form the appropriate convex combination of the Gauss-Newton direction
    // and the scaled gradient direction.
    let temp = (1.0 - alpha) * sgnorm.min(delta);
    for j in 1..=n {
        x[j] = temp * wa1[j] + alpha * x[j];
    }
}

/// MINPACK's `r1updt`: given the packed upper trapezoidal `s` (`m × n`, by
/// rows), computes an orthogonal `Q` such that `(s + u vᵀ) Q` is again upper
/// trapezoidal; `v` and `w` return the Givens rotations. Returns `sing`
/// (a zero diagonal element of the result).
fn r1updt(m: usize, n: usize, s: &mut V, u: &V, v: &mut V, w: &mut V) -> bool {
    // Initialize the diagonal element pointer.
    let mut jj = (n * (2 * m - n + 1)) / 2 - (m - n);
    // Move the nontrivial part of the last column of s into w.
    let mut l = jj;
    for i in n..=m {
        w[i] = s[l];
        l += 1;
    }
    // Rotate the vector v into a multiple of the n-th unit vector in such a
    // way that a spike is introduced into w.
    let nm1 = n.saturating_sub(1);
    if nm1 >= 1 {
        for nmj in 1..=nm1 {
            let j = n - nmj;
            jj -= m - j + 1;
            w[j] = 0.0;
            if v[j] != 0.0 {
                // Determine a Givens rotation which eliminates the j-th
                // element of v.
                let (sin, cos, tau);
                if v[n].abs() < v[j].abs() {
                    let cotan = v[n] / v[j];
                    sin = 0.5 / (0.25 + 0.25 * (cotan * cotan)).sqrt();
                    cos = sin * cotan;
                    tau = if cos.abs() * GIANT > 1.0 {
                        1.0 / cos
                    } else {
                        1.0
                    };
                } else {
                    let tan = v[j] / v[n];
                    cos = 0.5 / (0.25 + 0.25 * (tan * tan)).sqrt();
                    sin = cos * tan;
                    tau = sin;
                }
                // Apply the transformation to v and store the information
                // necessary to recover the Givens rotation.
                v[n] = sin * v[j] + cos * v[n];
                v[j] = tau;
                // Apply the transformation to s and extend the spike in w.
                let mut l = jj;
                for i in j..=m {
                    let temp = cos * s[l] - sin * w[i];
                    w[i] = sin * s[l] + cos * w[i];
                    s[l] = temp;
                    l += 1;
                }
            }
        }
    }
    // Add the spike from the rank 1 update to w.
    for i in 1..=m {
        w[i] += v[n] * u[i];
    }
    // Eliminate the spike.
    let mut sing = false;
    if nm1 >= 1 {
        for j in 1..=nm1 {
            if w[j] != 0.0 {
                // Determine a Givens rotation which eliminates the j-th
                // element of the spike.
                let (sin, cos, tau);
                if s[jj].abs() < w[j].abs() {
                    let cotan = s[jj] / w[j];
                    sin = 0.5 / (0.25 + 0.25 * (cotan * cotan)).sqrt();
                    cos = sin * cotan;
                    tau = if cos.abs() * GIANT > 1.0 {
                        1.0 / cos
                    } else {
                        1.0
                    };
                } else {
                    let tan = w[j] / s[jj];
                    cos = 0.5 / (0.25 + 0.25 * (tan * tan)).sqrt();
                    sin = cos * tan;
                    tau = sin;
                }
                // Apply the transformation to s and reduce the spike in w.
                let mut l = jj;
                for i in j..=m {
                    let temp = cos * s[l] + sin * w[i];
                    w[i] = -sin * s[l] + cos * w[i];
                    s[l] = temp;
                    l += 1;
                }
                // Store the information necessary to recover the Givens
                // rotation.
                w[j] = tau;
            }
            // Test for zero diagonal elements in the output s.
            if s[jj] == 0.0 {
                sing = true;
            }
            jj += m - j + 1;
        }
    }
    // Move w back into the last column of the output s.
    let mut l = jj;
    for i in n..=m {
        s[l] = w[i];
        l += 1;
    }
    if s[jj] == 0.0 {
        sing = true;
    }
    sing
}

/// The cosine and sine of a Givens rotation stored by [`r1updt`].
#[inline]
fn rotation(t: f64) -> (f64, f64) {
    if t.abs() > 1.0 {
        let cos = 1.0 / t;
        (cos, (1.0 - cos * cos).sqrt())
    } else {
        let sin = t;
        ((1.0 - sin * sin).sqrt(), sin)
    }
}

/// MINPACK's `r1mpyq` on the `m × n` matrix `a`: applies the two sets of
/// Givens rotations of [`r1updt`] (`v`, `w`) from the right.
fn r1mpyq(m: usize, n: usize, a: &mut [f64], lda: usize, v: &V, w: &V) {
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * lda;
    let nm1 = n.saturating_sub(1);
    if nm1 < 1 {
        return;
    }
    // Apply the first set of Givens rotations to a.
    for nmj in 1..=nm1 {
        let j = n - nmj;
        let (cos, sin) = rotation(v[j]);
        for i in 1..=m {
            let temp = cos * a[at(i, j)] - sin * a[at(i, n)];
            a[at(i, n)] = sin * a[at(i, j)] + cos * a[at(i, n)];
            a[at(i, j)] = temp;
        }
    }
    // Apply the second set of Givens rotations to a.
    for j in 1..=nm1 {
        let (cos, sin) = rotation(w[j]);
        for i in 1..=m {
            let temp = cos * a[at(i, j)] + sin * a[at(i, n)];
            a[at(i, n)] = -sin * a[at(i, j)] + cos * a[at(i, n)];
            a[at(i, j)] = temp;
        }
    }
}

/// Options of [`hybrj`] (MINPACK's arguments; [`HybrjOptions::default`]
/// gives scipy's `fsolve` defaults for `n` variables via [`fsolve`]).
#[derive(Clone, Debug, PartialEq)]
pub struct HybrjOptions {
    /// Termination when the relative error between two consecutive iterates
    /// is at most `xtol` (scipy default `1.49012e-8`).
    pub xtol: f64,
    /// Maximum number of function evaluations (Jacobian evaluations are not
    /// counted); 0 means scipy's default `100·(n + 1)`.
    pub maxfev: usize,
    /// Initial step bound factor (scipy default 100).
    pub factor: f64,
    /// Positive variable scales (MINPACK `mode = 2`); `None` scales by the
    /// column norms of the Jacobian (`mode = 1`, scipy's default).
    pub diag: Option<Vec<f64>>,
}

impl Default for HybrjOptions {
    fn default() -> Self {
        Self {
            xtol: 1.49012e-8,
            maxfev: 0,
            factor: 100.0,
            diag: None,
        }
    }
}

/// The result of [`hybrj`].
#[derive(Clone, Debug, PartialEq)]
pub struct HybrjResult {
    /// The final estimate of the solution.
    pub x: Vec<f64>,
    /// The function values at `x`.
    pub fvec: Vec<f64>,
    /// MINPACK's `info`: 1 converged (`xtol`), 2 too many function
    /// evaluations, 3 `xtol` too small, 4 not making progress (Jacobian
    /// evaluations), 5 not making progress (iterations).
    pub info: i32,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Number of Jacobian evaluations.
    pub njev: usize,
}

impl HybrjResult {
    /// scipy's message for [`HybrjResult::info`].
    pub fn message(&self) -> &'static str {
        match self.info {
            1 => "The solution converged.",
            2 => "The number of calls to function has reached maxfev.",
            3 => "xtol is too small, no further improvement in the approximate solution is possible.",
            4 => "The iteration is not making good progress, as measured by the improvement from the last five Jacobian evaluations.",
            5 => "The iteration is not making good progress, as measured by the improvement from the last ten iterations.",
            _ => "Improper input parameters.",
        }
    }
}

/// MINPACK's `hybrj`: finds a zero of the `n` functions `fcn(x)` (length
/// `n`) with Jacobian `jac(x)` (`n × n`, `J[i, j] = ∂f_i/∂x_j`) from `x0`.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for improper input (MINPACK's `info =
/// 0`: empty `x0`, negative `xtol`, zero `maxfev`, non-positive `factor` or
/// `diag`, a Jacobian of the wrong shape), and the errors of `fcn` and `jac`.
pub fn hybrj<F, J>(
    mut fcn: F,
    mut jac: J,
    x0: &[f64],
    options: &HybrjOptions,
) -> Result<HybrjResult>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>>,
    J: FnMut(&[f64]) -> Result<Matrix>,
{
    let n = x0.len();
    let xtol = options.xtol;
    let maxfev = options.maxfev;
    let factor = options.factor;
    let mode = if options.diag.is_some() { 2 } else { 1 };
    if n == 0 || xtol < 0.0 || maxfev == 0 || factor <= 0.0 {
        return Err(BotorchError::InvalidArgument(
            "improper input parameters for hybrj (n, xtol, maxfev or factor)".into(),
        ));
    }
    let mut diag = V::zeros(n);
    if let Some(d) = &options.diag {
        if d.len() != n || d.iter().any(|&v| v <= 0.0) {
            return Err(BotorchError::InvalidArgument(
                "diag must have one positive entry per variable".into(),
            ));
        }
        diag = V(d.clone());
    }
    let lr = n * (n + 1) / 2;
    let mut x = V(x0.to_vec());
    let mut fvec = V::zeros(n);
    let mut fjac = M {
        n,
        a: vec![0.0; n * n],
    };
    let mut r = V::zeros(lr);
    let mut qtf = V::zeros(n);
    let mut wa1 = V::zeros(n);
    let mut wa2 = V::zeros(n);
    let mut wa3 = V::zeros(n);
    let mut wa4 = V::zeros(n);

    let eval_f = |fcn: &mut F, x: &V, out: &mut V| -> Result<()> {
        let f = fcn(&x.0)?;
        if f.len() != n {
            return Err(BotorchError::DimensionMismatch(format!(
                "the function returned {} values for {n} variables",
                f.len()
            )));
        }
        out.0.copy_from_slice(&f);
        Ok(())
    };
    let eval_j = |jac: &mut J, x: &V, out: &mut M| -> Result<()> {
        let j = jac(&x.0)?;
        if j.shape() != (n, n) {
            return Err(BotorchError::DimensionMismatch(format!(
                "the Jacobian must be {n}x{n}, got {}x{}",
                j.nrows(),
                j.ncols()
            )));
        }
        for c in 0..n {
            for rr in 0..n {
                out.a[rr + c * n] = j[(rr, c)];
            }
        }
        Ok(())
    };

    // Evaluate the function at the starting point and calculate its norm.
    eval_f(&mut fcn, &x, &mut fvec)?;
    let mut nfev = 1usize;
    let mut njev = 0usize;
    let mut fnorm = enorm_v(n, &fvec);

    // Initialize iteration counter and monitors.
    let mut iter = 1usize;
    let mut ncsuc = 0usize;
    let mut ncfail = 0usize;
    let mut nslow1 = 0usize;
    let mut nslow2 = 0usize;
    let mut xnorm = 0.0;
    let mut delta = 0.0;
    let info;

    // Beginning of the outer loop.
    'outer: loop {
        let mut jeval = true;
        // Calculate the Jacobian matrix.
        eval_j(&mut jac, &x, &mut fjac)?;
        njev += 1;
        // Compute the QR factorization of the Jacobian.
        qrfac(n, n, &mut fjac, &mut wa1, &mut wa2, &mut wa3);
        // On the first iteration and if mode is 1, scale according to the
        // norms of the columns of the initial Jacobian.
        if iter == 1 {
            if mode != 2 {
                for j in 1..=n {
                    diag[j] = wa2[j];
                    if wa2[j] == 0.0 {
                        diag[j] = 1.0;
                    }
                }
            }
            // On the first iteration, calculate the norm of the scaled x and
            // initialize the step bound delta.
            for j in 1..=n {
                wa3[j] = diag[j] * x[j];
            }
            xnorm = enorm_v(n, &wa3);
            delta = factor * xnorm;
            if delta == 0.0 {
                delta = factor;
            }
        }
        // Form (q transpose)*fvec and store in qtf.
        for i in 1..=n {
            qtf[i] = fvec[i];
        }
        for j in 1..=n {
            if fjac[(j, j)] != 0.0 {
                let mut sum = 0.0;
                for i in j..=n {
                    sum += fjac[(i, j)] * qtf[i];
                }
                let temp = -sum / fjac[(j, j)];
                for i in j..=n {
                    qtf[i] += fjac[(i, j)] * temp;
                }
            }
        }
        // Copy the triangular factor of the QR factorization into r.
        let mut sing = false;
        for j in 1..=n {
            let mut l = j;
            if j >= 2 {
                for i in 1..j {
                    r[l] = fjac[(i, j)];
                    l = l + n - i;
                }
            }
            r[l] = wa1[j];
            if wa1[j] == 0.0 {
                sing = true;
            }
        }
        let _ = sing;
        // Accumulate the orthogonal factor in fjac.
        qform(n, n, &mut fjac, &mut wa1);
        // Rescale if necessary.
        if mode != 2 {
            for j in 1..=n {
                diag[j] = diag[j].max(wa2[j]);
            }
        }

        // Beginning of the inner loop.
        loop {
            // Determine the direction p.
            dogleg(n, &r, &diag, &qtf, delta, &mut wa1, &mut wa2, &mut wa3);
            // Store the direction p and x + p. Calculate the norm of p.
            for j in 1..=n {
                wa1[j] = -wa1[j];
                wa2[j] = x[j] + wa1[j];
                wa3[j] = diag[j] * wa1[j];
            }
            let pnorm = enorm_v(n, &wa3);
            // On the first iteration, adjust the initial step bound.
            if iter == 1 {
                delta = delta.min(pnorm);
            }
            // Evaluate the function at x + p and calculate its norm.
            eval_f(&mut fcn, &wa2, &mut wa4)?;
            nfev += 1;
            let fnorm1 = enorm_v(n, &wa4);
            // Compute the scaled actual reduction.
            let mut actred = -1.0;
            if fnorm1 < fnorm {
                let t = fnorm1 / fnorm;
                actred = 1.0 - t * t;
            }
            // Compute the scaled predicted reduction.
            let mut l = 1;
            for i in 1..=n {
                let mut sum = 0.0;
                for j in i..=n {
                    sum += r[l] * wa1[j];
                    l += 1;
                }
                wa3[i] = qtf[i] + sum;
            }
            let temp = enorm_v(n, &wa3);
            let mut prered = 0.0;
            if temp < fnorm {
                let t = temp / fnorm;
                prered = 1.0 - t * t;
            }
            // Compute the ratio of the actual to the predicted reduction.
            let mut ratio = 0.0;
            if prered > 0.0 {
                ratio = actred / prered;
            }
            // Update the step bound.
            if ratio < 0.1 {
                ncsuc = 0;
                ncfail += 1;
                delta *= 0.5;
            } else {
                ncfail = 0;
                ncsuc += 1;
                if ratio >= 0.5 || ncsuc > 1 {
                    delta = delta.max(pnorm / 0.5);
                }
                if (ratio - 1.0).abs() <= 0.1 {
                    delta = pnorm / 0.5;
                }
            }
            // Test for successful iteration.
            if ratio >= 1e-4 {
                // Successful iteration. Update x, fvec, and their norms.
                for j in 1..=n {
                    x[j] = wa2[j];
                    wa2[j] = diag[j] * x[j];
                    fvec[j] = wa4[j];
                }
                xnorm = enorm_v(n, &wa2);
                fnorm = fnorm1;
                iter += 1;
            }
            // Determine the progress of the iteration.
            nslow1 += 1;
            if actred >= 1e-3 {
                nslow1 = 0;
            }
            if jeval {
                nslow2 += 1;
            }
            if actred >= 0.1 {
                nslow2 = 0;
            }
            // Test for convergence.
            if delta <= xtol * xnorm || fnorm == 0.0 {
                info = 1;
                break 'outer;
            }
            // Tests for termination and stringent tolerances.
            let mut stop = 0;
            if nfev >= maxfev {
                stop = 2;
            }
            if 0.1 * (0.1 * delta).max(pnorm) <= EPSMCH * xnorm {
                stop = 3;
            }
            if nslow2 == 5 {
                stop = 4;
            }
            if nslow1 == 10 {
                stop = 5;
            }
            if stop != 0 {
                info = stop;
                break 'outer;
            }
            // Criterion for recalculating the Jacobian.
            if ncfail == 2 {
                continue 'outer;
            }
            // Calculate the rank one modification to the Jacobian and update
            // qtf if necessary.
            for j in 1..=n {
                let mut sum = 0.0;
                for i in 1..=n {
                    sum += fjac[(i, j)] * wa4[i];
                }
                wa2[j] = (sum - wa3[j]) / pnorm;
                wa1[j] = diag[j] * ((diag[j] * wa1[j]) / pnorm);
                if ratio >= 1e-4 {
                    qtf[j] = sum;
                }
            }
            // Compute the QR factorization of the updated Jacobian.
            r1updt(n, n, &mut r, &wa1, &mut wa2, &mut wa3);
            r1mpyq(n, n, &mut fjac.a, n, &wa2, &wa3);
            r1mpyq(1, n, &mut qtf.0, 1, &wa2, &wa3);
            // End of the inner loop.
            jeval = false;
        }
    }
    Ok(HybrjResult {
        x: x.0,
        fvec: fvec.0,
        info,
        nfev,
        njev,
    })
}

/// `scipy.optimize.fsolve(func, x0, fprime=jac, xtol=xtol, maxfev=maxfev)`
/// (`maxfev = 0`: scipy's default `100·(n + 1)`), returning the full result
/// (scipy returns `x` also when the solver did not converge).
///
/// # Errors
/// See [`hybrj`].
pub fn fsolve<F, J>(fcn: F, jac: J, x0: &[f64], xtol: f64, maxfev: usize) -> Result<HybrjResult>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>>,
    J: FnMut(&[f64]) -> Result<Matrix>,
{
    let maxfev = if maxfev == 0 {
        100 * (x0.len() + 1)
    } else {
        maxfev
    };
    hybrj(
        fcn,
        jac,
        x0,
        &HybrjOptions {
            xtol,
            maxfev,
            ..HybrjOptions::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enorm_matches_plain_norm() {
        let x = [3.0, 4.0, 0.0, 1e-25, 1e25];
        assert_eq!(enorm(2, &x), 5.0);
        assert!((enorm(5, &x) - 1e25).abs() / 1e25 < 1e-15);
        assert_eq!(enorm(1, &[0.0]), 0.0);
    }

    #[test]
    fn solves_rosenbrock_system() {
        let f = |x: &[f64]| Ok(vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]);
        let j = |x: &[f64]| {
            Ok(Matrix::from_row_slice(
                2,
                2,
                &[-20.0 * x[0], 10.0, -1.0, 0.0],
            ))
        };
        let r = fsolve(f, j, &[-1.2, 1.0], 1.49012e-8, 0).unwrap();
        assert_eq!(r.info, 1);
        assert!((r.x[0] - 1.0).abs() < 1e-10 && (r.x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn improper_input_is_an_error() {
        let f = |x: &[f64]| Ok(x.to_vec());
        let j = |_: &[f64]| Ok(Matrix::identity(1, 1));
        assert!(fsolve(f, j, &[], 1e-8, 0).is_err());
        let f = |x: &[f64]| Ok(x.to_vec());
        let j = |_: &[f64]| Ok(Matrix::identity(1, 1));
        assert!(fsolve(f, j, &[1.0], -1.0, 0).is_err());
    }
}
