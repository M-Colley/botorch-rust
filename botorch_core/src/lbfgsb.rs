//! L-BFGS-B 3.0 for bound-constrained minimization, with the semantics of
//! `scipy.optimize.minimize(method="L-BFGS-B")`.
//!
//! This is a direct Rust port of L-BFGS-B version 3.0 (Zhu, Byrd, Lu, Nocedal;
//! 3.0 update by Morales and Nocedal), translated from Stephen Becker's C
//! version (L-BFGS-B-C, BSD-3-Clause). The routine structure (`active`,
//! `cauchy`, `bmv`, `formk`, `formt`, `cmprlb`, `subsm`, `matupd`, `freev`,
//! `hpsolb`, `projgr`, `lnsrlb`, `dcsrch`, `dcstep`, `dpofa`, `dtrsl`) and the
//! floating-point operations follow the reference code; internally 1-based
//! indexing is kept so that the port can be checked line by line. The
//! reverse-communication driver is replaced by direct objective calls with the
//! stopping logic of scipy's driver (`maxiter` and `maxfun` are checked after
//! every iteration, `factr = ftol / eps`, at most `maxls` function evaluations
//! per line search).
//!
//! References:
//! * R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, "A limited memory algorithm for
//!   bound constrained optimization", SIAM J. Sci. Comput. 16(5), 1995.
//! * C. Zhu, R. H. Byrd, P. Lu and J. Nocedal, "Algorithm 778: L-BFGS-B",
//!   ACM TOMS 23(4), 1997.
//! * J. L. Morales and J. Nocedal, "Remark on Algorithm 778", ACM TOMS 38(1), 2011.
//!
//! License of the original code (retained as required): L-BFGS-B is released
//! under the "New BSD License" (3-clause). Copyright (c) 2011 Ciyou Zhu,
//! Richard Byrd, Jorge Nocedal and Jose Luis Morales; C version Copyright (c)
//! 2015 Stephen Becker. Redistribution and use in source and binary forms, with
//! or without modification, are permitted provided that the copyright notice,
//! this list of conditions and the disclaimer are retained; neither the names
//! of the copyright holders nor of the contributors may be used to endorse or
//! promote derived products without prior written permission. THE SOFTWARE IS
//! PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

#![allow(clippy::needless_range_loop)]

use std::fmt;
use std::ops::{Index, IndexMut};

/// Options of [`minimize`], named after `scipy.optimize.minimize(method="L-BFGS-B")`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LbfgsbOptions {
    /// Maximum number of variable-metric corrections (scipy default 10).
    pub maxcor: usize,
    /// Relative function reduction tolerance: the iteration stops when
    /// `(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= ftol`
    /// (scipy default `2.220446049250313e-9`, i.e. `factr = 1e7`).
    pub ftol: f64,
    /// Projected-gradient tolerance: stop when `max_i |proj g_i| <= gtol`
    /// (scipy default 1e-5).
    pub gtol: f64,
    /// Maximum number of iterations (scipy default 15000).
    pub maxiter: usize,
    /// Maximum number of function evaluations (scipy default 15000).
    pub maxfun: usize,
    /// Maximum number of line-search steps per iteration (scipy default 20).
    pub maxls: usize,
}

impl Default for LbfgsbOptions {
    fn default() -> Self {
        Self {
            maxcor: 10,
            ftol: 2.220_446_049_250_313e-9,
            gtol: 1e-5,
            maxiter: 15_000,
            maxfun: 15_000,
            maxls: 20,
        }
    }
}

/// Termination reason of [`minimize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LbfgsbStatus {
    /// `max |proj g| <= gtol`.
    ConvergedProjectedGradient,
    /// Relative reduction of the objective `<= ftol`.
    ConvergedRelativeReduction,
    /// The iteration limit was reached.
    MaxIterations,
    /// The function-evaluation limit was exceeded.
    MaxEvaluations,
    /// The callback requested termination.
    CallbackStop,
    /// The line search failed without stored corrections
    /// (`ABNORMAL_TERMINATION_IN_LNSRCH`).
    AbnormalLineSearch,
}

impl LbfgsbStatus {
    /// Whether the run converged (scipy's `success`).
    pub fn success(self) -> bool {
        matches!(
            self,
            LbfgsbStatus::ConvergedProjectedGradient | LbfgsbStatus::ConvergedRelativeReduction
        )
    }

    /// scipy's termination message.
    pub fn message(self) -> &'static str {
        match self {
            LbfgsbStatus::ConvergedProjectedGradient => {
                "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
            }
            LbfgsbStatus::ConvergedRelativeReduction => {
                "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
            }
            LbfgsbStatus::MaxIterations => "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
            LbfgsbStatus::MaxEvaluations => "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
            LbfgsbStatus::CallbackStop => "STOP: CALLBACK REQUESTED STOP",
            LbfgsbStatus::AbnormalLineSearch => "ABNORMAL: ABNORMAL_TERMINATION_IN_LNSRCH",
        }
    }
}

/// Result of [`minimize`] (scipy's `OptimizeResult`).
#[derive(Clone, Debug, PartialEq)]
pub struct LbfgsbResult {
    /// Final iterate.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub f: f64,
    /// Gradient at `x`.
    pub g: Vec<f64>,
    /// Number of iterations (scipy's `nit`).
    pub iterations: usize,
    /// Number of objective evaluations (scipy's `nfev`).
    pub evaluations: usize,
    /// Termination reason.
    pub status: LbfgsbStatus,
}

/// Errors of [`minimize`].
#[derive(Clone, Debug, PartialEq)]
pub enum LbfgsbError<E> {
    /// Invalid arguments (dimensions, bounds or options).
    InvalidInput(String),
    /// The objective returned an error; the optimization was aborted.
    Objective(E),
}

impl<E: fmt::Display> fmt::Display for LbfgsbError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LbfgsbError::InvalidInput(msg) => write!(f, "invalid L-BFGS-B input: {msg}"),
            LbfgsbError::Objective(e) => write!(f, "objective evaluation failed: {e}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for LbfgsbError<E> {}

/// Minimizes `fun` subject to `lower <= x <= upper` (use `±inf` for missing
/// bounds), starting from `x0` (clipped into the box). `fun` returns the value
/// and gradient; an error aborts the optimization.
///
/// # Errors
/// [`LbfgsbError::InvalidInput`] for inconsistent arguments and
/// [`LbfgsbError::Objective`] if `fun` fails.
pub fn minimize<F, E>(
    fun: F,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: &LbfgsbOptions,
) -> Result<LbfgsbResult, LbfgsbError<E>>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>,
{
    minimize_with_callback(fun, x0, lower, upper, options, |_, _, _| false)
}

/// [`minimize`] with a callback invoked after every iteration with the
/// iteration number, the current iterate and its objective value; returning
/// `true` stops the optimization (scipy's `callback` raising `StopIteration`).
///
/// # Errors
/// See [`minimize`].
pub fn minimize_with_callback<F, E, C>(
    mut fun: F,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: &LbfgsbOptions,
    mut callback: C,
) -> Result<LbfgsbResult, LbfgsbError<E>>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), E>,
    C: FnMut(usize, &[f64], f64) -> bool,
{
    let n = x0.len();
    if n == 0 {
        return Err(LbfgsbError::InvalidInput("x0 must not be empty".into()));
    }
    if lower.len() != n || upper.len() != n {
        return Err(LbfgsbError::InvalidInput(format!(
            "bounds have lengths {} and {} but x0 has length {n}",
            lower.len(),
            upper.len()
        )));
    }
    if options.maxcor == 0 {
        return Err(LbfgsbError::InvalidInput("maxcor must be positive".into()));
    }
    if options.ftol.is_nan() || options.ftol < 0.0 || options.gtol.is_nan() || options.gtol < 0.0 {
        return Err(LbfgsbError::InvalidInput(
            "ftol and gtol must be non-negative".into(),
        ));
    }
    if x0.iter().any(|v| v.is_nan()) || lower.iter().chain(upper).any(|v| v.is_nan()) {
        return Err(LbfgsbError::InvalidInput(
            "x0 and bounds must not contain NaN".into(),
        ));
    }
    // nbd: 0 unbounded, 1 lower only, 2 both, 3 upper only (scipy's mapping).
    let mut nbd = I1::new(n);
    let mut l = V1::new(n);
    let mut u = V1::new(n);
    let mut x = V1::new(n);
    for i in 1..=n {
        let (lo, hi) = (lower[i - 1], upper[i - 1]);
        let (has_lo, has_hi) = (lo.is_finite(), hi.is_finite());
        nbd[i] = match (has_lo, has_hi) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        };
        l[i] = if has_lo { lo } else { 0.0 };
        u[i] = if has_hi { hi } else { 0.0 };
        if has_lo && has_hi && lo > hi {
            return Err(LbfgsbError::InvalidInput(format!(
                "infeasible bounds for variable {}: {lo} > {hi}",
                i - 1
            )));
        }
        // scipy clips the starting point into the bounds.
        x[i] = x0[i - 1].max(lo).min(hi);
    }

    let m = options.maxcor;
    let factr = options.ftol / f64::EPSILON;
    let pgtol = options.gtol;
    let maxls = options.maxls;
    let mut ws = Mat::new(n, m);
    let mut wy = Mat::new(n, m);
    let mut sy = Mat::new(m, m);
    let mut ss = Mat::new(m, m);
    let mut wt = Mat::new(m, m);
    let mut wn = Mat::new(2 * m, 2 * m);
    let mut snd = Mat::new(2 * m, 2 * m);
    let mut z = V1::new(n);
    let mut r = V1::new(n);
    let mut d = V1::new(n);
    let mut t = V1::new(n);
    let mut xp = V1::new(n);
    let mut wa = V1::new(8 * m);
    let mut index = I1::new(n);
    let mut iwhere = I1::new(n);
    let mut indx2 = I1::new(n);
    let mut g = V1::new(n);

    let mut evaluations = 0usize;
    let mut evaluate =
        |x: &V1, g: &mut V1, evaluations: &mut usize| -> Result<f64, LbfgsbError<E>> {
            *evaluations += 1;
            let (f, grad) = fun(&x.0).map_err(LbfgsbError::Objective)?;
            if grad.len() != n {
                return Err(LbfgsbError::InvalidInput(format!(
                    "gradient has length {} but x has length {n}",
                    grad.len()
                )));
            }
            g.0.copy_from_slice(&grad);
            Ok(f)
        };

    // ---- START ------------------------------------------------------------
    let epsmch = f64::EPSILON;
    let mut col = 0usize;
    let mut head = 1usize;
    let mut theta = 1.0;
    let mut iupdat = 0usize;
    let mut updatd = false;
    let mut iback;
    let mut itail = 0usize;
    let mut ileave = 0usize;
    let mut nenter = 0usize;
    let mut fold;
    let mut dnorm;
    let mut gd;
    let mut stpmx;
    let mut stp;
    let mut gdold = 0.0;
    let mut dtd;
    let mut iter = 0usize;
    let mut nfree = n;
    let mut ifun;
    let tol = factr * epsmch;
    let mut info: i32;
    let mut n_iterations = 0usize;

    let (_prjctd, cnstnd, boxed) = active(n, &l, &u, &nbd, &mut x, &mut iwhere);

    // ---- FG_START ---------------------------------------------------------
    let mut f = evaluate(&x, &mut g, &mut evaluations)?;
    let mut sbgnrm = projgr(n, &l, &u, &nbd, &x, &g);
    if sbgnrm <= pgtol {
        return Ok(LbfgsbResult {
            x: x.0,
            f,
            g: g.0,
            iterations: 0,
            evaluations,
            status: LbfgsbStatus::ConvergedProjectedGradient,
        });
    }

    let status = 'outer: loop {
        // ---- L222: compute the search direction ----------------------------
        let wrk = if !cnstnd && col > 0 {
            z.0.copy_from_slice(&x.0);
            updatd
        } else {
            info = cauchy(
                n,
                &x,
                &l,
                &u,
                &nbd,
                &g,
                &mut indx2,
                &mut iwhere,
                &mut t,
                &mut d,
                &mut z,
                m,
                &wy,
                &ws,
                &sy,
                &wt,
                theta,
                col,
                head,
                &mut wa,
                sbgnrm,
                epsmch,
            );
            if info != 0 {
                // Singular triangular system in bmv: refresh the memory.
                col = 0;
                head = 1;
                theta = 1.0;
                iupdat = 0;
                updatd = false;
                continue 'outer;
            }
            freev(
                n,
                &mut nfree,
                &mut index,
                &mut nenter,
                &mut ileave,
                &mut indx2,
                &iwhere,
                updatd,
                cnstnd,
                iter,
            )
        };
        // ---- L333 ----------------------------------------------------------
        if nfree != 0 && col != 0 {
            info = 0;
            if wrk {
                info = formk(
                    n, nfree, &index, nenter, ileave, &indx2, iupdat, updatd, &mut wn, &mut snd, m,
                    &ws, &wy, &sy, theta, col, head,
                );
            }
            if info == 0 {
                info = cmprlb(
                    n, m, &x, &g, &ws, &wy, &sy, &wt, &z, &mut r, &mut wa, &index, theta, col,
                    head, nfree, cnstnd,
                );
            }
            if info == 0 {
                info = subsm(
                    n, m, nfree, &index, &l, &u, &nbd, &mut z, &mut r, &mut xp, &ws, &wy, theta,
                    &x, &g, col, head, &mut wa, &wn,
                );
            }
            if info != 0 {
                col = 0;
                head = 1;
                theta = 1.0;
                iupdat = 0;
                updatd = false;
                continue 'outer;
            }
        }
        // ---- L555: search direction d = z - x -------------------------------
        for i in 1..=n {
            d[i] = z[i] - x[i];
        }
        // ---- L666: line search (lnsrlb) -----------------------------------
        dtd = ddot(n, &d.0, &d.0);
        dnorm = dtd.sqrt();
        stpmx = 1e10;
        if cnstnd {
            if iter == 0 {
                stpmx = 1.0;
            } else {
                for i in 1..=n {
                    let a1 = d[i];
                    if nbd[i] != 0 {
                        if a1 < 0.0 && nbd[i] <= 2 {
                            let a2 = l[i] - x[i];
                            if a2 >= 0.0 {
                                stpmx = 0.0;
                            } else if a1 * stpmx < a2 {
                                stpmx = a2 / a1;
                            }
                        } else if a1 > 0.0 && nbd[i] >= 2 {
                            let a2 = u[i] - x[i];
                            if a2 <= 0.0 {
                                stpmx = 0.0;
                            } else if a1 * stpmx > a2 {
                                stpmx = a2 / a1;
                            }
                        }
                    }
                }
            }
        }
        stp = if iter == 0 && !boxed {
            (1.0 / dnorm).min(stpmx)
        } else {
            1.0
        };
        t.0.copy_from_slice(&x.0);
        r.0.copy_from_slice(&g.0);
        fold = f;
        ifun = 0usize;
        iback = 0usize;
        info = 0;
        let mut ls = LineSearch::new();
        loop {
            gd = ddot(n, &g.0, &d.0);
            if ifun == 0 {
                gdold = gd;
                if gd >= 0.0 {
                    // The directional derivative is non-negative.
                    info = -4;
                    break;
                }
            }
            ls.dcsrch(f, gd, &mut stp, FTOL, GTOL, XTOL, STPMIN, stpmx);
            if !ls.task.is_warning() && !ls.task.is_converged() {
                ifun += 1;
                iback = ifun - 1;
                if stp == 1.0 {
                    x.0.copy_from_slice(&z.0);
                } else {
                    for i in 1..=n {
                        x[i] = stp * d[i] + t[i];
                    }
                }
                if iback >= maxls {
                    break;
                }
                f = evaluate(&x, &mut g, &mut evaluations)?;
            } else {
                break;
            }
        }
        if info != 0 || iback >= maxls {
            // Restore the previous iterate.
            x.0.copy_from_slice(&t.0);
            g.0.copy_from_slice(&r.0);
            f = fold;
            if col == 0 {
                break 'outer LbfgsbStatus::AbnormalLineSearch;
            }
            // Refresh the memory and restart the iteration.
            col = 0;
            head = 1;
            theta = 1.0;
            iupdat = 0;
            updatd = false;
            continue 'outer;
        }
        // ---- NEW_X --------------------------------------------------------
        iter += 1;
        sbgnrm = projgr(n, &l, &u, &nbd, &x, &g);
        n_iterations += 1;
        if callback(n_iterations, &x.0, f) {
            break 'outer LbfgsbStatus::CallbackStop;
        }
        if n_iterations >= options.maxiter {
            break 'outer LbfgsbStatus::MaxIterations;
        }
        if evaluations > options.maxfun {
            break 'outer LbfgsbStatus::MaxEvaluations;
        }
        // ---- L777: convergence tests -------------------------------------
        if sbgnrm <= pgtol {
            break 'outer LbfgsbStatus::ConvergedProjectedGradient;
        }
        let ddum = fold.abs().max(f.abs()).max(1.0);
        if fold - f <= tol * ddum {
            break 'outer LbfgsbStatus::ConvergedRelativeReduction;
        }
        // Compute d = newx - oldx, r = newg - oldg, rr = y'y and dr = y's.
        for i in 1..=n {
            r[i] = g[i] - r[i];
        }
        let rr = ddot(n, &r.0, &r.0);
        let (dr, ddum) = if stp == 1.0 {
            (gd - gdold, -gdold)
        } else {
            let dr = (gd - gdold) * stp;
            for i in 1..=n {
                d[i] *= stp;
            }
            (dr, -gdold * stp)
        };
        if dr <= epsmch * ddum {
            // Skip the L-BFGS update.
            updatd = false;
            continue 'outer;
        }
        // Update the L-BFGS matrix.
        updatd = true;
        iupdat += 1;
        matupd(
            n, m, &mut ws, &mut wy, &mut sy, &mut ss, &d, &r, &mut itail, iupdat, &mut col,
            &mut head, &mut theta, rr, dr, stp, dtd,
        );
        if formt(m, &mut wt, &sy, &ss, col, theta) != 0 {
            col = 0;
            head = 1;
            theta = 1.0;
            iupdat = 0;
            updatd = false;
        }
    };
    Ok(LbfgsbResult {
        x: x.0,
        f,
        g: g.0,
        iterations: n_iterations,
        evaluations,
        status,
    })
}

// ---------------------------------------------------------------------------
// 1-based storage helpers
// ---------------------------------------------------------------------------

/// 1-based vector of `f64`.
#[derive(Clone, Debug)]
struct V1(Vec<f64>);

impl V1 {
    fn new(n: usize) -> Self {
        V1(vec![0.0; n])
    }
}

impl Index<usize> for V1 {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        &self.0[i - 1]
    }
}

impl IndexMut<usize> for V1 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i - 1]
    }
}

/// 1-based vector of integers.
#[derive(Clone, Debug)]
struct I1(Vec<i64>);

impl I1 {
    fn new(n: usize) -> Self {
        I1(vec![0; n])
    }
}

impl Index<usize> for I1 {
    type Output = i64;
    #[inline]
    fn index(&self, i: usize) -> &i64 {
        &self.0[i - 1]
    }
}

impl IndexMut<usize> for I1 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut i64 {
        &mut self.0[i - 1]
    }
}

/// Column-major matrix with 1-based `(row, col)` indexing (Fortran layout).
#[derive(Clone, Debug)]
struct Mat {
    data: Vec<f64>,
    ld: usize,
}

impl Mat {
    fn new(rows: usize, cols: usize) -> Self {
        Mat {
            data: vec![0.0; rows * cols],
            ld: rows,
        }
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[(i - 1) + (j - 1) * self.ld]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[(i - 1) + (j - 1) * self.ld]
    }
}

#[inline]
fn ddot(n: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

/// `pointr % m + 1` (advance a 1-based circular pointer).
#[inline]
fn next_ptr(p: usize, m: usize) -> usize {
    p % m + 1
}

// ---------------------------------------------------------------------------
// Subroutines of L-BFGS-B
// ---------------------------------------------------------------------------

/// Initializes `iwhere` and projects the initial `x` onto the feasible set.
/// Returns `(prjctd, cnstnd, boxed)`.
fn active(n: usize, l: &V1, u: &V1, nbd: &I1, x: &mut V1, iwhere: &mut I1) -> (bool, bool, bool) {
    let mut prjctd = false;
    let mut cnstnd = false;
    let mut boxed = true;
    for i in 1..=n {
        if nbd[i] > 0 {
            if nbd[i] <= 2 && x[i] <= l[i] {
                if x[i] < l[i] {
                    prjctd = true;
                    x[i] = l[i];
                }
            } else if nbd[i] >= 2 && x[i] > u[i] {
                prjctd = true;
                x[i] = u[i];
            }
        }
    }
    for i in 1..=n {
        if nbd[i] != 2 {
            boxed = false;
        }
        if nbd[i] == 0 {
            iwhere[i] = -1;
        } else {
            cnstnd = true;
            if nbd[i] == 2 && u[i] - l[i] <= 0.0 {
                iwhere[i] = 3;
            } else {
                iwhere[i] = 0;
            }
        }
    }
    (prjctd, cnstnd, boxed)
}

/// Product of the 2m x 2m middle matrix of the compact L-BFGS formula with a
/// 2col vector `v` (stored at `v[voff + 1 ..]`), written to `p[poff + 1 ..]`.
/// Returns the `dtrsl` info code.
#[allow(clippy::too_many_arguments)]
fn bmv(sy: &Mat, wt: &Mat, col: usize, v: &[f64], voff: usize, p: &mut [f64], poff: usize) -> i32 {
    if col == 0 {
        return 0;
    }
    let vv = |i: usize| v[voff + i - 1];
    // PART I: solve [ D^(1/2)      O ] [ p1 ] = [ v1 ]
    //               [ -L*D^(-1/2)  J ] [ p2 ]   [ v2 ]
    p[poff + col] = vv(col + 1);
    for i in 2..=col {
        let i2 = col + i;
        let mut sum = 0.0;
        for k in 1..i {
            sum += sy[(i, k)] * vv(k) / sy[(k, k)];
        }
        p[poff + i2 - 1] = vv(i2) + sum;
    }
    let info = dtrsl(wt, 0, col, p, poff + col, 11);
    if info != 0 {
        return info;
    }
    for i in 1..=col {
        p[poff + i - 1] = vv(i) / sy[(i, i)].sqrt();
    }
    // PART II: solve [ -D^(1/2)   D^(-1/2)*L'  ] [ p1 ] = [ p1 ]
    //                [  0         J'           ] [ p2 ]   [ p2 ]
    let info = dtrsl(wt, 0, col, p, poff + col, 1);
    if info != 0 {
        return info;
    }
    for i in 1..=col {
        p[poff + i - 1] = -p[poff + i - 1] / sy[(i, i)].sqrt();
    }
    for i in 1..=col {
        let mut sum = 0.0;
        for k in (i + 1)..=col {
            sum += sy[(k, i)] * p[poff + col + k - 1] / sy[(i, i)];
        }
        p[poff + i - 1] += sum;
    }
    0
}

/// Computes the generalized Cauchy point `xcp`. Returns the info code.
#[allow(clippy::too_many_arguments)]
fn cauchy(
    n: usize,
    x: &V1,
    l: &V1,
    u: &V1,
    nbd: &I1,
    g: &V1,
    iorder: &mut I1,
    iwhere: &mut I1,
    t: &mut V1,
    d: &mut V1,
    xcp: &mut V1,
    m: usize,
    wy: &Mat,
    ws: &Mat,
    sy: &Mat,
    wt: &Mat,
    theta: f64,
    col: usize,
    head: usize,
    wa: &mut V1,
    sbgnrm: f64,
    epsmch: f64,
) -> i32 {
    // Workspace layout (1-based): p = wa[1..2m], c = wa[2m+1..4m],
    // wbp = wa[4m+1..6m], v = wa[6m+1..8m].
    let (po, co, wbpo, vo) = (0usize, 2 * m, 4 * m, 6 * m);
    if sbgnrm <= 0.0 {
        xcp.0.copy_from_slice(&x.0);
        return 0;
    }
    let mut bnded = true;
    let mut nfree = n + 1;
    let mut nbreak = 0usize;
    let mut ibkmin = 0usize;
    let mut bkmin = 0.0;
    let col2 = 2 * col;
    let mut f1 = 0.0;
    for i in 1..=col2 {
        wa.0[po + i - 1] = 0.0;
    }
    let mut tl = 0.0;
    let mut tu = 0.0;
    for i in 1..=n {
        let neggi = -g[i];
        if iwhere[i] != 3 && iwhere[i] != -1 {
            if nbd[i] <= 2 {
                tl = x[i] - l[i];
            }
            if nbd[i] >= 2 {
                tu = u[i] - x[i];
            }
            let xlower = nbd[i] <= 2 && tl <= 0.0;
            let xupper = nbd[i] >= 2 && tu <= 0.0;
            iwhere[i] = 0;
            if xlower {
                if neggi <= 0.0 {
                    iwhere[i] = 1;
                }
            } else if xupper {
                if neggi >= 0.0 {
                    iwhere[i] = 2;
                }
            } else if neggi.abs() <= 0.0 {
                iwhere[i] = -3;
            }
        }
        let mut pointr = head;
        if iwhere[i] != 0 && iwhere[i] != -1 {
            d[i] = 0.0;
        } else {
            d[i] = neggi;
            f1 -= neggi * neggi;
            for j in 1..=col {
                wa.0[po + j - 1] += wy[(i, pointr)] * neggi;
                wa.0[po + col + j - 1] += ws[(i, pointr)] * neggi;
                pointr = next_ptr(pointr, m);
            }
            if nbd[i] <= 2 && nbd[i] != 0 && neggi < 0.0 {
                nbreak += 1;
                iorder[nbreak] = i as i64;
                t[nbreak] = tl / (-neggi);
                if nbreak == 1 || t[nbreak] < bkmin {
                    bkmin = t[nbreak];
                    ibkmin = nbreak;
                }
            } else if nbd[i] >= 2 && neggi > 0.0 {
                nbreak += 1;
                iorder[nbreak] = i as i64;
                t[nbreak] = tu / neggi;
                if nbreak == 1 || t[nbreak] < bkmin {
                    bkmin = t[nbreak];
                    ibkmin = nbreak;
                }
            } else {
                nfree -= 1;
                iorder[nfree] = i as i64;
                if neggi.abs() > 0.0 {
                    bnded = false;
                }
            }
        }
    }
    if theta != 1.0 {
        for j in 1..=col {
            wa.0[po + col + j - 1] *= theta;
        }
    }
    xcp.0.copy_from_slice(&x.0);
    if nbreak == 0 && nfree == n + 1 {
        return 0;
    }
    for j in 1..=col2 {
        wa.0[co + j - 1] = 0.0;
    }
    let mut f2 = -theta * f1;
    let f2_org = f2;
    if col > 0 {
        let (head_part, rest) = wa.0.split_at_mut(vo);
        let info = bmv(sy, wt, col, head_part, po, rest, 0);
        if info != 0 {
            return info;
        }
        f2 -= ddot(col2, &wa.0[vo..vo + col2], &wa.0[po..po + col2]);
    }
    let mut dtm = -f1 / f2;
    let mut tsum = 0.0;
    let mut skip_to_999 = false;
    if nbreak != 0 {
        let mut nleft = nbreak;
        let mut iter = 1usize;
        let mut tj = 0.0;
        loop {
            // L777
            let tj0 = tj;
            let ibp;
            if iter == 1 {
                tj = bkmin;
                ibp = iorder[ibkmin] as usize;
            } else {
                if iter == 2 && ibkmin != nbreak {
                    t[ibkmin] = t[nbreak];
                    iorder[ibkmin] = iorder[nbreak];
                }
                hpsolb(nleft, t, iorder, iter - 2);
                tj = t[nleft];
                ibp = iorder[nleft] as usize;
            }
            let dt = tj - tj0;
            if dtm < dt {
                break;
            }
            tsum += dt;
            nleft -= 1;
            iter += 1;
            let dibp = d[ibp];
            d[ibp] = 0.0;
            let zibp;
            if dibp > 0.0 {
                zibp = u[ibp] - x[ibp];
                xcp[ibp] = u[ibp];
                iwhere[ibp] = 2;
            } else {
                zibp = l[ibp] - x[ibp];
                xcp[ibp] = l[ibp];
                iwhere[ibp] = 1;
            }
            if nleft == 0 && nbreak == n {
                dtm = dt;
                skip_to_999 = true;
                break;
            }
            let dibp2 = dibp * dibp;
            f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
            f2 -= theta * dibp2;
            if col > 0 {
                for j in 1..=col2 {
                    wa.0[co + j - 1] += dt * wa.0[po + j - 1];
                }
                let mut pointr = head;
                for j in 1..=col {
                    wa.0[wbpo + j - 1] = wy[(ibp, pointr)];
                    wa.0[wbpo + col + j - 1] = theta * ws[(ibp, pointr)];
                    pointr = next_ptr(pointr, m);
                }
                let (head_part, rest) = wa.0.split_at_mut(vo);
                let info = bmv(sy, wt, col, head_part, wbpo, rest, 0);
                if info != 0 {
                    return info;
                }
                let wmc = ddot(col2, &wa.0[co..co + col2], &wa.0[vo..vo + col2]);
                let wmp = ddot(col2, &wa.0[po..po + col2], &wa.0[vo..vo + col2]);
                let wmw = ddot(col2, &wa.0[wbpo..wbpo + col2], &wa.0[vo..vo + col2]);
                for j in 1..=col2 {
                    wa.0[po + j - 1] += -dibp * wa.0[wbpo + j - 1];
                }
                f1 += dibp * wmc;
                f2 = f2 + dibp * 2.0 * wmp - dibp2 * wmw;
            }
            f2 = (epsmch * f2_org).max(f2);
            if nleft > 0 {
                dtm = -f1 / f2;
                continue;
            } else if bnded {
                dtm = 0.0;
            } else {
                dtm = -f1 / f2;
            }
            break;
        }
    }
    if !skip_to_999 {
        // L888
        if dtm <= 0.0 {
            dtm = 0.0;
        }
        tsum += dtm;
        for i in 1..=n {
            xcp[i] += tsum * d[i];
        }
    }
    // L999
    if col > 0 {
        for j in 1..=col2 {
            wa.0[co + j - 1] += dtm * wa.0[po + j - 1];
        }
    }
    0
}

/// Computes `r = -Z'B(xcp - xk) - Z'g` (reduced gradient at the Cauchy point).
#[allow(clippy::too_many_arguments)]
fn cmprlb(
    n: usize,
    m: usize,
    x: &V1,
    g: &V1,
    ws: &Mat,
    wy: &Mat,
    sy: &Mat,
    wt: &Mat,
    z: &V1,
    r: &mut V1,
    wa: &mut V1,
    index: &I1,
    theta: f64,
    col: usize,
    head: usize,
    nfree: usize,
    cnstnd: bool,
) -> i32 {
    if !cnstnd && col > 0 {
        for i in 1..=n {
            r[i] = -g[i];
        }
    } else {
        for i in 1..=nfree {
            let k = index[i] as usize;
            r[i] = -theta * (z[k] - x[k]) - g[k];
        }
        // wa(1..2m) <- M * c, where c = wa(2m+1..4m) was computed by cauchy.
        let (head_part, rest) = wa.0.split_at_mut(2 * m);
        let info = bmv(sy, wt, col, rest, 0, head_part, 0);
        if info != 0 {
            return -8;
        }
        let mut pointr = head;
        for j in 1..=col {
            let a1 = wa[j];
            let a2 = theta * wa[col + j];
            for i in 1..=nfree {
                let k = index[i] as usize;
                r[i] = r[i] + wy[(k, pointr)] * a1 + ws[(k, pointr)] * a2;
            }
            pointr = next_ptr(pointr, m);
        }
    }
    0
}

/// Forms the LEL^T factorization of the indefinite matrix K of the subspace
/// problem. Returns the info code.
#[allow(clippy::too_many_arguments)]
fn formk(
    n: usize,
    nsub: usize,
    ind: &I1,
    nenter: usize,
    ileave: usize,
    indx2: &I1,
    iupdat: usize,
    updatd: bool,
    wn: &mut Mat,
    wn1: &mut Mat,
    m: usize,
    ws: &Mat,
    wy: &Mat,
    sy: &Mat,
    theta: f64,
    col: usize,
    head: usize,
) -> i32 {
    let upcl = if updatd {
        if iupdat > m {
            // Shift old part of WN1.
            for jy in 1..m {
                let js = m + jy;
                for k in 0..(m - jy) {
                    wn1[(jy + k, jy)] = wn1[(jy + 1 + k, jy + 1)];
                }
                for k in 0..(m - jy) {
                    wn1[(js + k, js)] = wn1[(js + 1 + k, js + 1)];
                }
                for k in 0..(m - 1) {
                    wn1[(m + 1 + k, jy)] = wn1[(m + 2 + k, jy + 1)];
                }
            }
        }
        // Put new rows in blocks (1,1), (2,1) and (2,2).
        let pbegin = 1;
        let pend = nsub;
        let dbegin = nsub + 1;
        let dend = n;
        let iy = col;
        let is = m + col;
        let mut ipntr = head + col - 1;
        if ipntr > m {
            ipntr -= m;
        }
        let mut jpntr = head;
        for jy in 1..=col {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            for k in pbegin..=pend {
                let k1 = ind[k] as usize;
                temp1 += wy[(k1, ipntr)] * wy[(k1, jpntr)];
            }
            for k in dbegin..=dend {
                let k1 = ind[k] as usize;
                temp2 += ws[(k1, ipntr)] * ws[(k1, jpntr)];
                temp3 += ws[(k1, ipntr)] * wy[(k1, jpntr)];
            }
            wn1[(iy, jy)] = temp1;
            wn1[(is, js)] = temp2;
            wn1[(is, jy)] = temp3;
            jpntr = next_ptr(jpntr, m);
        }
        // Put new column in block (2,1).
        let jy = col;
        let mut jpntr = head + col - 1;
        if jpntr > m {
            jpntr -= m;
        }
        let mut ipntr = head;
        for i in 1..=col {
            let is = m + i;
            let mut temp3 = 0.0;
            for k in pbegin..=pend {
                let k1 = ind[k] as usize;
                temp3 += ws[(k1, ipntr)] * wy[(k1, jpntr)];
            }
            ipntr = next_ptr(ipntr, m);
            wn1[(is, jy)] = temp3;
        }
        col - 1
    } else {
        col
    };
    // Modify the old parts in blocks (1,1) and (2,2) due to changes in the set
    // of free variables.
    let mut ipntr = head;
    for iy in 1..=upcl {
        let is = m + iy;
        let mut jpntr = head;
        for jy in 1..=iy {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            let mut temp4 = 0.0;
            for k in 1..=nenter {
                let k1 = indx2[k] as usize;
                temp1 += wy[(k1, ipntr)] * wy[(k1, jpntr)];
                temp2 += ws[(k1, ipntr)] * ws[(k1, jpntr)];
            }
            for k in ileave..=n {
                let k1 = indx2[k] as usize;
                temp3 += wy[(k1, ipntr)] * wy[(k1, jpntr)];
                temp4 += ws[(k1, ipntr)] * ws[(k1, jpntr)];
            }
            wn1[(iy, jy)] = wn1[(iy, jy)] + temp1 - temp3;
            wn1[(is, js)] = wn1[(is, js)] - temp2 + temp4;
            jpntr = next_ptr(jpntr, m);
        }
        ipntr = next_ptr(ipntr, m);
    }
    // Modify the old parts in block (2,1).
    let mut ipntr = head;
    for is in (m + 1)..=(m + upcl) {
        let mut jpntr = head;
        for jy in 1..=upcl {
            let mut temp1 = 0.0;
            let mut temp3 = 0.0;
            for k in 1..=nenter {
                let k1 = indx2[k] as usize;
                temp1 += ws[(k1, ipntr)] * wy[(k1, jpntr)];
            }
            for k in ileave..=n {
                let k1 = indx2[k] as usize;
                temp3 += ws[(k1, ipntr)] * wy[(k1, jpntr)];
            }
            if is <= jy + m {
                wn1[(is, jy)] = wn1[(is, jy)] + temp1 - temp3;
            } else {
                wn1[(is, jy)] = wn1[(is, jy)] - temp1 + temp3;
            }
            jpntr = next_ptr(jpntr, m);
        }
        ipntr = next_ptr(ipntr, m);
    }
    // Form the upper triangle of WN = [D+Y'ZZ'Y/theta   -L_a'+R_z' ]
    //                                 [-L_a +R_z        S'AA'S*theta]
    for iy in 1..=col {
        let is = col + iy;
        let is1 = m + iy;
        for jy in 1..=iy {
            let js = col + jy;
            let js1 = m + jy;
            wn[(jy, iy)] = wn1[(iy, jy)] / theta;
            wn[(js, is)] = wn1[(is1, js1)] * theta;
        }
        for jy in 1..iy {
            wn[(jy, is)] = -wn1[(is1, jy)];
        }
        for jy in iy..=col {
            wn[(jy, is)] = wn1[(is1, jy)];
        }
        wn[(iy, iy)] += sy[(iy, iy)];
    }
    // Form the upper triangle of WN = [LL'            L^-1(-L_a'+R_z')]
    //                                 [(-L_a +R_z)L'^-1   S'AA'S*theta ]
    // First Cholesky factor (1,1) block of wn to get LL' with L' stored in the
    // upper triangle of wn.
    if dpofa(wn, 0, 0, col) != 0 {
        return -1;
    }
    // Then form L^-1(-L_a'+R_z') in the (1,2) block.
    let col2 = 2 * col;
    let mut colbuf = vec![0.0; col];
    for js in (col + 1)..=col2 {
        for i in 1..=col {
            colbuf[i - 1] = wn[(i, js)];
        }
        let info = dtrsl(wn, 0, col, &mut colbuf, 0, 11);
        if info != 0 {
            return info;
        }
        for i in 1..=col {
            wn[(i, js)] = colbuf[i - 1];
        }
    }
    // Form S'AA'S*theta + (L^-1(-L_a'+R_z'))'L^-1(-L_a'+R_z') in the upper
    // triangle of (2,2) block of wn.
    for is in (col + 1)..=col2 {
        for js in is..=col2 {
            let mut s = 0.0;
            for k in 1..=col {
                s += wn[(k, is)] * wn[(k, js)];
            }
            wn[(is, js)] += s;
        }
    }
    // Cholesky factorization of (2,2) block of wn.
    if dpofa(wn, col, col, col) != 0 {
        return -2;
    }
    0
}

/// Forms the upper half of `T = theta*SS + L*D^(-1)*L'` and its Cholesky
/// factor (stored in the upper triangle of `wt`). Returns the info code.
fn formt(m: usize, wt: &mut Mat, sy: &Mat, ss: &Mat, col: usize, theta: f64) -> i32 {
    let _ = m;
    for j in 1..=col {
        wt[(1, j)] = theta * ss[(1, j)];
    }
    for i in 2..=col {
        for j in i..=col {
            let k1 = i.min(j) - 1;
            let mut ddum = 0.0;
            for k in 1..=k1 {
                ddum += sy[(i, k)] * sy[(j, k)] / sy[(k, k)];
            }
            wt[(i, j)] = ddum + theta * ss[(i, j)];
        }
    }
    if dpofa(wt, 0, 0, col) != 0 {
        return -3;
    }
    0
}

/// Counts the entering and leaving variables and partitions the variables
/// into free and active sets. Returns `wrk` (whether `formk` must be called).
#[allow(clippy::too_many_arguments)]
fn freev(
    n: usize,
    nfree: &mut usize,
    index: &mut I1,
    nenter: &mut usize,
    ileave: &mut usize,
    indx2: &mut I1,
    iwhere: &I1,
    updatd: bool,
    cnstnd: bool,
    iter: usize,
) -> bool {
    *nenter = 0;
    *ileave = n + 1;
    if iter > 0 && cnstnd {
        // Count the entering and leaving variables.
        for i in 1..=*nfree {
            let k = index[i] as usize;
            if iwhere[k] > 0 {
                *ileave -= 1;
                indx2[*ileave] = k as i64;
            }
        }
        for i in (*nfree + 1)..=n {
            let k = index[i] as usize;
            if iwhere[k] <= 0 {
                *nenter += 1;
                indx2[*nenter] = k as i64;
            }
        }
    }
    let wrk = *ileave < n + 1 || *nenter > 0 || updatd;
    // Find the index set of free and active variables at the GCP.
    *nfree = 0;
    let mut iact = n + 1;
    for i in 1..=n {
        if iwhere[i] <= 0 {
            *nfree += 1;
            index[*nfree] = i as i64;
        } else {
            iact -= 1;
            index[iact] = i as i64;
        }
    }
    wrk
}

/// Heap sort helper: moves the least breakpoint to `t[n]` (see the reference).
fn hpsolb(n: usize, t: &mut V1, iorder: &mut I1, iheap: usize) {
    if iheap == 0 {
        // Rearrange the elements t(1) to t(n) to form a heap.
        for k in 2..=n {
            let ddum = t[k];
            let indxin = iorder[k];
            let mut i = k;
            while i > 1 {
                let j = i / 2;
                if ddum < t[j] {
                    t[i] = t[j];
                    iorder[i] = iorder[j];
                    i = j;
                } else {
                    break;
                }
            }
            t[i] = ddum;
            iorder[i] = indxin;
        }
    }
    // Assign to 'out' the value of t(1), the least member of the heap, and
    // rearrange the remaining members to form a heap as elements 1 to n-1.
    if n > 1 {
        let mut i = 1;
        let out = t[1];
        let indxou = iorder[1];
        let ddum = t[n];
        let indxin = iorder[n];
        loop {
            let mut j = i + i;
            if j < n {
                if t[j + 1] < t[j] {
                    j += 1;
                }
                if t[j] < ddum {
                    t[i] = t[j];
                    iorder[i] = iorder[j];
                    i = j;
                    continue;
                }
            }
            break;
        }
        t[i] = ddum;
        iorder[i] = indxin;
        // Put the least member in t(n).
        t[n] = out;
        iorder[n] = indxou;
    }
}

/// Updates the limited-memory matrices WS, WY, SY and SS.
#[allow(clippy::too_many_arguments)]
fn matupd(
    n: usize,
    m: usize,
    ws: &mut Mat,
    wy: &mut Mat,
    sy: &mut Mat,
    ss: &mut Mat,
    d: &V1,
    r: &V1,
    itail: &mut usize,
    iupdat: usize,
    col: &mut usize,
    head: &mut usize,
    theta: &mut f64,
    rr: f64,
    dr: f64,
    stp: f64,
    dtd: f64,
) {
    // Set pointers for matrices WS and WY.
    if iupdat <= m {
        *col = iupdat;
        *itail = (*head + iupdat - 2) % m + 1;
    } else {
        *itail = next_ptr(*itail, m);
        *head = next_ptr(*head, m);
    }
    // Update matrices WS and WY.
    for i in 1..=n {
        ws[(i, *itail)] = d[i];
        wy[(i, *itail)] = r[i];
    }
    // Set theta = yy/ys.
    *theta = rr / dr;
    // Form the middle matrix in B: update the upper triangle of SS and the
    // lower triangle of SY.
    if iupdat > m {
        // Move old information.
        for j in 1..*col {
            for k in 0..j {
                ss[(1 + k, j)] = ss[(2 + k, j + 1)];
            }
            for k in 0..(*col - j) {
                sy[(j + k, j)] = sy[(j + 1 + k, j + 1)];
            }
        }
    }
    // Add new information: the last row of SY and the last column of SS.
    let mut pointr = *head;
    for j in 1..*col {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for i in 1..=n {
            s1 += d[i] * wy[(i, pointr)];
            s2 += ws[(i, pointr)] * d[i];
        }
        sy[(*col, j)] = s1;
        ss[(j, *col)] = s2;
        pointr = next_ptr(pointr, m);
    }
    ss[(*col, *col)] = if stp == 1.0 { dtd } else { stp * stp * dtd };
    sy[(*col, *col)] = dr;
}

/// Infinity norm of the projected gradient.
fn projgr(n: usize, l: &V1, u: &V1, nbd: &I1, x: &V1, g: &V1) -> f64 {
    let mut sbgnrm: f64 = 0.0;
    for i in 1..=n {
        let mut gi = g[i];
        if nbd[i] != 0 {
            if gi < 0.0 {
                if nbd[i] >= 2 {
                    gi = (x[i] - u[i]).max(gi);
                }
            } else if nbd[i] <= 2 {
                gi = (x[i] - l[i]).min(gi);
            }
        }
        sbgnrm = sbgnrm.max(gi.abs());
    }
    sbgnrm
}

/// Subspace minimization (version 3.0, with the projected search of Morales
/// and Nocedal). `x` holds the Cauchy point on entry and the subspace
/// minimizer on exit; `d` holds the reduced gradient on entry. Returns the
/// info code.
#[allow(clippy::too_many_arguments)]
fn subsm(
    n: usize,
    m: usize,
    nsub: usize,
    ind: &I1,
    l: &V1,
    u: &V1,
    nbd: &I1,
    x: &mut V1,
    d: &mut V1,
    xp: &mut V1,
    ws: &Mat,
    wy: &Mat,
    theta: f64,
    xx: &V1,
    gg: &V1,
    col: usize,
    head: usize,
    wv: &mut V1,
    wn: &Mat,
) -> i32 {
    if nsub == 0 {
        return 0;
    }
    // Compute wv = W'Zd.
    let mut pointr = head;
    for i in 1..=col {
        let mut temp1 = 0.0;
        let mut temp2 = 0.0;
        for j in 1..=nsub {
            let k = ind[j] as usize;
            temp1 += wy[(k, pointr)] * d[j];
            temp2 += ws[(k, pointr)] * d[j];
        }
        wv[i] = temp1;
        wv[col + i] = theta * temp2;
        pointr = next_ptr(pointr, m);
    }
    // Compute wv := K^(-1) wv.
    let col2 = 2 * col;
    let info = dtrsl(wn, 0, col2, &mut wv.0, 0, 11);
    if info != 0 {
        return info;
    }
    for i in 1..=col {
        wv[i] = -wv[i];
    }
    let info = dtrsl(wn, 0, col2, &mut wv.0, 0, 1);
    if info != 0 {
        return info;
    }
    // Compute d = (1/theta)d + (1/theta**2)Z'W wv.
    let mut pointr = head;
    for jy in 1..=col {
        let js = col + jy;
        for i in 1..=nsub {
            let k = ind[i] as usize;
            d[i] = d[i] + wy[(k, pointr)] * wv[jy] / theta + ws[(k, pointr)] * wv[js];
        }
        pointr = next_ptr(pointr, m);
    }
    let inv_theta = 1.0 / theta;
    for i in 1..=nsub {
        d[i] *= inv_theta;
    }
    // Let us try the projection: d is the Newton direction.
    let mut hit_bound = false;
    xp.0.copy_from_slice(&x.0);
    for i in 1..=nsub {
        let k = ind[i] as usize;
        let dk = d[i];
        let mut xk = x[k];
        if nbd[k] != 0 {
            if nbd[k] == 1 {
                x[k] = l[k].max(xk + dk);
                if x[k] == l[k] {
                    hit_bound = true;
                }
            } else if nbd[k] == 2 {
                xk = l[k].max(xk + dk);
                x[k] = u[k].min(xk);
                if x[k] == l[k] || x[k] == u[k] {
                    hit_bound = true;
                }
            } else if nbd[k] == 3 {
                x[k] = u[k].min(xk + dk);
                if x[k] == u[k] {
                    hit_bound = true;
                }
            }
        } else {
            x[k] = xk + dk;
        }
    }
    if !hit_bound {
        return 0;
    }
    // Check sign of the directional derivative.
    let mut dd_p = 0.0;
    for i in 1..=n {
        dd_p += (x[i] - xx[i]) * gg[i];
    }
    if dd_p > 0.0 {
        x.0.copy_from_slice(&xp.0);
    } else {
        return 0;
    }
    // Backtracking (version 2.3 step) within the feasible region.
    let mut alpha = 1.0;
    let mut temp1 = alpha;
    let mut ibd = 0usize;
    for i in 1..=nsub {
        let k = ind[i] as usize;
        let dk = d[i];
        if nbd[k] != 0 {
            if dk < 0.0 && nbd[k] <= 2 {
                let temp2 = l[k] - x[k];
                if temp2 >= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha < temp2 {
                    temp1 = temp2 / dk;
                }
            } else if dk > 0.0 && nbd[k] >= 2 {
                let temp2 = u[k] - x[k];
                if temp2 <= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha > temp2 {
                    temp1 = temp2 / dk;
                }
            }
            if temp1 < alpha {
                alpha = temp1;
                ibd = i;
            }
        }
    }
    if alpha < 1.0 {
        let dk = d[ibd];
        let k = ind[ibd] as usize;
        if dk > 0.0 {
            x[k] = u[k];
            d[ibd] = 0.0;
        } else if dk < 0.0 {
            x[k] = l[k];
            d[ibd] = 0.0;
        }
    }
    for i in 1..=nsub {
        let k = ind[i] as usize;
        x[k] += alpha * d[i];
    }
    0
}

/// LINPACK `dpofa`: Cholesky factorization `A = R'R` of the `n x n` block of
/// `a` whose top-left element is `a(r0 + 1, c0 + 1)`; `R` overwrites the upper
/// triangle. Returns 0 on success or the order of the failing leading minor.
fn dpofa(a: &mut Mat, r0: usize, c0: usize, n: usize) -> i32 {
    let at = |i: usize, j: usize| (r0 + i, c0 + j);
    for j in 1..=n {
        let mut s = 0.0;
        for k in 1..j {
            let mut dot = 0.0;
            for i in 1..k {
                dot += a[at(i, k)] * a[at(i, j)];
            }
            let mut t = a[at(k, j)] - dot;
            t /= a[at(k, k)];
            a[at(k, j)] = t;
            s += t * t;
        }
        let s = a[at(j, j)] - s;
        if s <= 0.0 {
            return j as i32;
        }
        a[at(j, j)] = s.sqrt();
    }
    0
}

/// LINPACK `dtrsl`: solves `T x = b` or `T' x = b` for the `n x n` leading
/// triangular block of `t` (whose top-left element is `t(r0 + 1, r0 + 1)`),
/// where `b(i)` is stored at `b[boff + i - 1]`. `job`: 00 lower `T x = b`,
/// 01 upper `T x = b`, 10 lower `T' x = b`, 11 upper `T' x = b`.
/// Returns 0 or the index of a zero diagonal element.
fn dtrsl(t: &Mat, r0: usize, n: usize, b: &mut [f64], boff: usize, job: i32) -> i32 {
    let tt = |i: usize, j: usize| t[(r0 + i, r0 + j)];
    for info in 1..=n {
        if tt(info, info) == 0.0 {
            return info as i32;
        }
    }
    let bi = |i: usize| boff + i - 1;
    let mut case = 1;
    if job % 10 != 0 {
        case = 2;
    }
    if job % 100 / 10 != 0 {
        case += 2;
    }
    match case {
        1 => {
            // T lower, T x = b.
            b[bi(1)] /= tt(1, 1);
            for j in 2..=n {
                let temp = -b[bi(j - 1)];
                for k in 0..(n - j + 1) {
                    b[bi(j + k)] += temp * tt(j + k, j - 1);
                }
                b[bi(j)] /= tt(j, j);
            }
        }
        2 => {
            // T upper, T x = b.
            b[bi(n)] /= tt(n, n);
            for jj in 2..=n {
                let j = n - jj + 1;
                let temp = -b[bi(j + 1)];
                for k in 1..=j {
                    b[bi(k)] += temp * tt(k, j + 1);
                }
                b[bi(j)] /= tt(j, j);
            }
        }
        3 => {
            // T lower, T' x = b.
            b[bi(n)] /= tt(n, n);
            for jj in 2..=n {
                let j = n - jj + 1;
                let mut dot = 0.0;
                for k in 1..jj {
                    dot += tt(j + k, j) * b[bi(j + k)];
                }
                b[bi(j)] -= dot;
                b[bi(j)] /= tt(j, j);
            }
        }
        _ => {
            // T upper, T' x = b.
            b[bi(1)] /= tt(1, 1);
            for j in 2..=n {
                let mut dot = 0.0;
                for k in 1..j {
                    dot += tt(k, j) * b[bi(k)];
                }
                b[bi(j)] -= dot;
                b[bi(j)] /= tt(j, j);
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Line search (Moré-Thuente dcsrch / dcstep, MINPACK-2)
// ---------------------------------------------------------------------------

const FTOL: f64 = 1e-3;
const GTOL: f64 = 0.9;
const XTOL: f64 = 0.1;
const STPMIN: f64 = 0.0;

/// Task codes of `dcsrch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LsTask {
    Start,
    Fg,
    Convergence,
    WarningRound,
    WarningXtol,
    WarningStpmax,
    WarningStpmin,
    Error,
}

impl LsTask {
    fn is_warning(self) -> bool {
        matches!(
            self,
            LsTask::WarningRound
                | LsTask::WarningXtol
                | LsTask::WarningStpmax
                | LsTask::WarningStpmin
        )
    }

    fn is_converged(self) -> bool {
        self == LsTask::Convergence
    }
}

/// State of the Moré-Thuente line search.
#[derive(Clone, Debug)]
struct LineSearch {
    task: LsTask,
    brackt: bool,
    stage: i32,
    ginit: f64,
    gtest: f64,
    gx: f64,
    gy: f64,
    finit: f64,
    fx: f64,
    fy: f64,
    stx: f64,
    sty: f64,
    stmin: f64,
    stmax: f64,
    width: f64,
    width1: f64,
}

impl LineSearch {
    fn new() -> Self {
        LineSearch {
            task: LsTask::Start,
            brackt: false,
            stage: 1,
            ginit: 0.0,
            gtest: 0.0,
            gx: 0.0,
            gy: 0.0,
            finit: 0.0,
            fx: 0.0,
            fy: 0.0,
            stx: 0.0,
            sty: 0.0,
            stmin: 0.0,
            stmax: 0.0,
            width: 0.0,
            width1: 0.0,
        }
    }

    /// One step of `dcsrch`: given `f` and `g` (directional derivative) at
    /// `stp`, updates `stp` and `self.task`.
    #[allow(clippy::too_many_arguments)]
    fn dcsrch(
        &mut self,
        f: f64,
        g: f64,
        stp: &mut f64,
        ftol: f64,
        gtol: f64,
        xtol: f64,
        stpmin: f64,
        stpmax: f64,
    ) {
        if self.task == LsTask::Start {
            let mut error = false;
            if *stp < stpmin || *stp > stpmax || g >= 0.0 || ftol < 0.0 || gtol < 0.0 || xtol < 0.0
            {
                error = true;
            }
            if stpmin < 0.0 || stpmax < stpmin {
                error = true;
            }
            if error {
                // As in the reference driver, an error code is not treated as
                // convergence or warning, so the caller keeps iterating with
                // the (possibly uninitialized) search state.
                self.task = LsTask::Error;
                return;
            }
            self.brackt = false;
            self.stage = 1;
            self.finit = f;
            self.ginit = g;
            self.gtest = ftol * self.ginit;
            self.width = stpmax - stpmin;
            self.width1 = self.width / 0.5;
            self.stx = 0.0;
            self.fx = self.finit;
            self.gx = self.ginit;
            self.sty = 0.0;
            self.fy = self.finit;
            self.gy = self.ginit;
            self.stmin = 0.0;
            self.stmax = *stp + *stp * 4.0;
            self.task = LsTask::Fg;
            return;
        }
        let ftest = self.finit + *stp * self.gtest;
        if self.stage == 1 && f <= ftest && g >= 0.0 {
            self.stage = 2;
        }
        // Test for warnings.
        if self.brackt && (*stp <= self.stmin || *stp >= self.stmax) {
            self.task = LsTask::WarningRound;
        }
        if self.brackt && self.stmax - self.stmin <= xtol * self.stmax {
            self.task = LsTask::WarningXtol;
        }
        if *stp == stpmax && f <= ftest && g <= self.gtest {
            self.task = LsTask::WarningStpmax;
        }
        if *stp == stpmin && (f > ftest || g >= self.gtest) {
            self.task = LsTask::WarningStpmin;
        }
        // Test for convergence.
        if f <= ftest && g.abs() <= gtol * (-self.ginit) {
            self.task = LsTask::Convergence;
        }
        if self.task.is_warning() || self.task.is_converged() {
            return;
        }
        // A modified function is used to predict the step during the first
        // stage if a lower function value has been obtained but the decrease
        // is not sufficient.
        if self.stage == 1 && f <= self.fx && f > ftest {
            let fm = f - *stp * self.gtest;
            let mut fxm = self.fx - self.stx * self.gtest;
            let mut fym = self.fy - self.sty * self.gtest;
            let gm = g - self.gtest;
            let mut gxm = self.gx - self.gtest;
            let mut gym = self.gy - self.gtest;
            dcstep(
                &mut self.stx,
                &mut fxm,
                &mut gxm,
                &mut self.sty,
                &mut fym,
                &mut gym,
                stp,
                fm,
                gm,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
            self.fx = fxm + self.stx * self.gtest;
            self.fy = fym + self.sty * self.gtest;
            self.gx = gxm + self.gtest;
            self.gy = gym + self.gtest;
        } else {
            dcstep(
                &mut self.stx,
                &mut self.fx,
                &mut self.gx,
                &mut self.sty,
                &mut self.fy,
                &mut self.gy,
                stp,
                f,
                g,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
        }
        // Decide if a bisection step is needed.
        if self.brackt {
            if (self.sty - self.stx).abs() >= self.width1 * 0.66 {
                *stp = self.stx + (self.sty - self.stx) * 0.5;
            }
            self.width1 = self.width;
            self.width = (self.sty - self.stx).abs();
        }
        // Set the minimum and maximum steps allowed for stp.
        if self.brackt {
            self.stmin = self.stx.min(self.sty);
            self.stmax = self.stx.max(self.sty);
        } else {
            self.stmin = *stp + (*stp - self.stx) * 1.1;
            self.stmax = *stp + (*stp - self.stx) * 4.0;
        }
        // Force the step to be within the bounds stpmax and stpmin.
        *stp = stp.max(stpmin);
        *stp = stp.min(stpmax);
        // If further progress is not possible, let stp be the best point
        // obtained during the search.
        if (self.brackt && (*stp <= self.stmin || *stp >= self.stmax))
            || (self.brackt && self.stmax - self.stmin <= xtol * self.stmax)
        {
            *stp = self.stx;
        }
        self.task = LsTask::Fg;
    }
}

/// MINPACK-2 `dcstep`: safeguarded step for the line search.
#[allow(clippy::too_many_arguments)]
fn dcstep(
    stx: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    sty: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    stp: &mut f64,
    fp: f64,
    dp: f64,
    brackt: &mut bool,
    stpmin: f64,
    stpmax: f64,
) {
    let sgnd = dp * (*dx / dx.abs());
    let stpf;
    if fp > *fx {
        // First case: a higher function value. The minimum is bracketed.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let ts = theta / s;
        let mut gamma = s * (ts * ts - *dx / s * (dp / s)).sqrt();
        if *stp < *stx {
            gamma = -gamma;
        }
        let p = gamma - *dx + theta;
        let q = gamma - *dx + gamma + dp;
        let r = p / q;
        let stpc = *stx + r * (*stp - *stx);
        let stpq = *stx + *dx / ((*fx - fp) / (*stp - *stx) + *dx) / 2.0 * (*stp - *stx);
        if (stpc - *stx).abs() < (stpq - *stx).abs() {
            stpf = stpc;
        } else {
            stpf = stpc + (stpq - stpc) / 2.0;
        }
        *brackt = true;
    } else if sgnd < 0.0 {
        // Second case: lower function value and derivatives of opposite sign.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let ts = theta / s;
        let mut gamma = s * (ts * ts - *dx / s * (dp / s)).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = gamma - dp + theta;
        let q = gamma - dp + gamma + *dx;
        let r = p / q;
        let stpc = *stp + r * (*stx - *stp);
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if (stpc - *stp).abs() > (stpq - *stp).abs() {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        *brackt = true;
    } else if dp.abs() < dx.abs() {
        // Third case: lower function value, derivatives of the same sign, and
        // the magnitude of the derivative decreases.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let ts = theta / s;
        let mut gamma = s * (0.0f64.max(ts * ts - *dx / s * (dp / s))).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = gamma - dp + theta;
        let q = gamma + (*dx - dp) + gamma;
        let r = p / q;
        let stpc = if r < 0.0 && gamma != 0.0 {
            *stp + r * (*stx - *stp)
        } else if *stp > *stx {
            stpmax
        } else {
            stpmin
        };
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if *brackt {
            let mut sf = if (stpc - *stp).abs() < (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            if *stp > *stx {
                sf = (*stp + (*sty - *stp) * 0.66).min(sf);
            } else {
                sf = (*stp + (*sty - *stp) * 0.66).max(sf);
            }
            stpf = sf;
        } else {
            let mut sf = if (stpc - *stp).abs() > (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            sf = stpmax.min(sf);
            sf = stpmin.max(sf);
            stpf = sf;
        }
    } else {
        // Fourth case: lower function value, derivatives of the same sign, and
        // the magnitude of the derivative does not decrease.
        if *brackt {
            let theta = (fp - *fy) * 3.0 / (*sty - *stp) + *dy + dp;
            let s = theta.abs().max(dy.abs()).max(dp.abs());
            let ts = theta / s;
            let mut gamma = s * (ts * ts - *dy / s * (dp / s)).sqrt();
            if *stp > *sty {
                gamma = -gamma;
            }
            let p = gamma - dp + theta;
            let q = gamma - dp + gamma + *dy;
            let r = p / q;
            stpf = *stp + r * (*sty - *stp);
        } else if *stp > *stx {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }
    // Update the interval which contains a minimizer.
    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < 0.0 {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    *stp = stpf;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The extended Rosenbrock problem of the L-BFGS-B drivers (driver1):
    /// n = 25, bounds alternating between [1, 100] and [-100, 100], x0 = 3.
    fn rosenbrock(x: &[f64]) -> (f64, Vec<f64>) {
        let n = x.len();
        let mut f = 0.25 * (x[0] - 1.0).powi(2);
        for i in 1..n {
            f += (x[i] - x[i - 1].powi(2)).powi(2);
        }
        f *= 4.0;
        let mut g = vec![0.0; n];
        let mut t1 = x[1] - x[0].powi(2);
        g[0] = 2.0 * (x[0] - 1.0) - 16.0 * x[0] * t1;
        for i in 1..n - 1 {
            let t2 = t1;
            t1 = x[i + 1] - x[i].powi(2);
            g[i] = 8.0 * t2 - 16.0 * x[i] * t1;
        }
        g[n - 1] = 8.0 * t1;
        (f, g)
    }

    fn check_against_scipy(
        res: &LbfgsbResult,
        nit: usize,
        nfev: usize,
        f: f64,
        status: LbfgsbStatus,
        x: &[f64],
    ) {
        assert_eq!(res.status, status);
        assert_eq!(res.iterations, nit, "iterations");
        assert_eq!(res.evaluations, nfev, "evaluations");
        assert!(
            (res.f - f).abs() <= 1e-12 * f.abs() + 1e-15,
            "f = {} vs {}",
            res.f,
            f
        );
        for (a, e) in res.x.iter().zip(x) {
            assert!((a - e).abs() <= 1e-12 * e.abs().max(1.0), "x = {a} vs {e}");
        }
    }

    // Reference values: scipy 1.18 `minimize(fun, x0, jac=True, method="L-BFGS-B", ...)`.

    #[test]
    fn rosenbrock_bounded_matches_scipy() {
        // The problem of the L-BFGS-B driver1 (n = 25, maxcor = 5).
        let n = 25;
        let lower: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -100.0 })
            .collect();
        let upper = vec![100.0; n];
        let opts = LbfgsbOptions {
            maxcor: 5,
            ..Default::default()
        };
        let res = minimize(
            |x: &[f64]| Ok::<_, ()>(rosenbrock(x)),
            &vec![3.0; n],
            &lower,
            &upper,
            &opts,
        )
        .unwrap();
        check_against_scipy(
            &res,
            23,
            28,
            1.0834900835598926e-09,
            LbfgsbStatus::ConvergedRelativeReduction,
            &[1.000002325877184, 1.0000030446388726, 1.0000045143926763],
        );
    }

    #[test]
    fn rosenbrock_unbounded_matches_scipy() {
        let inf = f64::INFINITY;
        let res = minimize(
            |x: &[f64]| Ok::<_, ()>(rosenbrock(x)),
            &[3.0; 10],
            &[-inf; 10],
            &[inf; 10],
            &LbfgsbOptions::default(),
        )
        .unwrap();
        check_against_scipy(
            &res,
            16,
            18,
            5.464722490528811e-06,
            LbfgsbStatus::ConvergedRelativeReduction,
            &[1.0022456539375386, 1.0047780553421248, 1.009719304373149],
        );
    }

    #[test]
    fn rosenbrock_half_bounded_matches_scipy() {
        let inf = f64::INFINITY;
        let lower: Vec<f64> = (0..10)
            .map(|i| if i % 3 == 0 { -inf } else { -0.5 })
            .collect();
        let upper: Vec<f64> = (0..10)
            .map(|i| if i % 3 == 0 { 0.9 } else { inf })
            .collect();
        let res = minimize(
            |x: &[f64]| Ok::<_, ()>(rosenbrock(x)),
            &[-1.2; 10],
            &lower,
            &upper,
            &LbfgsbOptions::default(),
        )
        .unwrap();
        check_against_scipy(
            &res,
            26,
            27,
            0.010000000001572666,
            LbfgsbStatus::ConvergedProjectedGradient,
            &[0.9, 0.8099999855732031, 0.6561000396376608],
        );
    }

    #[test]
    fn quadratic_matches_scipy() {
        let a = [3.0, 1.0, 0.5];
        let c = [1.0, -2.0, 0.25];
        let f = |x: &[f64]| -> Result<(f64, Vec<f64>), ()> {
            let v: f64 = (0..3).map(|i| 0.5 * a[i] * (x[i] - c[i]).powi(2)).sum();
            let g = (0..3).map(|i| a[i] * (x[i] - c[i])).collect();
            Ok((v, g))
        };
        let inf = f64::INFINITY;
        let res = minimize(
            f,
            &[0.0; 3],
            &[-inf; 3],
            &[inf; 3],
            &LbfgsbOptions::default(),
        )
        .unwrap();
        check_against_scipy(
            &res,
            8,
            9,
            1.6353902705993632e-11,
            LbfgsbStatus::ConvergedProjectedGradient,
            &[0.9999986228581869, -2.000001091245403, 0.2500071871315783],
        );
    }

    #[test]
    fn active_bounds_are_respected() {
        // Minimum of (x - 2)^2 + (y + 1)^2 subject to x <= 1, y >= 0 is (1, 0).
        let f = |x: &[f64]| -> Result<(f64, Vec<f64>), ()> {
            Ok((
                (x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2),
                vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] + 1.0)],
            ))
        };
        let inf = f64::INFINITY;
        let res = minimize(
            f,
            &[-5.0, 5.0],
            &[-inf, 0.0],
            &[1.0, inf],
            &LbfgsbOptions::default(),
        )
        .unwrap();
        assert!(res.status.success());
        assert_eq!(res.x, vec![1.0, 0.0]);
    }

    #[test]
    fn objective_errors_abort() {
        let f = |_: &[f64]| -> Result<(f64, Vec<f64>), &'static str> { Err("boom") };
        let res = minimize(
            f,
            &[0.0],
            &[f64::NEG_INFINITY],
            &[f64::INFINITY],
            &LbfgsbOptions::default(),
        );
        assert_eq!(res, Err(LbfgsbError::Objective("boom")));
        let g = |_: &[f64]| -> Result<(f64, Vec<f64>), ()> { Ok((0.0, vec![0.0])) };
        assert!(matches!(
            minimize(g, &[0.0], &[1.0], &[0.0], &LbfgsbOptions::default()),
            Err(LbfgsbError::InvalidInput(_))
        ));
    }

    #[test]
    fn iteration_limit_is_reported() {
        let opts = LbfgsbOptions {
            maxiter: 3,
            ..Default::default()
        };
        let n = 10;
        let inf = f64::INFINITY;
        let res = minimize(
            |x: &[f64]| Ok::<_, ()>(rosenbrock(x)),
            &vec![3.0; n],
            &vec![-inf; n],
            &vec![inf; n],
            &opts,
        )
        .unwrap();
        assert_eq!(res.status, LbfgsbStatus::MaxIterations);
        assert_eq!(res.iterations, 3);
    }
}
