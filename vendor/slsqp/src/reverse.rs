//! A driver for Kraft's SLSQP with the semantics of SciPy's
//! `scipy.optimize.minimize(method="SLSQP")` (added for botorch-rs).
//!
//! The algorithm is the reverse-communication routine `slsqp` of this crate
//! (Kraft's SQP method with an L1 merit-function line search, BFGS updates of
//! an LDLᵀ factor and Lawson–Hanson least-squares subproblems). Compared with
//! the crate's [`crate::minimize`] (NLopt's driver), [`minimize_kraft`]
//!
//! * supports equality constraints (`c(x) = 0`) next to inequality
//!   constraints (`c(x) ≥ 0`, Kraft's and SciPy's sign convention);
//! * uses Kraft's own convergence test with the accuracy `acc` (SciPy's
//!   `ftol`) instead of NLopt's stopping rules, so that convergence does not
//!   depend on constraints being satisfied exactly;
//! * counts `maxiter` major iterations (SciPy's `maxiter`) and reports
//!   Kraft's exit modes with SciPy's messages;
//! * returns the final iterate, like SciPy.

use crate::slsqp::{length_work, slsqp, slsqpb_state};

/// The objective, the constraints and (optionally) their derivatives at a
/// point, as returned by the evaluation callback of [`minimize_kraft`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evaluation {
    /// Objective value `f(x)`.
    pub f: f64,
    /// Constraint values: the `meq` equality constraints (`c_j(x) = 0`)
    /// followed by the inequality constraints (`c_j(x) ≥ 0`).
    pub c: Vec<f64>,
    /// Objective gradient (length `n`); may be empty when no derivatives were
    /// requested.
    pub g: Vec<f64>,
    /// Constraint Jacobian, row-major `m × n` (row `j` is `∇c_j`); may be
    /// empty when no derivatives were requested.
    pub a: Vec<f64>,
}

/// Settings of [`minimize_kraft`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KraftOptions {
    /// Maximum number of major (SQP) iterations (SciPy's `maxiter`, default
    /// 100).
    pub maxiter: usize,
    /// Requested accuracy (SciPy's `ftol`, default `1e-6`): the iteration stops
    /// when the objective changes by less than `acc` (or the step is shorter
    /// than `acc`) while the constraint violation is below `acc`.
    pub acc: f64,
}

impl Default for KraftOptions {
    fn default() -> Self {
        Self {
            maxiter: 100,
            acc: 1e-6,
        }
    }
}

/// Result of [`minimize_kraft`].
#[derive(Clone, Debug, PartialEq)]
pub struct KraftResult {
    /// The final iterate.
    pub x: Vec<f64>,
    /// The objective value at the last evaluated point.
    pub f: f64,
    /// Kraft's exit mode (SciPy's `status`): 0 for success, see
    /// [`exit_message`].
    pub mode: i32,
    /// Number of major iterations.
    pub iterations: usize,
    /// Number of objective evaluations.
    pub evaluations: usize,
    /// Number of gradient evaluations.
    pub gradient_evaluations: usize,
}

impl KraftResult {
    /// Whether the optimization converged (mode 0, SciPy's `success`).
    pub fn success(&self) -> bool {
        self.mode == 0
    }

    /// SciPy's message for the exit mode.
    pub fn message(&self) -> &'static str {
        exit_message(self.mode)
    }
}

/// SciPy's description of an SLSQP exit mode.
pub fn exit_message(mode: i32) -> &'static str {
    match mode {
        -1 => "Gradient evaluation required (g & a)",
        0 => "Optimization terminated successfully",
        1 => "Function evaluation required (f & c)",
        2 => "More equality constraints than independent variables",
        3 => "More than 3*n iterations in LSQ subproblem",
        4 => "Inequality constraints incompatible",
        5 => "Singular matrix E in LSQ subproblem",
        6 => "Singular matrix C in LSQ subproblem",
        7 => "Rank-deficient equality constraint subproblem HFTI",
        8 => "Positive directional derivative for linesearch",
        9 => "Iteration limit reached",
        _ => "Unknown SLSQP exit mode",
    }
}

fn new_state() -> slsqpb_state {
    slsqpb_state {
        t: 0.0,
        f0: 0.0,
        h1: 0.0,
        h2: 0.0,
        h3: 0.0,
        h4: 0.0,
        n1: 0,
        n2: 0,
        n3: 0,
        t0: 0.0,
        gs: 0.0,
        tol: 0.0,
        line: 0,
        alpha: 0.0,
        iexact: 0,
        incons: 0,
        ireset: 0,
        itermx: 0,
        x0: std::ptr::null_mut(),
    }
}

fn clip(x: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(lower.iter().zip(upper))
        .map(|(&v, (&l, &u))| {
            if v < l {
                l
            } else if v > u {
                u
            } else {
                v
            }
        })
        .collect()
}

/// Copies the constraint values of an evaluation (checking their number).
fn store_constraints(ev: &Evaluation, m: usize, c: &mut [f64]) {
    assert_eq!(
        ev.c.len(),
        m,
        "expected {m} constraint values, got {}",
        ev.c.len()
    );
    c[..m].copy_from_slice(&ev.c);
}

/// Copies the derivatives of an evaluation into Kraft's arrays (`g` of length
/// `n + 1`, `a` column-major with leading dimension `la`).
fn store_derivatives(ev: &Evaluation, n: usize, m: usize, la: usize, g: &mut [f64], a: &mut [f64]) {
    assert_eq!(
        ev.g.len(),
        n,
        "expected a gradient of length {n}, got {}",
        ev.g.len()
    );
    assert_eq!(
        ev.a.len(),
        m * n,
        "expected a {m} x {n} constraint Jacobian, got {} values",
        ev.a.len()
    );
    g[..n].copy_from_slice(&ev.g);
    for j in 0..m {
        for i in 0..n {
            a[j + i * la] = ev.a[j * n + i];
        }
    }
}

/// Minimizes `f(x)` subject to `c_j(x) = 0` (`j < meq`), `c_j(x) ≥ 0`
/// (`meq ≤ j < m`) and `lower ≤ x ≤ upper` with Kraft's SLSQP, driven like
/// `scipy.optimize.minimize(method="SLSQP", jac=True)`.
///
/// `evaluate(x, with_derivatives)` returns the objective and the `m`
/// constraint values at `x` and, if `with_derivatives`, their gradients (it
/// may always return them). The starting point is clipped to the bounds and
/// every point passed to `evaluate` lies within the bounds (infinite bounds
/// are allowed). Errors of `evaluate` abort the optimization.
///
/// # Panics
/// If the bounds do not have the length of `x0`, `meq > m`, or an evaluation
/// returns values of the wrong length.
pub fn minimize_kraft<F, E>(
    mut evaluate: F,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    meq: usize,
    m: usize,
    options: &KraftOptions,
) -> Result<KraftResult, E>
where
    F: FnMut(&[f64], bool) -> Result<Evaluation, E>,
{
    let n = x0.len();
    assert!(
        lower.len() == n && upper.len() == n,
        "bounds must have the length of x0"
    );
    assert!(meq <= m, "meq must not exceed m");
    let la = m.max(1);
    let mut x = clip(x0, lower, upper);
    let mut c = vec![0.0; la];
    let mut g = vec![0.0; n + 1];
    let mut a = vec![0.0; la * (n + 1)];
    let (mut len_w, mut len_jw) = (0i32, 0i32);
    // SAFETY: `length_work` only writes the two integers.
    unsafe { length_work(&mut len_w, &mut len_jw, m as i32, meq as i32, n as i32) };
    let mut w = vec![0.0; len_w.max(1) as usize];
    let mut jw = vec![0i32; len_jw.max(1) as usize];
    let mut state = new_state();
    let (mut mi, mut meqi, mut lai, mut ni) = (m as i32, meq as i32, la as i32, n as i32);
    let mut acc = options.acc;
    let mut iter = i32::try_from(options.maxiter).unwrap_or(i32::MAX);
    let mut mode = 0i32;

    // Initial evaluation (SciPy evaluates f, g, c and their Jacobian at x0).
    let ev = evaluate(&x, true)?;
    let mut f = ev.f;
    store_constraints(&ev, m, &mut c);
    store_derivatives(&ev, n, m, la, &mut g, &mut a);
    let (mut evaluations, mut gradient_evaluations) = (1usize, 1usize);
    // Point at which the derivatives in `g` / `a` were computed.
    let mut derivatives_at: Option<Vec<f64>> = Some(x.clone());
    loop {
        // SAFETY: all arrays have the sizes required by Kraft's routine:
        // `x`, `lower`, `upper` of length n, `g` of length n + 1, `c` of length
        // max(m, 1), `a` of size max(m, 1) x (n + 1) (column-major, leading
        // dimension `la`), and the workspaces sized by `length_work`.
        unsafe {
            slsqp(
                &mut mi,
                &mut meqi,
                &mut lai,
                &mut ni,
                x.as_mut_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                &mut f,
                c.as_mut_ptr(),
                g.as_mut_ptr(),
                a.as_mut_ptr(),
                &mut acc,
                &mut iter,
                &mut mode,
                w.as_mut_ptr(),
                &mut len_w,
                jw.as_mut_ptr(),
                &mut len_jw,
                &mut state,
            );
        }
        match mode {
            // Function evaluation at a trial point of the line search; NLopt's
            // translation requests the derivatives along with the first trial
            // point (mode -2), which is usually accepted.
            1 | -2 => {
                let xc = clip(&x, lower, upper);
                let with_derivatives = mode == -2;
                let ev = evaluate(&xc, with_derivatives)?;
                evaluations += 1;
                f = ev.f;
                store_constraints(&ev, m, &mut c);
                if with_derivatives {
                    store_derivatives(&ev, n, m, la, &mut g, &mut a);
                    gradient_evaluations += 1;
                    derivatives_at = Some(x.clone());
                } else {
                    derivatives_at = None;
                }
            }
            // Derivatives at the accepted point.
            -1 => {
                if derivatives_at.as_deref() != Some(&x[..]) {
                    let xc = clip(&x, lower, upper);
                    let ev = evaluate(&xc, true)?;
                    store_derivatives(&ev, n, m, la, &mut g, &mut a);
                    gradient_evaluations += 1;
                    derivatives_at = Some(x.clone());
                }
            }
            _ => break,
        }
    }
    Ok(KraftResult {
        x,
        f,
        mode,
        iterations: iter.max(0) as usize,
        evaluations,
        gradient_evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SciPy's documentation example (`minimize` tutorial): minimize
    /// `(x0 - 1)² + (x1 - 2.5)²` subject to three linear inequalities and
    /// `x ≥ 0`; the solution is `(1.4, 1.7)`.
    #[test]
    fn scipy_tutorial_problem() {
        let eval = |x: &[f64], _: bool| -> Result<Evaluation, ()> {
            Ok(Evaluation {
                f: (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2),
                c: vec![
                    x[0] - 2.0 * x[1] + 2.0,
                    -x[0] - 2.0 * x[1] + 6.0,
                    -x[0] + 2.0 * x[1] + 2.0,
                ],
                g: vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.5)],
                a: vec![1.0, -2.0, -1.0, -2.0, -1.0, 2.0],
            })
        };
        let inf = f64::INFINITY;
        let r = minimize_kraft(
            eval,
            &[2.0, 0.0],
            &[0.0, 0.0],
            &[inf, inf],
            0,
            3,
            &KraftOptions::default(),
        )
        .unwrap();
        assert!(r.success(), "{}", r.message());
        assert!(
            (r.x[0] - 1.4).abs() < 1e-6 && (r.x[1] - 1.7).abs() < 1e-6,
            "{:?}",
            r.x
        );
    }

    /// An equality constraint: minimize `x0² + x1² + x2²` subject to
    /// `x0 + x1 + x2 = 1` (solution `1/3` each) and `x1 ≥ 0.5`.
    #[test]
    fn equality_and_inequality_constraints() {
        let eval = |x: &[f64], _: bool| -> Result<Evaluation, ()> {
            Ok(Evaluation {
                f: x.iter().map(|v| v * v).sum(),
                c: vec![x[0] + x[1] + x[2] - 1.0, x[1] - 0.5],
                g: x.iter().map(|v| 2.0 * v).collect(),
                a: vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            })
        };
        let r = minimize_kraft(
            eval,
            &[1.0, 1.0, 1.0],
            &[-5.0; 3],
            &[5.0; 3],
            1,
            2,
            &KraftOptions::default(),
        )
        .unwrap();
        assert!(r.success(), "{}", r.message());
        for (v, e) in r.x.iter().zip([0.25, 0.5, 0.25]) {
            assert!((v - e).abs() < 1e-6, "{:?}", r.x);
        }
        // The iteration limit is reported as mode 9.
        let r = minimize_kraft(
            eval,
            &[1.0, 1.0, 1.0],
            &[-5.0; 3],
            &[5.0; 3],
            1,
            2,
            &KraftOptions {
                maxiter: 1,
                acc: 1e-12,
            },
        )
        .unwrap();
        assert_eq!(r.mode, 9);
        assert_eq!(r.message(), "Iteration limit reached");
    }
}
