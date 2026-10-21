//! Constrained candidate generation with SLSQP, mirroring BoTorch's
//! `gen_candidates_scipy` when parameter constraints are present
//! ([`gen_candidates_slsqp`]) and `project_to_feasible_space_via_slsqp`
//! ([`project_to_feasible_space`]).
//!
//! BoTorch optimizes all initial conditions of a chunk (`batch_limit`
//! restarts, by default all restarts with linear constraints and one with
//! nonlinear constraints) as *one* SLSQP problem: the objective is the
//! negated sum of the acquisition values of the restarts, and the
//! constraints are replicated for every restart. [`gen_candidates_slsqp`]
//! does exactly this with Kraft's SLSQP (the vendored `slsqp` crate) driven
//! like `scipy.optimize.minimize(method="SLSQP")`: equality constraints
//! first, then the linear and nonlinear inequality constraints in SciPy's
//! order, `maxiter` major iterations and the accuracy `ftol`.

use std::collections::BTreeMap;

use rayon::prelude::*;
use slsqp::{minimize_kraft, Evaluation, KraftOptions};

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::optim::initializers::apply_fixed_features;
use crate::optim::parameter_constraints::{
    evaluate_feasibility, project_to_equality_constraints, unfixed_constraints, LinearConstraint,
    ParameterConstraints, CONSTRAINT_TOLERANCE,
};
use crate::Matrix;

/// Default maximum number of SLSQP iterations of acquisition optimization
/// (`gen_candidates_scipy`'s `options.get("maxiter", 2000)`).
pub const DEFAULT_SLSQP_MAXITER: usize = 2000;
/// SciPy's default SLSQP accuracy (`ftol`).
pub const DEFAULT_SLSQP_FTOL: f64 = 1e-6;

/// SLSQP settings (SciPy's `maxiter` and `ftol`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlsqpOptions {
    /// Maximum number of major iterations (default 2000, BoTorch's).
    pub maxiter: usize,
    /// Requested accuracy (default `1e-6`, SciPy's).
    pub ftol: f64,
}

impl Default for SlsqpOptions {
    fn default() -> Self {
        Self {
            maxiter: DEFAULT_SLSQP_MAXITER,
            ftol: DEFAULT_SLSQP_FTOL,
        }
    }
}

/// The result of one SLSQP problem over a chunk of restarts.
#[derive(Clone, Debug, PartialEq)]
pub struct SlsqpChunkResult {
    /// The optimized candidate sets (`q × d` each), clamped to the bounds.
    pub candidates: Vec<Matrix>,
    /// The acquisition values of the candidate sets.
    pub acq_values: Vec<f64>,
    /// SLSQP's exit mode (SciPy's `status`; 0 is success).
    pub mode: i32,
    /// SciPy's message for the exit mode.
    pub message: &'static str,
    /// Major iterations.
    pub iterations: usize,
    /// Objective evaluations (each evaluates every restart of the chunk).
    pub evaluations: usize,
    /// Messages corresponding to BoTorch's `OptimizationWarning`s (failed
    /// optimization other than the iteration limit, infeasible candidates
    /// with nonlinear constraints); they trigger BoTorch's retry.
    pub warnings: Vec<String>,
}

/// The constraint rows of an SLSQP problem over flattened variables: sparse
/// linear rows (`Σ c x - rhs`) and nonlinear constraints on points or
/// q-batches.
struct ProblemRows {
    /// `(entries, rhs)` of the linear equality rows.
    eq: Vec<(Vec<(usize, f64)>, f64)>,
    /// `(entries, rhs)` of the linear inequality rows.
    ineq: Vec<(Vec<(usize, f64)>, f64)>,
}

/// Linear rows for `b` restarts of `q` points with `nf` variables each
/// (SciPy's `make_scipy_linear_constraints` order: per constraint, per
/// restart, per point).
fn linear_rows(
    cs: &[LinearConstraint],
    b: usize,
    q: usize,
    nf: usize,
) -> Vec<(Vec<(usize, f64)>, f64)> {
    let mut rows = Vec::new();
    for c in cs {
        let per = c.rows(q, nf);
        for r in 0..b {
            for row in &per {
                rows.push((
                    row.iter().map(|&(col, v)| (r * q * nf + col, v)).collect(),
                    c.rhs(),
                ));
            }
        }
    }
    rows
}

fn dot(row: &[(usize, f64)], x: &[f64]) -> f64 {
    row.iter().map(|&(col, c)| c * x[col]).sum()
}

/// `columnwise_clamp` with BoTorch's `raise_on_violation` check
/// (`torch.allclose` tolerances).
pub(crate) fn clamp_to_bounds(x: &mut Matrix, bounds: &Matrix, raise: bool) -> Result<()> {
    for i in 0..x.nrows() {
        for k in 0..x.ncols() {
            let (lo, hi) = (bounds[(0, k)], bounds[(1, k)]);
            let v = x[(i, k)];
            let c = v.max(lo).min(hi);
            if raise && ((v - c).abs() > 1e-8 + 1e-5 * c.abs() || v.is_nan()) {
                return Err(BotorchError::InvalidArgument(format!(
                    "Original value(s) are out of bounds: x[{i}, {k}] = {v} not in [{lo}, {hi}]"
                )));
            }
            x[(i, k)] = c;
        }
    }
    Ok(())
}

/// Gradients of constraints as sparse rows `(column, value)`.
type SparseRows = Vec<Vec<(usize, f64)>>;

/// The nonlinear constraint evaluations of a set of candidate sets: for every
/// constraint (in order), every restart and every point (intra-point) or
/// q-batch (inter-point), the value and the gradient with respect to the
/// flattened variables.
fn nonlinear_rows(
    constraints: &ParameterConstraints,
    sets: &[Matrix],
    free: &[usize],
    with_grad: bool,
) -> Result<(Vec<f64>, SparseRows)> {
    let q = sets.first().map_or(0, |s| s.nrows());
    let d = sets.first().map_or(0, |s| s.ncols());
    let nf = free.len();
    let mut values = Vec::new();
    let mut grads = Vec::new();
    for c in &constraints.nonlinear {
        for (r, x) in sets.iter().enumerate() {
            if c.is_intrapoint() {
                for p in 0..q {
                    let row: Vec<f64> = x.row(p).iter().copied().collect();
                    let (v, g) = c.evaluate(&row)?;
                    if g.len() != d {
                        return Err(BotorchError::DimensionMismatch(format!(
                            "an intra-point constraint returned a gradient of length {}, \
                             expected {d}",
                            g.len()
                        )));
                    }
                    values.push(v);
                    if with_grad {
                        grads.push(
                            free.iter()
                                .enumerate()
                                .map(|(j, &k)| (r * q * nf + p * nf + j, g[k]))
                                .collect(),
                        );
                    }
                }
            } else {
                let flat: Vec<f64> = (0..q)
                    .flat_map(|p| x.row(p).iter().copied().collect::<Vec<_>>())
                    .collect();
                let (v, g) = c.evaluate(&flat)?;
                if g.len() != q * d {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "an inter-point constraint returned a gradient of length {}, expected {}",
                        g.len(),
                        q * d
                    )));
                }
                values.push(v);
                if with_grad {
                    let mut entries = Vec::with_capacity(q * nf);
                    for p in 0..q {
                        for (j, &k) in free.iter().enumerate() {
                            entries.push((r * q * nf + p * nf + j, g[p * d + k]));
                        }
                    }
                    grads.push(entries);
                }
            }
        }
    }
    Ok((values, grads))
}

/// Optimizes the acquisition function from the initial conditions (`q × d`
/// each) as one SLSQP problem (BoTorch's `gen_candidates_scipy` with
/// constraints): the objective is `-Σ_r acq(X_r)`, the constraints apply to
/// every restart, fixed features are removed from the optimization (with the
/// linear constraints reduced accordingly) and initial conditions are
/// projected onto the intra-point equality constraints first.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
/// [`BotorchError::InvalidArgument`] for initial conditions outside the bounds,
/// [`BotorchError::CandidateGeneration`] for violated constraints on fixed
/// features only, [`BotorchError::Numerical`] for NaN values or gradients,
/// and errors of the acquisition function and the constraints.
pub fn gen_candidates_slsqp(
    initial_conditions: &[Matrix],
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    fixed_features: &BTreeMap<usize, f64>,
    constraints: &ParameterConstraints,
    options: &SlsqpOptions,
) -> Result<SlsqpChunkResult> {
    let b = initial_conditions.len();
    let d = bounds.ncols();
    if b == 0 {
        return Err(BotorchError::InvalidArgument(
            "at least one initial condition is required".into(),
        ));
    }
    let q = initial_conditions[0].nrows();
    for ic in initial_conditions {
        if ic.shape() != (q, d) || q == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "initial conditions must be {q} x {d}, got {}x{}",
                ic.nrows(),
                ic.ncols()
            )));
        }
    }
    constraints.validate(q, d)?;
    let free: Vec<usize> = (0..d).filter(|k| !fixed_features.contains_key(k)).collect();
    let nf = free.len();
    let reduced = unfixed_constraints(constraints, fixed_features, d)?;
    let free_bounds = Matrix::from_fn(2, nf, |i, j| bounds[(i, free[j])]);
    // Initial conditions in the reduced space, clamped (raising on violations).
    let mut x0 = Vec::with_capacity(b * q * nf);
    for ic in initial_conditions {
        let mut r = Matrix::from_fn(q, nf, |p, j| ic[(p, free[j])]);
        clamp_to_bounds(&mut r, &free_bounds, true)?;
        if !reduced.equality.is_empty() {
            r = project_to_equality_constraints(&r, &reduced.equality)?;
            clamp_to_bounds(&mut r, &free_bounds, false)?;
        }
        for p in 0..q {
            x0.extend(r.row(p).iter());
        }
    }
    let lower: Vec<f64> = (0..b * q * nf).map(|i| free_bounds[(0, i % nf)]).collect();
    let upper: Vec<f64> = (0..b * q * nf).map(|i| free_bounds[(1, i % nf)]).collect();
    let rows = ProblemRows {
        eq: linear_rows(&reduced.equality, b, q, nf),
        ineq: linear_rows(&reduced.inequality, b, q, nf),
    };
    let n_nonlinear: usize = reduced
        .nonlinear
        .iter()
        .map(|c| if c.is_intrapoint() { b * q } else { b })
        .sum();
    let meq = rows.eq.len();
    let m = meq + rows.ineq.len() + n_nonlinear;
    let n = b * q * nf;
    let assemble = |x: &[f64]| -> Vec<Matrix> {
        (0..b)
            .map(|r| {
                let mut s = Matrix::zeros(q, d);
                for p in 0..q {
                    for (j, &k) in free.iter().enumerate() {
                        s[(p, k)] = x[r * q * nf + p * nf + j];
                    }
                }
                apply_fixed_features(&mut s, fixed_features);
                s
            })
            .collect()
    };
    let evaluate = |x: &[f64], with_grad: bool| -> Result<Evaluation> {
        let nan = x.iter().filter(|v| v.is_nan()).count();
        if nan > 0 {
            return Err(BotorchError::Numerical(format!(
                "{nan} elements of the {} element array `x` are NaN.",
                x.len()
            )));
        }
        let sets = assemble(x);
        let mut f = 0.0;
        let mut g = if with_grad { vec![0.0; n] } else { Vec::new() };
        if with_grad {
            let results = sets
                .par_iter()
                .map(|s| acq_function.forward_with_grad(s))
                .collect::<Result<Vec<_>>>()?;
            for (r, (v, grad)) in results.iter().enumerate() {
                f += -v;
                for p in 0..q {
                    for (j, &k) in free.iter().enumerate() {
                        g[r * q * nf + p * nf + j] = -grad[(p, k)];
                    }
                }
            }
            let nan = g.iter().filter(|v| v.is_nan()).count();
            if nan > 0 {
                return Err(BotorchError::Numerical(format!(
                    "{nan} elements of the {n} element gradient array `gradf` are NaN. This \
                     often indicates numerical issues."
                )));
            }
        } else {
            for v in acq_function.forward_many(&sets)? {
                f += -v;
            }
        }
        let mut c = Vec::with_capacity(m);
        for (row, rhs) in rows.eq.iter().chain(&rows.ineq) {
            c.push(dot(row, x) - rhs);
        }
        let (nl_values, nl_grads) = nonlinear_rows(&reduced, &sets, &free, with_grad)?;
        c.extend(nl_values);
        let mut a = Vec::new();
        if with_grad {
            a = vec![0.0; m * n];
            for (i, (row, _)) in rows.eq.iter().chain(&rows.ineq).enumerate() {
                for &(col, v) in row {
                    a[i * n + col] = v;
                }
            }
            let offset = rows.eq.len() + rows.ineq.len();
            for (i, entries) in nl_grads.iter().enumerate() {
                for &(col, v) in entries {
                    a[(offset + i) * n + col] = v;
                }
            }
        }
        Ok(Evaluation { f, c, g, a })
    };
    let kraft = KraftOptions {
        maxiter: options.maxiter,
        acc: options.ftol,
    };
    let res = minimize_kraft(evaluate, &x0, &lower, &upper, meq, m, &kraft)?;
    let mut candidates = assemble(&res.x);
    for s in candidates.iter_mut() {
        clamp_to_bounds(s, bounds, true)?;
        apply_fixed_features(s, fixed_features);
    }
    let mut warnings = Vec::new();
    if res.mode != 0 && res.mode != 9 {
        warnings.push(format!(
            "Optimization failed within `scipy.optimize.minimize` with status {} and message {}.",
            res.mode,
            res.message()
        ));
    }
    if !constraints.nonlinear.is_empty() {
        let only_nonlinear = ParameterConstraints {
            nonlinear: constraints.nonlinear.clone(),
            ..Default::default()
        };
        let mut infeasible = 0;
        for s in &candidates {
            if !evaluate_feasibility(s, &only_nonlinear, CONSTRAINT_TOLERANCE)? {
                infeasible += 1;
            }
        }
        if infeasible > 0 {
            warnings.push(format!(
                "The optimizer produced infeasible candidates. {infeasible} out of {b} batches \
                 of candidates were infeasible."
            ));
        }
    }
    let acq_values = acq_function.forward_many(&candidates)?;
    Ok(SlsqpChunkResult {
        candidates,
        acq_values,
        mode: res.mode,
        message: res.message(),
        iterations: res.iterations,
        evaluations: res.evaluations,
        warnings,
    })
}

/// Projects candidate sets onto the feasible set (BoTorch's
/// `project_to_feasible_space_via_slsqp`): the closest points (in the
/// Euclidean norm over all candidate sets jointly) that satisfy the bounds,
/// the fixed features (as equal bounds) and the linear and nonlinear
/// constraints, found by SLSQP with `ftol = 1e-8` and 100 iterations
/// (SciPy's default `maxiter`).
///
/// # Errors
/// [`BotorchError::CandidateGeneration`] with BoTorch's message if SLSQP does
/// not converge, [`BotorchError::InvalidArgument`] for candidates outside the
/// bounds, and errors of the nonlinear constraints.
pub fn project_to_feasible_space(
    sets: &[Matrix],
    bounds: &Matrix,
    constraints: &ParameterConstraints,
    fixed_features: &BTreeMap<usize, f64>,
) -> Result<Vec<Matrix>> {
    if constraints.is_empty() || sets.is_empty() {
        return Ok(sets.to_vec());
    }
    let rows = sets.len() * sets[0].nrows();
    let fixed = fixed_features
        .iter()
        .map(|(&index, &value)| (index, vec![value; rows]))
        .collect();
    project_to_feasible_space_with_fixed_values(
        sets,
        bounds,
        constraints,
        &fixed,
        CONSTRAINT_TOLERANCE,
    )
}

/// Feasible-space projection with per-row fixed values and a dtype-specific
/// SLSQP tolerance. Fixed feature vectors contain one value per point, in
/// candidate-set-major order.
///
/// # Errors
/// The errors of [`project_to_feasible_space`], plus inconsistent candidate,
/// bound or fixed-value dimensions.
pub fn project_to_feasible_space_with_fixed_values(
    sets: &[Matrix],
    bounds: &Matrix,
    constraints: &ParameterConstraints,
    fixed_features: &BTreeMap<usize, Vec<f64>>,
    tolerance: f64,
) -> Result<Vec<Matrix>> {
    if sets.is_empty() {
        return Ok(Vec::new());
    }
    let b = sets.len();
    let (q, d) = sets[0].shape();
    if q == 0 || d == 0 || bounds.shape() != (2, d) || sets.iter().any(|x| x.shape() != (q, d)) {
        return Err(BotorchError::DimensionMismatch(
            "inconsistent projection candidates or bounds".into(),
        ));
    }
    constraints.validate(q, d)?;
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(BotorchError::InvalidArgument(
            "projection tolerance must be finite and positive".into(),
        ));
    }
    if (0..d).any(|j| {
        bounds[(0, j)].is_nan() || bounds[(1, j)].is_nan() || bounds[(0, j)] > bounds[(1, j)]
    }) {
        return Err(BotorchError::InvalidArgument(
            "Lower bounds cannot exceed upper bounds or contain NaN.".into(),
        ));
    }
    let n = b * q * d;
    let target: Vec<f64> = sets
        .iter()
        .flat_map(|s| (0..q).flat_map(move |p| s.row(p).iter().copied().collect::<Vec<_>>()))
        .collect();
    let mut lower: Vec<f64> = (0..n).map(|i| bounds[(0, i % d)]).collect();
    let mut upper: Vec<f64> = (0..n).map(|i| bounds[(1, i % d)]).collect();
    for (&k, values) in fixed_features {
        if k >= d || values.len() != b * q || values.iter().any(|value| !value.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "invalid fixed feature index or per-row values".into(),
            ));
        }
        for (r, &value) in values.iter().enumerate() {
            lower[r * d + k] = value;
            upper[r * d + k] = value;
        }
    }
    let mut x0 = Vec::with_capacity(n);
    for s in sets {
        let mut c = s.clone();
        clamp_to_bounds(&mut c, bounds, true)?;
        for p in 0..q {
            x0.extend(c.row(p).iter());
        }
    }
    let all: Vec<usize> = (0..d).collect();
    let eq = linear_rows(&constraints.equality, b, q, d);
    let ineq = linear_rows(&constraints.inequality, b, q, d);
    let n_nonlinear: usize = constraints
        .nonlinear
        .iter()
        .map(|c| if c.is_intrapoint() { b * q } else { b })
        .sum();
    let meq = eq.len();
    let m = meq + ineq.len() + n_nonlinear;
    let to_sets = |x: &[f64]| -> Vec<Matrix> {
        (0..b)
            .map(|r| Matrix::from_fn(q, d, |p, k| x[r * q * d + p * d + k]))
            .collect()
    };
    let evaluate = |x: &[f64], with_grad: bool| -> Result<Evaluation> {
        let f = 0.5
            * x.iter()
                .zip(&target)
                .map(|(a, t)| (a - t) * (a - t))
                .sum::<f64>();
        let g = if with_grad {
            x.iter().zip(&target).map(|(a, t)| a - t).collect()
        } else {
            Vec::new()
        };
        let mut c = Vec::with_capacity(m);
        for (row, rhs) in eq.iter().chain(&ineq) {
            c.push(dot(row, x) - rhs);
        }
        let (nl_values, nl_grads) = nonlinear_rows(constraints, &to_sets(x), &all, with_grad)?;
        c.extend(nl_values);
        let mut a = Vec::new();
        if with_grad {
            a = vec![0.0; m * n];
            for (i, (row, _)) in eq.iter().chain(&ineq).enumerate() {
                for &(col, v) in row {
                    a[i * n + col] = v;
                }
            }
            let offset = eq.len() + ineq.len();
            for (i, entries) in nl_grads.iter().enumerate() {
                for &(col, v) in entries {
                    a[(offset + i) * n + col] = v;
                }
            }
        }
        Ok(Evaluation { f, c, g, a })
    };
    let kraft = KraftOptions {
        maxiter: 100,
        acc: tolerance,
    };
    let res = minimize_kraft(evaluate, &x0, &lower, &upper, meq, m, &kraft)?;
    if !res.success() {
        return Err(BotorchError::CandidateGeneration(format!(
            "Optimization failed: {}",
            res.message()
        )));
    }
    let final_constraints = evaluate(&res.x, false)?.c;
    let constraints_satisfied = final_constraints.iter().enumerate().all(|(i, value)| {
        value.is_finite()
            && if i < meq {
                value.abs() <= tolerance
            } else {
                *value >= -tolerance
            }
    });
    let bounds_satisfied = res.x.iter().enumerate().all(|(i, value)| {
        value.is_finite() && *value >= lower[i] - tolerance && *value <= upper[i] + tolerance
    });
    if !constraints_satisfied || !bounds_satisfied {
        return Err(BotorchError::CandidateGeneration(
            "Optimization failed: SLSQP returned an infeasible projection.".into(),
        ));
    }
    Ok(to_sets(&res.x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::optim::parameter_constraints::FnNonlinearConstraint;

    #[test]
    fn projection_respects_different_fixed_values_per_row() {
        let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        let constraints = ParameterConstraints {
            inequality: vec![
                LinearConstraint::intra_point(vec![0, 1], vec![1.0, 1.0], 1.2).unwrap(),
            ],
            ..Default::default()
        };
        let sets = vec![
            Matrix::from_row_slice(1, 2, &[0.2, 0.2]),
            Matrix::from_row_slice(1, 2, &[0.8, 0.1]),
        ];
        let mut fixed: BTreeMap<usize, Vec<f64>> = [(0, vec![0.3, 0.7])].into_iter().collect();
        let projected =
            project_to_feasible_space_with_fixed_values(&sets, &bounds, &constraints, &fixed, 1e-8)
                .unwrap();
        assert!((projected[0][(0, 0)] - 0.3).abs() < 1e-12);
        assert!((projected[0][(0, 1)] - 0.9).abs() < 1e-8);
        assert!((projected[1][(0, 1)] - 0.5).abs() < 1e-8);
        fixed.insert(1, vec![0.9, 0.5]);
        assert!(project_to_feasible_space_with_fixed_values(
            &sets,
            &bounds,
            &constraints,
            &fixed,
            1e-8
        )
        .is_ok());
    }

    #[test]
    fn projection_rejects_an_infeasible_success_status() {
        let bounds = Matrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let constraints = ParameterConstraints {
            inequality: vec![LinearConstraint::intra_point(vec![0], vec![1.0], 2.0).unwrap()],
            ..Default::default()
        };
        assert!(project_to_feasible_space(
            &[Matrix::from_row_slice(1, 1, &[0.2])],
            &bounds,
            &constraints,
            &BTreeMap::new(),
        )
        .is_err());
    }

    #[test]
    fn projection_reaches_the_feasible_set() {
        let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        let constraints = ParameterConstraints {
            inequality: vec![
                LinearConstraint::intra_point(vec![0, 1], vec![1.0, 1.0], 1.2).unwrap(),
            ],
            equality: vec![],
            nonlinear: vec![Arc::new(FnNonlinearConstraint::intrapoint(|x| {
                (0.9 - x[0], vec![-1.0, 0.0])
            }))],
        };
        let sets = vec![Matrix::from_row_slice(1, 2, &[0.95, 0.1])];
        let p = project_to_feasible_space(&sets, &bounds, &constraints, &BTreeMap::new()).unwrap();
        let x = &p[0];
        assert!(evaluate_feasibility(x, &constraints, CONSTRAINT_TOLERANCE).unwrap());
        // The projection of (0.95, 0.1) onto {x0 <= 0.9, x0 + x1 >= 1.2} is (0.9, 0.3).
        assert!(
            (x[(0, 0)] - 0.9).abs() < 1e-7 && (x[(0, 1)] - 0.3).abs() < 1e-7,
            "{x}"
        );
        // A fixed feature stays at its value.
        let fixed: BTreeMap<usize, f64> = [(1, 0.5)].into_iter().collect();
        let p = project_to_feasible_space(&sets, &bounds, &constraints, &fixed).unwrap();
        assert_eq!(p[0][(0, 1)], 0.5);
        assert!((p[0][(0, 0)] - 0.9).abs() < 1e-7);
    }
}
