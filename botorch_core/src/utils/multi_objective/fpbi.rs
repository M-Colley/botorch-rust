//! Focus point boundary intersection (FPBI), Schmitt et al. (2022),
//! <https://doi.org/10.3390/inventions7030046>, equations (16)–(27).
//!
//! This is a continuous constrained implementation: it solves each extreme
//! objective, builds the paper's focus hyperplane grid, and optimizes each
//! boundary intersection with SLSQP. Callbacks receive one decision vector.
//! Original inequalities use `g(x) >= 0` (SciPy / BoTorch convention).
//! Focus coordinates are normalized **costs**, regardless of `maximize`.

use super::pareto::is_non_dominated;
use crate::{BotorchError, Matrix, Result};
use slsqp::{minimize_kraft, Evaluation, KraftOptions, KraftResult};

/// A vector-valued objective or constraint callback.
pub type VectorFunction<'a> = dyn Fn(&[f64]) -> Result<Vec<f64>> + 'a;

/// FPBI solver and grid settings.
#[derive(Clone, Debug, PartialEq)]
pub struct FpbiOptions {
    /// BoTorch maximization orientation; `false` accepts the paper's costs.
    pub maximize: bool,
    /// Number of grid positions per basis direction (paper's `r_F`).
    pub grid_resolution: usize,
    /// Guard against an exponentially large `(r_F)^(m-1)` grid.
    pub max_grid_points: usize,
    /// Maximum SLSQP major iterations per restart.
    pub maxiter: usize,
    /// SLSQP objective/constraint stopping accuracy.
    pub ftol: f64,
    /// Maximum allowed final equality/inequality violation.
    pub feasibility_tolerance: f64,
    /// Relative finite-difference step, clipped to decision bounds.
    pub finite_difference_step: f64,
}
impl Default for FpbiOptions {
    fn default() -> Self {
        Self {
            maximize: true,
            grid_resolution: 5,
            max_grid_points: 10_000,
            maxiter: 200,
            ftol: 1e-8,
            feasibility_tolerance: 1e-6,
            finite_difference_step: 1e-6,
        }
    }
}

/// Geometric construction in normalized minimization coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct FpbiGeometry {
    /// Per-objective minima, in oriented original cost units.
    pub utopia: Vec<f64>,
    /// Actual per-objective maxima over the retained extrema. Store these
    /// directly: reconstructing `utopia + ranges` can lose finite endpoints.
    pub nadir: Vec<f64>,
    /// Range across individual objective minima (flat columns use zero).
    pub ranges: Vec<f64>,
    /// Which objectives vary over the nondominated extrema.
    pub active_objectives: Vec<bool>,
    /// Normalized extreme points, each row minimizes one objective.
    pub normalized_extremes: Matrix,
    /// Extreme indices defining the farthest pair (first tie wins).
    pub farthest_pair: (usize, usize),
    /// Midpoint of the farthest extreme pair.
    pub center: Vec<f64>,
    /// User-supplied normalized cost-space focus (zero by default).
    pub focus: Vec<f64>,
    /// Unnormalized search direction `focus - center`.
    pub search_direction: Vec<f64>,
    /// Basis directions as rows; first follows the farthest extreme pair.
    pub basis: Matrix,
    /// Objective-space grid in the paper's order.
    pub anchors: Matrix,
    /// Whether the retained extremes all coincide.
    pub degenerate: bool,
    /// True if deterministic orthogonal completion was needed for a
    /// rank-degenerate generalized cross product (extension to the paper).
    pub used_degenerate_fallback: bool,
}

/// Outcome of a scalar SLSQP solve after choosing the best restart.
#[derive(Clone, Debug, PartialEq)]
pub struct FpbiSolveStatus {
    /// Index of the minimized objective, or anchor, within its phase.
    pub index: usize,
    /// Whether this is an individual objective minimum.
    pub extreme: bool,
    /// Optimizer exit mode (0 means converged).
    pub mode: i32,
    /// SLSQP's exit message.
    pub message: String,
    /// Maximum final constraint violation; checked independently of SLSQP.
    pub max_violation: f64,
    /// Whether this final point satisfies the constraints within tolerance.
    pub feasible: bool,
    /// Whether SLSQP converged and the final point is feasible.
    pub success: bool,
    /// Number of major iterations for the selected restart.
    pub iterations: usize,
    /// Number of attempted restarts.
    pub restarts: usize,
    /// Boundary-intersection parameter; absent for extreme solves.
    pub kappa: Option<f64>,
}

/// FPBI front plus diagnostics. Converged, feasible anchor solutions and
/// individual extrema are filtered for dominance and duplicates among these
/// samples. Local scalar solves do not guarantee a globally Pareto front.
#[derive(Clone, Debug, PartialEq)]
pub struct FpbiResult {
    /// Decision vectors on the returned front (rows).
    pub x: Matrix,
    /// Original objective values on the returned front (rows).
    pub y: Matrix,
    /// Anchor index for each front row (`None` for an individual extreme).
    pub anchor_indices: Vec<Option<usize>>,
    /// Decision vectors minimizing each objective.
    pub extreme_x: Matrix,
    /// Original extreme objective values.
    pub extreme_y: Matrix,
    /// Grid and normalization details.
    pub geometry: FpbiGeometry,
    /// One selected-restart status per extreme followed by each anchor.
    pub statuses: Vec<FpbiSolveStatus>,
}

fn invalid(message: &str) -> BotorchError {
    BotorchError::InvalidArgument(message.into())
}

/// Validate the nominal grid before allocating it or solving any extrema.
fn grid_point_count(m: usize, options: &FpbiOptions) -> Result<usize> {
    if options.grid_resolution == 0 || options.max_grid_points == 0 {
        return Err(invalid(
            "grid_resolution and max_grid_points must be positive",
        ));
    }
    let exponent = u32::try_from(m - 1).map_err(|_| invalid("FPBI grid size overflows"))?;
    let count = options
        .grid_resolution
        .checked_pow(exponent)
        .ok_or_else(|| invalid("FPBI grid size overflows"))?;
    if count > options.max_grid_points {
        return Err(invalid(
            "FPBI grid exceeds max_grid_points; reduce grid_resolution",
        ));
    }
    Ok(count)
}

fn checked_values(
    f: &VectorFunction<'_>,
    x: &[f64],
    expected: usize,
    name: &str,
) -> Result<Vec<f64>> {
    let values = f(x)?;
    if values.len() != expected {
        return Err(BotorchError::DimensionMismatch(format!(
            "{name} returned {} values, expected {expected}",
            values.len()
        )));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(BotorchError::Numerical(format!(
            "{name} returned a non-finite value"
        )));
    }
    Ok(values)
}
fn norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |a, b| a.hypot(*b))
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn generalized_cross(rows: &[Vec<f64>], m: usize) -> Vec<f64> {
    (0..m)
        .map(|j| {
            let minor =
                Matrix::from_fn(m - 1, m - 1, |r, c| rows[r][if c < j { c } else { c + 1 }]);
            if j % 2 == 0 {
                minor.determinant()
            } else {
                -minor.determinant()
            }
        })
        .collect()
}

/// Construct FPBI geometry from individual objective minima in original
/// units. The asymmetric grid is source-faithful: positions are
/// `(k - r_F / 2) * L / r_F` for `k = 1, ..., r_F`.
pub fn fpbi_geometry(
    extremes: &Matrix,
    focus: Option<&[f64]>,
    options: &FpbiOptions,
) -> Result<FpbiGeometry> {
    let m = extremes.ncols();
    if m < 2 || extremes.nrows() != m {
        return Err(invalid("extremes must be an m x m matrix with m >= 2"));
    }
    if extremes.iter().any(|v| !v.is_finite()) {
        return Err(invalid("extremes must be finite"));
    }
    let count = grid_point_count(m, options)?;
    if let Some(f) = focus {
        if f.len() != m {
            return Err(BotorchError::DimensionMismatch(
                "focus must match the objective count".into(),
            ));
        }
        if f.iter().any(|v| !v.is_finite()) {
            return Err(invalid("focus must be finite normalized costs"));
        }
    }
    let costs = extremes.map(|v| if options.maximize { -v } else { v });
    let retained: Vec<_> = is_non_dominated(&costs, false, true)
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| v.then_some(i))
        .collect();
    let utopia: Vec<_> = (0..m)
        .map(|j| {
            retained
                .iter()
                .map(|&i| costs[(i, j)])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let nadir: Vec<_> = (0..m)
        .map(|j| {
            retained
                .iter()
                .map(|&i| costs[(i, j)])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    let ranges: Vec<_> = nadir.iter().zip(&utopia).map(|(hi, lo)| hi - lo).collect();
    if ranges.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "extreme objective ranges overflow; rescale objective units",
        ));
    }
    let active_objectives: Vec<_> = ranges.iter().map(|v| *v > 0.0).collect();
    let active: Vec<_> = active_objectives
        .iter()
        .enumerate()
        .filter_map(|(j, &v)| v.then_some(j))
        .collect();
    let focus = focus.map_or_else(|| vec![0.0; m], |f| f.to_vec());
    if focus
        .iter()
        .zip(&active_objectives)
        .any(|(v, active)| !active && *v != 0.0)
    {
        return Err(invalid("focus must be zero along flat objectives"));
    }
    let normalized_extremes = Matrix::from_fn(m, m, |i, j| {
        if ranges[j] > 0.0 {
            let hi = nadir[j];
            let scale = utopia[j].abs().max(hi.abs()).max(1.0);
            let lo_s = utopia[j] / scale;
            (costs[(i, j)] / scale - lo_s) / (hi / scale - lo_s)
        } else {
            0.0
        }
    });
    let mut pair = (retained[0], retained[0]);
    let mut length = -1.0;
    for &a in &retained {
        for &b in &retained {
            let d = norm(
                &(0..m)
                    .map(|j| normalized_extremes[(b, j)] - normalized_extremes[(a, j)])
                    .collect::<Vec<_>>(),
            );
            if d > length {
                pair = (a, b);
                length = d;
            }
        }
    }
    let center: Vec<_> = (0..m)
        .map(|j| (normalized_extremes[(pair.0, j)] + normalized_extremes[(pair.1, j)]) / 2.0)
        .collect();
    let search_direction: Vec<_> = (0..m).map(|j| focus[j] - center[j]).collect();
    let mut basis_rows = Vec::<Vec<f64>>::new();
    let mut fallback = false;
    let k = active.len();
    if length > 0.0 && k > 1 {
        let s: Vec<_> = active.iter().map(|&j| search_direction[j]).collect();
        basis_rows.push(
            active
                .iter()
                .map(|&j| {
                    (normalized_extremes[(pair.1, j)] - normalized_extremes[(pair.0, j)]) / length
                })
                .collect(),
        );
        let remaining: Vec<_> = retained
            .iter()
            .copied()
            .filter(|i| *i != pair.0 && *i != pair.1)
            .collect();
        let auxiliary: Vec<Vec<_>> = remaining
            .iter()
            .map(|&i| {
                active
                    .iter()
                    .map(|&j| normalized_extremes[(i, j)] - center[j])
                    .collect()
            })
            .collect();
        for l in 1..k - 1 {
            let mut rows = vec![s.clone()];
            rows.extend(basis_rows.clone());
            if auxiliary.len() >= k - 2 {
                rows.extend(auxiliary.iter().skip(l).cloned());
            }
            let mut direction = if rows.len() == k - 1 {
                generalized_cross(&rows, k)
            } else {
                vec![0.0; k]
            };
            let mut magnitude = norm(&direction);
            if magnitude <= 1e-12 || !magnitude.is_finite() {
                // The paper's generalized cross is undefined at rank loss.
                // Complete the direction basis deterministically instead.
                fallback = true;
                let mut orthogonal = Vec::<Vec<f64>>::new();
                for v in basis_rows.iter().chain(std::iter::once(&s)) {
                    let mut q = v.clone();
                    for u in &orthogonal {
                        let w = dot(&q, u);
                        for j in 0..k {
                            q[j] -= w * u[j];
                        }
                    }
                    let n = norm(&q);
                    if n > 1e-12 {
                        for v in &mut q {
                            *v /= n;
                        }
                        orthogonal.push(q);
                    }
                }
                magnitude = 0.0;
                for axis in 0..k {
                    let mut q = vec![0.0; k];
                    q[axis] = 1.0;
                    for u in &orthogonal {
                        let w = dot(&q, u);
                        for j in 0..k {
                            q[j] -= w * u[j];
                        }
                    }
                    let n = norm(&q);
                    if n > magnitude {
                        magnitude = n;
                        direction = q;
                    }
                }
            }
            if magnitude <= 1e-12 {
                return Err(BotorchError::Numerical(
                    "unable to construct FPBI direction basis".into(),
                ));
            }
            for v in &mut direction {
                *v /= magnitude;
            }
            basis_rows.push(direction);
        }
    }
    let mut basis = Matrix::zeros(m - 1, m);
    for (i, row) in basis_rows.iter().enumerate() {
        for (a, &j) in active.iter().enumerate() {
            basis[(i, j)] = row[a];
        }
    }
    let anchors = Matrix::from_fn(count, m, |i, j| {
        let mut value = center[j];
        let mut code = i;
        for l in (0..m - 1).rev() {
            let k = code % options.grid_resolution + 1;
            code /= options.grid_resolution;
            let gamma = (k as f64 - options.grid_resolution as f64 / 2.0) * length
                / options.grid_resolution as f64;
            value += gamma * basis[(l, j)];
        }
        value
    });
    Ok(FpbiGeometry {
        utopia,
        nadir,
        ranges,
        active_objectives,
        normalized_extremes,
        farthest_pair: pair,
        center,
        focus,
        search_direction,
        basis,
        anchors,
        degenerate: length == 0.0,
        used_degenerate_fallback: fallback,
    })
}

type ScalarEvaluation = (f64, Vec<f64>);
fn solve_scalar(
    function: &dyn Fn(&[f64]) -> Result<ScalarEvaluation>,
    starts: &[Vec<f64>],
    lower: &[f64],
    upper: &[f64],
    meq: usize,
    nconstraints: usize,
    options: &FpbiOptions,
) -> Result<(KraftResult, f64)> {
    let mut selected: Option<(KraftResult, f64)> = None;
    for start in starts {
        let evaluate = |x: &[f64], with_grad: bool| -> Result<Evaluation> {
            let (f, c) = function(x)?;
            if !f.is_finite() || c.len() != nconstraints || c.iter().any(|v| !v.is_finite()) {
                return Err(BotorchError::Numerical(
                    "non-finite or inconsistent scalar FPBI evaluation".into(),
                ));
            }
            let mut g = Vec::new();
            let mut a = Vec::new();
            if with_grad {
                g = vec![0.0; x.len()];
                a = vec![0.0; nconstraints * x.len()];
                for j in 0..x.len() {
                    let h = options.finite_difference_step * x[j].abs().max(1.0);
                    let lo = (x[j] - h).max(lower[j]);
                    let hi = (x[j] + h).min(upper[j]);
                    if hi <= lo {
                        continue;
                    }
                    let mut left = x.to_vec();
                    let mut right = x.to_vec();
                    left[j] = lo;
                    right[j] = hi;
                    let (fl, cl) = function(&left)?;
                    let (fr, cr) = function(&right)?;
                    if cl.len() != nconstraints || cr.len() != nconstraints {
                        return Err(invalid("constraint callback changed output shape"));
                    }
                    g[j] = (fr - fl) / (hi - lo);
                    for k in 0..nconstraints {
                        a[k * x.len() + j] = (cr[k] - cl[k]) / (hi - lo);
                    }
                }
                if g.iter().chain(&a).any(|v| !v.is_finite()) {
                    return Err(BotorchError::Numerical(
                        "non-finite FPBI finite-difference derivative".into(),
                    ));
                }
            }
            Ok(Evaluation { f, c, g, a })
        };
        let result = minimize_kraft(
            evaluate,
            start,
            lower,
            upper,
            meq,
            nconstraints,
            &KraftOptions {
                maxiter: options.maxiter,
                acc: options.ftol,
            },
        )?;
        let (_, constraints) = function(&result.x)?;
        let violation = constraints
            .iter()
            .enumerate()
            .map(|(i, v)| if i < meq { v.abs() } else { (-v).max(0.0) })
            .fold(0.0, f64::max);
        let feasible = violation <= options.feasibility_tolerance;
        let choose = selected.as_ref().is_none_or(|(previous, pv)| {
            let prior_feasible = *pv <= options.feasibility_tolerance;
            if feasible != prior_feasible {
                feasible
            } else if feasible && result.success() != previous.success() {
                result.success()
            } else if feasible {
                result.f < previous.f
            } else {
                violation < *pv
            }
        });
        if choose {
            selected = Some((result, violation));
        }
    }
    selected.ok_or_else(|| invalid("at least one initial condition is required"))
}

/// Generate a front by solving continuous FPBI subproblems.
///
/// `bounds` is `2 × d`, `initial_conditions` is `restarts × d`; these
/// starts are used for every individual objective and every anchor. Optional
/// constraints return vectors (`inequalities >= 0`, `equalities == 0`).
/// The intersection parameter is unrestricted, as in equation (27).
/// A failure to find a converged feasible extreme raises an error. Failed
/// anchor solves are reported in `statuses` and omitted from the front.
pub fn generate_fpbi_front(
    objective: &VectorFunction<'_>,
    bounds: &Matrix,
    initial_conditions: &Matrix,
    focus: Option<&[f64]>,
    inequalities: Option<&VectorFunction<'_>>,
    equalities: Option<&VectorFunction<'_>>,
    options: &FpbiOptions,
) -> Result<FpbiResult> {
    let d = bounds.ncols();
    if bounds.nrows() != 2 || d == 0 || initial_conditions.ncols() != d {
        return Err(BotorchError::DimensionMismatch(
            "bounds must be 2 x d and initial_conditions must be restarts x d".into(),
        ));
    }
    if (0..d).any(|j| {
        !bounds[(0, j)].is_finite()
            || !bounds[(1, j)].is_finite()
            || bounds[(1, j)] <= bounds[(0, j)]
            || !(bounds[(1, j)] - bounds[(0, j)]).is_finite()
    }) {
        return Err(invalid("bounds must be finite and have upper > lower"));
    }
    if initial_conditions.iter().any(|v| !v.is_finite())
        || (0..initial_conditions.nrows()).any(|i| {
            (0..d).any(|j| {
                initial_conditions[(i, j)] < bounds[(0, j)]
                    || initial_conditions[(i, j)] > bounds[(1, j)]
            })
        })
    {
        return Err(invalid(
            "initial conditions must be finite and within bounds",
        ));
    }
    // Optimize unit-box decisions so finite differences and local tolerances
    // do not depend on a decision variable's physical units.
    let physical = |u: &[f64]| -> Vec<f64> {
        (0..d)
            .map(|j| bounds[(0, j)] + u[j] * (bounds[(1, j)] - bounds[(0, j)]))
            .collect()
    };
    let normalized = Matrix::from_fn(initial_conditions.nrows(), d, |i, j| {
        (initial_conditions[(i, j)] - bounds[(0, j)]) / (bounds[(1, j)] - bounds[(0, j)])
    });
    let unit_bounds = Matrix::from_fn(2, d, |i, _| i as f64);
    let f = |u: &[f64]| objective(&physical(u));
    let gi = |u: &[f64]| match inequalities {
        Some(g) => g(&physical(u)),
        None => Ok(vec![]),
    };
    let ge = |u: &[f64]| match equalities {
        Some(g) => g(&physical(u)),
        None => Ok(vec![]),
    };
    let mut result = generate_fpbi_front_unit(
        &f,
        &unit_bounds,
        &normalized,
        focus,
        inequalities.map(|_| &gi as &VectorFunction<'_>),
        equalities.map(|_| &ge as &VectorFunction<'_>),
        options,
    )?;
    for i in 0..result.x.nrows() {
        for j in 0..d {
            result.x[(i, j)] =
                bounds[(0, j)] + result.x[(i, j)] * (bounds[(1, j)] - bounds[(0, j)]);
        }
    }
    for i in 0..result.extreme_x.nrows() {
        for j in 0..d {
            result.extreme_x[(i, j)] =
                bounds[(0, j)] + result.extreme_x[(i, j)] * (bounds[(1, j)] - bounds[(0, j)]);
        }
    }
    Ok(result)
}

fn generate_fpbi_front_unit(
    objective: &VectorFunction<'_>,
    bounds: &Matrix,
    initial_conditions: &Matrix,
    focus: Option<&[f64]>,
    inequalities: Option<&VectorFunction<'_>>,
    equalities: Option<&VectorFunction<'_>>,
    options: &FpbiOptions,
) -> Result<FpbiResult> {
    let d = bounds.ncols();
    if bounds.nrows() != 2
        || d == 0
        || initial_conditions.nrows() == 0
        || initial_conditions.ncols() != d
    {
        return Err(BotorchError::DimensionMismatch(
            "bounds must be 2 x d and initial_conditions must be nonempty restarts x d".into(),
        ));
    }
    if options.maxiter == 0
        || !options.ftol.is_finite()
        || options.ftol <= 0.0
        || !options.feasibility_tolerance.is_finite()
        || options.feasibility_tolerance < 0.0
        || !options.finite_difference_step.is_finite()
        || options.finite_difference_step <= 0.0
    {
        return Err(invalid("solver iterations, accuracy and finite difference step must be positive; feasibility tolerance must be nonnegative"));
    }
    let lower: Vec<_> = (0..d).map(|j| bounds[(0, j)]).collect();
    let upper: Vec<_> = (0..d).map(|j| bounds[(1, j)]).collect();
    if lower
        .iter()
        .zip(&upper)
        .any(|(a, b)| !a.is_finite() || !b.is_finite() || a > b)
    {
        return Err(invalid("decision bounds must be finite and ordered"));
    }
    let starts: Vec<Vec<_>> = (0..initial_conditions.nrows())
        .map(|i| initial_conditions.row(i).iter().copied().collect())
        .collect();
    if starts.iter().any(|x| {
        x.iter()
            .enumerate()
            .any(|(j, v)| !v.is_finite() || *v < lower[j] || *v > upper[j])
    }) {
        return Err(invalid(
            "initial conditions must be finite and within bounds",
        ));
    }
    let first = objective(&starts[0])?;
    let m = first.len();
    if m < 2 {
        return Err(invalid("FPBI needs at least two objectives"));
    }
    if first.iter().any(|v| !v.is_finite()) {
        return Err(BotorchError::Numerical(
            "objective returned a non-finite value".into(),
        ));
    }
    // Dimension is now known: reject invalid grids before constraint
    // callbacks or the potentially expensive individual-objective solves.
    grid_point_count(m, options)?;
    let ni = match inequalities {
        Some(f) => f(&starts[0])?.len(),
        None => 0,
    };
    let ne = match equalities {
        Some(f) => f(&starts[0])?.len(),
        None => 0,
    };
    let original_constraints = |x: &[f64]| -> Result<Vec<f64>> {
        let mut c = match equalities {
            Some(f) => checked_values(f, x, ne, "equalities")?,
            None => vec![],
        };
        if let Some(f) = inequalities {
            c.extend(checked_values(f, x, ni, "inequalities")?);
        }
        Ok(c)
    };
    let sign = if options.maximize { -1.0 } else { 1.0 };
    let mut extreme_x = Matrix::zeros(m, d);
    let mut extreme_y = Matrix::zeros(m, m);
    let mut statuses = Vec::new();
    for j in 0..m {
        let scalar = |x: &[f64]| -> Result<ScalarEvaluation> {
            Ok((
                sign * checked_values(objective, x, m, "objective")?[j],
                original_constraints(x)?,
            ))
        };
        let (result, violation) =
            solve_scalar(&scalar, &starts, &lower, &upper, ne, ne + ni, options)?;
        if !result.success() || violation > options.feasibility_tolerance {
            return Err(BotorchError::CandidateGeneration(format!(
                "FPBI extreme objective {j} failed: {} (violation {violation})",
                result.message()
            )));
        }
        let y = checked_values(objective, &result.x, m, "objective")?;
        for k in 0..d {
            extreme_x[(j, k)] = result.x[k];
        }
        for k in 0..m {
            extreme_y[(j, k)] = y[k];
        }
        statuses.push(FpbiSolveStatus {
            index: j,
            extreme: true,
            mode: result.mode,
            message: result.message().into(),
            max_violation: violation,
            feasible: true,
            success: true,
            iterations: result.iterations,
            restarts: starts.len(),
            kappa: None,
        });
    }
    let geometry = fpbi_geometry(&extreme_y, focus, options)?;
    let normalize = |y: &[f64]| -> Vec<f64> {
        (0..m)
            .map(|j| {
                if geometry.ranges[j] > 0.0 {
                    let hi = geometry.nadir[j];
                    let scale = geometry.utopia[j].abs().max(hi.abs()).max(1.0);
                    let lo_s = geometry.utopia[j] / scale;
                    (sign * y[j] / scale - lo_s) / (hi / scale - lo_s)
                } else {
                    0.0
                }
            })
            .collect()
    };
    let mut zlower = lower.clone();
    zlower.push(f64::NEG_INFINITY);
    let mut zupper = upper.clone();
    zupper.push(f64::INFINITY);
    let mut front_x: Vec<Vec<_>> = (0..m)
        .map(|i| extreme_x.row(i).iter().copied().collect())
        .collect();
    let mut front_y: Vec<Vec<_>> = (0..m)
        .map(|i| extreme_y.row(i).iter().copied().collect())
        .collect();
    let mut anchor_indices = vec![None; m];
    if !geometry.degenerate && !geometry.search_direction.iter().any(|v| *v < 0.0) {
        return Err(invalid(
            "focus - center must have a negative component; otherwise kappa is unbounded",
        ));
    }
    let mut ray_starts: Vec<Vec<_>> = (0..m)
        .map(|i| extreme_x.row(i).iter().copied().collect())
        .collect();
    ray_starts.extend(starts.clone());
    for i in 0..geometry.anchors.nrows() {
        if geometry.degenerate {
            break;
        }
        // Flat axes pad the grid with duplicate anchors. Solve each ray once.
        if let Some(previous) = (0..i).find(|&previous| {
            (0..m).all(|j| geometry.anchors[(previous, j)] == geometry.anchors[(i, j)])
        }) {
            if let Some(status) = statuses.iter().find(|s| !s.extreme && s.index == previous) {
                let mut reused = status.clone();
                reused.index = i;
                reused.iterations = 0;
                reused.restarts = 0;
                reused.message = format!("Reused identical anchor {previous}: {}", reused.message);
                statuses.push(reused);
            }
            continue;
        }
        let scalar = |z: &[f64]| -> Result<ScalarEvaluation> {
            let y = normalize(&checked_values(objective, &z[..d], m, "objective")?);
            let mut c = original_constraints(&z[..d])?;
            c.extend(
                (0..m)
                    .map(|j| geometry.anchors[(i, j)] + z[d] * geometry.search_direction[j] - y[j]),
            );
            Ok((-z[d], c))
        };
        let mut zstarts = Vec::new();
        for x in &ray_starts {
            let y = normalize(&checked_values(objective, x, m, "objective")?);
            let kappa = (0..m)
                .filter(|&j| geometry.search_direction[j] < 0.0)
                .map(|j| (geometry.anchors[(i, j)] - y[j]) / (-geometry.search_direction[j]))
                .fold(f64::INFINITY, f64::min);
            let mut z = x.clone();
            z.push(kappa.min(0.0));
            zstarts.push(z);
        }
        let (result, violation) = solve_scalar(
            &scalar,
            &zstarts,
            &zlower,
            &zupper,
            ne,
            ne + ni + m,
            options,
        )?;
        let feasible = violation <= options.feasibility_tolerance;
        let success = result.success() && feasible;
        if success {
            let x = result.x[..d].to_vec();
            let y = checked_values(objective, &x, m, "objective")?;
            front_x.push(x);
            front_y.push(y);
            anchor_indices.push(Some(i));
        }
        statuses.push(FpbiSolveStatus {
            index: i,
            extreme: false,
            mode: result.mode,
            message: result.message().into(),
            max_violation: violation,
            feasible,
            success,
            iterations: result.iterations,
            restarts: ray_starts.len(),
            kappa: Some(result.x[d]),
        });
    }
    let all_y = Matrix::from_fn(front_y.len(), m, |i, j| front_y[i][j]);
    let nondominated = is_non_dominated(&all_y, options.maximize, true);
    let mut kept = Vec::<usize>::new();
    for (i, &keep) in nondominated.iter().enumerate() {
        if !keep {
            continue;
        }
        let zi = normalize(&front_y[i]);
        if !kept.iter().any(|&k| {
            zi.iter()
                .zip(normalize(&front_y[k]))
                .all(|(a, b)| (a - b).abs() <= options.feasibility_tolerance)
        }) {
            kept.push(i);
        }
    }
    Ok(FpbiResult {
        x: Matrix::from_fn(kept.len(), d, |i, j| front_x[kept[i]][j]),
        y: Matrix::from_fn(kept.len(), m, |i, j| front_y[kept[i]][j]),
        anchor_indices: kept.iter().map(|&i| anchor_indices[i]).collect(),
        extreme_x,
        extreme_y,
        geometry,
        statuses,
    })
}
