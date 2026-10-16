//! Optimization over discrete sets (`botorch.optim.optimize`):
//! [`optimize_acqf_discrete`] (enumeration of a set of choices) and
//! [`optimize_acqf_discrete_local_search`] (local search on a lattice). For
//! `q > 1` both generate the candidates sequentially, conditioning on the
//! previous ones as pending points.

use rand::{Rng, RngExt};
use rayon::prelude::*;

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::generation::gen::torch_argmax;
use crate::optim::parameter_constraints::{ConstraintIndices, LinearConstraint};
use crate::Matrix;

/// The result of the discrete optimizers.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteResult {
    /// The candidates (`q' × d`, `q' <= q` if fewer unique choices remain).
    pub candidates: Matrix,
    /// The acquisition values: for [`optimize_acqf_discrete`] the value of
    /// every candidate conditional on the previous ones (one value for
    /// `q = 1`); for [`optimize_acqf_discrete_local_search`] the joint value
    /// of the candidates.
    pub acq_values: Vec<f64>,
    /// Messages of BoTorch's warnings (`OptimizationWarning`).
    pub warnings: Vec<String>,
}

/// The rows of a matrix as vectors.
fn rows(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| m.row(i).iter().copied().collect())
        .collect()
}

fn from_rows(rows: &[Vec<f64>], d: usize) -> Matrix {
    Matrix::from_fn(rows.len(), d, |i, j| rows[i][j])
}

/// Checks that the constraints are intra-point constraints on `d` features.
fn check_intra_point(constraints: &[LinearConstraint], d: usize) -> Result<()> {
    for c in constraints {
        match c.indices() {
            ConstraintIndices::IntraPoint(idx) => {
                if idx.iter().any(|&k| k >= d) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "Constraint indices cannot exceed the problem dimension d={d}."
                    )));
                }
            }
            ConstraintIndices::InterPoint(_) => {
                return Err(BotorchError::Unsupported(
                    "Inter-point constraints are not supported for discrete optimization.".into(),
                ))
            }
        }
    }
    Ok(())
}

/// Whether `x` satisfies the inequality constraints exactly (`Σ c x >= rhs`,
/// BoTorch's `_filter_infeasible`).
fn is_feasible(x: &[f64], constraints: &[LinearConstraint]) -> bool {
    constraints.iter().all(|c| match c.indices() {
        ConstraintIndices::IntraPoint(idx) => {
            let s: f64 = idx
                .iter()
                .zip(c.coefficients())
                .map(|(&k, &w)| x[k] * w)
                .sum();
            s >= c.rhs()
        }
        ConstraintIndices::InterPoint(_) => false,
    })
}

/// Whether `x` equals a row of `avoid` (BoTorch's `_filter_invalid`).
fn is_avoided(x: &[f64], avoid: &[Vec<f64>]) -> bool {
    avoid.iter().any(|a| a.as_slice() == x)
}

/// Values of the single points `xs` (each evaluated as a `1 × d` set), in
/// parallel.
fn point_values(acq_function: &dyn AcquisitionFunction, xs: &[Vec<f64>]) -> Result<Vec<f64>> {
    let d = acq_function.dim();
    xs.par_iter()
        .map(|x| acq_function.forward(&Matrix::from_row_slice(1, d, x)))
        .collect()
}

/// Appends `rows` to the base pending points.
fn pending_with(base: Option<&Matrix>, rows: &[Vec<f64>], d: usize) -> Matrix {
    let n0 = base.map_or(0, |b| b.nrows());
    Matrix::from_fn(n0 + rows.len(), d, |i, j| {
        if i < n0 {
            base.expect("base rows")[(i, j)]
        } else {
            rows[i - n0][j]
        }
    })
}

/// Optimizes the acquisition function over the discrete set `choices`
/// (`n × d`) by evaluating every choice (BoTorch's `optimize_acqf_discrete`):
///
/// * with `unique`, the rows of `x_avoid` are removed from the choices;
///   infeasible choices (`inequality_constraints`, intra-point) are removed;
/// * for `q > 1`, the candidates are chosen sequentially: the best choice is
///   added to the pending points (and, with `unique`, removed from the
///   choices) before the next one is chosen; if fewer than `q` unique choices
///   remain, `q` is reduced (with BoTorch's warning).
///
/// The choices are evaluated in parallel (BoTorch's `max_batch_size` only
/// bounds memory and does not change the result). `x_avoid_given` and
/// `constraints_given` select BoTorch's error message for an empty set.
///
/// # Errors
/// [`BotorchError::Unsupported`] for one-shot acquisition functions and
/// inter-point constraints, [`BotorchError::InputData`] if no choice remains,
/// [`BotorchError::DimensionMismatch`] for inconsistent shapes, and errors of
/// the acquisition function (including acquisition functions without pending
/// points for `q > 1`).
#[allow(clippy::too_many_arguments)]
pub fn optimize_acqf_discrete(
    acq_function: &dyn AcquisitionFunction,
    q: usize,
    choices: &Matrix,
    unique: bool,
    x_avoid: Option<&Matrix>,
    inequality_constraints: Option<&[LinearConstraint]>,
) -> Result<DiscreteResult> {
    if acq_function.as_one_shot().is_some() {
        return Err(BotorchError::Unsupported(
            "Discrete optimization is not supported for one-shot acquisition functions.".into(),
        ));
    }
    let d = acq_function.dim();
    if choices.ncols() != d {
        return Err(BotorchError::DimensionMismatch(format!(
            "choices must have d = {d} columns, got {}",
            choices.ncols()
        )));
    }
    if q == 0 {
        return Err(BotorchError::InvalidArgument("q must be positive".into()));
    }
    let mut choice_rows = rows(choices);
    if let (Some(avoid), true) = (x_avoid, unique) {
        if avoid.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "X_avoid must have d = {d} columns, got {}",
                avoid.ncols()
            )));
        }
        let avoid = rows(avoid);
        choice_rows.retain(|x| !is_avoided(x, &avoid));
    }
    if let Some(cs) = inequality_constraints {
        check_intra_point(cs, d)?;
        choice_rows.retain(|x| is_feasible(x, cs));
    }
    let mut warnings = Vec::new();
    let mut q = q;
    if choice_rows.is_empty() {
        let mut message = "`choices` must be non-empty.".to_string();
        if x_avoid.is_some() || inequality_constraints.is_some() {
            message.push_str(
                " No feasible points remain after removing `X_avoid` and filtering out \
                 infeasible points.",
            );
        }
        return Err(BotorchError::InputData(message));
    } else if choice_rows.len() < q && unique {
        warnings.push(format!(
            "Requested q={q} candidates from fully discrete search space, but only {} \
             possible choices remain. ",
            choice_rows.len()
        ));
        q = choice_rows.len();
    }
    if q == 1 {
        let values = point_values(acq_function, &choice_rows)?;
        let best = torch_argmax(&values);
        return Ok(DiscreteResult {
            candidates: from_rows(&choice_rows[best..=best], d),
            acq_values: vec![values[best]],
            warnings,
        });
    }
    let base = acq_function.x_pending();
    let mut selected: Vec<Vec<f64>> = Vec::with_capacity(q);
    let mut values_out = Vec::with_capacity(q);
    let result = (|| -> Result<()> {
        for _ in 0..q {
            let values = point_values(acq_function, &choice_rows)?;
            let best = torch_argmax(&values);
            selected.push(choice_rows[best].clone());
            values_out.push(values[best]);
            acq_function.set_x_pending(Some(&pending_with(base.as_ref(), &selected, d)))?;
            if unique {
                choice_rows.remove(best);
            }
        }
        Ok(())
    })();
    acq_function.set_x_pending(base.as_ref())?;
    result?;
    Ok(DiscreteResult {
        candidates: from_rows(&selected, d),
        acq_values: values_out,
        warnings,
    })
}

/// Settings of [`optimize_acqf_discrete_local_search`] (BoTorch's keyword
/// arguments).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalSearchOptions {
    /// Number of local searches per candidate (20).
    pub num_restarts: usize,
    /// Number of random lattice points drawn per try (4096).
    pub raw_samples: usize,
    /// Maximum number of tries to find `num_restarts` valid starting points
    /// (100).
    pub max_tries: usize,
    /// Exclude the previous candidates from later local searches (true).
    pub unique: bool,
}

impl Default for LocalSearchOptions {
    fn default() -> Self {
        Self {
            num_restarts: 20,
            raw_samples: 4096,
            max_tries: 100,
            unique: true,
        }
    }
}

/// The function type of a generator of lattice starting points (BoTorch's
/// `_gen_batch_initial_conditions_local_search`): given the points to
/// avoid, it returns the valid lattice points (`n × d`) from which the
/// `num_restarts` best are taken.
pub type LatticePointsFn<'a> = dyn Fn(&Matrix) -> Result<Matrix> + Sync + 'a;

/// BoTorch's `_gen_batch_initial_conditions_local_search`: random lattice
/// points (`raw_samples` per try, every coordinate drawn uniformly from its
/// choices), deduplicated and sorted (`torch.unique(dim=0)`), without the
/// rows of `x_avoid` and infeasible points, until at least `min_points`
/// remain.
///
/// # Errors
/// [`BotorchError::Runtime`] ("Failed to generate at least {min_points}
/// initial conditions") after `max_tries` tries.
pub fn gen_lattice_points<R: Rng + ?Sized>(
    discrete_choices: &[Vec<f64>],
    raw_samples: usize,
    x_avoid: &Matrix,
    inequality_constraints: &[LinearConstraint],
    min_points: usize,
    max_tries: usize,
    rng: &mut R,
) -> Result<Matrix> {
    let dim = discrete_choices.len();
    let avoid = rows(x_avoid);
    let mut x: Vec<Vec<f64>> = Vec::new();
    for _ in 0..max_tries {
        for _ in 0..raw_samples {
            x.push(
                discrete_choices
                    .iter()
                    .map(|c| c[rng.random_range(0..c.len())])
                    .collect(),
            );
        }
        x.sort_by(|a, b| lex_cmp(a, b));
        x.dedup();
        x.retain(|p| !is_avoided(p, &avoid) && is_feasible(p, inequality_constraints));
        if x.len() >= min_points {
            return Ok(from_rows(&x, dim));
        }
    }
    Err(BotorchError::Runtime(format!(
        "Failed to generate at least {min_points} initial conditions"
    )))
}

/// Lexicographic order of rows (`torch.unique(dim=0)`; NaN sorts last).
pub(crate) fn lex_cmp(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b) {
        match x.partial_cmp(y) {
            Some(std::cmp::Ordering::Equal) => continue,
            Some(o) => return o,
            None => {
                if x.is_nan() && !y.is_nan() {
                    return std::cmp::Ordering::Greater;
                }
                if y.is_nan() && !x.is_nan() {
                    return std::cmp::Ordering::Less;
                }
            }
        }
    }
    std::cmp::Ordering::Equal
}

/// The feasible lattice neighbors of `x` that differ in one coordinate
/// (BoTorch's `_generate_neighbors`), in the order of the dimensions and
/// choices, without `x` and the rows of `avoid`.
fn neighbors(
    x: &[f64],
    discrete_choices: &[Vec<f64>],
    avoid: &[Vec<f64>],
    constraints: &[LinearConstraint],
) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for (i, c) in discrete_choices.iter().enumerate() {
        for &v in c {
            let mut y = x.to_vec();
            y[i] = v;
            if y.as_slice() != x && !is_avoided(&y, avoid) && is_feasible(&y, constraints) {
                out.push(y);
            }
        }
    }
    out
}

/// Hill climbing from `x0`: moves to the best neighbor while it improves the
/// acquisition value (BoTorch's inner loop of
/// `optimize_acqf_discrete_local_search`).
fn local_search(
    acq_function: &dyn AcquisitionFunction,
    x0: &[f64],
    discrete_choices: &[Vec<f64>],
    avoid: &[Vec<f64>],
    constraints: &[LinearConstraint],
) -> Result<(Vec<f64>, f64)> {
    let d = x0.len();
    let mut curr = x0.to_vec();
    let mut curr_val = acq_function.forward(&Matrix::from_row_slice(1, d, &curr))?;
    loop {
        let loc = neighbors(&curr, discrete_choices, avoid, constraints);
        if loc.is_empty() {
            break;
        }
        let values = point_values(acq_function, &loc)?;
        let best = torch_argmax(&values);
        // `acqval_loc.max() <= curr_acqval` (a NaN maximum ends the search
        // here, where BoTorch would move to it and could cycle).
        if values[best].is_nan() || values[best] <= curr_val {
            break;
        }
        curr = loc[best].clone();
        curr_val = values[best];
    }
    Ok((curr, curr_val))
}

/// Optimizes the acquisition function over the lattice `discrete_choices[0]
/// × ... × discrete_choices[d-1]` by local search (BoTorch's
/// `optimize_acqf_discrete_local_search`): for each of the `q` candidates,
/// `num_restarts` starting points are taken (the given
/// `batch_initial_conditions` (`n × d`) for the first candidate, the rest the
/// best of the random lattice points of `lattice_points`, by default
/// [`gen_lattice_points`]), and from each a hill climb over the
/// one-coordinate neighbors (excluding `x_avoid`, and with `unique` the
/// previous candidates, and infeasible points) finds a local optimum; the
/// best one is the candidate. Later candidates are conditioned on the
/// previous ones as pending points. The local searches run in parallel.
///
/// Returns the candidates and their joint acquisition value. Unlike BoTorch,
/// the pending points of the acquisition function are restored also when the
/// joint value is not needed.
///
/// # Errors
/// [`BotorchError::Runtime`] if no starting points can be generated,
/// [`BotorchError::InvalidArgument`] / [`BotorchError::DimensionMismatch`]
/// for invalid arguments, and errors of the acquisition function.
#[allow(clippy::too_many_arguments)]
pub fn optimize_acqf_discrete_local_search<R: Rng + ?Sized>(
    acq_function: &dyn AcquisitionFunction,
    discrete_choices: &[Vec<f64>],
    q: usize,
    options: &LocalSearchOptions,
    inequality_constraints: &[LinearConstraint],
    x_avoid: Option<&Matrix>,
    batch_initial_conditions: Option<&Matrix>,
    lattice_points: Option<&LatticePointsFn<'_>>,
    return_acq_values: bool,
    rng: &mut R,
) -> Result<DiscreteResult> {
    let dim = discrete_choices.len();
    if dim != acq_function.dim() {
        return Err(BotorchError::DimensionMismatch(format!(
            "discrete_choices has {dim} dimensions but the acquisition function has d = {}",
            acq_function.dim()
        )));
    }
    if discrete_choices.iter().any(|c| c.is_empty()) {
        return Err(BotorchError::InvalidArgument(
            "every dimension needs at least one choice".into(),
        ));
    }
    check_intra_point(inequality_constraints, dim)?;
    if let Some(a) = x_avoid {
        if a.ncols() != dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "X_avoid must have d = {dim} columns, got {}",
                a.ncols()
            )));
        }
    }
    if let Some(b) = batch_initial_conditions {
        if b.ncols() != dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "batch_initial_conditions must be n x 1 x {dim}, got {} columns",
                b.ncols()
            )));
        }
    }
    let base_pending = if q > 1 {
        acq_function.x_pending()
    } else {
        None
    };
    let base_avoid: Option<Vec<Vec<f64>>> = x_avoid.map(rows);
    let mut avoid: Vec<Vec<f64>> = base_avoid.clone().unwrap_or_default();
    let mut bic: Option<Vec<Vec<f64>>> = batch_initial_conditions.map(rows);
    let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(q);
    let result =
        (|| -> Result<()> {
            for _ in 0..q {
                // Starting points (BoTorch's `_gen_starting_points_local_search`).
                let mut required = options.num_restarts as isize;
                let mut x0: Vec<Vec<f64>> = Vec::new();
                let had_bic = bic.is_some();
                if let Some(b) = bic.take() {
                    required -= b.len() as isize;
                    x0.extend(b.into_iter().filter(|x| {
                        !is_avoided(x, &avoid) && is_feasible(x, inequality_constraints)
                    }));
                }
                if required > 0 {
                    let avoid_m = from_rows(&avoid, dim);
                    let generated = match lattice_points {
                        Some(f) => f(&avoid_m)?,
                        None => gen_lattice_points(
                            discrete_choices,
                            options.raw_samples,
                            &avoid_m,
                            inequality_constraints,
                            options.num_restarts,
                            options.max_tries,
                            rng,
                        )?,
                    };
                    if generated.ncols() != dim || generated.nrows() < options.num_restarts {
                        return Err(BotorchError::DimensionMismatch(format!(
                            "the lattice points must be n x {dim} with n >= {}, got {}x{}",
                            options.num_restarts,
                            generated.nrows(),
                            generated.ncols()
                        )));
                    }
                    let generated = rows(&generated);
                    let values = point_values(acq_function, &generated)?;
                    let top = crate::optim::initializers::topk_indices(
                        &values,
                        options.num_restarts,
                        true,
                        true,
                    );
                    x0.extend(top.into_iter().map(|i| generated[i].clone()));
                }
                if !had_bic && required <= 0 {
                    return Err(BotorchError::InvalidArgument(
                        "Either `batch_initial_conditions` or `raw_samples` must be set.".into(),
                    ));
                }
                let results = x0
                    .par_iter()
                    .map(|x| {
                        local_search(
                            acq_function,
                            x,
                            discrete_choices,
                            &avoid,
                            inequality_constraints,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                if results.is_empty() {
                    return Err(BotorchError::Runtime(
                        "no valid starting points for the local search".into(),
                    ));
                }
                let values: Vec<f64> = results.iter().map(|r| r.1).collect();
                let best = torch_argmax(&values);
                candidates.push(results[best].0.clone());
                if q > 1 {
                    acq_function.set_x_pending(Some(&pending_with(
                        base_pending.as_ref(),
                        &candidates,
                        dim,
                    )))?;
                    if options.unique {
                        avoid = base_avoid.clone().unwrap_or_default();
                        avoid.extend(candidates.iter().cloned());
                    }
                }
            }
            Ok(())
        })();
    if q > 1 {
        acq_function.set_x_pending(base_pending.as_ref())?;
    }
    result?;
    let candidates = from_rows(&candidates, dim);
    let acq_values = if return_acq_values {
        vec![acq_function.forward(&candidates)?]
    } else {
        Vec::new()
    };
    Ok(DiscreteResult {
        candidates,
        acq_values,
        warnings: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbors_and_lex_order() {
        let choices = vec![vec![0.0, 1.0], vec![0.0, 0.5, 1.0]];
        let n = neighbors(&[0.0, 0.5], &choices, &[], &[]);
        assert_eq!(n, vec![vec![1.0, 0.5], vec![0.0, 0.0], vec![0.0, 1.0]]);
        let n = neighbors(&[0.0, 0.5], &choices, &[vec![0.0, 0.0]], &[]);
        assert_eq!(n.len(), 2);
        let mut v = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.5],
            vec![1.0, 0.0],
        ];
        v.sort_by(|a, b| lex_cmp(a, b));
        v.dedup();
        assert_eq!(v, vec![vec![0.0, 0.5], vec![0.0, 1.0], vec![1.0, 0.0]]);
    }
}
