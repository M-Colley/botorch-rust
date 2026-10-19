//! Parameter constraints of acquisition optimization
//! (`botorch.optim.parameter_constraints`): linear inequality and equality
//! constraints ([`LinearConstraint`], BoTorch's `(indices, coefficients,
//! rhs)` tuples, intra- or inter-point), nonlinear inequality constraints
//! ([`NonlinearInequalityConstraint`], BoTorch's `(callable, is_intrapoint)`
//! tuples), feasibility checks ([`evaluate_feasibility`]) and the reduction
//! of constraints to the features that are not fixed.
//!
//! Conventions (BoTorch's): a linear inequality constraint is
//! `Σ_i coefficients[i] · X[indices[i]] ≥ rhs`, an equality constraint the
//! same with `=`, and a nonlinear constraint `g(x) ≥ 0`.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::Matrix;

/// Feasibility tolerance of BoTorch's `get_constraint_tolerance` for double
/// precision: inequalities may be violated by up to this amount, equalities
/// must hold to within it.
pub const CONSTRAINT_TOLERANCE: f64 = 1e-8;

/// The entries of a candidate set (`q × d`) that a linear constraint involves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintIndices {
    /// Features of every point (BoTorch's one-dimensional `indices`): the
    /// constraint applies to each point of the q-batch separately.
    IntraPoint(Vec<usize>),
    /// `(point, feature)` pairs (BoTorch's two-dimensional `indices`): one
    /// constraint across the points of the q-batch.
    InterPoint(Vec<(usize, usize)>),
}

/// A linear constraint `Σ_i coefficients[i] · X[indices[i]] ≥ rhs` (or `=`
/// for equality constraints), BoTorch's `(indices, coefficients, rhs)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearConstraint {
    indices: ConstraintIndices,
    coefficients: Vec<f64>,
    rhs: f64,
}

impl LinearConstraint {
    /// A constraint on the features `indices` of every point.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for empty or mismatched indices and
    /// coefficients or non-finite values.
    pub fn intra_point(indices: Vec<usize>, coefficients: Vec<f64>, rhs: f64) -> Result<Self> {
        Self::new(ConstraintIndices::IntraPoint(indices), coefficients, rhs)
    }

    /// A constraint across the points of the q-batch on the entries
    /// `(point, feature)`.
    ///
    /// # Errors
    /// See [`LinearConstraint::intra_point`].
    pub fn inter_point(
        indices: Vec<(usize, usize)>,
        coefficients: Vec<f64>,
        rhs: f64,
    ) -> Result<Self> {
        Self::new(ConstraintIndices::InterPoint(indices), coefficients, rhs)
    }

    /// A constraint from its parts.
    ///
    /// # Errors
    /// See [`LinearConstraint::intra_point`].
    pub fn new(indices: ConstraintIndices, coefficients: Vec<f64>, rhs: f64) -> Result<Self> {
        let k = match &indices {
            ConstraintIndices::IntraPoint(i) => i.len(),
            ConstraintIndices::InterPoint(i) => i.len(),
        };
        if k == 0 {
            return Err(BotorchError::InvalidArgument(
                "`indices` must be at least one-dimensional".into(),
            ));
        }
        if coefficients.len() != k {
            return Err(BotorchError::InvalidArgument(format!(
                "a linear constraint has {k} indices but {} coefficients",
                coefficients.len()
            )));
        }
        if coefficients.iter().any(|c| !c.is_finite()) || !rhs.is_finite() {
            return Err(BotorchError::InvalidArgument(
                "linear constraint coefficients and right-hand sides must be finite".into(),
            ));
        }
        Ok(Self {
            indices,
            coefficients,
            rhs,
        })
    }

    /// The indices.
    pub fn indices(&self) -> &ConstraintIndices {
        &self.indices
    }

    /// The coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The right-hand side.
    pub fn rhs(&self) -> f64 {
        self.rhs
    }

    /// Whether the constraint couples the points of the q-batch.
    pub fn is_inter_point(&self) -> bool {
        matches!(self.indices, ConstraintIndices::InterPoint(_))
    }

    /// Checks the indices against `q` points of dimension `d` (BoTorch's
    /// errors of `transform_constraints` and `make_scipy_linear_constraints`).
    pub(crate) fn validate(&self, q: usize, d: usize) -> Result<()> {
        match &self.indices {
            ConstraintIndices::IntraPoint(idx) => {
                if idx.iter().any(|&k| k >= d) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "Constraint indices cannot exceed the problem dimension d={d}."
                    )));
                }
            }
            ConstraintIndices::InterPoint(idx) => {
                if idx.iter().any(|&(_, k)| k >= d) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "Constraint indices cannot exceed the problem dimension d={d}."
                    )));
                }
                if idx.iter().any(|&(p, _)| p >= q) {
                    return Err(BotorchError::Runtime(format!(
                        "Index out of bounds for {q}-batch"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The constraint as sparse rows over the flattened candidate set
    /// (index `p·d + k`): one row per point for intra-point constraints, one
    /// row for inter-point constraints. Duplicate entries are summed and the
    /// entries sorted by column (the dense rows SciPy evaluates).
    pub(crate) fn rows(&self, q: usize, d: usize) -> Vec<Vec<(usize, f64)>> {
        let mut rows = Vec::new();
        let mut push = |entries: Vec<(usize, f64)>| {
            let mut dense: BTreeMap<usize, f64> = BTreeMap::new();
            for (col, c) in entries {
                *dense.entry(col).or_insert(0.0) += c;
            }
            rows.push(dense.into_iter().collect::<Vec<_>>());
        };
        match &self.indices {
            ConstraintIndices::IntraPoint(idx) => {
                for p in 0..q {
                    push(
                        idx.iter()
                            .zip(&self.coefficients)
                            .map(|(&k, &c)| (p * d + k, c))
                            .collect(),
                    );
                }
            }
            ConstraintIndices::InterPoint(idx) => push(
                idx.iter()
                    .zip(&self.coefficients)
                    .map(|(&(p, k), &c)| (p * d + k, c))
                    .collect(),
            ),
        }
        rows
    }

    /// `Σ c · x` for every row of [`LinearConstraint::rows`] at the candidate
    /// set `x` (`q × d`).
    pub(crate) fn values(&self, x: &Matrix) -> Vec<f64> {
        let d = x.ncols();
        self.rows(x.nrows(), d)
            .iter()
            .map(|row| row.iter().map(|&(col, c)| c * x[(col / d, col % d)]).sum())
            .collect()
    }
}

/// A nonlinear inequality constraint `g(x) ≥ 0` (BoTorch's `(callable,
/// is_intrapoint)` tuples).
pub trait NonlinearInequalityConstraint: Send + Sync + Debug {
    /// Whether the constraint applies to every point separately (`true`) or
    /// to the whole q-batch.
    fn is_intrapoint(&self) -> bool;

    /// The value `g(x)` and its gradient (same length as `x`) at one point
    /// (`d` values) for intra-point constraints, or at the q-batch (`q·d`
    /// values, row-major) for inter-point constraints.
    ///
    /// # Errors
    /// Implementation-specific.
    fn evaluate(&self, x: &[f64]) -> Result<(f64, Vec<f64>)>;
}

/// The value and gradient of a nonlinear constraint function.
pub type ConstraintFn = dyn Fn(&[f64]) -> (f64, Vec<f64>) + Send + Sync;

/// A nonlinear inequality constraint given by a closure returning the value
/// and its gradient.
#[derive(Clone)]
pub struct FnNonlinearConstraint {
    f: Arc<ConstraintFn>,
    intrapoint: bool,
}

impl Debug for FnNonlinearConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnNonlinearConstraint")
            .field("intrapoint", &self.intrapoint)
            .finish()
    }
}

impl FnNonlinearConstraint {
    /// An intra-point constraint `f(x) ≥ 0` on every point (`x` of length `d`).
    pub fn intrapoint<F>(f: F) -> Self
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>) + Send + Sync + 'static,
    {
        Self {
            f: Arc::new(f),
            intrapoint: true,
        }
    }

    /// An inter-point constraint `f(X) ≥ 0` on the q-batch (`X` flattened
    /// row-major, length `q·d`).
    pub fn interpoint<F>(f: F) -> Self
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>) + Send + Sync + 'static,
    {
        Self {
            f: Arc::new(f),
            intrapoint: false,
        }
    }
}

impl NonlinearInequalityConstraint for FnNonlinearConstraint {
    fn is_intrapoint(&self) -> bool {
        self.intrapoint
    }

    fn evaluate(&self, x: &[f64]) -> Result<(f64, Vec<f64>)> {
        Ok((self.f)(x))
    }
}

/// The parameter constraints of an acquisition optimization (BoTorch's
/// `inequality_constraints`, `equality_constraints` and
/// `nonlinear_inequality_constraints`).
#[derive(Clone, Debug, Default)]
pub struct ParameterConstraints {
    /// Linear inequality constraints `Σ c x ≥ rhs`.
    pub inequality: Vec<LinearConstraint>,
    /// Linear equality constraints `Σ c x = rhs`.
    pub equality: Vec<LinearConstraint>,
    /// Nonlinear inequality constraints `g(x) ≥ 0`.
    pub nonlinear: Vec<Arc<dyn NonlinearInequalityConstraint>>,
}

impl ParameterConstraints {
    /// Whether there are no constraints.
    pub fn is_empty(&self) -> bool {
        self.inequality.is_empty() && self.equality.is_empty() && self.nonlinear.is_empty()
    }

    /// Whether there are linear constraints.
    pub fn has_linear(&self) -> bool {
        !(self.inequality.is_empty() && self.equality.is_empty())
    }

    /// Checks the linear constraints against `q` points of dimension `d`.
    pub(crate) fn validate(&self, q: usize, d: usize) -> Result<()> {
        for c in self.inequality.iter().chain(&self.equality) {
            c.validate(q, d)?;
        }
        Ok(())
    }

    /// BoTorch's check for sequential optimization: no inter-point
    /// constraints.
    pub(crate) fn validate_sequential(&self) -> Result<()> {
        let message = |i: usize, kind: &str| {
            BotorchError::Unsupported(format!(
                "Inter-point constraints are not supported for sequential optimization. But \
                 the {i}th {kind} constraint is defined as inter-point."
            ))
        };
        if let Some(i) = self.inequality.iter().position(|c| c.is_inter_point()) {
            return Err(message(i, "linear inequality"));
        }
        if let Some(i) = self.equality.iter().position(|c| c.is_inter_point()) {
            return Err(message(i, "linear equality"));
        }
        if let Some(i) = self.nonlinear.iter().position(|c| !c.is_intrapoint()) {
            return Err(message(i, "non-linear inequality"));
        }
        Ok(())
    }
}

/// The value of a nonlinear constraint for every point (intra-point) or for
/// the q-batch (inter-point) of `x` (`q × d`).
pub(crate) fn nonlinear_values(
    constraint: &dyn NonlinearInequalityConstraint,
    x: &Matrix,
) -> Result<Vec<f64>> {
    if constraint.is_intrapoint() {
        (0..x.nrows())
            .map(|p| {
                let row: Vec<f64> = x.row(p).iter().copied().collect();
                Ok(constraint.evaluate(&row)?.0)
            })
            .collect()
    } else {
        let flat: Vec<f64> = (0..x.nrows())
            .flat_map(|p| x.row(p).iter().copied().collect::<Vec<_>>())
            .collect();
        Ok(vec![constraint.evaluate(&flat)?.0])
    }
}

/// Whether the candidate set `x` (`q × d`) satisfies the constraints within
/// `tolerance` (BoTorch's `evaluate_feasibility`): inequalities `Σ c x ≥ rhs -
/// tol`, equalities `|Σ c x - rhs| < tol`, nonlinear `g(x) ≥ -tol`.
///
/// # Errors
/// Errors of the nonlinear constraints.
pub fn evaluate_feasibility(
    x: &Matrix,
    constraints: &ParameterConstraints,
    tolerance: f64,
) -> Result<bool> {
    for c in &constraints.inequality {
        if c.values(x)
            .iter()
            .any(|&v| v.is_nan() || v < c.rhs - tolerance)
        {
            return Ok(false);
        }
    }
    for c in &constraints.equality {
        if c.values(x)
            .iter()
            .any(|&v| v.is_nan() || (v - c.rhs).abs() >= tolerance)
        {
            return Ok(false);
        }
    }
    for c in &constraints.nonlinear {
        if nonlinear_values(c.as_ref(), x)?
            .iter()
            .any(|&v| v.is_nan() || v < -tolerance)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A linear constraint on the features that are not fixed (BoTorch's
/// `_generate_unfixed_lin_constraints`): fixed features move to the
/// right-hand side and the remaining feature indices are renumbered. Returns
/// `None` for a constraint whose features are all fixed and that is
/// satisfied.
///
/// # Errors
/// [`BotorchError::CandidateGeneration`] with BoTorch's message if a
/// constraint on fixed features only is violated.
pub(crate) fn unfixed_linear_constraint(
    c: &LinearConstraint,
    id: usize,
    fixed: &BTreeMap<usize, f64>,
    d: usize,
    eq: bool,
) -> Result<Option<LinearConstraint>> {
    let free: Vec<usize> = (0..d).filter(|k| !fixed.contains_key(k)).collect();
    let new_index = |k: usize| free.iter().position(|&f| f == k).expect("free feature");
    let mut rhs = c.rhs;
    let mut coefficients = Vec::new();
    let indices = match &c.indices {
        ConstraintIndices::IntraPoint(idx) => {
            let mut out = Vec::new();
            for (&k, &coef) in idx.iter().zip(&c.coefficients) {
                match fixed.get(&k) {
                    Some(&v) => rhs -= coef * v,
                    None => {
                        out.push(new_index(k));
                        coefficients.push(coef);
                    }
                }
            }
            ConstraintIndices::IntraPoint(out)
        }
        ConstraintIndices::InterPoint(idx) => {
            let mut out = Vec::new();
            for (&(p, k), &coef) in idx.iter().zip(&c.coefficients) {
                match fixed.get(&k) {
                    Some(&v) => rhs -= coef * v,
                    None => {
                        out.push((p, new_index(k)));
                        coefficients.push(coef);
                    }
                }
            }
            ConstraintIndices::InterPoint(out)
        }
    };
    if coefficients.is_empty() {
        if (eq && rhs != 0.0) || (!eq && rhs > 0.0) {
            let prefix = if eq { "Eq" } else { "Ineq" };
            return Err(BotorchError::CandidateGeneration(format!(
                "{prefix}uality constraint {id} not met with fixed_features."
            )));
        }
        return Ok(None);
    }
    Ok(Some(LinearConstraint {
        indices,
        coefficients,
        rhs,
    }))
}

/// The constraints on the features that are not fixed: linear constraints
/// reduced by [`unfixed_linear_constraint`]; nonlinear constraints are kept
/// (they are evaluated on the full points).
///
/// # Errors
/// See [`unfixed_linear_constraint`].
pub(crate) fn unfixed_constraints(
    constraints: &ParameterConstraints,
    fixed: &BTreeMap<usize, f64>,
    d: usize,
) -> Result<ParameterConstraints> {
    if fixed.is_empty() {
        return Ok(constraints.clone());
    }
    let reduce = |cs: &[LinearConstraint], eq: bool| -> Result<Vec<LinearConstraint>> {
        let mut out = Vec::new();
        for (i, c) in cs.iter().enumerate() {
            if let Some(r) = unfixed_linear_constraint(c, i, fixed, d, eq)? {
                out.push(r);
            }
        }
        Ok(out)
    };
    Ok(ParameterConstraints {
        inequality: reduce(&constraints.inequality, false)?,
        equality: reduce(&constraints.equality, true)?,
        nonlinear: constraints.nonlinear.clone(),
    })
}

/// Projects every point of `x` (`q × d`) onto the intra-point equality
/// constraints by least squares, `x + Aᵀ (A Aᵀ)⁻¹ (b - A x)` (BoTorch's
/// `project_to_equality_constraints`; inter-point constraints are ignored).
///
/// # Errors
/// [`BotorchError::Runtime`] if `A Aᵀ` is singular (as `torch.linalg.solve`).
pub fn project_to_equality_constraints(
    x: &Matrix,
    equality: &[LinearConstraint],
) -> Result<Matrix> {
    let intra: Vec<&LinearConstraint> = equality.iter().filter(|c| !c.is_inter_point()).collect();
    if intra.is_empty() {
        return Ok(x.clone());
    }
    let d = x.ncols();
    let k = intra.len();
    let mut a = Matrix::zeros(k, d);
    let mut b = crate::Vector::zeros(k);
    for (i, c) in intra.iter().enumerate() {
        c.validate(x.nrows(), d)?;
        if let ConstraintIndices::IntraPoint(idx) = &c.indices {
            // `A[i, indices] = coefficients` (the last duplicate wins, as in
            // torch's indexed assignment).
            for (&j, &coef) in idx.iter().zip(&c.coefficients) {
                a[(i, j)] = coef;
            }
        }
        b[i] = c.rhs;
    }
    let lu = (&a * a.transpose()).lu();
    if !lu.is_invertible() {
        return Err(BotorchError::Runtime(
            "linalg.solve: The solver failed because the input matrix is singular.".into(),
        ));
    }
    let mut out = x.clone();
    for p in 0..x.nrows() {
        let xp = x.row(p).transpose();
        let residual = &b - &a * &xp;
        let lam = lu.solve(&residual).expect("invertible");
        let correction = a.transpose() * lam;
        for j in 0..d {
            out[(p, j)] += correction[j];
        }
    }
    Ok(out)
}

/// Adjoint of the equality projection with respect to points, coefficients
/// and right-hand sides. Inter-point constraints have zero adjoints, since
/// the projection only operates on intra-point equalities.
#[derive(Clone, Debug)]
pub struct EqualityProjectionGrad {
    /// Adjoint of the input points.
    pub points: Matrix,
    /// Coefficient adjoints in the order of each input constraint.
    pub coefficients: Vec<Vec<f64>>,
    /// Right-hand-side adjoints in input constraint order.
    pub rhs: Vec<f64>,
}

/// Reverse derivative of [`project_to_equality_constraints`].
///
/// # Errors
/// The errors of the projection, and [`BotorchError::DimensionMismatch`]
/// when the output adjoint does not have the shape of `x`.
pub fn equality_projection_vjp(
    x: &Matrix,
    equality: &[LinearConstraint],
    output_adjoint: &Matrix,
) -> Result<EqualityProjectionGrad> {
    if output_adjoint.shape() != x.shape() {
        return Err(BotorchError::DimensionMismatch(
            "projection input and output adjoint shapes must agree".into(),
        ));
    }
    let intra: Vec<(usize, &LinearConstraint)> = equality
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_inter_point())
        .collect();
    let mut result = EqualityProjectionGrad {
        points: output_adjoint.clone(),
        coefficients: equality
            .iter()
            .map(|c| vec![0.0; c.coefficients.len()])
            .collect(),
        rhs: vec![0.0; equality.len()],
    };
    if intra.is_empty() {
        return Ok(result);
    }
    let d = x.ncols();
    let k = intra.len();
    let mut a = Matrix::zeros(k, d);
    let mut b = crate::Vector::zeros(k);
    for (row, (_, c)) in intra.iter().enumerate() {
        c.validate(x.nrows(), d)?;
        if let ConstraintIndices::IntraPoint(indices) = &c.indices {
            for (&column, &coefficient) in indices.iter().zip(&c.coefficients) {
                a[(row, column)] = coefficient;
            }
        }
        b[row] = c.rhs;
    }
    let lu = (&a * a.transpose()).lu();
    if !lu.is_invertible() {
        return Err(BotorchError::Runtime(
            "linalg.solve: The solver failed because the input matrix is singular.".into(),
        ));
    }
    let mut ga = Matrix::zeros(k, d);
    let mut gb = crate::Vector::zeros(k);
    for row in 0..x.nrows() {
        let point = x.row(row).transpose();
        let adjoint = output_adjoint.row(row).transpose();
        let lambda = lu.solve(&(&b - &a * &point)).expect("invertible");
        let u = lu.solve(&(&a * &adjoint)).expect("invertible");
        let projected = &point + a.transpose() * &lambda;
        let gx = adjoint - a.transpose() * &u;
        result.points.row_mut(row).copy_from(&gx.transpose());
        ga += &lambda * gx.transpose() - &u * projected.transpose();
        gb += u;
    }
    for (row, (original, c)) in intra.iter().enumerate() {
        if let ConstraintIndices::IntraPoint(indices) = &c.indices {
            // Indexed assignment's adjoint gathers every source index, as
            // torch does even when the forward indices contain duplicates.
            result.coefficients[*original] =
                indices.iter().map(|&column| ga[(row, column)]).collect();
        }
        result.rhs[*original] = gb[row];
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_projection_has_closed_form_adjoint() {
        let x = Matrix::from_row_slice(1, 2, &[0.2, 0.4]);
        let equality =
            vec![LinearConstraint::intra_point(vec![0, 1], vec![1.0, 1.0], 1.2).unwrap()];
        let projected = project_to_equality_constraints(&x, &equality).unwrap();
        assert!((projected[(0, 0)] - 0.5).abs() < 1e-15);
        assert!((projected[(0, 1)] - 0.7).abs() < 1e-15);
        let adjoint = Matrix::from_row_slice(1, 2, &[1.0, 2.0]);
        let grad = equality_projection_vjp(&x, &equality, &adjoint).unwrap();
        assert!((grad.points[(0, 0)] + 0.5).abs() < 1e-15);
        assert!((grad.points[(0, 1)] - 0.5).abs() < 1e-15);
        assert!((grad.rhs[0] - 1.5).abs() < 1e-15);
        for coefficient in &grad.coefficients[0] {
            assert!((coefficient + 0.9).abs() < 1e-15);
        }
    }

    #[test]
    fn rows_feasibility_and_fixed_features() {
        // x0 + 2 x2 >= 1 for every point; x[0, 1] - x[1, 1] = 0.25 across points.
        let ineq = LinearConstraint::intra_point(vec![2, 0], vec![2.0, 1.0], 1.0).unwrap();
        let eq =
            LinearConstraint::inter_point(vec![(0, 1), (1, 1)], vec![1.0, -1.0], 0.25).unwrap();
        assert_eq!(
            ineq.rows(2, 3),
            vec![vec![(0, 1.0), (2, 2.0)], vec![(3, 1.0), (5, 2.0)]]
        );
        assert_eq!(eq.rows(2, 3), vec![vec![(1, 1.0), (4, -1.0)]]);
        let constraints = ParameterConstraints {
            inequality: vec![ineq.clone()],
            equality: vec![eq.clone()],
            nonlinear: vec![Arc::new(FnNonlinearConstraint::intrapoint(|x| {
                (
                    1.0 - x[0] * x[0] - x[1] * x[1],
                    vec![-2.0 * x[0], -2.0 * x[1], 0.0],
                )
            }))],
        };
        let x = Matrix::from_row_slice(2, 3, &[0.2, 0.5, 0.4, 0.5, 0.25, 0.3]);
        assert!(evaluate_feasibility(&x, &constraints, CONSTRAINT_TOLERANCE).unwrap());
        let mut y = x.clone();
        y[(1, 2)] = 0.2; // 0.5 + 0.4 < 1
        assert!(!evaluate_feasibility(&y, &constraints, CONSTRAINT_TOLERANCE).unwrap());
        assert!(constraints.validate_sequential().is_err());
        assert!(constraints.validate(2, 3).is_ok());
        assert!(matches!(
            constraints.validate(1, 3),
            Err(BotorchError::Runtime(_))
        ));
        assert!(ineq.validate(1, 2).is_err());
        // Fixing x2 = 0.3 turns the inequality into x0 >= 0.4 on feature 0 of
        // the reduced space; fixing x0 too leaves a violated constraint.
        let fixed: BTreeMap<usize, f64> = [(2, 0.3)].into_iter().collect();
        let r = unfixed_linear_constraint(&ineq, 0, &fixed, 3, false)
            .unwrap()
            .unwrap();
        assert_eq!(r.indices(), &ConstraintIndices::IntraPoint(vec![0]));
        assert!((r.rhs() - 0.4).abs() < 1e-15);
        let both: BTreeMap<usize, f64> = [(0, 0.1), (2, 0.3)].into_iter().collect();
        let e = unfixed_linear_constraint(&ineq, 0, &both, 3, false).unwrap_err();
        assert_eq!(
            e.to_string(),
            "Inequality constraint 0 not met with fixed_features."
        );
        let ok: BTreeMap<usize, f64> = [(0, 0.9), (2, 0.3)].into_iter().collect();
        assert!(unfixed_linear_constraint(&ineq, 0, &ok, 3, false)
            .unwrap()
            .is_none());
        // Equality projection.
        let e2 = LinearConstraint::intra_point(vec![0, 1], vec![1.0, 1.0], 1.0).unwrap();
        let p = project_to_equality_constraints(&x, std::slice::from_ref(&e2)).unwrap();
        for i in 0..2 {
            assert!((p[(i, 0)] + p[(i, 1)] - 1.0).abs() < 1e-15);
            assert_eq!(p[(i, 2)], x[(i, 2)]);
        }
    }
}
