//! Regularization of acquisition functions (`botorch.acquisition.penalized`).
//!
//! * [`PenalizedAcquisitionFunction`]: `raw_acqf(X) - λ penalty(X)` with a
//!   [`Penalty`] of the candidate set: [`L2Penalty`], [`L1Penalty`],
//!   [`GaussianPenalty`], [`GroupLassoPenalty`], [`L0PenaltyApprox`] (or a
//!   user function, [`FnPenalty`]).
//! * [`PenalizedMcObjective`]: `objective(samples, X) - λ penalty(X)` with a
//!   [`PointPenalty`] of every point: [`L1PenaltyObjective`],
//!   [`L0PenaltyApproxObjective`].
//! * The helpers `group_lasso_regularizer`, `narrow_gaussian` and
//!   `nnz_approx` ([`L0Approximation`]).
//!
//! Gradients are exact (the subgradients that `torch.autograd` computes: the
//! maximum and minimum over the points pass the gradient to the first
//! extremal point, `|·|` has derivative 0 at 0 and the Euclidean norm has
//! gradient 0 at 0).

use std::fmt::Debug;
use std::sync::Arc;

use crate::acquisition::mc_objective::{McObjective, McSamples, SamplePullback, SampleValues};
use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::{Matrix, Vector};

/// A penalty of a candidate set (`q × d`) with its gradient (BoTorch's
/// penalty modules of `PenalizedAcquisitionFunction`).
pub trait Penalty: Send + Sync + Debug {
    /// The penalty of the candidate set `x` and, if `with_grad`, its gradient
    /// (`q × d`).
    ///
    /// # Errors
    /// Implementation-specific (e.g. shape errors).
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)>;
}

/// The reverse pass of per-point values: maps the adjoint of the values
/// (`n`) to the adjoint of the points (`n × d`).
pub type PointPullback = Box<dyn FnOnce(&Vector) -> Result<Matrix> + Send>;

/// A penalty of every point (`n × d` to `n` values, BoTorch's penalty
/// objectives of `PenalizedMCObjective`).
pub trait PointPenalty: Send + Sync + Debug {
    /// The penalties of the points `x` and, if `with_grad`, the reverse pass.
    ///
    /// # Errors
    /// Implementation-specific (e.g. shape errors).
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(Vector, Option<PointPullback>)>;
}

/// The pullback of per-point values whose gradients with respect to their own
/// points are `g` (`n × d`).
fn diagonal_pullback(g: Matrix) -> PointPullback {
    Box::new(move |bar: &Vector| {
        let mut out = g;
        for (i, b) in bar.iter().enumerate() {
            out.row_mut(i).scale_mut(*b);
        }
        Ok(out)
    })
}

fn check_point(point: &Vector, x: &Matrix) -> Result<()> {
    if point.len() != x.ncols() {
        return Err(BotorchError::DimensionMismatch(format!(
            "the reference point has {} entries but X has {} columns",
            point.len(),
            x.ncols()
        )));
    }
    Ok(())
}

/// `torch.sign`.
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The index of the first maximum (`torch.max(dim=-1)`).
fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] || (v[best].is_nan() && !x.is_nan()) {
            best = i;
        }
    }
    best
}

/// The index of the first minimum (`torch.min(dim=-1)`).
fn argmin(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x < v[best] {
            best = i;
        }
    }
    best
}

/// A penalty that is the maximum over the points of a per-point penalty.
fn max_over_points<F>(x: &Matrix, with_grad: bool, f: F) -> (f64, Option<Matrix>)
where
    F: Fn(usize, bool) -> (f64, Option<Vector>),
{
    let q = x.nrows();
    let vals: Vec<(f64, Option<Vector>)> = (0..q).map(|j| f(j, with_grad)).collect();
    let v: Vec<f64> = vals.iter().map(|(v, _)| *v).collect();
    let j = argmax(&v);
    let grad = with_grad.then(|| {
        let mut g = Matrix::zeros(q, x.ncols());
        if let Some(gj) = &vals[j].1 {
            g.row_mut(j).copy_from(&gj.transpose());
        }
        g
    });
    (v[j], grad)
}

/// BoTorch's `L2Penalty(init_point)`: `max_j ||x_j - p||²`.
#[derive(Clone, Debug, PartialEq)]
pub struct L2Penalty {
    init_point: Vector,
}

impl L2Penalty {
    /// The penalty with the reference point `init_point` (`d`).
    pub fn new(init_point: Vector) -> Self {
        Self { init_point }
    }
}

impl Penalty for L2Penalty {
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        check_point(&self.init_point, x)?;
        let p = &self.init_point;
        Ok(max_over_points(x, with_grad, |j, g| {
            let diff = Vector::from_fn(x.ncols(), |k, _| x[(j, k)] - p[k]);
            let n = diff.norm();
            (
                n * n,
                g.then(|| {
                    if n > 0.0 {
                        &diff * 2.0
                    } else {
                        diff.map(|_| 0.0)
                    }
                }),
            )
        }))
    }
}

/// BoTorch's `L1Penalty(init_point)`: `max_j ||x_j - p||_1`.
#[derive(Clone, Debug, PartialEq)]
pub struct L1Penalty {
    init_point: Vector,
}

impl L1Penalty {
    /// The penalty with the reference point `init_point` (`d`).
    pub fn new(init_point: Vector) -> Self {
        Self { init_point }
    }
}

impl Penalty for L1Penalty {
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        check_point(&self.init_point, x)?;
        let p = &self.init_point;
        Ok(max_over_points(x, with_grad, |j, g| {
            let diff = Vector::from_fn(x.ncols(), |k, _| x[(j, k)] - p[k]);
            (
                diff.iter().map(|v| v.abs()).sum(),
                g.then(|| diff.map(sign)),
            )
        }))
    }
}

/// BoTorch's `GaussianPenalty(init_point, sigma)`:
/// `max_j exp(||x_j - p||² / (2 σ²))`.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPenalty {
    init_point: Vector,
    sigma: f64,
}

impl GaussianPenalty {
    /// The penalty with the reference point `init_point` (`d`) and `sigma`.
    pub fn new(init_point: Vector, sigma: f64) -> Self {
        Self { init_point, sigma }
    }
}

impl Penalty for GaussianPenalty {
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        check_point(&self.init_point, x)?;
        let p = &self.init_point;
        let s2 = self.sigma * self.sigma;
        Ok(max_over_points(x, with_grad, |j, g| {
            let diff = Vector::from_fn(x.ncols(), |k, _| x[(j, k)] - p[k]);
            let n = diff.norm();
            let e = (n * n / 2.0 / s2).exp();
            (
                e,
                g.then(|| {
                    if n > 0.0 {
                        &diff * (e / s2)
                    } else {
                        diff.map(|_| 0.0)
                    }
                }),
            )
        }))
    }
}

/// BoTorch's `group_lasso_regularizer(X, groups)` of one point:
/// `Σ_g sqrt(|g|) ||x_g||_2`, and its gradient.
pub fn group_lasso_regularizer(x: &[f64], groups: &[Vec<usize>]) -> Result<(f64, Vector)> {
    let mut v = 0.0;
    let mut g = Vector::zeros(x.len());
    for grp in groups {
        if let Some(&k) = grp.iter().find(|&&k| k >= x.len()) {
            return Err(BotorchError::DimensionMismatch(format!(
                "index {k} is out of bounds for dimension with size {}",
                x.len()
            )));
        }
        let n = grp.iter().map(|&k| x[k] * x[k]).sum::<f64>().sqrt();
        let s = (grp.len() as f64).sqrt();
        v += s * n;
        if n > 0.0 {
            for &k in grp {
                g[k] += s * x[k] / n;
            }
        }
    }
    Ok((v, g))
}

/// BoTorch's `GroupLassoPenalty(init_point, groups)`: the group lasso of
/// `x - p` (`q = 1` only, as in BoTorch).
#[derive(Clone, Debug, PartialEq)]
pub struct GroupLassoPenalty {
    init_point: Vector,
    groups: Vec<Vec<usize>>,
}

impl GroupLassoPenalty {
    /// The penalty with the reference point `init_point` (`d`) and the groups
    /// of indices.
    pub fn new(init_point: Vector, groups: Vec<Vec<usize>>) -> Self {
        Self { init_point, groups }
    }
}

impl Penalty for GroupLassoPenalty {
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        if x.nrows() != 1 {
            return Err(BotorchError::Unsupported(
                "group-lasso has not been implemented for q>1 yet.".into(),
            ));
        }
        check_point(&self.init_point, x)?;
        let diff: Vec<f64> = (0..x.ncols())
            .map(|k| x[(0, k)] - self.init_point[k])
            .collect();
        let (v, g) = group_lasso_regularizer(&diff, &self.groups)?;
        Ok((
            v,
            with_grad.then(|| Matrix::from_row_slice(1, g.len(), g.as_slice())),
        ))
    }
}

/// BoTorch's `narrow_gaussian(X, a)`: `exp(-0.5 (x / a)²)`.
pub fn narrow_gaussian(x: f64, a: f64) -> f64 {
    (-0.5 * (x / a) * (x / a)).exp()
}

/// BoTorch's `L0Approximation(target_point, a)` / `nnz_approx(X,
/// target_point, a)`: the differentiable relaxation `d - Σ_k exp(-0.5
/// ((x_k - t_k) / a)²)` of `||x - t||_0` of every point.
#[derive(Clone, Debug, PartialEq)]
pub struct L0Approximation {
    target_point: Vector,
    a: f64,
}

impl L0Approximation {
    /// The relaxation with the target point `target_point` (`d`) and `a`.
    pub fn new(target_point: Vector, a: f64) -> Self {
        Self { target_point, a }
    }

    /// The relaxed `||x_j - t||_0` of every point (`n`) and its gradient.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `x` and the target point have
    /// different dimensions (BoTorch's message).
    pub fn nnz(&self, x: &Matrix, with_grad: bool) -> Result<(Vector, Option<Matrix>)> {
        let d = x.ncols();
        if d != self.target_point.len() {
            return Err(BotorchError::InvalidArgument(
                "X and target_point have different shapes.".into(),
            ));
        }
        let a = self.a;
        let n = x.nrows();
        let mut v = Vector::zeros(n);
        let mut g = Matrix::zeros(if with_grad { n } else { 0 }, d);
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..d {
                let z = x[(j, k)] - self.target_point[k];
                let e = narrow_gaussian(z, a);
                s += e;
                if with_grad {
                    g[(j, k)] = e * z / (a * a);
                }
            }
            v[j] = d as f64 - s;
        }
        Ok((v, with_grad.then_some(g)))
    }
}

/// BoTorch's `L0PenaltyApprox(target_point, a)`: the minimum over the points
/// of [`L0Approximation`].
#[derive(Clone, Debug, PartialEq)]
pub struct L0PenaltyApprox {
    inner: L0Approximation,
}

impl L0PenaltyApprox {
    /// The penalty with the target point `target_point` (`d`) and `a`
    /// (BoTorch's default 1).
    pub fn new(target_point: Vector, a: f64) -> Self {
        Self {
            inner: L0Approximation::new(target_point, a),
        }
    }
}

impl Penalty for L0PenaltyApprox {
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let (v, g) = self.inner.nnz(x, with_grad)?;
        let vals: Vec<f64> = v.iter().copied().collect();
        let j = argmin(&vals);
        let grad = g.map(|g| {
            let mut out = Matrix::zeros(x.nrows(), x.ncols());
            out.row_mut(j).copy_from(&g.row(j));
            out
        });
        Ok((vals[j], grad))
    }
}

/// The function of a [`FnPenalty`]: the penalty of a candidate set and,
/// if requested, its gradient.
pub type PenaltyFn = dyn Fn(&Matrix, bool) -> Result<(f64, Option<Matrix>)> + Send + Sync;

/// A penalty given by a function (e.g. a `torch.nn.Module` evaluated through
/// the Python bindings).
pub struct FnPenalty {
    f: Arc<PenaltyFn>,
}

impl Debug for FnPenalty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FnPenalty")
    }
}

impl FnPenalty {
    /// The penalty of the function `f(x, with_grad)`.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Matrix, bool) -> Result<(f64, Option<Matrix>)> + Send + Sync + 'static,
    {
        Self { f: Arc::new(f) }
    }
}

impl Penalty for FnPenalty {
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        (self.f)(x, with_grad)
    }
}

/// BoTorch's `PenalizedAcquisitionFunction(raw_acqf, penalty_func,
/// regularization_parameter)`: `raw_acqf(X) - λ penalty(X)`.
#[derive(Debug)]
pub struct PenalizedAcquisitionFunction {
    raw: Arc<dyn AcquisitionFunction>,
    penalty: Arc<dyn Penalty>,
    regularization_parameter: f64,
}

impl PenalizedAcquisitionFunction {
    /// The penalized acquisition function.
    pub fn new(
        raw_acqf: Arc<dyn AcquisitionFunction>,
        penalty: Arc<dyn Penalty>,
        regularization_parameter: f64,
    ) -> Self {
        Self {
            raw: raw_acqf,
            penalty,
            regularization_parameter,
        }
    }

    /// The raw acquisition function.
    pub fn raw_acqf(&self) -> &Arc<dyn AcquisitionFunction> {
        &self.raw
    }

    /// The regularization parameter `λ`.
    pub fn regularization_parameter(&self) -> f64 {
        self.regularization_parameter
    }
}

impl AcquisitionFunction for PenalizedAcquisitionFunction {
    fn model(&self) -> &Arc<dyn Model> {
        self.raw.model()
    }

    fn dim(&self) -> usize {
        self.raw.dim()
    }

    fn expected_q(&self) -> Option<usize> {
        self.raw.expected_q()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        let v = self.raw.forward(x)?;
        let (p, _) = self.penalty.evaluate(x, false)?;
        Ok(v - self.regularization_parameter * p)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.raw.forward_with_grad(x)?;
        let (p, gp) = self.penalty.evaluate(x, true)?;
        let gp =
            gp.ok_or_else(|| BotorchError::Runtime("the penalty returned no gradient".into()))?;
        Ok((
            v - self.regularization_parameter * p,
            g - gp * self.regularization_parameter,
        ))
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.raw.x_pending()
    }

    /// Sets the pending points of the raw acquisition function (analytic
    /// ones reject them).
    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.raw.set_x_pending(x_pending)
    }

    fn is_log(&self) -> bool {
        self.raw.is_log()
    }
}

/// BoTorch's `L1PenaltyObjective(init_point)`: `||x_j - p||_1` of every
/// point.
#[derive(Clone, Debug, PartialEq)]
pub struct L1PenaltyObjective {
    init_point: Vector,
}

impl L1PenaltyObjective {
    /// The penalty objective with the reference point `init_point` (`d`).
    pub fn new(init_point: Vector) -> Self {
        Self { init_point }
    }
}

impl PointPenalty for L1PenaltyObjective {
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(Vector, Option<PointPullback>)> {
        check_point(&self.init_point, x)?;
        let p = &self.init_point;
        let n = x.nrows();
        let v = Vector::from_fn(n, |j, _| {
            (0..x.ncols()).map(|k| (x[(j, k)] - p[k]).abs()).sum()
        });
        let g = with_grad.then(|| {
            diagonal_pullback(Matrix::from_fn(n, x.ncols(), |j, k| sign(x[(j, k)] - p[k])))
        });
        Ok((v, g))
    }
}

/// BoTorch's `L0PenaltyApproxObjective(target_point, a)`: [`L0Approximation`]
/// of every point.
#[derive(Clone, Debug, PartialEq)]
pub struct L0PenaltyApproxObjective {
    inner: L0Approximation,
}

impl L0PenaltyApproxObjective {
    /// The penalty objective with the target point `target_point` (`d`) and
    /// `a`.
    pub fn new(target_point: Vector, a: f64) -> Self {
        Self {
            inner: L0Approximation::new(target_point, a),
        }
    }
}

impl PointPenalty for L0PenaltyApproxObjective {
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(Vector, Option<PointPullback>)> {
        let (v, g) = self.inner.nnz(x, with_grad)?;
        Ok((v, g.map(diagonal_pullback)))
    }
}

/// The function of a [`FnPointPenalty`].
pub type PointPenaltyFn =
    dyn Fn(&Matrix, bool) -> Result<(Vector, Option<PointPullback>)> + Send + Sync;

/// A per-point penalty given by a function.
pub struct FnPointPenalty {
    f: Arc<PointPenaltyFn>,
}

impl Debug for FnPointPenalty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FnPointPenalty")
    }
}

impl FnPointPenalty {
    /// The penalty of the function `f(x, with_grad)`.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Matrix, bool) -> Result<(Vector, Option<PointPullback>)> + Send + Sync + 'static,
    {
        Self { f: Arc::new(f) }
    }
}

impl PointPenalty for FnPointPenalty {
    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(Vector, Option<PointPullback>)> {
        (self.f)(x, with_grad)
    }
}

/// BoTorch's `PenalizedMCObjective(objective, penalty_objective,
/// regularization_parameter)`: `objective(samples, X) - λ penalty(X)` with
/// the penalty of every point subtracted from all of its samples.
#[derive(Debug)]
pub struct PenalizedMcObjective {
    objective: Arc<dyn McObjective>,
    penalty: Arc<dyn PointPenalty>,
    regularization_parameter: f64,
}

impl PenalizedMcObjective {
    /// The penalized objective.
    pub fn new(
        objective: Arc<dyn McObjective>,
        penalty: Arc<dyn PointPenalty>,
        regularization_parameter: f64,
    ) -> Self {
        Self {
            objective,
            penalty,
            regularization_parameter,
        }
    }
}

impl McObjective for PenalizedMcObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let x = x.ok_or_else(|| {
            BotorchError::InvalidArgument(
                "PenalizedMCObjective requires the inputs X of the samples.".into(),
            )
        })?;
        let base = self.objective.evaluate(samples, Some(x), with_grad)?;
        let (p, gp) = self.penalty.evaluate(x, with_grad)?;
        let (n, s) = base.values.shape();
        if p.len() != n {
            return Err(BotorchError::DimensionMismatch(format!(
                "the penalty returned {} values for {n} points",
                p.len()
            )));
        }
        let lam = self.regularization_parameter;
        let values = Matrix::from_fn(n, s, |i, c| base.values[(i, c)] - lam * p[i]);
        let pullback: Option<SamplePullback> = match (base.pullback, gp) {
            (Some(pb), Some(gp)) => Some(Box::new(move |bar: &Matrix| {
                let (sbar, xbar) = pb(bar)?;
                let totals = Vector::from_fn(bar.nrows(), |i, _| -lam * bar.row(i).sum());
                let gpen = gp(&totals)?;
                let xb = match xbar {
                    Some(xb) => xb + gpen,
                    None => gpen,
                };
                Ok((sbar, Some(xb)))
            })),
            _ => None,
        };
        Ok(SampleValues { values, pullback })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd_check(p: &dyn Penalty, x: &Matrix) {
        let (_, g) = p.evaluate(x, true).unwrap();
        let g = g.unwrap();
        let h = 1e-6;
        for i in 0..x.nrows() {
            for k in 0..x.ncols() {
                let mut a = x.clone();
                let mut b = x.clone();
                a[(i, k)] += h;
                b[(i, k)] -= h;
                let fd = (p.evaluate(&a, false).unwrap().0 - p.evaluate(&b, false).unwrap().0)
                    / (2.0 * h);
                assert!(
                    (fd - g[(i, k)]).abs() < 1e-6,
                    "{p:?} ({i},{k}): {fd} vs {}",
                    g[(i, k)]
                );
            }
        }
    }

    #[test]
    fn penalties_and_gradients() {
        let x = Matrix::from_row_slice(2, 3, &[0.1, 0.5, -0.2, 0.7, 0.3, 0.4]);
        let p = Vector::from_vec(vec![0.2, 0.1, 0.0]);
        fd_check(&L2Penalty::new(p.clone()), &x);
        fd_check(&L1Penalty::new(p.clone()), &x);
        fd_check(&GaussianPenalty::new(p.clone(), 0.7), &x);
        fd_check(&L0PenaltyApprox::new(p.clone(), 0.5), &x);
        let x1 = x.rows(0, 1).into_owned();
        fd_check(
            &GroupLassoPenalty::new(p.clone(), vec![vec![0, 2], vec![1]]),
            &x1,
        );
        assert!(GroupLassoPenalty::new(p.clone(), vec![vec![0]])
            .evaluate(&x, false)
            .is_err());
        let (v, _) = L2Penalty::new(p.clone()).evaluate(&x, false).unwrap();
        let expected: f64 = (0..3).map(|k| (x[(1, k)] - p[k]).powi(2)).sum();
        assert!((v - expected).abs() < 1e-15);
    }
}
