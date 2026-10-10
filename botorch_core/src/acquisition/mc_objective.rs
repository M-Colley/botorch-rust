//! Monte-Carlo objectives (`botorch.acquisition.objective`):
//! [`IdentityMcObjective`], [`LinearMcObjective`], [`GenericMcObjective`]
//! and the legacy feasibility-weighted [`ConstrainedMcObjective`], operating
//! on posterior samples ([`McSamples`]).
//!
//! An MC acquisition function maps the joint posterior samples of its `n`
//! points (`q` candidates and pending or baseline points) and `m` model
//! outputs to one objective value per point and sample, `obj(samples, X)`,
//! and computes its utility from these values. Without an objective the model
//! must have a single output (or a scalarizing posterior transform), whose
//! samples are used directly.
//!
//! Objectives provide exact reverse-mode derivatives: [`McObjective::evaluate`]
//! with `with_grad = true` returns a [`SamplePullback`] that maps the adjoint
//! of the objective values to the adjoints of the samples and of the inputs
//! (for objectives that depend on `X` directly).

use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::utils::objective::{
    compute_smoothed_feasibility_indicator, McConstraint, OutcomeConstraints,
};
use crate::{Matrix, Vector};

/// Joint posterior samples of `n` points and `m` outputs for `S` MC samples:
/// BoTorch's `sample_shape x batch_shape x n x m` samples of one candidate set.
///
/// The values are stored output-major like the joint posteriors of this crate:
/// row `t·n + p` of [`McSamples::data`] holds output `t` at point `p` for every
/// MC sample (one column per sample).
#[derive(Clone, Debug, PartialEq)]
pub struct McSamples {
    data: Matrix,
    num_points: usize,
    num_outputs: usize,
}

impl McSamples {
    /// Samples from their output-major `(m·n) × S` matrix.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `data` does not have
    /// `num_points · num_outputs` rows or `num_outputs` is zero.
    pub fn new(data: Matrix, num_points: usize, num_outputs: usize) -> Result<Self> {
        if num_outputs == 0 || data.nrows() != num_points * num_outputs {
            return Err(BotorchError::DimensionMismatch(format!(
                "samples of {num_points} points and {num_outputs} outputs need {} rows, got {}",
                num_points * num_outputs,
                data.nrows()
            )));
        }
        Ok(Self {
            data,
            num_points,
            num_outputs,
        })
    }

    /// Samples from BoTorch's row-major `S × n × m` layout (index
    /// `(s·n + p)·m + t`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `values` does not have
    /// `S·n·m` elements.
    pub fn from_row_major(
        values: &[f64],
        num_samples: usize,
        num_points: usize,
        num_outputs: usize,
    ) -> Result<Self> {
        let (n, m) = (num_points, num_outputs);
        if values.len() != num_samples * n * m || m == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {num_samples} x {n} x {m} samples, got {} values",
                values.len()
            )));
        }
        let data = Matrix::from_fn(m * n, num_samples, |r, s| {
            let (t, p) = (r / n, r % n);
            values[(s * n + p) * m + t]
        });
        Ok(Self {
            data,
            num_points: n,
            num_outputs: m,
        })
    }

    /// The samples in BoTorch's row-major `S × n × m` layout.
    pub fn to_row_major(&self) -> Vec<f64> {
        let (n, m, s) = (self.num_points, self.num_outputs, self.num_samples());
        let mut out = Vec::with_capacity(s * n * m);
        for c in 0..s {
            for p in 0..n {
                for t in 0..m {
                    out.push(self.data[(t * n + p, c)]);
                }
            }
        }
        out
    }

    /// The output-major `(m·n) × S` matrix of samples.
    pub fn data(&self) -> &Matrix {
        &self.data
    }

    /// The output-major matrix of samples, consuming `self`.
    pub fn into_data(self) -> Matrix {
        self.data
    }

    /// Number of points `n`.
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    /// Number of outputs `m`.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Number of MC samples `S`.
    pub fn num_samples(&self) -> usize {
        self.data.ncols()
    }

    /// Output `t` at point `p` in MC sample `s`.
    pub fn value(&self, s: usize, p: usize, t: usize) -> f64 {
        self.data[(t * self.num_points + p, s)]
    }

    /// The samples of output `t` (`n × S`).
    pub fn output(&self, t: usize) -> Matrix {
        self.data
            .rows(t * self.num_points, self.num_points)
            .into_owned()
    }

    /// The samples of the points `points` (all outputs).
    pub fn points(&self, points: Range<usize>) -> McSamples {
        let (n, m) = (self.num_points, self.num_outputs);
        let k = points.len();
        let data = Matrix::from_fn(m * k, self.num_samples(), |r, s| {
            let (t, i) = (r / k, r % k);
            self.data[(t * n + points.start + i, s)]
        });
        McSamples {
            data,
            num_points: k,
            num_outputs: m,
        }
    }

    /// Embeds the adjoint of the samples of the points `points` (as returned
    /// for [`McSamples::points`]) into an adjoint of all `n` points.
    pub(crate) fn embed_points_adjoint(&self, points: Range<usize>, adjoint: &Matrix) -> Matrix {
        let (n, m) = (self.num_points, self.num_outputs);
        let k = points.len();
        let mut out = Matrix::zeros(m * n, self.num_samples());
        for t in 0..m {
            for i in 0..k {
                out.row_mut(t * n + points.start + i)
                    .copy_from(&adjoint.row(t * k + i));
            }
        }
        out
    }
}

/// Reverse-mode derivative of a map from posterior samples (and inputs) to
/// per-point values: maps the adjoint of the values (`n × S`) to the adjoint
/// of the samples (`(m·n) × S`, the layout of [`McSamples`]) and, if the
/// values depend on the inputs `X` directly, the adjoint of the inputs
/// (`n × d`).
pub type SamplePullback = Box<dyn FnOnce(&Matrix) -> Result<(Matrix, Option<Matrix>)> + Send>;

/// Per-point values computed from posterior samples (`n × S`, one column per
/// MC sample), with a pullback if derivatives were requested.
pub struct SampleValues {
    /// The values (`n × S`).
    pub values: Matrix,
    /// The reverse-mode derivative (if requested).
    pub pullback: Option<SamplePullback>,
}

impl Debug for SampleValues {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SampleValues")
            .field("values", &self.values)
            .field("pullback", &self.pullback.is_some())
            .finish()
    }
}

/// An MC objective (BoTorch's `MCAcquisitionObjective`): maps posterior
/// samples of `n` points to one value per point and MC sample, to be
/// maximized.
pub trait McObjective: Send + Sync + Debug {
    /// Objective values (`n × S`) of `samples` of the points `x` (`n × d`;
    /// `None` where BoTorch passes `X=None`); with `with_grad`, also the
    /// pullback.
    ///
    /// # Errors
    /// Implementation-specific (e.g. a wrong number of outputs).
    fn evaluate(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues>;

    /// Whether the value at a point depends only on the samples at that point
    /// (and not on `X`), which lets acquisition functions reuse values that
    /// do not depend on the candidates. Defaults to `false`.
    fn is_pointwise(&self) -> bool {
        false
    }

    /// Whether this is a [`ConstrainedMcObjective`] (which BoTorch's LogEI
    /// acquisition functions reject).
    fn is_constrained(&self) -> bool {
        false
    }
}

/// The error of BoTorch's `MCAcquisitionObjective.__call__` for objective
/// values whose number of points differs from the number of inputs.
pub(crate) fn q_batch_mismatch(got: usize, expected: usize) -> BotorchError {
    BotorchError::Runtime(format!(
        "The q-batch shape of the objective values does not agree with the q-batch shape \
         of X. Got {got} and {expected}. This may happen if you used a one-to-many input \
         transform but forgot to use a corresponding objective."
    ))
}

/// The trivial objective (BoTorch's `IdentityMCObjective`): the samples of a
/// single-output model, `samples.squeeze(-1)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IdentityMcObjective;

impl McObjective for IdentityMcObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let m = samples.num_outputs();
        if m != 1 {
            return Err(q_batch_mismatch(
                m,
                x.map_or(samples.num_points(), |x| x.nrows()),
            ));
        }
        let pullback: Option<SamplePullback> = if with_grad {
            Some(Box::new(|bar: &Matrix| Ok((bar.clone(), None))))
        } else {
            None
        };
        Ok(SampleValues {
            values: samples.data().clone(),
            pullback,
        })
    }

    fn is_pointwise(&self) -> bool {
        true
    }
}

/// A linear objective `Σ_t weights[t] · samples[..., t]` (BoTorch's
/// `LinearMCObjective(weights)`).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearMcObjective {
    weights: Vector,
}

impl LinearMcObjective {
    /// The objective with one weight per model output.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for empty or non-finite weights.
    pub fn new(weights: Vector) -> Result<Self> {
        if weights.is_empty() || weights.iter().any(|w| !w.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "weights must be a non-empty vector of finite values".into(),
            ));
        }
        Ok(Self { weights })
    }

    /// The weights.
    pub fn weights(&self) -> &Vector {
        &self.weights
    }
}

impl McObjective for LinearMcObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let (n, m) = (samples.num_points(), samples.num_outputs());
        if m != self.weights.len() {
            return Err(BotorchError::Runtime(
                "Output shape of samples not equal to that of weights".into(),
            ));
        }
        let data = samples.data();
        let s = samples.num_samples();
        let mut values = Matrix::zeros(n, s);
        for t in 0..m {
            let w = self.weights[t];
            for c in 0..s {
                for p in 0..n {
                    values[(p, c)] += data[(t * n + p, c)] * w;
                }
            }
        }
        let pullback: Option<SamplePullback> = if with_grad {
            let weights = self.weights.clone();
            Some(Box::new(move |bar: &Matrix| {
                let mut out = Matrix::zeros(m * n, bar.ncols());
                for t in 0..m {
                    out.rows_mut(t * n, n).copy_from(&(bar * weights[t]));
                }
                Ok((out, None))
            }))
        } else {
            None
        };
        Ok(SampleValues { values, pullback })
    }

    fn is_pointwise(&self) -> bool {
        true
    }
}

/// The value and the gradient (with respect to the `m` outputs) of a
/// pointwise function of the outputs at one point and sample.
pub type PointwiseFn = dyn Fn(&[f64]) -> (f64, Vec<f64>) + Send + Sync;

/// An objective given by a function of the outputs at one point, applied to
/// every point and MC sample (BoTorch's `GenericMCObjective` for callables
/// that act pointwise, e.g. `lambda Y, X: Y[..., 0] - Y[..., 1] ** 2`).
///
/// The closure returns the value and its gradient with respect to the `m`
/// outputs. Objectives that couple points or depend on `X` directly
/// implement [`McObjective`] instead.
#[derive(Clone)]
pub struct GenericMcObjective {
    f: Arc<PointwiseFn>,
}

impl Debug for GenericMcObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GenericMcObjective")
    }
}

impl GenericMcObjective {
    /// The objective `f(y)` with gradient (`f` returns `(value, ∂value/∂y)`).
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>) + Send + Sync + 'static,
    {
        Self { f: Arc::new(f) }
    }
}

/// Applies a pointwise function to every point and sample: values (`n × S`)
/// and, if requested, the gradients (`(m·n) × S`).
pub(crate) fn apply_pointwise(
    f: &PointwiseFn,
    samples: &McSamples,
    with_grad: bool,
) -> Result<(Matrix, Option<Matrix>)> {
    let (n, m, s) = (
        samples.num_points(),
        samples.num_outputs(),
        samples.num_samples(),
    );
    let data = samples.data();
    let mut values = Matrix::zeros(n, s);
    let mut grads = if with_grad {
        Some(Matrix::zeros(m * n, s))
    } else {
        None
    };
    let mut y = vec![0.0; m];
    for c in 0..s {
        for p in 0..n {
            for (t, yt) in y.iter_mut().enumerate() {
                *yt = data[(t * n + p, c)];
            }
            let (v, g) = f(&y);
            if g.len() != m {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the gradient of a pointwise function of {m} outputs has {} entries",
                    g.len()
                )));
            }
            values[(p, c)] = v;
            if let Some(gm) = grads.as_mut() {
                for (t, gt) in g.iter().enumerate() {
                    gm[(t * n + p, c)] = *gt;
                }
            }
        }
    }
    Ok((values, grads))
}

/// The pullback of a pointwise map with gradients `grads` (`(m·n) × S`).
pub(crate) fn pointwise_pullback(grads: Matrix, n: usize, m: usize) -> SamplePullback {
    Box::new(move |bar: &Matrix| {
        let mut out = grads;
        for t in 0..m {
            for p in 0..n {
                for c in 0..bar.ncols() {
                    out[(t * n + p, c)] *= bar[(p, c)];
                }
            }
        }
        Ok((out, None))
    })
}

impl McObjective for GenericMcObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let (values, grads) = apply_pointwise(self.f.as_ref(), samples, with_grad)?;
        let pullback =
            grads.map(|g| pointwise_pullback(g, samples.num_points(), samples.num_outputs()));
        Ok(SampleValues { values, pullback })
    }

    fn is_pointwise(&self) -> bool {
        true
    }
}

/// Feasibility-weighted objective (BoTorch's legacy `ConstrainedMCObjective`):
///
/// ```text
/// (max(objective(Y) + M, 0) · Π_i sigmoid(-constraint_i(Y) / eta_i)) - M
/// ```
///
/// with the infeasible cost `M` (BoTorch's `apply_constraints`). The objective
/// is evaluated without `X` (BoTorch calls it with `X=None`). BoTorch
/// recommends passing outcome constraints to the acquisition functions
/// instead (`constraints=...`), and its LogEI acquisition functions reject
/// this objective.
#[derive(Clone, Debug)]
pub struct ConstrainedMcObjective {
    objective: Arc<dyn McObjective>,
    constraints: OutcomeConstraints,
    infeasible_cost: f64,
}

impl ConstrainedMcObjective {
    /// The feasibility-weighted `objective` with outcome `constraints` (one
    /// temperature `eta` per constraint; BoTorch's default `eta` is `1e-3`)
    /// and infeasible cost `infeasible_cost` (BoTorch's default 0).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for a non-finite infeasible cost and
    /// for invalid temperatures (see [`OutcomeConstraints::new`]).
    pub fn new(
        objective: Arc<dyn McObjective>,
        constraints: Vec<Arc<dyn McConstraint>>,
        infeasible_cost: f64,
        eta: Vec<f64>,
    ) -> Result<Self> {
        if !infeasible_cost.is_finite() {
            return Err(BotorchError::InvalidArgument(format!(
                "infeasible_cost must be finite, got {infeasible_cost}"
            )));
        }
        Ok(Self {
            objective,
            constraints: OutcomeConstraints::new(constraints, eta)?,
            infeasible_cost,
        })
    }

    /// The infeasible cost `M`.
    pub fn infeasible_cost(&self) -> f64 {
        self.infeasible_cost
    }

    /// The outcome constraints.
    pub fn constraints(&self) -> &OutcomeConstraints {
        &self.constraints
    }
}

impl McObjective for ConstrainedMcObjective {
    fn evaluate(
        &self,
        samples: &McSamples,
        _x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let obj = self.objective.evaluate(samples, None, with_grad)?;
        let n = samples.num_points();
        if obj.values.nrows() != n {
            return Err(q_batch_mismatch(obj.values.nrows(), n));
        }
        let fat = vec![Some(false); self.constraints.len()];
        let ind = compute_smoothed_feasibility_indicator(
            &self.constraints,
            &fat,
            samples,
            false,
            with_grad,
        )?;
        let m_cost = self.infeasible_cost;
        // obj.add(M).clamp_min(0).mul(w).add(-M)
        let shifted = obj.values.map(|v| v + m_cost);
        let clamped = shifted.map(|v| v.max(0.0));
        let values = clamped.component_mul(&ind.values).map(|v| v + -m_cost);
        let pullback: Option<SamplePullback> = if with_grad {
            let obj_pullback = obj.pullback.expect("pullback requested");
            let w = ind.values.clone();
            Some(Box::new(move |bar: &Matrix| {
                // d/d obj: w where obj + M >= 0 (clamp_min's derivative).
                let obj_bar = Matrix::from_fn(bar.nrows(), bar.ncols(), |p, c| {
                    if shifted[(p, c)] >= 0.0 {
                        bar[(p, c)] * w[(p, c)]
                    } else {
                        0.0
                    }
                });
                let w_bar = bar.component_mul(&clamped);
                let (mut samples_bar, _) = obj_pullback(&obj_bar)?;
                samples_bar += ind.backward(&w_bar)?;
                Ok((samples_bar, None))
            }))
        } else {
            None
        };
        Ok(SampleValues { values, pullback })
    }

    fn is_pointwise(&self) -> bool {
        self.objective.is_pointwise() && self.constraints.is_pointwise()
    }

    fn is_constrained(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::objective::{FnOutcomeConstraint, LinearOutcomeConstraint};

    fn samples() -> McSamples {
        // S = 3 samples, n = 2 points, m = 2 outputs.
        let row_major: Vec<f64> = (0..12).map(|i| ((i * 7) % 11) as f64 / 5.0 - 1.0).collect();
        McSamples::from_row_major(&row_major, 3, 2, 2).unwrap()
    }

    #[test]
    fn layouts_round_trip() {
        let s = samples();
        assert_eq!(
            (s.num_samples(), s.num_points(), s.num_outputs()),
            (3, 2, 2)
        );
        let rm = s.to_row_major();
        assert_eq!(McSamples::from_row_major(&rm, 3, 2, 2).unwrap(), s);
        assert_eq!(s.value(2, 1, 0), rm[(2 * 2 + 1) * 2]);
        let p = s.points(1..2);
        assert_eq!(p.value(1, 0, 1), s.value(1, 1, 1));
        let adj = Matrix::from_element(2, 3, 1.0);
        let full = s.embed_points_adjoint(1..2, &adj);
        assert_eq!(full.row(0).sum(), 0.0);
        assert_eq!(full.row(1).sum(), 3.0);
        assert!(McSamples::new(Matrix::zeros(3, 2), 2, 2).is_err());
    }

    /// Central differences of `sum(bar ⊙ objective(samples))` with respect to
    /// every sample value.
    fn check_pullback(obj: &dyn McObjective, s: &McSamples) {
        let bar = Matrix::from_fn(s.num_points(), s.num_samples(), |p, c| {
            0.3 + 0.2 * p as f64 - 0.1 * c as f64
        });
        let out = obj.evaluate(s, None, true).unwrap();
        let (adj, _) = (out.pullback.unwrap())(&bar).unwrap();
        let rm = s.to_row_major();
        let h = 1e-6;
        for i in 0..rm.len() {
            let f = |delta: f64| {
                let mut v = rm.clone();
                v[i] += delta;
                let sp = McSamples::from_row_major(&v, 3, 2, 2).unwrap();
                obj.evaluate(&sp, None, false)
                    .unwrap()
                    .values
                    .component_mul(&bar)
                    .sum()
            };
            let fd = (f(h) - f(-h)) / (2.0 * h);
            let (c, rest) = (i / 4, i % 4);
            let (p, t) = (rest / 2, rest % 2);
            let an = adj[(t * 2 + p, c)];
            assert!((fd - an).abs() < 1e-6, "{i}: {an} vs {fd}");
        }
    }

    #[test]
    fn objectives_and_gradients() {
        let s = samples();
        let lin = LinearMcObjective::new(Vector::from_vec(vec![0.7, -1.3])).unwrap();
        let v = lin.evaluate(&s, None, false).unwrap().values;
        assert!((v[(1, 2)] - (0.7 * s.value(2, 1, 0) - 1.3 * s.value(2, 1, 1))).abs() < 1e-15);
        check_pullback(&lin, &s);
        assert!(matches!(
            LinearMcObjective::new(Vector::from_vec(vec![1.0]))
                .unwrap()
                .evaluate(&s, None, false),
            Err(BotorchError::Runtime(_))
        ));
        assert!(IdentityMcObjective.evaluate(&s, None, false).is_err());
        let gen = GenericMcObjective::new(|y| (y[0] - y[1] * y[1], vec![1.0, -2.0 * y[1]]));
        check_pullback(&gen, &s);
        let cons: Vec<Arc<dyn McConstraint>> = vec![
            Arc::new(LinearOutcomeConstraint::new(Vector::from_vec(vec![0.0, 1.0]), -0.2).unwrap()),
            Arc::new(FnOutcomeConstraint::new(|y| {
                (y[0] * y[1] - 0.1, vec![y[1], y[0]])
            })),
        ];
        let con = ConstrainedMcObjective::new(Arc::new(gen), cons, 0.8, vec![0.3, 0.5]).unwrap();
        assert!(con.is_constrained() && con.is_pointwise());
        check_pullback(&con, &s);
        // obj + M < 0 is clamped to zero before weighting: the value is -M.
        let lo = ConstrainedMcObjective::new(
            Arc::new(LinearMcObjective::new(Vector::from_vec(vec![1.0, 0.0])).unwrap()),
            vec![Arc::new(
                LinearOutcomeConstraint::new(Vector::from_vec(vec![0.0, 1.0]), 0.0).unwrap(),
            )],
            0.0,
            vec![1e-3],
        )
        .unwrap();
        let v = lo.evaluate(&s, None, false).unwrap().values;
        for p in 0..2 {
            for c in 0..3 {
                if s.value(c, p, 0) < 0.0 {
                    assert_eq!(v[(p, c)], 0.0);
                }
            }
        }
    }
}
