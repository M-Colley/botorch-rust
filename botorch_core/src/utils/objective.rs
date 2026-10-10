//! Outcome constraints of Monte-Carlo acquisition functions
//! (`botorch.utils.objective`): the [`McConstraint`] trait,
//! [`OutcomeConstraints`] (BoTorch's `constraints`, `eta` and `fat`
//! arguments), [`compute_feasibility_indicator`] and
//! [`compute_smoothed_feasibility_indicator`].
//!
//! A constraint maps posterior samples to a slack value per point and MC
//! sample; the point is feasible in that sample iff the slack is `≤ 0`.
//! Acquisition functions weight their utilities with a smooth approximation
//! of the feasibility indicator,
//!
//! ```text
//! ind = Π_i sigmoid(-c_i(Y) / eta_i)      (log ind = Σ_i log sigmoid(-c_i / eta_i))
//! ```
//!
//! with the logistic sigmoid (`fat = false`) or the fat-tailed `fatmoid`
//! (`fat = true`), or `c_i(Y)` itself for constraints that return
//! probabilities of feasibility (`fat = None`).

use std::fmt::Debug;
use std::sync::Arc;

use nalgebra::DMatrix;

use crate::acquisition::mc_objective::{
    apply_pointwise, pointwise_pullback, McSamples, PointwiseFn, SamplePullback, SampleValues,
};
use crate::error::{BotorchError, Result};
use crate::utils::safe_math::{log_fatmoid_with_grad, logexpit, logexpit_grad};
use crate::{Matrix, Vector};

/// An outcome constraint (one of BoTorch's `constraints` callables): maps
/// posterior samples of `n` points to slack values (`n × S`), feasible iff
/// `≤ 0` (or, for constraints used with `fat = None`, to probabilities of
/// feasibility).
pub trait McConstraint: Send + Sync + Debug {
    /// The slack values (`n × S`) of `samples`; with `with_grad`, also the
    /// pullback (whose input adjoint is always `None`).
    ///
    /// # Errors
    /// Implementation-specific.
    fn evaluate(&self, samples: &McSamples, with_grad: bool) -> Result<SampleValues>;

    /// Whether the value at a point depends only on the samples at that
    /// point. Defaults to `false`.
    fn is_pointwise(&self) -> bool {
        false
    }
}

/// A linear outcome constraint `Σ_t weights[t] · Y[..., t] + offset ≤ 0`
/// (e.g. `Y[..., 1] - 0.5` is `weights = e_1`, `offset = -0.5`).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearOutcomeConstraint {
    weights: Vector,
    offset: f64,
}

impl LinearOutcomeConstraint {
    /// The constraint with one weight per output.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for empty or non-finite weights or a
    /// non-finite offset.
    pub fn new(weights: Vector, offset: f64) -> Result<Self> {
        if weights.is_empty() || weights.iter().any(|w| !w.is_finite()) || !offset.is_finite() {
            return Err(BotorchError::InvalidArgument(
                "weights must be non-empty and finite and the offset finite".into(),
            ));
        }
        Ok(Self { weights, offset })
    }
}

impl McConstraint for LinearOutcomeConstraint {
    fn evaluate(&self, samples: &McSamples, with_grad: bool) -> Result<SampleValues> {
        let (n, m, s) = (
            samples.num_points(),
            samples.num_outputs(),
            samples.num_samples(),
        );
        if m != self.weights.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "the constraint has {} weights but the samples have {m} outputs",
                self.weights.len()
            )));
        }
        let data = samples.data();
        let mut values = Matrix::zeros(n, s);
        for c in 0..s {
            for p in 0..n {
                let mut v = 0.0;
                for t in 0..m {
                    v += self.weights[t] * data[(t * n + p, c)];
                }
                values[(p, c)] = v + self.offset;
            }
        }
        let pullback: Option<SamplePullback> = if with_grad {
            let w = self.weights.clone();
            Some(Box::new(move |bar: &Matrix| {
                let mut out = Matrix::zeros(m * n, bar.ncols());
                for t in 0..m {
                    out.rows_mut(t * n, n).copy_from(&(bar * w[t]));
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

/// An outcome constraint given by a function of the outputs at one point,
/// `f(y) -> (slack, ∂slack/∂y)`, applied to every point and MC sample.
#[derive(Clone)]
pub struct FnOutcomeConstraint {
    f: Arc<PointwiseFn>,
}

impl Debug for FnOutcomeConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FnOutcomeConstraint")
    }
}

impl FnOutcomeConstraint {
    /// The constraint `f(y) ≤ 0` (`f` returns the slack and its gradient).
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>) + Send + Sync + 'static,
    {
        Self { f: Arc::new(f) }
    }
}

impl McConstraint for FnOutcomeConstraint {
    fn evaluate(&self, samples: &McSamples, with_grad: bool) -> Result<SampleValues> {
        let (values, grads) = apply_pointwise(self.f.as_ref(), samples, with_grad)?;
        let pullback =
            grads.map(|g| pointwise_pullback(g, samples.num_points(), samples.num_outputs()));
        Ok(SampleValues { values, pullback })
    }

    fn is_pointwise(&self) -> bool {
        true
    }
}

/// Outcome constraints with their temperatures (BoTorch's `constraints` and
/// `eta` arguments of the MC acquisition functions). How the feasibility
/// indicator is smoothed (BoTorch's `fat`) is a property of the acquisition
/// function and is passed to [`compute_smoothed_feasibility_indicator`].
#[derive(Clone, Debug)]
pub struct OutcomeConstraints {
    constraints: Vec<Arc<dyn McConstraint>>,
    eta: Vec<f64>,
}

impl OutcomeConstraints {
    /// Constraints with one temperature `eta[i] > 0` per constraint
    /// (BoTorch's tensor-valued `eta`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] with BoTorch's messages for a
    /// mismatched number of temperatures or non-positive temperatures.
    pub fn new(constraints: Vec<Arc<dyn McConstraint>>, eta: Vec<f64>) -> Result<Self> {
        if eta.len() != constraints.len() {
            return Err(BotorchError::InvalidArgument(
                "Number of provided constraints and number of provided etas do not match.".into(),
            ));
        }
        if eta.iter().any(|&e| e.is_nan() || e <= 0.0) {
            return Err(BotorchError::InvalidArgument(
                "eta must be positive.".into(),
            ));
        }
        Ok(Self { constraints, eta })
    }

    /// Constraints that share the temperature `eta` (BoTorch's scalar `eta`,
    /// default `1e-3`).
    ///
    /// # Errors
    /// See [`OutcomeConstraints::new`].
    pub fn uniform(constraints: Vec<Arc<dyn McConstraint>>, eta: f64) -> Result<Self> {
        let k = constraints.len();
        Self::new(constraints, vec![eta; k])
    }

    /// The constraints.
    pub fn constraints(&self) -> &[Arc<dyn McConstraint>] {
        &self.constraints
    }

    /// The temperatures.
    pub fn eta(&self) -> &[f64] {
        &self.eta
    }

    /// Number of constraints.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Whether there are no constraints.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Whether all constraints are pointwise.
    pub fn is_pointwise(&self) -> bool {
        self.constraints.iter().all(|c| c.is_pointwise())
    }
}

/// Checks one smoothing kind per constraint (BoTorch's `fat` list).
pub(crate) fn check_fat(constraints: &OutcomeConstraints, fat: &[Option<bool>]) -> Result<()> {
    if fat.len() != constraints.len() {
        return Err(BotorchError::InvalidArgument(
            "Number of provided constraints and number of provided fats do not match.".into(),
        ));
    }
    Ok(())
}

/// Evaluates a constraint and checks the shape of its values.
fn evaluate_constraint(
    constraint: &dyn McConstraint,
    samples: &McSamples,
    with_grad: bool,
) -> Result<SampleValues> {
    let v = constraint.evaluate(samples, with_grad)?;
    if v.values.shape() != (samples.num_points(), samples.num_samples()) {
        return Err(BotorchError::DimensionMismatch(format!(
            "a constraint returned values of shape {:?}, expected {} points x {} samples",
            v.values.shape(),
            samples.num_points(),
            samples.num_samples()
        )));
    }
    Ok(v)
}

/// Hard feasibility of every point in every sample (`n × S`): `true` iff all
/// constraints are `≤ 0` (BoTorch's `compute_feasibility_indicator`; NaN
/// values are infeasible).
///
/// # Errors
/// Errors of the constraints.
pub fn compute_feasibility_indicator(
    constraints: &[Arc<dyn McConstraint>],
    samples: &McSamples,
) -> Result<DMatrix<bool>> {
    let (n, s) = (samples.num_points(), samples.num_samples());
    let mut feasible = DMatrix::from_element(n, s, true);
    for c in constraints {
        let v = evaluate_constraint(c.as_ref(), samples, false)?.values;
        for (f, &x) in feasible.iter_mut().zip(v.iter()) {
            *f = *f && x <= 0.0;
        }
    }
    Ok(feasible)
}

/// The smoothed feasibility indicator of [`compute_smoothed_feasibility_indicator`]
/// together with what is needed to backpropagate through it.
pub struct SmoothedFeasibility {
    /// The indicator values (`n × S`), or their logarithm.
    pub values: Matrix,
    log: bool,
    /// Per constraint, `∂ log ind / ∂ c_i` (`n × S`).
    dlog: Vec<Matrix>,
    pullbacks: Vec<SamplePullback>,
    samples_shape: (usize, usize),
}

impl Debug for SmoothedFeasibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmoothedFeasibility")
            .field("values", &self.values)
            .field("log", &self.log)
            .finish()
    }
}

impl SmoothedFeasibility {
    /// Maps the adjoint of [`SmoothedFeasibility::values`] (`n × S`) to the
    /// adjoint of the samples (`(m·n) × S`).
    ///
    /// # Errors
    /// Errors of the constraints' pullbacks (or if derivatives were not
    /// requested).
    pub fn backward(self, adjoint: &Matrix) -> Result<Matrix> {
        if self.pullbacks.len() != self.dlog.len() {
            return Err(BotorchError::Numerical(
                "the feasibility indicator was computed without derivatives".into(),
            ));
        }
        let log_bar = if self.log {
            adjoint.clone()
        } else {
            adjoint.component_mul(&self.values)
        };
        let mut out = Matrix::zeros(self.samples_shape.0, self.samples_shape.1);
        for (dl, pb) in self.dlog.iter().zip(self.pullbacks) {
            let (sb, _) = pb(&log_bar.component_mul(dl))?;
            out += sb;
        }
        Ok(out)
    }
}

/// Smoothed feasibility indicator of `constraints` for every point and
/// sample (BoTorch's `compute_smoothed_feasibility_indicator`):
/// `exp(Σ_i log sigmoid_i(-c_i / eta_i))`, or the sum itself if `log`, where
/// `fat[i]` selects the logistic sigmoid (`Some(false)`), the fat-tailed
/// `fatmoid` (`Some(true)`) or no transformation for constraints that return
/// probabilities of feasibility (`None`, contributing `log c_i`).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] unless there is one `fat` entry per
/// constraint, and errors of the constraints.
pub fn compute_smoothed_feasibility_indicator(
    constraints: &OutcomeConstraints,
    fat: &[Option<bool>],
    samples: &McSamples,
    log: bool,
    with_grad: bool,
) -> Result<SmoothedFeasibility> {
    check_fat(constraints, fat)?;
    let (n, s) = (samples.num_points(), samples.num_samples());
    let mut total = Matrix::zeros(n, s);
    let mut dlog = Vec::with_capacity(constraints.len());
    let mut pullbacks = Vec::with_capacity(constraints.len());
    for ((c, &eta), &fat) in constraints
        .constraints
        .iter()
        .zip(&constraints.eta)
        .zip(fat)
    {
        let v = evaluate_constraint(c.as_ref(), samples, with_grad)?;
        let mut d = Matrix::zeros(n, s);
        for ((t, dd), &x) in total.iter_mut().zip(d.iter_mut()).zip(v.values.iter()) {
            match fat {
                // Probabilities of feasibility: log(c).
                None => {
                    *t += x.ln();
                    *dd = 1.0 / x;
                }
                Some(fat) => {
                    let z = -x / eta;
                    let (l, g) = if fat {
                        log_fatmoid_with_grad(z, 1.0)
                    } else {
                        (logexpit(z), logexpit_grad(z))
                    };
                    *t += l;
                    *dd = -(g / eta);
                }
            }
        }
        if with_grad {
            dlog.push(d);
            pullbacks.push(v.pullback.expect("pullback requested"));
        }
    }
    let values = if log { total } else { total.map(f64::exp) };
    Ok(SmoothedFeasibility {
        values,
        log,
        dlog,
        pullbacks,
        samples_shape: samples.data().shape(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::safe_math::{log_fatmoid, sigmoid};

    fn samples() -> McSamples {
        let row_major: Vec<f64> = (0..12).map(|i| ((i * 5) % 7) as f64 / 3.0 - 1.0).collect();
        McSamples::from_row_major(&row_major, 3, 2, 2).unwrap()
    }

    #[test]
    fn indicators_match_their_definitions() {
        let s = samples();
        let c0: Arc<dyn McConstraint> =
            Arc::new(LinearOutcomeConstraint::new(Vector::from_vec(vec![0.0, 1.0]), -0.2).unwrap());
        let c1: Arc<dyn McConstraint> = Arc::new(FnOutcomeConstraint::new(|y| {
            (y[0] * y[0] - 0.5, vec![2.0 * y[0], 0.0])
        }));
        let oc = OutcomeConstraints::new(vec![c0.clone(), c1.clone()], vec![0.1, 0.4]).unwrap();
        let fat = [Some(false), Some(true)];
        let ind = compute_smoothed_feasibility_indicator(&oc, &fat, &s, false, false).unwrap();
        let log_ind = compute_smoothed_feasibility_indicator(&oc, &fat, &s, true, false).unwrap();
        let hard = compute_feasibility_indicator(oc.constraints(), &s).unwrap();
        for c in 0..3 {
            for p in 0..2 {
                let a = s.value(c, p, 1) - 0.2;
                let b = s.value(c, p, 0).powi(2) - 0.5;
                let e = sigmoid(-a / 0.1) * log_fatmoid(-b / 0.4, 1.0).exp();
                assert!((ind.values[(p, c)] - e).abs() < 1e-14);
                assert!((log_ind.values[(p, c)] - e.ln()).abs() < 1e-12);
                assert_eq!(hard[(p, c)], a <= 0.0 && b <= 0.0);
            }
        }
        // Gradients against central differences.
        for log in [false, true] {
            let ind = compute_smoothed_feasibility_indicator(&oc, &fat, &s, log, true).unwrap();
            let bar = Matrix::from_fn(2, 3, |p, c| 1.0 + 0.5 * p as f64 - 0.3 * c as f64);
            let adj = ind.backward(&bar).unwrap();
            let rm = s.to_row_major();
            for i in 0..rm.len() {
                let f = |delta: f64| {
                    let mut v = rm.clone();
                    v[i] += delta;
                    let sp = McSamples::from_row_major(&v, 3, 2, 2).unwrap();
                    compute_smoothed_feasibility_indicator(&oc, &fat, &sp, log, false)
                        .unwrap()
                        .values
                        .component_mul(&bar)
                        .sum()
                };
                let h = 1e-6;
                let fd = (f(h) - f(-h)) / (2.0 * h);
                let (c, rest) = (i / 4, i % 4);
                let (p, t) = (rest / 2, rest % 2);
                let an = adj[(t * 2 + p, c)];
                assert!(
                    (fd - an).abs() < 1e-5 * fd.abs().max(1.0),
                    "{i}: {an} vs {fd}"
                );
            }
        }
        // Probability constraints (`fat = None`) contribute log(c).
        let prob: Arc<dyn McConstraint> = Arc::new(FnOutcomeConstraint::new(|y| {
            (
                0.5 + 0.25 * y[0].tanh(),
                vec![0.25 * (1.0 - y[0].tanh().powi(2)), 0.0],
            )
        }));
        let po = OutcomeConstraints::uniform(vec![prob], 1e-3).unwrap();
        let v = compute_smoothed_feasibility_indicator(&po, &[None], &s, true, false).unwrap();
        assert!((v.values[(0, 1)] - (0.5 + 0.25 * s.value(1, 0, 0).tanh()).ln()).abs() < 1e-15);
        // Validation with BoTorch's messages.
        let e = OutcomeConstraints::new(vec![c0.clone()], vec![]).unwrap_err();
        assert!(e.to_string().contains("etas do not match"));
        let e = OutcomeConstraints::uniform(vec![c0.clone()], 0.0).unwrap_err();
        assert!(e.to_string().contains("eta must be positive"));
        let ok = OutcomeConstraints::uniform(vec![c0], 1e-3).unwrap();
        let e = compute_smoothed_feasibility_indicator(&ok, &[], &s, true, false).unwrap_err();
        assert!(e.to_string().contains("fats do not match"));
        assert!(OutcomeConstraints::uniform(vec![], 1e-3)
            .unwrap()
            .is_empty());
    }
}
