//! Parameter constraints mirroring `gpytorch.constraints.GreaterThan`,
//! `gpytorch.constraints.Positive` and `gpytorch.constraints.Interval`.
//!
//! A constraint maps an unconstrained *raw* parameter (the value that is
//! stored and optimized) to the *actual* parameter value. Two flavours exist:
//!
//! * **enforced** (a transform, GPyTorch's default): `GreaterThan` maps
//!   `actual = softplus(raw) + lower_bound`, `Interval` maps
//!   `actual = sigmoid(raw) · (upper - lower) + lower`; the raw parameter is
//!   unbounded.
//! * **not enforced** (`transform = None`, used by all of BoTorch's
//!   `SingleTaskGP` defaults and by the task kernel of `LCEMGP`):
//!   `actual = raw`, and the bounds are instead imposed by the optimizer
//!   (L-BFGS-B box constraints), exactly like
//!   `botorch.optim.utils.get_parameters_and_bounds`.
//!
//! [`Constraint`] is either kind; the covariance modules and likelihoods hold
//! a [`Constraint`].

use crate::error::{BotorchError, Result};

/// The transform used by an enforced constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterTransform {
    /// `torch.nn.Softplus()` (beta = 1, threshold = 20) with inverse
    /// `x + log(-expm1(-x))`.
    Softplus,
    /// `torch.sigmoid` with inverse `log(x) - log(1 - x)`
    /// (`gpytorch.utils.transforms.inv_sigmoid`).
    Sigmoid,
}

impl ParameterTransform {
    /// Applies the transform.
    pub fn apply(self, x: f64) -> f64 {
        match self {
            ParameterTransform::Softplus => softplus(x),
            ParameterTransform::Sigmoid => sigmoid(x),
        }
    }

    /// Derivative of the transform at `x`.
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            ParameterTransform::Softplus => softplus_derivative(x),
            ParameterTransform::Sigmoid => {
                let s = sigmoid(x);
                (1.0 - s) * s
            }
        }
    }

    /// Applies the inverse transform.
    pub fn inverse(self, y: f64) -> f64 {
        match self {
            ParameterTransform::Softplus => inv_softplus(y),
            ParameterTransform::Sigmoid => inv_sigmoid(y),
        }
    }

    /// The inverse transform evaluated in single precision, as GPyTorch
    /// computes the raw initial value of a constraint created from Python
    /// floats (float32 tensors).
    fn inverse_f32(self, y: f32) -> f32 {
        match self {
            ParameterTransform::Softplus => y + (-(-y).exp_m1()).ln(),
            ParameterTransform::Sigmoid => y.ln() - (1.0 - y).ln(),
        }
    }

    /// The name of the transform (`"softplus"` or `"sigmoid"`).
    pub fn name(self) -> &'static str {
        match self {
            ParameterTransform::Softplus => "softplus",
            ParameterTransform::Sigmoid => "sigmoid",
        }
    }
}

/// `torch.sigmoid`: `1 / (1 + exp(-x))`.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `gpytorch.utils.transforms.inv_sigmoid`: `log(x) - log(1 - x)`.
pub fn inv_sigmoid(x: f64) -> f64 {
    x.ln() - (1.0 - x).ln()
}

/// `torch.nn.functional.softplus` with `beta = 1` and `threshold = 20`.
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// Derivative of [`softplus`] (as computed by torch's softplus backward).
pub fn softplus_derivative(x: f64) -> f64 {
    if x > 20.0 {
        1.0
    } else {
        let z = x.exp();
        z / (z + 1.0)
    }
}

/// `gpytorch.utils.transforms.inv_softplus`: `x + log(-expm1(-x))`.
pub fn inv_softplus(x: f64) -> f64 {
    x + (-(-x).exp_m1()).ln()
}

/// Rounds a value to single precision and back, mirroring how PyTorch
/// materializes Python floats in its default dtype (`torch.float32`) before a
/// model is cast to double precision.
pub fn round_to_f32(x: f64) -> f64 {
    x as f32 as f64
}

/// A lower-bound constraint (`gpytorch.constraints.GreaterThan`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GreaterThan {
    lower_bound: f64,
    transform: Option<ParameterTransform>,
    initial_value: Option<f64>,
}

impl GreaterThan {
    /// Creates a constraint `actual >= lower_bound`.
    ///
    /// `initial_value` is given in the actual (constrained) space; like
    /// GPyTorch, a model initializes the parameter to it.
    ///
    /// The bound is stored exactly. GPyTorch materializes bounds in torch's
    /// default dtype (float32); use [`GreaterThan::rounded_to_f32`] to
    /// reproduce that rounding.
    pub fn new(
        lower_bound: f64,
        transform: Option<ParameterTransform>,
        initial_value: Option<f64>,
    ) -> Self {
        Self {
            lower_bound,
            transform,
            initial_value,
        }
    }

    /// `gpytorch.constraints.Positive()`: lower bound 0 enforced via softplus.
    pub fn positive() -> Self {
        Self::new(0.0, Some(ParameterTransform::Softplus), None)
    }

    /// Returns a copy whose bound and initial value are rounded to single
    /// precision, exactly as GPyTorch stores them when the constraint is
    /// created from Python floats.
    pub fn rounded_to_f32(self) -> Self {
        Self {
            lower_bound: round_to_f32(self.lower_bound),
            transform: self.transform,
            initial_value: self.initial_value.map(round_to_f32),
        }
    }

    /// The lower bound in actual space.
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    /// The transform (None means the constraint is not enforced).
    pub fn transform_kind(&self) -> Option<ParameterTransform> {
        self.transform
    }

    /// The initial value in actual space, if any.
    pub fn initial_value(&self) -> Option<f64> {
        self.initial_value
    }

    /// Whether the constraint is enforced through a transform.
    pub fn enforced(&self) -> bool {
        self.transform.is_some()
    }

    /// Maps a raw parameter value to the actual parameter value.
    pub fn transform(&self, raw: f64) -> f64 {
        match self.transform {
            Some(t) => t.apply(raw) + self.lower_bound,
            None => raw,
        }
    }

    /// Derivative `d actual / d raw`.
    pub fn transform_derivative(&self, raw: f64) -> f64 {
        match self.transform {
            Some(t) => t.derivative(raw),
            None => 1.0,
        }
    }

    /// Maps an actual parameter value to the raw parameter value.
    pub fn inverse_transform(&self, actual: f64) -> f64 {
        match self.transform {
            Some(t) => t.inverse(actual - self.lower_bound),
            None => actual,
        }
    }

    /// The raw initial value implied by [`GreaterThan::initial_value`].
    pub fn initial_raw_value(&self) -> Option<f64> {
        self.initial_value.map(|v| self.inverse_transform(v))
    }

    /// Bounds on the *raw* parameter used by the optimizer, as computed by
    /// `botorch.optim.utils.get_parameters_and_bounds`: the inverse transform
    /// of the constraint bounds (`(-inf, inf)` for enforced constraints).
    pub fn raw_bounds(&self) -> (f64, f64) {
        (
            self.inverse_transform(self.lower_bound),
            self.inverse_transform(f64::INFINITY),
        )
    }
}

/// A two-sided constraint (`gpytorch.constraints.Interval`):
/// `lower_bound <= actual <= upper_bound`.
///
/// Enforced through a transform (GPyTorch's default `sigmoid`), the actual
/// value is `transform(raw) · (upper - lower) + lower` and the raw parameter
/// is unbounded. Not enforced (`transform = None`), `actual = raw` and the
/// optimizer keeps the raw parameter in `[lower, upper]` (L-BFGS-B box
/// bounds, as BoTorch's `fit_gpytorch_mll` does via
/// `get_parameters_and_bounds`).
///
/// The inverse used by [`Interval::inverse_transform`] is stored separately,
/// as in GPyTorch: `Interval(lower, upper, transform=softplus)` keeps the
/// default inverse `inv_sigmoid` unless another one is given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    lower_bound: f64,
    upper_bound: f64,
    transform: Option<ParameterTransform>,
    inverse: ParameterTransform,
    initial_value: Option<f64>,
    initial_raw: Option<f64>,
}

impl Interval {
    /// Creates `Interval(lower_bound, upper_bound, transform, inv_transform,
    /// initial_value)`; `inverse = None` selects GPyTorch's default inverse
    /// (`inv_sigmoid`). `initial_value` is in the actual space.
    ///
    /// The bounds are stored exactly; [`Interval::rounded_to_f32`] reproduces
    /// GPyTorch's single-precision bounds and raw initial value.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] with GPyTorch's messages if
    /// `lower_bound >= upper_bound` or a bound is not finite (use
    /// [`GreaterThan`] for one-sided constraints), and if `initial_value` is
    /// not finite or lies outside the bounds.
    pub fn new(
        lower_bound: f64,
        upper_bound: f64,
        transform: Option<ParameterTransform>,
        inverse: Option<ParameterTransform>,
        initial_value: Option<f64>,
    ) -> Result<Self> {
        if lower_bound.is_nan() || upper_bound.is_nan() || lower_bound >= upper_bound {
            return Err(BotorchError::InvalidArgument(
                "Got parameter bounds with empty intervals.".into(),
            ));
        }
        if !(lower_bound.is_finite() && upper_bound.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "Cannot make an Interval directly with non-finite bounds. Use a derived class \
                 like GreaterThan or LessThan instead."
                    .into(),
            ));
        }
        if let Some(v) = initial_value {
            if !(v.is_finite() && v >= lower_bound && v <= upper_bound) {
                return Err(BotorchError::InvalidArgument(format!(
                    "initial_value {v} is outside the interval [{lower_bound}, {upper_bound}]"
                )));
            }
        }
        let mut c = Self {
            lower_bound,
            upper_bound,
            transform,
            inverse: inverse.unwrap_or(ParameterTransform::Sigmoid),
            initial_value,
            initial_raw: None,
        };
        c.initial_raw = initial_value.map(|v| c.inverse_transform(v));
        Ok(c)
    }

    /// Returns a copy that behaves like a GPyTorch `Interval` created from
    /// Python floats: the bounds and the initial value are float32 tensors,
    /// and the raw initial value is computed from them in single precision.
    pub fn rounded_to_f32(self) -> Self {
        let (l, u) = (self.lower_bound as f32, self.upper_bound as f32);
        let initial_raw = self.initial_value.map(|v| {
            let v = v as f32;
            match self.transform {
                None => v as f64,
                Some(_) => self.inverse.inverse_f32((v - l) / (u - l)) as f64,
            }
        });
        Self {
            lower_bound: l as f64,
            upper_bound: u as f64,
            transform: self.transform,
            inverse: self.inverse,
            initial_value: self.initial_value.map(round_to_f32),
            initial_raw,
        }
    }

    /// The lower bound in actual space.
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    /// The upper bound in actual space.
    pub fn upper_bound(&self) -> f64 {
        self.upper_bound
    }

    /// The transform (None means the constraint is not enforced).
    pub fn transform_kind(&self) -> Option<ParameterTransform> {
        self.transform
    }

    /// The transform whose inverse maps actual to raw values (GPyTorch's
    /// `inv_transform`).
    pub fn inverse_kind(&self) -> ParameterTransform {
        self.inverse
    }

    /// The initial value in actual space, if any.
    pub fn initial_value(&self) -> Option<f64> {
        self.initial_value
    }

    /// Whether the constraint is enforced through a transform.
    pub fn enforced(&self) -> bool {
        self.transform.is_some()
    }

    /// Maps a raw parameter value to the actual parameter value.
    pub fn transform(&self, raw: f64) -> f64 {
        match self.transform {
            Some(t) => t.apply(raw) * (self.upper_bound - self.lower_bound) + self.lower_bound,
            None => raw,
        }
    }

    /// Derivative `d actual / d raw`.
    pub fn transform_derivative(&self, raw: f64) -> f64 {
        match self.transform {
            Some(t) => (self.upper_bound - self.lower_bound) * t.derivative(raw),
            None => 1.0,
        }
    }

    /// Maps an actual parameter value to the raw parameter value.
    pub fn inverse_transform(&self, actual: f64) -> f64 {
        match self.transform {
            Some(_) => self
                .inverse
                .inverse((actual - self.lower_bound) / (self.upper_bound - self.lower_bound)),
            None => actual,
        }
    }

    /// The raw initial value implied by [`Interval::initial_value`]
    /// (GPyTorch's `constraint.initial_value`).
    pub fn initial_raw_value(&self) -> Option<f64> {
        self.initial_raw
    }

    /// Bounds on the *raw* parameter used by the optimizer
    /// (`get_parameters_and_bounds`: the inverse transform of both bounds),
    /// i.e. `(lower, upper)` if the constraint is not enforced and
    /// `(-inf, inf)` with the sigmoid transform.
    pub fn raw_bounds(&self) -> (f64, f64) {
        (
            self.inverse_transform(self.lower_bound),
            self.inverse_transform(self.upper_bound),
        )
    }
}

/// A parameter constraint: GPyTorch's `GreaterThan` / `Positive` or
/// `Interval`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Constraint {
    /// A lower bound (`GreaterThan`, `Positive`).
    GreaterThan(GreaterThan),
    /// Lower and upper bounds (`Interval`).
    Interval(Interval),
}

impl From<GreaterThan> for Constraint {
    fn from(c: GreaterThan) -> Self {
        Constraint::GreaterThan(c)
    }
}

impl From<Interval> for Constraint {
    fn from(c: Interval) -> Self {
        Constraint::Interval(c)
    }
}

impl Constraint {
    /// `gpytorch.constraints.Positive()`: lower bound 0 enforced via softplus.
    pub fn positive() -> Self {
        GreaterThan::positive().into()
    }

    /// See [`GreaterThan::rounded_to_f32`] and [`Interval::rounded_to_f32`].
    pub fn rounded_to_f32(self) -> Self {
        match self {
            Constraint::GreaterThan(c) => c.rounded_to_f32().into(),
            Constraint::Interval(c) => c.rounded_to_f32().into(),
        }
    }

    /// The lower bound in actual space.
    pub fn lower_bound(&self) -> f64 {
        match self {
            Constraint::GreaterThan(c) => c.lower_bound(),
            Constraint::Interval(c) => c.lower_bound(),
        }
    }

    /// The upper bound in actual space (infinite for `GreaterThan`).
    pub fn upper_bound(&self) -> f64 {
        match self {
            Constraint::GreaterThan(_) => f64::INFINITY,
            Constraint::Interval(c) => c.upper_bound(),
        }
    }

    /// The transform (None means the constraint is not enforced).
    pub fn transform_kind(&self) -> Option<ParameterTransform> {
        match self {
            Constraint::GreaterThan(c) => c.transform_kind(),
            Constraint::Interval(c) => c.transform_kind(),
        }
    }

    /// The initial value in actual space, if any.
    pub fn initial_value(&self) -> Option<f64> {
        match self {
            Constraint::GreaterThan(c) => c.initial_value(),
            Constraint::Interval(c) => c.initial_value(),
        }
    }

    /// Whether the constraint is enforced through a transform.
    pub fn enforced(&self) -> bool {
        self.transform_kind().is_some()
    }

    /// Maps a raw parameter value to the actual parameter value.
    pub fn transform(&self, raw: f64) -> f64 {
        match self {
            Constraint::GreaterThan(c) => c.transform(raw),
            Constraint::Interval(c) => c.transform(raw),
        }
    }

    /// Derivative `d actual / d raw`.
    pub fn transform_derivative(&self, raw: f64) -> f64 {
        match self {
            Constraint::GreaterThan(c) => c.transform_derivative(raw),
            Constraint::Interval(c) => c.transform_derivative(raw),
        }
    }

    /// Maps an actual parameter value to the raw parameter value.
    pub fn inverse_transform(&self, actual: f64) -> f64 {
        match self {
            Constraint::GreaterThan(c) => c.inverse_transform(actual),
            Constraint::Interval(c) => c.inverse_transform(actual),
        }
    }

    /// The raw initial value, if the constraint has an initial value.
    pub fn initial_raw_value(&self) -> Option<f64> {
        match self {
            Constraint::GreaterThan(c) => c.initial_raw_value(),
            Constraint::Interval(c) => c.initial_raw_value(),
        }
    }

    /// Optimizer bounds of the raw parameter (see [`GreaterThan::raw_bounds`]
    /// and [`Interval::raw_bounds`]).
    pub fn raw_bounds(&self) -> (f64, f64) {
        match self {
            Constraint::GreaterThan(c) => c.raw_bounds(),
            Constraint::Interval(c) => c.raw_bounds(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_without_transform_is_a_box_bound() {
        // LCEMGP's task kernel: Interval(0, 2, transform=None, initial_value=1).
        let c = Interval::new(0.0, 2.0, None, None, Some(1.0))
            .unwrap()
            .rounded_to_f32();
        assert!(!c.enforced());
        assert_eq!(c.transform(0.7), 0.7);
        assert_eq!(c.inverse_transform(0.7), 0.7);
        assert_eq!(c.transform_derivative(0.3), 1.0);
        assert_eq!(c.raw_bounds(), (0.0, 2.0));
        assert_eq!(c.initial_raw_value(), Some(1.0));
        let c: Constraint = c.into();
        assert_eq!(c.upper_bound(), 2.0);
        assert_eq!(c.raw_bounds(), (0.0, 2.0));
    }

    #[test]
    fn sigmoid_interval_roundtrip_and_derivative() {
        let c = Interval::new(0.1, 3.0, Some(ParameterTransform::Sigmoid), None, None).unwrap();
        assert!(c.enforced());
        assert_eq!(c.raw_bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        for &x in &[-6.0, -1.0, 0.0, 0.4, 3.0] {
            let v = c.transform(x);
            assert!(v > 0.1 && v < 3.0);
            assert!((c.inverse_transform(v) - x).abs() < 1e-9);
            let h = 1e-6;
            let fd = (c.transform(x + h) - c.transform(x - h)) / (2.0 * h);
            assert!((fd - c.transform_derivative(x)).abs() < 1e-8);
        }
        assert_eq!(c.transform(0.0), 0.5 * 2.9 + 0.1);
    }

    #[test]
    fn interval_rejects_empty_and_infinite_bounds() {
        assert!(Interval::new(1.0, 1.0, None, None, None).is_err());
        assert!(Interval::new(0.0, f64::INFINITY, None, None, None).is_err());
        assert!(Interval::new(0.0, 1.0, None, None, Some(2.0)).is_err());
    }

    #[test]
    fn float32_initial_raw_value_of_a_sigmoid_interval() {
        // GPyTorch: Interval(0.1, 3.0, initial_value=1.0)._initial_value
        // (float32 arithmetic): log(0.31034482) - log(0.6896552).
        let c = Interval::new(0.1, 3.0, Some(ParameterTransform::Sigmoid), None, Some(1.0))
            .unwrap()
            .rounded_to_f32();
        let x = (1.0f32 - 0.1f32) / (3.0f32 - 0.1f32);
        let expected = (x.ln() - (1.0f32 - x).ln()) as f64;
        assert_eq!(c.initial_raw_value(), Some(expected));
        assert_eq!(c.lower_bound(), 0.1f32 as f64);
    }

    #[test]
    fn softplus_roundtrip_and_derivative() {
        for &x in &[-30.0, -3.0, -0.1, 0.0, 0.5, 4.0, 19.0, 25.0] {
            let y = softplus(x);
            if x > -30.0 {
                assert!((inv_softplus(y) - x).abs() < 1e-9 * (1.0 + x.abs()));
            }
            let h = 1e-6;
            let fd = (softplus(x + h) - softplus(x - h)) / (2.0 * h);
            assert!((fd - softplus_derivative(x)).abs() < 1e-8);
        }
        assert_eq!(softplus(0.0), std::f64::consts::LN_2);
    }

    #[test]
    fn non_enforced_constraint_is_identity_with_box_bounds() {
        let c = GreaterThan::new(2.5e-2, None, Some(0.3)).rounded_to_f32();
        assert_eq!(c.lower_bound(), 0.02500000037252903);
        assert_eq!(c.transform(0.7), 0.7);
        assert_eq!(c.inverse_transform(0.7), 0.7);
        assert_eq!(c.raw_bounds(), (0.02500000037252903, f64::INFINITY));
        assert_eq!(c.initial_raw_value(), Some(0.3f32 as f64));
    }

    #[test]
    fn positive_constraint_is_unbounded_in_raw_space() {
        let c = GreaterThan::positive();
        assert!(c.enforced());
        assert_eq!(c.raw_bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(c.transform(0.0), std::f64::consts::LN_2);
        assert!((c.inverse_transform(c.transform(1.3)) - 1.3).abs() < 1e-14);
    }
}
