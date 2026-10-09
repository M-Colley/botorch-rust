//! Parameter constraints mirroring `gpytorch.constraints.GreaterThan` and
//! `gpytorch.constraints.Positive`.
//!
//! A constraint maps an unconstrained *raw* parameter (the value that is
//! stored and optimized) to the *actual* parameter value. Two flavours exist:
//!
//! * **enforced** (`transform = Some(Softplus)`, GPyTorch's default):
//!   `actual = softplus(raw) + lower_bound`; the raw parameter is unbounded.
//! * **not enforced** (`transform = None`, used by all of BoTorch's
//!   `SingleTaskGP` defaults): `actual = raw`, and the bound is instead
//!   imposed by the optimizer (L-BFGS-B box constraints), exactly like
//!   `botorch.optim.utils.get_parameters_and_bounds`.

/// The transform used by an enforced constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterTransform {
    /// `torch.nn.Softplus()` (beta = 1, threshold = 20) with inverse
    /// `x + log(-expm1(-x))`.
    Softplus,
}

impl ParameterTransform {
    /// Applies the transform.
    pub fn apply(self, x: f64) -> f64 {
        match self {
            ParameterTransform::Softplus => softplus(x),
        }
    }

    /// Derivative of the transform at `x`.
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            ParameterTransform::Softplus => softplus_derivative(x),
        }
    }

    /// Applies the inverse transform.
    pub fn inverse(self, y: f64) -> f64 {
        match self {
            ParameterTransform::Softplus => inv_softplus(y),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
