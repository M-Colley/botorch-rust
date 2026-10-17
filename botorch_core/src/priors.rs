//! Hyperparameter priors mirroring GPyTorch's `LogNormalPrior`,
//! `GammaPrior`, `NormalPrior`, `UniformPrior`, `HalfCauchyPrior`,
//! `HalfNormalPrior` and `SmoothedBoxPrior` and BoTorch's `BetaPrior` (i.e.
//! the corresponding `torch.distributions`), including all normalizing
//! constants.
//!
//! GPyTorch stores prior parameters given as Python floats in float32;
//! [`Prior::rounded_to_f32`] reproduces that rounding, and for the priors
//! whose torch distribution keeps (part of) its float32 state when the model
//! is cast to double precision (`LogNormal`, `HalfNormal`: the base
//! `Normal`; `Uniform`: `low` and `high`) the single-precision parts of the
//! evaluation as well. `HalfCauchyPrior` is the exception: torch evaluates
//! it entirely in single precision (`HalfCauchy.log_prob` casts its argument
//! to the float32 dtype of the base `Cauchy`); this crate evaluates it in
//! double precision with the float32-rounded scale (see the deviations in
//! the documentation of [`Prior::HalfCauchy`]).

use rand::Rng;
use rand_distr::Distribution;
use statrs::function::gamma::ln_gamma;

use crate::constraints::round_to_f32;
use crate::error::{BotorchError, Result};

/// `log(sqrt(2π))`, the normalizing constant of the standard normal density.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
/// `log(π)` (`math.log(math.pi)`).
const LN_PI: f64 = 1.144_729_885_849_400_2;

/// A univariate prior over a positive hyperparameter.
///
/// Priors are evaluated on the *actual* (constrained) parameter value, as in
/// GPyTorch's `ExactMarginalLogLikelihood`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prior {
    /// Log-normal prior: `log x ~ Normal(loc, scale²)`.
    LogNormal {
        /// Mean of `log x`.
        loc: f64,
        /// Standard deviation of `log x` (> 0).
        scale: f64,
        /// Emulate GPyTorch's `LogNormalPrior` created from Python floats: its
        /// base `Normal` keeps float32 parameters even after the model is cast
        /// to double precision, so `scale²` and `log(scale)` are evaluated in
        /// single precision (set by [`Prior::rounded_to_f32`]).
        float32: bool,
        /// The float32 `log(scale)` as torch computes it (its single-precision
        /// `log` is not always correctly rounded); `None`: computed here.
        torch_log: Option<f64>,
    },
    /// Gamma prior with density `rate^c / Γ(c) · x^(c-1) · exp(-rate·x)`.
    Gamma {
        /// Concentration (shape) `c > 0`.
        concentration: f64,
        /// Rate `> 0` (inverse scale).
        rate: f64,
    },
    /// Normal prior `Normal(loc, scale²)` (`gpytorch.priors.NormalPrior`).
    Normal {
        /// Mean.
        loc: f64,
        /// Standard deviation (> 0).
        scale: f64,
    },
    /// Uniform prior on `[low, high)` (`gpytorch.priors.UniformPrior`): the
    /// log density is `-log(high - low)` inside and `-inf` outside; values
    /// outside `[low, high]` are rejected (torch validates the argument).
    Uniform {
        /// Lower end.
        low: f64,
        /// Upper end (> `low`).
        high: f64,
        /// Emulate GPyTorch's float32 `low` and `high`, which are not cast
        /// with the model: `log(high - low)` is evaluated in single precision.
        float32: bool,
        /// The float32 `log(high - low)` as torch computes it; `None`:
        /// computed here.
        torch_log: Option<f64>,
    },
    /// Half-Cauchy prior with density `2 / (π s (1 + (x/s)²))` on `x ≥ 0`
    /// (`gpytorch.priors.HalfCauchyPrior`); negative values are rejected.
    ///
    /// Deviation: torch's `HalfCauchy.log_prob` casts its argument to the
    /// dtype of its base distribution, which GPyTorch leaves in float32 when
    /// the model is cast to double precision, so BoTorch evaluates this prior
    /// (and its gradient) in single precision. It is evaluated in double
    /// precision here (with the float32-rounded scale).
    HalfCauchy {
        /// Scale `s > 0`.
        scale: f64,
    },
    /// Half-normal prior with density `2 N(x | 0, s²)` on `x ≥ 0`
    /// (`gpytorch.priors.HalfNormalPrior`); negative values are rejected.
    HalfNormal {
        /// Scale `s > 0`.
        scale: f64,
        /// Emulate GPyTorch's float32 base `Normal` (as for
        /// [`Prior::LogNormal`]): `s²` and `log(s)` in single precision.
        float32: bool,
        /// The float32 `log(scale)` as torch computes it; `None`: computed
        /// here.
        torch_log: Option<f64>,
    },
    /// Beta prior `Beta(concentration1, concentration0)` on `[0, 1]`
    /// (`botorch.models.utils.priors.BetaPrior`), with density
    /// `x^(α-1) (1-x)^(β-1) / B(α, β)`.
    Beta {
        /// `α = concentration1 > 0`.
        concentration1: f64,
        /// `β = concentration0 > 0`.
        concentration0: f64,
    },
    /// GPyTorch's `SmoothedBoxPrior(a, b, sigma)`, a smoothed uniform prior
    /// on `[a, b]` with Gaussian tails: `log N(dist(x, [a, b]) | 0, σ²) - M`
    /// with `M = log(1 + (b - a) / (√(2π) σ))`. Its parameters are stored
    /// (as torch buffers) in the model's precision, so only their float32
    /// rounding at construction is emulated.
    SmoothedBox {
        /// Lower end `a`.
        a: f64,
        /// Upper end `b > a`.
        b: f64,
        /// Scale `σ > 0` of the tails.
        sigma: f64,
    },
}

impl Prior {
    /// Creates a [`Prior::LogNormal`].
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `scale` is not positive or `loc` is not finite.
    pub fn log_normal(loc: f64, scale: f64) -> Result<Self> {
        if !(scale > 0.0 && scale.is_finite()) || !loc.is_finite() {
            return Err(BotorchError::InvalidArgument(format!(
                "LogNormal prior requires finite loc and positive finite scale, got loc={loc}, scale={scale}"
            )));
        }
        Ok(Prior::LogNormal {
            loc,
            scale,
            float32: false,
            torch_log: None,
        })
    }

    /// Creates a [`Prior::Gamma`].
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `concentration` or `rate` is not positive.
    pub fn gamma(concentration: f64, rate: f64) -> Result<Self> {
        if !(concentration > 0.0 && concentration.is_finite() && rate > 0.0 && rate.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "Gamma prior requires positive finite concentration and rate, got {concentration}, {rate}"
            )));
        }
        Ok(Prior::Gamma {
            concentration,
            rate,
        })
    }

    /// Creates a [`Prior::Normal`].
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `loc` is finite and `scale`
    /// positive and finite.
    pub fn normal(loc: f64, scale: f64) -> Result<Self> {
        if !(scale > 0.0 && scale.is_finite()) || !loc.is_finite() {
            return Err(BotorchError::InvalidArgument(format!(
                "Normal prior requires finite loc and positive finite scale, got loc={loc}, scale={scale}"
            )));
        }
        Ok(Prior::Normal { loc, scale })
    }

    /// Creates a [`Prior::Uniform`].
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `low < high` are finite.
    pub fn uniform(low: f64, high: f64) -> Result<Self> {
        if !(low.is_finite() && high.is_finite() && low < high) {
            return Err(BotorchError::InvalidArgument(format!(
                "Uniform prior requires finite low < high, got low={low}, high={high}"
            )));
        }
        Ok(Prior::Uniform {
            low,
            high,
            float32: false,
            torch_log: None,
        })
    }

    /// Creates a [`Prior::HalfCauchy`].
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `scale` is positive and finite.
    pub fn half_cauchy(scale: f64) -> Result<Self> {
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "HalfCauchy prior requires a positive finite scale, got {scale}"
            )));
        }
        Ok(Prior::HalfCauchy { scale })
    }

    /// Creates a [`Prior::HalfNormal`].
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `scale` is positive and finite.
    pub fn half_normal(scale: f64) -> Result<Self> {
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "HalfNormal prior requires a positive finite scale, got {scale}"
            )));
        }
        Ok(Prior::HalfNormal {
            scale,
            float32: false,
            torch_log: None,
        })
    }

    /// Creates a [`Prior::SmoothedBox`].
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `a < b` and `sigma > 0` are
    /// finite (GPyTorch's "must have that a < b (element-wise)").
    pub fn smoothed_box(a: f64, b: f64, sigma: f64) -> Result<Self> {
        if !(a.is_finite() && b.is_finite() && a < b) {
            return Err(BotorchError::InvalidArgument(
                "must have that a < b (element-wise)".into(),
            ));
        }
        if !(sigma > 0.0 && sigma.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "SmoothedBoxPrior requires a positive finite sigma, got {sigma}"
            )));
        }
        Ok(Prior::SmoothedBox { a, b, sigma })
    }

    /// Creates a [`Prior::Beta`].
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless both concentrations are
    /// positive and finite.
    pub fn beta(concentration1: f64, concentration0: f64) -> Result<Self> {
        let ok = |c: f64| c > 0.0 && c.is_finite();
        if !(ok(concentration1) && ok(concentration0)) {
            return Err(BotorchError::InvalidArgument(format!(
                "Beta prior requires positive finite concentrations, got {concentration1}, \
                 {concentration0}"
            )));
        }
        Ok(Prior::Beta {
            concentration1,
            concentration0,
        })
    }

    /// Returns a copy that behaves like a GPyTorch prior created from Python
    /// floats: `torch.distributions` stores such parameters in torch's default
    /// dtype (float32), and casting the model to float64 afterwards does not
    /// restore the lost digits. For `LogNormal`, whose base `Normal` is not
    /// cast at all, `scale²` and `log(scale)` are additionally evaluated in
    /// single precision. All BoTorch presets in this crate use this so that
    /// log-probabilities match BoTorch to double precision.
    pub fn rounded_to_f32(self) -> Self {
        match self {
            Prior::LogNormal { loc, scale, .. } => Prior::LogNormal {
                loc: round_to_f32(loc),
                scale: round_to_f32(scale),
                float32: true,
                torch_log: None,
            },
            Prior::Gamma {
                concentration,
                rate,
            } => Prior::Gamma {
                concentration: round_to_f32(concentration),
                rate: round_to_f32(rate),
            },
            Prior::Normal { loc, scale } => Prior::Normal {
                loc: round_to_f32(loc),
                scale: round_to_f32(scale),
            },
            Prior::Uniform { low, high, .. } => Prior::Uniform {
                low: round_to_f32(low),
                high: round_to_f32(high),
                float32: true,
                torch_log: None,
            },
            Prior::HalfCauchy { scale } => Prior::HalfCauchy {
                scale: round_to_f32(scale),
            },
            Prior::HalfNormal { scale, .. } => Prior::HalfNormal {
                scale: round_to_f32(scale),
                float32: true,
                torch_log: None,
            },
            Prior::Beta {
                concentration1,
                concentration0,
            } => Prior::Beta {
                concentration1: round_to_f32(concentration1),
                concentration0: round_to_f32(concentration0),
            },
            Prior::SmoothedBox { a, b, sigma } => Prior::SmoothedBox {
                a: round_to_f32(a),
                b: round_to_f32(b),
                sigma: round_to_f32(sigma),
            },
        }
    }

    /// Returns a copy (of a prior emulating float32 torch distributions)
    /// whose single-precision log term -- `log(scale)` of `LogNormal` and
    /// `HalfNormal`, `log(high - low)` of `Uniform` -- is `value`, the
    /// value torch computes (torch's float32 `log`, e.g. MKL's, is not always
    /// correctly rounded; this crate's is). Other priors are returned
    /// unchanged.
    pub fn with_torch_log(self, value: f64) -> Self {
        match self {
            Prior::LogNormal {
                loc,
                scale,
                float32: true,
                ..
            } => Prior::LogNormal {
                loc,
                scale,
                float32: true,
                torch_log: Some(value),
            },
            Prior::Uniform {
                low,
                high,
                float32: true,
                ..
            } => Prior::Uniform {
                low,
                high,
                float32: true,
                torch_log: Some(value),
            },
            Prior::HalfNormal {
                scale,
                float32: true,
                ..
            } => Prior::HalfNormal {
                scale,
                float32: true,
                torch_log: Some(value),
            },
            other => other,
        }
    }

    /// The name of the distribution (`"LogNormal"`, `"Gamma"`, `"Normal"`,
    /// `"Uniform"`, `"HalfCauchy"`, `"HalfNormal"`, `"Beta"` or
    /// `"SmoothedBox"`).
    pub fn name(&self) -> &'static str {
        match self {
            Prior::LogNormal { .. } => "LogNormal",
            Prior::Gamma { .. } => "Gamma",
            Prior::Normal { .. } => "Normal",
            Prior::Uniform { .. } => "Uniform",
            Prior::HalfCauchy { .. } => "HalfCauchy",
            Prior::HalfNormal { .. } => "HalfNormal",
            Prior::Beta { .. } => "Beta",
            Prior::SmoothedBox { .. } => "SmoothedBox",
        }
    }

    /// The parameters as stored, in GPyTorch's argument order.
    pub fn parameters(&self) -> Vec<f64> {
        match *self {
            Prior::LogNormal { loc, scale, .. } | Prior::Normal { loc, scale } => vec![loc, scale],
            Prior::Gamma {
                concentration,
                rate,
            } => vec![concentration, rate],
            Prior::Uniform { low, high, .. } => vec![low, high],
            Prior::HalfCauchy { scale } | Prior::HalfNormal { scale, .. } => vec![scale],
            Prior::Beta {
                concentration1,
                concentration0,
            } => vec![concentration1, concentration0],
            Prior::SmoothedBox { a, b, sigma } => vec![a, b, sigma],
        }
    }

    /// Whether `x` lies in the support that torch validates for this prior
    /// (GPyTorch's `UniformPrior`, `HalfCauchyPrior` and `HalfNormalPrior`
    /// validate their arguments; the other priors do not and return their
    /// log density, possibly `-inf` or NaN).
    pub fn in_validated_support(&self, x: f64) -> bool {
        match *self {
            Prior::Uniform { low, high, .. } => x >= low && x <= high,
            Prior::HalfCauchy { .. } | Prior::HalfNormal { .. } => x >= 0.0,
            _ => !x.is_nan(),
        }
    }

    /// Log density at `x` (matches `torch.distributions.*.log_prob`).
    pub fn log_prob(&self, x: f64) -> f64 {
        match *self {
            Prior::LogNormal { loc, .. } => {
                let (var, log_scale) = self.log_normal_var_and_log_scale();
                let lx = x.ln();
                let z = lx - loc;
                -lx + (-(z * z) / (2.0 * var) - log_scale - LN_SQRT_2PI)
            }
            Prior::Gamma {
                concentration,
                rate,
            } => {
                xlogy(concentration, rate) + xlogy(concentration - 1.0, x)
                    - rate * x
                    - ln_gamma(concentration)
            }
            Prior::Normal { loc, scale } => {
                let z = x - loc;
                -(z * z) / (2.0 * (scale * scale)) - scale.ln() - LN_SQRT_2PI
            }
            Prior::Uniform {
                low,
                high,
                float32,
                torch_log,
            } => {
                // torch: log(low <= x) * (high > x)) - log(high - low).
                let inside = if low <= x && high > x {
                    0.0
                } else {
                    f64::NEG_INFINITY
                };
                let log_width = match (float32, torch_log) {
                    (true, Some(v)) => v,
                    (true, None) => ((high as f32) - (low as f32)).ln() as f64,
                    (false, _) => (high - low).ln(),
                };
                inside - log_width
            }
            Prior::HalfCauchy { scale } => {
                if x < 0.0 {
                    return f64::NEG_INFINITY;
                }
                let z = x / scale;
                -LN_PI - scale.ln() - (z * z).ln_1p() + std::f64::consts::LN_2
            }
            Prior::HalfNormal { .. } => {
                if x < 0.0 {
                    return f64::NEG_INFINITY;
                }
                let (var, log_scale) = self.half_normal_var_and_log_scale();
                -(x * x) / (2.0 * var) - log_scale - LN_SQRT_2PI + std::f64::consts::LN_2
            }
            Prior::Beta {
                concentration1: a,
                concentration0: b,
            } => {
                xlogy(a - 1.0, x) + xlogy(b - 1.0, 1.0 - x) + ln_gamma(a + b)
                    - (ln_gamma(a) + ln_gamma(b))
            }
            Prior::SmoothedBox { a, b, sigma } => {
                // GPyTorch: ((x - c).abs() - r).clamp(min=0), the distance
                // from the box, under Normal(0, σ), minus the normalizer.
                let c = (a + b) / 2.0;
                let r = (b - a) / 2.0;
                let z = ((x - c).abs() - r).max(0.0);
                let tails = -(z * z) / (2.0 * (sigma * sigma)) - sigma.ln() - LN_SQRT_2PI;
                let big_m = (1.0 + (b - a) / ((2.0 * std::f64::consts::PI).sqrt() * sigma)).ln();
                tails - big_m
            }
        }
    }

    /// Derivative of [`Prior::log_prob`] with respect to `x`.
    pub fn grad_log_prob(&self, x: f64) -> f64 {
        match *self {
            Prior::LogNormal { loc, .. } => {
                let (var, _) = self.log_normal_var_and_log_scale();
                -(1.0 + (x.ln() - loc) / var) / x
            }
            Prior::Gamma {
                concentration,
                rate,
            } => {
                if concentration == 1.0 {
                    -rate
                } else {
                    (concentration - 1.0) / x - rate
                }
            }
            Prior::Normal { loc, scale } => -(x - loc) / (scale * scale),
            Prior::Uniform { .. } => 0.0,
            Prior::HalfCauchy { scale } => {
                let z = x / scale;
                -2.0 * z / (scale * (1.0 + z * z))
            }
            Prior::HalfNormal { .. } => {
                let (var, _) = self.half_normal_var_and_log_scale();
                -x / var
            }
            Prior::Beta {
                concentration1: a,
                concentration0: b,
            } => {
                let ga = if a == 1.0 { 0.0 } else { (a - 1.0) / x };
                let gb = if b == 1.0 { 0.0 } else { (b - 1.0) / (1.0 - x) };
                ga - gb
            }
            Prior::SmoothedBox { a, b, sigma } => {
                let c = (a + b) / 2.0;
                let r = (b - a) / 2.0;
                let z = (x - c).abs() - r;
                if z > 0.0 {
                    -z / (sigma * sigma) * (x - c).signum()
                } else {
                    0.0
                }
            }
        }
    }

    /// Mode of the distribution (double-precision arithmetic).
    pub fn mode(&self) -> f64 {
        match *self {
            Prior::LogNormal { loc, scale, .. } => (loc - scale * scale).exp(),
            Prior::Gamma {
                concentration,
                rate,
            } => ((concentration - 1.0) / rate).max(0.0),
            Prior::Normal { loc, .. } => loc,
            Prior::Uniform { low, high, .. } => 0.5 * (low + high),
            Prior::HalfCauchy { .. } | Prior::HalfNormal { .. } => 0.0,
            Prior::Beta {
                concentration1: a,
                concentration0: b,
            } => (a - 1.0) / (a + b - 2.0),
            Prior::SmoothedBox { a, b, .. } => 0.5 * (a + b),
        }
    }

    /// Mode computed in single precision, reproducing `prior.mode` of a
    /// float32 torch distribution. BoTorch uses this value as the initial
    /// hyperparameter value of its default models.
    pub fn mode_f32(&self) -> f64 {
        match *self {
            Prior::LogNormal { loc, scale, .. } => {
                let (loc, scale) = (loc as f32, scale as f32);
                (loc - scale * scale).exp() as f64
            }
            Prior::Gamma {
                concentration,
                rate,
            } => {
                let (c, r) = (concentration as f32, rate as f32);
                ((c - 1.0) / r).max(0.0) as f64
            }
            other => round_to_f32(other.mode()),
        }
    }

    /// Draws a sample from the prior.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match *self {
            Prior::LogNormal { loc, scale, .. } => rand_distr::LogNormal::new(loc, scale)
                .expect("validated LogNormal parameters")
                .sample(rng),
            Prior::Gamma {
                concentration,
                rate,
            } => rand_distr::Gamma::new(concentration, 1.0 / rate)
                .expect("validated Gamma parameters")
                .sample(rng),
            Prior::Normal { loc, scale } => rand_distr::Normal::new(loc, scale)
                .expect("validated Normal parameters")
                .sample(rng),
            Prior::Uniform { low, high, .. } => rand_distr::Uniform::new(low, high)
                .expect("validated Uniform parameters")
                .sample(rng),
            Prior::HalfCauchy { scale } => rand_distr::Cauchy::new(0.0, scale)
                .expect("validated Cauchy parameters")
                .sample(rng)
                .abs(),
            Prior::HalfNormal { scale, .. } => rand_distr::Normal::new(0.0, scale)
                .expect("validated Normal parameters")
                .sample(rng)
                .abs(),
            Prior::Beta {
                concentration1,
                concentration0,
            } => rand_distr::Beta::new(concentration1, concentration0)
                .expect("validated Beta parameters")
                .sample(rng),
            Prior::SmoothedBox { a, b, sigma } => {
                // GPyTorch's `rsample`: the Gaussian tails with probability
                // 1 / (1 + (b - a) / (√(2π) σ)), else uniform on [a, b].
                let gauss_max = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * sigma);
                let gauss_weight = 1.0 / (gauss_max * (b - a) + 1.0);
                let u: f64 = rand_distr::StandardUniform.sample(rng);
                if u < gauss_weight {
                    let e: f64 = rand_distr::Normal::new(0.0, sigma)
                        .expect("validated Normal parameters")
                        .sample(rng);
                    e + if e < 0.0 { a } else { b }
                } else {
                    let v: f64 = rand_distr::StandardUniform.sample(rng);
                    a + v * (b - a)
                }
            }
        }
    }
}

impl Prior {
    /// `(scale², log(scale))` of a LogNormal prior, in single precision when
    /// emulating a float32 torch distribution.
    fn log_normal_var_and_log_scale(&self) -> (f64, f64) {
        match *self {
            Prior::LogNormal {
                scale,
                float32,
                torch_log,
                ..
            } => {
                if float32 {
                    let s = scale as f32;
                    ((s * s) as f64, torch_log.unwrap_or(s.ln() as f64))
                } else {
                    (scale * scale, scale.ln())
                }
            }
            _ => unreachable!("not a LogNormal prior"),
        }
    }

    /// `(scale², log(scale))` of a HalfNormal prior, in single precision when
    /// emulating a float32 torch distribution.
    fn half_normal_var_and_log_scale(&self) -> (f64, f64) {
        match *self {
            Prior::HalfNormal {
                scale,
                float32,
                torch_log,
            } => {
                if float32 {
                    let s = scale as f32;
                    ((s * s) as f64, torch_log.unwrap_or(s.ln() as f64))
                } else {
                    (scale * scale, scale.ln())
                }
            }
            _ => unreachable!("not a HalfNormal prior"),
        }
    }
}

/// `torch.xlogy`: `x * log(y)`, defined as 0 when `x == 0` (and `y` is not NaN).
fn xlogy(x: f64, y: f64) -> f64 {
    if y.is_nan() {
        f64::NAN
    } else if x == 0.0 {
        0.0
    } else {
        x * y.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_normal_matches_closed_form() {
        let p = Prior::log_normal(-4.0, 1.0).unwrap();
        // Reference values from torch.distributions.LogNormal(-4.0, 1.0).log_prob.
        let x: f64 = 0.01;
        let expected =
            -x.ln() - 0.5 * (x.ln() + 4.0).powi(2) - (2.0 * std::f64::consts::PI).sqrt().ln();
        assert!((p.log_prob(x) - expected).abs() < 1e-14);
        assert!((p.mode() - (-5.0f64).exp()).abs() < 1e-18);
    }

    #[test]
    fn float32_log_normal_matches_torch() {
        // torch: SingleTaskGP(...).covar_module.lengthscale_prior (d = 2)
        //            .log_prob(tensor([0.3, 0.7], dtype=float64))
        let p = Prior::LogNormal {
            loc: std::f64::consts::SQRT_2 + 2f64.ln() * 0.5,
            scale: 3f64.sqrt(),
            float32: false,
            torch_log: None,
        }
        .rounded_to_f32();
        assert_eq!(p.log_prob(0.3), -1.729238793903073);
        assert_eq!(p.log_prob(0.7), -1.8588440155232244);
        assert_eq!(LN_SQRT_2PI, (2.0 * std::f64::consts::PI).sqrt().ln());
    }

    #[test]
    fn gamma_matches_closed_form() {
        let p = Prior::gamma(3.0, 6.0).unwrap();
        let x: f64 = 0.4;
        let expected = 3.0 * 6.0f64.ln() + 2.0 * x.ln() - 6.0 * x - 2.0f64.ln();
        assert!((p.log_prob(x) - expected).abs() < 1e-13);
        assert!((p.mode() - 2.0 / 6.0).abs() < 1e-16);
    }

    #[test]
    fn gradients_match_finite_differences() {
        let priors = [
            Prior::log_normal(1.3, 1.7).unwrap(),
            Prior::gamma(1.1, 0.05).unwrap(),
            Prior::gamma(2.0, 0.15).unwrap(),
        ];
        for p in priors {
            for &x in &[0.03, 0.5, 2.0, 7.5] {
                let h = 1e-6 * x;
                let fd = (p.log_prob(x + h) - p.log_prob(x - h)) / (2.0 * h);
                let g = p.grad_log_prob(x);
                assert!(
                    (fd - g).abs() < 1e-6 * (1.0 + g.abs()),
                    "{p:?} {x} {fd} {g}"
                );
            }
        }
    }

    #[test]
    fn new_priors_match_closed_forms() {
        use std::f64::consts::PI;
        let x: f64 = 0.37;
        let n = Prior::normal(0.3, 1.7).unwrap();
        let e = -(x - 0.3f64).powi(2) / (2.0 * 1.7 * 1.7) - (1.7f64 * (2.0 * PI).sqrt()).ln();
        assert!((n.log_prob(x) - e).abs() < 1e-14);
        let u = Prior::uniform(0.1, 2.3).unwrap();
        assert!((u.log_prob(x) + 2.2f64.ln()).abs() < 1e-14);
        assert_eq!(u.log_prob(2.3), f64::NEG_INFINITY);
        assert!(!u.in_validated_support(2.4));
        assert!(u.in_validated_support(2.3));
        let hc = Prior::half_cauchy(0.7).unwrap();
        let e = (2.0 / (PI * 0.7 * (1.0 + (x / 0.7).powi(2)))).ln();
        assert!((hc.log_prob(x) - e).abs() < 1e-14);
        assert!(!hc.in_validated_support(-0.1));
        let hn = Prior::half_normal(0.7).unwrap();
        let e = (2.0 / (0.7 * (2.0 * PI).sqrt())).ln() - x * x / (2.0 * 0.49);
        assert!((hn.log_prob(x) - e).abs() < 1e-14);
        let b = Prior::beta(2.5, 1.5).unwrap();
        // B(2.5, 1.5) = Γ(2.5) Γ(1.5) / Γ(4) = (1.329340388 · 0.886226925) / 6.
        let beta_fn: f64 = 1.329_340_388_179_137 * 0.886_226_925_452_758 / 6.0;
        let e = 1.5 * x.ln() + 0.5 * (1.0 - x).ln() - beta_fn.ln();
        assert!((b.log_prob(x) - e).abs() < 1e-13);
        for p in [n, u, hc, hn, b] {
            for &x in &[0.15, 0.37, 0.8] {
                let h = 1e-6;
                let fd = (p.log_prob(x + h) - p.log_prob(x - h)) / (2.0 * h);
                assert!((fd - p.grad_log_prob(x)).abs() < 1e-6, "{p:?} {x}");
            }
        }
        // Float32 emulation of GPyTorch's HalfNormalPrior(0.7) on a double model
        // (torch: HalfNormalPrior(0.7).log_prob(tensor(0.123456789012345, float64))).
        let hn32 = Prior::half_normal(0.7).unwrap().rounded_to_f32();
        assert_eq!(hn32.log_prob(0.123456789012345), 0.11533098452750556);
        let g = hn32.grad_log_prob(0.123456789012345);
        assert!((g + 0.2519526410577779).abs() < 1e-15, "{g}");
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        assert!(Prior::log_normal(0.0, 0.0).is_err());
        assert!(Prior::gamma(-1.0, 1.0).is_err());
        assert!(Prior::gamma(1.0, f64::INFINITY).is_err());
        assert!(Prior::normal(0.0, -1.0).is_err());
        assert!(Prior::uniform(1.0, 1.0).is_err());
        assert!(Prior::half_cauchy(0.0).is_err());
        assert!(Prior::half_normal(f64::NAN).is_err());
        assert!(Prior::beta(1.0, 0.0).is_err());
    }

    #[test]
    fn samples_are_positive() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for p in [
            Prior::log_normal(0.0, 1.0).unwrap(),
            Prior::gamma(2.0, 3.0).unwrap(),
        ] {
            for _ in 0..100 {
                assert!(p.sample(&mut rng) > 0.0);
            }
        }
    }
}
