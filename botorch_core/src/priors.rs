//! Hyperparameter priors mirroring `gpytorch.priors.LogNormalPrior` and
//! `gpytorch.priors.GammaPrior` (i.e. `torch.distributions.LogNormal` and
//! `torch.distributions.Gamma`), including all normalizing constants.

use rand::Rng;
use rand_distr::Distribution;
use statrs::function::gamma::ln_gamma;

use crate::constraints::round_to_f32;
use crate::error::{BotorchError, Result};

/// `log(sqrt(2π))`, the normalizing constant of the standard normal density.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

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
    },
    /// Gamma prior with density `rate^c / Γ(c) · x^(c-1) · exp(-rate·x)`.
    Gamma {
        /// Concentration (shape) `c > 0`.
        concentration: f64,
        /// Rate `> 0` (inverse scale).
        rate: f64,
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
            },
            Prior::Gamma {
                concentration,
                rate,
            } => Prior::Gamma {
                concentration: round_to_f32(concentration),
                rate: round_to_f32(rate),
            },
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
        }
    }
}

impl Prior {
    /// `(scale², log(scale))` of a LogNormal prior, in single precision when
    /// emulating a float32 torch distribution.
    fn log_normal_var_and_log_scale(&self) -> (f64, f64) {
        match *self {
            Prior::LogNormal { scale, float32, .. } => {
                if float32 {
                    let s = scale as f32;
                    ((s * s) as f64, s.ln() as f64)
                } else {
                    (scale * scale, scale.ln())
                }
            }
            Prior::Gamma { .. } => unreachable!("not a LogNormal prior"),
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
    fn invalid_parameters_are_rejected() {
        assert!(Prior::log_normal(0.0, 0.0).is_err());
        assert!(Prior::gamma(-1.0, 1.0).is_err());
        assert!(Prior::gamma(1.0, f64::INFINITY).is_err());
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
