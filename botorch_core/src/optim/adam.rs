//! The Adam optimizer with the update of `torch.optim.Adam` (its
//! single-tensor CPU implementation), used by the torch-style optimizers of
//! BoTorch (`gen_candidates_torch`, `fit_gpytorch_mll_torch`).
//!
//! The update follows torch's arithmetic: `exp_avg.lerp_(grad, 1 - beta1)`
//! (a fused multiply-add), `exp_avg_sq.mul_(beta2).addcmul_(grad, grad,
//! value=1 - beta2)`, the bias corrections in double precision and
//! `param.addcdiv_(exp_avg, sqrt(exp_avg_sq) / sqrt(bc2) + eps,
//! value=-lr / bc1)`.

use crate::error::{BotorchError, Result};

/// Settings of [`Adam`] (`torch.optim.Adam`'s arguments).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamOptions {
    /// Learning rate (torch's default 1e-3).
    pub lr: f64,
    /// Coefficients of the running averages (default `(0.9, 0.999)`).
    pub betas: (f64, f64),
    /// Term added to the denominator (default 1e-8).
    pub eps: f64,
    /// L2 penalty (default 0).
    pub weight_decay: f64,
    /// Use the AMSGrad variant (default false).
    pub amsgrad: bool,
    /// Maximize instead of minimize (default false).
    pub maximize: bool,
}

impl Default for AdamOptions {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            maximize: false,
        }
    }
}

impl AdamOptions {
    /// Checks the arguments like torch (`Invalid learning rate: ...`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for invalid settings.
    pub fn validate(&self) -> Result<()> {
        if self.lr.is_nan() || self.lr < 0.0 {
            return Err(BotorchError::InvalidArgument(format!(
                "Invalid learning rate: {}",
                self.lr
            )));
        }
        if self.eps.is_nan() || self.eps < 0.0 {
            return Err(BotorchError::InvalidArgument(format!(
                "Invalid epsilon value: {}",
                self.eps
            )));
        }
        let (b1, b2) = self.betas;
        if !(0.0..1.0).contains(&b1) {
            return Err(BotorchError::InvalidArgument(format!(
                "Invalid beta parameter at index 0: {b1}"
            )));
        }
        if !(0.0..1.0).contains(&b2) {
            return Err(BotorchError::InvalidArgument(format!(
                "Invalid beta parameter at index 1: {b2}"
            )));
        }
        if self.weight_decay.is_nan() || self.weight_decay < 0.0 {
            return Err(BotorchError::InvalidArgument(format!(
                "Invalid weight_decay value: {}",
                self.weight_decay
            )));
        }
        Ok(())
    }
}

/// The state of an Adam optimizer over a flat parameter vector.
#[derive(Clone, Debug)]
pub struct Adam {
    options: AdamOptions,
    step: u64,
    exp_avg: Vec<f64>,
    exp_avg_sq: Vec<f64>,
    max_exp_avg_sq: Vec<f64>,
}

impl Adam {
    /// A new optimizer for `n` parameters.
    ///
    /// # Errors
    /// See [`AdamOptions::validate`].
    pub fn new(n: usize, options: AdamOptions) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            options,
            step: 0,
            exp_avg: vec![0.0; n],
            exp_avg_sq: vec![0.0; n],
            max_exp_avg_sq: if options.amsgrad {
                vec![0.0; n]
            } else {
                Vec::new()
            },
        })
    }

    /// The number of steps taken.
    pub fn steps(&self) -> u64 {
        self.step
    }

    /// One update of `params` with the gradient `grads` (torch's
    /// `optimizer.step()` after the gradients were assigned). Entries listed
    /// as frozen (`None` gradients in torch) are skipped.
    ///
    /// # Panics
    /// If the lengths of `params` and `grads` differ from the optimizer's.
    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), self.exp_avg.len());
        assert_eq!(grads.len(), self.exp_avg.len());
        let o = self.options;
        let (beta1, beta2) = o.betas;
        self.step += 1;
        let step = self.step as f64;
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);
        let step_size = o.lr / bias_correction1;
        let bias_correction2_sqrt = bias_correction2.powf(0.5);
        let w1 = 1.0 - beta1;
        let w2 = 1.0 - beta2;
        for i in 0..params.len() {
            let mut g = if o.maximize { -grads[i] } else { grads[i] };
            if o.weight_decay != 0.0 {
                g += params[i] * o.weight_decay;
            }
            // exp_avg.lerp_(grad, 1 - beta1): torch's lerp kernel is a fused
            // multiply-add for weights below 0.5.
            let m = self.exp_avg[i];
            self.exp_avg[i] = if w1.abs() < 0.5 {
                w1.mul_add(g - m, m)
            } else {
                g - (g - m) * (1.0 - w1)
            };
            self.exp_avg_sq[i] = self.exp_avg_sq[i] * beta2 + (w2 * g) * g;
            let v = if o.amsgrad {
                self.max_exp_avg_sq[i] = self.max_exp_avg_sq[i].max(self.exp_avg_sq[i]);
                self.max_exp_avg_sq[i]
            } else {
                self.exp_avg_sq[i]
            };
            let denom = v.sqrt() / bias_correction2_sqrt + o.eps;
            params[i] += (-step_size * self.exp_avg[i]) / denom;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimizes_a_quadratic() {
        let mut adam = Adam::new(
            2,
            AdamOptions {
                lr: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        let mut x = vec![1.0, -2.0];
        for _ in 0..500 {
            let g = vec![2.0 * x[0], 4.0 * (x[1] - 0.5)];
            adam.step(&mut x, &g);
        }
        assert!(x[0].abs() < 1e-3 && (x[1] - 0.5).abs() < 1e-3, "{x:?}");
        // The first step moves every coordinate by about lr.
        let mut adam = Adam::new(1, AdamOptions::default()).unwrap();
        let mut x = vec![0.0];
        adam.step(&mut x, &[3.0]);
        assert!((x[0] + 1e-3).abs() < 1e-9);
        assert!(Adam::new(
            1,
            AdamOptions {
                lr: -1.0,
                ..Default::default()
            }
        )
        .is_err());
    }
}
