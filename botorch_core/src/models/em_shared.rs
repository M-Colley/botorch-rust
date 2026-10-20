//! Exact joint likelihood fitting for GP lists with genuinely shared parameters.

use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::mll::ExactMarginalLogLikelihood;
use crate::models::SingleTaskGP;
use crate::optim::fit::MllModel;

/// A single mean/kernel fitted to independent datasets with distinct noises.
#[derive(Clone, Debug)]
pub struct SharedGPFit {
    models: Vec<SingleTaskGP>,
    raw: Vec<f64>,
    learn_mean: bool,
    freeze_noise: bool,
    kernel_size: usize,
}
impl SharedGPFit {
    /// Construct from exact scalar GP snapshots with matching covariance layouts.
    pub fn new(models: Vec<SingleTaskGP>, learn_mean: bool, freeze_noise: bool) -> Result<Self> {
        let Some(first) = models.first() else {
            return Err(BotorchError::InvalidArgument(
                "shared GP fitting requires datasets".into(),
            ));
        };
        let kernel_size = first.parameter_layout().kernel_range().len();
        if models.iter().any(|m| {
            m.dim() != first.dim()
                || m.parameter_layout().noise_index().is_none()
                || m.parameter_layout().kernel_range().len() != kernel_size
        }) {
            return Err(BotorchError::DimensionMismatch(
                "shared GPs require matching scalar Gaussian covariance layouts".into(),
            ));
        }
        let mut raw: Vec<f64> = models
            .iter()
            .map(|m| m.raw_parameters()[m.parameter_layout().noise_index().unwrap()])
            .collect();
        let initial = first.raw_parameters();
        if learn_mean {
            raw.push(initial[first.parameter_layout().constant_index()]);
        }
        raw.extend_from_slice(&initial.as_slice()[first.parameter_layout().kernel_range()]);
        let mut result = Self {
            models,
            raw: vec![],
            learn_mean,
            freeze_noise,
            kernel_size,
        };
        result.set_raw_parameters(&raw)?;
        Ok(result)
    }
    fn offset(&self) -> usize {
        self.models.len() + usize::from(self.learn_mean)
    }
    /// Flat joint parameter vector (all noises, shared constant, shared kernel).
    pub fn raw_parameters(&self) -> &[f64] {
        &self.raw
    }
    fn local_raw(&self, i: usize, raw: &[f64]) -> Result<Vec<f64>> {
        if raw.len() != self.offset() + self.kernel_size || raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "invalid shared GP raw parameters".into(),
            ));
        }
        let model = &self.models[i];
        let layout = model.parameter_layout();
        let mut local = model.raw_parameters().as_slice().to_vec();
        local[layout.noise_index().unwrap()] = raw[i];
        local[layout.constant_index()] = if self.learn_mean {
            raw[self.models.len()]
        } else {
            0.0
        };
        local[layout.kernel_range()].copy_from_slice(&raw[self.offset()..]);
        Ok(local)
    }
    /// Atomically update every independent GP from the joint vector.
    pub fn set_raw_parameters(&mut self, raw: &[f64]) -> Result<()> {
        let mut models = self.models.clone();
        for (i, model) in models.iter_mut().enumerate() {
            model.set_raw_parameters(&self.local_raw(i, raw)?)?;
        }
        self.models = models;
        self.raw = raw.to_vec();
        Ok(())
    }
    /// Exact parameter state of each scalar GP, for installing into Python owners.
    pub fn local_parameters(&self) -> Vec<Vec<f64>> {
        self.models
            .iter()
            .map(|m| m.raw_parameters().as_slice().to_vec())
            .collect()
    }
    /// Public de-duplicated SumMarginalLogLikelihood parameter shapes.
    pub fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let mut result: Vec<_> = (0..self.models.len())
            .map(|i| {
                (
                    format!("likelihood.likelihoods.{i}.noise_covar.raw_noise"),
                    vec![1],
                )
            })
            .collect();
        let first = &self.models[0];
        if self.learn_mean {
            result.push(("model.models.0.mean_module.raw_constant".into(), vec![]));
        }
        result.extend(
            first
                .parameter_element_shapes()
                .into_iter()
                .filter(|(name, _)| name.starts_with("covar_module."))
                .map(|(name, shape)| (format!("model.models.0.{name}"), shape)),
        );
        result
    }
    /// Raw bounds, pinning explicitly fixed observation variances.
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut result: Vec<_> = self
            .models
            .iter()
            .enumerate()
            .map(|(i, m)| {
                if self.freeze_noise {
                    (self.raw[i], self.raw[i])
                } else {
                    m.raw_parameter_bounds()[m.parameter_layout().noise_index().unwrap()]
                }
            })
            .collect();
        let first = &self.models[0];
        let bounds = first.raw_parameter_bounds();
        if self.learn_mean {
            result.push(bounds[first.parameter_layout().constant_index()]);
        }
        result.extend_from_slice(&bounds[first.parameter_layout().kernel_range()]);
        result
    }
    /// Mean normalized exact likelihood and accumulated shared raw derivatives.
    pub fn value_and_grad(&self, raw: &[f64]) -> Result<(f64, Vec<f64>)> {
        let mut value = 0.0;
        let mut gradient = vec![0.0; raw.len()];
        let count = self.models.len();
        for (i, model) in self.models.iter().enumerate() {
            let (v, g) =
                ExactMarginalLogLikelihood::new(model).value_and_grad(&self.local_raw(i, raw)?)?;
            let layout = model.parameter_layout();
            value += v;
            gradient[i] = g[layout.noise_index().unwrap()];
            if self.learn_mean {
                gradient[count] += g[layout.constant_index()];
            }
            for (dest, g) in gradient[self.offset()..]
                .iter_mut()
                .zip(&g.as_slice()[layout.kernel_range()])
            {
                *dest += g;
            }
        }
        for g in &mut gradient {
            *g /= count as f64;
        }
        Ok((value / count as f64, gradient))
    }
    fn sample_priors(&self, rng: &mut rand::rngs::StdRng) -> Vec<f64> {
        let mut raw = self.raw.clone();
        for (i, model) in self.models.iter().enumerate() {
            let sampled = crate::fit::sample_all_priors(model, model.hyperparameters(), rng);
            if !self.freeze_noise {
                raw[i] = sampled[model.parameter_layout().noise_index().unwrap()];
            }
            if i == 0 {
                if self.learn_mean {
                    raw[self.models.len()] = sampled[model.parameter_layout().constant_index()];
                }
                raw[self.offset()..]
                    .copy_from_slice(&sampled[model.parameter_layout().kernel_range()]);
            }
        }
        raw
    }
}
impl MllModel for SharedGPFit {
    fn raw_parameter_vector(&self) -> Vec<f64> {
        self.raw.clone()
    }
    fn set_raw_parameter_vector(&mut self, values: &[f64]) -> Result<()> {
        self.set_raw_parameters(values)
    }
    fn raw_bounds(&self) -> Vec<(f64, f64)> {
        self.raw_parameter_bounds()
    }
    fn mll_values_and_gradient(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        self.value_and_grad(x).map(|(v, g)| (vec![v], g))
    }
}
impl Fittable for SharedGPFit {
    type Output = FitResult;
    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        let (result, raw) = minimize_with_retries(
            &self.raw,
            &self.raw_parameter_bounds(),
            options,
            |raw| self.value_and_grad(raw),
            |rng| self.sample_priors(rng),
        )?;
        self.set_raw_parameters(&raw)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{KernelKind, KernelNode};
    use crate::models::SingleTaskGPOptions;
    use crate::transforms::OutcomeTransformChoice;
    use crate::Matrix;

    #[test]
    fn joint_shared_gradients_and_fixed_noises_are_correct() {
        for learn_mean in [false, true] {
            let models = (0..3)
                .map(|i| {
                    let x = Matrix::from_fn(3 + i, 1, |j, _| j as f64 / (2 + i) as f64);
                    let y = Matrix::from_fn(3 + i, 1, |j, _| ((j + i) as f64).sin());
                    SingleTaskGP::new(
                        x,
                        y,
                        SingleTaskGPOptions {
                            covar_tree: Some(KernelNode::scale(
                                KernelNode::stationary(KernelKind::Rbf, Some(1)).unwrap(),
                            )),
                            outcome_transform: OutcomeTransformChoice::None,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect();
            let mut model = SharedGPFit::new(models, learn_mean, true).unwrap();
            let raw = model.raw_parameters().to_vec();
            let (_, gradient) = model.value_and_grad(&raw).unwrap();
            for i in 0..raw.len() {
                let mut p = raw.clone();
                let mut m = raw.clone();
                let h = 1e-6;
                p[i] += h;
                m[i] -= h;
                let numerical = (model.value_and_grad(&p).unwrap().0
                    - model.value_and_grad(&m).unwrap().0)
                    / (2.0 * h);
                assert!(
                    (numerical - gradient[i]).abs() < 1e-5,
                    "raw{i}: {numerical} != {}",
                    gradient[i]
                );
            }
            model
                .fit_mll(FitOptions {
                    maxiter: 8,
                    max_attempts: 1,
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(&model.raw_parameters()[..3], &raw[..3]);
            let locals = model.local_parameters();
            for local in &locals[1..] {
                assert_eq!(&local[1..], &locals[0][1..]);
            }
        }
    }
}
