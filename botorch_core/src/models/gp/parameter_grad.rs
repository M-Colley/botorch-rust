//! Posterior pullbacks to the exact GP's raw hyperparameters.

use super::{Likelihood, SingleTaskGP};
use crate::error::{BotorchError, Result};
use crate::kernels::{BuiltCovar, CovarModule, Prepared};
use crate::transforms::input::TransformMode;
use crate::{Matrix, Vector};

impl SingleTaskGP {
    /// Derivative of a weighted Gaussian posterior mean and covariance with
    /// respect to every raw hyperparameter in [`Self::raw_parameters`] order.
    ///
    /// The adjoints address the original outcome space, or the Gaussian base
    /// of a nonlinear transformed posterior. This differentiates both the
    /// training covariance solve and the train/query kernel blocks. Learned
    /// input-transform parameters receive both training and query contributions.
    /// No hyperparameter priors are included in a posterior pullback.
    ///
    /// # Errors
    /// Shape and factorization errors follow [`Self::joint_posterior_vjp`].
    /// Sparse-outlier likelihood parameters and training-input snapshots that
    /// differ from the recorded input-transform program are not supported.
    pub fn posterior_parameter_vjp(
        &self,
        x: &Matrix,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
        observation_noise: bool,
    ) -> Result<Vector> {
        if self.sparse_outlier_noise.is_some() {
            return Err(BotorchError::Unsupported(
                "posterior hyperparameter adjoints with sparse-outlier noise".into(),
            ));
        }
        let xt = self.transform_inputs(x)?;
        let q = xt.nrows();
        if mean_adjoint.len() != q || covariance_adjoint.shape() != (q, q) {
            return Err(BotorchError::DimensionMismatch(format!(
                "posterior adjoints must have shapes {q} and {q} x {q}"
            )));
        }
        let cache = self.cache()?;
        let n = self.num_train();
        let (affine, _) = self.outcome_parts(&xt)?;
        let mbar = Vector::from_fn(q, |i, _| mean_adjoint[i] * affine.scale(i));
        let abar = Matrix::from_fn(q, q, |i, j| {
            (covariance_adjoint[(i, j)] + covariance_adjoint[(j, i)])
                * (0.5 * affine.scale(i) * affine.scale(j))
        });
        let xs = cache.kernel.prepare(&xt)?;
        let cross = cache.kernel.cross(&xs, &cache.train_scaled);
        let w = cache.chol.solve(&cross.transpose());
        let target_bar = &w * &mbar;
        let train_bar = &w * &abar * w.transpose()
            - (&target_bar * cache.alpha.transpose() + &cache.alpha * target_bar.transpose()) * 0.5;
        let cross_bar = &mbar * cache.alpha.transpose() - &abar * w.transpose() * 2.0;
        let all_x = Matrix::from_fn(n + q, xt.ncols(), |i, j| {
            if i < n {
                self.train_x_tf[(i, j)]
            } else {
                xt[(i - n, j)]
            }
        });
        let joint_bar = Matrix::from_fn(n + q, n + q, |i, j| match (i < n, j < n) {
            (true, true) => train_bar[(i, j)],
            (false, false) => abar[(i - n, j - n)],
            (false, true) => cross_bar[(i - n, j)] * 0.5,
            (true, false) => cross_bar[(j - n, i)] * 0.5,
        });
        let prepared = cache.kernel.prepare(&all_x)?;
        let layout = self.parameter_layout();
        let mut gradient = Vector::zeros(layout.len());
        gradient[layout.constant_index()] = mbar.sum() - target_bar.sum();
        if let (Some(index), Likelihood::Gaussian(spec), Some(raw)) = (
            layout.noise_index(),
            &self.likelihood,
            self.hyperparameters.raw_noise,
        ) {
            let training_noise_bar: f64 = (0..n)
                .filter(|&i| self.point_noise.as_ref().is_none_or(|p| p[i].is_none()))
                .map(|i| train_bar[(i, i)])
                .sum();
            gradient[index] = (training_noise_bar
                + if observation_noise { abar.trace() } else { 0.0 })
                * spec.noise_constraint.transform_derivative(raw);
        }
        match (&self.covar_module, &cache.kernel, &prepared) {
            (
                CovarModule::Standard(spec),
                BuiltCovar::Standard(kernel),
                Prepared::Standard(points),
            ) => {
                let (lengthscale, scale) = kernel.contract_hyper_grad(points, &joint_bar);
                for (j, index) in layout.lengthscale_range().enumerate() {
                    gradient[index] = lengthscale[j]
                        * spec
                            .lengthscale_constraint
                            .transform_derivative(self.hyperparameters.raw_lengthscale[j]);
                }
                if let (Some(index), Some(spec), Some(raw), Some(scale)) = (
                    layout.outputscale_index(),
                    spec.scale,
                    self.hyperparameters.raw_outputscale,
                    scale,
                ) {
                    gradient[index] = scale * spec.outputscale_constraint.transform_derivative(raw);
                }
            }
            (CovarModule::Tree(node), BuiltCovar::Tree(tree), Prepared::Tree(points)) => {
                let kernel_gradient = tree.contract(points, &joint_bar);
                let derivatives =
                    node.transform_derivatives(self.hyperparameters.raw_kernel.as_slice());
                for (j, index) in layout.kernel_range().enumerate() {
                    gradient[index] = kernel_gradient[j] * derivatives[j];
                }
            }
            _ => unreachable!("points prepared for the cached covariance module"),
        }
        if let Some(chain) = &self.input_chain {
            if chain.num_params() > 0 {
                let (train_tf, train_tape) =
                    chain.transform_with_tape_mode(&self.train_x, TransformMode::Train)?;
                if train_tf != self.train_x_tf {
                    return Err(BotorchError::Unsupported(
                        "posterior input-parameter adjoints require a matching training transform snapshot"
                            .into(),
                    ));
                }
                let (query_tf, query_tape) = chain.transform_with_tape(x)?;
                debug_assert_eq!(query_tf, xt);
                let mut point_bar = Matrix::zeros(n + q, xt.ncols());
                for i in 0..n + q {
                    let (_, derivatives) = cache.kernel.cross_with_grad(&prepared, i, &prepared);
                    let row = derivatives.tr_mul(&joint_bar.row(i).transpose()) * 2.0;
                    point_bar.row_mut(i).copy_from(&row.transpose());
                }
                let (_, train_parameters) = chain.backward_with_params(
                    &train_tape,
                    0..self.train_x.nrows(),
                    &point_bar.rows(0, n).into_owned(),
                )?;
                let (_, query_parameters) = chain.backward_with_params(
                    &query_tape,
                    0..x.nrows(),
                    &point_bar.rows(n, q).into_owned(),
                )?;
                for (j, index) in layout.input_range().enumerate() {
                    gradient[index] = train_parameters[j] + query_parameters[j];
                }
            }
        }
        Ok(gradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{KernelKind, KernelNode};
    use crate::models::SingleTaskGPOptions;
    use crate::transforms::input::{ChainedInputTransform, InputTransformStep};
    use crate::transforms::input_steps::{NewStep, WarpStep};
    use crate::{Normalize, OutcomeTransformChoice};

    fn models() -> Vec<SingleTaskGP> {
        let x = Matrix::from_row_slice(5, 2, &[0.1, 0.2, 0.3, 0.7, 0.5, 0.4, 0.7, 0.9, 0.9, 0.1]);
        let y = Matrix::from_column_slice(5, 1, &[1.2, 2.1, 1.7, 0.6, 1.4]);
        let tree = KernelNode::scale(
            KernelNode::additive(vec![
                KernelNode::stationary(KernelKind::Matern32, Some(2)).unwrap(),
                KernelNode::linear(Some(2)).unwrap(),
            ])
            .unwrap(),
        );
        let warp = ChainedInputTransform::new(vec![InputTransformStep::Step(NewStep::Warp(
            WarpStep::new(2, vec![0, 1], 1e-7, Some((vec![0.0; 2], vec![1.0; 2]))).unwrap(),
        ))])
        .unwrap();
        vec![
            SingleTaskGPOptions::default(),
            SingleTaskGPOptions {
                kernel: KernelKind::Matern52,
                use_scale_kernel: true,
                input_transform: Some(
                    Normalize::with_bounds(&Matrix::from_row_slice(2, 2, &[-0.2, -0.1, 1.2, 1.3]))
                        .unwrap(),
                ),
                ..Default::default()
            },
            SingleTaskGPOptions {
                train_yvar: Some(Matrix::from_element(5, 1, 0.03)),
                covar_tree: Some(tree.clone()),
                outcome_transform: OutcomeTransformChoice::None,
                ..Default::default()
            },
            SingleTaskGPOptions {
                covar_tree: Some(tree),
                input_chain: Some(warp),
                ..Default::default()
            },
        ]
        .into_iter()
        .map(|options| {
            let mut model = SingleTaskGP::new(x.clone(), y.clone(), options).unwrap();
            let mut raw = model.raw_parameters();
            raw[model.parameter_layout().constant_index()] = 0.23;
            model.set_raw_parameters(raw.as_slice()).unwrap();
            model
        })
        .collect()
    }

    #[test]
    fn posterior_parameters_match_finite_differences() {
        let x = Matrix::from_row_slice(3, 2, &[0.22, 0.31, 0.45, 0.62, 0.82, 0.42]);
        let mean = Vector::from_column_slice(&[0.3, -0.7, 1.2]);
        // Deliberately nonsymmetric: only the symmetric part contributes.
        let covariance =
            Matrix::from_row_slice(3, 3, &[0.8, -0.1, 0.4, 0.2, -0.3, 0.5, 0.1, 0.6, -0.2]);
        for (case, model) in models().into_iter().enumerate() {
            for noisy in [false, true] {
                let gradient = model
                    .posterior_parameter_vjp(&x, &mean, &covariance, noisy)
                    .unwrap();
                let raw = model.raw_parameters();
                let objective = |trial: &SingleTaskGP| {
                    let post = trial.posterior(&x, noisy).unwrap();
                    post.mean.dot(&mean) + post.covariance.component_mul(&covariance).sum()
                };
                for j in 0..raw.len() {
                    let step = 1e-6 * raw[j].abs().max(1.0);
                    let mut plus = model.clone();
                    let mut minus = model.clone();
                    let mut p = raw.clone();
                    p[j] += step;
                    plus.set_raw_parameters(p.as_slice()).unwrap();
                    p[j] -= 2.0 * step;
                    minus.set_raw_parameters(p.as_slice()).unwrap();
                    let fd = (objective(&plus) - objective(&minus)) / (2.0 * step);
                    assert!(
                        (fd - gradient[j]).abs() < 3e-6 * fd.abs().max(1.0),
                        "case {case}, noise {noisy}, parameter {j}: {} vs FD {fd}",
                        gradient[j]
                    );
                }
            }
        }
    }

    #[test]
    fn posterior_parameter_adjoint_shapes_and_empty_query() {
        let model = models().remove(0);
        let x = Matrix::zeros(0, 2);
        let gradient = model
            .posterior_parameter_vjp(&x, &Vector::zeros(0), &Matrix::zeros(0, 0), true)
            .unwrap();
        assert_eq!(gradient, Vector::zeros(model.raw_parameters().len()));
        assert!(model
            .posterior_parameter_vjp(&x, &Vector::zeros(1), &Matrix::zeros(0, 0), false)
            .is_err());
    }
}
