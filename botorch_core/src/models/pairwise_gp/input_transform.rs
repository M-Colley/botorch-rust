//! Trial-dependent input transforms and exact Laplace evidence pullbacks.

use super::*;

impl PairwiseGP {
    pub(super) fn kernel_raw_parameters(&self) -> &[f64] {
        &self.raw[..num_raw_parameters(&self.covar_module)]
    }

    /// A training forward at the trial concentrations, differentiated through
    /// the same two Newton steps as BoTorch (the fsolve start is held fixed).
    pub(super) fn evaluate_trial(
        &self,
        raw: &[f64],
        x0: &[f64],
        with_grad: bool,
    ) -> Result<(f64, Option<Vec<f64>>, LaplaceState)> {
        self.evaluate_trial_from_start(raw, Start::Warm(x0), with_grad)
    }

    pub(super) fn evaluate_trial_from_start(
        &self,
        raw: &[f64],
        start: Start<'_>,
        with_grad: bool,
    ) -> Result<(f64, Option<Vec<f64>>, LaplaceState)> {
        if raw.len() != self.raw.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                self.raw.len(),
                raw.len()
            )));
        }
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| BotorchError::Unsupported("the model has no data".into()))?;
        let kernel_parameters = num_raw_parameters(&self.covar_module);
        let kernel_raw = &raw[..kernel_parameters];
        let mut transform = self
            .input_transform
            .as_ref()
            .map(|chain| {
                let modes = vec![true; chain.training().len()];
                chain
                    .with_params(&raw[kernel_parameters..])?
                    .with_training(modes)
            })
            .transpose()?;
        let (points, tape) = match &mut transform {
            Some(chain) => {
                chain.fit(&data.datapoints)?;
                let (points, tape) =
                    chain.transform_with_tape_mode(&data.datapoints, TransformMode::Train)?;
                (points, Some(tape))
            }
            None => (data.datapoints.clone(), None),
        };
        let problem = self.problem(&points);
        let mut lap = laplace(&problem, &self.covar_module, kernel_raw, start)?;
        check_training_posterior(&lap.state, problem.jitter)?;
        let (value, covariance_bar) = mll_data(&problem, &lap, with_grad)?;
        let (mut prior, mut gradient) = priors_and_raw_grad(
            &self.covar_module,
            &lap.covar_built,
            &lap.prepared,
            kernel_raw,
            covariance_bar.as_ref(),
            &lap.state.nan_grads,
        )?;
        if let Some(chain) = transform {
            let (input_prior, mut input_gradient) = chain.log_prior_and_grad();
            prior += input_prior;
            if let Some(weight) = covariance_bar {
                if chain.num_params() > 0 {
                    let mut point_bar = Matrix::zeros(points.nrows(), points.ncols());
                    for row in 0..points.nrows() {
                        let (_, derivatives) =
                            lap.covar_built
                                .cross_with_grad(&lap.prepared, row, &lap.prepared);
                        // Both occurrences of a training point contribute
                        // to the symmetric training covariance contraction.
                        let bar = derivatives.tr_mul(&weight.row(row).transpose()) * 2.0;
                        point_bar.row_mut(row).copy_from(&bar.transpose());
                    }
                    let (_, data_gradient) = chain.backward_with_params(
                        &tape.expect("the chain recorded its training transform"),
                        0..data.datapoints.nrows(),
                        &point_bar,
                    )?;
                    for (target, value) in input_gradient.iter_mut().zip(data_gradient) {
                        *target += value;
                    }
                }
                gradient
                    .as_mut()
                    .expect("kernel gradient was requested")
                    .extend(input_gradient);
            }
        }
        lap.state.raw = raw.to_vec();
        Ok((value + prior, gradient, lap.state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{KernelKind, KernelNode};
    use crate::transforms::input::InputTransformStep;
    use crate::transforms::input_steps::{NewStep, WarpStep};
    use crate::Normalize;

    fn model(tree: bool, priors: bool, likelihood: PairwiseLikelihood) -> PairwiseGP {
        let (points, comparisons) = super::super::tests::data();
        let prior = priors.then(|| Prior::log_normal(0.0, 0.3).unwrap());
        let warp = WarpStep::new(2, vec![1], 2e-5, None)
            .unwrap()
            .with_priors(prior, prior);
        let transform = ChainedInputTransform::new(vec![
            InputTransformStep::Normalize(Normalize::new(2)),
            InputTransformStep::Step(NewStep::Warp(warp)),
        ])
        .unwrap();
        let options = PairwiseGPOptions {
            likelihood,
            input_transform: Some(transform),
            covar_module: tree.then(|| {
                CovarModule::Tree(KernelNode::scale(
                    KernelNode::stationary(KernelKind::Matern52, Some(2)).unwrap(),
                ))
            }),
            ..Default::default()
        };
        let mut model = PairwiseGP::new(
            Some((points, comparisons)),
            2,
            options,
            Some(&super::super::tests::draws(6)),
        )
        .unwrap();
        let mut raw = model.raw_parameters().to_vec();
        let kernel = num_raw_parameters(model.covar_module());
        raw[kernel] = 0.8;
        raw[kernel + 1] = 1.4;
        model.set_raw_parameters(&raw).unwrap();
        model.update(None).unwrap();
        model
    }

    #[test]
    fn warped_laplace_gradient_matches_fixed_solution_finite_differences() {
        for tree in [false, true] {
            for priors in [false, true] {
                for likelihood in [PairwiseLikelihood::Probit, PairwiseLikelihood::Logit] {
                    let model = model(tree, priors, likelihood);
                    let raw = model.raw_parameters().to_vec();
                    let (_, gradient, state) = model.mll_value_and_grad(&raw, true).unwrap();
                    let gradient = gradient.unwrap();
                    assert_eq!(gradient.len(), raw.len());
                    let kernel = num_raw_parameters(model.covar_module());
                    assert!(gradient[kernel..].iter().any(|g| g.abs() > 1e-5));
                    for j in 0..raw.len() {
                        let mut plus = raw.clone();
                        let mut minus = raw.clone();
                        plus[j] += 1e-6;
                        minus[j] -= 1e-6;
                        let fp = model
                            .evaluate_trial_from_start(&plus, Start::Solved(&state.x0), false)
                            .unwrap()
                            .0;
                        let fm = model
                            .evaluate_trial_from_start(&minus, Start::Solved(&state.x0), false)
                            .unwrap()
                            .0;
                        let finite = (fp - fm) / 2e-6;
                        assert!(
                            (finite - gradient[j]).abs() < 3e-6 * finite.abs().max(1.0),
                            "tree={tree}, priors={priors}, likelihood={likelihood:?}, j={j}: {} vs {finite}",
                            gradient[j]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn warped_fit_updates_concentrations_and_retains_matching_native_state() {
        let mut model = model(false, true, PairwiseLikelihood::Probit);
        let initial = model.raw_parameters().to_vec();
        let value = model.mll().unwrap();
        model
            .fit(FitOptions {
                maxiter: 100,
                max_attempts: 1,
                ..Default::default()
            })
            .unwrap();
        assert!(model.mll().unwrap() > value + 1e-4);
        let kernel = num_raw_parameters(model.covar_module());
        assert!(model.raw_parameters()[kernel..]
            .iter()
            .zip(&initial[kernel..])
            .any(|(a, b)| (a - b).abs() > 1e-4));
        assert_eq!(model.state().unwrap().raw, model.raw_parameters());
        assert_eq!(
            model.input_transform().unwrap().params(),
            model.raw_parameters()[kernel..]
        );
        assert!(!model.training());
    }
}
