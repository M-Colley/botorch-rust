//! Edge cases and error handling: tiny and degenerate data sets, duplicate
//! inputs (jittered Cholesky), fixed noise, inputs outside the unit cube,
//! invalid inputs, and the retry logic of `fit_gpytorch_mll`.

use botorch_core::kernels::get_matern_kernel_with_gamma_prior;
use botorch_core::{
    fit_gpytorch_mll, BotorchError, ExactMarginalLogLikelihood, FitOptions, FitStatus, Matrix,
    Normalize, OutcomeTransformChoice, SingleTaskGP, SingleTaskGPOptions, Standardize, Vector,
};

fn grid(n: usize, d: usize) -> Matrix {
    Matrix::from_fn(n, d, |i, j| ((i * (j + 3) + 1) % (n + 1)) as f64 / n as f64)
}

fn response(x: &Matrix) -> Matrix {
    Matrix::from_fn(x.nrows(), 1, |i, _| {
        x.row(i)
            .iter()
            .enumerate()
            .map(|(j, v)| ((j + 1) as f64 * 2.0 * v).sin())
            .sum::<f64>()
    })
}

#[test]
fn single_training_point() {
    let x = Matrix::from_row_slice(1, 2, &[0.3, 0.7]);
    let y = Matrix::from_element(1, 1, 4.2);
    let mut model = SingleTaskGP::new(x.clone(), y, SingleTaskGPOptions::default()).unwrap();
    // Standardize with n = 1: std = 1, the transformed target is 0.
    assert_eq!(model.outcome_transform().unwrap().stdvs(), &[1.0]);
    assert_eq!(model.transformed_train_targets()[0], 0.0);
    let fit = fit_gpytorch_mll(&mut model, FitOptions::default()).unwrap();
    assert_eq!(fit.status, FitStatus::Success);
    let post = model.posterior(&x, false).unwrap();
    assert!((post.mean[0] - 4.2).abs() < 1e-2);
    let far = model
        .posterior(&Matrix::from_row_slice(1, 2, &[50.0, -50.0]), false)
        .unwrap();
    // Far away the posterior reverts to the prior (mean = 4.2 + c, variance = 1).
    assert!((far.covariance[(0, 0)] - 1.0).abs() < 1e-12);
}

#[test]
fn constant_targets_are_only_demeaned() {
    let x = grid(6, 2);
    let y = Matrix::from_element(6, 1, -3.0);
    let mut model = SingleTaskGP::new(x.clone(), y, SingleTaskGPOptions::default()).unwrap();
    let st = model.outcome_transform().unwrap();
    assert_eq!(st.means(), &[-3.0]);
    assert_eq!(st.stdvs(), &[1.0]);
    assert!(model.transformed_train_targets().iter().all(|&v| v == 0.0));
    fit_gpytorch_mll(&mut model, FitOptions::default()).unwrap();
    let post = model.posterior(&x, false).unwrap();
    assert!(post.mean.iter().all(|m| (m + 3.0).abs() < 1e-6));
}

#[test]
fn duplicate_inputs_with_zero_noise_use_jitter() {
    // Three identical inputs make K(X, X) exactly singular; with a zero noise
    // variance the Cholesky factorization needs jitter (1e-8), like GPyTorch's
    // psd_safe_cholesky.
    let x = Matrix::from_row_slice(4, 2, &[0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.9, 0.1]);
    let y = Matrix::from_column_slice(4, 1, &[1.0, 1.0, 1.0, -1.0]);
    let mut model = SingleTaskGP::new(x.clone(), y, SingleTaskGPOptions::default()).unwrap();
    let mut h = model.hyperparameters().clone();
    h.raw_noise = Some(0.0);
    model.set_hyperparameters(h).unwrap();
    let chol = model.train_cholesky().unwrap();
    assert_eq!(chol.jitter(), 1e-8);
    // Compare with an explicit computation using K + 1e-8 I.
    let kernel = model.kernel().unwrap();
    let xt = model.transformed_train_inputs();
    let k = kernel.covariance_symmetric(xt).unwrap() + Matrix::identity(4, 4) * 1e-8;
    let xs = Matrix::from_row_slice(2, 2, &[0.5, 0.5, 0.2, 0.4]);
    let ks = kernel.covariance(&xs, xt).unwrap();
    // (Solve with an LU factorization: K is too ill-conditioned for an explicit inverse.)
    let lu = k.clone().lu();
    let ytf = model.transformed_train_targets().clone();
    let st = model.outcome_transform().unwrap();
    let (mu, sd) = (st.means()[0], st.stdvs()[0]);
    let mean_tf = &ks * lu.solve(&ytf).unwrap();
    let cov_tf =
        kernel.covariance_symmetric(&xs).unwrap() - &ks * lu.solve(&ks.transpose()).unwrap();
    let post = model.posterior(&xs, false).unwrap();
    for i in 0..2 {
        assert!((post.mean[i] - (mu + sd * mean_tf[i])).abs() < 1e-6);
        for j in 0..2 {
            assert!((post.covariance[(i, j)] - sd * sd * cov_tf[(i, j)]).abs() < 1e-6);
        }
    }
    // The LogNormal noise prior has no density at zero noise (torch raises a
    // support error, here the value is NaN), so the MLL is not finite ...
    assert!(!ExactMarginalLogLikelihood::new(&model)
        .value()
        .unwrap()
        .is_finite());
    // ... but fitting starts from the projection onto the noise bound (1e-4).
    let fit = fit_gpytorch_mll(&mut model, FitOptions::default()).unwrap();
    assert!(fit.loss.is_finite());
    assert!(model.likelihood_noise()[0] >= 1e-4 * (1.0 - 1e-7));
}

#[test]
fn one_dimensional_inputs() {
    let x = Matrix::from_column_slice(8, 1, &[0.05, 0.2, 0.33, 0.4, 0.61, 0.7, 0.88, 0.95]);
    let y = response(&x);
    let mut model = SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap();
    fit_gpytorch_mll(&mut model, FitOptions::default()).unwrap();
    let (mean, var, dmean, dvar) = model.posterior_q1_with_grad(&[0.5]).unwrap();
    assert!(mean.is_finite() && var > 0.0 && dmean.len() == 1 && dvar.len() == 1);
}

#[test]
fn fixed_noise_observation_noise_is_the_mean_training_noise() {
    let x = grid(7, 2);
    let y = response(&x);
    let yvar = Matrix::from_fn(7, 1, |i, _| 0.01 * (i + 1) as f64);
    let model = SingleTaskGP::new(
        x,
        y,
        SingleTaskGPOptions {
            train_yvar: Some(yvar.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(model.hyperparameters().raw_noise.is_none());
    assert_eq!(model.parameter_layout().len(), 1 + 2);
    let mean_yvar = yvar.sum() / 7.0;
    assert!((model.observation_noise() - mean_yvar).abs() < 1e-14);
    let xs = Matrix::from_row_slice(2, 2, &[0.1, 0.9, 0.5, 0.5]);
    let p = model.posterior(&xs, false).unwrap();
    let pn = model.posterior(&xs, true).unwrap();
    let diff = pn.covariance - p.covariance;
    assert!((diff[(0, 0)] - mean_yvar).abs() < 1e-12);
    assert!((diff[(1, 1)] - mean_yvar).abs() < 1e-12);
    assert!(diff[(0, 1)].abs() < 1e-15);
}

#[test]
fn zero_observation_variances_are_clamped() {
    let x = grid(5, 2);
    let y = response(&x);
    let model = SingleTaskGP::new(
        x,
        y,
        SingleTaskGPOptions {
            train_yvar: Some(Matrix::zeros(5, 1)),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(model
        .likelihood_noise()
        .iter()
        .all(|&v| v == botorch_core::models::MIN_FIXED_NOISE));
}

#[test]
fn inputs_outside_the_unit_cube_with_normalize() {
    let x = grid(9, 2).map(|v| -20.0 + 60.0 * v);
    let y = response(&x.map(|v| v / 40.0));
    let mut model = SingleTaskGP::new(
        x.clone(),
        y,
        SingleTaskGPOptions {
            input_transform: Some(Normalize::new(2)),
            ..Default::default()
        },
    )
    .unwrap();
    let xt = model.transformed_train_inputs();
    assert!(xt.min() >= 0.0 && xt.max() <= 1.0);
    fit_gpytorch_mll(&mut model, FitOptions::default()).unwrap();
    // Raw-space test points far outside the training range.
    let xs = Matrix::from_row_slice(2, 2, &[100.0, -100.0, 45.0, 0.0]);
    let post = model.posterior(&xs, true).unwrap();
    assert!(post.mean.iter().all(|m| m.is_finite()));
    assert!(post.variance().iter().all(|&v| v > 0.0));
    let g = model.joint_posterior_with_grad(&xs).unwrap();
    assert_eq!(g.dmean.len(), 4);
}

#[test]
fn normalize_ignores_constant_columns() {
    let mut x = grid(6, 2);
    for i in 0..6 {
        x[(i, 1)] = 3.5;
    }
    let y = response(&x);
    let model = SingleTaskGP::new(
        x,
        y,
        SingleTaskGPOptions {
            input_transform: Some(Normalize::new(2)),
            ..Default::default()
        },
    )
    .unwrap();
    let tf = model.input_transform().unwrap();
    assert_eq!(tf.offset()[1], 0.0);
    assert_eq!(tf.coefficient()[1], 1.0);
    assert!(model
        .transformed_train_inputs()
        .column(1)
        .iter()
        .all(|&v| v == 3.5));
}

#[test]
fn custom_standardize_and_no_outcome_transform() {
    let x = grid(5, 1);
    let y = response(&x).map(|v| 1e-9 * v);
    // min_stdv above the data's standard deviation: only de-mean.
    let model = SingleTaskGP::new(
        x.clone(),
        y.clone(),
        SingleTaskGPOptions {
            outcome_transform: OutcomeTransformChoice::Standardize(
                Standardize::new().with_min_stdv(1e-6),
            ),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(model.outcome_transform().unwrap().stdvs(), &[1.0]);
    let raw = SingleTaskGP::new(
        x,
        y.clone(),
        SingleTaskGPOptions {
            outcome_transform: OutcomeTransformChoice::None,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(raw.outcome_transform().is_none());
    assert_eq!(raw.transformed_train_targets().as_slice(), y.as_slice());
}

#[test]
fn empty_batches_of_test_points() {
    let x = grid(5, 2);
    let model = SingleTaskGP::new(x.clone(), response(&x), SingleTaskGPOptions::default()).unwrap();
    let post = model.posterior(&Matrix::zeros(0, 2), true).unwrap();
    assert!(post.is_empty());
    let g = model
        .joint_posterior_with_grad(&Matrix::zeros(0, 2))
        .unwrap();
    assert!(g.dmean.is_empty() && g.dcov.is_empty());
}

#[test]
fn invalid_inputs_are_rejected() {
    let x = grid(5, 2);
    let y = response(&x);
    let err = SingleTaskGP::new(
        x.clone(),
        Matrix::zeros(5, 2),
        SingleTaskGPOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, BotorchError::Unsupported(_)));
    let err = SingleTaskGP::new(
        x.clone(),
        Matrix::zeros(4, 1),
        SingleTaskGPOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, BotorchError::DimensionMismatch(_)));
    let mut xn = x.clone();
    xn[(2, 1)] = f64::NAN;
    let err = SingleTaskGP::new(xn, y.clone(), SingleTaskGPOptions::default()).unwrap_err();
    assert!(matches!(err, BotorchError::InputData(_)));
    let err = SingleTaskGP::new(
        x.clone(),
        y.clone(),
        SingleTaskGPOptions {
            train_yvar: Some(Matrix::from_element(5, 1, -0.1)),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, BotorchError::InputData(_)));
    let err = SingleTaskGP::new(
        x.clone(),
        y.clone(),
        SingleTaskGPOptions {
            input_transform: Some(Normalize::new(3)),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, BotorchError::DimensionMismatch(_)));
    let mut model = SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap();
    assert!(matches!(
        model.posterior(&Matrix::zeros(1, 3), false),
        Err(BotorchError::DimensionMismatch(_))
    ));
    assert!(matches!(
        model.posterior_q1_with_grad(&[0.1]),
        Err(BotorchError::DimensionMismatch(_))
    ));
    assert!(model.set_raw_parameters(&[0.1, 0.0]).is_err());
    assert!(model.set_raw_parameters(&[0.1, 0.0, -1.0, 0.3]).is_err());
    let mut h = model.hyperparameters().clone();
    h.raw_outputscale = Some(1.0);
    assert!(matches!(
        model.set_hyperparameters(h),
        Err(BotorchError::DimensionMismatch(_))
    ));
}

/// A model whose initial hyperparameters make the training covariance
/// numerically singular even with jitter (huge outputscale and duplicated
/// inputs), so that the first fitting attempt fails with a non-PSD error.
fn failing_model() -> SingleTaskGP {
    let mut x = grid(6, 2);
    for j in 0..2 {
        x[(5, j)] = x[(0, j)];
    }
    let y = response(&x);
    let mut model = SingleTaskGP::new(
        x,
        y,
        SingleTaskGPOptions {
            train_yvar: Some(Matrix::from_element(6, 1, 1e-6)),
            covar_module: Some(get_matern_kernel_with_gamma_prior(2)),
            ..Default::default()
        },
    )
    .unwrap();
    let mut h = model.hyperparameters().clone();
    h.raw_outputscale = Some(1e12);
    model.set_hyperparameters(h).unwrap();
    assert!(matches!(
        model.train_cholesky(),
        Err(BotorchError::NotPsd { .. })
    ));
    model
}

#[test]
fn fit_retries_from_prior_samples_after_failures() {
    let mut model = failing_model();
    let fit = fit_gpytorch_mll(&mut model, FitOptions::default()).unwrap();
    assert!(fit.attempts > 1, "expected a retry, got {fit:?}");
    assert_ne!(fit.status, FitStatus::Failure);
    assert!(model.outputscale().unwrap() < 1e6);
    assert!(model.train_cholesky().is_ok());
}

#[test]
fn fit_reports_failure_and_keeps_parameters_when_all_attempts_fail() {
    let mut model = failing_model();
    let before = model.raw_parameters();
    let err = fit_gpytorch_mll(
        &mut model,
        FitOptions {
            max_attempts: 1,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, BotorchError::ModelFitting(_)), "{err}");
    assert_eq!(model.raw_parameters(), before);
}

#[test]
fn fit_respects_the_iteration_limit() {
    let x = grid(10, 3);
    let y = response(&x);
    let mut model = SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap();
    let fit = fit_gpytorch_mll(
        &mut model,
        FitOptions {
            maxiter: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(fit.status, FitStatus::Stopped);
    assert_eq!(fit.iterations, 2);
    // Fitting again continues from the current values and improves the loss.
    let fit2 = fit_gpytorch_mll(&mut model, FitOptions::default()).unwrap();
    assert!(fit2.loss <= fit.loss);
    let lengthscale: Vector = model.lengthscale();
    assert!(lengthscale.iter().all(|&l| l >= 2.5e-2 * (1.0 - 1e-7)));
}

#[test]
fn models_can_be_shared_across_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SingleTaskGP>();
    assert_send_sync::<ExactMarginalLogLikelihood<'static>>();
    // The lazily computed prediction cache is shared safely.
    let x = grid(6, 2);
    let model = SingleTaskGP::new(x.clone(), response(&x), SingleTaskGPOptions::default()).unwrap();
    let xs = Matrix::from_row_slice(1, 2, &[0.3, 0.3]);
    let means: Vec<f64> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| s.spawn(|| model.posterior(&xs, false).unwrap().mean[0]))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(means.iter().all(|&m| m == means[0]));
}
