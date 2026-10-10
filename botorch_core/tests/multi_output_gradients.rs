//! Central finite-difference checks of the input gradients of multi-output,
//! batched and `ModelListGP` posteriors, of acquisition functions evaluated
//! through a `ScalarizedPosteriorTransform`, and of the constrained analytic
//! acquisition functions; consistency of `forward`, `forward_many` and
//! `forward_with_grad`; and the errors of the new APIs.

use std::sync::Arc;

use botorch_core::acquisition::{
    AcquisitionFunction, ConstrainedExpectedImprovement, ExpectedImprovement,
    LogConstrainedExpectedImprovement, LogExpectedImprovement, LogProbabilityOfImprovement,
    McAcquisitionOptions, OutputConstraint, PosteriorMean, PosteriorStandardDeviation,
    ProbabilityOfImprovement, PruneOptions, QExpectedImprovement, QLogExpectedImprovement,
    QLogNoisyExpectedImprovement, QLogNoisyExpectedImprovementOptions, QSimpleRegret,
    QUpperConfidenceBound, ScalarizedModel, ScalarizedPosteriorTransform, UpperConfidenceBound,
};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::{
    fit_gpytorch_mll, BatchedInputTransform, BatchedMultiOutputGP, BatchedMultiOutputGPOptions,
    BotorchError, FitOptions, FitStatus, KernelKind, Matrix, Model, ModelListGP, Normalize,
    SingleTaskGP, SingleTaskGPOptions, Vector,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn data(seed: u64, n: usize, d: usize, m: usize) -> (Matrix, Matrix) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Matrix::from_fn(n, d, |_, _| rng.random::<f64>());
    let y = Matrix::from_fn(n, m, |i, t| {
        let u: f64 = (0..d).map(|j| x[(i, j)]).sum();
        (1.0 + t as f64) * ((3.0 + t as f64) * u).sin() + 0.4 * u * u + 0.05 * rng.random::<f64>()
    });
    (x, y)
}

/// A 3-output model with fixed hyperparameters (one value set per output).
fn multi_output() -> BatchedMultiOutputGP {
    let (x, y) = data(1, 11, 2, 3);
    let mut gp = BatchedMultiOutputGP::multi_output(
        x,
        y,
        SingleTaskGPOptions {
            input_transform: Some(Normalize::new(2)),
            kernel: KernelKind::Matern52,
            ..Default::default()
        },
    )
    .unwrap();
    // raw_noise (3), raw_constant (3), raw_lengthscale (3 x 2).
    gp.set_raw_parameters(&[
        0.02, 0.05, 0.01, 0.1, -0.2, 0.0, 0.3, 0.5, 0.45, 0.25, 0.6, 0.7,
    ])
    .unwrap();
    gp
}

/// A model list with a multi-output and a single-output member with
/// different training data.
fn model_list() -> ModelListGP {
    let (x1, y1) = data(2, 9, 2, 2);
    let mut a = BatchedMultiOutputGP::multi_output(x1, y1, SingleTaskGPOptions::default()).unwrap();
    a.set_raw_parameters(&[0.03, 0.02, 0.0, 0.1, 0.4, 0.3, 0.5, 0.6])
        .unwrap();
    let (x2, y2) = data(3, 7, 2, 1);
    let mut b = SingleTaskGP::new(
        x2,
        y2,
        SingleTaskGPOptions {
            train_yvar: Some(Matrix::from_element(7, 1, 0.01)),
            ..Default::default()
        },
    )
    .unwrap();
    b.set_raw_parameters(&[0.2, 0.35, 0.55]).unwrap();
    ModelListGP::new(vec![a, b.into()]).unwrap()
}

fn points(q: usize, seed: u64) -> Matrix {
    let mut rng = StdRng::seed_from_u64(seed);
    Matrix::from_fn(q, 2, |_, _| 0.05 + 0.9 * rng.random::<f64>())
}

/// Checks the forward-mode posterior derivatives against central differences.
fn check_posterior_fd(name: &str, model: &dyn Model, x: &Matrix) {
    let g = model.joint_posterior_with_grad(x).unwrap();
    let p = model.posterior(x, false).unwrap();
    assert!((&g.mean - &p.mean).amax() < 1e-13, "{name}: mean");
    assert!(
        (&g.covariance - &p.covariance).amax() < 1e-13,
        "{name}: covariance"
    );
    let d = x.ncols();
    for pt in 0..x.nrows() {
        for k in 0..d {
            let h = 1e-6;
            let mut xp = x.clone();
            xp[(pt, k)] += h;
            let mut xm = x.clone();
            xm[(pt, k)] -= h;
            let pp = model.posterior(&xp, false).unwrap();
            let pm = model.posterior(&xm, false).unwrap();
            let j = pt * d + k;
            let fd_mean = (&pp.mean - &pm.mean) / (2.0 * h);
            let fd_cov = (&pp.covariance - &pm.covariance) / (2.0 * h);
            assert!((fd_mean - &g.dmean[j]).amax() < 2e-6, "{name}: dmean[{j}]");
            assert!((fd_cov - &g.dcov[j]).amax() < 2e-6, "{name}: dcov[{j}]");
        }
    }
}

#[test]
fn posterior_gradients_match_finite_differences() {
    let mo = multi_output();
    check_posterior_fd("multi-output", &mo, &points(3, 4));
    let list = model_list();
    check_posterior_fd("model list", &list, &points(2, 5));
    let t = ScalarizedPosteriorTransform::new(Vector::from_vec(vec![0.7, -0.3, 1.1]), 0.5).unwrap();
    let s = ScalarizedModel::new(Arc::new(list), t).unwrap();
    check_posterior_fd("scalarized model list", &s, &points(3, 6));
    // Batched model: per batch element.
    let (x0, y0) = data(7, 8, 2, 2);
    let (x1, y1) = data(8, 8, 2, 2);
    let b = BatchedMultiOutputGP::new(
        vec![x0, x1],
        vec![y0, y1],
        vec![2],
        BatchedMultiOutputGPOptions {
            input_transform: Some(BatchedInputTransform::PerBatch(vec![Normalize::new(2); 2])),
            ..Default::default()
        },
    )
    .unwrap();
    let x = points(2, 9);
    let g = b.joint_posterior_with_grad_at(1, &x).unwrap();
    let h = 1e-6;
    let mut xp = x.clone();
    xp[(1, 0)] += h;
    let mut xm = x.clone();
    xm[(1, 0)] -= h;
    let fd = (b.posterior_at(1, &xp, false).unwrap().covariance
        - b.posterior_at(1, &xm, false).unwrap().covariance)
        / (2.0 * h);
    assert!((fd - &g.dcov[2]).amax() < 2e-6);
}

/// Checks `forward_with_grad` against central differences of `forward`.
fn check_acqf(name: &str, acqf: &dyn AcquisitionFunction, x: &Matrix) {
    let (v, g) = acqf.forward_with_grad(x).unwrap();
    let v2 = acqf.forward(x).unwrap();
    assert!(
        (v - v2).abs() <= 1e-12 * v.abs().max(1e-12),
        "{name}: forward {v2} vs forward_with_grad {v}"
    );
    // Central differences have a round-off floor of about ε|f|/h.
    let scale = g.amax().max(1e-8);
    for i in 0..x.nrows() {
        for k in 0..x.ncols() {
            let h = 1e-6;
            let mut xp = x.clone();
            xp[(i, k)] += h;
            let mut xm = x.clone();
            xm[(i, k)] -= h;
            let fd = (acqf.forward(&xp).unwrap() - acqf.forward(&xm).unwrap()) / (2.0 * h);
            assert!(
                (g[(i, k)] - fd).abs() <= 1e-5 * scale + 1e-6 * fd.abs() + 1e-9 * v.abs().max(1.0),
                "{name}: d/dx[{i},{k}] analytic {} vs fd {fd}",
                g[(i, k)]
            );
        }
    }
}

fn sampler(seed: u64) -> Option<Arc<SobolQMCNormalSampler>> {
    Some(Arc::new(
        SobolQMCNormalSampler::new(128, Some(seed)).unwrap(),
    ))
}

type Acqfs = Vec<(&'static str, Box<dyn AcquisitionFunction>)>;

#[test]
fn scalarized_acquisition_gradients_match_finite_differences() {
    let models: Vec<(&str, Arc<dyn Model>, Vec<f64>)> = vec![
        (
            "multi-output",
            Arc::new(multi_output()),
            vec![0.6, -0.4, 0.25],
        ),
        ("model list", Arc::new(model_list()), vec![-0.5, 1.0, 0.75]),
    ];
    for (mname, model, w) in models {
        let t = ScalarizedPosteriorTransform::new(Vector::from_vec(w), 0.3).unwrap();
        let s = ScalarizedModel::wrap(model.clone(), Some(t)).unwrap();
        let xs: Vec<Matrix> = (0..4).map(|i| points(1, 20 + i)).collect();
        let (mean, _) = s
            .posterior_marginals(&Matrix::from_fn(4, 2, |i, j| xs[i][(0, j)]))
            .unwrap();
        let best = mean.max();
        let m = || s.clone();
        let analytic: Acqfs = vec![
            (
                "EI",
                Box::new(ExpectedImprovement::new(m(), best, true).unwrap()),
            ),
            (
                "LogEI",
                Box::new(LogExpectedImprovement::new(m(), best + 0.5, true).unwrap()),
            ),
            (
                "PI",
                Box::new(ProbabilityOfImprovement::new(m(), best, false).unwrap()),
            ),
            (
                "LogPI",
                Box::new(LogProbabilityOfImprovement::new(m(), best, true).unwrap()),
            ),
            (
                "UCB",
                Box::new(UpperConfidenceBound::new(m(), 0.7, true).unwrap()),
            ),
            ("PM", Box::new(PosteriorMean::new(m(), false).unwrap())),
            (
                "PSD",
                Box::new(PosteriorStandardDeviation::new(m(), true).unwrap()),
            ),
        ];
        for (aname, acqf) in &analytic {
            let many = acqf.forward_many(&xs).unwrap();
            for (i, x) in xs.iter().enumerate() {
                check_acqf(&format!("{mname} {aname}"), acqf.as_ref(), x);
                assert_eq!(many[i], acqf.forward(x).unwrap(), "{mname} {aname}");
            }
        }
        let opts = || McAcquisitionOptions {
            sampler: sampler(3),
            x_pending: Some(points(2, 40)),
        };
        let mc: Acqfs = vec![
            (
                "qEI",
                Box::new(QExpectedImprovement::new(m(), best - 0.3, opts()).unwrap()),
            ),
            (
                "qLogEI",
                Box::new(QLogExpectedImprovement::new(m(), best, Default::default()).unwrap()),
            ),
            (
                "qUCB",
                Box::new(QUpperConfidenceBound::new(m(), 0.5, opts()).unwrap()),
            ),
            ("qSR", Box::new(QSimpleRegret::new(m(), opts()).unwrap())),
            (
                "qLogNEI",
                Box::new(
                    QLogNoisyExpectedImprovement::new(
                        m(),
                        &points(5, 41),
                        QLogNoisyExpectedImprovementOptions {
                            sampler: sampler(4),
                            prune_options: PruneOptions {
                                seed: Some(0),
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                ),
            ),
        ];
        for (aname, acqf) in &mc {
            check_acqf(&format!("{mname} {aname}"), acqf.as_ref(), &points(2, 50));
        }
    }
}

#[test]
fn constrained_gradients_match_finite_differences() {
    let models: Vec<(&str, Arc<dyn Model>)> = vec![
        ("multi-output", Arc::new(multi_output())),
        ("model list", Arc::new(model_list())),
    ];
    for (mname, model) in models {
        let xs: Vec<Matrix> = (0..5).map(|i| points(1, 60 + i)).collect();
        let configs: Vec<(usize, Vec<OutputConstraint>, bool)> = vec![
            (0, vec![(1, Some(0.2), None)], true),
            (1, vec![(0, None, Some(0.8))], false),
            (
                0,
                vec![(1, Some(-0.5), Some(1.0)), (2, Some(0.0), None)],
                true,
            ),
            (
                2,
                vec![(0, Some(-1.0), Some(-0.2)), (1, None, Some(2.0))],
                true,
            ),
            // Feasibility deep in the tail.
            (0, vec![(2, Some(9.0), None)], true),
        ];
        for (obj, cons, maximize) in configs {
            let log = LogConstrainedExpectedImprovement::new(
                model.clone(),
                0.5,
                obj,
                cons.clone(),
                maximize,
            )
            .unwrap();
            let plain = ConstrainedExpectedImprovement::new(
                model.clone(),
                0.5,
                obj,
                cons.clone(),
                maximize,
            )
            .unwrap();
            assert!(log.is_log() && plain.is_nonnegative());
            let many = log.forward_many(&xs).unwrap();
            let many_plain = plain.forward_many(&xs).unwrap();
            for (i, x) in xs.iter().enumerate() {
                let name = format!("{mname} obj={obj} cons={cons:?}");
                check_acqf(&format!("{name} LogCEI"), &log, x);
                check_acqf(&format!("{name} CEI"), &plain, x);
                // CEI = exp(LogCEI) where it does not underflow.
                if many[i] > -300.0 {
                    let rel = (many_plain[i] - many[i].exp()).abs() / many[i].exp();
                    assert!(
                        rel < 1e-12,
                        "{name}: {} vs {}",
                        many_plain[i],
                        many[i].exp()
                    );
                }
            }
        }
    }
}

#[test]
fn invalid_arguments_are_rejected() {
    let model: Arc<dyn Model> = Arc::new(multi_output());
    let bad = |cons: Vec<OutputConstraint>, obj: usize| {
        LogConstrainedExpectedImprovement::new(model.clone(), 0.0, obj, cons, true).err()
    };
    assert!(matches!(
        bad(vec![], 0),
        Some(BotorchError::InvalidArgument(_))
    ));
    assert!(matches!(
        bad(vec![(0, Some(0.0), None)], 0),
        Some(BotorchError::InvalidArgument(_))
    ));
    assert!(matches!(
        bad(vec![(1, Some(1.0), Some(0.5))], 0),
        Some(BotorchError::InvalidArgument(_))
    ));
    assert!(matches!(
        bad(vec![(5, Some(0.0), None)], 0),
        Some(BotorchError::InvalidArgument(_))
    ));
    assert!(matches!(
        bad(vec![(1, Some(0.0), None)], 7),
        Some(BotorchError::InvalidArgument(_))
    ));
    assert!(matches!(
        bad(vec![(1, Some(0.0), None), (1, None, Some(1.0))], 0),
        Some(BotorchError::InvalidArgument(_))
    ));
    // (None, None) constraints are accepted and ignored, as in BoTorch.
    let a = LogConstrainedExpectedImprovement::new(
        model.clone(),
        0.0,
        0,
        vec![(1, None, None), (2, Some(0.0), None)],
        true,
    )
    .unwrap();
    let b = LogConstrainedExpectedImprovement::new(
        model.clone(),
        0.0,
        0,
        vec![(2, Some(0.0), None)],
        true,
    )
    .unwrap();
    let x = points(1, 1);
    assert_eq!(a.forward(&x).unwrap(), b.forward(&x).unwrap());
    // Multi-output models need a posterior transform.
    assert!(matches!(
        ExpectedImprovement::new(model.clone(), 0.0, true),
        Err(BotorchError::Unsupported(_))
    ));
    assert!(matches!(
        QSimpleRegret::new(model.clone(), McAcquisitionOptions::default()),
        Err(BotorchError::Unsupported(_))
    ));
    assert!(model.posterior_marginals(&x).is_err());
    assert!(model.posterior_q1_with_grad(&[0.5, 0.5]).is_err());
    // The transform needs one weight per output.
    let t = ScalarizedPosteriorTransform::new(Vector::from_vec(vec![1.0, 2.0]), 0.0).unwrap();
    assert!(matches!(
        ScalarizedModel::new(model.clone(), t),
        Err(BotorchError::DimensionMismatch(_))
    ));
    assert!(ScalarizedPosteriorTransform::new(Vector::from_vec(vec![f64::NAN]), 0.0).is_err());
    // Batched models cannot be used by acquisition functions.
    let (x0, y0) = data(1, 6, 2, 1);
    let (x1, y1) = data(2, 6, 2, 1);
    let batched: Arc<dyn Model> = Arc::new(
        BatchedMultiOutputGP::new(vec![x0, x1], vec![y0, y1], vec![2], Default::default()).unwrap(),
    );
    assert_eq!(batched.batch_shape(), &[2]);
    assert!(matches!(
        ExpectedImprovement::new(batched.clone(), 0.0, true),
        Err(BotorchError::Unsupported(_))
    ));
    assert!(matches!(
        LogConstrainedExpectedImprovement::new(
            batched.clone(),
            0.0,
            0,
            vec![(1, None, Some(1.0))],
            true
        ),
        Err(BotorchError::Unsupported(_))
    ));
    assert!(batched.posterior(&x, false).is_err());
    // Inconsistent training data.
    let (xa, ya) = data(1, 6, 2, 2);
    assert!(BatchedMultiOutputGP::new(
        vec![xa.clone()],
        vec![ya.clone()],
        vec![2],
        Default::default()
    )
    .is_err());
    assert!(BatchedMultiOutputGP::new(
        vec![xa.clone(), xa.clone()],
        vec![ya.clone(), ya.rows(0, 5).into_owned()],
        vec![2],
        Default::default()
    )
    .is_err());
    assert!(BatchedMultiOutputGP::multi_output(
        xa.clone(),
        ya.clone(),
        SingleTaskGPOptions {
            train_yvar: Some(Matrix::from_element(6, 1, 0.1)),
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn fits_are_independent_per_element() {
    // Fitting a multi-output model equals fitting every output on its own.
    let (x, y) = data(11, 12, 2, 2);
    let mut gp =
        BatchedMultiOutputGP::multi_output(x.clone(), y.clone(), SingleTaskGPOptions::default())
            .unwrap();
    let res = fit_gpytorch_mll(&mut gp, FitOptions::default()).unwrap();
    assert_eq!(res.results.len(), 2);
    assert_eq!(res.status, FitStatus::Success);
    assert_eq!(res.message, "Batched L-BFGS-B: 2/2 outputs converged.");
    for t in 0..2 {
        let mut single = SingleTaskGP::new(
            x.clone(),
            y.columns(t, 1).into_owned(),
            SingleTaskGPOptions::default(),
        )
        .unwrap();
        // BoTorch's batched L-BFGS-B does not limit function evaluations.
        let r = fit_gpytorch_mll(
            &mut single,
            FitOptions {
                maxfun: usize::MAX,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.loss, res.results[t].loss);
        assert_eq!(r.iterations, res.results[t].iterations);
        assert_eq!(single.raw_parameters(), gp.model(0, t).raw_parameters());
    }
    assert!((res.loss - res.results.iter().map(|r| r.loss).sum::<f64>()).abs() < 1e-15);
    // A model list fits its members.
    let mut list = model_list();
    let before = list.models()[1].raw_parameters();
    let results = fit_gpytorch_mll(&mut list, FitOptions::default()).unwrap();
    assert_eq!(results.len(), 2);
    assert_ne!(list.models()[1].raw_parameters(), before);
    // A failed fit leaves the model unchanged.
    let mut gp2 = BatchedMultiOutputGP::multi_output(x, y, SingleTaskGPOptions::default()).unwrap();
    let initial = gp2.raw_parameters();
    let err = fit_gpytorch_mll(
        &mut gp2,
        FitOptions {
            max_attempts: 0,
            ..Default::default()
        },
    );
    assert!(err.is_err());
    assert_eq!(gp2.raw_parameters(), initial);
}
