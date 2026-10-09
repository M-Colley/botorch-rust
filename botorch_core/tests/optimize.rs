//! Behavior of `optimize_acqf`: fixed features, sequential greedy
//! optimization with pending points, explicit initial conditions,
//! determinism given a seed, and error handling.

use std::collections::BTreeMap;
use std::sync::Arc;

use botorch_core::acquisition::{
    AcquisitionFunction, LogExpectedImprovement, McAcquisitionOptions, QExpectedImprovement,
    QLogExpectedImprovement, QLogExpectedImprovementOptions, QLogNoisyExpectedImprovement,
    QLogNoisyExpectedImprovementOptions, QUpperConfidenceBound,
};
use botorch_core::optim::{optimize_acqf, optimize_acqf_detailed, OptimizeAcqfOptions};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::utils::sampling::draw_sobol_samples;
use botorch_core::{BotorchError, Matrix, Model, SingleTaskGP, SingleTaskGPOptions};

fn bounds() -> Matrix {
    Matrix::from_row_slice(2, 3, &[0.0, -1.0, 0.0, 1.0, 1.0, 2.0])
}

fn model() -> Arc<dyn Model> {
    let b = bounds();
    let x = draw_sobol_samples(&b, 15, 1, Some(1)).unwrap();
    let x = Matrix::from_fn(15, 3, |i, j| x[i][(0, j)]);
    let y = Matrix::from_fn(15, 1, |i, _| {
        -(x[(i, 0)] - 0.3).powi(2) - (x[(i, 1)] + 0.2).powi(2) - 0.5 * (x[(i, 2)] - 1.4).powi(2)
    });
    let mut m = SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap();
    m.set_raw_parameters(&[0.01, 0.0, 0.6, 0.8, 1.0]).unwrap();
    Arc::new(m)
}

fn seeded(seed: u64) -> OptimizeAcqfOptions {
    OptimizeAcqfOptions {
        seed: Some(seed),
        ..Default::default()
    }
}

#[test]
fn optimize_logei_is_deterministic_and_within_bounds() {
    let m = model();
    let best_f = m.train_targets().max();
    let acqf = LogExpectedImprovement::new(m, best_f, true).unwrap();
    let b = bounds();
    let (x1, v1) = optimize_acqf(&acqf, &b, 1, 8, 256, &seeded(3)).unwrap();
    let (x2, v2) = optimize_acqf(&acqf, &b, 1, 8, 256, &seeded(3)).unwrap();
    assert_eq!(x1, x2);
    assert_eq!(v1, v2);
    for k in 0..3 {
        assert!(x1[(0, k)] >= b[(0, k)] && x1[(0, k)] <= b[(1, k)]);
    }
    // The optimum is at least as good as the best raw sample.
    let raw = draw_sobol_samples(&b, 256, 1, Some(3)).unwrap();
    let raw_best = acqf
        .forward_many(&raw)
        .unwrap()
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(v1 >= raw_best);
    // Single-threaded execution gives the same result.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (x3, v3) = pool.install(|| optimize_acqf(&acqf, &b, 1, 8, 256, &seeded(3)).unwrap());
    assert_eq!((x3, v3), (x1, v1));
}

#[test]
fn fixed_features_are_respected() {
    let m = model();
    let acqf = QUpperConfidenceBound::new(
        m,
        0.5,
        McAcquisitionOptions {
            sampler: Some(Arc::new(SobolQMCNormalSampler::new(64, Some(1)).unwrap())),
            ..Default::default()
        },
    )
    .unwrap();
    let b = bounds();
    let mut fixed = BTreeMap::new();
    fixed.insert(1, 0.25);
    let options = OptimizeAcqfOptions {
        fixed_features: fixed.clone(),
        ..seeded(0)
    };
    let res = optimize_acqf_detailed(&acqf, &b, 2, 4, 64, &options).unwrap();
    assert_eq!(res.candidates.shape(), (2, 3));
    assert!(res.candidates.column(1).iter().all(|&v| v == 0.25));
    for r in &res.restarts {
        assert!(r.candidates.column(1).iter().all(|&v| v == 0.25));
    }
    // All features fixed: the fixed point is returned.
    fixed.insert(0, 0.5);
    fixed.insert(2, 1.0);
    let options = OptimizeAcqfOptions {
        fixed_features: fixed,
        ..seeded(0)
    };
    let (x, v) = optimize_acqf(&acqf, &b, 2, 4, 64, &options).unwrap();
    assert_eq!(
        x,
        Matrix::from_row_slice(2, 3, &[0.5, 0.25, 1.0, 0.5, 0.25, 1.0])
    );
    assert_eq!(v, acqf.forward(&x).unwrap());
    // Out-of-range feature index.
    let mut bad = BTreeMap::new();
    bad.insert(7, 0.0);
    let options = OptimizeAcqfOptions {
        fixed_features: bad,
        ..seeded(0)
    };
    assert!(optimize_acqf(&acqf, &b, 1, 2, 16, &options).is_err());
}

#[test]
fn sequential_optimization_uses_pending_points() {
    let m = model();
    let best_f = m.train_targets().max();
    let pending0 = Matrix::from_row_slice(1, 3, &[0.9, 0.9, 0.1]);
    let acqf = QLogExpectedImprovement::new(
        m.clone(),
        best_f,
        QLogExpectedImprovementOptions {
            sampler: Some(Arc::new(SobolQMCNormalSampler::new(128, Some(2)).unwrap())),
            x_pending: Some(pending0.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let b = bounds();
    let options = OptimizeAcqfOptions {
        sequential: true,
        ..seeded(1)
    };
    let res = optimize_acqf_detailed(&acqf, &b, 3, 4, 128, &options).unwrap();
    assert_eq!(res.candidates.shape(), (3, 3));
    assert_eq!(res.acq_values.len(), 3);
    assert_eq!(res.acq_value, res.acq_values[2]);
    // The original pending points are restored.
    assert_eq!(acqf.x_pending(), Some(pending0.clone()));
    // Step i maximizes the acquisition conditioned on the base pending points and
    // the previous candidates: re-evaluating reproduces the reported value.
    let prev = Matrix::from_fn(3, 3, |i, j| {
        if i == 0 {
            pending0[(0, j)]
        } else {
            res.candidates[(i - 1, j)]
        }
    });
    acqf.set_x_pending(Some(&prev)).unwrap();
    let v = acqf
        .forward(&res.candidates.rows(2, 1).into_owned())
        .unwrap();
    assert_eq!(v, res.acq_values[2]);
    acqf.set_x_pending(Some(&pending0)).unwrap();
    // Greedy candidates are distinct.
    for i in 0..3 {
        for j in 0..i {
            assert!((res.candidates.row(i) - res.candidates.row(j)).amax() > 1e-3);
        }
    }

    // qLogNEI (incremental): pending points join the baseline.
    let nei = QLogNoisyExpectedImprovement::new(
        m.clone(),
        m.train_inputs(),
        QLogNoisyExpectedImprovementOptions {
            sampler: Some(Arc::new(SobolQMCNormalSampler::new(64, Some(5)).unwrap())),
            prune_options: botorch_core::acquisition::PruneOptions {
                seed: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let n_base = nei.x_baseline().nrows();
    let res = optimize_acqf_detailed(&nei, &b, 2, 3, 64, &options).unwrap();
    assert_eq!(res.candidates.shape(), (2, 3));
    assert_eq!(nei.x_pending(), None);
    assert!(nei.full_x_baseline().nrows() <= n_base);

    // Analytic acquisition functions do not support pending points.
    let logei = LogExpectedImprovement::new(m, best_f, true).unwrap();
    assert!(matches!(
        optimize_acqf(&logei, &b, 2, 2, 16, &options),
        Err(BotorchError::DimensionMismatch(_)) | Err(BotorchError::Unsupported(_))
    ));
}

#[test]
fn explicit_initial_conditions_and_errors() {
    let m = model();
    let best_f = m.train_targets().max();
    let acqf = LogExpectedImprovement::new(m, best_f, true).unwrap();
    let b = bounds();
    let ics = vec![
        Matrix::from_row_slice(1, 3, &[0.3, -0.2, 1.4]),
        Matrix::from_row_slice(1, 3, &[0.9, 0.9, 0.1]),
    ];
    let options = OptimizeAcqfOptions {
        batch_initial_conditions: Some(ics.clone()),
        ..seeded(0)
    };
    // Only the provided initial conditions are used (num_restarts = 2, no raw samples).
    let res = optimize_acqf_detailed(&acqf, &b, 1, 2, 0, &options).unwrap();
    assert_eq!(res.restarts.len(), 2);
    // More restarts than provided: the rest are generated from raw samples.
    let res = optimize_acqf_detailed(&acqf, &b, 1, 5, 64, &options).unwrap();
    assert_eq!(res.restarts.len(), 5);
    // Initial conditions outside the bounds are rejected.
    let options = OptimizeAcqfOptions {
        batch_initial_conditions: Some(vec![Matrix::from_row_slice(1, 3, &[5.0, 0.0, 0.0])]),
        ..seeded(0)
    };
    assert!(optimize_acqf(&acqf, &b, 1, 1, 0, &options).is_err());
    // Wrong q for an analytic acquisition function, wrong bounds.
    assert!(optimize_acqf(&acqf, &b, 2, 2, 16, &seeded(0)).is_err());
    assert!(optimize_acqf(&acqf, &Matrix::zeros(2, 2), 1, 2, 16, &seeded(0)).is_err());
    assert!(optimize_acqf(&acqf, &b, 1, 0, 16, &seeded(0)).is_err());
}

#[test]
fn zero_acquisition_values_fall_back_to_random_initial_conditions() {
    // qEI with an unreachable incumbent is exactly zero at every raw sample.
    let acqf = QExpectedImprovement::new(
        model(),
        1e6,
        McAcquisitionOptions {
            sampler: Some(Arc::new(SobolQMCNormalSampler::new(32, Some(0)).unwrap())),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(acqf.is_nonnegative());
    let res = optimize_acqf_detailed(&acqf, &bounds(), 1, 3, 16, &seeded(0)).unwrap();
    assert!(res
        .warnings
        .iter()
        .any(|w| w.contains("initial conditions are being selected randomly")));
    assert_eq!(res.restarts.len(), 3);
    assert_eq!(res.acq_value, 0.0);
}
