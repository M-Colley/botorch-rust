//! Central finite-difference checks of the analytic gradients of every
//! acquisition function (q = 1 and q > 1, with and without pending points,
//! with and without Normalize / Standardize), and consistency of values
//! between `forward`, `forward_with_grad` and `forward_many`.

use std::sync::Arc;

use botorch_core::acquisition::{
    AcquisitionFunction, ExpectedImprovement, LogExpectedImprovement, LogProbabilityOfImprovement,
    McAcquisitionOptions, PosteriorMean, PosteriorStandardDeviation, ProbabilityOfImprovement,
    PruneOptions, QExpectedImprovement, QLogExpectedImprovement, QLogExpectedImprovementOptions,
    QLogNoisyExpectedImprovement, QLogNoisyExpectedImprovementOptions, QProbabilityOfImprovement,
    QSimpleRegret, QUpperConfidenceBound, UpperConfidenceBound,
};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::{
    KernelKind, Matrix, Normalize, OutcomeTransformChoice, SingleTaskGP, SingleTaskGPOptions,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

type Named = (&'static str, Arc<SingleTaskGP>, f64, f64);

fn data(seed: u64, n: usize, d: usize, lo: f64, hi: f64) -> (Matrix, Matrix) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Matrix::from_fn(n, d, |_, _| lo + (hi - lo) * rng.random::<f64>());
    let y = Matrix::from_fn(n, 1, |i, _| {
        let u: f64 = (0..d).map(|j| (x[(i, j)] - lo) / (hi - lo)).sum();
        2.0 + 1.5 * ((3.0 * u).sin() + 0.3 * u * u) + 0.05 * rng.random::<f64>()
    });
    (x, y)
}

/// Models with fixed, non-default hyperparameters and their input ranges.
fn models() -> Vec<Named> {
    let mut out: Vec<Named> = Vec::new();
    let (x, y) = data(1, 10, 2, 0.0, 1.0);
    let mut m = SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap();
    m.set_raw_parameters(&[0.02, 0.1, 0.3, 0.45]).unwrap();
    out.push(("rbf + standardize", Arc::new(m), 0.0, 1.0));
    let (x, y) = data(2, 12, 3, -2.0, 3.0);
    let mut m = SingleTaskGP::new(
        x,
        y,
        SingleTaskGPOptions {
            input_transform: Some(Normalize::new(3)),
            kernel: KernelKind::Matern52,
            ..Default::default()
        },
    )
    .unwrap();
    m.set_raw_parameters(&[0.05, -0.2, 0.4, 0.6, 0.35]).unwrap();
    out.push(("matern + normalize + standardize", Arc::new(m), -2.0, 3.0));
    let (x, y) = data(3, 8, 2, 0.0, 1.0);
    let yvar = Matrix::from_fn(8, 1, |i, _| 0.01 + 0.003 * i as f64);
    let mut m = SingleTaskGP::new(
        x,
        y,
        SingleTaskGPOptions {
            train_yvar: Some(yvar),
            outcome_transform: OutcomeTransformChoice::None,
            ..Default::default()
        },
    )
    .unwrap();
    m.set_raw_parameters(&[1.5, 0.25, 0.5]).unwrap();
    out.push(("fixed noise, no outcome transform", Arc::new(m), 0.0, 1.0));
    out
}

fn points(q: usize, d: usize, lo: f64, hi: f64, seed: u64) -> Matrix {
    let mut rng = StdRng::seed_from_u64(seed);
    Matrix::from_fn(q, d, |_, _| lo + (hi - lo) * rng.random::<f64>())
}

/// Checks `forward_with_grad` against central differences of `forward`.
fn check(name: &str, acqf: &dyn AcquisitionFunction, x: &Matrix) {
    let (v, g) = acqf.forward_with_grad(x).unwrap();
    let v2 = acqf.forward(x).unwrap();
    assert!(
        (v - v2).abs() <= 1e-12 * v.abs().max(1e-12),
        "{name}: forward {v2} vs forward_with_grad {v}"
    );
    assert_eq!(g.shape(), x.shape());
    let scale = g.amax().max(1e-8);
    for i in 0..x.nrows() {
        for k in 0..x.ncols() {
            let h = 1e-6 * x[(i, k)].abs().max(1.0);
            let mut xp = x.clone();
            xp[(i, k)] += h;
            let mut xm = x.clone();
            xm[(i, k)] -= h;
            let fd = (acqf.forward(&xp).unwrap() - acqf.forward(&xm).unwrap()) / (2.0 * h);
            assert!(
                (g[(i, k)] - fd).abs() <= 1e-5 * scale + 1e-6 * fd.abs(),
                "{name}: d/dx[{i},{k}] analytic {} vs fd {fd}",
                g[(i, k)]
            );
        }
    }
}

type Acqfs = Vec<(&'static str, Box<dyn AcquisitionFunction>)>;

#[test]
fn analytic_gradients_match_finite_differences() {
    for (mname, model, lo, hi) in models() {
        let d = model.dim();
        let best = model.train_targets().max();
        for maximize in [true, false] {
            let bf = if maximize {
                best
            } else {
                model.train_targets().min()
            };
            let m = || model.clone();
            let acqfs: Acqfs = vec![
                (
                    "EI",
                    Box::new(ExpectedImprovement::new(m(), bf, maximize).unwrap()),
                ),
                (
                    "LogEI",
                    Box::new(LogExpectedImprovement::new(m(), bf, maximize).unwrap()),
                ),
                (
                    "LogEI far",
                    Box::new(LogExpectedImprovement::new(m(), bf + 25.0, maximize).unwrap()),
                ),
                (
                    "PI",
                    Box::new(ProbabilityOfImprovement::new(m(), bf, maximize).unwrap()),
                ),
                (
                    "LogPI",
                    Box::new(LogProbabilityOfImprovement::new(m(), bf + 1.0, maximize).unwrap()),
                ),
                (
                    "UCB",
                    Box::new(UpperConfidenceBound::new(m(), 2.0, maximize).unwrap()),
                ),
                ("PM", Box::new(PosteriorMean::new(m(), maximize).unwrap())),
                (
                    "PSD",
                    Box::new(PosteriorStandardDeviation::new(m(), maximize).unwrap()),
                ),
            ];
            for (aname, acqf) in &acqfs {
                assert_eq!(acqf.expected_q(), Some(1));
                let xs: Vec<Matrix> = (0..4).map(|s| points(1, d, lo, hi, 10 + s)).collect();
                for x in &xs {
                    check(
                        &format!("{mname} / {aname} / max={maximize}"),
                        acqf.as_ref(),
                        x,
                    );
                }
                let many = acqf.forward_many(&xs).unwrap();
                for (x, v) in xs.iter().zip(&many) {
                    assert_eq!(*v, acqf.forward(x).unwrap(), "{aname}: forward_many");
                }
                assert!(acqf.forward(&points(2, d, lo, hi, 1)).is_err());
                assert!(acqf.set_x_pending(None).is_err());
            }
        }
    }
}

fn sampler(seed: u64) -> Option<Arc<SobolQMCNormalSampler>> {
    Some(Arc::new(
        SobolQMCNormalSampler::new(256, Some(seed)).unwrap(),
    ))
}

#[test]
fn mc_gradients_match_finite_differences() {
    for (mname, model, lo, hi) in models() {
        let d = model.dim();
        let best = model.train_targets().max();
        for n_pending in [0usize, 2] {
            let pending = (n_pending > 0).then(|| points(n_pending, d, lo, hi, 99));
            let opts = || McAcquisitionOptions {
                sampler: sampler(7),
                x_pending: pending.clone(),
                ..Default::default()
            };
            let logei = |fat: bool, bf: f64| {
                QLogExpectedImprovement::new(
                    model.clone(),
                    bf,
                    QLogExpectedImprovementOptions {
                        sampler: sampler(7),
                        x_pending: pending.clone(),
                        fat,
                        ..Default::default()
                    },
                )
                .unwrap()
            };
            let m = || model.clone();
            let acqfs: Acqfs = vec![
                (
                    "qEI",
                    Box::new(QExpectedImprovement::new(m(), best - 0.3, opts()).unwrap()),
                ),
                ("qLogEI", Box::new(logei(true, best))),
                ("qLogEI far", Box::new(logei(true, best + 20.0))),
                ("qLogEI fat=false", Box::new(logei(false, best))),
                (
                    "qPI",
                    Box::new(QProbabilityOfImprovement::new(m(), best, 0.05, opts()).unwrap()),
                ),
                (
                    "qUCB",
                    Box::new(QUpperConfidenceBound::new(m(), 0.5, opts()).unwrap()),
                ),
                ("qSR", Box::new(QSimpleRegret::new(m(), opts()).unwrap())),
            ];
            for incremental in [true, false] {
                let nei = QLogNoisyExpectedImprovement::new(
                    model.clone(),
                    model.train_inputs(),
                    QLogNoisyExpectedImprovementOptions {
                        sampler: sampler(3),
                        x_pending: pending.clone(),
                        incremental,
                        prune_options: PruneOptions {
                            seed: Some(0),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .unwrap();
                for q in 1..=3 {
                    let x = points(q, d, lo, hi, 40 + q as u64);
                    let name = format!(
                        "{mname} / qLogNEI inc={incremental} / q={q} / pending={n_pending}"
                    );
                    check(&name, &nei, &x);
                }
            }
            for (aname, acqf) in &acqfs {
                assert_eq!(acqf.x_pending(), pending);
                for q in 1..=3 {
                    let x = points(q, d, lo, hi, 20 + q as u64);
                    let name = format!("{mname} / {aname} / q={q} / pending={n_pending}");
                    check(&name, acqf.as_ref(), &x);
                }
            }
        }
    }
}
