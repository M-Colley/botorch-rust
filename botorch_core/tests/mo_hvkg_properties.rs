//! Checks of the fantasy models and the hypervolume knowledge gradient that
//! do not need BoTorch:
//!
//! * gradients of qHVKG (posterior mean and posterior samples, log, pending
//!   points and evaluation masks, current value and cost-aware utilities,
//!   multi-fidelity projections, objectives) against central finite
//!   differences;
//! * the fantasy posterior means average to the posterior mean (antithetic
//!   base samples: exactly), and conditioning equals a model refitted on the
//!   enlarged data with the same hyperparameters;
//! * the knowledge gradient is non-negative in expectation: with the fantasy
//!   designs at a fixed set `A` and `current_value` the value of `A`, the
//!   average over many quasi-random fantasies is `>= 0` up to the QMC error
//!   (the hypervolume is convex in every output's coordinates and the
//!   outputs are fantasized independently);
//! * t-batches are evaluated independently, and the evaluation does not
//!   depend on the number of threads;
//! * `optimize_acqf` optimizes augmented candidate sets and returns the
//!   extracted candidates.

use std::sync::Arc;

use botorch_core::acquisition::cost_aware::{
    CostAwareUtility, GenericCostAwareUtility, InverseCostWeightedUtility, UtilityValues,
};
use botorch_core::acquisition::fidelity::TargetFidelityProjection;
use botorch_core::acquisition::multi_objective::{
    GenericMcMultiOutputObjective, HypervolumeValueFunction, McMultiOutputObjective,
    QHypervolumeKnowledgeGradient, QHypervolumeKnowledgeGradientOptions,
    QMultiFidelityHypervolumeKnowledgeGradient, QMultiFidelityHypervolumeKnowledgeGradientOptions,
};
use botorch_core::acquisition::AcquisitionFunction;
use botorch_core::models::cost::AffineFidelityCostModel;
use botorch_core::models::{BatchedMatrix, BoolMatrix, FantasySampler};
use botorch_core::optim::{optimize_acqf_detailed, OptimizeAcqfOptions};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::{
    BatchedMultiOutputGP, Matrix, Model, ModelListGP, Normalize, SingleTaskGP, SingleTaskGPOptions,
    Vector,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn data(seed: u64, n: usize, d: usize, t: usize) -> (Matrix, Matrix) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Matrix::from_fn(n, d, |_, _| rng.random::<f64>());
    let y = Matrix::from_fn(n, 1, |i, _| {
        let u: f64 = (0..d).map(|j| x[(i, j)] * (1.0 + (j * t) as f64)).sum();
        (1.0 + 0.3 * t as f64) * ((2.0 + t as f64) * u).sin() - 0.3 * u * u
    });
    (x, y)
}

/// A list of `m` single-output GPs on different data (the second one with
/// fixed noise) with explicit hyperparameters.
fn model_list(m: usize, d: usize) -> ModelListGP {
    let subs: Vec<BatchedMultiOutputGP> = (0..m)
        .map(|t| {
            let (x, y) = data(10 + t as u64, 8 + t, d, t);
            let yvar = (t == 1).then(|| Matrix::from_fn(8 + t, 1, |i, _| 0.01 + 0.003 * i as f64));
            let mut gp = SingleTaskGP::new(
                x,
                y,
                SingleTaskGPOptions {
                    train_yvar: yvar,
                    input_transform: Some(Normalize::new(d)),
                    ..Default::default()
                },
            )
            .unwrap();
            let mut h = gp.hyperparameters().clone();
            h.raw_lengthscale = Vector::from_fn(d, |j, _| 0.3 + 0.15 * (j + t) as f64);
            h.raw_constant = 0.1 * t as f64;
            if h.raw_noise.is_some() {
                h.raw_noise = Some(0.01 + 0.005 * t as f64);
            }
            gp.set_hyperparameters(h).unwrap();
            gp.into()
        })
        .collect();
    ModelListGP::new(subs).unwrap()
}

fn samplers(m: usize, n_f: usize, seed: u64) -> Vec<Arc<SobolQMCNormalSampler>> {
    (0..m)
        .map(|t| Arc::new(SobolQMCNormalSampler::new(n_f, Some(seed + t as u64)).unwrap()))
        .collect()
}

fn points(seed: u64, n: usize, d: usize) -> Matrix {
    let mut rng = StdRng::seed_from_u64(seed);
    Matrix::from_fn(n, d, |_, _| 0.05 + 0.9 * rng.random::<f64>())
}

/// Central finite differences of `acqf` at `x` against its gradient.
fn check_gradient(acqf: &dyn AcquisitionFunction, x: &Matrix, what: &str) {
    let (v, g) = acqf.forward_with_grad(x).unwrap();
    assert!(v.is_finite(), "{what}: value {v}");
    assert!((acqf.forward(x).unwrap() - v).abs() <= 1e-14 * v.abs().max(1.0));
    let h = 1e-6;
    let scale = g.amax().max(1e-3);
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let mut a = x.clone();
            let mut b = x.clone();
            a[(i, j)] += h;
            b[(i, j)] -= h;
            let fd = (acqf.forward(&a).unwrap() - acqf.forward(&b).unwrap()) / (2.0 * h);
            assert!(
                (fd - g[(i, j)]).abs() <= 2e-6 * scale,
                "{what}: ({i}, {j}) analytic {} vs finite differences {fd}",
                g[(i, j)]
            );
        }
    }
}

fn hvkg(
    model: &Arc<dyn Model>,
    ref_point: &[f64],
    o: QHypervolumeKnowledgeGradientOptions,
) -> QHypervolumeKnowledgeGradient {
    QHypervolumeKnowledgeGradient::new(Arc::clone(model), ref_point, o).unwrap()
}

#[test]
fn hvkg_gradients_match_finite_differences() {
    let (d, m, n_f, np) = (2, 2, 3, 2);
    let model: Arc<dyn Model> = Arc::new(model_list(m, d));
    let ref_point = [-1.5, -1.0];
    let base = || QHypervolumeKnowledgeGradientOptions {
        num_fantasies: n_f,
        num_pareto: np,
        samplers: Some(samplers(m, n_f, 3)),
        inner_sampler: Some(Arc::new(SobolQMCNormalSampler::new(8, Some(9)).unwrap())),
        ..Default::default()
    };
    let x = points(1, 2 + n_f * np, d);
    check_gradient(&hvkg(&model, &ref_point, base()), &x, "posterior mean");
    check_gradient(
        &hvkg(
            &model,
            &ref_point,
            QHypervolumeKnowledgeGradientOptions {
                use_posterior_mean: false,
                ..base()
            },
        ),
        &x,
        "posterior samples",
    );
    check_gradient(
        &hvkg(
            &model,
            &ref_point,
            QHypervolumeKnowledgeGradientOptions {
                log: true,
                current_value: Some(-3.0),
                ..base()
            },
        ),
        &x,
        "log with current value",
    );
    // Pending points and decoupled evaluations.
    let pending = points(5, 2, d);
    check_gradient(
        &hvkg(
            &model,
            &ref_point,
            QHypervolumeKnowledgeGradientOptions {
                x_pending: Some(pending.clone()),
                x_evaluation_mask: Some(BoolMatrix::from_row_slice(
                    2,
                    2,
                    &[true, false, false, true],
                )),
                x_pending_evaluation_mask: Some(BoolMatrix::from_row_slice(
                    2,
                    2,
                    &[false, true, true, true],
                )),
                use_posterior_mean: false,
                ..base()
            },
        ),
        &x,
        "masks and pending points",
    );
    // A cost-aware utility of the candidates and an objective of X.
    let cost: Arc<dyn CostAwareUtility> = Arc::new(
        InverseCostWeightedUtility::new(
            Arc::new(AffineFidelityCostModel::new(Some(&[(1, 2.0)]), 0.5).unwrap()),
            true,
            None,
            false,
        )
        .unwrap(),
    );
    let objective: Arc<dyn McMultiOutputObjective> =
        Arc::new(GenericMcMultiOutputObjective::new(|y: &[f64]| {
            (
                vec![y[0] + 0.2 * y[1], y[1] - 0.1 * y[0] * y[0]],
                vec![1.0, 0.2, -0.2 * y[0], 1.0],
            )
        }));
    check_gradient(
        &hvkg(
            &model,
            &ref_point,
            QHypervolumeKnowledgeGradientOptions {
                current_value: Some(0.5),
                cost_aware_utility: Some(cost),
                objective: Some(objective),
                ..base()
            },
        ),
        &x,
        "cost-aware",
    );
    // Multi-fidelity: the value function at the target fidelity.
    let mf = QMultiFidelityHypervolumeKnowledgeGradient::new(
        Arc::clone(&model),
        &ref_point,
        vec![(-1, 1.0)],
        QMultiFidelityHypervolumeKnowledgeGradientOptions {
            hvkg: base(),
            project: Some(Arc::new(
                TargetFidelityProjection::new(vec![(-1, 1.0)], d).unwrap(),
            )),
        },
    )
    .unwrap();
    check_gradient(&mf, &x, "multi-fidelity");
    // q = 0: the value of the fantasy designs under the current model.
    let x0 = points(2, n_f * np, d);
    check_gradient(&hvkg(&model, &ref_point, base()), &x0, "q = 0");
}

#[test]
fn generic_cost_utility_and_value_function() {
    let (d, m, n_f, np) = (2, 2, 2, 2);
    let model: Arc<dyn Model> = Arc::new(model_list(m, d));
    let ref_point = [-1.5, -1.0];
    let utility: Arc<dyn CostAwareUtility> = Arc::new(GenericCostAwareUtility::new(
        |x: &Matrix, deltas: &[f64], with_grad: bool| {
            // deltas / (1 + Σ x²).
            let c = 1.0 + x.iter().map(|v| v * v).sum::<f64>();
            let values = deltas.iter().map(|d| d / c).collect();
            let deltas = deltas.to_vec();
            let x = x.clone();
            Ok(UtilityValues {
                values,
                pullback: with_grad.then(|| {
                    Box::new(move |vbar: &[f64]| {
                        let dbar = vbar.iter().map(|v| v / c).collect();
                        let cbar: f64 = vbar
                            .iter()
                            .zip(&deltas)
                            .map(|(v, d)| -v * d / (c * c))
                            .sum();
                        Ok((dbar, x.map(|v| 2.0 * v * cbar)))
                    }) as botorch_core::acquisition::cost_aware::UtilityPullback
                }),
            })
        },
    ));
    let acqf = hvkg(
        &model,
        &ref_point,
        QHypervolumeKnowledgeGradientOptions {
            num_fantasies: n_f,
            num_pareto: np,
            samplers: Some(samplers(m, n_f, 4)),
            current_value: Some(0.1),
            cost_aware_utility: Some(Arc::clone(&utility)),
            ..Default::default()
        },
    );
    check_gradient(&acqf, &points(3, 1 + n_f * np, d), "generic utility");
    // A log knowledge gradient requires log utilities.
    let log = hvkg(
        &model,
        &ref_point,
        QHypervolumeKnowledgeGradientOptions {
            num_fantasies: n_f,
            num_pareto: np,
            current_value: Some(0.1),
            cost_aware_utility: Some(utility),
            log: true,
            ..Default::default()
        },
    );
    assert!(log.forward(&points(3, 1 + n_f * np, d)).is_err());
    // The value function (non-negative, linear) and its gradient.
    let vf = HypervolumeValueFunction::new(
        Arc::clone(&model),
        &ref_point,
        None,
        None,
        true,
        false,
        None,
    )
    .unwrap();
    assert!(vf.is_nonnegative());
    check_gradient(&vf, &points(4, 3, d), "value function");
}

#[test]
fn fantasy_means_average_to_the_posterior_mean() {
    let d = 2;
    let list = model_list(2, d);
    let n_f = 4;
    let xn = points(7, 3, d);
    // Antithetic base samples: the fantasies' targets average to the mean.
    let z = Matrix::from_fn(n_f, 3, |i, j| {
        let v = 0.3 + 0.4 * j as f64 + 0.2 * (i / 2) as f64;
        if i % 2 == 0 {
            v
        } else {
            -v
        }
    });
    let s: Vec<Arc<SobolQMCNormalSampler>> = (0..2)
        .map(|_| Arc::new(SobolQMCNormalSampler::from_base_samples(z.clone(), Some(0)).unwrap()))
        .collect();
    let fm = list
        .fantasize(
            &BatchedMatrix::single(xn),
            FantasySampler::PerOutput(&s),
            None,
            None,
        )
        .unwrap();
    let xt = points(8, 5, d);
    let base = Model::posterior(&list, &xt, false).unwrap();
    let mut avg = Vector::zeros(base.mean.len());
    for i in 0..n_f {
        avg += fm.posterior_at(i, &xt, false).unwrap().mean;
    }
    avg /= n_f as f64;
    assert!((avg - &base.mean).amax() < 1e-12);
}

#[test]
fn knowledge_gradient_is_nonnegative_in_expectation() {
    let (d, m) = (2, 2);
    let model: Arc<dyn Model> = Arc::new(model_list(m, d));
    let ref_point = [-1.5, -1.0];
    let np = 2;
    let n_f = 256;
    let vf = HypervolumeValueFunction::new(
        Arc::clone(&model),
        &ref_point,
        None,
        None,
        true,
        false,
        None,
    )
    .unwrap();
    for seed in 0..3u64 {
        let a = points(20 + seed, np, d);
        let current = vf.forward(&a).unwrap();
        let acqf = hvkg(
            &model,
            &ref_point,
            QHypervolumeKnowledgeGradientOptions {
                num_fantasies: n_f,
                num_pareto: np,
                samplers: Some(samplers(m, n_f, 100 + seed)),
                current_value: Some(current),
                ..Default::default()
            },
        );
        let q = 1;
        let mut x = Matrix::zeros(q + n_f * np, d);
        x.rows_mut(0, q).copy_from(&points(30 + seed, q, d));
        for i in 0..n_f {
            x.rows_mut(q + i * np, np).copy_from(&a);
        }
        let kg = acqf.forward(&x).unwrap();
        assert!(
            kg >= -1e-3 * current.abs().max(1.0),
            "seed {seed}: {kg} (value {current})"
        );
    }
}

#[test]
fn t_batches_are_independent_and_thread_count_does_not_matter() {
    let (d, m, n_f, np) = (2, 2, 3, 2);
    let model: Arc<dyn Model> = Arc::new(model_list(m, d));
    let acqf = hvkg(
        &model,
        &[-1.5, -1.0],
        QHypervolumeKnowledgeGradientOptions {
            num_fantasies: n_f,
            num_pareto: np,
            samplers: Some(samplers(m, n_f, 5)),
            use_posterior_mean: false,
            ..Default::default()
        },
    );
    let xs: Vec<Matrix> = (0..4).map(|s| points(40 + s, 1 + n_f * np, d)).collect();
    let many = acqf.forward_many(&xs).unwrap();
    for (x, v) in xs.iter().zip(&many) {
        assert_eq!(acqf.forward(x).unwrap(), *v);
    }
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| acqf.forward_with_grad(&xs[0]).unwrap());
    let par = acqf.forward_with_grad(&xs[0]).unwrap();
    assert_eq!(one.0, par.0);
    assert_eq!(one.1, par.1);
}

#[test]
fn optimize_acqf_extracts_the_candidates() {
    let (d, m, n_f, np) = (2, 2, 2, 2);
    let model: Arc<dyn Model> = Arc::new(model_list(m, d));
    let acqf = hvkg(
        &model,
        &[-1.5, -1.0],
        QHypervolumeKnowledgeGradientOptions {
            num_fantasies: n_f,
            num_pareto: np,
            samplers: Some(samplers(m, n_f, 6)),
            ..Default::default()
        },
    );
    let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
    let mut options = OptimizeAcqfOptions {
        seed: Some(0),
        ..Default::default()
    };
    options.one_shot.num_inner_restarts = 4;
    options.one_shot.raw_inner_samples = 64;
    let r = optimize_acqf_detailed(&acqf, &bounds, 1, 3, 32, &options).unwrap();
    assert_eq!(r.candidates.shape(), (1, d));
    assert_eq!(r.restarts.len(), 3);
    for rs in &r.restarts {
        assert_eq!(rs.candidates.nrows(), 1 + n_f * np);
        assert!(rs.candidates.iter().all(|v| (0.0..=1.0).contains(v)));
    }
    let best = r
        .restarts
        .iter()
        .map(|rs| rs.acq_value)
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(r.acq_value, best);
    // Sequential optimization is not supported for one-shot functions.
    let seq = OptimizeAcqfOptions {
        sequential: true,
        ..options.clone()
    };
    assert!(optimize_acqf_detailed(&acqf, &bounds, 2, 2, 16, &seq).is_err());
}
