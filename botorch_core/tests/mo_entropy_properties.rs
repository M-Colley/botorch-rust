//! Properties of pathwise sampling and the multi-objective information-
//! theoretic acquisition functions that hold independently of BoTorch:
//!
//! * Matheron paths of multi-output models and model lists have the
//!   posterior's mean and covariance (statistically);
//! * conditioning on observations equals the posterior of the GP with the
//!   observations appended to its training data;
//! * sampled Pareto fronts are non-dominated, are the path values at the
//!   Pareto sets and are covered by the sample box decompositions;
//! * for one objective and one candidate, the information gains of the
//!   lower-bound estimates are non-negative (truncating a Gaussian to a
//!   half-line, and conditioning, reduce its variance; up to the jitter that
//!   `"LB"` adds to the variance), and the Monte-Carlo
//!   estimate of the information gain is at least the lower bound (the
//!   Gaussian has the largest entropy for given moments);
//! * gradients of MES, JES and PES agree with central finite differences;
//! * results do not depend on the number of threads.

use std::sync::Arc;

use botorch_core::acquisition::multi_objective::{
    compute_sample_box_decomposition, sample_optimal_points, ConditionedOutput, EstimationType,
    LowerBoundMultiObjectiveEntropySearchOptions, QLowerBoundMultiObjectiveJointEntropySearch,
    QLowerBoundMultiObjectiveMaxValueEntropySearch, QMultiObjectivePredictiveEntropySearch,
    QMultiObjectivePredictiveEntropySearchOptions, SampleOptimalPointsOptions, SamplePartitioning,
};
use botorch_core::acquisition::AcquisitionFunction;
use botorch_core::sampling::pathwise::{draw_matheron_paths, MatheronPathOptions, SamplePath};
use botorch_core::utils::multi_objective::pareto::is_non_dominated;
use botorch_core::{
    BatchedMultiOutputGP, KernelKind, Matrix, Model, ModelListGP, Normalize,
    OutcomeTransformChoice, SingleTaskGP, SingleTaskGPOptions, SobolQMCNormalSampler, Vector,
};

/// Training data in `[0, 1]^d` with `m` smooth outputs.
fn data(n: usize, d: usize, m: usize, shift: f64) -> (Matrix, Matrix) {
    let x = Matrix::from_fn(n, d, |i, j| {
        (((i * 7 + j * 3) % n) as f64 + 0.5) / n as f64 + 0.01 * (j as f64 + shift)
    });
    let y = Matrix::from_fn(n, m, |i, t| {
        let s: f64 = x
            .row(i)
            .iter()
            .enumerate()
            .map(|(j, v)| v * (1.0 + 0.3 * j as f64))
            .sum();
        (3.0 * s + t as f64).sin() + 0.5 * (t as f64 + 1.0) * (2.0 * s - shift).cos()
    });
    (x, y)
}

/// A multi-output model with explicit hyperparameters.
fn multi_output(
    n: usize,
    d: usize,
    m: usize,
    kind: KernelKind,
    normalize: bool,
) -> BatchedMultiOutputGP {
    let (x, y) = data(n, d, m, 0.0);
    let mut gp = BatchedMultiOutputGP::multi_output(
        x,
        y,
        SingleTaskGPOptions {
            kernel: kind,
            input_transform: normalize.then(|| Normalize::new(d)),
            ..Default::default()
        },
    )
    .unwrap();
    for e in 0..gp.num_elements() {
        let mut h = gp.element_hyperparameters(e).clone();
        h.raw_lengthscale = Vector::from_fn(d, |j, _| 0.3 + 0.1 * j as f64 + 0.05 * e as f64);
        h.raw_noise = Some(0.01 + 0.01 * e as f64);
        h.raw_constant = 0.1 * e as f64;
        gp.set_element_hyperparameters(e, h).unwrap();
    }
    gp
}

fn model_list() -> ModelListGP {
    let a = multi_output(9, 2, 1, KernelKind::Rbf, true);
    let (x, y) = data(7, 2, 1, 0.4);
    let mut b: BatchedMultiOutputGP = SingleTaskGP::new(
        x,
        y,
        SingleTaskGPOptions {
            kernel: KernelKind::Matern52,
            ..Default::default()
        },
    )
    .unwrap()
    .into();
    let mut h = b.element_hyperparameters(0).clone();
    h.raw_lengthscale = Vector::from_vec(vec![0.25, 0.4]);
    h.raw_noise = Some(0.02);
    b.set_element_hyperparameters(0, h).unwrap();
    ModelListGP::new(vec![a, b]).unwrap()
}

fn candidates(q: usize, d: usize, seed: usize) -> Matrix {
    Matrix::from_fn(q, d, |i, j| {
        let v = ((seed * 31 + i * 17 + j * 7) % 97) as f64 / 97.0;
        0.05 + 0.9 * v
    })
}

/// Central finite differences of `f` at `x`, compared with `grad` relative
/// to the largest component.
fn check_gradient(label: &str, f: &dyn Fn(&Matrix) -> f64, x: &Matrix, grad: &Matrix, tol: f64) {
    let h = 1e-6;
    let scale = grad.amax().max(1e-3);
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let mut xp = x.clone();
            xp[(i, j)] += h;
            let mut xm = x.clone();
            xm[(i, j)] -= h;
            let fd = (f(&xp) - f(&xm)) / (2.0 * h);
            let err = (fd - grad[(i, j)]).abs() / scale;
            assert!(
                err < tol,
                "{label}: d/dx[{i},{j}] analytic {} vs finite differences {fd} (error {err:e})",
                grad[(i, j)]
            );
        }
    }
}

#[test]
fn matheron_paths_have_the_posterior_moments() {
    let models: Vec<(&str, Arc<dyn Model>)> = vec![
        (
            "multi-output",
            Arc::new(multi_output(10, 2, 2, KernelKind::Rbf, true)),
        ),
        ("model list", Arc::new(model_list())),
    ];
    for (name, model) in models {
        let s = 2048;
        let options = MatheronPathOptions {
            num_features: 2048,
            seed: Some(3),
            draws: None,
        };
        let paths = draw_matheron_paths(model.as_ref(), s, &options).unwrap();
        let x = candidates(3, 2, 1);
        let values = paths.forward(&x).unwrap();
        let post = model.posterior(&x, false).unwrap();
        let m = model.num_outputs();
        assert_eq!(values.len(), m);
        for (t, v) in values.iter().enumerate() {
            let mean = v.row_mean().transpose();
            let centered = Matrix::from_fn(s, 3, |k, i| v[(k, i)] - mean[i]);
            let cov = centered.transpose() * &centered / (s as f64 - 1.0);
            let post_t = post.select_outputs(&[t]).unwrap();
            let scale = post_t.covariance.diagonal().max().sqrt();
            let mean_err = (&mean - &post_t.mean).amax() / scale;
            let cov_err = (&cov - &post_t.covariance).amax() / (scale * scale);
            assert!(mean_err < 0.08, "{name} output {t}: mean error {mean_err}");
            assert!(
                cov_err < 0.1,
                "{name} output {t}: covariance error {cov_err}"
            );
        }
    }
}

#[test]
fn conditioning_equals_the_posterior_with_appended_data() {
    // Without transforms the conditioned posterior is the posterior of the
    // GP trained on the appended data (with the same hyperparameters).
    let (x, y) = data(8, 2, 1, 0.2);
    let options = SingleTaskGPOptions {
        outcome_transform: OutcomeTransformChoice::None,
        ..Default::default()
    };
    let mut gp = SingleTaskGP::new(x.clone(), y.clone(), options.clone()).unwrap();
    let mut h = gp.hyperparameters().clone();
    h.raw_lengthscale = Vector::from_vec(vec![0.3, 0.45]);
    h.raw_noise = Some(0.03);
    gp.set_hyperparameters(h.clone()).unwrap();
    let z = Matrix::from_row_slice(2, 2, &[0.2, 0.7, 0.65, 0.35]);
    let yz = vec![1.3, -0.4];
    let noise = gp.observation_noise();
    let cond = ConditionedOutput::new(
        &gp,
        std::slice::from_ref(&z),
        std::slice::from_ref(&yz),
        noise,
    )
    .unwrap();
    let xa = Matrix::from_fn(10, 2, |i, j| if i < 8 { x[(i, j)] } else { z[(i - 8, j)] });
    let ya = Matrix::from_fn(10, 1, |i, _| if i < 8 { y[(i, 0)] } else { yz[i - 8] });
    let mut full = SingleTaskGP::new(xa, ya, options).unwrap();
    full.set_hyperparameters(h).unwrap();
    let xt = candidates(4, 2, 5);
    let (mean, var) = cond.marginals(&gp, &xt).unwrap();
    let (em, ev) = full.posterior_marginals_unclamped(&xt).unwrap();
    for i in 0..4 {
        assert!(
            (mean[(i, 0)] - em[i]).abs() < 1e-10,
            "{} vs {}",
            mean[(i, 0)],
            em[i]
        );
        assert!(
            (var[(i, 0)] - ev[i]).abs() < 1e-10,
            "{} vs {}",
            var[(i, 0)],
            ev[i]
        );
    }
    // The gradients of the conditioned marginals.
    let ma = Matrix::from_element(4, 1, 0.7);
    let va = Matrix::from_element(4, 1, -1.3);
    let g = cond.marginals_vjp(&gp, &xt, 4, &ma, &va).unwrap();
    let f = |xx: &Matrix| {
        let (m, v) = cond.marginals(&gp, xx).unwrap();
        0.7 * m.sum() - 1.3 * v.sum()
    };
    check_gradient("conditioned marginals", &f, &xt, &g, 1e-6);
}

#[test]
fn sampled_pareto_fronts_and_boxes() {
    let model = multi_output(10, 2, 2, KernelKind::Matern52, true);
    let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
    let options = SampleOptimalPointsOptions {
        seed: Some(11),
        num_features: 256,
        ..Default::default()
    };
    let (ps, pf) = sample_optimal_points(&model, &bounds, 4, 5, true, &options).unwrap();
    let cells =
        compute_sample_box_decomposition(&pf, SamplePartitioning::Dominated, true, 0).unwrap();
    for (k, (s, f)) in ps.iter().zip(&pf).enumerate() {
        assert!(
            is_non_dominated(f, true, true).iter().all(|&b| b),
            "sample {k}"
        );
        // Every Pareto point is the upper corner of a dominated cell (the
        // dominated region reaches the front).
        for i in 0..f.nrows() {
            let covered = (0..cells.num_cells()).any(|c| {
                let (lo, up) = cells.cell(k, c);
                (0..2).all(|t| lo[t] <= f[(i, t)] && f[(i, t)] <= up[t])
            });
            assert!(covered, "sample {k} point {i} is not in a dominated cell");
        }
        assert!(s.iter().all(|v| (0.0..=1.0).contains(v)));
    }
}

/// Entropy search acquisition functions of a model and sampled optima.
struct Setup {
    model: Arc<dyn Model>,
    ps: Vec<Matrix>,
    pf: Vec<Matrix>,
    cells: botorch_core::utils::multi_objective::box_decompositions::CellBounds,
}

fn setup(model: Arc<dyn Model>, num_points: usize, seed: u64) -> Setup {
    let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
    let options = SampleOptimalPointsOptions {
        seed: Some(seed),
        num_features: 256,
        ..Default::default()
    };
    let (ps, pf) =
        sample_optimal_points(model.as_ref(), &bounds, 3, num_points, true, &options).unwrap();
    let cells =
        compute_sample_box_decomposition(&pf, SamplePartitioning::Dominated, true, 0).unwrap();
    Setup {
        model,
        ps,
        pf,
        cells,
    }
}

fn entropy_options(
    et: EstimationType,
    pending: Option<Matrix>,
    samples: usize,
) -> LowerBoundMultiObjectiveEntropySearchOptions {
    LowerBoundMultiObjectiveEntropySearchOptions {
        x_pending: pending,
        estimation_type: et,
        sampler: Some(Arc::new(
            SobolQMCNormalSampler::new(samples, Some(5)).unwrap(),
        )),
    }
}

#[test]
fn single_objective_information_gains_are_non_negative() {
    let su = setup(
        Arc::new(multi_output(9, 2, 1, KernelKind::Rbf, false)),
        1,
        4,
    );
    for et in [
        EstimationType::Noiseless,
        EstimationType::LowerBound,
        EstimationType::LowerBound2,
    ] {
        let mes = QLowerBoundMultiObjectiveMaxValueEntropySearch::new(
            su.model.clone(),
            su.cells.clone(),
            entropy_options(et, None, 16),
        )
        .unwrap();
        let jes = QLowerBoundMultiObjectiveJointEntropySearch::new(
            su.model.clone(),
            su.ps.clone(),
            su.pf.clone(),
            su.cells.clone(),
            entropy_options(et, None, 16),
        )
        .unwrap();
        // "LB" adds 1e-6 to the moment-matched variance, which lowers the
        // gain by up to 0.5e-6 / (v + noise) where the truncation changes
        // little.
        let tol = if et == EstimationType::LowerBound {
            1e-5
        } else {
            1e-12
        };
        for k in 0..40 {
            let x = candidates(1, 2, k);
            let (vm, vj) = (mes.forward(&x).unwrap(), jes.forward(&x).unwrap());
            assert!(vm >= -tol, "MES {et:?} at {x}: {vm}");
            assert!(vj >= -tol, "JES {et:?} at {x}: {vj}");
        }
    }
}

#[test]
fn monte_carlo_information_gain_exceeds_the_lower_bound() {
    let su = setup(
        Arc::new(multi_output(10, 2, 2, KernelKind::Rbf, true)),
        4,
        8,
    );
    let lb = QLowerBoundMultiObjectiveMaxValueEntropySearch::new(
        su.model.clone(),
        su.cells.clone(),
        entropy_options(EstimationType::LowerBound, None, 16),
    )
    .unwrap();
    let mc = QLowerBoundMultiObjectiveMaxValueEntropySearch::new(
        su.model.clone(),
        su.cells.clone(),
        entropy_options(EstimationType::MonteCarlo, None, 8192),
    )
    .unwrap();
    for k in 0..10 {
        let x = candidates(1, 2, k);
        let (a, b) = (lb.forward(&x).unwrap(), mc.forward(&x).unwrap());
        assert!(b >= a - 2e-3 * a.abs().max(1.0), "at {x}: MC {b} < LB {a}");
    }
}

#[test]
fn entropy_search_gradients_match_finite_differences() {
    let models: Vec<(&str, Arc<dyn Model>, usize)> = vec![
        (
            "multi-output",
            Arc::new(multi_output(10, 2, 2, KernelKind::Matern52, true)),
            4,
        ),
        ("model list", Arc::new(model_list()), 3),
        (
            "single output",
            Arc::new(multi_output(8, 2, 1, KernelKind::Rbf, false)),
            1,
        ),
    ];
    for (name, model, num_points) in models {
        let su = setup(model, num_points, 21);
        for et in [
            EstimationType::Noiseless,
            EstimationType::LowerBound,
            EstimationType::LowerBound2,
            EstimationType::MonteCarlo,
        ] {
            for (q, pending) in [(1, None), (2, Some(candidates(1, 2, 40)))] {
                let mes = QLowerBoundMultiObjectiveMaxValueEntropySearch::new(
                    su.model.clone(),
                    su.cells.clone(),
                    entropy_options(et, pending.clone(), 16),
                )
                .unwrap();
                let jes = QLowerBoundMultiObjectiveJointEntropySearch::new(
                    su.model.clone(),
                    su.ps.clone(),
                    su.pf.clone(),
                    su.cells.clone(),
                    entropy_options(et, pending.clone(), 16),
                )
                .unwrap();
                let acqfs: [(&str, &dyn AcquisitionFunction); 2] = [("MES", &mes), ("JES", &jes)];
                for (acqf_name, acqf) in acqfs {
                    let x = candidates(q, 2, 7);
                    let (v, g) = acqf.forward_with_grad(&x).unwrap();
                    assert_eq!(v, acqf.forward(&x).unwrap());
                    let label = format!("{name} {acqf_name} {et:?} q={q}");
                    check_gradient(&label, &|xx| acqf.forward(xx).unwrap(), &x, &g, 1e-6);
                }
            }
        }
    }
}

#[test]
fn pes_gradients_match_finite_differences() {
    let models: Vec<(&str, Arc<dyn Model>, usize)> = vec![
        (
            "multi-output",
            Arc::new(multi_output(8, 2, 2, KernelKind::Rbf, false)),
            3,
        ),
        ("model list", Arc::new(model_list()), 3),
        (
            "single output",
            Arc::new(multi_output(8, 2, 1, KernelKind::Matern52, true)),
            1,
        ),
    ];
    for (name, model, num_points) in models {
        let su = setup(model, num_points, 33);
        for (maximize, q, pending) in [
            (true, 1, None),
            (true, 2, None),
            (false, 1, Some(candidates(1, 2, 50))),
        ] {
            let pes = QMultiObjectivePredictiveEntropySearch::new(
                su.model.clone(),
                su.ps.clone(),
                QMultiObjectivePredictiveEntropySearchOptions {
                    maximize,
                    x_pending: pending,
                    ..Default::default()
                },
            )
            .unwrap();
            let x = candidates(q, 2, 13);
            let (v, g) = pes.forward_with_grad(&x).unwrap();
            assert_eq!(v, pes.forward(&x).unwrap());
            let label = format!("{name} PES maximize={maximize} q={q}");
            check_gradient(&label, &|xx| pes.forward(xx).unwrap(), &x, &g, 1e-5);
        }
    }
}

#[test]
fn results_do_not_depend_on_the_number_of_threads() {
    let su = setup(
        Arc::new(multi_output(10, 2, 2, KernelKind::Rbf, true)),
        4,
        2,
    );
    let xs: Vec<Matrix> = (0..6).map(|k| candidates(2, 2, k)).collect();
    let build = || -> Vec<Box<dyn AcquisitionFunction>> {
        vec![
            Box::new(
                QLowerBoundMultiObjectiveJointEntropySearch::new(
                    su.model.clone(),
                    su.ps.clone(),
                    su.pf.clone(),
                    su.cells.clone(),
                    entropy_options(EstimationType::MonteCarlo, None, 32),
                )
                .unwrap(),
            ),
            Box::new(
                QMultiObjectivePredictiveEntropySearch::new(
                    su.model.clone(),
                    su.ps.clone(),
                    QMultiObjectivePredictiveEntropySearchOptions::default(),
                )
                .unwrap(),
            ),
        ]
    };
    let many: Vec<Vec<f64>> = build()
        .iter()
        .map(|a| a.forward_many(&xs).unwrap())
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let one: Vec<Vec<f64>> = pool.install(|| {
        build()
            .iter()
            .map(|a| a.forward_many(&xs).unwrap())
            .collect()
    });
    assert_eq!(many, one);
}
