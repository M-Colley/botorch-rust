//! Checks of the multi-objective acquisition functions that do not need
//! BoTorch:
//!
//! * gradients of qEHVI, qLogEHVI, qNEHVI, qLogNEHVI (with and without
//!   `cache_root`, incremental or not, with pending points), the analytic
//!   EHVI and qLogNParEGO, with objectives and outcome constraints, against
//!   central finite differences;
//! * qEHVI and qNEHVI equal the Monte-Carlo average of exact hypervolume
//!   improvements computed with the dimension-sweep `Hypervolume` from the
//!   same posterior samples;
//! * the low-rank (`cache_root`) and joint samplings agree;
//! * analytic EHVI agrees with qEHVI as the number of MC samples grows;
//! * qLogEHVI approaches `log(qEHVI)` as the temperatures go to zero;
//! * pending points: cached decompositions equal fresh ones, and the
//!   non-incremental NEHVI of cached pending points.

use std::sync::Arc;

use botorch_core::acquisition::multi_objective::{
    ExpectedHypervolumeImprovement, FeasibilityWeightedMcMultiOutputObjective,
    GenericMcMultiOutputObjective, McMultiOutputObjective, QExpectedHypervolumeImprovement,
    QExpectedHypervolumeImprovementOptions, QLogExpectedHypervolumeImprovement,
    QLogExpectedHypervolumeImprovementOptions, QLogNParEGO, QLogNParEGOOptions,
    QLogNoisyExpectedHypervolumeImprovement, QLogNoisyExpectedHypervolumeImprovementOptions,
    QNoisyExpectedHypervolumeImprovement, QNoisyExpectedHypervolumeImprovementOptions,
    WeightedMcMultiOutputObjective,
};
use botorch_core::acquisition::AcquisitionFunction;
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::utils::multi_objective::box_decompositions::{
    BoxDecomposition, FastNondominatedPartitioning, NondominatedPartitioning,
};
use botorch_core::utils::multi_objective::hypervolume::Hypervolume;
use botorch_core::utils::objective::{
    FnOutcomeConstraint, LinearOutcomeConstraint, McConstraint, OutcomeConstraints,
};
use botorch_core::{
    BatchedMultiOutputGP, KernelKind, Matrix, Model, ModelListGP, Normalize, SingleTaskGP,
    SingleTaskGPOptions, Vector,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn data(seed: u64, n: usize, d: usize, m: usize) -> (Matrix, Matrix) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Matrix::from_fn(n, d, |_, _| rng.random::<f64>());
    let y = Matrix::from_fn(n, m, |i, t| {
        let u: f64 = (0..d)
            .map(|j| x[(i, j)] * (1.0 + j as f64 * t as f64))
            .sum();
        (1.0 + 0.3 * t as f64) * ((2.0 + t as f64) * u).sin() - 0.3 * u * u
            + 0.05 * rng.random::<f64>()
    });
    (x, y)
}

/// A multi-output SingleTaskGP with explicit hyperparameters.
fn multi_output(m: usize, seed: u64) -> (Arc<dyn Model>, Matrix, Matrix) {
    let (x, y) = data(seed, 10, 2, m);
    let mut gp = BatchedMultiOutputGP::multi_output(
        x.clone(),
        y.clone(),
        SingleTaskGPOptions {
            input_transform: Some(Normalize::new(2)),
            kernel: KernelKind::Matern52,
            ..Default::default()
        },
    )
    .unwrap();
    let mut raw = Vec::new();
    raw.extend((0..m).map(|t| 0.01 + 0.01 * t as f64));
    raw.extend((0..m).map(|t| 0.05 * t as f64 - 0.05));
    for t in 0..m {
        raw.extend([0.3 + 0.1 * t as f64, 0.5 - 0.05 * t as f64]);
    }
    gp.set_raw_parameters(&raw).unwrap();
    (Arc::new(gp), x, y)
}

/// A model list of a two-output and a single-output model.
fn model_list(seed: u64) -> (Arc<dyn Model>, Matrix) {
    let (x1, y1) = data(seed, 9, 2, 2);
    let mut a =
        BatchedMultiOutputGP::multi_output(x1.clone(), y1, SingleTaskGPOptions::default()).unwrap();
    a.set_raw_parameters(&[0.03, 0.02, 0.0, 0.1, 0.4, 0.3, 0.5, 0.6])
        .unwrap();
    let (x2, y2) = data(seed + 1, 7, 2, 1);
    let mut b = SingleTaskGP::new(x2, y2, SingleTaskGPOptions::default()).unwrap();
    b.set_raw_parameters(&[0.02, 0.35, 0.55, 0.4]).unwrap();
    (Arc::new(ModelListGP::new(vec![a, b.into()]).unwrap()), x1)
}

fn vstack(parts: &[&Matrix]) -> Matrix {
    let d = parts[0].ncols();
    let n: usize = parts.iter().map(|p| p.nrows()).sum();
    let mut out = Matrix::zeros(n, d);
    let mut r = 0;
    for p in parts {
        out.rows_mut(r, p.nrows()).copy_from(p);
        r += p.nrows();
    }
    out
}

fn points(q: usize, seed: u64) -> Matrix {
    let mut rng = StdRng::seed_from_u64(seed);
    Matrix::from_fn(q, 2, |_, _| 0.05 + 0.9 * rng.random::<f64>())
}

fn sampler(n: usize, seed: u64) -> Arc<SobolQMCNormalSampler> {
    Arc::new(SobolQMCNormalSampler::new(n, Some(seed)).unwrap())
}

/// A reference point below the posterior means at `x` and a partitioning of
/// the space not dominated by a weakened front (the means pulled 40% of the
/// way towards the reference point), so that new points improve on it.
fn front(model: &dyn Model, x: &Matrix) -> (Vec<f64>, FastNondominatedPartitioning) {
    let mean = model.posterior(x, false).unwrap().mean_matrix();
    let m = mean.ncols();
    let ref_point: Vec<f64> = (0..m)
        .map(|t| mean.column(t).min() - 0.1 * (mean.column(t).max() - mean.column(t).min()))
        .collect();
    let weak = Matrix::from_fn(mean.nrows(), m, |i, t| {
        ref_point[t] + 0.6 * (mean[(i, t)] - ref_point[t])
    });
    let p = FastNondominatedPartitioning::new(&ref_point, Some(&weak)).unwrap();
    (ref_point, p)
}

/// Central differences of the acquisition value against its gradient.
fn check_grad(acqf: &dyn AcquisitionFunction, x: &Matrix, label: &str) {
    let (v, g) = acqf.forward_with_grad(x).unwrap();
    assert!(
        (v - acqf.forward(x).unwrap()).abs() <= 1e-12 * v.abs().max(1.0),
        "{label}"
    );
    assert!(v.is_finite(), "{label}: {v}");
    let h = 1e-6;
    let scale = g.amax().max(1e-8);
    for i in 0..x.nrows() {
        for k in 0..x.ncols() {
            let mut xp = x.clone();
            xp[(i, k)] += h;
            let mut xm = x.clone();
            xm[(i, k)] -= h;
            let fd = (acqf.forward(&xp).unwrap() - acqf.forward(&xm).unwrap()) / (2.0 * h);
            assert!(
                (fd - g[(i, k)]).abs() <= 2e-5 * scale + 1e-7 * fd.abs(),
                "{label} ({i},{k}): grad {} vs fd {fd} (scale {scale})",
                g[(i, k)]
            );
        }
    }
}

fn constraint_on(t: usize, m: usize, offset: f64) -> Arc<dyn McConstraint> {
    let mut w = vec![0.0; m];
    w[t] = 1.0;
    Arc::new(LinearOutcomeConstraint::new(Vector::from_vec(w), offset).unwrap())
}

#[test]
fn gradients_match_finite_differences() {
    let (model, x, _) = multi_output(2, 1);
    let (ref_point, part) = front(model.as_ref(), &x);
    let pending = points(1, 99);
    let quad: Arc<dyn McConstraint> = Arc::new(FnOutcomeConstraint::new(|y| {
        (y[0] * y[1] - 0.3, vec![y[1], y[0]])
    }));
    for q in 1..=3 {
        let xq = points(q, 10 + q as u64);
        for (with_constraints, with_pending) in [(false, false), (true, false), (true, true)] {
            let constraints = with_constraints.then(|| {
                OutcomeConstraints::new(
                    vec![constraint_on(0, 2, -0.8), quad.clone()],
                    vec![0.1, 0.3],
                )
                .unwrap()
            });
            let x_pending = with_pending.then(|| pending.clone());
            let label = format!("q={q} c={with_constraints} p={with_pending}");
            let a = QExpectedHypervolumeImprovement::new(
                model.clone(),
                &ref_point,
                &part,
                QExpectedHypervolumeImprovementOptions {
                    sampler: Some(sampler(32, 3)),
                    constraints: constraints.clone(),
                    x_pending: x_pending.clone(),
                    ..Default::default()
                },
            )
            .unwrap();
            check_grad(&a, &xq, &format!("qEHVI {label}"));
            for fat in [true, false] {
                let a = QLogExpectedHypervolumeImprovement::new(
                    model.clone(),
                    &ref_point,
                    &part,
                    QLogExpectedHypervolumeImprovementOptions {
                        sampler: Some(sampler(32, 3)),
                        constraints: constraints.clone(),
                        x_pending: x_pending.clone(),
                        fat,
                        tau_relu: 1e-3,
                        ..Default::default()
                    },
                )
                .unwrap();
                check_grad(&a, &xq, &format!("qLogEHVI fat={fat} {label}"));
            }
            for cache_root in [true, false] {
                for incremental in [true, false] {
                    let a = QNoisyExpectedHypervolumeImprovement::new(
                        model.clone(),
                        &ref_point,
                        &x,
                        QNoisyExpectedHypervolumeImprovementOptions {
                            sampler: Some(sampler(16, 5)),
                            constraints: constraints.clone(),
                            x_pending: x_pending.clone(),
                            cache_root,
                            incremental_nehvi: incremental,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    check_grad(
                        &a,
                        &xq,
                        &format!("qNEHVI cr={cache_root} inc={incremental} {label}"),
                    );
                    let a = QLogNoisyExpectedHypervolumeImprovement::new(
                        model.clone(),
                        &ref_point,
                        &x,
                        QLogNoisyExpectedHypervolumeImprovementOptions {
                            sampler: Some(sampler(16, 5)),
                            constraints: constraints.clone(),
                            x_pending: x_pending.clone(),
                            cache_root,
                            incremental_nehvi: incremental,
                            tau_relu: 1e-3,
                            tau_max: 1e-2,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    check_grad(
                        &a,
                        &xq,
                        &format!("qLogNEHVI cr={cache_root} inc={incremental} {label}"),
                    );
                }
            }
        }
    }
    let a = ExpectedHypervolumeImprovement::new(model.clone(), &ref_point, &part).unwrap();
    for s in 0..4 {
        check_grad(&a, &points(1, 40 + s), "EHVI");
    }
}

#[test]
fn objectives_and_model_lists() {
    let (model, x) = model_list(3);
    let m = model.num_outputs();
    assert_eq!(m, 3);
    // Weighted objective on two outputs; the third output is a constraint.
    let weighted: Arc<dyn McMultiOutputObjective> = Arc::new(
        WeightedMcMultiOutputObjective::new(Vector::from_vec(vec![1.0, -0.5]), Some(&[0, 1]), None)
            .unwrap(),
    );
    let generic: Arc<dyn McMultiOutputObjective> =
        Arc::new(GenericMcMultiOutputObjective::new(|y| {
            (
                vec![y[0] + 0.1 * y[2], -(y[1] * y[1])],
                vec![1.0, 0.0, 0.1, 0.0, -2.0 * y[1], 0.0],
            )
        }));
    let feas: Arc<dyn McMultiOutputObjective> = Arc::new(
        FeasibilityWeightedMcMultiOutputObjective::new(model.as_ref(), &x, &[-1], None).unwrap(),
    );
    for (name, obj) in [
        ("weighted", weighted),
        ("generic", generic),
        ("feasibility", feas),
    ] {
        let samples_mean = model.posterior(&x, false).unwrap();
        let s = botorch_core::acquisition::McSamples::new(
            Matrix::from_column_slice(m * x.nrows(), 1, samples_mean.mean.as_slice()),
            x.nrows(),
            m,
        )
        .unwrap();
        let ym = obj.evaluate(&s, None, false).unwrap().values;
        let ymat = Matrix::from_fn(x.nrows(), 2, |p, t| ym.value(0, p, t));
        let ref_point: Vec<f64> = (0..2).map(|t| ymat.column(t).min() - 0.2).collect();
        let part = FastNondominatedPartitioning::new(&ref_point, Some(&ymat)).unwrap();
        let constraints =
            Some(OutcomeConstraints::uniform(vec![constraint_on(2, 3, -0.5)], 0.05).unwrap());
        let xq = points(2, 7);
        let a = QLogExpectedHypervolumeImprovement::new(
            model.clone(),
            &ref_point,
            &part,
            QLogExpectedHypervolumeImprovementOptions {
                sampler: Some(sampler(32, 1)),
                objective: Some(obj.clone()),
                constraints: constraints.clone(),
                tau_relu: 1e-3,
                ..Default::default()
            },
        )
        .unwrap();
        check_grad(&a, &xq, &format!("qLogEHVI list {name}"));
        for cache_root in [true, false] {
            let a = QNoisyExpectedHypervolumeImprovement::new(
                model.clone(),
                &ref_point,
                &x,
                QNoisyExpectedHypervolumeImprovementOptions {
                    sampler: Some(sampler(16, 2)),
                    objective: Some(obj.clone()),
                    constraints: constraints.clone(),
                    cache_root,
                    ..Default::default()
                },
            )
            .unwrap();
            check_grad(&a, &xq, &format!("qNEHVI list {name} cr={cache_root}"));
        }
        let p = QLogNParEGO::new(
            model.clone(),
            &x,
            QLogNParEGOOptions {
                objective: Some(obj.clone()),
                sampler: Some(sampler(16, 4)),
                weights_seed: Some(3),
                constraints: constraints.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p.scalarization_weights().len(), 2);
        assert!((p.scalarization_weights().iter().sum::<f64>() - 1.0).abs() < 1e-12);
        check_grad(&p, &xq, &format!("qLogNParEGO {name}"));
    }
}

/// The base samples BoTorch's NEHVI uses for `N` cached points and `q` new
/// points: the draw for all points with the cached points' rows replaced.
fn nehvi_base(s: &SobolQMCNormalSampler, n: usize, q: usize, m: usize) -> Matrix {
    let mut z = (*s.base_samples((n + q) * m).unwrap()).clone();
    let zb = s.base_samples(n * m).unwrap();
    z.columns_mut(0, n * m).copy_from(&zb);
    z
}

/// Hypervolume of the rows of `y` (all better than the reference point
/// counted, dominated points included).
fn hv(r: &[f64], y: &Matrix) -> f64 {
    Hypervolume::new(r).compute(y).unwrap()
}

#[test]
fn nehvi_is_the_average_exact_improvement() {
    for m in [2usize, 3] {
        let (model, x, _) = multi_output(m, 20 + m as u64);
        let (ref_point, _) = front(model.as_ref(), &x);
        let n = x.nrows();
        for q in 1..=2 {
            let xq = points(q, 30 + q as u64);
            let s = sampler(24, 9);
            let z = nehvi_base(&s, n, q, m);
            let post = model.posterior(&vstack(&[&x, &xq]), false).unwrap();
            let samples = post.rsample_from_base_samples(&z).unwrap();
            let mut expected = 0.0;
            for c in 0..z.nrows() {
                let yb = Matrix::from_fn(n, m, |p, t| samples.value(c, p, t));
                let ya = Matrix::from_fn(n + q, m, |p, t| samples.value(c, p, t));
                expected += hv(&ref_point, &ya) - hv(&ref_point, &yb);
            }
            expected /= z.nrows() as f64;
            for cache_root in [true, false] {
                let a = QNoisyExpectedHypervolumeImprovement::new(
                    model.clone(),
                    &ref_point,
                    &x,
                    QNoisyExpectedHypervolumeImprovementOptions {
                        sampler: Some(s.clone()),
                        cache_root,
                        ..Default::default()
                    },
                )
                .unwrap();
                let v = a.forward(&xq).unwrap();
                assert!(
                    (v - expected).abs() <= 1e-10 * expected.abs().max(1e-3),
                    "m={m} q={q} cache_root={cache_root}: {v} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn qehvi_is_the_average_exact_improvement_and_ehvi_its_limit() {
    let (model, x, _) = multi_output(2, 5);
    let (ref_point, part) = front(model.as_ref(), &x);
    let pareto = part.pareto_y().unwrap().matrix(0);
    let base_hv = hv(&ref_point, &pareto);
    let xq = points(2, 77);
    let s = sampler(64, 1);
    let a = QExpectedHypervolumeImprovement::new(
        model.clone(),
        &ref_point,
        &part,
        QExpectedHypervolumeImprovementOptions {
            sampler: Some(s.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let z = s.base_samples(2 * 2).unwrap();
    let samples = model
        .posterior(&xq, false)
        .unwrap()
        .rsample_from_base_samples(&z)
        .unwrap();
    let mut expected = 0.0;
    for c in 0..z.nrows() {
        let mut all = pareto.clone().insert_rows(pareto.nrows(), 2, 0.0);
        for p in 0..2 {
            for t in 0..2 {
                all[(pareto.nrows() + p, t)] = samples.value(c, p, t);
            }
        }
        expected += hv(&ref_point, &all) - base_hv;
    }
    expected /= z.nrows() as f64;
    let v = a.forward(&xq).unwrap();
    assert!(
        (v - expected).abs() <= 1e-12 * expected.max(1e-6),
        "{v} vs {expected}"
    );
    // Analytic EHVI is the limit of qEHVI (q = 1) for many samples.
    let ehvi = ExpectedHypervolumeImprovement::new(model.clone(), &ref_point, &part).unwrap();
    for k in 0..3 {
        let x1 = points(1, 200 + k);
        let exact = ehvi.forward(&x1).unwrap();
        let mc = QExpectedHypervolumeImprovement::new(
            model.clone(),
            &ref_point,
            &part,
            QExpectedHypervolumeImprovementOptions {
                sampler: Some(sampler(1 << 14, 7)),
                ..Default::default()
            },
        )
        .unwrap()
        .forward(&x1)
        .unwrap();
        assert!(
            (mc - exact).abs() <= 2e-3 * exact.max(1e-4),
            "EHVI {exact} vs qEHVI {mc}"
        );
    }
    // An approximate partitioning gives a lower bound of the same order.
    let approx = NondominatedPartitioning::new(&ref_point, Some(&pareto), 0.1).unwrap();
    assert!(approx.get_hypercell_bounds().unwrap().num_cells() >= 1);
}

#[test]
fn log_qehvi_approaches_log_qehvi() {
    let (model, x, _) = multi_output(3, 8);
    let (ref_point, part) = front(model.as_ref(), &x);
    let xq = points(2, 5);
    let exact = QExpectedHypervolumeImprovement::new(
        model.clone(),
        &ref_point,
        &part,
        QExpectedHypervolumeImprovementOptions {
            sampler: Some(sampler(32, 2)),
            ..Default::default()
        },
    )
    .unwrap()
    .forward(&xq)
    .unwrap()
    .ln();
    let mut prev = f64::INFINITY;
    for tau in [1e-2, 1e-3, 1e-4, 1e-5] {
        let v = QLogExpectedHypervolumeImprovement::new(
            model.clone(),
            &ref_point,
            &part,
            QLogExpectedHypervolumeImprovementOptions {
                sampler: Some(sampler(32, 2)),
                tau_relu: tau,
                tau_max: tau,
                ..Default::default()
            },
        )
        .unwrap()
        .forward(&xq)
        .unwrap();
        let err = (v - exact).abs();
        assert!(err <= prev, "tau={tau}: {err} > {prev}");
        prev = err;
    }
    assert!(prev < 1e-3, "{prev}");
}

#[test]
fn cache_root_matches_joint_sampling() {
    for (model, x) in [
        {
            let (m, x, _) = multi_output(2, 41);
            (m, x)
        },
        model_list(42),
    ] {
        let mo = model.num_outputs();
        let (ref_point, _) = front(model.as_ref(), &x);
        let ref_point = &ref_point[..mo.min(ref_point.len())];
        let s = sampler(32, 6);
        let make = |cache_root: bool| {
            QLogNoisyExpectedHypervolumeImprovement::new(
                model.clone(),
                ref_point,
                &x,
                QLogNoisyExpectedHypervolumeImprovementOptions {
                    sampler: Some(s.clone()),
                    cache_root,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let (a, b) = (make(true), make(false));
        for q in 1..=3 {
            let xq = points(q, 60 + q as u64);
            let (va, ga) = a.forward_with_grad(&xq).unwrap();
            let (vb, gb) = b.forward_with_grad(&xq).unwrap();
            assert!(
                (va - vb).abs() <= 1e-9 * va.abs().max(1.0),
                "q={q}: {va} vs {vb}"
            );
            assert!((&ga - &gb).amax() <= 1e-7 * ga.amax().max(1.0), "q={q}");
            let many = a.forward_many(&[xq.clone(), points(q, 1)]).unwrap();
            assert_eq!(many[0], va);
        }
    }
}

#[test]
fn pending_points_are_cached_like_fresh_decompositions() {
    let (model, x, _) = multi_output(2, 51);
    let (ref_point, _) = front(model.as_ref(), &x);
    let s = sampler(16, 3);
    let p1 = points(1, 1);
    let p2 = points(2, 2);
    let xq = points(1, 3);
    let opts =
        |incremental: bool, pending: Option<Matrix>| QNoisyExpectedHypervolumeImprovementOptions {
            sampler: Some(s.clone()),
            incremental_nehvi: incremental,
            x_pending: pending,
            ..Default::default()
        };
    for incremental in [true, false] {
        // Appending pending points one at a time vs at construction.
        let a = QNoisyExpectedHypervolumeImprovement::new(
            model.clone(),
            &ref_point,
            &x,
            opts(incremental, None),
        )
        .unwrap();
        a.set_x_pending(Some(&p1)).unwrap();
        let both = vstack(&[&p1, &p2]);
        a.set_x_pending(Some(&both)).unwrap();
        assert_eq!(a.x_baseline_and_pending().nrows(), x.nrows() + 3);
        let v_inc = a.forward(&xq).unwrap();
        assert!(v_inc.is_finite());
        // Replacing the pending points discards the cached ones.
        a.set_x_pending(Some(&p2)).unwrap();
        assert_eq!(a.x_baseline_and_pending().nrows(), x.nrows() + 2);
        let fresh = QNoisyExpectedHypervolumeImprovement::new(
            model.clone(),
            &ref_point,
            &x,
            opts(incremental, None),
        )
        .unwrap();
        fresh.set_x_pending(Some(&p2)).unwrap();
        let (va, vf) = (a.forward(&xq).unwrap(), fresh.forward(&xq).unwrap());
        assert!(
            (va - vf).abs() <= 1e-12 * va.abs().max(1e-9),
            "{va} vs {vf}"
        );
        if !incremental {
            assert!(a.prev_nehvi() > 0.0);
            assert!((a.prev_nehvi() - fresh.prev_nehvi()).abs() < 1e-15);
            let hvs = a.hypervolumes();
            let base = a.baseline_hvs().unwrap();
            let expected: f64 = hvs
                .iter()
                .zip(&base)
                .map(|(h, b)| (h - b).max(0.0))
                .sum::<f64>()
                / hvs.len() as f64;
            assert!((a.prev_nehvi() - expected).abs() < 1e-15);
        }
        // Clearing them restores the baseline-only decomposition.
        a.set_x_pending(None).unwrap();
        assert_eq!(a.x_baseline_and_pending().nrows(), x.nrows());
        assert_eq!(a.prev_nehvi(), 0.0);
    }
}
