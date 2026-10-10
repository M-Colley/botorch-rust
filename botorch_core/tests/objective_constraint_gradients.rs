//! Central finite-difference checks of the input gradients of MC acquisition
//! functions with multi-output sampling, MC objectives (including a direct
//! dependence on `X`) and outcome constraints, of qLogNEI with feasibility-
//! weighted incumbents and of qLogProbabilityOfFeasibility; consistency of
//! `forward`, `forward_many` and `forward_with_grad`; and BoTorch's argument
//! validation.

use std::sync::Arc;

use botorch_core::acquisition::{
    AcquisitionFunction, ConstrainedMcObjective, GenericMcObjective, IdentityMcObjective,
    LinearMcObjective, McAcquisitionOptions, McObjective, McSamples, QExpectedImprovement,
    QLogExpectedImprovement, QLogExpectedImprovementOptions, QLogNoisyExpectedImprovement,
    QLogNoisyExpectedImprovementOptions, QLogProbabilityOfFeasibility,
    QLogProbabilityOfFeasibilityOptions, QProbabilityOfImprovement, QSimpleRegret,
    QUpperConfidenceBound, SamplePullback, SampleValues,
};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::utils::objective::{
    FnOutcomeConstraint, LinearOutcomeConstraint, McConstraint, OutcomeConstraints,
};
use botorch_core::{
    BatchedMultiOutputGP, BotorchError, KernelKind, Matrix, Model, ModelListGP, Normalize, Result,
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

fn multi_output() -> Arc<dyn Model> {
    let (x, y) = data(11, 12, 2, 2);
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
    gp.set_raw_parameters(&[0.02, 0.05, 0.1, -0.2, 0.3, 0.5, 0.25, 0.6])
        .unwrap();
    Arc::new(gp)
}

fn model_list() -> Arc<dyn Model> {
    let (x1, y1) = data(12, 9, 2, 2);
    let mut a = BatchedMultiOutputGP::multi_output(x1, y1, SingleTaskGPOptions::default()).unwrap();
    a.set_raw_parameters(&[0.03, 0.02, 0.0, 0.1, 0.4, 0.3, 0.5, 0.6])
        .unwrap();
    let (x2, y2) = data(13, 7, 2, 1);
    let mut b = SingleTaskGP::new(x2, y2, SingleTaskGPOptions::default()).unwrap();
    b.set_raw_parameters(&[0.02, 0.35, 0.55, 0.4]).unwrap();
    Arc::new(ModelListGP::new(vec![a, b.into()]).unwrap())
}

fn single_output() -> Arc<dyn Model> {
    let (x, y) = data(14, 10, 2, 1);
    let mut gp = SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap();
    gp.set_raw_parameters(&[0.02, 0.1, 0.3, 0.45]).unwrap();
    Arc::new(gp)
}

fn points(q: usize, seed: u64) -> Matrix {
    let mut rng = StdRng::seed_from_u64(seed);
    Matrix::from_fn(q, 2, |_, _| 0.05 + 0.9 * rng.random::<f64>())
}

fn sampler(seed: u64) -> Option<Arc<SobolQMCNormalSampler>> {
    Some(Arc::new(
        SobolQMCNormalSampler::new(64, Some(seed)).unwrap(),
    ))
}

/// Central differences of `forward` against `forward_with_grad`.
fn check_fd(name: &str, acqf: &dyn AcquisitionFunction, x: &Matrix) {
    let (v, g) = acqf.forward_with_grad(x).unwrap();
    assert_eq!(acqf.forward(x).unwrap(), v, "{name}: forward");
    assert_eq!(
        acqf.forward_many(std::slice::from_ref(x)).unwrap()[0],
        v,
        "{name}: forward_many"
    );
    assert!(v.is_finite(), "{name}: value {v}");
    let scale = g.amax().max(1e-3);
    for p in 0..x.nrows() {
        for k in 0..x.ncols() {
            let h = 1e-6;
            let mut xp = x.clone();
            xp[(p, k)] += h;
            let mut xm = x.clone();
            xm[(p, k)] -= h;
            let fd = (acqf.forward(&xp).unwrap() - acqf.forward(&xm).unwrap()) / (2.0 * h);
            assert!(
                (fd - g[(p, k)]).abs() < 2e-5 * scale + 1e-7,
                "{name}: d/dx[{p},{k}] = {} vs finite difference {fd}",
                g[(p, k)]
            );
        }
    }
}

/// A quadratic objective `y0 - 0.3 y1²` (generic, pointwise).
fn quadratic() -> Arc<dyn McObjective> {
    Arc::new(GenericMcObjective::new(|y| {
        let v = y[0] - 0.3 * y[1] * y[1];
        let mut g = vec![0.0; y.len()];
        g[0] = 1.0;
        g[1] = -0.6 * y[1];
        (v, g)
    }))
}

/// An objective that depends on `X` directly: `y0 · (1 + x_{p,0})` (BoTorch's
/// `GenericMCObjective(lambda Y, X: Y[..., 0] * (1 + X[..., 0]))`).
#[derive(Debug)]
struct XWeighted;

impl McObjective for XWeighted {
    fn evaluate(
        &self,
        samples: &McSamples,
        x: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<SampleValues> {
        let x = x.expect("X is passed").clone();
        let (n, s) = (samples.num_points(), samples.num_samples());
        let y0 = samples.output(0);
        let values = Matrix::from_fn(n, s, |p, c| y0[(p, c)] * (1.0 + x[(p, 0)]));
        let pullback: Option<SamplePullback> = if with_grad {
            let m = samples.num_outputs();
            Some(Box::new(move |bar: &Matrix| {
                let mut sbar = Matrix::zeros(m * n, bar.ncols());
                let mut xbar = Matrix::zeros(n, x.ncols());
                for p in 0..n {
                    for c in 0..bar.ncols() {
                        sbar[(p, c)] = bar[(p, c)] * (1.0 + x[(p, 0)]);
                        xbar[(p, 0)] += bar[(p, c)] * y0[(p, c)];
                    }
                }
                Ok((sbar, Some(xbar)))
            }))
        } else {
            None
        };
        Ok(SampleValues { values, pullback })
    }
}

/// `y_last <= 0.3` and `y0² <= 1` for `m` outputs.
fn constraints_m(m: usize) -> OutcomeConstraints {
    let mut w = vec![0.0; m];
    w[m - 1] = 1.0;
    let c0: Arc<dyn McConstraint> =
        Arc::new(LinearOutcomeConstraint::new(Vector::from_vec(w), -0.3).unwrap());
    let c1: Arc<dyn McConstraint> = Arc::new(FnOutcomeConstraint::new(|y| {
        let mut g = vec![0.0; y.len()];
        g[0] = 2.0 * y[0];
        (y[0] * y[0] - 1.0, g)
    }));
    OutcomeConstraints::new(vec![c0, c1], vec![0.2, 0.5]).unwrap()
}

fn constraints() -> OutcomeConstraints {
    constraints_m(2)
}

#[test]
fn mc_acquisition_gradients_with_objectives_and_constraints() {
    for (mname, model) in [
        ("multi-output", multi_output()),
        ("model list", model_list()),
    ] {
        let m = model.num_outputs();
        let w: Vec<f64> = [0.8, -0.5, 0.3][..m].to_vec();
        let linear: Arc<dyn McObjective> =
            Arc::new(LinearMcObjective::new(Vector::from_vec(w)).unwrap());
        let objectives: Vec<(&str, Arc<dyn McObjective>)> = vec![
            ("linear", linear.clone()),
            ("generic", quadratic()),
            ("x-weighted", Arc::new(XWeighted)),
            (
                "constrained",
                Arc::new(
                    ConstrainedMcObjective::new(
                        quadratic(),
                        constraints_m(m).constraints().to_vec(),
                        1.5,
                        vec![0.2, 0.5],
                    )
                    .unwrap(),
                ),
            ),
        ];
        for (oname, obj) in &objectives {
            for (q, pending) in [(1, None), (2, Some(points(2, 40)))] {
                let x = points(q, 7 + q as u64);
                let opts = || McAcquisitionOptions {
                    sampler: sampler(3),
                    x_pending: pending.clone(),
                    objective: Some(obj.clone()),
                    constraints: None,
                };
                let name = |a: &str| format!("{mname} {oname} {a} q={q}");
                check_fd(
                    &name("qEI"),
                    &QExpectedImprovement::new(model.clone(), -0.5, opts()).unwrap(),
                    &x,
                );
                check_fd(
                    &name("qPI"),
                    &QProbabilityOfImprovement::new(model.clone(), 0.0, 0.2, opts()).unwrap(),
                    &x,
                );
                check_fd(
                    &name("qSR"),
                    &QSimpleRegret::new(model.clone(), opts()).unwrap(),
                    &x,
                );
                check_fd(
                    &name("qUCB"),
                    &QUpperConfidenceBound::new(model.clone(), 0.3, opts()).unwrap(),
                    &x,
                );
                if !obj.is_constrained() {
                    check_fd(
                        &name("qLogEI"),
                        &QLogExpectedImprovement::new(
                            model.clone(),
                            0.0,
                            QLogExpectedImprovementOptions {
                                sampler: sampler(3),
                                x_pending: pending.clone(),
                                objective: Some(obj.clone()),
                                tau_max: 0.05,
                                tau_relu: 1e-2,
                                ..Default::default()
                            },
                        )
                        .unwrap(),
                        &x,
                    );
                }
            }
        }
        // Outcome constraints (sigmoid for qEI/qPI, fatmoid / sigmoid for
        // qLogEI, a probability constraint with fat = None).
        for (q, pending) in [(1, None), (2, Some(points(1, 41)))] {
            let x = points(q, 17 + q as u64);
            let name = |a: &str| format!("{mname} constrained {a} q={q}");
            let opts = || McAcquisitionOptions {
                sampler: sampler(5),
                x_pending: pending.clone(),
                objective: Some(linear.clone()),
                constraints: Some(constraints_m(m)),
            };
            check_fd(
                &name("qEI"),
                &QExpectedImprovement::new(model.clone(), -0.5, opts()).unwrap(),
                &x,
            );
            check_fd(
                &name("qPI"),
                &QProbabilityOfImprovement::new(model.clone(), 0.0, 0.2, opts()).unwrap(),
                &x,
            );
            for fat in [true, false] {
                check_fd(
                    &name(&format!("qLogEI fat={fat}")),
                    &QLogExpectedImprovement::new(
                        model.clone(),
                        0.0,
                        QLogExpectedImprovementOptions {
                            sampler: sampler(5),
                            x_pending: pending.clone(),
                            fat,
                            tau_max: 0.05,
                            tau_relu: 1e-2,
                            objective: Some(linear.clone()),
                            constraints: Some(constraints_m(m)),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                    &x,
                );
            }
            let prob: Arc<dyn McConstraint> = Arc::new(FnOutcomeConstraint::new(|y| {
                let s = 1.0 / (1.0 + (-2.0 * y[0]).exp());
                let mut g = vec![0.0; y.len()];
                g[0] = 2.0 * s * (1.0 - s);
                (s, g)
            }));
            let mixed = OutcomeConstraints::new(
                vec![constraints_m(m).constraints()[0].clone(), prob],
                vec![0.2, 0.2],
            )
            .unwrap();
            check_fd(
                &name("qLogEI fat=[true, None]"),
                &QLogExpectedImprovement::new(
                    model.clone(),
                    0.0,
                    QLogExpectedImprovementOptions {
                        sampler: sampler(5),
                        x_pending: pending.clone(),
                        objective: Some(linear.clone()),
                        constraints: Some(mixed),
                        constraint_fat: Some(vec![Some(true), None]),
                        tau_relu: 1e-2,
                        ..Default::default()
                    },
                )
                .unwrap(),
                &x,
            );
            check_fd(
                &name("qLogPF"),
                &QLogProbabilityOfFeasibility::new(
                    model.clone(),
                    constraints_m(m),
                    QLogProbabilityOfFeasibilityOptions {
                        sampler: sampler(5),
                        x_pending: pending.clone(),
                        tau_max: 0.05,
                        ..Default::default()
                    },
                )
                .unwrap(),
                &x,
            );
        }
    }
}

#[test]
fn qlognei_gradients_with_objectives_and_constraints() {
    for (mname, model) in [
        ("multi-output", multi_output()),
        ("model list", model_list()),
    ] {
        let m = model.num_outputs();
        let w: Vec<f64> = [0.8, -0.5, 0.3][..m].to_vec();
        let xb = points(6, 50);
        for (k, (cons, incremental, pending)) in [
            (Some(constraints_m(m)), true, None),
            (Some(constraints_m(m)), true, Some(points(2, 51))),
            (Some(constraints_m(m)), false, Some(points(2, 52))),
            (None, true, Some(points(1, 53))),
        ]
        .into_iter()
        .enumerate()
        {
            let objective: Arc<dyn McObjective> = if k == 3 {
                Arc::new(XWeighted)
            } else {
                Arc::new(LinearMcObjective::new(Vector::from_vec(w.clone())).unwrap())
            };
            let acqf = QLogNoisyExpectedImprovement::new(
                model.clone(),
                &xb,
                QLogNoisyExpectedImprovementOptions {
                    sampler: sampler(9),
                    x_pending: pending,
                    prune_baseline: false,
                    incremental,
                    tau_relu: 1e-2,
                    tau_max: 0.05,
                    objective: Some(objective),
                    constraints: cons,
                    ..Default::default()
                },
            )
            .unwrap();
            for q in 1..=2 {
                check_fd(
                    &format!("{mname} qLogNEI case {k} q={q}"),
                    &acqf,
                    &points(q, 60 + q as u64),
                );
            }
        }
        // A baseline that is infeasible in every sample: the incumbents are
        // BoTorch's lower bound of the objective.
        let never: Arc<dyn McConstraint> =
            Arc::new(LinearOutcomeConstraint::new(Vector::from_vec(vec![0.0; m]), 1.0).unwrap());
        let acqf = QLogNoisyExpectedImprovement::new(
            model.clone(),
            &xb,
            QLogNoisyExpectedImprovementOptions {
                sampler: sampler(9),
                prune_baseline: true,
                prune_options: botorch_core::acquisition::PruneOptions {
                    seed: Some(0),
                    ..Default::default()
                },
                tau_relu: 1e-2,
                objective: Some(Arc::new(
                    LinearMcObjective::new(Vector::from_vec(w.clone())).unwrap(),
                )),
                constraints: Some(OutcomeConstraints::uniform(vec![never], 0.1).unwrap()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(acqf.lower_bound_points().shape(), (32, 2));
        check_fd(
            &format!("{mname} qLogNEI infeasible"),
            &acqf,
            &points(2, 70),
        );
    }
}

#[test]
fn botorch_argument_validation() {
    let mo = multi_output();
    let so = single_output();
    let none = McAcquisitionOptions::default;
    // Multi-output models need an objective (or a posterior transform).
    let e = QExpectedImprovement::new(mo.clone(), 0.0, none()).unwrap_err();
    assert!(e.to_string().contains("Must specify an objective"), "{e}");
    // An identity objective on a multi-output model fails when evaluated.
    let acqf = QExpectedImprovement::new(
        mo.clone(),
        0.0,
        McAcquisitionOptions {
            objective: Some(Arc::new(IdentityMcObjective)),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        acqf.forward(&points(2, 1)),
        Err(BotorchError::Runtime(_))
    ));
    // A linear objective with the wrong number of weights.
    let acqf = QSimpleRegret::new(
        mo.clone(),
        McAcquisitionOptions {
            objective: Some(Arc::new(
                LinearMcObjective::new(Vector::from_vec(vec![1.0, 2.0, 3.0])).unwrap(),
            )),
            ..Default::default()
        },
    )
    .unwrap();
    let e = acqf.forward(&points(1, 2)).unwrap_err();
    assert!(e.to_string().contains("Output shape of samples"), "{e}");
    // ConstrainedMcObjective and constraints together; in a LogEI function.
    let con_obj: Arc<dyn McObjective> = Arc::new(
        ConstrainedMcObjective::new(
            Arc::new(IdentityMcObjective),
            constraints().constraints().to_vec(),
            0.0,
            vec![1e-3, 1e-3],
        )
        .unwrap(),
    );
    let e = QExpectedImprovement::new(
        so.clone(),
        0.0,
        McAcquisitionOptions {
            objective: Some(con_obj.clone()),
            constraints: Some(constraints()),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(e
        .to_string()
        .contains("ConstrainedMCObjective as well as constraints"));
    let e = QLogExpectedImprovement::new(
        so.clone(),
        0.0,
        QLogExpectedImprovementOptions {
            objective: Some(con_obj),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("Log-Improvement should not be used"));
    // qSimpleRegret / qUCB do not take outcome constraints.
    let with_constraints = McAcquisitionOptions {
        constraints: Some(OutcomeConstraints::uniform(vec![], 1e-3).unwrap()),
        ..Default::default()
    };
    assert!(QSimpleRegret::new(so.clone(), with_constraints.clone()).is_err());
    assert!(QUpperConfidenceBound::new(so.clone(), 0.1, with_constraints).is_err());
    // Mismatched fat lists.
    let e = QLogExpectedImprovement::new(
        so,
        0.0,
        QLogExpectedImprovementOptions {
            constraints: Some(constraints()),
            constraint_fat: Some(vec![Some(true)]),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("fats do not match"));
}
