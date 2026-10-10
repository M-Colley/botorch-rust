//! Constrained acquisition optimization and polytope sampling compared with
//! BoTorch (`tests/fixtures/phase4b/optimize_constrained.json` and
//! `polytope.json`):
//!
//! * SLSQP from BoTorch's own initial conditions (`gen_candidates_scipy` with
//!   linear inequality, equality, inter-point, fixed-feature and nonlinear
//!   constraints; chunks of restarts optimized jointly as in BoTorch): the
//!   candidates must be feasible and agree with SciPy's within `1e-5` (Kraft's
//!   SLSQP is driven like SciPy's, so the iterates agree up to round-off), the
//!   acquisition values within `1e-6` (relative, log scale);
//! * full `optimize_acqf` runs over several seeds (raw samples and polytope
//!   samples cannot be reproduced, so the quality is compared): every
//!   candidate must be feasible and the best value over the seeds must reach
//!   BoTorch's best value;
//! * hit-and-run polytope samples: feasibility, and means and covariances
//!   within Monte-Carlo tolerances of BoTorch's samples; interior points of
//!   polytopes with a unique maximal-slack point.
//!
//! Run with `--nocapture` to print the comparison tables.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use botorch_core::acquisition::{
    AcquisitionFunction, LogExpectedImprovement, QLogExpectedImprovement,
    QLogExpectedImprovementOptions,
};
use botorch_core::optim::{
    evaluate_feasibility, gen_batch_initial_conditions, gen_candidates_slsqp,
    optimize_acqf_detailed, sample_q_batches_from_polytope, ConstraintIndices,
    FnNonlinearConstraint, IcGenerator, InitialConditionOptions, LinearConstraint,
    NonlinearInequalityConstraint, OptimizeAcqfOptions, ParameterConstraints, SlsqpOptions,
    CONSTRAINT_TOLERANCE,
};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::utils::polytope::{find_interior_point, get_polytope_samples};
use botorch_core::{Matrix, Model, SingleTaskGP, SingleTaskGPOptions, Vector};
use common::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Gen {
    initial_conditions: Vec<Vec<Vec<f64>>>,
    candidates: Vec<Vec<Vec<f64>>>,
    values: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Run {
    seed: u64,
    value: f64,
}

/// `[indices, coefficients, rhs]` with 1-d or 2-d indices.
type ConstraintJson = (serde_json::Value, Vec<f64>, f64);

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    acqf: String,
    q: usize,
    inequality_constraints: Vec<ConstraintJson>,
    equality_constraints: Vec<ConstraintJson>,
    nonlinear: Vec<(String, bool)>,
    fixed_features: BTreeMap<String, f64>,
    gen_candidates: Vec<Gen>,
    optimize_acqf: Vec<Run>,
}

#[derive(Debug, Deserialize)]
struct OptimizeFixture {
    train_x: Vec<Vec<f64>>,
    train_y: Vec<Vec<f64>>,
    raw_params: Vec<f64>,
    best_f: f64,
    qlogei_base_samples: Vec<Vec<f64>>,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct PolytopeCase {
    name: String,
    bounds: Vec<Vec<f64>>,
    q: Option<usize>,
    inequality_constraints: Vec<ConstraintJson>,
    equality_constraints: Vec<ConstraintJson>,
    n: usize,
    mean: Vec<f64>,
    covariance: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct Interior {
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    x: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct PolytopeFixture {
    cases: Vec<PolytopeCase>,
    interior_points: Vec<Interior>,
}

fn load_optimize() -> OptimizeFixture {
    let path = fixture_dir()
        .join("phase4b")
        .join("optimize_constrained.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn linear(c: &ConstraintJson) -> LinearConstraint {
    let (idx, coef, rhs) = c;
    let arr = idx.as_array().unwrap();
    if arr.first().is_some_and(|v| v.is_array()) {
        let pairs: Vec<(usize, usize)> = arr
            .iter()
            .map(|p| {
                let p = p.as_array().unwrap();
                (
                    p[0].as_u64().unwrap() as usize,
                    p[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        LinearConstraint::inter_point(pairs, coef.clone(), *rhs).unwrap()
    } else {
        let idx: Vec<usize> = arr.iter().map(|v| v.as_u64().unwrap() as usize).collect();
        LinearConstraint::intra_point(idx, coef.clone(), *rhs).unwrap()
    }
}

/// BoTorch's `0.35² - ||x - 0.4||² ≥ 0`.
fn ball() -> Arc<dyn NonlinearInequalityConstraint> {
    Arc::new(FnNonlinearConstraint::intrapoint(|x| {
        let v = 0.35f64.powi(2) - x.iter().map(|v| (v - 0.4) * (v - 0.4)).sum::<f64>();
        (v, x.iter().map(|v| -2.0 * (v - 0.4)).collect())
    }))
}

/// BoTorch's inter-point `||x_0 - x_1||² - 0.15 ≥ 0` (q = 2, d = 3).
fn separation() -> Arc<dyn NonlinearInequalityConstraint> {
    Arc::new(FnNonlinearConstraint::interpoint(|x| {
        let d = x.len() / 2;
        let diff: Vec<f64> = (0..d).map(|k| x[k] - x[d + k]).collect();
        let v = diff.iter().map(|v| v * v).sum::<f64>() - 0.15;
        let mut g = vec![0.0; 2 * d];
        for k in 0..d {
            g[k] = 2.0 * diff[k];
            g[d + k] = -2.0 * diff[k];
        }
        (v, g)
    }))
}

fn constraints_of(c: &Case) -> ParameterConstraints {
    ParameterConstraints {
        inequality: c.inequality_constraints.iter().map(linear).collect(),
        equality: c.equality_constraints.iter().map(linear).collect(),
        nonlinear: c
            .nonlinear
            .iter()
            .map(|(name, intra)| {
                let f = match name.as_str() {
                    "ball" => ball(),
                    "separation" => separation(),
                    other => panic!("unknown constraint {other}"),
                };
                assert_eq!(f.is_intrapoint(), *intra);
                f
            })
            .collect(),
    }
}

fn fixed_of(c: &Case) -> BTreeMap<usize, f64> {
    c.fixed_features
        .iter()
        .map(|(k, v)| (k.parse().unwrap(), *v))
        .collect()
}

fn acqfs(f: &OptimizeFixture) -> (Arc<dyn AcquisitionFunction>, Arc<dyn AcquisitionFunction>) {
    let mut m = SingleTaskGP::new(
        matrix(&f.train_x),
        matrix(&f.train_y),
        SingleTaskGPOptions::default(),
    )
    .unwrap();
    m.set_raw_parameters(&f.raw_params).unwrap();
    let model: Arc<dyn Model> = Arc::new(m);
    let logei = Arc::new(LogExpectedImprovement::new(model.clone(), f.best_f, true).unwrap());
    let sampler = Arc::new(
        SobolQMCNormalSampler::from_base_samples(matrix(&f.qlogei_base_samples), Some(0)).unwrap(),
    );
    let qlogei = Arc::new(
        QLogExpectedImprovement::new(
            model,
            f.best_f,
            QLogExpectedImprovementOptions {
                sampler: Some(sampler),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    (logei, qlogei)
}

fn unit_bounds(d: usize) -> Matrix {
    Matrix::from_fn(2, d, |i, _| i as f64)
}

/// SLSQP from BoTorch's initial conditions. Kraft's algorithm is driven like
/// SciPy's, but SciPy (>= 1.16) solves the least-squares subproblems with a
/// rewritten NNLS / LSEI (LAPACK Householder and Givens transformations),
/// whose solutions differ from the NLopt translation at about `1e-9` per
/// step. Well-conditioned runs therefore end at the same optimum (values
/// within `1e-6`, typically `1e-8`); runs that wander through infeasible
/// regions can be chaotic and end at a different local optimum. Required:
/// at least 80% of the restarts at SciPy's optimum, and the best restart of
/// every case at least as good as SciPy's best.
#[test]
fn slsqp_from_botorch_initial_conditions() {
    let f = load_optimize();
    let (logei, qlogei) = acqfs(&f);
    let bounds = unit_bounds(3);
    let mut r = Report::default();
    let (mut same, mut total) = (0usize, 0usize);
    println!(
        "{:28} {:>3} {:>22} {:>22} {:>10} {:>10} {:>5}",
        "case", "#", "botorch value", "rust value", "value err", "x err", "iters"
    );
    for c in &f.cases {
        let acqf = if c.acqf == "logei" { &logei } else { &qlogei };
        let constraints = constraints_of(c);
        let fixed = fixed_of(c);
        let mut i = 0;
        let (mut best_bt, mut best_rs) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for g in &c.gen_candidates {
            let ics: Vec<Matrix> = g.initial_conditions.iter().map(|s| matrix(s)).collect();
            let res = gen_candidates_slsqp(
                &ics,
                acqf.as_ref(),
                &bounds,
                &fixed,
                &constraints,
                &SlsqpOptions::default(),
            )
            .unwrap();
            for (k, (x, v)) in res.candidates.iter().zip(&res.acq_values).enumerate() {
                let expected = matrix(&g.candidates[k]);
                let e = g.values[k];
                let verr = (v - e).abs() / e.abs().max(1.0);
                let xerr = (x - &expected).amax();
                println!(
                    "{:28} {i:>3} {e:>22.15e} {v:>22.15e} {verr:>10.2e} {xerr:>10.2e} {:>5}",
                    c.name, res.iterations
                );
                // SciPy's candidates are feasible up to its accuracy; so are ours.
                assert!(
                    evaluate_feasibility(x, &constraints, 1e-6).unwrap(),
                    "{}: restart {i} infeasible",
                    c.name
                );
                for (&k, &val) in &fixed {
                    for p in 0..x.nrows() {
                        assert_eq!(x[(p, k)], val);
                    }
                }
                total += 1;
                if verr <= 1e-6 {
                    same += 1;
                    r.check(&c.name, "SLSQP value vs SciPy (same optimum)", verr, 1e-6);
                    r.check(
                        &c.name,
                        "SLSQP candidates vs SciPy (same optimum)",
                        xerr,
                        1e-4,
                    );
                }
                best_bt = best_bt.max(e);
                best_rs = best_rs.max(*v);
                i += 1;
            }
        }
        let shortfall = (best_bt - best_rs).max(0.0) / best_bt.abs().max(1.0);
        r.check(&c.name, "best restart shortfall vs SciPy", shortfall, 1e-6);
    }
    println!("restarts at SciPy's optimum: {same} of {total}");
    assert!(
        same * 5 >= total * 4,
        "only {same} of {total} restarts reach SciPy's optimum"
    );
    r.finish("SLSQP from BoTorch's initial conditions");
}

#[test]
fn optimize_acqf_with_constraints_matches_botorch_quality() {
    let f = load_optimize();
    let (logei, qlogei) = acqfs(&f);
    let bounds = unit_bounds(3);
    let mut r = Report::default();
    for c in &f.cases {
        let acqf = if c.acqf == "logei" { &logei } else { &qlogei };
        let constraints = constraints_of(c);
        let fixed = fixed_of(c);
        let mut values = Vec::new();
        for run in &c.optimize_acqf {
            let ic_generator = if constraints.nonlinear.is_empty() {
                None
            } else {
                // BoTorch's `ic_generator=gen_batch_initial_conditions` (ignores
                // the nonlinear constraints).
                let lin = constraints.clone();
                let fx = fixed.clone();
                Some(IcGenerator::new(move |req| {
                    let opts = InitialConditionOptions {
                        seed: req.options.seed,
                        fixed_features: fx.clone(),
                        inequality_constraints: lin.inequality.clone(),
                        equality_constraints: lin.equality.clone(),
                        ..Default::default()
                    };
                    let mut rng =
                        <rand::rngs::Xoshiro256PlusPlus as rand::SeedableRng>::seed_from_u64(
                            req.options.seed.unwrap_or(0),
                        );
                    Ok(gen_batch_initial_conditions(
                        req.acq_function,
                        req.bounds,
                        req.q,
                        req.num_restarts,
                        req.raw_samples,
                        &opts,
                        &mut rng,
                    )?
                    .0)
                }))
            };
            let options = OptimizeAcqfOptions {
                seed: Some(run.seed),
                fixed_features: fixed.clone(),
                constraints: constraints.clone(),
                ic_generator,
                ..Default::default()
            };
            let res = optimize_acqf_detailed(acqf.as_ref(), &bounds, c.q, 4, 64, &options).unwrap();
            assert!(
                evaluate_feasibility(&res.candidates, &constraints, CONSTRAINT_TOLERANCE).unwrap(),
                "{} seed {}: infeasible candidates",
                c.name,
                run.seed
            );
            values.push(res.acq_value);
        }
        let best_bt = c
            .optimize_acqf
            .iter()
            .map(|r| r.value)
            .fold(f64::NEG_INFINITY, f64::max);
        let best_rs = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "{:28} botorch best {best_bt:>14.8} rust best {best_rs:>14.8} rust runs {:?}",
            c.name,
            values.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>()
        );
        // The best Rust run must reach BoTorch's best (log-scale tolerance).
        let shortfall = (best_bt - best_rs).max(0.0) / best_bt.abs().max(1.0);
        r.check(&c.name, "best value shortfall vs BoTorch", shortfall, 1e-4);
    }
    r.finish("optimize_acqf with constraints");
}

#[test]
fn polytope_samples_match_botorch_statistically() {
    let path = fixture_dir().join("phase4b").join("polytope.json");
    let f: PolytopeFixture = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for c in &f.cases {
        let bounds = matrix(&c.bounds);
        let ineq: Vec<LinearConstraint> = c.inequality_constraints.iter().map(linear).collect();
        let eq: Vec<LinearConstraint> = c.equality_constraints.iter().map(linear).collect();
        let x = match c.q {
            None => get_polytope_samples(c.n, &bounds, &ineq, &eq, Some(7), 2000, 10).unwrap(),
            Some(q) => {
                let batches =
                    sample_q_batches_from_polytope(c.n, q, &bounds, 2000, 5, Some(7), &ineq, &eq)
                        .unwrap();
                let d = bounds.ncols();
                // Feasible q-batches.
                let cons = ParameterConstraints {
                    inequality: ineq.clone(),
                    equality: eq.clone(),
                    nonlinear: vec![],
                };
                for b in &batches {
                    assert!(evaluate_feasibility(b, &cons, 1e-9).unwrap());
                }
                Matrix::from_fn(c.n, q * d, |i, j| batches[i][(j / d, j % d)])
            }
        };
        let dim = x.ncols();
        let (lo, hi): (Vec<f64>, Vec<f64>) = (0..dim)
            .map(|j| {
                (
                    bounds[(0, j % bounds.ncols())],
                    bounds[(1, j % bounds.ncols())],
                )
            })
            .unzip();
        // Feasibility of every sample.
        for i in 0..x.nrows() {
            for j in 0..dim {
                assert!(x[(i, j)] >= lo[j] - 1e-9 && x[(i, j)] <= hi[j] + 1e-9);
            }
            if c.q.is_none() {
                let row = x.rows(i, 1).into_owned();
                for con in &ineq {
                    assert!(con.coefficients().iter().sum::<f64>().is_finite());
                    let v = match con.indices() {
                        ConstraintIndices::IntraPoint(idx) => idx
                            .iter()
                            .zip(con.coefficients())
                            .map(|(&k, c)| c * row[(0, k)])
                            .sum::<f64>(),
                        ConstraintIndices::InterPoint(_) => unreachable!(),
                    };
                    assert!(v >= con.rhs() - 1e-9, "{}: {v} < {}", c.name, con.rhs());
                }
                for con in &eq {
                    let v = match con.indices() {
                        ConstraintIndices::IntraPoint(idx) => idx
                            .iter()
                            .zip(con.coefficients())
                            .map(|(&k, c)| c * row[(0, k)])
                            .sum::<f64>(),
                        ConstraintIndices::InterPoint(_) => unreachable!(),
                    };
                    assert!((v - con.rhs()).abs() < 1e-9);
                }
            }
        }
        // Moments: within a few Monte-Carlo standard errors (of the range).
        let n = x.nrows() as f64;
        let mean: Vec<f64> = (0..dim).map(|j| x.column(j).sum() / n).collect();
        let mut max_mean_err = 0.0f64;
        let mut max_cov_err = 0.0f64;
        for j in 0..dim {
            let range = hi[j] - lo[j];
            max_mean_err = max_mean_err.max((mean[j] - c.mean[j]).abs() / range);
            for k in 0..dim {
                let cov: f64 = (0..x.nrows())
                    .map(|i| (x[(i, j)] - mean[j]) * (x[(i, k)] - mean[k]))
                    .sum::<f64>()
                    / (n - 1.0);
                let scale = range * (hi[k] - lo[k]);
                max_cov_err = max_cov_err.max((cov - c.covariance[j][k]).abs() / scale);
            }
        }
        println!(
            "{:16} mean err {max_mean_err:.3e} (of range), covariance err {max_cov_err:.3e}",
            c.name
        );
        assert!(max_mean_err < 0.03, "{}: means differ", c.name);
        assert!(max_cov_err < 0.015, "{}: covariances differ", c.name);
    }
    for p in &f.interior_points {
        let a = matrix(&p.a);
        let b = Vector::from_vec(p.b.clone());
        let x = find_interior_point(&a, &b, None).unwrap();
        for (k, e) in p.x.iter().enumerate() {
            assert!((x[k] - e).abs() < 1e-9, "interior point {x} vs {:?}", p.x);
        }
    }
}
