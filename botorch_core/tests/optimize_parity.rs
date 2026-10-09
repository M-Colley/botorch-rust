//! Candidate generation compared with BoTorch on a fitted Hartmann-6 model
//! (`tests/fixtures/phase2/optimize_hartmann6.json`):
//!
//! * L-BFGS-B from BoTorch's own initial conditions (`gen_candidates_scipy`)
//!   for analytic LogEI and qLogEI (q = 2, explicit base samples): the final
//!   acquisition values must match (the port of L-BFGS-B reproduces scipy's
//!   iterates, so the candidates agree as well up to round-off amplification);
//! * full `optimize_acqf` runs with different seeds (raw samples cannot be
//!   reproduced, so the quality is compared statistically): the best Rust run
//!   must reach BoTorch's best value, and the fraction of runs that reach it
//!   must not be significantly lower than BoTorch's (for the multimodal
//!   q = 2 problem neither library finds the global optimum from every seed).
//!
//! Run with `--nocapture` to print the comparison tables.

mod common;

use std::sync::Arc;

use botorch_core::acquisition::{
    AcquisitionFunction, LogExpectedImprovement, QLogExpectedImprovement,
    QLogExpectedImprovementOptions,
};
use botorch_core::optim::{
    default_lbfgsb_options, gen_candidates_lbfgsb, optimize_acqf_detailed, OptimizeAcqfOptions,
};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::{Matrix, Model, SingleTaskGP, SingleTaskGPOptions};
use common::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Run {
    seed: u64,
    value: f64,
}

#[derive(Debug, Deserialize)]
struct AcqfRuns {
    base_samples: Option<Vec<Vec<f64>>>,
    initial_conditions: serde_json::Value,
    candidates: serde_json::Value,
    values: Vec<f64>,
    optimize_acqf: Vec<Run>,
}

#[derive(Debug, Deserialize)]
struct OptimizeFixture {
    train_x: Vec<Vec<f64>>,
    train_y: Vec<Vec<f64>>,
    raw_params: Vec<f64>,
    best_f: f64,
    logei: AcqfRuns,
    qlogei: AcqfRuns,
}

fn load() -> OptimizeFixture {
    let path = fixture_dir().join("phase2").join("optimize_hartmann6.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn model(f: &OptimizeFixture) -> Arc<dyn Model> {
    let mut m = SingleTaskGP::new(
        matrix(&f.train_x),
        matrix(&f.train_y),
        SingleTaskGPOptions::default(),
    )
    .unwrap();
    m.set_raw_parameters(&f.raw_params).unwrap();
    Arc::new(m)
}

/// `n × q × d` (or `n × d` for q = 1) JSON arrays as candidate sets.
fn sets(v: &serde_json::Value, q: usize) -> Vec<Matrix> {
    let arr = v.as_array().unwrap();
    arr.iter()
        .map(|e| {
            if q == 1 {
                let row: Vec<f64> = serde_json::from_value(e.clone()).unwrap();
                matrix(&[row])
            } else {
                let rows: Vec<Vec<f64>> = serde_json::from_value(e.clone()).unwrap();
                matrix(&rows)
            }
        })
        .collect()
}

fn unit_bounds(d: usize) -> Matrix {
    Matrix::from_fn(2, d, |i, _| i as f64)
}

/// Runs L-BFGS-B from BoTorch's initial conditions and compares the results.
fn check_gen_candidates(name: &str, acqf: &dyn AcquisitionFunction, runs: &AcqfRuns, q: usize) {
    let ics = sets(&runs.initial_conditions, q);
    let expected = sets(&runs.candidates, q);
    let bounds = unit_bounds(6);
    let res = gen_candidates_lbfgsb(
        &ics,
        acqf,
        &bounds,
        &Default::default(),
        &default_lbfgsb_options(),
    )
    .unwrap();
    println!("== {name}: L-BFGS-B from BoTorch's initial conditions ==");
    println!(
        "{:>3} {:>22} {:>22} {:>10} {:>10} {:>5}",
        "#", "botorch value", "rust value", "value err", "x err", "iters"
    );
    let mut worst = 0.0f64;
    for (i, r) in res.iter().enumerate() {
        let e = runs.values[i];
        let verr = (r.acq_value - e).abs() / e.abs().max(1.0);
        let xerr = (&r.candidates - &expected[i]).amax();
        worst = worst.max(verr);
        println!(
            "{i:>3} {e:>22.15e} {:>22.15e} {verr:>10.2e} {xerr:>10.2e} {:>5}",
            r.acq_value, r.iterations
        );
        // The same local optimum is reached, to optimization accuracy.
        assert!(
            verr < 1e-6,
            "{name} restart {i}: {} vs BoTorch {e}",
            r.acq_value
        );
        assert!(
            xerr < 1e-3,
            "{name} restart {i}: candidates differ by {xerr}"
        );
    }
    println!("worst value error: {worst:.2e}");
}

#[test]
fn gen_candidates_from_botorch_initial_conditions() {
    let f = load();
    let m = model(&f);
    let logei = LogExpectedImprovement::new(m.clone(), f.best_f, true).unwrap();
    check_gen_candidates("LogEI", &logei, &f.logei, 1);
    let z = matrix(f.qlogei.base_samples.as_ref().unwrap());
    let qlogei = QLogExpectedImprovement::new(
        m,
        f.best_f,
        QLogExpectedImprovementOptions {
            sampler: Some(Arc::new(
                SobolQMCNormalSampler::from_base_samples(z, Some(0)).unwrap(),
            )),
            ..Default::default()
        },
    )
    .unwrap();
    check_gen_candidates("qLogEI (q = 2)", &qlogei, &f.qlogei, 2);
}

fn check_optimize(name: &str, acqf: &dyn AcquisitionFunction, runs: &AcqfRuns, q: usize) {
    let bounds = unit_bounds(6);
    let botorch_best = runs
        .optimize_acqf
        .iter()
        .map(|r| r.value)
        .fold(f64::NEG_INFINITY, f64::max);
    println!("== {name}: optimize_acqf(q={q}, num_restarts=10, raw_samples=512) ==");
    println!("{:>5} {:>20} {:>20}", "seed", "botorch", "rust");
    let mut below = 0;
    let mut rust_best = f64::NEG_INFINITY;
    for run in &runs.optimize_acqf {
        let res = optimize_acqf_detailed(
            acqf,
            &bounds,
            q,
            10,
            512,
            &OptimizeAcqfOptions {
                seed: Some(run.seed),
                ..Default::default()
            },
        )
        .unwrap();
        println!(
            "{:>5} {:>20.12} {:>20.12}",
            run.seed, run.value, res.acq_value
        );
        assert_eq!(res.candidates.shape(), (q, 6));
        assert!(res.candidates.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert_eq!(res.restarts.len(), 10);
        // The returned value is the best restart and matches a re-evaluation.
        let best = res
            .restarts
            .iter()
            .map(|r| r.acq_value)
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(best, res.acq_value);
        assert_eq!(acqf.forward(&res.candidates).unwrap(), res.acq_value);
        rust_best = rust_best.max(res.acq_value);
        if res.acq_value < botorch_best - 1e-6 * botorch_best.abs().max(1.0) {
            below += 1;
        }
    }
    let n = runs.optimize_acqf.len();
    let tol = 1e-6 * botorch_best.abs().max(1.0);
    let botorch_hits = runs
        .optimize_acqf
        .iter()
        .filter(|r| r.value >= botorch_best - tol)
        .count();
    let rust_hits = n - below;
    println!(
        "BoTorch best {botorch_best:.12}, Rust best {rust_best:.12}; runs reaching it: BoTorch {botorch_hits}/{n}, Rust {rust_hits}/{n}"
    );
    assert!(
        rust_best >= botorch_best - tol,
        "best Rust value {rust_best} below BoTorch's {botorch_best}"
    );
    // Success rates: allow a binomial fluctuation of about two standard errors.
    let p = botorch_hits as f64 / n as f64;
    let slack = (2.0 * (n as f64 * p * (1.0 - p)).sqrt()).ceil() as usize + 1;
    assert!(
        rust_hits + slack >= botorch_hits,
        "Rust reached the optimum in {rust_hits}/{n} runs, BoTorch in {botorch_hits}/{n}"
    );
}

#[test]
fn optimize_acqf_matches_botorch_quality() {
    let f = load();
    let m = model(&f);
    let logei = LogExpectedImprovement::new(m.clone(), f.best_f, true).unwrap();
    check_optimize("LogEI", &logei, &f.logei, 1);
    let z = matrix(f.qlogei.base_samples.as_ref().unwrap());
    let qlogei = QLogExpectedImprovement::new(
        m,
        f.best_f,
        QLogExpectedImprovementOptions {
            sampler: Some(Arc::new(
                SobolQMCNormalSampler::from_base_samples(z, Some(0)).unwrap(),
            )),
            ..Default::default()
        },
    )
    .unwrap();
    check_optimize("qLogEI", &qlogei, &f.qlogei, 2);
}
