//! Parity of the multi-objective test problems with BoTorch
//! (`tests/fixtures/phase5a_tf/`, `scripts/fixtures_mo_test_functions.py`):
//! objectives, negated objectives and constraint slacks at random points and
//! at the corners of the domain, bounds, reference points, maximum
//! hypervolumes and the deterministic Pareto fronts; for problems whose
//! definition botorch-rust corrects (BoTorch pull requests #3402 and #3405),
//! agreement with the corrected definition and a check that BoTorch main
//! indeed differs. Also the transformations behind the random DTLZ fronts
//! (BoTorch's `sample_simplex` and `sample_hypersphere`).

mod common;

use std::collections::BTreeMap;

use botorch_core::test_functions::multi_objective::{
    hypersphere_from_normal, simplex_from_uniform,
};
use botorch_core::test_functions::{MultiObjectiveKind, MultiObjectiveTestProblem};
use common::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Kwargs {
    dim: Option<usize>,
    num_objectives: Option<usize>,
}

/// BoTorch main's own values for problems that botorch-rust corrects.
#[derive(Debug, Deserialize)]
struct BotorchMain {
    max_hv: Option<f64>,
    values: Vec<Vec<f64>>,
    slack: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Deserialize)]
struct Entry {
    name: String,
    kwargs: Kwargs,
    reference: String,
    deviation: Option<String>,
    dim: usize,
    num_objectives: usize,
    num_constraints: usize,
    bounds: Vec<Vec<f64>>,
    raw_ref_point: Vec<f64>,
    ref_point: Vec<f64>,
    ref_point_negated: Vec<f64>,
    max_hv: Option<f64>,
    x: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    values_negated: Vec<Vec<f64>>,
    slack: Option<Vec<Vec<f64>>>,
    pareto_fronts: Option<BTreeMap<String, Vec<Vec<f64>>>>,
    pareto_fronts_negated: Option<BTreeMap<String, Vec<Vec<f64>>>>,
    /// Penicillin: BoTorch's own largest change of each value when the inputs
    /// are perturbed by one unit in the last place.
    sensitivity: Option<Vec<Vec<f64>>>,
    botorch: Option<BotorchMain>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    problems: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
struct SimplexCase {
    uniform: Vec<Vec<f64>>,
    samples: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct SphereCase {
    normal: Vec<Vec<f64>>,
    samples: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct Sampling {
    sample_simplex: Vec<SimplexCase>,
    sample_hypersphere: Vec<SphereCase>,
}

/// Tolerance (relative, per output column) of values and slacks: rounding
/// only. Penicillin's explicit Euler simulation is ill-conditioned (perturbing
/// the inputs by one unit in the last place changes BoTorch's own values by up
/// to 8e-11 at the fixture points and 7e-4 at random points), so ulp-level
/// differences between math libraries (exp, expm1, pow) are amplified: its
/// values are compared with `max(TOL, NOISE_FLOOR_FACTOR × BoTorch's own
/// change)`, per value (see `noise_floor_err`).
const TOL: f64 = 1e-12;
/// Multiple of BoTorch's sensitivity to rounding allowed for Penicillin.
const NOISE_FLOOR_FACTOR: f64 = 10.0;
/// Tolerance of constants computed from closed forms (`max_hv`).
const TOL_CONST: f64 = 1e-15;

fn load() -> Fixture {
    let path = fixture_dir()
        .join("phase5a_tf")
        .join("mo_test_functions.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn label(e: &Entry) -> String {
    let mut s = e.name.clone();
    let mut args = vec![];
    if let Some(d) = e.kwargs.dim {
        args.push(format!("d={d}"));
    }
    if let Some(m) = e.kwargs.num_objectives {
        args.push(format!("M={m}"));
    }
    if !args.is_empty() {
        s.push_str(&format!("({})", args.join(",")));
    }
    s
}

fn build(e: &Entry) -> Option<MultiObjectiveTestProblem> {
    let kind = MultiObjectiveKind::from_name(&e.name)?;
    Some(MultiObjectiveTestProblem::new(kind, e.kwargs.dim, e.kwargs.num_objectives).unwrap())
}

/// The worst [`rel_err`] over the columns of row-major `n × width` values
/// (outputs of very different scales are compared separately). Non-finite
/// values must match exactly.
fn col_rel_err(actual: &[f64], expected: &[Vec<f64>]) -> f64 {
    let width = expected.first().map_or(1, Vec::len);
    let flat = flatten(expected);
    assert_eq!(actual.len(), flat.len(), "length mismatch");
    (0..width)
        .map(|j| {
            let a: Vec<f64> = actual.iter().skip(j).step_by(width).copied().collect();
            let e: Vec<f64> = flat.iter().skip(j).step_by(width).copied().collect();
            if a.iter().zip(&e).any(|(x, y)| x.is_nan() != y.is_nan()) {
                return f64::INFINITY;
            }
            rel_err(&a, &e)
        })
        .fold(0.0, f64::max)
}

/// For values with a known sensitivity to rounding (`sens`, absolute): the
/// worst ratio of the error to `max(TOL, NOISE_FLOOR_FACTOR × sens)`, both
/// relative like [`col_rel_err`], scaled to `TOL` (so that it is at most `TOL`
/// if all values are within their tolerances).
fn noise_floor_err(actual: &[f64], expected: &[Vec<f64>], sens: &[Vec<f64>]) -> f64 {
    let width = expected[0].len();
    let (flat, flat_sens) = (flatten(expected), flatten(sens));
    let mut worst: f64 = 0.0;
    for j in 0..width {
        let scale = flat
            .iter()
            .skip(j)
            .step_by(width)
            .fold(0.0f64, |m, v| m.max(v.abs()));
        let floor = if scale > 0.0 { scale } else { 1.0 };
        for i in (j..flat.len()).step_by(width) {
            let denom = flat[i].abs() + floor;
            let err = (actual[i] - flat[i]).abs() / denom;
            let tol = TOL.max(NOISE_FLOOR_FACTOR * flat_sens[i] / denom);
            worst = worst.max(err / tol * TOL);
        }
    }
    worst
}

fn rel(a: f64, e: f64) -> f64 {
    (a - e).abs() / e.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn parity_mo_test_functions() {
    let fixture = load();
    let mut r = Report::default();
    for e in &fixture.problems {
        let Some(p) = build(e) else {
            panic!("{}: unknown problem", e.name);
        };
        let label = label(e);
        assert_eq!(p.dim(), e.dim, "{label}");
        assert_eq!(p.num_objectives(), e.num_objectives, "{label}");
        assert_eq!(p.num_constraints(), e.num_constraints, "{label}");
        assert_eq!(
            matrix_rows(p.bounds()),
            flatten(&e.bounds),
            "{label}: bounds"
        );
        assert_eq!(p.raw_ref_point(), e.raw_ref_point, "{label}: _ref_point");
        assert_eq!(p.ref_point(), e.ref_point, "{label}: ref_point");
        let neg = p.clone().with_negate(true);
        assert_eq!(neg.ref_point(), e.ref_point_negated, "{label}: ref_point");
        match (p.max_hv(), e.max_hv) {
            (Some(a), Some(b)) => r.check(&label, "max_hv", rel(a, b), TOL_CONST),
            (None, None) => {}
            other => panic!("{label}: max_hv {other:?}"),
        }
        let x = flatten(&e.x);
        let d = e.dim;
        let v = p.evaluate_true_rows(&x, d).unwrap();
        let vn = neg.forward_rows(&x, d, false).unwrap();
        if let Some(sens) = &e.sensitivity {
            // Penicillin: relative to BoTorch's own sensitivity to rounding.
            r.check(&label, "values (raw)", col_rel_err(&v, &e.values), f64::MAX);
            let err = noise_floor_err(&v, &e.values, sens);
            r.check(&label, "values (scaled to noise floor)", err, TOL);
            let err = noise_floor_err(&vn, &e.values_negated, sens);
            r.check(&label, "values (negate, scaled)", err, TOL);
        } else {
            r.check(&label, "values", col_rel_err(&v, &e.values), TOL);
            let err = col_rel_err(&vn, &e.values_negated);
            r.check(&label, "values (negate)", err, TOL);
        }
        match &e.slack {
            Some(expected) => {
                let c = p.evaluate_slack_true_rows(&x, d).unwrap();
                r.check(&label, "slack", col_rel_err(&c, expected), TOL);
                // Slacks are not negated.
                let cn = neg.evaluate_slack_rows(&x, d, false).unwrap();
                assert_eq!(c, cn, "{label}: negated slack");
            }
            None => assert!(p.evaluate_slack_true_rows(&x, d).is_err(), "{label}"),
        }
        if let (Some(fronts), Some(fronts_neg)) = (&e.pareto_fronts, &e.pareto_fronts_negated) {
            for (n, expected) in fronts {
                let front_label = format!("{label}, n={n}");
                let n: usize = n.parse().unwrap();
                let f = p.gen_pareto_front(n, None).unwrap();
                assert_eq!(f.nrows(), expected.len(), "{front_label}: front size");
                let err = col_rel_err(&matrix_rows(&f), expected);
                r.check(&front_label, "pareto front", err, TOL);
                let fneg = neg.gen_pareto_front(n, None).unwrap();
                let expected_neg = &fronts_neg[&n.to_string()];
                let err = col_rel_err(&matrix_rows(&fneg), expected_neg);
                r.check(&front_label, "pareto front (negate)", err, TOL);
            }
        }
        // Problems that botorch-rust corrects: BoTorch main must differ.
        let kind = p.kind();
        assert_eq!(
            kind.deviation().is_some(),
            e.deviation.is_some(),
            "{label}: deviation"
        );
        if let Some(bt) = &e.botorch {
            assert!(e.reference != "BoTorch" || e.deviation.is_none(), "{label}");
            let dev = e.deviation.as_deref().unwrap();
            let values_differ = flatten(&bt.values) != flatten(&e.values);
            let slack_differs = match (&bt.slack, &e.slack) {
                (Some(a), Some(b)) => flatten(a) != flatten(b),
                (None, Some(_)) => true,
                _ => false,
            };
            let max_hv_differs = bt.max_hv != e.max_hv;
            assert!(
                values_differ || slack_differs || max_hv_differs,
                "{label}: BoTorch main does not deviate ({dev})"
            );
            if e.name == "GMM" {
                // BoTorch main evaluates in float32: agreement to float32 precision.
                let err = col_rel_err(&v, &bt.values);
                r.check(&label, "values vs float32 BoTorch main", err, 1e-6);
            }
        }
    }
    r.finish("multi-objective test functions");
}

#[test]
fn every_botorch_problem_is_covered() {
    let fixture = load();
    let mut names: Vec<&str> = fixture.problems.iter().map(|e| e.name.as_str()).collect();
    names.dedup();
    let kinds: Vec<&str> = MultiObjectiveKind::ALL.iter().map(|k| k.name()).collect();
    assert_eq!(names, kinds);
}

#[test]
fn parity_front_samplers() {
    let path = fixture_dir().join("phase5a_tf").join("sampling.json");
    let fixture: Sampling = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut r = Report::default();
    for (i, c) in fixture.sample_simplex.iter().enumerate() {
        let s = simplex_from_uniform(&matrix(&c.uniform));
        r.check_slices(
            &format!("sample_simplex[{i}]"),
            "simplex",
            &matrix_rows(&s),
            &flatten(&c.samples),
            1e-15,
        );
    }
    for (i, c) in fixture.sample_hypersphere.iter().enumerate() {
        let s = hypersphere_from_normal(&matrix(&c.normal));
        r.check_slices(
            &format!("sample_hypersphere[{i}]"),
            "hypersphere",
            &matrix_rows(&s),
            &flatten(&c.samples),
            1e-15,
        );
    }
    r.finish("Pareto-front samplers");
}
