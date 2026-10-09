//! Parity of the synthetic test functions with BoTorch
//! (`tests/fixtures/phase2/test_functions.json`): values at random points and
//! at the known optimizers, negation, bounds, optimal values, optimizers and
//! the minimization flag.

mod common;

use botorch_core::test_functions::{SyntheticTestFunction, TestFunctionKind};
use common::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Kwargs {
    dim: Option<usize>,
    m: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Entry {
    function: String,
    kwargs: Kwargs,
    dim: usize,
    bounds: Vec<Vec<f64>>,
    is_minimization_problem: bool,
    is_minimization_problem_negated: bool,
    optimal_value: Option<f64>,
    optimal_value_negated: Option<f64>,
    optimizers: Option<Vec<Vec<f64>>>,
    x: Vec<Vec<f64>>,
    values: Vec<f64>,
    values_negated: Vec<f64>,
    values_at_optimizers: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    functions: Vec<Entry>,
}

fn build(e: &Entry) -> SyntheticTestFunction {
    let kind = TestFunctionKind::ALL
        .into_iter()
        .find(|k| k.name() == e.function)
        .unwrap_or_else(|| panic!("unknown test function {}", e.function));
    match kind {
        TestFunctionKind::Shekel => {
            SyntheticTestFunction::shekel(e.kwargs.m.unwrap_or(10)).unwrap()
        }
        _ => SyntheticTestFunction::new(kind, e.kwargs.dim).unwrap(),
    }
}

#[test]
fn parity_test_functions() {
    let path = fixture_dir().join("phase2").join("test_functions.json");
    let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut r = Report::default();
    for e in &fixture.functions {
        let f = build(e);
        let label = format!(
            "{}(dim={}{})",
            e.function,
            e.dim,
            e.kwargs.m.map(|m| format!(", m={m}")).unwrap_or_default()
        );
        assert_eq!(f.dim(), e.dim, "{label}");
        assert_eq!(
            matrix_rows(f.bounds()),
            flatten(&e.bounds),
            "{label}: bounds"
        );
        assert_eq!(
            f.is_minimization_problem(),
            e.is_minimization_problem,
            "{label}"
        );
        assert_eq!(f.optimal_value(), e.optimal_value, "{label}: optimal value");
        let neg = f.clone().with_negate(true);
        assert_eq!(
            neg.is_minimization_problem(),
            e.is_minimization_problem_negated,
            "{label}"
        );
        assert_eq!(neg.optimal_value(), e.optimal_value_negated, "{label}");
        match (&f.optimizers(), &e.optimizers) {
            (Some(a), Some(b)) => assert_eq!(matrix_rows(a), flatten(b), "{label}: optimizers"),
            (None, None) => {}
            other => panic!("{label}: optimizers {other:?}"),
        }
        let x = matrix(&e.x);
        let v = f.evaluate_true(&x).unwrap();
        r.check_slices(&label, "values", v.as_slice(), &e.values, 1e-12);
        let vn = neg.forward(&x, false).unwrap();
        r.check_slices(
            &label,
            "values (negate)",
            vn.as_slice(),
            &e.values_negated,
            1e-12,
        );
        if let (Some(opt), Some(expected)) = (f.optimizers(), &e.values_at_optimizers) {
            let v = f.evaluate_true(&opt).unwrap();
            r.check_slices(
                &label,
                "values at optimizers",
                v.as_slice(),
                expected,
                1e-12,
            );
        }
    }
    r.finish("synthetic test functions");
}
