//! Shared helpers for the integration tests: BoTorch fixture loading, model
//! construction from a fixture configuration and tolerance checks.

#![allow(dead_code)]

pub mod kernel_spec;

use std::collections::BTreeMap;
use std::path::PathBuf;

use botorch_core::kernels::get_matern_kernel_with_gamma_prior;
use botorch_core::{
    get_gaussian_likelihood_with_gamma_prior, KernelKind, Matrix, Normalize,
    OutcomeTransformChoice, SingleTaskGP, SingleTaskGPOptions, Vector,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub d: usize,
    pub n: usize,
    pub covar: String,
    pub likelihood: String,
    pub yvar: Option<String>,
    pub normalize: Option<String>,
    pub outcome: String,
    pub bounds: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Deserialize)]
pub struct Transformed {
    pub train_x: Vec<Vec<f64>>,
    pub train_y: Vec<f64>,
    pub test_x: Vec<Vec<f64>>,
    pub normalize_offset: Option<Vec<f64>>,
    pub normalize_coefficient: Option<Vec<f64>>,
    pub standardize_mean: Option<Vec<f64>>,
    pub standardize_std: Option<Vec<f64>>,
    pub train_yvar: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct KernelFixture {
    pub k_train: Vec<Vec<f64>>,
    pub k_test_train: Vec<Vec<f64>>,
    pub dk_draw: BTreeMap<String, Vec<Vec<Vec<f64>>>>,
    pub lengthscale: Vec<f64>,
    pub outputscale: Option<f64>,
    pub likelihood_noise: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct MllFixture {
    pub value: f64,
    pub grad: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct JointFixture {
    pub mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub covariance_noisy: Vec<Vec<f64>>,
    /// q x q x d: d mean[a] / d x[p, k]
    pub dmean_dx: Vec<Vec<Vec<f64>>>,
    /// q x q x q x d: d cov[a, b] / d x[p, k]
    pub dcov_dx: Vec<Vec<Vec<Vec<f64>>>>,
}

#[derive(Debug, Deserialize)]
pub struct PosteriorFixture {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub variance_noisy: Vec<f64>,
    pub dmean_dx: Vec<Vec<f64>>,
    pub dvariance_dx: Vec<Vec<f64>>,
    pub joint: JointFixture,
}

#[derive(Debug, Deserialize)]
pub struct FitFixture {
    pub initial_raw_params: Vec<f64>,
    pub raw_params: Vec<f64>,
    pub loss: f64,
    pub grad: Vec<f64>,
    pub fval: f64,
    pub iterations: usize,
    pub status: String,
    pub message: String,
    pub attempts: usize,
    pub free_indices: Vec<usize>,
    pub hessian_eigenvalues: Vec<f64>,
    pub posterior_mean: Vec<f64>,
    pub posterior_variance: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub name: String,
    pub config: Config,
    pub train_x: Vec<Vec<f64>>,
    pub train_y: Vec<Vec<f64>>,
    pub train_yvar: Option<Vec<Vec<f64>>>,
    pub test_x: Vec<Vec<f64>>,
    pub param_names: Vec<(String, usize)>,
    pub default_raw_params: Vec<f64>,
    pub default_mll: MllFixture,
    pub raw_params: Vec<f64>,
    pub transform_derivative: Vec<f64>,
    pub transformed: Transformed,
    pub kernel: KernelFixture,
    pub mll: MllFixture,
    pub posterior: PosteriorFixture,
    pub fit: FitFixture,
}

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Loads every fixture (sorted by name).
pub fn load_fixtures() -> Vec<Fixture> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(fixture_dir())
        .expect("fixture directory exists")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no fixtures found; run scripts/generate_fixtures.py"
    );
    paths
        .iter()
        .map(|p| {
            let text = std::fs::read_to_string(p).unwrap();
            serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("failed to parse {}: {e}", p.display()))
        })
        .collect()
}

pub fn matrix(rows: &[Vec<f64>]) -> Matrix {
    let n = rows.len();
    let d = if n == 0 { 0 } else { rows[0].len() };
    Matrix::from_fn(n, d, |i, j| rows[i][j])
}

pub fn vector(v: &[f64]) -> Vector {
    Vector::from_column_slice(v)
}

/// Builds the botorch_core model corresponding to a fixture configuration.
pub fn build_model(f: &Fixture) -> SingleTaskGP {
    build_model_from(&f.config, &f.train_x, &f.train_y, f.train_yvar.as_ref())
}

/// Builds the botorch_core model for a configuration and training data.
pub fn build_model_from(
    c: &Config,
    train_x: &[Vec<f64>],
    train_y: &[Vec<f64>],
    train_yvar: Option<&Vec<Vec<f64>>>,
) -> SingleTaskGP {
    let mut options = SingleTaskGPOptions {
        train_yvar: train_yvar.map(|v| matrix(v)),
        ..Default::default()
    };
    match c.covar.as_str() {
        "rbf" => options.kernel = KernelKind::Rbf,
        "matern" => options.kernel = KernelKind::Matern52,
        "matern_gamma" => options.covar_module = Some(get_matern_kernel_with_gamma_prior(c.d)),
        "rbf_scale" => {
            options.kernel = KernelKind::Rbf;
            options.use_scale_kernel = true;
        }
        other => panic!("unknown covar {other}"),
    }
    if c.likelihood == "gamma" {
        options.likelihood = Some(get_gaussian_likelihood_with_gamma_prior());
    }
    match c.normalize.as_deref() {
        None => {}
        Some("learned") => options.input_transform = Some(Normalize::new(c.d)),
        Some("explicit") => {
            let b = matrix(c.bounds.as_ref().expect("explicit bounds"));
            options.input_transform = Some(Normalize::with_bounds(&b).unwrap());
        }
        Some(other) => panic!("unknown normalize {other}"),
    }
    if c.outcome == "none" {
        options.outcome_transform = OutcomeTransformChoice::None;
    }
    SingleTaskGP::new(matrix(train_x), matrix(train_y), options).unwrap()
}

/// Mixed relative/normwise error: `max_i |a_i - e_i| / (|e_i| + max_j |e_j|·floor)`,
/// i.e. elementwise relative error for entries of the order of the largest
/// entry and normwise relative error for tiny entries.
pub fn rel_err(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    let scale = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let floor = if scale > 0.0 { scale } else { 1.0 };
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() / (e.abs() + floor))
        .fold(0.0, f64::max)
}

pub fn flatten(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

pub fn matrix_rows(m: &Matrix) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.len());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.push(m[(i, j)]);
        }
    }
    out
}

/// Collects per-fixture error statistics and reports violations at the end.
#[derive(Default)]
pub struct Report {
    rows: Vec<(String, String, f64, f64)>,
}

impl Report {
    pub fn check(&mut self, fixture: &str, quantity: &str, err: f64, tol: f64) {
        self.rows
            .push((fixture.to_string(), quantity.to_string(), err, tol));
    }

    pub fn check_slices(
        &mut self,
        fixture: &str,
        quantity: &str,
        actual: &[f64],
        expected: &[f64],
        tol: f64,
    ) {
        let err = rel_err(actual, expected);
        self.check(fixture, quantity, err, tol);
    }

    /// Like [`Report::check_slices`], but entries whose absolute error is at
    /// most `atol` count as exact. Used for gradients, whose components can
    /// be tiny results of cancellation (e.g. a prior gradient at the prior
    /// mode), where only absolute agreement is meaningful.
    pub fn check_slices_atol(
        &mut self,
        fixture: &str,
        quantity: &str,
        actual: &[f64],
        expected: &[f64],
        tol: f64,
        atol: f64,
    ) {
        let filtered: Vec<(f64, f64)> = actual
            .iter()
            .zip(expected)
            .map(|(&a, &e)| {
                if (a - e).abs() <= atol {
                    (e, e)
                } else {
                    (a, e)
                }
            })
            .collect();
        let (a, e): (Vec<f64>, Vec<f64>) = filtered.into_iter().unzip();
        self.check_slices(fixture, quantity, &a, &e, tol);
    }

    /// Prints a table (visible with `--nocapture`) and panics on violations.
    pub fn finish(self, title: &str) {
        let mut worst: BTreeMap<String, f64> = BTreeMap::new();
        println!("== {title} ==");
        for (fx, q, err, tol) in &self.rows {
            let flag = if err <= tol && err.is_finite() {
                ""
            } else {
                "  <-- FAIL"
            };
            println!("{fx:40} {q:28} err={err:9.2e} tol={tol:7.1e}{flag}");
            let w = worst.entry(q.clone()).or_insert(0.0);
            *w = w.max(*err);
        }
        println!("-- worst error per quantity --");
        for (q, w) in &worst {
            println!("{q:28} {w:9.2e}");
        }
        let failures: Vec<_> = self
            .rows
            .iter()
            .filter(|(_, _, e, t)| !(e <= t && e.is_finite()))
            .collect();
        assert!(
            failures.is_empty(),
            "{} parity violations: {failures:?}",
            failures.len()
        );
    }
}
