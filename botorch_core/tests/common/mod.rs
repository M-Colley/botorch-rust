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

// ---------------------------------------------------------------------------
// Comparison of fits on twins (two stages)
// ---------------------------------------------------------------------------

/// Fit losses within this distance count as equal: the tolerance of every
/// fit comparison and the ties of the statistical checks.
pub const FIT_TOL: f64 = 1e-6;
/// BoTorch's outcomes within this distance of its best loss belong to its
/// best optimum; their spread scales the stage-1 tolerance.
pub const FIT_CLUSTER: f64 = 1e-4;
/// Twins (besides the model itself) compared in stage 1. The fixtures hold
/// BoTorch's outcomes on 64 twins, all of which stage 2 compares.
pub const STAGE1_TWINS: usize = 16;
/// One-sided significance level of each statistical check of stage 2.
pub const FIT_ALPHA: f64 = 1e-3;
/// The standard normal quantile `Φ⁻¹(1 - FIT_ALPHA)`.
pub const FIT_Z: f64 = 3.090_232_306_167_813;

/// The z statistic of the one-sided Mann–Whitney U test that the values
/// `a` are stochastically larger than `b`: `U` counts the pairs with
/// `a_i > b_j + tie` (pairs within `tie` count one half), standardized with
/// the no-ties variance and a continuity correction. Ties and positive
/// dependence within twin pairs only reduce the variance of `U`, so the
/// test is conservative.
pub fn mann_whitney_z(a: &[f64], b: &[f64], tie: f64) -> f64 {
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let mut u = 0.0;
    for &x in a {
        for &y in b {
            if x > y + tie {
                u += 1.0;
            } else if (x - y).abs() <= tie {
                u += 0.5;
            }
        }
    }
    let sd = (n1 * n2 * (n1 + n2 + 1.0) / 12.0).sqrt();
    (u - n1 * n2 / 2.0 - 0.5) / sd
}

/// The smallest `k` such that all `k` smallest of `n_rs + n_bt`
/// exchangeable values come from the `n_bt` with probability at most
/// [`FIT_ALPHA`] (`C(n_bt, k) / C(n_rs + n_bt, k) <= FIT_ALPHA`), or
/// `n_bt + 1`.
pub fn rank_bound_k(n_rs: usize, n_bt: usize) -> usize {
    let mut p = 1.0;
    for k in 1..=n_bt {
        p *= (n_bt - k + 1) as f64 / (n_rs + n_bt - k + 1) as f64;
        if p <= FIT_ALPHA {
            return k;
        }
    }
    n_bt + 1
}

/// One-sided Fisher exact test that botorch_rs's first fit attempts fail
/// (i.e. need a retry from prior samples) more often than BoTorch's on the
/// same model and twins: `rs_failed` of `rs_total` against `bt_failed` of
/// `bt_total`. The p value is the hypergeometric probability of at least
/// `rs_failed` failures among botorch_rs's fits given the total number of
/// failures; the check (reported as `-log10 p`) fails if `p <= FIT_ALPHA`,
/// under the null hypothesis of equal failure rates with probability at
/// most [`FIT_ALPHA`].
pub fn compare_first_attempt_failures(
    report: &mut Report,
    name: &str,
    label: &str,
    (bt_failed, bt_total): (usize, usize),
    (rs_failed, rs_total): (usize, usize),
) {
    use statrs::distribution::{DiscreteCDF, Hypergeometric};
    let p = if rs_failed == 0 {
        1.0
    } else {
        Hypergeometric::new(
            (bt_total + rs_total) as u64,
            (bt_failed + rs_failed) as u64,
            rs_total as u64,
        )
        .unwrap()
        .sf(rs_failed as u64 - 1)
    };
    report.check(
        name,
        &format!("{label}: first attempts failing more often (-log10 Fisher p)"),
        0.0 - p.log10(),
        -FIT_ALPHA.log10(),
    );
}

/// The outcome of a fit: the final loss and the number of attempts.
pub type FitOutcome = (f64, usize);

/// Stage 2 of a [`FitComparison`]: the Mann–Whitney z of botorch_rs's
/// losses, the rank `k` ([`rank_bound_k`]) and BoTorch's `k`-th best loss.
#[derive(Debug, Clone)]
pub struct Stage2 {
    pub z: f64,
    pub k: usize,
    pub kth: f64,
}

/// The two-stage comparison of a fit case (see [`two_stage_fit_comparison`]).
#[derive(Debug, Clone)]
pub struct FitComparison {
    /// The number of twins (besides the model) of the deciding stage.
    pub twins: usize,
    /// BoTorch's and botorch_rs's first-attempt losses of the deciding stage.
    pub bt: Vec<f64>,
    pub rs: Vec<f64>,
    /// Stage 1: BoTorch's spread at its best optimum, the tolerance
    /// `max(FIT_TOL, 10 × spread)`, and the excesses (in units of the
    /// tolerance) of botorch_rs's best loss over BoTorch's best and of its
    /// worst loss over BoTorch's worst.
    pub spread: f64,
    pub tol: f64,
    pub best_excess: f64,
    pub worst_excess: f64,
    /// `None` if stage 1 decided.
    pub stage2: Option<Stage2>,
}

impl FitComparison {
    /// The deciding stage (1 or 2).
    pub fn stage(&self) -> usize {
        if self.stage2.is_some() {
            2
        } else {
            1
        }
    }

    /// botorch_rs's best first-attempt loss of the deciding stage.
    pub fn rs_best(&self) -> f64 {
        self.rs.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// BoTorch's best and worst first-attempt losses of the deciding stage.
    pub fn bt_range(&self) -> (f64, f64) {
        let best = self.bt.iter().copied().fold(f64::INFINITY, f64::min);
        let worst = self.bt.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (best, worst)
    }

    pub fn passed(&self) -> bool {
        match &self.stage2 {
            None => self.best_excess <= 1.0 && self.worst_excess <= 1.0,
            Some(s) => s.z <= FIT_Z && self.rs_best() - s.kth <= FIT_TOL,
        }
    }

    /// The deciding values, and for stage 2 the sorted outcomes of both
    /// libraries.
    pub fn diagnostics(&self, label: &str) -> String {
        let mut out = format!(
            "{label}: stage 1: best above BoTorch's best {:.3} x tol, worst above BoTorch's \
             worst {:.3} x tol (tol {:.1e}, BoTorch's spread at its best {:.1e})",
            self.best_excess, self.worst_excess, self.tol, self.spread
        );
        if let Some(s) = &self.stage2 {
            let sorted = |v: &[f64]| {
                let mut v = v.to_vec();
                v.sort_by(f64::total_cmp);
                v.iter()
                    .map(|x| format!("{x:+.10}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            out += &format!(
                "\nstage 2 on {} BoTorch and {} botorch_rs first-attempt outcomes: Mann-Whitney \
                 z = {:.3} (limit {FIT_Z:.3}), k = {}, BoTorch's k-th best {:+.10}, botorch_rs's \
                 best {:+.10} (tolerance {FIT_TOL:e}): {}\nBoTorch (sorted): {}\nbotorch_rs \
                 (sorted): {}",
                self.bt.len(),
                self.rs.len(),
                s.z,
                s.k,
                s.kth,
                self.rs_best(),
                if self.passed() { "pass" } else { "FAIL" },
                sorted(&self.bt),
                sorted(&self.rs)
            );
        }
        out
    }
}

/// The two-stage comparison of fits from identical initial values (see
/// `parity_helpers.two_stage_fit_comparisons` in the Python tests and
/// `docs/fitting.md`, "Comparing fits"). `bt` holds BoTorch's outcomes on
/// the model (index 0) and its twins `1..` (training inputs perturbed at
/// rounding level, 64 in the fixtures), `bt_extra` further first-attempt
/// outcomes of BoTorch on the model (e.g. with L-BFGS-B's variables
/// reversed), `rs_own` botorch_rs's outcome on the model and `rs_twin(t)`
/// its outcome on twin `t`. Only first attempts are compared (retries start
/// from prior samples drawn by different generators).
///
/// 1. On the model and the first [`STAGE1_TWINS`] twins: botorch_rs's best
///    loss must reach BoTorch's best, and none of its losses may end above
///    BoTorch's worst (its own fit's, and every twin's), both within
///    `max(FIT_TOL, 10 × BoTorch's spread at its best optimum)`. If so, the
///    case passes. (The live Python tests check only botorch_rs's fit of the
///    model itself against BoTorch's worst, as the former bound did: their
///    stage 2 needs up to 64 more BoTorch fits. Here stage 2 costs only
///    botorch_rs's fits, and checking every stage-1 fit sends more
///    regressions to it; see `docs/fitting.md`.)
/// 2. Otherwise botorch_rs fits the remaining twins as well, and on all
///    outcomes its losses must pass a one-sided Mann–Whitney U test at level
///    [`FIT_ALPHA`] (not stochastically larger than BoTorch's; ties within
///    [`FIT_TOL`]), and its best loss must reach BoTorch's `k`-th best
///    ([`rank_bound_k`]) within [`FIT_TOL`].
///
/// The case fails only if both stages fail; stage 2 alone fails by chance
/// (botorch_rs's outcomes another sample of BoTorch's) with probability at
/// most `2 × FIT_ALPHA`, so the procedure does as well.
pub fn two_stage_fit_comparison(
    bt: &[FitOutcome],
    bt_extra: &[f64],
    rs_own: FitOutcome,
    rs_twin: &mut dyn FnMut(usize) -> FitOutcome,
) -> FitComparison {
    let first = |out: &[FitOutcome]| -> Vec<f64> {
        out.iter()
            .filter(|&&(_, a)| a == 1)
            .map(|&(l, _)| l)
            .collect()
    };
    let n_twins = bt.len() - 1;
    let n1 = STAGE1_TWINS.min(n_twins);
    let mut rs_out = vec![rs_own];
    rs_out.extend((1..=n1).map(&mut *rs_twin));
    let mut bt1 = first(&bt[..=n1]);
    bt1.extend_from_slice(bt_extra);
    assert!(!bt1.is_empty(), "BoTorch's fits all needed retries");
    let rs1 = first(&rs_out);
    let best = bt1.iter().copied().fold(f64::INFINITY, f64::min);
    let worst = bt1.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let spread = bt1
        .iter()
        .copied()
        .filter(|&l| l <= best + FIT_CLUSTER)
        .fold(best, f64::max)
        - best;
    let tol = FIT_TOL.max(10.0 * spread);
    let rs_best = rs1.iter().copied().fold(f64::INFINITY, f64::min);
    let rs_worst = rs1.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut c = FitComparison {
        twins: n1,
        bt: bt1,
        rs: rs1,
        spread,
        tol,
        best_excess: (rs_best - best).max(0.0) / tol,
        worst_excess: (rs_worst - worst).max(0.0) / tol,
        stage2: None,
    };
    if c.passed() || n_twins <= n1 {
        return c;
    }
    rs_out.extend((n1 + 1..=n_twins).map(&mut *rs_twin));
    c.twins = n_twins;
    c.bt = first(bt);
    c.bt.extend_from_slice(bt_extra);
    c.rs = first(&rs_out);
    let z = if c.rs.is_empty() {
        f64::INFINITY
    } else {
        mann_whitney_z(&c.rs, &c.bt, FIT_TOL)
    };
    let k = rank_bound_k(c.rs.len(), c.bt.len());
    let mut sorted = c.bt.clone();
    sorted.sort_by(f64::total_cmp);
    let kth = sorted.get(k - 1).copied().unwrap_or(f64::INFINITY);
    c.stage2 = Some(Stage2 { z, k, kth });
    c
}

/// [`two_stage_fit_comparison`], with the deciding quantities checked in
/// `report` under `label`. A case decided by stage 2 prints its diagnostics
/// (stage 1's excesses, z, k, BoTorch's `k`-th best and the sorted outcomes
/// of both libraries; visible in the output of a failing test).
pub fn compare_fits(
    report: &mut Report,
    name: &str,
    label: &str,
    bt: &[FitOutcome],
    bt_extra: &[f64],
    rs_own: FitOutcome,
    rs_twin: &mut dyn FnMut(usize) -> FitOutcome,
) -> FitComparison {
    let c = two_stage_fit_comparison(bt, bt_extra, rs_own, rs_twin);
    match &c.stage2 {
        None => {
            report.check(
                name,
                &format!("{label}: best loss above BoTorch's best (scaled)"),
                c.best_excess,
                1.0,
            );
            report.check(
                name,
                &format!("{label}: worst loss above BoTorch's worst (scaled)"),
                c.worst_excess,
                1.0,
            );
        }
        Some(s) => {
            println!("{name}: {}", c.diagnostics(label));
            report.check(
                name,
                &format!("{label} (stage 2): stage-1 excess (scaled)"),
                c.best_excess.max(c.worst_excess),
                f64::INFINITY,
            );
            report.check(
                name,
                &format!("{label} (stage 2): Mann-Whitney z"),
                s.z.max(0.0),
                FIT_Z,
            );
            report.check(
                name,
                &format!("{label} (stage 2): best above BoTorch's k-th best"),
                (c.rs_best() - s.kth).max(0.0),
                FIT_TOL,
            );
        }
    }
    c
}

/// [`compare_fits`] and a one-sided Fisher exact test that botorch_rs's
/// first fit attempts do not fail (end in a retry from prior samples) more
/// often than BoTorch's on the same fits ([`compare_first_attempt_failures`]).
/// The test runs on the fits of the stage that decided [`compare_fits`]; if
/// botorch_rs's first attempts failed more often than BoTorch's there,
/// botorch_rs also fits the remaining twins and the test runs on all of
/// them. The case fails only if the test on all twins fails, so with
/// probability at most [`FIT_ALPHA`] under equal failure rates. Each twin is
/// fitted at most once.
pub fn compare_fits_and_first_attempts(
    report: &mut Report,
    name: &str,
    label: &str,
    bt: &[FitOutcome],
    bt_extra: &[f64],
    rs_own: FitOutcome,
    rs_twin: &mut dyn FnMut(usize) -> FitOutcome,
) -> FitComparison {
    let n_twins = bt.len() - 1;
    let mut cache: Vec<Option<FitOutcome>> = vec![None; n_twins + 1];
    let mut cached = |t: usize, cache: &mut Vec<Option<FitOutcome>>| -> FitOutcome {
        *cache[t].get_or_insert_with(|| rs_twin(t))
    };
    let c = compare_fits(report, name, label, bt, bt_extra, rs_own, &mut |t| {
        cached(t, &mut cache)
    });
    let failed = |out: &[FitOutcome]| out.iter().filter(|&&(_, a)| a != 1).count();
    let decided = c.twins;
    let rs_decided: Vec<FitOutcome> = std::iter::once(rs_own)
        .chain((1..=decided).map(|t| cached(t, &mut cache)))
        .collect();
    let (mut rs_all, mut bt_all) = (rs_decided, bt[..=decided].to_vec());
    if failed(&rs_all) > failed(&bt_all) && decided < n_twins {
        rs_all.extend((decided + 1..=n_twins).map(|t| cached(t, &mut cache)));
        bt_all = bt.to_vec();
    }
    compare_first_attempt_failures(
        report,
        name,
        label,
        (failed(&bt_all), bt_all.len()),
        (failed(&rs_all), rs_all.len()),
    );
    c
}
