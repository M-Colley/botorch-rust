//! Parity of the multi-objective utilities with BoTorch
//! (`tests/fixtures/phase5a_utils/*.json`, `--phase 5a-utils`):
//! `is_non_dominated` (all flags, batches, BoTorch's loop variant,
//! infinities; NaN outcomes against BoTorch with meta-pytorch/botorch#3392),
//! `Hypervolume`, `infer_reference_point`,
//! `get_hypervolume_maximizing_subset` (replaying BoTorch's random
//! choices), the three box decompositions after construction, incremental
//! updates and `reset` (Pareto sets, cells and local upper bounds in
//! BoTorch's order, hypervolumes), the helpers of `box_decompositions.utils`,
//! the Chebyshev scalarization and the transformations of `sample_simplex`
//! and `sample_hypersphere`.
//!
//! Discrete outputs (masks, Pareto sets, chosen subsets, numbers of cells and
//! local upper bounds) must be identical; values must agree to a relative
//! 1e-12 (hypervolumes are sums over many cells whose summation order differs
//! from torch's reductions).

mod common;

use botorch_core::utils::multi_objective::box_decompositions::utils::{
    compute_local_upper_bounds, expand_ref_point, get_partition_bounds, pad_batch_pareto_frontier,
    update_local_upper_bounds_incremental,
};
use botorch_core::utils::multi_objective::box_decompositions::{
    compute_dominated_hypercell_bounds_2d, compute_non_dominated_hypercell_bounds_2d,
    BoxDecomposition, DominatedPartitioning, FastNondominatedPartitioning, FastPartitioning,
    LocalUpperBounds, NondominatedPartitioning,
};
use botorch_core::utils::multi_objective::hypervolume::{
    get_hypervolume_maximizing_subset, infer_reference_point, Hypervolume,
};
use botorch_core::utils::multi_objective::pareto::is_non_dominated_batch;
use botorch_core::utils::multi_objective::scalarization::get_chebyshev_scalarization;
use botorch_core::utils::multi_objective::PointBatch;
use botorch_core::utils::sampling::{hypersphere_from_normal, simplex_from_uniform};
use botorch_core::Matrix;
use common::{fixture_dir, Report};
use serde::de::DeserializeOwned;
use serde::Deserialize;

const VALUE_TOL: f64 = 1e-12;

/// A number of a fixture; the writer encodes non-finite numbers as the
/// strings "inf", "-inf" and "nan".
#[derive(Debug, Clone, Copy)]
struct F(f64);

impl<'de> Deserialize<'de> for F {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            F(f64),
            S(String),
        }
        Ok(F(match Raw::deserialize(d)? {
            Raw::F(v) => v,
            Raw::S(s) => match s.as_str() {
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                "nan" => f64::NAN,
                other => return Err(serde::de::Error::custom(format!("not a number: {other}"))),
            },
        }))
    }
}

fn vals(v: &[F]) -> Vec<f64> {
    v.iter().map(|f| f.0).collect()
}

fn load<T: DeserializeOwned>(name: &str) -> T {
    let path = fixture_dir().join("phase5a_utils").join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}; run scripts/generate_fixtures.py", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Identical values, including infinities and NaNs.
fn same(a: &[f64], e: &[f64]) -> bool {
    a.len() == e.len()
        && a.iter()
            .zip(e)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

/// Relative error of the finite entries; non-finite entries must match.
fn err(a: &[f64], e: &[f64]) -> f64 {
    assert_eq!(a.len(), e.len(), "length mismatch");
    let mut worst = 0.0f64;
    for (x, y) in a.iter().zip(e) {
        if !y.is_finite() || !x.is_finite() {
            if !(x == y || (x.is_nan() && y.is_nan())) {
                return f64::INFINITY;
            }
            continue;
        }
        worst = worst.max((x - y).abs() / y.abs().max(f64::MIN_POSITIVE));
    }
    worst
}

/// A point batch from flat values and a `(*batch, n, m)` shape.
fn batch(values: &[f64], shape: &[usize]) -> PointBatch {
    let (m, rest) = shape.split_last().unwrap();
    let (n, b) = rest.split_last().unwrap();
    PointBatch::new(b.to_vec(), *n, *m, values.to_vec()).unwrap()
}

#[derive(Deserialize)]
struct ParetoCase {
    name: String,
    y: Vec<F>,
    shape: Vec<usize>,
    masks: std::collections::BTreeMap<String, Vec<bool>>,
}

#[derive(Deserialize)]
struct ParetoFixture {
    cases: Vec<ParetoCase>,
}

fn check_pareto(file: &str) -> usize {
    let fixture: ParetoFixture = load(file);
    let mut checked = 0;
    for c in &fixture.cases {
        let y = batch(&vals(&c.y), &c.shape);
        for (key, expected) in &c.masks {
            let (mx, dd) = key.split_once('_').unwrap();
            let mask = is_non_dominated_batch(&y, mx == "True", dd == "True");
            assert_eq!(&mask, expected, "{file} {} {key}", c.name);
            checked += 1;
        }
    }
    checked
}

#[test]
fn parity_is_non_dominated() {
    assert_eq!(check_pareto("pareto.json"), 380);
}

#[test]
fn parity_is_non_dominated_with_nan() {
    // Expected masks recorded with BoTorch including meta-pytorch/botorch#3392.
    assert_eq!(check_pareto("pareto_nan.json"), 16);
}

#[derive(Deserialize)]
struct HvCase {
    name: String,
    ref_point: Vec<F>,
    y: Vec<Vec<F>>,
    hv: F,
}

#[derive(Deserialize)]
struct RefCase {
    name: String,
    pareto_y: Vec<Vec<F>>,
    shape: Vec<usize>,
    max_ref_point: Option<Vec<F>>,
    scale: f64,
    scale_max_ref_point: bool,
    ref_point: Vec<F>,
}

#[derive(Deserialize)]
struct Choice {
    remaining: Vec<usize>,
    choice: usize,
}

#[derive(Deserialize)]
struct SubsetCase {
    name: String,
    n: usize,
    y: Vec<Vec<F>>,
    ref_point: Vec<F>,
    subset: Vec<Vec<F>>,
    indices: Vec<usize>,
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct HypervolumeFixture {
    hypervolume: Vec<HvCase>,
    infer_reference_point: Vec<RefCase>,
    maximizing_subset: Vec<SubsetCase>,
}

fn rows_matrix(rows: &[Vec<F>], m: usize) -> Matrix {
    Matrix::from_fn(rows.len(), m, |i, j| rows[i][j].0)
}

#[test]
fn parity_hypervolume() {
    let fixture: HypervolumeFixture = load("hypervolume.json");
    let mut r = Report::default();
    for c in &fixture.hypervolume {
        let refp = vals(&c.ref_point);
        let y = rows_matrix(&c.y, refp.len());
        let hv = Hypervolume::new(&refp).compute(&y).unwrap();
        r.check(
            &c.name,
            "Hypervolume.compute",
            err(&[hv], &[c.hv.0]),
            VALUE_TOL,
        );
    }
    for c in &fixture.infer_reference_point {
        let y = Matrix::from_fn(c.shape[0], c.shape[1], |i, j| c.pareto_y[i][j].0);
        let max_ref = c.max_ref_point.as_ref().map(|v| vals(v));
        let out =
            infer_reference_point(&y, max_ref.as_deref(), c.scale, c.scale_max_ref_point).unwrap();
        r.check(
            &c.name,
            "infer_reference_point",
            err(&out, &vals(&c.ref_point)),
            VALUE_TOL,
        );
    }
    for c in &fixture.maximizing_subset {
        let refp = vals(&c.ref_point);
        let y = rows_matrix(&c.y, refp.len());
        let mut choices = c.choices.iter();
        let (subset, idx) = get_hypervolume_maximizing_subset(c.n, &y, &refp, |remaining| {
            let rec = choices.next().expect("BoTorch made fewer random choices");
            assert_eq!(remaining, rec.remaining.as_slice(), "{}", c.name);
            rec.choice
        })
        .unwrap();
        assert!(
            choices.next().is_none(),
            "{}: BoTorch made more random choices",
            c.name
        );
        assert_eq!(idx, c.indices, "{}", c.name);
        let expected = rows_matrix(&c.subset, refp.len());
        assert_eq!(subset, expected, "{}", c.name);
    }
    assert!(fixture
        .maximizing_subset
        .iter()
        .any(|c| !c.choices.is_empty()));
    r.finish("hypervolume utilities");
}

#[derive(Deserialize)]
struct State {
    pareto_y: Vec<F>,
    pareto_shape: Vec<usize>,
    hv: Vec<F>,
    cells: Vec<F>,
    cells_shape: Vec<usize>,
    u: Option<Vec<F>>,
    z: Option<Vec<F>>,
    num_bounds: Option<usize>,
}

#[derive(Deserialize)]
struct BdCase {
    name: String,
    class: String,
    alpha: f64,
    ref_point: Vec<F>,
    ref_shape: Vec<usize>,
    y: Vec<F>,
    y_shape: Vec<usize>,
    updates: Vec<Vec<F>>,
    update_shapes: Vec<Vec<usize>>,
    reset: bool,
    states: Vec<State>,
}

#[derive(Deserialize)]
struct BdFixture {
    cases: Vec<BdCase>,
}

fn check_state(r: &mut Report, label: &str, d: &dyn BoxDecomposition, s: &State) {
    let pareto = d.pareto_y().unwrap();
    assert_eq!(pareto.shape(), s.pareto_shape, "{label}: pareto shape");
    assert!(
        same(pareto.values(), &vals(&s.pareto_y)),
        "{label}: Pareto points (in order)"
    );
    let cells = d.get_hypercell_bounds().unwrap();
    assert_eq!(cells.shape(), s.cells_shape, "{label}: number of cells");
    r.check(
        label,
        "hypercell bounds",
        err(&cells.to_flat(), &vals(&s.cells)),
        VALUE_TOL,
    );
    let hv = d.compute_hypervolume();
    r.check(
        label,
        "compute_hypervolume",
        err(&hv, &vals(&s.hv)),
        VALUE_TOL,
    );
}

fn check_lub(label: &str, lub: Option<&LocalUpperBounds>, s: &State) {
    match (&s.u, &s.z, lub) {
        (Some(u), Some(z), Some(lub)) => {
            assert_eq!(
                Some(lub.len()),
                s.num_bounds,
                "{label}: number of local upper bounds"
            );
            assert!(same(lub.upper_bounds(), &vals(u)), "{label}: U (in order)");
            assert!(
                same(lub.defining_points(), &vals(z)),
                "{label}: Z (in order)"
            );
        }
        (None, None, _) => {}
        _ => panic!("{label}: local upper bounds missing"),
    }
}

/// Runs a case with a typed fast partitioning so that its local upper
/// bounds can be compared too.
fn run_fast<D: FastPartitioning>(r: &mut Report, c: &BdCase, mut d: D) {
    for (k, s) in c.states.iter().enumerate() {
        if k > 0 {
            if k <= c.updates.len() {
                let y = batch(&vals(&c.updates[k - 1]), &c.update_shapes[k - 1]);
                d.update(&y).unwrap();
            } else {
                assert!(c.reset);
                d.reset().unwrap();
            }
        }
        let label = format!("{} [{k}]", c.name);
        check_state(r, &label, &d, s);
        check_lub(&label, d.local_upper_bounds(), s);
    }
}

#[test]
fn parity_box_decompositions() {
    let fixture: BdFixture = load("box_decompositions.json");
    let mut r = Report::default();
    let mut n_states = 0;
    for c in &fixture.cases {
        let refp = vals(&c.ref_point);
        let y = batch(&vals(&c.y), &c.y_shape);
        match c.class.as_str() {
            "FastNondominatedPartitioning" => run_fast(
                &mut r,
                c,
                FastNondominatedPartitioning::with_ref_point_shape(&refp, &c.ref_shape, Some(&y))
                    .unwrap(),
            ),
            "DominatedPartitioning" => run_fast(
                &mut r,
                c,
                DominatedPartitioning::with_ref_point_shape(&refp, &c.ref_shape, Some(&y)).unwrap(),
            ),
            "NondominatedPartitioning" => {
                let mut d = NondominatedPartitioning::with_ref_point_shape(
                    &refp,
                    &c.ref_shape,
                    Some(&y),
                    c.alpha,
                )
                .unwrap();
                for (k, s) in c.states.iter().enumerate() {
                    if k > 0 {
                        if k <= c.updates.len() {
                            let upd = batch(&vals(&c.updates[k - 1]), &c.update_shapes[k - 1]);
                            d.update(&upd).unwrap();
                        } else {
                            d.reset().unwrap();
                        }
                    }
                    check_state(&mut r, &format!("{} [{k}]", c.name), &d, s);
                }
            }
            other => panic!("unknown class {other}"),
        }
        n_states += c.states.len();
    }
    assert!(n_states > 300, "{n_states} states");
    r.finish("box decompositions");
}

#[derive(Deserialize)]
struct LubStep {
    z: Vec<F>,
    u: Vec<F>,
    z_points: Vec<F>,
    len: usize,
}

#[derive(Deserialize)]
struct LubCase {
    m: usize,
    u0: Vec<F>,
    z0: Vec<F>,
    steps: Vec<LubStep>,
    incremental_u: Vec<F>,
    incremental_z: Vec<F>,
    partition_bounds: Vec<F>,
    partition_shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Cells2dCase {
    y: Vec<F>,
    y_shape: Vec<usize>,
    ref_point: Vec<F>,
    non_dominated: Vec<F>,
    non_dominated_shape: Vec<usize>,
    dominated: Vec<F>,
    dominated_shape: Vec<usize>,
}

#[derive(Deserialize)]
struct PadCase {
    y: Vec<F>,
    y_shape: Vec<usize>,
    ref_point: Vec<F>,
    is_pareto: bool,
    mask: Option<Vec<bool>>,
    out: Vec<F>,
    out_shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Expand {
    ref_point: Vec<F>,
    batch_shape: Vec<usize>,
    out: Vec<F>,
}

#[derive(Deserialize)]
struct BoxUtilsFixture {
    local_upper_bounds: Vec<LubCase>,
    cells_2d: Vec<Cells2dCase>,
    pad: Vec<PadCase>,
    expand: Expand,
}

#[test]
fn parity_box_decomposition_utils() {
    let fixture: BoxUtilsFixture = load("box_utils.json");
    for c in &fixture.local_upper_bounds {
        let m = c.m;
        let mut lub = LocalUpperBounds::new(m, vals(&c.u0), vals(&c.z0)).unwrap();
        let initial = lub.clone();
        let mut points = Vec::new();
        for (i, s) in c.steps.iter().enumerate() {
            let z = vals(&s.z);
            lub = compute_local_upper_bounds(&lub, &z).unwrap();
            assert_eq!(lub.len(), s.len, "m={m} step {i}");
            assert!(same(lub.upper_bounds(), &vals(&s.u)), "m={m} step {i}: U");
            assert!(
                same(lub.defining_points(), &vals(&s.z_points)),
                "m={m} step {i}: Z"
            );
            points.extend(z);
        }
        let new_pareto = PointBatch::from_rows(c.steps.len(), m, points).unwrap();
        let inc = update_local_upper_bounds_incremental(&new_pareto, &initial).unwrap();
        assert!(
            same(inc.upper_bounds(), &vals(&c.incremental_u)),
            "m={m}: incremental U"
        );
        assert!(
            same(inc.defining_points(), &vals(&c.incremental_z)),
            "m={m}: incremental Z"
        );
        let bounds = get_partition_bounds(&lub, &vec![0.0; m]).unwrap();
        assert_eq!(bounds.shape(), c.partition_shape, "m={m}");
        assert!(
            same(&bounds.to_flat(), &vals(&c.partition_bounds)),
            "m={m}: bounds"
        );
    }
    for c in &fixture.cells_2d {
        let y = batch(&vals(&c.y), &c.y_shape);
        let r = vals(&c.ref_point);
        let nd = compute_non_dominated_hypercell_bounds_2d(&y, &r).unwrap();
        assert_eq!(nd.shape(), c.non_dominated_shape);
        assert!(same(&nd.to_flat(), &vals(&c.non_dominated)));
        let d = compute_dominated_hypercell_bounds_2d(&y, &r).unwrap();
        assert_eq!(d.shape(), c.dominated_shape);
        assert!(same(&d.to_flat(), &vals(&c.dominated)));
    }
    for c in &fixture.pad {
        let y = batch(&vals(&c.y), &c.y_shape);
        let out =
            pad_batch_pareto_frontier(&y, &vals(&c.ref_point), c.is_pareto, c.mask.as_deref())
                .unwrap();
        assert_eq!(out.shape(), c.out_shape);
        assert!(same(out.values(), &vals(&c.out)));
    }
    let e = &fixture.expand;
    let out = expand_ref_point(&vals(&e.ref_point), &[2], &e.batch_shape).unwrap();
    assert!(same(&out, &vals(&e.out)));
}

#[derive(Deserialize)]
struct ScalarizationCase {
    name: String,
    weights: Vec<f64>,
    y: Vec<F>,
    n: usize,
    alpha: f64,
    x: Vec<F>,
    values: Vec<F>,
    grads: Vec<F>,
}

#[derive(Deserialize)]
struct ScalarizationFixture {
    cases: Vec<ScalarizationCase>,
}

#[test]
fn parity_chebyshev_scalarization() {
    let fixture: ScalarizationFixture = load("scalarization.json");
    let mut r = Report::default();
    for c in &fixture.cases {
        let m = c.weights.len();
        let y = Matrix::from_row_slice(c.n, m, &vals(&c.y));
        let s = get_chebyshev_scalarization(&c.weights, &y, c.alpha).unwrap();
        let x = vals(&c.x);
        let values = s.evaluate_rows(&x);
        r.check(&c.name, "values", err(&values, &vals(&c.values)), VALUE_TOL);
        let mut grads = vec![0.0; x.len()];
        for (row, g) in x.chunks_exact(m).zip(grads.chunks_exact_mut(m)) {
            s.evaluate_with_grad(row, g);
        }
        r.check_slices(&c.name, "gradients", &grads, &vals(&c.grads), VALUE_TOL);
    }
    r.finish("Chebyshev scalarization");
}

#[derive(Deserialize)]
struct SimplexCase {
    d: usize,
    n: usize,
    u: Vec<F>,
    out: Vec<F>,
}

#[derive(Deserialize)]
struct SphereCase {
    d: usize,
    n: usize,
    z: Vec<F>,
    out: Vec<F>,
}

#[derive(Deserialize)]
struct SamplingFixture {
    simplex: Vec<SimplexCase>,
    hypersphere: Vec<SphereCase>,
}

#[test]
fn parity_sampling_transformations() {
    let fixture: SamplingFixture = load("sampling.json");
    let mut r = Report::default();
    for c in &fixture.simplex {
        let u = Matrix::from_row_slice(c.n, c.d - 1, &vals(&c.u));
        let out = simplex_from_uniform(&u);
        let flat: Vec<f64> = common::matrix_rows(&out);
        r.check(
            &format!("d={}", c.d),
            "sample_simplex",
            err(&flat, &vals(&c.out)),
            0.0,
        );
    }
    for c in &fixture.hypersphere {
        let z = Matrix::from_row_slice(c.n, c.d, &vals(&c.z));
        let out = hypersphere_from_normal(&z);
        let flat: Vec<f64> = common::matrix_rows(&out);
        // torch.linalg.norm sums the squares in a different order.
        r.check(
            &format!("d={}", c.d),
            "sample_hypersphere",
            err(&flat, &vals(&c.out)),
            1e-15,
        );
    }
    r.finish("sampling transformations");
}
