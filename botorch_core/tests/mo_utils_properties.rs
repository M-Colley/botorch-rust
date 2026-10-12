//! Properties of the multi-objective utilities that do not depend on
//! BoTorch: the hypervolume of the dimension sweep, of all three box
//! decompositions, of an exact grid computation and of a Monte-Carlo
//! estimate agree; the dominated and non-dominated cells tile the box
//! between the reference point and the ideal point (every point lies in
//! exactly one cell of the right kind); batched decompositions equal the
//! per-element ones; incremental updates equal fresh decompositions;
//! `BoxDecompositionList` pads and updates correctly in parallel; the
//! simplex and hypersphere samplers have the right distributions; and the
//! types are thread-safe.

use botorch_core::utils::multi_objective::box_decompositions::{
    BoxDecomposition, BoxDecompositionList, CellBounds, DominatedPartitioning,
    FastNondominatedPartitioning, NondominatedPartitioning,
};
use botorch_core::utils::multi_objective::hypervolume::Hypervolume;
use botorch_core::utils::multi_objective::pareto::is_non_dominated;
use botorch_core::utils::multi_objective::scalarization::ChebyshevScalarization;
use botorch_core::utils::multi_objective::PointBatch;
use botorch_core::utils::sampling::{sample_hypersphere, sample_simplex};
use botorch_core::Matrix;

/// A small deterministic generator (SplitMix64).
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Points near the positive unit sphere (large Pareto fronts), on a coarse
/// grid (ties) every third set.
fn points(rng: &mut Rng, n: usize, m: usize, grid: bool) -> Matrix {
    let mut y = Matrix::zeros(n, m);
    for i in 0..n {
        let mut norm = 0.0;
        for j in 0..m {
            y[(i, j)] = rng.uniform() + 1e-3;
            norm += y[(i, j)] * y[(i, j)];
        }
        let s = (1.0 - 0.2 * rng.uniform()) / norm.sqrt();
        for j in 0..m {
            y[(i, j)] = if grid {
                (4.0 * y[(i, j)] * s).round() / 4.0
            } else {
                y[(i, j)] * s
            };
        }
    }
    y
}

/// The exact hypervolume by coordinate compression: the dominated region is
/// a union of grid cells spanned by the sorted coordinates.
fn grid_hypervolume(y: &Matrix, r: &[f64]) -> f64 {
    let (n, m) = y.shape();
    let pts: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| y[(i, j)]).collect())
        .filter(|p: &Vec<f64>| p.iter().zip(r).all(|(v, r)| v >= r))
        .collect();
    let coords: Vec<Vec<f64>> = (0..m)
        .map(|j| {
            let mut c: Vec<f64> = pts.iter().map(|p| p[j]).chain([r[j]]).collect();
            c.sort_by(|a, b| a.partial_cmp(b).unwrap());
            c.dedup();
            c
        })
        .collect();
    let dims: Vec<usize> = coords.iter().map(|c| c.len() - 1).collect();
    let total: usize = dims.iter().product();
    let mut hv = 0.0;
    let mut idx = vec![0usize; m];
    for _ in 0..total {
        // The cell is dominated iff its upper corner is weakly dominated.
        let upper: Vec<f64> = (0..m).map(|j| coords[j][idx[j] + 1]).collect();
        if pts
            .iter()
            .any(|p| p.iter().zip(&upper).all(|(v, u)| v >= u))
        {
            hv += (0..m)
                .map(|j| coords[j][idx[j] + 1] - coords[j][idx[j]])
                .product::<f64>();
        }
        for k in 0..m {
            idx[k] += 1;
            if idx[k] < dims[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    hv
}

fn decompositions(r: &[f64], y: &Matrix) -> Vec<Box<dyn BoxDecomposition>> {
    vec![
        Box::new(NondominatedPartitioning::new(r, Some(y), 0.0).unwrap()),
        Box::new(FastNondominatedPartitioning::new(r, Some(y)).unwrap()),
        Box::new(DominatedPartitioning::new(r, Some(y)).unwrap()),
    ]
}

#[test]
fn hypervolumes_agree() {
    let mut rng = Rng(1);
    let mut worst = 0.0f64;
    for m in 2..=5 {
        for (k, n) in [1usize, 4, 9, 16].into_iter().enumerate() {
            if m == 5 && n > 9 {
                continue;
            }
            let y = points(&mut rng, n, m, k % 3 == 2);
            let r = vec![0.05; m];
            let exact = grid_hypervolume(&y, &r);
            let hso = Hypervolume::new(&r).compute(&y).unwrap();
            let mut values = vec![hso];
            for d in decompositions(&r, &y) {
                values.push(d.compute_hypervolume()[0]);
            }
            for v in values {
                let e = (v - exact).abs() / exact.max(1e-300);
                worst = worst.max(e);
                assert!(e < 1e-12, "m={m} n={n}: {v} vs {exact}");
            }
        }
    }
    println!("worst relative error vs the grid hypervolume: {worst:.2e}");
}

#[test]
fn monte_carlo_hypervolume() {
    let mut rng = Rng(2);
    for m in [2usize, 3, 4] {
        let y = points(&mut rng, 25, m, false);
        let r = vec![0.0; m];
        let hv = Hypervolume::new(&r).compute(&y).unwrap();
        // Estimate within the unit box (all points are in [0, 1]^m).
        let draws = 200_000;
        let mut hits = 0usize;
        let mut x = vec![0.0; m];
        for _ in 0..draws {
            for v in x.iter_mut() {
                *v = rng.uniform();
            }
            if (0..25).any(|i| (0..m).all(|j| y[(i, j)] >= x[j])) {
                hits += 1;
            }
        }
        let p = hits as f64 / draws as f64;
        let se = (p * (1.0 - p) / draws as f64).sqrt();
        assert!((p - hv).abs() < 5.0 * se, "m={m}: MC {p} +- {se} vs {hv}");
    }
}

/// Whether `x` lies in the half-open cell `[l, u)`.
fn in_cell(x: &[f64], l: &[f64], u: &[f64]) -> bool {
    x.iter()
        .zip(l)
        .zip(u)
        .all(|((x, l), u)| *l <= *x && *x < *u)
}

fn count_cells(cells: &CellBounds, x: &[f64]) -> usize {
    (0..cells.num_cells())
        .filter(|&k| {
            let (l, u) = cells.cell(0, k);
            in_cell(x, l, u)
        })
        .count()
}

#[test]
fn cells_tile_the_box() {
    let mut rng = Rng(3);
    for m in 2..=4 {
        for (n, grid) in [(6usize, false), (12, true), (20, false)] {
            let y = points(&mut rng, n, m, grid);
            let r = vec![0.1; m];
            let ds = decompositions(&r, &y);
            let ideal: Vec<f64> = (0..m)
                .map(|j| (0..n).map(|i| y[(i, j)]).fold(f64::NEG_INFINITY, f64::max))
                .collect();
            // Random points in the box between the reference point and a bit
            // beyond the ideal point.
            for _ in 0..3000 {
                let x: Vec<f64> = (0..m)
                    .map(|j| r[j] + (ideal[j] + 0.1 - r[j]) * rng.uniform())
                    .collect();
                // Strictly dominated by a Pareto point (boundaries have measure 0).
                let dominated = (0..n).any(|i| (0..m).all(|j| y[(i, j)] > x[j]));
                let nd_exact = count_cells(ds[0].get_hypercell_bounds().unwrap(), &x);
                let nd_fast = count_cells(ds[1].get_hypercell_bounds().unwrap(), &x);
                let dom = count_cells(ds[2].get_hypercell_bounds().unwrap(), &x);
                assert_eq!(nd_exact, usize::from(!dominated), "m={m} n={n} {x:?}");
                assert_eq!(nd_fast, usize::from(!dominated), "m={m} n={n} {x:?}");
                assert_eq!(dom, usize::from(dominated), "m={m} n={n} {x:?}");
            }
            // The dominated cells and the non-dominated cells clipped to the
            // box have the volume of the box.
            let clip = |c: &CellBounds| -> f64 {
                (0..c.num_cells())
                    .map(|k| {
                        let (l, u) = c.cell(0, k);
                        (0..m)
                            .map(|j| (u[j].min(ideal[j]) - l[j].max(r[j])).max(0.0))
                            .product::<f64>()
                    })
                    .sum()
            };
            let box_volume: f64 = (0..m).map(|j| ideal[j] - r[j]).product();
            let dominated = ds[2].get_hypercell_bounds().unwrap().volumes()[0];
            for nd in &ds[..2] {
                let total = dominated + clip(nd.get_hypercell_bounds().unwrap());
                assert!(
                    ((total - box_volume) / box_volume).abs() < 1e-12,
                    "m={m} n={n}"
                );
            }
        }
    }
}

/// The cells of a batch element without BoTorch's padding (empty cells),
/// sorted.
fn cell_set(c: &CellBounds, b: usize) -> Vec<Vec<u64>> {
    let mut out: Vec<Vec<u64>> = (0..c.num_cells())
        .filter_map(|k| {
            let start = (b * c.num_cells() + k) * c.num_outcomes();
            let (l, u) = (
                &c.lower()[start..start + c.num_outcomes()],
                &c.upper()[start..start + c.num_outcomes()],
            );
            if l.iter().zip(u).any(|(l, u)| l == u) {
                return None;
            }
            Some(l.iter().chain(u).map(|v| v.to_bits()).collect())
        })
        .collect();
    out.sort();
    out
}

#[test]
fn batched_decompositions_equal_their_elements() {
    let mut rng = Rng(4);
    let mut ys: Vec<Matrix> = (0..6).map(|k| points(&mut rng, 15, 2, k == 2)).collect();
    ys[3] = Matrix::from_element(15, 2, -1.0); // no point better than the reference point
    ys[4] = ys[4].map(|v| 0.5 * v);
    let r = [0.0, 0.0];
    let batched: Vec<Box<dyn BoxDecomposition>> = vec![
        Box::new(NondominatedPartitioning::new_batched(&r, &ys, 0.0).unwrap()),
        Box::new(FastNondominatedPartitioning::new_batched(&r, &ys).unwrap()),
        Box::new(DominatedPartitioning::new_batched(&r, &ys).unwrap()),
    ];
    for (k, y) in ys.iter().enumerate() {
        for (d, single) in batched.iter().zip(decompositions(&r, y)) {
            let hv = d.compute_hypervolume()[k];
            let expected = single.compute_hypervolume()[0];
            assert!(
                (hv - expected).abs() <= 1e-14 * expected.max(1.0),
                "{}",
                d.class_name()
            );
            assert_eq!(
                cell_set(d.get_hypercell_bounds().unwrap(), k),
                cell_set(single.get_hypercell_bounds().unwrap(), 0),
                "{} element {k}",
                d.class_name()
            );
        }
    }
    assert!(FastNondominatedPartitioning::new_batched(&[0.0; 3], &[Matrix::zeros(2, 3)]).is_err());
}

#[test]
fn incremental_updates_equal_fresh_decompositions() {
    let mut rng = Rng(5);
    for m in [3usize, 4] {
        let y = points(&mut rng, 20, m, false);
        let r = vec![0.1; m];
        let mut fast = FastNondominatedPartitioning::new(&r, Some(&y.rows(0, 8).into())).unwrap();
        let mut dom = DominatedPartitioning::new(&r, Some(&y.rows(0, 8).into())).unwrap();
        for chunk in [(8, 5), (13, 7)] {
            let upd: Matrix = y.rows(chunk.0, chunk.1).into();
            fast.update_matrix(&upd).unwrap();
            dom.update_matrix(&upd).unwrap();
        }
        let fresh_fast = FastNondominatedPartitioning::new(&r, Some(&y)).unwrap();
        let fresh_dom = DominatedPartitioning::new(&r, Some(&y)).unwrap();
        let hv = Hypervolume::new(&r).compute(&y).unwrap();
        for d in [
            &fast as &dyn BoxDecomposition,
            &dom,
            &fresh_fast,
            &fresh_dom,
        ] {
            assert!((d.compute_hypervolume()[0] - hv).abs() < 1e-13 * hv);
        }
        assert_eq!(
            cell_set(fast.get_hypercell_bounds().unwrap(), 0).len(),
            cell_set(fresh_fast.get_hypercell_bounds().unwrap(), 0).len()
        );
        assert_eq!(
            cell_set(dom.get_hypercell_bounds().unwrap(), 0),
            cell_set(fresh_dom.get_hypercell_bounds().unwrap(), 0)
        );
        // Updates without new Pareto points or without points better than
        // the reference point change nothing.
        let before = fast.get_hypercell_bounds().unwrap().clone();
        fast.update_matrix(&Matrix::from_element(2, m, 0.05))
            .unwrap();
        fast.update_matrix(&Matrix::from_element(1, m, 0.11))
            .unwrap();
        assert_eq!(fast.get_hypercell_bounds().unwrap(), &before);
        assert_eq!(fast.y().unwrap().n(), 23);
    }
}

#[test]
fn box_decomposition_lists() {
    let mut rng = Rng(6);
    let ys: Vec<Matrix> = (0..12).map(|k| points(&mut rng, 6 + k, 3, false)).collect();
    let r = [0.1, 0.1, 0.1];
    let mut list =
        BoxDecompositionList::par_build(&ys, |y| FastNondominatedPartitioning::new(&r, Some(y)))
            .unwrap();
    assert_eq!(list.len(), 12);
    let cells = list.get_hypercell_bounds().unwrap();
    assert_eq!(cells.batch_shape(), &[12]);
    for (k, d) in list.decompositions().iter().enumerate() {
        let own = d.get_hypercell_bounds().unwrap();
        let len = own.num_cells() * 3;
        assert_eq!(&cells.lower_of(k)[..len], own.lower());
        assert!(cells.lower_of(k)[len..].iter().all(|&v| v == 0.0));
        assert!(cells.upper_of(k)[len..].iter().all(|&v| v == 0.0));
    }
    let hvs = list.compute_hypervolume();
    for (y, hv) in ys.iter().zip(&hvs) {
        let expected = Hypervolume::new(&r).compute(y).unwrap();
        assert!((hv[0] - expected).abs() < 1e-13 * expected);
    }
    let updates: Vec<PointBatch> = (0..12)
        .map(|k| PointBatch::from_matrix(&points(&mut rng, 3 + k % 2, 3, false)))
        .collect();
    list.update(&updates).unwrap();
    for (k, d) in list.decompositions().iter().enumerate() {
        let mut all = ys[k]
            .clone()
            .resize_vertically(ys[k].nrows() + updates[k].n(), 0.0);
        for i in 0..updates[k].n() {
            for j in 0..3 {
                all[(ys[k].nrows() + i, j)] = updates[k].point(0, i)[j];
            }
        }
        let expected = Hypervolume::new(&r).compute(&all).unwrap();
        assert!((d.compute_hypervolume()[0] - expected).abs() < 1e-13 * expected);
    }
    assert!(list.update(&updates[..3]).is_err());
}

#[test]
fn pareto_sets_have_the_dominance_properties() {
    let mut rng = Rng(7);
    let y = points(&mut rng, 200, 3, true);
    let mask = is_non_dominated(&y, true, true);
    let kept: Vec<usize> = (0..200).filter(|&i| mask[i]).collect();
    for &i in &kept {
        for &k in &kept {
            if i != k {
                assert!(
                    (0..3).any(|j| y[(i, j)] != y[(k, j)]),
                    "duplicates are removed"
                );
            }
        }
    }
    for i in (0..200).filter(|&i| !mask[i]) {
        // Every removed point is weakly dominated by a kept point.
        assert!(kept.iter().any(|&k| (0..3).all(|j| y[(k, j)] >= y[(i, j)])));
    }
}

#[test]
fn sampler_distributions() {
    for qmc in [false, true] {
        let d = 4;
        let x = sample_simplex(d, 4096, qmc, Some(11)).unwrap();
        for i in 0..x.nrows() {
            assert!((x.row(i).sum() - 1.0).abs() < 1e-12 && x.row(i).min() >= 0.0);
        }
        // Dirichlet(1, ..., 1): mean 1/d, variance (d - 1) / (d²(d + 1)).
        for j in 0..d {
            let col = x.column(j);
            let mean = col.mean();
            let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 4095.0;
            assert!((mean - 0.25).abs() < 0.01, "qmc={qmc} mean {mean}");
            assert!((var - 3.0 / 80.0).abs() < 0.004, "qmc={qmc} var {var}");
        }
        let s = sample_hypersphere(3, 4096, qmc, Some(12)).unwrap();
        for i in 0..s.nrows() {
            assert!((s.row(i).norm() - 1.0).abs() < 1e-12);
        }
        // Uniform on the sphere: mean 0, covariance I / d.
        for j in 0..3 {
            let col = s.column(j);
            assert!(col.mean().abs() < 0.03, "qmc={qmc}");
            let second = col.iter().map(|v| v * v).sum::<f64>() / 4096.0;
            assert!((second - 1.0 / 3.0).abs() < 0.02, "qmc={qmc}");
        }
    }
    assert_eq!(
        sample_simplex(1, 3, false, Some(0)).unwrap(),
        Matrix::from_element(3, 1, 1.0)
    );
    assert!(sample_hypersphere(1, 64, true, Some(0))
        .unwrap()
        .iter()
        .all(|v| v.abs() == 1.0));
    assert!(sample_simplex(0, 3, false, None).is_err());
    assert_eq!(
        sample_simplex(3, 5, true, Some(4)).unwrap(),
        sample_simplex(3, 5, true, Some(4)).unwrap()
    );
}

#[test]
fn types_are_thread_safe() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NondominatedPartitioning>();
    assert_send_sync::<FastNondominatedPartitioning>();
    assert_send_sync::<DominatedPartitioning>();
    assert_send_sync::<BoxDecompositionList>();
    assert_send_sync::<Hypervolume>();
    assert_send_sync::<ChebyshevScalarization>();
    assert_send_sync::<CellBounds>();
    assert_send_sync::<PointBatch>();
    // Concurrent use of a shared decomposition gives identical results.
    let mut rng = Rng(8);
    let y = points(&mut rng, 30, 3, false);
    let d = FastNondominatedPartitioning::new(&[0.0; 3], Some(&y)).unwrap();
    let hv = d.compute_hypervolume();
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| assert_eq!(d.compute_hypervolume(), hv));
        }
    });
}
