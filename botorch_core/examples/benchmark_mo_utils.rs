//! Native timings of the multi-objective utilities (the Rust side of
//! `scripts/benchmark_mo_utils.py`, without the Python bindings):
//!
//! * `Hypervolume::compute` and the three box decompositions (construction
//!   plus hypervolume) for m = 2, 3, 4 objectives and n = 10, 50, 100, 200
//!   points near the positive unit sphere (mostly Pareto optimal);
//! * the decompositions of 128 Monte-Carlo samples of 20 points, as qNEHVI
//!   builds them: one batched `FastNondominatedPartitioning` (m = 2) and a
//!   `BoxDecompositionList` built with `par_build` (m = 3), on one thread
//!   and on all threads.
//!
//! The single-threaded tasks run on the main thread; the batched tasks run
//! in rayon pools of one thread and of all threads (on CPUs with
//! performance and efficiency cores, a pool's worker can be slower than the
//! main thread).
//!
//! ```sh
//! cargo run --release --example benchmark_mo_utils -- [repeats]
//! ```

use std::time::Instant;

use botorch_core::utils::multi_objective::box_decompositions::{
    BoxDecomposition, BoxDecompositionList, DominatedPartitioning, FastNondominatedPartitioning,
    NondominatedPartitioning,
};
use botorch_core::utils::multi_objective::hypervolume::Hypervolume;
use botorch_core::utils::multi_objective::pareto::is_non_dominated;
use botorch_core::{BotorchError, Matrix};

/// Points near the positive unit sphere (a deterministic generator).
fn front(n: usize, m: usize, seed: u64) -> Matrix {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut y = Matrix::zeros(n, m);
    for i in 0..n {
        let row: Vec<f64> = (0..m).map(|_| uniform() + 1e-3).collect();
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        let s = (1.0 - 0.05 * uniform()) / norm;
        for j in 0..m {
            y[(i, j)] = row[j] * s;
        }
    }
    y
}

/// Median seconds per call over `repeats` runs of at least 0.1 s each.
fn median_time(repeats: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut times: Vec<f64> = (0..repeats)
        .map(|_| {
            let (mut calls, start) = (0u32, Instant::now());
            loop {
                f();
                calls += 1;
                let elapsed = start.elapsed().as_secs_f64();
                if elapsed >= 0.1 {
                    return elapsed / f64::from(calls);
                }
            }
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn fmt(seconds: f64) -> String {
    if seconds >= 1.0 {
        format!("{seconds:.2} s")
    } else if seconds >= 1e-3 {
        format!("{:.2} ms", seconds * 1e3)
    } else {
        format!("{:.1} µs", seconds * 1e6)
    }
}

fn main() -> Result<(), BotorchError> {
    let repeats: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    println!("| Task | m | n | time |");
    println!("|---|---|---|---|");
    {
        for m in [2usize, 3, 4] {
            let r = vec![0.0; m];
            for n in [10usize, 50, 100, 200] {
                let y = front(n, m, (n * m) as u64);
                let mask = is_non_dominated(&y, true, true);
                let rows: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
                let pareto = y.select_rows(&rows);
                let hv = Hypervolume::new(&r);
                let t = median_time(repeats, || {
                    hv.compute(&pareto).unwrap();
                });
                println!("| Hypervolume::compute | {m} | {n} | {} |", fmt(t));
                if !(m == 4 && n > 50) {
                    let t = median_time(repeats, || {
                        let d = NondominatedPartitioning::new(&r, Some(&y), 0.0).unwrap();
                        d.compute_hypervolume();
                    });
                    println!("| NondominatedPartitioning | {m} | {n} | {} |", fmt(t));
                }
                let t = median_time(repeats, || {
                    let d = FastNondominatedPartitioning::new(&r, Some(&y)).unwrap();
                    d.compute_hypervolume();
                });
                println!("| FastNondominatedPartitioning | {m} | {n} | {} |", fmt(t));
                let t = median_time(repeats, || {
                    let d = DominatedPartitioning::new(&r, Some(&y)).unwrap();
                    d.compute_hypervolume();
                });
                println!("| DominatedPartitioning | {m} | {n} | {} |", fmt(t));
            }
        }
    }
    let samples2: Vec<Matrix> = (0..128).map(|k| front(20, 2, 1000 + k)).collect();
    let samples3: Vec<Matrix> = (0..128).map(|k| front(20, 3, 2000 + k)).collect();
    let all = rayon::current_num_threads();
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("thread pool");
    for (threads, pool) in [
        (1, single),
        (
            all,
            rayon::ThreadPoolBuilder::new()
                .num_threads(all)
                .build()
                .expect("thread pool"),
        ),
    ] {
        pool.install(|| {
            let t = median_time(repeats, || {
                let d = FastNondominatedPartitioning::new_batched(&[0.0; 2], &samples2).unwrap();
                d.get_hypercell_bounds().unwrap();
            });
            println!(
                "| batched FastNondominatedPartitioning, 128 samples ({threads} threads) | 2 | 20 | {} |",
                fmt(t)
            );
            let t = median_time(repeats, || {
                let list = BoxDecompositionList::par_build(&samples3, |y| {
                    FastNondominatedPartitioning::new(&[0.0; 3], Some(y))
                })
                .unwrap();
                list.get_hypercell_bounds().unwrap();
            });
            println!(
                "| BoxDecompositionList::par_build, 128 samples ({threads} threads) | 3 | 20 | {} |",
                fmt(t)
            );
        });
    }
    Ok(())
}
