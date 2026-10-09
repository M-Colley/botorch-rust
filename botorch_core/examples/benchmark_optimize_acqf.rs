//! Timing of `optimize_acqf` on the problem of
//! `scripts/benchmark_optimize_acqf.py` (an identical model in BoTorch):
//! 50 training inputs from a Weyl sequence in `[0, 1]^6`, targets
//! `-Hartmann6(x)`, a default `SingleTaskGP` with fixed hyperparameters
//! (noise 1e-3, constant 0, lengthscales 0.5).
//!
//! * `optimize_acqf(LogEI, q = 1, num_restarts = 10, raw_samples = 512)`
//! * `optimize_acqf(qLogEI with 512 Sobol samples, q = 3, num_restarts = 10,
//!   raw_samples = 512)`
//!
//! each on one thread and on all threads (rayon's global pool).
//!
//! ```sh
//! cargo run --release --example benchmark_optimize_acqf -- [repeats]
//! ```

use std::sync::Arc;
use std::time::Instant;

use botorch_core::acquisition::{
    AcquisitionFunction, LogExpectedImprovement, QLogExpectedImprovement,
    QLogExpectedImprovementOptions,
};
use botorch_core::optim::{optimize_acqf, OptimizeAcqfOptions};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::test_functions::SyntheticTestFunction;
use botorch_core::{BotorchError, Matrix, Model, SingleTaskGP, SingleTaskGPOptions};

/// The same Weyl sequence as the Python script.
fn weyl(n: usize, d: usize) -> Matrix {
    let primes = [2.0f64, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0];
    Matrix::from_fn(n, d, |i, j| {
        let alpha = primes[j].sqrt() % 1.0;
        (0.5 + (i + 1) as f64 * alpha).rem_euclid(1.0)
    })
}

fn bench(
    name: &str,
    make: &dyn Fn() -> Box<dyn AcquisitionFunction>,
    q: usize,
    repeats: usize,
    threads: usize,
) -> Result<(), BotorchError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| BotorchError::InvalidArgument(e.to_string()))?;
    let bounds = Matrix::from_fn(2, 6, |i, _| i as f64);
    let mut times = Vec::new();
    let mut best = f64::NEG_INFINITY;
    for s in 0..repeats {
        let acqf = make();
        let options = OptimizeAcqfOptions {
            seed: Some(s as u64),
            ..Default::default()
        };
        let t = Instant::now();
        let (_, v) =
            pool.install(|| optimize_acqf(acqf.as_ref(), &bounds, q, 10, 512, &options))?;
        times.push(t.elapsed().as_secs_f64());
        best = best.max(v);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    println!(
        "{name:28} threads={threads:2}: median {:8.4} s (min {:.4}, max {:.4}); best value {best:.6}",
        times[times.len() / 2],
        times[0],
        times[times.len() - 1]
    );
    Ok(())
}

fn main() -> Result<(), BotorchError> {
    let repeats: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let x = weyl(50, 6);
    let problem = SyntheticTestFunction::hartmann(6)?.with_negate(true);
    let y = Matrix::from_column_slice(50, 1, problem.forward(&x, false)?.as_slice());
    let mut gp = SingleTaskGP::new(x, y.clone(), SingleTaskGPOptions::default())?;
    gp.set_raw_parameters(&[1e-3, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])?;
    let model: Arc<dyn Model> = Arc::new(gp);
    let best_f = y.max();
    let all = std::thread::available_parallelism().map_or(1, |n| n.get());
    for threads in [1, all] {
        let m = model.clone();
        bench(
            "LogEI q=1",
            &move || Box::new(LogExpectedImprovement::new(m.clone(), best_f, true).unwrap()),
            1,
            repeats,
            threads,
        )?;
        let m = model.clone();
        bench(
            "qLogEI q=3 (512 samples)",
            &move || {
                Box::new(
                    QLogExpectedImprovement::new(
                        m.clone(),
                        best_f,
                        QLogExpectedImprovementOptions {
                            sampler: Some(Arc::new(
                                SobolQMCNormalSampler::new(512, Some(0)).unwrap(),
                            )),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                )
            },
            3,
            repeats,
            threads,
        )?;
    }
    Ok(())
}
