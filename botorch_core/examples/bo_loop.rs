//! Closed-loop Bayesian optimization of the Hartmann-6 function
//! (minimum -3.32237 at x* = (0.20169, 0.150011, 0.476874, 0.275332,
//! 0.311652, 0.6573)), following BoTorch's "closed loop" tutorial:
//!
//! 1. evaluate a scrambled Sobol design of `n_init` points;
//! 2. repeat: fit a `SingleTaskGP` (Standardize, BoTorch's default priors)
//!    by maximizing the marginal likelihood, optimize an acquisition function
//!    with `optimize_acqf` (10 restarts, 512 raw samples) and evaluate the
//!    candidates.
//!
//! The problem is negated (`negate=True`) so that BO maximizes `-Hartmann`.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --example bo_loop -- [logei|qlogei|qlognei] [iterations] [seed] [q]
//! ```
//!
//! Defaults: `logei`, 30 iterations, seed 0, `q = 1` (`q = 2` for the batch
//! acquisition functions if not given).

use std::sync::Arc;
use std::time::Instant;

use botorch_core::acquisition::{
    AcquisitionFunction, LogExpectedImprovement, QLogExpectedImprovement,
    QLogExpectedImprovementOptions, QLogNoisyExpectedImprovement,
    QLogNoisyExpectedImprovementOptions,
};
use botorch_core::optim::{optimize_acqf, OptimizeAcqfOptions};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::test_functions::SyntheticTestFunction;
use botorch_core::utils::sampling::draw_sobol_samples;
use botorch_core::{
    fit_gpytorch_mll, BotorchError, FitOptions, Matrix, Model, SingleTaskGP, SingleTaskGPOptions,
};

fn vstack(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.nrows() + b.nrows(), a.ncols());
    out.rows_mut(0, a.nrows()).copy_from(a);
    out.rows_mut(a.nrows(), b.nrows()).copy_from(b);
    out
}

fn main() -> Result<(), BotorchError> {
    let args: Vec<String> = std::env::args().collect();
    let acqf_name = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("logei")
        .to_string();
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let default_q = if acqf_name == "logei" { 1 } else { 2 };
    let q: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_q);
    if acqf_name == "logei" && q != 1 {
        return Err(BotorchError::InvalidArgument(
            "analytic LogEI requires q = 1".into(),
        ));
    }

    let problem = SyntheticTestFunction::hartmann(6)?.with_negate(true);
    let bounds = problem.bounds().clone();
    let d = problem.dim();
    let n_init = 2 * (d + 1);

    let init = draw_sobol_samples(&bounds, n_init, 1, Some(seed))?;
    let mut train_x = Matrix::from_fn(n_init, d, |i, j| init[i][(0, j)]);
    let mut train_y =
        Matrix::from_column_slice(n_init, 1, problem.forward(&train_x, false)?.as_slice());
    println!(
        "Hartmann-6 (negated), {acqf_name}, q = {q}, {n_init} Sobol points + {iterations} \
         iterations, seed {seed}; optimum -3.32237"
    );
    println!(
        "{:>4} {:>6} {:>14} {:>14} {:>9} {:>9} {:>9}",
        "iter", "n", "best f(x)", "acq value", "fit [ms]", "acq [ms]", "total [s]"
    );
    let start = Instant::now();
    for it in 0..iterations {
        let t0 = Instant::now();
        let mut gp = SingleTaskGP::new(
            train_x.clone(),
            train_y.clone(),
            SingleTaskGPOptions::default(),
        )?;
        fit_gpytorch_mll(&mut gp, FitOptions::default())?;
        let fit_ms = t0.elapsed().as_secs_f64() * 1e3;
        let model: Arc<dyn Model> = Arc::new(gp);
        let best_f = train_y.max();
        let sampler = || {
            Some(Arc::new(
                SobolQMCNormalSampler::new(512, Some(seed + it as u64)).unwrap(),
            ))
        };
        let acqf: Box<dyn AcquisitionFunction> = match acqf_name.as_str() {
            "logei" => Box::new(LogExpectedImprovement::new(model, best_f, true)?),
            "qlogei" => Box::new(QLogExpectedImprovement::new(
                model,
                best_f,
                QLogExpectedImprovementOptions {
                    sampler: sampler(),
                    ..Default::default()
                },
            )?),
            "qlognei" => Box::new(QLogNoisyExpectedImprovement::new(
                model,
                &train_x,
                QLogNoisyExpectedImprovementOptions {
                    sampler: sampler(),
                    ..Default::default()
                },
            )?),
            other => {
                return Err(BotorchError::InvalidArgument(format!(
                    "unknown acquisition function {other} (logei, qlogei, qlognei)"
                )))
            }
        };
        let t1 = Instant::now();
        let options = OptimizeAcqfOptions {
            seed: Some(seed.wrapping_mul(1000) + it as u64),
            ..Default::default()
        };
        let (candidates, acq_value) = optimize_acqf(acqf.as_ref(), &bounds, q, 10, 512, &options)?;
        let acq_ms = t1.elapsed().as_secs_f64() * 1e3;
        let y_new = problem.forward(&candidates, false)?;
        train_x = vstack(&train_x, &candidates);
        train_y = vstack(&train_y, &Matrix::from_column_slice(q, 1, y_new.as_slice()));
        // Report the best Hartmann value found (the negation of the best observation).
        println!(
            "{:>4} {:>6} {:>14.6} {:>14.6} {:>9.1} {:>9.1} {:>9.2}",
            it + 1,
            train_x.nrows(),
            -train_y.max(),
            acq_value,
            fit_ms,
            acq_ms,
            start.elapsed().as_secs_f64()
        );
    }
    let best = train_y.column(0).argmax().0;
    println!(
        "best value {:.6} (optimum -3.32237) at x = {:.4}",
        -train_y[(best, 0)],
        train_x.row(best)
    );
    Ok(())
}
