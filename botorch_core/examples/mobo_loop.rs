//! Closed-loop multi-objective Bayesian optimization, following BoTorch's
//! "Multi-Objective BO with qNEHVI / qLogNEHVI" tutorial:
//!
//! 1. evaluate a scrambled Sobol design of `2 (d + 1)` points;
//! 2. repeat: fit a multi-output `SingleTaskGP` (or a `ModelListGP` of
//!    single-output models) with `Normalize` (the problem's bounds) and
//!    `Standardize` by maximizing the marginal likelihood, build the
//!    acquisition function (128 Sobol samples, the problem's reference
//!    point, observed points as the baseline with `prune_baseline=True`),
//!    optimize it for `q` candidates (`sequential=True`, 10 restarts, 512
//!    raw samples) and evaluate them;
//! 3. report the hypervolume of the observed outcomes after every
//!    iteration (and its log distance to the maximum hypervolume).
//!
//! The problems are negated (`negate=True`) so that all objectives are
//! maximized: `branincurrin` (d = 2, m = 2) and `dtlz2` (d = 6, m = 2 or
//! `dtlz2-3`: m = 3).
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --example mobo_loop -- \
//!     [branincurrin|dtlz2|dtlz2-3] [qlognehvi|qnehvi|qlogehvi|qlognparego] \
//!     [iterations] [seed] [q] [multi|list] [noise_std]
//! ```
//!
//! Defaults: `branincurrin`, `qlognehvi`, 20 iterations, seed 0, `q = 2`,
//! a multi-output model, noise-free observations.

use std::sync::Arc;
use std::time::Instant;

use botorch_core::acquisition::multi_objective::{
    QLogExpectedHypervolumeImprovement, QLogExpectedHypervolumeImprovementOptions, QLogNParEGO,
    QLogNParEGOOptions, QLogNoisyExpectedHypervolumeImprovement,
    QLogNoisyExpectedHypervolumeImprovementOptions, QNoisyExpectedHypervolumeImprovement,
    QNoisyExpectedHypervolumeImprovementOptions,
};
use botorch_core::acquisition::AcquisitionFunction;
use botorch_core::optim::{optimize_acqf, OptimizeAcqfOptions};
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::test_functions::NoiseStd;
use botorch_core::utils::multi_objective::box_decompositions::{
    BoxDecomposition, DominatedPartitioning, FastNondominatedPartitioning,
};
use botorch_core::utils::sampling::draw_sobol_samples;
use botorch_core::{
    fit_gpytorch_mll, BatchedInputTransform, BatchedMultiOutputGP, BatchedMultiOutputGPOptions,
    BotorchError, FitOptions, Matrix, Model, ModelListGP, MultiObjectiveKind,
    MultiObjectiveTestProblem, Normalize,
};

fn vstack(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.nrows() + b.nrows(), a.ncols());
    out.rows_mut(0, a.nrows()).copy_from(a);
    out.rows_mut(a.nrows(), b.nrows()).copy_from(b);
    out
}

/// A GP over `y` (`n × k`) with `Normalize` to the problem's bounds and the
/// default `Standardize`, fitted by maximizing the marginal likelihood.
fn fit(
    x: &Matrix,
    y: Matrix,
    bounds: &Matrix,
    seed: u64,
) -> Result<BatchedMultiOutputGP, BotorchError> {
    let options = BatchedMultiOutputGPOptions {
        input_transform: Some(BatchedInputTransform::Shared(Normalize::with_bounds(
            bounds,
        )?)),
        ..Default::default()
    };
    let mut gp = BatchedMultiOutputGP::new(vec![x.clone()], vec![y], vec![], options)?;
    fit_gpytorch_mll(
        &mut gp,
        FitOptions {
            seed,
            ..Default::default()
        },
    )?;
    Ok(gp)
}

/// The hypervolume of the observed outcomes with respect to `ref_point`.
fn hypervolume(ref_point: &[f64], y: &Matrix) -> Result<f64, BotorchError> {
    Ok(DominatedPartitioning::new(ref_point, Some(y))?.compute_hypervolume()[0])
}

fn main() -> Result<(), BotorchError> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, default: &str| args.get(i).cloned().unwrap_or_else(|| default.into());
    let problem_name = arg(1, "branincurrin");
    let acqf_name = arg(2, "qlognehvi");
    let iterations: usize = arg(3, "20").parse().unwrap_or(20);
    let seed: u64 = arg(4, "0").parse().unwrap_or(0);
    let q: usize = arg(5, "2").parse().unwrap_or(2);
    let model_kind = arg(6, "multi");
    let noise_std: f64 = arg(7, "0").parse().unwrap_or(0.0);

    let (kind, dim, m) = match problem_name.as_str() {
        "branincurrin" => (MultiObjectiveKind::BraninCurrin, None, None),
        "dtlz2" => (MultiObjectiveKind::Dtlz2, Some(6), Some(2)),
        "dtlz2-3" => (MultiObjectiveKind::Dtlz2, Some(6), Some(3)),
        other => {
            return Err(BotorchError::InvalidArgument(format!(
                "unknown problem {other} (branincurrin, dtlz2, dtlz2-3)"
            )))
        }
    };
    let mut problem = MultiObjectiveTestProblem::new(kind, dim, m)?
        .with_negate(true)
        .with_seed(seed);
    if noise_std > 0.0 {
        problem = problem.with_noise_std(NoiseStd::Scalar(noise_std))?;
    }
    let bounds = problem.bounds().clone();
    let (d, m) = (problem.dim(), problem.num_objectives());
    let ref_point = problem.ref_point();
    let max_hv = problem.max_hv();
    let n_init = 2 * (d + 1);

    let init = draw_sobol_samples(&bounds, n_init, 1, Some(seed))?;
    let mut train_x = Matrix::from_fn(n_init, d, |i, j| init[i][(0, j)]);
    let mut train_y = problem.forward(&train_x, noise_std > 0.0)?;
    // The hypervolume is computed from the noise-free outcomes.
    let mut true_y = problem.forward(&train_x, false)?;
    println!(
        "{} (negated, d = {d}, m = {m}), {acqf_name}, {model_kind} model, q = {q}, \
         {n_init} Sobol points + {iterations} iterations, seed {seed}, noise std {noise_std}; \
         reference point {ref_point:?}, maximum hypervolume {}",
        problem.name(),
        max_hv.map_or("unknown".into(), |v| format!("{v:.6}"))
    );
    println!(
        "{:>4} {:>6} {:>12} {:>12} {:>9} {:>9} {:>9}",
        "iter", "n", "hypervolume", "log10 gap", "fit [ms]", "acq [ms]", "total [s]"
    );
    let report = |it: usize, n: usize, hv: f64, fit_ms: f64, acq_ms: f64, total: f64| {
        let gap = max_hv.map_or(f64::NAN, |v| (v - hv).max(1e-16).log10());
        println!("{it:>4} {n:>6} {hv:>12.6} {gap:>12.4} {fit_ms:>9.1} {acq_ms:>9.1} {total:>9.2}");
    };
    report(0, n_init, hypervolume(&ref_point, &true_y)?, 0.0, 0.0, 0.0);
    let start = Instant::now();
    for it in 0..iterations {
        let t0 = Instant::now();
        let fit_seed = seed.wrapping_mul(1000) + it as u64;
        let model: Arc<dyn Model> = match model_kind.as_str() {
            "multi" => Arc::new(fit(&train_x, train_y.clone(), &bounds, fit_seed)?),
            "list" => Arc::new(ModelListGP::new(
                (0..m)
                    .map(|j| {
                        fit(
                            &train_x,
                            train_y.columns(j, 1).into_owned(),
                            &bounds,
                            fit_seed,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?),
            other => {
                return Err(BotorchError::InvalidArgument(format!(
                    "unknown model {other} (multi, list)"
                )))
            }
        };
        let fit_ms = t0.elapsed().as_secs_f64() * 1e3;
        let sampler = Some(Arc::new(SobolQMCNormalSampler::new(
            128,
            Some(seed.wrapping_mul(7919) + it as u64),
        )?));
        let acqf: Box<dyn AcquisitionFunction> = match acqf_name.as_str() {
            "qlognehvi" => Box::new(QLogNoisyExpectedHypervolumeImprovement::new(
                model,
                &ref_point,
                &train_x,
                QLogNoisyExpectedHypervolumeImprovementOptions {
                    sampler,
                    prune_baseline: true,
                    ..Default::default()
                },
            )?),
            "qnehvi" => Box::new(QNoisyExpectedHypervolumeImprovement::new(
                model,
                &ref_point,
                &train_x,
                QNoisyExpectedHypervolumeImprovementOptions {
                    sampler,
                    prune_baseline: true,
                    ..Default::default()
                },
            )?),
            "qlogehvi" => {
                // Partitioning of the posterior mean at the observed points.
                let mean = model.posterior(&train_x, false)?.mean_matrix();
                let partitioning = FastNondominatedPartitioning::new(&ref_point, Some(&mean))?;
                Box::new(QLogExpectedHypervolumeImprovement::new(
                    model,
                    &ref_point,
                    &partitioning,
                    QLogExpectedHypervolumeImprovementOptions {
                        sampler,
                        ..Default::default()
                    },
                )?)
            }
            "qlognparego" => Box::new(QLogNParEGO::new(
                model,
                &train_x,
                QLogNParEGOOptions {
                    sampler,
                    weights_seed: Some(seed.wrapping_mul(31) + it as u64),
                    ..Default::default()
                },
            )?),
            other => {
                return Err(BotorchError::InvalidArgument(format!(
                "unknown acquisition function {other} (qlognehvi, qnehvi, qlogehvi, qlognparego)"
            )))
            }
        };
        let t1 = Instant::now();
        let options = OptimizeAcqfOptions {
            seed: Some(seed.wrapping_mul(1000) + it as u64),
            sequential: true,
            ..Default::default()
        };
        let (candidates, _) = optimize_acqf(acqf.as_ref(), &bounds, q, 10, 512, &options)?;
        let acq_ms = t1.elapsed().as_secs_f64() * 1e3;
        train_y = vstack(&train_y, &problem.forward(&candidates, noise_std > 0.0)?);
        true_y = vstack(&true_y, &problem.forward(&candidates, false)?);
        train_x = vstack(&train_x, &candidates);
        report(
            it + 1,
            train_x.nrows(),
            hypervolume(&ref_point, &true_y)?,
            fit_ms,
            acq_ms,
            start.elapsed().as_secs_f64(),
        );
    }
    Ok(())
}
