//! The optima of posterior sample paths (BoTorch's
//! `botorch.acquisition.utils.get_optimal_samples` and
//! `botorch.utils.sampling.optimize_posterior_samples`), the input of the
//! single-objective entropy search acquisition functions
//! ([`super::joint_entropy_search`], [`super::predictive_entropy_search`]).
//!
//! `num_optima` Matheron paths of the model's posterior are drawn
//! ([`crate::sampling::pathwise::get_matheron_path_model`]); every path is
//! evaluated at `raw_samples` quasi-random candidates, its `num_restarts`
//! best candidates are optimized by one joint L-BFGS-B problem over all
//! paths (BoTorch's `gen_candidates_scipy(..., use_parallel_mode=False)`),
//! and the best optimized point of every path is its optimum.
//!
//! The candidates come from a Sobol sequence seeded by `seed` (BoTorch's
//! `SobolEngine` is unseeded) or are given explicitly; the paths' random
//! numbers come from `seed` or explicit draws, like
//! [`crate::sampling::pathwise::MatheronPathOptions`].

use std::sync::Arc;

use rayon::prelude::*;

use crate::acquisition::mc_objective::{McObjective, McSamples};
use crate::acquisition::objective::ScalarizedPosteriorTransform;
use crate::error::{BotorchError, Result};
use crate::lbfgsb::{self, LbfgsbError, LbfgsbOptions};
use crate::models::Model;
use crate::sampling::pathwise::utils::derive_seed;
use crate::sampling::pathwise::{
    get_matheron_path_model, MatheronPathDraws, MatheronPathModel, MatheronPathOptions, SamplePath,
    DEFAULT_NUM_FEATURES,
};
use crate::utils::sampling::{check_bounds, random_seed, SobolEngine};
use crate::Matrix;

/// A transform of the path values to the values that are maximized
/// (BoTorch's `sample_transform`: `posterior_transform.evaluate` or an MC
/// objective).
#[derive(Clone, Debug, Default)]
pub enum SampleTransform {
    /// The values of a single-output model.
    #[default]
    Identity,
    /// A scalarization of the outputs (`ScalarizedPosteriorTransform.evaluate`).
    Scalarize(ScalarizedPosteriorTransform),
    /// An MC objective applied to the values of every path.
    Objective(Arc<dyn McObjective>),
}

/// Options of [`get_optimal_samples`] (BoTorch's keyword arguments and the
/// random numbers).
#[derive(Clone, Debug)]
pub struct OptimalSamplesOptions {
    /// The number of quasi-random candidates (default 1024).
    pub raw_samples: usize,
    /// The number of best candidates optimized per path (default 20).
    pub num_restarts: usize,
    /// The transform of the path values (a posterior transform or an MC
    /// objective; default: none).
    pub transform: SampleTransform,
    /// Return the transformed optimal values (`num_optima × 1`) instead of
    /// the untransformed outputs of the paths (`num_optima × m`).
    pub return_transformed: bool,
    /// The number of Fourier features of the paths (default 1024).
    pub num_features: usize,
    /// The seed of the candidates and of the paths (default: random).
    pub seed: Option<u64>,
    /// Explicit random numbers of the paths (one entry per output, with
    /// `num_optima` samples each).
    pub draws: Option<Vec<MatheronPathDraws>>,
    /// Explicit candidates (`raw_samples × d`, e.g. BoTorch's Sobol draws).
    pub candidates: Option<Matrix>,
    /// The L-BFGS-B settings (default: SciPy's with `maxiter = 2000`).
    pub lbfgsb: LbfgsbOptions,
}

impl Default for OptimalSamplesOptions {
    fn default() -> Self {
        Self {
            raw_samples: 1024,
            num_restarts: 20,
            transform: SampleTransform::Identity,
            return_transformed: false,
            num_features: DEFAULT_NUM_FEATURES,
            seed: None,
            draws: None,
            candidates: None,
            lbfgsb: LbfgsbOptions {
                maxiter: 2000,
                ..LbfgsbOptions::default()
            },
        }
    }
}

/// The transformed values (`n`) of path sample `s` at the points `x` and,
/// with gradients, their derivatives (`n × d`), from the values and
/// gradients of all outputs.
fn transformed_values(
    transform: &SampleTransform,
    values: &[Vec<f64>],
    grads: Option<&[Matrix]>,
    x: &Matrix,
) -> Result<(Vec<f64>, Option<Matrix>)> {
    let m = values.len();
    let n = x.nrows();
    match transform {
        SampleTransform::Identity => {
            if m != 1 {
                return Err(BotorchError::Unsupported(
                    "Multi-output models require a `posterior_transform` or an `objective` in \
                     get_optimal_samples."
                        .into(),
                ));
            }
            Ok((values[0].clone(), grads.map(|g| g[0].clone())))
        }
        SampleTransform::Scalarize(t) => {
            if t.num_outputs() != m {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Output shape not equal to that of weights. Output shape is {m} and weights \
                     are [{}]",
                    t.num_outputs()
                )));
            }
            let w = t.weights();
            let v = (0..n)
                .map(|i| t.offset() + (0..m).map(|k| values[k][i] * w[k]).sum::<f64>())
                .collect();
            let g = grads.map(|g| {
                let mut out = Matrix::zeros(n, x.ncols());
                for (k, gk) in g.iter().enumerate() {
                    out += gk * w[k];
                }
                out
            });
            Ok((v, g))
        }
        SampleTransform::Objective(obj) => {
            let data = Matrix::from_fn(m * n, 1, |r, _| values[r / n][r % n]);
            let samples = McSamples::new(data, n, m)?;
            let out = obj.evaluate(&samples, None, grads.is_some())?;
            if out.values.shape() != (n, 1) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the objective returned {} values for {n} points",
                    out.values.nrows()
                )));
            }
            let v: Vec<f64> = (0..n).map(|i| out.values[(i, 0)]).collect();
            let g = match (grads, out.pullback) {
                (Some(g), Some(pb)) => {
                    let (sbar, xbar) = pb(&Matrix::from_element(n, 1, 1.0))?;
                    let mut gx = Matrix::zeros(n, x.ncols());
                    for (k, gk) in g.iter().enumerate() {
                        for i in 0..n {
                            let a = sbar[(k * n + i, 0)];
                            for j in 0..x.ncols() {
                                gx[(i, j)] += a * gk[(i, j)];
                            }
                        }
                    }
                    if let Some(xb) = xbar {
                        gx += xb;
                    }
                    Some(gx)
                }
                (Some(_), None) => {
                    return Err(BotorchError::Runtime(
                        "the objective returned no reverse pass".into(),
                    ))
                }
                (None, _) => None,
            };
            Ok((v, g))
        }
    }
}

/// The transformed values of path sample `s` at the points `x`.
fn path_values(
    paths: &MatheronPathModel,
    transform: &SampleTransform,
    s: usize,
    x: &Matrix,
    with_grad: bool,
) -> Result<(Vec<f64>, Option<Matrix>)> {
    if with_grad {
        let (values, grads) = paths.paths().forward_with_grad(x)?;
        let vs: Vec<Vec<f64>> = values
            .iter()
            .map(|v| v.row(s).iter().copied().collect())
            .collect();
        let gs: Vec<Matrix> = grads.iter().map(|g| g[s].clone()).collect();
        transformed_values(transform, &vs, Some(&gs), x)
    } else {
        let values = paths.paths().forward(x)?;
        let vs: Vec<Vec<f64>> = values
            .iter()
            .map(|v| v.row(s).iter().copied().collect())
            .collect();
        transformed_values(transform, &vs, None, x)
    }
}

/// BoTorch's `optimize_posterior_samples(paths, bounds, raw_samples,
/// num_restarts, sample_transform, return_transformed)` for the paths of a
/// [`MatheronPathModel`] with `num_optima` samples: the optimal inputs
/// (`num_optima × d`) and outputs (`num_optima × m`, or `num_optima × 1`
/// with `return_transformed`). `candidates` are the quasi-random candidates
/// (`raw_samples × d`, default: a Sobol sequence seeded by `seed`).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for `num_restarts > raw_samples` or
/// non-positive sizes, [`BotorchError::DimensionMismatch`] for candidates of
/// the wrong shape, and errors of the paths, the transform and L-BFGS-B.
#[allow(clippy::too_many_arguments)]
pub fn optimize_posterior_samples(
    paths: &MatheronPathModel,
    bounds: &Matrix,
    raw_samples: usize,
    num_restarts: usize,
    transform: &SampleTransform,
    return_transformed: bool,
    candidates: Option<&Matrix>,
    seed: Option<u64>,
    lbfgsb_options: &LbfgsbOptions,
) -> Result<(Matrix, Matrix)> {
    let d = check_bounds(bounds)?;
    if raw_samples == 0 || num_restarts == 0 || num_restarts > raw_samples {
        return Err(BotorchError::InvalidArgument(format!(
            "need 0 < num_restarts ({num_restarts}) <= raw_samples ({raw_samples})"
        )));
    }
    let candidate_set = match candidates {
        Some(c) => {
            if c.shape() != (raw_samples, d) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected candidates of shape {raw_samples}x{d}, got {}x{}",
                    c.nrows(),
                    c.ncols()
                )));
            }
            c.clone()
        }
        None => {
            let engine = SobolEngine::new(d, Some(seed.unwrap_or_else(random_seed)))?;
            let u = engine.draw(raw_samples, 0)?;
            Matrix::from_fn(raw_samples, d, |i, j| {
                bounds[(0, j)] + (bounds[(1, j)] - bounds[(0, j)]) * u[(i, j)]
            })
        }
    };
    let n_opt = paths.num_samples();
    // The values of every path at the candidates and the best candidates.
    let all = paths.paths().forward(&candidate_set)?;
    let top: Vec<Vec<usize>> = (0..n_opt)
        .map(|s| {
            let vs: Vec<Vec<f64>> = all
                .iter()
                .map(|v| v.row(s).iter().copied().collect())
                .collect();
            let (vals, _) = transformed_values(transform, &vs, None, &candidate_set)?;
            let mut idx: Vec<usize> = (0..raw_samples).collect();
            idx.sort_by(|&a, &b| {
                vals[b]
                    .partial_cmp(&vals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            idx.truncate(num_restarts);
            Ok(idx)
        })
        .collect::<Result<_>>()?;
    // One joint L-BFGS-B problem over all paths and restarts.
    let r = num_restarts;
    let mut x0 = Vec::with_capacity(n_opt * r * d);
    let mut lower = Vec::with_capacity(n_opt * r * d);
    let mut upper = Vec::with_capacity(n_opt * r * d);
    for t in &top {
        for &i in t {
            for j in 0..d {
                x0.push(candidate_set[(i, j)].clamp(bounds[(0, j)], bounds[(1, j)]));
                lower.push(bounds[(0, j)]);
                upper.push(bounds[(1, j)]);
            }
        }
    }
    let block = |x: &[f64], s: usize| Matrix::from_row_slice(r, d, &x[s * r * d..(s + 1) * r * d]);
    let objective = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
        let parts = (0..n_opt)
            .into_par_iter()
            .map(|s| path_values(paths, transform, s, &block(x, s), true))
            .collect::<Result<Vec<_>>>()?;
        let mut f = 0.0;
        let mut g = vec![0.0; x.len()];
        for (s, (v, gr)) in parts.into_iter().enumerate() {
            f -= v.iter().sum::<f64>();
            let gr = gr.expect("gradient requested");
            for i in 0..r {
                for j in 0..d {
                    g[(s * r + i) * d + j] = -gr[(i, j)];
                }
            }
        }
        Ok((f, g))
    };
    let res = match lbfgsb::minimize(objective, &x0, &lower, &upper, lbfgsb_options) {
        Ok(r) => r,
        Err(LbfgsbError::Objective(e)) => return Err(e),
        Err(LbfgsbError::InvalidInput(msg)) => return Err(BotorchError::InvalidArgument(msg)),
    };
    let mut x_opt = Matrix::zeros(n_opt, d);
    let mut f_best = vec![0.0; n_opt];
    for (s, fb) in f_best.iter_mut().enumerate() {
        let mut xs = block(&res.x, s);
        for i in 0..r {
            for j in 0..d {
                xs[(i, j)] = xs[(i, j)].clamp(bounds[(0, j)], bounds[(1, j)]);
            }
        }
        let (v, _) = path_values(paths, transform, s, &xs, false)?;
        // torch.max: the first maximum.
        let mut best = 0;
        for i in 1..r {
            if v[i] > v[best] {
                best = i;
            }
        }
        *fb = v[best];
        x_opt.row_mut(s).copy_from(&xs.row(best));
    }
    if return_transformed {
        return Ok((x_opt, Matrix::from_column_slice(n_opt, 1, &f_best)));
    }
    let values = paths.paths().forward(&x_opt)?;
    let m = values.len();
    let f_opt = Matrix::from_fn(n_opt, m, |s, k| values[k][(s, s)]);
    Ok((x_opt, f_opt))
}

/// BoTorch's `get_optimal_samples(model, bounds, num_optima, raw_samples,
/// num_restarts, posterior_transform, objective, return_transformed)`: the
/// optima of `num_optima` posterior sample paths of `model` (see the
/// [module documentation](self)): optimal inputs (`num_optima × d`) and
/// outputs (`num_optima × m`, or the transformed values `num_optima × 1`
/// with `return_transformed`).
///
/// # Errors
/// See [`optimize_posterior_samples`] and
/// [`crate::sampling::pathwise::draw_matheron_paths`].
pub fn get_optimal_samples(
    model: &dyn Model,
    bounds: &Matrix,
    num_optima: usize,
    options: &OptimalSamplesOptions,
) -> Result<(Matrix, Matrix)> {
    if num_optima == 0 {
        return Err(BotorchError::InvalidArgument(
            "num_optima must be positive".into(),
        ));
    }
    let seed = options.seed.unwrap_or_else(random_seed);
    let paths = get_matheron_path_model(
        model,
        num_optima,
        &MatheronPathOptions {
            num_features: options.num_features,
            seed: Some(derive_seed(seed, 0)),
            draws: options.draws.clone(),
        },
    )?;
    optimize_posterior_samples(
        &paths,
        bounds,
        options.raw_samples,
        options.num_restarts,
        &options.transform,
        options.return_transformed,
        options.candidates.as_ref(),
        Some(derive_seed(seed, 1)),
        &options.lbfgsb,
    )
}
