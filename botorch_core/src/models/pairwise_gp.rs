//! `PairwiseGP`: preference learning with a Gaussian process and a Laplace
//! approximation (`botorch.models.pairwise_gp`), with its marginal log
//! likelihood (`PairwiseLaplaceMarginalLogLikelihood`).
//!
//! The model learns a latent utility `f` from comparisons of its
//! datapoints (Chu & Ghahramani 2005). With the prior `f ~ N(μ, K)` (a
//! `ConstantMean` and a `ScaleKernel` covariance module) and a pairwise
//! likelihood ([`PairwiseLikelihood`]), BoTorch's `_update` computes the
//! Laplace approximation exactly as follows, and so does this port:
//!
//! 1. `K` is the training covariance, made positive definite by
//!    `_ensure_psd_with_jitter` (jitter `0`, `j`, `10 j` added to `K / s`,
//!    `s` the outputscale), and `L` the Cholesky factor of `K / s` with
//!    `psd_safe_cholesky`'s jitter (`j, 10 j, ...`, 6 tries), scaled by `√s`.
//! 2. The MAP utility solves `g(f) = ∇(-log p(D | f)) + (L Lᵀ)⁻¹ (f - μ) = 0`
//!    with `scipy.optimize.fsolve` (MINPACK `hybrj`, ported in
//!    [`crate::minpack`]) and the Jacobian `H(f) + (L Lᵀ)⁻¹` (`H` the
//!    likelihood's [`PairwiseLikelihood::negative_log_hessian_sum`]), `xtol`
//!    (default 1e-6) and `maxfev` (default 100), started from the previous
//!    solution (BoTorch's `_x0`) or, for new data, from the standardized
//!    win counts plus `0.05` times standard normal draws (numpy's global
//!    generator in BoTorch; supplied by the caller here), times `√s`.
//! 3. Two Newton steps `f ← f - (K H(f) + I)⁻¹ K g(f)` follow (BoTorch's
//!    `_util_newton_updates`); the utility is their result `f₂` and the
//!    likelihood Hessian is `H(f₁)` (computed in the second step).
//!
//! The posterior at test points `X*` is BoTorch's eval-mode `forward`: mean
//! `k*ᵀ (L Lᵀ)⁻¹ (f₂ - μ) + μ` and covariance
//! `K** - k*ᵀ (H(f₁) K + I)⁻¹ H(f₁) k*`, whose Cholesky factor (of the
//! covariance divided by `s`, with jitter) times `√s` defines the reported
//! covariance `L* L*ᵀ`, as GPyTorch's `RootLinearOperator`.
//!
//! The marginal log likelihood is BoTorch's
//! `-Σ log p(D | f₂) + ½ f₂ᵀ (L Lᵀ)⁻¹ f₂` (clamped at zero and negated)
//! `- ½ log det(K H(f₁) + I)` plus the log priors, and its gradient is
//! what autograd computes in BoTorch: the derivative through the two Newton
//! steps (with the fsolve solution held fixed), computed here in reverse
//! mode ([`PairwiseGP::mll_value_and_grad`]). At the solution this is the
//! implicit-function derivative of the Laplace evidence; the second Newton
//! step removes the error of the first to second order.
//!
//! Comparisons are given as `(preferred, other)` pairs of datapoint
//! indices. Unbatched data are consolidated first (`consolidate_duplicates`:
//! datapoints within `atol + rtol·max(|x|)` in the max norm are merged);
//! batched models ([`BatchedPairwiseGP`], e.g. fantasy models) are not.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::constraints::{GreaterThan, Interval};
use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::kernels::eval::TreeKernel;
use crate::kernels::{BuiltCovar, CovarModule, KernelKind, KernelSpec, Prepared, ScaleKernelSpec};
use crate::linalg::psd_safe_cholesky_with;
use crate::minpack::fsolve;
use crate::mll::raw_gradient;
use crate::models::likelihoods::{Comparisons, PairwiseLikelihood};
use crate::models::Model;
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::priors::Prior;
use crate::transforms::input::{ChainedInputTransform, TransformMode};
use crate::{Matrix, Vector};

mod input_transform;

/// BoTorch's default `jitter` of `PairwiseGP`.
pub const DEFAULT_JITTER: f64 = 1e-6;
/// The default `xtol` of the MAP solve (`_update`).
pub const DEFAULT_XTOL: f64 = 1e-6;
/// The default `maxfev` of the MAP solve (`_update`).
pub const DEFAULT_MAXFEV: usize = 100;
/// The default `consolidate_atol`.
pub const DEFAULT_CONSOLIDATE_ATOL: f64 = 1e-4;
/// `linear_operator.settings.cholesky_max_tries` as BoTorch sets it.
const CHOLESKY_MAX_TRIES: usize = 6;
/// The tries of `_ensure_psd_with_jitter`.
const ENSURE_PSD_TRIES: usize = 3;
/// The message of `torch.linalg.solve` for a singular matrix.
const SINGULAR: &str =
    "torch.linalg.solve: The solver failed because the input matrix is singular.";

/// BoTorch's default covariance module of `PairwiseGP` for `d` inputs: a
/// `ScaleKernel(RBFKernel(ard_num_dims=d))` with lengthscale prior
/// `Gamma(2.4, 2.7)` and constraint `GreaterThan(1e-4, transform=None,
/// initial_value=prior mode)`, outputscale prior `SmoothedBox(0.01, 100)`
/// and constraint `Interval(0.005, 200, initial_value=1)`. With `float32`,
/// the constants are rounded as GPyTorch stores Python floats under torch's
/// default dtype float32.
pub fn pairwise_default_covar_module(d: usize, float32: bool) -> Result<KernelSpec> {
    let gamma = Prior::Gamma {
        concentration: 2.4,
        rate: 2.7,
    };
    let (gamma, mode) = if float32 {
        let g = gamma.rounded_to_f32();
        (g, g.mode_f32())
    } else {
        (gamma, (2.4 - 1.0) / 2.7)
    };
    let ls = GreaterThan::new(1e-4, None, Some(mode));
    let ls = if float32 { ls.rounded_to_f32() } else { ls };
    let os_prior = Prior::smoothed_box(0.01, 100.0, 0.01)?;
    let os_prior = if float32 {
        os_prior.rounded_to_f32()
    } else {
        os_prior
    };
    let os = Interval::new(
        0.01 * 0.5,
        100.0 * 2.0,
        Some(crate::constraints::ParameterTransform::Sigmoid),
        None,
        Some(1.0),
    )?;
    let os = if float32 { os.rounded_to_f32() } else { os };
    let mut spec = KernelSpec::new(KernelKind::Rbf, d);
    spec.lengthscale_prior = Some(gamma);
    spec.lengthscale_constraint = ls.into();
    Ok(spec.with_scale_kernel(ScaleKernelSpec {
        outputscale_prior: Some(os_prior),
        outputscale_constraint: os.into(),
    }))
}

// ---------------------------------------------------------------------------
// Consolidation of duplicates
// ---------------------------------------------------------------------------

/// BoTorch's `detect_duplicates`: the pairs `(duplicate, original)` of rows
/// of `x` within `atol + rtol · max(max|x_i|, max|x_j|)` in the max norm
/// (strictly), one per duplicate row (the first closest original).
pub fn detect_duplicates(x: &Matrix, rtol: f64, atol: f64) -> Vec<(usize, usize)> {
    let n = x.nrows();
    let rval: Vec<f64> = (0..n)
        .map(|i| {
            x.row(i)
                .iter()
                .fold(f64::NEG_INFINITY, |m, v| m.max(v.abs()))
        })
        .collect();
    let mut out = Vec::new();
    for j in 0..n {
        let mut best = f64::INFINITY;
        let mut arg = 0;
        for i in 0..n {
            let tol = if rtol != 0.0 {
                atol + rtol * rval[i].max(rval[j])
            } else {
                atol
            };
            let dist = if i < j {
                (0..x.ncols()).fold(0.0f64, |m, k| m.max((x[(i, k)] - x[(j, k)]).abs()))
            } else {
                f64::INFINITY
            };
            let diff = dist - tol;
            // torch's `min(dim)` returns the first minimal index.
            if diff < best {
                best = diff;
                arg = i;
            }
        }
        if best < 0.0 {
            out.push((j, arg));
        }
    }
    out
}

/// The result of [`consolidate_duplicates`].
#[derive(Clone, Debug, PartialEq)]
pub struct Consolidated {
    /// The datapoints without duplicates.
    pub datapoints: Matrix,
    /// The comparisons re-indexed to the consolidated datapoints.
    pub comparisons: Vec<(usize, usize)>,
    /// The new index of every original datapoint.
    pub indices: Vec<usize>,
}

/// BoTorch's `consolidate_duplicates`: drops duplicated datapoints and
/// re-indexes the comparisons (a datapoint that is both a duplicate and an
/// original is kept, as in BoTorch).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for a comparison index out of range
/// (Python's `KeyError` in BoTorch, when duplicates are present).
pub fn consolidate_duplicates(
    x: &Matrix,
    comparisons: &[(usize, usize)],
    rtol: f64,
    atol: f64,
) -> Result<Consolidated> {
    let n = x.nrows();
    let found = detect_duplicates(x, rtol, atol);
    let keys: std::collections::BTreeSet<usize> = found.iter().map(|&(k, _)| k).collect();
    let values: std::collections::BTreeSet<usize> = found.iter().map(|&(_, v)| v).collect();
    // BoTorch drops a pair if its duplicate or its original is both a
    // duplicate and an original.
    let common: std::collections::BTreeSet<usize> = keys.intersection(&values).copied().collect();
    let dup: Vec<(usize, usize)> = found
        .into_iter()
        .filter(|&(k, v)| !common.contains(&k) && !common.contains(&v))
        .collect();
    if dup.is_empty() {
        return Ok(Consolidated {
            datapoints: x.clone(),
            comparisons: comparisons.to_vec(),
            indices: (0..n).collect(),
        });
    }
    let dup_set: std::collections::BTreeSet<usize> = dup.iter().map(|&(k, _)| k).collect();
    let unique: Vec<usize> = (0..n).filter(|i| !dup_set.contains(i)).collect();
    let mut new_idx = vec![usize::MAX; n];
    for (r, &u) in unique.iter().enumerate() {
        new_idx[u] = r;
    }
    for &(k, v) in &dup {
        new_idx[k] = new_idx[v];
    }
    let map = |i: usize| -> Result<usize> {
        new_idx
            .get(i)
            .copied()
            .ok_or_else(|| BotorchError::InvalidArgument(format!("KeyError: {i}")))
    };
    let comps = comparisons
        .iter()
        .map(|&(a, b)| Ok((map(a)?, map(b)?)))
        .collect::<Result<Vec<_>>>()?;
    let datapoints = Matrix::from_fn(unique.len(), x.ncols(), |r, c| x[(unique[r], c)]);
    Ok(Consolidated {
        datapoints,
        comparisons: comps,
        indices: new_idx,
    })
}

// ---------------------------------------------------------------------------
// Numerical helpers (BoTorch's operations, in BoTorch's order)
// ---------------------------------------------------------------------------

fn try_chol(a: &Matrix) -> Option<Matrix> {
    if a.iter().any(|v| v.is_nan()) {
        return None;
    }
    nalgebra::Cholesky::new(a.clone()).map(|c| c.unpack())
}

/// BoTorch's `_ensure_psd_with_jitter(matrix, scale, jitter)`.
fn ensure_psd_with_jitter(matrix: &Matrix, scale: f64, jitter: f64) -> Result<Matrix> {
    let mut scaled = matrix.map(|v| v / scale);
    let mut new_jitter = 0.0;
    let mut psd = false;
    let mut last = 0;
    for i in 0..ENSURE_PSD_TRIES {
        if new_jitter != 0.0 {
            for k in 0..scaled.nrows() {
                scaled[(k, k)] += new_jitter;
            }
        }
        last = i;
        if try_chol(&scaled).is_some() {
            psd = true;
            break;
        }
        new_jitter = jitter * 10f64.powi(i as i32) - new_jitter;
    }
    if !psd {
        return Err(BotorchError::NotPsd {
            jitter: jitter * 10f64.powi(last as i32),
        });
    }
    Ok(scaled.map(|v| v * scale))
}

/// BoTorch's `_scaled_psd_safe_cholesky(matrix, scale, jitter)`.
fn scaled_psd_safe_cholesky(matrix: &Matrix, scale: f64, jitter: f64) -> Result<Matrix> {
    let scaled = matrix.map(|v| v / scale);
    let f = psd_safe_cholesky_with(&scaled, jitter, CHOLESKY_MAX_TRIES)?;
    let root = scale.sqrt();
    Ok(f.l().map(|v| v * root))
}

/// `torch.cholesky_solve(b, L)`: `(L Lᵀ)⁻¹ b`.
fn chol_solve(l: &Matrix, b: &Vector) -> Vector {
    let y = l
        .solve_lower_triangular(b)
        .expect("nonsingular Cholesky factor");
    l.tr_solve_lower_triangular(&y)
        .expect("nonsingular Cholesky factor")
}

/// `torch.cholesky_solve(B, L)` for a matrix `B`.
fn chol_solve_mat(l: &Matrix, b: &Matrix) -> Matrix {
    let y = l
        .solve_lower_triangular(b)
        .expect("nonsingular Cholesky factor");
    l.tr_solve_lower_triangular(&y)
        .expect("nonsingular Cholesky factor")
}

/// `torch.cholesky_inverse(L)`.
fn chol_inverse(l: &Matrix) -> Matrix {
    chol_solve_mat(l, &Matrix::identity(l.nrows(), l.nrows()))
}

/// `torch.linalg.solve(A, B)` (LU with partial pivoting).
fn lu_solve(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    a.clone()
        .lu()
        .solve(b)
        .ok_or_else(|| BotorchError::Runtime(SINGULAR.into()))
}

/// `A⁻¹` (LU with partial pivoting).
fn lu_inverse(a: &Matrix) -> Result<Matrix> {
    a.clone()
        .lu()
        .try_inverse()
        .ok_or_else(|| BotorchError::Runtime(SINGULAR.into()))
}

/// `torch.logdet(A)` of a general matrix: `log |det A|`, NaN for a negative
/// determinant.
fn logdet(a: &Matrix) -> f64 {
    let lu = a.clone().lu();
    let det_sign = lu.determinant().signum();
    let u = lu.u();
    let mut s = 0.0;
    for i in 0..u.nrows() {
        s += u[(i, i)].abs().ln();
    }
    if det_sign < 0.0 {
        f64::NAN
    } else {
        s
    }
}

/// numpy's pairwise summation of a contiguous float64 array (`np.add.reduce`).
fn numpy_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, rk) in r.iter_mut().enumerate() {
                *rk += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_sum(&a[..n2]) + numpy_sum(&a[n2..])
    }
}

/// BoTorch's heuristic initial utility for new data:
/// `((wc - mean) / max(std, 1e-6) + 0.05 · draws) · √s` (numpy, `ddof = 0`).
fn initial_utility(comps: &Comparisons, draws: &[f64], sqrt_scale: f64) -> Vec<f64> {
    let wc = comps.win_count();
    let n = wc.len() as f64;
    let mean = numpy_sum(wc.as_slice()) / n;
    let dev: Vec<f64> = wc.iter().map(|&v| (v - mean) * (v - mean)).collect();
    let std = (numpy_sum(&dev) / n).sqrt().max(1e-6);
    wc.iter()
        .zip(draws)
        .map(|(&v, &e)| ((v - mean) / std + 0.05 * e) * sqrt_scale)
        .collect()
}

/// The kernel for raw parameters `raw` (GPyTorch accepts any finite values).
fn build_covar(module: &CovarModule, raw: &[f64], d: usize) -> Result<BuiltCovar> {
    match module {
        CovarModule::Standard(spec) => {
            if spec.scale.is_none() {
                return Err(BotorchError::Unsupported(
                    "PairwiseGP must be used with a ScaleKernel.".into(),
                ));
            }
            if raw.len() != 1 + spec.ard_num_dims {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected {} raw kernel parameters, got {}",
                    1 + spec.ard_num_dims,
                    raw.len()
                )));
            }
            Ok(BuiltCovar::Standard(
                spec.build_unchecked(&raw[1..], Some(raw[0]))?,
            ))
        }
        CovarModule::Tree(node) => Ok(BuiltCovar::Tree(TreeKernel::build(node, raw, d)?)),
    }
}

/// The outputscale of the root `ScaleKernel`.
fn outputscale_of(covar: &BuiltCovar) -> f64 {
    match covar {
        BuiltCovar::Standard(k) => k.outputscale().expect("ScaleKernel"),
        BuiltCovar::Tree(t) => t.values()[0],
    }
}

/// The number of raw parameters of a covariance module.
pub fn num_raw_parameters(module: &CovarModule) -> usize {
    match module {
        CovarModule::Standard(s) => usize::from(s.scale.is_some()) + s.ard_num_dims,
        CovarModule::Tree(t) => t.num_raw(),
    }
}

/// The initial raw parameters of a covariance module (GPyTorch's).
pub fn initial_raw_parameters(module: &CovarModule) -> Vec<f64> {
    match module {
        CovarModule::Standard(s) => {
            let mut v = Vec::new();
            if let Some(os) = s.initial_raw_outputscale() {
                v.push(os);
            }
            v.extend(std::iter::repeat_n(
                s.initial_raw_lengthscale(),
                s.ard_num_dims,
            ));
            v
        }
        CovarModule::Tree(t) => t.initial_raw(),
    }
}

/// The optimizer bounds of the raw parameters (non-enforced constraints).
pub fn raw_parameter_bounds(module: &CovarModule) -> Vec<(f64, f64)> {
    match module {
        CovarModule::Standard(s) => {
            let mut v = Vec::new();
            if let Some(sc) = s.scale {
                v.push(sc.outputscale_constraint.raw_bounds());
            }
            v.extend(std::iter::repeat_n(
                s.lengthscale_constraint.raw_bounds(),
                s.ard_num_dims,
            ));
            v
        }
        CovarModule::Tree(t) => t.raw_bounds(),
    }
}

fn check_scale_kernel(module: &CovarModule) -> Result<()> {
    let ok = match module {
        CovarModule::Standard(s) => s.scale.is_some(),
        CovarModule::Tree(t) => matches!(t.kind(), crate::kernels::KernelType::Scale),
    };
    if ok {
        Ok(())
    } else {
        Err(BotorchError::Unsupported(
            "PairwiseGP must be used with a ScaleKernel.".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// The Laplace approximation
// ---------------------------------------------------------------------------

/// The state of the Laplace approximation after BoTorch's `_update`.
#[derive(Clone, Debug)]
pub struct LaplaceState {
    /// The raw hyperparameters it was computed at.
    pub raw: Vec<f64>,
    /// The outputscale `s`.
    pub outputscale: f64,
    /// BoTorch's `covar`: the training covariance after
    /// `_ensure_psd_with_jitter`.
    pub covar: Matrix,
    /// BoTorch's `covar_chol` (`L`, with `L Lᵀ = covar` up to jitter).
    pub covar_chol: Matrix,
    /// BoTorch's `covar_inv`: `(L Lᵀ)⁻¹`.
    pub covar_inv: Matrix,
    /// The fsolve solution (BoTorch's `_x0`, the next warm start).
    pub x0: Vec<f64>,
    /// MINPACK's `info` of the solve.
    pub fsolve_info: i32,
    /// The function evaluations of the solve.
    pub fsolve_nfev: usize,
    /// The utility after the two Newton steps (BoTorch's `utility`).
    pub utility: Vector,
    /// BoTorch's `likelihood_hess`: the Hessian at the first Newton iterate.
    pub likelihood_hess: Matrix,
    /// The first Newton iterate `f₁`.
    pub x1: Vector,
    /// GPyTorch's NaN hyperparameter gradients (see
    /// [`crate::kernels::nonfinite`]).
    nan_grads: Vec<usize>,
}

impl LaplaceState {
    /// BoTorch's `hlcov_eye`: `likelihood_hess @ covar + I`.
    pub fn hlcov_eye(&self) -> Matrix {
        let n = self.covar.nrows();
        &self.likelihood_hess * &self.covar + Matrix::identity(n, n)
    }
}

/// Where the MAP solve starts.
#[derive(Clone, Copy, Debug)]
enum Start<'a> {
    /// The previous solution.
    Warm(&'a [f64]),
    /// New data: the heuristic with these standard normal draws.
    Fresh(&'a [f64]),
    /// No solve: these values are taken as the fsolve solution (the
    /// function BoTorch differentiates, with the solution held fixed).
    Solved(&'a [f64]),
}

/// The data of a Laplace computation.
struct Problem<'a> {
    likelihood: PairwiseLikelihood,
    comps: &'a Comparisons,
    x_tf: &'a Matrix,
    mean: f64,
    jitter: f64,
    xtol: f64,
    maxfev: usize,
}

/// The pieces of a Newton step kept for the reverse pass.
struct NewtonRecord {
    x: Vector,
    h: Matrix,
    a_inv: Matrix,
    g: Vector,
    delta: Vector,
    r: Vector,
}

/// The result of [`laplace`] with what the MLL gradient needs.
struct Laplace {
    state: LaplaceState,
    covar_built: BuiltCovar,
    prepared: Prepared,
    steps: Vec<NewtonRecord>,
}

fn newton_step(
    p: &Problem<'_>,
    k: &Matrix,
    l: &Matrix,
    x: &Vector,
) -> Result<(Vector, NewtonRecord)> {
    let n = x.len();
    let h = p.likelihood.negative_log_hessian_sum(x, p.comps);
    let mut a = k * &h;
    for i in 0..n {
        a[(i, i)] += 1.0;
    }
    let r = chol_solve(l, &x.add_scalar(-p.mean));
    let g = p.likelihood.negative_log_gradient_sum(x, p.comps)? + &r;
    let cov_g = k * &g;
    let a_inv = lu_inverse(&a)?;
    let delta = lu_solve(&a, &Matrix::from_column_slice(n, 1, cov_g.as_slice()))?;
    let delta = Vector::from_column_slice(delta.as_slice());
    let next = x - &delta;
    Ok((
        next,
        NewtonRecord {
            x: x.clone(),
            h,
            a_inv,
            g,
            delta,
            r,
        },
    ))
}

fn laplace(
    p: &Problem<'_>,
    module: &CovarModule,
    raw: &[f64],
    start: Start<'_>,
) -> Result<Laplace> {
    let n = p.x_tf.nrows();
    let covar_built = build_covar(module, raw, p.x_tf.ncols())?;
    let prepared = covar_built.prepare(p.x_tf)?;
    let mut k = covar_built.symmetric(&prepared);
    let nan = covar_built.training_nan(p.x_tf, false);
    nan.inject(&mut k);
    let s = outputscale_of(&covar_built);
    let covar = ensure_psd_with_jitter(&k, s, p.jitter)?;
    let l = scaled_psd_safe_cholesky(&covar, s, p.jitter)?;
    let covar_inv = chol_inverse(&l);

    let x0: Vec<f64> = match start {
        Start::Warm(x) => x.to_vec(),
        Start::Fresh(draws) => initial_utility(p.comps, draws, s.sqrt()),
        Start::Solved(x) => x.to_vec(),
    };
    let solved = matches!(start, Start::Solved(_));
    let res = if solved {
        crate::minpack::HybrjResult {
            x: x0.clone(),
            fvec: Vec::new(),
            info: 1,
            nfev: 0,
            njev: 0,
        }
    } else {
        fsolve(
            |u| {
                let u = Vector::from_column_slice(u);
                let b = p.likelihood.negative_log_gradient_sum(&u, p.comps)?;
                let g = b + chol_solve(&l, &u.add_scalar(-p.mean));
                Ok(g.as_slice().to_vec())
            },
            |u| {
                let u = Vector::from_column_slice(u);
                Ok(p.likelihood.negative_log_hessian_sum(&u, p.comps) + &covar_inv)
            },
            &x0,
            p.xtol,
            p.maxfev,
        )?
    };
    let f = Vector::from_column_slice(&res.x);
    let (x1, s0) = newton_step(p, &covar, &l, &f)?;
    let (x2, s1) = newton_step(p, &covar, &l, &x1)?;
    let likelihood_hess = s1.h.clone();
    debug_assert_eq!(x2.len(), n);
    Ok(Laplace {
        state: LaplaceState {
            raw: raw.to_vec(),
            outputscale: s,
            covar,
            covar_chol: l,
            covar_inv,
            x0: res.x.clone(),
            fsolve_info: res.info,
            fsolve_nfev: res.nfev,
            utility: x2,
            likelihood_hess,
            x1,
            nan_grads: nan.nan_gradients().to_vec(),
        },
        covar_built,
        prepared,
        steps: vec![s0, s1],
    })
}

/// The data part of the Laplace MLL (without priors) and, with `with_grad`,
/// its gradient with respect to the training covariance `K` (symmetric).
fn mll_data(p: &Problem<'_>, lap: &Laplace, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
    let st = &lap.state;
    let n = st.utility.len();
    let f = &st.utility;
    let log_lik = p.likelihood.log_p(f, p.comps);
    let neg_ll_sum = -log_lik.sum();
    let inv_prod = chol_solve(&st.covar_chol, f);
    let log_prior = 0.5 * f.dot(&inv_prod);
    let s_loss = neg_ll_sum + log_prior;
    let log_posterior = -(s_loss.max(0.0));
    let mut a1 = &st.covar * &st.likelihood_hess;
    for i in 0..n {
        a1[(i, i)] += 1.0;
    }
    let value = -0.5 * logdet(&a1) + log_posterior;
    if !with_grad {
        return Ok((value, None));
    }
    let k = &st.covar;
    let l = &st.covar_chol;
    let mut kbar = Matrix::zeros(n, n);
    let mut ktbar = Matrix::zeros(n, n);
    // -S (clamped at zero from below; torch >= 2.14: no gradient at 0).
    let mut xbar = Vector::zeros(n);
    if s_loss > 0.0 {
        xbar = p.likelihood.log_p_sum_grad(f, p.comps) - &inv_prod;
        ktbar.ger(0.5, &inv_prod, &inv_prod, 1.0);
    }
    // -½ log det(A₁), A₁ = K H(f₁) + I (the matrix of the second Newton step).
    let s1 = &lap.steps[1];
    let mut abar1 = s1.a_inv.transpose() * -0.5;
    // Reverse through the second Newton step f₂ = f₁ - A₁⁻¹ K g(f₁).
    let x1bar = reverse_step(p, k, l, s1, &xbar, &mut abar1, &mut kbar, &mut ktbar, true)?;
    // ... and the first (f₀, the fsolve solution, is constant).
    let s0 = &lap.steps[0];
    let mut abar0 = Matrix::zeros(n, n);
    reverse_step(
        p, k, l, s0, &x1bar, &mut abar0, &mut kbar, &mut ktbar, false,
    )?;
    let total = kbar + ktbar;
    let w = (&total + total.transpose()) * 0.5;
    Ok((value, Some(w)))
}

/// The reverse pass of one Newton step `x' = x - A⁻¹ K g(x)` with
/// `A = K H(x) + I`: accumulates the adjoints of `K` (`kbar`, through `A`
/// and `K g`) and of `L Lᵀ` (`ktbar`, through `g`) given the adjoint of
/// `x'` and the extra adjoint `abar` of `A`; returns the adjoint of `x` if
/// `need_x`.
#[allow(clippy::too_many_arguments)]
fn reverse_step(
    p: &Problem<'_>,
    k: &Matrix,
    l: &Matrix,
    s: &NewtonRecord,
    xnext_bar: &Vector,
    abar: &mut Matrix,
    kbar: &mut Matrix,
    ktbar: &mut Matrix,
    need_x: bool,
) -> Result<Vector> {
    let delta_bar = -xnext_bar;
    let cbar = s.a_inv.transpose() * &delta_bar;
    abar.ger(-1.0, &cbar, &s.delta, 1.0);
    kbar.ger(1.0, &cbar, &s.g, 1.0);
    let gbar = k * &cbar;
    let kt_inv_gbar = chol_solve(l, &gbar);
    ktbar.ger(-1.0, &kt_inv_gbar, &s.r, 1.0);
    *kbar += &*abar * &s.h;
    if !need_x {
        return Ok(Vector::zeros(xnext_bar.len()));
    }
    let hbar = k * &*abar;
    let jb = p.likelihood.gradient_jacobian(&s.x, p.comps)?;
    let mut xbar = xnext_bar.clone();
    xbar += jb * &gbar;
    xbar += kt_inv_gbar;
    xbar += p.likelihood.hessian_vjp(&s.x, p.comps, &hbar);
    Ok(xbar)
}

/// BoTorch's training-mode `forward` also factorizes the training
/// posterior covariance `K - K (H K + I)⁻¹ H K` (raising `NotPSDError` if
/// it fails even with jitter).
fn check_training_posterior(st: &LaplaceState, jitter: f64) -> Result<()> {
    let n = st.covar.nrows();
    let hl_cov = &st.likelihood_hess * &st.covar;
    let mut hl_cov_i = hl_cov.clone();
    for i in 0..n {
        hl_cov_i[(i, i)] += 1.0;
    }
    let output = &st.covar - &st.covar * lu_solve(&hl_cov_i, &hl_cov)?;
    scaled_psd_safe_cholesky(&output, st.outputscale, jitter).map(|_| ())
}

/// The log priors of the hyperparameters and their gradients with respect
/// to the raw parameters (`grad` is the data gradient with respect to the
/// actual values, mapped to raw values and combined with the priors).
fn priors_and_raw_grad(
    module: &CovarModule,
    covar: &BuiltCovar,
    prepared: &Prepared,
    raw: &[f64],
    w: Option<&Matrix>,
    nan_grads: &[usize],
) -> Result<(f64, Option<Vec<f64>>)> {
    match (module, covar) {
        (CovarModule::Standard(spec), BuiltCovar::Standard(kernel)) => {
            let scale = spec.scale.expect("ScaleKernel");
            let s = kernel.outputscale().expect("ScaleKernel");
            let ls = kernel.lengthscale();
            let mut lp = 0.0;
            if let Some(prior) = scale.outputscale_prior {
                check_support(&prior, s)?;
                lp += prior.log_prob(s);
            }
            if let Some(prior) = spec.lengthscale_prior {
                for &v in ls.iter() {
                    check_support(&prior, v)?;
                    lp += prior.log_prob(v);
                }
            }
            let Some(w) = w else {
                return Ok((lp, None));
            };
            let Prepared::Standard(points) = prepared else {
                unreachable!("points prepared for another kernel")
            };
            let (gl, gs) = kernel.contract_hyper_grad(points, w);
            let mut grad = vec![0.0; raw.len()];
            let deriv = scale.outputscale_constraint.transform_derivative(raw[0]);
            grad[0] = raw_gradient(
                gs.expect("outputscale gradient"),
                scale.outputscale_prior,
                s,
                deriv,
                1.0,
            );
            for j in 0..spec.ard_num_dims {
                let deriv = spec.lengthscale_constraint.transform_derivative(raw[1 + j]);
                grad[1 + j] = raw_gradient(gl[j], spec.lengthscale_prior, ls[j], deriv, 1.0);
            }
            for &j in nan_grads {
                grad[1 + j] = f64::NAN;
            }
            Ok((lp, Some(grad)))
        }
        (CovarModule::Tree(node), BuiltCovar::Tree(tree)) => {
            let Some(w) = w else {
                return Ok((tree.log_prior()?, None));
            };
            let (lp, gp) = tree.log_prior_and_grad()?;
            let Prepared::Tree(points) = prepared else {
                unreachable!("points prepared for another kernel")
            };
            let gk = tree.contract(points, w);
            let derivs = node.transform_derivatives(raw);
            let mut grad: Vec<f64> = (0..raw.len())
                .map(|j| (gk[j] + gp[j]) * derivs[j])
                .collect();
            for &j in nan_grads {
                grad[j] = f64::NAN;
            }
            Ok((lp, Some(grad)))
        }
        _ => unreachable!("covariance module and kernel of different kinds"),
    }
}

fn check_support(prior: &Prior, v: f64) -> Result<()> {
    if !prior.in_validated_support(v) {
        return Err(BotorchError::InvalidArgument(format!(
            "Expected value argument ({v}) to be within the support of the distribution \
             {}Prior, but found invalid values",
            prior.name()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// Options of [`PairwiseGP`] (BoTorch's constructor arguments).
#[derive(Clone, Debug)]
pub struct PairwiseGPOptions {
    /// The likelihood (default: probit).
    pub likelihood: PairwiseLikelihood,
    /// The covariance module; must be a `ScaleKernel` (default:
    /// [`pairwise_default_covar_module`]).
    pub covar_module: Option<CovarModule>,
    /// Raw hyperparameters (default: the module's initial values).
    pub raw_parameters: Option<Vec<f64>>,
    /// An input transform without learned parameters.
    pub input_transform: Option<ChainedInputTransform>,
    /// `jitter` (default 1e-6).
    pub jitter: f64,
    /// `xtol` of the MAP solve (`None`: 1e-6).
    pub xtol: Option<f64>,
    /// `consolidate_rtol` (default 0).
    pub consolidate_rtol: f64,
    /// `consolidate_atol` (default 1e-4).
    pub consolidate_atol: f64,
    /// `maxfev` of the MAP solve (`None`: 100).
    pub maxfev: Option<usize>,
    /// The constant of the `ConstantMean` (default 0, not fitted).
    pub mean_constant: f64,
    /// Whether the default covariance module rounds its constants to
    /// float32 (torch's default dtype).
    pub float32: bool,
    /// Seed of the generator of the initial perturbations when none are
    /// supplied.
    pub seed: u64,
}

impl Default for PairwiseGPOptions {
    fn default() -> Self {
        Self {
            likelihood: PairwiseLikelihood::Probit,
            covar_module: None,
            raw_parameters: None,
            input_transform: None,
            jitter: DEFAULT_JITTER,
            xtol: None,
            consolidate_rtol: 0.0,
            consolidate_atol: DEFAULT_CONSOLIDATE_ATOL,
            maxfev: None,
            mean_constant: 0.0,
            float32: true,
            seed: 0,
        }
    }
}

/// The training data of a [`PairwiseGP`].
#[derive(Clone, Debug)]
pub struct PairwiseData {
    /// The datapoints as given (BoTorch's `unconsolidated_datapoints`).
    pub unconsolidated_datapoints: Matrix,
    /// The comparisons as given.
    pub unconsolidated_comparisons: Vec<(usize, usize)>,
    /// The consolidated datapoints (BoTorch's `datapoints`).
    pub datapoints: Matrix,
    /// The consolidated comparisons (BoTorch's `comparisons`).
    pub comparisons: Comparisons,
    /// The new index of every original datapoint (`None` in batch mode).
    pub consolidated_indices: Option<Vec<usize>>,
}

/// BoTorch's `PairwiseGP` (one batch element; see [`BatchedPairwiseGP`]).
#[derive(Clone, Debug)]
pub struct PairwiseGP {
    likelihood: PairwiseLikelihood,
    covar_module: CovarModule,
    raw: Vec<f64>,
    mean_constant: f64,
    input_transform: Option<ChainedInputTransform>,
    jitter: f64,
    xtol: Option<f64>,
    consolidate_rtol: f64,
    consolidate_atol: f64,
    maxfev: Option<usize>,
    dim: usize,
    data: Option<PairwiseData>,
    training: bool,
    x0: Option<Vec<f64>>,
    state: Option<Arc<LaplaceState>>,
    rng: u64,
}

impl PairwiseGP {
    /// A model with datapoints (`n × d`) and comparisons `(preferred,
    /// other)`, or a prior-only model of dimension `dim` without data. The
    /// Laplace approximation is computed (in train mode, like BoTorch's
    /// constructor) with the initial perturbations `draws` (`n` standard
    /// normal values after consolidation; drawn from the model's own
    /// generator if `None`).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for a covariance module that is not a
    /// `ScaleKernel` or a one-to-many input transform;
    /// [`BotorchError::InvalidArgument`] for comparisons out of range or
    /// wrong `draws`; numerical errors of the Laplace approximation.
    pub fn new(
        data: Option<(Matrix, Vec<(usize, usize)>)>,
        dim: usize,
        options: PairwiseGPOptions,
        draws: Option<&[f64]>,
    ) -> Result<Self> {
        Self::new_impl(data, dim, options, draws, true)
    }

    fn new_impl(
        data: Option<(Matrix, Vec<(usize, usize)>)>,
        dim: usize,
        options: PairwiseGPOptions,
        draws: Option<&[f64]>,
        consolidate: bool,
    ) -> Result<Self> {
        let covar_module = match options.covar_module {
            Some(m) => m,
            None => CovarModule::Standard(pairwise_default_covar_module(dim, options.float32)?),
        };
        check_scale_kernel(&covar_module)?;
        let mut input_transform = options.input_transform;
        let kernel_parameters = num_raw_parameters(&covar_module);
        let input_parameters = input_transform.as_ref().map_or(0, |t| t.num_params());
        let mut raw = options
            .raw_parameters
            .unwrap_or_else(|| initial_raw_parameters(&covar_module));
        // The constructor also accepts the historical kernel-only initial
        // vector, with transform concentrations supplied by the chain itself.
        if raw.len() == kernel_parameters {
            if let Some(chain) = &input_transform {
                raw.extend(chain.params());
            }
        }
        if raw.len() != kernel_parameters + input_parameters {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters (kernel then input transform), got {}",
                kernel_parameters + input_parameters,
                raw.len()
            )));
        }
        if let Some(t) = &mut input_transform {
            t.set_params(&raw[kernel_parameters..])?;
            if t.is_one_to_many() {
                return Err(BotorchError::Unsupported(
                    "one-to-many input transforms (InputPerturbation) are not supported by \
                     PairwiseGP"
                        .into(),
                ));
            }
        }
        let mut model = Self {
            likelihood: options.likelihood,
            covar_module,
            raw,
            mean_constant: options.mean_constant,
            input_transform,
            jitter: options.jitter,
            xtol: options.xtol,
            consolidate_rtol: options.consolidate_rtol,
            consolidate_atol: options.consolidate_atol,
            maxfev: options.maxfev,
            dim,
            data: None,
            training: true,
            x0: None,
            state: None,
            rng: options.seed,
        };
        if let Some((x, comps)) = data {
            model.set_train_data_impl(x, comps, consolidate, true, draws)?;
        }
        Ok(model)
    }

    // -- data -------------------------------------------------------------

    fn set_train_data_impl(
        &mut self,
        x: Matrix,
        comps: Vec<(usize, usize)>,
        consolidate: bool,
        update: bool,
        draws: Option<&[f64]>,
    ) -> Result<()> {
        if x.ncols() != self.dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected datapoints with {} columns, got {}",
                self.dim,
                x.ncols()
            )));
        }
        let (dp, cp, idx) = if consolidate {
            let c =
                consolidate_duplicates(&x, &comps, self.consolidate_rtol, self.consolidate_atol)?;
            (c.datapoints, c.comparisons, Some(c.indices))
        } else {
            (x.clone(), comps.clone(), None)
        };
        let comparisons = Comparisons::new(cp, dp.nrows())?;
        self.data = Some(PairwiseData {
            unconsolidated_datapoints: x,
            unconsolidated_comparisons: comps,
            datapoints: dp,
            comparisons,
            consolidated_indices: idx,
        });
        if update {
            self.update(draws)?;
        }
        Ok(())
    }

    /// BoTorch's `set_train_data(datapoints, comparisons, update_model)`.
    ///
    /// # Errors
    /// See [`PairwiseGP::new`].
    pub fn set_train_data(
        &mut self,
        x: Matrix,
        comps: Vec<(usize, usize)>,
        update_model: bool,
        draws: Option<&[f64]>,
    ) -> Result<()> {
        let consolidate = self
            .data
            .as_ref()
            .is_none_or(|d| d.consolidated_indices.is_some());
        self.set_train_data_impl(x, comps, consolidate, update_model, draws)
    }

    /// The training data (`None` for a prior-only model).
    pub fn data(&self) -> Option<&PairwiseData> {
        self.data.as_ref()
    }

    /// The number of (consolidated) datapoints `n` (0 without data).
    pub fn num_datapoints(&self) -> usize {
        self.data.as_ref().map_or(0, |d| d.datapoints.nrows())
    }

    /// Whether a fresh start of the MAP solve is needed (BoTorch draws
    /// `n` standard normal values then): no previous solution of the right
    /// size.
    pub fn needs_fresh_start(&self) -> bool {
        let n = self.num_datapoints();
        n > 0 && self.x0.as_ref().is_none_or(|x| x.len() != n)
    }

    /// The likelihood.
    pub fn likelihood(&self) -> PairwiseLikelihood {
        self.likelihood
    }

    /// The covariance module.
    pub fn covar_module(&self) -> &CovarModule {
        &self.covar_module
    }

    /// The raw hyperparameters: kernel parameters followed by learned input
    /// transform parameters (concentration0 then concentration1 for Warp).
    pub fn raw_parameters(&self) -> &[f64] {
        &self.raw
    }

    /// Sets the raw hyperparameters (the Laplace approximation is not
    /// recomputed, as in BoTorch; call [`PairwiseGP::update`]).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of values and
    /// [`BotorchError::InvalidArgument`] for non-finite values.
    pub fn set_raw_parameters(&mut self, raw: &[f64]) -> Result<()> {
        if raw.len() != self.raw.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                self.raw.len(),
                raw.len()
            )));
        }
        if raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "non-finite raw hyperparameters: {raw:?}"
            )));
        }
        self.raw = raw.to_vec();
        let kernel_parameters = num_raw_parameters(&self.covar_module);
        if let Some(chain) = &mut self.input_transform {
            chain.set_params(&raw[kernel_parameters..])?;
        }
        Ok(())
    }

    /// The optimizer bounds of the raw hyperparameters.
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = raw_parameter_bounds(&self.covar_module);
        if let Some(chain) = &self.input_transform {
            bounds.extend(chain.param_bounds());
        }
        bounds
    }

    /// The `ConstantMean` constant.
    pub fn mean_constant(&self) -> f64 {
        self.mean_constant
    }

    /// Sets the `ConstantMean` constant (not fitted; the Laplace
    /// approximation is not recomputed).
    pub fn set_mean_constant(&mut self, c: f64) {
        self.mean_constant = c;
    }

    /// The input dimension.
    pub fn input_dim(&self) -> usize {
        self.dim
    }

    /// `jitter`, `xtol`, `consolidate_rtol`, `consolidate_atol`, `maxfev`.
    pub fn settings(&self) -> (f64, Option<f64>, f64, f64, Option<usize>) {
        (
            self.jitter,
            self.xtol,
            self.consolidate_rtol,
            self.consolidate_atol,
            self.maxfev,
        )
    }

    /// The input transform (fitted).
    pub fn input_transform(&self) -> Option<&ChainedInputTransform> {
        self.input_transform.as_ref()
    }

    /// Whether the model is in train mode.
    pub fn training(&self) -> bool {
        self.training
    }

    /// Sets train (`true`) or eval mode.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        if let Some(chain) = self.input_transform.take() {
            let steps = chain.training().len();
            self.input_transform = Some(
                chain
                    .with_training(vec![training; steps])
                    .expect("one mode flag per transform step"),
            );
        }
    }

    /// The Laplace approximation (`None` without data).
    pub fn state(&self) -> Option<&LaplaceState> {
        self.state.as_deref()
    }

    /// The warm start of the next MAP solve (BoTorch's `_x0`).
    pub fn warm_start(&self) -> Option<&[f64]> {
        self.x0.as_deref()
    }

    /// Sets the warm start (BoTorch's `_x0`).
    pub fn set_warm_start(&mut self, x0: Option<Vec<f64>>) {
        self.x0 = x0;
    }

    // -- the Laplace approximation ------------------------------------------

    /// The transformed consolidated datapoints in the current mode (train
    /// mode relearns the transform's coefficients, as BoTorch's transforms
    /// do in train mode).
    fn transformed_datapoints(&mut self) -> Result<Matrix> {
        let dp = &self.data.as_ref().expect("data").datapoints;
        match &mut self.input_transform {
            None => Ok(dp.clone()),
            Some(t) if self.training => t.fit(dp),
            Some(t) => t.transform_mode(dp, TransformMode::Eval),
        }
    }

    fn fresh_draws(&mut self, n: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(self.rng);
        self.rng = self.rng.wrapping_add(1);
        (0..n).map(|_| StandardNormal.sample(&mut rng)).collect()
    }

    fn problem<'a>(&'a self, x_tf: &'a Matrix) -> Problem<'a> {
        let d = self.data.as_ref().expect("data");
        Problem {
            likelihood: self.likelihood,
            comps: &d.comparisons,
            x_tf,
            mean: self.mean_constant,
            jitter: self.jitter,
            xtol: self.xtol.unwrap_or(DEFAULT_XTOL),
            maxfev: self.maxfev.unwrap_or(DEFAULT_MAXFEV),
        }
    }

    /// BoTorch's `_update`: recomputes the Laplace approximation at the
    /// current hyperparameters, warm-started from the previous solution, or
    /// from the heuristic with the standard normal `draws` (length `n`;
    /// drawn from the model's generator if `None`) when there is none of
    /// the right size.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for `draws` of the wrong length;
    /// numerical errors (`NotPsd`, a singular Newton system).
    pub fn update(&mut self, draws: Option<&[f64]>) -> Result<()> {
        if self.data.is_none() {
            return Ok(());
        }
        let n = self.num_datapoints();
        let x_tf = self.transformed_datapoints()?;
        let fresh: Option<Vec<f64>> = if self.needs_fresh_start() {
            Some(match draws {
                Some(d) if d.len() == n => d.to_vec(),
                Some(d) => {
                    return Err(BotorchError::InvalidArgument(format!(
                        "expected {n} initial draws, got {}",
                        d.len()
                    )))
                }
                None => self.fresh_draws(n),
            })
        } else {
            None
        };
        let lap = {
            let p = self.problem(&x_tf);
            let start = match &fresh {
                Some(d) => Start::Fresh(d),
                None => Start::Warm(self.x0.as_deref().expect("warm start")),
            };
            laplace(&p, &self.covar_module, self.kernel_raw_parameters(), start)?
        };
        self.x0 = Some(lap.state.x0.clone());
        let mut state = lap.state;
        state.raw = self.raw.clone();
        self.state = Some(Arc::new(state));
        Ok(())
    }

    /// The Laplace state at the current hyperparameters from the given
    /// fsolve solution `x0` (the model keeps its state): BoTorch's two Newton
    /// steps of `_update` without the MAP solve. With another
    /// implementation's fsolve solution this gives both the same input to
    /// everything downstream of the solve.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] without data or with `x0` of the
    /// wrong length; errors of the covariance and its factorization.
    pub fn laplace_from_solution(&mut self, x0: &[f64]) -> Result<LaplaceState> {
        if self.data.is_none() || x0.len() != self.num_datapoints() {
            return Err(BotorchError::InvalidArgument(format!(
                "expected an fsolve solution of {} values, got {}",
                self.num_datapoints(),
                x0.len()
            )));
        }
        let x_tf = self.transformed_datapoints()?;
        let p = self.problem(&x_tf);
        let mut state = laplace(
            &p,
            &self.covar_module,
            self.kernel_raw_parameters(),
            Start::Solved(x0),
        )?
        .state;
        state.raw = self.raw.clone();
        Ok(state)
    }

    /// The Laplace marginal log likelihood (BoTorch's
    /// `PairwiseLaplaceMarginalLogLikelihood`, priors included) at the raw
    /// hyperparameters `raw`, warm-started from the current solution, and
    /// its gradient with respect to `raw` (what BoTorch's autograd computes
    /// through the two Newton steps). The model is not changed; the new
    /// state is returned (the next warm start is its `x0`).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for a model without data; numerical
    /// errors of the Laplace approximation (BoTorch's `NotPSDError`, a
    /// singular Newton system).
    pub fn mll_value_and_grad(
        &self,
        raw: &[f64],
        with_grad: bool,
    ) -> Result<(f64, Option<Vec<f64>>, LaplaceState)> {
        let x0 = self
            .x0
            .clone()
            .ok_or_else(|| BotorchError::Unsupported("the model has no data".into()))?;
        self.evaluate_trial(raw, &x0, with_grad)
    }

    /// The transformed datapoints as a training-mode forward pass sees them.
    fn training_inputs_tf(&self) -> Result<Matrix> {
        let d = self
            .data
            .as_ref()
            .ok_or_else(|| BotorchError::Unsupported("the model has no data".into()))?;
        match &self.input_transform {
            None => Ok(d.datapoints.clone()),
            Some(t) => {
                let mut t = t.clone();
                t.fit(&d.datapoints)
            }
        }
    }

    /// The MLL at the current state (no recomputation; BoTorch's
    /// `mll(model(datapoints), comparisons)` after the forward pass).
    ///
    /// # Errors
    /// See [`PairwiseGP::mll_value_and_grad`].
    pub fn mll(&self) -> Result<f64> {
        Ok(self.mll_value_and_grad(&self.raw, false)?.0)
    }

    // -- the posterior --------------------------------------------------------

    /// The test-point transform and its Jacobians (`None` without a
    /// transform).
    fn transform_test(&self, x: &Matrix, with_grad: bool) -> Result<(Matrix, Option<Vec<Matrix>>)> {
        if x.ncols() != self.dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X with {} columns (d), got {}",
                self.dim,
                x.ncols()
            )));
        }
        match &self.input_transform {
            None => Ok((x.clone(), None)),
            Some(t) => {
                let mode = if self.training {
                    TransformMode::Train
                } else {
                    TransformMode::Eval
                };
                if with_grad {
                    let (xt, tape) = t.transform_with_tape_mode(x, mode)?;
                    let jacs = tape.jacobians_through(t)?;
                    Ok((xt, Some(jacs)))
                } else {
                    Ok((t.transform_mode(x, mode)?, None))
                }
            }
        }
    }

    /// The posterior (eval-mode `forward`) at raw inputs `x` with, if
    /// requested, the derivatives with respect to the inputs.
    fn posterior_impl(&self, x: &Matrix, with_grad: bool) -> Result<JointPosteriorGrad> {
        let (xt, jacs) = self.transform_test(x, with_grad)?;
        let q = xt.nrows();
        let dt = xt.ncols();
        let covar = build_covar(&self.covar_module, self.kernel_raw_parameters(), dt)?;
        let s = outputscale_of(&covar);
        let pn = covar.prepare(&xt)?;
        let kxx = covar.symmetric(&pn);
        let kxx_psd = ensure_psd_with_jitter(&kxx, s, self.jitter)?;
        let mut dmean_t: Vec<Vector> = Vec::new();
        let mut dcov_t: Vec<Matrix> = Vec::new();
        let (mean, pred) = match (&self.data, &self.state) {
            (Some(data), Some(st)) => {
                let x_train = match &self.input_transform {
                    None => data.datapoints.clone(),
                    Some(t) => t.transform_mode(&data.datapoints, TransformMode::Eval)?,
                };
                let pt = covar.prepare(&x_train)?;
                let n = x_train.nrows();
                let mu = self.mean_constant;
                let alpha = chol_solve(&st.covar_chol, &st.utility.add_scalar(-mu));
                let m = lu_solve(&st.hlcov_eye(), &st.likelihood_hess)?;
                let mut kx = Matrix::zeros(q, n);
                let mut dks = Vec::with_capacity(q);
                for p in 0..q {
                    let (k, dk) = covar.cross_with_grad(&pn, p, &pt);
                    kx.set_row(p, &k.transpose());
                    dks.push(dk);
                }
                let mean = (&kx * &alpha).add_scalar(mu);
                let fac = &m * kx.transpose(); // n × q
                let pred = &kxx_psd - &kx * &fac;
                if with_grad {
                    let t = &kx * &m; // q × n
                    for p in 0..q {
                        let (_, dkxx) = covar.cross_with_grad(&pn, p, &pn);
                        let (_, ddiag) = covar.diag_with_grad(&pn, p);
                        for k in 0..dt {
                            let r = dks[p].column(k);
                            let mut dm = Vector::zeros(q);
                            dm[p] = r.dot(&alpha);
                            let mut da = Matrix::zeros(q, q);
                            for b in 0..q {
                                let v = if b == p {
                                    ddiag.as_ref().map_or(0.0, |g| g[k])
                                } else {
                                    dkxx[(b, k)]
                                };
                                da[(p, b)] += v;
                                if b != p {
                                    da[(b, p)] += v;
                                }
                            }
                            for b in 0..q {
                                da[(p, b)] -= r.dot(&fac.column(b));
                                da[(b, p)] -= t.row(b).transpose().dot(&r);
                            }
                            dmean_t.push(dm);
                            dcov_t.push((&da + da.transpose()) * 0.5);
                        }
                    }
                }
                (mean, pred)
            }
            _ => {
                // Prior mode.
                if with_grad {
                    for p in 0..q {
                        let (_, dkxx) = covar.cross_with_grad(&pn, p, &pn);
                        let (_, ddiag) = covar.diag_with_grad(&pn, p);
                        for k in 0..dt {
                            let mut da = Matrix::zeros(q, q);
                            for b in 0..q {
                                let v = if b == p {
                                    ddiag.as_ref().map_or(0.0, |g| g[k])
                                } else {
                                    dkxx[(b, k)]
                                };
                                da[(p, b)] += v;
                                if b != p {
                                    da[(b, p)] += v;
                                }
                            }
                            dmean_t.push(Vector::zeros(q));
                            dcov_t.push((&da + da.transpose()) * 0.5);
                        }
                    }
                }
                (Vector::from_element(q, self.mean_constant), kxx_psd)
            }
        };
        let root = scaled_psd_safe_cholesky(&pred, s, self.jitter)?;
        let covariance = &root * root.transpose();
        if !with_grad {
            return Ok(JointPosteriorGrad {
                mean,
                covariance,
                dmean: Vec::new(),
                dcov: Vec::new(),
                num_outputs: 1,
            });
        }
        let d = x.ncols();
        let (dmean, dcov) = match jacs {
            None => (dmean_t, dcov_t),
            Some(jacs) => {
                let mut dmean = vec![Vector::zeros(q); q * d];
                let mut dcov = vec![Matrix::zeros(q, q); q * d];
                for i in 0..q {
                    for l in 0..dt {
                        for j in 0..d {
                            let f = jacs[i][(l, j)];
                            if f != 0.0 {
                                dmean[i * d + j].axpy(f, &dmean_t[i * dt + l], 1.0);
                                dcov[i * d + j] += &dcov_t[i * dt + l] * f;
                            }
                        }
                    }
                }
                (dmean, dcov)
            }
        };
        Ok(JointPosteriorGrad {
            mean,
            covariance,
            dmean,
            dcov,
            num_outputs: 1,
        })
    }

    // -- conditioning ---------------------------------------------------------

    /// BoTorch's `condition_on_observations(X, Y)` with unbatched pairwise
    /// observations `Y` (comparisons among the new points `X`, `n' × d`): a
    /// copy of the model with the new datapoints appended to the
    /// consolidated ones (indices shifted by `n`), consolidated again, with
    /// the Laplace approximation recomputed in the model's mode (`draws`:
    /// the `n + n'` initial perturbations of the fresh start, see
    /// [`PairwiseGP::update`]). A model without data takes `(X, Y)` as its
    /// data.
    ///
    /// # Errors
    /// See [`PairwiseGP::new`].
    pub fn condition_on_observations(
        &self,
        x: &Matrix,
        comps: &[(usize, usize)],
        draws: Option<&[f64]>,
    ) -> Result<PairwiseGP> {
        let mut new = self.clone();
        let (dp, cp) = new.appended(x, comps);
        let consolidate = self
            .data
            .as_ref()
            .is_none_or(|d| d.consolidated_indices.is_some());
        new.set_train_data_impl(dp, cp, consolidate, true, draws)?;
        Ok(new)
    }

    /// The model with the datapoints `x` and comparisons `comps` (indices
    /// into `x`) appended, without recomputing the Laplace approximation
    /// (see [`PairwiseGP::update`]; the next update starts fresh unless the
    /// number of datapoints is unchanged). In `batch_mode` (BoTorch's
    /// batched `condition_on_observations`, e.g. for fantasies) the data
    /// are not consolidated; otherwise they are if the model's are.
    ///
    /// # Errors
    /// See [`PairwiseGP::new`].
    pub fn with_observations(
        &self,
        x: &Matrix,
        comps: &[(usize, usize)],
        batch_mode: bool,
    ) -> Result<PairwiseGP> {
        let mut new = self.clone();
        let (dp, cp) = new.appended(x, comps);
        let consolidate = !batch_mode
            && self
                .data
                .as_ref()
                .is_none_or(|d| d.consolidated_indices.is_some());
        new.set_train_data_impl(dp, cp, consolidate, false, None)?;
        Ok(new)
    }

    /// The datapoints and comparisons after appending `(x, comps)`.
    fn appended(&self, x: &Matrix, comps: &[(usize, usize)]) -> (Matrix, Vec<(usize, usize)>) {
        match &self.data {
            None => (x.clone(), comps.to_vec()),
            Some(d) => {
                let n = d.datapoints.nrows();
                let mut dp = Matrix::zeros(n + x.nrows(), self.dim);
                dp.rows_mut(0, n).copy_from(&d.datapoints);
                dp.rows_mut(n, x.nrows()).copy_from(x);
                let mut cp = d.comparisons.pairs().to_vec();
                cp.extend(comps.iter().map(|&(a, b)| (a + n, b + n)));
                (dp, cp)
            }
        }
    }

    // -- fitting --------------------------------------------------------------

    /// BoTorch's `sample_all_priors` on the covariance module.
    fn sample_priors(&self, rng: &mut StdRng) -> Vec<f64> {
        let mut raw = self.raw.clone();
        let kernel_parameters = num_raw_parameters(&self.covar_module);
        match &self.covar_module {
            CovarModule::Standard(spec) => {
                let scale = spec.scale.expect("ScaleKernel");
                if let Some(prior) = scale.outputscale_prior {
                    raw[0] = scale
                        .outputscale_constraint
                        .inverse_transform(prior.sample(rng));
                }
                if let Some(prior) = spec.lengthscale_prior {
                    for v in raw[1..kernel_parameters].iter_mut() {
                        *v = spec
                            .lengthscale_constraint
                            .inverse_transform(prior.sample(rng));
                    }
                }
            }
            CovarModule::Tree(tree) => {
                crate::kernels::tree::sample_tree_priors(tree, &mut raw[..kernel_parameters], rng);
            }
        }
        if let Some(chain) = &self.input_transform {
            for (value, prior) in raw[kernel_parameters..]
                .iter_mut()
                .zip(chain.param_priors())
            {
                if let Some(prior) = prior {
                    *value = prior.sample(rng);
                }
            }
        }
        raw
    }

    /// `fit_gpytorch_mll` with a `PairwiseLaplaceMarginalLogLikelihood`
    /// (BoTorch's `_fit_fallback` with `fit_gpytorch_mll_scipy`): L-BFGS-B
    /// on the raw hyperparameters, every evaluation recomputing the Laplace
    /// approximation warm-started from the previous one (BoTorch's training
    /// forward pass), with retries from prior samples. On success the model
    /// keeps the state of the last evaluation at the optimum and is put in
    /// eval mode.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for a model without data; the errors of
    /// [`crate::fit::fit_gpytorch_mll`].
    pub fn fit(&mut self, options: FitOptions) -> Result<FitResult> {
        if self.data.is_none() {
            return Err(BotorchError::Unsupported(
                "datapoints and comparisons cannot be None in training mode.".into(),
            ));
        }
        self.set_training(true);
        let bounds = self.raw_parameter_bounds();
        let x_init = self.raw.clone();
        let warm = std::cell::RefCell::new(self.x0.clone().expect("warm start"));
        let last: std::cell::RefCell<Option<LaplaceState>> = std::cell::RefCell::new(None);
        let (result, x) = {
            let this: &Self = self;
            minimize_with_retries(
                &x_init,
                &bounds,
                options,
                |raw| {
                    let x0 = warm.borrow().clone();
                    match this.evaluate_trial(raw, &x0, true) {
                        Ok((v, g, st)) => {
                            *warm.borrow_mut() = st.x0.clone();
                            *last.borrow_mut() = Some(st);
                            Ok((v, g.expect("gradient")))
                        }
                        // torch.linalg.solve's singular error: NaN (BoTorch's
                        // `_handle_numerical_errors`).
                        Err(BotorchError::Runtime(m)) if m == SINGULAR => {
                            Ok((f64::NAN, vec![f64::NAN; raw.len()]))
                        }
                        Err(e) => Err(e),
                    }
                },
                |rng| this.sample_priors(rng),
            )?
        };
        self.set_raw_parameters(&x)?;
        // Install the fitted transform coefficients as well as concentrations
        // even when the final Laplace state can be reused without a new solve.
        self.transformed_datapoints()?;
        self.x0 = Some(warm.into_inner());
        let last = last.into_inner();
        match last {
            Some(st) if st.raw == x => {
                self.state = Some(Arc::new(st));
            }
            _ => self.update(None)?,
        }
        self.set_training(false);
        Ok(result)
    }
}

impl Fittable for PairwiseGP {
    type Output = FitResult;

    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        self.fit(options)
    }
}

impl Model for PairwiseGP {
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_outputs(&self) -> usize {
        1
    }

    /// The posterior (`observation_noise` is ignored: noise is not
    /// identifiable from the scale in probit models).
    fn posterior(&self, x: &Matrix, _observation_noise: bool) -> Result<Posterior> {
        let g = self.posterior_impl(x, false)?;
        Ok(Posterior::single_output(g.mean, g.covariance))
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.posterior_impl(x, true)
    }
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

/// A batch of [`PairwiseGP`]s (BoTorch's `PairwiseGP` with batched data,
/// e.g. the fantasy models of `fantasize`): every element has its own data
/// and Laplace approximation and is not consolidated; the hyperparameters
/// are a global vector of which every element uses the entries
/// [`BatchedPairwiseGP::param_map`] (shared by all elements, or independent
/// parameters such as the default batched lengthscales and outputscales).
#[derive(Clone, Debug)]
pub struct BatchedPairwiseGP {
    batch_shape: Vec<usize>,
    elements: Vec<PairwiseGP>,
    raw: Vec<f64>,
    param_map: Vec<Vec<usize>>,
    shared_input_transform: Option<ChainedInputTransform>,
}

fn pooled_inputs(inputs: &[&Matrix], dim: usize) -> Matrix {
    let rows = inputs.iter().map(|x| x.nrows()).sum();
    let mut pooled = Matrix::zeros(rows, dim);
    let mut offset = 0;
    for input in inputs {
        pooled.rows_mut(offset, input.nrows()).copy_from(input);
        offset += input.nrows();
    }
    pooled
}

impl BatchedPairwiseGP {
    fn check_learned_input_parameters(&self) -> Result<()> {
        if self.elements[0]
            .input_transform
            .as_ref()
            .is_some_and(|t| t.num_params() > 0)
        {
            return Err(BotorchError::Unsupported(
                "fitting or changing learned input parameters of batched PairwiseGP requires shared/per-member parameter mapping".into(),
            ));
        }
        Ok(())
    }
    /// A batch of models with the datapoints and comparisons of every
    /// element (row-major in `batch_shape`), global raw hyperparameters
    /// `raw` and the global index of every element's raw parameter
    /// (`param_map`, `None`: all shared). `draws`: the initial
    /// perturbations of all elements (`B·n`, row-major).
    ///
    /// # Errors
    /// See [`PairwiseGP::new`]; [`BotorchError::DimensionMismatch`] for
    /// inconsistent shapes.
    pub fn new(
        batch_shape: Vec<usize>,
        data: Vec<(Matrix, Vec<(usize, usize)>)>,
        dim: usize,
        mut options: PairwiseGPOptions,
        raw: Option<Vec<f64>>,
        param_map: Option<Vec<Vec<usize>>>,
        draws: Option<&[f64]>,
    ) -> Result<Self> {
        let b: usize = batch_shape.iter().product();
        if data.len() != b || b == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {b} batch elements, got {}",
                data.len()
            )));
        }
        let n = data[0].0.nrows();
        let m = data[0].1.len();
        if data.iter().any(|(x, c)| x.nrows() != n || c.len() != m) {
            return Err(BotorchError::DimensionMismatch(
                "all batch elements must have the same numbers of datapoints and comparisons"
                    .into(),
            ));
        }
        if let Some(d) = draws {
            if d.len() != b * n {
                return Err(BotorchError::InvalidArgument(format!(
                    "expected {} initial draws, got {}",
                    b * n,
                    d.len()
                )));
            }
        }
        if options
            .input_transform
            .as_ref()
            .is_some_and(|t| t.num_params() > 0)
        {
            return Err(BotorchError::Unsupported(
                "batched PairwiseGP learned input transforms require shared/per-member parameter mapping".into(),
            ));
        }
        // Shared coefficient-learning transforms reduce over every input
        // batch, as Normalize(batch_shape=()) does in BoTorch. Individual
        // Laplace solves must reuse these pooled coefficients.
        let mut shared_input_transform = options.input_transform.clone();
        if let Some(chain) = &mut shared_input_transform {
            let inputs: Vec<&Matrix> = data.iter().map(|(x, _)| x).collect();
            chain.fit(&pooled_inputs(&inputs, dim))?;
            options.input_transform = Some(chain.frozen());
        }
        let template = PairwiseGP::new_impl(None, dim, options.clone(), None, false)?;
        let k = template.raw.len();
        let param_map = param_map.unwrap_or_else(|| vec![(0..k).collect(); b]);
        let raw = raw.unwrap_or_else(|| template.raw.clone());
        Self::check_map(&param_map, b, k, raw.len())?;
        let mut elements = Vec::with_capacity(b);
        for (e, (x, c)) in data.into_iter().enumerate() {
            let local: Vec<f64> = param_map[e].iter().map(|&i| raw[i]).collect();
            let opts = PairwiseGPOptions {
                raw_parameters: Some(local),
                covar_module: Some(template.covar_module.clone()),
                seed: options.seed.wrapping_add(e as u64),
                ..options.clone()
            };
            let el = PairwiseGP::new_impl(
                Some((x, c)),
                dim,
                opts,
                draws.map(|d| &d[e * n..(e + 1) * n]),
                false,
            )?;
            elements.push(el);
        }
        Ok(Self {
            batch_shape,
            elements,
            raw,
            param_map,
            shared_input_transform,
        })
    }

    fn check_map(map: &[Vec<usize>], b: usize, k: usize, len: usize) -> Result<()> {
        if map.len() != b
            || map
                .iter()
                .any(|m| m.len() != k || m.iter().any(|&i| i >= len))
        {
            return Err(BotorchError::DimensionMismatch(
                "invalid parameter map of a batched PairwiseGP".into(),
            ));
        }
        Ok(())
    }

    /// The batch from elements sharing all hyperparameters (e.g. the
    /// fantasies of an unbatched model).
    pub fn from_elements(batch_shape: Vec<usize>, elements: Vec<PairwiseGP>) -> Result<Self> {
        let shared = elements.first().and_then(|e| e.input_transform.clone());
        Self::from_elements_with_transform(batch_shape, elements, shared)
    }

    /// A batch retaining the original shared transform's coefficient-learning
    /// flags (individual member views may have frozen coefficients).
    pub fn from_elements_with_transform(
        batch_shape: Vec<usize>,
        elements: Vec<PairwiseGP>,
        shared_input_transform: Option<ChainedInputTransform>,
    ) -> Result<Self> {
        let b: usize = batch_shape.iter().product();
        if elements.len() != b || b == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {b} batch elements, got {}",
                elements.len()
            )));
        }
        let raw = elements[0].raw.clone();
        let k = raw.len();
        Ok(Self {
            batch_shape,
            elements,
            raw,
            param_map: vec![(0..k).collect(); b],
            shared_input_transform,
        })
    }

    /// The shared input transform, with its original learning flags.
    pub fn shared_input_transform(&self) -> Option<&ChainedInputTransform> {
        self.shared_input_transform.as_ref()
    }

    fn prepare_shared_transform(&mut self) -> Result<()> {
        if let Some(chain) = &mut self.shared_input_transform {
            if self.elements[0].training {
                let inputs: Vec<&Matrix> = self
                    .elements
                    .iter()
                    .map(|e| &e.data.as_ref().expect("batch member has data").datapoints)
                    .collect();
                chain.fit(&pooled_inputs(&inputs, self.elements[0].dim))?;
            }
            let frozen = chain.frozen();
            for element in &mut self.elements {
                element.input_transform = Some(frozen.clone());
            }
        }
        Ok(())
    }

    /// The batch shape.
    pub fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    /// The elements (row-major).
    pub fn elements(&self) -> &[PairwiseGP] {
        &self.elements
    }

    /// The elements, mutably (keep the hyperparameters consistent with
    /// [`BatchedPairwiseGP::set_raw_parameters`]).
    pub fn elements_mut(&mut self) -> &mut [PairwiseGP] {
        &mut self.elements
    }

    /// The global raw hyperparameters.
    pub fn raw_parameters(&self) -> &[f64] {
        &self.raw
    }

    /// The global index of every element's raw parameters.
    pub fn param_map(&self) -> &[Vec<usize>] {
        &self.param_map
    }

    /// Sets the global raw hyperparameters (no recomputation).
    ///
    /// # Errors
    /// See [`PairwiseGP::set_raw_parameters`].
    pub fn set_raw_parameters(&mut self, raw: &[f64]) -> Result<()> {
        self.check_learned_input_parameters()?;
        if raw.len() != self.raw.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                self.raw.len(),
                raw.len()
            )));
        }
        for (e, el) in self.elements.iter_mut().enumerate() {
            let local: Vec<f64> = self.param_map[e].iter().map(|&i| raw[i]).collect();
            el.set_raw_parameters(&local)?;
        }
        self.raw = raw.to_vec();
        Ok(())
    }

    /// Sets train or eval mode of every element.
    pub fn set_training(&mut self, training: bool) {
        for e in &mut self.elements {
            e.set_training(training);
        }
        if let Some(chain) = self.shared_input_transform.take() {
            let steps = chain.training().len();
            self.shared_input_transform = Some(
                chain
                    .with_training(vec![training; steps])
                    .expect("one mode flag per transform step"),
            );
        }
    }

    /// Recomputes the Laplace approximations (`draws`: `B·n` values for
    /// elements that need a fresh start, row-major; see
    /// [`PairwiseGP::update`]).
    ///
    /// # Errors
    /// See [`PairwiseGP::update`].
    pub fn update(&mut self, draws: Option<&[f64]>) -> Result<()> {
        self.prepare_shared_transform()?;
        let n = self.elements[0].num_datapoints();
        for (e, el) in self.elements.iter_mut().enumerate() {
            el.update(draws.map(|d| &d[e * n..(e + 1) * n]))?;
        }
        Ok(())
    }

    /// Whether the elements need a fresh start.
    pub fn needs_fresh_start(&self) -> bool {
        self.elements.iter().any(PairwiseGP::needs_fresh_start)
    }

    /// The sum of the elements' MLLs (priors counted once, on the global
    /// parameters) and its gradient with respect to the global raw
    /// parameters, warm-started from every element's solution; with the
    /// new states.
    ///
    /// # Errors
    /// See [`PairwiseGP::mll_value_and_grad`].
    pub fn mll_value_and_grad(
        &self,
        raw: &[f64],
        with_grad: bool,
    ) -> Result<(f64, Option<Vec<f64>>, Vec<LaplaceState>)> {
        self.check_learned_input_parameters()?;
        let mut total = 0.0;
        let mut grad = vec![0.0; raw.len()];
        let mut states = Vec::with_capacity(self.elements.len());
        for (e, el) in self.elements.iter().enumerate() {
            let local: Vec<f64> = self.param_map[e].iter().map(|&i| raw[i]).collect();
            let x_tf = el.training_inputs_tf()?;
            let x0 = el.x0.clone().expect("warm start");
            let p = el.problem(&x_tf);
            let lap = laplace(&p, &el.covar_module, &local, Start::Warm(&x0))?;
            check_training_posterior(&lap.state, p.jitter)?;
            let (data, w) = mll_data(&p, &lap, with_grad)?;
            total += data;
            if let Some(w) = w {
                // The data gradient only (no priors): contract with a zero
                // prior by subtracting the prior part.
                let (_, g_all) = priors_and_raw_grad(
                    &el.covar_module,
                    &lap.covar_built,
                    &lap.prepared,
                    &local,
                    Some(&w),
                    &lap.state.nan_grads,
                )?;
                let (_, g_prior) = prior_raw_grad(&el.covar_module, &local)?;
                for (j, &gi) in self.param_map[e].iter().enumerate() {
                    grad[gi] += g_all.as_ref().expect("gradient")[j] - g_prior[j];
                }
            }
            states.push(lap.state);
        }
        let (lp, gp) = self.global_priors(raw)?;
        total += lp;
        for (g, p) in grad.iter_mut().zip(&gp) {
            *g += p;
        }
        Ok((total, with_grad.then_some(grad), states))
    }

    /// The log priors of the global parameters and their raw gradients.
    fn global_priors(&self, raw: &[f64]) -> Result<(f64, Vec<f64>)> {
        let el = &self.elements[0];
        let identity = self
            .param_map
            .iter()
            .all(|m| m.iter().enumerate().all(|(j, &i)| i == j))
            && raw.len() == el.raw.len();
        if identity {
            return prior_raw_grad(&el.covar_module, raw);
        }
        let CovarModule::Standard(_) = &el.covar_module else {
            return Err(BotorchError::Unsupported(
                "batched kernel trees with per-element parameters are not supported".into(),
            ));
        };
        // Every global parameter once, with the prior of a local parameter
        // that maps to it.
        let mut owner = vec![None; raw.len()];
        for m in &self.param_map {
            for (j, &i) in m.iter().enumerate() {
                owner[i].get_or_insert(j);
            }
        }
        let mut lp = 0.0;
        let mut grad = vec![0.0; raw.len()];
        for (i, o) in owner.iter().enumerate() {
            let Some(j) = *o else { continue };
            let (prior, constraint) = standard_param(&el.covar_module, j);
            let v = constraint.transform(raw[i]);
            if let Some(p) = prior {
                check_support(&p, v)?;
                lp += p.log_prob(v);
                grad[i] = raw_gradient(
                    0.0,
                    Some(p),
                    v,
                    constraint.transform_derivative(raw[i]),
                    1.0,
                );
            }
        }
        Ok((lp, grad))
    }

    /// `fit_gpytorch_mll` on the batch: one L-BFGS-B problem over the
    /// global raw parameters (BoTorch fits a batched `PairwiseGP` with one
    /// closure; its MLL is the sum over the batch).
    ///
    /// # Errors
    /// See [`PairwiseGP::fit`].
    pub fn fit(&mut self, options: FitOptions) -> Result<FitResult> {
        self.check_learned_input_parameters()?;
        self.set_training(true);
        self.prepare_shared_transform()?;
        let bounds = self.global_bounds();
        let x_init = self.raw.clone();
        let warm: std::cell::RefCell<Vec<Vec<f64>>> = std::cell::RefCell::new(
            self.elements
                .iter()
                .map(|e| e.x0.clone().expect("warm start"))
                .collect(),
        );
        let last: std::cell::RefCell<Option<(Vec<f64>, Vec<LaplaceState>)>> =
            std::cell::RefCell::new(None);
        let (result, x) = {
            let this: &Self = self;
            minimize_with_retries(
                &x_init,
                &bounds,
                options,
                |raw| {
                    let mut probe = this.clone();
                    for (el, x0) in probe.elements.iter_mut().zip(warm.borrow().iter()) {
                        el.x0 = Some(x0.clone());
                    }
                    match probe.mll_value_and_grad(raw, true) {
                        Ok((v, g, states)) => {
                            *warm.borrow_mut() = states.iter().map(|s| s.x0.clone()).collect();
                            *last.borrow_mut() = Some((raw.to_vec(), states));
                            Ok((v, g.expect("gradient")))
                        }
                        Err(BotorchError::Runtime(m)) if m == SINGULAR => {
                            Ok((f64::NAN, vec![f64::NAN; raw.len()]))
                        }
                        Err(e) => Err(e),
                    }
                },
                |rng| {
                    let el = &this.elements[0];
                    let mut raw = this.raw.clone();
                    for m in &this.param_map {
                        let s = el.sample_priors(rng);
                        for (j, &i) in m.iter().enumerate() {
                            raw[i] = s[j];
                        }
                    }
                    raw
                },
            )?
        };
        self.set_raw_parameters(&x)?;
        for (el, x0) in self.elements.iter_mut().zip(warm.into_inner()) {
            el.x0 = Some(x0);
        }
        match last.into_inner() {
            Some((raw, states)) if raw == x => {
                for (el, st) in self.elements.iter_mut().zip(states) {
                    el.state = Some(Arc::new(st));
                }
            }
            _ => {
                for el in &mut self.elements {
                    el.update(None)?;
                }
            }
        }
        self.set_training(false);
        Ok(result)
    }

    /// The optimizer bounds of the global parameters.
    pub fn global_bounds(&self) -> Vec<(f64, f64)> {
        let local = self.elements[0].raw_parameter_bounds();
        let mut out = vec![(f64::NEG_INFINITY, f64::INFINITY); self.raw.len()];
        for m in &self.param_map {
            for (j, &i) in m.iter().enumerate() {
                out[i] = local[j];
            }
        }
        out
    }
}

/// The prior and the constraint of the local parameter `j` of a standard
/// module (`0`: outputscale, `1..`: lengthscales).
fn standard_param(module: &CovarModule, j: usize) -> (Option<Prior>, crate::Constraint) {
    let CovarModule::Standard(spec) = module else {
        unreachable!("standard module")
    };
    if j == 0 {
        let s = spec.scale.expect("ScaleKernel");
        (s.outputscale_prior, s.outputscale_constraint)
    } else {
        (spec.lengthscale_prior, spec.lengthscale_constraint)
    }
}

/// The log priors of a module at raw values and their raw gradients.
fn prior_raw_grad(module: &CovarModule, raw: &[f64]) -> Result<(f64, Vec<f64>)> {
    match module {
        CovarModule::Standard(_) => {
            let mut lp = 0.0;
            let mut grad = vec![0.0; raw.len()];
            for (j, &r) in raw.iter().enumerate() {
                let (prior, constraint) = standard_param(module, j);
                let v = constraint.transform(r);
                if let Some(p) = prior {
                    check_support(&p, v)?;
                    lp += p.log_prob(v);
                    grad[j] =
                        raw_gradient(0.0, Some(p), v, constraint.transform_derivative(r), 1.0);
                }
            }
            Ok((lp, grad))
        }
        CovarModule::Tree(node) => {
            let tree = TreeKernel::build(node, raw, input_dim_of(node))?;
            let (lp, gp) = tree.log_prior_and_grad()?;
            let derivs = node.transform_derivatives(raw);
            Ok((lp, gp.iter().zip(&derivs).map(|(g, d)| g * d).collect()))
        }
    }
}

/// An input dimension that the tree accepts (for evaluating its priors).
fn input_dim_of(node: &crate::kernels::KernelNode) -> usize {
    (1..=64).find(|&d| node.validate(d).is_ok()).unwrap_or(1)
}

impl Fittable for BatchedPairwiseGP {
    type Output = FitResult;

    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        self.fit(options)
    }
}

impl Model for BatchedPairwiseGP {
    fn dim(&self) -> usize {
        self.elements[0].dim
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    fn batch_element(&self, b: usize) -> Option<&dyn Model> {
        self.elements.get(b).map(|e| e as &dyn Model)
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        if self.elements.len() == 1 {
            return self.elements[0].posterior(x, observation_noise);
        }
        Err(BotorchError::Unsupported(
            "the posterior of a batched PairwiseGP is evaluated per batch element".into(),
        ))
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        if self.elements.len() == 1 {
            return self.elements[0].joint_posterior_with_grad(x);
        }
        Err(BotorchError::Unsupported(
            "the posterior of a batched PairwiseGP is evaluated per batch element".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn data() -> (Matrix, Vec<(usize, usize)>) {
        let x = Matrix::from_row_slice(
            6,
            2,
            &[
                0.1, 0.2, 0.5, 0.9, 0.3, 0.4, 0.8, 0.1, 0.65, 0.55, 0.2, 0.75,
            ],
        );
        (x, vec![(0, 1), (2, 3), (1, 4), (5, 2), (4, 0)])
    }

    pub(super) fn draws(n: usize) -> Vec<f64> {
        (0..n).map(|i| ((i as f64) * 0.7).sin()).collect()
    }

    #[test]
    fn shared_batch_normalization_uses_pooled_data_and_retains_learning_flags() {
        use crate::transforms::input::InputTransformStep;
        use crate::Normalize;

        let (x, comps) = data();
        let shifted = x.add_scalar(2.0);
        let options = PairwiseGPOptions {
            input_transform: Some(
                ChainedInputTransform::new(vec![InputTransformStep::Normalize(Normalize::new(2))])
                    .unwrap(),
            ),
            ..Default::default()
        };
        let mut batch = BatchedPairwiseGP::new(
            vec![2],
            vec![(x.clone(), comps.clone()), (shifted.clone(), comps)],
            2,
            options,
            None,
            None,
            Some(&draws(12)),
        )
        .unwrap();
        let mut expected = Normalize::new(2);
        expected.fit(&pooled_inputs(&[&x, &shifted], 2)).unwrap();
        for element in batch.elements() {
            let raw = &element.data().unwrap().datapoints;
            assert_eq!(
                element.training_inputs_tf().unwrap(),
                expected.transform(raw).unwrap()
            );
        }
        // A training-mode conditioned batch must relearn the shared bounds
        // from all of its augmented data, even though member views are frozen.
        let new_x = Matrix::from_row_slice(2, 2, &[4.0, 3.0, 5.0, 4.0]);
        let shared = batch.shared_input_transform().cloned();
        let elements = batch
            .elements()
            .iter()
            .map(|e| e.with_observations(&new_x, &[(0, 1)], true).unwrap())
            .collect();
        batch = BatchedPairwiseGP::from_elements_with_transform(vec![2], elements, shared).unwrap();
        batch.update(None).unwrap();
        let raw: Vec<&Matrix> = batch
            .elements()
            .iter()
            .map(|e| &e.data().unwrap().datapoints)
            .collect();
        expected.fit(&pooled_inputs(&raw, 2)).unwrap();
        for (element, input) in batch.elements().iter().zip(raw) {
            assert_eq!(
                element.training_inputs_tf().unwrap(),
                expected.transform(input).unwrap()
            );
        }
    }

    #[test]
    fn mll_gradient_matches_finite_differences() {
        for lik in [PairwiseLikelihood::Probit, PairwiseLikelihood::Logit] {
            let (x, c) = data();
            let opts = PairwiseGPOptions {
                likelihood: lik,
                ..Default::default()
            };
            let m = PairwiseGP::new(Some((x, c)), 2, opts, Some(&draws(6))).unwrap();
            let raw: Vec<f64> = vec![-5.0, 0.4, 0.7];
            let (_, g, st) = m.mll_value_and_grad(&raw, true).unwrap();
            let g = g.unwrap();
            // BoTorch's gradient differentiates the two Newton steps from
            // the fsolve solution held fixed.
            let x_tf = m.training_inputs_tf().unwrap();
            let p = m.problem(&x_tf);
            let fixed = |r: &[f64]| -> f64 {
                let lap = laplace(&p, &m.covar_module, r, Start::Solved(&st.x0)).unwrap();
                let (v, _) = mll_data(&p, &lap, false).unwrap();
                let (lp, _) = priors_and_raw_grad(
                    &m.covar_module,
                    &lap.covar_built,
                    &lap.prepared,
                    r,
                    None,
                    &[],
                )
                .unwrap();
                v + lp
            };
            for j in 0..raw.len() {
                let h = 1e-6;
                let mut a = raw.clone();
                let mut b = raw.clone();
                a[j] += h;
                b[j] -= h;
                let fa = fixed(&a);
                let fb = fixed(&b);
                let fd = (fa - fb) / (2.0 * h);
                println!("{lik:?} {j}: fd {fd:.12e} an {:.12e}", g[j]);
                assert!(
                    (fd - g[j]).abs() < 1e-6 * (1.0 + g[j].abs()),
                    "{lik:?} param {j}: fd {fd} vs {}",
                    g[j]
                );
            }
        }
    }

    #[test]
    fn posterior_gradients_match_finite_differences() {
        let (x, c) = data();
        let mut m = PairwiseGP::new(Some((x, c)), 2, Default::default(), Some(&draws(6))).unwrap();
        m.set_training(false);
        let xt = Matrix::from_row_slice(3, 2, &[0.15, 0.3, 0.6, 0.6, 0.4, 0.05]);
        let g = m.joint_posterior_with_grad(&xt).unwrap();
        for j in 0..6 {
            let (p, k) = (j / 2, j % 2);
            let h = 1e-6;
            let mut a = xt.clone();
            let mut b = xt.clone();
            a[(p, k)] += h;
            b[(p, k)] -= h;
            let pa = m.posterior(&a, false).unwrap();
            let pb = m.posterior(&b, false).unwrap();
            let dm = (&pa.mean - &pb.mean) / (2.0 * h);
            let dc = (&pa.covariance - &pb.covariance) / (2.0 * h);
            assert!((dm - &g.dmean[j]).amax() < 1e-6, "mean {j}");
            assert!((dc - &g.dcov[j]).amax() < 1e-6, "cov {j}");
        }
    }

    #[test]
    fn duplicates_are_consolidated() {
        let x = Matrix::from_row_slice(4, 1, &[0.0, 1.0, 0.00001, 2.0]);
        let c = consolidate_duplicates(&x, &[(0, 1), (2, 3)], 0.0, 1e-4).unwrap();
        assert_eq!(c.datapoints.nrows(), 3);
        assert_eq!(c.comparisons, vec![(0, 1), (0, 2)]);
        assert_eq!(c.indices, vec![0, 1, 0, 2]);
    }
}
