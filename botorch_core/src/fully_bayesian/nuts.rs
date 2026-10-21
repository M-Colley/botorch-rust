//! The No-U-Turn sampler of NumPyro (`numpyro.infer.NUTS` run by
//! `numpyro.infer.MCMC`), which BoTorch's `fit_fully_bayesian_model_nuts`
//! uses to sample the hyperparameters of its fully Bayesian models.
//!
//! This is a port of NumPyro 0.21's algorithm, step by step:
//!
//! * Hamiltonian dynamics with the velocity Verlet integrator and a dense or
//!   diagonal Euclidean metric (`velocity_verlet`,
//!   `euclidean_kinetic_energy`);
//! * the iterative NUTS tree builder of NumPyro (`build_tree`,
//!   `_iterative_build_subtree`, `_double_tree`, `_combine_tree`): biased
//!   progressive sampling between doublings, uniform progressive sampling
//!   inside a subtree, the generalized U-turn criterion of Betancourt
//!   (2017) on the momentum checkpoints of the subtree
//!   (`_leaf_idx_to_ckpt_idxs`, `_is_iterative_turning`) and on the whole
//!   tree, divergences at an energy error above `max_delta_energy = 1000`;
//! * the warmup adaptation of NumPyro (`warmup_adapter`): dual averaging of
//!   the log step size (`t0 = 10`, `κ = 0.75`, `γ = 0.05`, restarted at the
//!   end of every mass-matrix window around `log(10 ε)`), Stan's windows
//!   (`build_adaptation_schedule`: buffers of 75 / 50 and a first window of
//!   25 doubling iterations, or 15% / 75% / 10% of short warmups, a single
//!   window below 20 iterations), Welford estimates of the (co)variance of
//!   the samples of a window with Stan's regularization
//!   `n / (n + 5) Σ + 10⁻³ · 5 / (n + 5) I`, and the final step size
//!   `exp(x̄)` of the averaged iterate at the last warmup iteration.
//!
//! NumPyro draws its random numbers with JAX's counter-based PRNG. The
//! sampler takes them from a [`NutsRandom`] source instead: the
//! standard-normal momentum draws (`random.normal` of the momentum
//! generator, the first of them at initialization), and the uniform draws
//! behind NumPyro's `random.bernoulli` calls (`uniform < p`: the direction of
//! every doubling with `p = 1/2`, the transitions inside subtrees and between
//! trees). [`RecordedDraws`] replays draws recorded from NumPyro (then the
//! sampler retraces NumPyro's chain); [`SeededDraws`] draws them from a
//! seeded generator (botorch_rs's `fit_fully_bayesian_model_nuts`).

use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::error::{BotorchError, Result};
use crate::Matrix;

/// A differentiable potential energy `U(z)` (the negative log density of
/// the unconstrained parameters `z`).
pub trait Potential {
    /// The number of unconstrained parameters.
    fn dim(&self) -> usize;
    /// `U(z)` and its gradient. Non-finite values are allowed (e.g. a
    /// covariance matrix that is not positive definite): the sampler treats
    /// them as NumPyro does (a diverging transition).
    fn value_and_grad(&self, z: &[f64]) -> (f64, Vec<f64>);
}

/// The kind of a uniform draw (NumPyro's `random.bernoulli` calls).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniformKind {
    /// The direction of a doubling (`p = 1/2`; right if `u < 1/2`).
    Direction,
    /// A transition inside a subtree (uniform progressive sampling).
    SubtreeTransition,
    /// A transition between the tree and a new subtree (biased progressive
    /// sampling).
    TreeTransition,
}

/// The source of the random numbers of the sampler.
pub trait NutsRandom {
    /// `n` standard-normal draws (a momentum before its scaling by the
    /// square root of the mass matrix).
    fn normal(&mut self, n: usize) -> Result<Vec<f64>>;
    /// A uniform draw on `[0, 1)` for a Bernoulli trial with success
    /// probability `p` (success if the draw is below `p`).
    fn uniform(&mut self, kind: UniformKind, p: f64) -> Result<f64>;
}

/// Draws from a seeded generator (xoshiro256++).
#[derive(Clone, Debug)]
pub struct SeededDraws {
    rng: Xoshiro256PlusPlus,
}

impl SeededDraws {
    /// A generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// A uniform draw on `[-radius, radius)` (NumPyro's `init_to_uniform`).
    pub fn symmetric_uniform(&mut self, radius: f64) -> f64 {
        let u: f64 = self.rng.random::<f64>();
        u * 2.0 * radius - radius
    }
}

impl NutsRandom for SeededDraws {
    fn normal(&mut self, n: usize) -> Result<Vec<f64>> {
        Ok((0..n)
            .map(|_| StandardNormal.sample(&mut self.rng))
            .collect())
    }

    fn uniform(&mut self, _kind: UniformKind, _p: f64) -> Result<f64> {
        Ok(self.rng.random::<f64>())
    }
}

/// One recorded draw of NumPyro's sampler.
#[derive(Clone, Debug, PartialEq)]
pub enum RecordedDraw {
    /// A standard-normal momentum draw.
    Normal(Vec<f64>),
    /// A uniform draw and the success probability NumPyro compared it with.
    Uniform {
        /// The draw.
        u: f64,
        /// NumPyro's probability `p` (the trial succeeded if `u < p`).
        p: f64,
    },
}

/// Replays draws recorded from NumPyro, in order, and records how closely the
/// sampler's Bernoulli probabilities follow NumPyro's.
#[derive(Clone, Debug)]
pub struct RecordedDraws {
    draws: Vec<RecordedDraw>,
    position: usize,
    /// The largest `|p - p_numpyro|` of the uniform draws so far.
    pub max_probability_error: f64,
    /// The index of the first draw whose outcome differs from NumPyro's
    /// (`u < p` versus `u < p_numpyro`), if any: from there on the chain is
    /// no longer NumPyro's.
    pub first_flip: Option<usize>,
    /// The smallest `|u - p_numpyro|` over the uniform draws (how close a
    /// trial came to flipping).
    pub min_margin: f64,
}

impl RecordedDraws {
    /// Replays `draws`.
    pub fn new(draws: Vec<RecordedDraw>) -> Self {
        Self {
            draws,
            position: 0,
            max_probability_error: 0.0,
            first_flip: None,
            min_margin: f64::INFINITY,
        }
    }

    /// The number of draws consumed so far.
    pub fn consumed(&self) -> usize {
        self.position
    }

    /// The number of recorded draws.
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Whether there are no recorded draws.
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    fn next(&mut self, what: &str) -> Result<&RecordedDraw> {
        let i = self.position;
        let draw = self.draws.get(i).ok_or_else(|| {
            BotorchError::InvalidArgument(format!(
                "the recorded draws ran out at draw {i} (expected a {what} draw)"
            ))
        })?;
        self.position += 1;
        Ok(draw)
    }
}

impl NutsRandom for RecordedDraws {
    fn normal(&mut self, n: usize) -> Result<Vec<f64>> {
        let i = self.position;
        match self.next("normal")? {
            RecordedDraw::Normal(v) if v.len() == n => Ok(v.clone()),
            other => Err(BotorchError::InvalidArgument(format!(
                "recorded draw {i} is {other:?}, but the sampler needs {n} normal draws"
            ))),
        }
    }

    fn uniform(&mut self, _kind: UniformKind, p: f64) -> Result<f64> {
        let i = self.position;
        let (u, p_ref) = match self.next("uniform")? {
            RecordedDraw::Uniform { u, p } => (*u, *p),
            other => {
                return Err(BotorchError::InvalidArgument(format!(
                    "recorded draw {i} is {other:?}, but the sampler needs a uniform draw"
                )))
            }
        };
        let err = if p.is_nan() && p_ref.is_nan() {
            0.0
        } else {
            (p - p_ref).abs()
        };
        if !err.is_nan() {
            self.max_probability_error = self.max_probability_error.max(err);
        }
        if (u < p) != (u < p_ref) && self.first_flip.is_none() {
            self.first_flip = Some(i);
        }
        let margin = (u - p_ref).abs();
        if !margin.is_nan() {
            self.min_margin = self.min_margin.min(margin);
        }
        Ok(u)
    }
}

/// The options of the sampler (NumPyro's `NUTS` and `MCMC` arguments).
#[derive(Clone, Debug)]
pub struct NutsOptions {
    /// Warmup (adaptation) iterations (`num_warmup`).
    pub num_warmup: usize,
    /// Iterations after the warmup whose states are returned (`num_samples`).
    pub num_samples: usize,
    /// The maximal tree depth during and after the warmup (`max_tree_depth`).
    pub max_tree_depth: (usize, usize),
    /// The target acceptance probability of the step-size adaptation.
    pub target_accept_prob: f64,
    /// A dense (`true`) or diagonal mass matrix.
    pub dense_mass: bool,
    /// The initial step size.
    pub step_size: f64,
    /// Adapt the step size during the warmup (dual averaging).
    pub adapt_step_size: bool,
    /// Adapt the mass matrix during the warmup (Welford).
    pub adapt_mass_matrix: bool,
    /// Regularize the estimated mass matrix (Stan's shrinkage).
    pub regularize_mass_matrix: bool,
    /// The energy error above which a transition diverges.
    pub max_delta_energy: f64,
    /// Stop (instead of failing) at the first iteration whose random draws
    /// cannot be taken, e.g. when a replayed chain leaves the recorded one:
    /// the run then ends after the last complete iteration
    /// ([`NutsRun::stopped_at`]).
    pub stop_when_draws_fail: bool,
}

impl Default for NutsOptions {
    /// NumPyro's defaults with BoTorch's `fit_fully_bayesian_model_nuts`
    /// settings (dense mass matrix, maximal tree depth 6, 512 warmup
    /// iterations and 256 samples).
    fn default() -> Self {
        Self {
            num_warmup: 512,
            num_samples: 256,
            max_tree_depth: (6, 6),
            target_accept_prob: 0.8,
            dense_mass: true,
            step_size: 1.0,
            adapt_step_size: true,
            adapt_mass_matrix: true,
            regularize_mass_matrix: true,
            max_delta_energy: 1000.0,
            stop_when_draws_fail: false,
        }
    }
}

/// A mass matrix (or its inverse, or its square root): dense or diagonal.
#[derive(Clone, Debug, PartialEq)]
pub enum MassMatrix {
    /// A dense matrix.
    Dense(Matrix),
    /// A diagonal matrix (its diagonal).
    Diagonal(Vec<f64>),
}

impl MassMatrix {
    fn identity(dim: usize, dense: bool) -> Self {
        if dense {
            MassMatrix::Dense(Matrix::identity(dim, dim))
        } else {
            MassMatrix::Diagonal(vec![1.0; dim])
        }
    }

    /// `self · v` (`jnp.matmul` / `jnp.multiply`).
    fn apply(&self, v: &[f64]) -> Vec<f64> {
        match self {
            MassMatrix::Diagonal(d) => d.iter().zip(v).map(|(a, b)| a * b).collect(),
            MassMatrix::Dense(m) => (0..m.nrows())
                .map(|i| {
                    let mut s = 0.0;
                    for (j, vj) in v.iter().enumerate() {
                        s += m[(i, j)] * vj;
                    }
                    s
                })
                .collect(),
        }
    }
}

/// One iteration of the chain.
#[derive(Clone, Debug)]
pub struct NutsIteration {
    /// The state after the iteration.
    pub z: Vec<f64>,
    /// The potential energy of `z`.
    pub potential_energy: f64,
    /// The step size used by the next iteration.
    pub step_size: f64,
    /// The mean acceptance probability of the trajectory.
    pub accept_prob: f64,
    /// The number of leapfrog steps of the trajectory.
    pub num_steps: usize,
    /// Whether the trajectory diverged.
    pub diverging: bool,
}

/// The result of [`sample_nuts`].
#[derive(Clone, Debug)]
pub struct NutsRun {
    /// The states of the `num_samples` iterations after the warmup.
    pub samples: Vec<Vec<f64>>,
    /// Every iteration (warmup and sampling).
    pub trace: Vec<NutsIteration>,
    /// The final (adapted) step size.
    pub step_size: f64,
    /// The final (adapted) inverse mass matrix.
    pub inverse_mass_matrix: MassMatrix,
    /// The iteration at which the run stopped because its random draws
    /// failed ([`NutsOptions::stop_when_draws_fail`]), if it did.
    pub stopped_at: Option<usize>,
}

impl NutsRun {
    /// The number of divergent transitions after the warmup.
    pub fn num_divergences(&self, num_warmup: usize) -> usize {
        self.trace
            .iter()
            .skip(num_warmup)
            .filter(|it| it.diverging)
            .count()
    }
}

/// NumPyro's `init_to_uniform(radius=2)` with `find_valid_initial_params`:
/// unconstrained values drawn uniformly from `[-2, 2)` until the potential
/// energy and its gradient are finite (at most 100 attempts).
///
/// # Errors
/// [`BotorchError::Runtime`] (NumPyro's `RuntimeError`) if no attempt gives
/// a finite potential and gradient.
pub fn find_valid_initial_params<P: Potential + ?Sized>(
    potential: &P,
    draws: &mut SeededDraws,
) -> Result<Vec<f64>> {
    for _ in 0..100 {
        let z: Vec<f64> = (0..potential.dim())
            .map(|_| draws.symmetric_uniform(2.0))
            .collect();
        let (pe, grad) = potential.value_and_grad(&z);
        if pe.is_finite() && grad.iter().all(|g| g.is_finite()) {
            return Ok(z);
        }
    }
    Err(BotorchError::Runtime(
        "Cannot find valid initial parameters. Please check your model again.".into(),
    ))
}

/// A window of the warmup adaptation (`start..=end`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptWindow {
    /// The first iteration of the window.
    pub start: usize,
    /// The last iteration of the window.
    pub end: usize,
}

/// NumPyro's `build_adaptation_schedule` (Stan's windows).
pub fn build_adaptation_schedule(num_steps: usize) -> Vec<AdaptWindow> {
    let mut schedule = Vec::new();
    if num_steps < 20 {
        schedule.push(AdaptWindow {
            start: 0,
            end: num_steps.wrapping_sub(1),
        });
        return schedule;
    }
    let mut start_buffer_size = 75usize;
    let mut end_buffer_size = 50usize;
    let mut init_window_size = 25usize;
    if start_buffer_size + end_buffer_size + init_window_size > num_steps {
        start_buffer_size = (0.15 * num_steps as f64) as usize;
        end_buffer_size = (0.1 * num_steps as f64) as usize;
        init_window_size = num_steps - start_buffer_size - end_buffer_size;
    }
    schedule.push(AdaptWindow {
        start: 0,
        end: start_buffer_size.wrapping_sub(1),
    });
    let end_window_start = num_steps - end_buffer_size;
    let mut next_window_size = init_window_size;
    let mut next_window_start = start_buffer_size;
    while next_window_start < end_window_start {
        let cur_window_start = next_window_start;
        let mut cur_window_size = next_window_size;
        if 3 * cur_window_size <= end_window_start - cur_window_start {
            next_window_size = 2 * cur_window_size;
        } else {
            cur_window_size = end_window_start - cur_window_start;
        }
        next_window_start = cur_window_start + cur_window_size;
        schedule.push(AdaptWindow {
            start: cur_window_start,
            end: next_window_start - 1,
        });
    }
    schedule.push(AdaptWindow {
        start: end_window_start,
        end: num_steps - 1,
    });
    schedule
}

/// The dual-averaging state `(x_t, x̄_t, ḡ_t, t, μ)`.
#[derive(Clone, Debug)]
struct DualAveraging {
    x: f64,
    x_avg: f64,
    g_avg: f64,
    t: u64,
    prox_center: f64,
}

const DA_T0: f64 = 10.0;
const DA_KAPPA: f64 = 0.75;
const DA_GAMMA: f64 = 0.05;

impl DualAveraging {
    fn new(prox_center: f64) -> Self {
        Self {
            x: 0.0,
            x_avg: 0.0,
            g_avg: 0.0,
            t: 0,
            prox_center,
        }
    }

    fn update(&mut self, g: f64) {
        self.t += 1;
        let t = self.t as f64;
        self.g_avg = (1.0 - 1.0 / (t + DA_T0)) * self.g_avg + g / (t + DA_T0);
        self.x = self.prox_center - t.powf(0.5) / DA_GAMMA * self.g_avg;
        let weight = t.powf(-DA_KAPPA);
        self.x_avg = (1.0 - weight) * self.x_avg + weight * self.x;
    }
}

/// Welford's online (co)variance estimate.
#[derive(Clone, Debug)]
struct Welford {
    mean: Vec<f64>,
    m2: Vec<f64>,
    n: u64,
    dense: bool,
}

impl Welford {
    fn new(dim: usize, dense: bool) -> Self {
        Self {
            mean: vec![0.0; dim],
            m2: vec![0.0; if dense { dim * dim } else { dim }],
            n: 0,
            dense,
        }
    }

    fn update(&mut self, sample: &[f64]) {
        let dim = self.mean.len();
        self.n += 1;
        let n = self.n as f64;
        let delta_pre: Vec<f64> = sample.iter().zip(&self.mean).map(|(s, m)| s - m).collect();
        for (m, d) in self.mean.iter_mut().zip(&delta_pre) {
            *m += d / n;
        }
        let delta_post: Vec<f64> = sample.iter().zip(&self.mean).map(|(s, m)| s - m).collect();
        if self.dense {
            // `m2 + jnp.outer(delta_post, delta_pre)` (row-major).
            for (row, post) in self.m2.chunks_mut(dim).zip(&delta_post) {
                for (m, pre) in row.iter_mut().zip(&delta_pre) {
                    *m += post * pre;
                }
            }
        } else {
            for ((m, pre), post) in self.m2.iter_mut().zip(&delta_pre).zip(&delta_post) {
                *m += pre * post;
            }
        }
    }

    /// `(inverse mass matrix, mass matrix square root)`.
    fn finalize(&self, regularize: bool) -> (MassMatrix, MassMatrix) {
        let dim = self.mean.len();
        let n = self.n as f64;
        let scale = n / (n + 5.0);
        let shrinkage = 1e-3 * (5.0 / (n + 5.0));
        if self.dense {
            let cov = Matrix::from_fn(dim, dim, |i, j| {
                let c = self.m2[i * dim + j] / (n - 1.0);
                if regularize {
                    scale * c + if i == j { shrinkage } else { 0.0 }
                } else {
                    c
                }
            });
            let sqrt = mass_matrix_sqrt_from_inverse(&cov);
            (MassMatrix::Dense(cov), MassMatrix::Dense(sqrt))
        } else {
            let cov: Vec<f64> = self
                .m2
                .iter()
                .map(|m| {
                    let c = m / (n - 1.0);
                    if regularize {
                        scale * c + shrinkage
                    } else {
                        c
                    }
                })
                .collect();
            let sqrt = cov.iter().map(|c| 1.0 / c.sqrt()).collect();
            (MassMatrix::Diagonal(cov), MassMatrix::Diagonal(sqrt))
        }
    }
}

/// The square root `S` (lower triangular) of the mass matrix `M = Σ⁻¹`
/// with `M = S Sᵀ`, from the inverse mass matrix `Σ`, as NumPyro computes
/// it: `Σ = U Uᵀ` with `U` upper triangular (the Cholesky factor of `Σ`
/// with its rows and columns reversed; `jnp.linalg.cholesky` symmetrizes its
/// input) and `S = U⁻ᵀ`. NaN if `Σ` is not positive definite (as JAX).
fn mass_matrix_sqrt_from_inverse(cov: &Matrix) -> Matrix {
    let dim = cov.nrows();
    let rev = Matrix::from_fn(dim, dim, |i, j| {
        let (a, b) = (dim - 1 - i, dim - 1 - j);
        (cov[(a, b)] + cov[(b, a)]) / 2.0
    });
    let Some(l) = cholesky_lower(&rev) else {
        return Matrix::from_element(dim, dim, f64::NAN);
    };
    // tril_inv = (reverse(L))ᵀ, lower triangular.
    let tril_inv = Matrix::from_fn(dim, dim, |i, j| l[(dim - 1 - j, dim - 1 - i)]);
    // Solve tril_inv · S = I by forward substitution.
    let mut s = Matrix::zeros(dim, dim);
    for c in 0..dim {
        for i in 0..dim {
            let mut v = if i == c { 1.0 } else { 0.0 };
            for k in 0..i {
                v -= tril_inv[(i, k)] * s[(k, c)];
            }
            s[(i, c)] = v / tril_inv[(i, i)];
        }
    }
    s
}

/// The lower Cholesky factor of a symmetric matrix (`None` if it is not
/// positive definite or not finite).
fn cholesky_lower(a: &Matrix) -> Option<Matrix> {
    let n = a.nrows();
    let mut l = Matrix::zeros(n, n);
    for j in 0..n {
        let mut d = a[(j, j)];
        for k in 0..j {
            d -= l[(j, k)] * l[(j, k)];
        }
        if d.is_nan() || d <= 0.0 || !d.is_finite() {
            return None;
        }
        let djj = d.sqrt();
        l[(j, j)] = djj;
        for i in (j + 1)..n {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = s / djj;
        }
    }
    Some(l)
}

/// The adaptation state (NumPyro's `HMCAdaptState`).
#[derive(Clone, Debug)]
struct AdaptState {
    step_size: f64,
    inverse_mass_matrix: MassMatrix,
    mass_matrix_sqrt: MassMatrix,
    ss: DualAveraging,
    mm: Welford,
    window_idx: usize,
}

/// The kinetic energy `½ rᵀ Σ r` (`euclidean_kinetic_energy`).
fn kinetic_energy(inverse_mass_matrix: &MassMatrix, r: &[f64]) -> f64 {
    let v = inverse_mass_matrix.apply(r);
    0.5 * dot(&v, r)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// `jnp.logaddexp`.
fn logaddexp(a: f64, b: f64) -> f64 {
    let amax = a.max(b);
    let delta = a - b;
    if delta.is_nan() {
        a + b
    } else {
        amax + (-delta.abs()).exp().ln_1p()
    }
}

/// `jnp.minimum(x, 1)` (NaN propagates).
fn clip_one(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else {
        x.min(1.0)
    }
}

/// A NUTS (sub)tree (NumPyro's `TreeInfo`).
#[derive(Clone, Debug)]
struct Tree {
    z_left: Vec<f64>,
    r_left: Vec<f64>,
    grad_left: Vec<f64>,
    z_right: Vec<f64>,
    r_right: Vec<f64>,
    grad_right: Vec<f64>,
    z_proposal: Vec<f64>,
    pe_proposal: f64,
    grad_proposal: Vec<f64>,
    energy_proposal: f64,
    depth: usize,
    weight: f64,
    r_sum: Vec<f64>,
    turning: bool,
    diverging: bool,
    sum_accept_probs: f64,
    num_proposals: usize,
}

struct Integrator<'a, P: Potential + ?Sized> {
    potential: &'a P,
    inverse_mass_matrix: &'a MassMatrix,
    step_size: f64,
    max_delta_energy: f64,
    energy_current: f64,
}

impl<P: Potential + ?Sized> Integrator<'_, P> {
    /// One velocity Verlet step from `(z, r)` (with the gradient `grad` of
    /// the potential at `z`) in the given direction (`_build_basetree`).
    fn base_tree(&self, z: &[f64], r: &[f64], grad: &[f64], going_right: bool) -> Tree {
        let step = if going_right {
            self.step_size
        } else {
            -self.step_size
        };
        let half = 0.5 * step;
        let r_half: Vec<f64> = r.iter().zip(grad).map(|(r, g)| r - half * g).collect();
        let v = self.inverse_mass_matrix.apply(&r_half);
        let z_new: Vec<f64> = z.iter().zip(&v).map(|(z, v)| z + step * v).collect();
        let (pe_new, grad_new) = self.potential.value_and_grad(&z_new);
        let r_new: Vec<f64> = r_half
            .iter()
            .zip(&grad_new)
            .map(|(r, g)| r - half * g)
            .collect();
        let energy_new = pe_new + kinetic_energy(self.inverse_mass_matrix, &r_new);
        let mut delta_energy = energy_new - self.energy_current;
        if delta_energy.is_nan() {
            delta_energy = f64::INFINITY;
        }
        let accept_prob = clip_one((-delta_energy).exp());
        Tree {
            z_left: z_new.clone(),
            r_left: r_new.clone(),
            grad_left: grad_new.clone(),
            z_right: z_new.clone(),
            r_right: r_new.clone(),
            grad_right: grad_new.clone(),
            z_proposal: z_new,
            pe_proposal: pe_new,
            grad_proposal: grad_new,
            energy_proposal: energy_new,
            depth: 0,
            weight: -delta_energy,
            r_sum: r_new,
            turning: false,
            diverging: delta_energy > self.max_delta_energy,
            sum_accept_probs: accept_prob,
            num_proposals: 1,
        }
    }

    /// NumPyro's `_is_turning`.
    fn is_turning(&self, r_left: &[f64], r_right: &[f64], r_sum: &[f64]) -> bool {
        let v_left = self.inverse_mass_matrix.apply(r_left);
        let v_right = self.inverse_mass_matrix.apply(r_right);
        let rs: Vec<f64> = r_sum
            .iter()
            .zip(r_left.iter().zip(r_right))
            .map(|(s, (l, r))| s - (l + r) / 2.0)
            .collect();
        dot(&v_left, &rs) <= 0.0 || dot(&v_right, &rs) <= 0.0
    }

    /// NumPyro's `_combine_tree`.
    fn combine(
        &self,
        current: &Tree,
        new: &Tree,
        going_right: bool,
        biased: bool,
        rng: &mut (impl NutsRandom + ?Sized),
    ) -> Result<Tree> {
        let (left, right) = if going_right {
            (current, new)
        } else {
            (new, current)
        };
        let r_sum: Vec<f64> = current
            .r_sum
            .iter()
            .zip(&new.r_sum)
            .map(|(a, b)| a + b)
            .collect();
        let (prob, turning, kind) = if biased {
            let prob = if new.turning || new.diverging {
                0.0
            } else {
                clip_one((new.weight - current.weight).exp())
            };
            let turning = new.turning || self.is_turning(&left.r_left, &right.r_right, &r_sum);
            (prob, turning, UniformKind::TreeTransition)
        } else {
            let prob = 1.0 / (1.0 + (-(new.weight - current.weight)).exp());
            (prob, current.turning, UniformKind::SubtreeTransition)
        };
        let u = rng.uniform(kind, prob)?;
        let proposal = if u < prob { new } else { current };
        Ok(Tree {
            z_left: left.z_left.clone(),
            r_left: left.r_left.clone(),
            grad_left: left.grad_left.clone(),
            z_right: right.z_right.clone(),
            r_right: right.r_right.clone(),
            grad_right: right.grad_right.clone(),
            z_proposal: proposal.z_proposal.clone(),
            pe_proposal: proposal.pe_proposal,
            grad_proposal: proposal.grad_proposal.clone(),
            energy_proposal: proposal.energy_proposal,
            depth: current.depth + 1,
            weight: logaddexp(current.weight, new.weight),
            r_sum,
            turning,
            diverging: new.diverging,
            sum_accept_probs: current.sum_accept_probs + new.sum_accept_probs,
            num_proposals: current.num_proposals + new.num_proposals,
        })
    }

    /// NumPyro's `_iterative_build_subtree`: a subtree of `2^depth` leaves
    /// from the edge of `prototype` (stopped early by a U-turn or a
    /// divergence).
    fn build_subtree(
        &self,
        prototype: &Tree,
        going_right: bool,
        num_ckpts: usize,
        rng: &mut (impl NutsRandom + ?Sized),
    ) -> Result<Tree> {
        let dim = prototype.r_sum.len();
        let max_num_proposals = 1usize << prototype.depth;
        let mut r_ckpts = vec![vec![0.0; dim]; num_ckpts];
        let mut r_sum_ckpts = vec![vec![0.0; dim]; num_ckpts];
        let mut tree = prototype.clone();
        tree.num_proposals = 0;
        let mut turning = false;
        while tree.num_proposals < max_num_proposals && !turning && !tree.diverging {
            let new_leaf = if going_right {
                self.base_tree(&tree.z_right, &tree.r_right, &tree.grad_right, true)
            } else {
                self.base_tree(&tree.z_left, &tree.r_left, &tree.grad_left, false)
            };
            let leaf_idx = tree.num_proposals;
            let new_tree = if leaf_idx == 0 {
                new_leaf.clone()
            } else {
                self.combine(&tree, &new_leaf, going_right, false, rng)?
            };
            let (idx_min, idx_max) = leaf_idx_to_ckpt_idxs(leaf_idx);
            if leaf_idx.is_multiple_of(2) {
                r_ckpts[idx_max as usize] = new_leaf.r_right.clone();
                r_sum_ckpts[idx_max as usize] = new_tree.r_sum.clone();
            }
            // `_is_iterative_turning`.
            let mut i = idx_max;
            turning = false;
            while i >= idx_min && !turning {
                let k = i as usize;
                let subtree_r_sum: Vec<f64> = new_tree
                    .r_sum
                    .iter()
                    .zip(r_sum_ckpts[k].iter().zip(&r_ckpts[k]))
                    .map(|(s, (sc, rc))| s - sc + rc)
                    .collect();
                turning = self.is_turning(&r_ckpts[k], &new_leaf.r_right, &subtree_r_sum);
                i -= 1;
            }
            tree = new_tree;
        }
        tree.depth = prototype.depth;
        tree.turning = turning;
        Ok(tree)
    }
}

/// NumPyro's `_leaf_idx_to_ckpt_idxs`.
fn leaf_idx_to_ckpt_idxs(n: usize) -> (i64, i64) {
    let n = n as i64;
    let idx_max = (n >> 1).count_ones() as i64;
    let num_subtrees = ((!n & (n + 1)) - 1).count_ones() as i64;
    (idx_max - num_subtrees + 1, idx_max)
}

/// Runs NumPyro's NUTS from `z0` for `num_warmup + num_samples` iterations.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for an empty or mismatching `z0`, or
/// if `rng` runs out of (recorded) draws or they do not match the sampler.
pub fn sample_nuts<P: Potential + ?Sized>(
    potential: &P,
    z0: &[f64],
    options: &NutsOptions,
    rng: &mut (impl NutsRandom + ?Sized),
) -> Result<NutsRun> {
    sample_nuts_with_callback(potential, z0, options, rng, |_, _| Ok(()))
}

/// Runs [`sample_nuts`], reporting each completed warmup or sample transition.
///
/// The callback receives a one-based transition count and its diagnostics.
/// It runs before the next transition, does not draw random numbers, and can
/// cancel the run by returning an error. No partial result is returned on error.
///
/// # Errors
/// Returns the validation and random-draw errors of [`sample_nuts`], or an
/// error returned by the callback.
pub fn sample_nuts_with_callback<P: Potential + ?Sized>(
    potential: &P,
    z0: &[f64],
    options: &NutsOptions,
    rng: &mut (impl NutsRandom + ?Sized),
    mut callback: impl FnMut(usize, &NutsIteration) -> Result<()>,
) -> Result<NutsRun> {
    let dim = potential.dim();
    if z0.len() != dim {
        return Err(BotorchError::InvalidArgument(format!(
            "expected {dim} initial values, got {}",
            z0.len()
        )));
    }
    let schedule = build_adaptation_schedule(options.num_warmup);
    let num_windows = schedule.len();
    let (mut pe, mut grad) = potential.value_and_grad(z0);
    let mut z = z0.to_vec();
    let identity = MassMatrix::identity(dim, options.dense_mass);
    let mut adapt = AdaptState {
        step_size: options.step_size,
        inverse_mass_matrix: identity.clone(),
        mass_matrix_sqrt: identity,
        ss: DualAveraging::new((10.0 * options.step_size).ln()),
        mm: Welford::new(dim, options.dense_mass),
        window_idx: 0,
    };
    // The momentum of `init_kernel` (only used for the initial energy).
    rng.normal(dim)?;
    let num_ckpts = options.max_tree_depth.0.max(options.max_tree_depth.1);
    let total = options.num_warmup + options.num_samples;
    let mut trace = Vec::with_capacity(total);
    let mut samples = Vec::with_capacity(options.num_samples);
    let mut stopped_at = None;
    for i in 0..total {
        let warmup = i < options.num_warmup;
        let max_depth = if warmup {
            options.max_tree_depth.0
        } else {
            options.max_tree_depth.1
        };
        let tree = match nuts_iteration(
            potential,
            (&z, pe, &grad),
            &adapt,
            options,
            max_depth,
            num_ckpts,
            rng,
        ) {
            Ok(tree) => tree,
            Err(_) if options.stop_when_draws_fail => {
                stopped_at = Some(i);
                break;
            }
            Err(e) => return Err(e),
        };
        let accept_prob = tree.sum_accept_probs / tree.num_proposals as f64;
        z = tree.z_proposal;
        pe = tree.pe_proposal;
        grad = tree.grad_proposal;
        if warmup {
            adapt_update(
                &mut adapt,
                i,
                accept_prob,
                &z,
                options,
                &schedule,
                num_windows,
            );
        }
        let iteration = NutsIteration {
            z: z.clone(),
            potential_energy: pe,
            step_size: adapt.step_size,
            accept_prob,
            num_steps: tree.num_proposals,
            diverging: tree.diverging,
        };
        callback(i + 1, &iteration)?;
        trace.push(iteration);
        if !warmup {
            samples.push(z.clone());
        }
    }
    Ok(NutsRun {
        samples,
        trace,
        step_size: adapt.step_size,
        inverse_mass_matrix: adapt.inverse_mass_matrix,
        stopped_at,
    })
}

/// One NUTS transition from the state `(z, U(z), ∇U(z))`: a momentum
/// draw and NumPyro's `build_tree`.
fn nuts_iteration<P: Potential + ?Sized>(
    potential: &P,
    (z, pe, grad): (&[f64], f64, &[f64]),
    adapt: &AdaptState,
    options: &NutsOptions,
    max_depth: usize,
    num_ckpts: usize,
    rng: &mut (impl NutsRandom + ?Sized),
) -> Result<Tree> {
    let eps = rng.normal(z.len())?;
    let r = adapt.mass_matrix_sqrt.apply(&eps);
    let integrator = Integrator {
        potential,
        inverse_mass_matrix: &adapt.inverse_mass_matrix,
        step_size: adapt.step_size,
        max_delta_energy: options.max_delta_energy,
        energy_current: pe + kinetic_energy(&adapt.inverse_mass_matrix, &r),
    };
    let mut tree = Tree {
        z_left: z.to_vec(),
        r_left: r.clone(),
        grad_left: grad.to_vec(),
        z_right: z.to_vec(),
        r_right: r.clone(),
        grad_right: grad.to_vec(),
        z_proposal: z.to_vec(),
        pe_proposal: pe,
        grad_proposal: grad.to_vec(),
        energy_proposal: integrator.energy_current,
        depth: 0,
        weight: 0.0,
        r_sum: r,
        turning: false,
        diverging: false,
        sum_accept_probs: 0.0,
        num_proposals: 0,
    };
    while tree.depth < max_depth && !tree.turning && !tree.diverging {
        let going_right = rng.uniform(UniformKind::Direction, 0.5)? < 0.5;
        let subtree = integrator.build_subtree(&tree, going_right, num_ckpts, rng)?;
        tree = integrator.combine(&tree, &subtree, going_right, true, rng)?;
    }
    Ok(tree)
}

/// NumPyro's `warmup_adapter` update at warmup iteration `t`.
fn adapt_update(
    state: &mut AdaptState,
    t: usize,
    accept_prob: f64,
    z: &[f64],
    options: &NutsOptions,
    schedule: &[AdaptWindow],
    num_windows: usize,
) {
    if options.adapt_step_size {
        state.ss.update(options.target_accept_prob - accept_prob);
        let log_step_size = if t + 1 == options.num_warmup {
            state.ss.x_avg
        } else {
            state.ss.x
        };
        state.step_size = log_step_size.exp().clamp(f64::MIN_POSITIVE, f64::MAX);
    }
    let window_idx = state.window_idx;
    let is_middle_window = 0 < window_idx && window_idx + 1 < num_windows;
    if options.adapt_mass_matrix && is_middle_window {
        state.mm.update(z);
    }
    // JAX clamps the out-of-range index after the last window.
    let window_end = schedule[window_idx.min(num_windows - 1)].end;
    let at_window_end = t == window_end;
    if at_window_end {
        state.window_idx += 1;
    }
    if at_window_end && is_middle_window {
        if options.adapt_mass_matrix {
            let (inverse, sqrt) = state.mm.finalize(options.regularize_mass_matrix);
            state.inverse_mass_matrix = inverse;
            state.mass_matrix_sqrt = sqrt;
            state.mm = Welford::new(z.len(), options.dense_mass);
        }
        if options.adapt_step_size {
            state.ss = DualAveraging::new(10f64.ln() + state.step_size.ln());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptation_schedules_match_numpyro() {
        let w = |s: usize, e: usize| AdaptWindow { start: s, end: e };
        assert_eq!(build_adaptation_schedule(10), vec![w(0, 9)]);
        assert_eq!(
            build_adaptation_schedule(32),
            vec![w(0, 3), w(4, 28), w(29, 31)]
        );
        assert_eq!(
            build_adaptation_schedule(512),
            vec![
                w(0, 74),
                w(75, 99),
                w(100, 149),
                w(150, 249),
                w(250, 461),
                w(462, 511)
            ]
        );
    }

    #[test]
    fn checkpoint_indices_match_numpyro() {
        // n: (idx_min, idx_max) from NumPyro's `_leaf_idx_to_ckpt_idxs`.
        let expected = [
            (0, (1, 0)),
            (1, (0, 0)),
            (2, (2, 1)),
            (3, (0, 1)),
            (4, (2, 1)),
            (5, (1, 1)),
            (6, (3, 2)),
            (7, (0, 2)),
            (13, (2, 2)),
            (30, (5, 4)),
            (31, (0, 4)),
        ];
        for (n, e) in expected {
            assert_eq!(leaf_idx_to_ckpt_idxs(n), e, "n = {n}");
        }
    }

    struct Gaussian;

    impl Potential for Gaussian {
        fn dim(&self) -> usize {
            2
        }
        fn value_and_grad(&self, z: &[f64]) -> (f64, Vec<f64>) {
            // Correlated Gaussian with precision [[2, 0.6], [0.6, 1]].
            let g = vec![2.0 * z[0] + 0.6 * z[1], 0.6 * z[0] + z[1]];
            (0.5 * (z[0] * g[0] + z[1] * g[1]), g)
        }
    }

    #[test]
    fn iteration_callbacks_preserve_seeded_trajectories_and_can_cancel() {
        let options = NutsOptions {
            num_warmup: 4,
            num_samples: 3,
            max_tree_depth: (2, 2),
            ..Default::default()
        };
        let expected =
            sample_nuts(&Gaussian, &[1.0, -1.0], &options, &mut SeededDraws::new(17)).unwrap();
        let mut reported = Vec::new();
        let actual = sample_nuts_with_callback(
            &Gaussian,
            &[1.0, -1.0],
            &options,
            &mut SeededDraws::new(17),
            |completed, iteration| {
                assert_eq!(completed, reported.len() + 1);
                reported.push(iteration.clone());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(actual.samples, expected.samples);
        assert_eq!(actual.step_size, expected.step_size);
        assert_eq!(reported.len(), 7);
        for (a, b) in reported.iter().zip(&expected.trace) {
            assert_eq!(a.z, b.z);
            assert_eq!(a.potential_energy, b.potential_energy);
            assert_eq!(a.accept_prob, b.accept_prob);
            assert_eq!(a.num_steps, b.num_steps);
            assert_eq!(a.diverging, b.diverging);
        }
        let mut completed = 0;
        let error = sample_nuts_with_callback(
            &Gaussian,
            &[1.0, -1.0],
            &options,
            &mut SeededDraws::new(17),
            |i, _| {
                completed = i;
                if i == 2 {
                    Err(BotorchError::Runtime("cancelled during warmup".into()))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert_eq!(completed, 2);
        assert!(error.to_string().contains("cancelled during warmup"));
    }

    #[test]
    fn samples_a_gaussian() {
        let options = NutsOptions {
            num_warmup: 300,
            num_samples: 2000,
            ..Default::default()
        };
        let mut rng = SeededDraws::new(3);
        let run = sample_nuts(&Gaussian, &[1.0, -1.0], &options, &mut rng).unwrap();
        let n = run.samples.len() as f64;
        let mean: Vec<f64> = (0..2)
            .map(|k| run.samples.iter().map(|s| s[k]).sum::<f64>() / n)
            .collect();
        // Covariance = inverse precision = [[0.6098, -0.3659], [-0.3659, 1.2195]].
        let cov = |a: usize, b: usize| {
            run.samples
                .iter()
                .map(|s| (s[a] - mean[a]) * (s[b] - mean[b]))
                .sum::<f64>()
                / n
        };
        assert!(mean[0].abs() < 0.1 && mean[1].abs() < 0.1, "{mean:?}");
        assert!((cov(0, 0) - 0.6098).abs() < 0.1, "{}", cov(0, 0));
        assert!((cov(1, 1) - 1.2195).abs() < 0.2, "{}", cov(1, 1));
        assert!((cov(0, 1) + 0.3659).abs() < 0.1, "{}", cov(0, 1));
        assert_eq!(run.num_divergences(options.num_warmup), 0);
    }
}
