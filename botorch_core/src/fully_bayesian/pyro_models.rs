//! The probabilistic models of BoTorch's fully Bayesian GPs
//! (`botorch.models.fully_bayesian`: `SaasPyroModel`, `MaternPyroModel`,
//! `LinearPyroModel`; `botorch.models.fully_bayesian_multitask`:
//! `MultitaskSaasPyroModel`) as potential energies for NUTS.
//!
//! BoTorch writes them as NumPyro models; NumPyro maps every latent site to
//! an unconstrained value (`exp` for the positive sites, with its
//! log-Jacobian) and differentiates the negative log joint density with JAX.
//! [`PyroModel`] evaluates the same potential with analytic gradients:
//!
//! * the GP marginal likelihood `N(y | μ, K)` with
//!   `K = s · k(X) (∘ K_task) + (10⁻⁴ + σ²) I` (or `+ diag(train_Yvar)`),
//!   where `k` is the Matérn-5/2 kernel on `X / ℓ`
//!   (`d = sqrt(max(‖Δ‖², 10⁻³⁰))`, as BoTorch's `compute_dists`) or the
//!   linear kernel `Σ_k w_k (x_k - ½)(x'_k - ½)`;
//! * the SAAS prior `τ ~ HC(0.1)`, `ρ_k ~ HC(1)`, `ℓ_k = (τ ρ_k)^{-1/2}`,
//!   the outputscale `s ~ Gamma(2, 0.15)`; the dimension-scaled prior
//!   `ℓ_k ~ LogNormal(√2 + ln(d)/2, √3)` of `MaternPyroModel`; the linear
//!   model's `τ ~ HC(0.1)`, `ω_k ~ HC(1)`, `w_k = sqrt(τ ω_k)`;
//! * the mean `μ ~ N(0, 1)` (one per task for multi-task models), the
//!   noise `σ² ~ Gamma(0.9, 10)`;
//! * optional Kumaraswamy input warping `1 - (1 - x'^{c₁})^{c₀}` of the
//!   inputs mapped to `x' = x (1 - 2ε) + ε` (the `Warp` transform that the
//!   fitted model applies), with `c₀, c₁ ~ LogNormal(0, √0.1)`. BoTorch's
//!   NumPyro model warps with `1 - (1 - clamp(x, ε, 1 - ε)^{c₀})^{c₁}`
//!   instead, so that its samples of `c₀` and `c₁` do not belong to the
//!   warping its fitted models use (a BoTorch bug, see
//!   `docs/drafts/10a-fully-bayesian.md`);
//! * for multi-task models, the task covariance `k₅⸝₂(e / λ)` of latent task
//!   features `e ~ N(0, 1)` (`num_tasks × rank`) with lengthscales
//!   `λ ~ Gamma(6, 3)`.
//!
//! The unconstrained parameters are ordered as NumPyro flattens them: by
//! site name (`ravel_pytree` of a dict, BoTorch's dense mass matrix over
//! `sorted(z)`), every site in row-major order.

use statrs::function::gamma::ln_gamma;

use crate::error::{BotorchError, Result};
use crate::fully_bayesian::nuts::Potential;
use crate::{Matrix, Vector};
use nalgebra::Cholesky;

pub use crate::models::MIN_INFERRED_NOISE_LEVEL;

const SQRT5: f64 = 2.236_067_977_499_79;

/// The kernel of a fully Bayesian model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyroKernel {
    /// Matérn-5/2 with the SAAS prior (`SaasPyroModel`).
    Saas,
    /// Matérn-5/2 with the dimension-scaled lengthscale prior
    /// (`MaternPyroModel`).
    Matern,
    /// The linear kernel (`LinearPyroModel`).
    Linear,
}

/// The task structure of a multi-task model
/// (`LatentFeatureMultiTaskPyroMixin`).
#[derive(Clone, Debug, PartialEq)]
pub struct TaskStructure {
    /// The column of the task indices (`0..num_tasks`) in the inputs.
    pub task_feature: usize,
    /// The number of tasks.
    pub num_tasks: usize,
    /// The number of latent task features (`task_rank`).
    pub rank: usize,
}

/// The specification of a fully Bayesian model.
#[derive(Clone, Debug)]
pub struct PyroModelSpec {
    /// The kernel and its priors.
    pub kernel: PyroKernel,
    /// The `Gamma(concentration, rate)` prior of the outputscale, or `None`
    /// for an outputscale of one (Matérn and SAAS kernels).
    pub outputscale_prior: Option<(f64, f64)>,
    /// The (input-transformed) training inputs, `n × D`.
    pub train_x: Matrix,
    /// The (outcome-transformed) training targets.
    pub train_y: Vec<f64>,
    /// The observed noise variances, or `None` to infer the noise.
    pub train_yvar: Option<Vec<f64>>,
    /// The warped columns of the inputs (`None` without input warping).
    pub warp_indices: Option<Vec<usize>>,
    /// The clamping margin of the warped inputs.
    pub warp_eps: f64,
    /// The task structure of a multi-task model.
    pub task: Option<TaskStructure>,
}

/// The support of a latent site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    /// Positive values (unconstrained by `exp`).
    Positive,
    /// Real values.
    Real,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SitePrior {
    HalfCauchy(f64),
    Gamma(f64, f64),
    Normal(f64, f64),
    LogNormal(f64, f64),
}

/// A latent site of the model.
#[derive(Clone, Debug, PartialEq)]
pub struct Site {
    /// NumPyro's site name.
    pub name: &'static str,
    /// The shape of the site (`[]` for scalars).
    pub shape: Vec<usize>,
    /// The offset of the site in the unconstrained parameter vector.
    pub offset: usize,
    /// The support of the site.
    pub support: Support,
    prior: SitePrior,
}

impl Site {
    /// The number of values of the site.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A fully Bayesian model as a potential energy (see the
/// [module documentation](self)).
#[derive(Clone, Debug)]
pub struct PyroModel {
    spec: PyroModelSpec,
    sites: Vec<Site>,
    dim: usize,
    /// The feature columns (all columns but the task column).
    features: Vec<usize>,
    /// The task of every training point.
    task_index: Vec<usize>,
}

fn site(name: &'static str, shape: Vec<usize>, support: Support, prior: SitePrior) -> Site {
    Site {
        name,
        shape,
        offset: 0,
        support,
        prior,
    }
}

impl PyroModel {
    /// The model of `spec`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes,
    /// [`BotorchError::InvalidArgument`] for invalid task indices or warped
    /// columns.
    pub fn new(spec: PyroModelSpec) -> Result<Self> {
        let n = spec.train_x.nrows();
        let cols = spec.train_x.ncols();
        if spec.train_y.len() != n {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {n} training targets, got {}",
                spec.train_y.len()
            )));
        }
        if let Some(v) = &spec.train_yvar {
            if v.len() != n {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected {n} noise variances, got {}",
                    v.len()
                )));
            }
        }
        let mut task_index = Vec::new();
        let features: Vec<usize> = match &spec.task {
            Some(t) => {
                if t.task_feature >= cols || t.num_tasks == 0 || t.rank == 0 {
                    return Err(BotorchError::InvalidArgument(format!(
                        "invalid task structure {t:?} for inputs with {cols} columns"
                    )));
                }
                for i in 0..n {
                    let v = spec.train_x[(i, t.task_feature)];
                    if !(v >= 0.0 && v < t.num_tasks as f64 && v.fract() == 0.0) {
                        return Err(BotorchError::InvalidArgument(format!(
                            "task index {v} is not one of 0..{}",
                            t.num_tasks
                        )));
                    }
                    task_index.push(v as usize);
                }
                (0..cols).filter(|&c| c != t.task_feature).collect()
            }
            None => (0..cols).collect(),
        };
        let d = features.len();
        if let Some(idx) = &spec.warp_indices {
            let mut unique = idx.clone();
            unique.sort_unstable();
            unique.dedup();
            if idx.is_empty()
                || unique.len() != idx.len()
                || idx
                    .iter()
                    .any(|&i| i >= cols || spec.task.as_ref().is_some_and(|t| i == t.task_feature))
                || !spec.warp_eps.is_finite()
                || spec.warp_eps <= 0.0
                || spec.warp_eps >= 0.5
            {
                return Err(BotorchError::InvalidArgument(format!(
                    "warped columns {idx:?} out of range for {cols} columns"
                )));
            }
        }
        let dw = spec.warp_indices.as_ref().map_or(0, Vec::len);
        let lognormal_c = SitePrior::LogNormal(0.0, 0.1f64.powf(0.5));
        let mut sites = Vec::new();
        let num_means = spec.task.as_ref().map_or(0, |t| t.num_tasks);
        let mean_shape = if num_means > 0 {
            vec![num_means]
        } else {
            vec![]
        };
        match spec.kernel {
            PyroKernel::Saas => {
                sites.push(site(
                    "_kernel_inv_length_sq",
                    vec![d],
                    Support::Positive,
                    SitePrior::HalfCauchy(1.0),
                ));
                sites.push(site(
                    "kernel_tausq",
                    vec![],
                    Support::Positive,
                    SitePrior::HalfCauchy(0.1),
                ));
            }
            PyroKernel::Matern => {
                let loc = 2f64.sqrt() + (d as f64).ln() * 0.5;
                sites.push(site(
                    "lengthscale",
                    vec![d],
                    Support::Positive,
                    SitePrior::LogNormal(loc, 3f64.sqrt()),
                ));
            }
            PyroKernel::Linear => {
                sites.push(site(
                    "_weight_variance_sq",
                    vec![d],
                    Support::Positive,
                    SitePrior::HalfCauchy(1.0),
                ));
                sites.push(site(
                    "tau_sq",
                    vec![],
                    Support::Positive,
                    SitePrior::HalfCauchy(0.1),
                ));
            }
        }
        if dw > 0 {
            sites.push(site("c0", vec![dw], Support::Positive, lognormal_c));
            sites.push(site("c1", vec![dw], Support::Positive, lognormal_c));
        }
        sites.push(site(
            "mean",
            mean_shape,
            Support::Real,
            SitePrior::Normal(0.0, 1.0),
        ));
        if spec.train_yvar.is_none() {
            sites.push(site(
                "noise",
                vec![],
                Support::Positive,
                SitePrior::Gamma(0.9, 10.0),
            ));
        }
        if spec.kernel != PyroKernel::Linear {
            if let Some((c, r)) = spec.outputscale_prior {
                sites.push(site(
                    "outputscale",
                    vec![],
                    Support::Positive,
                    SitePrior::Gamma(c, r),
                ));
            }
        }
        if let Some(t) = &spec.task {
            sites.push(site(
                "latent_features",
                vec![t.num_tasks, t.rank],
                Support::Real,
                SitePrior::Normal(0.0, 1.0),
            ));
            sites.push(site(
                "task_lengthscale",
                vec![t.rank],
                Support::Positive,
                SitePrior::Gamma(6.0, 3.0),
            ));
        }
        sites.sort_by(|a, b| a.name.cmp(b.name));
        let mut offset = 0;
        for s in &mut sites {
            s.offset = offset;
            offset += s.size();
        }
        Ok(Self {
            spec,
            sites,
            dim: offset,
            features,
            task_index,
        })
    }

    /// The latent sites, sorted by name (the order of the unconstrained
    /// parameters).
    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    /// The model specification.
    pub fn spec(&self) -> &PyroModelSpec {
        &self.spec
    }

    fn site(&self, name: &str) -> Option<&Site> {
        self.sites.iter().find(|s| s.name == name)
    }

    /// The constrained value of every latent site at `z` (NumPyro's
    /// `constrain_fn`).
    pub fn constrain(&self, z: &[f64]) -> Vec<(&'static str, Vec<f64>)> {
        self.sites
            .iter()
            .map(|s| {
                let raw = &z[s.offset..s.offset + s.size()];
                let values = match s.support {
                    Support::Positive => raw.iter().map(|v| v.exp()).collect(),
                    Support::Real => raw.to_vec(),
                };
                (s.name, values)
            })
            .collect()
    }

    /// The samples as BoTorch's `postprocess_mcmc_samples` returns them for
    /// the unconstrained states `samples`: `(name, shape of one sample,
    /// values of all samples)` with the derived sites (`lengthscale` and
    /// `kernel_inv_length_sq` of the SAAS model, `weight_variance` of the
    /// linear model) and without the auxiliary ones (`kernel_tausq`,
    /// `_kernel_inv_length_sq`, `tau_sq`, `_weight_variance_sq`).
    ///
    /// Unlike BoTorch's, the `noise` samples are the noise variances of the
    /// likelihood, `10⁻⁴ + σ²` (BoTorch returns `σ²`, and its models use
    /// `max(σ², 10⁻⁴)` instead of the `10⁻⁴ + σ²` they were sampled with).
    pub fn postprocess(&self, samples: &[Vec<f64>]) -> Vec<(String, Vec<usize>, Vec<f64>)> {
        let mut out: Vec<(String, Vec<usize>, Vec<f64>)> = Vec::new();
        let constrained: Vec<_> = samples.iter().map(|z| self.constrain(z)).collect();
        let get = |k: usize, name: &str| -> Vec<f64> {
            constrained[k]
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        };
        let d = self.features.len();
        match self.spec.kernel {
            PyroKernel::Saas => {
                let mut ils_all = Vec::new();
                let mut ls_all = Vec::new();
                for k in 0..samples.len() {
                    let tausq = get(k, "kernel_tausq")[0];
                    for v in get(k, "_kernel_inv_length_sq") {
                        let ils = tausq * v;
                        ils_all.push(ils);
                        ls_all.push(1.0 / ils.sqrt());
                    }
                }
                out.push(("kernel_inv_length_sq".into(), vec![d], ils_all));
                out.push(("lengthscale".into(), vec![d], ls_all));
            }
            PyroKernel::Linear => {
                let mut wv_all = Vec::new();
                for k in 0..samples.len() {
                    let tau = get(k, "tau_sq")[0];
                    wv_all.extend(
                        get(k, "_weight_variance_sq")
                            .iter()
                            .map(|w| (tau * w).sqrt()),
                    );
                }
                out.push(("weight_variance".into(), vec![d], wv_all));
            }
            PyroKernel::Matern => {}
        }
        for s in &self.sites {
            if matches!(
                s.name,
                "kernel_tausq" | "_kernel_inv_length_sq" | "tau_sq" | "_weight_variance_sq"
            ) {
                continue;
            }
            let mut all = Vec::new();
            for k in 0..samples.len() {
                all.extend(get(k, s.name));
            }
            if s.name == "noise" {
                // The noise variance of the likelihood the samples were
                // drawn with (BoTorch loads the sampled value itself,
                // clamped at the level, see the module documentation).
                for v in &mut all {
                    *v += MIN_INFERRED_NOISE_LEVEL;
                }
            }
            out.push((s.name.to_string(), s.shape.clone(), all));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The potential energy (negative log joint density of the
    /// unconstrained parameters) and its gradient.
    pub fn potential(&self, z: &[f64]) -> (f64, Vec<f64>) {
        let mut grad = vec![0.0; self.dim];
        // Log prior densities with the log-Jacobians of the transforms.
        let mut log_joint = 0.0;
        for s in &self.sites {
            for k in 0..s.size() {
                let zi = z[s.offset + k];
                let (lp, dlp) = match s.support {
                    Support::Positive => {
                        let x = zi.exp();
                        let (lp, dlp_dz) = log_prior_positive(s.prior, x);
                        (lp + zi, dlp_dz + 1.0)
                    }
                    Support::Real => log_prior_real(s.prior, zi),
                };
                log_joint += lp;
                grad[s.offset + k] = dlp;
            }
        }
        let n = self.spec.train_x.nrows();
        if n > 0 {
            match self.log_likelihood(z) {
                Some((ll, ll_grad)) => {
                    log_joint += ll;
                    for (g, h) in grad.iter_mut().zip(&ll_grad) {
                        *g += h;
                    }
                }
                None => return (f64::NAN, vec![f64::NAN; self.dim]),
            }
        }
        (-log_joint, grad.iter().map(|g| -g).collect())
    }

    fn values(&self, z: &[f64], name: &str) -> Option<(usize, Vec<f64>)> {
        let s = self.site(name)?;
        let raw = &z[s.offset..s.offset + s.size()];
        let v = match s.support {
            Support::Positive => raw.iter().map(|v| v.exp()).collect(),
            Support::Real => raw.to_vec(),
        };
        Some((s.offset, v))
    }

    /// The GP log likelihood and its gradient with respect to `z` (`None`
    /// if the covariance matrix is not positive definite).
    #[allow(clippy::too_many_lines)]
    fn log_likelihood(&self, z: &[f64]) -> Option<(f64, Vec<f64>)> {
        let spec = &self.spec;
        let n = spec.train_x.nrows();
        let d = self.features.len();
        let mut grad = vec![0.0; self.dim];
        // Inputs, warped.
        let mut x_full = spec.train_x.clone();
        let warp = self.values(z, "c0").zip(self.values(z, "c1"));
        // d x_w / d c0, d x_w / d c1 for every warped column and row.
        let mut warp_derivs: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();
        if let (Some(((_, c0), (_, c1))), Some(idx)) = (&warp, &spec.warp_indices) {
            let eps = spec.warp_eps;
            for (k, &col) in idx.iter().enumerate() {
                let mut d0 = vec![0.0; n];
                let mut d1 = vec![0.0; n];
                for i in 0..n {
                    // BoTorch's `kumaraswamy_warp` (the `Warp` transform of
                    // the fitted model): `1 - (1 - x'^c1)^c0`.
                    let x =
                        (spec.train_x[(i, col)] * (1.0 - 2.0 * eps) + eps).clamp(eps, 1.0 - eps);
                    let u = x.powf(c1[k]);
                    let one_minus_u = 1.0 - u;
                    let p = one_minus_u.powf(c0[k]);
                    x_full[(i, col)] = 1.0 - p;
                    d0[i] = -p * one_minus_u.ln();
                    d1[i] = c0[k] * one_minus_u.powf(c0[k] - 1.0) * u * x.ln();
                }
                warp_derivs.push((col, d0, d1));
            }
        }
        let xf = Matrix::from_fn(n, d, |i, k| x_full[(i, self.features[k])]);
        // Task covariance.
        let task = spec.task.as_ref().map(|t| {
            let (lf_off, lf) = self.values(z, "latent_features").unwrap();
            let (tl_off, tl) = self.values(z, "task_lengthscale").unwrap();
            let e = Matrix::from_fn(t.num_tasks, t.rank, |a, k| lf[a * t.rank + k] / tl[k]);
            let (tk, tp) = matern52_matrix(&e);
            (lf_off, tl_off, e, tl, tk, tp)
        });
        let task_cov = |i: usize, j: usize| -> f64 {
            match &task {
                Some((.., tk, _)) => tk[(self.task_index[i], self.task_index[j])],
                None => 1.0,
            }
        };
        // Data covariance (without the outputscale for Matérn kernels).
        let os = self.values(z, "outputscale");
        let s = os.as_ref().map_or(1.0, |(_, v)| v[0]);
        let (k_data, matern_aux) = match spec.kernel {
            PyroKernel::Saas | PyroKernel::Matern => {
                let ls: Vec<f64> = if spec.kernel == PyroKernel::Saas {
                    let (_, ils) = self.values(z, "_kernel_inv_length_sq").unwrap();
                    let (_, tau) = self.values(z, "kernel_tausq").unwrap();
                    ils.iter().map(|v| 1.0 / (tau[0] * v).sqrt()).collect()
                } else {
                    self.values(z, "lengthscale").unwrap().1
                };
                let scaled = Matrix::from_fn(n, d, |i, k| xf[(i, k)] / ls[k]);
                let (km, kp) = matern52_matrix(&scaled);
                (km, Some((ls, scaled, kp)))
            }
            PyroKernel::Linear => {
                let (_, w) = self.values(z, "_weight_variance_sq").unwrap();
                let (_, tau) = self.values(z, "tau_sq").unwrap();
                let wv: Vec<f64> = w.iter().map(|w| (tau[0] * w).sqrt()).collect();
                let xw = Matrix::from_fn(n, d, |i, k| (xf[(i, k)] - 0.5) * wv[k].sqrt());
                (&xw * xw.transpose(), None)
            }
        };
        // Full covariance and mean.
        let noise = match &spec.train_yvar {
            Some(v) => v.clone(),
            None => {
                let (_, v) = self.values(z, "noise").unwrap();
                vec![MIN_INFERRED_NOISE_LEVEL + v[0]; n]
            }
        };
        let mut kfull = Matrix::from_fn(n, n, |i, j| s * k_data[(i, j)] * task_cov(i, j));
        for (i, nv) in noise.iter().enumerate() {
            kfull[(i, i)] += nv;
        }
        let (mean_off, mean) = self.values(z, "mean").unwrap();
        let mean_of = |i: usize| -> f64 {
            if spec.task.is_some() {
                mean[self.task_index[i]]
            } else {
                mean[0]
            }
        };
        if kfull.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let chol = Cholesky::new(kfull)?;
        let l = chol.l_dirty();
        if (0..n).any(|i| l[(i, i)].is_nan() || l[(i, i)] <= 0.0 || !l[(i, i)].is_finite()) {
            return None;
        }
        let r = Vector::from_fn(n, |i, _| spec.train_y[i] - mean_of(i));
        // w = L⁻¹ r, α = L⁻ᵀ w.
        let w = l.solve_lower_triangular(&r)?;
        let alpha = l.tr_solve_lower_triangular(&w)?;
        let mahalanobis: f64 = w.iter().map(|v| v * v).sum();
        let half_log_det: f64 = (0..n).map(|i| l[(i, i)].ln()).sum();
        let ll = -0.5 * mahalanobis
            - (half_log_det + 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln());
        // dll/dK = ½ (α αᵀ - K⁻¹) =: G (symmetric: from the lower triangle
        // of K⁻¹ = L⁻ᵀ L⁻¹).
        let kinv = inverse_lower(l);
        let g = Matrix::from_fn(n, n, |i, j| {
            let kij = if i >= j { kinv[(i, j)] } else { kinv[(j, i)] };
            0.5 * (alpha[i] * alpha[j] - kij)
        });
        // Mean.
        for (i, a) in alpha.iter().enumerate() {
            let t = if spec.task.is_some() {
                self.task_index[i]
            } else {
                0
            };
            grad[mean_off + t] += a;
        }
        // Noise.
        if spec.train_yvar.is_none() {
            let (off, v) = self.values(z, "noise").unwrap();
            let tr: f64 = (0..n).map(|i| g[(i, i)]).sum();
            grad[off] += tr * v[0];
        }
        // Outputscale.
        if let Some((off, v)) = &os {
            let mut acc = 0.0;
            for j in 0..n {
                acc += g[(j, j)] * k_data[(j, j)] * task_cov(j, j);
                for i in (j + 1)..n {
                    acc += 2.0 * g[(i, j)] * k_data[(i, j)] * task_cov(i, j);
                }
            }
            grad[*off] += acc * v[0];
        }
        // Kernel hyperparameters and warped inputs: dll/dx_f (n × d,
        // row-major; only needed with input warping).
        let need_dx = warp.is_some();
        let mut dx = vec![0.0; if need_dx { n * d } else { 0 }];
        match &matern_aux {
            Some((ls, scaled, kp)) => {
                // W_ij = G_ij s T_ij (5/3)(1 + √5 D) e^{-√5 D}, dK_ij/dℓ_k = W Δ²/ℓ_k
                // (every pair i > j twice; the diagonal has Δ = 0).
                let rows = row_major(scaled);
                let inv_ls: Vec<f64> = ls.iter().map(|l| 1.0 / l).collect();
                let mut dls = vec![0.0; d];
                for j in 0..n {
                    let xj = &rows[j * d..(j + 1) * d];
                    for i in (j + 1)..n {
                        let wij = 2.0 * g[(i, j)] * s * task_cov(i, j) * kp[(i, j)];
                        if wij == 0.0 {
                            continue;
                        }
                        let xi = &rows[i * d..(i + 1) * d];
                        for k in 0..d {
                            let delta = xi[k] - xj[k];
                            dls[k] += wij * delta * delta * inv_ls[k];
                        }
                        if need_dx {
                            for k in 0..d {
                                let t = wij * (xi[k] - xj[k]) * inv_ls[k];
                                dx[i * d + k] -= t;
                                dx[j * d + k] += t;
                            }
                        }
                    }
                }
                if spec.kernel == PyroKernel::Saas {
                    let ils_site = self.site("_kernel_inv_length_sq").unwrap().offset;
                    let tau_site = self.site("kernel_tausq").unwrap().offset;
                    for k in 0..d {
                        let dz = dls[k] * (-0.5 * ls[k]);
                        grad[ils_site + k] += dz;
                        grad[tau_site] += dz;
                    }
                } else {
                    let off = self.site("lengthscale").unwrap().offset;
                    for k in 0..d {
                        grad[off + k] += dls[k] * ls[k];
                    }
                }
            }
            None => {
                let (w_off, w) = self.values(z, "_weight_variance_sq").unwrap();
                let (tau_off, tau) = self.values(z, "tau_sq").unwrap();
                let wv: Vec<f64> = w.iter().map(|w| (tau[0] * w).sqrt()).collect();
                let rows: Vec<f64> = row_major(&xf).iter().map(|v| v - 0.5).collect();
                let mut dwv = vec![0.0; d];
                for j in 0..n {
                    let xj = &rows[j * d..(j + 1) * d];
                    for i in j..n {
                        let xi = &rows[i * d..(i + 1) * d];
                        // Every pair i > j twice.
                        let gij = g[(i, j)] * s * task_cov(i, j);
                        let m = if i == j { 1.0 } else { 2.0 };
                        for k in 0..d {
                            dwv[k] += m * gij * xi[k] * xj[k];
                        }
                        if need_dx {
                            for k in 0..d {
                                dx[i * d + k] += 2.0 * gij * wv[k] * xj[k];
                                if i != j {
                                    dx[j * d + k] += 2.0 * gij * wv[k] * xi[k];
                                }
                            }
                        }
                    }
                }
                for k in 0..d {
                    let dz = dwv[k] * 0.5 * wv[k];
                    grad[w_off + k] += dz;
                    grad[tau_off] += dz;
                }
            }
        }
        // Warping parameters.
        if let Some(((c0_off, c0), (c1_off, c1))) = &warp {
            for (k, (col, d0, d1)) in warp_derivs.iter().enumerate() {
                let Some(f) = self.features.iter().position(|c| c == col) else {
                    continue; // the task column: warped, then dropped
                };
                let mut acc0 = 0.0;
                let mut acc1 = 0.0;
                for i in 0..n {
                    acc0 += dx[i * d + f] * d0[i];
                    acc1 += dx[i * d + f] * d1[i];
                }
                grad[c0_off + k] += acc0 * c0[k];
                grad[c1_off + k] += acc1 * c1[k];
            }
        }
        // Task covariance.
        if let (Some((lf_off, tl_off, e, tl, _, tp)), Some(t)) = (&task, &spec.task) {
            // H_ab = Σ_{t_i = a, t_j = b} G_ij s k_ij.
            let mut h = Matrix::zeros(t.num_tasks, t.num_tasks);
            for i in 0..n {
                for j in 0..n {
                    h[(self.task_index[i], self.task_index[j])] += g[(i, j)] * s * k_data[(i, j)];
                }
            }
            let mut de = Matrix::zeros(t.num_tasks, t.rank);
            for a in 0..t.num_tasks {
                for b in 0..t.num_tasks {
                    let w = h[(a, b)] * tp[(a, b)];
                    if w == 0.0 {
                        continue;
                    }
                    for k in 0..t.rank {
                        de[(a, k)] -= 2.0 * w * (e[(a, k)] - e[(b, k)]);
                    }
                }
            }
            for a in 0..t.num_tasks {
                for k in 0..t.rank {
                    grad[lf_off + a * t.rank + k] += de[(a, k)] / tl[k];
                    // dλ/dz = λ; de/dλ = -e/λ.
                    grad[tl_off + k] -= de[(a, k)] * e[(a, k)];
                }
            }
        }
        Some((ll, grad))
    }
}

impl Potential for PyroModel {
    fn dim(&self) -> usize {
        self.dim
    }

    fn value_and_grad(&self, z: &[f64]) -> (f64, Vec<f64>) {
        self.potential(z)
    }
}

/// The Matérn-5/2 kernel matrix of the rows of `scaled` (BoTorch's
/// `matern52_kernel` with `compute_dists`) and, per pair,
/// `(5/3)(1 + √5 d) e^{-√5 d}` (zero where the distance is clamped), the
/// factor of the derivatives with respect to the squared distance.
fn matern52_matrix(scaled: &Matrix) -> (Matrix, Matrix) {
    let (n, d) = scaled.shape();
    let rows = row_major(scaled);
    let mut k = Matrix::zeros(n, n);
    let mut p = Matrix::zeros(n, n);
    for j in 0..n {
        let xj = &rows[j * d..(j + 1) * d];
        for i in j..n {
            let xi = &rows[i * d..(i + 1) * d];
            let sq: f64 = xi.iter().zip(xj).map(|(a, b)| (a - b) * (a - b)).sum();
            let dist = sq.max(1e-30).sqrt();
            let s5d = SQRT5 * dist;
            let e = (-s5d).exp();
            let kv = (1.0 + s5d + 5.0 / 3.0 * (dist * dist)) * e;
            k[(i, j)] = kv;
            k[(j, i)] = kv;
            if sq > 1e-30 {
                let pv = 5.0 / 3.0 * (1.0 + s5d) * e;
                p[(i, j)] = pv;
                p[(j, i)] = pv;
            }
        }
    }
    (k, p)
}

/// The lower triangle of `(L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹` for the lower-triangular
/// `l` (the upper triangle of `l` is ignored and that of the result is
/// zero): `L⁻¹` by column-oriented forward substitution and the products
/// of its columns, `n³/3` multiplications on contiguous columns.
fn inverse_lower(l: &Matrix) -> Matrix {
    let n = l.nrows();
    let ls = l.as_slice();
    let mut linv = vec![0.0; n * n];
    for c in 0..n {
        let x = &mut linv[c * n..(c + 1) * n];
        x[c] = 1.0;
        for k in c..n {
            let xk = x[k] / ls[k * n + k];
            x[k] = xk;
            if xk != 0.0 {
                let col = &ls[k * n + k + 1..(k + 1) * n];
                for (xi, lik) in x[k + 1..].iter_mut().zip(col) {
                    *xi -= lik * xk;
                }
            }
        }
    }
    let mut out = Matrix::zeros(n, n);
    for j in 0..n {
        for i in j..n {
            // Σ_{k ≥ i} L⁻¹_ki L⁻¹_kj.
            let a = &linv[i * n + i..(i + 1) * n];
            let b = &linv[j * n + i..(j + 1) * n];
            out[(i, j)] = a.iter().zip(b).map(|(u, v)| u * v).sum();
        }
    }
    out
}

/// The entries of `m` row by row (contiguous rows for the pair loops).
fn row_major(m: &Matrix) -> Vec<f64> {
    let (n, d) = m.shape();
    let mut out = Vec::with_capacity(n * d);
    for i in 0..n {
        out.extend((0..d).map(|c| m[(i, c)]));
    }
    out
}

/// `log p(x)` of a positive site and its derivative with respect to
/// `z = log x` (without the log-Jacobian `z`).
fn log_prior_positive(prior: SitePrior, x: f64) -> (f64, f64) {
    match prior {
        SitePrior::HalfCauchy(scale) => {
            let ratio = x / scale;
            let lp = -std::f64::consts::PI.ln() - scale.ln() - (ratio * ratio).ln_1p() + 2f64.ln();
            (lp, -2.0 * ratio * ratio / (1.0 + ratio * ratio))
        }
        SitePrior::Gamma(c, r) => {
            let normalize = ln_gamma(c) - c * r.ln();
            let lp = (c - 1.0) * x.ln() - r * x - normalize;
            (lp, (c - 1.0) - r * x)
        }
        SitePrior::LogNormal(loc, scale) => {
            let lx = x.ln();
            let v = (lx - loc) / scale;
            let lp = -0.5 * v * v - ((2.0 * std::f64::consts::PI).sqrt() * scale).ln() - lx;
            (lp, -v / scale - 1.0)
        }
        SitePrior::Normal(..) => unreachable!("normal priors are on real sites"),
    }
}

/// `log p(x)` of a real site and its derivative.
fn log_prior_real(prior: SitePrior, x: f64) -> (f64, f64) {
    match prior {
        SitePrior::Normal(loc, scale) => {
            let v = (x - loc) / scale;
            (
                -0.5 * v * v - ((2.0 * std::f64::consts::PI).sqrt() * scale).ln(),
                -v / scale,
            )
        }
        _ => unreachable!("only normal priors are on real sites"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kernel: PyroKernel, warp: bool, task: bool, yvar: bool) -> PyroModelSpec {
        let n = 7;
        let cols = if task { 3 } else { 2 };
        let x = Matrix::from_fn(n, cols, |i, c| {
            if task && c == 1 {
                (i % 3) as f64
            } else {
                ((i * 7 + c * 3) % 11) as f64 / 11.0 + 0.03
            }
        });
        let y = (0..n).map(|i| ((i * 5) % 7) as f64 / 3.0 - 1.0).collect();
        PyroModelSpec {
            kernel,
            outputscale_prior: match kernel {
                PyroKernel::Saas => Some((2.0, 0.15)),
                _ => None,
            },
            train_x: x,
            train_y: y,
            train_yvar: yvar.then(|| (0..n).map(|i| 0.01 + 0.002 * i as f64).collect()),
            warp_indices: warp.then(|| (0..cols).filter(|&c| !task || c != 1).collect()),
            warp_eps: 1e-7,
            task: task.then_some(TaskStructure {
                task_feature: 1,
                num_tasks: 3,
                rank: 2,
            }),
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        for kernel in [PyroKernel::Saas, PyroKernel::Matern, PyroKernel::Linear] {
            for (warp, task, yvar) in [
                (false, false, false),
                (true, false, true),
                (false, true, false),
                (true, true, false),
            ] {
                let model = PyroModel::new(spec(kernel, warp, task, yvar)).unwrap();
                let z: Vec<f64> = (0..model.dim())
                    .map(|i| ((i * 37) % 13) as f64 / 13.0 - 0.5)
                    .collect();
                let (_, grad) = model.potential(&z);
                for k in 0..model.dim() {
                    let h = 1e-6;
                    let mut zp = z.clone();
                    zp[k] += h;
                    let mut zm = z.clone();
                    zm[k] -= h;
                    let fd = (model.potential(&zp).0 - model.potential(&zm).0) / (2.0 * h);
                    assert!(
                        (fd - grad[k]).abs() <= 1e-6 * (1.0 + fd.abs()),
                        "{kernel:?} warp={warp} task={task} k={k}: {fd} vs {}",
                        grad[k]
                    );
                }
            }
        }
    }

    #[test]
    fn sites_are_sorted_by_name() {
        let model = PyroModel::new(spec(PyroKernel::Saas, true, true, false)).unwrap();
        let names: Vec<_> = model.sites().iter().map(|s| s.name).collect();
        assert_eq!(
            names,
            [
                "_kernel_inv_length_sq",
                "c0",
                "c1",
                "kernel_tausq",
                "latent_features",
                "mean",
                "noise",
                "outputscale",
                "task_lengthscale"
            ]
        );
    }
}
