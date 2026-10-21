//! BoTorch's `HigherOrderGP` (Zhe et al. 2019, "Scalable high-order
//! Gaussian process regression") and its posterior
//! (`HigherOrderGPPosterior`).
//!
//! The outputs of every input point form a tensor of shape `s₁ × … × s_k`
//! (BoTorch's `target_shape`); the covariance of the `N = n·S` training
//! values (`S = s₁⋯s_k`) is the Kronecker product
//!
//! ```text
//! K = K₀(X, X) ⊗ K₁(L₁, L₁) ⊗ … ⊗ K_k(L_k, L_k) + σ² I
//! ```
//!
//! of a kernel on the inputs and kernels on learned *latent* coordinates
//! `L_i` (`s_i × r_i`) of every output dimension, with a zero mean. All
//! computations use the eigendecompositions `K_i = Q_i Λ_i Q_iᵀ` and mode
//! products of the `n × s₁ × … × s_k` tensors (Kronecker algebra), as
//! linear_operator does:
//!
//! * the exact MLL `[-½ (yᵀ K⁻¹ y + log|K| + N log 2π) + log p(θ)] / N`
//!   and its gradient with respect to the raw hyperparameters, the latent
//!   coordinates and the noise;
//! * the posterior mean and the marginal variances
//!   (`make_posterior_variances`), the exact joint posterior covariance,
//!   and posterior samples by Matheron's rule from given base samples
//!   (`HigherOrderGPPosterior.rsample_from_base_samples`, BoTorch's layout
//!   of `(n + q)·S + n·S` base samples per draw, the root of the joint
//!   prior covariance being the Kronecker product of the factors'
//!   Cholesky factors).

use std::ops::Range;
use std::sync::OnceLock;

use nalgebra::SymmetricEigen;
use rand::rngs::StdRng;

use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::kernels::eval::TreePrep;
use crate::kernels::tree::{sample_tree_priors, KernelNode};
use crate::kernels::TreeKernel;
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::models::gp::GaussianLikelihoodSpec;
use crate::models::model::check_vjp_shapes_expanded;
use crate::models::Model;
use crate::optim::fit::MllModel;
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::transforms::input::{ChainedInputTransform, TransformMode};
use crate::{Matrix, Vector};

mod matheron;
mod variance;
pub use variance::{posterior_variances_factors_vjp, posterior_variances_from_factors};

/// `log(2π)`.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

// ---------------------------------------------------------------------------
// Dense tensors and mode products
// ---------------------------------------------------------------------------

/// A dense row-major tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// The dimensions.
    pub dims: Vec<usize>,
    /// The values (row-major).
    pub data: Vec<f64>,
}

impl Tensor {
    /// A tensor of dimensions `dims` from row-major values.
    ///
    /// # Panics
    /// If the number of values does not match.
    pub fn new(dims: Vec<usize>, data: Vec<f64>) -> Self {
        assert_eq!(
            dims.iter().product::<usize>(),
            data.len(),
            "tensor size mismatch"
        );
        Self { dims, data }
    }

    /// The mode-`mode` product `out[…, r, …] = Σ_a m[r, a] self[…, a, …]`.
    ///
    /// # Panics
    /// If `m` has the wrong number of columns.
    pub fn mode_mul(&self, mode: usize, m: &Matrix) -> Tensor {
        let n = self.dims[mode];
        assert_eq!(m.ncols(), n, "mode product dimension mismatch");
        let r = m.nrows();
        let outer: usize = self.dims[..mode].iter().product();
        let inner: usize = self.dims[mode + 1..].iter().product();
        let mut out = vec![0.0; outer * r * inner];
        for o in 0..outer {
            for a in 0..n {
                let src = &self.data[(o * n + a) * inner..(o * n + a + 1) * inner];
                for rr in 0..r {
                    let w = m[(rr, a)];
                    if w == 0.0 {
                        continue;
                    }
                    let dst = &mut out[(o * r + rr) * inner..(o * r + rr + 1) * inner];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d += w * s;
                    }
                }
            }
        }
        let mut dims = self.dims.clone();
        dims[mode] = r;
        Tensor { dims, data: out }
    }

    /// `Σ_{other indices} a[…, i, …] b[…, j, …]` (the mode-`mode`
    /// unfoldings' product `A₍ₘ₎ B₍ₘ₎ᵀ`).
    fn mode_gram(a: &Tensor, b: &Tensor, mode: usize) -> Matrix {
        let n = a.dims[mode];
        let outer: usize = a.dims[..mode].iter().product();
        let inner: usize = a.dims[mode + 1..].iter().product();
        let mut out = Matrix::zeros(n, n);
        for o in 0..outer {
            for i in 0..n {
                let ai = &a.data[(o * n + i) * inner..(o * n + i + 1) * inner];
                for j in 0..n {
                    let bj = &b.data[(o * n + j) * inner..(o * n + j + 1) * inner];
                    out[(i, j)] += ai.iter().zip(bj).map(|(x, y)| x * y).sum::<f64>();
                }
            }
        }
        out
    }
}

/// The Kronecker product of vectors (row-major over the factors).
fn kron_vec(vs: &[Vector]) -> Vec<f64> {
    let mut out = vec![1.0];
    for v in vs {
        let mut next = Vec::with_capacity(out.len() * v.len());
        for a in &out {
            for b in v.iter() {
                next.push(a * b);
            }
        }
        out = next;
    }
    out
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// The `MultivariateNormalPrior` of the latent coordinates of an output
/// dimension (BoTorch's `latent_init="gp"`): `N(0, C)` on the first latent
/// column (`r_i = 1`).
#[derive(Clone, Debug, PartialEq)]
pub struct LatentPrior {
    /// The covariance `C` (`s_i × s_i`).
    pub covariance: Matrix,
}

/// Options of [`HigherOrderGP::new`].
#[derive(Clone, Debug)]
pub struct HigherOrderGPOptions {
    /// The kernels: on the inputs, then on the latent coordinates of every
    /// output dimension.
    pub covar_modules: Vec<KernelNode>,
    /// The Gaussian likelihood.
    pub noise: GaussianLikelihoodSpec,
    /// The priors of the latent coordinates (`latent_init="gp"`), one per
    /// output dimension, or none.
    pub latent_priors: Vec<Option<LatentPrior>>,
    /// Whether the latent coordinates are learned (BoTorch's
    /// `learn_latent_pars`).
    pub learn_latent_pars: bool,
    /// The input transform applied to the points of posteriors (fitted on
    /// the training inputs, eval mode); the training inputs given to the
    /// model are already transformed.
    pub input_transform: Option<ChainedInputTransform>,
    /// `FlattenedStandardize`'s mean and standard deviation of all outputs
    /// (the targets given to the model are standardized), or none.
    pub outcome: Option<(f64, f64)>,
}

#[derive(Clone, Debug)]
struct Factor {
    kernel: TreeKernel,
    prep: TreePrep,
    k: Matrix,
    q: Matrix,
    lambda: Vector,
}

#[derive(Clone, Debug)]
struct Cache {
    factors: Vec<Factor>,
    noise: f64,
    /// `D = kron(λ) + σ²` (row-major over `n × s₁ × …`).
    d: Vec<f64>,
    /// `α = K⁻¹ y` as a tensor.
    alpha: Tensor,
    /// `α ×_{i≥1} K_i` (the coefficients of `K₀(x, X)` in the mean).
    alpha_lat: Tensor,
}

/// BoTorch's `HigherOrderGP` (unbatched; see the [module
/// documentation](self)).
#[derive(Clone, Debug)]
pub struct HigherOrderGP {
    train_x: Matrix,
    /// Standardized training targets, `n × S` (row-major over the target
    /// shape).
    train_y: Matrix,
    target_shape: Vec<usize>,
    latent_dims: Vec<usize>,
    dim: usize,
    options: HigherOrderGPOptions,
    raw: Vec<f64>,
    kernel_offsets: Vec<Range<usize>>,
    latent_offsets: Vec<Range<usize>>,
    cache: OnceLock<std::result::Result<Cache, BotorchError>>,
}

impl HigherOrderGP {
    /// Replace the fitted inverse affine map without refitting targets or noise.
    ///
    /// # Errors
    /// The offset must be finite and the scale finite and positive.
    pub fn set_fitted_outcome_affine(&mut self, outcome: Option<(f64, f64)>) -> Result<bool> {
        if outcome.is_some_and(|(offset, scale)| {
            !offset.is_finite() || !scale.is_finite() || scale <= 0.0
        }) {
            return Err(BotorchError::InvalidArgument(
                "Invalid fitted outcome statistics".into(),
            ));
        }
        let changed = self.options.outcome != outcome;
        self.options.outcome = outcome;
        Ok(changed)
    }

    /// A model from the transformed training inputs `train_x` (`n × d'`),
    /// the (standardized) targets `train_y` (`n × S`, row-major over
    /// `target_shape`), the number of latent coordinates of every output
    /// dimension, the raw input dimension `dim` and the flat raw
    /// parameters `raw` (the noise, every kernel's raw parameters, then the
    /// latent coordinates `L_i` row-major; `None` for the kernels' and the
    /// noise's initial values with zero latents).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent shapes and
    /// [`BotorchError::InputData`] for NaN data.
    pub fn new(
        train_x: Matrix,
        train_y: Matrix,
        target_shape: Vec<usize>,
        latent_dims: Vec<usize>,
        dim: usize,
        raw: Option<Vec<f64>>,
        options: HigherOrderGPOptions,
    ) -> Result<Self> {
        let k = target_shape.len();
        if k == 0 {
            return Err(BotorchError::DimensionMismatch(
                "a HigherOrderGP needs targets with at least one output dimension".into(),
            ));
        }
        if train_x.nrows() == 0
            || train_x.ncols() == 0
            || target_shape.contains(&0)
            || latent_dims.contains(&0)
        {
            return Err(BotorchError::InvalidArgument(
                "training inputs, target dimensions and latent dimensions must be nonempty".into(),
            ));
        }
        let s: usize = target_shape.iter().product();
        if train_y.nrows() != train_x.nrows() || train_y.ncols() != s {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} x {s} training targets, got {} x {}",
                train_x.nrows(),
                train_y.nrows(),
                train_y.ncols()
            )));
        }
        if latent_dims.len() != k || options.covar_modules.len() != k + 1 {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} latent dimensions and {} covariance modules, got {} and {}",
                k,
                k + 1,
                latent_dims.len(),
                options.covar_modules.len()
            )));
        }
        if options.latent_priors.len() != k && !options.latent_priors.is_empty() {
            return Err(BotorchError::DimensionMismatch(
                "one latent prior (or none) per output dimension".into(),
            ));
        }
        if train_x.iter().chain(train_y.iter()).any(|v| v.is_nan()) {
            return Err(BotorchError::InputData(
                "Input data contains NaN values.".into(),
            ));
        }
        let dt = train_x.ncols();
        options.covar_modules[0].validate(dt)?;
        for (i, node) in options.covar_modules[1..].iter().enumerate() {
            node.validate(latent_dims[i])?;
        }
        if let Some(chain) = &options.input_transform {
            if chain.points_per_input() != 1 {
                return Err(BotorchError::Unsupported(
                    "one-to-many input transforms in HigherOrderGP".into(),
                ));
            }
            chain.check_mode_dim(dim, dt, TransformMode::Eval)?;
        } else if dim != dt {
            return Err(BotorchError::DimensionMismatch(format!(
                "input dimension {dim} differs from the training inputs' {dt}"
            )));
        }
        if let Some((mean, stdv)) = options.outcome {
            if !mean.is_finite() || !stdv.is_finite() || stdv <= 0.0 {
                return Err(BotorchError::InvalidArgument(
                    "outcome statistics must be finite with positive standard deviation".into(),
                ));
            }
        }
        for (i, prior) in options.latent_priors.iter().enumerate() {
            if let Some(prior) = prior {
                if prior.covariance.shape() != (target_shape[i], target_shape[i]) {
                    return Err(BotorchError::DimensionMismatch(
                        "the latent prior covariance must match its output dimension".into(),
                    ));
                }
                if latent_dims[i] != 1 {
                    return Err(BotorchError::Unsupported(
                        "GP latent priors require one coordinate per output dimension".into(),
                    ));
                }
            }
        }
        let mut off = 1;
        let mut kernel_offsets = Vec::with_capacity(k + 1);
        for node in &options.covar_modules {
            kernel_offsets.push(off..off + node.num_raw());
            off += node.num_raw();
        }
        let mut latent_offsets = Vec::with_capacity(k);
        for i in 0..k {
            let len = target_shape[i] * latent_dims[i];
            latent_offsets.push(off..off + len);
            off += len;
        }
        let raw = match raw {
            Some(r) => {
                if r.len() != off {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "expected {off} raw parameters, got {}",
                        r.len()
                    )));
                }
                r
            }
            None => {
                let mut r = vec![0.0; off];
                r[0] = options.noise.initial_raw_noise();
                for (node, range) in options.covar_modules.iter().zip(&kernel_offsets) {
                    r[range.clone()].copy_from_slice(&node.initial_raw());
                }
                r
            }
        };
        if raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "the raw parameters must be finite".into(),
            ));
        }
        Ok(Self {
            train_x,
            train_y,
            target_shape,
            latent_dims,
            dim,
            options,
            raw,
            kernel_offsets,
            latent_offsets,
            cache: OnceLock::new(),
        })
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// The flat raw parameters.
    pub fn raw_parameters(&self) -> &[f64] {
        &self.raw
    }

    /// Sets the flat raw parameters.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong length and
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
            return Err(BotorchError::InvalidArgument(
                "the raw parameters must be finite".into(),
            ));
        }
        self.raw = raw.to_vec();
        self.cache = OnceLock::new();
        Ok(())
    }

    /// The target shape `s₁ × … × s_k`.
    pub fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    /// The number of outputs per point, `S`.
    pub fn num_targets(&self) -> usize {
        self.target_shape.iter().product()
    }

    /// The transformed training inputs.
    pub fn train_inputs(&self) -> &Matrix {
        &self.train_x
    }

    /// The standardized training targets (`n × S`).
    pub fn train_targets(&self) -> &Matrix {
        &self.train_y
    }

    /// The options.
    pub fn options(&self) -> &HigherOrderGPOptions {
        &self.options
    }

    /// The numbers of latent coordinates.
    pub fn latent_dims(&self) -> &[usize] {
        &self.latent_dims
    }

    /// `(name, size)` of every raw parameter (BoTorch's names in
    /// `ExactMarginalLogLikelihood(likelihood, model).named_parameters()`).
    pub fn parameter_names(&self) -> Vec<(String, usize)> {
        let mut out = vec![("likelihood.noise_covar.raw_noise".to_string(), 1)];
        for (i, node) in self.options.covar_modules.iter().enumerate() {
            node.visit_params(&format!("model.covar_modules.{i}."), &mut |name, p| {
                out.push((name, p.numel()));
            });
        }
        for (i, r) in self.latent_offsets.iter().enumerate() {
            out.push((format!("model.latent_parameters.{i}"), r.len()));
        }
        out
    }

    /// The optimizer bounds of the raw parameters.
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut b = vec![(f64::NEG_INFINITY, f64::INFINITY); self.raw.len()];
        b[0] = self.options.noise.noise_constraint.raw_bounds();
        for (node, r) in self.options.covar_modules.iter().zip(&self.kernel_offsets) {
            b[r.clone()].copy_from_slice(&node.raw_bounds());
        }
        b
    }

    /// The indices of the parameters that are optimized (all, except the
    /// latent coordinates without `learn_latent_pars`).
    pub fn free_parameters(&self) -> Vec<usize> {
        let end = if self.options.learn_latent_pars {
            self.raw.len()
        } else {
            self.latent_offsets
                .first()
                .map_or(self.raw.len(), |r| r.start)
        };
        (0..end).collect()
    }

    /// The noise variance (transformed outcome space).
    pub fn noise(&self) -> f64 {
        self.options.noise.noise_constraint.transform(self.raw[0])
    }

    fn latent_at(&self, raw: &[f64], i: usize) -> Matrix {
        Matrix::from_row_slice(
            self.target_shape[i],
            self.latent_dims[i],
            &raw[self.latent_offsets[i].clone()],
        )
    }

    /// The latent coordinates `L_i` (`s_i × r_i`).
    pub fn latent(&self, i: usize) -> Matrix {
        self.latent_at(&self.raw, i)
    }

    /// The kernel `i` (0: on the inputs) at the current parameters.
    ///
    /// # Errors
    /// See [`TreeKernel::build`].
    pub fn kernel(&self, i: usize) -> Result<TreeKernel> {
        self.kernel_at(&self.raw, i)
    }

    fn kernel_at(&self, raw: &[f64], i: usize) -> Result<TreeKernel> {
        let dimk = if i == 0 {
            self.train_x.ncols()
        } else {
            self.latent_dims[i - 1]
        };
        TreeKernel::build(
            &self.options.covar_modules[i],
            &raw[self.kernel_offsets[i].clone()],
            dimk,
        )
    }

    fn factors_at(&self, raw: &[f64]) -> Result<Vec<Factor>> {
        let k = self.target_shape.len();
        let mut out = Vec::with_capacity(k + 1);
        for i in 0..=k {
            let kernel = self.kernel_at(raw, i)?;
            let pts = if i == 0 {
                self.train_x.clone()
            } else {
                self.latent_at(raw, i - 1)
            };
            let prep = kernel.prepare(&pts)?;
            let km = kernel.symmetric(&prep);
            if km.iter().any(|v| !v.is_finite()) {
                return Err(BotorchError::Nan {
                    count: km.iter().filter(|v| v.is_nan()).count(),
                    total: km.len(),
                });
            }
            let eig = SymmetricEigen::new(km.clone());
            out.push(Factor {
                kernel,
                prep,
                k: km,
                q: eig.eigenvectors,
                lambda: eig.eigenvalues,
            });
        }
        Ok(out)
    }

    fn y_tensor(&self) -> Tensor {
        let mut dims = vec![self.train_x.nrows()];
        dims.extend(&self.target_shape);
        let mut data = Vec::with_capacity(self.train_y.len());
        for i in 0..self.train_y.nrows() {
            data.extend(self.train_y.row(i).iter());
        }
        Tensor::new(dims, data)
    }

    /// `(K + σ²)⁻¹ v` for a tensor `v` (`n × s₁ × …`).
    fn solve(factors: &[Factor], d: &[f64], v: &Tensor) -> Tensor {
        let mut t = v.clone();
        for (i, f) in factors.iter().enumerate() {
            t = t.mode_mul(i, &f.q.transpose());
        }
        for (x, di) in t.data.iter_mut().zip(d) {
            *x /= di;
        }
        for (i, f) in factors.iter().enumerate() {
            t = t.mode_mul(i, &f.q);
        }
        t
    }

    /// Solve the noisy training covariance against a tensor-shaped right-hand side.
    /// This reuses the Kronecker eigendecompositions without forming the full matrix.
    pub(crate) fn solve_training(&self, rhs: &Matrix) -> Result<Matrix> {
        if rhs.shape() != self.train_y.shape() {
            return Err(BotorchError::DimensionMismatch(
                "training solve right-hand side has the wrong shape".into(),
            ));
        }
        let cache = self.cache()?;
        let mut dims = vec![rhs.nrows()];
        dims.extend(&self.target_shape);
        let values = (0..rhs.nrows())
            .flat_map(|i| rhs.row(i).iter().copied().collect::<Vec<_>>())
            .collect();
        let solved = Self::solve(&cache.factors, &cache.d, &Tensor::new(dims, values));
        Ok(Matrix::from_row_slice(
            rhs.nrows(),
            rhs.ncols(),
            &solved.data,
        ))
    }

    fn compute_cache(&self) -> Result<Cache> {
        let factors = self.factors_at(&self.raw)?;
        let noise = self.noise();
        let lambdas: Vec<Vector> = factors.iter().map(|f| f.lambda.clone()).collect();
        let d: Vec<f64> = kron_vec(&lambdas).into_iter().map(|l| l + noise).collect();
        let alpha = Self::solve(&factors, &d, &self.y_tensor());
        let mut alpha_lat = alpha.clone();
        for (i, f) in factors.iter().enumerate().skip(1) {
            alpha_lat = alpha_lat.mode_mul(i, &f.k);
        }
        Ok(Cache {
            factors,
            noise,
            d,
            alpha,
            alpha_lat,
        })
    }

    fn cache(&self) -> Result<&Cache> {
        self.cache
            .get_or_init(|| self.compute_cache())
            .as_ref()
            .map_err(Clone::clone)
    }

    // ------------------------------------------------------------------
    // Marginal log likelihood
    // ------------------------------------------------------------------

    /// The exact MLL (divided by `N = n·S`, priors included) at the raw
    /// parameters `raw` and its gradient (if `with_grad`).
    ///
    /// # Errors
    /// [`BotorchError::Nan`] for NaN kernel matrices,
    /// [`BotorchError::InvalidArgument`] for parameters outside the support
    /// of their priors.
    pub fn mll_value_and_grad(&self, raw: &[f64], with_grad: bool) -> Result<(f64, Vec<f64>)> {
        if raw.len() != self.raw.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw parameters, got {}",
                self.raw.len(),
                raw.len()
            )));
        }
        let factors = self.factors_at(raw)?;
        let nc = self.options.noise.noise_constraint;
        let noise = nc.transform(raw[0]);
        let lambdas: Vec<Vector> = factors.iter().map(|f| f.lambda.clone()).collect();
        let d: Vec<f64> = kron_vec(&lambdas).into_iter().map(|l| l + noise).collect();
        let y = self.y_tensor();
        let alpha = Self::solve(&factors, &d, &y);
        let nn = y.data.len() as f64;
        let quad: f64 = y.data.iter().zip(&alpha.data).map(|(a, b)| a * b).sum();
        let logdet: f64 = d.iter().map(|v| v.ln()).sum();
        let mut value = -0.5 * (quad + logdet + nn * LN_2PI);
        let mut grad = vec![0.0; raw.len()];
        // Priors.
        let mut log_prior = 0.0;
        if let Some(p) = self.options.noise.noise_prior {
            log_prior += p.log_prob(noise);
            if with_grad {
                grad[0] += p.grad_log_prob(noise) / nn * nc.transform_derivative(raw[0]);
            }
        }
        for (i, f) in factors.iter().enumerate() {
            let r = self.kernel_offsets[i].clone();
            if with_grad {
                let (lp, g) = f.kernel.log_prior_and_grad()?;
                log_prior += lp;
                let deriv = self.options.covar_modules[i].transform_derivatives(&raw[r.clone()]);
                for (k, idx) in r.enumerate() {
                    grad[idx] += g[k] * deriv[k] / nn;
                }
            } else {
                log_prior += f.kernel.log_prior()?;
            }
        }
        for (i, prior) in self.options.latent_priors.iter().enumerate() {
            let Some(prior) = prior else { continue };
            let (lp, g) = mvn_log_prob_and_grad(&prior.covariance, &self.latent_at(raw, i))?;
            log_prior += lp;
            if with_grad {
                for (k, idx) in self.latent_offsets[i].clone().enumerate() {
                    grad[idx] += g[k] / nn;
                }
            }
        }
        value += log_prior;
        if !with_grad {
            return Ok((value / nn, grad));
        }
        // dF/dσ² = ½ (‖α‖² - Σ 1/D).
        let a2: f64 = alpha.data.iter().map(|v| v * v).sum();
        let inv_sum: f64 = d.iter().map(|v| 1.0 / v).sum();
        grad[0] += 0.5 * (a2 - inv_sum) / nn * nc.transform_derivative(raw[0]);
        // dF/dK_i = ½ (A₍ᵢ₎ B₍ᵢ₎ᵀ - Q_i diag(c) Q_iᵀ), B = α ×_{j≠i} K_j.
        let k = factors.len();
        let inv_d: Vec<f64> = d.iter().map(|v| 1.0 / v).collect();
        for i in 0..k {
            let mut b = alpha.clone();
            for (j, f) in factors.iter().enumerate() {
                if j != i {
                    b = b.mode_mul(j, &f.k);
                }
            }
            let s = Tensor::mode_gram(&alpha, &b, i);
            let s = (&s + s.transpose()) * 0.5;
            // c_l = Σ_{others} Π_{j≠i} λ_j / D.
            let mut weights = Vec::with_capacity(k);
            for (j, f) in factors.iter().enumerate() {
                if j == i {
                    weights.push(Vector::from_element(f.lambda.len(), 1.0));
                } else {
                    weights.push(f.lambda.clone());
                }
            }
            let wk = kron_vec(&weights);
            let ni = factors[i].lambda.len();
            let dims: Vec<usize> = factors.iter().map(|f| f.lambda.len()).collect();
            let inner: usize = dims[i + 1..].iter().product();
            let mut c = vec![0.0; ni];
            for (idx, (w, id)) in wk.iter().zip(&inv_d).enumerate() {
                c[(idx / inner) % ni] += w * id;
            }
            let q = &factors[i].q;
            let g = q * Matrix::from_diagonal(&Vector::from_vec(c)) * q.transpose();
            let w = (s - g) * (0.5 / nn);
            let gv = factors[i].kernel.contract(&factors[i].prep, &w);
            let r = self.kernel_offsets[i].clone();
            let deriv = self.options.covar_modules[i].transform_derivatives(&raw[r.clone()]);
            for (kk, idx) in r.enumerate() {
                grad[idx] += gv[kk] * deriv[kk];
            }
            if i >= 1 && self.options.learn_latent_pars {
                // The latent coordinates: 2 Σ_b W[a, b] ∂₁k(l_a, l_b).
                let li = i - 1;
                let ri = self.latent_dims[li];
                let off = self.latent_offsets[li].start;
                for a in 0..ni {
                    let (_, dk) =
                        factors[i]
                            .kernel
                            .cross_with_grad(&factors[i].prep, a, &factors[i].prep);
                    for l in 0..ri {
                        let mut gsum = 0.0;
                        for bb in 0..ni {
                            gsum += 2.0 * w[(a, bb)] * dk[(bb, l)];
                        }
                        grad[off + a * ri + l] += gsum;
                    }
                }
            }
        }
        Ok((value / nn, grad))
    }

    // ------------------------------------------------------------------
    // Posterior
    // ------------------------------------------------------------------

    fn check_columns(&self, x: &Matrix) -> Result<()> {
        if x.ncols() != self.dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected inputs with {} columns, got {}",
                self.dim,
                x.ncols()
            )));
        }
        Ok(())
    }

    /// The raw-space points transformed by the input transform (eval mode).
    ///
    /// # Errors
    /// For a column mismatch.
    pub fn transform_inputs(&self, x: &Matrix) -> Result<Matrix> {
        self.check_columns(x)?;
        match &self.options.input_transform {
            Some(chain) => chain.transform_mode(x, TransformMode::Eval),
            None => Ok(x.clone()),
        }
    }

    fn affine(&self) -> (f64, f64) {
        self.options.outcome.unwrap_or((0.0, 1.0))
    }

    /// The posterior mean (`q × S`) and marginal variances (`q × S`) at the
    /// raw-space points `x`, in the original outcome space (BoTorch's
    /// `posterior(X).mean` and `.variance`, from `make_posterior_variances`).
    ///
    /// # Errors
    /// See [`HigherOrderGP::posterior`].
    pub fn mean_and_variance(
        &self,
        x: &Matrix,
        observation_noise: bool,
    ) -> Result<(Matrix, Matrix)> {
        self.mean_and_variance_in(x, observation_noise, true)
    }

    /// [`HigherOrderGP::mean_and_variance`] in the original
    /// (`original_space`) or the standardized outcome space.
    ///
    /// # Errors
    /// See [`HigherOrderGP::posterior`].
    pub fn mean_and_variance_in(
        &self,
        x: &Matrix,
        observation_noise: bool,
        original_space: bool,
    ) -> Result<(Matrix, Matrix)> {
        let xt = self.transform_inputs(x)?;
        let c = self.cache()?;
        let f0 = &c.factors[0];
        let xp = f0.kernel.prepare(&xt)?;
        let kxs = f0.kernel.cross(&xp, &f0.prep); // q x n
        let q = xt.nrows();
        let s = self.num_targets();
        let mean_t = c.alpha_lat.mode_mul(0, &kxs);
        // var = diag(K**) ⊗ diag(K_lat) - (H₀ ⊗ H₁ ⊗ …) (1/D), H_i = (K_i Q_i)∘².
        let mut h = Tensor::new(c.alpha.dims.clone(), c.d.iter().map(|v| 1.0 / v).collect());
        let p0 = &kxs * &f0.q;
        h = h.mode_mul(0, &p0.map(|v| v * v));
        let mut diag_lat = Vec::with_capacity(c.factors.len() - 1);
        for (i, f) in c.factors.iter().enumerate().skip(1) {
            let pi = &f.k * &f.q;
            h = h.mode_mul(i, &pi.map(|v| v * v));
            diag_lat.push(f.k.diagonal());
        }
        let kd = kron_vec(&diag_lat);
        let (om, os) = if original_space {
            self.affine()
        } else {
            (0.0, 1.0)
        };
        let noise = if observation_noise { c.noise } else { 0.0 };
        let mut mean = Matrix::zeros(q, s);
        let mut var = Matrix::zeros(q, s);
        for p in 0..q {
            let kpp = f0.kernel.diag(&xp, p);
            for t in 0..s {
                mean[(p, t)] = om + os * mean_t.data[p * s + t];
                var[(p, t)] = os * os * (kpp * kd[t] - h.data[p * s + t] + noise);
            }
        }
        Ok((mean, var))
    }

    /// The exact joint posterior at transformed points `xt` over the `q·S`
    /// values (output-major: index `t·q + p` for output `t` at point `p`),
    /// in the original outcome space, with the factors' quantities needed
    /// for derivatives.
    fn joint_transformed(&self, xt: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let c = self.cache()?;
        let f0 = &c.factors[0];
        let xp = f0.kernel.prepare(xt)?;
        let kxs = f0.kernel.cross(&xp, &f0.prep); // q x n
        let kss = f0.kernel.symmetric(&xp);
        let q = xt.nrows();
        let s = self.num_targets();
        let mean_t = c.alpha_lat.mode_mul(0, &kxs);
        // Σ = K** ⊗ Kl - P D⁻¹ Pᵀ with P = (K*₀Q₀) ⊗ (K₁Q₁) ⊗ ….
        let mut p = &kxs * &f0.q;
        let mut kl = Matrix::from_element(1, 1, 1.0);
        for f in c.factors.iter().skip(1) {
            p = p.kronecker(&(&f.k * &f.q));
            kl = kl.kronecker(&f.k);
        }
        let inv_d = Vector::from_iterator(c.d.len(), c.d.iter().map(|v| 1.0 / v));
        let pd = Matrix::from_fn(p.nrows(), p.ncols(), |a, b| p[(a, b)] * inv_d[b]);
        let cov_pm = kss.kronecker(&kl) - pd * p.transpose(); // point-major (p s + t)
        let (om, os) = self.affine();
        let qs = q * s;
        let mut mean = Vector::zeros(qs);
        let mut cov = Matrix::zeros(qs, qs);
        let idx = |k: usize| (k % q) * s + k / q; // output-major k = t q + p -> p s + t
        for a in 0..qs {
            mean[a] = om + os * mean_t.data[idx(a)];
            for b in 0..qs {
                cov[(a, b)] = os * os * cov_pm[(idx(a), idx(b))];
            }
            if observation_noise {
                cov[(a, a)] += os * os * c.noise;
            }
        }
        Ok(Posterior::multi_output(mean, cov, s))
    }

    /// The exact joint posterior at the raw-space points `x` over all `q·S`
    /// values (output-major; the target tensors flattened row-major).
    ///
    /// # Errors
    /// For a column mismatch and numerical failures.
    pub fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let xt = self.transform_inputs(x)?;
        self.joint_transformed(&xt, observation_noise)
    }

    fn joint_with_grad_transformed(&self, xt: &Matrix) -> Result<JointPosteriorGrad> {
        let c = self.cache()?;
        let f0 = &c.factors[0];
        let post = self.joint_transformed(xt, false)?;
        let xp = f0.kernel.prepare(xt)?;
        let kxs = f0.kernel.cross(&xp, &f0.prep);
        let (q, dt) = xt.shape();
        let s = self.num_targets();
        let qs = q * s;
        let (_, os) = self.affine();
        // M = (Q₀ ⊗ K₁Q₁ ⊗ …) D⁻¹ (…)ᵀ (N × N): Σ = K**⊗Kl - (K*₀ ⊗ I) M (K*₀ ⊗ I)ᵀ.
        let mut r = f0.q.clone();
        let mut kl = Matrix::from_element(1, 1, 1.0);
        for f in c.factors.iter().skip(1) {
            r = r.kronecker(&(&f.k * &f.q));
            kl = kl.kronecker(&f.k);
        }
        let inv_d: Vec<f64> = c.d.iter().map(|v| 1.0 / v).collect();
        let rd = Matrix::from_fn(r.nrows(), r.ncols(), |a, b| r[(a, b)] * inv_d[b]);
        let mm = rd * r.transpose();
        let eye_s = Matrix::identity(s, s);
        let cmat = kxs.kronecker(&eye_s); // qS x N (point-major)
        let alat = &c.alpha_lat; // n x S
        let n = kxs.ncols();
        let idx = |k: usize| (k % q) * s + k / q;
        let mut dmean = vec![Vector::zeros(qs); q * dt];
        let mut dcov = vec![Matrix::zeros(qs, qs); q * dt];
        for p in 0..q {
            let (_, dcx) = f0.kernel.cross_with_grad(&xp, p, &f0.prep); // n x dt
            let (_, dkx) = f0.kernel.cross_with_grad(&xp, p, &xp); // q x dt
            let (_, dkpp) = f0.kernel.diag_with_grad(&xp, p);
            for l in 0..dt {
                let j = p * dt + l;
                // d mean[p, t] = Σ_a dK*[p, a] α_lat[a, t].
                let mut dm_pm = vec![0.0; qs];
                for t in 0..s {
                    let mut v = 0.0;
                    for a in 0..n {
                        v += dcx[(a, l)] * alat.data[a * s + t];
                    }
                    dm_pm[p * s + t] = os * v;
                }
                // dC (only rows p·S..p·S+S): dC[p s + t, a s + t'] = dK*[p, a] δ_tt'.
                // d(C M Cᵀ) = dC M Cᵀ + C M dCᵀ; (dC M)[p s + t, :] = Σ_a dK*[p, a] M[a s + t, :].
                let mut dcm = Matrix::zeros(s, n * s);
                for t in 0..s {
                    for a in 0..n {
                        let w = dcx[(a, l)];
                        if w == 0.0 {
                            continue;
                        }
                        for col in 0..n * s {
                            dcm[(t, col)] += w * mm[(a * s + t, col)];
                        }
                    }
                }
                let dcmc = &dcm * cmat.transpose(); // s x qS
                let mut dc_pm = Matrix::zeros(qs, qs);
                for t in 0..s {
                    for b in 0..qs {
                        dc_pm[(p * s + t, b)] -= dcmc[(t, b)];
                        dc_pm[(b, p * s + t)] -= dcmc[(t, b)];
                    }
                }
                // d(K** ⊗ Kl): rows/cols of point p.
                for pp in 0..q {
                    let dk = if pp == p { dkpp[l] } else { dkx[(pp, l)] };
                    for t in 0..s {
                        for u in 0..s {
                            let v = dk * kl[(t, u)];
                            if pp == p {
                                dc_pm[(p * s + t, p * s + u)] += v;
                            } else {
                                dc_pm[(p * s + t, pp * s + u)] += v;
                                dc_pm[(pp * s + u, p * s + t)] += v;
                            }
                        }
                    }
                }
                for a in 0..qs {
                    dmean[j][a] = dm_pm[idx(a)];
                    for b in 0..qs {
                        dcov[j][(a, b)] = os * os * dc_pm[(idx(a), idx(b))];
                    }
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean: post.mean,
            covariance: post.covariance,
            dmean,
            dcov,
            num_outputs: s,
        })
    }

    /// The joint posterior with its derivatives with respect to the
    /// raw-space points (forward mode; dense, for small `q·S`).
    ///
    /// # Errors
    /// See [`HigherOrderGP::posterior`].
    pub fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.check_columns(x)?;
        let Some(chain) = &self.options.input_transform else {
            return self.joint_with_grad_transformed(x);
        };
        let (xt, tape) = chain.transform_with_tape(x)?;
        let g = self.joint_with_grad_transformed(&xt)?;
        let jacs = tape.jacobians_through(chain)?;
        let (q, d) = x.shape();
        let dt = xt.ncols();
        let len = g.mean.len();
        let mut dmean = vec![Vector::zeros(len); q * d];
        let mut dcov = vec![Matrix::zeros(len, len); q * d];
        for i in 0..q {
            for l in 0..dt {
                for j in 0..d {
                    let f = jacs[i][(l, j)];
                    if f != 0.0 {
                        dmean[i * d + j].axpy(f, &g.dmean[i * dt + l], 1.0);
                        dcov[i * d + j] += &g.dcov[i * dt + l] * f;
                    }
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean: g.mean,
            covariance: g.covariance,
            dmean,
            dcov,
            num_outputs: g.num_outputs,
        })
    }

    /// Posterior samples by Matheron's rule from BoTorch's base samples
    /// (`HigherOrderGPPosterior.rsample_from_base_samples`): `base` is `S_mc
    /// × ((n + q)·S + n·S)` (the joint prior draws, then the training noise
    /// draws); returns `S_mc × (q·S)` samples (point-major, original outcome
    /// space).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for base samples of the wrong
    /// size, and errors of the Cholesky factorizations.
    pub fn rsample_from_base_samples(&self, x: &Matrix, base: &Matrix) -> Result<Matrix> {
        self.rsample_in(x, base, true)
    }

    /// [`HigherOrderGP::rsample_from_base_samples`] in the original
    /// (`original_space`) or the standardized outcome space.
    ///
    /// # Errors
    /// See [`HigherOrderGP::rsample_from_base_samples`].
    pub fn rsample_in(&self, x: &Matrix, base: &Matrix, original_space: bool) -> Result<Matrix> {
        let xt = self.transform_inputs(x)?;
        let c = self.cache()?;
        let f0 = &c.factors[0];
        let q = xt.nrows();
        let n = self.train_x.nrows();
        let s = self.num_targets();
        let joint = (n + q) * s;
        let total = joint + n * s;
        if base.ncols() != total {
            return Err(BotorchError::DimensionMismatch(format!(
                "base samples must have {total} entries per sample ((n + q)·S + n·S), got {}",
                base.ncols()
            )));
        }
        // Roots: Cholesky factors of the joint data covariance and of the
        // latent kernels (linear_operator's Kronecker Cholesky).
        let mut stacked = Matrix::zeros(n + q, xt.ncols());
        for i in 0..n {
            stacked.row_mut(i).copy_from(&self.train_x.row(i));
        }
        for i in 0..q {
            stacked.row_mut(n + i).copy_from(&xt.row(i));
        }
        let sp = f0.kernel.prepare(&stacked)?;
        let mut kj = f0.kernel.symmetric(&sp);
        // The train-train block is the training covariance itself.
        for a in 0..n {
            for b in 0..n {
                kj[(a, b)] = f0.k[(a, b)];
            }
        }
        let mut roots: Vec<CholeskyFactor> = vec![psd_safe_cholesky(&kj)?];
        for f in c.factors.iter().skip(1) {
            roots.push(psd_safe_cholesky(&f.k)?);
        }
        let xp = f0.kernel.prepare(&xt)?;
        let kxs = f0.kernel.cross(&xp, &f0.prep);
        let noise_sd = c.noise.sqrt();
        let (om, os) = if original_space {
            self.affine()
        } else {
            (0.0, 1.0)
        };
        let y = self.y_tensor();
        let mut dims_joint = vec![n + q];
        dims_joint.extend(&self.target_shape);
        let mut out = Matrix::zeros(base.nrows(), q * s);
        for smp in 0..base.nrows() {
            let z: Vec<f64> = (0..joint).map(|k| base[(smp, k)]).collect();
            let mut t = Tensor::new(dims_joint.clone(), z);
            for (i, r) in roots.iter().enumerate() {
                t = t.mode_mul(i, r.l());
            }
            // Training part: the first n·S values; test part: the rest.
            let mut rhs = y.clone();
            for k in 0..n * s {
                let train = t.data[k] + noise_sd * base[(smp, joint + k)];
                rhs.data[k] -= train;
            }
            let kinv = Self::solve(&c.factors, &c.d, &rhs);
            let mut upd = kinv.mode_mul(0, &kxs);
            for (i, f) in c.factors.iter().enumerate().skip(1) {
                upd = upd.mode_mul(i, &f.k);
            }
            for k in 0..q * s {
                let v = t.data[n * s + k] + upd.data[k];
                out[(smp, k)] = om + os * v;
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Conditioning
    // ------------------------------------------------------------------

    /// The model conditioned on new observations (BoTorch's
    /// `condition_on_observations`, GPyTorch's `get_fantasy_model`): the
    /// transformed points `x_tf` and standardized targets `y_tf` (`n' × S`)
    /// appended to the training data, with the same hyperparameters.
    ///
    /// # Errors
    /// For shape mismatches.
    pub fn condition_on_observations(&self, x_tf: &Matrix, y_tf: &Matrix) -> Result<Self> {
        if x_tf.ncols() != self.train_x.ncols()
            || y_tf.ncols() != self.num_targets()
            || x_tf.nrows() != y_tf.nrows()
        {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {}-column inputs and {}-column targets with the same number of rows",
                self.train_x.ncols(),
                self.num_targets()
            )));
        }
        let n = self.train_x.nrows();
        let m = x_tf.nrows();
        let x = Matrix::from_fn(n + m, self.train_x.ncols(), |i, j| {
            if i < n {
                self.train_x[(i, j)]
            } else {
                x_tf[(i - n, j)]
            }
        });
        let y = Matrix::from_fn(n + m, self.num_targets(), |i, j| {
            if i < n {
                self.train_y[(i, j)]
            } else {
                y_tf[(i - n, j)]
            }
        });
        Self::new(
            x,
            y,
            self.target_shape.clone(),
            self.latent_dims.clone(),
            self.dim,
            Some(self.raw.clone()),
            self.options.clone(),
        )
    }

    /// BoTorch's `sample_all_priors(model)`: kernel parameters with priors
    /// drawn from them (the noise prior too; latent priors have no setting
    /// closure in BoTorch and keep their values).
    pub fn sample_all_priors(&self, raw: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let mut out = raw.to_vec();
        if let Some(p) = self.options.noise.noise_prior {
            out[0] = self
                .options
                .noise
                .noise_constraint
                .inverse_transform(p.sample(rng));
        }
        for (node, r) in self.options.covar_modules.iter().zip(&self.kernel_offsets) {
            sample_tree_priors(node, &mut out[r.clone()], rng);
        }
        out
    }
}

/// `log N(x; 0, C)` of the first column of `latent` (BoTorch's
/// `MultivariateNormalPrior(..., transform=lambda x: x.squeeze(-1))` for
/// one latent coordinate) and its gradient with respect to `latent`
/// (row-major).
fn mvn_log_prob_and_grad(cov: &Matrix, latent: &Matrix) -> Result<(f64, Vec<f64>)> {
    if latent.ncols() != 1 {
        return Err(BotorchError::Unsupported(
            "latent_init='gp' with more than one latent coordinate per output dimension".into(),
        ));
    }
    let s = cov.nrows();
    let x = Vector::from_iterator(s, latent.column(0).iter().copied());
    let chol = psd_safe_cholesky(cov)?;
    let a = chol.solve_vec(&x);
    let lp = -0.5 * (x.dot(&a) + chol.logdet() + s as f64 * LN_2PI);
    Ok((lp, (-a).iter().copied().collect()))
}

impl Model for HigherOrderGP {
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_outputs(&self) -> usize {
        self.num_targets()
    }

    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        HigherOrderGP::posterior(self, x, observation_noise)
    }

    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        HigherOrderGP::joint_posterior_with_grad(self, x)
    }

    fn joint_posterior_vjp(
        &self,
        x: &Matrix,
        rows: Range<usize>,
        mean_adjoint: &Vector,
        covariance_adjoint: &Matrix,
    ) -> Result<Matrix> {
        check_vjp_shapes_expanded(
            x,
            &rows,
            mean_adjoint,
            covariance_adjoint,
            self.num_targets(),
            1,
        )?;
        let g = self.joint_posterior_with_grad(x)?;
        let d = x.ncols();
        let mut out = Matrix::zeros(rows.len(), d);
        for (r, p) in rows.enumerate() {
            for k in 0..d {
                let j = p * d + k;
                out[(r, k)] = mean_adjoint.dot(&g.dmean[j])
                    + covariance_adjoint.component_mul(&g.dcov[j]).sum();
            }
        }
        Ok(out)
    }
}

/// The HOGP's exact MLL as the objective of [`crate::fit::fit_gpytorch_mll`]
/// (BoTorch's `fit_gpytorch_mll_scipy` with retries) and of
/// [`crate::optim::fit::fit_gpytorch_mll_torch`]; the latent coordinates are
/// optimized only with `learn_latent_pars`.
#[derive(Clone, Debug)]
pub struct HogpFit {
    /// The model (holds the fitted parameters afterwards).
    pub model: HigherOrderGP,
}

impl Fittable for HogpFit {
    type Output = FitResult;

    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        let full = self.model.raw.clone();
        let free = self.model.free_parameters();
        let bounds_all = self.model.raw_parameter_bounds();
        let x0: Vec<f64> = free.iter().map(|&i| full[i]).collect();
        let bounds: Vec<(f64, f64)> = free.iter().map(|&i| bounds_all[i]).collect();
        let (result, x) = {
            let m = &self.model;
            let expand = |x: &[f64]| {
                let mut f = full.clone();
                for (j, &i) in free.iter().enumerate() {
                    f[i] = x[j];
                }
                f
            };
            minimize_with_retries(
                &x0,
                &bounds,
                options,
                |x| {
                    let (v, g) = m.mll_value_and_grad(&expand(x), true)?;
                    Ok((v, free.iter().map(|&i| g[i]).collect()))
                },
                |rng| {
                    let s = m.sample_all_priors(&full, rng);
                    free.iter().map(|&i| s[i]).collect()
                },
            )?
        };
        let mut f = full;
        for (j, &i) in free.iter().enumerate() {
            f[i] = x[j];
        }
        self.model.set_raw_parameters(&f)?;
        Ok(result)
    }
}

impl MllModel for HogpFit {
    fn raw_parameter_vector(&self) -> Vec<f64> {
        self.model.raw.clone()
    }

    fn set_raw_parameter_vector(&mut self, values: &[f64]) -> Result<()> {
        self.model.set_raw_parameters(values)
    }

    fn raw_bounds(&self) -> Vec<(f64, f64)> {
        self.model.raw_parameter_bounds()
    }

    fn mll_values_and_gradient(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let (v, mut g) = self.model.mll_value_and_grad(x, true)?;
        if !self.model.options.learn_latent_pars {
            for r in &self.model.latent_offsets {
                for i in r.clone() {
                    g[i] = 0.0;
                }
            }
        }
        Ok((vec![v], g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{GreaterThan, ParameterTransform};
    use crate::kernels::KernelKind;
    use crate::models::gp::get_gaussian_likelihood_with_lognormal_prior;

    pub(super) fn model(prior: bool, k3: bool) -> HigherOrderGP {
        let n = 5;
        let shape = if k3 { vec![2, 3] } else { vec![3] };
        let s: usize = shape.iter().product();
        let x = Matrix::from_fn(n, 2, |i, j| ((i * 3 + j * 5) % 7) as f64 / 7.0);
        let y = Matrix::from_fn(n, s, |i, t| ((i + 1) as f64 * 0.7 + t as f64).sin());
        let mut covs = vec![KernelNode::stationary(KernelKind::Rbf, Some(2)).unwrap()];
        for _ in &shape {
            covs.push(KernelNode::scale(
                KernelNode::stationary(KernelKind::Matern52, Some(1)).unwrap(),
            ));
        }
        let latent_priors = shape
            .iter()
            .map(|&si| {
                prior.then(|| LatentPrior {
                    covariance: Matrix::from_fn(si, si, |a, b| {
                        (-((a as f64 - b as f64).powi(2)) / 2.0).exp()
                            + if a == b { 0.1 } else { 0.0 }
                    }),
                })
            })
            .collect();
        let options = HigherOrderGPOptions {
            covar_modules: covs,
            noise: if prior {
                get_gaussian_likelihood_with_lognormal_prior()
            } else {
                GaussianLikelihoodSpec {
                    noise_prior: None,
                    noise_constraint: GreaterThan::new(
                        1e-4,
                        Some(ParameterTransform::Softplus),
                        None,
                    )
                    .into(),
                }
            },
            latent_priors,
            learn_latent_pars: true,
            input_transform: None,
            outcome: Some((0.3, 1.7)),
        };
        let mut m = HigherOrderGP::new(x, y, shape.clone(), vec![1; shape.len()], 2, None, options)
            .unwrap();
        let mut raw = m.raw_parameters().to_vec();
        for (k, v) in raw.iter_mut().enumerate() {
            *v += 0.2 * ((k as f64) * 1.37).sin() + if k == 0 { 0.05 } else { 0.0 };
        }
        m.set_raw_parameters(&raw).unwrap();
        m
    }

    #[test]
    fn mll_matches_dense_and_gradient_matches_finite_differences() {
        for (prior, k3) in [(false, false), (true, true)] {
            let m = model(prior, k3);
            let raw = m.raw_parameters().to_vec();
            let (v, g) = m.mll_value_and_grad(&raw, true).unwrap();
            // Dense reference value.
            let c = m.factors_at(&raw).unwrap();
            let mut kd = c[0].k.clone();
            for f in &c[1..] {
                kd = kd.kronecker(&f.k);
            }
            let nn = kd.nrows();
            let noise = m.noise();
            for i in 0..nn {
                kd[(i, i)] += noise;
            }
            let y = Vector::from_vec(m.y_tensor().data);
            let chol = psd_safe_cholesky(&kd).unwrap();
            let a = chol.solve_vec(&y);
            let dense = -0.5 * (y.dot(&a) + chol.logdet() + nn as f64 * LN_2PI) / nn as f64;
            if !prior {
                assert!(
                    (v - dense).abs() < 1e-12 * dense.abs().max(1.0),
                    "{v} vs {dense}"
                );
            }
            let gmax = g.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            for k in 0..raw.len() {
                let h = 1e-6;
                let mut rp = raw.clone();
                rp[k] += h;
                let mut rm = raw.clone();
                rm[k] -= h;
                let fd = (m.mll_value_and_grad(&rp, false).unwrap().0
                    - m.mll_value_and_grad(&rm, false).unwrap().0)
                    / (2.0 * h);
                assert!(
                    (fd - g[k]).abs() < 1e-6 * gmax.max(1.0),
                    "param {k}: {fd} vs {}",
                    g[k]
                );
            }
            assert!(v.is_finite());
        }
    }

    #[test]
    fn posterior_matches_dense_formulas_and_samples() {
        let m = model(false, true);
        let x = Matrix::from_row_slice(2, 2, &[0.15, 0.8, 0.6, 0.35]);
        let (mean, var) = m.mean_and_variance(&x, false).unwrap();
        let post = m.posterior(&x, false).unwrap();
        let (q, s) = (2, m.num_targets());
        for p in 0..q {
            for t in 0..s {
                assert!((post.mean[t * q + p] - mean[(p, t)]).abs() < 1e-12);
                assert!((post.covariance[(t * q + p, t * q + p)] - var[(p, t)]).abs() < 1e-12);
            }
        }
        // Dense exact posterior.
        let c = m.cache().unwrap();
        let mut kl = Matrix::from_element(1, 1, 1.0);
        for f in &c.factors[1..] {
            kl = kl.kronecker(&f.k);
        }
        let k0 = &c.factors[0];
        let xp = k0.kernel.prepare(&x).unwrap();
        let kxs = k0.kernel.cross(&xp, &k0.prep).kronecker(&kl);
        let kss = k0.kernel.symmetric(&xp).kronecker(&kl);
        let mut ktt = k0.k.kronecker(&kl);
        for i in 0..ktt.nrows() {
            ktt[(i, i)] += c.noise;
        }
        let chol = psd_safe_cholesky(&ktt).unwrap();
        let y = Vector::from_vec(m.y_tensor().data);
        let mu = &kxs * chol.solve_vec(&y);
        let cov = &kss - &kxs * chol.solve(&kxs.transpose());
        let (om, os) = m.affine();
        for p in 0..q {
            for t in 0..s {
                assert!((mean[(p, t)] - (om + os * mu[p * s + t])).abs() < 1e-10);
                for p2 in 0..q {
                    for t2 in 0..s {
                        let e = os * os * cov[(p * s + t, p2 * s + t2)];
                        assert!((post.covariance[(t * q + p, t2 * q + p2)] - e).abs() < 1e-10);
                    }
                }
            }
        }
        // Matheron samples: zero base samples give the posterior mean of the
        // training noise-free data only when y is the prior draw... check
        // linearity instead: the samples are affine in the base samples.
        let n = m.train_x.nrows();
        let total = (n + q) * s + n * s;
        let b0 = Matrix::zeros(1, total);
        let s0 = m.rsample_from_base_samples(&x, &b0).unwrap();
        for k in 0..q * s {
            assert!((s0[(0, k)] - mean[(k / s, k % s)]).abs() < 1e-10);
        }
    }

    #[test]
    fn posterior_gradients_match_finite_differences() {
        let m = model(true, false);
        let x = Matrix::from_row_slice(2, 2, &[0.15, 0.8, 0.6, 0.35]);
        let g = m.joint_posterior_with_grad(&x).unwrap();
        for p in 0..2 {
            for k in 0..2 {
                let h = 1e-6;
                let mut xp = x.clone();
                xp[(p, k)] += h;
                let mut xm = x.clone();
                xm[(p, k)] -= h;
                let pp = m.posterior(&xp, false).unwrap();
                let pm = m.posterior(&xm, false).unwrap();
                let dm = (&pp.mean - &pm.mean) / (2.0 * h);
                let dc = (&pp.covariance - &pm.covariance) / (2.0 * h);
                assert!((&dm - &g.dmean[p * 2 + k]).amax() < 1e-6);
                assert!((&dc - &g.dcov[p * 2 + k]).amax() < 1e-6);
            }
        }
    }
}
