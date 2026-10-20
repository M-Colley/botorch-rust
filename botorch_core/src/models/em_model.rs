//! Exact conditioning and analytic fitting of fixed empirical inducing priors.

use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::kernels::{KernelNode, TreeKernel};
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::models::em_gp::{EMDataset, EMInterpolation};
use crate::models::{ConstantMeanSpec, GaussianLikelihoodSpec, Model};
use crate::optim::fit::MllModel;
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::{Matrix, Vector};

/// Immutable empirical inducing moments, extended by a mutable parametric GP.
#[derive(Clone, Debug)]
pub struct EMPrior {
    /// Inducing locations.
    pub x: Matrix,
    /// Estimated mean at inducing points.
    pub mean: Vector,
    /// Estimated covariance at inducing points.
    pub covariance: Matrix,
}

/// Native configuration for an EM-prior-conditioned exact scalar GP.
#[derive(Clone, Debug)]
pub struct EMModelOptions {
    /// Parametric covariance used by shift interpolation.
    pub kernel: KernelNode,
    /// Current raw parametric covariance values.
    pub kernel_raw: Vec<f64>,
    /// A constant mean specification, or a zero mean when absent.
    pub mean: Option<ConstantMeanSpec>,
    /// Current constant (ignored for zero mean).
    pub constant: f64,
    /// Gaussian observation model.
    pub likelihood: GaussianLikelihoodSpec,
    /// Initial raw observation variance.
    pub noise_raw: f64,
    /// Optional full additive covariance and its raw parameters.
    pub base: Option<(KernelNode, Vec<f64>)>,
    /// Predict through shift interpolation instead of nearest inducing lookup.
    pub interpolate: bool,
    /// Historical datasets explicitly used inducing-point interpolation.
    pub inducing_observations: bool,
    /// Freeze the parametric mean and interpolation kernel during fitting.
    pub freeze_pretrained: bool,
}

#[derive(Clone, Debug)]
struct Cache {
    interpolation: EMInterpolation,
    base: Option<TreeKernel>,
    factor: CholeskyFactor,
    alpha: Vector,
}

/// Exact GP conditioned on target data, retaining fixed EM historical moments.
#[derive(Clone, Debug)]
pub struct EMEmpiricalGP {
    train_x: Matrix,
    train_y: Vector,
    datasets: Vec<EMDataset>,
    prior: EMPrior,
    options: EMModelOptions,
    raw: Vec<f64>,
    cache: Cache,
}

fn index_rows(x: &Matrix, inducing: &Matrix) -> Result<Vec<usize>> {
    (0..x.nrows())
        .map(|r| {
            let (idx, distance) = (0..inducing.nrows())
                .map(|i| {
                    let distance = (0..x.ncols())
                        .map(|j| (x[(r, j)] - inducing[(i, j)]).powi(2))
                        .sum::<f64>();
                    (i, distance)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("validated nonempty inducing inputs");
            if distance > 1e-12 {
                Err(BotorchError::InvalidArgument(
                    "query point not found in historical inputs; enable interpolation".into(),
                ))
            } else {
                Ok(idx)
            }
        })
        .collect()
}

impl EMEmpiricalGP {
    /// Validate fixed moments and construct the exact target conditioning cache.
    pub fn new(
        train_x: Matrix,
        train_y: Vector,
        datasets: Vec<EMDataset>,
        prior: EMPrior,
        options: EMModelOptions,
    ) -> Result<Self> {
        let n = prior.x.nrows();
        let d = prior.x.ncols();
        if n == 0
            || d == 0
            || prior.mean.len() != n
            || prior.covariance.shape() != (n, n)
            || prior
                .x
                .iter()
                .chain(prior.mean.iter())
                .chain(prior.covariance.iter())
                .any(|v| !v.is_finite())
            || (&prior.covariance - prior.covariance.transpose()).amax() > 1e-9
            || datasets.is_empty()
        {
            return Err(BotorchError::DimensionMismatch(
                "invalid fixed EM prior".into(),
            ));
        }
        let eigen = prior.covariance.clone().symmetric_eigen();
        if eigen.eigenvalues.min() < -1e-9 {
            return Err(BotorchError::InvalidArgument(
                "EM inducing covariance must be positive semidefinite".into(),
            ));
        }
        for data in datasets.iter().chain(std::iter::once(&EMDataset {
            x: train_x.clone(),
            y: train_y.clone(),
        })) {
            if data.x.nrows() == 0
                || data.x.ncols() != d
                || data.y.len() != data.x.nrows()
                || data.x.iter().chain(data.y.iter()).any(|v| !v.is_finite())
            {
                return Err(BotorchError::DimensionMismatch(
                    "EM models require finite n x d inputs and n outcomes".into(),
                ));
            }
        }
        if options.kernel_raw.len() != options.kernel.num_raw()
            || options
                .base
                .as_ref()
                .is_some_and(|(n, p)| n.num_raw() != p.len())
        {
            return Err(BotorchError::DimensionMismatch(
                "EM covariance raw parameter count differs".into(),
            ));
        }
        let mut raw = vec![options.noise_raw];
        if options.mean.is_some() {
            raw.push(options.constant);
        }
        raw.extend_from_slice(&options.kernel_raw);
        if let Some((_, values)) = &options.base {
            raw.extend_from_slice(values);
        }
        let cache = Self::build_cache(&train_x, &train_y, &prior, &options, &raw)?;
        Ok(Self {
            train_x,
            train_y,
            datasets,
            prior,
            options,
            raw,
            cache,
        })
    }
    fn kernel_offset(options: &EMModelOptions) -> usize {
        1 + usize::from(options.mean.is_some())
    }
    fn components(
        prior: &EMPrior,
        options: &EMModelOptions,
        raw: &[f64],
    ) -> Result<(EMInterpolation, Option<TreeKernel>)> {
        let start = Self::kernel_offset(options);
        let end = start + options.kernel.num_raw();
        let size = end + options.base.as_ref().map_or(0, |(n, _)| n.num_raw());
        if raw.len() != size || raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "invalid EM raw parameters".into(),
            ));
        }
        let kernel = TreeKernel::build(&options.kernel, &raw[start..end], prior.x.ncols())?;
        let constant = if options.mean.is_some() { raw[1] } else { 0.0 };
        let interpolation = EMInterpolation::new(
            kernel,
            prior.x.clone(),
            constant,
            prior.mean.clone(),
            prior.covariance.clone(),
        )?;
        let base = options
            .base
            .as_ref()
            .map(|(node, _)| TreeKernel::build(node, &raw[end..], prior.x.ncols()))
            .transpose()?;
        Ok((interpolation, base))
    }
    fn prior_at(
        x: &Matrix,
        prior: &EMPrior,
        interpolation: &EMInterpolation,
        base: Option<&TreeKernel>,
        interpolate: bool,
    ) -> Result<(Vector, Matrix)> {
        if x.ncols() != prior.x.ncols() || x.nrows() == 0 || x.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::DimensionMismatch(
                "invalid EM query input dimensions or values".into(),
            ));
        }
        let (mean, mut covariance) = if interpolate {
            let (m, c, _) = interpolation.moments(x)?;
            (m, c)
        } else {
            let indices = index_rows(x, &prior.x)?;
            (
                Vector::from_fn(x.nrows(), |i, _| prior.mean[indices[i]]),
                Matrix::from_fn(x.nrows(), x.nrows(), |i, j| {
                    prior.covariance[(indices[i], indices[j])]
                }),
            )
        };
        if let Some(kernel) = base {
            covariance += kernel.covariance_symmetric(x)?;
        }
        Ok((mean, covariance))
    }
    fn build_cache(
        x: &Matrix,
        y: &Vector,
        prior: &EMPrior,
        options: &EMModelOptions,
        raw: &[f64],
    ) -> Result<Cache> {
        let (interpolation, base) = Self::components(prior, options, raw)?;
        let (mean, mut covariance) =
            Self::prior_at(x, prior, &interpolation, base.as_ref(), options.interpolate)?;
        let noise = options.likelihood.noise_constraint.transform(raw[0]);
        if !noise.is_finite() || noise < 0.0 {
            return Err(BotorchError::InvalidArgument(
                "EM noise variance must be finite and nonnegative".into(),
            ));
        }
        for i in 0..x.nrows() {
            covariance[(i, i)] += noise;
        }
        let factor = psd_safe_cholesky(&covariance)?;
        let alpha = factor.solve_vec(&(y - mean));
        Ok(Cache {
            interpolation,
            base,
            factor,
            alpha,
        })
    }
    /// Raw parameters in likelihood, initial mean, initial kernel, base order.
    pub fn raw_parameters(&self) -> &[f64] {
        &self.raw
    }
    /// Replace all raw parameters atomically after validating the new cache.
    pub fn set_raw_parameters(&mut self, raw: &[f64]) -> Result<()> {
        let cache = Self::build_cache(
            &self.train_x,
            &self.train_y,
            &self.prior,
            &self.options,
            raw,
        )?;
        self.raw = raw.to_vec();
        self.cache = cache;
        Ok(())
    }
    /// Actual Gaussian observation variance.
    pub fn noise(&self) -> f64 {
        self.options
            .likelihood
            .noise_constraint
            .transform(self.raw[0])
    }
    /// Raw state after assigning one constrained observation variance.
    pub fn raw_parameter_vector_for_noise(&self, value: f64) -> Result<Vec<f64>> {
        let constraint = self.options.likelihood.noise_constraint;
        if !value.is_finite()
            || value < constraint.lower_bound()
            || value > constraint.upper_bound()
        {
            return Err(BotorchError::InvalidArgument(
                "noise is outside its constraint".into(),
            ));
        }
        let mut raw = self.raw.clone();
        raw[0] = constraint.inverse_transform(value);
        Ok(raw)
    }
    /// Actual parametric constant (zero for a zero mean).
    pub fn constant(&self) -> f64 {
        if self.options.mean.is_some() {
            self.raw[1]
        } else {
            0.0
        }
    }
    /// Raw state after assigning the parametric constant mean.
    pub fn raw_parameter_vector_for_constant(&self, value: f64) -> Result<Vec<f64>> {
        if self.options.mean.is_none() || !value.is_finite() {
            return Err(BotorchError::InvalidArgument(
                "a finite constant requires ConstantMean".into(),
            ));
        }
        let mut raw = self.raw.clone();
        raw[1] = value;
        Ok(raw)
    }
    /// Freeze or unfreeze the interpolation mean/kernel during fitting.
    pub fn freeze_pretrained(&mut self, freeze: bool) {
        self.options.freeze_pretrained = freeze;
    }
    /// Draw native hyperpriors, preserving frozen interpolation parameters.
    pub fn sample_all_priors<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let mut raw = self.raw.clone();
        if let Some(prior) = self.options.likelihood.noise_prior {
            raw[0] = self
                .options
                .likelihood
                .noise_constraint
                .inverse_transform(prior.sample(rng));
        }
        let offset = Self::kernel_offset(&self.options);
        let end = offset + self.options.kernel.num_raw();
        if !self.options.freeze_pretrained {
            if let Some(prior) = self.options.mean.and_then(|mean| mean.prior) {
                raw[1] = prior.sample(rng);
            }
            crate::kernels::tree::sample_tree_priors(
                &self.options.kernel,
                &mut raw[offset..end],
                rng,
            );
        }
        if let Some((base, _)) = &self.options.base {
            crate::kernels::tree::sample_tree_priors(base, &mut raw[end..], rng);
        }
        raw
    }
    /// Registered public parameter shapes.
    pub fn parameter_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let mut shapes = vec![("likelihood.noise_covar.raw_noise".into(), vec![1])];
        if self.options.mean.is_some() {
            shapes.push(("initial_mean_module.raw_constant".into(), vec![]));
        }
        shapes.extend(self.options.kernel.param_shapes("initial_covar_module."));
        if let Some((base, _)) = &self.options.base {
            shapes.extend(base.param_shapes("_additive_base."));
        }
        shapes
    }
    /// Raw bounds, pinning frozen interpolation parameters at their current values.
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = vec![self.options.likelihood.noise_constraint.raw_bounds()];
        if let Some(mean) = self.options.mean {
            bounds.push(mean.raw_bounds());
        }
        bounds.extend(self.options.kernel.raw_bounds());
        let stop = bounds.len();
        if let Some((base, _)) = &self.options.base {
            bounds.extend(base.raw_bounds());
        }
        if self.options.freeze_pretrained {
            for (i, bound) in bounds.iter_mut().enumerate().take(stop).skip(1) {
                *bound = (self.raw[i], self.raw[i]);
            }
        }
        bounds
    }
    /// Historical observed-data MLL and its analytic raw-parameter gradient.
    pub fn mll_value_and_grad(&self, raw: &[f64]) -> Result<(f64, Vec<f64>)> {
        let (interpolation, base) = Self::components(&self.prior, &self.options, raw)?;
        let kernel = &interpolation.kernel;
        let noise = self.options.likelihood.noise_constraint.transform(raw[0]);
        if !noise.is_finite() || noise < 0.0 {
            return Err(BotorchError::InvalidArgument(
                "invalid EM observation noise".into(),
            ));
        }
        let offset = Self::kernel_offset(&self.options);
        let end = offset + self.options.kernel.num_raw();
        let mut value = 0.0;
        let mut grad = vec![0.0; raw.len()];
        let mut observations = 0;
        for data in &self.datasets {
            let n = data.x.nrows();
            let m = self.prior.x.nrows();
            let (mean, mut covariance) = Self::prior_at(
                &data.x,
                &self.prior,
                &interpolation,
                base.as_ref(),
                self.options.inducing_observations,
            )?;
            for i in 0..n {
                covariance[(i, i)] += noise;
            }
            let factor = psd_safe_cholesky(&covariance)?;
            let residual = &data.y - mean;
            let alpha = factor.solve_vec(&residual);
            value += -0.5
                * (residual.dot(&alpha) + factor.logdet() + n as f64 * std::f64::consts::TAU.ln());
            let covariance_grad = (&alpha * alpha.transpose() - factor.inverse()) * 0.5;
            grad[0] += covariance_grad.trace();
            if self.options.inducing_observations {
                let cross = kernel.covariance(&data.x, &self.prior.x)?;
                let inverse = interpolation.cholesky.inverse();
                let weights = &cross * &inverse;
                let delta = &self.prior.mean - Vector::from_element(m, interpolation.constant);
                let weights_grad = &alpha * delta.transpose() - &covariance_grad * &cross
                    + (&covariance_grad + covariance_grad.transpose())
                        * &weights
                        * &self.prior.covariance;
                let cross_grad =
                    -covariance_grad.transpose() * &weights + &weights_grad * inverse.transpose();
                let inverse_grad = cross.transpose() * weights_grad;
                let inducing_grad = -inverse.transpose() * inverse_grad * inverse.transpose();
                let joint = Matrix::from_fn(n + m, data.x.ncols(), |i, j| {
                    if i < n {
                        data.x[(i, j)]
                    } else {
                        self.prior.x[(i - n, j)]
                    }
                });
                let mut adjoint = Matrix::zeros(n + m, n + m);
                adjoint.view_mut((0, 0), (n, n)).copy_from(&covariance_grad);
                adjoint
                    .view_mut((0, n), (n, m))
                    .copy_from(&(&cross_grad * 0.5));
                adjoint
                    .view_mut((n, 0), (m, n))
                    .copy_from(&(cross_grad.transpose() * 0.5));
                adjoint
                    .view_mut((n, n), (m, m))
                    .copy_from(&((&inducing_grad + inducing_grad.transpose()) * 0.5));
                let kg = kernel.contract(&kernel.prepare(&joint)?, &adjoint);
                for (dest, g) in grad[offset..end].iter_mut().zip(kg) {
                    *dest += g;
                }
                if self.options.mean.is_some() {
                    grad[1] += alpha.sum() - alpha.dot(&(weights * Vector::from_element(m, 1.0)));
                }
            }
            if let Some(base) = &base {
                let bg = base.contract(&base.prepare(&data.x)?, &covariance_grad);
                for (dest, g) in grad[end..].iter_mut().zip(bg) {
                    *dest += g;
                }
            }
            observations += n;
        }
        let (lp, prior_grad) = kernel.log_prior_and_grad()?;
        value += lp;
        let transforms = self.options.kernel.transform_derivatives(&raw[offset..end]);
        for ((dest, prior), transform) in
            grad[offset..end].iter_mut().zip(prior_grad).zip(transforms)
        {
            *dest = (*dest + prior) * transform;
        }
        if let (Some(base), Some((node, _))) = (&base, &self.options.base) {
            let (lp, prior_grad) = base.log_prior_and_grad()?;
            value += lp;
            let transforms = node.transform_derivatives(&raw[end..]);
            for ((dest, prior), transform) in grad[end..].iter_mut().zip(prior_grad).zip(transforms)
            {
                *dest = (*dest + prior) * transform;
            }
        }
        if let Some(mean) = self.options.mean {
            if let Some(prior) = mean.prior {
                value += prior.log_prob(raw[1]);
                grad[1] += prior.grad_log_prob(raw[1]);
            }
        }
        if let Some(prior) = self.options.likelihood.noise_prior {
            value += prior.log_prob(noise);
            grad[0] += prior.grad_log_prob(noise);
        }
        grad[0] *= self
            .options
            .likelihood
            .noise_constraint
            .transform_derivative(raw[0]);
        let scale = observations as f64;
        for g in &mut grad {
            *g /= scale;
        }
        Ok((value / scale, grad))
    }
    fn combined(&self, x: &Matrix) -> Result<(Vector, Matrix, Matrix)> {
        let n = self.train_x.nrows();
        let q = x.nrows();
        let joint = Matrix::from_fn(n + q, self.dim(), |i, j| {
            if i < n {
                self.train_x[(i, j)]
            } else {
                x[(i - n, j)]
            }
        });
        let (mean, covariance) = Self::prior_at(
            &joint,
            &self.prior,
            &self.cache.interpolation,
            self.cache.base.as_ref(),
            self.options.interpolate,
        )?;
        let cross = covariance.view((n, 0), (q, n)).into_owned();
        let posterior_mean = mean.rows(n, q) + &cross * &self.cache.alpha;
        let posterior_covariance =
            covariance.view((n, n), (q, q)) - &cross * self.cache.factor.solve(&cross.transpose());
        Ok((posterior_mean, posterior_covariance, joint))
    }
}

impl Model for EMEmpiricalGP {
    fn dim(&self) -> usize {
        self.prior.x.ncols()
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn posterior(&self, x: &Matrix, noise: bool) -> Result<Posterior> {
        if x.ncols() != self.dim() || x.nrows() == 0 || x.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::DimensionMismatch(
                "invalid EM posterior inputs".into(),
            ));
        }
        let (mean, mut covariance, _) = self.combined(x)?;
        if noise {
            for i in 0..x.nrows() {
                covariance[(i, i)] += self.noise();
            }
        }
        Ok(Posterior::single_output(mean, covariance))
    }
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.moments_with_grad(x, true)
    }
}

impl EMEmpiricalGP {
    fn moments_with_grad(&self, x: &Matrix, conditioned: bool) -> Result<JointPosteriorGrad> {
        let posterior = if conditioned {
            self.posterior(x, false)?
        } else {
            let (mean, covariance) = Self::prior_at(
                x,
                &self.prior,
                &self.cache.interpolation,
                self.cache.base.as_ref(),
                self.options.interpolate,
            )?;
            Posterior::single_output(mean, covariance)
        };
        let joint = if conditioned {
            self.combined(x)?.2
        } else {
            x.clone()
        };
        let n = if conditioned { self.train_x.nrows() } else { 0 };
        let q = x.nrows();
        let d = self.dim();
        let count = n + q;
        let mut dmean = vec![Vector::zeros(q); q * d];
        let mut dcov = vec![Matrix::zeros(q, q); q * d];
        let interpolation = &self.cache.interpolation;
        let kernel = &interpolation.kernel;
        let prep = kernel.prepare(&joint)?;
        let inducing_prep = kernel.prepare(&self.prior.x)?;
        let cross = kernel.cross(&prep, &inducing_prep);
        let weights = interpolation.cholesky.solve(&cross.transpose()).transpose();
        let delta =
            &self.prior.mean - Vector::from_element(self.prior.mean.len(), interpolation.constant);
        let base_prep = self
            .cache
            .base
            .as_ref()
            .map(|k| k.prepare(&joint))
            .transpose()?;
        let (_, covariance) = Self::prior_at(
            &joint,
            &self.prior,
            interpolation,
            self.cache.base.as_ref(),
            self.options.interpolate,
        )?;
        let solved = if conditioned {
            self.cache
                .factor
                .solve(&covariance.view((0, n), (n, q)).into_owned())
        } else {
            Matrix::zeros(0, q)
        };
        for r in 0..q {
            let point = n + r;
            let (_, direct_grad) = kernel.cross_with_grad(&prep, point, &prep);
            let (_, cross_grad) = kernel.cross_with_grad(&prep, point, &inducing_prep);
            let weight_grad = interpolation.cholesky.solve(&cross_grad);
            let base_grad = self
                .cache
                .base
                .as_ref()
                .zip(base_prep.as_ref())
                .map(|(k, p)| k.cross_with_grad(p, point, p).1);
            for j in 0..d {
                let mut prior_derivative = Matrix::zeros(count, count);
                if self.options.interpolate {
                    dmean[r * d + j][r] = weight_grad.column(j).dot(&delta);
                    for a in 0..count {
                        let gradient = direct_grad[(a, j)]
                            - weight_grad.column(j).dot(&cross.row(a).transpose())
                            + weight_grad
                                .column(j)
                                .dot(&(&self.prior.covariance * weights.row(a).transpose()));
                        prior_derivative[(point, a)] += gradient;
                        prior_derivative[(a, point)] += gradient;
                    }
                }
                if let Some(base_grad) = &base_grad {
                    for a in 0..count {
                        prior_derivative[(point, a)] += base_grad[(a, j)];
                        prior_derivative[(a, point)] += base_grad[(a, j)];
                    }
                }
                let cross_derivative = prior_derivative.view((n, 0), (q, n)).into_owned();
                if conditioned {
                    dmean[r * d + j] += &cross_derivative * &self.cache.alpha;
                }
                dcov[r * d + j] = prior_derivative.view((n, n), (q, q))
                    - &cross_derivative * &solved
                    - solved.transpose() * cross_derivative.transpose();
            }
        }
        Ok(JointPosteriorGrad {
            mean: posterior.mean,
            covariance: posterior.covariance,
            dmean,
            dcov,
            num_outputs: 1,
        })
    }
}

/// A prior distribution view, sharing an immutable native model snapshot.
#[derive(Clone, Debug)]
pub struct EMGPPrior(pub std::sync::Arc<EMEmpiricalGP>);

impl Model for EMGPPrior {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn posterior(&self, x: &Matrix, noise: bool) -> Result<Posterior> {
        let (mean, mut covariance) = EMEmpiricalGP::prior_at(
            x,
            &self.0.prior,
            &self.0.cache.interpolation,
            self.0.cache.base.as_ref(),
            self.0.options.interpolate,
        )?;
        if noise {
            for i in 0..x.nrows() {
                covariance[(i, i)] += self.0.noise();
            }
        }
        Ok(Posterior::single_output(mean, covariance))
    }
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.0.moments_with_grad(x, false)
    }
}

impl MllModel for EMEmpiricalGP {
    fn raw_parameter_vector(&self) -> Vec<f64> {
        self.raw.clone()
    }
    fn set_raw_parameter_vector(&mut self, values: &[f64]) -> Result<()> {
        self.set_raw_parameters(values)
    }
    fn raw_bounds(&self) -> Vec<(f64, f64)> {
        self.raw_parameter_bounds()
    }
    fn mll_values_and_gradient(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        self.mll_value_and_grad(x).map(|(v, g)| (vec![v], g))
    }
}

impl Fittable for EMEmpiricalGP {
    type Output = FitResult;
    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        self.fit_mll_with_mask(options, None, None)
    }
}

impl EMEmpiricalGP {
    /// Fit selected raw entries, preserving excluded parameters exactly.
    pub fn fit_mll_with_mask(
        &mut self,
        options: FitOptions,
        mask: Option<&[bool]>,
        bounds: Option<&[(f64, f64)]>,
    ) -> Result<FitResult> {
        let mut limits = bounds.map_or_else(|| self.raw_parameter_bounds(), <[_]>::to_vec);
        if limits.len() != self.raw.len() || mask.is_some_and(|m| m.len() != self.raw.len()) {
            return Err(BotorchError::DimensionMismatch(
                "EM parameter mask/bounds length differs".into(),
            ));
        }
        if let Some(mask) = mask {
            for (i, free) in mask.iter().enumerate() {
                if !free {
                    limits[i] = (self.raw[i], self.raw[i]);
                }
            }
        }
        let (result, raw) = minimize_with_retries(
            &self.raw,
            &limits,
            options,
            |raw| self.mll_value_and_grad(raw),
            |rng| self.sample_all_priors(rng),
        )?;
        self.set_raw_parameters(&raw)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{GreaterThan, ParameterTransform};
    use crate::kernels::KernelKind;

    fn model(inducing: bool, base: bool) -> EMEmpiricalGP {
        let z = Matrix::from_row_slice(3, 2, &[0.0, 0.0, 0.4, 0.7, 1.0, 0.3]);
        let prior = EMPrior {
            x: z.clone(),
            mean: Vector::from_vec(vec![0.3, 0.5, -0.2]),
            covariance: Matrix::from_row_slice(
                3,
                3,
                &[0.9, 0.2, -0.1, 0.2, 0.6, 0.1, -0.1, 0.1, 0.7],
            ),
        };
        let x = if inducing {
            Matrix::from_row_slice(3, 2, &[0.1, 0.2, 0.55, 0.6, 0.8, 0.1])
        } else {
            z.clone()
        };
        let data = vec![
            EMDataset {
                x: x.rows(0, 2).into_owned(),
                y: Vector::from_vec(vec![0.2, -0.3]),
            },
            EMDataset {
                x: x.clone(),
                y: Vector::from_vec(vec![0.8, 0.4, 0.7]),
            },
        ];
        let options = EMModelOptions {
            kernel: KernelNode::scale(KernelNode::stationary(KernelKind::Rbf, Some(2)).unwrap()),
            kernel_raw: vec![0.1, -0.2, 0.4],
            mean: Some(ConstantMeanSpec::default()),
            constant: 0.15,
            likelihood: GaussianLikelihoodSpec {
                noise_prior: None,
                noise_constraint: GreaterThan::new(1e-4, Some(ParameterTransform::Softplus), None)
                    .into(),
            },
            noise_raw: -2.0,
            base: base.then(|| {
                (
                    KernelNode::stationary(KernelKind::Matern52, Some(2)).unwrap(),
                    vec![0.2, -0.1],
                )
            }),
            interpolate: true,
            inducing_observations: inducing,
            freeze_pretrained: false,
        };
        EMEmpiricalGP::new(
            x,
            Vector::from_vec(vec![0.5, 0.2, -0.1]),
            data,
            prior,
            options,
        )
        .unwrap()
    }
    #[test]
    fn historical_mll_raw_gradients_match_finite_differences() {
        for inducing in [false, true] {
            for base in [false, true] {
                let model = model(inducing, base);
                let raw = model.raw_parameters();
                let (_, gradient) = model.mll_value_and_grad(raw).unwrap();
                for i in 0..raw.len() {
                    let mut plus = raw.to_vec();
                    let mut minus = raw.to_vec();
                    let h = 1e-6;
                    plus[i] += h;
                    minus[i] -= h;
                    let numerical = (model.mll_value_and_grad(&plus).unwrap().0
                        - model.mll_value_and_grad(&minus).unwrap().0)
                        / (2.0 * h);
                    assert!(
                        (numerical - gradient[i]).abs() < 2e-6,
                        "inducing={inducing} base={base} raw{i}: {numerical} != {}",
                        gradient[i]
                    );
                }
            }
        }
    }
    #[test]
    fn posterior_input_gradients_and_noise_are_exact() {
        for base in [false, true] {
            let model = model(true, base);
            let x = Matrix::from_row_slice(2, 2, &[0.22, 0.17, 0.72, 0.52]);
            let g = model.joint_posterior_with_grad(&x).unwrap();
            for j in 0..4 {
                let mut plus = x.clone();
                let mut minus = x.clone();
                let h = 1e-6;
                plus[(j / 2, j % 2)] += h;
                minus[(j / 2, j % 2)] -= h;
                let p = model.posterior(&plus, false).unwrap();
                let m = model.posterior(&minus, false).unwrap();
                assert!(((p.mean - m.mean) / (2.0 * h) - &g.dmean[j]).amax() < 2e-6);
                assert!(((p.covariance - m.covariance) / (2.0 * h) - &g.dcov[j]).amax() < 2e-6);
            }
            let noisy = model.posterior(&x, true).unwrap();
            assert!(
                (noisy.covariance - &g.covariance - Matrix::identity(2, 2) * model.noise()).amax()
                    < 1e-12
            );
        }
    }
}
