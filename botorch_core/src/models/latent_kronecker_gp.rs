//! Exact GP inference on Cartesian feature/time grids with missing observations.
//!
//! Complete grids reuse HigherOrderGP's Kronecker eigensolves. Missing entries
//! use exact dense masked solves. Sampling uses the correct conditional prior
//! covariance K_test - A^T A, followed by Matheron's observation update.

use std::ops::Range;
use std::sync::OnceLock;

use rand::rngs::StdRng;

use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::kernels::tree::{sample_tree_priors, KernelNode};
use crate::kernels::TreeKernel;
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::mll::ExactMarginalLogLikelihood;
use crate::models::approximate_gp::VariationalMean;
use crate::models::gp::{GaussianLikelihoodSpec, SingleTaskGP, SingleTaskGPOptions};
use crate::models::higher_order_gp::{HigherOrderGP, HigherOrderGPOptions, Tensor};
use crate::models::Model;
use crate::optim::fit::MllModel;
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::transforms::input::{ChainedInputTransform, TransformMode};
use crate::transforms::OutcomeTransformChoice;
use crate::{Matrix, Vector};

/// Modules and fitted transforms of a latent Kronecker GP.
#[derive(Clone, Debug)]
pub struct LatentKroneckerGPOptions {
    /// Covariance kernel on transformed feature inputs.
    pub covar_x: KernelNode,
    /// Covariance kernel on scalar time inputs.
    pub covar_t: KernelNode,
    /// Mean factor on features (constant or zero).
    pub mean_x: VariationalMean,
    /// Mean factor on time (constant or zero); the joint mean is their product.
    pub mean_t: VariationalMean,
    /// Gaussian observation noise.
    pub noise: GaussianLikelihoodSpec,
    /// Fitted static feature transform; training X is already transformed.
    pub input_transform: Option<ChainedInputTransform>,
    /// Fitted scalar outcome mean and standard deviation.
    pub outcome: Option<(f64, f64)>,
}

#[derive(Clone, Debug)]
enum Backend {
    Full(Box<HigherOrderGP>),
    Masked(Box<SingleTaskGP>),
}

/// Native latent Kronecker GP with unbatched training inputs.
#[derive(Clone, Debug)]
pub struct LatentKroneckerGP {
    x: Matrix,
    t: Matrix,
    y: Matrix,
    observed: Vec<usize>,
    dim: usize,
    options: LatentKroneckerGPOptions,
    mean_indices: [Option<usize>; 2],
    kernel_ranges: [Range<usize>; 2],
    raw: Vec<f64>,
    cache: OnceLock<std::result::Result<Backend, BotorchError>>,
    masked_cholesky: OnceLock<std::result::Result<CholeskyFactor, BotorchError>>,
}

fn row_values(matrix: &Matrix) -> Vec<f64> {
    (0..matrix.nrows())
        .flat_map(|i| matrix.row(i).iter().copied().collect::<Vec<_>>())
        .collect()
}

/// Feature/time Cartesian product, in feature-major order.
pub fn cartesian_inputs(x: &Matrix, t: &Matrix) -> Matrix {
    Matrix::from_fn(x.nrows() * t.nrows(), x.ncols() + 1, |i, j| {
        if j < x.ncols() {
            x[(i / t.nrows(), j)]
        } else {
            t[(i % t.nrows(), 0)]
        }
    })
}

impl LatentKroneckerGP {
    /// Build from transformed X, scalar time inputs, and standardized observations.
    /// Nonfinite Y entries are missing; every other shape/value must be valid.
    pub fn new(
        x: Matrix,
        t: Matrix,
        y: Matrix,
        dim: usize,
        raw: Option<Vec<f64>>,
        options: LatentKroneckerGPOptions,
    ) -> Result<Self> {
        if x.nrows() == 0
            || x.ncols() == 0
            || t.nrows() == 0
            || t.ncols() != 1
            || y.shape() != (x.nrows(), t.nrows())
        {
            return Err(BotorchError::DimensionMismatch(
                "expected X n x d, T t x 1, and Y n x t with nonempty dimensions".into(),
            ));
        }
        if x.iter().chain(t.iter()).any(|v| !v.is_finite()) {
            return Err(BotorchError::InputData(
                "feature and time inputs must be finite".into(),
            ));
        }
        let observed: Vec<usize> = row_values(&y)
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.is_finite().then_some(i))
            .collect();
        if observed.is_empty() {
            return Err(BotorchError::InputData(
                "at least one finite training observation is required".into(),
            ));
        }
        if options
            .outcome
            .is_some_and(|(mean, std)| !mean.is_finite() || !std.is_finite() || std <= 0.0)
        {
            return Err(BotorchError::InvalidArgument(
                "outcome mean must be finite and standard deviation positive".into(),
            ));
        }
        options.covar_x.validate(x.ncols())?;
        options.covar_t.validate(1)?;
        if let Some(chain) = &options.input_transform {
            if chain.points_per_input() != 1 || chain.num_params() != 0 {
                return Err(BotorchError::Unsupported(
                    "LatentKroneckerGP requires a static one-to-one input transform".into(),
                ));
            }
            chain.check_mode_dim(dim, x.ncols(), TransformMode::Eval)?;
        } else if dim != x.ncols() {
            return Err(BotorchError::DimensionMismatch(
                "raw and transformed feature dimensions differ".into(),
            ));
        }
        let mut initial = vec![options.noise.initial_raw_noise()];
        let mut mean_indices = [None, None];
        for (i, mean) in [options.mean_x, options.mean_t].iter().enumerate() {
            if let VariationalMean::Constant(spec) = mean {
                mean_indices[i] = Some(initial.len());
                initial.push(spec.initial_raw());
            }
        }
        let start_x = initial.len();
        initial.extend(options.covar_x.initial_raw());
        let start_t = initial.len();
        initial.extend(options.covar_t.initial_raw());
        let kernel_ranges = [start_x..start_t, start_t..initial.len()];
        let raw = raw.unwrap_or(initial);
        if raw.len() != kernel_ranges[1].end || raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "raw parameters have invalid length or nonfinite values".into(),
            ));
        }
        Ok(Self {
            x,
            t,
            y,
            observed,
            dim,
            options,
            mean_indices,
            kernel_ranges,
            raw,
            cache: OnceLock::new(),
            masked_cholesky: OnceLock::new(),
        })
    }

    /// Transformed training features.
    pub fn train_inputs(&self) -> &Matrix {
        &self.x
    }
    /// Training time inputs.
    pub fn train_t(&self) -> &Matrix {
        &self.t
    }
    /// Standardized finite training observations in feature-major order.
    pub fn train_targets(&self) -> Vector {
        let y = row_values(&self.y);
        Vector::from_iterator(self.observed.len(), self.observed.iter().map(|&i| y[i]))
    }
    /// Flattened indices of observed entries.
    pub fn observed_indices(&self) -> &[usize] {
        &self.observed
    }
    /// Whether the training grid is complete.
    pub fn is_complete(&self) -> bool {
        self.observed.len() == self.y.len()
    }
    /// Module specifications.
    pub fn options(&self) -> &LatentKroneckerGPOptions {
        &self.options
    }
    /// Flat raw parameters, in MLL registration order.
    pub fn raw_parameters(&self) -> &[f64] {
        &self.raw
    }
    /// Update raw parameters and clear prediction caches.
    pub fn set_raw_parameters(&mut self, raw: &[f64]) -> Result<()> {
        if raw.len() != self.raw.len() || raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "raw parameters have invalid length or nonfinite values".into(),
            ));
        }
        self.raw = raw.to_vec();
        self.cache = OnceLock::new();
        self.masked_cholesky = OnceLock::new();
        Ok(())
    }
    /// Names and sizes of all parameters.
    pub fn parameter_names(&self) -> Vec<(String, usize)> {
        let mut names = vec![("likelihood.noise_covar.raw_noise".into(), 1)];
        for (i, name) in ["X", "T"].iter().enumerate() {
            if self.mean_indices[i].is_some() {
                names.push((format!("model.mean_module_{name}.raw_constant"), 1));
            }
        }
        for (node, name) in [(&self.options.covar_x, "X"), (&self.options.covar_t, "T")] {
            node.visit_params(&format!("model.covar_module_{name}."), &mut |name, spec| {
                names.push((name, spec.numel()))
            });
        }
        names
    }
    /// Optimizer bounds of all raw parameters.
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = vec![self.options.noise.noise_constraint.raw_bounds()];
        for mean in [self.options.mean_x, self.options.mean_t] {
            if let VariationalMean::Constant(spec) = mean {
                bounds.push(spec.raw_bounds());
            }
        }
        bounds.extend(self.options.covar_x.raw_bounds());
        bounds.extend(self.options.covar_t.raw_bounds());
        bounds
    }
    fn mean_parts(&self, raw: &[f64]) -> [f64; 2] {
        let mut values = [0.0, 0.0];
        for (i, mean) in [self.options.mean_x, self.options.mean_t]
            .iter()
            .enumerate()
        {
            if let VariationalMean::Constant(spec) = mean {
                let value = raw[self.mean_indices[i].unwrap()];
                values[i] = spec.constraint.map_or(value, |c| c.transform(value));
            }
        }
        values
    }
    fn mean(&self, raw: &[f64]) -> f64 {
        self.mean_parts(raw).iter().product()
    }
    /// Gaussian likelihood noise variance in standardized units.
    pub fn noise(&self) -> f64 {
        self.options.noise.noise_constraint.transform(self.raw[0])
    }
    fn kernels(&self, raw: &[f64]) -> Result<(TreeKernel, TreeKernel)> {
        Ok((
            TreeKernel::build(
                &self.options.covar_x,
                &raw[self.kernel_ranges[0].clone()],
                self.x.ncols(),
            )?,
            TreeKernel::build(
                &self.options.covar_t,
                &raw[self.kernel_ranges[1].clone()],
                1,
            )?,
        ))
    }
    fn dense(&self, raw: &[f64]) -> Result<SingleTaskGP> {
        let all = cartesian_inputs(&self.x, &self.t);
        let inputs = Matrix::from_fn(self.observed.len(), all.ncols(), |i, j| {
            all[(self.observed[i], j)]
        });
        let targets = self.train_targets().add_scalar(-self.mean(raw));
        let kx = self
            .options
            .covar_x
            .clone()
            .with_parent_active_dims(&(0..self.x.ncols()).collect::<Vec<_>>())?;
        let kt = self
            .options
            .covar_t
            .clone()
            .with_parent_active_dims(&[self.x.ncols()])?;
        let mut gp = SingleTaskGP::new(
            inputs,
            Matrix::from_column_slice(targets.len(), 1, targets.as_slice()),
            SingleTaskGPOptions {
                covar_tree: Some(KernelNode::product(vec![kx, kt])?),
                likelihood: Some(self.options.noise),
                outcome_transform: OutcomeTransformChoice::None,
                ..Default::default()
            },
        )?;
        let mut parameters = vec![raw[0], 0.0];
        parameters.extend_from_slice(&raw[self.kernel_ranges[0].start..]);
        gp.set_raw_parameters(&parameters)?;
        Ok(gp)
    }
    fn backend_at(&self, raw: &[f64]) -> Result<Backend> {
        if !self.is_complete() {
            return self
                .dense(raw)
                .map(|model| Backend::Masked(Box::new(model)));
        }
        let mut parameters = vec![raw[0]];
        parameters.extend_from_slice(&raw[self.kernel_ranges[0].start..]);
        parameters.extend(row_values(&self.t));
        let model = HigherOrderGP::new(
            self.x.clone(),
            self.y.add_scalar(-self.mean(raw)),
            vec![self.t.nrows()],
            vec![1],
            self.x.ncols(),
            Some(parameters),
            HigherOrderGPOptions {
                covar_modules: vec![self.options.covar_x.clone(), self.options.covar_t.clone()],
                noise: self.options.noise,
                latent_priors: vec![],
                learn_latent_pars: false,
                input_transform: None,
                outcome: None,
            },
        )?;
        Ok(Backend::Full(Box::new(model)))
    }
    fn backend(&self) -> Result<&Backend> {
        self.cache
            .get_or_init(|| self.backend_at(&self.raw))
            .as_ref()
            .map_err(Clone::clone)
    }
    fn solve_observed(&self, rhs: &Vector) -> Result<Vector> {
        match self.backend()? {
            Backend::Full(model) => Ok(Vector::from_vec(row_values(&model.solve_training(
                &Matrix::from_row_slice(self.x.nrows(), self.t.nrows(), rhs.as_slice()),
            )?))),
            Backend::Masked(_) => {
                let factor = self
                    .masked_cholesky
                    .get_or_init(|| {
                        let (kx, kt) = self.kernels(&self.raw)?;
                        let full = kx
                            .covariance_symmetric(&self.x)?
                            .kronecker(&kt.covariance_symmetric(&self.t)?);
                        let covariance =
                            Matrix::from_fn(self.observed.len(), self.observed.len(), |i, j| {
                                full[(self.observed[i], self.observed[j])]
                                    + if i == j { self.noise() } else { 0.0 }
                            });
                        psd_safe_cholesky(&covariance)
                    })
                    .as_ref()
                    .map_err(Clone::clone)?;
                Ok(factor.solve_vec(rhs))
            }
        }
    }
    /// MLL and exact gradient at arbitrary raw parameters.
    pub fn mll_value_and_grad(&self, raw: &[f64], with_grad: bool) -> Result<(f64, Vec<f64>)> {
        if raw.len() != self.raw.len() || raw.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InvalidArgument(
                "invalid raw parameters".into(),
            ));
        }
        let owned;
        let backend = if raw == self.raw {
            self.backend()?
        } else {
            owned = self.backend_at(raw)?;
            &owned
        };
        let mut gradient = vec![0.0; raw.len()];
        let (mut value, mean_gradient) = match backend {
            Backend::Full(model) => {
                let (value, grad) = model.mll_value_and_grad(model.raw_parameters(), with_grad)?;
                gradient[0] = grad[0];
                gradient[self.kernel_ranges[0].start..]
                    .copy_from_slice(&grad[1..1 + raw.len() - self.kernel_ranges[0].start]);
                let gm = if with_grad {
                    model
                        .solve_training(&self.y.add_scalar(-self.mean(raw)))?
                        .sum()
                        / self.observed.len() as f64
                } else {
                    0.0
                };
                (value, gm)
            }
            Backend::Masked(model) => {
                let mll = ExactMarginalLogLikelihood::new(model);
                if with_grad {
                    let (value, grad) = mll.value_and_grad(model.raw_parameters().as_slice())?;
                    gradient[0] = grad[0];
                    gradient[self.kernel_ranges[0].start..].copy_from_slice(&grad.as_slice()[2..]);
                    (value, grad[1])
                } else {
                    (mll.value()?, 0.0)
                }
            }
        };
        let means = self.mean_parts(raw);
        for (i, mean) in [self.options.mean_x, self.options.mean_t]
            .iter()
            .enumerate()
        {
            if let VariationalMean::Constant(spec) = mean {
                let index = self.mean_indices[i].unwrap();
                let derivative = spec
                    .constraint
                    .map_or(1.0, |c| c.transform_derivative(raw[index]));
                gradient[index] = mean_gradient * means[1 - i] * derivative;
                if let Some(prior) = spec.prior {
                    value += prior.log_prob(means[i]) / self.observed.len() as f64;
                    gradient[index] +=
                        prior.grad_log_prob(means[i]) * derivative / self.observed.len() as f64;
                }
            }
        }
        Ok((value, gradient))
    }
    fn transformed(&self, x: &Matrix) -> Result<Matrix> {
        if x.nrows() == 0 || x.ncols() != self.dim {
            return Err(BotorchError::DimensionMismatch(
                "expected nonempty feature inputs of the model dimension".into(),
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::InputData(
                "query feature inputs must be finite".into(),
            ));
        }
        self.options.input_transform.as_ref().map_or_else(
            || Ok(x.clone()),
            |c| c.transform_mode(x, TransformMode::Eval),
        )
    }
    fn validate_t(t: &Matrix) -> Result<()> {
        if t.nrows() == 0 || t.ncols() != 1 || t.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::DimensionMismatch(
                "time queries must be a nonempty finite t x 1 matrix".into(),
            ));
        }
        Ok(())
    }
    fn affine(&self) -> (f64, f64) {
        self.options.outcome.unwrap_or((0.0, 1.0))
    }
    /// Exact posterior on X x T, flattened in feature-major order.
    pub fn posterior_grid(&self, x: &Matrix, t: &Matrix) -> Result<Posterior> {
        Self::validate_t(t)?;
        let xt = self.transformed(x)?;
        let mut posterior = match self.backend()? {
            Backend::Full(model) if t == &self.t => {
                let source = model.posterior(&xt, false)?;
                let q = x.nrows();
                let tasks = t.nrows();
                let index = |i: usize| (i % tasks) * q + i / tasks;
                Posterior::single_output(
                    Vector::from_fn(q * tasks, |i, _| source.mean[index(i)]),
                    Matrix::from_fn(q * tasks, q * tasks, |i, j| {
                        source.covariance[(index(i), index(j))]
                    }),
                )
            }
            Backend::Masked(model) => model.posterior(&cartesian_inputs(&xt, t), false)?,
            Backend::Full(_) => self
                .dense(&self.raw)?
                .posterior(&cartesian_inputs(&xt, t), false)?,
        };
        let (offset, scale) = self.affine();
        posterior.mean = posterior
            .mean
            .map(|v| offset + scale * (v + self.mean(&self.raw)));
        posterior.covariance *= scale * scale;
        Ok(posterior)
    }
    /// Analytic derivatives with respect to raw feature inputs on an X x T grid.
    pub fn posterior_grid_with_grad(&self, x: &Matrix, t: &Matrix) -> Result<JointPosteriorGrad> {
        Self::validate_t(t)?;
        let xt = self.transformed(x)?;
        let cartesian = cartesian_inputs(&xt, t);
        let dense;
        let model = match self.backend()? {
            Backend::Masked(model) => model.as_ref(),
            Backend::Full(_) => {
                dense = self.dense(&self.raw)?;
                &dense
            }
        };
        let source = model.joint_posterior_with_grad(&cartesian)?;
        let q = x.nrows();
        let tasks = t.nrows();
        let dt = xt.ncols();
        let d = x.ncols();
        let jacobians = self
            .options
            .input_transform
            .as_ref()
            .map(|c| {
                let (_, tape) = c.transform_with_tape_mode(x, TransformMode::Eval)?;
                tape.jacobians_through(c)
            })
            .transpose()?;
        let mut dmean = vec![Vector::zeros(q * tasks); q * d];
        let mut dcov = vec![Matrix::zeros(q * tasks, q * tasks); q * d];
        let (_, scale) = self.affine();
        for point in 0..q {
            for input in 0..d {
                for transformed in 0..dt {
                    let factor = jacobians
                        .as_ref()
                        .map_or(if input == transformed { 1.0 } else { 0.0 }, |j| {
                            j[point][(transformed, input)]
                        });
                    for time in 0..tasks {
                        let index = (point * tasks + time) * (dt + 1) + transformed;
                        dmean[point * d + input] += &source.dmean[index] * (scale * factor);
                        dcov[point * d + input] += &source.dcov[index] * (scale * scale * factor);
                    }
                }
            }
        }
        let posterior = self.posterior_grid(x, t)?;
        Ok(JointPosteriorGrad {
            mean: posterior.mean,
            covariance: posterior.covariance,
            dmean,
            dcov,
            num_outputs: 1,
        })
    }
    /// Correct Matheron samples from [full training prior, observed noise, test residual] draws.
    pub fn samples(&self, x: &Matrix, t: &Matrix, base: &Matrix) -> Result<Matrix> {
        Self::validate_t(t)?;
        let xt = self.transformed(x)?;
        let full = self.y.len();
        let observed = self.observed.len();
        let test = x.nrows() * t.nrows();
        if base.ncols() != full + observed + test {
            return Err(BotorchError::DimensionMismatch(
                "wrong base sample width".into(),
            ));
        }
        let (kx, kt) = self.kernels(&self.raw)?;
        let root_x = psd_safe_cholesky(&kx.covariance_symmetric(&self.x)?)?;
        let root_t = psd_safe_cholesky(&kt.covariance_symmetric(&self.t)?)?;
        let cross_x = kx.covariance(&self.x, &xt)?;
        let cross_t = kt.covariance(&self.t, t)?;
        let mut ax = cross_x.clone();
        root_x.solve_lower_mut(&mut ax);
        let mut at = cross_t.clone();
        root_t.solve_lower_mut(&mut at);
        let a = ax.kronecker(&at);
        let covariance_test = kx
            .covariance_symmetric(&xt)?
            .kronecker(&kt.covariance_symmetric(t)?);
        let residual_root = psd_safe_cholesky(&(covariance_test - a.transpose() * &a))?;
        let cross = cross_x.kronecker(&cross_t);
        let observed_cross = Matrix::from_fn(observed, test, |i, j| cross[(self.observed[i], j)]);
        let target = self.train_targets().add_scalar(-self.mean(&self.raw));
        let (offset, scale) = self.affine();
        let mut output = Matrix::zeros(base.nrows(), test);
        for sample in 0..base.nrows() {
            let z = Vector::from_iterator(full, (0..full).map(|i| base[(sample, i)]));
            let prior_train =
                Tensor::new(vec![self.x.nrows(), self.t.nrows()], z.as_slice().to_vec())
                    .mode_mul(0, root_x.l())
                    .mode_mul(1, root_t.l());
            let rhs = Vector::from_fn(observed, |i, _| {
                target[i]
                    - prior_train.data[self.observed[i]]
                    - self.noise().sqrt() * base[(sample, full + i)]
            });
            let w =
                Vector::from_iterator(test, (0..test).map(|i| base[(sample, full + observed + i)]));
            let values = a.transpose() * z
                + residual_root.l() * w
                + observed_cross.transpose() * self.solve_observed(&rhs)?;
            for i in 0..test {
                output[(sample, i)] = offset + scale * (self.mean(&self.raw) + values[i]);
            }
        }
        Ok(output)
    }
    /// Sample each registered scalar prior for fitting retries.
    pub fn sample_all_priors(&self, raw: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let mut values = raw.to_vec();
        if let Some(prior) = self.options.noise.noise_prior {
            values[0] = self
                .options
                .noise
                .noise_constraint
                .inverse_transform(prior.sample(rng));
        }
        for (i, mean) in [self.options.mean_x, self.options.mean_t]
            .iter()
            .enumerate()
        {
            if let VariationalMean::Constant(spec) = mean {
                if let Some(prior) = spec.prior {
                    let value = prior.sample(rng);
                    values[self.mean_indices[i].unwrap()] = spec
                        .constraint
                        .map_or(value, |c| c.inverse_transform(value));
                }
            }
        }
        sample_tree_priors(
            &self.options.covar_x,
            &mut values[self.kernel_ranges[0].clone()],
            rng,
        );
        sample_tree_priors(
            &self.options.covar_t,
            &mut values[self.kernel_ranges[1].clone()],
            rng,
        );
        values
    }
}

impl Model for LatentKroneckerGP {
    fn dim(&self) -> usize {
        self.dim
    }
    fn num_outputs(&self) -> usize {
        self.t.nrows()
    }
    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let grid = self.posterior_grid(x, &self.t)?;
        let q = x.nrows();
        let tasks = self.t.nrows();
        let index = |i: usize| (i % q) * tasks + i / q;
        let mut covariance = Matrix::from_fn(q * tasks, q * tasks, |i, j| {
            grid.covariance[(index(i), index(j))]
        });
        if observation_noise {
            let scale = self.affine().1;
            for i in 0..covariance.nrows() {
                covariance[(i, i)] += self.noise() * scale * scale;
            }
        }
        Ok(Posterior::multi_output(
            Vector::from_fn(q * tasks, |i, _| grid.mean[index(i)]),
            covariance,
            tasks,
        ))
    }
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let grid = self.posterior_grid_with_grad(x, &self.t)?;
        let q = x.nrows();
        let tasks = self.t.nrows();
        let index = |i: usize| (i % q) * tasks + i / q;
        let vector = |v: &Vector| Vector::from_fn(q * tasks, |i, _| v[index(i)]);
        let matrix =
            |v: &Matrix| Matrix::from_fn(q * tasks, q * tasks, |i, j| v[(index(i), index(j))]);
        Ok(JointPosteriorGrad {
            mean: vector(&grid.mean),
            covariance: matrix(&grid.covariance),
            dmean: grid.dmean.iter().map(vector).collect(),
            dcov: grid.dcov.iter().map(matrix).collect(),
            num_outputs: tasks,
        })
    }
}

/// Native optimization objective for a latent Kronecker GP.
#[derive(Clone, Debug)]
pub struct LatentKroneckerFit {
    /// The model, containing fitted parameters after optimization.
    pub model: LatentKroneckerGP,
}
impl MllModel for LatentKroneckerFit {
    fn raw_parameter_vector(&self) -> Vec<f64> {
        self.model.raw.clone()
    }
    fn set_raw_parameter_vector(&mut self, raw: &[f64]) -> Result<()> {
        self.model.set_raw_parameters(raw)
    }
    fn raw_bounds(&self) -> Vec<(f64, f64)> {
        self.model.raw_parameter_bounds()
    }
    fn mll_values_and_gradient(&self, raw: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let (v, g) = self.model.mll_value_and_grad(raw, true)?;
        Ok((vec![v], g))
    }
}
impl Fittable for LatentKroneckerFit {
    type Output = FitResult;
    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        let (result, raw) = minimize_with_retries(
            &self.model.raw,
            &self.model.raw_parameter_bounds(),
            options,
            |raw| self.model.mll_value_and_grad(raw, true),
            |rng| self.model.sample_all_priors(&self.model.raw, rng),
        )?;
        self.model.set_raw_parameters(&raw)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{GreaterThan, ParameterTransform};
    use crate::kernels::KernelKind;
    use crate::models::gp::ConstantMeanSpec;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    fn model(missing: bool) -> LatentKroneckerGP {
        let x = Matrix::from_column_slice(4, 1, &[0.1, 0.4, 0.8, 1.2]);
        let t = Matrix::from_column_slice(3, 1, &[0.1, 0.5, 1.0]);
        let mut y = Matrix::from_fn(4, 3, |i, j| (x[(i, 0)] * 2.0 + t[(j, 0)]).sin());
        if missing {
            y[(0, 1)] = f64::NAN;
            y[(2, 0)] = f64::INFINITY;
        }
        let options = LatentKroneckerGPOptions {
            covar_x: KernelNode::stationary(KernelKind::Matern52, Some(1)).unwrap(),
            covar_t: KernelNode::scale(
                KernelNode::stationary(KernelKind::Matern52, Some(1)).unwrap(),
            ),
            mean_x: VariationalMean::Constant(ConstantMeanSpec::default()),
            mean_t: VariationalMean::Constant(ConstantMeanSpec::default()),
            noise: GaussianLikelihoodSpec {
                noise_prior: None,
                noise_constraint: GreaterThan::new(1e-4, Some(ParameterTransform::Softplus), None)
                    .into(),
            },
            input_transform: None,
            outcome: Some((2.0, 1.7)),
        };
        let mut model = LatentKroneckerGP::new(x, t, y, 1, None, options).unwrap();
        let mut raw = model.raw.clone();
        raw[1] = 0.3;
        raw[2] = -0.4;
        model.set_raw_parameters(&raw).unwrap();
        model
    }

    #[test]
    fn latent_kronecker_mll_matches_dense_and_parameter_finite_differences() {
        for missing in [false, true] {
            let model = model(missing);
            let raw = model.raw_parameters();
            let (value, gradient) = model.mll_value_and_grad(raw, true).unwrap();
            let (kx, kt) = model.kernels(raw).unwrap();
            let full = kx
                .covariance_symmetric(&model.x)
                .unwrap()
                .kronecker(&kt.covariance_symmetric(&model.t).unwrap());
            let covariance = Matrix::from_fn(model.observed.len(), model.observed.len(), |i, j| {
                full[(model.observed[i], model.observed[j])]
                    + if i == j { model.noise() } else { 0.0 }
            });
            let factor = psd_safe_cholesky(&covariance).unwrap();
            let residual = model.train_targets().add_scalar(-model.mean(raw));
            let expected = -0.5
                * (residual.dot(&factor.solve_vec(&residual))
                    + factor.logdet()
                    + residual.len() as f64 * (2.0 * std::f64::consts::PI).ln())
                / residual.len() as f64;
            assert!((value - expected).abs() < 1e-12);
            for i in 0..raw.len() {
                let mut plus = raw.to_vec();
                plus[i] += 1e-6;
                let mut minus = raw.to_vec();
                minus[i] -= 1e-6;
                let fd = (model.mll_value_and_grad(&plus, false).unwrap().0
                    - model.mll_value_and_grad(&minus, false).unwrap().0)
                    / 2e-6;
                assert!(
                    (fd - gradient[i]).abs() < 1e-6 * (1.0 + fd.abs()),
                    "missing={missing}, param={i}, {fd} vs {}",
                    gradient[i]
                );
            }
        }
    }

    #[test]
    fn latent_kronecker_posteriors_and_feature_derivatives_match_dense() {
        for missing in [false, true] {
            let model = model(missing);
            let query = Matrix::from_column_slice(2, 1, &[0.25, 0.95]);
            for time in [
                model.t.clone(),
                Matrix::from_column_slice(2, 1, &[0.3, 0.7]),
            ] {
                let posterior = model.posterior_grid(&query, &time).unwrap();
                let dense = model
                    .dense(&model.raw)
                    .unwrap()
                    .posterior(&cartesian_inputs(&query, &time), false)
                    .unwrap();
                let (offset, scale) = model.affine();
                let expected_mean = dense
                    .mean
                    .map(|v| offset + scale * (model.mean(&model.raw) + v));
                assert!((&posterior.mean - expected_mean).amax() < 1e-12);
                assert!(
                    (&posterior.covariance - dense.covariance * (scale * scale)).amax() < 1e-12
                );
                let gradient = model.posterior_grid_with_grad(&query, &time).unwrap();
                for point in 0..query.nrows() {
                    let mut plus = query.clone();
                    plus[(point, 0)] += 1e-6;
                    let mut minus = query.clone();
                    minus[(point, 0)] -= 1e-6;
                    let pp = model.posterior_grid(&plus, &time).unwrap();
                    let pm = model.posterior_grid(&minus, &time).unwrap();
                    assert!(((&pp.mean - &pm.mean) / 2e-6 - &gradient.dmean[point]).amax() < 1e-6);
                    assert!(
                        ((&pp.covariance - &pm.covariance) / 2e-6 - &gradient.dcov[point]).amax()
                            < 1e-6
                    );
                }
            }
        }
    }

    #[test]
    fn latent_kronecker_matheron_samples_have_correct_covariance() {
        for missing in [false, true] {
            let model = model(missing);
            let query = Matrix::from_column_slice(1, 1, &[3.0]);
            for time in [
                model.t.clone(),
                Matrix::from_column_slice(2, 1, &[0.2, 1.4]),
            ] {
                let posterior = model.posterior_grid(&query, &time).unwrap();
                let width = model.y.len() + model.observed.len() + time.nrows();
                let mut rng = StdRng::seed_from_u64(32);
                let base = Matrix::from_fn(8192, width, |_, _| StandardNormal.sample(&mut rng));
                let samples = model.samples(&query, &time, &base).unwrap();
                let mean = Vector::from_fn(samples.ncols(), |i, _| {
                    samples.column(i).sum() / samples.nrows() as f64
                });
                assert!((&mean - &posterior.mean).amax() < 0.08);
                let centered = Matrix::from_fn(samples.nrows(), samples.ncols(), |i, j| {
                    samples[(i, j)] - mean[j]
                });
                let covariance = centered.transpose() * centered / (samples.nrows() - 1) as f64;
                let tolerance = 0.07 * posterior.covariance.amax();
                assert!((covariance - posterior.covariance).amax() < tolerance);
            }
        }
    }
}
