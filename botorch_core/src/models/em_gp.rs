//! Empirical-Bayes Gaussian priors estimated by exact native EM.

use crate::error::{BotorchError, Result};
use crate::kernels::TreeKernel;
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::utils::empirical::{project_psd, trace_matched_shrinkage};
use crate::{Matrix, Vector};

/// One independently drawn historical function observed at its own inputs.
#[derive(Clone, Debug)]
pub struct EMDataset {
    /// Observed input rows.
    pub x: Matrix,
    /// Scalar outcomes at those rows.
    pub y: Vector,
}

/// Exact EM configuration, using either direct indices or shift interpolation.
#[derive(Clone, Debug)]
pub struct EMOptions {
    /// Homoskedastic observation variance.
    pub noise: f64,
    /// Maximum number of EM updates.
    pub iterations: usize,
    /// Kernel prior on the empirical mean, centered on the parametric mean.
    pub mean_prior: bool,
    /// Inverse-Wishart prior on the empirical covariance.
    pub covariance_prior: bool,
    /// Inverse-Wishart degrees of freedom, defaulting to inducing count plus two.
    pub iw_nu: Option<f64>,
    /// Maximum entrywise mean/covariance change for early stopping.
    pub tolerance: Option<f64>,
    /// Initialize from zero mean and identity covariance instead of the kernel.
    pub naive_init: bool,
    /// Trace-matched covariance shrinkage intensity.
    pub shrinkage: f64,
    /// Retain one immutable state after each update.
    pub record_history: bool,
}
impl Default for EMOptions {
    fn default() -> Self {
        Self {
            noise: 1e-10,
            iterations: 16,
            mean_prior: false,
            covariance_prior: false,
            iw_nu: None,
            tolerance: Some(1e-6),
            naive_init: false,
            shrinkage: 0.0,
            record_history: false,
        }
    }
}

/// Estimated moments and fixed interpolation cache.
#[derive(Clone, Debug)]
pub struct EMResult {
    /// Empirical mean at inducing points.
    pub mean: Vector,
    /// Empirical covariance at inducing points.
    pub covariance: Matrix,
    /// Cholesky factor of the parametric inducing covariance.
    pub kernel_cholesky: Matrix,
    /// Mean displacement from the parametric mean.
    pub delta_mean: Vector,
    /// Optional per-iteration moment snapshots.
    pub history: Vec<(Vector, Matrix)>,
    /// Number of completed updates.
    pub iterations: usize,
}

/// Shift interpolation from inducing moments using the exact Nyström residual.
#[derive(Clone, Debug)]
pub struct EMInterpolation {
    /// Fixed parametric covariance.
    pub kernel: TreeKernel,
    /// Inducing inputs.
    pub inducing: Matrix,
    /// Parametric constant mean.
    pub constant: f64,
    /// Current estimated inducing mean.
    pub mean: Vector,
    /// Current estimated inducing covariance.
    pub covariance: Matrix,
    /// Factorization of the parametric inducing covariance.
    pub cholesky: CholeskyFactor,
}
impl EMInterpolation {
    /// Validate dimensions and cache the inducing factorization.
    pub fn new(
        kernel: TreeKernel,
        inducing: Matrix,
        constant: f64,
        mean: Vector,
        covariance: Matrix,
    ) -> Result<Self> {
        let n = inducing.nrows();
        if n == 0
            || inducing.ncols() == 0
            || mean.len() != n
            || covariance.shape() != (n, n)
            || !constant.is_finite()
            || inducing
                .iter()
                .chain(mean.iter())
                .chain(covariance.iter())
                .any(|v| !v.is_finite())
        {
            return Err(BotorchError::DimensionMismatch(
                "invalid EM interpolation dimensions or nonfinite values".into(),
            ));
        }
        let cholesky = psd_safe_cholesky(&kernel.covariance_symmetric(&inducing)?)?;
        Ok(Self {
            kernel,
            inducing,
            constant,
            mean,
            covariance,
            cholesky,
        })
    }
    /// Parametric weights and Nyström residual at arbitrary input rows.
    pub fn weights_and_residual(&self, x: &Matrix) -> Result<(Matrix, Matrix)> {
        let cross = self.kernel.covariance(x, &self.inducing)?;
        let weights = self.cholesky.solve(&cross.transpose()).transpose();
        let residual = self.kernel.covariance_symmetric(x)? - &weights * cross.transpose();
        Ok((weights, residual))
    }
    /// Interpolated moments and inducing-to-query cross covariance.
    pub fn moments(&self, x: &Matrix) -> Result<(Vector, Matrix, Matrix)> {
        let (weights, residual) = self.weights_and_residual(x)?;
        let delta = &self.mean - Vector::from_element(self.mean.len(), self.constant);
        let mean = Vector::from_element(x.nrows(), self.constant) + &weights * delta;
        let cross = &self.covariance * weights.transpose();
        let covariance = residual + weights * &cross;
        Ok((mean, covariance, cross))
    }
}

enum Observation {
    Indexed(Vec<usize>),
    Interpolated { weights: Matrix, residual: Matrix },
}

/// Fit a prior at `inducing`, using exact conditioning in every E-step.
///
/// `indices=None` selects inducing-point shift interpolation. Otherwise each
/// dataset supplies indices into the common inducing/input set.
pub fn pretrain_em_prior(
    datasets: &[EMDataset],
    kernel: TreeKernel,
    inducing: Matrix,
    constant: f64,
    indices: Option<&[Vec<usize>]>,
    options: &EMOptions,
) -> Result<EMResult> {
    run_em_prior(datasets, kernel, inducing, constant, indices, options, None)
}

/// Continue EM from detached learned moments, retaining fresh parametric priors.
#[allow(clippy::too_many_arguments)]
pub fn run_em_prior(
    datasets: &[EMDataset],
    kernel: TreeKernel,
    inducing: Matrix,
    constant: f64,
    indices: Option<&[Vec<usize>]>,
    options: &EMOptions,
    initial: Option<(&Vector, &Matrix)>,
) -> Result<EMResult> {
    let n = inducing.nrows();
    let d = inducing.ncols();
    let count = datasets.len();
    if !options.noise.is_finite() || options.noise < 0.0 {
        return Err(BotorchError::InvalidArgument(
            "EM observation noise must be finite and nonnegative".into(),
        ));
    }
    if !(0.0..=1.0).contains(&options.shrinkage) {
        return Err(BotorchError::InvalidArgument(
            "EM covariance shrinkage must lie in [0, 1]".into(),
        ));
    }
    if options.tolerance.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return Err(BotorchError::InvalidArgument(
            "EM convergence tolerance must be finite and nonnegative".into(),
        ));
    }
    if count == 0
        || n == 0
        || d == 0
        || !constant.is_finite()
        || indices.is_some_and(|v| v.len() != count)
        || inducing.iter().any(|v| !v.is_finite())
    {
        return Err(BotorchError::InvalidArgument(
            "invalid EM prior data or options".into(),
        ));
    }
    let nu = options.iw_nu.unwrap_or(n as f64 + 2.0);
    if options.covariance_prior && (!nu.is_finite() || nu <= n as f64 - 1.0) {
        return Err(BotorchError::InvalidArgument(
            "iw_nu must be greater than inducing count minus one".into(),
        ));
    }
    let kernel_covariance = kernel.covariance_symmetric(&inducing)?;
    let kernel_cholesky = psd_safe_cholesky(&kernel_covariance)?;
    let initial_covariance = if options.naive_init {
        Matrix::identity(n, n)
    } else {
        kernel_covariance.clone()
    };
    let (mut mean, mut covariance) = if let Some((mean, covariance)) = initial {
        if mean.len() != n
            || covariance.shape() != (n, n)
            || mean.iter().chain(covariance.iter()).any(|v| !v.is_finite())
            || (covariance - covariance.transpose()).amax() > 1e-9
            || covariance.clone().symmetric_eigen().eigenvalues.min() < -1e-9
        {
            return Err(BotorchError::InvalidArgument(
                "invalid EM warm-start moments".into(),
            ));
        }
        (mean.clone(), covariance.clone())
    } else {
        (
            Vector::from_element(n, if options.naive_init { 0.0 } else { constant }),
            initial_covariance.clone(),
        )
    };
    let mut observations = Vec::with_capacity(count);
    for (i, data) in datasets.iter().enumerate() {
        if data.x.nrows() == 0
            || data.x.ncols() != d
            || data.y.len() != data.x.nrows()
            || data.x.iter().chain(data.y.iter()).any(|v| !v.is_finite())
        {
            return Err(BotorchError::DimensionMismatch(
                "EM datasets require finite n x d inputs and n outcomes".into(),
            ));
        }
        observations.push(if let Some(indices) = indices {
            if indices[i].len() != data.y.len() || indices[i].iter().any(|&j| j >= n) {
                return Err(BotorchError::DimensionMismatch(
                    "EM observation indices differ from the inducing set".into(),
                ));
            }
            if indices[i]
                .iter()
                .enumerate()
                .any(|(r, &j)| (0..d).any(|c| data.x[(r, c)] != inducing[(j, c)]))
            {
                return Err(BotorchError::InvalidArgument(
                    "EM observation indices do not identify the corresponding input rows".into(),
                ));
            }
            Observation::Indexed(indices[i].clone())
        } else {
            let cross = kernel.covariance(&data.x, &inducing)?;
            let weights = kernel_cholesky.solve(&cross.transpose()).transpose();
            let residual = kernel.covariance_symmetric(&data.x)? - &weights * cross.transpose();
            Observation::Interpolated { weights, residual }
        });
    }
    let mean_prior_inverse = if options.mean_prior {
        Some(psd_safe_cholesky(&initial_covariance)?.inverse())
    } else {
        None
    };
    let mut history = Vec::new();
    let mut completed = 0;
    for iteration in 0..options.iterations {
        let mut means = Vec::with_capacity(count);
        let mut covariances = Vec::with_capacity(count);
        for (data, observation) in datasets.iter().zip(&observations) {
            let size = data.y.len();
            let (observed_mean, mut observed_covariance, cross) = match observation {
                Observation::Indexed(indices) => (
                    Vector::from_fn(size, |i, _| mean[indices[i]]),
                    Matrix::from_fn(size, size, |i, j| covariance[(indices[i], indices[j])]),
                    Matrix::from_fn(n, size, |i, j| covariance[(i, indices[j])]),
                ),
                Observation::Interpolated { weights, residual } => {
                    let cross = &covariance * weights.transpose();
                    (
                        Vector::from_element(size, constant)
                            + weights * (&mean - Vector::from_element(n, constant)),
                        residual + weights * &cross,
                        cross,
                    )
                }
            };
            for j in 0..size {
                observed_covariance[(j, j)] += options.noise;
            }
            let chol = psd_safe_cholesky(&observed_covariance)?;
            let alpha = chol.solve_vec(&(&data.y - observed_mean));
            means.push(&mean + &cross * alpha);
            let mut lower = cross.transpose();
            chol.solve_lower_mut(&mut lower);
            covariances.push(&covariance - lower.transpose() * lower);
        }
        let sum = means.iter().fold(Vector::zeros(n), |sum, v| sum + v);
        let updated_mean = if let Some(inverse_prior) = &mean_prior_inverse {
            let inverse = psd_safe_cholesky(&covariance)?.inverse();
            let precision = &inverse * count as f64 + inverse_prior;
            let prior_mean = Vector::from_element(n, constant);
            psd_safe_cholesky(&precision)?.solve_vec(&(inverse * sum + inverse_prior * prior_mean))
        } else {
            sum / count as f64
        };
        let mut scatter = Matrix::zeros(n, n);
        for (conditional_mean, conditional_covariance) in means.iter().zip(&covariances) {
            let diff = conditional_mean - &updated_mean;
            scatter += conditional_covariance + &diff * diff.transpose();
        }
        let mut updated_covariance = if options.covariance_prior {
            let scale = nu + n as f64 + 1.0;
            (scatter + &initial_covariance * scale) / (count as f64 + scale)
        } else {
            scatter / count as f64
        };
        if options.shrinkage > 0.0 {
            updated_covariance = trace_matched_shrinkage(
                &updated_covariance,
                &kernel_covariance,
                options.shrinkage,
            )?;
        }
        updated_covariance = project_psd(&updated_covariance, 1e-10)?;
        let change_mean = (&updated_mean - &mean).amax();
        let change_covariance = (&updated_covariance - &covariance).amax();
        mean = updated_mean;
        covariance = updated_covariance;
        completed = iteration + 1;
        if options.record_history {
            history.push((mean.clone(), covariance.clone()));
        }
        if options
            .tolerance
            .is_some_and(|tol| change_mean < tol && change_covariance < tol)
        {
            break;
        }
    }
    let delta_mean = &mean - Vector::from_element(n, constant);
    Ok(EMResult {
        mean,
        covariance,
        kernel_cholesky: kernel_cholesky.l().clone(),
        delta_mean,
        history,
        iterations: completed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{KernelKind, KernelNode};
    fn data() -> (Vec<EMDataset>, Matrix, TreeKernel) {
        let z = Matrix::from_column_slice(3, 1, &[0.0, 0.4, 1.0]);
        let data = (0..4)
            .map(|r| EMDataset {
                x: z.clone(),
                y: Vector::from_fn(3, |j, _| ((r + j * 2 + 1) as f64).sin() + r as f64 * 0.1),
            })
            .collect();
        let node = KernelNode::stationary(KernelKind::Rbf, Some(1)).unwrap();
        let kernel = TreeKernel::build(&node, &[0.1], 1).unwrap();
        (data, z, kernel)
    }
    #[test]
    fn complete_noiseless_em_recovers_sample_moments() {
        let (data, z, kernel) = data();
        let indices = vec![vec![0, 1, 2]; 4];
        let result = pretrain_em_prior(
            &data,
            kernel,
            z,
            0.0,
            Some(&indices),
            &EMOptions {
                noise: 0.0,
                iterations: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let mean = data.iter().fold(Vector::zeros(3), |s, d| s + &d.y) / 4.0;
        let covariance = data.iter().fold(Matrix::zeros(3, 3), |s, d| {
            let delta = &d.y - &mean;
            s + &delta * delta.transpose()
        }) / 4.0;
        assert!((&result.mean - mean).amax() < 1e-12);
        assert!((&result.covariance - covariance).amax() < 2e-10);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn map_mean_prior_preserves_parametric_location() {
        let (data, z, kernel) = data();
        let indices = vec![vec![0, 1, 2]; 4];
        let offset = 7.0;
        let shifted: Vec<_> = data
            .iter()
            .map(|d| EMDataset {
                x: d.x.clone(),
                y: &d.y + Vector::from_element(d.y.len(), offset),
            })
            .collect();
        for inducing in [false, true] {
            let options = EMOptions {
                noise: 0.03,
                iterations: 4,
                mean_prior: true,
                covariance_prior: true,
                shrinkage: 0.2,
                tolerance: None,
                ..Default::default()
            };
            let selection = if inducing {
                None
            } else {
                Some(indices.as_slice())
            };
            let base =
                pretrain_em_prior(&data, kernel.clone(), z.clone(), 0.35, selection, &options)
                    .unwrap();
            let shifted = pretrain_em_prior(
                &shifted,
                kernel.clone(),
                z.clone(),
                0.35 + offset,
                selection,
                &options,
            )
            .unwrap();
            assert!((&shifted.mean - &base.mean - Vector::from_element(3, offset)).amax() < 1e-10);
            assert!((&shifted.covariance - &base.covariance).amax() < 1e-10);
        }
    }

    #[test]
    fn naive_initialization_keeps_the_configured_map_prior_center() {
        let (_, _, kernel) = data();
        let z = Matrix::from_element(1, 1, 0.4);
        let data = vec![EMDataset {
            x: z.clone(),
            y: Vector::from_element(1, 5.0),
        }];
        let indices = vec![vec![0]];
        let result = pretrain_em_prior(
            &data,
            kernel,
            z,
            5.0,
            Some(&indices),
            &EMOptions {
                noise: 0.1,
                iterations: 1,
                mean_prior: true,
                naive_init: true,
                ..Default::default()
            },
        )
        .unwrap();
        // E[f | y] = 5/1.1; precision-weighting it with N(5,1) gives this MAP mean.
        assert!((result.mean[0] - (5.0 / 1.1 + 5.0) / 2.0).abs() < 1e-12);
    }
    #[test]
    fn indexed_and_exact_inducing_em_match_with_map_and_shrinkage() {
        let (data, z, kernel) = data();
        let indices = vec![vec![0, 1, 2]; 4];
        for prior in [false, true] {
            let options = EMOptions {
                noise: 0.03,
                iterations: 4,
                mean_prior: prior,
                covariance_prior: prior,
                shrinkage: 0.2,
                record_history: true,
                ..Default::default()
            };
            let a = pretrain_em_prior(
                &data,
                kernel.clone(),
                z.clone(),
                0.3,
                Some(&indices),
                &options,
            )
            .unwrap();
            let b =
                pretrain_em_prior(&data, kernel.clone(), z.clone(), 0.3, None, &options).unwrap();
            assert!((&a.mean - &b.mean).amax() < 1e-10);
            assert!((&a.covariance - &b.covariance).amax() < 1e-10);
            assert_eq!(a.history.len(), 4);
            assert!((&a.history[0].0 - &a.history[3].0).amax() > 1e-6);
        }
        let mut invalid = indices;
        invalid[0].reverse();
        assert!(
            pretrain_em_prior(&data, kernel, z, 0.0, Some(&invalid), &EMOptions::default())
                .is_err()
        );
    }
}
