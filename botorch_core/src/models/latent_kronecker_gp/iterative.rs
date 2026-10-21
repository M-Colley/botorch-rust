//! Scoped native conjugate-gradient inference and Lanczos log determinants.

use super::*;
use crate::models::matheron::{cross_self_query_vjp, kronecker_vjp};
use nalgebra::SymmetricEigen;
use rand::{RngExt, SeedableRng};

/// Solver settings captured by a native latent-grid model snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterativeOptions {
    /// Relative residual tolerance of each conjugate-gradient right-hand side.
    pub tol: f64,
    /// Maximum conjugate-gradient and Lanczos iterations.
    pub max_iter: usize,
    /// Enable iterative training solves above the Cholesky threshold.
    pub solves: bool,
    /// Enable Lanczos log determinants above the Cholesky threshold.
    pub log_prob: bool,
    /// At or below this size, retain exact inference.
    pub max_cholesky_size: usize,
    /// Rademacher trace probes; zero uses every coordinate basis vector.
    pub num_trace_samples: usize,
    /// Maximum Lanczos quadrature order.
    pub max_lanczos_quadrature_iterations: usize,
    /// Seed of retained trace probes, shared across objective evaluations.
    pub seed: u64,
}

impl Default for IterativeOptions {
    fn default() -> Self {
        Self {
            tol: 0.01,
            max_iter: 10000,
            solves: true,
            log_prob: true,
            max_cholesky_size: 800,
            num_trace_samples: 10,
            max_lanczos_quadrature_iterations: 20,
            seed: 0,
        }
    }
}

impl IterativeOptions {
    fn validate(&self) -> Result<()> {
        if !self.tol.is_finite()
            || self.tol <= 0.0
            || self.max_iter == 0
            || self.max_lanczos_quadrature_iterations == 0
        {
            return Err(BotorchError::InvalidArgument(
                "iterative tolerance and iteration limits must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct CgResult {
    solution: Matrix,
    converged: bool,
}

fn conjugate_gradient(a: &Matrix, rhs: &Matrix, options: IterativeOptions) -> Result<CgResult> {
    let n = a.nrows();
    if a.ncols() != n || rhs.nrows() != n {
        return Err(BotorchError::DimensionMismatch(
            "invalid conjugate-gradient system".into(),
        ));
    }
    if a.diagonal().iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(BotorchError::Numerical(
            "conjugate gradients require a positive finite diagonal".into(),
        ));
    }
    let mut result = Matrix::zeros(n, rhs.ncols());
    let mut converged = true;
    for column in 0..rhs.ncols() {
        let target = rhs.column(column);
        let norm = target.norm();
        if norm == 0.0 {
            continue;
        }
        let mut residual = target.into_owned();
        let mut preconditioned = Vector::from_fn(n, |i, _| residual[i] / a[(i, i)]);
        let mut direction = preconditioned.clone();
        let mut product = residual.dot(&preconditioned);
        let mut solution = Vector::zeros(n);
        for iteration in 0..options.max_iter {
            if residual.norm() <= options.tol * norm {
                let actual = target - a * &solution;
                if actual.norm() <= options.tol * norm {
                    break;
                }
                residual = actual;
                preconditioned = Vector::from_fn(n, |i, _| residual[i] / a[(i, i)]);
                direction = preconditioned.clone();
                product = residual.dot(&preconditioned);
            }
            let applied = a * &direction;
            let denominator = direction.dot(&applied);
            if denominator <= 0.0 || !denominator.is_finite() || !product.is_finite() {
                return Err(BotorchError::Numerical(
                    "conjugate gradients encountered a nonpositive curvature".into(),
                ));
            }
            let step = product / denominator;
            solution.axpy(step, &direction, 1.0);
            residual.axpy(-step, &applied, 1.0);
            if (iteration + 1) % n == 0 {
                // Finite precision can require more than n iterations. Refresh the
                // actual residual and restart rather than truncating max_iter to n.
                residual = target - a * &solution;
                preconditioned = Vector::from_fn(n, |i, _| residual[i] / a[(i, i)]);
                direction = preconditioned.clone();
                product = residual.dot(&preconditioned);
                continue;
            }
            preconditioned = Vector::from_fn(n, |i, _| residual[i] / a[(i, i)]);
            let updated = residual.dot(&preconditioned);
            direction *= updated / product;
            direction += &preconditioned;
            product = updated;
        }
        // Check the actual residual to avoid a false convergence after recurrence drift.
        converged &= (target - a * &solution).norm() <= options.tol * norm;
        result.set_column(column, &solution);
    }
    Ok(CgResult {
        solution: result,
        converged,
    })
}

fn probes(n: usize, options: IterativeOptions) -> Matrix {
    if options.num_trace_samples == 0 {
        return Matrix::identity(n, n);
    }
    let mut rng = StdRng::seed_from_u64(options.seed);
    Matrix::from_fn(n, options.num_trace_samples, |_, _| {
        if rng.random::<bool>() {
            1.0
        } else {
            -1.0
        }
    })
}

fn lanczos_logdet(a: &Matrix, probes: &Matrix, options: IterativeOptions) -> Result<f64> {
    let order = options
        .max_iter
        .min(options.max_lanczos_quadrature_iterations)
        .min(a.nrows());
    let mut estimate = 0.0;
    for probe in probes.column_iter() {
        let norm = probe.norm();
        let mut vector = probe / norm;
        let mut basis: Vec<Vector> = Vec::with_capacity(order);
        let mut diagonal = Vec::with_capacity(order);
        let mut off_diagonal = Vec::with_capacity(order.saturating_sub(1));
        for i in 0..order {
            basis.push(vector.clone());
            let mut residual = a * &vector;
            let alpha = vector.dot(&residual);
            diagonal.push(alpha);
            // Full reorthogonalization makes the quadrature stable near repeated eigenvalues.
            for previous in &basis {
                let projection = previous.dot(&residual);
                residual.axpy(-projection, previous, 1.0);
            }
            let beta = residual.norm();
            if i + 1 == order || beta <= f64::EPSILON * 10.0 * a.norm().max(1.0) {
                break;
            }
            off_diagonal.push(beta);
            vector = residual / beta;
        }
        let k = diagonal.len();
        let tridiagonal = Matrix::from_fn(k, k, |i, j| {
            if i == j {
                diagonal[i]
            } else if i + 1 == j {
                off_diagonal[i]
            } else if j + 1 == i {
                off_diagonal[j]
            } else {
                0.0
            }
        });
        let eigen = SymmetricEigen::new(tridiagonal);
        if eigen
            .eigenvalues
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(BotorchError::Numerical(
                "Lanczos encountered a nonpositive covariance eigenvalue".into(),
            ));
        }
        estimate += norm
            * norm
            * (0..k)
                .map(|i| eigen.eigenvectors[(0, i)].powi(2) * eigen.eigenvalues[i].ln())
                .sum::<f64>();
    }
    Ok(if options.num_trace_samples == 0 {
        estimate
    } else {
        estimate / probes.ncols() as f64
    })
}

impl LatentKroneckerGP {
    /// Clone this model with validated iterative settings; prediction snapshots retain them.
    pub fn with_iterative_options(&self, options: IterativeOptions) -> Result<Self> {
        options.validate()?;
        let mut model = self.clone();
        model.iterative = Some(options);
        model.iteration_limit = Arc::new(AtomicBool::new(false));
        Ok(model)
    }

    /// Consume the flag indicating an unconverged conjugate-gradient solve.
    pub fn take_iterative_warning(&self) -> bool {
        self.iteration_limit.swap(false, Ordering::Relaxed)
    }

    pub(super) fn iterative_enabled(&self, log_prob: bool) -> bool {
        self.iterative.is_some_and(|o| {
            self.observed.len() > o.max_cholesky_size
                && if log_prob { o.log_prob } else { o.solves }
        })
    }

    fn observed_covariance_at(&self, raw: &[f64]) -> Result<Matrix> {
        let (kx, kt) = self.kernels(raw)?;
        let full = kx
            .covariance_symmetric(&self.x)?
            .kronecker(&kt.covariance_symmetric(&self.t)?);
        let noise = self.options.noise.noise_constraint.transform(raw[0]);
        Ok(Matrix::from_fn(
            self.observed.len(),
            self.observed.len(),
            |i, j| full[(self.observed[i], self.observed[j])] + if i == j { noise } else { 0.0 },
        ))
    }

    pub(super) fn iterative_covariance(&self) -> Result<&Matrix> {
        self.iterative_covariance
            .get_or_init(|| self.observed_covariance_at(&self.raw))
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn iterative_solve(&self, matrix: &Matrix, rhs: &Matrix) -> Result<Matrix> {
        let options = self.iterative.expect("iterative configuration exists");
        let result = conjugate_gradient(matrix, rhs, options)?;
        if !result.converged {
            self.iteration_limit.store(true, Ordering::Relaxed);
        }
        Ok(result.solution)
    }

    fn configured_solve(&self, matrix: &Matrix, rhs: &Matrix) -> Result<Matrix> {
        if self.iterative_enabled(false) {
            self.iterative_solve(matrix, rhs)
        } else {
            Ok(psd_safe_cholesky(matrix)?.solve(rhs))
        }
    }

    pub(super) fn iterative_posterior(&self, x: &Matrix, t: &Matrix) -> Result<Posterior> {
        let (kx, kt) = self.kernels(&self.raw)?;
        let full_cross = kx
            .covariance(&self.x, x)?
            .kronecker(&kt.covariance(&self.t, t)?);
        let cross = Matrix::from_fn(self.observed.len(), x.nrows() * t.nrows(), |i, j| {
            full_cross[(self.observed[i], j)]
        });
        let matrix = self.iterative_covariance()?;
        let residual = self.train_targets().add_scalar(-self.mean(&self.raw));
        let alpha = self.iterative_solve(
            matrix,
            &Matrix::from_column_slice(residual.len(), 1, residual.as_slice()),
        )?;
        let inverse_cross = self.iterative_solve(matrix, &cross)?;
        let (offset, scale) = self.affine();
        let mean = (cross.transpose() * alpha)
            .column(0)
            .map(|v| offset + scale * (v + self.mean(&self.raw)));
        let covariance = (kx
            .covariance_symmetric(x)?
            .kronecker(&kt.covariance_symmetric(t)?)
            - cross.transpose() * inverse_cross)
            * (scale * scale);
        Ok(Posterior::single_output(
            mean,
            (&covariance + covariance.transpose()) * 0.5,
        ))
    }

    pub(super) fn iterative_grid_vjp(
        &self,
        x: &Matrix,
        t: &Matrix,
        mean: &Vector,
        covariance: &Matrix,
    ) -> Result<(Matrix, Matrix)> {
        let xt = self.transformed(x)?;
        let (kx, kt) = self.kernels(&self.raw)?;
        let cx = kx.covariance(&self.x, &xt)?;
        let ct = kt.covariance(&self.t, t)?;
        let cross = cx.kronecker(&ct);
        let observed_cross = Matrix::from_fn(self.observed.len(), mean.len(), |i, j| {
            cross[(self.observed[i], j)]
        });
        let matrix = self.iterative_covariance()?;
        let residual = self.train_targets().add_scalar(-self.mean(&self.raw));
        let alpha = self.iterative_solve(
            matrix,
            &Matrix::from_column_slice(residual.len(), 1, residual.as_slice()),
        )?;
        let inverse_cross = self.iterative_solve(matrix, &observed_cross)?;
        let scale = self.affine().1;
        let observed_bar = alpha * (mean * scale).transpose()
            - inverse_cross * (covariance + covariance.transpose()) * (scale * scale);
        let mut cross_bar = Matrix::zeros(cross.nrows(), cross.ncols());
        for (i, &row) in self.observed.iter().enumerate() {
            cross_bar.row_mut(row).copy_from(&observed_bar.row(i));
        }
        let (cx_bar, ct_bar) = kronecker_vjp(&cx, &ct, &cross_bar);
        let sx = kx.covariance_symmetric(&xt)?;
        let st = kt.covariance_symmetric(t)?;
        let (sx_bar, st_bar) = kronecker_vjp(&sx, &st, &(covariance * (scale * scale)));
        let features = cross_self_query_vjp(&kx, &self.x, &xt, &cx_bar, &sx_bar)?;
        let times = cross_self_query_vjp(&kt, &self.t, t, &ct_bar, &st_bar)?;
        Ok((self.feature_pullback(x, &features)?, times))
    }

    pub(super) fn iterative_posterior_with_grad(
        &self,
        x: &Matrix,
        t: &Matrix,
    ) -> Result<JointPosteriorGrad> {
        let p = self.posterior_grid(x, t)?;
        let q = x.nrows() * t.nrows();
        let mut dmean = vec![Vector::zeros(q); x.len()];
        let mut dcov = vec![Matrix::zeros(q, q); x.len()];
        for i in 0..q {
            let mut mean = Vector::zeros(q);
            mean[i] = 1.0;
            let (gradient, _) = self.iterative_grid_vjp(x, t, &mean, &Matrix::zeros(q, q))?;
            for row in 0..x.nrows() {
                for col in 0..x.ncols() {
                    dmean[row * x.ncols() + col][i] = gradient[(row, col)];
                }
            }
        }
        for i in 0..q {
            for j in 0..=i {
                let mut covariance = Matrix::zeros(q, q);
                covariance[(i, j)] = 1.0;
                let (gradient, _) =
                    self.iterative_grid_vjp(x, t, &Vector::zeros(q), &covariance)?;
                for row in 0..x.nrows() {
                    for col in 0..x.ncols() {
                        dcov[row * x.ncols() + col][(i, j)] = gradient[(row, col)];
                        dcov[row * x.ncols() + col][(j, i)] = gradient[(row, col)];
                    }
                }
            }
        }
        Ok(JointPosteriorGrad {
            mean: p.mean,
            covariance: p.covariance,
            dmean,
            dcov,
            num_outputs: 1,
        })
    }

    pub(super) fn iterative_mll(&self, raw: &[f64], with_grad: bool) -> Result<(f64, Vec<f64>)> {
        let matrix = self.observed_covariance_at(raw)?;
        let n = matrix.nrows();
        let nf = n as f64;
        let options = self.iterative.expect("iterative configuration exists");
        let residual = self.train_targets().add_scalar(-self.mean(raw));
        let alpha = self
            .configured_solve(
                &matrix,
                &Matrix::from_column_slice(n, 1, residual.as_slice()),
            )?
            .column(0)
            .into_owned();
        let trace_probes = probes(n, options);
        let logdet = if self.iterative_enabled(true) {
            lanczos_logdet(&matrix, &trace_probes, options)?
        } else {
            psd_safe_cholesky(&matrix)?.logdet()
        };
        let (kx, kt) = self.kernels(raw)?;
        let noise = self.options.noise.noise_constraint.transform(raw[0]);
        let mut value =
            -0.5 * (residual.dot(&alpha) + logdet + nf * (2.0 * std::f64::consts::PI).ln());
        if let Some(prior) = self.options.noise.noise_prior {
            value += prior.log_prob(noise);
        }
        value += kx.log_prior()? + kt.log_prior()?;
        let means = self.mean_parts(raw);
        for (i, mean) in [self.options.mean_x, self.options.mean_t]
            .iter()
            .enumerate()
        {
            if let VariationalMean::Constant(spec) = mean {
                if let Some(prior) = spec.prior {
                    value += prior.log_prob(means[i]);
                }
            }
        }
        value /= nf;
        let mut gradient = vec![0.0; raw.len()];
        if !with_grad {
            return Ok((value, gradient));
        }
        // Match the inverse-trace estimator used by iterative Gaussian MLL backward.
        let inverse = if self.iterative_enabled(true) {
            let solved = self.configured_solve(&matrix, &trace_probes)?;
            let estimate = solved * trace_probes.transpose();
            let estimate = if options.num_trace_samples == 0 {
                estimate
            } else {
                estimate / trace_probes.ncols() as f64
            };
            (&estimate + estimate.transpose()) * 0.5
        } else {
            psd_safe_cholesky(&matrix)?.inverse()
        };
        let w = (&alpha * alpha.transpose() - inverse) * 0.5;
        let noise_prior = self
            .options
            .noise
            .noise_prior
            .map_or(0.0, |p| p.scaled_grad_log_prob(noise, 1.0 / nf));
        gradient[0] = (w.trace() / nf + noise_prior)
            * self
                .options
                .noise
                .noise_constraint
                .transform_derivative(raw[0]);
        let mut full_bar = Matrix::zeros(self.y.len(), self.y.len());
        for (i, &a) in self.observed.iter().enumerate() {
            for (j, &b) in self.observed.iter().enumerate() {
                full_bar[(a, b)] = w[(i, j)];
            }
        }
        let sx = kx.covariance_symmetric(&self.x)?;
        let st = kt.covariance_symmetric(&self.t)?;
        let (x_bar, t_bar) = kronecker_vjp(&sx, &st, &full_bar);
        for (i, (kernel, node, input, bar)) in [
            (&kx, &self.options.covar_x, &self.x, &x_bar),
            (&kt, &self.options.covar_t, &self.t, &t_bar),
        ]
        .into_iter()
        .enumerate()
        {
            let values = kernel.contract(&kernel.prepare(input)?, bar);
            let prior = kernel.log_prior_scaled_grad(1.0 / nf)?;
            let derivatives = node.transform_derivatives(&raw[self.kernel_ranges[i].clone()]);
            for (j, index) in self.kernel_ranges[i].clone().enumerate() {
                gradient[index] = (values[j] / nf + prior[j]) * derivatives[j];
            }
        }
        for (i, mean) in [self.options.mean_x, self.options.mean_t]
            .iter()
            .enumerate()
        {
            if let VariationalMean::Constant(spec) = mean {
                let index = self.mean_indices[i].unwrap();
                let derivative = spec
                    .constraint
                    .map_or(1.0, |c| c.transform_derivative(raw[index]));
                let prior = spec
                    .prior
                    .map_or(0.0, |p| p.scaled_grad_log_prob(means[i], 1.0 / nf));
                gradient[index] = (alpha.sum() * means[1 - i] / nf + prior) * derivative;
            }
        }
        Ok((value, gradient))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_trace_options() -> IterativeOptions {
        IterativeOptions {
            tol: 1e-11,
            max_cholesky_size: 0,
            num_trace_samples: 0,
            max_lanczos_quadrature_iterations: 30,
            ..Default::default()
        }
    }

    #[test]
    fn cg_handles_zero_shared_rhs_and_reports_iteration_limit() {
        let a = Matrix::from_row_slice(3, 3, &[4.0, 1.0, 0.4, 1.0, 3.0, 0.2, 0.4, 0.2, 2.0]);
        let rhs = Matrix::from_row_slice(
            3,
            4,
            &[
                1.0, 0.0, 2.0, 1.0, -0.3, 0.0, -0.6, -0.3, 0.7, 0.0, 1.4, 0.7,
            ],
        );
        let result = conjugate_gradient(&a, &rhs, exact_trace_options()).unwrap();
        assert!(result.converged);
        assert!((&result.solution - psd_safe_cholesky(&a).unwrap().solve(&rhs)).amax() < 1e-12);
        let limited = conjugate_gradient(
            &a,
            &rhs,
            IterativeOptions {
                max_iter: 1,
                ..exact_trace_options()
            },
        )
        .unwrap();
        assert!(!limited.converged);
        assert_eq!(limited.solution.column(1).norm(), 0.0);
        assert!((&a * limited.solution - rhs).norm() > 0.1);
    }

    #[test]
    fn lanczos_matches_independent_spectral_trace_and_actual_probes() {
        let a = Matrix::from_row_slice(3, 3, &[4.0, 1.0, 0.4, 1.0, 3.0, 0.2, 0.4, 0.2, 2.0]);
        let spectral = SymmetricEigen::new(a.clone());
        let log_a = &spectral.eigenvectors
            * Matrix::from_diagonal(&spectral.eigenvalues.map(f64::ln))
            * spectral.eigenvectors.transpose();
        for samples in [0, 1, 7] {
            let options = IterativeOptions {
                num_trace_samples: samples,
                ..exact_trace_options()
            };
            let draws = probes(3, options);
            let expected = draws
                .column_iter()
                .map(|z| z.dot(&(&log_a * z)))
                .sum::<f64>()
                / if samples == 0 { 1.0 } else { samples as f64 };
            assert!((lanczos_logdet(&a, &draws, options).unwrap() - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn all_iterative_flag_combinations_match_dense_mll_and_raw_derivatives() {
        for missing in [false, true] {
            let model = super::super::tests::model(missing);
            let expected = model.mll_value_and_grad(&model.raw, true).unwrap();
            for solves in [false, true] {
                for log_prob in [false, true] {
                    let iterative = model
                        .with_iterative_options(IterativeOptions {
                            solves,
                            log_prob,
                            ..exact_trace_options()
                        })
                        .unwrap();
                    let (value, gradient) = iterative.mll_value_and_grad(&model.raw, true).unwrap();
                    assert!((value - expected.0).abs() < 1e-10);
                    for (i, &g) in gradient.iter().enumerate() {
                        assert!(
                            (g - expected.1[i]).abs() < 1e-9,
                            "raw {i}: {g} vs {}",
                            expected.1[i]
                        );
                        let mut plus = model.raw.clone();
                        plus[i] += 1e-6;
                        let mut minus = model.raw.clone();
                        minus[i] -= 1e-6;
                        let fd = (iterative.mll_value_and_grad(&plus, false).unwrap().0
                            - iterative.mll_value_and_grad(&minus, false).unwrap().0)
                            / 2e-6;
                        assert!((g - fd).abs() < 2e-6 * (1.0 + fd.abs()));
                    }
                }
            }
        }
    }

    #[test]
    fn iterative_mll_retains_noise_kernel_and_factor_mean_priors() {
        use crate::priors::Prior;
        let source = super::super::tests::model(true);
        let mut options = source.options.clone();
        options.noise.noise_prior = Some(Prior::Gamma {
            concentration: 2.0,
            rate: 3.0,
        });
        options.mean_x = VariationalMean::Constant(crate::models::gp::ConstantMeanSpec {
            prior: Some(Prior::Normal {
                loc: 0.2,
                scale: 0.7,
            }),
            constraint: None,
        });
        let model = LatentKroneckerGP::new(
            source.x.clone(),
            source.t.clone(),
            source.y.clone(),
            source.dim,
            Some(source.raw.clone()),
            options,
        )
        .unwrap();
        let iterative = model.with_iterative_options(exact_trace_options()).unwrap();
        let expected = model.mll_value_and_grad(&model.raw, true).unwrap();
        let actual = iterative.mll_value_and_grad(&model.raw, true).unwrap();
        assert!((actual.0 - expected.0).abs() < 1e-10);
        for (a, b) in actual.1.iter().zip(expected.1) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn iterative_posterior_sample_law_and_both_query_adjoints_match_dense() {
        for missing in [false, true] {
            let model = super::super::tests::model(missing);
            let iterative = model.with_iterative_options(exact_trace_options()).unwrap();
            let x = Matrix::from_column_slice(2, 1, &[0.28, 0.68]);
            let t = Matrix::from_column_slice(2, 1, &[0.23, 0.77]);
            let expected = model.posterior_grid(&x, &t).unwrap();
            let actual = iterative.posterior_grid(&x, &t).unwrap();
            assert!((&actual.mean - &expected.mean).amax() < 1e-10);
            assert!((&actual.covariance - &expected.covariance).amax() < 1e-10);
            let base = Matrix::from_fn(4, model.y.len() + model.observed.len() + 4, |i, j| {
                ((i * 3 + j * 11) as f64 * 0.17).sin()
            });
            let expected_samples = model.samples(&x, &t, &base).unwrap();
            let actual_samples = iterative.samples(&x, &t, &base).unwrap();
            assert!((&actual_samples - expected_samples).amax() < 1e-10);
            let ma = Vector::from_fn(4, |i, _| 0.2 + i as f64 * 0.3);
            let ca = Matrix::from_fn(4, 4, |i, j| ((i * 3 + j * 7) as f64 * 0.21).cos());
            let sa = Matrix::from_fn(4, 4, |i, j| ((i + j * 7) as f64 * 0.23).cos());
            for samples in [false, true] {
                let expected = if samples {
                    model.samples_vjp(&x, &t, &base, &sa)
                } else {
                    model.posterior_grid_vjp(&x, &t, &ma, &ca)
                }
                .unwrap();
                let actual = if samples {
                    iterative.samples_vjp(&x, &t, &base, &sa)
                } else {
                    iterative.posterior_grid_vjp(&x, &t, &ma, &ca)
                }
                .unwrap();
                assert!((&actual.0 - expected.0).amax() < 1e-9);
                assert!((&actual.1 - expected.1).amax() < 1e-9);
            }
            let full_gradient = iterative.posterior_grid_with_grad(&x, &t).unwrap();
            let exact_gradient = model.posterior_grid_with_grad(&x, &t).unwrap();
            for i in 0..x.len() {
                assert!((&full_gradient.dmean[i] - &exact_gradient.dmean[i]).amax() < 1e-9);
                assert!((&full_gradient.dcov[i] - &exact_gradient.dcov[i]).amax() < 1e-9);
            }
        }
    }

    #[test]
    fn settings_validate_threshold_and_iteration_limit_warning() {
        let model = super::super::tests::model(true);
        for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(model
                .with_iterative_options(IterativeOptions {
                    tol,
                    ..Default::default()
                })
                .is_err());
        }
        for options in [
            IterativeOptions {
                max_iter: 0,
                ..Default::default()
            },
            IterativeOptions {
                max_lanczos_quadrature_iterations: 0,
                ..Default::default()
            },
        ] {
            assert!(model.with_iterative_options(options).is_err());
        }
        let exact = model
            .with_iterative_options(IterativeOptions {
                max_iter: 1,
                ..Default::default()
            })
            .unwrap();
        let x = Matrix::from_column_slice(1, 1, &[0.28]);
        assert_eq!(
            exact.posterior_grid(&x, &model.t).unwrap().mean,
            model.posterior_grid(&x, &model.t).unwrap().mean
        );
        assert!(!exact.take_iterative_warning());
        let limited = model
            .with_iterative_options(IterativeOptions {
                max_iter: 1,
                ..exact_trace_options()
            })
            .unwrap();
        limited.posterior_grid(&x, &model.t).unwrap();
        assert!(limited.take_iterative_warning());
        assert!(!limited.take_iterative_warning());
    }
}
