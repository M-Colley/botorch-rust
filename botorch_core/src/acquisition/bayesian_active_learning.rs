//! Bayesian active learning (`botorch.acquisition.bayesian_active_learning`):
//! BALD (Houlsby et al., *Bayesian Active Learning for Classification and
//! Preference Learning*, 2011) in its batch version (Kirsch et al.,
//! *BatchBALD*, NeurIPS 2019) for fully Bayesian GPs.
//!
//! [`QBayesianActiveLearningByDisagreement`] is the mutual information
//! between the observations `y` at the candidates (with observation noise)
//! and the hyperparameters, estimated by Monte Carlo as BoTorch's
//! `qBayesianActiveLearningByDisagreement` does. With the `S` MCMC samples
//! (mixture components) of posterior `N(μ_s, Σ_s)` and `N` base samples
//! `ε_n` shared by the components, `y_{n,s} = μ_s + R_s ε_n` (GPyTorch's
//! covariance root `R_s`) and `ℓ_{n,s',s} = log N(y_{n,s'} | μ_s, Σ_s)`:
//!
//! ```text
//! BALD = -1/(N S) Σ_{n,s'} log(1/S Σ_s exp ℓ_{n,s',s}) + 1/(N S) Σ_{n,s} ℓ_{n,s,s},
//! ```
//!
//! the entropy of the mixture minus the mean entropy of the components.
//! Gradients with respect to the candidates are exact (through the
//! samples, the Cholesky factors and the posteriors of the components).

use std::sync::{Arc, RwLock};

use crate::acquisition::monte_carlo::{read_lock, vstack, write_lock};
use crate::acquisition::{check_candidates, check_pending, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::linalg::{psd_safe_cholesky, CovarianceRoot};
use crate::models::Model;
use crate::sampling::SobolQMCNormalSampler;
use crate::{Matrix, Vector};

/// BoTorch's `qBayesianActiveLearningByDisagreement` on the components
/// (MCMC samples) of a fully Bayesian GP: see the [module
/// documentation](self).
#[derive(Debug)]
pub struct QBayesianActiveLearningByDisagreement {
    members: Vec<Arc<dyn Model>>,
    sampler: Arc<SobolQMCNormalSampler>,
    x_pending: RwLock<Option<Matrix>>,
}

/// The posterior of one component at the candidates.
struct Component {
    mean: Vector,
    root: CovarianceRoot,
    chol: crate::linalg::CholeskyFactor,
    inv: Matrix,
}

impl QBayesianActiveLearningByDisagreement {
    /// `qBayesianActiveLearningByDisagreement(model, sampler, X_pending)`
    /// for the components `members` (unbatched Gaussian models, e.g.
    /// the MCMC samples of a `SaasFullyBayesianSingleTaskGP`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] without components,
    /// [`BotorchError::Unsupported`] for batched or transformed components
    /// and [`BotorchError::DimensionMismatch`] for components or pending
    /// points of different dimensions.
    pub fn new(
        members: Vec<Arc<dyn Model>>,
        sampler: Arc<SobolQMCNormalSampler>,
        x_pending: Option<Matrix>,
    ) -> Result<Self> {
        let Some(first) = members.first() else {
            return Err(BotorchError::InvalidArgument(
                "BALD needs the components (MCMC samples) of a fully Bayesian model".into(),
            ));
        };
        let d = first.dim();
        let outputs = first.num_outputs();
        if outputs == 0 {
            return Err(BotorchError::InvalidArgument(
                "BALD needs nonempty Gaussian outputs".into(),
            ));
        }
        for m in &members {
            if !m.batch_shape().is_empty() || m.points_per_input() != 1 {
                return Err(BotorchError::Unsupported(
                    "BALD supports unbatched Gaussian components with one posterior point per input"
                        .into(),
                ));
            }
            crate::acquisition::require_gaussian_model(m.as_ref(), "BALD")?;
            if m.num_outputs() != outputs {
                return Err(BotorchError::DimensionMismatch(
                    "BALD components must have the same number of outputs".into(),
                ));
            }
            if m.dim() != d {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the components have {} and {d} input dimensions",
                    m.dim()
                )));
            }
        }
        if let Some(p) = &x_pending {
            check_pending(p, d)?;
        }
        Ok(Self {
            members,
            sampler,
            x_pending: RwLock::new(x_pending),
        })
    }

    /// The components.
    pub fn members(&self) -> &[Arc<dyn Model>] {
        &self.members
    }

    fn evaluate(
        &self,
        x: &Matrix,
        with_grad: bool,
        with_parameters: bool,
    ) -> Result<(f64, Option<Matrix>, Option<Vec<Vector>>)> {
        let d = self.members[0].dim();
        check_candidates(x, d, None)?;
        let xf = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let q = xf.nrows();
        let outputs = self.members[0].num_outputs();
        let event = q * outputs;
        let base = self.sampler.base_samples(event)?;
        // Public normals are point-major (q x m); native joint Gaussian
        // moments and roots are output-major (m x q).
        let base = Matrix::from_fn(base.nrows(), event, |i, j| {
            base[(i, (j % q) * outputs + j / q)]
        });
        let reverse = with_grad || with_parameters;
        let n = base.nrows();
        let s_count = self.members.len();
        let comps: Vec<Component> = self
            .members
            .iter()
            .map(|m| {
                let post = m.posterior(&xf, true)?;
                let root = CovarianceRoot::new(&post.covariance)?;
                let chol = psd_safe_cholesky(&post.covariance)?;
                let inv = chol.inverse();
                Ok(Component {
                    mean: post.mean,
                    root,
                    chol,
                    inv,
                })
            })
            .collect::<Result<_>>()?;
        // Samples y[s'][n] (joint q*m event vectors).
        let roots: Vec<Matrix> = comps.iter().map(|c| c.root.to_matrix()).collect();
        let samples: Vec<Vec<Vector>> = comps
            .iter()
            .zip(&roots)
            .map(|(c, r)| {
                (0..n)
                    .map(|i| {
                        let eps = base.row(i).transpose();
                        &c.mean + r * eps
                    })
                    .collect()
            })
            .collect();
        let log2pi = (2.0 * std::f64::consts::PI).ln();
        let logdets: Vec<f64> = comps.iter().map(|c| c.chol.logdet()).collect();
        // ℓ[s'][n][s] and the residuals' solves a = Σ_s⁻¹ (y - μ_s).
        let mut ell = vec![vec![vec![0.0; s_count]; n]; s_count];
        let mut solves: Vec<Vec<Vec<Vector>>> = Vec::new();
        for (sp, ys) in samples.iter().enumerate() {
            let mut per_n = Vec::with_capacity(n);
            for (i, y) in ys.iter().enumerate() {
                let mut per_s = Vec::with_capacity(s_count);
                for (s, c) in comps.iter().enumerate() {
                    let r = y - &c.mean;
                    let w = c.chol.solve_lower_vec(&r);
                    let quad = w.dot(&w);
                    ell[sp][i][s] = -0.5 * (quad + logdets[s] + event as f64 * log2pi);
                    if reverse {
                        per_s.push(c.chol.solve_lower_transpose_vec(&w));
                    }
                }
                per_n.push(per_s);
            }
            solves.push(per_n);
        }
        // Mixture log densities (logsumexp over the components).
        let ln_s = (s_count as f64).ln();
        let mut prev = 0.0;
        let mut weights = vec![vec![vec![0.0; s_count]; n]; s_count];
        for sp in 0..s_count {
            for i in 0..n {
                let row = &ell[sp][i];
                let mx = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lse = if mx.is_finite() {
                    mx + row.iter().map(|v| (v - mx).exp()).sum::<f64>().ln()
                } else {
                    mx
                };
                prev += lse - ln_s;
                if reverse {
                    for s in 0..s_count {
                        weights[sp][i][s] = (row[s] - lse).exp();
                    }
                }
            }
        }
        let ns = (n * s_count) as f64;
        let prev_entropy = -prev / ns;
        let post_entropy: f64 = (0..s_count)
            .map(|s| -(0..n).map(|i| ell[s][i][s]).sum::<f64>() / n as f64)
            .sum::<f64>()
            / s_count as f64;
        let value = prev_entropy - post_entropy;
        if !reverse {
            return Ok((value, None, None));
        }
        // Adjoints of the means and covariances of the components.
        let mut mean_bar: Vec<Vector> = vec![Vector::zeros(event); s_count];
        let mut cov_bar: Vec<Matrix> = vec![Matrix::zeros(event, event); s_count];
        let mut root_bar: Vec<Matrix> = vec![Matrix::zeros(event, event); s_count];
        // Σ g of every component (for the -½ g Σ_s⁻¹ terms).
        let mut g_total = vec![0.0; s_count];
        for sp in 0..s_count {
            for i in 0..n {
                let mut ybar = Vector::zeros(event);
                for s in 0..s_count {
                    let g = (if s == sp { 1.0 } else { 0.0 } - weights[sp][i][s]) / ns;
                    if g == 0.0 {
                        continue;
                    }
                    let a = &solves[sp][i][s];
                    // ∂ℓ/∂μ_s = a, ∂ℓ/∂Σ_s = ½ (a aᵀ - Σ_s⁻¹), ∂ℓ/∂y = -a.
                    mean_bar[s].axpy(g, a, 1.0);
                    cov_bar[s].ger(0.5 * g, a, a, 1.0);
                    g_total[s] += g;
                    ybar.axpy(-g, a, 1.0);
                }
                // y = μ_{s'} + R_{s'} ε.
                mean_bar[sp] += &ybar;
                let eps = base.row(i).transpose();
                root_bar[sp].ger(1.0, &ybar, &eps, 1.0);
            }
        }
        let rows = x.nrows();
        let mut grad = Matrix::zeros(rows, d);
        let mut parameters = Vec::with_capacity(s_count);
        for (s, m) in self.members.iter().enumerate() {
            let cb = &cov_bar[s] - &comps[s].inv * (0.5 * g_total[s])
                + comps[s].root.backward(&root_bar[s])?;
            if with_grad {
                grad += m.joint_posterior_vjp(&xf, 0..rows, &mean_bar[s], &cb)?;
            }
            if with_parameters {
                parameters.push(m.posterior_parameter_vjp(&xf, &mean_bar[s], &cb, true)?);
            }
        }
        Ok((
            value,
            with_grad.then_some(grad),
            with_parameters.then_some(parameters),
        ))
    }

    /// Value and raw-parameter adjoints of each MCMC component, in member order.
    /// The observation-noise law is differentiated as well as the latent GP.
    ///
    /// # Errors
    /// Ordinary BALD errors or a component's unsupported parameter pullback.
    pub fn forward_with_member_parameter_gradients(
        &self,
        x: &Matrix,
    ) -> Result<(f64, Vec<Vector>)> {
        let (value, _, gradients) = self.evaluate(x, false, true)?;
        Ok((value, gradients.expect("parameter gradients requested")))
    }
}

impl AcquisitionFunction for QBayesianActiveLearningByDisagreement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.members[0]
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g, _) = self.evaluate(x, true, false)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        read_lock(&self.x_pending).clone()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.members[0].dim())?;
        }
        *write_lock(&self.x_pending) = x_pending.cloned();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::{JointPosteriorGrad, Posterior};

    #[derive(Clone, Debug)]
    struct JointLaw {
        slope: f64,
        correlation: f64,
        noise: f64,
    }

    impl Model for JointLaw {
        fn dim(&self) -> usize {
            1
        }
        fn num_outputs(&self) -> usize {
            2
        }

        fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
            let q = x.nrows();
            let mean = Vector::from_fn(2 * q, |i, _| {
                let output = i / q;
                (self.slope + 0.4 * output as f64) * x[(i % q, 0)] + 0.1 * output as f64
            });
            let covariance = Matrix::from_fn(2 * q, 2 * q, |i, j| {
                let amplitude = if i / q != j / q {
                    self.correlation
                } else if i / q == 0 {
                    1.0
                } else {
                    1.4
                };
                let difference = x[(i % q, 0)] - x[(j % q, 0)];
                amplitude * (-0.5 * difference * difference).exp()
                    + if observation_noise && i == j {
                        self.noise
                    } else {
                        0.0
                    }
            });
            Ok(Posterior::multi_output(mean, covariance, 2))
        }

        fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
            let q = x.nrows();
            let p = self.posterior(x, false)?;
            let dmean = (0..q)
                .map(|point| {
                    Vector::from_fn(2 * q, |i, _| {
                        if i % q == point {
                            self.slope + 0.4 * (i / q) as f64
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            let dcov = (0..q)
                .map(|point| {
                    Matrix::from_fn(2 * q, 2 * q, |i, j| {
                        let sign = f64::from(j % q == point) - f64::from(i % q == point);
                        sign * (x[(i % q, 0)] - x[(j % q, 0)]) * p.covariance[(i, j)]
                    })
                })
                .collect();
            Ok(JointPosteriorGrad {
                mean: p.mean,
                covariance: p.covariance,
                dmean,
                dcov,
                num_outputs: 2,
            })
        }

        fn posterior_parameter_vjp(
            &self,
            x: &Matrix,
            mean: &Vector,
            covariance: &Matrix,
            observation_noise: bool,
        ) -> Result<Vector> {
            let q = x.nrows();
            let slope = mean
                .iter()
                .enumerate()
                .map(|(i, b)| b * x[(i % q, 0)])
                .sum();
            let mut correlation = 0.0;
            for i in 0..2 * q {
                for j in 0..2 * q {
                    if i / q != j / q {
                        let difference = x[(i % q, 0)] - x[(j % q, 0)];
                        correlation += covariance[(i, j)] * (-0.5 * difference * difference).exp();
                    }
                }
            }
            Ok(Vector::from_vec(vec![
                slope,
                correlation,
                if observation_noise {
                    covariance.trace()
                } else {
                    0.0
                },
            ]))
        }
    }

    fn models() -> Vec<JointLaw> {
        vec![
            JointLaw {
                slope: 0.4,
                correlation: 0.35,
                noise: 0.04,
            },
            JointLaw {
                slope: 1.1,
                correlation: -0.25,
                noise: 0.08,
            },
        ]
    }

    fn normals(event: usize) -> Matrix {
        Matrix::from_fn(7, event, |i, j| ((i * event + j) as f64 * 0.37).sin())
    }

    fn acquisition(
        laws: &[JointLaw],
        pending: Option<Matrix>,
    ) -> QBayesianActiveLearningByDisagreement {
        let q = 2 + pending.as_ref().map_or(0, Matrix::nrows);
        let sampler = SobolQMCNormalSampler::from_base_samples(normals(2 * q), Some(9)).unwrap();
        QBayesianActiveLearningByDisagreement::new(
            laws.iter()
                .map(|law| Arc::new(law.clone()) as Arc<dyn Model>)
                .collect(),
            Arc::new(sampler),
            pending,
        )
        .unwrap()
    }

    #[test]
    fn joint_bald_matches_independent_component_densities_and_query_finite_differences() {
        for with_pending in [false, true] {
            let x = Matrix::from_column_slice(2, 1, &[0.2, 0.7]);
            let pending = with_pending.then(|| Matrix::from_element(1, 1, 0.45));
            let full = pending
                .as_ref()
                .map_or_else(|| x.clone(), |p| vstack(&[&x, p], 1));
            let q = full.nrows();
            let laws = models();
            let acq = acquisition(&laws, pending);
            let posts = laws
                .iter()
                .map(|law| law.posterior(&full, true).unwrap())
                .collect::<Vec<_>>();
            let chols = posts
                .iter()
                .map(|p| nalgebra::Cholesky::new(p.covariance.clone()).unwrap())
                .collect::<Vec<_>>();
            let base = normals(2 * q);
            let mut expected = 0.0;
            for origin in 0..2 {
                for sample in 0..base.nrows() {
                    let eps = Vector::from_fn(2 * q, |j, _| base[(sample, (j % q) * 2 + j / q)]);
                    let draw = &posts[origin].mean + chols[origin].l() * eps;
                    let density = (0..2)
                        .map(|target| {
                            let residual = &draw - &posts[target].mean;
                            -0.5 * (residual.dot(&chols[target].solve(&residual))
                                + (2 * q) as f64 * (2.0 * std::f64::consts::PI).ln())
                                - chols[target].l().diagonal().map(f64::ln).sum()
                        })
                        .collect::<Vec<_>>();
                    let max = density[0].max(density[1]);
                    let mixture = max + ((density[0] - max).exp() + (density[1] - max).exp()).ln()
                        - 2.0_f64.ln();
                    expected += (density[origin] - mixture) / (2 * base.nrows()) as f64;
                }
            }
            let (value, gradient) = acq.forward_with_grad(&x).unwrap();
            assert!((value - expected).abs() < 2e-14, "{value} != {expected}");
            for i in 0..2 {
                let mut plus = x.clone();
                let mut minus = x.clone();
                plus[(i, 0)] += 1e-6;
                minus[(i, 0)] -= 1e-6;
                let fd = (acq.forward(&plus).unwrap() - acq.forward(&minus).unwrap()) / 2e-6;
                assert!(
                    (gradient[(i, 0)] - fd).abs() < 2e-8,
                    "query {i}: {} != {fd}",
                    gradient[(i, 0)]
                );
            }
        }
    }

    #[test]
    fn joint_bald_member_parameter_vjp_includes_cross_covariance_and_observation_noise() {
        let x = Matrix::from_column_slice(2, 1, &[0.2, 0.7]);
        let laws = models();
        let (value, gradients) = acquisition(&laws, None)
            .forward_with_member_parameter_gradients(&x)
            .unwrap();
        assert_eq!(value, acquisition(&laws, None).forward(&x).unwrap());
        assert_eq!(gradients.len(), 2);
        for (member, member_gradient) in gradients.iter().enumerate() {
            assert_eq!(member_gradient.len(), 3);
            for (parameter, &analytic) in member_gradient.iter().enumerate() {
                let mut plus = laws.clone();
                let mut minus = laws.clone();
                let perturb = |law: &mut JointLaw, step: f64| match parameter {
                    0 => law.slope += step,
                    1 => law.correlation += step,
                    _ => law.noise += step,
                };
                perturb(&mut plus[member], 1e-6);
                perturb(&mut minus[member], -1e-6);
                let fd = (acquisition(&plus, None).forward(&x).unwrap()
                    - acquisition(&minus, None).forward(&x).unwrap())
                    / 2e-6;
                assert!(
                    (analytic - fd).abs() < 2e-8,
                    "member {member} parameter {parameter}: {} != {fd}",
                    analytic
                );
            }
        }
    }
}
