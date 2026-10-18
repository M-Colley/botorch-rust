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
    /// for the components `members` (unbatched single-output models, e.g.
    /// the MCMC samples of a `SaasFullyBayesianSingleTaskGP`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] without components,
    /// [`BotorchError::Unsupported`] for batched or multi-output components
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
        for m in &members {
            if !m.batch_shape().is_empty() || m.num_outputs() != 1 || m.points_per_input() != 1 {
                return Err(BotorchError::Unsupported(
                    "BALD supports unbatched single-output components (the MCMC samples of a \
                     single-output fully Bayesian GP)"
                        .into(),
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

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.members[0].dim();
        check_candidates(x, d, None)?;
        let xf = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let q = xf.nrows();
        let base = self.sampler.base_samples(q)?;
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
        // Samples y[s'][n] (q-vectors).
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
                    ell[sp][i][s] = -0.5 * (quad + logdets[s] + q as f64 * log2pi);
                    if with_grad {
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
                if with_grad {
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
        if !with_grad {
            return Ok((value, None));
        }
        // Adjoints of the means and covariances of the components.
        let mut mean_bar: Vec<Vector> = vec![Vector::zeros(q); s_count];
        let mut cov_bar: Vec<Matrix> = vec![Matrix::zeros(q, q); s_count];
        let mut root_bar: Vec<Matrix> = vec![Matrix::zeros(q, q); s_count];
        // Σ g of every component (for the -½ g Σ_s⁻¹ terms).
        let mut g_total = vec![0.0; s_count];
        for sp in 0..s_count {
            for i in 0..n {
                let mut ybar = Vector::zeros(q);
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
        for (s, m) in self.members.iter().enumerate() {
            let cb = &cov_bar[s] - &comps[s].inv * (0.5 * g_total[s])
                + comps[s].root.backward(&root_bar[s])?;
            grad += m.joint_posterior_vjp(&xf, 0..rows, &mean_bar[s], &cb)?;
        }
        Ok((value, Some(grad)))
    }
}

impl AcquisitionFunction for QBayesianActiveLearningByDisagreement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.members[0]
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
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
