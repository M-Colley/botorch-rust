//! Single-objective joint entropy search
//! (`botorch.acquisition.joint_entropy_search`; Hvarfner et al., *Joint
//! Entropy Search for Maximally-Informed Bayesian Optimization*, NeurIPS
//! 2022; Tu et al., *Joint Entropy Search for Multi-Objective Bayesian
//! Optimization*, NeurIPS 2022).
//!
//! [`QJointEntropySearch`] (qJES) is the mutual information between the
//! observations at the candidates and the optimal input–output pair, with
//! the conditional entropy estimated per candidate point under models
//! conditioned on sampled optima (from [`super::optimal_samples`]): the
//! lower bound `"LB"` (the entropy of a Gaussian with the variance of the
//! truncated posterior) or the Monte-Carlo estimate `"MC"`.
//!
//! The values follow BoTorch step by step (the clamps at float32's `eps`,
//! `torch.distributions.Normal`'s CDF and log density, `torch.logdet`);
//! gradients are exact (dual numbers seeded on the posterior quantities of
//! every conditioned model, propagated through the models).
//!
//! **Differences to BoTorch main:**
//!
//! * The optimal inputs are input-transformed once when the model is
//!   conditioned on them (meta-pytorch/botorch#3399; BoTorch main passes
//!   `model.transform_inputs(optimal_inputs)` to `condition_on_observations`,
//!   which transforms them again).
//! * The `"MC"` estimate compares the (transformed) optimal values with the
//!   posterior through the posterior transform (#3399; BoTorch main uses the
//!   untransformed optimal outputs).
//! * The `"MC"` estimate of `q > 1` points (meta-pytorch/botorch#3428): the
//!   conditional entropy of every point uses that point's marginal density,
//!   and the entropy of the `q` observations includes the constant
//!   `0.5 (1 + log 2π)` of every point. BoTorch pairs every point's
//!   truncation ratio with the *joint* density of the `q` samples and adds
//!   the constant once, so its estimate is not the information gain of the
//!   batch (e.g. it is not additive over independent points). These density
//!   and entropy-constant corrections do not affect `q = 1`.
//! * The explicit noise of the noiseless conditioning enters the conditioned
//!   models' posterior mean as well (meta-pytorch/botorch#3396).
//! * The `"MC"` conditional normal CDF is standardized by
//!   `sqrt(1 - rho^2)`, the conditional standard deviation. BoTorch's
//!   pinned b0dd01be implementation divides by the variance instead.

use std::sync::{Arc, RwLock};

use rayon::prelude::*;

use crate::acquisition::dual::{Dual, Scalar};
use crate::acquisition::monte_carlo::{read_lock, vstack, write_lock};
use crate::acquisition::objective::ScalarizedPosteriorTransform;
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::linalg::psd_safe_cholesky;
use crate::models::{Model, SingleTaskGP};
use crate::posterior::MIN_VARIANCE;
use crate::sampling::SobolQMCNormalSampler;
use crate::{Matrix, Vector};

/// `torch.finfo(torch.float32).eps`, BoTorch's `CLAMP_LB` of the joint
/// entropy search.
pub const CLAMP_LB: f64 = 1.192_092_895_507_812_5e-7;
/// `0.5 (1 + log 2π)`, BoTorch's `MC_ADD_TERM`.
pub const MC_ADD_TERM: f64 = 1.418_938_533_204_672_7;
/// BoTorch's `MIN_INFERRED_NOISE_LEVEL`, the noise variance of the
/// noiseless conditioning on the optima.
pub const MIN_INFERRED_NOISE_LEVEL: f64 = 1e-4;
/// BoTorch's default number of MC samples of the `"MC"` estimate.
pub const DEFAULT_JES_MC_SAMPLES: usize = 64;

/// `log(sqrt(2π))`.
const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// The estimate of the conditional entropy (BoTorch's `estimation_type`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JesEstimationType {
    /// `"LB"` (default): the lower bound of the information gain.
    #[default]
    LowerBound,
    /// `"MC"`: the Monte-Carlo estimate.
    MonteCarlo,
}

impl JesEstimationType {
    /// The estimation type of BoTorch's name (`"LB"`, `"MC"`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for other names (BoTorch's message).
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "LB" => Ok(Self::LowerBound),
            "MC" => Ok(Self::MonteCarlo),
            other => Err(BotorchError::InvalidArgument(format!(
                "Estimation type {other} is not valid. Please specify any of ['MC', 'LB']"
            ))),
        }
    }

    /// BoTorch's name.
    pub fn name(self) -> &'static str {
        match self {
            Self::LowerBound => "LB",
            Self::MonteCarlo => "MC",
        }
    }
}

/// Options of [`QJointEntropySearch`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QJointEntropySearchOptions {
    /// Condition on noiseless optimal observations `f*` (Hvarfner et al.,
    /// default `true`, noise variance `1e-4`) or on noisy ones `y*` (Tu et
    /// al., the likelihood's noise).
    pub condition_noiseless: bool,
    /// The noise variance of every output's conditioning in that output's
    /// outcome-transformed space (default: `1e-4` for every output with
    /// `condition_noiseless`, BoTorch's `SingleTaskGP` semantics; a
    /// `ModelListGP` takes the noise in the original space, i.e. `1e-4 /
    /// std²`).
    pub condition_noise: Option<Vec<f64>>,
    /// A scalarizing posterior transform (required for multi-output models).
    pub posterior_transform: Option<ScalarizedPosteriorTransform>,
    /// Pending points (`m × d`), appended to every candidate set.
    pub x_pending: Option<Matrix>,
    /// The entropy estimate (default `"LB"`).
    pub estimation_type: JesEstimationType,
    /// The sampler of the `"MC"` estimate (default: 64 Sobol samples with a
    /// random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
}

impl Default for QJointEntropySearchOptions {
    fn default() -> Self {
        Self {
            condition_noiseless: true,
            condition_noise: None,
            posterior_transform: None,
            x_pending: None,
            estimation_type: JesEstimationType::LowerBound,
            sampler: None,
        }
    }
}

/// The conditional entropy (summed over the `q` points) of an optimum with
/// value `f_star` under the conditioned posterior with mean `mean` (`q`),
/// noiseless covariance `cov` (`q × q`, transformed) and noise variances
/// `noise` (`q`), for the `"LB"` or (with base samples `z`, `S × q`) the
/// `"MC"` estimate.
fn conditional_entropy<S: Scalar>(
    mean: &[S],
    cov: &[Vec<S>],
    noise: &[f64],
    f_star: f64,
    z: Option<&Matrix>,
) -> Result<S> {
    let q = mean.len();
    let zero = mean[0].lift(0.0);
    let mut total = zero;
    let Some(z) = z else {
        for j in 0..q {
            // `.variance` of the noiseless and noisy posteriors.
            let v = cov[j][j].clamp_min(MIN_VARIANCE);
            let var_m = (cov[j][j] + noise[j]).clamp_min(MIN_VARIANCE);
            let stdv = v.sqrt();
            let normalized = (-mean[j] + f_star) / stdv;
            let cdf = normalized.ncdf().clamp_min(CLAMP_LB);
            let pdf = (-(normalized * normalized) / 2.0 - LOG_SQRT_2PI).exp();
            let ratio = pdf / cdf;
            let var_truncated =
                v * (-((normalized + ratio) * ratio) + 1.0).clamp_min(CLAMP_LB) + (var_m - v);
            total = total + var_truncated.ln() * 0.5;
        }
        return Ok(total);
    };
    // Joint samples of the noisy posterior (Cholesky root, as GPyTorch's
    // root decomposition).
    let mut noisy = cov.to_vec();
    for (j, row) in noisy.iter_mut().enumerate() {
        row[j] = row[j] + noise[j];
    }
    let l = crate::acquisition::max_value_entropy_search::cholesky_scalar(&noisy)?;
    let s_n = z.nrows();
    for j in 0..q {
        let v = cov[j][j].clamp_min(MIN_VARIANCE);
        let var_m = noisy[j][j].clamp_min(MIN_VARIANCE).clamp_min(CLAMP_LB);
        let rho = (v / var_m).sqrt();
        let normalized_mvs = (-mean[j] + f_star) / v.sqrt();
        let cdf_mvs = normalized_mvs.ncdf().clamp_min(CLAMP_LB);
        let sd_m = var_m.sqrt();
        // The marginal log density of point j (as MultivariateNormal.log_prob
        // of one point).
        let c_jj = noisy[j][j];
        let mut acc = zero;
        for k in 0..s_n {
            let mut sample = mean[j];
            for i in 0..=j {
                sample = sample + l[j][i] * z[(k, i)];
            }
            let normalized_sample = (sample - mean[j]) / sd_m;
            let rescaled = (normalized_mvs - rho * normalized_sample)
                / (-(rho * rho) + 1.0).clamp_min(CLAMP_LB).sqrt();
            let cdf_rescaled = rescaled.ncdf().clamp_min(CLAMP_LB);
            let ratio = cdf_rescaled / cdf_mvs;
            let diff = sample - mean[j];
            let log_prob = -(diff * diff / c_jj) * 0.5 - c_jj.ln() * 0.5 - LOG_SQRT_2PI;
            acc = acc + ratio * (ratio.ln() + log_prob);
        }
        total = total - acc / s_n as f64;
    }
    Ok(total)
}

/// BoTorch's `qJointEntropySearch`: see the [module documentation](self).
#[derive(Debug)]
pub struct QJointEntropySearch {
    model: Arc<dyn Model>,
    transform: Option<ScalarizedPosteriorTransform>,
    optimal_inputs: Matrix,
    optimal_outputs: Matrix,
    optimal_output_values: Vec<f64>,
    /// The conditioned GP of every optimum (rows) and output (columns).
    conditioned: Vec<Vec<SingleTaskGP>>,
    estimation_type: JesEstimationType,
    sampler: Arc<SobolQMCNormalSampler>,
    x_pending: RwLock<Option<Matrix>>,
    condition_noiseless: bool,
}

impl QJointEntropySearch {
    /// `qJointEntropySearch(model, optimal_inputs, optimal_outputs,
    /// condition_noiseless, posterior_transform, X_pending, estimation_type,
    /// num_samples)` with the optimal inputs (`num_optima × d`) and outputs
    /// (`num_optima × m`) of [`super::optimal_samples::get_optimal_samples`].
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for models without independent exact GP
    /// outputs, multi-output models without a posterior transform and
    /// batched models; [`BotorchError::DimensionMismatch`] for inputs of the
    /// wrong shapes; errors of the conditioning (e.g. a fixed-noise model
    /// with `condition_noiseless = false`).
    pub fn new(
        model: Arc<dyn Model>,
        optimal_inputs: &Matrix,
        optimal_outputs: &Matrix,
        options: QJointEntropySearchOptions,
    ) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        let gps = model.output_gps().ok_or_else(|| {
            BotorchError::Unsupported(
                "qJointEntropySearch requires a model of exact GPs that supports conditioning \
                 (SingleTaskGP or ModelListGP)"
                    .into(),
            )
        })?;
        let m = gps.len();
        let d = model.dim();
        let n_opt = optimal_inputs.nrows();
        if optimal_inputs.ncols() != d || n_opt == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "optimal_inputs must be a non-empty `num_optima x {d}` matrix, got {}x{}",
                optimal_inputs.nrows(),
                optimal_inputs.ncols()
            )));
        }
        if optimal_outputs.shape() != (n_opt, m) {
            return Err(BotorchError::DimensionMismatch(format!(
                "optimal_outputs must be `num_optima x {m}`, got {}x{}",
                optimal_outputs.nrows(),
                optimal_outputs.ncols()
            )));
        }
        let o = options;
        let optimal_output_values: Vec<f64> = match &o.posterior_transform {
            Some(t) => t.evaluate(optimal_outputs)?.iter().copied().collect(),
            None => {
                if m != 1 {
                    return Err(BotorchError::Unsupported(
                        "qJointEntropySearch requires a posterior transform for multi-output \
                         models"
                            .into(),
                    ));
                }
                optimal_outputs.column(0).iter().copied().collect()
            }
        };
        if let Some(p) = &o.x_pending {
            check_pending(p, d)?;
        }
        let noise: Option<Vec<f64>> = match (&o.condition_noise, o.condition_noiseless) {
            (Some(v), _) => {
                if v.len() != m {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "expected one conditioning noise per output ({m}), got {}",
                        v.len()
                    )));
                }
                Some(v.clone())
            }
            (None, true) => Some(vec![MIN_INFERRED_NOISE_LEVEL; m]),
            (None, false) => None,
        };
        let conditioned = (0..n_opt)
            .into_par_iter()
            .map(|i| {
                let xi = optimal_inputs.rows(i, 1).into_owned();
                gps.iter()
                    .enumerate()
                    .map(|(t, g)| {
                        let yi = Matrix::from_element(1, 1, optimal_outputs[(i, t)]);
                        let nz = noise.as_ref().map(|v| Matrix::from_element(1, 1, v[t]));
                        g.condition_on_observations(&xi, &yi, nz.as_ref())
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let sampler = match o.sampler {
            Some(s) => s,
            None => Arc::new(SobolQMCNormalSampler::new(DEFAULT_JES_MC_SAMPLES, None)?),
        };
        Ok(Self {
            model,
            transform: o.posterior_transform,
            optimal_inputs: optimal_inputs.clone(),
            optimal_outputs: optimal_outputs.clone(),
            optimal_output_values,
            conditioned,
            estimation_type: o.estimation_type,
            sampler,
            x_pending: RwLock::new(o.x_pending),
            condition_noiseless: o.condition_noiseless,
        })
    }

    /// The optimal inputs (`num_optima × d`).
    pub fn optimal_inputs(&self) -> &Matrix {
        &self.optimal_inputs
    }

    /// The optimal outputs (`num_optima × m`).
    pub fn optimal_outputs(&self) -> &Matrix {
        &self.optimal_outputs
    }

    /// The (transformed) optimal values (BoTorch's `optimal_output_values`).
    pub fn optimal_output_values(&self) -> &[f64] {
        &self.optimal_output_values
    }

    /// The estimation type.
    pub fn estimation_type(&self) -> JesEstimationType {
        self.estimation_type
    }

    /// Whether the conditioning is on noiseless observations.
    pub fn condition_noiseless(&self) -> bool {
        self.condition_noiseless
    }

    /// The sampler of the `"MC"` estimate.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.sampler
    }

    /// The joint posterior (transformed; mean, noiseless covariance and noise
    /// variances) of the GPs `gps` (one per output) at `x`.
    fn joint(&self, gps: &[&SingleTaskGP], x: &Matrix) -> Result<(Vector, Matrix, Vec<f64>)> {
        let n = x.nrows();
        let m = gps.len();
        let posts = gps
            .iter()
            .map(|g| Ok((g.posterior(x, false)?, g.posterior(x, true)?)))
            .collect::<Result<Vec<_>>>()?;
        let (mean, cov, noise): (Vector, Matrix, Vec<f64>) = match &self.transform {
            Some(t) => {
                let w = t.weights();
                let mean = Vector::from_fn(n, |p, _| {
                    t.offset() + (0..m).map(|k| w[k] * posts[k].0.mean[p]).sum::<f64>()
                });
                let cov = Matrix::from_fn(n, n, |a, b| {
                    (0..m)
                        .map(|k| w[k] * w[k] * posts[k].0.covariance[(a, b)])
                        .sum::<f64>()
                });
                let noise = (0..n)
                    .map(|p| {
                        (0..m)
                            .map(|k| {
                                w[k] * w[k]
                                    * (posts[k].1.covariance[(p, p)]
                                        - posts[k].0.covariance[(p, p)])
                            })
                            .sum()
                    })
                    .collect();
                (mean, cov, noise)
            }
            None => {
                let (a, b) = &posts[0];
                let noise = (0..n)
                    .map(|p| b.covariance[(p, p)] - a.covariance[(p, p)])
                    .collect();
                (a.mean.clone(), a.covariance.clone(), noise)
            }
        };
        Ok((mean, cov, noise))
    }

    /// The gradient with respect to `x` of `⟨mean_bar, μ⟩ + ⟨cov_bar, Σ⟩` of
    /// the transformed joint posterior of `gps` at `x`.
    fn joint_vjp(
        &self,
        gps: &[&SingleTaskGP],
        x: &Matrix,
        rows: usize,
        mean_bar: &Vector,
        cov_bar: &Matrix,
    ) -> Result<Matrix> {
        let mut grad = Matrix::zeros(rows, x.ncols());
        for (k, g) in gps.iter().enumerate() {
            let w = self.transform.as_ref().map_or(1.0, |t| t.weights()[k]);
            grad += g.joint_posterior_vjp(x, 0..rows, &(mean_bar * w), &(cov_bar * (w * w)))?;
        }
        Ok(grad)
    }

    /// The conditional entropy of optimum `i` at `x` and, with gradients, its
    /// gradient with respect to the first `q` rows of `x`.
    fn optimum_entropy(
        &self,
        i: usize,
        x: &Matrix,
        q: usize,
        z: Option<&Matrix>,
        with_grad: bool,
    ) -> Result<(f64, Option<Matrix>)> {
        let gps: Vec<&SingleTaskGP> = self.conditioned[i].iter().collect();
        let (mean, cov, noise) = self.joint(&gps, x)?;
        let n = x.nrows();
        let f_star = self.optimal_output_values[i];
        if !with_grad {
            let mv: Vec<f64> = mean.iter().copied().collect();
            let cv: Vec<Vec<f64>> = (0..n)
                .map(|a| (0..n).map(|b| cov[(a, b)]).collect())
                .collect();
            return Ok((conditional_entropy(&mv, &cv, &noise, f_star, z)?, None));
        }
        // Tangents: the means and the upper triangle of the covariance (the
        // "LB" estimate reads only the variances).
        let lb = z.is_none();
        let pairs: Vec<(usize, usize)> = if lb {
            (0..n).map(|a| (a, a)).collect()
        } else {
            (0..n).flat_map(|a| (a..n).map(move |b| (a, b))).collect()
        };
        let ntan = n + pairs.len();
        macro_rules! with_dual {
            ($nn:expr) => {{
                let mv: Vec<Dual<$nn>> = (0..n)
                    .map(|a| Dual::<$nn>::variable(mean[a], a, ntan))
                    .collect();
                let mut cv: Vec<Vec<Dual<$nn>>> = (0..n)
                    .map(|a| {
                        (0..n)
                            .map(|b| Dual::<$nn>::constant(cov[(a, b)], ntan))
                            .collect()
                    })
                    .collect();
                for (k, &(a, b)) in pairs.iter().enumerate() {
                    let v = Dual::<$nn>::variable(cov[(a, b)], n + k, ntan);
                    cv[a][b] = v;
                    cv[b][a] = v;
                }
                let r = conditional_entropy(&mv, &cv, &noise, f_star, z)?;
                (r.v, r.grad().to_vec())
            }};
        }
        if ntan > 256 {
            return Err(BotorchError::Unsupported(
                "too many points for the gradient of qJointEntropySearch".into(),
            ));
        }
        let (v, g) = if ntan <= 4 {
            with_dual!(4)
        } else if ntan <= 12 {
            with_dual!(12)
        } else if ntan <= 40 {
            with_dual!(40)
        } else {
            with_dual!(256)
        };
        let mean_bar = Vector::from_fn(n, |a, _| g[a]);
        let mut cov_bar = Matrix::zeros(n, n);
        for (k, &(a, b)) in pairs.iter().enumerate() {
            cov_bar[(a, b)] = g[n + k];
        }
        Ok((v, Some(self.joint_vjp(&gps, x, q, &mean_bar, &cov_bar)?)))
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let q = x.nrows();
        let xf = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x, p], d),
            None => x.clone(),
        };
        let n = xf.nrows();
        let base_gps = self
            .model
            .output_gps()
            .ok_or_else(|| BotorchError::Unsupported("the model lost its output GPs".into()))?;
        // The initial entropy 0.5 logdet Σ (+ the MC constant of every point).
        let (_, cov0, noise0) = self.joint(&base_gps, &xf)?;
        let mut noisy0 = cov0.clone();
        for p in 0..n {
            noisy0[(p, p)] += noise0[p];
        }
        let chol0 = psd_safe_cholesky(&noisy0)?;
        let mut initial = 0.5 * chol0.logdet();
        let z = match self.estimation_type {
            JesEstimationType::LowerBound => None,
            JesEstimationType::MonteCarlo => {
                initial += MC_ADD_TERM * n as f64;
                Some(self.sampler.base_samples(n)?)
            }
        };
        let n_opt = self.conditioned.len();
        let parts = (0..n_opt)
            .into_par_iter()
            .map(|i| self.optimum_entropy(i, &xf, q, z.as_deref(), with_grad))
            .collect::<Result<Vec<_>>>()?;
        let value = initial - parts.iter().map(|(v, _)| *v).sum::<f64>() / n_opt as f64;
        if !with_grad {
            return Ok((value, None));
        }
        // d(0.5 logdet Σ)/dΣ = 0.5 Σ⁻¹ (noise is constant).
        let inv = chol0.inverse() * 0.5;
        let mut grad = self.joint_vjp(&base_gps, &xf, q, &Vector::zeros(n), &inv)?;
        for (_, g) in &parts {
            grad -= g.as_ref().expect("gradient requested") / n_opt as f64;
        }
        Ok((value, Some(grad)))
    }
}

impl AcquisitionFunction for QJointEntropySearch {
    fn model(&self) -> &Arc<dyn Model> {
        &self.model
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
            check_pending(p, self.model.dim())?;
        }
        *write_lock(&self.x_pending) = x_pending.cloned();
        Ok(())
    }
}

#[cfg(test)]
mod conditional_normal_tests {
    use super::*;

    #[test]
    fn mc_conditional_cdf_uses_standard_deviation() {
        // F ~ N(0, .3), Y = F + E, E ~ N(0, .7). At Y=1,
        // F|Y=1 ~ N(.3, .21), hence P(F <= 0|Y=1)
        // = Phi(-sqrt(3/7)), rather than Phi(-sqrt(.3)/.7).
        let conditional_cdf = 0.256_345_380_130_961_8_f64;
        let ratio = conditional_cdf / 0.5;
        let expected = -ratio * (ratio.ln() - 0.5 - LOG_SQRT_2PI);
        let samples = Matrix::from_element(1, 1, 1.0);
        let actual = conditional_entropy(&[0.0], &[vec![0.3]], &[0.7], 0.0, Some(&samples))
            .expect("valid scalar Gaussian");
        assert!((actual - expected).abs() < 2e-14, "{actual} != {expected}");
    }
}
