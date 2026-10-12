//! The hypervolume knowledge gradient
//! (`botorch.acquisition.multi_objective.hypervolume_knowledge_gradient`;
//! Daulton, Balandat and Bakshy, *Hypervolume Knowledge Gradient: A Lookahead
//! Approach for Multi-Objective Bayesian Optimization with Partial
//! Information*, ICML 2023).
//!
//! [`QHypervolumeKnowledgeGradient`] (qHVKG) is the expected hypervolume of
//! the best set of `num_pareto` designs after observing the candidates,
//! estimated with `n_f = num_fantasies` fantasies of the observations and
//! optimized in a *one-shot* fashion: a candidate set holds the `q`
//! candidates followed by `n_f · num_pareto` fantasy designs, the
//! `num_pareto` designs of fantasy `i` in rows `q + i·num_pareto ..`. For
//! fantasy `i`, the model (a list of independent GPs) is conditioned on the
//! fantasized observations at the candidates and pending points (on the
//! observed outputs only, with evaluation masks: *decoupled* evaluations),
//! and the value function — the hypervolume (qEHVI over an empty Pareto
//! front, or qLogEHVI with `log`) of the posterior mean (or of posterior
//! samples) at fantasy `i`'s designs — is averaged over the fantasies. With
//! a `current_value`, it is subtracted (the knowledge gradient proper), and
//! a cost-aware utility can weight the improvement by the cost of the
//! candidates. [`QMultiFidelityHypervolumeKnowledgeGradient`] evaluates the
//! value function at projected designs (e.g. the target fidelities).
//!
//! Values and exact gradients are computed through the fantasy models
//! ([`crate::models::gp`]: the posterior at the candidates, the fantasized
//! observations, the rank-update of the Cholesky factor and the posteriors
//! at the fantasy designs) and the hypervolume of the posterior means or
//! samples ([`super::hypervolume`]).
//!
//! **Differences to BoTorch:**
//!
//! * **t-batches.** BoTorch reshapes the fantasy designs of a `b × (q +
//!   n_f·num_pareto) × d` input to `n_f × b × num_pareto × d` without
//!   permuting the dimensions, so for `b > 1` the value of t-batch `j` uses
//!   fantasy designs of other t-batches (contradicting its documentation and
//!   coupling the restarts of `optimize_acqf`). Here every t-batch is
//!   evaluated on its own designs, which equals BoTorch for `b = 1`.
//! * **Gradients with posterior samples.** With `use_posterior_mean =
//!   false`, BoTorch detaches the covariance cache of the fantasy models
//!   (GPyTorch's default `detach_test_caches`), so its gradient with respect
//!   to the candidates misses the dependence of the fantasy posteriors'
//!   covariances on the candidates. The gradients here are exact (BoTorch's
//!   with `gpytorch.settings.detach_test_caches(False)`).
//! * **Cost-aware utilities.** BoTorch passes `X_evaluation_mask` to every
//!   utility, which fails for `GenericCostAwareUtility`, and samples the cost
//!   of an `InverseCostWeightedUtility(use_mean=False)` with the fantasy
//!   `ListSampler`, which fails for deterministic cost models. Both work
//!   here (the samples of a deterministic cost equal its values).
//! * **Pending points.** Setting pending points without an evaluation mask
//!   clears the previous mask (meta-pytorch/botorch#3391).

use std::sync::{Arc, RwLock};

use rand::Rng;
use rayon::prelude::*;

use crate::acquisition::cost_aware::CostAwareUtility;
use crate::acquisition::fidelity::{InputProjection, TargetFidelityProjection};
use crate::acquisition::mc_objective::McSamples;
use crate::acquisition::multi_objective::hypervolume::{HypercellBounds, LogSettings};
use crate::acquisition::multi_objective::objective::McMultiOutputObjective;
use crate::acquisition::multi_objective::{MoMcCore, Reduction};
use crate::acquisition::one_shot::OneShotAcquisitionFunction;
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::linalg::CovarianceRoot;
use crate::models::gp::{FantasyAdjoint, FantasyGp, FantasyPosterior};
use crate::models::{BoolMatrix, Model, SingleTaskGP};
use crate::optim::OptimizeAcqfOptions;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::safe_math::{logdiffexp, logmeanexp_with_grad};
use crate::{Matrix, Vector};

/// BoTorch's default number of fantasies of qHVKG.
pub const DEFAULT_NUM_FANTASIES: usize = 8;
/// BoTorch's default number of Pareto-optimal designs per fantasy.
pub const DEFAULT_NUM_PARETO: usize = 10;
/// BoTorch's default number of MC samples of the inner sampler (used with
/// `use_posterior_mean = false`).
pub const DEFAULT_INNER_SAMPLES: usize = 32;

/// qLogEHVI's defaults in the log value function (`fat=True`,
/// `tau_relu=1e-6`, `tau_max=1e-2`).
const LOG_SETTINGS: LogSettings = LogSettings {
    fat: true,
    tau_relu: 1e-6,
    tau_max: 1e-2,
};

/// Below this amount of work per fantasy, the fantasies are evaluated
/// sequentially.
const PAR_WORK: usize = 1 << 14;

/// Options of [`QHypervolumeKnowledgeGradient`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QHypervolumeKnowledgeGradientOptions {
    /// The number of fantasies `n_f` (default 8).
    pub num_fantasies: usize,
    /// The number of Pareto-optimal designs per fantasy (default 10).
    pub num_pareto: usize,
    /// One sampler per output for the fantasy observations, each with
    /// `num_fantasies` samples (BoTorch's `ListSampler`; default: Sobol
    /// samplers with random seeds).
    pub samplers: Option<Vec<Arc<SobolQMCNormalSampler>>>,
    /// The multi-output objective of the value function (default: the
    /// identity).
    pub objective: Option<Arc<dyn McMultiOutputObjective>>,
    /// The sampler of the value function with `use_posterior_mean = false`
    /// (default: 32 Sobol samples with a random seed).
    pub inner_sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Which outputs are observed at which candidate (`q × m`, BoTorch's
    /// `X_evaluation_mask`; default: all).
    pub x_evaluation_mask: Option<BoolMatrix>,
    /// Pending points (`n' × d`), conditioned on together with the candidates.
    pub x_pending: Option<Matrix>,
    /// Which outputs are observed at which pending point (`n' × m`).
    pub x_pending_evaluation_mask: Option<BoolMatrix>,
    /// The current value (log value with `log`), subtracted from the value
    /// of every fantasy; required with a cost-aware utility.
    pub current_value: Option<f64>,
    /// Use the hypervolume of the posterior mean (default `true`) instead of
    /// the expected hypervolume of posterior samples.
    pub use_posterior_mean: bool,
    /// A cost-aware utility of the improvements and the candidates.
    pub cost_aware_utility: Option<Arc<dyn CostAwareUtility>>,
    /// Return the log value (qLogEHVI value function, `logmeanexp` over the
    /// fantasies; `current_value` and the utility in log space).
    pub log: bool,
}

impl Default for QHypervolumeKnowledgeGradientOptions {
    fn default() -> Self {
        Self {
            num_fantasies: DEFAULT_NUM_FANTASIES,
            num_pareto: DEFAULT_NUM_PARETO,
            samplers: None,
            objective: None,
            inner_sampler: None,
            x_evaluation_mask: None,
            x_pending: None,
            x_pending_evaluation_mask: None,
            current_value: None,
            use_posterior_mean: true,
            cost_aware_utility: None,
            log: false,
        }
    }
}

/// The value function of the hypervolume knowledge gradient (BoTorch's
/// `_get_hv_value_function`): the hypervolume of the posterior mean
/// (`use_posterior_mean`) or the expected hypervolume of posterior samples
/// at a set of designs, computed by qEHVI (qLogEHVI with `log`) over an
/// empty Pareto front, optionally at projected designs. The shared part of
/// [`HypervolumeValueFunction`] and the knowledge gradient.
#[derive(Debug)]
struct ValueCore {
    mc: MoMcCore,
    cells: Arc<HypercellBounds>,
    use_posterior_mean: bool,
    log: bool,
    project: Option<Arc<dyn InputProjection>>,
}

/// The value of a fantasy and, with gradients, its derivatives with respect
/// to the designs (`np × d`) and the fantasy models' training data.
struct FantasyValue {
    value: f64,
    grad: Option<(Matrix, Vec<FantasyAdjoint>)>,
}

impl ValueCore {
    fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        objective: Option<Arc<dyn McMultiOutputObjective>>,
        inner_sampler: Arc<SobolQMCNormalSampler>,
        use_posterior_mean: bool,
        log: bool,
        project: Option<Arc<dyn InputProjection>>,
    ) -> Result<Self> {
        if ref_point.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "the reference point must not be empty".into(),
            ));
        }
        let m = ref_point.len();
        let cells = HypercellBounds::new(1, 1, m, ref_point.to_vec(), vec![f64::INFINITY; m])?;
        Ok(Self {
            mc: MoMcCore::new(model, Some(inner_sampler), objective, None, false)?,
            cells: Arc::new(cells),
            use_posterior_mean,
            log,
            project,
        })
    }

    fn reduction(&self) -> Reduction {
        if self.log {
            Reduction::Log(LOG_SETTINGS)
        } else {
            Reduction::Linear
        }
    }

    /// The value of fantasy `i` of the fantasy GPs `gps` (one per output)
    /// at the designs `x` (`np × d`, before the projection).
    fn fantasy_value(
        &self,
        gps: &[FantasyGp<'_>],
        i: usize,
        x: &Matrix,
        with_grad: bool,
    ) -> Result<FantasyValue> {
        let m = gps.len();
        let xp = match &self.project {
            Some(p) => p.project(x)?,
            None => x.clone(),
        };
        let n = xp.nrows();
        let with_cov = !self.use_posterior_mean;
        let posts: Vec<FantasyPosterior> = gps
            .iter()
            .map(|g| g.posterior(i, &xp, with_cov))
            .collect::<Result<_>>()?;
        // Posterior means (one "sample") or samples with the inner base
        // samples (S × n·m, column p·m + k).
        let (samples, sampled) = if self.use_posterior_mean {
            let data = Matrix::from_fn(m * n, 1, |r, _| posts[r / n].mean[r % n]);
            (McSamples::new(data, n, m)?, None)
        } else {
            let base = self.mc.sampler.base_samples(n * m)?;
            let s = base.nrows();
            let mut data = Matrix::zeros(m * n, s);
            let mut roots = Vec::with_capacity(m);
            for (k, post) in posts.iter().enumerate() {
                let cov = post.covariance.as_ref().expect("covariance requested");
                let root = CovarianceRoot::new(cov)?;
                let r = root.to_matrix();
                let zk = Matrix::from_fn(s, n, |a, c| base[(a, c * m + k)]);
                let y = &r * zk.transpose();
                for p in 0..n {
                    for a in 0..s {
                        data[(k * n + p, a)] = y[(p, a)] + post.mean[p];
                    }
                }
                roots.push((root, zk));
            }
            (McSamples::new(data, n, m)?, Some(roots))
        };
        let r = self
            .mc
            .reduce_samples(&samples, &xp, &self.cells, self.reduction(), with_grad)?;
        if !with_grad {
            return Ok(FantasyValue {
                value: r.value,
                grad: None,
            });
        }
        let sbar = r.samples_bar.expect("gradient requested");
        let mut gxp = r.x_bar.unwrap_or_else(|| Matrix::zeros(n, xp.ncols()));
        let mut adjoints = Vec::with_capacity(m);
        for (k, (g, post)) in gps.iter().zip(&posts).enumerate() {
            let block = sbar.rows(k * n, n);
            let mean_bar = Vector::from_fn(n, |p, _| block.row(p).sum());
            let cov_bar = match &sampled {
                Some(roots) => {
                    let (root, zk) = &roots[k];
                    Some(root.backward(&(block * zk))?)
                }
                None => None,
            };
            let mut acc = g.adjoint();
            gxp += g.posterior_backward(post, &mean_bar, cov_bar.as_ref(), &mut acc);
            adjoints.push(acc);
        }
        let gx = match &self.project {
            Some(p) => p.backward(x, &gxp)?,
            None => gxp,
        };
        Ok(FantasyValue {
            value: r.value,
            grad: Some((gx, adjoints)),
        })
    }
}

/// The independent single-output GPs of the outputs of `model`.
fn output_gps(model: &dyn Model) -> Result<Vec<&SingleTaskGP>> {
    model.output_gps().ok_or_else(|| {
        BotorchError::Unsupported(
            "the hypervolume knowledge gradient requires a model of independent single-output \
             GPs (a ModelListGP)"
                .into(),
        )
    })
}

/// The value function of the hypervolume knowledge gradient as an
/// acquisition function of `num_pareto` designs (BoTorch's
/// `_get_hv_value_function(model, ref_point, objective, sampler, project,
/// use_posterior_mean, log)`): the hypervolume of the posterior mean of the
/// designs (or their expected hypervolume over the posterior samples of the
/// sampler), qEHVI (`log`: qLogEHVI) over an empty Pareto front.
#[derive(Debug)]
pub struct HypervolumeValueFunction {
    core: ValueCore,
}

impl HypervolumeValueFunction {
    /// The value function of `model` (a model of independent single-output
    /// GPs) with the reference point `ref_point`, the objective (default:
    /// the identity), the sampler of posterior samples (used without
    /// `use_posterior_mean`; default: qEHVI's 128 Sobol samples), `log` and an
    /// optional projection of the designs (BoTorch's
    /// `ProjectedAcquisitionFunction`).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for other models,
    /// [`BotorchError::InvalidArgument`] for an empty reference point.
    pub fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        objective: Option<Arc<dyn McMultiOutputObjective>>,
        sampler: Option<Arc<SobolQMCNormalSampler>>,
        use_posterior_mean: bool,
        log: bool,
        project: Option<Arc<dyn InputProjection>>,
    ) -> Result<Self> {
        output_gps(model.as_ref())?;
        let sampler = match sampler {
            Some(s) => s,
            None => Arc::new(SobolQMCNormalSampler::new(
                super::DEFAULT_MO_MC_SAMPLES,
                None,
            )?),
        };
        Ok(Self {
            core: ValueCore::new(
                model,
                ref_point,
                objective,
                sampler,
                use_posterior_mean,
                log,
                project,
            )?,
        })
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let model = &self.core.mc.model;
        check_candidates(x, model.dim(), None)?;
        let gps = output_gps(model.as_ref())?
            .into_iter()
            .map(|g| {
                FantasyGp::new(
                    g,
                    &Matrix::zeros(0, x.ncols()),
                    Arc::new(Matrix::zeros(1, 0)),
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let v = self.core.fantasy_value(&gps, 0, x, with_grad)?;
        Ok((v.value, v.grad.map(|(g, _)| g)))
    }
}

impl AcquisitionFunction for HypervolumeValueFunction {
    fn model(&self) -> &Arc<dyn Model> {
        &self.core.mc.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if x_pending.is_some() {
            return Err(BotorchError::Unsupported(
                "the hypervolume value function does not support pending points".into(),
            ));
        }
        Ok(())
    }

    /// qEHVI is non-negative (BoTorch's `is_nonnegative`); qLogEHVI and the
    /// projected value function are not.
    fn is_nonnegative(&self) -> bool {
        !self.core.log && self.core.project.is_none()
    }

    fn is_log(&self) -> bool {
        self.core.log
    }
}

/// Pending points and their evaluation mask.
#[derive(Clone, Debug, Default)]
struct Pending {
    x: Option<Matrix>,
    mask: Option<BoolMatrix>,
}

/// The multi-fidelity configuration of a [`QHypervolumeKnowledgeGradient`].
#[derive(Clone, Debug)]
pub(crate) struct MultiFidelity {
    pub target_fidelities: Vec<(isize, f64)>,
}

/// BoTorch's `qHypervolumeKnowledgeGradient`: see the [module
/// documentation](self).
#[derive(Debug)]
pub struct QHypervolumeKnowledgeGradient {
    model: Arc<dyn Model>,
    ref_point: Vec<f64>,
    num_fantasies: usize,
    num_pareto: usize,
    samplers: Vec<Arc<SobolQMCNormalSampler>>,
    inner_sampler: Arc<SobolQMCNormalSampler>,
    objective: Option<Arc<dyn McMultiOutputObjective>>,
    value: ValueCore,
    x_evaluation_mask: Option<BoolMatrix>,
    pending: RwLock<Pending>,
    current_value: Option<f64>,
    use_posterior_mean: bool,
    cost_aware_utility: Option<Arc<dyn CostAwareUtility>>,
    log: bool,
    multi_fidelity: Option<MultiFidelity>,
}

/// The evaluation of one t-batch.
struct Evaluation {
    value: f64,
    grad: Option<Matrix>,
}

impl QHypervolumeKnowledgeGradient {
    /// `qHypervolumeKnowledgeGradient(model, ref_point, ...)` for a model of
    /// independent single-output GPs (a [`crate::ModelListGP`] of
    /// single-output models; BoTorch requires a `ModelList`).
    ///
    /// # Errors
    /// * [`BotorchError::Unsupported`] for other or batched models and for a
    ///   cost-aware utility without `current_value` (BoTorch's message);
    /// * [`BotorchError::InvalidArgument`] for samplers whose number of
    ///   samples differs from `num_fantasies` (BoTorch's message), a wrong
    ///   number of samplers, zero fantasies or Pareto designs, and pending
    ///   points without an evaluation mask when the candidates have one;
    /// * [`BotorchError::DimensionMismatch`] for evaluation masks or pending
    ///   points of the wrong shape.
    pub fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        options: QHypervolumeKnowledgeGradientOptions,
    ) -> Result<Self> {
        Self::build(model, ref_point, options, None, None)
    }

    fn build(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        options: QHypervolumeKnowledgeGradientOptions,
        project: Option<Arc<dyn InputProjection>>,
        multi_fidelity: Option<MultiFidelity>,
    ) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        let m = output_gps(model.as_ref())?.len();
        let o = options;
        if o.num_fantasies == 0 || o.num_pareto == 0 {
            return Err(BotorchError::InvalidArgument(
                "num_fantasies and num_pareto must be positive".into(),
            ));
        }
        let samplers = match o.samplers {
            Some(s) => {
                if s.first().map(|v| v.num_samples()) != Some(o.num_fantasies) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "The sampler shape must match num_fantasies={}.",
                        o.num_fantasies
                    )));
                }
                if s.len() != m || s.iter().any(|v| v.num_samples() != o.num_fantasies) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "expected one sampler with {} samples per output ({m}), got {}",
                        o.num_fantasies,
                        s.len()
                    )));
                }
                s
            }
            None => (0..m)
                .map(|_| SobolQMCNormalSampler::new(o.num_fantasies, None).map(Arc::new))
                .collect::<Result<_>>()?,
        };
        let inner_sampler = match o.inner_sampler {
            Some(s) => s,
            None => Arc::new(SobolQMCNormalSampler::new(DEFAULT_INNER_SAMPLES, None)?),
        };
        if o.current_value.is_none() && o.cost_aware_utility.is_some() {
            return Err(BotorchError::Unsupported(
                "Cost-aware HVKG requires current_value to be specified.".into(),
            ));
        }
        if let Some(mask) = &o.x_evaluation_mask {
            check_mask_columns(mask, m)?;
        }
        let value = ValueCore::new(
            Arc::clone(&model),
            ref_point,
            o.objective.clone(),
            Arc::clone(&inner_sampler),
            o.use_posterior_mean,
            o.log,
            project,
        )?;
        let acqf = Self {
            model,
            ref_point: ref_point.to_vec(),
            num_fantasies: o.num_fantasies,
            num_pareto: o.num_pareto,
            samplers,
            inner_sampler,
            objective: o.objective,
            value,
            x_evaluation_mask: o.x_evaluation_mask,
            pending: RwLock::new(Pending::default()),
            current_value: o.current_value,
            use_posterior_mean: o.use_posterior_mean,
            cost_aware_utility: o.cost_aware_utility,
            log: o.log,
            multi_fidelity,
        };
        acqf.set_x_pending_with_mask(o.x_pending.as_ref(), o.x_pending_evaluation_mask)?;
        Ok(acqf)
    }

    /// The reference point.
    pub fn ref_point(&self) -> &[f64] {
        &self.ref_point
    }

    /// The number of fantasies `n_f`.
    pub fn num_fantasies(&self) -> usize {
        self.num_fantasies
    }

    /// The number of Pareto-optimal designs per fantasy.
    pub fn num_pareto(&self) -> usize {
        self.num_pareto
    }

    /// `n_f · num_pareto`, the number of fantasy designs of a candidate set
    /// (BoTorch's `num_pseudo_points`).
    pub fn num_pseudo_points(&self) -> usize {
        self.num_fantasies * self.num_pareto
    }

    /// The fantasy samplers (one per output).
    pub fn samplers(&self) -> &[Arc<SobolQMCNormalSampler>] {
        &self.samplers
    }

    /// The sampler of the value function.
    pub fn inner_sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.inner_sampler
    }

    /// The objective of the value function, if given.
    pub fn objective(&self) -> Option<&Arc<dyn McMultiOutputObjective>> {
        self.objective.as_ref()
    }

    /// The current value, if given.
    pub fn current_value(&self) -> Option<f64> {
        self.current_value
    }

    /// Whether the value function uses the posterior mean.
    pub fn use_posterior_mean(&self) -> bool {
        self.use_posterior_mean
    }

    /// The evaluation mask of the candidates, if given.
    pub fn x_evaluation_mask(&self) -> Option<&BoolMatrix> {
        self.x_evaluation_mask.as_ref()
    }

    /// The evaluation mask of the pending points, if given.
    pub fn x_pending_evaluation_mask(&self) -> Option<BoolMatrix> {
        read(&self.pending).mask.clone()
    }

    /// The cost-aware utility, if given.
    pub fn cost_aware_utility(&self) -> Option<&Arc<dyn CostAwareUtility>> {
        self.cost_aware_utility.as_ref()
    }

    pub(crate) fn multi_fidelity(&self) -> Option<&MultiFidelity> {
        self.multi_fidelity.as_ref()
    }

    /// The value function of the current model (without projection), as
    /// used by the one-shot initial conditions.
    ///
    /// # Errors
    /// See [`HypervolumeValueFunction::new`].
    pub fn value_function(&self) -> Result<HypervolumeValueFunction> {
        HypervolumeValueFunction::new(
            Arc::clone(&self.model),
            &self.ref_point,
            self.objective.clone(),
            Some(Arc::clone(&self.inner_sampler)),
            self.use_posterior_mean,
            false,
            None,
        )
    }

    /// BoTorch's `DecoupledAcquisitionFunction.set_X_pending(X_pending,
    /// X_pending_evaluation_mask)`: pending points (`n' × d`) and which
    /// outputs are observed at them (`n' × m`; `None`: all). Pending points
    /// without a mask clear the previous mask (meta-pytorch/botorch#3391).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for shapes that do not match, and
    /// [`BotorchError::InvalidArgument`] for pending points without a mask
    /// when the candidates have an evaluation mask (BoTorch's message).
    pub fn set_x_pending_with_mask(
        &self,
        x_pending: Option<&Matrix>,
        mask: Option<BoolMatrix>,
    ) -> Result<()> {
        let m = self.samplers.len();
        let mut new = Pending::default();
        if let Some(p) = x_pending {
            check_pending(p, self.model.dim())?;
            match mask {
                Some(mk) => {
                    if mk.shape() != (p.nrows(), m) {
                        return Err(BotorchError::DimensionMismatch(format!(
                            "Expected `X_pending_evaluation_mask` of shape `{} x {m}`, but got \
                             {}x{}.",
                            p.nrows(),
                            mk.nrows(),
                            mk.ncols()
                        )));
                    }
                    new.mask = Some(mk);
                }
                None if self.x_evaluation_mask.is_some() => {
                    return Err(BotorchError::InvalidArgument(
                        "If `self.X_evaluation_mask` is not None, then \
                         `X_pending_evaluation_mask` must be provided."
                            .into(),
                    ))
                }
                None => {}
            }
            new.x = Some(p.clone());
        } else {
            new.mask = mask;
        }
        *write(&self.pending) = new;
        Ok(())
    }

    /// BoTorch's `construct_evaluation_mask` for `q` candidates: the
    /// candidates' mask (or all outputs) followed by the pending points'
    /// mask, if the pending points have one; otherwise the candidates' mask.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the candidates' mask does not
    /// have `q` rows.
    pub fn construct_evaluation_mask(&self, q: usize) -> Result<Option<BoolMatrix>> {
        let pending = read(&self.pending);
        let Some(pm) = &pending.mask else {
            return Ok(self.x_evaluation_mask.clone());
        };
        let m = self.samplers.len();
        let xm = match &self.x_evaluation_mask {
            Some(mk) if mk.nrows() != q => {
                return Err(BotorchError::DimensionMismatch(
                    "Expected the -2 dimension of X and X_evaluation_mask to match.".into(),
                ))
            }
            Some(mk) => mk.clone(),
            None => BoolMatrix::from_element(q, m, true),
        };
        let mut out = BoolMatrix::from_element(q + pm.nrows(), m, false);
        out.rows_mut(0, q).copy_from(&xm);
        out.rows_mut(q, pm.nrows()).copy_from(pm);
        Ok(Some(out))
    }

    /// The candidates (`q × d`) and the fantasy designs (`n_f` matrices of
    /// `num_pareto × d`) of an augmented candidate set (BoTorch's
    /// `_split_hvkg_fantasy_points` for one t-batch).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `x` has fewer than
    /// `n_f · num_pareto` points (BoTorch's message).
    pub fn split_fantasy_points(&self, x: &Matrix) -> Result<(Matrix, Vec<Matrix>)> {
        split_hvkg_fantasy_points(x, self.num_fantasies, self.num_pareto)
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<Evaluation> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let (x_actual, x_fant) = self.split_fantasy_points(x)?;
        let q = x_actual.nrows();
        let mask = self.construct_evaluation_mask(q)?;
        let x_cond = {
            let pending = read(&self.pending);
            match &pending.x {
                Some(p) => {
                    let mut out = Matrix::zeros(q + p.nrows(), d);
                    out.rows_mut(0, q).copy_from(&x_actual);
                    out.rows_mut(q, p.nrows()).copy_from(p);
                    out
                }
                None => x_actual.clone(),
            }
        };
        let nc = x_cond.nrows();
        let m = self.samplers.len();
        if let Some(mk) = &mask {
            if mk.shape() != (nc, m) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected evaluation_mask of shape `{nc} x {m}`, but got {}x{}.",
                    mk.nrows(),
                    mk.ncols()
                )));
            }
        }
        // The fantasy models of every output.
        let gps = output_gps(self.model.as_ref())?;
        let rows: Vec<Vec<usize>> = (0..m)
            .map(|k| match &mask {
                Some(mk) => (0..nc).filter(|&r| mk[(r, k)]).collect(),
                None => (0..nc).collect(),
            })
            .collect();
        let fantasies = gps
            .iter()
            .zip(&rows)
            .zip(&self.samplers)
            .map(|((g, rk), sampler)| {
                let xk = Matrix::from_fn(rk.len(), d, |i, j| x_cond[(rk[i], j)]);
                let base = if rk.is_empty() {
                    Arc::new(Matrix::zeros(self.num_fantasies, 0))
                } else {
                    sampler.base_samples(rk.len())?
                };
                FantasyGp::new(g, &xk, base, self.num_fantasies)
            })
            .collect::<Result<Vec<_>>>()?;
        // The value of every fantasy (in parallel when there is enough work).
        let work = fantasies.iter().map(|f| f.num_new() + 1).sum::<usize>()
            * self.num_pareto
            * x.ncols()
            * gps.iter().map(|g| g.num_train()).max().unwrap_or(1);
        let eval = |i: usize| {
            self.value
                .fantasy_value(&fantasies, i, &x_fant[i], with_grad)
        };
        let values: Vec<FantasyValue> = if work >= PAR_WORK && self.num_fantasies > 1 {
            (0..self.num_fantasies)
                .into_par_iter()
                .map(eval)
                .collect::<Result<_>>()?
        } else {
            (0..self.num_fantasies).map(eval).collect::<Result<_>>()?
        };
        // Current value, cost-aware utility and the mean over the fantasies.
        let raw: Vec<f64> = values.iter().map(|v| v.value).collect();
        let (deltas, delta_grad): (Vec<f64>, Vec<f64>) = match self.current_value {
            Some(cv) if self.log => raw
                .iter()
                .map(|&v| (logdiffexp(cv, v), 1.0 / -(cv - v).exp_m1()))
                .unzip(),
            Some(cv) => raw.iter().map(|&v| (v - cv, 1.0)).unzip(),
            None => (raw.clone(), vec![1.0; raw.len()]),
        };
        let (utilities, utility_pullback) = match &self.cost_aware_utility {
            Some(u) => {
                if self.log && u.log() != Some(true) {
                    return Err(BotorchError::Runtime(
                        "Cost-aware HVKG has _log=True and requires cost_aware_utility to \
                         output log utilities."
                            .into(),
                    ));
                }
                let out = u.evaluate(
                    &x_actual,
                    &deltas,
                    self.x_evaluation_mask.as_ref(),
                    with_grad,
                )?;
                (out.values, out.pullback)
            }
            None => (deltas, None),
        };
        let n_f = utilities.len() as f64;
        let (value, weights) = if self.log {
            logmeanexp_with_grad(&utilities)
        } else {
            (
                utilities.iter().sum::<f64>() / n_f,
                vec![1.0 / n_f; utilities.len()],
            )
        };
        if !with_grad {
            return Ok(Evaluation { value, grad: None });
        }
        let mut grad = Matrix::zeros(x.nrows(), d);
        let deltas_bar = match utility_pullback {
            Some(pb) => {
                let (dbar, xbar) = pb(&weights)?;
                let mut rows = grad.rows_mut(0, q);
                rows += &xbar;
                dbar
            }
            None => weights,
        };
        let mut accs: Vec<FantasyAdjoint> = fantasies.iter().map(|f| f.adjoint()).collect();
        for (i, v) in values.iter().enumerate() {
            let scale = deltas_bar[i] * delta_grad[i];
            let (gx, adjoints) = v.grad.as_ref().expect("gradient requested");
            let start = q + i * self.num_pareto;
            let mut rows = grad.rows_mut(start, self.num_pareto);
            rows += gx * scale;
            for (acc, a) in accs.iter_mut().zip(adjoints) {
                acc.add_scaled(a, scale);
            }
        }
        for ((f, acc), rk) in fantasies.iter().zip(&accs).zip(&rows) {
            let gk = f.backward(acc)?;
            for (i, &r) in rk.iter().enumerate() {
                if r < q {
                    for j in 0..d {
                        grad[(r, j)] += gk[(i, j)];
                    }
                }
            }
        }
        Ok(Evaluation {
            value,
            grad: Some(grad),
        })
    }
}

fn read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

fn check_mask_columns(mask: &BoolMatrix, m: usize) -> Result<()> {
    if mask.ncols() != m {
        return Err(BotorchError::DimensionMismatch(format!(
            "Expected X_evaluation_mask to be `q x m`, but got shape {}x{}.",
            mask.nrows(),
            mask.ncols()
        )));
    }
    Ok(())
}

/// BoTorch's `_split_hvkg_fantasy_points` for one t-batch: the first
/// `q = x.nrows() - n_f · num_pareto` points of `x` are the candidates, and
/// the designs of fantasy `i` are the rows `q + i·num_pareto ..
/// q + (i + 1)·num_pareto`.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `x` has fewer than
/// `n_f · num_pareto` points (BoTorch's message).
pub fn split_hvkg_fantasy_points(
    x: &Matrix,
    n_f: usize,
    num_pareto: usize,
) -> Result<(Matrix, Vec<Matrix>)> {
    let total = n_f * num_pareto;
    if total > x.nrows() {
        return Err(BotorchError::InvalidArgument(format!(
            "`n_f*num_pareto` ({total}) must be less than the `q`-batch dimension of `X` ({}).",
            x.nrows()
        )));
    }
    let q = x.nrows() - total;
    let actual = x.rows(0, q).into_owned();
    let fantasies = (0..n_f)
        .map(|i| x.rows(q + i * num_pareto, num_pareto).into_owned())
        .collect();
    Ok((actual, fantasies))
}

impl AcquisitionFunction for QHypervolumeKnowledgeGradient {
    fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.value)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let e = self.evaluate(x, true)?;
        Ok((e.value, e.grad.expect("gradient requested")))
    }

    fn x_pending(&self) -> Option<Matrix> {
        read(&self.pending).x.clone()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.set_x_pending_with_mask(x_pending, None)
    }

    fn is_log(&self) -> bool {
        self.log
    }

    fn as_one_shot(&self) -> Option<&dyn OneShotAcquisitionFunction> {
        Some(self)
    }
}

impl OneShotAcquisitionFunction for QHypervolumeKnowledgeGradient {
    fn get_augmented_q_batch_size(&self, q: usize) -> usize {
        q + self.num_pseudo_points()
    }

    fn extract_candidates(&self, x_full: &Matrix) -> Result<Matrix> {
        let n = self.num_pseudo_points();
        if x_full.nrows() < n {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected at least {n} points, got {}",
                x_full.nrows()
            )));
        }
        Ok(x_full.rows(0, x_full.nrows() - n).into_owned())
    }

    fn gen_one_shot_initial_conditions(
        &self,
        bounds: &Matrix,
        q: usize,
        num_restarts: usize,
        raw_samples: usize,
        options: &OptimizeAcqfOptions,
        rng: &mut dyn Rng,
    ) -> Result<(Vec<Matrix>, Vec<String>)> {
        crate::optim::one_shot::gen_one_shot_hvkg_initial_conditions(
            self,
            bounds,
            q,
            num_restarts,
            raw_samples,
            options,
            rng,
        )
    }
}

/// Options of [`QMultiFidelityHypervolumeKnowledgeGradient`].
#[derive(Clone, Debug, Default)]
pub struct QMultiFidelityHypervolumeKnowledgeGradientOptions {
    /// The options shared with [`QHypervolumeKnowledgeGradient`].
    pub hvkg: QHypervolumeKnowledgeGradientOptions,
    /// The projection of the fantasy designs before the value function is
    /// evaluated (BoTorch's `project`; default: the identity), e.g. a
    /// [`TargetFidelityProjection`].
    pub project: Option<Arc<dyn InputProjection>>,
}

/// BoTorch's `qMultiFidelityHypervolumeKnowledgeGradient`: the hypervolume
/// knowledge gradient whose value function is evaluated at projected fantasy
/// designs (`project`, e.g. onto the target fidelities), typically with a
/// cost-aware utility. `target_fidelities` are used by the one-shot initial
/// conditions (the value function is optimized at the target fidelities).
/// BoTorch's `valfunc_cls` and `valfunc_argfac` are not used by BoTorch's
/// implementation either; trace observations (`expand`) are not supported,
/// as in BoTorch.
#[derive(Debug)]
pub struct QMultiFidelityHypervolumeKnowledgeGradient {
    inner: QHypervolumeKnowledgeGradient,
}

impl QMultiFidelityHypervolumeKnowledgeGradient {
    /// `qMultiFidelityHypervolumeKnowledgeGradient(model, ref_point,
    /// target_fidelities, ..., project)`; `target_fidelities` maps columns
    /// (negative ones count from the end) to target values.
    ///
    /// # Errors
    /// See [`QHypervolumeKnowledgeGradient::new`];
    /// [`BotorchError::DimensionMismatch`] for target fidelities out of
    /// bounds.
    pub fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        target_fidelities: Vec<(isize, f64)>,
        options: QMultiFidelityHypervolumeKnowledgeGradientOptions,
    ) -> Result<Self> {
        TargetFidelityProjection::new(target_fidelities.clone(), model.dim())?;
        let project = options.project;
        Ok(Self {
            inner: QHypervolumeKnowledgeGradient::build(
                model,
                ref_point,
                options.hvkg,
                project,
                Some(MultiFidelity { target_fidelities }),
            )?,
        })
    }

    /// The target fidelities.
    pub fn target_fidelities(&self) -> &[(isize, f64)] {
        &self
            .inner
            .multi_fidelity()
            .expect("multi-fidelity configuration")
            .target_fidelities
    }

    /// The underlying knowledge gradient (all accessors of
    /// [`QHypervolumeKnowledgeGradient`]).
    pub fn as_hvkg(&self) -> &QHypervolumeKnowledgeGradient {
        &self.inner
    }
}

impl AcquisitionFunction for QMultiFidelityHypervolumeKnowledgeGradient {
    fn model(&self) -> &Arc<dyn Model> {
        self.inner.model()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        self.inner.forward(x)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        self.inner.forward_with_grad(x)
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.inner.x_pending()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.inner.set_x_pending(x_pending)
    }

    fn is_log(&self) -> bool {
        self.inner.is_log()
    }

    fn as_one_shot(&self) -> Option<&dyn OneShotAcquisitionFunction> {
        Some(&self.inner)
    }
}
