//! The batch knowledge gradient with one-shot optimization
//! (`botorch.acquisition.knowledge_gradient`; Balandat et al., *BoTorch: A
//! Framework for Efficient Monte-Carlo Bayesian Optimization*, NeurIPS 2020;
//! Frazier, Powell and Dayanik 2008; Wu and Frazier 2016).
//!
//! [`QKnowledgeGradient`] (qKG) is the expected best value of a *value
//! function* after observing the candidates, estimated with `n_f =
//! num_fantasies` fantasies of the observations and optimized in a one-shot
//! fashion: a candidate set holds the `q` candidates followed by `n_f`
//! fantasy points, one per fantasy. For fantasy `i`, the model is conditioned
//! on observations fantasized at the candidates and pending points (BoTorch's
//! `fantasize`, with the base samples of the sampler), and the value
//! function is evaluated at fantasy point `i`: the posterior mean
//! (`PosteriorMean`, through the posterior transform) or, with an MC
//! objective, the simple regret of posterior samples (`qSimpleRegret` with
//! the inner sampler), or a value function of the classes BoTorch's
//! `valfunc_cls` selects ([`KgValueFunction`]). With a `current_value` it is
//! subtracted (the knowledge gradient proper), and the values are averaged
//! over the fantasies.
//!
//! [`QMultiFidelityKnowledgeGradient`] (qMFKG) evaluates the value function
//! at projected fantasy points (`project`, e.g. onto the target fidelities),
//! fantasizes at expanded candidate sets (`expand`, e.g. trace
//! observations) and weights the improvements by a cost-aware utility of the
//! candidates.
//!
//! Values and exact gradients with respect to the candidates and the
//! fantasy points are computed through the fantasy models
//! ([`crate::models::gp`]: the posterior at the candidates, the fantasized
//! observations, the rank update of the Cholesky factor and the posteriors
//! at the fantasy points).
//!
//! **Differences to BoTorch:**
//!
//! * **Gradients with posterior samples.** With an MC objective (or another
//!   value function that uses the posterior covariance), BoTorch evaluates
//!   the value function under GPyTorch's `fast_pred_var`, whose covariance
//!   cache of the fantasy models is detached from the candidates, so its
//!   gradient with respect to the candidates misses the dependence of the
//!   fantasy posteriors' covariances on them (fixed in
//!   meta-pytorch/botorch#3427). The gradients here are exact; the values
//!   agree with BoTorch.
//! * **`evaluate`** applies the cost-aware utility to the candidates without
//!   the pending points and before the expansion, as `forward` does; BoTorch's
//!   `evaluate` passes the expanded candidates and pending points.
//! * **`valfunc_argfac`** is evaluated by the caller (the Python layer calls
//!   it once with the current model): the arguments of the value function do
//!   not depend on the fantasy model, while BoTorch calls it with every
//!   fantasy model.

use std::sync::{Arc, RwLock};

use rand::Rng;
use rayon::prelude::*;

use crate::acquisition::cost_aware::CostAwareUtility;
use crate::acquisition::fidelity::InputProjection;
use crate::acquisition::mc_objective::McObjective;
use crate::acquisition::monte_carlo::{
    read_lock, vstack, write_lock, JointSamples, SampleUtility, SampledObjective,
    MULTI_OUTPUT_MESSAGE,
};
use crate::acquisition::objective::ScalarizedPosteriorTransform;
use crate::acquisition::one_shot::OneShotAcquisitionFunction;
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::gp::{FantasyAdjoint, FantasyGp, FantasyPosterior};
use crate::models::{Model, SingleTaskGP};
use crate::optim::OptimizeAcqfOptions;
use crate::posterior::{Posterior, MIN_VARIANCE};
use crate::sampling::{SobolQMCNormalSampler, DEFAULT_MC_SAMPLES};
use crate::utils::probability::{log_ndtr, log_ndtr_grad, ndtr, phi};
use crate::{Matrix, Vector};

/// BoTorch's default number of fantasies of qKG.
pub const DEFAULT_NUM_FANTASIES: usize = 64;
/// BoTorch's default number of MC samples of the inner sampler (used with an
/// MC objective).
pub const DEFAULT_INNER_SAMPLES: usize = 128;

/// Below this amount of work per evaluation, the fantasies are evaluated
/// sequentially.
const PAR_WORK: usize = 1 << 14;

/// The terminal value function of the knowledge gradient, evaluated on the
/// (scalarized) posterior of every fantasy model at its fantasy point(s):
/// BoTorch's `_get_value_function` (the posterior mean, or `qSimpleRegret`
/// with an MC objective) and the native classes of `valfunc_cls` with the
/// arguments of `valfunc_argfac`.
#[derive(Clone, Debug)]
pub enum KgValueFunction {
    /// `PosteriorMean(model, posterior_transform, maximize)`.
    PosteriorMean {
        /// Whether larger values are better.
        maximize: bool,
    },
    /// `PosteriorStandardDeviation(model, posterior_transform, maximize)`.
    PosteriorStandardDeviation {
        /// Whether larger values are better.
        maximize: bool,
    },
    /// `UpperConfidenceBound(model, beta, posterior_transform, maximize)`.
    UpperConfidenceBound {
        /// The trade-off parameter.
        beta: f64,
        /// Whether larger values are better.
        maximize: bool,
    },
    /// `ExpectedImprovement(model, best_f, posterior_transform, maximize)`.
    ExpectedImprovement {
        /// The incumbent.
        best_f: f64,
        /// Whether larger values are better.
        maximize: bool,
    },
    /// `LogExpectedImprovement(model, best_f, posterior_transform, maximize)`.
    LogExpectedImprovement {
        /// The incumbent.
        best_f: f64,
        /// Whether larger values are better.
        maximize: bool,
    },
    /// `ProbabilityOfImprovement(model, best_f, posterior_transform, maximize)`.
    ProbabilityOfImprovement {
        /// The incumbent.
        best_f: f64,
        /// Whether larger values are better.
        maximize: bool,
    },
    /// `LogProbabilityOfImprovement(model, best_f, posterior_transform,
    /// maximize)`.
    LogProbabilityOfImprovement {
        /// The incumbent.
        best_f: f64,
        /// Whether larger values are better.
        maximize: bool,
    },
    /// `ScalarizedPosteriorMean(model, weights, posterior_transform)`: the
    /// posterior means of the `q` points weighted by `weights`.
    ScalarizedPosteriorMean {
        /// One weight per point.
        weights: Vector,
    },
    /// A sample-reducing MC acquisition function (`qSimpleRegret`,
    /// `qExpectedImprovement`, `qProbabilityOfImprovement`,
    /// `qUpperConfidenceBound`, `qLowerConfidenceBound`,
    /// `qPosteriorStandardDeviation`) with the knowledge gradient's objective
    /// and the inner sampler (a default sampler of 512 samples if there is
    /// none).
    MonteCarlo(SampleUtility),
}

impl KgValueFunction {
    /// Whether the value function reads the posterior covariance.
    pub(crate) fn uses_covariance(&self) -> bool {
        !matches!(
            self,
            Self::PosteriorMean { .. } | Self::ScalarizedPosteriorMean { .. }
        )
    }

    /// Whether the value function is analytic (`q = 1` and a single output).
    fn is_analytic(&self) -> bool {
        !matches!(self, Self::MonteCarlo(_))
    }

    /// The BoTorch class name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PosteriorMean { .. } => "PosteriorMean",
            Self::PosteriorStandardDeviation { .. } => "PosteriorStandardDeviation",
            Self::UpperConfidenceBound { .. } => "UpperConfidenceBound",
            Self::ExpectedImprovement { .. } => "ExpectedImprovement",
            Self::LogExpectedImprovement { .. } => "LogExpectedImprovement",
            Self::ProbabilityOfImprovement { .. } => "ProbabilityOfImprovement",
            Self::LogProbabilityOfImprovement { .. } => "LogProbabilityOfImprovement",
            Self::ScalarizedPosteriorMean { .. } => "ScalarizedPosteriorMean",
            Self::MonteCarlo(u) => match u {
                SampleUtility::ExpectedImprovement { .. } => "qExpectedImprovement",
                SampleUtility::ProbabilityOfImprovement { .. } => "qProbabilityOfImprovement",
                SampleUtility::SimpleRegret => "qSimpleRegret",
                SampleUtility::ConfidenceBound { beta_prime } if *beta_prime < 0.0 => {
                    "qLowerConfidenceBound"
                }
                SampleUtility::ConfidenceBound { .. } => "qUpperConfidenceBound",
                SampleUtility::PosteriorStandardDeviation => "qPosteriorStandardDeviation",
            },
        }
    }
}

/// The value of a value function on a posterior and the adjoints of the
/// posterior's mean, covariance and of the points (and of `best_f`, for the
/// value functions that have one; zero for the others).
pub(crate) struct PosteriorValue {
    pub value: f64,
    pub mean_bar: Vector,
    pub cov_bar: Option<Matrix>,
    pub x_bar: Option<Matrix>,
    pub best_f_bar: f64,
}

/// `(u, du/dμ, du/dσ)` of the scaled improvement `u = ±(μ - best_f) / σ`.
fn scaled_improvement(mean: f64, sigma: f64, best_f: f64, maximize: bool) -> (f64, f64, f64) {
    let diff = mean - best_f;
    let u = diff / sigma;
    let (du_dm, du_ds) = (1.0 / sigma, -diff / (sigma * sigma));
    if maximize {
        (u, du_dm, du_ds)
    } else {
        (-u, -du_dm, -du_ds)
    }
}

/// A value function with its context: the MC objective and sampler of the
/// sample-based value functions.
#[derive(Clone, Debug)]
pub(crate) struct ValueFunctionCore {
    pub kind: KgValueFunction,
    pub objective: Option<Arc<dyn McObjective>>,
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
}

impl ValueFunctionCore {
    /// The value at the posterior `post` (single-output for the analytic
    /// value functions) of the points `x` (`n × d`).
    pub(crate) fn evaluate(
        &self,
        post: &Posterior,
        x: &Matrix,
        with_grad: bool,
    ) -> Result<PosteriorValue> {
        let n = post.num_points();
        let m = post.num_outputs();
        if self.kind.is_analytic() {
            if m != 1 {
                return Err(BotorchError::Unsupported(
                    "Must specify a posterior transform when using a multi-output model.".into(),
                ));
            }
            if let KgValueFunction::ScalarizedPosteriorMean { weights } = &self.kind {
                if weights.len() != n {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "ScalarizedPosteriorMean has {} weights for {n} points",
                        weights.len()
                    )));
                }
                return Ok(PosteriorValue {
                    value: post.mean.dot(weights),
                    mean_bar: weights.clone(),
                    cov_bar: None,
                    x_bar: None,
                    best_f_bar: 0.0,
                });
            }
            if n != 1 {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected X to be `batch_shape x q=1 x d`, but got X with {n} points."
                )));
            }
            return Ok(self.analytic(post.mean[0], post.covariance[(0, 0)]));
        }
        let KgValueFunction::MonteCarlo(utility) = &self.kind else {
            unreachable!("analytic value functions are handled above")
        };
        if self.objective.is_none() && m != 1 {
            return Err(BotorchError::Unsupported(MULTI_OUTPUT_MESSAGE.into()));
        }
        let sampler = self.sampler.as_ref().ok_or_else(|| {
            BotorchError::Runtime("sample-based value functions need a sampler".into())
        })?;
        let base = sampler.base_samples(n * m)?;
        let js = JointSamples::from_posterior(x.clone(), post.clone(), base, false)?;
        let so = SampledObjective::from_joint(js, self.objective.as_deref(), with_grad)?;
        let r = utility.reduce(&so.obj, None, with_grad);
        if !with_grad {
            return Ok(PosteriorValue {
                value: r.value,
                mean_bar: Vector::zeros(0),
                cov_bar: None,
                x_bar: None,
                best_f_bar: 0.0,
            });
        }
        let obj_bar = r.obj_bar.expect("gradient requested");
        // The utilities depend on `obj - best_f`.
        let best_f_bar = match utility {
            SampleUtility::ExpectedImprovement { .. }
            | SampleUtility::ProbabilityOfImprovement { .. } => -obj_bar.sum(),
            _ => 0.0,
        };
        let (mean_bar, cov_bar, x_bar) = so.posterior_adjoints(&obj_bar, None)?;
        Ok(PosteriorValue {
            value: r.value,
            mean_bar,
            cov_bar: Some(cov_bar),
            x_bar,
            best_f_bar,
        })
    }

    /// An analytic value function at a single-output, single-point posterior
    /// with mean `mean` and (unclamped) variance `var`; `sigma` is BoTorch's
    /// `sqrt(max(variance, 1e-12))` of the clamped variance.
    fn analytic(&self, mean: f64, var: f64) -> PosteriorValue {
        let clamped = var <= MIN_VARIANCE;
        let v = if clamped { MIN_VARIANCE } else { var };
        let sigma = v.max(crate::acquisition::analytic::MIN_VAR).sqrt();
        // dσ/dvar (zero where the variance is clamped).
        let ds_dv = if clamped { 0.0 } else { 0.5 / sigma };
        let (value, dm, ds) = match self.kind {
            KgValueFunction::PosteriorMean { maximize } => {
                (if maximize { mean } else { -mean }, sgn(maximize), 0.0)
            }
            KgValueFunction::PosteriorStandardDeviation { maximize } => {
                (if maximize { sigma } else { -sigma }, 0.0, sgn(maximize))
            }
            KgValueFunction::UpperConfidenceBound { beta, maximize } => {
                let sb = beta.sqrt();
                (
                    (if maximize { mean } else { -mean }) + sb * sigma,
                    sgn(maximize),
                    sb,
                )
            }
            KgValueFunction::ExpectedImprovement { best_f, maximize } => {
                let (u, du_dm, du_ds) = scaled_improvement(mean, sigma, best_f, maximize);
                let h = crate::acquisition::analytic::ei_helper(u);
                let dh = ndtr(u);
                (sigma * h, sigma * dh * du_dm, h + sigma * dh * du_ds)
            }
            KgValueFunction::LogExpectedImprovement { best_f, maximize } => {
                let (u, du_dm, du_ds) = scaled_improvement(mean, sigma, best_f, maximize);
                let g = crate::acquisition::analytic::log_ei_helper_grad(u);
                (
                    crate::acquisition::analytic::log_ei_helper(u) + sigma.ln(),
                    g * du_dm,
                    g * du_ds + 1.0 / sigma,
                )
            }
            KgValueFunction::ProbabilityOfImprovement { best_f, maximize } => {
                let (u, du_dm, du_ds) = scaled_improvement(mean, sigma, best_f, maximize);
                let p = phi(u);
                (ndtr(u), p * du_dm, p * du_ds)
            }
            KgValueFunction::LogProbabilityOfImprovement { best_f, maximize } => {
                let (u, du_dm, du_ds) = scaled_improvement(mean, sigma, best_f, maximize);
                let g = log_ndtr_grad(u);
                (log_ndtr(u), g * du_dm, g * du_ds)
            }
            KgValueFunction::ScalarizedPosteriorMean { .. } | KgValueFunction::MonteCarlo(_) => {
                unreachable!("not a single-point analytic value function")
            }
        };
        // The improvement-based values depend on `mean - best_f`.
        let best_f_bar = match self.kind {
            KgValueFunction::ExpectedImprovement { .. }
            | KgValueFunction::LogExpectedImprovement { .. }
            | KgValueFunction::ProbabilityOfImprovement { .. }
            | KgValueFunction::LogProbabilityOfImprovement { .. } => -dm,
            _ => 0.0,
        };
        PosteriorValue {
            value,
            mean_bar: Vector::from_element(1, dm),
            cov_bar: Some(Matrix::from_element(1, 1, ds * ds_dv)),
            x_bar: None,
            best_f_bar,
        }
    }
}

fn sgn(maximize: bool) -> f64 {
    if maximize {
        1.0
    } else {
        -1.0
    }
}

/// Options of [`QKnowledgeGradient`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QKnowledgeGradientOptions {
    /// The number of fantasies `n_f` (BoTorch's default 64); `None` takes it
    /// from the sampler.
    pub num_fantasies: Option<usize>,
    /// The sampler of the fantasy observations (default: `n_f` Sobol samples
    /// with a random seed). Its base samples drive the fantasies of every
    /// output (BoTorch's `fantasize`: a multi-output GP's fantasies use base
    /// samples for all of its outputs, a `ModelListGP`'s sub-models the same
    /// base samples).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// The MC objective of the value function: `None` uses the analytic
    /// posterior mean, an objective the simple regret of posterior samples
    /// (BoTorch's `qSimpleRegret` with the inner sampler).
    pub objective: Option<Arc<dyn McObjective>>,
    /// A posterior transform applied before the value function (required for
    /// multi-output models without an objective).
    pub posterior_transform: Option<ScalarizedPosteriorTransform>,
    /// The sampler of the value function (default with an objective: 128
    /// Sobol samples with a random seed).
    pub inner_sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// Pending points (`n' × d`), fantasized together with the candidates.
    pub x_pending: Option<Matrix>,
    /// The current value, subtracted from the value of every fantasy.
    pub current_value: Option<f64>,
}

impl Default for QKnowledgeGradientOptions {
    fn default() -> Self {
        Self {
            num_fantasies: Some(DEFAULT_NUM_FANTASIES),
            sampler: None,
            objective: None,
            posterior_transform: None,
            inner_sampler: None,
            x_pending: None,
            current_value: None,
        }
    }
}

/// Options of [`QMultiFidelityKnowledgeGradient`] (BoTorch's keyword
/// arguments).
#[derive(Clone, Debug, Default)]
pub struct QMultiFidelityKnowledgeGradientOptions {
    /// The options shared with [`QKnowledgeGradient`].
    pub kg: QKnowledgeGradientOptions,
    /// A cost-aware utility of the improvements and the candidates (requires
    /// `current_value`).
    pub cost_aware_utility: Option<Arc<dyn CostAwareUtility>>,
    /// The projection of the fantasy points before the value function is
    /// evaluated (BoTorch's `project`; default: the identity), e.g. a
    /// [`crate::acquisition::fidelity::TargetFidelityProjection`].
    pub project: Option<Arc<dyn InputProjection>>,
    /// The expansion of the candidates (and pending points) before they are
    /// fantasized (BoTorch's `expand`; default: the identity), e.g. a
    /// [`crate::acquisition::fidelity::TraceObservationExpansion`].
    pub expand: Option<Arc<dyn InputProjection>>,
    /// The value function (BoTorch's `valfunc_cls` with the arguments of
    /// `valfunc_argfac`); `None` is BoTorch's default (the posterior mean or,
    /// with an objective, `qSimpleRegret`).
    pub value_function: Option<KgValueFunction>,
}

/// The value of one fantasy and, with gradients, its derivatives with
/// respect to the fantasy point(s) (`k × d`) and the fantasy models'
/// training data.
struct FantasyValue {
    value: f64,
    grad: Option<(Matrix, Vec<FantasyAdjoint>)>,
}

/// The multi-fidelity configuration of a [`QKnowledgeGradient`].
#[derive(Clone, Debug, Default)]
struct MultiFidelity {
    cost_aware_utility: Option<Arc<dyn CostAwareUtility>>,
    project: Option<Arc<dyn InputProjection>>,
    expand: Option<Arc<dyn InputProjection>>,
}

/// The mean utility of the fantasies and, with gradients, the adjoints of
/// the fantasy values and of the candidates.
type ReducedValue = (f64, Option<(Vec<f64>, Option<Matrix>)>);

/// BoTorch's `qKnowledgeGradient`: see the [module documentation](self).
#[derive(Debug)]
pub struct QKnowledgeGradient {
    model: Arc<dyn Model>,
    num_fantasies: usize,
    sampler: Arc<SobolQMCNormalSampler>,
    objective: Option<Arc<dyn McObjective>>,
    posterior_transform: Option<ScalarizedPosteriorTransform>,
    inner_sampler: Option<Arc<SobolQMCNormalSampler>>,
    x_pending: RwLock<Option<Matrix>>,
    current_value: Option<f64>,
    value: ValueFunctionCore,
    mf: MultiFidelity,
}

/// The evaluation of one t-batch.
struct Evaluation {
    value: f64,
    grad: Option<Matrix>,
}

/// The independent single-output GPs of the outputs of `model` and the
/// groups of outputs that are fantasized jointly.
fn output_gps(model: &dyn Model) -> Result<(Vec<&SingleTaskGP>, Vec<usize>)> {
    let err = || {
        BotorchError::Unsupported(
            "the knowledge gradient requires a model of exact GPs that supports fantasies \
             (SingleTaskGP or ModelListGP)"
                .into(),
        )
    };
    let gps = model.output_gps().ok_or_else(err)?;
    let groups = model.fantasy_output_groups().ok_or_else(err)?;
    if groups.iter().sum::<usize>() != gps.len() {
        return Err(err());
    }
    Ok((gps, groups))
}

impl QKnowledgeGradient {
    /// `qKnowledgeGradient(model, num_fantasies, sampler, objective,
    /// posterior_transform, inner_sampler, X_pending, current_value)` for a
    /// model of exact GPs (a single- or multi-output `SingleTaskGP` or a
    /// `ModelListGP`).
    ///
    /// # Errors
    /// * [`BotorchError::InvalidArgument`] without `num_fantasies` and
    ///   sampler, or for a sampler whose number of samples differs from
    ///   `num_fantasies` (BoTorch's messages);
    /// * [`BotorchError::Unsupported`] for other or batched models and for
    ///   multi-output models without an objective or a posterior transform
    ///   (BoTorch's message);
    /// * [`BotorchError::DimensionMismatch`] for pending points or a
    ///   transform of the wrong dimension.
    pub fn new(model: Arc<dyn Model>, options: QKnowledgeGradientOptions) -> Result<Self> {
        Self::build(model, options, MultiFidelity::default(), None)
    }

    fn build(
        model: Arc<dyn Model>,
        options: QKnowledgeGradientOptions,
        mf: MultiFidelity,
        value_function: Option<KgValueFunction>,
    ) -> Result<Self> {
        let o = options;
        let (sampler, num_fantasies) = match (o.sampler, o.num_fantasies) {
            (None, None) => {
                return Err(BotorchError::InvalidArgument(
                    "Must specify `num_fantasies` if no `sampler` is provided.".into(),
                ))
            }
            (None, Some(n)) => (Arc::new(SobolQMCNormalSampler::new(n, None)?), n),
            (Some(s), Some(n)) => {
                if s.num_samples() != n {
                    return Err(BotorchError::InvalidArgument(format!(
                        "The sampler shape must match num_fantasies={n}."
                    )));
                }
                (s, n)
            }
            (Some(s), None) => {
                let n = s.num_samples();
                (s, n)
            }
        };
        check_unbatched(model.as_ref())?;
        let inner_sampler = match (&o.objective, o.inner_sampler) {
            (Some(_), None) => Some(Arc::new(SobolQMCNormalSampler::new(
                DEFAULT_INNER_SAMPLES,
                None,
            )?)),
            (_, s) => s,
        };
        if o.objective.is_none() && model.num_outputs() != 1 && o.posterior_transform.is_none() {
            return Err(BotorchError::Unsupported(MULTI_OUTPUT_MESSAGE.into()));
        }
        if let Some(t) = &o.posterior_transform {
            if t.num_outputs() != model.num_outputs() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Output shape not equal to that of weights. Output shape is {} and weights \
                     are [{}]",
                    model.num_outputs(),
                    t.num_outputs()
                )));
            }
        }
        if let Some(p) = &o.x_pending {
            check_pending(p, model.dim())?;
        }
        output_gps(model.as_ref())?;
        let kind = match value_function {
            Some(v) => v,
            None if o.objective.is_some() => {
                KgValueFunction::MonteCarlo(SampleUtility::SimpleRegret)
            }
            None => KgValueFunction::PosteriorMean { maximize: true },
        };
        let value_sampler = match (&kind, &inner_sampler) {
            (KgValueFunction::MonteCarlo(_), None) => Some(Arc::new(SobolQMCNormalSampler::new(
                DEFAULT_MC_SAMPLES,
                None,
            )?)),
            (_, s) => s.clone(),
        };
        let value = ValueFunctionCore {
            objective: if matches!(kind, KgValueFunction::MonteCarlo(_)) {
                o.objective.clone()
            } else {
                None
            },
            kind,
            sampler: value_sampler,
        };
        if value.kind.is_analytic() && model.num_outputs() != 1 && o.posterior_transform.is_none() {
            return Err(BotorchError::Unsupported(
                "Must specify a posterior transform when using a multi-output model.".into(),
            ));
        }
        Ok(Self {
            model,
            num_fantasies,
            sampler,
            objective: o.objective,
            posterior_transform: o.posterior_transform,
            inner_sampler,
            x_pending: RwLock::new(o.x_pending),
            current_value: o.current_value,
            value,
            mf,
        })
    }

    /// The number of fantasies `n_f`.
    pub fn num_fantasies(&self) -> usize {
        self.num_fantasies
    }

    /// The sampler of the fantasy observations.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.sampler
    }

    /// The sampler of the value function, if any.
    pub fn inner_sampler(&self) -> Option<&Arc<SobolQMCNormalSampler>> {
        self.inner_sampler.as_ref()
    }

    /// The MC objective, if given.
    pub fn objective(&self) -> Option<&Arc<dyn McObjective>> {
        self.objective.as_ref()
    }

    /// The posterior transform, if given.
    pub fn posterior_transform(&self) -> Option<&ScalarizedPosteriorTransform> {
        self.posterior_transform.as_ref()
    }

    /// The current value, if given.
    pub fn current_value(&self) -> Option<f64> {
        self.current_value
    }

    /// The value function evaluated on the fantasy models.
    pub fn value_function_kind(&self) -> &KgValueFunction {
        &self.value.kind
    }

    /// The projection of the fantasy points (qMFKG), if any.
    pub fn project(&self) -> Option<&Arc<dyn InputProjection>> {
        self.mf.project.as_ref()
    }

    /// The cost-aware utility (qMFKG), if any.
    pub fn cost_aware_utility(&self) -> Option<&Arc<dyn CostAwareUtility>> {
        self.mf.cost_aware_utility.as_ref()
    }

    /// The candidates (`q × d`) and the fantasy points (`n_f × d`) of an
    /// augmented candidate set (BoTorch's `_split_fantasy_points` for one
    /// t-batch).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `x` has fewer than `n_f` points
    /// (BoTorch's message).
    pub fn split_fantasy_points(&self, x: &Matrix) -> Result<(Matrix, Matrix)> {
        split_fantasy_points(x, self.num_fantasies)
    }

    /// The value function of the *current* model as an acquisition function
    /// (BoTorch's `_get_value_function(model, objective, posterior_transform,
    /// sampler=inner_sampler, project)` as used by the one-shot initial
    /// conditions: the posterior mean or `qSimpleRegret`, at projected points
    /// for qMFKG; `valfunc_cls` is not used there, as in BoTorch).
    ///
    /// # Errors
    /// Errors of the constructors.
    pub fn current_value_function(&self) -> Result<Arc<dyn AcquisitionFunction>> {
        let kind = if self.objective.is_some() {
            KgValueFunction::MonteCarlo(SampleUtility::SimpleRegret)
        } else {
            KgValueFunction::PosteriorMean { maximize: true }
        };
        let sampler = match (&kind, &self.inner_sampler) {
            (KgValueFunction::MonteCarlo(_), None) => Some(Arc::new(SobolQMCNormalSampler::new(
                DEFAULT_MC_SAMPLES,
                None,
            )?)),
            (_, s) => s.clone(),
        };
        let vf = ModelValueFunction::new(
            Arc::clone(&self.model),
            ValueFunctionCore {
                objective: self.objective.clone(),
                kind,
                sampler,
            },
            self.posterior_transform.clone(),
        )?;
        let vf: Arc<dyn AcquisitionFunction> = Arc::new(vf);
        Ok(match &self.mf.project {
            Some(p) => Arc::new(ProjectedAcquisitionFunction::new(vf, Arc::clone(p))),
            None => vf,
        })
    }

    /// The fantasy GPs of every output at the points `x_new` (`k × d`, the
    /// expanded candidates and pending points).
    fn fantasy_gps<'a>(
        &self,
        gps: &[&'a SingleTaskGP],
        groups: &[usize],
        x_new: &Matrix,
    ) -> Result<Vec<FantasyGp<'a>>> {
        let k = x_new.nrows();
        let n_f = self.num_fantasies;
        let mut out = Vec::with_capacity(gps.len());
        let mut offset = 0;
        for &mg in groups {
            let z = if k > 0 {
                Some(self.sampler.base_samples(k * mg)?)
            } else {
                None
            };
            for j in 0..mg {
                let base = match &z {
                    Some(z) if mg == 1 => Arc::clone(z),
                    Some(z) => Arc::new(Matrix::from_fn(n_f, k, |i, p| z[(i, p * mg + j)])),
                    None => Arc::new(Matrix::zeros(n_f, 0)),
                };
                out.push(FantasyGp::new(gps[offset + j], x_new, base, n_f)?);
            }
            offset += mg;
        }
        Ok(out)
    }

    /// The value of fantasy `i` at the point(s) `x` (before the projection).
    fn fantasy_value(
        &self,
        fantasies: &[FantasyGp<'_>],
        i: usize,
        x: &Matrix,
        with_grad: bool,
    ) -> Result<FantasyValue> {
        let m = fantasies.len();
        let xp = match &self.mf.project {
            Some(p) => p.project(x)?,
            None => x.clone(),
        };
        let n = xp.nrows();
        let with_cov = self.value.kind.uses_covariance();
        let posts: Vec<FantasyPosterior> = fantasies
            .iter()
            .map(|g| g.posterior(i, &xp, with_cov))
            .collect::<Result<_>>()?;
        let post = assemble_posterior(&posts, n);
        let (post_t, pv) = match &self.posterior_transform {
            Some(t) => {
                let pt = t.transform_posterior(&post)?;
                let pv = self.value.evaluate(&pt, &xp, with_grad)?;
                (Some(t), pv)
            }
            None => (None, self.value.evaluate(&post, &xp, with_grad)?),
        };
        if !with_grad {
            return Ok(FantasyValue {
                value: pv.value,
                grad: None,
            });
        }
        let (mean_bar, cov_bar) = match post_t {
            Some(t) => {
                let cb = pv.cov_bar.clone().unwrap_or_else(|| Matrix::zeros(n, n));
                let (mb, cbf) = t.pull_back(&pv.mean_bar, &cb);
                (mb, pv.cov_bar.is_some().then_some(cbf))
            }
            None => (pv.mean_bar, pv.cov_bar),
        };
        let mut gxp = pv.x_bar.unwrap_or_else(|| Matrix::zeros(n, xp.ncols()));
        let mut adjoints = Vec::with_capacity(m);
        for (t, (g, post)) in fantasies.iter().zip(&posts).enumerate() {
            let mb = Vector::from_fn(n, |p, _| mean_bar[t * n + p]);
            let cb = cov_bar
                .as_ref()
                .filter(|_| post.covariance.is_some())
                .map(|c| c.view((t * n, t * n), (n, n)).into_owned());
            let mut acc = g.adjoint();
            gxp += g.posterior_backward(post, &mb, cb.as_ref(), &mut acc);
            adjoints.push(acc);
        }
        let gx = match &self.mf.project {
            Some(p) => p.backward(x, &gxp)?,
            None => gxp,
        };
        Ok(FantasyValue {
            value: pv.value,
            grad: Some((gx, adjoints)),
        })
    }

    /// The candidates and pending points, and their expansion.
    fn conditioning_points(&self, x_actual: &Matrix) -> Result<(Matrix, Matrix)> {
        let d = self.model.dim();
        let x_eval = match read_lock(&self.x_pending).as_ref() {
            Some(p) => vstack(&[x_actual, p], d),
            None => x_actual.clone(),
        };
        let x_new = match &self.mf.expand {
            Some(e) => e.project(&x_eval)?,
            None => x_eval.clone(),
        };
        if x_new.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "`expand` must return points with {d} columns, got {}",
                x_new.ncols()
            )));
        }
        Ok((x_eval, x_new))
    }

    /// The deltas (values minus the current value) of the fantasies, the
    /// utilities, their mean and the adjoints of the deltas and candidates.
    fn reduce(&self, x_actual: &Matrix, raw: &[f64], with_grad: bool) -> Result<ReducedValue> {
        let deltas: Vec<f64> = match self.current_value {
            Some(cv) => raw.iter().map(|&v| v - cv).collect(),
            None => raw.to_vec(),
        };
        let (utilities, pullback) = match &self.mf.cost_aware_utility {
            Some(u) => {
                let out = u.evaluate(x_actual, &deltas, None, with_grad)?;
                (out.values, out.pullback)
            }
            None => (deltas, None),
        };
        let n_f = utilities.len() as f64;
        let value = utilities.iter().sum::<f64>() / n_f;
        if !with_grad {
            return Ok((value, None));
        }
        let weights = vec![1.0 / n_f; utilities.len()];
        Ok(match pullback {
            Some(pb) => {
                let (dbar, xbar) = pb(&weights)?;
                (value, Some((dbar, Some(xbar))))
            }
            None => (value, Some((weights, None))),
        })
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<Evaluation> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let (x_actual, x_fant) = self.split_fantasy_points(x)?;
        let q = x_actual.nrows();
        let n_f = self.num_fantasies;
        let (x_eval, x_new) = self.conditioning_points(&x_actual)?;
        let (gps, groups) = output_gps(self.model.as_ref())?;
        let fantasies = self.fantasy_gps(&gps, &groups, &x_new)?;
        let work = (x_new.nrows() + 1)
            * d
            * gps.len()
            * gps.iter().map(|g| g.num_train()).max().unwrap_or(1);
        let eval = |i: usize| {
            let xi = x_fant.rows(i, 1).into_owned();
            self.fantasy_value(&fantasies, i, &xi, with_grad)
        };
        let values: Vec<FantasyValue> = if work >= PAR_WORK && n_f > 1 {
            (0..n_f).into_par_iter().map(eval).collect::<Result<_>>()?
        } else {
            (0..n_f).map(eval).collect::<Result<_>>()?
        };
        let raw: Vec<f64> = values.iter().map(|v| v.value).collect();
        let (value, bars) = self.reduce(&x_actual, &raw, with_grad)?;
        let Some((deltas_bar, x_actual_bar)) = bars else {
            return Ok(Evaluation { value, grad: None });
        };
        let mut grad = Matrix::zeros(x.nrows(), d);
        if let Some(xb) = x_actual_bar {
            let mut rows = grad.rows_mut(0, q);
            rows += &xb;
        }
        let mut accs: Vec<FantasyAdjoint> = fantasies.iter().map(|f| f.adjoint()).collect();
        for (i, v) in values.iter().enumerate() {
            let scale = deltas_bar[i];
            let (gx, adjoints) = v.grad.as_ref().expect("gradient requested");
            let mut row = grad.rows_mut(q + i, 1);
            row += gx * scale;
            for (acc, a) in accs.iter_mut().zip(adjoints) {
                acc.add_scaled(a, scale);
            }
        }
        let mut new_bar = Matrix::zeros(x_new.nrows(), d);
        for (f, acc) in fantasies.iter().zip(&accs) {
            let g = f.backward(acc)?;
            if g.nrows() == new_bar.nrows() {
                new_bar += g;
            }
        }
        let eval_bar = match &self.mf.expand {
            Some(e) => e.backward(&x_eval, &new_bar)?,
            None => new_bar,
        };
        let mut rows = grad.rows_mut(0, q);
        rows += eval_bar.rows(0, q);
        Ok(Evaluation {
            value,
            grad: Some(grad),
        })
    }

    /// BoTorch's `evaluate(X, bounds, num_restarts, raw_samples, options,
    /// scipy_options)`: the knowledge gradient of the candidates `x` (`q ×
    /// d`, without fantasy points; pending points are appended) with the
    /// inner problems solved: the value function of every fantasy model is
    /// maximized over `bounds` from `num_restarts` initial conditions of
    /// [`crate::optim::one_shot::gen_value_function_initial_conditions`] by
    /// one joint L-BFGS-B problem over all restarts and fantasies (BoTorch's
    /// `gen_candidates_scipy` with `use_parallel_mode=False`, `lbfgsb` being
    /// its `scipy_options`), and the largest value of every fantasy is used.
    /// `rng` replaces torch's global generator.
    ///
    /// # Errors
    /// Shape errors and errors of the optimization.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_with_inner_optimization(
        &self,
        x: &Matrix,
        bounds: &Matrix,
        num_restarts: usize,
        raw_samples: usize,
        options: &crate::optim::one_shot::ValueFunctionInitialConditionOptions,
        lbfgsb: &crate::lbfgsb::LbfgsbOptions,
        rng: &mut dyn Rng,
    ) -> Result<f64> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let (_x_eval, x_new) = self.conditioning_points(x)?;
        let (gps, groups) = output_gps(self.model.as_ref())?;
        let fantasies = self.fantasy_gps(&gps, &groups, &x_new)?;
        let ics = crate::optim::one_shot::gen_value_function_initial_conditions(
            self,
            &fantasies,
            bounds,
            num_restarts,
            raw_samples,
            options,
            rng,
        )?;
        let values = crate::optim::one_shot::maximize_fantasy_values(
            self, &fantasies, &ics, bounds, lbfgsb,
        )?;
        Ok(self.reduce(x, &values, false)?.0)
    }

    /// The value of fantasy `i` at the point(s) `x` (`k × d`) and its
    /// gradient with respect to them (used by the inner optimization).
    pub(crate) fn fantasy_value_and_grad(
        &self,
        fantasies: &[FantasyGp<'_>],
        i: usize,
        x: &Matrix,
        with_grad: bool,
    ) -> Result<(f64, Option<Matrix>)> {
        let v = self.fantasy_value(fantasies, i, x, with_grad)?;
        Ok((v.value, v.grad.map(|(g, _)| g)))
    }
}

/// The joint (block-diagonal) posterior of the independent outputs of a
/// fantasy model from the posteriors of its outputs at `n` points.
fn assemble_posterior(posts: &[FantasyPosterior], n: usize) -> Posterior {
    let m = posts.len();
    let mean = Vector::from_fn(n * m, |r, _| posts[r / n].mean[r % n]);
    let mut cov = Matrix::zeros(n * m, n * m);
    for (t, p) in posts.iter().enumerate() {
        if let Some(c) = &p.covariance {
            cov.view_mut((t * n, t * n), (n, n)).copy_from(c);
        }
    }
    Posterior::multi_output(mean, cov, m)
}

/// BoTorch's `_split_fantasy_points` for one t-batch: the first
/// `q = x.nrows() - n_f` points of `x` are the candidates and the last `n_f`
/// the fantasy points (one per fantasy).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] if `x` has fewer than `n_f` points
/// (BoTorch's message).
pub fn split_fantasy_points(x: &Matrix, n_f: usize) -> Result<(Matrix, Matrix)> {
    if n_f > x.nrows() {
        return Err(BotorchError::InvalidArgument(format!(
            "n_f ({n_f}) must be less than the q-batch dimension of X ({})",
            x.nrows()
        )));
    }
    let q = x.nrows() - n_f;
    Ok((x.rows(0, q).into_owned(), x.rows(q, n_f).into_owned()))
}

impl AcquisitionFunction for QKnowledgeGradient {
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
        read_lock(&self.x_pending).clone()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        if let Some(p) = x_pending {
            check_pending(p, self.model.dim())?;
        }
        *write_lock(&self.x_pending) = x_pending.cloned();
        Ok(())
    }

    fn as_one_shot(&self) -> Option<&dyn OneShotAcquisitionFunction> {
        Some(self)
    }
}

impl OneShotAcquisitionFunction for QKnowledgeGradient {
    fn get_augmented_q_batch_size(&self, q: usize) -> usize {
        q + self.num_fantasies
    }

    fn extract_candidates(&self, x_full: &Matrix) -> Result<Matrix> {
        let n = self.num_fantasies;
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
        crate::optim::one_shot::gen_one_shot_kg_initial_conditions(
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

/// BoTorch's `qMultiFidelityKnowledgeGradient`: the knowledge gradient with
/// a cost-aware utility, the projection of the fantasy points (`project`),
/// the expansion of the candidates (`expand`) and a custom value function
/// (`valfunc_cls` / `valfunc_argfac`); without any of them it equals
/// [`QKnowledgeGradient`].
#[derive(Debug)]
pub struct QMultiFidelityKnowledgeGradient {
    inner: QKnowledgeGradient,
}

impl QMultiFidelityKnowledgeGradient {
    /// `qMultiFidelityKnowledgeGradient(model, num_fantasies, sampler,
    /// objective, posterior_transform, inner_sampler, X_pending,
    /// current_value, cost_aware_utility, project, expand, valfunc_cls,
    /// valfunc_argfac)`.
    ///
    /// # Errors
    /// See [`QKnowledgeGradient::new`]; [`BotorchError::Unsupported`] for a
    /// cost-aware utility without `current_value` (BoTorch's message).
    pub fn new(
        model: Arc<dyn Model>,
        options: QMultiFidelityKnowledgeGradientOptions,
    ) -> Result<Self> {
        if options.kg.current_value.is_none() && options.cost_aware_utility.is_some() {
            return Err(BotorchError::Unsupported(
                "Cost-aware KG requires current_value to be specified.".into(),
            ));
        }
        let mf = MultiFidelity {
            cost_aware_utility: options.cost_aware_utility,
            project: options.project,
            expand: options.expand,
        };
        Ok(Self {
            inner: QKnowledgeGradient::build(model, options.kg, mf, options.value_function)?,
        })
    }

    /// The underlying knowledge gradient (all accessors of
    /// [`QKnowledgeGradient`]).
    pub fn as_kg(&self) -> &QKnowledgeGradient {
        &self.inner
    }

    /// The expansion of the candidates, if any.
    pub fn expand(&self) -> Option<&Arc<dyn InputProjection>> {
        self.inner.mf.expand.as_ref()
    }
}

impl AcquisitionFunction for QMultiFidelityKnowledgeGradient {
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

    fn as_one_shot(&self) -> Option<&dyn OneShotAcquisitionFunction> {
        Some(&self.inner)
    }
}

/// A value function of the current model as an acquisition function (the
/// analytic and sample-based value functions of [`KgValueFunction`] through
/// a posterior transform).
#[derive(Debug)]
pub(crate) struct ModelValueFunction {
    model: Arc<dyn Model>,
    value: ValueFunctionCore,
    transform: Option<ScalarizedPosteriorTransform>,
}

impl ModelValueFunction {
    pub fn new(
        model: Arc<dyn Model>,
        value: ValueFunctionCore,
        transform: Option<ScalarizedPosteriorTransform>,
    ) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        Ok(Self {
            model,
            value,
            transform,
        })
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        check_candidates(x, self.model.dim(), None)?;
        let post = self.model.posterior(x, false)?;
        post.require_gaussian("the value function")?;
        let n = post.num_points();
        let (pv, transform) = match &self.transform {
            Some(t) => (
                self.value
                    .evaluate(&t.transform_posterior(&post)?, x, with_grad)?,
                Some(t),
            ),
            None => (self.value.evaluate(&post, x, with_grad)?, None),
        };
        if !with_grad {
            return Ok((pv.value, None));
        }
        let cb = pv.cov_bar.unwrap_or_else(|| Matrix::zeros(n, n));
        let (mean_bar, cov_bar) = match transform {
            Some(t) => t.pull_back(&pv.mean_bar, &cb),
            None => (pv.mean_bar, cb),
        };
        let mut grad = self
            .model
            .joint_posterior_vjp(x, 0..x.nrows(), &mean_bar, &cov_bar)?;
        if let Some(xb) = pv.x_bar {
            grad += xb;
        }
        Ok((pv.value, Some(grad)))
    }
}

impl AcquisitionFunction for ModelValueFunction {
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

    fn is_log(&self) -> bool {
        matches!(
            self.value.kind,
            KgValueFunction::LogExpectedImprovement { .. }
                | KgValueFunction::LogProbabilityOfImprovement { .. }
        )
    }
}

/// BoTorch's `ProjectedAcquisitionFunction(base_value_function, project)`:
/// the wrapped acquisition function evaluated at `project(X)`, with the
/// gradient pulled back through the projection.
#[derive(Debug)]
pub struct ProjectedAcquisitionFunction {
    base: Arc<dyn AcquisitionFunction>,
    project: Arc<dyn InputProjection>,
}

impl ProjectedAcquisitionFunction {
    /// Wraps `base` with the projection `project`.
    pub fn new(base: Arc<dyn AcquisitionFunction>, project: Arc<dyn InputProjection>) -> Self {
        Self { base, project }
    }

    /// The wrapped acquisition function (BoTorch's `base_value_function`).
    pub fn base_value_function(&self) -> &Arc<dyn AcquisitionFunction> {
        &self.base
    }

    /// The projection.
    pub fn project(&self) -> &Arc<dyn InputProjection> {
        &self.project
    }
}

impl AcquisitionFunction for ProjectedAcquisitionFunction {
    fn model(&self) -> &Arc<dyn Model> {
        self.base.model()
    }

    fn dim(&self) -> usize {
        self.base.dim()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        check_candidates(x, self.dim(), None)?;
        self.base.forward(&self.project.project(x)?)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        check_candidates(x, self.dim(), None)?;
        let xp = self.project.project(x)?;
        let (v, g) = self.base.forward_with_grad(&xp)?;
        Ok((v, self.project.backward(x, &g)?))
    }

    fn is_log(&self) -> bool {
        self.base.is_log()
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.base.set_x_pending(x_pending)
    }

    fn x_pending(&self) -> Option<Matrix> {
        self.base.x_pending()
    }
}
