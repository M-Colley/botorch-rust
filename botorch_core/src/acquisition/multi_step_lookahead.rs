//! Multi-step look-ahead with one-shot optimization
//! (`botorch.acquisition.multi_step_lookahead`; Jiang et al., *Efficient
//! Nonmyopic Bayesian Optimization via One-Shot Multi-Step Trees*, NeurIPS
//! 2020).
//!
//! [`QMultiStepLookahead`] performs a `k`-step look-ahead by repeated
//! fantasizing: the `q` candidates `X_0` (with the pending points) are
//! fantasized with the base samples of the first sampler (`f_1` fantasies),
//! every fantasy model is fantasized at its own `q_1` points `X_1` with the
//! second sampler (`f_2` fantasies each), and so on, which gives a tree of
//! fantasy models with `f_1 ⋯ f_j` nodes at depth `j`. Every node at depth
//! `j` evaluates the stage value function of depth `j` (BoTorch's
//! `valfunc_cls[j]`, if any) at its points; the value is the sum over the
//! stages of the averages of the stage values over the nodes (weighted by
//! the samplers' `base_weights`, if any). The default, a posterior mean at
//! the leaves only, is the multi-step knowledge gradient.
//!
//! A candidate set holds the `q` candidates followed by the points of all
//! nodes: for every depth `j = 1..k`, `f_1 ⋯ f_j` blocks of `q_j` points,
//! the block of the node with fantasy indices `(i_1, ..., i_j)` at position
//! `i_1 + f_1 i_2 + f_1 f_2 i_3 + ...` (BoTorch's
//! `get_multi_step_tree_input_representation`).
//!
//! The fantasy models are exact GPs conditioned on the fantasized
//! observations ([`crate::models::gp`]), with the noise of BoTorch's
//! `fantasize` (the inferred noise level, or the mean training noise of a
//! fixed-noise model). Values and exact gradients with respect to every
//! point of the tree are computed through all levels of conditioning.
//!
//! **Differences to BoTorch:**
//!
//! * **Gradients.** BoTorch evaluates the value functions of the fantasy
//!   models with detached test caches (`propagate_grads` is only set for
//!   the posteriors of the fantasized observations) and, at depth two and
//!   more, under `fast_pred_var`, so its gradients with respect to the
//!   candidates and the points of inner nodes miss their dependence through
//!   the fantasy models. The gradients here are exact (BoTorch under
//!   `propagate_grads(True)` without `fast_pred_var` agrees with them).
//! * **Base samples without `collapse_fantasy_base_samples`** are
//!   independent between the nodes of a depth and shared by the t-batches
//!   (BoTorch's `batch_range_override` does this for multi-output models;
//!   for single-output posteriors it collapses the fantasy dimensions
//!   instead and draws different base samples for every t-batch).
//! * **The points of a node with `q_j > 1`** are rows `ψ q_j .. (ψ + 1)
//!   q_j` of its depth's block of every candidate set (BoTorch's reshape
//!   mixes the blocks of different t-batches when there are several).
//! * **`best_f` from the model** (BoTorch's `make_best_f` as
//!   `valfunc_argfacs`) is the best observed or fantasized value of every
//!   node in the outcome space of its posterior (BoTorch takes the maximum
//!   of the outcome-transformed training targets, and fails for analytic
//!   value functions, whose `best_f` it broadcasts against the outputs).

use std::sync::{Arc, RwLock};

use rand::Rng;
use rayon::prelude::*;

use crate::acquisition::knowledge_gradient::{KgValueFunction, ValueFunctionCore};
use crate::acquisition::mc_objective::McObjective;
use crate::acquisition::monte_carlo::{read_lock, vstack, write_lock, SampleUtility};
use crate::acquisition::objective::ScalarizedPosteriorTransform;
use crate::acquisition::one_shot::OneShotAcquisitionFunction;
use crate::acquisition::{check_candidates, check_pending, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::linalg::CovarianceRoot;
use crate::models::gp::{CondAdjoint, CondFactor, CondGp, CondPosterior};
use crate::models::{Model, SingleTaskGP};
use crate::optim::OptimizeAcqfOptions;
use crate::posterior::Posterior;
use crate::sampling::SobolQMCNormalSampler;
use crate::{Matrix, Vector};

/// BoTorch's default number of samples of the inner samplers of MC value
/// functions.
pub const DEFAULT_INNER_MC_SAMPLES: usize = 32;

/// Below this number of nodes at a depth, they are evaluated sequentially.
const PAR_NODES: usize = 4;

/// A stage value function of [`QMultiStepLookahead`] (BoTorch's
/// `valfunc_cls[j]` with the arguments of `valfunc_argfacs[j]`).
#[derive(Clone, Debug)]
pub struct StageValueFunction {
    /// The value function (one of the native classes of the knowledge
    /// gradient's `valfunc_cls`).
    pub kind: KgValueFunction,
    /// Whether `best_f` is the best observed or fantasized value of every
    /// node (BoTorch's `make_best_f`); the `best_f` of `kind` is then
    /// ignored. Requires a single-output model.
    pub best_f_from_model: bool,
    /// The sampler of an MC value function (default:
    /// [`DEFAULT_INNER_MC_SAMPLES`] Sobol samples with a random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
}

impl StageValueFunction {
    /// A value function with its argument `best_f` (if any) fixed.
    pub fn new(kind: KgValueFunction) -> Self {
        Self {
            kind,
            best_f_from_model: false,
            sampler: None,
        }
    }

    /// Whether the value function has a `best_f`.
    fn has_best_f(&self) -> bool {
        matches!(
            self.kind,
            KgValueFunction::ExpectedImprovement { .. }
                | KgValueFunction::LogExpectedImprovement { .. }
                | KgValueFunction::ProbabilityOfImprovement { .. }
                | KgValueFunction::LogProbabilityOfImprovement { .. }
                | KgValueFunction::MonteCarlo(SampleUtility::ExpectedImprovement { .. })
                | KgValueFunction::MonteCarlo(SampleUtility::ProbabilityOfImprovement { .. })
        )
    }

    /// The value function with `best_f` replaced by `b`.
    fn with_best_f(&self, b: f64) -> KgValueFunction {
        match &self.kind {
            KgValueFunction::ExpectedImprovement { maximize, .. } => {
                KgValueFunction::ExpectedImprovement {
                    best_f: b,
                    maximize: *maximize,
                }
            }
            KgValueFunction::LogExpectedImprovement { maximize, .. } => {
                KgValueFunction::LogExpectedImprovement {
                    best_f: b,
                    maximize: *maximize,
                }
            }
            KgValueFunction::ProbabilityOfImprovement { maximize, .. } => {
                KgValueFunction::ProbabilityOfImprovement {
                    best_f: b,
                    maximize: *maximize,
                }
            }
            KgValueFunction::LogProbabilityOfImprovement { maximize, .. } => {
                KgValueFunction::LogProbabilityOfImprovement {
                    best_f: b,
                    maximize: *maximize,
                }
            }
            KgValueFunction::MonteCarlo(SampleUtility::ExpectedImprovement { .. }) => {
                KgValueFunction::MonteCarlo(SampleUtility::ExpectedImprovement { best_f: b })
            }
            KgValueFunction::MonteCarlo(SampleUtility::ProbabilityOfImprovement {
                tau, ..
            }) => KgValueFunction::MonteCarlo(SampleUtility::ProbabilityOfImprovement {
                best_f: b,
                tau: *tau,
            }),
            other => other.clone(),
        }
    }
}

/// Options of [`QMultiStepLookahead`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct QMultiStepLookaheadOptions {
    /// The batch sizes `[q_1, ..., q_k]` of the look-ahead steps.
    pub batch_sizes: Vec<usize>,
    /// The numbers of fantasies `[f_1, ..., f_k]` (exactly one of
    /// `num_fantasies` and `samplers` must be given).
    pub num_fantasies: Option<Vec<usize>>,
    /// The samplers of the fantasies of every step.
    pub samplers: Option<Vec<Arc<SobolQMCNormalSampler>>>,
    /// Weights of the fantasies of every step (the samplers' `base_weights`,
    /// normalized to sum to one; `None`: uniform).
    pub sample_weights: Vec<Option<Vec<f64>>>,
    /// The `k + 1` stage value functions (`None`: no stage value); default:
    /// the posterior mean at the leaves only (the multi-step knowledge
    /// gradient).
    pub value_functions: Option<Vec<Option<StageValueFunction>>>,
    /// The MC objective of the MC value functions.
    pub objective: Option<Arc<dyn McObjective>>,
    /// A posterior transform applied before the value functions.
    pub posterior_transform: Option<ScalarizedPosteriorTransform>,
    /// Pending points (`m × d`), fantasized with the candidates.
    pub x_pending: Option<Matrix>,
    /// Whether the base samples are shared by all nodes of a depth (BoTorch's
    /// `collapse_fantasy_base_samples`, default `true`).
    pub collapse_fantasy_base_samples: bool,
}

impl Default for QMultiStepLookaheadOptions {
    fn default() -> Self {
        Self {
            batch_sizes: Vec::new(),
            num_fantasies: None,
            samplers: None,
            sample_weights: Vec::new(),
            value_functions: None,
            objective: None,
            posterior_transform: None,
            x_pending: None,
            collapse_fantasy_base_samples: true,
        }
    }
}

/// BoTorch's `qMultiStepLookahead`: see the [module documentation](self).
#[derive(Debug)]
pub struct QMultiStepLookahead {
    model: Arc<dyn Model>,
    batch_sizes: Vec<usize>,
    num_fantasies: Vec<usize>,
    samplers: Vec<Arc<SobolQMCNormalSampler>>,
    /// Normalized fantasy weights of every step.
    weights: Vec<Vec<f64>>,
    stages: Vec<Option<ValueFunctionCore>>,
    best_f_from_model: Vec<bool>,
    posterior_transform: Option<ScalarizedPosteriorTransform>,
    x_pending: RwLock<Option<Matrix>>,
    collapse: bool,
}

/// The independent single-output GPs of the outputs of `model` and the
/// groups of outputs that are fantasized jointly.
fn output_gps(model: &dyn Model) -> Result<(Vec<&SingleTaskGP>, Vec<usize>)> {
    let err = || {
        BotorchError::Unsupported(
            "qMultiStepLookahead requires a model of exact GPs that supports fantasies \
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

/// The evaluation of a stage value function at a node: its value and the
/// adjoints of the posteriors of every output.
struct StageEval {
    value: f64,
    mean_bars: Vec<Vector>,
    cov_bars: Vec<Option<Matrix>>,
    x_bar: Option<Matrix>,
    /// With `best_f` from the model: the index of the best target and the
    /// adjoint of `best_f`.
    best_f: Option<(usize, f64)>,
}

/// The fantasies of a node: per output, the root of the posterior
/// covariance with observation noise and the base samples of the
/// fantasies (`f × q'`).
struct FantasyEval {
    roots: Vec<CovarianceRoot>,
    base: Vec<Matrix>,
}

/// A node of the fantasy tree of one candidate set.
struct Node<'a> {
    /// The model (one conditioned GP per output).
    gps: Vec<CondGp<'a>>,
    /// The weight of the node (the product of the fantasy weights along its
    /// path).
    weight: f64,
    /// The node's points (`q' × d`).
    x: Matrix,
    /// The rows of the candidate set of the node's points (`None`: pending
    /// points).
    x_rows: Vec<Option<usize>>,
    /// The rows of the candidate set of the appended points of the model.
    appended_rows: Arc<Vec<Option<usize>>>,
    /// The posterior of every output at the node's points.
    posts: Vec<CondPosterior>,
    stage: Option<StageEval>,
    fantasy: Option<FantasyEval>,
}

/// What the reverse pass of a node passes on: its contributions to the
/// gradient (rows of the candidate set) and, per output, the adjoints of the
/// targets appended by its ancestors (`older`) and of its own fantasized
/// targets (`own`, the last block).
struct NodeBackward {
    rows: Vec<(usize, Vector)>,
    older: Vec<Vector>,
    own: Vec<Vector>,
}

impl QMultiStepLookahead {
    /// `qMultiStepLookahead(model, batch_sizes, num_fantasies, samplers,
    /// valfunc_cls, valfunc_argfacs, objective, posterior_transform,
    /// inner_mc_samples, X_pending, collapse_fantasy_base_samples)` for a
    /// model of exact GPs (a single- or multi-output `SingleTaskGP` or a
    /// `ModelListGP`).
    ///
    /// # Errors
    /// * [`BotorchError::Unsupported`] unless exactly one of `num_fantasies`
    ///   and `samplers` is given (BoTorch's message), for other or batched
    ///   models, for multi-output models without a posterior transform or
    ///   objective, for a missing terminal value function and for `best_f`
    ///   from a multi-output model;
    /// * [`BotorchError::InvalidArgument`] for inconsistent lengths, empty
    ///   steps or invalid weights;
    /// * [`BotorchError::DimensionMismatch`] for pending points or a
    ///   transform of the wrong dimension.
    pub fn new(model: Arc<dyn Model>, options: QMultiStepLookaheadOptions) -> Result<Self> {
        let o = options;
        check_unbatched(model.as_ref())?;
        let k = o.batch_sizes.len();
        let samplers =
            match (o.num_fantasies, o.samplers) {
                (Some(nf), None) => nf
                    .iter()
                    .map(|&f| SobolQMCNormalSampler::new(f, None).map(Arc::new))
                    .collect::<Result<Vec<_>>>()?,
                (None, Some(s)) => s,
                _ => return Err(BotorchError::Unsupported(
                    "qMultiStepLookahead requires exactly one of `num_fantasies` or `samplers` \
                     as arguments."
                        .into(),
                )),
            };
        if samplers.len() != k {
            return Err(BotorchError::InvalidArgument(format!(
                "expected {k} samplers or numbers of fantasies (one per look-ahead step), got {}",
                samplers.len()
            )));
        }
        if o.batch_sizes.contains(&0) {
            return Err(BotorchError::InvalidArgument(
                "the batch sizes of the look-ahead steps must be positive".into(),
            ));
        }
        let num_fantasies: Vec<usize> = samplers.iter().map(|s| s.num_samples()).collect();
        let mut weights = Vec::with_capacity(k);
        for j in 0..k {
            let f = num_fantasies[j];
            let w = match o.sample_weights.get(j).cloned().flatten() {
                None => vec![1.0 / f as f64; f],
                Some(w) => {
                    let total: f64 = w.iter().sum();
                    if w.len() != f || !(total.is_finite() && total != 0.0) {
                        return Err(BotorchError::InvalidArgument(format!(
                            "the base weights of sampler {j} must be {f} values with a finite, \
                             non-zero sum"
                        )));
                    }
                    w.iter().map(|v| v / total).collect()
                }
            };
            weights.push(w);
        }
        let value_functions = match o.value_functions {
            Some(v) => v,
            None => {
                let mut v = vec![None; k];
                v.push(Some(StageValueFunction::new(
                    KgValueFunction::PosteriorMean { maximize: true },
                )));
                v
            }
        };
        if value_functions.len() != k + 1 {
            return Err(BotorchError::InvalidArgument(format!(
                "expected {} value functions (one per look-ahead step and a terminal one), got {}",
                k + 1,
                value_functions.len()
            )));
        }
        if value_functions[k].is_none() {
            return Err(BotorchError::Unsupported(
                "The terminal value function (the last element of `valfunc_cls`) must not be \
                 None."
                    .into(),
            ));
        }
        let m = model.num_outputs();
        if m != 1 && o.posterior_transform.is_none() {
            let analytic = value_functions
                .iter()
                .flatten()
                .any(|v| !matches!(v.kind, KgValueFunction::MonteCarlo(_)));
            if o.objective.is_none() || analytic {
                return Err(BotorchError::Unsupported(
                    "Must specify an objective or a posterior transform when using a \
                     multi-output model."
                        .into(),
                ));
            }
        }
        if let Some(t) = &o.posterior_transform {
            if t.num_outputs() != m {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Output shape not equal to that of weights. Output shape is {m} and weights \
                     are [{}]",
                    t.num_outputs()
                )));
            }
        }
        if let Some(p) = &o.x_pending {
            check_pending(p, model.dim())?;
        }
        output_gps(model.as_ref())?;
        let mut stages = Vec::with_capacity(k + 1);
        let mut best_f_from_model = Vec::with_capacity(k + 1);
        for v in value_functions {
            let Some(v) = v else {
                stages.push(None);
                best_f_from_model.push(false);
                continue;
            };
            if v.best_f_from_model && !v.has_best_f() {
                return Err(BotorchError::InvalidArgument(format!(
                    "{} has no `best_f`",
                    v.kind.name()
                )));
            }
            if v.best_f_from_model && m != 1 {
                return Err(BotorchError::Unsupported(
                    "`best_f` from the model (make_best_f) requires a single-output model".into(),
                ));
            }
            let mc = matches!(v.kind, KgValueFunction::MonteCarlo(_));
            let sampler = match (mc, v.sampler) {
                (true, None) => Some(Arc::new(SobolQMCNormalSampler::new(
                    DEFAULT_INNER_MC_SAMPLES,
                    None,
                )?)),
                (true, s) => s,
                (false, _) => None,
            };
            best_f_from_model.push(v.best_f_from_model);
            stages.push(Some(ValueFunctionCore {
                objective: if mc { o.objective.clone() } else { None },
                kind: v.kind,
                sampler,
            }));
        }
        Ok(Self {
            model,
            batch_sizes: o.batch_sizes,
            num_fantasies,
            samplers,
            weights,
            stages,
            best_f_from_model,
            posterior_transform: o.posterior_transform,
            x_pending: RwLock::new(o.x_pending),
            collapse: o.collapse_fantasy_base_samples,
        })
    }

    /// The batch sizes `[q_1, ..., q_k]` of the look-ahead steps.
    pub fn batch_sizes(&self) -> &[usize] {
        &self.batch_sizes
    }

    /// The numbers of fantasies `[f_1, ..., f_k]`.
    pub fn num_fantasies(&self) -> &[usize] {
        &self.num_fantasies
    }

    /// The samplers of the fantasies of every step.
    pub fn samplers(&self) -> &[Arc<SobolQMCNormalSampler>] {
        &self.samplers
    }

    /// The samplers of the MC stage value functions (`None` for analytic
    /// and missing ones).
    pub fn inner_samplers(&self) -> Vec<Option<Arc<SobolQMCNormalSampler>>> {
        self.stages
            .iter()
            .map(|s| s.as_ref().and_then(|s| s.sampler.clone()))
            .collect()
    }

    /// The number of auxiliary points of a candidate set, `Σ_j q_j f_1 ⋯
    /// f_j` (BoTorch's `_num_auxiliary`).
    pub fn num_auxiliary(&self) -> usize {
        let mut p = 1;
        let mut total = 0;
        for (q, f) in self.batch_sizes.iter().zip(&self.num_fantasies) {
            p *= f;
            total += q * p;
        }
        total
    }

    /// The number of points of every node at depth `j` (the candidates and
    /// pending points at depth 0).
    fn points_at(&self, j: usize, q0: usize) -> usize {
        if j == 0 {
            q0
        } else {
            self.batch_sizes[j - 1]
        }
    }

    /// BoTorch's `get_split_shapes` for one t-batch: the number of
    /// candidates `q` and the number of rows of every depth's block (`q`,
    /// then `f_1 ⋯ f_j q_j`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `q_aug` is not larger than the
    /// number of auxiliary points.
    pub fn split_sizes(&self, q_aug: usize) -> Result<(usize, Vec<usize>)> {
        let aux = self.num_auxiliary();
        if q_aug <= aux {
            return Err(BotorchError::DimensionMismatch(format!(
                "a candidate set of qMultiStepLookahead needs more than {aux} points (q \
                 candidates and the {aux} points of the look-ahead tree), got {q_aug}"
            )));
        }
        let q = q_aug - aux;
        let mut sizes = vec![q];
        let mut p = 1;
        for (qj, f) in self.batch_sizes.iter().zip(&self.num_fantasies) {
            p *= f;
            sizes.push(qj * p);
        }
        Ok((q, sizes))
    }

    /// The base samples of the fantasies of node `nu` (of `count` nodes) at
    /// depth `j` with `q` points, for output `jj` of a group of `mg` outputs
    /// (`f_{j+1} × q`).
    fn node_base(
        &self,
        j: usize,
        nu: usize,
        count: usize,
        q: usize,
        mg: usize,
        jj: usize,
    ) -> Result<Matrix> {
        let sampler = &self.samplers[j];
        let f = sampler.num_samples();
        let (z, offset) = if self.collapse || j == 0 {
            (sampler.base_samples(q * mg)?, 0)
        } else {
            (sampler.base_samples(count * q * mg)?, nu * q)
        };
        Ok(Matrix::from_fn(f, q, |i, p| z[(i, (offset + p) * mg + jj)]))
    }

    /// The forward pass of one node: posteriors, stage value and fantasies.
    fn eval_node(
        &self,
        node: &mut Node<'_>,
        j: usize,
        nu: usize,
        count: usize,
        groups: &[usize],
        with_grad: bool,
    ) -> Result<()> {
        let leaf = j == self.batch_sizes.len();
        let stage = self.stages[j].as_ref();
        let with_cov = !leaf || stage.is_some_and(|s| s.kind.uses_covariance());
        node.posts = node
            .gps
            .iter()
            .map(|g| g.posterior(&node.x, with_cov))
            .collect::<Result<_>>()?;
        if let Some(core) = stage {
            node.stage = Some(self.eval_stage(node, j, core, with_grad)?);
        }
        if leaf {
            return Ok(());
        }
        let q = node.x.nrows();
        let mut roots = Vec::with_capacity(node.gps.len());
        let mut base = Vec::with_capacity(node.gps.len());
        let mut offset = 0;
        for &mg in groups {
            for jj in 0..mg {
                let t = offset + jj;
                let g = &node.gps[t];
                let (_, sd) = g.outcome_affine()?;
                let noise = g.posterior_noise()?;
                let mut cov = node.posts[t]
                    .covariance
                    .clone()
                    .expect("computed for inner nodes");
                for p in 0..q {
                    cov[(p, p)] += sd * sd * noise;
                }
                roots.push(CovarianceRoot::new(&cov)?);
                base.push(self.node_base(j, nu, count, q, mg, jj)?);
            }
            offset += mg;
        }
        node.fantasy = Some(FantasyEval { roots, base });
        Ok(())
    }

    /// The stage value function `core` at a node.
    fn eval_stage(
        &self,
        node: &Node<'_>,
        j: usize,
        core: &ValueFunctionCore,
        with_grad: bool,
    ) -> Result<StageEval> {
        let m = node.gps.len();
        let n = node.x.nrows();
        let mut mean = Vector::zeros(n * m);
        let mut cov = Matrix::zeros(n * m, n * m);
        for (t, p) in node.posts.iter().enumerate() {
            mean.rows_mut(t * n, n).copy_from(&p.mean);
            if let Some(c) = &p.covariance {
                cov.view_mut((t * n, t * n), (n, n)).copy_from(c);
            }
        }
        let post = Posterior::multi_output(mean, cov, m);
        let mut best_f = None;
        let local;
        let core = if self.best_f_from_model[j] {
            let g = &node.gps[0];
            let targets = g.targets();
            let (imax, tmax) =
                targets
                    .iter()
                    .enumerate()
                    .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv {
                            (i, v)
                        } else {
                            (bi, bv)
                        }
                    });
            let (mu, sd) = g.outcome_affine()?;
            best_f = Some((imax, 0.0));
            let kind = StageValueFunction {
                kind: core.kind.clone(),
                best_f_from_model: true,
                sampler: None,
            }
            .with_best_f(mu + sd * tmax);
            local = ValueFunctionCore {
                kind,
                objective: core.objective.clone(),
                sampler: core.sampler.clone(),
            };
            &local
        } else {
            core
        };
        let (pv, transform) = match &self.posterior_transform {
            Some(t) => (
                core.evaluate(&t.transform_posterior(&post)?, &node.x, with_grad)?,
                Some(t),
            ),
            None => (core.evaluate(&post, &node.x, with_grad)?, None),
        };
        if !with_grad {
            return Ok(StageEval {
                value: pv.value,
                mean_bars: Vec::new(),
                cov_bars: Vec::new(),
                x_bar: None,
                best_f,
            });
        }
        let (mean_bar, cov_bar) = match transform {
            Some(t) => {
                let cb = pv.cov_bar.clone().unwrap_or_else(|| Matrix::zeros(n, n));
                let (mb, cbf) = t.pull_back(&pv.mean_bar, &cb);
                (mb, pv.cov_bar.is_some().then_some(cbf))
            }
            None => (pv.mean_bar, pv.cov_bar),
        };
        let mean_bars = (0..m)
            .map(|t| Vector::from_fn(n, |p, _| mean_bar[t * n + p]))
            .collect();
        let cov_bars = (0..m)
            .map(|t| {
                cov_bar
                    .as_ref()
                    .map(|c| c.view((t * n, t * n), (n, n)).into_owned())
            })
            .collect();
        Ok(StageEval {
            value: pv.value,
            mean_bars,
            cov_bars,
            x_bar: pv.x_bar,
            best_f: best_f.map(|(i, _)| (i, pv.best_f_bar)),
        })
    }

    /// The children of a node (in the order of the fantasies).
    fn children<'a>(
        &self,
        node: &Node<'a>,
        j: usize,
        nu: usize,
        x: &Matrix,
        block_start: usize,
    ) -> Result<Vec<(usize, Node<'a>)>> {
        let fantasy = node.fantasy.as_ref().expect("inner node");
        let f = self.num_fantasies[j];
        let q = node.x.nrows();
        let count: usize = self.num_fantasies[..j].iter().product();
        // The appended rows of the children: the node's and its points.
        let mut rows = (*node.appended_rows).clone();
        rows.extend(node.x_rows.iter().copied());
        let rows = Arc::new(rows);
        let factors: Vec<Arc<CondFactor>> = node
            .gps
            .iter()
            .map(|g| g.enlarge(&node.x, g.fantasy_noise()?))
            .collect::<Result<_>>()?;
        let qn = self.batch_sizes[j];
        let mut out = Vec::with_capacity(f);
        for i in 0..f {
            let child_nu = i * count + nu;
            let mut gps = Vec::with_capacity(node.gps.len());
            for (t, g) in node.gps.iter().enumerate() {
                let (mu, sd) = g.outcome_affine()?;
                let r = fantasy.roots[t].to_matrix();
                let z = fantasy.base[t].row(i).transpose();
                let y = &node.posts[t].mean + &r * z;
                let y_tf: Vec<f64> = y.iter().map(|&v| (v - mu) / sd).collect();
                debug_assert_eq!(y_tf.len(), q);
                gps.push(g.child(Arc::clone(&factors[t]), &y_tf));
            }
            let first = block_start + child_nu * qn;
            let x_rows: Vec<Option<usize>> = (first..first + qn).map(Some).collect();
            out.push((
                child_nu,
                Node {
                    gps,
                    weight: node.weight * self.weights[j][i],
                    x: x.rows(first, qn).into_owned(),
                    x_rows,
                    appended_rows: Arc::clone(&rows),
                    posts: Vec::new(),
                    stage: None,
                    fantasy: None,
                },
            ));
        }
        Ok(out)
    }

    /// The reverse pass of one node, given the adjoints of the targets
    /// appended by it and its ancestors from its children.
    fn backward_node(
        &self,
        node: &Node<'_>,
        incoming: &[Vector],
        children_own: &[Vec<Vector>],
        own_len: usize,
    ) -> Result<NodeBackward> {
        let d = self.model.dim();
        let mut rows = Vec::new();
        let mut older = Vec::with_capacity(node.gps.len());
        let mut own = Vec::with_capacity(node.gps.len());
        for (t, g) in node.gps.iter().enumerate() {
            let post = &node.posts[t];
            let q = node.x.nrows();
            let mut acc: CondAdjoint = g.adjoint();
            if !incoming.is_empty() {
                acc.tbar += &incoming[t];
            }
            let mut mean_bar = Vector::zeros(q);
            let mut cov_bar: Option<Matrix> = None;
            if let Some(s) = &node.stage {
                mean_bar += &s.mean_bars[t] * node.weight;
                cov_bar = s.cov_bars[t].as_ref().map(|c| c * node.weight);
                if let Some((imax, bbar)) = s.best_f {
                    let n = g.gp().num_train();
                    if imax >= n {
                        let (_, sd) = g.outcome_affine()?;
                        acc.tbar[imax - n] += node.weight * bbar * sd;
                    }
                }
            }
            if let Some(fantasy) = &node.fantasy {
                let (_, sd) = g.outcome_affine()?;
                let base = &fantasy.base[t];
                let mut root_bar = Matrix::zeros(q, q);
                for (i, child) in children_own.iter().enumerate() {
                    let ybar = &child[t] / sd;
                    mean_bar += &ybar;
                    root_bar += &ybar * base.row(i);
                }
                let cb = fantasy.roots[t].backward(&root_bar)?;
                cov_bar = Some(match cov_bar {
                    Some(c) => c + cb,
                    None => cb,
                });
            }
            let gx = g.posterior_backward(post, &mean_bar, cov_bar.as_ref(), &mut acc);
            for (p, r) in node.x_rows.iter().enumerate() {
                if let Some(r) = r {
                    rows.push((*r, gx.row(p).transpose()));
                }
            }
            let ga = g.appended_backward(&acc);
            for (p, r) in node.appended_rows.iter().enumerate() {
                if let Some(r) = r {
                    rows.push((*r, ga.row(p).transpose()));
                }
            }
            let a = acc.tbar.len();
            older.push(acc.tbar.rows(0, a - own_len).into_owned());
            own.push(acc.tbar.rows(a - own_len, own_len).into_owned());
        }
        if let Some(s) = &node.stage {
            if let Some(xb) = &s.x_bar {
                for (p, r) in node.x_rows.iter().enumerate() {
                    if let Some(r) = r {
                        rows.push((*r, xb.row(p).transpose() * node.weight));
                    }
                }
            }
        }
        debug_assert!(rows.iter().all(|(_, g)| g.len() == d));
        Ok(NodeBackward { rows, older, own })
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.model.dim();
        check_candidates(x, d, None)?;
        let (q, sizes) = self.split_sizes(x.nrows())?;
        let k = self.batch_sizes.len();
        let (gps, groups) = output_gps(self.model.as_ref())?;
        // Block starts of every depth.
        let mut starts = Vec::with_capacity(k + 1);
        let mut acc = 0;
        for s in &sizes {
            starts.push(acc);
            acc += s;
        }
        // The root: the candidates and pending points.
        let pending = read_lock(&self.x_pending).clone();
        let x0 = x.rows(0, q).into_owned();
        let (x_root, mut x_rows): (Matrix, Vec<Option<usize>>) = match &pending {
            Some(p) => (vstack(&[&x0, p], d), (0..q).map(Some).collect()),
            None => (x0, (0..q).map(Some).collect()),
        };
        if let Some(p) = &pending {
            x_rows.extend(std::iter::repeat_n(None, p.nrows()));
        }
        let root = Node {
            gps: gps.iter().map(|g| CondGp::root(g)).collect::<Result<_>>()?,
            weight: 1.0,
            x: x_root,
            x_rows,
            appended_rows: Arc::new(Vec::new()),
            posts: Vec::new(),
            stage: None,
            fantasy: None,
        };
        let mut levels: Vec<Vec<Node<'_>>> = vec![vec![root]];
        for j in 0..=k {
            let count = levels[j].len();
            let q_j = self.points_at(j, levels[j][0].x.nrows());
            let level = &mut levels[j];
            let eval = |(nu, node): (usize, &mut Node<'_>)| {
                self.eval_node(node, j, nu, count, &groups, with_grad)
            };
            if count >= PAR_NODES {
                level
                    .par_iter_mut()
                    .enumerate()
                    .map(eval)
                    .collect::<Result<()>>()?;
            } else {
                level
                    .iter_mut()
                    .enumerate()
                    .map(eval)
                    .collect::<Result<()>>()?;
            }
            debug_assert!(levels[j].iter().all(|n| n.x.nrows() == q_j));
            if j == k {
                break;
            }
            let block_start = starts[j + 1];
            let level = &levels[j];
            let make = |(nu, node)| self.children(node, j, nu, x, block_start);
            let kids: Vec<Vec<(usize, Node<'_>)>> = if count >= PAR_NODES {
                level
                    .par_iter()
                    .enumerate()
                    .map(make)
                    .collect::<Result<_>>()?
            } else {
                level.iter().enumerate().map(make).collect::<Result<_>>()?
            };
            let total = count * self.num_fantasies[j];
            let mut next: Vec<Option<Node<'_>>> = (0..total).map(|_| None).collect();
            for (nu_child, child) in kids.into_iter().flatten() {
                next[nu_child] = Some(child);
            }
            levels.push(
                next.into_iter()
                    .map(|n| n.expect("every child is built"))
                    .collect(),
            );
        }
        let value: f64 = levels
            .iter()
            .flatten()
            .filter_map(|n| n.stage.as_ref().map(|s| n.weight * s.value))
            .sum();
        if !with_grad {
            return Ok((value, None));
        }
        let mut grad = Matrix::zeros(x.nrows(), d);
        let m = gps.len();
        // Reverse pass, deepest level first. `incoming[nu][t]`: adjoints of
        // the targets appended by node `nu` and its ancestors from its
        // children; `own[nu][i][t]`: the adjoint of the targets of fantasy
        // `i` of node `nu`.
        let mut incoming: Vec<Vec<Vector>> = Vec::new();
        let mut own_from_children: Vec<Vec<Vec<Vector>>> = Vec::new();
        for j in (0..=k).rev() {
            let level = &levels[j];
            let count = level.len();
            let own_len = if j == 0 {
                0
            } else {
                levels[j - 1][0].x.nrows()
            };
            let back = |(nu, node): (usize, &Node<'_>)| {
                let inc: &[Vector] = incoming.get(nu).map_or(&[], |v| v.as_slice());
                let kids: &[Vec<Vector>] = own_from_children.get(nu).map_or(&[], |v| v.as_slice());
                self.backward_node(node, inc, kids, own_len)
            };
            let results: Vec<NodeBackward> = if count >= PAR_NODES {
                level
                    .par_iter()
                    .enumerate()
                    .map(back)
                    .collect::<Result<_>>()?
            } else {
                level.iter().enumerate().map(back).collect::<Result<_>>()?
            };
            if j == 0 {
                for r in results {
                    for (row, g) in r.rows {
                        let mut gr = grad.row_mut(row);
                        gr += g.transpose();
                    }
                }
                break;
            }
            let parents = levels[j - 1].len();
            let f = self.num_fantasies[j - 1];
            let mut new_incoming: Vec<Vec<Vector>> = levels[j - 1]
                .iter()
                .map(|p| {
                    p.gps
                        .iter()
                        .map(|g| Vector::zeros(g.num_appended()))
                        .collect()
                })
                .collect();
            let mut new_own: Vec<Vec<Vec<Vector>>> =
                (0..parents).map(|_| vec![Vec::new(); f]).collect();
            for (nu, r) in results.into_iter().enumerate() {
                for (row, g) in r.rows {
                    let mut gr = grad.row_mut(row);
                    gr += g.transpose();
                }
                let parent = nu % parents;
                let i = nu / parents;
                for t in 0..m {
                    new_incoming[parent][t] += &r.older[t];
                }
                new_own[parent][i] = r.own;
            }
            incoming = new_incoming;
            own_from_children = new_own;
        }
        Ok((value, Some(grad)))
    }
}

impl AcquisitionFunction for QMultiStepLookahead {
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

    fn as_one_shot(&self) -> Option<&dyn OneShotAcquisitionFunction> {
        Some(self)
    }
}

impl OneShotAcquisitionFunction for QMultiStepLookahead {
    fn get_augmented_q_batch_size(&self, q: usize) -> usize {
        q + self.num_auxiliary()
    }

    /// BoTorch's `optimize_acqf` is called with the augmented size (`q =
    /// get_augmented_q_batch_size(q)`) for `qMultiStepLookahead`.
    fn optimized_q_batch_size(&self, q: usize) -> usize {
        q
    }

    fn extract_candidates(&self, x_full: &Matrix) -> Result<Matrix> {
        let (q, _) = self.split_sizes(x_full.nrows())?;
        Ok(x_full.rows(0, q).into_owned())
    }

    /// BoTorch's default `gen_batch_initial_conditions` of the augmented
    /// candidate sets (`q` is the augmented size).
    fn gen_one_shot_initial_conditions(
        &self,
        bounds: &Matrix,
        q: usize,
        num_restarts: usize,
        raw_samples: usize,
        options: &OptimizeAcqfOptions,
        rng: &mut dyn Rng,
    ) -> Result<(Vec<Matrix>, Vec<String>)> {
        let (ics, fallback) = crate::optim::gen_batch_initial_conditions(
            self,
            bounds,
            q,
            num_restarts,
            raw_samples,
            &options.initial_condition_options(),
            rng,
        )?;
        let warnings = if fallback {
            vec![
                "Unable to find non-zero acquisition function values - initial conditions \
                  are being selected randomly."
                    .to_string(),
            ]
        } else {
            Vec::new()
        };
        Ok((ics, warnings))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SingleTaskGP, SingleTaskGPOptions};
    use crate::transforms::Normalize;

    fn model(fixed: bool) -> SingleTaskGP {
        let x = Matrix::from_fn(7, 2, |i, j| {
            ((i * 5 + j * 3) % 7) as f64 / 7.0 + 0.04 * j as f64
        });
        let y = Matrix::from_fn(7, 1, |i, _| {
            let s: f64 = x.row(i).iter().sum();
            (4.0 * s).sin() + 0.3 * s
        });
        let yvar = fixed.then(|| Matrix::from_fn(7, 1, |i, _| 0.02 + 0.003 * i as f64));
        let mut gp = SingleTaskGP::new(
            x,
            y,
            SingleTaskGPOptions {
                train_yvar: yvar,
                input_transform: Some(Normalize::new(2)),
                ..Default::default()
            },
        )
        .unwrap();
        let mut h = gp.hyperparameters().clone();
        h.raw_lengthscale = Vector::from_vec(vec![0.35, 0.6]);
        if h.raw_noise.is_some() {
            h.raw_noise = Some(0.03);
        }
        gp.set_hyperparameters(h).unwrap();
        gp
    }

    fn sampler(f: usize, q: usize, seed: u64) -> Arc<SobolQMCNormalSampler> {
        let z = Matrix::from_fn(f, q, |i, p| {
            (((i * 7 + p * 3 + seed as usize) % 11) as f64 - 5.0) / 3.0
        });
        Arc::new(SobolQMCNormalSampler::from_base_samples(z, Some(seed)).unwrap())
    }

    /// Reference value of the two-step tree with the posterior mean at the
    /// leaves and qEI-free stages: nested `fantasize` of the model.
    fn reference(
        gp: &SingleTaskGP,
        x: &Matrix,
        s0: &SobolQMCNormalSampler,
        s1: &SobolQMCNormalSampler,
    ) -> f64 {
        let x0 = x.rows(0, 1).into_owned();
        let f1 = s0.num_samples();
        let f2 = s1.num_samples();
        let models = gp.fantasize(&x0, s0, None).unwrap();
        let mut total = 0.0;
        for (i1, m1) in models.iter().enumerate() {
            let x1 = x.rows(1 + i1, 1).into_owned();
            let models2 = m1.fantasize(&x1, s1, None).unwrap();
            for (i2, m2) in models2.iter().enumerate() {
                let x2 = x.rows(1 + f1 + i2 * f1 + i1, 1).into_owned();
                total += m2.posterior(&x2, false).unwrap().mean[0];
            }
        }
        total / (f1 * f2) as f64
    }

    #[test]
    fn values_match_nested_fantasies_and_gradients_match_differences() {
        for fixed in [false, true] {
            let gp = model(fixed);
            let m: Arc<dyn Model> = Arc::new(gp.clone());
            let (s0, s1) = (sampler(3, 1, 1), sampler(2, 1, 2));
            let acqf = QMultiStepLookahead::new(
                Arc::clone(&m),
                QMultiStepLookaheadOptions {
                    batch_sizes: vec![1, 1],
                    samplers: Some(vec![Arc::clone(&s0), Arc::clone(&s1)]),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(acqf.num_auxiliary(), 3 + 6);
            let x = Matrix::from_fn(10, 2, |i, j| ((i * 3 + j * 7) % 10) as f64 / 10.0 + 0.05);
            let (v, g) = acqf.forward_with_grad(&x).unwrap();
            let r = reference(&gp, &x, &s0, &s1);
            assert!(
                (v - r).abs() < 1e-12 * r.abs().max(1.0),
                "fixed={fixed}: {v} vs {r}"
            );
            let h = 1e-6;
            for row in 0..x.nrows() {
                for c in 0..2 {
                    let (mut a, mut b) = (x.clone(), x.clone());
                    a[(row, c)] += h;
                    b[(row, c)] -= h;
                    let fd = (acqf.forward(&a).unwrap() - acqf.forward(&b).unwrap()) / (2.0 * h);
                    assert!(
                        (fd - g[(row, c)]).abs() < 1e-6 * fd.abs().max(1.0),
                        "fixed={fixed} ({row},{c}): {} vs {fd}",
                        g[(row, c)]
                    );
                }
            }
        }
    }

    #[test]
    fn stage_values_best_f_and_mc_gradients_match_differences() {
        let gp = model(false);
        let m: Arc<dyn Model> = Arc::new(gp);
        let inner = sampler(5, 2, 9);
        let qei = StageValueFunction {
            kind: KgValueFunction::MonteCarlo(SampleUtility::ExpectedImprovement { best_f: 0.0 }),
            best_f_from_model: true,
            sampler: Some(Arc::clone(&inner)),
        };
        let ei = StageValueFunction {
            kind: KgValueFunction::ExpectedImprovement {
                best_f: 0.0,
                maximize: true,
            },
            best_f_from_model: true,
            sampler: None,
        };
        for collapse in [true, false] {
            let acqf = QMultiStepLookahead::new(
                Arc::clone(&m),
                QMultiStepLookaheadOptions {
                    batch_sizes: vec![2, 1],
                    samplers: Some(vec![sampler(2, 2, 3), sampler(3, 4, 4)]),
                    sample_weights: vec![Some(vec![1.0, 3.0]), None],
                    value_functions: Some(vec![
                        Some(qei.clone()),
                        Some(qei.clone()),
                        Some(ei.clone()),
                    ]),
                    x_pending: Some(Matrix::from_row_slice(1, 2, &[0.3, 0.8])),
                    collapse_fantasy_base_samples: collapse,
                    ..Default::default()
                },
            )
            .unwrap();
            let n = acqf.get_augmented_q_batch_size(1);
            assert_eq!(n, 1 + 2 * 2 + 6);
            let x = Matrix::from_fn(n, 2, |i, j| ((i * 5 + j * 3) % 11) as f64 / 11.0 + 0.03);
            let (_, g) = acqf.forward_with_grad(&x).unwrap();
            let h = 1e-6;
            for row in 0..n {
                for c in 0..2 {
                    let (mut a, mut b) = (x.clone(), x.clone());
                    a[(row, c)] += h;
                    b[(row, c)] -= h;
                    let fd = (acqf.forward(&a).unwrap() - acqf.forward(&b).unwrap()) / (2.0 * h);
                    assert!(
                        (fd - g[(row, c)]).abs() < 1e-6 * fd.abs().max(1.0),
                        "collapse={collapse} ({row},{c}): {} vs {fd}",
                        g[(row, c)]
                    );
                }
            }
            assert_eq!(
                acqf.extract_candidates(&x).unwrap(),
                x.rows(0, 1).into_owned()
            );
        }
    }
}
