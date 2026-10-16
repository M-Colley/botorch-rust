//! Initial conditions of one-shot acquisition functions
//! (`botorch.optim.initializers.gen_one_shot_hvkg_initial_conditions`) and
//! Boltzmann sampling (`botorch.utils.sampling.boltzmann_sample`).
//!
//! The fantasy designs of the hypervolume knowledge gradient are initialized
//! from hypervolume-maximizing sets of the *current* posterior mean (the
//! value function optimized from `num_inner_restarts` starts), since the
//! maximizing sets of the fantasy models are often close to them: for a
//! fraction `1 - frac_random` of the restarts, every fantasy's designs are
//! drawn (Boltzmann sampling) from the optimized sets and the `q` candidates
//! are initialized by `gen_batch_initial_conditions` with these designs
//! fixed; the other restarts are drawn entirely by
//! `gen_batch_initial_conditions`.

use std::sync::Arc;

use rand::{Rng, RngExt};

use crate::acquisition::knowledge_gradient::QKnowledgeGradient;
use crate::acquisition::multi_objective::hypervolume_knowledge_gradient::{
    HypervolumeValueFunction, QHypervolumeKnowledgeGradient,
};
use crate::acquisition::one_shot::OneShotAcquisitionFunction;
use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::lbfgsb::LbfgsbOptions;
use crate::models::gp::FantasyGp;
use crate::models::Model;
use crate::optim::initializers::gen_batch_initial_conditions_with_fantasies;
use crate::optim::optimize::{optimize_acqf_detailed, CandidateGenerator, OptimizeAcqfOptions};
use crate::Matrix;

/// BoTorch's default fraction of random restarts (`options["frac_random"]`).
pub const DEFAULT_FRAC_RANDOM: f64 = 0.1;
/// BoTorch's default number of restarts of the inner optimization
/// (`options["num_inner_restarts"]`).
pub const DEFAULT_NUM_INNER_RESTARTS: usize = 20;
/// BoTorch's default number of raw samples of the inner optimization
/// (`options["raw_inner_samples"]`).
pub const DEFAULT_RAW_INNER_SAMPLES: usize = 1024;
/// BoTorch's default Boltzmann temperature of the fantasy designs
/// (`options.get("eta", 2.0)`).
pub const DEFAULT_ONE_SHOT_ETA: f64 = 2.0;

/// The options of the one-shot initial conditions (the entries of BoTorch's
/// `options` dictionary that only `gen_one_shot_hvkg_initial_conditions`
/// reads).
#[derive(Clone, Debug, PartialEq)]
pub struct OneShotInitialConditionOptions {
    /// The fraction of restarts whose fantasy designs are random
    /// (`frac_random`, in `(0, 1)`, default 0.1).
    pub frac_random: f64,
    /// Restarts of the optimization of the value function
    /// (`num_inner_restarts`, default 20).
    pub num_inner_restarts: usize,
    /// Raw samples of the optimization of the value function
    /// (`raw_inner_samples`, default 1024).
    pub raw_inner_samples: usize,
    /// The Boltzmann temperature of the selection of the optimized designs.
    /// BoTorch reads `options["eta"]`, which also sets the temperature of
    /// `initialize_q_batch`, and defaults to 2 here (and to 1 there); `None`
    /// is 2.
    pub eta: Option<f64>,
}

impl Default for OneShotInitialConditionOptions {
    fn default() -> Self {
        Self {
            frac_random: DEFAULT_FRAC_RANDOM,
            num_inner_restarts: DEFAULT_NUM_INNER_RESTARTS,
            raw_inner_samples: DEFAULT_RAW_INNER_SAMPLES,
            eta: None,
        }
    }
}

/// BoTorch's `standardize` of a vector: zero mean and unit (unbiased)
/// standard deviation, with standard deviations below `1e-9` replaced by 1.
fn standardize(v: &[f64]) -> Vec<f64> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let std = if v.len() > 1 {
        (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        f64::NAN
    };
    let std = if std >= 1e-9 { std } else { 1.0 };
    v.iter().map(|x| (x - mean) / std).collect()
}

/// `n` indices drawn with probabilities proportional to `weights`, with or
/// without replacement (`torch.multinomial`).
fn multinomial<R: Rng + ?Sized>(
    weights: &[f64],
    n: usize,
    replacement: bool,
    rng: &mut R,
) -> Result<Vec<usize>> {
    if weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) || weights.iter().sum::<f64>() <= 0.0 {
        return Err(BotorchError::InvalidArgument(
            "invalid multinomial distribution (weights must be finite, non-negative and not \
             all zero)"
                .into(),
        ));
    }
    if !replacement && n > weights.iter().filter(|&&w| w > 0.0).count() {
        return Err(BotorchError::InvalidArgument(
            "cannot sample more indices than there are categories without replacement".into(),
        ));
    }
    let mut w = weights.to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let total: f64 = w.iter().sum();
        let mut u = rng.random::<f64>() * total;
        let mut pick = w.iter().rposition(|&x| x > 0.0).expect("a positive weight");
        for (i, &wi) in w.iter().enumerate() {
            if wi > 0.0 {
                if u < wi {
                    pick = i;
                    break;
                }
                u -= wi;
            }
        }
        out.push(pick);
        if !replacement {
            w[pick] = 0.0;
        }
    }
    Ok(out)
}

/// BoTorch's `boltzmann_sample(function_values, num_samples, eta,
/// replacement)`: indices drawn with probabilities proportional to
/// `exp(eta · z)` for the standardized values `z`; `eta` is halved while a
/// weight overflows. `rng` replaces torch's global generator.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for non-finite values or more samples
/// than values without replacement.
pub fn boltzmann_sample<R: Rng + ?Sized>(
    function_values: &[f64],
    num_samples: usize,
    eta: f64,
    replacement: bool,
    rng: &mut R,
) -> Result<Vec<usize>> {
    if function_values.is_empty() || num_samples == 0 {
        return Err(BotorchError::InvalidArgument(format!(
            "cannot sample n_sample <= 0 samples (got {num_samples} samples from {} values)",
            function_values.len()
        )));
    }
    let z = standardize(function_values);
    let mut eta = eta;
    let mut weights: Vec<f64> = z.iter().map(|v| (eta * v).exp()).collect();
    while weights.iter().any(|w| w.is_infinite()) {
        eta *= 0.5;
        weights = z.iter().map(|v| (eta * v).exp()).collect();
    }
    multinomial(&weights, num_samples, replacement, rng)
}

/// The value function at the target fidelities (BoTorch's
/// `FixedFeatureAcquisitionFunction(value_function, d, fidelity_dims,
/// targets)`): evaluated on the non-fidelity columns, with the fidelity
/// columns inserted.
#[derive(Debug)]
struct FixedFidelityValue {
    inner: HypervolumeValueFunction,
    d: usize,
    /// `(column, value)` of the fixed columns, sorted by column.
    fixed: Vec<(usize, f64)>,
}

impl FixedFidelityValue {
    fn full(&self, x: &Matrix) -> Matrix {
        let mut out = Matrix::zeros(x.nrows(), self.d);
        let mut src = 0;
        for c in 0..self.d {
            match self.fixed.iter().find(|(k, _)| *k == c) {
                Some(&(_, v)) => out.column_mut(c).fill(v),
                None => {
                    out.column_mut(c).copy_from(&x.column(src));
                    src += 1;
                }
            }
        }
        out
    }

    fn free_columns(&self) -> Vec<usize> {
        (0..self.d)
            .filter(|c| !self.fixed.iter().any(|(k, _)| k == c))
            .collect()
    }
}

impl AcquisitionFunction for FixedFidelityValue {
    fn model(&self) -> &Arc<dyn Model> {
        self.inner.model()
    }

    fn dim(&self) -> usize {
        self.d - self.fixed.len()
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        crate::acquisition::check_candidates(x, self.dim(), None)?;
        self.inner.forward(&self.full(x))
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        crate::acquisition::check_candidates(x, self.dim(), None)?;
        let (v, g) = self.inner.forward_with_grad(&self.full(x))?;
        let free = self.free_columns();
        Ok((
            v,
            Matrix::from_fn(x.nrows(), free.len(), |p, j| g[(p, free[j])]),
        ))
    }

    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        self.inner.set_x_pending(x_pending)
    }
}

/// The optimized hypervolume-maximizing sets of the value function (the
/// inner optimization of [`gen_one_shot_hvkg_initial_conditions`]).
#[derive(Clone, Debug)]
pub struct ValueFunctionSolutions {
    /// One optimized set of `num_pareto` designs per inner restart (all `d`
    /// columns; at the target fidelities for multi-fidelity functions).
    pub candidates: Vec<Matrix>,
    /// The value function at every set.
    pub values: Vec<f64>,
    /// BoTorch's warnings of the optimization.
    pub warnings: Vec<String>,
}

/// The inner optimization of [`gen_one_shot_hvkg_initial_conditions`]:
/// `optimize_acqf` of the value function of the current model (BoTorch's
/// `_get_hv_value_function` with the knowledge gradient's objective, inner
/// sampler and `use_posterior_mean`) for `num_pareto` designs, from
/// `options.one_shot.num_inner_restarts` starts selected among
/// `options.one_shot.raw_inner_samples` raw samples, returning every restart.
/// For multi-fidelity knowledge gradients the designs are optimized over the
/// non-fidelity columns with the fidelities at their targets (BoTorch's
/// `FixedFeatureAcquisitionFunction`).
///
/// # Errors
/// [`BotorchError::Unsupported`] for parameter constraints with a
/// multi-fidelity knowledge gradient; errors of `optimize_acqf`.
pub fn optimize_hvkg_value_function(
    acq_function: &QHypervolumeKnowledgeGradient,
    bounds: &Matrix,
    options: &OptimizeAcqfOptions,
) -> Result<ValueFunctionSolutions> {
    let os = &options.one_shot;
    let d = bounds.ncols();
    let np = acq_function.num_pareto();
    let value_function = acq_function.value_function()?;
    // BoTorch's inner `optimize_acqf` receives the `options` dictionary only:
    // not the outer call's `post_processing_func`, `gen_candidates` or
    // `timeout_sec`.
    let inner_options = OptimizeAcqfOptions {
        batch_initial_conditions: None,
        sequential: false,
        ic_generator: None,
        post_processing: None,
        gen_candidates: CandidateGenerator::Scipy,
        timeout_sec: None,
        ..options.clone()
    };
    let (candidates, values, warnings) = match acq_function.multi_fidelity() {
        None => {
            let r = optimize_acqf_detailed(
                &value_function,
                bounds,
                np,
                os.num_inner_restarts,
                os.raw_inner_samples,
                &inner_options,
            )?;
            let (c, v): (Vec<Matrix>, Vec<f64>) = r
                .restarts
                .into_iter()
                .map(|r| (r.candidates, r.acq_value))
                .unzip();
            (c, v, r.warnings)
        }
        Some(mf) => {
            if !options.constraints.is_empty() {
                return Err(BotorchError::Unsupported(
                    "parameter constraints are not supported by the one-shot initial conditions \
                     of the multi-fidelity hypervolume knowledge gradient"
                        .into(),
                ));
            }
            let mut fixed: Vec<(usize, f64)> = Vec::new();
            for &(k, v) in &mf.target_fidelities {
                let c = if k < 0 { d as isize + k } else { k } as usize;
                fixed.retain(|(f, _)| *f != c);
                fixed.push((c, v));
            }
            fixed.sort_by_key(|&(c, _)| c);
            let wrapped = FixedFidelityValue {
                inner: value_function,
                d,
                fixed,
            };
            let free = wrapped.free_columns();
            let reduced = Matrix::from_fn(2, free.len(), |i, j| bounds[(i, free[j])]);
            let mut reduced_options = inner_options;
            reduced_options.fixed_features = options
                .fixed_features
                .iter()
                .filter_map(|(&k, &v)| free.iter().position(|&c| c == k).map(|j| (j, v)))
                .collect();
            let r = optimize_acqf_detailed(
                &wrapped,
                &reduced,
                np,
                os.num_inner_restarts,
                os.raw_inner_samples,
                &reduced_options,
            )?;
            let (c, v): (Vec<Matrix>, Vec<f64>) = r
                .restarts
                .into_iter()
                .map(|r| (wrapped.full(&r.candidates), r.acq_value))
                .unzip();
            (c, v, r.warnings)
        }
    };
    Ok(ValueFunctionSolutions {
        candidates,
        values,
        warnings,
    })
}

/// Records BoTorch's `BadInitialCandidatesWarning` of a random fallback.
fn fallback_warning(fallback: bool, warnings: &mut Vec<String>) {
    if fallback {
        warnings.push(
            "Unable to find non-zero acquisition function values - initial conditions are \
             being selected randomly."
                .into(),
        );
    }
}

/// BoTorch's `gen_one_shot_hvkg_initial_conditions(acq_function, bounds, q,
/// num_restarts, raw_samples, fixed_features, options, inequality_constraints,
/// equality_constraints)` (see the [module documentation](self)): returns
/// `num_restarts` augmented candidate sets (`(q + n_f·num_pareto) × d`) and
/// the messages of BoTorch's warnings. The settings come from `options`
/// (seed, `eta`, `alpha`, batch limits, fixed features, linear constraints,
/// L-BFGS-B options of the inner optimization and
/// [`OptimizeAcqfOptions::one_shot`]); `rng` replaces torch's global
/// generator.
///
/// For multi-fidelity knowledge gradients, the value function is optimized
/// over the non-fidelity columns with the fidelities at their targets, as in
/// BoTorch (negative target-fidelity indices are resolved, which BoTorch does
/// not do here; fixed features are mapped to the non-fidelity columns).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for `frac_random` outside `(0, 1)`
/// (BoTorch's message); errors of the optimization of the value function and
/// of [`crate::optim::gen_batch_initial_conditions`].
pub fn gen_one_shot_hvkg_initial_conditions(
    acq_function: &QHypervolumeKnowledgeGradient,
    bounds: &Matrix,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    options: &OptimizeAcqfOptions,
    rng: &mut dyn Rng,
) -> Result<(Vec<Matrix>, Vec<String>)> {
    let os = &options.one_shot;
    let frac_random = os.frac_random;
    if !(0.0 < frac_random && frac_random < 1.0) {
        return Err(BotorchError::InvalidArgument(format!(
            "frac_random must take on values in (0,1). Value: {frac_random}"
        )));
    }
    let d = bounds.ncols();
    let n_f = acq_function.num_fantasies();
    let np = acq_function.num_pareto();
    let mut warnings = Vec::new();
    let inner = optimize_hvkg_value_function(acq_function, bounds, options)?;
    warnings.extend(inner.warnings);
    let (cands, vals) = (inner.candidates, inner.values);
    let num_optim = (num_restarts as f64 * (1.0 - frac_random)).round_ties_even() as usize;
    let num_optim = num_optim.min(num_restarts);
    let mut optim_ics: Vec<Matrix> = Vec::with_capacity(num_optim);
    if num_optim > 0 {
        let eta = os.eta.unwrap_or(DEFAULT_ONE_SHOT_ETA);
        let idx = boltzmann_sample(&vals, num_optim * n_f, eta, true, rng)?;
        for r in 0..num_optim {
            let mut m = Matrix::zeros(n_f * np, d);
            for i in 0..n_f {
                m.rows_mut(i * np, np).copy_from(&cands[idx[r * n_f + i]]);
            }
            optim_ics.push(m);
        }
    }
    // BoTorch passes all `options` to `gen_batch_initial_conditions`
    // (including `topn` and `sample_around_best`).
    let ic_options = options.initial_condition_options();
    let num_random = num_restarts - num_optim;
    let mut base_ics: Vec<Matrix>;
    let optim_idcs: Vec<usize>;
    if num_random > 0 {
        let q_aug = q + n_f * np;
        let (ics, fallback) = gen_batch_initial_conditions_with_fantasies(
            acq_function,
            bounds,
            q_aug,
            num_restarts,
            raw_samples,
            &ic_options,
            None,
            rng,
        )?;
        fallback_warning(fallback, &mut warnings);
        base_ics = ics;
        if num_optim > 0 {
            let probs = vec![1.0 / num_restarts as f64; num_restarts];
            let optim_idxr = multinomial(&probs, num_optim, false, rng)?;
            for (j, &r) in optim_idxr.iter().enumerate() {
                base_ics[r].rows_mut(q, n_f * np).copy_from(&optim_ics[j]);
            }
            optim_idcs = optim_idxr;
        } else {
            optim_idcs = Vec::new();
        }
    } else {
        base_ics = optim_ics
            .iter()
            .map(|o| {
                let mut m = Matrix::zeros(q + n_f * np, d);
                m.rows_mut(q, n_f * np).copy_from(o);
                m
            })
            .collect();
        optim_idcs = (0..num_restarts).collect();
    }
    if num_optim == 0 {
        return Ok((base_ics, warnings));
    }
    let mut out = Vec::with_capacity(num_restarts);
    for (i, base) in base_ics.iter().enumerate() {
        if optim_idcs.contains(&i) {
            let fixed = base.rows(q, n_f * np).into_owned();
            let (ics, fallback) = gen_batch_initial_conditions_with_fantasies(
                acq_function,
                bounds,
                q,
                1,
                raw_samples,
                &ic_options,
                Some(&fixed),
                rng,
            )?;
            fallback_warning(fallback, &mut warnings);
            out.extend(ics);
        } else {
            out.push(base.clone());
        }
    }
    Ok((out, warnings))
}

/// BoTorch's `gen_one_shot_kg_initial_conditions(acq_function, bounds, q,
/// num_restarts, raw_samples, fixed_features, options,
/// inequality_constraints, equality_constraints)`: `num_restarts` augmented
/// candidate sets (`(q + n_f) × d`) from `gen_batch_initial_conditions` of the
/// knowledge gradient, whose last `int((1 - frac_random) n_f)` fantasy points
/// are replaced by maximizers of the value function of the current model
/// (`optimize_acqf` from `num_inner_restarts` starts among
/// `raw_inner_samples` raw samples), drawn by Boltzmann sampling of their
/// values (temperature `eta`, default 2). Returns the initial conditions and
/// the messages of BoTorch's warnings; `rng` replaces torch's global
/// generator.
///
/// BoTorch fails when no fantasy point is replaced (a single fantasy:
/// `torch.multinomial` of zero samples); the initial conditions of
/// `gen_batch_initial_conditions` are returned then.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for `frac_random` outside `(0, 1)`
/// (BoTorch's message), [`BotorchError::Unsupported`] for inter-point
/// constraints (BoTorch's `NotImplementedError`), and errors of the
/// optimization and of `gen_batch_initial_conditions`.
pub fn gen_one_shot_kg_initial_conditions(
    acq_function: &QKnowledgeGradient,
    bounds: &Matrix,
    q: usize,
    num_restarts: usize,
    raw_samples: usize,
    options: &OptimizeAcqfOptions,
    rng: &mut dyn Rng,
) -> Result<(Vec<Matrix>, Vec<String>)> {
    let os = &options.one_shot;
    let frac_random = os.frac_random;
    if !(0.0 < frac_random && frac_random < 1.0) {
        return Err(BotorchError::InvalidArgument(format!(
            "frac_random must take on values in (0,1). Value: {frac_random}"
        )));
    }
    for (constraints, kind) in [
        (&options.constraints.inequality, "inequality"),
        (&options.constraints.equality, "equality"),
    ] {
        if constraints.iter().any(|c| c.is_inter_point()) {
            return Err(BotorchError::Unsupported(format!(
                "Indices must be one-dimensional in gen_one_shot_kg_initial_conditions. \
                 Received inter-point indices for {kind} constrained."
            )));
        }
    }
    let q_aug = acq_function.get_augmented_q_batch_size(q);
    // BoTorch passes all `options` to `gen_batch_initial_conditions`
    // (including `topn` and `sample_around_best`, whose points are drawn for
    // the augmented candidate sets).
    let ic_options = options.initial_condition_options();
    let mut warnings = Vec::new();
    let (mut ics, fallback) = gen_batch_initial_conditions_with_fantasies(
        acq_function,
        bounds,
        q_aug,
        num_restarts,
        raw_samples,
        &ic_options,
        None,
        rng,
    )?;
    fallback_warning(fallback, &mut warnings);
    // The maximizers of the value function of the current model (BoTorch's
    // inner `optimize_acqf` with its default options, the fixed features and
    // the linear constraints).
    let value_function = acq_function.current_value_function()?;
    let mut inner = OptimizeAcqfOptions {
        seed: options.seed,
        fixed_features: options.fixed_features.clone(),
        ..Default::default()
    };
    inner.constraints.inequality = options.constraints.inequality.clone();
    inner.constraints.equality = options.constraints.equality.clone();
    let r = optimize_acqf_detailed(
        value_function.as_ref(),
        bounds,
        1,
        os.num_inner_restarts,
        os.raw_inner_samples,
        &inner,
    )?;
    warnings.extend(r.warnings);
    let (cands, vals): (Vec<Matrix>, Vec<f64>) = r
        .restarts
        .into_iter()
        .map(|r| (r.candidates, r.acq_value))
        .unzip();
    let n_value = ((1.0 - frac_random) * (q_aug - q) as f64) as usize;
    if n_value == 0 {
        return Ok((ics, warnings));
    }
    let eta = os.eta.unwrap_or(DEFAULT_ONE_SHOT_ETA);
    let idx = boltzmann_sample(&vals, num_restarts * n_value, eta, true, rng)?;
    for (r, ic) in ics.iter_mut().enumerate() {
        for j in 0..n_value {
            let row = cands[idx[r * n_value + j]].row(0).into_owned();
            ic.row_mut(q_aug - n_value + j).copy_from(&row);
        }
    }
    Ok((ics, warnings))
}

/// BoTorch's default fraction of random raw samples of
/// `gen_value_function_initial_conditions` (`options.get("frac_random", 0.6)`).
pub const DEFAULT_VALUE_FUNCTION_FRAC_RANDOM: f64 = 0.6;

/// The options of [`gen_value_function_initial_conditions`] (the entries of
/// BoTorch's `options` dictionary of `qKnowledgeGradient.evaluate`).
#[derive(Clone, Debug, PartialEq)]
pub struct ValueFunctionInitialConditionOptions {
    /// The fraction of raw samples that are random (`frac_random`, in
    /// `(0, 1)`, default 0.6).
    pub frac_random: f64,
    /// Restarts of the optimization of the current value function
    /// (`num_inner_restarts`, default 20).
    pub num_inner_restarts: usize,
    /// Raw samples of the optimization of the current value function
    /// (`raw_inner_samples`, default 1024).
    pub raw_inner_samples: usize,
    /// The Boltzmann temperature of the resampling of the optimized points
    /// and of `initialize_q_batch` (`eta`, default 2).
    pub eta: f64,
    /// The seed of the Sobol raw samples (`seed`).
    pub seed: Option<u64>,
}

impl Default for ValueFunctionInitialConditionOptions {
    fn default() -> Self {
        Self {
            frac_random: DEFAULT_VALUE_FUNCTION_FRAC_RANDOM,
            num_inner_restarts: DEFAULT_NUM_INNER_RESTARTS,
            raw_inner_samples: DEFAULT_RAW_INNER_SAMPLES,
            eta: DEFAULT_ONE_SHOT_ETA,
            seed: None,
        }
    }
}

/// BoTorch's `gen_value_function_initial_conditions(acq_function, bounds,
/// num_restarts, raw_samples, current_model, options)` for the value
/// functions of the fantasy models of a knowledge gradient: the current value
/// function is maximized (`num_inner_restarts`, `raw_inner_samples`), and for
/// every fantasy `int((1 - frac_random) raw_samples)` raw points are drawn
/// from its maximizers by Boltzmann sampling and the others from a Sobol
/// sequence (a block of `raw_samples - n_value` points per fantasy); the
/// fantasy's value function selects `num_restarts` of them with
/// `initialize_q_batch` (temperature `eta`). Returns, per fantasy, the
/// initial conditions (`1 × d` each).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for `frac_random` outside `(0, 1)`
/// (BoTorch's message), and errors of the optimization and the selection.
pub(crate) fn gen_value_function_initial_conditions(
    acq_function: &QKnowledgeGradient,
    fantasies: &[FantasyGp<'_>],
    bounds: &Matrix,
    num_restarts: usize,
    raw_samples: usize,
    options: &ValueFunctionInitialConditionOptions,
    rng: &mut dyn Rng,
) -> Result<Vec<Vec<Matrix>>> {
    let frac_random = options.frac_random;
    if !(0.0 < frac_random && frac_random < 1.0) {
        return Err(BotorchError::InvalidArgument(format!(
            "frac_random must take on values in (0,1). Value: {frac_random}"
        )));
    }
    let n_f = acq_function.num_fantasies();
    let value_function = acq_function.current_value_function()?;
    let inner = OptimizeAcqfOptions {
        seed: options.seed,
        ..Default::default()
    };
    let r = optimize_acqf_detailed(
        value_function.as_ref(),
        bounds,
        1,
        options.num_inner_restarts,
        options.raw_inner_samples,
        &inner,
    )?;
    let (cands, vals): (Vec<Matrix>, Vec<f64>) = r
        .restarts
        .into_iter()
        .map(|r| (r.candidates, r.acq_value))
        .unzip();
    let n_value = ((1.0 - frac_random) * raw_samples as f64) as usize;
    let n_random = raw_samples - n_value;
    let randomized = if n_random > 0 {
        crate::utils::sampling::draw_sobol_samples(bounds, n_random * n_f, 1, options.seed)?
    } else {
        Vec::new()
    };
    let mut out = Vec::with_capacity(n_f);
    for i in 0..n_f {
        let mut x_rnd: Vec<Matrix> = Vec::with_capacity(raw_samples);
        if n_value > 0 {
            let idx = boltzmann_sample(&vals, n_value, options.eta, true, rng)?;
            x_rnd.extend(idx.iter().map(|&k| cands[k].clone()));
        }
        x_rnd.extend(randomized[i * n_random..(i + 1) * n_random].iter().cloned());
        let acq_vals = x_rnd
            .iter()
            .map(|x| {
                Ok(acq_function
                    .fantasy_value_and_grad(fantasies, i, x, false)?
                    .0)
            })
            .collect::<Result<Vec<f64>>>()?;
        let (idcs, _) = crate::optim::initializers::initialize_q_batch(
            &acq_vals,
            num_restarts,
            options.eta,
            rng,
        )?;
        out.push(idcs.into_iter().map(|k| x_rnd[k].clone()).collect());
    }
    Ok(out)
}

/// The inner optimization of BoTorch's `qKnowledgeGradient.evaluate`: one
/// L-BFGS-B problem over the points of all restarts and fantasies,
/// maximizing the sum of the fantasies' value functions
/// (`gen_candidates_scipy(..., use_parallel_mode=False)`), followed by the
/// largest value of every fantasy over its restarts.
///
/// # Errors
/// Errors of the value functions and of L-BFGS-B.
pub(crate) fn maximize_fantasy_values(
    acq_function: &QKnowledgeGradient,
    fantasies: &[FantasyGp<'_>],
    initial_conditions: &[Vec<Matrix>],
    bounds: &Matrix,
    lbfgsb: &LbfgsbOptions,
) -> Result<Vec<f64>> {
    use rayon::prelude::*;

    let d = bounds.ncols();
    let n_f = initial_conditions.len();
    if n_f == 0 {
        return Ok(Vec::new());
    }
    let r = initial_conditions[0].len();
    // Variables in BoTorch's order: restart-major (`num_restarts x n_f x 1 x d`).
    let mut x0 = Vec::with_capacity(r * n_f * d);
    let mut lower = Vec::with_capacity(r * n_f * d);
    let mut upper = Vec::with_capacity(r * n_f * d);
    for k in 0..r {
        for ics in initial_conditions {
            for j in 0..d {
                x0.push(ics[k][(0, j)].clamp(bounds[(0, j)], bounds[(1, j)]));
                lower.push(bounds[(0, j)]);
                upper.push(bounds[(1, j)]);
            }
        }
    }
    let point = |x: &[f64], k: usize, i: usize| -> Matrix {
        let off = (k * n_f + i) * d;
        Matrix::from_row_slice(1, d, &x[off..off + d])
    };
    let objective = |x: &[f64]| -> Result<(f64, Vec<f64>)> {
        let parts = (0..r * n_f)
            .into_par_iter()
            .map(|ki| {
                let (k, i) = (ki / n_f, ki % n_f);
                acq_function.fantasy_value_and_grad(fantasies, i, &point(x, k, i), true)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut f = 0.0;
        let mut g = vec![0.0; x.len()];
        for (ki, (v, gr)) in parts.into_iter().enumerate() {
            f -= v;
            let gr = gr.expect("gradient requested");
            for j in 0..d {
                g[ki * d + j] = -gr[(0, j)];
            }
        }
        Ok((f, g))
    };
    let res = match crate::lbfgsb::minimize(objective, &x0, &lower, &upper, lbfgsb) {
        Ok(r) => r,
        Err(crate::lbfgsb::LbfgsbError::Objective(e)) => return Err(e),
        Err(crate::lbfgsb::LbfgsbError::InvalidInput(msg)) => {
            return Err(BotorchError::InvalidArgument(msg))
        }
    };
    let mut best = vec![f64::NEG_INFINITY; n_f];
    for k in 0..r {
        for (i, b) in best.iter_mut().enumerate() {
            let mut p = point(&res.x, k, i);
            for j in 0..d {
                p[(0, j)] = p[(0, j)].clamp(bounds[(0, j)], bounds[(1, j)]);
            }
            let v = acq_function
                .fantasy_value_and_grad(fantasies, i, &p, false)?
                .0;
            if v > *b || b.is_nan() {
                *b = v;
            }
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::Xoshiro256PlusPlus;
    use rand::SeedableRng;

    #[test]
    fn boltzmann_sampling_prefers_large_values() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let vals = [0.0, 1.0, 5.0, 2.0];
        let idx = boltzmann_sample(&vals, 4000, 2.0, true, &mut rng).unwrap();
        let mut counts = [0usize; 4];
        for i in idx {
            counts[i] += 1;
        }
        // Probabilities ∝ exp(2 z) with z the standardized values.
        let z = standardize(&vals);
        let w: Vec<f64> = z.iter().map(|v| (2.0 * v).exp()).collect();
        let tot: f64 = w.iter().sum();
        for k in 0..4 {
            let p = w[k] / tot;
            let f = counts[k] as f64 / 4000.0;
            assert!((f - p).abs() < 0.03, "{k}: {f} vs {p}");
        }
        let idx = boltzmann_sample(&vals, 3, 1.0, false, &mut rng).unwrap();
        let mut s = idx.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), 3);
        // Huge temperatures are halved until the weights are finite.
        let idx = boltzmann_sample(&vals, 50, 1e6, true, &mut rng).unwrap();
        assert!(idx.iter().all(|&i| i == 2));
        assert!(boltzmann_sample(&vals, 0, 1.0, true, &mut rng).is_err());
    }
}
