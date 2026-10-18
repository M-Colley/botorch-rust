# Input constructors, multi-step look-ahead and custom MC acquisition functions

The rest of BoTorch's single-objective acquisition (roadmap phase 8b): the
supported API, the deviations from BoTorch, the equivalence results and the
BoTorch bugs found. Reference: BoTorch main at da00e0401 and BoTorch 0.18.1,
under torch 2.13 and 2.14.1.

This phase ports the parts of Phase 8 that remained (`docs/roadmap.md`):
the input constructors with `get_acquisition_function`,
`qMultiStepLookahead`, `SampleReducingMCAcquisitionFunction` as a public
base class for Python subclasses, t-batched candidate sets of the max-value
samplers and t-batched fixed values of `FixedFeatureAcquisitionFunction`.

## Supported API

| BoTorch | botorch_rs | Notes |
|---|---|---|
| `botorch.acquisition.input_constructors.acqf_input_constructor`, `get_acqf_input_constructor`, `allow_only_specific_variable_kwargs` | same path (`get_acqf_input_constructor` also in `botorch_rs.acquisition`) | the registry, keyed by the botorch_rs classes; every constructor accepts `training_data` as a `SupervisedDataset` of botorch_rs or BoTorch or a dict of them (block designs), objectives, posterior transforms and constraints of either library; keyword arguments of constructors with `**kwargs` (e.g. `optimizer_options` of `construct_inputs_qKG`) are passed through (BoTorch rejects them, see bug 6) |
| `construct_inputs_posterior_mean`, `_best_f` (EI, LogEI, PI, LogPI), `_ucb`, `_pof`, `_logcei`, `_noisy_ei` (NEI, LogNEI) | same path | analytic acquisition functions; `best_f` from `get_best_f_analytic` |
| `construct_inputs_qSimpleRegret`, `_qEI`, `_qLogEI`, `_qPI`, `_qUCB`, `_qNEI`, `_qLogNEI`, `_LogPF` | same path | `best_f` from `get_best_f_mc` (feasibility-weighted with `constraints`, `compute_best_feasible_objective`), `X_baseline`, `prune_baseline`, `cache_root`, `fat`, `tau_max` / `tau_relu`, `eta`, `marginalize_dim`; `constraints` without an `objective` in `qSimpleRegret` use the identity objective (bug 7) |
| `construct_inputs_EHVI`, `_qEHVI`, `_qNEHVI`, `_qLogNEHVI`, `_qLogNParEGO` | same path | reference points from `objective_thresholds` (with `get_ref_point` / outcome constraints), partitionings (`alpha`), `X_baseline` / `prune_baseline`, constraints from `constraints_tuple` |
| `construct_inputs_qKG`, `_qMFKG`, `_qHVKG`, `_qMFHVKG`, `_mf_base` | same path | `current_value` from `optimize_objective` (botorch_rs's `optimize_acqf`; `optimizer_options`, `batch_initial_conditions` for deterministic results), `target_fidelities` / `fidelity_weights` / `cost_intercept` (the native `project_to_target_fidelity` / `expand_trace_observations` partials) |
| `construct_inputs_qMES`, `_qMFMES`, `_qJES`, `_NIPV` | same path | candidate sets drawn with the training data's RNG (torch's for tensors, so `torch.manual_seed` reproduces BoTorch's), optimal samples with botorch_rs's `get_optimal_samples`, NIPV's points with botorch_rs's `draw_sobol_samples` |
| `construct_inputs_analytic_eubo`, `construct_inputs_qeubo` | same path | added with Phase 10b (`docs/drafts/10b-preference.md`): preferential BO (`model` is the preference model) and BOPE (`pref_model`; the outcome model becomes `FixedSingleSampleModel(model, w)` with `w = torch.randn(pref_model.dim) * sample_multiplier` from torch's global generator, Python's `random` without torch) |
| `get_best_f_analytic`, `get_best_f_mc`, `optimize_objective` | same path | `posterior_transform`, `objective`, `constraints`, `fixed_features`, `linear_constraints`, `qmc` (`qmc=False` raises `UnsupportedError`: botorch_rs's `IIDNormalSampler` samples posteriors but the MC acquisition functions do not take it) |
| `botorch.acquisition.factory.get_acquisition_function` | same path (also `botorch_rs.acquisition`) | every name BoTorch handles (`qEI`, `qLogEI`, `qPI`, `qLogPF`, `qNEI`, `qLogNEI`, `qSR`, `qUCB`, `qEHVI`, `qLogEHVI`, `qNEHVI`, `qLogNEHVI`, `pTS`) with BoTorch's errors; Sobol sampler, or `IndexSampler` for ensemble models |
| `botorch.acquisition.utils.compute_best_feasible_objective`, `get_infeasible_cost`, `repeat_to_match_aug_dim`, `project_to_sample_points`, `get_acquisition_function` (deprecated) | same path | BoTorch's operations (torch for tensors); `get_infeasible_cost` from the posterior mean minus six standard deviations at `X` |
| `botorch.acquisition.multi_step_lookahead.qMultiStepLookahead` | same path (also `botorch_rs.acquisition`) | native fantasy tree (Rust): `batch_sizes`, `num_fantasies` or `samplers` (with `base_weights`), `valfunc_cls` (the knowledge gradient's native value functions: `PosteriorMean`, `PosteriorStandardDeviation`, `UpperConfidenceBound`, `(Log)ExpectedImprovement`, `(Log)ProbabilityOfImprovement`, `ScalarizedPosteriorMean`, `qSimpleRegret`, `qExpectedImprovement`, `qProbabilityOfImprovement`, `qUpper/LowerConfidenceBound`, `qPosteriorStandardDeviation`), `valfunc_argfacs` (`make_best_f` per node; others once, see deviations), `objective` (MC), `posterior_transform`, `inner_mc_samples`, `X_pending`, `collapse_fantasy_base_samples`; `get_augmented_q_batch_size`, `get_split_shapes`, `extract_candidates`, `get_induced_fantasy_model`; exact gradients (BoTorch's are wrong, bug 1); `SingleTaskGP` (one or more outputs, inferred or fixed noise, transforms) and `ModelListGP`. Rust: `botorch_core::acquisition::multi_step_lookahead` |
| `make_best_f`, `warmstart_multistep`, `get_multi_step_tree_input_representation`, `extract_candidates` | same path | `make_best_f` is the best observed value in the outcome space (bug 4); `warmstart_multistep` draws with torch's RNG for tensors (BoTorch's draws) |
| `optimize_acqf` of `qMultiStepLookahead` | `botorch_rs.optim.optimize_acqf` | one-shot: the augmented `q`, `gen_batch_initial_conditions`, `extract_candidates`; `batch_initial_conditions` from `warmstart_multistep` |
| `botorch.acquisition.monte_carlo.SampleReducingMCAcquisitionFunction`, `botorch.acquisition.logei.LogImprovementMCAcquisitionFunction` | same path (also `botorch_rs.acquisition`) | public base classes for Python subclasses: `_sample_forward` (and `objective`, `constraints`, `eta`, `fat`, `sample_reduction`, `q_reduction`) in torch; posterior samples from Rust with exact vector-Jacobian products; `optimize_acqf` works on subclasses. botorch_rs's own subclasses (`qEI`, `qPI`, `qSR`, `qUCB`, `qLCB`, `qPSD`, `qNEI`, `qLogEI`, `qLogNEI`, `qLogPF`) remain native and are now instances of these classes |
| `_sample_max_value_Gumbel`, `_sample_max_value_Thompson` | same path | `batch_shape x n x d` candidate sets: `num_samples x batch_shape` max values (`num_samples x 1` unbatched) |
| `FixedFeatureAcquisitionFunction` | same path | also t-batched values (`batch_shape x q x d_f`, broadcast against `X`); shared and per-point values stay native |

## What remains unported (in this area)

- The input constructor of `qBayesianActiveLearningByDisagreement`, which
  botorch_rs does not have (needs fully Bayesian models, Phase 10a). It can
  be registered with `acqf_input_constructor` when the class lands. The
  constructors of `AnalyticExpectedUtilityOfBestOption` and
  `qExpectedUtilityOfBestOption` came with Phase 10b
  (`docs/drafts/10b-preference.md`).
- `qMultiStepLookahead` with value functions outside the native list above
  (e.g. `qNoisyExpectedImprovement`, entropy searches, user classes), with
  argument factories other than `make_best_f` whose arguments depend on the
  fantasy model, and on models other than exact `SingleTaskGP` /
  `ModelListGP` (multi-task, approximate, fully Bayesian, ...): these raise
  `UnsupportedError`. `make_best_f` as an argument factory requires a
  single-output model (BoTorch's takes the maximum of every output of a
  multi-output model, which a scalarizing transform does not see).
- Batched candidate sets of the max-value entropy search classes
  (`qMaxValueEntropy(model, candidate_set)` with `candidate_set` of shape
  `b x n x d`): BoTorch documents `n x d` and its batched behaviour is broken
  (bug 8), so botorch_rs keeps raising `UnsupportedError`. The sampling
  functions take batched candidate sets.
- `get_best_f_mc` / `optimize_objective` with `qmc=False` (the MC
  acquisition functions do not take botorch_rs's `IIDNormalSampler`):
  `UnsupportedError`.

## Deviations from BoTorch (migration notes)

**`qMultiStepLookahead`.**

- *Gradients are exact.* BoTorch evaluates the stage value functions of the
  fantasy models with detached test caches and, from depth two, under
  `fast_pred_var`; its gradients miss the dependence through the fantasy
  models (bug 1). botorch_rs differentiates through all levels of
  conditioning; BoTorch under `settings.propagate_grads(True)` and
  `gpytorch.settings.fast_pred_var(False)` agrees with it.
- *`make_best_f`* gives every node the best observed or fantasized value in
  the outcome space of its posterior (BoTorch: the maximum of the
  *outcome-transformed* training targets, bug 4). Analytic value functions
  work with it (BoTorch fails, bug 4). The function `make_best_f(model, X)`
  itself returns the maximum of the untransformed targets.
- *Other argument factories* are called once, with the current model and
  `X=None`, and their arguments are used at every node (BoTorch calls them
  with every fantasy model). Arguments must be `best_f`, `beta`, `tau`,
  `maximize` or `weights`.
- *`collapse_fantasy_base_samples=False`* draws base samples that differ
  between the nodes of a depth and are shared by the t-batches, for every
  model (BoTorch does this for multi-output models only; for single-output
  posteriors its base samples differ between t-batches, bug 2).
- *Batch sizes `q_j > 1`*: the points of node `ν` are rows
  `ν q_j .. (ν + 1) q_j` of the depth's block of every candidate set
  (BoTorch mixes the blocks of different t-batches, bug 3).
- *One fantasy followed by several* and *multi-output models with one
  t-batch* work (BoTorch raises, bugs 5 and 9).
- `samplers` must be botorch_rs `SobolQMCNormalSampler`s;
  `get_induced_fantasy_model` requires `collapse_fantasy_base_samples=True`
  (it returns the conditioned botorch_rs models).

**Input constructors and factory.**

- The registry accepts the extra keyword arguments of constructors that
  take `**kwargs` (BoTorch's `allow_only_specific_variable_kwargs` rejects
  them, so `construct_inputs_qKG(..., optimizer_options=...)` fails through
  the registry, bug 6).
- `construct_inputs_qSimpleRegret` with `constraints` and no `objective`
  uses the identity objective (BoTorch passes `objective=None` to a
  constrained objective that then fails at evaluation, bug 7).
- Random numbers come from botorch_rs where BoTorch's own algorithms draw
  them (optimal samples of qJES, Sobol points of NIPV, `optimize_acqf` for
  KG current values); with the same draws (tests replay BoTorch's) the
  results agree. The MES candidate set uses torch's generator for tensors.

**`SampleReducingMCAcquisitionFunction` subclasses** evaluate the objective,
`_sample_forward`, constraints and reductions with torch (gradients by
autograd), with posterior samples from Rust; botorch_rs's built-in
subclasses stay native. The smoothed constraint indicator uses `eta` in
the samples' dtype (BoTorch rounds float hyperparameters to float32, bug
10).

**`FixedFeatureAcquisitionFunction`** with t-batched values evaluates the
base acquisition function on the full points (`_construct_X_full`); shared
and per-point values remain native.

## Equivalence

All comparisons use the same models (identical hyperparameters), the same
base samples (BoTorch's samplers' base samples installed in botorch_rs) and
BoTorch's own random draws where BoTorch draws them. Tolerances: values
1e-9, gradients 1e-8 (relative); constructed quantities from the training
data must be identical (worst errors 0). Corrected references for BoTorch's
bugs: `qMultiStepLookahead` gradients from BoTorch under
`propagate_grads(True)` without `fast_pred_var`, finite differences, and
the duplicated-fantasy identity (one fantasy followed by several equals the
same tree with the fantasy duplicated).

Worst errors over all configurations: BoTorch main (da00e0401) and
0.18.1, each under torch 2.13 and 2.14.1. A gradient error of 0 means that
every entry agrees within 1e-14 absolute (the comparisons ignore smaller
differences).

### Rust fixtures (`botorch_core/tests/acquisition_rest_parity.rs`, BoTorch main)

Fixtures of `scripts/fixtures_8b_acquisition_rest.py` (`SingleTaskGP` with
one output (RBF / Matern-5/2, `Normalize`, with and without `Standardize`,
fixed noise), two outputs, `ModelListGP`): values from BoTorch, gradients
from the corrected BoTorch (`propagate_grads(True)`, no `fast_pred_var`),
`make_best_f` corrected, `optimize_acqf` from BoTorch's initial conditions.

| quantity | tolerance | worst error |
|---|---|---|
| qMultiStepLookahead values (2 and 3 steps, `q_1 = 2`, EI / UCB / qEI stages with `best_f` from the model, pending points, weights, transforms, MC objectives, `collapse_fantasy_base_samples=False`) | 1e-9 | 1.2e-14 |
| qMultiStepLookahead gradients (same configurations) | 1e-8 | 0 |
| optimize_acqf of qMultiStepLookahead: restarts below BoTorch's value | 0 | 0 |
| optimize_acqf of qMultiStepLookahead: candidates of agreeing restarts | 1e-3 | 1.2e-7 |

### Live parity (`test_parity_8b_acquisition_rest.py`, 239 tests with `test_8b_acquisition_rest.py`)

| quantity | tolerance | worst error |
|---|---|---|
| constructed `best_f`, `X_baseline`, reference points, MES candidate sets, NIPV points, `project` / `expand`, scalarization weights | 0 (identical) | 0 |
| constructed partitionings, infeasible costs, `best_f` without feasible observations | 1e-9 | 1.0e-15 |
| constructed current values (qKG, qMFKG, qHVKG, qMFHVKG) and qJES optimal samples (`optimize_acqf` / L-BFGS-B) | 1e-6 | 5.0e-10 |
| `get_best_f_analytic`, `get_best_f_mc`, `compute_best_feasible_objective` | 0 (1e-9 without feasible point) | 5.6e-16 |
| `get_infeasible_cost` | 1e-9 | 2.6e-15 |
| acquisition functions built from the constructors (all 33 constructors): values | 1e-9 | 8.3e-12 |
| the same: gradients | 1e-8 | 4.3e-12 |
| `get_acquisition_function` (every name): `best_f` / values / gradients | 1e-9 / 1e-9 / 1e-8 | 2.6e-15 / 1.4e-14 / 5.4e-14 |
| `optimize_objective`: values | 1e-6 | 6.2e-13 |
| `optimize_objective`: candidates | 1e-4 | 2.6e-5 |
| qMultiStepLookahead values (6 models x 8 configurations, BoTorch's classes and argfacs) | 1e-9 | 1.2e-15 |
| qMultiStepLookahead gradients vs corrected BoTorch | 1e-8 | 0 |
| qMultiStepLookahead gradients vs central differences of BoTorch's values | 1e-7 | 2.7e-9 |
| `make_best_f`, `warmstart_multistep`, tree representation, `extract_candidates` | 0 (identical) | 0 |
| `get_induced_fantasy_model` posterior | 1e-9 | 3.7e-15 |
| optimize_acqf (qMultiStepLookahead, SampleReducing subclass): BoTorch's value at botorch_rs's candidates (shortfall) | 1e-6 | 1.3e-16 |
| SampleReducing / LogImprovement subclasses (objectives, transforms, constraints, `fat`, reductions, pending points): values | 1e-9 | 8.0e-15 |
| the same: gradients | 1e-8 | 8.0e-15 |
| FixedFeature with t-batched values: `X_full` / values / gradients | 0 / 1e-9 / 1e-8 | 0 / 2.2e-16 / 0 |
| `_sample_max_value_Thompson`, batched candidate sets | 1e-9 | 9.7e-16 |
| `_sample_max_value_Gumbel`, batched = one batch at a time | 0 | 0 |
| `_sample_max_value_Gumbel`, batched, vs BoTorch (float32 quantiles, phase 8 bug 7) | 1e-6 | 2.4e-8 |

Tolerances above nominal, with measurements: the values of `optimize_acqf`
/ `optimize_objective` and the optimal samples of qJES (L-BFGS-B stops on
rounding-dependent criteria; worst value difference 6.2e-13 and 5.0e-10 on
BoTorch 0.18.1, 1.1e-15 on main); `optimize_objective`'s candidates 1e-4
(BoTorch 0.18.1 stops elsewhere on a flat optimum: candidates 2.6e-5 apart,
values 2.0e-13; on main 3.4e-14); central differences with step 1e-6
(truncation and rounding error, measured 2.7e-9); the Gumbel quantiles,
which BoTorch rounds to float32 (2.4e-8, phase 8). The largest values error
(8.3e-12, analytic EHVI) is the summation order of the hypercells. Restarts
of `optimize_acqf` whose L-BFGS-B runs stop on different iterations are
allowed for one restart per problem (BoTorch 0.18.1 stops one restart
early; its value at botorch_rs's candidates is never lower).

### Randomized sweep (`test_equivalence_8b_acquisition_rest.py`, `-m equivalence`, 48 tests)

12 seeded scenarios (one and two outputs and `ModelListGP`, n = 6-14,
d = 1-3, inferred or fixed noise, transforms, RBF / Matern-5/2), random
look-ahead configurations (1-3 steps, 1-3 fantasies, batch sizes 1-2, stage
value functions, transforms, objectives, pending points, weights, collapse),
Python subclasses and constructors.

| quantity | tolerance | worst error |
|---|---|---|
| qMultiStepLookahead values | 1e-9 | 2.4e-15 |
| qMultiStepLookahead gradients vs corrected BoTorch | 1e-8 | 0 |
| SampleReducing / LogImprovement subclasses: values / gradients | 1e-9 / 1e-8 | 5.6e-14 / 1.1e-13 |
| constructors: `best_f`, `X_baseline` | 0 | 0 |
| acquisition functions from the constructors: values / gradients | 1e-9 / 1e-8 | 1.5e-12 / 3.0e-14 |
| `get_acquisition_function` `best_f` | 1e-9 | 5.6e-16 |

### Test runs

| BoTorch / torch | `pytest botorch_pyo3/tests -m "not equivalence"` | `-m equivalence` |
|---|---|---|
| main, 2.13 | 2809 passed, 27 skipped | 1007 passed, 15 skipped |
| main, 2.14.1 | 2809 passed, 27 skipped | 1007 passed, 15 skipped |
| 0.18.1, 2.13 | 2789 passed, 47 skipped | 1007 passed, 15 skipped |
| 0.18.1, 2.14.1 | 2789 passed, 47 skipped | 1007 passed, 15 skipped |

The six skips of this phase (independent of the version) are look-ahead
configurations on the two-output and `ModelListGP` models: `best_f` from
the model (single-output models only, see "Not ported") and `q_1 = 2` (BoTorch
fails for one t-batch, bug 9, and mixes the blocks of several, bug 3).

## Performance

Value and gradient of 32 candidate sets (`SingleTaskGP`, n = 30, d = 4),
best of five; BoTorch main with one torch thread (and its default, wrong,
gradients: the corrected ones cost more), botorch_rs with 5 Rayon threads.

| workload | BoTorch | botorch_rs | speed-up |
|---|---|---|---|
| qMultiStepLookahead: 2 steps, 4 x 4 fantasies, posterior mean | 68.0 ms | 2.53 ms | 27x |
| qMultiStepLookahead: 3 steps, 3 x 3 x 3 fantasies, posterior mean | 204.8 ms | 6.60 ms | 31x |
| qMultiStepLookahead: 2 steps (`q_1 = 2`), 4 x 4 fantasies, qEI stages (16 inner samples) | 92.7 ms | 3.13 ms | 30x |
| Python subclass of SampleReducingMCAcquisitionFunction (q = 2, 256 samples) | 6.7 ms | 2.25 ms | 3.0x |

The subclass runs its `_sample_forward` in torch on botorch_rs's posterior
samples, so the speed-up is that of the posterior and its backward pass.

## BoTorch bugs found

All reproduced on BoTorch main (da00e0401) and 0.18.1 (torch 2.13) with

```python
import torch
from botorch.models import SingleTaskGP
torch.manual_seed(0)
X = torch.rand(8, 2, dtype=torch.float64)
Y = torch.sin(6 * X).sum(-1, keepdim=True)
model = SingleTaskGP(X, Y).eval()
```

1. **`qMultiStepLookahead` gradients are wrong.** The stage values of the
   fantasy models are computed with detached test caches (and from depth
   two under `fast_pred_var`), so the gradient misses the dependence through
   the fantasy models.
   ```python
   acqf = qMultiStepLookahead(model, batch_sizes=[1], num_fantasies=[3])
   x = torch.rand(1, 4, 2, dtype=torch.float64, requires_grad=True)
   torch.autograd.grad(acqf(x).sum(), x)[0][0, 0]
   # [-0.3442, 0.0510]; central differences [-0.3182, 0.1222]; with
   # settings.propagate_grads(True), fast_pred_var(False): [-0.3182, 0.1222]
   ```
2. **`collapse_fantasy_base_samples=False` with single-output posteriors**
   makes a t-batch's value depend on the other t-batches: the samplers'
   `batch_range_override` is set for multi-output models only, so the
   base samples (shape `(2, 3, 1)`) differ between t-batches.
   ```python
   acqf = qMultiStepLookahead(model, batch_sizes=[1, 1], num_fantasies=[2, 2],
                              collapse_fantasy_base_samples=False)
   x1 = torch.rand(1, 7, 2, dtype=torch.float64)
   acqf(x1), acqf(torch.cat([x1, torch.rand(2, 7, 2, dtype=torch.float64)]))[0]
   # 0.1174 vs 0.1329
   ```
3. **`q_j > 1` with several t-batches** mixes the blocks of different
   t-batches (`get_split_shapes` / reshape in `_step`):
   ```python
   acqf = qMultiStepLookahead(model, batch_sizes=[2, 1], num_fantasies=[2, 2])
   x1 = torch.rand(1, 9, 2, dtype=torch.float64)
   acqf(x1), acqf(torch.cat([x1, torch.rand(2, 9, 2, dtype=torch.float64)]))[0]
   # 0.2736 vs 0.2547
   ```
4. **`make_best_f`** returns the maximum of `model.train_targets`, which are
   outcome-transformed (standardized): `make_best_f(model, X)["best_f"]` is
   1.302 while the best observation is 1.738. With an analytic value
   function, its shape (`batch` instead of `batch x 1`) fails:
   `qMultiStepLookahead(model, [1], [2], valfunc_cls=[None,
   ExpectedImprovement], valfunc_argfacs=[None, make_best_f])` raises
   `AssertionError: Expected the output shape to match either the t-batch
   shape of X, or the model.batch_shape ...`, as does the documented batch
   mode of `ExpectedImprovement(model, best_f=torch.tensor([0.1, 0.2]))` on
   `X` of shape `2 x 1 x 2`.
5. **One fantasy followed by several** fails:
   `qMultiStepLookahead(model, batch_sizes=[1, 1], num_fantasies=[1, 2])`
   on `X` of shape `2 x 4 x 2` raises `RuntimeError: shape '[2, 1, 2, 1]' is
   invalid for input of size 8`.
6. **The input-constructor registry rejects `optimize_objective_kwargs`**:
   `get_acqf_input_constructor(qKnowledgeGradient)(model=model,
   training_data=ds, bounds=[(0, 1)] * 2, optimizer_options={...})` raises
   `TypeError: Unexpected keyword argument optimizer_options when
   constructing input arguments for construct_inputs_qKG` (also qMFKG,
   qHVKG, qMFHVKG): `allow_only_specific_variable_kwargs` filters the
   `**optimize_objective_kwargs` that these constructors take.
7. **`construct_inputs_qSimpleRegret` with constraints and no objective**
   builds `ConstrainedMCObjective(objective=None, ...)`;
   `qSimpleRegret(**construct_inputs_qSimpleRegret(model, constraints=[lambda
   Z: Z[..., 0]], X_baseline=X))(torch.rand(1, 1, 2))` raises `TypeError:
   'NoneType' object is not callable`.
8. **Max-value entropy search with a batched candidate set** concatenates
   the training inputs along dimension 0: `qMaxValueEntropy(model,
   torch.rand(3, 5, 2))` raises `RuntimeError: Sizes of tensors must match
   except in dimension 0`, and with `n = n_train = 8` the candidate set
   silently becomes `(6, 8, 2)` (the training inputs as three more
   "batches") and the max values `(10, 6)`.
9. **Multi-output `qMultiStepLookahead` with one t-batch** fails (as the
   knowledge gradient did, phase 8): with
   `ScalarizedPosteriorTransform([0.5, 0.5])` on a two-output
   `SingleTaskGP`, `X` of shape `1 x 3 x 2` raises `RuntimeError: shape '[2,
   1, 2, 1]' is invalid for input of size 8` (two t-batches work).
10. **Float hyperparameters become float32 buffers** (precision): `best_f`
    (analytic and MC acquisition functions), `beta` (analytic
    `UpperConfidenceBound`), `tau` (`qProbabilityOfImprovement`) are stored
    with `torch.as_tensor(value)`, and `eta` (`ConstrainedMCObjective`,
    `compute_smoothed_feasibility_indicator`) with `torch.full(..., eta)`,
    i.e. in float32 for Python floats, even with float64 models; e.g.
    `qExpectedImprovement(model, best_f=0.2)` uses `0.20000000298` and its
    values are off by 3e-9. The constructors return tensors for `best_f`,
    but pass the user's floats `beta`, `tau` and `eta` through. botorch_rs
    keeps float64; the tests use float32-exact values (`0.25`, `0.75`,
    `2**-10`) so that both libraries see the same numbers.

BoTorch 0.18.1 also fails for `_sample_max_value_Gumbel` with several
candidate batches (fixed on main, #3353); the tests accept that failure on
0.18.1.

### Upstream status

| Bug | Fix |
|---|---|
| 1 | meta-pytorch/botorch#3451 |
| 2, 3 | meta-pytorch/botorch#3452 |
| 4 | `make_best_f`: meta-pytorch/botorch#3391; batched `best_f` in analytic functions: #3453 |
| 5, 9 | cornellius-gp/gpytorch#2804 |
| 6 | meta-pytorch/botorch#3454 |
| 7 | meta-pytorch/botorch#3455 |
| 8 | meta-pytorch/botorch#3456 (rejects batched candidate sets) |
| 10 | not filed: a design decision (see meta-pytorch/botorch#2724, #2725) |
