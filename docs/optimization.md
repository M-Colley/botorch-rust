# Optimization and candidate generation

The port of BoTorch's optimization and candidate generation (roadmap phase 9):
the supported API, the deviations from BoTorch, the equivalence results and the
BoTorch bugs found. Reference: BoTorch main at da00e0401 and BoTorch 0.18.1,
both with SciPy 1.18 and torch 2.x on CPU.

## Supported API

| BoTorch / GPyTorch | botorch_rs | Notes |
|---|---|---|
| `botorch.optim.optimize_acqf` (additions) | `botorch_rs.optim.optimize_acqf` | `post_processing_func` (applied to all restarts before the best is chosen, values recomputed), `timeout_sec` (split over sequential steps and chunks; L-BFGS-B and Adam stop after the iteration that exceeds it), `gen_candidates` (`gen_candidates_scipy`, `gen_candidates_torch` natively with Adam, or any callable with BoTorch's signature, called from Rust per chunk), `options`: `topn` (with `largest`, `sorted`), `sample_around_best` (with `sample_around_best_sigma`, `_subset_sigma`, `_prob_perturb`), `max_optimization_problem_aggregation_size` (BoTorch's serial path: groups of restarts as one L-BFGS-B problem, run in parallel), `factr` / `pgtol` / `tol` / `max_cor` aliases, `optimizer_options` / `stopping_criterion_options` for `gen_candidates_torch`. All of these work with the one-shot acquisition functions (`qKnowledgeGradient`, `qHypervolumeKnowledgeGradient` and their multi-fidelity variants): post-processing sees and re-evaluates the full trees (`b x q_aug x d`), and `topn` / `sample_around_best` reach the one-shot initial conditions, as in BoTorch. Both of BoTorch's L-BFGS-B paths: independent runs per restart (parallel path) and the aggregated problem (serial path, taken with `timeout_sec` or options outside the parallel path's) |
| `botorch.optim.optimize.optimize_acqf_mixed` | `botorch_rs.optim.optimize_acqf_mixed` | `fixed_features_list` (the combinations run in parallel), `return_best_only`, sequential `q > 1` with the previous candidates as pending points, parameter constraints, `post_processing_func`, `gen_candidates`, `batch_initial_conditions`, `ic_generator`; combinations that fail with `CandidateGenerationError` are skipped with BoTorch's warning. `optimize_acqf_list` uses it for `fixed_features_list` |
| `botorch.optim.optimize.optimize_acqf_discrete` | `botorch_rs.optim.optimize_acqf_discrete` | `choices`, `q` (sequential, pending points), `max_batch_size`, `unique`, `X_avoid`, `inequality_constraints`, `return_acq_values`; batched evaluation in Rust |
| `botorch.optim.optimize.optimize_acqf_discrete_local_search` | `botorch_rs.optim.optimize_acqf_discrete_local_search` | all arguments; the local searches of the restarts run in parallel; random lattice points from botorch_rs's generator (keyword-only `seed`); a replaced `_gen_batch_initial_conditions_local_search` (module attribute) is honored |
| `botorch.optim.optimize.optimize_acqf_cyclic` | `botorch_rs.optim.optimize_acqf_cyclic` | `cyclic_options` (`ExpMAStoppingCriterion` settings), the arguments of `optimize_acqf` |
| `botorch.optim.optimize_mixed.optimize_acqf_mixed_alternating` and its building blocks | `botorch_rs.optim.optimize_mixed.*` (also `botorch_rs.optim`) | discrete, binary and categorical dimensions, continuous relaxation of discrete dimensions with more than `max_discrete_values` values, random subsets of large categorical dimensions, all `options` (`initialization_strategy` `"continuous_relaxation"` / `"random"`, `tol`, `maxiter_*`, `num_spray_points`, `std_cont_perturbation`, `batch_limit`, `init_batch_limit`), fixed features, linear constraints, `post_processing_func`; the starting points run in parallel. Keyword-only `seed`. Also `round_discrete_dims`, `get_nearest_neighbors`, `get_categorical_neighbors`, `get_spray_points`, `sample_feasible_points`, `generate_starting_points` (a replaced module attribute is honored), `discrete_step`, `continuous_step`, `complement_indices`, `complement_indices_like`, `should_use_mixed_alternating_optimizer` and the module constants |
| `botorch.optim.optimize_homotopy.optimize_acqf_homotopy`, `prune_candidates` | `botorch_rs.optim.optimize_homotopy.*` (also `botorch_rs.optim`) | all arguments, including `discrete_dims` / `cat_dims` (mixed or mixed-alternating inner optimizer, as in BoTorch) |
| `botorch.optim.homotopy.*` | `botorch_rs.optim.homotopy.*` | `FixedHomotopySchedule`, `LinearHomotopySchedule`, `LogLinearHomotopySchedule` (torch's `linspace` formula), `HomotopyParameter` (a torch `Parameter`, a model attribute or a `HomotopyValue`), `Homotopy` with callbacks |
| `botorch.optim.initializers.initialize_q_batch`, `initialize_q_batch_nonneg`, `initialize_q_batch_topn`, `sample_points_around_best`, `is_nonnegative` | `botorch_rs.optim.initializers.*` | torch's draws when torch is importable (`torch.multinomial` / `torch.randperm` under the global seed, as in BoTorch), Python's `random` otherwise; `gen_batch_initial_conditions` with `topn` and `sample_around_best` options (natively) |
| `botorch.utils.sampling.sample_truncated_normal_perturbations`, `sample_perturbed_subset_dims`, `batched_multinomial`, `boltzmann_sample` | `botorch_rs.utils.sampling.*` | Sobol or uniform draws, BoTorch's truncated normal by inverse CDF |
| `botorch.generation.gen.gen_candidates_scipy`, `gen_candidates_torch`, `get_best_candidates` | `botorch_rs.generation.*` | `gen_candidates_scipy`: L-BFGS-B (both paths) or SLSQP with constraints, `fixed_features`, `timeout_sec`; `gen_candidates_torch`: Adam natively (torch's update rule, BoTorch's `ExpMAStoppingCriterion` on the loss, `callback`, `fixed_features`, `timeout_sec`), other torch optimizers run BoTorch's loop with torch |
| `botorch.generation.sampling.MaxPosteriorSampling`, `BoltzmannSampling`, `ConstrainedMaxPosteriorSampling`, `SamplingStrategy` | `botorch_rs.generation.sampling.*` (also `botorch_rs.generation`) | `objective`, `posterior_transform`, `replacement`, batched `X`; joint posterior samples with GPyTorch's base-sample layout (`torch.randn` under the global seed when torch is importable) and the maximization in Rust |
| `botorch.optim.fit.fit_gpytorch_mll_torch`, `fit_gpytorch_mll_scipy` | `botorch_rs.optim.fit.*` | native for botorch_rs GPs (all outputs and batch elements in parallel): torch's Adam update and `ExpMAStoppingCriterion`, `step_limit`, `parameters`, `bounds`, `timeout_sec`; L-BFGS-B with BoTorch's options; other optimizers, closures, schedulers and callbacks run BoTorch's generic loop on the botorch_rs closure. `fit_gpytorch_mll(..., optimizer=fit_gpytorch_mll_torch)` retries like BoTorch |
| `botorch.optim.core.OptimizationResult`, `OptimizationStatus`, `scipy_minimize`, `torch_minimize` | `botorch_rs.optim.core.*` | the generic minimizers run SciPy / torch on Python closures (NumPy / torch required) |
| `botorch.optim.stopping.StoppingCriterion`, `ExpMAStoppingCriterion` | `botorch_rs.optim.stopping.*` | torch's float32 weights (bitwise for the defaults) |
| `botorch.optim.closures.ForwardBackwardClosure`, `NdarrayOptimizationClosure`, `get_loss_closure`, `get_loss_closure_with_grads` | `botorch_rs.optim.closures.*` | loss and gradients computed in Rust; native variational ELBOs support actual external DataLoader minibatches and epoch cycling |
| `botorch.optim.utils.columnwise_clamp`, `fix_features`, `get_X_baseline`, `get_parameters`, `get_parameters_and_bounds`, `get_name_filter`, `sample_all_priors`, `TorchAttr` | `botorch_rs.optim.utils.*` | BoTorch's parameter names and order (likelihood first) |
| `botorch.cross_validation.gen_loo_cv_folds`, `batch_cross_validation`, `CVFolds`, `CVResults` | `botorch_rs.cross_validation.*` | batched LOO folds (also `train_Yvar`, multi-output), one batched model fitted per fold in parallel, `observation_noise`, `model_init_kwargs`; `MultiTaskGP` raises `UnsupportedError` as in BoTorch |
| `botorch.exceptions.errors.OptimizationTimeoutError`, `OptimizationGradientError` | `botorch_rs.exceptions.errors.*` | with `current_x` and `runtime` |

## What remains unported (in this area)

- per-t-batch (tensor-valued) fixed features in `gen_candidates_*`, and
  `timeout_sec` on the constrained (SLSQP) path, which runs to completion.
- External-data loss closures for exact MLLs. Variational minibatches are
  supported; see [the native closure protocol](variational-data-loader-closures.md).
- BoTorch's private `_optimize_acqf*` entry points: `OptimizeAcqfInputs` is
  accepted by the building blocks of `optimize_mixed` (`discrete_step`,
  `continuous_step`, `generate_starting_points`, ...) only.
- Batch-shaped inputs of `initialize_q_batch*` (`batch x n x q x d`); the
  `n x q x d` case is supported.

Native batched L-BFGS-B, timeout and NumPy optimization adapters are integrated
([solver details](batched-lbfgsb.md)). Cached `loo_cv`, `efficient_loo_cv` and
`ensemble_loo_cv` are also integrated ([cross-validation](cross-validation.md)).
`SingleTaskGP` families support best-of-all L-BFGS-B and Adam fitting attempts
with rollback and native prior resampling ([scope and tests](fit-attempts.md)).

## Deviations from BoTorch (migration notes)

- **Random numbers.** Raw samples, initial-condition selection, lattice
  points, starting points of the alternating optimizer, spray points and
  perturbations come from botorch_rs's generator (`options["seed"]` or a
  keyword-only `seed`), not torch's global generator. The pure-Python
  helpers (`initialize_q_batch*`, `MaxPosteriorSampling`,
  `BoltzmannSampling`) use torch's generator when torch is importable, so the
  same `torch.manual_seed` gives BoTorch's draws. Module-level hooks
  (`_gen_batch_initial_conditions_local_search`, `generate_starting_points`)
  replaced by the user are honored, as in BoTorch.
- **Initialization options do not change the L-BFGS-B path.**
  `n_thinning`, `topn`, `sample_around_best_*` and the other
  initial-condition options are separated from the solver options, so
  `optimize_acqf` stays on the parallel path (BoTorch main falls back to the
  serial path, bug 3). `ftol`, `max_cor` and `iprint` work on the parallel
  path (bugs 1 and 2; `iprint` and `disp` are ignored).
- **`gen_candidates_torch` with fixed features and bounds** clamps the free
  features to their own bounds (BoTorch fails, bug 4); only Adam runs
  natively (other torch optimizers use BoTorch's loop on torch tensors).
- **`sample_points_around_best`** perturbs the subset dimensions with
  `subset_sigma` (BoTorch uses `sigma`, bug 5), and every point is perturbed
  around itself (BoTorch mixes coordinates of different best points, bug
  6). With `ConstrainedExpectedImprovement` (dict constraints) the best
  points are ranked by the objective (BoTorch raises a `TypeError`, bug 9).
- **`BoltzmannSampling` with batched `X`** standardizes the acquisition
  values over the `N` points of every batch (BoTorch standardizes over the
  batch dimension, which makes the sampling uniform for one batch, bug 7).
  Unbatched inputs are identical.
- **`optimize_acqf_discrete_local_search`** restores the acquisition
  function's pending points also with `return_acq_values=False` (bug 8);
  a NaN neighbor value ends a local search instead of being moved to.
- **`optimize_acqf_cyclic`** keeps a copy of the values for the stopping
  criterion (BoTorch's criterion stores a view that the loop then
  overwrites, bug 11). With the default `minimize=True`, increasing values
  stop the loop as soon as the window is full either way; with
  `minimize=False` the number of cycles can differ.
- **Batched `fit_gpytorch_mll_scipy`** reports `STOPPED` when an output
  reaches the iteration limit (BoTorch reports `FAILURE`, bug 12).
- **`optimize_acqf_mixed`** never selects a restart that is infeasible after
  the projection onto the constraints (BoTorch can, bug 13).
- **`OptimizationResult.runtime`** of the native fits is the wall time of the
  native loop; `message` strings follow BoTorch's.
- **`tol` of the alternating optimizer** is always applied as the
  continuous steps' gradient tolerance (BoTorch main's parallel path); BoTorch
  drops it on the serial path (bug 14). `optimize_acqf` itself mirrors
  BoTorch's option handling on each path.
- **One-shot initial conditions.** `gen_one_shot_kg_initial_conditions`
  and `gen_one_shot_hvkg_initial_conditions` (also as the defaults of
  `optimize_acqf`) pass `topn`, `sample_around_best`, `nonnegative`,
  `n_burnin` and `n_thinning` on to `gen_batch_initial_conditions`, as
  BoTorch does with its `options`; `sample_around_best` draws
  `n * q_aug` points around the best training points. The inner
  optimization of the hypervolume value function receives the `options`
  only (not `post_processing_func`, `gen_candidates` or `timeout_sec`, as in
  BoTorch). Deviation: with `timeout_sec`, the outer call takes the serial
  L-BFGS-B path, and so does that inner optimization (BoTorch's inner
  call has no timeout and can take the parallel path); the initial
  conditions use botorch_rs's random draws in either case.
- **No global warning filters** are changed (BoTorch's
  `minimize_with_timeout` adds an `error` filter to the global list, bug 10).

## Equivalence

Tolerances: values 1e-9 (relative), gradients 1e-8; candidates of
continuous optimizers 1e-8 in the Rust fixture tests and 1e-7 in the live
tests (`optim9_helpers.XTOL`): at a converged optimum the objective is flat,
so gradient round-off of ~1e-15 moves the final iterate by up to
~sqrt(eps); the values at those candidates agree to ~1e-14. Candidates of
discrete optimizers must agree exactly. Every comparison starts from
identical inputs (initial conditions, replaced or recorded random draws).

### Rust fixture tests (`cargo test --test phase9_optim_parity`)

Fixtures: `botorch_core/tests/fixtures/phase9_optim/optim_d3.json`,
`optim_d4.json` (`scripts/generate_fixtures.py --phase 9-optim`, BoTorch
main).

| quantity | tolerance | worst error |
|---|---|---|
| `optimize_acqf_discrete` candidates (q = 1, 3, not unique) | 1e-9 | 0 |
| `optimize_acqf_discrete` values | 1e-9 | 3.1e-15 |
| `optimize_acqf_discrete_local_search` candidates (q = 1, 3) | 1e-9 | 0 |
| `optimize_acqf_discrete_local_search` values | 1e-9 | 2.6e-15 |
| `optimize_acqf_mixed` candidates (q = 1 best and all restarts, q = 2) | 1e-8 | 1.6e-9 |
| `optimize_acqf_mixed` values | 1e-9 | 4.7e-15 |
| `optimize_acqf_mixed_alternating` candidates (recorded starts, q = 1, 2) | 1e-8 | 2.5e-10 |
| `optimize_acqf_mixed_alternating` values | 1e-9 | 3.3e-15 |
| `optimize_acqf_cyclic` candidates | 1e-8 | 2.2e-10 |
| `optimize_acqf_cyclic` values | 1e-9 | 1.2e-15 |
| `gen_candidates_torch` candidates (defaults, custom Adam and stopping settings) | 1e-8 | 6.7e-16 |
| `gen_candidates_torch` values | 1e-9 | 5.7e-16 |
| `gen_candidates_scipy` serial (joint) path candidates | 1e-8 | 1.8e-12 |
| `gen_candidates_scipy` serial path values | 1e-9 | 4.1e-16 |
| `optimize_acqf` with `post_processing_func`: candidates / value | 1e-9 | 0 / 7.5e-16 |
| `ExpMAStoppingCriterion` weights, defaults (`eta = 1`, `n_window = 10`) | bitwise | 0 |
| `ExpMAStoppingCriterion` weights, other settings | 3e-7 | 5.9e-8 |
| `prune_candidates`, `MaxPosteriorSampling` selections, constrained scores | exact | 0 |
| truncated normal perturbations (`sample_around_best`) | 1e-12 | 2.4e-14 |
| `fit_gpytorch_mll_torch`, 200 steps: raw parameters / loss | 1e-9 | 4.5e-13 / 6.2e-15 |
| `fit_gpytorch_mll_torch`, custom learning rate: raw parameters / loss | 1e-9 | 1.8e-13 / 4.4e-14 |
| `fit_gpytorch_mll_torch` until the stopping criterion: loss outside BoTorch's twins | 1e-9 | 0 |
| `fit_gpytorch_mll_scipy`: raw parameters / loss | 1e-9 | 1.4e-11 / 9.9e-15 |
| `batch_cross_validation`: folds | exact | 0 |
| `batch_cross_validation`: fitted raw parameters | 1e-9 | 1.9e-10 |
| `batch_cross_validation`: posterior mean / variance | 1e-9 | 6.6e-11 / 4.7e-11 |

The ExpMA weights are float32 values in BoTorch (torch's default dtype):
torch's vectorized `exp` and `sum` are not correctly rounded, so weights
other than the defaults (hard-coded bitwise) differ by about one float32 ulp
(6e-8 relative); 3e-7 allows a few ulps.

The long Adam fit (`fit_gpytorch_mll_torch` until the stopping criterion,
900-1200 steps) amplifies round-off chaotically: BoTorch on the same data
perturbed by 1e-14..1e-11 ("twins") stops after 884-1063 steps (d = 3). The
test requires botorch-rs's step count and loss to lie within the twins'
range; the 200-step and custom-learning-rate fits are compared directly.

### Live parity tests (`test_parity_9_optim.py`, 59 tests)

Worst error per group of checks against BoTorch main (da00e0401) and
0.18.1 ("checks": number of distinct quantities in the group; "-": not
checked in that version). Full suites: 2600 passed and 23 skipped against
main, 2580 passed and 43 skipped against 0.18.1 (the extra skips are other
phases' main-only cases; the phase-9 files skip only the 7 alternating sweep
scenarios without a continuous dimension).

| quantity | kind | checks | tolerance | worst (main) | worst (0.18.1) |
|---|---|---|---|---|---|
| `optimize_acqf_discrete` pending points restored | selections (exact) | 1 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `discrete_step` | candidates / points | 1 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `discrete_step` | values | 1 | 1e-09 | 1.6e-15 | 1.6e-15 |
| `continuous_step` | candidates / points | 1 | 1e-07 | 1.2e-16 | - |
| `continuous_step` | value shortfall | 1 | 1e-09 | - | 2.2e-16 |
| `continuous_step` | values | 1 | 1e-09 | 6.4e-16 | - |
| `optimize_acqf_discrete` | candidates / points | 10 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `optimize_acqf_discrete` | values | 8 | 1e-09 | 5.4e-16 | 5.4e-16 |
| `optimize_acqf_discrete_local_search` | candidates / points | 5 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `optimize_acqf_discrete_local_search` | values | 5 | 1e-09 | 3.9e-16 | 3.9e-16 |
| `optimize_acqf_mixed`, infeasible combination (SLSQP) | values | 1 | 1e-06 | 1.7e-16 | 1.7e-16 |
| `optimize_acqf_mixed` | candidates / points | 8 | 1e-07 | 4.6e-15 | 1.1e-16 |
| `optimize_acqf_mixed` | values | 8 | 1e-09 | 1.5e-15 | 1.5e-15 |
| `optimize_acqf_mixed_alternating`, own random starts (statistical) | value shortfall | 1 | 1e-06 | 0.0e+00 | 0.0e+00 |
| `optimize_acqf_mixed_alternating` | candidates / points | 6 | 1e-07 | 3.6e-09 | - |
| `optimize_acqf_mixed_alternating` | value shortfall | 6 | 1e-09 | - | 2.2e-16 |
| `optimize_acqf_mixed_alternating` | values | 6 | 1e-09 | 5.2e-16 | - |
| neighbors and rounding | candidates / points | 2 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `optimize_acqf_cyclic` | candidates / points | 6 | 1e-07 | 2.7e-11 | 2.9e-13 |
| `optimize_acqf_cyclic` | values | 6 | 1e-09 | 1.3e-16 | 1.3e-16 |
| `optimize_acqf_homotopy`, schedules | candidates / points | 3 | 1e-07 | 6.0e-17 | 6.0e-17 |
| `optimize_acqf_homotopy`, schedules | values | 4 | 1e-09 | 8.0e-17 | 8.0e-17 |
| `gen_candidates_torch` | candidates / points | 7 | 1e-07 | 5.1e-14 | 5.1e-14 |
| `gen_candidates_torch` | values | 8 | 1e-09 | 1.1e-15 | 1.1e-15 |
| `gen_candidates_scipy` | candidates / points | 5 | 1e-07 | 2.4e-16 | 1.4e-16 |
| `gen_candidates_scipy` | values | 5 | 1e-09 | 6.0e-16 | 3.0e-16 |
| `optimize_acqf` (post-processing, timeout, aggregation, Adam) | candidates / points | 7 | 1e-07 | 2.0e-16 | 2.0e-16 |
| `optimize_acqf` (post-processing, timeout, aggregation, Adam) | values | 7 | 1e-09 | 5.3e-16 | 5.3e-16 |
| `initialize_q_batch*` | candidates / points | 10 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `initialize_q_batch*` | values | 10 | 1e-09 | 0.0e+00 | 0.0e+00 |
| `sample_around_best` points | candidates / points | 1 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `sample_around_best` points | values | 1 | 1e-12 | 7.6e-16 | 7.6e-16 |
| sampling strategies, `get_best_candidates` | selections (exact) | 10 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `fit_gpytorch_mll_torch` | parameters / gradients | 10 | 1e-09 | 7.0e-13 | 7.0e-13 |
| `fit_gpytorch_mll_torch` | values | 7 | 1e-09 | 1.1e-13 | 1.1e-13 |
| `fit_gpytorch_mll_scipy` | parameters / gradients | 6 | 1e-08 | 9.0e-12 | 9.0e-12 |
| `fit_gpytorch_mll_scipy` | values | 6 | 1e-09 | 9.6e-15 | 9.6e-15 |
| stopping criterion, closures, `prune_candidates` | candidates / points | 1 | 0e+00 | 0.0e+00 | 0.0e+00 |
| stopping criterion, closures, `prune_candidates` | parameters / gradients | 3 | 1e-08 | 2.5e-16 | 2.5e-16 |
| stopping criterion, closures, `prune_candidates` | selections (exact) | 1 | 0e+00 | 0.0e+00 | 0.0e+00 |
| stopping criterion, closures, `prune_candidates` | values | 1 | 1e-09 | 0.0e+00 | 0.0e+00 |
| cross-validation | data (exact) | 10 | 0e+00 | 0.0e+00 | 0.0e+00 |
| cross-validation | values | 8 | 1e-09 | 4.6e-11 | 4.6e-11 |

Notes:

- **Value shortfall (0.18.1).** BoTorch 0.18.1 with SciPy 1.18 always takes
  the serial L-BFGS-B path, which drops the `tol` of the alternating
  optimizer's continuous steps (bug 14), so its continuous steps stop at
  SciPy's `gtol = 1e-5` and its candidates differ from main's by up to
  ~2e-8. botorch_rs follows main; against 0.18.1 the check is that
  botorch_rs's value is not below BoTorch's by more than 1e-9 (relative).
- **`optimize_acqf_mixed`, infeasible combination (SLSQP), 1e-6.** The
  constrained path runs SLSQP, which SciPy >= 1.16 solves with a C
  reimplementation (LAPACK least squares) and botorch-rs with Kraft's
  algorithm (`vendor/slsqp`): the iterates agree to the solver's accuracy
  only (the tolerance of the existing constrained-optimization parity tests);
  measured 1.7e-16 here.
- **`optimize_acqf_mixed_alternating` with its own random starting points,
  1e-6 (statistical).** Without replaced starting points the random draws
  differ; the best value over three seeds must reach BoTorch's.
- `gen_candidates_torch` with fixed features and `BoltzmannSampling` with
  batched `X` are compared against corrected references (BoTorch fails or
  samples uniformly, bugs 4 and 7); the cyclic optimizer against BoTorch
  with a copying stopping criterion (bug 11); batched
  `fit_gpytorch_mll_scipy` expects `STOPPED` where BoTorch reports `FAILURE`
  (bug 12).

`optimize_acqf` with a `qKnowledgeGradient` (phase 8) and this phase's
options from identical augmented initial conditions and fantasy base
samples (`test_knowledge_gradient_with_phase9_options`; post-processing,
`timeout_sec`, `gen_candidates_torch`, `return_full_tree` with
post-processing, on both L-BFGS-B paths where BoTorch has them):
candidates ≤ 5.4e-14 (tolerance 1e-7), values ≤ 6.3e-17 (1e-9) against
BoTorch main and 0.18.1.

### Randomized equivalence suite (`test_equivalence_9_optim.py`, `-m equivalence`)

24 seeded scenarios (d = 2-5, n = 8-20, RBF or Matérn-5/2, with or without
`Normalize` and fixed noise): discrete (q = 1-3, `unique`, `X_avoid`, a
linear constraint), mixed (q = 1-2, both L-BFGS-B paths), local search,
mixed alternating, `gen_candidates_torch` (random Adam and stopping
settings), cyclic and `MaxPosteriorSampling`.

| quantity | kind | checks | tolerance | worst (main) | worst (0.18.1) |
|---|---|---|---|---|---|
| `optimize_acqf_discrete` | candidates / points | 3 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `optimize_acqf_discrete` | values | 3 | 1e-09 | 7.1e-15 | 7.1e-15 |
| `optimize_acqf_discrete_local_search` | candidates / points | 3 | 0e+00 | 0.0e+00 | 0.0e+00 |
| `optimize_acqf_discrete_local_search` | values | 3 | 1e-09 | 2.1e-15 | 2.1e-15 |
| `optimize_acqf_mixed` | candidates / points | 4 | 1e-07 | 2.0e-08 | 3.7e-09 |
| `optimize_acqf_mixed` | values | 4 | 1e-09 | 1.8e-15 | 1.8e-15 |
| `optimize_acqf_mixed_alternating` | candidates / points | 2 | 1e-07 | 9.1e-09 | - |
| `optimize_acqf_mixed_alternating` | value shortfall | 2 | 1e-09 | - | 5.3e-16 |
| `optimize_acqf_mixed_alternating` | values | 2 | 1e-09 | 1.1e-15 | - |
| `optimize_acqf_cyclic` | candidates / points | 1 | 1e-07 | 4.9e-09 | 4.4e-10 |
| `optimize_acqf_cyclic` | values | 1 | 1e-09 | 1.5e-15 | 2.2e-15 |
| `gen_candidates_torch` | candidates / points | 1 | 1e-07 | 3.3e-15 | 3.3e-15 |
| `gen_candidates_torch` | values | 1 | 1e-09 | 2.9e-15 | 2.9e-15 |
| sampling strategies, `get_best_candidates` | selections (exact) | 1 | 0e+00 | 0.0e+00 | 0.0e+00 |

The worst candidate error, 2.0e-8 (`optimize_acqf_mixed` with q = 2 on the
parallel path), is the flat-optimum effect described above; the values at
those candidates agree to 1.8e-15. Seven alternating scenarios are skipped
(all dimensions discrete).

## BoTorch bugs found

Reproduced with BoTorch main (da00e0401) and 0.18.1 unless noted (bug 13
with a stand-in for a failed projection, see there). Repro scripts were run
against both versions. Common
setup:

```python
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

torch.manual_seed(0)
X = torch.rand(10, 2, dtype=torch.double)
Y = (X - 0.5).pow(2).sum(-1, keepdim=True)
model = SingleTaskGP(X, Y)
acqf = LogExpectedImprovement(model, best_f=Y.max())
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
```

1. **`ftol` breaks the parallel L-BFGS-B path (main only).** `ftol` is on
   the fast path's whitelist, but `fmin_l_bfgs_b_batched` also receives its
   default `factr`:
   `optimize_acqf(acqf, bounds, 1, 2, 16, options={"ftol": 1e-12})` raises
   `AssertionError: ftol and factr cannot be used together, set factr
   explicitly to None.`
2. **`max_cor` and `iprint` break the parallel path (main only).** Both are
   whitelisted (`get_reasons_against_fast_path`) but not accepted by
   `fmin_l_bfgs_b_batched`: `options={"max_cor": 5}` raises `TypeError:
   fmin_l_bfgs_b_batched() got an unexpected keyword argument 'max_cor'`,
   `options={"iprint": -1}` likewise.
3. **Initialization options silently select the serial path (main only).**
   `options={"n_thinning": 32}` (or `topn`, `sample_around_best_*`, ...) is
   counted as a solver option: `get_reasons_against_fast_path` returns
   `["options={'n_thinning'} are not accepted"]`, so `optimize_acqf` falls
   back to the serial `scipy.optimize.minimize` path without a warning.
4. **`gen_candidates_torch` with `fixed_features` and bounds fails.**
   `gen_candidates_torch(torch.rand(3, 1, 2, dtype=torch.double), acqf,
   bounds[0], bounds[1], fixed_features={0: 0.5})` raises `RuntimeError: The
   expanded size of the tensor (1) must match the existing size (2) at
   non-singleton dimension 2`: the bounds of all `d` features are broadcast
   against the free features only.
5. **`sample_points_around_best` ignores `subset_sigma`.** It passes
   `sigma` to `sample_perturbed_subset_dims`; spying on that call with
   `sample_points_around_best(acqf, n_discrete_points=8, sigma=1e-3,
   bounds=bounds, subset_sigma=0.3, prob_perturb=0.5)` shows `sigma=0.001`.
6. **`sample_perturbed_subset_dims` mixes base points.** The perturbed
   coordinates of each sample come from a different base point than the
   unperturbed ones:
   ```python
   from botorch.utils.sampling import sample_perturbed_subset_dims
   torch.manual_seed(1)
   Xb = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
   out = sample_perturbed_subset_dims(Xb, bounds, n_discrete_points=200, sigma=1e-6, prob_perturb=0.5)
   ((out[:, 0] - out[:, 1]).abs() > 0.5).sum()  # 65 of 200 samples mix (0, 0) and (1, 1)
   ```
7. **`BoltzmannSampling` is uniform for batched `X`.** `standardize`
   normalizes a `batch x N` tensor over the batch dimension, so a single
   batch standardizes to zeros:
   ```python
   from botorch.acquisition import UpperConfidenceBound
   from botorch.generation.sampling import BoltzmannSampling
   ucb = UpperConfidenceBound(model, beta=0.0)
   Xc = torch.linspace(0, 1, 50, dtype=torch.double).unsqueeze(-1).repeat(1, 2)
   flat = BoltzmannSampling(ucb, eta=50.0)(Xc, num_samples=2000)  # best point: 45% of the samples
   batched = BoltzmannSampling(ucb, eta=50.0)(Xc.unsqueeze(0), num_samples=2000)[0]  # best point: 2% (uniform)
   ```
8. **`optimize_acqf_discrete_local_search` leaves `X_pending` set.** With
   `return_acq_values=False` the pending points of the previous candidates
   are not reset: after `optimize_acqf_discrete_local_search(qei, [c, c],
   q=2, num_restarts=2, raw_samples=16, return_acq_values=False)` (with
   `qei = qExpectedImprovement(model, best_f=Y.max())`,
   `c = torch.tensor([0.0, 0.5, 1.0], dtype=torch.double)`), `qei.X_pending`
   holds the two candidates.
9. **`sample_around_best` fails with `ConstrainedExpectedImprovement`.**
   `sample_points_around_best` calls the entries of `acq_function.constraints`,
   which is a dict of bounds for `ConstrainedExpectedImprovement`:
   `optimize_acqf(ConstrainedExpectedImprovement(m2, best_f=0.1,
   objective_index=0, constraints={1: (None, 0.0)}), bounds, 1, 2, 16,
   options={"sample_around_best": True})` with a two-output `m2` raises
   `TypeError: 'int' object is not callable`.
10. **`minimize_with_timeout` changes the global warning filters.** It
    calls `warnings.filterwarnings("error", message="Method .* cannot
    handle", ...)` outside a `catch_warnings` block; after one call the
    `error` filter stays in `warnings.filters` for the whole process.
11. **`optimize_acqf_cyclic` corrupts the stopping criterion's window.**
    `ExpMAStoppingCriterion` stores the first `fvals` as a view, and
    `optimize_acqf_cyclic` updates its value tensor in place:
    ```python
    from botorch.optim.stopping import ExpMAStoppingCriterion
    c = ExpMAStoppingCriterion(n_window=2)
    v = torch.tensor([1.0, 2.0])
    c(v)
    v[0] = 5.0  # what optimize_acqf_cyclic does between evaluations
    c(v)
    c._prev_fvals  # [[5., 2.], [5., 2.]]: the first row should be [1., 2.]
    ```
12. **Batched `fit_gpytorch_mll_scipy` reports `FAILURE` at the iteration
    limit.** The batched path checks `r.get("warnflag", 0) == 1`, but its
    batched L-BFGS-B results carry the flag as `status`: for a two-output
    `SingleTaskGP`, `fit_gpytorch_mll_scipy(mll, options={"maxiter": 3})`
    returns `FAILURE` ("Batched L-BFGS-B: 0/2 outputs converged.") where the
    single-output model returns `STOPPED`.
13. **`optimize_acqf_mixed` can return an infeasible candidate.** It
    optimizes every combination with `return_best_only=False`, where
    restarts that are still infeasible after the projection keep their
    values (only `return_best_only=True` sets them to `-inf`), and then takes
    the maximum. With the projection standing in for a failed one
    (`botorch.optim.optimize.project_to_feasible_space_via_slsqp = lambda X,
    **kw: X`) and a `gen_candidates` that returns one infeasible restart
    `(0.1, 0.1)` with the higher value and one feasible restart `(0.6, 0.6)`
    under `x0 + x1 >= 1`, `optimize_acqf` returns `(0.6, 0.6)` and
    `optimize_acqf_mixed(..., fixed_features_list=[{}])` returns `(0.1, 0.1)`.

14. **`tol` is dropped on the serial L-BFGS-B path.** On the parallel path
    `tol` is used as L-BFGS-B's `pgtol`; on the serial path it is passed to
    `scipy.optimize.minimize` as an L-BFGS-B option, which does not exist
    (SciPy's `OptimizeWarning: Unknown solver options: tol`, swallowed by
    `optimize_acqf`). `optimize_acqf_mixed_alternating` sets `tol` for its
    continuous steps, so on the serial path (always with BoTorch 0.18.1 and
    SciPy 1.18; with `timeout_sec` or other options on main) they stop at
    SciPy's default `gtol = 1e-5`:
    ```python
    from botorch.generation.gen import gen_candidates_scipy
    ics = torch.rand(2, 1, 2, dtype=torch.double)
    gen_candidates_scipy(ics, acqf, bounds[0], bounds[1], options={"tol": 1e-8, "maxfun": 15000})
    # OptimizeWarning: Unknown solver options: tol (main and 0.18.1;
    # 0.18.1 also without maxfun)
    ```

## botorch-rs bug fixed on the way

- **Endless SLSQP line search** (pre-existing in botorch-rs main). The
  vendored `slsqp` crate (NLopt's translation of Kraft's SLSQP) halves the
  step without limit while the merit function is not finite, where Kraft's
  and SciPy's code stop the line search after 10 trials. A NaN search
  direction (inconsistent constraints, e.g. a fixed feature that makes a
  linear constraint unsatisfiable) therefore never ended:
  `optimize_acqf(acqf, [[0.0] * 3, [1.0] * 3], 1, 2, 8, fixed_features={0:
  0.0}, inequality_constraints=[([0, 1], [1.0, 1.0], 1.9)],
  options={"seed": 7})` hung in the feasibility projection. The patch in
  `vendor/slsqp/src/slsqp.rs` (documented in its `Cargo.toml`, regression
  test in `reverse.rs`) ends the line search after 10 trials as SciPy does;
  the call now raises `CandidateGenerationError` in milliseconds, as BoTorch
  does (BoTorch's message comes from its failing projection, "Optimization
  failed: Positive directional derivative for linesearch").
