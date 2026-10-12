# Draft: fantasies, cost models and the hypervolume knowledge gradient

This draft collects the rows and sections that the port of BoTorch's
hypervolume knowledge gradient (qHVKG) and the model infrastructure it needs
(conditioning and fantasies) adds to `README.md` and `docs/api_coverage.md`.
Reference: BoTorch main at da00e0401; the pull requests are
meta-pytorch/botorch#3391, #3392, #3395, #3396, #3399 and #3407.

## Supported API (rows for the README table)

| BoTorch / GPyTorch | botorch_rs | Notes |
|---|---|---|
| `SingleTaskGP.condition_on_observations`, `fantasize` | same methods | `condition_on_observations(X, Y, noise=None)`: new points appended with the same hyperparameters and the fitted transforms (eval mode), GPyTorch's batch rules (`Y` may have one more batch dimension than `X`), explicit noise in the outcome-transformed space; `fantasize(X, sampler, observation_noise=None)`: `n_f` fantasy models (batch shape `n_f x batch`), observations sampled from the posterior with observation noise (the inferred noise level, or the mean training noise of a fixed-noise model), base samples shared over the batch as in BoTorch. Rank-k Cholesky updates of the cached training factor, equal to refactorizing (checked) |
| `ModelListGP.condition_on_observations`, `fantasize` | same methods | single-output sub-models; `X` per output, noise in the original outcome space (meta-pytorch/botorch#3399); `fantasize` with a `ListSampler` (one sampler per output), a shared sampler, or an `evaluation_mask` (`n' x m`: output `i` is fantasized only at its observed points) |
| `botorch.sampling.list_sampler.ListSampler` | `botorch_rs.sampling.ListSampler` | samplers of the same sample shape |
| `botorch.models.cost.AffineFidelityCostModel`, `FixedCostModel` | `botorch_rs.models.cost.*` | `fidelity_weights` (default `{-1: 1.0}`, negative indices), `fixed_cost`; differentiable with `torch.autograd`; `posterior(X)` |
| `botorch.acquisition.cost_aware.InverseCostWeightedUtility`, `GenericCostAwareUtility` | `botorch_rs.acquisition.cost_aware.*` | cost models of botorch_rs, botorch_rs GPs (posterior mean) or torch models; `use_mean`, `cost_objective`, `log`, `X_evaluation_mask`; gradients with respect to `X` and `deltas` |
| `botorch.acquisition.utils.project_to_target_fidelity`, `expand_trace_observations` | same paths | torch operations for tensors (differentiable), Rust otherwise |
| `qHypervolumeKnowledgeGradient` | `botorch_rs.acquisition.multi_objective.*` | `ModelListGP` of single-output models; `num_fantasies` (8), `num_pareto` (10), `sampler` (`ListSampler`), `objective`, `inner_sampler` (32 samples), `X_evaluation_mask`, `X_pending` / `set_X_pending` with `X_pending_evaluation_mask`, `current_value`, `use_posterior_mean`, `cost_aware_utility`, `log`; `get_augmented_q_batch_size`, `extract_candidates`, `construct_evaluation_mask`, `cost_sampler`; exact gradients computed in Rust (also through the fantasy models' covariances) |
| `qMultiFidelityHypervolumeKnowledgeGradient` | same path | `target_fidelities`, `project` (`functools.partial(project_to_target_fidelity, ...)` runs in Rust, other callables on torch tensors), `valfunc_cls` / `valfunc_argfac` (stored, unused as in BoTorch); `expand` raises `NotImplementedError` as in BoTorch |
| `hypervolume_knowledge_gradient._get_hv_value_function`, `_split_hvkg_fantasy_points` | same path | the value function (qEHVI / qLogEHVI over an empty front of the posterior mean or of samples, default 128) |
| `botorch.acquisition.acquisition.OneShotAcquisitionFunction` | same path | |
| `botorch.optim.initializers.gen_one_shot_hvkg_initial_conditions` | `botorch_rs.optim.initializers.*` | `frac_random`, `num_inner_restarts`, `raw_inner_samples`, `eta`, `fixed_features`, linear constraints (qHVKG); target fidelities of qMFHVKG |
| `optimize_acqf` with one-shot acquisition functions | `botorch_rs.optim.optimize_acqf` | augmented `q`, default one-shot initial conditions, `extract_candidates` of the best restart, `return_full_tree`; `gen_batch_initial_conditions(..., fixed_X_fantasies=...)` |

## Equivalence with BoTorch (README section)

### Fantasies and the hypervolume knowledge gradient

**Models and settings covered:**
- `SingleTaskGP` with one and two outputs, inferred and fixed noise,
  `Normalize` / `Standardize`, RBF and Matern-5/2 kernels; conditioning on
  unbatched observations, batched targets and batched inputs; fantasies of
  unbatched and batched inputs, with explicit noise;
- `ModelListGP` fantasies (`ListSampler`, evaluation masks, a shared
  sampler) and conditioning;
- qHVKG and qMFHVKG: posterior mean and posterior samples, `log`,
  `current_value`, pending points with evaluation masks, decoupled
  candidates, all multi-output objectives, inverse-cost-weighted utilities
  (fixed, affine and log costs), target-fidelity projections and `q = 0`;
- the value function, `_split_hvkg_fantasy_points`, cost models,
  utilities and fidelity helpers;
- `optimize_acqf` from identical initial conditions and the steps of
  `gen_one_shot_hvkg_initial_conditions`.

**Where the tests are:**
- Rust fixtures: `botorch_core/tests/mo_hvkg_parity.rs` (BoTorch's values
  in `botorch_core/tests/fixtures/phase5c_hvkg/`, generated by
  `scripts/generate_fixtures.py --phase 5c-hvkg` and, from the pull-request
  branches, `--phase 5c-hvkg-prs`), and BoTorch-independent properties in
  `botorch_core/tests/mo_hvkg_properties.rs`.
- Live Python tests: `test_parity_mo_hvkg.py`, 61 tests, and API tests for
  every array type in `test_mo_hvkg.py` (42 tests).
- A randomized suite: `test_equivalence_mo_hvkg.py` (`-m equivalence`), 48
  tests over 8 random model lists with random conditioning, fantasies and
  HVKG options.

| Quantity | Tolerance | Rust fixtures | Python parity | Randomized suite |
|---|---|---|---|---|
| Conditioned / fantasy models: posterior means | 1e-9 | 2.7e-15 | 2.4e-14 | 5.0e-15 |
| Conditioned / fantasy models: covariances | 1e-9 | 1.2e-14 | 2.7e-14 | 3.4e-14 |
| qHVKG / qMFHVKG values (log values) | 1e-9 | 9.9e-15 (1.3e-15) | 5.0e-15 (8.3e-16) | 5.6e-13 (9.6e-15) |
| qHVKG / qMFHVKG gradients, posterior mean | 1e-8 | 5.4e-14 (all cases) | 3.2e-15 | 7.9e-14 |
| qHVKG / qMFHVKG gradients, posterior samples | 1e-8 | as above | < 1e-14 (absolute) | 2.1e-11 |
| Value function values / gradients | 1e-9 / 1e-8 | 6.7e-16 / 3.9e-15 | 1.5e-15 / 5.0e-14 | |
| Cost models, utilities, projections, trace expansion | 1e-14 | exact | 8.7e-16 (GP cost model), else exact | |
| `_split_hvkg_fantasy_points` (one t-batch) | identical | identical | identical | |

Gradient references with posterior samples are BoTorch's with GPyTorch's
`detach_test_caches(False)`; BoTorch's default gradients are wrong there
(see the migration notes).

**Independent of BoTorch:**
- Conditioning equals refitting on the augmented data with the same
  hyperparameters (worst 1.0e-14 for means, 2.1e-15 for covariances), and
  rank-k updates of the Cholesky factor equal the full factorization.
- Gradients agree with central finite differences (worst 7.7e-9 relative to
  the largest component, Python suite; Rust tests for every option).
- The average of the fantasy posterior means over antithetic base samples
  equals the posterior mean, and qHVKG is non-negative in expectation over
  256 fantasies.
- t-batches are independent and results do not depend on the number of
  threads.

**Optimization.** From identical initial conditions, `optimize_acqf`
restarts reach BoTorch's values (to 4.8e-14): 6 of 6 in the Python test, 5 of
6 in the Rust fixtures (one restart parts from BoTorch's at a flat direction
of dominated fantasy designs; BoTorch's ends higher, 11.02 vs 9.79). The
inner optimization of `gen_one_shot_hvkg_initial_conditions` reaches
BoTorch's value in 18 of 18 restarts, and the candidates initialized with
BoTorch's fantasy designs are identical. End to end, with each library's
own one-shot initial conditions, botorch_rs finds better qHVKG values than
BoTorch's default `optimize_acqf` (BoTorch's joint optimization of the
restarts mixes their fantasy designs); with `batch_limit=1` the two are
statistically indistinguishable (see the benchmark).

## Benchmarks (README section)

`scripts/benchmark_mo_hvkg.py`: a `ModelListGP` of two GPs on DTLZ2 (d = 6,
30 points, fixed hyperparameters). BoTorch runs single-threaded; botorch_rs
with one thread and with `RAYON_NUM_THREADS=6`.

Medians of 5 runs on a 6-core Windows workstation (other jobs running;
timings vary by about 1.5x between sessions):

| Task | BoTorch (1 thread) | botorch_rs (1 thread) | botorch_rs (6 threads) | Speedup |
|---|---:|---:|---:|---:|
| qHVKG values + gradients, 8 sets of 81 points, posterior mean | 108.7 ms | 17.3 ms | 4.9 ms | 6.3x / 22x |
| the same with 32 posterior samples | 1.16 s | 141.0 ms | 26.1 ms | 8.2x / 45x |
| `gen_one_shot_hvkg_initial_conditions` (4 restarts, 4 fantasies x 5 designs) | 1.22 s | 37.1 ms | 7.4 ms | 33x / 165x |
| `optimize_acqf` qHVKG (q = 1, 4 restarts, 4 fantasies x 5 designs) | 9.41 s | 102 ms | 22.8 ms | 92x / 412x |
| the same, BoTorch with `batch_limit=1` | 42.4 s | 102 ms | 22.8 ms | 414x / 1857x |

Values agree to 1.2e-15 (BoTorch's values of the single candidate sets).
The best qHVKG values found (each evaluated on its own candidate set):
BoTorch's default `optimize_acqf` median 0.275 (0.20-0.33), with
`batch_limit=1` median 0.458 (0.38-0.48, 6 runs), botorch_rs median 0.441
(0.42-0.45). Over 12 seeds, cross-optimizing each library's initial
conditions with both optimizers (`batch_limit=1`) gives identical values
(to 8e-9) and indistinguishable distributions (median 0.441 from BoTorch's
initial conditions, 0.440 from botorch_rs's).

## Migration notes and known differences

- **t-batches are independent** (a BoTorch bug, kept fixed in botorch_rs).
  BoTorch's `_split_hvkg_fantasy_points` reshapes the fantasy designs
  `b x (n_f num_pareto) x d` into `n_f x b x num_pareto x d` without
  permuting, so with several t-batches the value of candidate set `b` uses
  fantasy designs of other t-batches (`X_fantasies[i, b]` is not row block
  `i` of `X[b]`). botorch_rs evaluates every t-batch on its own designs:
  values of a single t-batch agree with BoTorch's; t-batched values equal
  BoTorch's values of the single candidate sets. `optimize_acqf` optimizes
  the restarts of a batch jointly (`batch_limit`, by default all restarts),
  so BoTorch optimizes a mixed objective and ranks the restarts by mixed
  values: on the benchmark problem its returned candidate sets have true
  values of 0.20-0.33 (claimed 0.37-0.48), against 0.38-0.48 with
  `batch_limit=1` and 0.42-0.45 for botorch_rs.
- **Exact gradients with posterior samples.** BoTorch's fantasy models
  detach their test caches (GPyTorch's `detach_test_caches`, on by
  default), so with `use_posterior_mean=False` its gradient with respect to
  the candidates misses the dependence of the fantasy covariances on them
  (an absolute error of 0.27 against a largest gradient component of 0.42
  in a small repro; up to 35% relative error in the fixture cases). botorch_rs's gradients are exact; they equal BoTorch's under
  `detach_test_caches(False)` and finite differences.
- **Cost-aware utilities.** `GenericCostAwareUtility` fails in BoTorch's
  qHVKG (`TypeError`: qHVKG passes `X_evaluation_mask`, which its `forward`
  does not accept), and `InverseCostWeightedUtility(use_mean=False)` fails
  there as well; both work in botorch_rs (`use_mean=False` requires a
  deterministic cost model, whose samples are its values).
- **Explicit noise of homoskedastic models** (meta-pytorch/botorch#3396):
  the posterior mean of a model conditioned with explicit `noise` uses that
  noise; BoTorch main uses it for the covariance only (its mean cache is
  recomputed with the inferred noise level).
- **`ModelListGP.condition_on_observations`** (#3399): every sub-model is
  conditioned through its own transforms and the noise is in the original
  outcome space; BoTorch main passes untransformed inputs and transformed
  targets to GPyTorch.
- **Pending points** (#3391): `set_X_pending(X)` without a mask clears the
  previous `X_pending_evaluation_mask`.
- **`AffineFidelityCostModel`** keeps its weights as float64 values; BoTorch
  stores them in torch's default dtype (float32 unless changed), which
  rounds weights that are not representable in float32.
- **Samplers.** qHVKG and `fantasize` take botorch_rs samplers
  (`SobolQMCNormalSampler`, `ListSampler`), whose base samples are shared
  over the batch dimensions as in BoTorch; BoTorch's base samples can be
  assigned to them. The random choices of
  `gen_one_shot_hvkg_initial_conditions` (raw samples, Boltzmann sampling)
  come from botorch_rs's generator, seeded by the `seed` option, instead of
  torch's global generator.
- **Multi-fidelity initial conditions.** BoTorch's
  `gen_one_shot_hvkg_initial_conditions` for qMFHVKG does not resolve
  negative target-fidelity indices and passes `fixed_features` of the full
  space to the inner optimization over the non-fidelity columns;
  botorch_rs resolves both. Linear constraints with qMFHVKG raise
  `UnsupportedError`.
- **Fixed-noise conditioning with batched targets of unbatched inputs**
  fails in GPyTorch ("All tensors must have the same number of
  dimensions"); botorch_rs broadcasts.
- `ModelListGP.fantasize` and qHVKG require single-output sub-models;
  BoTorch also accepts multi-output sub-models in `ModelList.fantasize`.
- qHVKG with `sequential=True` raises BoTorch's error; `expand` (trace
  observations) raises `NotImplementedError` as in BoTorch.

## BoTorch issues found (BoTorch main da00e0401, minimal repros)

Not fixed in any of the pull requests above; botorch_rs keeps BoTorch's
behaviour except where it is clearly wrong (as noted).

1. `_split_hvkg_fantasy_points` mixes the fantasy designs of different
   t-batches (perturbing t-batch 1's designs changes t-batch 0's value,
   2.98 to 2.20); `optimize_acqf`'s default joint optimization of the
   restarts therefore optimizes and ranks mixed values. Fixed in botorch_rs.
2. With `use_posterior_mean=False`, qHVKG's gradient with respect to the
   candidates is wrong under GPyTorch's default `detach_test_caches`.
   Exact in botorch_rs.
3. qHVKG with `GenericCostAwareUtility` raises `TypeError:
   GenericCostAwareUtility.forward() got an unexpected keyword argument
   'X_evaluation_mask'`. Works in botorch_rs.
4. qHVKG with `InverseCostWeightedUtility(use_mean=False)` raises
   `AttributeError: 'EnsemblePosterior' object has no attribute
   'posteriors'` (the cost sampler is a `ListSampler`). Works in botorch_rs
   for deterministic cost models.
5. `qMultiFidelityHypervolumeKnowledgeGradient` stores `valfunc_cls` and
   `valfunc_argfac` but never uses them (kept: documented).
6. `gen_one_shot_hvkg_initial_conditions` for qMFHVKG fails for negative
   target-fidelity indices (`ValueError: Feature dimension d' (3) of input
   must be d - d_f (2).`) and applies `fixed_features` to the wrong columns
   when a fidelity column precedes them (`target_fidelities={0: 1.0}`,
   `fixed_features={1: 0.5}` fixes column 2). Resolved in botorch_rs.
7. GPyTorch's fantasy strategy fails to condition a fixed-noise model on
   batched targets of unbatched inputs. botorch_rs broadcasts.

## API coverage (rows for docs/api_coverage.md)

Newly ported: `acquisition.cost_aware.CostAwareUtility`,
`GenericCostAwareUtility`, `InverseCostWeightedUtility`;
`acquisition.utils.project_to_target_fidelity`,
`expand_trace_observations`; `acquisition.acquisition.OneShotAcquisitionFunction`;
`acquisition.multi_objective.qHypervolumeKnowledgeGradient`,
`qMultiFidelityHypervolumeKnowledgeGradient`; `models.cost.AffineFidelityCostModel`,
`FixedCostModel` (and `models.AffineFidelityCostModel`);
`sampling.ListSampler` (`sampling.list_sampler.ListSampler`);
`optim.initializers.gen_one_shot_hvkg_initial_conditions`;
`exceptions.DeprecationError`; `Model.condition_on_observations` /
`fantasize` of `SingleTaskGP` and `ModelListGP`.

## Not ported yet

- qKnowledgeGradient, qMultiFidelityKnowledgeGradient and
  `gen_one_shot_kg_initial_conditions` (the fantasy machinery,
  `FantasyGp` and the cost utilities are general and reusable for them).
- Fantasies of multi-output sub-models of a `ModelListGP`, of batched
  `ModelListGP` sub-models, and `condition_on_observations` for models
  other than `SingleTaskGP` / `ModelListGP`.
- `InverseCostWeightedUtility(use_mean=False)` with stochastic cost models
  (sampled costs).
- Trace observations (`expand`) in qMFHVKG (not supported by BoTorch).
- Linear constraints in the multi-fidelity one-shot initial conditions.
