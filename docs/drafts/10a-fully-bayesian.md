# Phase 10a: fully Bayesian GPs

Draft for README, `docs/api_coverage.md` and `docs/models.md` (phase 10a,
branch `phase10/fully-bayesian`). Reference: BoTorch main (da00e0401) and
the released 0.18.1; both fit fully Bayesian GPs with NumPyro (JAX), not
Pyro.

This is the historical phase-10 audit, rather than the current full-port
inventory. Optional public NumPyro tracing was subsequently implemented and
audited against official revision `b0dd01be6`; see
[`../numpyro-trace-interoperability.md`](../numpyro-trace-interoperability.md).
Native NUTS remains the default fitter. Native fully Bayesian multitask
warping was subsequently implemented and audited against the same pinned
revision; see [the current warping audit](../fully-bayesian-multitask-warping.md).

## Supported API (rows for README)

| BoTorch | botorch_rs | Notes |
| --- | --- | --- |
| `botorch.models.fully_bayesian.SaasFullyBayesianSingleTaskGP` | same path | SAAS prior; the samples form a batched `SingleTaskGP` (batch `num_mcmc_samples`); `GaussianMixturePosterior`; `median_lengthscale`, `num_mcmc_samples`, `batch_shape`, `load_mcmc_samples`, `condition_on_observations`, `construct_inputs`, `train_Yvar`, input warping, `Normalize`/`Standardize` transforms |
| `FullyBayesianSingleTaskGP`, `FullyBayesianLinearSingleTaskGP`, `AbstractFullyBayesianSingleTaskGP` | same path | Matérn-5/2 with a log-normal lengthscale prior; linear kernel with SAAS-style weight variances (`median_weight_variance`) |
| `PyroModel`, `MaternPyroModel`, `SaasPyroModel`, `LinearPyroModel` | same path | native Rust potential and NUTS remain the default; public `sample` / `sample_*` support optional NumPyro tracing (subsequent implementation linked above); `set_inputs`, `postprocess_mcmc_samples`, `load_mcmc_samples`, `get_dummy_mcmc_samples`, `warp` |
| `compute_dists`, `matern52_kernel`, `linear_kernel`, `reshape_and_detach`, `MCMC_DIM` | same path | on torch tensors, NumPy arrays or lists instead of JAX arrays; kernel helpers run in Rust and support arbitrary batch and elementwise scale broadcasting |
| `Model.transform_inputs` | same path | applies an explicit transform or the model's input transform; the identity without a transform |
| `botorch.models.fully_bayesian_multitask.SaasFullyBayesianMultiTaskGP` | same path | `MultitaskSaasPyroModel`, `MultiTaskPyroMixin`, `LatentFeatureMultiTaskPyroMixin`; latent task features with a Matérn task kernel; `posterior` (task column or `output_indices`), `condition_on_observations`, `median_lengthscale`; `fantasize` raises like BoTorch |
| `botorch.fit.fit_fully_bayesian_model_nuts` | `botorch_rs.fit.fit_fully_bayesian_model_nuts` | port of NumPyro's NUTS (dense mass matrix, dual averaging, Stan windows, `init_to_uniform(radius=2)`) in Rust, double precision; `max_tree_depth`, `warmup_steps`, `num_samples`, `thinning`, `seed`; extensions `draws`, `init_params` replay a NumPyro chain |
| `botorch.acquisition.bayesian_active_learning.qBayesianActiveLearningByDisagreement` | same path | BatchBALD on the samples with exact gradients in Rust; `FullyBayesianAcquisitionFunction`, `check_negative_info_gain`, `FULLY_BAYESIAN_ERROR_MSG`, `NEGATIVE_INFOGAIN_WARNING` |
| `GaussianMixturePosterior.mixture_covariance_matrix` | same | also for multi-output (multi-task) posteriors, where BoTorch fails |
| analytic and MC acquisition functions on fully Bayesian GPs | same names | averaged over the samples as BoTorch's `average_over_ensemble_models` (7b machinery), now including the fully Bayesian GPs |
| qEHVI, qNEHVI, qLogEHVI, qLogNEHVI, `prune_inferior_points_multi_objective` on a `ModelListGP` of fully Bayesian GPs | same names | `marginalize_dim=-3`: the members' samples averaged over the MCMC samples |
| `qLogNParEGO` on a `ModelListGP` of fully Bayesian GPs | same | one Chebyshev scalarization normalized by the ensemble's `Y_baseline`; baseline pruning under it |
| `PathwiseThompsonSampling` on fully Bayesian GPs (and lists of them) | same | `ensemble_indices`; `redraw(batch_size, draws=..., ensemble_indices=...)` (extension) |
| `botorch.acquisition.input_constructors.construct_inputs_BALD` | same path | registered for `qBayesianActiveLearningByDisagreement`; the other constructors and `get_acquisition_function` work on fully Bayesian GPs (a `best_f` with one value per MCMC sample, as `get_acquisition_function` computes it, applies sample by sample, as BoTorch's broadcasting) |
| Rust | `botorch_core::fully_bayesian::{PyroModel, PyroModelSpec, PyroKernel, TaskStructure, sample_nuts, NutsOptions, SeededDraws, RecordedDraws, find_valid_initial_params, build_adaptation_schedule}`, `acquisition::QBayesianActiveLearningByDisagreement`, `PathwiseThompsonSampling::new_ensemble`, `QLogNParEGOOptions::y_baseline` | |

## Not ported (and why)

- The NumPyro-program gap recorded in this phase is now resolved by the
  optional trace interface linked above. Custom trace overrides are supported
  with explicit NumPyro inference; the native Rust fitter rejects custom
  sampling overrides that its potential does not implement.
- `load_state_dict` of the fully Bayesian GPs (botorch_rs models have no
  state dicts) and `forward` (the batched prior; raises
  `UnsupportedError`).
- `botorch_community`'s active-learning functions
  (`qBayesianVarianceReduction`, `qBayesianQueryByComittee`,
  `qStatisticalDistanceActiveLearning`): community package, follow-up.
- Acquisition functions that BoTorch does not support on fully Bayesian
  GPs raise `UnsupportedError` (BoTorch fails with other errors): qKG,
  qMFKG, qHVKG, qNIPV (no `fantasize` on the single-task models), qMES,
  GIBBON, qPES (shape errors in BoTorch). qJES raises BoTorch's
  `NotImplementedError` (`FULLY_BAYESIAN_ERROR_MSG` of
  `joint_entropy_search`).
- `PathwiseThompsonSampling` of linear or input-warped fully Bayesian GPs
  and of `EnsembleMapSaasSingleTaskGP`: BoTorch fails on all three (bugs 7
  and 8 below; no random features of linear kernels); botorch_rs raises
  `UnsupportedError`.
- Baseline pruning of ensembles with an objective or constraints
  (`UnsupportedError`, as in 7b).
- Replaying BoTorch's default single-precision NUTS: botorch_rs samples in
  double precision; equivalence with BoTorch's default is statistical (see
  below).

## Deviations (migration notes)

1. **NUTS random numbers.** botorch_rs draws its random numbers from a
   seeded xoshiro256++ generator (`seed`, default 0), not from JAX's PRNG:
   the same seed gives a different chain. `fit_fully_bayesian_model_nuts(...,
   draws=..., init_params=...)` replays the draws recorded from a NumPyro
   run (momenta and the uniforms of NumPyro's Bernoulli trials, see
   `botorch_pyo3/tests/fully_bayesian_helpers.py`), which reproduces
   NumPyro's chain.
2. **Precision.** botorch_rs runs NUTS in double precision; BoTorch runs
   NumPyro in single precision unless JAX's `jax_enable_x64` is set
   (bug 6).
3. **Input warping.** botorch_rs's potential warps the inputs as the fitted
   model does (`Warp`: `x' = clamp(x (1 - 2 eps) + eps, eps, 1 - eps)`,
   `1 - (1 - x'^c1)^c0`); BoTorch's potential uses another warping, so the
   samples of `c0`, `c1` belong to a model other than the one used for
   predictions (bug 1).
4. **Inferred noise.** The loaded noise is `1e-4 + s` for the sampled `s`
   (the noise of the potential); BoTorch loads `max(s, 1e-4)` (bug 2).
5. **Fixed-noise conditioning** repeats the observation noise over the
   samples (BoTorch fails, bug 3).
6. **Multi-task warping.** The initial rejection recorded in this phase is
   superseded: `SaasFullyBayesianMultiTaskGP` now fits and predicts with
   sampled non-task warps natively. The current audit linked above covers
   physical feature selection, transforms, derivatives, and the corrections
   to the official loader (bug 5).
7. **Monte Carlo sampling of linear-kernel GPs.** GPyTorch evaluates the
   posterior covariance of a `LinearKernel` lazily as a low-rank root, so
   BoTorch's base samples have the rank of the kernel as dimension;
   botorch_rs samples with the dense covariance (identical to BoTorch under
   `gpytorch.settings.lazily_evaluate_kernels(False)`, which the tests use).
8. `mixture_covariance_matrix` of multi-output posteriors is computed in
   output-major order (BoTorch fails, bug 4); `PosteriorStandardDeviation`
   averages over the samples (BoTorch fails, 7b bug 2).
9. botorch_rs fixes in shared code: a batched `Warp` inside a per-sample
   `ChainedInputTransform` used all samples' concentrations for every
   sample; qLogNEI's `X_baseline` of an ensemble raised; qLogNParEGO of an
   ensemble normalized every member by its own `Y_baseline`.
10. `jit_compile` is accepted as the pinned reference's legacy no-op.
    `disable_progbar=False` now reports actual completed NUTS transitions;
    see [iteration reporting](../fully-bayesian-nuts-progress.md).
11. Dummy warping samples use `len(indices_to_warp)` concentrations when
    only a subset of the input features is warped; BoTorch uses the full
    input dimension and cannot load its own samples in this case.
12. A custom `PyroModel(eps=...)` passes that margin to the loaded `Warp`,
    keeping the sampler's potential and the fitted model consistent.
    BoTorch resets the loaded transform's margin to its default of 1e-7.
13. Direct calls to a batched `Warp` broadcast the input batch dimensions
    as BoTorch does. The previous port incorrectly required an exact batch
    shape on forward transformations (that restriction belongs to inverse
    transformations).

## Performance

The potential energy and its gradient are analytic (one Cholesky
factorization, `K⁻¹` from the columns of `L⁻¹`, pair loops over the
kernel's derivatives): 24 µs, 93 µs, 0.56 ms and 3.4 ms per evaluation
for `(n, d) = (20, 4), (50, 10), (100, 20), (200, 20)`.
`fit_fully_bayesian_model_nuts` with BoTorch's defaults (512 warmup
steps, 256 samples) on SAAS GPs, wall time on the laptop (BoTorch: NumPyro
in single precision, including JAX's compilation):

| `n`, `d` | BoTorch | botorch_rs |
| --- | --- | --- |
| 20, 4 | 10.8 s | 0.2 s |
| 50, 10 | 15.5 s | 1.2 s |
| 100, 20 | 24.2 s | 8.1 s |

## Equivalence

Tolerances: values 1e-9 and gradients 1e-8 (relative), unless stated. All
model-level comparisons load the same MCMC samples into both libraries.

### NUTS: replay of NumPyro's chains

Re-using an open-source sampler was checked first: no NUTS crate (nuts-rs
or other) is in `~/.cargo/registry` (and downloads were not allowed), and
an existing sampler would not reproduce NumPyro's chains anyway (nuts-rs,
PyMC's sampler, builds and adapts differently: other tree sampling, step
size and mass-matrix adaptation, random numbers). The sampler is therefore a line-by-line port of NumPyro's
`NUTS`/`MCMC` (velocity Verlet, iterative tree building with NumPyro's
checkpoints and turning criterion, biased top-level and uniform subtree
transitions, `max_delta_energy = 1000`, dual averaging with NumPyro's
restarts, Stan adaptation windows, Welford dense covariance with NumPyro's
regularization, the mass-matrix square root by reversed Cholesky).

The tests record every random draw of NumPyro's run in double precision
(momenta and the uniforms of its Bernoulli trials; the recorder recomputes
NumPyro's draws inside `jax.debug.callback`, so NumPyro's samples are
bitwise unchanged), replay them in botorch_rs from NumPyro's initial
parameters and require:

- **the same control flow** (exact): every draw consumed in NumPyro's
  order and every random decision NumPyro's;
- **the same states**: the chains amplify rounding errors (1e-15 at the
  start, up to 1e-9 .. 3e-5 after 30 to 50 iterations), as much as
  BoTorch's own chain does under perturbations of its training inputs
  (twins: training inputs perturbed alternately at 1e-14 and 1e-13, the
  low end of the brief's 1e-14 .. 1e-11 range, which gives a smaller,
  stricter spread; 4 twins in the fixtures, 6 in the live tests and the
  sweep). The state and the step size after every iteration must be within
  10x the twins' largest change (floor 1e-12). The worst measured ratio is
  0.16, i.e. botorch_rs's error is at most 1.6x BoTorch's own rounding
  sensitivity. Fewer twins underestimate the spread: with 3 twins, one
  sweep chain measured 7.9x their spread, with 8 twins 1.3x.
- **multi-task chains** are chaotic from their first iterations: BoTorch's
  twins leave NumPyro's decisions after 4 to 54 draws. botorch_rs must
  follow NumPyro's decisions at least as long as every twin (it follows all
  658 draws of `mt_saas` and 841 of 1290 of `mt_saas_fixed`, as long as
  the best twin), with the states within 10x the twins' spread where the
  twins agree.

| quantity (Rust fixtures, 9 cases) | tolerance | worst |
| --- | --- | --- |
| potential energy | 1e-9 | 5.4e-15 |
| potential gradient | 1e-8 | 2.9e-14 |
| unconsumed recorded draws / differing decisions | 0 (exact) | 0 |
| draws BoTorch's twins follow beyond botorch_rs | 0 (exact) | 0 |
| chain states / (10x twin spread) | 1 | 0.087 |
| step sizes / (10x twin spread) | 1 | 0.097 |
| post-processed samples / (10x twin spread) | 1 | 0.069 |

| quantity (live, BoTorch main) | tolerance | worst |
| --- | --- | --- |
| potential energy / gradient | 1e-9 / 1e-8 | 2.0e-15 / 6.3e-15 |
| multi-task potential energy / gradient | 1e-9 / 1e-8 | 1.3e-15 / 6.4e-14 |
| NUTS replay: unconsumed draws, differing decisions | 0 | 0 |
| NUTS replay: chain states / (10x twin spread) | 1 | 0.039 |
| NUTS replay: loaded samples / (10x twin spread) | 1 | 0.045 (noise) |
| multi-task replay: draws the twins follow beyond botorch_rs | 0 | 0 |

| quantity (sweep, `-m equivalence`) | tolerance | worst |
| --- | --- | --- |
| 12 random potentials / gradients | 1e-9 / 1e-8 | 1.6e-15 / 9.2e-15 |
| 6 random replays: draws the twins follow beyond botorch_rs | 0 | 0 |
| 6 random replays: states / (10x twin spread) | 1 | 0.16 |

**Statistical comparison with BoTorch's default.** BoTorch's default
single-precision chains cannot be replayed in double precision. Two chains
of each library (warmup 150, 300 samples) on three models are compared by
the standardized difference of the posterior means of every unconstrained
hyperparameter, with Monte-Carlo standard errors from effective sample
sizes (`|z| <= 4.5`, a false-positive rate of about 7e-6 per parameter for
normal estimates, 5e-5 per test). Worst `|z|` = 1.04 (0.23 of the
threshold).

### Models and acquisition functions

| quantity (Rust fixtures) | tolerance | worst |
| --- | --- | --- |
| MLL of every sample / gradient | 1e-9 / 1e-8 | 5.9e-15 / 3.1e-14 |
| q=1 posterior mean / variance / noisy variance | 1e-9 | 7.6e-15 / 1.8e-14 / 1.6e-14 |
| q=1 d mean / dx, d variance / dx | 1e-8 | 4.5e-15 / 3.0e-14 |
| joint posterior mean / covariance / noisy covariance | 1e-9 | 1.1e-14 / 2.1e-14 / 1.8e-14 |
| joint posterior vector-Jacobian product | 1e-8 | 3.4e-15 |
| qBALD values (relative, floor 1e-3) / gradients (floor 1e-4) | 1e-9 / 1e-8 | 8.9e-13 / 1.3e-11 |

The BALD floors are absolute 1e-12: BALD is a difference of entropies of
order one, and where the samples coincide (`saas_d3`, whose short chain
stays at one state after the warmup) it vanishes, with BoTorch's values
and gradients at rounding noise (at most 2.2e-16 and 1.2e-15).

| quantity (live, BoTorch main) | tolerance | worst |
| --- | --- | --- |
| posterior mean / variance / covariance (per sample) | 1e-9 | 5.3e-13 / 3.5e-14 / 3.5e-14 |
| mixture mean / variance / covariance | 1e-9 | 1.7e-13 / 1.2e-12 / 1.2e-12 |
| mixture quantiles (BoTorch's bisection tolerance 1e-6) | 1e-6 | 2.2e-13 |
| MLL of every sample / gradient | 1e-9 / 1e-8 | 7.4e-14 / 2.2e-13 |
| conditioned models (posteriors, mixtures) | 1e-9 | 6.8e-13 |
| analytic acquisition values / gradients | 1e-9 / 1e-8 | 2.0e-13 / 2.1e-12 |
| MC acquisition values / gradients (qEI, qLogEI, qNEI, qLogNEI, qPI, qSR, qUCB) | 1e-9 / 1e-8 | 8.9e-12 / 1.7e-11 |
| `prune_inferior_points` kept points | 0 | 0 |
| qBALD values / gradients | 1e-9 / 1e-8 | 7.0e-15 / 7.1e-14 |
| multi-task posteriors (task column, output tasks), conditioning | 1e-9 | 1.8e-13 |
| multi-task qLogEI / qLogNEI values / gradients | 1e-9 / 1e-8 | 5.6e-15 / 1.6e-14 |
| mixture covariance of several outputs | 1e-9 | 4.7e-14 |
| qEHVI / qNEHVI / qLogEHVI / qLogNEHVI on lists | 1e-9 / 1e-8 | 1.3e-13 / 1.2e-13 |
| `prune_inferior_points_multi_objective`, pruned baselines | 0 | 0 |
| qLogNParEGO on lists: `Y_baseline`, values, gradients | 1e-9 / 1e-8 | 2.9e-15 / 6.4e-15 / 1.1e-13 |
| PathwiseThompsonSampling (SAAS, Matérn, lists) values / gradients | 1e-9 / 1e-8 | 1.3e-14 / 1.2e-14 |
| `PyroModel.warp`, kernel helpers, `median_lengthscale` | 1e-9 | 2.0e-16 |
| `construct_inputs_BALD` kwargs (exact) and qBALD values / gradients from them | 0; 1e-9 / 1e-8 | 0; 6.7e-15 / 3.2e-14 |
| qLogEI / qLogNEI constructors on fully Bayesian GPs: `best_f`, `X_baseline` (exact), values / gradients | 0; 1e-9 / 1e-8 | 0; 5.0e-14 / 2.2e-13 |
| `get_acquisition_function` (qLogEI, qLogNEI, qUCB): per-sample `best_f`, `X_baseline`, values / gradients | 1e-9; 0; 1e-9 / 1e-8 | 2.7e-16; 0; 7.6e-16 / 1.7e-13 |

| quantity (sweep: 30 random models) | tolerance | worst |
| --- | --- | --- |
| posterior mean / variance / covariance | 1e-9 | 1.0e-12 / 2.8e-13 / 2.8e-13 |
| mixture mean / variance | 1e-9 | 3.3e-13 / 2.5e-12 |
| MLL / gradient | 1e-9 / 1e-8 | 5.5e-13 / 2.8e-12 |
| LogEI, qLogEI values / gradients | 1e-9 / 1e-8 | 3.4e-12 / 3.1e-11 |
| qBALD values (absolute 1e-12) / gradients (floor 1e-12) | 1e-9 / 1e-8 | 1.5e-12 / 2.5e-11 |
| qLogNEI values / gradients over the training baseline, / (10x BoTorch's twin spread at 1e-14 .. 1e-11, at least nominal) | 1 | 0.069 / 0.15 |

The qLogNEI baseline posteriors are nearly singular: BoTorch's own values
move by up to 4.8e-10 and its gradients by 1.1e-8 under input perturbations
of 1e-14 .. 1e-11 (measured per case), which sets the tolerance of that
case.

All live tests pass against BoTorch main and 0.18.1, under torch 2.13 and
torch 2.14.1 (the scratch venv has no JAX: its runs append the main
environment's site-packages, with JAX and NumPyro, after the venv's own
packages; nothing was downloaded).

## BoTorch bugs found (minimal reproductions)

All reproduce on BoTorch main (da00e0401) and 0.18.1.
Bug 4 is fixed on the local M-Colley fork's `fix/audit-correctness-fixes`
branch (c98dd9452); the parity test checks that implementation directly
when available. This does not claim that the fix has landed upstream.

1. **The NUTS potential and the fitted model warp the inputs
   differently.** `PyroModel._warp_jax` computes `1 - (1 - x^c0)^c1` on
   inputs clipped to `[eps, 1 - eps]`; the `Warp` transform loaded from the
   same sample computes `1 - (1 - x'^c1)^c0` on `x' = x (1 - 2 eps) + eps`
   (`kumaraswamy_warp`). The concentrations are sampled for one model and
   used in another:

   ```python
   import torch, jax.numpy as jnp, numpy as np
   from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
   torch.set_default_dtype(torch.float64)
   X, Y = torch.rand(8, 2), torch.randn(8, 1)
   m = SaasFullyBayesianSingleTaskGP(X, Y, use_input_warping=True)
   c0, c1 = torch.tensor([[2.0, 0.5]]), torch.tensor([[0.5, 3.0]])
   m.load_mcmc_samples({"mean": torch.zeros(1), "noise": torch.ones(1), "outputscale": torch.ones(1),
                        "lengthscale": torch.ones(1, 2), "c0": c0, "c1": c1})
   nuts_warp = m.pyro_model._warp_jax(jnp.asarray(X.numpy()), jnp.asarray(c0[0].numpy()), jnp.asarray(c1[0].numpy()))
   print((torch.as_tensor(np.asarray(nuts_warp)) - m.input_transform(X)[0]).abs().max())  # 0.9
   ```

   botorch_rs uses the `Warp` convention in the potential.

2. **The inferred noise is loaded without the 1e-4 of the NUTS model.**
   `sample_noise` returns `MIN_INFERRED_NOISE_LEVEL + s` to the likelihood
   of the potential, but `load_mcmc_samples` sets the likelihood's noise to
   `s` clamped at `GreaterThan(1e-4)`: samples below 1e-4 become 1e-4, the
   others lose the 1e-4.

   ```python
   import torch
   from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
   X, Y = torch.rand(8, 2, dtype=torch.float64), torch.randn(8, 1, dtype=torch.float64)
   m = SaasFullyBayesianSingleTaskGP(X, Y)
   m.load_mcmc_samples({"mean": torch.zeros(2, dtype=torch.float64),
                        "noise": torch.tensor([1e-5, 0.5], dtype=torch.float64),
                        "outputscale": torch.ones(2, dtype=torch.float64),
                        "lengthscale": torch.ones(2, 2, dtype=torch.float64)})
   print(m.likelihood.noise.flatten().tolist())  # [0.0001, 0.5]; the potential used [1.1e-4, 0.5001]
   ```

3. **`condition_on_observations` of fixed-noise fully Bayesian GPs
   fails**: the noise of the training data is batched over the samples, the
   new noise is not.

   ```python
   import torch
   from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
   torch.set_default_dtype(torch.float64)
   X, Y, Yvar = torch.rand(8, 2), torch.randn(8, 1), torch.full((8, 1), 0.01)
   m = SaasFullyBayesianSingleTaskGP(X, Y, Yvar)
   m.load_mcmc_samples({"mean": torch.zeros(3), "outputscale": torch.ones(3), "lengthscale": torch.ones(3, 2)})
   m.eval(); m.posterior(torch.rand(2, 2))
   m.condition_on_observations(torch.rand(2, 2), torch.randn(2, 1), noise=torch.full((2, 1), 0.01))
   # RuntimeError: expand(torch.DoubleTensor{[3, 8]}, size=[8]) ...
   ```

4. **`GaussianMixturePosterior.mixture_covariance_matrix` fails for
   multi-output posteriors** (adds `q x q` mean outer products to
   `(q m) x (q m)` covariances):

   ```python
   import torch
   from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
   torch.set_default_dtype(torch.float64)
   X = torch.cat([torch.rand(8, 2), torch.tensor([0.0, 1.0] * 4).unsqueeze(-1)], -1)
   m = SaasFullyBayesianMultiTaskGP(X, torch.randn(8, 1), task_feature=-1)
   m.load_mcmc_samples(m.pyro_model.get_dummy_mcmc_samples(num_mcmc_samples=3, dtype=torch.float64))
   m.eval()
   m.posterior(torch.rand(4, 2)).mixture_covariance_matrix  # RuntimeError
   ```

5. **`SaasFullyBayesianMultiTaskGP` drops the input warping of its
   `MultitaskSaasPyroModel`**: NUTS samples `c0`, `c1` with warped inputs,
   `load_mcmc_samples` discards the returned `Warp`, and predictions use
   unwarped inputs.

   ```python
   import torch
   from botorch.models.fully_bayesian_multitask import MultitaskSaasPyroModel, SaasFullyBayesianMultiTaskGP
   torch.set_default_dtype(torch.float64)
   X = torch.cat([torch.rand(8, 2), torch.tensor([0.0, 1.0] * 4).unsqueeze(-1)], -1)
   pm = MultitaskSaasPyroModel(use_input_warping=True)
   m = SaasFullyBayesianMultiTaskGP(X, torch.randn(8, 1), task_feature=-1, pyro_model=pm)
   s = pm.get_dummy_mcmc_samples(num_mcmc_samples=2, dtype=torch.float64)
   print(sorted(s))  # [..., 'c0', 'c1', ...]
   m.load_mcmc_samples(s)
   print(getattr(m, "input_transform", None))  # None
   ```

   Reconfirmed on official revision `b0dd01be6`. The native port now keeps
   each sampled warp in training and prediction, with analytic query
   derivatives. The current warping audit above also documents the official
   loader's full-input dimension and feature-index errors.

6. **NUTS runs in single precision for double-precision models**
   (possibly intended): JAX's `jax_enable_x64` is never set, so
   `train_X_jax` and the whole chain are float32 although the model is
   float64 (Cholesky factors of `n x n` kernel matrices in single
   precision). Repro: `SaasFullyBayesianSingleTaskGP(X64, Y64).pyro_model`
   after `set_inputs` has `train_X_jax.dtype == float32`.

7. **`PathwiseThompsonSampling` fails on fully Bayesian GPs with input
   warping** (the batched `Warp` of the paths):

   ```python
   import torch
   from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
   from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
   torch.set_default_dtype(torch.float64)
   X, Y = torch.rand(8, 2), torch.randn(8, 1)
   m = SaasFullyBayesianSingleTaskGP(X, Y, use_input_warping=True)
   m.load_mcmc_samples({"mean": torch.zeros(3), "noise": torch.ones(3), "outputscale": torch.ones(3),
                        "lengthscale": torch.ones(3, 2), "c0": torch.ones(3, 2), "c1": torch.ones(3, 2)})
   m.eval()
   PathwiseThompsonSampling(m)(torch.rand(2, 2, 2))  # RuntimeError (without warping: fine)
   ```

8. **`PathwiseThompsonSampling` fails on `EnsembleMapSaasSingleTaskGP`**
   for `1 < q != num_taus` (its kernel has no batch shape, so the path
   weights are not batched over the ensemble; for `q == num_taus` the
   weights broadcast against the members instead of the paths):

   ```python
   import torch
   from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
   from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP
   torch.set_default_dtype(torch.float64)
   X, Y = torch.rand(10, 4), torch.randn(10, 1)
   m = EnsembleMapSaasSingleTaskGP(X, Y, num_taus=3).eval()
   PathwiseThompsonSampling(m)(torch.rand(5, 2, 4))  # RuntimeError for q = 2
   ```

9. **Dummy samples fail to load with subset input warping.**
   `_common_dummy_samples` returns `c0` and `c1` with shape
   `num_samples x ard_num_dims`; `load_mcmc_samples` reshapes them to
   `num_samples x 1 x len(indices_to_warp)`. These shapes differ when a
   subset is warped. Both SAAS/Matérn and linear models are affected.
   Use the number of warped features when constructing dummy concentrations.

10. **Loading warping samples loses a custom `eps`.**
    `SaasPyroModel(use_input_warping=True, eps=0.02)` and
    `LinearPyroModel` create their loaded `Warp` without `eps=self._eps`,
    so predictions use 1e-7. Pass the configured margin to the transform.
    Bugs 9 and 10 also reproduce on the local M-Colley fork's
    `fix/audit-correctness-fixes` branch (c98dd9452).

    ```python
    import torch
    from botorch.models.fully_bayesian import SaasPyroModel
    pm = SaasPyroModel(use_input_warping=True, indices_to_warp=[1], eps=0.02)
    pm.set_inputs(torch.rand(6, 3, dtype=torch.float64), torch.randn(6, 1, dtype=torch.float64))
    samples = pm.get_dummy_mcmc_samples(2, dtype=torch.float64)
    print(samples["c0"].shape)  # [2, 3]; loading raises: expected [2, 1, 1]
    samples["c0"] = samples["c1"] = torch.ones(2, 1, dtype=torch.float64)
    print(pm.load_mcmc_samples(samples)[-1]._eps)  # 1e-7; should be 0.02
    ```

Known from earlier phases and seen again: re-chaining the input
transforms when re-loading fully Bayesian samples (fixed on the
M-Colley fork, commit 6329e9fb1), `PosteriorStandardDeviation` on ensembles
(7b bug 2).

## Continuation audit (2026-10-20)

The local reference for this audit is the M-Colley BoTorch fork's
`fix/audit-correctness-fixes` branch at c98dd9452, torch 2.13.0, Python
3.13.12. This reference includes fixes not asserted to be upstream.

- The original phase-10a API and parity run passed 200 tests; one stale
  assertion required bug 4 to remain present. The revised test validates
  the fixed covariance directly when available and always checks the
  covariance assembled independently from the mixture components.
- 45 focused kernel, warping, input dispatch and mixture covariance tests
  pass after the fixes. New batched kernel errors are at most 1.23e-16;
  direct `Warp` broadcast outputs match the reference exactly.
- 13 existing warp and batched-transform API/parity tests pass.
- Rust formatting, Ruff, and `cargo check -p botorch_pyo3 --release --offline`
  pass. A release extension build succeeds.
  A fresh debug fixture build initially exhausted the disk; the generated
  debug crate cache was cleaned and subsequent builds use release mode.
