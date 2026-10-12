# Draft README additions: pathwise sampling and information-theoretic multi-objective acquisition functions

This draft collects the README changes for the `mo/entropy-search` branch:
rows for "Supported API", the changes to "What is not ported yet", a section
for "Equivalence with BoTorch", benchmark rows, and the migration notes and
known differences. Numbers come from BoTorch main at da00e04 (the released
0.18.1 gives the same results for every compared quantity) with GPyTorch
1.15.2, linear_operator 0.6.1, PyTorch 2.13 (CPU) and Python 3.13 on Windows
11.

## Supported API (rows to add)

| BoTorch / GPyTorch | botorch_rs | Notes |
|---|---|---|
| `botorch.sampling.pathwise.draw_matheron_paths` | same path | `SingleTaskGP` (one or several outputs) and `ModelListGP` (also of multi-output models); any `sample_shape`; `prior_sampler=draw_kernel_feature_paths` (also `functools.partial(..., num_features=k)`), `update_strategy=gaussian_update`; keyword-only `seed` or explicit `draws` (one `(frequencies, weights, noise)` per output, e.g. BoTorch's); returns a `MatheronPath` (a `ModelListGP`: a `PathList` of them) whose values have BoTorch's shapes (`sample_shape` broadcast with the input batch shape, outputs as the last batch dimension), with `["prior_paths"]`, `["update_paths"]`, their `weight` and `feature_map.weight`, and exact gradients for torch inputs |
| `botorch.sampling.pathwise.draw_kernel_feature_paths`, `gaussian_update` | same paths | random Fourier features of RBF and Matérn-5/2 kernels with or without `ScaleKernel`, `num_features`; the exact Gaussian update (inferred or fixed noise) with keyword-only `noise` or `seed` |
| `botorch.sampling.pathwise.PathDict`, `PathList`, `SamplePath`, `GeneralizedLinearPath`, `MatheronPath`, `get_matheron_path_model`; `botorch.models.deterministic.MatheronPathModel` | same paths | `PathDict` and `PathList` combine any paths (`join`, transforms); `MatheronPathModel(model, sample_shape, ensemble_as_batch, seed)`, `posterior(X).mean` of shape `[sample_shape x] batch x n x m`, `output_indices` |
| `botorch.acquisition.multi_objective.utils.sample_optimal_points`, `random_search_optimizer`, `compute_sample_box_decomposition` | same paths | the default random search runs in Rust (paths drawn and optimized in parallel); custom `optimizer`s receive one `MatheronPathModel` at a time; `random_search_optimizer` accepts any model with `posterior(X).mean`; keyword-only `seed`, `num_features` and explicit `draws` and `candidates`; all three partitionings, `maximize`, `num_constraints` |
| `botorch.acquisition.multi_objective.max_value_entropy_search.qLowerBoundMultiObjectiveMaxValueEntropySearch` | same path | `hypercell_bounds`, `X_pending` / `set_X_pending`, `estimation_type` (`"0"`, `"LB"`, `"LB2"`, `"MC"`), `num_samples`, and a keyword-only `sampler` (for explicit base samples); values and exact gradients in Rust |
| `botorch.acquisition.multi_objective.joint_entropy_search.qLowerBoundMultiObjectiveJointEntropySearch`, `LowerBoundMultiObjectiveEntropySearch`, `_compute_entropy_noiseless`, `_compute_entropy_upper_bound`, `_compute_entropy_monte_carlo` | same path | as above, with `pareto_sets` and `pareto_fronts`; the conditioning on the sampled optima is an exact rank update of the posterior (meta-pytorch/botorch#3399, see the migration notes) |
| `botorch.acquisition.multi_objective.predictive_entropy_search.qMultiObjectivePredictiveEntropySearch`, `log_cdf_robust` | same path | `pareto_sets`, `maximize`, `X_pending`, `max_ep_iterations`, `ep_jitter`, `test_jitter`, `threshold`; BoTorch's EP (damping, fallbacks, convergence test) in Rust, values and exact gradients of the test-time update; `ep_iterations`, `ep_damping` and `cached_factors` describe the result of EP |

## What is not ported yet (changes)

- Under **Models**, "deterministic, ensemble and cost models" becomes
  "deterministic models other than `MatheronPathModel` (`GenericDeterministicModel`,
  `AffineDeterministicModel`, `PosteriorMeanModel`, `FixedSingleSampleModel`),
  ensemble and cost models".
- Under **Single-objective acquisition functions**, "pathwise Thompson
  sampling" stays (the single-objective `qMaxValueEntropy`, GIBBON,
  `qJointEntropySearch` and `qPredictiveEntropySearch` are not ported), but
  the paths it needs are now available.
- Under **Multi-objective optimization**, remove "multi-objective entropy
  search (max-value, joint and predictive)" and the bullet on
  `compute_sample_box_decomposition`, `random_search_optimizer` and
  `sample_optimal_points`. Add: "`MCSamplerMixin`'s `get_posterior_samples` and
  `sample_shape` on MES and JES".
- Under **Sampling and utilities**, "and pathwise (Matheron) sampling" becomes:
  "pathwise sampling of variational and other approximate GPs, ensembles and
  batched models; `gen_kernel_features`, `KernelEvaluationMap` and the
  transforms of `botorch.sampling.pathwise.utils` (they are part of the paths
  computed in Rust); constructing `GeneralizedLinearPath` or `MatheronPath`
  objects from feature maps and weights (paths come from the samplers)".
- Coverage (`scripts/api_coverage.py`): 186 of 773 public classes and
  functions (24%, from 165); `acquisition.multi_objective` 23 of 33 (70%),
  `models` 5 of 66, `sampling.pathwise` 11 of 25 (44%).

## Equivalence with BoTorch (new section)

### Pathwise sampling and information-theoretic acquisition functions

BoTorch draws the random numbers of its paths (Fourier frequencies, prior
weights, update noise) and the candidate sets of its random search from
torch's global generator. The comparisons pass BoTorch's draws to
botorch-rust, which then reproduces BoTorch's paths, sampled optima and
acquisition values; with its own seeds botorch-rust draws different numbers
from the same distributions.

**Models and settings covered:**
- single- and multi-output `SingleTaskGP`s (RBF, Matérn-5/2, `ScaleKernel`,
  `Normalize`, with and without `Standardize`, inferred and fixed noise) and
  `ModelListGP`s of single- and multi-output models;
- sample shapes `()`, `(3,)`, `(2, 2)` and inputs with batch dimensions that
  broadcast (or fail to broadcast, with the same error) with them;
- `maximize` true and false, one and several objectives, random searches that
  find too few optimal points (the same error);
- the estimates "0", "LB", "LB2" and "MC" (identical base samples), q = 1, 2,
  3, pending points and t-batches;
- PES with converging and non-converging EP (250 iterations), `maximize`
  true and false.

**Where the tests are:**
- Rust fixtures: `botorch_core/tests/mo_entropy_parity.rs` (five models,
  7 tests over 20 MES/JES cases and 3 PES cases per model, and BoTorch's EP
  iterates step by step), `botorch_core/tests/mo_entropy_properties.rs`
  (8 tests independent of BoTorch).
- Live Python tests: `test_parity_mo_entropy.py`, 102 tests.
- A randomized suite: `test_equivalence_mo_entropy.py`, 60 tests over 12
  scenarios (m = 1, 2, 3, n = 6-14, d = 1-3, random noise, transforms,
  kernels and input ranges, three `ModelListGP`s) with random sample shapes,
  numbers of features, test and pending points, q and partitionings (4
  skipped: BoTorch's path models reject the list of a multi-output model).
- API tests without BoTorch: `test_mo_entropy.py`, 35 tests.

The entropy estimates are differences of entropies of order one, so their
rounding errors are amplified; where BoTorch's own values change by more than
the nominal tolerance under a 1e-14 relative perturbation of the training
inputs (its "noise floor"), the tolerance is ten times that change. For PES,
the noise floor also includes BoTorch's change under last-bit noise in `exp`
and `erf` (EP evaluates `log1p(-exp(s))` near `s = 0`, which amplifies a
one-ulp difference to percent-level changes of the EP factors).

| Quantity | Tolerance | Rust fixtures | Python parity | Randomized suite |
|---|---|---|---|---|
| Matheron path values / gradients | 1e-9 / 1e-8 | 1.3e-15 / 2.9e-15 | 1.3e-15 / 2.2e-15 | 2.6e-15 / 1.6e-14 |
| Prior and update paths, `gaussian_update`, path weights | 1e-9 | 4.5e-15 | 4.4e-15 | 4.3e-15 |
| `MatheronPathModel` values | 1e-9 | | 7.0e-16 | 3.3e-15 |
| Sampled Pareto sets, random search inputs, box decompositions | identical | identical | identical | identical |
| Sampled Pareto fronts | 1e-9 | 7.6e-16 | 8.4e-16 | 1.2e-15 |
| `_compute_entropy_*` values / gradients, `log_cdf_robust` | 1e-12 | 5.7e-16 / exact | 2.1e-16 | |
| MES values / gradients (unscaled; worst noise floor) | 1e-9 / 1e-8, or 10× the noise floor | 9.3e-11 / 1.8e-8 (floor 4e-8) | 4.6e-13 / 9.5e-14 (floors 1.3e-12 / 4.4e-13) | 6.4e-12 / 7.0e-13 (floors 1.2e-10 / 4.9e-13) |
| JES values / gradients (unscaled; worst noise floor) | as above | 3.2e-13 / 4.6e-13 | 9.1e-14 / 1.1e-13 (floors 2.1e-13 / 1.7e-13) | 7.9e-14 / 2.3e-12 (floors 4.5e-13 / 1.8e-11) |
| PES values / gradients, converged EP | as above | 1.5e-7 / 5.4e-7 (floors up to 1e-6) | 1.5e-8 / 1.4e-8 (floors 3.9e-8 / 7.3e-8) | 5.3e-9 / 7.2e-9 (floors 6.5e-7 / 8.8e-7) |
| PES cached EP factors | as above | 2.1e-7 | 3.7e-8 | 2.4e-8 |
| PES EP iteration counts, damping factors, convergence decisions | identical | identical | identical | identical |
| BoTorch's EP iterates step by step (given BoTorch's inputs): factors / marginals / damped Cholesky factors | 1e-9 relative to 10× the last-bit floor / 1e-9 / 1e-9 | 9.8e-11 (scaled) / exact / 5.2e-15 | | |
| PES where BoTorch's EP itself depends on rounding (iteration counts change under the noise) | 0.5 (gross) | 1.9e-3 | 2.3e-15 | 1.9e-15 |
| `optimize_acqf` with JES from identical initial conditions (value / candidate) | 1e-6 / 1e-4 | | 8.5e-15 / 3.5e-15 | |

Note on PES fixtures: in the two fixture models whose EP does not converge
within 250 iterations (`m2_fixed_noise_d2_n8`, and the minimization case of
`m3_matern_d3_n10_normalize`), BoTorch's own values change by up to 0.2
under the perturbations; botorch-rust stays within ten times that change
(worst unscaled value error 0.16).

**Independent of BoTorch:**
- Matheron paths have the posterior's mean and covariance (2048 paths with
  2048 features; mean and standard deviation within 0.1 posterior standard
  deviations).
- Prior paths plus Gaussian updates with the noise of the same seed are the
  Matheron paths of that seed.
- Conditioning on the sampled optima equals a GP refitted with them appended
  (and its derivatives agree with finite differences).
- Sampled Pareto fronts are non-dominated and their cells cover the
  dominated space.
- Single-objective MES is non-negative (up to BoTorch's 1e-6 jitter of the
  lower bounds), and "MC" with 8192 samples is at least "LB".
- All gradients (MES, JES, PES, paths) agree with central finite differences.
- Results do not depend on the number of threads (bitwise).

## Benchmarks (new rows)

`scripts/benchmark_mo_entropy.py`: a two-output `SingleTaskGP` with
`Normalize` (d = 3), BoTorch single-threaded, botorch-rust with one thread
and with 6 threads (`RAYON_NUM_THREADS=6`), on the machine of the other
benchmarks. Medians of 3 runs of at least 0.2 s.

| Task (n = 20 / n = 50 training points) | BoTorch | botorch-rust, 1 thread | botorch-rust, 6 threads | Speed-up (1 thread) |
|---|---|---|---|---|
| `draw_matheron_paths`, 16 paths, 1024 features | 92.7 / 69.6 ms | 4.9 / 4.5 ms | 3.1 / 4.1 ms | 19× / 16× |
| Paths: values and gradients at 1024 points | 336 / 566 ms | 150 / 162 ms | 107 / 140 ms | 2.2× / 3.5× |
| `sample_optimal_points`, 10 sets of 10 points | 1.56 s / 907 ms | 868 / 660 ms | 176 / 197 ms | 1.8× / 1.4× |
| `compute_sample_box_decomposition`, 10 fronts | 8.0 / 11.0 ms | 0.16 / 0.27 ms | 0.22 / 0.24 ms | 49× / 41× |
| JES-LB construction | 15.7 / 27.8 ms | 0.67 / 1.7 ms | 0.45 / 1.2 ms | 24× / 17× |
| MES-LB, 256 × q = 1, values and gradients | 115 / 233 ms | 31 / 53 ms | 7.9 / 10.2 ms | 3.7× / 4.4× |
| JES-LB, 256 × q = 1, values and gradients | 203 / 604 ms | 45 / 65 ms | 11.2 / 15.3 ms | 4.5× / 9.2× |
| JES-LB, 256 × q = 2, values and gradients | 825 ms / 1.06 s | 178 / 108 ms | 43 / 24 ms | 4.6× / 9.8× |
| MES-MC, 256 × q = 1, values and gradients | 1.23 s / 999 ms | 614 / 484 ms | 168 / 138 ms | 2.0× / 2.1× |
| JES-MC, 256 × q = 1, values and gradients | 1.63 / 1.46 s | 794 / 429 ms | 196 / 102 ms | 2.1× / 3.4× |
| JES-MC, 256 × q = 2, values and gradients | 3.46 / 5.15 s | 1.20 / 2.44 s | 328 / 461 ms | 2.9× / 2.1× |
| PES construction (EP: 250 / 19 iterations) | 2.54 s / 743 ms | 262 / 201 ms | 203 / 89 ms | 9.7× / 3.7× |
| PES, 256 × q = 1, values and gradients | 777 ms / 1.64 s | 219 / 324 ms | 63 / 62 ms | 3.5× / 5.1× |

The Monte-Carlo estimates and the random search are dominated by scalar
transcendental functions (normal CDFs, logarithms, sines and cosines), which
PyTorch vectorizes; their single-threaded speed-ups are the smallest.

## Migration notes and known differences (new entries)

- **Random numbers.** Paths, sampled optima and the default MC sampler use
  botorch-rust's generators: `torch.manual_seed` has no effect, and the same
  seed gives different numbers than in BoTorch. Pass `seed=` (keyword-only)
  for reproducibility, or BoTorch's random numbers as `draws=` /
  `candidates=` to reproduce BoTorch's results exactly. Sample `i` of
  `sample_optimal_points` draws its path from stream `2i` of the seed and its
  candidate sets from stream `2i + 1`, so the results do not depend on the
  number of threads.
- **JES conditions on the raw Pareto sets** (meta-pytorch/botorch#3399).
  BoTorch main passes `model.transform_inputs(pareto_sets)` to
  `condition_on_observations`, which transforms them again: with an input
  transform such as `Normalize`, BoTorch conditions on the sampled fronts at
  the wrong inputs. botorch-rust transforms them once (without input
  transforms both agree). Model lists condition each sub-model on its own
  outputs, also with multi-output sub-models, which BoTorch main does not
  handle (it raises a `BotorchTensorDimensionError` or mixes up the outputs).
- **PES uses the raw training inputs** (a BoTorch bug without a pull request
  yet). BoTorch reads `model.train_inputs` at construction; a model in eval
  mode (e.g. after fitting) stores them transformed, and `model.posterior`
  transforms them again, so with `Normalize` BoTorch runs EP at the wrong
  inputs (in a minimal example its value is 0.1925 instead of 0.2111). In
  train mode BoTorch uses the raw inputs and agrees with botorch-rust.
- **PES evaluates every candidate set on its own.** BoTorch evaluates a
  batch of candidate sets jointly, and the fallback of its single test-time
  EP update (on NaNs) and its damping decisions are taken for the whole
  batch, so a candidate's value can depend on the other candidates in the
  batch. Without such fallbacks the two agree.
- **PES gradients where BoTorch's are NaN.** Where BoTorch's autograd
  differentiates a clamped logarithm of zero and returns NaN, botorch-rust
  returns the derivative of the clamped function (finite).
- **PES Cholesky failures.** BoTorch ignores the `info` of `cholesky_ex` in
  `_initialize_predictive_matrices` and continues with a partial factor;
  botorch-rust raises `NumericalError` there.
- **Pathwise sampling with input transforms** (meta-pytorch/botorch#3385):
  paths apply input transforms with eval-mode semantics (BoTorch main re-fits
  a learnable `Normalize` to the test inputs when a path of a model in train
  mode is evaluated), and any number of Fourier features is supported
  (BoTorch's Sobol engine is limited to 21201 dimensions; #3385 falls back to
  i.i.d. weights there, botorch-rust draws them from its scrambled Sobol
  sequence).
- **Lists of multi-output models.** `MatheronPathModel` and
  `sample_optimal_points` reject them with BoTorch's error, as BoTorch does;
  the Rust crate supports them, and so do the acquisition functions.
- **Values kept from BoTorch that look wrong** (reported, not changed):
  - the initial entropy of MES and JES uses the constant of an
    `M`-dimensional Gaussian, `0.5 M (1 + log 2π)`, for the joint posterior
    of `q` points (dimension `qM`): for q > 1 every value is shifted by
    `-0.5 (q - 1) M (1 + log 2π)` (for two independent candidates,
    `value([x1, x2]) = value(x1) + value(x2) - 2.84` with M = 2), which makes
    q-batch values negative but does not change their maximizer;
  - at test time PES computes both "noiseless" and noisy predictive
    covariances with `observation_noise=True` (the comment says "without
    observation noise"), so its observation-noise term is zero and the
    Pareto conditions are applied to noisy observations.

## Not ported (pathwise sampling and entropy search)

- Pathwise sampling of approximate GPs, ensembles, batched models and
  `sample_shape`-batched path models with `ensemble_as_batch`; custom
  `prior_sampler`s and `update_strategy`s (only BoTorch's defaults, with
  `num_features`); kernels other than RBF and Matérn-5/2 (with or without
  `ScaleKernel`).
- Constructing path objects from feature maps and weights;
  `gen_kernel_features` and `KernelEvaluationMap` in Python (both exist in the
  Rust crate) and the transforms of `pathwise.utils`.
- The single-objective entropy search acquisition functions
  (`qMaxValueEntropy`, `qLowerBoundMaxValueEntropy`, GIBBON,
  `qJointEntropySearch`, `qPredictiveEntropySearch`) and their multi-fidelity
  variants.
- Autograd through `log_cdf_robust` and the `_compute_entropy_*` functions
  in Python (values only; the acquisition functions have exact gradients).
