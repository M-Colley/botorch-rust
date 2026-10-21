# Multi-task GPs

The port of BoTorch's multi-task GPs, `MultiTaskGP` and `KroneckerMultiTaskGP`
(roadmap phase 7a): the supported API, the deviations from BoTorch, the
equivalence results and the BoTorch and GPyTorch bugs found. Reference: BoTorch
main at da00e0401 and BoTorch 0.18.1 (GPyTorch 1.15.2, linear_operator 0.6.1).

This is the historical phase-7a audit. Later native and Torch implementations
extend several capabilities listed below; see [Torch backend](torch-backend.md)
for the current device model scope. The MES/GIBBON baseline issue in bug 10 was
independently rechecked against official b0dd01be6 and repaired in the port;
its prepared upstream patch and current validation are recorded below.

Rust: `botorch_core/src/models/multitask/` (`mod.rs`: the `TaskLayout` shared
with `LCEMGP`; `gp.rs`: `MultiTaskGP`; `kronecker.rs`: `KroneckerMultiTaskGP`).
Bindings: `botorch_pyo3/src/multitask.rs`. Python:
`botorch_rs.models.multitask`, `botorch_rs.posteriors.multitask`,
`botorch_rs.likelihoods`, `botorch_rs.kernels.MultitaskKernel`,
`botorch_rs.priors.SmoothedBoxPrior`. `LCEMGP` (which shares `TaskLayout` and
subclasses `MultiTaskGP` in Python) is unchanged; its fixtures pass bitwise.

## Supported API

| BoTorch | botorch_rs | Notes |
| --- | --- | --- |
| `botorch.models.MultiTaskGP` | `botorch_rs.models.MultiTaskGP` | Long-format data, `task_feature` (any column, negative allowed), `output_tasks`, `all_tasks` (tasks without data: `eval()` sets their task factors to the observed mean, as BoTorch), `rank`, `task_covar_prior` (default `BetaPrior(2.5, 1.5)` on the `PositiveIndexKernel` correlations as in BoTorch main; any prior, incl. `LKJCovariancePrior`, or `None`), `covar_module` (any botorch_rs kernel tree; `active_dims` relative to all columns, default the non-task columns), `mean_module` (`None` = per-task `MultitaskMean`, or a shared `ConstantMean`), `likelihood` (`HadamardGaussianLikelihood` or a shared `GaussianLikelihood`), `train_Yvar` (fixed noise), `outcome_transform` (`Standardize(m=1)` default, `None`), `input_transform` (`Normalize`, also with `indices`), `validate_task_values`. `posterior` over the output tasks (X without the task column; `output_indices` = other task values) or single-output (X with the task column), batched, `observation_noise=True`, `posterior_transform`; `condition_on_observations` (incl. `noise`, batched/broadcast), `fantasize`, `construct_inputs` (`SupervisedDataset`, `MultiTaskDataset`, `prior_config`, `map_heterogeneous_to_full`), `get_all_tasks`, `named_parameters` / `initialize` / `deepcopy`, fits (`fit_gpytorch_mll`), the analytic and MC acquisition functions (single- and multi-objective, compared: qLogNEI, qEI, qUCB, qSimpleRegret, LogEI, PosteriorMean, PosteriorStandardDeviation, qLogNEHVI via `ModelListGP`), in `ModelListGP`. |
| `botorch.models.KroneckerMultiTaskGP` | `botorch_rs.models.KroneckerMultiTaskGP` | Block design; `data_covar_module` (any kernel tree), `rank`, `task_covar_prior` (default `LKJCovariancePrior(m, eta, sd_prior)`, keyword arguments `eta=1.5` and `sd_prior=SmoothedBoxPrior(e^-6, e^1.25, 0.05)`), `likelihood` / `likelihood_rank` (task noise covariance of rank `r`), `outcome_transform` (`Standardize(m)`; none by default as in BoTorch), `input_transform` (`Normalize`). Exact MLL and gradients, `posterior` (`MultitaskGPPosterior`), fits, acquisition functions (on the exact joint posterior). |
| `botorch.posteriors.MultitaskGPPosterior` | `botorch_rs.posteriors.MultitaskGPPosterior` | `mean`, `variance` (exact), `distribution` (interleaved, marginal variances as BoTorch), `base_sample_shape` (`(n + q) m + n m [+ q m]`), `batch_range`, `rsample`, `rsample_from_base_samples` (Matheron's rule; output-shaped base samples drive the first `q m` entries as in BoTorch). |
| `gpytorch.likelihoods.MultitaskGaussianLikelihood` | `botorch_rs.likelihoods.MultitaskGaussianLikelihood` | `num_tasks`, `rank` (0: `task_noises`, `>0`: `task_noise_covar_factor`), `noise_prior`, `noise_constraint`; global and task noise. |
| `gpytorch.likelihoods.HadamardGaussianLikelihood` | `botorch_rs.likelihoods.HadamardGaussianLikelihood` | Now constructible (`num_tasks`, `noise_prior`, `noise_constraint`, `task_feature_index`) and passable to `MultiTaskGP`. |
| `gpytorch.kernels.MultitaskKernel` | `botorch_rs.kernels.MultitaskKernel` | The covariance module of `KroneckerMultiTaskGP` (evaluable stand-alone on unbatched inputs). |
| `gpytorch.priors.SmoothedBoxPrior` | `botorch_rs.priors.SmoothedBoxPrior` | Scalar `a`, `b`, `sigma`. |
| `PositiveIndexKernel(task_prior=LKJCovariancePrior)` | same | An LKJ prior on the task covariance matrix. |
| `Normalize(indices=...)` | same | Only the given columns (as BoTorch); for `MultiTaskGP`, and since phase 6b for `SingleTaskGP` and transform chains too. |

## What remains unported (in this area)

- **`HeterogeneousMTGP`** and its heterogeneous kernels (`botorch.models.
  heterogeneous_multitask`, `botorch.models.kernels.heterogeneous_multitask`):
  scope 3 of the phase, not reached.
- Outcome transforms other than `Standardize` (e.g. `StratifiedStandardize`,
  `Log`) and input transforms other than `Normalize` (center 0.5, no
  `reverse`, all `transform_on_*` flags) on the multi-task GPs: ported for
  `SingleTaskGP` in phase 6b, not for these models (`UnsupportedError`).
- Batched training data (BoTorch blocks batched `MultiTaskGP`s;
  `KroneckerMultiTaskGP` with batched data raises `UnsupportedError`).
- Acquisition candidates **with** the task column (single-output acquisition
  over given tasks): botorch_rs's acquisition functions take candidates
  without the task column (over the output tasks), the common use.
- `qKnowledgeGradient` and `qNegIntegratedPosteriorVariance` with a
  `MultiTaskGP` (`UnsupportedError`: botorch_rs's fantasy machinery needs a
  `SingleTaskGP`; BoTorch fails on candidates without the task column too,
  see bug 10). `qMaxValueEntropy` evaluates on a `MultiTaskGP` in
  botorch_rs but fails in BoTorch (bug 10), so it is not compared.
- `KroneckerMultiTaskGP.condition_on_observations` / `fantasize`: BoTorch
  raises `RuntimeError` (no prediction strategy); so does botorch_rs.
- `MultitaskGaussianLikelihood` without global or task noise and with a
  `task_prior` (`UnsupportedError`); batched likelihoods, means and kernels.
- `posterior_transform` and `output_indices` subsets for
  `KroneckerMultiTaskGP` (`NotImplementedError`, as BoTorch for the former).

## Deviations from BoTorch (migration notes)

1. Per-task noise levels of a `HadamardGaussianLikelihood` are looked up by
   task **index** (BoTorch looks them up by the raw task value: bug 1,
   meta-pytorch/botorch#3396, fixed in M-Colley/botorch
   `fix/models-posteriors`).
2. An `LKJCovariancePrior` task prior acts on the task covariance matrix
   (BoTorch fails, bug 2); its correlation term is counted once (GPyTorch
   counts it `n` times with a univariate `sd_prior`, see
   `docs/kernels-and-priors.md`).
3. A `ProductKernel` `covar_module` keeps acting on the non-task columns, and
   `eval()` finds the task kernel as the last factor (bug 3).
4. `KroneckerMultiTaskGP` posterior variances are exact also with
   task-dependent noise or `likelihood_rank > 0` (bug 4). With a
   `Standardize` outcome transform the posterior is still a
   `MultitaskGPPosterior` in the original space (BoTorch: a
   `TransformedPosterior`), with the same values.
5. `KroneckerMultiTaskGP` acquisition functions sample the exact joint
   posterior (`q m` base samples per draw, Cholesky root); BoTorch samples by
   Matheron's rule (`(n + q) m + n m` base samples). Same distribution, other
   draws. `MultitaskGPPosterior.rsample_from_base_samples` uses Matheron's
   rule with Cholesky roots (BoTorch: eigendecomposition-based roots): same
   base sample shape and distribution, other draws.
6. `KroneckerMultiTaskGP.posterior(output_indices=...)` other than all
   outputs raises `NotImplementedError` (BoTorch main ignores it, bug 6).
7. The `active_dims` of `data_covar_module` apply (GPyTorch's
   `MultitaskKernel` ignores them, bug 7).
8. Fixed-noise `MultiTaskGP` with `observation_noise=True`: the noise of a
   task without data is the mean of all training noise levels (BoTorch:
   NaN, bug 8).
9. `condition_on_observations(..., noise=...)` of a model with learned noise:
   the given noise enters the posterior mean and covariance (#3396; BoTorch
   main uses it for the covariance only).
10. `construct_inputs(prior_config=...)` counts the task values in the task
    column (bug 5).
11. Exceptions as BoTorch: invalid task values (`ValueError`, same messages),
    `output_indices` with a task column (`ValueError`), tensor
    `observation_noise` (`NotImplementedError`), `subset_output`
    (`UnsupportedError`), conditioning in train mode (`RuntimeError`),
    fantasies of fixed-noise models (`NotImplementedError`; BoTorch fails
    earlier with `IndexError` for several output tasks, bug 11), boolean
    `observation_noise` in `fantasize` (`DeprecationError`); fantasies with
    candidates without the task column raise a clear `RuntimeError` (BoTorch
    fails with a shape error, bug 10).

## Equivalence

Fixtures: `scripts/generate_fixtures.py --phase 7a-multitask`
(`scripts/fixtures_7a_multitask.py`, cases shared with the live tests in
`botorch_pyo3/tests/multitask_helpers.py`), 12 `MultiTaskGP` cases (default,
rank 1 / task column 0 / output tasks, a task without data, non-contiguous
task values in column 1, fixed noise, Matérn + scale on d = 3, additive and
product kernels, LKJ prior, Gamma task prior + constant mean + Gaussian
likelihood, `Normalize(indices)` without standardization, single output task)
and 4 `KroneckerMultiTaskGP` cases (default m = 3, rank 1 + likelihood rank 2
+ Standardize + Normalize, Matérn + eta + SmoothedBox sd prior, data kernel
`active_dims`), generated against BoTorch main (da00e0401); they regenerate
byte for byte. Rust test: `botorch_core/tests/multitask_7a_parity.rs`. Live:
`botorch_pyo3/tests/test_parity_7a_multitask.py` (BoTorch main and 0.18.1);
randomized sweeps (30 `MultiTaskGP` cases with random tasks, priors, ranks,
output/unobserved tasks, fixed noise and random kernel trees; 16
`KroneckerMultiTaskGP` cases with random ranks, likelihood ranks,
transforms and kernel trees) marked `equivalence`. API and error tests:
`botorch_pyo3/tests/test_7a_multitask.py` (all array types).

The reference is BoTorch with its bugs fixed (`multitask_helpers.fix_reference`:
task-mapped noise, LKJ on the covariance matrix in double precision with the
correlation term once, product-kernel `active_dims`, `eval()` on the last
factor); `test_botorch_bugs` measures what the bugs change. Kronecker
variances are compared with the exact posterior computed densely in torch
from BoTorch's hyperparameters (which is also the distribution of BoTorch's
Matheron samples, measured through their linear map), and with BoTorch's
variances where its closed form is exact (equal task noises).

### Tolerances above nominal

- **Fits** (as in 6a): BoTorch's own fits from twins (training inputs
  perturbed at 1e-14 .. 1e-11) end at different optima. botorch_rs's best
  first-attempt loss must be within `max(1e-6, 10 x BoTorch's spread at its
  best optimum)` of BoTorch's best, and its own fit within that of BoTorch's
  worst; when BoTorch's outcomes range over more than 1e-3 (chaotic, e.g.
  `default_t3_d2`: BoTorch [0.09241, 0.10305] on 17 twins), 16 twins are
  compared statistically (best below BoTorch's median, median below its
  upper quartile, 1e-6). Reported values are scaled by the tolerance.
- **qLogNEI over the training baseline / qLogNEHVI** (nearly singular joint
  posteriors): BoTorch's outcomes jump between Cholesky-jitter regimes under
  rounding-level perturbations; compared with the regime botorch_rs falls
  into (`check_against_twins`: within 10 x the median spread of that
  regime, at least the nominal 1e-9 / 1e-8). Nearest-regime errors are
  listed unscaled.
- **Sweep, `exact_kernel_products`**: GPyTorch multiplies two non-dense
  kernel matrices (e.g. a `LinearKernel`'s low-rank root or an
  `AdditiveKernel`'s sum, by the task kernel's `InterpolatedLinearOperator`)
  through a `MulLinearOperator`, which takes Cholesky roots of both factors;
  the `n x n` task covariance of `n` data from fewer tasks is singular, the
  factorization fails and is retried with 1e-8 jitter. Measured MLL changes
  3.1e-9 .. 4.8e-9 (relative) in 4 of 30 sweep cases (3.6e-7 for a
  `LinearKernel` data kernel); with exact products the errors are
  1.5e-16 .. 2.2e-16. The sweep evaluates BoTorch with exact (dense)
  products, and with `zero_diagonal_distances` (quirk 7 in
  `docs/kernels-and-priors.md`).

### Fixtures (Rust, `multitask_7a_parity.rs`)

`MultiTaskGP` (12 cases):

| Quantity | Tolerance | Worst error |
| --- | --- | --- |
| default raw parameters (incl. random task factors) | exact | 0 |
| default MLL / gradient | 1e-9 / 1e-8 | 1.9e-14 / 1.6e-13 |
| MLL / gradient (random raw parameters, all parameters incl. task factors and LKJ) | 1e-9 / 1e-8 | 4.6e-15 / 1.8e-15 |
| raw parameters after `eval()` | 1e-15 | 2.4e-16 |
| point posterior mean / output covariance (with noise) | 1e-9 | 2.4e-15 / 2.1e-15 (1.8e-15) |
| point posterior d mean / dx, d variance / dx | 1e-8 | 2.2e-15, 5.1e-15 |
| joint posterior mean / covariance / with noise | 1e-9 | 1.9e-15 / 5.8e-15 / 5.3e-15 |
| joint posterior d (weighted sum) / dx | 1e-8 | 7.1e-15 |
| joint posterior with task inputs: mean / covariance / with noise / d/dx | 1e-9 / 1e-8 | 1.7e-15 / 3.1e-15 / 2.1e-15 / 1.9e-15 |
| conditioned posterior mean / covariance | 1e-9 | 3.1e-15 / 4.1e-15 |
| fantasy posterior mean / covariance (recorded base samples) | 1e-9 | 1.9e-15 / 4.1e-15 |
| qLogNEI q = 1, 2 (twins, scaled) | 1 | 1.7e-4, 5.3e-4 (nearest values 4.5e-13, gradients 5.3e-12) |
| qUCB (scalarized) values / gradients | 1e-9 / 1e-8 | 9.3e-16 / 4.3e-15 |
| LogEI values / gradients | 1e-9 / 1e-8 | 1.1e-15 / 4.9e-15 |
| PosteriorMean (scalarized) values / gradients | 1e-9 / 1e-8 | 1.0e-15 / 1.7e-15 |
| fit: best above BoTorch's best / own above BoTorch's worst (scaled) | 1 | 8.9e-10 / 0 |
| fit (chaotic): best vs median / median vs upper quartile | 1e-6 | 0 / 0 |

`KroneckerMultiTaskGP` (4 cases):

| Quantity | Tolerance | Worst error |
| --- | --- | --- |
| default raw parameters | exact | 0 |
| default MLL / gradient | 1e-9 / 1e-8 | 2.4e-15 / 1.0e-13 |
| MLL / gradient | 1e-9 / 1e-8 | 2.4e-16 / 1.3e-15 |
| posterior mean vs BoTorch / exact / own Matheron mean | 1e-9 | 8.9e-16 / 1.4e-15 / 1.6e-15 |
| covariance vs exact / with noise | 1e-9 | 3.3e-15 / 8.3e-16 |
| covariance vs BoTorch's Matheron samples / with noise | 1e-9 | 1.1e-14 / 2.4e-15 |
| own Matheron samples' covariance vs exact | 1e-9 | 1.4e-14 |
| d (weighted sum of mean and covariance) / dx | 1e-8 | 2.8e-15 |
| qEI (linear objective) values / gradients | 1e-9 / 1e-8 | 3.5e-16 / 1.2e-15 |
| fit: best above BoTorch's best / own above worst (scaled) | 1 | 1.5e-3 / 0 |

(Kronecker fit `default_m3_d2`: BoTorch [0.27748067, 0.27749277] on 17 twins,
botorch_rs best 0.27748085.)

### Live parity (`test_parity_7a_multitask.py`)

Against BoTorch main (da00e0401); the run against 0.18.1 passes too.

| Quantity | Tolerance | Worst error |
| --- | --- | --- |
| `MultiTaskGP`: parameter names, shapes, optimizer bounds | exact | equal |
| default raw parameters (incl. random task factors) | exact | 0 |
| default MLL / gradient | 1e-9 / 1e-8 | 1.1e-14 / 4.8e-14 |
| MLL / gradient (random raw parameters) | 1e-9 / 1e-8 | 3.2e-15 / 7.7e-15 |
| raw parameters after `eval()` | 1e-15 | 2.7e-16 |
| task covariance matrix, likelihood noise | 1e-9 | 1.8e-16, 0 |
| posterior over output tasks: mean / variance / covariance (with noise) | 1e-9 | 1.2e-15 / 1.7e-15 / 2.0e-15 (1.8e-15) |
| posterior with task inputs: mean / covariance (with noise) | 1e-9 | 1.5e-15 / 1.3e-15 (1.1e-15) |
| batched posterior: mean / covariance (with noise) | 1e-9 | 1.3e-15 / 3.4e-15 (2.9e-15) |
| posterior (`output_indices`): mean / covariance | 1e-9 | 1.3e-15 / 8.5e-15 |
| samples from base samples | 1e-9 | 4.8e-15 |
| conditioned posterior mean / covariance (single, batched, batched noise) | 1e-9 | 1.5e-15 / 3.8e-15 |
| conditioned on explicit noise vs dense exact posterior: mean / covariance | 1e-9 | 1.1e-16 / 9.3e-16 |
| fantasy posterior mean / covariance (BoTorch's base samples) | 1e-9 | 1.6e-15 / 3.8e-15 |
| PosteriorMean (scalarized) values / gradients | 1e-9 / 1e-8 | 1.6e-15 / 5.5e-15 |
| qSimpleRegret (scalarized) values / gradients | 1e-9 / 1e-8 | 2.8e-15 / 9.4e-15 |
| qEI, qUCB values / gradients | 1e-9 / 1e-8 | 1.1e-15, 3.5e-16 / 3.8e-15, 4.7e-15 |
| LogEI, PosteriorMean, PosteriorStandardDeviation values / gradients | 1e-9 / 1e-8 | 4.5e-15 / 3.1e-15 |
| qLogNEI q = 1, 2 (twins, scaled; nearest values / gradients) | 1 | 1.2e-5 (3.1e-15 / 1.2e-13) |
| `ModelListGP` of two `MultiTaskGP`s: posterior mean / covariance | 1e-9 | 4.3e-16 / 1.1e-15 |
| qLogNEHVI on that `ModelListGP` (twins, scaled; nearest values / gradients) | 1 | 8.4e-6 (6.9e-15 / 8.4e-14) |
| fits: best above BoTorch's best / own above BoTorch's worst (scaled) | 1 | 3.9e-2 / 0.42 |
| `construct_inputs` (`SupervisedDataset`, `MultiTaskDataset`) | exact | equal |
| `KroneckerMultiTaskGP`: parameter names, shapes, bounds | exact | equal |
| default raw parameters (incl. random factors) | exact | 0 |
| default MLL / gradient | 1e-9 / 1e-8 | 9.8e-15 / 6.1e-14 |
| MLL / gradient | 1e-9 / 1e-8 | 1.9e-16 / 6.2e-15 |
| posterior mean vs BoTorch / exact (with noise) | 1e-9 | 1.9e-15 / 1.5e-15 |
| posterior variance vs exact (with noise) | 1e-9 | 6.8e-16 (4.8e-16) |
| variance vs BoTorch with equal task noises | 1e-9 | 1.2e-15 |
| BoTorch's Matheron samples' covariance vs exact (with noise) | 1e-9 | 5.6e-15 (3.0e-15) |
| botorch_rs's Matheron samples' mean / covariance vs exact | 1e-9 | 1.5e-15 / 3.1e-15 |
| qEI (linear objective) values / gradients (exact joint posterior) | 1e-9 / 1e-8 | 9.3e-16 / 3.2e-15 |
| qLogNEHVI values / gradients (exact joint posterior) | 1e-9 / 1e-8 | 8.9e-13 / 3.1e-12 |
| fits (scaled) | 1 | 0 / 0 |
| sweep (30 cases): MultiTaskGP MLL / gradient | 1e-9 / 1e-8 | 4.9e-15 / 6.8e-14 |
| sweep: posterior mean / covariance (with or without noise) | 1e-9 | 1.4e-14 / 6.8e-15 |
| sweep: PosteriorMean (scalarized) values / gradients | 1e-9 / 1e-8 | 3.0e-14 / 3.8e-14 |
| sweep (16 cases): Kronecker MLL / gradient | 1e-9 / 1e-8 | 3.8e-16 / 5.8e-15 |
| sweep: Kronecker posterior mean (BoTorch) / variance (exact) | 1e-9 | 2.3e-15 / 3.2e-15 |

Measured effects of BoTorch's bugs (`test_botorch_bugs`): noise by raw task
value, MLL change 3.1e-3; `ProductKernel` `active_dims`, MLL change 6.8e-2;
Kronecker variance with task-dependent noise, relative error 2.6e-2.

## BoTorch / GPyTorch bugs found

Reproductions against BoTorch main (da00e0401), double precision; `X`
random in the unit cube with the task column last unless stated.

1. **`HadamardGaussianLikelihood` looks up the per-task noise by the raw task
   value**, not the task index (`MultiTaskGP` maps task values for the
   kernel only), so non-contiguous task values get the wrong noise (or
   none):

   ```python
   X = torch.rand(8, 2, dtype=torch.float64); X[:, -1] = torch.tensor([0., 2.] * 4)
   m = MultiTaskGP(X, torch.randn(8, 1, dtype=torch.float64), task_feature=-1)
   m.likelihood.noise_covar.noise = torch.tensor([0.1, 0.5], dtype=torch.float64)
   m.train(); (m.likelihood(m(X), X).covariance_matrix - m(X).covariance_matrix).diagonal()[:2]
   # tensor([0.1000, 0.0000]): task 2 gets no noise instead of 0.5
   ```

   Already reported as meta-pytorch/botorch#3396 (fix in M-Colley/botorch
   `fix/models-posteriors`); MLL change 3.1e-3 in `test_botorch_bugs`.
2. **`MultiTaskGP(task_covar_prior=LKJCovariancePrior(...))` fails**: the
   `PositiveIndexKernel` evaluates its task prior on the vector of
   correlations, which the LKJ prior cannot take:

   ```python
   X[:, -1] = torch.tensor([0., 1., 2.] * 3)  # 9 rows
   m = MultiTaskGP(X, Y, task_feature=-1,
                   task_covar_prior=LKJCovariancePrior(3, torch.tensor(1.5), GammaPrior(2.0, 1.5)))
   ExactMarginalLogLikelihood(m.likelihood, m)(m(X), m.train_targets, X)
   # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
   ```

   (`construct_inputs(prior_config={"use_LKJ_prior": True})` produces
   exactly this prior.)
3. **A `ProductKernel` `covar_module` loses its `active_dims`**: BoTorch sets
   `active_dims` on the product, then `covar_module * task_kernel` flattens
   it into `ProductKernel(RBF, Matern, PositiveIndexKernel)` without them,
   so the data kernels act on the task column too; and `eval()` takes
   `covar_module.kernels[1]` as the task kernel:

   ```python
   X = torch.rand(8, 3, dtype=torch.float64); X[:, -1] = torch.tensor([0., 1.] * 4)
   m = MultiTaskGP(X, Y, task_feature=-1, covar_module=ProductKernel(RBFKernel(), MaternKernel()))
   [k.active_dims for k in m.covar_module.kernels]  # [None, None, tensor([2])]
   MultiTaskGP(X, Y, task_feature=-1, all_tasks=[0, 1, 2],
               covar_module=ProductKernel(RBFKernel(), MaternKernel())).eval()
   # AttributeError: 'MaternKernel' object has no attribute 'covar_factor'
   ```

4. **`KroneckerMultiTaskGP.posterior` variances are wrong with
   task-dependent noise** (`likelihood_rank > 0`, or unequal `task_noises`):
   the closed form assumes noise `sigma^2 I`. With `X` 6 x 1,
   `task_noises = [0.01, 0.5]`, BoTorch's variances at `x = 0.3` are
   `[0.6215, 0.8849]`, the dense exact posterior's `[0.2857, 1.0116]`.
   (BoTorch's own Matheron samples have the exact covariance, 1e-14.)
   Relative error 2.6e-2 in `test_botorch_bugs` (random hyperparameters);
   fixed in M-Colley/botorch `fix/models-posteriors` (7db8b54cf).
5. **`MultiTaskGP.construct_inputs(prior_config={"use_LKJ_prior": True})`
   counts the distinct values of the row `training_data.X[task_feature]`**
   instead of the task column: with 3 tasks and rows of 4 values the LKJ
   prior is for `n = 4` tasks (and then fails, bug 2).
6. **`KroneckerMultiTaskGP.posterior(X, output_indices=[1])` ignores
   `output_indices`** (BoTorch main; shape `q x 2` for 2 tasks).
7. **GPyTorch's `MultitaskKernel` ignores the `active_dims` of its
   `data_covar_module`** (it calls the data kernel's `forward`): with
   `RBFKernel(active_dims=[0])` on 2-d inputs, `k(x)` differs from
   `k(x[:, :1])` by 0.48.
8. **Fixed-noise `MultiTaskGP` with a task without data:
   `posterior(X, observation_noise=True)` is NaN for that task** (the mean of
   no training noise levels):

   ```python
   m = MultiTaskGP(X, Y, task_feature=-1, train_Yvar=torch.full((8, 1), 0.01), all_tasks=[0, 1, 2])
   m.eval(); m.posterior(torch.rand(2, 1), observation_noise=True).variance  # [:, 2] is nan
   ```

9. **Fixed-noise `MultiTaskGP.condition_on_observations` with non-batched `X`
   and batched `Y` and `noise` fails** (`RuntimeError: All tensors must have
   the same number of dimensions`, GPyTorch's
   `FixedNoiseGaussianLikelihood` concatenates the noise without
   broadcasting). Batched `X`, or non-batched `noise`, works; botorch_rs
   broadcasts all three.
10. **`MultiTaskGP.fantasize` with candidates without the task column fails**
    (`RuntimeError: Unsupported batch shapes ...`), and so does
    `qKnowledgeGradient` on a single-output-task `MultiTaskGP`
    (`RuntimeError: Sizes of tensors must match ...`), and so does
    `qMaxValueEntropy(m, candidate_set)(X)` and GIBBON (`RuntimeError: Sizes of
    tensors must match ...`). The entropy failure occurs at construction:
    `MaxValueBase` appends long-format `train_inputs` (including the task column)
    to the task-free `candidate_set`. The port now removes that column only
    when designs have `num_non_task_features` columns. Their posterior retains
    the selected `output_tasks`; explicit task-labelled designs and reduced
    user baselines are preserved. Native default baselines already used
    continuous coordinates; explicit baselines and the Torch constructor now
    use the same law. Task-free fantasies/KG remain a separate model boundary.
11. **`Model.fantasize` of a fixed-noise `MultiTaskGP` with several output
    tasks fails with `IndexError`**: for `num_outputs > 1` it transposes
    `likelihood.noise`, a vector (`n`) for a `MultiTaskGP`:

    ```python
    m = MultiTaskGP(X, Y, task_feature=-1, train_Yvar=torch.full((8, 1), 0.01))  # tasks 0, 1
    m.eval(); m.fantasize(Xn, SobolQMCNormalSampler(torch.Size([3])))  # Xn with task column
    # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
    ```

    With one output task it reaches the intended `NotImplementedError`
    (tensor observation noise), which botorch_rs raises in both cases.
12. Quirk (not a bug): `best_f` given as a Python float is stored in torch's
    default dtype (float32) by BoTorch's analytic acquisition functions
    (LogEI change 3e-8); the tests pass double-precision tensors.

GPyTorch numerical quirk fixed for the sweep comparisons:
`MulLinearOperator` root decompositions with jitter in products of non-dense
kernel matrices (see "Tolerances above nominal").

### Upstream status

| Bug | Fix |
|---|---|
| 1 | meta-pytorch/botorch#3396 |
| 2, 5 | meta-pytorch/botorch#3446 |
| 3 | meta-pytorch/botorch#3447 |
| 4, 6 | meta-pytorch/botorch#3396 (bug 6: raises `NotImplementedError`) |
| 7 | [GPyTorch #2809](https://github.com/cornellius-gp/gpytorch/pull/2809) |
| 8 | meta-pytorch/botorch#3389 and #3374 |
| 9 | meta-pytorch/botorch#3448 (also affects `SingleTaskGP`) |
| 10 | `fantasize` and qKG: meta-pytorch/botorch#3374. MES/GIBBON: [BoTorch #3504](https://github.com/meta-pytorch/botorch/pull/3504). |
| 11 | meta-pytorch/botorch#3374 |

The MES/GIBBON follow-up passes 161 port CPU/CUDA max-entropy and multitask
tests, including 34 new cases for first/interior/last task columns, external
task labels, selected output tasks, full-labelled candidates, explicit
baselines, query/parameter gradients, second backward and prohibited host
conversion guards. Native explicit-baseline parity covers lists, NumPy and
Torch inputs. The unchanged pinned source fails all twelve task-free
CPU/CUDA/column/acquisition regression subcases; the proposed source repair
passes its full max-entropy suite and new regressions (10 tests, 14 subcases).
Reference numerical tests use fixed noise to isolate this issue from the
already reported inferred-noise task-remapping bug 1. This is a proposed
upstream patch, not a claim that official main already includes it.

## Verification

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
warnings`, `cargo test --workspace` (356 tests pass; the 3 doctests pass
with `--test-threads=1`, while linking them in parallel intermittently fails
on this machine with MSVC's `LNK1318: Unexpected PDB error`, unrelated to the
code); `scripts/develop.py`; ruff check / format (with
`scripts/fixtures_7a_multitask.py` added to CI's lists); mypy. `pytest
botorch_pyo3/tests` with `BOTORCH_RS_REQUIRE_BOTORCH=1` (sweeps included):
against BoTorch main 3186 passed, 36 skipped (the 3 failures of the first run,
an obsolete "Normalize(indices) is unsupported" assertion, and the 7a files
changed afterwards were rerun and pass); against 0.18.1 3169 passed, 56
skipped. The package imports (and fits / predicts both models) on Python 3.11
without torch and NumPy. The fixtures regenerate byte for byte against
BoTorch main.

Tests changed outside this phase: `test_lcemgp.py`,
`test_parity_lcemgp.py` and `test_models.py` asserted that `MultiTaskGP` and
`Normalize(indices=...)` are unsupported; they now assert the opposite.
