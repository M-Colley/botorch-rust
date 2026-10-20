# Multi-fidelity, mixed, MAP SAAS, ensemble and list models

The port of BoTorch's models other than the multi-task GPs (roadmap phase 7b):
the supported API, the deviations from BoTorch, the equivalence results and the
BoTorch bugs found. Ported: `SingleTaskMultiFidelityGP`, `MixedSingleTaskGP`,
the MAP SAAS models, `AffineDeterministicModel`, `FixedSingleSampleModel`,
`EnsembleModel` / `EnsemblePosterior` / `IndexSampler`, `ModelList` /
`ModelDict` / `PosteriorList`, `FantasizeMixin`, `subset_output` and
`construct_inputs`, `OrthogonalAdditiveGP` (including component posteriors),
and the ensemble paths of the acquisition functions. Reference: BoTorch main at
da00e0401 and BoTorch 0.18.1 (GPyTorch 1.15.2, linear_operator 0.6.1).

This chapter records the native CPU implementation and its phase-specific
verification. The integrated [Torch backend](torch-backend.md) also provides
device and higher-order graphs for the documented mixed, multi-fidelity,
MAP SAAS, ensemble and additive model paths. The configuration limits below
describe the native CPU route; Tensor configurations have separate family notes.

## Supported API

| BoTorch | botorch_rs | Notes |
|---|---|---|
| `botorch.models.SingleTaskMultiFidelityGP` | same path (also `botorch_rs.models`) | `data_fidelities`, `iteration_fidelity`, `linear_truncated` (`LinearTruncatedFidelityKernel` with `nu`) or `False` (`ExponentialDecayKernel` × `DownsamplingKernel`s × base kernel), custom `covar_module`, `train_Yvar`, `m ≥ 1` outputs, `Normalize`, `Standardize`; `construct_inputs(training_data, fidelity_features)`; `subset_output`; fits, posteriors, conditioning and fantasies as for `SingleTaskGP`; `qMultiFidelityKnowledgeGradient`, `qMultiFidelityMaxValueEntropy` / `qMultiFidelityLowerBoundMaxValueEntropy` and `qMultiFidelityHypervolumeKnowledgeGradient` on it |
| `botorch.models.MixedSingleTaskGP` | same path (also `botorch_rs.models`) | `cat_dims` (negative indices too), `cont_kernel_factory`, BoTorch's kernel `ScaleKernel(K_cont + ScaleKernel(K_cat)) + ScaleKernel(K_cont · K_cat)` (or `ScaleKernel(K_cat)` for an all-categorical space), `CategoricalKernel` with `GreaterThan(1e-6)` lengthscales, `train_Yvar`, `m ≥ 1`; `construct_inputs(training_data, categorical_features)`; `optimize_acqf_mixed` and `optimize_acqf_mixed_alternating` |
| `botorch.models.map_saas.get_map_saas_model`, `add_saas_prior`, `SaasPriorHelper`, `get_additive_map_saas_covar_module`, `get_mean_module_with_normal_prior`, `get_gaussian_likelihood_with_gamma_prior` | `botorch_rs.models.map_saas.*` (`add_saas_prior` also in `botorch_rs.models`) | half-Cauchy SAAS prior on `tau / lengthscale²`, fixed `tau` (float or tensor; a Python float is stored in float32, as in BoTorch) or learned `raw_tau` (`LogTransformedInterval(1e-3, 10, initial_value=0.1)` with `HalfCauchyPrior(0.1)`), BoTorch's `EPS = 1e-8` clamps; `NormalPrior(0, 1)` constant mean, `GammaPrior(0.9, 10)` noise with `LogTransformedInterval(1e-4, 1, initial_value=1e-2)` |
| `botorch.models.map_saas.AdditiveMapSaasSingleTaskGP` | same path (also `botorch_rs.models`) | sum of `num_taus` SAAS kernels with taus drawn as BoTorch does (torch's global generator) |
| `botorch.models.map_saas.EnsembleMapSaasSingleTaskGP` | same path (also `botorch_rs.models`) | batch of `num_taus` SAAS models (`taus`), `_is_ensemble`, `posterior` → `GaussianMixturePosterior` (`mixture_mean`, `mixture_variance`, `mixture_covariance_matrix`, `quantile`), `construct_inputs(num_taus)`; analytic and MC acquisition functions averaged over the members (`average_over_ensemble_models`), `prune_inferior_points` |
| `botorch.models.deterministic.AffineDeterministicModel` | same path (also `botorch_rs.models`) | `b + X a`; BoTorch's errors; `b` from a Python float is float32 as in BoTorch; usable as the model of acquisition functions (Rust model with analytic gradients) |
| `botorch.models.deterministic.FixedSingleSampleModel` | same path | `mean + sqrt(variance + jitter) · w` of a botorch_rs acquisition model (e.g. a `SingleTaskGP`), `w` from `torch.randn` by default; usable as the model of acquisition functions |
| `botorch.models.ensemble.EnsembleModel` | same path | abstract `forward` (torch); `posterior` → `EnsemblePosterior`; MC acquisition functions with an `IndexSampler` (members evaluated through `forward` with `torch.autograd` gradients); analytic acquisition functions on the mean and variance of the `EnsemblePosterior` (`ExpectedImprovement`, `LogExpectedImprovement`, `ProbabilityOfImprovement`, `LogProbabilityOfImprovement`, `UpperConfidenceBound`, `PosteriorMean`, `PosteriorStandardDeviation`, `ConstrainedExpectedImprovement`, `LogConstrainedExpectedImprovement`, `LogProbabilityOfFeasibility`). Acquisition functions need the attribute `dim` (botorch_rs extension) |
| `botorch.posteriors.ensemble.EnsemblePosterior`, `botorch.posteriors.fully_bayesian.GaussianMixturePosterior`, `botorch.posteriors.posterior_list.PosteriorList` | same paths (`PosteriorList` also in `botorch_rs.posteriors`) | weights, mixture moments, `rsample`, `rsample_from_base_samples`, quantiles (`batched_bisect`) |
| `botorch.sampling.index_sampler.IndexSampler` | same path | draws the member indices from torch's generator seeded with `seed` (the same draws as BoTorch) |
| `botorch.models.model.ModelList`, `ModelDict`, `FantasizeMixin` | same paths (`ModelList` also in `botorch_rs.models`) | `ModelList` of botorch_rs models (GPs and deterministic models): `posterior` (`PosteriorList`), `batch_shape`, `num_outputs`, `subset_output`, `transform_inputs`, `fantasize` (per sub-model), acquisition functions on the joint model; `ModelDict` with BoTorch's `InputDataError` |
| `Model.subset_output`, `Model.construct_inputs` | same names | `subset_output` for `SingleTaskGP` (any covariance module), `SingleTaskMultiFidelityGP`, `MixedSingleTaskGP`, the MAP SAAS models, `ModelListGP`, `ModelList`, the deterministic models; `construct_inputs` for every model above with BoTorch's or botorch_rs's `SupervisedDataset` |
| `botorch.utils.datasets.SupervisedDataset` | same path | `X`, `Y`, `Yvar`, names, validation, `clone`, equality |
| `botorch.models.additive_gp.OrthogonalAdditiveGP` | same path (not exported from `botorch_rs.models`, as in BoTorch) | `SingleTaskGP` with an `OrthogonalAdditiveKernel` (first and second order); conditioned component posteriors with `infer_all_components=True`, component metadata and `evaluate_first_order_on_grid` ([scope](additive-components.md)) |
| `gpytorch.constraints.LogTransformedInterval`, `botorch.utils.constraints.LogTransformedInterval` / `NonTransformedInterval` | `botorch_rs.utils.constraints.*`, `botorch_rs.constraints` | raw = `log(value)`, raw bounds `(log lb, log ub)`, float32 bounds and initial raw value as in BoTorch |
| `gpytorch.means.ConstantMean(constant_prior, constant_constraint)` | `botorch_rs.means.ConstantMean` | priors on the constant (MLL term, gradient, prior sampling); non-enforced constraints only |

<a id="what-remains-unported-in-this-area"></a>

## Native CPU configuration limits

- `OrthogonalAdditiveGP` component inference requires a single output and an
  affine outcome transform. Nonlinear outcome transforms do not preserve the
  additive decomposition and are rejected for component inference; full-model
  posteriors retain their transform support ([details](additive-components.md)).
- Ensemble paths of the acquisition functions other than the listed ones
  (knowledge gradient, entropy searches, HVKG and the multi-objective
  acquisition functions on `EnsembleMapSaasSingleTaskGP` /
  `EnsembleModel`): raise `UnsupportedError`.
- Pruning the baseline of an ensemble (`prune_inferior_points`) with an
  objective, posterior transform or constraints: `UnsupportedError`
  (`prune_baseline=False` works).
- Posterior transforms on `EnsembleModel` (BoTorch's `scalarize_posterior`
  raises `NotImplementedError` for an `EnsemblePosterior` too).
- `EnsembleModel` with an input or outcome transform in acquisition
  functions (the posterior supports them).
- Multi-task models: phase 7a.
- Outcome transforms other than `Standardize` on
  `EnsembleMapSaasSingleTaskGP` (`UnsupportedError`; BoTorch cannot build
  its posterior either). The other models of this phase take every input
  and outcome transform of phase 6b (see `docs/drafts/6b-transforms.md`).

## Deviations from BoTorch (migration notes)

- `subset_output` works for every `SingleTaskGP` covariance module (BoTorch
  raises `NotImplementedError` without `_subset_batch_dict`, e.g. for a
  custom `covar_module`); every botorch_rs module has independent
  parameters per output.
- `SingleTaskMultiFidelityGP(..., data_fidelities=[])` is treated as
  `None` (BoTorch builds an empty `ProductKernel` and fails on evaluation,
  bug 1).
- `PosteriorStandardDeviation` on an `EnsembleMapSaasSingleTaskGP` is the
  average of the members' standard deviations, like every other analytic
  acquisition function there (BoTorch raises, bug 2).
- `EnsembleMapSaasSingleTaskGP.fantasize(X)` returns a `num_fantasies x
  num_taus` batch of conditioned members (BoTorch fails for every shape of
  `X`, bug 6); `ModelList.fantasize` fantasizes every sub-model with its
  own sampler and outputs (BoTorch indexes sub-models by output, bug 4).
- An `EnsembleModel` acquisition function needs the attribute `dim` (the
  input dimension), and the members selected by the `IndexSampler` are the
  same for every t-batch of one evaluation (BoTorch draws the indices per
  t-batch, `sample_shape x posterior.batch_shape`; values agree for one
  t-batch, which is what the tests compare).
- The analytic acquisition functions of an `EnsembleModel` use the
  unclamped `EnsemblePosterior.variance` (then `_mean_and_sigma`'s `1e-12`
  clamp), as BoTorch; GPyTorch's `1e-10` variance clamp does not apply.
- A deterministic column of a `ModelList` (e.g. `GP + GenericDeterministicModel`)
  has an exactly zero covariance block; its Monte-Carlo samples equal the
  mean for any base samples (previously the jittered root added noise of
  about `2e-4`).
- The LogEI tail: as in phase 8, the gradient of `LogExpectedImprovement`
  far in the tail (`u ≲ -1e3`, e.g. the `1e-6` standard deviation of an
  ensemble whose members agree) is exact, while BoTorch's autograd has a
  relative error of about `eps · u²` (measured 2.1e-5 at `|u| ≈ 8e5`).

## Equivalence tests

Tolerances: values 1e-9, gradients 1e-8 (relative); structure, bounds and
defaults exact.

### Rust fixtures (`botorch_core/tests/models_7b_parity.rs`, 14 models)

Fixtures `botorch_core/tests/fixtures/7b_models/*.json` (generated against
BoTorch main with `scripts/generate_fixtures.py --phase 7b-models`): six
multi-fidelity models (linear truncated with `nu` 0.5 / 1.5 / 2.5,
downsampling, data and iteration fidelities, fixed noise, two outputs),
three mixed models (one or two categorical columns, all-categorical, two
outputs) and five MAP SAAS models (fixed and learned tau, fixed noise with
two outputs, additive, ensemble).

| Quantity | Tolerance | Worst error |
|---|---|---|
| parameter names, shapes, bounds, defaults | exact | 0 |
| MLL (random raw parameters) | 1e-9 | 2.1e-14 |
| MLL gradient (random raw parameters) | 1e-8 | 3.5e-14 |
| MLL (default parameters) | 1e-9 | 4.0e-15 |
| MLL gradient (default parameters) | 1e-8 | 1.7e-13 |
| prior kernel matrix | 1e-9 | 1.3e-16 |
| `q = 1` posterior mean / variance | 1e-9 | 4.1e-16 / 1.1e-15 |
| `q = 1` variance with observation noise | 1e-9 | 9.4e-16 |
| `q = 1` input gradients of mean / variance | 1e-8 | 9.5e-15 / 1.7e-14 |
| joint `q = 3` mean / covariance | 1e-9 | 3.2e-16 / 9.5e-14 |
| joint covariance with observation noise | 1e-9 | 1.4e-15 |
| joint `q = 3` input gradient | 1e-8 | 1.2e-11 |
| fits vs BoTorch's outcomes on the model and 16 twins (scaled distance to the nearest outcome) | best twin within 1e-4 | worst 1.7e-2 for one chaotic or multimodal model compared statistically, 0 for the others |

`mixed_last` is a multimodal likelihood: only 1 of BoTorch's 17 fits (model
and twins) reaches its best outcome within 1e-4, so it is compared
statistically (botorch_rs's outcome must lie within BoTorch's spread of
outcomes), like chaotic fits in phase 6a.

### Live parity

`botorch_pyo3/tests/test_parity_7b_models.py` (live parity, 85 tests),
`test_equivalence_7b_models.py` (randomized sweep, `equivalence` marker:
45 random multi-fidelity / mixed / MAP SAAS models and 20 random
`EnsembleModel`s) and `test_7b_models.py` (API and errors, 16 tests).
Worst errors against BoTorch main (da00e0401):

| Quantity | Tolerance | Worst error (parity) | Worst error (sweep) |
|---|---|---|---|
| parameter names, shapes, bounds, default raw parameters | exact | 0 | 0 |
| MLL (default and random parameters; MF, mixed, MAP SAAS, `OrthogonalAdditiveGP`, `subset_output`) | 1e-9 | 5.2e-15 | 1.0e-13 |
| MLL gradient | 1e-8 | 1.7e-13 | 1.8e-13 |
| posterior mean / variance / covariance (with and without observation noise) | 1e-9 | 1.1e-14 | 1.5e-13 |
| `PosteriorMean` / `PosteriorStandardDeviation` values | 1e-9 | 5.8e-15 | 8.5e-14 |
| `PosteriorMean` / `PosteriorStandardDeviation` input gradients | 1e-8 | 4.5e-14 | 8.7e-14 |
| `EnsembleMapSaasSingleTaskGP`: member and mixture moments | 1e-9 | 8.2e-15 | 1.2e-14 |
| `EnsembleMapSaasSingleTaskGP`: mixture quantiles (bisection) | 1e-6 | 1.6e-15 | |
| fits (scaled loss above BoTorch's best outcome on the model and 16 twins; above its worst) | 1 (scaled) | 3.5e-2; 0 | |
| analytic acquisition functions averaged over a MAP SAAS ensemble: values / gradients | 1e-9 / 1e-8 | 1.0e-15 / 8.6e-15 | 1.7e-15 / 4.8e-15 (LogEI) |
| `PosteriorStandardDeviation` on a MAP SAAS ensemble vs the member average (BoTorch bug 2): values / gradients | 1e-9 / 1e-8 | 1.3e-15 / 5.1e-15 | |
| MC acquisition functions on a MAP SAAS ensemble (qEI, qLogEI, qPI, qSR, qUCB, qNEI, qLogNEI): values / gradients | 1e-9 / 1e-8 | 8.8e-15 / 4.0e-14 | |
| `prune_inferior_points` on an ensemble: kept points | exact | 0 | |
| qMFKG (projection, cost, `valfunc_ei`), qMFMES, qMFGIBBON, qMFHVKG on MF GPs: values / gradients | 1e-9 / 1e-8 | 3.4e-15 / 4.8e-15 | |
| qLogNEI on a mixed GP: values / gradients | 1e-9 / 1e-8 | 8.0e-17 / 2.6e-14 | |
| `optimize_acqf_mixed`, `optimize_acqf_mixed_alternating` on a mixed GP: candidates / values | 1e-7 / 1e-9 | 7.9e-11 / 4.0e-16 | |
| `condition_on_observations` / `fantasize` (MF, mixed, MAP SAAS incl. ensemble): posteriors | 1e-9 | 1.5e-14 | |
| `subset_output` (MF, `[0]`, `[1]`, `[0, 1]`, `[1, 0]`): MLL / gradient / posteriors | 1e-9 / 1e-8 / 1e-9 | 7.7e-16 / 1.9e-14 / 1.0e-15 | |
| `AffineDeterministicModel` posterior, `subset_output` | 1e-15 | 0 | |
| `FixedSingleSampleModel`, `ModelList(GP, deterministic)`, `EnsembleModel` (`IndexSampler`): MC values / gradients | 1e-9 / 1e-8 | 8.4e-16 / 6.3e-15 | |
| `EnsemblePosterior` attributes; `IndexSampler` draws (same seed) | 1e-14; exact | 9.2e-17; 0 | |
| analytic acquisition functions on an `EnsembleModel` (moments of the `EnsemblePosterior`): values / gradients | 1e-9 / 1e-8 | 3.4e-16 / 5.3e-16 | 4.9e-16 / 8.5e-15 |
| LogEI / LogCEI gradient on an `EnsembleModel` with identical members (`|u|` up to 8e5) vs exact (mpmath) | 1e-10 | 9.6e-17 | |
| `ModelList` posterior, `output_indices`, `subset_output` | 1e-9 | 1.9e-16 | |
| `construct_inputs` (BoTorch's and botorch_rs's datasets) | exact | 0 | |

Recorded BoTorch noise (not compared): BoTorch's own LogEI gradient error
at `|u| ≈ 8e5` is 2.1e-5 (exact reference: mpmath); GPyTorch's `sq_dist`
rounding noise on the diagonal of `K(X, X)` when the inputs require
gradients changes BoTorch's Matérn-1/2 multi-fidelity MLLs by up to 5.5e-10
and their gradients by up to 9.8e-9 (relative). The live MLL comparisons
evaluate BoTorch with that diagonal zeroed (as the fixtures and phase 6a
do); without it the worst MLL gradient error would be 9.8e-9, just under
the tolerance. The candidates of the mixed optimizers use phase 9's
optimizer tolerance (1e-7).

All quantities are the same against BoTorch 0.18.1, except where 0.18.1 has
bugs that main fixed (see the verification).

### Verification

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`: clean; `cargo test --workspace`: 356 tests pass (unit tests of
  `LogTransformedInterval`, the SAAS prior terms, `AffineDeterministicModel`,
  `FixedSingleSampleModel`, `EnsembleAverage` and `EnsembleMomentsModel`
  with finite-difference gradients; `models_7b_parity.rs`; all earlier
  fixtures unchanged).
- `scripts/develop.py` (release), `ruff check` / `ruff format --check`,
  `mypy python/botorch_rs` (145 files): clean.
- `pytest botorch_pyo3/tests` with `BOTORCH_RS_REQUIRE_BOTORCH=1` against
  BoTorch main: 3230 passed, 36 skipped (none of them phase 7b tests);
  against BoTorch 0.18.1: 3209 passed, 56 skipped, 1 failed: the
  `optimize_acqf_mixed` candidates differed by 4.5e-7 (tolerance 1e-7)
  because 0.18.1 runs L-BFGS-B on its serial path, which stops at SciPy's
  `gtol = 1e-5`. The test now follows phase 9's convention for 0.18.1
  (botorch_rs's value must reach BoTorch's: shortfall 0, tolerance 1e-9),
  and the three phase 7b test files pass on both versions after that change
  (166 tests each; the other files are unaffected by it).
- `pytest -m equivalence` against main: 843 passed, 15 skipped.
- Python 3.11.15 without torch and NumPy: `botorch_rs` and every new module
  import; a `SingleTaskMultiFidelityGP` and an `AffineDeterministicModel`
  evaluate on plain lists.

## BoTorch bugs found

Reproduced on BoTorch main (da00e0401) and 0.18.1 (GPyTorch 1.15.2,
linear_operator 0.6.1), with `torch.set_default_dtype(torch.float64)`.

1. **`SingleTaskMultiFidelityGP` with `data_fidelities=[]` and an iteration
   fidelity builds an empty `ProductKernel`**: evaluating it fails.

   ```python
   m = SingleTaskMultiFidelityGP(torch.rand(8, 3), torch.rand(8, 1),
                                 iteration_fidelity=2, data_fidelities=[])
   m.posterior(torch.rand(2, 3))
   # IndexError: index 0 is out of range (torch/nn/modules/container.py)
   ```

   `_setup_multifidelity_covar_module` loops over the empty list for the
   data-fidelity kernels and multiplies their (empty) product into the
   kernel. Fix: treat `[]` as `None` (meta-pytorch/botorch#3443).

2. **`PosteriorStandardDeviation` fails on ensemble models**: the
   `average_over_ensemble_models` decorator sees a `std` of shape
   `b x num_taus`, but `forward` reshapes it to `X.shape[:-2]` first.

   ```python
   m = EnsembleMapSaasSingleTaskGP(torch.rand(8, 3), torch.rand(8, 1), num_taus=2)
   PosteriorStandardDeviation(m)(torch.rand(5, 1, 3))
   # RuntimeError: shape '[5]' is invalid for input of size 10 (acquisition/analytic.py:1171)
   ```

   Fix: return `std` without `.view(X.shape[:-2])` (as `PosteriorMean`
   returns its mean), so that the decorator averages over the ensemble
   dimension (meta-pytorch/botorch#3444). botorch_rs averages the members' standard deviations (tested
   against that reference).

3. **linear_operator: `MulLinearOperator._mul_constant` uses `if other > 0`
   on a tensor** (`mul_linear_operator.py:85`), which fails for a
   multi-output `SingleTaskMultiFidelityGP` with two data fidelities at a
   `b x 1 x d` batch of points:

   ```python
   m = SingleTaskMultiFidelityGP(torch.rand(8, 4), torch.rand(8, 2), data_fidelities=[2, 3])
   m.posterior(torch.rand(4, 1, 4))
   # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
   ```

   (`torch.rand(3, 4)` and `torch.rand(4, 2, 4)` work.) The fixture
   generator evaluates such models point by point with dense kernels.

4. **`ModelList.fantasize` indexes the sub-models by output index** (known:
   meta-pytorch/botorch#2398, fixed by #3374), so it fails as soon as a
   sub-model has several outputs:

   ```python
   X = torch.rand(8, 2)
   m = ModelList(SingleTaskGP(X, torch.rand(8, 2)), SingleTaskGP(X, torch.rand(8, 1)))
   m.posterior(X[:2])
   s = ListSampler(*[SobolQMCNormalSampler(torch.Size([2])) for _ in range(2)])
   m.fantasize(torch.rand(3, 2), sampler=s)
   # IndexError: index 2 is out of range
   ```

5. **`posterior(X, output_indices=[0, 2])` of a multi-output `SingleTaskGP`
   with `Standardize` fails** (`Standardize.untransform_posterior` uses all
   `m` outputs; known from the audit, fixed on the M-Colley fork):

   ```python
   m = SingleTaskGP(torch.rand(8, 2), torch.rand(8, 3))
   m.posterior(torch.rand(4, 2), output_indices=[0, 2])
   # RuntimeError: The size of tensor a (3) must match the size of tensor b (2) ... (transforms/outcome.py:487)
   ```

   The parity tests use whole or single-output subsets only.

6. **`EnsembleMapSaasSingleTaskGP.fantasize` fails for every shape of `X`**:
   its `posterior` inserts the ensemble dimension in front of `X`'s batch
   (`X.unsqueeze(MCMC_DIM)`), so the fantasized targets have one batch
   dimension more than GPyTorch's `get_fantasy_model` accepts (tried
   `n' x d`, `1 x n' x d`, `num_taus x n' x d`, `num_taus x 1 x n' x d`).

   ```python
   m = EnsembleMapSaasSingleTaskGP(torch.rand(8, 3), torch.rand(8, 1), num_taus=2)
   m.posterior(torch.rand(2, 3))
   m.fantasize(torch.rand(2, 3), SobolQMCNormalSampler(torch.Size([4])))
   # RuntimeError: Unsupported batch shapes: The target batch shape (torch.Size([4, 2])) must
   # have either the same dimension as or one more dimension than the input batch shape (torch.Size([]))
   ```

   Fix: sample the fantasies from the members' posteriors (the
   `SingleTaskGP` base class's `posterior` on a `num_taus x n' x d` batch of
   `X`). botorch_rs does so; its fantasy posteriors are tested against
   BoTorch's `fantasize` with that patch. meta-pytorch/botorch#3445 fixes
   it by inserting the ensemble dimension in `condition_on_observations`.

Not a bug but worth knowing: BoTorch's analytic acquisition functions on an
`EnsembleModel` are *not* averaged over the members (it has no
`_is_ensemble` flag); they use the Gaussian moments of the
`EnsemblePosterior`, and botorch_rs does the same.
