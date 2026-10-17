# Phase 6b: input and outcome transforms

This phase ports BoTorch's input and outcome transforms
(`botorch.models.transforms.input`, `.outcome`, `.utils`, `.factory`) and
their use in models: training (`fit_gpytorch_mll` with the learned
parameters of `Warp` and `LearnedFeatureImputation`), posteriors
(`TransformedPosterior` and `PosteriorList` with BoTorch's semantics),
conditioning, fantasies and acquisition functions. All 35 public names of
`botorch.models.transforms` now exist in botorch_rs with BoTorch's
signatures (`scripts/api_coverage.py` against BoTorch main: 35 of 35, and
every public method and property).

The core is native: an input transform is a chain of steps
(`botorch_core::transforms::input_steps`) applied according to its
`transform_on_train` / `transform_on_eval` / `transform_on_fantasize` flags
and the mode (train, eval, fantasize), with a tape for exact reverse-mode
derivatives with respect to the inputs and the learned parameters. Outcome
transforms are per-output chains of steps
(`botorch_core::transforms::outcome`), whose posteriors carry an elementwise
sample transform (`Posterior::sample_transform`) and, for `Log`, BoTorch's
log-normal moments. The existing `Normalize` / `Standardize` /
`InputPerturbation` paths are unchanged (every earlier fixture passes
bitwise-unchanged code paths: plain `Normalize`, `Standardize` and
`InputPerturbation` use the original implementations).

## Supported API (rows for the README table)

| BoTorch | botorch_rs | Notes |
|---|---|---|
| `botorch.models.transforms.input.Normalize` | same path | all arguments: `d`, `indices`, `bounds` (also batched), `batch_shape`, `transform_on_train/eval/fantasize`, `reverse`, `min_range`, `learn_bounds`, `almost_zero`, `center`; `bounds`, `offset`, `coefficient`, `learn_coefficients`, `get_init_args` (with `center` and unshifted bounds, meta-pytorch/botorch#3389), `equals`, `preprocess_transform`, `untransform` |
| `InputStandardize`, `AffineInputTransform`, `ReversibleInputTransform`, `InputTransform` | same paths | `indices`, `batch_shape`, flags, `reverse`, `min_std`; coefficients learned in train mode (BoTorch's `learn_coefficients`), fixed for `AffineInputTransform` |
| `Warp` | same path | Kumaraswamy CDF warping with learned `concentration0` / `concentration1` (`GreaterThan(1e-4)` without transform, optimizer bounds as BoTorch's), `concentration0_prior` / `concentration1_prior`, `indices`, `bounds`, `eps`, `batch_shape` (single-output batched models), flags; fitted jointly with the GP hyperparameters by `fit_gpytorch_mll` (exact gradients) |
| `Round` | same path | `integer_indices`, `categorical_features` (one-hot, rounded to the largest entry), `approximate` (BoTorch's `approximate_round` with `tau`), flags |
| `Log10` | same path | `indices`, `reverse`, flags |
| `AppendFeatures` | same path | `feature_set` or `f` (a torch callable with `indices`, `fkwargs`), `skip_expand`, flags; one-to-many (`n_f` points per input) in posteriors and Monte-Carlo acquisition functions (e.g. with risk measures); training inputs that already contain the feature columns (BoTorch's usual setup: the model's input dimension is then that of the posterior points) |
| `InteractionFeatures` | same path | `indices` (all dimensions by default) |
| `FilterFeatures` | same path | `feature_indices`, flags |
| `OneHotToNumeric`, `NumericToCategoricalEncoding` | same paths | `dim`, `categorical_features`, `encoders` (one-hot encoders evaluated natively, other callables through torch) |
| `LearnedFeatureImputation` | same path | `feature_indices`, `d`, `target_task`, `bounds` (BoTorch's `Interval` constraint on the raw values); learned imputation values fitted with the model |
| `InputPerturbation` | same path | unchanged, plus `batch_shape` (shape of the last perturbations) and `equals` (meta-pytorch/botorch#3389) |
| `ChainedInputTransform` | same path | any chain of the transforms above (keys, `equals`, `preprocess_transform`, `untransform`); steps in eval mode at construction (e.g. `get_rounding_input_transform`) as BoTorch |
| `BatchBroadcastedInputTransform` | same path | one chain per batch element of a batched model, `broadcast_index` |
| `botorch.models.transforms.outcome.Log`, `Power`, `Bilog` | same paths | `outputs` (some outputs only); `Log` transforms observation noise by the delta method, `Power` / `Bilog` raise BoTorch's `NotImplementedError`; `subset_output` re-indexes `outputs` as BoTorch main since meta-pytorch/botorch#3380; `untransform_posterior` gives a `TransformedPosterior` |
| `Standardize` | same path | now also `outputs` (some outputs only) and `subset_output` (meta-pytorch/botorch#3380) |
| `StratifiedStandardize` | same path | `stratification_idx`, `all_task_values`, `min_stdv` (strata without data or unknown values are not standardized, as BoTorch) |
| `ChainedOutcomeTransform`, `OutcomeTransform` | same paths | any chain of the outcome transforms above |
| `botorch.posteriors.TransformedPosterior`, `PosteriorList` | same paths | `mean` / `variance` (log-normal moments for `Log`; BoTorch's `NotImplementedError("No mean transform provided.")` for `Power` / `Bilog`), `rsample`, `rsample_from_base_samples`, `base_sample_shape` |
| `botorch.models.transforms.utils.*` | same path | `expand_and_copy_tensor`, `interaction_features`, `kumaraswamy_warp`, `inv_kumaraswamy_warp`, `nanstd`, `subset_transform`, `norm_to_lognorm`, `norm_to_lognorm_mean`, `norm_to_lognorm_variance`, `lognorm_to_norm` (native) |
| `botorch.models.transforms.factory.get_rounding_input_transform` | same path | `integer_indices`, `categorical_features`, `initialization`, `return_numeric`, `approximate` |
| `SingleTaskGP(input_transform=..., outcome_transform=...)` | same | every transform above (single- and multi-output, batched with `BatchBroadcastedInputTransform` / batched `Warp`); MLL and its exact gradient with respect to the learned transform parameters (with their priors); posteriors with and without observation noise; `condition_on_observations` and `fantasize` (fantasize mode skips steps with `transform_on_fantasize=False`, e.g. `InputPerturbation`, as BoTorch); analytic acquisition functions use the posterior moments, Monte-Carlo ones the transformed samples (exact gradients) |
| Monte-Carlo acquisition functions on transformed samples | same | t-batches whose samples are NaN (a `Power` transform whose inverse is undefined at negative base samples) have NaN values and gradients, as BoTorch's torch reductions give (`qExpectedImprovement`, `qProbabilityOfImprovement`, `qSimpleRegret`, `qUpperConfidenceBound`, `qLogExpectedImprovement`, `qNoisyExpectedImprovement`, `qLogNoisyExpectedImprovement`, ...) |
| `SingleTaskMultiFidelityGP`, `MixedSingleTaskGP`, `get_map_saas_model`, `AdditiveMapSaasSingleTaskGP`, `OrthogonalAdditiveGP` (phase 7b) with input and outcome transforms | same | every transform above (they are `SingleTaskGP`s in both libraries), e.g. `Warp` on the non-fidelity inputs, `Normalize(indices)` on the continuous inputs of a mixed model, `Log` / `Bilog` / `Power` outcomes; `subset_output` subsets general outcome transforms as BoTorch (`outcome_transform.subset_output`) |
| `ModelListGP` and `ModelList` with transformed sub-models | same | posterior a `PosteriorList` (as BoTorch) when a sub-model has a nonlinear outcome transform; `ScalarizedPosteriorTransform` moment-matches at `q = 1` (BoTorch's `scalarize_posterior` of a `PosteriorList`); Monte-Carlo acquisition functions sample every sub-model |

## Unported / unsupported (with BoTorch's behaviour)

- `Warp` with `reverse=True` in a model (BoTorch: inverse warping of the
  training inputs) raises `UnsupportedError`; stand-alone `untransform`
  works.
- Learned parameters shared by several batch elements of a model, e.g. a
  `Warp` (without `batch_shape`) on a multi-output `SingleTaskGP`, raise
  `UnsupportedError`: the outputs would be coupled through the shared
  concentrations. BoTorch builds such models but its `fit_gpytorch_mll`
  fails on them (see the bugs below).
- A step with learnable coefficients (e.g. a `Normalize` without bounds)
  after a step with learned parameters (`Warp`, `LearnedFeatureImputation`)
  in train mode raises `UnsupportedError`: its coefficients would depend on
  the learned parameters (BoTorch relearns them at every training forward).
- Models whose posterior points have other dimensions than their training
  inputs (e.g. `AppendFeatures` with the features in the training inputs):
  feature or perturbation functions (and `InteractionFeatures` of all
  dimensions) in their input transforms, and conditioning them, raise
  `UnsupportedError`. Fixed feature sets work.
- Native code paths built on Gaussian posteriors reject models with
  nonlinear outcome transforms (`UnsupportedError`) instead of porting their
  `TransformedPosterior` variants: `qKnowledgeGradient` (BoTorch supports a
  `Log` outcome transform there through the log-normal posterior mean of
  the fantasy models), `qHypervolumeKnowledgeGradient`, max-value entropy
  search, the noiseless fantasy models of the entropy searches,
  `qAnalyticProbabilityOfImprovement`, the `output_gps`-based helpers, and
  `LCEMGP` (only `Standardize`). Pathwise sampling supports `Normalize`
  input transforms only.
- `StratifiedStandardize` with a `batch_shape`.
- Transforms other than `Normalize` (center 0.5, no `reverse`, all
  `transform_on_*` flags; `indices` for `MultiTaskGP`) and `Standardize` (of
  all outputs) on `MultiTaskGP`, `KroneckerMultiTaskGP` and `LCEMGP`:
  `UnsupportedError` (BoTorch supports them, e.g. `StratifiedStandardize` on
  a `MultiTaskGP`; their Rust models take the plain transforms only).
  Before the merge, `StratifiedStandardize` and `Standardize(outputs=...)`
  were silently treated as a plain `Standardize` there.
- Outcome transforms other than `Standardize` on
  `EnsembleMapSaasSingleTaskGP`: `posterior` and the members of its
  acquisition functions raise `UnsupportedError` (BoTorch's `posterior`
  fails with an `AttributeError`: a `TransformedPosterior` has no
  `distribution` for the `GaussianMixturePosterior`).

## Deviations (migration notes)

- `ScalarizedPosteriorTransform` on the posterior of a single model with a
  nonlinear outcome transform raises BoTorch's message as an
  `UnsupportedError` (a `NotImplementedError`, as BoTorch's); on a
  `ModelListGP` of single-output models it moment-matches at `q = 1` without
  observation noise, as BoTorch's `scalarize_posterior` of a
  `PosteriorList` (with observation noise: `UnsupportedError`).
- Models with transforms of some outputs (`Log(outputs=[0])` and
  `Standardize(m=2)` on a two-output model): botorch_rs untransforms the
  posterior per output (each output's chain); BoTorch raises
  `NotImplementedError` in `untransform_posterior`. The live tests compare
  with single-output BoTorch models with the same parameters. Stand-alone
  `untransform_posterior` with `outputs` raises BoTorch's
  `NotImplementedError`.
- Monte-Carlo acquisition functions on a `ModelListGP` with transformed
  sub-models take one `SobolQMCNormalSampler` whose base samples stack those
  of the outputs on the last dimension (BoTorch: a `ListSampler` of one
  sampler per sub-model, with the same draws per output); `ListSampler`s are
  rejected there.
- Stand-alone `Standardize.untransform_posterior` returns a
  `TransformedPosterior` with the affine maps (BoTorch: a rescaled
  `GPyTorchPosterior`); values are identical. Models untransform natively.
- A step with learnable coefficients that is in eval mode when the model
  is built keeps its coefficients (BoTorch relearns them from the training
  inputs the next time the model is in train mode, e.g. in
  `fit_gpytorch_mll`). Fixed coefficients (as in
  `get_rounding_input_transform`) behave identically.
- `model.posterior(...)` results (moments and samples) are not connected to
  torch's autograd (as before this phase); input gradients come through the
  acquisition functions (native exact gradients, compared with BoTorch's
  autograd in the tests), and `posterior_samples_with_vjp` in Rust.
- `Normalize.indices` (and that of the other affine transforms) is a list
  of non-negative indices (negative ones are converted, as BoTorch's tensor
  indexing; phase 7a's convention), not a tensor.
- The analytic acquisition functions clamp the variance of each output at
  GPyTorch's `1e-10` only for Gaussian outputs (the moments of a
  `TransformedPosterior` are not clamped, then `_mean_and_sigma`'s `1e-12`
  applies, as BoTorch); this per-output rule replaces phase 7b's per-model
  `clamps_variance` flag where both apply.
- `SingleTaskGP.dim` is the dimension of the posterior points: for
  `AppendFeatures` not applied in training it is smaller than that of the
  training inputs.
- `SingleTaskGP.train_inputs` stay in the original input space (as before
  this phase; BoTorch's are transformed in eval mode).

## Equivalence

All tolerances are the nominal ones (values 1e-9, gradients 1e-8,
relative); none was loosened.

### Rust fixtures (`botorch_core/tests/transforms_6b_parity.rs`)

33 fixtures generated against BoTorch main
(`scripts/generate_fixtures.py --phase 6b-transforms`): every input
transform alone (`Normalize` with indices / center / bounds / reverse /
flags, `InputStandardize`, `AffineInputTransform`, `Log10`, `Warp` with and
without priors and bounds, `Round` exact and approximate, `FilterFeatures`,
`InteractionFeatures` with and without indices, `OneHotToNumeric`,
`NumericToCategoricalEncoding`, `LearnedFeatureImputation`,
`AppendFeatures`), chains (`Normalize` + `Warp`, `InputPerturbation` +
`AppendFeatures(skip_expand)`, BoTorch's rounding chain in eval mode,
`Normalize` + `Warp` + `InteractionFeatures` with `Bilog` + `Standardize`),
every outcome transform alone and chained (`Log`, `Log` + `Standardize`,
`Standardize` + `Bilog`, `Power(0.5)`, `Power(2)` + `Standardize`,
`Bilog`, `Bilog` + `Standardize`, fixed noise, `outputs=` subsets on
two-output models, `StratifiedStandardize`), and a rejected configuration
(`Warp` on a two-output model).

| Quantity | Tolerance | Worst error |
|---|---|---|
| parameter names and shapes, optimizer bounds, default raw parameters | exact | 0 |
| transformed training inputs | 1e-9 | 3.0e-16 |
| MLL (default / random parameters) | 1e-9 | 1.2e-14 / 1.9e-15 |
| MLL gradient incl. `Warp` / imputation parameters (default / random) | 1e-8 | 9.7e-14 / 1.6e-14 |
| q = 1 mean / variance (TransformedPosterior moments) | 1e-9 | 1.1e-15 / 5.0e-15 |
| q = 1 mean / variance with observation noise | 1e-9 | 9.3e-16 / 5.1e-15 |
| d mean / dx, d variance / dx | 1e-8 | 3.6e-15 / 6.0e-15 |
| posterior samples from given base samples | 1e-9 | 6.2e-15 |
| d (weighted samples) / dx | 1e-8 | 1.7e-13 |
| `ExpectedImprovement` value / gradient | 1e-9 / 1e-8 | 2.2e-15 / 9.7e-15 |
| `qExpectedImprovement` value / gradient | 1e-9 / 1e-8 | 2.6e-15 / 2.2e-14 |
| fits (best of model + 16 twins vs BoTorch's best, scaled by max(1e-6, 10 x spread)) | 1 | 3.5e-5 |
| fits (own fit above BoTorch's worst outcome, scaled) | 1 | 0 |

The same file tests BoTorch's `NotImplementedError("No mean transform
provided.")` for `Power` posteriors and the NaN values and gradients of
`qExpectedImprovement` at NaN samples.

Fits (all first attempts, 17 of 17 in both libraries): `log`, `log_std`,
`bilog`, `warp_priors_bounds` reach BoTorch's loss to 1e-10 with the same
number of iterations; `warp` and `normalize_warp` (flat warping optima,
BoTorch's own twins spread over 5.8e-7 / 6.2e-7) reach BoTorch's best twin
outcome (+0.6010006267 vs +0.6010006266; +0.8726639843 vs +0.8726639843).

### Live parity (`botorch_pyo3/tests/test_parity_6b_transforms.py`)

Against BoTorch main (da00e0401, with the #3389 fixes patched in where
noted) and 0.18.1; 97 tests including the randomized sweep (48 random
chains of input transforms of random dimensions with random outcome
transforms, marked `equivalence`; one of them, `InputStandardize` with
`Power(2)`, has NaN samples in a t-batch of `qExpectedImprovement`, whose
NaN value and gradients botorch_rs reproduces). Gradients are compared with
matching NaNs required.

| Quantity | Tolerance | Worst error |
|---|---|---|
| default raw parameters, names, shapes, bounds | exact | 0 |
| transformed training inputs | 1e-9 | 2.1e-16 |
| MLL / gradient (all models, default and random parameters, incl. sweep) | 1e-9 / 1e-8 | 4.0e-14 / 3.1e-13 |
| q = 1 mean / variance (with and without noise) | 1e-9 | 1.5e-15 / 2.2e-14 |
| samples from identical base samples | 1e-9 | 1.6e-13 |
| `PosteriorMean`, `ExpectedImprovement` value / gradient | 1e-9 / 1e-8 | 4.3e-15 / 2.4e-14 |
| `qExpectedImprovement`, `qSimpleRegret` value / gradient | 1e-9 / 1e-8 | 5.9e-15 / 2.0e-14 |
| one-to-many (`InputPerturbation`, `AppendFeatures`, both) with `CVaR` / `Expectation`, value / gradient | 1e-9 / 1e-8 | 1.4e-15 / 1.6e-13 |
| `ModelListGP`: `PosteriorList`, `qSimpleRegret` (ListSampler draws), EI with `ScalarizedPosteriorTransform` | 1e-9 / 1e-8 | 3.4e-16 / 3.1e-15 |
| outputs subsets vs single-output BoTorch models (mean, variance, samples, MLL) | 1e-9 | 1.4e-15 |
| conditioning / fantasies (`Log`, chains, `StratifiedStandardize`, `Normalize` + `Warp`, `InputPerturbation`) | 1e-9 | 1.1e-14 / 1.3e-14 |
| batched `Warp` and `BatchBroadcastedInputTransform` (MLL, gradient, posterior) | 1e-9 / 1e-8 | 3.0e-15 / 9.7e-14 |
| fits (`log`, `warp`, `warp_priors_bounds`; 4 twins) | 1 (scaled) | 2.3e-8 |
| phase-7b models with transforms (multi-fidelity + `Log`, + `Warp` + `Bilog`/`Standardize`; mixed + `Normalize(indices)` + `Log`; MAP SAAS + `InputStandardize` + `Log`; additive MAP SAAS + `Power`; orthogonal additive + `Log`): MLL / gradient | 1e-9 / 1e-8 | 4.5e-16 / 1.8e-14 |
| the same: q = 1 mean / variance, samples, `qExpectedImprovement` / gradient, `ExpectedImprovement` / gradient | 1e-9 / 1e-8 | 1.3e-15 / 2.4e-14, 4.1e-15, 4.7e-15 / 8.0e-15, 7.3e-15 / 7.5e-15 |

API and error tests: `botorch_pyo3/tests/test_6b_transforms.py` (closed
forms of the utilities, stand-alone transforms, `subset_output`,
`get_rounding_input_transform`, transformed posterior moments, BoTorch's
exceptions and messages; runs without torch and BoTorch).

## BoTorch bugs found

1. **`BatchBroadcastedInputTransform` does not register its transforms**
   (main; fixed by meta-pytorch/botorch#3389). `self.transforms` is a plain
   list, so `model.eval()` does not reach the sub-transforms: a learned
   `Normalize` keeps learning its bounds from the posterior points.
   ```python
   tf = BatchBroadcastedInputTransform([Normalize(d=2), Normalize(d=2)])
   tf.eval(); [t.training for t in tf.transforms]  # [True, True]
   ```
   botorch_rs: the transforms follow the model's mode (tests patch #3389
   into BoTorch).
2. **`InputStandardize` on a single point gives NaN** (main; fixed by
   #3389): the unbiased standard deviation of one point is NaN and
   `coefficient < min_std` is False for NaN.
   `InputStandardize(d=2)(torch.rand(1, 2))` is NaN; botorch_rs treats it as
   zero spread (coefficient 1).
3. **`Normalize.get_init_args` drops `center`** and returns the shifted
   bounds (main; fixed by #3389), so `Normalize(**n.get_init_args())`
   differs from `n` for `center != 0.5`. botorch_rs returns `center` and the
   original bounds.
4. **`Standardize.untransform_posterior`'s `TransformedPosterior` fallback
   uses the per-stratum buffers** of `StratifiedStandardize` instead of the
   per-input statistics (main; fixed by #3389). botorch_rs computes the
   per-point affine maps.
5. **Conditioning a model with `StratifiedStandardize` uses the raw inputs**
   (main and 0.18.1; not yet reported).
   `BatchedMultiOutputGPyTorchModel.condition_on_observations` calls
   `self.outcome_transform(Y, X=X)` before the input transform, while the
   model fits the transform on the transformed training inputs and
   `posterior` untransforms at transformed inputs. With an input transform
   that changes the stratification column, new observations are
   standardized with the wrong stratum:
   ```python
   X = torch.rand(12, 3); X[:, 2] = torch.arange(12) % 3
   Y = torch.randn(12, 1) + 3 * X[:, 2:]
   model = SingleTaskGP(
       X, Y,
       input_transform=AffineInputTransform(d=3, coefficient=torch.tensor([1., 1., .5]), offset=torch.zeros(3)),
       outcome_transform=StratifiedStandardize(stratification_idx=2, all_task_values=torch.arange(5)),
   )  # strata 0, 2, 4 of the transformed inputs
   model.posterior(X[:3])
   cm = model.condition_on_observations(torch.tensor([[.3, .4, 1.]]), torch.tensor([[3.5]]))
   cm.train_targets[-1]  # 3.5: the new point is standardized with stratum 1 (no data), not 2
   ```
   botorch_rs uses the transformed inputs throughout
   (`test_stratified_conditioning_uses_the_transformed_inputs`).
6. **`fit_gpytorch_mll` fails on multi-output models with a `Warp`**
   (main and 0.18.1; not yet reported). The concentrations (shape `(k,)`)
   are shared by the outputs, but the batched fit of a multi-output model
   reshapes every parameter to the model's batch shape:
   ```python
   model = SingleTaskGP(X, Y2, input_transform=Warp(d=2, indices=[0]))  # Y2: n x 2
   fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
   # RuntimeError: shape '[2, 1]' is invalid for input of size 1
   ```
   botorch_rs rejects such models (shared learned parameters across batch
   elements) when they are built.
7. **`subset_output` with `outputs`** kept the original indices in BoTorch
   0.18.1 and in the reference checkout da00e0401 (fixed in main by
   meta-pytorch/botorch#3380); botorch_rs implements #3380's re-indexing.

Also fixed in botorch_rs (a latent bug of its own): the fantasy models of
`qKnowledgeGradient` ignored a `Normalize` inside a `ChainedInputTransform`
in their input gradients (they used the Jacobian of a stand-alone
`Normalize` only); they now use the chain's Jacobian and reject chains
whose Jacobian is not constant.
