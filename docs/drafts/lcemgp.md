# Contextual multi-output GP (LCEMGP) and the Interval constraint (draft)

Draft documentation of branch `unity/lcemgp`, to be merged into the README and
`docs/api_coverage.md`. It adds what the contextual mode of the Unity project
BOforUnity (`context_support.build_contextual_model`, used by `bo.py` and
`mobo.py`) needs: BoTorch's latent context embedding multi-output GP `LCEMGP`
with the two variants the Unity project runs, and GPyTorch's `Interval`
constraint. Reference: BoTorch main `da00e0401`; the live tests also run
against the released BoTorch 0.18.1.

## Supported API

| BoTorch / GPyTorch | botorch_rs (Python) | botorch_core (Rust) | Notes |
|---|---|---|---|
| `gpytorch.constraints.Interval(lower_bound, upper_bound, transform=sigmoid, inv_transform=inv_sigmoid, initial_value=None)` | `botorch_rs.constraints.Interval` (same signature; transforms `sigmoid`, `softplus`, `None`) | `constraints::Interval`, `Constraint::Interval` | value `sigmoid(raw) · (u − l) + l` (GPyTorch's default inverse is `inv_sigmoid` whatever the transform, as in GPyTorch); `transform=None`: not enforced, the raw value is the value and `fit_gpytorch_mll` keeps it in the box `[l, u]` (L-BFGS-B bounds, as `get_parameters_and_bounds`); bounds stored in float32 and the raw initial value computed in float32, like GPyTorch; GPyTorch's errors for empty intervals and infinite bounds. Usable as `lengthscale_constraint`, `outputscale_constraint` and `noise_constraint`. `GreaterThan` and `Positive` are now subclasses of `Interval`, as in GPyTorch. |
| `models.contextual_multioutput.LCEMGP(train_X, train_Y, task_feature, train_Yvar=None, mean_module=None, covar_module=None, likelihood=None, context_cat_feature=None, context_emb_feature=None, embs_dim_list=None, output_tasks=None, all_tasks=None, outcome_transform=DEFAULT, input_transform=None)` | same path (also `botorch_rs.models`), plus the keyword-only `seed=None` | `models::LCEMGP::new(train_x, train_y, task_feature, LCEMGPOptions)` | learned embeddings (`torch.nn.Embedding(n, dim, max_norm=1.0)` per column of `context_cat_feature`) and fixed embeddings `context_emb_feature`; per-task `ConstantMean`s (`MultitaskMean`) and noise levels (`HadamardGaussianLikelihood`), `Standardize`, BoTorch's dimension-scaled RBF data kernel times an RBF kernel on the embeddings (`task_covar_module_base`, `Interval(0, 2, transform=None)` lengthscales). `posterior(X, output_indices=None, observation_noise=False, posterior_transform=None)` over the output tasks for `X` without the task column, over the given tasks with it, batched; `eval()` / `train()`; `named_parameters()` and `initialize(**{name: value})` with BoTorch's names; `emb_layers[i].weight`; `task_covar_module_base` (assignable), `covar_module`, `mean_module.base_means[t].constant`, `likelihood.noise`; `_task_embeddings()`, `_eval_context_covar()`, `task_covar_module(task_idcs)`; `copy.deepcopy`; task values validated like BoTorch (`validate_task_values=True`). |
| the Unity project's `_ProvidedEmbeddingLCEMGP` (not in BoTorch) | `botorch_rs.models.contextual_multioutput.ProvidedEmbeddingLCEMGP` (also `botorch_rs.models`), same signature as `LCEMGP`, `context_emb_feature` required | `LCEMGPOptions::provided_embedding` | botorch-rust extension: no learned embeddings (`emb_layers == []`); the task kernel `RBFKernel(ard_num_dims=m, lengthscale_constraint=Interval(0, 2, transform=None, initial_value=1))` sees only the provided embeddings; context covariance `(1 − w) · k_e(E, E) + w · I` with `w = context_specific_weight` (parameter `raw_context_specific_weight`, shape `[1]`, initial 0, `Interval(0, 1, transform=None)`, first in `named_parameters()`). `context_cat_feature` is accepted and ignored. `HAS_PROVIDED_EMBEDDING_LCEMGP = True` in the same module. |
| `models.multitask.MultiTaskGP` | same path (also `botorch_rs.models`) | `models::multitask::TaskLayout` | only as the base class of `LCEMGP` (task feature, `all_tasks` including tasks without data, `output_tasks`, task-value mapping and validation, `get_all_tasks`); creating a `MultiTaskGP` raises `UnsupportedError`. |
| `gpytorch.likelihoods.HadamardGaussianLikelihood`, `gpytorch.means.MultitaskMean` | `botorch_rs.likelihoods.HadamardGaussianLikelihood`, `botorch_rs.means.MultitaskMean` | parts of `LCEMGP` | created by `LCEMGP` (`noise` of shape `num_tasks`; `base_means`); not constructible on their own. |
| `ModelListGP(*lcemgps)`, `SumMarginalLogLikelihood` | `botorch_rs.models.ModelListGP`, `botorch_rs.mlls.SumMarginalLogLikelihood` | `models::ModelList` (a list of any `Model`s) | block-diagonal joint posterior; fitting with `fit_gpytorch_mll` (each sub-model separately, as BoTorch's `_fit_list`). `condition_on_observations` / `fantasize` require `SingleTaskGP` sub-models. |
| `ExactMarginalLogLikelihood(model.likelihood, model)` of an `LCEMGP`, `fit_gpytorch_mll` | same | `LCEMGP::mll_value_and_grad`, `Fittable for LCEMGP` | value and analytic gradient; fit with L-BFGS-B over BoTorch's parameters in BoTorch's order and bounds, retries from `sample_all_priors`; NaN losses (task lengthscale at its bound 0) handled like BoTorch's `_handle_numerical_errors`. |
| `qLogNoisyExpectedImprovement`, `qLogNoisyExpectedHypervolumeImprovement`, `prune_inferior_points(_multi_objective)`, `optimize_acqf` on `LCEMGP` / `ModelListGP` of `LCEMGP`s | same | `Model for LCEMGP`, `Model for ModelList` | joint posteriors with exact input gradients (reverse mode); all Monte-Carlo acquisition functions that accept a `SingleTaskGP` accept these models. |

What is not ported (changes to the README list): `MultiTaskGP` itself
(`IndexKernel` / `PositiveIndexKernel` task covariance with `rank`, `LKJ` and
`BetaPrior` task priors), `KroneckerMultiTaskGP`, `HeterogeneousMTGP`, the
`contextual.SACGP` / `LCEAGP` additive contextual models; for `LCEMGP`:
`train_Yvar` (fixed noise), custom `mean_module` / `likelihood`, input
transforms, outcome transforms other than `Standardize`, batched training
data, `condition_on_observations` / `fantasize` and `construct_inputs`
(`subset_output` raises `UnsupportedError`, as in BoTorch); GPyTorch's other
constraint classes (`LessThan`) and transforms other than softplus / sigmoid.

## The Unity project's models and the deviations from BoTorch

botorch_rs implements the two models of BOforUnity's `context_support.py` (the
equivalence references are those classes on BoTorch):

1. **Learned embeddings**: `LCEMGP(train_X, train_Y, task_feature,
   context_cat_feature=arange(K)[:, None], embs_dim_list=[1],
   output_tasks=[current], all_tasks=list(range(K)))`, BoTorch's `LCEMGP` with
   the Unity project's `eval()` (`_UnobservedContextLCEMGP`).
2. **Provided embeddings**: `ProvidedEmbeddingLCEMGP(train_X, train_Y,
   task_feature, context_cat_feature=..., context_emb_feature=E,
   output_tasks=[current], all_tasks=list(range(K)))` (the Unity project's
   `_ProvidedEmbeddingLCEMGP`).

`LCEMGP` with learned and fixed embeddings together (BoTorch's own use) is
supported too.

Deliberate deviations from BoTorch (the first three are what the Unity
project's models do, so botorch_rs equals them):

- **Contexts without data.** BoTorch's `MultiTaskGP.eval()` initializes the
  `IndexKernel` factors of tasks without observations, which `LCEMGP` does not
  have: BoTorch 0.18.1 and main raise an `AttributeError` for any `LCEMGP`
  with an unobserved task. botorch_rs's `eval()` (called by every `posterior`)
  instead sets the constant mean of every task without data to the average of
  the observed tasks' constants (an unobserved constant never leaves its
  initial 0 otherwise, which offsets the predictions for a new participant).
  Observed tasks are unaffected. This is the defined behaviour of botorch_rs
  for unobserved tasks.
- **Default task kernel size.** BoTorch sizes the task kernel for the learned
  embedding dimensions only (`ard_num_dims = sum(embs_dim_list)`) but
  evaluates it on learned and fixed embeddings, which fails with a
  `context_emb_feature`. botorch_rs gives it one lengthscale per embedding
  dimension (learned + fixed).
- **Per-task noise of raw task values.** The task values are mapped to task
  indices before the noise levels are looked up (meta-pytorch/botorch#3396);
  identical to BoTorch for task values `0..K-1`, as in the Unity project.
- **Initialization of learned embeddings.** botorch_rs draws them from
  `N(0, 1)` with its own generator, seeded by the keyword-only `seed`
  (random if `None`); BoTorch draws them from torch's global generator.
  Explicit values (`model.emb_layers[0].weight = W` or
  `initialize(**{"emb_layers.0.weight": W})`) reproduce any BoTorch model.
  For repeatable runs, pass `seed`.
- **Raw initial values of sigmoid `Interval`s** are computed in float32 like
  GPyTorch, but torch's float32 `log` is not correctly rounded: botorch_rs's
  raw initial value can differ from GPyTorch's by one float32 ulp (≤ 1.2e-7
  relative). `Interval(..., transform=None)` (the Unity models' constraints)
  is exact.

### Embedding renormalization and fitting

`torch.nn.Embedding(max_norm=1.0)` renormalizes the looked-up rows of its
weight in place (under `no_grad`) on every forward pass: a row with norm
`‖w‖ > 1` is scaled by `1 / (‖w‖ + 1e-7)`, and autograd differentiates the
lookup at the renormalized weights without the derivative of the
renormalization. botorch_rs reproduces this exactly: `mll()`, `posterior()`,
`_task_embeddings()`, `_eval_context_covar()` and `task_covar_module()`
renormalize the stored weights in place, and the MLL gradient with respect to
the weights is the gradient at the renormalized weights. During
`fit_gpytorch_mll`, BoTorch's L-BFGS-B state is the unrenormalized parameter
vector while every evaluation renormalizes it; botorch_rs's fit keeps the same
state, so both optimizers see the same objective and gradient and follow the
same iterates up to rounding. The fitted weights are the final L-BFGS-B
iterate (renormalized by the next forward pass), as in BoTorch.

Long L-BFGS-B runs on the flat likelihoods of these models amplify rounding:
BoTorch itself ends at a different final loss when the training inputs are
perturbed by 1e-14 or the optimizer's variables are reversed (up to 3.4e-4
in the fixtures). Fits are therefore compared by their final loss, which must
not exceed BoTorch's by more than `max(1e-6, 10 ×` that spread`)`; short fits
end at the same iterate.

## Equivalence with BoTorch

Rust fixture tests (`botorch_core/tests/lcemgp_parity.rs`, fixtures from
`scripts/fixtures_lcemgp.py`, `generate_fixtures.py --phase unity-lcemgp`,
BoTorch main `da00e0401`): the Unity models with K = 2, 3, 5 contexts
(including contexts without data), m = 1, 3 embedding dimensions, d = 1–4;
`LCEMGP` with learned and fixed embeddings; `SingleTaskGP`s with `Interval`
constraints. Worst error per quantity:

| Quantity | Tolerance | Worst error |
|---|---|---|
| default raw parameters, parameter names and bounds | exact | 0 |
| MLL at default / random parameters | 1e-9 | 3.5e-14 / 8.2e-16 |
| MLL gradient at default / random parameters | 1e-8 | 3.5e-14 / 2.0e-14 |
| renormalized embedding weights after a forward pass | exact | 0 |
| constant means after `eval()` (contexts without data) | 1e-15 | 0 |
| posterior mean / covariance (with and without observation noise, all tasks, task-column inputs) | 1e-9 | 1.4e-15 / 1.0e-14 |
| posterior VJP (input gradients) | 1e-8 | 2.5e-15 |
| fit: final loss above BoTorch's, scaled by `max(1e-6, 10 × BoTorch's spread)` | ≤ 1 | 1.4e-2 (provided K=5, m=3, d=4: +2.7e-6 with a spread of 2.0e-5; the largest unscaled excess, +1.1e-5, is learned K=3, d=2 with a spread of 3.4e-4; short fits end at the same iterate) |
| qLogNEI, q = 1, 2 (identical base samples): values / gradients, scaled by `max(nominal, 10 × noise floor)` | ≤ 1 | 0.21 / 0.10 |
| qLogNEHVI on a `ModelList` of two Unity models, q = 1, 2: values / gradients, scaled | ≤ 1 | 0.23 / 0.19 |
| `Interval` transform, inverse, derivative | 1e-15 | 0 / 0 / 5.9e-17 |
| `Interval` raw initial value | 1 float32 ulp | 1.2e-7 |
| `SingleTaskGP` with `Interval` constraints: MLL / gradient | 1e-9 / 1e-8 | 6.6e-16 / 6.8e-15 |
| `SingleTaskGP` with `Interval` constraints: fit (scaled) | ≤ 1 | 6.0e-5 |

Live tests (`botorch_pyo3/tests/test_parity_lcemgp.py`, both public Python
APIs side by side, the Unity models built exactly as `build_contextual_model`
builds them; `-m equivalence` selects a randomized sweep), against BoTorch
0.18.1 and main:

| Quantity | Tolerance | Worst error (0.18.1) | Worst error (main) |
|---|---|---|---|
| named parameters after construction, `initialize`, forward passes, `eval()` | exact | 0 | 0 |
| MLL / gradient (random parameters through `initialize`) | 1e-9 / 1e-8 | 7.0e-16 / 3.1e-15 | 7.0e-16 / 3.1e-15 |
| posterior mean / covariance (output task, all tasks, task column, batched, observation noise) | 1e-9 | 1.9e-15 / 1.2e-14 | 1.9e-15 / 1.2e-14 |
| context covariance, `task_covar_module(task_idcs)`, embeddings | 1e-9 | 1.3e-16 / 5.8e-17 / 0 | 1.3e-16 / 5.8e-17 / 0 |
| `SumMarginalLogLikelihood` of a `ModelListGP` of two Unity models | 1e-9 | 1.8e-16 | 1.8e-16 |
| fit from identical initial values: final loss above BoTorch's (unscaled; scaled) | ≤ 1 scaled | 3.3e-7; 0.16 | 3.3e-7; 0.16 |
| `ModelListGP` fit: MLL below BoTorch's | 1e-6 | 0 | 0 |
| qLogNEI values / gradients, scaled (unscaled: within BoTorch's own noise floor) | ≤ 1 | 0.10 / 0.10 | 0.10 / 0.10 |
| qLogNEHVI values / gradients, scaled (unscaled 7.7e-10 / 6.4e-7) | ≤ 1 | 0.13 / 0.06 | 0.13 / 0.06 |
| `optimize_acqf` (qLogNEI) from 4 identical initial conditions (learned, provided): restarts ending at another value than BoTorch's (relative 1e-6) | ≤ 1 of 4 | 1 | 0 |
| `Interval` values / raw initial values (float32 ulps) | 1e-15 / 2.4e-7 | 0 / 7.5e-8 | 0 / 7.5e-8 |
| `SingleTaskGP` with `Interval`: MLL / gradient; fit (scaled) | 1e-9 / 1e-8; ≤ 1 | 8.7e-16 / 6.6e-15; 0.45 | 8.7e-16 / 6.6e-15; 0.45 |
| sweep (8 random scenarios): MLL / gradient, posterior, qLogNEI values (scaled) | 1e-9 / 1e-8 / 1e-9 / ≤ 1 | 5.6e-16 / 1.7e-15 / 5.9e-16 / 4.7e-6 | 5.6e-16 / 1.7e-15 / 5.9e-16 / 4.7e-6 |

## The Unity project's contextual mode on botorch-rust

End-to-end checks with BOforUnity-rs (`origin/main` `fafb2be`; a scratch copy,
the Unity repository itself unchanged):

- `tests/test_contextual_integration.py` unchanged, with this botorch_rs on
  the path: 23 of 23 pass; runs with learned embeddings now use botorch_rs
  (`has_capability(CONTEXTUAL_CAPABILITY)`), runs with manual or image
  embeddings still fall back to BoTorch. In the whole Unity suite in one
  process, three tests that simulate a botorch_rs *without* `LCEMGP`
  (`fake_botorch_rs(lcemgp=False)`, and likewise three without
  `PosteriorMeanModel`) fail because `mock.patch.dict(sys.modules, ...)`
  leaves the real `botorch_rs.models.contextual_multioutput` imported by
  earlier tests in place; they pass on their own. The fake package should
  map the missing modules to `None`.
- With `bo_library.contextual_capability` accepting manual / image embeddings
  when `botorch_rs.models.contextual_multioutput.HAS_PROVIDED_EMBEDDING_LCEMGP`
  exists, and `_build_botorch_rs_contextual_model` building
  `ProvidedEmbeddingLCEMGP(...)` for provided embeddings and
  `LCEMGP(..., embs_dim_list=[1], seed=...)` for learned ones: the project's
  model tests (`ContextualModelTests`: twin embeddings, a new participant's
  similarity from the provided embeddings only and independent of the seed,
  the observed mean level of an unobserved context, embedding magnitudes)
  pass on botorch_rs with botorch_rs's `fit_gpytorch_mll`, and the contextual
  `bo.py` / `mobo.py` loops (warm start, new participant; learned and
  provided embeddings) run on botorch_rs without fallback and repeat exactly
  for a fixed `Seed`. The only failing test is the one that asserts the
  fallback for provided embeddings.

Suggestion quality (BoTorch and botorch-rust draw different Monte-Carlo,
restart and embedding-initialization samples, so the suggestions agree in
distribution, not bit for bit). Synthetic contextual objectives (d = 2, three
participants whose optima move with their embeddings), 5 warm-start rows each
for user_A and user_B, the current participant user_B (with its warm-start
rows) or user_C (new, no data); Unity's defaults (10 restarts, 1024 raw
samples, 128 MC samples); the real `bo.py` / `mobo.py` loops; paired by seed
(same warm start), differences botorch-rust − BoTorch with the Wilcoxon
signed-rank test:

| backend | embeddings | current participant | metric | seeds | BoTorch (mean ± se) | botorch-rust (mean ± se) | difference | p |
|---|---|---|---|---|---|---|---|---|
| `bo.py` (qLogNEI, 10 iterations) | provided | existing (user_B) | final simple regret | 12 | 0.09 ± 0.04 | 0.51 ± 0.42 | +0.42 ± 0.42 | 0.68 |
| | | | mean simple regret | 12 | 2.22 ± 0.40 | 2.21 ± 0.49 | −0.01 ± 0.15 | 0.38 |
| | provided | new (user_C) | final simple regret | 12 | 2.72 ± 1.00 | 2.82 ± 1.07 | +0.10 ± 0.27 | 0.79 |
| | | | mean simple regret | 12 | 4.61 ± 0.80 | 4.33 ± 0.85 | −0.28 ± 0.28 | 0.38 |
| | learned | existing (user_B) | final simple regret | 12 | 0.028 ± 0.009 | 0.033 ± 0.009 | +0.005 ± 0.009 | 0.68 |
| | | | mean simple regret | 12 | 1.72 ± 0.29 | 1.99 ± 0.37 | +0.27 ± 0.20 | 0.23 |
| | learned | new (user_C) | final simple regret | 52 | 1.36 ± 0.32 | 1.20 ± 0.29 | −0.15 ± 0.36 | 0.90 |
| | | | mean simple regret | 52 | 3.68 ± 0.31 | 3.78 ± 0.31 | +0.10 ± 0.21 | 0.58 |
| `mobo.py` (qLogNEHVI, 8 iterations) | provided | existing (user_B) | final hypervolume | 12 | 2.41 ± 0.12 | 2.55 ± 0.09 | +0.14 ± 0.12 | 0.47 |
| | | | mean hypervolume | 12 | 1.45 ± 0.16 | 1.59 ± 0.13 | +0.14 ± 0.11 | 0.68 |
| | provided | new (user_C) | final hypervolume | 12 | 3.04 ± 0.33 | 3.03 ± 0.33 | −0.01 ± 0.10 | 0.76 |
| | | | mean hypervolume | 12 | 1.74 ± 0.25 | 1.64 ± 0.26 | −0.11 ± 0.10 | 0.57 |
| | learned | existing (user_B) | final hypervolume | 12 | 2.52 ± 0.09 | 2.50 ± 0.11 | −0.01 ± 0.10 | 1.00 |
| | | | mean hypervolume | 12 | 1.60 ± 0.15 | 1.63 ± 0.15 | +0.03 ± 0.11 | 0.91 |
| | learned | new (user_C) | final hypervolume | 12 | 2.93 ± 0.30 | 2.68 ± 0.36 | −0.25 ± 0.45 | 0.57 |
| | | | mean hypervolume | 12 | 1.60 ± 0.27 | 1.45 ± 0.27 | −0.15 ± 0.19 | 0.42 |

Simple regret on the objective's 0–10 scale (lower is better; the mean over
the iterations measures how fast a library gets there), hypervolume of the
current participant's observations in `mobo.py`'s normalized space (higher is
better). No difference is significant (all p ≥ 0.23; the learned / new case,
p = 0.05 for the mean regret after 12 seeds, was rerun with 52 seeds).

For fixed observations (30 data sets per case: warm start plus 0–4 rows of the
current participant), BoTorch's own qLogNEI (on BoTorch's fitted model, 2048
fixed Sobol samples) scores botorch-rust's suggestion against BoTorch's; as a
reference, BoTorch suggests a second time with another torch seed:

| embeddings | current participant | initial embedding | data sets | botorch-rust − BoTorch: mean / median | BoTorch (other seed) − BoTorch: mean / median | p (botorch-rust vs other seed) | true objective, botorch-rust − BoTorch | true objective, other seed − BoTorch |
|---|---|---|---|---|---|---|---|---|
| provided | existing (user_B) | — | 30 | +0.19 / +0.0003 | +0.15 / 0.0000 | 0.07 | +0.21 ± 0.25 | +0.24 ± 0.24 |
| provided | new (user_C) | — | 30 | +0.13 / +0.0012 | −0.03 / 0.0000 | 0.10 | +0.13 ± 0.11 | −0.09 ± 0.11 |
| learned | existing (user_B) | each library's own | 30 | −0.07 / −0.0000 | −0.28 / −0.0000 | 0.48 | +0.35 ± 0.27 | +0.37 ± 0.40 |
| learned | new (user_C) | each library's own | 30 | −0.40 / −0.0045 | −1.86 / −0.0247 | 0.45 | −0.63 ± 0.30 | −0.45 ± 0.35 |
| learned | new (user_C) | BoTorch's | 30 | −0.03 / 0.0000 | −1.86 / −0.0247 | < 0.01 | −0.03 ± 0.05 | −0.45 ± 0.35 |

(qLogNEI in log space; positive: botorch-rust's suggestion has a higher
BoTorch acquisition value than BoTorch's own suggestion.) botorch-rust's
suggestions are as good as BoTorch's by BoTorch's own criterion: within the
spread of BoTorch's suggestions with another seed.

With learned embeddings, a context without data keeps its random initial
embedding, so its predictions depend on that draw in either library; given
BoTorch's initial embedding, botorch-rust's suggestion matches BoTorch's much
more closely than BoTorch's own suggestion with another seed.

Time per suggestion in the same loops, new participant (user_C), Unity's
defaults (`bo.py`: one torch thread and one botorch_rs thread; `mobo.py`:
torch's default threads, botorch_rs on 4 threads), medians over 30 (`bo.py`)
and 12 (`mobo.py`) iterations, measured while other jobs ran on the machine:

| backend | embeddings | BoTorch: fit / acquisition / suggestion | botorch-rust: fit / acquisition / suggestion | speed-up (suggestion) |
|---|---|---|---|---|
| `bo.py` | provided | 0.52 s / 0.34 s / 0.89 s | 3.6 ms / 21 ms / 24 ms | 37x |
| `bo.py` | learned | 0.77 s / 0.36 s / 1.14 s | 3.5 ms / 14 ms / 18 ms | 62x |
| `mobo.py` (two LCEMGPs) | provided | 2.35 s / 2.31 s / 4.41 s | 4.3 ms / 50 ms / 56 ms | 79x |
| `mobo.py` (two LCEMGPs) | learned | 2.16 s / 1.87 s / 4.07 s | 5.5 ms / 47 ms / 52 ms | 78x |

"Fit" is `fit_gpytorch_mll`, "acquisition" the construction and
`optimize_acqf` of qLogNEI / qLogNEHVI (10 restarts, 1024 raw samples, 128 MC
samples). The fits are 140–550 times faster.

## Unity integration (for `context_support._build_botorch_rs_contextual_model`)

- Capability: `botorch_rs.models.contextual_multioutput.HAS_PROVIDED_EMBEDDING_LCEMGP`
  (`True`), or the attribute `ProvidedEmbeddingLCEMGP` of that module (the
  `(module, attribute)` form of `bo_library.has_capability`).
- Learned embeddings: `LCEMGP(train_X=..., train_Y=y[:, j:j+1],
  task_feature=d, context_cat_feature=arange(K)[:, None], embs_dim_list=[1],
  output_tasks=[current], all_tasks=list(range(K)), seed=...)`; do not pass a
  `context_emb_feature` (that would add fixed embeddings to the learned ones,
  BoTorch's stock model). `seed` (botorch_rs extension, keyword-only) makes
  the embedding initialization repeatable; without it, it is random per model.
- Provided (manual / image) embeddings: `ProvidedEmbeddingLCEMGP(train_X=...,
  train_Y=y[:, j:j+1], task_feature=d, context_cat_feature=arange(K)[:, None],
  context_emb_feature=E, output_tasks=[current], all_tasks=list(range(K)))`.
- Both work with `ExactMarginalLogLikelihood`, `ModelListGP(*models)` with
  `SumMarginalLogLikelihood`, `fit_gpytorch_mll`, qLogNEI / qLogNEHVI and
  `optimize_acqf` from botorch_rs, and need no BoTorch workaround: `eval()`
  already averages the constants of contexts without data.
