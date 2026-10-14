# Deterministic models, composite acquisition functions and release wheels (draft)

Draft documentation of branch `unity/metataf`, to be merged into the README and
`docs/api_coverage.md`. It adds what openbo's multi-objective transfer acquisition
function (MO-TAF, used by the Unity project's MetaTAF backend) needs, and wheels
for pip-only installs. Reference: BoTorch main `da00e0401`; the live tests also
run against the released BoTorch 0.18.1.

## Supported API

| BoTorch | botorch_rs (Python) | botorch_core (Rust) | Notes |
|---|---|---|---|
| `models.deterministic.PosteriorMeanModel` | same path (also `botorch_rs.models`) | `models::PosteriorMeanModel` | the posterior mean of a `SingleTaskGP`, `ModelListGP` or deterministic model; `forward`, `posterior` (zero variance, `rsample`), `num_outputs`, `batch_shape`. The model of every acquisition function (analytic, Monte-Carlo, hypervolume), evaluated in Rust with exact gradients through the mean. |
| `models.deterministic.GenericDeterministicModel` | same path (also `botorch_rs.models`) | `Model` of the bindings | `f(X)` on torch tensors (`batch x q x d` to `batch x q x m`), `num_outputs`, `batch_shape`, `subset_output`; in acquisition functions `f` is called from Rust and differentiated with `torch.autograd`, and the input dimension must be given as the keyword-only `dim` (botorch_rs extension). `f` must be differentiable with torch (botorch_rs models are not: use `PosteriorMeanModel`). |
| `Model.is_deterministic` (no BoTorch counterpart) | `Model.is_deterministic` of the model handle | `Model::is_deterministic` | a deterministic model's Monte-Carlo samples use a zero covariance root (samples equal the mean, whatever the base samples) and its variances are not clamped (BoTorch's single-member `EnsemblePosterior`), so analytic acquisition functions use `σ = sqrt(1e-12)` like BoTorch's `_mean_and_sigma`. |
| `sampling.stochastic_samplers.StochasticSampler` | same path (also `botorch_rs.sampling`) | — | `forward(posterior)` is `posterior.rsample(sample_shape)`; `seed` is ignored for sampling, as in BoTorch. Accepted by the Monte-Carlo acquisition functions on deterministic models (BoTorch's own `StochasticSampler` too); with other models `UnsupportedError`, because botorch_rs's Monte-Carlo acquisition functions use fixed base samples. |
| a `torch.nn.Module` returning `logsumexp(log w_i + acqf_i(X)) - log Σ w_i` | `acquisition.composite.WeightedLogMeanExp(acq_functions, weights)` (also `botorch_rs.acquisition`) | `acquisition::WeightedLogMeanExp` | botorch-rust extension: `log(Σ w_i exp(α_i(X)) / Σ w_i)`, the weighted mean of log-scale acquisition functions in linear space (openbo's `_CompositeTAFEHVI`); evaluated in Rust exactly like `torch.logsumexp`, gradient `Σ softmax_i ∇α_i`; non-negative weights (terms with weight 0 are left out of both sums and not evaluated); `X_pending` of the first term, `set_X_pending` on every term; callables as terms are wrapped in `CallableAcquisitionFunction`. |
| `optimize_acqf(acq_function=<any torch callable>)` | `optim.optimize_acqf`, `acquisition.composite.CallableAcquisitionFunction(acq_function, dim)` | `AcquisitionFunction` of the bindings | like BoTorch, `optimize_acqf` accepts any callable on torch tensors (`batch x q x d` to `batch` values, e.g. a `torch.nn.Module` combining acquisition functions of either library): values come from Python, gradients from `torch.autograd`; a batch of candidate sets (raw samples) is evaluated in one call; calls hold the GIL, so restarts are serialized in Python (Rust parts of botorch_rs terms still run in parallel); `X_pending` / `set_X_pending` of the callable are used for sequential optimization; exceptions raised by the callable propagate unchanged. |
| hyperparameter setters (`outputscale`, `noise`, `constant`, `lengthscale`, `initialize`) | same | — | values with as many elements as the parameter are reshaped when they cannot be broadcast, as GPyTorch's `Module.initialize` does (`expand_as`, else `view_as`): e.g. `covar_module.outputscale = torch.tensor([v])`. |

What is not ported yet (changes to the README list): the deterministic models
`AffineDeterministicModel` and `FixedSingleSampleModel` (`PosteriorMeanModel`,
`GenericDeterministicModel` and `MatheronPathModel` are ported), and the samplers
`ForkedRNGSampler`, `IIDNormalSampler` and `IndexSampler` (`StochasticSampler`
only for deterministic models).

## Equivalence with BoTorch

Rust fixture tests (`botorch_core/tests/deterministic_parity.rs`, fixtures from
`scripts/fixtures_deterministic.py`, `generate_fixtures.py --phase
5d-deterministic`, BoTorch main `da00e0401`): two MO-TAF configurations (d = 2
and 3; two and three sources built like openbo's `_materialize_source_mean`:
`ScaleKernel` of Matern-5/2 or RBF, `Standardize`, fixed hyperparameters; a
two-output target `SingleTaskGP`), worst error per quantity:

| Quantity | Tolerance | Worst error |
|---|---|---|
| `PosteriorMeanModel(ModelListGP)` posterior mean | 1e-9 | 1.3e-15 |
| qLogEHVI on `PosteriorMeanModel` (`tau_max=1e-3`, `fat=True`), q = 1 and 2: values / gradients | 1e-9 / 1e-8 | 9.3e-15 / 1.7e-13 |
| qEHVI on `PosteriorMeanModel`, q = 1 and 2: values / gradients | 1e-9 / 1e-8 | 1.6e-14 / 0 |
| target qLogNEHVI (BoTorch's recorded base samples): values / gradients | 1e-9 / 1e-8 | 4.6e-15 / 3.6e-15 |
| `WeightedLogMeanExp` of target and sources (six weight vectors, some zero): values / gradients | 1e-9 / 1e-8 | 7.3e-15 / 2.1e-14 |
| `optimize_acqf` of the blend from BoTorch's initial conditions: restarts below BoTorch's value / candidates | ≤ 1 / 1e-6 | 0 of 8 / 5.7e-14 |

Live tests (`botorch_pyo3/tests/test_parity_deterministic.py`, both public Python
APIs side by side; `-m equivalence` adds a randomized sweep of composites):

| Quantity | Tolerance | Worst error |
|---|---|---|
| `PosteriorMeanModel` posterior mean, samples, `forward` (two `ModelListGP`s, a two-output `SingleTaskGP`) | 1e-9 | 7.4e-16 |
| qLogEHVI / qEHVI on `PosteriorMeanModel` (`StochasticSampler` of BoTorch or botorch_rs, Sobol; q = 1, 2, 3; pending points): values / gradients | 1e-9 / 1e-8 | 3.3e-15 / 1.7e-15 |
| analytic EHVI on `PosteriorMeanModel`: values / gradients | 1e-9 / 1e-8 | 3.4e-16 / 0 |
| LogEI, EI, LogPI within a few `σ = sqrt(1e-12)` of the mean: values / gradients | 1e-9 / 1e-8 | 2.5e-10 / 3.6e-10 (the improvement `μ - best_f ~ 1e-6` amplifies the 1e-16 differences of the means) |
| UCB, posterior mean, posterior standard deviation: values / gradients | 1e-9 / 1e-8 | 1.0e-16 / 0 |
| qEI, qLogEI, qUCB, qSR, qPI, qLogNEI (pruned or not) on `PosteriorMeanModel`: values / gradients | 1e-9 / 1e-8 | 6.5e-16 / 4.3e-15 |
| `GenericDeterministicModel` (a torch function) in qLogEHVI, qEHVI and LogEI: values / gradients | 1e-9 / 1e-8 | 0 / 4.0e-16 |
| LogEI far in the tail (`\|u\|` ~ 1e5) vs an exact mpmath reference: values / gradients | 1e-13 / 1e-10 | 4.3e-15 / 3.4e-16 (BoTorch: 7.6e-16 / 8.1e-6) |
| 0-dim hyperparameters set from tensors of shape `()`, `(1,)`, `(1, 1)`: stored values; posterior mean / variance | 1e-15; 1e-9 | 0; 2.7e-16 / 1.2e-16 |
| MO-TAF blend (target qLogNEHVI with identical base samples, sources qLogEHVI on `PosteriorMeanModel`), native `WeightedLogMeanExp` and the same `torch.nn.Module` over botorch_rs terms: values / gradients | 1e-9 / 1e-8 | 1.1e-14 / 3.8e-14 |
| randomized sweep (one to three sources, random weights, zero weights): values / gradients | 1e-9 / 1e-8 | 4.0e-14 / 3.8e-13 |
| `optimize_acqf` of the blend from identical initial conditions (native and callable): restarts below BoTorch's value / candidates / returned values | ≤ 1 / — / 1e-9 | 0 / 1.1e-15 / 2.2e-15 |

## Migration notes

- **Deterministic models.** As in BoTorch, posterior transforms raise
  `UnsupportedError` (BoTorch: `NotImplementedError` from
  `scalarize_posterior`) and `observation_noise=True` is ignored. Where BoTorch
  fails, botorch_rs works: a `SobolQMCNormalSampler` on a deterministic model
  (BoTorch's `EnsemblePosterior` rejects it), and qNEHVI / qLogNEHVI / qLogNParEGO
  with a `StochasticSampler` (BoTorch: "StochasticSampler does not implement
  `_update_base_samples`"). The sampler never changes the values.
- **LogEI far in the tail.** A deterministic model has `σ = sqrt(1e-12)`, so
  LogEI is evaluated at `|u|` up to about 1e6. botorch_rs's gradient is exact
  there; BoTorch's autograd has relative errors up to about 1e-5 (see the
  deliberate deviations).
- **Callable acquisition functions** are slower than native ones: every
  evaluation calls Python under the GIL. For openbo's MO-TAF the native
  `WeightedLogMeanExp` is used instead.
- **`GenericDeterministicModel`** needs `dim=d` in acquisition functions, and
  `f` must be differentiable with `torch.autograd`; a non-differentiable `f`
  raises a `RuntimeError` when gradients are requested (BoTorch fails too).

## MO-TAF (openbo) on botorch-rust

openbo's `BO_MODEL_LIBRARY` switch (branch `botorch-rust-library` of
M-Colley/openbo) runs MO-TAF with `PosteriorMeanModel` source terms and the
native `WeightedLogMeanExp` blend.

On a 3-d, two-objective problem with two sources (openbo's defaults: 5
restarts, 64 raw samples, 128 MC samples; 15 iterations after 5 initial points;
openbo's `MOTAFSequentialOptimizer` with `model_library` set per run, measured while other jobs ran on the
machine):

| | BoTorch | botorch-rust, 1 thread | botorch-rust, 4 threads |
|---|---|---|---|
| `suggest()` (fit, acquisition, `optimize_acqf`), median | 2.19 s | 54 ms (41x) | 28 ms (79x) |
| 15 suggestions | 37 s | 0.93 s | 0.44 s |

Suggestion quality (BoTorch and botorch-rust draw different Monte-Carlo and
initialization samples, so suggestions agree in distribution, not bit for bit):

- Over 10 seeds (identical initial designs), the final hypervolume is 3.924 ±
  0.021 with BoTorch and 3.942 ± 0.017 with botorch-rust (paired difference
  +0.018 ± 0.010, p = 0.10): no significant difference.
- For 30 fixed sets of observations, BoTorch's own MO-TAF acquisition function
  values botorch-rust's suggestion 0.020 (log scale) above its own on average
  (median 0), while BoTorch's suggestion with another Monte-Carlo seed is 0.026
  below (median 0): botorch-rust reaches BoTorch's optimum within BoTorch's own
  Monte-Carlo and multi-start variability (paired p = 0.27).

## Installation from wheels

The release workflow (`.github/workflows/release.yml`) builds abi3 wheels (one
per platform for CPython 3.11 and later) with maturin: manylinux x86_64 and
aarch64, Windows x86_64, macOS arm64 and x86_64, plus the sdist. It tests the
native wheels without NumPy or torch on Python 3.11 and 3.13 and, for a tag
`v*`, attaches all files to a GitHub Release (a manual run keeps them as
workflow artifacts). Installing a wheel needs no Rust toolchain:

```console
$ pip install botorch-rust --find-links https://github.com/M-Colley/botorch-rust/releases/expanded_assets/v0.1.0
$ pip install https://github.com/M-Colley/botorch-rust/releases/download/v0.1.0/<wheel file>
```

Releasing:

1. Set the version in `botorch_pyo3/pyproject.toml` (PEP 440, e.g. `0.1.0`) and
   in the workspace `Cargo.toml` (SemVer, e.g. `0.1.0`); the workflow's
   `version` job fails if they or the tag disagree.
2. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
3. PyPI (once): create a pending trusted publisher on pypi.org for project
   `botorch-rust` (owner `M-Colley`, repository `botorch-rust`, workflow
   `release.yml`, environment `pypi`), create the environment `pypi` in the
   repository settings and uncomment the `pypi` job of the workflow. Then
   `pip install botorch-rust` works.

Building wheels locally (needs Rust and maturin):

```console
$ pip install maturin
$ cd botorch_pyo3 && maturin build --release --out dist   # abi3 wheel of this platform
$ maturin sdist --out dist
```
