# botorch-rust

*botorch-rust* is a fast implementation of [BoTorch](https://botorch.org) in
Rust, with Python bindings that provide essentially the same API as BoTorch.
It is an independent project, not affiliated with or endorsed by Meta or the
BoTorch team.

## Why botorch-rust?

### Speed

Bayesian optimization usually works with a few dozen to a few hundred
observations. At that scale BoTorch's run time is dominated by overhead
rather than arithmetic: PyTorch dispatch and autograd bookkeeping for tiny
tensors, and Python-level loops around SciPy's optimizers. botorch-rust
evaluates the same mathematics natively in double precision, with analytic
gradients and multi-start optimizations running in parallel. A complete
30-iteration BO loop on Hartmann-6 (model fitting plus acquisition
optimization in every iteration) takes about 0.2 s instead of 18 s, and a
`LogExpectedImprovement` optimization with 10 restarts about 1 ms instead of
0.1 s (see [Benchmarks](#benchmarks)). This makes BO practical inside tight
loops: many repetitions for benchmarking, cheap objective functions, or
interactive use.

### No PyTorch dependency

The Python package `botorch_rs` has no runtime dependencies, not even NumPy.
Arrays may be torch tensors, NumPy arrays or nested lists of floats, and
results come back with the type of the corresponding input. The Rust crate
`botorch_core` can be used directly from Rust.

### The same API

Typical BoTorch code runs after changing the imports (GPyTorch's modules
that BoTorch users touch, such as `gpytorch.mlls`, live in `botorch_rs` too):

```python
import torch
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms import Normalize, Standardize
from botorch_rs.mlls import ExactMarginalLogLikelihood   # gpytorch.mlls in BoTorch
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.acquisition import LogExpectedImprovement
from botorch_rs.optim import optimize_acqf
from botorch_rs.test_functions import Hartmann

problem = Hartmann(dim=6, negate=True)
train_X = torch.rand(20, 6, dtype=torch.float64)
train_Y = problem(train_X).unsqueeze(-1)
model = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=6),
                     outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
acqf = LogExpectedImprovement(model, best_f=train_Y.max())
candidate, value = optimize_acqf(acqf, bounds=problem.bounds, q=1,
                                 num_restarts=10, raw_samples=512)
```

## Design philosophy

Like [Rustuna](https://github.com/optuna/rustuna) for Optuna, botorch-rust
balances compatibility and performance:

- **Essentially the same API.** Module paths, class names, argument names,
  order and defaults follow BoTorch. Where botorch-rust cannot honor an
  argument (a `post_processing_func` or custom `gen_candidates` in
  `optimize_acqf`, a multi-output objective, ...) it raises
  `botorch_rs.exceptions.UnsupportedError`, a `NotImplementedError`, instead
  of silently ignoring it.
- **The same numbers.** Every component is validated against BoTorch itself:
  identical data and hyperparameters give identical posteriors, marginal
  likelihoods and acquisition values to about 1e-10 (see
  [Equivalence with BoTorch](#equivalence-with-botorch)).
- **Not a complete replacement.** botorch-rust covers the core of Bayesian
  optimization with independent GPs (single- and multi-output, batched, model
  lists), MC objectives, outcome constraints and constrained acquisition
  optimization. BoTorch remains the place for multi-task and multi-fidelity
  models, multi-objective optimization, knowledge gradient and entropy-based
  acquisition functions, GPUs, and everything built on autograd through
  models.

## Installation

### Python

botorch-rust is not yet on PyPI. Install it from a source checkout (requires a
Rust toolchain, see [rustup.rs](https://rustup.rs)):

```console
$ pip install ./botorch_pyo3
```

This builds an abi3 wheel with [maturin](https://www.maturin.rs) that works on
CPython 3.11 and later. For development, `python scripts/develop.py` builds the
extension into the source tree instead (see [Development](#development)).

### Rust

```toml
[dependencies]
botorch_core = { git = "https://github.com/M-Colley/botorch-rust" }
```

## Quickstart

### Python

A Bayesian optimization loop with batch acquisition (`q = 2`) and the noisy
LogEI acquisition function; all arrays are torch tensors here, but NumPy
arrays or nested lists work the same way:

```python
import torch
from botorch_rs.acquisition import qLogNoisyExpectedImprovement
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms import Normalize
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.test_functions import Branin
from botorch_rs.utils.sampling import draw_sobol_samples

problem = Branin(negate=True, noise_std=0.1)
bounds = problem.bounds
train_X = draw_sobol_samples(bounds, n=8, q=1, seed=0).squeeze(-2)
train_Y = problem(train_X).unsqueeze(-1)
for iteration in range(10):
    model = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=2))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    acqf = qLogNoisyExpectedImprovement(
        model, X_baseline=train_X, sampler=SobolQMCNormalSampler(torch.Size([256]), seed=iteration)
    )
    candidates, _ = optimize_acqf(acqf, bounds, q=2, num_restarts=10, raw_samples=256,
                                  options={"seed": iteration})
    train_X = torch.cat([train_X, candidates])
    train_Y = torch.cat([train_Y, problem(candidates).unsqueeze(-1)])

posterior = model.posterior(train_X)
print(posterior.mean.shape, posterior.variance.shape)  # (28, 1), (28, 1)
```

### Rust

```rust
use std::sync::Arc;

use botorch_core::acquisition::LogExpectedImprovement;
use botorch_core::optim::{optimize_acqf, OptimizeAcqfOptions};
use botorch_core::{fit_gpytorch_mll, FitOptions, Matrix, Model, SingleTaskGP, SingleTaskGPOptions};

let train_x = Matrix::from_fn(12, 2, |i, j| ((i * 7 + j * 3) % 12) as f64 / 11.0);
let train_y = Matrix::from_fn(12, 1, |i, _| (6.0 * train_x[(i, 0)]).sin() + train_x[(i, 1)]);
let mut model = SingleTaskGP::new(train_x, train_y.clone(), SingleTaskGPOptions::default())?;
fit_gpytorch_mll(&mut model, FitOptions::default())?;

let model: Arc<dyn Model> = Arc::new(model);
let acqf = LogExpectedImprovement::new(model, train_y.max(), true)?;
let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
let options = OptimizeAcqfOptions { seed: Some(0), ..Default::default() };
let (candidate, value) = optimize_acqf(&acqf, &bounds, 1, 10, 512, &options)?;
```

`botorch_core/examples/bo_loop.rs` runs a complete loop on Hartmann-6
(`cargo run --release --example bo_loop`); the module documentation of
`botorch_core` (`cargo doc --open`) maps every BoTorch function to its Rust
counterpart.

## Supported API

| BoTorch / GPyTorch | botorch_rs | Notes |
|---|---|---|
| `botorch.models.SingleTaskGP` | `botorch_rs.models.SingleTaskGP` | `batch_shape x n x d` inputs and `batch_shape x n x m` targets (`m ≥ 1` independent outputs, any batch shape: independent hyperparameters per output and batch element, BoTorch's `_aug_batch_shape`); inferred noise or fixed `train_Yvar`; `posterior(X, output_indices, observation_noise, posterior_transform)` with BoTorch's shapes (`mean`/`variance` `batch x q x m`, `MultitaskMultivariateNormal` covariance `batch x qm x qm` in BoTorch's output-major layout, batch broadcasting, `rsample`, `rsample_from_base_samples`), `train_inputs`, `train_targets`, `named_parameters`, `initialize`, `likelihood.noise`, `covar_module.lengthscale`, ... |
| `botorch.models.ModelListGP` | `botorch_rs.models.ModelListGP` | independent single- or multi-output `SingleTaskGP`s with different training data; block-diagonal joint posterior over the concatenated outputs, `output_indices`, `posterior_transform`, `likelihood` (`LikelihoodList`), `named_parameters` |
| `botorch.models.transforms.Normalize`, `Standardize` | `botorch_rs.models.transforms.Normalize`, `Standardize` | learned or fixed bounds over all inputs, `center=0.5`; `Standardize(m, batch_shape)`, `Normalize(d, batch_shape)` / batched bounds (single-output batched models) |
| `botorch.models.utils.gpytorch_modules.*` | `botorch_rs.models.utils.gpytorch_modules.*` | BoTorch's default (dimension-scaled LogNormal) and legacy (Gamma) priors |
| `gpytorch.kernels.RBFKernel`, `MaternKernel(nu=2.5)`, `ScaleKernel` | `botorch_rs.kernels.*` | ARD kernels |
| `gpytorch.priors.LogNormalPrior`, `GammaPrior` | `botorch_rs.priors.*` | |
| `gpytorch.constraints.GreaterThan`, `Positive` | `botorch_rs.constraints.*` | softplus or no transform |
| `gpytorch.likelihoods.GaussianLikelihood`, `gpytorch.means.ConstantMean` | `botorch_rs.likelihoods.*`, `botorch_rs.means.*` | |
| `gpytorch.mlls.ExactMarginalLogLikelihood`, `SumMarginalLogLikelihood` | `botorch_rs.mlls.*` | `mll()` evaluates it on the training data (one value per output and batch element; the sum over a `ModelListGP`'s sub-models divided by their number) |
| `botorch.fit.fit_gpytorch_mll` | `botorch_rs.fit.fit_gpytorch_mll` | L-BFGS-B (port of L-BFGS-B 3.0 reproducing SciPy's iterates), retries from prior samples, `optimizer_kwargs={"options": {...}}`, `max_attempts`; multi-output and batched models are fitted per element like BoTorch's batched independent fitting, `ModelListGP`s per sub-model |
| `ExpectedImprovement`, `LogExpectedImprovement`, `ProbabilityOfImprovement`, `LogProbabilityOfImprovement`, `UpperConfidenceBound`, `PosteriorMean`, `PosteriorStandardDeviation` | `botorch_rs.acquisition.*` | analytic, `q = 1`, `maximize`, `posterior_transform` |
| `LogConstrainedExpectedImprovement`, `ConstrainedExpectedImprovement` | `botorch_rs.acquisition.*` | analytic, `objective_index`, `constraints={i: (lower, upper)}` (`None`: unbounded), `maximize` |
| `qExpectedImprovement`, `qProbabilityOfImprovement`, `qUpperConfidenceBound`, `qSimpleRegret`, `qLogExpectedImprovement`, `qLogNoisyExpectedImprovement` | `botorch_rs.acquisition.*` | any `q`, `X_pending` / `set_X_pending`, baseline pruning, `incremental`, `fat` (also a per-constraint list), `tau_max`, `tau_relu`, `posterior_transform`, `objective`; outcome `constraints` (callables, feasible iff `<= 0`) and `eta` (a float or one per constraint) for qEI, qPI, qLogEI and qLogNEI (feasibility-weighted incumbents and BoTorch's lower bound for infeasible baselines) |
| `qLogProbabilityOfFeasibility` | `botorch_rs.acquisition.qLogProbabilityOfFeasibility` | `constraints`, `eta`, `fat`, `tau_max`, `posterior_transform` |
| `IdentityMCObjective`, `LinearMCObjective`, `GenericMCObjective`, `ConstrainedMCObjective`, `MCAcquisitionObjective` | `botorch_rs.acquisition.objective.*` | linear objectives are evaluated in Rust; generic objectives, `ConstrainedMCObjective` (`infeasible_cost`, `eta`), subclasses and outcome constraints are callables on torch tensors, evaluated from Rust with gradients from `torch.autograd` (requires torch); BoTorch's own objective objects are accepted too |
| `botorch.acquisition.objective.ScalarizedPosteriorTransform` | same path | `weights`, `offset`; `evaluate`, `posterior_transform(posterior)`; accepted by `model.posterior`, all acquisition functions and `prune_inferior_points` |
| `botorch.acquisition.utils.prune_inferior_points` | same path | `objective`, `posterior_transform`, `constraints`, `sampler` |
| `acqf(X)`, `torch.autograd.grad(acqf(X), X)` | same | exact gradients computed in Rust; `acqf.value_and_grad(X)` for NumPy and lists |
| `botorch.sampling.SobolQMCNormalSampler` | `botorch_rs.sampling.SobolQMCNormalSampler` | explicit `base_samples` supported |
| `botorch.optim.optimize_acqf` | `botorch_rs.optim.optimize_acqf` | box bounds, `inequality_constraints` / `equality_constraints` (BoTorch's `(indices, coefficients, rhs)`, intra- and inter-point), `nonlinear_inequality_constraints` (`(callable, is_intrapoint)`, torch callables), `fixed_features`, `batch_initial_conditions`, `ic_generator` (called with BoTorch's keyword arguments; `gen_batch_initial_conditions` runs natively), `sequential`, `return_best_only`, `options` (`seed`, `method` (`"L-BFGS-B"`, `"SLSQP"`), `batch_limit`, `init_batch_limit`, `eta`, `alpha`, `nonnegative`, `maxiter`, `maxfun`, `ftol`, `gtol`, `maxcor`, `maxls`, `n_burnin`, `n_thinning`); with constraints, SLSQP over chunks of `batch_limit` restarts, projection of infeasible results and `CandidateGenerationError` as in BoTorch |
| `botorch.optim.initializers.gen_batch_initial_conditions`, `sample_q_batches_from_polytope`, `transform_constraints` | `botorch_rs.optim.initializers.*` | linear constraints (hit-and-run polytope sampling of the raw samples), `fixed_features`, `options` |
| `botorch.optim.parameter_constraints.evaluate_feasibility` | same path | linear and nonlinear constraints, `tolerance` |
| `botorch.utils.sampling.draw_sobol_samples`, `draw_sobol_normal_samples` | `botorch_rs.utils.sampling.*` | |
| `botorch.utils.sampling.get_polytope_samples`, `HitAndRunPolytopeSampler`, `find_interior_point`, `sample_polytope`, `sparse_to_dense_constraints` | `botorch_rs.utils.sampling.*` | |
| `botorch.test_functions.*` | `botorch_rs.test_functions.*` | Ackley, Beale, Branin, Bukin, Cosine8, DixonPrice, DropWave, EggHolder, Griewank, Hartmann, HolderTable, Levy, Michalewicz, Powell, Rastrigin, Rosenbrock, Shekel, SixHumpCamel, StyblinskiTang, ThreeHumpCamel |
| `botorch.exceptions` | `botorch_rs.exceptions` | `BotorchRsError` hierarchy (also `ValueError`, `RuntimeError`, `NotImplementedError` subclasses) and BoTorch's warnings |

Everything else in BoTorch is not ported yet: see
[What is not ported yet](#what-is-not-ported-yet).

## What is not ported yet

botorch-rust covers the core of Bayesian optimization with exact Gaussian
processes, which is a small part of BoTorch: it provides **92 of the 773
public classes and functions** defined in BoTorch's modules (12%; BoTorch main
at da00e04). `scripts/api_coverage.py` compares the two libraries module by
module and writes every missing class, function, parameter and method to
[docs/api_coverage.md](docs/api_coverage.md). The main gaps:

- **Models:**
  - multi-task GPs (`MultiTaskGP`, `KroneckerMultiTaskGP`,
    `HeterogeneousMTGP`);
  - multi-fidelity and mixed or categorical GPs (`SingleTaskMultiFidelityGP`,
    `MixedSingleTaskGP`);
  - fully Bayesian and MAP SAAS models (`SaasFullyBayesianSingleTaskGP` with
    NUTS, `get_map_saas_model`);
  - variational GPs (`SingleTaskVariationalGP`), `HigherOrderGP` and
    `LatentKroneckerGP`;
  - preference learning (`PairwiseGP`);
  - contextual and additive GPs;
  - deterministic, ensemble and cost models;
  - relevance pursuit, and the empirical and hierarchical model packages.
- **Model features:**
  - `condition_on_observations`, `fantasize`, `subset_output` and
    `construct_inputs`;
  - autograd through models (acquisition values do support `torch.autograd`
    with respect to their inputs);
  - kernels other than RBF, Matérn-5/2 and `ScaleKernel`, including BoTorch's
    own kernels for categorical inputs, fidelities and additive structure;
  - priors other than LogNormal and Gamma, and likelihoods other than
    Gaussian;
  - input transforms other than `Normalize` (`InputStandardize`, `Warp`,
    `Round`, `AppendFeatures`, `InputPerturbation`, chained transforms, ...);
  - outcome transforms other than `Standardize` (`Log`, `Power`, `Bilog`,
    chained transforms, ...);
  - GPU execution and float32 arithmetic: everything runs in float64 on the
    CPU.
- **Single-objective acquisition functions:**
  - `qNoisyExpectedImprovement` and the analytic noisy EI variants;
  - `qLowerConfidenceBound`, `qPosteriorStandardDeviation`,
    `ScalarizedPosteriorMean`, and the analytic `LogProbabilityOfFeasibility`;
  - knowledge gradient (`qKnowledgeGradient` and its multi-fidelity variant);
  - entropy search (max-value entropy search, GIBBON, joint and predictive
    entropy search) and `qMultiStepLookahead`;
  - active learning (`qNegIntegratedPosteriorVariance`, BALD) and preference
    learning (EUBO);
  - risk measures (VaR, CVaR, ...) and cost-aware utilities;
  - penalized, proximal, prior-guided, fixed-feature and decoupled wrappers,
    multi-output acquisition functions, and pathwise Thompson sampling.
- **Multi-objective optimization:**
  - all multi-objective acquisition functions (`qLogNoisyExpectedHypervolumeImprovement`,
    `qLogExpectedHypervolumeImprovement`, `qLogNParEGO`,
    `qHypervolumeKnowledgeGradient`, `MOMF`, multi-objective entropy search,
    ...);
  - their objectives, risk measures, and hypervolume and Pareto utilities.
- **Objectives and transforms:** posterior transforms other than
  `ScalarizedPosteriorTransform`, learned and multi-output MC objectives, and
  `get_acquisition_function` with the input constructors.
- **Optimization and candidate generation:**
  - `optimize_acqf_cyclic`, `optimize_acqf_list`, `optimize_acqf_mixed`
    (also the alternating variant), `optimize_acqf_discrete` (also with local
    search) and `optimize_acqf_homotopy`;
  - in `optimize_acqf`: custom `gen_candidates` (`gen_candidates_torch`),
    `post_processing_func`, `timeout_sec`, `sample_around_best` and
    batch-shaped fixed features;
  - initial conditions for one-shot acquisition functions;
  - `MaxPosteriorSampling` and `BoltzmannSampling`;
  - fitting with torch optimizers, BoTorch's closures and stopping criteria.
- **Sampling and utilities:**
  - samplers other than `SobolQMCNormalSampler` (`IIDNormalSampler`,
    `StochasticSampler`, `ListSampler`, ...) and pathwise (Matheron) sampling;
  - most of `botorch.utils`: multi-objective, probability (bivariate and
    truncated multivariate normal, elliptical slice sampling), low-rank and
    dataset utilities;
  - cross-validation and `settings`.
- **Test functions and community code:** test functions other than the 20
  synthetic ones above (multi-objective, multi-fidelity, constrained
  engineering problems, sensitivity analysis, ...), and everything in
  `botorch_community`.

<details>
<summary>Coverage by area (public classes and functions)</summary>

| Area | Public classes and functions | In botorch_rs | Share |
|---|---|---|---|
| `acquisition` | 145 | 27 | 19% |
| `acquisition.multi_objective` | 33 | 0 | 0% |
| `cross_validation` | 7 | 0 | 0% |
| `exceptions` | 20 | 15 | 75% |
| `fit` | 3 | 1 | 33% |
| `generation` | 8 | 0 | 0% |
| `logging` | 1 | 0 | 0% |
| `models` | 66 | 3 | 5% |
| `models.empirical_gps` | 36 | 0 | 0% |
| `models.hierarchical` | 6 | 0 | 0% |
| `models.kernels` | 19 | 0 | 0% |
| `models.likelihoods` | 5 | 0 | 0% |
| `models.transforms` | 35 | 4 | 11% |
| `models.utils` | 26 | 4 | 15% |
| `optim` | 58 | 5 | 9% |
| `optim.closures` | 5 | 0 | 0% |
| `optim.utils` | 14 | 0 | 0% |
| `posteriors` | 13 | 2 | 15% |
| `sampling` | 14 | 2 | 14% |
| `sampling.pathwise` | 25 | 0 | 0% |
| `settings` | 3 | 0 | 0% |
| `test_functions` | 84 | 21 | 25% |
| `test_utils` | 2 | 0 | 0% |
| `utils` | 97 | 8 | 8% |
| `utils.multi_objective` | 22 | 0 | 0% |
| `utils.probability` | 26 | 0 | 0% |
| **all of `botorch`** | **773** | **92** | **12%** |
| `botorch_community` | 55 | 0 | 0% |

</details>

## Equivalence with BoTorch

botorch-rust is correct when it computes what BoTorch computes. Agreement is
tested at two levels:

1. **Rust fixture tests.** `scripts/generate_fixtures.py` runs a reference
   BoTorch and records inputs and results as JSON files
   (`botorch_core/tests/fixtures/`; `--phase 1` GP core, `--phase 2`
   acquisition functions, optimization and test functions, `--phase 4`
   multi-output, batched and list models, posterior transforms and
   constrained EI, `--phase 4b` MC objectives, outcome constraints,
   constrained optimization and polytope sampling). The
   `botorch_core/tests/*parity*.rs` tests reproduce them with the Rust core
   alone, without Python (`cargo test`).
2. **Live side-by-side Python tests.** The Python tests build the same models
   in BoTorch and in botorch_rs (identical data, hyperparameters and
   Monte-Carlo base samples) and compare the results through the public APIs:
   `test_parity.py` (GP core, acquisition functions, optimization, test
   functions and API signatures), `test_parity_multi_output.py`
   (multi-output, batched and list models, posterior transforms, constrained
   EI), `test_parity_constraints.py` (MC objectives, outcome constraints,
   parameter constraints, polytope sampling, constrained optimization) and
   the unified, randomized suite `test_equivalence.py`.

Every parity module prints the worst error of every quantity in the pytest
terminal summary.

### Running the comparisons

```console
$ cargo test --workspace                       # Rust fixture tests
$ python scripts/generate_fixtures.py --phase 4b  # regenerate fixtures (needs BoTorch)
$ python scripts/develop.py                    # build the extension
$ export BOTORCH_RS_REQUIRE_BOTORCH=1          # a missing BoTorch is an error, not a skip
$ PYTHONPATH=/path/to/botorch:botorch_pyo3/python python -m pytest botorch_pyo3/tests -q
$ PYTHONPATH=/path/to/botorch:botorch_pyo3/python python -m pytest botorch_pyo3/tests -q -m equivalence
```

The last command runs only the unified suite (about 3 minutes); without a
BoTorch checkout on `PYTHONPATH`, the installed BoTorch is used (separate
`PYTHONPATH` entries with `;` on Windows). CI's parity job runs the Python
tests and, as a separate step, the equivalence suite against the released
BoTorch.

### The unified equivalence suite

`test_equivalence.py` (pytest marker `equivalence`) draws 18 scenarios from a
seeded generator: n ∈ {6, 10, 16} training points, d ∈ {1, …, 4} inputs,
m ∈ {1, 2, 3} outputs, inferred or fixed noise, with and without `Normalize`
and `Standardize`, RBF or Matérn-5/2 kernels, three input ranges, batched
models and four `ModelListGP`s. Data, hyperparameters, test points, base
samples and the remaining choices (q, pending points, objectives, outcome
constraints, `eta`, `fat`, posterior transforms, parameter constraints) are
random but seeded. For every scenario it compares initialization, MLL values
and gradients, posteriors (joint, batched, with observation noise,
scalarized) and samples from base samples, fits, all analytic acquisition
functions, all Monte-Carlo acquisition functions with identity, linear,
generic (input-dependent) and constrained objectives, posterior transforms
and outcome constraints, `prune_inferior_points`, and `optimize_acqf` with
and without linear (intra- and inter-point, equality) and nonlinear
constraints; plus all synthetic test functions and random polytopes. Results
of the last run (157 tests in 3 minutes; BoTorch main at da00e04):

| Quantity | Tolerance | Worst observed |
|---|---|---|
| Default raw hyperparameters | 1e-12 | exact |
| MLL values, MLL gradients | 1e-9 (relative) | 6.5e-15, 8.1e-15 |
| Posterior mean, variance, covariance (incl. observation noise and posterior transforms) | 1e-9 | 4.2e-15, 7.7e-14, 9.4e-14 |
| Samples from explicit base samples | 1e-9 | 6.6e-15 |
| Fit: final loss above BoTorch's | 1e-6 | 9.6e-15 |
| Analytic acquisition values / gradients (EI, LogEI, PI, LogPI, UCB, posterior mean and standard deviation, CEI, LogCEI) | 1e-9 / 1e-8 | 1.3e-13 / 1.2e-13 |
| MC acquisition values / gradients (qEI, qPI, qLogEI, qSR, qUCB, qLogPF; all objectives, transforms and constraints) | 1e-9 / 1e-8, or 10× BoTorch's noise floor if larger (see below); errors scaled to the nominal tolerance | 1.2e-13 / 5.8e-13 |
| qLogNEI values / gradients (incl. tight and infeasible baselines) | as above | 9.1e-11 / 9.5e-10 |
| Unscaled errors of all these Monte-Carlo evaluations (values / gradients) | | 3.7e-9 / 9.3e-9 |
| MC evaluations on numerically singular joint posteriors (jitter-dependent) | finite, 0.1 | 7 evaluations, 2.8e-3 |
| `prune_inferior_points` | identical sets | identical |
| `optimize_acqf` from identical initial conditions, L-BFGS-B / SLSQP (incl. nonlinear constraints) | ≥ 80% of the restarts within 1e-6 | 159 of 160 restarts; agreeing restarts within 1.7e-14 (L-BFGS-B) and 2.3e-7 (SLSQP) |
| `optimize_acqf` end to end over 3 seeds (statistical) | best value ≥ BoTorch's median − 1e-3·max(1, \|v\|) | 8.4e-9 below; 1.1e-2 below BoTorch's best seed |
| Returned acquisition values vs BoTorch's value at the returned candidates | 1e-9 | 1.6e-14 |
| Infeasible results of constrained runs | none | none |
| Test functions: values, bounds, optimal values | 1e-12, exact, 1e-15 | 2.9e-16, exact, exact |
| Polytope samples: mean / covariance (statistical) | 0.05 range / 0.02 range² | 1.6e-2 / 3.6e-3 |
| `find_interior_point` slack, `evaluate_feasibility`, `transform_constraints` | 1e-9, identical, identical | 2e-17, identical, identical |

Two numerical effects need tolerances beyond plain rounding. *Nearly singular
joint posteriors* (qLogNEI and pruning sample the posterior at the training
points) amplify rounding: perturbing BoTorch's own training inputs by 1e-14
(relative) changes its MC values by up to 4.1e-9 and gradients by up to
4.6e-8. The suite therefore compares Monte-Carlo quantities against
`max(1e-9, 10 × that change)` (1e-8 for gradients), measured per evaluation
with two perturbed BoTorch twins, and reports errors scaled to the nominal
tolerance. *Numerically singular* covariance blocks (smallest eigenvalue below
1e-12 of the largest) make the decision whether Cholesky needs jitter depend
on rounding: LAPACK and botorch-rust can decide differently, which moves the
samples along the singular directions; such evaluations are counted and only
checked loosely. The single diverging optimization restart starts on a flat
part of the acquisition function (log value −44) and ends in another local
optimum (−1.7194 vs −1.7143); end to end, BoTorch's initial conditions come
from torch's global random number generator, so its best seed can be better
(or worse) than botorch-rust's.

### Targeted parity tests

| Quantity | Tolerance (relative) | Observed, Rust fixtures | Observed, Python API |
|---|---|---|---|
| Default hyperparameter initialization (incl. GPyTorch's float32 rounding) | 1e-12 | exact | exact |
| Kernel matrices, transforms | 1e-10 | ≤ 6e-14 | |
| Exact MLL value | 1e-10 / 1e-9 | ≤ 6e-14 (3e-12 for condition number ~3e6) | ≤ 6e-15 |
| MLL gradient | 1e-10 | ≤ 6e-14 | |
| Posterior mean, variance, joint covariance (with and without observation noise) | 1e-10 / 1e-9 | ≤ 6e-14 | ≤ 2e-14 (2e-12 for a mean close to zero) |
| Posterior derivatives with respect to the inputs | 1e-10 | ≤ 6e-14 | |
| Analytic acquisition values | 1e-10 / 1e-9 | ≤ 6e-11 (EI/PI deep in the tail), typically 1e-14 | ≤ 5e-13 |
| Analytic acquisition gradients | 1e-10 / 1e-8 | ≤ 6e-11 | ≤ 2e-14 |
| MC acquisition values (qEI, qLogEI, qPI, qUCB, qSR, qLogNEI; q ≤ 3, pending points) | 1e-10 / 1e-9 | ≤ 6e-13 | ≤ 3e-14 |
| MC acquisition gradients | 1e-10 / 1e-8 | ≤ 3e-11 | ≤ 2e-14 |
| `prune_inferior_points` | identical selection | identical | identical |
| Synthetic test functions | 1e-12 | ≤ 3e-16 | ≤ 1e-14 |
| Multi-output, batched and `ModelListGP` models: MLL values and gradients, joint posteriors (batched and broadcast inputs, observation noise) and their input derivatives, samples from base samples | 1e-10 / 1e-9 | ≤ 5e-14 | ≤ 3e-14 |
| Acquisition values with a `ScalarizedPosteriorTransform` (analytic, MC, qLogNEI) | 1e-10 / 1e-9 | ≤ 4e-13 | ≤ 3e-13 |
| Acquisition gradients with a `ScalarizedPosteriorTransform` | 1e-10 / 1e-8 | ≤ 6e-13 | ≤ 3e-11 |
| `LogConstrainedExpectedImprovement`, `ConstrainedExpectedImprovement` values and gradients | 1e-10 / 1e-9 | ≤ 4e-12 | ≤ 9e-14 |
| Samples of multi-output posteriors from BoTorch's `S x q x m` base samples | 1e-10 / 1e-9 | ≤ 4e-15 | ≤ 2e-15 |
| MC acquisition values with linear, generic and constrained objectives and outcome constraints (qEI, qPI, qLogEI incl. `fat` lists, qSR, qUCB, qLogPF) | 1e-10 / 1e-9 | ≤ 4e-13 | ≤ 2.3e-13 |
| ... their gradients | 1e-10 / 1e-8 | ≤ 3e-13 | ≤ 3.7e-13 |
| qLogNEI with objectives and constraints (baselines feasible, partly or entirely infeasible) values / gradients | 1e-10 / 1e-9, 1e-8 | ≤ 6e-14 / ≤ 1.5e-12 | ≤ 3.7e-11 / ≤ 2e-10 |
| `get_infeasible_cost`; `prune_inferior_points` with objectives and constraints | 1e-10; identical | ≤ 2e-14; identical | identical |
| SLSQP from identical initial conditions: restarts at SciPy's result (value within 1e-6) | — | 21 of 22 (errors ≤ 3.8e-7) | 48 of 50 (errors ≤ 2.3e-7; one diverging restart better in each library) |
| Constrained `optimize_acqf` over seeds: best value below BoTorch's | 1e-4 | ≤ 4.6e-9 | ≤ 4.8e-8 |
| Polytope samples: mean / covariance error (statistical) | 0.03 range / 0.015 (Rust), 0.01 (Python) range² | ≤ 1.3e-2 / ≤ 3.6e-3 | ≤ 1.9e-2 / ≤ 2.7e-3 |
| `find_interior_point` (unique maximizers), `sparse_to_dense_constraints`, `transform_constraints`, `evaluate_feasibility`, error types and messages | 1e-9; identical | ≤ 1e-9 | identical |

Gradient errors are relative to the largest gradient component. The Rust
fixtures cover 17 GP configurations (d ∈ {1, 2, 5}, n ∈ {1, …, 20}, RBF and
Matérn-5/2, `ScaleKernel`, fixed noise, learned and explicit `Normalize`,
with and without `Standardize`, legacy Gamma priors, degenerate data), three
acquisition models, 13 multi-output configurations (`m` = 1, 2, 3; batch
shapes `3`, `2`, `2 x 2`; three `ModelListGP`s) and, for objectives and
constraints, five single-output, multi-output and list models; the Python
parity tests cover eight single-output and 15 multi-output model
configurations, including custom kernels and priors built from
`botorch_rs.kernels` and `botorch_rs.priors`, and six models for objectives
and constraints. Monte-Carlo acquisition functions are compared with
identical base samples, and Python floats that BoTorch would round to float32
(`best_f`, `tau`, `beta`, `eta`, `infeasible_cost`) are passed as float64
tensors.

Fitting reaches BoTorch's final loss (required: no worse than BoTorch's +
1e-6): from BoTorch's default initialization, 16 of 17 Rust fixture problems
take the same number of L-BFGS-B iterations, and the largest excess observed
is 3.4e-7 (1e-11 for the Python configurations, typically 1e-15). On flat
likelihood surfaces, L-BFGS-B's relative-reduction stopping rule can end at
points that differ by round-off along long trajectories (up to ~2e-6 in the
loss on BO-loop data); in those cases BoTorch's own optimizer, restarted from
botorch-rust's solution, makes no further progress (checked along the BO loop of
`test_bo_loop.py`). Multi-output and batched models reach BoTorch's final loss
of every output and batch element (largest excess 5.5e-13 on the fixtures, 1e-10
through the Python API), with the same iteration counts for all 36 fitted
fixture elements. `optimize_acqf` reaches BoTorch's best acquisition value
on fixed models (LogEI and qLogEI on Hartmann-6), and started from BoTorch's
initial conditions it reproduces BoTorch's optimized candidates to about
1e-15 (L-BFGS-B). SLSQP iterates agree with SciPy's to about 1e-9 per step:
SciPy 1.16 and later solve the SLSQP subproblems with LAPACK, botorch-rust with
the port of Kraft's original code, so long runs on flat landscapes can end in
different local optima, in either direction.

### Versions checked

The numbers above come from BoTorch's main branch at da00e04 with GPyTorch
1.15.2, linear_operator 0.6.1, PyTorch 2.13 (CPU), SciPy 1.18 and Python 3.13
on Windows 11; CI runs the Python comparisons against the latest BoTorch
release. An earlier run against the released BoTorch 0.18.1 found two
behaviors that differ from main, which the tests now account for: 0.18.1
rejects SLSQP starts that violate nonlinear constraints (main and botorch-rust
accept them), and with SciPy 1.18 it optimizes all L-BFGS-B restarts in one
joint problem instead of one problem per restart.

## Benchmarks

Wall time through the Python API (`scripts/benchmark.py`; median of 5 runs,
3 for the BO loop, after a warm-up run) on an Intel Core Ultra 7 155H laptop
(22 logical CPUs, Windows 11), Python 3.13, PyTorch 2.13 on the CPU, BoTorch
main at da00e04. "1 thread" sets `torch.set_num_threads(1)` and
`botorch_rs.set_num_threads(1)`; otherwise each library uses its default
(BoTorch/PyTorch: 16 threads, botorch-rust: 22).

| Task | botorch_rs, 1 thread | BoTorch, 1 thread | speedup | botorch_rs, 22 threads | BoTorch, 16 threads | speedup |
|---|---|---|---|---|---|---|
| fit, n = 20 | 1.5 ms | 322.9 ms | 218x | 1.9 ms | 459.6 ms | 247x |
| fit, n = 100 | 25.1 ms | 201.9 ms | 8.0x | 19.0 ms | 305.4 ms | 16x |
| fit, n = 300 | 335.9 ms | 418.2 ms | 1.2x | 276.6 ms | 535.0 ms | 1.9x |
| optimize_acqf, LogEI, q = 1 | 1.0 ms | 112.2 ms | 110x | 1.3 ms | 165.7 ms | 128x |
| optimize_acqf, qLogEI, q = 3 | 104.6 ms | 2.82 s | 27x | 38.3 ms | 2.57 s | 67x |
| optimize_acqf, qLogNEI, q = 1 | 24.4 ms | 207.7 ms | 8.5x | 12.0 ms | 247.0 ms | 21x |
| BO loop, 30 iterations | 210.0 ms | 17.67 s | 84x | 233.4 ms | 21.09 s | 90x |

Tasks (Hartmann-6): `fit` is `SingleTaskGP` + `fit_gpytorch_mll` on n Weyl
points from BoTorch's default initialization; `optimize_acqf` uses 10 restarts
and 512 raw samples on a 50-point model with fixed hyperparameters (qLogEI
with the same 512 explicit base samples in both libraries); the BO loop is the
loop of `examples/bo_loop.rs` (14 Sobol points, then 30 × fit + LogEI
optimization). Both libraries reach the same results: identical fitted losses
(to 1e-6), the same LogEI optimum, qLogEI −0.900625 (botorch-rust) vs −0.900679
(BoTorch), and best Hartmann values −3.160 vs −3.106 after the loop (a single
design; over 30 designs the loops are statistically indistinguishable, median
−3.156 vs −3.159). Timings vary by about ±20% between runs on this laptop.

The speedups are largest where BoTorch's per-call overhead dominates: small
models and many cheap evaluations. Fitting cost grows as O(n³); at n = 300 the
dense linear algebra dominates and BoTorch's optimized BLAS/LAPACK catches up
(botorch-rust uses `nalgebra`, single-threaded). Restarts of `optimize_acqf` run
in parallel, which pays off for the MC acquisition functions; for analytic
LogEI one optimization takes about a millisecond either way.

## Migration notes and known differences

- **Imports.** Replace `botorch.` by `botorch_rs.`; GPyTorch's
  `mlls`, `kernels`, `priors`, `constraints`, `likelihoods` and `means` are
  `botorch_rs.mlls`, `botorch_rs.kernels`, ... (only the classes listed above).
- **Arrays.** Inputs may be torch tensors, NumPy arrays or nested lists;
  results have the array type, floating dtype and device of the input they are
  computed from (float64 for integer inputs). All computations are in float64.
  Results that are not computed from an array (a test function's `bounds`,
  `draw_sobol_normal_samples`) are float64 tensors if torch has been imported
  and lists otherwise; pass `dtype=` (a torch or NumPy dtype) to choose.
- **Random numbers.** Quasi-random points come from an Owen-scrambled Sobol
  sequence (`sobol_burley`), not torch's `SobolEngine`: for the same seed, raw
  samples and MC base samples differ from BoTorch's (with the same
  distribution); explicit `sampler.base_samples` reproduce BoTorch exactly.
  `torch.manual_seed` does not seed botorch-rust: use `options={"seed": ...}` in
  `optimize_acqf` (which also seeds the random choice of initial conditions)
  and `SobolQMCNormalSampler(..., seed=...)`. qLogNEI prunes its baseline with
  a seed derived from its sampler's seed, so whole BO loops are reproducible.
  `posterior.rsample` and test-function noise do use the caller's library
  (torch, NumPy) random number generator.
- **Python floats.** `best_f`, `beta`, `tau`, the constraint temperatures
  `eta`, `infeasible_cost` and the bounds of the constrained EI are used as
  exact float64 values; BoTorch stores Python floats passed for them in
  float32 (torch's default dtype). Pass float64 tensors (e.g.
  `best_f=train_Y.max()`) to BoTorch for bit-comparable results.
- **Multi-output and batched models.** Hyperparameters are independent per
  output and batch element (BoTorch's default modules with `batch_shape =
  _aug_batch_shape`); custom modules must have that batch shape, and modules
  shared by all outputs raise `UnsupportedError` (BoTorch's batched fitting
  cannot fit them either). Fitting runs one L-BFGS-B problem per element like
  BoTorch's `fmin_l_bfgs_b_batched` (without a `maxfun` limit; passing
  `maxfun` warns, as in BoTorch); a failed element is retried on its own from
  prior samples, whereas BoTorch accepts an abnormal termination of an element
  and restarts all elements after a numerical error. A `Normalize` with a
  batch shape is only supported for single-output batched models: for several
  outputs BoTorch broadcasts the per-batch bounds against the output
  dimension. `output_indices` of a multi-output `SingleTaskGP` select outputs
  (BoTorch fails for proper subsets). Acquisition functions require unbatched
  models.
- **`ModelListGP` posteriors** are the block-diagonal concatenation of the
  sub-model posteriors. With linear_operator 0.6.1, BoTorch's own result
  mixes up the covariance blocks of multi-output sub-models in some
  configurations (`BlockDiagLinearOperator` slicing in `separate_mtmvn`), and
  its posterior transforms fail for `q > 1` (block indexing in
  `scalarize_posterior`); the parity references evaluate these BoTorch code
  paths with dense covariances.
- **Model snapshots.** An acquisition function uses the model as it is when
  the acquisition function is created; after refitting a model, create a new
  acquisition function (the usual BO-loop pattern).
- **qLogNEI** always uses the plain joint computation (BoTorch's
  `cache_root=False`); `cache_root` is accepted and ignored. Both agree in
  distribution; objectives and constraints are evaluated on the joint samples
  of the baseline and the candidates, as with `cache_root=False` (this only
  matters for callables that couple the points of a batch). When some sample
  has no feasible baseline point, the objective's lower bound is estimated at
  32 points drawn from a generator seeded by the sampler (BoTorch:
  `torch.rand`); pass them explicitly with the keyword-only
  `lower_bound_points` (`acqf.lower_bound_points` returns them). BoTorch's
  warning that qLogProbabilityOfFeasibility is the better choice is emitted
  at the first evaluation that needs the bound.
- **Objectives and constraints.** Python callables (generic objectives,
  `ConstrainedMCObjective`, outcome constraints, nonlinear parameter
  constraints) receive float64 torch tensors of BoTorch's shapes for a single
  candidate set (`sample_shape x 1 x q x m` samples, `1 x q x d` inputs) and
  are called from the Rust computation (the GIL is re-acquired); an exception
  raised by a callable propagates unchanged. Mismatched numbers of `eta` or
  `fat` entries are reported when the acquisition function is created
  (BoTorch: at the first evaluation), with BoTorch's messages.
- **Constrained optimization.** SLSQP follows SciPy's driver (`ftol` 1e-6,
  `maxiter` 2000), but solves its subproblems with the port of Kraft's code,
  whereas SciPy 1.16 and later use LAPACK; restarts that start on flat parts
  of the acquisition function can therefore end in different local optima
  (in either direction). Initial conditions that violate nonlinear
  constraints are accepted, as on BoTorch's main branch (the released 0.18.1
  rejects them). `options={"method": "L-BFGS-B"}` with constraints
  raises `RuntimeWarning("Method L-BFGS-B cannot handle constraints.")` like
  BoTorch's batched L-BFGS-B path. Polytope samplers use their own random
  streams, and `find_interior_point` solves its linear program with `microlp`
  (BoTorch: HiGHS), which may return a different maximizer when it is not
  unique.
- **Nearly singular covariances.** Whether a Cholesky factorization needs
  jitter is decided by rounding when a covariance block is numerically
  singular (smallest eigenvalue around 1e-16 of the largest); botorch-rust and
  LAPACK can decide differently there, which changes Monte-Carlo samples
  along the singular directions (see [Equivalence with
  BoTorch](#equivalence-with-botorch)).
- **Marginal log likelihood.** `mll()` without arguments evaluates the MLL on
  the model's training data (GPyTorch: `mll(model(*model.train_inputs),
  model.train_targets)`); `mll.value_and_grad()` also returns its gradient.
- **Gradients.** There is no autograd through models. Acquisition values
  support `torch.autograd` (exact gradients computed in Rust), and
  `acqf.value_and_grad(X)` returns gradients for any array type.
- **Fitting.** `pick_best_of_all_attempts`, custom optimizers and closures are
  not supported; retries sample hyperparameters from their priors with a
  seeded generator (`fit_gpytorch_mll(mll, seed=...)`).
- **Warnings and errors** follow BoTorch (`InputDataWarning` for unscaled
  data, `NumericsWarning` for legacy EI, `BadInitialCandidatesWarning`).
  Errors raised by the Rust core derive from
  `botorch_rs.exceptions.BotorchRsError` and from the matching built-in
  exception.
- **Threads.** `botorch_rs.set_num_threads(n)` sets the number of threads
  (default: one per logical CPU); the GIL is released during all computations.
- **Deliberate deviations.** `prune_inferior_points` with
  `max_frac < 1` keeps the most frequent maximizers (BoTorch indexes the wrong
  points there), the LogEI gradient is accurate far in the tail, and posterior
  samples of a single point use `sqrt(max(v, 0))` where BoTorch yields NaN.

## Development

```console
$ cargo fmt --all --check
$ cargo clippy --workspace --all-targets -- -D warnings
$ cargo test --workspace                     # unit, BoTorch-parity and doc tests
$ python scripts/develop.py                  # build the extension into botorch_pyo3/python
$ PYTHONPATH=botorch_pyo3/python python -m pytest botorch_pyo3/tests -q
```

The Python parity tests run when BoTorch is importable; to test against a
source checkout put it first on the path (`PYTHONPATH=/path/to/botorch:botorch_pyo3/python`).
`-m equivalence` selects the unified equivalence suite (see [Equivalence with
BoTorch](#equivalence-with-botorch)).
`python scripts/generate_fixtures.py` regenerates the Rust parity fixtures
(`--phase 1` for the GP core, `--phase 2` for acquisition functions,
optimization and test functions, `--phase 4` for multi-output, batched and
list models, posterior transforms and constrained EI, `--phase 4b` for MC
objectives, outcome constraints, constrained optimization and polytope
sampling), and `python scripts/benchmark.py` measures the timings above, and
`python scripts/api_coverage.py` regenerates the API coverage report
([docs/api_coverage.md](docs/api_coverage.md)). See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT (see `LICENSE`). `botorch_core/src/lbfgsb.rs` is a port of L-BFGS-B 3.0
(Zhu, Byrd, Lu, Nocedal, Morales; C version by S. Becker), distributed under
the BSD 3-clause license whose notice is retained in that file. Quasi-random
numbers use the `sobol_burley` crate (Burley's hash-based Owen scrambling with
Joe–Kuo direction numbers).

## Acknowledgements and citation

botorch-rust re-implements algorithms designed and implemented in
[BoTorch](https://github.com/meta-pytorch/botorch) and
[GPyTorch](https://github.com/cornellius-gp/gpytorch); its structure follows
[Rustuna](https://github.com/optuna/rustuna). If you use botorch-rust, please
cite BoTorch:

```bibtex
@inproceedings{balandat2020botorch,
  title={{BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization}},
  author={Balandat, Maximilian and Karrer, Brian and Jiang, Daniel R. and Daulton, Samuel and Letham, Benjamin and Wilson, Andrew Gordon and Bakshy, Eytan},
  booktitle={Advances in Neural Information Processing Systems 33},
  year={2020},
  url={http://arxiv.org/abs/1910.06403}
}
```

and, for the LogEI family of acquisition functions, Ament et al.,
*Unexpected Improvements to Expected Improvement for Bayesian Optimization*
(NeurIPS 2023), and for the default priors, Hvarfner et al., *Vanilla
Bayesian Optimization Performs Great in High Dimensions* (ICML 2024).
