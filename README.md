# botorch-rs

*botorch-rs* is a fast implementation of [BoTorch](https://botorch.org) in
Rust, with Python bindings that provide essentially the same API as BoTorch.

## Why botorch-rs?

### Speed

Bayesian optimization usually works with a few dozen to a few hundred
observations. At that scale BoTorch's run time is dominated by overhead
rather than arithmetic: PyTorch dispatch and autograd bookkeeping for tiny
tensors, and Python-level loops around SciPy's optimizers. botorch-rs
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

Like [Rustuna](https://github.com/optuna/rustuna) for Optuna, botorch-rs
balances compatibility and performance:

- **Essentially the same API.** Module paths, class names, argument names,
  order and defaults follow BoTorch. Where botorch-rs cannot honor an
  argument (an `objective`, `posterior_transform`, outcome `constraints`,
  linear constraints in `optimize_acqf`, ...) it raises
  `botorch_rs.exceptions.UnsupportedError`, a `NotImplementedError`, instead
  of silently ignoring it.
- **The same numbers.** Every component is validated against BoTorch itself:
  identical data and hyperparameters give identical posteriors, marginal
  likelihoods and acquisition values to about 1e-10 (see
  [Accuracy](#accuracy-and-parity-with-botorch)).
- **Not a complete replacement.** botorch-rs covers the core of single-output
  Bayesian optimization with GPs. BoTorch remains the place for multi-output
  and multi-fidelity models, custom objectives, knowledge gradient and
  entropy-based acquisition functions, GPUs, and everything built on autograd.

## Installation

### Python

botorch-rs is not yet on PyPI. Install it from a source checkout (requires a
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
botorch_core = { git = "https://github.com/botorch-rs/botorch-rs" }
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
| `botorch.models.SingleTaskGP` | `botorch_rs.models.SingleTaskGP` | single output (`n x d` inputs, `n x 1` targets); inferred noise or fixed `train_Yvar`; `posterior` (`mean`, `variance`, `distribution.covariance_matrix`, `rsample`, `observation_noise=True`), `train_inputs`, `train_targets`, `named_parameters`, `initialize`, `likelihood.noise`, `covar_module.lengthscale`, ... |
| `botorch.models.transforms.Normalize`, `Standardize` | `botorch_rs.models.transforms.Normalize`, `Standardize` | learned or fixed bounds over all inputs, `center=0.5` |
| `botorch.models.utils.gpytorch_modules.*` | `botorch_rs.models.utils.gpytorch_modules.*` | BoTorch's default (dimension-scaled LogNormal) and legacy (Gamma) priors |
| `gpytorch.kernels.RBFKernel`, `MaternKernel(nu=2.5)`, `ScaleKernel` | `botorch_rs.kernels.*` | ARD kernels |
| `gpytorch.priors.LogNormalPrior`, `GammaPrior` | `botorch_rs.priors.*` | |
| `gpytorch.constraints.GreaterThan`, `Positive` | `botorch_rs.constraints.*` | softplus or no transform |
| `gpytorch.likelihoods.GaussianLikelihood`, `gpytorch.means.ConstantMean` | `botorch_rs.likelihoods.*`, `botorch_rs.means.*` | |
| `gpytorch.mlls.ExactMarginalLogLikelihood` | `botorch_rs.mlls.ExactMarginalLogLikelihood` | `mll()` evaluates it on the training data |
| `botorch.fit.fit_gpytorch_mll` | `botorch_rs.fit.fit_gpytorch_mll` | L-BFGS-B (port of L-BFGS-B 3.0 reproducing SciPy's iterates), retries from prior samples, `optimizer_kwargs={"options": {...}}`, `max_attempts` |
| `ExpectedImprovement`, `LogExpectedImprovement`, `ProbabilityOfImprovement`, `LogProbabilityOfImprovement`, `UpperConfidenceBound`, `PosteriorMean`, `PosteriorStandardDeviation` | `botorch_rs.acquisition.*` | analytic, `q = 1`, `maximize` |
| `qExpectedImprovement`, `qProbabilityOfImprovement`, `qUpperConfidenceBound`, `qSimpleRegret`, `qLogExpectedImprovement`, `qLogNoisyExpectedImprovement` | `botorch_rs.acquisition.*` | any `q`, `X_pending` / `set_X_pending`, baseline pruning, `incremental`, `fat`, `tau_max`, `tau_relu` |
| `botorch.acquisition.utils.prune_inferior_points` | same path | |
| `acqf(X)`, `torch.autograd.grad(acqf(X), X)` | same | exact gradients computed in Rust; `acqf.value_and_grad(X)` for NumPy and lists |
| `botorch.sampling.SobolQMCNormalSampler` | `botorch_rs.sampling.SobolQMCNormalSampler` | explicit `base_samples` supported |
| `botorch.optim.optimize_acqf` | `botorch_rs.optim.optimize_acqf` | box bounds, `fixed_features`, `batch_initial_conditions`, `sequential`, `return_best_only`, `options` (`seed`, `batch_limit`, `init_batch_limit`, `eta`, `alpha`, `nonnegative`, `maxiter`, `maxfun`, `ftol`, `gtol`, `maxcor`, `maxls`) |
| `botorch.utils.sampling.draw_sobol_samples`, `draw_sobol_normal_samples` | `botorch_rs.utils.sampling.*` | |
| `botorch.test_functions.*` | `botorch_rs.test_functions.*` | Ackley, Beale, Branin, Bukin, Cosine8, DixonPrice, DropWave, EggHolder, Griewank, Hartmann, HolderTable, Levy, Michalewicz, Powell, Rastrigin, Rosenbrock, Shekel, SixHumpCamel, StyblinskiTang, ThreeHumpCamel |
| `botorch.exceptions` | `botorch_rs.exceptions` | `BotorchRsError` hierarchy (also `ValueError`, `RuntimeError`, `NotImplementedError` subclasses) and BoTorch's warnings |

Not available: multi-output, batched and other model classes (`ModelListGP`,
`MultiTaskGP`, ...), `condition_on_observations` / `fantasize`, objectives,
posterior transforms and outcome constraints, input transforms other than
`Normalize`, other acquisition functions (knowledge gradient, entropy search,
multi-objective), linear and nonlinear constraints and custom
`gen_candidates` in `optimize_acqf`, and GPU execution.

## Accuracy and parity with BoTorch

Correctness is defined by agreement with BoTorch. `scripts/generate_fixtures.py`
records reference values from BoTorch for the Rust tests, and the Python tests
compare against BoTorch directly (`botorch_pyo3/tests/test_parity.py`, run in
CI against the released BoTorch):

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

Gradient errors are relative to the largest gradient component. The Rust
fixtures cover 17 GP configurations (d ∈ {1, 2, 5}, n ∈ {1, …, 20}, RBF and
Matérn-5/2, `ScaleKernel`, fixed noise, learned and explicit `Normalize`,
with and without `Standardize`, legacy Gamma priors, degenerate data) and
three acquisition models; the Python tests cover eight model configurations,
including custom kernels and priors built from `botorch_rs.kernels` and
`botorch_rs.priors`.

Monte-Carlo acquisition functions are compared with identical base samples.
Fitting reaches BoTorch's final loss (required: no worse than BoTorch's +
1e-6): from BoTorch's default initialization, 16 of 17 Rust fixture problems
take the same number of L-BFGS-B iterations, and the largest excess observed
is 3.4e-7 (1e-11 for the Python configurations, typically 1e-15). On flat
likelihood surfaces, L-BFGS-B's relative-reduction stopping rule can end at
points that differ by round-off along long trajectories (up to ~2e-6 in the
loss on BO-loop data); in those cases BoTorch's own optimizer, restarted from
botorch-rs' solution, makes no further progress (checked along the BO loop of
`test_bo_loop.py`). `optimize_acqf` reaches BoTorch's best acquisition value
on fixed models (LogEI and qLogEI on Hartmann-6), and started from BoTorch's
initial conditions it reproduces BoTorch's optimized candidates to about
1e-15.

## Benchmarks

Wall time through the Python API (`scripts/benchmark.py`; median of 5 runs,
3 for the BO loop, after a warm-up run) on an Intel Core Ultra 7 155H laptop
(22 logical CPUs, Windows 11), Python 3.13, PyTorch 2.13 on the CPU, BoTorch
main at da00e04. "1 thread" sets `torch.set_num_threads(1)` and
`botorch_rs.set_num_threads(1)`; otherwise each library uses its default
(BoTorch/PyTorch: 16 threads, botorch-rs: 22).

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
(to 1e-6), the same LogEI optimum, qLogEI −0.900625 (botorch-rs) vs −0.900679
(BoTorch), and best Hartmann values −3.160 vs −3.106 after the loop (a single
design; over 30 designs the loops are statistically indistinguishable, median
−3.156 vs −3.159). Timings vary by about ±20% between runs on this laptop.

The speedups are largest where BoTorch's per-call overhead dominates: small
models and many cheap evaluations. Fitting cost grows as O(n³); at n = 300 the
dense linear algebra dominates and BoTorch's optimized BLAS/LAPACK catches up
(botorch-rs uses `nalgebra`, single-threaded). Restarts of `optimize_acqf` run
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
  `torch.manual_seed` does not seed botorch-rs: use `options={"seed": ...}` in
  `optimize_acqf` (which also seeds the random choice of initial conditions)
  and `SobolQMCNormalSampler(..., seed=...)`. qLogNEI prunes its baseline with
  a seed derived from its sampler's seed, so whole BO loops are reproducible.
  `posterior.rsample` and test-function noise do use the caller's library
  (torch, NumPy) random number generator.
- **Python floats.** `best_f`, `beta` and `tau` are used as exact float64
  values; BoTorch stores Python floats passed for them in float32 (torch's
  default dtype). Pass float64 tensors (e.g. `best_f=train_Y.max()`) to
  BoTorch for bit-comparable results.
- **Model snapshots.** An acquisition function uses the model as it is when
  the acquisition function is created; after refitting a model, create a new
  acquisition function (the usual BO-loop pattern).
- **qLogNEI** always uses the plain joint computation (BoTorch's
  `cache_root=False`); `cache_root` is accepted and ignored. Both agree in
  distribution.
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
`python scripts/generate_fixtures.py` regenerates the Rust parity fixtures
(`--phase 1` for the GP core, `--phase 2` for acquisition functions,
optimization and test functions), and `python scripts/benchmark.py` measures
the timings above. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT (see `LICENSE`). `botorch_core/src/lbfgsb.rs` is a port of L-BFGS-B 3.0
(Zhu, Byrd, Lu, Nocedal, Morales; C version by S. Becker), distributed under
the BSD 3-clause license whose notice is retained in that file. Quasi-random
numbers use the `sobol_burley` crate (Burley's hash-based Owen scrambling with
Joe–Kuo direction numbers).

## Acknowledgements and citation

botorch-rs re-implements algorithms designed and implemented in
[BoTorch](https://github.com/meta-pytorch/botorch) and
[GPyTorch](https://github.com/cornellius-gp/gpytorch); its structure follows
[Rustuna](https://github.com/optuna/rustuna). If you use botorch-rs, please
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
