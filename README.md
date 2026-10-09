# botorch-rs

A Rust implementation of [BoTorch](https://botorch.org) (Bayesian optimization
with Gaussian processes), in the spirit of
[optuna/rustuna](https://github.com/optuna/rustuna): a dependency-light Rust
core crate, with Python bindings (planned) that expose essentially the same
API as the original.

## Status

The `botorch_core` crate implements, in double precision:

**Phase 1: Gaussian-process core.** BoTorch's `SingleTaskGP` for a single
output:

- RBF and Matérn-5/2 ARD kernels (GPyTorch parameterization), optional
  `ScaleKernel`, with gradients with respect to hyperparameters and inputs;
- LogNormal and Gamma priors, `GreaterThan`/`Positive` constraints;
- BoTorch's default priors and initialization
  (`get_covar_module_with_dim_scaled_prior`,
  `get_gaussian_likelihood_with_lognormal_prior`) and the legacy Gamma-prior
  presets;
- homoskedastic and fixed-noise Gaussian likelihoods;
- `Normalize` input and `Standardize` outcome transforms;
- posteriors (mean and joint covariance, optional observation noise) and their
  derivatives with respect to the test inputs (forward and reverse mode);
- the exact marginal log likelihood with analytic gradients, and
  `fit_gpytorch_mll` built on a port of L-BFGS-B 3.0 that reproduces scipy's
  iterates.

**Phase 2: acquisition functions and their optimization.**

- a `Model` trait (implemented by `SingleTaskGP`) through which acquisition
  functions access posteriors and their derivatives;
- analytic acquisition functions (`q = 1`): `ExpectedImprovement`,
  `LogExpectedImprovement`, `ProbabilityOfImprovement`,
  `LogProbabilityOfImprovement`, `UpperConfidenceBound`, `PosteriorMean`,
  `PosteriorStandardDeviation`;
- Monte-Carlo acquisition functions for any `q`, with pending points:
  `qExpectedImprovement`, `qLogExpectedImprovement`,
  `qProbabilityOfImprovement`, `qUpperConfidenceBound`, `qSimpleRegret` and
  `qLogNoisyExpectedImprovement` (with `prune_inferior_points`), using a seeded
  quasi-MC `SobolQMCNormalSampler` (sample-average approximation);
- exact gradients of every acquisition function (reverse mode through the
  posterior samples, the Cholesky factor and the posterior);
- `optimize_acqf`: Sobol raw samples, Boltzmann / non-negative selection of
  initial conditions, multi-start L-BFGS-B with restarts in parallel (rayon),
  fixed features, sequential greedy optimization;
- `draw_sobol_samples`, `draw_sobol_normal_samples`, numerically safe special
  functions (`log_ndtr`, `log1mexp`, `fatplus`, `fatmax`, ...);
- 20 synthetic test functions (`Hartmann`, `Branin`, `Ackley`, ...).

Numbers are validated against BoTorch itself: `scripts/generate_fixtures.py`
records reference values from BoTorch/GPyTorch, and the parity tests require
agreement to 1e-10 for kernels, transforms, posteriors, MLL values and
gradients (observed about 1e-14), fitted losses no worse than BoTorch's, and
acquisition values and gradients (observed at most 6e-11, typically 1e-14;
Monte-Carlo acquisition functions are compared with identical base samples).
L-BFGS-B started from BoTorch's initial conditions reproduces BoTorch's
optimized candidates to about 1e-15.

Planned: PyO3 Python bindings (phase 3).

## Usage

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

A complete Bayesian optimization loop on Hartmann-6:

```sh
cargo run --release --example bo_loop -- [logei|qlogei|qlognei] [iterations] [seed] [q]
```

## Performance

`optimize_acqf` with 10 restarts and 512 raw samples on a 50-point
Hartmann-6 model with fixed hyperparameters
(`examples/benchmark_optimize_acqf.rs` and the identical
`scripts/benchmark_optimize_acqf.py`; median of 9 runs on a laptop with 22
logical CPUs; BoTorch main at da00e04 (0.18.1+), PyTorch 2.13):

| | botorch-rs, 1 thread | botorch-rs, 22 threads | BoTorch, 1 thread | BoTorch, 16 threads (default) |
|---|---|---|---|---|
| `LogExpectedImprovement`, q = 1 | 1.1 ms | 0.8 ms | 111 ms | 171 ms |
| `qLogExpectedImprovement`, q = 3, 512 MC samples | 53 ms | 13 ms | 939 ms | 1692 ms |

The Hartmann-6 loop of `examples/bo_loop.rs` (LogEI, 14 + 30 evaluations,
including model fitting) takes about 0.2 s; over 30 identical initial designs
its results are statistically indistinguishable from the same loop run with
BoTorch (median best value -3.156 vs -3.159).

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test                                     # unit, parity, gradient and behavior tests
cargo test --test parity -- --nocapture        # print the BoTorch parity tables (GP core)
cargo test --test acquisition_parity -- --nocapture
cargo test --test optimize_parity -- --nocapture
```

Regenerate the reference fixtures with BoTorch installed (see the header of
`scripts/generate_fixtures.py`; `--phase 1` only regenerates the GP fixtures,
`--phase 2` the acquisition, optimization and test-function fixtures):

```sh
python scripts/generate_fixtures.py
```

## License

MIT (see `LICENSE`). `botorch_core/src/lbfgsb.rs` is a port of L-BFGS-B 3.0
(Zhu, Byrd, Lu, Nocedal, Morales; C version by S. Becker), distributed under
the BSD 3-clause license whose notice is retained in that file. The Sobol
sequence uses the `sobol_burley` crate (Burley's hash-based Owen scrambling,
Joe–Kuo direction numbers).
