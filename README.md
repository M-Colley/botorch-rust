# botorch-rs

A Rust implementation of [BoTorch](https://botorch.org) (Bayesian optimization
with Gaussian processes), in the spirit of
[optuna/rustuna](https://github.com/optuna/rustuna): a dependency-light Rust
core crate, with Python bindings (planned) that expose essentially the same
API as the original.

## Status

**Phase 1 (this release): Gaussian-process core.** The `botorch_core` crate
implements BoTorch's `SingleTaskGP` for a single output, in double precision:

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
  derivatives with respect to the test inputs;
- the exact marginal log likelihood with analytic gradients, and
  `fit_gpytorch_mll` built on a port of L-BFGS-B 3.0 that reproduces scipy's
  iterates.

Numbers are validated against BoTorch itself: `scripts/generate_fixtures.py`
records reference values from BoTorch/GPyTorch, and the parity tests require
agreement to 1e-10 (observed: about 1e-14) for kernels, transforms, posteriors,
MLL values and gradients, and fitted losses no worse than BoTorch's.

Planned: acquisition functions and `optimize_acqf` (phase 2), PyO3 Python
bindings (phase 3).

## Usage

```rust
use botorch_core::{fit_gpytorch_mll, FitOptions, Matrix, Normalize, SingleTaskGP, SingleTaskGPOptions};

let train_x = Matrix::from_fn(12, 2, |i, j| ((i * 7 + j * 3) % 12) as f64 / 11.0);
let train_y = Matrix::from_fn(12, 1, |i, _| (6.0 * train_x[(i, 0)]).sin() + train_x[(i, 1)]);
let options = SingleTaskGPOptions { input_transform: Some(Normalize::new(2)), ..Default::default() };
let mut model = SingleTaskGP::new(train_x, train_y, options)?;
fit_gpytorch_mll(&mut model, FitOptions::default())?;
let posterior = model.posterior(&Matrix::from_row_slice(1, 2, &[0.3, 0.6]), false)?;
```

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test                                     # unit, parity, gradient and edge-case tests
cargo test --test parity -- --nocapture        # print the BoTorch parity tables
```

Regenerate the reference fixtures with BoTorch installed (see the header of
`scripts/generate_fixtures.py`):

```sh
python scripts/generate_fixtures.py
```

## License

MIT (see `LICENSE`). `botorch_core/src/lbfgsb.rs` is a port of L-BFGS-B 3.0
(Zhu, Byrd, Lu, Nocedal, Morales; C version by S. Becker), distributed under
the BSD 3-clause license whose notice is retained in that file.
