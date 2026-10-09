# botorch-rs

A Rust implementation of [BoTorch](https://botorch.org) (Bayesian optimization
with Gaussian processes), in the spirit of
[optuna/rustuna](https://github.com/optuna/rustuna): a dependency-light Rust
core crate, with Python bindings (planned) that expose essentially the same
API as the original.

## Status

Work in progress. Phase 1 implements the Gaussian-process core of BoTorch
(`SingleTaskGP`, its priors and transforms, the exact marginal log likelihood
and `fit_gpytorch_mll`) in the `botorch_core` crate, validated against BoTorch
itself. Acquisition functions and `optimize_acqf` (phase 2) and PyO3 Python
bindings (phase 3) will follow.

## License

MIT (see `LICENSE`).
