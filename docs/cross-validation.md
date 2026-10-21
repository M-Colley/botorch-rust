# Efficient leave-one-out cross-validation

`botorch_rs.cross_validation.loo_cv(model, observation_noise=True,
untransform=True)` dispatches to `efficient_loo_cv` for exact independent-output
GPs and `ensemble_loo_cv` for their ensembles. Unlike `batch_cross_validation`,
these functions retain the fitted hyperparameters rather than refitting each
fold. They put the supplied model in evaluation mode.

The native Rust method `SingleTaskGP::loo_predictions` reuses the training
Cholesky factor. For noisy training covariance `K`, targets `y` and prior
mean `c`, it computes

```
alpha = solve(K, y - c)
d[i] = sum_j inverse(L)[j, i]^2, where K = L L^T
mean[i] = y[i] - alpha[i] / d[i]
variance[i] = 1 / d[i]
```

This takes cubic time per output or model batch, rather than a separate
factorization for every held-out point. With `observation_noise=False`, it
subtracts each point's own noise variance, including heteroskedastic fixed
noise. The native diagonal posterior sampler uses the supplied base samples.

Ordinary results have shape `batch_shape x n x 1 x m`. Ensemble results
have shape `batch_shape x n x num_models x 1 x m`, with the mixture reporting
methods of `GaussianMixturePosterior`. Nonlinear outcome transforms retain
their sample transforms instead of being approximated by Gaussians. Ensemble
members must share transformed training targets and fixed observation noise.

The current implementation supports exact `SingleTaskGP`-backed models,
including their mixed and multi-fidelity variants and fully Bayesian or
MAP-SAAS ensembles. It rejects other model backends explicitly. It does not
differentiate these diagnostics through model hyperparameters. Specialized
outcome transforms retain their existing restrictions.

## Upstream failures and solutions

The parity tests reproduce these failures in released BoTorch 0.18.1 and
official upstream revision `b0dd01be61d517c387e234903a6e2cc38c057ea2`:

1. Batched `Standardize` statistics broadcast against the LOO fold dimension.
   For example, a model batch of size 2 and 6 observations produces a posterior
   shaped `2 x 6 x 1 x m`; statistics shaped `2 x 1 x m` do not broadcast
   against that layout. Untransform the predictions as `2 x 6 x m` and then
   restore the singleton `q=1` dimension. The Rust wrapper does this for both
   observed data and predictions; nonlinear samples use the same layout rule.
2. Fully Bayesian targets are stored as `n` observations, while prior means
   are batched over MCMC samples. The ensemble consistency checker indexes
   dimension `-2` of those unbatched targets and raises `IndexError`. Expand
   the targets to the predictive batch shape before checking consistency.
   Return observations expanded at the ensemble dimension as well, so they
   broadcast with the posterior. The native wrapper obtains observations
   directly in the model's predictive batch layout.

The local BoTorch fork at `c98dd9452` already contains proposed fixes for
these issues. Tests against the official release use the raw official LOO
Gaussian and explicit affine untransforms for the first failure, and expand
the unchanged reference targets for the second. They do not silently skip
failed reference cases.

The Rust fixture test `botorch_core/tests/loo_cv.rs` independently compares
the identity with explicit conditional Gaussian calculations using recorded
BoTorch kernel matrices. The Python tests cover lists, NumPy and torch;
batched and multi-output models; fixed noise; normalized inputs; standardized
and log-transformed outcomes; explicit base samples; and fully Bayesian
ensemble means, variances and mixture covariance.
