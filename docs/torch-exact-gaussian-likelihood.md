# Supplied Gaussian likelihoods in exact Tensor GPs

`SingleTaskGP(..., backend="torch", likelihood=module)` retains supported
Gaussian likelihood modules and their Parameter identities. Actual GPyTorch
`_GaussianLikelihoodBase` modules receive a lazy Gaussian prior and the source
input protocol; their actual marginal defines the observed mean and full
covariance. No public `.noise` vector is substituted for a correlated
`noise_covar`. Returned Gaussian means and covariance Tensors must retain the
prior event size and broadcastable batches. Non-Gaussian exact likelihoods
remain unsupported, as in GPyTorch ExactGP.

Conditioning and Gaussian solves use the port's native Tensor operations.
GPyTorch distributions/operators are optional interoperability for an explicitly
supplied likelihood, and no upstream BoTorch model/inference is called.
Scalar, batched and independent multiple-output GPs retain CPU/CUDA data,
query, noise and parameter graphs through mixed higher derivatives.

A supplied likelihood owns observation noise even when the constructor also
receives `train_Yvar`, matching BoTorch. Its own noise is already in transformed
outcome units; it is not silently rescaled from constructor noise. The default
likelihood and constructor-fixed-noise paths retain their existing native
blockwise calculations and caching. Custom modules conservatively recompute
their training marginal, as they can depend on state outside registered
parameter mutation counters.

Exact marginal likelihood calls accept explicit likelihood arguments, such as
`mll(prior, targets, model.train_inputs)`, for input-dependent noise modules.
Calls without arguments infer the transformed training-input sequence. Actual
likelihood kwargs are forwarded to its Gaussian marginal. The latent prior
continues to honor a public model's virtual `forward` implementation.

Supplied fixed-noise conditioning appends the likelihood's actual fixed values
and requires explicit new noise, following its fantasy protocol. The original
likelihood remains unchanged, while the conditioned model retains shared learned
parameters, separate additional learned noise and live appended-noise graphs.
Fantasies infer new fixed noise from the likelihood's mean noise, matching
BoTorch, and support differentiable posterior samples.

The actual supplied marginal remains authoritative. For example, GPyTorch
1.15.2 forwards an explicit `noise` kwarg to both the fixed and additional
learned noise components of `FixedNoiseGaussianLikelihood`. Both components
then use that override, so this particular module doubles the provided noise.
The adapter preserves that observed module behavior rather than silently
rewriting caller-owned likelihood semantics. A separate upstream regression
and repair are being prepared; this does not affect the port's default
fixed-noise likelihood.

Validation adds 44 CPU/CUDA cases. Independent dense Gaussian formulas check
full correlated noise, scalar/multiple-output and training/query batches,
explicit prediction-noise overrides, complete first derivatives and mixed
HVPs. Pinned BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2`
provides direct posterior/MLL values and independent first-gradient controls.
Fixed-noise updates preserve original module state and learned parameter
identity. Device guards reject Tensor `cpu`, `numpy` and `tolist` conversions
during prediction, MLL and differentiation. All 494 focused new/existing
Tensor core, module, cache, subclass and fully Bayesian loader cases pass.
Torch 2.14 CPU passes 22 new cases and skips 22 CUDA cases. Ruff and Mypy pass
for the changed production modules.
