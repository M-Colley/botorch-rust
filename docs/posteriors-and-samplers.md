# Posterior and sampler API details

Native Gaussian posteriors provide `sample`, `rsample`, base-sample sampling,
marginal `quantile` and `density`, and affine scalarization through
`scalarize_posterior_gpytorch` and `scalarize_posterior`. Scalarization keeps
model-backed posteriors lazy and uses Rust for covariance contraction and
sampling. It also handles explicit Gaussian moments (including LOO results)
and the single-point, single-output `PosteriorList` case supported by BoTorch.

`Posterior.sample` suppresses gradient tracking. Marginal density and quantile
evaluations support first derivatives with respect to the evaluation values.
Scalarization weights are passed to the native model as constants; gradients
with respect to those weights are currently unavailable, as with the existing
`ScalarizedPosteriorTransform`.

`MCSamplerMixin` lazily chooses and reuses a sampler through `get_sampler`.
It supports normal, transformed, list and ensemble posteriors. A `ListSampler`
samples its components and concatenates outputs, including lists containing
index samplers. Native `ModelListGP.fantasize` explicitly requires normal
samplers for its Gaussian members.

Native MC and multi-objective acquisition constructors currently initialize
their backend samplers eagerly. Their `get_posterior_samples` method uses
that configured sampler. In particular, calling it on a `PosteriorList` with
the constructor's default normal sampler remains unsupported; standalone
`MCSamplerMixin` selects a `ListSampler` correctly. Rebinding an acquisition's
Python sampler does not rebuild its compiled native acquisition.

`TorchPosterior` is an optional adapter for a PyTorch distribution explicitly
supplied by the caller. It delegates sampling, density, quantile and other
distribution attributes to that distribution. It imports PyTorch only when
constructed or used, and its default sample shape is empty, matching BoTorch.
Wrapping a PyTorch distribution does not add a PyTorch dependency to native GP
models or change their Rust computations.

The adapter finds device and dtype recursively through nested distribution
parameters. This fixes the metadata failure in the official BoTorch reference
at `b0dd01be6`: `TorchPosterior(Independent(Normal(...), 1)).device` and `.dtype`
raise `AttributeError` there because `Independent` holds its tensors inside
`base_dist`. Recursing through the stored parameters also handles
`TransformedDistribution`, dictionaries and lists; a visited set avoids cycles.
Pickle and deepcopy retain the wrapped distribution and metadata.
