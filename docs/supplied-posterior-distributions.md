# Explicit Gaussian distribution interop

`GPyTorchPosterior(distribution=distribution)` and
`GaussianMixturePosterior(distribution=distribution)` accept caller-supplied
GPyTorch multivariate normals, including both multitask covariance layouts.
The positional distribution form also works. A Torch MultivariateNormal can be
supplied without GPyTorch's base-sample sampling protocol.

This is an explicit optional Torch interop boundary. It preserves the supplied
distribution object and its parameter graphs. Native models continue to construct
their posteriors and execute inference and query adjoints in Rust; they do not
construct these external distributions internally. Importing the package still
imports neither Torch nor GPyTorch.

The adapter implements marginal moments, device/dtype metadata, default and
explicit sample shapes, fixed-normal sampling, distribution forwarding,
no-grad `sample`, quantile/density, normal sampler dispatch, and affine
scalarization. Fixed normals and scalarization weights retain their Torch graphs
on this supplied-distribution path. Native scalarization weights retain their
previous held-fixed semantics. Supplied Gaussian mixtures also preserve graphs
through total mean/variance/covariance; their multitask covariance uses the
correct flattened event layout rather than the incompatible between-component
term in the pinned official reference.

Tests compare values, samples, and parameter/weight derivatives with official
BoTorch `b0dd01be6`. Serialization and deepcopy are checked for detached caller
distributions. Torch itself rejects deepcopy of nonleaf graph tensors, and the
adapter preserves that behavior.
