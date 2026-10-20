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

## Structured constructors

HigherOrderGPPosterior and MultitaskGPPosterior accept their public supplied
distribution/covariance/target constructor arguments, including the positional
forms. Their supplied-operator path samples with Matheron's rule: draw from the
joint prior and training noise, solve the observed training covariance against
the target residual, apply the test/train covariance, and optionally add test
noise and the test mean. Operator roots and solves use the supplied Torch /
linear_operator objects, retaining their graphs. There is no BoTorch import or
automatic conversion of a native model to an external model.

HOGP uses the explicitly represented training-noise term when supplied as a sum,
and otherwise recovers noise by subtracting the joint training-prior block from
the observed training covariance. This supports a complete noise covariance
rather than assuming every training noise variance equals its first diagonal
entry. The caller must supply a valid positive-semidefinite joint/noise system.

LatentKroneckerGPPosterior accepts `(model, distribution, X, T)`. A native Rust
LatentKroneckerGP keeps its native Matheron sampler. An explicitly supplied
external model must expose its training-grid metadata and implement
`_rsample_from_base_samples(X, T, base_samples)`; that caller owns sampling.
The adapter validates base/sample layouts and preserves the returned graph.
Native structured Matheron query adjoints remain unsupported and are rejected
explicitly when gradient tracking is requested.

Normal samplers now recognize structured base layouts, including the additional
training-prior/noise draws. This fixes a preexisting path that passed only
`q * m` normals to a native MultitaskGPPosterior. Default sampler dispatch now
also handles native/supplied HigherOrder and LatentKronecker posteriors.

Structured parity tests use official `b0dd01be6`, covering batched and unbatched
HOGP/Multitask Matheron samples, diagonal and correlated task noise, optional test
noise, query/base-normal gradients, arbitrary sample shapes, and an explicit
Latent model protocol. For batched Multitask posterior-constructor parity, the
tests expand valid unbatched official operators: the official model's default
batched kernel with this environment's GPyTorch repeats its task batch twice and
fails before constructing a posterior. That separate model/kernel integration
failure does not prevent testing the public posterior constructor itself.
