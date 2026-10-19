# Expectation posterior transform and native GP base

`ExpectationPosteriorTransform(n_w, weights=None)` reduces consecutive
groups of `n_w` posterior points to weighted expectations. The outputs keep
their output dimension. Weights have shape `n_w x m` (one column may be
broadcast to all outputs), are nonnegative, and are normalized independently
per output. Normalization, weighted outcome evaluation, and full covariance
contractions execute through Rust array primitives. Cross-point and
cross-output covariance terms are preserved.

The transform can be called directly on a native `GPyTorchPosterior` or
passed to `SingleTaskGP.posterior`, including one-to-many input transforms
such as `InputPerturbation`. It returns a native Gaussian posterior with
means, marginal variances, full covariance, quantile/density evaluation,
explicit-base-sample sampling, and IID/Sobol normal samplers. Interleaved
multitask distributions are rejected, matching official BoTorch b0dd01be6.

`evaluate` supports lists, NumPy, and torch, including first-order torch
gradients in outcomes and weights. Posterior moment construction follows
the existing native GP API's value semantics; it does not expose gradients
with respect to posterior-transform weights or query inputs through torch
autograd. Compiled acquisition constructors currently support affine
output scalarization only. They explicitly reject expectation transforms;
this change does not claim acquisition gradient support for this transform.

Official BoTorch b0dd01be6 accepts all-zero weight columns and divides by
their zero sum, producing NaN weights, outcomes, and posterior moments.
This implementation rejects zero-sum and nonfinite columns with a useful
`ValueError`. An upstream fix is to validate finite nonnegative weights
and strictly positive column sums before normalization. A strictly positive
integer `n_w` and a point count divisible by `n_w` are also checked here.

`botorch_rs.models.gpytorch.GPyTorchModel` is the public base protocol for
the native exact, variational, higher-order, and multitask GP adapters. It
provides batch/output properties and tensor shape/dtype validation with
BoTorch-compatible strict errors, non-strict shape warnings, and precision
warnings. It keeps package imports free of torch. Numerical inference is
provided by native subclasses; the base does not wrap arbitrary Python
GPyTorch models or perform Python GP inference.
