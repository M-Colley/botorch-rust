# Native posterior query autograd

The native Gaussian posterior now connects Torch query tensors to first-order
input derivatives of its mean, variance, covariance matrix, and reparameterized
samples. Gaussian inference, Cholesky sample adjoints, outcome-transform
derivatives, output selection, and scalarization derivatives execute in Rust.
The optional Torch adapter only passes values and adjoints between the graph and
the native model. Importing `botorch_rs` does not import Torch.

The adapter retains a clone of the original query graph separately from the flat
native query buffer. It uses the posterior's immutable model snapshot, so later
model parameter changes do not change either its values or its input derivatives.
Model/query batch broadcasting sums adjoints back into the original query shape.
Float32 and float64 outputs retain the query's dtype and device; native numerical
work remains on the CPU.

`rsample`, explicit base-sample draws, and the IID/Sobol normal samplers preserve
query connectivity. Random normals and model hyperparameters are held fixed in
this derivative. `sample` and `torch.no_grad()` disable the query graph. The
adapter deliberately supports first-order derivatives only.

## Validation

The focused suite compares exact GP values and input gradients with the official
BoTorch checkout at `b0dd01be6`, including single/multiple outputs, batched models,
query broadcasting, observation noise, Standardize and Log outcomes, Normalize
and Warp inputs, selected/reordered outputs, affine scalarization, covariance
layout, and fixed-normal samples. ModelList, MultiTaskGP, and the component
posteriors of fully Bayesian single-task and multitask models also have native
finite-difference checks. Existing posterior, sampling, snapshot, protocol, and
fully Bayesian tests are rerun separately.

## Current boundaries

HigherOrderGP, LatentKroneckerGP, and KroneckerMultiTaskGP use specialized structured/Matheron sampling
paths. Their public posterior methods reject query tensors requiring gradients
when autograd is enabled until those paths have dedicated native adjoints. Their
ordinary values and native acquisition input gradients remain available.

This change does not add hyperparameter autograd, second-order query derivatives,
gradients with respect to scalarization weights, or query connectivity through
explicit-moment posterior transformations such as ExpectationPosteriorTransform.
Mixture aggregate statistics/quantiles retain their existing value-only path;
the component means, variances, covariance matrices, and samples have query
connectivity. More general dynamic model families require their own numerical
parity coverage before claiming all configurations are covered.

## Verified official reference bug and solution

At official revision `b0dd01be6`, a two-output batched SingleTaskGP queried with
`output_indices=[1]` raises `ValueError: Must provide at least 2 MVNs to form a
MultitaskMultivariateNormal`. The inherited posterior method passes a one-element
list to `MultitaskMultivariateNormal.from_independent_mvns` in
`botorch/models/gpytorch.py`. The Rust port returns the selected single-output
Gaussian law. Tests explicitly reproduce the reference exception and compare
with `reference.subset_output([1]).posterior(X)`, which has the intended law.

The upstream fix is to return the sole multivariate normal directly when the
selected list has length one, and only construct a multitask distribution when
there are at least two selected outputs. This finding concerns the pinned
official revision, not the separate locally corrected BoTorch fork.
