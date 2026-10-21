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

HigherOrderGP, LatentKroneckerGP, and KroneckerMultiTaskGP now have dedicated
native Matheron query adjoints as described below. Gradients through batches of
HigherOrderGP fantasy posteriors are still rejected explicitly.

This change does not add hyperparameter autograd, second-order query derivatives,
gradients with respect to scalarization weights, or query connectivity through
explicit-moment posterior transformations such as ExpectationPosteriorTransform.
Mixture aggregate statistics/quantiles retain their existing value-only path;
the component means, variances, covariance matrices, and samples have query
connectivity. More general dynamic model families require their own numerical
parity coverage before claiming all configurations are covered.

## Structured native posteriors

HigherOrderGP and KroneckerMultiTaskGP differentiate their actual fixed-base
Matheron maps. With training points first in the data Cholesky factor, its
training rows do not depend on query coordinates. Native pullbacks therefore
differentiate the test rows of that factor and the test/training covariance
update, retaining the supplied normals and model parameters. Tensor mode
products, task-noise covariance, standardization, and input transforms retain
their existing sampling semantics.

LatentKroneckerGP differentiates its complete- or missing-grid sampler with
respect to both feature queries and custom time queries. Its map uses
`Aᵀ z + chol(K_test - Aᵀ A) w + K_observed,testᵀ solve(residual)`; the native
pullback differentiates the residual Cholesky factor and both Kronecker kernel
factors, then sums broadcast adjoints back to the original X and T shapes.

The public marginal means, variances, and exposed covariance views also retain
query graphs. HOGP's exposed base distribution has diagonal marginal covariance
blocks, matching its official distribution; its default standardized posterior
is a TransformedPosterior. Gaussian sampling and marginal evaluation remain
native. Affine outcome-transform helpers now preserve the incoming query graph
instead of flattening it away. `sample`, no-grad evaluation, float32/float64,
normal samplers, and model snapshots retain the ordinary posterior behavior.

Three Rust finite-difference tests cover tensor targets, correlated noise ranks,
input/output normalization, missing grids, and feature/time adjoints. Fifty
focused Python cases cover raw gradients and batch broadcasting, fixed normals,
custom times, missing observations, transforms, snapshots, dtype/no-grad, and
live official HOGP value/sample/gradient parity. Native Kronecker samples use
their established Cholesky roots, which can differ from official eigen-based
roots for the same normals; their native gradients are checked against the
actual native sampling function.

The moment pullbacks reuse exact native model VJPs. Some HOGP/full-grid moment
paths still form dense intermediates; this milestone does not claim that every
large structured configuration has an optimized backward memory footprint.

## Fixed Kronecker posterior snapshot bug

The specialized MultitaskGPPosterior previously snapshotted its generic Gaussian
model handle but retained a mutable Kronecker sampling source. After changing
`model.likelihood.noise` to 100, an old posterior kept its cached mean
`[0.2020, 0.6863]` while its zero-base samples changed to `[0.0113, 0.00735]`.
The constructor now retains `model_rs.copy()`, so cached moments and Matheron
samples use the same immutable parameter state. Float32 and float64 regression
tests check both sample values and query gradients after a model mutation.

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
