# Orthogonal additive GP components

`OrthogonalAdditiveGP.posterior(X, infer_all_components=True)` now conditions
each additive process on the **full** training covariance, using the native
cached Cholesky factor and representer weights. For component covariance
`K_c`, its posterior is

```
mean_c = prior_mean_c + K_c(X*, X) K_full(X, X)^-1 (y - prior_mean)
cov_c  = K_c(X*, X*) - K_c(X*, X) K_full(X, X)^-1 K_c(X, X*)
```

Only the bias carries the constant mean and an outcome transform's location.
The returned leading batch dimension enumerates bias, first-order terms,
then lexicographically ordered interactions. Training and query batches
broadcast behind it. `component_indices`, `get_component_index`,
`num_components`, and `evaluate_first_order_on_grid` expose this structure.
The kernel also provides native `k`, `normalizer`, and individual covariance
components; normalizers reflect current parameters without stale eval caches.

Component means sum to the full posterior mean. Component variances do **not**
sum to the full posterior variance: conditioning correlates the components,
and this interface returns their marginal processes rather than cross-component
covariances. Each marginal has genuine native query derivatives and supports
the Gaussian posterior sampling protocol. Explicit observation noise adds
the likelihood noise to each returned marginal, matching BoTorch's convention.

Components require a single output and an affine outcome transform. Nonlinear
outcome transforms do not preserve an additive decomposition and are rejected
for component inference. Full model posteriors retain their existing transform
support. The default batched kernel uses independent parameter vectors per
training batch, consistent with this port's current native batched GP fitting;
sharing a single learned parameter vector across training batches is a remaining
compatibility limitation. Standalone kernel inspection returns values without
building a hyperparameter autograd graph.

Validation includes 32 Python cases for values, covariances, samples, input
gradients, batching, quadrature, grid evaluation and snapshots. Native finite
differences independently check every component's mean and covariance Jacobian
for RBF, Matérn-3/2 and Matérn-5/2 kernels, including original-unit reconstruction.

## Verified official BoTorch defects

Reference: `b0dd01be61d517c387e234903a6e2cc38c057ea2`.

1. `_non_reduced_forward` builds the constant bias with the kernel's batch
   shape, while first-order terms include broadcast input batches. Concatenation
   fails with tensors of different ranks. Component GP inference also uses the
   training batch shape to move the component axis even when the test covariance
   has fewer or more batch dimensions. Expand both point sets to their common
   batch shape, and expand the bias before concatenation. The
   [proposed patch](../patches/botorch-oak-component-bias-broadcast.patch) applies
   cleanly to the reference and was tested in six combinations of interaction
   mode and query batch layout, against the native means and covariances.

2. With default `Standardize`, the common outcome location is added to every
   component. Their means therefore sum to `full_mean + (C - 1) * outcome_mean`.
   A deterministic seven-point, three-input example has an error of
   `48.03513772249775`, exactly the repeated location term. Apply the outcome
   location only to the bias; scale every component covariance by the outcome
   variance. The native implementation and regression test enforce this formula.

The official unbatched, untransformed cases are compared directly. Batched
cases use the conditioning equations and official kernel values independently,
because the unmodified reference crashes. The upstream repairs and regressions
are submitted in [BoTorch #3486](https://github.com/meta-pytorch/botorch/pull/3486).
