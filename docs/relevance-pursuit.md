# Native relevance pursuit and robust Gaussian noise

The port implements the nine public symbols from official BoTorch revision
`b0dd01be6` in `models.relevance_pursuit`,
`models.robust_relevance_pursuit_model`, and
`models.likelihoods.sparse_outlier_noise`. These are working model components,
support searches, and fitters; the Rust extension never imports BoTorch.

`SparseOutlierGaussianLikelihood` can be supplied to `SingleTaskGP` with a
native Gaussian base-noise module. `RobustRelevancePursuitSingleTaskGP` also
supports fixed observation noise through `train_Yvar`. Support expansion,
contraction, sparse/dense conversion, snapshots, forward/backward pursuit,
and maximum-posterior support selection operate on the actual model state.
The ordinary native acquisition-model handle and posterior methods retain
the outlier covariance. Snapshots use copy-on-write model state.

For a training covariance diagonal `d = diag(K + Sigma)`, the convex
parameterization adds `d_i * r_i / (1-r_i)` at active observations. The direct
parameterization adds `r_i`. A single native Cholesky factorization evaluates
the exact MLL, all dense hyperparameter gradients, and rho gradients. The
convex chain rule also scales the diagonal adjoints for base noise and kernel
parameters. The native bounded L-BFGS-B optimizer fits rho jointly with the
mean, kernel, noise, and learned input-transform parameters. Priors on rho
apply to the dense variance factors, including inactive zeros, as upstream.
Dense parameter representation exposes all `dim` parameters; fitting through
the pursuit routines switches back to the active representation.

Observation noise follows the sparse likelihood's semantics: rho applies
only at the complete matching training inputs, convex rho then scales the
posterior diagonal, and fixed base noise outside its event size contributes
zero. Rho constraints must be non-transformed, nonnegative, and strictly
below one in the convex parameterization. The lower bound is zero because
inactive variance factors are zero. A supplied constraint initial value is
honored. Public setters validate exact bounds; native line-search evaluations
project excursions of a few ulps back to box endpoints.

The implementation currently supports unbatched single-output exact GPs.
Batched sparse parameters, variational inference, samplewise likelihood
evaluation, changing training inputs, sparse support remapping for fantasy
models, and time limits on an entire pursuit explicitly raise unsupported
configuration errors. Native optimizer iteration and evaluation limits work.
Support posterior probabilities intentionally retain upstream's use of the
normalized MLL rather than silently changing its model-comparison formula.

The atomic transformed-target/fixed-noise replacement method preserves
fitted transforms and hyperparameters, updates original-space bookkeeping,
and invalidates training factors. It supports batched and multi-output
ordinary GPs as well as the single-output sparse model. Target/noise edits
commit together after all elements validate. The number of observations
must continue to match the existing training inputs.

## Confirmed upstream defects and corresponding corrections

These defects were reproduced against the official checkout above, separate
from the older local fork and the installed BoTorch 0.18.1 baseline.

* `SparseOutlierNoise._optimal_rhos` solves against `Y` while its covariance
  has a possibly nonzero mean. Its LOO expansion criterion is consequently
  incorrect after fitting the constant mean. Solve against `Y - F.mean`
  instead; equivalently use centered residuals in the precision/LOO
  identities. The Rust implementation uses this correction, with a regression
  against the corrected reference algebra and a nonzero-mean counterexample
  to the uncorrected implementation.
* Batched rho selection writes `_rho_selection_indices[j] = i` along the
  first axis. For batch size two, `dim=5`, and support `[2]`, reading `.rho`
  raises `IndexError`. Selection indices must be assigned with `[..., j]`
  and gathered along the final dimension; the generic pursuit mixin's sparse
  indexing also needs consistent final-axis handling. This port rejects that
  configuration until the full batched semantics are implemented.
* `expand_support([1, 1])` accepts duplicate indices, creating invalid support
  state. Validate uniqueness, integer range, and disjointness with the existing
  support before mutation. The port performs these checks atomically.
* The robust model's default `int(0.2 * dim)` is zero for fewer than five
  observations. Support posterior calculation then raises `ValueError: math
  domain error` at `log(prior_mean_of_support)`. This port uses
  `max(1, int(0.2 * dim))` and validates explicitly supplied prior means.
* A failure during `custom_fit` leaves `mll.model` pointing at the temporary
  standard model. Restore the original model and likelihood in `finally`;
  only install a selected fitted snapshot after the search succeeds. An
  injected optimizer failure reproduces the upstream state leak and tests
  that the port preserves the original state.
* The noise module caches its input tensor by reference. Mutating that tensor
  changes both compared inputs, incorrectly retaining rho during evaluation.
  Cache an independent copy. The port caches independent flattened values,
  with a regression that mutates the caller's tensor after training.
* With fixed base noise and a test event size different from the training
  size, `FixedGaussianNoise` returns a scalar `ZeroLinearOperator`.
  `SparseOutlierNoise.forward` indexes its empty shape with `[-1]` and crashes
  during `posterior(..., observation_noise=True)`. Preserve the requested
  event shape on this zero operator (or use the requested event size for
  the shape check), skip rho, and return zero added noise. The regression
  proves the official exception and compares the port against the actual
  latent reference posterior, which is the correct zero-noise result.

Validation includes Rust finite differences for every joint parameter in
both parameterizations, bounded optimization and cache invalidation, and
support-validation rollback. Python reference tests compare exact MLL values,
all hyperparameter gradients, candidate posteriors, fixed-support optimizer
iterates, standalone likelihoods, support-search snapshots, robust fixed-noise
MAP selection, failed-fit restoration, and atomic target/noise replacement.
BoTorch default prior factories are authored in their canonical float32 mode
before constructing double-precision models in the parity tests.

## Persistent parameter layout during pursuit

Changing sparse support changes the native raw-parameter layout. Torch
synchronization refreshes the layout before pulling values, retaining
unchanged kernel, mean and noise leaves and replacing a resized `raw_rho`
leaf with its new shape. Its trainability flag is preserved. Public sparse
updates first push pending Torch edits, so support changes cannot discard
those edits or write a stale parameter vector into Rust. Recreate an
external optimizer after changing support because its parameter list and
state may refer to the previous sparse layout. Float32/float64 regressions
check expansion, contraction, pending edits and leaf identities; the full
forward/backward pursuit and robust-model fitting tests cover repeated
layout changes.

