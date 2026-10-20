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

The native CPU numerical backend supports unbatched single-output exact GPs.
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
  indexing also needs consistent final-axis handling. The native CPU backend
  rejects that configuration; the Tensor backend below implements the final-axis
  likelihood law and rejects undefined batched discrete ranking.
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


## Prepared upstream repair and dependency follow-up

The official-source repair is prepared separately on branch
`codex/robust-noise-correctness` (base `b0dd01be6`). Its regression suite covers
all seven issues above on CPU and CUDA, including actual gradients through
batched rho gathers, sparse/dense/support edits along the final axis, frozen
parameters, failure during support search or model comparison, and successful
installation of the selected likelihood. Discrete pursuit still has a single
shared support; batched ranking now raises before mutation instead of inventing
a reduction across batches or using the wrong indexing axis. Batched likelihood
values and per-batch expansion objectives remain available.

The fixed-noise repair also exposed a separate `linear_operator` 0.6.1 defect.
`ZeroLinearOperator` records a requested dtype/device, but `to_dense` calls
`torch.zeros(*self.sizes)` without either keyword. This minimal regression
fails on CPU for dtype and on CUDA for both dtype and device:

```python
import torch
from linear_operator.operators import ZeroLinearOperator

for device in ["cpu", "cuda"]:
    zero = ZeroLinearOperator(2, 3, 3, dtype=torch.double,
                              device=torch.device(device))
    expected = torch.zeros(2, 3, 3, dtype=torch.double, device=device)
    torch.testing.assert_close(zero.to_dense(), expected)
```

The proposed dependency repair is to pass `dtype=self.dtype` and
`device=self.device` to `torch.zeros` in `ZeroLinearOperator.to_dense`, with
both device cases as regressions. The sparse-noise repair uses a
`DiagLinearOperator` of an explicitly allocated zero diagonal, preserving
shape, dtype and device even before that dependency repair is released.
Publication and deduplication are tracked centrally in
[upstream-pull-requests.md](upstream-pull-requests.md).

## Tensor backend: CPU, CUDA, and higher-order autograd

`RobustRelevancePursuitSingleTaskGP(..., backend="torch")` uses the Rust Tensor
factorization, prediction, and exact-MLL operations. CUDA training data or data
requiring gradients select this path automatically. A sparse likelihood supplied
to `SingleTaskGP(..., backend="torch")` selects the same exact observation law.
An actual Torch noise module supplied to `SparseOutlierNoise` or
`SparseOutlierGaussianLikelihood` creates a device-resident module directly;
registered base-noise and rho priors remain in the MLL and parameter traversal.
The base noise module's own forward is used, including fixed-noise event-size
behavior, and custom Torch covariance/mean modules retain their live parameters.

Fixed-support likelihoods and exact posteriors support independent output and
model batches. Sparse/dense conversion, expansion, and contraction act on the
last parameter axis, preserve trainability, and validate support changes before
replacing the rho leaf. The covariance contains the actual diagonal correction,
not an independent substitute model. Tensor mean, covariance, samples, normalized
MLL, transforms, training inputs/targets/fixed noise, and model parameters retain
first- and second-order graphs on their original device. Matching-training-input
observation noise applies convex rho to the **posterior** covariance diagonal,
as the public upstream likelihood does; training factorization instead uses the
prior covariance diagonal. Fixed-noise query values are recomputed from current
transform statistics when the event count matches.

Forward/backward discrete pursuit performs genuine joint hyperparameter/rho
fits, either through the existing bounded SciPy adapter or a supplied
`fit_gpytorch_mll_torch` optimizer. Tensor covariance calculations stay on device;
an explicitly selected SciPy optimizer exchanges its optimizer state with SciPy.
Support ranking uses Tensor scores; only the selected integer support labels are
converted to Python metadata. The corrected LOO criterion centers targets by the
actual GP mean. Gradient probes restore the original rho leaf, trainability, and
sparse/dense representation on success or failure. Support posterior evaluation
uses Tensor logsumexp and calls a custom support prior on scalar Tensor sizes,
matching official BoTorch. Fitted model-selection probabilities are detached
diagnostics; the separate support-posterior function retains its evaluation graph.
Support selection and optimization are not advertised as differentiable mappings
from training data to the chosen fitted parameters.

Discrete support ranking requires an unbatched single-output model and rejects
other layouts before changing support or parameters. Upstream has one shared
support and no specified aggregation of different batch ranking scores; the
batched likelihood law does not establish a batched discrete search law.
Variational and
samplewise sparse likelihood evaluation remain unsupported. Recreate an external
optimizer after support resizing because the rho leaf changes identity. The first
Tensor implementation recomputes training factors rather than caching them.

`test_torch_relevance_pursuit.py` checks both CPU and CUDA, float32 stability,
float64 values and mixed first/second derivatives against an independent dense
Schur formula, official unbatched prior/MLL/posterior parity, batched rho layouts,
observation-noise matching, fixed-noise/Normalize/Standardize graphs, actual
support-search fits and model selection, registered external Torch noise priors,
parameter bounds, and injected-failure rollback. The existing native relevance
suite remains unchanged and passes alongside it.

### Conditioning, fantasies, and timeout configuration

Tensor sparse models now append observation rows with **zero added outlier
variance**. Existing observation indices retain their rho values and their exact
persistent parameter leaves. This is an observation-label policy: appending a
second measurement at the same coordinates does not reuse the earlier row's
outlier label. Supplied conditioning noise remains active, and known observation
noise is extended with the existing Tensor GP convention. Original fitted input
and outcome-transform statistics remain fixed. Model, output, fantasy, and
noise-only leading batch axes broadcast without flattening through the CPU
backend. The original model's data and support metadata remain unchanged;
kernel, mean, learned base-noise, and existing rho leaves are shared, preserving
the complete conditional graph. Deepcopy a model first when independent fitted
parameter objects are desired.

If the original rho parameter is in dense representation, every old row is an
explicit parameter, including any currently zero values outside support metadata.
The conditioned model represents those old rows as a sparse support covering
the old observation range. This keeps the original dense leaf intact and adds
no detached parameter copies; appended rows still have zero rho. A later support
resize replaces the conditioned model's leaf as usual. Discrete ranking detects
fantasy/model batch axes even when existing parameter leaves broadcast over them.
Cached support-selection diagnostics are cleared on the conditioned model.

Default fantasies sample new observations with **base noise**, including the
per-output mean known noise for fixed-noise models. They exclude old observation
outlier labels even at matching coordinates, then condition with exactly that
sampling noise. Explicit predictive fantasy noise and `propagate_grads=True`
retain their graphs. This avoids sampling a noisy law and conditioning with a
different measurement variance.

The Tensor robust fitter honors the pinned official `timeout_sec` contract:
each numerical fit receives `timeout_sec / len(numbers_of_outliers)`. This is an
apportioned per-fit timeout, not a strict deadline for Python support scoring,
model comparison, or the whole search; the initial support fit can also be an
additional fit. Optimizer kwargs are copied before the timeout is inserted.
Finite nonnegative timeouts, optimizer step/iteration limits, and failure cleanup
are supported without background workers. The native CPU robust fitter retains
its separately documented timeout restriction.

`test_torch_relevance_conditioning.py` verifies the exact Gaussian conditional
mean/covariance and mixed Hessians, sparse and dense old layouts, shared leaves,
multi-output/model/noise/fantasy batches, fixed noise and transforms, base-noise
fantasy sampling, and timeout delivery to every actual numerical fit.
