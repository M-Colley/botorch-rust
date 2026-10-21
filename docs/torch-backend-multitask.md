# Device-resident MultiTaskGP

`MultiTaskGP(..., backend="torch")` uses the installed Torch device and full
autograd graph. CUDA training inputs, training tensors requiring gradients, or
custom Torch modules select this backend automatically. CPU data with no such
request keep the existing native CPU backend. For higher-order gradients with
respect to queries alone, select `backend="torch"` when constructing the model.

This implementation owns its task covariance, task means, and per-task noise
modules. Gaussian factorization, prediction, and exact log marginal likelihood
run through the Rust tensor bindings. It never delegates inference to a BoTorch
or GPyTorch model. Custom Torch kernels and Gaussian likelihood modules may be
used explicitly; their existing tensors and parameter graphs remain live.

The task covariance is `softplus(F) @ softplus(F).T + diag(softplus(v))`,
normalized by task zero's variance, matching the default PositiveIndexKernel at
the pinned official reference `b0dd01be61d517c387e234903a6e2cc38c057ea2`.
The default includes its Beta correlation prior, dimension-scaled RBF prior,
per-task constant means, and Hadamard noise priors in the normalized exact MLL.
Fixed training noise retains gradients through its original tensor and the
outcome transform. Known tasks with noncontiguous labels use mapped indices for
all covariance, mean, and noise lookups; this also preserves the existing CPU
port's correction to upstream's raw-label Hadamard noise lookup.

Training data have shape `n x (d+1)` and `n x 1`; batched training is rejected,
as in the pinned reference. Queries may have arbitrary batch dimensions.
Queries with task labels return one output per row. Queries containing only
continuous columns return the correlated outputs specified by `output_tasks`.
`output_indices` overrides those **raw task labels**, rather than selecting
positions within `output_tasks`. Covariances use output-major ordering.
Task features are categorical and have zero gradients. Continuous training and
query inputs, targets, noise, raw task factors/variances, kernel parameters,
and means support first and higher derivatives and ordinary Torch optimizers.

Task metadata may be inspected at construction. Tensor values used by the
numerical model are never flattened, detached, or copied through NumPy/CPU
computation. Input transforms must preserve task labels; use Normalize/Warp
indices that exclude the task feature. Port outcome transforms broadcast their
single long-format statistics across selected tasks, including nonlinear chains.

`all_tasks` may include unobserved tasks. Evaluation initializes their positive
factor rows from the observed-task average, following upstream's explicit
parameter mutation. No mathematical gradient tie to the observed rows is
introduced by that initialization. Fixed predictive noise for an unobserved
task uses the global observed mean when no task-specific average exists,
following the corrected native/empirical rule proposed in
[BoTorch #3389](https://github.com/meta-pytorch/botorch/pull/3389).

Conditioning appends observations in the original long-format input layout,
including the task column, with a single target column. Observation and model
batches broadcast; targets and supplied conditioning noise retain their graphs.
Noise follows the reference conditioning convention: it is already in the
outcome-transformed space. Fixed-noise conditioning requires that noise.
For a learned-noise model, explicitly supplied noise enters both the posterior
mean and covariance, retaining the native port's repair for upstream #3396.
The original model's training data and fixed-noise inspection buffers stay intact.

## Direct calls and caller-defined priors

`forward(X)` always returns the latent prior. In training mode it applies the
input transform; in evaluation mode it accepts inputs already transformed by
the caller. An evaluation-mode `model(X)` conditions on the training data and
returns the Gaussian in fitted, outcome-transformed target units. It accepts
the complete long-format input, including its physical task column. It neither
applies the input transform again nor untransforms the outcome. Public
`posterior(raw_X)` separately applies the input transform, selects/expands
requested tasks, and untransforms the resulting outcome. These distinctions
match the pinned source's ExactGP protocol.

Caller subclasses of MultiTaskGP and LCEMGP retain their actual `forward`
method. Prediction evaluates that prior jointly on transformed training and
query inputs to obtain its cross-covariance; the MLL also evaluates the actual
prior. Cooperative `super().forward`, added parameters, serialization and full
data/query/parameter Hessian-vector products remain supported. Built-in priors
retain their blockwise kernel evaluation.

Explicit Gaussian likelihoods retain their actual marginal and object identity,
including input-dependent, correlated noise covariance and mean changes. A
canonical supplied Gaussian need not expose a public `.noise` attribute. It
owns the observation law even if constructor `train_Yvar` was also supplied,
as in the reference. Task indices passed to likelihoods are mapped indices,
preserving the existing #3396 raw-label repair. Fixed likelihood conditioning
appends the actual fixed component while preserving separate learned noise
parameters and the graphs of original/new noise tensors. Explicit `noise=`
overrides follow the supplied likelihood's own marginal implementation.

Thirty-two new CPU/CUDA cases compare direct prior/conditional values and first
derivatives against official BoTorch
`b0dd01be61d517c387e234903a6e2cc38c057ea2`, and independently check complete
Gaussian conditioning/MLL values and all data/noise/parameter mixed Hessians.
They cover Normalize/Standardize, interior task columns, noncontiguous labels,
query batches, actual correlated likelihoods, subclass serialization and strict
host-conversion guards. The direct source comparisons use contiguous inferred-
noise labels to isolate the already reported #3396 defect, and omit fixed
LCEM embeddings to isolate the source's already reported ARD-width defect;
independent Gaussian controls retain those configurations. All 303 related
MT/LCEM/fully Bayesian/subclass/exact-likelihood regressions pass, excluding
two separately tested NUTS replay cases. Torch 2.14.1 CPU passes 16 new cases
and skips 16 CUDA cases. Ruff and Mypy pass.

Fantasies sample the correlated observed posterior and append each sample set
as a model batch. Multiple sample axes and `propagate_grads=True` preserve
mixed higher derivatives through fantasy inputs, sampled targets and all model
parameters. The fixed-noise extension uses actual per-task predictive noise
averages, rather than a global average. Batched fixed-noise prediction gathers
those averages along the task axis after broadcasting model and query batches.

Boundaries are explicit: Tensor-valued posterior/fantasy observation noise and
unsupported custom module/prior configurations raise errors. Conditioning and
fantasies require inputs including the task column, as in the reference.
This checkpoint recomputes the training factor on each evaluation;
it does not claim the single-task backend's prediction-cache speedup.
With `validate_task_values=False`, the raw task mapper retains NaNs for absent
labels like upstream. Numerical evaluation rejects invalid mapped indices before
a CUDA gather can trigger a device-side assertion. With validation enabled,
unexpected labels raise ValueError directly.

The focused suite compares moments, full joint covariance, normalized MLL, and
parameter gradients with the pinned reference on CPU/CUDA and float32/float64.
It also checks finite-difference first/second gradients through training inputs,
targets, fixed noise, queries, and every raw parameter; correlated sample
pullbacks; nonlinear outcome chains; task remapping; custom modules; dtype/device
moves; live optimizer updates; and explicit unsupported cases.

`test_torch_multitask_conditioning.py` adds CPU/CUDA checks of broadcast
conditioning, both task-column positions, noncontiguous labels, shared-normal
fantasies with multiple sample axes, fixed predictive-noise averaging, every
source/new-data/noise/parameter first derivative and mixed Hessian-vector
products. Device tests forbid CPU/list/NumPy conversion during conditioning,
sampling and double backward.

The pinned official fixed-noise fantasy update has an additional broadcast
failure: unbatched `new_X` with three batches of `new_Y`/noise reaches
`linear_operator.operators.LinearOperator.cat_rows` with covariance blocks of
different ranks and raises `RuntimeError: All tensors must have the same number
of dimensions`. The regression asserts that unmodified failure, then compares
the port's three conditional laws against three separate official updates.
The existing [BoTorch PR #3448](https://github.com/meta-pytorch/botorch/pull/3448)
already addresses this failure: broadcast the new inputs to the complete noise
batch before invoking `get_fantasy_model`. This makes the old/cross/new
covariance blocks compatible in the fantasy update. The port forms the exact
augmented covariance directly and therefore retains the same correlated law
without depending on that update shortcut. This is an existing upstream issue,
not a newly distinct bug report.
