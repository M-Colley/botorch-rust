# Empirical learning-curve models

Reference: official BoTorch commit `b0dd01be6` in `.reference/botorch`.
The separate checkout on the Desktop is an audit fork and is not the authority
for this parity work.

The first model milestone implements these public classes:

- `BaseAugmentedEmpiricalKernel`
- `EmpiricalOneDimensionalMean`, `EmpiricalOneDimensionalKernel`,
  `EmpiricalOneDimensionalGP`
- `MultiOutputEmpiricalOneDimensionalMean`,
  `MultiOutputEmpiricalOneDimensionalKernel`,
  `MultiOutputEmpiricalOneDimensionalGP`

On the native CPU backend, interpolation, basis decomposition, empirical covariance, Cholesky inference,
MLL evaluation, parameter derivatives, input derivatives, sampling and both
L-BFGS-B and Adam fitting run in Rust. Python arranges array shapes and module
metadata. Torch and NumPy are optional array containers; neither implements
the model's numerical inference. Native computation currently uses CPU float64;
Torch output arrays preserve the caller's dtype and device.

Historical curves have shape `curves x progression x outputs`, with at least
two distinct progression coordinates. Means interpolate the historical sample
mean. Covariances interpolate centered curves and normalize by the **original**
curve count minus the correction, including when a compressed basis is used.
Independent-output models share ARD weights across outputs and learn separate
noise values. Their native fitter sums the per-output normalized likelihoods
and accumulates derivatives of shared parameters. Correlated-output models
condition jointly on point-major interleaved observations and expose the full
cross-output covariance. Both support fixed observation variances and an
additive native base covariance on progression.

Validation includes official-source mean/kernel parity with and without
compression and ARD, unequal cross-input diagonals, query derivatives, MLL
values and raw gradients, independent and correlated posterior moments and
joint covariances, fixed/inferred noise, list/NumPy/Torch arrays, both native
fitters, immutable posterior snapshots and additive-kernel parameter binding.
Native tests independently finite-difference all shared raw parameters and the
entire posterior mean/covariance Jacobian.

## EM prior pretraining

`pretrain_em_prior` and `EMPriorContainer` perform exact native EM over
independent historical datasets. They support partial observations on the
union input set, explicit inducing points with shift interpolation, Gaussian
mean and inverse-Wishart covariance priors, kernel or identity initialization,
trace-matched covariance shrinkage, PSD stabilization, convergence stopping,
and optional iteration histories. All matrix operations run in Rust.
Historical data and inducing locations are fixed; gradients through EM
pretraining are rejected explicitly. The mean must be an unbatched native
`ZeroMean` or `ConstantMean`, and the covariance an unbatched native kernel.

Official-source tests compare every EM iteration and the resulting moments,
inducing Cholesky factors and shifts for both observation layouts, ML/MAP
updates, both initializations, and shrinkage. Saved containers preserve array
metadata and native mean, kernel, constraint and prior descriptors without
serializing mutable model backreferences. This save format is native to the
port and does not interchange files with `torch.save` upstream.

## EM prediction and observed-data fitting

`EMEmpiricalGaussianProcess` conditions a learned inducing prior on new target
observations. Shift interpolation preserves the full Nyström residual and an
optional additive base kernel contributes its full covariance at query points.
When interpolation is disabled, queries must identify historical locations
within the upstream `1e-6` distance tolerance. Explicit inducing points always
enable interpolation. Queries may have batch dimensions; training is scalar
and unbatched. Both prior distributions (`forward`) and conditioned posteriors
have native first query derivatives, full joint covariances and sampling.
Posterior objects retain immutable snapshots when model parameters change.

`EMEmpiricalMarginalLogLikelihood` sums historical observed-data likelihoods
and adds registered hyperpriors once, then divides by the total historical
observation count. For `from_pretrained` models it differentiates through the
parametric mean, interpolation kernel, Gaussian noise and optional additive
kernel while holding learned inducing moments fixed. The default factory
freezes the pretrained mean/kernel; `freeze_pretrained=False` enables their
joint fitting. Native L-BFGS-B and Adam use these analytic gradients.

Constructing from historical datasets performs native initial EM. Training
reruns EM from the parametric kernel prior and differentiates through every
conditioning and moment update in Rust. The derivatives include Gaussian
mean priors, inverse-Wishart covariance priors, trace-matched shrinkage and
spectral PSD clipping. Both native fitters use the resulting historical MLL
gradients and refresh the learned moments and target posterior after parameter
updates. Default construction and fitting match official BoTorch values,
raw parameter gradients and posterior moments; independent finite differences
exercise indexed and explicit-inducing layouts with ML and MAP updates.
Warm-start EM initializes each rerun from the previous detached learned
moments, while evaluating mean/IW priors and shrinkage against the current
parametric kernel. Each observed-data likelihood call advances the state
once and differentiates through that run; previous runs contribute no gradient.
The native L-BFGS-B and Adam closures retain this state between evaluations.
Parameter assignment preserves warm moments until a training forward or MLL
call refines them. Explicit-parameter `value_and_grad(raw_parameters)` evaluates
against the current seed without changing model state, allowing reproducible
finite-difference diagnostics. Existing posterior snapshots remain immutable.
With explicit `inducing_points`, `learnable_inducing_points=True` exposes
`_X_inducing` as a native optimizer parameter. Analytic kernel input adjoints
differentiate the inducing Gram matrix, interpolation weights, Nyström residual
and complete cold or warm EM updates. Both fitters can optimize coordinates
alone or jointly with the mean, covariance and observation noise; callers may
set locations using `model.initialize(_X_inducing=...)`. Pretrained models can
also learn locations while their inducing moments stay fixed. Moving the
historical input grid without explicit inducing points remains unsupported.
Arbitrary Python mean/likelihood objects and tensor-valued predictive noise
are rejected.

`build_shared_gp_model_list` returns exact scalar `SingleTaskGP` components
sharing the caller's actual native mean/kernel descriptors. Its returned
`SumMarginalLogLikelihood` aggregates the per-dataset normalized likelihoods
and gradients of shared parameters in one native optimization. Gaussian
noises are learned independently or pinned exactly when supplied. List,
NumPy and Torch array interfaces and native L-BFGS-B/Adam are supported;
custom Python modules and batched historical datasets are explicit boundaries.
Custom optimizers and closures, parameter subsets, callbacks, schedulers and
retry policies now use the generic fitting drivers over that same joint
objective and global parameter layout. They preserve shared parameters and
fixed noises. Native EM retry snapshots include the learned warm moments, not
only raw hyperparameters; see [fitting protocols](fitting-protocol.md).

`coordinate_ascent_em` alternates native EM with native fixed-moment kernel
fitting, pins observation noise, updates caller-owned parametric modules, and
reruns EM on the complete historical data. It supports L-BFGS-B or Adam,
historical-dataset subsampling and immutable container histories. Inducing
locations are currently fixed. Recorded histories snapshot descriptors and
moments so later kernel mutations do not change earlier trajectory states.

Two native derivative regressions cover eigenvalues immediately on opposite
sides of the PSD floor and the exact trace-clamp boundary. Spectral divided
differences now follow both clipping branches symmetrically, including when
the eigenvalue gap is smaller than `1e-12`; a positive-semidefinite matrix at
the floor uses Torch's inclusive clamp derivative. Trace-shrinkage derivatives
also use the inclusive boundary convention. These correct port bugs rather
than changing the reference's behavior.

Float32 EM containers can round a rank-deficient covariance to a slightly
indefinite matrix. The constructor repairs only negative eigenvalues within
the input dtype's storage precision using native projection; it continues to
reject materially indefinite covariances. Regression tests exercise the
rank-deficient Float32 case and rejection of a negative identity matrix.

## Verified upstream EM issues

### Freezing computed EM state

Official revision `b0dd01be6` also raises a leaf-variable `RuntimeError` when
`freeze_pretrained_parameters()` follows an observed-data MLL evaluation with
trainable mean, kernel or inducing locations. The MLL refresh creates computed
mean/covariance and interpolation-cache tensors. Calling
`requires_grad_(False)` on those non-leaf tensors fails. The regression
`test_freezing_after_mll_evaluation_handles_computed_em_caches` reproduces this
with a learned inducing set; the native model freezes its parametric modules
successfully and keeps inducing coordinates available for fitting.

An upstream fix should freeze the actual module parameters, replace learned
moment tensors with their detached values, and rebuild the interpolation cache
from the current frozen modules. Rebuilding retains the correct coordinate
derivatives when `_X_inducing` is still learnable. Cached factors are computed
values rather than optimizer parameters and should not have their gradient
flags mutated in place. No reference source was modified.

### Historical dataset subsampling

In official revision `b0dd01be6`, `coordinate_ascent_em` replaces
`model.datasets` with a random subset but leaves
`model._unique_inputs_obs.experiment_indices` in the original dataset order.
For a prior fitted without explicit inducing points, the MLL loop then pairs
the selected outcomes with another historical dataset's mean/covariance
indices. Equal-sized datasets silently produce the wrong likelihood;
different sizes can fail with a dimension error.

For example, fit a prior to datasets observed at `[0, 1]` and `[1, 2]`, then
assign `model.datasets = [datasets[1]]`. The MLL evaluates the second dataset
at indices `[0, 1]`, while the correct indices in the original union input
set are `[1, 2]`. The regression
`test_subsampled_historical_likelihood_uses_correct_indices` reproduces a
nonzero likelihood error against an explicit Gaussian calculation.

A proposed upstream fix is to look up each selected dataset against the
unchanged full historical input set inside the non-inducing MLL path:

```python
indices = self.model._find_indices_in_historical(dataset.X)
mu_S, Sigma_SS = self.model._get_prior_at_indices(indices)
```

Alternatively retain the selected original dataset indices and use those to
select the original `experiment_indices` entries. Rebuilding the union input
set from the subset would lose the correspondence with the learned inducing
moments. The port derives indices from the actual dataset inputs and refreshes
the native likelihood state when `model.datasets` is replaced. The reference
source was not modified.

The default non-inducing coordinate step also has no data dependence on the
parametric kernel once EM moments are detached. With a `ZeroMean`, an RBF
kernel without hyperpriors and fixed noise, upstream calls backward on a
constant likelihood and raises `RuntimeError: element 0 of tensors does not
require grad`. The port returns the mathematically correct zero kernel
gradient and completes that coordinate step without changing the kernel.
`test_coordinate_default_without_phi_dependence_is_a_valid_noop` reproduces
both behaviors. An upstream solution is to skip optimization when the MLL has
no trainable parameter dependence, or to request explicit inducing points
when observed-data kernel refinement is desired.

## Current boundaries

Training data must be unbatched, although posterior queries may have batch
dimensions. Historical data and their decomposition are fixed. The models
require corresponding empirical mean/covariance modules; arbitrary custom
means, input/outcome transforms, batched additive kernels and explicit
tensor-valued posterior noise are rejected. Additive base-kernel parameters
are shared across independent outputs. Model calls use the port's existing
`mll()` interface, rather than accepting a Python GPyTorch distribution as an
MLL argument. Independent and correlated empirical one-dimensional GPs now
support native conditioning and fantasy batches
([scope and upstream workaround](empirical-conditioning.md)); heterogeneous
multitask empirical conditioning remains unsupported.
Multi-output model component descriptors directly expose their empirical
kernel/mean, rather than the upstream internal vectorization wrappers.

The heterogeneous-domain multi-task milestone adds
`MultiTaskEmpiricalOneDimensionalMean`,
`MultiTaskEmpiricalOneDimensionalKernel` and
`MultiTaskEmpiricalOneDimensionalGP`. Historical task grids may have different
lengths and domains; curve identities are paired across tasks. Models accept
arbitrary observed task subsets and predict selected tasks, including tasks
absent from training. Queries may specify their own task per row. Both task
column positions, task-dependent fixed noise, prebuilt fixed-module reuse,
wide-format and homogeneous historical-data constructors are supported.
Task selection occurs before interpolation, so an unrequested task's domain
does not constrain a selected task's query range. Analytic input derivatives
include the interpolated mean and all posterior covariance blocks; task-label
derivatives are zero. Historical task labels within `1e-6` of integers are
rounded, matching the official validation.

The EM prior/model/coordinate-ascent families are integrated as described
above, including cold/warm reruns and learned explicit inducing locations.
The device-resident implementation below retains their higher-order graphs.
Historical union-grid locations and observation membership remain fixed when
explicit inducing points are absent; other configuration limits remain as
documented for each backend. Public exports alone do not establish exhaustive
empirical-family parity.

## Upstream standalone kernel shape bug

At official revision `b0dd01be6`, the ordinary GPyTorch call path assumes an
`n1 x n2` result for both empirical kernels because their `batch_shape` is
empty. With two historical outputs:

```python
import torch
from botorch.models.empirical_gps.empirical_1d_gp import EmpiricalOneDimensionalKernel
from botorch.models.empirical_gps.multioutput_empirical_1d_gp import MultiOutputEmpiricalOneDimensionalKernel

X = torch.linspace(0, 1, 4, dtype=torch.double).unsqueeze(-1)
Y = torch.randn(6, 4, 2, dtype=torch.double)
Q = X[:3]
EmpiricalOneDimensionalKernel(X, Y)(Q, Q).to_dense()
# RuntimeError: expected [3, 3], got [2, 3, 3]
MultiOutputEmpiricalOneDimensionalKernel(X, Y)(Q, Q).to_dense()
# RuntimeError: expected [3, 3], got [6, 6]
```

Their direct `forward(Q, Q)` methods return the intended covariance. The
native public call returns the intended result and tests compare to those
official forward methods. A proposed upstream fix is to give the independent
kernel the output batch shape expected by GPyTorch and to implement
`num_outputs_per_input` for the correlated kernel (returning the historical
output count). Preserve the model's existing vectorization behavior and add
standalone `kernel(Q, Z).to_dense()` and diagonal regressions when applying
the fix. Neither reference checkout was changed.

The independent output-batch and query-broadcasting repair is submitted in
[BoTorch #3500](https://github.com/meta-pytorch/botorch/pull/3500); the correlated
event-count repair is in [#3499](https://github.com/meta-pytorch/botorch/pull/3499).

## Upstream fixed-noise prediction of an unobserved task

The empirical multi-task model explicitly permits a task absent from training.
In official revision `b0dd01be6`, `MultiTaskGPyTorchModel._apply_noise` computes
the task's predictive observation noise by averaging an empty selection of
training noise values. `posterior(Q, observation_noise=True)` therefore has
NaN variance for an unobserved historical task, although its latent posterior
is finite. The regression in `test_empirical_multitask_gp.py` reproduces this
with two historical tasks, observations only from task zero, and uniform
fixed variance `0.023`; the task-one noisy variance becomes NaN upstream.

The native model averages fixed noise separately for observed tasks and uses
the global observed-noise average when a historical task has no training
observations. An upstream fix with the same convention is:

```python
mask = train_task_features == task_feature
noise_by_task[..., task_feature] = (
    self.likelihood.noise[..., mask].mean(dim=-1)
    if mask.any()
    else self.likelihood.noise.mean(dim=-1)
)
```

This preserves all observed-task estimates and follows the global averaging
convention already used by single-output fixed-noise GPs. Callers needing a
different noise assumption for unseen tasks can supply explicit predictive
noise upstream; explicit tensor-valued predictive noise is a current port
boundary. The reference source was not modified.

## Device-resident empirical curve models

`EmpiricalOneDimensionalGP` and `MultiOutputEmpiricalOneDimensionalGP` now
accept `backend="torch"`; CUDA or differentiable training/historical inputs
select this backend automatically. Their mean and kernel constructors accept
the same selection. The independent model retains independent outputs and
shared curve-weight leaves. The correlated model retains the full historical
cross-output covariance and conditions jointly on the interleaved observation
grid. Its public training inputs and prior distribution use the expanded
point/output layout, and its public posterior exposes the actual joint Gaussian.

Interpolation, centering, compression, and basis contractions use device-local
Torch Tensor operations. Exact factorization, Gaussian prediction, normalized
MLL and sampling use the Rust Torch dispatcher. No Tensor numerical evaluation
is flattened or transferred to the native CPU float64 backend. Historical knots
and curve values are actual registered buffers; they retain their data graphs.
Statistics and compressed bases are recomputed when evaluated, so independent
backward calls do not reuse an already-consumed construction graph. Knots must
be distinct, and out-of-range queries raise as in the reference. Derivatives of
piecewise interpolation apply away from interval/order changes. Spectral
derivatives use Torch's own eigendecomposition/SVD rules; repeated singular or
eigenvalues retain the usual spectral derivative limitations.

Both models support model batches, ARD, fixed observation variances, learned
Gaussian likelihoods, additive base kernels, custom Torch modules, first- and
second-order training/query/history/parameter derivatives, base-sample sampling,
conditioning, and fantasies. Correlated predictive Tensor noise broadcasts over
the point/output grid and is cast to the query dtype/device without detaching its
graph. Native empirical descriptors are converted from their actual fixed
interpolation buffers and curve weights, including descriptors built from
different historical values than the model's default history argument. Such
conversion preserves the descriptor's fixed-data status; it does not invent
gradients through the descriptor's earlier native decomposition. ARD parameters
and actual custom Torch Gaussian likelihood objects retain their live identities.
Fixed-noise models are constructed with `train_Yvar`; arbitrary likelihood types
raise. Input/outcome transforms and correlated `output_indices` remain explicitly
unsupported, matching these upstream model APIs. Independent output subsetting
updates the historical output tables as well as the likelihood and training data.

Validation in
`test_torch_empirical_gp.py` includes CPU/CUDA official mean, kernel, full
posterior and MLL parity, independent dense historical-curve mixed Hessians,
compressed-basis historical Hessians, additive kernels and real fits, native
descriptor conversion, explicit predictive/conditioning noise, live likelihood
identity, sampling, conditioning and fantasies. The existing native model and
conditioning suites pass alongside these tests. The standalone correlated lazy
metadata regression also proves the previously documented upstream repair:
`num_outputs_per_input` returning the historical output count makes the unchanged
forward covariance usable through the ordinary GPyTorch lazy call.


### Paired-history empirical multitask models

`MultitaskEmpiricalOneDimensionalGP` and its empirical mean/kernel now select
`backend="torch"` explicitly or automatically for CUDA, differentiable data,
and custom Torch modules. Each historical task retains its own progression
knots. Paired historical curves determine the full cross-task covariance;
the implementation does not replace it with independent task posteriors.
Only rows belonging to a task are interpolated against that task's knots, so
unrelated task domains do not impose spurious interpolation restrictions.

Exact training, MLL, conditioning, fantasies, sampling, and batched prediction
reuse the Rust Tensor Gaussian operations. Continuous training/query coordinates,
historical knots and curves, observed targets, learned noise, and additive
kernel parameters retain first- and second-order graphs on CPU and CUDA.
Categorical task labels have zero derivatives and are validated before gather.
Actual Tensor mean/kernel modules and converted native interpolation descriptors
are supported. Wide and homogeneous constructors assemble their data on device.
Fixed-noise prediction averages observations by task and uses the global observed
mean for tasks with no observations, as in the corrected ordinary multitask law.

The public model supports model batches, query batches, and fantasy batches;
`set_train_data` validates and updates the task metadata atomically. Predictive
Tensor noise remains an explicit limitation of the common multitask API, matching
the pinned reference's limitation. Validation in
`test_torch_empirical_multitask.py` compares unmodified official mean, covariance,
posterior and MLL results and independently checks paired-curve mixed Hessians,
noise-only automatic backend selection, historical-buffer mutation, additive
kernels, fits, constructors, conditioning and fantasies. The combined empirical
Tensor and legacy suites passed 137 tests at this checkpoint.


### Device-resident EM empirical priors and models

`pretrain_em_prior`, `EMEmpiricalGaussianProcess` and its `from_pretrained`
constructor accept `backend="torch"`. CUDA, differentiable inputs or actual
Torch modules select the Tensor path automatically. Indexed and explicit-inducing
E-steps, Gaussian-mean and inverse-Wishart MAP updates, trace-matched shrinkage,
PSD stabilization, convergence stopping and both initializations run on the
input device. Rust dispatches safe Cholesky and Gaussian prediction; the EM
updates use differentiable Torch matrix operations. Neither models nor priors
call upstream BoTorch at runtime or transfer numerical evaluation to CPU.

The MAP mean prior is centered on the actual parametric mean at the inducing
points: its right-hand side includes `K_mu.inverse() @ m(Z)`. Pinned upstream
`b0dd01be6` omits this term at the caller, shrinking toward zero. The port uses
the corrected law already proposed in [BoTorch #3388](https://github.com/meta-pytorch/botorch/pull/3388).
The Tensor regression oracle compiles an isolated copy of the official source
with exactly this missing `m_0` argument added; the source checkout stays
unchanged. Offset-equivariance tests verify that shifting historical and target
values together with the parametric mean shifts predictions without changing
the covariance or observed-data MLL.

Every training forward and historical MLL evaluation owns a fresh EM graph.
Cold reruns differentiate through every update; warm reruns start from detached
previous moments. Persistent learned moments are detached registered buffers,
so an older backward, a later warm update, serialization, device/dtype moves and
subsequent independent posterior backward calls do not reuse a consumed graph.
The historical MLL uses all historical datasets and their full indexing,
independently of the target conditioning observations. Both SciPy and Torch
fitters optimize the actual Tensor leaves through this objective. Scalar
Gaussian noise and custom Torch Gaussian likelihood, mean and kernel modules
are supported; the likelihood must remain homoscedastic as upstream requires.
`ExperimentDataset.Yvar` is retained as metadata and is not consumed by the EM
likelihood, matching the source law.

Pretrained containers own deep historical/module snapshots and detached moments.
Freezing pretrained modules preserves derivatives through live shift weights,
the full Schur residual, explicit inducing locations and an optional full
additive covariance. Target training/query data, predictive noise, parameters,
historical values and (with explicit inducing points) historical coordinates
retain mixed first/second derivatives. Without explicit inducing points, union
grid locations and observation membership are fixed discrete metadata. Inducing
locations become actual learnable Parameters only when requested. Conditioning,
fantasies, sampling, direct-index queries and batched prediction preserve the
same learned prior.

PSD projection uses a gap-centered spectral projector and an analytic
matrix-absolute/Sylvester rule for a mixed clipped spectrum, including repeated
eigenvalues. Centering inside the actual spectral gap avoids cancellation near
tiny positive floors. The backward solve has its
own analytic backward, so higher derivatives never differentiate eigenvectors.
All-positive and all-clipped spectra retain direct identity/constant branches.
Mixed float32 projections work internally in float64 before restoring dtype.
CPU/CUDA finite-difference `gradcheck` and `gradgradcheck` verify repeated mixed
spectra; independent divided differences and valid local finite differences
check first derivatives and Hessian-vector products near zero. Float32
third-derivative graphs are finite. The clipping threshold
itself remains nonsmooth. Training data are scalar
and unbatched; query/fantasy batches are supported. Input/outcome transforms
remain outside this upstream model API.

`test_torch_em_empirical_gp.py` covers CPU/CUDA corrected-source moments,
every recorded EM iteration, cold/warm MLL and parameter gradients, independent
mixed Hessians through explicit inducing points and historical data, immutable
warm snapshots, float32 zero-noise stabilization, reusable-container round trips,
native-prior conversion, condition/fantasy/sample derivatives, and both fitting
routes. Native EM and all existing empirical model regressions run alongside it.


### Tensor KL, coordinate ascent and shared-kernel fitting

The remaining empirical-prior public helpers have device-resident Tensor paths:
`kl_divergence_mvn`, `KLPriorFitMLL`, `fit_kernel_to_em_prior`,
`coordinate_ascent_em`, and `build_shared_gp_model_list`. Direct Gaussian KL
retains all four input graphs, including broadcast batches and mixed Hessians.
The KL fitting MLL detaches the target empirical moments as its upstream contract
requires, while keeping live reference-coordinate and model-parameter graphs.
Neither observation noise nor its priors enter the KL objective.

Coordinate ascent performs actual fixed-moment historical likelihood fitting,
then re-estimates EM moments using all historical datasets. Both SciPy and Adam
steps, fixed observation variance, device RNG subsampling, optional inducing
location optimization, automatic Adam selection, and deep container histories
are supported. Native module descriptors are converted at the boundary; fitted
parameter values are installed back into the caller-owned descriptors. Actual
Torch modules retain their parameter identities. With no explicit inducing grid,
the indexed fixed-moment objective can be constant in the kernel parameters;
this is handled as an exact no-op rather than the pinned source's backward error.
Registered parameter priors still participate if they give that objective a
gradient. Optimizing inducing locations requires explicit locations.

Shared model lists use one actual Tensor mean module and one actual Tensor
kernel module across all children, so shared gradients accumulate at those same
leaves and fitting optimizes the joint objective. Training targets remain on
their raw scale (`outcome_transform=None`), following the correction in
[BoTorch #3388](https://github.com/meta-pytorch/botorch/pull/3388); the native shared
factory now uses this corrected convention too. Each child's normalized exact
MLL is averaged over children, matching SumMarginalLogLikelihood. Learned noises
remain independent; explicitly fixed noises retain their values and frozen
flags. Port-native descriptors are converted to shared Tensor modules and
synchronized after the high-level fit; external Tensor optimizers should use
the actual modules exposed by the returned models. No independent native copies
are rebound and presented as shared Tensor leaves.

`test_torch_em_helpers.py` checks CPU/CUDA all-input KL Hessians, live reference
coordinates with detached empirical targets, actual KL recovery, caller module
identity and descriptor updates, raw-target joint likelihood and shared Hessians,
both coordinate optimizers with fixed/learned locations and selected datasets,
full-data EM refreshes, default no-op/zero-cycle behavior, and independent history
snapshots. The helper, Tensor EM and legacy KL suites passed 76 tests together.


### Public numerical utilities on Tensor inputs

The public functions in `models.empirical_gps.utils` now keep Tensor numerical
work and graphs on their device: curve centering, sample covariance, orthogonal
bases, mean/basis interpolants and weighted basis evaluation, PSD projection,
trace-matched shrinkage, and EM-prior synthesis. Unique-row grids and integer
membership maps also stay on device, with explicit detached/discrete membership
semantics. The native fixed-data basis implementation remains available with
`compute_orthogonal_basis(..., backend="native")`. Lists and NumPy keep their
native numerical paths. ARD initialization allocates Tensor weights on the
requested device and constraint checks compare actual Tensor bounds there.

`LinearInterpolation1D` becomes an actual Torch module for Tensor data, retaining
its four persistent state keys and the source's logical default fill dtype.
Raw source buffers retain data graphs; sorting and gathering are rebuilt during
each forward so independent backward calls work. Device/dtype moves preserve
these live sources. Loading state or explicitly replacing public table buffers
installs the supplied tables as interpolation data; it does not recreate the
external constructor tensors' former graphs. Public table snapshots own their
storage, so external updates to live constructor tensors preserve their data
graphs even with `assume_sorted=True`. Queries retain gradients through
loaded tables. Out-of-range filling and NaN query propagation match the source.
Mean/basis builders retain their ordinary construction graphs; rebuilding a
builder reconstructs its statistics or decomposition for a new training graph.
Spectral basis orientation and rank selection retain their usual nonsmooth
boundaries at repeated singular/eigenvalues and threshold crossings.

Two pinned-source defects are isolated in `test_torch_empirical_utils.py`:

- For a wide `3 x 8` matrix with `diag(1,2,3)` in the first three columns,
  `compute_orthogonal_basis(A, "eigh")` forms an `8 x 8` column Gram with five
  repeated artificial zero eigenvalues. Even the first derivative of
  `||(B.T @ B)||²` is NaN. The port uses the smaller row Gram when rows are fewer
  than columns and returns `U.T @ A`; rows retain their singular scaling and
  `B.T @ B == A.T @ A`. No artificial nullspace enters the derivative.
  The source repair is [BoTorch #3507](https://github.com/meta-pytorch/botorch/pull/3507).
- `LinearInterpolation1D` sorts differentiable knots/values once in its
  constructor. A second independent backward through the interpolant then
  raises a consumed-graph error. Rebuilding the sort/gather from raw sources
  during forward fixes this without retaining an old graph. The source repair
  is [BoTorch #3508](https://github.com/meta-pytorch/botorch/pull/3508).

CPU/CUDA tests compare complete covariance and basis Gram derivatives and mixed
Hessians, compressed and uncompressed interpolation pipelines, repeated module
backward calls, state round trips, fill behavior, synthetic moment recovery,
PSD gradients in mixed repeated eigenspaces, shrinkage, metadata and constraints.
The new device utility and legacy utility suites passed 73 tests together.
