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

Interpolation, basis decomposition, empirical covariance, Cholesky inference,
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

## Upstream EM subsampling index mismatch

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
above. Warm starts and learned inducing locations remain configuration work;
the public exports do not establish full empirical-family completion.

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
