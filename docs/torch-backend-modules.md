# Torch backend parameter modules

The optional Torch model backend uses port-owned `torch.nn.Module` components
from `models/_torch_backend_modules.py`. It does not bind parameters to the
CPU/f64 parameter mirror. Parameters and buffers move with `.to(...)`, and
ordinary Torch optimizers consume the same parameter objects used in inference.
Covariance evaluation dispatches through Rust `tensor_kernel` to the installed
Torch runtime, retaining the input device, dtype, and autograd graph.

`sample_all_priors` samples actual Torch-backend Parameters in place. The eight
owned univariate families (Normal, LogNormal, Gamma, Uniform, HalfCauchy,
HalfNormal, Beta and SmoothedBox) generate draws on their prior's device and
dtype. SmoothedBox retains its uniform middle and Gaussian tails with their
correct mixture weight; it is not approximated by a bounded uniform. Its
discrete mixture selection follows the reference sampler, rather than claiming
a pathwise derivative of the mixture weight. Model prior sampling runs under
`no_grad` and preserves Parameter identities and trainability flags.

Module and model-list aliases are visited once. Registered GPyTorch priors and
their setting closures are supported, including scalar singleton prior axes
and non-scalar prior batches. User-supplied prior implementations retain their
own sampling behavior; outputs are converted to the target Parameter's dtype
and device before its setter. This also handles GPyTorch transformed priors
whose cached base distribution remains CPU/float32 after `Module.to` moves
their buffers. The owned prior samplers do not have that cache/device mismatch.

An explicit `seed` isolates and restores CPU and participating CUDA RNG states.
Without a seed the usual Torch RNG states advance. Constraint failures retry up
to `max_retries`, restoring the previous raw values after each rejected draw;
an exhausted attempt limit raises. Missing inverse-setting closures and
unrelated sampler failures raise directly. A foreign prior without sampling
support emits `BotorchWarning`, as in the reference. The focused 51 CPU/CUDA
sampling cases compare owned draws against independent Torch distributions,
check SmoothedBox tail probability and moments, and exercise exact/SVGP
Parameters, aliases, retries, foreign priors, seeded RNG isolation and surviving
posterior graphs.

The current converter supports RBF, Matérn (0.5, 1.5, 2.5), RQ, Periodic,
Cosine, Linear, Polynomial, Scale, Additive and Product kernels. ARD, active
dimensions, model batches, feature batches, shared kernel objects and configured
constraints/priors retain their respective parameter layouts. Custom Torch
kernel and mean modules can also be supplied directly. Other port kernel
families raise `UnsupportedError` rather than using a CPU numerical fallback.

ConstantMean, ZeroMean, Gaussian likelihoods, fixed observation noise,
Normalize, Warp and ChainedInputTransform are implemented as Torch components.
Warp concentrations are actual Parameters with optimizer bounds. Normalize and
Standardize expose differentiable `fit` methods; a model can rebuild fitting
statistics from its original training tensors before each differentiable
evaluation. This retains the functional derivative through training-data
preprocessing; BoTorch's ordinary eval-mode cached-training-input path detaches
that graph. Values and the differentiable transform formulas match the
reference, while the port deliberately retains training-data gradients in this
backend. Query evaluation reuses fitted statistics. Standardize converts
independent output covariance blocks with shape `(..., m, q, q)`.

The Gaussian likelihood registers its learned parameter at
`noise_covar.raw_noise`; its convenience `raw_noise` accessor refers to that same
leaf. Calling the likelihood on function samples returns conditional Normal
distributions. Calling it on a latent Gaussian adds noise before any required
Cholesky factorization. Fixed-noise inspection buffers are detached initial
snapshots; the model recomputes its transformed observation noise from the
original training Tensor for inference and gradients. Noise floors are 1e-4 for
float32 and 1e-6 for float64, matching GPyTorch defaults.

Log, Bilog, Power and ChainedOutcomeTransform adapters preserve complete Torch
sample graphs through `TransformedPosterior`. Log provides analytic lognormal
means and variances and the reference delta-method observation-noise transform.
Power and Bilog retain the reference's explicit missing-moment and
observation-noise errors. Standardize un-transforms Gaussian covariance directly
and affine-transforms nonlinear posterior samples and moments where available.
Returned posterior transforms capture their fitted statistics so later refitting
does not change an existing posterior. Plain Torch Standardize modules are
converted by their parameter/statistics protocol without importing BoTorch.

Custom Torch modules with GPyTorch's `named_priors` registration protocol can
use `module_log_prior(module, batch_shape)`. The helper evaluates the actual
registered prior closures with Tensor operations, reduces event dimensions,
preserves batch axes and shared leaves, and requires no BoTorch numerical code.

Prior terms reduce parameter event dimensions and preserve model/output batch
axes. Shared kernel/prior objects are visited once. The default LogNormal prior
retains its construction-precision variance and log-scale constants, matching
GPyTorch's cached Normal base distribution even when parameters use float64.
The buffers reside on the selected device; parameter log densities retain first
and higher derivatives.

## Verification and reference limitations

`test_torch_backend_modules.py` and `test_torch_backend_outcomes.py` pass 262
cases on the available CPU and CUDA device. Tests compare values and first derivatives to the installed GPyTorch
1.15.2 / official BoTorch reference, compare Hessian-vector products to central
differences of reference first derivatives, check full versus diagonal
covariance and finite self-covariance Hessians in float32/float64, and exercise
optimizer updates, held leaf identities, shared parameters and transform
statistics gradients. These checks establish the listed components; they do
not establish GPU/autograd coverage for every model or acquisition family.

Three reference details require care when constructing an independent oracle:

* GPyTorch Matérn evaluations use `torch.cdist`, whose backward does not implement
  a second derivative in the tested Torch runtime. First derivatives are valid;
  central differences of those derivatives check the port's differentiable
  pairwise-distance calculation. A differentiable broadcast difference and
  squared-distance implementation repairs the missing higher-order path.
* GPyTorch's composite lazy covariance wrapper can predict the original feature
  dimension after child active-dimension selection. Eager reference evaluation
  avoids that incorrect shape assertion while retaining the kernel formula.
* GPyTorch PolynomialKernel's offset broadcasting omits the feature axis for
  simultaneous model and feature batches. Distinct per-batch offsets expose
  either a shape error or an incorrect broadcast when dimensions happen to
  agree. The correct offset has an additional singleton feature axis. The
  regression oracle evaluates each selected feature with the unchanged
  non-feature reference kernel and stacks those results.

Prior constructors taking Python floats can also retain cached CPU/float32
transformed-distribution state after GPyTorch `.to(...)`; tests of a general
device-local prior construct the reference from Tensor hyperparameters on the
intended device and dtype. A separate default-prior regression checks the
historical cached-constant behavior to 2e-14 instead of relaxing MLL tolerance.

Bilog's reference expression `sign(y) * log1p(abs(y))` gives Torch derivative
zero at `y=0`, although its mathematical derivative is one. The Torch adapter
uses domain-safe positive/negative branches, giving slope one for both Bilog
and its inverse. Away from zero, values and first/second derivatives match the
reference. The function is C1 and is not C2 at zero; no unique Hessian exists
there. The repair is checked with a centered finite difference at the origin.

Constant-target Standardize gradients also need a reference repair. Computing
`sqrt(variance)` at zero before replacing small standard deviations with one
can backpropagate `0 * inf` through the inactive branch. For example, training
`Standardize(1)` on `torch.ones(3, 1, requires_grad=True)` gives NaN first and
second target derivatives in the tested reference. The adapter clamps variance
to `min_stdv**2` before taking its square root, then applies the original
constant-one branch according to the original variance. Values are unchanged.
CPU/CUDA float32/float64 regressions cover constant, near-constant and single
observations, with finite differences small enough to stay below the threshold.

Direct public `Log`, `Power` and `Bilog` calls now select device Tensor algebra
for Tensor outcomes, including inverse mappings, selected columns and supported
noise transforms. Their public posterior sample transforms also preserve the
incoming Gaussian graph. Previously these direct entry points flattened Tensors
into native arrays even when a model's converted transform worked correctly.
Eighteen new CPU/CUDA cases check source values, complete first/higher graphs,
inverse round trips and no-host-copy guards. The combined direct/model outcome,
native transform and transform parity suites pass 207 cases. Torch 2.14 CPU
passes all 59 selected direct/domain/randomized checks and skips nine CUDA
cases. Undefined inverse-power samples are classified separately as described
in [equivalence verification](equivalence-verification.md).

Direct evaluation-mode calls on exact scalar/independent-output Tensor GPs now
consume already transformed coordinates, as ExactGP does. They broadcast the
existing model/output axes and return Gaussian laws in transformed outcome
units. Previously this path transformed the query again and could insert a
duplicate independent-output axis. `forward` remains the latent prior;
`posterior` retains its separate raw-input/output-axis/outcome-untransform
protocol. The distinction also propagates through scalar fully Bayesian,
supplied Gaussian likelihood, relevance-pursuit and empirical overrides.

Thirty-six new CPU/CUDA cases compare source conditional/prior values and
query gradients across normalization, warping, model/query/output batches and
actual supplied likelihoods. Independent dense Gaussian calculations verify
complete data/parameter Hessians, including actual fully Bayesian loader
transforms and correlated noise. The initial 32 cases and existing exact,
subclass, cache, relevance, fully Bayesian loader and empirical regressions
pass 322 combined cases; the four additional loader controls pass separately.

## Direct standardization transforms

Public `Standardize` and `StratifiedStandardize` calls now dispatch Tensor
data to port-owned device calculations. Training mode fits graph-bearing
means and unbiased standard deviations; eval mode applies the current
statistics. Observation-noise scaling, inverse transforms and selected
outputs retain first and higher derivatives. Constant and single-observation
data use the documented scale-one branch with finite derivatives. NaN
observations remain NaN in the result and are excluded from fitted statistics;
masked observations cannot contaminate derivatives of the observed entries.

`to`, `cpu`, `cuda`, `float` and `double` migrate actual Tensor statistics.
`state_dict` and `load_state_dict` use the four reference statistics keys
(`means`, `stdvs`, `_stdvs_sq`, `_is_trained`). Stratified task remapping also
migrates, as a nonpersistent buffer. These descriptors retain optional-package
support and do not claim the complete `torch.nn.Module` protocol. Explicitly
returning to list or NumPy calls transfers the fitted statistics to Rust;
batched stratification requires Tensor inputs.
The source mapping-migration repair is submitted in
[BoTorch #3521](https://github.com/meta-pytorch/botorch/pull/3521).

Stratified fitting reduces each batch and task independently and applies an
identity transform to unknown tasks. Affine posterior conversion scales the
complete Gaussian covariance, including cross-output terms, preserves
Gaussian-mixture posteriors, and keeps an immutable snapshot of the fitted
law. Other posterior types retain differentiable sample and moment closures.
As in the reference, `StratifiedStandardize.subset_output` and standardization
of selected outputs through `untransform_posterior` remain unsupported.

An already-fitted eval `Standardize` descriptor passed to a Tensor GP keeps
its calibration statistics. Calling `model.train()` enables fitting on the
model's training targets again. Training calls produce fresh fitting graphs;
eval mode retains the normal Torch graph lifetime of the stored statistics.
Reusing an already-consumed external calibration graph still requires a new
fit or the caller's usual `retain_graph` handling.
