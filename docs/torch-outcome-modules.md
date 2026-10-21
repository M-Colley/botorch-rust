# Public outcome transforms as Torch modules

When Torch is already loaded, constructing a public `Standardize`,
`StratifiedStandardize`, `FlattenedStandardize`, `Log`, `Power`, `Bilog`, or
`ChainedOutcomeTransform` produces an actual `torch.nn.Module`. Chained
transforms also implement the actual `nn.ModuleDict` interface. Importing and
using the native package without Torch retains the native descriptor path.

Descriptors constructed before Torch is imported promote in place when the
caller explicitly requests migration, state serialization, buffer/parameter
traversal, or module hooks. Existing aliases and fitted statistics are
preserved, and chained children promote before module registration. Native
pickles loaded after importing Torch restore directly into real modules.
Native array calls alone never import Torch.

These modules support registration in a parent module, registered Parameters
and buffers, forward hooks, module traversal, device and dtype migration,
`state_dict` loading including `assign=True`, deepcopy, pickle, and Torch
serialization. Standardization statistics use the source's flat buffer names;
the private numerical implementation introduces no extra state prefix.
Stratification mapping is a nonpersistent buffer and follows device moves.

Tensor models keep the identity of supplied public outcome modules. Their
training calibration and posterior transformations retain training data,
query, supplied transform Parameter, and higher derivative graphs. Chained
transforms execute children through their actual module call protocol, so
child hooks run too. Chains can include supplied external Torch outcome
modules; calibration refits their training statistics without changing their
evaluation mode. A supplied `Power` Parameter remains the same registered
Parameter through direct calls and posterior sample transformations.

Custom supplied Torch modules and BoTorch Standardize subclasses retain their
actual forward callbacks, hooks, registered state, and object/Parameter
ownership. Calibration supplies training input context and refreshes its
complete data and Parameter graph. Custom public port subclasses use the same
forward calibration protocol. For explicitly supplied BoTorch Gaussian
callbacks, the port constructs their expected Gaussian envelope and converts
the result back to its differentiable Tensor law; GP fitting and prediction
remain implemented in the port. Canonical source built-ins without custom
callbacks retain their existing native Tensor conversion.

Supplied BoTorch chains retain their source call behavior, including the
source's child-hook suppression. The public port chain honors child hooks;
the source repair is proposed in
[BoTorch PR #3528](https://github.com/meta-pytorch/botorch/pull/3528).

The existing Tensor model extensions for selected output affine maps and
repeated output subsetting remain available. Unselected standardization
columns have zero means and unit scales, including in transformed Gaussian
posteriors. Array-based direct calls retain their existing native behavior.

Unchanged built-in CPU descriptors remain eligible for automatic native
inference. CUDA state, live gradient buffers, caller Parameters, custom
classes, and numerical method overrides select Tensor inference through the
shared backend selection policy. The actual transform module is available
for explicit `backend="torch"` model attachment.

For native models, the registered fitted outcome buffers also control the
inverse predictive map. Mutating means or scales, or loading fitted outcome
state, refreshes that map before prediction, conditioning, or constructing an
acquisition snapshot. Stored transformed training targets, observation noise,
and training factors remain unchanged. This includes independent output and
model batches, multitask and structured models, and all fully Bayesian members
and LCEM fantasy members. Existing native acquisition snapshots retain their
usual parameter-state validation contract.

Stratified transforms use the actual registered `strata_mapping` table during
calibration and inverse transformations, including remapping or sharing table
rows. Native inverse maps read the same table. The source's separate mapping
migration defect is tracked by
[BoTorch PR #3521](https://github.com/meta-pytorch/botorch/pull/3521); source
CUDA equivalence controls construct task values on the destination device.
First and second derivative controls use the source's unmodified `Standardize`
on each observed stratum, since its masked `nanstd` backward defect is tracked
by [BoTorch PR #3520](https://github.com/meta-pytorch/botorch/pull/3520).

Validation covers all seven families on CPU/CUDA, native automatic selection,
source numerical controls, flat registered state, serialization and migration,
module ownership in all three structured Tensor models, custom registered
outcome modules, and full first and mixed second derivatives for supplied
Power Parameters and posterior samples. The standalone module checks also
run with BoTorch, GPyTorch, linear_operator, SciPy, and probabilistic runtimes
blocked from import.

The native buffer-authority and mapping regressions passed together with the
existing direct, native-state, module, and Tensor outcome checks: 248 tests on
Torch 2.13 CPU/CUDA with the rebuilt 0.3.0 native extension. Negative controls
reproduce the original native mean error and the ignored mapping permutation.
