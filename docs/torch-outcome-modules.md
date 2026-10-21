# Public outcome transforms as Torch modules

When Torch is already loaded, constructing a public `Standardize`,
`StratifiedStandardize`, `FlattenedStandardize`, `Log`, `Power`, `Bilog`, or
`ChainedOutcomeTransform` produces an actual `torch.nn.Module`. Chained
transforms also implement the actual `nn.ModuleDict` interface. Importing and
using the native package without Torch retains the native descriptor path.

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

The existing Tensor model extensions for selected output affine maps and
repeated output subsetting remain available. Unselected standardization
columns have zero means and unit scales, including in transformed Gaussian
posteriors. Array-based direct calls retain their existing native behavior.

Unchanged built-in CPU descriptors remain eligible for automatic native
inference. CUDA state, live gradient buffers, caller Parameters, custom
classes, and numerical method overrides select Tensor inference through the
shared backend selection policy. The actual transform module is available
for explicit `backend="torch"` model attachment.

Validation covers all seven families on CPU/CUDA, native automatic selection,
source numerical controls, flat registered state, serialization and migration,
module ownership in all three structured Tensor models, custom registered
outcome modules, and full first and mixed second derivatives for supplied
Power Parameters and posterior samples. The standalone module checks also
run with BoTorch, GPyTorch, linear_operator, SciPy, and probabilistic runtimes
blocked from import.
