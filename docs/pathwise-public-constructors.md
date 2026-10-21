# Public path constructors

Audited against official BoTorch revision `b0dd01be6`.

Direct Tensor `OutcomeUntransformer` calls use the actual outcome transform's
Tensor inverse, preserving calibration-target and path-value graphs. The former
`Standardize` shortcut exported statistics to native arrays and detached
calibration derivatives. Twelve new CPU/CUDA cases check batched/independent
outputs, source values and gradients, independent mixed higher derivatives and
no-host-copy guards; the new and existing path utility selection passes 48
cases. Native array callers retain their affine path.

`GeneralizedLinearPath(feature_map, weight, bias_module=None,
input_transform=None, output_transform=None, is_ensemble=False,
ensemble_as_batch=False)` evaluates the supplied feature map and multiplies
its features by the supplied weights. Native feature maps and all matrix
products use Rust, including their first-order Torch query, frequency, and
weight adjoints. Bias modules, keyword arguments to feature maps, and both
ensemble layouts are evaluated. Explicit caller callables are optional
interoperability: their own implementation controls their tensor graphs.
The package does not import Torch to construct or evaluate list/NumPy paths.

`MatheronPath(prior_paths, update_paths, input_transform=None,
output_transform=None)` is a real `PathDict` with keys `prior_paths` and
`update_paths`. Its join adds arrays elementwise, including nested lists.
Both public constructors keep caller-supplied tensors instead of flattening
them at construction, preserving their graphs. Explicit paths containing
picklable callables can be pickled; native path handles are immutable and
support Python deep copy by sharing the fixed native snapshot, but do not
implement native-handle pickle serialization.

Factory-generated Matheron paths retain the direct native whole-path
evaluation when their components, join, and transforms are unchanged.
Replacing/deleting dictionary entries or replacing a transform now evaluates
the current composition. Previously those edits were silently ignored.
`forward` returns the sum before the outer outcome untransform, matching
BoTorch. The factory snapshots the fitted outcome transform, and
`OutcomeUntransformer(Standardize(...))` uses native affine operations for array
paths and the actual Tensor inverse for device paths, retaining calibration
graphs as well as path-value derivatives.

Validation covers list, NumPy, and Torch values; official constructor values,
batch and ensemble layouts; query/weight/frequency gradients; explicit bias
and transform graphs; compositions of native prior/update paths; dictionary
edits; single/multiple-output standardization; outcome-transform snapshot
independence; deep copy; and explicit-callable serialization.

Tensor generalized-linear paths, Tensor feature maps, and compositions of
Torch modules are actual `torch.nn.Module` instances. They retain supplied
Parameter identities, register ordinary path weights and feature arrays as
buffers, and register their kernel, bias, transform and component modules.
Forward hooks see transformed inputs and the output before the outer inverse,
matching the source mixin. State dictionaries, explicit dtype/device migration,
whole-object serialization with picklable callables, and shared component aliases
are covered separately from numerical parity. Generated device paths use
picklable joins and inverse-transform modules, and migrate their explicit draw
records with the path. Array constructors keep their native wrappers and do not
require importing Torch. When Torch is already in use, the six public path
transform families also expose actual module hooks, child registration, state
dictionaries and migration. Their integer selector and output-count metadata
are registered buffers. Fifty new CPU/CUDA cases exercise these contracts and
independent first/mixed second derivatives; without reference dependencies,
46 pass and four source comparisons skip. The path-transform subset passes 156
cases. These focused counts are not a release certificate or evidence for every
inherited Torch API.

Caller-defined subclasses of the public path, feature-map and transform
classes retain their constructors and forward overrides while exposing the
same module registration, hooks, migration and serialization protocols. Forty-four
additional CPU/CUDA controls cover eleven subclass families and compare values,
all active first derivatives and mixed second derivatives with the unmodified
reference. The combined module and subclass selection passes 226 cases;
the subclass controls also pass 22 cases on Torch 2.14 CPU and 22 without
reference dependencies (22 reference-only cases skip).

Factory-produced native generalized-linear paths still expose their existing
fixed weight/draw metadata and reject assignment to their immutable native
weight snapshots. Their feature metadata is not a replacement for the fully
callable maps returned by `gen_kernel_features` or supplied to the public
constructor. Default feature generation supports RBF/Matérn kernels, optionally
scaled, including ARD, active dimensions and model batches. Native loaded
SAAS/Matérn fully Bayesian models retain their MCMC batch and ensemble query
layout. Device paths also accept actual GPyTorch stationary kernel modules;
their feature algebra retains first- and higher-order graphs.
`MatheronPathModel` evaluates these composed native batch paths without assuming
a shared native handle. It preserves the source ensemble flag even with an
empty path sample shape. Its explicit seed also controls feature generation
when the default prior is partially bound to a feature count.

`draw_kernel_feature_paths` accepts `map_generator`, `weight_generator`,
`input_transform`, `output_transform`, and model-list `join` overrides. The
weight generator receives the actual feature count returned by the map.
`draw_matheron_paths` invokes arbitrary prior/update callables. During these
callbacks `model.outcome_transform` is absent, matching the reference, while
functional training-data helpers retain the fitted transform needed to compute
transformed targets. Scoped state is restored on normal and exceptional exits.
Prior callbacks must return callable sample paths with a callable `forward`
method; update callbacks must return callable paths. Malformed returns raise
`TypeError` at construction and preserve the model's outcome-transform state.

Exact `gaussian_update` supports alternate transformed `points`,
`target_values`, full `noise_covariance`, Gaussian likelihoods and a caller's
`scale_tril`. Native list/NumPy paths factor and solve in Rust. Tensor-backend
paths retain all input, target, sample, covariance, factor and Gaussian-noise
parameter graphs on CPU/CUDA. A native model rejects differentiable override
arrays explicitly; construct it with `backend="torch"` for those graphs.
Fixed training noise contributes zero covariance when alternate point counts
have no corresponding noise entries, matching `FixedGaussianNoise`.
For a supplied actual Gaussian likelihood, Tensor updates evaluate its full
Gaussian marginal at the update points. The observation covariance includes
correlated input-dependent noise, and any additive observation-mean shift is
subtracted from the residual. A diagonal `.noise` view is not substituted for
that marginal. Independent dense conditioning checks include all training,
query, likelihood, kernel, prior-sample, and base-normal first and mixed second
derivatives, with repeated independent backward calls.

The override suite compares independent official values, full noise/factor
derivatives and Hessians, callback transform semantics and restoration, all
native array kinds, loaded fully Bayesian ensemble layouts and actual GPyTorch
feature modules. The reference's kernel-distance Hessian is unavailable in
some versions, so its independent Hessian comparison uses the smooth
root/noise/target algebra; port query Hessians are also checked for finiteness.

Pinned `b0dd01be6` accepts a Tensor noise covariance but calls its removed
`Tensor.cholesky()` method on Torch 2.14. The port already uses the supported
linear-algebra API. Those override regressions therefore use the independent
reference kernel and a dense Gaussian solve; they retain strict values, complete
first derivatives and the supported Hessian comparison on the symmetric
covariance and lower-triangular factor domains. The source repair is published
as [BoTorch #3516](https://github.com/meta-pytorch/botorch/pull/3516), head
`87cfecc4ad5569e9b6ecf84159fe6ec519505579`. Against the unchanged source, the
three new methods expose ten failing CPU subcases on Torch 2.14.1. With the
Tensor-only repair, all seven update/posterior-sampling methods pass on Torch
2.13 CPU/CUDA and Torch 2.14.1 CPU; the original LinearOperator branch remains
unchanged. The new methods cover twenty CPU/CUDA dtype/factor subcases on 2.13
and ten CPU subcases on 2.14.1. Source formatting and lint checks pass.

The port-specific `draws` replay extension applies to the default optimized
factories. Arbitrary callback compositions do not expose a joint draw record
and cannot combine custom generators with explicit native `draws`. Native
scalar variational posterior updates remain a separate unsupported model
dispatch; device scalar whitened variational updates are implemented. The
reference also does not register independent multitask variational Matheron
updates or observation covariance overrides for variational updates.

Optional Pyro/NumPyro generative effect-handler
interoperability is documented separately in
[`numpyro-trace-interoperability.md`](numpyro-trace-interoperability.md).
