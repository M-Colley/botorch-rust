# Public path constructors

Audited against official BoTorch revision `b0dd01be6`.

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
`OutcomeUntransformer(Standardize(...))` uses native affine operations so
editing a composed path preserves query gradients through standardization.

Validation covers list, NumPy, and Torch values; official constructor values,
batch and ensemble layouts; query/weight/frequency gradients; explicit bias
and transform graphs; compositions of native prior/update paths; dictionary
edits; single/multiple-output standardization; outcome-transform snapshot
independence; deep copy; and explicit-callable serialization.

These classes remain array-oriented wrappers rather than `torch.nn.Module`
subclasses; they do not provide automatic Torch parameter registration.
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

The override suite compares independent official values, full noise/factor
derivatives and Hessians, callback transform semantics and restoration, all
native array kinds, loaded fully Bayesian ensemble layouts and actual GPyTorch
feature modules. The reference's kernel-distance Hessian is unavailable in
some versions, so its independent Hessian comparison uses the smooth
root/noise/target algebra; port query Hessians are also checked for finiteness.

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
