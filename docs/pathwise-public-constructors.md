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
constructor. Native model sampling restrictions, including unbatched
`SingleTaskGP`/`ModelListGP` factories and supported stationary kernels,
continue to apply. Public Pyro/NumPyro generative effect-handler methods are
separate remaining work.
