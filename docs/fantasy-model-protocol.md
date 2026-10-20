# Generic fantasy conditioning

The shared `FantasizeMixin` now requires a posterior, input transform and
conditioning implementation, and scopes the fantasy and gradient settings
around posterior evaluation and sampling. It computes the average observed
noise for fixed-noise models, preserves output dimensions, expands explicit
noise over candidate batches, and handles empty queries without invoking a
sampler. Lists, NumPy arrays and Torch tensors follow the same noise layout.
The Torch path retains a caller's query and explicit-noise graphs.

`Model.batch_shape` supplies the same abstract-property error as BoTorch.
Concrete native models retain their own batch metadata. An abstract base
cannot infer a model batch from an arbitrary callable.

The focused protocol tests include official-reference comparisons, scalar
and multiple outputs, multidimensional sample shapes, context restoration
on exceptions, and empty queries. Pairwise posterior/conditioning and
acquisition parity tests also pass with the shared implementation.

## Multidimensional sample noise

At official revision `b0dd01be61d517c387e234903a6e2cc38c057ea2`,
`FantasizeMixin.fantasize` expands noise to `Y_fantasized.shape[1:]`.
This removes exactly one sample dimension. With sample shape `(2, 3)`,
two query points and one output, the samples have shape `(2, 3, 2, 1)`;
the forwarded noise has shape `(3, 2, 1)` and includes a fantasy axis.
Noise should instead have shape `(2, 1)`, broadcast across both fantasy
dimensions by the conditioning implementation.

The native mixin removes `len(sampler.sample_shape)` dimensions. The
proposed `patches/botorch-fantasy-noise-sample-shape.patch` makes that same
change in both the empty and nonempty official-reference branches.
This addresses noise layout; a particular model may separately restrict
the fantasy batch shapes that it can condition on. No upstream submission
has been made.

Native models still have separate training-data differentiation limits.
Setting `propagate_grads=True` scopes the generic model protocol and allows
custom differentiable model implementations to observe the setting; it
does not add training-data adjoints to models that lack them.
