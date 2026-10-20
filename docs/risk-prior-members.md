Public risk-measure and prior methods
====================================

`MVaR.make_differentiable` computes each component from the mean of all
exactly matching prepared sample components in Rust. Its native adjoint
shares gradients across ties, returns no gradient for the supplied MVaR
locations, and preserves upstream NaN output / zero sample gradient when a
component is unattained.

`MARS.chebyshev_objective` caches the scalarization callable until its
weights or baseline change. Native array operations preserve input
autograd, first-minimum tie gradients, reference-point normalization, and
infinite values. Previously retrieved callables retain their normalization
bounds while reading the current weights, as in BoTorch.

`BetaPrior.expand` preserves the scalar native prior, validation setting,
array type, and Tensor.expand shape rules. Concentrations may be scalars or
single-element arrays; heterogeneous concentration arrays remain outside
the native scalar-prior representation. Density and input derivatives are
evaluated natively after broadcasting, including empty batch dimensions.

Reference and upstream correction
---------------------------------

The parity tests target official BoTorch revision `b0dd01be6`. They also
support released BoTorch 0.18.1 through its underlying PyTorch Beta density
when its prior constructor fails with the current PyTorch read-only
concentration property. No runtime reference dependency is used by the port.

Official `BetaPrior.expand((0, 3))` with `validate_args=True` reconstructs a
new distribution whose concentration validation fails with an ambiguous
empty-tensor `reshape(..., -1)`. The port supports this valid empty
expansion. Upstream can preserve validation without revalidating empty
parameters by expanding the existing Beta / Dirichlet distribution, as
PyTorch's `Beta.expand` already does. The regression compares with that
actual distribution expansion, including empty shapes and input gradients.
