# Empirical GP utilities

`botorch_rs.models.empirical_gps.utils` implements the 17 public utility
names in the official BoTorch source audited at revision `b0dd01be6`, plus
its two private interpolation helpers. This is the numerical foundation for
the empirical model families; the model families themselves are not claimed
complete here.

Numerical work runs in Rust: economy SVD and Gram eigendecomposition,
sample covariance, centering, interpolation, basis weighting, PSD projection,
trace-matched shrinkage and synthesis of curves from an EM prior. Python
converts arrays, permutes shapes, routes indices and registers metadata.
No NumPy, PyTorch or SciPy linear algebra or interpolation is used as a
numerical fallback. The matrix and interpolation bindings release the GIL.

The utilities preserve nested-list, NumPy and torch array kinds. Covariance
and basis decompositions support batches; interpolation uses the same
outer batch convention as the official helper. The covariance divisor uses
the original curve count, even when an orthogonal basis compresses it.
`build_basis_interpolant` supports per-output compression and vectorized
compression that retains cross-output covariance. SVD/eigenvector signs can
differ from torch; the defining Gram matrix is preserved.

First-order torch autograd is connected to native vector-Jacobian products
for covariance, PSD projection, shrinkage and interpolation. Interpolation
derivatives include knots, curve values and query locations. Native scalar
operations and reductions support centering and weighting derivatives.
Historical SVD/eigh decompositions and the covariance decomposition used to
synthesize EM curves currently require fixed historical data: requesting
their derivatives raises UnsupportedError. The EM mean remains
differentiable. Higher derivatives are not implemented.

Synthetic EM curves use deterministic orthonormal Helmert contrasts, with
`rank + 1` curves. Their mean and centered sample covariance reproduce the
requested prior, including the degrees-of-freedom correction. Individual
synthetic curves can differ from the official QR construction; their
probabilistic moments match. Interpolation requires at least two strictly
increasing finite knots; duplicate knots raise a clear error.

`instantiate_ard` attaches nonnegative native parameter-constraint metadata
and respects optional registration methods. Optional torch/GPyTorch modules
receive a torch-module metadata container around the native untransformed
constraint, including serializable bounds; this performs no tensor arithmetic.
The interpolation object stores
its data in a state dictionary and moves torch buffers with `.to`. It is a
native callable descriptor rather than a torch module subclass.

Validation includes native Gram/covariance/EM invariants, finite differences
of interpolation and matrix derivatives, and live official-source parity
for covariance, basis and mean interpolants, projection and shrinkage,
including their first-order derivatives. Metadata, bounds, shape validation
and all supported array kinds are checked through the public Python API.
