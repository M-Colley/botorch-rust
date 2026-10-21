# Device graphs in public transformation utilities

All ten functions in `models.transforms.utils` now operate directly on Tensor
inputs on their CPU/CUDA device. Normal/lognormal means, variances and full
covariances preserve batch broadcasting and use `expm1`/`log1p` for small values.
`expand_and_copy_tensor` creates separate storage and folds expansion gradients
back to the original batch. `subset_transform` retains declared column order,
negative indices, boolean masks, unchanged columns, transformation parameters
and broadcast batches. Interaction features and forward/inverse Kumaraswamy
warping retain input and concentration graphs. Tensor values are never flattened
through the CPU Rust array bridge. List/NumPy calls retain that native path
without importing Torch; ordinary non-Tensor constants accompanying a Tensor
are converted to its device and dtype.

The `nanstd` value still ignores missing entries and applies Bessel's
correction. Missing values are masked **before** subtraction and squaring,
so they cannot contaminate valid observed-coordinate derivatives. Constant
slices select the zero subgradient, with a zero differentiated branch; a
mathematical Hessian at the norm's nondifferentiable origin is not claimed.
Slices with fewer than two observations retain their undefined NaN output.

Pinned official BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2`
computes `nanstd([[.2, nan, .8, .5]], -1) == .3`, but its first gradient is
`[[nan, nan, nan, nan]]`. It squares a NaN before reducing with `nanmean`,
allowing inactive arithmetic to contaminate backward. The port's derivative
matches standard deviation computed only on observed entries. A distinct
upstream helper repair is being prepared; existing BoTorch #3468 changes
outcome-transform code rather than this public helper.

Validation adds 46 CPU/CUDA cases. Analytical lognormal marginal derivatives
and directional Hessian products are checked independently, including tiny
variance precision. Full-covariance conversions pass numerical gradcheck and
gradgradcheck; a round trip has the identity Jacobian and zero Hessian. Tests
also check expansion gradient folding, feature interactions, physical-column
selection and parameter graphs, masked-standard-deviation derivatives,
concentration derivatives and numerical second derivatives of both warps.
Direct source values/first derivatives and strict host-conversion guards
complement these independent controls. All 233 focused new/legacy transform,
native-array parity and Tensor outcome cases pass on Torch 2.13 CPU/CUDA.
Torch 2.14 CPU passes 23 new cases and skips 23 CUDA cases. Ruff and Mypy pass.
