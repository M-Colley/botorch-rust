# Public scalar math on Torch tensors

`utils.safe_math`, the univariate Gaussian helpers in
`utils.probability.utils`, and the scalar algebra and reductions shared through
`_ops` now execute device-resident Torch expressions. CUDA tensors retain their
device and first, second and mixed derivatives; no numeric tensor is flattened,
detached or transferred to CPU. NumPy and list arguments continue to execute
the native Rust algorithms. The Torch module loads lazily, so list-only imports
do not require Torch.

This includes safe arithmetic, logarithmic and fat-tailed smooth functions,
temperature reductions, Gaussian CDF/density/log-tail functions and normal
interval probabilities. Feasibility probabilities retain the graph through
means, standard deviations and tensor constraint bounds. Temperature tensors
retain their actual graph, including softplus's dependence on inverse
temperature. Percentile ranks execute on device with the reference's discrete
counting semantics; they are not differentiable membership operations.

The public bivariate Gaussian integrators (`bvn`, `bvnu`, `bvnmom`), MVNXPB and
other multivariate integration/sampling classes retain their existing native
CPU implementations. Their old first-order bridges do not constitute GPU or
higher-order-autograd support. This checkpoint makes no such claim.

The tests in `test_torch_public_math.py` compare float32/float64 values, weighted
gradients and Hessian-vector products with the pinned official reference on
CPU/CUDA; they also run gradcheck/gradgradcheck, mixed temperature/bound checks,
scalar dtype-promotion checks, and forbid host-buffer conversion at runtime.
The earlier native array parity tests continue to cover NumPy and lists.

## Stable complementary-error-function tails

The unmodified reference's `log_erfcx` evaluates a direct `erfcx` backward whose
large positive derivative subtracts nearly equal quantities. With a float64
input of `1e50` on Torch 2.13, it returns a first derivative of `0` and a second
derivative of `2`; the asymptotic derivatives are `-1e-50` and `1e-100`.

The port uses the standard large-positive expansion

`log(erfcx(x)) = -log(x) - log(pi)/2 + log(1 + sum_k (-1)^k (2k-1)!!/(2*x*x)^k)`.

It evaluates twelve inverse-square terms above 8 for float32 and 20 for
float64. This avoids cancellation and preserves derivatives of the full
expression. Ordinary arguments use Torch's special function; negative
arguments use `log(erfc(x)) + x*x`. Branch inputs are masked before evaluation,
which also avoids inactive-branch NaNs at infinite limits. The isolated
`test_log_erfcx_large_tail_has_correct_first_and_second_derivatives` regression
checks the mathematical tail independently of the faulty reference gradient.
