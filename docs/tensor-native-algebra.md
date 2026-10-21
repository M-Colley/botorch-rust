# Differentiable tensor algebra

The optional `tensor_*` extension functions dispatch Torch operations from Rust
without linking a particular libtorch binary interface. Tensor values stay on
their original CPU or CUDA device. Torch owns the numerical kernels and their
autograd graphs; these functions do not copy tensor values into Rust arrays.
Importing the extension does not import Torch.

The exact Gaussian conditioning and log-likelihood functions accept covariance
blocks, means, targets and diagonal noise as tensors. Their graphs include
training inputs through the kernel blocks, training targets, query inputs,
means, likelihood noise and kernel parameters. Independent outputs can occupy
ordinary broadcast batch dimensions. The likelihood function returns the
Gaussian data log probability; `normalize=True` divides by the number of
training points. Model-prior terms belong in the model's likelihood wrapper.

`tensor_exact_gp_factor` exposes the training Cholesky factor and residual
weights. `tensor_exact_gp_predict` reuses those tensors and evaluates only the
candidate covariance blocks and triangular solve. Neither function detaches its
inputs or stores a hidden cache. Model wrappers decide when frozen training
state permits caching; when the training graph is live, reuse still preserves
all its gradients. Tests check both complete training graphs and second query
derivatives with frozen factors, and count one training factorization across
multiple predictions.

Query-only batch dimensions are packed into columns of one triangular solve.
For example, sixteen `n x 2` cross-covariance blocks sharing one training factor
become one `n x 32` right-hand side. This avoids Torch materializing sixteen
copies of the `n x n` factor. Model batch dimensions retain their own factors,
including singleton dimensions that broadcast against query batches. The result
is reshaped back before posterior covariances are formed. Regression tests
compare values, every input gradient and Hessian-vector products with the
original broadcast solve on CPU and CUDA.

Explicit jitter in these functions is fixed configuration. Sampling can use
`tensor_psd_safe_cholesky`, which first attempts the original matrix and adds
increasing diagonal jitter only to failed batch members. This decision reads
the small factorization-status tensor. Factors and arithmetic remain on the
original device, and the selected jitter branch retains its derivative graph.
The function does not modify its input matrix.

Dense tensor primitives cover RBF, Matérn, rational quadratic, periodic,
cosine, linear and polynomial kernels. Actual constrained tensor parameters
enter the calculation directly. The trailing `parameters` dictionary carries
`alpha`, `period_length`, `variance`, or `offset` as appropriate; polynomial
`power` is a fixed finite Python number. Unknown parameters are rejected.
Module wrappers handle active dimensions, parameter transformations and kernel
composition separately.

The Matérn-3/2 and Matérn-5/2 kernels use their exact Taylor terms at coincident
points to retain the existing second and fourth input derivatives,
respectively. Matérn-1/2 is not differentiable at coincidence; its zero-distance
derivative follows the usual zero-gradient convention.

Cosine uses the identity

```
exp([[0, -pi² s], [1, 0]])[0, 0] = cos(pi sqrt(s)),
```

where `s` is squared distance after period scaling. The entire matrix
exponential avoids the artificial singularity from differentiating `sqrt(0)`.
Tests verify its first six derivatives at zero. This costs more than a direct
cosine evaluation and is not a performance claim for this kernel. Pairwise
distance arithmetic also uses memory proportional to the coordinate dimension.

The entire-function branch is now restricted to squared distances below `1e-4`.
Away from zero, a separately masked square root and ordinary cosine preserve
large-phase float32 accuracy. The previous all-matrix-exponential calculation
had errors up to `2.92e-5` in a three-point example on both CPU and CUDA.
The new regression fails against the frozen `52c03c6` binary and passes with
the repair, including first/sixth derivatives at zero and mixed Hessian-vector
products away from zero. The focused native tensor suites pass 132 cases.

Tests exercise CPU and the available NVIDIA RTX 4050 CUDA device, float32 and
float64, arbitrary broadcast batches, noncontiguous inputs, scalar and ARD
parameters, numerical gradient checks, second derivatives, and likelihood and
posterior derivatives with respect to all supplied tensor inputs.

## Reference issues and solutions

The installed GPyTorch 1.15.2 periodic and cosine forward paths call
`torch.cdist`. With Torch 2.13.0, differentiating their query gradients again
raises `NotImplementedError: the derivative for '_cdist_backward' is not
implemented`. The port uses differentiable pairwise arithmetic for periodic
and the matrix identity above for cosine. Tests first compare actual reference
values and first derivatives, then replace only the reference distance
primitive to validate the same formula's second derivatives.

GPyTorch's polynomial offset also misses the feature batch dimension when
`last_dim_is_batch=True` and the model has a batch shape. For example, a model
batch of two and three features causes both full and diagonal evaluation to
raise a broadcasting error. A concrete upstream repair is to align the offset
once and use that value in both paths:

```python
offset = self.offset.view(*self.batch_shape, 1, 1)
if last_dim_is_batch:
    offset = offset.unsqueeze(-3)
    x1 = x1.transpose(-1, -2).unsqueeze(-1)
    x2 = x2.transpose(-1, -2).unsqueeze(-1)
if diag:
    return ((x1 * x2).sum(-1) + offset.squeeze(-1)).pow(self.power)
return (x1 @ x2.transpose(-2, -1) + offset).pow(self.power)
```

The tensor implementation aligns the offset correctly. Its regression oracle
stacks the actual reference kernel evaluated separately on each feature; it
does not skip the affected configurations.

Upstream repairs are filed as
[GPyTorch #2814](https://github.com/cornellius-gp/gpytorch/pull/2814) for polynomial
feature offsets and
[GPyTorch #2816](https://github.com/cornellius-gp/gpytorch/pull/2816) for periodic
and cosine higher derivatives. Their upstream regressions fail on unchanged
main and pass with the proposed fixes.
