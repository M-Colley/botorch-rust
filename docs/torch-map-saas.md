# MAP SAAS on the Torch backend

`get_map_saas_model`, `get_fitted_map_saas_model`,
`AdditiveMapSaasSingleTaskGP` and `EnsembleMapSaasSingleTaskGP` accept
`backend="torch"`. CUDA tensors, training tensors requiring gradients, or
Torch transform modules select that backend automatically. Explicit
`backend="native"` retains the existing CPU implementation.

The device implementation uses the same sparse axis-aligned prior:
`tau / lengthscale**2 ~ HalfCauchy(1)`. A learned global `tau` has a
`HalfCauchy(0.1)` prior. Both lengthscales and learned `tau` are actual
`nn.Parameter` leaves with the declared logarithmic or interval constraint.
Prior evaluation sums parameter event axes while preserving ordinary model
and output batches. These coupled terms are included by the exact MLL,
which supports standard Torch optimizers and the port's SciPy/Torch fitters.
SciPy transfers optimizer scalars and parameter vectors at its explicit
boundary; kernels, training covariance matrices, likelihoods and predictions
remain on the training device.

Conversion of an existing port RBF/Matérn kernel preserves its SAAS metadata,
raw shrinkage value and coupled prior. It must not reduce that kernel to an
ordinary stationary covariance with no shrinkage term. The public
`add_saas_prior` and `SaasPriorHelper` closures also operate on the port-owned
Torch RBF/Matérn modules without flattening their parameters. Setters retain
leaf identity; prior draws use the same constraint-aware setters. Fixed
Tensor-valued `tau` retains its external first and higher derivatives through
the prior. Conditional predictions at fixed lengthscales correctly have zero
derivative with respect to the prior-only shrinkage parameter.

Additive models have independently sampled fixed sparsity levels and learned
component lengthscales/outputscales. Their random `tau` draws use a Torch
distribution on the requested device. Cross-device seeds need not produce
the same random values; parity comparisons explicitly match the sampled
levels. `get_additive_map_saas_covar_module` honors explicit dtype/device
requests and can return a port-owned Tensor module with `backend="torch"`.

Ensemble models repeat training tensors with differentiable device operations
and return `TorchGaussianMixturePosterior`. The final model-batch axis indexes
the sparsity levels. Query batches receive a separate singleton ensemble axis,
including explicit prediction noise. Both fitted parameters and fixed sparsity
buffers survive state-dict round trips. Ensemble outcome transforms must
preserve Gaussian component laws; nonlinear transforms raise explicitly.

The focused CPU/CUDA tests compare scalar, batched and multioutput MLL values,
every data/parameter first derivative, and mixed Hessian-vector products with
official BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2`. They also compare
posterior moments, data/query/parameter gradients, real Adam updates and
SciPy/Torch fitting, both constraint forms, external fixed-`tau` derivatives,
prior sampling, conversion and state restoration. GPyTorch's distinct-point
Matérn posterior path raises on second derivatives of `_cdist_backward` in
the tested runtime, so its independent first-gradient differences provide
the posterior Hessian-vector oracle. The port's own complete second graph
is evaluated directly.

## Reference additive covariance factory dtype/device

The pinned reference helper applies explicit dtype/device arguments to each
Matérn base kernel, then creates its enclosing ScaleKernel with default CPU
parameters. With default float32 and a requested double CUDA module:

```python
import torch
from botorch.models.map_saas import get_additive_map_saas_covar_module

kernel = get_additive_map_saas_covar_module(
    2, num_taus=1, dtype=torch.double, device=torch.device("cuda")
)
print([(name, p.dtype, p.device) for name, p in kernel.named_parameters()])
# kernels.0.raw_outputscale: float32, CPU
# kernels.0.base_kernel.raw_lengthscale: float64, CUDA
```

This also leaves the outputscale float32 for a requested double CPU kernel.
The full reference GP constructor eventually calls `.to(train_X)`, so the
issue is visible at the public standalone covariance factory boundary. A
minimal upstream repair moves the ScaleKernel to the requested dtype/device
after constructing it, or moves the completed AdditiveKernel before returning.
The port returns a consistent module immediately. Its independent covariance
oracle explicitly calls `.to(dtype=..., device=...)` on the reference helper
result before matching parameter leaves; fixed-`tau` randomness does not affect
these covariance values at fixed lengthscales.
