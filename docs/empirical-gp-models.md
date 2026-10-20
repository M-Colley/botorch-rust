# Empirical learning-curve models

Reference: official BoTorch commit `b0dd01be6` in `.reference/botorch`.
The separate checkout on the Desktop is an audit fork and is not the authority
for this parity work.

The first model milestone implements these public classes:

- `BaseAugmentedEmpiricalKernel`
- `EmpiricalOneDimensionalMean`, `EmpiricalOneDimensionalKernel`,
  `EmpiricalOneDimensionalGP`
- `MultiOutputEmpiricalOneDimensionalMean`,
  `MultiOutputEmpiricalOneDimensionalKernel`,
  `MultiOutputEmpiricalOneDimensionalGP`

Interpolation, basis decomposition, empirical covariance, Cholesky inference,
MLL evaluation, parameter derivatives, input derivatives, sampling and both
L-BFGS-B and Adam fitting run in Rust. Python arranges array shapes and module
metadata. Torch and NumPy are optional array containers; neither implements
the model's numerical inference. Native computation currently uses CPU float64;
Torch output arrays preserve the caller's dtype and device.

Historical curves have shape `curves x progression x outputs`, with at least
two distinct progression coordinates. Means interpolate the historical sample
mean. Covariances interpolate centered curves and normalize by the **original**
curve count minus the correction, including when a compressed basis is used.
Independent-output models share ARD weights across outputs and learn separate
noise values. Their native fitter sums the per-output normalized likelihoods
and accumulates derivatives of shared parameters. Correlated-output models
condition jointly on point-major interleaved observations and expose the full
cross-output covariance. Both support fixed observation variances and an
additive native base covariance on progression.

Validation includes official-source mean/kernel parity with and without
compression and ARD, unequal cross-input diagonals, query derivatives, MLL
values and raw gradients, independent and correlated posterior moments and
joint covariances, fixed/inferred noise, list/NumPy/Torch arrays, both native
fitters, immutable posterior snapshots and additive-kernel parameter binding.
Native tests independently finite-difference all shared raw parameters and the
entire posterior mean/covariance Jacobian.

## Current boundaries

Training data must be unbatched, although posterior queries may have batch
dimensions. Historical data and their decomposition are fixed. The models
require corresponding empirical mean/covariance modules; arbitrary custom
means, input/outcome transforms, batched additive kernels and explicit
tensor-valued posterior noise are rejected. Additive base-kernel parameters
are shared across independent outputs. Model calls use the port's existing
`mll()` interface, rather than accepting a Python GPyTorch distribution as an
MLL argument. Conditioning/fantasizing these models is not implemented yet.
Multi-output model component descriptors directly expose their empirical
kernel/mean, rather than the upstream internal vectorization wrappers.

The heterogeneous-domain multi-task model and EM prior/model/coordinate-ascent
families are the next milestones. These seven exports do not establish full
empirical-family completion.

## Upstream standalone kernel shape bug

At official revision `b0dd01be6`, the ordinary GPyTorch call path assumes an
`n1 x n2` result for both empirical kernels because their `batch_shape` is
empty. With two historical outputs:

```python
import torch
from botorch.models.empirical_gps.empirical_1d_gp import EmpiricalOneDimensionalKernel
from botorch.models.empirical_gps.multioutput_empirical_1d_gp import MultiOutputEmpiricalOneDimensionalKernel

X = torch.linspace(0, 1, 4, dtype=torch.double).unsqueeze(-1)
Y = torch.randn(6, 4, 2, dtype=torch.double)
Q = X[:3]
EmpiricalOneDimensionalKernel(X, Y)(Q, Q).to_dense()
# RuntimeError: expected [3, 3], got [2, 3, 3]
MultiOutputEmpiricalOneDimensionalKernel(X, Y)(Q, Q).to_dense()
# RuntimeError: expected [3, 3], got [6, 6]
```

Their direct `forward(Q, Q)` methods return the intended covariance. The
native public call returns the intended result and tests compare to those
official forward methods. A proposed upstream fix is to give the independent
kernel the output batch shape expected by GPyTorch and to implement
`num_outputs_per_input` for the correlated kernel (returning the historical
output count). Preserve the model's existing vectorization behavior and add
standalone `kernel(Q, Z).to_dense()` and diagonal regressions when applying
the fix. Neither reference checkout was changed.
