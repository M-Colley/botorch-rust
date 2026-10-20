# Infinite-width BNN diagonal conventions and numerical stability

Audited against official BoTorch revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`, Torch 2.13, and the native release
checkpoint `4a8b8bd`. This concerns the ReLU infinite-width neural-network
GP covariance implemented by `InfiniteWidthBNNKernel`; no separate neural
tangent-kernel recurrence is involved.

## Deterministic public diagonal difference

The official kernel intentionally clips correlations to
`[-1 + acos_eps, 1 - acos_eps]` before evaluating `acos` in its full-matrix
path. Its `diag=True` path instead uses the exact self-variance recurrence,
without clipping. These two public evaluations therefore disagree even at
ordinary finite inputs, independently of floating-point norm accuracy.
The native kernel reproduces both conventions, including their derivatives.

```python
import torch
from botorch.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel

x = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
kernel = InfiniteWidthBNNKernel(depth=1, acos_eps=1e-7).double()
kernel.weight_var = 1.0
kernel.bias_var = 1.0
diagonal = kernel(x, diag=True).squeeze()
full_self = kernel(x).to_dense().squeeze()
print(diagonal.item(), full_self.item())
print(torch.autograd.grad(diagonal, x, retain_graph=True)[0].item())
print(torch.autograd.grad(full_self, x)[0].item())
```

Observed official and native values:

| Depth | `diag=True` | Full self covariance | Query derivative: diagonal / full |
|---|---:|---:|---:|
| 1 | 1.9999999999999996 | 1.99999990000949 | 0.9999999999999998 / 0.99999990000949 |
| 3 | 1.9999999999999996 | 1.99999990000949 | 0.2499999999999999 / 0.24999997500237245 |

The value difference is `9.99905096499e-8` in this example. Depth zero has
no arc-cosine update and the two paths agree. The deliberate `acos_eps`
stabilization is not itself evidence of an implementation defect. The
important public contract issue is that the two routes expose different
self-covariances, so substituting one for the other changes an objective.

## Consequence for scalar analytic acquisitions

An analytic acquisition such as LogEI consumes a model's marginal mean and
variance. If that model computes its variance using the lazy kernel
diagonal, a native acquisition or parameter-adjoint tape must use that same
diagonal convention and its derivatives. Taking the diagonal of an eagerly
materialized joint covariance is not an equivalent replacement here.

For an SVGP, both routes can agree on the inducing-point correction while
differing in the prior query diagonal. Thus matching ordinary posterior
means and variances does not establish that a separate full-covariance
acquisition tape uses the same Gaussian moments. An order-`1e-7` variance
change can produce order-`1e-8` LogEI differences, and the query and kernel
parameter pullbacks must follow the chosen variance path as well. Tight
equivalence tests should not be loosened to hide this semantic difference.

This document validates the kernel behavior and proposed repairs below. It
does not, by itself, claim validation of a particular acquisition-tape fix.

## Closed self-variance recurrences

Write the positive weight and bias variances as `w` and `b`, and let
`r = sum(x * x) / d`. The exact self variance is

\[
v_0 = wr+b,\qquad v_{\ell+1}=\tfrac{w}{2}v_\ell+b.
\]

This is the existing `diag=True` result after `depth` updates. It avoids
`acos` entirely, retains both appearances of the self input, and is directly
differentiable with respect to `x`, `w`, and `b`. For example,
`dv_depth/dx = (w/2)**depth * (2*w/d)*x`.

The existing clipped full self covariance also has a closed form. For
`0 < eps < 1`, nonnegative `w,b`, and positive finite variances, define

\[
\rho=1-\varepsilon,\quad
a_\varepsilon=
\frac{\sqrt{\varepsilon(2-\varepsilon)}
 +(\pi-\arccos\rho)\rho}{\pi}.
\]

For positive depth `L`, the old full self result is
`b + (w/2) * a_eps * v_(L-1)`. The self correlation remains in the upper
clipped region at each layer: positive bias ensures that the previous
clipping deficit cannot exceed the next variance's clipping threshold.
For depth zero the result is `v_0`. This formula matches the official full
diagonal up to floating-point rounding and can be differentiated through
the variance recurrence without subtracting singular `acos` derivatives.

Two viable upstream choices are therefore:

1. Keep the existing exact marginal recurrence and make full self entries
   use it too, while retaining correlation clipping for other entries.
2. Preserve the existing clipped full-matrix convention and make the lazy
   diagonal use its closed form, with the corresponding gradients.

Either choice changes one existing public route; it should be an explicit
compatibility decision. Merely copying diagonal values under `no_grad`, or
detaching them, would give incorrect input or hyperparameter derivatives.

## Proposed differentiable full-matrix repair

For `K(X, X)`, the following reference-only prototype implements the first
choice. It removes self entries from the `acos` calculation before restoring
their differentiable closed recurrence. Using a product of square roots
also avoids overflow in the intermediate variance product.

```python
def repaired_same_inputs(kernel, x):
    w, b = kernel.weight_var, kernel.bias_var
    cross = w * (x @ x.transpose(-2, -1) / x.shape[-1]) + b
    variance = kernel._initialize_var(x)
    self_mask = torch.eye(x.shape[-2], dtype=torch.bool, device=x.device)
    for layer in range(kernel.depth):
        if layer:
            variance = w * variance / 2 + b
        root = variance.sqrt()
        scale = root @ root.transpose(-2, -1)
        fraction = (cross / scale).clamp(
            -1 + kernel.acos_eps, 1 - kernel.acos_eps
        )
        fraction = torch.where(self_mask, torch.zeros_like(fraction), fraction)
        angle = fraction.acos()
        other_entries = (
            w / (2 * torch.pi) * scale
            * (angle.sin() + (torch.pi - angle) * fraction) + b
        )
        self_entries = torch.diag_embed((w * variance / 2 + b).squeeze(-1))
        cross = torch.where(self_mask, self_entries, other_entries)
    return cross
```

Independent checks compared this prototype's self values and gradients
with respect to inputs, raw weight variance, and raw bias variance to the
official lazy diagonal for depths 1, 3, and 7, with both `acos_eps=1e-7`
and `acos_eps=0`: all six configurations matched within `1e-13`. The
clipped closed form was also compared with the official full diagonal in
18 positive-variance configurations.

Two further official numerical edge cases were reproduced separately:

- With `acos_eps=0`, `x=[[1]]`, and unit weight/bias variances, the full
  value is finite but its query derivative is NaN. The symmetric self
  recurrence avoids the cancelling singular derivatives.
- With `x=[[1e100]]`, positive default epsilon, and unit variances, the
  full route returns infinity because it forms `v * v` before taking its
  square root. The lazy diagonal is finite, approximately `5e199`. The
  prototype returns the same finite value and query derivative
  approximately `1e100`.

The prototype specifically addresses self entries of a same-input Gram
matrix. A complete upstream implementation must preserve active dimensions
and batch broadcasting, and define paired `diag=True` semantics for
different input sets. Duplicate off-diagonal inputs with zero epsilon and
zero-variance parameter limits require their own stable treatment. No
production kernel source is changed by this audit.
