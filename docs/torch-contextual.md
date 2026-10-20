# Contextual covariance on the Torch backend

`SACGP` and `LCEAGP` accept `backend="torch"`. CUDA inputs, or training
inputs/targets/noise requiring gradients, select that backend automatically.
`SACKernel` and `LCEAKernel` also accept the backend argument; a CUDA `device`
selects it automatically. A native CPU kernel descriptor can be supplied to a
Torch `SingleTaskGP`, which converts its parameters once at construction.

Covariance, likelihood, exact Gaussian conditioning, priors, and optimization
retain Tensor dtype, device, and first and higher derivative graphs. RBF
evaluation uses the Rust Torch dispatcher; context selection, embedding lookup,
and covariance contraction use port-owned Tensor operations. Runtime numerical
evaluation does not flatten to the native CPU/f64 model or call BoTorch models.

SAC shares one ARD spatial kernel across context scales and counts its prior
once, including when the same child appears under multiple module names. LCEA
supports learned categorical embeddings, multiple categorical features, optional
fixed embedding features, fixed population weights, and batched kernel
parameters. Batched learned embedding tables are independent Tensor parameters.
As with `nn.Embedding(max_norm=1)`, an oversized embedding row is projected under
`no_grad` before lookup; gradients are with respect to the resulting parameter,
not a differentiable norm-projection map. Context indices and fixed features are
registered buffers, so `.to(device=..., dtype=...)` migrates them correctly.

SAC models support ordinary model batches and independent output augmentation.
LCEA models retain the official unbatched, one-output training contract; LCEA
kernels can be batched independently. Both kernels implement aligned diagonal
evaluation and unequal input grids. `last_dim_is_batch=True` is rejected: the
context decomposition already defines grouped feature axes. LCEM and orthogonal
additive Tensor models are separate work and are not claimed by this checkpoint.

`tests/test_torch_contextual.py` covers CPU/CUDA float32 and float64, official
kernel and posterior values and first derivatives, independent context-pair
formulas and full parameter/input Hessian-vector products, repeated training
data backward, actual Adam updates, conditioning, state reload, shared parameter
identity, default factory selection, and embedding projection. Reference source
is official BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2`. For batched
learned LCEA, each corrected unbatched source kernel is an independent oracle;
the source batched multiple-feature lookup is already a documented upstream
defect. Decomposition order follows the explicitly listed feature order rather
than the source helper's sorted permutation, preserving cross-context alignment.

## Verified source device-migration defect

With CUDA available, the following official-source call fails:

```python
import torch
from botorch.models.kernels.contextual_sac import SACKernel

k = SACKernel({"a": [0, 1], "b": [2, 3]}, torch.Size()).double().cuda()
x = torch.rand(3, 4, device="cuda", dtype=torch.double)
k(x).to_dense()  # index is on cpu, different from other tensors on cuda
```

`_indexers` contains tensors in an ordinary dictionary, so module migration does
not move them. Constructing with `device="cuda"` is a workaround. The proposed
upstream repair registers each indexer as a buffer (with stable internal buffer
names), then resolves those buffers during evaluation. The port does this and
tests CPU construction followed by CUDA migration. This is distinct from the
previously reported duplicate shared-prior counting defect.
