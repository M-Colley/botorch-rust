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
context decomposition already defines grouped feature axes. Orthogonal additive
Tensor kernels and models are documented in [additive-components.md](additive-components.md).

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
previously reported duplicate shared-prior counting defect. The migration repair
is submitted in [BoTorch #3493](https://github.com/meta-pytorch/botorch/pull/3493).

## Latent context embedding multitask GP

`LCEMGP(..., backend="torch")` uses the same device-resident exact Gaussian
solver as `MultiTaskGP`, with a separately learned RBF covariance on context
embeddings. CUDA or differentiable Tensor arguments select it automatically.
Learned embedding layers, per-task means/noise, continuous covariance parameters,
and the context lengthscales are actual persistent `nn.Parameter` objects.
Unlike the native CPU LCEM adapter, the Tensor implementation also supports fixed
observation noise, custom mean/data covariance/Gaussian likelihood modules, input
transforms, conditioning, and fantasies through the common multitask protocol.

The context RBF has one lengthscale for every learned **and** supplied embedding
dimension. Supplied Tensor embedding features retain their data gradients.
Each categorical feature is independently mapped to contiguous indices, as are
noncontiguous task labels. These address the previously documented source
dimension and label-indexing defects. For an unobserved context, evaluation keeps
its embedding and mean parameters; it skips the source `MultiTaskGP.eval`
IndexKernel-factor update, since that factor does not exist in LCEM's embedding
covariance. Training remains unbatched long format, with any task-column
position and freely batched prediction inputs.

`task_covar_module(task_idcs)` accepts mapped indices in `[0, num_tasks)`; model
prediction inputs carry the original task labels. The task kernel can be replaced
by a port descriptor or a Torch module. Fixed embedding/category arrays are
registered buffers, and `device` follows the training Tensor after migration.
The optional `seed` controls the private learned-embedding initialization stream.
`ProvidedEmbeddingLCEMGP`, a project extension outside the official main package,
retains its native CPU adapter and is not included in this Tensor checkpoint.

`tests/test_torch_lcem.py` adds 21 CPU/CUDA cases: official full covariance,
posterior/query/parameter gradients, and MLL values/gradients for all task-column
positions and both learned/fixed noise; independent dense conditioning and
first/second derivatives through training inputs/targets/noise, supplied and
learned embeddings, and queries; noncontiguous labels; transformed float32
models; unobserved contexts; Adam and bounded SciPy fitting; conditioning; task
kernel replacement; and state reload. The reference posterior parameter-gradient
oracle enables BoTorch's `propagate_grads` setting. The runtime never calls the
reference model.

`LCEMGP.construct_inputs` now accepts supervised and multitask datasets, task
selection, categorical/fixed context features and embedding sizes. It delegates
the shared dataset/task protocol to `MultiTaskGP.construct_inputs`, preserves
original Tensor identities and forwards optional context arguments unchanged.
Constructed configurations work with native list/NumPy/Tensor predictions and
the explicit Torch backend. The dataset helper and existing Tensor LCEM suite
pass 37 combined cases, including CPU/CUDA training/embedding/query first and
higher derivatives. Native LCEM conditioning/fantasies still require selecting
the existing Torch backend; this helper does not add those native operations.
