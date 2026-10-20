# Native MLL parameter protocol

`ExactMarginalLogLikelihood`, `SumMarginalLogLikelihood` and `VariationalELBO`
expose `parameters(recurse=True)` and
`named_parameters(prefix="", recurse=True, remove_duplicate=True)`. Ordinary
CPU Torch exact and variational models return their persistent parameter
leaves, so standard Torch optimizers can train native losses:

```python
mll = ExactMarginalLogLikelihood(model.likelihood, model)
optimizer = torch.optim.Adam(mll.parameters(), lr=0.0003)
optimizer.zero_grad()
loss = -mll().sum()
loss.backward()
optimizer.step()
```

Default traversal registers likelihood parameters before model parameters and
omits their aliases under the model. For a single exact GP, typical names are
`likelihood.noise_covar.raw_noise`, `model.mean_module.raw_constant`, and
`model.covar_module.raw_lengthscale`. A summed likelihood uses
`likelihood.likelihoods.<i>.*` and `model.models.<i>.*`. Repeated child models
and genuinely shared leaves appear once. Setting `remove_duplicate=False`
exposes every registration path, including likelihood aliases and the summed
likelihood's `mlls.<i>.*` paths. `prefix` prepends a namespace. MLLs have no
direct parameters, so `recurse=False` returns an empty iterator.

`zero_grad(set_to_none=True)` clears managed Torch gradients. Passing `False`
zeros existing gradient tensors in place. `requires_grad_(flag)` changes the
persistent leaves' flags and returns the MLL. Native model evaluation and
optimizers retain their explicit-gradient and parameter-bound behavior.

Lists and NumPy models expose the same native parameter descriptors and naming
rules. They have no managed Torch gradients, so `zero_grad` is a no-op and
`requires_grad_` raises an explicit unsupported error. Torch-backed families
that do not yet expose persistent leaves have the same flag boundary.

This parameter protocol does not make native MLLs `torch.nn.Module` instances
or imply support for arbitrary module registration, hooks, device migration,
or state dictionaries. Kernel/GP evaluation and exact parameter derivatives
remain in Rust. Tests compare registration paths and aliases with GPyTorch,
run Adam and SGD through native exact and summed losses, and check freezing,
gradient clearing, non-Torch descriptors, and native optimizer-name mapping.
