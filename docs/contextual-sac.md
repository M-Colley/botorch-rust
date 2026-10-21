# Structural additive contextual Gaussian processes

`botorch_rs.models.kernels.contextual_sac.SACKernel` evaluates the sum of
scaled RBF kernels on equal-sized context decompositions. It stores one
shared ARD lengthscale vector and one outputscale per context. The native
kernel evaluator accumulates every component's lengthscale derivatives into
the same parameter offsets. It also accumulates input derivatives correctly
when contexts overlap or repeat an input column.

The shared RBF uses BoTorch's dimension-scaled lengthscale prior and
constraint. Each outputscale has `GammaPrior(2, 15)` and a positive
constraint. Model batches and multiple independent outputs have independent
parameter sets, matching the corresponding BoTorch batch dimensions.

`botorch_rs.models.contextual.SACGP` uses this covariance in the native
`SingleTaskGP` implementation. Outcomes are standardized by default.
`train_Yvar=None` selects inferred Gaussian noise. The class accepts lists,
NumPy arrays, and torch tensors without calling a Python GP implementation.
`construct_inputs` returns arguments that can directly construct a model,
including `train_Yvar=None` for a dataset without observed noise.

## Official BoTorch audit

The reference revision is official BoTorch **b0dd01be6**. This is distinct
from the local corrected fork at c98dd9452.

### Shared lengthscale prior is counted M+1 times

For M contexts, official `SACKernel` registers `base_kernel` once directly
and again through every context's `ScaleKernel`. PyTorch's parameter
enumeration deduplicates that shared object, but GPyTorch's prior traversal
does not. With two contexts, this reproduces the problem:

```python
import torch
from botorch.models.kernels.contextual_sac import SACKernel

k = SACKernel({"a": [0, 1], "b": [2, 3]}, torch.Size())
print([name for name, _ in k.named_parameters()])
# One base_kernel.raw_lengthscale, plus two raw_outputscale parameters.
print([name for name, *_ in k.named_priors()
       if name.endswith("lengthscale_prior")])
# base_kernel.lengthscale_prior
# kernel_dict.a.base_kernel.lengthscale_prior
# kernel_dict.b.base_kernel.lengthscale_prior
```

Consequently, the official normalized exact MLL exceeds the corrected MLL
by `M * lengthscale_prior.log_prob(lengthscale).sum() / n`. This also changes
the lengthscale gradient and fitted hyperparameters. Kernel matrices and
posteriors at identical parameters are unaffected. The Rust implementation
counts the prior once, consistently with the single shared parameter.
Reference tests explicitly subtract the M duplicate terms before comparing
MLL values and gradients; unmodified official fitting objectives differ.

An upstream solution is to add a visited-module set to GPyTorch's recursive
prior enumeration, as PyTorch does for shared parameters. Another is to
register the base kernel only at the top-level SAC module and have the
context evaluators refer to it without registering aliases. Prior sampling
should deduplicate the same shared parameter for the same reason.

### Dataset construction omits a required noise argument

Official `SACGP.construct_inputs` inherits `SingleTaskGP.construct_inputs`,
which omits `train_Yvar` when a dataset has no observed noise. However,
`SACGP.__init__` requires the `train_Yvar` argument. Thus
`SACGP(**SACGP.construct_inputs(dataset, decomposition))` raises `TypeError`
for such a dataset. The fix is to include `train_Yvar=training_data.Yvar`
unconditionally. This port applies that fix.

## Scope

SAC uses native exact inference, model fitting, posteriors, and analytic
acquisition input derivatives. The tests cover explicit context matrices,
shared parameter aliases, MLL values and gradients after correcting the
reference prior duplication, batches, multiple outputs, inferred and fixed
noise, fitting, cloning, and conditioning. This implementation does not
provide the separate learned-context embedding kernel `LCEAKernel` or
`LCEAGP`.
