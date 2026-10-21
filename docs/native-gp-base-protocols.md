# Native GP base protocols

The four public classes in `botorch_rs.models.gpytorch` expose shared behavior
for native Rust GP models. The interface audit and reference tests use official
BoTorch revision `b0dd01be6` from the repository's `.reference/botorch` checkout.
They do not turn an arbitrary Python GPyTorch `ExactGP` into a native model.

| Base | Shared behavior | Concrete native subclasses |
| --- | --- | --- |
| `GPyTorchModel` | Tensor rank and dtype validation, external batch and output metadata | Exact, approximate, higher-order, and multitask wrappers |
| `BatchedMultiOutputGPyTorchModel` | Independent-output batch dimensions, training tensor layout, native posterior validation, output selection, observation-noise and posterior-transform dispatch | `SingleTaskGP` and its subclasses; `HigherOrderGP` retains its specialized posterior implementation |
| `ModelListGPyTorchModel` | Native joint posterior construction, output grouping, nonlinear `PosteriorList` selection, recursive evaluation mode, distribution batch expansion | `ModelListGP`, which also inherits the public `ModelList` grouping and transformation protocol |
| `MultiTaskGPyTorchModel` | Long-format task-column validation, task selection, native posterior construction, training-target and fixed-noise extraction | `MultiTaskGP`, `LCEMGP`, and their subclasses |

`get_batch_dimensions`, `_set_dimensions`, and `_transform_tensor_args` agree
with the official independent-output base for single and multiple outputs and
batched and unbatched data. Their list, NumPy, and torch forms use structural
array operations. Posterior inference, covariance construction, and Gaussian
sampling execute in Rust. `_broadcast_mvns` accepts native Gaussian distribution
views; expanding their moments keeps the resulting native sampler usable.

Native training dimensions cannot be changed in place by `_set_dimensions`.
The GP-list posterior requires a common model batch shape, although the public
base's `batch_shape` property computes and warns about a broadcastable shape as
BoTorch does. Tensor-valued `observation_noise`, arbitrary Python likelihood
callbacks, and GPyTorch's private mutable training-data restoration hooks are
not implemented by these native protocols. Concrete model lifecycle methods
continue to define supported conditioning, fantasies, copying, and fitting.

The refactor also fixes a selection bug in nonlinear `ModelListGP` posteriors.
For a two-output first model and one-output second model, selecting `[2, 1, 0]`
previously iterated the physical model list and returned the first model's
outputs before the second model's output. The shared implementation follows the
public grouping map and returns the second model first, then outputs `[1, 0]`
from the first model. A regression checks means and variances for all three
array forms.

Validation before integration: 331 regression tests passed across contextual
SAC/LCEA, LCEM, multitask, higher-order, expectation transforms, Gaussian
posteriors, TorchPosterior, and multi-output models. The focused base suite
adds direct comparisons with the official dimension/layout helpers and
checks shared posterior dispatch, output grouping, inheritance, modes, fixed
noise extraction, and native Gaussian batch expansion.
