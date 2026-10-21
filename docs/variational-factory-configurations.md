# Variational constructor configurations

`SingleTaskVariationalGP` and the generic default
`ApproximateGPyTorchModel(model=None, ...)` accept a supplied variational
distribution whose batch dimensions broadcast with the model's augmented
batch shape. Its parameters retain their original shape and identity; they
are not expanded into independently trainable copies. This includes a single
`q(u)` shared across model batches or output tasks, a singleton model batch
with distinct task distributions, and additional distribution batches that
broadcast with the kernel. Port Cholesky/MeanField/Delta specifications and
actual caller-supplied GPyTorch distribution modules follow the same rule.
The inducing event size must still match the number of inducing points.

ELBO targets broadcast to the resulting `q(f)` batch and event dimensions,
as in the Gaussian likelihood protocol. Independent multitask KL reduction
retains the reference strategy's actual reduction over its variational batch
axis, including singleton/shared distributions. It does not invent a second
copy of a shared distribution or multiply its scalar KL by the output count.

The regression module `test_torch_variational_shared_distribution.py`
checks scalar and multiple outputs, shared and singleton model/task batches,
additional distribution batches, source values and all first derivatives,
and first/second derivatives against an independent dense Gaussian formula.
It also checks posterior snapshots after later distribution parameter edits.
CPU and CUDA run the same tests; reference-dependent cases require the
optional reference packages, while port-specification dense-law cases do not.

Constructor compatibility is separate from implementing every strategy in
GPyTorch. Direct `GridInterpolationVariationalStrategy`,
`NNVariationalStrategy`, and `OrthogonallyDecoupledVariationalStrategy` are
not interchangeable with the constructor call made by the pinned BoTorch
wrapper: they reject its `inducing_points` or
`learn_inducing_locations` keyword arguments. A caller can instead supply a
compatible factory or an already-constructed approximate model. Working
source factories, subclasses, `LargeBatchVariationalStrategy`, and targetless
outcome-transform attachment are additional audited configurations; they
must preserve the supplied strategy's actual behavior rather than replacing
a subclass with a canonical strategy by name.
