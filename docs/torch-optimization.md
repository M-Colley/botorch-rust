# Device acquisition optimization

The Tensor optimization driver supports candidate `batch_limit` and
`max_optimization_problem_aggregation_size` as actual limits on groups of
restarts. Bounds and per-restart fixed values are sliced consistently;
`fixed_features={i: None}` retains the original frozen values. Torch
optimizers retain all candidate algebra on the device. SciPy transfers only
its explicit candidate/value/gradient interface, as before. Every group
receives a share of the remaining timeout budget.

Automatic box initialization and best-restart selection support model
batches. Boltzmann restarts are drawn independently per model batch, and
the final candidate is selected independently for each batch. Nonnegative
restart selection likewise applies its positive-quota/threshold law per
batch. Public `initialize_q_batch`, `initialize_q_batch_nonneg` and
`initialize_q_batch_topn` retain Tensor values/graphs on their device.
The public `gen_batch_initial_conditions` also accepts a Tensor raw-sample
generator and `fixed_X_fantasies`, without host inference conversions.

`optimize_acqf(..., acq_function_sequence=[...], sequential=True)` runs one
actual device acquisition per step. Each acquisition retains its own
original pending set, appends the previous selected candidates during its
step, and has its original state restored on success or failure. Joint
one-shot acquisitions and inter-point constraints remain excluded from
sequential mode, as in BoTorch.

The 88-case checkpoint combines new CPU/CUDA configuration tests, existing
Tensor optimizer/initializer tests and native optimization smoke tests.
It checks actual Torch/SciPy chunk sizes and convergence, fixed-value
mapping, model-batched initialization and best selection, sequence cleanup
on errors, public generators/fantasies, seeded reference laws and forbidden
host conversions. Targeted mypy and Ruff checks pass.

The pinned official reference is BoTorch
`b0dd01be61d517c387e234903a6e2cc38c057ea2`. Batched Boltzmann parity uses the
category-axis standardization correction already proposed in
[PR 3390](https://github.com/meta-pytorch/botorch/pull/3390), matching the
existing native port. The unmodified reference standardizes across batches
and can sample nearly uniformly despite unequal values within a batch.
Scalar laws continue to match the unmodified reference and RNG.

Constrained automatic sampling, sample-around-best, specialized KG starting
point heuristics, mixed/discrete/cyclic optimizers and further public
initialization helpers are subsequent checkpoints. Forced parallel SciPy
mode remains unsupported; this driver uses serial SciPy calls for each
restart group. No optimization speedup is claimed here.
