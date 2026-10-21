# Device pruning and constraint utilities

Public scalar and multi-objective baseline pruning now evaluate the posterior,
draw samples, apply objectives and constraints, marginalize a supplied model
batch dimension, and select points on device. Selection is discrete; returned
points retain their input Tensor graph. Both interfaces retain the source's
restriction to unbatched input point sets. No-pruning noisy scalar acquisitions
accept batched baselines, and ignore the pruning-only `marginalize_dim` option
as the source does. Infeasible-cost sampling respects every baseline batch's
bounding box.

The public Pareto mask uses device comparisons, with a bounded-memory loop for
large problems. Maximization/minimization, duplicate retention, empty batches
and the established correction for partial-NaN rows are covered. Capped scalar
pruning retains actual point labels when sorting winner counts, the correction
already proposed in [BoTorch #3391](https://github.com/meta-pytorch/botorch/pull/3391).

Boolean feasibility marginalization stays on device. Smoothed feasibility keeps
the complete constraint and Tensor-temperature graphs for ordinary, fat-tailed
and directly probability-valued constraints. Scalar temperatures retain the
reference's initial default-dtype rounding.

The focused scalar/constraint regression run passes 225 cases with one recorded
skip; two further batched infeasible-cost cases pass. The Pareto, pruning and
legacy multi-objective run passes 112 cases, including overlapping scalar cases.
Checks compare unmodified reference values, gradients and Hessians, common
sampling draws, explicit probability caps, selected-input derivatives and guards
against host data conversions. Counts are not combined into a release total.
