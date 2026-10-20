# Hierarchical conditional kernels and models

`HierarchicalConditionalKernel` partitions a feature forest into blocks that
activate together. Its covariance sums one Matérn-5/2 covariance per block,
multiplied by a deterministic PSD gate `a(x) * a(y)`. A gate checks every
ancestor on the root-to-block path using BoTorch's `isclose` tolerances.
Branches have zero input derivatives; active continuous features retain
their kernel derivatives. Native products handle parameter derivatives and
the marginal likelihood without Python numerical inference.

The public `kernels` list contains the actual Matérn or ScaleKernel objects.
Lengthscales, optional SAAS priors and bounded outputscales are trainable.
Native parameter paths include an internal activation-product component;
public subkernel attributes continue to address their native parameters.

`HierarchicalConditionalKernelGP` supports unbatched single-output training,
inferred or fixed noise, the usual SingleTaskGP transforms, exact likelihoods,
native fitting, posterior inference and cloning. Shared kernel parameters
across model batches or outputs are explicitly unsupported by the current
exact-GP backend. `HierarchicalConditionalKernelMultiTaskGP` uses the existing
MultiTaskGP backend and its documented likelihood, transform and output-task
restrictions. It maps non-task feature indices correctly when the task
column is first, last or between other features.

## Confirmed upstream problems and solutions

The reference is official BoTorch `b0dd01be6`; numerical parity also runs
against the installed 0.18.1 release.

- The official constructor compares all parent category values globally.
  Separate parents can legitimately have close labels: `{0: {1.0: [1]},
  2: {1.000001: [3]}}` is unambiguous. Only compare categories belonging to
  the same parent. The native adapter applies this validation locally.
- Branch values are derived before an initially unfitted Normalize learns
  its training bounds. With training parent values 2 and 4, transformed
  inputs use 0 and 1 while gates can retain 2 and 4. Fit the transform on
  actual training inputs before deriving branch values, then use its
  preprocessing operation without further fitting. Empty dependency graphs
  remain valid ordinary feature blocks.
- The multitask constructor applies non-task dependency indices directly
  to full input columns when transforming category values. For an interior
  task column, those index systems differ. Map each dependency index through
  the ordered non-task columns before transforming its category value.
- A closed cycle with no root can silently disappear from the official
  forest traversal. Validate every connected component before constructing
  paths. The iterative port rejects cycles, duplicate children and invalid
  indices, and handles forests deeper than Python's recursion limit.

Tests compare full and paired-diagonal covariances, PSD, active/inactive
input adjoints, marginal likelihoods and all parameter gradients, posterior
means and covariance, cloning, normalized branch values and task-column
mapping. Configuration errors are tested explicitly.
