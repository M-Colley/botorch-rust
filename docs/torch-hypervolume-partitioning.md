# Device binary hypervolume partitioning

The development Torch backend implements the reference Couckuyt binary
partitioning algorithm in addition to the existing exact Lacour decomposition.
Noisy EHVI and LogEHVI accept nonzero `alpha`; above two objectives they use the
same relative-volume stopping rule and rank-based splits as
`NondominatedPartitioning`. Two-objective partitioning remains exact, as in the
reference. Nonpositive noisy-acquisition `alpha` selects the fast exact
decomposition.

Each posterior sample has its own Pareto count, feasibility mask and rank grid.
Search cells are padded across sample and model batches. Numerical boundary
coordinates stay on the Tensor device and retain their first and higher
derivatives. Only integer widths are read to compact the ragged search tree.
Breadth-first traversal changes cell order but preserves the accepted cell set.
Observed differentiable fronts support both fast and binary partitionings,
including nonzero approximation thresholds.

Noisy baseline pruning uses the public multi-objective pruning protocol,
including a supplied model-batch `marginalize_dim`, objectives and constraints.
That argument applies only when pruning is enabled, matching the reference.
Batched baseline inputs remain unsupported by both implementations.

Noisy geometry uses a standalone baseline-prefix draw with the existing common
normal samples. This preserves the fixed reference frontier when repeated
candidates require jitter in the full joint factor. The prefix posterior keeps
the query's batch axes so evaluating it does not regenerate sampler coordinates.
Full-joint candidate sampling and the documented root-cache fallback remain.

`test_torch_binary_partition.py` compares exact accepted cells on CPU/CUDA,
float32/float64, three/four objectives, four approximation levels, multiple
sample axes, duplicate points, empty and ragged feasible fronts. It also checks
observed/reference-coordinate derivatives, noisy acquisition values and full
first/higher graphs against the pinned official source, host-copy guards,
ordinary model-batch pruning, and ignored pruning-only arguments.
The new cases, existing hypervolume regressions and acquisition-subclass tests
pass together: 162 CPU/CUDA cases with no tolerance changes. The repeated-point
case exposed and verifies the baseline-prefix stabilization repair.

Pending points may carry batch axes. Dynamic and cached prefixes preserve the
point axis during append, replacement, promotion past `max_iep`, and clearing.
Non-incremental cached improvements retain their decision-batch axes when
averaging posterior samples. Twenty CPU/CUDA tests compare batched transitions
against independent decisions with the same sampler history and compare
uncached values/full first/higher graphs against independent unmodified-source
decisions. Together with the existing hypervolume suite, 109 cases pass. The
source currently rejects even uncached batched pending points because its empty
cached prefix has a different batch rank; the regression is reproduced in a
separate upstream fix. The port's cached batched-pending behavior extends the
source's currently working configurations.

These development checks supplement the frozen release certificate; they do
not establish full-package behavioral or GPU equivalence.
