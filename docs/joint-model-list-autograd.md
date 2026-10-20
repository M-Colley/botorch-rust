# Joint ModelListGP parameter gradients

Joint Gaussian `ModelListGP` posteriors now attach the actual persistent
parameter leaves of ordinary CPU Torch `SingleTaskGP` children. Mean,
variance, covariance and reparameterized samples support selected outputs,
scalarization, different training sets, and independent multioutput children.
Each posterior retains native child snapshots and original leaf identities.
Repeated child/parameter aliases accumulate contributions through Torch's
normal graph semantics. Query-input derivatives remain attached separately.

The native list stores a broadcast parent batch shape and an explicit index
map into each original child. Parameter pullbacks lift selected/scalarized
bars into the full joint Gaussian, extract each independent child block,
expand queries to the parent batch shape, and apply the child's native
analytic pullback. This last expansion preserves repetitions introduced by
other children, such as `(2, 1)` and `(3,)` broadcasting to `(2, 3)`.
Gradients reduce into each child's original parameter shape. Cross-child
covariance derivatives vanish because the model is independent across children.

Unsupported child families cause an explicit parameter-backward error when
mixed with supported persistent child leaves. Their existing query-input
interfaces remain available. Common unbatched analytic/scalar MC acquisition
parameter graphs are documented in `acquisition-parameter-gradients.md`.
Specialized all-unsupported lists retain their
existing parameter protocols; no new parameter differentiation is claimed.

## Reference defects and validated solutions

Tests use official BoTorch `b0dd01be6`, Torch `2.13.0+cu132`, GPyTorch
`1.15.2`, and `linear_operator` `0.6.1`. The reference is the official pinned
checkout, not the separate local audit fork or the installed BoTorch release.

For a mixed list containing a two-output `SingleTaskGP`, official
`separate_mtmvn` slices a lazy `BlockDiagLinearOperator` into task blocks.
On these dependency versions, that slice can return a different matrix from
the corresponding slice of its dense covariance: within-task off-diagonals
are lost and variances move between tasks. The list's mean remains correct,
so checking means alone misses the error. The regression in
`test_model_list_parameter_autograd.py` reproduces the discrepancy, then
executes the proposed one-line fix and checks joint covariance and parameter
gradients against the correct conditional.

`docs/patches/botorch-model-list-task-blocks.patch` materializes the covariance
before task indexing. This is a conservative correctness workaround and can
increase memory use. A more efficient library-level solution is to correct
lazy block indexing or extract contiguous blocks directly from the
`BlockDiagLinearOperator` base operator. The Rust port constructs its
independent blocks directly and does not reproduce the lazy indexing bug.

Some official multioutput GP batches also fail during `ExactGP` predictive
cache reshaping when query and model batch axes broadcast. Joint parity
tests therefore assemble the exact Gaussian conditional using the actual
official kernel, mean, likelihood-noise and outcome-transform state with
Torch Cholesky solves. This corrected reference retains the official
parameter graph and differentiates the full training solve. Separate tests
verify its values and gradients against ordinary official child posteriors
on supported layouts. Native directional finite differences independently
check the newly supported parent/child broadcast cases. Evaluating independent
conditional blocks and broadcasting them explicitly is a working solution
for the reference batch-cache failure; this document does not claim a
one-line upstream cache fix.
