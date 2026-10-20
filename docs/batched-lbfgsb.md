# Native batched L-BFGS-B

`botorch_rs.optim.batched_lbfgs_b.fmin_l_bfgs_b_batched` runs independent
Rust L-BFGS-B states and groups their pending objective requests. A row that
has converged stops receiving evaluations; `batch_indices` identifies each
active row in the original batch. Bounds, tolerances, limited-memory
corrections, line searches, iteration limits and callback termination are
handled independently.

The resumable state shares the synchronous solver's numerical subroutines.
It can be copied during a line search, preserving curvature and search
history. Results contain a SciPy `OptimizeResult` and `LinearOperator`;
inverse-Hessian products use the native two-loop recurrence. This NumPy API
loads NumPy and SciPy when called. The core Rust solver requires neither.
SciPy's private work arrays are not exposed by `OptimState`.

Validation includes 3 native tests comparing every accepted step against
the existing synchronous solver, 34 Python tests covering official parity,
callbacks, independent batch schedules, suspended copies and inverse
Hessians, and all 16 tests from the pinned official batched-optimizer test
module with its solver import switched to the port.

## Incorrect objective values after failed upstream line searches

Official BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2` with
SciPy 1.18.0 can report an objective value from a rejected trial together
with the restored point and gradient. For the Rosenbrock objective
`100*(y-x*x)**2 + (1-x)**2`, start a batch at `[-1.2, 1]` and `[2, 2]`,
use bounds `[(0, 0.8), (0, 0.7)]`, and set `maxls=1`.

Both solvers return points `[[0, 0.7], [0.8, 0.7]]`. Their actual objective
values are `[50, 0.4]`. Upstream reports `[41, 41]`. The Rust state restores
the previous point, objective value and gradient together and reports the
correct values. The regression checks the objective at the returned point
instead of reproducing the erroneous upstream values.

This is inherited from SciPy's C interface: its Python extension parses the
objective as a scalar, passes a pointer to a local copy, and discards the
restored scalar on return. Changing the Python scalar to a zero-dimensional
array does not correct it. The same mismatch occurs in SciPy's ordinary
`minimize(method="L-BFGS-B")` on this example.

The proposed [BoTorch patch](../patches/botorch-batched-lbfgsb-restored-value.patch)
retains each accepted objective value, including the initial point, and
restores it on abnormal line-search termination. It preserves batching and
does not require extra objective evaluations. The patch applies cleanly to
the pinned source and was verified in 24 configurations: three bound
layouts, four solver-option combinations and both batch-index modes.
It has not been submitted upstream.

Source inspection:
[BoTorch driver](https://github.com/meta-pytorch/botorch/blob/b0dd01be61d517c387e234903a6e2cc38c057ea2/botorch/optim/batched_lbfgs_b.py),
[SciPy C interface](https://github.com/scipy/scipy/blob/v1.18.0/scipy/optimize/__lbfgsb.h),
[SciPy Python driver](https://github.com/scipy/scipy/blob/v1.18.0/scipy/optimize/_lbfgsb_py.py).
