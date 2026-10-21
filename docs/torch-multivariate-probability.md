# Multivariate Gaussian probabilities on Torch

The public `bvn`, `bvnu` and `bvnmom` routines execute Genz polar/Taylor
quadrature and the truncated-moment formulas directly on Tensor devices.
Their ordinary nonsingular domains retain higher derivatives with respect to
correlations and finite bounds. NumPy and list inputs retain native Rust.
The quadrature degree is chosen per correlation, preserving the native
implementation's independence from unrelated batch members.

Inactive singular formulas are not evaluated. Infinite-bound limits have
finite zero bound derivatives; correlations at exactly ±1 use the degenerate
one-dimensional probability law. Correlation sensitivity at those singular
boundaries is not a smooth interior derivative, and is not claimed as such.

`test_torch_bvn.py` checks CPU/CUDA float32/float64 values, reference first and
second graphs away from known source defects, independent Plackett density
identities, gradcheck/gradgradcheck, truncated moments and host-transfer guards.

## Independence-point correlation derivative

The pinned source `bvnu` handles `r == 0` by returning `Phi(-h)*Phi(-k)`.
Although the value is correct, that expression drops all dependence on the
correlation. At `r=0, h=.4, k=-.2`, the source returns no correlation gradient;
with only `r` tracked, backward raises because the result has no graph.
The true derivative is `phi(h)*phi(k)`, approximately `0.1440`, confirmed by
centered differences of the unmodified source probabilities.

The Tensor port evaluates the polar formula at zero, preserving this
derivative and its higher pullbacks. The isolated reference regression records
the original defect. Legacy gradient comparisons correct only the source's
zero-correlation entries using the analytic density formula, keeping all other
reference checks intact. A concrete upstream repair is to remove the `r == 0`
shortcut, allowing the polar integration branch to handle independence.
