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

The quadrature forward uses exact analytic Gaussian partials for reverse and
forward automatic differentiation. Their Torch expressions retain all higher
pullbacks; `torch.func.jvp`, `jacrev` and `vmap` are also tested. In particular,
the reference's high-correlation Taylor expression contains `max` and `abs`
cusps whose cancellation is lost by its second backward at `h=k` (or `h=-k`
for negative correlation). The distribution is smooth there for `abs(r)<1`.
The regression at `r=.98,h=k=.4` checks the true mixed derivative against the
bivariate density and centered differences of the unmodified source gradient.

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

## Stateful multivariate approximation

`MVNXPB` uses the same bivariate-conditioning approximation on Tensor devices,
including standardization, greedy variable prioritization, truncated-mean
plug-ins and odd-dimension marginal corrections. This remains an approximation
to a general multivariate Gaussian integral; the bivariate blocks use Genz
quadrature. Initial covariance diagonal entries must be positive.

`solve(num_steps)`, resumption, `select_pivot`, manual pivots, supplied
precomputed factors, `augment`, batch indexing/concatenation/expansion, cloning
and explicit detachment keep their public state semantics. Tensor factor and
plug-in updates create new arrays, so later state mutations do not invalidate
an earlier forward's backward graph. Integer permutations and pivot choices
are discrete; gradients flow through the chosen numerical branch. This also
avoids the source's incorrect batched in-place pivot gradients.

`augment_cholesky` stays on the Tensor device even without gradients. Augmenting
a completed solve accepts cross-covariances in the existing solver permutation
order, as in the source API. If a Schur factorization fails, the existing
device-native safe-Cholesky routine supplies the configured jitter recovery.
Quadrature constants are cached per device and dtype. Repeated variables with
zero conditional variance use the limiting deterministic interval event;
singular covariance derivatives are not advertised as smooth interior graphs.

`test_torch_mvnxpb.py` compares one through six dimensions and float32/float64
against the pinned reference, using memberwise source derivatives where its
batched pivot graph is wrong. It checks gradcheck/gradgradcheck for fresh and
resumed solves, old-forward preservation, augmentation, state views and
no-host-transfer operation with and without gradients. The existing phase11
state and native-array regressions remain enabled.
