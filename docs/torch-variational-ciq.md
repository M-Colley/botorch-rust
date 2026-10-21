# CIQ variational inference

`SingleTaskVariationalGP` accepts the port or GPyTorch
`CiqVariationalStrategy` class, including a `partial` configuring finite
nonnegative `jitter_val`. Automatic backend selection chooses Tensor inference
for this strategy on CPU and CUDA. It uses symmetric whitening:

```text
A = (K(Z,Z) + jitter*I)^(-1/2) K(Z,X)
mean = mean_prior(X) + A.T @ q.mean
covariance = K(X,X) + A.T @ (q.covariance - I) @ A
```

This is a different coordinate system from Cholesky whitening; the same
variational parameters generally produce different predictions. The port
computes the inverse-root product using the elliptic contour resolvent sum
from the CIQ method. Arithmetic-geometric means and Landen descent compute
the Jacobi contour nodes entirely with Torch operations on the input device.
Shifted `torch.linalg.solve` operations retain first and higher derivative
graphs. Detached spectral bounds select the quadrature contour, following
the source's treatment of quadrature setup. No NumPy/SciPy transfer, CPU
flattening, eigenvector derivative, or delegated GPyTorch model inference is
used. The GPyTorch `num_contour_quadrature` setting is honored when available,
with the same default of 15 nodes.

The current implementation uses dense eigenvalue bounds and dense shifted
solves. Its setup and solves do not provide the source's iterative large-M
complexity advantage; no speedup for large inducing sets is claimed. These
choices retain complete solve graphs, including repeated-eigenvalue cases.

Cholesky, MeanField, Delta and TrilNatural distributions retain the source's
full conditional covariance and its two additions of data jitter. Their KL
uses the whitened standard Gaussian prior, with the Delta/MAP density
convention described in the family documentation. Supplied distribution
modules retain their registered parameters and gradients.

`NaturalVariationalDistribution` follows the source's specialized NGD-CIQ
convention instead: diagonal predictive covariance with one data-jitter
addition, and a zero KL primal whose backward contributes the canonical
Gaussian KL gradient. The port represents that last operation as
`KL - KL.detach()`. It is an intentional gradient surrogate, not an
ordinary scalar objective with that numerical derivative. The module's
registered natural-gradient VJP is retained; these parameter gradients are
not ordinary Euclidean derivatives. As in source CIQ, this special branch
reads the Natural coordinate law even for subclasses, rather than an
arbitrary subclass `forward` override. First-call initialization also follows
the source's fixed `1e-3` natural-vector perturbation.

The pinned Independent wrapper cannot evaluate the Natural CIQ ELBO because
it reconstructs KL through a property that CIQ explicitly forbids. The
delegation repair is [GPyTorch #2823](https://github.com/cornellius-gp/gpytorch/pull/2823).
A regression demonstrates the unchanged failure. Only that defective wrapper's
KL call is replaced by delegation in the independent-output Natural oracle;
other source implementations remain unchanged.

Strict source Natural comparisons start with diagonal precision so its Jacobi
preconditioner is exact. A separate independent dense Gaussian calculation
checks nondiagonal Natural precision, all parameter/data first derivatives
and the source KL surrogate. This avoids treating finite CG solve error as
an exact oracle. Actual multi-step optimizer replay still uses the unmodified
source solver, including its convergence warnings.

The focused tests cover scalar and batched models, independent outputs,
CPU/CUDA float32/float64, all first conditional/ELBO gradients, actual optimizer
steps, immutable saved laws, query batches and prior bypass. Mixed model
Hessian-vector products are compared with finite differences of independent
reference first gradients. Primitive numerical gradcheck and gradgradcheck
include repeated eigenvalues, where an eigenvector-differentiation shortcut
would be singular. The focused suite passes 54 cases on Torch 2.13 CPU/CUDA
and 28 on Torch 2.14.1 CPU.

The source does not register default CIQ Matheron updates. The port raises
explicitly for that dispatcher; custom pathwise callbacks remain available.
Other strategy constructors that require a base strategy, grid or task
arguments are separate configurations, not aliases for this class.
