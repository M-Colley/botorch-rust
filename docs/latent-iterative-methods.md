# Latent Kronecker iterative Tensor inference

Tensor `LatentKroneckerGP.use_iterative_methods` accepts the source's
`tol`, `max_iter`, `covar_root_decomposition`, `log_prob` and `solves`
configuration. The context restores nested settings on normal and exceptional
exit. The configuration does not require GPyTorch; when callers have imported
GPyTorch, supplied modules also see its corresponding settings.

Above the Cholesky threshold (800 observations by default), inference uses
Jacobi-preconditioned conjugate gradients for training solves. Gaussian MLLs
use stochastic Lanczos quadrature for the log determinant when `log_prob=True`.
Both algorithms use ordinary Tensor operations and retain mixed higher
derivatives through inputs, observations and model parameters on CPU and CUDA.
The implementation currently operates on the model's dense covariance
snapshots. Iterative execution reduces factorization work but does not make
those snapshots matrix free.

Imported GPyTorch settings for `max_cholesky_size`, `num_trace_samples` and
`max_lanczos_quadrature_iterations` are captured by the context. Zero trace
samples requests the exact basis trace. Otherwise the log determinant is a
stochastic estimate; equivalence requires comparing the actual probes or
statistical tolerances, rather than expecting identical draws across
implementations. `max_iter` limits both the CG solve and Lanczos iterations.
Exhausted CG tolerance produces a numerical warning and returns its computed
iterate. Setting `solves=False` or `log_prob=False` selects exact Cholesky for
that operation.

Matheron posterior objects retain the solver configuration used at their
creation, including after the context exits. As in the reference's explicit
pathwise sampling implementation, prior/noise factors still use Cholesky;
the `covar_root_decomposition` setting is visible to supplied lazy modules
without replacing explicit Cholesky calls.

The iterative test suite checks forced-CG values and query derivatives against
the unmodified reference, full first and mixed second derivatives against
independent dense algebra, complete masked-grid sample covariance, actual
stochastic probes, shared/zero right-hand sides and iteration-limit behavior.
This Tensor checkpoint does not claim completion of the separate native
lists/NumPy solver route.
