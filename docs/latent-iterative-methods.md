# Latent Kronecker iterative inference

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

The native lists/NumPy/CPU-Tensor route also captures these scoped settings.
It uses native Jacobi-preconditioned CG and fully reorthogonalized Lanczos
quadrature, including complete and masked grids, parameter fitting, posterior
moments, feature/time query adjoints, and Matheron sampling. Native posterior
and deep-copy snapshots retain the settings after the scope exits; parameter
updates made inside nested scopes remain effective after normal or exceptional
exit. Native inference continues to use float64 and does not import Torch.
Its covariance snapshots are dense, as on the Tensor route.

Native trace probes are retained Rademacher draws with an independent seed,
which keeps an optimizer's stochastic objective reproducible between calls.
Native MLL gradients use the inverse-trace estimator of iterative Gaussian MLL
backward; a stochastic log-determinant estimate is not an exact objective or a
bitwise replay of another implementation's probes. Zero trace samples with
full quadrature checks exact values and every raw-parameter derivative against
independent dense inference. Native query derivatives use the implicit Gaussian
solve adjoint; selecting the Tensor backend retains the full higher-order
iteration graph. CG honors the requested iteration limit even above the matrix
size, refreshing its true residual when finite precision requires a restart.

Validation includes native solver edge cases, independent spectral/probe
quadrature, all solver/log-probability flag combinations, mean/kernel/noise
priors, posterior and sampling query adjoints, array-kind and nested-scope
protocols, Torch 2.14 CPU, and execution with reference packages unavailable.
Strict source controls use configurations whose iterative errors satisfy the
asserted moment/gradient tolerances; tighter requested source tolerances can
reach its division guard and emit nonconvergence warnings. Those failed audit
trials remain preserved separately from the successful controls.

The [native verification receipt](verification-native-latent-iterative-2026-10-21.json)
records the immutable binary, 303 committed build inputs, 552 passing Rust tests,
strict Clippy, and focused integrated/source/Torch-version checks. Later Python
configuration changes still require a fresh complete verification before release.
