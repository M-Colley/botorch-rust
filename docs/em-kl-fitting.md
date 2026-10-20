# Native Gaussian KL and empirical-prior kernel fitting

`kl_divergence_mvn(mu_p, Sigma_p, mu_q, Sigma_q)` computes
`KL(N(mu_p, Sigma_p) || N(mu_q, Sigma_q))` in Rust using PSD-safe Cholesky
factorizations. Torch inputs support exact first derivatives in all four
arguments; NumPy and list inputs use the same native calculation. The usual
float64 native computation and caller dtype/device restoration apply.

`KLPriorFitMLL` and `fit_kernel_to_em_prior` fit the parametric mean and
covariance at fixed reference locations. They minimize Gaussian KL against
the supplied empirical mean/covariance, using native kernel contractions,
constraint derivatives, and the common L-BFGS-B bound/retry machinery.
The target log determinant is cached across objective evaluations. Fitted
parameters update the caller's modules only after a successful fit.

The objective contains neither likelihood noise nor hyperparameter prior
terms. `KLPriorFitMLL.value_and_grad` follows the model's full raw parameter
order and returns zero for parameters absent from the KL objective.
`fit_gpytorch_mll` dispatches this MLL to its actual KL fitting path.
As upstream documents, matching a kernel to an EM prior is an empirical-Bayes
heuristic, rather than the EM M-step for free prior means and covariances.

The fitting interface currently supports unbatched native kernels with
`ConstantMean` or `ZeroMean`, fixed reference locations, and native L-BFGS-B.
It uses the native MLL value/explicit raw-gradient protocol; it does not create
a Torch graph through model hyperparameters. Grad-enabled reference locations
raise an explicit unsupported error. The standalone Gaussian KL utility has
its own first-order Torch adapter and does propagate all four argument gradients.

Verification: 16 Python checks cover Torch/NumPy/list values, all four matrix
derivatives against Torch, official BoTorch KL-MLL values/raw-gradient order,
fitting dispatch, and recovery of a known Gaussian prior with zero or learned
constant mean. Native tests independently check symmetric covariance finite
differences and raw kernel/mean parameter derivatives.
