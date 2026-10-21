# Batch-decoupled variational inference

`SingleTaskVariationalGP` accepts port or GPyTorch
`BatchDecoupledVariationalStrategy` factories on the Tensor backend. The strategy
stores two independently trainable inducing sets. The first produces the mean
interpolation; the second produces the covariance interpolation. With `A_mu`
and `A_var` denoting their separately factored, whitened cross-covariances,

```text
mean = mean_module(X) + A_mu.T @ q.mean
cov  = K_var(X,X) + jitter*I + A_var.T @ (q.covariance - I) @ A_var
```

Both training and evaluation retain the full covariance correction, matching the
source decoupled strategy. Each inducing matrix is factored in double precision
and cast to the data dtype, following the reference. Distinct mean/covariance
kernel and mean parameters use `mean_var_batch_dim`, supplied through a partial
factory; negative axis layouts and shared hyperparameters are covered. Scalar
model batches and unbatched independent outputs are implemented. As documented
by the source wrapper, batches of multi-output models should use a model list.

The regularizer is the source's decoupled convention:

```text
KL(Delta(q.mean), Normal(0,I)) + KL(Normal(0,q.covariance), Normal(0,I))
```

It includes the Delta term's prior-density normalizer. It differs from the
ordinary Gaussian KL by `M*log(2*pi)/2` per inducing distribution. Cholesky,
mean-field and supplied Gaussian modules, including Natural and TrilNatural,
retain their actual parameter and gradient semantics. A Delta variational
family is rejected, as required by the source decoupled constructor.

Pinned GPyTorch's `IndependentMultitaskVariationalStrategy.kl_divergence` calls
its superclass instead of the base strategy's override. This loses the
decoupled constant for independent outputs: with two outputs and three inducing
points it drops `5.513631199228036` before ELBO scaling. The port delegates to
the base strategy. A separate regression proves the unchanged source behavior
against the analytic Gaussian/Delta expression. Only the affected independent
output reference terms are corrected using that same source base strategy;
other conditional, likelihood, ELBO and derivative comparisons use unchanged
source implementations. An upstream repair is being prepared for this wrapper
and the equivalent LMC delegation defect.

The focused suite checks distinct inducing effects, complete conditional/ELBO
first derivatives, mixed data/parameter Hessian-vector products against finite
differences of independent reference gradients, Natural VJPs, actual Adam
updates, CPU/CUDA float32/float64 execution and physical covariance layout.
Pinned `linear_operator` matrix-Hessian limitations are not treated as the
second-derivative oracle.

The default common-inducing-set Gaussian Matheron updater is not a decoupled
sampler. It raises an explicit request for a custom update rather than treating
the two sets as extra model batches. Arbitrary callback composition remains
available. CIQ's different symmetric whitening law is a separate implementation
gap; this strategy does not substitute ordinary or CIQ whitening.
