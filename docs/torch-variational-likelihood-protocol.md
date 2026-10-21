# Supplied variational likelihood modules

The Tensor `SingleTaskVariationalGP` accepts supplied scalar GPyTorch
non-Gaussian likelihood modules. The actual likelihood object, registered
Parameters, buffers and overridden methods remain in use. Gaussian inference
and variational updates belong to the port; no upstream GP model computes them.
GPyTorch is needed only when supplying one of its module objects.

The ELBO calls the supplied `expected_log_prob` protocol. One-dimensional
likelihoods use marginal Gaussian moments without materializing an N-by-N
training covariance. Other scalar likelihoods receive the full Gaussian law,
so a joint observation mapping can use off-diagonal covariances. Saved q(f)
outputs capture likelihood Tensor state with differentiable clones, preserving
the original leaf gradients after a later likelihood update.

Observed predictions return the supplied likelihood's actual distribution,
including its Monte Carlo sample axes. The adapter provides a GPyTorch Gaussian
input with port-owned differentiable Cholesky sampling, avoiding incomplete
lazy-operator sampling Hessians. Discrete samples retain their distribution's
ordinary PyTorch differentiation semantics.

`test_torch_variational_external_likelihood.py` has 42 CPU/CUDA cases covering
Laplace, Student-t, Beta and Poisson likelihoods, a Student-t subclass with an
extra learned offset, and a joint linear observation likelihood. Values and all
first derivatives agree with the unmodified pinned BoTorch/GPyTorch models.
Independent dense Gaussian algebra and direct likelihood quadrature check mixed
second derivatives; two convergent finite differences of unmodified source
first gradients separately verify that oracle. Observed-distribution graphs,
saved-state behavior and no-host-copy/no-full-covariance guards are included.
The combined likelihood/probit/distribution/snapshot regressions pass 174 cases;
Torch 2.14.1 CPU passes all 21 available new cases.

Multiple latent outputs with a non-Gaussian likelihood remain outside this
checkpoint. Gaussian and canonical probit likelihoods retain their existing
port-owned numerical paths.
