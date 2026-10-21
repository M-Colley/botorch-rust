# Supplied variational likelihood modules

The Tensor `SingleTaskVariationalGP` accepts supplied GPyTorch
Gaussian and non-Gaussian likelihood modules. The actual likelihood object, registered
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

Multiple latent outputs now retain their multitask Gaussian layout when passed
to the supplied likelihood. `SoftmaxLikelihood` supports categorical targets,
learned class mixing or identity mixing, model batches, observed categorical
probabilities, and full parameter/query derivative graphs. The ELBO follows the
supplied likelihood's event reduction and normalization, and forwards likelihood
keyword arguments, including differentiable Tensor arguments. It retains the
actual module and does not replace classification by a Gaussian-noise objective.
Construct a categorical model with `train_Y=None`, explicit `num_outputs` and
pass integer labels to the ELBO, as with the reference constructor.

The scalar and categorical likelihood suites together pass 60 CPU/CUDA cases;
all 30 available cases pass on Torch 2.14.1 CPU. Training objective values and
complete first derivatives match unmodified source models. Independent dense
Gaussian moments, direct likelihood quadrature and common-normal categorical
calculations verify the higher graphs. The source's optimized variational
predictive double backward is repaired separately in
[GPyTorch #2824](https://github.com/cornellius-gp/gpytorch/pull/2824).
Native Gaussian specifications and canonical scalar probit likelihoods retain
their existing port-owned numerical paths.

Actual Gaussian likelihoods also retain subclass overrides and multitask noise
operators. The ELBO calls their actual `expected_log_prob`, including likelihood
keyword arguments and saved parameter/buffer state. Standard Gaussian and
multitask Gaussian expectations use only marginal variances, preserving the
O(NM) training path. A custom Gaussian expectation can inspect the full saved
q(f) covariance. Observed predictions use the actual Gaussian marginal,
including shifted means and off-diagonal task noise, then convert its explicit
event ordering into the port's posterior ordering. Parameter identities remain
unchanged; full covariance and common-normal sample graphs remain on device.

The adapter materializes supplied Gaussian noise operators through their Tensor
multiplication rather than the generic LinearOperator Matmul autograd wrapper.
That wrapper's inherited bilinear-derivative fallback detaches higher graphs
for Kronecker noise; independent dense task-noise formulas check the complete
sample Hessians instead. The source repair is submitted in
[linear_operator #155](https://github.com/cornellius-gp/linear_operator/pull/155).

`test_torch_variational_gaussian_protocol.py` adds 42 CPU/CUDA cases: canonical
Gaussian module identity, shifted Gaussian subclass semantics, diagonal and
rank-one correlated multitask noise, model batches, unmodified source ELBO
values and first derivatives, independent dense all-parameter/data higher
derivatives, common-normal observed sampling, saved state, likelihood Tensor
keywords, deepcopy, repeated backward, and no-host-copy/no-square-training-
covariance guards. The combined Gaussian/non-Gaussian/categorical/probit,
distribution, noise and snapshot suites pass 246 cases.
