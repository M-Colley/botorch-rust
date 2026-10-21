# Built-in model subclasses

Caller subclasses of `SingleTaskGP`, `MultiTaskGP`, and `LCEMGP` participate in
Torch backend selection. Explicit `backend="torch"`, CUDA data, differentiable
data, and supplied Torch modules preserve the caller's cooperative constructor,
methods, buffers, class identity, deepcopy, and pickle through the shared model
subtype adapter. Explicit `backend="native"` retains the native family and its
configuration restrictions. CPU Tensor LCEM inputs also select Torch when fixed
noise, a custom mean/likelihood, or an input transform requires that backend;
this includes positional `train_Yvar` and port-owned component specifications.

For `SingleTaskGP`, `forward` defines the latent Gaussian prior in both training
and evaluation modes. Training `model(X)` invokes this virtual method, and
an evaluation call conditions the prior on the observed training data.
`posterior` and the exact MLL honor a caller's `forward` override. The conditional
law uses its training Gaussian and its joint training/query Gaussian, retaining
continuous data, target, mean, covariance, and parameter graphs. The default
prior continues to use the efficient covariance-block/factor-cache path. Custom
priors bypass that cache because unregistered caller state can change their law.
No upstream model is used at runtime.

`test_builtin_model_subclasses.py` covers the three factories on CPU and CUDA,
explicit/automatic/native selection, serialization, LCEM configuration selection,
and scalar/multi-output custom priors with and without model batches. Posterior
values and all first derivatives match the pinned source. An independent dense
Torch Gaussian calculation checks the mixed higher derivatives without calling
port numerical bindings or source operators.

The multi-output source comparison exposed a separate double-backward defect:
for the unbatched two-output example, values/first derivatives agree to about
8e-15, while the original source's Hessian-vector product differs by up to
0.213. The port and independent Gaussian Hessian agree to 2e-12. Central
differences of the unmodified source's first gradients converge to the port
(maximum error 3.7e-4, 3.3e-5, and 3.7e-6 at steps 1e-5, 3e-6, and 1e-6 for the
unnormalized direction). The tests retain source first-gradient checks and a
normalized-direction convergence check; they do not relax value tolerances or
patch the reference checkout. The cause is `CatLinearOperator`'s detached generic
bilinear derivative in the independent-MVN covariance assembly, separately from
the triangular operator defect. The analytic child derivative in
[linear_operator #153](https://github.com/cornellius-gp/linear_operator/pull/153)
reduces the complete model Hessian discrepancy to 2.3e-12. Applying only the
previous triangular repair does not fix this case.

This checkpoint does not extend the other specialized model family factories,
or change `MultiTaskGP`/`LCEMGP` virtual-prior semantics. Those paths remain
separate from the `SingleTaskGP` prior override implementation described here.
