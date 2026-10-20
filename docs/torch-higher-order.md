# Tensor higher-order Gaussian processes

`HigherOrderGP(..., backend="torch")` implements the structured correlated
law on CPU and CUDA with real PyTorch parameters. CUDA or gradient-bearing
training tensors and real Torch modules also select this backend automatically.
The native CPU/list backend remains available.

For a target grid of size `s1 x ... x sk`, its covariance is
`K_X ⊗ K_Z1 ⊗ ... ⊗ K_Zk`. Each `Zi` is a learned latent coordinate tensor;
outputs are not replaced by independent GPs. Input and latent kernels,
Gaussian likelihood noise, latent coordinates, training inputs/targets and
query inputs retain their first and higher derivatives. Exact likelihoods
and prediction solves call the Rust Tensor bindings, which issue Torch
operations without host copies. Actual custom Torch kernel and Gaussian
likelihood modules retain their parameter identities and derivative contracts.

The default outcome transform standardizes all training grid cells together,
separately per model batch. Statistics retain training-target graphs, including
finite derivatives at constant targets. `latent_init="gp"` stores the actual
initial prior covariance, contributes its density to the MLL and honors
`learn_latent_pars`; multiple latent coordinates per grid axis are supported.
MLL normalization divides by the number of observed grid cells.

Posterior means, variances and samples have shape
`sample_shape x batch x q x target_shape`. `distribution` exposes the full
point-major joint covariance (rather than only marginal diagonal blocks).
Matheron sampling draws the joint input/grid prior and independent training
noise, then applies the exact conditional update; its normal layout is
`(n+q)*grid_size + n*grid_size`, plus `q*grid_size` when requesting noisy test
observations. The roots differ from the reference eigen roots while preserving
the same Gaussian law. Conditioning and fantasies append structured targets,
broadcast fantasy/model batches and retain the original transform statistics.

This initial Tensor implementation materializes the full Kronecker covariance
and uses a dense exact solve: memory grows quadratically and factorization
cubically with `n*grid_size`. It preserves correlation and full graphs but
does not yet claim the reference operator solver's large-grid scalability.
Fixed per-observation conditioning noise, output selection and arbitrary
posterior/outcome transforms are explicit unsupported configurations.

The pinned official reference (`b0dd01be6`) detaches latent covariance during
evaluation. Tests therefore compare full training/latent/query derivatives
against an independent Gaussian conditional built from its actual kernel
modules, then check mixed Hessian products by finite differences. Its noisy
HOGP posterior recomputes latent marginal variances and omits test noise from
the Matheron draw; this backend instead adds the requested independent
Gaussian observation noise consistently to covariance and samples. Tests
verify that law directly without treating detached/marginal-only reference
objects as a joint autograd oracle.

Validation: `test_torch_higher_order.py` covers float32/float64, CPU/CUDA,
model batches, exact values/MLL, all-parameter first derivatives, mixed
second derivatives, prior density, custom modules, structured transforms,
conditioning and fantasies. Sample means/covariances are checked against
the actual joint conditional, including noisy observations.
