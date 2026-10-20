# Mixed-input and multi-fidelity Tensor GPs

`MixedSingleTaskGP` and `SingleTaskMultiFidelityGP` accept `backend="torch"`.
CUDA training tensors or training tensors requiring gradients select it
automatically. Ordinary CPU arrays retain the native implementation; select
the backend at construction for higher-order query gradients alone.

The mixed model preserves the official categorical/continuous sum and product
covariance tree, its separate lengthscales and output scales, and its priors.
Categorical inputs enter equality comparisons and therefore have zero query
derivatives; continuous inputs and all raw covariance parameters retain their
ordinary Tensor graph. A purely categorical model uses the official scaled
categorical covariance. Custom continuous-kernel factories can return port
kernels or actual Torch kernel modules. External modules retain their original
Parameters and registered priors; their adapter adds a `module` component to
the parameter namespace. Custom Torch modules retain their own derivative
support; an upstream custom kernel without double backward is not repaired by
the adapter.

The multi-fidelity model preserves the actual linear-truncated covariance,
including one/two-fidelity bias interactions and the biased/unbiased Matérn
kernels. The alternative covariance uses the actual product of ordinary,
exponential-decay iteration-fidelity and downsampling data-fidelity kernels.
Multiple data-fidelity indices, negative indices, configured Matérn smoothness,
custom non-fidelity covariance modules, kernel priors and likelihood priors
retain their semantics. As in the reference, supplying a custom non-fidelity
kernel with `linear_truncated=True` raises an error.

These are exact Gaussian models using the shared Rust Tensor factorization,
prediction and log-marginal-likelihood dispatcher. Covariance construction and
all intermediate values stay on their Tensor device; inference does not use
upstream BoTorch/GPyTorch models or the CPU/f64 backend. Batch dimensions and
independent output dimensions retain independent hyperparameters. The existing
Tensor model's transforms, output subsetting, conditioning, fantasies,
samplers, fitting and acquisitions operate on the same covariance modules.
First and higher derivatives include the original training inputs, targets,
fixed noise, queries and raw Parameters. Categorical labels remain discrete
metadata; mixed/discrete acquisition search is a separate optimizer feature.

`test_torch_specialized_exact.py` passes 94 CPU/CUDA cases against official
BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2`. Tests cover float32/float64,
model batches and multiple outputs, posterior means/full covariances, normalized
MLLs and priors, parameter/query first derivatives, custom live Torch kernels,
subsetting and fantasy sample derivatives. An independent conditional law
formed from actual reference kernel modules verifies every training/noise/query
and parameter first derivative without the reference's detached eval cache.
Centered differences verify mixed Hessian-vector products over all those
arguments. Tensor-forward tests forbid CPU, NumPy and list conversions.

The fidelity adapters preserve the native port's previously documented repair
to the multi-dimensional downsampling diagonal: it is the actual diagonal of
the full covariance, unlike the reference's power-of-summed-products branch.
Categorical feature-batch covariance similarly preserves point-axis order.
Neither model implies GPU support for correlated Kronecker or higher-order
families; those require their own correlated joint laws.
