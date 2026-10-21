# Supplied Gaussian likelihoods in structured Tensor models

The exact MLLs of Tensor `KroneckerMultiTaskGP`, `HigherOrderGP` and
`LatentKroneckerGP` evaluate the actual supplied Gaussian observation module.
The module and its registered Parameters retain their identities. Scalar
Gaussian likelihood subclasses may supply a full correlated covariance or
change the observation mean. Multitask likelihood subclasses may override
their marginal and shaped noise covariance. Additional MLL positional
arguments and keywords reach the supplied marginal, including explicit
observation-noise tensors for scalar likelihoods.

Canonical multitask noise uses the existing direct Tensor factor/diagonal
formula, preserving higher derivatives without a lazy-operator custom VJP.
Custom modules retain the derivative semantics of their own implementation;
an upstream lazy operator with an incomplete higher backward does not acquire
a new higher-order contract merely by being supplied to this port.

`FixedNoiseGaussianLikelihood` is accepted for higher-order and latent
Kronecker training and MLL evaluation. Noise is indexed by observed flattened
grid cells, including the latent model's missing-value mask. Implicit MLL
evaluation uses that event shape, rather than inferring a smaller noise shape
from the number of feature-grid rows. Explicit extra likelihood inputs retain
the supplied module's own shape semantics. Latent Kronecker implicit MLLs also
evaluate a subclass's actual `forward` training prior.

This first checkpoint changes training likelihood evaluation. Prediction and
Matheron sampling with noncanonical observation modules are a separate pending
integration; callers should not infer those paths are complete from successful
MLL evaluation. The reference explicitly rejects a fixed-noise likelihood in
`LatentKroneckerGP.posterior`.

`test_torch_structured_gaussian_likelihoods.py` independently builds complete
observation covariances and checks values, every data/parameter gradient and
mixed Hessian products on CPU/CUDA. It also checks unmodified pinned reference
data log probabilities, actual likelihood identity, supplied noise gradients,
and the absence of host copies during the tested operations.
