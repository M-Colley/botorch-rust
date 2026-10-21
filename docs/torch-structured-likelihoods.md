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
formula. Supplied GPyTorch marginals retain their actual covariance operators;
their evaluated Tensor multiplication materializes the covariance without the
generic lazy-operator backward that detaches representation derivatives.
This preserves higher derivatives for custom shaped Kronecker noise, including
`3 * super()._shaped_noise_covar(...)`. Deliberate detach operations inside a
supplied module retain that module's derivative semantics.

`FixedNoiseGaussianLikelihood` is accepted for higher-order and latent
Kronecker training and MLL evaluation. Noise is indexed by observed flattened
grid cells, including the latent model's missing-value mask. Implicit MLL
evaluation uses that event shape, rather than inferring a smaller noise shape
from the number of feature-grid rows. Explicit extra likelihood inputs retain
the supplied module's own shape semantics. Latent Kronecker implicit MLLs also
evaluate a subclass's actual `forward` training prior.

Prediction and Matheron sampling use the complete supplied observation law:
the observed training mean, correlated training noise, and query observation
noise where the public posterior supports it. Retained posterior objects keep
their captured laws when the model's likelihood parameters subsequently
change. Latent Kronecker prediction evaluates its virtual training prior and
separate `_get_test_prior_mean_and_covariances` hook, matching the reference's
separation of those extension points. Its full-grid sampling layout preserves
the observed training mask and correlated observation noise.

The reference explicitly rejects a fixed-noise likelihood in
`LatentKroneckerGP.posterior`; the port preserves this boundary while permitting
the supplied module for direct model evaluation and training. Supplied-noise
conditioning and the iterative solver configuration require separate audits.

`test_torch_structured_gaussian_likelihoods.py` independently builds complete
observation covariances and checks values, every data/parameter gradient and
mixed Hessian products on CPU/CUDA. It also checks unmodified pinned reference
data log probabilities, actual likelihood identity, supplied noise gradients,
and the absence of host copies during the tested operations.
