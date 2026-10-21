# Device probit variational observations

Scalar `SingleTaskVariationalGP` accepts the port's `BernoulliLikelihood()`
specification and actual GPyTorch `BernoulliLikelihood` instances. These
configurations select the Tensor backend on CPU as well as CUDA. The port
evaluates their observation law, marginal and expected log likelihood using
device Tensor operations; it does not invoke a BoTorch or GPyTorch model or
likelihood for numerical predictions or losses.

The conditional probability is `Phi(f)`. For a Gaussian latent marginal,
the observed probability is `Phi(mean / sqrt(1 + variance))`. ELBO expectations
use Gauss-Hermite integration of `log Phi((2*y - 1)*f)`. Port specifications
default to 20 nodes and allow `num_locs`; nodes and weights are constructed
from the Hermite Jacobi eigensystem on the model device. They follow the
reference's initial default-dtype quantization. Supplied GPyTorch instances
contribute their actual configured quadrature nodes and weights, including
nondefault rules. Snapshot ELBOs retain their original quadrature constants.

Cholesky, mean-field and Delta/MAP inducing families and whitened/unwhitened
conditioning retain the same latent GP laws. The classifier's default posterior
is Gaussian in latent space. `observation_noise=True` returns the Bernoulli
observation distribution through `TorchPosterior`, matching the source's shape.
Discrete Bernoulli observations do not have reparameterized samples; `rsample`
raises `NotImplementedError`, like the underlying source distribution. This
does not assert a pathwise derivative for discrete observations.

The implementation uses `torch.special.log_ndtr`, which has the mathematical
negative-tail derivatives. The installed GPyTorch `LogNormalCDF` custom function
instead approximates the value and saves rational-approximation intermediates
outside its derivative graph. At double-precision `z=-2`, it returns
`-3.7832322852931073` instead of `-3.7831843336820317`, its first derivative is
`2.3733293350595472` instead of `2.3732155328228406`, and its second derivative is
zero instead of `-0.8857208995859178`. Replacing that scalar primitive with
`torch.special.log_ndtr` repairs both the numerical value and higher derivatives.
Reference classifier comparisons disclose this one-primitive repair and retain
the independent reference GP, variational, likelihood, quadrature and MLL code.
Full mixed data/parameter Hessians are additionally compared against finite
differences of those independent reference gradients.

Actual supplied homoskedastic GPyTorch Gaussian likelihoods are also converted
to a port evaluator while keeping their noise parameter and constraint module
identities, trainability flags, priors and optimizer bounds. Their Gaussian
conditional/marginal computations use Tensor operations in the port.

This checkpoint covers scalar probit observations and homoskedastic Gaussian
component interoperability. Independent multitask probit likelihoods, other
non-Gaussian observation families and arbitrary custom likelihood modules are
separate configuration gaps. An explicit observation covariance override is
not a Bernoulli likelihood and raises an error. Source-unsupported unwhitened
and independent-multitask Matheron update boundaries still apply.
