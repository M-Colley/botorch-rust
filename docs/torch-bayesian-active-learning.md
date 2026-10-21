# Fully Bayesian disagreement on the Torch backend

`qBayesianActiveLearningByDisagreement` now dispatches Tensor fully Bayesian
models and fully Bayesian `ModelListGP` models to a device-resident joint
Gaussian-mixture calculation. It draws component samples from the port's
posterior with observation noise, evaluates each sample under every component,
and estimates mixture entropy minus mean conditional entropy. Gaussian
log densities, sample reuse, log-sum-exp, and reductions retain CPU/CUDA graphs.
Pending points and scalarizing posterior transforms are supported. A missing
sampler uses 512 Sobol samples, matching the reference default.

Dense covariance and event values use the same output-major order. This also
supports an unscalarized multi-output ModelList as a joint information-gain
extension; the documented reference interface asks callers to scalarize a
multi-output model. No distribution ordering metadata is mutated. Multiple
sample axes and an empty sample shape reduce over all MC samples while keeping
the MCMC component axis distinct. Negative MC estimates retain the reference
warning rather than being clipped, which would change the estimator.

Eighteen focused CPU/CUDA cases compare scalar values and query derivatives to
pinned official BoTorch `b0dd01be61d517c387e234903a6e2cc38c057ea2`, and compare
query Hessian-vector products to reference first-gradient differences.
Scalarized ModelList values also match the reference. A separate explicit
component-by-component Gaussian-density oracle checks unscalarized joint
covariance layout and multidimensional MC reduction. Training inputs, targets,
fixed noise, query, posterior-transform weights, and loaded parameters retain
first and second derivatives. A single-component mixture has zero information
gain and finite second derivatives. These changes do not extend the existing
native CPU BALD restriction on multi-output or ModelList models.
