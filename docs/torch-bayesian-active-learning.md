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
calculation. No distribution ordering metadata is mutated. Multiple
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
gain and finite second derivatives.

An additional CPU/CUDA control compares the unscalarized ModelList directly
with the pinned source using the same flattened Gaussian root coordinates.
The source's event-order metadata change induces a fixed permutation of the
public `q x m` base-normal array; accounting for that permutation makes
values and query gradients agree at the existing strict tolerances. The
metadata mutation alone is therefore not established as an upstream bug,
and no PR was filed for it.

The native CPU path also accepts scalarized fully Bayesian ModelLists.
Each MCMC component is the native joint model of the corresponding members,
then the requested scalarizing posterior transform is applied before the
existing Rust information-gain and query-adjoint calculation. Singleton
component batches broadcast across larger component batches; incompatible
non-singleton sizes still fail. IID and Sobol normal samplers are accepted,
including explicit IID normals, seeded redraws when q changes, pending
points and public cache reuse. Native gradients through supplied base
normals require the Tensor backend and raise an explicit error otherwise.
Unscalarized native multi-output BALD now uses the same full joint law.
Each component keeps its complete `q*m` mean and covariance, including
cross-output blocks; public `q x m` normals are converted to the native
output-major root coordinates. The mixture component is shared across outputs.
The calculation is not the sum of marginal output information gains.

Eighteen new native tests compare two kernel families, equal/singleton
MCMC sizes, pending points, and both normal samplers with pinned official
values and query gradients. They also check seeded IID cache contents,
list/NumPy/Torch query parity, cache replacement and precise boundary
errors.

The joint extension adds an independent Rust two-output law with nonzero
cross-output covariance. Its mixture values compare to component Gaussian
densities, query adjoints compare to finite differences with and without
pending points, and per-member raw-parameter VJPs compare to finite differences
of the mean slope, cross-output correlation and observation noise. The core
`forward_with_member_parameter_gradients` method returns those member vectors
in each model's raw parameter order. Python persistent native-FB parameter
leaf wiring is separate; use the Tensor backend for complete parameter and
higher-order graphs.

Python regressions compare native joint ModelLists against independent dense
Gaussian densities and the pinned source with canonical root coordinates,
covering Matern/linear members, pending points, IID/Sobol normals, changing
point counts and list/NumPy/Torch inputs. The scalarized regressions remain
unchanged.

Validation of the joint checkpoint: both new Rust tests pass and all 46 native
and Tensor BALD Python cases pass against a fresh optimized test-profile ABI.
