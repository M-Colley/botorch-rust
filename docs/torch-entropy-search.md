# Device-resident entropy search

The Torch backend implements `qMaxValueEntropy`, `qLowerBoundMaxValueEntropy`
(GIBBON), `qMultiFidelityMaxValueEntropy` and
`qMultiFidelityLowerBoundMaxValueEntropy`. GP inference calls the port's
`model.posterior`; acquisition formulas and conditional Gaussian algebra use
Torch tensors on the model device. No runtime calculation delegates to BoTorch,
SciPy or NumPy. Native CPU models retain their Rust entropy implementations.

This checkpoint supports initially unbatched scalar Gaussian models, including
supplied Gaussian-distribution adapters. Queries have `batch_shape x 1 x d`
shape, including multiple query batch dimensions. Pending-point fantasies
support greedy batch filling. Multifidelity MES supports projection, trace
observations and cost weighting; multifidelity GIBBON supports projection and
costs, and rejects trace observations. Initially model-batched and fully Bayesian
ensembles remain explicit boundaries at this stage. MES observation/fantasy
samplers currently require one sample dimension, and MES requires at least two
observation samples for its control-variate estimate.

Thompson maxima and Gumbel fitting run on device. The Gumbel sampler solves the
independent Gaussian maximum's three quantiles with batched log-CDF bisection.
It retains posterior precision, consistent with the existing native CPU sampler;
the pinned BoTorch helper rounds its SciPy roots through the default dtype.
Independent float64 SciPy roots provide the stricter numerical oracle, with the
unmodified reference compared separately. Standalone Tensor Gumbel sampling
retains implicit first and second training-data derivatives through Newton
refinements at the solved root. Acquisition constructors deliberately sample
their maxima under `no_grad`, matching the fixed-maximum-sample estimator.

The default Gumbel constructors preserve the reference's random-draw order,
including observation/fantasy sampler seeds and the multifidelity redraw.
Explicit `seed`, `sampler`, `fantasies_sampler`, `uniforms` and `base_samples`
extensions allow controlled sampling. Pending observations are detached when
fantasized as in the reference; kernel, mean, noise, training-data and query
operations otherwise retain their Torch graphs. Cost samplers have independent
state. Raw training inputs are appended to the maximum candidate set, avoiding
the existing reference double-input-transform issue.

Validation uses official revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`. The focused CPU/CUDA suite has 84
tests, including float32 and float64, synchronized maximum/observation draws,
minimization/maximization, pending fantasies, traces, costs, query batch shapes,
first/mixed second derivatives and complete training/query `gradcheck` and
`gradgradcheck`. Tests forbid `.cpu()`, `.numpy()` and `.tolist()` during
construction, inference and double backward. Nine existing native max-entropy
constructor/sampling regressions also pass. No speed claim is established for
these entropy acquisitions yet.

## Verified reference defects and concrete repairs

**GIBBON pending covariance uses inconsistent output units.** With Standardize,
the reference obtains cross-covariance from `model(X_batches)` in transformed
units, but its pending block and marginal variance from `model.posterior` in
physical output units. The Schur complement then changes with the arbitrary
output scale. The new test multiplies the training outcomes by four and compares
against the complete noisy joint posterior. The repair is to obtain cross and
pending blocks from that same posterior, applying any posterior transform
consistently. The port uses this corrected covariance space. The unmodified
reference discrepancy is asserted separately from the independent corrected
Gaussian-law comparison.
This output-unit repair is already proposed in
[BoTorch PR #3391](https://github.com/meta-pytorch/botorch/pull/3391);
it is not a newly discovered independent upstream defect.

**GIBBON assumes one query batch dimension.** `acq.mean(dim=1)` averages the
wrong axis for a `3 x 2 x 1 x d` query, and pending expansion/indexing also fixes
the number of batch axes. The unmodified pinned implementation raises for that
query shape. Average the final maximum-sample axis and use ellipsis indexing
and broadcasting across `X.shape[:-2]`. The port's batched values equal separate
row evaluations and retain all parameter derivatives.

**MES control-variate correlation is undefined for constant observation
draws.** An explicit all-zero normal base-sample tensor yields zero variance for
the log-density control variate. The pinned implementation divides by a zero
variance product and returns NaN. Set the control coefficient to zero in this
case; select a finite positive product before the square root to keep inactive
branches safe under double backward. Tests assert the reference NaN and the
port's finite value and second derivative.
With a single observation, skip variance evaluation entirely: the unbiased
variance has undefined derivatives even when its NaN value is masked later.
The one/eight-observation regressions run on both CPU and CUDA. The native
control mean is centered before summation, so an exactly constant control
retains exactly zero variance instead of a floating-point mean residual.

These are local implementations and proposed upstream repairs. Root-level
publication work tracks their issues and pull requests; this chapter does not
claim an upstream submission.

## Joint entropy search

`qJointEntropySearch` now provides device-native `LB` and `MC` estimates for
initially unbatched Gaussian models with differentiable conditioning. Candidate
batches, pending points, raw optimal locations and scalarizing posterior
transforms retain the same semantics as the corrected native CPU estimator.
The initial model must support conditioning; fully Bayesian mixtures and
initial model batches remain explicitly unsupported for JES. The focused
50-case CPU/CUDA suite includes synchronized official Gaussian sampling,
independent Schur-complement conditioning for explicit fantasy noise,
Normalize, `q > 1`, multiple query-batch axes, training/optimal/query first and
second derivatives, scalarized multi-output and ModelListGP models, float32
and float64, all parameter Hessian products, and checks forbidding
inference-data host conversion. This does not establish a speed claim.

The scalar JES Monte Carlo CDF has a further verified scaling defect in the
official pinned revision. At
`botorch/acquisition/joint_entropy_search.py:346`, it evaluates
`(normalized_mvs - rho * normalized_samples) / (1 - rho**2)`. For standardized
jointly Gaussian latent and noisy observations, the conditional variance is
`1 - rho**2`; standardizing a normal CDF requires its square root. For example,
`F ~ N(0, 0.3)` and `Y = F + E`, with independent `E ~ N(0, 0.7)`, give
`F | Y=1 ~ N(0.3, 0.21)`. Therefore
`P(F <= 0 | Y=1) = Phi(-sqrt(3/7)) = 0.25634538013096175`.
The current reference instead evaluates `Phi(-sqrt(0.3)/0.7)`, approximately
`0.2169719081489336`. Replace the denominator by `(1 - rho**2).clamp_min(CLAMP_LB).sqrt()`;
the clamp follows the existing multi-objective entropy estimator's numerical
boundary convention. The Rust and Tensor implementations apply this repair.

Tests separately assert the unmodified reference discrepancy and the corrected
Gaussian-law result. Existing live Monte Carlo comparisons explicitly install
the corrected reference oracle. Four static Rust MC fixture entries were
regenerated from the official revision using their identical recorded model
parameters, query locations and normal draws; other fixture entries remain
unchanged. Earlier conditioning/input-transform/marginal-density corrections
are described in [the native acquisition chapter](single-objective-acquisition.md).
The new Rust scalar regression tests the closed Gaussian example directly.

## Predictive entropy search

`qPredictiveEntropySearch` and `qMultiObjectivePredictiveEntropySearch` now
run their expectation-propagation algebra on device. Both support pending
points, `q > 1`, several query-batch axes, maximization/minimization and scalar
or independent multi-output Gaussian models, including ModelListGP. Ordinary
initial model batches are rejected in the same way as the official family;
fully Bayesian EP mixtures are also outside this implementation's supported
scope. Raw training locations are used when building factors, consistent with
the native port's existing input-transform correction. The reference's
deliberately noisy candidate EP covariance convention is preserved.

PES damping and fallback select discrete branches. The implementation retains
exact Torch derivatives within the selected branch; it does not claim
differentiability at a damping/convergence/fallback boundary. It recomputes the
EP factor graph for differentiable evaluations, retaining training-data,
optimal-location and hyperparameter dependence and permitting repeated backward
calls. Functional factor assembly avoids in-place modification of tensors saved
by autograd. Cholesky branch choices are made without recording invalid factors;
only the final positive matrix is factored on the differentiable tape.
`log_cdf_robust` preserves the reference's CDF rounding/clamps and tail
approximation while masking logarithm-of-zero branches before evaluation.

The 30-case CPU/CUDA suite checks reference values/query derivatives, complete
training/optimal/query `gradcheck` and `gradgradcheck`, all parameter Hessian
products, repeated backward calls, Normalize, pending points, scalar/MO/list
models, float32/float64, minimization, arbitrary query batches and forbidden
inference-data `.cpu()`, `.numpy()` and `.tolist()` conversions. Scalar device
convergence/PSD decisions synchronize metadata with Python, as in the reference.
Two additional JES regressions check real device candidate optimization and
ensure model parameter `.grad` fields remain untouched. No speed improvement
has been measured for this EP implementation.

**PES can accept a failed first Cholesky pivot.** The official
`multi_objective/predictive_entropy_search.py:1032,1046` checks whether the sum
of Cholesky `info` codes exceeds one. Every nonzero code is a failure, including
`info=1`. On CPU, calling `_update_damping(I2, diag(-1,1), tensor(1.), 1e-4)`
returns damping one and a finite invalid factor. The additional NaN test does
not detect it. CUDA's failed factor can contain NaNs and trigger that separate
fallback, so the exact unmodified symptom is backend dependent. Reject any
nonzero code; in this example damping one half produces the valid matrix
`diag(1e-4, 1.0001)` after jitter. Both native Rust tests and CPU/CUDA Tensor
tests cover the repair, with the unmodified CPU failure asserted separately.
The Rust change also rejects a previous precision whose first pivot fails.

## Multiobjective lower-bound MES and JES

`qLowerBoundMultiObjectiveMaxValueEntropySearch` and
`qLowerBoundMultiObjectiveJointEntropySearch` now support all four reference
estimators (`0`, `LB`, `LB2`, `MC`) with device-resident posterior statistics,
Gaussian draws/densities and truncated moments. They preserve the existing
native/reference Gaussian-constant convention for batches, including its
`q > 1` behavior. This checkpoint does not change that numerical definition.
JES conditions on raw Pareto locations, retaining gradients through the
conditioning observations. The public entropy helper functions also retain
Tensor graphs when a Tensor is on CUDA or requires gradients.

The 80-case CPU/CUDA suite checks scalar and multiple objectives, ModelListGP,
Normalize, pending points, arbitrary query batches, synchronized Monte Carlo
draws, float32/float64, all model-parameter Hessian products and complete
training/Pareto/bound/query `gradcheck` and `gradgradcheck`. Bound Hessian checks
use disjoint boxes in the smooth interior. Independently perturbing duplicate
shared partition faces can make boxes overlap and cross the reference's
probability-mass clamp, where a classical Hessian need not exist. No inference
arrays are converted to host Python/NumPy data. Initial model batches are
rejected as in the official family; JES conditioning on fixed-noise models
requires unspecified Pareto observation noise and is explicitly rejected,
matching the source's unsupported configuration.

**Infinite integration bounds need normal-tail limits.** In the pinned
multiobjective entropy helpers, `g * phi(g)` is evaluated directly at
`g = +/-inf`. A single box spanning the full Gaussian space therefore yields
non-finite noiseless/moment entropy (`NaN`, or `-inf` for CUDA log-determinant
paths), although its exact entropy is finite. The Monte Carlo forward value
can remain finite while its variance derivative becomes NaN because
standardizing an infinite bound records an infinite variance derivative.
The port masks infinite bounds *before* standardization, substitutes exact
CDF limits `0/1`, and uses the tail limit `g * phi(g) = 0`.
Tests compare the resulting values with the closed Gaussian entropy and
assert finite first/second derivatives; the original non-finite values or
variance gradients are reproduced separately. The usual finite-bound
estimators remain compared directly with the unmodified pinned reference.

The native CPU box implementation applies the same limits: infinite endpoints
have zero `g*phi(g)` moment and zero posterior-parameter adjoints, including
the rescaled endpoints in the MC estimator. New Rust regressions verify the
full-space Gaussian entropy/adjoints for all four estimators and a half-line's
closed entropy with finite-difference mean/variance/noise derivatives. MES
has a separate exact-law regression for one/eight zero-normal observation
draws, checking its finite value and Dual derivatives.
