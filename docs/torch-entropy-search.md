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
`b0dd01be61d517c387e234903a6e2cc38c057ea2`. The focused CPU/CUDA suite has 82
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
