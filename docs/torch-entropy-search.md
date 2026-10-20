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
