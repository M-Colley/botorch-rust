Noisy improvement sampling and state
====================================

`qNoisyExpectedImprovement`, `qLogNoisyExpectedImprovement`, and
`qLogNParEGO` share a native sampler that supports both cached baseline
Cholesky roots and full joint sampling. The log variants apply the stable
log-improvement reduction to these same sample paths, including per-outcome
constraint smoothing, fat-tailed approximations, prior point weights, and
analytic input gradients. `cache_root` now selects the actual algorithm in
all three classes; the former log variants ignored this argument.

The cache mode draws and stores baseline samples independently, copies
their base-sample rows into the joint draw, and uses a native low-rank
Cholesky update for new points. Unsupported numerical roots use the existing
joint fallback. Replacing a sampler's base-sample matrix invalidates the
derived joint base-sample entry, preserving the fixed cached baseline while
using the new candidate draws.

`compute_best_f(obj)` uses the actual best feasible objective in every MC
sample. With a cached root this is available before a forward call. Without
one, the native evaluator returns its incumbents alongside its value and
gradient; the Python bridge gathers the results in candidate-batch order
after parallel evaluation completes. This prevents one worker's state from
overwriting another worker's incumbents. The result follows `obj`'s array
type, dtype, device, and sample/batch broadcasting shape and has no gradient
through baseline selection. Changing pending points invalidates previous
uncached forward state and rebuilds an incremental cached baseline.

The old log-only memoization keyed incumbents by joint point count and
covariance jitter. That key omitted the sampler's actual base samples, so
replacing draws of the same shape could retain stale incumbents. The shared
uncached path computes incumbents directly from its current joint samples.

Model batch dimensions remain subject to the acquisition layer's existing
unbatched-model restriction. Cached ensemble incumbents support the member
dimension. Public `compute_best_f` for an uncached ensemble explicitly
rejects missing per-member forward state rather than synthesizing it.

Parity targets official BoTorch revision `b0dd01be6`; historical joint-draw
fixtures now specify `cache_root=False` explicitly. Dedicated live tests
exercise both cache modes, non-incremental and incremental pending points,
constraints, native gradients, parallel candidate batches, contextual
objectives, and changed base samples.

The multidimensional sample-shape regression also reproduces an official
upstream bug: `NormalMCSampler._update_base_samples` computes
`base_ndims = current_base_samples.dim() - 1`, counting only one sample
axis. With `sample_shape=(4, 8)`, the cached forward attempts an invalid
reshape. The correction is to subtract `len(self.sample_shape)`. The port
already preserves all sample axes; the test compares its actual forward
incumbents with the official correctly initialized cached incumbents and
records the reference forward exception.

Cached numerical fallback correctness
-------------------------------------

The native cached-root fallback previously evaluated a contextual objective
over `[baseline; candidates]`, while the ordinary low-rank cached path
evaluated it over candidates alone. A forced-fallback regression reproduced
an improvement of `0.491707425270575` instead of `0.464582425270575` for an
objective that adds the mean of its input points. The corrected fallback
evaluates the same candidate-only objective and embeds its sample and direct
input adjoints into the full joint posterior layout. The regression checks
linear and log reduction values, unchanged baseline incumbents, matching
analytic gradients, and central finite differences.

Exact-GP block computation
-------------------------

For models exposing independent exact output GPs, the cached path also
stores their fixed-point training solves and the baseline normal matrices.
Each candidate set requests only its new posterior mean, its covariance
with the baseline, and its own covariance through `cross_posterior`.
Gradients use `cross_posterior_vjp` on these blocks; they allocate no full
baseline covariance adjoint. Models without compatible GP caches retain
the general joint-posterior route and its numerical fallback.

A native regression counts model queries: after initialization, cached
value-and-gradient evaluation calls neither the full-model posterior nor
its joint posterior VJP. Disabling only the GP caches forces the general
route and yields matching values and gradients. The contextual-objective
fallback regression also compares against the optimized block path.
