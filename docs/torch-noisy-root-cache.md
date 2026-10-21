# Tensor noisy acquisition root caching

`qNoisyExpectedImprovement`, `qLogNoisyExpectedImprovement`,
`qNoisyExpectedHypervolumeImprovement`, and
`qLogNoisyExpectedHypervolumeImprovement` support `cache_root=True` on the
Tensor backend. Independent Gaussian output blocks are sampled on the input
device, using the baseline Cholesky factor and a Schur conditional factor
for the new candidate points. The public cached-Cholesky mixin also computes
Tensor roots without flattening their device or autograd graph.

Baseline and candidate draws share the retained normal coordinates. The MC
objective receives the complete joint draw once, including when it is a
stochastic `LearnedObjective`. Evaluating that learned objective separately
on baseline and candidates would change its probability law; the port keeps
the joint evaluation with both cached and full-factor sampling.

For frozen models, the root cache checks model parameter/buffer mutation
counters, model configuration, posterior-transform state, baseline points,
and baseline covariance. A changed public parameter, training value,
baseline, dtype, or device invalidates the corresponding root on the next
evaluation. Append-only pending prefixes can extend the leading factor;
replacement and clearing preserve the common baseline normal coordinates.
At most the original and current prefix roots are retained.

When training data, parameters, baseline points, or scalarization weights
require an autograd graph, each forward computes a fresh baseline factor.
The candidate draw still uses a Schur solve, but the baseline graph is never
reused after backward. First and higher derivatives therefore include the
baseline factor, covariance, means, and all live data/parameter inputs.
This differs from the pinned reference's constructor-time detached baseline
cache. No speedup is claimed for live-parameter training.

Root caching is disabled with a warning when explicitly requested for
unsupported model/sampler combinations, nonlinear transformed posteriors,
or correlated output blocks. A failed Schur factorization falls back to
full-joint sampling with a numerical warning, following the reference.
Ordinary repeated-point stabilization uses the same adaptive Cholesky
jitter as the Tensor posterior. As in the reference, if stabilization is
needed in only the conditional block, its nugget need not equal the nugget
used for the standalone baseline frontier. One-to-many posterior point
layouts currently fall back to full-joint sampling.

Focused CPU/CUDA tests compare cached values, query gradients, and
directional Hessians with the pinned official cached acquisition path.
An independent dense Gaussian Cholesky oracle checks all query, baseline,
training-data, parameter, and scalarization-weight first and second
derivatives, including repeated backward calls, independent output/model
batches, pending-prefix transitions, and dtype migration. Learned-objective
regressions compare the complete joint probability law using common
explicit normals. These checks do not measure runtime performance.
