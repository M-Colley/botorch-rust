# HigherOrderGP fantasy updates and supplied covariance factors

Audited against official BoTorch revision `b0dd01be6`.

`get_fantasy_model(inputs, targets)` appends data already in the model's
transformed input/outcome spaces. It shares the native conditioning helper
with `condition_on_observations`, which accepts raw inputs and raw targets
and applies each fitted transform exactly once. This prevents repeated
standardization of fantasy targets. `fantasize` follows the raw-data route.
Structured targets and a flattened `q * prod(target_shape)` vector are
accepted; input and fantasy-target batches broadcast. Public training data
properties now expose every fantasy model's actual stacked data. Explicit
noise and unsupported keyword arguments raise instead of being ignored.

`make_posterior_variances(joint_covariance_matrix)` evaluates the factors
supplied by the caller: the first factor contains the data covariance over
training then test points; subsequent factors contain the output-mode
covariances. A Kronecker operator with `linear_ops` or a sequence of dense
list/NumPy/Torch factors is accepted. Explicit caller operators can expose
`evaluate_kernel` / `to_dense`; individual factors are materialized without
materializing the full joint Kronecker product. All subsequent variance
algebra runs in Rust. The result is flattened in point-major order and in
transformed outcome units, matching the official method. Model noise enters
the training solve; the method does not add test observation noise or apply
the outer outcome untransform.

The forward algorithm diagonalizes the training factors and uses tensor
mode products for
`diag(Ktest) - diag(Kcross (Ktrain + noise I)^-1 Kcross.T)`.
Modified test/train blocks and output factors directly change the result.
Factor and noise batches broadcast, with gradients reduced back over
broadcast dimensions. Optional first-order Torch adjoints are analytic
native contractions; they include the train covariance, test/train blocks,
test diagonals, and training noise. The optional adjoint implementation
currently materializes dense training/cross solve intermediates; the forward
variance algorithm remains factorized. Second-order derivatives and
gradients through conditioning/training targets are not provided.

Validation includes official already-transformed fantasy training data,
with and without `FlattenedStandardize`; nontrivial input normalization;
fantasy batch broadcasting; raw-condition/fantasy consistency; list, NumPy,
and Torch covariance factors; changed test/train blocks; batched factors;
official variance and factor-gradient comparisons; native entrywise finite
differences for every factor and noise; float32/float64; and no-grad mode.

The existing unbatched training-model boundary remains. A collection of
fantasy models supports shared unbatched query points; conditioning an
existing fantasy collection and query gradients through that collection
remain explicit unsupported configurations.

The audit found an additional official HOGP bug: after making a posterior,
call `get_fantasy_model` with a leading fantasy dimension and then call
`posterior` on the fantasy model. At `b0dd01be6`, the fantasy cache is a
generic `AddedDiagLinearOperator`, while `HigherOrderGP.posterior` assumes
the Kronecker-specific `.linear_op` attribute (line 503), raising
`AttributeError`. Adding a fallback to `.linear_ops[0]` alone then exposes
a `BatchRepeatLinearOperator`, which also does not expose Kronecker factors
through `.linear_ops`. A complete upstream solution should extract/rebuild
the individual covariance factors through these wrapper types and expand
query batches before reshaping posterior outputs. Native fantasy inference
uses the actual appended training data and the existing factorized native
GP. Tests compare the failing official fantasy's training data/parameters
against an independent dense Gaussian posterior formula, and compare
unbatched official fantasy predictions directly where the official method
works. This is a documented reference limitation, not runtime delegation.
