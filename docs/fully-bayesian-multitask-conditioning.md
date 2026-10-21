# Fully Bayesian multitask conditioning batches

Native `SaasFullyBayesianMultiTaskGP.condition_on_observations` now broadcasts
observation batches with the model's final MCMC axis. For `S` components,
inputs shaped `B x 1 x n x (d+1)` and targets shaped `B x S x n x 1` create
`B x S` conditioned models. Further leading fantasy axes can be added by a
second conditioning call. Native bordered updates group only observations
that share the same source model, inputs and noise; they do not reuse the
wrong source model's solve.

Conditioned native model metadata and direct prior evaluation retain this
axis order: fantasy axes precede the final component axis. `train_inputs`,
transformed `train_targets`, and public member-module lists describe the
conditioned data and native handles. Tensor fully Bayesian multitask
`batch_shape` and `_input_batch_shape` now reflect their actual conditioned
training tensors rather than always reporting only `S`.

Seventeen focused tests cover native list/NumPy/Torch conditioning, Warp,
Normalize/Standardize, nested fantasies, prior values, original-model
preservation and invalid-broadcast atomicity. Native query derivatives and
Tensor CPU/CUDA derivatives through new inputs, targets, noise and query
are compared with independent exact Gaussian conditioning of the official
prior; Tensor mixed Hessian-vector products use the same independent law.
The complete new and existing multitask/outcome regression set passes 209
cases against the immutable development ABI containing the batched binding.

The independent oracle is necessary because GPyTorch 1.15.2's
`DefaultPredictionStrategy.get_fantasy_strategy` infers `batch_shape` from
`full_inputs[0]`. Shared inputs can have batch `(B,1)` while a fully Bayesian
prior broadcasts to `(B,S)`. Its subsequent mean `view` incorrectly folds
the `S` components into the event dimension. In the one-observation fixture
the Schur system is `1 x 1` while its right-hand side incorrectly has 15
rows, and the official conditioning call fails. Using
`full_output.batch_shape` preserves the actual prior's batch shape. A scoped
reference-only prototype of that repair agrees with the independent dense
law to `1.78e-15` for means and `2.73e-16` for covariance. The production port
does not patch or call the reference implementation.

The source repair is submitted as [GPyTorch #2819](https://github.com/cornellius-gp/gpytorch/pull/2819),
against official main `e1302d24cbd334649ada70608cbe98e903a47a83`. Its new
regression fails all eight CPU/CUDA subcases before the repair and checks
means, covariances and fantasy-input/target/query gradients against dense
Gaussian conditioning afterward. The exact-GP, derivative-fantasy,
ModelList, Gaussian-likelihood and fixed-noise fantasy suites run 117 tests
successfully on Torch 2.13 CPU/CUDA (one existing expected failure), and on
Torch 2.14 CPU (one CUDA skip and one existing expected failure). The PR is
an open proposal; the pinned reference remains unchanged.
