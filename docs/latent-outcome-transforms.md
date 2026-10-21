# General LatentKronecker outcome transforms

The Tensor backend accepts source-supported outcome modules beyond scalar
`Standardize`: `Log`, `Power`, `Bilog`, chains, custom differentiable affine
modules, and input-dependent outcomes where their declared feature/outcome
shapes are compatible. The actual supplied module remains attached to the
model. Training targets are transformed as observed scalar events; missing
grid cells are excluded before calibration.

The posterior retains the structured Matheron base-normal layout and grid
sample shape. A scalar event mapping applies the outcome inverse after the
Gaussian draw. `Log`, including a `Log` followed by `Standardize` chain,
provides its analytic raw-space mean and marginal variance. Affine modules
can expose their transformed Gaussian distribution. `Power` and `Bilog`
provide samples; analytic moments remain undefined, matching the outcome
transform API's `NotImplementedError` boundary.

Posterior snapshots clone fitted buffers and transform parameters through
ordinary Tensor operations. Functional calls use that snapshot for samples
and moments, so refitting or mutating the live transform cannot change an
existing posterior's law. These clones preserve derivatives to the original
training data and supplied Parameter leaves. They do not replace model
Parameters or change their identity.

Conditioning and the port's configured-noise fantasy extension support these
outcomes. The observation likelihood is applied in transformed outcome units,
then the actual outcome inverse maps the sampled observations back to raw
units before conditioning. No extra random draws are introduced outside the
supplied sampler. Input context is supplied during calibration, training
transformation, and inversion; the supported stratified source case with one
time location is tested.

Pinned BoTorch revision `b0dd01be6` applies the inverse outcome map to samples
but leaves its posterior distribution in transformed units. Its sampler also
loses terms of the conditional covariance. These source bugs are already
covered by [BoTorch PR #3488](https://github.com/meta-pytorch/botorch/pull/3488),
which includes nonlinear outcome moments; no duplicate PR was filed. The
port uses consistent raw-space moments and the complete Gaussian conditional.

Validation compares transformed targets and Gaussian evaluations with the
unchanged pinned source. Independent dense Gaussian conditionals and full
joint Matheron constructions check values, samples, every training/query/time
and model/transform leaf derivative, and mixed Hessian products on CPU/CUDA.
The same tests cover model batches, missing cells, snapshot stability,
conditioning, fantasies, and source-valid stratified input context.
