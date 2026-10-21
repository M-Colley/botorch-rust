# General Kronecker and higher-order outcome transforms

Tensor `KroneckerMultiTaskGP` and `HigherOrderGP` accept source-supported
nonlinear and custom outcome modules in addition to their default
standardization. `Log`, `Power`, `Bilog`, chains, and custom differentiable
affine transforms retain the actual supplied public port module and its
registered Parameters. Higher-order chains can include public or canonical
source `FlattenedStandardize` children.

The existing full Gaussian conditional and Matheron base-normal layout are
preserved. Outcome mappings act on the task event grid for Kronecker models
and the complete target-grid shape for higher-order models. Log mappings
provide analytic raw-space moments; Power and Bilog retain their sampling
API and explicit unavailable-moment errors. Selected nonlinear Kronecker
outputs and repeated output subsets keep their numerical and gradient laws.

Functional parameter/buffer snapshots preserve the posterior's fitted
outcome law when the live transform is refitted. Snapshot clones retain
derivatives to the original data and transform leaves. Conditioning and
fantasies use the same mapping and structured sampler, including higher
derivatives through query locations and original/future model Parameters.

Explicit higher-order conditioning noise is supplied in raw outcome units.
Re-evaluation transforms it alongside the actual appended raw observations;
it never substitutes zero-valued observations for a nonlinear delta-method
variance map. Both learned-noise and supplied fixed-noise paths retain the
raw observation/noise derivative graphs. Kronecker conditioning noise keeps
its existing outcome-transformed-unit contract.

`test_torch_grid_outcomes.py` covers CPU/CUDA, model batches, source target and
available-moment controls, every model/data/transform leaf first and mixed
second derivative, selected outputs, saved posterior stability, conditioning,
fantasies, and nonlinear conditioning-noise conversion. The suite also runs
under Torch 2.14 and with BoTorch/GPyTorch and probabilistic runtimes blocked.
The source's posterior observation-noise override bugs remain tracked in
[BoTorch PR #3396](https://github.com/meta-pytorch/botorch/pull/3396); the port
keeps their explicit unsupported boundary rather than silently ignoring a
Tensor override.
