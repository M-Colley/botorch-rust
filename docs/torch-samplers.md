# Public sampler module protocols

When Torch is loaded, the public Monte Carlo sampler classes construct lazy
`nn.Module` subtypes. They retain their public `isinstance` relationships and
execute their actual `forward` methods through Torch's hook machinery. The
sampler bases support caller subclasses using `super().__init__`, including
custom normal-base construction, supplied Parameters and child modules. Custom
`__call__` and `__setattr__` overrides remain in effect.

This covers `MCSampler`, `NormalMCSampler`, `IIDNormalSampler`,
`SobolQMCNormalSampler`, `PairwiseMCSampler`, `PairwiseIIDNormalSampler`,
`PairwiseSobolQMCNormalSampler`, `IndexSampler`, `ForkedRNGSampler`,
`StochasticSampler` and `ListSampler`. The standalone QMC engines remain plain
engines, as in the source. Importing the package or constructing native Sobol
samplers with list inputs does not require importing Torch or NumPy.

Tensor base samples are registered under the source name `base_samples`.
Buffers participate in parent state, train/eval mode, dtype/device migration,
deepcopy and whole-object Torch/pickle serialization. Caller base-sample
Parameters retain their actual identity and complete derivative graph. A
`ListSampler` registers its actual children in a `ModuleList`, including caller
Torch samplers, and forwards baseline-prefix updates to each child. Public
Gaussian and ensemble posterior protocols are accepted without requiring
private port markers. Pairwise comparisons select integer preferences on the
sample device, preserving the documented preferred-item-first repair.

Cache migration preserves existing normals while their shape is unchanged,
matching the source. Default seed generation consumes the same Torch random
draws as the source, including the pairwise constructors. Native CPU sampling
retains its Rust sampler; serialization rebuilds that disposable native cache
from its seed and available public normals. Source-style strict state loading
requires materializing a previously `None` base-sample buffer first.

Tensor acquisitions retain supplied sampler objects, their registered leaves
and custom numerical methods. Independent cost/fantasy copies preserve custom
constructor state through deepcopy. The native acquisition bridge still
consumes its Rust normal sampler rather than arbitrary overridden callbacks.
It explicitly rejects custom methods, Parameters, live base samples and hooks
with guidance to construct a Torch-backend model, so caller behavior cannot be
silently discarded. Canonical CPU normal descriptors retain their supplied
public base-sample cache, including actual source samplers. Registering samplers
as modules does not remove that explicit native backend boundary.

The focused regression module includes 55 CPU/CUDA module cases against source revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`, including actual source sampler
modules, all registered state, migration, loading, serialization, custom
callbacks, Parameters, hooks and independent first/mixed second derivatives,
plus native rejection and explicit-cache controls.
Source-independent cases also run with BoTorch, GPyTorch, linear_operator and
SciPy blocked. This certifies these configurations without asserting every
possible posterior layout or a sampler speed improvement.
