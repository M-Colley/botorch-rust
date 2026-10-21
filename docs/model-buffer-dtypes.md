# Logical model buffers

`Model.dtypes_of_buffers` now inspects real registered state. The Python
`named_buffers()` / `buffers()` protocol exposes native state as Torch tensor
snapshots, restoring its caller-visible dtype rather than Rust's internal
CPU float64 storage. Inspection imports Torch when requested; importing the
port and constructing a model from lists does not import Torch.

The descriptors read current Rust prior parameters, constraint bounds,
standardization statistics, affine input-transform coefficients, kernel
active dimensions, and PairwiseGP's registered Laplace/data buffers. Native
module construction records argument and default floating dtypes before
scalar conversion erases that information. Binding a component to training
data records the same floating-buffer conversion as BoTorch's `Module.to`;
integer indices and Boolean flags retain their dtypes. Shared prior objects
retain their own conversion state, including later casts by another model.

ModelList children, deterministic wrappers, supported transforms, and basic
kernel/prior/constraint children are traversed recursively. Actual Torch
module children delegate to Torch's real registered buffers. Custom port
models can call `register_buffer`; ordinary tensor attributes, parameters,
and plain lists of modules do not become buffers automatically. Native
ModelList and transform container representations are recognized explicitly.
Prefixing, recursion control, `None`, nonpersistent buffers, and duplicate
module/tensor filtering are supported.

Ordinary exact GP training data, fixed observation noise,
`EnsembleModel.ensemble_weights`, and `FixedSingleSampleModel.w` are not
registered buffers in official BoTorch and are excluded. PairwiseGP is an
exception: its documented training/Laplace tensors are registered buffers.
Standardize contributes its Boolean `_is_trained` flag. Normalize/Warp/Round
can contribute integer index buffers. GPyTorch's Uniform and LKJ priors
register no own buffers; LKJCovariancePrior still traverses its actual child
prior modules.

This is a read protocol, not a full Torch Module/state-dict implementation.
Editing a returned native snapshot does not mutate Rust state. Use the
component's existing supported setters for mutation. Exact/fixed-noise
SingleTaskGP, PairwiseGP, standard deterministic/cost/ensemble wrappers,
ModelList, basic kernels, supported transforms and univariate/LKJ priors have
concrete descriptors. Special native model/kernel states that do not yet
have descriptors raise `UnsupportedError`; they do not return an incomplete
or fabricated empty dtype set. In particular, InputPerturbation's cached
expanded perturbations are not exposed by its native handle yet, so its
buffer inspection raises explicitly. HigherOrder, latent/variational and
empirical model inspection also remains outside the concrete model mapping.

The audit also corrected three dtype bugs in the port: affine deterministic
offsets now retain their own tensor dtype when the weights have another
dtype (including output subsetting); affine cost weights retain the default
dtype from construction; Beta concentrations use Torch's promoted dtype
when their input tensors have different dtypes.

`botorch_pyo3/tests/test_model_buffers.py` checks buffer names, dtypes and
values against official upstream revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`, with float32/float64 GP inputs,
mixed integer/Boolean buffers, mixed priors, custom Torch modules, aliasing,
containers, transform states, NumPy inputs, and lazy-import behavior. No
upstream bug was identified in this buffer audit.
