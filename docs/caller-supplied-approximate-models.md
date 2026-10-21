# Caller-supplied approximate models

`ApproximateGPyTorchModel(model=module, likelihood=likelihood, num_outputs=m)`
accepts an actual Torch module on CPU or CUDA. The wrapper registers that exact
module and likelihood, preserving their Parameters, buffers, state, modes,
subclass methods and device/dtype migration. Its forward and posterior methods
execute the caller's module; the port neither imports nor constructs a BoTorch
model. `num_outputs` remains explicit, matching the source wrapper.

Scalar, independent multitask and model/query-batched Gaussian outputs retain
their actual distributions. Other distributions use `TorchPosterior`. The
supplied likelihood executes for observation-noise predictions, including
non-Gaussian likelihoods. Attached Torch input/outcome transforms remain actual
modules; port Normalize, Warp, chained input transforms, Standardize, Log, Bilog,
Power and chained outcome specifications have device conversions. Fitted
transform state remains fixed during posterior evaluation. Posterior transforms
retain their graphs. Output-index selection and Tensor-valued observation noise
remain unsupported by this wrapper, as in the pinned source implementation.

`VariationalELBO(likelihood, model, num_data, beta, combine_terms)` accepts the
actual supplied module with a variational strategy. It uses its
`kl_divergence()`, the likelihood's `expected_log_prob()`, registered priors and
added loss terms. The MLL retains the original model identity. Explicit outputs
and targets, a loss closure, or a data loader drive training when that module
does not store training data. Actual Torch and SciPy optimizers update those
leaves; retry rollback includes a separately supplied likelihood. Untransformed
GPyTorch constraints retain scalar or per-element Tensor bounds and their
device. Transformed constraints act through the caller's parameterization;
their constrained-space bounds are not incorrectly imposed on raw leaves.
SciPy transfers flattened optimizer state and bounds to NumPy at its explicit
interoperability boundary. Posterior and ELBO Tensor computations stay on device.

The caller owns its module's mathematical law and derivative implementation.
Consequently, source custom-autograd/operator defects in that module remain
visible through this adapter; it does not substitute another model or silently
detach those graphs. The tests compare values, first derivatives and higher
derivatives with the same actual modules and the pinned source wrapper/ELBO,
alongside independent port-only modules, state and host-transfer checks.

This adapter targets an explicitly supplied module. Default port variational
model construction remains available through `SingleTaskVariationalGP`; this
checkpoint does not add a new Tensor default model for the generic
`ApproximateGPyTorchModel(model=None)` constructor.
