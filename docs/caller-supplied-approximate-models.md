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

`ApproximateGPyTorchModel(model=None, train_X=X, backend="torch", ...)` also
constructs the port's actual Tensor variational model. CUDA or differentiable
inputs/inducing locations, input batches, supplied Torch components and supported
alternative variational configurations select that path automatically. Ordinary
native/list construction remains available. Its generic constructor binds the
inner model's arguments, ignores `train_Y` as documented by the source, and
retains the explicit `num_outputs`. Default Gaussian likelihoods have no
likelihood prior, matching this generic source wrapper rather than the separate
`SingleTaskVariationalGP` defaults. Train against explicit targets or a closure/
loader; constructor targets are not retained or used to infer the output count.

The default Tensor owner, its inner model and variational strategy are the
actual objects used for inference and ELBO fitting. Deep copies and pickle
round trips restore their owner links, modes, leaves, buffers and module aliases.
Attached transforms update through their ordinary train/eval forward protocol;
the generic wrapper does not fit them to the unused constructor targets.

Caller subclasses of both public approximate model constructors retain their
constructors, forward/posterior overrides and additional Parameters when the
Tensor backend is selected. Native built-in allocation and already concrete
Tensor classes keep their own factory paths.
