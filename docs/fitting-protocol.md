# Fitting protocols

`fit_gpytorch_mll` supports caller-supplied optimizer and gradient closures,
closure keyword arguments, warning handling, caught exception policies,
prior resampling, best-of-attempt selection, and state rollback. Frozen
parameters survive prior resampling. A successful fit restores the selected
attempt and enters evaluation mode; failed attempts restore the original state.
The native default exact-GP fitter retains its Rust L-BFGS-B fast path.
Advanced options and Torch models use the generic fitting protocol.
Native generic SciPy fitting expands bounds using the complete parameter shape
and retains the Rust parameter state through caller closures and closure keyword
arguments. Generic Torch fitting bridges list and NumPy parameters to optimizer
leaves, writes the returned gradients, and commits the final values to Rust.

`ForwardBackwardClosure` leaves the returned gradients in the actual
Parameters' `.grad` fields, matching the reference contract. This matters for
Torch optimizers, which read those fields when stepping. The former port
cleared them before returning and could therefore report successful fits
without changing parameters.

CPU/CUDA regressions check actual SGD updates, unused-parameter gradients,
custom callbacks, warning and exception retries, frozen leaves, closure
arguments, best-attempt state selection, and rollback after failure. These
checks cover both native exact models and Torch models. Broader fitting,
variational, optimization, and parameter-gradient regressions passed 117 cases
before the additional nonfinite-objective and iteration-limit policy cases.
A further 59-case run covers the generic fitting and minibatch closure suites,
including 18 native exact/variational array-kind regressions. These verify real
parameter changes and objective improvement with callbacks and custom closures;
they caught and repaired truncated bounds and a missing Rust-state connection.

External-data closures now also evaluate Tensor exact MLLs and caller-supplied
model/MLL protocols, including multiple model inputs, additional targets and
closure keyword arguments. Iterators restart after an epoch and reject malformed
or empty batches. Eight new CPU/CUDA cases compare external exact loss values,
complete gradients and directional Hessians and exercise actual optimizer steps.
The Hessian oracle uses central differences of the reference's first derivatives:
its optimized RBF kernel loses a lengthscale derivative term in double backward.
Native exact MLL closures continue to use stored data; select `backend="torch"`
for the external-data exact-GP interface.

KL-prior fitting, EM historical-data fitting, and shared-parameter model lists
also follow this protocol. Their native gradient closures call the actual KL,
EM, or joint shared objective, rather than substituting an ordinary exact-GP
likelihood. Native default L-BFGS-B remains available without SciPy. Supplied
optimizers, closures and closure keyword arguments, parameter selection, bounds,
callbacks, schedulers, warning policies, retries and best-attempt selection use
the generic drivers. SciPy methods require the optional SciPy dependency;
Torch optimizers require Torch. Native Tensor descriptors are bridged to actual
optimizer leaves and synchronized with Rust, including caller-supplied closures.

Shared lists are fitted as one joint objective over the global parameter layout,
with one mean/kernel and separate likelihood noises. Ordinary independent model
lists retain per-child fitting. Restart draws preserve sharing and explicitly
fixed noises. A failed native EM attempt restores the complete native model copy,
including warm EM moments; Tensor models use their actual registered state.
Native higher-order and latent-Kronecker best-of-attempt fitting uses their
native prior sampler and the same generic selection/rollback policy.

The specialized extensions are covered by 47 focused CPU/CUDA and native tests,
including actual SGD/BFGS optimization, independent native restart comparisons,
the pinned reference's KL fitting protocol, shared identity, and failed warm-EM
rollback. A further 259 existing fitting, closure, variational and EM tests pass.
With reference libraries and SciPy blocked, the focused CPU suite passes 26 tests
and skips eight dependency-specific cases. Existing model restrictions remain:
native KL models are unbatched and single-output with fixed reference locations,
and EM objectives use their fixed historical datasets. External-data loaders are
not substituted for those historical datasets. Models without a native
prior-sampling protocol still reject automatic resampling instead of silently
repeating the same initialization.
