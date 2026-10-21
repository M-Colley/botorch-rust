# Fitting protocols

`fit_gpytorch_mll` supports caller-supplied optimizer and gradient closures,
closure keyword arguments, warning handling, caught exception policies,
prior resampling, best-of-attempt selection, and state rollback. Frozen
parameters survive prior resampling. A successful fit restores the selected
attempt and enters evaluation mode; failed attempts restore the original state.
The native default exact-GP fitter retains its Rust L-BFGS-B fast path.
Advanced options and Torch models use the generic fitting protocol.

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

This work does not imply support for every model-specific fitting extension.
Specialized empirical models and external-data closure configurations still
need their separate behavior audit.
