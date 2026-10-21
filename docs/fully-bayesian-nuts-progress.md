# Fully Bayesian NUTS iteration reporting

`fit_fully_bayesian_model_nuts(..., disable_progbar=False)` now reports each
completed warmup and sampling transition to standard error on both native
and Tensor CPU/CUDA backends. The report includes the transition count,
acceptance probability, leapfrog-step count and cumulative divergences.
`disable_progbar=True` suppresses it. The optional NumPyro custom-program
bridge continues to use NumPyro's own progress reporting.

The native core's existing `sample_nuts` entry point retains its signature.
A separate `sample_nuts_with_callback` calls a fallible callback between
transitions. The binding and Tensor sampler expose the same optional
`progress(completed, total, warmup, accept_prob, num_steps, diverging)`
callback. It observes diagnostics without changing states or consuming
random numbers. Python callback exceptions, including cancellation, stop
sampling; the progress context closes the output line on every exit. The
native fitter now waits until sampling succeeds before resetting a loaded
model, so interruption preserves its previous posterior.

These diagnostics require scalar synchronization on CUDA; the NUTS state,
momenta, covariance estimates and mass-matrix algebra remain on their
existing backend. Progress reporting does not compile the eager Tensor
sampler. The accepted legacy `jit_compile` flag remains the pinned
reference's no-op.
