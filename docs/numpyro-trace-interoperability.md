# Optional NumPyro trace interoperability

Audited against official BoTorch revision `b0dd01be6`.

The public `sample` and `sample_*` methods on `MaternPyroModel`,
`SaasPyroModel`, `LinearPyroModel`, `MultiTaskPyroMixin`, and
`LatentFeatureMultiTaskPyroMixin` now define actual NumPyro programs.
They use the official named stochastic/deterministic sites, priors, inferred
noise floor, known-noise behavior, latent task embeddings, and observation
likelihoods. NumPyro `handlers.seed`, `trace`, and `substitute`, `log_density`,
JAX `jit`/`value_and_grad`, and NumPyro MCMC operate on these programs.
`_prior_mode` creates the `f` and unobserved `Y` sites, including saved
`f_prior_sample` / `Y_prior_sample` values; empty training targets omit `Y`.

JAX/NumPyro imports are lazy. Importing the package, constructing a model,
and running the native potential do not import JAX, NumPyro, or Torch.
Calling a trace method explicitly requires installed `jax`, `jaxlib`, and
`numpyro`. Training arrays are converted lazily to JAX arrays for this
interface. JAX precision follows the caller's `jax_enable_x64` setting;
the adapter does not change global precision settings.

`compute_dists`, `matern52_kernel`, and `linear_kernel` retain native Rust
execution for list, NumPy, and Torch arrays. Explicit JAX arrays/tracers
use JAX operations so traced values and their derivatives are preserved.
This is optional framework interoperability, not an inference fallback:
`fit_fully_bayesian_model_nuts` still uses the native Rust potential and
native NUTS. A custom public sampling-method override works with NumPyro
directly; native fitting now explicitly rejects overrides it cannot
represent, rather than silently ignoring them. Native fitting likewise
does not implement the trace's prior-predictive mode.

For an explicit NumPyro run, use
`numpyro.infer.MCMC(numpyro.infer.NUTS(model.pyro_model.sample), ...)`,
then `model.pyro_model.postprocess_mcmc_samples(mcmc.get_samples())` and
`model.load_mcmc_samples(...)`. The loaded model's GP predictions remain
native. The port's existing postprocessing correction adds the inferred
noise floor before loading, keeping the sampled likelihood and predictions
consistent.

The trace preserves the port's verified input-warp correction: rescale to
the interior interval and apply `1 - (1 - x**c1)**c0`, matching the fitted
native `Warp`. Official `b0dd01be6` uses reversed concentrations and omits
the rescaling in its NumPyro program. For multitask traces with default
warp indices, the adapter selects all non-task columns; the official
`range(ard_num_dims)` selection can warp the task column and omit a real
feature when the task column is not last. Native fully Bayesian multitask
fitting and prediction now use the same per-sample transforms; see
[the native multitask warping audit](fully-bayesian-multitask-warping.md).
Direct multitask Pyro inputs require task indices in `[0, num_tasks)`;
the fully Bayesian GP wrapper maps external task labels automatically.
Invalid direct identifiers raise clearly rather than allowing JAX's
out-of-range indexing to silently clamp task assignments.

Another verified official bug affects prior mode with known noise. Given
`train_Yvar.shape == (n, 1)`, the official `Normal(loc=f, scale=sqrt(noise))`
broadcasts a length-`n` latent vector with `n × 1` scales into an `n × n`
`Y` site. Reproduce with a `MaternPyroModel` configured with known noise,
set `_prior_mode = True`, then inspect the seeded trace's `Y.value.shape`.
Flattening the known per-observation scales to length `n` fixes the site.
This adapter implements that fix. Reference tests apply this correction
and the documented warp corrections explicitly; all other named sites,
values, priors, and densities compare to the unchanged official program.

Validation covers all four concrete model programs, inferred/known noise,
warping, seeded/substituted traces, JIT density gradients, prior mode,
empty targets, custom overrides, JAX kernel-helper adjoints, optional-import
isolation, unconstrained potential/gradient parity with native Rust, and a
real short NumPyro MCMC run loaded into a native GP.
