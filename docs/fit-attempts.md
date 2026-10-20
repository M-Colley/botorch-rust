# Selecting the best fitting attempt

`fit_gpytorch_mll(..., pick_best_of_all_attempts=True)` supports native
`SingleTaskGP` families, including independent output and training batches.
It performs the requested `max_attempts` fits and retains the successful
attempt with the smallest negative marginal log likelihood. The first
attempt starts from the original parameters. Each later attempt restores
the original state and draws registered priors through the existing native
sampler. Parameters without priors retain their original values.

Both native L-BFGS-B and `optimizer=fit_gpytorch_mll_torch` (native Adam)
are supported. Sum marginal likelihoods retain their existing independent
submodel fitting semantics. Numerical failures discard the trial; if all
attempts fail, the original parameters are restored and `ModelFittingError`
is raised. Unexpected errors also restore the original state before
propagating. A nonpositive attempt budget is rejected before mutation.

For L-BFGS-B, `_fit_result` records the total attempt count, selected attempt
and independent summaries. Adam retains its `OptimizationResult` type;
both expose `_fit_attempt_results` and `_fit_selected_attempt` on the MLL.
These summaries do not contain recursive references.

The port's optional `seed` fixes restart samples; native and Torch random
generators produce different draws from equal integer seeds. Tests compare
the result with independently executed native restarts for list, NumPy and
Torch models, scalar and multiple outputs, and both native optimizers. A
separate official-source comparison supplies identical restart values to
both implementations and compares their fitted parameters and likelihoods.
Failure injection checks rollback after all unsuccessful attempts.

Best-of-all fitting for specialized non-SingleTaskGP model families remains
unsupported. The default first-success fitting behavior is unchanged.
