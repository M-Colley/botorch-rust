# Training data and optimization diagnostics

`extract_targets_and_noise_single_output` and
`restore_targets_and_noise_single_output` operate in the model's transformed
outcome space. Their public arguments use trailing single-output columns.
Native target/noise replacement preserves fitted transforms and current
hyperparameters, updates original-space bookkeeping, and invalidates the
factorization cache. Existing posterior objects retain their snapshots.
Both arrays are validated before a native replacement, so an invalid target
shape cannot leave the model with partly updated fixed noise.

The native replacement supports the existing number of observations and
model batch/output dimensions. `strict=False` does not authorize changing
those native dimensions. Adding observations uses
`condition_on_observations`; replacing inputs and changing observation
counts in place remain separate work.

`get_data_for_optimization_help(model, path)` writes local JSON diagnostics.
For a Python model implementing `state_dict`, its output matches BoTorch's
JSON layout. Native models export their actual named raw parameters under
`state_dict`, plus an explicit `native_info` section containing component
descriptions and original/transformed training arrays. This section makes
the native representation visible; the file is not a cross-library model
checkpoint. JSON serialization completes before an existing destination is
opened. The function does not upload or submit the data anywhere.
