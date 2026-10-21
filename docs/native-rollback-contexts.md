# Native model rollback contexts

`module_rollback_ctx` checkpoints and restores the raw parameters of exact GPs
and variational GPs with a fixed native parameter layout. Pending Torch leaf
edits are synchronized before the checkpoint. Direct leaf edits, public setters
and native fitting inside the context are rolled back on normal and exception
exits. Restoration updates held parameter leaves in place and preserves their
identities and trainability flags. Saved native loss graphs retain their own
source snapshots.

Tracked pending leaf edits may contain NaN or infinity: their checkpoint values
replace those edits before native validation, so restoration still completes and
an original exception from the body is preserved. With a partial checkpoint,
valid pending edits to untracked parameters are committed alongside the restored
tracked values. Non-finite untracked edits cannot be preserved in native storage;
they raise an explicit validation error after restoring tracked values and
committing valid untracked edits. Invalid untracked leaf edits remain pending
for the caller to repair or include in a larger checkpoint. Checkpoint values
themselves must be finite.

`name_filter` limits the saved parameters. Passing an explicit checkpoint uses
those values, and changes to its `TensorCheckpoint.values` determine the restored
state. Passing an empty checkpoint or clearing the yielded checkpoint commits
changes; pending Torch leaf edits are synchronized to native storage on exit.
Torch checkpoint copying honors `dtype` and `device` arguments. Lists and NumPy
arrays retain their native descriptor and restoration behavior.

This is raw-parameter rollback, not full model-state serialization. It does not
restore changed training data, transforms, lazy initialization buffers or model
structure. Sparse support or other parameter-layout changes cannot be recovered
from the raw vector alone and raise an explicit unsupported error when
restoration is requested. Simultaneously conflicting direct native and Torch
edits retain the model's existing synchronization error; callers should use
public setters or synchronize between those edits.

Tests cover both model families and all array kinds, pending entry edits,
direct in-context leaf edits, public setters, native fitting, normal and exception
exits, filtered and caller-edited checkpoints, explicit commit semantics,
checkpoint dtype conversion, immutable old graphs, and sparse support resizing.
