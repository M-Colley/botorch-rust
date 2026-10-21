# Sequential acquisition sequences

`optimize_acqf(acq_function_sequence=..., sequential=True)` now optimizes one
actual acquisition per candidate through the existing native q=1 optimizer.
The sequence must have length q, with q > 1. Each acquisition keeps its own
base pending points; previously selected, postprocessed candidates are added
for its step. All original pending states are restored on success and on
errors in initialization, candidate generation, or evaluation.

Options, linear and nonlinear intra-point constraints, fixed features,
postprocessing, custom initializers and candidate generators are forwarded
to each step. The total timeout is divided by q. The result is q-by-d
candidates and q conditional values, or `None` for the values when
`return_acq_values=False`. Torch, NumPy and lists remain supported; callable
Torch acquisitions receive Torch pending tensors even when bounds use
another array representation. The sequence can supply all acquisitions
with `acq_function=None`.

As in official BoTorch, a supplied `batch_initial_conditions` array is
rejected for sequential optimization. Supply q=1 starts through
`ic_generator` and a `raw_samples` count. Inter-point constraints,
`return_best_only=False`, and one-shot acquisitions are explicitly rejected.
Native joint optimization is unchanged.

The audit reproduced three official upstream bugs at revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`:

1. `acq_function=None` with a valid sequence passes initial validation, but
   the first `dataclasses.replace` clears the sequence before selecting its
   first acquisition, causing `ValueError` before optimization.
2. A second-step exception leaves that acquisition's pending points
   augmented, because restoration only occurs after successful completion.
3. When all features are fixed, the early shortcut ignores the sequence,
   evaluates the primary acquisition on the entire joint batch, and returns
   one scalar instead of q conditional acquisition values.

The port corrects all three. The proposed upstream solution is
[patches/botorch-sequence-state.patch](patches/botorch-sequence-state.patch):
select the initial acquisition before replacing inputs, restore pending in
`finally`, and perform sequence dispatch before the all-fixed shortcut.
`git apply --check` succeeds against the pinned official revision. No
external reference files are modified by the port or the tests.

`test_optimize_sequence.py` includes live parity of distinct native
qSimpleRegret/qExpectedImprovement acquisitions, exact pending histories,
constraints/fixed features/postprocessing, user initializer keywords,
custom native candidate-generation callbacks and per-step timeout values,
all three bug reproductions, error restoration, and invalid configurations.
