# Direct public input-transform Tensor calls

Two device checkpoints cover direct calls to fifteen concrete families in
`models.transforms.input`. Tensor numerical calls use port-owned Torch operations;
list/NumPy calls and explicit native model descriptors retain the Rust path.
Torch is not imported by an array-only program.

| Family | Direct Tensor behavior verified | Status |
| --- | --- | --- |
| AffineInputTransform | selected columns, batch broadcast, reverse, fixed coefficients | Implemented |
| Normalize | learned/fixed bounds, selected columns, batch reductions, center, inverse, preprocess, native state export | Implemented |
| InputStandardize | learned means/unbiased variance, batch reductions, inverse, constant-slice no-op | Implemented |
| Warp | selected normalization, epsilon rescaling, forward/inverse, batch concentrations, live Parameters | Implemented |
| Round | smooth approximation and deliberate integer/one-hot straight-through derivatives | Implemented |
| Log10 | selected log/power maps and reverse | Implemented |
| FilterFeatures | declared physical column order | Implemented |
| InteractionFeatures | selected pair products and unchanged original features | Implemented |
| AppendFeatures | fixed/callable/one-to-many feature expansion and skip-expand layout | Implemented |
| OneHotToNumeric | categorical reduction and inverse | Implemented |
| NumericToCategoricalEncoding | actual caller encoders, including live embedding parameters, and feature insertion | Implemented |
| LearnedFeatureImputation | persistent raw parameters, sigmoid constraints, target-task insertion and observed/missing task masks | Implemented |
| InputPerturbation | fixed/callable additive/multiplicative expansion, partial-feature selection and bounds | Implemented |
| ChainedInputTransform | child flags, order, inverse and preprocessing | Implemented |
| BatchBroadcastedInputTransform | device split/stack and child paths | Implemented |

Affine fitted statistics follow the reference's deliberate graph policy:
learning in training mode keeps their input graph; evaluation and preprocessing
use detached fitted statistics. Fixed supplied affine coefficient/offset
buffers are also detached, as in the source. Input derivatives remain live.
Model adapters separately retain their complete training-data graph policy.
Constant standardized columns use the no-op branch without differentiating an
inactive square root at zero. The existing native singleton no-op remains;
the source's undefined unbiased singleton standard deviation is not used as a
finite derivative oracle.

Warp concentrations become persistent Torch Parameters on direct Tensor use.
An explicitly supplied Parameter retains its identity; initialize/setter value
updates copy into the existing leaf. The Tensor GP adapter shares those actual
parameters with the caller's transform. Numerical calls do not export data to
native arrays; exporting state to an array call or native model is an explicit
boundary. Fitted normalization and concentration values survive that boundary.

Native owner updates also refresh a Warp's existing direct concentration leaves
when read: MLL trial values and fitted values are authoritative, and held
Parameter identities remain unchanged. Selected-column Normalize objects export
their current Tensor coefficient/offset pair to the native descriptor used by
MultiTaskGP, as well as to generic native chains. These two state-boundary
regressions are checked independently and against the original eleven failing
native Pairwise/fully Bayesian multitask parity cases.

Forty-six CPU/CUDA cases compare values, fitted state, first derivatives and
mixed Hessian-vector products against official BoTorch
`b0dd01be61d517c387e234903a6e2cc38c057ea2`. They include reversed/selected/batched
affines, full Warp input/parameter graphs, straight-through rounding, inverses,
mode/fantasy flags and strict CPU/NumPy/list-conversion guards around constructor,
numerical calls and model integration. All 34 native-array transform tests and
257 selected native/Tensor module/model parity regressions also pass. Torch
2.14.1 CPU passes 23 new cases and skips 23 CUDA cases. Ruff and Mypy pass.

The second checkpoint adds 38 CPU/CUDA cases for fixed/callable features,
skip-expand ordering, fixed/callable perturbations, batch and selected-feature
layouts, additive/multiplicative operations, bound graphs, categorical
reductions/inverses, caller embedding graphs, imputation parameters and masks,
chains and batch wrappers. Full first/higher derivatives are compared with the
actual source; the imputation source is constructed in its supported global
double dtype so GPyTorch Interval does not insert a float32 bound-gradient
intermediate. Strict host-conversion guards cover constructors and calls;
array-export and equality checks read current Tensor values. Torch 2.14.1 CPU
passes 19 cases and skips 19 CUDA cases.

A probe of the original port implementation at `cd25bd9` confirms detached
outputs and CPU/list conversion for all eight initial families on both CPU and
CUDA. This is a port-only gap, repaired in the main port PR; it is not a new
upstream finding.

The model bridge now registers the actual public input transform in a Torch
module shell, including all fifteen families above and caller subclasses. It
calls the original numerical hooks, rather than substituting a canonical
implementation for a subclass. Actual Warp/imputation Parameters, callback
modules and fitted/static Tensor buffers stay registered; nested aliases retain
one module/parameter/prior owner. Model training-data statistics are rebuilt
with their full graph, while initially frozen affine statistics retain the
direct API's detached fitted-buffer policy. Device/dtype conversion, model
state dictionaries, deepcopy and actual callback parameter derivatives are
covered separately.

SingleTaskGP builds its default kernel from the transformed training feature
width. Query validation checks the resulting feature width, allowing declared
query expansion such as AppendFeatures with already augmented training inputs.
Training transforms must preserve the number of target rows. The no-fit
registration path accepts only a dtype/device reference and no fabricated
training inputs or feature width, for external approximate-model attachment.
CPU automatic selection also recognizes actual live transform leaves, feature
Tensors and callback modules; explicit `backend="native"` preserves native
selection.

The model tests compare each family's Gaussian posterior and complete
training-data/query/parameter first and higher derivatives with independent
closed-form transforms and Gaussian conditioning on CPU/CUDA. Eight families
also have direct official-model value/query-derivative controls. Feature-width
changes, one-to-many query ordering, frozen child modes, alias ownership,
parameter identity, strict host-conversion guards and registration without
fitting are tested explicitly. This describes the verified scalar model and
generic-builder scope; it does not establish every specialized model's
transform configuration. Eight families now have conditioning controls with
unmodified source models and independent dense Gaussian laws, including
transforms that change feature width. Layouts with different original raw
training/query widths fail the source's original-input concatenation too; an
AppendFeatures source boundary control records that behavior explicitly.

Whole-model pickle now retains every family's actual Tensor buffers, caller
callback modules and Parameters. Ordinary pickle memoization preserves shared
transforms and aliases to caller-owned modules/Parameters saved alongside the
model. Native input descriptors are regenerated from their primitive
configuration and cached statistics instead of being pickled. Univariate Warp
priors retain their exact native factory precision and Torch logarithm terms.
The resulting model supports posterior values, complete first/higher
derivatives and device/dtype conversion after loading. Seven unmodified source
model families independently establish that whole-object pickle is a supported
source operation. Caller callbacks still need to satisfy normal Python pickle
requirements, just as in the source.

The bridge checkpoint passes 68 new CPU/CUDA cases and 507 integrated
transform/module/Tensor-model cases. The existing native transform parity
selection passes 48 cases without tolerance changes. Torch 2.14.1 CPU passes
34 new cases and skips 34 CUDA cases; Ruff and Mypy pass for the six changed
implementation modules.

The persistence checkpoint adds 58 CPU/CUDA cases and two exact native prior
precision controls. All 271 selected persistence, bridge, direct transform and
prior cases pass. A separate selection covering persistence, the bridge and
empirical models passes 180 cases. Training row validation checks the actual
input transform against the original target rows, preserving empirical
subclasses that use a flattened correlated-output training representation.
Torch 2.14.1 CPU passes 29 persistence cases and skips 29 CUDA cases (before
adding the two native precision controls). It does not certify the complete
package.

Module encoders stored inside NumericToCategoricalEncoding's dictionary are
registered with the Tensor model too, retaining caller identity, optimizer
visibility, state, modes and device/dtype migration. The corresponding source
registration defect is proposed for repair in BoTorch #3522.

Scalar fantasize now enters the public fantasy context, so the default
InputPerturbation environmental expansion is skipped during sampling and
conditioning. With propagate_grads=False, only the training prediction factor
and alpha are detached during sampling; the sampled query path remains live.
The source checks explicitly enable gradient propagation and rebuild its
prediction cache for subsequent posterior comparisons. Both flags have
independent dense first/mixed higher derivative controls. The 280-case
integrated selection includes knowledge gradient, its initializers,
multi-step lookahead and supplied Gaussian likelihood conditioning.

Conditioned scalar and ordinary multitask models now copy their child modules
and Parameters, matching the source's independent ownership. Appended training
data remain connected to the original data and fantasy sample leaves. The
source controls compare the original and fantasy parameter partials separately;
they no longer sum the two hyperparameter paths. Updating a conditioned
hyperparameter, changing its modes or migrating its dtype leaves the original
model unchanged. Supplied fixed-noise likelihoods also copy their additional
learned noise Parameter; two unmodified source controls establish this behavior.
The older test that required aliasing that Parameter has been corrected.

The ownership checkpoint passes 413 integrated cases across input transforms,
scalar/multitask conditioning, supplied Gaussian likelihoods, empirical models,
knowledge gradient, its initializers and multi-step lookahead. Torch 2.14.1 CPU
passes 52 cases and skips 38 CUDA cases in the affected conditioning selection.
The standalone public classes now acquire an actual Torch Module protocol when
Torch is loaded. The cooperative class keeps the original public numerical
methods, actual caller Parameters and child modules. All fifteen families
support hooks, module/parameter/buffer traversal, device/dtype conversion,
state dictionaries, and whole-object pickle. Objects created before importing
Torch can acquire the protocol in place on an explicit module-method request,
preserving caller aliases. Array construction and calls still do not import
Torch. Unchanged CPU builtins remain eligible for native automatic selection;
actual caller Parameters, modules, numerical overrides and hooks select Tensor
inference.

Native affine snapshots stay observational until a Tensor call or explicit
migration, preserving the full precision of native array coefficients. The
source state keys, values and loading behavior are checked directly for ten
families, including Warp constraints and imputation masks. Cached fixed
InputPerturbation tensors now own writable storage, repairing an own-state
loading defect also proposed upstream in BoTorch #3525. The existing Warp
graph fixture now migrates both actual and reference before initialization;
four additional controls establish matching source factory precision when
initialization precedes migration.

The standalone checkpoint passes 608 integrated existing/new cases before the
last four factory-precision cases. The latest protocol and buffer selection
passes 96 cases; Torch 2.14.1 CPU passes 103 bridge/protocol/buffer cases and
skips 61 CUDA cases. Mypy passes all six changed implementation modules. This
does not establish complete package behavior parity.

Conditioning now stores the actual transformed coordinates of the old and
appended data. New coordinates are transformed once in the active
eval/fantasy phase before copying component Parameters; future queries use the
copied transform Parameters. This preserves skipped Warp coordinates and
expanded environmental draws, including when the transform's fantasy flag
differs from its training flag. Existing training-data transformation graphs
remain live as the Tensor extension's documented capability, while the source
caches its original training preprocessing under no_grad. The source controls
therefore compare query and copied-transform partials, and independent dense
laws verify complete original/copied-parameter mixed Hessians. Returning to
training discards the saved phase coordinates and rebuilds preprocessing on
the raw appended training data.

Ordinary multitask posterior layouts use the actual transformed point count,
preserve task-major and point/perturbation-major order, and accept repeated
categorical task labels in one-to-many transforms. Training noise uses those
same physical coordinates. Twenty-four source/dense CPU/CUDA cases cover these
paths; 22 fail against the frozen preceding backend. The integrated affected
selection passes 265 cases, and Torch 2.14.1 CPU passes 43 cases and skips 43
CUDA cases. The generic source fixed-noise multitask fantasy defect encountered
during these probes is already addressed in existing BoTorch #3374; supported
learned-noise source configurations establish the phase comparisons.

Ordinary multitask construction now derives its feature count, task column,
labels and default kernel from transformed training coordinates. Source-first
train-only FilterFeatures and AppendFeatures cases establish supported width
changes: eval queries already use the selected/augmented feature space. The
source itself classifies query task columns before transforming them, so this
does not claim arbitrary simultaneous training/query width relocation.
Four dense Gaussian controls verify complete data/query/parameter Hessians.
CPU auto selection routes configurations outside the native MT transform
support to Tensor inference, including non-Normalize inputs and nonlinear
outcomes. Explicit backend="native" retains its documented limits.

The remaining finite Input audit item is actual GPyTorch-specific
Warp/imputation prior/constraint calls. A native-only limitation is not a
missing Tensor implementation; source-supported configurations remain
available through the Tensor backend before claiming full package coverage.
