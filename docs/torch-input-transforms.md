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

This matrix describes the tested direct-call scope; it is not a claim that
every input-transform model configuration is available on the Tensor backend.
The next checkpoint expands the model builder, which previously converted only
Normalize/Warp/chains, and checks transformed feature widths and training/query
row semantics separately from these direct-call results.
