# Torch acquisition subclass protocols

Acquisition dispatch resolves the nearest registered public class in a user
subclass's MRO and verifies that class's actual exported identity. A user
class with the same name as a built-in acquisition therefore retains its
own behavior. The generated Tensor subtype remains an instance of the user
class and inserts the existing Tensor implementation before the native
constructor in cooperative `super()` calls.

User constructors, `forward`, posterior/moment hooks, and `_sample_forward`
methods run normally. Their parameters, buffers and child modules register
with `torch.nn.Module`; state dictionaries, deepcopy and pickle preserve
the user class. Explicit user pickle reducers take precedence. As with
ordinary Python classes, locally defined classes/callables need their own
serialization strategy.

Subclasses of the generic `AcquisitionFunction`, `AnalyticAcquisitionFunction`
and `MCAcquisitionFunction` bases can implement their own Tensor `forward`.
The generic module provides the model and detached pending-point protocol.
The analytic and MC bases additionally provide their existing moment and
posterior-sampling helpers. Missing required `forward` or per-sample utility
implementations remain abstract errors.

`SampleReducingMCAcquisitionFunction` subclasses evaluate their actual
`_sample_forward` and supplied q/sample reduction callables. User reduction
method overrides and cooperative calls to the default reductions are also
preserved. Callable identity is retained even for callable objects with a
false truth value. MC sample axes are reduced before the separate ensemble
reduction, retaining the source's behavior for nonlinear or sum reductions.
Constraint weighting and deliberate log-space behavior use the existing
Tensor calculations; no EI utility is substituted for a custom utility.

Fifteen focused CPU/CUDA cases exercise CPU graph-based and CUDA automatic
backend selection, cooperative constructors, posterior and sample hooks,
custom generic bases, supplied/replaced reduction methods, fully Bayesian
subclasses, abstract requirements, state changes and serialization.
Values and complete data/query/parameter first and second derivatives are
checked against explicit formulas and pinned official acquisition protocols
using the same port posterior. The tests also prohibit host conversions
during an analytic graph calculation. Torch 2.14 CPU passes eight available
cases with seven CUDA skips. The existing scalar acquisition suite passes
102 cases, and the combined subclass, analytic-PI, fully Bayesian list and
BALD regression set passes 83 cases.
