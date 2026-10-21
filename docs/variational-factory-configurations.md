# Variational constructor configurations

`SingleTaskVariationalGP` and the generic default
`ApproximateGPyTorchModel(model=None, ...)` accept a supplied variational
distribution whose batch dimensions broadcast with the model's augmented
batch shape. Its parameters retain their original shape and identity; they
are not expanded into independently trainable copies. This includes a single
`q(u)` shared across model batches or output tasks, a singleton model batch
with distinct task distributions, and additional distribution batches that
broadcast with the kernel. Port Cholesky/MeanField/Delta specifications and
actual caller-supplied GPyTorch distribution modules follow the same rule.
The inducing event size must still match the number of inducing points.

ELBO targets broadcast to the resulting `q(f)` batch and event dimensions,
as in the Gaussian likelihood protocol. Independent multitask KL reduction
retains the reference strategy's actual reduction over its variational batch
axis, including singleton/shared distributions. It does not invent a second
copy of a shared distribution or multiply its scalar KL by the output count.

The regression module `test_torch_variational_shared_distribution.py`
checks scalar and multiple outputs, shared and singleton model/task batches,
additional distribution batches, source values and all first derivatives,
and first/second derivatives against an independent dense Gaussian formula.
It also checks posterior snapshots after later distribution parameter edits.
CPU and CUDA run the same tests; reference-dependent cases require the
optional reference packages, while port-specification dense-law cases do not.

## Caller-supplied strategy factories

A compatible callable `variational_strategy` now executes its actual
constructor. This includes subclasses, functions, callable objects and
`functools.partial` with extra constructor options or positional prefixes.
The returned strategy module, variational distribution, registered parameters,
buffers and virtual `forward` method are retained. Constructor-supplied
inducing points and learn-location options override partial keyword defaults
through the ordinary Python call protocol. A malformed factory return is
rejected explicitly.

The optional strategy host is a port-defined approximate GP whose Gaussian
prior uses the port's mean and covariance modules. It imports GPyTorch for
the caller's strategy-component protocol and never imports or delegates to a
BoTorch model. It wraps prior covariance in a dense linear operator without
factoring it eagerly: a joint prior containing repeated inducing/training
inputs can be singular before the strategy adds its intended jitter.
Port Gaussian likelihood descriptors gain the public `expected_log_prob`
protocol without changing parameter names or duplicating their leaves.
Actual supplied likelihood modules keep their identity. The ordinary
VariationalELBO module adapter supplies fitting, priors, KL and loss terms.

Canonical `VariationalStrategy`, `UnwhitenedVariationalStrategy`,
`BatchDecoupledVariationalStrategy` and `CiqVariationalStrategy` still use the
port's device-resident algebra. Arbitrary caller components retain their own
autograd behavior, including any limitations in external operations they
choose to call. In particular, a subclass using GPyTorch's optimized
`ComputePredictiveUpdates` inherits the upstream higher-derivative defect
reported in [GPyTorch #2824](https://github.com/cornellius-gp/gpytorch/pull/2824).
The new independent dense-Gaussian HVP controls use a caller-defined strategy
whose operations retain the full higher-order graph; they do not treat a
known broken external second derivative as an oracle.

`test_torch_variational_factories.py` checks source values and every first
derivative, custom strategy parameters and forward laws, independent dense
HVPs, actual Adam steps and public Torch fitting, Gaussian default descriptors,
module links, modes, state loading, deepcopy and pickle on CPU and CUDA.
Training data graphs remain shared across deepcopy, as with the existing
outer data buffers; module parameters and mutable state are copied. No hidden
owner shell or weak reference is introduced for the actual strategy host.

## Input-transform construction

Default kernel ARD width, inducing-point coordinates, target row validation,
and default inducing count use the actual transformed training inputs. The
public posterior still receives inputs in the original feature coordinates.
This permits feature-filtering transforms and training transforms that expand
rows when targets describe those transformed rows. Target model batches may
broadcast with the training-input batches. Source controls verify the changed
feature width, selected-feature gradients, actual training rows and allocator
count.

## Optional targets and outcome context

An outcome transform can be attached when `train_Y=None`. Stateless transforms
and already-calibrated Standardize transforms retain their inverse posterior
law without fabricating observations or fitting on input coordinates. An
uncalibrated Standardize raises the source's error when inverse-transforming
a posterior. The public Torch standardization module now enforces that check
on its direct posterior path as well as its value path.

Outcome fitting, explicit training targets and posterior inversion receive
the actual transformed input coordinates. This includes
`StratifiedStandardize` and chains containing it, with both canonical port
inference and actual caller strategy components. Regression tests compare
source values and query/module derivatives, and compare all data, query and
parameter gradients/HVPs with independent per-stratum formulas. The source's
NaN masked-calibration derivative is checked explicitly as the known
[BoTorch #3520](https://github.com/meta-pytorch/botorch/pull/3520) defect;
it is not used as a correct calibration-gradient oracle.

## Grouped large-batch prediction

`LargeBatchVariationalStrategy`, supplied either as a port specification or
as the actual canonical GPyTorch class, uses the port's grouped algorithm.
It factors the inducing covariance in float64, solves for the mean weights
and the inducing-sized covariance middle term, then multiplies those factors
by the cross covariance. None of its three triangular solves has one RHS
column per query point. Canonical inference does not import GPyTorch or
BoTorch. Caller subclasses still retain their actual component protocol.

Scalar training predictions retain the source's diagonal correction even
when the query has fewer rows than inducing points; evaluation, multiple
outputs and explicit `diag=False` retain the full correction. Float32 input
models use float64 grouped systems and cast the updates back as in the
source. All grouping and quadratic-form operations use ordinary Torch
autograd, including higher derivatives; they avoid the external cached
custom-backward defect reported in
[GPyTorch #2829](https://github.com/cornellius-gp/gpytorch/pull/2829).

Predictions snapshot their evaluated law and later calls reflect parameter
changes. The implementation recomputes its small grouped systems instead of
retaining detached inference factors across calls, so each call retains the
active parameter graph. The regression module
`test_torch_variational_large_batch.py` checks source float32/float64 values
and every first derivative, independent dense-Gaussian all-leaf HVPs,
training/evaluation covariance modes, grouped solve dimensions, device-only
execution, snapshots, state loading, deepcopy and whole-model pickle.

## Source constructor boundaries

Constructor compatibility is separate from implementing every strategy in
GPyTorch. Direct `GridInterpolationVariationalStrategy`,
`NNVariationalStrategy`, and `OrthogonallyDecoupledVariationalStrategy` are
not interchangeable with the constructor call made by the pinned BoTorch
wrapper: they reject its `inducing_points` or
`learn_inducing_locations` keyword arguments. A caller can instead supply a
compatible factory or an already-constructed approximate model. A caller's
factory must adapt the actual constructor arguments and supply a coherent
strategy; accepting a class name cannot substitute for that protocol.
