r"""
A registry of helpers for generating inputs to acquisition function
constructors programmatically from a consistent input format
(``botorch.acquisition.input_constructors``).

The constructors take BoTorch's ``training_data`` (a ``SupervisedDataset``
of botorch_rs or BoTorch, or a dict of them), objectives, posterior
transforms and constraints (of botorch_rs or BoTorch), and return the
keyword arguments of the botorch_rs acquisition functions with BoTorch's
values and defaults. Computations on the training data use torch for torch
tensors (with BoTorch's operations, so that e.g. ``best_f`` is exactly
BoTorch's) and plain Python otherwise; callables (objectives and constraints
of BoTorch, ``GenericMCObjective``) require torch.

Random numbers: the candidate set of ``qMaxValueEntropy`` is drawn with the
random number generator of the training data's array library (torch's for
tensors, so that ``torch.manual_seed`` reproduces BoTorch's draws), the
Sobol points of ``qNegIntegratedPosteriorVariance`` with botorch_rs's
``draw_sobol_samples``, the optimal samples of ``qJointEntropySearch`` with
botorch_rs's ``get_optimal_samples`` and the current values of the knowledge
gradients with botorch_rs's ``optimize_acqf`` (pass
``batch_initial_conditions`` through ``optimize_objective_kwargs`` for
deterministic results). The fixed sample ``w`` of the outcome model of
the EUBO constructors is ``torch.randn(pref_model.dim) * sample_multiplier``
with torch's global generator (BoTorch's draw), or Python's ``random``
without torch.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import math
import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any, TypeVar

from botorch_rs._arrays import (
    ArrayKind,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch_rs.acquisition.analytic import (
    ExpectedImprovement,
    LogConstrainedExpectedImprovement,
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    LogProbabilityOfFeasibility,
    LogProbabilityOfImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch_rs.acquisition.bayesian_active_learning import (
    qBayesianActiveLearningByDisagreement,
)
from botorch_rs.acquisition.cost_aware import InverseCostWeightedUtility
from botorch_rs.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch_rs.acquisition.joint_entropy_search import qJointEntropySearch
from botorch_rs.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch_rs.acquisition.logei import (
    TAU_MAX,
    TAU_RELU,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qLogProbabilityOfFeasibility,
)
from botorch_rs.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch_rs.acquisition.monte_carlo import (
    qExpectedImprovement,
    qLowerConfidenceBound,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch_rs.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch_rs.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch_rs.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch_rs.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch_rs.acquisition.multi_objective.parego import qLogNParEGO
from botorch_rs.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch_rs.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
    _botorch_class_names,
)
from botorch_rs.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    qExpectedUtilityOfBestOption,
)
from botorch_rs.acquisition.utils import (
    compute_best_feasible_objective,
    expand_trace_observations,
    get_infeasible_cost,
    get_optimal_samples,
    project_to_target_fidelity,
)
from botorch_rs.exceptions.errors import BotorchError, UnsupportedError
from botorch_rs.models.cost import AffineFidelityCostModel
from botorch_rs.models.deterministic import FixedSingleSampleModel
from botorch_rs.sampling.normal import SobolQMCNormalSampler
from botorch_rs.utils.datasets import SupervisedDataset

ACQF_INPUT_CONSTRUCTOR_REGISTRY: dict[Any, Callable[..., dict[str, Any]]] = {}

T = TypeVar("T")
#: A value or a dict of values (BoTorch's ``MaybeDict``).
MaybeDict = Any
#: The types of the keyword arguments of ``optimize_objective``.
TOptimizeObjectiveKwargs = Any


# --------------------------------------------------------------------------- #
# Array helpers: torch operations for torch tensors (BoTorch's, so that the
# results are bit-identical), plain Python on flat data otherwise.
# --------------------------------------------------------------------------- #


def _torch() -> Any:
    from botorch_rs._torch_bridge import require_torch

    require_torch()
    return importlib.import_module("torch")


def _shape(x: Any) -> tuple[int, ...]:
    if hasattr(x, "shape"):
        return tuple(x.shape)
    return tuple(to_flat(x, "data")[1])


def _equal(a: Any, b: Any) -> bool:
    if is_torch_tensor(a) and is_torch_tensor(b):
        return bool(_torch().equal(a, b))
    fa, fb = to_flat(a, "data"), to_flat(b, "data")
    return tuple(fa[1]) == tuple(fb[1]) and list(fa[0]) == list(fb[0])


def _cat_last(arrays: Sequence[Any]) -> Any:
    """``torch.cat(arrays, dim=-1)``."""
    if all(is_torch_tensor(a) for a in arrays):
        return _torch().cat(list(arrays), dim=-1)
    flats = [to_flat(a, "Y") for a in arrays]
    lead = tuple(flats[0][1][:-1])
    if any(tuple(f[1][:-1]) != lead for f in flats):
        raise RuntimeError("Sizes of tensors must match except in the last dimension.")
    widths = [f[1][-1] for f in flats]
    rows = math.prod(lead)
    out: list[float] = []
    for r in range(rows):
        for (data, _), w in zip(flats, widths, strict=True):
            out.extend(data[r * w : (r + 1) * w])
    return from_flat(out, (*lead, sum(widths)), kind_of(arrays[0]))


def _amax(x: Any, dim: int) -> Any:
    """``x.max(dim).values`` (``dim`` negative)."""
    if is_torch_tensor(x):
        return x.max(dim).values
    data, shape = to_flat(x, "x")
    nd = len(shape)
    axis = nd + dim
    outer = math.prod(shape[:axis])
    size = shape[axis]
    inner = math.prod(shape[axis + 1 :])
    out = [
        max(data[(o * size + i) * inner + j] for i in range(size))
        for o in range(outer)
        for j in range(inner)
    ]
    return from_flat(out, (*shape[:axis], *shape[axis + 1 :]), kind_of(x))


def _squeeze_last(x: Any) -> Any:
    if is_torch_tensor(x):
        return x.squeeze(-1)
    data, shape = to_flat(x, "x")
    if shape and shape[-1] == 1:
        shape = shape[:-1]
    return from_flat(data, tuple(shape), kind_of(x))


def _as_tensor(x: Any) -> Any:
    torch = _torch()
    if is_torch_tensor(x):
        return x
    data, shape = to_flat(x, "x")
    return torch.tensor(list(data), dtype=torch.float64).reshape(shape)


def _restore(value: Any, kind: ArrayKind) -> Any:
    """A torch result as an array of ``kind``."""
    if kind.lib == "torch" or not is_torch_tensor(value):
        return value
    return from_flat(value.reshape(-1).tolist(), tuple(value.shape), kind)


def _bounds_tensor(bounds: Any, like: Any) -> Any:
    """``torch.as_tensor(bounds, dtype=X.dtype, device=X.device)`` for a
    list of ``(lower, upper)`` pairs (an array of the kind of ``like``)."""
    if is_torch_tensor(like):
        return _torch().as_tensor(bounds, dtype=like.dtype, device=like.device)
    data, shape = to_flat(bounds, "bounds")
    return from_flat(list(data), tuple(shape), kind_of(like))


def _transpose(x: Any) -> Any:
    """``x.t()`` of a 2-dim array."""
    if is_torch_tensor(x):
        return x.t()
    data, (r, c) = to_flat(x, "x")
    return from_flat([data[i * c + j] for j in range(c) for i in range(r)], (c, r), kind_of(x))


def _is_botorch_object(obj: Any) -> bool:
    return bool(_botorch_class_names(obj))


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def _field_is_shared(
    datasets: Iterable[SupervisedDataset] | dict[Hashable, SupervisedDataset],
    fieldname: str,
) -> bool:
    r"""Determines whether or not a given field is shared by all datasets."""
    if isinstance(datasets, dict):
        datasets = datasets.values()

    base = None
    for dataset in datasets:
        if not hasattr(dataset, fieldname):
            raise AttributeError(f"{type(dataset)} object has no field `{fieldname}`.")

        obj = getattr(dataset, fieldname)
        if base is None:
            base = obj
        elif not _equal(base, obj):
            return False

    return True


def _get_dataset_field(
    dataset: MaybeDict,
    fieldname: str,
    transform: Callable[[Any], Any] | None = None,
    join_rule: Callable[[Sequence[Any]], Any] | None = None,
    first_only: bool = False,
    assert_shared: bool = False,
) -> Any:
    r"""Convenience method for extracting a given field from one or more datasets."""
    if isinstance(dataset, dict):
        if assert_shared and not _field_is_shared(dataset, fieldname):
            raise ValueError(f"Field `{fieldname}` must be shared.")

        if not first_only:
            fields = (_get_dataset_field(d, fieldname, transform) for d in dataset.values())
            return join_rule(tuple(fields)) if join_rule else tuple(fields)

        dataset = next(iter(dataset.values()))

    field = getattr(dataset, fieldname)
    return transform(field) if transform else field


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def get_acqf_input_constructor(
    acqf_cls: type[AcquisitionFunction],
) -> Callable[..., dict[str, Any]]:
    r"""Get acquisition function input constructor from registry.

    Args:
        acqf_cls: The AcquisitionFunction class (not instance) for which
            to retrieve the input constructor.

    Returns:
        The input constructor associated with ``acqf_cls``.

    """
    if acqf_cls not in ACQF_INPUT_CONSTRUCTOR_REGISTRY:
        raise RuntimeError(
            f"Input constructor for acquisition class `{acqf_cls.__name__}` not "
            "registered. Use the `@acqf_input_constructor` decorator to register "
            "a new method."
        )
    return ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls]


def allow_only_specific_variable_kwargs(f: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for allowing a function to accept keyword arguments that are not
    explicitly listed in the function signature, but only specific ones.

    This decorator is applied in ``acqf_input_constructor`` so that all constructors
    obtained with ``acqf_input_constructor`` allow keyword
    arguments such as ``training_data`` and ``objective``, even if they do not appear
    in the signature of ``f``. Any other keyword arguments will raise an error,
    unless ``f`` takes ``**kwargs`` (e.g. the ``optimize_objective_kwargs`` of
    ``construct_inputs_qKG``), which receives them (BoTorch raises for them).
    """
    allowed = {
        # ``training_data`` and/or ``X_baseline`` are needed to compute baselines
        # for some EI-type acquisition functions.
        "training_data",
        "X_baseline",
        # Objective thresholds are needed for defining hypervolumes in
        # multi-objective optimization.
        "objective_thresholds",
        # ref_point is the new preferred way to pass reference points
        # for multi-objective optimization.
        "ref_point",
        # Used in input constructors for some lookahead acquisition functions
        # such as qKnowledgeGradient.
        "bounds",
        # Needed for LogProbabilityOfFeasibility
        # and LogConstrainedExpectedImprovement
        "constraints_tuple",
        "posterior_transform",
        # not used by analytic acquisition functions
        "objective",
        "constraints",
    }

    params = inspect.signature(f).parameters
    # Constructors with ``**kwargs`` (the ``optimize_objective_kwargs`` of the
    # knowledge gradients) accept every keyword argument (BoTorch rejects the
    # ones that are not named in the signature).
    var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def g(*args: Any, **kwargs: Any) -> T:
        new_kwargs = {}
        accepted_kwargs = params.keys()
        for k, v in kwargs.items():
            if k in accepted_kwargs or var_keyword:
                new_kwargs[k] = v
            elif k not in allowed:
                raise TypeError(
                    f"Unexpected keyword argument `{k}` when"
                    f" constructing input arguments for {f.__name__}."
                )
        return f(*args, **new_kwargs)

    return g


def acqf_input_constructor(
    *acqf_cls: type[AcquisitionFunction],
) -> Callable[..., Any]:
    r"""Decorator for registering acquisition function input constructors.

    Args:
        acqf_cls: The AcquisitionFunction classes (not instances) for which
            to register the input constructor.
    """
    for acqf_cls_ in acqf_cls:
        if acqf_cls_ in ACQF_INPUT_CONSTRUCTOR_REGISTRY:
            raise ValueError(
                "Cannot register duplicate arg constructor for acquisition "
                f"class `{acqf_cls_.__name__}`"
            )

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        method_kwargs = allow_only_specific_variable_kwargs(method)
        for acqf_cls_ in acqf_cls:
            ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls_] = method_kwargs
        return method

    return decorator


def _register_acqf_input_constructor(
    acqf_cls: type[AcquisitionFunction],
    input_constructor: Callable[..., dict[str, Any]],
) -> None:
    ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls] = input_constructor


# --------------------- Input argument constructors --------------------- #


@acqf_input_constructor(PosteriorMean)
def construct_inputs_posterior_mean(
    model: Any,
    posterior_transform: Any = None,
) -> dict[str, Any]:
    r"""Construct kwargs for PosteriorMean acquisition function.

    Args:
        model: The model to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {"model": model, "posterior_transform": posterior_transform}


@acqf_input_constructor(
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    LogProbabilityOfImprovement,
)
def construct_inputs_best_f(
    model: Any,
    training_data: MaybeDict,
    posterior_transform: Any = None,
    best_f: Any = None,
    maximize: bool = True,
) -> dict[str, Any]:
    r"""Construct kwargs for the acquisition functions requiring ``best_f``.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
            Used to determine default value for ``best_f``.
        best_f: Threshold above (or below) which improvement is defined.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if best_f is None:
        best_f = get_best_f_analytic(
            training_data=training_data,
            posterior_transform=posterior_transform,
        )

    return {
        "model": model,
        "posterior_transform": posterior_transform,
        "best_f": best_f,
        "maximize": maximize,
    }


@acqf_input_constructor(LogProbabilityOfFeasibility)
def construct_inputs_pof(model: Any, constraints_tuple: tuple[Any, Any]) -> dict[str, Any]:
    r"""Construct kwargs for the log probability of feasibility acquisition function.

    Args:
        model: The model to be used in the acquisition function.
        constraints_tuple: A tuple of ``(A, b)``. For ``k`` outcome constraints
            and ``m`` outputs at `f(x)``, ``A`` is ``k x m`` and ``b`` is ``k x 1`` such
            that ``A f(x) <= b``.


    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # Construct a constraint dictionary from constraint_tuple
    constraints_dict = _construct_constraint_dict_from_tuple(
        constraints_tuple, LogProbabilityOfFeasibility
    )

    return {"model": model, "constraints": constraints_dict}


@acqf_input_constructor(LogConstrainedExpectedImprovement)
def construct_inputs_logcei(
    model: Any,
    training_data: MaybeDict,
    objective_index: int,
    constraints_tuple: tuple[Any, Any],
    best_f: Any = None,
    maximize: bool = True,
) -> dict[str, Any]:
    r"""Construct kwargs for the log constrained expected improvement
    acquisition function.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
            Used to determine default value for ``best_f``.
        objective_index: The index of the objective.
        constraints_tuple: A tuple of ``(A, b)``. For ``k`` outcome constraints
            and ``m`` outputs at `f(x)``, ``A`` is ``k x m`` and ``b`` is ``k x 1`` such
            that ``A f(x) <= b``.
        best_f: Either a scalar or a ``b``-dim Tensor (batch mode) representing
                the best feasible function value observed so far (assumed noiseless).
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """

    # If no best_f provided, compute it from the training data
    # For LogCEI, posterior_transform is not used.
    if best_f is None:
        best_f = get_best_f_analytic(
            training_data=training_data,
        )

    # Construct a constraint dictionary from constraint_tuple
    constraints_dict = _construct_constraint_dict_from_tuple(
        constraints_tuple, LogConstrainedExpectedImprovement
    )

    return {
        "model": model,
        "best_f": best_f,
        "objective_index": objective_index,
        "constraints": constraints_dict,
        "maximize": maximize,
    }


@acqf_input_constructor(UpperConfidenceBound)
def construct_inputs_ucb(
    model: Any,
    posterior_transform: Any = None,
    beta: Any = 0.2,
    maximize: bool = True,
) -> dict[str, Any]:
    r"""Construct kwargs for ``UpperConfidenceBound``.

    Args:
        model: The model to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        beta: Either a scalar or a one-dim tensor with ``b`` elements (batch mode)
            representing the trade-off parameter between mean and covariance
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        "model": model,
        "posterior_transform": posterior_transform,
        "beta": beta,
        "maximize": maximize,
    }


@acqf_input_constructor(NoisyExpectedImprovement, LogNoisyExpectedImprovement)
def construct_inputs_noisy_ei(
    model: Any,
    training_data: MaybeDict,
    num_fantasies: int = 20,
    maximize: bool = True,
) -> dict[str, Any]:
    r"""Construct kwargs for ``NoisyExpectedImprovement``.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        num_fantasies: The number of fantasies to generate. The higher this
            number the more accurate the model (at the expense of model
            complexity and performance).
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # TODO: Add prune_baseline functionality as for qNEI
    X = _get_dataset_field(training_data, "X", first_only=True, assert_shared=True)
    return {
        "model": model,
        "X_observed": X,
        "num_fantasies": num_fantasies,
        "maximize": maximize,
    }


@acqf_input_constructor(qSimpleRegret)
def construct_inputs_qSimpleRegret(
    model: Any,
    objective: Any = None,
    posterior_transform: Any = None,
    X_pending: Any = None,
    sampler: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    X_baseline: Any = None,
) -> dict[str, Any]:
    r"""Construct kwargs for qSimpleRegret.

    Args:
        model: The model to be used in the acquisition function.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A ``batch_shape, m x d``-dim Tensor of ``m`` design points
            that have points that have been submitted for function evaluation
            but have not yet been evaluated.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        X_baseline: A ``batch_shape x r x d``-dim Tensor of ``r`` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first ``X``.

    Returns:
        A dict mapping kwarg names of the constructor to values.

    With constraints and without an objective, the objective of the
    ``ConstrainedMCObjective`` is the identity (BoTorch passes ``None``, which
    fails when the acquisition function is evaluated).
    """
    if constraints is not None:
        if X_baseline is None:
            raise ValueError("Constraints require an X_baseline.")
        if objective is None:
            objective = IdentityMCObjective()
        objective = ConstrainedMCObjective(
            objective=objective,
            constraints=constraints,
            infeasible_cost=get_infeasible_cost(X=X_baseline, model=model, objective=objective),
        )
    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
    }


@acqf_input_constructor(qExpectedImprovement)
def construct_inputs_qEI(
    model: Any,
    training_data: MaybeDict,
    objective: Any = None,
    posterior_transform: Any = None,
    X_pending: Any = None,
    sampler: Any = None,
    best_f: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    eta: Any = 1e-3,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``qExpectedImprovement`` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        best_f: Threshold above (or below) which improvement is defined.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of ``compute_smoothed_feasibility_indicator``.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if best_f is None:
        best_f = get_best_f_mc(
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            constraints=constraints,
            model=model,
        )

    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "best_f": best_f,
        "constraints": constraints,
        "eta": eta,
    }


@acqf_input_constructor(qLogExpectedImprovement)
def construct_inputs_qLogEI(
    model: Any,
    training_data: MaybeDict,
    objective: Any = None,
    posterior_transform: Any = None,
    X_pending: Any = None,
    sampler: Any = None,
    best_f: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    eta: Any = 1e-3,
    fat: bool = True,
    tau_max: float = TAU_MAX,
    tau_relu: float = TAU_RELU,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``qLogExpectedImprovement`` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        best_f: Threshold above (or below) which improvement is defined.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of ``compute_smoothed_feasibility_indicator``.
        fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
            approximation to the ReLU.
        tau_max: Temperature parameter controlling the sharpness of the smooth
            approximations to max.
        tau_relu: Temperature parameter controlling the sharpness of the smooth
            approximations to ReLU.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        **construct_inputs_qEI(
            model=model,
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sampler=sampler,
            best_f=best_f,
            constraints=constraints,
            eta=eta,
        ),
        "fat": fat,
        "tau_max": tau_max,
        "tau_relu": tau_relu,
    }


@acqf_input_constructor(qLogProbabilityOfFeasibility)
def construct_inputs_LogPF(
    model: Any,
    constraints: list[Callable[[Any], Any]],
    posterior_transform: Any = None,
    X_pending: Any = None,
    sampler: Any = None,
    eta: Any = 1e-3,
    fat: bool = True,
    tau_max: float = TAU_MAX,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``qLogProbabilityOfFeasibility`` constructor.

    Args:
        model: The model to be used in the acquisition function.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of ``compute_smoothed_feasibility_indicator``.
        fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
            approximation to the ReLU.
        tau_max: Temperature parameter controlling the sharpness of the smooth
            approximations to max.

    Returns:
        A dictionary mapping kwarg names of the constructor to values.
    """
    return {
        "model": model,
        "constraints": constraints,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "eta": eta,
        "fat": fat,
        "tau_max": tau_max,
    }


@acqf_input_constructor(qNoisyExpectedImprovement)
def construct_inputs_qNEI(
    model: Any,
    training_data: MaybeDict,
    objective: Any = None,
    posterior_transform: Any = None,
    X_pending: Any = None,
    sampler: Any = None,
    X_baseline: Any = None,
    prune_baseline: bool | None = True,
    cache_root: bool | None = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    eta: Any = 1e-3,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``qNoisyExpectedImprovement`` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A ``batch_shape x r x d``-dim Tensor of ``r`` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first ``X``.
        prune_baseline: If True, remove points in ``X_baseline`` that are
            highly unlikely to be the best point. This can significantly
            improve performance and is generally recommended.
        cache_root: A boolean indicating whether to cache the root
            decomposition over ``X_baseline`` and use low-rank updates.
            If None, will be set to True if the model supports it and False
            otherwise.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of ``compute_smoothed_feasibility_indicator``.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if X_baseline is None:
        X_baseline = _get_dataset_field(
            training_data,
            fieldname="X",
            assert_shared=True,
            first_only=True,
        )
    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "X_baseline": X_baseline,
        "prune_baseline": prune_baseline,
        "cache_root": cache_root,
        "constraints": constraints,
        "eta": eta,
    }


@acqf_input_constructor(qLogNoisyExpectedImprovement)
def construct_inputs_qLogNEI(
    model: Any,
    training_data: MaybeDict,
    objective: Any = None,
    posterior_transform: Any = None,
    X_pending: Any = None,
    sampler: Any = None,
    X_baseline: Any = None,
    prune_baseline: bool | None = True,
    cache_root: bool | None = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    eta: Any = 1e-3,
    fat: bool = True,
    tau_max: float = TAU_MAX,
    tau_relu: float = TAU_RELU,
    incremental: bool = True,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``qLogNoisyExpectedImprovement`` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A ``batch_shape x r x d``-dim Tensor of ``r`` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first ``X``.
        prune_baseline: If True, remove points in ``X_baseline`` that are
            highly unlikely to be the best point. This can significantly
            improve performance and is generally recommended.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of ``compute_smoothed_feasibility_indicator``.
        fat: Toggles the use of the fat-tailed non-linearities to smoothly approximate
            the constraints indicator function.
        tau_max: Temperature parameter controlling the sharpness of the smooth
            approximations to max.
        tau_relu: Temperature parameter controlling the sharpness of the smooth
            approximations to ReLU.
        incremental: Whether to compute incremental EI over the pending points
            or compute EI of the joint batch improvement (including pending
            points).

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        **construct_inputs_qNEI(
            model=model,
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sampler=sampler,
            X_baseline=X_baseline,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
            constraints=constraints,
            eta=eta,
        ),
        "fat": fat,
        "tau_max": tau_max,
        "tau_relu": tau_relu,
        "incremental": incremental,
    }


@acqf_input_constructor(qProbabilityOfImprovement)
def construct_inputs_qPI(
    model: Any,
    training_data: MaybeDict,
    objective: Any = None,
    posterior_transform: Any = None,
    X_pending: Any = None,
    sampler: Any = None,
    tau: float = 1e-3,
    best_f: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    eta: Any = 1e-3,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``qProbabilityOfImprovement`` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        tau: The temperature parameter used in the sigmoid approximation
            of the step function. Smaller values yield more accurate
            approximations of the function, but result in gradients
            estimates with higher variance.
        best_f: The best objective value observed so far (assumed noiseless). Can
            be a ``batch_shape``-shaped tensor, which in case of a batched model
            specifies potentially different values for each element of the batch.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of ``compute_smoothed_feasibility_indicator``.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if best_f is None:
        best_f = get_best_f_mc(
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            constraints=constraints,
            model=model,
        )

    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "tau": tau,
        "best_f": best_f,
        "constraints": constraints,
        "eta": eta,
    }


@acqf_input_constructor(qLowerConfidenceBound, qUpperConfidenceBound)
def construct_inputs_qUCB(
    model: Any,
    objective: Any = None,
    posterior_transform: Any = None,
    X_pending: Any = None,
    sampler: Any = None,
    X_baseline: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    beta: float = 0.2,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``qUpperConfidenceBound`` constructor.

    Args:
        model: The model to be used in the acquisition function.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A ``batch_shape x r x d``-dim Tensor of ``r`` design points
            that have already been observed. These points are used to
            compute with infeasible cost when there are constraints.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        beta: Controls tradeoff between mean and standard deviation in UCB.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if constraints is not None:
        if X_baseline is None:
            raise ValueError("Constraints require an X_baseline.")
        if objective is None:
            objective = IdentityMCObjective()
        objective = ConstrainedMCObjective(
            objective=objective,
            constraints=constraints,
            infeasible_cost=get_infeasible_cost(X=X_baseline, model=model, objective=objective),
        )
    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "beta": beta,
    }


def _get_sampler(mc_samples: int, qmc: bool) -> Any:
    """Set up MC sampler for q(N)EHVI (botorch_rs has only the
    ``SobolQMCNormalSampler``)."""
    # initialize the sampler
    if qmc:
        return SobolQMCNormalSampler(sample_shape=(mc_samples,))
    raise UnsupportedError(
        "qmc=False (an IIDNormalSampler) is not supported by botorch_rs; use qmc=True."
    )


def _is_gpytorch_model(model: Any) -> bool:
    """Whether BoTorch would consider ``model`` a ``GPyTorchModel`` (the
    exact GP models of botorch_rs)."""
    from botorch_rs.models.gp_regression import SingleTaskGP
    from botorch_rs.models.model_list_gp_regression import ModelListGP
    from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP

    if isinstance(model, ModelListGP):
        return False
    return isinstance(model, (SingleTaskGP, MultiTaskGP, KroneckerMultiTaskGP))


def _posterior_mean(model: Any, X: Any) -> Any:
    """``model.posterior(X).mean`` (computed without gradients)."""
    return model.posterior(X).mean


def _feasible_rows(Y: Any, constraints: list[Callable[[Any], Any]]) -> Any:
    """``Y[feas]`` for ``feas = all(c(Y) <= 0)`` (torch)."""
    torch = _torch()
    kind = kind_of(Y)
    Yt = _as_tensor(Y)
    feas = torch.stack([c(Yt) <= 0 for c in constraints], dim=-1).all(dim=-1)
    return _restore(Yt[feas], kind)


def _is_risk_measure(objective: Any) -> bool:
    from botorch_rs.acquisition.risk_measures import RiskMeasureMCObjective

    return isinstance(objective, RiskMeasureMCObjective) or (
        "RiskMeasureMCObjective" in _botorch_class_names(objective)
    )


def _is_mo_risk_measure_class(objective: Any) -> bool:
    return any(c.__name__ == "MultiOutputRiskMeasureMCObjective" for c in type(objective).__mro__)


def _apply_objective(objective: Any, Y: Any) -> Any:
    """``objective(Y)`` (a risk measure's ``preprocessing_function``), on
    torch tensors for BoTorch objectives."""
    if _is_risk_measure(objective) or _is_mo_risk_measure_class(objective):
        return objective.preprocessing_function(Y)
    if _is_botorch_object(objective) and not is_torch_tensor(Y):
        return _restore(objective(_as_tensor(Y)), kind_of(Y))
    return objective(Y)


def _partitioning(ref_point: Any, Y: Any, alpha: float) -> Any:
    from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
        NondominatedPartitioning,
    )

    if alpha > 0:
        return NondominatedPartitioning(ref_point=ref_point, Y=Y, alpha=alpha)
    return FastNondominatedPartitioning(ref_point=ref_point, Y=Y)


def _num_objectives(ref_point: Any, drop_nan: bool = False) -> int:
    data, shape = to_flat(ref_point, "ref_point")
    if drop_nan:
        return sum(1 for v in data if not math.isnan(v))
    return shape[0]


@acqf_input_constructor(ExpectedHypervolumeImprovement)
def construct_inputs_EHVI(
    model: Any,
    training_data: MaybeDict,
    objective_thresholds: Any = None,
    posterior_transform: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    alpha: float | None = None,
    Y_pmean: Any = None,
    ref_point: Any = None,
) -> dict[str, Any]:
    r"""Construct kwargs for ``ExpectedHypervolumeImprovement`` constructor."""
    ref_point = _get_ref_point(
        objective_thresholds=objective_thresholds,
        ref_point=ref_point,
    )
    num_objectives = _num_objectives(ref_point)
    if constraints is not None:
        raise NotImplementedError("EHVI does not yet support outcome constraints.")

    X = _get_dataset_field(
        training_data,
        fieldname="X",
        first_only=True,
        assert_shared=True,
    )

    alpha = (
        get_default_partitioning_alpha(num_objectives=num_objectives) if alpha is None else alpha
    )

    # Compute posterior mean (for ref point computation ref pareto frontier)
    # if one is not provided among arguments.
    if Y_pmean is None:
        Y_pmean = _posterior_mean(model, X)
    partitioning = _partitioning(ref_point, Y_pmean, alpha)

    kwargs = {
        "model": model,
        "ref_point": ref_point,
        "partitioning": partitioning,
    }
    if posterior_transform is not None:
        kwargs["posterior_transform"] = posterior_transform
    return kwargs


@acqf_input_constructor(qExpectedHypervolumeImprovement, qLogExpectedHypervolumeImprovement)
def construct_inputs_qEHVI(
    model: Any,
    training_data: MaybeDict,
    objective_thresholds: Any = None,
    objective: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    alpha: float | None = None,
    sampler: Any = None,
    X_pending: Any = None,
    eta: float = 1e-3,
    mc_samples: int = 128,
    qmc: bool = True,
    ref_point: Any = None,
) -> dict[str, Any]:
    r"""
    Construct kwargs for ``qExpectedHypervolumeImprovement`` and
    ``qLogExpectedHypervolumeImprovement``.
    """
    X = _get_dataset_field(
        training_data,
        fieldname="X",
        first_only=True,
        assert_shared=True,
    )

    # compute posterior mean (for ref point computation ref pareto frontier)
    Y_pmean = _posterior_mean(model, X)
    # For HV-based acquisition functions we pass the constraint transform directly
    if constraints is not None:
        # Adjust ``Y_pmean`` to contain feasible points only.
        Y_pmean = _feasible_rows(Y_pmean, constraints)

    ref_point = _get_ref_point(
        objective_thresholds=objective_thresholds,
        objective=objective,
        ref_point=ref_point,
    )
    num_objectives = _num_objectives(ref_point)

    alpha = (
        get_default_partitioning_alpha(num_objectives=num_objectives) if alpha is None else alpha
    )

    Y = Y_pmean if objective is None else _apply_objective(objective, Y_pmean)
    partitioning = _partitioning(ref_point, Y, alpha)

    if sampler is None and _is_gpytorch_model(model):
        sampler = _get_sampler(mc_samples=mc_samples, qmc=qmc)

    return {
        "model": model,
        "ref_point": ref_point,
        "partitioning": partitioning,
        "sampler": sampler,
        "X_pending": X_pending,
        "constraints": constraints,
        "eta": eta,
        "objective": objective,
    }


@acqf_input_constructor(qNoisyExpectedHypervolumeImprovement)
def construct_inputs_qNEHVI(
    model: Any,
    training_data: MaybeDict,
    objective_thresholds: Any = None,
    objective: Any = None,
    X_baseline: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    alpha: float | None = None,
    sampler: Any = None,
    X_pending: Any = None,
    eta: float = 1e-3,
    fat: bool = False,
    mc_samples: int = 128,
    qmc: bool = True,
    prune_baseline: bool = True,
    cache_pending: bool = True,
    max_iep: int = 0,
    incremental_nehvi: bool = True,
    cache_root: bool | None = None,
    ref_point: Any = None,
) -> dict[str, Any]:
    r"""Construct kwargs for ``qNoisyExpectedHypervolumeImprovement``'s constructor."""
    if X_baseline is None:
        X_baseline = _get_dataset_field(
            training_data,
            fieldname="X",
            first_only=True,
            assert_shared=True,
        )
    # This selects the objectives (a subset of the outcomes) and set each
    # objective threshold to have the proper optimization direction.
    if objective is None:
        objective = IdentityMCMultiOutputObjective()

    if constraints is not None:
        if _is_risk_measure(objective):
            raise UnsupportedError(
                "Outcome constraints are not supported with risk measures. "
                "Use a feasibility-weighted risk measure instead."
            )

    if sampler is None and _is_gpytorch_model(model):
        sampler = _get_sampler(mc_samples=mc_samples, qmc=qmc)

    ref_point = _get_ref_point(
        objective_thresholds=objective_thresholds,
        objective=objective,
        ref_point=ref_point,
    )

    num_objectives = _num_objectives(ref_point, drop_nan=True)
    if alpha is None:
        alpha = get_default_partitioning_alpha(num_objectives=num_objectives)

    return {
        "model": model,
        "ref_point": ref_point,
        "X_baseline": X_baseline,
        "sampler": sampler,
        "objective": objective,
        "constraints": constraints,
        "X_pending": X_pending,
        "eta": eta,
        "fat": fat,
        "prune_baseline": prune_baseline,
        "alpha": alpha,
        "cache_pending": cache_pending,
        "max_iep": max_iep,
        "incremental_nehvi": incremental_nehvi,
        "cache_root": cache_root,
    }


@acqf_input_constructor(qLogNoisyExpectedHypervolumeImprovement)
def construct_inputs_qLogNEHVI(
    model: Any,
    training_data: MaybeDict,
    objective_thresholds: Any = None,
    objective: Any = None,
    X_baseline: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    alpha: float | None = None,
    sampler: Any = None,
    X_pending: Any = None,
    eta: float = 1e-3,
    fat: bool = True,
    mc_samples: int = 128,
    qmc: bool = True,
    prune_baseline: bool = True,
    cache_pending: bool = True,
    max_iep: int = 0,
    incremental_nehvi: bool = True,
    cache_root: bool | None = None,
    tau_relu: float = TAU_RELU,
    tau_max: float = TAU_MAX,
    ref_point: Any = None,
) -> dict[str, Any]:
    """
    Construct kwargs for ``qLogNoisyExpectedHypervolumeImprovement``'s constructor."
    """
    return {
        **construct_inputs_qNEHVI(
            model=model,
            training_data=training_data,
            objective_thresholds=objective_thresholds,
            objective=objective,
            ref_point=ref_point,
            X_baseline=X_baseline,
            constraints=constraints,
            alpha=alpha,
            sampler=sampler,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            mc_samples=mc_samples,
            qmc=qmc,
            prune_baseline=prune_baseline,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
        ),
        "tau_relu": tau_relu,
        "tau_max": tau_max,
    }


@acqf_input_constructor(qLogNParEGO)
def construct_inputs_qLogNParEGO(
    model: Any,
    training_data: MaybeDict,
    scalarization_weights: Any = None,
    objective: Any = None,
    X_pending: Any = None,
    sampler: Any = None,
    X_baseline: Any = None,
    prune_baseline: bool | None = True,
    cache_root: bool | None = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    eta: Any = 1e-3,
    fat: bool = True,
    tau_max: float = TAU_MAX,
    tau_relu: float = TAU_RELU,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``qLogNParEGO`` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        scalarization_weights: A ``m``-dim Tensor of weights to be used in the
            Chebyshev scalarization. If omitted, samples from the unit simplex.
        objective: The MultiOutputMCAcquisitionObjective under which the samples are
            evaluated before applying Chebyshev scalarization.
            Defaults to ``IdentityMultiOutputObjective()``.
        X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A ``batch_shape x r x d``-dim Tensor of ``r`` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first ``X``.
        prune_baseline: If True, remove points in ``X_baseline`` that are
            highly unlikely to be the best point. This can significantly
            improve performance and is generally recommended.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension ``sample_shape x batch-shape x q x m``-dim to a
            ``sample_shape x batch-shape x q``-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of ``compute_smoothed_feasibility_indicator``.
        fat: Toggles the use of the fat-tailed non-linearities to smoothly approximate
            the constraints indicator function.
        tau_max: Temperature parameter controlling the sharpness of the smooth
            approximations to max.
        tau_relu: Temperature parameter controlling the sharpness of the smooth
            approximations to ReLU.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_qLogNEI(
        model=model,
        training_data=training_data,
        objective=objective,
        X_pending=X_pending,
        sampler=sampler,
        X_baseline=X_baseline,
        prune_baseline=prune_baseline,
        cache_root=cache_root,
        constraints=constraints,
        eta=eta,
        fat=fat,
        tau_max=tau_max,
        tau_relu=tau_relu,
    )
    base_inputs.pop("posterior_transform", None)
    return {
        **base_inputs,
        "scalarization_weights": scalarization_weights,
    }


def _uniform(n: int, d: int, like: Any) -> Any:
    """``torch.rand(n, d)`` with the generator of ``like``'s array library
    (torch's for tensors), as an array of its kind."""
    if is_torch_tensor(like):
        return _torch().rand(n, d, device=like.device, dtype=like.dtype)
    import random

    return from_flat([random.random() for _ in range(n * d)], (n, d), kind_of(like))


@acqf_input_constructor(qMaxValueEntropy)
def construct_inputs_qMES(
    model: Any,
    training_data: MaybeDict,
    bounds: list[tuple[float, float]],
    posterior_transform: Any = None,
    candidate_size: int = 1000,
    maximize: bool = True,
    # TODO: qMES also supports other inputs, such as num_fantasies
) -> dict[str, Any]:
    r"""Construct kwargs for ``qMaxValueEntropy`` constructor.

    The candidate set is uniform in ``bounds`` (drawn with torch's generator
    for torch training data, as in BoTorch).
    """
    X = _get_dataset_field(training_data, "X", first_only=True)
    _rvs = _uniform(candidate_size, len(bounds), X)
    _bounds = _transpose(_bounds_tensor(bounds, X))
    if is_torch_tensor(_rvs):
        candidate_set = _bounds[0] + (_bounds[1] - _bounds[0]) * _rvs
    else:
        bdata, _ = to_flat(_bounds, "bounds")
        d = len(bounds)
        rdata, _ = to_flat(_rvs, "rvs")
        vals = [
            bdata[j] + (bdata[d + j] - bdata[j]) * rdata[i * d + j]
            for i in range(candidate_size)
            for j in range(d)
        ]
        candidate_set = from_flat(vals, (candidate_size, d), kind_of(X))
    return {
        "model": model,
        "posterior_transform": posterior_transform,
        "candidate_set": candidate_set,
        "maximize": maximize,
    }


def _project_target_fidelity(target_fidelities: dict[int, Any]) -> Callable[[Any], Any]:
    def project(X: Any) -> Any:
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities, d=_shape(X)[-1])

    return project


def _expand_trace(target_fidelities: dict[int, Any], num_trace_observations: int) -> Any:
    def expand(X: Any) -> Any:
        return expand_trace_observations(
            X=X,
            fidelity_dims=sorted(target_fidelities),
            num_trace_obs=num_trace_observations,
        )

    return expand


def construct_inputs_mf_base(
    target_fidelities: dict[int, int | float],
    fidelity_weights: dict[int, float] | None = None,
    cost_intercept: float = 1.0,
    num_trace_observations: int = 0,
) -> dict[str, Any]:
    r"""Construct kwargs for a multifidelity acquisition function's constructor.

    ``project`` and ``expand`` are functions of ``X`` (like BoTorch's
    lambdas); the constructors of the multi-fidelity acquisition functions
    replace them by the ``functools.partial`` objects that botorch_rs
    evaluates natively, with the same values.
    """
    if fidelity_weights is None:
        fidelity_weights = dict.fromkeys(target_fidelities, 1.0)

    if set(target_fidelities) != set(fidelity_weights):
        raise RuntimeError(
            "Must provide the same indices for target_fidelities "
            f"({set(target_fidelities)}) and fidelity_weights "
            f" ({set(fidelity_weights)})."
        )

    cost_aware_utility = InverseCostWeightedUtility(
        cost_model=AffineFidelityCostModel(
            fidelity_weights=fidelity_weights, fixed_cost=cost_intercept
        )
    )

    return {
        "cost_aware_utility": cost_aware_utility,
        "expand": _expand_trace(target_fidelities, num_trace_observations),
        "project": _project_target_fidelity(target_fidelities),
    }


def _native_mf_inputs(
    inputs_mf: dict[str, Any],
    target_fidelities: dict[int, Any],
    num_trace_observations: int,
    d: int,
) -> dict[str, Any]:
    """``construct_inputs_mf_base``'s outputs with ``project`` / ``expand`` as
    the ``functools.partial`` objects that botorch_rs evaluates natively
    (inputs of ``d`` columns)."""
    out = dict(inputs_mf)
    if "project" in out:
        out["project"] = functools.partial(
            project_to_target_fidelity, target_fidelities=target_fidelities, d=d
        )
    if "expand" in out:
        out["expand"] = functools.partial(
            expand_trace_observations,
            fidelity_dims=sorted(target_fidelities),
            num_trace_obs=num_trace_observations,
        )
    return out


@acqf_input_constructor(qKnowledgeGradient)
def construct_inputs_qKG(
    model: Any,
    training_data: MaybeDict,
    bounds: list[tuple[float, float]],
    objective: Any = None,
    posterior_transform: Any = None,
    num_fantasies: int = 64,
    with_current_value: bool = False,
    **optimize_objective_kwargs: TOptimizeObjectiveKwargs,
) -> dict[str, Any]:
    r"""Construct kwargs for ``qKnowledgeGradient`` constructor."""

    inputs_qkg = {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "num_fantasies": num_fantasies,
    }

    if with_current_value:
        X = _get_dataset_field(training_data, "X", first_only=True)
        _bounds = _bounds_tensor(bounds, X)

        _, current_value = optimize_objective(
            model=model,
            bounds=_transpose(_bounds),
            q=1,
            objective=objective,
            posterior_transform=posterior_transform,
            **optimize_objective_kwargs,
        )
        inputs_qkg["current_value"] = _detached_max(current_value)

    return inputs_qkg


def _detached_max(value: Any) -> Any:
    """``value.detach().cpu().max()``."""
    if is_torch_tensor(value):
        return value.detach().cpu().max()
    data, _ = to_flat(value, "current_value")
    return from_flat([max(data)], (), kind_of(value))


@acqf_input_constructor(qHypervolumeKnowledgeGradient)
def construct_inputs_qHVKG(
    model: Any,
    training_data: MaybeDict,
    bounds: list[tuple[float, float]],
    objective_thresholds: Any = None,
    objective: Any = None,
    posterior_transform: Any = None,
    num_fantasies: int = 8,
    num_pareto: int = 10,
    ref_point: Any = None,
    **optimize_objective_kwargs: TOptimizeObjectiveKwargs,
) -> dict[str, Any]:
    r"""Construct kwargs for ``qKnowledgeGradient`` constructor."""

    X = _get_dataset_field(training_data, "X", first_only=True)
    _bounds = _bounds_tensor(bounds, X)

    ref_point = _get_ref_point(
        objective_thresholds=objective_thresholds,
        objective=objective,
        ref_point=ref_point,
    )

    acq_function = _get_hv_value_function(
        model=model,
        ref_point=ref_point,
        use_posterior_mean=True,
        objective=objective,
    )

    _, current_value = optimize_objective(
        model=model,
        bounds=_transpose(_bounds),
        q=num_pareto,
        acq_function=acq_function,
        **optimize_objective_kwargs,
    )

    return {
        "model": model,
        "objective": objective,
        "ref_point": ref_point,
        "num_fantasies": num_fantasies,
        "num_pareto": num_pareto,
        "current_value": _detached_max(current_value),
    }


@acqf_input_constructor(qMultiFidelityKnowledgeGradient)
def construct_inputs_qMFKG(
    model: Any,
    training_data: MaybeDict,
    bounds: list[tuple[float, float]],
    target_fidelities: dict[int, int | float],
    objective: Any = None,
    posterior_transform: Any = None,
    fidelity_weights: dict[int, float] | None = None,
    cost_intercept: float = 1.0,
    num_trace_observations: int = 0,
    num_fantasies: int = 64,
    **optimize_objective_kwargs: TOptimizeObjectiveKwargs,
) -> dict[str, Any]:
    r"""Construct kwargs for ``qMultiFidelityKnowledgeGradient`` constructor."""

    X = _get_dataset_field(training_data, "X", first_only=True)
    _bounds = _bounds_tensor(bounds, X)

    inputs_mf = construct_inputs_mf_base(
        target_fidelities=target_fidelities,
        fidelity_weights=fidelity_weights,
        cost_intercept=cost_intercept,
        num_trace_observations=num_trace_observations,
    )
    inputs_mf = _native_mf_inputs(
        inputs_mf, target_fidelities, num_trace_observations, _shape(X)[-1]
    )

    _, current_value = optimize_objective(
        model=model,
        bounds=_transpose(_bounds),
        q=1,
        objective=objective,
        posterior_transform=posterior_transform,
        fixed_features=target_fidelities,
        **optimize_objective_kwargs,
    )

    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "num_fantasies": num_fantasies,
        "current_value": _detached_max(current_value),
        **inputs_mf,
    }


@acqf_input_constructor(qMultiFidelityHypervolumeKnowledgeGradient)
def construct_inputs_qMFHVKG(
    model: Any,
    training_data: MaybeDict,
    bounds: list[tuple[float, float]],
    target_fidelities: dict[int, int | float],
    objective_thresholds: Any = None,
    objective: Any = None,
    posterior_transform: Any = None,
    fidelity_weights: dict[int, float] | None = None,
    cost_intercept: float = 1.0,
    num_trace_observations: int = 0,
    num_fantasies: int = 8,
    num_pareto: int = 10,
    ref_point: Any = None,
    **optimize_objective_kwargs: TOptimizeObjectiveKwargs,
) -> dict[str, Any]:
    r"""
    Construct kwargs for ``qMultiFidelityHypervolumeKnowledgeGradient`` constructor.
    """

    inputs_mf = construct_inputs_mf_base(
        target_fidelities=target_fidelities,
        fidelity_weights=fidelity_weights,
        cost_intercept=cost_intercept,
        num_trace_observations=num_trace_observations,
    )

    if num_trace_observations > 0:
        raise NotImplementedError(
            "Trace observations are not currently supported "
            "by `qMultiFidelityHypervolumeKnowledgeGradient`."
        )

    del inputs_mf["expand"]

    X = _get_dataset_field(training_data, "X", first_only=True)
    _bounds = _bounds_tensor(bounds, X)
    inputs_mf = _native_mf_inputs(
        inputs_mf, target_fidelities, num_trace_observations, _shape(X)[-1]
    )

    ref_point = _get_ref_point(
        objective_thresholds=objective_thresholds,
        objective=objective,
        ref_point=ref_point,
    )

    acq_function = _get_hv_value_function(
        model=model,
        ref_point=ref_point,
        use_posterior_mean=True,
        objective=objective,
    )

    _, current_value = optimize_objective(
        model=model,
        bounds=_transpose(_bounds),
        q=num_pareto,
        acq_function=acq_function,
        fixed_features=target_fidelities,
        **optimize_objective_kwargs,
    )

    return {
        "model": model,
        "objective": objective,
        "ref_point": ref_point,
        "num_fantasies": num_fantasies,
        "num_pareto": num_pareto,
        "current_value": _detached_max(current_value),
        "target_fidelities": target_fidelities,
        **inputs_mf,
    }


@acqf_input_constructor(qMultiFidelityMaxValueEntropy)
def construct_inputs_qMFMES(
    model: Any,
    training_data: MaybeDict,
    bounds: list[tuple[float, float]],
    target_fidelities: dict[int, int | float],
    num_fantasies: int = 64,
    fidelity_weights: dict[int, float] | None = None,
    cost_intercept: float = 1.0,
    num_trace_observations: int = 0,
    candidate_size: int = 1000,
    maximize: bool = True,
) -> dict[str, Any]:
    r"""Construct kwargs for ``qMultiFidelityMaxValueEntropy`` constructor."""
    inputs_mf = construct_inputs_mf_base(
        target_fidelities=target_fidelities,
        fidelity_weights=fidelity_weights,
        cost_intercept=cost_intercept,
        num_trace_observations=num_trace_observations,
    )
    X = _get_dataset_field(training_data, "X", first_only=True)
    inputs_mf = _native_mf_inputs(
        inputs_mf, target_fidelities, num_trace_observations, _shape(X)[-1]
    )

    inputs_qmes = construct_inputs_qMES(
        model=model,
        training_data=training_data,
        bounds=bounds,
        candidate_size=candidate_size,
        maximize=maximize,
    )

    return {**inputs_mf, **inputs_qmes, "num_fantasies": num_fantasies}


def _one_sample_outcome_model(model: Any, pref_model: Any, sample_multiplier: Any) -> Any:
    """BoTorch's EUBO-zeta outcome model: ``FixedSingleSampleModel(model, w)``
    with ``w = torch.randn(pref_model.dim) * sample_multiplier`` (torch's
    global generator and float32 arithmetic, as BoTorch; Python's ``random``
    without torch). ``pref_model.dim`` rather than ``model.num_outputs``, as
    the number of outputs of a multi-task model can be its number of tasks."""
    dim = pref_model.dim
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        import random

        w: Any = [random.gauss(0.0, 1.0) * sample_multiplier for _ in range(int(dim))]
    else:
        w = torch.randn(dim) * sample_multiplier
    return FixedSingleSampleModel(model=model, w=w)


@acqf_input_constructor(AnalyticExpectedUtilityOfBestOption)
def construct_inputs_analytic_eubo(
    model: Any,
    pref_model: Any = None,
    previous_winner: Any = None,
    sample_multiplier: float | None = 1.0,
    objective: Any = None,
    posterior_transform: Any = None,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``AnalyticExpectedUtilityOfBestOption`` constructor.

    ``model`` is the primary model defined over the parameter space. It can be the
    outcome model in BOPE or the preference model in PBO. ``pref_model`` is the model
    defined over the outcome/metric space, which is typically the preference model
    in BOPE.

    If both model and pref_model exist, we are performing Bayesian Optimization with
    Preference Exploration (BOPE). When only pref_model is None, we are performing
    preferential BO (PBO).

    Args:
        model: The outcome model to be used in the acquisition function in BOPE
            when pref_model exists; otherwise, model is the preference model and
            we are doing Preferential BO
        pref_model: The preference model to be used in preference exploration as in
            BOPE; if None, we are doing PBO and model is the preference model.
        previous_winner: The previous winner of the best option.
        sample_multiplier: The scale factor for the single-sample model.
        objective: Ignored. This argument is allowed to be passed then ignored
            because of the way that EUBO is typically used in a BOPE loop.
        posterior_transform: Ignored. This argument is allowed to be passed then
            ignored because of the way that EUBO is typically used in a BOPE
            loop.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if pref_model is None:
        return {
            "pref_model": model,
            "outcome_model": None,
            "previous_winner": previous_winner,
        }
    # A deterministic fixed single sample model from ``model``, i.e. EUBO-zeta
    # as described in https://arxiv.org/abs/2203.11382.
    return {
        "pref_model": pref_model,
        "outcome_model": _one_sample_outcome_model(model, pref_model, sample_multiplier),
        "previous_winner": previous_winner,
    }


@acqf_input_constructor(qExpectedUtilityOfBestOption)
def construct_inputs_qeubo(
    model: Any,
    pref_model: Any = None,
    sample_multiplier: float | None = 1.0,
    sampler: SobolQMCNormalSampler | None = None,
    objective: Any = None,
    posterior_transform: Any = None,
    X_pending: Any = None,
) -> dict[str, Any]:
    r"""Construct kwargs for the ``qExpectedUtilityOfBestOption`` (qEUBO) constructor.

    ``model`` is the primary model defined over the parameter space. It can be the
    outcome model in BOPE or the preference model in PBO. ``pref_model`` is the model
    defined over the outcome/metric space, which is typically the preference model
    in BOPE.

    If both model and pref_model exist, we are performing Bayesian Optimization with
    Preference Exploration (BOPE). When only pref_model is None, we are performing
    preferential BO (PBO).

    Args:
        model: The outcome model to be used in the acquisition function in BOPE
            when pref_model exists; otherwise, model is the preference model and
            we are doing Preferential BO
        pref_model: The preference model to be used in preference exploration as in
            BOPE; if None, we are doing PBO and model is the preference model.
        sample_multiplier: The scale factor for the single-sample model.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if pref_model is None:
        return {
            "pref_model": model,
            "outcome_model": None,
            "sampler": sampler,
            "objective": objective,
            "posterior_transform": posterior_transform,
            "X_pending": X_pending,
        }
    # A deterministic fixed single sample model from ``model``, i.e. EUBO-zeta
    # as described in https://arxiv.org/abs/2203.11382.
    return {
        "pref_model": pref_model,
        "outcome_model": _one_sample_outcome_model(model, pref_model, sample_multiplier),
        "sampler": sampler,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
    }


def get_best_f_analytic(
    training_data: MaybeDict,
    posterior_transform: Any = None,
) -> Any:
    """The best observed (scalarized) outcome of the training data: the
    default ``best_f`` of the analytic acquisition functions (a tensor of
    shape ``batch_shape``, or the corresponding array of the training data's
    kind)."""
    if isinstance(training_data, dict) and not _field_is_shared(training_data, fieldname="X"):
        raise NotImplementedError("Currently only block designs are supported.")

    Y = _get_dataset_field(
        training_data,
        fieldname="Y",
        join_rule=_cat_last,
    )

    if posterior_transform is not None:
        if _is_botorch_object(posterior_transform) and not is_torch_tensor(Y):
            values = _restore(posterior_transform.evaluate(Y=_as_tensor(Y), X=None), kind_of(Y))
        else:
            values = posterior_transform.evaluate(Y=Y, X=None)
        return _amax(values, -1)
    if _shape(Y)[-1] > 1:
        raise NotImplementedError(
            "Analytic acquisition functions currently only work with "
            "multi-output models if provided with a `ScalarizedObjective`."
        )
    return _squeeze_last(_amax(Y, -2))


def _is_identity_objective(objective: Any) -> bool:
    return objective is None or isinstance(objective, IdentityMCObjective)


def _linear_objective_values(Y: Any, weights: list[float]) -> Any:
    """``Y @ weights`` of a ``(batch_shape) x q x m`` array of any kind
    (botorch_rs's ``LinearMCObjective`` of one sample, without torch)."""
    data, shape = to_flat(Y, "Y")
    m = shape[-1]
    if m != len(weights):
        raise RuntimeError("Output shape of samples not equal to that of weights")
    values = [
        sum(data[i * m + j] * w for j, w in enumerate(weights)) for i in range(len(data) // m)
    ]
    return from_flat(values, tuple(shape[:-1]), kind_of(Y))


def get_best_f_mc(
    training_data: MaybeDict,
    objective: Any = None,
    posterior_transform: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    model: Any = None,
) -> Any:
    """
    Computes the maximum value of the objective over the training data.

    Args:
        training_data: Has fields Y, which is evaluated by ``objective``, and X,
            which is used as ``X_baseline``. ``Y`` is of shape
            ``batch_shape x q x m``.
        objective: The objective under which to evaluate the training data. If
            omitted, uses ``IdentityMCObjective``.
        posterior_transform: An optional PosteriorTransform to apply to ``Y``
            before computing the objective.
        constraints: For assessing feasibility.
        model: Used by ``compute_best_feasible_objective`` when there are no
            feasible observations.

    Returns:
        A Tensor of shape ``batch_shape``.
    """
    if isinstance(training_data, dict) and not _field_is_shared(training_data, fieldname="X"):
        raise NotImplementedError("Currently only block designs are supported.")

    X_baseline = _get_dataset_field(
        training_data,
        fieldname="X",
        assert_shared=True,
        first_only=True,
    )

    Y = _get_dataset_field(
        training_data,
        fieldname="Y",
        join_rule=_cat_last,
    )  # batch_shape x q x m
    kind = kind_of(Y)
    # Without torch tensors, the identity and botorch_rs's linear objective
    # are computed natively (other objectives are torch callables).
    native = (
        not is_torch_tensor(Y)
        and constraints is None
        and (_is_identity_objective(objective) or type(objective) is LinearMCObjective)
        and (posterior_transform is None or not _is_botorch_object(posterior_transform))
    )
    if not native:
        # BoTorch's computation on torch tensors.
        Y = _as_tensor(Y)
        X_baseline = _as_tensor(X_baseline)

    if posterior_transform is not None:
        # retain the original tensor dimension since objective expects explicit
        # output dimension.
        Y_dim = len(_shape(Y))
        Y = posterior_transform.evaluate(Y=Y, X=X_baseline)
        if len(_shape(Y)) < Y_dim:
            if is_torch_tensor(Y):
                Y = Y.unsqueeze(-1)
            else:
                data, shape = to_flat(Y, "Y")
                Y = from_flat(data, (*shape, 1), kind_of(Y))
    if objective is None:
        if _shape(Y)[-1] > 1:
            raise UnsupportedError(
                "Acquisition functions require an objective when "
                "used with multi-output models (except for multi-objective"
                "acquisition functions)."
            )
        objective = IdentityMCObjective()
    if native and isinstance(objective, LinearMCObjective):
        # ``LinearMCObjective`` of a single sample: ``obj = Y @ weights``.
        obj = _linear_objective_values(Y, objective._weights)
    elif native:
        # The identity objective of a single sample: ``obj = Y.squeeze(-1)``.
        obj = _squeeze_last(Y)
    else:
        # ``Y`` is of shape ``(batch_shape) x q x m``; ``MCAcquisitionObjective``s
        # expect inputs ``sample_shape x (batch_shape) x q x m``.
        # For most objectives, ``obj`` will have shape ``1 x (batch_shape) x q``, but
        # with a ``LearnedObjective`` it can be ``num_samples x (batch_shape) x q``.
        obj = objective(Y.unsqueeze(0), X=X_baseline)
        obj = obj.mean(dim=0)  # taking mean over monte carlo samples
    best = compute_best_feasible_objective(
        samples=Y,
        obj=obj,
        constraints=constraints,
        model=model,
        objective=objective,
        posterior_transform=posterior_transform,
        X_baseline=X_baseline,
    )
    return _restore(best, kind)


def optimize_objective(
    model: Any,
    bounds: Any,
    q: int,
    acq_function: AcquisitionFunction | None = None,
    objective: Any = None,
    posterior_transform: Any = None,
    linear_constraints: tuple[Any, Any] | None = None,
    fixed_features: dict[int, float] | None = None,
    qmc: bool = True,
    mc_samples: int = 512,
    seed_inner: int | None = None,
    optimizer_options: dict[str, Any] | None = None,
    post_processing_func: Callable[[Any], Any] | None = None,
    batch_initial_conditions: Any = None,
    sequential: bool = False,
) -> tuple[Any, Any]:
    r"""Optimize an objective under the given model.

    Args:
        model: The model to be used in the objective.
        bounds: A ``2 x d`` tensor of lower and upper bounds for each column of ``X``.
        q: The cardinality of input sets on which the objective is to be evaluated.
        objective: The objective to optimize.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        linear_constraints: A tuple of (A, b). Given ``k`` linear constraints on a
            ``d``-dimensional space, ``A`` is ``k x d`` and ``b`` is ``k x 1`` such that
            ``A x <= b``. (Not used by single task models).
        fixed_features: A dictionary of feature assignments
            ``{feature_index: value}`` to hold fixed during generation.
        qmc: Toggle for enabling (qmc=1) or disabling (qmc=0) use of Quasi Monte Carlo.
        mc_samples: Integer number of samples used to estimate Monte Carlo objectives.
        seed_inner: Integer seed used to initialize the sampler passed to MCObjective.
        optimizer_options: Table used to lookup keyword arguments for the optimizer.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e. according to ``round-trip`` transformations).
        batch_initial_conditions: A Tensor of initial values for the optimizer.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.

    Returns:
        A tuple containing the best input locations and corresponding objective values.
    """
    from botorch_rs.optim.optimize import optimize_acqf

    if optimizer_options is None:
        optimizer_options = {}

    if acq_function is None:
        if objective is None:
            acq_function = PosteriorMean(model=model, posterior_transform=posterior_transform)
        else:
            if not qmc:
                raise UnsupportedError(
                    "qmc=False (an IIDNormalSampler) is not supported by botorch_rs; use qmc=True."
                )
            acq_function = qSimpleRegret(
                model=model,
                objective=objective,
                posterior_transform=posterior_transform,
                sampler=SobolQMCNormalSampler(sample_shape=(mc_samples,), seed=seed_inner),
            )

    d = _shape(bounds)[-1]
    if fixed_features:
        acq_function = FixedFeatureAcquisitionFunction(
            acq_function=acq_function,
            d=d,
            columns=list(fixed_features.keys()),
            values=list(fixed_features.values()),
        )
        free_feature_dims = sorted(set(range(d)) - set(fixed_features.keys()))
        if is_torch_tensor(bounds):
            free_feature_bounds = bounds[:, free_feature_dims]  # (2, d' <= d)
        else:
            bdata, _ = to_flat(bounds, "bounds")
            free_feature_bounds = from_flat(
                [bdata[r * d + j] for r in range(2) for j in free_feature_dims],
                (2, len(free_feature_dims)),
                kind_of(bounds),
            )
    else:
        free_feature_bounds = bounds

    if linear_constraints is None:
        inequality_constraints = None
    else:
        A, b = linear_constraints
        inequality_constraints = []
        if is_torch_tensor(A):
            k = A.shape[0]
            for i in range(k):
                indices = A[i, :].nonzero(as_tuple=False).squeeze()
                coefficients = -A[i, indices]
                rhs = -b[i, 0]
                inequality_constraints.append((indices, coefficients, rhs))
        else:
            adata, (k, da) = to_flat(A, "A")
            bdata, _ = to_flat(b, "b")
            for i in range(k):
                idx = [j for j in range(da) if adata[i * da + j] != 0]
                inequality_constraints.append((idx, [-adata[i * da + j] for j in idx], -bdata[i]))

    options = {
        "batch_limit": optimizer_options.get("batch_limit", 8),
        "maxiter": optimizer_options.get("maxiter", 200),
        "nonnegative": optimizer_options.get("nonnegative", False),
    }
    if "method" in optimizer_options:
        options["method"] = optimizer_options.pop("method")

    return optimize_acqf(
        acq_function=acq_function,
        bounds=free_feature_bounds,
        q=q,
        num_restarts=optimizer_options.get("num_restarts", 60),
        raw_samples=optimizer_options.get("raw_samples", 1024),
        options=options,
        inequality_constraints=inequality_constraints,
        fixed_features=None,  # handled inside the acquisition function
        post_processing_func=post_processing_func,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=True,
        sequential=sequential,
    )


@acqf_input_constructor(qJointEntropySearch)
def construct_inputs_qJES(
    model: Any,
    bounds: list[tuple[float, float]],
    num_optima: int = 64,
    condition_noiseless: bool = True,
    posterior_transform: Any = None,
    X_pending: Any = None,
    estimation_type: str = "LB",
    num_samples: int = 64,
) -> dict[str, Any]:
    """Construct inputs for ``qJointEntropySearch``: optimal samples from
    ``get_optimal_samples`` (botorch_rs's paths and optimizer)."""
    like = model.train_inputs[0]
    optimal_inputs, optimal_outputs = get_optimal_samples(
        model=model,
        bounds=_transpose(_bounds_tensor(bounds, like)),
        num_optima=num_optima,
        posterior_transform=posterior_transform,
        return_transformed=True,
    )

    inputs = {
        "model": model,
        "optimal_inputs": optimal_inputs,
        "optimal_outputs": optimal_outputs,
        "condition_noiseless": condition_noiseless,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "estimation_type": estimation_type,
        "num_samples": num_samples,
    }
    return inputs


@acqf_input_constructor(qBayesianActiveLearningByDisagreement)
def construct_inputs_BALD(
    model: Any,
    X_pending: Any = None,
    sampler: Any = None,
    posterior_transform: Any = None,
) -> dict[str, Any]:
    """Construct inputs for ``qBayesianActiveLearningByDisagreement`` (a
    fully Bayesian model)."""
    inputs = {
        "model": model,
        "X_pending": X_pending,
        "sampler": sampler,
        "posterior_transform": posterior_transform,
    }
    return inputs


@acqf_input_constructor(qNegIntegratedPosteriorVariance)
def construct_inputs_NIPV(
    model: Any,
    bounds: list[tuple[float, float]],
    num_mc_points: int = 128,
    X_pending: Any = None,
    posterior_transform: Any = None,
) -> dict[str, Any]:
    """Construct inputs for qNegIntegratedPosteriorVariance (the integration
    points from botorch_rs's ``draw_sobol_samples``)."""
    from botorch_rs.utils import sampling as _sampling

    like = model.train_inputs[0]
    b = _transpose(_bounds_tensor(bounds, like))
    mc_points = _sampling.draw_sobol_samples(bounds=b, n=num_mc_points, q=1)
    if is_torch_tensor(mc_points):
        mc_points = mc_points.squeeze(-2)
    else:
        data, shape = to_flat(mc_points, "mc_points")
        mc_points = from_flat(data, (shape[0], shape[-1]), kind_of(mc_points))
    inputs = {
        "model": model,
        "mc_points": mc_points,
        "X_pending": X_pending,
        "posterior_transform": posterior_transform,
    }
    return inputs


def _get_ref_point(
    objective_thresholds: Any = None,
    objective: Any = None,
    ref_point: Any = None,
) -> Any:
    """Get the reference point for multi-objective acquisition functions.

    Args:
        objective_thresholds: Deprecated. Raw objective thresholds that will be
            transformed through the objective (if provided) to produce the
            reference point. Use ``ref_point`` instead.
        objective: The multi-output objective, used only with the deprecated
            ``objective_thresholds`` path to transform thresholds into the
            objective space.
        ref_point: The maximization-aligned reference point of shape
            ``(num_objectives,)``, used directly without any further processing.
            This is the preferred way to specify the reference point.

    Returns:
        A ``(num_objectives,)``-dim Tensor representing the reference point
        in the objective space, suitable for hypervolume computation.
    """
    if ref_point is not None:
        if objective_thresholds is not None:
            raise ValueError("Cannot specify both `ref_point` and `objective_thresholds`.")
        return ref_point
    if objective_thresholds is None:
        raise ValueError("Either `ref_point` or `objective_thresholds` must be provided.")
    warnings.warn(
        "`objective_thresholds` is deprecated in favor of `ref_point`. "
        "Unlike `objective_thresholds`, which gets transformed through the "
        "objective, `ref_point` should be a maximization-aligned reference "
        "point of shape `(num_objectives,)` and is used directly without "
        "any further processing.",
        DeprecationWarning,
        stacklevel=3,
    )
    if objective is None:
        return objective_thresholds
    return _apply_objective(objective, objective_thresholds)


def _construct_constraint_dict_from_tuple(
    constraints_tuple: tuple[Any, Any], acqf_class: type[AcquisitionFunction]
) -> dict[int, Any]:
    """
    Construct a dictionary of the form ``{i: [lower, upper]}``,
    where ``i`` is the output index, and ``lower`` and ``upper`` are
    lower and upper bounds on that output (resp. interpreted
    as -Inf / Inf if None).
    """
    weights, bounds = constraints_tuple
    if is_torch_tensor(weights):
        torch = _torch()
        constraints_dict: dict[int, Any] = {}
        for w, b in zip(weights, bounds, strict=False):
            nonzero_w = w.nonzero()
            if nonzero_w.numel() != 1:
                raise BotorchError(
                    f"{acqf_class.__name__} only support constraints on single outcomes."
                )
            i = nonzero_w.item()
            w_i = w[i]
            is_ub = torch.sign(w_i) == 1.0
            b = b.item()
            bound = (None, b / w_i) if is_ub else (b / w_i, None)
            constraints_dict[i] = bound
        return constraints_dict
    wdata, (k, m) = to_flat(weights, "weights")
    bdata, _ = to_flat(bounds, "bounds")
    out: dict[int, Any] = {}
    for r in range(k):
        nz = [j for j in range(m) if wdata[r * m + j] != 0]
        if len(nz) != 1:
            raise BotorchError(
                f"{acqf_class.__name__} only support constraints on single outcomes."
            )
        i = nz[0]
        w_i = wdata[r * m + i]
        out[i] = (None, bdata[r] / w_i) if w_i > 0 else (bdata[r] / w_i, None)
    return out


__all__ = [
    "ACQF_INPUT_CONSTRUCTOR_REGISTRY",
    "acqf_input_constructor",
    "allow_only_specific_variable_kwargs",
    "construct_inputs_BALD",
    "construct_inputs_EHVI",
    "construct_inputs_LogPF",
    "construct_inputs_NIPV",
    "construct_inputs_best_f",
    "construct_inputs_logcei",
    "construct_inputs_mf_base",
    "construct_inputs_noisy_ei",
    "construct_inputs_pof",
    "construct_inputs_posterior_mean",
    "construct_inputs_qEHVI",
    "construct_inputs_qEI",
    "construct_inputs_qHVKG",
    "construct_inputs_qJES",
    "construct_inputs_qKG",
    "construct_inputs_qLogEI",
    "construct_inputs_qLogNEHVI",
    "construct_inputs_qLogNEI",
    "construct_inputs_qLogNParEGO",
    "construct_inputs_qMES",
    "construct_inputs_qMFHVKG",
    "construct_inputs_qMFKG",
    "construct_inputs_qMFMES",
    "construct_inputs_qNEHVI",
    "construct_inputs_qNEI",
    "construct_inputs_qPI",
    "construct_inputs_qSimpleRegret",
    "construct_inputs_qUCB",
    "construct_inputs_ucb",
    "get_acqf_input_constructor",
    "get_best_f_analytic",
    "get_best_f_mc",
    "optimize_objective",
]
