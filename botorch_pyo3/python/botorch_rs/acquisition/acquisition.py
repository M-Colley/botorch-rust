"""Base classes of the acquisition functions (``botorch.acquisition.acquisition``)."""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    ArrayKind,
    Flat,
    as_float,
    from_flat,
    from_pair,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs._botorch_rs import Model as _RsModel
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP


def check_unsupported(**kwargs: Any) -> None:
    """Raises ``UnsupportedError`` for BoTorch arguments that are set."""
    for name, value in kwargs.items():
        if (
            value is not None
            and value is not False
            and not (isinstance(value, (list, tuple, dict)) and len(value) == 0)
        ):
            raise UnsupportedError(f"`{name}` is not supported by botorch_rs.")


def scalar_arg(value: Any, name: str) -> float:
    """A scalar argument such as ``best_f`` as a float (batched values are
    not supported)."""
    data, _ = to_flat(value, name)
    if len(data) != 1:
        raise UnsupportedError(f"Batched `{name}` (one value per t-batch) is not supported.")
    return as_float(value, name)


def pending_arg(X_pending: Any, d: int) -> Flat | None:
    """``X_pending`` (``m x d``) as a flat array."""
    if X_pending is None:
        return None
    flat = to_flat(X_pending, "X_pending")
    if len(flat[1]) != 2 or flat[1][1] != d:
        raise BotorchTensorDimensionError(
            f"Expected X_pending to be an `m x d` tensor with d = {d}, got shape {flat[1]}."
        )
    return flat


def is_acquisition_model(model: Any) -> bool:
    """Whether botorch_rs acquisition functions can be built on ``model``: a
    ``SingleTaskGP`` (or one of its subclasses), a ``ModelListGP``, a
    ``MultiTaskGP`` (also an ``LCEMGP``), a ``KroneckerMultiTaskGP``, a
    deterministic model (``PosteriorMeanModel``, ``GenericDeterministicModel``,
    ``AffineDeterministicModel``, ``FixedSingleSampleModel``) or a
    ``ModelList`` of them (and, internally, the moments of an
    ``EnsembleModel``)."""
    from botorch_rs.models.approximate_gp import ApproximateGPyTorchModel
    from botorch_rs.models.deterministic import (
        AffineDeterministicModel,
        FixedSingleSampleModel,
        GenericDeterministicModel,
        PosteriorMeanModel,
    )
    from botorch_rs.models.ensemble import _EnsembleMoments
    from botorch_rs.models.model import ModelList
    from botorch_rs.models.pairwise_gp import PairwiseGP

    return isinstance(
        model,
        (
            ApproximateGPyTorchModel,
            SingleTaskGP,
            ModelListGP,
            MultiTaskGP,
            KroneckerMultiTaskGP,
            PosteriorMeanModel,
            GenericDeterministicModel,
            AffineDeterministicModel,
            FixedSingleSampleModel,
            ModelList,
            _EnsembleMoments,
            PairwiseGP,
        ),
    )


#: The error message for other models.
ACQUISITION_MODEL_MESSAGE = (
    "botorch_rs acquisition functions require a botorch_rs SingleTaskGP, ModelListGP, "
    "MultiTaskGP, KroneckerMultiTaskGP or LCEMGP model, a deterministic model or a "
    "ModelList of them."
)


#: The acquisition functions whose values on an ensemble model are the
#: averages (``mean``, or ``logmeanexp`` for log-scale values) of their
#: values on the members, with the same arguments and Monte-Carlo base
#: samples (BoTorch's ``average_over_ensemble_models`` and the ensemble
#: dimension of ``SampleReducingMCAcquisitionFunction``), verified against
#: BoTorch.
ENSEMBLE_ACQUISITION_FUNCTIONS = frozenset(
    {
        "ExpectedImprovement",
        "LogExpectedImprovement",
        "ProbabilityOfImprovement",
        "LogProbabilityOfImprovement",
        "UpperConfidenceBound",
        "PosteriorMean",
        "PosteriorStandardDeviation",
        "qExpectedImprovement",
        "qLogExpectedImprovement",
        "qProbabilityOfImprovement",
        "qSimpleRegret",
        "qUpperConfidenceBound",
        "qLowerConfidenceBound",
        "qPosteriorStandardDeviation",
        "qNoisyExpectedImprovement",
        "qLogNoisyExpectedImprovement",
        # Multi-objective (lists of fully Bayesian GPs, phase 10a): BoTorch's
        # `marginalize_dim` / `average_over_ensemble_models`.
        "qExpectedHypervolumeImprovement",
        "qNoisyExpectedHypervolumeImprovement",
        "qLogExpectedHypervolumeImprovement",
        "qLogNoisyExpectedHypervolumeImprovement",
    }
)


#: The analytic acquisition functions that only use the marginal means and
#: variances of the posterior (BoTorch's ``_mean_and_sigma``): on an
#: ``EnsembleModel`` (whose posterior is an ``EnsemblePosterior``, not an
#: averaged ensemble), they use the ensemble's mean and variance.
_GAUSSIAN_MOMENT_ACQUISITION_FUNCTIONS = frozenset(
    {
        "ExpectedImprovement",
        "LogExpectedImprovement",
        "ProbabilityOfImprovement",
        "LogProbabilityOfImprovement",
        "UpperConfidenceBound",
        "PosteriorMean",
        "PosteriorStandardDeviation",
        "ConstrainedExpectedImprovement",
        "LogConstrainedExpectedImprovement",
        "LogProbabilityOfFeasibility",
    }
)


def _supports_ensemble(cls: type, names: frozenset[str] = ENSEMBLE_ACQUISITION_FUNCTIONS) -> bool:
    return any(c.__name__ in names and c.__module__.startswith("botorch_rs.") for c in cls.__mro__)


def _init_on_ensemble(self: Any, init: Any, args: tuple, kwargs: dict[str, Any]) -> None:
    """Builds the acquisition function ``self`` of an ensemble model as the
    (weighted) average of the acquisition functions of its members (see
    ``EnsembleAverage``): every member's is built by ``init`` with the same
    arguments and one shared sampler. For the noisy expected improvements,
    the baseline is pruned on the ensemble first (BoTorch's
    ``prune_inferior_points``). The analytic acquisition functions of an
    ``EnsembleModel`` (``_GAUSSIAN_MOMENT_ACQUISITION_FUNCTIONS``) are not
    averaged: they use the mean and variance of its ``EnsemblePosterior``,
    as BoTorch's."""
    from botorch_rs.acquisition.utils import _prune_ensemble

    cls = type(self)
    model: Any = args[0] if args else kwargs.get("model")
    fully_bayesian_error = getattr(cls, "_FULLY_BAYESIAN_ERROR", None)
    if fully_bayesian_error is not None and getattr(model, "_is_fully_bayesian", False):
        raise NotImplementedError(fully_bayesian_error)
    if not hasattr(model, "_ensemble_members") and _supports_ensemble(
        cls, _GAUSSIAN_MOMENT_ACQUISITION_FUNCTIONS
    ):
        # An analytic acquisition function of an EnsembleModel: the Gaussian
        # moments of its EnsemblePosterior (not an average over members).
        from botorch_rs.models.ensemble import _EnsembleMoments

        if args:
            init(self, _EnsembleMoments(model), *args[1:], **kwargs)
        else:
            init(self, **{**kwargs, "model": _EnsembleMoments(model)})
        self.model = model
        return
    if not _supports_ensemble(cls):
        raise UnsupportedError(
            f"{cls.__name__} does not support ensemble models (EnsembleMapSaasSingleTaskGP, "
            "EnsembleModel, fully Bayesian GPs) in botorch_rs; supported are "
            f"{sorted(ENSEMBLE_ACQUISITION_FUNCTIONS)} (and, on an EnsembleModel, the analytic "
            f"acquisition functions {sorted(_GAUSSIAN_MOMENT_ACQUISITION_FUNCTIONS)})."
        )
    sig = inspect.signature(init)
    params = list(sig.parameters.values())
    bound = sig.bind(None, *args, **kwargs)
    arguments = dict(bound.arguments)
    arguments.pop(params[0].name)
    extra: dict[str, Any] = {}
    for p in params:
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            extra = arguments.pop(p.name, {})
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            raise UnsupportedError(f"{cls.__name__} is not supported with ensemble models.")
    names = sig.parameters
    model_name = params[1].name
    deterministic_members = not hasattr(model, "_ensemble_members")
    if deterministic_members and "sampler" not in names:
        raise UnsupportedError(
            "botorch_rs supports the Monte-Carlo acquisition functions (sampled with an "
            "IndexSampler) and the analytic acquisition functions "
            f"{sorted(_GAUSSIAN_MOMENT_ACQUISITION_FUNCTIONS)} of an EnsembleModel, not "
            f"{cls.__name__}."
        )
    members, weights, sampler = model._ensemble_acquisition(arguments.get("sampler"))
    if "sampler" in names:
        arguments["sampler"] = sampler
    if "marginalize_dim" in names:
        # The ensemble dimension (BoTorch's MCMC_DIM) is the only one
        # botorch_rs marginalizes; the members have none.
        if arguments.get("marginalize_dim") not in (None, -3):
            raise UnsupportedError(
                "botorch_rs marginalizes the ensemble dimension of an ensemble model "
                "(marginalize_dim=-3)."
            )
        arguments["marginalize_dim"] = None
    prepare = getattr(cls, "_prepare_ensemble", None)
    if prepare is not None and not deterministic_members:
        # Class-specific arguments of the members (e.g. qLogNParEGO's).
        prepare(init, model, members, arguments, model_name)
    if "prune_baseline" in names:
        prune = arguments.get("prune_baseline", names["prune_baseline"].default)
        if prune:
            for key in ("objective", "posterior_transform", "constraints"):
                if arguments.get(key) is not None:
                    raise UnsupportedError(
                        "Pruning the baseline of an ensemble model with an objective, a "
                        "posterior transform or constraints is not supported by botorch_rs; "
                        "pass prune_baseline=False."
                    )
            if "ref_point" in names:
                from botorch_rs.acquisition.multi_objective.utils import (
                    _prune_ensemble_multi_objective,
                )

                arguments["X_baseline"] = _prune_ensemble_multi_objective(
                    model, arguments["X_baseline"], arguments["ref_point"], members=members
                )
            else:
                arguments["X_baseline"] = _prune_ensemble(
                    model, arguments["X_baseline"], members=members
                )
            arguments["prune_baseline"] = False
    # One `best_f` per member (BoTorch broadcasts a `best_f` with one value
    # per MCMC sample against the ensemble dimension, e.g. that of
    # `get_acquisition_function` from the samples' posterior means).
    per_member: list[dict[str, Any]] = [{} for _ in members]
    best_f = arguments.get("best_f")
    if best_f is not None and not deterministic_members and len(members) > 1:
        values, _ = to_flat(best_f, "best_f")
        if len(values) == len(members):
            kind = kind_of(best_f)
            per_member = [{"best_f": from_flat([v], (), kind)} for v in values]
    objs = []
    for member, own in zip(members, per_member, strict=True):
        obj = object.__new__(cls)
        init(obj, **{**arguments, model_name: member, **own}, **extra)
        objs.append(obj)
    self.__dict__.update(objs[0].__dict__)
    if per_member[0]:
        self.best_f = best_f
    self.model = model
    if "sampler" in names and not deterministic_members:
        self.sampler = sampler
    if "prune_baseline" in names:
        self.prune_baseline = bool(
            dict(bound.arguments).get("prune_baseline", names["prune_baseline"].default)
        )
    self._ensemble_acqfs = objs
    self._rs = _rs.ensemble_average([o._rs for o in objs], weights)


def _is_ensemble_model(model: Any) -> bool:
    """Whether acquisition functions average over the members of ``model``
    (an ``EnsembleMapSaasSingleTaskGP`` or an ``EnsembleModel``)."""
    return hasattr(model, "_ensemble_acquisition")


class MCSamplerMixin:
    """Sampler functionality shared by Monte Carlo acquisition functions.

    A missing sampler is selected lazily from the posterior type by
    ``get_sampler``, and reused on subsequent calls.
    """

    _default_sample_shape = (512,)

    def __init__(self, sampler: Any = None) -> None:
        self.sampler = sampler

    def get_posterior_samples(self, posterior: Any) -> Any:
        """Draw posterior samples with the configured or lazily chosen sampler."""
        if self.sampler is None:
            from botorch_rs.sampling.get_sampler import get_sampler

            self.sampler = get_sampler(posterior=posterior, sample_shape=self._default_sample_shape)
        return self.sampler(posterior=posterior)

    @property
    def sample_shape(self) -> Any:
        """The sampler's sample shape, or the default before lazy selection."""
        return self._default_sample_shape if self.sampler is None else self.sampler.sample_shape


class AcquisitionFunction:
    """Base class of the acquisition functions.

    ``acqf(X)`` evaluates the candidate sets ``X`` (``batch_shape x q x d``, or
    ``q x d`` for a single set) and returns ``batch_shape`` values (``(1,)`` for
    a single set), like BoTorch. If ``X`` is a torch tensor that requires
    gradients, the result is connected to the autograd graph (with exact
    gradients computed in Rust).

    On an ensemble model (``EnsembleMapSaasSingleTaskGP``, ``EnsembleModel``),
    the acquisition functions in ``ENSEMBLE_ACQUISITION_FUNCTIONS`` average
    their values on the members, as BoTorch's (the analytic ones of an
    ``EnsembleModel`` use the moments of its ``EnsemblePosterior`` instead);
    others raise ``UnsupportedError``.
    """

    _rs: _rs.AcquisitionFunction
    _log: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        model = args[0] if args else kwargs.get("model")
        if model is None:
            model = kwargs.get("acq_function", kwargs.get("base_value_function"))
        if getattr(model, "_tensor_backend", None) == "torch":
            from botorch_rs.acquisition._torch_backend_acquisition import tensor_acquisition

            return tensor_acquisition(cls)
        return object.__new__(cls)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is None or getattr(init, "_ensemble_aware", False):
            return

        @functools.wraps(init)
        def __init__(self: Any, *args: Any, **kw: Any) -> None:
            model = args[0] if args else kw.get("model")
            if _is_ensemble_model(model):
                _init_on_ensemble(self, init, args, kw)
                return
            init(self, *args, **kw)
            # Native wrappers reuse the wrapped acquisition's original model
            # law, which may predate the model's current parameter state.
            for argument in (*args, *kw.values()):
                if isinstance(argument, AcquisitionFunction):
                    inherited = argument.__dict__.get("_parameter_snapshot")
                    if inherited is not None and inherited.model is self.model:
                        self._parameter_snapshot = inherited
                        break

        __init__._ensemble_aware = True  # type: ignore[attr-defined]
        cls.__init__ = __init__  # type: ignore[method-assign]

    def __init__(self, model: Any) -> None:
        if not is_acquisition_model(model):
            raise UnsupportedError(ACQUISITION_MODEL_MESSAGE)
        if model.batch_shape:
            raise UnsupportedError(
                "botorch_rs acquisition functions require unbatched models, got a model with "
                f"batch shape {model.batch_shape}."
            )
        self.model = model
        self._pending_kind: ArrayKind = model._kind
        from botorch_rs.acquisition._parameter_autograd import capture

        self._parameter_snapshot = capture(model)

    def _model_handle(self, posterior_transform: Any = None) -> _RsModel:
        """The model seen through ``posterior_transform`` (if given)."""
        native = self.model._acquisition_model(posterior_transform)
        from botorch_rs.acquisition._parameter_autograd import capture

        self._parameter_snapshot = capture(self.model)
        return native

    def _candidates(self, X: Any) -> tuple[Flat, ArrayKind, tuple[int, ...]]:
        """``X`` as a flat ``batch x q x d`` array, its kind and original shape."""
        kind = kind_of(X)
        data, shape = to_flat(X, "X")
        if len(shape) < 2:
            raise ValueError(
                f"X is {len(shape)}-dim but must be at least 2-dim (`batch_shape x q x d`)."
            )
        shape3 = (1, *shape) if len(shape) == 2 else shape
        return (data, shape3), kind, shape

    def _after_evaluation(self) -> None:
        """Hook run after every evaluation (warnings of the computation)."""

    def forward(self, X: Any) -> Any:
        """Acquisition values at ``X`` (shape ``batch_shape``)."""
        snapshot = self.__dict__.get("_parameter_snapshot")
        if snapshot is not None and not self.__dict__.get("_ensemble_acqfs"):
            snapshot.validate(self.model)
        if is_torch_tensor(X) and X.dim() >= 2:
            from botorch_rs import _torch_autograd

            if snapshot is not None and not self.__dict__.get("_ensemble_acqfs"):
                import torch

                if torch.is_grad_enabled() and any(p.requires_grad for _, p in snapshot.entries):
                    from botorch_rs.acquisition._parameter_autograd import evaluate

                    X3 = X if X.dim() > 2 else X.unsqueeze(0)
                    result = evaluate(self, X3, snapshot)
                    self._after_evaluation()
                    return result
            if _torch_autograd.requires_grad(X):
                X3 = X if X.dim() > 2 else X.unsqueeze(0)
                out = _torch_autograd.evaluate(self._rs, X3)
                self._after_evaluation()
                return out
        flat, kind, _ = self._candidates(X)
        values, batch = self._rs.forward(flat)
        self._after_evaluation()
        return from_flat(values, batch, kind)

    def __call__(self, X: Any) -> Any:
        return self.forward(X)

    def value_and_grad(self, X: Any) -> tuple[Any, Any]:
        """Acquisition values (shape ``batch_shape``) and their gradients with
        respect to ``X`` (shape of ``X``) -- a botorch_rs extension."""
        if getattr(self, "_tensor_backend", None) == "torch":
            import torch

            with torch.enable_grad():
                variable = X if X.requires_grad else X.detach().requires_grad_(True)
                value = self(variable)
                gradient = torch.autograd.grad(value.sum(), variable, create_graph=True)[0]
            return value, gradient
        snapshot = self.__dict__.get("_parameter_snapshot")
        if snapshot is not None and not self.__dict__.get("_ensemble_acqfs"):
            snapshot.validate(self.model)
        flat, kind, shape = self._candidates(X)
        use_tape = False
        if (
            snapshot is not None
            and snapshot.model_error is None
            and not self.__dict__.get("_ensemble_acqfs")
            and is_torch_tensor(X)
            and self._rs.supports_parameter_gradients
        ):
            import torch

            use_tape = torch.is_grad_enabled() and any(p.requires_grad for _, p in snapshot.entries)
        if use_tape:
            # Match forward's Gaussian-adjoint evaluation exactly: the optimized
            # query-only analytic route can differ by a few float64 ULPs.
            (values, batch), gradient, _ = self._rs.forward_with_adjoint_tape(flat, True)
            assert gradient is not None
            grads, _ = gradient
        else:
            (values, batch), (grads, _) = self._rs.forward_with_grad(flat)
        self._after_evaluation()
        return from_flat(values, batch, kind), from_flat(grads, shape, kind)

    def set_X_pending(self, X_pending: Any = None) -> None:
        """Informs the acquisition function about pending points (``m x d``);
        ``None`` removes them."""
        flat = pending_arg(X_pending, self._rs.dim)
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        self._rs.set_x_pending(flat)

    @property
    def X_pending(self) -> Any:
        """The pending points (``m x d``), or ``None``."""
        pending = self._rs.x_pending()
        return None if pending is None else from_pair(pending, self._pending_kind)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MultiModelAcquisitionFunction(AcquisitionFunction, ABC):
    """Base for custom acquisition functions using a dictionary of models."""

    def __init__(self, model_dict: Any) -> None:
        self.model_dict = model_dict

    @abstractmethod
    def forward(self, X: Any) -> Any:
        """Evaluate the custom acquisition function using ``model_dict``."""
        raise NotImplementedError


class OneShotAcquisitionFunction(AcquisitionFunction):
    """Base class of the one-shot acquisition functions (e.g. the hypervolume
    knowledge gradient), whose candidate sets are augmented by auxiliary
    points that ``optimize_acqf`` optimizes jointly with the candidates."""

    def get_augmented_q_batch_size(self, q: int) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def extract_candidates(self, X_full: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = [
    "ENSEMBLE_ACQUISITION_FUNCTIONS",
    "AcquisitionFunction",
    "MCSamplerMixin",
    "MultiModelAcquisitionFunction",
    "OneShotAcquisitionFunction",
]
