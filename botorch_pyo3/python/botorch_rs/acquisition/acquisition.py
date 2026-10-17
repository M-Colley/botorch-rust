"""Base classes of the acquisition functions (``botorch.acquisition.acquisition``)."""

from __future__ import annotations

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
    ``SingleTaskGP``, a ``ModelListGP``, a ``MultiTaskGP`` (also an
    ``LCEMGP``), a ``KroneckerMultiTaskGP``, or a ``PosteriorMeanModel`` or
    ``GenericDeterministicModel``."""
    from botorch_rs.models.deterministic import GenericDeterministicModel, PosteriorMeanModel

    return isinstance(
        model,
        (
            SingleTaskGP,
            ModelListGP,
            MultiTaskGP,
            KroneckerMultiTaskGP,
            PosteriorMeanModel,
            GenericDeterministicModel,
        ),
    )


#: The error message for other models.
ACQUISITION_MODEL_MESSAGE = (
    "botorch_rs acquisition functions require a botorch_rs SingleTaskGP, ModelListGP, "
    "MultiTaskGP, KroneckerMultiTaskGP or LCEMGP model (or a PosteriorMeanModel or "
    "GenericDeterministicModel)."
)


class AcquisitionFunction:
    """Base class of the acquisition functions.

    ``acqf(X)`` evaluates the candidate sets ``X`` (``batch_shape x q x d``, or
    ``q x d`` for a single set) and returns ``batch_shape`` values (``(1,)`` for
    a single set), like BoTorch. If ``X`` is a torch tensor that requires
    gradients, the result is connected to the autograd graph (with exact
    gradients computed in Rust).
    """

    _rs: _rs.AcquisitionFunction
    _log: bool = False

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

    def _model_handle(self, posterior_transform: Any = None) -> _RsModel:
        """The model seen through ``posterior_transform`` (if given)."""
        return self.model._acquisition_model(posterior_transform)

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
        if is_torch_tensor(X) and X.dim() >= 2:
            from botorch_rs import _torch_autograd

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
        flat, kind, shape = self._candidates(X)
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


class OneShotAcquisitionFunction(AcquisitionFunction):
    """Base class of the one-shot acquisition functions (e.g. the hypervolume
    knowledge gradient), whose candidate sets are augmented by auxiliary
    points that ``optimize_acqf`` optimizes jointly with the candidates."""

    def get_augmented_q_batch_size(self, q: int) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def extract_candidates(self, X_full: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["AcquisitionFunction", "OneShotAcquisitionFunction"]
