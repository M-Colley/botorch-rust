r"""Abstract base class of decoupled acquisition functions
(``botorch.acquisition.decoupled``): the bookkeeping of the evaluation masks
(which outcomes are evaluated for every candidate and pending point).

Subclasses provide the Rust acquisition function (``_rs``) that uses the
masks, as ``qHypervolumeKnowledgeGradient`` does with its own masks.
"""

from __future__ import annotations

import warnings
from abc import ABC
from typing import Any

from botorch_rs._arrays import is_torch_tensor, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.models.model_list_gp_regression import ModelListGP


def _shape(mask: Any) -> tuple[int, ...]:
    if hasattr(mask, "shape"):
        return tuple(int(s) for s in mask.shape)
    return tuple(to_flat(mask, "mask")[1])


class DecoupledAcquisitionFunction(AcquisitionFunction, ABC):
    """Abstract base class for decoupled acquisition functions.

    A decoupled acquisition function where one may intend to evaluate a design
    on only a subset of the outcomes. The ``X_evaluation_mask`` (``q x m``)
    specifies which outcomes should be evaluated for each design; with ``n'``
    pending points, ``X_pending_evaluation_mask`` (``n' x m``) is concatenated
    with it to obtain the full evaluation mask. If ``X(_pending)_evaluation_mask``
    is None, it is assumed that ``X(_pending)`` will be evaluated on all
    outcomes.
    """

    def __init__(self, model: Any, X_evaluation_mask: Any = None, **kwargs: Any) -> None:
        if not isinstance(model, ModelListGP):
            raise ValueError(f"{self.__class__.__name__} requires using a ModelList.")
        super().__init__(model, **kwargs)
        self.num_outputs = model.num_outputs
        self.X_evaluation_mask = X_evaluation_mask
        self.X_pending_evaluation_mask: Any = None
        self._X_pending: Any = None

    @property
    def X_evaluation_mask(self) -> Any:
        r"""Get the evaluation indices for the new candidate."""
        return self._X_evaluation_mask

    @X_evaluation_mask.setter
    def X_evaluation_mask(self, X_evaluation_mask: Any = None) -> None:
        r"""Set the evaluation indices for the new candidate."""
        if X_evaluation_mask is not None:
            shape = _shape(X_evaluation_mask)
            if len(shape) != 2 or shape[-1] != self.num_outputs:
                raise BotorchTensorDimensionError(
                    "Expected X_evaluation_mask to be `q x m`, but got shape"
                    f" {'x'.join(str(s) for s in shape)}."
                )
        self._X_evaluation_mask = X_evaluation_mask

    @property
    def X_pending(self) -> Any:
        return self._X_pending

    def set_X_pending(self, X_pending: Any = None, X_pending_evaluation_mask: Any = None) -> None:
        r"""Informs the acquisition function about pending design points for
        different outcomes (``X_pending_evaluation_mask``: ``n' x m``
        booleans; ``None``: all outcomes)."""
        if X_pending is not None:
            if is_torch_tensor(X_pending) and X_pending.requires_grad:
                warnings.warn(
                    "Pending points require a gradient but the acquisition function"
                    " will not provide a gradient to these points.",
                    stacklevel=2,
                )
                X_pending = X_pending.detach().clone()
            self._X_pending = X_pending
            if X_pending_evaluation_mask is not None:
                shape = _shape(X_pending_evaluation_mask)
                n = _shape(X_pending)[0]
                if len(shape) != 2 or shape[0] != n or shape[1] != self.num_outputs:
                    raise BotorchTensorDimensionError(
                        f"Expected `X_pending_evaluation_mask` of shape "
                        f"`{n} x {self.num_outputs}`, but "
                        f"got {'x'.join(str(s) for s in shape)}."
                    )
                self.X_pending_evaluation_mask = X_pending_evaluation_mask
            elif self.X_evaluation_mask is not None:
                raise ValueError(
                    "If `self.X_evaluation_mask` is not None, then "
                    "`X_pending_evaluation_mask` must be provided."
                )
        else:
            self._X_pending = X_pending
            self.X_pending_evaluation_mask = X_pending_evaluation_mask

    def construct_evaluation_mask(self, X: Any) -> Any:
        r"""The boolean evaluation mask (``n + n' x m``) of ``X`` (``batch_shape x
        n x d``) and the pending points."""
        if self.X_pending_evaluation_mask is not None:
            import torch

            X_evaluation_mask = self.X_evaluation_mask
            n = _shape(X)[-2]
            if X_evaluation_mask is None:
                X_evaluation_mask = torch.ones(n, self.num_outputs, dtype=torch.bool)
            elif _shape(X_evaluation_mask)[0] != n:
                raise BotorchTensorDimensionError(
                    "Expected the -2 dimension of X and X_evaluation_mask to match."
                )
            return torch.cat(
                [
                    torch.as_tensor(X_evaluation_mask, dtype=torch.bool),
                    torch.as_tensor(self.X_pending_evaluation_mask, dtype=torch.bool),
                ],
                dim=-2,
            )
        return self.X_evaluation_mask


__all__ = ["DecoupledAcquisitionFunction"]
