"""Base class of the models (``botorch.models.model``)."""

from __future__ import annotations

from typing import Any


class Model:
    """Abstract base class of botorch_rs models."""

    training: bool = True

    def train(self, mode: bool = True) -> Model:
        """Sets the training flag (for API compatibility; it has no effect)."""
        self.training = mode
        return self

    def eval(self) -> Model:
        return self.train(False)

    def posterior(self, X: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["Model"]
