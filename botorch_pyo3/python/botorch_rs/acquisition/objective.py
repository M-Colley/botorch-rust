"""Posterior transforms (``botorch.acquisition.objective``):
``ScalarizedPosteriorTransform``."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import as_float, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError


class PosteriorTransform:
    """Base class of the posterior transforms."""

    #: Whether the transform maps a multi-output posterior to a single output.
    scalarize: bool = False

    def evaluate(self, Y: Any, X: Any = None) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, posterior: Any, X: Any = None) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, posterior: Any, X: Any = None) -> Any:
        return self.forward(posterior, X)


class ScalarizedPosteriorTransform(PosteriorTransform):
    r"""An affine posterior transform for scalarizing multi-output posteriors.

    For a Gaussian posterior with mean ``mu`` and covariance ``Sigma`` over
    ``m`` outputs, this yields the single-output posterior of
    ``offset + weights^T f`` (mean ``offset + weights^T mu``, covariance
    ``weights^T Sigma weights`` for every pair of points).

    Example:
        >>> weights = torch.tensor([0.5, 0.25], dtype=torch.float64)
        >>> posterior_transform = ScalarizedPosteriorTransform(weights)
        >>> EI = ExpectedImprovement(model, best_f=0.1, posterior_transform=posterior_transform)
    """

    scalarize = True

    def __init__(self, weights: Any, offset: float = 0.0) -> None:
        data, shape = to_flat(weights, "weights")
        if len(shape) != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.weights = weights
        self.offset = offset
        self._weights = [float(w) for w in data]
        self._offset = as_float(offset, "offset")

    def evaluate(self, Y: Any, X: Any = None) -> Any:
        """The transform applied to outcomes ``Y`` (``batch_shape x q x m``):
        ``offset + Y @ weights`` (shape ``batch_shape x q``)."""
        data, shape = to_flat(Y, "Y")
        m = len(self._weights)
        if not shape or shape[-1] != m:
            raise BotorchTensorDimensionError(
                f"Expected outcomes with {m} outputs (the number of weights), got shape {shape}."
            )
        out = [
            self._offset + sum(data[i * m + t] * self._weights[t] for t in range(m))
            for i in range(len(data) // m)
        ]
        return from_flat(out, tuple(shape[:-1]), kind_of(Y))

    def forward(self, posterior: Any, X: Any = None) -> Any:
        """The scalarized (single-output) posterior of a botorch_rs posterior."""
        from botorch_rs.posteriors.gpytorch import GPyTorchPosterior

        if not isinstance(posterior, GPyTorchPosterior):
            raise UnsupportedError("ScalarizedPosteriorTransform requires a botorch_rs posterior.")
        if posterior._transform is not None:
            raise UnsupportedError("The posterior has already been transformed.")
        if posterior._m != len(self._weights):
            raise RuntimeError(
                "Output shape not equal to that of weights. Output shape is "
                f"{posterior._m} and weights are {len(self._weights)}"
            )
        return GPyTorchPosterior(
            posterior._rs,
            posterior._x,
            posterior._kind,
            posterior._observation_noise,
            num_outputs=posterior._m,
            model_batch_shape=posterior._batch,
            output_indices=posterior._output_indices,
            transform=(self._weights, self._offset),
        )

    def __repr__(self) -> str:
        return f"ScalarizedPosteriorTransform(weights={self._weights}, offset={self._offset})"


def transform_arg(
    posterior_transform: Any, num_outputs: int | None = None
) -> tuple[list[float], float] | None:
    """The ``(weights, offset)`` of a posterior transform for the extension
    (``None`` without a transform), checked against ``num_outputs``."""
    if posterior_transform is None:
        return None
    if not isinstance(posterior_transform, ScalarizedPosteriorTransform):
        raise UnsupportedError(
            "botorch_rs supports posterior transforms of type "
            "botorch_rs.acquisition.objective.ScalarizedPosteriorTransform, got "
            f"{type(posterior_transform).__name__}."
        )
    weights = posterior_transform._weights
    if num_outputs is not None and len(weights) != num_outputs:
        raise RuntimeError(
            "Output shape not equal to that of weights. Output shape is "
            f"{num_outputs} and weights are {len(weights)}"
        )
    return weights, posterior_transform._offset


__all__ = ["PosteriorTransform", "ScalarizedPosteriorTransform"]
