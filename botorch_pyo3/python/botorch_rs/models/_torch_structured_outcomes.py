"""Scalar event outcome mappings around structured Matheron posteriors."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.posteriors.transformed import TransformedPosterior


def is_scalar_standardize(transform: Any) -> bool:
    from botorch_rs.models._backend_selection import native_port_descriptor_type
    from botorch_rs.models._torch_backend_modules import TorchStandardize
    from botorch_rs.models.transforms.outcome import Standardize

    return (
        type(transform) is TorchStandardize or native_port_descriptor_type(transform) is Standardize
    )


class _OutcomeMapping(nn.Module):
    transform: Any

    def __init__(self, transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, mode: str, value: Any, X: Tensor) -> Any:
        if mode == "samples":
            return self.transform.untransform(value, X=X)[0]
        posterior = self.transform.untransform_posterior(value, X=X)
        return getattr(posterior, mode)


class TorchStructuredOutcomePosterior(TransformedPosterior):
    """Retain structured base normals while mapping scalar outcome events."""

    _uses_matheron = True

    def __init__(
        self, posterior: Any, transform: nn.Module, X: Tensor, scalar_events: bool = True
    ) -> None:
        self._mapping = _OutcomeMapping(transform)
        # Functional calls preserve source leaf partials and immutable fitted
        # statistics even if another posterior refits the live model transform.
        self._state = {
            name: value.clone()
            for name, value in (
                *self._mapping.named_parameters(),
                *self._mapping.named_buffers(),
            )
        }
        self._context = X
        shape = posterior.mean.shape
        self._flat = posterior
        if scalar_events:
            q, m = shape[-2:]
            index = torch.arange(q * m, device=posterior.device).reshape(m, q).T.flatten()
            covariance = posterior._covariance_tensor.index_select(-2, index).index_select(
                -1, index
            )
            self._flat = TorchGPyTorchPosterior(
                posterior.mean.flatten(-2).unsqueeze(-1), covariance
            )

        def samples(value: Tensor) -> Tensor:
            flat = value.flatten(-2).unsqueeze(-1) if scalar_events else value
            return self._evaluate("samples", flat).reshape(value.shape)

        super().__init__(
            posterior,
            sample_transform=samples,
            mean_transform=lambda mean, variance: self._evaluate("mean", self._flat).reshape(shape),
            variance_transform=lambda mean, variance: self._evaluate(
                "variance", self._flat
            ).reshape(shape),
        )

    def _evaluate(self, mode: str, value: Any) -> Any:
        return torch.func.functional_call(
            self._mapping, self._state, (mode, value, self._context), strict=False
        )

    @property
    def distribution(self) -> Any:
        return self._evaluate("distribution", self._flat)
