"""Adapter for a caller-supplied PyTorch distribution.

PyTorch is imported only when this adapter is constructed or used. Native
GP models continue to evaluate and sample their posteriors in Rust.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from botorch_rs.posteriors.gpytorch import Posterior


def _parameter_tensor(value: Any, tensor_type: type, visited: set[int]) -> Any:
    """Find tensor parameters inside distributions and their containers."""
    if isinstance(value, tensor_type):
        return value
    identity = id(value)
    if identity in visited:
        return None
    visited.add(identity)
    children: Iterable[Any]
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    elif hasattr(value, "__dict__"):
        children = vars(value).values()
    else:
        return None
    for child in children:
        tensor = _parameter_tensor(child, tensor_type, visited)
        if tensor is not None:
            return tensor
    return None


class TorchPosterior(Posterior):
    """Expose a supplied PyTorch distribution through the posterior API.

    Attributes not defined on the posterior are forwarded to the distribution.
    Nested parameter tensors supply device and dtype metadata, including for
    ``Independent`` and ``TransformedDistribution`` wrappers.
    """

    def __init__(self, distribution: Any) -> None:
        import torch

        self.distribution = distribution
        parameter = _parameter_tensor(distribution, torch.Tensor, set())
        if parameter is None:
            raise TypeError(
                "Cannot infer device and dtype: the distribution has no tensor parameters."
            )
        self._device = parameter.device
        self._dtype = parameter.dtype

    def rsample(self, sample_shape: Any = None) -> Any:
        """Reparameterized samples; the default sample shape is empty."""
        import torch

        shape = torch.Size() if sample_shape is None else torch.Size(sample_shape)
        return self.distribution.rsample(sample_shape=shape)

    @property
    def device(self) -> Any:
        return self._device

    @property
    def dtype(self) -> Any:
        return self._dtype

    def __getattr__(self, name: str) -> Any:
        distribution = self.__dict__.get("distribution")
        if distribution is None:
            raise AttributeError(name)
        return getattr(distribution, name)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _value(self, value: Any) -> Any:
        import torch

        return (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, dtype=self.dtype, device=self.device)
        )

    def quantile(self, value: Any) -> Any:
        """Quantiles computed by the caller's distribution's ``icdf``."""
        import torch

        value = self._value(value)
        if value.numel() > 1:
            return torch.stack([self.quantile(v) for v in value], dim=0)
        return self.distribution.icdf(value)

    def density(self, value: Any) -> Any:
        """Density or probability mass from the distribution's ``log_prob``."""
        import torch

        value = self._value(value)
        if value.numel() > 1:
            return torch.stack([self.density(v) for v in value], dim=0)
        return self.distribution.log_prob(value).exp()

    def _extended_shape(self, sample_shape: Any = ()) -> Any:
        import torch

        return self.distribution._extended_shape(sample_shape=torch.Size(sample_shape))


__all__ = ["TorchPosterior"]
