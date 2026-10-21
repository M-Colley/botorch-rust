"""Explicit copying conversions between optional PyTorch and JAX arrays."""

from __future__ import annotations

import importlib
from typing import Any


def _optional(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            "The JAX conversion utilities require torch, numpy and jax. "
            "Install the missing optional libraries before calling them."
        ) from error


def torch_to_jax(t: Any) -> Any:
    """Detach and copy a torch tensor to a JAX array on JAX's default device."""
    jnp = _optional("jax.numpy")
    return jnp.array(t.detach().cpu().numpy(), copy=True)


def jax_to_torch(a: Any, device: Any, dtype: Any) -> Any:
    """Copy a JAX array to a torch tensor of the requested device and dtype."""
    torch = _optional("torch")
    np = _optional("numpy")
    return torch.tensor(np.asarray(a), device=device, dtype=dtype)


__all__ = ["jax_to_torch", "torch_to_jax"]
