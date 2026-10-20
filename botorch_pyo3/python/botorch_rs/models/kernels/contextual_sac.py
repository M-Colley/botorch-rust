"""Structural additive contextual covariance (``botorch.models.kernels.contextual_sac``)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, from_flat
from botorch_rs._module import GPModule
from botorch_rs.constraints import Interval
from botorch_rs.kernels import Kernel, ScaleKernel
from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch_rs.priors import GammaPrior, Prior


class _ContextKernels(dict[str, ScaleKernel]):
    """ModuleDict-style dotted lookup used by ``Kernel.initialize``."""

    def __getattr__(self, name: str) -> ScaleKernel:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class SACKernel(Kernel):
    r"""Sum of scaled RBF kernels on equal-sized context decompositions.

    All components share a single ARD lengthscale vector with BoTorch's
    dimension-scaled prior. Each context has its own outputscale with a
    ``GammaPrior(2, 15)``. Values and all model derivatives run in Rust;
    shared lengthscale derivatives accumulate across contexts.

    The shared lengthscale prior is evaluated once. Official BoTorch
    b0dd01be6 inadvertently counts it once per module alias (M+1 times).
    """

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        from botorch_rs.models.map_saas import _use_torch_backend

        device = kwargs.get("device", args[2] if len(args) > 2 else None)
        if cls is SACKernel and (
            _use_torch_backend(backend)
            or (backend == "auto" and device is not None and str(device) != "cpu")
        ):
            from botorch_rs.models._torch_backend_contextual import TorchSACKernel

            return object.__new__(TorchSACKernel)
        return object.__new__(cls)

    def __init__(
        self,
        decomposition: dict[str, list[int]],
        batch_shape: Any,
        device: Any = None,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__(batch_shape=batch_shape)
        if not decomposition:
            raise ValueError("SACKernel requires at least one context.")
        d = len(next(iter(decomposition.values())))
        if d == 0 or any(len(dims) != d for dims in decomposition.values()):
            raise ValueError("num of parameters needs to be same and nonzero across all contexts")
        for name, dims in decomposition.items():
            if not isinstance(name, str) or not name or "." in name:
                raise ValueError("Context names must be nonempty strings without dots.")
            if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in dims):
                raise ValueError("Context parameter indices must be nonnegative integers.")
        self.decomposition = {name: list(dims) for name, dims in decomposition.items()}
        self._device = device
        self._indexers = {name: tuple(dims) for name, dims in self.decomposition.items()}
        self.base_kernel = get_covar_module_with_dim_scaled_prior(d, batch_shape=batch_shape)
        self.kernel_dict = _ContextKernels(
            (name, ScaleKernel(self.base_kernel, outputscale_prior=GammaPrior(2.0, 15.0)))
            for name in self.decomposition
        )

    @property
    def device(self) -> Any:
        return self._device

    @property
    def is_stationary(self) -> bool:
        return True

    def _parameter_modules(self) -> Iterator[tuple[str, Kernel]]:
        yield "base_kernel", self.base_kernel
        for name, kernel in self.kernel_dict.items():
            yield f"kernel_dict.{name}", kernel

    def _own_params(self) -> list[tuple[str, Interval | None, Prior | None]]:
        # Flatten the unique parameters, retaining the public GPyTorch paths.
        # Traversing ordinary ScaleKernel children would duplicate the base.
        return [
            (f"{prefix}.{name}", constraint, prior)
            for prefix, kernel in self._parameter_modules()
            for name, constraint, prior in kernel._own_params()
        ]

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.contextual_sac(
            list(self.decomposition.items()), self.base_kernel._node()
        )

    def _parameter_module(self, name: str) -> tuple[Kernel, str]:
        if name.startswith("base_kernel."):
            return self.base_kernel, name.removeprefix("base_kernel.")
        _, context, raw_name = name.split(".")
        return self.kernel_dict[context], raw_name

    def _raw_values(self, name: str) -> list[float]:
        if self._model is not None:
            return super()._raw_values(name)
        module, raw_name = self._parameter_module(name)
        return module._raw_values(raw_name)

    def _set_raw(self, name: str, value: Any) -> None:
        if self._model is not None:
            super()._set_raw(name, value)
        else:
            module, raw_name = self._parameter_module(name)
            module._set_raw(raw_name, value)

    def _all_kernels(self) -> Iterator[Kernel]:
        yield self
        yield from (kernel for _, kernel in self._parameter_modules())

    def named_sub_kernels(self) -> Iterator[tuple[str, Kernel]]:
        yield from self._parameter_modules()

    def _bind(self, model: Any, prefix: str = "") -> None:
        super()._bind(model, prefix)
        self.base_kernel._bind(model, prefix + "base_kernel.")
        for context, kernel in self.kernel_dict.items():
            # Bind only each scale's unique parameter. Its base is the same
            # object already bound to the canonical top-level base path.
            GPModule._bind(kernel, model)
            kernel._prefix = f"{prefix}kernel_dict.{context}."
            for name, values in kernel._local.items():
                model._set_raw_param(
                    kernel._prefix + name, from_flat(values, kernel._param_shape(name), LIST)
                )
            kernel._local = {}

    forward = Kernel.__call__


__all__ = ["SACKernel"]
