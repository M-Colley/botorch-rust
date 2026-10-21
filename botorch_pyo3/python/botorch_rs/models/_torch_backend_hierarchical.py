"""Device-resident hierarchical covariance gates and exact model graphs."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import TorchKernel, build_kernel
from botorch_rs.models._torch_backend_multitask import TorchMultiTaskGP
from botorch_rs.models.hierarchical.conditional_kernel_gp import (
    ATOL,
    RTOL,
    HierarchicalConditionalKernel,
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
)


class TorchHierarchicalConditionalKernel(TorchKernel, HierarchicalConditionalKernel):  # type: ignore[misc]
    """One differentiable covariance per jointly activated feature block."""

    @property
    def batch_shape(self) -> Any:
        return self._tensor_batch_shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._tensor_batch_shape = torch.Size(value)

    def __init__(
        self,
        *args: Any,
        _like: Tensor | None = None,
        _source: HierarchicalConditionalKernel | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("backend", None)
        source = HierarchicalConditionalKernel(*args, **kwargs) if _source is None else _source
        like = torch.empty((), dtype=torch.get_default_dtype()) if _like is None else _like
        TorchKernel.__init__(self, like, source.active_dims)
        self.batch_shape = torch.Size()
        self.dim = source.dim
        self.hierarchical_dependencies = source.hierarchical_dependencies
        self.use_saas_prior = source.use_saas_prior
        self.use_outputscale = source.use_outputscale
        self.partition, self.paths = source.partition, source.paths
        self.ancestor_indices = source.ancestor_indices
        self.kernels = nn.ModuleList(  # type: ignore[assignment]
            [build_kernel(kernel, self.dim, (), like) for kernel in source.kernels]
        )
        self.gate_input_transform: nn.Module | None = None
        self.register_buffer("_gate_template", None)
        self._feature_columns = list(range(self.dim))
        self._output_batch = False

    def configure_gate_transform(
        self, transform: nn.Module | None, train_X: Tensor, feature_columns: list[int]
    ) -> None:
        # The model owns this module; an alias here must not enumerate its
        # parameter priors a second time as covariance priors.
        object.__setattr__(self, "gate_input_transform", transform)
        self._gate_template = train_X[..., :1, :]
        self._feature_columns = feature_columns

    def _targets(self, path: Any, like: Tensor) -> Tensor:
        values = like.new_tensor([value for _, value in path])
        if not path or self.gate_input_transform is None:
            return values
        assert self._gate_template is not None
        columns = torch.tensor(
            [self._feature_columns[parent] for parent, _ in path],
            dtype=torch.long,
            device=like.device,
        )
        template = self._gate_template
        dummy = template.expand(*template.shape[:-2], len(path), template.shape[-1]).clone()
        rows = torch.arange(len(path), device=like.device)
        dummy[..., rows, columns] = values
        transform = getattr(
            self.gate_input_transform, "preprocess_transform", self.gate_input_transform
        )
        transformed = transform(dummy)
        if transformed.shape != dummy.shape:
            raise UnsupportedError("Hierarchical transforms must preserve rows and features.")
        targets = transformed[..., rows, columns]
        return targets.unsqueeze(-2) if self._output_batch else targets

    @property
    def ancestor_values(self) -> Any:  # type: ignore[override]
        return [self._targets(path, self._anchor) for path in self.paths]

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if last_dim_is_batch:
            raise UnsupportedError("Hierarchical kernels do not define feature batches.")
        result: Tensor | None = None
        for kernel, indices, path in zip(
            self.kernels, self.ancestor_indices, self.paths, strict=True
        ):
            targets = self._targets(path, x1)
            left = torch.isclose(x1[..., indices], targets.unsqueeze(-2), rtol=RTOL, atol=ATOL).all(
                -1
            )
            right = torch.isclose(
                x2[..., indices], targets.unsqueeze(-2), rtol=RTOL, atol=ATOL
            ).all(-1)
            gate = (left & right) if diag else (left.unsqueeze(-1) & right.unsqueeze(-2))
            value = kernel(x1, x2, diag=diag) * gate
            result = value if result is None else result + value
        if result is None:
            shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
            tail = (x1.shape[-2],) if diag else (x1.shape[-2], x2.shape[-2])
            return x1.new_zeros(*shape, *tail)
        return result


def _arguments(public: Any, instance: Any, args: Any, kwargs: Any) -> dict[str, Any]:
    bound = inspect.signature(public.__init__).bind(instance, *args, **kwargs)
    bound.apply_defaults()
    result = dict(bound.arguments)
    result.pop("self")
    result.pop("backend", None)
    return result


def _kernel(values: dict[str, Any], like: Tensor, dim: int) -> TorchHierarchicalConditionalKernel:
    return TorchHierarchicalConditionalKernel(
        dim,
        values["hierarchical_dependencies"],
        values["eval_hierarchical_features"],
        values["separate_hierarchical_features"],
        values["use_saas_prior"],
        values["use_outputscale"],
        _like=like,
    )


class TorchHierarchicalConditionalKernelGP(TorchSingleTaskGP, HierarchicalConditionalKernelGP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        values = _arguments(HierarchicalConditionalKernelGP, self, args, kwargs)
        X = values["train_X"]
        kernel = _kernel(values, X, X.shape[-1])
        TorchSingleTaskGP.__init__(
            self,
            X,
            values["train_Y"],
            values["train_Yvar"],
            covar_module=kernel,
            input_transform=values["input_transform"],
            outcome_transform=values["outcome_transform"],
        )
        kernel.configure_gate_transform(self.input_transform, X, list(range(X.shape[-1])))
        kernel._output_batch = self.num_outputs > 1


class TorchHierarchicalConditionalKernelMultiTaskGP(
    TorchMultiTaskGP, HierarchicalConditionalKernelMultiTaskGP
):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        values = _arguments(HierarchicalConditionalKernelMultiTaskGP, self, args, kwargs)
        X = values["train_X"]
        kernel = _kernel(values, X, X.shape[-1] - 1)
        for name in (
            "hierarchical_dependencies",
            "eval_hierarchical_features",
            "separate_hierarchical_features",
            "use_saas_prior",
            "use_outputscale",
        ):
            values.pop(name)
        TorchMultiTaskGP.__init__(self, **values, covar_module=kernel)
        kernel.configure_gate_transform(
            self.input_transform,
            X,
            [i for i in range(X.shape[-1]) if i != self._task_feature],
        )
