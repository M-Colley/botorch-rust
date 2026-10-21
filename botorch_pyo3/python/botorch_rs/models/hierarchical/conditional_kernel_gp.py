"""Exact GPs for feature forests with native PSD activation gates."""

from __future__ import annotations

import math
import operator
import sys
from collections.abc import Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.kernels import AdditiveKernel, Kernel, MaternKernel, ScaleKernel, _KernelList
from botorch_rs.models._backend_selection import requires_tensor_backend
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.hierarchical.utils import get_blocks_with_paths
from botorch_rs.models.map_saas import add_saas_prior
from botorch_rs.models.multitask import MultiTaskGP
from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch_rs.utils.constraints import LogTransformedInterval
from botorch_rs.utils.types import DEFAULT

RTOL, ATOL = 1e-5, 1e-8
LOG_OUTPUTSCALE_CONSTRAINT = LogTransformedInterval(1e-2, 1e4, initial_value=10)


def _torch_requested(backend: str, *values: Any) -> bool:
    if backend not in {"auto", "native", "torch"}:
        raise ValueError("backend must be 'auto', 'native', or 'torch'.")
    if backend != "auto":
        return backend == "torch"
    torch = sys.modules.get("torch")
    return any(
        getattr(value, "requires_grad", False)
        or getattr(getattr(value, "device", None), "type", "cpu") != "cpu"
        or (torch is not None and requires_tensor_backend(value))
        for value in values
    )


class _ActivatedKernel(Kernel):
    """A private native product retaining the actual public subkernel object."""

    def __init__(self, base: Kernel, path: Sequence[tuple[int, float]]) -> None:
        self._base, self._path = base, path
        super().__init__()

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.product([_rs.KernelNode.activation(self._path), self._base._node()])

    def _children(self) -> list[tuple[str, Kernel]]:
        return [("kernels.1", self._base)]


class HierarchicalConditionalKernel(AdditiveKernel):
    """Sum block kernels when both points activate the block's ancestor path."""

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        if cls is HierarchicalConditionalKernel and _torch_requested(backend):
            from botorch_rs.models._torch_backend_hierarchical import (
                TorchHierarchicalConditionalKernel,
            )

            return object.__new__(TorchHierarchicalConditionalKernel)
        return object.__new__(cls)

    def __init__(
        self,
        dim: int,
        hierarchical_dependencies: Any,
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        *,
        backend: str = "auto",
    ) -> None:
        self._components: list[Kernel] = []
        Kernel.__init__(self)
        self.dim = operator.index(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        self.hierarchical_dependencies = hierarchical_dependencies
        self.use_saas_prior, self.use_outputscale = use_saas_prior, use_outputscale
        for branches in hierarchical_dependencies.values():
            values = sorted(branches)
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Hierarchical parent values must be finite.")
            for a, b in zip(values, values[1:], strict=False):
                if b - a <= ATOL + RTOL * max(abs(a), abs(b)):
                    raise ValueError(
                        f"Float parent values {a} and {b} are closer than the default tolerance."
                    )
        self.partition, self.paths = get_blocks_with_paths(
            self.dim,
            hierarchical_dependencies,
            eval_hierarchical_features,
            separate_hierarchical_features,
        )
        self.ancestor_indices = [[i for i, _ in path] for path in self.paths]
        self.ancestor_values = [[value for _, value in path] for path in self.paths]
        self.kernels = _KernelList(self.construct_individual_kernels())
        self._components = [
            _ActivatedKernel(kernel, path)
            for kernel, path in zip(self.kernels, self.paths, strict=True)
        ]

    def construct_individual_kernels(self) -> list[Kernel]:
        kernels: list[Kernel] = []
        for indices in self.partition:
            base: Kernel
            if self.use_saas_prior:
                base = MaternKernel(nu=2.5, ard_num_dims=len(indices), active_dims=indices)
                add_saas_prior(base)
            else:
                base = get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=len(indices), use_rbf_kernel=False, active_dims=indices
                )
                base.lengthscale = 1.0
            kernels.append(
                ScaleKernel(
                    base, outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10)
                )
                if self.use_outputscale
                else base
            )
        return kernels

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.additive([component._node() for component in self._components])

    def _node(self) -> _rs.KernelNode:
        node = self._configure(self._base_node())
        return node if self.active_dims is None else node.with_parent_active_dims(self.active_dims)

    def _children(self) -> list[tuple[str, Kernel]]:
        return [(f"kernels.{i}", kernel) for i, kernel in enumerate(self._components)]

    def forward(self, x1: Any, x2: Any, diag: bool = False, **params: Any) -> Any:
        return self(x1, x2, diag=diag, **params)


def _transform_dependencies(
    train_X: Any, dependencies: Any, transform: Any, feature_columns: list[int] | None = None
) -> Any:
    if transform is None or not dependencies:
        return dependencies
    data, shape = to_flat(train_X)
    d = shape[-1]
    columns = list(range(d)) if feature_columns is None else feature_columns
    # Learn Normalize statistics from the actual training data before deriving
    # branch values; deriving them from an unfitted transform corrupts gates.
    training = transform.training
    try:
        transform.train(True)
        transform(train_X)
    finally:
        transform.train(training)
    pairs = [(parent, value) for parent, branches in dependencies.items() for value in branches]
    rows: list[float] = []
    for parent, value in pairs:
        row = list(data[:d])
        row[columns[parent]] = value
        rows.extend(row)
    dummy = from_flat(rows, (len(pairs), d), kind_of(train_X))
    values, out_shape = to_flat(transform.preprocess_transform(dummy))
    if tuple(out_shape) != (len(pairs), d):
        raise ValueError("Hierarchical transforms must preserve input dimensions and rows.")
    result: dict[int, dict[float, list[int]]] = {}
    for i, (parent, original) in enumerate(pairs):
        value = values[i * d + columns[parent]]
        result.setdefault(parent, {})[value] = list(dependencies[parent][original])
    return result


class HierarchicalConditionalKernelGP(SingleTaskGP):
    """Single-task exact GP with correlations over common active features."""

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        if cls is HierarchicalConditionalKernelGP and _torch_requested(
            backend, *args, *kwargs.values()
        ):
            from botorch_rs.models._torch_backend_hierarchical import (
                TorchHierarchicalConditionalKernelGP,
            )

            return object.__new__(TorchHierarchicalConditionalKernelGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        hierarchical_dependencies: Any,
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        train_Yvar: Any = None,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        input_transform: Any = None,
        outcome_transform: Any = DEFAULT,
        *,
        backend: str = "auto",
    ) -> None:
        kernel = HierarchicalConditionalKernel(
            to_flat(train_X)[1][-1],
            _transform_dependencies(train_X, hierarchical_dependencies, input_transform),
            eval_hierarchical_features,
            separate_hierarchical_features,
            use_saas_prior,
            use_outputscale,
        )
        super().__init__(
            train_X,
            train_Y,
            train_Yvar,
            covar_module=kernel,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

    @classmethod
    def construct_inputs(  # type: ignore[override]
        cls,
        training_data: Any,
        hierarchical_dependencies: Any,
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
    ) -> dict[str, Any]:
        return {
            **SingleTaskGP.construct_inputs(training_data),
            "hierarchical_dependencies": hierarchical_dependencies,
            "eval_hierarchical_features": eval_hierarchical_features,
            "separate_hierarchical_features": separate_hierarchical_features,
            "use_saas_prior": use_saas_prior,
            "use_outputscale": use_outputscale,
        }


class HierarchicalConditionalKernelMultiTaskGP(MultiTaskGP):
    """Multi-task GP combining hierarchical input blocks and a task covariance."""

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        if cls is HierarchicalConditionalKernelMultiTaskGP and _torch_requested(
            backend, *args, *kwargs.values()
        ):
            from botorch_rs.models._torch_backend_hierarchical import (
                TorchHierarchicalConditionalKernelMultiTaskGP,
            )

            return object.__new__(TorchHierarchicalConditionalKernelMultiTaskGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        hierarchical_dependencies: Any,
        train_Yvar: Any = None,
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        likelihood: Any = None,
        task_covar_prior: Any = None,
        output_tasks: Any = None,
        rank: Any = None,
        all_tasks: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        validate_task_values: bool = True,
        *,
        backend: str = "auto",
    ) -> None:
        d = to_flat(train_X)[1][-1]
        feature = operator.index(task_feature)
        if not -d <= feature < d:
            raise ValueError("task_feature is out of range.")
        columns = [i for i in range(d) if i != feature % d]
        deps = _transform_dependencies(train_X, hierarchical_dependencies, input_transform, columns)
        kernel = HierarchicalConditionalKernel(
            len(columns),
            deps,
            eval_hierarchical_features,
            separate_hierarchical_features,
            use_saas_prior,
            use_outputscale,
        )
        super().__init__(
            train_X,
            train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            covar_module=kernel,
            likelihood=likelihood,
            task_covar_prior=task_covar_prior,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            validate_task_values=validate_task_values,
        )

    @classmethod
    def construct_inputs(  # type: ignore[override]
        cls,
        training_data: Any,
        task_feature: int,
        hierarchical_dependencies: Any,
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        output_tasks: Any = None,
        task_covar_prior: Any = None,
        rank: Any = None,
    ) -> dict[str, Any]:
        result = MultiTaskGP.construct_inputs(
            training_data,
            task_feature=task_feature,
            output_tasks=output_tasks,
            task_covar_prior=task_covar_prior,
            rank=rank,
        )
        result.update(
            hierarchical_dependencies=hierarchical_dependencies,
            eval_hierarchical_features=eval_hierarchical_features,
            separate_hierarchical_features=separate_hierarchical_features,
            use_saas_prior=use_saas_prior,
            use_outputscale=use_outputscale,
        )
        return result
