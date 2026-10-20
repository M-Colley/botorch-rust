"""Feature partitions and exact binary activation kernels for heterogeneous tasks."""

from __future__ import annotations

from operator import index as integer_index
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.kernels import Kernel, MaternKernel, ScaleKernel, _KernelList
from botorch_rs.models.hierarchical.conditional_kernel_gp import _torch_requested
from botorch_rs.models.map_saas import add_saas_prior
from botorch_rs.utils.constraints import LogTransformedInterval

LOG_OUTPUTSCALE_CONSTRAINT = LogTransformedInterval(1e-2, 1e4, initial_value=10)


def _features(feature_indices: list[list[int]]) -> list[set[int]]:
    if not feature_indices:
        raise ValueError("At least one task feature list is required.")
    output = []
    for features in feature_indices:
        indices = [integer_index(value) for value in features]
        if any(value < 0 for value in indices):
            raise ValueError("Feature indices must be nonnegative integers.")
        if len(set(indices)) != len(indices):
            raise ValueError("Each task feature list must contain unique indices.")
        output.append(set(indices))
    return output


def find_subsets(feature_indices: list[list[int]]) -> list[set[int]]:
    """Return maximal disjoint feature groups with the same active tasks."""
    features = _features(feature_indices)
    subsets = [features[0]]
    for task in feature_indices[1:]:
        remaining_task = set(task)
        refined = []
        for subset in subsets:
            common = remaining_task & subset
            remaining = subset - common
            if common:
                refined.append(common)
                remaining_task = remaining_task.difference(common)
            if remaining:
                refined.append(remaining)
        if remaining_task:
            refined.append(remaining_task)
        subsets = refined
    return subsets


def map_subsets(
    subsets: list[set[int]], feature_indices: list[list[int]]
) -> tuple[dict[tuple[int, ...], list[int]], list[list[int]]]:
    """Map a disjoint feature partition to task memberships and binary rows."""
    features = _features(feature_indices)
    keys = [tuple(integer_index(value) for value in subset) for subset in subsets]
    normalized = [set(key) for key in keys]
    all_features = set().union(*features)
    seen: set[int] = set()
    for subset in normalized:
        if any(value < 0 for value in subset):
            raise ValueError("Feature indices must be nonnegative integers.")
        if seen & subset:
            raise ValueError("Feature subsets must be disjoint; found duplicate indices.")
        if not subset <= all_features:
            raise ValueError("A subset contains a feature absent from every task.")
        if not any(subset <= task for task in features):
            raise ValueError("Each feature subset must be fully active in at least one task.")
        seen |= subset
    if seen != all_features:
        raise ValueError("Feature subsets must cover every task feature.")
    if len(normalized) != len({tuple(subset) for subset in normalized}):
        raise ValueError("Feature subsets must be unique.")
    mapping = {
        key: [i for i, task in enumerate(features) if subset <= task]
        for key, subset in zip(keys, normalized, strict=True)
    }
    binary = [[int(subset <= task) for subset in normalized] for task in features]
    return mapping, binary


class DeltaKernel(Kernel):
    """Exact zero/one covariance ``1[x1 == 1] * 1[x2 == 1]`` for one feature."""

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        if cls is DeltaKernel and _torch_requested(backend):
            from botorch_rs.models._torch_backend_heterogeneous import TorchDeltaKernel

            return object.__new__(TorchDeltaKernel)
        return object.__new__(cls)

    def __init__(self, *args: Any, backend: str = "auto", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _base_node(self) -> _rs.KernelNode:
        node_type: Any = _rs.KernelNode
        return node_type.activation([(0, 1.0)])

    def _standard_spec(self, d: int) -> _rs.KernelSpec | None:
        assert (len(self.active_dims) if self.active_dims is not None else d) == 1, (
            "DeltaKernel expects 1D inputs!"
        )
        return None

    def __call__(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Any:
        for value in (x1, x2):
            if value is not None:
                shape = to_flat(value)[1]
                dimensions = (
                    len(self.active_dims)
                    if self.active_dims is not None
                    else shape[-1]
                    if len(shape) > 1
                    else 1
                )
                assert dimensions == 1, "DeltaKernel expects 1D inputs!"
        return super().__call__(x1, x2, diag, last_dim_is_batch, **params)

    forward = __call__


class CombinatorialCovarModule(ScaleKernel):
    """Native scaled Matérn-5/2 covariance on binary feature memberships."""

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        if cls is CombinatorialCovarModule and _torch_requested(backend):
            from botorch_rs.models._torch_backend_heterogeneous import TorchCombinatorialCovarModule

            return object.__new__(TorchCombinatorialCovarModule)
        return object.__new__(cls)

    def __init__(self, ard_num_dims: int | None = None, *, backend: str = "auto") -> None:
        base = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        if ard_num_dims is not None and ard_num_dims > 1:
            add_saas_prior(base)
        super().__init__(base, outputscale_constraint=LOG_OUTPUTSCALE_CONSTRAINT)


class MultiTaskConditionalKernel(Kernel):
    """Native sum of spatial covariances gated by shared task features.

    The final optional covariance evaluates a Matérn kernel on the tasks'
    binary subset memberships. Task indicators are discrete integer indices.
    """

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        if cls is MultiTaskConditionalKernel and _torch_requested(backend):
            from botorch_rs.models._torch_backend_heterogeneous import (
                TorchMultiTaskConditionalKernel,
            )

            return object.__new__(TorchMultiTaskConditionalKernel)
        return object.__new__(cls)

    def __init__(
        self,
        feature_indices: list[list[int]],
        task_feature_index: int = -1,
        use_saas_prior: bool = True,
        use_combinatorial_kernel: bool = True,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        features = _features(feature_indices)
        if any(not features_for_task for features_for_task in features):
            raise ValueError("Each heterogeneous task must contain at least one feature.")
        self.feature_indices = [list(indices) for indices in feature_indices]
        self.task_feature_index = integer_index(task_feature_index)
        self.use_saas_prior = bool(use_saas_prior)
        self.use_combinatorial_kernel = bool(use_combinatorial_kernel)
        self.active_index_map, self.binary_map = map_subsets(
            find_subsets(feature_indices), feature_indices
        )
        self.kernels = _KernelList(self.construct_individual_kernels())
        # This remains registered, with its priors, when disabled, matching
        # BoTorch's named parameters and marginal-likelihood prior terms.
        self.combinatorial_kernel = CombinatorialCovarModule(ard_num_dims=len(self.kernels))
        self.delta_kernel = DeltaKernel()

    def construct_individual_kernels(self) -> list[Kernel]:
        kernels: list[Kernel] = []
        for indices in self.active_index_map:
            base = MaternKernel(nu=2.5, ard_num_dims=len(indices), active_dims=indices)
            if self.use_saas_prior:
                add_saas_prior(base)
            kernels.append(ScaleKernel(base, outputscale_constraint=LOG_OUTPUTSCALE_CONSTRAINT))
        return kernels

    def map_task_to_binary(self, x_task: Any) -> Any:
        data, shape = to_flat(x_task, "x_task")
        mapped: list[float] = []
        for value in data:
            if not 0 <= value < len(self.binary_map) or value != int(value):
                raise ValueError(f"Task index {value} is invalid for {len(self.binary_map)} tasks.")
            mapped.extend(self.binary_map[int(value)])
        return from_flat(mapped, (*shape, len(self.kernels)), kind_of(x_task))

    def _children(self) -> list[tuple[str, Kernel]]:
        return [
            *((f"kernels.{i}", kernel) for i, kernel in enumerate(self.kernels)),
            ("combinatorial_kernel", self.combinatorial_kernel),
        ]

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.heterogeneous(
            self.task_feature_index,
            [[bool(value) for value in row] for row in self.binary_map],
            [kernel._node() for kernel in self.kernels],
            self.combinatorial_kernel._node(),
            self.use_combinatorial_kernel,
        )

    def forward(self, x1: Any, x2: Any, diag: bool = False, **params: Any) -> Any:
        return self(x1, x2, diag=diag, **params)


__all__ = [
    "DeltaKernel",
    "CombinatorialCovarModule",
    "MultiTaskConditionalKernel",
    "find_subsets",
    "map_subsets",
]
