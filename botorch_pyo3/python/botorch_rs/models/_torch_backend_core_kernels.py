"""Explicit device factories for the public elementary and task kernel trees."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from botorch_rs import kernels as public
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend_kernel_catalogue import (
    TorchPositiveIndexKernel,
    _BatchShape,
    _TorchLKJCovariancePrior,
)
from botorch_rs.models._torch_backend_modules import (
    TorchAlgebraKernel,
    TorchCompositeKernel,
    TorchConstraint,
    TorchKernel,
    TorchScaleKernel,
    TorchStationaryKernel,
    _tensor,
    build_kernel,
)
from botorch_rs.priors import LKJCovariancePrior


def _metadata(kwargs: dict[str, Any], children: Any = ()) -> Tensor:
    kwargs.pop("backend", None)
    dtype, device = kwargs.pop("dtype", None), kwargs.pop("device", None)
    like = None
    for child in children:
        if isinstance(child, nn.Module):
            like = next(iter(child.parameters()), getattr(child, "_anchor", None))
            if like is not None:
                break
    return torch.empty(
        (),
        dtype=dtype if dtype is not None else (None if like is None else like.dtype),
        device=device if device is not None else (None if like is None else like.device),
    )


class _CoreFactory(_BatchShape):
    _public_class: Any
    _implementation: Any
    _tensor_backend = "torch"

    def __init__(
        self, *args: Any, _source: Any = None, _like: Tensor | None = None, **kwargs: Any
    ) -> None:
        like = _metadata(kwargs) if _like is None else _like
        if _source is None:
            _source = self._public_class(*args, backend="native", **kwargs)
        self._implementation.__init__(self, _source, like)


class TorchCoreIndexKernel(_BatchShape, TorchKernel, public.IndexKernel):  # type: ignore[misc]
    """Unconstrained low-rank factors and positive diagonal task variances."""

    _tensor_backend = "torch"
    raw_var_constraint: Any
    IndexKernelPrior: Any

    def __init__(
        self, *args: Any, _source: Any = None, _like: Tensor | None = None, **kwargs: Any
    ) -> None:
        like = _metadata(kwargs) if _like is None else _like
        source = (
            public.IndexKernel(*args, backend="native", **kwargs) if _source is None else _source
        )
        TorchKernel.__init__(self, like, source.active_dims)
        self.batch_shape = source.batch_shape
        self.num_tasks, self.rank = source.num_tasks, source.rank
        self.raw_var_constraint = TorchConstraint(source.raw_var_constraint, like)
        for name in ("covar_factor", "raw_var"):
            if self.rank == 0:
                # Native KernelNode requires positive rank, but the public
                # GPyTorch diagonal-only rank-zero parameterization is valid.
                # Its construction values need no CPU covariance evaluation.
                shape = (
                    (*self.batch_shape, self.num_tasks, 0)
                    if name == "covar_factor"
                    else (*self.batch_shape, self.num_tasks)
                )
                original = source._local.get(name)
                if original is None:
                    original = source.raw_var_constraint.inverse_transform(
                        source.raw_var_constraint.initial_value
                    )
                original = (
                    _tensor(original, like).reshape(shape)
                    if name in source._local
                    else _tensor(original, like).expand(shape)
                )
            else:
                original = getattr(source, name)
            self.register_parameter(
                name,
                nn.Parameter(
                    _tensor(original, like).clone(),
                    requires_grad=not isinstance(original, nn.Parameter) or original.requires_grad,
                ),
            )
        if isinstance(source.IndexKernelPrior, LKJCovariancePrior):
            self.IndexKernelPrior = _TorchLKJCovariancePrior(source.IndexKernelPrior, like)
            self._prior_parameters["IndexKernelPrior"] = "covar_matrix", 0
        else:
            self._register_prior("IndexKernelPrior", source.IndexKernelPrior, "covar_matrix", 2)

    @property
    def var(self) -> Tensor:
        return self.raw_var_constraint.transform(self.raw_var)

    @var.setter
    def var(self, value: Any) -> None:
        self._assign("raw_var", value, self.raw_var_constraint)

    @property
    def covar_matrix(self) -> Tensor:
        return self.covar_factor @ self.covar_factor.transpose(-1, -2) + self.var.diag_embed()

    @covar_matrix.setter
    def covar_matrix(self, value: Any) -> None:
        covariance = _tensor(value, self._anchor).expand(
            *self.batch_shape, self.num_tasks, self.num_tasks
        )
        values, vectors = torch.linalg.eigh(covariance)
        if bool((values <= 0).any()):
            raise ValueError("Task covariance must be positive definite.")
        if self.rank == self.num_tasks:
            # An unconstrained square factor and positive residual diagonal can
            # represent the sampled covariance exactly. Lower rank is a genuine
            # projected initialization, as documented for PositiveIndexKernel.
            variance = values[..., :1].mul(0.5).expand(*self.batch_shape, self.num_tasks)
            factor = torch.linalg.cholesky(covariance - variance.diag_embed())
        elif self.rank == 0:
            factor = covariance.new_empty(*self.batch_shape, self.num_tasks, 0)
            variance = covariance.diagonal(dim1=-2, dim2=-1)
        else:
            factor = vectors[..., :, -self.rank :] * values[..., None, -self.rank :].sqrt()
            variance = (covariance.diagonal(dim1=-2, dim2=-1) - factor.square().sum(-1)).clamp_min(
                1e-8
            )
        self._assign("covar_factor", factor)
        self.var = variance

    def _eval_covar_matrix(self) -> Tensor:
        return self.covar_matrix

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        # Both covariance parameterizations use the same checked discrete gather.
        return TorchPositiveIndexKernel._forward_selected(
            cast(Any, self), x1, x2, diag, last_dim_is_batch
        )


class TorchCoreScaleKernel(_BatchShape, TorchScaleKernel, public.ScaleKernel):  # type: ignore[misc]
    _tensor_backend = "torch"

    def __init__(
        self,
        base_kernel: Any = None,
        *args: Any,
        _source: Any = None,
        _like: Tensor | None = None,
        _convert: Any = None,
        **kwargs: Any,
    ) -> None:
        if _source is not None:
            base_kernel = _source.base_kernel
        like = _metadata(kwargs, (base_kernel,)) if _like is None else _like
        child: Any = (
            build_kernel(base_kernel, 0, base_kernel.batch_shape, like)
            if _convert is None
            else _convert(base_kernel)
        )
        if _source is None:
            # This descriptor initializes only the new scale. The actual child
            # stays a shared Tensor module with its existing parameter leaves.
            dummy = public.RBFKernel(batch_shape=child.batch_shape, active_dims=child.active_dims)
            _source = public.ScaleKernel(dummy, *args, backend="native", **kwargs)
        TorchScaleKernel.__init__(self, _source, child, like)


class _CompositeFactory(_BatchShape, TorchCompositeKernel):
    _public_class: Any
    _tensor_backend = "torch"

    def __init__(
        self,
        *kernels: Any,
        _source: Any = None,
        _like: Tensor | None = None,
        _convert: Any = None,
        **kwargs: Any,
    ) -> None:
        if _source is not None:
            kernels = tuple(_source.kernels)
        like = _metadata(kwargs, kernels) if _like is None else _like
        active_dims = kwargs.pop("active_dims", None if _source is None else _source.active_dims)
        batch = kwargs.pop("batch_shape", () if _source is None else _source._batch_shape)
        ard = kwargs.pop("ard_num_dims", None)
        if kwargs:
            raise UnsupportedError(f"Unsupported composite kernel options: {sorted(kwargs)}")
        memo: dict[int, Any] = {}

        def convert(child: Any) -> Any:
            if id(child) not in memo:
                memo[id(child)] = build_kernel(child, 0, child.batch_shape, like)
            return memo[id(child)]

        children = [(convert if _convert is None else _convert)(child) for child in kernels]
        kind = "add" if self._public_class is public.AdditiveKernel else "product"
        TorchCompositeKernel.__init__(self, children, kind, like)
        self.batch_shape = torch.broadcast_shapes(batch, self.batch_shape)
        if active_dims is not None:
            self.active_dims = torch.as_tensor(active_dims, dtype=torch.long, device=like.device)
        self.ard_num_dims = ard

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if self.ard_num_dims is not None and x1.shape[-1] != self.ard_num_dims:
            raise ValueError(
                f"Expected {self.ard_num_dims} selected input dimensions, got {x1.shape[-1]}."
            )
        value = super()._forward_selected(x1, x2, diag, last_dim_is_batch)
        tail = (1 if diag else 2) + int(last_dim_is_batch)
        batch = torch.broadcast_shapes(self.batch_shape, x1.shape[:-2], x2.shape[:-2])
        return value.expand(*batch, *value.shape[-tail:])


class TorchCoreMultitaskKernel(_BatchShape, TorchKernel, public.MultitaskKernel):  # type: ignore[misc]
    _tensor_backend = "torch"
    data_covar_module: Any
    task_covar_module: Any

    def __init__(
        self,
        data_covar_module: Any = None,
        num_tasks: int | None = None,
        rank: int | None = 1,
        task_covar_prior: Any = None,
        _source: Any = None,
        _like: Tensor | None = None,
        _convert: Any = None,
        **kwargs: Any,
    ) -> None:
        if _source is not None:
            data_covar_module, num_tasks = _source.data_covar_module, _source.num_tasks
        like = _metadata(kwargs, (data_covar_module,)) if _like is None else _like
        active_dims = kwargs.pop("active_dims", None if _source is None else _source.active_dims)
        batch = kwargs.pop("batch_shape", () if _source is None else _source._batch_shape)
        if kwargs:
            raise UnsupportedError(f"Unsupported multitask kernel options: {sorted(kwargs)}")
        TorchKernel.__init__(self, like, active_dims)
        if num_tasks is None:
            raise TypeError("num_tasks is required")
        self.num_tasks = num_tasks
        convert = (
            (lambda child: build_kernel(child, 0, child.batch_shape, like))
            if _convert is None
            else _convert
        )
        data = convert(data_covar_module)
        self.task_covar_module = (
            TorchCoreIndexKernel(
                num_tasks, rank, prior=task_covar_prior, batch_shape=batch, _like=like
            )
            if _source is None
            else convert(_source.task_covar_module)
        )
        self.data_covar_module = data
        self.batch_shape = torch.broadcast_shapes(
            batch, self.data_covar_module.batch_shape, self.task_covar_module.batch_shape
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if last_dim_is_batch:
            raise UnsupportedError("MultitaskKernel does not accept last_dim_is_batch=True.")
        spatial = self.data_covar_module(x1, x2, diag=diag)
        task = self.task_covar_module.covar_matrix
        if diag:
            value = spatial.unsqueeze(-1) * task.diagonal(dim1=-2, dim2=-1).unsqueeze(-2)
            return value.flatten(-2)
        value = spatial.unsqueeze(-1).unsqueeze(-3) * task.unsqueeze(-2).unsqueeze(-4)
        return value.reshape(
            *value.shape[:-4],
            spatial.shape[-2] * self.num_tasks,
            spatial.shape[-1] * self.num_tasks,
        )


_FACTORIES: dict[type, type] = {}


def tensor_kernel_class(cls: type) -> type:
    if cls not in _FACTORIES:
        result: type
        if cls is public.IndexKernel:
            result = TorchCoreIndexKernel
        elif cls is public.ScaleKernel:
            result = TorchCoreScaleKernel
        elif cls is public.MultitaskKernel:
            result = TorchCoreMultitaskKernel
        elif cls in (public.AdditiveKernel, public.ProductKernel):
            result = type(
                f"Torch{cls.__name__}",
                (_CompositeFactory, cls),
                {
                    "__module__": __name__,
                    "_public_class": cls,
                },
            )
        else:
            implementation = (
                TorchStationaryKernel
                if cls in (public.RBFKernel, public.MaternKernel)
                else TorchAlgebraKernel
            )
            result = type(
                f"Torch{cls.__name__}",
                (_CoreFactory, implementation, cls),
                {
                    "__module__": __name__,
                    "_public_class": cls,
                    "_implementation": implementation,
                },
            )
        _FACTORIES[cls] = result
        globals()[result.__name__] = result
    return _FACTORIES[cls]


def __getattr__(name: str) -> Any:
    if name.startswith("Torch"):
        cls = getattr(public, name.removeprefix("Torch"), None)
        if cls is not None:
            return tensor_kernel_class(cls)
    raise AttributeError(name)
