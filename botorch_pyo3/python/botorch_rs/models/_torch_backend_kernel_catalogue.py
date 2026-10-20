"""Public Tensor factories for categorical, fidelity and positive task kernels."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.models._torch_backend_modules import (
    TorchConstraint,
    TorchKernel,
    TorchPrior,
    _tensor,
    _TorchModule,
    build_kernel,
    module_log_prior,
)
from botorch_rs.models._torch_backend_specialized import TorchCategoricalKernel as _Categorical
from botorch_rs.models._torch_backend_specialized import TorchFidelityKernel as _Fidelity
from botorch_rs.models.kernels.categorical import CategoricalKernel
from botorch_rs.models.kernels.downsampling import DownsamplingKernel
from botorch_rs.models.kernels.exponential_decay import ExponentialDecayKernel
from botorch_rs.models.kernels.linear_truncated_fidelity import LinearTruncatedFidelityKernel
from botorch_rs.models.kernels.positive_index import PositiveIndexKernel
from botorch_rs.priors import LKJCovariancePrior


class _BatchShape:
    @property
    def batch_shape(self) -> Any:
        return self._tensor_batch_shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._tensor_batch_shape = torch.Size(value)


def _initial(
    source_class: Any, args: Any, kwargs: Any, source: Any, like: Tensor | None
) -> tuple[Any, Tensor]:
    kwargs.pop("backend", None)
    dtype, device = kwargs.pop("dtype", None), kwargs.pop("device", None)
    if source is None:
        source = source_class(*args, backend="native", **kwargs)
    return source, torch.empty((), dtype=dtype, device=device) if like is None else like


class TorchCategoricalKernel(_BatchShape, _Categorical, CategoricalKernel):  # type: ignore[misc]
    def __init__(
        self, *args: Any, _source: Any = None, _like: Tensor | None = None, **kwargs: Any
    ) -> None:
        source, like = _initial(CategoricalKernel, args, kwargs, _source, _like)
        _Categorical.__init__(self, source, like)


def _fidelity_initial(
    self: Any,
    source_class: Any,
    args: Any,
    kwargs: Any,
    source: Any,
    like: Tensor | None,
    convert: Any,
) -> None:
    source, like = _initial(source_class, args, kwargs, source, like)
    if convert is None:
        memo: dict[int, Any] = {}

        def convert(child: Any) -> Any:
            if id(child) not in memo:
                memo[id(child)] = build_kernel(child, 0, child.batch_shape, like)
            return memo[id(child)]

    _Fidelity.__init__(self, source, like, convert)


class TorchDownsamplingKernel(_BatchShape, _Fidelity, DownsamplingKernel):  # type: ignore[misc]
    def __init__(
        self,
        *args: Any,
        _source: Any = None,
        _like: Tensor | None = None,
        _convert: Any = None,
        **kwargs: Any,
    ) -> None:
        _fidelity_initial(self, DownsamplingKernel, args, kwargs, _source, _like, _convert)


class TorchExponentialDecayKernel(_BatchShape, _Fidelity, ExponentialDecayKernel):  # type: ignore[misc]
    def __init__(
        self,
        *args: Any,
        _source: Any = None,
        _like: Tensor | None = None,
        _convert: Any = None,
        **kwargs: Any,
    ) -> None:
        _fidelity_initial(self, ExponentialDecayKernel, args, kwargs, _source, _like, _convert)


class TorchLinearTruncatedFidelityKernel(_BatchShape, _Fidelity, LinearTruncatedFidelityKernel):  # type: ignore[misc]
    def __init__(
        self,
        *args: Any,
        _source: Any = None,
        _like: Tensor | None = None,
        _convert: Any = None,
        **kwargs: Any,
    ) -> None:
        external: dict[str, nn.Module] = {}
        if _source is None:
            metadata = {key: kwargs.pop(key) for key in ("dtype", "device") if key in kwargs}
            values = inspect.signature(LinearTruncatedFidelityKernel.__init__).bind(
                None, *args, **kwargs
            )
            values.apply_defaults()
            config = dict(values.arguments)
            config.pop("self")
            extra = config.pop("kwargs")
            for name in ("covar_module_unbiased", "covar_module_biased"):
                if isinstance(config[name], nn.Module):
                    external[name] = config[name]
                    config[name] = None
            config["backend"] = "native"
            _source = LinearTruncatedFidelityKernel(**config, **extra)
            kwargs = metadata
        _fidelity_initial(self, LinearTruncatedFidelityKernel, (), kwargs, _source, _like, _convert)
        self._external_kernel_names = tuple(external)
        for name, module in external.items():
            setattr(self, name, build_kernel(module, 0, self.batch_shape, self._anchor))

    def log_prior(self) -> Tensor:
        value = super().log_prior()
        seen: set[int] = set()
        for name in self._external_kernel_names:
            module = getattr(self, name)
            if id(module) not in seen and not isinstance(module, _TorchModule):
                value = value + module_log_prior(module, self.batch_shape)
                seen.add(id(module))
        return value


class _TorchLKJCovariancePrior(nn.Module):
    """One correlation log density plus independent marginal-scale densities."""

    concentration: Tensor

    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__()
        self.n = source.n
        self.register_buffer("concentration", like.new_tensor(source.eta))
        self.sd_prior = TorchPrior(source.sd_prior, like)

    def _extended_shape(self) -> torch.Size:
        return torch.Size([self.n, self.n])

    def log_prob(self, covariance: Tensor) -> Tensor:
        scales = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
        correlation = covariance / (scales.unsqueeze(-1) * scales.unsqueeze(-2))
        if self.n == 1:
            return self.sd_prior.log_prob(scales).sum(-1)
        factor = torch.linalg.cholesky(correlation)
        return torch.distributions.LKJCholesky(self.n, self.concentration).log_prob(
            factor
        ) + self.sd_prior.log_prob(scales).sum(-1)

    def sample(self, shape: Any = ()) -> Tensor:
        shape = torch.Size(shape)
        with torch.no_grad():
            factor = (
                self.concentration.new_ones(*shape, 1, 1)
                if self.n == 1
                else torch.distributions.LKJCholesky(self.n, self.concentration).sample(shape)
            )
            scales = self.sd_prior.sample(shape + (self.n,))
            return (factor @ factor.transpose(-1, -2)) * scales.unsqueeze(-1) * scales.unsqueeze(-2)


def _index_initial_parameter(source: Any, name: str, like: Tensor) -> Any:
    if source.rank != 0 or source._model is not None:
        return getattr(source, name)
    shape = (
        (*source.batch_shape, source.num_tasks, 0)
        if name in ("covar_factor", "raw_covar_factor")
        else (*source.batch_shape, source.num_tasks)
    )
    original = source._local.get(name)
    if original is not None:
        return _tensor(original, like).reshape(shape)
    constraint = source.raw_var_constraint
    return _tensor(constraint.inverse_transform(constraint.initial_value), like).expand(shape)


class TorchPositiveIndexKernel(_BatchShape, TorchKernel, PositiveIndexKernel):  # type: ignore[misc]
    """Positive low-rank task covariance with arbitrary kernel/input batches."""

    _tensor_backend = "torch"
    raw_var_constraint: Any
    raw_covar_factor_constraint: Any

    def __init__(
        self, *args: Any, _source: Any = None, _like: Tensor | None = None, **kwargs: Any
    ) -> None:
        source, like = _initial(PositiveIndexKernel, args, kwargs, _source, _like)
        TorchKernel.__init__(self, like, source.active_dims)
        self.batch_shape = source.batch_shape
        self.num_tasks, self.rank = source.num_tasks, source.rank
        self.unit_scale_for_target, self.target_task_index = (
            source.unit_scale_for_target,
            source.target_task_index,
        )
        self.normalize_covar_matrix = source.normalize_covar_matrix
        self.raw_var_constraint = TorchConstraint(source.raw_var_constraint, like)
        self.raw_covar_factor_constraint = TorchConstraint(source.raw_covar_factor_constraint, like)
        for name in ("raw_var", "raw_covar_factor"):
            original = _index_initial_parameter(source, name, like)
            self.register_parameter(
                name,
                nn.Parameter(
                    _tensor(original, like).clone(),
                    requires_grad=not isinstance(original, nn.Parameter) or original.requires_grad,
                ),
            )
        if isinstance(source.IndexKernelPrior, LKJCovariancePrior):
            self.IndexKernelPrior = _TorchLKJCovariancePrior(source.IndexKernelPrior, like)  # type: ignore[assignment]
            self._prior_parameters["IndexKernelPrior"] = ("covar_matrix", 0)
        else:
            self._register_prior(
                "IndexKernelPrior", source.IndexKernelPrior, "_lower_triangle_corr", 1
            )

    @property
    def var(self) -> Tensor:
        return self.raw_var_constraint.transform(self.raw_var)

    @var.setter
    def var(self, value: Any) -> None:
        self._assign("raw_var", value, self.raw_var_constraint)

    @property
    def covar_factor(self) -> Tensor:
        return self.raw_covar_factor_constraint.transform(self.raw_covar_factor)

    @covar_factor.setter
    def covar_factor(self, value: Any) -> None:
        self._assign("raw_covar_factor", value, self.raw_covar_factor_constraint)

    def _set_covar_factor(self, value: Any) -> None:
        self.covar_factor = value

    @property
    def covar_matrix(self) -> Tensor:
        factor = self.covar_factor
        value = factor @ factor.transpose(-1, -2) + self.var.diag_embed()
        return (
            value / value[..., self.target_task_index, self.target_task_index, None, None]
            if self.unit_scale_for_target
            else value
        )

    @covar_matrix.setter
    def covar_matrix(self, value: Any) -> None:
        # A matrix prior cannot generally be inverted into a fixed-rank,
        # elementwise-positive factor. As with the source correlation setter,
        # project the draw into the actual parameterization for initialization.
        covariance = _tensor(value, self._anchor).expand(
            *self.batch_shape, self.num_tasks, self.num_tasks
        )
        factor = torch.linalg.cholesky(covariance)[..., :, : self.rank].clamp_min(1e-6)
        variance = (covariance.diagonal(dim1=-2, dim2=-1) - factor.square().sum(-1)).clamp_min(1e-6)
        self.covar_factor = factor
        self.var = variance

    def _eval_covar_matrix(self) -> Tensor:
        return self.covar_matrix

    @property
    def _lower_triangle_corr(self) -> Tensor:
        covariance = self.covar_matrix
        std = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
        correlation = covariance / (std.unsqueeze(-1) * std.unsqueeze(-2))
        pairs = torch.tril_indices(self.num_tasks, self.num_tasks, -1, device=std.device)
        return correlation[..., pairs[0], pairs[1]]

    @_lower_triangle_corr.setter
    def _lower_triangle_corr(self, value: Any) -> None:
        self._set_lower_triangle_corr(value)

    def _set_lower_triangle_corr(self, value: Any) -> None:
        pairs = torch.tril_indices(self.num_tasks, self.num_tasks, -1, device=self._anchor.device)
        correlations = (
            _tensor(value, self._anchor).expand(*self.batch_shape, pairs.shape[-1]).clamp(0, 1)
        )
        matrix = (
            torch.eye(self.num_tasks, dtype=self._anchor.dtype, device=self._anchor.device)
            .expand(*self.batch_shape, self.num_tasks, self.num_tasks)
            .clone()
        )
        matrix[..., pairs[0], pairs[1]] = correlations
        matrix[..., pairs[1], pairs[0]] = correlations
        values, vectors = torch.linalg.eigh(matrix)
        matrix = (vectors * values.clamp_min(1e-6).unsqueeze(-2)) @ vectors.transpose(-1, -2)
        std = matrix.diagonal(dim1=-2, dim2=-1).sqrt()
        matrix = matrix / (std.unsqueeze(-1) * std.unsqueeze(-2))
        self.covar_factor = torch.linalg.cholesky(matrix)[..., :, : self.rank].clamp_min(1e-6)

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
        if not last_dim_is_batch and (x1.shape[-1] != 1 or x2.shape[-1] != 1):
            raise ValueError("Task index kernels expect one selected task column.")
        if diag and x1.shape[-2] != x2.shape[-2]:
            raise ValueError("diag=True requires equal numbers of input points.")
        batch = torch.broadcast_shapes(self.batch_shape, x1.shape[:-2], x2.shape[:-2])
        covariance = self.covar_matrix.expand(*batch, self.num_tasks, self.num_tasks)
        left, right = (
            (x1.transpose(-1, -2), x2.transpose(-1, -2))
            if last_dim_is_batch
            else (x1[..., 0], x2[..., 0])
        )
        if last_dim_is_batch:
            features = torch.broadcast_shapes((left.shape[-2],), (right.shape[-2],))[0]
            batch = batch + (features,)
            covariance = covariance.unsqueeze(-3).expand(*batch, self.num_tasks, self.num_tasks)
        left, right = (
            left.long().expand(*batch, x1.shape[-2]),
            right.long().expand(*batch, x2.shape[-2]),
        )
        if not bool(
            ((left >= 0) & (left < self.num_tasks)).all()
            & ((right >= 0) & (right < self.num_tasks)).all()
        ):
            raise IndexError("Task index is outside the covariance matrix.")
        if diag:
            return covariance.flatten(-2).gather(-1, left * self.num_tasks + right)
        rows = covariance.gather(
            -2, left.unsqueeze(-1).expand(*batch, x1.shape[-2], self.num_tasks)
        )
        return rows.gather(-1, right.unsqueeze(-2).expand(*batch, x1.shape[-2], x2.shape[-2]))
