"""Mixed-input and fidelity exact GPs with device-resident covariance graphs."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from botorch_rs.constraints import GreaterThan
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import AdditiveKernel, ProductKernel, RBFKernel, ScaleKernel
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import (
    TorchAlgebraKernel,
    TorchCompositeKernel,
    TorchConstraint,
    TorchKernel,
    TorchScaleKernel,
    _tensor,
    build_kernel,
    module_log_prior,
)
from botorch_rs.models.gp_regression_fidelity import (
    SingleTaskMultiFidelityGP,
    _setup_multifidelity_covar_module,
)
from botorch_rs.models.gp_regression_mixed import MixedSingleTaskGP
from botorch_rs.models.kernels.categorical import CategoricalKernel
from botorch_rs.models.kernels.downsampling import DownsamplingKernel
from botorch_rs.models.kernels.exponential_decay import ExponentialDecayKernel
from botorch_rs.models.kernels.linear_truncated_fidelity import LinearTruncatedFidelityKernel
from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch_rs.utils.transforms import normalize_indices
from botorch_rs.utils.types import DEFAULT


class TorchCategoricalKernel(TorchAlgebraKernel):
    """Hamming covariance; categorical coordinates deliberately have zero slope."""

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if diag:
            if x1.shape[-2] != x2.shape[-2]:
                raise ValueError("diag=True requires equally many points.")
            distances = (x1 != x2).to(x1) / self.lengthscale
            if last_dim_is_batch:
                return (-distances.transpose(-1, -2)).exp()
        else:
            distances = (x1.unsqueeze(-2) != x2.unsqueeze(-3)).to(x1)
            distances = distances / self.lengthscale.unsqueeze(-2)
            if last_dim_is_batch:
                return (-distances.movedim(-1, -3)).exp()
        return (-distances.mean(-1)).exp()


class TorchFidelityKernel(TorchAlgebraKernel):
    """Actual downsampling, exponential-decay and linear-truncated covariances."""

    power: Any

    def __init__(self, source: Any, like: Tensor, convert: Any) -> None:
        TorchKernel.__init__(self, like, source.active_dims)
        self.batch_shape = torch.Size(source.batch_shape)
        self.ard_num_dims = source.ard_num_dims
        self.kind = type(source).__name__.removesuffix("Kernel").lower()
        self._logical_parameters = {}
        shapes = source._element_shapes()
        for raw_name, constraint, prior in source._own_params():
            logical = raw_name.removeprefix("raw_")
            original = getattr(source, raw_name)
            raw = _tensor(original, like).reshape(source._param_shape(raw_name)).clone()
            self.register_parameter(
                raw_name,
                nn.Parameter(
                    raw,
                    requires_grad=not isinstance(original, nn.Parameter) or original.requires_grad,
                ),
            )
            self.add_module(f"{raw_name}_constraint", TorchConstraint(constraint, like))
            self._logical_parameters[logical] = raw_name
            self._register_prior(f"{logical}_prior", prior, logical, len(shapes[raw_name]))
        self.fidelity_dims = tuple(getattr(source, "fidelity_dims", ()))
        if isinstance(source, LinearTruncatedFidelityKernel):
            self.covar_module_unbiased = convert(source.covar_module_unbiased)
            self.covar_module_biased = convert(source.covar_module_biased)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if diag and x1.shape[-2] != x2.shape[-2]:
            raise ValueError("diag=True requires equally many points.")
        if self.kind == "lineartruncatedfidelity":
            if last_dim_is_batch:
                raise UnsupportedError(
                    "Linear-truncated fidelity does not support feature batches."
                )
            continuous = [i for i in range(x1.shape[-1]) if i not in self.fidelity_dims]
            if not continuous:
                raise ValueError("Linear-truncated fidelity requires a non-fidelity coordinate.")
            unbiased = self.covar_module_unbiased(
                x1[..., continuous], x2[..., continuous], diag=diag
            )
            biased = self.covar_module_biased(x1[..., continuous], x2[..., continuous], diag=diag)
            power = self.power if diag else self.power.unsqueeze(-1)
            products, crosses = [], []
            for feature in self.fidelity_dims:
                a, b = x1[..., feature].clamp(0, 1), x2[..., feature].clamp(0, 1)
                if not diag:
                    a, b = a.unsqueeze(-1), b.unsqueeze(-2)
                products.append(a * b)
                crosses.append((1 - a) * (1 - b))
            bias = crosses[0] * (1 + products[0]).pow(power)
            if len(products) == 2:
                bias = bias + crosses[1] * (1 + products[1]).pow(power)
                bias = bias + crosses[0] * crosses[1] * (1 + products[0] + products[1]).pow(power)
            return unbiased + bias * biased

        offset, power = self.offset, self.power
        if self.kind == "exponentialdecay":
            x1, x2 = x1 / self.lengthscale, x2 / self.lengthscale
        if last_dim_is_batch:
            x1, x2 = x1.transpose(-1, -2).unsqueeze(-1), x2.transpose(-1, -2).unsqueeze(-1)
            offset, power = offset.unsqueeze(-1), power.unsqueeze(-1)
        if not diag:
            offset, power = offset.unsqueeze(-1), power.unsqueeze(-1)
        if self.kind == "downsampling":
            a, b = (1 - x1).pow(1 + power.unsqueeze(-1) if diag else 1 + power), (1 - x2)
            if diag:
                # Preserve the actual full-covariance diagonal for d>1, too.
                return offset + (a * b.pow(1 + power.unsqueeze(-1))).sum(-1)
            return offset + a @ b.pow(1 + power).transpose(-1, -2)

        delta = x1 + x2 if diag else x1.unsqueeze(-2) + x2.unsqueeze(-3)
        square = delta.square().sum(-1)
        distance = torch.where(
            square > 0, square.clamp_min(torch.finfo(square.dtype).tiny).sqrt(), 0
        )
        return offset + (1 + distance).pow(-power)


class _ExternalKernel(TorchKernel):
    def __init__(self, source: nn.Module, like: Tensor) -> None:
        super().__init__(like, getattr(source, "active_dims", None))
        self.module = source.to(like)
        self.batch_shape = torch.Size(getattr(source, "batch_shape", ()))

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
        result = self.module.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch)
        return result.to_dense() if hasattr(result, "to_dense") else result


class _SpecializedCompositeKernel(TorchCompositeKernel):
    def log_prior(self) -> Tensor:
        total = super().log_prior()
        seen = set()
        for module in self.modules():
            if isinstance(module, _ExternalKernel) and id(module.module) not in seen:
                seen.add(id(module.module))
                total = total + module_log_prior(module.module, self.batch_shape)
        return total


def specialized_kernel(source: Any, d: int, batch_shape: Any, like: Tensor) -> TorchKernel:
    """Convert the original kernel tree, preserving shared parameter aliases."""
    memo: dict[int, TorchKernel] = {}

    def convert(kernel: Any) -> TorchKernel:
        if id(kernel) in memo:
            return memo[id(kernel)]
        if isinstance(kernel, TorchKernel):
            result = kernel.to(like)
        elif isinstance(kernel, nn.Module):
            result = _ExternalKernel(kernel, like)
        elif isinstance(kernel, CategoricalKernel):
            result = TorchCategoricalKernel(kernel, like)
        elif isinstance(
            kernel, (DownsamplingKernel, ExponentialDecayKernel, LinearTruncatedFidelityKernel)
        ):
            result = TorchFidelityKernel(kernel, like, convert)
        elif type(kernel) is ScaleKernel:
            result = TorchScaleKernel(kernel, convert(kernel.base_kernel), like)
        elif type(kernel) in (AdditiveKernel, ProductKernel):
            result = _SpecializedCompositeKernel(
                [convert(k) for k in kernel.kernels],
                "add" if type(kernel) is AdditiveKernel else "product",
                like,
            )
        else:
            result = cast(TorchKernel, build_kernel(kernel, d, batch_shape, like))
        memo[id(kernel)] = result
        return result

    return convert(source)


def _scaled(base: TorchKernel, batch_shape: Any, like: Tensor) -> TorchScaleKernel:
    result = TorchScaleKernel(ScaleKernel(RBFKernel(batch_shape=batch_shape)), base, like)
    result.active_dims = base.active_dims
    return result


class TorchMixedSingleTaskGP(TorchSingleTaskGP, MixedSingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        cont_kernel_factory: Any = None,
        likelihood: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        self._validate_data(train_X, train_Y, train_Yvar)
        if not cat_dims:
            raise ValueError("Must specify categorical dimensions for MixedSingleTaskGP")
        d, m = train_X.shape[-1], train_Y.shape[-1]
        categories = normalize_indices(list(cat_dims), d)
        assert categories is not None
        if len(set(categories)) != len(categories):
            raise ValueError("Categorical dimensions must be unique.")
        continuous = sorted(set(range(d)) - set(categories))
        batch = train_X.shape[:-2] + ((m,) if m > 1 else ())
        factory = cont_kernel_factory or get_covar_module_with_dim_scaled_prior

        def categorical(selected: Any = None) -> TorchKernel:
            return specialized_kernel(
                CategoricalKernel(
                    batch_shape=batch,
                    ard_num_dims=len(categories),
                    active_dims=selected,
                    lengthscale_constraint=GreaterThan(1e-6),
                ),
                d,
                batch,
                train_X,
            )

        def ordinary() -> TorchKernel:
            return specialized_kernel(
                factory(batch_shape=batch, ard_num_dims=len(continuous), active_dims=continuous),
                d,
                batch,
                train_X,
            )

        kernel: TorchKernel
        if not continuous:
            kernel = _scaled(categorical(), batch, train_X)
        else:
            cat_sum = _scaled(categorical(categories), batch, train_X)
            additive = _SpecializedCompositeKernel([ordinary(), cat_sum], "add", train_X)
            product = _SpecializedCompositeKernel(
                [ordinary(), categorical(categories)], "product", train_X
            )
            kernel = _SpecializedCompositeKernel(
                [_scaled(additive, batch, train_X), _scaled(product, batch, train_X)],
                "add",
                train_X,
            )
        super().__init__(
            train_X,
            train_Y,
            train_Yvar,
            likelihood=likelihood,
            covar_module=kernel,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._ignore_X_dims_scaling_check = categories


class TorchSingleTaskMultiFidelityGP(TorchSingleTaskGP, SingleTaskMultiFidelityGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Any = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        covar_module: Any = None,
        likelihood: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        mean_module: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        self._validate_data(train_X, train_Y, train_Yvar)
        if iteration_fidelity is None and not data_fidelities:
            raise UnsupportedError(
                "SingleTaskMultiFidelityGP requires at least one fidelity parameter."
            )
        batch = train_X.shape[:-2] + ((train_Y.shape[-1],) if train_Y.shape[-1] > 1 else ())
        external = isinstance(covar_module, nn.Module) and not linear_truncated
        source, subset = _setup_multifidelity_covar_module(
            train_X.shape[-1],
            batch,
            iteration_fidelity,
            data_fidelities,
            linear_truncated,
            nu,
            None if external else covar_module,
        )
        kernel = specialized_kernel(source, train_X.shape[-1], batch, train_X)
        if external:
            assert isinstance(kernel, TorchCompositeKernel)
            kernel.kernels[0] = _ExternalKernel(covar_module, train_X)
            subset.pop("covar_module.kernels.0.raw_lengthscale", None)
        super().__init__(
            train_X,
            train_Y,
            train_Yvar,
            likelihood=likelihood,
            covar_module=kernel,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._subset_batch_dict = subset
        self._init_args = dict(
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
        )
