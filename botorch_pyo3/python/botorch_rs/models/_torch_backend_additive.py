"""Device-resident orthogonal additive kernels and component GP inference."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import (
    TorchConstraint,
    TorchKernel,
    _tensor,
    build_kernel,
    module_log_prior,
)
from botorch_rs.models.additive_gp import OrthogonalAdditiveGP
from botorch_rs.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.utils.types import DEFAULT


class TorchOrthogonalAdditiveKernel(TorchKernel, OrthogonalAdditiveKernel):  # type: ignore[misc]
    """Quadrature-orthogonalized one-dimensional kernels, with full graphs."""

    _tensor_backend = "torch"
    base_kernel: Any
    coeff_constraint: Any
    z: Tensor
    w: Tensor
    _pairs: Tensor

    @property
    def batch_shape(self) -> Any:
        return self._tensor_batch_shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._tensor_batch_shape = torch.Size(value)

    def __init__(
        self, *args: Any, _source: Any = None, _like: Tensor | None = None, **kwargs: Any
    ) -> None:
        kwargs.pop("backend", None)
        source = (
            OrthogonalAdditiveKernel(*args, backend="native", **kwargs)
            if _source is None
            else _source
        )
        like = (
            torch.empty((), dtype=source._dtype, device=getattr(source.z, "device", None))
            if _like is None
            else _like
        )
        TorchKernel.__init__(self, like)
        self.dim, self.quad_deg = source.dim, source.quad_deg
        self.batch_shape = source.batch_shape
        self.second_order, self.check_hypercube = source.second_order, source.check_hypercube
        self.per_dim_lengthscales, self._per_dim = source.per_dim_lengthscales, source._per_dim
        self.base_kernel = build_kernel(source.base_kernel, 1, source.base_kernel.batch_shape, like)
        self.coeff_constraint = TorchConstraint(source.coeff_constraint, like)
        for name in ("raw_offset", "raw_coeffs_1", "raw_coeffs_2"):
            if name == "raw_coeffs_2" and not self.second_order:
                continue
            original = getattr(source, name)
            self.register_parameter(name, nn.Parameter(_tensor(original, like).clone()))
            self.add_module(f"{name}_constraint", self.coeff_constraint)
        for name, parameter, event in (
            ("offset_prior", "offset", 0),
            ("coeffs_1_prior", "coeffs_1", 1),
            ("coeffs_2_prior", "_positive_coeffs_2", 1),
        ):
            self._register_prior(name, getattr(source, name), parameter, event)
        self.register_buffer(
            "z",
            _tensor(source.z, like)
            .reshape(self.quad_deg, 1)
            .expand(self.quad_deg, self.dim)
            .clone(),
        )
        self.register_buffer("w", _tensor(source.w, like).reshape(self.quad_deg, 1))
        self.register_buffer(
            "_pairs", torch.triu_indices(self.dim, self.dim, 1, device=like.device)
        )

    @property
    def offset(self) -> Tensor:
        return self.coeff_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: Any) -> None:
        self._assign("raw_offset", value, self.coeff_constraint)

    @property
    def coeffs_1(self) -> Tensor:
        return self.coeff_constraint.transform(self.raw_coeffs_1)

    @coeffs_1.setter
    def coeffs_1(self, value: Any) -> None:
        self._assign("raw_coeffs_1", value, self.coeff_constraint)

    @property
    def _positive_coeffs_2(self) -> Tensor:
        return self.coeff_constraint.transform(self.raw_coeffs_2)

    @_positive_coeffs_2.setter
    def _positive_coeffs_2(self, value: Any) -> None:
        self._assign("raw_coeffs_2", value, self.coeff_constraint)

    @property
    def coeffs_2(self) -> Any:
        if not self.second_order:
            return None
        coefficients = self._positive_coeffs_2
        flat = coefficients.new_zeros(*self.batch_shape, self.dim * self.dim)
        return flat.scatter(
            -1,
            (self._pairs[0] * self.dim + self._pairs[1]).expand(*self.batch_shape, -1),
            coefficients,
        ).reshape(*self.batch_shape, self.dim, self.dim)

    @coeffs_2.setter
    def coeffs_2(self, value: Any) -> None:
        if not self.second_order:
            raise UnsupportedError("Second-order coefficients are disabled.")
        matrix = _tensor(value, self._anchor).expand(*self.batch_shape, self.dim, self.dim)
        self._assign(
            "raw_coeffs_2", matrix[..., self._pairs[0], self._pairs[1]], self.coeff_constraint
        )

    @property
    def component_indices(self) -> dict[str, Tensor]:
        result = {
            "bias": torch.tensor([0], device=self._anchor.device),
            "first_order": torch.arange(self.dim, device=self._anchor.device),
        }
        if self.second_order:
            result["second_order"] = self._pairs.T
        return result

    def log_prior(self) -> Tensor:
        value = self._anchor.new_zeros(self.batch_shape)
        for name, (attribute, event) in self._prior_parameters.items():
            term = getattr(self, name).log_prob(getattr(self, attribute))
            value = value + (term.sum(tuple(range(-event, 0))) if event else term)
        base = module_log_prior(self.base_kernel, self.base_kernel.batch_shape)
        if isinstance(base, Tensor) and base.ndim > len(self.batch_shape):
            base = base.sum(-1)
        return value + base

    def k(self, x1: Tensor, x2: Tensor, diag: bool = False) -> Tensor:
        if x1.shape[-1] != self.dim or x2.shape[-1] != self.dim:
            raise ValueError(f"OAK inputs must contain {self.dim} features.")
        values = []
        for i in range(self.dim):
            lengthscale = self.base_kernel.lengthscale
            if self._per_dim:
                lengthscale = lengthscale.select(-3, i)
            elif lengthscale.ndim == len(self.batch_shape) + 3:
                lengthscale = lengthscale.squeeze(-3)
            values.append(
                _rs.tensor_kernel(
                    self.base_kernel.kind,
                    x1[..., i : i + 1],
                    x2[..., i : i + 1],
                    lengthscale,
                    self.base_kernel.nu,
                    diag,
                    False,
                )
            )
        return torch.stack(values, -2 if diag else -3)

    def normalizer(self, eps: float = 1e-6) -> Tensor:
        # Recompute on every call in both modes: eval parameters and higher
        # derivatives are part of the public Tensor graph contract.
        weights = self.w.squeeze(-1)
        return torch.einsum("i,...dij,j->...d", weights, self.k(self.z, self.z), weights).clamp_min(
            eps
        )[..., None, None]

    def _orthogonal_base_kernels(self, x1: Tensor, x2: Tensor, diag: bool = False) -> Tensor:
        if self.check_hypercube:
            for name, x in (("x1", x1), ("x2", x2)):
                if bool(((x < -1e-6) | (x > 1 + 1e-6)).any()):
                    raise ValueError(f"{name} is not in hypercube [0, 1]^d.")
        weights = self.w / self.normalizer().sqrt()
        first = self.k(x1, self.z) @ weights
        second = first if x1 is x2 else self.k(x2, self.z) @ weights
        correction = (first * second).squeeze(-1) if diag else first @ second.transpose(-1, -2)
        return self.k(x1, x2, diag=diag) - correction

    def _non_reduced_forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False
    ) -> Tensor:
        if last_dim_is_batch:
            raise UnsupportedError("OrthogonalAdditiveKernel does not support last_dim_is_batch.")
        orthogonal = self._orthogonal_base_kernels(x1, x2, diag)
        trailing = 1 if diag else 2
        axis = -trailing - 1
        shape = orthogonal.shape[:axis] + (1,) + orthogonal.shape[-trailing:]
        bias = self.offset[(...,) + (None,) * (trailing + 1)].expand(shape)
        first = self.coeffs_1[(...,) + (None,) * trailing] * orthogonal
        values = [bias, first]
        if self.second_order:
            interactions = orthogonal.index_select(axis, self._pairs[0]) * orthogonal.index_select(
                axis, self._pairs[1]
            )
            values.append(self._positive_coeffs_2[(...,) + (None,) * trailing] * interactions)
        return torch.cat(values, dim=axis)

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
        return self._non_reduced_forward(x1, x2, diag, last_dim_is_batch).sum(-2 if diag else -3)


class TorchOrthogonalAdditiveGP(TorchSingleTaskGP, OrthogonalAdditiveGP):
    covar_module: Any

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        covar_module: Any = None,
        second_order: bool = False,
        likelihood: Any = None,
        mean_module: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        if covar_module is None:
            batch = train_Y.shape[:-2] + ((train_Y.shape[-1],) if train_Y.shape[-1] > 1 else ())
            covar_module = TorchOrthogonalAdditiveKernel(
                train_X.shape[-1],
                second_order=second_order,
                batch_shape=batch,
                dtype=train_X.dtype,
                device=train_X.device,
                _like=train_X,
            )
        elif not isinstance(covar_module, OrthogonalAdditiveKernel):
            raise TypeError("covar_module must be an OrthogonalAdditiveKernel.")
        TorchSingleTaskGP.__init__(
            self,
            train_X,
            train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        infer_all_components: bool = False,
        **kwargs: Any,
    ) -> Any:
        if not infer_all_components:
            return TorchSingleTaskGP.posterior(
                self, X, output_indices, observation_noise, posterior_transform, **kwargs
            )
        if kwargs:
            raise UnsupportedError(f"Unsupported posterior arguments: {sorted(kwargs)}.")
        if self.num_outputs != 1:
            raise UnsupportedError("Additive component inference requires one model output.")
        if output_indices is not None and list(output_indices) not in ([0], [-1]):
            raise IndexError("A component model has one output.")
        outcome = self.outcome_transform
        if outcome is not None and not getattr(outcome, "_is_linear", False):
            raise UnsupportedError(
                "Additive component inference requires an affine outcome transform."
            )
        # Reuse validation, transformation fitting and the full training law;
        # component marginals always condition on the full covariance.
        if (
            not isinstance(X, Tensor)
            or X.ndim < 2
            or X.shape[-1] != self._train_X.shape[-1]
            or X.dtype != self._train_X.dtype
            or X.device != self._train_X.device
        ):
            raise ValueError("X must share the model input dimensions, dtype and device.")
        self.eval()
        train, Y, fixed = self._training_values()
        query = self.transform_inputs(X)
        factor, alpha = _rs.tensor_exact_gp_factor(
            Y,
            self._covariance(train, train),
            self.mean_module(train),
            self._training_noise(Y, fixed),
        )
        cross = self.covar_module._non_reduced_forward(train, query).movedim(-3, 0)
        prior = self.covar_module._non_reduced_forward(query, query).movedim(-3, 0)
        batch = torch.broadcast_shapes(train.shape[:-2], query.shape[:-2], self.batch_shape)
        query_mean = self.mean_module(query).expand(*batch, query.shape[-2])
        bias = query_mean.new_zeros(self.num_components)
        bias[0] = 1
        prior_mean = bias.reshape(
            self.num_components, *(1 for _ in range(len(batch) + 1))
        ) * query_mean.unsqueeze(0)
        query_noise = (
            self._noise_for(query.shape[-2])
            if observation_noise is True
            else observation_noise.squeeze(-1)
            if isinstance(observation_noise, Tensor)
            else None
        )
        if query_noise is None and observation_noise is not False:
            raise TypeError("observation_noise must be bool or Tensor.")
        mean, covariance = _rs.tensor_exact_gp_predict(
            factor, alpha, cross, prior, prior_mean, observation_noise=query_noise
        )
        posterior = TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance)
        if outcome is not None:
            posterior = outcome.untransform_posterior(posterior, X=X)
            zero = TorchGPyTorchPosterior(
                torch.zeros_like(mean).unsqueeze(-1), torch.zeros_like(covariance)
            )
            location = outcome.untransform_posterior(zero, X=X).mean
            keep = bias.reshape(self.num_components, *(1 for _ in range(location.ndim - 1)))
            posterior = TorchGPyTorchPosterior(
                posterior.mean - (1 - keep) * location, posterior.distribution.covariance_matrix
            )
        return posterior if posterior_transform is None else posterior_transform(posterior, X=X)

    def evaluate_first_order_on_grid(self, grid_1d: Tensor) -> Any:
        if grid_1d.ndim != 1 or not grid_1d.numel():
            raise ValueError("grid_1d must be a nonempty one-dimensional grid.")
        grid = grid_1d.unsqueeze(-1).expand(-1, self.covar_module.dim)
        posterior = self.posterior(grid, infer_all_components=True)
        mean, variance = posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)
        return (mean[0].mean(-1), variance[0].mean(-1)), (
            mean[1 : 1 + self.covar_module.dim],
            variance[1 : 1 + self.covar_module.dim],
        )
