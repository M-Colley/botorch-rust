"""Device-resident contour quadrature for symmetric variational whitening."""

from __future__ import annotations

import math
import sys
from typing import Any

import torch
from torch import Tensor

from botorch_rs.models._torch_backend_variational import (
    _dense,
    _distribution_moments,
    _TorchVariationalOutput,
    _TorchVariationalStrategy,
)


def _contour_parameters(matrix: Tensor, count: int) -> tuple[Tensor, Tensor]:
    """Elliptic contour nodes via arithmetic-geometric means and Landen descent.

    The spectral bounds and contour are quadrature setup, as in the source.
    They stay on the input device but do not enter the derivative graph. The
    resolvent solves below supply every derivative of the rational approximation.
    """
    if count < 1:
        raise ValueError("The contour quadrature count must be positive.")
    with torch.no_grad():
        spectrum = torch.linalg.eigvalsh(matrix)
        lower, upper = spectrum[..., 0], spectrum[..., -1]
        if not bool(torch.isfinite(spectrum).all()) or not bool((lower > 0).all()):
            raise RuntimeError("CIQ requires a finite positive definite inducing covariance.")
        ratio = lower / upper
        a, b = torch.ones_like(ratio), ratio.sqrt()
        descent = []
        for _ in range(32):
            c, next_a = (a - b) / 2, (a + b) / 2
            descent.append((c, next_a))
            a, b = next_a, (a * b).sqrt()
            if bool((c.abs() <= torch.finfo(matrix.dtype).eps * a).all()):
                break
        else:
            raise RuntimeError("CIQ elliptic contour setup did not converge.")
        complete_integral = math.pi / (2 * a)
        fractions = (torch.arange(count, dtype=matrix.dtype, device=matrix.device) + 0.5) / count
        arguments = fractions.reshape(count, *([1] * ratio.ndim)) * complete_integral
        amplitude = (2 ** len(descent)) * a * arguments
        for c, next_a in reversed(descent):
            correction = (c / next_a * amplitude.sin()).clamp(-1, 1).asin()
            amplitude = (amplitude + correction) / 2
        sn, cn = amplitude.sin(), amplitude.cos()
        # This form avoids cancellation in 1 - (1-ratio) * sn**2.
        dn = (cn.square() + ratio * sn.square()).sqrt()
        shifts = lower * (sn / cn).square()
        weights = (2 * complete_integral * lower.sqrt() / (math.pi * count)) * dn / cn.square()
        return weights, shifts


def contour_inverse_root_matmul(matrix: Tensor, rhs: Tensor, count: int = 15) -> Tensor:
    """Approximate ``matrix**(-1/2) @ rhs`` with an elliptic resolvent contour.

    Dense shifted solves are intentional: unlike the reference's detached
    MINRES custom backward, Torch's solves retain higher derivative graphs.
    No eigenvectors or Cholesky whitening are used.
    """
    batch = torch.broadcast_shapes(matrix.shape[:-2], rhs.shape[:-2])
    matrix = matrix.expand(*batch, *matrix.shape[-2:])
    rhs = rhs.expand(*batch, *rhs.shape[-2:])
    weights, shifts = _contour_parameters(matrix, count)
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    systems = matrix.unsqueeze(0) + shifts[..., None, None] * eye
    solves = torch.linalg.solve(systems, rhs.unsqueeze(0))
    return (weights[..., None, None] * solves).sum(0)


class TorchCiqVariationalStrategy(_TorchVariationalStrategy):
    symmetric_whitening = True
    _last_ngd_kl: Tensor

    def _ngd(self) -> bool:
        gpytorch = sys.modules.get("gpytorch")
        return gpytorch is not None and isinstance(
            self._variational_distribution, gpytorch.variational.NaturalVariationalDistribution
        )

    @property
    def variational_distribution(self) -> Any:
        if self._ngd():
            raise RuntimeError(
                "Variational distribution for NGD-CIQ should be computed during forward calls."
            )
        return super().variational_distribution

    def _initialize(self) -> None:
        if not self._ngd():
            return super()._initialize()
        if not bool(self.variational_params_initialized):
            distribution = self._variational_distribution
            with torch.no_grad():
                distribution.natural_vec.copy_(torch.randn_like(distribution.natural_vec) * 1e-3)
                distribution.natural_mat.copy_(
                    -0.5
                    * torch.eye(
                        distribution.num_inducing_points,
                        dtype=self.inducing_points.dtype,
                        device=self.inducing_points.device,
                    )
                )
                self.variational_params_initialized.fill_(1)

    def kl_divergence(self, moments: tuple[Tensor, Tensor, bool] | None = None) -> Tensor:
        if not self._ngd():
            return super().kl_divergence(moments)
        if moments is None:
            if not hasattr(self, "_last_ngd_kl"):
                raise RuntimeError(
                    "KL divergence for NGD-CIQ should be computed during forward calls."
                )
            return self._last_ngd_kl
        regularizer = super().kl_divergence(moments)
        # Source NGD-CIQ skips the expensive KL primal but returns its canonical
        # parameter derivatives. Preserve that explicit surrogate convention.
        self._last_ngd_kl = regularizer - regularizer.detach()
        return self._last_ngd_kl

    def variational_output(self, X: Tensor) -> Any:
        owner = self._owner_ref()
        X = X.clone()
        query = owner._augment(X)
        points = self.inducing_points
        inducing = points if points.ndim == owner._train_X.ndim + 1 else owner._augment(points)
        kernel = owner.model.covar_module
        m, n = points.shape[-2], X.shape[-2]
        eye = torch.eye(m, dtype=X.dtype, device=X.device)
        matrix = _dense(kernel(inducing, inducing)) + self.jitter_val * eye
        gpytorch = sys.modules.get("gpytorch")
        count = 15 if gpytorch is None else gpytorch.settings.num_contour_quadrature.value()
        interpolation = contour_inverse_root_matmul(
            matrix, _dense(kernel(inducing, query)), count=count
        )
        if self._ngd():
            assert gpytorch is not None
            # Source CIQ reads Natural's natural_vec/natural_mat directly,
            # including for subclasses. Use its registered canonical VJP,
            # rather than an arbitrary subclass forward override, for the
            # same natural-coordinate law.
            value = gpytorch.variational.NaturalVariationalDistribution.forward(
                self._variational_distribution
            )
            moments = value.mean.clone(), value.covariance_matrix, False
        else:
            moments = _distribution_moments(self._variational_distribution)
        variational_mean, covariance_q, _ = moments
        mean = owner.model.mean_module(query) + (
            interpolation.mT @ variational_mean.unsqueeze(-1)
        ).squeeze(-1)
        prior = _dense(kernel(query, query))
        covariance = prior + interpolation.mT @ ((covariance_q - eye) @ interpolation)
        if self._ngd():
            variance = covariance.diagonal(dim1=-2, dim2=-1) + self.jitter_val
            variance = variance.clamp_min(1e-6 if X.dtype == torch.float32 else 1e-10)
            covariance = torch.diag_embed(variance)
        else:
            # The pinned CIQ strategy adds data jitter twice in its GD branch.
            covariance = covariance + 2 * self.jitter_val * torch.eye(
                n, dtype=X.dtype, device=X.device
            )
            variance = covariance.diagonal(dim1=-2, dim2=-1)
        return _TorchVariationalOutput(
            owner, mean, variance, lambda: covariance, variational_moments=moments
        )
