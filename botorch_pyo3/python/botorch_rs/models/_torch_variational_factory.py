"""A prior host for a caller's actual variational-strategy component.

This optional interoperability path never imports BoTorch. Canonical port
strategies use the port's Tensor algebra instead of this host.
"""

from __future__ import annotations

from typing import Any

import gpytorch
from linear_operator.operators import DenseLinearOperator
from torch import Tensor, nn


class CallerStrategyGP(gpytorch.models.ApproximateGP):
    """Run the supplied strategy with our mean/kernel as its actual GP prior."""

    _tensor_backend = "torch"

    def __init__(
        self,
        factory: Any,
        points: Tensor,
        distribution: nn.Module,
        learn: bool,
        covariance: nn.Module,
        mean: nn.Module,
        outputs: int,
        inputs: Tensor,
        targets: Tensor | None,
    ) -> None:
        self._aug_batch_shape = inputs.shape[:-2] + ((outputs,) if outputs > 1 else ())
        strategy = factory(
            self,
            inducing_points=points,
            variational_distribution=distribution,
            learn_inducing_locations=learn,
        )
        if not isinstance(strategy, nn.Module) or not callable(
            getattr(strategy, "kl_divergence", None)
        ):
            raise TypeError("variational_strategy must return a Torch module with kl_divergence().")
        if outputs > 1:
            strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                strategy, num_tasks=outputs, task_dim=-1
            )
        super().__init__(strategy)
        self.mean_module, self.covar_module = mean, covariance
        # These are the same public training-data attributes that the source
        # wrapper attaches to its approximate model, not a hidden outer owner.
        self.train_inputs = [inputs]
        if targets is not None:
            self.train_targets = targets.squeeze(-1)

    def forward(self, X: Tensor) -> Any:
        from botorch_rs.models._torch_backend_variational import _dense

        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(X), DenseLinearOperator(_dense(self.covar_module(X)))
        )


def caller_distribution(distribution: nn.Module, family: str, like: Tensor) -> nn.Module:
    """Preserve actual modules; convert port specifications without copying leaves."""
    if isinstance(distribution, gpytorch.variational._VariationalDistribution):
        return distribution
    classes = {
        "cholesky": gpytorch.variational.CholeskyVariationalDistribution,
        "meanfield": gpytorch.variational.MeanFieldVariationalDistribution,
        "delta": gpytorch.variational.DeltaVariationalDistribution,
    }
    result = classes[family](
        distribution.num_inducing_points,
        batch_shape=distribution.batch_shape,
        mean_init_std=distribution.mean_init_std,
    ).to(like)
    for name, parameter in distribution.named_parameters(recurse=False):
        result.register_parameter(name, parameter)
    return result


def caller_likelihood(likelihood: nn.Module, supplied: Any, like: Tensor) -> nn.Module:
    """Retain user modules and give port Gaussian descriptors the public protocol."""
    from botorch_rs.models._torch_backend_approximate import (
        _MultitaskLikelihood,
        _ScalarLikelihood,
    )
    from botorch_rs.models._torch_backend_modules import TorchGaussianLikelihood
    from botorch_rs.models._torch_backend_variational import _TorchMultitaskNoise

    if isinstance(supplied, nn.Module):
        return supplied.to(like)
    if isinstance(likelihood, TorchGaussianLikelihood):
        adapted: nn.Module = object.__new__(_ScalarLikelihood)
    elif isinstance(likelihood, _TorchMultitaskNoise):
        adapted = object.__new__(_MultitaskLikelihood)
    else:
        return likelihood
    # This is the same real likelihood state and the same Parameter leaves,
    # with expected_log_prob added; no wrapper parameter prefix is introduced.
    adapted.__dict__ = likelihood.__dict__
    return adapted
