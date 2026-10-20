"""Explicit optional Torch interop for supplied structured posterior operators."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs.exceptions.errors import BotorchTensorDimensionError


def marginal(posterior: Any, value: Any, density: bool) -> Any:
    """Gaussian marginal evaluation using native elementwise operations."""
    from botorch_rs import _ops
    from botorch_rs._arrays import from_flat, kind_of, to_flat

    mean, variance = posterior.mean, posterior.variance
    data, shape = to_flat(value, "value")
    target = (*shape, *(1 for _ in to_flat(mean)[1])) if len(data) > 1 else ()
    value = (
        value.reshape(target)
        if hasattr(value, "reshape")
        else from_flat(data, target, kind_of(mean))
    )
    sd = _ops.apply("sqrt", variance)
    if density:
        z = _ops.apply("div", _ops.apply("sub", value, mean), sd)
        return _ops.apply("exp", _ops.apply("sub", _ops.apply("log_phi", z), _ops.apply("log", sd)))
    return _ops.apply("add", mean, _ops.apply("mul", sd, _ops.apply("ndtri", value)))


def _operator(value: Any) -> Any:
    from linear_operator.operators import to_linear_operator

    return to_linear_operator(value)


def _draw(covariance: Any, normals: Any) -> Any:
    root = _operator(covariance).root_decomposition().root
    if root.shape[-1] > normals.shape[-1]:
        raise BotorchTensorDimensionError("Covariance root needs more base samples than supplied.")
    return root.matmul(normals[..., : root.shape[-1]].unsqueeze(-1)).squeeze(-1)


class SuppliedMatheron:
    """Matheron updates of explicitly supplied higher-order/multitask operators."""

    def __init__(
        self,
        distribution: Any,
        joint_covariance_matrix: Any,
        train_train_covar: Any,
        test_train_covar: Any,
        train_targets: Any,
        output_shape: Any,
        *,
        train_noise: Any = None,
        test_noise: Any = None,
        test_mean: Any = None,
    ) -> None:
        from botorch_rs.posteriors._distribution import GaussianDistributionAdapter

        GaussianDistributionAdapter(distribution)
        self.distribution = distribution
        self.joint_covariance_matrix = _operator(joint_covariance_matrix)
        self.train_train_covar = _operator(train_train_covar)
        self.test_train_covar = _operator(test_train_covar)
        self.train_targets = train_targets
        self.train_diff = train_targets
        self.output_shape = tuple(output_shape)
        self.test_mean = test_mean
        self.test_noise = None if test_noise is None else _operator(test_noise)
        self.multitask = test_mean is not None
        if not hasattr(train_targets, "shape") or len(train_targets.shape) < (
            2 if self.multitask else 1
        ):
            raise BotorchTensorDimensionError(
                "Expected tensor-valued training responses with explicit event axes."
            )
        self.num_tasks = train_targets.shape[-1] if self.multitask else 1
        self.observation_noise = self.test_noise is not None
        n = self.train_train_covar.shape[-1]
        if train_noise is None:
            # HOGP's public train covariance includes likelihood noise. Keep
            # the supplied noise operator (and its graph) when explicitly
            # represented as a sum, otherwise recover its covariance.
            terms: Any = getattr(self.train_train_covar, "linear_ops", ())
            noise = (
                sum(terms[1:], terms[0] * 0)
                if len(terms) > 1
                else (self.train_train_covar - self.joint_covariance_matrix[..., :n, :n])
            )
            self.train_noise = _operator(noise)
            self.observed_covariance = self.train_train_covar
        else:
            self.train_noise = _operator(train_noise)
            self.observed_covariance = self.train_train_covar + self.train_noise
        joint_size = self.joint_covariance_matrix.shape[-1]
        test_size = joint_size - n
        if (
            n <= 0
            or test_size <= 0
            or self.joint_covariance_matrix.shape[-2] != joint_size
            or self.train_train_covar.shape[-2] != n
            or self.test_train_covar.shape[-2:] != (test_size, n)
            or self.train_noise.shape[-2:] != (n, n)
            or (
                self.test_noise is not None and self.test_noise.shape[-2:] != (test_size, test_size)
            )
        ):
            raise BotorchTensorDimensionError(
                "Supplied Matheron operators have incompatible shapes."
            )
        self.num_train = n
        self.test_size = test_size
        self.batch_shape = tuple(self.joint_covariance_matrix.shape[:-2])
        if math.prod(self.output_shape) != math.prod(self.batch_shape) * test_size:
            raise BotorchTensorDimensionError(
                "output_shape does not match the conditional event size."
            )
        if tuple(distribution.mean.shape) != self.output_shape:
            raise BotorchTensorDimensionError("distribution.mean must have output_shape.")
        target_shape = tuple(train_targets.shape)
        count = (
            math.prod(target_shape[-2:])
            if self.multitask
            else (
                target_shape[-2]
                if len(target_shape) >= 2 and target_shape[-1] == 1
                else target_shape[-1]
            )
        )
        if count != n:
            raise BotorchTensorDimensionError(
                "Training targets do not match the observed event size."
            )

    @property
    def mean(self) -> Any:
        return self.distribution.mean

    @property
    def variance(self) -> Any:
        return self.distribution.variance

    @property
    def base_sample_shape(self) -> tuple[int, ...]:
        size = self.joint_covariance_matrix.shape[-1] + self.num_train
        if self.test_noise is not None:
            size += self.test_size
        return (*self.batch_shape, size)

    def prepare(self, sample_shape: Any, base_samples: Any = None) -> Any:
        import torch

        shape = torch.Size(sample_shape)
        total = self.base_sample_shape[-1]
        target = shape + self.base_sample_shape
        if base_samples is None:
            return torch.randn(target, dtype=self.mean.dtype, device=self.mean.device)
        if base_samples.shape[: len(shape)] != shape:
            raise RuntimeError("sample_shape disagrees with shape of base_samples.")
        if base_samples.ndim == len(target) + 1 and base_samples.shape[-1] == 1:
            base_samples = base_samples.squeeze(-1)
        if base_samples.shape[-1] != total:
            partial = shape + self.batch_shape + (self.test_size,)
            if base_samples.numel() != math.prod(partial):
                raise BotorchTensorDimensionError(
                    "Base samples do not match the Matheron base layout."
                )
            base_samples = base_samples.reshape(partial)
            additional = torch.randn(
                *partial[:-1],
                total - self.test_size,
                dtype=base_samples.dtype,
                device=base_samples.device,
            )
            base_samples = torch.cat([base_samples, additional], dim=-1)
        try:
            return base_samples.expand(target)
        except RuntimeError as error:
            raise BotorchTensorDimensionError(
                "Base sample batches do not match the posterior."
            ) from error

    def rsample(
        self, sample_shape: Any = None, base_samples: Any = None, train_diff: Any = None
    ) -> Any:
        import torch

        shape = torch.Size([1]) if sample_shape is None else torch.Size(sample_shape)
        normals = self.prepare(shape, base_samples)
        joint_size, n = self.joint_covariance_matrix.shape[-1], self.num_train
        joint = _draw(self.joint_covariance_matrix, normals[..., :joint_size])
        noise = _draw(self.train_noise, normals[..., joint_size : joint_size + n])
        targets = self.train_targets if train_diff is None else train_diff
        if self.multitask:
            targets = targets.flatten(-2)
        elif targets.shape[-1:] == (1,):
            targets = targets.squeeze(-1)
        rhs = targets - joint[..., :n] - noise
        solved = self.observed_covariance.solve(rhs.unsqueeze(-1))
        samples = joint[..., n:] + self.test_train_covar.matmul(solved).squeeze(-1)
        if self.test_noise is not None:
            samples = samples + _draw(self.test_noise, normals[..., joint_size + n :])
        samples = samples.reshape(*shape, *self.output_shape)
        return samples if self.test_mean is None else samples + self.test_mean


class SuppliedLatent:
    """Moment distribution plus an explicitly supplied model sampling protocol."""

    def __init__(self, model: Any, distribution: Any, X: Any, T: Any) -> None:
        import torch

        from botorch_rs.posteriors._distribution import GaussianDistributionAdapter

        GaussianDistributionAdapter(distribution)
        method = getattr(model, "_rsample_from_base_samples", None)
        if not callable(method):
            raise TypeError(
                "The supplied model must implement _rsample_from_base_samples(X, T, base_samples)."
            )
        if not all(
            hasattr(model, name)
            for name in ("batch_shape", "train_inputs", "train_T", "train_targets")
        ):
            raise TypeError(
                "The supplied model must expose batch_shape and training-grid metadata."
            )
        if not isinstance(X, torch.Tensor) or not isinstance(T, torch.Tensor):
            raise TypeError("Supplied model interoperability requires Torch query tensors.")
        if X.ndim < 2 or T.ndim < 2 or T.shape[-1] != 1:
            raise BotorchTensorDimensionError("Expected X batch x q x d and T batch x t x 1.")
        self.model, self.distribution, self.X, self.T = model, distribution, X, T
        self.batch_shape = tuple(model.batch_shape)
        self.output_batch_shape = tuple(
            torch.broadcast_shapes(self.batch_shape, X.shape[:-2], T.shape[:-2])
        )
        self.output_shape = (*self.output_batch_shape, X.shape[-2], T.shape[-2])
        if tuple(distribution.mean.shape) != self.output_shape:
            raise BotorchTensorDimensionError("distribution.mean must have batch x q x t shape.")
        full = model.train_inputs[0].shape[-2] * model.train_T.shape[-2]
        observed = model.train_targets.shape[-1]
        self.base_sample_shape = (*self.batch_shape, full + observed + X.shape[-2] * T.shape[-2])

    def rsample(self, sample_shape: Any = None, base_samples: Any = None) -> Any:
        import torch

        shape = torch.Size([1]) if sample_shape is None else torch.Size(sample_shape)
        target = shape + self.base_sample_shape
        if base_samples is None:
            base_samples = torch.randn(target, dtype=self.X.dtype, device=self.X.device)
        if base_samples.shape[: len(shape)] != shape:
            raise RuntimeError("sample_shape disagrees with shape of base_samples.")
        if base_samples.shape[-1] != self.base_sample_shape[-1]:
            raise BotorchTensorDimensionError(
                "Base samples do not match the latent Matheron layout."
            )
        result = self.model._rsample_from_base_samples(self.X, self.T, base_samples.expand(target))
        if not isinstance(result, torch.Tensor):
            raise TypeError("The supplied model must return a Torch sample tensor.")
        if tuple(result.shape) != (*shape, *self.output_shape):
            raise BotorchTensorDimensionError(
                "The supplied model returned an incompatible sample shape."
            )
        return result
