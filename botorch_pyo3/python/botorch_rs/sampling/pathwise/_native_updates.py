"""Arbitrary exact Gaussian path updates with native array linear algebra."""

from __future__ import annotations

import math
import secrets
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import broadcast_shapes, first_kind, from_flat, to_flat
from botorch_rs._ops import apply, expand, full, getitem, shape_of
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.pathwise.features.maps import KernelEvaluationMap, _matmul
from botorch_rs.sampling.pathwise.paths import GeneralizedLinearPath


def _dense(value: Any) -> Any:
    return value.to_dense() if hasattr(value, "to_dense") else value


def _cholesky(value: Any, kind: Any) -> Any:
    values, shape = to_flat(value)
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError("Path noise/covariance must be square matrices.")
    data, _ = _rs.psd_safe_cholesky(values, math.prod(shape[:-2]), shape[-1])
    return from_flat(data, shape, kind)


def _diagonal_noise(noise: Any, points: Any, kind: Any) -> Any:
    batch = broadcast_shapes(shape_of(noise)[:-1], shape_of(points)[:-2])
    if batch is None:
        raise ValueError("Likelihood noise and update point batches do not broadcast.")
    shape = (*batch, shape_of(points)[-2])
    values = to_flat(expand(noise, shape))[0]
    n = shape[-1]
    data = [
        values[b * n + i] if i == j else 0.0
        for b in range(math.prod(shape[:-1]))
        for i in range(n)
        for j in range(n)
    ]
    return from_flat(data, (*shape[:-1], n, n), kind)


def native_gaussian_update(
    model: Any,
    sample_values: Any,
    likelihood: Any,
    target_values: Any,
    noise: Any,
    seed: Any,
    noise_covariance: Any,
    points: Any,
    scale_tril: Any,
) -> Any:
    from botorch_rs.sampling.pathwise.update_strategies import DEFAULT
    from botorch_rs.sampling.pathwise.utils import get_train_inputs, get_train_targets

    if points is None:
        (points,) = get_train_inputs(model, transformed=True)
    target = get_train_targets(model, transformed=True) if target_values is None else target_values
    kind = first_kind(sample_values, points, target)
    n = shape_of(points)[-2]
    if shape_of(sample_values)[-1:] != (n,):
        raise ValueError("sample_values must end in the update-point dimension.")
    if likelihood is DEFAULT:
        likelihood = model.likelihood
    if noise_covariance is None:
        if likelihood is None:
            covariance_noise = None
        elif callable(getattr(likelihood, "noise_covar", None)):
            covariance_noise = likelihood.noise_covar(shape=shape_of(points)[:-1])
            covariance_noise = (
                None
                if type(covariance_noise).__name__ == "ZeroLinearOperator"
                else _dense(covariance_noise)
            )
        else:
            try:
                variances = likelihood.noise
            except AttributeError:
                constraint = likelihood.noise_constraint
                raw = constraint.initial_value
                variances = full((1,), constraint.transform(0.0 if raw is None else raw), kind)
            from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood

            covariance_noise = (
                None
                if isinstance(likelihood, FixedNoiseGaussianLikelihood)
                and shape_of(variances)[-1] != n
                else _diagonal_noise(variances, points, kind)
            )
    elif type(noise_covariance).__name__ == "ZeroLinearOperator":
        covariance_noise = None
    else:
        covariance_noise = _dense(noise_covariance)
    inputs = (sample_values, target, points, scale_tril, covariance_noise, noise)
    if any(getattr(value, "requires_grad", False) for value in inputs if value is not None):
        raise UnsupportedError(
            "Differentiable path-update overrides require a model constructed with backend='torch'."
        )
    errors = apply("sub_raw", target, sample_values)
    if covariance_noise is not None:
        if noise is None:
            shape = shape_of(sample_values)
            data, _ = _rs.torch_randn(
                secrets.randbits(63) if seed is None else seed, math.prod(shape)
            )
            noise = from_flat(data, shape, kind)
        elif shape_of(noise) != shape_of(sample_values):
            raise ValueError("Standard normal noise must match sample_values.shape.")
        factor_noise = _cholesky(covariance_noise, kind)
        noise_values = getitem(_matmul(factor_noise, getitem(noise, (..., None))), (..., 0))
        errors = apply("sub_raw", errors, noise_values)
    if scale_tril is None:
        covariance = model.covar_module(points, points)
        if covariance_noise is not None:
            covariance = apply("add_raw", covariance, covariance_noise)
        scale_tril = _cholesky(covariance, kind)
    else:
        scale_tril = _dense(scale_tril)
    factor_shape, error_shape = shape_of(scale_tril), shape_of(errors)
    if factor_shape[-2:] != (n, n):
        raise ValueError("scale_tril must end in n x n for the update points.")
    batch = broadcast_shapes(factor_shape[:-2], error_shape[:-1])
    if batch is None:
        raise ValueError("Path factor and error batch dimensions do not broadcast.")
    weights = _rs.pathwise_cholesky_solve(
        to_flat(expand(scale_tril, (*batch, n, n))),
        to_flat(expand(errors, (*batch, n))),
    )
    return GeneralizedLinearPath(
        feature_map=KernelEvaluationMap(
            model.covar_module, points, input_transform=getattr(model, "input_transform", None)
        ),
        weight=from_flat(weights[0], weights[1], kind),
        is_ensemble=bool(getattr(model, "_is_ensemble", False)),
    )
