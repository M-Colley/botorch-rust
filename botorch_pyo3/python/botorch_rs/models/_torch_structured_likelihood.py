"""Actual Gaussian observation laws for structured Tensor models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from botorch_rs.models._torch_gaussian_likelihood import (
    gaussian_marginal,
    is_fixed_noise,
    is_gpytorch_gaussian,
)


def validate_scalar_likelihood(likelihood: Any) -> None:
    """Accept scalar Gaussian modules without replacing their registered leaves."""
    from botorch_rs.models._torch_backend_modules import (
        TorchFixedNoiseLikelihood,
        TorchGaussianLikelihood,
    )

    if not is_gpytorch_gaussian(likelihood) and not isinstance(
        likelihood, (TorchGaussianLikelihood, TorchFixedNoiseLikelihood)
    ):
        raise TypeError("Structured exact inference requires a Gaussian likelihood module.")


def scalar_observed_law(
    likelihood: Any,
    mean: Tensor,
    covariance: Tensor,
    inputs: tuple[Tensor, ...],
    *,
    params: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    training: bool = False,
    fixed_noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Keep a supplied scalar marginal, its inputs and observation overrides."""
    arguments = ((list(inputs),) if training else inputs) if params is None else params
    if params is None and is_fixed_noise(likelihood):
        # A structured event has one observation per grid cell, rather than
        # one per feature row. Let the actual event size select fixed noise.
        arguments = ()
    if is_gpytorch_gaussian(likelihood):
        from gpytorch.distributions import MultivariateNormal
        from linear_operator.operators import DenseLinearOperator

        from botorch_rs.models._torch_variational_external_likelihood import (
            posterior_from_gaussian,
        )

        distribution = MultivariateNormal(mean, DenseLinearOperator(covariance))
        if fixed_noise is None:
            observed = likelihood(distribution, *arguments, **(kwargs or {}))
        else:
            observed = torch.func.functional_call(
                likelihood,
                {"noise_covar.noise": fixed_noise},
                (distribution, *arguments),
                kwargs or {},
            )
        posterior = posterior_from_gaussian(observed)
        return posterior.mean.squeeze(-1), posterior._covariance_tensor
    if fixed_noise is not None:
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

        distribution = TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution
        observed = torch.func.functional_call(
            likelihood, {"noise": fixed_noise}, (distribution,), kwargs or {}
        )
        return observed.mean, observed.covariance_matrix
    return gaussian_marginal(
        likelihood,
        mean,
        covariance,
        inputs[0],
        params=() if params is None else params,
        kwargs=kwargs,
    )


def _canonical_multitask(likelihood: Any) -> bool:
    from botorch_rs.models._torch_backend_kronecker import TorchMultitaskGaussianLikelihood

    if isinstance(likelihood, TorchMultitaskGaussianLikelihood):
        return (
            getattr(likelihood.forward, "__func__", None)
            is TorchMultitaskGaussianLikelihood.forward
            and type(likelihood).__call__ is torch.nn.Module.__call__
            and not likelihood._forward_hooks
            and not likelihood._forward_pre_hooks
        )
    if not is_gpytorch_gaussian(likelihood):
        return False
    from gpytorch.likelihoods import Likelihood
    from gpytorch.likelihoods.multitask_gaussian_likelihood import (
        MultitaskGaussianLikelihood,
        _MultitaskGaussianLikelihoodBase,
    )

    return (
        getattr(likelihood.marginal, "__func__", None) is MultitaskGaussianLikelihood.marginal
        and getattr(likelihood._shaped_noise_covar, "__func__", None)
        is _MultitaskGaussianLikelihoodBase._shaped_noise_covar
        and type(likelihood).__call__ is Likelihood.__call__
    )


def multitask_observed_law(
    model: Any,
    mean: Tensor,
    covariance: Tensor,
    inputs: Tensor,
    *,
    params: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """A point-major multitask marginal, including actual module overrides."""
    from botorch_rs.models._torch_backend_kronecker import tensor_kron

    likelihood = model.likelihood
    if _canonical_multitask(likelihood) and not kwargs:
        # Construct the canonical covariance using ordinary Tensor algebra so
        # covariance-factor Hessians do not enter lazy-operator custom VJPs.
        eye = torch.eye(mean.shape[-2], dtype=mean.dtype, device=mean.device)
        return mean, covariance + tensor_kron(eye, model._noise_matrix())
    arguments = (([inputs],) if training else (inputs,)) if params is None else params
    if is_gpytorch_gaussian(likelihood):
        from gpytorch.distributions import MultitaskMultivariateNormal
        from linear_operator.operators import DenseLinearOperator

        distribution = MultitaskMultivariateNormal(
            mean, DenseLinearOperator(covariance), interleaved=True
        )
        observed = likelihood(distribution, *arguments, **(kwargs or {}))
        from botorch_rs.models._torch_variational_external_likelihood import (
            posterior_from_gaussian,
        )

        posterior = posterior_from_gaussian(observed)
        observed_mean = posterior.mean
        q, m = mean.shape[-2:]
        index = torch.arange(q * m, device=mean.device).reshape(m, q).T.flatten()
        observed_covariance = posterior._covariance_tensor.index_select(-2, index).index_select(
            -1, index
        )
    else:
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

        q, m = mean.shape[-2:]
        index = torch.arange(q * m, device=mean.device).reshape(q, m).T.flatten()
        output_major = covariance.index_select(-2, index).index_select(-1, index)
        observed = likelihood(
            TorchGPyTorchPosterior(mean, output_major).distribution,
            *arguments,
            **(kwargs or {}),
        )
        observed_mean = observed.mean
        covariance = observed.covariance_matrix
        index = torch.arange(q * m, device=mean.device).reshape(m, q).T.flatten()
        observed_covariance = covariance.index_select(-2, index).index_select(-1, index)
    if not isinstance(observed_mean, Tensor) or not isinstance(observed_covariance, Tensor):
        raise TypeError("A Gaussian marginal must return Tensor mean and covariance_matrix.")
    if (
        observed_mean.shape[-2:] != mean.shape[-2:]
        or observed_covariance.shape[-2:] != covariance.shape[-2:]
    ):
        raise ValueError("A Gaussian marginal must preserve the structured event shape.")
    return observed_mean, observed_covariance


def condition_likelihood(
    model: Any, conditioned: Any, noise: Any, Y: Tensor, X: Tensor | None = None
) -> None:
    """Retain the likelihood's fantasy protocol and explicit observation noise."""
    if noise is not None:
        if (
            not isinstance(noise, Tensor)
            or noise.dtype != Y.dtype
            or noise.device != Y.device
            or not bool(torch.isfinite(noise).all() & (noise >= 0).all())
            or torch.broadcast_shapes(noise.shape, Y.shape) != Y.shape
        ):
            raise ValueError("Conditioning noise must be finite, nonnegative and broadcast to Y.")
        noise = noise.expand_as(Y)
    dimensions = len(getattr(model, "target_shape", (model.num_outputs,)))
    likelihood = model.likelihood
    hook = getattr(likelihood, "get_fantasy_likelihood", None)
    if is_fixed_noise(likelihood) and noise is None:
        raise RuntimeError("FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg")
    transformed = noise
    if noise is not None and model.outcome_transform is not None and hasattr(model, "target_shape"):
        model._training_values()
        transformed = model.outcome_transform(Y, noise, X=X)[1]
    if callable(hook):
        options = {} if transformed is None else {"noise": transformed.flatten(-dimensions - 1)}
        conditioned.likelihood = hook(**options)
        validate_scalar = not hasattr(conditioned, "_noise_matrix")
        if validate_scalar:
            validate_scalar_likelihood(conditioned.likelihood)
        elif not isinstance(conditioned.likelihood, torch.nn.Module):
            raise TypeError("The fantasy likelihood must be a Gaussian module.")
    elif is_fixed_noise(likelihood):
        import copy

        conditioned.likelihood = copy.copy(likelihood)
        conditioned.likelihood._buffers = likelihood._buffers.copy()
        original = likelihood.noise
        flat = transformed.flatten(-dimensions - 1)
        batch = torch.broadcast_shapes(original.shape[:-1], flat.shape[:-1])
        conditioned.likelihood.noise = torch.cat(
            (original.expand(*batch, original.shape[-1]), flat.expand(*batch, flat.shape[-1])),
            -1,
        )
    if noise is not None:
        segments = getattr(model, "_conditioned_noise_segments", ())
        name = f"_botorch_conditioning_noise_{len(segments)}"
        conditioned.register_buffer(name, noise)
        conditioned._conditioned_noise_segments = (*segments, (model._train_X.shape[-2], name))
    if is_fixed_noise(likelihood):
        if not hasattr(model, "_conditioning_fixed_noise"):
            conditioned.register_buffer(
                "_conditioning_fixed_noise",
                likelihood.noise_covar.noise
                if is_gpytorch_gaussian(likelihood)
                else likelihood.noise,
            )


def conditioned_observed_law(
    model: Any, mean: Tensor, covariance: Tensor, observed: Tensor
) -> Tensor:
    """Replace specified fantasy blocks without rebuilding registered modules."""
    segments = getattr(model, "_conditioned_noise_segments", ())
    if not segments or is_fixed_noise(model.likelihood):
        return observed
    n, m = model._train_X.shape[-2], getattr(model, "_grid_size", model.num_outputs)
    dimensions = len(getattr(model, "target_shape", (m,)))
    batch = torch.broadcast_shapes(covariance.shape[:-2], model.batch_shape)
    keep = covariance.new_ones(n * m)
    diagonal = covariance.new_zeros(*batch, n * m)
    for start, name in segments:
        noise = getattr(model, name)
        count = noise.shape[-dimensions - 1]
        if model.outcome_transform is not None and hasattr(model, "target_shape"):
            axis = -dimensions - 1
            raw_values = model._train_Y.narrow(axis, start, count)
            inputs = model._train_X.narrow(-2, start, count)
            noise = model.outcome_transform(raw_values, noise, X=inputs)[1]
        flat = noise.flatten(-dimensions - 1)
        keep[start * m : (start + count) * m] = 0
        # cat keeps the supplied noise graph and allows broadcast fantasy axes.
        diagonal = torch.cat(
            (
                diagonal[..., : start * m],
                flat.expand(*batch, count * m),
                diagonal[..., (start + count) * m :],
            ),
            -1,
        )
    return (
        covariance
        + (observed - covariance) * keep[:, None] * keep[None, :]
        + torch.diag_embed(diagonal)
    )


def fixed_conditioning_noise(model: Any) -> Tensor | None:
    segments = getattr(model, "_conditioned_noise_segments", ())
    if not segments or not is_fixed_noise(model.likelihood):
        return None
    original = model._conditioning_fixed_noise
    dimensions = len(getattr(model, "target_shape", (model.num_outputs,)))
    values = [original]
    for start, name in segments:
        noise = getattr(model, name)
        if model.outcome_transform is not None:
            axis = -dimensions - 1
            count = noise.shape[axis]
            raw_values = model._train_Y.narrow(axis, start, count)
            inputs = model._train_X.narrow(-2, start, count)
            noise = model.outcome_transform(raw_values, noise, X=inputs)[1]
        values.append(noise.flatten(-dimensions - 1))
    batch = torch.broadcast_shapes(*(value.shape[:-1] for value in values))
    return torch.cat([value.expand(*batch, value.shape[-1]) for value in values], -1)
