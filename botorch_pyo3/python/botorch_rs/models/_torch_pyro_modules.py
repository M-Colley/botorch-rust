"""Device module construction for the public probabilistic loader hooks."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import GreaterThan
from botorch_rs.kernels import LinearKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models._torch_backend_modules import (
    TorchCompositeKernel,
    TorchInputChain,
    build_input_transform,
    build_kernel,
    build_likelihood,
    build_mean,
)
from botorch_rs.models.transforms.input import Normalize, Warp


def mean_module(program: Any, samples: dict[str, Any], batch: Any) -> nn.Module:
    result: Any = build_mean(ConstantMean(batch_shape=batch), batch, program.train_X)
    result.constant = samples["mean"].detach().reshape(batch)
    return result


def likelihood_module(program: Any, samples: dict[str, Any], batch: Any) -> nn.Module:
    fixed = program.train_Yvar
    fixed = None if fixed is None else fixed.squeeze(-1).expand(*batch, fixed.shape[-2])
    result: Any = build_likelihood(
        GaussianLikelihood(batch_shape=batch, noise_constraint=GreaterThan(1e-4))
        if fixed is None
        else None,
        fixed,
        batch,
        program.train_X,
    )
    if fixed is None:
        result.noise = samples["noise"].detach().clamp_min(1e-4).reshape(*batch, 1)
    else:
        # Retain the explicit data-noise graph: the ordinary inspection-only
        # fixed-noise builder deliberately stores a detached initial snapshot.
        minimum = 1e-4 if fixed.dtype == torch.float32 else 1e-6
        result.noise = fixed.clamp_min(minimum)
    return result


def warping_module(program: Any, samples: dict[str, Any], batch: Any) -> Any:
    if not program.use_input_warping:
        return None
    like = program.train_X
    d = like.shape[-1]
    indices = program._warp_indices()
    result: Any = build_input_transform(
        Warp(
            d=d, indices=indices, batch_shape=batch, bounds=[[0.0] * d, [1.0] * d], eps=program._eps
        ),
        d,
        like,
    )
    with torch.no_grad():
        for i in (0, 1):
            getattr(result, f"concentration{i}").copy_(
                samples[f"c{i}"].reshape(*batch, 1, len(indices))
            )
    return result


def family_modules(
    program: Any,
    samples: dict[str, Any],
    linear: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Call actual loader hooks without flattening Tensor samples on the host."""
    like = program.train_X
    count = samples["mean"].shape[0]
    batch = torch.Size([count])
    tkwargs = {"dtype": like.dtype, "device": like.device}
    mean = program._build_mean_module(samples, batch, **tkwargs)
    if linear:
        covariance = build_kernel(
            LinearKernel(ard_num_dims=program.ard_num_dims, batch_shape=batch),
            program.ard_num_dims,
            batch,
            like,
        )
        covariance.variance = (
            samples["weight_variance"].detach().reshape(count, 1, program.ard_num_dims)
        )
    else:
        covariance = program._get_covar_module(
            use_scale_kernel="outputscale" in samples, batch_shape=batch, **tkwargs
        )
        if not isinstance(covariance, nn.Module):
            covariance = build_kernel(covariance, program.ard_num_dims, batch, like)
        base: Any = covariance
        if hasattr(covariance, "raw_outputscale"):
            covariance.outputscale = samples["outputscale"].detach().reshape(batch)
            base = covariance.base_kernel
        base.lengthscale = samples["lengthscale"].detach().reshape(count, 1, program.ard_num_dims)
    covariance = program._build_multitask_covariance(samples, covariance, batch, **tkwargs)
    likelihood = program._likelihood(samples, batch)
    transform = program._warping_function(samples, batch)
    if linear:
        center: Any
        if hasattr(program, "task_feature"):
            from botorch_rs.models._torch_backend_fully_bayesian_multitask import _CenterFeatures

            indices = torch.tensor(
                [i for i in range(like.shape[-1]) if i != program.task_feature], device=like.device
            )
            center = _CenterFeatures(indices)
        else:
            center = build_input_transform(
                Normalize(
                    d=like.shape[-1],
                    bounds=[[0.0] * like.shape[-1], [1.0] * like.shape[-1]],
                    center=0.0,
                ),
                like.shape[-1],
                like,
            )
        transform = TorchInputChain(
            {**({"warp": transform} if transform is not None else {}), "center": center}, like
        )
    return mean, covariance, likelihood, transform


def multitask_mean(program: Any, samples: dict[str, Any], batch: Any) -> nn.Module:
    from botorch_rs.models._torch_backend_multitask import TorchTaskMean

    like = program.train_X
    result = TorchTaskMean(program.num_tasks, like)
    result.base_means = nn.ModuleList(
        [build_mean(ConstantMean(batch_shape=batch), batch, like) for _ in range(program.num_tasks)]
    )
    for task, component in enumerate(result.base_means):
        component.constant = samples["mean"][:, task].detach()
    return result


def multitask_covariance(
    program: Any, samples: dict[str, Any], data: nn.Module, batch: Any
) -> nn.Module:
    from botorch_rs.models._torch_backend_fully_bayesian_multitask import _SampleIndexKernel
    from botorch_rs.models._torch_backend_multitask import _ContinuousKernel

    like = program.train_X
    indices = torch.tensor(
        [i for i in range(like.shape[-1]) if i != program.task_feature], device=like.device
    )
    data = _ContinuousKernel(data, indices, like)
    task = _SampleIndexKernel(program.num_tasks, batch, program.task_feature, like)
    latent = samples["latent_features"].detach()
    covariance = _rs.tensor_kernel(
        "matern", latent, latent, samples["task_lengthscale"].detach().unsqueeze(-2), nu=2.5
    )
    with torch.no_grad():
        task.covar_factor.copy_(_rs.tensor_psd_safe_cholesky(covariance))
        task.raw_var.fill_(-torch.inf)
    return TorchCompositeKernel([cast(Any, data), cast(Any, task)], "product", like)
