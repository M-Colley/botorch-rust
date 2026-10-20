"""Independent model lists retaining device tensors and their autograd graph."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.functional import pad

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior, TorchPosteriorList


class TorchLikelihoodList(torch.nn.Module):
    def __init__(self, *likelihoods: Any) -> None:
        super().__init__()
        self.likelihoods = torch.nn.ModuleList(likelihoods)

    def forward(self, *distributions: Any, **kwargs: Any) -> Any:
        return [
            module(distribution, **kwargs)
            for module, distribution in zip(self.likelihoods, distributions, strict=True)
        ]


class TorchModelListGP(torch.nn.Module, ModelListGP):
    _tensor_backend = "torch"
    models: Any
    likelihood: Any

    def __init__(self, *models: Any) -> None:
        torch.nn.Module.__init__(self)
        if not models or any(
            getattr(model, "_tensor_backend", None) != "torch" for model in models
        ):
            raise UnsupportedError(
                "A Torch ModelListGP requires only Torch-backend component models."
            )
        dimensions = {model._train_X.shape[-1] for model in models}
        if len(dimensions) != 1:
            raise ValueError("All component models must have the same input dimension.")
        kinds = {(model._train_X.dtype, model._train_X.device) for model in models}
        if len(kinds) != 1:
            raise ValueError("All component models must share dtype and device.")
        torch.broadcast_shapes(*(model.batch_shape for model in models))
        self.models = torch.nn.ModuleList(models)
        self.likelihood = TorchLikelihoodList(*(model.likelihood for model in models))

    @property
    def num_outputs(self) -> int:
        return sum(model.num_outputs for model in self.models)

    @property
    def batch_shape(self) -> Any:
        return torch.broadcast_shapes(*(model.batch_shape for model in self.models))

    def posterior(
        self,
        X: Any,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        batch = torch.broadcast_shapes(self.batch_shape, X.shape[:-2])
        X = X.expand(*batch, *X.shape[-2:])
        parts = []
        start = 0
        for model in self.models:
            noise = observation_noise
            if isinstance(noise, torch.Tensor):
                noise = noise[..., start : start + model.num_outputs]
            parts.append(model.posterior(X, observation_noise=noise, **kwargs))
            start += model.num_outputs
        if any(not isinstance(posterior, TorchGPyTorchPosterior) for posterior in parts):
            result = TorchPosteriorList(*parts)
            if output_indices is not None:
                result = result.subset_output(output_indices)
            return result if posterior_transform is None else posterior_transform(result, X=X)
        batch = torch.broadcast_shapes(*(p.batch_shape for p in parts))
        q = parts[0]._q
        mean = torch.cat([p.mean.expand(*batch, q, p._m) for p in parts], dim=-1)
        size = q * self.num_outputs
        covariance = mean.new_zeros(*batch, size, size)
        start = 0
        for posterior in parts:
            width = posterior._m * q
            covariance = covariance + pad(
                posterior.distribution.covariance_matrix,
                (start, size - start - width, start, size - start - width),
            )
            start += width
        posterior = TorchGPyTorchPosterior(mean, covariance)
        if output_indices is not None:
            posterior = posterior.subset_output(output_indices)
        return posterior if posterior_transform is None else posterior_transform(posterior, X=X)

    def forward(self, *inputs: Any) -> Any:
        return [model(value) for model, value in zip(self.models, inputs, strict=True)]

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        if len(X) not in (len(self.models), self.num_outputs):
            raise ValueError("Provide one conditioning input per component or output.")
        models = []
        start = 0
        for i, model in enumerate(self.models):
            stop = start + model.num_outputs
            inputs = X[i] if len(X) == len(self.models) else X[start]
            # A component GP shares training locations across its outputs.
            if len(X) == self.num_outputs and model.num_outputs > 1:
                if any(not torch.equal(inputs, X[j]) for j in range(start + 1, stop)):
                    raise UnsupportedError(
                        "A multi-output component requires matching inputs for its outputs."
                    )
            variance = None if noise is None else noise[..., start:stop]
            if variance is not None and model.outcome_transform is not None:
                variance = model.outcome_transform(Y[..., start:stop], variance)[1]
            models.append(
                model.condition_on_observations(
                    inputs, Y[..., start:stop], noise=variance, **kwargs
                )
            )
            start = stop
        return TorchModelListGP(*models).eval()

    def fantasize(
        self,
        X: Any,
        sampler: Any,
        observation_noise: Any = None,
        evaluation_mask: Any = None,
        **kwargs: Any,
    ) -> Any:
        samplers = getattr(sampler, "samplers", None)
        if evaluation_mask is not None and samplers is None:
            raise ValueError("Decoupled fantasization requires a ListSampler.")
        models = []
        start = 0
        for i, model in enumerate(self.models):
            stop = start + model.num_outputs
            inputs = X
            if evaluation_mask is not None:
                if model.num_outputs != 1:
                    raise UnsupportedError(
                        "Decoupled fantasies require single-output component models."
                    )
                inputs = X[..., evaluation_mask[:, i], :]
            member_sampler = sampler if samplers is None else samplers[i]
            noise = None if observation_noise is None else observation_noise[..., start:stop]
            models.append(
                model.fantasize(inputs, member_sampler, observation_noise=noise, **kwargs)
            )
            start = stop
        return TorchModelListGP(*models).eval()

    def _sync_torch_parameters(self) -> Any:
        return None
