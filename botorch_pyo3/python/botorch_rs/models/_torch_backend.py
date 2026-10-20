"""Device-resident exact GPs using the Rust Torch-operation dispatcher."""

from __future__ import annotations

import copy
from typing import Any

import torch

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior, independent_covariance
from botorch_rs.utils.types import DEFAULT


class TorchSingleTaskGP(torch.nn.Module, SingleTaskGP):
    """An exact GP with ordinary Torch autograd through every Tensor operation.

    Rust dispatches kernels and Gaussian conditioning to the installed Torch
    runtime. No input, intermediate moment, or gradient is copied to a CPU
    numerical backend. ``SingleTaskGP(..., backend="torch")`` selects this
    implementation; CUDA inputs select it automatically.
    """

    _tensor_backend = "torch"

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        likelihood: Any = None,
        covar_module: Any = None,
        mean_module: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        from botorch_rs.models._torch_backend_modules import (
            build_input_transform,
            build_kernel,
            build_likelihood,
            build_mean,
            build_outcome_transform,
        )

        torch.nn.Module.__init__(self)
        self._validate_data(train_X, train_Y, train_Yvar)
        self._output_count = train_Y.shape[-1]
        self.register_buffer("_train_X", train_X)
        self.register_buffer("_train_Y", train_Y)
        self.register_buffer("_train_Yvar", train_Yvar)
        self.register_buffer("_noise_override", None)
        # Sources for fitted transform statistics stay unchanged when adding
        # fantasies. Recompute their graph on every evaluation, allowing
        # independent backward calls without retaining a stale fit graph.
        self.register_buffer("_transform_fit_X", train_X)
        self.register_buffer("_transform_fit_Y", train_Y)
        self.input_transform = build_input_transform(input_transform, train_X.shape[-1], train_X)
        self.outcome_transform = build_outcome_transform(
            outcome_transform, self.num_outputs, train_Y
        )
        _, _, noise = self._training_values()
        self.covar_module = build_kernel(
            covar_module, train_X.shape[-1], self._aug_batch_shape, train_X
        )
        self.mean_module = build_mean(mean_module, self._aug_batch_shape, train_X)
        self.likelihood = build_likelihood(likelihood, noise, self._aug_batch_shape, train_X)

    @staticmethod
    def _validate_data(X: Any, Y: Any, Yvar: Any = None) -> None:
        if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
            raise TypeError("The Torch backend requires Tensor training inputs and targets.")
        if X.ndim < 2 or Y.ndim != X.ndim or X.shape[:-1] != Y.shape[:-1]:
            raise ValueError("Expected train_X (..., n, d) and train_Y (..., n, m).")
        if not X.shape[-2] or not X.shape[-1] or not Y.shape[-1]:
            raise ValueError("Training data dimensions must be nonempty.")
        if X.dtype not in (torch.float32, torch.float64) or Y.dtype != X.dtype:
            raise TypeError("Training inputs and targets must share float32 or float64 dtype.")
        if X.device != Y.device:
            raise ValueError("Training inputs and targets must share a device.")
        if Yvar is not None:
            if (
                not isinstance(Yvar, torch.Tensor)
                or Yvar.device != X.device
                or Yvar.dtype != X.dtype
            ):
                raise TypeError("train_Yvar must be a Tensor with the training dtype and device.")
            if torch.broadcast_shapes(Y.shape, Yvar.shape) != Y.shape:
                raise ValueError("train_Yvar must broadcast to train_Y.shape.")
            if bool((Yvar < 0).any()):
                raise ValueError("Observation variances must be nonnegative.")
        if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(Y).all()):
            raise ValueError("Training inputs and targets must be finite.")
        if Yvar is not None and not bool(torch.isfinite(Yvar).all()):
            raise ValueError("Observation variances must be finite.")

    @property
    def num_outputs(self) -> int:
        return self._output_count

    @property
    def _num_outputs(self) -> int:
        return self.num_outputs

    @property
    def batch_shape(self) -> Any:
        return self._train_Y.shape[:-2]

    @property
    def _input_batch_shape(self) -> Any:
        return self.batch_shape

    @property
    def _aug_batch_shape(self) -> Any:
        return self.batch_shape + ((self.num_outputs,) if self.num_outputs > 1 else ())

    def _augment_X(self, X: Any) -> Any:
        return (
            X.unsqueeze(-3).expand(*X.shape[:-2], self.num_outputs, *X.shape[-2:])
            if self.num_outputs > 1
            else X
        )

    def _augment_Y(self, Y: Any) -> Any:
        return Y.transpose(-1, -2) if self.num_outputs > 1 else Y.squeeze(-1)

    def _fit_transform_graphs(self) -> None:
        for transform, values in (
            (self.input_transform, self._transform_fit_X),
            (self.outcome_transform, self._transform_fit_Y),
        ):
            if transform is None:
                continue
            if torch.is_grad_enabled() and values.requires_grad:
                # Only port-owned fitted transforms have this marker. An
                # arbitrary user module controls its own fitting semantics.
                if getattr(transform, "_tensor_backend", None) == "torch":
                    transform.train()
                    transform(values)
            transform.eval()

    def _training_values(self) -> tuple[Any, Any, Any]:
        self._fit_transform_graphs()
        X = self._train_X
        if self.input_transform is not None:
            X = self.input_transform(X)
        Y, noise = self._train_Y, self._train_Yvar
        if self.outcome_transform is not None:
            Y, noise = self.outcome_transform(Y, noise, X=self._train_X)
        return (
            self._augment_X(X),
            self._augment_Y(Y),
            None if noise is None else self._augment_Y(noise.expand_as(Y)),
        )

    @property
    def train_inputs(self) -> tuple[Any]:
        return (self._training_values()[0],)

    @property
    def train_targets(self) -> Any:
        return self._training_values()[1]

    def transform_inputs(self, X: Any, input_transform: Any = None) -> Any:
        transform = self.input_transform if input_transform is None else input_transform
        return X if transform is None else transform(X)

    def _training_noise(self, Y: Any, fixed_noise: Any) -> Any:
        noise = fixed_noise if fixed_noise is not None else self._noise_for(Y.shape[-1])
        if self._noise_override is not None:
            noise = torch.where(self._noise_override.isnan(), noise, self._noise_override)
        return noise

    def _noise_for(self, count: int) -> Any:
        method = getattr(self.likelihood, "noise_for", None)
        if method is not None:
            return method(count)
        noise = self.likelihood.noise
        return noise.expand(*noise.shape[:-1], count)

    def _covariance(self, X1: Any, X2: Any) -> Any:
        value = self.covar_module(X1, X2)
        # GPyTorch kernel modules and user kernels may expose lazy covariance
        # objects. Densifying them retains their Torch graph and device.
        return value if isinstance(value, torch.Tensor) else value.to_dense()

    def posterior(
        self,
        X: Any,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported posterior arguments: {sorted(kwargs)}.")
        if not isinstance(X, torch.Tensor) or X.ndim < 2 or X.shape[-1] != self._train_X.shape[-1]:
            raise ValueError("X must be a Tensor with shape (..., q, d).")
        if X.dtype != self._train_X.dtype or X.device != self._train_X.device:
            raise ValueError("X must share the model's dtype and device; use model.to(X).")
        self.eval()
        train_X, Y, fixed_noise = self._training_values()
        query_X = self._augment_X(self.transform_inputs(X))
        train_noise = self._training_noise(Y, fixed_noise)
        query_noise = None
        if isinstance(observation_noise, torch.Tensor):
            query_noise = self._augment_Y(observation_noise)
        elif observation_noise is True:
            query_noise = (
                fixed_noise.mean(-1, keepdim=True)
                if fixed_noise is not None
                else self._noise_for(query_X.shape[-2])
            )
        elif observation_noise is not False:
            raise TypeError("observation_noise must be bool or a Tensor.")
        mean, covariance = _rs.tensor_exact_gp_posterior(
            Y,
            self._covariance(train_X, train_X),
            self._covariance(train_X, query_X),
            self._covariance(query_X, query_X),
            self.mean_module(train_X),
            self.mean_module(query_X),
            train_noise,
            observation_noise=query_noise,
        )
        if self.num_outputs == 1:
            mean, blocks = mean.unsqueeze(-1), covariance.unsqueeze(-3)
        else:
            mean, blocks = mean.transpose(-1, -2), covariance
        if self.outcome_transform is not None:
            mean, blocks = self.outcome_transform.untransform_moments(mean, blocks)
        if output_indices is not None:
            indices = torch.as_tensor(output_indices, device=mean.device, dtype=torch.long)
            if indices.ndim != 1 or not indices.numel():
                raise ValueError("output_indices must be a nonempty one-dimensional sequence.")
            if bool(((indices < -self.num_outputs) | (indices >= self.num_outputs)).any()):
                raise IndexError("output_indices contains an out-of-range output.")
            indices = indices.remainder(self.num_outputs)
            mean, blocks = mean.index_select(-1, indices), blocks.index_select(-3, indices)
        posterior = TorchGPyTorchPosterior(mean, independent_covariance(blocks))
        return posterior if posterior_transform is None else posterior_transform(posterior, X=X)

    def forward(self, X: Any) -> Any:
        if not self.training:
            return self.posterior(X).distribution
        # As in ExactGP, training __call__ returns the latent prior on the
        # already transformed, output-augmented inputs in train_inputs.
        return torch.distributions.MultivariateNormal(
            self.mean_module(X), covariance_matrix=self._covariance(X, X)
        )

    def _log_prior(self) -> Any:
        result = self._train_X.new_zeros(())
        seen: set[int] = set()
        for module in (self.likelihood, self.mean_module, self.covar_module, self.input_transform):
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                method = getattr(module, "log_prior", None)
                if method is not None:
                    result = result + method()
        return result

    def _exact_mll(self, output: Any = None, target: Any = None) -> Any:
        if output is None:
            X, Y, fixed_noise = self._training_values()
            mean, covariance = self.mean_module(X), self._covariance(X, X)
        else:
            _, _, fixed_noise = self._training_values()
            Y = self.train_targets if target is None else target
            mean, covariance = output.mean, output.covariance_matrix
        value = _rs.tensor_exact_mll(
            Y, covariance, mean, self._training_noise(Y, fixed_noise), normalize=False
        )
        return (value + self._log_prior()) / Y.shape[-1]

    def set_train_data(self, inputs: Any = None, targets: Any = None, strict: bool = True) -> None:
        X = self._train_X if inputs is None else inputs[0] if isinstance(inputs, tuple) else inputs
        Y = self._train_Y if targets is None else targets
        if Y.ndim == X.ndim - 1 and self.num_outputs == 1:
            Y = Y.unsqueeze(-1)
        self._validate_data(X, Y, self._train_Yvar)
        if Y.shape[-1] != self.num_outputs:
            raise ValueError("set_train_data cannot change the number of outputs.")
        if strict and (X.shape != self._train_X.shape or Y.shape != self._train_Y.shape):
            raise RuntimeError("Cannot modify the training data shape with strict=True.")
        self._train_X, self._train_Y = X, Y
        self._transform_fit_X, self._transform_fit_Y = X, Y

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported conditioning arguments: {sorted(kwargs)}.")
        if X.shape[-2] != Y.shape[-2] or Y.shape[-1] != self.num_outputs:
            raise ValueError("Conditioning X and Y dimensions must match the model.")
        batch = torch.broadcast_shapes(self.batch_shape, X.shape[:-2], Y.shape[:-2])
        model = copy.copy(self)
        model._buffers = self._buffers.copy()
        model._modules = self._modules.copy()
        model._train_X = torch.cat(
            (
                self._train_X.expand(*batch, *self._train_X.shape[-2:]),
                X.expand(*batch, *X.shape[-2:]),
            ),
            dim=-2,
        )
        model._train_Y = torch.cat(
            (
                self._train_Y.expand(*batch, *self._train_Y.shape[-2:]),
                Y.expand(*batch, *Y.shape[-2:]),
            ),
            dim=-2,
        )
        if self._train_Yvar is not None:
            if noise is None:
                noise = self._train_Yvar.mean(-2, keepdim=True).expand_as(Y)
            elif self.outcome_transform is not None:
                # Conditioning noise follows BoTorch's transformed-space API.
                noise = self.outcome_transform.untransform(torch.zeros_like(Y), noise)[1]
            model._train_Yvar = torch.cat(
                (
                    self._train_Yvar.expand(*batch, *self._train_Y.shape[-2:]),
                    noise.expand(*batch, *Y.shape[-2:]),
                ),
                dim=-2,
            )
        elif noise is not None or self._noise_override is not None:
            previous = self._noise_override
            if previous is None:
                previous = torch.full_like(self._augment_Y(self._train_Y), torch.nan)
            extra = (
                torch.full_like(self._augment_Y(Y), torch.nan)
                if noise is None
                else self._augment_Y(noise)
            )
            augmented_batch = batch + ((self.num_outputs,) if self.num_outputs > 1 else ())
            model._noise_override = torch.cat(
                (
                    previous.expand(*augmented_batch, previous.shape[-1]),
                    extra.expand(*augmented_batch, Y.shape[-2]),
                ),
                dim=-1,
            )
        return model.eval()

    def fantasize(self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any) -> Any:
        propagate = kwargs.pop("propagate_grads", False)
        if kwargs:
            raise UnsupportedError(f"Unsupported fantasy arguments: {sorted(kwargs)}.")
        if isinstance(observation_noise, bool):
            raise TypeError("fantasize observation_noise must be None or a Tensor.")
        posterior = self.posterior(
            X, observation_noise=True if observation_noise is None else observation_noise
        )
        Y = sampler(posterior)
        if not propagate:
            Y = Y.detach()
        return self.condition_on_observations(X, Y, noise=observation_noise)

    def _sync_torch_parameters(self) -> Any:
        # Parameters already are the tensors consumed by the inference graph.
        return None

    def initialize(self, **kwargs: Any) -> Any:
        for name, value in kwargs.items():
            path, _, attribute = name.rpartition(".")
            module = self.get_submodule(path) if path else self
            current = getattr(module, attribute)
            if isinstance(current, torch.nn.Parameter):
                with torch.no_grad():
                    current.copy_(
                        torch.as_tensor(value, dtype=current.dtype, device=current.device)
                    )
            else:
                setattr(module, attribute, value)
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        new = object.__new__(type(self))
        memo[id(self)] = new
        for name, value in self.__dict__.items():
            if name == "_buffers":
                # Keep graph connectivity for training tensors and fantasies.
                new.__dict__[name] = value.copy()
            else:
                new.__dict__[name] = copy.deepcopy(value, memo)
        return new
