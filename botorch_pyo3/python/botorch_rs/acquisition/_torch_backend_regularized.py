"""Compositional acquisition penalties and prior/proximity weighting on device."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.exceptions.errors import BotorchRsError, UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning


class _Regularized(nn.Module):
    _tensor_backend = "torch"

    def value_and_grad(self, X: Tensor) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            variable = X if X.requires_grad else X.detach().requires_grad_(True)
            value = self(variable)
            gradient = torch.autograd.grad(
                value.sum(), variable, create_graph=True, allow_unused=True
            )[0]
            return value, variable * 0 if gradient is None else gradient


class TensorPenalized(_Regularized):
    def __init__(self, raw_acqf: Any, penalty_func: Any, regularization_parameter: Any) -> None:
        nn.Module.__init__(self)
        self.raw_acqf = raw_acqf
        self.model = raw_acqf.model
        self.penalty_func = penalty_func
        self.regularization_parameter = regularization_parameter
        self._log = getattr(raw_acqf, "_log", False)

    @property
    def X_pending(self) -> Any:
        return self.raw_acqf.X_pending

    def set_X_pending(self, X_pending: Any = None) -> None:
        self.raw_acqf.set_X_pending(X_pending)

    def forward(self, X: Tensor) -> Tensor:
        return self.raw_acqf(X) - self.regularization_parameter * self.penalty_func(X)


class TensorPriorGuided(_Regularized):
    _X_pending: Tensor | None

    def __init__(
        self,
        acq_function: Any,
        prior_module: Any,
        log: bool = False,
        prior_exponent: Any = 1.0,
        X_pending: Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)
        from botorch_rs.acquisition.monte_carlo import SampleReducingMCAcquisitionFunction

        if getattr(acq_function, "X_pending", None) is not None:
            raise BotorchRsError(
                "X_pending is set on acq_function, but should be set on "
                "`PriorGuidedAcquisitionFunction`."
            )
        self.acq_func = acq_function
        self.model = acq_function.model
        self.prior_module = prior_module
        self._log = log
        self._prior_exponent = prior_exponent
        self._is_sample_reducing_af = isinstance(acq_function, SampleReducingMCAcquisitionFunction)
        self.register_buffer("_X_pending", None)
        self.set_X_pending(X_pending)

    @property
    def X_pending(self) -> Tensor | None:
        return self._X_pending

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if X_pending is not None:
            if X_pending.requires_grad:
                warnings.warn(
                    "Pending points are copied without gradients.", BotorchWarning, stacklevel=2
                )
            X_pending = X_pending.detach().clone()
        self._X_pending = X_pending

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim < 2:
            raise ValueError("X must have at least two dimensions.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        if self._X_pending is not None:
            pending = self._X_pending.to(X)
            batch = torch.broadcast_shapes(X.shape[:-2], pending.shape[:-2])
            X = torch.cat(
                (X.expand(*batch, *X.shape[-2:]), pending.expand(*batch, *pending.shape[-2:])),
                dim=-2,
            )
        prior = self.prior_module(X)
        if self._is_sample_reducing_af:
            value = self.acq_func._non_reduced_forward(X)
        else:
            if prior.shape[-1] > 1:
                raise NotImplementedError("q>1 requires a sample-reducing MC acquisition function.")
            value = self.acq_func(X).unsqueeze(-1)
        weighted = (
            value + prior * self._prior_exponent
            if self._log
            else value * prior.pow(self._prior_exponent)
        )
        return (
            self.acq_func._reduce(weighted) if self._is_sample_reducing_af else weighted.squeeze(-1)
        )


def _same_transform(first: Any, second: Any) -> bool:
    if first is second:
        return True
    if type(first) is not type(second) or first is None:
        return False
    if hasattr(first, "state_dict") and hasattr(second, "state_dict"):
        left, right = first.state_dict(), second.state_dict()
        return left.keys() == right.keys() and all(
            torch.equal(left[key], right[key]) for key in left
        )
    return all(
        hasattr(first, key)
        and hasattr(second, key)
        and torch.equal(torch.as_tensor(getattr(first, key)), torch.as_tensor(getattr(second, key)))
        for key in ("offset", "coefficient")
    )


class TensorProximal(_Regularized):
    proximal_weights: Tensor
    beta: Tensor | None

    def __init__(
        self,
        acq_function: Any,
        proximal_weights: Tensor,
        transformed_weighting: bool | None = True,
        beta: Any = None,
    ) -> None:
        nn.Module.__init__(self)
        if getattr(acq_function, "X_pending", None) is not None:
            raise UnsupportedError("Proximal acquisition function requires `X_pending` to be None.")
        self.acq_func = acq_function
        self.model = acq_function.model
        self._log = False
        models = list(getattr(self.model, "models", [self.model]))
        self._source_model = models[0]
        if not hasattr(self._source_model, "train_inputs"):
            raise UnsupportedError("Acquisition function model must have `train_inputs`.")
        train_X = self._source_model.train_inputs[0]
        for model in models[1:]:
            if not torch.equal(model.train_inputs[0], train_X):
                raise UnsupportedError(
                    "Proximal acquisition function requires equal training inputs."
                )
            if not _same_transform(
                getattr(model, "input_transform", None),
                getattr(self._source_model, "input_transform", None),
            ):
                raise UnsupportedError(
                    "Proximal acquisition function requires identical input transforms."
                )
        if self.model.batch_shape and train_X.shape[1] != 1:
            raise UnsupportedError("Proximal acquisition function requires a single batch model.")
        if proximal_weights.ndim != 1 or proximal_weights.shape[0] != train_X.shape[-1]:
            raise ValueError("proximal_weights must have the model's feature dimension.")
        self.register_buffer("proximal_weights", proximal_weights)
        self.transformed_weighting = transformed_weighting
        self.register_buffer("beta", None if beta is None else torch.as_tensor(beta))

    @property
    def X_pending(self) -> None:
        return None

    def set_X_pending(self, X_pending: Any = None) -> None:
        if X_pending is not None:
            raise UnsupportedError("Proximal acquisition functions do not support X_pending.")

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim < 2 or X.shape[-2] != 1:
            raise ValueError("Proximal acquisition functions require q=1.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        value = self.acq_func(X)
        if self.beta is None:
            if bool((value < 0).any()):
                raise RuntimeError(
                    "Set beta to use proximal biasing with negative acquisition values."
                )
        else:
            beta = self.beta.to(value)
            scaled = value * beta
            high = scaled > 20
            value = torch.where(
                high,
                value,
                torch.nn.functional.softplus(scaled.masked_fill(high, 0)) / beta,
            )
        model = self._source_model
        # The port stores raw training inputs; reference eval-mode exact models
        # expose transformed inputs. Transform both raw arguments consistently.
        train_X = getattr(model, "_train_X", model.train_inputs[0])
        last_X = train_X[..., -1:, :]
        transform = getattr(model, "input_transform", None)
        if transform is not None and self.transformed_weighting:
            difference = transform.transform(X) - transform.transform(last_X)
        else:
            difference = X - last_X
        weight = (-0.5 * (difference / self.proximal_weights.to(X)).square().sum(-1)).exp()
        return value * weight.squeeze(-1)
