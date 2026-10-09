"""Marginal log likelihoods (``gpytorch.mlls``)."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import from_flat, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.likelihoods import Likelihood
from botorch_rs.models.gp_regression import SingleTaskGP


class MarginalLogLikelihood:
    """Base class of the marginal log likelihoods."""

    training: bool = True
    #: Summary of the last ``fit_gpytorch_mll`` call (scipy-style keys).
    _fit_result: dict[str, Any] | None = None

    def __init__(self, likelihood: Likelihood, model: SingleTaskGP) -> None:
        self.likelihood = likelihood
        self.model = model

    def train(self, mode: bool = True) -> MarginalLogLikelihood:
        self.training = mode
        self.model.train(mode)
        return self

    def eval(self) -> MarginalLogLikelihood:
        return self.train(False)


class ExactMarginalLogLikelihood(MarginalLogLikelihood):
    r"""The exact marginal log likelihood of a GP, divided by the number of
    training points, with the log priors of the hyperparameters included (as
    in GPyTorch).

    Calling it without arguments, ``mll()``, evaluates it on the model's
    training data at the current hyperparameters (GPyTorch's
    ``mll(model(*model.train_inputs), model.train_targets)``).
    """

    def __init__(self, likelihood: Likelihood, model: SingleTaskGP) -> None:
        if not isinstance(model, SingleTaskGP):
            raise UnsupportedError("ExactMarginalLogLikelihood requires a botorch_rs SingleTaskGP.")
        if likelihood is not model.likelihood:
            raise UnsupportedError(
                "The likelihood must be the model's likelihood (`model.likelihood`)."
            )
        super().__init__(likelihood, model)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The MLL value (a 0-dim array of the model's array type)."""
        if args or kwargs:
            raise UnsupportedError(
                "botorch_rs evaluates the MLL on the model's training data: call `mll()` "
                "without arguments."
            )
        return from_flat([self.model._rs.mll()], (), self.model._kind)

    forward = __call__

    def value_and_grad(self, raw_parameters: Any = None) -> tuple[float, list[float]]:
        """The MLL and its gradient with respect to the raw hyperparameters (in
        the order of ``model.named_parameters()``), at the current or the
        given flat raw hyperparameters."""
        params = None if raw_parameters is None else to_flat(raw_parameters)[0]
        return self.model._rs.mll_value_and_grad(params)


__all__ = ["ExactMarginalLogLikelihood", "MarginalLogLikelihood"]
