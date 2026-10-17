"""Marginal log likelihoods (``gpytorch.mlls``)."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import broadcast_shapes, broadcast_to, from_flat, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.likelihoods import Likelihood
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP


def _forward_pass(model: Any) -> None:
    """The side effects of a forward pass of the model (the in-place
    renormalization of LCEMGP's embeddings)."""
    hook = getattr(model, "_forward_pass", None)
    if hook is not None:
        hook()


class MarginalLogLikelihood:
    """Base class of the marginal log likelihoods."""

    training: bool = True
    #: Summary of the last ``fit_gpytorch_mll`` call (scipy-style keys).
    _fit_result: Any = None

    def __init__(self, likelihood: Likelihood, model: Any) -> None:
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
    ``mll(model(*model.train_inputs), model.train_targets)``): a 0-dim array
    for a single-output model, and one value per output and batch element
    (shape ``_aug_batch_shape``) otherwise.
    """

    def __init__(
        self, likelihood: Likelihood, model: SingleTaskGP | MultiTaskGP | KroneckerMultiTaskGP
    ) -> None:
        if not isinstance(model, (SingleTaskGP, MultiTaskGP, KroneckerMultiTaskGP)):
            raise UnsupportedError(
                "ExactMarginalLogLikelihood requires a botorch_rs SingleTaskGP, MultiTaskGP, "
                "KroneckerMultiTaskGP or LCEMGP."
            )
        if likelihood is not model.likelihood:
            raise UnsupportedError(
                "The likelihood must be the model's likelihood (`model.likelihood`)."
            )
        super().__init__(likelihood, model)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The MLL values (shape ``_aug_batch_shape``, of the model's array type)."""
        if args or kwargs:
            raise UnsupportedError(
                "botorch_rs evaluates the MLL on the model's training data: call `mll()` "
                "without arguments."
            )
        _forward_pass(self.model)
        data, shape = self.model._rs.mll()
        return from_flat(data, tuple(shape), self.model._kind)

    forward = __call__

    def value_and_grad(self, raw_parameters: Any = None) -> tuple[Any, list[float]]:
        """The MLL and the gradient of its sum over the elements with respect
        to the raw hyperparameters (in the order of
        ``model.named_parameters()``), at the current or the given flat raw
        hyperparameters. The value is a float for a single-output model and a
        list of values (one per element) otherwise."""
        params = None if raw_parameters is None else to_flat(raw_parameters)[0]
        if params is None:
            _forward_pass(self.model)
        (values, shape), grad = self.model._rs.mll_value_and_grad(params)
        return (values[0] if not shape else list(values)), grad


class SumMarginalLogLikelihood(MarginalLogLikelihood):
    r"""The sum of the exact marginal log likelihoods of the sub-models of a
    ``ModelListGP``, divided by the number of sub-models (GPyTorch's
    ``SumMarginalLogLikelihood``; the per-model values are broadcast against
    each other as in GPyTorch).
    """

    def __init__(
        self,
        likelihood: Likelihood,
        model: ModelListGP,
        mll_cls: type = ExactMarginalLogLikelihood,
    ) -> None:
        if not isinstance(model, ModelListGP):
            raise UnsupportedError("SumMarginalLogLikelihood requires a botorch_rs ModelListGP.")
        if likelihood is not model.likelihood:
            raise UnsupportedError(
                "The likelihood must be the model's likelihood (`model.likelihood`)."
            )
        if mll_cls is not ExactMarginalLogLikelihood:
            raise UnsupportedError("Only ExactMarginalLogLikelihood sub-MLLs are supported.")
        super().__init__(likelihood, model)
        self.mlls = [mll_cls(m.likelihood, m) for m in model.models]

    def train(self, mode: bool = True) -> SumMarginalLogLikelihood:
        self.training = mode
        for mll in self.mlls:
            mll.train(mode)
        self.model.training = mode
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """``sum_i mll_i() / k`` on the training data of the sub-models."""
        if args or kwargs:
            raise UnsupportedError(
                "botorch_rs evaluates the MLL on the models' training data: call `mll()` "
                "without arguments."
            )
        for mll in self.mlls:
            _forward_pass(mll.model)
        parts = [mll.model._rs.mll() for mll in self.mlls]
        shape = broadcast_shapes(*(tuple(s) for _, s in parts))
        if shape is None:
            raise RuntimeError(
                "The MLLs of the sub-models have shapes that cannot be broadcast: "
                f"{[tuple(s) for _, s in parts]}."
            )
        total = [0.0] * math.prod(shape)
        for data, s in parts:
            for i, v in enumerate(broadcast_to(data, tuple(s), shape)):
                total[i] += v
        k = len(self.mlls)
        return from_flat([v / k for v in total], shape, self.model._kind)

    forward = __call__


__all__ = ["ExactMarginalLogLikelihood", "MarginalLogLikelihood", "SumMarginalLogLikelihood"]
