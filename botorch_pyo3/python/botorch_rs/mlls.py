"""Marginal log likelihoods (``gpytorch.mlls``)."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from typing import Any

from botorch_rs._arrays import broadcast_shapes, broadcast_to, from_flat, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.likelihoods import Likelihood
from botorch_rs.models.empirical_gps.empirical_1d_gp import EmpiricalOneDimensionalGP
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.higher_order_gp import HigherOrderGP
from botorch_rs.models.latent_kronecker_gp import LatentKroneckerGP
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP


def _persistent_parameter(value: Any) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.nn.Parameter)


def _model_parameter_paths(model: Any) -> Iterator[tuple[str, Any, Any]]:
    """Native parameter paths, retaining aliases until final traversal filtering."""
    if isinstance(model, ModelListGP):
        for index, child in enumerate(model.models):
            for name, value, key in _model_parameter_paths(child):
                yield f"models.{index}.{name}", value, key
        for index, child in enumerate(model.models):
            for name, value, key in _model_parameter_paths(child):
                if name.startswith("likelihood."):
                    yield f"likelihood.likelihoods.{index}.{name[11:]}", value, key
        return
    for name, value in model.named_parameters():
        key = ("torch", id(value)) if _persistent_parameter(value) else ("native", id(model), name)
        yield name, value, key


def _named_mll_parameters(
    paths: Any, prefix: str, remove_duplicate: bool
) -> Iterator[tuple[str, Any]]:
    seen = set()
    for name, value, key in paths:
        if remove_duplicate and key in seen:
            continue
        seen.add(key)
        yield f"{prefix}.{name}" if prefix else name, value


def _forward_pass(model: Any) -> None:
    """The side effects of a forward pass of the model (the in-place
    renormalization of LCEMGP's embeddings)."""
    sync = getattr(model, "_sync_torch_parameters", None)
    if sync is not None:
        sync()
    hook = getattr(model, "_forward_pass", None)
    if hook is not None:
        hook()


class MarginalLogLikelihood:
    """Native MLL with parameter traversal and basic Torch gradient controls.

    This is a focused parameter protocol, not a ``torch.nn.Module`` subclass.
    """

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

    def _parameter_paths(self) -> Iterator[tuple[str, Any, Any]]:
        paths = list(_model_parameter_paths(self.model))
        # GPyTorch registers the likelihood before the model. The model also
        # registers that same likelihood; remove_duplicate=False exposes both.
        for name, value, key in paths:
            if isinstance(self.model, ModelListGP):
                _, child_index, child_name = name.split(".", 2)
                if child_name.startswith("likelihood."):
                    yield f"likelihood.likelihoods.{child_index}.{child_name[11:]}", value, key
            elif name.startswith("likelihood."):
                yield name, value, key
        for name, value, key in paths:
            yield f"model.{name}", value, key
        for index, mll in enumerate(getattr(self, "mlls", ())):
            for name, value, key in mll._parameter_paths():
                yield f"mlls.{index}.{name}", value, key

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Any]]:
        """Parameters in likelihood-then-model registration order.

        Native MLLs have no direct parameters, so ``recurse=False`` is empty.
        Array-backed models expose native parameter descriptors; CPU Torch
        exact and variational models expose their persistent leaves.
        """
        if recurse:
            yield from _named_mll_parameters(self._parameter_paths(), prefix, remove_duplicate)

    def parameters(self, recurse: bool = True) -> Iterator[Any]:
        if recurse:
            for _, value in self.named_parameters():
                yield value

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear managed Torch gradients; native array descriptors have none."""
        for parameter in self.parameters():
            if not _persistent_parameter(parameter) or parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                if parameter.grad.grad_fn is not None:
                    parameter.grad.detach_()
                else:
                    parameter.grad.requires_grad_(False)
                parameter.grad.zero_()

    def requires_grad_(self, requires_grad: bool = True) -> MarginalLogLikelihood:
        """Set trainability of persistent CPU Torch leaves in place."""
        parameters = tuple(self.parameters())
        if any(not _persistent_parameter(parameter) for parameter in parameters):
            raise UnsupportedError(
                "requires_grad_ requires persistent CPU Torch model parameters; "
                "native array descriptors do not store trainability flags."
            )
        for parameter in parameters:
            parameter.requires_grad_(requires_grad)
        return self


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
        self,
        likelihood: Likelihood,
        model: SingleTaskGP
        | MultiTaskGP
        | KroneckerMultiTaskGP
        | HigherOrderGP
        | LatentKroneckerGP,
    ) -> None:
        if type(likelihood).__name__ in ("PairwiseProbitLikelihood", "PairwiseLogitLikelihood"):
            # GPyTorch's check of the likelihood (e.g. a PairwiseGP).
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        if not isinstance(
            model,
            (
                SingleTaskGP,
                MultiTaskGP,
                KroneckerMultiTaskGP,
                HigherOrderGP,
                LatentKroneckerGP,
                EmpiricalOneDimensionalGP,
            ),
        ):
            raise UnsupportedError(
                "ExactMarginalLogLikelihood requires a botorch_rs SingleTaskGP, MultiTaskGP, "
                "KroneckerMultiTaskGP, LCEMGP, HigherOrderGP or LatentKroneckerGP."
            )
        if likelihood is not model.likelihood:
            raise UnsupportedError(
                "The likelihood must be the model's likelihood (`model.likelihood`)."
            )
        super().__init__(likelihood, model)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The MLL values (shape ``_aug_batch_shape``, of the model's array type)."""
        if getattr(self.model, "_tensor_backend", None) == "torch":
            if kwargs or len(args) > 2:
                raise TypeError(
                    "Call mll(), or mll(model(*model.train_inputs), model.train_targets)."
                )
            return self.model._exact_mll(*args)
        if args or kwargs:
            raise UnsupportedError(
                "botorch_rs evaluates the MLL on the model's training data: call `mll()` "
                "without arguments."
            )
        _forward_pass(self.model)
        data, shape = self.model._rs.mll()
        from botorch_rs.models._mll_autograd import mll_output

        return mll_output(self.model, data, tuple(shape))

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
        for sub in model.models:
            if type(sub.likelihood).__name__ in (
                "PairwiseProbitLikelihood",
                "PairwiseLogitLikelihood",
            ):
                raise RuntimeError("Likelihood must be Gaussian for exact inference")
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
        if getattr(self.model, "_tensor_backend", None) == "torch":
            if kwargs or len(args) not in (0, 2):
                raise TypeError("Call mll(), or mll(outputs, targets).")
            outputs = (
                [mll() for mll in self.mlls]
                if not args
                else [
                    mll(output, target)
                    for mll, output, target in zip(self.mlls, *args, strict=True)
                ]
            )
            return sum(outputs) / len(outputs)
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
        if self.model._kind.lib == "torch":
            from botorch_rs import _ops
            from botorch_rs.models._mll_autograd import mll_output

            outputs = [
                mll_output(mll.model, data, tuple(s))
                for mll, (data, s) in zip(self.mlls, parts, strict=True)
            ]
            total = outputs[0]
            for output in outputs[1:]:
                total = _ops.apply("add", total, output, kind=self.model._kind)
            return _ops.apply("div", total, len(outputs), kind=self.model._kind)
        total = [0.0] * math.prod(shape)
        for data, s in parts:
            for i, v in enumerate(broadcast_to(data, tuple(s), shape)):
                total[i] += v
        k = len(self.mlls)
        return from_flat([v / k for v in total], shape, self.model._kind)

    forward = __call__


class _ApproximateMarginalLogLikelihood(MarginalLogLikelihood):
    r"""An approximate marginal log likelihood (typically a bound) for
    approximate GP models (GPyTorch's ``_ApproximateMarginalLogLikelihood``).

    Args:
        likelihood: The likelihood of the model (``model.likelihood`` of a
            botorch_rs ``SingleTaskVariationalGP``).
        model: The approximate GP model (``model.model``).
        num_data: The total number of training data points (necessary for
            SGD).
        beta: A multiplicative factor for the KL divergence term. Setting it
            to 1 (default) recovers true variational inference (as derived in
            `Scalable Variational Gaussian Process Classification`_). Setting
            it to anything less than 1 reduces the regularization effect of
            the model (similarly to what was proposed in `the beta-VAE
            paper`_).
        combine_terms: Whether or not to sum the expected NLL with the KL
            and prior terms (default True).
    """

    def __init__(
        self,
        likelihood: Likelihood,
        model: Any,
        num_data: int,
        beta: float = 1.0,
        combine_terms: bool = True,
    ) -> None:
        from botorch_rs.models.approximate_gp import _SingleTaskVariationalGP

        if not isinstance(model, _SingleTaskVariationalGP) or model._outer is None:
            raise UnsupportedError(
                "The variational ELBO requires the `model.model` of a botorch_rs "
                "SingleTaskVariationalGP (or ApproximateGPyTorchModel)."
            )
        if likelihood is not model._outer.likelihood:
            raise UnsupportedError(
                "The likelihood must be the model's likelihood (`model.likelihood`)."
            )
        super().__init__(likelihood, model)
        self.num_data = num_data
        self.beta = beta
        self.combine_terms = combine_terms

    @property
    def _rs(self) -> Any:
        return self.model._outer._rs

    @property
    def _kind(self) -> Any:
        return self.model._outer._kind

    def _elbo_data(self, approximate_dist_f: Any, target: Any) -> tuple[Any, Any]:
        """Validate and prepare the same data for native values and gradients."""
        from botorch_rs.models.approximate_gp import _ApproximateOutput

        outer = self.model._outer
        if approximate_dist_f is None:
            outer._ensure_initialized()
            x = to_flat(self.model.train_inputs[0], "X")
            target = self.model.train_targets if target is None else target
        else:
            if not isinstance(approximate_dist_f, _ApproximateOutput):
                raise UnsupportedError(
                    "botorch_rs evaluates the ELBO on the output of `model.model(X)`."
                )
            if approximate_dist_f._owner is not outer:
                raise UnsupportedError("The output is not one of this model's.")
            if approximate_dist_f._is_prior:
                raise UnsupportedError(
                    "VariationalELBO requires q(f) from `model.model(X)`. Prior outputs "
                    "from `model.model.forward(X)` or `model.model(X, prior=True)` are "
                    "not supported as ELBO inputs."
                )
            x = approximate_dist_f._x
            if target is None:
                raise TypeError("missing the target values")
        m = outer.num_outputs
        ydata, yshape = to_flat(target, "target")
        n = x[1][0]
        allowed = {(n, m)} if m > 1 else {(n,), (n, 1)}
        if tuple(yshape) not in allowed:
            raise RuntimeError(
                f"The target of shape {tuple(yshape)} does not match the output of event "
                f"shape {(n, m) if m > 1 else (n,)}."
            )
        return x, (list(ydata), [n, m])

    def _terms(self, approximate_dist_f: Any, target: Any) -> tuple[float, float, float]:
        x, y = self._elbo_data(approximate_dist_f, target)
        source = self._rs if approximate_dist_f is None else approximate_dist_f._native
        return source.elbo(x, y, float(self.num_data), float(self.beta))  # type: ignore[no-any-return]

    def __call__(self, approximate_dist_f: Any = None, target: Any = None, **kwargs: Any) -> Any:
        r"""The ELBO of the targets ``target`` under ``approximate_dist_f =
        model.model(X)`` (on the model's training data if both are omitted):
        a 0-dim array, or the tuple ``(log_likelihood, kl_divergence,
        log_prior)`` with ``combine_terms=False``."""
        if kwargs:
            raise UnsupportedError(f"Unsupported arguments: {sorted(kwargs)}.")
        x, y = self._elbo_data(approximate_dist_f, target)
        source = self._rs if approximate_dist_f is None else approximate_dist_f._native
        leaves = None if approximate_dist_f is None else approximate_dist_f._parameter_leaves
        terms = source.elbo(x, y, float(self.num_data), float(self.beta))
        from botorch_rs.models._elbo_autograd import elbo_output

        return elbo_output(
            self.model._outer,
            x,
            y,
            float(self.num_data),
            float(self.beta),
            terms,
            self.combine_terms,
            native=source,
            leaves=leaves,
        )

    forward = __call__

    def value_and_grad(self, raw_parameters: Any = None) -> tuple[float, list[float]]:
        """The ELBO on the training data and its gradient with respect to the
        raw parameters (in the order of ``mll.named_parameters()``), at the
        current or the given flat raw parameters (a botorch_rs extension)."""
        self.model._outer._ensure_initialized()
        params = None if raw_parameters is None else list(to_flat(raw_parameters)[0])
        value, grad = self._rs.elbo_value_and_grad(float(self.num_data), float(self.beta), params)
        return value, list(grad)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Any:
        """``(name, raw value)`` of every parameter (the likelihood's first)."""
        if recurse:
            paths = (
                (name, value, ("torch", id(value)) if _persistent_parameter(value) else name)
                for name, value in self.model._outer._named_raw()
            )
            yield from _named_mll_parameters(paths, prefix, remove_duplicate)

    def _param_entries(self) -> list[tuple[str, Any, int, int]]:
        offsets = self.model._outer._offsets()
        return [(n, v, *offsets[n]) for n, v in self.named_parameters()]

    def parameters(self, recurse: bool = True) -> Any:
        for _, value in self.named_parameters(recurse=recurse):
            yield value

    def train(self, mode: bool = True) -> _ApproximateMarginalLogLikelihood:
        self.training = mode
        self.model.train(mode)
        return self


class VariationalELBO(_ApproximateMarginalLogLikelihood):
    r"""
    The variational evidence lower bound (ELBO) (GPyTorch's
    ``VariationalELBO``), used to train approximate GP models:

    .. math::

       \mathcal{L}_\text{ELBO} = \sum_{i=1}^N \mathbb{E}_{q(f_i)} \left[
       \log p(y_i \mid f_i) \right] - \beta \, \text{KL} \left[ q(\mathbf u)
       \Vert p(\mathbf u) \right] (+ \log p(\theta))

    divided by the number of data points (the likelihood term by the number
    of points of the batch, the KL divergence and the log priors by
    ``num_data``).

    Example:
        >>> mll = VariationalELBO(model.likelihood, model.model, num_data=100)
        >>> loss = -mll(model.model(train_x), train_y)
    """


__all__ = [
    "ExactMarginalLogLikelihood",
    "MarginalLogLikelihood",
    "SumMarginalLogLikelihood",
    "VariationalELBO",
]
