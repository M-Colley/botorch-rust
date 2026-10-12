r"""Risk measures (``botorch.acquisition.risk_measures``; Cakmak et al.,
*Bayesian Optimization of Risk Measures*, NeurIPS 2020): MC objectives that
map the posterior samples of the ``n_w`` perturbations of every candidate
(``sample_shape x batch_shape x (q * n_w) x m``, from a model with an
``InputPerturbation``) to a risk measure of the candidate
(``sample_shape x batch_shape x q``).

``CVaR``, ``VaR``, ``WorstCase`` and ``Expectation`` are evaluated in Rust,
inside the acquisition functions and when called directly (with exact
gradients through torch's autograd for torch tensors). As in BoTorch they
work on the lower tail (larger values are better); multi-output samples must
be scalarized by a ``preprocessing_function`` (an MC objective of
botorch_rs or BoTorch, or a callable on torch tensors).

Example:
    >>> model = SingleTaskGP(train_X, train_Y, input_transform=InputPerturbation(w_set))
    >>> objective = CVaR(alpha=0.8, n_w=w_set.shape[0])
    >>> acqf = qLogNoisyExpectedImprovement(model, train_X, objective=objective)
"""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition import _native_objective
from botorch_rs.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    objective_arg,
)
from botorch_rs.exceptions.errors import BotorchError


def _view_n_w(samples: Any, n_w: int) -> Any:
    return samples.view(*samples.shape[:-1], -1, n_w)


class RiskMeasureMCObjective(MCAcquisitionObjective):
    r"""Base class of the risk measures: objectives transforming the joint
    posterior samples of the ``n_w`` points of every candidate into samples
    of a risk measure (the first ``n_w`` points belong to the first
    candidate, and so on).

    Subclasses implement ``forward(samples, X=None)`` on torch tensors (see
    ``_prepare_samples``); the risk measures of this module are computed in
    Rust.
    """

    #: The extension object of a risk measure computed in Rust.
    _rs: Any = None

    def __init__(
        self,
        n_w: int,
        preprocessing_function: Callable[[Any], Any] | None = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            n_w: The size of the ``w_set`` to calculate the risk measure over.
            preprocessing_function: A preprocessing function to apply to the
                samples before computing the risk measure (e.g. to scalarize
                multi-output samples, or to apply feasibility weighting).
                Given a ``batch x m``-dim tensor of samples, it returns a
                ``batch``-dim tensor.
        """
        self.n_w = int(n_w)
        if preprocessing_function is None:
            preprocessing_function = self._default_preprocessing()
        self.preprocessing_function = preprocessing_function

    def _default_preprocessing(self) -> Any:
        return IdentityMCObjective()

    def _preprocessing_spec(self) -> tuple[Any, ...] | None:
        """The preprocessing function for the extension (``None``: the
        samples of a single-output model)."""
        pre = self.preprocessing_function
        if type(pre) is IdentityMCObjective:
            return None
        if isinstance(pre, MCAcquisitionObjective) or (
            "MCAcquisitionObjective" in _class_names(pre)
        ):
            spec = objective_arg(pre)
            return None if spec == ("identity",) else spec
        if not callable(pre):
            raise BotorchError("`preprocessing_function` must be callable.")
        return ("callable", pre)

    def _prepare_samples(self, samples: Any) -> Any:
        r"""BoTorch's ``_prepare_samples`` (torch tensors): the preprocessed
        samples as ``sample_shape x batch_shape x q x n_w``."""
        if samples.shape[-1] > 1 and isinstance(self.preprocessing_function, IdentityMCObjective):
            raise RuntimeError(
                "Multi-output samples should be scalarized using a `preprocessing_function`."
            )
        return _view_n_w(self.preprocessing_function(samples), self.n_w)

    def forward(self, samples: Any, X: Any = None) -> Any:
        """The risk measure of the samples (``sample_shape x batch_shape x
        q``); ``X`` is ignored."""
        if self._rs is None:  # pragma: no cover - abstract
            raise NotImplementedError
        return _native_objective.evaluate(self._rs, samples)

    def _spec(self) -> tuple[Any, ...]:
        """Computed in Rust unless a subclass overrides ``forward``."""
        if self._rs is not None and type(self).forward in _NATIVE_FORWARDS:
            return ("native", self._rs)
        return ("callable", self)


def _class_names(obj: Any) -> list[str]:
    return [c.__name__ for c in type(obj).__mro__]


class CVaR(RiskMeasureMCObjective):
    r"""The Conditional Value-at-Risk: the mean of the ``n_w - alpha_idx``
    smallest values of every candidate, ``alpha_idx = ceil(n_w * alpha) - 1``
    (the expectation of the worst outcomes with total probability
    ``1 - alpha``).

    Note: With a discrete ``w_set`` the VaR and CVaR are (possibly biased)
    Monte-Carlo approximations of the true risk measures.
    """

    def __init__(
        self,
        alpha: float,
        n_w: int,
        preprocessing_function: Callable[[Any], Any] | None = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            alpha: The risk level, float in ``(0.0, 1.0]``.
            n_w: The size of the ``w_set`` to calculate the risk measure over.
            preprocessing_function: A preprocessing function to apply to the
                samples before computing the risk measure.
        """
        super().__init__(n_w=n_w, preprocessing_function=preprocessing_function)
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0.0, 1.0]")
        self.alpha = alpha
        self.alpha_idx = ceil(n_w * alpha) - 1
        self._rs = self._build()

    def _build(self) -> Any:
        return _rs.RiskMeasure.cvar(float(self.alpha), self.n_w, self._preprocessing_spec())


class VaR(CVaR):
    r"""The Value-at-Risk: the ``1 - alpha_idx / n_w`` quantile
    (``interpolation="lower"``) of the values of every candidate, the
    smallest reward after excluding the worst outcomes with total
    probability ``1 - alpha``."""

    def __init__(
        self,
        alpha: float,
        n_w: int,
        preprocessing_function: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(n_w=n_w, alpha=alpha, preprocessing_function=preprocessing_function)
        self._q = 1 - self.alpha_idx / n_w

    def _build(self) -> Any:
        return _rs.RiskMeasure.var(float(self.alpha), self.n_w, self._preprocessing_spec())


class WorstCase(RiskMeasureMCObjective):
    r"""The worst-case risk measure: the smallest value of every candidate."""

    def __init__(
        self,
        n_w: int,
        preprocessing_function: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(n_w=n_w, preprocessing_function=preprocessing_function)
        self._rs = _rs.RiskMeasure.worst_case(self.n_w, self._preprocessing_spec())


class Expectation(RiskMeasureMCObjective):
    r"""The expectation risk measure: the mean value of every candidate (the
    expected outcome under the perturbations; not a risk measure in the
    strict sense)."""

    def __init__(
        self,
        n_w: int,
        preprocessing_function: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(n_w=n_w, preprocessing_function=preprocessing_function)
        self._rs = _rs.RiskMeasure.expectation(self.n_w, self._preprocessing_spec())


#: The ``forward`` methods computed in Rust (subclasses that override
#: ``forward`` are evaluated by calling them).
_NATIVE_FORWARDS: set[Any] = {RiskMeasureMCObjective.forward}


def risk_measure_from_botorch(objective: Any) -> RiskMeasureMCObjective | None:
    """The botorch_rs equivalent of one of BoTorch's single-output risk
    measures (``CVaR``, ``VaR``, ``WorstCase``, ``Expectation``, ``MARS``),
    or ``None`` for other objects (including subclasses)."""
    names = [
        c.__name__
        for c in type(objective).__mro__
        if getattr(c, "__module__", "").startswith("botorch.acquisition.")
    ]
    name = type(objective).__name__
    if name not in names:
        return None
    pre = getattr(objective, "preprocessing_function", None)
    if name in ("CVaR", "VaR"):
        cls = CVaR if name == "CVaR" else VaR
        return cls(alpha=float(objective.alpha), n_w=int(objective.n_w), preprocessing_function=pre)
    if name == "WorstCase":
        return WorstCase(n_w=int(objective.n_w), preprocessing_function=pre)
    if name == "Expectation":
        return Expectation(n_w=int(objective.n_w), preprocessing_function=pre)
    if name == "MARS":
        from botorch_rs.acquisition.multi_objective.multi_output_risk_measures import MARS

        ref = objective.ref_point
        return MARS(
            alpha=float(objective.alpha),
            n_w=int(objective.n_w),
            chebyshev_weights=objective.chebyshev_weights,
            baseline_Y=objective.baseline_Y,
            ref_point=None if ref is None else ref,
            preprocessing_function=pre,
        )
    return None


__all__ = [
    "CVaR",
    "Expectation",
    "RiskMeasureMCObjective",
    "VaR",
    "WorstCase",
]
