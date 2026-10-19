r"""
Preference acquisition functions (``botorch.acquisition.preference``). This
includes: Analytical EUBO acquisition function as introduced in
[Lin2022preference]_ and its MC-based generalization qEUBO as proposed in
[Astudillo2023qeubo]_, and the Bayesian active learning by disagreement of
[Houlsby2011bald]_ for pairs.

.. [Astudillo2023qeubo]
    Astudillo, R., Lin, Z.J., Bakshy, E. and Frazier, P.I. qEUBO: A Decision-Theoretic
    Acquisition Function for Preferential Bayesian Optimization. International
    Conference on Artificial Intelligence and Statistics (AISTATS), 2023.

.. [Lin2022preference]
    Lin, Z.J., Astudillo, R., Frazier, P.I. and Bakshy, E. Preference Exploration
    for Efficient Bayesian Optimization with Multiple Outcomes. International
    Conference on Artificial Intelligence and Statistics (AISTATS), 2022.

.. [Houlsby2011bald]
    Houlsby, N., Huszár, F., Ghahramani, Z. and Lengyel, M.
    Bayesian Active Learning for Gaussian Process Classification.
    NIPS Workshop on Bayesian optimization, experimental design and bandits:
    Theory and applications, 2011.

Values and exact gradients are computed in Rust (also through the outcome
model, which must be a deterministic botorch_rs model such as
``PosteriorMeanModel``, ``FixedSingleSampleModel`` or
``GenericDeterministicModel``).
"""

from __future__ import annotations

import importlib
import math
import random
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.acquisition.monte_carlo import MCAcquisitionFunction
from botorch_rs.exceptions.errors import UnsupportedError

SIGMA_JITTER = 1e-8


def _outcome_handle(outcome_model: Any) -> Any:
    if outcome_model is None:
        return None
    handle = getattr(outcome_model, "_acquisition_model", None)
    if handle is None:
        raise UnsupportedError(
            "`outcome_model` must be a deterministic botorch_rs model (e.g. a "
            "PosteriorMeanModel, FixedSingleSampleModel or GenericDeterministicModel)."
        )
    return handle(None)


def _require_data(pref_model: Any) -> None:
    """BoTorch reads ``pref_model.datapoints.dtype`` (an AttributeError for a
    ``PairwiseGP`` without data)."""
    if getattr(pref_model, "datapoints", 0) is None:
        # BoTorch: a PairwiseGP without data has no `num_outputs` yet.
        raise AttributeError(f"'{type(pref_model).__name__}' object has no attribute 'num_outputs'")


class AnalyticExpectedUtilityOfBestOption(AcquisitionFunction):
    r"""Analytic Preferential Expected Utility of Best Options, i.e., Analytical EUBO"""

    def __init__(
        self,
        pref_model: Any,
        outcome_model: Any = None,
        previous_winner: Any = None,
    ) -> None:
        r"""Analytic implementation of Expected Utility of the Best Option under the
        Laplace model (assumes a PairwiseGP is used as the preference model) as
        proposed in [Lin2022preference]_.

        Args:
            pref_model: The preference model that maps the outcomes (i.e., Y) to
                scalar-valued utility.
            outcome_model: A deterministic model that maps parameters (i.e., X) to
                outcomes (i.e., Y). The outcome model f defines the search space of
                Y = f(X). If model is None, we are directly calculating EUBO on
                the parameter space. When used with ``OneSamplePosteriorDrawModel``,
                we are obtaining EUBO-zeta as described in [Lin2022preference]_.
            previous_winner: Tensor representing the previous winner in the Y space.
        """
        super().__init__(pref_model)
        _require_data(pref_model)
        if pref_model.num_outputs != 1:
            raise UnsupportedError(
                "Must specify a posterior transform when using a multi-output model."
            )
        self.outcome_model = outcome_model
        self.previous_winner = previous_winner
        winner = None
        if previous_winner is not None:
            data, shape = to_flat(previous_winner, "previous_winner")
            if len(shape) > 2 or (len(shape) == 2 and shape[0] != 1):
                raise UnsupportedError(
                    "`previous_winner` must be a single point (`m` or `1 x m` values)."
                )
            winner = [float(v) for v in data]
        self.posterior_transform = None
        self._rs = _rs.analytic_eubo(
            self._model_handle(None), _outcome_handle(outcome_model), winner
        )


class qExpectedUtilityOfBestOption(MCAcquisitionFunction):
    r"""MC-based Expected Utility of Best Option (qEUBO)

    This computes qEUBO by
    (1) sampling the joint posterior over q points
    (2) evaluating the maximum objective value across the q points
    (3) averaging over the samples

    ``qEUBO(X) = E[max Y], Y ~ f(X), where X = (x_1,...,x_q)``
    """

    def __init__(
        self,
        pref_model: Any,
        outcome_model: Any = None,
        sampler: Any = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
    ) -> None:
        r"""MC-based Expected Utility of Best Option (qEUBO) as proposed
        in [Astudillo2023qeubo]_.

        Args:
            pref_model: The preference model that maps the outcomes (i.e., Y) to
                scalar-valued utility.
            outcome_model: A deterministic model that maps parameters (i.e., X) to
                outcomes (i.e., Y). The outcome model f defines the search space of
                Y = f(X). If model is None, we are directly calculating qEUBO on
                the parameter space.
            sampler: The sampler used to draw base samples. See
                ``MCAcquisitionFunction`` more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to ``IdentityMCObjective()``.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set
                to have no gradient.
        """
        self.outcome_model = outcome_model
        self._outcome = _outcome_handle(outcome_model)
        super().__init__(
            model=pref_model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self._build("qSimpleRegret")

    def _model_handle(self, posterior_transform: Any = None) -> Any:
        """The preference model seen through the outcome model."""
        return _rs.compose_models(self.model._acquisition_model(posterior_transform), self._outcome)


class _BaldDraws:
    """The extension object of a BALD evaluation with fixed draws (for the
    torch autograd bridge)."""

    def __init__(self, bald: Any, draws: Any) -> None:
        self._bald = bald
        self._draws = draws

    def forward_with_grad(self, flat: Any) -> Any:
        return self._bald.forward_with_grad(flat, self._draws)


def _standard_normal(shape: tuple[int, ...], X: Any) -> tuple[list[float], tuple[int, ...]]:
    """Standard normal draws of ``shape`` from the generator of ``X``'s array
    library (BoTorch's ``Normal.rsample`` draws ``torch.empty(shape).normal_()``)."""
    n = math.prod(shape)
    if is_torch_tensor(X):
        torch = importlib.import_module("torch")
        return torch.empty(shape, dtype=torch.float64).normal_().reshape(-1).tolist(), shape
    if kind_of(X).lib == "numpy":
        np = importlib.import_module("numpy")
        return [float(v) for v in np.random.standard_normal(shape).reshape(-1)], shape
    return [random.gauss(0.0, 1.0) for _ in range(n)], shape


class PairwiseBayesianActiveLearningByDisagreement(AcquisitionFunction):
    r"""MC Bayesian Active Learning by Disagreement"""

    def __init__(
        self,
        pref_model: Any,
        outcome_model: Any = None,
        num_samples: int | None = 1024,
        std_noise: float | None = 0.0,
    ) -> None:
        """
        Monte Carlo implementation of Bayesian Active Learning by Disagreement (BALD)
        proposed in [Houlsby2011bald]_.

        Every evaluation draws ``num_samples`` standard normal values per
        candidate pair from the generator of the input's array library (as
        BoTorch draws them from torch's), so that values are random; within
        ``optimize_acqf``, the draws come from botorch_rs's own generator.

        Args:
            pref_model: The preference model that maps the outcomes (i.e., Y) to
                scalar-valued utility.
            outcome_model: A deterministic model that maps parameters (i.e., X) to
                outcomes (i.e., Y). The outcome model f defines the search space of
                Y = f(X). If model is None, we are directly calculating BALD on
                the parameter space.
            num_samples: number of samples to approximate the conditional_entropy.
            std_noise: Additional observational noise to include. Defaults to 0.
        """
        super().__init__(pref_model)
        if pref_model.num_outputs != 1:
            raise UnsupportedError(
                "Must specify an objective or a posterior transform when using "
                "a multi-output model."
            )
        self.outcome_model = outcome_model
        self.num_samples = num_samples
        self.std_noise = std_noise
        self._bald = _rs.PairwiseBALD(
            self._model_handle(None),
            _outcome_handle(outcome_model),
            int(num_samples if num_samples is not None else 1024),
            float(std_noise if std_noise is not None else 0.0),
            random.randint(0, 2**31 - 1),
        )
        self._rs = self._bald.acquisition_function()

    def _prepare(self, X: Any) -> tuple[Any, Any, tuple[int, ...]]:
        kind = kind_of(X)
        flat = to_flat(X, "X")
        shape = tuple(flat[1])
        if len(shape) < 2:
            raise ValueError(
                f"X is {len(shape)}-dim but must be at least 2-dim (`batch_shape x q x d`)."
            )
        if shape[-2] != 2:
            raise AssertionError(
                "Expected X to be `batch_shape x q=2 x d`, but got X with shape "
                f"{shape if kind.lib != 'torch' else X.shape}."
            )
        shape3 = (1, *shape) if len(shape) == 2 else shape
        return (flat[0], shape3), kind, shape3[:-2]

    def forward(self, X: Any) -> Any:
        r"""Evaluate MC BALD on the candidate set ``X``.

        Args:
            X: A ``batch_shape x 2 x d``-dim Tensor of t-batches with ``q=2``
                ``d``-dim design points each.

        Returns:
            A ``batch_shape'``-dim Tensor of MC BALD values at the given
            design points pair ``X``.
        """
        flat, kind, batch = self._prepare(X)
        draws = _standard_normal((self._bald.num_samples, *batch), X)
        if is_torch_tensor(X) and X.dim() >= 2:
            from botorch_rs import _torch_autograd

            if _torch_autograd.requires_grad(X):
                X3 = X if X.dim() > 2 else X.unsqueeze(0)
                return _torch_autograd.evaluate(_BaldDraws(self._bald, draws), X3)
        values, vshape = self._bald.forward(flat, draws)
        return from_flat(values, vshape, kind)

    __call__ = forward

    def value_and_grad(self, X: Any) -> tuple[Any, Any]:
        """Values and gradients (with new draws) -- a botorch_rs extension."""
        flat, kind, _ = self._prepare(X)
        batch = tuple(flat[1][:-2])
        draws = _standard_normal((self._bald.num_samples, *batch), X)
        (values, vshape), (grads, _) = self._bald.forward_with_grad(flat, draws)
        return from_flat(values, vshape, kind), from_flat(grads, to_flat(X)[1], kind)


__all__ = [
    "AnalyticExpectedUtilityOfBestOption",
    "PairwiseBayesianActiveLearningByDisagreement",
    "qExpectedUtilityOfBestOption",
]
