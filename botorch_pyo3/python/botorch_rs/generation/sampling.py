"""Sampling-based generation strategies (``botorch.generation.sampling``).

A ``SamplingStrategy`` returns samples from the input points rather than the
value for a set of points, as acquisition functions do. The posterior samples
come from the botorch_rs model (their base samples are drawn with the random
number generator of the caller's array library: with torch tensors,
``torch.manual_seed`` gives BoTorch's draws), the selections are computed in
Rust.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError


def _gather(X: Any, idcs: list[int], batch: tuple[int, ...], num_samples: int) -> Any:
    """``torch.gather`` of the rows ``idcs`` (``batch x num_samples``) from
    ``X`` (``batch' x N x d``, expanded to ``batch``)."""
    data, shape = to_flat(X, "X")
    n, d = shape[-2], shape[-1]
    nb = math.prod(batch)
    xb = math.prod(shape[:-2])
    out: list[float] = []
    for b in range(nb):
        src = b if xb == nb else 0
        for s in range(num_samples):
            i = idcs[b * num_samples + s]
            off = (src * n + i) * d
            out.extend(data[off : off + d])
    return from_flat(out, (*batch, num_samples, d), kind_of(X))


def _rsample(posterior: Any, num_samples: int, kind: ArrayKind) -> Any:
    """``posterior.rsample((num_samples,))``; for single-output posteriors of
    torch inputs, the base samples are drawn like GPyTorch's
    ``MultivariateNormal.rsample`` (``torch.randn(*batch, N, num_samples)``),
    so that ``torch.manual_seed`` reproduces BoTorch's samples."""
    if kind.lib == "torch" and len(posterior.base_sample_shape) == len(posterior.batch_shape) + 1:
        import torch

        batch = tuple(posterior.batch_shape)
        n = posterior.base_sample_shape[-1]
        z = torch.randn(*batch, n, num_samples, dtype=torch.float64)
        z = z.permute(-1, *range(len(batch) + 1)).contiguous()
        return posterior.rsample_from_base_samples((num_samples,), z)
    return posterior.rsample(sample_shape=(num_samples,))


class SamplingStrategy(ABC):
    """Abstract base class for sampling-based generation strategies."""

    @abstractmethod
    def forward(self, X: Any, num_samples: int = 1) -> Any:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A ``batch_shape x N x d`` array from which to sample (in the
                ``N`` dimension).
            num_samples: The number of samples to draw.

        Returns:
            A ``batch_shape x num_samples x d`` array of samples from ``X``.
        """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class MaxPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value
    (BoTorch's ``MaxPosteriorSampling``, Thompson sampling over a discrete
    set).

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Any,
        objective: Any = None,
        posterior_transform: Any = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
        """
        from botorch_rs.acquisition.objective import IdentityMCObjective

        self.model = model
        self.objective = IdentityMCObjective() if objective is None else objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement

    def forward(self, X: Any, num_samples: int = 1, observation_noise: bool = False) -> Any:
        r"""Sample from the model posterior.

        Args:
            X: A ``batch_shape x N x d`` array from which to sample (in the
                ``N`` dimension) according to the maximum posterior value
                under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A ``batch_shape x num_samples x d`` array of samples from ``X``.
        """
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        samples = _rsample(posterior, num_samples, kind_of(X))
        return self.maximize_samples(X, samples, num_samples)

    def _objective_values(self, samples: Any, X: Any) -> tuple[list[float], tuple[int, ...]]:
        """The objective values of the samples (``num_samples x batch x N``)."""
        from botorch_rs.acquisition.objective import IdentityMCObjective

        data, shape = to_flat(samples, "samples")
        if isinstance(self.objective, IdentityMCObjective):
            if shape[-1] != 1:
                raise UnsupportedError(
                    "MaxPosteriorSampling with the identity objective requires a single "
                    "output; pass a posterior transform or an objective."
                )
            return [float(v) for v in data], tuple(shape[:-1])
        out = self.objective(samples, X=X)
        odata, oshape = to_flat(out, "objective values")
        return [float(v) for v in odata], tuple(oshape)

    def maximize_samples(self, X: Any, samples: Any, num_samples: int = 1) -> Any:
        r"""The points of ``X`` that maximize the objective of every sample
        (``samples`` of shape ``num_samples x batch_shape x N x m``); without
        replacement, BoTorch's deduplication of the top maximizers."""
        obj, oshape = self._objective_values(samples, X)
        n = oshape[-1]
        batch = oshape[1:-1]
        if len(batch) > 1 and not self.replacement:
            raise NotImplementedError(
                "MaxPosteriorSampling without replacement for more than a single batch "
                "dimension is not yet implemented."
            )
        idcs = _rs.maximize_samples_py(
            obj, int(oshape[0]), math.prod(batch), int(n), bool(self.replacement)
        )
        return _gather(X, list(idcs), batch, int(oshape[0]))


class BoltzmannSampling(SamplingStrategy):
    r"""Sample from a set of points according to a tempered acquisition value
    (BoTorch's ``BoltzmannSampling``): a multinomial distribution over the
    points with weights ``exp(eta * standardize(acq_func(X[..., i, :])))``.

    The acquisition values are standardized over the ``N`` points of every
    batch (BoTorch standardizes batched inputs over the batch dimension,
    which makes its sampling uniform for a single batch, see the
    documentation of the port).

    Example:
        >>> UCB = UpperConfidenceBound(model, beta=0.1)
        >>> BMUCB = BoltzmannSampling(UCB, eta=0.5)
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = BMUCB(X, num_samples=5)
    """

    def __init__(self, acq_func: Any, eta: float = 1.0, replacement: bool = True) -> None:
        r"""Boltzmann Acquisition Value Sampling.

        Args:
            acq_func: The acquisition function; evaluated at the individual
                points.
            eta: The temperature parameter in the softmax.
            replacement: If True, sample with replacement.
        """
        self.acq_func = acq_func
        self.eta = eta
        self.replacement = replacement

    def forward(self, X: Any, num_samples: int = 1) -> Any:
        r"""Sample from a tempered value of the acquisition function value.

        Args:
            X: A ``batch_shape x N x d`` array from which to sample.
            num_samples: The number of samples to draw.

        Returns:
            A ``batch_shape x num_samples x d`` array of samples from ``X``.
        """
        data, shape = to_flat(X, "X")
        n, d = shape[-2], shape[-1]
        batch = tuple(shape[:-2])
        nb = math.prod(batch)
        kind = kind_of(X)
        # The points as `N x batch x 1 x d` (BoTorch's evaluation order).
        rows: list[float] = []
        for i in range(n):
            for b in range(nb):
                off = (b * n + i) * d
                rows.extend(data[off : off + d])
        values = to_flat(self.acq_func(from_flat(rows, (n * nb, 1, d), kind)), "values")[0]
        idcs: list[int] = []
        for b in range(nb):
            v = [float(values[i * nb + b]) for i in range(n)]
            mean = sum(v) / n
            std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else math.nan
            std = std if std >= 1e-9 else 1.0
            weights = [math.exp(self.eta * (x - mean) / std) for x in v]
            idcs.extend(batched_multinomial_indices(weights, num_samples, self.replacement, kind))
        return _gather(X, idcs, batch, num_samples)


class ConstrainedMaxPosteriorSampling(MaxPosteriorSampling):
    r"""Constrained max posterior sampling (BoTorch's
    ``ConstrainedMaxPosteriorSampling``).

    Posterior sampling where we try to maximize an objective function while
    simultaneously satisfying constraints ``c_i(x) <= 0``, each modeled by
    a separate GP (a ``ModelListGP``, or a single model). If a realization
    has a feasible point, infeasible points score ``-inf``; otherwise the
    score is the negative total violation.
    """

    def __init__(
        self,
        model: Any,
        constraint_model: Any,
        objective: Any = None,
        posterior_transform: Any = None,
        replacement: bool = True,
    ) -> None:
        if objective is not None:
            raise NotImplementedError(
                "`objective` is not supported for `ConstrainedMaxPosteriorSampling`."
            )
        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.constraint_model = constraint_model

    def forward(self, X: Any, num_samples: int = 1, observation_noise: bool = False) -> Any:
        r"""Sample from the model posteriors (``batch_shape x N x d`` points)."""
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        Y = _rsample(posterior, num_samples, kind_of(X))
        models = getattr(self.constraint_model, "models", None) or [self.constraint_model]
        c_parts = []
        for c_model in models:
            c_post = c_model.posterior(X, observation_noise=observation_noise)
            c_parts.append(to_flat(_rsample(c_post, num_samples, kind_of(X)), "C"))
        ydata, yshape = to_flat(Y, "Y")
        rows = math.prod(yshape[:-2])
        n = yshape[-2]
        k = sum(s[-1] for _, s in c_parts)
        cdata: list[float] = []
        for r in range(rows):
            for i in range(n):
                for data, shape in c_parts:
                    m = shape[-1]
                    off = (r * n + i) * m
                    cdata.extend(float(v) for v in data[off : off + m])
        scores = _rs.constrained_scores_py([float(v) for v in ydata], cdata, rows, n, k)
        samples = from_flat(scores, tuple(yshape), kind_of(Y))
        return self.maximize_samples(X=X, samples=samples, num_samples=num_samples)


def batched_multinomial_indices(
    weights: list[float], num_samples: int, replacement: bool, kind: ArrayKind
) -> list[int]:
    """Multinomial draws from one row of weights with the generator of the
    array library (``torch.multinomial`` for torch arrays)."""
    if kind.lib == "torch":
        import torch

        w = torch.tensor(weights, dtype=torch.float64)
        return [int(i) for i in torch.multinomial(w, num_samples, replacement=replacement).tolist()]
    import random

    if replacement:
        return random.choices(range(len(weights)), weights=weights, k=num_samples)
    from botorch_rs.optim.initializers import _Draws

    return _Draws(kind).multinomial(weights, num_samples)


__all__ = [
    "BoltzmannSampling",
    "ConstrainedMaxPosteriorSampling",
    "MaxPosteriorSampling",
    "SamplingStrategy",
]
