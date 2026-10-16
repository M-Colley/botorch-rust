r"""Pathwise Thompson sampling (``botorch.acquisition.thompson_sampling``),
computed in Rust with exact gradients.

The value of a candidate set is the sum over its ``q`` points of ``q``
independent posterior sample paths (Matheron paths), path ``j`` at point
``j``, through the posterior transform or the MC objective; maximizing it
yields ``q`` Thompson samples. The paths are drawn at the first evaluation
with the candidate sets' ``q`` (``redraw(batch_size)``).

Random numbers: botorch_rs draws the paths' random numbers from its own
generator (``seed``, a botorch_rs extension; every redraw uses a new stream)
or takes them explicitly (``redraw(batch_size, draws=...)``, one
``(frequencies, weights, noise)`` tuple per output, see
``draw_matheron_paths``).
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.acquisition.objective import objective_arg, transform_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.pathwise.prior_samplers import (
    DEFAULT_NUM_FEATURES,
    draws_arg,
    seed_arg,
)

#: BoTorch's error when the batch size changes.
BATCH_SIZE_CHANGE_ERROR = (
    "The batch size of PathwiseThompsonSampling should not change during a "
    "forward pass - was {}, now {}. Please re-initialize the acquisition if you "
    "want to change the batch size."
)


class PathwiseThompsonSampling(AcquisitionFunction):
    r"""Single-outcome Thompson Sampling packaged as an (analytic)
    acquisition function. Querying the acquisition function gives the summed
    values of one or more draws from a pathwise drawn posterior sample, and
    thus its maximization yields one (or multiple) Thompson sample(s).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> TS = PathwiseThompsonSampling(model)

    botorch_rs extensions (keyword-only): ``num_features`` of the paths
    (BoTorch's default 1024) and ``seed``.
    """

    def __init__(
        self,
        model: Any,
        objective: Any = None,
        posterior_transform: Any = None,
        *,
        num_features: int = DEFAULT_NUM_FEATURES,
        seed: int | None = None,
    ) -> None:
        super().__init__(model)
        if objective is None and model.num_outputs != 1:
            if posterior_transform is None:
                raise UnsupportedError(
                    "Must specify an objective or a posterior transform when using "
                    "a multi-output model."
                )
            if not getattr(posterior_transform, "scalarize", False):
                raise UnsupportedError(
                    "If using a multi-output model without an objective, "
                    "posterior_transform must scalarize the output."
                )
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.num_features = int(num_features)
        self._rs, self._state = _rs.pathwise_thompson_sampling(
            model._acquisition_model(None),
            objective_arg(objective),
            transform_arg(posterior_transform, model.num_outputs),
            self.num_features,
            seed_arg(seed),
        )

    @property
    def batch_size(self) -> int | None:
        """The number of drawn paths (``None`` before the first evaluation)."""
        return self._state.batch_size

    def redraw(self, batch_size: int, *, draws: Any = None) -> None:
        """Draws ``batch_size`` new paths (from explicit ``draws``, a
        botorch_rs extension, or the generator)."""
        self._state.redraw(int(batch_size), draws_arg(draws, ("frequencies", "weights", "noise")))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(batch_size={self.batch_size})"


__all__ = ["BATCH_SIZE_CHANGE_ERROR", "PathwiseThompsonSampling"]
