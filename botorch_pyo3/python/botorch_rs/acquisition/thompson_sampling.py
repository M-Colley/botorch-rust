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

On an ensemble model (a fully Bayesian GP), every member draws ``q`` paths
and path ``j`` of output ``t`` is that of the member ``ensemble_indices[j,
0, t]``, drawn uniformly at every redraw (BoTorch's
``select_from_ensemble_models``; explicit with ``redraw(batch_size,
draws=..., ensemble_indices=...)``, the draws of every member and output,
member-major).
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, _is_ensemble_model
from botorch_rs.acquisition.objective import objective_arg, transform_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.pathwise.prior_samplers import (
    DEFAULT_NUM_FEATURES,
    draws_arg,
    seed_arg,
)
from botorch_rs.utils.transforms import is_fully_bayesian

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
        self._is_ensemble = _is_ensemble_model(model)
        if self._is_ensemble:
            # The paths of every member (the ensemble's batch shape is its
            # members').
            self.model = model
            self._pending_kind = model._kind
        else:
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
        members_of = getattr(model, "_ensemble_members", None)
        self._state: Any  # ThompsonSamplingState or EnsembleThompsonSamplingState
        if self._is_ensemble:
            if members_of is None or not is_fully_bayesian(model):
                # BoTorch's paths of an EnsembleMapSaasSingleTaskGP (whose
                # kernel has no batch shape) fail for 1 < q != num_taus.
                raise UnsupportedError(
                    "botorch_rs supports PathwiseThompsonSampling on the ensembles of fully "
                    f"Bayesian GPs (and lists of them), not on {type(model).__name__}."
                )
            self._rs, self._state = _rs.ensemble_pathwise_thompson_sampling(
                [m._acquisition_model(None) for m in members_of()],
                objective_arg(objective),
                transform_arg(posterior_transform, model.num_outputs),
                self.num_features,
                seed_arg(seed),
            )
        else:
            self._rs, self._state = _rs.pathwise_thompson_sampling(
                model._acquisition_model(None),
                objective_arg(objective),
                transform_arg(posterior_transform, model.num_outputs),
                self.num_features,
                seed_arg(seed),
            )

    __init__._ensemble_aware = True  # type: ignore[attr-defined]

    @property
    def batch_size(self) -> int | None:
        """The number of drawn paths (``None`` before the first evaluation)."""
        return self._state.batch_size

    def redraw(self, batch_size: int, *, draws: Any = None, ensemble_indices: Any = None) -> None:
        """Draws ``batch_size`` new paths (from explicit ``draws`` and, for
        an ensemble, ``ensemble_indices`` (``batch_size x 1 x m``), botorch_rs
        extensions, or the generator)."""
        flat = draws_arg(draws, ("frequencies", "weights", "noise"))
        if not self._is_ensemble:
            if ensemble_indices is not None:
                raise ValueError("`ensemble_indices` are only used by ensemble models.")
            self._state.redraw(int(batch_size), flat)
            return
        idx = None
        if ensemble_indices is not None:
            values, _ = to_flat(ensemble_indices, "ensemble_indices")
            idx = [int(v) for v in values]
        self._state.redraw(int(batch_size), flat, idx)

    @property
    def ensemble_indices(self) -> Any:
        """BoTorch's ``ensemble_indices`` (``q x 1 x m``, int64): the member
        of every path and output (``None`` for a single model or before the
        first evaluation)."""
        if not self._is_ensemble:
            return None
        idx = self._state.ensemble_indices
        if idx is None:
            return None
        m = self.model.num_outputs
        q = len(idx) // m
        lib = self.model._kind.lib
        if lib == "torch":
            import torch

            return torch.tensor(idx, dtype=torch.long, device=self.model._kind.device).reshape(
                q, 1, m
            )
        if lib == "numpy":
            import numpy as np

            return np.asarray(idx, dtype=np.int64).reshape(q, 1, m)
        return [[[idx[j * m + t] for t in range(m)]] for j in range(q)]

    def select_from_ensemble_models(self, values: Any) -> Any:
        """Gather each drawn path/output's fixed ensemble member.

        Inputs have shape ``... x paths x ensemble x outputs``. Selection
        preserves input-array gradients and does not redraw the native paths.
        """
        if not self._is_ensemble:
            return values
        ids = self._state.ensemble_indices
        if ids is None:
            raise RuntimeError("Draw the paths before selecting ensemble members.")
        shape = _ops.shape_of(values)
        if len(shape) < 3:
            raise ValueError("Expected values with shape (..., paths, ensemble, outputs).")
        q, size, m = shape[-3:]
        if len(ids) != q * m or m != self.model.num_outputs or any(not 0 <= i < size for i in ids):
            raise ValueError("Values do not match the drawn path/output/ensemble dimensions.")
        kind = kind_of(values)
        target = (*shape[:-3], q, 1, m)
        if kind.lib == "torch":
            import torch

            index = (
                torch.as_tensor(ids, dtype=torch.long, device=values.device)
                .reshape(q, 1, m)
                .expand(target)
            )
            return values.gather(-2, index).squeeze(-2)
        if kind.lib == "numpy":
            import numpy as np

            numpy_index = np.broadcast_to(np.asarray(ids, dtype=np.int64).reshape(q, 1, m), target)
            return np.take_along_axis(values, numpy_index, axis=-2).squeeze(-2)
        data = to_flat(values, "values")[0]
        selected = [
            data[((batch * q + path) * size + int(ids[path * m + output])) * m + output]
            for batch in range(math.prod(shape[:-3]))
            for path in range(q)
            for output in range(m)
        ]
        return from_flat(selected, (*shape[:-3], q, m), kind)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(batch_size={self.batch_size})"


__all__ = ["BATCH_SIZE_CHANGE_ERROR", "PathwiseThompsonSampling"]
