"""Best-of-all native fitting with rollback between independent attempts."""

from __future__ import annotations

import copy
import math
from typing import Any

from botorch_rs.exceptions.errors import InvalidArgumentError, ModelFittingError, NotPSDError


def fit_all_attempts(
    mll: Any,
    optimizer: Any,
    optimizer_kwargs: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> Any:
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.optim.utils.model_utils import sample_all_priors

    attempts = int(kwargs.pop("max_attempts", 5))
    if attempts < 1:
        raise InvalidArgumentError("max_attempts must be at least 1")
    seed = kwargs.pop("seed", None)
    original = list(mll.model._rs.raw_parameters())
    best: tuple[float, list[float], Any, int] | None = None
    summaries: list[Any] = []
    try:
        for attempt in range(1, attempts + 1):
            mll.model._rs.set_raw_parameters(original)
            mll.train()
            if attempt > 1:
                sample_all_priors(mll.model, seed=None if seed is None else int(seed) + attempt)
            try:
                fit_gpytorch_mll(
                    mll,
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                    max_attempts=1,
                    **kwargs,
                )
            except (ModelFittingError, NotPSDError):
                summaries.append(None)
                continue
            info = copy.deepcopy(mll._fit_result)
            loss = float(info["fval"] if isinstance(info, dict) else info.fval)
            summaries.append(info)
            if math.isfinite(loss) and (best is None or loss < best[0]):
                best = (loss, list(mll.model._rs.raw_parameters()), info, attempt)
    finally:
        # An unexpected error also restores the original model, rather than
        # leaving an unreviewed intermediate trial installed.
        mll.model._rs.set_raw_parameters(original)
        mll.train()
    if best is None:
        raise ModelFittingError("All attempts to fit the model have failed.")
    _, raw, info, selected = best
    mll.model._rs.set_raw_parameters(raw)
    info = copy.deepcopy(info)
    if isinstance(info, dict):
        info.update(attempts=attempts, selected_attempt=selected, attempt_results=summaries)
    mll._fit_attempt_results = summaries
    mll._fit_selected_attempt = selected
    mll._fit_result = info
    return mll.eval()
