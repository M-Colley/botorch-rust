"""In-sample model fit metrics (``botorch.utils.evaluation``)."""

from __future__ import annotations

import math
from math import log
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.utils.transforms import is_fully_bayesian

MLL = "MLL"
AIC = "AIC"
BIC = "BIC"


def compute_in_sample_model_fit_metric(model: Any, criterion: str) -> float:
    """Compute a in-sample model fit metric.

    Args:
        model: A fitted ExactGP.
        criterion: Evaluation criterion. One of "MLL", "AIC", "BIC". AIC
            penalizes the MLL based on the number of parameters. BIC uses
            a slightly different penalty based on the number of parameters
            and data points.

    Returns:
        The in-sample evaluation metric.

    For a botorch_rs ``SingleTaskGP``, the log marginal likelihood of the
    (transformed) training targets is computed in Rust (averaged over the
    outputs and batch elements of a batched model, as BoTorch averages
    ``mll``), and the number of parameters is the number of raw
    hyperparameter values.
    """
    if criterion not in (AIC, BIC, MLL):
        raise ValueError(f"Invalid evaluation criterion {criterion}.")
    from botorch_rs.models.gp_regression import SingleTaskGP

    if not isinstance(model, SingleTaskGP):
        raise UnsupportedError(
            "compute_in_sample_model_fit_metric requires a botorch_rs SingleTaskGP (or a subclass)."
        )
    values, _ = _rs.data_log_likelihoods(model._rs)
    vals = [float(v) for v in values]
    mll_scalar = math.fsum(vals) / len(vals)
    num_params: float = float(sum(math.prod(shape) for _, shape in model._rs.parameter_shapes()))
    if is_fully_bayesian(model=model):
        num_params /= len(vals)
    n = model.train_inputs[0]
    from botorch_rs._arrays import to_flat

    n_train = to_flat(n, "train_inputs")[1][-2]
    if criterion == AIC:
        return 2 * num_params - 2 * mll_scalar
    elif criterion == BIC:
        return num_params * log(n_train) - 2 * mll_scalar
    return mll_scalar


__all__ = ["AIC", "BIC", "MLL", "compute_in_sample_model_fit_metric"]
