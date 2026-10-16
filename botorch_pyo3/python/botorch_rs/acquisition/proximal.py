r"""Proximal biasing of acquisition functions (``botorch.acquisition.proximal``),
computed in Rust with exact gradients.

The base acquisition function is weighted by a squared exponential
``exp(-0.5 ||(x - x_last) / w||^2)`` around the last training point (in the
model's input-transformed space with ``transformed_weighting``), for ``q =
1``. With ``beta``, ``softplus(acqf(x), beta)`` replaces the base value.

References:
    R. Roussel et al. Proximal biasing for Bayesian optimization and
    characterization of physical systems. arXiv:2209.04587.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.model_list_gp_regression import ModelListGP


class ProximalAcquisitionFunction(AcquisitionFunction):
    """A wrapper around AcquisitionFunctions to add proximal weighting of the
    acquisition function. The acquisition function is weighted via a squared
    exponential centered at the last training point, with varying lengthscales
    corresponding to ``proximal_weights``. Can only be used with acquisition
    functions based on single batch models (``q = 1``).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.0)
        >>> proximal_weights = torch.ones(d)
        >>> EI_proximal = ProximalAcquisitionFunction(EI, proximal_weights)
        >>> eip = EI_proximal(test_X)
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        proximal_weights: Any,
        transformed_weighting: bool | None = True,
        beta: float | None = None,
    ) -> None:
        if not isinstance(acq_function, AcquisitionFunction):
            raise UnsupportedError("`acq_function` must be a botorch_rs acquisition function.")
        model = acq_function.model
        super().__init__(model)
        self.acq_func = acq_function
        if acq_function.X_pending is not None:
            raise UnsupportedError("Proximal acquisition function requires `X_pending` to be None.")
        if isinstance(model, ModelListGP):
            first = model.models[0]
            x0 = to_flat(first.train_inputs[0], "train_inputs")
            tf0 = getattr(first, "input_transform", None)
            for sub in model.models[1:]:
                if to_flat(sub.train_inputs[0], "train_inputs") != x0:
                    raise UnsupportedError(
                        "Proximal acquisition function does not support unequal training inputs"
                    )
                if getattr(sub, "input_transform", None) is not tf0 and not _same_transform(
                    getattr(sub, "input_transform", None), tf0
                ):
                    raise UnsupportedError(
                        "Proximal acquisition function does not support non-identical "
                        "input transforms"
                    )
        data, shape = to_flat(proximal_weights, "proximal_weights")
        d = int(model._acquisition_model(None).dim)
        if len(shape) != 1 or shape[0] != d:
            raise ValueError(
                "`proximal_weights` must be a one dimensional tensor with "
                "same feature dimension as model."
            )
        self.proximal_weights = proximal_weights
        self.transformed_weighting = transformed_weighting
        self.beta = beta
        self._log = False
        self._rs = _rs.proximal_acquisition_function(
            acq_function._rs,
            model._acquisition_model(None),
            [float(w) for w in data],
            bool(transformed_weighting),
            None if beta is None else float(beta),
        )

    def _after_evaluation(self) -> None:
        self.acq_func._after_evaluation()


def _same_transform(a: Any, b: Any) -> bool:
    """Whether two input transforms are equal (both absent, or the same
    fitted ``Normalize``)."""
    if a is None or b is None:
        return a is None and b is None
    try:
        return bool(
            to_flat(a.offset, "offset") == to_flat(b.offset, "offset")
            and to_flat(a.coefficient, "coefficient") == to_flat(b.coefficient, "coefficient")
        )
    except AttributeError:
        return False


__all__ = ["ProximalAcquisitionFunction"]
