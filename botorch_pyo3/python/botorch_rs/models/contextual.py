"""Gaussian processes with structural additive context decompositions."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import to_flat
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.kernels.contextual_sac import SACKernel


class SACGP(SingleTaskGP):
    """Exact GP with a native structural additive contextual covariance.

    Training outcomes are standardized by default, matching BoTorch.
    ``train_Yvar=None`` selects the inferred-noise likelihood.
    """

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any,
        decomposition: dict[str, list[int]],
    ) -> None:
        yshape = to_flat(train_Y, "train_Y")[1]
        batch = yshape[:-2] + ((yshape[-1],) if yshape[-1] > 1 else ())
        kernel = SACKernel(decomposition, batch, device=getattr(train_X, "device", None))
        super().__init__(train_X, train_Y, train_Yvar, covar_module=kernel)
        self.decomposition = kernel.decomposition

    @classmethod
    def construct_inputs(  # type: ignore[override]
        cls, training_data: Any, decomposition: dict[str, list[int]]
    ) -> dict[str, Any]:
        return {
            "train_Yvar": None,
            **super().construct_inputs(training_data),
            "decomposition": decomposition,
        }


__all__ = ["SACGP"]
