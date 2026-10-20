"""Gaussian processes with structural additive context decompositions."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import to_flat
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.kernels.contextual_lcea import LCEAKernel
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


class LCEAGP(SingleTaskGP):
    """Exact GP with latent-context additive covariance and native fitting.

    Like BoTorch's contextual model, training is unbatched with one output.
    Outcomes retain their original scale (no default standardization).
    """

    covar_module: LCEAKernel

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any,
        decomposition: dict[str, list[int]],
        train_embedding: bool = True,
        cat_feature_dict: dict | None = None,
        embs_feature_dict: dict | None = None,
        embs_dim_list: list[int] | None = None,
        context_weight_dict: dict | None = None,
    ) -> None:
        xs, ys = to_flat(train_X)[1], to_flat(train_Y)[1]
        if len(xs) != 2 or len(ys) != 2 or ys[-1] != 1:
            raise ValueError("LCEAGP requires unbatched n x d inputs and n x 1 outcomes.")
        kernel = LCEAKernel(
            decomposition,
            (),
            train_embedding,
            cat_feature_dict,
            embs_feature_dict,
            embs_dim_list,
            context_weight_dict,
            getattr(train_X, "device", None),
        )
        super().__init__(train_X, train_Y, train_Yvar, covar_module=kernel, outcome_transform=None)
        self.decomposition = kernel.decomposition

    def _forward_pass(self) -> None:
        self.covar_module._forward_pass()

    def eval(self) -> LCEAGP:
        self._forward_pass()
        self.covar_module.eval()
        super().eval()
        return self

    @classmethod
    def construct_inputs(  # type: ignore[override]
        cls,
        training_data: Any,
        decomposition: dict[str, list[int]],
        train_embedding: bool = True,
        cat_feature_dict: dict | None = None,
        embs_feature_dict: dict | None = None,
        embs_dim_list: list[int] | None = None,
        context_weight_dict: dict | None = None,
    ) -> dict[str, Any]:
        return {
            "train_Yvar": None,
            **super().construct_inputs(training_data),
            "decomposition": decomposition,
            "train_embedding": train_embedding,
            "cat_feature_dict": cat_feature_dict,
            "embs_feature_dict": embs_feature_dict,
            "embs_dim_list": embs_dim_list,
            "context_weight_dict": context_weight_dict,
        }


__all__ = ["SACGP", "LCEAGP"]
