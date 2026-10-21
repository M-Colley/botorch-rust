"""Robust exact GP regression with native relevance pursuit and support selection."""

from __future__ import annotations

import copy
import math
import sys
from typing import Any, cast

from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.likelihoods import Likelihood
from botorch_rs.models._backend_selection import requires_tensor_backend
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.likelihoods.sparse_outlier_noise import (
    SparseOutlierGaussianLikelihood,
    SparseOutlierNoise,
)
from botorch_rs.models.relevance_pursuit import (
    backward_relevance_pursuit,
    get_posterior_over_support,
)
from botorch_rs.utils.types import DEFAULT

FRACTIONS_OF_OUTLIERS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


class RobustRelevancePursuitMixin:
    """Fit a sequence of supports and retain the maximum posterior model."""

    likelihood: Likelihood

    def __init__(
        self,
        base_likelihood: Any,
        dim: int,
        prior_mean_of_support: float | None = None,
        convex_parameterization: bool = True,
        cache_model_trace: bool = False,
    ) -> None:
        mean = (
            float(max(1, int(0.2 * dim)))
            if prior_mean_of_support is None
            else float(prior_mean_of_support)
        )
        if not math.isfinite(mean) or mean <= 0:
            raise ValueError("prior_mean_of_support must be finite and positive.")
        self.likelihood = SparseOutlierGaussianLikelihood(
            base_likelihood.noise_covar, dim, convex_parameterization=convex_parameterization
        )
        self.likelihood._bind(self)
        self.prior_mean_of_support = mean
        self.cache_model_trace = bool(cache_model_trace)
        self.bmc_support_sizes: Any = None
        self.bmc_probabilities: Any = None
        self.model_trace: list[Any] | None = None

    def to_standard_model(self) -> SingleTaskGP:
        if not isinstance(self, SingleTaskGP):
            raise UnsupportedError("A robust model must implement to_standard_model().")
        model: Any = copy.deepcopy(self)
        model.__class__ = SingleTaskGP
        return cast(SingleTaskGP, model)

    def load_standard_model(self, standard_model: SingleTaskGP) -> RobustRelevancePursuitMixin:
        SparseOutlierNoise._from_model(standard_model)
        metadata = {
            name: getattr(self, name)
            for name in (
                "prior_mean_of_support",
                "cache_model_trace",
                "bmc_support_sizes",
                "bmc_probabilities",
                "model_trace",
            )
        }
        memo: dict[int, Any] = {id(standard_model): self}
        state = {
            key: (value.copy() if key == "_rs" else copy.deepcopy(value, memo))
            for key, value in standard_model.__dict__.items()
        }
        self.__dict__.clear()
        self.__dict__.update(state)
        self.__dict__.update(metadata)
        return self

    def custom_fit(
        self,
        mll: Any,
        *,
        numbers_of_outliers: Any = None,
        fractions_of_outliers: Any = None,
        timeout_sec: float | None = None,
        relevance_pursuit_optimizer: Any = backward_relevance_pursuit,
        reset_parameters: bool = True,
        reset_dense_parameters: bool = False,
        closure: Any = None,
        optimizer: Any = None,
        closure_kwargs: Any = None,
        optimizer_kwargs: Any = None,
    ) -> Any:
        from botorch_rs.mlls import ExactMarginalLogLikelihood

        if not isinstance(mll, ExactMarginalLogLikelihood):
            raise UnsupportedError("Robust relevance pursuit requires an exact Gaussian MLL.")
        if timeout_sec is not None:
            raise UnsupportedError(
                "A timeout for native relevance pursuit is not supported yet; use optimizer maxiter/maxfun limits."
            )
        original = mll.model
        if not isinstance(original, RobustRelevancePursuitMixin):
            raise TypeError("custom_fit requires a robust relevance-pursuit model.")
        n = SparseOutlierNoise._from_model(original).dim
        if numbers_of_outliers is None:
            fractions = (
                FRACTIONS_OF_OUTLIERS
                if fractions_of_outliers is None
                else list(fractions_of_outliers)
            )
            if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in fractions):
                raise ValueError("fractions_of_outliers must lie in [0, 1].")
            numbers_of_outliers = [int(float(p) * n) for p in fractions]
        original_likelihood = mll.likelihood
        try:
            mll.model = original.to_standard_model()
            mll.likelihood = mll.model.likelihood
            sparse = SparseOutlierNoise._from_model(mll.model)
            _, trace = relevance_pursuit_optimizer(
                sparse_module=sparse,
                mll=mll,
                sparsity_levels=numbers_of_outliers,
                record_model_trace=True,
                reset_parameters=reset_parameters,
                reset_dense_parameters=reset_dense_parameters,
                closure=closure,
                optimizer=optimizer,
                closure_kwargs=closure_kwargs,
                optimizer_kwargs=optimizer_kwargs,
            )
            sizes, probabilities = get_posterior_over_support(
                SparseOutlierNoise, trace, prior_mean_of_support=original.prior_mean_of_support
            )
            values = list(to_flat(probabilities)[0])
            selected = max(range(len(values)), key=values.__getitem__)
            original.load_standard_model(trace[selected])
            original.bmc_support_sizes = sizes
            original.bmc_probabilities = probabilities
            original.model_trace = trace if original.cache_model_trace else None
        finally:
            mll.model = original
            mll.likelihood = getattr(original, "likelihood", original_likelihood)
        return mll


class RobustRelevancePursuitSingleTaskGP(SingleTaskGP, RobustRelevancePursuitMixin):
    """Exact native GP whose observation noise includes selected outliers."""

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        if backend not in ("auto", "native", "torch"):
            raise ValueError("backend must be 'auto', 'native', or 'torch'.")
        torch = sys.modules.get("torch")
        values = (train_X, train_Y, kwargs.get("train_Yvar"), *args[:1])
        modules = (
            *args[1:],
            *(
                kwargs.get(name)
                for name in (
                    "likelihood",
                    "covar_module",
                    "mean_module",
                    "input_transform",
                    "outcome_transform",
                )
            ),
        )
        if cls is RobustRelevancePursuitSingleTaskGP and (
            backend == "torch"
            or (
                backend == "auto"
                and (
                    any(
                        getattr(value, "requires_grad", False)
                        or getattr(getattr(value, "device", None), "type", "cpu") != "cpu"
                        for value in values
                    )
                    or (
                        torch is not None
                        and any(requires_tensor_backend(module) for module in modules)
                    )
                )
            )
        ):
            from botorch_rs.models._torch_backend_relevance import TorchRobustRelevancePursuitGP

            return object.__new__(TorchRobustRelevancePursuitGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        likelihood: Any = None,
        covar_module: Any = None,
        mean_module: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        convex_parameterization: bool = True,
        prior_mean_of_support: float | None = None,
        cache_model_trace: bool = False,
        *,
        backend: str = "auto",
    ) -> None:
        shape = to_flat(train_Y)[1]
        if len(shape) != 2 or shape[-1] != 1:
            raise UnsupportedError(
                "Robust relevance pursuit currently requires unbatched single-output data."
            )
        SingleTaskGP.__init__(
            self,
            train_X,
            train_Y,
            train_Yvar,
            likelihood,
            covar_module,
            mean_module,
            outcome_transform,
            input_transform,
        )
        RobustRelevancePursuitMixin.__init__(
            self,
            self.likelihood,
            shape[-2],
            prior_mean_of_support,
            convex_parameterization,
            cache_model_trace,
        )


__all__ = ["RobustRelevancePursuitMixin", "RobustRelevancePursuitSingleTaskGP"]
