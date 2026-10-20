"""Exact model lists whose native fitter accumulates shared-parameter gradients."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import Kernel
from botorch_rs.means import ConstantMean, Mean, ZeroMean
from botorch_rs.mlls import SumMarginalLogLikelihood
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model_list_gp_regression import ModelListGP


class _SharedBinding:
    def __init__(self, models: list[Any]) -> None:
        self.models = models
        self._kind = models[0]._kind
        self._rs = self
        self.aug_batch_shape: list[int] = []

    @property
    def constant(self) -> Any:
        return self.models[0]._rs.constant

    def set_constant(self, value: Any) -> None:
        for model in self.models:
            model._rs.set_constant(value)

    def _raw_param(self, name: str) -> Any:
        return self.models[0]._raw_param(name)

    def _raw_param_shape(self, name: str) -> Any:
        return self.models[0]._raw_param_shape(name)

    def _set_raw_param(self, name: str, value: Any) -> None:
        for model in self.models:
            model._set_raw_param(name, value)


class _SharedBackend:
    def __init__(self, models: list[Any], mean: Mean, kernel: Kernel, freeze_noise: bool) -> None:
        self.models, self.mean, self.kernel = models, mean, kernel
        self._native = _rs.SharedGPFit(
            [m._rs for m in models], isinstance(mean, ConstantMean), freeze_noise
        )

    def raw_parameters(self) -> list[float]:
        raw = [float(model._raw_param("raw_noise")[0]) for model in self.models]
        if isinstance(self.mean, ConstantMean):
            raw.extend(to_flat(self.mean.raw_constant)[0])
        raw.extend(self.kernel._raw_flat())
        return raw

    def _sync(self) -> None:
        raw = self.raw_parameters()
        if raw != self._native.raw_parameters():
            self._native.set_raw_parameters(raw)

    def _install(self) -> None:
        for model, raw in zip(self.models, self._native.local_parameters(), strict=True):
            model._rs.set_raw_parameters(raw)

    def set_raw_parameters(self, raw: Any) -> None:
        self._native.set_raw_parameters(list(raw))
        self._install()

    def raw_parameter_bounds(self) -> Any:
        self._sync()
        return self._native.raw_parameter_bounds()

    def parameter_shapes(self) -> Any:
        return self._native.parameter_shapes()

    def parameter_names(self) -> Any:
        return self._native.parameter_names()

    def mll(self, raw: Any = None) -> Any:
        self._sync()
        return self._native.mll(raw)

    def mll_value_and_grad(self, raw: Any = None) -> Any:
        self._sync()
        return self._native.mll_value_and_grad(raw)

    def fit(self, **kwargs: Any) -> Any:
        self._sync()
        result = self._native.fit(**kwargs)
        self._install()
        return result

    def fit_torch(self, **kwargs: Any) -> Any:
        self._sync()
        result = self._native.fit_torch(**kwargs)
        self._install()
        return result


class _SharedSumMarginalLogLikelihood(SumMarginalLogLikelihood):
    _rs_shared_fit = True

    def __init__(self, model: ModelListGP, backend: _SharedBackend) -> None:
        super().__init__(model.likelihood, model)
        self._rs, self._kind = backend, model.models[0]._kind

    def _validate_fit(self) -> None:
        self._rs._sync()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError("Shared GP likelihoods accept no keyword arguments.")
        value = self._rs.mll()
        return from_flat(value[0], tuple(value[1]), self._kind)

    __call__ = forward

    def value_and_grad(self, raw_parameters: Any = None) -> tuple[float, list[float]]:
        raw = None if raw_parameters is None else list(to_flat(raw_parameters)[0])
        value, gradient = self._rs.mll_value_and_grad(raw)
        return float(value[0][0]), list(gradient)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Any:
        # This backend already stores each globally shared parameter once.
        if not recurse:
            return
        raw, offset = self._rs.raw_parameters(), 0
        for name, shape in self._rs.parameter_shapes():
            count = math.prod(shape)
            yield (
                f"{prefix}.{name}" if prefix else name,
                from_flat(raw[offset : offset + count], tuple(shape), self._kind),
            )
            offset += count

    def _param_entries(self) -> list[Any]:
        entries, offset = [], 0
        for name, value in self.named_parameters():
            size = math.prod(to_flat(value)[1])
            entries.append((name, value, offset, size))
            offset += size
        return entries

    def _fit(
        self,
        closure: Any = None,
        optimizer: Any = None,
        closure_kwargs: Any = None,
        optimizer_kwargs: Any = None,
        **kwargs: Any,
    ) -> Any:
        if closure is not None or closure_kwargs is not None:
            raise UnsupportedError("Shared GP fitting requires native analytic likelihoods.")
        name = getattr(optimizer, "__name__", None)
        settings = dict(optimizer_kwargs or {})
        self.train()
        if name == "fit_gpytorch_mll_torch":
            if kwargs:
                raise UnsupportedError(f"Unsupported shared Adam fit arguments: {sorted(kwargs)}.")
            result = optimizer(self, **settings)
            self._fit_result = dict(
                loss=result.fval, iterations=result.step, status=result.status.name
            )
        elif optimizer is None or name == "fit_gpytorch_mll_scipy":
            if settings.pop("method", "L-BFGS-B") != "L-BFGS-B":
                raise UnsupportedError("Shared GP fitting supports native L-BFGS-B or Adam.")
            options = dict(settings.pop("options", None) or {})
            allowed = {"maxiter", "maxfun", "ftol", "gtol", "maxcor", "maxls", "disp", "iprint"}
            if settings or set(options) - allowed:
                raise UnsupportedError("Unsupported shared GP optimizer arguments.")
            options.pop("disp", None)
            options.pop("iprint", None)
            if "gtol" in options:
                options["pgtol"] = options.pop("gtol")
            attempts, seed = kwargs.pop("max_attempts", 5), kwargs.pop("seed", None)
            if kwargs:
                raise UnsupportedError(f"Unsupported shared GP fit arguments: {sorted(kwargs)}.")
            loss, iterations, evaluations, status, message, attempts = self._rs.fit(
                max_attempts=attempts, seed=seed, **options
            )
            self._fit_result = dict(
                loss=loss,
                iterations=iterations,
                evaluations=evaluations,
                status=status,
                message=message,
                attempts=attempts,
            )
        else:
            raise UnsupportedError("Shared GP fitting supports native L-BFGS-B or Adam.")
        self.eval()
        return self


def build_shared_gp_model_list(
    datasets: list[Any],
    mean_module: Mean,
    covar_module: Kernel,
    observation_noise: float | None = None,
) -> tuple[ModelListGP, SumMarginalLogLikelihood]:
    """Build exact GPs with one jointly fitted mean/kernel and separate noises."""
    from botorch_rs.models.empirical_gps.em_empirical_gp import _copy_modules

    if not datasets:
        raise ValueError("At least one historical dataset is required.")
    if not isinstance(mean_module, (ZeroMean, ConstantMean)) or not isinstance(
        covar_module, Kernel
    ):
        raise UnsupportedError(
            "Shared GP lists require native ZeroMean/ConstantMean and covariance kernels."
        )
    if (
        mean_module.batch_shape
        or covar_module.batch_shape
        or mean_module._model is not None
        or covar_module._model is not None
    ):
        raise UnsupportedError("Shared GP modules must initially be unbatched and unbound.")
    if observation_noise is not None and (
        not math.isfinite(observation_noise) or observation_noise < 1e-4
    ):
        raise ValueError(
            "observation_noise must satisfy the Gaussian likelihood's lower bound 1e-4."
        )
    models: list[Any] = []
    for data in datasets:
        xs, ys = to_flat(data.X)[1], to_flat(data.Y)[1]
        if len(xs) != 2 or xs[0] == 0 or ys != (xs[0], 1):
            raise ValueError("Shared GP datasets require unbatched n x d inputs and n x 1 targets.")
        mean, kernel = _copy_modules(mean_module, covar_module)
        actual_mean = mean if isinstance(mean, ConstantMean) else ConstantMean()
        model: Any = SingleTaskGP(data.X, data.Y, mean_module=actual_mean, covar_module=kernel)
        if observation_noise is not None:
            model.likelihood.noise = observation_noise
        models.append(model)
    binding = _SharedBinding(models)
    mean_module._bind(binding)
    covar_module._bind(binding, "covar_module.")
    for model in models:
        model.mean_module, model.covar_module = mean_module, covar_module
    result = ModelListGP(*models)
    backend = _SharedBackend(models, mean_module, covar_module, observation_noise is not None)
    return result, _SharedSumMarginalLogLikelihood(result, backend)


__all__ = ["build_shared_gp_model_list"]
