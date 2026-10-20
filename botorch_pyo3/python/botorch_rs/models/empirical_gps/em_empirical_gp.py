"""Exact native EM estimation of empirical Gaussian-process priors."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import Kernel
from botorch_rs.means import ConstantMean, Mean, ZeroMean
from botorch_rs.models.empirical_gps.utils import (
    ExperimentDataset,
    UniqueInputs,
    build_unique_inputs,
)
from botorch_rs.models.gpytorch import GPyTorchModel


def _mean_constant(module: Any) -> float:
    if isinstance(module, ZeroMean):
        return 0.0
    if not isinstance(module, ConstantMean) or tuple(module.batch_shape):
        raise UnsupportedError(
            "EM priors currently require an unbatched native ZeroMean or ConstantMean."
        )
    data = to_flat(module.constant)[0]
    if len(data) != 1:
        raise UnsupportedError("EM priors require a scalar parametric mean.")
    return float(data[0])


def _kernel_state(module: Any) -> tuple[Any, list[float]]:
    if not isinstance(module, Kernel) or module.batch_shape:
        raise UnsupportedError("EM priors require an unbatched native covariance kernel.")
    node = module._node()
    raw = module._raw_flat()
    if len(raw) != node.num_raw:
        raise UnsupportedError("Batched covariance parameters are not supported in EM priors.")
    return node, raw


class _PriorPickler(pickle.Pickler):
    """Serialize descriptor snapshots without mutable native model backreferences."""

    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, Kernel):
            state = dict(obj.__dict__)
            state.update(_model=None, _prefix="", _preset=None, _local_kind=obj._kind_out())
            state["_local"] = {
                name: list(obj._raw_values(name)) for name, _, _ in obj._own_params()
            }
            if getattr(obj, "ard", False) and "curve_weights" in state["_local"]:
                raw = state["_local"]["curve_weights"]
                state["_weight_array"] = from_flat(raw, (len(raw),), obj._kind_out())
            return "kernel", id(obj), type(obj), state
        if isinstance(obj, Mean):
            state = dict(obj.__dict__)
            state["_model"] = None
            if isinstance(obj, ConstantMean):
                state.update(
                    _local_raw=[_mean_constant(obj)],
                    _local_kind=kind_of(obj.constant),
                    _has_local_constant=True,
                )
            return "mean", id(obj), type(obj), state
        if isinstance(obj, _rs.Constraint):
            return (
                "constraint",
                obj.kind,
                obj.lower_bound,
                obj.upper_bound,
                obj.transform_name,
                obj.initial_value,
                obj.initial_raw_value,
            )
        if isinstance(obj, _rs.Prior):
            return "prior", _rs.em_prior_state(obj)
        return None


class _PriorUnpickler(pickle.Unpickler):
    def __init__(self, file: Any) -> None:
        super().__init__(file)
        self._modules: dict[int, Any] = {}

    def persistent_load(self, key: Any) -> Any:
        tag, *values = key
        if tag in ("kernel", "mean"):
            identity, cls, state = values
            if identity not in self._modules:
                result = cls.__new__(cls)
                result.__dict__.update(state)
                self._modules[identity] = result
            return self._modules[identity]
        if tag == "constraint":
            name, lower, upper, transform, initial, raw_initial = values
            if name == "GreaterThan":
                return _rs.Constraint.greater_than(lower, transform, initial, False)
            if name == "LessThan":
                return _rs.Constraint.less_than(upper, transform, initial, False)
            if name == "LogTransformedInterval":
                return _rs.Constraint.log_transformed_interval(
                    lower, upper, initial, False, raw_initial
                )
            if name == "Interval":
                return _rs.Constraint.interval(lower, upper, transform, None, initial, False)
        if tag == "prior":
            name, params, float32, torch_log = values[0]
            names = {
                "LogNormal": "log_normal",
                "Gamma": "gamma",
                "Normal": "normal",
                "Uniform": "uniform",
                "HalfNormal": "half_normal",
                "HalfCauchy": "half_cauchy",
                "Beta": "beta",
                "SmoothedBox": "smoothed_box",
            }
            result = getattr(_rs.Prior, names[name])(*params, float32=float32)
            return result if torch_log is None else result.with_torch_log(torch_log)
        raise pickle.UnpicklingError("Unknown native EM prior descriptor.")


@dataclass(frozen=True)
class EMPriorContainer:
    """Reusable empirical inducing moments and their native module descriptors."""

    datasets: list[ExperimentDataset]
    mean_module: Mean
    covar_module: Kernel
    likelihood_noise: Any
    num_em_iterations: int
    use_mean_prior: bool
    use_covar_prior: bool
    iw_nu: float | None
    em_convergence_tol: float | None
    enable_interpolation: bool
    use_inducing_points: bool
    X_inducing: Any
    mu_inducing: Any
    Sigma_inducing: Any
    L_kernel_inducing: Any
    delta_mu: Any
    unique_inputs: UniqueInputs

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            _PriorPickler(file, protocol=pickle.HIGHEST_PROTOCOL).dump(self)

    @classmethod
    def load(cls, path: str) -> EMPriorContainer:
        with open(path, "rb") as file:
            result = _PriorUnpickler(file).load()
        if not isinstance(result, cls):
            raise TypeError("Saved file does not contain an EMPriorContainer.")
        return result


def pretrain_em_prior(
    datasets: list[ExperimentDataset],
    mean_module: Mean,
    covar_module: Kernel,
    likelihood_noise: Any = None,
    inducing_points: Any = None,
    num_em_iterations: int = 16,
    use_mean_prior: bool = False,
    use_covar_prior: bool = False,
    iw_nu: float | None = None,
    em_convergence_tol: float | None = 1e-6,
    enable_interpolation: bool = True,
    init_mode: str = "kernel",
    iteration_history: list[tuple[Any, Any]] | None = None,
    covariance_shrinkage: float = 0.0,
) -> EMPriorContainer:
    """Estimate exact Gaussian inducing moments from independent historical curves.

    Indexed and inducing-point E-steps, mean/IW MAP updates, PSD stabilization
    and trace-matched shrinkage are native. Historical observations are fixed.
    """
    if not datasets:
        raise ValueError("At least one historical dataset is required.")
    if not isinstance(num_em_iterations, int) or num_em_iterations < 0:
        raise ValueError("num_em_iterations must be a nonnegative integer.")
    if init_mode not in ("kernel", "naive"):
        raise ValueError("init_mode must be 'kernel' or 'naive'.")
    for data in datasets:
        GPyTorchModel._validate_tensor_args(data.X, data.Y)
        x, y = _ops.shape_of(data.X), _ops.shape_of(data.Y)
        if len(x) != 2 or x[0] == 0 or x[1] == 0 or y != (x[0], 1):
            raise ValueError("EM datasets require unbatched n x d inputs and n x 1 outcomes.")
        if _ops._requires_grad((data.X, data.Y)):
            raise UnsupportedError("Historical EM observations are fixed data.")
    unique = build_unique_inputs(datasets)
    X = unique.X_all if inducing_points is None else inducing_points
    shape = _ops.shape_of(X)
    if len(shape) != 2 or shape[0] == 0 or shape[1] != _ops.shape_of(datasets[0].X)[-1]:
        raise ValueError(
            "inducing_points must be a nonempty matrix matching the data feature dimension."
        )
    if _ops._requires_grad((X,)):
        raise UnsupportedError("EM pretraining requires fixed inducing locations.")
    kind = kind_of(X)
    values = [1e-10] if likelihood_noise is None else list(to_flat(likelihood_noise)[0])
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] < 0:
        raise ValueError("likelihood_noise must be a finite nonnegative scalar variance.")
    node, raw = _kernel_state(covar_module)
    indices = (
        None
        if inducing_points is not None
        else [[int(v) for v in to_flat(i)[0]] for i in unique.experiment_indices]
    )
    state = _rs.em_pretrain(
        [to_flat(data.X) for data in datasets],
        [to_flat(data.Y) for data in datasets],
        to_flat(X),
        node,
        raw,
        _mean_constant(mean_module),
        indices,
        values[0],
        num_em_iterations,
        use_mean_prior,
        use_covar_prior,
        iw_nu,
        em_convergence_tol,
        init_mode == "naive",
        covariance_shrinkage,
        iteration_history is not None,
    )
    mu, sigma, chol, delta, history, _ = state

    def output(pair: Any) -> Any:
        return from_flat(pair[0], tuple(pair[1]), kind)

    if iteration_history is not None:
        iteration_history.extend((output(m), output(c)) for m, c in history)
    return EMPriorContainer(
        datasets,
        mean_module,
        covar_module,
        from_flat(values, (), kind),
        num_em_iterations,
        use_mean_prior,
        use_covar_prior,
        float(shape[0] + 2) if use_covar_prior and iw_nu is None else iw_nu,
        em_convergence_tol,
        enable_interpolation,
        inducing_points is not None,
        X,
        output(mu),
        output(sigma),
        output(chol),
        output(delta),
        unique,
    )


__all__ = ["EMPriorContainer", "pretrain_em_prior"]
