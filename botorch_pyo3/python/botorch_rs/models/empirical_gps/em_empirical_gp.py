"""Exact native EM estimation of empirical Gaussian-process priors."""

from __future__ import annotations

import copy
import io
import math
import pickle
from dataclasses import dataclass
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import broadcast_to, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import Kernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean, Mean, ZeroMean
from botorch_rs.mlls import MarginalLogLikelihood
from botorch_rs.models.empirical_gps.utils import (
    ExperimentDataset,
    UniqueInputs,
    _reshape,
    build_unique_inputs,
)
from botorch_rs.models.gpytorch import GPyTorchModel
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior


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


def _copy_modules(*modules: Any) -> Any:
    buffer = io.BytesIO()
    _PriorPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(modules)
    buffer.seek(0)
    return _PriorUnpickler(buffer).load()


class EMEmpiricalGaussianProcess(GPyTorchModel):
    """Exact native conditioning on a fixed EM prior, with analytic fitting.

    Pretrained mean/kernel parameters may be frozen or fitted through shift
    interpolation. Historical EM moments stay fixed during this fitting path.
    """

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        em_prior: EMPriorContainer | None = None,
        datasets: list[ExperimentDataset] | None = None,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        likelihood: Any = None,
        num_em_iterations: int = 16,
        use_mean_prior: bool = False,
        use_covar_prior: bool = False,
        iw_nu: float | None = None,
        inducing_points: Any = None,
        enable_interpolation: bool = True,
        warm_start_em: bool = False,
        em_convergence_tol: float | None = 1e-6,
        init_mode: str = "kernel",
        learnable_inducing_points: bool = False,
        covariance_shrinkage: float = 0.0,
    ) -> None:
        from botorch_rs.models.higher_order_gp import _initial_raw_noise

        if learnable_inducing_points or warm_start_em:
            raise UnsupportedError(
                "Learnable inducing locations and warm-start EM gradients are not implemented."
            )
        if likelihood is None:
            likelihood = GaussianLikelihood()
        if not isinstance(likelihood, GaussianLikelihood) or likelihood.batch_shape:
            raise ValueError("EMEmpiricalGaussianProcess requires an unbatched GaussianLikelihood.")
        self._validate_tensor_args(train_X, train_Y)
        xs, ys = _ops.shape_of(train_X), _ops.shape_of(train_Y)
        if len(xs) != 2 or xs[0] == 0 or ys != (xs[0], 1):
            raise ValueError("EM models require unbatched n x d inputs and n x 1 targets.")
        raw_noise = (
            _initial_raw_noise(likelihood)
            if likelihood._model is None
            else to_flat(likelihood.raw_noise)[0][0]
        )
        using_pretrained = em_prior is not None
        if em_prior is not None and covariance_shrinkage > 0:
            raise ValueError("covariance_shrinkage only applies to EM pretraining from datasets.")
        if em_prior is None:
            if datasets is None or mean_module is None or covar_module is None:
                raise ValueError(
                    "Provide an em_prior, or datasets with mean_module and covar_module."
                )
            em_prior = pretrain_em_prior(
                datasets,
                mean_module,
                covar_module,
                likelihood_noise=likelihood.noise_constraint._rs.transform(raw_noise),
                inducing_points=inducing_points,
                num_em_iterations=num_em_iterations,
                use_mean_prior=use_mean_prior,
                use_covar_prior=use_covar_prior,
                iw_nu=iw_nu,
                em_convergence_tol=em_convergence_tol,
                enable_interpolation=enable_interpolation,
                init_mode=init_mode,
                covariance_shrinkage=covariance_shrinkage,
            )
        if not isinstance(em_prior, EMPriorContainer):
            raise TypeError("em_prior must be an EMPriorContainer.")
        if _ops.shape_of(em_prior.X_inducing)[-1] != xs[-1]:
            raise ValueError("The EM prior and train_X feature dimensions differ.")
        mean, covar = _copy_modules(em_prior.mean_module, em_prior.covar_module)
        self.initial_mean_module, self.initial_covar_module = mean, covar
        self.likelihood = GaussianLikelihood(likelihood.noise_prior, likelihood.noise_constraint)
        self._train_X, self._train_Y = _ops.clone(train_X), _ops.clone(train_Y)
        self._kind = kind_of(train_X)
        self._prior_container = em_prior
        self.datasets = em_prior.datasets
        self._using_pretrained_prior = using_pretrained
        self.enable_interpolation = em_prior.enable_interpolation
        self._use_inducing_points = em_prior.use_inducing_points
        self._X_inducing = _ops.clone(em_prior.X_inducing)
        self._mu_inducing = _ops.clone(em_prior.mu_inducing)
        self._Sigma_inducing = _ops.clone(em_prior.Sigma_inducing)
        self._cached_L_kernel_inducing = _ops.clone(em_prior.L_kernel_inducing)
        self._cached_delta_mu = _ops.clone(em_prior.delta_mu)
        self._unique_inputs_obs = em_prior.unique_inputs
        self._N_inducing = _ops.shape_of(self._X_inducing)[0]
        self.num_em_iterations = em_prior.num_em_iterations
        self.use_mean_prior, self.use_covar_prior = (
            em_prior.use_mean_prior,
            em_prior.use_covar_prior,
        )
        self.iw_nu, self.em_convergence_tol = em_prior.iw_nu, em_prior.em_convergence_tol
        self.warm_start_em = False
        self._num_outputs = 1
        self._additive_base = None
        self._construct_native(raw_noise, freeze=False)
        self.training = True

    def _construct_native(self, noise_raw: float, freeze: bool) -> None:
        mean, covar = self.initial_mean_module, self.initial_covar_module
        node, raw = _kernel_state(covar)
        base = self._additive_base
        base_raw: list[float]
        base_node, base_raw = (None, []) if base is None else _kernel_state(base)
        self._freeze_pretrained = freeze
        constraint = getattr(mean, "raw_constant_constraint", None)
        prior = getattr(mean, "mean_prior", None)
        self._rs = _rs.EMGaussianProcess(
            to_flat(self._train_X),
            to_flat(self._train_Y),
            [to_flat(data.X) for data in self.datasets],
            [to_flat(data.Y) for data in self.datasets],
            to_flat(self._X_inducing),
            to_flat(self._mu_inducing),
            to_flat(self._Sigma_inducing),
            node,
            raw,
            self.likelihood._spec,
            noise_raw,
            constant=_mean_constant(mean),
            learn_mean=isinstance(mean, ConstantMean),
            mean_prior=None if prior is None else prior._rs,
            mean_constraint=None if constraint is None else constraint._rs,
            base=base_node,
            base_raw=base_raw,
            interpolate=self.enable_interpolation or self._use_inducing_points,
            inducing_observations=self._use_inducing_points,
            freeze_pretrained=freeze,
        )
        self._param_shapes = {name: tuple(shape) for name, shape in self._rs.parameter_shapes()}
        self.likelihood._bind(self)
        mean._bind(self)
        covar._bind(self, "initial_covar_module.")
        if base is not None:
            base._bind(self, "_additive_base.")

    @classmethod
    def from_pretrained(
        cls,
        em_prior: EMPriorContainer,
        train_X: Any,
        train_Y: Any,
        likelihood: Any = None,
        freeze_pretrained: bool = True,
        learnable_inducing_points: bool = False,
        base_covar_module: Kernel | None = None,
    ) -> EMEmpiricalGaussianProcess:
        model = cls(
            train_X,
            train_Y,
            em_prior=em_prior,
            likelihood=likelihood,
            learnable_inducing_points=learnable_inducing_points,
        )
        if base_covar_module is not None:
            (base,) = _copy_modules(base_covar_module)
            model._additive_base = base
            model._construct_native(model._rs.raw_parameters()[0], freeze_pretrained)
        elif freeze_pretrained:
            model.freeze_pretrained_parameters()
        return model

    def freeze_pretrained_parameters(self) -> None:
        self._rs.freeze_pretrained(True)
        self._freeze_pretrained = True

    @property
    def datasets(self) -> list[ExperimentDataset]:
        return self._datasets

    @datasets.setter
    def datasets(self, value: list[ExperimentDataset]) -> None:
        if not value:
            raise ValueError("At least one historical dataset is required.")
        self._datasets = list(value)
        if hasattr(self, "_rs"):
            self._construct_native(self._rs.raw_parameters()[0], self._freeze_pretrained)

    @property
    def batch_shape(self) -> tuple[()]:
        return ()

    @property
    def train_inputs(self) -> tuple[Any]:
        return (self._train_X,)

    @property
    def train_targets(self) -> Any:
        return _reshape(self._train_Y, (_ops.shape_of(self._train_X)[0],))

    def _offsets(self) -> dict[str, tuple[int, int]]:
        result = {}
        offset = 0
        for name, shape in self._rs.parameter_shapes():
            size = math.prod(shape)
            result[name] = (offset, size)
            offset += size
        return result

    def _param_index(self, suffix: str) -> tuple[str, int, int]:
        offsets = self._offsets()
        if suffix in offsets:
            return suffix, *offsets[suffix]
        matches = [
            (name, offset, size)
            for name, (offset, size) in offsets.items()
            if name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise AttributeError(f"The EM model has no unique parameter {suffix!r}.")
        return matches[0]

    def _raw_param(self, suffix: str) -> list[float]:
        _, offset, size = self._param_index(suffix)
        return self._rs.raw_parameters()[offset : offset + size]

    def _raw_param_shape(self, suffix: str) -> tuple[int, ...]:
        return self._param_shapes[self._param_index(suffix)[0]]

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        name, offset, size = self._param_index(suffix)
        data, shape = to_flat(value)
        raw = self._rs.raw_parameters()
        raw[offset : offset + size] = broadcast_to(data, shape, self._param_shapes[name])
        self._rs.set_raw_parameters(raw)

    def named_parameters(self) -> Any:
        raw = self._rs.raw_parameters()
        for name, (offset, size) in self._offsets().items():
            yield name, from_flat(raw[offset : offset + size], self._param_shapes[name], self._kind)

    def parameters(self) -> Any:
        return (value for _, value in self.named_parameters())

    def _param_entries(self) -> list[Any]:
        return [(name, value, *self._offsets()[name]) for name, value in self.named_parameters()]

    def posterior(
        self,
        X: Any,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        from botorch_rs.acquisition.objective import transform_arg

        if kwargs or not isinstance(observation_noise, bool):
            raise UnsupportedError(
                "EM posteriors require boolean observation_noise and no extra arguments."
            )
        if output_indices is not None and list(output_indices) != [0]:
            raise ValueError("EMEmpiricalGaussianProcess has one output.")
        self.eval()
        return GPyTorchPosterior(
            self._rs.model_handle(),
            to_flat(X),
            kind_of(X),
            observation_noise,
            num_outputs=1,
            transform=transform_arg(posterior_transform),
            query_inputs=X,
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> Any:
        from botorch_rs.acquisition.objective import transform_arg

        model = self._rs.model_handle()
        transform = transform_arg(posterior_transform)
        return model if transform is None else _rs.Model.from_models([model], transform)

    def forward(self, X: Any) -> Any:
        if self.training and not self._using_pretrained_prior:
            raise UnsupportedError(
                "Training-time EM reruns are not implemented; use eval() for cached prediction or construct from a pretrained prior for fitting."
            )
        return GPyTorchPosterior(
            self._rs.prior_model_handle(),
            to_flat(X),
            kind_of(X),
            False,
            num_outputs=1,
            query_inputs=X,
        ).distribution

    __call__ = forward

    def __deepcopy__(self, memo: dict[int, Any]) -> EMEmpiricalGaussianProcess:
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            result.__dict__[name] = value.copy() if name == "_rs" else copy.deepcopy(value, memo)
        return result


class EMEmpiricalMarginalLogLikelihood(MarginalLogLikelihood):
    """Native observed-data likelihood over all fixed historical datasets."""

    _rs_fit = True
    _rs_em_fit = True

    def __init__(self, likelihood: Any, model: EMEmpiricalGaussianProcess) -> None:
        if not isinstance(model, EMEmpiricalGaussianProcess):
            raise TypeError("EMEmpiricalMarginalLogLikelihood requires an EM model.")
        if likelihood is not model.likelihood:
            raise ValueError("The likelihood must be the EM model's likelihood.")
        super().__init__(likelihood, model)

    def _validate_fit(self) -> None:
        if not self.model._using_pretrained_prior:
            raise UnsupportedError(
                "Differentiating through EM iterations is not implemented; pretrain_em_prior then from_pretrained supports analytic fitting of the fixed prior."
            )

    def forward(self, function_dist: Any = None, target: Any = None) -> Any:
        self._validate_fit()
        value = self.model._rs.mll()
        return from_flat(value[0], tuple(value[1]), self.model._kind)

    __call__ = forward

    def value_and_grad(self, raw_parameters: Any = None) -> tuple[float, list[float]]:
        self._validate_fit()
        raw = None if raw_parameters is None else list(to_flat(raw_parameters)[0])
        value, gradient = self.model._rs.mll_value_and_grad(raw)
        return float(value[0][0]), list(gradient)

    def _fit(
        self,
        closure: Any = None,
        optimizer: Any = None,
        closure_kwargs: Any = None,
        optimizer_kwargs: Any = None,
        **kwargs: Any,
    ) -> EMEmpiricalMarginalLogLikelihood:
        self._validate_fit()
        if closure is not None or closure_kwargs is not None:
            raise UnsupportedError("EM fitting requires the native analytic likelihood closure.")
        options = dict(optimizer_kwargs or {})
        name = getattr(optimizer, "__name__", None)
        self.train()
        if name == "fit_gpytorch_mll_torch":
            if kwargs:
                raise UnsupportedError(f"Unsupported EM Adam fit arguments: {sorted(kwargs)}.")
            result = optimizer(self, **options)
            self._fit_result = {
                "loss": result.fval,
                "iterations": result.step,
                "status": result.status.name,
            }
        elif optimizer is None or name == "fit_gpytorch_mll_scipy":
            if options.pop("method", "L-BFGS-B") != "L-BFGS-B":
                raise UnsupportedError("EM fitting supports native L-BFGS-B or Adam.")
            settings = dict(options.pop("options", None) or {})
            for ignored in ("disp", "iprint"):
                settings.pop(ignored, None)
            allowed = {"maxiter", "maxfun", "ftol", "gtol", "maxcor", "maxls"}
            if options or set(settings) - allowed:
                raise UnsupportedError(
                    f"Unsupported EM optimizer options: {sorted(set(options) | (set(settings) - allowed))}."
                )
            attempts, seed = kwargs.pop("max_attempts", 5), kwargs.pop("seed", None)
            if kwargs:
                raise UnsupportedError(f"Unsupported EM fit arguments: {sorted(kwargs)}.")
            if "gtol" in settings:
                settings["pgtol"] = settings.pop("gtol")
            loss, iterations, evaluations, status, message, attempts = self.model._rs.fit(
                max_attempts=attempts, seed=seed, **settings
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
            raise UnsupportedError("EM fitting supports native L-BFGS-B or Adam.")
        self.eval()
        return self


from botorch_rs.models.empirical_gps.shared_gp import build_shared_gp_model_list  # noqa: E402

__all__ = [
    "EMEmpiricalGaussianProcess",
    "EMEmpiricalMarginalLogLikelihood",
    "EMPriorContainer",
    "pretrain_em_prior",
    "build_shared_gp_model_list",
]
