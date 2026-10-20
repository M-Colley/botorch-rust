"""Empirical multi-task GP with heterogeneous progression domains."""

from __future__ import annotations

import copy
import math
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import Kernel
from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from botorch_rs.means import Mean
from botorch_rs.models.empirical_gps.empirical_1d_gp import (
    BaseAugmentedEmpiricalKernel,
    EmpiricalOneDimensionalGP,
    EmpiricalOneDimensionalKernel,
    _SharedKernelBinding,
    _stack,
    _use_tensor_backend,
)
from botorch_rs.models.empirical_gps.utils import (
    LinearInterpolation1D,
    _moveaxis,
    _reshape,
    center_curves,
    compute_sample_covariance,
)
from botorch_rs.models.gpytorch import MultiTaskGPyTorchModel
from botorch_rs.models.higher_order_gp import _initial_raw_noise
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior


def _history(Xs: Any, Ys: Any) -> int:
    if (
        not isinstance(Xs, (list, tuple))
        or not isinstance(Ys, (list, tuple))
        or not Xs
        or len(Xs) != len(Ys)
    ):
        raise ValueError("Historical inputs and curves must be equally sized nonempty lists.")
    count = _ops.shape_of(Ys[0])[0] if len(_ops.shape_of(Ys[0])) == 2 else 0
    for x, y in zip(Xs, Ys, strict=True):
        xs, ys = _ops.shape_of(x), _ops.shape_of(y)
        if len(ys) != 2 or count == 0 or ys[0] != count or ys[1] < 2 or xs != (ys[1], 1):
            raise ValueError(
                "Historical tasks require paired curves x progression arrays and progression x 1 inputs."
            )
        if _ops._requires_grad((x, y)) or any(not math.isfinite(v) for v in to_flat(y)[0]):
            raise UnsupportedError("Historical tasks require fixed finite curve data.")
    return count


def _inputs(x: Any, task_feature: int, num_tasks: int) -> tuple[Any, list[int], tuple[int, ...]]:
    values, shape = to_flat(x)
    if len(shape) < 2 or shape[-1] != 2 or task_feature not in (0, 1, -1, -2):
        raise ValueError("Task inputs require two columns and a valid task_feature.")
    task_feature %= 2
    tasks = []
    for value in values[task_feature::2]:
        if not math.isfinite(value):
            raise ValueError("Task indices must be finite integers.")
        task = round(value)
        if abs(value - task) > 1e-6 or not 0 <= task < num_tasks:
            raise ValueError("Task indices must be integers in the historical task range.")
        tasks.append(task)
    return _ops.getitem(x, (..., 1 - task_feature)), tasks, shape[:-1]


class MultiTaskEmpiricalOneDimensionalMean(Mean):
    """Per-task interpolation of paired historical sample means."""

    def __new__(
        cls,
        historical_Xs: Any = None,
        historical_Ys: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        values = (
            *(historical_Xs if historical_Xs is not None else []),
            *(historical_Ys if historical_Ys is not None else []),
        )
        if cls is MultiTaskEmpiricalOneDimensionalMean and _use_tensor_backend(values, backend):
            from botorch_rs.models.empirical_gps._torch_multitask import TensorTaskEmpiricalMean

            return object.__new__(TensorTaskEmpiricalMean)
        return object.__new__(cls)

    def __init__(
        self, historical_Xs: Any, historical_Ys: Any, task_feature: int, *, backend: str = "auto"
    ) -> None:
        _history(historical_Xs, historical_Ys)
        if task_feature not in (0, 1, -1, -2):
            raise ValueError("Invalid task_feature.")
        self.task_feature = task_feature % 2
        self.num_tasks = len(historical_Ys)
        self.interpolants = []
        for x, y in zip(historical_Xs, historical_Ys, strict=True):
            mean, _ = center_curves(y, curve_dim=0)
            self.interpolants.append(
                LinearInterpolation1D(_reshape(x, (_ops.shape_of(x)[0],)), mean)
            )

    def forward(self, x: Any) -> Any:
        progression, tasks, shape = _inputs(x, self.task_feature, self.num_tasks)
        progression = _reshape(progression, (len(tasks),))
        if not tasks:
            return from_flat([], shape, kind_of(x))
        values = [
            self.interpolants[task](_ops.getitem(progression, i)) for i, task in enumerate(tasks)
        ]
        return _reshape(_stack(values), shape)

    __call__ = forward

    def _mean_node(self) -> Any:
        return _rs.KernelNode.empirical_basis(
            [list(to_flat(f._x)[0]) for f in self.interpolants],
            [to_flat(_reshape(f._y, (1, _ops.shape_of(f._y)[0]))) for f in self.interpolants],
            [1.0],
            False,
            1.0,
            1,
            False,
        )


class MultiTaskEmpiricalOneDimensionalKernel(EmpiricalOneDimensionalKernel):
    """Paired historical-curve covariance across different task domains."""

    _is_task_kernel = True
    _vectorize_outputs = False

    def __new__(
        cls,
        historical_Xs: Any = None,
        historical_Ys: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        values = (
            *(historical_Xs if historical_Xs is not None else []),
            *(historical_Ys if historical_Ys is not None else []),
        )
        if cls is MultiTaskEmpiricalOneDimensionalKernel and _use_tensor_backend(values, backend):
            from botorch_rs.models.empirical_gps._torch_multitask import TensorTaskEmpiricalKernel

            return object.__new__(TensorTaskEmpiricalKernel)
        return object.__new__(cls)

    def __init__(
        self,
        historical_Xs: Any,
        historical_Ys: Any,
        task_feature: int,
        num_curves: int,
        correction: int = 0,
        *,
        backend: str = "auto",
    ) -> None:
        curves = _history(historical_Xs, historical_Ys)
        if curves != num_curves or not isinstance(correction, int) or correction >= curves:
            raise ValueError(
                "num_curves must match paired histories and correction must be smaller."
            )
        if task_feature not in (0, 1, -1, -2):
            raise ValueError("Invalid task_feature.")
        Kernel.__init__(self)
        self.task_feature = task_feature % 2
        self.num_tasks = self.num_outputs = len(historical_Ys)
        self.num_curves, self.correction = curves, correction
        self.ard = False
        self._weight_array = None
        self._effective_num_curves = curves
        self._use_svd = False
        self.interpolants = []
        for x, y in zip(historical_Xs, historical_Ys, strict=True):
            _, centered = center_curves(y, curve_dim=0)
            self.interpolants.append(
                LinearInterpolation1D(_reshape(x, (_ops.shape_of(x)[0],)), centered)
            )

    def _canonical_node(self) -> Any:
        return _rs.KernelNode.empirical_basis(
            [list(to_flat(f._x)[0]) for f in self.interpolants],
            [to_flat(f._y) for f in self.interpolants],
            [1.0] * self.num_curves,
            False,
            float(self.num_curves - self.correction),
            1,
            False,
        )

    def _base_node(self) -> Any:
        return self._canonical_node().with_parent_active_dims(
            [1 - self.task_feature, self.task_feature]
        )

    def _basis(self, x: Any) -> Any:
        progression, tasks, shape = _inputs(x, self.task_feature, self.num_tasks)
        progression = _reshape(progression, (len(tasks),))
        if not tasks:
            return from_flat([], (*shape[:-1], self.num_curves, shape[-1]), kind_of(x))
        rows = [
            self.interpolants[task](_ops.getitem(progression, i)) for i, task in enumerate(tasks)
        ]
        return _moveaxis(_reshape(_stack(rows), (*shape, self.num_curves)), -1, -2)

    def __call__(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Any:
        if last_dim_is_batch or params:
            raise UnsupportedError("Unsupported empirical task kernel arguments.")
        a = self._basis(x1)
        b = a if x2 is None or x2 is x1 else self._basis(x2)
        return compute_sample_covariance(
            a, b, self.num_curves, diag=diag, correction=self.correction
        )

    forward = __call__


class MultiTaskEmpiricalOneDimensionalGP(EmpiricalOneDimensionalGP, MultiTaskGPyTorchModel):
    """Exact empirical GP conditioned on arbitrary observed task subsets."""

    _correlated = True

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        task_feature: Any = None,
        historical_Xs: Any = None,
        historical_Ys: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        values = (
            train_X,
            train_Y,
            *(historical_Xs if historical_Xs is not None else []),
            *(historical_Ys if historical_Ys is not None else []),
            kwargs.get("train_Yvar"),
            *args[:1],
        )
        torch = sys.modules.get("torch")
        modules = (
            *args[3:],
            *(kwargs.get(name) for name in ("mean_module", "covar_module", "base_covar_module")),
        )
        selected = _use_tensor_backend(values, backend) or (
            backend == "auto"
            and torch is not None
            and any(isinstance(module, torch.nn.Module) for module in modules)
        )
        if cls is MultiTaskEmpiricalOneDimensionalGP and selected:
            from botorch_rs.models.empirical_gps._torch_multitask import TensorTaskEmpiricalGP

            return object.__new__(TensorTaskEmpiricalGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        historical_Xs: Any = None,
        historical_Ys: Any = None,
        train_Yvar: Any = None,
        output_tasks: Any = None,
        correction: int = 0,
        mean_module: Any = None,
        covar_module: Any = None,
        base_covar_module: Any = None,
        *,
        backend: str = "auto",
    ) -> None:
        if (mean_module is None) != (covar_module is None):
            raise ValueError("mean_module and covar_module must be provided together.")
        if mean_module is None:
            count = _history(historical_Xs, historical_Ys)
            mean_module = MultiTaskEmpiricalOneDimensionalMean(
                historical_Xs, historical_Ys, task_feature
            )
            covar_module = MultiTaskEmpiricalOneDimensionalKernel(
                historical_Xs, historical_Ys, task_feature, count, correction
            )
        elif correction != 0:
            raise ValueError("Set correction when constructing a pre-built covariance module.")
        if not isinstance(mean_module, MultiTaskEmpiricalOneDimensionalMean) or not isinstance(
            covar_module, MultiTaskEmpiricalOneDimensionalKernel
        ):
            raise UnsupportedError("Expected native empirical task mean and covariance modules.")
        if (
            mean_module.num_tasks != covar_module.num_tasks
            or mean_module.task_feature != task_feature % 2
            or covar_module.task_feature != task_feature % 2
        ):
            raise ValueError("Mean, covariance and task_feature configuration must match.")
        # These descriptors have no learned parameters. Reusing their fixed
        # historical arrays must not rebind a previously constructed model.
        if mean_module._model is not None:
            mean_module = copy.copy(mean_module)
            mean_module._model = None
        if covar_module._model is not None:
            covar_module = copy.copy(covar_module)
            covar_module._model = None
            covar_module._prefix = ""
        self._validate_tensor_args(train_X, train_Y, train_Yvar)
        if len(_ops.shape_of(train_X)) != 2 or _ops.shape_of(train_Y) != (
            _ops.shape_of(train_X)[0],
            1,
        ):
            raise UnsupportedError(
                "Empirical task training requires unbatched n x 2 and n x 1 arrays."
            )
        progression, tasks, _ = _inputs(train_X, task_feature, mean_module.num_tasks)
        outputs = list(range(mean_module.num_tasks)) if output_tasks is None else list(output_tasks)
        if not outputs or any(
            not isinstance(s, int) or s < 0 or s >= mean_module.num_tasks for s in outputs
        ):
            raise ValueError("output_tasks must contain valid historical task indices.")
        self._kind = kind_of(train_X)
        self._task_feature = task_feature % 2
        self.num_tasks, self.num_curves = mean_module.num_tasks, covar_module.num_curves
        self.num_non_task_features = 1
        self._num_outputs, self._output_tasks = len(outputs), outputs
        self._aug_batch_shape = ()
        self._has_base_kernel = base_covar_module is not None
        canonical = covar_module._canonical_node()
        if base_covar_module is not None:
            if not isinstance(base_covar_module, Kernel) or base_covar_module.batch_shape:
                raise UnsupportedError("Expected an unbatched native additive base kernel.")
            canonical = _rs.KernelNode.additive(
                [
                    canonical,
                    base_covar_module._node().with_parent_active_dims(
                        [1 - self._task_feature, self._task_feature]
                    ),
                ]
            )
            covar_module = BaseAugmentedEmpiricalKernel(covar_module, base_covar_module)
        for module in (mean_module, covar_module):
            if module._model is not None:
                raise UnsupportedError(
                    "Empirical modules must be unbound when constructing a model."
                )
        likelihood: Any = GaussianLikelihood()
        n = len(tasks)
        xx = (
            [v for p, t in zip(to_flat(progression)[0], tasks, strict=True) for v in (p, float(t))],
            (n, 2),
        )
        variance = None if train_Yvar is None else [to_flat(train_Yvar)]
        self._rs = _rs.EmpiricalGP(
            [xx],
            [to_flat(train_Y)],
            variance,
            mean_module._mean_node(),
            canonical,
            likelihood._spec,
            covar_module._raw_flat(),
            [_initial_raw_noise(likelihood)],
            True,
        )
        self._param_shapes = {
            self._public_name(name): tuple(shape)
            for name, shape in canonical.param_shapes("covar_module.")
        }
        if variance is None:
            self._param_shapes["likelihood.noise_covar.raw_noise"] = (1,)
        else:
            likelihood = FixedNoiseGaussianLikelihood._create()
        self._train_X, self._train_Y, self._train_Yvar = train_X, train_Y, train_Yvar
        self.likelihood, self.mean_module, self.covar_module = likelihood, mean_module, covar_module
        likelihood._bind(self)
        mean_module._bind(self)
        self._shared_kernel_binding = _SharedKernelBinding(self)
        covar_module._bind(self._shared_kernel_binding, "covar_module.")
        self.training = True

    @property
    def train_inputs(self) -> tuple[Any]:
        # The explicit task column already encodes the joint training event.
        # Correlated wide-output models repeat scalar progression inputs;
        # applying that representation here loses task labels and doubles rows.
        return (self._train_X,)

    @property
    def train_targets(self) -> Any:
        return _reshape(self._train_Y, (_ops.shape_of(self._train_X)[0],))

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
                "Empirical task posteriors accept boolean observation_noise and no extra arguments."
            )
        shape = _ops.shape_of(X)
        if len(shape) < 2 or shape[-1] not in (1, 2):
            raise ValueError("Queries require progression or progression plus task columns.")
        explicit = shape[-1] == 2
        if explicit and output_indices is not None:
            raise UnsupportedError("output_indices cannot accompany explicit task columns.")
        outputs = self._output_tasks if output_indices is None else list(output_indices)
        source = self._rs.task_model_handle(outputs, self._task_feature if explicit else None)
        self.eval()
        return GPyTorchPosterior(
            source,
            to_flat(X),
            kind_of(X),
            observation_noise,
            num_outputs=1 if explicit else len(outputs),
            transform=transform_arg(posterior_transform),
            query_inputs=X,
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> Any:
        from botorch_rs.acquisition.objective import transform_arg

        source = self._rs.task_model_handle(self._output_tasks)
        transform = transform_arg(posterior_transform)
        return source if transform is None else _rs.Model.from_models([source], transform)

    def _map_tasks(self, task_values: Any) -> Any:
        return task_values

    def _split_inputs(self, x: Any) -> tuple[Any, Any, Any]:
        _, tasks, shape = _inputs(x, self._task_feature, self.num_tasks)
        kind = kind_of(x)
        indices: Any
        if kind.lib == "torch":
            import torch

            indices = torch.tensor(tasks, dtype=torch.long, device=kind.device).reshape((*shape, 1))
        elif kind.lib == "numpy":
            import numpy as np

            indices = np.asarray(tasks, dtype=np.int64).reshape((*shape, 1))
        else:
            indices = from_flat([float(t) for t in tasks], (*shape, 1), kind)
        before = _ops.getitem(x, (..., slice(None), slice(None, self._task_feature)))
        after = _ops.getitem(x, (..., slice(None), slice(self._task_feature + 1, None)))
        return before, indices, after

    @classmethod
    def from_homogeneous_data(
        cls,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        historical_X: Any,
        historical_Y: Any,
        **kwargs: Any,
    ) -> Any:
        shape = _ops.shape_of(historical_Y)
        if len(shape) != 3:
            raise ValueError("historical_Y must have three dimensions.")
        return cls(
            train_X,
            train_Y,
            task_feature,
            [historical_X] * shape[-1],
            [_ops.getitem(historical_Y, (..., s)) for s in range(shape[-1])],
            **kwargs,
        )

    @classmethod
    def from_wide_format(
        cls,
        train_X: Any,
        train_Y: Any,
        historical_Xs: Any,
        historical_Ys: Any,
        train_Yvar: Any = None,
        output_tasks: Any = None,
        correction: int = 0,
        *,
        backend: str = "auto",
    ) -> Any:
        values = (
            train_X,
            train_Y,
            *(historical_Xs if historical_Xs is not None else []),
            *(historical_Ys if historical_Ys is not None else []),
            train_Yvar,
        )
        if _use_tensor_backend(values, backend):
            from botorch_rs.models.empirical_gps._torch_multitask import TensorTaskEmpiricalGP

            return TensorTaskEmpiricalGP.from_wide_format(
                train_X,
                train_Y,
                historical_Xs,
                historical_Ys,
                train_Yvar,
                output_tasks,
                correction,
            )
        x, xs = to_flat(train_X)
        y, ys = to_flat(train_Y)
        if (
            len(xs) != 2
            or xs[-1] != 1
            or len(ys) != 2
            or ys[0] != xs[0]
            or ys[1] != len(historical_Ys)
        ):
            raise ValueError("Wide training inputs and target dimensions do not match histories.")
        if train_Yvar is not None and _ops.shape_of(train_Yvar) != ys:
            raise ValueError("train_Yvar must match train_Y.")
        kind = kind_of(train_X)
        n, m = ys
        xx = from_flat([v for p in x for s in range(m) for v in (p, float(s))], (n * m, 2), kind)
        yy = from_flat(y, (n * m, 1), kind)
        vv = None if train_Yvar is None else _reshape(train_Yvar, (n * m, 1))
        return cls(
            xx, yy, 1, historical_Xs, historical_Ys, vv, output_tasks, correction, backend=backend
        )


__all__ = [
    "MultiTaskEmpiricalOneDimensionalGP",
    "MultiTaskEmpiricalOneDimensionalMean",
    "MultiTaskEmpiricalOneDimensionalKernel",
]
