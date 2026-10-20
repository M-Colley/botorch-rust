"""Base protocol shared by native Gaussian-process models.

The public name mirrors BoTorch's GPyTorchModel. Numerical inference is
implemented by the Rust-backed subclasses; this base does not wrap an
arbitrary Python GPyTorch ExactGP.
"""

from __future__ import annotations

import warnings
from typing import Any, Self

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import broadcast_shapes, broadcast_to, kind_of, to_flat
from botorch_rs._ops import dtype_name
from botorch_rs.exceptions.errors import (
    BotorchTensorDimensionError,
    InputDataError,
    UnsupportedError,
)
from botorch_rs.exceptions.warnings import BotorchTensorDimensionWarning, InputDataWarning
from botorch_rs.models.model import Model, ModelList
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior


class GPyTorchModel(Model):
    """Base protocol for native GP posterior, batch, and output interfaces."""

    @staticmethod
    def _validate_tensor_args(X: Any, Y: Any, Yvar: Any = None, strict: bool = True) -> None:
        xs, ys = to_flat(X, "X")[1], to_flat(Y, "Y")[1]
        if len(xs) != len(ys):
            if len(xs) - len(ys) == 1 and xs[:-1] == ys:
                message = (
                    "An explicit output dimension is required for targets."
                    f" Expected Y with dimension {len(xs)} (got Y.dim()={len(ys)})."
                )
            else:
                message = (
                    "Expected X and Y to have the same number of dimensions"
                    f" (got X with dimension {len(xs)} and Y with dimension {len(ys)})."
                )
            if strict:
                raise BotorchTensorDimensionError(message)
            warnings.warn(
                "Non-strict enforcement of botorch tensor conventions. The following "
                "error would have been raised with strict enforcement: " + message,
                BotorchTensorDimensionWarning,
                stacklevel=2,
            )
        if Yvar is not None:
            vs = to_flat(Yvar, "Yvar")[1]
            if ys[-len(vs) :] != vs:
                raise BotorchTensorDimensionError(
                    "An explicit output dimension is required for observation noise."
                    f" Expected Yvar with shape: {ys[-len(vs) :]} (got {vs})."
                )
        xd, yd = dtype_name(X) or "float64", dtype_name(Y) or "float64"
        vd = (dtype_name(Yvar) or "float64") if Yvar is not None else None
        if xd != yd or (vd is not None and yd != vd):
            raise InputDataError(
                f"Expected all inputs to share the same dtype. Got {xd} for X, "
                f"{yd} for Y, and {vd} for Yvar."
            )
        if xd != "float64":
            warnings.warn(
                f"The model inputs are of type {xd}. It is strongly recommended to use "
                "double precision in BoTorch, as this improves both precision and stability "
                "and can help avoid numerical errors.",
                InputDataWarning,
                stacklevel=3,
            )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(to_flat(self.train_inputs[0], "train_inputs")[1][:-2])  # type: ignore[attr-defined]

    @property
    def num_outputs(self) -> int:
        return int(self._num_outputs)  # type: ignore[attr-defined]


class BatchedMultiOutputGPyTorchModel(GPyTorchModel):
    """Native independent-output GP posterior and dimension protocol."""

    _rs: Any
    _kind: Any

    @staticmethod
    def get_batch_dimensions(train_X: Any, train_Y: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
        r"""Get the raw batch shape and output-augmented batch shape of the inputs.

        Args:
            train_X: A ``n x d`` or ``batch_shape x n x d`` (batch mode) tensor of
                training features.
            train_Y: A ``n x m`` or ``batch_shape x n x m`` (batch mode) tensor of
                training observations.

        Returns:
            2-element tuple containing

            - The ``input_batch_shape``
            - The output-augmented batch shape: ``input_batch_shape x (m)``
        """
        input_batch_shape = tuple(to_flat(train_X, "train_X")[1][:-2])
        num_outputs = to_flat(train_Y, "train_Y")[1][-1]
        aug_batch_shape = input_batch_shape + ((num_outputs,) if num_outputs > 1 else ())
        return input_batch_shape, aug_batch_shape

    def _set_dimensions(self, train_X: Any, train_Y: Any) -> None:
        batch, aug = self.get_batch_dimensions(train_X, train_Y)
        outputs = to_flat(train_Y)[1][-1]
        if getattr(self, "_rs", None) is not None:
            if self.num_outputs != outputs or self.batch_shape != batch:
                raise UnsupportedError(
                    "A native GP's training dimensions cannot be changed in place."
                )
            return
        self.__dict__.update(_num_outputs=outputs, _input_batch_shape=batch, _aug_batch_shape=aug)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.__dict__["_input_batch_shape"])

    def _transform_tensor_args(self, X: Any, Y: Any, Yvar: Any = None) -> tuple[Any, Any, Any]:
        from botorch_rs import _ops
        from botorch_rs.models.utils.assorted import multioutput_to_batch_mode_transform

        if self.num_outputs > 1:
            return multioutput_to_batch_mode_transform(X, Y, self.num_outputs, Yvar)

        def squeeze(value: Any) -> Any:
            shape = to_flat(value)[1]
            return _ops.getitem(value, (Ellipsis, 0)) if shape[-1] == 1 else value

        return X, squeeze(Y), None if Yvar is None else squeeze(Yvar)

    def _posterior_args(
        self, output_indices: Any, posterior_transform: Any
    ) -> tuple[list[int] | None, tuple[list[float], float] | None]:
        from botorch_rs.acquisition.objective import transform_arg

        m = self.num_outputs
        oi = None
        if output_indices is not None:
            oi = [int(t) for t in output_indices]
            if not oi or any(not -m <= t < m for t in oi):
                raise BotorchTensorDimensionError(
                    f"output_indices {list(output_indices)} are invalid for a model with "
                    f"{m} outputs."
                )
            oi = [t % m for t in oi]
            if oi == list(range(m)):
                oi = None
        return oi, transform_arg(posterior_transform)

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """The posterior at ``X`` (``batch_shape x q x d``), in the original
        outcome space, over the outputs ``output_indices`` (all by default),
        scalarized by ``posterior_transform`` if given; with
        ``observation_noise=True`` the likelihood noise is added (the mean
        training noise for fixed-noise models)."""
        if getattr(self, "_rs", None) is None:
            raise NotImplementedError(
                "Posterior inference requires a native Rust GP backing model."
            )
        if not isinstance(observation_noise, bool):
            raise UnsupportedError("`observation_noise` must be a bool in botorch_rs.")
        from botorch_rs.acquisition.objective import ExpectationPosteriorTransform
        from botorch_rs.models.gp_regression import _model_posterior

        if isinstance(posterior_transform, ExpectationPosteriorTransform):
            posterior = self.posterior(X, output_indices, observation_noise, **kwargs)
            return posterior_transform(posterior, X=X)
        oi, transform = self._posterior_args(output_indices, posterior_transform)
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if len(flat[1]) < 2:
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d`, got shape {flat[1]}."
            )
        if flat[1][-1] != self._rs.dim:
            raise BotorchTensorDimensionError(
                f"Expected X with {self._rs.dim} columns (d), got {flat[1][-1]}."
            )
        self.eval()
        return _model_posterior(
            self._rs,
            flat,
            kind,
            observation_noise,
            num_outputs=self.num_outputs,
            model_batch_shape=self.batch_shape,
            output_indices=oi,
            transform=transform,
            points_per_input=self._rs.points_per_input,
            transformed=list(self._rs.transformed_outputs),
            is_list=False,
            query_inputs=X,
            parameter_owner=self,
        )


class ModelListGPyTorchModel(ModelList, GPyTorchModel):
    """Joint Gaussian posterior protocol for lists of native GP models."""

    def __init__(self, *models: Any) -> None:
        if not models:
            raise ValueError("A native GP list requires at least one model.")
        if any(getattr(model, "_rs", None) is None for model in models):
            raise UnsupportedError("GP list inference requires native Rust GP backing models.")
        super().__init__(*models)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        shapes = {tuple(model.batch_shape) for model in self.models}
        if len(shapes) == 1:
            return next(iter(shapes))
        shape = broadcast_shapes(*shapes)
        message = f"Component models of {type(self).__name__} have different batch shapes"
        if shape is None:
            raise NotImplementedError(message + " that are not broadcastable.")
        warnings.warn(message + ". Broadcasting batch shapes.", stacklevel=2)
        return shape

    @property
    def _all_single_task(self) -> bool:
        from botorch_rs.models.gp_regression import SingleTaskGP

        return all(isinstance(m, SingleTaskGP) for m in self.models)

    def _sync_torch_parameters(self) -> Any:
        """Collect actual child leaves; native list pullbacks preserve order."""
        from types import SimpleNamespace

        from botorch_rs.models.gp_regression import SingleTaskGP

        entries: list[tuple[str, Any]] = []
        error: str | None = None
        for i, model in enumerate(self.models):
            sync = getattr(model, "_sync_torch_parameters", None)
            state = None if sync is None else sync()
            if state is None or type(model) is not SingleTaskGP:
                error = (
                    "Joint ModelListGP parameter gradients require every child to be an "
                    "ordinary CPU Torch SingleTaskGP. Unsupported child: "
                    f"models.{i} ({type(model).__name__})."
                )
                if state is None:
                    continue
            entries.extend((f"models.{i}.{name}", leaf) for name, leaf in state.entries)
            error = error or state.gradient_error
        return SimpleNamespace(entries=entries, gradient_error=error) if entries else None

    def _rs_list(self) -> Any:
        """A list of the current snapshots of the sub-models (a generic list
        of model handles if a sub-model is not a ``SingleTaskGP``)."""
        for model in self.models:
            sync = getattr(model, "_sync_torch_parameters", None)
            if sync is not None:
                sync()
        if self._all_single_task:
            return _rs.ModelListGP([m._rs for m in self.models])
        return _rs.Model.from_models([m._acquisition_model(None) for m in self.models])

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        """The posterior at ``X`` (``batch_shape x q x d``) over the outputs
        ``output_indices`` (all by default), scalarized by
        ``posterior_transform`` if given, with the observation noise of every
        sub-model if ``observation_noise=True``. If a sub-model has a
        nonlinear outcome transform, a ``PosteriorList`` of the sub-models'
        posteriors (a posterior transform then scalarizes the marginal moments
        of a single point, as BoTorch's ``scalarize_posterior``)."""
        from botorch_rs.acquisition.objective import transform_arg

        if not isinstance(observation_noise, bool):
            raise UnsupportedError("`observation_noise` must be a bool in botorch_rs.")
        m = self.num_outputs
        oi = None
        if output_indices is not None:
            oi = [int(t) for t in output_indices]
            if not oi or any(not -m <= t < m for t in oi):
                raise BotorchTensorDimensionError(
                    f"output_indices {list(output_indices)} are invalid for a model with "
                    f"{m} outputs."
                )
            oi = [t % m for t in oi]
            if oi == list(range(m)):
                oi = None
        kind = kind_of(X)
        flat = to_flat(X, "X")
        d = self.models[0]._rs.dim
        if len(flat[1]) < 2 or flat[1][-1] != d:
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d` with d = {d}, got shape {flat[1]}."
            )
        self.eval()
        transform = transform_arg(posterior_transform)
        transformed = self._transformed_outputs()
        selected = transformed if oi is None else [transformed[t] for t in oi]
        if any(selected) and transform is None:
            # BoTorch's `ModelList.posterior`: a `PosteriorList` of the
            # sub-models' posteriors over their selected outputs.
            from botorch_rs.posteriors.posterior_list import PosteriorList

            posteriors = []
            for index, local in self._get_group_subset_indices(oi).items():
                sub = self.models[index]
                k = sub.num_outputs
                if local is None or local:
                    posteriors.append(
                        sub.posterior(
                            X,
                            output_indices=None
                            if local is None or local == list(range(k))
                            else local,
                            observation_noise=observation_noise,
                        )
                    )
            return PosteriorList(*posteriors)
        rs_list = self._rs_list()
        return GPyTorchPosterior(
            rs_list,
            flat,
            kind,
            observation_noise,
            num_outputs=m,
            model_batch_shape=self.batch_shape,
            output_indices=oi,
            transform=transform,
            points_per_input=rs_list.points_per_input,
            query_inputs=X,
            parameter_owner=self,
        )

    def _transformed_outputs(self) -> list[bool]:
        """Whether each output's posterior is a ``TransformedPosterior``."""
        out: list[bool] = []
        for sub in self.models:
            flags = getattr(sub._rs, "transformed_outputs", None)
            out.extend(list(flags) if flags is not None else [False] * sub.num_outputs)
        return out

    def train(self, mode: bool = True) -> Self:
        self.training = mode
        for m in self.models:
            m.train(mode)
        return self

    def _broadcast_mvns(self, mvns: list[Any]) -> list[Any]:
        """Broadcast native Gaussian moment views without Python GP inference."""
        from botorch_rs.posteriors.gpytorch import MultivariateNormal, _MomentPosterior

        if not mvns or any(not isinstance(mvn, MultivariateNormal) for mvn in mvns):
            raise UnsupportedError("_broadcast_mvns requires native Gaussian distributions.")
        batches = [tuple(mvn.batch_shape) for mvn in mvns]
        if len(set(batches)) == 1:
            return mvns
        model_shape = self.batch_shape
        rank = max(map(len, batches))
        insert_at = rank - len(model_shape)
        aligned = []
        for batch in batches:
            while len(batch) < rank:
                batch = (*batch[:insert_at], 1, *batch[insert_at:])
            aligned.append(batch)
        target = broadcast_shapes(*aligned)
        if target is None:
            raise RuntimeError("Gaussian distribution batch shapes are not broadcastable.")
        for i, (mvn, batch) in enumerate(zip(mvns, aligned, strict=True)):
            posterior = mvn._p
            (mean, _), (covariance, _) = posterior._get_joint()
            q, m = posterior._q, posterior._m
            new_mean = broadcast_to(mean, (*batch, q, m), (*target, q, m))
            new_covariance = broadcast_to(
                covariance, (*batch, q * m, q * m), (*target, q * m, q * m)
            )
            mean_shape = (*target, q) if m == 1 else (*target, q, m)
            expanded = _MomentPosterior(
                (new_mean, mean_shape),
                (new_covariance, (*target, q * m, q * m)),
                posterior._kind,
                num_outputs=m,
            )
            mvns[i] = expanded.distribution
        return mvns


class MultiTaskGPyTorchModel(GPyTorchModel):
    """Long-format task selection and native multitask posterior protocol."""

    _rs: Any
    _kind: Any
    _task_feature: int
    _output_tasks: list[int]
    num_non_task_features: int

    @property
    def train_targets(self) -> Any:
        raise NotImplementedError("Native multitask training targets are required.")

    def _native_task_source(self, tasks: list[int] | None, includes_task_feature: bool) -> Any:
        raise NotImplementedError("A native multitask posterior source is required.")

    def _check_task_values(self, values: list[float]) -> None:
        raise NotImplementedError("A native model must provide task-value validation.")

    def _extract_targets_and_noise(self) -> tuple[Any, Any]:
        from botorch_rs import _ops
        from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood

        targets = _ops.getitem(self.train_targets, (Ellipsis, None))
        likelihood = getattr(self, "likelihood", None)
        noise = None
        if isinstance(likelihood, FixedNoiseGaussianLikelihood):
            noise = _ops.getitem(likelihood.noise, (Ellipsis, None))
        return targets, noise

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""The posterior at ``X`` in the original outcome space.

        ``X`` (``batch_shape x q x d``) without the task column gives the
        posterior over the output tasks (``output_indices``: other task
        values), a ``MultitaskMultivariateNormal`` in BoTorch's output-major
        layout for several tasks. With the task column (``... x (d + 1)``),
        the posterior is single-output over the given tasks.
        ``observation_noise=True`` adds the noise level of each point's task
        (for fixed noise, the mean training noise of the task).
        """
        from botorch_rs.acquisition.objective import transform_arg

        if not isinstance(observation_noise, bool):
            raise NotImplementedError(
                "Passing a tensor of observations is not supported by MultiTaskGP."
            )
        kind = kind_of(X)
        flat = to_flat(X, "X")
        d = self.num_non_task_features
        if len(flat[1]) < 2 or flat[1][-1] not in (d, d + 1):
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d` with d = {d} (or d + 1 with the task "
                f"feature), got shape {tuple(flat[1])}."
            )
        if flat[1][-1] == d + 1:
            if output_indices is not None:
                raise ValueError("`output_indices` must be None when `X` includes task features.")
            cols = d + 1
            self._check_task_values(
                sorted({flat[0][i + self._task_feature] for i in range(0, len(flat[0]), cols)})
            )
            self.eval()
            hook = getattr(self, "_forward_pass", None)
            if hook is not None:
                hook()
            source = self._native_task_source(None, True)
            num_outputs = 1
        else:
            tasks = None if output_indices is None else [int(t) for t in output_indices]
            self._check_task_values([float(t) for t in (tasks or self._output_tasks)])
            self.eval()
            hook = getattr(self, "_forward_pass", None)
            if hook is not None:
                hook()
            source = self._native_task_source(tasks, False)
            num_outputs = len(tasks) if tasks is not None else self.num_outputs
        return GPyTorchPosterior(
            source,
            flat,
            kind,
            observation_noise,
            num_outputs=num_outputs,
            model_batch_shape=self.batch_shape,
            transform=transform_arg(posterior_transform),
            query_inputs=X,
        )

    def subset_output(self, idcs: list[int]) -> Any:
        raise UnsupportedError("Subsetting outputs is not supported by `MultiTaskGPyTorchModel`.")


__all__ = [
    "GPyTorchModel",
    "BatchedMultiOutputGPyTorchModel",
    "ModelListGPyTorchModel",
    "MultiTaskGPyTorchModel",
]
