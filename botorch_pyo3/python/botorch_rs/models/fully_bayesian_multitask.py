r"""Multi-task Gaussian Process Regression models with fully Bayesian
inference (``botorch.models.fully_bayesian_multitask``).

The hyperparameters (the SAAS data kernel, per-task means, the noise and the
latent task features of an ICM task covariance) are sampled with NUTS
(``fit_fully_bayesian_model_nuts``) from the potential energy of BoTorch's
NumPyro model (evaluated in Rust). Every MCMC sample is then a
``MultiTaskGP`` whose task covariance is ``L L^T`` for the Cholesky factor
``L`` of the Matérn-5/2 kernel matrix of the latent task features (BoTorch's
``IndexKernel`` with ``covar_factor = L`` and ``var = 0``); the posterior is
the mixture of their posteriors.

``backend="torch"`` (also selected automatically for CUDA inputs or data
requiring gradients) constructs device-resident correlated components and
uses the Torch-dispatched multinomial NUTS sampler. See
``docs/torch-fully-bayesian-multitask.md`` for supported compositions and
validation of data, parameter and query derivatives.
"""

from __future__ import annotations

import copy
import math
from typing import Any, NoReturn

from botorch_rs._arrays import broadcast_shapes, broadcast_to, from_flat, kind_of, to_flat
from botorch_rs.constraints import GreaterThan
from botorch_rs.exceptions.errors import InputDataError, UnsupportedError
from botorch_rs.kernels import IndexKernel, Kernel, LinearKernel, MaternKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.models import _numpyro
from botorch_rs.models.fully_bayesian import (
    MCMC_DIM,
    MIN_INFERRED_NOISE_LEVEL,
    SaasPyroModel,
    _median,
    _values,
    reshape_and_detach,
)
from botorch_rs.models.gp_regression import _validate_input_scaling
from botorch_rs.models.multitask import MultiTaskGP
from botorch_rs.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
    Warp,
)
from botorch_rs.models.transforms.outcome import OutcomeTransform
from botorch_rs.posteriors.fully_bayesian import _StackedMixturePosterior


class MultiTaskPyroMixin:
    r"""Mixin with universal multi-task logic for PyroModel subclasses.

    Stores task-related attributes (``task_feature``, ``num_tasks``,
    ``task_rank``) and adjusts ``ard_num_dims`` to exclude the task column.
    Overrides ``sample_mean`` to return per-task means and
    ``_prepare_features`` to strip the task column.

    Place before the ``PyroModel`` subclass in the MRO.
    """

    task_feature: int
    num_tasks: int
    task_rank: int

    def set_inputs(  # type: ignore[override]
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        task_feature: int | None = None,
        task_rank: int | None = None,
        all_tasks: list[int] | None = None,
    ) -> None:
        """Set training data and configure multi-task attributes.

        Args:
            train_X: Training inputs (n x (d + 1)), including a task column.
            train_Y: Training targets (n x 1).
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            task_feature: The index of the task feature column.
            task_rank: The number of learned task embeddings. Defaults to
                the number of tasks.
            all_tasks: A list of all task indices. If omitted, all tasks will be
                inferred from the task feature column of the training data.
        """
        super().set_inputs(train_X, train_Y, train_Yvar)  # type: ignore[misc]
        data, shape = to_flat(train_X, "train_X")
        cols = shape[-1]
        assert task_feature is not None
        task_feature = task_feature % cols
        self.task_feature = task_feature
        if all_tasks is None:
            all_tasks = sorted({int(data[i * cols + task_feature]) for i in range(shape[0])})
        self.num_tasks = len(all_tasks)
        self.task_rank = task_rank or self.num_tasks
        self.ard_num_dims = cols - 1  # type: ignore[attr-defined]

    def _task_structure(self) -> tuple[int, int, int] | None:
        return (self.task_feature, self.num_tasks, self.task_rank)

    def _warp_indices(self) -> list[int] | None:
        if not self.use_input_warping:  # type: ignore[attr-defined]
            return None
        cols = self.ard_num_dims + 1  # type: ignore[attr-defined]
        selected = self.indices  # type: ignore[attr-defined]
        indices = (
            [i for i in range(cols) if i != self.task_feature]
            if selected is None
            else [int(i) % cols if -cols <= int(i) < 0 else int(i) for i in selected]
        )
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(i < 0 or i >= cols or i == self.task_feature for i in indices)
        ):
            raise InputDataError("Warp indices must be unique non-task input columns.")
        return indices

    def sample_mean(self) -> Any:
        return _numpyro.mean(self, self.num_tasks)

    def _get_task_indices_and_base_idxr_jax(self) -> tuple[Any, Any]:
        jnp, _, _ = _numpyro.backend()
        values, shape = to_flat(self.train_X, "train_X")  # type: ignore[attr-defined]
        tasks_values = values[self.task_feature :: shape[-1]]
        if any(int(t) != t or t < 0 or t >= self.num_tasks for t in tasks_values):
            raise InputDataError(
                "Pyro task identifiers must be mapped to indices in [0, num_tasks). "
                "SaasFullyBayesianMultiTaskGP performs this mapping automatically."
            )
        base = jnp.asarray([i for i in range(self.ard_num_dims + 1) if i != self.task_feature])  # type: ignore[attr-defined]
        tasks = self.train_X_jax[..., self.task_feature].astype(jnp.int32)  # type: ignore[attr-defined]
        return tasks, base

    def _prepare_features(self, X: Any) -> Any:
        _, base = self._get_task_indices_and_base_idxr_jax()
        return X[..., base]


class LatentFeatureMultiTaskPyroMixin(MultiTaskPyroMixin):
    r"""Mixin that adds ICM-style multi-task capabilities via latent features.

    The task covariance is the Matérn-5/2 kernel of learned latent task
    features (``latent_features ~ N(0, 1)``, ``num_tasks x task_rank``) with
    lengthscales ``task_lengthscale ~ Gamma(6, 3)``; the means are per task.
    """

    def sample_latent_features(self) -> Any:
        return _numpyro.latent_features(self.num_tasks, self.task_rank)

    def sample_task_lengthscale(self, concentration: float = 6.0, rate: float = 3.0) -> Any:
        return _numpyro.task_lengthscale(self.task_rank, concentration, rate)

    def _build_task_covar(self) -> tuple[Any, Any]:
        tasks, _ = self._get_task_indices_and_base_idxr_jax()
        features = self.sample_latent_features()[tasks]
        return _numpyro.kernel(features, self.sample_task_lengthscale(), "matern52"), tasks

    def _maybe_multitask_transform(self, K_noiseless: Any, mean: Any) -> tuple[Any, Any]:
        task_covar, tasks = self._build_task_covar()
        return K_noiseless * task_covar, mean[tasks]

    def _member_modules(self, mcmc_samples: dict[str, Any], s: int) -> dict[str, Any]:
        """The hyperparameters of sample ``s`` of a multi-task model."""
        means, _ = _values(mcmc_samples, "mean")
        t = self.num_tasks
        latent, lshape = _values(mcmc_samples, "latent_features")
        r = lshape[-1]
        task_ls, _ = _values(mcmc_samples, "task_lengthscale")
        return {
            "mean": means[s * t : (s + 1) * t],
            "latent_features": latent[s * t * r : (s + 1) * t * r],
            "task_lengthscale": task_ls[s * r : (s + 1) * r],
            "rank": r,
        }

    def get_dummy_mcmc_samples(self, num_mcmc_samples: int, **tkwargs: Any) -> dict[str, Any]:
        """Return dummy MCMC samples for state dict loading."""
        mcmc_samples = super().get_dummy_mcmc_samples(  # type: ignore[misc]
            num_mcmc_samples=num_mcmc_samples, **tkwargs
        )
        ones = self._ones  # type: ignore[attr-defined]
        mcmc_samples["mean"] = ones((num_mcmc_samples, self.num_tasks), tkwargs)
        mcmc_samples["task_lengthscale"] = ones((num_mcmc_samples, self.task_rank), tkwargs)
        mcmc_samples["latent_features"] = ones(
            (num_mcmc_samples, self.num_tasks, self.task_rank), tkwargs
        )
        return mcmc_samples


class MultitaskSaasPyroModel(LatentFeatureMultiTaskPyroMixin, SaasPyroModel):
    r"""
    Multi-task SAAS model. Backward-compatible subclass that composes
    ``LatentFeatureMultiTaskPyroMixin`` with ``SaasPyroModel``.
    """


def _mcmc_slice(value: Any, batch: tuple[int, ...], sample: int) -> Any:
    """Broadcast full event arrays, then remove their final MCMC batch axis."""
    if value is None:
        return None
    data, shape = to_flat(value)
    if len(shape) < 2:
        raise ValueError("MCMC inputs and observations require at least two event dimensions.")
    expanded = broadcast_to(data, tuple(shape), (*batch, *shape[-2:]))
    width = math.prod(shape[-2:])
    count = batch[-1]
    selected = [
        v
        for outer in range(math.prod(batch[:-1]))
        for v in expanded[(outer * count + sample) * width : (outer * count + sample + 1) * width]
    ]
    return from_flat(selected, (*batch[:-1], *shape[-2:]), kind_of(value))


def _stack_mcmc_events(values: list[Any]) -> Any:
    """Stack equal-shaped arrays before their trailing two event dimensions."""
    flats = [to_flat(value) for value in values]
    shape = tuple(flats[0][1])
    width = math.prod(shape[-2:])
    data = [
        v
        for outer in range(math.prod(shape[:-2]))
        for flat, _ in flats
        for v in flat[outer * width : (outer + 1) * width]
    ]
    return from_flat(data, (*shape[:-2], len(values), *shape[-2:]), kind_of(values[0]))


def _cholesky(a: list[list[float]]) -> list[list[float]]:
    """linear_operator's ``psd_safe_cholesky`` of a small dense matrix: the
    plain factor, else with a jitter of 1e-8, 1e-7, 1e-6 on the diagonal."""
    n = len(a)
    for jitter in (0.0, 1e-8, 1e-7, 1e-6):
        lo = [[0.0] * n for _ in range(n)]
        ok = True
        for j in range(n):
            d = a[j][j] + jitter - sum(lo[j][k] ** 2 for k in range(j))
            if not d > 0:
                ok = False
                break
            lo[j][j] = math.sqrt(d)
            for i in range(j + 1, n):
                lo[i][j] = (a[i][j] - sum(lo[i][k] * lo[j][k] for k in range(j))) / lo[j][j]
        if ok:
            return lo
    raise RuntimeError("The task covariance of an MCMC sample is not positive definite.")


class _SampleMultiTaskGP(MultiTaskGP):
    """An MCMC sample of a fully Bayesian multi-task GP: a ``MultiTaskGP``
    with an ``IndexKernel`` task covariance of the sample (and no
    adjustment of the task factors of unobserved tasks in ``eval()``)."""

    _eval_unobserved_tasks = False
    outcome_transform: Any

    def __init__(self, task_kernel: Kernel, **kwargs: Any) -> None:
        self._sample_task_kernel = task_kernel
        super().__init__(**kwargs)

    def _make_task_covar_module(
        self, num_tasks: int, rank: int, task_covar_prior: Any, task_feature: int
    ) -> Kernel:
        kernel = self._sample_task_kernel
        kernel.active_dims = (task_feature,)
        return kernel


class _MCMCInputWarp(Warp):
    """Fixed sampled concentrations with synchronized native member state."""

    def _set(self, i: int, value: Any) -> None:
        previous = self._steps
        super()._set(i, value)
        member = self.__dict__.get("_member")
        if member is not None:
            try:
                self._install(member)
            except Exception:
                self._steps = previous
                raise

    def _install(self, member: Any) -> None:
        member._rs.set_input_warp(
            list(self.indices),
            list(to_flat(self.concentration0)[0]),
            list(to_flat(self.concentration1)[0]),
            self._eps,
        )
        self._member = member


class SaasFullyBayesianMultiTaskGP(MultiTaskGP):
    r"""A fully Bayesian multi-task GP model with the SAAS prior.
    This model assumes that the inputs have been normalized to [0, 1]^d and that the
    output has been stratified standardized to have zero mean and unit variance for
    each task. The SAAS model [Eriksson2021saasbo]_ with a Matern-5/2 is used as data
    kernel by default.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model as it
    isn't compatible with ``fit_gpytorch_mll``.

    Example:
        >>> X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
        >>> i1, i2 = torch.zeros(10, 1), torch.ones(20, 1)
        >>> train_X = torch.cat([
        >>>     torch.cat([X1, i1], -1), torch.cat([X2, i2], -1),
        >>> ])
        >>> train_Y = torch.cat(f1(X1), f2(X2)).unsqueeze(-1)
        >>> train_Yvar = 0.01 * torch.ones_like(train_Y)
        >>> mtsaas_gp = SaasFullyBayesianMultiTaskGP(
        >>>     train_X, train_Y, train_Yvar, task_feature=-1,
        >>> )
        >>> fit_fully_bayesian_model_nuts(mtsaas_gp)
        >>> posterior = mtsaas_gp.posterior(test_X)

    In botorch_rs, the fitted model holds one ``MultiTaskGP`` per MCMC
    sample (``_samples``); its posterior is the mixture of theirs, and the
    acquisition functions average over them.
    """

    _is_fully_bayesian = True
    _is_ensemble = True
    outcome_transform: Any
    _fb_conditioned = False

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        from botorch_rs.models.map_saas import _use_torch_backend

        if cls is SaasFullyBayesianMultiTaskGP and _use_torch_backend(
            backend,
            train_X,
            train_Y,
            *args[1:2],
            kwargs.get("train_Yvar"),
            *(kwargs.get(name) for name in ("input_transform", "outcome_transform")),
        ):
            from botorch_rs.models._torch_backend_fully_bayesian_multitask import (
                TorchSaasFullyBayesianMultiTaskGP,
            )

            return object.__new__(TorchSaasFullyBayesianMultiTaskGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        train_Yvar: Any = None,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        pyro_model: MultitaskSaasPyroModel | None = None,
        validate_task_values: bool = True,
        *,
        backend: str = "auto",
    ) -> None:
        r"""Initialize the fully Bayesian multi-task GP model.

        Args:
            train_X: Training inputs (n x (d + 1))
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). If None, we infer the noise.
                Note that the inferred noise is common across all tasks.
            task_feature: The index of the task feature (``-d <= task_feature <= d``).
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            rank: The num of learned task embeddings to be used in the task kernel.
                If omitted, use a full rank (i.e. number of tasks) kernel.
            all_tasks: A list of all task indices. If omitted, all tasks will be
                inferred from the task feature column of the training data. Used to
                inform the model about the total number of tasks, including any
                unobserved tasks.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale).
                Note that ``.train()`` will be called on the outcome transform during
                instantiation of the model.
            input_transform: An input transform that is applied to the inputs ``X``
                in the model's forward pass.
            pyro_model: Optional ``PyroModel`` that has the same signature as
                ``MultitaskSaasPyroModel``. Defaults to ``MultitaskSaasPyroModel``.
            validate_task_values: If True, validate that the task values supplied in the
                input are expected tasks values. If false, unexpected task values
                will be mapped to the first output_task if supplied.

        botorch_rs supports ``Standardize`` as the outcome transform and
        ``Normalize`` as the common input transform (those of ``MultiTaskGP``).
        With input warping each MCMC member additionally applies its sampled
        Kumaraswamy transform on selected non-task columns.
        """
        x = to_flat(train_X, "train_X")
        y = to_flat(train_Y, "train_Y")
        if not (len(x[1]) == len(y[1]) == 2 and x[1][0] == y[1][0] and y[1][-1] == 1):
            raise ValueError("Expected train_X to have shape n x d and train_Y to have shape n x 1")
        if train_Yvar is not None and tuple(to_flat(train_Yvar, "train_Yvar")[1]) != tuple(y[1]):
            raise ValueError("Expected train_Yvar to be None or have the same shape as train_Y")
        from botorch_rs.models._fully_bayesian_outcomes import validate_native_outcome

        validate_native_outcome(outcome_transform)
        kind = kind_of(train_X)
        raw_Y = train_Y
        transformed_X = train_X
        if input_transform is not None:
            transformed_X = input_transform(train_X)
        Yvar = train_Yvar
        if outcome_transform is not None:
            outcome_transform.train()  # Ensure we learn parameters here on init
            train_Y, Yvar = outcome_transform(Y=train_Y, Yvar=train_Yvar, X=transformed_X)
        if Yvar is not None:  # Clamp after transforming
            vdata, vshape = to_flat(Yvar, "train_Yvar")
            Yvar = from_flat(
                [max(float(v), MIN_INFERRED_NOISE_LEVEL) for v in vdata], tuple(vshape), kind
            )
        if any(math.isnan(v) for v in list(x[0]) + list(to_flat(train_Y, "train_Y")[0])):
            raise InputDataError("Input data contains NaN values.")
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=Yvar,
            task_feature=task_feature,
            output_tasks=output_tasks,
            rank=rank,
            # We already transformed the data above, this avoids applying the
            # default ``Standardize`` transform twice.
            outcome_transform=None,
            all_tasks=all_tasks,
            validate_task_values=validate_task_values,
        )
        self.mean_module: Any = None
        self.covar_module: Any = None  # type: ignore[assignment]
        self.likelihood: Any = None  # type: ignore[assignment]
        self._samples: list[Any] | None = None
        self._fb_args = {
            "train_X": train_X,
            "raw_Y": raw_Y,
            "train_Y": train_Y,
            "train_Yvar": Yvar,
            "task_feature": task_feature,
            "output_tasks": output_tasks,
            "all_tasks": all_tasks,
            "validate_task_values": validate_task_values,
        }
        self._fb_outcome_transform = outcome_transform
        self._fb_input_transform = input_transform
        if pyro_model is None:
            pyro_model = MultitaskSaasPyroModel()
        # The task values mapped to task indices (BoTorch's `_map_tasks`).
        tx = to_flat(transformed_X, "train_X")
        cols = tx[1][-1]
        tf = self._task_feature
        mapped = list(tx[0])
        for i in range(tx[1][0]):
            mapped[i * cols + tf] = float(self._map_task_value(mapped[i * cols + tf]))
        pyro_model.set_inputs(
            train_X=from_flat(mapped, tuple(tx[1]), kind),
            train_Y=train_Y,
            train_Yvar=Yvar,
            task_feature=tf,
            task_rank=self._rank,
            all_tasks=all_tasks,
        )
        self.pyro_model: MultitaskSaasPyroModel = pyro_model
        if pyro_model.use_input_warping:
            pyro_model._warp_indices()
            if not math.isfinite(pyro_model._eps) or not 0 < pyro_model._eps < 0.5:
                raise ValueError("Warp eps must be finite and in (0, 0.5).")
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        y_tf = to_flat(train_Y, "train_Y")
        if y_tf[1][0] > 0:
            _validate_input_scaling(to_flat(transformed_X, "X"), y_tf, y_tf[1][0])

    def _map_task_value(self, value: float) -> int:
        """The index of a task value (BoTorch's ``_task_mapper``)."""
        if self._rs.tasks_are_indices:
            return int(value)
        return self.all_tasks.index(int(value))

    # -- fitted state --------------------------------------------------------

    def _check_if_fitted(self) -> None:
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    def train(  # type: ignore[override]
        self, mode: bool = True, reset: bool = True
    ) -> SaasFullyBayesianMultiTaskGP:
        r"""Puts the model in ``train`` mode.

        Args:
            mode: A boolean indicating whether to put the model in training mode.
            reset: A boolean indicating whether to reset the model to its initial
                state. If ``mode`` is False, this argument is ignored.

        Returns:
            The model itself.
        """
        self.training = mode
        if mode and reset:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None
            self._samples = None
        for name in ("input_transform", "outcome_transform"):
            tf = getattr(self, name, None)
            if tf is not None:
                tf.train(mode)
        return self

    def eval(self) -> SaasFullyBayesianMultiTaskGP:
        r"""Puts the model in eval mode (without the adjustment of the task
        factors of unobserved tasks, as BoTorch)."""
        self._check_if_fitted()
        return self.train(False)

    @property
    def median_lengthscale(self) -> Any:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        assert self._samples is not None
        covars: list[Any] = [m.covar_module for m in self._samples]
        per = [list(to_flat(c.kernels[0].base_kernel.lengthscale, "ls")[0]) for c in covars]
        d = len(per[0])
        out = [_median([p[k] for p in per]) for k in range(d)]
        return from_flat(out, (d,), self._kind)

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        assert self._samples is not None
        return len(self._samples)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""Batch shape of the model, equal to the number of MCMC samples.
        Note that ``SaasFullyBayesianMultiTaskGP`` does not support batching
        over input data at this point.
        """
        self._check_if_fitted()
        assert self._samples is not None
        return (*self._samples[0].batch_shape, len(self._samples))

    @property
    def train_inputs(self) -> tuple[Any]:
        if not getattr(self, "_fb_conditioned", False):
            return super().train_inputs
        assert self._samples is not None
        return (_stack_mcmc_events([member.train_inputs[0] for member in self._samples]),)

    @property
    def train_targets(self) -> Any:
        if not getattr(self, "_fb_conditioned", False):
            return super().train_targets
        assert self._samples is not None
        values = []
        for member in self._samples:
            data, shape = to_flat(member.train_targets)
            values.append(from_flat(data, (*shape, 1), self._kind))
        data, shape = to_flat(_stack_mcmc_events(values))
        return from_flat(data, tuple(shape[:-1]), self._kind)

    def fantasize(self, *args: Any, **kwargs: Any) -> NoReturn:  # type: ignore[override]
        raise NotImplementedError("Fantasize is not implemented!")

    def load_mcmc_samples(self, mcmc_samples: dict[str, Any]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by ``fit_fully_bayesian_model_nuts`` when the model
        has been fitted in order to create a batched MultiTaskGP model.
        """
        pm = self.pyro_model
        if pm._rs_kernel not in {"saas", "matern", "linear"}:
            raise UnsupportedError(
                "Native multitask loading requires a declared Matern, SAAS or Linear kernel family."
            )
        means, _ = _values(mcmc_samples, "mean")
        num = len(means) // pm.num_tasks
        linear = pm._rs_kernel == "linear"
        parameters, _ = _values(mcmc_samples, "weight_variance" if linear else "lengthscale")
        outputscale = (
            _values(mcmc_samples, "outputscale")[0]
            if "outputscale" in mcmc_samples and not linear
            else None
        )
        noise = None if pm.train_Yvar is not None else _values(mcmc_samples, "noise")[0]
        d = pm.ard_num_dims
        args = dict(self._fb_args)
        user_ot = self._fb_outcome_transform
        from botorch_rs.models._fully_bayesian_outcomes import native_outcome_affine

        affine = None if user_ot is None else native_outcome_affine(user_ot, self._kind)
        samples = []
        for s in range(num):
            mod = pm._member_modules(mcmc_samples, s)
            t, r = pm.num_tasks, mod["rank"]
            latent_kernel = MaternKernel(nu=2.5, ard_num_dims=r)
            latent_kernel.lengthscale = from_flat(mod["task_lengthscale"], (1, r), kind_of(None))
            latent = from_flat(mod["latent_features"], (t, r), kind_of(None))
            task_covar, _ = to_flat(latent_kernel(latent), "task_covar")
            chol = _cholesky([[float(task_covar[i * t + j]) for j in range(t)] for i in range(t)])
            task_kernel = IndexKernel(num_tasks=t, rank=t)
            task_kernel.covar_factor = from_flat(
                [v for row in chol for v in row], (t, t), kind_of(None)
            )
            # var = 0 (BoTorch: raw_var = -inf; softplus(-1000) = 0 exactly).
            task_kernel.raw_var = from_flat([-1000.0] * t, (t,), kind_of(None))
            base: Any = (
                LinearKernel(ard_num_dims=d) if linear else MaternKernel(nu=2.5, ard_num_dims=d)
            )
            setattr(
                base,
                "variance" if linear else "lengthscale",
                from_flat(parameters[s * d : (s + 1) * d], (1, d), kind_of(None)),
            )
            data_kernel = base
            if outputscale is not None:
                data_kernel = ScaleKernel(base)
                data_kernel.outputscale = from_flat([outputscale[s]], (), kind_of(None))
            likelihood = None
            if noise is not None:
                likelihood = GaussianLikelihood(
                    noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL)
                )
            member = _SampleMultiTaskGP(
                task_kernel=task_kernel,
                train_X=args["train_X"],
                train_Y=args["train_Y"],
                task_feature=args["task_feature"],
                train_Yvar=args["train_Yvar"],
                covar_module=data_kernel,
                likelihood=likelihood,
                output_tasks=args["output_tasks"],
                rank=t,
                all_tasks=args["all_tasks"],
                outcome_transform=None,
                input_transform=copy.deepcopy(self._fb_input_transform),
                validate_task_values=args["validate_task_values"],
            )
            if affine is not None:
                assert user_ot is not None
                member._rs._set_fitted_outcome_affine(*affine)
                member.outcome_transform = copy.deepcopy(user_ot).eval()
            if pm.use_input_warping:
                indices = pm._warp_indices()
                assert indices is not None
                k = len(indices)
                c0, cshape = _values(mcmc_samples, "c0")
                c1, c1shape = _values(mcmc_samples, "c1")
                if cshape != (num, k) or c1shape != (num, k):
                    raise ValueError(f"Expected sampled concentrations shaped {(num, k)}.")
                cols = d + 1
                warp = _MCMCInputWarp(
                    d=cols,
                    indices=indices,
                    eps=pm._eps,
                    bounds=from_flat([0.0] * cols + [1.0] * cols, (2, cols), self._kind),
                )
                warp.concentration0 = from_flat(c0[s * k : (s + 1) * k], (k,), self._kind)
                warp.concentration1 = from_flat(c1[s * k : (s + 1) * k], (k,), self._kind)
                warp._install(member)
                normalize = getattr(member, "input_transform", None)
                member.input_transform = (
                    warp
                    if normalize is None
                    else ChainedInputTransform(normalize=normalize, warp=warp)
                )
            if linear:
                cols = d + 1
                indices = [i for i in range(cols) if i != self._task_feature]
                member._rs.set_input_feature_offsets(indices, [-0.5] * d)
                center = Normalize(
                    d=cols,
                    indices=indices,
                    bounds=from_flat([0.5] * cols + [1.5] * cols, (2, cols), self._kind),
                )
                previous = getattr(member, "input_transform", None)
                member.input_transform = (
                    center
                    if previous is None
                    else ChainedInputTransform(previous=previous, center=center)
                )
            for k in range(t):
                member.mean_module.base_means[k].constant = mod["mean"][k]
            if noise is not None:
                lik: Any = member.likelihood
                lik.noise = max(noise[s], MIN_INFERRED_NOISE_LEVEL)
            member.eval()
            samples.append(member)
        self._samples = samples
        self.mean_module = [m.mean_module for m in samples]
        self.covar_module = [m.covar_module for m in samples]  # type: ignore[assignment]
        self.likelihood = [m.likelihood for m in samples]  # type: ignore[assignment]

    def forward(self, X: Any) -> Any:
        """The loaded prior at inputs including the categorical task column.

        Training mode applies the fitted input transformations. Evaluation
        mode accepts already transformed inputs, matching the public GP
        ``forward`` contract rather than computing a conditional posterior.
        """
        self._check_if_fitted()
        assert self._samples is not None
        kind = kind_of(X)
        if kind.lib == "torch":
            import torch

            from botorch_rs.models._torch_backend_modules import build_input_transform, build_kernel
            from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

            if X.ndim < 2 or X.shape[-1] != self.pyro_model.ard_num_dims + 1:
                raise ValueError("Prior inputs must include the task feature column.")
            batch = torch.broadcast_shapes(X.shape[:-2], self.batch_shape)
            expanded = X.expand(*batch, *X.shape[-2:])
            means, covariances = [], []
            for s, member in enumerate(self._samples):
                query = expanded.select(-3, s)
                transform = getattr(member, "input_transform", None)
                if self.training and transform is not None:

                    def prior_transform(source: Any) -> Any:
                        if isinstance(source, _MCMCInputWarp):
                            # The subclass only synchronizes native setters;
                            # its prior evaluation is exactly the public Warp.
                            result: Any = copy.copy(source)
                            result.__class__ = Warp
                            return result
                        if isinstance(source, ChainedInputTransform):
                            return ChainedInputTransform(
                                **{name: prior_transform(child) for name, child in source.items()}
                            )
                        return source

                    transform = prior_transform(transform)
                    module = build_input_transform(transform, X.shape[-1], X)
                    assert module is not None
                    query = module.train()(query)
                labels = query[..., self._task_feature]
                tasks = torch.zeros_like(labels, dtype=torch.long)
                valid = torch.zeros_like(labels, dtype=torch.bool)
                for i, label in enumerate(self.all_tasks):
                    match = labels == label
                    tasks = torch.where(match, i, tasks)
                    valid = valid | match
                if self._validate_task_values and not bool(valid.all()):
                    raise ValueError("Received invalid task labels; expected labels in all_tasks.")
                mapped = torch.cat(
                    (
                        query[..., : self._task_feature],
                        tasks.to(query).unsqueeze(-1),
                        query[..., self._task_feature + 1 :],
                    ),
                    dim=-1,
                )
                constants = torch.stack(
                    [
                        torch.as_tensor(mean.constant, dtype=query.dtype, device=query.device)
                        for mean in member.mean_module.base_means
                    ],
                    dim=-1,
                )
                constants = constants.expand(*query.shape[:-2], self.num_tasks)
                means.append(
                    constants.unsqueeze(-2)
                    .expand(*tasks.shape, self.num_tasks)
                    .gather(-1, tasks.unsqueeze(-1))
                    .squeeze(-1)
                )
                kernel = build_kernel(member.covar_module, X.shape[-1], member.batch_shape, X)
                covariances.append(kernel(mapped, mapped))
            return TorchGPyTorchPosterior(
                torch.stack(means, dim=-2).unsqueeze(-1), torch.stack(covariances, dim=-3)
            ).distribution

        from botorch_rs.posteriors.gpytorch import _MomentPosterior

        _, shape = to_flat(X, "X")
        if len(shape) < 2 or shape[-1] != self.pyro_model.ard_num_dims + 1:
            raise ValueError("Prior inputs must include the task feature column.")
        batch = broadcast_shapes(tuple(shape[:-2]), self.batch_shape)
        if batch is None:
            raise RuntimeError("Query and MCMC batch dimensions do not broadcast.")
        q, cols = shape[-2:]
        component_means, component_covariances = [], []
        for s, member in enumerate(self._samples):
            query = _mcmc_slice(X, batch, s)
            if self.training:
                query = member.transform_inputs(query)
            values = list(to_flat(query, "X")[0])
            flat_constants = []
            for mean in member.mean_module.base_means:
                constant_data, constant_shape = to_flat(mean.constant)
                flat_constants.append(
                    broadcast_to(constant_data, (*constant_shape, 1), (*batch[:-1], q))
                )
            flat_means = []
            for row in range(math.prod(batch[:-1]) * q):
                task = self._map_task_value(values[row * cols + self._task_feature])
                flat_means.append(flat_constants[task][row])
                values[row * cols + self._task_feature] = float(task)
            mapped = from_flat(values, (*batch[:-1], q, cols), kind)
            component_means.append(from_flat(flat_means, (*batch[:-1], q, 1), kind))
            component_covariances.append(member.covar_module(mapped))
        mean_data, mean_shape = to_flat(_stack_mcmc_events(component_means))
        covariance_data, covariance_shape = to_flat(_stack_mcmc_events(component_covariances))
        return _MomentPosterior(
            (mean_data, tuple(mean_shape[:-1])), (covariance_data, tuple(covariance_shape)), kind
        ).distribution

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        r"""Computes the posterior over model outputs at the provided points.

        Returns:
            A ``GaussianMixturePosterior`` object (the mixture of the samples'
            posteriors along ``MCMC_DIM``). Includes observation noise if
            specified.
        """
        self._check_if_fitted()
        assert self._samples is not None
        posteriors = [
            m.posterior(
                X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )
            for m in self._samples
        ]
        return _StackedMixturePosterior(posteriors, kind_of(X))

    def condition_on_observations(  # type: ignore[override]
        self, X: Any, Y: Any, noise: Any = None, **kwargs: Any
    ) -> SaasFullyBayesianMultiTaskGP:
        """Conditions on additional observations for a Fully Bayesian model (either
        identical across models or unique per-model).

        Args:
            X: A ``batch_shape x num_samples x d``-dim Tensor, where ``d`` is
                the dimension of the feature space and ``batch_shape`` is the number of
                sampled models.
            Y: A ``batch_shape x num_samples x 1``-dim Tensor, where ``d`` is
                the dimension of the feature space and ``batch_shape`` is the number of
                sampled models.

        Returns:
            A fully bayesian model conditioned on given observations (every
            MCMC sample on its observations).
        """
        self._check_if_fitted()
        assert self._samples is not None
        shapes = [tuple(to_flat(value)[1]) for value in (X, Y, noise) if value is not None]
        if any(len(shape) < 2 for shape in shapes):
            raise ValueError("Conditioning data require shapes (...,n,d) and (...,n,1).")
        batch = self.batch_shape
        for shape in shapes:
            expanded = broadcast_shapes(batch, shape[:-2])
            if expanded is None:
                raise ValueError("Conditioning and MCMC batch dimensions do not broadcast.")
            batch = expanded
        new = copy.copy(self)
        new._samples = [
            m.condition_on_observations(
                _mcmc_slice(X, batch, s),
                _mcmc_slice(Y, batch, s),
                noise=_mcmc_slice(noise, batch, s),
                **kwargs,
            )
            for s, m in enumerate(self._samples)
        ]
        new.mean_module = [m.mean_module for m in new._samples]
        new.covar_module = [m.covar_module for m in new._samples]
        new.likelihood = [m.likelihood for m in new._samples]
        new._fb_conditioned = True
        new.training = False
        return new

    # -- ensembles -----------------------------------------------------------

    def _ensemble_members(self) -> list[MultiTaskGP]:
        """The MCMC samples (``MultiTaskGP``s)."""
        self._check_if_fitted()
        assert self._samples is not None
        return list(self._samples)

    def _ensemble_acquisition(self, sampler: Any) -> tuple[list[Any], list[float] | None, Any]:
        """The members, their (equal) weights and the sampler shared by the
        members' acquisition functions."""
        from botorch_rs.acquisition.monte_carlo import DEFAULT_MC_SAMPLES
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(DEFAULT_MC_SAMPLES,))
        return self._ensemble_members(), None, sampler

    def __repr__(self) -> str:
        fitted = "unfitted" if self.covar_module is None else f"{self.num_mcmc_samples} samples"
        return f"SaasFullyBayesianMultiTaskGP(num_tasks={self.num_tasks}, {fitted})"


__all__ = [
    "MCMC_DIM",
    "LatentFeatureMultiTaskPyroMixin",
    "MultiTaskPyroMixin",
    "MultitaskSaasPyroModel",
    "SaasFullyBayesianMultiTaskGP",
    "reshape_and_detach",
]
