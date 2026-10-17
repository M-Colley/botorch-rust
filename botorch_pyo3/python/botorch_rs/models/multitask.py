"""Multi-task GPs (``botorch.models.multitask``): ``MultiTaskGP`` and
``KroneckerMultiTaskGP``.

References:

* E. Bonilla, K. Chai and C. Williams. Multi-task Gaussian Process Prediction.
  NeurIPS 2007.
* K. Swersky, J. Snoek and R. Adams. Multi-Task Bayesian Optimization.
  NeurIPS 2013.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import broadcast_to, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.kernels import AdditiveKernel, Kernel, ProductKernel
from botorch_rs.likelihoods import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    HadamardGaussianLikelihood,
    Likelihood,
)
from botorch_rs.means import ConstantMean, MultitaskMean
from botorch_rs.models.model import Model
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior
from botorch_rs.utils.types import DEFAULT


def _is_default(value: Any) -> bool:
    return value is DEFAULT or type(value).__name__ == "_DefaultType"


def _int_list(values: Any, name: str) -> list[int] | None:
    if values is None:
        return None
    data, shape = to_flat(values, name)
    if len(shape) > 1:
        raise BotorchTensorDimensionError(f"`{name}` must be a list of task values.")
    return [int(v) for v in data]


class MultiTaskGP(Model):
    r"""Multi-task exact GP with an ICM (intrinsic coregionalization model)
    kernel (BoTorch's ``MultiTaskGP``), see Bonilla et al. (2007) and Swersky
    et al. (2013).

    The training data are in "long format": one column of ``train_X``
    (``task_feature``) holds the task of every observation. The covariance is
    ``covar_module = data_covar_module * task_covar_module``: a kernel on the
    non-task features (``covar_module``, by default BoTorch's
    dimension-scaled RBF kernel, acting on the non-task columns unless it has
    ``active_dims`` of its own) times a ``PositiveIndexKernel(num_tasks,
    rank)`` on the task column with the task prior ``task_covar_prior``
    (default ``BetaPrior(2.5, 1.5)`` on the task correlations). Every task has
    its own constant mean (``MultitaskMean``) and, unless ``train_Yvar`` is
    given, its own noise level (``HadamardGaussianLikelihood``). Outcomes are
    standardized over all rows by default.

    ``posterior(X)`` for ``X`` without the task column (``batch_shape x q x
    d``) is over the output tasks (``output_tasks``, or the task values
    ``output_indices``), a ``MultitaskMultivariateNormal`` with correlated
    outputs; with the task column (``... x (d + 1)``) it is single-output over
    the given tasks.

    Differences from BoTorch (see the deviations in the documentation):

    * per-task noise levels are looked up by task index (BoTorch reads the
      raw task value, meta-pytorch/botorch#3396);
    * the ``active_dims`` of a ``ProductKernel`` ``covar_module`` apply to its
      sub-kernels, and ``eval()`` finds the task kernel as the last sub-kernel
      of ``covar_module`` (BoTorch: ``kernels[1]``);
    * an ``LKJCovariancePrior`` task prior acts on the task covariance matrix
      (BoTorch evaluates it on the correlations and fails);
    * acquisition functions take candidates without the task column (over the
      output tasks); ``qKnowledgeGradient`` is not supported.

    Example:
        >>> X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
        >>> i1, i2 = torch.zeros(10, 1), torch.ones(20, 1)
        >>> train_X = torch.cat([torch.cat([X1, i1], -1), torch.cat([X2, i2], -1)])
        >>> train_Y = torch.cat([f1(X1), f2(X2)]).unsqueeze(-1)
        >>> model = MultiTaskGP(train_X, train_Y, task_feature=-1)
    """

    _supports_batched_models = False
    _supports_cache_root = False

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        train_Yvar: Any = None,
        mean_module: Any = None,
        covar_module: Kernel | None = None,
        likelihood: Likelihood | None = None,
        task_covar_prior: Any = DEFAULT,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        validate_task_values: bool = True,
    ) -> None:
        from botorch_rs.models.gp_regression import _validate_tensor_args
        from botorch_rs.models.kernels.positive_index import PositiveIndexKernel
        from botorch_rs.models.transforms.input import Normalize
        from botorch_rs.models.utils.gpytorch_modules import (
            get_covar_module_with_dim_scaled_prior,
        )
        from botorch_rs.models.utils.priors import BetaPrior

        kind = kind_of(train_X)
        x = to_flat(train_X, "train_X")
        y = to_flat(train_Y, "train_Y")
        yvar = to_flat(train_Yvar, "train_Yvar") if train_Yvar is not None else None
        if yvar is not None and len(yvar[1]) == len(y[1]) - 1:
            yvar = (yvar[0], (*yvar[1], 1))
        _validate_tensor_args(x[1], y[1], yvar[1] if yvar is not None else None)
        if len(x[1]) != 2:
            # Batch mode multi-task GPs are blocked in GPyTorch.
            raise ValueError(f"Unsupported shape {tuple(x[1])} for train_X.")
        cols = x[1][-1]
        d = cols - 1
        # Input transform (a Normalize of all columns or of `indices`).
        it_rs, it_indices = None, None
        if input_transform is not None:
            if not isinstance(input_transform, Normalize):
                raise UnsupportedError(
                    "`input_transform` of a MultiTaskGP must be a botorch_rs Normalize."
                )
            if input_transform.batch_shape:
                raise UnsupportedError("A Normalize with a batch shape is not supported.")
            it_rs = input_transform._rs
            if input_transform.indices is not None:
                if input_transform._d != cols:
                    raise BotorchTensorDimensionError(
                        f"The input transform is for d = {input_transform._d} inputs, but "
                        f"train_X has {cols} columns."
                    )
                it_indices = list(input_transform.indices)
        # The task feature (shifted to be non-negative, as BoTorch does).
        if task_feature < 0:
            task_feature += cols
        all_tasks_inferred, tf, _ = self.get_all_tasks(train_X, task_feature, output_tasks)
        task_values = _int_list(all_tasks, "all_tasks")
        num_tasks = len(task_values) if task_values else len(all_tasks_inferred)
        # Outcome transform.
        if _is_default(outcome_transform):
            ot_obj: Standardize | None = Standardize(m=1)
            ot_rs, std_default = None, True
        elif outcome_transform is None:
            ot_obj, ot_rs, std_default = None, None, False
        elif isinstance(outcome_transform, Standardize):
            if outcome_transform._m != 1:
                raise BotorchTensorDimensionError(
                    f"The outcome transform is for m = {outcome_transform._m} outputs, got m = 1."
                )
            ot_obj, ot_rs, std_default = outcome_transform, outcome_transform._rs, False
        else:
            raise UnsupportedError(
                "`outcome_transform` must be a Standardize, DEFAULT or None (StratifiedStandardize "
                "is not ported yet)."
            )
        # Likelihood.
        lik_spec, lik_kind = None, "hadamard"
        if likelihood is not None:
            if yvar is not None:
                raise UnsupportedError(
                    "A custom likelihood cannot be combined with `train_Yvar` in botorch_rs."
                )
            if isinstance(likelihood, HadamardGaussianLikelihood):
                if likelihood.num_tasks != num_tasks:
                    raise BotorchTensorDimensionError(
                        f"The likelihood has {likelihood.num_tasks} tasks, the model {num_tasks}."
                    )
                lik_spec, lik_kind = likelihood._spec, "hadamard"
            elif isinstance(likelihood, GaussianLikelihood):
                if likelihood.batch_shape:
                    raise UnsupportedError("A likelihood with a batch shape is not supported.")
                lik_spec, lik_kind = likelihood._spec, "gaussian"
            else:
                raise UnsupportedError(
                    "`likelihood` must be a botorch_rs HadamardGaussianLikelihood or "
                    "GaussianLikelihood."
                )
            if likelihood._model is not None:
                raise UnsupportedError(
                    "This likelihood is already part of another model; create a new one."
                )
        # Mean.
        mean_kind = "multitask"
        if mean_module is not None:
            if not isinstance(mean_module, ConstantMean):
                raise UnsupportedError(
                    "`mean_module` must be None (a MultitaskMean of ConstantMeans) or a "
                    "botorch_rs ConstantMean."
                )
            if mean_module.batch_shape:
                raise UnsupportedError("A mean module with a batch shape is not supported.")
            mean_kind = "constant"
        # Covariance modules: the data kernel on the non-task columns and the
        # task kernel (BoTorch's PositiveIndexKernel, which draws its initial
        # values from torch's generator).
        base_idxr = [c for c in range(cols) if c != tf]
        if covar_module is None:
            data_kernel = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=d, active_dims=base_idxr
            )
        else:
            if not isinstance(covar_module, Kernel):
                raise UnsupportedError(
                    "`covar_module` must be a botorch_rs kernel (from botorch_rs.kernels or "
                    "botorch_rs.models.kernels)."
                )
            if covar_module._model is not None:
                raise UnsupportedError(
                    "This kernel is already part of another model; create a new one for each model."
                )
            if covar_module.batch_shape:
                raise UnsupportedError("A covariance module with a batch shape is not supported.")
            data_kernel = covar_module
            if data_kernel.active_dims is None:
                data_kernel.active_dims = tuple(base_idxr)
        if isinstance(data_kernel, (AdditiveKernel, ProductKernel)):
            data_node = data_kernel._configure(data_kernel._base_node())
            data_dims = None if data_kernel.active_dims is None else list(data_kernel.active_dims)
        else:
            data_node = data_kernel._node()
            data_dims = None
        if _is_default(task_covar_prior):
            task_covar_prior = BetaPrior(concentration1=2.5, concentration0=1.5)
        self._rank = int(rank) if rank is not None else num_tasks
        task_kernel = PositiveIndexKernel(
            num_tasks=num_tasks,
            rank=self._rank,
            task_prior=task_covar_prior,
            active_dims=[tf],
        )
        self._rs = _rs.MultiTaskGP(
            x,
            y,
            int(task_feature),
            yvar,
            data_node,
            data_dims,
            task_kernel._node(),
            True,
            lik_spec,
            lik_kind,
            mean_kind,
            _int_list(output_tasks, "output_tasks"),
            self._rank,
            task_values,
            ot_rs,
            std_default,
            it_rs,
            it_indices,
            bool(validate_task_values),
        )
        self._kind = kind
        rs = self._rs
        # BoTorch's attributes.
        self.num_tasks: int = rs.num_tasks
        self.all_tasks: list[int] = list(rs.all_tasks)
        self._output_tasks: list[int] = list(rs.output_tasks)
        self._num_outputs: int = len(self._output_tasks)
        self._task_feature: int = rs.task_feature
        self.num_non_task_features: int = rs.dim
        self._base_idxr: list[int] = base_idxr
        observed = set(rs.observed_task_indices)
        self._observed_task_indices = sorted(observed)
        self._unobserved_task_indices = [k for k in range(self.num_tasks) if k not in observed]
        self._expected_task_values = set(self.all_tasks)
        self._validate_task_values = bool(validate_task_values)
        # Modules bound to the model's hyperparameters.
        if yvar is not None:
            self.likelihood: Likelihood = FixedNoiseGaussianLikelihood._create(())
            self.likelihood._bind(self)
        elif likelihood is None:
            spec = rs.likelihood_spec()
            assert spec is not None
            self.likelihood = HadamardGaussianLikelihood._create(
                self, spec, self.num_tasks, self._task_feature
            )
        else:
            likelihood._bind(self)
            self.likelihood = likelihood
        if mean_module is None:
            self.mean_module: Any = MultitaskMean._create(self, self.num_tasks)
        else:
            mean_module._bind(self)
            self.mean_module = mean_module
        covar = data_kernel * task_kernel
        covar._bind(self, "covar_module.")
        self.covar_module: Kernel = covar
        if input_transform is not None:
            fitted = rs.input_transform(input_transform.learn_bounds)
            assert fitted is not None
            input_transform._set_fitted(fitted, kind)
            self.input_transform: Any = input_transform
        if ot_obj is not None:
            fitted_ot = rs.outcome_transform()
            assert fitted_ot is not None
            ot_obj._set_fitted(fitted_ot, kind_of(train_Y))
            self.outcome_transform = ot_obj
        self.training = True

    # -- tasks -------------------------------------------------------------

    @classmethod
    def get_all_tasks(
        cls,
        train_X: Any,
        task_feature: int,
        output_tasks: list[int] | None = None,
    ) -> tuple[list[int], int, int]:
        """The sorted task values of ``train_X``, the (non-negative) task
        feature and the number of non-task features (BoTorch's
        ``get_all_tasks``)."""
        data, shape = to_flat(train_X, "train_X")
        if len(shape) != 2:
            raise ValueError(f"Unsupported shape {tuple(shape)} for train_X.")
        d = shape[-1] - 1
        if not (-d <= task_feature <= d):
            raise ValueError(f"Must have that -{d} <= task_feature <= {d}")
        task_feature = task_feature % (d + 1)
        n, c = shape
        all_tasks = sorted({int(data[i * c + task_feature]) for i in range(n)})
        return all_tasks, task_feature, d

    def _check_task_values(self, values: list[float]) -> None:
        """BoTorch's ``_map_tasks`` validation (``validate_task_values=True``)."""
        if not self._validate_task_values:
            return
        longs = {int(v) for v in values}
        if self._rs.tasks_are_indices:
            if not all(0 <= v < self.num_tasks for v in values):
                raise ValueError(
                    "Expected all task features in `X` to be between 0 and "
                    f"self.num_tasks - 1. Got {values}."
                )
        elif longs - self._expected_task_values:
            raise ValueError(
                "Received invalid raw task values. Expected raw value to be in "
                f"{self._expected_task_values}, but got unexpected task values: "
                f"{longs - self._expected_task_values}."
            )

    # -- data --------------------------------------------------------------

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape (non-empty for fantasy models)."""
        return tuple(self._rs.batch_shape)

    @property
    def _input_batch_shape(self) -> tuple[int, ...]:
        return self.batch_shape

    @property
    def _aug_batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def train_inputs(self) -> tuple[Any]:
        """``(train_X,)`` including the task column, in the original space."""
        data, shape = self._rs.train_inputs()
        return (from_flat(data, tuple(shape), self._kind),)

    @property
    def train_targets(self) -> Any:
        """The training targets after the outcome transform (``n``)."""
        data, shape = self._rs.transformed_train_targets()
        return from_flat(data, tuple(shape), self._kind)

    # -- modes -------------------------------------------------------------

    def eval(self) -> MultiTaskGP:
        """Evaluation mode. As in BoTorch, the rows of the task kernel's
        ``covar_factor`` of tasks without training data become the mean of
        the observed tasks' rows (``posterior`` calls ``eval()``)."""
        if self._unobserved_task_indices:
            self._rs.eval_unobserved()
        return self.train(False)

    def train(self, mode: bool = True) -> MultiTaskGP:
        self.training = mode
        for name in ("input_transform", "outcome_transform"):
            tf = getattr(self, name, None)
            if tf is not None:
                tf.train(mode)
        return self

    # -- posterior ---------------------------------------------------------

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
            source = _rs.multitask_task_inputs_model(self._rs)
            num_outputs = 1
        else:
            tasks = None if output_indices is None else [int(t) for t in output_indices]
            self._check_task_values([float(t) for t in (tasks or self._output_tasks)])
            self.eval()
            source = _rs.multitask_model(self._rs, tasks)
            num_outputs = len(tasks) if tasks is not None else self._num_outputs
        return GPyTorchPosterior(
            source,
            flat,
            kind,
            observation_noise,
            num_outputs=num_outputs,
            model_batch_shape=self.batch_shape,
            transform=transform_arg(posterior_transform),
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle that acquisition functions are built on (over the
        output tasks; candidates without the task column)."""
        from botorch_rs.acquisition.objective import transform_arg

        self.eval()
        return _rs.multitask_model(
            self._rs, None, transform_arg(posterior_transform, self.num_outputs)
        )

    def subset_output(self, idcs: list[int]) -> Any:
        raise UnsupportedError("Subsetting outputs is not supported by `MultiTaskGPyTorchModel`.")

    # -- conditioning and fantasies ------------------------------------------

    def _check_full_inputs(self, flat: Any, what: str) -> None:
        d = self.num_non_task_features
        if len(flat[1]) < 2 or flat[1][-1] != d + 1:
            raise BotorchTensorDimensionError(
                f"{what} of a MultiTaskGP needs inputs with the task feature (d + 1 = {d + 1} "
                f"columns), got shape {tuple(flat[1])}."
            )

    def _with_rs(self, rs: _rs.MultiTaskGP) -> MultiTaskGP:
        """A copy of this model (with its own modules) around the Rust model
        ``rs``, in eval mode (e.g. a conditioned or fantasy model)."""
        new = self.__class__.__new__(self.__class__)
        memo: dict[int, Any] = {id(self): new}
        for key, value in self.__dict__.items():
            new.__dict__[key] = rs if key == "_rs" else copy.deepcopy(value, memo)
        new.training = False
        return new

    def condition_on_observations(
        self, X: Any, Y: Any, noise: Any = None, **kwargs: Any
    ) -> MultiTaskGP:
        r"""The model conditioned on new observations (BoTorch's
        ``condition_on_observations``): ``X`` (``batch_shape x n' x (d + 1)``,
        with the task feature), ``Y`` (``batch_shape' x n' x 1``, original
        outcome space) and optional observation-noise variances ``noise``
        (shape of ``Y``, in the *outcome-transformed* space, as BoTorch
        assumes; required for a fixed-noise model). The result has the
        broadcast batch shape and the same hyperparameters. Explicit noise of
        a model with learned noise enters the posterior mean and covariance
        (meta-pytorch/botorch#3396; BoTorch main uses it for the covariance
        only).

        As in BoTorch, the model must be in eval mode (e.g. after a call of
        ``posterior``).
        """
        if kwargs:
            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
        if self.training:
            raise RuntimeError(
                "Fantasy observations can only be added after making predictions with a model "
                "so that all test independent caches exist. Call the model on some data first!"
            )
        x = to_flat(X, "X")
        self._check_full_inputs(x, "condition_on_observations")
        y = to_flat(Y, "Y")
        if len(y[1]) == len(x[1]) - 1:
            y = (y[0], (*y[1], 1))
        if y[1][-1] != 1:
            raise BotorchTensorDimensionError(
                f"Expected Y with one output (`... x n' x 1`), got shape {tuple(y[1])}."
            )
        nz = None
        if noise is not None:
            nz = to_flat(noise, "noise")
            if tuple(nz[1]) == tuple(y[1][:-1]):
                # (`batch_shape x n'`: without the output dimension.)
                nz = (nz[0], (*nz[1], 1))
        return self._with_rs(self._rs.condition_on_observations(x, y, nz))

    def fantasize(
        self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any
    ) -> MultiTaskGP:
        r"""A fantasy model (BoTorch's ``fantasize``): ``n_f`` sets of
        observations at ``X`` (``batch_shape x n' x (d + 1)``, with the task
        feature) are sampled from the posterior with observation noise using
        ``sampler`` (a ``SobolQMCNormalSampler`` with ``sample_shape=(n_f,)``;
        its base samples are shared by the batch elements), and the model is
        conditioned on each of them. The fantasy model has the batch shape
        ``n_f x batch_shape``. As in BoTorch, ``observation_noise`` tensors and
        fixed-noise models are not supported (``NotImplementedError``)."""
        from botorch_rs.exceptions.errors import DeprecationError
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        if kwargs:
            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
        if isinstance(observation_noise, bool):
            raise DeprecationError(
                "`fantasize` no longer accepts a boolean for `observation_noise`."
            )
        if not isinstance(sampler, SobolQMCNormalSampler):
            raise UnsupportedError("`sampler` must be a botorch_rs SobolQMCNormalSampler.")
        x = to_flat(X, "X")
        d = self.num_non_task_features
        if len(x[1]) >= 2 and x[1][-1] == d:
            raise RuntimeError(
                "MultiTaskGP.fantasize needs inputs with the task feature (d + 1 = "
                f"{d + 1} columns), got shape {tuple(x[1])} (BoTorch fails as well)."
            )
        self._check_full_inputs(x, "fantasize")
        if observation_noise is not None:
            raise NotImplementedError(
                "Passing a tensor of observations is not supported by MultiTaskGP."
            )
        self.eval()
        return self._with_rs(self._rs.fantasize(x, sampler._rs, None))

    # -- construct_inputs ----------------------------------------------------

    @classmethod
    def construct_inputs(
        cls,
        training_data: Any,
        task_feature: int,
        output_tasks: list[int] | None = None,
        task_covar_prior: Any = DEFAULT,
        prior_config: dict[str, Any] | None = None,
        rank: int | None = None,
        map_heterogeneous_to_full: bool = False,
    ) -> dict[str, Any]:
        r"""Construct ``Model`` keyword arguments from a dataset (BoTorch's
        ``MultiTaskGP.construct_inputs``).

        ``training_data`` is a ``SupervisedDataset`` or ``MultiTaskDataset``
        (BoTorch's classes, or any object with ``X``, ``Y`` and ``Yvar`` --
        and ``datasets``, ``task_feature_index`` for a multi-task dataset).
        ``prior_config={"use_LKJ_prior": True, "eta": ...}`` gives an
        ``LKJCovariancePrior(num_tasks, eta, GammaPrior(1.0, 0.15))`` with
        ``num_tasks`` the number of task values in ``X``'s task column
        (BoTorch counts the distinct values of the row ``X[task_feature]``).
        """
        from botorch_rs.priors import GammaPrior, LKJCovariancePrior

        if (
            not _is_default(task_covar_prior)
            and task_covar_prior is not None
            and prior_config is not None
        ):
            raise ValueError(
                "Only one of `task_covar_prior` and `prior_config` arguments expected."
            )
        is_multi_task = hasattr(training_data, "datasets") and hasattr(
            training_data, "task_feature_index"
        )
        if prior_config is not None:
            if not prior_config.get("use_LKJ_prior"):
                raise ValueError("Currently only config for LKJ prior is supported.")
            data, shape = to_flat(training_data.X, "X")
            c = shape[-1]
            num_tasks = len({data[i + task_feature % c] for i in range(0, len(data), c)})
            sd_prior = GammaPrior(1.0, 0.15)
            eta = prior_config.get("eta", 0.5)
            if not isinstance(eta, float) and not isinstance(eta, int):
                raise ValueError(f"eta must be a real number, your eta was {eta}.")
            task_covar_prior = LKJCovariancePrior(num_tasks, eta, sd_prior)
        if (
            map_heterogeneous_to_full
            and is_multi_task
            and getattr(training_data, "has_heterogeneous_features", False)
        ):
            all_datasets, feature_indices, full_dim = (
                training_data.get_heterogeneous_feature_mapping()
            )
            rows: list[list[float]] = []
            ys: list[float] = []
            yvars: list[float] | None = []
            for task_idx, (ds, fi) in enumerate(zip(all_datasets, feature_indices, strict=True)):
                xd, xs = to_flat(ds.X, "X")
                k = xs[-1]
                fi_list = [int(v) for v in to_flat(fi, "feature_indices")[0]]
                for i in range(0, len(xd), k):
                    row = [0.0] * (full_dim + 1)
                    for j, col in enumerate(fi_list):
                        row[col] = xd[i + j]
                    row[-1] = float(task_idx)
                    rows.append(row)
                ys.extend(to_flat(ds.Y, "Y")[0])
                if ds.Yvar is None:
                    yvars = None
                elif yvars is not None:
                    yvars.extend(to_flat(ds.Yvar, "Yvar")[0])
            kind = kind_of(all_datasets[0].X)
            base_inputs: dict[str, Any] = {
                "train_X": from_flat([v for r in rows for v in r], (len(rows), full_dim + 1), kind),
                "train_Y": from_flat(ys, (len(ys), 1), kind),
            }
            if yvars is not None and all_datasets[0].Yvar is not None:
                base_inputs["train_Yvar"] = from_flat(yvars, (len(yvars), 1), kind)
            base_inputs["task_feature"] = -1
            base_inputs["all_tasks"] = list(range(len(all_datasets)))
            base_inputs["output_tasks"] = output_tasks
            if not _is_default(task_covar_prior):
                base_inputs["task_covar_prior"] = task_covar_prior
            if rank is not None:
                base_inputs["rank"] = rank
            return base_inputs
        if not all(hasattr(training_data, a) for a in ("X", "Y", "Yvar")):
            raise TypeError(
                "Expected `training_data` to be a `SupervisedDataset`, but got "
                f"{type(training_data)}."
            )
        base_inputs = {"train_X": training_data.X, "train_Y": training_data.Y}
        if training_data.Yvar is not None:
            base_inputs["train_Yvar"] = training_data.Yvar
        if is_multi_task and training_data.task_feature_index is None:
            base_inputs["all_tasks"] = list(range(len(training_data.datasets)))
        if not _is_default(task_covar_prior):
            base_inputs["task_covar_prior"] = task_covar_prior
        if rank is not None:
            base_inputs["rank"] = rank
        base_inputs["task_feature"] = task_feature
        base_inputs["output_tasks"] = output_tasks
        return base_inputs

    # -- parameters --------------------------------------------------------

    def _param_index(self, suffix: str) -> tuple[str, int, int]:
        """The full name, offset and size of the parameter named ``suffix``
        (a full name, or the end of one)."""
        names = self._rs.parameter_names()
        for exact in (True, False):
            offset = 0
            for name, size in names:
                if name == suffix if exact else name.endswith(suffix):
                    return name, offset, size
                offset += size
        raise AttributeError(f"The model has no parameter {suffix!r}.")

    def _raw_param(self, suffix: str) -> list[float]:
        _, offset, size = self._param_index(suffix)
        return self._rs.raw_parameters()[offset : offset + size]

    def _raw_param_shape(self, suffix: str) -> tuple[int, ...]:
        name, _, _ = self._param_index(suffix)
        return tuple(dict(self._rs.parameter_shapes())[name])

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        data, shape = to_flat(value, suffix)
        name, _, size = self._param_index(suffix)
        target = tuple(dict(self._rs.parameter_shapes())[name])
        try:
            new = broadcast_to(data, shape, target)
        except ValueError:
            # GPyTorch's `Module.initialize`: `expand_as`, else `view_as`.
            if len(data) != size:
                raise BotorchTensorDimensionError(
                    f"{name} has shape {target}, got a value of shape {tuple(shape)}."
                ) from None
            new = [float(v) for v in data]
        self._rs.set_raw_parameter(name, new)

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of every hyperparameter, with BoTorch's names,
        order and shapes."""
        values = self._rs.raw_parameters()
        offset = 0
        for name, shape in self._rs.parameter_shapes():
            size = math.prod(shape)
            data = values[offset : offset + size]
            offset += size
            yield name, from_flat(data, tuple(shape), self._kind)

    def parameters(self) -> Iterator[Any]:
        for _, value in self.named_parameters():
            yield value

    def initialize(self, **kwargs: Any) -> MultiTaskGP:
        """Sets hyperparameters by (dotted) name, like GPyTorch's
        ``Module.initialize``, e.g. ``model.initialize(**{"likelihood.noise":
        1e-3, "covar_module.kernels.0.lengthscale": 0.5})``; raw names (those
        of ``named_parameters()``) work too."""
        for name, value in kwargs.items():
            obj: Any = self
            *path, attr = name.split(".")
            for part in path:
                if isinstance(obj, list) and part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            if not isinstance(getattr(type(obj), attr, None), property):
                raise AttributeError(f"Unknown hyperparameter {name!r}.")
            setattr(obj, attr, value)
        return self

    # -- copies ------------------------------------------------------------

    def __deepcopy__(self, memo: dict[int, Any]) -> MultiTaskGP:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_rs":
                new.__dict__[key] = value.copy()
            else:
                new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def __repr__(self) -> str:
        extra = f", batch_shape={self.batch_shape}" if self.batch_shape else ""
        return (
            f"MultiTaskGP(n={self._rs.num_train}, d={self.num_non_task_features}, "
            f"num_tasks={self.num_tasks}, output_tasks={self._output_tasks}, "
            f"rank={self._rank}{extra})"
        )


class KroneckerMultiTaskGP(Model):
    r"""Multi-task GP with Kronecker structure, using an ICM kernel (BoTorch's
    ``KroneckerMultiTaskGP``), for the "block design": all ``m`` tasks
    (the columns of ``train_Y``) are observed at all ``n`` training points.

    The covariance of tasks ``s, t`` at points ``x, x'`` is ``k_x(x, x') B[s,
    t]`` (``covar_module = MultitaskKernel(data_covar_module, m, rank,
    task_covar_prior)``): a data kernel (by default BoTorch's
    dimension-scaled RBF kernel) and an ``IndexKernel`` with the task prior
    (by default ``LKJCovariancePrior(m, eta, sd_prior)`` with ``eta=1.5`` and
    ``sd_prior=SmoothedBoxPrior(exp(-6), exp(1.25), 0.05)``, both keyword
    arguments). The noise is a ``MultitaskGaussianLikelihood`` (per-task and
    global noise, or a task noise covariance of rank ``likelihood_rank``),
    the mean one constant per task. There is no outcome transform by default.

    ``posterior(X)`` is a ``MultitaskGPPosterior`` over all tasks: the exact
    posterior means and variances, and samples by Matheron's rule (see
    ``botorch_rs.posteriors.multitask``). Acquisition functions use the exact
    joint posterior.

    Differences from BoTorch: the posterior variances are exact also for
    task-dependent noise (BoTorch's closed form assumes homoskedastic noise,
    fixed in M-Colley/botorch ``fix/models-posteriors``); ``output_indices``
    other than all outputs and tensor ``observation_noise`` raise
    ``NotImplementedError`` (BoTorch main ignores the former); the
    ``active_dims`` of ``data_covar_module`` apply (GPyTorch's
    ``MultitaskKernel`` calls the data kernel's ``forward`` and ignores
    them); batched training data are not supported.

    Example:
        >>> train_X = torch.rand(10, 2)
        >>> train_Y = torch.cat([f_1(train_X), f_2(train_X)], dim=-1)
        >>> model = KroneckerMultiTaskGP(train_X, train_Y)
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        likelihood: Any = None,
        data_covar_module: Kernel | None = None,
        task_covar_prior: Any = None,
        rank: int | None = None,
        outcome_transform: Any = None,
        input_transform: Any = None,
        **kwargs: Any,
    ) -> None:
        import math as _math

        from botorch_rs.kernels import MultitaskKernel
        from botorch_rs.likelihoods import MultitaskGaussianLikelihood
        from botorch_rs.models.gp_regression import _validate_tensor_args
        from botorch_rs.models.transforms.input import Normalize
        from botorch_rs.models.utils.gpytorch_modules import (
            get_covar_module_with_dim_scaled_prior,
        )
        from botorch_rs.priors import LKJCovariancePrior, Prior, SmoothedBoxPrior

        kind = kind_of(train_X)
        x = to_flat(train_X, "train_X")
        y = to_flat(train_Y, "train_Y")
        unknown = set(kwargs) - {"eta", "sd_prior", "likelihood_rank"}
        if unknown:
            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(unknown)}.")
        it_rs = None
        if input_transform is not None:
            if not isinstance(input_transform, Normalize) or input_transform.indices is not None:
                raise UnsupportedError(
                    "`input_transform` of a KroneckerMultiTaskGP must be a botorch_rs Normalize "
                    "of all inputs."
                )
            if input_transform.batch_shape:
                raise UnsupportedError("A Normalize with a batch shape is not supported.")
            it_rs = input_transform._rs
        ot_rs = None
        if outcome_transform is not None:
            if not isinstance(outcome_transform, Standardize):
                raise UnsupportedError(
                    "`outcome_transform` must be a botorch_rs Standardize or None."
                )
            ot_rs = outcome_transform._rs
        _validate_tensor_args(x[1], y[1], None)
        if len(x[1]) != 2:
            raise UnsupportedError(
                "Batched training data are not supported by KroneckerMultiTaskGP in botorch_rs."
            )
        d, m = x[1][-1], y[1][-1]
        if outcome_transform is not None and outcome_transform._m != m:
            raise BotorchTensorDimensionError(
                f"The outcome transform is for m = {outcome_transform._m} outputs, got m = {m}."
            )
        self._rank = int(rank) if rank is not None else m
        likelihood_rank = int(kwargs.get("likelihood_rank", 0))
        if likelihood is None:
            from botorch_rs.models.utils.gpytorch_modules import (
                get_gaussian_likelihood_with_lognormal_prior,
            )

            spec = get_gaussian_likelihood_with_lognormal_prior()._spec
            likelihood = MultitaskGaussianLikelihood._from_spec(spec, m, likelihood_rank)
        elif not isinstance(likelihood, MultitaskGaussianLikelihood):
            raise UnsupportedError("`likelihood` must be a botorch_rs MultitaskGaussianLikelihood.")
        elif likelihood.num_tasks != m:
            raise BotorchTensorDimensionError(
                f"The likelihood has {likelihood.num_tasks} tasks, the model {m}."
            )
        else:
            likelihood_rank = likelihood.rank
        if task_covar_prior is None:
            eta = kwargs.get("eta", 1.5)
            sd_prior = kwargs.get(
                "sd_prior", SmoothedBoxPrior(_math.exp(-6), _math.exp(1.25), 0.05)
            )
            task_covar_prior = LKJCovariancePrior(n=m, eta=eta, sd_prior=sd_prior)
        elif not isinstance(task_covar_prior, Prior):
            raise TypeError(
                f"Expected gpytorch.priors.Prior but got {type(task_covar_prior).__name__}"
            )
        if data_covar_module is None:
            data_covar_module = get_covar_module_with_dim_scaled_prior(ard_num_dims=d)
        elif not isinstance(data_covar_module, Kernel):
            raise UnsupportedError("`data_covar_module` must be a botorch_rs kernel.")
        elif data_covar_module._model is not None:
            raise UnsupportedError(
                "This kernel is already part of another model; create a new one for each model."
            )
        covar = MultitaskKernel(
            data_covar_module=data_covar_module,
            num_tasks=m,
            rank=self._rank,
            task_covar_prior=task_covar_prior,
        )
        self._rs = _rs.KroneckerMultiTaskGP(
            x,
            y,
            data_covar_module._node(),
            covar.task_covar_module._node(),
            likelihood._spec,
            likelihood_rank,
            self._rank,
            ot_rs,
            it_rs,
        )
        self._kind = kind
        self._num_outputs = m
        likelihood._bind(self)
        self.likelihood: Any = likelihood
        self.mean_module = MultitaskMean._create(self, m)
        covar._bind(self, "covar_module.")
        self.covar_module: Kernel = covar
        if input_transform is not None:
            fitted = self._rs.input_transform(input_transform.learn_bounds)
            assert fitted is not None
            input_transform._set_fitted(fitted, kind)
            self.input_transform: Any = input_transform
        if outcome_transform is not None:
            fitted_ot = self._rs.outcome_transform()
            assert fitted_ot is not None
            outcome_transform._set_fitted(fitted_ot, kind_of(train_Y))
            self.outcome_transform = outcome_transform
        self.training = True

    # -- data --------------------------------------------------------------

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def _input_batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def _aug_batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def train_inputs(self) -> tuple[Any]:
        """``(train_X,)`` in the original input space."""
        data, shape = self._rs.train_inputs()
        return (from_flat(data, tuple(shape), self._kind),)

    @property
    def train_targets(self) -> Any:
        """The training targets after the outcome transform (``n x m``)."""
        data, shape = self._rs.transformed_train_targets()
        return from_flat(data, tuple(shape), self._kind)

    def train(self, mode: bool = True) -> KroneckerMultiTaskGP:
        self.training = mode
        for name in ("input_transform", "outcome_transform"):
            tf = getattr(self, name, None)
            if tf is not None:
                tf.train(mode)
        return self

    # -- posterior ---------------------------------------------------------

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        r"""The posterior over all tasks at ``X`` (``batch_shape x q x d``):
        a ``MultitaskGPPosterior`` (see the class documentation)."""
        from botorch_rs.posteriors.multitask import MultitaskGPPosterior

        self.eval()
        if posterior_transform is not None:
            raise NotImplementedError(
                f"Posterior transforms currently not supported for {self.__class__.__name__}"
            )
        if output_indices is not None and list(output_indices) != list(range(self.num_outputs)):
            raise NotImplementedError(
                "Subsetting the outputs via `output_indices` is currently not "
                f"supported for {self.__class__.__name__}."
            )
        if not isinstance(observation_noise, bool):
            raise NotImplementedError(
                "Passing a tensor of observation noise is currently not supported "
                f"for {self.__class__.__name__}."
            )
        kind = kind_of(X)
        flat = to_flat(X, "X")
        d = self._rs.dim
        if len(flat[1]) < 2 or flat[1][-1] != d:
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d` with d = {d}, got shape {tuple(flat[1])}."
            )
        ot = getattr(self, "outcome_transform", None)
        scales = None
        if ot is not None:
            scales = [float(v) for v in to_flat(ot.stdvs, "stdvs")[0]]
        return MultitaskGPPosterior(
            _rs.kronecker_model(self._rs),
            self._rs,
            flat,
            kind,
            observation_noise,
            num_outputs=self.num_outputs,
            num_train=self._rs.num_train,
            scales=scales,
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        """The model handle that acquisition functions are built on (the exact
        joint posterior over all tasks)."""
        if posterior_transform is not None:
            raise NotImplementedError(
                f"Posterior transforms currently not supported for {self.__class__.__name__}"
            )
        self.eval()
        return _rs.kronecker_model(self._rs)

    def subset_output(self, idcs: list[int]) -> Any:
        raise UnsupportedError(
            f"{self.__class__.__name__}.subset_output is not supported (BoTorch does not "
            "implement it)."
        )

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        """As in BoTorch, which never creates the prediction strategy that
        GPyTorch's fantasy models need (its posterior is custom), a
        ``RuntimeError``."""
        raise RuntimeError(
            "Fantasy observations can only be added after making predictions with a model "
            "so that all test independent caches exist. Call the model on some data first!"
        )

    def fantasize(self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any) -> Any:
        """As in BoTorch (see ``condition_on_observations``), a
        ``RuntimeError``."""
        return self.condition_on_observations(X, None)

    # -- parameters --------------------------------------------------------

    _param_index = MultiTaskGP._param_index
    _raw_param = MultiTaskGP._raw_param
    _raw_param_shape = MultiTaskGP._raw_param_shape
    _set_raw_param = MultiTaskGP._set_raw_param
    named_parameters = MultiTaskGP.named_parameters
    parameters = MultiTaskGP.parameters
    initialize = MultiTaskGP.initialize  # type: ignore[assignment]

    def __deepcopy__(self, memo: dict[int, Any]) -> KroneckerMultiTaskGP:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_rs":
                new.__dict__[key] = value.copy()
            else:
                new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def __repr__(self) -> str:
        return (
            f"KroneckerMultiTaskGP(n={self._rs.num_train}, d={self._rs.dim}, "
            f"num_tasks={self.num_outputs}, rank={self._rank})"
        )


__all__ = ["KroneckerMultiTaskGP", "MultiTaskGP"]
