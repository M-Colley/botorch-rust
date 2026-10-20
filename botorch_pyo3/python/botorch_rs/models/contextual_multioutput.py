"""The latent context embedding multi-output GP
(``botorch.models.contextual_multioutput``): ``LCEMGP``.

References: Q. Feng, B. Letham, H. Mao and E. Bakshy. High-Dimensional
Contextual Policy Search with Unknown Context Rewards using Bayesian
Optimization. NeurIPS 2020.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs
from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, ArrayKind, Flat, from_flat, kind_of, to_flat
from botorch_rs.constraints import Interval
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.kernels import Kernel, RBFKernel
from botorch_rs.likelihoods import HadamardGaussianLikelihood
from botorch_rs.means import MultitaskMean
from botorch_rs.models.gp_regression import _is_default, _validate_tensor_args
from botorch_rs.models.multitask import MultiTaskGP
from botorch_rs.models.transforms.outcome import Standardize, is_plain_standardize
from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch_rs.utils.types import DEFAULT


class _KernelSlot:
    """Presents one kernel of an ``LCEMGP`` (the data kernel ``"data"`` or the
    task kernel ``"task"``) to the kernel classes, which read and set their
    hyperparameters through the model they are bound to."""

    aug_batch_shape: list[int] = []

    def __init__(self, model: LCEMGP, which: str, prefix: str) -> None:
        self._model = model
        self._which = which
        self._prefix = prefix

    @property
    def _kind(self) -> ArrayKind:
        return self._model._kind

    @property
    def _rs(self) -> _KernelSlot:
        return self

    @property
    def lengthscale(self) -> list[float]:
        return self._model._rs.lengthscale_of(self._which)

    def set_lengthscale(self, value: Flat) -> None:
        self._model._rs.set_lengthscale_of(self._which, list(value[0]))

    @property
    def outputscale(self) -> list[float] | None:
        v = self._model._rs.outputscale_of(self._which)
        return None if v is None else [v]

    def set_outputscale(self, value: Flat) -> None:
        if len(value[0]) != 1:
            raise BotorchTensorDimensionError(
                f"The outputscale is a scalar, got a value of shape {tuple(value[1])}."
            )
        self._model._rs.set_outputscale_of(self._which, value[0][0])

    def _full_name(self, suffix: str) -> str:
        for name, _ in self._model._rs.parameter_names():
            if name.startswith(self._prefix + ".") and name.endswith(suffix):
                return name
        raise AttributeError(f"The kernel has no parameter {suffix!r}.")

    def _raw_param(self, suffix: str) -> list[float]:
        return self._model._raw_param(self._full_name(suffix))

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        self._model._set_raw_param(self._full_name(suffix), value)

    def __deepcopy__(self, memo: dict[int, Any]) -> _KernelSlot:
        return _KernelSlot(copy.deepcopy(self._model, memo), self._which, self._prefix)


class Embedding:
    """An embedding layer of an ``LCEMGP``
    (``torch.nn.Embedding(num_embeddings, embedding_dim, max_norm=1.0)``).

    ``weight`` reads and sets the model's weights. Like torch, looking up
    rows (calling the layer, or any forward pass of the model such as
    ``posterior``) renormalizes the looked-up rows of the weight in place to
    a norm of at most ``max_norm``.
    """

    max_norm = 1.0
    norm_type = 2.0

    def __init__(self, model: LCEMGP, index: int, num_embeddings: int, embedding_dim: int) -> None:
        self._model = model
        self._index = index
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    @property
    def weight(self) -> Any:
        """The weight (``num_embeddings x embedding_dim``)."""
        data, shape = self._model._rs.embedding_weights()[self._index]
        return from_flat(data, tuple(shape), self._model._kind)

    @weight.setter
    def weight(self, value: Any) -> None:
        self._model._set_raw_param(f"emb_layers.{self._index}.weight", value)

    def __call__(self, indices: Any) -> Any:
        """The (renormalized) rows ``indices``; renormalizes the looked-up
        rows of the model's weight in place, like torch."""
        self._model._forward_pass()
        data, shape = self._model._rs.embedding_weights()[self._index]
        idx, ishape = to_flat(indices, "indices")
        rows = [int(i) for i in idx]
        dim = shape[1]
        out = [data[r * dim + j] for r in rows for j in range(dim)]
        return from_flat(out, (*ishape, dim), kind_of(indices))

    def __deepcopy__(self, memo: dict[int, Any]) -> Embedding:
        return Embedding(
            copy.deepcopy(self._model, memo), self._index, self.num_embeddings, self.embedding_dim
        )

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim}, max_norm=1.0)"


def _int_list(values: Any, name: str) -> list[int] | None:
    if values is None:
        return None
    data, shape = to_flat(values, name)
    if len(shape) > 1:
        raise BotorchTensorDimensionError(f"`{name}` must be a list of task values.")
    return [int(v) for v in data]


def _matrix_arg(value: Any, name: str) -> Flat | None:
    if value is None:
        return None
    flat = to_flat(value, name)
    if len(flat[1]) != 2:
        raise BotorchTensorDimensionError(
            f"`{name}` must be a 2-dimensional tensor (one row per context), got shape "
            f"{tuple(flat[1])}."
        )
    return flat


class LCEMGP(MultiTaskGP):
    r"""The multi-task GP with the latent context embedding multi-output
    (LCE-M) kernel (BoTorch's ``LCEMGP``).

    The covariance of two observations ``(x, t)`` and ``(x', t')`` is
    ``k_x(x, x') * k_e(E_t, E_t')``: a data kernel (``covar_module``,
    BoTorch's dimension-scaled RBF kernel) times an RBF kernel on context
    embeddings (``task_covar_module_base``, lengthscales constrained to
    ``Interval(0, 2)`` without transform, i.e. optimizer box bounds). The
    embedding of a context concatenates learned embeddings (one
    ``Embedding(num_values, dim, max_norm=1.0)`` per column of
    ``context_cat_feature``) and the fixed ``context_emb_feature`` row. The
    model has per-task constant means and noise levels and standardizes
    the outcomes over all rows.

    ``posterior(X)`` for ``X`` without the task column (``batch_shape x q x
    d``) is over ``output_tasks``; with the task column it is over the given
    tasks. Tasks in ``all_tasks`` without training data are supported.

    Differences from BoTorch (the behavior the Unity project gives
    ``LCEMGP``, which fixes two BoTorch problems):

    * ``eval()`` (also called by ``posterior``) sets the constant mean of
      every task without training data to the average of the observed tasks'
      constants; BoTorch's ``MultiTaskGP.eval()`` instead initializes
      ``IndexKernel`` factors, which raises an ``AttributeError`` for
      ``LCEMGP`` with tasks without data, and an unobserved constant would
      otherwise stay at its initial 0. Observed tasks are unaffected.
    * The default task kernel has one lengthscale per embedding dimension,
      learned plus fixed. BoTorch sizes it for the learned dimensions only,
      which fails with a ``context_emb_feature``.

    Learned embeddings are initialized from ``N(0, 1)`` with botorch_rs's
    generator, seeded with the keyword-only ``seed`` (random if ``None``);
    BoTorch draws them from torch's global generator. Set them with
    ``model.emb_layers[i].weight = ...`` or
    ``model.initialize(**{"emb_layers.0.weight": ...})``.

    Example:
        >>> model = LCEMGP(train_X, train_Y, task_feature=-1,
        ...                context_emb_feature=embeddings, output_tasks=[2],
        ...                all_tasks=[0, 1, 2])
        >>> fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        >>> posterior = model.posterior(test_X)  # over task 2
    """

    _rs: _botorch_rs.LCEMGP  # type: ignore[assignment]

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        from botorch_rs.models.map_saas import _use_torch_backend

        if cls is LCEMGP and _use_torch_backend(backend, train_X, train_Y, *args, *kwargs.values()):
            from botorch_rs.models._torch_backend_lcem import TorchLCEMGP

            return object.__new__(TorchLCEMGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        train_Yvar: Any = None,
        mean_module: Any = None,
        covar_module: Kernel | None = None,
        likelihood: Any = None,
        context_cat_feature: Any = None,
        context_emb_feature: Any = None,
        embs_dim_list: list[int] | None = None,
        output_tasks: list[int] | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        seed: int | None = None,
        backend: str = "auto",
    ) -> None:
        self._init(
            train_X,
            train_Y,
            task_feature,
            train_Yvar,
            mean_module,
            covar_module,
            likelihood,
            context_cat_feature,
            context_emb_feature,
            embs_dim_list,
            output_tasks,
            all_tasks,
            outcome_transform,
            input_transform,
            seed,
            provided_embedding=False,
        )

    def _init(
        self,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        train_Yvar: Any,
        mean_module: Any,
        covar_module: Kernel | None,
        likelihood: Any,
        context_cat_feature: Any,
        context_emb_feature: Any,
        embs_dim_list: list[int] | None,
        output_tasks: list[int] | None,
        all_tasks: list[int] | None,
        outcome_transform: Any,
        input_transform: Any,
        seed: int | None,
        provided_embedding: bool,
    ) -> None:
        if train_Yvar is not None:
            raise UnsupportedError("LCEMGP with fixed noise (`train_Yvar`) is not supported.")
        if mean_module is not None:
            raise UnsupportedError("A custom `mean_module` is not supported for LCEMGP.")
        if likelihood is not None:
            raise UnsupportedError("A custom `likelihood` is not supported for LCEMGP.")
        if input_transform is not None:
            raise UnsupportedError("Input transforms are not supported for LCEMGP.")
        kind = kind_of(train_X)
        x = to_flat(train_X, "train_X")
        y = to_flat(train_Y, "train_Y")
        _validate_tensor_args(x[1], y[1], None)
        if len(x[1]) != 2:
            raise ValueError(f"Unsupported shape {tuple(x[1])} for train_X.")
        d = x[1][-1] - 1
        if _is_default(outcome_transform):
            ot_obj: Standardize | None = Standardize(m=1)
            ot_rs, std_default = None, True
        elif outcome_transform is None:
            ot_obj, ot_rs, std_default = None, None, False
        elif isinstance(outcome_transform, Standardize):
            if not is_plain_standardize(outcome_transform):
                raise UnsupportedError(
                    "`outcome_transform` must be a Standardize (of all outputs), DEFAULT or None: "
                    f"{type(outcome_transform).__name__} is not supported by LCEMGP."
                )
            if outcome_transform._m != 1:
                raise BotorchTensorDimensionError(
                    f"The outcome transform is for m = {outcome_transform._m} outputs, got m = 1."
                )
            ot_obj, ot_rs, std_default = outcome_transform, outcome_transform._rs, False
        else:
            raise UnsupportedError("`outcome_transform` must be a Standardize, DEFAULT or None.")
        if covar_module is not None and not isinstance(covar_module, Kernel):
            raise UnsupportedError(
                "`covar_module` must be a botorch_rs kernel (RBFKernel, MaternKernel(nu=2.5) "
                "or ScaleKernel)."
            )
        covar = covar_module
        covar_spec = covar._spec(d) if covar is not None and d > 0 else None
        self._rs = _rs.LCEMGP(
            x,
            y,
            int(task_feature),
            _int_list(output_tasks, "output_tasks"),
            _int_list(all_tasks, "all_tasks"),
            _matrix_arg(context_cat_feature, "context_cat_feature"),
            _matrix_arg(context_emb_feature, "context_emb_feature"),
            None if embs_dim_list is None else [int(e) for e in embs_dim_list],
            ot_rs,
            std_default,
            covar_spec,
            None,
            None if seed is None else int(seed),
            provided_embedding,
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
        observed = set(rs.observed_task_indices)
        self._observed_task_indices = sorted(observed)
        self._unobserved_task_indices = [k for k in range(self.num_tasks) if k not in observed]
        self._expected_task_values = set(self.all_tasks)
        self._validate_task_values = True
        if context_cat_feature is None:
            # BoTorch's default: the task values (in the order of `all_tasks`).
            given = _int_list(all_tasks, "all_tasks") or self.all_tasks
            context_cat_feature = from_flat([float(t) for t in given], (len(given), 1), kind)
        self.context_cat_feature = context_cat_feature
        self.context_emb_feature = context_emb_feature
        self.emb_dims: list[tuple[int, int]] = [tuple(s) for s in rs.embedding_shapes]  # type: ignore[misc]
        # Modules bound to the model's hyperparameters.
        self.likelihood = HadamardGaussianLikelihood._create(
            self, rs.likelihood_spec(), self.num_tasks, self._task_feature
        )
        self.mean_module = MultitaskMean._create(self, self.num_tasks)
        if covar is None:
            covar = get_covar_module_with_dim_scaled_prior(ard_num_dims=d)
        covar._bind(_KernelSlot(self, "data", "covar_module"))  # type: ignore[arg-type]
        self.covar_module: Kernel = covar
        self.emb_layers = [Embedding(self, i, n, e) for i, (n, e) in enumerate(self.emb_dims)]
        task_kernel = RBFKernel(
            ard_num_dims=rs.embedding_dim,
            lengthscale_constraint=Interval(0.0, 2.0, transform=None, initial_value=1.0),
        )
        task_kernel._bind(_KernelSlot(self, "task", "task_covar_module_base"))  # type: ignore[arg-type]
        self._task_covar_module_base: Kernel = task_kernel
        if ot_obj is not None:
            fitted = rs.outcome_transform()
            assert fitted is not None
            ot_obj._set_fitted(fitted, kind_of(train_Y))
            self.outcome_transform = ot_obj
        self.training = True

    # -- the task kernel ---------------------------------------------------

    @property
    def task_covar_module_base(self) -> Kernel:
        """The kernel on the context embeddings. Assigning a new kernel (an
        ``RBFKernel``, ``MaternKernel`` or ``ScaleKernel`` with one lengthscale
        per embedding dimension) replaces it; its hyperparameters start at
        their initial values, as in BoTorch."""
        return self._task_covar_module_base

    @task_covar_module_base.setter
    def task_covar_module_base(self, kernel: Kernel) -> None:
        if not isinstance(kernel, Kernel):
            raise UnsupportedError(
                "`task_covar_module_base` must be a botorch_rs kernel (RBFKernel, "
                "MaternKernel(nu=2.5) or ScaleKernel)."
            )
        spec = kernel._spec(self._rs.embedding_dim)
        self._rs.set_task_covar_module(spec)
        kernel._bind(_KernelSlot(self, "task", "task_covar_module_base"))  # type: ignore[arg-type]
        self._task_covar_module_base = kernel

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
        """``(train_X,)`` including the task column."""
        data, shape = self._rs.train_inputs()
        return (from_flat(data, tuple(shape), self._kind),)

    @property
    def train_targets(self) -> Any:
        """The training targets after the outcome transform (``n``)."""
        data, shape = self._rs.transformed_train_targets()
        return from_flat(data, tuple(shape), self._kind)

    # -- forward-pass side effects ------------------------------------------

    def _forward_pass(self) -> None:
        """The in-place renormalization of the embedding weights that every
        forward pass of BoTorch's model performs (``max_norm=1.0``)."""
        self._rs.renorm_embeddings()

    def _task_embeddings(self) -> Any:
        """The embeddings of all contexts (``num_contexts x D``: learned,
        then fixed); renormalizes the learned weights like BoTorch."""
        self._forward_pass()
        data, shape = self._rs.task_embeddings()
        return from_flat(data, tuple(shape), self._kind)

    def _eval_context_covar(self) -> Any:
        """The task covariance of all contexts (``num_contexts x
        num_contexts``)."""
        self._forward_pass()
        data, shape = self._rs.task_covariance()
        return from_flat(data, tuple(shape), self._kind)

    def task_covar_module(self, task_idcs: Any) -> Any:
        """The task covariance of the task indices ``task_idcs`` (``n x 1`` or
        ``b x n x 1``), shape ``n x n`` or ``b x n x n``."""
        idx, ishape = to_flat(task_idcs, "task_idcs")
        if not ishape or ishape[-1] != 1:
            raise BotorchTensorDimensionError("task_idcs must have shape `(b x) n x 1`.")
        self._forward_pass()
        data, shape = self._rs.task_covariance()
        k = shape[0]
        n = ishape[-2]
        batch = math.prod(ishape[:-2])
        out: list[float] = []
        for b in range(batch):
            rows = [int(v) for v in idx[b * n : (b + 1) * n]]
            out.extend(data[r * k + c] for r in rows for c in rows)
        return from_flat(out, (*ishape[:-2], n, n), kind_of(task_idcs))

    def _native_task_source(self, tasks: list[int] | None, includes_task_feature: bool) -> Any:
        return (
            _rs.Model.from_lcemgp_task_inputs(self._rs)
            if includes_task_feature
            else _rs.Model.from_lcemgp(self._rs, tasks)
        )

    # -- posterior ---------------------------------------------------------

    def _check_task_values(self, values: list[float]) -> None:
        """BoTorch's ``_map_tasks`` validation (``validate_task_values=True``)."""
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

    def _acquisition_model(self, posterior_transform: Any = None) -> _botorch_rs.Model:
        """The model handle that acquisition functions are built on."""
        from botorch_rs.acquisition.objective import transform_arg

        # BoTorch's acquisition functions evaluate `model.posterior`, which
        # calls `eval()` and runs a forward pass.
        self.eval()
        self._forward_pass()
        return _rs.Model.from_lcemgp(
            self._rs, None, transform_arg(posterior_transform, self.num_outputs)
        )

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        raise UnsupportedError(
            "condition_on_observations is not supported for LCEMGP in botorch_rs."
        )

    def fantasize(self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any) -> Any:
        raise UnsupportedError("fantasize is not supported for LCEMGP in botorch_rs.")

    @classmethod
    def construct_inputs(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        raise UnsupportedError("LCEMGP.construct_inputs is not ported to botorch_rs.")

    # -- parameters --------------------------------------------------------

    def _raw_param(self, name: str) -> list[float]:
        values = self._rs.raw_parameters()
        offset = 0
        for n, size in self._rs.parameter_names():
            if n == name:
                return values[offset : offset + size]
            offset += size
        raise AttributeError(f"The model has no parameter {name!r}.")

    def _set_raw_param(self, name: str, value: Any) -> None:
        from botorch_rs._arrays import broadcast_to

        data, shape = to_flat(value, name)
        shapes = dict(self._rs.parameter_shapes())
        if name not in shapes:
            raise AttributeError(f"The model has no parameter {name!r}.")
        target = tuple(shapes[name])
        try:
            new = broadcast_to(data, shape, target)
        except ValueError:
            raise BotorchTensorDimensionError(
                f"{name} has shape {target}, got a value of shape {tuple(shape)}."
            ) from None
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

    def initialize(self, **kwargs: Any) -> LCEMGP:
        """Sets hyperparameters by (dotted) name, like GPyTorch's
        ``Module.initialize``, e.g. ``model.initialize(**{"likelihood.noise":
        1e-3, "emb_layers.0.weight": w, "task_covar_module_base.lengthscale":
        0.5})``; raw names (those of ``named_parameters()``) work too."""
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

    # -- modes and copies ----------------------------------------------------

    def eval(self) -> LCEMGP:
        """Evaluation mode. As in the Unity project's LCEMGP, the constant
        mean of every task without training data becomes the average of the
        observed tasks' constants (``posterior`` calls ``eval()``)."""
        self._rs.eval_unobserved_means()
        return self.train(False)

    def train(self, mode: bool = True) -> LCEMGP:
        self.training = mode
        tf = getattr(self, "outcome_transform", None)
        if tf is not None:
            tf.train(mode)
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> LCEMGP:
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
            f"LCEMGP(n={self._rs.num_train}, d={self.num_non_task_features}, "
            f"num_tasks={self.num_tasks}, output_tasks={self._output_tasks}, "
            f"emb_dims={self.emb_dims}, embedding_dim={self._rs.embedding_dim})"
        )


class ProvidedEmbeddingLCEMGP(LCEMGP):
    r"""An LCEMGP whose context embeddings are exactly the provided ones
    (the Unity project's ``_ProvidedEmbeddingLCEMGP``; a botorch_rs
    extension, not part of BoTorch).

    There is no learned embedding (``emb_layers`` is empty): the task kernel
    ``task_covar_module_base`` (``RBFKernel(ard_num_dims=m)`` with
    ``Interval(0, 2)`` lengthscales without transform) sees only
    ``context_emb_feature`` (``num_tasks x m``). The context covariance is
    ``(1 - w) * k_e(E, E) + w * I`` with a learned context-specific weight
    ``w = context_specific_weight`` (parameter ``raw_context_specific_weight``
    of shape ``1``, constraint ``Interval(0, 1, transform=None,
    initial_value=0)``, i.e. optimizer box bounds), so that contexts with
    identical embeddings can still be fitted apart. Everything else is
    :class:`LCEMGP`'s, including ``eval()`` for tasks without data;
    ``context_cat_feature`` is accepted for BoTorch's signature but not used.

    Example:
        >>> model = ProvidedEmbeddingLCEMGP(
        ...     train_X, train_Y, task_feature=-1,
        ...     context_cat_feature=torch.arange(3.0).unsqueeze(-1),
        ...     context_emb_feature=embeddings, output_tasks=[2], all_tasks=[0, 1, 2])
    """

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        train_Yvar: Any = None,
        mean_module: Any = None,
        covar_module: Kernel | None = None,
        likelihood: Any = None,
        context_cat_feature: Any = None,
        context_emb_feature: Any = None,
        embs_dim_list: list[int] | None = None,
        output_tasks: list[int] | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        if context_emb_feature is None:
            raise ValueError("ProvidedEmbeddingLCEMGP requires a `context_emb_feature`.")
        self._init(
            train_X,
            train_Y,
            task_feature,
            train_Yvar,
            mean_module,
            covar_module,
            likelihood,
            context_cat_feature,
            context_emb_feature,
            None,
            output_tasks,
            all_tasks,
            outcome_transform,
            input_transform,
            seed,
            provided_embedding=True,
        )
        self.raw_context_specific_weight_constraint = Interval(
            0.0, 1.0, transform=None, initial_value=0.0
        )

    @property
    def raw_context_specific_weight(self) -> Any:
        """The raw context-specific weight (shape ``1``)."""
        return from_flat(self._raw_param("raw_context_specific_weight"), (1,), self._kind)

    @raw_context_specific_weight.setter
    def raw_context_specific_weight(self, value: Any) -> None:
        self._set_raw_param("raw_context_specific_weight", value)

    @property
    def context_specific_weight(self) -> Any:
        """The context-specific weight ``w`` (shape ``1``)."""
        w = self._rs.context_specific_weight
        assert w is not None
        return from_flat([w], (1,), self._kind)

    @context_specific_weight.setter
    def context_specific_weight(self, value: Any) -> None:
        data, shape = to_flat(value, "context_specific_weight")
        raw = [self.raw_context_specific_weight_constraint.inverse_transform(v) for v in data]
        self._set_raw_param("raw_context_specific_weight", from_flat(raw, tuple(shape), LIST))

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of every hyperparameter in BoTorch's
        ``named_parameters()`` order (the model's own
        ``raw_context_specific_weight`` first)."""
        items = list(super().named_parameters())
        own = [item for item in items if item[0] == "raw_context_specific_weight"]
        return iter(own + [item for item in items if item[0] != "raw_context_specific_weight"])


#: botorch_rs provides the Unity project's contextual models: ``LCEMGP`` with
#: the averaged constant means of unobserved tasks and ``ProvidedEmbeddingLCEMGP``.
HAS_PROVIDED_EMBEDDING_LCEMGP = True

__all__ = ["HAS_PROVIDED_EMBEDDING_LCEMGP", "Embedding", "LCEMGP", "ProvidedEmbeddingLCEMGP"]
