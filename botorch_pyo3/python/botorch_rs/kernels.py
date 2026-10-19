"""Covariance modules (``gpytorch.kernels``).

The kernels of GPyTorch that BoTorch users pass as ``covar_module``:
``RBFKernel``, ``MaternKernel`` (``nu`` = 0.5, 1.5, 2.5), ``RQKernel``,
``PeriodicKernel``, ``LinearKernel``, ``PolynomialKernel``,
``CosineKernel`` and ``IndexKernel``, combined with ``ScaleKernel``,
``AdditiveKernel`` and ``ProductKernel`` (also ``k1 + k2`` and ``k1 * k2``)
into trees, with ``active_dims``, ``ard_num_dims``, ``batch_shape``,
constraints and priors. BoTorch's own kernels are in
:mod:`botorch_rs.models.kernels`.

A kernel passed to a model as ``covar_module`` is bound to it: its
hyperparameter attributes (``lengthscale``, ``raw_lengthscale``,
``outputscale``, ...) then read and set the model's hyperparameters (with
the model's ``_aug_batch_shape`` as leading dimensions, as in GPyTorch).
Before that, they hold the kernel's own values (GPyTorch's initial values,
or what was assigned), which the model starts from. The kernel matrices are
computed in Rust (``kernel(x1, x2)``, the model's posteriors, the marginal
log likelihood and their derivatives).

Deviations from GPyTorch: kernel calls return dense arrays (the array type
of ``x1``) instead of ``LinearOperator`` objects; ``last_dim_is_batch`` is
not supported; unknown keyword arguments raise ``UnsupportedError`` (GPyTorch
ignores them); ``IndexKernel`` draws its random initial values from torch's
global generator if torch is installed (as GPyTorch), else from Python's
``random`` module.
"""

from __future__ import annotations

import copy
import importlib
import math
import random
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    LIST,
    ArrayKind,
    broadcast_shapes,
    broadcast_to,
    from_flat,
    kind_of,
    to_flat,
)
from botorch_rs._module import GPModule
from botorch_rs.constraints import Interval, Positive
from botorch_rs.constraints import _from_rs as _constraint_from_rs
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.priors import LKJCovariancePrior, Prior
from botorch_rs.priors import _from_rs as _prior_from_rs


def _check_batch_shape(batch_shape: Any) -> tuple[int, ...]:
    """A module's ``batch_shape`` as a tuple of ints."""
    if batch_shape is None:
        return ()
    try:
        return tuple(int(b) for b in batch_shape)
    except TypeError as e:
        raise UnsupportedError(f"Invalid batch_shape {batch_shape!r}.") from e


def _check_prior(prior: Any, name: str) -> None:
    if prior is not None and not isinstance(prior, Prior):
        raise TypeError(f"Expected gpytorch.priors.Prior but got {type(prior).__name__}")


def _check_constraint(constraint: Any, name: str) -> None:
    if constraint is not None and not isinstance(constraint, Interval):
        raise UnsupportedError(
            f"`{name}` must be a botorch_rs.constraints constraint (GreaterThan, Positive, "
            "LessThan or Interval)."
        )


def _check_kwargs(kwargs: dict[str, Any], cls: str) -> None:
    kwargs.pop("eps", None)  # GPyTorch's deprecated argument, ignored
    kwargs.pop("device", None)
    kwargs.pop("dtype", None)
    if kwargs:
        raise UnsupportedError(f"Unsupported arguments of {cls}: {sorted(kwargs)}.")


def _active_dims(active_dims: Any) -> tuple[int, ...] | None:
    if active_dims is None:
        return None
    data, _ = to_flat(active_dims, "active_dims")
    dims = tuple(int(v) for v in data)
    if any(d != v for d, v in zip(dims, data, strict=True)):
        raise ValueError(f"active_dims must be integers, got {active_dims!r}.")
    return dims


def _randn(shape: Sequence[int], uniform: bool = False) -> list[float]:
    """``torch.randn(*shape)`` (``torch.rand`` if ``uniform``) from torch's
    global generator, as GPyTorch initializes ``IndexKernel``; Python's
    ``random`` module without torch."""
    n = math.prod(shape)
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        if uniform:
            return [random.random() for _ in range(n)]
        return [random.gauss(0.0, 1.0) for _ in range(n)]
    t = torch.rand(*shape) if uniform else torch.randn(*shape)
    return [float(v) for v in t.reshape(-1).tolist()]


class _KernelList(list):  # type: ignore[type-arg]
    """``torch.nn.ModuleList`` of sub-kernels (``kernels[0]`` or
    ``kernels.0`` in dotted names)."""

    def __getattr__(self, name: str) -> Any:
        if name.isdigit() and int(name) < len(self):
            return self[int(name)]
        raise AttributeError(name)


def _properties(raw_name: str) -> tuple[property, property]:
    """The ``raw_<name>`` and ``<name>`` properties of a kernel parameter."""

    def get_raw(self: Kernel) -> Any:
        return self._get_raw(raw_name)

    def set_raw(self: Kernel, value: Any) -> None:
        self._set_raw(raw_name, value)

    def get_value(self: Kernel) -> Any:
        return self._get_value(raw_name)

    def set_value(self: Kernel, value: Any) -> None:
        self._set_value(raw_name, value)

    return property(get_raw, set_raw), property(get_value, set_value)


class Kernel(GPModule):
    """Base class of the covariance modules (``gpytorch.kernels.Kernel``)."""

    #: Whether the kernel has a ``lengthscale``.
    has_lengthscale = False
    #: The kernel's own parameters in GPyTorch's order: ``(raw name,
    #: constraint attribute, prior attribute)``.
    _params: tuple[tuple[str, str | None, str | None], ...] = ()
    #: A BoTorch preset building the kernel's ``KernelSpec`` for ``d`` inputs.
    _preset: Callable[[int], _rs.KernelSpec] | None = None

    def __init__(
        self,
        ard_num_dims: int | None = None,
        batch_shape: Any = None,
        active_dims: Any = None,
        lengthscale_prior: Prior | None = None,
        lengthscale_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        _check_kwargs(kwargs, type(self).__name__)
        self._batch_shape = _check_batch_shape(batch_shape)
        self.active_dims = _active_dims(active_dims)
        if ard_num_dims is not None and int(ard_num_dims) <= 0:
            raise ValueError(f"ard_num_dims must be positive, got {ard_num_dims}.")
        self.ard_num_dims = None if ard_num_dims is None else int(ard_num_dims)
        self._local: dict[str, list[float]] = {}
        self._local_kind: ArrayKind = LIST
        self._prefix = ""
        self._preset = None
        if self.has_lengthscale:
            _check_prior(lengthscale_prior, "lengthscale_prior")
            _check_constraint(lengthscale_constraint, "lengthscale_constraint")
            self.lengthscale_prior = lengthscale_prior
            self.raw_lengthscale_constraint = (
                lengthscale_constraint if lengthscale_constraint is not None else Positive()
            )

    # -- structure ---------------------------------------------------------

    def _base_node(self) -> _rs.KernelNode:  # pragma: no cover - abstract
        raise NotImplementedError

    def _children(self) -> list[tuple[str, Kernel]]:
        """``(attribute, sub-kernel)`` in GPyTorch's module order."""
        return []

    def _own_params(self) -> list[tuple[str, Interval | None, Prior | None]]:
        """``(raw name, constraint, prior)`` of the kernel's own parameters
        (including parameters registered later, e.g. the ``raw_tau`` of a
        SAAS prior)."""
        out = []
        for name, c_attr, p_attr in (*type(self)._params, *getattr(self, "_extra_params", ())):
            c = getattr(self, c_attr, None) if c_attr else None
            p = getattr(self, p_attr, None) if p_attr else None
            out.append((name, c, p))
        return out

    def _configure(self, node: _rs.KernelNode) -> _rs.KernelNode:
        """Applies constraints, priors and initial values of the own parameters."""
        for name, constraint, prior in self._own_params():
            if constraint is not None or prior is not None:
                node = node.with_param(
                    name,
                    constraint._rs if constraint is not None else None,
                    prior._rs if prior is not None else None,
                )
        return node

    def _node(self) -> _rs.KernelNode:
        """The kernel tree (``botorch_rs._botorch_rs.KernelNode``)."""
        node = self._base_node()
        saas = getattr(self, "_saas", None)
        if saas is not None:
            node = saas.apply(self, node)
        node = self._configure(node)
        if self.active_dims is not None:
            node = node.with_active_dims(list(self.active_dims))
        return node

    def _standard_spec(self, d: int) -> _rs.KernelSpec | None:
        """The ``KernelSpec`` of BoTorch's preset modules (evaluated by the
        specialized kernel of botorch_rs's default models); ``None`` for other
        kernels, which are evaluated as kernel trees."""
        if self._preset is not None and self.active_dims is None:
            self._node().validate(d)
            return self._preset(d)
        return None

    def _spec(self, d: int) -> _rs.KernelSpec:
        """The ``KernelSpec`` (models that support standard kernels only: an
        RBF or Matern kernel with one lengthscale per input, optionally in a
        ``ScaleKernel``, without ``active_dims``)."""
        spec = self._standard_spec(d) or self._node().as_kernel_spec(d)
        if spec is None:
            raise UnsupportedError(
                "This model supports RBFKernel and MaternKernel with one lengthscale per input "
                "(ard_num_dims=d), optionally in a ScaleKernel, without active_dims."
            )
        return spec

    def _all_kernels(self) -> Iterator[Kernel]:
        yield self
        for _, child in self._children():
            yield from child._all_kernels()

    def named_sub_kernels(self) -> Iterator[tuple[str, Kernel]]:
        """The sub-kernels with their attribute names (GPyTorch's
        ``named_sub_kernels``)."""
        for name, child in self._children():
            yield name, child
            for sub, k in child.named_sub_kernels():
                yield f"{name}.{sub}", k

    def sub_kernels(self) -> Iterator[Kernel]:
        for _, k in self.named_sub_kernels():
            yield k

    @property
    def batch_shape(self) -> tuple[int, ...]:  # type: ignore[override]
        """The batch shape: the kernel's own, broadcast with its sub-kernels'."""
        shape = self._batch_shape
        for _, child in self._children():
            b = broadcast_shapes(shape, child.batch_shape)
            if b is None:
                raise RuntimeError(
                    f"Shape mismatch: the batch shapes {shape} and {child.batch_shape} of the "
                    "sub-kernels do not broadcast."
                )
            shape = b
        return shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._batch_shape = _check_batch_shape(value)

    @property
    def is_stationary(self) -> bool:
        return self.has_lengthscale

    # -- parameters --------------------------------------------------------

    def _element_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(shape) for name, shape, _ in self._node().own_params()}

    def _initial(self, name: str) -> float:
        for n, _, init in self._node().own_params():
            if n == name:
                return float(init)
        raise AttributeError(f"{type(self).__name__} has no parameter {name!r}.")

    def _param_shape(self, name: str) -> tuple[int, ...]:
        """The full shape of the parameter ``name``."""
        if self._model is not None:
            getter = getattr(self._model, "_raw_param_shape", None)
            if getter is not None:
                return tuple(getter(self._prefix + name))
            return (*self._aug, *self._element_shapes()[name])
        return (*self._batch_shape, *self._element_shapes()[name])

    def _raw_values(self, name: str) -> list[float]:
        if self._model is not None:
            return list(self._model._raw_param(self._prefix + name))
        if name in self._local:
            return list(self._local[name])
        return [self._initial(name)] * math.prod(self._param_shape(name))

    def _kind_out(self) -> ArrayKind:
        return self._model._kind if self._model is not None else self._local_kind

    def _constraint_of(self, name: str) -> Interval | None:
        for n, c, _ in self._own_params():
            if n == name:
                return c
        raise AttributeError(f"{type(self).__name__} has no parameter {name!r}.")

    def _get_raw(self, name: str) -> Any:
        return from_flat(self._raw_values(name), self._param_shape(name), self._kind_out())

    def _set_raw(self, name: str, value: Any) -> None:
        if self._model is not None:
            self._model._set_raw_param(self._prefix + name, value)
            return
        data, shape = to_flat(value, name)
        target = self._param_shape(name)
        try:
            new = broadcast_to(data, shape, target)
        except ValueError:
            if len(data) != math.prod(target):
                raise BotorchTensorDimensionError(
                    f"{name} has shape {target}, got a value of shape {tuple(shape)}."
                ) from None
            new = [float(v) for v in data]
        self._local[name] = new
        self._local_kind = kind_of(value)

    def _get_value(self, name: str) -> Any:
        c = self._constraint_of(name)
        raw = self._raw_values(name)
        values = raw if c is None else [c.transform(v) for v in raw]
        return from_flat(values, self._param_shape(name), self._kind_out())

    def _set_value(self, name: str, value: Any) -> None:
        c = self._constraint_of(name)
        data, shape = to_flat(value, name)
        raw = data if c is None else [c.inverse_transform(v) for v in data]
        self._set_raw(name, from_flat(list(raw), tuple(shape), LIST))
        if self._model is None:
            self._local_kind = kind_of(value)

    def _raw_flat(self) -> list[float]:
        """The raw values of all parameters (own and sub-kernels'), flat in
        ``named_parameters()`` order."""
        out: list[float] = []
        for name, _, _ in self._own_params():
            out.extend(self._raw_values(name))
        for _, child in self._children():
            out.extend(child._raw_flat())
        return out

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of every parameter (GPyTorch's
        ``named_parameters``)."""
        for name, _, _ in self._own_params():
            yield name, self._get_raw(name)
        for attr, child in self._children():
            for name, value in child.named_parameters():
                yield f"{attr}.{name}", value

    def parameters(self) -> Iterator[Any]:
        for _, value in self.named_parameters():
            yield value

    def initialize(self, **kwargs: Any) -> Kernel:
        """Sets hyperparameters by (dotted) name, like GPyTorch's
        ``Module.initialize``."""
        for name, value in kwargs.items():
            obj: Any = self
            *path, attr = name.split(".")
            for part in path:
                obj = getattr(obj, part)
            if not isinstance(getattr(type(obj), attr, None), property):
                raise AttributeError(f"Unknown parameter {name!r} for {type(obj).__name__}.")
            setattr(obj, attr, value)
        return self

    # -- binding -------------------------------------------------------------

    def _bind(self, model: Any, prefix: str = "") -> None:
        super()._bind(model)
        self._prefix = prefix
        for name, values in self._local.items():
            model._set_raw_param(prefix + name, from_flat(values, self._param_shape(name), LIST))
        self._local = {}
        for attr, child in self._children():
            child._bind(model, f"{prefix}{attr}.")

    # -- lengthscale -------------------------------------------------------

    @property
    def lengthscale(self) -> Any:
        """The lengthscale (``None`` for kernels without one)."""
        if not self.has_lengthscale:
            return None
        return self._get_value("raw_lengthscale")

    @lengthscale.setter
    def lengthscale(self, value: Any) -> None:
        if not self.has_lengthscale:
            raise RuntimeError("Kernel has no lengthscale.")
        self._set_value("raw_lengthscale", value)

    @property
    def raw_lengthscale(self) -> Any:
        if not self.has_lengthscale:
            raise AttributeError(f"{type(self).__name__} has no lengthscale.")
        return self._get_raw("raw_lengthscale")

    @raw_lengthscale.setter
    def raw_lengthscale(self, value: Any) -> None:
        self._set_raw("raw_lengthscale", value)

    @property
    def lengthscale_constraint(self) -> Interval:
        """The lengthscale constraint (GPyTorch's ``raw_lengthscale_constraint``)."""
        return self.raw_lengthscale_constraint

    @property
    def raw_tau(self) -> Any:
        """The global shrinkage parameter of a SAAS prior with a learned
        ``tau`` (``add_saas_prior(kernel, tau=None)``)."""
        if "raw_tau" not in (name for name, _, _ in self._own_params()):
            raise AttributeError(f"{type(self).__name__} has no parameter 'raw_tau'.")
        return self._get_raw("raw_tau")

    @raw_tau.setter
    def raw_tau(self, value: Any) -> None:
        if "raw_tau" not in (name for name, _, _ in self._own_params()):
            raise AttributeError(f"{type(self).__name__} has no parameter 'raw_tau'.")
        self._set_raw("raw_tau", value)

    # -- evaluation ----------------------------------------------------------

    def __call__(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Any:
        """The kernel matrix ``K(x1, x2)`` (``... x n1 x n2``), or its
        diagonal (``diag=True``) as GPyTorch computes it, with the current
        hyperparameters; a dense array of the type of ``x1``."""
        if last_dim_is_batch:
            raise UnsupportedError("last_dim_is_batch is not supported by botorch_rs.")
        if params:
            raise UnsupportedError(f"Unsupported arguments: {sorted(params)}.")
        kind = kind_of(x1)
        a = to_flat(x1, "x1")
        if len(a[1]) == 1:
            a = (a[0], (a[1][0], 1))
        b = None
        if x2 is not None:
            b = to_flat(x2, "x2")
            if len(b[1]) == 1:
                b = (b[0], (b[1][0], 1))
            if b[1][-1] != a[1][-1]:
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")
        node = self._node()
        kbatch = self.batch_shape
        if self._model is not None:
            kbatch = self._aug
        abatch = tuple(a[1][:-2])
        bbatch = tuple(b[1][:-2]) if b is not None else abatch
        out_batch = broadcast_shapes(kbatch, abatch, bbatch)
        if out_batch is None:
            raise RuntimeError(
                f"The batch shapes of the kernel {kbatch} and of the inputs {abatch}, "
                f"{bbatch} do not broadcast."
            )
        raws = self._raw_per_element(len(kbatch))
        results: list[float] = []
        shape: tuple[int, ...] = ()
        for idx in _indices(out_batch):
            raw = raws[_flat_index(idx, kbatch)]
            xa = _batch_matrix(a, idx)
            xb = _batch_matrix(b, idx) if b is not None else None
            data, out_shape = node.evaluate(raw, xa, xb, diag)
            shape = tuple(out_shape)
            results.extend(data)
        return from_flat(results, (*out_batch, *shape), kind)

    def _raw_per_element(self, nbatch: int) -> list[list[float]]:
        """The flat raw parameters of every kernel batch element."""
        k = self._node().num_raw
        flat = self._raw_flat()
        n = max(len(flat) // max(k, 1), 1) if k else 1
        if k == 0:
            batch = self._aug if self._model is not None else self.batch_shape
            return [[]] * max(math.prod(batch), 1)
        # Parameters are stored parameter by parameter over the batch.
        sizes = [math.prod(s) for _, s in self._node().param_shapes("")]
        per: list[list[float]] = [[] for _ in range(n)]
        offset = 0
        for size in sizes:
            for e in range(n):
                per[e].extend(flat[offset + e * size : offset + (e + 1) * size])
            offset += size * n
        return per

    def __add__(self, other: Kernel) -> Kernel:
        kernels: list[Kernel] = []
        kernels += list(self.kernels) if isinstance(self, AdditiveKernel) else [self]
        kernels += list(other.kernels) if isinstance(other, AdditiveKernel) else [other]
        return AdditiveKernel(*kernels)

    def __mul__(self, other: Kernel) -> Kernel:
        kernels: list[Kernel] = []
        kernels += list(self.kernels) if isinstance(self, ProductKernel) else [self]
        kernels += list(other.kernels) if isinstance(other, ProductKernel) else [other]
        return ProductKernel(*kernels)

    def __deepcopy__(self, memo: dict[int, Any]) -> Kernel:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def __repr__(self) -> str:
        args = []
        if self.ard_num_dims is not None:
            args.append(f"ard_num_dims={self.ard_num_dims}")
        if self.active_dims is not None:
            args.append(f"active_dims={list(self.active_dims)}")
        return f"{type(self).__name__}({', '.join(args)})"


def _indices(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    if not shape:
        yield ()
        return
    idx = [0] * len(shape)
    for _ in range(math.prod(shape)):
        yield tuple(idx)
        for k in range(len(shape) - 1, -1, -1):
            idx[k] += 1
            if idx[k] < shape[k]:
                break
            idx[k] = 0


def _flat_index(idx: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """The row-major index in ``shape`` of the broadcast index ``idx``."""
    off = len(idx) - len(shape)
    flat = 0
    for i, s in enumerate(shape):
        flat = flat * s + (0 if s == 1 else idx[off + i])
    return flat


def _batch_matrix(x: tuple[Any, Any], idx: tuple[int, ...]) -> tuple[list[float], tuple[int, int]]:
    data, shape = x
    n, d = shape[-2], shape[-1]
    batch = tuple(shape[:-2])
    e = _flat_index(idx, batch) if batch else 0
    return list(data[e * n * d : (e + 1) * n * d]), (n, d)


# ----------------------------------------------------------------------
# Stationary kernels
# ----------------------------------------------------------------------


class RBFKernel(Kernel):
    """Squared-exponential kernel ``exp(-r**2 / 2)`` with
    ``r = ||(x1 - x2) / lengthscale||`` (``gpytorch.kernels.RBFKernel``)."""

    has_lengthscale = True
    _params = (("raw_lengthscale", "raw_lengthscale_constraint", "lengthscale_prior"),)
    _kernel_kind = "rbf"

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.stationary(self._kernel_kind, self.ard_num_dims)


class MaternKernel(Kernel):
    """Matern kernel with smoothness ``nu`` (0.5, 1.5 or 2.5)
    (``gpytorch.kernels.MaternKernel``)."""

    has_lengthscale = True
    _params = (("raw_lengthscale", "raw_lengthscale_constraint", "lengthscale_prior"),)

    def __init__(self, nu: float | None = 2.5, **kwargs: Any) -> None:
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super().__init__(**kwargs)
        self.nu = nu

    @property
    def _kernel_kind(self) -> str:
        return {0.5: "matern12", 1.5: "matern32", 2.5: "matern52"}[float(self.nu)]  # type: ignore[arg-type]

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.stationary(self._kernel_kind, self.ard_num_dims)

    def __repr__(self) -> str:
        args = [f"nu={self.nu}"]
        if self.ard_num_dims is not None:
            args.append(f"ard_num_dims={self.ard_num_dims}")
        if self.active_dims is not None:
            args.append(f"active_dims={list(self.active_dims)}")
        return f"MaternKernel({', '.join(args)})"


class RQKernel(Kernel):
    """Rational quadratic kernel ``(1 + r**2 / (2 alpha))**(-alpha)``
    (``gpytorch.kernels.RQKernel``)."""

    has_lengthscale = True
    _params = (
        ("raw_lengthscale", "raw_lengthscale_constraint", "lengthscale_prior"),
        ("raw_alpha", "raw_alpha_constraint", None),
    )

    def __init__(self, alpha_constraint: Interval | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _check_constraint(alpha_constraint, "alpha_constraint")
        self.raw_alpha_constraint = alpha_constraint if alpha_constraint is not None else Positive()

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.rational_quadratic(self.ard_num_dims)

    raw_alpha, alpha = _properties("raw_alpha")


class PeriodicKernel(Kernel):
    """Periodic kernel ``exp(-2 sum_j sin(pi |x1_j - x2_j| / p_j)**2 / l_j)``
    (``gpytorch.kernels.PeriodicKernel``)."""

    has_lengthscale = True
    _params = (
        ("raw_lengthscale", "raw_lengthscale_constraint", "lengthscale_prior"),
        ("raw_period_length", "raw_period_length_constraint", "period_length_prior"),
    )

    def __init__(
        self,
        period_length_prior: Prior | None = None,
        period_length_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _check_prior(period_length_prior, "period_length_prior")
        _check_constraint(period_length_constraint, "period_length_constraint")
        self.period_length_prior = period_length_prior
        self.raw_period_length_constraint = (
            period_length_constraint if period_length_constraint is not None else Positive()
        )

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.periodic(self.ard_num_dims)

    raw_period_length, period_length = _properties("raw_period_length")


class LinearKernel(Kernel):
    """Linear kernel ``sum_j v_j x1_j x2_j`` (``gpytorch.kernels.LinearKernel``)."""

    _params = (("raw_variance", "raw_variance_constraint", "variance_prior"),)

    def __init__(
        self,
        ard_num_dims: int | None = None,
        variance_prior: Prior | None = None,
        variance_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)
        _check_prior(variance_prior, "variance_prior")
        _check_constraint(variance_constraint, "variance_constraint")
        self.variance_prior = variance_prior
        self.raw_variance_constraint = (
            variance_constraint if variance_constraint is not None else Positive()
        )

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.linear(self.ard_num_dims)

    raw_variance, variance = _properties("raw_variance")


class PolynomialKernel(Kernel):
    """Polynomial kernel ``(x1^T x2 + offset)**power``
    (``gpytorch.kernels.PolynomialKernel``)."""

    _params = (("raw_offset", "raw_offset_constraint", "offset_prior"),)

    def __init__(
        self,
        power: int,
        offset_prior: Prior | None = None,
        offset_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        data, _ = to_flat(power, "power")
        if len(data) > 1:
            raise RuntimeError("Cant create a Polynomial kernel with more than one power")
        self.power = power if isinstance(power, (int, float)) else data[0]
        _check_prior(offset_prior, "offset_prior")
        _check_constraint(offset_constraint, "offset_constraint")
        self.offset_prior = offset_prior
        self.raw_offset_constraint = (
            offset_constraint if offset_constraint is not None else Positive()
        )

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.polynomial(float(self.power))

    raw_offset, offset = _properties("raw_offset")


class CosineKernel(Kernel):
    """Cosine kernel ``cos(pi ||x1 - x2|| / period_length)``
    (``gpytorch.kernels.CosineKernel``)."""

    _params = (("raw_period_length", "raw_period_length_constraint", "period_length_prior"),)

    def __init__(
        self,
        period_length_prior: Prior | None = None,
        period_length_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _check_prior(period_length_prior, "period_length_prior")
        _check_constraint(period_length_constraint, "period_length_constraint")
        self.period_length_prior = period_length_prior
        self.raw_period_length_constraint = (
            period_length_constraint if period_length_constraint is not None else Positive()
        )

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.cosine()

    raw_period_length, period_length = _properties("raw_period_length")


class IndexKernel(Kernel):
    """Task covariance ``B[i, j]`` with ``B = F F^T + diag(var)`` for integer
    task indices ``i, j`` (``gpytorch.kernels.IndexKernel``); the input is a
    single column (``active_dims=[task_feature]``) truncated to integers."""

    _params: tuple[tuple[str, str | None, str | None], ...] = (
        ("covar_factor", None, None),
        ("raw_var", "raw_var_constraint", None),
    )

    def __init__(
        self,
        num_tasks: int,
        rank: int | None = 1,
        prior: Prior | None = None,
        var_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        rank = 1 if rank is None else int(rank)
        if rank > num_tasks:
            raise RuntimeError(
                "Cannot create a task covariance matrix larger than the number of tasks"
            )
        super().__init__(**kwargs)
        _check_prior(prior, "prior")
        _check_constraint(var_constraint, "var_constraint")
        self.num_tasks = int(num_tasks)
        self.rank = rank
        self.raw_var_constraint = var_constraint if var_constraint is not None else Positive()
        self.IndexKernelPrior = prior
        batch = self._batch_shape
        self._local["covar_factor"] = _randn((*batch, self.num_tasks, rank))
        raw_var = _randn((*batch, self.num_tasks))
        if self.raw_var_constraint.initial_value is None:
            self._local["raw_var"] = raw_var

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.index(self.num_tasks, self.rank)

    def _node(self) -> _rs.KernelNode:
        node = super()._node()
        prior = self.IndexKernelPrior
        if isinstance(prior, LKJCovariancePrior):
            return node.with_task_prior_lkj(prior.n, prior.eta, prior.sd_prior._rs)
        if prior is not None:
            return node.with_task_prior(prior._rs)
        return node

    @property
    def covar_factor(self) -> Any:
        return self._get_raw("covar_factor")

    @covar_factor.setter
    def covar_factor(self, value: Any) -> None:
        self._set_raw("covar_factor", value)

    raw_var, var = _properties("raw_var")

    def _eval_covar_matrix(self) -> Any:
        """``F F^T + diag(var)`` (``batch_shape x num_tasks x num_tasks``)."""
        return _task_covariance(self, normalize=None)

    @property
    def covar_matrix(self) -> Any:
        return self._eval_covar_matrix()


def _task_covariance(kernel: Kernel, normalize: int | None) -> Any:
    """The task covariance of an index kernel (per batch element)."""
    t = kernel.num_tasks  # type: ignore[attr-defined]
    r = kernel.rank  # type: ignore[attr-defined]
    factor_name = (
        "covar_factor" if "covar_factor" in kernel._element_shapes() else "raw_covar_factor"
    )
    fc = kernel._constraint_of(factor_name)
    f_raw = kernel._raw_values(factor_name)
    f = f_raw if fc is None else [fc.transform(v) for v in f_raw]
    vc = kernel._constraint_of("raw_var")
    var = [vc.transform(v) if vc is not None else v for v in kernel._raw_values("raw_var")]
    nb = len(var) // t
    out: list[float] = []
    for e in range(nb):
        fe = f[e * t * r : (e + 1) * t * r]
        b = [
            [
                sum(fe[i * r + k] * fe[j * r + k] for k in range(r))
                + (var[e * t + i] if i == j else 0.0)
                for j in range(t)
            ]
            for i in range(t)
        ]
        if normalize is not None:
            s = b[normalize][normalize]
            b = [[v / s for v in row] for row in b]
        out.extend(v for row in b for v in row)
    shape = kernel._param_shape("raw_var")[:-1]
    return from_flat(out, (*shape, t, t), kernel._kind_out())


# ----------------------------------------------------------------------
# Composite kernels
# ----------------------------------------------------------------------


class ScaleKernel(Kernel):
    """``outputscale * base_kernel`` (``gpytorch.kernels.ScaleKernel``)."""

    _params = (("raw_outputscale", "raw_outputscale_constraint", "outputscale_prior"),)

    def __init__(
        self,
        base_kernel: Kernel,
        outputscale_prior: Prior | None = None,
        outputscale_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(base_kernel, Kernel):
            raise UnsupportedError("ScaleKernel needs a botorch_rs kernel as base_kernel.")
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super().__init__(**kwargs)
        _check_prior(outputscale_prior, "outputscale_prior")
        _check_constraint(outputscale_constraint, "outputscale_constraint")
        self.base_kernel = base_kernel
        # GPyTorch creates ``raw_outputscale`` with the batch shape broadcast
        # with the base kernel's (e.g. a ScaleKernel without ``batch_shape``
        # around a batched kernel has batched outputscales).
        self._batch_shape = self.batch_shape
        self.outputscale_prior = outputscale_prior
        self.raw_outputscale_constraint = (
            outputscale_constraint if outputscale_constraint is not None else Positive()
        )

    def _children(self) -> list[tuple[str, Kernel]]:
        return [("base_kernel", self.base_kernel)]

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.scale(self.base_kernel._node())

    raw_outputscale, outputscale = _properties("raw_outputscale")

    @property
    def outputscale_constraint(self) -> Interval:
        return self.raw_outputscale_constraint

    @property
    def is_stationary(self) -> bool:
        return self.base_kernel.is_stationary

    def __repr__(self) -> str:
        return f"ScaleKernel({self.base_kernel!r})"


class _CompositeKernel(Kernel):
    """A combination of sub-kernels (``kernels``)."""

    def __init__(self, *kernels: Kernel) -> None:
        super().__init__()
        if not kernels:
            raise UnsupportedError(f"{type(self).__name__} needs at least one kernel.")
        for k in kernels:
            if not isinstance(k, Kernel):
                raise UnsupportedError(f"{type(self).__name__} needs botorch_rs kernels.")
        self.kernels = _KernelList(kernels)

    def _children(self) -> list[tuple[str, Kernel]]:
        return [(f"kernels.{i}", k) for i, k in enumerate(self.kernels)]

    @property
    def is_stationary(self) -> bool:
        return all(k.is_stationary for k in self.kernels)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(repr(k) for k in self.kernels)})"


class AdditiveKernel(_CompositeKernel):
    """The sum of kernels (``gpytorch.kernels.AdditiveKernel``)."""

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.additive([k._node() for k in self.kernels])


class ProductKernel(_CompositeKernel):
    """The elementwise product of kernels (``gpytorch.kernels.ProductKernel``)."""

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.product([k._node() for k in self.kernels])


class MultitaskKernel(Kernel):
    r"""The kernel of a ``KroneckerMultiTaskGP``
    (``gpytorch.kernels.MultitaskKernel``): ``K_XX kron K_TT`` of a data
    kernel ``data_covar_module`` and an ``IndexKernel(num_tasks, rank,
    prior=task_covar_prior)`` ``task_covar_module`` (interleaved: index
    ``i * num_tasks + t``)."""

    def __init__(
        self,
        data_covar_module: Kernel,
        num_tasks: int,
        rank: int | None = 1,
        task_covar_prior: Prior | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(data_covar_module, Kernel):
            raise UnsupportedError(
                "MultitaskKernel needs a botorch_rs kernel as data_covar_module."
            )
        super().__init__(**kwargs)
        self.task_covar_module = IndexKernel(
            num_tasks=num_tasks, batch_shape=self._batch_shape, rank=rank, prior=task_covar_prior
        )
        self.data_covar_module = data_covar_module
        self.num_tasks = int(num_tasks)

    def _children(self) -> list[tuple[str, Kernel]]:
        return [
            ("task_covar_module", self.task_covar_module),
            ("data_covar_module", self.data_covar_module),
        ]

    def _base_node(self) -> _rs.KernelNode:
        raise UnsupportedError(
            "MultitaskKernel is supported as the covariance module of a KroneckerMultiTaskGP only."
        )

    def __call__(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Any:
        """``K_XX kron K_TT`` (``n1 m x n2 m``, interleaved); unbatched inputs."""
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        kx = to_flat(self.data_covar_module(x1, x2, **params), "K")
        if len(kx[1]) != 2:
            raise UnsupportedError("MultitaskKernel evaluates unbatched inputs only.")
        kt = to_flat(self.task_covar_module.covar_matrix, "B")
        n1, n2 = kx[1]
        m = self.num_tasks
        out = [
            kx[0][i * n2 + j] * kt[0][s * m + t]
            for i in range(n1)
            for s in range(m)
            for j in range(n2)
            for t in range(m)
        ]
        if diag:
            out = [out[a * n2 * m + a] for a in range(min(n1, n2) * m)]
            return from_flat(out, (n1 * m,), kind_of(x1))
        return from_flat(out, (n1 * m, n2 * m), kind_of(x1))

    def __repr__(self) -> str:
        return f"MultitaskKernel({self.data_covar_module!r}, num_tasks={self.num_tasks})"


def _kernel_from_spec(
    spec: _rs.KernelSpec,
    preset: Callable[[int], _rs.KernelSpec],
    batch_shape: tuple[int, ...] = (),
    active_dims: Any = None,
) -> Kernel:
    """A kernel object describing ``spec`` that builds itself with ``preset``."""
    kind = spec.kind
    base: Kernel
    if kind == "rbf":
        base = RBFKernel(
            ard_num_dims=spec.ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
            lengthscale_prior=_prior_from_rs(spec.lengthscale_prior),
            lengthscale_constraint=_constraint_from_rs(spec.lengthscale_constraint),
        )
    else:
        nu = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}[kind]
        base = MaternKernel(
            nu=nu,
            ard_num_dims=spec.ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
            lengthscale_prior=_prior_from_rs(spec.lengthscale_prior),
            lengthscale_constraint=_constraint_from_rs(spec.lengthscale_constraint),
        )
    if not spec.has_scale_kernel:
        base._preset = preset
        return base
    outputscale_constraint = spec.outputscale_constraint
    kernel = ScaleKernel(
        base,
        outputscale_prior=_prior_from_rs(spec.outputscale_prior),
        outputscale_constraint=(
            _constraint_from_rs(outputscale_constraint)
            if outputscale_constraint is not None
            else None
        ),
        batch_shape=batch_shape,
    )
    kernel._preset = preset
    return kernel


__all__ = [
    "AdditiveKernel",
    "CosineKernel",
    "IndexKernel",
    "Kernel",
    "LinearKernel",
    "MultitaskKernel",
    "MaternKernel",
    "PeriodicKernel",
    "PolynomialKernel",
    "ProductKernel",
    "RBFKernel",
    "RQKernel",
    "ScaleKernel",
]
