"""Relevance pursuit with native exact-GP objectives and bounded optimization."""

from __future__ import annotations

import copy
import math
import sys
import warnings
from abc import ABC, abstractmethod
from operator import index as integer_index
from typing import Any

from botorch_rs import _ops
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError


class RelevancePursuitMixin(ABC):
    """Support bookkeeping for a parameter with one indexing dimension."""

    def __init__(self, dim: int, support: list[int] | None) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self._support = _indices(support or [], dim)
        self._is_sparse = True
        self._expansion_modifier: Any = None
        self._contraction_modifier: Any = None

    @property
    @abstractmethod
    def sparse_parameter(self) -> Any:
        """The active parameter values, or their dense representation."""

    @abstractmethod
    def set_sparse_parameter(self, value: Any) -> None:
        """Replace parameter values without changing the support."""

    @staticmethod
    def _from_model(model: Any) -> RelevancePursuitMixin:
        raise NotImplementedError("A relevance-pursuit subclass must locate its sparse module.")

    @property
    def support(self) -> list[int]:
        return self._support

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse

    @property
    def is_active(self) -> Any:
        values = [i in self.support for i in range(self.dim)]
        kind = kind_of(self.sparse_parameter)
        if kind.lib == "torch":
            torch = sys.modules["torch"]
            return torch.tensor(values, dtype=torch.bool, device=kind.device)
        if kind.lib == "numpy":
            return sys.modules["numpy"].asarray(values, dtype=bool)
        return values

    @property
    def inactive_indices(self) -> Any:
        values = [i for i in range(self.dim) if i not in self.support]
        kind = kind_of(self.sparse_parameter)
        if kind.lib == "torch":
            torch = sys.modules["torch"]
            return torch.tensor(values, dtype=torch.long, device=kind.device)
        if kind.lib == "numpy":
            return sys.modules["numpy"].asarray(values, dtype="int64")
        return values

    def _dense(self) -> list[float]:
        values, shape = to_flat(self.sparse_parameter)
        expected = len(self.support) if self.is_sparse else self.dim
        if shape != (expected,):
            raise UnsupportedError("Relevance pursuit requires a one-dimensional sparse parameter.")
        if not self.is_sparse:
            return list(values)
        dense = [0.0] * self.dim
        for i, value in zip(self.support, values, strict=True):
            dense[i] = value
        return dense

    def _replace(self, support: list[int], dense: list[float], sparse: bool) -> None:
        old = self.sparse_parameter
        values = [dense[i] for i in support] if sparse else dense
        value = from_flat(values, (len(values),), kind_of(old))
        if type(old).__name__ == "Parameter" and kind_of(old).lib == "torch":
            value = sys.modules["torch"].nn.Parameter(value, requires_grad=old.requires_grad)
        previous = self._support, self._is_sparse
        self._support, self._is_sparse = support, sparse
        try:
            self.set_sparse_parameter(value)
        except Exception:
            self._support, self._is_sparse = previous
            raise

    def to_sparse(self) -> RelevancePursuitMixin:
        if not self.is_sparse:
            self._replace(self.support.copy(), self._dense(), True)
        return self

    def to_dense(self) -> RelevancePursuitMixin:
        if self.is_sparse:
            self._replace(self.support.copy(), self._dense(), False)
        return self

    def expand_support(self, indices: list[int]) -> RelevancePursuitMixin:
        indices = _indices(indices, self.dim)
        if any(i in self.support for i in indices):
            raise ValueError("Cannot expand support with already active indices.")
        self._replace(self.support + indices, self._dense(), self.is_sparse)
        return self

    def contract_support(self, indices: list[int]) -> RelevancePursuitMixin:
        indices = _indices(indices, self.dim)
        if any(i not in self.support for i in indices):
            raise ValueError("Cannot contract support with inactive indices.")
        dense = self._dense()
        for i in indices:
            dense[i] = 0.0
        self._replace([i for i in self.support if i not in indices], dense, self.is_sparse)
        return self

    def full_support(self) -> RelevancePursuitMixin:
        self.expand_support([i for i in range(self.dim) if i not in self.support])
        return self.to_dense()

    def remove_support(self) -> RelevancePursuitMixin:
        self._replace([], [0.0] * self.dim, self.is_sparse)
        return self

    def _sparse_parameter_gradient(self, mll: Any) -> Any:
        rs = getattr(mll.model, "_rs", None)
        if rs is None or not hasattr(rs, "sparse_outlier_expansion"):
            raise UnsupportedError("Sparse gradients require a native sparse outlier GP.")
        mll.train()
        values = rs.sparse_outlier_expansion(False)
        inactive = [i for i in range(self.dim) if i not in self.support]
        return from_flat(
            [values[i] for i in inactive], (len(inactive),), kind_of(self.sparse_parameter)
        )

    def expansion_objective(self, mll: Any) -> Any:
        return self._sparse_parameter_gradient(mll)

    def support_expansion(self, mll: Any, n: int = 1, modifier: Any = None) -> bool:
        if len(self.support) == self.dim or n <= 0:
            return False
        g = self.expansion_objective(mll)
        modifier = self._expansion_modifier if modifier is None else modifier
        values = list(to_flat(g if modifier is None else modifier(g))[0])
        inactive = [i for i in range(self.dim) if i not in self.support]
        if len(values) != len(inactive):
            raise ValueError("Expansion objective must return one value per inactive index.")
        selected = [
            inactive[j]
            for j in sorted(range(len(values)), key=lambda j: -values[j])[:n]
            if values[j] > 0
        ]
        if not selected:
            return False
        self.expand_support(selected)
        return True

    def support_contraction(self, mll: Any, n: int = 1, modifier: Any = None) -> bool:
        if not self.support or n <= 0:
            return False
        values = from_flat(
            [self._dense()[i] for i in self.support],
            (len(self.support),),
            kind_of(self.sparse_parameter),
        )
        modifier = self._contraction_modifier if modifier is None else modifier
        data = list(to_flat(values if modifier is None else modifier(values))[0])
        indices = sorted(range(len(data)), key=data.__getitem__)[:n]
        self.contract_support([self.support[j] for j in indices])
        return True

    def optimize_mll(
        self,
        mll: Any,
        model_trace: list[Any] | None = None,
        reset_parameters: bool = True,
        reset_dense_parameters: bool = False,
        closure: Any = None,
        optimizer: Any = None,
        closure_kwargs: Any = None,
        optimizer_kwargs: Any = None,
    ) -> Any:
        from botorch_rs.fit import fit_gpytorch_mll

        if reset_parameters:
            self._replace(self.support.copy(), [0.0] * self.dim, self.is_sparse)
        if reset_dense_parameters:
            initialize_dense_parameters(mll.model)
        self.to_sparse()
        fit_gpytorch_mll(
            mll,
            closure=closure,
            optimizer=optimizer,
            closure_kwargs=closure_kwargs,
            optimizer_kwargs=optimizer_kwargs,
        )
        if model_trace is not None:
            model_trace.append(copy.deepcopy(mll.model))
        return mll


def _indices(indices: Any, dim: int) -> list[int]:
    try:
        values = [integer_index(i) if not isinstance(i, bool) else -1 for i in indices]
    except TypeError:
        raise ValueError("Support indices must be integers in range(dim).") from None
    if any(not 0 <= i < dim for i in values):
        raise ValueError("Support indices must be integers in range(dim).")
    if len(set(values)) != len(values):
        raise ValueError("Support indices must be distinct.")
    return values


def _pursuit(
    sparse_module: RelevancePursuitMixin,
    mll: Any,
    backwards: bool,
    sparsity_levels: Any,
    reset_parameters: bool,
    reset_dense_parameters: bool,
    record_model_trace: bool,
    initial_support: Any,
    closure: Any,
    optimizer: Any,
    closure_kwargs: Any,
    optimizer_kwargs: Any,
) -> tuple[RelevancePursuitMixin, Any]:
    guard = getattr(sparse_module, "_require_unbatched_ranking", None)
    if guard is not None:
        guard()
    if backwards and initial_support is None:
        sparse_module.full_support()
    else:
        sparse_module.remove_support()
        if initial_support is not None:
            sparse_module.expand_support(initial_support)
    start = len(sparse_module.support)
    levels = (
        list(range(start + 1) if backwards else range(start, sparse_module.dim + 1))
        if sparsity_levels is None
        else list(sparsity_levels)
    )
    if not levels or any(
        isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= sparse_module.dim
        for k in levels
    ):
        raise ValueError("sparsity_levels must be a nonempty sequence of integers in [0, dim].")
    if any(k > start if backwards else k < start for k in levels):
        raise ValueError(
            "Sparsity levels must follow the direction of the pursuit from initial_support."
        )
    levels = sorted(set(levels), reverse=backwards)
    options = copy.deepcopy(optimizer_kwargs or {})
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    if (
        getattr(mll.model, "_tensor_backend", None) == "torch"
        and optimizer is fit_gpytorch_mll_torch
    ):
        options.setdefault("step_limit", 10000)
    else:
        native_options = options.setdefault("options", {})
        native_options.setdefault("maxiter", 10000)
        native_options.setdefault("ftol", 1e-8)
        native_options.setdefault("gtol", 1e-8)
    trace: list[Any] | None = [] if record_model_trace else None
    kw = dict(
        model_trace=trace,
        reset_parameters=reset_parameters,
        reset_dense_parameters=reset_dense_parameters,
        closure=closure,
        optimizer=optimizer,
        closure_kwargs=closure_kwargs,
        optimizer_kwargs=options,
    )
    sparse_module.optimize_mll(mll, **kw)
    for level in levels:
        current = len(sparse_module.support)
        if level == current:
            continue
        changed = (
            sparse_module.support_contraction(mll, current - level)
            if backwards
            else sparse_module.support_expansion(mll, level - current)
        )
        if not changed:
            warnings.warn(
                "Relevance pursuit terminated because the support could not be changed.",
                RuntimeWarning,
                stacklevel=3,
            )
            break
        sparse_module.optimize_mll(mll, **kw)
    return sparse_module, trace


def forward_relevance_pursuit(
    sparse_module: RelevancePursuitMixin,
    mll: Any,
    sparsity_levels: Any = None,
    reset_parameters: bool = True,
    reset_dense_parameters: bool = False,
    record_model_trace: bool = True,
    initial_support: Any = None,
    closure: Any = None,
    optimizer: Any = None,
    closure_kwargs: Any = None,
    optimizer_kwargs: Any = None,
) -> tuple[RelevancePursuitMixin, Any]:
    return _pursuit(
        sparse_module,
        mll,
        False,
        sparsity_levels,
        reset_parameters,
        reset_dense_parameters,
        record_model_trace,
        initial_support,
        closure,
        optimizer,
        closure_kwargs,
        optimizer_kwargs,
    )


def backward_relevance_pursuit(
    sparse_module: RelevancePursuitMixin,
    mll: Any,
    sparsity_levels: Any = None,
    reset_parameters: bool = True,
    reset_dense_parameters: bool = False,
    record_model_trace: bool = True,
    initial_support: Any = None,
    closure: Any = None,
    optimizer: Any = None,
    closure_kwargs: Any = None,
    optimizer_kwargs: Any = None,
) -> tuple[RelevancePursuitMixin, Any]:
    return _pursuit(
        sparse_module,
        mll,
        True,
        sparsity_levels,
        reset_parameters,
        reset_dense_parameters,
        record_model_trace,
        initial_support,
        closure,
        optimizer,
        closure_kwargs,
        optimizer_kwargs,
    )


def get_posterior_over_support(
    rp_class: Any,
    model_trace: list[Any],
    log_support_prior: Any = None,
    prior_mean_of_support: float | None = None,
) -> tuple[Any, Any]:
    """BoTorch's support posterior using normalized exact MLL values."""
    from botorch_rs.mlls import ExactMarginalLogLikelihood

    if not model_trace:
        raise ValueError("model_trace must not be empty")
    if log_support_prior is None and (
        prior_mean_of_support is None
        or not math.isfinite(prior_mean_of_support)
        or prior_mean_of_support <= 0
    ):
        raise ValueError(
            "A log_support_prior or a finite positive prior_mean_of_support is required."
        )
    if getattr(model_trace[0], "_tensor_backend", None) == "torch":
        from botorch_rs.models._torch_backend_relevance import tensor_support_posterior

        return tensor_support_posterior(
            rp_class, model_trace, log_support_prior, prior_mean_of_support
        )
    kind = model_trace[0]._kind
    sizes = [float(len(rp_class._from_model(m).support)) for m in model_trace]
    values = []
    for size, model in zip(sizes, model_trace, strict=True):
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        value = to_flat(mll())[0]
        if len(value) != 1:
            raise UnsupportedError(
                "Support model comparison currently requires unbatched single-output models."
            )
        prior = (
            (
                -size / float(prior_mean_of_support or 1.0)
                - math.log(float(prior_mean_of_support or 1.0))
            )
            if log_support_prior is None
            else float(to_flat(log_support_prior(from_flat([size], (), kind)))[0][0])
        )
        values.append(value[0] + prior)
    logits = from_flat(values, (len(values),), kind)
    probabilities = _ops.apply(
        "exp", _ops.apply("sub", logits, _ops.reduce("logsumexp", logits, -1))
    )
    return from_flat(sizes, (len(sizes),), kind), probabilities


def initialize_dense_parameters(model: Any) -> tuple[Any, dict[str, Any]]:
    """Initialize constrained native parameters by BoTorch's bound-based rules."""
    if getattr(model, "_tensor_backend", None) == "torch":
        from botorch_rs.models._torch_backend_relevance import tensor_initialize_dense

        return tensor_initialize_dense(model)
    values: dict[str, Any] = {}
    for name, value in model.named_parameters():
        parent = model
        *path, attr = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        constraint = getattr(parent, attr + "_constraint", None)
        if constraint is None:
            raw = 0.0
        elif constraint.initial_value is not None:
            raw = constraint.initial_value
        else:
            lower, upper = constraint.lower_bound, constraint.upper_bound
            chosen = (
                (0.0 if lower == -math.inf else lower + 1.0)
                if upper == math.inf
                else (
                    upper - 1.0 if lower == -math.inf else lower + min(1.0, (upper - lower) / 2.0)
                )
            )
            raw = constraint.inverse_transform(chosen)
        shape = to_flat(value)[1]
        values[name] = from_flat([raw] * math.prod(shape), shape, kind_of(value))
        if hasattr(model, "_set_raw_param"):
            model._set_raw_param(name, values[name])
    if not hasattr(model, "_set_raw_param"):
        model.initialize(**values)
    return model, values


__all__ = [
    "RelevancePursuitMixin",
    "forward_relevance_pursuit",
    "backward_relevance_pursuit",
    "get_posterior_over_support",
    "initialize_dense_parameters",
]
