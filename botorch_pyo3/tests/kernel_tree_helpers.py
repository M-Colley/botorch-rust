"""Kernel trees described as data, built in GPyTorch / BoTorch and in
botorch_rs (``test_parity_6a_kernels.py`` and ``scripts/fixtures_6a_kernels.py``).

A *spec* is a JSON-compatible dict ``{"type": <class name>, "args": [...],
"kwargs": {...}}`` whose values may be

* ``{"__kernel__": spec}``: a sub-kernel;
* ``{"__prior__": [class name, [args]]}``: a prior (``LKJCovariancePrior``
  takes ``[n, eta, prior spec]``);
* ``{"__constraint__": [class name, [args], {kwargs}]}``: a constraint
  (``"transform": null`` disables the transform);
* ``{"__dtype__": "float64"}``: ``torch.float64`` (BoTorch) / the string
  (botorch_rs).

``random_spec`` draws random trees over all supported kernels, with
``active_dims``, constraints and priors.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import math
import random
from collections.abc import Iterator
from typing import Any

#: Kernel classes of GPyTorch (``gpytorch.kernels``) and of BoTorch
#: (``botorch.models.kernels``) by name.
GPYTORCH_KERNELS = {
    "RBFKernel",
    "MaternKernel",
    "RQKernel",
    "PeriodicKernel",
    "LinearKernel",
    "PolynomialKernel",
    "CosineKernel",
    "IndexKernel",
    "ScaleKernel",
    "AdditiveKernel",
    "ProductKernel",
}
BOTORCH_KERNELS = {
    "CategoricalKernel": "categorical",
    "DownsamplingKernel": "downsampling",
    "ExponentialDecayKernel": "exponential_decay",
    "InfiniteWidthBNNKernel": "infinite_width_bnn",
    "LinearTruncatedFidelityKernel": "linear_truncated_fidelity",
    "OrthogonalAdditiveKernel": "orthogonal_additive_kernel",
    "PositiveIndexKernel": "positive_index",
}
GPYTORCH_PRIORS = {
    "GammaPrior",
    "LogNormalPrior",
    "NormalPrior",
    "UniformPrior",
    "HalfCauchyPrior",
    "HalfNormalPrior",
    "LKJCovariancePrior",
}


def _class(lib: str, name: str) -> Any:
    if name in GPYTORCH_KERNELS:
        mod = "gpytorch.kernels" if lib == "bt" else "botorch_rs.kernels"
    elif name in BOTORCH_KERNELS:
        pkg = "botorch.models.kernels" if lib == "bt" else "botorch_rs.models.kernels"
        mod = f"{pkg}.{BOTORCH_KERNELS[name]}"
    elif name in GPYTORCH_PRIORS:
        mod = "gpytorch.priors" if lib == "bt" else "botorch_rs.priors"
    elif name == "BetaPrior":
        mod = "botorch.models.utils.priors" if lib == "bt" else "botorch_rs.models.utils.priors"
    else:
        mod = "gpytorch.constraints" if lib == "bt" else "botorch_rs.constraints"
    return getattr(importlib.import_module(mod), name)


def _value(lib: str, v: Any) -> Any:
    if isinstance(v, dict):
        if "__kernel__" in v:
            return build_kernel(v["__kernel__"], lib)
        if "__prior__" in v:
            name, args = v["__prior__"]
            if name == "LKJCovariancePrior":
                n, eta, sd = args
                sd_prior = _value(lib, {"__prior__": sd})
                if lib == "bt":
                    import torch

                    return _class(lib, name)(n, torch.tensor(float(eta)), sd_prior)
                return _class(lib, name)(n, eta, sd_prior)
            return _class(lib, name)(*args)
        if "__constraint__" in v:
            name, args, kwargs = v["__constraint__"]
            return _class(lib, name)(*args, **dict(kwargs))
        if "__dtype__" in v:
            if lib == "bt":
                import torch

                return getattr(torch, v["__dtype__"])
            return v["__dtype__"]
        return {k: _value(lib, x) for k, x in v.items()}
    if isinstance(v, list):
        return [_value(lib, x) for x in v]
    return v


def build_kernel(spec: dict[str, Any], lib: str) -> Any:
    """The kernel of ``spec`` in BoTorch (``lib="bt"``) or botorch_rs (``"rs"``)."""
    cls = _class(lib, spec["type"])
    args = [_value(lib, a) for a in spec.get("args", [])]
    kwargs = {k: _value(lib, v) for k, v in spec.get("kwargs", {}).items()}
    return cls(*args, **kwargs)


def kernel(type_: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"type": type_, "args": list(args), "kwargs": kwargs}


def sub(spec: dict[str, Any]) -> dict[str, Any]:
    return {"__kernel__": spec}


def prior(name: str, *args: Any) -> dict[str, Any]:
    return {"__prior__": [name, list(args)]}


def constraint(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"__constraint__": [name, list(args), kwargs]}


def with_batch(spec: dict[str, Any], batch: list[int]) -> dict[str, Any]:
    """``spec`` with ``batch_shape=batch`` on every kernel that takes one."""
    spec = copy.deepcopy(spec)

    def walk(s: dict[str, Any]) -> None:
        t = s["type"]
        if t not in ("AdditiveKernel", "ProductKernel"):
            s["kwargs"]["batch_shape"] = list(batch)
        for v in list(s.get("args", [])) + list(s["kwargs"].values()):
            if isinstance(v, dict) and "__kernel__" in v:
                walk(v["__kernel__"])

    walk(spec)
    return spec


# ----------------------------------------------------------------------
# Random trees
# ----------------------------------------------------------------------

_SCALAR_PRIORS = [
    ("GammaPrior", lambda r: [r.choice([1.5, 2.0, 3.0]), r.choice([0.5, 2.0, 6.0])]),
    ("LogNormalPrior", lambda r: [r.choice([-0.5, 0.0, 0.5]), r.choice([0.7, 1.0])]),
    ("NormalPrior", lambda r: [r.choice([0.2, 0.5, 1.0]), r.choice([0.5, 1.5])]),
    ("HalfNormalPrior", lambda r: [r.choice([0.7, 1.5, 3.0])]),
]


def _maybe_prior(r: random.Random, p: float = 0.35) -> dict[str, Any] | None:
    if r.random() > p:
        return None
    name, args = r.choice(_SCALAR_PRIORS)
    return prior(name, *args(r))


def _maybe_constraint(r: random.Random, p: float = 0.25) -> dict[str, Any] | None:
    if r.random() > p:
        return None
    choice = r.randrange(3)
    if choice == 0:
        return constraint("GreaterThan", r.choice([1e-3, 0.05]))
    if choice == 1:
        return constraint("Interval", 0.02, r.choice([3.0, 8.0]))
    return constraint("GreaterThan", 0.05, transform=None)


def _add(kwargs: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        kwargs[key] = value


def _leaf(r: random.Random, dims: list[int]) -> dict[str, Any]:
    """A random leaf kernel acting on ``dims`` (as ``active_dims``)."""
    k = len(dims)
    kinds = ["RBFKernel", "MaternKernel", "RQKernel", "PeriodicKernel", "LinearKernel",
             "PolynomialKernel", "CategoricalKernel", "ExponentialDecayKernel",
             "InfiniteWidthBNNKernel"]  # fmt: skip
    if k == 1:
        kinds.append("CosineKernel")  # positive definite in one dimension only
    kind = r.choice(kinds)
    kw: dict[str, Any] = {"active_dims": list(dims)}
    ard = r.choice([None, k])
    if kind in ("RBFKernel", "MaternKernel", "RQKernel", "PeriodicKernel", "CategoricalKernel",
                "ExponentialDecayKernel"):  # fmt: skip
        if ard is not None:
            kw["ard_num_dims"] = ard
        _add(kw, "lengthscale_prior", _maybe_prior(r))
        _add(kw, "lengthscale_constraint", _maybe_constraint(r))
    if kind == "MaternKernel":
        kw["nu"] = r.choice([0.5, 1.5, 2.5])
    elif kind == "RQKernel":
        _add(kw, "alpha_constraint", _maybe_constraint(r))
    elif kind == "PeriodicKernel":
        _add(kw, "period_length_prior", _maybe_prior(r))
    elif kind == "LinearKernel":
        if ard is not None:
            kw["ard_num_dims"] = ard
        _add(kw, "variance_prior", _maybe_prior(r))
    elif kind == "PolynomialKernel":
        kw["power"] = r.choice([1, 2, 3])
        _add(kw, "offset_prior", _maybe_prior(r))
    elif kind == "CosineKernel":
        _add(kw, "period_length_prior", _maybe_prior(r))
    elif kind == "ExponentialDecayKernel":
        _add(kw, "power_prior", _maybe_prior(r))
        _add(kw, "offset_prior", _maybe_prior(r))
    elif kind == "InfiniteWidthBNNKernel":
        kw["depth"] = r.choice([1, 2, 3])
    return kernel(kind, **kw)


def random_spec(r: random.Random, d: int, depth: int = 0) -> dict[str, Any]:
    """A random kernel tree on ``d`` inputs."""
    if depth >= 2 or r.random() < 0.3:
        dims = sorted(r.sample(range(d), r.randint(1, d)))
        node = _leaf(r, dims)
    else:
        n = r.randint(2, 3)
        kids = [sub(random_spec(r, d, depth + 1)) for _ in range(n)]
        node = kernel(r.choice(["AdditiveKernel", "ProductKernel"]), *kids)
    if r.random() < 0.35:
        kw: dict[str, Any] = {}
        _add(kw, "outputscale_prior", _maybe_prior(r))
        _add(kw, "outputscale_constraint", _maybe_constraint(r))
        node = kernel("ScaleKernel", sub(node), **kw)
    return node


# ----------------------------------------------------------------------
# BoTorch's numerical quirks, fixed for comparisons
# ----------------------------------------------------------------------


@contextlib.contextmanager
def double_precision_half_cauchy() -> Iterator[None]:
    """``HalfCauchyPrior.log_prob`` in double precision: torch's
    ``HalfCauchy.log_prob`` casts its argument to the dtype of the base
    ``Cauchy``, which GPyTorch leaves in float32 when the model is cast to
    double precision. This evaluates the base ``Cauchy`` with its (float32)
    scale in float64 instead."""
    import torch
    from torch.distributions import HalfCauchy

    orig = HalfCauchy.log_prob

    def log_prob(self: Any, value: Any) -> Any:
        scale = self.base_dist.scale.to(value.dtype)
        lp = -math.log(math.pi) - scale.log() - ((value / scale) ** 2).log1p() + math.log(2)
        return torch.where(value >= 0, lp, -math.inf)

    HalfCauchy.log_prob = log_prob
    try:
        yield
    finally:
        HalfCauchy.log_prob = orig


@contextlib.contextmanager
def zero_diagonal_distances() -> Iterator[None]:
    """GPyTorch's ``sq_dist`` with the diagonal of a set's distances to
    itself zeroed also when the inputs require gradients."""
    import gpytorch.kernels.kernel as kk
    import torch

    orig = kk.sq_dist

    def sq_dist(x1: Any, x2: Any, x1_eq_x2: bool = False) -> Any:
        res = orig(x1, x2, x1_eq_x2=x1_eq_x2)
        if x1_eq_x2 and (x1.requires_grad or x2.requires_grad):
            res = res - torch.diag_embed(res.diagonal(dim1=-2, dim2=-1))
        return res

    kk.sq_dist = sq_dist
    try:
        yield
    finally:
        kk.sq_dist = orig
