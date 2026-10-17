"""Specs of input and outcome transforms for the phase-6b equivalence tests.

A spec is JSON-serializable, so that the fixture generator, the Rust test
(``botorch_core/tests/transforms_6b_parity.rs``) and the live Python tests
build the same transforms with BoTorch (``lib="bt"``) and botorch_rs
(``lib="rs"``):

* an input spec is a list of steps (a ``ChainedInputTransform`` of them with
  the keys ``s0``, ``s1``, ... if there are several), every step a dict with
  ``type`` (the class name) and its keyword arguments (``flags``: the
  transform flags; ``training: false``: the step is put in eval mode before
  the model is built);
* an outcome spec is a list of steps likewise (a ``ChainedOutcomeTransform``
  if there are several);
* priors are ``{"__prior__": [class name, [args]]}``.

``random_input_spec`` / ``random_outcome_spec`` draw the random chains of the
equivalence sweep. ``botorch_fixes`` patches BoTorch main's transform bugs
for the comparisons (meta-pytorch/botorch#3389: the transforms of a
``BatchBroadcastedInputTransform`` are registered as submodules, and the
standard deviation of ``InputStandardize`` on a single point is treated as
zero spread).
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import random
from collections.abc import Iterator
from typing import Any

import torch

DT = torch.float64


def _lib(lib: str, module: str) -> Any:
    prefix = "botorch" if lib == "bt" else "botorch_rs"
    return importlib.import_module(f"{prefix}.{module}")


def _prior(lib: str, v: Any) -> Any:
    if v is None:
        return None
    name, args = v["__prior__"]
    if name == "BetaPrior":
        return _lib(lib, "models.utils.priors").BetaPrior(*args)
    mod = importlib.import_module("gpytorch.priors" if lib == "bt" else "botorch_rs.priors")
    return getattr(mod, name)(*args)


def prior(name: str, *args: Any) -> dict[str, Any]:
    return {"__prior__": [name, list(args)]}


def _tensor(v: Any, lib: str, long: bool = False) -> Any:
    if v is None:
        return None
    t = torch.tensor(v, dtype=torch.long if long else DT)
    return t


def _flags(
    s: dict[str, Any], defaults: tuple[bool, bool, bool] = (True, True, True)
) -> dict[str, bool]:
    f = s.get("flags", defaults)
    return {
        "transform_on_train": bool(f[0]),
        "transform_on_eval": bool(f[1]),
        "transform_on_fantasize": bool(f[2]),
    }


def build_input_step(s: dict[str, Any], lib: str) -> Any:
    tfs = _lib(lib, "models.transforms.input")
    t = s["type"]
    if t == "Normalize":
        kw: dict[str, Any] = {}
        for key in ("indices", "center", "reverse", "min_range", "learn_bounds"):
            if key in s:
                kw[key] = s[key]
        if "bounds" in s:
            kw["bounds"] = _tensor(s["bounds"], lib)
        if "flags" in s:
            kw.update(_flags(s))
        return tfs.Normalize(d=s["d"], **kw)
    if t == "InputStandardize":
        kw = {k: s[k] for k in ("indices", "reverse", "min_std") if k in s}
        if "flags" in s:
            kw.update(_flags(s))
        return tfs.InputStandardize(d=s["d"], **kw)
    if t == "AffineInputTransform":
        kw = {k: s[k] for k in ("indices", "reverse") if k in s}
        if "flags" in s:
            kw.update(_flags(s))
        return tfs.AffineInputTransform(
            d=s["d"],
            coefficient=_tensor(s["coefficient"], lib),
            offset=_tensor(s["offset"], lib),
            **kw,
        )
    if t == "Warp":
        kw = {k: s[k] for k in ("eps", "reverse") if k in s}
        if "flags" in s:
            kw.update(_flags(s))
        if s.get("bounds") is not None:
            kw["bounds"] = _tensor(s["bounds"], lib)
        if s.get("batch_shape"):
            kw["batch_shape"] = torch.Size(s["batch_shape"])
        return tfs.Warp(
            d=s["d"],
            indices=s["indices"],
            concentration0_prior=_prior(lib, s.get("c0_prior")),
            concentration1_prior=_prior(lib, s.get("c1_prior")),
            **kw,
        )
    if t == "Round":
        cat = s.get("categorical_features")
        return tfs.Round(
            integer_indices=s.get("integer_indices"),
            categorical_features={int(k): int(v) for k, v in cat.items()} if cat else None,
            approximate=s.get("approximate", False),
            tau=s.get("tau", 1e-3),
            **_flags(s),
        )
    if t == "Log10":
        return tfs.Log10(indices=s["indices"], reverse=s.get("reverse", False), **_flags(s))
    if t == "AppendFeatures":
        return tfs.AppendFeatures(
            feature_set=_tensor(s["feature_set"], lib),
            skip_expand=s.get("skip_expand", False),
            **_flags(s, (False, True, False)),
        )
    if t == "InteractionFeatures":
        return tfs.InteractionFeatures(indices=s.get("indices"))
    if t == "FilterFeatures":
        return tfs.FilterFeatures(
            feature_indices=torch.tensor(s["feature_indices"], dtype=torch.long), **_flags(s)
        )
    if t == "OneHotToNumeric":
        cat = {int(k): int(v) for k, v in s["categorical_features"].items()}
        return tfs.OneHotToNumeric(dim=s["dim"], categorical_features=cat, **_flags(s))
    if t == "NumericToCategoricalEncoding":
        from torch.nn.functional import one_hot

        cat = {int(k): int(v) for k, v in s["categorical_features"].items()}
        enc = {k: functools.partial(one_hot, num_classes=v) for k, v in cat.items()}
        return tfs.NumericToCategoricalEncoding(
            dim=s["dim"], categorical_features=cat, encoders=enc, **_flags(s)
        )
    if t == "LearnedFeatureImputation":
        fi = {int(k): list(v) for k, v in s["feature_indices"].items()}
        return tfs.LearnedFeatureImputation(
            feature_indices=fi,
            d=s["d"],
            target_task=s.get("target_task"),
            bounds=_tensor(s.get("bounds"), lib),
            **_flags(s),
        )
    if t == "InputPerturbation":
        return tfs.InputPerturbation(
            perturbation_set=_tensor(s["perturbation_set"], lib),
            bounds=_tensor(s.get("bounds"), lib),
            indices=s.get("indices"),
            multiplicative=s.get("multiplicative", False),
            **_flags(s, (False, True, False)),
        )
    if t == "BatchBroadcastedInputTransform":
        return tfs.BatchBroadcastedInputTransform(
            transforms=[build_input(sub, lib) for sub in s["transforms"]],
            broadcast_index=s.get("broadcast_index", -3),
        )
    raise ValueError(f"unknown input transform {t}")


def build_input(spec: list[dict[str, Any]], lib: str) -> Any:
    steps = [build_input_step(s, lib) for s in spec]
    for s, tf in zip(spec, steps, strict=True):
        if s.get("training") is False:
            tf.eval()
    if len(steps) == 1:
        return steps[0]
    tfs = _lib(lib, "models.transforms.input")
    return tfs.ChainedInputTransform(**{f"s{i}": tf for i, tf in enumerate(steps)})


def build_outcome_step(s: dict[str, Any], lib: str, m: int) -> Any:
    tfs = _lib(lib, "models.transforms.outcome")
    t = s["type"]
    outputs = s.get("outputs")
    if t == "Standardize":
        return tfs.Standardize(
            m=s.get("m", m),
            outputs=outputs,
            batch_shape=torch.Size(s.get("batch_shape", [])),
            min_stdv=s.get("min_stdv", 1e-8),
        )
    if t == "Log":
        return tfs.Log(outputs=outputs)
    if t == "Power":
        return tfs.Power(power=s["power"], outputs=outputs)
    if t == "Bilog":
        return tfs.Bilog(outputs=outputs)
    if t == "StratifiedStandardize":
        return tfs.StratifiedStandardize(
            stratification_idx=s["stratification_idx"],
            all_task_values=torch.tensor(s["task_values"], dtype=torch.long),
        )
    raise ValueError(f"unknown outcome transform {t}")


def build_outcome(spec: list[dict[str, Any]] | None, lib: str, m: int) -> Any:
    if spec is None:
        return None
    steps = [build_outcome_step(s, lib, m) for s in spec]
    if len(steps) == 1:
        return steps[0]
    tfs = _lib(lib, "models.transforms.outcome")
    return tfs.ChainedOutcomeTransform(**{f"o{i}": tf for i, tf in enumerate(steps)})


def has_moments(spec: list[dict[str, Any]] | None) -> bool:
    """Whether the posterior of a model with this outcome transform has a
    mean and variance (no ``Power`` or ``Bilog``)."""
    return not any(s["type"] in ("Power", "Bilog") for s in (spec or []))


def is_nonlinear(spec: list[dict[str, Any]] | None) -> bool:
    return any(s["type"] in ("Log", "Power", "Bilog") for s in (spec or []))


@contextlib.contextmanager
def botorch_fixes() -> Iterator[None]:
    """BoTorch main with the transform fixes of meta-pytorch/botorch#3389:
    ``BatchBroadcastedInputTransform`` registers its transforms as submodules
    (so that they follow the model's mode) and ``InputStandardize`` treats
    the NaN standard deviation of a single point as zero spread."""
    import botorch.models.transforms.input as bi
    from torch.nn import ModuleList

    orig_init = bi.BatchBroadcastedInputTransform.__init__
    orig_update = bi.InputStandardize._update_coefficients

    def init(self: Any, transforms: Any, broadcast_index: int = -3) -> None:
        orig_init(self, transforms, broadcast_index)
        self.transforms = ModuleList(self.transforms)

    def update(self: Any, X: Any) -> None:
        batch_ndim = min(len(self.batch_shape), X.ndim - 2)
        reduce_dims = (*range(X.ndim - batch_ndim - 2), X.ndim - 2)
        coefficient, offset = (
            values.unsqueeze(-2) for values in torch.std_mean(X, dim=reduce_dims, unbiased=True)
        )
        almost_zero = ~(coefficient >= self.min_std)
        self._coefficient = torch.where(almost_zero, 1.0, coefficient)
        self._offset = torch.where(almost_zero, 0.0, offset)

    bi.BatchBroadcastedInputTransform.__init__ = init
    bi.InputStandardize._update_coefficients = update
    try:
        yield
    finally:
        bi.BatchBroadcastedInputTransform.__init__ = orig_init
        bi.InputStandardize._update_coefficients = orig_update


# -----------------------------------------------------------------------------
# Random chains (equivalence sweep)
# -----------------------------------------------------------------------------


def _random_flags(r: random.Random) -> list[bool]:
    return [True, True, r.random() < 0.8]


def random_input_spec(r: random.Random, d: int) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """A random chain of one-to-one input transforms of ``d``-dimensional
    inputs in the unit cube, its output dimension and data hints (``log10``:
    shift the inputs away from zero)."""
    steps: list[dict[str, Any]] = []
    hints: dict[str, Any] = {}
    dim = d
    k = r.randint(1, 3)
    learned_after = False
    for _ in range(k):
        kind = r.choice(
            [
                "Normalize",
                "InputStandardize",
                "Affine",
                "Warp",
                "Log10",
                "Round",
                "Interaction",
                "Filter",
            ]
        )
        idx = sorted(r.sample(range(dim), r.randint(1, dim)))
        if kind == "Normalize" and not learned_after:
            s: dict[str, Any] = {"type": "Normalize", "d": dim}
            if r.random() < 0.5:
                s["indices"] = idx
            if r.random() < 0.3:
                s["center"] = r.choice([0.0, 0.25])
            steps.append(s)
        elif kind == "InputStandardize" and not learned_after:
            s = {"type": "InputStandardize", "d": dim}
            if r.random() < 0.5:
                s["indices"] = idx
            steps.append(s)
        elif kind == "Affine":
            kk = len(idx)
            steps.append(
                {
                    "type": "AffineInputTransform",
                    "d": dim,
                    "indices": idx,
                    "coefficient": [round(r.uniform(0.5, 2.0), 3) for _ in range(kk)],
                    "offset": [round(r.uniform(-0.3, 0.3), 3) for _ in range(kk)],
                }
            )
        elif kind == "Warp" and not steps:
            s = {"type": "Warp", "d": dim, "indices": idx}
            if r.random() < 0.5:
                s["c0_prior"] = prior("LogNormalPrior", 0.0, 0.75**0.5)
                s["c1_prior"] = prior("LogNormalPrior", 0.0, 0.75**0.5)
            steps.append(s)
            learned_after = True
        elif kind == "Log10" and not steps:
            steps.append({"type": "Log10", "indices": idx})
            hints["log10"] = True
        elif kind == "Round" and not steps:
            steps.append(
                {
                    "type": "Round",
                    "integer_indices": idx[:1],
                    "approximate": True,
                    "tau": 0.2,
                    "flags": _random_flags(r),
                }
            )
        elif kind == "Interaction" and dim <= 3:
            steps.append({"type": "InteractionFeatures", "indices": idx if len(idx) > 1 else None})
            n = len(idx) if len(idx) > 1 else dim
            dim += n * (n - 1) // 2
        elif kind == "Filter" and dim > 1:
            keep = sorted(r.sample(range(dim), r.randint(1, dim - 1)))
            steps.append({"type": "FilterFeatures", "feature_indices": keep})
            dim = len(keep)
    if not steps:
        steps.append({"type": "Normalize", "d": d, "center": 0.25})
    return steps, dim, hints


def random_outcome_spec(r: random.Random) -> list[dict[str, Any]] | None:
    """A random outcome transform of positive targets."""
    kind = r.choice(["Log", "LogStd", "Power", "Bilog", "BilogStd", "Std", "None"])
    return {
        "Log": [{"type": "Log"}],
        "LogStd": [{"type": "Log"}, {"type": "Standardize"}],
        "Power": [{"type": "Power", "power": r.choice([0.5, 2.0])}],
        "Bilog": [{"type": "Bilog"}],
        "BilogStd": [{"type": "Bilog"}, {"type": "Standardize"}],
        "Std": [{"type": "Standardize"}],
        "None": None,
    }[kind]
