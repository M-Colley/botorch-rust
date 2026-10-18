"""Shared helpers of the live comparisons of phase 8b with BoTorch
(``test_parity_8b_acquisition_rest.py``,
``test_equivalence_8b_acquisition_rest.py``): model and dataset pairs, the
corrected BoTorch of the multi-step look-ahead, and the Python subclasses of
the sample-reducing acquisition functions used on both sides.

Import only after ``importorskip_botorch()``: this module requires torch and
BoTorch.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import Iterator
from typing import Any

import gpytorch
import torch
from botorch import settings as bt_settings
from botorch.acquisition import multi_step_lookahead as bt_msl
from botorch.utils.datasets import SupervisedDataset as BtDataset

from botorch_rs.utils.datasets import SupervisedDataset as RsDataset
from parity_helpers import ModelSpec, list_pair, make_data, model_pair

DT = torch.float64

#: The model cases: ``(spec(s), seed)``.
CASES: dict[str, tuple[Any, int]] = {
    "m1": (ModelSpec(n=8, d=2), 3),
    "m1_norm": (ModelSpec(n=9, d=3, normalize=True, kernel="matern52"), 5),
    "m1_fixed": (ModelSpec(n=7, d=2, fixed_noise=True), 7),
    "m1_nostd": (ModelSpec(n=8, d=2, standardize=False), 9),
    "m2": (ModelSpec(n=8, d=2, m=2), 11),
    "list": ([ModelSpec(n=8, d=2), ModelSpec(n=8, d=2, normalize=True)], 13),
}


@functools.cache
def case(name: str) -> tuple[Any, Any, Any, Any]:
    """``(bt, rs, X, Y)``: the BoTorch and botorch_rs models (identical
    hyperparameters, eval mode), the training inputs and the (untransformed)
    training targets of all outputs (the sub-models of the model list share
    their inputs)."""
    spec, seed = CASES[name]
    if isinstance(spec, list):
        bt, rs, X = list_pair(spec, seed)
        ys = []
        for i, s in enumerate(spec):
            Xi, Yi, _ = make_data(s, seed + 13 * i)
            ys.append(Yi)
            # The sub-models are built on their own inputs: the datasets use
            # the inputs of the first one (a block design needs shared X).
            del Xi
        return bt, rs, X, torch.cat(ys, dim=-1)
    bt, rs, X = model_pair(spec, seed)
    _, Y, _ = make_data(spec, seed)
    return bt, rs, X, Y


def datasets(X: Any, Y: Any, lib: str, block: bool = False) -> Any:
    """A ``SupervisedDataset`` of BoTorch or botorch_rs (``lib``), or with
    ``block`` a dict of one-output datasets with shared inputs."""
    cls = BtDataset if lib == "botorch" else RsDataset
    names = [f"x{i}" for i in range(X.shape[-1])]
    if not block:
        return cls(X, Y, feature_names=names, outcome_names=[f"y{i}" for i in range(Y.shape[-1])])
    return {
        i: cls(X, Y[..., i : i + 1], feature_names=names, outcome_names=[f"y{i}"])
        for i in range(Y.shape[-1])
    }


@contextlib.contextmanager
def corrected_lookahead() -> Iterator[None]:
    """BoTorch's ``qMultiStepLookahead`` with exact gradients: the value
    functions of the fantasy models are evaluated with test caches that
    track the fantasies (``propagate_grads``) and without ``fast_pred_var``."""
    with bt_settings.propagate_grads(True), gpytorch.settings.fast_pred_var(False):
        yield


@contextlib.contextmanager
def single_output_collapse_fix() -> Iterator[None]:
    """BoTorch's ``collapse_fantasy_base_samples=False`` with the batch range
    of single-output posteriors (whose base samples have no output
    dimension): independent base samples per fantasy, shared by the
    t-batches (BoTorch's override is that of multi-output posteriors)."""
    from botorch.models import ModelListGP

    orig = bt_msl.qMultiStepLookahead._set_samplers_batch_range

    def fixed(self: Any, batch_shape: Any) -> None:
        if self.model.num_outputs > 1 and not isinstance(self.model, ModelListGP):
            # Multi-output posteriors: BoTorch's override is right.
            orig(self, batch_shape)
            return
        start = -1 - len(batch_shape)
        for s in self.samplers:
            s.batch_range_override = (start, -1)

    bt_msl.qMultiStepLookahead._set_samplers_batch_range = fixed
    try:
        yield
    finally:
        bt_msl.qMultiStepLookahead._set_samplers_batch_range = orig


def best_f_argfac(analytic: bool) -> Any:
    """The corrected ``make_best_f`` of BoTorch: the best untransformed target
    (with a trailing dimension for analytic value functions, which BoTorch
    broadcasts against the outputs)."""

    def argfac(model: Any, X: Any) -> dict:
        t = model.train_targets
        if hasattr(model, "outcome_transform"):
            t = model.outcome_transform.untransform(t.unsqueeze(-1))[0].squeeze(-1)
        b = t.max(dim=-1).values
        return {"best_f": b.unsqueeze(-1) if analytic else b}

    return argfac


def copy_lookahead_samplers(a_bt: Any, a_rs: Any) -> None:
    """Installs the base samples of BoTorch's fantasy and inner samplers (drawn
    by a first evaluation) in botorch_rs's."""
    for sb, sr in zip(a_bt.samplers, a_rs.samplers, strict=True):
        z = sb.base_samples
        f = z.shape[0]
        sr._rs.set_base_samples((z.reshape(f, -1).flatten().tolist(), (f, z[0].numel())))
    for sb, sr in zip(a_bt.inner_samplers, a_rs.inner_samplers, strict=True):
        if sb is None or sb.base_samples is None:
            continue
        z = sb.base_samples
        f = z.shape[0]
        sr._rs.set_base_samples((z.reshape(f, -1).flatten().tolist(), (f, z[0].numel())))


def values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.detach().clone().requires_grad_(True)
    v = acqf(Xg)
    (g,) = torch.autograd.grad(v.sum(), Xg)
    return v.detach(), g


# ---------------------------------------------------------------------------
# Python subclasses of the sample-reducing acquisition functions (the same
# code for BoTorch's and botorch_rs's base classes)
# ---------------------------------------------------------------------------


def improvement_subclass(base: type) -> type:
    """A qEI-like subclass: ``max(obj - best_f, 0)``."""

    class Improvement(base):  # type: ignore[misc, valid-type]
        def __init__(self, model: Any, best_f: float, **kwargs: Any) -> None:
            super().__init__(model, **kwargs)
            self.best_f = best_f

        def _sample_forward(self, obj: Any) -> Any:
            return (obj - self.best_f).clamp_min(0)

    return Improvement


def bounded_subclass(base: type) -> type:
    """A subclass with a smooth, signed utility: ``tanh(obj) + obj^2 / 4``."""

    class Bounded(base):  # type: ignore[misc, valid-type]
        def _sample_forward(self, obj: Any) -> Any:
            return torch.tanh(obj) + 0.25 * obj.square()

    return Bounded


def log_improvement_subclass(base: type) -> type:
    """A qLogEI-like subclass of ``LogImprovementMCAcquisitionFunction``:
    ``log(softplus(obj - best_f, beta=20))``."""

    class LogImprovement(base):  # type: ignore[misc, valid-type]
        def __init__(self, model: Any, best_f: float, **kwargs: Any) -> None:
            super().__init__(model, **kwargs)
            self.best_f = best_f

        def _sample_forward(self, obj: Any) -> Any:
            return torch.nn.functional.softplus(obj - self.best_f, beta=20.0).log()

    return LogImprovement


def accepts(fn: Any, name: str) -> bool:
    return name in inspect.signature(fn).parameters
