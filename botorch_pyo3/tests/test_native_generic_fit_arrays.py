"""Generic fitters retain Rust state across array kinds and custom closures."""

from __future__ import annotations

import math
from functools import partial

import pytest

from botorch_rs.mlls import ExactMarginalLogLikelihood, VariationalELBO
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.optim.closures import get_loss_closure_with_grads
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch_rs.optim.utils import get_parameters
from helpers import as_kind, flatten


def _scale(value, scale):
    return [_scale(item, scale) for item in value] if isinstance(value, list) else value * scale


def _mll(kind, variational):
    x = as_kind([[0.05, 0.13], [0.22, 0.31], [0.43, 0.52], [0.67, 0.73], [0.88, 0.96]], kind)
    y = as_kind([[0.12], [0.62], [0.45], [-0.17], [0.31]], kind)
    if variational:
        model = SingleTaskVariationalGP(x, y, inducing_points=3, backend="native")
        return VariationalELBO(model.likelihood, model.model, num_data=5), model.model._rs
    model = SingleTaskGP(x, y, backend="native")
    return ExactMarginalLogLikelihood(model.likelihood, model), model._rs


@pytest.mark.parametrize("variational", [False, True])
@pytest.mark.parametrize("custom_closure", [False, True])
def test_scipy_native_array_shapes_custom_closures_and_callbacks(kind, variational, custom_closure):
    pytest.importorskip("scipy")
    mll, native = _mll(kind, variational)
    parameters = get_parameters(mll, requires_grad=True)
    base = get_loss_closure_with_grads(mll, parameters)
    initial = list(native.raw_parameters())
    before = flatten(base()[0])[0]
    calls = []

    def closure(*, scale):
        value, gradients = base()
        return _scale(value, scale), tuple(_scale(g, scale) for g in gradients)

    result = fit_gpytorch_mll_scipy(
        mll,
        closure=closure if custom_closure else None,
        closure_kwargs={"scale": 1.0} if custom_closure else None,
        callback=lambda params, result: calls.append(result.fval),
        options={"maxiter": 4},
    )
    assert math.isfinite(result.fval) and calls
    assert list(native.raw_parameters()) != initial
    assert flatten(base()[0])[0] <= before + 1e-8


@pytest.mark.parametrize("variational", [False, True])
def test_torch_optimizer_bridges_native_arrays_and_supplied_gradient_closures(kind, variational):
    torch = pytest.importorskip("torch")
    mll, native = _mll(kind, variational)
    parameters = get_parameters(mll, requires_grad=True)
    base = get_loss_closure_with_grads(mll, parameters)
    initial = list(native.raw_parameters())
    before = flatten(base()[0])[0]

    def closure(*, scale):
        value, gradients = base()
        return _scale(value, scale), tuple(_scale(g, scale) for g in gradients)

    result = fit_gpytorch_mll_torch(
        mll,
        closure=closure,
        closure_kwargs={"scale": 1.0},
        optimizer=partial(torch.optim.SGD, lr=1e-5),
        step_limit=3,
        stopping_criterion=None,
    )
    assert math.isfinite(result.fval)
    assert list(native.raw_parameters()) != initial
    assert flatten(base()[0])[0] < before
