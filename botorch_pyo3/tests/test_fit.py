"""ExactMarginalLogLikelihood and fit_gpytorch_mll."""

from __future__ import annotations

import pytest

from botorch_rs.exceptions import BotorchRsError, UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms import Normalize
from helpers import as_kind, check_kind, flatten, make_data


def test_fit_improves_the_mll(kind: str) -> None:
    X, Y = make_data(n=15, d=3, seed=2)
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind), input_transform=Normalize(d=3))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = mll()
    check_kind(before, kind)
    assert mll.training
    out = fit_gpytorch_mll(mll)
    assert out is mll and not mll.training and not model.training
    after = float(mll())
    assert after > float(before)
    info = mll._fit_result
    assert info["status"] == "SUCCESS" and info["attempts"] == 1
    assert info["fval"] == pytest.approx(-after, rel=1e-12)
    # The fitted noise respects BoTorch's lower bound of 1e-4.
    assert flatten(model.likelihood.noise)[0] >= 1e-4


def test_fit_options() -> None:
    X, Y = make_data(n=10, d=2, seed=3)
    model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2, "disp": False}})
    assert mll._fit_result["status"] == "STOPPED"
    assert mll._fit_result["nit"] == 2
    fit_gpytorch_mll(mll, optimizer_kwargs={"method": "L-BFGS-B"}, max_attempts=2, seed=1)
    assert mll._fit_result["status"] == "SUCCESS"
    for kwargs in (
        {"closure": lambda: None},
        {"optimizer": object()},
        {"optimizer_kwargs": {"method": "Adam"}},
        {"optimizer_kwargs": {"options": {"unknown": 1}}},
        {"optimizer_kwargs": {"timeout_sec": 1.0}},
        {"pick_best_of_all_attempts": True},
    ):
        with pytest.raises(UnsupportedError):
            fit_gpytorch_mll(mll, **kwargs)
    with pytest.raises(BotorchRsError):
        fit_gpytorch_mll(mll, max_attempts=0)
    with pytest.raises(UnsupportedError):
        fit_gpytorch_mll(object())


def test_mll_requires_the_model_likelihood() -> None:
    X, Y = make_data(n=6, d=2, seed=4)
    model = SingleTaskGP(X, Y)
    other = SingleTaskGP(X, Y)
    with pytest.raises(UnsupportedError):
        ExactMarginalLogLikelihood(other.likelihood, model)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with pytest.raises(UnsupportedError):
        mll(object(), object())


def test_mll_gradient_matches_finite_differences() -> None:
    X, Y = make_data(n=12, d=2, seed=5)
    model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    theta = [0.05, 0.1, 0.4, 0.7]  # raw noise, constant, lengthscales
    value, grad = mll.value_and_grad(theta)

    def at(params: list[float]) -> float:
        (values, shape) = model._rs.mll(params)
        assert shape == []
        return values[0]

    assert value == pytest.approx(at(theta), rel=1e-15)
    h = 1e-6
    for i in range(len(theta)):
        up = list(theta)
        down = list(theta)
        up[i] += h
        down[i] -= h
        fd = (at(up) - at(down)) / (2 * h)
        assert grad[i] == pytest.approx(fd, rel=1e-6, abs=1e-8)
