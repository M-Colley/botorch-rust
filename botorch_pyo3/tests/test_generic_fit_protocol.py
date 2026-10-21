"""High-level fitting preserves the public optimizer/closure and retry protocols."""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.exceptions.errors import ModelFittingError, NotPSDError  # noqa: E402
from botorch_rs.exceptions.warnings import OptimizationWarning  # noqa: E402
from botorch_rs.fit import fit_gpytorch_mll  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood  # noqa: E402
from botorch_rs.models import SingleTaskGP  # noqa: E402
from botorch_rs.optim.core import OptimizationResult, OptimizationStatus  # noqa: E402
from botorch_rs.optim.utils import get_parameters  # noqa: E402

CONFIGS = [
    ("native", "cpu"),
    ("torch", "cpu"),
    pytest.param(
        "torch",
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
    ),
]


def _model(backend, device):
    x = torch.linspace(0, 1, 6, dtype=torch.double, device=device).unsqueeze(-1)
    model = SingleTaskGP(x, (4 * x).sin(), backend=backend)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    parameters = get_parameters(mll)
    mean = next(value for name, value in parameters.items() if name.endswith("raw_constant"))
    frozen = next(value for name, value in parameters.items() if "lengthscale" in name)
    frozen.requires_grad_(False)
    return mll, mean, frozen


def _result(loss):
    return OptimizationResult(
        3, loss, OptimizationStatus.SUCCESS, runtime=0.1, message="custom optimizer"
    )


@pytest.mark.parametrize("backend,device", CONFIGS)
def test_custom_optimizer_retries_with_frozen_parameters_and_closure_kwargs(
    monkeypatch, backend, device
):
    mll, mean, frozen = _model(backend, device)
    frozen_initial = frozen.detach().clone()
    calls, closures = [], []

    def sample(model, **kwargs):
        with torch.no_grad():
            mean.fill_(7)
            frozen.fill_(99)

    monkeypatch.setattr("botorch_rs.optim.utils.model_utils.sample_all_priors", sample)

    def closure(*, offset):
        closures.append(offset)
        return mean.sum() + offset, (torch.ones_like(mean),)

    def optimizer(module, closure, *, label):
        calls.append((float(mean.detach()), label))
        torch.testing.assert_close(frozen, frozen_initial)
        closure()
        with torch.no_grad():
            mean.fill_(2 if len(calls) > 1 else 9)
        if len(calls) == 1:
            warnings.warn("retry this fit", OptimizationWarning, stacklevel=2)
        return _result(1.0)

    with pytest.warns(OptimizationWarning, match="retry this fit"):
        assert (
            fit_gpytorch_mll(
                mll,
                optimizer=optimizer,
                closure=closure,
                closure_kwargs={"offset": 4},
                optimizer_kwargs={"label": "accepted"},
                max_attempts=2,
            )
            is mll
        )
    assert [item[1] for item in calls] == ["accepted", "accepted"]
    assert calls[1][0] == 7
    assert closures == [4, 4]
    assert float(mean.detach()) == 2
    assert not mll.training and not mll.model.training
    assert mll._fit_result["attempts"] == 2
    torch.testing.assert_close(frozen, frozen_initial)


@pytest.mark.parametrize("backend,device", CONFIGS)
def test_best_attempt_and_failed_optimizer_restore_actual_model(monkeypatch, backend, device):
    mll, mean, _ = _model(backend, device)
    monkeypatch.setattr(
        "botorch_rs.optim.utils.model_utils.sample_all_priors", lambda *a, **kw: None
    )
    calls = []

    def optimizer(module, closure):
        calls.append(float(mean.detach()))
        with torch.no_grad():
            mean.fill_(len(calls))
        if len(calls) == 2:
            raise NotPSDError("failed second trial")
        return _result(1.0 if len(calls) == 1 else 3.0)

    fit_gpytorch_mll(mll, optimizer=optimizer, max_attempts=3, pick_best_of_all_attempts=True)
    assert calls == [0.0, 0.0, 0.0]
    assert float(mean.detach()) == 1
    assert mll._fit_selected_attempt == 1
    initial = {name: value.detach().clone() for name, value in get_parameters(mll).items()}

    def failing(module, closure):
        with torch.no_grad():
            mean.fill_(123)
        raise NotPSDError("cannot fit")

    with pytest.raises(ModelFittingError):
        fit_gpytorch_mll(mll, optimizer=failing, max_attempts=2)
    for name, value in get_parameters(mll).items():
        torch.testing.assert_close(value, initial[name])
    assert mll.training


@pytest.mark.parametrize("backend,device", CONFIGS)
def test_custom_warning_handler_and_caught_exception_policy(monkeypatch, backend, device):
    mll, mean, _ = _model(backend, device)
    monkeypatch.setattr(
        "botorch_rs.optim.utils.model_utils.sample_all_priors", lambda *a, **kw: None
    )
    calls = []

    def optimizer(module, closure):
        calls.append(1)
        with torch.no_grad():
            mean.fill_(len(calls))
        if len(calls) == 1:
            raise ArithmeticError("retry custom exception")
        warnings.warn("caller accepts this warning", OptimizationWarning, stacklevel=2)
        return _result(0.0)

    fit_gpytorch_mll(
        mll,
        optimizer=optimizer,
        max_attempts=2,
        caught_exception_types=(ArithmeticError,),
        warning_handler=lambda message: True,
    )
    assert len(calls) == 2
    assert float(mean.detach()) == 2


@pytest.mark.parametrize("backend,device", CONFIGS)
def test_supplied_gradient_closure_runs_real_sgd_without_parameter_argument(backend, device):
    from botorch_rs.optim.closures import get_loss_closure_with_grads
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    mll, mean, _ = _model(backend, device)
    initial = mean.detach().clone()
    closure = get_loss_closure_with_grads(mll, get_parameters(mll, requires_grad=True))
    fit_gpytorch_mll(
        mll,
        optimizer=fit_gpytorch_mll_torch,
        closure=closure,
        optimizer_kwargs={
            "optimizer": lambda values: torch.optim.SGD(values, lr=1e-4),
            "step_limit": 3,
            "stopping_criterion": None,
        },
    )
    assert not torch.equal(mean, initial)
    assert not mll.training


@pytest.mark.parametrize("backend,device", CONFIGS)
def test_best_attempt_rejects_nonfinite_objectives_and_restores_initial_state(
    monkeypatch, backend, device
):
    mll, mean, _ = _model(backend, device)
    initial = mean.detach().clone()
    monkeypatch.setattr(
        "botorch_rs.optim.utils.model_utils.sample_all_priors", lambda *a, **kw: None
    )
    values = iter([float("nan"), float("inf")])

    def optimizer(module, closure):
        with torch.no_grad():
            mean.fill_(123)
        return _result(next(values))

    with pytest.raises(ModelFittingError):
        fit_gpytorch_mll(mll, optimizer=optimizer, max_attempts=2, pick_best_of_all_attempts=True)
    torch.testing.assert_close(mean, initial)
    assert mll.training


@pytest.mark.parametrize("backend,device", CONFIGS)
def test_default_warning_policy_accepts_iteration_limit_without_retry(backend, device):
    mll, _, _ = _model(backend, device)
    calls = []

    def optimizer(module, closure):
        calls.append(1)
        warnings.warn(
            "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT", OptimizationWarning, stacklevel=2
        )
        return _result(1.0)

    fit_gpytorch_mll(mll, optimizer=optimizer, max_attempts=3)
    assert calls == [1]
    assert not mll.training
