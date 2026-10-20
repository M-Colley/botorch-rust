"""Best-of-all fitting selects real native fits and rolls back failed trials."""

from __future__ import annotations

import copy
import json

import pytest

from botorch_rs.exceptions.errors import InvalidArgumentError, ModelFittingError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim.utils.model_utils import sample_all_priors
from helpers import as_kind, assert_close, make_data


@pytest.mark.parametrize("outputs", [1, 2])
def test_best_native_attempt_matches_independent_restarts(kind, outputs):
    x, y = make_data(n=9, d=2, seed=22)
    y = [[row[0] * (j + 1) + j for j in range(outputs)] for row in y]
    model = SingleTaskGP(as_kind(x, kind), as_kind(y, kind))
    initial = list(model._rs.raw_parameters())
    trials = []
    options = {"options": {"maxiter": 5}}
    for attempt in range(1, 4):
        candidate = copy.deepcopy(model)
        candidate._rs.set_raw_parameters(initial)
        if attempt > 1:
            sample_all_priors(candidate, seed=31 + attempt)
        mll = ExactMarginalLogLikelihood(candidate.likelihood, candidate)
        fit_gpytorch_mll(mll, optimizer_kwargs=options, max_attempts=1)
        trials.append((mll._fit_result["fval"], candidate._rs.raw_parameters()))
    expected = min(range(3), key=lambda i: trials[i][0])
    actual = ExactMarginalLogLikelihood(model.likelihood, model)
    assert (
        fit_gpytorch_mll(
            actual,
            optimizer_kwargs=options,
            pick_best_of_all_attempts=True,
            max_attempts=3,
            seed=31,
        )
        is actual
    )
    assert_close(model._rs.raw_parameters(), trials[expected][1], rtol=0, atol=0)
    assert actual._fit_result["fval"] == trials[expected][0]
    assert actual._fit_result["attempts"] == 3
    assert actual._fit_result["selected_attempt"] == expected + 1
    assert len(actual._fit_result["attempt_results"]) == 3
    json.dumps(actual._fit_result)
    assert not model.training and not actual.training


def test_all_failed_attempts_restore_original_state(monkeypatch):
    import botorch_rs.fit as fit_module

    x, y = make_data(n=7, d=2, seed=3)
    model = SingleTaskGP(x, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    original = model._rs.raw_parameters()
    calls = []

    def numerical_failure(mll, **kwargs):
        calls.append(list(mll.model._rs.raw_parameters()))
        mll.model.mean_module.constant = 13.0
        raise ModelFittingError("injected numerical failure")

    monkeypatch.setattr(fit_module, "fit_gpytorch_mll", numerical_failure)
    with pytest.raises(ModelFittingError, match="All attempts"):
        fit_gpytorch_mll(mll, pick_best_of_all_attempts=True, max_attempts=3, seed=4)
    assert len(calls) == 3
    assert calls[0] == original
    assert calls[1] != original
    assert model._rs.raw_parameters() == original
    assert mll.training and model.training


def test_best_of_all_runs_native_adam_attempts():
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    x, y = make_data(n=7, d=2, seed=3)
    model = SingleTaskGP(x, y)
    trials = []
    options = {"step_limit": 3, "stopping_criterion": None}
    for attempt in range(1, 4):
        candidate = copy.deepcopy(model)
        if attempt > 1:
            sample_all_priors(candidate, seed=17 + attempt)
        mll = ExactMarginalLogLikelihood(candidate.likelihood, candidate)
        fit_gpytorch_mll(
            mll, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs=options, max_attempts=1
        )
        trials.append((mll._fit_result.fval, candidate._rs.raw_parameters()))
    expected = min(range(3), key=lambda i: trials[i][0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(
        mll,
        optimizer=fit_gpytorch_mll_torch,
        optimizer_kwargs=options,
        pick_best_of_all_attempts=True,
        max_attempts=3,
        seed=17,
    )
    assert_close(model._rs.raw_parameters(), trials[expected][1], rtol=0, atol=0)
    assert mll._fit_result.fval == trials[expected][0]
    assert mll._fit_selected_attempt == expected + 1
    assert len(mll._fit_attempt_results) == 3


def test_invalid_attempt_budget_preserves_model():
    x, y = make_data(n=7, d=2, seed=3)
    model = SingleTaskGP(x, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    original = model._rs.raw_parameters()
    with pytest.raises(InvalidArgumentError, match="at least 1"):
        fit_gpytorch_mll(mll, pick_best_of_all_attempts=True, max_attempts=0)
    assert model._rs.raw_parameters() == original


def test_best_attempt_matches_official_with_identical_restart_values(monkeypatch):
    torch = pytest.importorskip("torch")
    official = pytest.importorskip("botorch.fit")
    from botorch.models import SingleTaskGP as ReferenceGP

    pytest.importorskip("gpytorch")
    from gpytorch.mlls import ExactMarginalLogLikelihood as ReferenceMLL

    import botorch_rs.optim.utils.model_utils as native_utils

    x, y = make_data(n=9, d=2, seed=22)
    xt, yt = torch.tensor(x, dtype=torch.double), torch.tensor(y, dtype=torch.double)
    actual, expected = SingleTaskGP(xt, yt), ReferenceGP(xt, yt)
    restarts = [[0.04, -0.2, 0.2, 0.5], [0.08, 0.3, 0.6, 0.25]]
    native_starts, reference_starts = iter(restarts), iter(restarts)

    def restart_native(model, seed=None):
        model._rs.set_raw_parameters(next(native_starts))

    def restart_reference(model):
        values = iter(next(reference_starts))
        with torch.no_grad():
            for _, parameter in model.named_parameters():
                parameter.copy_(
                    torch.tensor(
                        [next(values) for _ in range(parameter.numel())],
                        dtype=parameter.dtype,
                        device=parameter.device,
                    ).reshape(parameter.shape)
                )

    monkeypatch.setattr(native_utils, "sample_all_priors", restart_native)
    monkeypatch.setattr(official, "sample_all_priors", restart_reference)
    a = ExactMarginalLogLikelihood(actual.likelihood, actual)
    b = ReferenceMLL(expected.likelihood, expected)
    settings = {"options": {"maxiter": 5}}
    fit_gpytorch_mll(a, pick_best_of_all_attempts=True, max_attempts=3, optimizer_kwargs=settings)
    official.fit_gpytorch_mll(
        b, pick_best_of_all_attempts=True, max_attempts=3, optimizer_kwargs=settings
    )
    b.train()
    expected_value = b(expected(*expected.train_inputs), expected.train_targets)
    torch.testing.assert_close(a(), expected_value, rtol=1e-7, atol=1e-8)
    for (name, av), (other_name, bv) in zip(
        actual.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(av, bv, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("pick_best", [False, True])
@pytest.mark.parametrize("method", ["L-BFGS-B", "Adam"])
def test_high_level_fitting_preserves_frozen_leaves(outputs, pick_best, method):
    torch = pytest.importorskip("torch")
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    x, y = make_data(n=9, d=2, seed=22)
    y = [[row[0] * (j + 1) + j for j in range(outputs)] for row in y]
    model = SingleTaskGP(torch.tensor(x, dtype=torch.double), torch.tensor(y, dtype=torch.double))
    parameters = dict(model.named_parameters())
    noise = parameters["likelihood.noise_covar.raw_noise"]
    noise.requires_grad_(False)
    before = noise.detach().clone()
    free_before = model.mean_module.raw_constant.detach().clone()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    options = {"options": {"maxiter": 5}} if method == "L-BFGS-B" else {"step_limit": 3}
    fit_gpytorch_mll(
        mll,
        optimizer=None if method == "L-BFGS-B" else fit_gpytorch_mll_torch,
        optimizer_kwargs=options,
        max_attempts=3,
        pick_best_of_all_attempts=pick_best,
        seed=13,
    )
    assert dict(model.named_parameters())["likelihood.noise_covar.raw_noise"] is noise
    torch.testing.assert_close(noise, before, rtol=0, atol=0)
    assert not noise.requires_grad
    assert not torch.equal(model.mean_module.raw_constant, free_before)
    assert not mll.training and not model.training


def test_fitting_a_fully_frozen_model_is_an_exact_noop():
    torch = pytest.importorskip("torch")
    x, y = make_data(n=7, d=2, seed=3)
    model = SingleTaskGP(torch.tensor(x, dtype=torch.double), torch.tensor(y, dtype=torch.double))
    model.requires_grad_(False)
    original = list(model._rs.raw_parameters())
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    expected = -float(mll())
    fit_gpytorch_mll(mll)
    assert model._rs.raw_parameters() == original
    assert mll._fit_result.step == 0
    assert mll._fit_result.fval == expected
