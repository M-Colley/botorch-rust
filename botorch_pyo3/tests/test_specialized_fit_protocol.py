"""Specialized objectives retain their real loss, shared state and fit protocol."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import warnings

import pytest

torch = pytest.importorskip("torch")

from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import ModelFittingError
from botorch_rs.exceptions.warnings import OptimizationWarning
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import RBFKernel, ScaleKernel
from botorch_rs.means import ConstantMean
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.empirical_gps import (
    EMEmpiricalGaussianProcess,
    EMEmpiricalMarginalLogLikelihood,
    build_shared_gp_model_list,
)
from botorch_rs.models.empirical_gps.em_coordinate_ascent import KLPriorFitMLL
from botorch_rs.models.empirical_gps.utils import ExperimentDataset
from botorch_rs.optim.closures import get_loss_closure_with_grads
from botorch_rs.optim.core import OptimizationResult, OptimizationStatus
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch_rs.optim.utils import get_parameters, sample_all_priors
from botorch_rs.priors import NormalPrior

CONFIGS = [("native", "cpu"), ("torch", "cpu")] + (
    [("torch", "cuda")] if torch.cuda.is_available() else []
)
FAMILIES = ["kl", "em", "shared"]


def _make(family, backend, device, *, warm=False, fixed=False):
    x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double)
    mean = ConstantMean(constant_prior=NormalPrior(0.2, 0.5)).initialize(constant=0.2)
    kernel = ScaleKernel(RBFKernel())
    kernel.base_kernel.lengthscale = 0.4
    kernel.outputscale = 1.1
    histories = [
        ExperimentDataset(x, (3 * x).sin() + 0.2),
        ExperimentDataset(x[[0, 2, 3]], (2 * x[[0, 2, 3]]).sin() - 0.3),
    ]
    if family == "shared":
        _, mll = build_shared_gp_model_list(
            histories,
            mean,
            kernel,
            observation_noise=0.04 if fixed else None,
            backend=backend,
        )
        return mll
    if family == "em":
        model = EMEmpiricalGaussianProcess(
            x[:2],
            histories[0].Y[:2],
            datasets=histories,
            mean_module=mean,
            covar_module=kernel,
            num_em_iterations=2,
            em_convergence_tol=None,
            warm_start_em=warm,
            backend=backend,
        )
        return EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    model = SingleTaskGP(
        x,
        histories[0].Y,
        mean_module=mean,
        covar_module=kernel,
        outcome_transform=None,
        backend=backend,
    )
    covariance = x.new_tensor(
        [[0.9, 0.2, 0.1, 0.04], [0.2, 1.0, 0.3, 0.1], [0.1, 0.3, 0.8, 0.2], [0.04, 0.1, 0.2, 0.7]]
    )
    return KLPriorFitMLL(model.likelihood, model, x, x.new_full((4,), 0.7), covariance)


def _mean(mll):
    model = mll.model.models[0] if hasattr(mll.model, "models") else mll.model
    return model.initial_mean_module if hasattr(model, "initial_mean_module") else model.mean_module


def _constant(mll, value):
    mean = _mean(mll)
    mean.constant = mean.raw_constant.new_tensor(value)


def _float(value):
    return float(to_flat(value)[0][0])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("backend,device", CONFIGS)
def test_default_gradient_closure_uses_specialized_objective(family, backend, device):
    mll = _make(family, backend, device)
    parameters = get_parameters(mll, requires_grad=True)
    loss, gradients = get_loss_closure_with_grads(mll, parameters)()
    if backend == "torch":
        expected = -mll()
        derivatives = torch.autograd.grad(expected, tuple(parameters.values()), allow_unused=True)
        torch.testing.assert_close(loss, expected)
        for actual, reference, parameter in zip(
            gradients, derivatives, parameters.values(), strict=True
        ):
            if reference is None:
                assert actual is None or torch.count_nonzero(actual) == 0
            else:
                torch.testing.assert_close(actual, reference)
                assert actual.device == parameter.device
    else:
        value, raw_gradient = mll.value_and_grad()
        assert _float(loss) == pytest.approx(-value, abs=1e-10, rel=1e-10)
        from botorch_rs.optim.utils.model_utils import _entries

        entries = {entry.name: entry for entry in _entries(mll)}
        for name, actual in zip(parameters, gradients, strict=True):
            entry = entries[name]
            expected = [-g for g in raw_gradient[entry.offset : entry.offset + entry.size]]
            torch.testing.assert_close(
                torch.as_tensor(to_flat(actual)[0]), torch.as_tensor(expected)
            )
        if family == "shared":
            assert len(parameters) == len(mll._rs.parameter_shapes())
            assert list(parameters) == [name for name, _ in mll._rs.parameter_shapes()]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("backend,device", CONFIGS)
def test_custom_optimizer_closure_kwargs_best_attempt_and_failure_rollback(
    monkeypatch, family, backend, device
):
    mll = _make(family, backend, device)
    parameters = get_parameters(mll, requires_grad=True)
    base_closure = get_loss_closure_with_grads(mll, parameters)
    calls, values, offsets = [], [], []
    identities = tuple(id(value) for value in mll.model.parameters()) if backend == "torch" else ()
    monkeypatch.setattr(
        "botorch_rs.optim.utils.model_utils.sample_all_priors", lambda *a, **kw: None
    )

    def closure(*, offset):
        offsets.append(offset)
        value, gradients = base_closure()
        return value + offset, gradients

    def optimizer(module, closure, *, label):
        assert module is mll  # A shared MLL is optimized jointly, never child by child.
        calls.append((_float(_mean(module).constant), label))
        _constant(module, [1.2, 0.2, 0.7][len(calls) - 1])
        value, _ = closure()
        values.append(_float(value))
        return OptimizationResult(1, values[-1], OptimizationStatus.SUCCESS)

    assert (
        fit_gpytorch_mll(
            mll,
            optimizer=optimizer,
            closure=closure,
            closure_kwargs={"offset": 0.125},
            optimizer_kwargs={"label": "real objective"},
            pick_best_of_all_attempts=True,
            max_attempts=3,
        )
        is mll
    )
    selected = min(range(3), key=values.__getitem__)
    assert mll._fit_selected_attempt == selected + 1
    assert _float(_mean(mll).constant) == pytest.approx([1.2, 0.2, 0.7][selected])
    assert [value for value, _ in calls] == pytest.approx([0.2] * 3)
    assert offsets == [0.125] * 3
    assert not mll.training
    if backend == "torch":
        assert tuple(id(value) for value in mll.model.parameters()) == identities
    initial = _float(_mean(mll).constant)

    def failing(module, closure):
        _constant(module, 12)
        raise ArithmeticError("injected failure")

    with pytest.raises(ModelFittingError):
        fit_gpytorch_mll(
            mll, optimizer=failing, max_attempts=2, caught_exception_types=(ArithmeticError,)
        )
    assert _float(_mean(mll).constant) == initial
    assert mll.training


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("backend,device", CONFIGS)
def test_real_sgd_with_gradient_closure_scheduler_and_callback(family, backend, device):
    mll = _make(family, backend, device)
    parameters = get_parameters(mll, requires_grad=True)
    closure = get_loss_closure_with_grads(mll, parameters)
    initial = _float(closure()[0])
    callbacks = []
    fit_gpytorch_mll(
        mll,
        optimizer=fit_gpytorch_mll_torch,
        closure=closure,
        optimizer_kwargs={
            "optimizer": lambda values: torch.optim.SGD(values, lr=1e-4),
            "scheduler": lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.95
            ),
            "step_limit": 3,
            "stopping_criterion": None,
            "callback": lambda values, result: callbacks.append((len(values), result.step)),
        },
        max_attempts=1,
    )
    assert _float(mll()) > -initial
    assert len(callbacks) == 3
    assert not mll.training


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("backend,device", CONFIGS)
def test_scipy_selected_parameters_and_callback_optimize_same_objective(family, backend, device):
    pytest.importorskip("scipy")
    mll = _make(family, backend, device)
    parameters = get_parameters(mll)
    selected = {name: value for name, value in parameters.items() if name.endswith("raw_constant")}
    assert len(selected) == 1
    original = {
        name: to_flat(value)[0] for name, value in parameters.items() if name not in selected
    }
    callbacks = []
    before = _float(mll())
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={
            "parameters": selected,
            "method": "BFGS",
            "options": {"maxiter": 4},
            "callback": lambda values, result: callbacks.append(result.step),
        },
        max_attempts=1,
    )
    assert _float(mll()) >= before - 1e-9
    assert callbacks
    for name, value in get_parameters(mll).items():
        if name in original:
            assert to_flat(value)[0] == original[name]


def test_native_warm_em_failed_attempt_restores_full_moments():
    mll = _make("em", "native", "cpu", warm=True)
    raw = mll.model._rs.raw_parameters()
    moments = mll.model._rs.em_state()
    calls = []

    def failing(module, closure):
        _constant(module, 8)
        get_loss_closure_with_grads(module, get_parameters(module))()
        calls.append(module.model._rs.em_state())
        raise ArithmeticError("warm-state failure")

    with pytest.raises(ModelFittingError):
        fit_gpytorch_mll(
            mll,
            optimizer=failing,
            max_attempts=2,
            caught_exception_types=(ArithmeticError,),
            seed=9,
        )
    assert calls[0] != moments
    assert mll.model._rs.raw_parameters() == raw
    assert mll.model._rs.em_state() == moments


@pytest.mark.parametrize("fixed", [False, True])
def test_native_shared_prior_restart_preserves_tying_and_fixed_noises(fixed):
    mll = _make("shared", "native", "cpu", fixed=fixed)
    initial = mll._rs.raw_parameters()
    sample_all_priors(mll.model, seed=7)
    assert mll._rs.raw_parameters() != initial
    locals_ = [model._rs.raw_parameters() for model in mll.model.models]
    assert locals_[0][1:] == locals_[1][1:]
    if fixed:
        assert [value[0] for value in locals_] == initial[:2]
    with pytest.warns(OptimizationWarning, match="force retry"):
        calls = []

        def optimizer(module, closure):
            calls.append(module._rs.raw_parameters())
            if len(calls) == 1:
                warnings.warn("force retry", OptimizationWarning, stacklevel=2)
            return OptimizationResult(1, -_float(module()), OptimizationStatus.SUCCESS)

        fit_gpytorch_mll(mll, optimizer=optimizer, max_attempts=2, seed=17)
    assert len(calls) == 2 and calls[0] != calls[1]
    locals_ = [model._rs.raw_parameters() for model in mll.model.models]
    assert locals_[0][1:] == locals_[1][1:]


@pytest.mark.parametrize("family", FAMILIES)
def test_default_specialized_native_fit_retains_native_solver_without_scipy(monkeypatch, family):
    mll = _make(family, "native", "cpu")

    def forbidden(*args, **kwargs):
        raise AssertionError("default specialized fitting must retain its native solver")

    monkeypatch.setattr("botorch_rs.optim.fit._fit_generic_scipy", forbidden)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2}}, max_attempts=1)
    assert not mll.training


@pytest.mark.parametrize("family", ["higher_order", "latent_kronecker"])
def test_non_single_task_best_attempt_matches_independent_native_restarts(family):
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.models.higher_order_gp import HigherOrderGP
    from botorch_rs.models.latent_kronecker_gp import LatentKroneckerGP

    x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], dtype=torch.double)
    y = torch.cat(((3 * x).sin(), (4 * x).cos()), dim=-1)
    if family == "higher_order":
        model = HigherOrderGP(x, y, outcome_transform=None, backend="native")
    else:
        model = LatentKroneckerGP(
            x,
            torch.tensor([[0.1], [0.7]], dtype=x.dtype),
            y,
            outcome_transform=None,
            backend="native",
        )
    trials = []
    for attempt in (1, 2):
        candidate = copy.deepcopy(model)
        if attempt > 1:
            sample_all_priors(candidate, seed=11 + attempt)
        trial = ExactMarginalLogLikelihood(candidate.likelihood, candidate)
        result = fit_gpytorch_mll_scipy(trial, options={"maxiter": 2})
        trials.append((result.fval, candidate._rs.raw_parameters()))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={"options": {"maxiter": 2}},
        pick_best_of_all_attempts=True,
        max_attempts=2,
        seed=11,
    )
    selected = min(range(2), key=lambda index: trials[index][0])
    assert mll._fit_selected_attempt == selected + 1
    assert mll._fit_result["fval"] == pytest.approx(trials[selected][0], abs=1e-12, rel=1e-12)
    assert model._rs.raw_parameters() == trials[selected][1]


@pytest.mark.parametrize("backend,device", CONFIGS)
def test_kl_custom_fit_protocol_matches_pinned_reference(backend, device):
    pytest.importorskip("botorch")
    gpytorch = pytest.importorskip("gpytorch")
    from botorch.fit import fit_gpytorch_mll as reference_fit
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.models.empirical_gps.em_coordinate_ascent import KLPriorFitMLL as ReferenceKL

    actual = _make("kl", backend, device)
    x = actual._X_ref
    mean = gpytorch.means.ConstantMean().to(x)
    mean.constant = x.new_tensor(0.2)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(x)
    kernel.base_kernel.lengthscale = x.new_tensor(0.4)
    kernel.outputscale = x.new_tensor(1.1)
    model = ReferenceGP(
        x, torch.sin(3 * x) + 0.2, mean_module=mean, covar_module=kernel, outcome_transform=None
    )
    reference = ReferenceKL(model.likelihood, model, x, actual._mu_em, actual._Sigma_em)
    traces = []
    for module, fit in ((actual, fit_gpytorch_mll), (reference, reference_fit)):
        values = []

        def closure(*, offset, module=module):
            return -module.forward(None, None) + offset, ()

        def optimizer(mll, closure, *, label, module=module, values=values):
            assert mll is module and label == "same optimizer"
            _constant(mll, [1.2, 0.2, 0.7][len(values)])
            loss = _float(closure()[0])
            values.append(loss)
            return OptimizationResult(1, loss, OptimizationStatus.SUCCESS)

        fit(
            module,
            optimizer=optimizer,
            closure=closure,
            closure_kwargs={"offset": 0.125},
            optimizer_kwargs={"label": "same optimizer"},
            max_attempts=3,
            pick_best_of_all_attempts=True,
        )
        traces.append((values, _float(_mean(module).constant), module.training))
    assert traces[0][0] == pytest.approx(traces[1][0], abs=3e-9, rel=3e-9)
    assert traces[0][1:] == traces[1][1:]
