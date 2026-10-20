"""Noisy acquisition tapes preserve cached constants and actual sampling paths."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition.logei import qLogNoisyExpectedImprovement as BtLogNEI  # noqa: E402
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement as BtNEI  # noqa: E402
from botorch.acquisition.objective import GenericMCObjective as BtObjective  # noqa: E402
from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtScalarize  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from botorch.settings import propagate_grads  # noqa: E402
from gpytorch.settings import fast_pred_var  # noqa: E402
from so_acquisition_helpers import rs_sampler, seeded_draws  # noqa: E402
from test_posterior_parameter_gradients import model_pair  # noqa: E402

from botorch_rs.acquisition.logei import qLogNoisyExpectedImprovement as LogNEI  # noqa: E402
from botorch_rs.acquisition.monte_carlo import qNoisyExpectedImprovement as NEI  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    GenericMCObjective,  # noqa: E402
    ScalarizedPosteriorTransform,  # noqa: E402
)

S = 32
pytestmark = pytest.mark.filterwarnings("ignore")


def refresh_live_training_solve(model):
    # Baseline construction intentionally runs no_grad and may populate detached
    # predictive caches. Preserve the acquisition's detached baseline quantities,
    # but rebuild the current posterior's full training solve under propagate_grads.
    for child in getattr(model, "models", (model,)):
        child.eval()
        if hasattr(child, "input_transform"):
            child.set_train_data(
                inputs=child.input_transform(child._original_train_inputs), strict=False
            )
        else:
            child._clear_cache()


def setup_pair(mode):
    model, reference = model_pair(
        outputs=2 if mode in ("scalar", "custom") else 1,
        tree=mode == "plain",
        warp=mode == "warp",
    )
    kwargs, expected_kwargs, outputs = {}, {}, model.num_outputs
    if mode == "list":
        from botorch.models import ModelListGP as BtList

        from botorch_rs.models import ModelListGP

        # Repeated leaves must receive both child blocks' contributions.
        model, reference = ModelListGP(model, model), BtList(reference, reference)
        outputs = 2
    if mode in ("scalar", "list"):
        weights = torch.tensor([0.4, -0.2], dtype=torch.double)
        kwargs["posterior_transform"] = ScalarizedPosteriorTransform(weights)
        expected_kwargs["posterior_transform"] = BtScalarize(weights)
        outputs = 1
    if mode == "custom":

        def objective(samples, X=None):
            return samples[..., 0] + 0.15 * samples[..., 1] + 0.2 * X[..., 0]

        def constraint(samples):
            # Always feasible, with a nonconstant smooth candidate indicator.
            return -samples[..., 1].square() - 0.05

        kwargs.update(objective=GenericMCObjective(objective), constraints=[constraint], eta=0.2)
        expected_kwargs.update(objective=BtObjective(objective), constraints=[constraint], eta=0.2)
    child = getattr(model, "models", [model])[0]
    train = child.train_inputs[0]
    baseline = train.reshape(-1, *train.shape[-2:])[0, :3]
    return model, reference, baseline, outputs, kwargs, expected_kwargs


@pytest.mark.parametrize("name", ["nei", "lognei"])
@pytest.mark.parametrize("cache_root", [False, True])
@pytest.mark.parametrize("mode", ["plain", "scalar", "custom", "warp", "list"])
def test_noisy_parameter_query_and_incumbent_graph_parity(name, cache_root, mode):
    model, reference, baseline, outputs, kwargs, expected_kwargs = setup_pair(mode)
    cls, expected_cls = (NEI, BtNEI) if name == "nei" else (LogNEI, BtLogNEI)
    sampler = rs_sampler(S, [3 * outputs, 5 * outputs])
    shared = dict(X_baseline=baseline, prune_baseline=False, cache_root=cache_root)
    with seeded_draws():
        acquisition = cls(model, sampler=sampler, **shared, **kwargs)
        expected_acquisition = expected_cls(
            reference, sampler=BtSampler(torch.Size([S]), seed=0), **shared, **expected_kwargs
        )
        query = torch.linspace(0.15, 0.85, 12, dtype=torch.double).reshape(3, 2, 2).requires_grad_()
        expected_query = query.detach().clone().requires_grad_()
        actual = acquisition(query)
        with propagate_grads(True), fast_pred_var(False):
            refresh_live_training_solve(reference)
            expected = expected_acquisition(expected_query)
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
        upstream = torch.tensor([-0.3, 0.0, 0.7], dtype=torch.double)
        gradients = torch.autograd.grad((actual * upstream).sum(), (*model.parameters(), query))
        expected_gradients = torch.autograd.grad(
            (expected * upstream).sum(), (*reference.parameters(), expected_query)
        )
        for result, want in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(result, want, rtol=4e-6, atol=4e-7)
        dummy = torch.zeros(S, 3, 2, dtype=torch.double)
        torch.testing.assert_close(
            acquisition.compute_best_f(dummy),
            expected_acquisition.compute_best_f(dummy),
            rtol=2e-9,
            atol=2e-10,
        )
        assert not expected_acquisition.compute_best_f(dummy).requires_grad
        if cache_root:
            assert not expected_acquisition._baseline_L.requires_grad


@pytest.mark.parametrize("cache_root", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_lognei_pending_members_preserve_incremental_parameter_graph(cache_root, incremental):
    model, reference = model_pair()
    baseline = model.train_inputs[0][:3]
    pending = torch.tensor([[0.4, 0.6]], dtype=torch.double)
    shared = dict(
        X_baseline=baseline,
        X_pending=pending,
        prune_baseline=False,
        cache_root=cache_root,
        incremental=incremental,
    )
    sampler = rs_sampler(S, [4 if incremental else 3, 6])
    with seeded_draws():
        acquisition = LogNEI(model, sampler=sampler, **shared)
        expected_acquisition = BtLogNEI(
            reference, sampler=BtSampler(torch.Size([S]), seed=0), **shared
        )
        query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double, requires_grad=True)
        expected_query = query.detach().clone().requires_grad_()
        actual = acquisition(query)
        with propagate_grads(True), fast_pred_var(False):
            refresh_live_training_solve(reference)
            expected = expected_acquisition(expected_query)
        gradients = torch.autograd.grad(actual.sum(), (*model.parameters(), query))
        expected_gradients = torch.autograd.grad(
            expected.sum(), (*reference.parameters(), expected_query)
        )
        for result, want in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(result, want, rtol=4e-6, atol=4e-7)


@pytest.mark.parametrize("cache_root", [False, True])
def test_noisy_old_graph_ignores_later_sampler_pending_and_model_changes(cache_root):
    model, _ = model_pair()
    sampler = rs_sampler(S, [3, 5, 6])
    acquisition = LogNEI(
        model,
        model.train_inputs[0][:3],
        sampler=sampler,
        prune_baseline=False,
        cache_root=cache_root,
    )
    query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double)
    original = acquisition(query)
    expected = torch.autograd.grad(original.sum(), tuple(model.parameters()), retain_graph=True)
    sampler.base_samples = torch.full((S, 6), 0.25, dtype=torch.double)
    acquisition.set_X_pending(torch.tensor([[0.4, 0.6]], dtype=torch.double))
    with torch.no_grad():
        model.mean_module.raw_constant.add_(0.1)
    with pytest.raises(RuntimeError, match="Reconstruct the acquisition"):
        acquisition(query)
    actual = torch.autograd.grad(original.sum(), tuple(model.parameters()))
    for result, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, want)


def test_noisy_parameter_backward_does_not_repeat_objective():
    model, _ = model_pair()
    calls = []

    def objective(samples, X=None):
        calls.append(tuple(samples.shape))
        return samples[..., 0]

    acquisition = LogNEI(
        model,
        model.train_inputs[0][:3],
        prune_baseline=False,
        objective=GenericMCObjective(objective),
    )
    query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double)
    value = acquisition(query)
    count = len(calls)
    value.sum().backward()
    assert len(calls) == count


@pytest.mark.parametrize("name", ["qLogExpectedImprovement", "qLogNoisyExpectedImprovement"])
@pytest.mark.parametrize("batched", [False, True])
def test_learned_objective_tapes_preserve_member_dispatch_and_outcome_gradients(name, batched):
    from mo_acquisition_helpers import synced_draws
    from test_batched_learned_objective_mc import acquisition_pair
    from test_learned_objective_mc import acquisitions

    reference, acquisition = acquisition_pair(name, (2,)) if batched else acquisitions(name)
    query = torch.tensor(
        [[[0.25, 0.45], [0.65, 0.55]], [[0.35, 0.65], [0.8, 0.3]]],
        dtype=torch.double,
        requires_grad=True,
    )
    expected_query = query.detach().clone().requires_grad_()
    actual = acquisition(query)
    with synced_draws(85), propagate_grads(True), fast_pred_var(False):
        refresh_live_training_solve(reference.model)
        expected = reference(expected_query)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    weights = torch.tensor([-0.3, 0.7], dtype=torch.double)
    gradients = torch.autograd.grad(
        (actual * weights).sum(), (*acquisition.model.parameters(), query)
    )
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(), (*reference.model.parameters(), expected_query)
    )
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("cache_root", [False, True])
@pytest.mark.parametrize("minimize_first", [False, True])
def test_parego_parameter_tape_preserves_fixed_scalarization(cache_root, minimize_first):
    from botorch.acquisition.multi_objective.parego import qLogNParEGO as BtParEGO

    from botorch_rs.acquisition.multi_objective.parego import qLogNParEGO

    model, reference = model_pair(outputs=2)
    baseline = model.train_inputs[0][0, :3]
    shared = dict(
        X_baseline=baseline,
        prune_baseline=False,
        cache_root=cache_root,
        scalarization_weights=torch.tensor(
            [-0.35 if minimize_first else 0.35, 0.65], dtype=torch.double
        ),
    )
    with seeded_draws():
        acquisition = qLogNParEGO(model, sampler=rs_sampler(S, [6, 10]), **shared)
        expected_acquisition = BtParEGO(
            reference, sampler=BtSampler(torch.Size([S]), seed=0), **shared
        )
        query = torch.linspace(0.15, 0.85, 12, dtype=torch.double).reshape(3, 2, 2).requires_grad_()
        expected_query = query.detach().clone().requires_grad_()
        actual = acquisition(query)
        with propagate_grads(True), fast_pred_var(False):
            refresh_live_training_solve(reference)
            expected = expected_acquisition(expected_query)
        torch.testing.assert_close(actual, expected, rtol=4e-6, atol=4e-7)
        weights = torch.tensor([-0.3, 0.0, 0.7], dtype=torch.double)
        gradients = torch.autograd.grad((actual * weights).sum(), (*model.parameters(), query))
        expected_gradients = torch.autograd.grad(
            (expected * weights).sum(), (*reference.parameters(), expected_query)
        )
        for result, want in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(result, want, rtol=4e-6, atol=4e-7)
        assert not acquisition.Y_baseline.requires_grad
        assert not expected_acquisition.Y_baseline.requires_grad
        dummy = torch.zeros(S, 3, 2, dtype=torch.double)
        torch.testing.assert_close(
            acquisition.compute_best_f(dummy),
            expected_acquisition.compute_best_f(dummy),
            rtol=2e-9,
            atol=2e-10,
        )


@pytest.mark.parametrize("cache_root", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_parego_custom_objective_constraints_and_pending_graph(cache_root, incremental):
    from botorch.acquisition.multi_objective.objective import (
        GenericMCMultiOutputObjective as BtObjective,
    )
    from botorch.acquisition.multi_objective.parego import qLogNParEGO as BtParEGO

    from botorch_rs.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
    from botorch_rs.acquisition.multi_objective.parego import qLogNParEGO

    model, reference = model_pair(outputs=2)
    baseline = model.train_inputs[0][0, :3]
    pending = torch.tensor([[0.4, 0.6]], dtype=torch.double)

    def objective(samples, X=None):
        result = torch.stack([samples[..., 1], -samples[..., 0]], dim=-1)
        return result if X is None else result + 0.1 * X[..., :1]

    def constraint(samples):
        return -samples[..., 1].square() - 0.05

    shared = dict(
        X_baseline=baseline,
        X_pending=pending,
        prune_baseline=False,
        cache_root=cache_root,
        incremental=incremental,
        constraints=[constraint],
        eta=0.2,
        scalarization_weights=torch.tensor([-0.35, 0.65], dtype=torch.double),
    )
    with seeded_draws():
        acquisition = qLogNParEGO(
            model,
            sampler=rs_sampler(S, [8 if incremental else 6, 12]),
            objective=GenericMCMultiOutputObjective(objective),
            **shared,
        )
        expected_acquisition = BtParEGO(
            reference,
            sampler=BtSampler(torch.Size([S]), seed=0),
            objective=BtObjective(objective),
            **shared,
        )
        query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double, requires_grad=True)
        expected_query = query.detach().clone().requires_grad_()
        actual = acquisition(query)
        with propagate_grads(True), fast_pred_var(False):
            refresh_live_training_solve(reference)
            expected = expected_acquisition(expected_query)
        torch.testing.assert_close(actual, expected, rtol=4e-6, atol=4e-7)
        gradients = torch.autograd.grad(actual.sum(), (*model.parameters(), query))
        expected_gradients = torch.autograd.grad(
            expected.sum(), (*reference.parameters(), expected_query)
        )
        for result, want in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(result, want, rtol=4e-6, atol=4e-7)


@pytest.mark.parametrize("requested_cache", [False, True])
def test_nonlinear_outcome_joint_tape(requested_cache):
    from botorch.models import SingleTaskGP as BtGP
    from botorch.models.transforms.outcome import Log as BtLog

    from botorch_rs.models import SingleTaskGP
    from botorch_rs.models.transforms.outcome import Log

    source, _ = model_pair()
    x, y = source.train_inputs[0], torch.exp(source.train_targets).unsqueeze(-1)
    model, reference = (
        SingleTaskGP(x, y, outcome_transform=Log()),
        BtGP(x, y, outcome_transform=BtLog()),
    )
    shared = dict(X_baseline=x[:3], prune_baseline=False, cache_root=requested_cache)
    with seeded_draws():
        acquisition = LogNEI(model, sampler=rs_sampler(S, [3, 5]), **shared)
        expected_acquisition = BtLogNEI(
            reference, sampler=BtSampler(torch.Size([S]), seed=0), **shared
        )
        query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double)
        actual = acquisition(query)
        with propagate_grads(True), fast_pred_var(False):
            refresh_live_training_solve(reference)
            expected = expected_acquisition(query)
        assert not acquisition._rs.cache_root
        torch.testing.assert_close(actual, expected, rtol=4e-6, atol=4e-7)
        gradients = torch.autograd.grad(actual.sum(), tuple(model.parameters()))
        expected_gradients = torch.autograd.grad(expected.sum(), tuple(reference.parameters()))
        for result, want in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(result, want, rtol=4e-6, atol=4e-7)
