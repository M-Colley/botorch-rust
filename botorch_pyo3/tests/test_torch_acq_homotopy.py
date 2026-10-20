# ruff: noqa: E402
"""Homotopy stages, pruning and state rollback on the acquisition device."""

import pytest

pytest.importorskip("torch")


import torch
from test_torch_acq_discrete import DEVICES, no_host
from test_torch_acq_optimization_configs import Quadratic

from botorch_rs.exceptions.errors import CandidateGenerationError
from botorch_rs.generation.gen import gen_candidates_torch
from botorch_rs.optim import homotopy as schedule
from botorch_rs.optim.optimize_homotopy import optimize_acqf_homotopy, prune_candidates


class Peak(Quadratic):
    def __init__(self, device, dimension=1, fail=False):
        super().__init__()
        self.s = torch.nn.Parameter(torch.tensor(1.7, dtype=torch.double, device=device))
        self.target = torch.tensor([0.73] * dimension, dtype=torch.double, device=device)
        self.fail = fail

    def forward(self, X):
        if self.fail and self.X_pending is not None and self.X_pending.shape[-2] > 1:
            raise RuntimeError("deliberate homotopy failure")
        return -((X - self.target) / self.s).square().sum((-2, -1))


def make_schedule(acquisition, library=schedule, callbacks=None):
    return library.Homotopy(
        [library.HomotopyParameter(acquisition.s, library.FixedHomotopySchedule([1.0, 0.5, 0.2]))],
        callbacks=callbacks,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tolerance", [0.0, 1e-4, 0.1])
def test_device_pruning_matches_reference_and_retains_selected_graph(device, tolerance):
    from botorch.optim.optimize_homotopy import prune_candidates as reference

    points = torch.tensor(
        [[0.1, 0.2], [0.10001, 0.2], [0.4, 0.7], [0.9, 0.2]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    values = points.new_tensor([0.1, 0.4, 0.2, 0.3])
    expected = reference(points, values, tolerance)
    expected_grad = torch.autograd.grad(expected.sum(), points)[0]
    with no_host():
        actual = prune_candidates(points, values, tolerance)
        actual_grad = torch.autograd.grad(actual.sum(), points)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [1, 2])
def test_real_device_homotopy_matches_reference_stages_and_resets(device, q):
    from botorch.generation.gen import gen_candidates_torch as reference_generator
    from botorch.optim import homotopy as reference_schedule
    from botorch.optim.optimize_homotopy import optimize_acqf_homotopy as reference

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    initial = bounds.new_tensor([[[0.1]], [[0.5]], [[0.9]]])
    actual, expected = Peak(device), Peak(device)
    originals = []
    for acquisition in (actual, expected):
        pending = bounds.new_tensor([[0.95]])
        acquisition.set_X_pending(pending)
        originals.append(pending)
    histories = [[], []]
    actual_h = make_schedule(
        actual, callbacks=[lambda: histories[0].append(float(actual.s.detach()))]
    )
    reference_h = make_schedule(
        expected,
        reference_schedule,
        callbacks=[lambda: histories[1].append(float(expected.s.detach()))],
    )
    options = {
        "optimizer_options": {"lr": 0.05},
        "stopping_criterion_options": {"maxiter": 80, "n_window": 80},
    }
    kwargs = dict(
        bounds=bounds,
        q=q,
        num_restarts=3,
        batch_initial_conditions=initial,
        options=options,
        final_options=options,
    )

    def detached_reference(initial_conditions, *args, **kwargs):
        # The source generator fails when a later homotopy stage receives its
        # graph-attached previous output. Each stage optimizes new decisions.
        return reference_generator(initial_conditions.detach(), *args, **kwargs)

    expected_result = reference(
        expected, homotopy=reference_h, gen_candidates=detached_reference, **kwargs
    )
    with no_host():
        actual_result = optimize_acqf_homotopy(
            actual, homotopy=actual_h, gen_candidates=gen_candidates_torch, **kwargs
        )
    torch.testing.assert_close(actual_result[0], expected_result[0], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(actual_result[1], expected_result[1], atol=1e-8, rtol=1e-8)
    assert histories[0] == histories[1]
    torch.testing.assert_close(actual.s, actual.s.new_tensor(1.7))
    if q > 1:
        assert actual.X_pending is originals[0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("alternating", [False, True])
def test_device_homotopy_uses_mixed_and_alternating_domains(device, alternating):
    acquisition = Peak(device, dimension=2)
    acquisition.target = acquisition.target.new_tensor([1.0, 0.73])
    bounds = acquisition.target.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    if alternating:
        discrete = {0: torch.linspace(0, 1, 12, dtype=bounds.dtype, device=device)}
        options = {
            "num_spray_points": 0,
            "initialization_strategy": "random",
            "maxiter_alternating": 3,
            "maxiter_continuous": 8,
        }
    else:
        discrete = {0: [0.0, 1.0]}
        options = {
            "seed": 13,
            "optimizer_options": {"lr": 0.05},
            "stopping_criterion_options": {"maxiter": 80, "n_window": 80},
        }
    homotopy = make_schedule(acquisition)
    with no_host():
        candidate, value = optimize_acqf_homotopy(
            acquisition,
            bounds,
            1,
            3,
            homotopy,
            raw_samples=8,
            discrete_dims=discrete,
            options=options,
            final_options=options,
            gen_candidates=gen_candidates_torch,
        )
    torch.testing.assert_close(candidate, bounds.new_tensor([[1.0, 0.73]]), atol=0.01, rtol=0)
    assert value.shape == (1,) and torch.isfinite(value).all()
    torch.testing.assert_close(acquisition.s, acquisition.s.new_tensor(1.7))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("failure", ["bounds", "fixed", "constraint", "domain"])
def test_homotopy_rejects_infeasible_postprocessing_and_restores_schedule(device, failure):
    acquisition = Peak(device, dimension=2)
    bounds = acquisition.target.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    options = {"stopping_criterion_options": {"maxiter": 2}}
    kwargs = {}
    if failure == "fixed":
        kwargs["fixed_features"] = {0: 0.5}
    elif failure == "constraint":
        kwargs["inequality_constraints"] = [
            (torch.tensor([0], device=device), bounds.new_tensor([1.0]), 0.2)
        ]
        options = {"maxiter": 3}
    elif failure == "domain":
        kwargs["discrete_dims"] = {0: [0.0, 1.0]}

    def transform(X):
        return X + 2 if failure == "bounds" else X * 0 + (0.3 if failure == "domain" else 0)

    with pytest.raises(CandidateGenerationError):
        optimize_acqf_homotopy(
            acquisition,
            bounds,
            1,
            2,
            make_schedule(acquisition),
            raw_samples=8,
            options=options,
            final_options=options,
            gen_candidates=None if failure == "constraint" else gen_candidates_torch,
            post_processing_func=transform,
            **kwargs,
        )
    torch.testing.assert_close(acquisition.s, acquisition.s.new_tensor(1.7))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cleanup_error", [False, True])
def test_homotopy_failure_rolls_back_pending_and_schedule_preserving_primary_error(
    device, cleanup_error
):
    acquisition = Peak(device, fail=True)
    bounds = acquisition.target.new_tensor([[0.0], [1.0]])
    pending = bounds.new_tensor([[0.95]])
    acquisition.set_X_pending(pending)
    homotopy = make_schedule(acquisition)
    if cleanup_error:

        def callback():
            if float(acquisition.s.detach()) == 1.7:
                raise RuntimeError("secondary cleanup failure")

        homotopy._callbacks = [callback]
    options = {
        "optimizer_options": {"lr": 0.05},
        "stopping_criterion_options": {"maxiter": 30, "n_window": 30},
    }
    with pytest.raises(RuntimeError, match="deliberate homotopy failure"), no_host():
        optimize_acqf_homotopy(
            acquisition,
            bounds,
            2,
            2,
            homotopy,
            raw_samples=8,
            gen_candidates=gen_candidates_torch,
            options=options,
            final_options=options,
        )
    assert acquisition.X_pending is pending
    torch.testing.assert_close(acquisition.s, acquisition.s.new_tensor(1.7))
