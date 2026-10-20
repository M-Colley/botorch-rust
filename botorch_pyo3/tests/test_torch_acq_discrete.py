# ruff: noqa: E402
"""Device discrete/mixed/cyclic search with independent reference checks."""

import pytest

pytest.importorskip("torch")


from unittest.mock import patch

import torch
from test_torch_acq_optimization_configs import Quadratic

from botorch_rs.generation.gen import gen_candidates_torch
from botorch_rs.optim.optimize import (
    optimize_acqf_cyclic,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
    optimize_acqf_list,
    optimize_acqf_mixed,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def no_host():
    from contextlib import ExitStack

    stack = ExitStack()
    for name in ("cpu", "numpy", "tolist"):
        stack.enter_context(
            patch.object(torch.Tensor, name, side_effect=AssertionError("host transfer"))
        )
    return stack


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [1, 3])
@pytest.mark.parametrize("unique", [False, True])
def test_discrete_matches_reference_and_chunk_limit(device, q, unique):
    from botorch.optim.optimize import optimize_acqf_discrete as reference

    choices = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.double, device=device)
    avoid = choices.new_tensor([[0.7]])
    constraints = [(torch.tensor([0], device=device), choices.new_tensor([1.0]), 0.2)]
    kwargs = dict(
        q=q,
        choices=choices,
        max_batch_size=2,
        unique=unique,
        X_avoid=avoid,
        inequality_constraints=constraints,
    )
    actual_acq, reference_acq = Quadratic(), Quadratic()
    pending = choices.new_tensor([[0.96]])
    actual_acq.set_X_pending(pending)
    reference_acq.set_X_pending(pending)
    expected = reference(reference_acq, **kwargs)
    with no_host():
        actual = optimize_acqf_discrete(actual_acq, **kwargs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert max(actual_acq.seen) <= 2
    assert actual_acq.X_pending is pending


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("return_values", [False, True])
def test_discrete_local_search_reference_initials_and_cleanup(device, return_values):
    from botorch.optim.optimize import optimize_acqf_discrete_local_search as reference

    grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.double, device=device)
    starts = grid.new_tensor([[[0.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]]])
    kwargs = dict(
        discrete_choices=[grid, grid],
        q=2,
        num_restarts=4,
        raw_samples=32,
        batch_initial_conditions=starts,
        max_batch_size=3,
        X_avoid=grid.new_tensor([[0.75, 0.75]]),
        return_acq_values=return_values,
    )
    acq, ref = Quadratic(), Quadratic()
    pending = grid.new_tensor([[0.9, 0.9]])
    acq.set_X_pending(pending)
    ref.set_X_pending(pending)
    torch.manual_seed(13)
    expected = reference(ref, **kwargs)
    torch.manual_seed(13)
    with no_host():
        actual = optimize_acqf_discrete_local_search(acq, **kwargs)
    torch.testing.assert_close(actual[0], expected[0])
    if return_values:
        torch.testing.assert_close(actual[1], expected[1])
    else:
        assert actual[1] is None
    assert acq.X_pending is pending


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("return_values", [False, True])
def test_mixed_real_device_continuous_search_and_domain_selection(device, q, return_values):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    acq = Quadratic()
    pending = bounds.new_tensor([[0.96, 0.96]])
    acq.set_X_pending(pending)
    with no_host():
        candidate, value = optimize_acqf_mixed(
            acq,
            bounds,
            q,
            3,
            [{0: 0.0}, {0: 1.0}],
            raw_samples=16,
            gen_candidates=gen_candidates_torch,
            return_acq_values=return_values,
            options={
                "seed": 13,
                "batch_limit": 2,
                "optimizer_options": {"lr": 0.05},
                "stopping_criterion_options": {"maxiter": 90, "n_window": 90},
            },
        )
    torch.testing.assert_close(candidate[:, 0], candidate.new_ones(q))
    torch.testing.assert_close(candidate[:, 1], candidate.new_full((q,), 0.73), atol=0.015, rtol=0)
    if return_values:
        torch.testing.assert_close(value, acq(candidate))
    else:
        assert value is None
    assert acq.X_pending is pending


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_all_restarts_of_winning_combination_and_reference(device):
    from botorch.optim.optimize import optimize_acqf_mixed as reference

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    initial = bounds.new_tensor([[[0.4, 0.2]], [[0.5, 0.6]], [[0.6, 0.8]]])
    kwargs = dict(
        bounds=bounds,
        q=1,
        num_restarts=3,
        fixed_features_list=[{0: 0.0}, {0: 1.0}],
        batch_initial_conditions=initial,
        return_best_only=False,
        options={"maxiter": 40},
    )
    expected = reference(Quadratic(), **kwargs)
    actual = optimize_acqf_mixed(Quadratic(), **kwargs)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_acquisition_list_device_search_and_restore(device):
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    acquisitions = [Quadratic(t) for t in (0.2, 0.5, 0.8)]
    original = [bounds.new_tensor([[0.9 + i * 0.01]]) for i in range(3)]
    for acq, pending in zip(acquisitions, original, strict=True):
        acq.set_X_pending(pending)
    with no_host():
        candidates, values = optimize_acqf_list(
            acquisitions,
            bounds,
            2,
            raw_samples=8,
            options={
                "seed": 13,
                "optimizer_options": {"lr": 0.05},
                "stopping_criterion_options": {"maxiter": 90, "n_window": 90},
            },
            ic_gen_kwargs={"gen_candidates": gen_candidates_torch},
        )
    torch.testing.assert_close(
        candidates, bounds.new_tensor([[0.2], [0.5], [0.8]]), atol=0.015, rtol=0
    )
    assert values.shape == (3,)
    for acq, pending in zip(acquisitions, original, strict=True):
        assert acq.X_pending is pending


@pytest.mark.parametrize("device", DEVICES)
def test_cyclic_device_updates_and_pending_restoration(device):
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    acq = Quadratic()
    pending = bounds.new_tensor([[0.96]])
    acq.set_X_pending(pending)
    with no_host():
        candidate, value = optimize_acqf_cyclic(
            acq,
            bounds,
            2,
            3,
            raw_samples=16,
            gen_candidates=gen_candidates_torch,
            options={
                "seed": 13,
                "optimizer_options": {"lr": 0.04},
                "stopping_criterion_options": {"maxiter": 100, "n_window": 100},
            },
            cyclic_options={"maxiter": 2},
        )
    torch.testing.assert_close(candidate, candidate.new_full((2, 1), 0.73), atol=0.015, rtol=0)
    assert value.shape == (2,)
    assert acq.X_pending is pending


class PendingQuadratic(Quadratic):
    def __init__(self, tensor_backend, fail=False):
        super().__init__(0.2)
        self._tensor_backend = "torch" if tensor_backend else None
        self.fail = fail
        self.pending_seen = []

    def forward(self, X):
        self.pending_seen.append(self.X_pending)
        if self.fail:
            raise RuntimeError("deliberate acquisition failure")
        target = self.target
        if self.X_pending is not None:
            target += 0.1 * self.X_pending.mean(-2)
        return -(X - target).square().sum((-2, -1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tensor_backend", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_acquisition_list_own_baselines_and_exception_cleanup(device, tensor_backend, fail):
    if not tensor_backend and device == "cuda":
        pytest.skip("Native callback bridge is a CPU interface.")
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    acquisitions = [PendingQuadratic(tensor_backend, fail and i == 2) for i in range(3)]
    originals = [bounds.new_tensor([[0.7 + i * 0.1]]) for i in range(3)]
    for acquisition, pending in zip(acquisitions, originals, strict=True):
        acquisition.set_X_pending(pending)
    kwargs = dict(
        acq_function_list=acquisitions,
        bounds=bounds,
        num_restarts=1,
        raw_samples=2,
        fixed_features={0: 0.3},
    )
    if fail:
        with pytest.raises(Exception, match="deliberate acquisition failure"):
            optimize_acqf_list(**kwargs)
    else:
        candidates, values = optimize_acqf_list(**kwargs)
        torch.testing.assert_close(candidates, bounds.new_full((3, 1), 0.3))
        expected = []
        for i, original in enumerate(originals):
            pending = torch.cat([original, candidates[:i]], -2)
            expected.append(-((0.3 - (0.2 + 0.1 * pending.mean())) ** 2))
        torch.testing.assert_close(values, torch.stack(expected))
    for i, acquisition in enumerate(acquisitions):
        assert acquisition.X_pending is originals[i]
        assert acquisition.pending_seen
        expected_pending = torch.cat([originals[i], bounds.new_full((i, 1), 0.3)], -2)
        torch.testing.assert_close(acquisition.pending_seen[0], expected_pending)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("minimize", [False, True])
def test_tensor_stopping_matches_reference_and_retains_device(device, minimize):
    from botorch.optim.stopping import ExpMAStoppingCriterion as Reference

    from botorch_rs.optim.stopping import ExpMAStoppingCriterion

    criterion = ExpMAStoppingCriterion(n_window=3, minimize=minimize)
    reference = Reference(n_window=3, minimize=minimize)
    for i in range(15):
        values = torch.tensor([1.0 + 1 / (i + 1), 2.0 + 1 / (i + 1)], device=device)
        expected = reference(values.clone())
        with no_host():
            assert criterion(values) == expected
    criterion.reset()
    assert criterion.iter == 0
    assert not criterion(values)


@pytest.mark.parametrize("device", DEVICES)
def test_discrete_batched_exact_gp_values_and_device_choices(device):
    from botorch_rs.acquisition.analytic import PosteriorMean
    from botorch_rs.models import SingleTaskGP

    X = torch.linspace(0, 1, 7, dtype=torch.double, device=device).reshape(1, 7, 1).expand(2, 7, 1)
    Y = torch.stack(((X[0] * 5.3).sin(), (X[1] * 5.3).cos()))
    acquisition = PosteriorMean(SingleTaskGP(X, Y, backend="torch", outcome_transform=None))
    choices = torch.linspace(0, 1, 19, dtype=torch.double, device=device).unsqueeze(-1)
    scores = acquisition(choices[:, None, None, :].expand(19, 2, 1, 1))
    with no_host():
        candidate, value = optimize_acqf_discrete(acquisition, 1, choices, max_batch_size=3)
    expected = choices[scores.argmax(0)].unsqueeze(-2)
    torch.testing.assert_close(candidate, expected)
    torch.testing.assert_close(value, scores.amax(0))


class FailAfterPending(Quadratic):
    def forward(self, X):
        if self.X_pending.shape[-2] > 1:
            raise RuntimeError("pending-dependent failure")
        return super().forward(X)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["discrete", "local", "mixed", "cyclic"])
def test_search_exception_restores_pending(device, method):
    grid = torch.linspace(0, 1, 5, dtype=torch.double, device=device)
    acquisition = FailAfterPending()
    pending = grid.new_tensor([[0.91]])
    acquisition.set_X_pending(pending)
    with pytest.raises(RuntimeError, match="pending-dependent failure"), no_host():
        if method == "discrete":
            optimize_acqf_discrete(acquisition, 2, grid.unsqueeze(-1))
        elif method == "local":
            optimize_acqf_discrete_local_search(
                acquisition,
                [grid],
                2,
                2,
                8,
                batch_initial_conditions=grid.new_tensor([[[0.0]], [[1.0]]]),
            )
        else:
            kwargs = dict(
                acq_function=acquisition,
                bounds=grid.new_tensor([[0.0], [1.0]]),
                q=2,
                num_restarts=2,
                raw_samples=8,
                gen_candidates=gen_candidates_torch,
                options={"stopping_criterion_options": {"maxiter": 2}},
            )
            if method == "mixed":
                optimize_acqf_mixed(**kwargs, fixed_features_list=[{0: 0.3}, {0: 0.7}])
            else:
                optimize_acqf_cyclic(**kwargs)
    assert acquisition.X_pending is pending
