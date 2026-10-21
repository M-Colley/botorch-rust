# ruff: noqa: E402
"""Restart/model batches, real generation chunks and acquisition sequences."""

import pytest

pytest.importorskip("torch")


from unittest.mock import patch

import torch

from botorch_rs.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch_rs.optim import _torch_acq_initializers as initializer
from botorch_rs.optim import optimize_acqf

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class Quadratic(torch.nn.Module):
    _tensor_backend = "torch"

    def __init__(self, target=0.73):
        super().__init__()
        self.target = target
        self.X_pending = None
        self.seen = []

    def set_X_pending(self, X_pending=None):
        self.X_pending = X_pending

    def forward(self, X):
        self.seen.append(X.shape[0])
        return -(X - self.target).square().sum((-2, -1))


@pytest.mark.parametrize("device", DEVICES)
def test_batched_boltzmann_initializer_matches_reference(device):
    pytest.importorskip("scipy")
    pytest.importorskip("botorch")
    from botorch.optim.initializers import initialize_q_batch as reference

    X = torch.linspace(0, 1, 8 * 2 * 3 * 2 * 2, dtype=torch.double, device=device).reshape(
        8, 2, 3, 2, 2
    )
    scores = (X[..., 0, 0] * 7).sin() + X[..., 1, 1]
    torch.manual_seed(13)
    # Existing upstream PR3390 repairs boltzmann_sample's category-axis
    # standardization. The port already follows that corrected contract.
    from botorch.utils.transforms import standardize

    with patch(
        "botorch.utils.sampling.standardize",
        lambda values: standardize(values.unsqueeze(-1)).squeeze(-1),
    ):
        expected = reference(X, scores, 4, eta=1.7)
    actual = initializer.initialize_q_batch(
        X, scores, 4, eta=1.7, generator=torch.Generator(device=device).manual_seed(13)
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("generator", [gen_candidates_torch, gen_candidates_scipy])
def test_lowlevel_aggregation_and_highlevel_batch_limits_keep_fixed_values(device, generator):
    if generator is not gen_candidates_torch:
        pytest.importorskip("scipy")
    initial = torch.linspace(0.1, 0.9, 7 * 2 * 2, dtype=torch.double, device=device).reshape(
        7, 2, 2
    )
    fixed = initial[:, 0, 1].clone()
    bounds = initial.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    acq = Quadratic()
    options = {"max_optimization_problem_aggregation_size": 2}
    options.update(
        {
            "optimizer_options": {"lr": 0.05},
            "stopping_criterion_options": {"maxiter": 80, "n_window": 80},
        }
        if generator is gen_candidates_torch
        else {"maxiter": 80}
    )
    candidate, value = generator(
        initial, acq, lower_bounds=0.0, upper_bounds=1.0, fixed_features={1: fixed}, options=options
    )
    assert max(acq.seen) <= 2
    torch.testing.assert_close(candidate[..., 1], fixed[:, None].expand(7, 2))
    torch.testing.assert_close(
        candidate[..., 0], candidate.new_full((7, 2), 0.73), atol=0.015, rtol=0
    )
    torch.testing.assert_close(value, Quadratic()(candidate))
    acq.seen.clear()
    candidate, value = optimize_acqf(
        acq,
        bounds,
        2,
        7,
        batch_initial_conditions=initial,
        gen_candidates=generator,
        fixed_features={1: fixed},
        return_best_only=False,
        options={"batch_limit": 3, **options},
    )
    assert max(acq.seen) <= 2
    torch.testing.assert_close(candidate[..., 1], fixed[:, None].expand(7, 2))
    torch.testing.assert_close(value, Quadratic()(candidate))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("topn", [False, True])
def test_model_batched_initialization_and_best_restart_selection(device, topn):
    from botorch_rs.acquisition.analytic import PosteriorMean
    from botorch_rs.models import SingleTaskGP

    X = (
        torch.tensor([[0.05], [0.3], [0.65], [0.9]], dtype=torch.double, device=device)
        .expand(2, 4, 1)
        .clone()
    )
    Y = torch.stack(((X[0] * 5.3).sin(), (X[1] * 5.3).cos()))
    acq = PosteriorMean(SingleTaskGP(X, Y, backend="torch", outcome_transform=None))
    bounds = X.new_tensor([[0.0], [1.0]])
    recorded = []

    def retain(initial_conditions, acquisition_function, **kwargs):
        recorded.append(initial_conditions)
        return initial_conditions, acquisition_function(initial_conditions)

    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        result, value = optimize_acqf(
            acq,
            bounds,
            1,
            4,
            24,
            gen_candidates=retain,
            options={"seed": 13, "topn": topn, "batch_limit": 2, "init_batch_limit": 5},
        )
    candidates = torch.cat(recorded, 0)
    values = acq(candidates)
    expected_indices = values.argmax(0)
    expected = candidates[expected_indices, torch.arange(2, device=device)]
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(value, values.amax(0))
    assert result.shape == (2, 1, 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fail", [False, True])
def test_acquisition_sequence_pending_restoration_and_device_search(device, fail):
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    acquisitions = [Quadratic(t) for t in (0.2, 0.5, 0.8)]
    originals = [bounds.new_tensor([[0.91 + i * 0.01]]) for i in range(3)]
    for acq, pending in zip(acquisitions, originals, strict=True):
        acq.set_X_pending(pending)
    calls = []

    def retain(initial_conditions, acquisition_function, **kwargs):
        calls.append(acquisition_function.X_pending.detach().clone())
        if fail and len(calls) == 2:
            raise RuntimeError("deliberate generator failure")
        candidates = initial_conditions.new_full(
            initial_conditions.shape, acquisition_function.target
        )
        return candidates, acquisition_function(candidates)

    kwargs = dict(
        acq_function=None,
        bounds=bounds,
        q=3,
        num_restarts=2,
        raw_samples=8,
        sequential=True,
        acq_function_sequence=acquisitions,
        gen_candidates=retain,
    )
    if fail:
        with pytest.raises(RuntimeError, match="deliberate"):
            optimize_acqf(**kwargs)
    else:
        with (
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
            patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
            patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
        ):
            candidate, values = optimize_acqf(**kwargs)
        torch.testing.assert_close(candidate, bounds.new_tensor([[0.2], [0.5], [0.8]]))
        torch.testing.assert_close(values, bounds.new_zeros(3))
        assert [len(pending) for pending in calls] == [1, 2, 3]
    for acq, pending in zip(acquisitions, originals, strict=True):
        assert acq.X_pending is pending


@pytest.mark.parametrize("device", DEVICES)
def test_public_device_initializers_and_explicit_raw_generator(device):
    from botorch_rs.optim.initializers import gen_batch_initial_conditions, initialize_q_batch

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    queries = torch.linspace(0.1, 0.9, 8 * 2 * 2 * 2, dtype=bounds.dtype, device=device).reshape(
        8, 2, 2, 2
    )
    scores = -(queries - 0.73).square().sum((-2, -1))
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        selected, values = initialize_q_batch(queries, scores, 3)
        assert selected.shape == (3, 2, 2, 2) and values.shape == (3, 2)
        calls = []

        def raw(n, q, seed):
            calls.append((n, q, seed))
            return queries[:, 0]

        fantasies = bounds.new_tensor([[0.3, 0.8]])
        initial = gen_batch_initial_conditions(
            Quadratic(),
            bounds,
            2,
            3,
            8,
            generator=raw,
            fixed_X_fantasies=fantasies,
            options={"topn": True, "seed": 13},
        )
        assert initial.shape == (3, 3, 2)
        torch.testing.assert_close(initial[:, -1], fantasies.expand(3, 2))
        assert calls == [(8, 2, 13)]
