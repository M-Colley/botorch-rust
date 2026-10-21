"""NUTS reports completed transitions without changing trajectories or fitted state."""

# ruff: noqa: E402
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.fit import fit_fully_bayesian_model_nuts
from botorch_rs.models import _nuts_progress
from botorch_rs.models._torch_backend_mcmc import sample_torch_nuts
from botorch_rs.models.fully_bayesian import FullyBayesianSingleTaskGP

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]
BACKENDS = [
    ("native", "cpu"),
    ("torch", "cpu"),
    pytest.param(
        "torch",
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
    ),
]


class GaussianPotential:
    def value_and_grad(self, z):
        return z.square().sum() / 2, z.clone()


def _model(backend, device):
    x = torch.tensor([[0.12], [0.34], [0.65], [0.88]], device=device, dtype=torch.double)
    y = (x * 4).sin()
    return FullyBayesianSingleTaskGP(
        x, y, torch.full_like(y, 0.03), backend=backend, outcome_transform=None
    )


def test_native_callbacks_report_in_order_preserve_draws_and_clean_errors():
    potential = _model("native", "cpu").pyro_model._potential()
    options = dict(
        num_warmup=2,
        num_samples=3,
        max_tree_depth=(2, 2),
        seed=29,
        z0=[0.0] * potential.dim,
    )
    expected = potential.sample_nuts(**options)
    calls = []

    def progress(completed, total, warmup, accept, steps, diverging):
        assert completed == len(calls) + 1
        assert total == 5 and warmup == (completed <= 2)
        assert math.isfinite(accept) and steps > 0 and isinstance(diverging, bool)
        # Reentrant native calls are valid while the original run is in flight.
        assert math.isfinite(potential.potential(options["z0"])[0])
        calls.append(completed)

    assert potential.sample_nuts(**options, progress=progress) == expected
    assert calls == [1, 2, 3, 4, 5]
    failure = KeyboardInterrupt("cancel this run")

    def cancel(*args):
        if args[0] == 2:
            raise failure

    with pytest.raises(KeyboardInterrupt, match="cancel this run") as caught:
        potential.sample_nuts(**options, progress=cancel)
    assert caught.value is failure
    assert potential.sample_nuts(**options) == expected
    with pytest.raises(ValueError, match="progress must be callable"):
        potential.sample_nuts(**options, progress=42)


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_callbacks_report_in_order_and_preserve_seeded_samples(device):
    initial = torch.tensor([0.2, -0.3], device=device, dtype=torch.double)
    options = dict(num_warmup=2, num_samples=3, max_tree_depth=2, seed=29)
    expected = sample_torch_nuts(GaussianPotential(), initial, **options)
    calls = []

    def progress(completed, total, warmup, accept, steps, diverging):
        assert completed == len(calls) + 1
        assert total == 5 and warmup == (completed <= 2)
        assert math.isfinite(accept) and steps > 0 and isinstance(diverging, bool)
        calls.append(completed)

    actual = sample_torch_nuts(GaussianPotential(), initial, **options, progress=progress)
    torch.testing.assert_close(actual["samples"], expected["samples"], atol=0, rtol=0)
    for actual_row, expected_row in zip(actual["trace"], expected["trace"], strict=True):
        for key in actual_row:
            torch.testing.assert_close(actual_row[key], expected_row[key], atol=0, rtol=0)
    assert calls == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="progress must be callable"):
        sample_torch_nuts(GaussianPotential(), initial, **options, progress=42)


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_public_progress_runs_before_fit_returns_and_can_be_disabled(
    backend, device, monkeypatch, capsys
):
    model = _model(backend, device)
    calls = []
    original = _nuts_progress._NutsProgress.__call__

    def report(self, *args):
        assert "_nuts_result" not in model.__dict__
        calls.append(args[0])
        original(self, *args)

    monkeypatch.setattr(_nuts_progress._NutsProgress, "__call__", report)
    options = dict(warmup_steps=2, num_samples=3, thinning=1, max_tree_depth=2, seed=29)
    fit_fully_bayesian_model_nuts(model, **options, disable_progbar=False)
    assert calls == [1, 2, 3, 4, 5]
    text = capsys.readouterr().err
    assert "Warmup: 1/5" in text and "Sample: 5/5" in text and text.endswith("\n")
    other = _model(backend, device)
    fit_fully_bayesian_model_nuts(other, **options, disable_progbar=True)
    assert capsys.readouterr().err == ""
    torch.testing.assert_close(
        torch.as_tensor(model._nuts_result["samples"]),
        torch.as_tensor(other._nuts_result["samples"]),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("backend,device", BACKENDS)
def test_cancelled_fit_closes_progress_and_preserves_loaded_model(
    backend, device, monkeypatch, capsys
):
    model = _model(backend, device)
    like = model.train_inputs[0]
    model.load_mcmc_samples(
        {
            "mean": like.new_tensor([0.2, 0.4]),
            "lengthscale": like.new_tensor([[0.5], [0.8]]),
        }
    )
    model.eval()
    query = like.new_tensor([[0.25]])
    expected = model.posterior(query).mean.clone()
    original = _nuts_progress._NutsProgress.__call__

    def cancel(self, *args):
        original(self, *args)
        raise KeyboardInterrupt("stop during warmup")

    with monkeypatch.context() as patch:
        patch.setattr(_nuts_progress._NutsProgress, "__call__", cancel)
        with pytest.raises(KeyboardInterrupt, match="stop during warmup"):
            fit_fully_bayesian_model_nuts(
                model, warmup_steps=2, num_samples=2, max_tree_depth=2, thinning=1
            )
    assert capsys.readouterr().err.endswith("\n")
    torch.testing.assert_close(model.posterior(query).mean, expected, atol=0, rtol=0)
    assert not model.training and "_nuts_result" not in model.__dict__
