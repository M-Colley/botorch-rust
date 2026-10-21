"""Device model-fit metrics use the actual likelihood rather than CPU state."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models import SingleTaskGP  # noqa: E402
from botorch_rs.models.transforms.input import Normalize  # noqa: E402
from botorch_rs.utils.evaluation import compute_in_sample_model_fit_metric as metric  # noqa: E402

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("transforms", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_metrics_match_reference_with_outputs_batches_transforms_and_noise(
    device, m, fixed, transforms, batched
):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.models.transforms.input import Normalize as ReferenceNormalize
    from botorch.utils.evaluation import compute_in_sample_model_fit_metric as reference_metric

    x = torch.linspace(0.1, 0.9, 6, dtype=torch.double, device=device).unsqueeze(-1)
    y = torch.cat([(x * (3 + i)).sin() for i in range(m)], dim=-1)
    if batched:
        x = torch.stack((x, x + 0.04))
        y = torch.stack((y, y + 0.2))
    noise = torch.full_like(y, 0.035) if fixed else None
    a_kwargs, e_kwargs = {}, {}
    if transforms:
        a_kwargs["input_transform"] = Normalize(d=1, bounds=x.new_tensor([[0.0], [1.0]]))
        e_kwargs["input_transform"] = ReferenceNormalize(d=1, bounds=x.new_tensor([[0.0], [1.0]]))
    else:
        a_kwargs["outcome_transform"] = e_kwargs["outcome_transform"] = None
    actual = SingleTaskGP(x, y, noise, backend="torch", **a_kwargs)
    expected = ReferenceGP(x, y, noise, **e_kwargs)
    for criterion in ("MLL", "AIC", "BIC"):
        a, e = metric(actual, criterion), reference_metric(expected, criterion)
        assert isinstance(a, float)
        assert a == pytest.approx(e, rel=2e-9, abs=2e-9)
        assert actual.training is False


@pytest.mark.parametrize("device", DEVICES)
def test_metric_has_no_training_graph_or_host_array_transfer_and_excludes_priors(device):
    x = (
        torch.linspace(0.05, 0.95, 5, dtype=torch.double, device=device)
        .unsqueeze(-1)
        .requires_grad_()
    )
    y = (x.detach() * 4).sin().requires_grad_()
    model = SingleTaskGP(x, y, backend="torch", outcome_transform=None)
    model.train()
    with torch.no_grad():
        expected = (
            model.likelihood(model(*model.train_inputs)).log_prob(model.train_targets).mean().item()
        )
    error = AssertionError("Metric must retain numerical arrays on their device")
    with (
        patch.object(torch.Tensor, "cpu", side_effect=error),
        patch.object(torch.Tensor, "numpy", side_effect=error),
        patch.object(torch.Tensor, "tolist", side_effect=error),
    ):
        actual = metric(model, "MLL")
    assert actual == pytest.approx(expected, abs=1e-12)
    assert x.grad is None and y.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training
    with pytest.raises(ValueError, match="Invalid evaluation criterion"):
        metric(model, "bad")


@pytest.mark.parametrize("device", DEVICES)
def test_loaded_fully_bayesian_metrics_preserve_samples_and_average_components(device):
    pytest.importorskip("botorch")
    pytest.importorskip("jax")
    pytest.importorskip("numpyro")
    from botorch.utils.evaluation import compute_in_sample_model_fit_metric as reference_metric
    from test_torch_fully_bayesian import _pair

    (actual, expected), _, _ = _pair(device, "FullyBayesianSingleTaskGP", transforms=True)
    kernel = actual.covar_module
    for criterion in ("MLL", "AIC", "BIC"):
        assert metric(actual, criterion) == pytest.approx(
            reference_metric(expected, criterion), rel=2e-8, abs=2e-8
        )
        assert actual.covar_module is kernel and actual.num_mcmc_samples == 3


@pytest.mark.parametrize("device", DEVICES)
def test_multitask_metric_uses_correlated_task_law_and_task_noise(device):
    pytest.importorskip("botorch")
    from botorch.utils.evaluation import compute_in_sample_model_fit_metric as reference_metric
    from test_torch_backend_multitask import data, paired_models

    x, y, _ = data(device)
    actual, expected = paired_models(x, y)
    # The pinned utility drops the task inputs required by its own default
    # Hadamard likelihood. Retain the raw failure and form the correct law
    # explicitly rather than changing the reference checkout.
    with pytest.raises(ValueError, match="Task indices must be provided"):
        reference_metric(expected, "MLL")
    expected.train()
    with torch.no_grad():
        distribution = expected.likelihood(expected(*expected.train_inputs), *expected.train_inputs)
        log_probability = distribution.log_prob(expected.train_targets).mean().item()
    expected.eval()
    parameters = sum(parameter.numel() for parameter in expected.parameters())
    correct = {
        "MLL": log_probability,
        "AIC": 2 * parameters - 2 * log_probability,
        "BIC": parameters * math.log(x.shape[-2]) - 2 * log_probability,
    }
    for criterion in ("MLL", "AIC", "BIC"):
        assert metric(actual, criterion) == pytest.approx(correct[criterion], rel=2e-8, abs=2e-8)
