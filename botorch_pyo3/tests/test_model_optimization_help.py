"""Diagnostic JSON and atomic transformed-target/fixed-noise restoration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.models.utils.assorted import (
    extract_targets_and_noise_single_output,
    get_data_for_optimization_help,
    restore_targets_and_noise_single_output,
)
from helpers import KINDS, as_kind, assert_close, importorskip_botorch, to_list


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fixed", [False, True])
def test_native_diagnostics_capture_parameters_and_original_transformed_data(tmp_path, kind, fixed):
    x = as_kind([[2.0], [5.0], [8.0]], kind)
    y = as_kind([[10.0], [40.0], [20.0]], kind)
    yvar = as_kind([[0.1], [0.2], [0.3]], kind) if fixed else None
    model = SingleTaskGP(
        x, y, train_Yvar=yvar, input_transform=Normalize(d=1), outcome_transform=Standardize(m=1)
    )
    path = tmp_path / "diagnostics.json"
    get_data_for_optimization_help(model, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dtype"] == "float64"
    assert payload["train_X"] == to_list(x)
    assert_close(payload["train_Y"], [[v] for v in to_list(model.train_targets)])
    for name, value in model.named_parameters():
        assert_close(payload["state_dict"][name], value)
    info = payload["native_info"]
    assert info["class"] == "SingleTaskGP"
    assert_close(info["train_X_transformed"], [[0.0], [0.5], [1.0]])
    assert_close(info["train_Y_original"], y)
    if fixed:
        assert_close(info["train_Yvar_original"], yvar)
    else:
        assert info["train_Yvar_original"] is None


def test_gpytorch_diagnostic_json_matches_official(tmp_path):
    importorskip_botorch()
    import torch
    from botorch.models import SingleTaskGP as OfficialGP
    from botorch.models.utils.assorted import get_data_for_optimization_help as official

    model = OfficialGP(
        torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double),
        torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.double),
    )
    a, b = tmp_path / "ours.json", tmp_path / "official.json"
    get_data_for_optimization_help(model, str(a))
    official(model, str(b))
    assert json.loads(a.read_text()) == json.loads(b.read_text())


def test_unserializable_state_preserves_existing_destination(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("preserve me")
    model = SimpleNamespace(
        train_inputs=([[0.0]],), train_targets=[1.0], state_dict=lambda: {"invalid": object()}
    )
    with pytest.raises(TypeError):
        get_data_for_optimization_help(model, str(path))
    assert path.read_text() == "preserve me"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fixed", [False, True])
def test_restore_native_data_invalidates_cache_preserves_old_posterior_and_hyperparameters(
    kind, fixed
):
    x = as_kind([[0.1], [0.4], [0.8]], kind)
    y = as_kind([[1.0], [2.0], [4.0]], kind)
    noise = as_kind([[0.01], [0.02], [0.03]], kind) if fixed else None
    model = SingleTaskGP(x, y, train_Yvar=noise, outcome_transform=Standardize(m=1))
    query = as_kind([[0.3], [0.6]], kind)
    posterior = model.posterior(query)
    original_mean = to_list(posterior.mean)
    raw = list(model._rs.raw_parameters())
    targets, variance = extract_targets_and_noise_single_output(model)
    from botorch_rs import _ops

    new_targets = _ops.apply("add", targets, 0.5)
    new_variance = _ops.apply("mul", variance, 2.0) if fixed else None
    restore_targets_and_noise_single_output(model, new_targets, new_variance, True)
    actual_targets, actual_variance = extract_targets_and_noise_single_output(model)
    assert_close(actual_targets, new_targets)
    if fixed:
        assert_close(actual_variance, new_variance)
    assert list(model._rs.raw_parameters()) == raw
    assert to_list(model.posterior(query).mean) != original_mean
    assert_close(posterior.mean, original_mean)
    restore_targets_and_noise_single_output(model, targets, variance, True)
    assert_close(model.posterior(query).mean, original_mean)


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_native_target_and_noise_replacement_is_atomic(kind):
    model = SingleTaskGP(
        as_kind([[0.1], [0.4], [0.8]], kind),
        as_kind([[1.0], [2.0], [4.0]], kind),
        train_Yvar=as_kind([[0.01], [0.02], [0.03]], kind),
        outcome_transform=None,
    )
    before_y, before_noise = extract_targets_and_noise_single_output(model)
    with pytest.raises((ValueError, RuntimeError)):
        restore_targets_and_noise_single_output(
            model, as_kind([[9.0], [8.0]], kind), as_kind([[0.2], [0.3], [0.4]], kind), True
        )
    after_y, after_noise = extract_targets_and_noise_single_output(model)
    assert_close(after_y, before_y)
    assert_close(after_noise, before_noise)


def test_restore_python_model_protocol_matches_official():
    importorskip_botorch()
    import torch
    from botorch.models import SingleTaskGP as OfficialGP

    model = OfficialGP(
        torch.tensor([[0.1], [0.4]], dtype=torch.double),
        torch.tensor([[1.0], [2.0]], dtype=torch.double),
        train_Yvar=torch.tensor([[0.01], [0.02]], dtype=torch.double),
        outcome_transform=None,
    )
    targets = torch.tensor([[3.0], [4.0]], dtype=torch.double)
    noise = torch.tensor([[0.2], [0.3]], dtype=torch.double)
    restore_targets_and_noise_single_output(model, targets, noise, True)
    torch.testing.assert_close(model.train_targets, targets.squeeze(-1))
    torch.testing.assert_close(model.likelihood.noise, noise.squeeze(-1))
