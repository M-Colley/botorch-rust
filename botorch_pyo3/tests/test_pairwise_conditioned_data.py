"""Conditioned preference data, ownership, and atomic data replacement."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

pytest.importorskip("numpy")


import numpy as np

from helpers import assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from test_pairwise_batched_warp import models
from test_pairwise_warp import models as unbatched_models


def conditioned_pair(training, nested):
    bm, rm, _, _ = models(per_member=True, chained=True)
    bm.train(training)
    rm.train(training)
    points = torch.tensor(
        [[[0.18, 0.35], [0.73, 0.3]], [[2.4, 2.8], [2.65, 2.6]]], dtype=torch.float64
    )
    comparisons = torch.tensor([[[[0, 1]], [[1, 0]]], [[[1, 0]], [[0, 1]]], [[[0, 1]], [[0, 1]]]])
    np.random.seed(271)
    # Official cached non-leaf utilities cannot be deep-copied for a second
    # conditioning step; no-grad conditioning avoids that separate limitation.
    with torch.no_grad():
        bc = bm.condition_on_observations(points, comparisons)
    np.random.seed(271)
    rc = rm.condition_on_observations(points, comparisons)
    if nested:
        points = points + 0.06
        comparisons = comparisons.expand(2, *comparisons.shape)
        np.random.seed(272)
        with torch.no_grad():
            bc = bc.condition_on_observations(points, comparisons)
        np.random.seed(272)
        rc = rc.condition_on_observations(points, comparisons)
    return bc, rc


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_conditioned_training_metadata_contains_all_observations(training, nested):
    bc, rc = conditioned_pair(training, nested)
    assert_close(rc.train_inputs[0], bc.train_inputs[0])
    assert_close(rc.train_targets, bc.train_targets)
    assert tuple(rc.train_inputs[0].shape) == tuple(bc.train_inputs[0].shape)
    assert tuple(rc.train_targets.shape) == tuple(bc.train_targets.shape)
    assert rc.train_targets.dtype == torch.long
    assert rc.train_inputs[0] is rc.unconsolidated_datapoints
    assert rc.train_targets is rc.unconsolidated_comparisons
    assert_close(rc.covar, bc.covar, rtol=4e-7, atol=4e-8)
    assert_close(rc.input_transform.norm.bounds, bc.input_transform.norm.bounds)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_conditioned_data_replacement_retains_original_parameter_batch(training, nested):
    from test_pairwise_warp import reference_mll_gradient

    from botorch_rs.models import PairwiseLaplaceMarginalLogLikelihood as RsMLL

    bc, rc = conditioned_pair(training, nested)
    held_kernel, held_transform = rc.covar_module, rc.input_transform
    held_warp = held_transform.warp
    parameters = {
        name: torch.as_tensor(value, dtype=torch.float64).clone()
        for name, value in rc.named_parameters()
    }
    points = bc.datapoints + torch.tensor([0.15, -0.1], dtype=torch.float64)
    comparisons = bc.comparisons
    np.random.seed(273)
    bc.set_train_data(points, comparisons)
    np.random.seed(273)
    rc.set_train_data(points, comparisons)
    assert rc.covar_module is held_kernel
    assert rc.input_transform is held_transform
    assert rc.input_transform.warp is held_warp
    assert held_kernel._model is rc and held_warp._model is rc
    assert rc._parameter_batch == (2,)
    for name, value in rc.named_parameters():
        assert_close(value, parameters[name])
        assert_close(value, dict(bc.named_parameters())[name])
    assert_close(rc.train_inputs[0], bc.train_inputs[0])
    assert_close(rc.train_targets, bc.train_targets)
    assert_close(rc.covar, bc.covar, rtol=4e-7, atol=4e-8)
    assert_close(rc.utility, bc.utility, rtol=4e-7, atol=4e-8)
    assert_close(held_transform.norm.bounds, bc.input_transform.norm.bounds)
    value, gradients = reference_mll_gradient(bc)
    actual, gradient = RsMLL(rc.likelihood, rc).value_and_grad()
    assert_close(actual, value, rtol=4e-7, atol=4e-8)
    assert_close(
        gradient,
        torch.cat([gradients[n].reshape(-1) for n, _ in rc._param_names()]),
        rtol=4e-6,
        atol=4e-7,
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("failure", ["indices", "batch"])
def test_failed_batched_replacement_preserves_snapshot_modules_and_posterior(nested, failure):
    _, model = conditioned_pair(False, nested)
    held_kernel, held_transform = model.covar_module, model.input_transform
    held_warp = held_transform.warp
    native = model._rs
    raw = native.raw_parameters()
    bounds = held_transform.norm.bounds.clone()
    concentrations = torch.as_tensor(held_warp.concentration0, dtype=torch.float64).clone()
    covar = model.covar.clone()
    inputs, targets = model.train_inputs[0], model.train_targets
    query = torch.tensor([[[0.3, 0.6]], [[2.3, 2.6]]], dtype=torch.float64)
    expected = model.posterior(query).mean.clone()
    if failure == "indices":
        points = model.datapoints
        comparisons = model.comparisons.clone()
        comparisons[..., 0, 0] = 10_000
    else:
        points = model.datapoints.reshape(-1, model.n, model.dim)[0].unsqueeze(0).expand(3, -1, -1)
        comparisons = torch.tensor([[[0, 1]]] * 3)
    with pytest.raises((RuntimeError, ValueError)):
        model.set_train_data(points, comparisons)
    assert model._rs is native
    assert native.raw_parameters() == raw
    assert model.covar_module is held_kernel and model.input_transform is held_transform
    assert held_kernel._model is model and held_warp._model is model
    assert model.train_inputs[0] is inputs and model.train_targets is targets
    assert_close(held_transform.norm.bounds, bounds)
    assert_close(held_warp.concentration0, concentrations)
    assert_close(model.covar, covar)
    assert_close(model.posterior(query).mean, expected)


def test_failed_unbatched_replacement_preserves_native_and_bound_modules():
    _, model, _, held_warp = unbatched_models(chained=True)
    held_kernel, held_transform = model.covar_module, model.input_transform
    native = model._rs
    raw = native.raw_parameters()
    query = torch.tensor([[0.3, 0.6]], dtype=torch.float64)
    expected = model.posterior(query).mean.clone()
    with pytest.raises((RuntimeError, ValueError)):
        model.set_train_data(model.datapoints, torch.tensor([[0, 10_000]]))
    assert model._rs is native
    assert native.raw_parameters() == raw
    assert model.covar_module is held_kernel and model.input_transform is held_transform
    assert held_kernel._model is model and held_warp._model is model
    assert_close(model.posterior(query).mean, expected)
