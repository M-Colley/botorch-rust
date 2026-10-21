"""Real cached Kronecker training covariance / predictive mean solves."""

from __future__ import annotations

import copy

import pytest

from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape

importorskip_botorch()
torch = pytest.importorskip("torch")

import multitask_helpers as h  # noqa: E402

from botorch_rs.models import KroneckerMultiTaskGP  # noqa: E402


@pytest.mark.parametrize("case", h.kron_cases(), ids=lambda case: case["name"])
@pytest.mark.parametrize("kind", KINDS)
def test_public_cache_values_against_official_and_dense_solve(case, kind):
    x, y = h.kron_data(case, 123)
    reference, original = h.build_kron_pair(case, x, y, 543)
    reference.eval()
    # Preserve the identical hyperparameters, while returning each array kind.
    if kind == "torch":
        model = original
    else:
        torch.manual_seed(543)
        model = KroneckerMultiTaskGP(
            as_kind(x.tolist(), kind), as_kind(y.tolist(), kind), **h.kron_kwargs("rs", case)
        )
        model._rs.set_raw_parameters(original._rs.raw_parameters())
    full = model.train_full_covar
    alpha = model.predictive_mean_cache
    check_kind(full, kind)
    check_kind(alpha, kind)
    nm = case["n"] * case["m"]
    assert shape(full) == (nm, nm) and shape(alpha) == (nm,)
    assert_close(full, reference.train_full_covar.to_dense(), rtol=2e-8, atol=2e-9)
    assert_close(alpha, reference.predictive_mean_cache, rtol=2e-8, atol=2e-9)
    noisy = reference.train_full_covar.to_dense() + torch.kron(
        torch.eye(case["n"], dtype=torch.float64), h.kron_noise(reference)
    )
    residual = reference.train_targets - reference.mean_module(reference.train_inputs[0])
    expected = torch.linalg.solve(noisy, residual.reshape(-1))
    assert_close(alpha, expected, rtol=2e-8, atol=2e-9)
    if kind == "torch":
        assert not full.requires_grad and not alpha.requires_grad


@pytest.mark.parametrize("kind", KINDS)
def test_cache_parameter_changes_and_output_mutation_do_not_change_snapshots(kind):
    case = h.kron_cases()[0]
    x, y = h.kron_data(case, 222)
    torch.manual_seed(54)
    model = KroneckerMultiTaskGP(as_kind(x.tolist(), kind), as_kind(y.tolist(), kind))
    before = model.predictive_mean_cache
    covariance = model.train_full_covar
    snapshot = copy.deepcopy(model)
    # Noise changes the solve, while the noiseless covariance remains identical.
    model.likelihood.noise = 2.3
    after = model.predictive_mean_cache
    assert not torch.allclose(torch.as_tensor(before), torch.as_tensor(after))
    assert_close(model.train_full_covar, covariance)
    assert_close(snapshot.predictive_mean_cache, before)
    # Kernel mutation invalidates the training covariance as well.
    model.covar_module.data_covar_module.lengthscale = 0.8
    changed = model.train_full_covar
    assert not torch.allclose(torch.as_tensor(changed), torch.as_tensor(covariance))
    assert_close(snapshot.train_full_covar, covariance)
    # Arrays returned to callers are independent copies of the native cache.
    if kind == "list":
        changed[0][0] += 17
    else:
        changed[0, 0] += 17
    assert not torch.allclose(torch.as_tensor(changed), torch.as_tensor(model.train_full_covar))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cache_dtype_and_flattened_interleaving(dtype):
    x = torch.tensor([[0.1], [0.5], [0.9]], dtype=dtype)
    y = torch.cat((x.sin(), x.cos()), -1)
    model = KroneckerMultiTaskGP(x, y)
    covariance = model.train_full_covar
    alpha = model.predictive_mean_cache
    assert covariance.dtype == alpha.dtype == dtype
    assert covariance.device == alpha.device == x.device
    task = model.covar_module.task_covar_module.covar_matrix
    data = model.covar_module.data_covar_module(x)
    torch.testing.assert_close(covariance, torch.kron(data, task), rtol=2e-5, atol=2e-6)
