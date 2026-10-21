# ruff: noqa: E402
"""Iterative masked-grid inference retains complete Tensor derivative graphs."""

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models import LatentKroneckerGP
from botorch_rs.models._structured_iterative import (
    IterativeSettings,
    current,
    use_iterative_methods,
)
from botorch_rs.models._torch_structured_iterative import conjugate_gradient, logdet

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def gradients(loss, leaves, create_graph=False):
    values = torch.autograd.grad(
        loss, leaves, create_graph=create_graph, retain_graph=True, allow_unused=True
    )
    return [
        leaf * 0 if value is None else value for leaf, value in zip(leaves, values, strict=True)
    ]


def test_scoped_iterative_flags_restore_on_nested_exception():
    assert current() is None
    with use_iterative_methods(1e-7, 23, True, False, True):
        outer = current()
        assert (
            outer.tol,
            outer.max_iter,
            outer.covar_root_decomposition,
            outer.log_prob,
            outer.solves,
        ) == (1e-7, 23, True, False, True)
        with pytest.raises(RuntimeError, match="restore"):
            with use_iterative_methods(1e-4, 9, False, True, False):
                assert not current().solves
                raise RuntimeError("restore")
        assert current() is outer
    assert current() is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batched", [False, True])
def test_cg_shared_rhs_and_zero_columns_all_leaf_hessians(device, batched):
    torch.manual_seed(83)
    shape = (2, 7, 7) if batched else (7, 7)
    raw = torch.randn(*shape, dtype=torch.double, device=device, requires_grad=True)
    matrix = raw @ raw.transpose(-1, -2) + 0.4 * torch.eye(7, device=device, dtype=raw.dtype)
    rhs = torch.randn(7, 2, dtype=raw.dtype, device=device, requires_grad=True)
    values = torch.cat((rhs, torch.zeros_like(rhs[..., :1])), -1)
    actual = conjugate_gradient(matrix, values, IterativeSettings(tol=1e-11))
    expected = torch.linalg.solve(matrix, values)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    aa = gradients(actual.square().sum(), (raw, rhs), True)
    bb = gradients(expected.square().sum(), (raw, rhs), True)
    for a, b in zip(aa, bb, strict=True):
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-8)
    for a, b in zip(
        gradients(sum(g.square().sum() for g in aa), (raw, rhs)),
        gradients(sum(g.square().sum() for g in bb), (raw, rhs)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_lanczos_full_trace_logdet_all_leaf_hessians(device):
    torch.manual_seed(87)
    raw = torch.randn(6, 6, dtype=torch.double, device=device, requires_grad=True)
    matrix = raw @ raw.T + 0.7 * torch.eye(6, device=device, dtype=raw.dtype)
    config = IterativeSettings(tol=1e-10, max_cholesky_size=0, num_trace_samples=0)
    actual = logdet(matrix, config)
    expected = torch.linalg.slogdet(matrix)[1]
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    aa = gradients(actual, (raw,), True)
    bb = gradients(expected, (raw,), True)
    torch.testing.assert_close(aa[0], bb[0], atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(
        gradients(aa[0].square().sum(), (raw,))[0],
        gradients(bb[0].square().sum(), (raw,))[0],
        atol=2e-7,
        rtol=2e-7,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_stochastic_lanczos_matches_its_actual_probes_and_hessians(device):
    torch.manual_seed(188)
    raw = torch.randn(6, 6, dtype=torch.double, device=device, requires_grad=True)
    matrix = raw @ raw.T + 0.7 * torch.eye(6, device=device, dtype=raw.dtype)
    config = IterativeSettings(max_cholesky_size=0, num_trace_samples=13)
    torch.manual_seed(190)
    actual = logdet(matrix, config)
    torch.manual_seed(190)
    probes = matrix.new_empty(6, 13).bernoulli_(0.5) * 2 - 1
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    expected = ((eigenvectors.T @ probes).square() * eigenvalues.log()[:, None]).sum(0).mean()
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    aa, bb = gradients(actual, (raw,), True), gradients(expected, (raw,), True)
    torch.testing.assert_close(aa[0], bb[0], atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(
        gradients(aa[0].square().sum(), (raw,))[0],
        gradients(bb[0].square().sum(), (raw,))[0],
        atol=2e-7,
        rtol=2e-7,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_cg_iteration_limit_warns_and_returns_actual_iterate(device):
    matrix = torch.diag(torch.tensor([1.0, 2.0, 7.0], dtype=torch.double, device=device))
    matrix = matrix + 0.2
    rhs = matrix.new_tensor([[0.2], [1.2], [0.7]])
    from botorch_rs.exceptions.warnings import NumericsWarning

    with pytest.warns(NumericsWarning, match="iteration limit"):
        actual = conjugate_gradient(matrix, rhs, IterativeSettings(tol=1e-12, max_iter=1))
    assert not torch.allclose(actual, torch.linalg.solve(matrix, rhs), atol=1e-8, rtol=1e-8)
    assert torch.isfinite(actual).all()


def model_data(device):
    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device, requires_grad=True)
    T = X.new_tensor([[0.0], [0.6]])
    Y = (4 * X + T.squeeze(-1)).sin().detach()
    Y[1, 0] = float("nan")
    Y.requires_grad_()
    model = LatentKroneckerGP(X, T, Y, outcome_transform=None, backend="torch")
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    return model, X, Y, Q


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log_prob_flag", [False, True])
def test_masked_iterative_mll_posterior_sampling_and_full_leaf_hessians(device, log_prob_flag):
    model, X, Y, Q = model_data(device)
    leaves = (X, Y, Q, *model.parameters())
    exact = model.posterior(Q)
    exact_mll = model._exact_mll()
    config = IterativeSettings(
        tol=1e-10, max_cholesky_size=0, num_trace_samples=0, log_prob=log_prob_flag
    )
    with patch("botorch_rs.models._structured_iterative.setting") as settings:
        settings.side_effect = lambda name, default: getattr(config, name, default)
        with model.use_iterative_methods(tol=1e-10, log_prob=log_prob_flag):
            with (
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")),
                patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host copy")),
            ):
                posterior = model.posterior(Q)
                mll = model._exact_mll()
    assert posterior._iterative_settings.max_cholesky_size == 0
    torch.testing.assert_close(mll, exact_mll, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(posterior.mean, exact.mean, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(
        posterior._covariance_tensor, exact._covariance_tensor, atol=2e-9, rtol=2e-9
    )
    aa = gradients(
        mll + posterior.mean.square().sum() + posterior._covariance_tensor.square().sum(),
        leaves,
        True,
    )
    bb = gradients(
        exact_mll + exact.mean.square().sum() + exact._covariance_tensor.square().sum(),
        leaves,
        True,
    )
    for a, b in zip(aa, bb, strict=True):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)
    for a, b in zip(
        gradients(sum(g.square().sum() for g in aa), leaves),
        gradients(sum(g.square().sum() for g in bb), leaves),
        strict=True,
    ):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    size = posterior.base_sample_shape[-1]
    samples = posterior.rsample_from_base_samples(
        (size,), torch.eye(size, dtype=X.dtype, device=X.device)
    ).flatten(-2)
    centered = samples - posterior.mean.flatten(-2)
    torch.testing.assert_close(
        centered.T @ centered, posterior.distribution.covariance_matrix, atol=2e-8, rtol=2e-8
    )
    assert all(
        torch.isfinite(value).all()
        for value in gradients(samples.square().sum(), (X, Q, *model.parameters()))
    )


@pytest.mark.parametrize("device", DEVICES)
def test_source_forced_cg_values_and_query_derivatives(device):
    gpytorch = pytest.importorskip("gpytorch")
    pytest.importorskip("botorch")
    from botorch.models.latent_kronecker_gp import LatentKroneckerGP as Reference

    model, X, Y, Q = model_data(device)
    reference = Reference(X, model.train_T, Y, outcome_transform=None).to(X)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            dict(reference.named_parameters())[name].copy_(parameter)
    model.eval()
    reference.eval()
    with (
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.detach_test_caches(False),
        reference.use_iterative_methods(tol=1e-10, max_iter=200, log_prob=False),
    ):
        expected = reference(Q)
        mean, covariance = expected.mean, expected.covariance_matrix
        expected_gradient = gradients(mean.square().sum() + covariance.square().sum(), (Q,))[0]
    with (
        gpytorch.settings.max_cholesky_size(0),
        model.use_iterative_methods(tol=1e-10, max_iter=200, log_prob=False),
    ):
        actual = model(Q)
        torch.testing.assert_close(actual.mean, mean, atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(actual.covariance_matrix, covariance, atol=2e-8, rtol=2e-8)
        actual_gradient = gradients(
            actual.mean.square().sum() + actual.covariance_matrix.square().sum(), (Q,)
        )[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-7, rtol=2e-7)
