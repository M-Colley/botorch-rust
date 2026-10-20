"""Torch-dispatched exact GP algebra retains device and higher-order graphs."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from botorch_rs import _botorch_rs as native  # noqa: E402

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


def kernel(x, z, lengthscale, scale, kind="rbf", nu=2.5):
    return scale * native.tensor_kernel(kind, x, z, lengthscale, nu=nu)


def inputs(device="cpu", dtype=torch.double):
    values = [
        [[0.1, 0.2], [0.4, 0.7], [0.9, 0.6]],
        [0.3, -0.2, 0.5],
        [[0.25, 0.35], [0.65, 0.45]],
        [0.05, 0.07, 0.04],
        [[0.6, 0.8]],
        1.2,
        0.15,
        [0.02, 0.03],
    ]
    return tuple(torch.tensor(v, dtype=dtype, device=device, requires_grad=True) for v in values)


def native_outputs(*args, kind="rbf", nu=2.5):
    x, y, query, noise, lengthscale, scale, mean, observation_noise = args
    ktt = kernel(x, x, lengthscale, scale, kind, nu)
    ktq = kernel(x, query, lengthscale, scale, kind, nu)
    kqq = kernel(query, query, lengthscale, scale, kind, nu)
    posterior = native.tensor_exact_gp_posterior(
        y,
        ktt,
        ktq,
        kqq,
        mean.expand_as(y),
        mean.expand(query.shape[:-1]),
        noise,
        observation_noise=observation_noise,
    )
    mll = native.tensor_exact_mll(y, ktt, mean.expand_as(y), noise, normalize=True)
    return *posterior, mll


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_conditioning_and_mll_match_independent_torch_linalg(device, dtype):
    args = inputs(device, dtype)
    x, y, query, noise, lengthscale, scale, mean, observation_noise = args

    def expected_kernel(a, b):
        delta = (a.unsqueeze(-2) - b.unsqueeze(-3)) / lengthscale
        return scale * torch.exp(-0.5 * delta.square().sum(-1))

    ktt, ktq, kqq = expected_kernel(x, x), expected_kernel(x, query), expected_kernel(query, query)
    covariance = ktt + torch.diag_embed(noise)
    residual = y - mean
    expected_mean = mean + (ktq.T @ torch.linalg.solve(covariance, residual.unsqueeze(-1))).squeeze(
        -1
    )
    expected_covariance = kqq - ktq.T @ torch.linalg.solve(covariance, ktq)
    expected_covariance = expected_covariance + torch.diag_embed(observation_noise)
    expected_mll = (
        torch.distributions.MultivariateNormal(
            mean.expand_as(y), covariance_matrix=covariance
        ).log_prob(y)
        / y.shape[-1]
    )
    actual = native_outputs(*args)
    expected = (expected_mean, expected_covariance, expected_mll)
    for value, want in zip(actual, expected, strict=True):
        assert value.device.type == device and value.dtype == dtype and value.grad_fn is not None
        torch.testing.assert_close(
            value,
            want,
            rtol=2e-5 if dtype == torch.float32 else 2e-12,
            atol=2e-6 if dtype == torch.float32 else 2e-12,
        )
    loss = actual[0].square().sum() + actual[1].square().sum() + actual[2]
    expected_loss = expected[0].square().sum() + expected[1].square().sum() + expected[2]
    gradients = torch.autograd.grad(loss, args, create_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, args, create_graph=True)
    for value, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(
            value,
            want,
            rtol=5e-5 if dtype == torch.float32 else 2e-11,
            atol=5e-6 if dtype == torch.float32 else 2e-11,
        )
    hvp = torch.autograd.grad(sum(g.square().sum() for g in gradients), args)
    expected_hvp = torch.autograd.grad(sum(g.square().sum() for g in expected_gradients), args)
    for value, want in zip(hvp, expected_hvp, strict=True):
        torch.testing.assert_close(
            value,
            want,
            rtol=2e-4 if dtype == torch.float32 else 2e-10,
            atol=2e-5 if dtype == torch.float32 else 2e-10,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("kind", "nu"), [("rbf", 2.5), ("matern", 1.5), ("matern", 2.5)])
def test_exact_gp_all_tensor_inputs_pass_gradcheck_and_gradgradcheck(device, kind, nu):
    args = inputs(device)

    def function(*values):
        return native_outputs(*values, kind=kind, nu=nu)

    assert torch.autograd.gradcheck(function, args, fast_mode=True, atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(function, args, fast_mode=True, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("kind", "nu", "curvature"),
    [("rbf", 2.5, 1.0), ("matern", 1.5, 3.0), ("matern", 2.5, 5.0 / 3.0)],
)
def test_coincident_kernel_has_correct_hessian(device, kind, nu, curvature):
    point = torch.tensor([[0.3, 0.6]], dtype=torch.double, device=device, requires_grad=True)
    other = point.detach().clone()
    lengthscale = torch.tensor([[0.7, 0.9]], dtype=torch.double, device=device, requires_grad=True)
    value = native.tensor_kernel(kind, point, other, lengthscale, nu=nu).sum()
    (gradient,) = torch.autograd.grad(value, point, create_graph=True)
    assert torch.equal(gradient, torch.zeros_like(gradient))
    rows = [torch.autograd.grad(gradient[0, j], point, retain_graph=True)[0][0] for j in range(2)]
    hessian = torch.stack(rows)
    torch.testing.assert_close(hessian, -torch.diag(curvature / lengthscale[0].square()))


@pytest.mark.parametrize("device", DEVICES)
def test_arbitrary_broadcast_batches_and_independent_outputs(device):
    args = inputs(device)
    x, y, query, noise, lengthscale, scale, mean, observation_noise = args
    x = x.expand(2, 1, 3, 2)
    y = torch.stack([y, y + 0.1]).unsqueeze(1)
    query = torch.stack([query, query + 0.05, query + 0.1]).unsqueeze(0)
    noise = noise.expand(2, 1, 3)
    values = native_outputs(x, y, query, noise, lengthscale, scale, mean, observation_noise)
    assert values[0].shape == (2, 3, 2)
    assert values[1].shape == (2, 3, 2, 2)
    assert values[2].shape == (2, 1)
    for output in range(2):
        for candidate in range(3):
            direct = native_outputs(
                x[output, 0],
                y[output, 0],
                query[0, candidate],
                noise[output, 0],
                lengthscale,
                scale,
                mean,
                observation_noise,
            )
            torch.testing.assert_close(values[0][output, candidate], direct[0])
            torch.testing.assert_close(values[1][output, candidate], direct[1])
    gradients = torch.autograd.grad(sum(v.sum() for v in values), args, create_graph=True)
    assert all(torch.isfinite(g).all() and g.device.type == device for g in gradients)


@pytest.mark.parametrize("device", DEVICES)
def test_kernel_diag_last_dimension_and_noncontiguous_inputs(device):
    x = torch.arange(12, dtype=torch.double, device=device).reshape(3, 4).T / 12
    z = x + 0.1
    lengthscale = torch.tensor([[0.6, 0.8, 0.9]], dtype=torch.double, device=device)
    full = native.tensor_kernel("rbf", x, z, lengthscale)
    diag = native.tensor_kernel("rbf", x, z, lengthscale, diag=True)
    torch.testing.assert_close(diag, full.diagonal())
    by_dimension = native.tensor_kernel("rbf", x, z, lengthscale, last_dim_is_batch=True)
    expected = torch.exp(
        -0.5 * (((x / lengthscale).T.unsqueeze(-1) - (z / lengthscale).T.unsqueeze(-2)) ** 2)
    )
    torch.testing.assert_close(by_dimension, expected)


def test_jitter_is_explicit_and_shape_errors_are_clear():
    y = torch.tensor([0.1, 0.2], dtype=torch.double)
    bad_covariance = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.double)
    mean, noise = torch.zeros_like(y), torch.zeros_like(y)
    with pytest.raises(torch.linalg.LinAlgError):
        native.tensor_exact_mll(y, bad_covariance, mean, noise)
    value = native.tensor_exact_mll(y, bad_covariance, mean, noise, jitter=2.0)
    assert torch.isfinite(value)
    with pytest.raises(ValueError, match="jitter"):
        native.tensor_exact_mll(y, bad_covariance, mean, noise, jitter=math.nan)
    with pytest.raises(ValueError, match="k_train"):
        native.tensor_exact_mll(y, torch.eye(3), mean, noise)
    with pytest.raises(ValueError, match="Torch tensor"):
        native.tensor_kernel("rbf", [[0.1]], [[0.2]])
    with pytest.raises(NotImplementedError, match="nu"):
        native.tensor_kernel("matern", y.unsqueeze(-1), y.unsqueeze(-1), nu=0.8)


@pytest.mark.parametrize("device", DEVICES)
def test_psd_safe_cholesky_pd_path_has_exact_graph_and_higher_derivatives(device):
    raw = torch.tensor(
        [[1.2, 0.2], [0.1, 0.8]], dtype=torch.double, device=device, requires_grad=True
    )

    def function(value):
        matrix = value @ value.T + torch.eye(2, dtype=value.dtype, device=value.device) * 0.2
        return native.tensor_psd_safe_cholesky(matrix)

    assert torch.autograd.gradcheck(function, (raw,), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (raw,), fast_mode=True)
    matrix = raw @ raw.T + torch.eye(2, dtype=raw.dtype, device=raw.device) * 0.2
    actual = native.tensor_psd_safe_cholesky(matrix)
    assert actual.device == matrix.device and actual.grad_fn is not None
    torch.testing.assert_close(actual, torch.linalg.cholesky(matrix), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_psd_safe_cholesky_only_jitters_failing_batch_members(device, dtype):
    from botorch_rs.exceptions.warnings import NumericsWarning

    matrix = (
        torch.stack([torch.eye(2), torch.diag(torch.tensor([1.0, 0.0]))])
        .to(device=device, dtype=dtype)
        .requires_grad_()
    )
    original = matrix.detach().clone()
    jitter = 1e-8 if dtype == torch.float64 else 1e-6
    with pytest.warns(NumericsWarning, match="added jitter") as warnings:
        actual = native.tensor_psd_safe_cholesky(matrix)
    assert len(warnings) == 1
    assert torch.equal(matrix, original)
    additions = torch.tensor([[0.0, 0.0], [jitter, jitter]], dtype=dtype, device=device)
    expected = torch.linalg.cholesky(matrix + torch.diag_embed(additions))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (gradient,) = torch.autograd.grad(actual.sum(), matrix, create_graph=True)
    (expected_gradient,) = torch.autograd.grad(expected.sum(), matrix, create_graph=True)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    (hessian_vector,) = torch.autograd.grad(gradient.sum(), matrix)
    (expected_hessian_vector,) = torch.autograd.grad(expected_gradient.sum(), matrix)
    torch.testing.assert_close(hessian_vector, expected_hessian_vector, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_psd_safe_cholesky_retains_successful_batches_during_later_retries(device):
    from botorch_rs.exceptions.warnings import NumericsWarning

    matrix = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, -5e-8]]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    with pytest.warns(NumericsWarning) as warnings:
        lower = native.tensor_psd_safe_cholesky(matrix, jitter=1e-8, max_tries=2)
    assert len(warnings) == 2
    additions = torch.tensor([[1e-8, 1e-8], [1e-7, 1e-7]], dtype=matrix.dtype, device=device)
    torch.testing.assert_close(lower, torch.linalg.cholesky(matrix + torch.diag_embed(additions)))
    assert torch.isfinite(torch.autograd.grad(lower.sum(), matrix)[0]).all()


def test_psd_safe_cholesky_exhaustion_and_invalid_options_are_explicit():
    from botorch_rs.exceptions.errors import NotPSDError
    from botorch_rs.exceptions.warnings import NumericsWarning

    bad = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.double)
    with pytest.warns(NumericsWarning) as warnings, pytest.raises(NotPSDError):
        native.tensor_psd_safe_cholesky(bad, max_tries=2)
    assert len(warnings) == 2
    with pytest.raises(NotPSDError):
        native.tensor_psd_safe_cholesky(bad, max_tries=0)
    with pytest.raises(ValueError, match="jitter"):
        native.tensor_psd_safe_cholesky(bad, jitter=-1.0)
    with pytest.raises(ValueError, match="n x n"):
        native.tensor_psd_safe_cholesky(torch.ones(2, 3))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reused_training_factors_preserve_complete_training_and_query_graphs(device, dtype):
    args = inputs(device, dtype)
    x, y, query, noise, lengthscale, scale, mean, observation_noise = args
    ktt = kernel(x, x, lengthscale, scale)
    chol, alpha = native.tensor_exact_gp_factor(y, ktt, mean.expand_as(y), noise)
    assert chol.shape == (3, 3) and alpha.shape == (3,)
    assert chol.grad_fn is not None and alpha.grad_fn is not None
    covariance = ktt + torch.diag_embed(noise)
    actual_loss, expected_loss = 0.0, 0.0
    for weight, offset in [(-0.3, 0.0), (0.7, 0.05)]:
        current = query + offset
        cross = kernel(x, current, lengthscale, scale)
        prior = kernel(current, current, lengthscale, scale)
        values, cov = native.tensor_exact_gp_predict(
            chol, alpha, cross, prior, mean.expand(current.shape[:-1]), observation_noise
        )
        expected = mean + (
            cross.T @ torch.linalg.solve(covariance, (y - mean).unsqueeze(-1))
        ).squeeze(-1)
        expected_cov = (
            prior
            - cross.T @ torch.linalg.solve(covariance, cross)
            + torch.diag_embed(observation_noise)
        )
        actual_loss = actual_loss + weight * (values.square().sum() + cov.square().sum())
        expected_loss = expected_loss + weight * (
            expected.square().sum() + expected_cov.square().sum()
        )
    gradients = torch.autograd.grad(actual_loss, args, create_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, args, create_graph=True)
    tolerance = (
        dict(rtol=2e-4, atol=2e-5) if dtype == torch.float32 else dict(rtol=2e-10, atol=2e-10)
    )
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, **tolerance)
    hvp = torch.autograd.grad(sum(g.square().sum() for g in gradients), args, retain_graph=True)
    expected_hvp = torch.autograd.grad(sum(g.square().sum() for g in expected_gradients), args)
    for actual, expected in zip(hvp, expected_hvp, strict=True):
        torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_frozen_training_cache_retains_higher_order_query_graph(device):
    x, y, query, noise, lengthscale, scale, mean, observation_noise = (
        t.detach() for t in inputs(device)
    )
    query.requires_grad_()
    with torch.no_grad():
        chol, alpha = native.tensor_exact_gp_factor(
            y, kernel(x, x, lengthscale, scale), mean.expand_as(y), noise
        )
    assert not chol.requires_grad and not alpha.requires_grad

    def function(q):
        return native.tensor_exact_gp_predict(
            chol,
            alpha,
            kernel(x, q, lengthscale, scale),
            kernel(q, q, lengthscale, scale),
            mean.expand(q.shape[:-1]),
            observation_noise,
        )

    assert torch.autograd.gradcheck(function, (query,), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (query,), fast_mode=True)


def test_predict_does_not_repeat_training_cholesky(monkeypatch):
    x, y, query, noise, lengthscale, scale, mean, _ = inputs()
    calls = []
    original = torch.linalg.cholesky

    def counted(matrix, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    chol, alpha = native.tensor_exact_gp_factor(
        y, kernel(x, x, lengthscale, scale), mean.expand_as(y), noise
    )
    for offset in (0.0, 0.05, 0.1):
        q = query + offset
        native.tensor_exact_gp_predict(
            chol,
            alpha,
            kernel(x, q, lengthscale, scale),
            kernel(q, q, lengthscale, scale),
            mean.expand(q.shape[:-1]),
        )
    assert calls == [(3, 3)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "factor_batch,cross_batch", [((), (2, 3)), ((3,), (2, 3)), ((1, 3), (2, 4, 3))]
)
def test_packed_query_rhs_matches_broadcast_solve_and_full_derivatives(
    device, factor_batch, cross_batch, monkeypatch
):
    torch.manual_seed(41)
    raw = torch.randn(*factor_batch, 4, 4, dtype=torch.double, device=device, requires_grad=True)
    chol = raw.tril(-1) * 0.1 + torch.eye(4, dtype=torch.double, device=device) * 1.3
    alpha = torch.randn(*factor_batch, 4, dtype=torch.double, device=device, requires_grad=True)
    cross = (
        torch.randn(*cross_batch, 4, 2, dtype=torch.double, device=device, requires_grad=True) * 0.1
    )
    query = torch.eye(2, dtype=torch.double, device=device).expand(*cross_batch, 2, 2)
    mean = torch.zeros(*cross_batch, 2, dtype=torch.double, device=device)
    original = torch.linalg.solve_triangular
    calls = []

    def counted(a, b, **kwargs):
        calls.append((tuple(a.shape), tuple(b.shape)))
        return original(a, b, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve_triangular", counted)
    values, covariance = native.tensor_exact_gp_predict(chol, alpha, cross, query, mean)
    assert len(calls) == 1
    # Query-only batch dimensions must become RHS columns, not repeated factors.
    assert len(calls[0][1]) == len(chol.shape)
    expected_mean = (cross.transpose(-1, -2) @ alpha.unsqueeze(-1)).squeeze(-1)
    solution = original(chol, cross, upper=False)
    expected_covariance = query - solution.transpose(-1, -2) @ solution
    torch.testing.assert_close(values, expected_mean, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(covariance, expected_covariance, atol=1e-12, rtol=1e-12)
    leaves = (raw, alpha, cross)
    gradient = torch.autograd.grad(
        values.square().sum() + covariance.square().sum(), leaves, create_graph=True
    )
    expected_gradient = torch.autograd.grad(
        expected_mean.square().sum() + expected_covariance.square().sum(), leaves, create_graph=True
    )
    for actual, expected in zip(gradient, expected_gradient, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradient), leaves, retain_graph=True)
    expected_second = torch.autograd.grad(sum(g.square().sum() for g in expected_gradient), leaves)
    for actual, expected in zip(second, expected_second, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
