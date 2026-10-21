"""Public transformation helpers preserve device values and complete graphs."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models.transforms import utils

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _derivatives(value, leaves, directions):
    first = torch.autograd.grad(value, leaves, create_graph=True)
    hvp = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(first, directions, strict=True)), leaves
    )
    return (*first, *hvp)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.double, torch.float])
def test_lognormal_marginals_broadcast_values_and_analytic_gradient_hvp(device, dtype):
    mu = torch.linspace(0.1, 0.4, 6, device=device, dtype=dtype).reshape(2, 1, 3).requires_grad_()
    var = torch.linspace(0.2, 0.6, 12, device=device, dtype=dtype).reshape(1, 4, 3).requires_grad_()
    weights = torch.linspace(0.2, 0.8, 24, device=device, dtype=dtype).reshape(2, 4, 3)
    directions = (torch.full_like(mu, 0.3), torch.full_like(var, -0.2))
    mean, variance = utils.norm_to_lognorm_mean(mu, var), utils.norm_to_lognorm_variance(mu, var)
    actual = _derivatives((weights * (mean + variance)).sum(), (mu, var), directions)
    E = torch.exp(mu + var / 2)
    F = torch.exp(2 * mu + 2 * var) - torch.exp(2 * mu + var)
    G = 2 * torch.exp(2 * mu + 2 * var) - torch.exp(2 * mu + var)
    J = 4 * torch.exp(2 * mu + 2 * var) - torch.exp(2 * mu + var)
    dm, dv = directions
    expected = (
        (weights * (E + 2 * F)).sum_to_size(mu.shape),
        (weights * (E / 2 + G)).sum_to_size(var.shape),
        (weights * (E * (dm + dv / 2) + 2 * (2 * F * dm + G * dv))).sum_to_size(mu.shape),
        (weights * (E * (dm + dv / 2) / 2 + 2 * G * dm + J * dv)).sum_to_size(var.shape),
    )
    tolerance = 3e-5 if dtype == torch.float else 3e-12
    torch.testing.assert_close(mean, E, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(variance, F, atol=tolerance, rtol=tolerance)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=tolerance, rtol=tolerance)
    # expm1 must retain variances far smaller than exp(var)'s rounding unit.
    tiny = var.new_full((1,), 1e-14 if dtype == torch.double else 1e-8)
    torch.testing.assert_close(
        utils.norm_to_lognorm_variance(tiny * 0, tiny), tiny, atol=0, rtol=1e-6
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("inverse", [False, True])
def test_full_lognormal_moments_match_source_and_pass_gradgradcheck(device, inverse):
    pytest.importorskip("botorch")
    from botorch.models.transforms import utils as source

    mu = torch.tensor([[0.2, -0.1]], dtype=torch.double, device=device)
    covariance = mu.new_tensor([[[0.3, 0.08], [0.08, 0.2]], [[0.4, 0.12], [0.12, 0.3]]])
    if inverse:
        mu, covariance = source.norm_to_lognorm(mu, covariance)
    mu, covariance = mu.detach().requires_grad_(), covariance.detach().requires_grad_()
    function = utils.lognorm_to_norm if inverse else utils.norm_to_lognorm
    reference = source.lognorm_to_norm if inverse else source.norm_to_lognorm
    actual, expected = function(mu, covariance), reference(mu, covariance)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=3e-14, rtol=3e-13)
    directions = (torch.full_like(mu, 0.2), torch.full_like(covariance, -0.1))
    for a, e in zip(
        _derivatives(sum(v.sin().sum() for v in actual), (mu, covariance), directions),
        _derivatives(sum(v.sin().sum() for v in expected), (mu, covariance), directions),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-12, rtol=3e-11)
    assert torch.autograd.gradcheck(function, (mu, covariance), atol=2e-6, rtol=2e-5)
    assert torch.autograd.gradgradcheck(function, (mu, covariance), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_lognormal_roundtrip_retains_identity_jacobian_and_zero_hessian(device):
    mu = torch.tensor([[0.1, 0.3]], device=device, dtype=torch.double, requires_grad=True)
    covariance = mu.new_tensor(
        [[[0.2, 0.04], [0.04, 0.3]], [[0.3, 0.05], [0.05, 0.4]]], requires_grad=True
    )
    restored = utils.lognorm_to_norm(*utils.norm_to_lognorm(mu, covariance))
    torch.testing.assert_close(restored[0], mu.expand(2, 2), atol=3e-16, rtol=3e-15)
    torch.testing.assert_close(restored[1], covariance, atol=3e-16, rtol=3e-15)
    first = torch.autograd.grad(sum(v.sum() for v in restored), (mu, covariance), create_graph=True)
    torch.testing.assert_close(first[0], torch.full_like(mu, 2), atol=3e-15, rtol=3e-15)
    torch.testing.assert_close(first[1], torch.ones_like(covariance), atol=3e-15, rtol=3e-15)
    second = torch.autograd.grad(sum(v.sum() for v in first), (mu, covariance))
    for value in second:
        torch.testing.assert_close(value, torch.zeros_like(value), atol=3e-14, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(4,), (3, 1), (0,)])
def test_expand_copy_broadcasts_and_folds_gradients_without_aliasing(device, batch):
    X = (
        torch.linspace(0.1, 0.6, 6, dtype=torch.double, device=device)
        .reshape(1, 2, 3)
        .requires_grad_()
    )
    result = utils.expand_and_copy_tensor(X, torch.Size(batch))
    repeats = result.numel() // X.numel()
    (first,) = torch.autograd.grad(result.square().sum(), X, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), X)
    torch.testing.assert_close(first, 2 * X * repeats, atol=0, rtol=0)
    torch.testing.assert_close(second, torch.full_like(X, 2 * repeats), atol=0, rtol=0)
    if result.numel():
        assert result.data_ptr() != X.data_ptr()
    with pytest.raises(RuntimeError, match="not broadcastable"):
        utils.expand_and_copy_tensor(X.expand(2, 2, 3), (3,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("indices", ["unsorted", "negative", "boolean"])
@pytest.mark.parametrize("batched", [False, True])
def test_subset_transform_keeps_physical_columns_and_full_parameter_graph(device, indices, batched):
    pytest.importorskip("botorch")
    from botorch.models.transforms.utils import subset_transform as reference

    X = (
        torch.linspace(0.1, 0.6, 6, dtype=torch.double, device=device)
        .reshape(1, 2, 3)
        .requires_grad_()
    )
    columns = [2, 0] if indices != "boolean" else [0, 2]
    selection = {"unsorted": [2, 0], "negative": [-1, 0], "boolean": [True, False, True]}[indices]

    class Transform:
        def __init__(self):
            self.indices = torch.tensor(selection, device=device)
            self.weight = (
                X.new_tensor([0.7, 0.9])
                .expand(*((3, 1) if batched else ()), 2)
                .clone()
                .requires_grad_()
            )
            if batched:
                self.batch_shape = torch.Size([3])

        def apply(self, inputs, offset):
            return (inputs * self.weight).sin() + offset

    transform = Transform()
    offset = X.new_tensor(0.2, requires_grad=True)
    actual = utils.subset_transform(Transform.apply)(transform, X, offset=offset)
    expected = reference(Transform.apply)(transform, X, offset=offset)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    selected = transform.apply(X[..., columns], offset)
    independent = torch.stack(
        [
            selected[..., columns.index(i)]
            if i in columns
            else X[..., i].expand(selected.shape[:-1])
            for i in range(3)
        ],
        -1,
    )
    torch.testing.assert_close(actual, independent, atol=0, rtol=0)
    leaves = X, transform.weight, offset
    directions = tuple(torch.full_like(value, 0.1) for value in leaves)
    for a, e in zip(
        _derivatives(actual.square().sum(), leaves, directions),
        _derivatives(independent.square().sum(), leaves, directions),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-13, rtol=3e-12)
    assert actual.data_ptr() != X.data_ptr()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dimensions", [1, 3, 4])
def test_interaction_features_have_exact_independent_query_gradient_and_hvp(device, dimensions):
    X = (
        torch.linspace(0.1, 0.9, 2 * dimensions, dtype=torch.double, device=device)
        .reshape(2, dimensions)
        .requires_grad_()
    )
    direction = torch.full_like(X, 0.2)
    actual = utils.interaction_features(X)
    expected = (
        torch.stack(
            [X[..., i] * X[..., j] for i in range(dimensions) for j in range(i + 1, dimensions)], -1
        ).unsqueeze(-2)
        if dimensions > 1
        else X[..., :0].unsqueeze(-2)
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    first, hvp = _derivatives(actual.sum(), (X,), (direction,))
    torch.testing.assert_close(first, X.sum(-1, keepdim=True) - X, atol=3e-16, rtol=3e-15)
    torch.testing.assert_close(
        hvp, direction.sum(-1, keepdim=True) - direction, atol=3e-16, rtol=3e-15
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("keepdim", [False, True])
def test_nanstd_ignores_missing_values_before_arithmetic_and_keeps_full_derivatives(
    device, keepdim
):
    X = torch.tensor(
        [[0.2, float("nan"), 0.8, 0.5], [0.1, 0.7, float("nan"), 0.4]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    actual = utils.nanstd(X, dim=-1, keepdim=keepdim)
    expected = torch.stack([X[0, [0, 2, 3]].std(), X[1, [0, 1, 3]].std()])
    if keepdim:
        expected = expected.unsqueeze(-1)
    torch.testing.assert_close(actual, expected, atol=3e-16, rtol=3e-15)
    direction = torch.full_like(X, 0.1)
    for a, e in zip(
        _derivatives(actual.square().sum() + actual.sin().sum(), (X,), (direction,)),
        _derivatives(expected.square().sum() + expected.sin().sum(), (X,), (direction,)),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-13, rtol=3e-12)
        assert bool(torch.isfinite(a).all())
    constants = X.new_tensor([[0.5, 0.5, float("nan")]], requires_grad=True)
    std = utils.nanstd(constants, -1)
    (first,) = torch.autograd.grad(std.sum(), constants, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), constants)
    torch.testing.assert_close(std, std.new_zeros(1), atol=0, rtol=0)
    for value in (first, second):
        torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
    assert bool(
        torch.isnan(
            utils.nanstd(X.new_tensor([[float("nan"), float("nan")], [0.5, float("nan")]]), -1)
        ).all()
    )


@pytest.mark.parametrize("device", DEVICES)
def test_nanstd_matches_source_values_and_smooth_first_derivatives(device):
    pytest.importorskip("botorch")
    from botorch.models.transforms.utils import nanstd as source

    X = torch.tensor(
        [[0.2, 0.3, 0.8], [0.4, 0.7, 0.5]], dtype=torch.double, device=device, requires_grad=True
    )
    actual, expected = utils.nanstd(X, 0), source(X, 0)
    torch.testing.assert_close(actual, expected, atol=3e-16, rtol=3e-15)
    direction = torch.linspace(0.1, 0.4, X.numel(), device=device, dtype=X.dtype).reshape_as(X)
    for a, e in zip(
        _derivatives(actual.square().sum(), (X,), (direction,)),
        _derivatives(expected.square().sum(), (X,), (direction,)),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-13, rtol=3e-12)
    with_nan = X.detach().clone()
    with_nan[0, 0] = float("nan")
    torch.testing.assert_close(
        utils.nanstd(with_nan, -1), source(with_nan, -1), atol=3e-16, rtol=3e-15
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("inverse", [False, True])
def test_kumaraswamy_values_complete_graphs_and_numerical_second_derivatives(device, inverse):
    pytest.importorskip("botorch")
    from botorch.models.transforms import utils as source

    X = torch.tensor(
        [[0.2, 0.5], [0.7, 0.4]], dtype=torch.double, device=device, requires_grad=True
    )
    c0 = X.new_tensor([0.7, 1.3], requires_grad=True)
    c1 = X.new_tensor([1.1, 0.8], requires_grad=True)
    function = utils.inv_kumaraswamy_warp if inverse else utils.kumaraswamy_warp
    reference = source.inv_kumaraswamy_warp if inverse else source.kumaraswamy_warp
    actual, expected = function(X, c0, c1, eps=0.03), reference(X, c0, c1, eps=0.03)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    leaves, directions = (X, c0, c1), tuple(torch.full_like(value, 0.1) for value in (X, c0, c1))
    for a, e in zip(
        _derivatives(actual.sin().sum(), leaves, directions),
        _derivatives(expected.sin().sum(), leaves, directions),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-13, rtol=3e-12)
    assert torch.autograd.gradgradcheck(
        lambda x, a, b: function(x, a, b, eps=0.03), leaves, atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("device", DEVICES)
def test_transform_utilities_do_not_copy_tensor_values_to_host(device, monkeypatch):
    X = torch.tensor(
        [[0.2, 0.5], [0.7, 0.4]], dtype=torch.double, device=device, requires_grad=True
    )
    c0, c1 = (
        X.new_tensor([0.7, 1.3], requires_grad=True),
        X.new_tensor([1.1, 0.8], requires_grad=True),
    )

    class Transform:
        indices = [1]

        @utils.subset_transform
        def apply(self, value):
            return value.square()

    def reject(*args, **kwargs):
        raise AssertionError("unexpected host Tensor conversion")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, reject)
        mu, covariance = (
            X.mean(0),
            X.transpose(-1, -2) @ X + 0.2 * torch.eye(2, device=device, dtype=X.dtype),
        )
        lognormal = utils.norm_to_lognorm(mu, covariance)
        values = [
            *lognormal,
            *utils.lognorm_to_norm(*lognormal),
            utils.norm_to_lognorm_mean(mu, mu.square()),
            utils.norm_to_lognorm_variance(mu, mu.square()),
            utils.expand_and_copy_tensor(X, (3,)),
            Transform().apply(X),
            utils.interaction_features(X),
            utils.nanstd(X, 0),
            utils.kumaraswamy_warp(X, c0, c1, 0.03),
            utils.inv_kumaraswamy_warp(X, c0, c1, 0.03),
        ]
        first = torch.autograd.grad(
            sum(value.sin().sum() for value in values), (X, c0, c1), create_graph=True
        )
        second = torch.autograd.grad(sum(value.square().sum() for value in first), (X, c0, c1))
    assert all(
        value.device == X.device and bool(torch.isfinite(value).all())
        for value in (*values, *first, *second)
    )
