"""Exact native component conditioning, reconstruction and query derivatives."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models.additive_gp import OrthogonalAdditiveGP as BtGP  # noqa: E402
from botorch.models.kernels.orthogonal_additive_kernel import (  # noqa: E402
    OrthogonalAdditiveKernel as BtKernel,
)

from botorch_rs.models.additive_gp import OrthogonalAdditiveGP  # noqa: E402
from botorch_rs.models.kernels.orthogonal_additive_kernel import (  # noqa: E402
    OrthogonalAdditiveKernel,
)
from botorch_rs.models.transforms.outcome import Standardize  # noqa: E402

DT = torch.float64


def pair(second_order, batch=()):
    generator = torch.Generator().manual_seed(291)
    x = torch.rand(*batch, 9, 3, generator=generator, dtype=DT)
    y = x[..., :1].sin() + x[..., 1:2].square() - 0.4 * x[..., 2:3]
    actual = OrthogonalAdditiveGP(x, y, second_order=second_order, outcome_transform=None)
    reference = BtGP(x, y, second_order=second_order, outcome_transform=None)
    actual.mean_module.constant = torch.tensor(0.37, dtype=DT)
    reference.mean_module.initialize(constant=torch.tensor(0.37, dtype=DT))
    return actual, reference


@pytest.mark.parametrize("second_order", [False, True])
@pytest.mark.parametrize("batch,query_batch", [((), ()), ((2,), ()), ((2,), (2,)), ((2,), (3, 1))])
@pytest.mark.parametrize("noise", [False, True])
def test_component_conditioning_matches_official(second_order, batch, query_batch, noise):
    actual, reference = pair(second_order, batch)
    query = torch.linspace(
        0.18, 0.83, max(torch.Size(query_batch).numel(), 1) * 6, dtype=DT
    ).reshape(*query_batch, 2, 3)
    ap = actual.posterior(query, infer_all_components=True, observation_noise=noise)
    if not batch:
        bp = reference.posterior(query, infer_all_components=True, observation_noise=noise)
        expected_mean, expected_covariance = bp.mean, bp.distribution.covariance_matrix
    else:
        # Official component inference fails to broadcast the constant bias.
        # Validate the conditioning equations using its actual full training
        # kernel and independently evaluated orthogonal one-dimensional terms.
        bx = reference.train_inputs[0]
        common = torch.broadcast_shapes(query.shape[:-2], bx.shape[:-2])
        q = query.expand(*common, *query.shape[-2:])
        tx = bx.expand(*common, *bx.shape[-2:])
        kernel = reference.covar_module

        def parts(a, b):
            bases = kernel._orthogonal_base_kernels(a, b)
            bias = kernel.offset.expand(*common, a.shape[-2], b.shape[-2]).unsqueeze(-3)
            result = [bias, kernel.coeffs_1[..., None, None] * bases]
            if second_order:
                result.extend(
                    (kernel.coeffs_2[i, j] * bases[..., i, :, :] * bases[..., j, :, :]).unsqueeze(
                        -3
                    )
                    for i in range(3)
                    for j in range(i + 1, 3)
                )
            return torch.cat(result, dim=-3).movedim(-3, 0)

        full = (
            kernel(tx, tx).to_dense()
            + torch.eye(tx.shape[-2], dtype=DT) * reference.likelihood.noise[..., None]
        )
        cross, prior = parts(q, tx), parts(q, q)
        residual = reference.train_targets.expand(*common, tx.shape[-2]) - reference.mean_module(tx)
        alpha = torch.linalg.solve(full, residual.unsqueeze(-1))
        expected_mean = cross @ alpha
        expected_mean[0] += reference.mean_module(q).unsqueeze(-1)
        expected_covariance = prior - cross @ torch.linalg.solve(full, cross.transpose(-1, -2))
        if noise:
            expected_covariance += (
                torch.eye(q.shape[-2], dtype=DT) * reference.likelihood.noise[..., None]
            )
    torch.testing.assert_close(ap.mean, expected_mean, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(
        ap.distribution.covariance_matrix, expected_covariance, rtol=2e-8, atol=2e-10
    )
    torch.testing.assert_close(ap.mean.sum(0), actual.posterior(query).mean, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("mode", ["mean", "variance", "covariance", "samples"])
@pytest.mark.parametrize("second_order", [False, True])
def test_component_query_gradients_match_official(second_order, mode):
    actual, reference = pair(second_order)
    query = torch.tensor([[0.27, 0.64, 0.39], [0.76, 0.43, 0.61]], dtype=DT, requires_grad=True)
    other = query.detach().clone().requires_grad_()
    ap, bp = (
        actual.posterior(query, infer_all_components=True),
        reference.posterior(other, infer_all_components=True),
    )
    if mode == "covariance":
        a, b = ap.distribution.covariance_matrix, bp.distribution.covariance_matrix
    elif mode == "samples":
        z = torch.linspace(
            -0.8, 0.7, 3 * torch.Size(ap.base_sample_shape).numel(), dtype=DT
        ).reshape(3, *ap.base_sample_shape)
        a = ap.rsample_from_base_samples((3,), z)
        b = bp.rsample_from_base_samples(torch.Size([3]), z)
    else:
        a, b = getattr(ap, mode), getattr(bp, mode)
    torch.testing.assert_close(a, b, rtol=2e-7, atol=2e-8)
    weights = torch.linspace(0.2, 0.9, a.numel(), dtype=DT).reshape(a.shape)
    ga = torch.autograd.grad((a * weights).sum(), query)[0]
    gb = torch.autograd.grad((b * weights).sum(), other)[0]
    torch.testing.assert_close(ga, gb, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("second_order", [False, True])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_base_quadrature_and_component_covariances(second_order, batch):
    kwargs = dict(
        dim=3, batch_shape=torch.Size(batch), second_order=second_order, quad_deg=12, dtype=DT
    )
    actual, reference = OrthogonalAdditiveKernel(**kwargs), BtKernel(**kwargs)
    x = torch.tensor([[0.15, 0.63, 0.82], [0.75, 0.42, 0.26]], dtype=DT)
    y = 0.9 - 0.6 * x
    for diag in (False, True):
        torch.testing.assert_close(
            actual.k(x, y, diag), reference.k(x, y, diag), rtol=2e-12, atol=2e-12
        )
        right = x if diag else y
        values = actual._non_reduced_forward(x, right, diag)
        torch.testing.assert_close(
            values, reference._non_reduced_forward(x, right, diag), rtol=2e-11, atol=2e-12
        )
        torch.testing.assert_close(
            values.sum(-2 if diag else -3), actual(x, right, diag=diag), rtol=2e-11, atol=2e-12
        )
    torch.testing.assert_close(actual.normalizer(), reference.normalizer(), rtol=2e-12, atol=2e-12)
    # A parameter mutation must invalidate any normalization state.
    actual.base_kernel.lengthscale = torch.tensor(0.21, dtype=DT)
    reference.base_kernel.lengthscale = torch.tensor(0.21, dtype=DT)
    torch.testing.assert_close(
        actual.normalizer(0.3), reference.normalizer(0.3), rtol=2e-12, atol=2e-12
    )
    for name, indices in actual.component_indices.items():
        torch.testing.assert_close(indices, reference.component_indices[name])


@pytest.mark.parametrize("batch", [(), (2,)])
def test_grid_statistics_and_snapshot(batch):
    actual, reference = pair(True, batch)
    grid = torch.linspace(0.1, 0.9, 7, dtype=DT)
    a = actual.evaluate_first_order_on_grid(grid)
    if batch:
        parts = []
        for index in range(batch[0]):
            leaf = BtGP(
                reference.train_inputs[0][index],
                reference.train_targets[index, :, None],
                second_order=True,
                outcome_transform=None,
            )
            leaf.mean_module.initialize(constant=torch.tensor(0.37, dtype=DT))
            parts.append(leaf.evaluate_first_order_on_grid(grid))
        b = (
            tuple(torch.stack([item[0][k] for item in parts], dim=0) for k in range(2)),
            tuple(torch.stack([item[1][k] for item in parts], dim=1) for k in range(2)),
        )
    else:
        b = reference.evaluate_first_order_on_grid(grid)
    for ag, bg in zip(a, b, strict=True):
        for av, bv in zip(ag, bg, strict=True):
            torch.testing.assert_close(av, bv, rtol=3e-9, atol=3e-10)
    post = actual.posterior(grid[:, None].expand(7, 3), infer_all_components=True)
    old = post.mean.clone()
    actual.mean_module.constant = 5.0
    torch.testing.assert_close(post.mean, old)


def test_standardized_components_reconstruct_original_units():
    actual, _ = pair(True)
    x = actual.train_inputs[0]
    # Use large nonzero outcome location to catch repeated untransform offsets.
    targets = (x[..., :1].sin() + x[..., 1:2].square() - 0.4 * x[..., 2:3]) * 2.5 + 7.0
    model = OrthogonalAdditiveGP(x, targets, second_order=True, outcome_transform=Standardize(1))
    query = torch.tensor([[0.27, 0.64, 0.39], [0.76, 0.43, 0.61]], dtype=DT)
    post = model.posterior(query, infer_all_components=True)
    torch.testing.assert_close(
        post.mean.sum(0), model.posterior(query).mean, rtol=1e-10, atol=1e-10
    )
    reference = BtGP(x, targets, second_order=True)
    other = reference.posterior(query, infer_all_components=True)
    means = other.mean.clone()
    if not torch.allclose(means.sum(0), reference.posterior(query).mean):
        # b0dd01be adds Standardize's outcome location to every component.
        # Only the bias may contain the location of the full process.
        means[1:] -= reference.outcome_transform.means
    torch.testing.assert_close(post.mean, means, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(
        post.distribution.covariance_matrix,
        other.distribution.covariance_matrix,
        rtol=2e-8,
        atol=2e-10,
    )


def test_component_index_validation():
    kernel = OrthogonalAdditiveKernel(3, second_order=True)
    assert kernel.get_component_index("second_order", (1, 2)) == 6
    with pytest.raises(ValueError):
        kernel.get_component_index("second_order", (2, 1))
    with pytest.raises(IndexError):
        kernel.get_component_index("first_order", 3)
