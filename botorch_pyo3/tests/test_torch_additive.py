"""Orthogonal additive Tensor kernels, component laws and higher derivatives."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models.additive_gp import OrthogonalAdditiveGP as ReferenceGP
from botorch.models.kernels.orthogonal_additive_kernel import (
    OrthogonalAdditiveKernel as ReferenceKernel,
)

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import MaternKernel, RBFKernel
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.additive_gp import OrthogonalAdditiveGP
from botorch_rs.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from botorch_rs.models.transforms.outcome import Log
from botorch_rs.optim.utils.model_utils import sample_all_priors
from botorch_rs.priors import GammaPrior

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _copy(reference, port, index=None):
    source = dict(port.named_parameters())
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            value = source[name]
            parameter.copy_(value if index is None else value[index])


def _manual_components(kernel, x, z):
    values = []
    for i in range(kernel.dim):
        lengthscale = kernel.base_kernel.lengthscale
        lengthscale = lengthscale[..., i, 0, 0] if kernel._per_dim else lengthscale[..., 0, 0]

        def covariance(a, b, lengthscale=lengthscale):
            return (
                -0.5 * ((a.unsqueeze(-1) - b.unsqueeze(-2)) / lengthscale[..., None, None]).square()
            ).exp()

        nodes, weights = kernel.z[:, i], kernel.w[:, 0]
        integral = torch.einsum(
            "i,...ij,j->...", weights, covariance(nodes, nodes), weights
        ).clamp_min(1e-6)
        left = covariance(x[..., i], nodes) @ weights
        right = covariance(z[..., i], nodes) @ weights
        values.append(
            covariance(x[..., i], z[..., i])
            - left.unsqueeze(-1) * right.unsqueeze(-2) / integral[..., None, None]
        )
    ortho = torch.stack(values, -3)
    batch = ortho.shape[:-3]
    bias = kernel.offset[..., None, None, None].expand(*batch, 1, x.shape[-2], z.shape[-2])
    first = kernel.coeffs_1[..., None, None] * ortho
    parts = [bias, first]
    if kernel.second_order:
        pairs = torch.triu_indices(kernel.dim, kernel.dim, 1, device=x.device)
        interactions = ortho.index_select(-3, pairs[0]) * ortho.index_select(-3, pairs[1])
        parts.append(
            kernel.coeff_constraint.transform(kernel.raw_coeffs_2)[..., None, None] * interactions
        )
    return torch.cat(parts, -3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "batch,per_dim,second,nu",
    [
        ((), True, False, None),
        ((2,), True, True, None),
        ((2,), False, True, None),
        ((), True, True, 1.5),
        ((), False, False, 2.5),
        ((2,), False, False, 2.5),
    ],
)
def test_kernel_law_components_priors_and_higher_graphs(device, batch, per_dim, second, nu):
    torch.manual_seed(18)
    base_batch = batch + ((3,) if per_dim else ())
    base = (
        RBFKernel(batch_shape=base_batch)
        if nu is None
        else MaternKernel(nu=nu, batch_shape=base_batch)
    )
    kernel = OrthogonalAdditiveKernel(
        3,
        base,
        per_dim,
        quad_deg=10,
        second_order=second,
        batch_shape=batch,
        dtype=torch.double,
        backend="torch",
    ).to(device)
    x = torch.rand(*batch, 3, 3, dtype=torch.double, device=device, requires_grad=True)
    z = torch.rand(
        *((1,) if batch else ()), 4, 3, dtype=torch.double, device=device, requires_grad=True
    )
    value = kernel(x, z)
    components = kernel._non_reduced_forward(x, z)
    torch.testing.assert_close(components.sum(-3), value)
    torch.testing.assert_close(kernel(x, x, diag=True), kernel(x, x).diagonal(dim1=-2, dim2=-1))
    if nu is None:
        manual = _manual_components(kernel, x, z)
        torch.testing.assert_close(components, manual, atol=3e-14, rtol=3e-12)
        leaves = (x, z, *kernel.parameters())
        first = torch.autograd.grad(
            components.square().sum(), leaves, create_graph=True, retain_graph=True
        )
        other = torch.autograd.grad(
            manual.square().sum(), leaves, create_graph=True, retain_graph=True
        )
        for a, b in zip(first, other, strict=True):
            torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-10)
        second_grad = torch.autograd.grad(sum(g.sum() for g in first), leaves, retain_graph=True)
        manual_second = torch.autograd.grad(sum(g.sum() for g in other), leaves)
        for a, b in zip(second_grad, manual_second, strict=True):
            torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)
    else:
        first = torch.autograd.grad(value.square().sum(), x, create_graph=True)[0]
        assert torch.isfinite(torch.autograd.grad(first.sum(), x)[0]).all()
    # The official shared-lengthscale model-batch layout does not insert a
    # feature batch axis. Independent unbatched kernels define its corrected
    # per-batch law without changing the formula or tolerances.
    oracle_batches = range(batch[0]) if batch and not per_dim else [None]
    for index in oracle_batches:
        ref_batch = torch.Size() if index is not None else torch.Size(batch)
        rb = ref_batch + ((3,) if per_dim else ())
        ref_base = (
            gpytorch.kernels.RBFKernel(batch_shape=rb)
            if nu is None
            else gpytorch.kernels.MaternKernel(nu=nu, batch_shape=rb)
        )
        reference = ReferenceKernel(
            3,
            ref_base,
            per_dim,
            quad_deg=10,
            second_order=second,
            batch_shape=ref_batch,
            dtype=torch.double,
            device=device,
        )
        _copy(reference, kernel, index)
        a, b = (x, z) if index is None else (x[index], z[0])
        expected = reference(a, b).to_dense()
        actual = value if index is None else value[index]
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-11)
    kernel.eval()
    for _ in range(2):
        g = torch.autograd.grad(
            kernel.normalizer().sum(), kernel.base_kernel.raw_lengthscale, create_graph=True
        )[0]
        assert torch.isfinite(
            torch.autograd.grad(g.sum(), kernel.base_kernel.raw_lengthscale)[0]
        ).all()
    with pytest.raises(UnsupportedError, match="last_dim_is_batch"):
        kernel(x, last_dim_is_batch=True)
    with pytest.raises(ValueError, match="hypercube"):
        kernel(x + 2)


def _component_oracle(model, query):
    train, y, fixed = model._training_values()
    kernel = model.covar_module
    covariance = _manual_components(kernel, train, train).sum(-3)
    noise = model._training_noise(y, fixed)
    factor = torch.linalg.cholesky(covariance + noise.diag_embed())
    alpha = torch.cholesky_solve((y - model.mean_module(train)).unsqueeze(-1), factor)
    cross = _manual_components(kernel, train, query).movedim(-3, 0)
    prior = _manual_components(kernel, query, query).movedim(-3, 0)
    mean = (cross.transpose(-1, -2) @ alpha).squeeze(-1)
    bias = torch.zeros_like(mean)
    bias[0] = model.mean_module(query)
    mean = mean + bias
    posterior_covariance = prior - cross.transpose(-1, -2) @ torch.cholesky_solve(cross, factor)
    return mean, posterior_covariance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("second", [False, True])
def test_component_posterior_all_data_parameter_hessians_and_training(device, batch, second):
    torch.manual_seed(114)
    x = torch.rand(*batch, 6, 3, dtype=torch.double, device=device, requires_grad=True)
    y = torch.randn(*batch, 6, 1, dtype=x.dtype, device=device, requires_grad=True)
    model = OrthogonalAdditiveGP(x, y, second_order=second, outcome_transform=None, backend="torch")
    query = torch.rand(*batch, 2, 3, dtype=x.dtype, device=device, requires_grad=True)
    leaves = (x, y, query, *model.parameters())
    for _ in range(2):
        posterior = model.posterior(query, infer_all_components=True)
        expected_mean, expected_cov = _component_oracle(model, query)
        torch.testing.assert_close(
            posterior.mean.squeeze(-1), expected_mean, atol=3e-12, rtol=3e-10
        )
        torch.testing.assert_close(
            posterior.distribution.covariance_matrix, expected_cov, atol=3e-12, rtol=3e-10
        )
        value = (
            posterior.mean.square().sum() + posterior.distribution.covariance_matrix.square().sum()
        )
        expected = expected_mean.square().sum() + expected_cov.square().sum()
        first = torch.autograd.grad(value, leaves, create_graph=True, retain_graph=True)
        ref_first = torch.autograd.grad(expected, leaves, create_graph=True, retain_graph=True)
        for a, b in zip(first, ref_first, strict=True):
            torch.testing.assert_close(a, b, atol=5e-10, rtol=3e-8)
        second_grad = torch.autograd.grad(sum(g.sum() for g in first), leaves, retain_graph=True)
        ref_second = torch.autograd.grad(sum(g.sum() for g in ref_first), leaves)
        for a, b in zip(second_grad, ref_second, strict=True):
            torch.testing.assert_close(a, b, atol=1e-8, rtol=2e-7)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(mll.parameters(), lr=0.002)
    before = model.covar_module.raw_coeffs_1.detach().clone()
    for _ in range(2):
        model.train()
        optimizer.zero_grad()
        (-mll(model(*model.train_inputs), model.train_targets).sum()).backward()
        optimizer.step()
    assert not torch.equal(before, model.covar_module.raw_coeffs_1)
    grid = torch.linspace(0.05, 0.95, 5, dtype=x.dtype, device=device, requires_grad=True)
    bias, first_order = model.evaluate_first_order_on_grid(grid)
    assert first_order[0].shape == (3, *batch, 5)
    assert torch.isfinite(torch.autograd.grad(first_order[0].sum(), grid)[0]).all()
    assert bias[0].shape == batch


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
def test_original_units_component_sum_samples_and_observation_noise(device, batch):
    torch.manual_seed(44)
    x = torch.rand(*batch, 6, 3, dtype=torch.double, device=device, requires_grad=True)
    y = (5 + torch.randn(*batch, 6, 1, dtype=x.dtype, device=device)).requires_grad_()
    model = OrthogonalAdditiveGP(x, y, second_order=True, backend="torch")
    query = torch.rand(*batch, 2, 3, dtype=x.dtype, device=device, requires_grad=True)
    full = model.posterior(query)
    parts = model.posterior(query, infer_all_components=True)
    torch.testing.assert_close(parts.mean.sum(0), full.mean, atol=1e-11, rtol=2e-10)
    noisy = model.posterior(query, infer_all_components=True, observation_noise=True)
    added = noisy.variance - parts.variance
    expected_noise = model.likelihood.noise * model.outcome_transform.stdvs.squeeze(-1).square()
    torch.testing.assert_close(
        added, expected_noise.reshape(1, *batch, 1, 1).expand_as(added), atol=2e-12, rtol=2e-10
    )
    samples = parts.rsample(torch.Size([4]))
    gradient = torch.autograd.grad(samples.square().sum(), (x, y, query), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradient), query)[0]
    assert torch.isfinite(second).all()
    assert model.get_component_index("second_order", (0, 2)) == 5
    if not batch:
        # Source component inference is used directly in its supported
        # unbatched/untransformed configuration (its location defect is fixed
        # by the original-unit sum regression above).
        reference = ReferenceGP(
            x.detach(), y.detach(), second_order=True, outcome_transform=None
        ).to(x)
        _copy(reference, model)
        source = reference.posterior(query, infer_all_components=True)
        untransformed = OrthogonalAdditiveGP(
            x, y, second_order=True, outcome_transform=None, backend="torch"
        )
        untransformed.load_state_dict(
            {
                name: value
                for name, value in model.state_dict().items()
                if not name.startswith("outcome_transform.")
            }
        )
        actual = untransformed.posterior(query, infer_all_components=True)
        torch.testing.assert_close(actual.mean, source.mean, atol=1e-10, rtol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_positive_packed_prior_migration_and_nonlinear_boundary(device):
    kernel = (
        OrthogonalAdditiveKernel(
            3, second_order=True, quad_deg=8, coeffs_2_prior=GammaPrior(2, 3), backend="torch"
        )
        .double()
        .to(device)
    )
    value = kernel.log_prior()
    assert value.isfinite()
    gradient = torch.autograd.grad(value, kernel.raw_coeffs_2, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), kernel.raw_coeffs_2)[0]).all()
    assert all(buffer.device.type == device for buffer in kernel.buffers())
    x = torch.rand(5, 3, dtype=torch.float32, device=device)
    y = torch.rand(5, 1, dtype=x.dtype, device=device) + 0.3
    model = OrthogonalAdditiveGP(x, y, outcome_transform=Log(), backend="torch")
    assert torch.isfinite(model.posterior(x[:2]).mean).all()
    with pytest.raises(UnsupportedError, match="affine"):
        model.posterior(x[:2], infer_all_components=True)


@pytest.mark.parametrize("device", DEVICES)
def test_shared_kernel_query_broadcast_coeff_setters_and_prior_sampling(device):
    torch.manual_seed(14)
    kernel = OrthogonalAdditiveKernel(
        3, second_order=True, dtype=torch.double, coeffs_2_prior=GammaPrior(2, 3), backend="torch"
    ).to(device)
    held = kernel.raw_coeffs_2
    matrix = torch.ones(3, 3, dtype=torch.double, device=device) * 0.2
    kernel.coeffs_2 = matrix
    torch.testing.assert_close(kernel.coeffs_2.triu(1), matrix.triu(1))
    assert kernel.raw_coeffs_2 is held
    x = torch.rand(2, 5, 3, dtype=torch.double, device=device, requires_grad=True)
    y = torch.randn(2, 5, 1, dtype=x.dtype, device=device, requires_grad=True)
    model = OrthogonalAdditiveGP(x, y, kernel, backend="torch")
    before = held.detach().clone()
    sample_all_priors(model)
    assert not torch.equal(before, held)
    query = torch.rand(3, 1, 2, 3, dtype=x.dtype, device=device, requires_grad=True)
    posterior = model.posterior(query, infer_all_components=True)
    assert posterior.mean.shape == (7, 3, 2, 2, 1)
    torch.testing.assert_close(
        posterior.mean.sum(0), model.posterior(query).mean, atol=2e-11, rtol=3e-10
    )
    first = torch.autograd.grad(posterior.mean.sum(), (query, x, y, held), create_graph=True)
    assert torch.isfinite(
        torch.autograd.grad(sum(value.square().sum() for value in first), query)[0]
    ).all()
    assert kernel.component_indices["second_order"].device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_independent_multioutput_full_posterior(device):
    x = torch.rand(5, 3, device=device)
    y = torch.rand(5, 2, device=device)
    model = OrthogonalAdditiveGP(x, y, second_order=True, backend="torch")
    query = torch.rand(2, 3, device=device, requires_grad=True)
    posterior = model.posterior(query)
    assert posterior.mean.shape == (2, 2)
    first = torch.autograd.grad(posterior.mean.square().sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(first.sum(), query)[0]).all()
    with pytest.raises(UnsupportedError, match="one model output"):
        model.posterior(query, infer_all_components=True)
