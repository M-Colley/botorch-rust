"""Correlated Kronecker Tensor laws, priors, Matheron samples and derivatives."""

from unittest.mock import patch

import pytest
import torch

from botorch_rs.models import KroneckerMultiTaskGP
from botorch_rs.models._torch_backend_kronecker import tensor_kron
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device, dtype=torch.double, batch=False):
    X = torch.tensor([[0.1], [0.4], [0.75], [0.9]], dtype=dtype, device=device)
    Y = torch.cat(((X * 4).sin(), (X * 3).cos()), -1)
    Q = X.new_tensor([[0.25], [0.6]])
    if batch:
        X = X.unsqueeze(0).expand(2, -1, -1).clone()
        Y = Y.unsqueeze(0).expand(2, -1, -1).clone()
        Y[1] *= 1.3
        Q = Q.unsqueeze(0).expand(2, -1, -1).clone()
    return X, Y, Q


def paired(X, Y, rank=2, likelihood_rank=0, standardize=False):
    from botorch.models import KroneckerMultiTaskGP as Reference
    from botorch.models.transforms.outcome import Standardize as ReferenceStandardize

    a = KroneckerMultiTaskGP(
        X,
        Y,
        rank=rank,
        likelihood_rank=likelihood_rank,
        outcome_transform=Standardize(2) if standardize else None,
        backend="torch",
    )
    b = Reference(
        X,
        Y,
        rank=rank,
        likelihood_rank=likelihood_rank,
        outcome_transform=ReferenceStandardize(2) if standardize else None,
    ).to(X)
    expected = dict(b.named_parameters())
    actual = dict(a.named_parameters())
    assert expected.keys() == actual.keys()
    with torch.no_grad():
        for name, value in actual.items():
            value.copy_(expected[name])
    return a, b


def reference_kernel(model, X, Q):
    if X.ndim == 2:
        return model.covar_module(X, Q).to_dense()
    data = model.covar_module.data_covar_module(X, Q).to_dense()
    task = model.covar_module.task_covar_module.covar_matrix.to_dense()
    return torch.einsum("...ij,...kl->...ikjl", data, task).reshape(
        *data.shape[:-2], data.shape[-2] * 2, data.shape[-1] * 2
    )


def reference_law(model, X, Y, Q):
    # Actual reference covariance modules, with the complete correlated noise.
    noise = model.likelihood._shaped_noise_covar(
        torch.Size((*X.shape[:-2], X.shape[-2], 2))
    ).to_dense()
    ktt = reference_kernel(model, X, X) + noise
    ktq = reference_kernel(model, X, Q)
    kqq = reference_kernel(model, Q, Q)
    mean_train, mean_query = model.mean_module(X), model.mean_module(Q)
    alpha = torch.linalg.solve(ktt, (Y - mean_train).flatten(-2).unsqueeze(-1))
    mean = (mean_query.flatten(-2) + (ktq.transpose(-1, -2) @ alpha).squeeze(-1)).reshape(
        *Q.shape[:-2], Q.shape[-2], 2
    )
    covariance = kqq - ktq.transpose(-1, -2) @ torch.linalg.solve(ktt, ktq)
    q = Q.shape[-2]
    index = torch.arange(q * 2, device=X.device).reshape(q, 2).T.flatten()
    return mean, covariance.index_select(-2, index).index_select(-1, index)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("rank,likelihood_rank", [(1, 0), (2, 1)])
def test_exact_correlated_law_and_data_mll_against_official_modules(
    device, dtype, batch, rank, likelihood_rank
):
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X, Y, Q = data(device, dtype, batch)
    actual, expected = paired(X, Y, rank, likelihood_rank)
    posterior = actual.posterior(Q)
    mean, covariance = reference_law(expected, X, Y, Q)
    tolerance = 2e-4 if dtype == torch.float32 else 2e-8
    torch.testing.assert_close(posterior.mean, mean, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        posterior.distribution.covariance_matrix, covariance, atol=tolerance, rtol=tolerance
    )
    assert posterior.mean.device == X.device
    expected.train()
    # The existing native port fixes duplicated LKJ correlation density. Subtract
    # the official prior aggregate, then add the port's documented joint prior.
    total = Y.shape[-2] * Y.shape[-1]
    reference_prior = sum(
        prior.log_prob(closure(module)).reshape(*X.shape[:-2], -1).sum(-1)
        for _, module, prior, closure, _ in expected.named_priors()
    )
    if batch:
        with pytest.raises(RuntimeError, match="incompatible for a Kronecker product"):
            expected.covar_module(X, X).to_dense()
        noise = expected.likelihood._shaped_noise_covar(
            torch.Size((*X.shape[:-2], X.shape[-2], 2))
        ).to_dense()
        gaussian = torch.distributions.MultivariateNormal(
            expected.mean_module(X).flatten(-2),
            covariance_matrix=reference_kernel(expected, X, X) + noise,
        )
        corrected = (gaussian.log_prob(Y.flatten(-2)) + actual._log_prior()) / total
    else:
        ordinary = ExactMarginalLogLikelihood(expected.likelihood, expected)(expected(X), Y)
        corrected = ordinary - reference_prior / total + actual._log_prior() / total
    torch.testing.assert_close(actual._exact_mll(), corrected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_training_query_parameter_first_and_mixed_second_gradients(device):
    X, Y, Q = (value.requires_grad_() for value in data(device))
    a, b = paired(X, Y, likelihood_rank=1)
    pa = a.posterior(Q)
    mean, covariance = reference_law(b, X, Y, Q)
    names = sorted(dict(a.named_parameters()))
    aa = (X, Y, Q, *(dict(a.named_parameters())[name] for name in names))
    bb = (X, Y, Q, *(dict(b.named_parameters())[name] for name in names))

    def loss(mean, cov):
        return 0.4 * mean.square().sum() + 0.3 * cov.square().sum()

    ga = torch.autograd.grad(
        loss(pa.mean, pa.distribution.covariance_matrix), aa, create_graph=True
    )
    gb = torch.autograd.grad(loss(mean, covariance), bb)
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, atol=5e-8, rtol=5e-8)
    directions = [torch.ones_like(v) * 0.03 for v in aa]
    second = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(ga, directions, strict=True)), aa
    )
    results = []
    step = 2e-5
    for sign in (1, -1):
        with torch.no_grad():
            for value, direction in zip(aa, directions, strict=True):
                value.add_(direction, alpha=sign * step)
        p = a.posterior(Q)
        results.append(torch.autograd.grad(loss(p.mean, p.distribution.covariance_matrix), aa))
        with torch.no_grad():
            for value, direction in zip(aa, directions, strict=True):
                value.add_(direction, alpha=-sign * step)
    for value, positive, negative in zip(second, *results, strict=True):
        torch.testing.assert_close(value, (positive - negative) / (2 * step), atol=2e-6, rtol=3e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("observation_noise", [False, True])
def test_matheron_base_layout_actual_joint_sample_law_and_second_derivatives(
    device, observation_noise
):
    X, Y, Q = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    Q.requires_grad_()
    model, _ = paired(X, Y, likelihood_rank=1)
    posterior = model.posterior(Q, observation_noise=observation_noise)
    assert posterior.base_sample_shape == (20 + (4 if observation_noise else 0),)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host copy")),
    ):
        base = torch.randn(5, *posterior.base_sample_shape, dtype=X.dtype, device=X.device)
        samples = posterior.rsample_from_base_samples((5,), base)
        leaves = (X, Y, Q, *model.parameters())
        first = torch.autograd.grad(samples.square().sum(), leaves, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
        assert all(bool(torch.isfinite(g).all()) for g in second)
    with torch.no_grad():
        draws = posterior.rsample((24000,)).transpose(-1, -2).flatten(-2)
        mean = posterior.mean.T.flatten()
        torch.testing.assert_close(draws.mean(0), mean, atol=0.035, rtol=0.04)
        torch.testing.assert_close(
            torch.cov(draws.T), posterior.distribution.covariance_matrix, atol=0.035, rtol=0.09
        )


@pytest.mark.parametrize("device", DEVICES)
def test_standardize_conditioning_fantasies_and_output_selection(device):
    X, Y, Q = data(device)
    a, b = paired(X, Y, standardize=True)
    pa, pb = a.posterior(Q), b.posterior(Q)
    torch.testing.assert_close(pa.mean, pb.mean, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(pa.variance, pb.variance, atol=2e-8, rtol=2e-8)
    assert pa.rsample((2,)).shape == (2, 2, 2)
    selected = a.posterior(Q, output_indices=[1, 0])
    torch.testing.assert_close(selected.mean, pa.mean.flip(-1))
    Q.requires_grad_()
    fantasy = a.fantasize(Q, IIDNormalSampler((2,), seed=13), propagate_grads=True)
    law = fantasy.posterior(Q)
    first = torch.autograd.grad(law.mean.square().sum() + law.variance.sum(), Q, create_graph=True)[
        0
    ]
    assert bool(torch.isfinite(torch.autograd.grad(first.sum(), Q)[0]).all())
    assert fantasy.batch_shape == (2,)


def test_tensor_kron_is_true_correlated_product():
    a = torch.tensor([[1.0, 0.3], [0.3, 2.0]])
    b = torch.tensor([[2.0, 0.7], [0.7, 3.0]])
    torch.testing.assert_close(tensor_kron(a, b), torch.kron(a, b))


@pytest.mark.parametrize("device", DEVICES)
def test_lkj_density_has_one_correlation_term_and_live_second_derivatives(device):
    X, Y, _ = data(device)
    actual, expected = paired(X, Y)
    kernel = actual.covar_module.task_covar_module
    matrix = kernel.covar_matrix
    standard = matrix.diagonal().sqrt()
    correlation = matrix / standard[:, None] / standard[None, :]
    prior = expected.covar_module.task_covar_module.IndexKernelPrior
    reference = (
        prior.correlation_prior.log_prob(correlation) + prior.sd_prior.log_prob(standard).sum()
    )
    value = kernel.log_prior()
    # The native prior retains its documented double-precision constants rather
    # than GPyTorch's float32 SmoothedBox construction constants.
    torch.testing.assert_close(value, reference, atol=2e-7, rtol=2e-7)
    leaves = (kernel.covar_factor, kernel.raw_var)
    first = torch.autograd.grad(value, leaves, create_graph=True)
    reference_first = torch.autograd.grad(reference, leaves, retain_graph=True)
    for a, b in zip(first, reference_first, strict=True):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)
    assert all(
        bool(torch.isfinite(v).all())
        for v in torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_standalone_official_multitask_kernel_repeats_existing_task_batch(device):
    from gpytorch.kernels import MultitaskKernel, RBFKernel
    from linear_operator.operators import KroneckerProductLinearOperator, to_linear_operator

    X, _, _ = data(device, batch=True)
    kernel = MultitaskKernel(
        RBFKernel(batch_shape=torch.Size([2])), num_tasks=2, rank=1, batch_shape=torch.Size([2])
    ).to(X)
    with pytest.raises(RuntimeError, match="incompatible for a Kronecker product"):
        kernel(X, X).to_dense()
    data_covariance = kernel.data_covar_module(X, X).to_dense()
    task_covariance = kernel.task_covar_module.covar_matrix.to_dense()
    expected = torch.stack([torch.kron(data_covariance[i], task_covariance[i]) for i in range(2)])
    torch.testing.assert_close(tensor_kron(data_covariance, task_covariance), expected)

    def repaired(module, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise NotImplementedError
        task = module.task_covar_module.covar_matrix
        data = to_linear_operator(module.data_covar_module.forward(x1, x2, **params))
        result = KroneckerProductLinearOperator(data, task)
        return result.diagonal(dim1=-2, dim2=-1) if diag else result

    with patch.object(MultitaskKernel, "forward", repaired):
        corrected = kernel(X, X).to_dense()
        torch.testing.assert_close(corrected, expected)
        torch.testing.assert_close(kernel(X, X, diag=True), expected.diagonal(dim1=-2, dim2=-1))


@pytest.mark.parametrize("device", DEVICES)
def test_custom_multitask_likelihood_noise_flags_and_live_parameters(device):
    from gpytorch.likelihoods import MultitaskGaussianLikelihood

    X, Y, Q = data(device)
    likelihood = MultitaskGaussianLikelihood(2, rank=1, has_global_noise=True).to(X)
    model = KroneckerMultiTaskGP(X, Y, likelihood=likelihood, backend="torch")
    assert model.likelihood is likelihood
    posterior = model.posterior(Q, observation_noise=True)
    first = torch.autograd.grad(
        posterior.mean.square().sum() + posterior.variance.sum(),
        likelihood.task_noise_covar_factor,
        create_graph=True,
    )[0]
    assert bool(
        torch.isfinite(
            torch.autograd.grad(first.sum(), likelihood.task_noise_covar_factor)[0]
        ).all()
    )
