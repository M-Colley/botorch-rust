# ruff: noqa: E402
"""True masked feature/time Tensor laws and full correlated sampling graphs."""

import pytest

pytest.importorskip("torch")


from unittest.mock import patch

import torch

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.means import ConstantMean
from botorch_rs.models import LatentKroneckerGP
from botorch_rs.models._torch_backend_kronecker import tensor_kron
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device, dtype=torch.double, batch=False, missing=False):
    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=dtype, device=device)
    T = X.new_tensor([[0.0], [0.5], [1.0]])
    Y = (X * 4 + T.squeeze(-1) * 2).sin() + 0.7
    Q, U = X.new_tensor([[0.25], [0.7]]), X.new_tensor([[0.2], [0.8]])
    if missing:
        Y[1, 1] = float("nan")
        Y[2, 0] = float("inf")
    if batch:
        X, Y = X.expand(2, -1, -1).clone(), Y.expand(2, -1, -1).clone()
        Y[1] = 0.2 + 1.3 * Y[1]
    return X, T, Y, Q, U


def paired(X, T, Y, standardize=False):
    from botorch.models.latent_kronecker_gp import LatentKroneckerGP as Reference

    kwargs = {} if standardize else {"outcome_transform": None}
    a = LatentKroneckerGP(X, T, Y, backend="torch", **kwargs)
    b = Reference(X, T, Y, **kwargs).to(X)
    actual, expected = dict(a.named_parameters()), dict(b.named_parameters())
    assert actual.keys() == expected.keys()
    with torch.no_grad():
        for name in actual:
            actual[name].copy_(expected[name])
    return a, b


def law(model, X, T, Q, U):
    kernel_x, kernel_t = model.covar_module_X, model.covar_module_T
    kt = kernel_t(T, T).to_dense()
    full = tensor_kron(kernel_x(X, X).to_dense(), kt)
    cross = tensor_kron(kernel_x(X, Q).to_dense(), kernel_t(T, U).to_dense())[
        ..., model.mask_valid, :
    ]
    test = tensor_kron(kernel_x(Q, Q).to_dense(), kernel_t(U, U).to_dense())
    covariance = full[..., model.mask_valid, :][..., :, model.mask_valid]
    covariance = covariance + torch.diag_embed(
        model.likelihood.noise.expand_as(model.train_targets)
    )
    mean_train = model._get_mean(X, T, model.mask_valid)
    mean_test = model._get_mean(Q, U)
    alpha = torch.linalg.solve(covariance, (model.train_targets - mean_train).unsqueeze(-1))
    mean = mean_test + (cross.transpose(-1, -2) @ alpha).squeeze(-1)
    posterior = test - cross.transpose(-1, -2) @ torch.linalg.solve(covariance, cross)
    if hasattr(model, "outcome_transform"):
        mean = mean * model.outcome_transform.stdvs.squeeze(
            -1
        ) + model.outcome_transform.means.squeeze(-1)
        posterior = posterior * model.outcome_transform.stdvs.square()
    return mean, posterior, covariance, mean_train


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_actual_official_masked_component_law_and_normalized_mll(
    device, dtype, batch, missing, standardize
):
    X, T, Y, Q, U = data(device, dtype, batch, missing)
    a, b = paired(X, T, Y, standardize)
    p = a.posterior(Q, U)
    mean, covariance, train, mean_train = law(b, X, T, Q, U)
    tol = 3e-5 if dtype == torch.float32 else 2e-8
    torch.testing.assert_close(p.mean.flatten(-2), mean, atol=tol, rtol=tol)
    torch.testing.assert_close(p.distribution.covariance_matrix, covariance, atol=tol, rtol=tol)
    torch.testing.assert_close(
        p.variance.flatten(-2), covariance.diagonal(dim1=-2, dim2=-1), atol=tol, rtol=tol
    )
    target = b.train_targets
    expected = (
        torch.distributions.MultivariateNormal(mean_train, covariance_matrix=train).log_prob(target)
        / target.shape[-1]
    )
    torch.testing.assert_close(a._exact_mll(), expected, atol=tol, rtol=tol)
    assert p.mean.device == X.device
    assert a.train_targets.shape[-1] == (7 if missing else 9)
    assert p.base_sample_shape == X.shape[:-2] + (9 + target.shape[-1] + 4,)


@pytest.mark.parametrize("device", DEVICES)
def test_all_training_time_query_target_and_parameter_first_gradients_and_hvp(device):
    X, T, Y, Q, U = [v.requires_grad_() for v in data(device, missing=True)]
    a, b = paired(X, T, Y)
    p = a.posterior(Q, U)
    mean, covariance, _, _ = law(b, X, T, Q, U)
    names = sorted(dict(a.named_parameters()))
    aa = (X, T, Y, Q, U, *(dict(a.named_parameters())[n] for n in names))
    bb = (X, T, Y, Q, U, *(dict(b.named_parameters())[n] for n in names))

    def loss(mean, cov):
        return 0.4 * mean.square().sum() + 0.3 * cov.square().sum()

    ga = torch.autograd.grad(
        loss(p.mean.flatten(-2), p.distribution.covariance_matrix), aa, create_graph=True
    )
    gb = torch.autograd.grad(loss(mean, covariance), bb)
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-7)
    assert ga[2][1, 1] == 0 and ga[2][2, 0] == 0
    directions = [torch.full_like(v, 0.02) for v in aa]
    # Missing entries are metadata; no perturbation crosses the fixed mask.
    directions[2][~torch.isfinite(Y)] = 0
    second = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(ga, directions, strict=True)), aa
    )
    step, results = 1e-5, []
    for sign in (1, -1):
        with torch.no_grad():
            for value, direction in zip(aa, directions, strict=True):
                value.add_(direction, alpha=sign * step)
        q = a.posterior(Q, U)
        results.append(
            torch.autograd.grad(loss(q.mean.flatten(-2), q.distribution.covariance_matrix), aa)
        )
        with torch.no_grad():
            for value, direction in zip(aa, directions, strict=True):
                value.add_(direction, alpha=-sign * step)
    for value, positive, negative in zip(second, *results, strict=True):
        torch.testing.assert_close(value, (positive - negative) / (2 * step), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("missing", [False, True])
def test_matheron_true_joint_covariance_and_first_second_derivatives(device, missing):
    X, T, Y, Q, U = [v.requires_grad_() for v in data(device, missing=missing)]
    model = LatentKroneckerGP(X, T, Y, backend="torch")
    p = model.posterior(Q, U)
    normal = torch.randn(2, 3, *p.base_sample_shape, dtype=X.dtype, device=X.device)
    inputs = (X, T, Y, Q, U, *model.parameters())
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host copy")),
    ):
        sample = p.rsample_from_base_samples(torch.Size([2, 3]), normal)
        first = torch.autograd.grad(sample.square().sum(), inputs, create_graph=True)
        second = torch.autograd.grad(sum(v.square().sum() for v in first), inputs)
    assert sample.shape == (2, 3, 2, 2)
    assert all(torch.isfinite(v).all() for v in (*first, *second))
    with torch.no_grad():
        values = p.rsample(torch.Size([16000])).flatten(-2)
        torch.testing.assert_close(values.mean(0), p.mean.flatten(-2), atol=0.035, rtol=0.06)
        centered = values - values.mean(0)
        torch.testing.assert_close(
            centered.T @ centered / (values.shape[0] - 1),
            p.distribution.covariance_matrix,
            atol=0.035,
            rtol=0.09,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_model_query_time_batches_nonzero_product_means_and_custom_modules(device):
    from gpytorch.kernels import RBFKernel

    X, T, Y, Q, U = data(device, batch=True, missing=True)
    mx, mt = ConstantMean(batch_shape=(2,)), ConstantMean(batch_shape=(2,))
    mx.constant = torch.tensor([0.7, 0.9])
    mt.constant = torch.tensor([0.3, 0.6])
    kernel = RBFKernel(ard_num_dims=1).to(X)
    a = LatentKroneckerGP(
        X,
        T,
        Y,
        backend="torch",
        mean_module_X=mx,
        mean_module_T=mt,
        covar_module_X=kernel,
        outcome_transform=None,
    )
    assert a.covar_module_X is kernel
    Q = Q.expand(3, 1, -1, -1).clone().requires_grad_()
    U = U.expand(3, 1, -1, -1).clone().requires_grad_()
    p = a.posterior(Q, U)
    assert p.mean.shape == (3, 2, 2, 2)
    assert p.rsample(torch.Size([2, 3])).shape == (2, 3, 3, 2, 2, 2)
    gradient = torch.autograd.grad(
        p.mean.square().sum(), (Q, U, *a.parameters()), create_graph=True
    )
    assert all(torch.isfinite(v).all() for v in gradient)
    a._exact_mll().sum().backward()
    assert a.mean_module_X.raw_constant.grad is not None


@pytest.mark.parametrize("device", DEVICES)
def test_conditioning_and_deterministic_sampler_fantasies_preserve_stats(device):
    X, T, Y, Q, U = data(device, missing=True)
    a = LatentKroneckerGP(X, T, Y, backend="torch")
    initial = a.posterior(Q)
    values = initial.mean.detach().expand(3, -1, -1).clone()
    values[:, 0, 1] = float("nan")
    updated = a.condition_on_observations(Q, values)
    assert updated.posterior(Q).mean.shape == (3, 2, 3)
    assert updated.mask_valid.sum() == 12
    torch.testing.assert_close(updated.outcome_transform.means, a.outcome_transform.means)
    assert a._train_Y.shape == (3, 3)
    Q.requires_grad_()
    sampler = IIDNormalSampler(torch.Size([2]), seed=123)
    fa = a.fantasize(Q, sampler, propagate_grads=True)
    fb = a.fantasize(Q, sampler, propagate_grads=True)
    torch.testing.assert_close(fa._train_Y, fb._train_Y, equal_nan=True)
    first = torch.autograd.grad(fa.posterior(Q).mean.square().sum(), Q, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), Q)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_masks_shape_changes_and_explicit_boundaries(device):
    X, T, Y, Q, U = data(device, batch=True, missing=True)
    invalid = Y.clone()
    invalid[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Pattern"):
        LatentKroneckerGP(X, T, invalid, backend="torch")
    with pytest.raises(ValueError, match="finite observation"):
        LatentKroneckerGP(X, T, torch.full_like(Y, float("nan")), backend="torch")
    a = LatentKroneckerGP(X, T, Y, backend="torch")
    with pytest.raises(UnsupportedError, match="iterative"):
        a.use_iterative_methods()
    with pytest.raises(UnsupportedError, match="latent outputs"):
        a.posterior(Q, observation_noise=True)
    a._train_Y[0, 0, 0] = float("nan")
    a._train_Y[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="pattern changed"):
        a.posterior(Q)


@pytest.mark.parametrize("device", DEVICES)
def test_exact_mll_first_and_second_training_gradients_and_public_fitter(device):
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    X, T, Y, Q, U = [value.requires_grad_() for value in data(device, missing=True)]
    a, b = paired(X, T, Y, standardize=True)
    _, _, covariance, mean = law(b, X, T, Q, U)
    expected = (
        torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance).log_prob(
            b.train_targets
        )
        / b.train_targets.shape[-1]
    )
    names = sorted(dict(a.named_parameters()))
    aa = (X, T, Y, *(dict(a.named_parameters())[name] for name in names))
    bb = (X, T, Y, *(dict(b.named_parameters())[name] for name in names))
    first = torch.autograd.grad(a._exact_mll(), aa, create_graph=True)
    reference = torch.autograd.grad(expected, bb)
    for left, right in zip(first, reference, strict=True):
        torch.testing.assert_close(left, right, atol=2e-8, rtol=2e-8)
    second = torch.autograd.grad(sum(value.square().sum() for value in first), aa)
    assert all(torch.isfinite(value).all() for value in second)
    mll = ExactMarginalLogLikelihood(a.likelihood, a)
    a.train()
    torch.testing.assert_close(mll(a(*a.train_inputs), a.train_targets), mll())
    before = {name: value.detach().clone() for name, value in a.named_parameters()}
    fitted = fit_gpytorch_mll(
        mll,
        optimizer=fit_gpytorch_mll_torch,
        optimizer_kwargs={"step_limit": 3, "stopping_criterion": None},
    )
    assert fitted.model is a
    assert any(not torch.equal(value, before[name]) for name, value in a.named_parameters())
    assert torch.isfinite(a.posterior(Q, U).mean).all()


@pytest.mark.parametrize("device", DEVICES)
def test_learned_input_transform_preserves_time_axis_and_full_hessian(device):
    from botorch_rs.models.transforms.input import ChainedInputTransform, Normalize, Warp

    X, T, Y, Q, U = [value.requires_grad_() for value in data(device, missing=True)]
    transform = ChainedInputTransform(
        normalize=Normalize(1, bounds=X.new_tensor([[0.0], [1.0]])), warp=Warp(d=1, indices=[0])
    )
    model = LatentKroneckerGP(X, T, Y, backend="torch", input_transform=transform)
    assert model.transform_inputs(model.train_T) is model.train_T
    inputs = (X, T, Y, Q, U, *model.parameters())
    p = model.posterior(Q, U)
    first = torch.autograd.grad(
        p.mean.square().sum() + model._exact_mll(), inputs, create_graph=True
    )
    second = torch.autograd.grad(sum(value.square().sum() for value in first), inputs)
    assert all(torch.isfinite(value).all() for value in (*first, *second))
