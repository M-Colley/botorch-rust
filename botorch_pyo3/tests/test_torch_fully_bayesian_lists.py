# ruff: noqa: E402
"""Fully Bayesian lists retain mixture axes, component graphs and transforms."""

import pytest

pytest.importorskip("torch")


import warnings

import torch

from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch_rs.models.transforms.outcome import Log
from botorch_rs.posteriors._torch_backend import TorchPosteriorList
from botorch_rs.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def pair(device, offset=0, count=3, reference=False):
    if reference:
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP as ReferenceGP

        cls = ReferenceGP
    else:
        cls = SaasFullyBayesianSingleTaskGP
    X = torch.tensor([[0.08], [0.29], [0.61], [0.87]], dtype=torch.double, device=device)
    Y = (X * 5 + offset).sin()
    X.requires_grad_()
    Y.requires_grad_()
    model = cls(X, Y, **({} if reference else {"backend": "torch"}))
    model.load_mcmc_samples(
        {
            "mean": torch.linspace(-0.1, 0.15, count, dtype=X.dtype, device=device),
            "lengthscale": torch.linspace(0.25, 0.7, count, dtype=X.dtype, device=device).reshape(
                count, 1
            ),
            "outputscale": torch.linspace(0.8, 1.2, count, dtype=X.dtype, device=device),
            "noise": torch.linspace(0.03, 0.06, count, dtype=X.dtype, device=device),
        }
    )
    return model.eval()


def query(device):
    return (
        torch.tensor([[0.21], [0.53]], dtype=torch.double, device=device)
        .expand(2, 4, 2, 1)
        .clone()
        .requires_grad_()
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("noise", [False, True, "tensor"])
def test_gaussian_mixture_lists_match_reference_or_explicit_component_noise(device, mixed, noise):
    from botorch.models import ModelListGP as ReferenceList
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    first, first_ref = pair(device), pair(device, reference=True)
    second, second_ref = pair(device, 0.4), pair(device, 0.4, reference=True)
    if mixed:
        X, Y = first._train_X, first._train_Y + 0.3
        second = SingleTaskGP(X, Y, backend="torch")
        second_ref = ReferenceGP(X, Y)
    model, reference = ModelListGP(first, second), ReferenceList(first_ref, second_ref)
    points = query(device)
    observation_noise = (
        points.new_full((*points.shape[:-1], 2), 0.04) if noise == "tensor" else noise
    )
    actual = model.posterior(points, observation_noise=observation_noise)
    if noise == "tensor":
        from botorch.posteriors.fully_bayesian import GaussianMixturePosterior as ReferenceMixture
        from gpytorch.distributions import MultitaskMultivariateNormal

        # Official fully Bayesian posterior adds an axis to X but omits the
        # corresponding axis for explicit observation noise. Assert its failure
        # before constructing the intended independent component noise law.
        with pytest.raises(RuntimeError, match="expanded size"):
            reference.posterior(points, observation_noise=observation_noise)
        mvns = []
        for i, member in enumerate((first_ref, second_ref)):
            member_noise = observation_noise[..., i : i + 1]
            ensemble = getattr(member, "_is_ensemble", False)
            if ensemble:
                member_noise = member_noise.unsqueeze(-3)
            mvn = member.posterior(points, observation_noise=member_noise).distribution
            if not ensemble:
                mvn = mvn.unsqueeze(-1).expand(torch.Size((2, 4, 3)))
            mvns.append(mvn)
        expected = ReferenceMixture(MultitaskMultivariateNormal.from_independent_mvns(mvns))
    else:
        expected = reference.posterior(points, observation_noise=observation_noise)
    assert isinstance(actual, GaussianMixturePosterior)
    assert actual.mean.shape == (2, 4, 3, 2, 2)
    torch.testing.assert_close(actual.mean, expected.mean, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(actual.variance, expected.variance, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(actual.mixture_mean, expected.mixture_mean, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(
        actual.mixture_variance, expected.mixture_variance, rtol=2e-9, atol=2e-10
    )
    samples = SobolQMCNormalSampler((12,), seed=31)(actual)
    reference_samples = ReferenceSampler(torch.Size((12,)), seed=31)(expected)
    torch.testing.assert_close(samples, reference_samples, rtol=3e-8, atol=3e-9)
    assert samples.device == points.device
    gradient = torch.autograd.grad(
        (actual.mean.square() + actual.variance).sum(), points, create_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(
        (expected.mean.square() + expected.variance).sum(), points
    )[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=3e-7, atol=3e-8)
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), points)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_ordinary_batches_remain_distinct_from_mcmc(device):
    first = pair(device)
    X = first._train_X.expand(2, -1, -1)
    Y = first._train_Y.expand(2, -1, -1) + X.new_tensor([0.0, 0.4]).reshape(2, 1, 1)
    second = SingleTaskGP(X, Y, backend="torch")
    model = ModelListGP(first, second)
    points = query(device)[:, :1]
    actual = model.posterior(points)
    assert model.batch_shape == (2, 3)
    assert actual.mean.shape == (2, 2, 3, 2, 2)
    ordinary = second.posterior(points)
    torch.testing.assert_close(
        actual.mean[..., 1], ordinary.mean[..., 0].unsqueeze(-2).expand(2, 2, 3, 2)
    )
    assert torch.isfinite(
        torch.autograd.grad(actual.mixture_covariance_matrix.square().sum(), points)[0]
    ).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batched", [False, True])
def test_nonlinear_component_permutation_preserves_samples_and_mixed_gradients(device, batched):
    first = pair(device)
    X, Y = first._train_X, first._train_Y.exp()
    if batched:
        X, Y = X.expand(2, -1, -1), Y.expand(2, -1, -1) * Y.new_tensor([1.0, 1.3]).reshape(2, 1, 1)
    second = SingleTaskGP(X, Y, outcome_transform=Log(), backend="torch")
    model = ModelListGP(first, second)
    points = query(device)[:, :1] if batched else query(device)
    actual = model.posterior(points)
    assert isinstance(actual, TorchPosteriorList)
    outer = (2, 2) if batched else (2, 4)
    assert actual.mean.shape == (*outer, 3, 2, 2)
    source = second.posterior(points)
    torch.testing.assert_close(
        actual.mean[..., 1], source.mean[..., 0].unsqueeze(-2).expand(*outer, 3, 2)
    )
    torch.testing.assert_close(
        actual.variance[..., 1], source.variance[..., 0].unsqueeze(-2).expand(*outer, 3, 2)
    )
    normals = torch.randn((5, 1, 1, 1, 2, 2), dtype=points.dtype, device=device)
    samples = actual.rsample_from_base_samples((5,), normals)
    gaussian = first.posterior(points).rsample_from_base_samples((5,), normals[..., 0])
    transformed = source.rsample_from_base_samples((5,), normals[..., 0, :, 1])
    torch.testing.assert_close(samples[..., 0], gaussian[..., 0].expand_as(samples[..., 0]))
    torch.testing.assert_close(
        samples[..., 1], transformed[..., 0].unsqueeze(-2).expand(5, *outer, 3, 2)
    )
    leaves = (points, first._train_X, first._train_Y, *model.parameters())
    gradients = torch.autograd.grad(samples.square().sum(), leaves, create_graph=True)
    second_gradients = torch.autograd.grad(sum(g.square().sum() for g in gradients), leaves)
    assert all(torch.isfinite(g).all() and g.device == points.device for g in second_gradients)


@pytest.mark.parametrize("device", DEVICES)
def test_selected_outputs_and_scalarization_keep_mixture_metadata_and_graphs(device):
    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform

    model = ModelListGP(pair(device), pair(device, 0.4))
    points = query(device)
    selected = model.posterior(points, output_indices=[1, 0, 1])
    assert isinstance(selected, GaussianMixturePosterior)
    torch.testing.assert_close(selected.mean[..., 0], selected.mean[..., 2])
    weights = points.new_tensor([0.3, 0.7], requires_grad=True)
    posterior = model.posterior(points, posterior_transform=ScalarizedPosteriorTransform(weights))
    assert isinstance(posterior, GaussianMixturePosterior)
    gradient = torch.autograd.grad(
        posterior.mixture_covariance_matrix.square().sum(), (points, weights), create_graph=True
    )
    assert all(
        torch.isfinite(g).all()
        for g in torch.autograd.grad(sum(g.square().sum() for g in gradient), (points, weights))
    )


@pytest.mark.parametrize("device", DEVICES)
def test_single_component_mixture_broadcasts_without_losing_member_axis(device):
    model = ModelListGP(pair(device, count=1), pair(device))
    points = query(device)
    posterior = model.posterior(points)
    assert model.batch_shape == (3,)
    assert isinstance(posterior, GaussianMixturePosterior)
    torch.testing.assert_close(
        posterior.mean[..., 0],
        model.models[0].posterior(points).mean[..., 0].expand(2, 4, 3, 2),
    )
    torch.testing.assert_close(
        posterior.mixture_mean,
        posterior.mean.mean(-3),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_loaded_fully_bayesian_list_hypervolume_reduction_has_no_host_transfer(device, monkeypatch):
    from botorch_rs.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement

    model = ModelListGP(pair(device), pair(device, 0.4))
    points = query(device)
    baseline = model.models[0]._train_X[::2]

    def forbidden(*args, **kwargs):
        raise AssertionError("Mixture list copied inference data to the host")

    for name in ("cpu", "numpy", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model,
            [-0.3, -0.3],
            baseline,
            sampler=SobolQMCNormalSampler((12,), seed=31),
            cache_root=False,
            prune_baseline=False,
        )
    value, gradient = acq.value_and_grad(points)
    assert value.shape == points.shape[:-2]
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), points)[0]).all()
