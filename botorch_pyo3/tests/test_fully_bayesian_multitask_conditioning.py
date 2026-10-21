"""Fantasy axes are distinct from the final fully Bayesian component axis."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("jax")

from test_torch_fully_bayesian_multitask import DEVICES, _pair, _samples

from botorch_rs.models.fully_bayesian_multitask import (
    MultitaskSaasPyroModel,
    SaasFullyBayesianMultiTaskGP,
)
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize


def _models(device, warp, transforms):
    (tensor, reference), data = _pair(device, fixed=True, warp=warp, transforms=transforms)
    x, y, noise = data[0]
    native = SaasFullyBayesianMultiTaskGP(
        x.cpu(),
        y.cpu(),
        1,
        noise.cpu(),
        rank=2,
        pyro_model=MultitaskSaasPyroModel(use_input_warping=warp, eps=0.03),
        input_transform=Normalize(3, indices=[0, 2], bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        if transforms
        else None,
        outcome_transform=Standardize(1) if transforms else None,
        backend="native",
    )
    native.load_mcmc_samples({key: value.cpu() for key, value in _samples(x).items()})
    native.eval()
    return native, tensor, reference


def _observations(like):
    X = like.new_tensor([[[[0.25, 7.0, 0.45]]], [[[0.35, 3.0, 0.55]]]])
    Y = like.new_tensor([[[[0.8]], [[0.9]], [[1.0]]], [[[1.2]], [[1.3]], [[1.4]]]])
    return X, Y, torch.full_like(Y, 0.04)


def _dense_reference(reference, X, Y, noise, query):
    """Condition the official prior directly, avoiding its fantasy-cache bug."""
    batch = torch.broadcast_shapes(reference.batch_shape, X.shape[:-2], Y.shape[:-2])
    original_train = getattr(reference, "_original_train_inputs", reference.train_inputs[0])
    if original_train is None:
        original_train = reference.train_inputs[0]
    train_x = original_train.expand(*batch, *original_train.shape[-2:])
    query_full = torch.cat(
        [
            torch.cat((query[:, :1], query.new_full((len(query), 1), task), query[:, 1:]), -1)
            for task in (3, 7)
        ]
    ).expand(*batch, 2 * len(query), 3)
    full = torch.cat((train_x, X.expand(*batch, *X.shape[-2:]), query_full), -2)
    transformed = reference.transform_inputs(full)
    prior = reference.forward(transformed)
    mapped = transformed.clone()
    mapped[..., 1] = reference._map_tasks(mapped[..., 1])
    covariance = (
        reference.covar_module.kernels[0](mapped).to_dense()
        * reference.covar_module.kernels[1](mapped).to_dense()
    )
    means = prior.mean
    n = train_x.shape[-2]
    targets = torch.cat(
        (reference.train_targets.expand(*batch, n), Y.squeeze(-1).expand(*batch, Y.shape[-2])), -1
    )
    # Original stored targets are already transformed, while new targets are
    # supplied in original outcome units. Keep the fitted statistics fixed.
    transform = getattr(reference, "outcome_transform", None)
    offset, scale = 0.0, 1.0
    if transform is not None:
        extra, _ = transform(Y)
        targets = torch.cat(
            (
                reference.train_targets.expand(*batch, n),
                extra.squeeze(-1).expand(*batch, Y.shape[-2]),
            ),
            -1,
        )
        offset, variance = transform.untransform(Y.new_zeros(1, 1), Y.new_ones(1, 1))
        offset, scale = offset.squeeze(), variance.squeeze().sqrt()
    train_noise = reference.likelihood.noise.expand(*batch, n)
    all_noise = torch.cat((train_noise, noise.squeeze(-1).expand(*batch, Y.shape[-2])), -1)
    count = targets.shape[-1]
    K = covariance[..., :count, :count] + torch.diag_embed(all_noise)
    cross = covariance[..., count:, :count]
    mean = means[..., count:] + (
        cross @ torch.linalg.solve(K, (targets - means[..., :count]).unsqueeze(-1))
    ).squeeze(-1)
    cov = covariance[..., count:, count:] - cross @ torch.linalg.solve(K, cross.transpose(-1, -2))
    return (mean * scale + offset).reshape(*batch, 2, len(query)).transpose(
        -1, -2
    ), cov * scale * scale


@pytest.mark.parametrize("warp", [False, True])
@pytest.mark.parametrize("transforms", [False, True])
@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_native_fantasy_and_mcmc_axes_match_official_dense_gaussian_law(warp, transforms, kind):
    native, _, reference = _models("cpu", warp, transforms)
    X, Y, noise = _observations(reference.train_inputs[0])
    convert = (
        (lambda t: t.tolist())
        if kind == "list"
        else (lambda t: t.numpy())
        if kind == "numpy"
        else (lambda t: t)
    )
    conditioned = native.condition_on_observations(convert(X), convert(Y), noise=convert(noise))
    assert conditioned.batch_shape == (2, 3)
    assert tuple(conditioned.train_inputs[0].shape) == (2, 3, 7, 3)
    assert tuple(conditioned.train_targets.shape) == (2, 3, 7)
    query = X.new_tensor([[0.2, 0.7], [0.6, 0.4]], requires_grad=True)
    actual = conditioned.posterior(query)
    expected_mean, expected_cov = _dense_reference(reference, X, Y, noise, query)
    torch.testing.assert_close(actual.mean, expected_mean, atol=3e-10, rtol=2e-9)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected_cov, atol=3e-10, rtol=2e-9
    )
    expected_gradient = torch.autograd.grad(
        expected_mean.square().sum() + expected_cov.square().sum(), query
    )[0]
    actual_gradient = torch.autograd.grad(
        actual.mean.square().sum() + actual.distribution.covariance_matrix.square().sum(), query
    )[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-9, rtol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("warp", [False, True])
def test_tensor_fantasy_axes_metadata_and_complete_conditioning_graphs(device, warp):
    _, model, reference = _models(device, warp, True)
    X, Y, noise = (value.requires_grad_() for value in _observations(model._train_X))
    conditioned = model.condition_on_observations(X, Y, noise=noise)
    assert conditioned.batch_shape == torch.Size([2, 3])
    assert conditioned._input_batch_shape == torch.Size([2, 3])
    query = X.new_tensor([[0.2, 0.7], [0.6, 0.4]], requires_grad=True)
    actual = conditioned.posterior(query)
    expected_mean, expected_cov = _dense_reference(reference, X, Y, noise, query)
    torch.testing.assert_close(actual.mean, expected_mean, atol=3e-10, rtol=2e-9)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected_cov, atol=3e-10, rtol=2e-9
    )
    targets = (X, Y, noise, query)
    derivatives = []
    for mean, covariance in (
        (actual.mean, actual.distribution.covariance_matrix),
        (expected_mean, expected_cov),
    ):
        first = torch.autograd.grad(
            mean.square().sum() + covariance.square().sum(), targets, create_graph=True
        )
        directions = [torch.full_like(value, 0.17) for value in targets]
        directions[0][..., 1] = 0
        second = torch.autograd.grad(
            sum((g * d).sum() for g, d in zip(first, directions, strict=True)), targets
        )
        derivatives.append((*first, *second))
    for a, b in zip(*derivatives, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=2e-7)


def test_native_nested_conditioning_metadata_prior_and_failure_atomicity():
    model, _, _ = _models("cpu", True, True)
    X, Y, noise = _observations(model.train_inputs[0])
    first = model.condition_on_observations(X, Y, noise=noise)
    nested_X = X.new_tensor(
        [
            [[[[0.28, 3.0, 0.63]]]],
            [[[[0.48, 7.0, 0.53]]]],
            [[[[0.58, 3.0, 0.43]]]],
            [[[[0.68, 7.0, 0.33]]]],
        ]
    )
    nested_Y = Y.new_full((4, 2, 3, 1, 1), 0.95)
    second = first.condition_on_observations(
        nested_X, nested_Y, noise=nested_Y.new_full(nested_Y.shape, 0.03)
    )
    assert second.batch_shape == (4, 2, 3)
    assert second.train_inputs[0].shape == (4, 2, 3, 8, 3)
    assert second.train_targets.shape == (4, 2, 3, 8)
    query = X.new_tensor([[0.2, 0.7], [0.6, 0.4]])
    saved = second.posterior(query).mean.clone()
    full_query = X.new_tensor([[0.2, 3.0, 0.7], [0.6, 7.0, 0.4]])
    priors = [
        second.forward(full_query.tolist()),
        second.forward(full_query.numpy()),
        second.forward(full_query),
    ]
    for prior in priors:
        assert tuple(prior.batch_shape) == (4, 2, 3)
        torch.testing.assert_close(
            torch.as_tensor(prior.covariance_matrix, dtype=X.dtype),
            priors[-1].covariance_matrix,
            atol=2e-12,
            rtol=2e-11,
        )
    with pytest.raises(ValueError, match="do not broadcast"):
        second.condition_on_observations(X.new_zeros(5, 1, 1, 3), nested_Y, noise=noise)
    torch.testing.assert_close(second.posterior(query).mean, saved, atol=0, rtol=0)
    assert model.train_inputs[0].shape == (6, 3)
