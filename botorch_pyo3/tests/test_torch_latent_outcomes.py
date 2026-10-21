"""General LatentKronecker outcome laws retain structured sampling graphs."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from botorch_rs.models.latent_kronecker_gp import LatentKroneckerGP
from botorch_rs.models.transforms.outcome import (
    Bilog,
    ChainedOutcomeTransform,
    Log,
    OutcomeTransform,
    Power,
    Standardize,
    StratifiedStandardize,
)
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent")
    ),
]


class AffineOutcome(OutcomeTransform):
    _is_linear = True

    def __init__(self, like):
        self.scale = nn.Parameter(like.new_tensor(1.7))
        self.register_buffer("offset", like.new_tensor(0.3))

    def forward(self, Y, Yvar=None, X=None):
        return (Y - self.offset) / self.scale, None if Yvar is None else Yvar / self.scale.square()

    def untransform(self, Y, Yvar=None, X=None):
        return Y * self.scale + self.offset, None if Yvar is None else Yvar * self.scale.square()

    def untransform_posterior(self, posterior, X=None):
        return TorchGPyTorchPosterior(
            posterior.mean * self.scale + self.offset,
            posterior._covariance_tensor * self.scale.square(),
        )


class ExternalAffineOutcome(nn.Module):
    _is_linear = True

    def __init__(self, like):
        super().__init__()
        self.scale = nn.Parameter(like.new_tensor(1.7))
        self.register_buffer("offset", like.new_tensor(0.3))

    forward = AffineOutcome.forward
    untransform = AffineOutcome.untransform
    untransform_posterior = AffineOutcome.untransform_posterior


class ShiftedStandardize(Standardize):
    def __init__(self, like):
        super().__init__(1, batch_shape=like.shape[:-2])
        self.shift = nn.Parameter(like.new_tensor(0.7))

    def forward(self, Y, Yvar=None, X=None):
        values, variance = super().forward(Y, Yvar, X=X)
        return values + self.shift, variance

    def untransform(self, Y, Yvar=None, X=None):
        return super().untransform(Y - self.shift, Yvar, X=X)

    def untransform_posterior(self, posterior, X=None):
        shifted = TorchGPyTorchPosterior(posterior.mean - self.shift, posterior._covariance_tensor)
        return super().untransform_posterior(shifted, X=X)


def _data(device, missing=True, batch=False):
    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device)
    T = X.new_tensor([[0.0], [1.0]])
    Y = 1.8 + (X * 2 + T.squeeze(-1) * 0.7).sin()
    Q, U = X.new_tensor([[0.25], [0.7]]), X.new_tensor([[0.2], [0.8]])
    if missing:
        Y[1, 1] = float("nan")
    if batch:
        X, Y = X.expand(2, -1, -1).clone(), Y.expand(2, -1, -1).clone()
        Y[1] = Y[1] + 0.2
    return tuple(value.requires_grad_() for value in (X, T, Y, Q, U))


def _transform(family, Y):
    return {
        "log": lambda: Log(),
        "power": lambda: Power(nn.Parameter(Y.new_tensor(2.3))),
        "bilog": lambda: Bilog(),
        "chain": lambda: ChainedOutcomeTransform(
            log=Log(), standardize=Standardize(1, batch_shape=Y.shape[:-2])
        ),
        "affine": lambda: AffineOutcome(Y),
        "external": lambda: ExternalAffineOutcome(Y),
        "custom_standardize": lambda: ShiftedStandardize(Y),
    }[family]()


def _training_and_inverse(family, outcome, Y, mask):
    observed = Y.flatten(-2)[..., mask]
    if family == "log":
        return observed.log(), torch.exp, None
    if family == "power":
        return observed.pow(outcome.power), lambda value: value.pow(1 / outcome.power), None
    if family == "bilog":
        return (
            observed.log1p(),
            lambda value: torch.where(value >= 0, value.expm1(), -(-value).expm1()),
            None,
        )
    if family == "chain":
        values = observed.log()
        means, scales = values.mean(-1, keepdim=True), values.std(-1, keepdim=True)

        def inverse(value):
            return (value * scales + means).exp()

        return (values - means) / scales, inverse, (means, scales)
    if family == "custom_standardize":
        means, scales = observed.mean(-1, keepdim=True), observed.std(-1, keepdim=True)
        return (
            (observed - means) / scales + outcome.shift,
            lambda value: (value - outcome.shift) * scales + means,
            (means, scales),
        )
    return (
        (observed - outcome.offset) / outcome.scale,
        lambda value: value * outcome.scale + outcome.offset,
        None,
    )


def _law(model, X, T, Q, U, target, base):
    full = model._covariance(X, T, X, T)
    cross_full = model._covariance(X, T, Q, U)
    test = model._covariance(Q, U, Q, U)
    mask = model.mask_valid
    cross = cross_full[..., mask, :]
    covariance = full[..., mask, :][..., :, mask]
    noise = model.likelihood.noise.expand_as(target)
    matrix = covariance + torch.diag_embed(noise)
    train_mean, test_mean = model._get_mean(X, T, mask), model._get_mean(Q, U)
    mean = test_mean + (
        cross.transpose(-1, -2) @ torch.linalg.solve(matrix, (target - train_mean).unsqueeze(-1))
    ).squeeze(-1)
    cov = test - cross.transpose(-1, -2) @ torch.linalg.solve(matrix, cross)
    nfull, nobs = mask.numel(), int(mask.sum())
    joint = torch.cat(
        (torch.cat((full, cross_full), -1), torch.cat((cross_full.transpose(-1, -2), test), -1)), -2
    )
    normal = torch.cat((base[..., :nfull], base[..., nfull + nobs :]), -1)
    prior = (torch.linalg.cholesky(joint) @ normal.unsqueeze(-1)).squeeze(-1)
    residual = (
        target
        - train_mean
        - prior[..., :nfull][..., mask]
        - noise.sqrt() * base[..., nfull : nfull + nobs]
    )
    samples = (
        test_mean
        + prior[..., nfull:]
        + (cross.transpose(-1, -2) @ torch.linalg.solve(matrix, residual.unsqueeze(-1))).squeeze(-1)
    )
    return mean, cov, samples


def _compare(actual, expected, leaves):
    torch.testing.assert_close(actual, expected, atol=5e-9, rtol=5e-9)
    first = torch.autograd.grad(actual, leaves, create_graph=True, retain_graph=True)
    reference = torch.autograd.grad(expected, leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    ref_second = torch.autograd.grad(
        sum(g.square().sum() for g in reference), leaves, retain_graph=True
    )
    for a, b in zip((*first, *second), (*reference, *ref_second), strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=3e-6, rtol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family", ["log", "power", "bilog", "chain", "affine", "external", "custom_standardize"]
)
@pytest.mark.parametrize("batch", [False, True])
def test_general_outcomes_dense_matheron_law_and_all_leaf_hessians(device, family, batch):
    X, T, Y, Q, U = _data(device, batch=batch)
    outcome = _transform(family, Y)
    model = LatentKroneckerGP(X, T, Y, outcome_transform=outcome, backend="torch")
    assert model.outcome_transform is outcome
    posterior = model.posterior(Q, U)
    bases = torch.linspace(
        -0.3, 0.5, 3 * posterior.base_sample_shape.numel(), dtype=Y.dtype, device=device
    ).reshape(3, *posterior.base_sample_shape)
    target, inverse, calibration = _training_and_inverse(family, outcome, Y, model.mask_valid)
    torch.testing.assert_close(model.train_targets, target)
    mean, cov, draws = _law(model, X, T, Q, U, target, bases)
    actual_samples = posterior.rsample_from_base_samples(torch.Size([3]), bases)
    expected_samples = inverse(draws).reshape(actual_samples.shape)
    actual, expected = actual_samples.sin().sum(), expected_samples.sin().sum()
    variance = cov.diagonal(dim1=-2, dim2=-1)
    if family in ("log", "chain"):
        location, scale = (0.0, 1.0) if calibration is None else calibration
        raw_mean = (mean * scale + location + 0.5 * variance * scale**2).exp()
        raw_variance = (variance * scale**2).expm1() * (
            2 * (mean * scale + location) + variance * scale**2
        ).exp()
        torch.testing.assert_close(posterior.mean.flatten(-2), raw_mean)
        torch.testing.assert_close(posterior.variance.flatten(-2), raw_variance)
        actual = actual + posterior.mean.sin().sum() + posterior.variance.square().sum()
        expected = expected + raw_mean.sin().sum() + raw_variance.square().sum()
    elif family in ("affine", "external", "custom_standardize"):
        scale = outcome.scale if calibration is None else calibration[1]
        torch.testing.assert_close(posterior.mean.flatten(-2), inverse(mean))
        torch.testing.assert_close(
            posterior.distribution.covariance_matrix,
            cov * scale.unsqueeze(-1).square()
            if isinstance(scale, torch.Tensor) and scale.ndim
            else cov * scale**2,
        )
        actual = actual + posterior.mean.sin().sum() + posterior.variance.square().sum()
        expected = expected + inverse(mean).sin().sum() + (variance * scale**2).square().sum()
    else:
        with pytest.raises(NotImplementedError):
            _ = posterior.mean
        with pytest.raises(NotImplementedError):
            _ = posterior.variance
    _compare(actual, expected, (X, T, Y, Q, U, *model.parameters()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["log", "chain", "affine", "external"])
def test_general_outcome_conditioning_fantasies_and_frozen_posterior_statistics(device, family):
    X, T, Y, Q, _ = _data(device)
    model = LatentKroneckerGP(X, T, Y, outcome_transform=_transform(family, Y), backend="torch")
    posterior = model.posterior(Q)
    expected_mean, expected_variance = posterior.mean.clone(), posterior.variance.clone()
    bases = torch.linspace(
        -0.2, 0.3, 2 * posterior.base_sample_shape.numel(), device=device, dtype=Y.dtype
    ).reshape(2, *posterior.base_sample_shape)
    expected_samples = posterior.rsample_from_base_samples(torch.Size([2]), bases)
    if family == "chain":
        model.outcome_transform.train()(Y.flatten(-2)[..., model.mask_valid].unsqueeze(-1) * 1.4)
    elif family in ("affine", "external"):
        model.outcome_transform.offset.add_(0.2)
    torch.testing.assert_close(posterior.mean, expected_mean)
    torch.testing.assert_close(posterior.variance, expected_variance)
    torch.testing.assert_close(
        posterior.rsample_from_base_samples(torch.Size([2]), bases), expected_samples
    )
    conditioned = model.condition_on_observations(Q[:1], Y.new_full((1, 2), 2.4))
    assert conditioned.outcome_transform is not model.outcome_transform
    assert conditioned.posterior(Q).mean.shape == (2, 2)
    fantasy = model.fantasize(
        Q[:1], IIDNormalSampler(torch.Size([2]), seed=413), propagate_grads=True
    )
    loss = fantasy.posterior(Q).mean.sin().sum()
    gradients = torch.autograd.grad(
        loss, (Y, Q, *model.parameters()), create_graph=True, allow_unused=True
    )
    assert all(value is None or torch.isfinite(value).all() for value in gradients)


@pytest.mark.parametrize("device", DEVICES)
def test_source_valid_single_time_stratified_outcome_uses_feature_context(device):
    X = torch.tensor([[0.1, 1], [0.3, 3], [0.6, 1], [0.9, 3]], dtype=torch.double, device=device)
    T = X.new_tensor([[0.0]])
    Y = X.new_tensor([[1.1], [2.4], [1.8], [3.7]]).requires_grad_()
    outcome = StratifiedStandardize(1, [1, 3])
    model = LatentKroneckerGP(X, T, Y, outcome_transform=outcome, backend="torch")
    Q = X.new_tensor([[0.2, 1], [0.8, 3]]).requires_grad_()
    posterior = model.posterior(Q)
    internal = model._posterior(Q, untransform=False)
    expected = outcome.untransform_posterior(internal, X=Q)
    _compare(
        posterior.mean.sin().sum() + posterior.variance.square().sum(),
        expected.mean.sin().sum() + expected.variance.square().sum(),
        (Y, Q, *model.parameters()),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["log", "power", "bilog", "chain"])
def test_pinned_source_training_targets_and_transformed_gaussian_evaluation(device, family):
    reference = pytest.importorskip("botorch.models.latent_kronecker_gp")
    outcomes = pytest.importorskip("botorch.models.transforms.outcome")
    X, T, Y, Q, U = _data(device)
    actual = LatentKroneckerGP(X, T, Y, outcome_transform=_transform(family, Y), backend="torch")
    source_outcome = {
        "log": lambda: outcomes.Log(),
        "power": lambda: outcomes.Power(nn.Parameter(Y.new_tensor(2.3))),
        "bilog": lambda: outcomes.Bilog(),
        "chain": lambda: outcomes.ChainedOutcomeTransform(
            log=outcomes.Log(), standardize=outcomes.Standardize(1)
        ),
    }[family]()
    expected = reference.LatentKroneckerGP(X, T, Y, outcome_transform=source_outcome).to(X)
    a_params, b_params = dict(actual.named_parameters()), dict(expected.named_parameters())
    assert a_params.keys() == b_params.keys()
    with torch.no_grad():
        for name in a_params:
            a_params[name].copy_(b_params[name])
    torch.testing.assert_close(actual.train_targets, expected.train_targets)
    expected.eval()
    a_law = actual._posterior(Q, U, untransform=False).distribution
    b_law = expected.posterior(Q, U).distribution
    torch.testing.assert_close(a_law.mean, b_law.mean, atol=3e-9, rtol=3e-8)
    torch.testing.assert_close(
        a_law.covariance_matrix, b_law.covariance_matrix, atol=3e-9, rtol=3e-8
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["log", "chain", "affine"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_latent_model_state_roundtrip_preserves_grid_targets_and_outcome_law(
    device, family, batch, missing
):
    X, T, Y, Q, U = _data(device, missing=missing, batch=batch)
    model = LatentKroneckerGP(X, T, Y, outcome_transform=_transform(family, Y), backend="torch")
    with torch.no_grad():
        initial = model.posterior(Q, U)
        means, variances = initial.mean.clone(), initial.variance.clone()
        targets = model.train_targets.clone()
        state = copy.deepcopy(model.state_dict())
        for parameter in model.parameters():
            parameter.add_(0.13)
        model.load_state_dict(state)
        restored = model.posterior(Q, U)
    torch.testing.assert_close(restored.mean, means)
    torch.testing.assert_close(restored.variance, variances)
    torch.testing.assert_close(model.train_targets, targets)
    assert model.train_targets.shape == targets.shape
    assert model.train_T.shape[-2] == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["log", "chain", "affine"])
def test_latent_fantasy_state_roundtrip_has_writable_time_batches(device, family):
    X, T, Y, Q, U = _data(device)
    model = LatentKroneckerGP(X, T, Y, outcome_transform=_transform(family, Y), backend="torch")
    with torch.no_grad():
        fantasy = model.fantasize(Q[:1], IIDNormalSampler(torch.Size([2]), seed=178))
        before = fantasy.posterior(Q, U)
        means, variances = before.mean.clone(), before.variance.clone()
        fantasy.load_state_dict(copy.deepcopy(fantasy.state_dict()))
        restored = fantasy.posterior(Q, U)
    torch.testing.assert_close(restored.mean, means)
    torch.testing.assert_close(restored.variance, variances)
