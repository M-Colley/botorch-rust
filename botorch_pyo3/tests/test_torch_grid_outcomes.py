"""General outcome maps on Kronecker and higher-order Tensor grids."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from botorch_rs.models.higher_order_gp import FlattenedStandardize, HigherOrderGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP
from botorch_rs.models.transforms.outcome import (
    Bilog,
    ChainedOutcomeTransform,
    Log,
    OutcomeTransform,
    Power,
    Standardize,
)
from botorch_rs.posteriors.transformed import TransformedPosterior
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
        return TransformedPosterior(
            posterior,
            lambda samples: samples * self.scale + self.offset,
            lambda mean, var: mean * self.scale + self.offset,
            lambda mean, var: var * self.scale.square(),
        )


def _data(kind, device, batch=False):
    X = torch.tensor([[0.1], [0.45], [0.9]], device=device, dtype=torch.double)
    outputs = torch.arange(4 if kind == "higher" else 2, device=device, dtype=X.dtype)
    Y = 1.8 + (X * 2 + outputs * 0.3).sin()
    if kind == "higher":
        Y = Y.reshape(3, 2, 2)
    Q = X.new_tensor([[0.25], [0.7]])
    if batch:
        X = X.expand(2, -1, -1).clone()
        Y = Y.expand(2, *Y.shape).clone()
        Y[1] += 0.2
    return tuple(value.requires_grad_() for value in (X, Y, Q))


def _outcome(kind, family, Y):
    batch = Y.shape[:-3] if kind == "higher" else Y.shape[:-2]
    return {
        "log": Log,
        "power": lambda: Power(nn.Parameter(Y.new_tensor(2.3))),
        "bilog": Bilog,
        "chain": lambda: ChainedOutcomeTransform(
            log=Log(),
            standardize=FlattenedStandardize((2, 2), batch)
            if kind == "higher"
            else Standardize(2, batch_shape=batch),
        ),
        "affine": lambda: AffineOutcome(Y),
    }[family]()


def _manual(kind, family, Y, transform):
    if family == "log":
        return Y.log(), torch.exp, (0.0, 1.0)
    if family == "power":
        return Y.pow(transform.power), lambda value: value.pow(1 / transform.power), None
    if family == "bilog":
        return (
            Y.log1p(),
            lambda value: torch.where(value >= 0, value.expm1(), -(-value).expm1()),
            None,
        )
    if family == "chain":
        values = Y.log()
        if kind == "higher":
            flat = values.flatten(-3)
            shape = values.shape[:-3] + (1, 1, 1)
            location, scale = flat.mean(-1).reshape(shape), flat.std(-1).reshape(shape)
        else:
            location, scale = values.mean(-2, keepdim=True), values.std(-2, keepdim=True)
        return (
            (values - location) / scale,
            lambda value: (value * scale + location).exp(),
            (location, scale),
        )
    return (
        (Y - transform.offset) / transform.scale,
        lambda value: value * transform.scale + transform.offset,
        None,
    )


def _model(kind, X, Y, outcome, **kwargs):
    cls = HigherOrderGP if kind == "higher" else KroneckerMultiTaskGP
    return cls(X, Y, outcome_transform=outcome, backend="torch", **kwargs)


def _compare(actual, expected, leaves):
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
    first = torch.autograd.grad(actual, leaves, create_graph=True, retain_graph=True)
    reference = torch.autograd.grad(expected, leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    ref_second = torch.autograd.grad(
        sum(g.square().sum() for g in reference), leaves, retain_graph=True
    )
    for a, b in zip((*first, *second), (*reference, *ref_second), strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=4e-6, rtol=4e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["kronecker", "higher"])
@pytest.mark.parametrize("family", ["log", "power", "bilog", "chain", "affine"])
@pytest.mark.parametrize("batch", [False, True])
def test_grid_outcome_training_moments_and_structured_samples_all_leaf_hessians(
    device, kind, family, batch
):
    X, Y, Q = _data(kind, device, batch)
    transform = _outcome(kind, family, Y)
    model = _model(kind, X, Y, transform)
    assert model.outcome_transform is transform
    posterior, internal = model.posterior(Q), model._posterior(Q, untransform=False)
    target, inverse, calibration = _manual(kind, family, Y, transform)
    torch.testing.assert_close(
        model.train_targets, target.flatten(-3) if kind == "higher" else target
    )
    base = torch.linspace(
        -0.2, 0.3, 3 * posterior.base_sample_shape.numel(), device=device, dtype=Y.dtype
    ).reshape(3, *posterior.base_sample_shape)
    actual_samples = posterior.rsample_from_base_samples(torch.Size([3]), base)
    expected_samples = inverse(internal.rsample_from_base_samples(torch.Size([3]), base))
    actual, expected = actual_samples.sin().sum(), expected_samples.sin().sum()
    if family in ("log", "chain"):
        location, scale = calibration
        mean, variance = internal.mean * scale + location, internal.variance * scale**2
        raw_mean = (mean + 0.5 * variance).exp()
        raw_variance = variance.expm1() * (2 * mean + variance).exp()
    elif family == "affine":
        raw_mean, raw_variance = (
            inverse(internal.mean),
            internal.variance * transform.scale.square(),
        )
    else:
        with pytest.raises(NotImplementedError):
            _ = posterior.mean
        with pytest.raises(NotImplementedError):
            _ = posterior.variance
    if family in ("log", "chain", "affine"):
        torch.testing.assert_close(posterior.mean, raw_mean)
        torch.testing.assert_close(posterior.variance, raw_variance)
        actual = actual + posterior.mean.sin().sum() + posterior.variance.square().sum()
        expected = expected + raw_mean.sin().sum() + raw_variance.square().sum()
    _compare(actual, expected, (X, Y, Q, *model.parameters()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["kronecker", "higher"])
@pytest.mark.parametrize("family", ["log", "chain", "affine"])
def test_grid_general_outcome_conditioning_fantasies_and_snapshot_stability(device, kind, family):
    X, Y, Q = _data(kind, device)
    model = _model(kind, X, Y, _outcome(kind, family, Y))
    posterior = model.posterior(Q)
    means, variances = posterior.mean.clone(), posterior.variance.clone()
    base = Y.new_zeros(2, *posterior.base_sample_shape)
    samples = posterior.rsample_from_base_samples(torch.Size([2]), base)
    if family == "chain":
        model.outcome_transform.train()(Y * 1.4)
    elif family == "affine":
        model.outcome_transform.offset.add_(0.2)
    torch.testing.assert_close(posterior.mean, means)
    torch.testing.assert_close(posterior.variance, variances)
    torch.testing.assert_close(posterior.rsample_from_base_samples(torch.Size([2]), base), samples)
    conditioned = model.condition_on_observations(Q[:1], Y[:1] * 1.1)
    assert conditioned.outcome_transform is not model.outcome_transform
    assert conditioned.posterior(Q).mean.shape == Y.shape[:0] + (2, *Y.shape[1:])
    fantasy = model.fantasize(
        Q[:1], IIDNormalSampler(torch.Size([2]), seed=612), propagate_grads=True
    )
    first = torch.autograd.grad(
        fantasy.posterior(Q).mean.sin().sum(),
        (Y, Q, *model.parameters()),
        create_graph=True,
        allow_unused=True,
    )
    assert all(value is None or torch.isfinite(value).all() for value in first)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_higher_log_explicit_conditioning_noise_uses_actual_raw_observations(device, fixed):
    X, Y, Q = _data("higher", device)
    old_noise = Y.new_full((Y.numel(),), 0.06, requires_grad=True)
    likelihood = (
        pytest.importorskip("gpytorch.likelihoods").FixedNoiseGaussianLikelihood(old_noise)
        if fixed
        else None
    )
    model = _model("higher", X, Y, Log(), likelihood=likelihood)
    new_Y = (Y[:1].detach() * 1.1).requires_grad_()
    new_noise = new_Y.new_full(new_Y.shape, 0.03, requires_grad=True)
    conditioned = model.condition_on_observations(Q[:1], new_Y, noise=new_noise)
    from botorch_rs.models._torch_structured_likelihood import (
        conditioned_observed_law,
        fixed_conditioning_noise,
    )

    if fixed:
        actual = fixed_conditioning_noise(conditioned)[..., -new_Y.numel() :]
    else:
        count = conditioned._train_X.shape[-2] * conditioned._grid_size
        covariance = Y.new_zeros(count, count)
        observed = torch.eye(count, dtype=Y.dtype, device=device) * 0.06
        actual = conditioned_observed_law(
            conditioned, Y.new_zeros(count), covariance, observed
        ).diagonal(dim1=-2, dim2=-1)[-new_Y.numel() :]
    expected = (new_noise / new_Y.square()).flatten()
    _compare(actual.square().sum(), expected.square().sum(), (new_Y, new_noise))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["kronecker", "higher"])
@pytest.mark.parametrize("family", ["log", "power", "bilog", "chain"])
def test_grid_outcome_source_targets_and_available_moments(device, kind, family):
    reference = pytest.importorskip("botorch.models")
    outcomes = pytest.importorskip("botorch.models.transforms.outcome")
    higher = pytest.importorskip("botorch.models.higher_order_gp")
    X, Y, Q = _data(kind, device)
    source_transform = {
        "log": outcomes.Log,
        "power": lambda: outcomes.Power(nn.Parameter(Y.new_tensor(2.3))),
        "bilog": outcomes.Bilog,
        "chain": lambda: outcomes.ChainedOutcomeTransform(
            log=outcomes.Log(),
            standardize=higher.FlattenedStandardize(torch.Size([2, 2]))
            if kind == "higher"
            else outcomes.Standardize(2),
        ),
    }[family]()
    expected = (higher.HigherOrderGP if kind == "higher" else reference.KroneckerMultiTaskGP)(
        X, Y, outcome_transform=source_transform
    ).to(X)
    actual = _model(kind, X, Y, _outcome(kind, family, Y))
    a_params, b_params = dict(actual.named_parameters()), dict(expected.named_parameters())
    assert a_params.keys() == b_params.keys()
    with torch.no_grad():
        for name in a_params:
            a_params[name].copy_(b_params[name])
    torch.testing.assert_close(actual.train_targets, expected.train_targets)
    if family in ("log", "chain"):
        a_posterior, b_posterior = actual.posterior(Q), expected.posterior(Q)
        torch.testing.assert_close(a_posterior.mean, b_posterior.mean, atol=3e-7, rtol=3e-7)
        torch.testing.assert_close(a_posterior.variance, b_posterior.variance, atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("source_chain", [False, True])
def test_source_flattened_standardize_inside_public_chain_is_supported(device, source_chain):
    source = pytest.importorskip("botorch.models.higher_order_gp")
    outcomes = pytest.importorskip("botorch.models.transforms.outcome")
    X, Y, Q = _data("higher", device)
    transform = ChainedOutcomeTransform(
        log=Log(), standardize=source.FlattenedStandardize(torch.Size([2, 2])).to(Y)
    )
    if source_chain:
        transform = outcomes.ChainedOutcomeTransform(
            log=outcomes.Log(), standardize=source.FlattenedStandardize(torch.Size([2, 2])).to(Y)
        )
    model = _model("higher", X, Y, transform)
    if not source_chain:
        assert model.outcome_transform is transform
    posterior = model.posterior(Q)
    first = torch.autograd.grad(
        posterior.mean.sin().sum(), (Y, Q, *model.parameters()), create_graph=True
    )
    second = torch.autograd.grad(
        sum(value.square().sum() for value in first), (Y, Q, *model.parameters())
    )
    assert all(torch.isfinite(value).all() for value in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_kronecker_selected_nonlinear_outputs_and_repeated_subsets(device):
    X, Y, Q = _data("kronecker", device)
    model = _model("kronecker", X, Y, Log(outputs=[1]))
    posterior = model.posterior(Q)
    latent = model._posterior(Q, untransform=False)
    mean = torch.cat(
        (latent.mean[..., :1], (latent.mean[..., 1:] + 0.5 * latent.variance[..., 1:]).exp()), -1
    )
    variance = torch.cat(
        (
            latent.variance[..., :1],
            latent.variance[..., 1:].expm1()
            * (2 * latent.mean[..., 1:] + latent.variance[..., 1:]).exp(),
        ),
        -1,
    )
    torch.testing.assert_close(posterior.mean, mean)
    torch.testing.assert_close(posterior.variance, variance)
    subset = model.posterior(Q, output_indices=[1, 0, 1])
    torch.testing.assert_close(subset.mean, mean[..., [1, 0, 1]])
    torch.testing.assert_close(subset.variance, variance[..., [1, 0, 1]])
    _compare(
        subset.mean.sin().sum() + subset.variance.square().sum(),
        mean[..., [1, 0, 1]].sin().sum() + variance[..., [1, 0, 1]].square().sum(),
        (X, Y, Q, *model.parameters()),
    )
