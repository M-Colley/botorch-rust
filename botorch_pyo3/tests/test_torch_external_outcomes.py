"""Supplied outcome modules retain callbacks, state, and calibration graphs."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from botorch_rs.models._torch_backend_modules import build_outcome_transform
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.transforms.outcome import OutcomeTransform
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


class CalibratedOutcome(nn.Module):
    _is_linear = True

    def __init__(self, like):
        super().__init__()
        self.scale = nn.Parameter(like.new_tensor(1.7))
        self.context = nn.Parameter(like.new_tensor(0.3))
        self.register_buffer("location", like.new_zeros(1, 1))

    def forward(self, Y, Yvar=None, X=None):
        if self.training:
            self.location = Y.mean(-2, keepdim=True) + self.context * X.mean(-2, keepdim=True)
        return (Y - self.location) / self.scale, (
            None if Yvar is None else Yvar / self.scale.square()
        )

    def untransform(self, Y, Yvar=None, X=None):
        return Y * self.scale + self.location, (
            None if Yvar is None else Yvar * self.scale.square()
        )

    def untransform_posterior(self, posterior, X=None):
        return TorchGPyTorchPosterior(
            posterior.mean * self.scale + self.location,
            posterior._covariance_tensor * self.scale.square(),
        )


class CalibratedPublicOutcome(OutcomeTransform):
    _is_linear = True

    def __init__(self, like):
        self.scale = nn.Parameter(like.new_tensor(1.7))
        self.context = nn.Parameter(like.new_tensor(0.3))
        self.register_buffer("location", like.new_zeros(1, 1))

    forward = CalibratedOutcome.forward
    untransform = CalibratedOutcome.untransform
    untransform_posterior = CalibratedOutcome.untransform_posterior


def _compare(actual, expected, leaves):
    torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
    first = torch.autograd.grad(actual, leaves, create_graph=True, retain_graph=True)
    reference = torch.autograd.grad(expected, leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    ref_second = torch.autograd.grad(
        sum(g.square().sum() for g in reference), leaves, retain_graph=True
    )
    for a, b in zip((*first, *second), (*reference, *ref_second), strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outcome_type", [CalibratedOutcome, CalibratedPublicOutcome])
def test_pure_torch_calibration_keeps_context_and_all_leaf_hessians(device, outcome_type):
    x = torch.tensor([[0.1], [0.35], [0.7], [0.9]], dtype=torch.double, device=device)
    x.requires_grad_()
    y = (1.4 + x.detach().sin()).requires_grad_()
    outcome = outcome_type(y)
    model = SingleTaskGP(x, y, outcome_transform=outcome, backend="torch")
    assert model.outcome_transform is outcome
    model._training_values()
    location = y.mean(-2, keepdim=True) + outcome.context * x.mean(-2, keepdim=True)
    target = (y - location) / outcome.scale
    _compare(
        model.train_targets.sin().sum(),
        target.squeeze(-1).sin().sum(),
        (x, y, *outcome.parameters()),
    )
    query = x.detach()[:2].clone().requires_grad_()
    law = model.posterior(query)
    first = torch.autograd.grad(
        law.mean.sin().sum() + law.variance.square().sum(),
        (x, y, query, *model.parameters()),
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first), (x, y, query, *model.parameters())
    )
    assert all(torch.isfinite(value).all() for value in (*first, *second))
    detached = build_outcome_transform(outcome_type(y), 1, y.detach(), train_X=x.detach())
    for duplicate in (copy.deepcopy(detached), pickle.loads(pickle.dumps(detached))):
        torch.testing.assert_close(duplicate(y, X=x)[0], detached(y, X=x)[0])
        assert dict(duplicate.named_parameters()).keys() == {"scale", "context"}
        assert dict(duplicate.named_buffers()).keys() == {"location"}


def _shift_hook(module, arguments, result):
    return result[0] + module.shift, result[1]


def _source_transform(family, y):
    source = pytest.importorskip("botorch.models.transforms.outcome")
    if family in ("custom", "chain"):

        class ShiftedStandardize(source.Standardize):
            def forward(self, Y, Yvar=None, X=None):
                values, variance = super().forward(Y, Yvar, X=X)
                return values + self.shift, variance

        result = ShiftedStandardize(y.shape[-1]).to(y)
    else:
        result = source.Standardize(y.shape[-1]).to(y)
    result.register_parameter("shift", nn.Parameter(y.new_tensor(0.4)))
    if family == "hook":
        result.register_forward_hook(_shift_hook)
    if family == "chain":
        result = source.ChainedOutcomeTransform(log=source.Log(), standardize=result)
    return result


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["hook", "custom", "chain"])
def test_actual_source_callbacks_and_calibration_graphs_are_preserved(device, family):
    y = torch.tensor([[1.1], [1.6], [2.4], [3.1]], dtype=torch.double, device=device)
    y.requires_grad_()
    source = _source_transform(family, y)
    transform = build_outcome_transform(source, 1, y)
    assert transform is source
    transform.fit(y)
    child = source["standardize"] if family == "chain" else source
    values = y.log() if family == "chain" else y
    expected = (values - values.mean(-2, keepdim=True)) / values.std(-2, keepdim=True) + child.shift
    _compare(transform(y)[0].sin().sum(), expected.sin().sum(), (y, child.shift))
    query = y.new_tensor([[0.2], [0.6]]).requires_grad_()
    model = SingleTaskGP(y.detach() / 4, y, outcome_transform=source, backend="torch")
    assert model.outcome_transform is source
    posterior = model.posterior(query)
    first = torch.autograd.grad(
        posterior.mean.sin().sum() + posterior.variance.square().sum(),
        (y, query, *model.parameters()),
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(value.square().sum() for value in first), (y, query, *model.parameters())
    )
    assert all(torch.isfinite(value).all() for value in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["hook", "custom", "chain"])
def test_source_gaussian_callback_interoperability_preserves_multitask_covariance(device, family):
    source_posteriors = pytest.importorskip("botorch.posteriors.gpytorch")
    distributions = pytest.importorskip("gpytorch.distributions")
    operators = pytest.importorskip("linear_operator.operators")
    y = torch.tensor([[1.1], [1.6], [2.4], [3.1]], dtype=torch.double, device=device)
    y.requires_grad_()
    source = _source_transform(family, y)
    original_type = type(source)
    transform = build_outcome_transform(source, 1, y)
    transform.fit(y)
    mean = y.new_tensor([[0.2, 0.5], [0.7, 0.9]]).requires_grad_()
    root = y.new_tensor(
        [[0.8, 0.0, 0.0, 0.0], [0.1, 0.7, 0.0, 0.0], [0.2, 0.1, 0.9, 0.0], [0.0, 0.2, 0.1, 0.8]]
    ).requires_grad_()
    covariance = root @ root.transpose(-1, -2)
    expected = original_type.untransform_posterior(
        source,
        source_posteriors.GPyTorchPosterior(
            distributions.MultitaskMultivariateNormal(
                mean, operators.DenseLinearOperator(covariance), interleaved=False
            )
        ),
    )
    actual = transform.untransform_posterior(TorchGPyTorchPosterior(mean, covariance))
    value = actual.mean.sin().sum() + actual.variance.square().sum()
    reference = expected.mean.sin().sum() + expected.variance.square().sum()
    if family != "chain":
        identity = torch.eye(4, device=device, dtype=y.dtype)
        expected_covariance = (
            expected.distribution.lazy_covariance_matrix.evaluate_kernel()._matmul(identity)
        )
        torch.testing.assert_close(actual._covariance_tensor, expected_covariance)
        value = value + actual._covariance_tensor.sin().sum()
        reference = reference + expected_covariance.sin().sum()
    _compare(value, reference, (y, mean, root))
