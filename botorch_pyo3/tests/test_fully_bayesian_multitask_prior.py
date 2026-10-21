"""Multitask prior contracts and native latent-feature kernel compositions."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_torch_fully_bayesian_multitask import DEVICES, _data, _pair, _samples

from botorch_rs.models.fully_bayesian import LinearPyroModel, MaternPyroModel
from botorch_rs.models.fully_bayesian_multitask import (
    LatentFeatureMultiTaskPyroMixin,
    MultitaskSaasPyroModel,
    SaasFullyBayesianMultiTaskGP,
)


def _construct(family, backend="native", device="cpu", warp=False):
    x, y, noise = _data(device, feature=1)
    base = {"matern": MaternPyroModel, "linear": LinearPyroModel}.get(family)
    program_class = (
        MultitaskSaasPyroModel
        if base is None
        else type("LatentTaskProgram", (LatentFeatureMultiTaskPyroMixin, base), {})
    )
    model = SaasFullyBayesianMultiTaskGP(
        x,
        y,
        1,
        noise,
        rank=2,
        pyro_model=program_class(use_input_warping=warp, eps=0.03),
        backend=backend,
    )
    samples = _samples(x)
    if family != "saas":
        del samples["outputscale"]
    if family == "linear":
        samples["weight_variance"] = samples.pop("lengthscale")
    model.load_mcmc_samples(samples)
    return model, samples, (x, y, noise)


def _matern(x, lengthscale):
    distance = ((x.unsqueeze(-2) - x.unsqueeze(-3)) / lengthscale.unsqueeze(-2)).square().sum(-1)
    # The positive-distance branch suffices for off-diagonal query derivatives;
    # keep the exact self-diagonal independent of sqrt'(0).
    positive = distance > 0
    radius = torch.where(positive, distance, torch.ones_like(distance)).sqrt()
    radius = torch.where(positive, radius, torch.zeros_like(radius)) * (5.0**0.5)
    return (1 + radius + radius.square() / 3) * (-radius).exp()


def _prior(samples, query, family, warp, training):
    count = len(samples["mean"])
    batch = torch.broadcast_shapes(query.shape[:-2], (count,))
    query = query.expand(*batch, *query.shape[-2:])
    tasks = torch.where(query[..., 1] == 3, 0, 1)
    features = query[..., [0, 2]]
    if training and warp:
        inner = (0.03 + 0.94 * features).clamp(0.03, 0.97)
        features = 1 - (1 - inner.pow(samples["c1"].unsqueeze(-2))).pow(samples["c0"].unsqueeze(-2))
    if training and family == "linear":
        features = features - 0.5
    if family == "linear":
        data = (features * samples["weight_variance"].unsqueeze(-2)) @ features.transpose(-1, -2)
    else:
        data = _matern(features, samples["lengthscale"].unsqueeze(-2))
        if "outputscale" in samples:
            data = data * samples["outputscale"][..., None, None]
    task_covar = _matern(samples["latent_features"], samples["task_lengthscale"].unsqueeze(-2))
    task_covar = task_covar.expand(*batch, 2, 2)
    task_rows = task_covar.gather(-2, tasks.unsqueeze(-1).expand(*batch, tasks.shape[-1], 2))
    task_block = task_rows.gather(
        -1, tasks.unsqueeze(-2).expand(*batch, tasks.shape[-1], tasks.shape[-1])
    )
    means = samples["mean"].expand(*batch, 2).gather(-1, tasks)
    return means, data * task_block


def _query(like):
    return like.new_tensor(
        [[[[0.2, 7.0, 0.7], [0.6, 3.0, 0.4]]], [[[0.3, 3.0, 0.6], [0.8, 7.0, 0.2]]]]
    )


@pytest.mark.parametrize("family", ["saas", "matern", "linear"])
@pytest.mark.parametrize("warp", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_native_multitask_prior_matches_closed_form(family, warp, training, kind):
    model, samples, (x, _, _) = _construct(family, warp=warp)
    model.train(training, reset=False)
    query = _query(x)
    value = query.tolist() if kind == "list" else query.numpy() if kind == "numpy" else query
    actual = model.forward(value)
    mean, covariance = _prior(samples, query, family, warp, training)
    torch.testing.assert_close(
        torch.as_tensor(actual.mean, dtype=x.dtype), mean, atol=2e-12, rtol=2e-11
    )
    torch.testing.assert_close(
        torch.as_tensor(actual.covariance_matrix, dtype=x.dtype), covariance, atol=2e-12, rtol=2e-11
    )
    assert tuple(actual.batch_shape) == (2, 3)
    assert tuple(actual.event_shape) == (2,)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["saas", "matern", "linear"])
@pytest.mark.parametrize("warp", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_tensor_multitask_prior_closed_form_and_query_hvp(device, family, warp, training):
    model, samples, (x, _, _) = _construct(family, backend="torch", device=device, warp=warp)
    model.train(training, reset=False)
    query = _query(x).requires_grad_()
    actual = model.forward(query)
    mean, covariance = _prior(samples, query, family, warp, training)
    torch.testing.assert_close(actual.mean, mean, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(actual.covariance_matrix, covariance, atol=2e-12, rtol=2e-11)
    direction = torch.linspace(0.1, 0.7, query.numel(), device=device, dtype=x.dtype).reshape_as(
        query
    )
    first, second = [], []
    for covar in (actual.covariance_matrix, covariance):
        (gradient,) = torch.autograd.grad(covar.square().sum(), query, create_graph=True)
        (hvp,) = torch.autograd.grad((gradient * direction).sum(), query)
        first.append(gradient)
        second.append(hvp)
    torch.testing.assert_close(first[0], first[1], atol=3e-10, rtol=2e-9)
    torch.testing.assert_close(second[0], second[1], atol=3e-10, rtol=2e-9)
    torch.testing.assert_close(first[0][..., 1], torch.zeros_like(first[0][..., 1]), atol=0, rtol=0)
    law = model.forward(query)
    # A latent prior excludes likelihood noise, and eval forward expects
    # transformed coordinates so it also excludes the input-transform graph.
    targets = tuple(
        p
        for name, p in model.named_parameters()
        if not name.startswith("likelihood.")
        and (training or not name.startswith("input_transform."))
    )
    gradients = torch.autograd.grad(
        law.mean.square().sum() + law.covariance_matrix.square().sum(), targets, create_graph=True
    )
    hvps = torch.autograd.grad(sum(v.square().sum() for v in gradients), targets)
    assert all(bool(v.isfinite().all()) for v in (*gradients, *hvps))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("training", [False, True])
def test_tensor_multitask_forward_matches_official_prior(device, training):
    (model, reference), _ = _pair(device, fixed=True, transforms=True, warp=True)
    query = _query(model._train_X)
    for obj in (model, reference):
        obj.train(training, reset=False)
    actual, expected = model.forward(query), reference.forward(query)
    torch.testing.assert_close(actual.mean, expected.mean, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(
        actual.covariance_matrix, expected.covariance_matrix, atol=2e-12, rtol=2e-11
    )


@pytest.mark.parametrize("family", ["matern", "linear"])
@pytest.mark.parametrize("warp", [False, True])
def test_native_latent_compositions_posterior_and_conditioning_match_tensor(family, warp):
    native, _, (x, _, _) = _construct(family, warp=warp)
    tensor, _, _ = _construct(family, backend="torch", warp=warp)
    native.eval()
    tensor.eval()
    query = x.new_tensor([[0.2, 0.7], [0.6, 0.4]])
    added_x = x.new_tensor([[0.25, 7.0, 0.45]])
    added_y = x.new_tensor([[0.8]])
    noise = x.new_tensor([[0.04]])
    for a, b in [
        (native, tensor),
        (
            native.condition_on_observations(added_x, added_y, noise=noise),
            tensor.condition_on_observations(added_x, added_y, noise=noise),
        ),
    ]:
        actual, expected = a.posterior(query), b.posterior(query)
        torch.testing.assert_close(actual.mean, expected.mean, atol=3e-10, rtol=2e-9)
        torch.testing.assert_close(
            actual.distribution.covariance_matrix,
            expected.distribution.covariance_matrix,
            atol=3e-10,
            rtol=2e-9,
        )
