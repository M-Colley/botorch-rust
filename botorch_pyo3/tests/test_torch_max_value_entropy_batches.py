"""Batched MES retains independent model, fantasy and query axes."""

# ruff: noqa: E402
from __future__ import annotations

import itertools
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch_rs.models import SingleTaskGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class NoTrainingBaseline(torch.nn.Module):
    """A legitimate posterior/fantasy protocol without exposed training inputs."""

    _tensor_backend = "torch"

    def __init__(self, model):
        super().__init__()
        self.base_model = model

    @property
    def batch_shape(self):
        return self.base_model.batch_shape

    @property
    def num_outputs(self):
        return 1

    def posterior(self, *args, **kwargs):
        return self.base_model.posterior(*args, **kwargs)

    def fantasize(self, *args, **kwargs):
        return self.base_model.fantasize(*args, **kwargs)


class ModelSlice(NoTrainingBaseline):
    def __init__(self, model, indices):
        super().__init__(model)
        self.indices = indices

    @property
    def batch_shape(self):
        return self.base_model.batch_shape[: -len(self.indices)]

    def posterior(self, X, **kwargs):
        X = X.reshape(*X.shape[:-2], *((1,) * len(self.indices)), *X.shape[-2:])
        posterior = self.base_model.posterior(X, **kwargs)
        event = (slice(None), slice(None))
        selector = (Ellipsis, *self.indices, *event)
        return TorchGPyTorchPosterior(
            posterior.mean[selector], posterior.distribution.covariance_matrix[selector]
        )


def project(X):
    return torch.cat((X[..., :1], torch.ones_like(X[..., 1:])), -1)


def model(device, batch=(2,), multi=False, reference=False):
    x = torch.tensor([[0.04], [0.27], [0.56], [0.81]], device=device, dtype=torch.double)
    if multi:
        x = torch.cat((x, 0.3 + 0.6 * x), -1)
    x = x.expand(*batch, *x.shape).clone().requires_grad_()
    scale = torch.arange(1, batch.numel() + 1, device=device, dtype=x.dtype).reshape(*batch, 1, 1)
    y = ((5.3 * x[..., :1]).sin() * (1 + scale / 4)).detach().requires_grad_()
    cls = SingleTaskGP
    if reference:
        pytest.importorskip("botorch")
        from botorch.models import SingleTaskGP as cls
    result = cls(x, y, outcome_transform=None, **({} if reference else {"backend": "torch"}))
    return (NoTrainingBaseline(result) if multi else result), x, y


def acquisition(model, X, *, multi=False, source=False, fantasies=3):
    cls = qMultiFidelityMaxValueEntropy if multi else qMaxValueEntropy
    if source:
        from botorch.acquisition.max_value_entropy_search import (
            qMaxValueEntropy as SourceMES,
        )
        from botorch.acquisition.max_value_entropy_search import (
            qMultiFidelityMaxValueEntropy as SourceMF,
        )

        cls = SourceMF if multi else SourceMES
        from botorch.sampling.normal import SobolQMCNormalSampler as SourceSampler
    candidates = X.new_tensor([[0.08], [0.38], [0.71]])
    if multi:
        candidates = torch.cat((candidates, torch.full_like(candidates, 0.7)), -1)
    options = {"project": project} if multi else {"train_inputs": X.reshape(-1, *X.shape[-2:])[0]}
    acq = cls(
        model,
        candidates,
        num_mv_samples=4,
        num_y_samples=8,
        num_fantasies=fantasies,
        use_gumbel=False,
        **options,
        **(
            {}
            if source
            else {
                "sampler": SobolQMCNormalSampler((8,), seed=13),
                "fantasies_sampler": SobolQMCNormalSampler((fantasies,), seed=17),
                "seed": 19,
            }
        ),
    )
    if source:
        acq.sampler = SourceSampler(torch.Size([8]), seed=13)
        acq.fantasies_sampler = SourceSampler(torch.Size([fantasies]), seed=17)
    return acq


def maxima(X, batch):
    values = X.new_tensor([0.8, 1.1, 1.4, 1.7])
    offsets = torch.arange(batch.numel(), device=X.device, dtype=X.dtype).reshape(*batch) / 5
    return values.reshape(4, *((1,) * len(batch))) + offsets


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_batched_mes_matches_source_values_query_gradients_and_no_pending_parameters(
    device, multi, pending
):
    pytest.importorskip("botorch")
    import gpytorch
    from botorch.settings import propagate_grads

    values, gradients = [], []
    for source in (False, True):
        gp, x, y = model(device, batch=torch.Size([2]), multi=multi, reference=source)
        acq = acquisition(gp, x, multi=multi, source=source, fantasies=1)
        if pending:
            pending_X = x.new_tensor([[0.91, 0.8]] if multi else [[0.91]])
            acq.set_X_pending(pending_X.expand(2, *pending_X.shape) if source else pending_X)
        acq.posterior_max_values = maxima(x, torch.Size([2]))
        query = x.new_tensor(
            [[[0.19, 0.6]], [[0.43, 0.8]]] if multi else [[[0.19]], [[0.43]]],
            requires_grad=True,
        )
        if source:
            base = gp.base_model if multi else gp
            base.prediction_strategy = None
            acq.model.prediction_strategy = None
        with gpytorch.settings.detach_test_caches(False), propagate_grads(True):
            value = acq(query)
            values.append(value)
            leaves = (query,)
            if not pending:
                leaves += (x, y, *(p for _, p in sorted(gp.named_parameters())))
            gradients.append(torch.autograd.grad(value.sum(), leaves, allow_unused=True))
    torch.testing.assert_close(*values, atol=4e-8, rtol=4e-8)
    for actual, expected in zip(*gradients, strict=True):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("batch", [torch.Size([2]), torch.Size([2, 3])])
def test_model_and_fantasy_axes_equal_independent_component_average_with_full_hvps(
    device, multi, pending, batch
):
    gp, x, y = model(device, batch, multi)
    acq = acquisition(gp, x, multi=multi)
    if pending:
        acq.set_X_pending(x.new_tensor([[0.91, 0.8]] if multi else [[0.91]]))
    acq.posterior_max_values = maxima(x, batch).requires_grad_()
    query = x.new_tensor(
        [[[[0.19, 0.6]], [[0.43, 0.8]]]] if multi else [[[[0.19]], [[0.43]]]],
        requires_grad=True,
    )
    actual = acq(query)
    component_values = []
    for index in itertools.product(*(range(length) for length in batch)):
        initial = ModelSlice(gp, index)
        component = acquisition(initial, x, multi=multi)
        component.model = ModelSlice(acq.model, index)
        component.posterior_max_values = acq.posterior_max_values[(slice(None), *index)].unsqueeze(
            -1
        )
        component_values.append(component(query))
    expected = torch.stack(component_values).mean(0)
    torch.testing.assert_close(actual, expected, atol=4e-8, rtol=4e-8)
    leaves = (query, x, y, acq.posterior_max_values, *dict.fromkeys(acq.parameters()))
    derivatives = [
        torch.autograd.grad(value.sum(), leaves, create_graph=True, allow_unused=True)
        for value in (actual, expected)
    ]
    for a, b in zip(*derivatives, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)
            assert torch.isfinite(a).all()
    hvps = [
        torch.autograd.grad(
            sum(value.square().sum() / 2 for value in derivative if value is not None),
            leaves,
            allow_unused=True,
            retain_graph=True,
        )
        for derivative in derivatives
    ]
    for a, b in zip(*hvps, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
            assert torch.isfinite(a).all()


@pytest.mark.parametrize("device", DEVICES)
def test_batched_mes_construction_and_higher_derivatives_do_not_copy_inference_to_host(device):
    gp, x, y = model(device, torch.Size([2]))
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Tensor.cpu")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("Tensor.numpy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("Tensor.tolist")),
    ):
        acq = acquisition(gp, x)
        acq.set_X_pending(x.new_tensor([[0.91]]))
        query = x.new_tensor([[[0.19]], [[0.43]]], requires_grad=True)
        value = acq(query)
        first = torch.autograd.grad(value.sum(), (query, x, y), create_graph=True)
        second = torch.autograd.grad(sum(v.square().sum() / 2 for v in first), (query, x, y))
    assert all(
        value.device.type == device and torch.isfinite(value).all() for value in (*first, *second)
    )
