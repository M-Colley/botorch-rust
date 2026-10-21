"""Public standardization APIs keep device statistics and complete Torch graphs."""

# ruff: noqa: E402
from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models.gp_regression import SingleTaskGP as RefGP
from botorch.models.transforms.outcome import Standardize as RefStandardize
from botorch.models.transforms.outcome import StratifiedStandardize as RefStratified

from botorch_rs.models._torch_backend_fully_bayesian import TorchGaussianMixturePosterior
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Log,
    Standardize,
    StratifiedStandardize,
)
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.posteriors.ensemble import EnsemblePosterior

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent")
    ),
]


def _compare_graphs(actual, expected, leaves):
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    first = torch.autograd.grad(actual, leaves, create_graph=True, retain_graph=True)
    reference = torch.autograd.grad(expected, leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    reference_second = torch.autograd.grad(
        sum(g.square().sum() for g in reference), leaves, retain_graph=True
    )
    for a, b in zip((*first, *second), (*reference, *reference_second), strict=True):
        assert bool(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("outputs", [None, [0, 2]])
def test_direct_standardize_source_values_training_noise_and_mixed_hessians(device, batch, outputs):
    generator = torch.Generator(device=device).manual_seed(184)
    Y = torch.randn(
        *batch, 5, 3, dtype=torch.double, device=device, generator=generator
    ).requires_grad_()
    V = (Y.detach().square() * 0.03 + 0.05).requires_grad_()
    actual, expected = Standardize(3, outputs, batch), RefStandardize(3, outputs, torch.Size(batch))
    for _ in range(2):
        # Each training call owns a fresh statistics graph. Earlier returned
        # tensors remain valid when a later call replaces the module buffers.
        a, av = actual(Y, V)
        b, bv = expected(Y, V)
        weights = torch.linspace(0.2, 1.1, a.numel(), device=device, dtype=Y.dtype).reshape_as(a)
        _compare_graphs(
            (a.sin() * weights).sum() + av.square().sum(),
            (b.sin() * weights).sum() + bv.square().sum(),
            (Y, V),
        )
        for name in ("means", "stdvs", "_stdvs_sq"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    actual.eval()
    expected.eval()
    query = torch.randn(4, *batch, 2, 3, dtype=Y.dtype, device=device).requires_grad_()
    qvar = (query.detach().square() * 0.1 + 0.04).requires_grad_()
    for transform, reference in ((actual, expected),):
        a, av = transform(query, qvar)
        b, bv = reference(query, qvar)
        ar, avr = transform.untransform(a, av)
        br, bvr = reference.untransform(b, bv)
        torch.testing.assert_close(ar, query)
        torch.testing.assert_close(avr, qvar)
        _compare_graphs(
            a.square().sum() + av.sin().sum() + ar.square().sum() + avr.sum(),
            b.square().sum() + bv.sin().sum() + br.square().sum() + bvr.sum(),
            (Y, query, qvar),
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stratified", [False, True])
def test_direct_standardize_state_roundtrip_migration_and_native_transfer(device, stratified):
    Y = torch.tensor([[0.2], [0.7], [1.4], [-0.3]], dtype=torch.double, device=device)
    X = Y.new_tensor([[0, 1], [0.1, 3], [0.2, 1], [0.3, 3]])
    make = (lambda: StratifiedStandardize(1, [1, 3])) if stratified else (lambda: Standardize(1))
    transform = make().to(Y)
    reference = (
        RefStratified(1, torch.tensor([1, 3], device=device), dtype=Y.dtype)
        if stratified
        else RefStandardize(1)
    ).to(Y)
    actual = transform(Y, Y.square() * 0.1, X=X)
    expected = reference(Y, Y.square() * 0.1, X=X)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    state = copy.deepcopy(transform.state_dict())
    assert state.keys() == reference.state_dict().keys()
    for name, tensor in state.items():
        torch.testing.assert_close(tensor, reference.state_dict()[name].to(tensor))
    restored = make().to(Y)
    restored.load_state_dict(state)
    restored.eval()
    query = Y[:2] + 0.3
    torch.testing.assert_close(restored(query, X=X[:2])[0], transform.eval()(query, X=X[:2])[0])
    restored.float().cpu()
    assert restored.means.dtype == torch.float and restored.means.device.type == "cpu"
    assert dict(restored.named_buffers())["means"] is restored.means
    assert list(restored.parameters()) == []
    if device == "cuda":
        restored.cuda(torch.device("cuda", torch.cuda.current_device()))
        assert restored.means.device.type == "cuda"
        restored.cpu()
    with pytest.raises(ValueError, match="CUDA device"):
        restored.cuda("cpu")
    # Returning explicitly to an array API is a numerical state transfer,
    # while subsequent Tensor calls retain the stored destination device.
    a = restored(query.cpu().float().tolist(), X=X[:2].cpu().float().tolist())[0]
    b = restored(query.cpu().float(), X=X[:2].cpu().float())[0]
    torch.testing.assert_close(torch.tensor(a), b)
    native = make()
    native(Y.cpu().tolist(), X=X.cpu().tolist())
    native.eval()
    torch.testing.assert_close(native(query, X=X[:2])[0], transform(query, X=X[:2])[0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stratified", [False, True])
def test_direct_standardize_gaussian_posterior_and_immutable_statistics_graph(device, stratified):
    Y = torch.tensor(
        [[0.2], [0.7], [1.4], [-0.3]], dtype=torch.double, device=device, requires_grad=True
    )
    X = Y.new_tensor([[0, 0], [0.1, 1], [0.2, 0], [0.3, 1]])
    transform = StratifiedStandardize(1, [0, 1]) if stratified else Standardize(1)
    transform(Y, X=X)
    transform.eval()
    mean = Y.new_tensor([[0.1], [0.4]], requires_grad=True)
    factor = Y.new_tensor([[0.7, 0], [0.2, 0.5]], requires_grad=True)
    posterior = TorchGPyTorchPosterior(mean, factor @ factor.mT)
    result = transform.untransform_posterior(posterior, X=X[:2])
    if stratified:
        offset = torch.stack((Y[::2].mean(), Y[1::2].mean())).unsqueeze(-1)
        scale = torch.stack((Y[::2].std(), Y[1::2].std())).unsqueeze(-1)
    else:
        offset, scale = Y.mean(-2, keepdim=True), Y.std(-2, keepdim=True)
    expected_mean = mean * scale + offset
    expected_cov = (factor @ factor.mT) * scale.squeeze(-1).unsqueeze(-1) * scale.squeeze(-1)
    # Refit replaces the transform law, but cannot mutate an existing posterior.
    transform.train()(Y + torch.arange(4, device=device).unsqueeze(-1), X=X)
    _compare_graphs(
        result.mean.sin().sum() + result.distribution.covariance_matrix.square().sum(),
        expected_mean.sin().sum() + expected_cov.square().sum(),
        (Y, mean, factor),
    )
    normals = Y.new_tensor([[-0.4, 0.1], [0.7, -0.2], [0.6, 0.3]])
    samples = result.rsample_from_base_samples(torch.Size([3]), normals)
    expected_samples = expected_mean.squeeze(-1) + normals @ torch.linalg.cholesky(expected_cov).mT
    torch.testing.assert_close(samples.squeeze(-1), expected_samples)


@pytest.mark.parametrize("device", DEVICES)
def test_direct_standardize_multitask_subset_and_generic_sample_graph(device):
    Y = torch.tensor(
        [[0.2, 0.1], [0.7, 0.8], [1.4, -0.3]], dtype=torch.double, device=device, requires_grad=True
    )
    transform = Standardize(2)
    transform(Y)
    transform.eval()
    subset = transform.subset_output([1, 0])
    torch.testing.assert_close(subset.means, transform.means[..., [1, 0]])
    values = torch.stack((Y[:2] + 0.3, Y[:2] - 0.2), -3)
    result = transform.untransform_posterior(EnsemblePosterior(values))
    expected_samples = values * Y.std(-2, keepdim=True) + Y.mean(-2, keepdim=True)
    _compare_graphs(
        result.mean.square().sum() + result.variance.square().sum(),
        expected_samples.mean(-3).square().sum() + expected_samples.var(-3).square().sum(),
        (Y,),
    )
    indices = torch.tensor([0, 1], device=device)
    samples = result.rsample_from_base_samples(torch.Size([2]), indices)
    torch.testing.assert_close(samples, expected_samples)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
def test_stratified_direct_batched_statistics_matches_independent_per_task_law(device, batch):
    Y = torch.tensor([[0.2], [1.1], [0.7], [-0.3], [0.4]], dtype=torch.double, device=device)
    Y = Y.expand(*batch, *Y.shape).clone().requires_grad_()
    X = Y.detach().new_tensor([[0.0, 1], [0.2, 3], [0.4, 1], [0.6, 3], [0.8, 1]])
    X = X.expand(*batch, *X.shape).clone()
    V = (Y.detach().square() * 0.1 + 0.02).requires_grad_()
    transform = StratifiedStandardize(-1, [1, 3], batch_shape=batch)
    with patch(
        "botorch_rs.models.transforms.outcome.to_flat",
        side_effect=AssertionError("host conversion"),
    ):
        actual, av = transform(Y, V, X=X)
    # Independent per-task means and unbiased standard deviations.
    means = torch.stack([Y[..., [0, 2, 4], :].mean(-2), Y[..., [1, 3], :].mean(-2)], -2)
    stds = torch.stack([Y[..., [0, 2, 4], :].std(-2), Y[..., [1, 3], :].std(-2)], -2)
    order = torch.tensor([0, 1, 0, 1, 0], device=device)
    mean, std = means.index_select(-2, order), stds.index_select(-2, order)
    expected, ev = (Y - mean) / std, V / std.square()
    _compare_graphs(
        actual.sin().sum() + av.square().sum(), expected.sin().sum() + ev.square().sum(), (Y, V)
    )
    transform.eval()
    unknown = X[..., :2, :].clone()
    unknown[..., 0, -1] = 2
    with pytest.warns(UserWarning, match="identity transform"):
        values, variance = transform(Y[..., :2, :], V[..., :2, :], X=unknown)
    torch.testing.assert_close(values[..., 0, :], Y[..., 0, :])
    torch.testing.assert_close(variance[..., 0, :], V[..., 0, :])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("observations", [1, 4])
def test_direct_standardize_constant_data_has_finite_first_second_derivatives(device, observations):
    Y = torch.ones(observations, 2, dtype=torch.double, device=device, requires_grad=True)
    values, variance = Standardize(2)(Y, Y.square() * 0.1)
    gradient = torch.autograd.grad(values.square().sum() + variance.sum(), Y, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient.sum(), Y)[0]
    assert bool(torch.isfinite(gradient).all()) and bool(torch.isfinite(hessian).all())


@pytest.mark.parametrize("device", DEVICES)
def test_direct_standardize_chain_and_error_protocols(device):
    Y = torch.tensor([[0.2], [0.7], [1.3]], device=device, dtype=torch.double, requires_grad=True)
    transform = ChainedOutcomeTransform(log=Log(), standardize=Standardize(1))
    values, _ = transform(Y)
    restored, _ = transform.untransform(values)
    torch.testing.assert_close(restored, Y)
    gradient = torch.autograd.grad(restored.square().sum(), Y, create_graph=True)[0]
    torch.testing.assert_close(gradient, 2 * Y)
    with pytest.raises(RuntimeError, match="fitted"):
        Standardize(1).untransform(Y)
    with pytest.raises(RuntimeError, match="broadcast-compatible"):
        Standardize(1)(Y, Y[:2])
    with pytest.raises(ValueError, match="X is required"):
        StratifiedStandardize(0, [0, 1])(Y)
    with pytest.raises(NotImplementedError):
        StratifiedStandardize(0, [0, 1]).subset_output([0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mixture", [False, True])
def test_direct_standardize_full_cross_output_covariance_and_mixture_class(device, mixture):
    Y = torch.tensor(
        [[0.2, -0.1], [0.7, 0.8], [1.4, 0.4]], dtype=torch.double, device=device, requires_grad=True
    )
    transform = Standardize(2)
    transform(Y)
    mean = Y.new_tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
    factor = (torch.arange(16, device=device, dtype=Y.dtype).reshape(4, 4) / 80).tril() + torch.eye(
        4, device=device, dtype=Y.dtype
    )
    factor.requires_grad_()
    covariance = factor @ factor.mT
    cls = TorchGaussianMixturePosterior if mixture else TorchGPyTorchPosterior
    posterior = (
        cls(mean.unsqueeze(0).expand(3, 2, 2), covariance.expand(3, 4, 4))
        if mixture
        else cls(mean, covariance)
    )
    result = transform.untransform_posterior(posterior)
    assert type(result) is cls
    scale = Y.std(-2).repeat_interleave(2)
    expected_cov = covariance * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    expected_mean = posterior.mean * Y.std(-2) + Y.mean(-2)
    _compare_graphs(
        result.mean.square().sum() + result.distribution.covariance_matrix.sin().sum(),
        expected_mean.square().sum()
        + expected_cov.expand_as(result.distribution.covariance_matrix).sin().sum(),
        (Y, mean, factor),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_direct_standardize_nan_observations_and_fresh_training_graph(device, dtype):
    Y = torch.tensor(
        [[0.1, 0.4], [float("nan"), 0.8], [0.7, float("nan")]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    transform = Standardize(2)
    for _ in range(2):
        actual, _ = transform(Y)
        expected = torch.stack(((Y[0, 0] + Y[2, 0]) / 2, (Y[0, 1] + Y[1, 1]) / 2))
        torch.testing.assert_close(transform.means.squeeze(-2), expected)
        loss = (
            actual.nan_to_num().sin().sum()
            + transform.means.square().sum()
            + transform.stdvs.square().sum()
        )
        first = torch.autograd.grad(loss, Y, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), Y)[0]
        assert bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("supplied", ["port", "source"])
def test_gp_constructor_refits_supplied_eval_descriptor_as_source(device, supplied):
    X = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double, device=device)
    Y = torch.tensor([[0.2], [0.7], [0.3]], dtype=X.dtype, device=device, requires_grad=True)
    calibration = torch.tensor(
        [[2.1], [4.3], [3.4]], dtype=X.dtype, device=device, requires_grad=True
    )
    transform = Standardize(1) if supplied == "port" else RefStandardize(1).to(Y)
    transform(calibration)
    transform.eval()
    model = SingleTaskGP(X, Y, outcome_transform=transform, backend="torch")
    source_transform = RefStandardize(1).to(Y)
    source_transform(calibration)
    source_transform.eval()
    source = RefGP(X, Y, outcome_transform=source_transform)
    query = X.new_tensor([[0.3], [0.6]], requires_grad=True)
    for name in ("means", "stdvs"):
        torch.testing.assert_close(
            getattr(model.outcome_transform, name), getattr(source.outcome_transform, name)
        )
    torch.testing.assert_close(model.outcome_transform.means, Y.mean(-2, keepdim=True))
    posterior = model.posterior(query)
    gradient = torch.autograd.grad(
        posterior.mean.square().sum() + posterior.variance.sum(),
        (Y, query),
        create_graph=True,
    )
    for tensor in gradient:
        assert bool(torch.isfinite(tensor).all())
    second = torch.autograd.grad(sum(g.square().sum() for g in gradient), (Y, query))
    for tensor in second:
        assert bool(torch.isfinite(tensor).all())
    model.train()
    model.posterior(query)
    torch.testing.assert_close(model.outcome_transform.means, Y.mean(-2, keepdim=True))


@pytest.mark.parametrize("device", DEVICES)
def test_stratified_eval_broadcasts_sample_batches_and_single_observation_graph(device):
    Y = torch.tensor([[[0.2]], [[0.7]]], dtype=torch.double, device=device, requires_grad=True)
    X = Y.new_tensor([[[0, 0]], [[0.4, 1]]])
    transform = StratifiedStandardize(-1, [0, 1], batch_shape=(2,))
    values, _ = transform(Y, X=X)
    torch.testing.assert_close(values, torch.zeros_like(Y))
    transform.eval()
    query = (Y.detach().unsqueeze(0).expand(3, 2, 1, 1) + 0.2).requires_grad_()
    result, _ = transform.untransform(query, X=X)
    expected = query + Y
    _compare_graphs(result.square().sum(), expected.square().sum(), (Y, query))
