# ruff: noqa: E402
"""Actual device GP entropy inference, synchronized sampling and full graphs."""

import pytest

pytest.importorskip("torch")


from functools import partial

import torch

from botorch_rs.acquisition import max_value_entropy_search as port
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NAMES = [
    "qMaxValueEntropy",
    "qLowerBoundMaxValueEntropy",
    "qMultiFidelityMaxValueEntropy",
    "qMultiFidelityLowerBoundMaxValueEntropy",
]


def data(device, multi=False):
    X = torch.tensor([[0.04], [0.27], [0.56], [0.81]], dtype=torch.double, device=device)
    if multi:
        X = torch.cat((X, 0.3 + 0.6 * X), -1)
    Y = (5.3 * X[..., :1]).sin()
    return X, Y


def project(X):
    return torch.cat((X[..., :-1], torch.ones_like(X[..., -1:])), -1)


def make_pair(device, name, pending=False, maximize=True, trace=False, dtype=torch.double):
    from botorch.acquisition import max_value_entropy_search as reference
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    multi = "MultiFidelity" in name
    lower = "LowerBound" in name
    X, Y = data(device, multi)
    X, Y = X.to(dtype), Y.to(dtype)
    model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
    ref_model = ReferenceGP(X, Y, outcome_transform=None)
    candidates = torch.linspace(0.05, 0.95, 7, dtype=X.dtype, device=device).unsqueeze(-1)
    if multi:
        candidates = torch.cat((candidates, torch.full_like(candidates, 0.7)), -1)
    kwargs = dict(num_mv_samples=4, use_gumbel=False, maximize=maximize)
    if multi:
        kwargs["project"] = project
        if trace:
            from botorch.acquisition.utils import expand_trace_observations as RefExpand

            from botorch_rs.acquisition.utils import expand_trace_observations

            kwargs["expand"] = partial(
                expand_trace_observations, fidelity_dims=[1], num_trace_obs=1
            )
            ref_kwargs = dict(kwargs, expand=partial(RefExpand, fidelity_dims=[1], num_trace_obs=1))
        else:
            ref_kwargs = dict(kwargs)
    else:
        ref_kwargs = dict(kwargs)
    if not lower or multi:
        kwargs.update(num_fantasies=3, num_y_samples=16)
        ref_kwargs.update(num_fantasies=3, num_y_samples=16)
    ref = getattr(reference, name)(ref_model, candidates, **ref_kwargs)
    extension = (
        {}
        if lower
        else dict(
            sampler=SobolQMCNormalSampler((16,), seed=13),
            fantasies_sampler=SobolQMCNormalSampler((3,), seed=17),
        )
    )
    actual = getattr(port, name)(model, candidates, seed=19, **kwargs, **extension)
    if not lower:
        ref.sampler = ReferenceSampler(torch.Size((16,)), seed=13)
        ref.fantasies_sampler = ReferenceSampler(torch.Size((3,)), seed=17)
    if pending:
        value = X.new_tensor([[0.91, 0.8]] if multi else [[0.91]])
        actual.set_X_pending(value)
        ref.set_X_pending(value)
    maxima = X.new_tensor([0.75, 1.0, 1.2, 1.4] if maximize else [-0.4, 0.1, 0.3, 0.5]).unsqueeze(
        -1
    )
    actual.posterior_max_values = maxima
    ref.posterior_max_values = maxima
    return actual, ref, X


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_entropy_values_and_first_query_gradients_match_pinned_reference(
    device, name, pending, maximize
):
    actual, reference, X = make_pair(device, name, pending, maximize)
    query = (
        X.new_tensor([[[0.19, 0.6]], [[0.43, 0.8]]])
        if "MultiFidelity" in name
        else X.new_tensor([[[0.19]], [[0.43]]])
    )
    query.requires_grad_()
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=2e-7, atol=2e-8)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=3e-5, atol=3e-6)
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
    assert isinstance(actual, getattr(port, name)) and value.device == query.device


@pytest.mark.parametrize("device", DEVICES)
def test_trace_observations_and_cost_sampler_independence(device):
    actual, reference, X = make_pair(device, "qMultiFidelityMaxValueEntropy", trace=True)
    query = X.new_tensor([[[0.19, 0.6]], [[0.43, 0.8]]], requires_grad=True)
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=3e-7, atol=3e-8)
    assert actual.cost_sampler is not actual.fantasies_sampler
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["qMaxValueEntropy", "qLowerBoundMaxValueEntropy"])
def test_entropy_complete_training_and_query_graph_passes_gradgradcheck(device, name):
    X, Y = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    query = X.new_tensor([[0.23]], requires_grad=True)
    candidate = X.detach()

    def evaluate(train_X, train_Y, points):
        model = SingleTaskGP(train_X, train_Y, backend="torch", outcome_transform=None)
        kwargs = dict(num_mv_samples=4, use_gumbel=False, seed=19)
        if name == "qMaxValueEntropy":
            kwargs["sampler"] = SobolQMCNormalSampler((12,), seed=13)
        acq = getattr(port, name)(model, candidate, **kwargs)
        acq.posterior_max_values = X.new_tensor([[0.75], [1.0], [1.2], [1.4]])
        return acq(points)

    assert torch.autograd.gradcheck(evaluate, (X, Y, query), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (X, Y, query), fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
def test_entropy_constructor_forward_and_second_backward_never_copy_inference_to_host(
    device, name, monkeypatch
):
    X, Y = data(device, "MultiFidelity" in name)
    model = SingleTaskGP(
        X.requires_grad_(), Y.requires_grad_(), backend="torch", outcome_transform=None
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Entropy inference copied Tensor data to the host")

    for field in ("cpu", "numpy", "tolist"):
        monkeypatch.setattr(torch.Tensor, field, forbidden)
    kwargs = dict(num_mv_samples=4, seed=13)
    if "LowerBound" not in name or "MultiFidelity" in name:
        kwargs.update(num_fantasies=3, num_y_samples=12)
    acq = getattr(port, name)(model, X.detach(), **kwargs)
    query = (X[:1] * 0.7 + 0.1).detach().requires_grad_()
    value, gradient = acq.value_and_grad(query)
    assert value.device == X.device and torch.isfinite(value).all()
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_gumbel_quantiles_match_float64_oracle_and_reference_rng(device):
    from botorch.acquisition.max_value_entropy_search import _sample_max_value_Gumbel as reference
    from botorch.models import SingleTaskGP as ReferenceGP

    X, Y = data(device)
    model, ref = (
        SingleTaskGP(X, Y, backend="torch", outcome_transform=None),
        ReferenceGP(X, Y, outcome_transform=None),
    )
    candidate = torch.linspace(0.03, 0.97, 9, device=device, dtype=X.dtype).unsqueeze(-1)
    with torch.no_grad():
        torch.manual_seed(31)
        expected = reference(ref, candidate, 7)
        actual = port._sample_max_value_Gumbel(model, candidate, 7, seed=31)
    # The public reference rounds the SciPy roots to its default float32.
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-8)
    from math import log

    from scipy.optimize import brentq
    from scipy.stats import norm

    with torch.no_grad():
        posterior = ref.posterior(candidate)
        mean = posterior.mean.squeeze(-1).cpu().numpy()
        sigma = posterior.variance.squeeze(-1).clamp_min(1e-8).sqrt().cpu().numpy()
        distribution = norm(mean, sigma)
        lower, upper = (mean - 3 * sigma).min(), (mean + 5 * sigma).max()
        quantiles = [
            brentq(lambda y, p=p: distribution.logcdf(y).sum() - log(p), lower, upper)
            for p in (0.25, 0.5, 0.75)
        ]
        q25, q50, q75 = quantiles
        scale = (q25 - q75) / (log(log(4 / 3)) - log(log(4)))
        location = q50 + scale * log(log(2))
        generator = torch.Generator(device=device).manual_seed(31)
        uniforms = torch.rand((7, 1), dtype=X.dtype, device=device, generator=generator)
        oracle = location - scale * (-uniforms.log()).log()
    torch.testing.assert_close(actual, oracle, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_standalone_gumbel_sampler_retains_implicit_training_derivatives(device):
    X, Y = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    candidate = X.detach() * 0.8 + 0.1
    uniforms = X.new_tensor([[0.2], [0.5], [0.8]])

    def evaluate(train_X, train_Y):
        model = SingleTaskGP(train_X, train_Y, backend="torch", outcome_transform=None)
        return port._sample_max_value_Gumbel(model, candidate, 3, uniforms=uniforms)

    assert torch.autograd.gradcheck(evaluate, (X, Y), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (X, Y), fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
def test_float32_entropy_reference_values_and_gradients(device, name):
    actual, reference, X = make_pair(device, name, dtype=torch.float32)
    query = (X[:2] * 0.8 + 0.1).unsqueeze(-2).detach().requires_grad_()
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=3e-3, atol=3e-4)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=3e-3, atol=3e-4)
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
def test_default_gumbel_constructor_preserves_reference_rng_order(device, name):
    from botorch.acquisition import max_value_entropy_search as reference
    from botorch.models import SingleTaskGP as ReferenceGP

    X, Y = data(device, "MultiFidelity" in name)
    model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
    ref_model = ReferenceGP(X, Y, outcome_transform=None)
    kwargs = dict(num_mv_samples=4)
    if name != "qLowerBoundMaxValueEntropy":
        kwargs.update(num_fantasies=3, num_y_samples=16)
    torch.manual_seed(37)
    ref = getattr(reference, name)(ref_model, X, **kwargs)
    expected_cpu, expected_device = torch.rand(3), torch.rand(3, device=device)
    torch.manual_seed(37)
    actual = getattr(port, name)(model, X, **kwargs)
    torch.testing.assert_close(
        actual.posterior_max_values, ref.posterior_max_values, rtol=2e-7, atol=2e-8
    )
    torch.testing.assert_close(torch.rand(3), expected_cpu, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(3, device=device), expected_device, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
def test_entropy_multiple_query_batches_equal_independent_rows_and_keep_all_parameter_graphs(
    device, name
):
    actual, _, X = make_pair(device, name, pending=True)
    raw = (
        X.new_tensor([[[0.19, 0.6]], [[0.43, 0.8]]])
        if "MultiFidelity" in name
        else X.new_tensor([[[0.19]], [[0.43]]])
    )
    query = raw.repeat(3, 1, 1, 1).requires_grad_()
    value = actual(query)
    assert value.shape == (3, 2)
    rows = torch.stack([actual(row) for row in query])
    torch.testing.assert_close(value, rows, rtol=1e-10, atol=1e-11)
    leaves = (query, *actual.parameters())
    gradients = torch.autograd.grad(value.sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradients), leaves)
    assert all(torch.isfinite(g).all() for g in second)


@pytest.mark.parametrize("device", DEVICES)
def test_gibbon_standardized_pending_repulsion_uses_one_consistent_posterior_space(device):
    from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy as Reference
    from botorch.models import SingleTaskGP as ReferenceGP

    X, Y = data(device)
    model = SingleTaskGP(X, 4 * Y, backend="torch")
    ref_model = ReferenceGP(X, 4 * Y)
    pending = X.new_tensor([[0.91]])
    acq = port.qLowerBoundMaxValueEntropy(model, X, X_pending=pending, use_gumbel=False, seed=3)
    reference = Reference(ref_model, X, X_pending=pending, use_gumbel=False)
    maxima = X.new_tensor([[3.0], [4.0], [4.8], [5.6]])
    acq.posterior_max_values = reference.posterior_max_values = maxima
    query = X.new_tensor([[[0.19]], [[0.43]]], requires_grad=True)
    value = acq(query)
    # The official pending term mixes standardized model(...) cross-covariance
    # with untransformed posterior variances. The intended Schur complement is
    # computed from one complete noisy posterior in physical output units.
    reference.set_X_pending(None)
    quality = reference(query)
    joint = torch.cat((query, pending.expand(query.shape[0], 1, 1)), -2)
    covariance = ref_model.posterior(joint, observation_noise=True).distribution.covariance_matrix
    cross, block = covariance[..., :1, 1:], covariance[..., 1:, 1:]
    conditional = covariance[..., :1, :1] - cross @ torch.linalg.solve(
        block, cross.transpose(-1, -2)
    )
    expected = quality + 0.5 * (conditional[..., 0, 0].log() - covariance[..., 0, 0].log())
    torch.testing.assert_close(value, expected, rtol=2e-8, atol=2e-9)
    reference.set_X_pending(pending)
    assert not torch.allclose(value, reference(query), rtol=1e-4, atol=1e-5)
    actual_gradient = torch.autograd.grad(value.sum(), query)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_samples", [1, 8])
def test_mes_zero_control_variate_variance_has_finite_double_backward(device, num_samples):
    from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy as ReferenceMES
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    X, Y = data(device)
    model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
    sampler = SobolQMCNormalSampler((num_samples,), seed=13)
    sampler.base_samples = X.new_zeros((num_samples, 1, 1))
    acq = port.qMaxValueEntropy(model, X, sampler=sampler, use_gumbel=False, seed=3)
    acq.posterior_max_values = X.new_tensor([[0.75], [1.0], [1.2], [1.4]])
    query = X.new_tensor([[0.23]], requires_grad=True)
    value, gradient = acq.value_and_grad(query)
    assert torch.isfinite(value).all()
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
    reference = ReferenceMES(
        ReferenceGP(X, Y, outcome_transform=None), X, use_gumbel=False, num_y_samples=num_samples
    )
    reference.sampler = ReferenceSampler(torch.Size((num_samples,)), seed=13)
    reference.sampler.base_samples = X.new_zeros((num_samples, 1, 1, 1))
    reference.posterior_max_values = acq.posterior_max_values
    assert not torch.isfinite(reference(query)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_gibbon_reference_multiple_query_batches_repro(device):
    actual, reference, X = make_pair(device, "qLowerBoundMaxValueEntropy")
    query = X.new_tensor([[[0.19]], [[0.43]]]).repeat(3, 1, 1, 1)
    with pytest.raises((AssertionError, RuntimeError)):
        reference(query)
    torch.testing.assert_close(actual(query), torch.stack([actual(row) for row in query]))


@pytest.mark.parametrize("device", DEVICES)
def test_supplied_gaussian_distribution_model_keeps_device_samples_and_gradients(device):
    from botorch_rs.posteriors.gpytorch import GPyTorchPosterior

    X, Y = data(device)

    class GaussianAdapter(torch.nn.Module):
        _tensor_backend = "torch"
        num_outputs = 1
        batch_shape = torch.Size()

        def __init__(self):
            super().__init__()
            self.gp = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)

        def posterior(self, X, **kwargs):
            posterior = self.gp.posterior(X, **kwargs)
            return GPyTorchPosterior(
                distribution=torch.distributions.MultivariateNormal(
                    posterior.mean.squeeze(-1),
                    covariance_matrix=posterior.distribution.covariance_matrix,
                )
            )

    model = GaussianAdapter()
    acq = port.qMaxValueEntropy(model, X, use_gumbel=False, seed=3)
    query = X.new_tensor([[0.23]], requires_grad=True)
    value, gradient = acq.value_and_grad(query)
    assert torch.isfinite(value).all()
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
