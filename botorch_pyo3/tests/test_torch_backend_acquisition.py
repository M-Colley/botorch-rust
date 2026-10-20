"""Tensor-backend acquisition formulas, device residency, and higher derivatives."""

import importlib
import warnings

import pytest
import torch
from torch import nn

from botorch_rs.posteriors.gpytorch import GPyTorchPosterior

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
ANALYTIC = [
    ("ExpectedImprovement", {"best_f": 0.25}),
    ("LogExpectedImprovement", {"best_f": 0.25}),
    ("ProbabilityOfImprovement", {"best_f": 0.25}),
    ("LogProbabilityOfImprovement", {"best_f": 0.25}),
    ("UpperConfidenceBound", {"beta": 0.7}),
    ("PosteriorMean", {}),
    ("PosteriorStandardDeviation", {}),
]
MC = [
    ("qExpectedImprovement", {"best_f": 0.25}),
    ("qLogExpectedImprovement", {"best_f": 0.25}),
    ("qProbabilityOfImprovement", {"best_f": 0.25, "tau": 0.08}),
    ("qUpperConfidenceBound", {"beta": 0.7}),
    ("qLowerConfidenceBound", {"beta": 0.7}),
    ("qSimpleRegret", {}),
    ("qPosteriorStandardDeviation", {}),
]


class TensorModel(nn.Module):
    """Independent test-only exact GP, using the port's supplied-distribution API."""

    _tensor_backend = "torch"
    num_outputs = 1
    batch_shape = torch.Size()

    def __init__(self, device):
        super().__init__()
        self.train_X = nn.Parameter(
            torch.tensor([[0.05], [0.35], [0.72], [0.95]], device=device, dtype=torch.double)
        )
        self.train_Y = nn.Parameter(
            torch.tensor([0.12, 0.62, -0.17, 0.31], device=device, dtype=torch.double)
        )
        self.raw_lengthscale = nn.Parameter(torch.tensor(-0.9, device=device, dtype=torch.double))
        self.raw_noise = nn.Parameter(torch.tensor(-3.5, device=device, dtype=torch.double))

    def posterior(self, X, posterior_transform=None):
        def kernel(a, b):
            return (
                -0.5
                * ((a.unsqueeze(-2) - b.unsqueeze(-3)) / self.raw_lengthscale.exp())
                .square()
                .sum(-1)
            ).exp()

        eye = torch.eye(self.train_X.shape[-2], device=X.device, dtype=X.dtype)
        train = kernel(self.train_X, self.train_X) + self.raw_noise.exp() * eye
        cross = kernel(X, self.train_X)
        mean = cross @ torch.linalg.solve(train, self.train_Y)
        covariance = kernel(X, X) - cross @ torch.linalg.solve(train, cross.transpose(-1, -2))
        covariance = covariance + torch.eye(X.shape[-2], device=X.device, dtype=X.dtype) * 1e-8
        posterior = GPyTorchPosterior(
            distribution=torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance)
        )
        return posterior if posterior_transform is None else posterior_transform(posterior)


class FixedSampler(nn.Module):
    def __init__(self, q, device, sample_shape=(24,)):
        super().__init__()
        self.sample_shape = torch.Size(sample_shape)
        self.seed = 11
        generator = torch.Generator(device=device).manual_seed(self.seed)
        self.register_buffer(
            "base_samples",
            torch.randn(
                *sample_shape,
                1,
                q,
                device=device,
                dtype=torch.double,
                generator=generator,
            ),
        )

    def forward(self, posterior):
        singleton_batch = (1,) * (len(posterior.base_sample_shape) - 1)
        base_samples = self.base_samples.reshape(*self.sample_shape, *singleton_batch, -1)
        return posterior.rsample_from_base_samples(self.sample_shape, base_samples)


def acquisition(name, reference=False):
    prefix = "botorch" if reference else "botorch_rs"
    module = (
        "logei"
        if name.startswith("qLog")
        else "monte_carlo"
        if name.startswith("q")
        else "analytic"
    )
    return getattr(importlib.import_module(f"{prefix}.acquisition.{module}"), name)


def derivatives(value, query, model):
    leaves = (query, *model.parameters())
    gradients = torch.autograd.grad(
        value.sum(), leaves, create_graph=True, retain_graph=True, allow_unused=True
    )
    first = tuple(
        g if g is not None else leaf * 0 for g, leaf in zip(gradients, leaves, strict=True)
    )
    # A directional derivative of the complete gradient exercises mixed as well
    # as query/parameter/train-data second derivatives.
    gradients = torch.autograd.grad(
        sum(x.square().sum() for x in first),
        leaves,
        retain_graph=True,
        allow_unused=True,
    )
    second = tuple(
        g if g is not None else leaf * 0 for g, leaf in zip(gradients, leaves, strict=True)
    )
    return first, second


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,kwargs", ANALYTIC + MC)
def test_tensor_acquisition_reference_graph_and_device(device, name, kwargs):
    model = TensorModel(device)
    q = 2 if name.startswith("q") else 1
    query = torch.tensor([[0.21], [0.56]][:q], device=device, dtype=torch.double).requires_grad_()
    arguments = dict(kwargs)
    if name.startswith("q"):
        arguments["sampler"] = FixedSampler(q, device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(name)(model, **arguments)
        expected = acquisition(name, True)(model, **arguments)
    assert isinstance(actual, acquisition(name))
    assert isinstance(actual, nn.Module)
    assert actual.model is model
    actual_value, expected_value = actual(query), expected(query)
    assert actual_value.device == query.device
    torch.testing.assert_close(actual_value, expected_value, rtol=2e-10, atol=2e-11)
    for actual_order, expected_order in zip(
        derivatives(actual_value, query, model),
        derivatives(expected_value, query, model),
        strict=True,
    ):
        for actual_gradient, expected_gradient in zip(actual_order, expected_order, strict=True):
            assert actual_gradient.device == query.device
            assert torch.isfinite(actual_gradient).all()
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-7, atol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["qNoisyExpectedImprovement", "qLogNoisyExpectedImprovement"])
def test_noisy_reference_default_incumbent_detachment(device, name):
    model = TensorModel(device)
    baseline = model.train_X.detach()[::2]
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double).requires_grad_()
    arguments = dict(
        X_baseline=baseline,
        sampler=FixedSampler(4, device),
        cache_root=False,
        prune_baseline=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(name)(model, **arguments)
        expected = acquisition(name, True)(model, **arguments)
    value, reference = actual(query), expected(query)
    torch.testing.assert_close(value, reference, rtol=2e-10, atol=2e-11)
    for actual_order, expected_order in zip(
        derivatives(value, query, model),
        derivatives(reference, query, model),
        strict=True,
    ):
        for actual_gradient, expected_gradient in zip(actual_order, expected_order, strict=True):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-7, atol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["qNoisyExpectedImprovement", "qLogNoisyExpectedImprovement"])
def test_noisy_full_joint_gradient_extension_matches_finite_difference(device, name):
    model = TensorModel(device)
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double).requires_grad_()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(name)(
            model,
            X_baseline=model.train_X.detach()[::2],
            sampler=FixedSampler(4, device),
            cache_root=False,
            prune_baseline=False,
            detach_baseline=False,
        )
    value, gradients = actual.value_and_grad(query)
    epsilon = 1e-5
    for index in range(query.numel()):
        direction = torch.zeros_like(query)
        direction.view(-1)[index] = epsilon
        difference = (actual(query + direction) - actual(query - direction)) / (2 * epsilon)
        torch.testing.assert_close(
            gradients.view(-1)[index], difference.squeeze(), rtol=1e-5, atol=1e-6
        )
    assert torch.isfinite(torch.autograd.grad(gradients.square().sum(), query)[0]).all()
    assert value.device == query.device


def test_tensor_acquisition_rejects_unknown_options_and_unsupported_family():
    model = TensorModel("cpu")
    with pytest.raises(TypeError, match="unexpected keyword"):
        acquisition("PosteriorMean")(model, nonexistent=True)
    with pytest.raises(NotImplementedError, match="does not yet support"):
        from botorch_rs.acquisition.analytic import NoisyExpectedImprovement

        NoisyExpectedImprovement(model, X_observed=model.train_X)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name,fat",
    [
        ("qExpectedImprovement", False),
        ("qLogExpectedImprovement", False),
        ("qLogExpectedImprovement", True),
        ("qLogExpectedImprovement", None),
    ],
)
def test_constraints_objective_pending_and_multidimensional_samples(device, name, fat):
    from botorch.acquisition.objective import GenericMCObjective as ReferenceObjective

    from botorch_rs.acquisition.objective import GenericMCObjective

    model = TensorModel(device)
    query = torch.tensor(
        [[[[0.21], [0.56]], [[0.18], [0.81]]]], device=device, dtype=torch.double
    ).requires_grad_()
    pending = torch.tensor([[0.92]], device=device, dtype=torch.double).requires_grad_()
    constraint = (
        (lambda y: (0.4 - y[..., 0]).sigmoid()) if fat is None else lambda y: y[..., 0] - 0.4
    )

    def function(y, X):
        return y[..., 0].square() + 0.07 * X[..., 0]

    arguments = dict(
        best_f=0.1,
        sampler=FixedSampler(3, device, (3, 4)),
        X_pending=pending,
        constraints=[constraint],
        eta=0.2,
    )
    # qEI has canonical sigmoid constraints; qLogEI exposes both tails and
    # probability-valued constraints through its fat option.
    if name.startswith("qLog"):
        arguments["fat"] = [fat]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(name)(model, objective=GenericMCObjective(function), **arguments)
        expected = acquisition(name, True)(
            model, objective=ReferenceObjective(function), **arguments
        )
    assert not actual.X_pending.requires_grad
    value, reference = actual(query), expected(query)
    assert value.shape == torch.Size((1, 2))
    torch.testing.assert_close(value, reference, rtol=2e-10, atol=2e-11)
    for actual_order, expected_order in zip(
        derivatives(value, query, model),
        derivatives(reference, query, model),
        strict=True,
    ):
        for actual_gradient, expected_gradient in zip(actual_order, expected_order, strict=True):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-7, atol=2e-8)
    actual.set_X_pending(None)
    assert actual.X_pending is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("incremental", [False, True])
def test_noisy_pending_constraint_reference(device, incremental):
    model = TensorModel(device)
    query = torch.tensor(
        [[[0.21], [0.56]], [[0.18], [0.81]]], device=device, dtype=torch.double
    ).requires_grad_()
    baseline = model.train_X.detach()[::2]
    pending = torch.tensor([[0.92]], device=device, dtype=torch.double)
    arguments = dict(
        X_baseline=baseline,
        sampler=FixedSampler(5, device),
        cache_root=False,
        prune_baseline=False,
        X_pending=pending,
        incremental=incremental,
        constraints=[lambda y: y[..., 0] - y[..., 0].amin(-1, keepdim=True) - 0.1],
        eta=0.1,
    )
    name = "qLogNoisyExpectedImprovement"
    actual = acquisition(name)(model, **arguments)
    expected = acquisition(name, True)(model, **arguments)
    value, reference = actual(query), expected(query)
    torch.testing.assert_close(value, reference, rtol=2e-10, atol=2e-11)
    for actual_order, expected_order in zip(
        derivatives(value, query, model),
        derivatives(reference, query, model),
        strict=True,
    ):
        for actual_gradient, expected_gradient in zip(actual_order, expected_order, strict=True):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_ei_tails_match_values_and_two_derivatives(device, dtype):
    from botorch.acquisition.analytic import _log_ei_helper

    from botorch_rs.acquisition._torch_backend_acquisition import _log_ei

    u = torch.tensor(
        [-1e7, -1e4, -200, -20, -2, -1.1, -1, -0.9, 0, 20], device=device, dtype=dtype
    ).requires_grad_()
    actual, expected = _log_ei(u), _log_ei_helper(u)
    tolerance = 3e-6 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    first = torch.autograd.grad(actual.sum(), u, create_graph=True)[0]
    reference_first = torch.autograd.grad(expected.sum(), u, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), u)[0]
    reference_second = torch.autograd.grad(reference_first.sum(), u)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    torch.testing.assert_close(first, reference_first, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(second, reference_second, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_forward_and_double_backward_do_not_transfer_tensor_data_to_host(device, monkeypatch):
    import botorch_rs._arrays as arrays

    model = TensorModel(device)
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double).requires_grad_()
    sampler = FixedSampler(2, device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor acquisition performed a host conversion")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(arrays, "to_flat", forbidden)
    actual = acquisition("qLogExpectedImprovement")(model, best_f=0.25, sampler=sampler)
    value, gradient = actual.value_and_grad(query)
    second = torch.autograd.grad(gradient.square().sum(), query)[0]
    assert value.device == query.device and second.device == query.device
    assert torch.isfinite(second).all()


def test_noisy_cache_warning_and_invalid_constraint_temperature():
    model = TensorModel("cpu")
    with pytest.warns(RuntimeWarning, match="full-joint"):
        actual = acquisition("qLogNoisyExpectedImprovement")(
            model,
            X_baseline=model.train_X.detach(),
            cache_root=True,
            prune_baseline=False,
        )
    assert actual.cache_root is False
    constrained = acquisition("qLogExpectedImprovement")(
        model,
        best_f=0.25,
        sampler=FixedSampler(2, "cpu"),
        constraints=[lambda y: y[..., 0]],
        eta=0,
    )
    with pytest.raises(ValueError, match="eta must be positive"):
        constrained(torch.tensor([[0.21], [0.56]], dtype=torch.double))


def native_data(device):
    X = torch.tensor([[0.05], [0.35], [0.72], [0.95]], device=device, dtype=torch.double)
    Y = torch.tensor([[0.12], [0.62], [-0.17], [0.31]], device=device, dtype=torch.double)
    return X, Y


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name,kwargs",
    ANALYTIC + MC + [("qNoisyExpectedImprovement", {}), ("qLogNoisyExpectedImprovement", {})],
)
def test_real_native_torch_gp_acquisition_reference_and_higher_order_graph(device, name, kwargs):
    from botorch.models import SingleTaskGP as ReferenceGP
    from gpytorch import settings

    from botorch_rs.models import SingleTaskGP

    X, Y = (value.requires_grad_() for value in native_data(device))
    model = SingleTaskGP(X, Y, backend="torch")
    reference = ReferenceGP(X, Y)
    model.covar_module.lengthscale = reference.covar_module.lengthscale.detach()
    model.mean_module.constant = reference.mean_module.constant.detach()
    model.likelihood.noise = reference.likelihood.noise.detach()
    q = 2 if name.startswith("q") else 1
    query = torch.tensor([[0.21], [0.56]][:q], device=device, dtype=torch.double).requires_grad_()
    arguments = dict(kwargs)
    if name.startswith("q"):
        noisy = "Noisy" in name
        arguments["sampler"] = FixedSampler(4 if noisy else q, device)
        if noisy:
            arguments.update(X_baseline=X.detach()[::2], cache_root=False, prune_baseline=False)
    with (
        warnings.catch_warnings(),
        settings.detach_test_caches(False),
        settings.fast_pred_var(False),
    ):
        warnings.simplefilter("ignore")
        actual = acquisition(name)(model, **arguments)
        expected = acquisition(name, True)(reference, **arguments)
        value, expected_value = actual(query), expected(query)
        torch.testing.assert_close(value, expected_value, rtol=1e-8, atol=1e-9)
        actual_first = torch.autograd.grad(
            value.sum(), query, create_graph=True, retain_graph=True
        )[0]
        expected_first = torch.autograd.grad(
            expected_value.sum(), query, create_graph=True, retain_graph=True
        )[0]
        torch.testing.assert_close(actual_first, expected_first, rtol=2e-7, atol=2e-8)
        # GPyTorch's sampled-posterior custom backward can miss nonlinear
        # second-order terms. Compare acquisition Hessians using the same port
        # posterior graph; the independent gradgradcheck below verifies the
        # complete inference + acquisition graph against finite differences.
        formula_reference = acquisition(name, True)(model, **arguments)(query)
        formula_first = torch.autograd.grad(
            formula_reference.sum(), query, create_graph=True, retain_graph=True
        )[0]
        actual_second = torch.autograd.grad(actual_first.sum(), query, retain_graph=True)[0]
        expected_second = torch.autograd.grad(formula_first.sum(), query, retain_graph=True)[0]
        torch.testing.assert_close(actual_second, expected_second, rtol=2e-6, atol=2e-7)
    leaves = (X, Y, query, *model.parameters())
    gradients = torch.autograd.grad(
        value.sum(), leaves, create_graph=True, retain_graph=True, allow_unused=True
    )
    differentiable = tuple(g for g in gradients if g is not None and g.requires_grad)
    second = torch.autograd.grad(
        sum(g.square().sum() for g in differentiable), leaves, allow_unused=True
    )
    assert all(
        g is None or (g.device == query.device and torch.isfinite(g).all())
        for g in (*gradients, *second)
    )
    # Re-evaluation builds a fresh graph; no model or acquisition cache consumes
    # transform statistics retained from the first backward.
    assert torch.isfinite(torch.autograd.grad(actual(query).sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name", ["LogExpectedImprovement", "qLogExpectedImprovement", "qLogNoisyExpectedImprovement"]
)
def test_real_native_acquisition_training_and_query_gradcheck(device, name):
    from botorch_rs.models import SingleTaskGP

    X, Y = (value.requires_grad_() for value in native_data(device))
    q = 1 if name == "LogExpectedImprovement" else 2
    query = torch.tensor([[0.21], [0.56]][:q], device=device, dtype=torch.double).requires_grad_()

    def evaluate(train_X, train_Y, candidate):
        model = SingleTaskGP(train_X, train_Y, backend="torch", outcome_transform=None)
        kwargs = {"best_f": 0.25}
        if name.startswith("q"):
            kwargs["sampler"] = FixedSampler(4 if "Noisy" in name else q, device)
        if "Noisy" in name:
            kwargs.pop("best_f")
            kwargs.update(
                X_baseline=train_X[::2],
                cache_root=False,
                prune_baseline=False,
                detach_baseline=False,
            )
        return acquisition(name)(model, **kwargs)(candidate)

    assert torch.autograd.gradcheck(evaluate, (X, Y, query), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (X, Y, query), fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
def test_real_native_default_sampler_acquisition_has_no_host_conversion(device, monkeypatch):
    import botorch_rs._arrays as arrays
    from botorch_rs.models import SingleTaskGP

    X, Y = (value.requires_grad_() for value in native_data(device))
    model = SingleTaskGP(X, Y, backend="torch")
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double).requires_grad_()

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor acquisition performed a host conversion")

    for method in ("cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    monkeypatch.setattr(arrays, "to_flat", forbidden)
    actual = acquisition("qLogExpectedImprovement")(model, best_f=0.25)
    value, gradient = actual.value_and_grad(query)
    assert actual.sampler.base_samples.device == query.device
    second = torch.autograd.grad(
        gradient.square().sum(), (X, Y, query, *model.parameters()), allow_unused=True
    )
    assert all(g is None or (g.device == query.device and torch.isfinite(g).all()) for g in second)
    assert value.device == query.device
