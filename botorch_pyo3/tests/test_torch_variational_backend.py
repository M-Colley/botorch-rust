"""Device-resident whitened SVGP values, parameter graphs and ELBO training."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from botorch.models.approximate_gp import SingleTaskVariationalGP as ReferenceGP

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import RBFKernel
from botorch_rs.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _base(model, outputs):
    strategy = model.model.variational_strategy
    return strategy.base_variational_strategy if outputs > 1 else strategy


def _models(device, outputs, learn=True):
    x = torch.linspace(0.1, 0.9, 16, dtype=torch.double, device=device).reshape(8, 2)
    y = torch.stack((x.sum(-1).sin(), x.sum(-1).cos()), -1)[..., :outputs]
    inputs = [x.clone().requires_grad_(), x.clone().requires_grad_()]
    targets = [y.clone().requires_grad_(), y.clone().requires_grad_()]
    batch = torch.Size([outputs]) if outputs > 1 else torch.Size()
    likelihood = MultitaskGaussianLikelihood(outputs) if outputs > 1 else GaussianLikelihood()
    ref_like = (
        gpytorch.likelihoods.MultitaskGaussianLikelihood(outputs)
        if outputs > 1
        else gpytorch.likelihoods.GaussianLikelihood()
    )
    actual = SingleTaskVariationalGP(
        inputs[0],
        targets[0],
        likelihood=likelihood,
        covar_module=RBFKernel(ard_num_dims=2, batch_shape=batch),
        inducing_points=x[[0, 3, 6]],
        learn_inducing_points=learn,
        backend="torch",
    )
    reference = ReferenceGP(
        inputs[1],
        targets[1],
        likelihood=ref_like.to(x),
        covar_module=gpytorch.kernels.RBFKernel(ard_num_dims=2, batch_shape=batch).to(x),
        inducing_points=x[[0, 3, 6]],
        learn_inducing_points=learn,
    ).to(x)
    for model in (actual, reference):
        model.model.covar_module.lengthscale = x.new_tensor(0.6)
        model.model.mean_module.constant = x.new_tensor(0.2)
        model.likelihood.noise = x.new_tensor(0.15)
        if outputs > 1:
            model.likelihood.task_noises = x.new_tensor(0.03)
        strategy = _base(model, outputs)
        with torch.no_grad():
            distribution = strategy._variational_distribution
            distribution.variational_mean.copy_(
                torch.linspace(
                    -0.1, 0.2, distribution.variational_mean.numel(), device=device, dtype=x.dtype
                ).reshape_as(distribution.variational_mean)
            )
            distribution.chol_variational_covar.copy_(
                torch.eye(3, dtype=x.dtype, device=device) * 0.8
                + torch.tril(torch.ones(3, 3, dtype=x.dtype, device=device), diagonal=-1) * 0.1
            )
            strategy.variational_params_initialized.fill_(1)
    return actual, reference, inputs, targets


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("training", [False, True])
def test_whitened_q_prior_and_elbo_match_official_first_gradients(device, outputs, training):
    model, reference, inputs, targets = _models(device, outputs)
    model.train(training)
    reference.train(training)
    actual, expected = model.model(inputs[0]), reference.model(inputs[1])
    torch.testing.assert_close(actual.mean, expected.mean, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(actual.variance, expected.variance, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        actual.covariance_matrix, expected.covariance_matrix, atol=1e-10, rtol=1e-9
    )
    mll = VariationalELBO(model.likelihood, model.model, num_data=13, beta=0.7, combine_terms=False)
    ref_mll = gpytorch.mlls.VariationalELBO(
        reference.likelihood, reference.model, num_data=13, beta=0.7, combine_terms=False
    )
    target = [value.squeeze(-1) if outputs == 1 else value for value in targets]
    terms, ref_terms = mll(actual, target[0]), ref_mll(expected, target[1])
    for a, b in zip(terms, ref_terms, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    losses = [terms[0] - terms[1] + terms[2], ref_terms[0] - ref_terms[1] + ref_terms[2]]
    parameters = [dict(model.named_parameters()), dict(reference.named_parameters())]
    assert parameters[0].keys() == parameters[1].keys()
    gradients = [
        torch.autograd.grad(
            loss.sum(), [inputs[i], targets[i], *parameters[i].values()], create_graph=True
        )
        for i, loss in enumerate(losses)
    ]
    # Model registration order differs; compare data derivatives then named leaves.
    for a, b in zip(gradients[0][:2], gradients[1][:2], strict=True):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-8)
    named = [
        dict(zip(mapping, grads[2:], strict=True))
        for mapping, grads in zip(parameters, gradients, strict=True)
    ]
    for name in named[0]:
        torch.testing.assert_close(named[0][name], named[1][name], atol=2e-9, rtol=2e-8)
    # The port's full graph supports a second backward through inducing-point
    # locations, covariance solves, q(v), training inputs and training targets.
    hessian = torch.autograd.grad(
        sum(value.square().sum() for value in gradients[0]),
        [inputs[0], targets[0], *parameters[0].values()],
    )
    assert all(torch.isfinite(value).all() for value in hessian)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("learn", [False, True])
def test_variational_posterior_query_graph_and_real_optimizer_minibatches(device, outputs, learn):
    model, reference, inputs, targets = _models(device, outputs, learn)
    model.eval()
    reference.eval()
    query = inputs[0].detach()[[1, 5]].clone().requires_grad_()
    query_ref = query.detach().clone().requires_grad_()
    observed, expected = model.posterior(query), reference.posterior(query_ref)
    torch.testing.assert_close(observed.mean, expected.mean, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(observed.variance, expected.variance, atol=1e-10, rtol=1e-9)
    first = [
        torch.autograd.grad(p.mean.sum() + p.variance.sum(), value, create_graph=True)[0]
        for p, value in [(observed, query), (expected, query_ref)]
    ]
    torch.testing.assert_close(*first, atol=1e-9, rtol=1e-8)
    second = torch.autograd.grad(first[0].square().sum(), query)[0]
    assert torch.isfinite(second).all()
    strategy = _base(model, outputs)
    assert isinstance(strategy.inducing_points, torch.nn.Parameter) is learn
    original = strategy.inducing_points.detach().clone()
    target = targets[0].detach().squeeze(-1) if outputs == 1 else targets[0].detach()
    model.train()
    mll = VariationalELBO(model.likelihood, model.model, num_data=8)
    optimizer = torch.optim.Adam(mll.parameters(), lr=0.005)
    for indices in ([0, 1, 2, 3], [4, 5, 6, 7], [1, 2, 5, 6]):
        optimizer.zero_grad()
        loss = -mll(model.model(inputs[0].detach()[indices]), target[indices]).sum()
        loss.backward()
        optimizer.step()
    assert (
        not torch.equal(strategy.inducing_points, original)
        if learn
        else torch.equal(strategy.inducing_points, original)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_variational_prior_does_not_initialize_rng_and_saved_output_law(device):
    x = torch.rand(8, 2, dtype=torch.double, device=device)
    model = SingleTaskVariationalGP(
        x, x.sum(-1, keepdim=True), inducing_points=x[:3], backend="torch"
    )
    strategy = model.model.variational_strategy
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    prior = model.model.forward(x)
    assert not bool(strategy.variational_params_initialized)
    assert torch.equal(
        rng, torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    )
    mll = VariationalELBO(model.likelihood, model.model, 8)
    with pytest.raises(UnsupportedError, match="q\\(f\\).*(prior|law)"):
        mll(prior, x.sum(-1))
    output = model.model(x)
    mean, variance = output.mean.clone(), output.variance.clone()
    expected_loss = mll(output, x.sum(-1)).clone()
    # Covariance is still lazy: public setters must not change this saved law.
    model.model.covar_module.lengthscale = 0.3
    model.likelihood.noise = 0.2
    torch.testing.assert_close(output.mean, mean)
    torch.testing.assert_close(output.variance, variance)
    torch.testing.assert_close(mll(output, x.sum(-1)), expected_loss)
    assert torch.isfinite(output.covariance_matrix).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("outputs", [1, 2])
def test_batched_variational_data_graphs_and_higher_order_elbo(device, dtype, outputs):
    x = torch.linspace(0.1, 0.9, 32, dtype=dtype, device=device).reshape(2, 8, 2).requires_grad_()
    y = torch.stack((x.sum(-1).sin(), x.sum(-1).cos()), -1)[..., :outputs].detach().requires_grad_()
    model = SingleTaskVariationalGP(
        x, y, inducing_points=x.detach()[..., [0, 3, 6], :], backend="torch"
    )
    model.eval()
    mll = VariationalELBO(model.likelihood, model.model, 8)
    loss = -mll().sum()
    variables = [x, y, *model.parameters()]
    first = torch.autograd.grad(loss, variables, create_graph=True)
    second = torch.autograd.grad(sum(grad.square().sum() for grad in first), variables)
    assert all(torch.isfinite(grad).all() for grad in [*first, *second])
    posterior = model.posterior(x[..., :2, :])
    assert posterior.mean.shape == (2, 2, outputs)
    assert posterior.mean.device == x.device and posterior.mean.dtype == dtype
    assert mll().shape == (2,)


@pytest.mark.parametrize("device", DEVICES)
def test_default_pivoted_cholesky_allocation_and_reinitialization(device):
    generator = torch.Generator().manual_seed(117)
    x = torch.rand(15, 2, generator=generator, dtype=torch.double).to(device)
    model = SingleTaskVariationalGP(x, inducing_points=4, backend="torch")
    reference = ReferenceGP(x, inducing_points=4).to(x)
    strategy = model.model.variational_strategy
    torch.testing.assert_close(
        strategy.inducing_points, reference.model.variational_strategy.inducing_points
    )
    model.model(x)
    held = strategy.inducing_points
    model.init_inducing_points(x.flip(0))
    assert strategy.inducing_points is held and not bool(strategy.variational_params_initialized)
    clone = copy.deepcopy(model)
    assert clone.model._outer is clone
    assert clone.model.variational_strategy._owner_ref() is clone
    with torch.no_grad():
        clone.model.mean_module.constant = 0.7
    assert not torch.equal(clone.model(x).mean, model.model(x).mean)


def test_scalar_elbo_second_derivatives_match_reference_gradient_finite_difference():
    model, reference, inputs, targets = _models("cpu", 1)
    model.eval()
    reference.eval()
    mll = VariationalELBO(model.likelihood, model.model, 8)
    ref_mll = gpytorch.mlls.VariationalELBO(reference.likelihood, reference.model, 8)
    actual_loss = mll(model.model(inputs[0]), targets[0].squeeze(-1))
    actual_first = torch.autograd.grad(actual_loss, inputs[0], create_graph=True)[0]
    direction = torch.linspace(-0.2, 0.3, inputs[0].numel(), dtype=inputs[0].dtype).reshape_as(
        inputs[0]
    )
    actual_hvp = torch.autograd.grad((actual_first * direction).sum(), inputs[0])[0]
    gradients = []
    h = 1e-5
    for sign in (-1, 1):
        query = (inputs[1].detach() + sign * h * direction).requires_grad_()
        value = ref_mll(reference.model(query), targets[1].squeeze(-1))
        gradients.append(torch.autograd.grad(value, query)[0])
    torch.testing.assert_close(
        actual_hvp, (gradients[1] - gradients[0]) / (2 * h), atol=2e-7, rtol=2e-6
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_all_data_and_parameter_elbo_hvps_match_independent_reference_difference(device, outputs):
    model, reference, inputs, targets = _models(device, outputs)
    model.eval()
    reference.eval()
    mll = VariationalELBO(model.likelihood, model.model, 8)
    ref_mll = gpytorch.mlls.VariationalELBO(reference.likelihood, reference.model, 8)
    parameters = dict(model.named_parameters())
    ref_parameters = dict(reference.named_parameters())
    variables = [inputs[0], targets[0], *parameters.values()]
    target = targets[0].squeeze(-1) if outputs == 1 else targets[0]
    value = mll(model.model(inputs[0]), target)
    first = torch.autograd.grad(value, variables, create_graph=True)
    directions = [torch.full_like(item, 0.03 + 0.01 * i) for i, item in enumerate(variables)]
    actual = torch.autograd.grad(
        sum(
            (gradient * direction).sum()
            for gradient, direction in zip(first, directions, strict=True)
        ),
        variables,
    )
    originals = {name: parameter.detach().clone() for name, parameter in ref_parameters.items()}
    direction_map = dict(zip(parameters, directions[2:], strict=True))
    rows = []
    h = 1e-5
    for sign in (-1, 1):
        with torch.no_grad():
            for name, parameter in ref_parameters.items():
                parameter.copy_(originals[name] + sign * h * direction_map[name])
        _base(reference, outputs)._clear_cache()
        x = (inputs[1].detach() + sign * h * directions[0]).requires_grad_()
        y = (targets[1].detach() + sign * h * directions[1]).requires_grad_()
        target = y.squeeze(-1) if outputs == 1 else y
        ref_value = ref_mll(reference.model(x), target)
        row = torch.autograd.grad(ref_value, [x, y, *(ref_parameters[name] for name in parameters)])
        rows.append(row)
    for observed, low, high in zip(actual, *rows, strict=True):
        torch.testing.assert_close(observed, (high - low) / (2 * h), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_multitask_likelihood_conditionals_and_marginal_log_density(device):
    model, reference, inputs, targets = _models(device, 2)
    model.eval()
    reference.eval()
    observed = model.model(inputs[0])
    expected = reference.model(inputs[1])
    marginal = model.likelihood(observed)
    ref_marginal = reference.likelihood(expected)
    torch.testing.assert_close(marginal.mean, ref_marginal.mean)
    torch.testing.assert_close(marginal.variance, ref_marginal.variance)
    torch.testing.assert_close(marginal.log_prob(targets[0]), ref_marginal.log_prob(targets[1]))
    conditional = model.likelihood(observed.mean)
    torch.testing.assert_close(
        conditional.variance,
        (model.likelihood.noise + model.likelihood.task_noises).expand_as(observed.mean),
    )
