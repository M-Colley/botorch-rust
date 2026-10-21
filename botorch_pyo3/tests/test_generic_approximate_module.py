"""Caller-owned approximate modules retain their actual law and training protocol."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import importlib.util
import pickle

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import ApproximateGPyTorchModel
from botorch_rs.posteriors.transformed import TransformedPosterior

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _Strategy(torch.nn.Module):
    def __init__(self, batch, like):
        super().__init__()
        self.raw_kl = torch.nn.Parameter(like.new_full((*batch, 1), 0.3))

    def kl_divergence(self):
        return 0.5 * self.raw_kl.square().sum(-1)


class _DenseApproximate(torch.nn.Module):
    def __init__(self, device="cpu", batch=()):
        super().__init__()
        self.batch_shape = torch.Size(batch)
        self.scale = torch.nn.Parameter(
            torch.full((*batch, 1), 0.4, dtype=torch.double, device=device)
        )
        self.location = torch.nn.Parameter(
            torch.full((*batch, 1), 0.2, dtype=torch.double, device=device)
        )
        self.variational_strategy = _Strategy(batch, self.scale)
        self.register_buffer("index", torch.tensor(2, device=device))
        self.register_buffer("flag", torch.tensor(True, device=device))

    def forward(self, x):
        mean = self.location + self.scale * x[..., 0].sin()
        covariance = (-0.5 * (x.unsqueeze(-2) - x.unsqueeze(-3)).square().sum(-1)).exp()
        covariance = covariance + 0.2 * torch.eye(x.shape[-2], device=x.device, dtype=x.dtype)
        return torch.distributions.MultivariateNormal(mean, covariance)


class _Likelihood(torch.nn.Module):
    def __init__(self, like):
        super().__init__()
        self.raw_noise = torch.nn.Parameter(like.new_tensor(-0.3))

    def forward(self, distribution):
        return torch.distributions.Normal(
            distribution.mean, (distribution.variance + self.raw_noise.exp()).sqrt()
        )

    def expected_log_prob(self, target, function, multiplier=1):
        noise = self.raw_noise.exp()
        return (
            multiplier
            * -0.5
            * (
                ((target - function.mean).square() + function.variance) / noise
                + noise.log()
                + torch.log(noise.new_tensor(2 * torch.pi))
            )
        )


class _PortGaussianApproximate(_DenseApproximate):
    def forward(self, x):
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

        distribution = super().forward(x)
        return TorchGPyTorchPosterior(
            distribution.mean.unsqueeze(-1), distribution.covariance_matrix
        ).distribution


class _Input(torch.nn.Module):
    def __init__(self, like):
        super().__init__()
        self.scale = torch.nn.Parameter(like.new_tensor(0.8))

    def forward(self, x):
        return x * self.scale


class _Outcome(torch.nn.Module):
    def __init__(self, like):
        super().__init__()
        self.scale = torch.nn.Parameter(like.new_tensor(1.3))

    def untransform_posterior(self, posterior, X=None):
        scale = self.scale.clone()
        return TransformedPosterior(
            posterior, lambda s: s * scale, lambda m, v: m * scale, lambda m, v: v * scale.square()
        )


class TaggedApproximate(ApproximateGPyTorchModel):
    def __init__(self, *args, label="kept", **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def method(self):
        return self.label


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
def test_port_only_module_identity_state_modes_full_graph_and_pickle(device, batch, monkeypatch):
    supplied = _DenseApproximate(device, batch)
    likelihood = _Likelihood(supplied.scale)
    model = TaggedApproximate(supplied, likelihood=likelihood, label="actual")
    assert isinstance(model, TaggedApproximate) and isinstance(model, torch.nn.Module)
    assert model.model is supplied and model.likelihood is likelihood
    assert model.num_outputs == 1 and model.batch_shape == supplied.batch_shape
    assert model.method() == "actual"
    assert {id(p) for p in model.parameters()} == {
        id(p) for p in (*supplied.parameters(), *likelihood.parameters())
    }
    assert {torch.int64, torch.bool} <= model.dtypes_of_buffers
    for replica in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert replica.method() == "actual" and replica.model is not supplied
        assert replica.state_dict().keys() == model.state_dict().keys()
    model.input_transform = _Input(supplied.scale)
    model.outcome_transform = _Outcome(supplied.scale)
    x = supplied.scale.new_tensor([[0.2], [0.6]]).requires_grad_()
    model.train()
    torch.testing.assert_close(model(x).mean, supplied(model.input_transform(x)).mean)
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            raise AssertionError("Tensor forward must not transfer or detach data")

        for name in ("cpu", "numpy", "tolist", "detach"):
            patch.setattr(torch.Tensor, name, forbidden)
        posterior = model.posterior(x, observation_noise=True)
    assert not model.training and not supplied.training and not likelihood.training
    expected = likelihood(supplied(model.input_transform(x)))
    torch.testing.assert_close(posterior.mean, expected.mean * model.outcome_transform.scale)
    torch.testing.assert_close(
        posterior.variance, expected.variance * model.outcome_transform.scale.square()
    )
    leaves = (x, *model.parameters())
    gradients = torch.autograd.grad(
        posterior.mean.square().sum() + posterior.variance.sum(),
        leaves,
        create_graph=True,
        allow_unused=True,
    )
    higher = torch.autograd.grad(
        sum(g.square().sum() for g in gradients if g is not None), leaves, allow_unused=True
    )
    assert all(g is None or torch.isfinite(g).all() for g in (*gradients, *higher))
    model.float()
    assert model.model is supplied and all(p.dtype == torch.float32 for p in model.parameters())


def _source_pair(device, outputs=1, batch=(), likelihood_kind="supplied"):
    gpytorch = pytest.importorskip("gpytorch")
    pytest.importorskip("botorch")
    from botorch.models.approximate_gp import ApproximateGPyTorchModel as Reference

    class GP(gpytorch.models.ApproximateGP):
        def __init__(self):
            shape = torch.Size(batch) + ((outputs,) if outputs > 1 else ())
            points = torch.tensor([[0.1], [0.5], [0.9]], device=device, dtype=torch.double)
            q = gpytorch.variational.CholeskyVariationalDistribution(
                3, batch_shape=shape, mean_init_std=0
            )
            strategy = gpytorch.variational.VariationalStrategy(self, points, q)
            if outputs > 1:
                strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                    strategy, num_tasks=outputs
                )
            super().__init__(strategy)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=shape)
            self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=shape)
            self.to(points)
            with torch.no_grad():
                strategy = self.variational_strategy
                if outputs > 1:
                    strategy = strategy.base_variational_strategy
                strategy.variational_params_initialized.fill_(1)
                q.variational_mean.copy_(points.new_tensor([-0.3, 0.1, 0.4]))
                q.chol_variational_covar.copy_(
                    torch.eye(3, dtype=points.dtype, device=points.device) * 0.8
                )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    module = GP()
    if likelihood_kind == "default":
        likelihood = None
    elif likelihood_kind == "nonGaussian":
        likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(
            device=device, dtype=torch.double
        )
    elif outputs > 1:
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=outputs).to(
            device=device, dtype=torch.double
        )
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=torch.double)
    own = ApproximateGPyTorchModel(module, likelihood, num_outputs=outputs)
    reference = Reference(copy.deepcopy(module), copy.deepcopy(likelihood), num_outputs=outputs).to(
        device=device, dtype=torch.double
    )
    return own, reference


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "outputs,batch,query_batch", [(1, (), ()), (1, (2,), (3, 1)), (2, (), (3,)), (2, (2,), (3, 1))]
)
@pytest.mark.parametrize("likelihood_kind", ["default", "supplied", "nonGaussian"])
def test_actual_approximate_gp_matches_source_wrapper_values_and_graphs(
    device, outputs, batch, query_batch, likelihood_kind
):
    own, reference = _source_pair(device, outputs, batch, likelihood_kind)
    x = (
        torch.tensor([[0.2], [0.6]], device=device, dtype=torch.double)
        .expand(*query_batch, 2, 1)
        .clone()
        .requires_grad_()
    )
    rx = x.detach().clone().requires_grad_()
    first, second = [], []
    names = tuple(dict(own.named_parameters()))
    for wrapper, query in ((own, x), (reference, rx)):
        posterior = wrapper.posterior(query, observation_noise=True)
        first.append((posterior.mean, posterior.variance))
        parameters = dict(wrapper.named_parameters())
        params = (query, *(parameters[name] for name in names))
        gradient = torch.autograd.grad(
            posterior.mean.square().sum() + posterior.variance.sum(),
            params,
            create_graph=True,
            allow_unused=True,
        )
        higher = torch.autograd.grad(
            sum(g.square().sum() for g in gradient if g is not None), params, allow_unused=True
        )
        second.append((gradient, higher))
    assert own.num_outputs == outputs
    for a, b in zip(*first, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-7, atol=2e-8)
    for actual, expected in zip(*second, strict=True):
        for a, b in zip(actual, expected, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_actual_generic_elbo_training_closure_and_dataloader(device):
    from functools import partial

    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.optim.closures import get_loss_closure_with_grads
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch
    from botorch_rs.optim.utils.model_utils import get_parameters

    supplied = _DenseApproximate(device)
    likelihood = _Likelihood(supplied.scale)
    wrapped = ApproximateGPyTorchModel(supplied, likelihood)
    mll = VariationalELBO(likelihood, supplied, num_data=7, beta=0.4)
    assert mll.model is supplied and wrapped.model is supplied and mll.likelihood is likelihood
    x = supplied.scale.new_tensor([[0.1], [0.3], [0.7]])
    y = supplied.scale.new_tensor([0.3, 0.6, 0.2])
    output = supplied(x)
    expected = (
        likelihood.expected_log_prob(y, output).mean(-1)
        - 0.4 / 7 * supplied.variational_strategy.kl_divergence()
    )
    torch.testing.assert_close(mll(output, y), expected)
    multiplier = supplied.scale.new_tensor(0.7, requires_grad=True)
    torch.testing.assert_close(
        mll(output, y, multiplier=multiplier),
        likelihood.expected_log_prob(y, output, multiplier).mean(-1)
        - 0.4 / 7 * supplied.variational_strategy.kl_divergence(),
    )
    loader = [(x[:2], y[:2]), (x[2:], y[2:])]
    closure = get_loss_closure_with_grads(mll, get_parameters(mll, requires_grad=True), loader)
    first = closure()[0]
    second = closure()[0]
    assert not torch.allclose(first, second)
    start = supplied.location.detach().clone()
    fitted = fit_gpytorch_mll(
        mll,
        data_loader=loader,
        optimizer=partial(fit_gpytorch_mll_torch, step_limit=4, stopping_criterion=None),
        max_attempts=1,
    )
    assert fitted is mll and not torch.equal(start, supplied.location)
    assert not mll.training and not supplied.training and not likelihood.training


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("combine", [False, True])
def test_generic_elbo_actual_kl_priors_added_losses_and_first_higher_graphs(
    device, outputs, combine
):
    gpytorch = pytest.importorskip("gpytorch")
    own, _ = _source_pair(device, outputs, (), "supplied")
    model = own.model
    model.covar_module.register_prior(
        "ls_prior", gpytorch.priors.NormalPrior(0.4, 0.7), "lengthscale"
    )

    class Added(gpytorch.mlls.AddedLossTerm):
        def loss(self):
            return 0.03 * model.mean_module.constant.square().sum()

    model.register_added_loss_term("regularizer")
    model.update_added_loss_term("regularizer", Added())
    actual = VariationalELBO(own.likelihood, model, num_data=11, beta=0.4, combine_terms=combine)
    reference = gpytorch.mlls.VariationalELBO(
        own.likelihood, model, num_data=11, beta=0.4, combine_terms=combine
    )
    x = next(model.parameters()).new_tensor([[0.2], [0.4], [0.6], [0.8]]).requires_grad_()
    y = torch.sin(x[..., 0])
    if outputs > 1:
        y = torch.stack((y, y + 0.1), -1)
    y = y.detach().requires_grad_()
    model.train()
    output = model(x)
    observed, expected = actual(output, y), reference(output, y)
    if combine:
        torch.testing.assert_close(observed, expected)
        observed_loss, expected_loss = observed.sum(), expected.sum()
    else:
        assert len(observed) == len(expected) == 4
        for a, b in zip(observed, expected, strict=True):
            torch.testing.assert_close(a, b)
        observed_loss = (observed[0] - observed[1] + observed[2] - observed[3]).sum()
        expected_loss = (expected[0] - expected[1] + expected[2] - expected[3]).sum()
    params = (x, y, *actual.parameters())
    first = torch.autograd.grad(observed_loss, params, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(expected_loss, params, create_graph=True, retain_graph=True)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b)
    higher = torch.autograd.grad(sum(g.square().sum() for g in first), params, retain_graph=True)
    source_higher = torch.autograd.grad(sum(g.square().sum() for g in second), params)
    for a, b in zip(higher, source_higher, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_generic_tensor_scipy_fit_and_complete_failure_rollback(device):
    pytest.importorskip("scipy")
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.optim.closures import ForwardBackwardClosure
    from botorch_rs.optim.fit import fit_gpytorch_mll_scipy
    from botorch_rs.optim.utils.model_utils import get_parameters

    supplied = _DenseApproximate(device)
    likelihood = _Likelihood(supplied.scale)
    mll = VariationalELBO(likelihood, supplied, num_data=5)
    x = supplied.scale.new_tensor([[0.1], [0.4], [0.8]])
    y = supplied.scale.new_tensor([0.2, 0.6, 0.3])
    parameters = get_parameters(mll, requires_grad=True)

    def loss():
        return -mll(supplied(x), y)

    closure = ForwardBackwardClosure(loss, parameters)
    before = float(loss().detach())
    result = fit_gpytorch_mll_scipy(mll, closure=closure, options={"maxiter": 4})
    assert result.step > 0 and float(loss().detach()) < before
    snapshots = {name: parameter.detach().clone() for name, parameter in mll.named_parameters()}

    def fail(mll, closure=None, **kwargs):
        with torch.no_grad():
            for parameter in mll.parameters():
                parameter.add_(2)
        raise RuntimeError("injected failure after changing model and likelihood")

    with pytest.raises(RuntimeError, match="injected failure"):
        fit_gpytorch_mll(mll, optimizer=fail, closure=closure, max_attempts=1)
    for name, parameter in mll.named_parameters():
        torch.testing.assert_close(parameter, snapshots[name])


@pytest.mark.parametrize("device", DEVICES)
def test_attached_transform_modules_and_posterior_scalarization_match_source(device):
    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform

    pytest.importorskip("botorch")
    from botorch.acquisition.objective import ScalarizedPosteriorTransform as ReferenceTransform

    own, reference = _source_pair(device, outputs=2)
    for wrapper in (own, reference):
        wrapper.input_transform = _Input(next(wrapper.parameters()))
    x = next(own.parameters()).new_tensor([[0.23], [0.67]]).requires_grad_()
    rx = x.detach().clone().requires_grad_()
    weights = x.new_tensor([0.3, 0.7], requires_grad=True)
    ref_weights = weights.detach().clone().requires_grad_()
    observed = own.posterior(
        x, posterior_transform=ScalarizedPosteriorTransform(weights, offset=0.1)
    )
    expected = reference.posterior(
        rx, posterior_transform=ReferenceTransform(ref_weights, offset=0.1)
    )
    torch.testing.assert_close(observed.mean, expected.mean)
    torch.testing.assert_close(observed.variance, expected.variance)
    rows = []
    names = tuple(dict(own.named_parameters()))
    for wrapper, query, w, posterior in (
        (own, x, weights, observed),
        (reference, rx, ref_weights, expected),
    ):
        params = (query, w, *(dict(wrapper.named_parameters())[name] for name in names))
        first = torch.autograd.grad(
            posterior.mean.square().sum() + posterior.variance.sum(),
            params,
            create_graph=True,
            allow_unused=True,
        )
        higher = torch.autograd.grad(
            sum(g.square().sum() for g in first if g is not None), params, allow_unused=True
        )
        rows.append((first, higher))
    for first, second in zip(*rows, strict=True):
        for a, b in zip(first, second, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_port_transform_specs_keep_fixed_state_on_supplied_module(device):
    from botorch_rs.models.transforms.input import Normalize
    from botorch_rs.models.transforms.outcome import Standardize

    supplied = _PortGaussianApproximate(device)
    model = ApproximateGPyTorchModel(supplied, _Likelihood(supplied.scale))
    input_transform = Normalize(1, bounds=[[0.1], [0.9]])
    model.input_transform = input_transform
    transform = Standardize(1)
    transform(torch.tensor([[0.2], [0.6], [0.8]], dtype=torch.double))
    model.outcome_transform = transform
    query = supplied.scale.new_tensor([[0.2], [0.6]]).requires_grad_()
    actual = model.posterior(query)
    raw = supplied((query - 0.1) / 0.8)
    scale = model.outcome_transform.stdvs
    mean = model.outcome_transform.means
    torch.testing.assert_close(actual.mean, raw.mean.unsqueeze(-1) * scale + mean)
    first = torch.autograd.grad(actual.mean.square().sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), query)[0]).all()
    assert "input_transform._offset" in model.state_dict()
    assert "outcome_transform.stdvs" in model.state_dict()
    assert model.input_transform is input_transform
    state = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        input_transform._offset.add_(0.2)
    assert not torch.equal(input_transform._offset, state["input_transform._offset"])
    model.load_state_dict(state)
    torch.testing.assert_close(model.posterior(query).mean, actual.mean)
    if importlib.util.find_spec("botorch") is not None:
        from botorch.models.transforms.input import Normalize as SourceNormalize

        source = SourceNormalize(1, bounds=query.new_tensor([[0.1], [0.9]]))
        assert source.state_dict().keys() == input_transform.state_dict().keys()


@pytest.mark.parametrize("device", DEVICES)
def test_default_optimizer_with_supplied_closure_needs_no_fabricated_training_data(device):
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.optim.closures import ForwardBackwardClosure
    from botorch_rs.optim.utils.model_utils import get_parameters

    supplied = _DenseApproximate(device)
    mll = VariationalELBO(_Likelihood(supplied.scale), supplied, num_data=4)
    x = supplied.scale.new_tensor([[0.2], [0.5]])
    y = supplied.scale.new_tensor([0.3, 0.7])
    closure = ForwardBackwardClosure(
        lambda: -mll(supplied(x), y), get_parameters(mll, requires_grad=True)
    )
    before = supplied.location.detach().clone()
    fit_gpytorch_mll(
        mll,
        closure=closure,
        optimizer_kwargs={"step_limit": 3, "stopping_criterion": None},
        max_attempts=1,
    )
    assert not torch.equal(supplied.location, before)
    assert not hasattr(supplied, "train_inputs")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("solver", ["torch", "scipy"])
def test_actual_untransformed_constraints_preserve_element_bounds_and_fitting(device, solver):
    gpytorch = pytest.importorskip("gpytorch")
    if solver == "scipy":
        pytest.importorskip("scipy")
    from functools import partial

    from botorch_rs.optim.closures import ForwardBackwardClosure
    from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
    from botorch_rs.optim.utils.model_utils import get_parameters_and_bounds

    class Constrained(gpytorch.Module):
        def __init__(self):
            super().__init__()
            self.raw_coeff = torch.nn.Parameter(
                torch.tensor([0.2, 0.5], device=device, dtype=torch.double)
            )
            self.raw_bias = torch.nn.Parameter(self.raw_coeff.new_tensor(0.2))
            self.raw_transformed = torch.nn.Parameter(self.raw_coeff.new_tensor(-0.4))
            self.register_constraint(
                "raw_coeff",
                gpytorch.constraints.Interval(
                    self.raw_coeff.new_tensor([0.1, 0.4]),
                    self.raw_coeff.new_tensor([0.3, 0.7]),
                    transform=None,
                ),
            )
            self.register_constraint(
                "raw_bias", gpytorch.constraints.Interval(0.0, 0.6, transform=None)
            )
            self.register_constraint("raw_transformed", gpytorch.constraints.Positive())
            self.variational_strategy = _Strategy((), self.raw_coeff)
            self.to(self.raw_coeff)

    model = Constrained()
    mll = VariationalELBO(_Likelihood(model.raw_coeff), model, num_data=3)
    parameters, bounds = get_parameters_and_bounds(mll)
    direct_parameters, direct_bounds = get_parameters_and_bounds(model)
    key = "model.raw_coeff"
    assert bounds[key][0] is model.raw_coeff_constraint.lower_bound
    assert bounds[key][1] is model.raw_coeff_constraint.upper_bound
    assert bounds[key][0].device == model.raw_coeff.device
    assert "model.raw_transformed" not in bounds
    assert direct_parameters["raw_coeff"] is model.raw_coeff
    assert direct_bounds["raw_coeff"][0] is bounds[key][0]
    if importlib.util.find_spec("botorch") is not None:
        from botorch.optim.utils.model_utils import get_parameters_and_bounds as source_bounds

        source_parameters, reference = source_bounds(model)
        assert {f"model.{name}" for name in source_parameters} <= parameters.keys()
        for name, (lower, upper) in bounds.items():
            torch.testing.assert_close(lower, reference[name.removeprefix("model.")][0])
            torch.testing.assert_close(upper, reference[name.removeprefix("model.")][1])
    selected = {key: model.raw_coeff, "model.raw_bias": model.raw_bias}
    closure = ForwardBackwardClosure(
        lambda: (model.raw_coeff - 2).square().sum() + (model.raw_bias + 1).square(),
        selected,
    )
    if solver == "torch":
        result = fit_gpytorch_mll_torch(
            mll,
            parameters=selected,
            closure=closure,
            optimizer=partial(torch.optim.SGD, lr=1),
            step_limit=3,
            stopping_criterion=None,
        )
    else:
        result = fit_gpytorch_mll_scipy(
            mll, parameters=selected, closure=closure, options={"maxiter": 5}
        )
    assert result.step > 0
    torch.testing.assert_close(model.raw_coeff, model.raw_coeff_constraint.upper_bound)
    torch.testing.assert_close(model.raw_bias, model.raw_bias_constraint.lower_bound)


@pytest.mark.parametrize("device", DEVICES)
def test_frozen_generic_module_fit_evaluates_actual_supplied_closure(device):
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    supplied = _DenseApproximate(device)
    mll = VariationalELBO(_Likelihood(supplied.scale), supplied, num_data=4)
    mll.requires_grad_(False)
    calls = []

    def closure(offset):
        calls.append(offset)
        return supplied.scale.new_tensor(offset), ()

    result = fit_gpytorch_mll_torch(mll, closure=closure, closure_kwargs={"offset": 2.3})
    assert result.step == 0 and result.fval == pytest.approx(2.3)
    assert calls == [2.3]
    assert not hasattr(supplied, "train_inputs")
