"""The generic default approximate constructor builds the port's real Tensor GP."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import ApproximateGPyTorchModel, SingleTaskVariationalGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TaggedDefault(ApproximateGPyTorchModel):
    def __init__(self, *args, label="preserved", **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def marker(self):
        return self.label


class TaggedVariational(SingleTaskVariationalGP):
    def __init__(self, *args, label="kept", **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.forward_gain = torch.nn.Parameter(self._train_X.new_tensor(0.8))
        self.posterior_gain = torch.nn.Parameter(self._train_X.new_tensor(0.9))
        self.calls = []

    def forward(self, X):
        self.calls.append("forward")
        return super().forward(X * self.forward_gain)

    def posterior(self, X, **kwargs):
        self.calls.append("posterior")
        return super().posterior(X * self.posterior_gain, **kwargs)


class DenseKernel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_scale = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.double))

    def forward(self, x, y=None, diag=False):
        y = x if y is None else y
        difference = x - y if diag else x.unsqueeze(-2) - y.unsqueeze(-3)
        return (-0.5 * difference.square().sum(-1) / self.raw_scale.exp()).exp()


class InputScale(torch.nn.Module):
    def __init__(self, like):
        super().__init__()
        self.scale = torch.nn.Parameter(like.new_tensor(0.8))
        self.calls = []

    def forward(self, x):
        self.calls.append((x.shape, self.training))
        return x * self.scale


def _inputs(device, batch=()):
    return (
        torch.tensor(
            [[0.1, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.8]], device=device, dtype=torch.double
        )
        .expand(*batch, 4, 2)
        .clone()
    )


def _initialize(model, outputs):
    strategy = model.model.variational_strategy
    if outputs > 1:
        strategy = strategy.base_variational_strategy
    with torch.no_grad():
        strategy.variational_params_initialized.fill_(1)
        distribution = strategy._variational_distribution
        distribution.variational_mean.copy_(
            distribution.variational_mean.new_tensor([-0.3, 0.1, 0.4])
        )
        distribution.chol_variational_covar.copy_(
            torch.eye(
                3,
                dtype=distribution.variational_mean.dtype,
                device=distribution.variational_mean.device,
            )
            * 0.8
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_generic_default_inducing_allocation_and_plain_likelihood_match_source(device, outputs):
    pytest.importorskip("botorch")
    from botorch.models.approximate_gp import ApproximateGPyTorchModel as Reference

    x = _inputs(device)
    own = ApproximateGPyTorchModel(train_X=x, num_outputs=outputs, backend="torch")
    reference = Reference(train_X=x, num_outputs=outputs).to(x)
    actual_strategy = own._strategy
    reference_strategy = reference.model.variational_strategy
    if outputs > 1:
        reference_strategy = reference_strategy.base_variational_strategy
    assert actual_strategy.inducing_points.shape[-2] == 1
    torch.testing.assert_close(actual_strategy.inducing_points, reference_strategy.inducing_points)
    assert dict(own.named_parameters()).keys() == dict(reference.named_parameters()).keys()
    for strategy in (actual_strategy, reference_strategy):
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.zero_()
            strategy._variational_distribution.chol_variational_covar.fill_(1)
    torch.testing.assert_close(own.posterior(x).mean, reference.posterior(x).mean)
    torch.testing.assert_close(
        own.posterior(x, observation_noise=True).variance,
        reference.posterior(x, observation_noise=True).variance,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch", [(1, ()), (1, (2,)), (2, ()), (2, (2,))])
def test_default_matches_source_params_q_posterior_elbo_first_graph(device, outputs, batch):
    pytest.importorskip("botorch")
    from botorch.models.approximate_gp import ApproximateGPyTorchModel as Reference
    from gpytorch.mlls import VariationalELBO as ReferenceELBO

    x = _inputs(device, batch)
    points = x[..., [0, 2, 3], :].clone()
    # The generic source inner ignores these targets and keeps explicit m.
    ignored_y = x.new_zeros(7, outputs + 3).requires_grad_()
    actual = ApproximateGPyTorchModel(
        None, None, outputs, x, ignored_y, inducing_points=points, backend="torch"
    )
    reference = Reference(None, None, outputs, x, ignored_y, inducing_points=points).to(x)
    assert actual.num_outputs == reference.num_outputs == outputs
    assert actual._train_Y is None
    assert actual.model._outer is actual and actual._strategy._owner_ref() is actual
    _initialize(actual, outputs)
    _initialize(reference, outputs)
    actual_parameters, reference_parameters = map(
        dict, (actual.named_parameters(), reference.named_parameters())
    )
    assert actual_parameters.keys() == reference_parameters.keys()
    for name, parameter in actual_parameters.items():
        torch.testing.assert_close(parameter, reference_parameters[name])
    assert not any("likelihood" in name and "prior" in name for name, _ in actual.named_buffers())
    query = x.new_tensor([[0.2, 0.6], [0.8, 0.3]]).requires_grad_()
    rq = query.detach().clone().requires_grad_()
    actual_p = actual.posterior(query, observation_noise=True)
    reference_p = reference.posterior(rq, observation_noise=True)
    torch.testing.assert_close(actual_p.mean, reference_p.mean)
    torch.testing.assert_close(
        actual_p.distribution.covariance_matrix,
        reference_p.distribution.covariance_matrix
        if outputs == 1
        else reference_p.distribution.covariance_matrix.reshape(*batch, 2, outputs, 2, outputs)
        .transpose(-3, -4)
        .transpose(-1, -2)
        .reshape(*batch, 2 * outputs, 2 * outputs),
    )
    names = tuple(actual_parameters)
    gradients = []
    for posterior, params in (
        (actual_p, (query, *(actual_parameters[name] for name in names))),
        (reference_p, (rq, *(reference_parameters[name] for name in names))),
    ):
        gradients.append(
            torch.autograd.grad(
                posterior.mean.square().sum() + posterior.variance.sum(), params, allow_unused=True
            )
        )
    for a, b in zip(*gradients, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
    target = x[..., 0].sin()
    if outputs > 1:
        target = target.unsqueeze(-1).expand(*batch, 4, outputs)
    for wrapper in (actual, reference):
        wrapper.train()
    observed = VariationalELBO(actual.likelihood, actual.model, num_data=9, beta=0.4)(
        actual(x), target
    )
    expected = ReferenceELBO(reference.likelihood, reference.model, num_data=9, beta=0.4)(
        reference(x), target
    )
    torch.testing.assert_close(observed, expected)
    first = torch.autograd.grad(
        observed.sum(), tuple(actual_parameters.values()), allow_unused=True
    )
    second = torch.autograd.grad(
        expected.sum(), tuple(reference_parameters[name] for name in names), allow_unused=True
    )
    for a, b in zip(first, second, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)


def _dense_q(model, query):
    points = model._strategy.inducing_points
    inducing = model._augment(points)
    x = model._augment(query)
    kernel = model.model.covar_module
    jitter = model._strategy.jitter_val
    eye = torch.eye(points.shape[-2], dtype=query.dtype, device=query.device)
    kzz = kernel(inducing, inducing) + jitter * eye
    interpolation = torch.linalg.solve_triangular(
        torch.linalg.cholesky(kzz), kernel(inducing, x), upper=False
    )
    distribution = model._strategy._variational_distribution
    mean = model.model.mean_module(x) + (
        interpolation.mT @ distribution.variational_mean.unsqueeze(-1)
    ).squeeze(-1)
    chol = distribution.chol_variational_covar.tril()
    covariance = kernel(x, x) + interpolation.mT @ (chol @ chol.mT - eye) @ interpolation
    covariance = covariance + jitter * torch.eye(
        query.shape[-2], dtype=query.dtype, device=query.device
    )
    return mean, covariance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_default_full_graph_matches_independent_gaussian_and_no_host_transfers(
    device, outputs, monkeypatch
):
    x = _inputs(device)
    model = ApproximateGPyTorchModel(
        train_X=x, num_outputs=outputs, inducing_points=x[[0, 2, 3]], backend="torch"
    )
    _initialize(model, outputs)
    query = x.new_tensor([[0.23, 0.61], [0.73, 0.32]]).requires_grad_()
    model.eval()
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            raise AssertionError("Tensor inference must not transfer or detach data")

        for name in ("cpu", "numpy", "tolist", "detach"):
            patch.setattr(torch.Tensor, name, forbidden)
        actual = model.posterior(query)
    mean, covariance = _dense_q(model, query)
    expected_mean = mean if outputs == 1 else mean.mT
    expected_variance = covariance.diagonal(dim1=-2, dim2=-1)
    expected_variance = expected_variance if outputs == 1 else expected_variance.mT
    torch.testing.assert_close(
        actual.mean.squeeze(-1) if outputs == 1 else actual.mean, expected_mean
    )
    torch.testing.assert_close(
        actual.variance.squeeze(-1) if outputs == 1 else actual.variance, expected_variance
    )
    parameters = (query, *model.parameters())
    rows = []
    for loss in (
        actual.mean.square().sum() + actual.variance.square().sum(),
        expected_mean.square().sum() + expected_variance.square().sum(),
    ):
        first = torch.autograd.grad(
            loss, parameters, create_graph=True, retain_graph=True, allow_unused=True
        )
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first if g is not None),
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        rows.append((first, second))
    for observed, expected in zip(*rows, strict=True):
        for a, b in zip(observed, expected, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_default_subclass_state_modes_copy_pickle_and_explicit_target_fitting(device, outputs):
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.optim.closures import ForwardBackwardClosure
    from botorch_rs.optim.utils.model_utils import get_parameters

    x = _inputs(device)
    model = TaggedDefault(
        train_X=x, num_outputs=outputs, inducing_points=x[[0, 2, 3]], backend="torch", label="kept"
    )
    _initialize(model, outputs)
    model.kernel_alias = model.model.covar_module
    for replica in (model, copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert replica.marker() == "kept" and isinstance(replica, TaggedDefault)
        assert replica.model._outer is replica and replica._strategy._owner_ref() is replica
        assert replica.kernel_alias is replica.model.covar_module
        replica.eval()
        assert (
            not replica.training and not replica.model.training and not replica.likelihood.training
        )
        for name, value in replica.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name])
        torch.testing.assert_close(replica.posterior(x).mean, model.posterior(x).mean)
        replica.train()
        assert replica.model.training and replica.likelihood.training
    target = x[..., 0].sin()
    if outputs > 1:
        target = target.unsqueeze(-1).expand(4, outputs)
    mll = VariationalELBO(model.likelihood, model.model, num_data=4)
    closure = ForwardBackwardClosure(
        lambda: -mll(model(x), target), get_parameters(mll, requires_grad=True)
    )
    initial = model._strategy._variational_distribution.variational_mean.detach().clone()
    fit_gpytorch_mll(
        mll,
        closure=closure,
        max_attempts=1,
        optimizer_kwargs={"step_limit": 3, "stopping_criterion": None},
    )
    assert not torch.equal(initial, model._strategy._variational_distribution.variational_mean)
    model.float()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    with pytest.raises(NotImplementedError, match="output indices"):
        model.posterior(x.float(), output_indices=[0])


def test_default_native_and_tensor_configuration_selection():
    from botorch_rs.variational import (
        MeanFieldVariationalDistribution,
        UnwhitenedVariationalStrategy,
    )

    x = _inputs("cpu")
    native = ApproximateGPyTorchModel(train_X=x.tolist(), inducing_points=x[:3].tolist())
    assert getattr(native, "_tensor_backend", None) != "torch"
    for options in (
        {"train_X": x.requires_grad_()},
        {"train_X": x.detach().expand(2, 4, 2)},
        {"train_X": x.detach(), "variational_distribution": MeanFieldVariationalDistribution(3)},
        {"train_X": x.detach(), "variational_strategy": UnwhitenedVariationalStrategy},
        {"train_X": x.detach(), "covar_module": DenseKernel()},
    ):
        model = ApproximateGPyTorchModel(inducing_points=x[:3].detach(), **options)
        assert isinstance(model, torch.nn.Module) and model._tensor_backend == "torch"
    builtin = SingleTaskVariationalGP(x.detach(), inducing_points=x[:3].detach(), backend="torch")
    restored = pickle.loads(pickle.dumps(builtin))
    assert restored.model._outer is restored and restored._strategy._owner_ref() is restored


@pytest.mark.parametrize("device", DEVICES)
def test_generic_attached_transform_only_updates_through_actual_forward(device):
    x = _inputs(device).requires_grad_()
    model = ApproximateGPyTorchModel(train_X=x, inducing_points=x[[0, 2, 3]].detach())
    _initialize(model, 1)
    transform = InputScale(x)
    model.input_transform = transform
    query = x.new_tensor([[0.2, 0.6], [0.8, 0.3]]).requires_grad_()
    actual = model.posterior(query)
    assert model.input_transform is transform
    assert transform.calls == [(query.shape, False)]
    mean, covariance = _dense_q(model, query * transform.scale)
    torch.testing.assert_close(actual.mean.squeeze(-1), mean)
    torch.testing.assert_close(actual.variance.squeeze(-1), covariance.diagonal(dim1=-2, dim2=-1))
    model.train()
    model(query)
    assert transform.calls[-1] == (query.shape, True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_default_explicit_target_elbo_matches_independent_gaussian_hessian(device, outputs):
    import math

    x = _inputs(device)
    model = ApproximateGPyTorchModel(
        train_X=x, num_outputs=outputs, inducing_points=x[[0, 2, 3]], backend="torch"
    )
    _initialize(model, outputs)
    query = x.new_tensor([[0.23, 0.61], [0.73, 0.32]]).requires_grad_()
    target = x.new_full((2,), 0.3) if outputs == 1 else x.new_full((2, outputs), 0.3)
    target.requires_grad_()
    mll = VariationalELBO(model.likelihood, model.model, num_data=9, beta=0.4)
    actual = mll(model(query), target)
    mean, covariance = _dense_q(model, query)
    if outputs == 1:
        noise = model.likelihood.noise
        y = target
    else:
        noise = model.likelihood.noise_matrix().diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
        y = target.mT
    log_likelihood = (
        -0.5
        * (
            ((y - mean).square() + covariance.diagonal(dim1=-2, dim2=-1)) / noise
            + noise.log()
            + math.log(2 * math.pi)
        ).sum()
        / query.shape[-2]
    )
    distribution = model._strategy._variational_distribution
    chol = distribution.chol_variational_covar.tril()
    kl = (
        0.5
        * (
            distribution.variational_mean.square().sum(-1)
            + chol.square().sum((-1, -2))
            - 3
            - 2 * chol.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
        ).sum()
    )
    expected = log_likelihood - 0.4 / 9 * kl + model._log_prior() / 9
    torch.testing.assert_close(actual, expected)
    parameters = (query, target, *model.parameters())
    rows = []
    for value in (actual, expected):
        first = torch.autograd.grad(value, parameters, create_graph=True, retain_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), parameters, retain_graph=True
        )
        rows.append((first, second))
    for observed, reference in zip(*rows, strict=True):
        for a, b in zip(observed, reference, strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_single_task_variational_subclass_factory_virtual_law_and_state(device):
    x = _inputs(device)
    options = {} if device == "cuda" else {"backend": "torch"}
    model = TaggedVariational(
        x, x[..., :1].sin(), inducing_points=x[[0, 2, 3]], label="actual", **options
    )
    assert model._tensor_backend == "torch" and isinstance(model, TaggedVariational)
    _initialize(model, 1)
    query = x.new_tensor([[0.2, 0.6], [0.7, 0.3]]).requires_grad_()
    actual = model(query)
    expected_mean, _ = _dense_q(model, query * model.forward_gain)
    torch.testing.assert_close(actual.mean, expected_mean)
    assert model.calls[-1] == "forward"
    actual_p = model.posterior(query)
    expected_mean, expected_covariance = _dense_q(model, query * model.posterior_gain)
    torch.testing.assert_close(actual_p.mean.squeeze(-1), expected_mean)
    torch.testing.assert_close(
        actual_p.variance.squeeze(-1), expected_covariance.diagonal(dim1=-2, dim2=-1)
    )
    assert model.calls[-1] == "posterior"
    leaves = (query, model.forward_gain, model.posterior_gain)
    first = torch.autograd.grad(
        actual.mean.square().sum() + actual_p.mean.square().sum(), leaves, create_graph=True
    )
    assert all(
        torch.isfinite(g).all()
        for g in torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    )
    for restored in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert restored.label == "actual" and isinstance(restored, TaggedVariational)
        assert restored.model._outer is restored and restored._strategy._owner_ref() is restored
        torch.testing.assert_close(restored.posterior(query).mean, model.posterior(query).mean)


@pytest.mark.parametrize("device", DEVICES)
def test_default_variational_fit_failure_restores_separate_likelihood(device):
    from botorch_rs.fit import fit_gpytorch_mll

    x = _inputs(device)
    model = ApproximateGPyTorchModel(train_X=x, inducing_points=x[:3], backend="torch")
    mll = VariationalELBO(model.likelihood, model.model, num_data=4)
    before = {name: value.detach().clone() for name, value in mll.named_parameters()}

    def fail(mll, closure=None, **kwargs):
        with torch.no_grad():
            for value in mll.parameters():
                value.add_(1)
        raise RuntimeError("injected likelihood mutation")

    with pytest.raises(RuntimeError, match="injected likelihood mutation"):
        fit_gpytorch_mll(mll, optimizer=fail, closure=lambda: (), max_attempts=1)
    for name, value in mll.named_parameters():
        torch.testing.assert_close(value, before[name])
