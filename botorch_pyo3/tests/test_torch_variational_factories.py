"""The public strategy factory executes the caller's actual component."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import functools
import pickle

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from test_torch_variational_shared_distribution import (
    DenseKernel,
    DenseMean,
    dense_terms,
    loss_terms,
)

from botorch_rs.mlls import VariationalELBO
from botorch_rs.models import SingleTaskVariationalGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class PlainStrategy(gpytorch.variational.VariationalStrategy):
    pass


class DenseStrategy(gpytorch.variational.VariationalStrategy):
    """A caller-defined q law with independent mean parameters and dense algebra."""

    def __init__(self, *args, gain=0.8, shift=0.2, label="custom", **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = torch.nn.Parameter(self.inducing_points.new_tensor(gain))
        self.shift = torch.nn.Parameter(self.inducing_points.new_tensor(shift))
        self.label = label

    def forward(self, x, inducing_points, inducing_values, **kwargs):
        kernel, mean_module = self.model.covar_module, self.model.mean_module
        m, n = inducing_points.shape[-2], x.shape[-2]
        eye = torch.eye(m, dtype=x.dtype, device=x.device)
        root = torch.linalg.cholesky(
            kernel(inducing_points, inducing_points) + self.jitter_val * eye
        )
        interpolation = torch.linalg.solve_triangular(root, kernel(inducing_points, x), upper=False)
        factor = self._variational_distribution.chol_variational_covar.tril()
        covariance_q = factor @ factor.mT
        mean = (
            mean_module(x)
            + self.gain * (interpolation.mT @ inducing_values.unsqueeze(-1)).squeeze(-1)
            + self.shift
        )
        covariance = (
            kernel(x, x)
            + self.jitter_val * torch.eye(n, dtype=x.dtype, device=x.device)
            + interpolation.mT @ ((covariance_q - eye) @ interpolation)
        )
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


def labelled_factory(label, model, **kwargs):
    return DenseStrategy(model, label=label, **kwargs)


class UnhashableFactory:
    __hash__ = None

    def __call__(self, model, **kwargs):
        return DenseStrategy(model, label="callable object", **kwargs)


class RowDoubler(torch.nn.Module):
    def forward(self, X):
        return X.repeat_interleave(2, dim=-2)


def make_model(device, factory, outputs=1, batch=(), *, reference=False):
    x = (
        torch.tensor(
            [[0.1, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.8]],
            device=device,
            dtype=torch.double,
        )
        .expand(*batch, 4, 2)
        .clone()
        .requires_grad_()
    )
    y = torch.cat([x[..., :1].sin() + i * 0.2 for i in range(outputs)], -1)
    y = y.detach().clone().requires_grad_()
    aug = batch + ((outputs,) if outputs > 1 else ())
    distribution = gpytorch.variational.CholeskyVariationalDistribution(
        3, batch_shape=torch.Size(aug), mean_init_std=0
    ).to(x)
    likelihood = (
        gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size(batch))
        if outputs == 1
        else gpytorch.likelihoods.MultitaskGaussianLikelihood(outputs)
    ).to(x)
    cls = SingleTaskVariationalGP
    if reference:
        pytest.importorskip("botorch")
        from botorch.models import SingleTaskVariationalGP as cls
    model = cls(
        x,
        y,
        likelihood=likelihood,
        covar_module=DenseKernel(aug, x),
        mean_module=DenseMean(aug, x),
        variational_strategy=factory,
        variational_distribution=distribution,
        inducing_points=x.new_tensor([[0.12, 0.2], [0.44, 0.51], [0.84, 0.92]]),
        **({} if reference else {"backend": "torch"}),
    ).to(x)
    strategy = model.model.variational_strategy
    if outputs > 1:
        strategy = strategy.base_variational_strategy
    assert strategy._variational_distribution is distribution
    assert model.likelihood is likelihood
    assert strategy.model is model.model
    assert strategy.model._aug_batch_shape == torch.Size(aug)
    with torch.no_grad():
        strategy.variational_params_initialized.fill_(1)
        distribution.variational_mean.copy_(x.new_tensor([-0.2, 0.15, 0.3]))
        distribution.chol_variational_covar.copy_(
            x.new_tensor([[0.8, 0, 0], [0.1, 0.9, 0], [-0.05, 0.12, 0.7]])
        )
        likelihood.noise = x.new_tensor(0.08)
        if outputs > 1:
            likelihood.task_noises = x.new_tensor([0.03, 0.04])
    return model, strategy, x, y


FACTORIES = [
    PlainStrategy,
    DenseStrategy,
    functools.partial(DenseStrategy, gain=0.7, shift=0.3),
    functools.partial(labelled_factory, "partial positional", shift=0.15),
    UnhashableFactory(),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("outputs,batch", [(1, ()), (2, (2,))])
def test_factory_identity_constructor_virtual_law_and_source_first_gradients(
    device, factory, outputs, batch
):
    pytest.importorskip("botorch")
    from gpytorch.mlls import VariationalELBO as ReferenceELBO

    models = [make_model(device, factory, outputs, batch, reference=r) for r in (False, True)]
    values, gradients = [], []
    for reference, (model, strategy, x, y) in enumerate(models):
        assert type(strategy) is (PlainStrategy if factory is PlainStrategy else DenseStrategy)
        if isinstance(strategy, DenseStrategy):
            assert (
                dict(model.named_parameters())[
                    "model.variational_strategy."
                    + ("base_variational_strategy." if outputs > 1 else "")
                    + "shift"
                ]
                is strategy.shift
            )
        inputs = x.unsqueeze(-3) if outputs > 1 and batch else x
        output = model.model(inputs)
        mll = (ReferenceELBO if reference else VariationalELBO)(
            model.likelihood, model.model, 11, beta=0.7
        )
        elbo = mll(output, y.squeeze(-1) if outputs == 1 else y)
        values.append((output.mean, output.variance, output.covariance_matrix, elbo))
        parameters = dict(model.named_parameters())
        gradients.append(
            torch.autograd.grad(
                loss_terms(output.mean, output.covariance_matrix, elbo),
                (*(parameters[n] for n in sorted(parameters)), x, y),
            )
        )
    assert (
        dict(models[0][0].named_parameters()).keys() == dict(models[1][0].named_parameters()).keys()
    )
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    query = models[0][2].detach()
    for observed in (False, True):
        actual, expected = [
            model.posterior(query, observation_noise=observed) for model, *_ in models
        ]
        torch.testing.assert_close(actual.mean, expected.mean)
        torch.testing.assert_close(actual.variance, expected.variance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch", [(1, ()), (2, (2,))])
def test_custom_strategy_complete_hvp_matches_independent_dense_gaussian(device, outputs, batch):
    model, strategy, x, y = make_model(device, DenseStrategy, outputs, batch)
    model.eval()
    inputs = x.unsqueeze(-3) if outputs > 1 and batch else x
    output = model.model(inputs)
    actual = VariationalELBO(model.likelihood, model.model, 11, beta=0.7)(
        output, y.squeeze(-1) if outputs == 1 else y
    )
    base_mean, covariance, expected = dense_terms(model, strategy, x, y)
    augmented = x.unsqueeze(-3) if outputs > 1 else x
    prior_mean = model.model.mean_module(augmented)
    mean = prior_mean + strategy.gain * (base_mean - prior_mean) + strategy.shift
    target = y.transpose(-1, -2) if outputs > 1 else y.squeeze(-1)
    noise = model.likelihood.noise
    if outputs > 1:
        noise = (noise + model.likelihood.task_noises).unsqueeze(-1)
    correction = -0.5 * ((target - mean).square() - (target - base_mean).square()) / noise
    correction = correction.mean(-1)
    if outputs > 1:
        correction = correction.sum(-1)
    expected = expected + correction
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    # Use the actual event layout on both sides; the independent formula's
    # output-batch mean is transposed for the MT distribution.
    mean = mean.transpose(-1, -2) if outputs > 1 else mean
    covariance_dense = covariance
    if outputs > 1:
        n = x.shape[-2]
        covariance_dense = torch.einsum(
            "...tij,tu->...itju",
            covariance,
            torch.eye(outputs, dtype=x.dtype, device=x.device),
        ).reshape(*covariance.shape[:-3], n * outputs, n * outputs)
    torch.testing.assert_close(output.covariance_matrix, covariance_dense)
    expected_loss = loss_terms(mean, covariance_dense, expected)
    actual_loss = loss_terms(output.mean, output.covariance_matrix, actual)
    leaves = (*model.parameters(), x, y)
    gradients = [
        torch.autograd.grad(v, leaves, create_graph=True) for v in (actual_loss, expected_loss)
    ]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-9)
    hvps = [torch.autograd.grad(sum(g.square().sum() / 2 for g in gs), leaves) for gs in gradients]
    for a, b in zip(*hvps, strict=True):
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_default_port_descriptors_work_with_actual_strategy_and_automatic_elbo(device, outputs):
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.double, device=device).unsqueeze(-1)
    y = x.sin().expand(8, outputs)
    model = SingleTaskVariationalGP(
        x,
        y,
        num_outputs=outputs,
        inducing_points=x[:3],
        variational_strategy=PlainStrategy,
        backend="torch",
    )
    mll = VariationalELBO(model.likelihood, model.model, 8)
    value = mll()
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.posterior(x, observation_noise=True).mean.shape == (8, outputs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["deepcopy", "pickle", "state"])
def test_factory_state_modes_and_serialization_preserve_component_links(device, kind):
    model, strategy, x, _ = make_model(device, DenseStrategy)
    model.train()
    if kind == "deepcopy":
        cloned = copy.deepcopy(model)
    elif kind == "pickle":
        cloned = pickle.loads(pickle.dumps(model))
    else:
        cloned, *_ = make_model(device, DenseStrategy)
        cloned.load_state_dict(model.state_dict())
    assert cloned.training and cloned.model.training and cloned._strategy.training
    assert cloned._strategy.model is cloned.model
    assert cloned._strategy is not strategy
    a, b = model.posterior(x.detach()), cloned.posterior(x.detach())
    torch.testing.assert_close(a.mean, b.mean)
    torch.testing.assert_close(a.variance, b.variance)
    with torch.no_grad():
        cloned._strategy.shift.add_(0.3)
    assert not torch.allclose(model.posterior(x.detach()).mean, cloned.posterior(x.detach()).mean)


def test_malformed_factory_is_checked_at_the_actual_return_protocol():
    x = torch.rand(8, 1, dtype=torch.double)
    with pytest.raises(TypeError, match="return a Torch module with kl_divergence"):
        SingleTaskVariationalGP(
            x, inducing_points=x[:3], variational_strategy=lambda *a, **k: None, backend="torch"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_actual_strategy_leaves_participate_in_source_equivalent_adam_steps(device):
    pytest.importorskip("botorch")
    from gpytorch.mlls import VariationalELBO as ReferenceELBO

    models = [make_model(device, DenseStrategy, reference=r) for r in (False, True)]
    mlls = [
        cls(model.likelihood, model.model, 11, beta=0.7)
        for cls, (model, *_rest) in zip((VariationalELBO, ReferenceELBO), models, strict=True)
    ]
    optimizers = [torch.optim.Adam(mll.parameters(), lr=0.015) for mll in mlls]
    before = models[0][1].shift.detach().clone()
    for _ in range(4):
        losses = []
        for mll, optimizer, (_, _, x, y) in zip(mlls, optimizers, models, strict=True):
            optimizer.zero_grad()
            loss = -mll(mll.model(x), y.squeeze(-1)).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses, atol=2e-9, rtol=2e-9)
    assert not torch.equal(before, models[0][1].shift)
    parameters = [dict(model.named_parameters()) for model, *_ in models]
    assert parameters[0].keys() == parameters[1].keys()
    for name, actual in parameters[0].items():
        torch.testing.assert_close(actual, parameters[1][name], atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_public_torch_fitter_discovers_custom_strategy_parameters(device):
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    model, strategy, _x, _y = make_model(device, DenseStrategy)
    mll = VariationalELBO(model.likelihood, model.model, 11, beta=0.7)
    before = strategy.shift.detach().clone()
    result = fit_gpytorch_mll_torch(
        mll,
        step_limit=3,
        stopping_criterion=None,
        optimizer=functools.partial(torch.optim.Adam, lr=0.01),
    )
    assert result.step == 3
    assert not torch.equal(before, strategy.shift)
    assert torch.isfinite(mll()).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("factory", [None, PlainStrategy])
def test_variational_modules_and_inducing_points_use_transformed_feature_width(device, factory):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskVariationalGP as Reference
    from botorch.models.transforms.input import FilterFeatures

    x = torch.tensor(
        [[0.1, 0.2, 0.4], [0.3, 0.7, 0.8], [0.6, 0.4, 0.9], [0.9, 0.8, 0.5]],
        device=device,
        dtype=torch.double,
    )
    models = []
    for cls in (SingleTaskVariationalGP, Reference):
        points = x[[0, 2, 3]][:, [0, 2]]
        options = {} if factory is None else {"variational_strategy": factory}
        model = cls(
            x,
            x[..., :1].sin(),
            inducing_points=points,
            input_transform=FilterFeatures(torch.tensor([0, 2], device=device)),
            **options,
            **({"backend": "torch"} if cls is SingleTaskVariationalGP else {}),
        ).to(x)
        strategy = model.model.variational_strategy
        assert strategy.inducing_points.shape[-1] == 2
        assert model.model.covar_module.ard_num_dims == 2
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.copy_(
                x.new_tensor([-0.2, 0.1, 0.3])
            )
            strategy._variational_distribution.chol_variational_covar.copy_(
                torch.eye(3, device=device) * 0.8
            )
        models.append(model)
    values, gradients = [], []
    for model in models:
        query = x.detach().clone().requires_grad_()
        posterior = model.posterior(query, observation_noise=True)
        values.append((posterior.mean, posterior.variance))
        parameters = dict(model.named_parameters())
        gradients.append(
            torch.autograd.grad(
                posterior.mean.square().sum() + posterior.variance.sum(),
                (query, *(parameters[n] for n in sorted(parameters))),
            )
        )
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
    assert torch.equal(gradients[0][0][:, 1], x.new_zeros(4))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("factory", [None, PlainStrategy])
def test_training_rows_and_default_allocation_follow_actual_input_transform(device, factory):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskVariationalGP as Reference
    from gpytorch.mlls import VariationalELBO as ReferenceELBO

    x = torch.linspace(0.1, 0.9, 4, device=device, dtype=torch.double).unsqueeze(-1)
    values, gradients = [], []
    for reference, cls in enumerate((SingleTaskVariationalGP, Reference)):
        y = x.sin().repeat_interleave(2, -2).requires_grad_()
        options = {} if factory is None else {"variational_strategy": factory}
        model = cls(
            x,
            y,
            input_transform=RowDoubler(),
            **options,
            **({"backend": "torch"} if not reference else {}),
        ).to(x)
        strategy = model.model.variational_strategy
        assert model.model.train_inputs[0].shape == (8, 1)
        assert strategy.inducing_points.shape == (2, 1)
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.copy_(x.new_tensor([-0.2, 0.3]))
            strategy._variational_distribution.chol_variational_covar.copy_(
                torch.eye(2, dtype=x.dtype, device=x.device) * 0.8
            )
        q = model.model(model.model.train_inputs[0])
        mll = (ReferenceELBO if reference else VariationalELBO)(model.likelihood, model.model, 8)
        elbo = mll(q, y.squeeze(-1))
        values.append((q.mean, q.variance, elbo))
        parameters = dict(model.named_parameters())
        gradients.append(
            torch.autograd.grad(elbo.sum(), (*(parameters[n] for n in sorted(parameters)), y))
        )
    for a, b in zip(*values, strict=True):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_generic_default_constructor_preserves_strategy_factory_and_ignored_targets(device):
    pytest.importorskip("botorch")
    from botorch.models.approximate_gp import ApproximateGPyTorchModel as Reference
    from gpytorch.mlls import VariationalELBO as ReferenceELBO

    from botorch_rs.models.approximate_gp import ApproximateGPyTorchModel

    x = torch.linspace(0.1, 0.9, 6, dtype=torch.double, device=device).unsqueeze(-1)
    ignored = x.new_zeros(99, 3).requires_grad_()
    values = []
    for reference, cls in enumerate((ApproximateGPyTorchModel, Reference)):
        model = cls(
            train_X=x,
            train_Y=ignored,
            num_outputs=2,
            inducing_points=x[[0, 2, 5]],
            variational_strategy=PlainStrategy,
            **({"backend": "torch"} if not reference else {}),
        ).to(x)
        assert model.num_outputs == 2
        assert not hasattr(model.model, "_outer_ref")
        strategy = model.model.variational_strategy.base_variational_strategy
        assert type(strategy) is PlainStrategy and strategy.model is model.model
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.fill_(0.2)
            strategy._variational_distribution.chol_variational_covar.copy_(
                torch.eye(3, dtype=x.dtype, device=x.device) * 0.8
            )
        output = model.model(x)
        target = x.sin().expand(6, 2)
        mll = (ReferenceELBO if reference else VariationalELBO)(model.likelihood, model.model, 6)
        values.append((output.mean, output.variance, mll(output, target)))
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    assert ignored.grad is None


@pytest.mark.parametrize("device", DEVICES)
def test_factory_runs_once_and_receives_constructor_overrides(device):
    x = torch.linspace(0.1, 0.9, 6, dtype=torch.double, device=device).unsqueeze(-1)
    calls = []

    def factory(model, **kwargs):
        calls.append((model, kwargs))
        return PlainStrategy(model, **kwargs)

    unused_distribution = gpytorch.variational.CholeskyVariationalDistribution(1)
    model = SingleTaskVariationalGP(
        x,
        inducing_points=x[:3],
        variational_strategy=functools.partial(
            factory,
            inducing_points=x[:1],
            variational_distribution=unused_distribution,
            learn_inducing_locations=False,
        ),
        backend="torch",
    )
    assert len(calls) == 1 and calls[0][0] is model.model
    assert calls[0][1]["inducing_points"].shape == (3, 1)
    assert calls[0][1]["variational_distribution"] is model._strategy._variational_distribution
    assert calls[0][1]["variational_distribution"] is not unused_distribution
    assert calls[0][1]["learn_inducing_locations"] is True
    assert isinstance(model._strategy.inducing_points, torch.nn.Parameter)
