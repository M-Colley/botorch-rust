"""An actual user strategy factory owns adaptation of its constructor inputs."""

# ruff: noqa: E402
from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskVariationalGP as Source
from test_torch_variational_shared_distribution import DenseKernel, DenseMean

from botorch_rs.models import SingleTaskVariationalGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def adapting_factory(kind, model, inducing_points, variational_distribution, **kwargs):
    if kind == "event":
        inducing_points = inducing_points[..., :2, :]
    elif kind == "features":
        inducing_points = inducing_points[..., :2]
    elif kind == "batches":
        inducing_points = inducing_points[0]
    elif kind == "distribution":
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(2)
        inducing_points = inducing_points[..., :2, :]
    return gpytorch.variational.VariationalStrategy(
        model, inducing_points, variational_distribution, **kwargs
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["event", "features", "batches", "distribution"])
def test_source_valid_factory_adapts_raw_points_and_distribution(device, kind):
    values, gradients = [], []
    for cls in (Source, SingleTaskVariationalGP):
        reference = cls is Source
        x = torch.tensor(
            [[0.1, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.8]],
            device=device,
            dtype=torch.double,
        )
        points = x
        if kind == "features":
            points = torch.cat((points, points + 0.01), -1)
        if kind == "batches":
            points = points.expand(3, 4, 2)
        count = 2 if kind == "event" else 4
        distribution = gpytorch.variational.CholeskyVariationalDistribution(count, mean_init_std=0)
        model = cls(
            x,
            inducing_points=points,
            variational_distribution=distribution,
            variational_strategy=functools.partial(adapting_factory, kind),
            covar_module=DenseKernel((), x),
            mean_module=DenseMean((), x),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            **({} if reference else {"backend": "torch"}),
        )
        strategy = model.model.variational_strategy
        actual_distribution = strategy._variational_distribution
        assert (actual_distribution is distribution) == (kind != "distribution")
        assert all(p.dtype == x.dtype and p.device == x.device for p in model.parameters())
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            actual_distribution.variational_mean.fill_(0.2)
            actual_distribution.chol_variational_covar.copy_(
                torch.eye(actual_distribution.num_inducing_points, device=device) * 0.8
            )
        query = x.new_tensor([[0.31, 0.48], [0.68, 0.28]], requires_grad=True)
        posterior = model.posterior(query, observation_noise=True)
        values.append((posterior.mean, posterior.variance))
        parameters = dict(model.named_parameters())
        gradients.append(
            torch.autograd.grad(
                posterior.mean.sin().sum() + posterior.variance.square().sum(),
                (query, *(parameters[name] for name in sorted(parameters))),
            )
        )
    for a, b in zip(*values, strict=True):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-8)


class FactoryDefaultStateStrategy(gpytorch.variational.VariationalStrategy):
    def __init__(self, *args, **kwargs):
        distribution = kwargs["variational_distribution"]
        self.constructor_dtype = distribution.variational_mean.dtype
        self.constructor_device = distribution.variational_mean.device
        super().__init__(*args, **kwargs)
        self.shift = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return gpytorch.distributions.MultivariateNormal(
            result.mean + self.shift, result.lazy_covariance_matrix
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("supplied", [False, True])
def test_factory_runs_before_model_dtype_device_conversion(device, supplied):
    records = []
    for cls in (Source, SingleTaskVariationalGP):
        reference = cls is Source
        x = torch.linspace(0.1, 0.9, 8, device=device, dtype=torch.double).unsqueeze(-1)
        options = {}
        if supplied:
            options["variational_distribution"] = (
                gpytorch.variational.CholeskyVariationalDistribution(3, mean_init_std=0)
            )
        model = cls(
            x,
            inducing_points=x[:3],
            variational_strategy=FactoryDefaultStateStrategy,
            **options,
            **({} if reference else {"backend": "torch"}),
        )
        strategy = model.model.variational_strategy
        records.append((strategy.constructor_dtype, strategy.constructor_device))
        assert all(p.dtype == x.dtype and p.device == x.device for p in model.parameters())
        assert strategy.shift.dtype == x.dtype and strategy.shift.device == x.device
    assert records[0] == records[1] == (torch.get_default_dtype(), torch.device("cpu"))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("learn", [False, True])
@pytest.mark.parametrize("point_kind", ["float32", "float16", "other_device"])
@pytest.mark.parametrize(
    "strategy_name",
    [
        "VariationalStrategy",
        "UnwhitenedVariationalStrategy",
        "BatchDecoupledVariationalStrategy",
        "CiqVariationalStrategy",
        "LargeBatchVariationalStrategy",
    ],
)
def test_canonical_inducing_points_follow_model_dtype_and_device(
    device, learn, point_kind, strategy_name
):
    if point_kind == "other_device" and not torch.cuda.is_available():
        pytest.skip("A second device is needed for this constructor conversion control.")
    records = []
    for cls in (Source, SingleTaskVariationalGP):
        reference = cls is Source
        x = torch.tensor(
            [[0.1, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.8]],
            device=device,
            dtype=torch.double,
        )
        points = x[[0, 2, 3]]
        points = (
            points.to("cuda" if device == "cpu" else "cpu")
            if point_kind == "other_device"
            else points.to(getattr(torch, point_kind))
        )
        model = cls(
            x,
            inducing_points=points,
            learn_inducing_points=learn,
            variational_strategy=getattr(gpytorch.variational, strategy_name),
            covar_module=DenseKernel((), x),
            mean_module=DenseMean((), x),
            **({} if reference else {"backend": "torch"}),
        )
        strategy = model.model.variational_strategy
        assert (
            strategy.inducing_points.dtype == x.dtype
            and strategy.inducing_points.device == x.device
        )
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.fill_(0.2)
            strategy._variational_distribution.chol_variational_covar.copy_(
                torch.eye(3, device=device) * 0.8
            )
        query = x.new_tensor([[0.31, 0.48], [0.68, 0.28]])
        posterior = model.posterior(query, observation_noise=True)
        records.append((posterior.mean, posterior.variance))
    for actual, expected in zip(*records, strict=True):
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
