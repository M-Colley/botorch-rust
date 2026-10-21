"""Supported Tensor-only variational configurations select their backend."""

# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.variational import (
    DeltaVariationalDistribution,
    MeanFieldVariationalDistribution,
    UnwhitenedVariationalStrategy,
)


@pytest.mark.parametrize(
    "kind",
    ["meanfield", "delta", "unwhitened", "external_distribution", "external_kernel", "batch"],
)
def test_cpu_float64_tensor_only_configurations_select_auto(kind):
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.double).reshape(6, 2)
    y = x[:, :1].sin()
    kwargs = {"inducing_points": x[:3].clone()}
    if kind in ("meanfield", "delta"):
        kwargs["variational_distribution"] = (
            MeanFieldVariationalDistribution
            if kind == "meanfield"
            else DeltaVariationalDistribution
        )(3)
    elif kind == "unwhitened":
        kwargs["variational_strategy"] = UnwhitenedVariationalStrategy
    elif kind == "external_distribution":
        kwargs["variational_distribution"] = gpytorch.variational.CholeskyVariationalDistribution(3)
    elif kind == "external_kernel":
        kwargs["covar_module"] = gpytorch.kernels.RBFKernel(ard_num_dims=2)
    else:
        x, y = x.expand(2, 6, 2).clone(), y.expand(2, 6, 1).clone()
    assert not x.requires_grad and not y.requires_grad and x.device.type == "cpu"
    actual = SingleTaskVariationalGP(x, y, outcome_transform=None, **kwargs)
    assert actual._tensor_backend == "torch"
    prediction = actual.posterior(x[..., :2, :])
    assert prediction.mean.shape == (*x.shape[:-2], 2, 1)
    assert torch.isfinite(prediction.mean).all()
    with pytest.raises((UnsupportedError, ValueError)):
        SingleTaskVariationalGP(x, y, outcome_transform=None, backend="native", **kwargs)


@pytest.mark.parametrize("positional", [False, True])
def test_positional_variational_specs_route_and_plain_native_default_is_preserved(positional):
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.double).reshape(6, 2)
    y = x[:, :1].sin()
    distribution = MeanFieldVariationalDistribution(3)
    model = (
        SingleTaskVariationalGP(
            x, y, None, 1, True, None, None, distribution, UnwhitenedVariationalStrategy, x[:3]
        )
        if positional
        else SingleTaskVariationalGP(
            x,
            y,
            variational_distribution=distribution,
            variational_strategy=UnwhitenedVariationalStrategy,
            inducing_points=x[:3],
        )
    )
    assert model._tensor_backend == "torch"
    assert not model._strategy.whitened
    native = SingleTaskVariationalGP(x, y, inducing_points=x[:3])
    assert getattr(native, "_tensor_backend", None) != "torch"
    assert not isinstance(native, torch.nn.Module)
