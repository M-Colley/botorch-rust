"""Strategy partials preserve constructor overrides and configured jitter."""

# ruff: noqa: E402
import functools

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskVariationalGP as ReferenceGP
from gpytorch.mlls import VariationalELBO as ReferenceELBO

from botorch_rs import variational
from botorch_rs.kernels import RBFKernel
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _pair(device, jitter, unwhitened, own_spec):
    results = []
    for reference in (False, True):
        x = torch.linspace(0.1, 0.9, 10, device=device, dtype=torch.double).reshape(5, 2)
        y = x[:, :1].sin().requires_grad_()
        x.requires_grad_()
        distribution = gpytorch.variational.CholeskyVariationalDistribution(3, mean_init_std=0).to(
            x
        )
        module = gpytorch.variational if reference or not own_spec else variational
        cls = getattr(
            module, "UnwhitenedVariationalStrategy" if unwhitened else "VariationalStrategy"
        )
        model = (ReferenceGP if reference else SingleTaskVariationalGP)(
            x,
            y,
            inducing_points=x.detach()[[0, 2, 4]],
            variational_distribution=distribution,
            variational_strategy=functools.partial(
                cls,
                jitter_val=jitter,
                # Caller-supplied constructor arguments override these defaults.
                inducing_points=x.new_zeros(1, 2),
                learn_inducing_locations=False,
            ),
            covar_module=(gpytorch.kernels.RBFKernel if reference else RBFKernel)(ard_num_dims=2),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            outcome_transform=None,
        ).to(x)
        with torch.no_grad():
            strategy = model.model.variational_strategy
            strategy.variational_params_initialized.fill_(1)
            distribution.variational_mean.copy_(x.new_tensor([-0.2, 0.1, 0.3]))
            distribution.chol_variational_covar.mul_(0.8)
        assert strategy.jitter_val == jitter
        assert strategy.inducing_points.shape == (3, 2)
        assert isinstance(strategy.inducing_points, torch.nn.Parameter)
        if not reference:
            assert model._tensor_backend == "torch"
        results.append((model, x, y))
    return results


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("jitter", [0.0, 1e-3, 0.02])
@pytest.mark.parametrize("unwhitened", [False, True])
@pytest.mark.parametrize("own_spec", [False, True])
def test_partial_strategy_q_elbo_and_complete_first_graphs(device, jitter, unwhitened, own_spec):
    values, gradients = [], []
    for reference, (model, x, y) in enumerate(_pair(device, jitter, unwhitened, own_spec)):
        cls = ReferenceELBO if reference else VariationalELBO
        output = model.model(x)
        loss = cls(model.likelihood, model.model, 5)(output, y.squeeze(-1))
        values.append((output.mean, output.variance, output.covariance_matrix, loss))
        parameters = dict(model.named_parameters())
        leaves = tuple(parameters[name] for name in sorted(parameters)) + (x, y)
        gradients.append(torch.autograd.grad(loss.sum(), leaves))
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("jitter", [-0.1, float("inf"), float("nan")])
def test_partial_strategy_rejects_invalid_jitter_explicitly(jitter):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _pair("cpu", jitter, False, True)
