"""CIQ symmetric whitening, source conventions and complete Torch graphs."""

# ruff: noqa: E402
import contextlib
import functools
import math
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskVariationalGP as ReferenceGP
from gpytorch.mlls import VariationalELBO as ReferenceELBO

from botorch_rs import variational
from botorch_rs.kernels import RBFKernel
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models._torch_backend_variational_ciq import contour_inverse_root_matmul
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.sampling.pathwise import draw_matheron_paths

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FAMILIES = ["Cholesky", "MeanField", "Delta", "Natural", "TrilNatural"]


@pytest.fixture(autouse=True)
def precise_reference_iterations():
    # Tighten the independent source's iterative solves, not port tolerances.
    with contextlib.ExitStack() as stack:
        for setting, value in (
            (gpytorch.settings.minres_tolerance, 1e-12),
            (gpytorch.settings.cg_tolerance, 1e-10),
            (gpytorch.settings.eval_cg_tolerance, 1e-10),
            (gpytorch.settings.max_cg_iterations, 100),
            (gpytorch.settings.num_contour_quadrature, 24),
        ):
            stack.enter_context(setting(value))
        yield


def _pair(device, family="Cholesky", layout="scalar", dtype=torch.double):
    pairs = []
    outputs = 2 if layout == "outputs" else 1
    for reference in (False, True):
        x = torch.linspace(0.08, 0.91, 10, dtype=dtype, device=device).reshape(5, 2)
        y = x[..., :1].sin()
        if layout == "batch":
            x = torch.stack((x, x + 0.02))
            y = torch.stack((y, y * 0.8 + 0.1))
        if outputs > 1:
            y = torch.cat((y, y * 0.6 + 0.2), -1)
        x, y = x.requires_grad_(), y.requires_grad_()
        batch = x.shape[:-2]
        augmented = batch + ((outputs,) if outputs > 1 else ())
        distribution = getattr(gpytorch.variational, family + "VariationalDistribution")(
            3, batch_shape=augmented, mean_init_std=0
        ).to(x)
        kernel = (gpytorch.kernels.RBFKernel if reference else RBFKernel)(
            ard_num_dims=2, batch_shape=augmented
        )
        mean = gpytorch.means.ConstantMean(batch_shape=augmented).to(x)
        likelihood = (
            gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch).to(x)
            if outputs == 1
            else None
        )
        strategy = (gpytorch.variational if reference else variational).CiqVariationalStrategy
        model = (ReferenceGP if reference else SingleTaskVariationalGP)(
            x,
            y,
            covar_module=kernel,
            mean_module=mean,
            likelihood=likelihood,
            inducing_points=x.detach()[..., [0, 2, 4], :],
            variational_distribution=distribution,
            variational_strategy=functools.partial(strategy, jitter_val=2e-4),
            outcome_transform=None,
        ).to(x)
        strategy = model.model.variational_strategy
        if outputs > 1:
            strategy = strategy.base_variational_strategy
        covariance = x.new_tensor([[0.7, 0.06, 0.02], [0.06, 0.9, -0.03], [0.02, -0.03, 0.8]])
        if family == "Natural":
            # Its source CG has a fixed 1e-10 internal denominator cutoff.
            # A diagonal precision makes its Jacobi preconditioner exact, so
            # these strict source comparisons are independent of that solver
            # error. A separate dense oracle covers nondiagonal precision.
            covariance = torch.diag_embed(covariance.diagonal())
        distribution.initialize_variational_distribution(
            gpytorch.distributions.MultivariateNormal(
                x.new_tensor([-0.16, 0.13, 0.23]).expand(*augmented, 3),
                covariance.expand(*augmented, 3, 3),
            )
        )
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            model.model.covar_module.lengthscale = x.new_tensor([0.3, 0.55])
            mean.constant.fill_(0.12)
            model.likelihood.noise = x.new_tensor(0.09)
        pairs.append((model, strategy, distribution, x, y))
    return pairs


def _evaluate(pair, reference):
    model, _, _, x, y = pair
    output = model.model(x)
    mll = (ReferenceELBO if reference else VariationalELBO)(
        model.likelihood, model.model, num_data=9, beta=0.7, combine_terms=False
    )
    if (
        reference
        and model.num_outputs > 1
        and type(pair[2]) is gpytorch.variational.NaturalVariationalDistribution
    ):
        # Only repair the known wrapper defect proposed in GPyTorch #2823.
        # The unchanged independent wrapper tries to access the forbidden
        # NGD-CIQ variational_distribution instead of delegating its base KL.
        wrapper = model.model.variational_strategy
        with mock.patch.object(
            wrapper,
            "kl_divergence",
            side_effect=lambda: wrapper.base_variational_strategy.kl_divergence().sum(-1),
        ):
            terms = mll(output, y)
    else:
        terms = mll(output, y if model.num_outputs > 1 else y.squeeze(-1))
    parameters = dict(model.named_parameters())
    leaves = tuple(parameters[name] for name in sorted(parameters)) + (x, y)
    return output, terms, leaves


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", ["scalar", "batch", "outputs"])
@pytest.mark.parametrize("family", FAMILIES)
def test_ciq_full_law_elbo_and_all_first_gradients_match_source(device, family, layout):
    values, gradients = [], []
    for reference, pair in enumerate(_pair(device, family, layout)):
        output, terms, leaves = _evaluate(pair, reference)
        values.append((output.mean, output.variance, output.covariance_matrix, *terms))
        # A nonlinear conditional term checks full-covariance pullbacks too.
        objective = (
            sum(term.sum() for term in terms) + output.covariance_matrix.square().sum() * 0.02
        )
        gradients.append(torch.autograd.grad(objective, leaves))
        assert all(value.device.type == device for value in values[-1])
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=8e-7, rtol=8e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("repeated", [False, True])
def test_contour_inverse_root_values_and_complete_higher_graphs(device, repeated):
    raw = torch.tensor([[1.1, 0.0], [0.2, 0.8]], dtype=torch.double, device=device)
    if repeated:
        raw = torch.eye(2, dtype=torch.double, device=device)
    raw.requires_grad_()
    rhs = torch.tensor(
        [[0.2, 0.4], [-0.1, 0.6]], dtype=torch.double, device=device, requires_grad=True
    )

    def operation(factor, value):
        matrix = factor @ factor.mT + 0.2 * torch.eye(2, dtype=factor.dtype, device=device)
        return contour_inverse_root_matmul(matrix, value, count=20)

    actual = operation(raw, rhs)
    matrix = raw @ raw.mT + 0.2 * torch.eye(2, dtype=raw.dtype, device=device)
    values, vectors = torch.linalg.eigh(matrix.detach())
    expected = (vectors * values.rsqrt()) @ vectors.mT @ rhs
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    assert torch.autograd.gradcheck(operation, (raw, rhs), eps=1e-5, atol=2e-7, rtol=2e-6)
    assert torch.autograd.gradgradcheck(operation, (raw, rhs), eps=1e-5, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", ["scalar", "batch"])
def test_ciq_complete_model_hvp_matches_independent_reference_first_gradient_differences(
    device, layout
):
    actual_pair, reference_pair = _pair(device, layout=layout)
    _, terms, leaves = _evaluate(actual_pair, False)
    first = torch.autograd.grad(sum(t.sum() for t in terms), leaves, create_graph=True)
    directions = tuple(torch.full_like(value, 0.03) for value in leaves)
    actual = torch.autograd.grad(
        sum((a * b).sum() for a, b in zip(first, directions, strict=True)), leaves
    )
    _, _, reference_leaves = _evaluate(reference_pair, True)
    saved = [value.detach().clone() for value in reference_leaves]
    firsts = []
    epsilon = 2e-5
    for sign in (-1, 1):
        with torch.no_grad():
            for value, base, direction in zip(reference_leaves, saved, directions, strict=True):
                value.copy_(base + sign * epsilon * direction)
        _, terms, current = _evaluate(reference_pair, True)
        firsts.append(torch.autograd.grad(sum(t.sum() for t in terms), current))
    for found, minus, plus in zip(actual, *firsts, strict=True):
        torch.testing.assert_close(found, (plus - minus) / (2 * epsilon), atol=2e-5, rtol=3e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["Cholesky", "Natural"])
def test_ciq_actual_optimizer_replay(device, family):
    pairs = _pair(device, family)
    optimizers = []
    for model, _, _, _, _ in pairs:
        # Natural parameters must follow their registered natural gradient direction.
        optimizers.append(torch.optim.SGD(model.parameters(), lr=0.002))
    for _ in range(3):
        for reference, (pair, optimizer) in enumerate(zip(pairs, optimizers, strict=True)):
            optimizer.zero_grad()
            _, terms, _ = _evaluate(pair, reference)
            (
                -sum(term.sum() * sign for term, sign in zip(terms, (1, -1, 1), strict=True))
            ).backward()
            optimizer.step()
    actual = dict(pairs[0][0].named_parameters())
    expected = dict(pairs[1][0].named_parameters())
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_ciq_symmetric_coordinates_and_source_natural_surrogate_are_explicit(device):
    model, strategy, distribution, x, _ = _pair(device, "Natural")[0]
    with pytest.raises(RuntimeError, match="during forward"):
        strategy.kl_divergence()
    with pytest.raises(RuntimeError, match="during forward"):
        _ = strategy.variational_distribution
    output = model.model(x)
    assert output._kl.item() == 0
    assert distribution is strategy._variational_distribution
    mean, covariance = distribution().mean, distribution().covariance_matrix
    analytic = 0.5 * (
        mean.square().sum()
        + covariance.diagonal().sum()
        - 3
        - torch.linalg.slogdet(covariance).logabsdet
    )
    expected = torch.autograd.grad(analytic, tuple(distribution.parameters()))
    found = torch.autograd.grad(output._kl, tuple(distribution.parameters()))
    for actual, wanted in zip(found, expected, strict=True):
        torch.testing.assert_close(actual, wanted)
    with pytest.raises(NotImplementedError, match="CiqVariationalStrategy"):
        draw_matheron_paths(model, torch.Size([2]))
    # Verify symmetric whitening is physically different from Cholesky whitening.
    z = strategy.inducing_points
    matrix = model.model.covar_module(z, z)
    matrix = matrix + strategy.jitter_val * torch.eye(3, dtype=x.dtype, device=device)
    right = model.model.covar_module(z, x)
    symmetric = contour_inverse_root_matmul(matrix, right)
    triangular = torch.linalg.solve_triangular(torch.linalg.cholesky(matrix), right, upper=False)
    assert (symmetric - triangular).abs().max() > 0.05


@pytest.mark.parametrize("device", DEVICES)
def test_ciq_float32_device_law_and_gradients(device):
    values, gradients = [], []
    for reference, pair in enumerate(_pair(device, dtype=torch.float32)):
        output, terms, leaves = _evaluate(pair, reference)
        values.append((output.mean, output.variance, *terms))
        gradients.append(torch.autograd.grad(sum(t.sum() for t in terms), leaves))
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_ciq_nondiagonal_natural_precision_matches_independent_dense_gaussian(device):
    values, gradients = [], []
    for reference, pair in enumerate(_pair(device, "Natural")):
        model, strategy, distribution, x, y = pair
        covariance = x.new_tensor([[0.7, 0.06, 0.02], [0.06, 0.9, -0.03], [0.02, -0.03, 0.8]])
        distribution.initialize_variational_distribution(
            gpytorch.distributions.MultivariateNormal(x.new_tensor([-0.16, 0.13, 0.23]), covariance)
        )
        if not reference:
            output, terms, leaves = _evaluate(pair, False)
            mean, variance = output.mean, output.variance
            loss = terms[0].sum() - terms[1].sum()
        else:
            # An exact Gaussian identity avoids the source Natural branch's
            # iterative precision solve and its fixed denominator cutoff.
            # This oracle uses an eigendecomposition of a distinct spectrum,
            # not the port's rational approximation or Cholesky coordinates.
            z = strategy.inducing_points
            kzz = model.model.covar_module(z, z).to_dense()
            kzz = kzz + strategy.jitter_val * torch.eye(3, dtype=x.dtype, device=device)
            eigenvalues, eigenvectors = torch.linalg.eigh(kzz)
            inverse_root = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.mT
            cross = model.model.covar_module(z, x).to_dense()
            interpolation = inverse_root @ cross
            q = distribution()
            mean = model.model.mean_module(x) + (interpolation.mT @ q.mean.unsqueeze(-1)).squeeze(
                -1
            )
            variance = (
                model.model.covar_module(x, x, diag=True)
                + strategy.jitter_val
                + (
                    interpolation
                    * (
                        (q.covariance_matrix - torch.eye(3, dtype=x.dtype, device=device))
                        @ interpolation
                    )
                ).sum(-2)
            )
            kl = 0.5 * (
                q.mean.square().sum()
                + q.covariance_matrix.diagonal().sum()
                - 3
                - torch.linalg.slogdet(q.covariance_matrix).logabsdet
            )
            noise = model.likelihood.noise
            log_prob = (
                -0.5
                * (
                    ((y.squeeze(-1) - mean).square() + variance) / noise
                    + noise.log()
                    + math.log(2 * math.pi)
                ).mean()
            )
            loss = log_prob - (kl - kl.detach()) * (0.7 / 9)
            parameters = dict(model.named_parameters())
            leaves = tuple(parameters[name] for name in sorted(parameters)) + (x, y)
        values.append((mean, variance, loss))
        gradients.append(torch.autograd.grad(loss, leaves))
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_ciq_saved_law_t_batches_and_prior_bypass(device):
    pair = _pair(device)[0]
    model, strategy, _, x, _ = pair
    model.eval()
    query = x[:2].detach().expand(4, 2, 2).clone().requires_grad_()
    output = model.model(query)
    expected = [value.detach().clone() for value in (output.mean, output.variance, output._kl)]
    # Materialize the covariance only after public parameter/point updates.
    with torch.no_grad():
        model.model.covar_module.lengthscale = x.new_tensor([0.6, 0.9])
        strategy.inducing_points.add_(0.02)
        strategy._variational_distribution.variational_mean.add_(0.1)
    for actual, saved in zip((output.mean, output.variance, output._kl), expected, strict=True):
        torch.testing.assert_close(actual, saved)
    assert output.covariance_matrix.shape == (4, 2, 2)
    assert torch.isfinite(
        torch.autograd.grad(output.mean.sum() + output.variance.sum(), query)[0]
    ).all()
    prior = model.model(query, prior=True)
    wanted = model.model.covar_module(query, query)
    torch.testing.assert_close(prior.covariance_matrix, wanted)


@pytest.mark.parametrize("device", DEVICES)
def test_ciq_port_and_supplied_factory_auto_routing_and_natural_initialization(device):
    for supplied in (False, True):
        x = torch.linspace(0.1, 0.9, 5, device=device, dtype=torch.double).unsqueeze(-1)
        distribution = gpytorch.variational.NaturalVariationalDistribution(3, mean_init_std=0).to(x)
        strategy_class = (gpytorch.variational if supplied else variational).CiqVariationalStrategy
        model = SingleTaskVariationalGP(
            x,
            x.sin(),
            inducing_points=x[[0, 2, 4]],
            variational_distribution=distribution,
            variational_strategy=functools.partial(strategy_class, jitter_val=0),
            covar_module=RBFKernel(),
            outcome_transform=None,
        ).to(x)
        assert model._tensor_backend == "torch"
        torch.manual_seed(235)
        expected = torch.randn_like(distribution.natural_vec) * 1e-3
        torch.manual_seed(235)
        model.model(x)
        torch.testing.assert_close(distribution.natural_vec, expected)
        torch.testing.assert_close(
            distribution.natural_mat, -0.5 * torch.eye(3, device=device, dtype=x.dtype)
        )
        assert model._strategy.jitter_val == 0


def test_unchanged_independent_natural_ciq_wrapper_has_the_known_upstream_kl_defect():
    model, _, _, x, y = _pair("cpu", "Natural", "outputs")[1]
    output = model.model(x)
    with pytest.raises(RuntimeError, match="NGD-CIQ"):
        ReferenceELBO(model.likelihood, model.model, num_data=5)(output, y)


def test_contour_rhs_broadcast_and_invalid_covariance():
    matrix = torch.eye(3, dtype=torch.double) * 2
    rhs = torch.arange(12, dtype=torch.double).reshape(2, 3, 2)
    torch.testing.assert_close(contour_inverse_root_matmul(matrix, rhs), rhs / math.sqrt(2))
    with pytest.raises(ValueError, match="positive"):
        contour_inverse_root_matmul(matrix, rhs, count=0)
    with pytest.raises(RuntimeError, match="positive definite"):
        contour_inverse_root_matmul(-matrix, rhs)
