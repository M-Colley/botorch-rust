"""Independent mean/covariance inducing laws and the decoupled regularizer."""

# ruff: noqa: E402
import functools
import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskVariationalGP as ReferenceGP
from gpytorch.mlls import VariationalELBO as ReferenceELBO

from botorch_rs import variational
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import RBFKernel
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.sampling.pathwise import draw_matheron_paths, gaussian_update

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
LAYOUTS = ["scalar", "shared_batch", "mean_var_last", "mean_var_first", "outputs"]


def _pair(device, layout="scalar", family="CholeskyVariationalDistribution", dtype=torch.double):
    pairs = []
    outputs = 2 if layout == "outputs" else 1
    batched = layout not in ("scalar", "outputs")
    for reference in (False, True):
        x = torch.linspace(0.1, 0.9, 10, dtype=dtype, device=device).reshape(5, 2)
        y = x[..., :1].sin()
        if outputs > 1:
            y = torch.cat((y, 0.7 * y + 0.2), -1)
        if batched:
            x = x.expand(3, 5, 2).clone() + x.new_tensor([0, 0.01, 0.03])[:, None, None]
            y = y.expand(3, 5, 1).clone()
        x.requires_grad_()
        y.requires_grad_()
        batch = x.shape[:-2]
        aug = batch + ((outputs,) if outputs > 1 else ())
        kernel_batch = {
            "scalar": (),
            "shared_batch": (),
            "mean_var_last": (3, 2),
            "mean_var_first": (2, 3),
            "outputs": (2,),
        }[layout]
        mean_var_dimension = {"mean_var_last": -1, "mean_var_first": -2}.get(layout)
        source = getattr(gpytorch.variational, family)(3, batch_shape=aug, mean_init_std=0).to(x)
        mean = gpytorch.means.ConstantMean(batch_shape=kernel_batch).to(x)
        kernel = (gpytorch.kernels.RBFKernel if reference else RBFKernel)(
            ard_num_dims=2, batch_shape=kernel_batch
        )
        likelihood = (
            gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch) if outputs == 1 else None
        )
        strategy_class = (
            gpytorch.variational if reference else variational
        ).BatchDecoupledVariationalStrategy
        model = (ReferenceGP if reference else SingleTaskVariationalGP)(
            x,
            y,
            inducing_points=x.detach()[..., [0, 2, 4], :],
            variational_distribution=source,
            variational_strategy=functools.partial(
                strategy_class, mean_var_batch_dim=mean_var_dimension, jitter_val=1e-4
            ),
            covar_module=kernel,
            mean_module=mean,
            likelihood=likelihood,
            outcome_transform=None,
        ).to(x)
        strategy = model.model.variational_strategy
        if outputs > 1:
            strategy = strategy.base_variational_strategy
        assert strategy._variational_distribution is source
        source.initialize_variational_distribution(
            gpytorch.distributions.MultivariateNormal(
                x.new_tensor([-0.2, 0.1, 0.3]).expand(*aug, 3),
                (0.8 * torch.eye(3, dtype=dtype, device=device)).expand(*aug, 3, 3),
            )
        )
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy.inducing_points.select((mean_var_dimension or -1) - 2, 0).sub_(0.025)
            strategy.inducing_points.select((mean_var_dimension or -1) - 2, 1).add_(0.035)
            model.model.covar_module.lengthscale = x.new_tensor([0.4, 0.7])
            if mean_var_dimension is not None:
                # Mean and covariance hyperparameters really differ as well.
                mean.constant.select(mean_var_dimension, 0).fill_(0.12)
                mean.constant.select(mean_var_dimension, 1).fill_(-0.08)
                shape = model.model.covar_module.raw_lengthscale.shape
                varying = x.new_tensor([0.35, 0.6]).expand(*shape).clone()
                varying.select(mean_var_dimension - 2, 1).mul_(1.4)
                model.model.covar_module.lengthscale = varying
            else:
                mean.constant.fill_(0.12)
            model.likelihood.noise = x.new_tensor(0.08)
        pairs.append((model, source, x, y, strategy))
    return pairs


def _loss_and_leaves(pair, reference):
    model, _, x, y, _ = pair
    output = model.model(x)
    mll = (ReferenceELBO if reference else VariationalELBO)(
        model.likelihood, model.model, 7, beta=0.6, combine_terms=False
    )
    terms = mll(output, y.squeeze(-1) if model.num_outputs == 1 else y)
    if reference and model.num_outputs > 1:
        # Pinned IndependentMultitaskVariationalStrategy ignores its base KL
        # override, dropping the decoupled Delta-mean normalizing term. Repair
        # only that known-defective reference term using its own base strategy;
        # retain its independent GP, likelihood, conditional and ELBO code.
        base = model.model.variational_strategy.base_variational_strategy
        corrected_kl = base.kl_divergence().sum(-1) * (mll.beta / mll.num_data)
        terms = (terms[0], corrected_kl, terms[2])
    parameters = dict(model.named_parameters())
    leaves = tuple(parameters[name] for name in sorted(parameters)) + (x, y)
    return output, terms, leaves


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "family", ["CholeskyVariationalDistribution", "MeanFieldVariationalDistribution"]
)
def test_decoupled_q_full_training_covariance_kl_elbo_and_all_first_vjps(device, layout, family):
    pairs = _pair(device, layout, family)
    values, gradients = [], []
    for reference, pair in enumerate(pairs):
        output, terms, leaves = _loss_and_leaves(pair, reference)
        values.append((output.mean, output.variance, output.covariance_matrix, *terms))
        gradients.append(torch.autograd.grad(sum(term.sum() for term in terms), leaves))
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-7)
    for reference, (model, _, x, _, _) in enumerate(pairs):
        posterior = model.posterior(x[..., :2, :])
        covariance = posterior.distribution.covariance_matrix
        if model.num_outputs > 1 and getattr(posterior.distribution, "_interleaved", False):
            n, m = posterior.mean.shape[-2:]
            order = torch.arange(n * m, device=x.device).reshape(n, m).mT.flatten()
            covariance = covariance.index_select(-2, order).index_select(-1, order)
        values[reference] = (
            posterior.mean,
            posterior.variance,
            covariance,
        )
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", LAYOUTS[:4])
def test_decoupled_complete_hessian_vector_product_matches_reference_gradient_differences(
    device, layout
):
    actual_pair, reference_pair = _pair(device, layout)
    _, terms, leaves = _loss_and_leaves(actual_pair, False)
    first = torch.autograd.grad(sum(term.sum() for term in terms), leaves, create_graph=True)
    directions = [torch.full_like(leaf, 0.07) for leaf in leaves]
    hvp = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(first, directions, strict=True)), leaves
    )
    _, _, reference_leaves = _loss_and_leaves(reference_pair, True)
    base = [leaf.detach().clone() for leaf in reference_leaves]
    finite = []
    epsilon = 1e-5
    for sign in (-1, 1):
        with torch.no_grad():
            for leaf, value, direction in zip(reference_leaves, base, directions, strict=True):
                leaf.copy_(value + sign * epsilon * direction)
        reference_pair[0].train()
        _, ref_terms, ref_leaves = _loss_and_leaves(reference_pair, True)
        finite.append(torch.autograd.grad(sum(term.sum() for term in ref_terms), ref_leaves))
    for actual, minus, plus in zip(hvp, *finite, strict=True):
        torch.testing.assert_close(actual, (plus - minus) / (2 * epsilon), atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family", ["NaturalVariationalDistribution", "TrilNaturalVariationalDistribution"]
)
def test_decoupled_retains_supplied_natural_gradient_rule(device, family):
    pairs = _pair(device, "mean_var_last", family)
    values, gradients = [], []
    for reference, pair in enumerate(pairs):
        _, terms, leaves = _loss_and_leaves(pair, reference)
        loss = sum(term.sum() for term in terms)
        values.append(loss)
        gradients.append(torch.autograd.grad(loss, leaves))
    torch.testing.assert_close(*values, atol=3e-8, rtol=3e-8)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_decoupled_mean_and_covariance_inducing_locations_are_distinct(device):
    model, _, x, _, strategy = _pair(device)[0]
    output = model.model(x)
    mean_gradient = torch.autograd.grad(
        output.mean.sum(), strategy.inducing_points, retain_graph=True
    )[0]
    covariance_gradient = torch.autograd.grad(
        output.covariance_matrix.sum(), strategy.inducing_points
    )[0]
    assert bool(mean_gradient[0].abs().sum() > 0)
    assert bool(covariance_gradient[1].abs().sum() > 0)
    torch.testing.assert_close(mean_gradient[1], torch.zeros_like(mean_gradient[1]), atol=0, rtol=0)
    torch.testing.assert_close(
        covariance_gradient[0], torch.zeros_like(covariance_gradient[0]), atol=0, rtol=0
    )


def test_decoupled_rejects_source_unsupported_delta_family():
    x = torch.linspace(0.1, 0.9, 6, dtype=torch.double).unsqueeze(-1)
    for family in (
        variational.DeltaVariationalDistribution,
        gpytorch.variational.DeltaVariationalDistribution,
    ):
        with pytest.raises(NotImplementedError, match="does not work with Delta"):
            SingleTaskVariationalGP(
                x,
                x.sin(),
                inducing_points=x[:3],
                variational_distribution=family(3),
                variational_strategy=variational.BatchDecoupledVariationalStrategy,
            )


@pytest.mark.parametrize("device", DEVICES)
def test_unmodified_source_independent_wrapper_drops_decoupled_kl_override(device):
    (actual, _, x, _, _), (reference, source, ref_x, _, base) = _pair(device, "outputs")
    actual.model(x)
    reference.model(ref_x)
    q = source()
    covariance = q.covariance_matrix
    gaussian_kl = 0.5 * (
        q.mean.square().sum(-1)
        + covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        - 3
        - torch.linalg.slogdet(covariance).logabsdet
    )
    constant = 0.5 * 3 * math.log(2 * math.pi)
    torch.testing.assert_close(base.kl_divergence(), gaussian_kl + constant, atol=1e-12, rtol=1e-12)
    # This is the unchanged-source failure evidence, not the corrected oracle.
    torch.testing.assert_close(
        reference.model.variational_strategy.kl_divergence(),
        gaussian_kl.sum(),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        actual.model.variational_strategy.kl_divergence(),
        gaussian_kl.sum() + 2 * constant,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_decoupled_adam_steps_match_independent_source(device):
    pairs = _pair(device, "mean_var_last")
    optimizers = [torch.optim.Adam(pair[0].parameters(), lr=0.005) for pair in pairs]
    for _ in range(4):
        losses = []
        for reference, (pair, optimizer) in enumerate(zip(pairs, optimizers, strict=True)):
            optimizer.zero_grad()
            _, terms, _ = _loss_and_leaves(pair, reference)
            loss = -(terms[0] - terms[1] + terms[2]).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses, atol=3e-8, rtol=3e-8)
    actual, reference = [dict(pair[0].named_parameters()) for pair in pairs]
    assert actual.keys() == reference.keys()
    for name, parameter in actual.items():
        torch.testing.assert_close(parameter, reference[name], atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", LAYOUTS[:4])
def test_decoupled_float32_device_values_and_first_vjps(device, layout):
    values, gradients = [], []
    for reference, pair in enumerate(_pair(device, layout, dtype=torch.float32)):
        output, terms, leaves = _loss_and_leaves(pair, reference)
        loss = terms[0] - terms[1] + terms[2]
        values.append((output.mean, output.variance, loss))
        gradients.append(torch.autograd.grad(loss.sum(), leaves))
        assert all(
            value.device == pair[2].device and value.dtype == torch.float32
            for value in gradients[-1]
        )
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_decoupled_default_pathwise_update_does_not_silently_treat_both_sets_as_models():
    model, _, x, _, _ = _pair("cpu")[0]
    with pytest.raises(UnsupportedError, match="custom pathwise update"):
        gaussian_update(model, x.new_zeros(3, 2, 3))
    with pytest.raises(UnsupportedError, match="custom pathwise update"):
        draw_matheron_paths(model, torch.Size([3]))
