"""Supplied q modules retain their forward law, identity and natural-gradient VJP."""

# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskVariationalGP as ReferenceGP
from gpytorch.mlls import VariationalELBO as ReferenceELBO

from botorch_rs.kernels import RBFKernel
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.sampling.pathwise import gaussian_update

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NATURAL = ["NaturalVariationalDistribution", "TrilNaturalVariationalDistribution"]


class ShiftedCholesky(gpytorch.variational.CholeskyVariationalDistribution):
    """A source-valid subclass whose extra leaf changes the actual q law."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter("shift", torch.nn.Parameter(torch.tensor(0.13)))
        self.register_prior("shift_prior", gpytorch.priors.NormalPrior(0, 1), "shift")
        self.register_buffer("covariance_multiplier", torch.tensor(0.81))
        self.initialization_calls = 0

    def initialize_variational_distribution(self, prior):
        self.initialization_calls += 1
        assert isinstance(prior, gpytorch.distributions.MultivariateNormal)
        super().initialize_variational_distribution(prior)

    def forward(self):
        q = super().forward()
        return gpytorch.distributions.MultivariateNormal(
            q.mean + self.shift, q.lazy_covariance_matrix * self.covariance_multiplier
        )


def _pair(device, family, unwhitened=False, outputs=1, batch=False, dtype=torch.double):
    pairs = []
    for reference in (False, True):
        x = torch.linspace(0.1, 0.9, 10, device=device, dtype=dtype).reshape(5, 2)
        y = x[..., :1].sin()
        if outputs > 1:
            y = torch.cat((y, y * 0.6 + 0.2), -1)
        if batch:
            x, y = x.expand(2, 5, 2).clone(), y.expand(2, 5, outputs).clone()
            x[1] += 0.02
            y[1] += 0.07
        x.requires_grad_()
        y.requires_grad_()
        aug = x.shape[:-2] + ((outputs,) if outputs > 1 else ())
        cls = family if isinstance(family, type) else getattr(gpytorch.variational, family)
        source = cls(3, batch_shape=aug, mean_init_std=0).to(x)
        kernel = (gpytorch.kernels.RBFKernel if reference else RBFKernel)(
            ard_num_dims=2, batch_shape=aug
        )
        likelihood = (
            gpytorch.likelihoods.GaussianLikelihood(batch_shape=x.shape[:-2])
            if outputs == 1
            else None
        )
        model = (ReferenceGP if reference else SingleTaskVariationalGP)(
            x,
            y,
            likelihood=likelihood,
            covar_module=kernel,
            inducing_points=x.new_tensor([[0.12, 0.2], [0.44, 0.51], [0.84, 0.92]]),
            variational_distribution=source,
            variational_strategy=getattr(
                gpytorch.variational,
                "UnwhitenedVariationalStrategy" if unwhitened else "VariationalStrategy",
            ),
            outcome_transform=None,
        ).to(x)
        strategy = model.model.variational_strategy
        if outputs > 1:
            strategy = strategy.base_variational_strategy
        assert strategy._variational_distribution is source
        # Fix a nontrivial q in a parameterization-independent way. This invokes
        # the supplied module's own initialization, exactly as the source does.
        source.initialize_variational_distribution(
            gpytorch.distributions.MultivariateNormal(
                x.new_tensor([-0.2, 0.15, 0.3]).expand(*aug, 3),
                x.new_tensor([[0.7, 0.04, 0.02], [0.04, 0.9, 0.06], [0.02, 0.06, 1.2]]).expand(
                    *aug, 3, 3
                ),
            )
        )
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            model.model.covar_module.lengthscale = x.new_tensor([0.4, 0.7])
            model.model.mean_module.constant = x.new_tensor(0.12)
            model.likelihood.noise = x.new_tensor(0.08)
        pairs.append((model, source, x, y))
    return pairs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", NATURAL + [ShiftedCholesky])
@pytest.mark.parametrize("unwhitened", [False, True])
@pytest.mark.parametrize("outputs,batch", [(1, False), (1, True), (2, False)])
def test_supplied_q_law_and_complete_elbo_vjps(device, family, unwhitened, outputs, batch):
    pairs = _pair(device, family, unwhitened, outputs, batch)
    values, gradients = [], []
    for reference, (model, source, x, y) in enumerate(pairs):
        model.train()
        output = model.model(x)
        mll = (ReferenceELBO if reference else VariationalELBO)(
            model.likelihood, model.model, 11, beta=0.7, combine_terms=False
        )
        terms = mll(output, y.squeeze(-1) if outputs == 1 else y)
        values.append((output.mean, output.variance, output.covariance_matrix, *terms))
        parameters = dict(model.named_parameters())
        leaves = tuple(parameters[name] for name in sorted(parameters)) + (x, y)
        gradients.append(torch.autograd.grad(sum(term.sum() for term in terms), leaves))
        assert all(
            value is dict(model.model.named_parameters())[name]
            for name, value in model.model.named_variational_parameters()
        )
        assert {id(p) for p in model.model.variational_parameters()} == {
            id(p) for p in source.parameters()
        }
        assert not {id(p) for p in model.model.variational_parameters()} & {
            id(p) for p in model.model.hyperparameters()
        }
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", NATURAL)
@pytest.mark.parametrize("unwhitened", [False, True])
def test_real_natural_gradient_steps_match_reference(device, family, unwhitened):
    pairs = _pair(device, family, unwhitened, batch=True)
    mlls = [
        (ReferenceELBO if reference else VariationalELBO)(model.likelihood, model.model, 5)
        for reference, (model, _, _, _) in enumerate(pairs)
    ]
    optimizers = [
        gpytorch.optim.NGD(model.model.variational_parameters(), num_data=5, lr=0.025)
        for model, _, _, _ in pairs
    ]
    for _ in range(4):
        losses = []
        for (model, source, x, y), mll, optimizer in zip(pairs, mlls, optimizers, strict=True):
            optimizer.zero_grad()
            loss = -mll(model.model(x), y.squeeze(-1)).sum()
            loss.backward()
            assert all(bool(torch.isfinite(p.grad).all()) for p in source.parameters())
            optimizer.step()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses, atol=3e-8, rtol=3e-8)
    for actual, expected in zip(pairs[0][1].parameters(), pairs[1][1].parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("unwhitened", [False, True])
def test_custom_module_initialization_identity_registration_and_hessians(device, unwhitened):
    (actual, source, x, y), (expected, reference, ref_x, ref_y) = _pair(
        device, ShiftedCholesky, unwhitened
    )
    source.shift.requires_grad_(False)
    reference.shift.requires_grad_(False)
    actual._strategy.variational_params_initialized.fill_(0)
    expected.model.variational_strategy.variational_params_initialized.fill_(0)
    losses, firsts, all_leaves = [], [], []
    for index, (model, data_x, data_y) in enumerate(((actual, x, y), (expected, ref_x, ref_y))):
        output = model.model(data_x)
        mll = (ReferenceELBO if index else VariationalELBO)(model.likelihood, model.model, 5)
        loss = -mll(output, data_y.squeeze(-1)).sum()
        parameters = dict(model.named_parameters())
        leaves = tuple(
            parameters[name] for name in sorted(parameters) if parameters[name].requires_grad
        ) + (data_x, data_y)
        first = torch.autograd.grad(loss, leaves, create_graph=not index)
        losses.append(loss)
        firsts.append(first)
        all_leaves.append(leaves)
    torch.testing.assert_close(*losses, atol=3e-8, rtol=3e-8)
    for a, b in zip(*firsts, strict=True):
        torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-7)
    directions = [torch.full_like(value, 0.13) for value in all_leaves[0]]
    actual_hvp = torch.autograd.grad(
        sum(
            (gradient * direction).sum()
            for gradient, direction in zip(firsts[0], directions, strict=True)
        ),
        all_leaves[0],
    )
    # Pinned linear_operator Triangular.solve/matmul has a known matrix-Hessian
    # defect (upstream PR152). Finite-difference the independent first gradients
    # rather than treating that detached second derivative as an oracle.
    base = [value.detach().clone() for value in all_leaves[1]]
    finite = []
    epsilon = 1e-5
    for sign in (-1, 1):
        with torch.no_grad():
            for leaf, value, direction in zip(all_leaves[1], base, directions, strict=True):
                leaf.copy_(value + sign * epsilon * direction)
        expected.train()
        mll = ReferenceELBO(expected.likelihood, expected.model, 5)
        loss = -mll(expected.model(ref_x), ref_y.squeeze(-1)).sum()
        finite.append(torch.autograd.grad(loss, all_leaves[1]))
    for actual_value, minus, plus in zip(actual_hvp, *finite, strict=True):
        torch.testing.assert_close(
            actual_value, (plus - minus) / (2 * epsilon), atol=3e-6, rtol=3e-6
        )
    assert source.initialization_calls == reference.initialization_calls == 2
    assert actual._strategy._variational_distribution is source
    assert not source.shift.requires_grad
    key = "model.variational_strategy._variational_distribution.shift"
    assert dict(actual.named_parameters())[key] is source.shift
    prefix = "model.variational_strategy._variational_distribution."
    assert {
        key.removeprefix(prefix) for key in actual.state_dict() if key.startswith(prefix)
    } == set(source.state_dict())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", NATURAL)
@pytest.mark.parametrize("unwhitened", [False, True])
def test_supplied_natural_first_call_initializes_from_the_correct_prior(device, family, unwhitened):
    pairs = _pair(device, family, unwhitened)
    outputs = []
    for reference, (model, source, x, y) in enumerate(pairs):
        strategy = model.model.variational_strategy
        strategy.variational_params_initialized.fill_(0)
        output = model.model(x)
        cls = ReferenceELBO if reference else VariationalELBO
        outputs.append(
            (
                output.mean,
                output.variance,
                cls(model.likelihood, model.model, 5)(output, y.squeeze(-1)),
            )
        )
        assert bool(strategy.variational_params_initialized)
        assert strategy._variational_distribution is source
    for actual, expected in zip(*outputs, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", NATURAL)
def test_float32_natural_values_and_vjps_remain_on_device(device, family):
    values, gradients = [], []
    for reference, (model, source, x, y) in enumerate(_pair(device, family, dtype=torch.float32)):
        cls = ReferenceELBO if reference else VariationalELBO
        loss = -cls(model.likelihood, model.model, 5)(model.model(x), y.squeeze(-1)).sum()
        values.append(loss)
        leaves = tuple(source.parameters()) + (x, y)
        gradients.append(torch.autograd.grad(loss, leaves))
        assert all(value.dtype == x.dtype and value.device == x.device for value in gradients[-1])
    torch.testing.assert_close(*values, atol=5e-5, rtol=5e-5)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)


def test_supplied_distribution_invalid_forward_output_is_not_coerced():
    class InvalidDistribution(ShiftedCholesky):
        def forward(self):
            return torch.distributions.Normal(self.variational_mean, 1)

    model, _, x, _ = _pair("cpu", InvalidDistribution)[0]
    with pytest.raises(RuntimeError, match="Invalid variational distribution"):
        model.model(x)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_forward_is_evaluated_once_for_each_saved_elbo_law(device):
    class CountingDistribution(ShiftedCholesky):
        calls = 0

        def forward(self):
            self.calls += 1
            q = super().forward()
            return gpytorch.distributions.MultivariateNormal(
                q.mean + 0.07 * self.calls,
                q.lazy_covariance_matrix * (1 + 0.1 * self.calls),
            )

    losses = []
    pairs = _pair(device, CountingDistribution)
    for reference, (model, source, x, y) in enumerate(pairs):
        cls = ReferenceELBO if reference else VariationalELBO
        mll = cls(model.likelihood, model.model, 5)
        output = model.model(x)
        losses.append(mll(output, y.squeeze(-1)))
        assert source.calls == 1
        if not reference:
            # A later forward must not replace this probability object's KL law.
            model.model(x)
            torch.testing.assert_close(mll(output, y.squeeze(-1)), losses[-1], atol=0, rtol=0)
            assert source.calls == 2
    torch.testing.assert_close(*losses, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", NATURAL + [ShiftedCholesky])
def test_supplied_current_q_matheron_update_retains_sampling_graph(device, family):
    model, source, x, _ = _pair(device, family)[0]
    model.eval()
    values = x.new_tensor([[0.1, -0.2, 0.3], [0.3, 0.4, -0.2]])
    normals = x.new_tensor([[-0.3, 0.2, 0.4], [0.5, -0.7, 0.1]])
    update = gaussian_update(model, values, noise=normals)
    z = model._strategy.inducing_points
    factor = torch.linalg.cholesky(model.model.covar_module(z, z))
    # q's own root, not an independently chosen root with a different coupling.
    target = model.model.mean_module(z) + (
        factor @ source().rsample(base_samples=normals).unsqueeze(-1)
    ).squeeze(-1)
    expected = torch.cholesky_solve((target - values).unsqueeze(-1), factor).squeeze(-1)
    torch.testing.assert_close(update.weight, expected, atol=3e-10, rtol=3e-10)
    loss = update(x).square().sum()
    gradients = torch.autograd.grad(loss, tuple(source.parameters()))
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
