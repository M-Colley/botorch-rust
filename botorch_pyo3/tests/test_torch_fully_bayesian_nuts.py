"""Device-resident potentials, NUTS transitions and fully Bayesian fitting."""

# ruff: noqa: E402
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from botorch_rs import _botorch_rs as _rs
from botorch_rs.models._torch_backend_mcmc import TorchMCMCPotential, sample_torch_nuts

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _data(device="cpu", gradients=False):
    x = torch.tensor(
        [[0.12, 0.7], [0.34, 0.24], [0.65, 0.6], [0.88, 0.9]],
        device=device,
        dtype=torch.double,
        requires_grad=gradients,
    )
    y = torch.tensor(
        [[0.1], [0.9], [0.4], [-0.2]], device=device, dtype=torch.double, requires_grad=gradients
    )
    noise = torch.full_like(y, 0.03, requires_grad=gradients)
    return x, y, noise


def _native(x, y, noise, kernel, warp):
    return _rs.PyroModel(
        kernel,
        (x.detach().cpu().reshape(-1).tolist(), list(x.shape)),
        y.detach().cpu().reshape(-1).tolist(),
        None if noise is None else noise.detach().cpu().reshape(-1).tolist(),
        outputscale_prior=(2.0, 0.15) if kernel == "saas" else None,
        warp_indices=[0, 1] if warp else None,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel", ["matern", "saas", "linear"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("warp", [False, True])
def test_potential_native_value_gradient_hessian_and_batch(device, kernel, fixed, warp):
    x, y, noise = _data(device, gradients=True)
    noise = noise if fixed else None
    potential = TorchMCMCPotential(
        x, y, noise, kernel=kernel, warp_indices=[0, 1] if warp else None
    )
    native = _native(x, y, noise, kernel, warp)
    assert potential.sites() == native.sites()
    z = torch.linspace(-0.6, 0.4, potential.dim, dtype=x.dtype, device=device, requires_grad=True)
    value, gradient = potential.value_and_grad(z, create_graph=True)
    expected, expected_gradient = native.potential(z.detach().cpu().tolist())
    torch.testing.assert_close(value, x.new_tensor(expected), atol=3e-11, rtol=3e-12)
    torch.testing.assert_close(gradient, x.new_tensor(expected_gradient), atol=3e-10, rtol=3e-11)
    direction = torch.linspace(0.1, 0.4, z.numel(), device=device, dtype=x.dtype)
    (hessian_vector,) = torch.autograd.grad((gradient * direction).sum(), z, retain_graph=True)
    eps = 2e-5
    _, plus = native.potential((z.detach() + eps * direction).cpu().tolist())
    _, minus = native.potential((z.detach() - eps * direction).cpu().tolist())
    torch.testing.assert_close(
        hessian_vector,
        x.new_tensor(plus).sub(x.new_tensor(minus)) / (2 * eps),
        atol=3e-8,
        rtol=2e-7,
    )
    data = (x, y) if noise is None else (x, y, noise)
    first = torch.autograd.grad(value, data, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), data)
    assert all(g.device == x.device and bool(torch.isfinite(g).all()) for g in (*first, *second))
    states = torch.stack((z.detach(), z.detach() + 0.1))
    batched = potential(states)
    torch.testing.assert_close(batched, torch.stack([potential(s) for s in states]))
    processed = potential.postprocess(states)
    for name, shape, values in native.postprocess(states.cpu().tolist()):
        torch.testing.assert_close(
            processed[name], x.new_tensor(values).reshape(2, *shape), atol=2e-14, rtol=2e-14
        )


@pytest.mark.parametrize("kernel", ["matern", "saas", "linear"])
def test_potential_independent_numpyro(kernel):
    pytest.importorskip("jax")
    pytest.importorskip("numpyro")
    pytest.importorskip("botorch")
    from botorch.models import fully_bayesian as reference

    pytest.importorskip("botorch")
    from fully_bayesian_helpers import jax_x64, numpyro_potential

    x, y, noise = _data()
    family = {
        "matern": "FullyBayesianSingleTaskGP",
        "saas": "SaasFullyBayesianSingleTaskGP",
        "linear": "FullyBayesianLinearSingleTaskGP",
    }[kernel]
    with jax_x64():
        model = getattr(reference, family)(
            x, y, noise, use_input_warping=True, outcome_transform=None
        )
    potential = TorchMCMCPotential(x, y, noise, kernel=kernel, warp_indices=[0, 1])
    z = torch.linspace(-0.4, 0.3, potential.dim, dtype=x.dtype)
    actual, gradient = potential.value_and_grad(z)
    expected, expected_gradient = numpyro_potential(model, z.tolist())
    torch.testing.assert_close(actual, x.new_tensor(expected), atol=2e-10, rtol=2e-11)
    torch.testing.assert_close(gradient, x.new_tensor(expected_gradient), atol=2e-9, rtol=2e-10)


@pytest.fixture(scope="module")
def reference_run():
    pytest.importorskip("jax")
    pytest.importorskip("numpyro")
    pytest.importorskip("botorch")
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

    pytest.importorskip("botorch")
    from fully_bayesian_helpers import bt_fit_recorded, jax_x64

    x, y, noise = _data()
    with jax_x64():
        model = SaasFullyBayesianSingleTaskGP(x, y, noise, outcome_transform=None)
    return bt_fit_recorded(model, warmup_steps=24, num_samples=5, thinning=1, max_tree_depth=3)


@pytest.mark.parametrize("device", DEVICES)
def test_nuts_replays_independent_numpyro_trajectory(device, reference_run):
    x, y, noise = _data(device)
    potential = TorchMCMCPotential(x, y, noise)
    result = sample_torch_nuts(
        potential,
        x.new_tensor(reference_run.z0),
        num_warmup=24,
        num_samples=5,
        max_tree_depth=3,
        draws=reference_run.draws,
    )
    assert result["replay_first_flip"] is None
    assert result["replay_draws_consumed"] == len(reference_run.draws)
    assert len(result["trace"]) == len(reference_run.states)
    for actual, (_, expected, step) in zip(result["trace"], reference_run.states, strict=True):
        assert actual["z"].device == x.device
        torch.testing.assert_close(actual["z"], x.new_tensor(expected), atol=3e-8, rtol=3e-9)
        assert actual["step_size"] == pytest.approx(step, abs=2e-9, rel=2e-9)
    assert result["samples"].device == x.device
    assert not result["samples"].requires_grad
    assert result["inverse_mass_matrix"].device == x.device


class _Gaussian:
    def __init__(self, like):
        self.covariance = like.new_tensor([[1.0, 0.6], [0.6, 2.0]])
        self.precision = torch.linalg.inv(self.covariance)

    def value_and_grad(self, z):
        gradient = self.precision @ z
        return 0.5 * torch.dot(z, gradient), gradient


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dense", [False, True])
def test_nuts_known_gaussian_seed_and_device(device, dense):
    initial = torch.zeros(2, device=device, dtype=torch.double)
    potential = _Gaussian(initial)
    state = torch.random.get_rng_state()
    result = sample_torch_nuts(
        potential,
        initial,
        num_warmup=75,
        num_samples=1500,
        max_tree_depth=4,
        seed=14,
        dense_mass=dense,
    )
    assert torch.equal(state, torch.random.get_rng_state())
    samples = result["samples"]
    assert samples.device == initial.device
    # Batch-means standard errors account for Markov-chain autocorrelation.
    # Thirty batches of fifty draws provide a finite Monte Carlo diagnostic
    # rather than treating correlated transitions as independent observations.
    batches = samples.reshape(30, 50, 2)
    standard_error = batches.mean(1).std(0) / math.sqrt(30)
    assert bool((samples.mean(0).abs() < 5 * standard_error).all())
    effective_samples = potential.covariance.diagonal() / standard_error.square()
    assert float(effective_samples.min()) > 60
    outer = batches.unsqueeze(-1) * batches.unsqueeze(-2)
    moment_error = outer.mean((0, 1)) - potential.covariance
    moment_standard_error = outer.mean(1).std(0) / math.sqrt(30)
    assert bool((moment_error.abs() < 5 * moment_standard_error).all())
    assert sum(row["diverging"] for row in result["trace"][75:]) < 15
    repeat = sample_torch_nuts(potential, initial, num_warmup=0, num_samples=3, seed=14)
    again = sample_torch_nuts(potential, initial, num_warmup=0, num_samples=3, seed=14)
    torch.testing.assert_close(repeat["samples"], again["samples"], atol=0, rtol=0)
    assert all(math.isfinite(float(item["potential_energy"])) for item in result["trace"])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel", ["matern", "saas", "linear"])
@pytest.mark.parametrize("fixed", [False, True])
def test_public_nuts_fit_has_device_samples_and_prediction_graphs(
    device, kernel, fixed, monkeypatch
):
    from botorch_rs.fit import fit_fully_bayesian_model_nuts
    from botorch_rs.models import fully_bayesian as models

    x, y, noise = _data(device, gradients=True)
    family = {
        "matern": "FullyBayesianSingleTaskGP",
        "saas": "SaasFullyBayesianSingleTaskGP",
        "linear": "FullyBayesianLinearSingleTaskGP",
    }[kernel]
    model = getattr(models, family)(
        x, y, noise if fixed else None, use_input_warping=True, backend="torch"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A Tensor NUTS fit must not create a CPU/f64 potential.")

    monkeypatch.setattr(model.pyro_model, "_potential", forbidden)
    before = torch.random.get_rng_state()
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=4,
        num_samples=5,
        thinning=2,
        max_tree_depth=2,
        seed=9,
        disable_progbar=True,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert model.num_mcmc_samples == 3
    assert not model.training
    result = model._nuts_result
    assert result["samples"].device == x.device
    assert result["samples"].shape[0] == 5
    assert len(result["trace"]) == 9
    assert all(parameter.device == x.device for parameter in model.parameters())
    query = x.detach()[:2].clone().requires_grad_()
    posterior = model.posterior(query)
    objective = posterior.mixture_mean.sum() + posterior.mixture_variance.sum()
    targets = (x, y, query, *model.parameters())
    if fixed:
        targets = (*targets, noise)
    first = torch.autograd.grad(objective, targets, create_graph=True)
    second = torch.autograd.grad(sum(item.square().sum() for item in first), targets)
    assert all(
        item.device == x.device and bool(torch.isfinite(item).all()) for item in (*first, *second)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_public_nuts_replay_and_failure_preserves_loaded_state(device, reference_run):
    from botorch_rs.fit import fit_fully_bayesian_model_nuts
    from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

    x, y, noise = _data(device)
    model = SaasFullyBayesianSingleTaskGP(x, y, noise, backend="torch")
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=24,
        num_samples=5,
        thinning=2,
        max_tree_depth=3,
        draws=reference_run.draws,
        init_params=reference_run.z0,
        disable_progbar=True,
    )
    torch.testing.assert_close(
        model._nuts_result["samples"],
        x.new_tensor([row[1] for row in reference_run.states[24:]]),
        atol=3e-8,
        rtol=3e-9,
    )
    original = model.posterior(x[:2]).mean.clone()
    leaves = tuple(model.parameters())
    cases = [
        ({"thinning": 0}, ValueError),
        ({"warmup_steps": -1}, ValueError),
        ({"draws": []}, ValueError),
        ({"draws": [], "init_params": reference_run.z0}, ValueError),
        ({"draws": [], "init_params": [float("nan")] * len(reference_run.z0)}, ValueError),
    ]
    for options, exception in cases:
        with pytest.raises(exception):
            fit_fully_bayesian_model_nuts(model, **options)
        assert tuple(model.parameters()) == leaves
        torch.testing.assert_close(model.posterior(x[:2]).mean, original, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_potential_and_chain_remain_float32(device):
    x, y, noise = (item.float() for item in _data(device))
    potential = TorchMCMCPotential(x, y, noise, kernel="matern")
    z = torch.zeros(potential.dim, device=device, dtype=x.dtype)
    result = sample_torch_nuts(potential, z, num_warmup=2, num_samples=2, max_tree_depth=2)
    assert result["samples"].dtype == torch.float32
    assert bool(torch.isfinite(result["samples"]).all())


def test_nuts_input_and_divergence_validation():
    x, y, noise = _data()
    potential = TorchMCMCPotential(x, y, noise)
    z = x.new_zeros(potential.dim)
    for options in (
        {"num_samples": 0},
        {"max_tree_depth": 0},
        {"step_size": math.nan},
        {"target_accept_prob": 1},
        {"num_warmup": -1},
    ):
        with pytest.raises(ValueError):
            sample_torch_nuts(potential, z, **options)
    with torch.inference_mode(), pytest.raises(Exception, match="inference_mode"):
        sample_torch_nuts(potential, z)
    result = sample_torch_nuts(
        potential, z, num_warmup=0, num_samples=3, max_tree_depth=2, step_size=1e4
    )
    assert all(item["diverging"] for item in result["trace"])
    torch.testing.assert_close(result["samples"], z.expand(3, -1))
