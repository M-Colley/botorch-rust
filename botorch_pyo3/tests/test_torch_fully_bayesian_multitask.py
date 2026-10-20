"""Device-resident latent-task mixtures, potentials and NUTS fitting."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from botorch_rs import _botorch_rs as _rs
from botorch_rs.fit import fit_fully_bayesian_model_nuts
from botorch_rs.models._torch_backend_mcmc import TorchMCMCPotential
from botorch_rs.models.fully_bayesian_multitask import (
    MultitaskSaasPyroModel,
    SaasFullyBayesianMultiTaskGP,
)
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _data(device, feature=1, gradients=False, labels=(3, 7)):
    base = torch.tensor(
        [[0.1, 0.3], [0.3, 0.8], [0.7, 0.2], [0.8, 0.6], [0.4, 0.7], [0.9, 0.4]],
        dtype=torch.double,
        device=device,
    )
    tasks = base.new_tensor([labels[0], labels[1]] * 3).unsqueeze(-1)
    x = torch.cat([base[:, :feature], tasks, base[:, feature:]], -1).requires_grad_(gradients)
    y = (base[:, :1].sin() + base[:, 1:].square() + tasks * 0.1).requires_grad_(gradients)
    noise = (
        torch.linspace(0.02, 0.05, len(x), device=device, dtype=x.dtype)
        .unsqueeze(-1)
        .requires_grad_(gradients)
    )
    return x, y, noise


def _samples(like, tasks=2, rank=2, count=3, width=2):
    samples = {
        "mean": torch.linspace(
            -0.2, 0.4, count * tasks, device=like.device, dtype=like.dtype
        ).reshape(count, tasks),
        "lengthscale": torch.linspace(
            0.5, 1.1, count * 2, device=like.device, dtype=like.dtype
        ).reshape(count, 2),
        "outputscale": torch.linspace(0.8, 1.3, count, device=like.device, dtype=like.dtype),
        "noise": torch.linspace(0.03, 0.06, count, device=like.device, dtype=like.dtype),
        "latent_features": torch.linspace(
            0.1, 1.3, count * tasks * rank, device=like.device, dtype=like.dtype
        ).reshape(count, tasks, rank),
        "task_lengthscale": torch.linspace(
            0.6, 1.2, count * rank, device=like.device, dtype=like.dtype
        ).reshape(count, rank),
    }
    for i in (0, 1):
        samples[f"c{i}"] = torch.linspace(
            0.7 + i * 0.1, 1.5 + i * 0.1, count * width, device=like.device, dtype=like.dtype
        ).reshape(count, width)
    return samples


def _pair(
    device, feature=1, fixed=False, transforms=False, warp=False, subset=False, gradients=False
):
    pytest.importorskip("jax")
    pytest.importorskip("botorch")
    from botorch.models import fully_bayesian_multitask as reference
    from botorch.models.transforms.input import ChainedInputTransform as ReferenceChain
    from botorch.models.transforms.input import Normalize as ReferenceNormalize
    from botorch.models.transforms.input import Warp as ReferenceWarp
    from botorch.models.transforms.outcome import Standardize as ReferenceStandardize

    pytest.importorskip("botorch")
    from fully_bayesian_helpers import jax_x64

    base = [i for i in range(3) if i != feature]
    indices = [base[-1]] if subset else base
    models, datasets = [], []
    for index in (0, 1):
        data = _data(device, feature, gradients)
        x, y, noise = data
        namespace = reference if index else None
        kwargs = dict(rank=2)
        if transforms:
            kwargs["input_transform"] = (ReferenceNormalize if index else Normalize)(
                d=3, indices=base, bounds=x.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            )
            kwargs["outcome_transform"] = (ReferenceStandardize if index else Standardize)(1)
        kwargs["pyro_model"] = (
            namespace.MultitaskSaasPyroModel if index else MultitaskSaasPyroModel
        )(use_input_warping=warp, indices_to_warp=indices if subset else None, eps=0.03)
        with jax_x64():
            model = (
                namespace.SaasFullyBayesianMultiTaskGP if index else SaasFullyBayesianMultiTaskGP
            )(
                x,
                y,
                feature,
                noise if fixed else None,
                **kwargs,
                **({} if index else {"backend": "torch"}),
            )
        samples = _samples(x, width=len(indices))
        if index and warp:
            # Pinned b0dd builds its discarded Warp with the wrong width,
            # then discards it. Correct only that documented loader defect.
            model.pyro_model.use_input_warping = False
            model.load_mcmc_samples(samples)
            model.pyro_model.use_input_warping = True
            transform = ReferenceWarp(
                3,
                indices,
                batch_shape=torch.Size([3]),
                bounds=x.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                eps=0.03,
            )
            transform.concentration0.data = samples["c0"].unsqueeze(-2).clone()
            transform.concentration1.data = samples["c1"].unsqueeze(-2).clone()
            common = getattr(model, "input_transform", None)
            model.input_transform = (
                transform if common is None else ReferenceChain(user=common, warp=transform)
            )
        else:
            model.load_mcmc_samples(samples)
        models.append(model.eval())
        datasets.append(data)
    return models, datasets


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel", ["matern", "saas", "linear"])
@pytest.mark.parametrize("feature", [0, 2])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("warp", [False, True])
def test_task_potential_native_value_gradient_hessian_and_batch(
    device, kernel, feature, fixed, warp
):
    x, y, noise = _data(device, feature, gradients=True, labels=(0, 1))
    noise = noise if fixed else None
    indices = [i for i in range(3) if i != feature] if warp else None
    potential = TorchMCMCPotential(
        x, y, noise, kernel=kernel, task=(feature, 2, 2), warp_indices=indices
    )
    native = _rs.PyroModel(
        kernel,
        (x.detach().cpu().flatten().tolist(), list(x.shape)),
        y.detach().cpu().flatten().tolist(),
        None if noise is None else noise.detach().cpu().flatten().tolist(),
        outputscale_prior=(2.0, 0.15) if kernel == "saas" else None,
        warp_indices=indices,
        task=(feature, 2, 2),
    )
    assert potential.sites() == native.sites()
    z = torch.linspace(-0.4, 0.5, potential.dim, device=device, dtype=x.dtype, requires_grad=True)
    energy, grad = potential.value_and_grad(z, create_graph=True)
    expected, expected_grad = native.potential(z.detach().cpu().tolist())
    torch.testing.assert_close(energy, x.new_tensor(expected), atol=5e-11, rtol=3e-12)
    torch.testing.assert_close(grad, x.new_tensor(expected_grad), atol=3e-10, rtol=3e-11)
    direction = torch.linspace(0.1, 0.3, z.numel(), device=device, dtype=x.dtype)
    (hvp,) = torch.autograd.grad((grad * direction).sum(), z, retain_graph=True)
    eps = 2e-5
    _, plus = native.potential((z.detach() + eps * direction).cpu().tolist())
    _, minus = native.potential((z.detach() - eps * direction).cpu().tolist())
    torch.testing.assert_close(
        hvp, (x.new_tensor(plus) - x.new_tensor(minus)) / (2 * eps), atol=8e-8, rtol=3e-7
    )
    data = (x, y) if noise is None else (x, y, noise)
    first = torch.autograd.grad(energy, data, create_graph=True)
    second = torch.autograd.grad(sum(item.square().sum() for item in first), data)
    assert all(bool(torch.isfinite(item).all()) for item in (*first, *second))
    torch.testing.assert_close(first[0][:, feature], x.new_zeros(len(x)), atol=0, rtol=0)
    states = torch.stack((z.detach(), z.detach() + 0.1))
    torch.testing.assert_close(potential(states), torch.stack([potential(row) for row in states]))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("feature", [0, 1, 2])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("transforms", [False, True])
def test_loaded_task_mixture_matches_official_all_tasks_and_labeled_queries(
    device, feature, fixed, transforms
):
    (model, reference), data = _pair(device, feature, fixed, transforms)
    assert isinstance(model, SaasFullyBayesianMultiTaskGP)
    assert model.num_mcmc_samples == 3
    query = data[0][0].new_tensor([[[0.2, 0.6], [0.6, 0.4]], [[0.25, 0.5], [0.7, 0.3]]])
    for selected in (None, [7]):
        actual, expected = (
            model.posterior(query, output_indices=selected),
            reference.posterior(query, output_indices=selected),
        )
        for name in ("mean", "variance", "mixture_mean", "mixture_variance"):
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), atol=4e-9, rtol=3e-8
            )
        torch.testing.assert_close(
            actual.distribution.covariance_matrix,
            expected.distribution.covariance_matrix,
            atol=4e-9,
            rtol=3e-8,
        )
    labels = query.new_tensor([[7.0], [3.0]]).expand(2, 2, 1)
    full = torch.cat((query[..., :feature], labels, query[..., feature:]), -1)
    actual, expected = model.posterior(full), reference.posterior(full)
    torch.testing.assert_close(actual.mean, expected.mean, atol=4e-9, rtol=3e-8)
    torch.testing.assert_close(
        model.median_lengthscale, reference.median_lengthscale, atol=1e-14, rtol=1e-14
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("feature", [0, 1, 2])
@pytest.mark.parametrize("subset", [False, True])
def test_sampled_task_warp_preserves_labels_and_query_gradient(device, feature, subset):
    (model, reference), _ = _pair(
        device, feature, fixed=True, transforms=True, warp=True, subset=subset
    )
    inputs = [
        model._train_X.new_tensor([[0.2, 0.6], [0.6, 0.4]], requires_grad=True) for _ in range(2)
    ]
    results = []
    for obj, query in zip((model, reference), inputs, strict=True):
        posterior = obj.posterior(query)
        loss = posterior.mean.square().sum() + posterior.variance.sum()
        (gradient,) = torch.autograd.grad(loss, query, create_graph=obj is model)
        if obj is model:
            (second,) = torch.autograd.grad(gradient.sum(), query)
            assert bool(torch.isfinite(second).all())
        results.append((posterior.mean, posterior.variance, gradient))
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, atol=8e-8, rtol=5e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_task_nuts_really_fits_and_preserves_loaded_law_on_invalid_refit(device, monkeypatch):
    x, y, noise = _data(device, gradients=True)
    model = SaasFullyBayesianMultiTaskGP(
        x,
        y,
        1,
        noise,
        rank=2,
        pyro_model=MultitaskSaasPyroModel(use_input_warping=True, eps=0.03),
        backend="torch",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor NUTS must not call a flattened CPU/f64 potential.")

    monkeypatch.setattr(model.pyro_model, "_potential", forbidden)
    before = torch.random.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(x.device) if device == "cuda" else None
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=4,
        num_samples=5,
        thinning=2,
        max_tree_depth=2,
        seed=12,
        disable_progbar=True,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    if cuda_before is not None:
        assert torch.equal(cuda_before, torch.cuda.get_rng_state(x.device))
    assert model.num_mcmc_samples == 3
    assert model._nuts_result["samples"].device == x.device
    query = x.new_tensor([[0.3, 0.5]], requires_grad=True)
    posterior = model.posterior(query)
    loss = posterior.mean.square().sum() + posterior.distribution.covariance_matrix.square().sum()
    targets = (x, y, noise, query, *model.parameters())
    first = torch.autograd.grad(loss, targets, create_graph=True)
    second = torch.autograd.grad(sum(item.sum() for item in first), targets)
    assert all(bool(torch.isfinite(value).all()) for value in (*first, *second))
    state = copy.deepcopy(model.state_dict())
    original = posterior.mean.clone()
    with pytest.raises(ValueError):
        fit_fully_bayesian_model_nuts(model, thinning=0)
    torch.testing.assert_close(model.posterior(query).mean, original, atol=0, rtol=0)
    model.train(reset=False)
    assert model.num_mcmc_samples == 3
    model.train()
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.posterior(query)
    model.load_state_dict(state)
    torch.testing.assert_close(model.posterior(query).mean, original, atol=2e-14, rtol=2e-14)


def _posterior_gradient(model, data, query, create_graph):
    pytest.importorskip("botorch")
    from botorch import settings

    model.eval()
    x, y, noise = data
    parameters = dict(model.named_parameters())
    targets = (x, y, noise, query, *(parameters[name] for name in sorted(parameters)))
    with settings.propagate_grads(True):
        posterior = model.posterior(query)
        value = (
            posterior.mean.square().sum()
            + 0.2 * posterior.distribution.covariance_matrix.square().sum()
        )
        gradient = torch.autograd.grad(value, targets, create_graph=create_graph)
    return targets, gradient


@pytest.mark.parametrize("device", DEVICES)
def test_complete_loaded_task_data_query_parameter_gradient_and_mixed_hvp(device):
    pytest.importorskip("botorch")
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Reference

    pytest.importorskip("botorch")
    from fully_bayesian_helpers import jax_x64

    (model, reference), data = _pair(device, fixed=True, gradients=True)
    queries = [
        model._train_X.new_tensor([[0.2, 0.6], [0.6, 0.4]], requires_grad=True) for _ in range(2)
    ]
    targets, actual = _posterior_gradient(model, data[0], queries[0], True)
    _, expected = _posterior_gradient(reference, data[1], queries[1], False)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=4e-7)
    directions = [torch.full_like(value, 0.17) for value in targets]
    directions[0][:, 1] = 0  # Categorical labels do not admit continuous perturbations.
    hvp = torch.autograd.grad(
        sum((a * b).sum() for a, b in zip(actual, directions, strict=True)), targets
    )
    perturbed = []
    epsilon = 2e-5
    for sign in (-1, 1):
        shifted = [
            (value.detach() + sign * epsilon * directions[index]).requires_grad_()
            for index, value in enumerate(data[1])
        ]
        with jax_x64():
            other = Reference(*shifted[:2], task_feature=1, train_Yvar=shifted[2], rank=2)
        other.load_mcmc_samples(_samples(shifted[0]))
        with torch.no_grad():
            original = dict(reference.named_parameters())
            for name, value in other.named_parameters():
                value.copy_(original[name] + sign * epsilon * 0.17)
        query = (queries[1].detach() + sign * epsilon * 0.17).requires_grad_()
        perturbed.append(_posterior_gradient(other, shifted, query, False)[1])
    for actual, minus, plus in zip(hvp, *perturbed, strict=True):
        torch.testing.assert_close(actual, (plus - minus) / (2 * epsilon), atol=2e-6, rtol=4e-6)


@pytest.mark.parametrize("warp", [False, True])
def test_task_potential_independent_jax_density_and_derivative(warp):
    pytest.importorskip("botorch")
    from botorch.models.fully_bayesian_multitask import (
        MultitaskSaasPyroModel as ReferencePyro,
    )
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Reference

    pytest.importorskip("botorch")
    from fully_bayesian_helpers import jax_x64, numpyro_potential

    x, y, noise = _data("cpu", feature=2, labels=(0, 1))
    with jax_x64():
        model = Reference(x, y, 2, noise, rank=2, pyro_model=ReferencePyro(use_input_warping=warp))
    potential = TorchMCMCPotential(
        x, y, noise, task=(2, 2, 2), warp_indices=[0, 1] if warp else None
    )
    z = torch.linspace(-0.4, 0.5, potential.dim, dtype=x.dtype)
    value, gradient = potential.value_and_grad(z)
    expected, expected_gradient = numpyro_potential(model, z.tolist())
    torch.testing.assert_close(value, x.new_tensor(expected), atol=3e-10, rtol=3e-11)
    torch.testing.assert_close(gradient, x.new_tensor(expected_gradient), atol=2e-9, rtol=3e-10)


@pytest.fixture(scope="module")
def task_reference_run():
    pytest.importorskip("botorch")
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Reference

    pytest.importorskip("botorch")
    from fully_bayesian_helpers import bt_fit_recorded, jax_x64

    x, y, noise = _data("cpu", feature=2, labels=(0, 1))
    with jax_x64():
        model = Reference(x, y, 2, noise, rank=2)
    return bt_fit_recorded(model, warmup_steps=24, num_samples=5, thinning=1, max_tree_depth=3)


@pytest.fixture(scope="module")
def task_reference_precision(task_reference_run):
    """Measure reference trajectory conditioning without changing task labels."""
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Reference
    from fully_bayesian_helpers import (
        TWIN_SCALES,
        bt_fit_recorded,
        first_difference,
        jax_x64,
        numpyro_potential,
    )

    x, y, noise = _data("cpu", feature=2, labels=(0, 1))
    twins = []
    for index, scale in enumerate(TWIN_SCALES * 3):
        shifted = x.clone()
        generator = torch.Generator().manual_seed(570 + index)
        shifted[:, :2] *= 1 + scale * torch.randn(6, 2, dtype=x.dtype, generator=generator)
        torch.testing.assert_close(shifted[:, 2], x[:, 2], atol=0, rtol=0)
        with jax_x64():
            model = Reference(shifted, y, 2, noise, rank=2)
        twin = bt_fit_recorded(model, warmup_steps=24, num_samples=5, thinning=1, max_tree_depth=3)
        # Only trajectories following the same discrete decisions constitute
        # a local floating-point sensitivity control.
        assert len(twin.draws) == len(task_reference_run.draws)
        assert first_difference(task_reference_run.draws, twin.draws) == len(twin.draws)
        assert all(
            original[1] == perturbed[1]
            for original, perturbed in zip(task_reference_run.draws, twin.draws, strict=True)
        )
        twins.append(twin)
    with jax_x64():
        model = Reference(x, y, 2, noise, rank=2)
    forces = [numpyro_potential(model, z) for _, z, _ in task_reference_run.states]
    return twins, forces


@pytest.mark.parametrize("device", DEVICES)
def test_task_nuts_replays_numpyro_warmup_and_sample_states(
    device, task_reference_run, task_reference_precision
):
    x, y, noise = _data(device, feature=2, labels=(0, 1))
    model = SaasFullyBayesianMultiTaskGP(x, y, 2, noise, rank=2, backend="torch")
    potential = TorchMCMCPotential.from_model(model)
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=24,
        num_samples=5,
        thinning=2,
        max_tree_depth=3,
        draws=task_reference_run.draws,
        init_params=task_reference_run.z0,
    )
    result = model._nuts_result
    assert result["replay_first_flip"] is None
    assert result["replay_draws_consumed"] == len(task_reference_run.draws)
    twins, forces = task_reference_precision
    max_ratio = max_energy_error = max_force_error = 0.0
    for index, (actual, (_, expected, step)) in enumerate(
        zip(result["trace"], task_reference_run.states, strict=True)
    ):
        expected = x.new_tensor(expected)
        spread = torch.stack(
            [(x.new_tensor(twin.states[index][1]) - expected).abs() for twin in twins]
        ).amax(0)
        # Keep the original tolerance as a floor, then account for this
        # transition's independently measured NumPyro self-sensitivity.
        bound = torch.maximum(2e-7 + 2e-8 * expected.abs(), 10 * spread)
        ratio = float(((actual["z"] - expected).abs() / bound).max())
        max_ratio = max(max_ratio, ratio)
        assert ratio <= 1, (index, ratio)
        step_spread = max(abs(twin.states[index][2] - step) for twin in twins)
        assert abs(actual["step_size"] - step) <= max(2e-8 + 2e-8 * abs(step), 10 * step_spread)
        # A trajectory envelope alone could hide a wrong force. Check the
        # identical saved state against the independent JAX density as well.
        energy, gradient = potential.value_and_grad(expected)
        expected_energy, expected_gradient = (x.new_tensor(value) for value in forces[index])
        torch.testing.assert_close(energy, expected_energy, atol=3e-10, rtol=3e-11)
        torch.testing.assert_close(gradient, expected_gradient, atol=2e-9, rtol=3e-10)
        max_energy_error = max(max_energy_error, float((energy - expected_energy).abs()))
        max_force_error = max(max_force_error, float((gradient - expected_gradient).abs().max()))
    print(
        f"{device}: replay max state/bound={max_ratio:.6g}, "
        f"pointwise potential error={max_energy_error:.6g}, "
        f"pointwise gradient error={max_force_error:.6g}"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["matern", "linear"])
def test_latent_feature_pyro_compositions_loaded_law_graphs_and_real_nuts(device, family):
    pytest.importorskip("botorch")
    from botorch.models import fully_bayesian as reference_fb
    from botorch.models.fully_bayesian_multitask import (
        LatentFeatureMultiTaskPyroMixin as ReferenceMixin,
    )
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Reference
    from botorch.models.transforms.input import Normalize as ReferenceNormalize

    pytest.importorskip("botorch")
    from fully_bayesian_helpers import jax_x64

    from botorch_rs.models import fully_bayesian as fb
    from botorch_rs.models.fully_bayesian_multitask import LatentFeatureMultiTaskPyroMixin

    own_base = fb.MaternPyroModel if family == "matern" else fb.LinearPyroModel
    reference_base = (
        reference_fb.MaternPyroModel if family == "matern" else reference_fb.LinearPyroModel
    )
    own_class = type("LatentTaskModel", (LatentFeatureMultiTaskPyroMixin, own_base), {})
    reference_class = type("ReferenceLatentTaskModel", (ReferenceMixin, reference_base), {})
    x, y, noise = _data(device, feature=1, gradients=True)
    model = SaasFullyBayesianMultiTaskGP(
        x, y, 1, noise, rank=2, pyro_model=own_class(), backend="torch"
    )
    with jax_x64():
        reference = Reference(
            x.detach(), y.detach(), 1, noise.detach(), rank=2, pyro_model=reference_class()
        )
    samples = _samples(x)
    samples.pop("outputscale")
    if family == "linear":
        samples["weight_variance"] = samples.pop("lengthscale")
    model.load_mcmc_samples(samples)
    if family == "linear":
        # The pinned Linear loader hardcodes a scalar ConstantMean and cannot
        # load the (S,T) mean produced by the public latent-feature mixin.
        with pytest.raises(RuntimeError, match="invalid for input of size"):
            reference.load_mcmc_samples(samples)

        def corrected_linear_load(mcmc_samples):
            scalar_samples = {**mcmc_samples, "mean": mcmc_samples["mean"][:, 0]}
            _, covariance, likelihood, transform = reference_base.load_mcmc_samples(
                reference.pyro_model, scalar_samples
            )
            batch = torch.Size([samples["mean"].shape[0]])
            kwargs = {"dtype": x.dtype, "device": x.device}
            mean = reference.pyro_model._build_mean_module(mcmc_samples, batch, **kwargs)
            covariance = reference.pyro_model._build_multitask_covariance(
                mcmc_samples, covariance, batch, **kwargs
            )
            return mean, covariance, likelihood, transform

        reference.pyro_model.load_mcmc_samples = corrected_linear_load
    reference.load_mcmc_samples(samples)
    if family == "linear":
        # The same pinned multitask loader drops LinearPyroModel's returned
        # centering transform, just as it drops the sampled Warp. Install the
        # correct full-width, non-task-only reference centering transform.
        reference.input_transform = ReferenceNormalize(
            3, indices=[0, 2], center=0.0, bounds=x.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        ).to(x)
    reference.eval()
    query = x.new_tensor([[0.2, 0.6], [0.6, 0.4]], requires_grad=True)
    actual, expected = model.posterior(query), reference.posterior(query)
    expected_mean = expected.mean
    expected_covariance = expected.distribution.covariance_matrix
    if family == "linear":
        # Lazy multiplication roots the singular Linear prior and adds jitter.
        # Evaluate the two reference kernels independently before multiplying,
        # then use the physical noisy Gaussian conditioning equations.
        full_query = torch.cat(
            [
                torch.cat([query[:, :1], query.new_full((2, 1), task), query[:, 1:]], -1)
                for task in (3, 7)
            ]
        )
        all_inputs = torch.cat([x.detach(), full_query], -2).unsqueeze(-3).expand(3, -1, -1)
        transformed = reference.transform_inputs(all_inputs)
        prior = reference.forward(transformed)
        mapped = transformed.clone()
        mapped[..., 1] = reference._map_tasks(transformed[..., 1])
        data_kernel, task_kernel = reference.covar_module.kernels
        covariance = data_kernel(mapped).to_dense() * task_kernel(mapped).to_dense()
        mean = prior.mean
        train_cov = covariance[..., :6, :6] + torch.diag_embed(noise.detach()[:, 0])
        cross = covariance[..., 6:, :6]
        residual = y.detach()[:, 0] - mean[..., :6]
        expected_mean = (
            (
                mean[..., 6:]
                + (cross @ torch.linalg.solve(train_cov, residual.unsqueeze(-1))).squeeze(-1)
            )
            .reshape(3, 2, 2)
            .transpose(-1, -2)
        )
        expected_covariance = covariance[..., 6:, 6:] - cross @ torch.linalg.solve(
            train_cov, cross.transpose(-1, -2)
        )
    torch.testing.assert_close(actual.mean, expected_mean, atol=3e-9, rtol=3e-8)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected_covariance,
        atol=3e-9,
        rtol=3e-8,
    )
    targets = (x, y, noise, query, *model.parameters())
    first = torch.autograd.grad(
        actual.mean.sum() + actual.variance.sum(), targets, create_graph=True
    )
    second = torch.autograd.grad(sum(value.square().sum() for value in first), targets)
    assert all(bool(torch.isfinite(value).all()) for value in (*first, *second))
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=2,
        num_samples=2,
        thinning=1,
        max_tree_depth=2,
        seed=17,
        disable_progbar=True,
    )
    assert model.num_mcmc_samples == 2
    assert bool(torch.isfinite(model.posterior(query).mean).all())


@pytest.mark.parametrize("device", DEVICES)
def test_float32_two_samples_shared_noise_atomic_reload_and_auto_backend(device):
    x, y, _ = (value.float().requires_grad_() for value in _data(device))
    model = SaasFullyBayesianMultiTaskGP(x, y, 1, rank=1)
    assert model._tensor_backend == "torch"
    samples = _samples(x, count=2, rank=1)
    model.load_mcmc_samples(samples)
    # S == T must not reinterpret the shared scalar noise as task-specific.
    actual_noise = model._noise_at(x.unsqueeze(-3).expand(2, -1, -1))
    torch.testing.assert_close(actual_noise, samples["noise"].unsqueeze(-1).expand(2, len(x)))
    query = x.new_tensor([[0.2, 0.4]], requires_grad=True)
    saved = model.posterior(query)
    saved_mean = saved.mean.clone()
    held = tuple(model.parameters())
    with pytest.raises(ValueError, match="latent_features"):
        model.load_mcmc_samples({**samples, "latent_features": x.new_zeros(2, 2, 2)})
    assert all(a is b for a, b in zip(held, model.parameters(), strict=True))
    torch.testing.assert_close(model.posterior(query).mean, saved_mean, atol=0, rtol=0)
    model.load_mcmc_samples({**samples, "mean": samples["mean"] + 0.1})
    torch.testing.assert_close(saved.mean, saved_mean, atol=0, rtol=0)
    posterior = model.posterior(query)
    loss = posterior.mean.sum() + posterior.variance.sum()
    targets = (x, y, query, *model.parameters())
    first = torch.autograd.grad(loss, targets, create_graph=True)
    second = torch.autograd.grad(sum(value.square().sum() for value in first), targets)
    assert all(bool(torch.isfinite(value).all()) for value in (*first, *second))


def test_task_custom_probabilistic_override_rejected_without_fitting():
    from botorch_rs.exceptions.errors import UnsupportedError

    class Custom(MultitaskSaasPyroModel):
        def sample_task_lengthscale(self):
            raise AssertionError("A custom law must not be silently replaced.")

    x, y, noise = _data("cpu")
    with pytest.raises(UnsupportedError, match="custom sample_task_lengthscale"):
        SaasFullyBayesianMultiTaskGP(x, y, 1, noise, pyro_model=Custom(), backend="torch")
