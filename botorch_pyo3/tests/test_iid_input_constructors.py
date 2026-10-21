"""Input-constructor IID requests reach real sampler and acquisition evaluation."""

from __future__ import annotations

import pytest

from botorch_rs.acquisition.input_constructors import _get_sampler, optimize_objective
from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler


@pytest.mark.parametrize("qmc", [False, True])
def test_requested_sampler_family_and_shape(qmc):
    sampler = _get_sampler(mc_samples=11, qmc=qmc)
    assert isinstance(sampler, SobolQMCNormalSampler if qmc else IIDNormalSampler)
    assert tuple(sampler.sample_shape) == (11,)


def test_iid_sampler_draws_match_official_shared_gaussian():
    torch = pytest.importorskip("torch")
    constructors = pytest.importorskip("botorch.acquisition.input_constructors")
    distributions = pytest.importorskip("gpytorch.distributions")
    reference_posteriors = pytest.importorskip("botorch.posteriors.gpytorch")
    from botorch_rs.posteriors.gpytorch import GPyTorchPosterior

    mean = torch.tensor([0.2, -0.1], dtype=torch.double)
    covariance = torch.tensor([[0.8, 0.15], [0.15, 0.4]], dtype=torch.double)
    distribution = distributions.MultivariateNormal(mean, covariance)
    torch.manual_seed(31)
    actual = _get_sampler(11, qmc=False)
    torch.manual_seed(31)
    expected = constructors._get_sampler(11, qmc=False)
    assert actual.seed == expected.seed
    torch.testing.assert_close(
        actual(GPyTorchPosterior(distribution=distribution)),
        expected(reference_posteriors.GPyTorchPosterior(distribution=distribution)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("backend,device", [("native", "cpu"), ("torch", "cpu"), ("torch", "cuda")])
@pytest.mark.parametrize("qmc", [False, True])
def test_optimize_objective_runs_requested_seeded_sampler(backend, device, qmc):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if backend == "torch":
        pytest.importorskip(
            "scipy", reason="The default Tensor optimize_objective driver uses SciPy."
        )
    from botorch_rs.acquisition.monte_carlo import qSimpleRegret
    from botorch_rs.acquisition.objective import GenericMCObjective
    from botorch_rs.models import SingleTaskGP

    X = torch.tensor([[0.05], [0.3], [0.62], [0.95]], device=device, dtype=torch.double)
    Y = (5 * X).sin()
    model = SingleTaskGP(X, Y, outcome_transform=None, backend=backend)
    objective = GenericMCObjective(lambda samples, X=None: samples[..., 0].tanh())
    bounds = torch.tensor([[0.0], [1.0]], device=device, dtype=torch.double)
    initial = torch.tensor([[[0.2]], [[0.7]]], device=device, dtype=torch.double)
    candidates, value = optimize_objective(
        model,
        bounds,
        q=1,
        objective=objective,
        qmc=qmc,
        mc_samples=16,
        seed_inner=71,
        optimizer_options={"num_restarts": 2, "maxiter": 4},
        batch_initial_conditions=initial,
    )
    sampler_cls = SobolQMCNormalSampler if qmc else IIDNormalSampler
    expected_acquisition = qSimpleRegret(
        model, objective=objective, sampler=sampler_cls(sample_shape=(16,), seed=71)
    )
    torch.testing.assert_close(
        value, expected_acquisition(candidates).squeeze(), rtol=1e-11, atol=1e-11
    )
    assert candidates.device == bounds.device
    assert bool(((candidates >= bounds[0]) & (candidates <= bounds[1])).all())
