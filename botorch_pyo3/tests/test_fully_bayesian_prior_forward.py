"""Loaded fully Bayesian priors and the pinned fitter's legacy JIT flag."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("jax")
pytest.importorskip("numpyro")

from fully_bayesian_helpers import DT, model_pair

from botorch_rs.fit import fit_fully_bayesian_model_nuts
from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP


def _query():
    return torch.tensor([[[[0.2, 0.4], [0.6, 0.3]]], [[[0.3, 0.7], [0.8, 0.2]]]], dtype=DT)


@pytest.mark.parametrize("family", ["matern", "saas", "linear"])
@pytest.mark.parametrize("warp", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_native_fully_bayesian_forward_matches_prior(family, warp, training, kind):
    reference, model, _, _ = model_pair(
        family, d=2, n=6, num=3, seed=952, warp=warp, transforms=False
    )
    query = _query()
    if kind == "list":
        actual_query = query.tolist()
    elif kind == "numpy":
        actual_query = query.numpy()
    else:
        actual_query = query
    for obj in (reference, model):
        obj.train(training, reset=False)
    expected, actual = reference.forward(query), model.forward(actual_query)
    for field in ("mean", "variance", "covariance_matrix"):
        torch.testing.assert_close(
            torch.as_tensor(getattr(actual, field), dtype=DT),
            getattr(expected, field),
            atol=2e-12,
            rtol=2e-11,
        )
    assert tuple(actual.batch_shape) == tuple(expected.batch_shape) == (2, 3)
    assert tuple(actual.event_shape) == (2,)
    assert tuple(torch.as_tensor(actual.rsample((4,))).shape) == (4, 2, 3, 2)


@pytest.mark.parametrize("family", ["matern", "saas", "linear"])
@pytest.mark.parametrize("warp", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_native_prior_forward_query_gradient_and_hvp(family, warp, training):
    reference, model, _, _ = model_pair(
        family, d=2, n=6, num=3, seed=952, warp=warp, transforms=False
    )
    results = []
    for obj in (reference, model):
        obj.train(training, reset=False)
        query = _query().requires_grad_()
        law = obj.forward(query)
        loss = law.mean.square().sum() + 0.2 * law.covariance_matrix.square().sum()
        (gradient,) = torch.autograd.grad(loss, query, create_graph=True)
        direction = torch.linspace(0.1, 0.7, query.numel(), dtype=DT).reshape_as(query)
        (hvp,) = torch.autograd.grad((gradient * direction).sum(), query)
        results.append((gradient, hvp))
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=2e-9)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_tensor_nuts_jit_flag_is_the_same_accepted_noop_as_reference(device):
    generator = torch.Generator().manual_seed(19)
    x = torch.rand(6, 2, generator=generator, dtype=DT).to(device)
    y = torch.sin(x[:, :1] * 3)
    noise = torch.full_like(y, 0.03)
    results = []
    for jit in (False, True):
        model = SaasFullyBayesianSingleTaskGP(x, y, noise, backend="torch")
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=2,
            num_samples=2,
            thinning=1,
            max_tree_depth=2,
            seed=7,
            jit_compile=jit,
        )
        results.append(model._nuts_result)
    torch.testing.assert_close(results[0]["samples"], results[1]["samples"], atol=0, rtol=0)
    for left, right in zip(results[0]["trace"], results[1]["trace"], strict=True):
        torch.testing.assert_close(left["z"], right["z"], atol=0, rtol=0)
        assert left["step_size"] == right["step_size"]
