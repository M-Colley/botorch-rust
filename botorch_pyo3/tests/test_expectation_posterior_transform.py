"""Native expectation transforms against official BoTorch b0dd01be6."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

from botorch_rs._arrays import kind_of, to_flat
from botorch_rs.acquisition.objective import ExpectationPosteriorTransform
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior
from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weights", [None, [[1.0, 3.0], [3.0, 1.0]]])
def test_evaluate_array_kinds(kind: str, weights: list | None) -> None:
    transform = ExpectationPosteriorTransform(2, as_kind(weights, kind) if weights else None)
    outcomes = as_kind([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], kind)
    result = transform.evaluate(outcomes)
    check_kind(result, kind)
    expected = [[[2.0, 3.0], [6.0, 7.0]]] if weights is None else [[[2.5, 2.5], [6.5, 6.5]]]
    assert shape(result) == (1, 2, 2)
    assert_close(result, expected)


@pytest.mark.parametrize(
    "n_w,weights",
    [
        (0, None),
        (-1, None),
        (True, None),
        (2.5, None),
        (2, [[0.0], [0.0]]),
        (2, [[-1.0], [2.0]]),
        (2, [[math.inf], [1.0]]),
        (2, [[1.0]]),
        (2, [1.0, 2.0]),
    ],
)
def test_invalid_weights(n_w: object, weights: object) -> None:
    with pytest.raises(ValueError):
        ExpectationPosteriorTransform(n_w, weights)  # type: ignore[arg-type]


def test_incompatible_q_and_outputs() -> None:
    transform = ExpectationPosteriorTransform(2, [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(BotorchTensorDimensionError, match="divisible"):
        transform.evaluate([[1.0, 2.0]])
    with pytest.raises(BotorchTensorDimensionError, match="columns"):
        transform.evaluate([[1.0], [2.0]])


@pytest.mark.parametrize("q,m,batch", [(1, 1, ()), (3, 1, (2,)), (2, 2, ()), (2, 3, (2,))])
@pytest.mark.parametrize("weighted", [False, True])
def test_full_correlated_gaussian_parity(q: int, m: int, batch: tuple, weighted: bool) -> None:
    importorskip_botorch()
    import torch
    from botorch.acquisition.objective import ExpectationPosteriorTransform as BtTransform
    from botorch.posteriors.gpytorch import GPyTorchPosterior as BtPosterior
    from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

    nw = 3
    generator = torch.Generator().manual_seed(906)
    mean = torch.randn(*batch, q * nw, m, generator=generator, dtype=torch.float64)
    n = q * nw * m
    factor = torch.randn(*batch, n, n, generator=generator, dtype=torch.float64)
    covariance = factor @ factor.transpose(-1, -2) + 0.1 * torch.eye(n, dtype=torch.float64)
    distribution = (
        MultivariateNormal(mean.squeeze(-1), covariance)
        if m == 1
        else MultitaskMultivariateNormal(mean, covariance, interleaved=False)
    )
    bt = BtPosterior(distribution)
    rs = GPyTorchPosterior(None, ([], (*batch, q * nw, 0)), kind_of(mean), False, num_outputs=m)
    rs._joint = (to_flat(mean), to_flat(covariance))
    variance = covariance.diagonal(dim1=-2, dim2=-1).reshape(*batch, m, q * nw).transpose(-1, -2)
    rs._marginals = (list(to_flat(mean)[0]), list(to_flat(variance)[0]))
    weights = torch.rand(nw, m, generator=generator, dtype=torch.float64) if weighted else None
    tb, tr = BtTransform(nw, weights), ExpectationPosteriorTransform(nw, weights)
    pb, pr = tb(bt), tr(rs)
    torch.testing.assert_close(pr.mean, pb.mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(pr.variance, pb.variance, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        pr.distribution.covariance_matrix, pb.distribution.covariance_matrix, rtol=1e-12, atol=1e-12
    )
    base_shape = (*batch, q, m) if m > 1 else (*batch, q)
    base = torch.randn(5, *base_shape, generator=generator, dtype=torch.float64)
    samples_b = pb.rsample_from_base_samples(torch.Size([5]), base)
    samples_r = pr.rsample_from_base_samples((5,), base)
    torch.testing.assert_close(samples_r, samples_b, rtol=1e-10, atol=1e-10)
    for sampler in (IIDNormalSampler((4,), seed=51), SobolQMCNormalSampler((4,), seed=51)):
        assert tuple(sampler(pr).shape) == (4, *batch, q, m)


def test_evaluate_torch_gradients_in_outcomes_and_weights() -> None:
    importorskip_botorch()
    import torch
    from botorch.acquisition.objective import ExpectationPosteriorTransform as BtTransform

    outcomes = torch.linspace(-1.0, 1.0, 24, dtype=torch.float64).reshape(2, 6, 2).requires_grad_()
    weights = torch.tensor(
        [[0.2, 0.3], [0.7, 0.1], [0.1, 0.6]], dtype=torch.float64, requires_grad=True
    )
    bt, rs = BtTransform(3, weights), ExpectationPosteriorTransform(3, weights)
    vb, vr = bt.evaluate(outcomes), rs.evaluate(outcomes)
    gb = torch.autograd.grad(vb.square().sum(), (outcomes, weights), retain_graph=True)
    gr = torch.autograd.grad(vr.square().sum(), (outcomes, weights))
    torch.testing.assert_close(vr, vb, rtol=1e-13, atol=1e-13)
    for actual, expected in zip(gr, gb, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_reject_interleaved_and_non_gaussian() -> None:
    posterior = GPyTorchPosterior(
        None, ([], (6, 0)), kind_of([]), False, num_outputs=2, interleaved=True
    )
    transform = ExpectationPosteriorTransform(3)
    with pytest.raises(UnsupportedError, match="interleaved"):
        transform(posterior)
    with pytest.raises(UnsupportedError, match="Gaussian"):
        transform(object())


@pytest.mark.parametrize("kind", KINDS)
def test_native_model_posterior_integration(kind: str) -> None:
    from botorch_rs.models import SingleTaskGP
    from botorch_rs.models.transforms.input import InputPerturbation

    x = as_kind([[0.1], [0.3], [0.6], [0.9]], kind)
    y = as_kind([[0.1], [0.9], [0.8], [0.2]], kind)
    perturbation = InputPerturbation(as_kind([[-0.03], [0.03]], kind))
    model = SingleTaskGP(x, y, input_transform=perturbation)
    query = as_kind([[0.2], [0.7]], kind)
    transform = ExpectationPosteriorTransform(2)
    actual = model.posterior(query, posterior_transform=transform)
    expected = transform(model.posterior(query))
    assert shape(actual.mean) == (2, 1)
    check_kind(actual.mean, kind)
    assert_close(actual.mean, expected.mean)
    assert_close(actual.distribution.covariance_matrix, expected.distribution.covariance_matrix)


def test_list_import_and_transform_remain_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from botorch_rs.acquisition.objective import ExpectationPosteriorTransform; t=ExpectationPosteriorTransform(2); assert t.evaluate([[1.0],[3.0]])==[[2.0]]; assert 'torch' not in sys.modules",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
