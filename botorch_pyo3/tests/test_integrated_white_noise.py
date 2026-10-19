"""Integrated white noise parity with the official BoTorch kernel.

The kernel was added after BoTorch 0.18.1. Set BOTORCH_RS_REFERENCE_ROOT to
an official checkout to compare against that source (audited at b0dd01be6).
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest

from botorch_rs.acquisition.analytic import PosteriorMean, PosteriorStandardDeviation
from botorch_rs.kernels import RBFKernel, ScaleKernel
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.kernels import IntegratedWhiteNoiseKernel
from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape


def official_kernel() -> type:
    importorskip_botorch()
    try:
        from botorch.models.kernels.integrated_white_noise import (
            IntegratedWhiteNoiseKernel as BtKernel,
        )
    except ImportError:
        reference = Path(
            os.environ.get(
                "BOTORCH_RS_REFERENCE_ROOT",
                Path(__file__).resolve().parents[2] / ".reference" / "botorch",
            )
        )
        source = reference / "botorch" / "models" / "kernels" / "integrated_white_noise.py"
        if not source.is_file():
            pytest.skip(
                "Requires BoTorch with IntegratedWhiteNoiseKernel or BOTORCH_RS_REFERENCE_ROOT"
            )
        spec = importlib.util.spec_from_file_location("official_integrated_white_noise", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        BtKernel = module.IntegratedWhiteNoiseKernel
    return BtKernel


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", [1, 2, 3, 4, 7])
def test_array_api_closed_forms_and_cross_diagonals(kind: str, order: int) -> None:
    kernel = IntegratedWhiteNoiseKernel(order)
    X = as_kind([[0.0], [0.2], [0.6], [1.5]], kind)
    Z = as_kind([[0.1], [0.4], [0.3], [2.0]], kind)
    covariance = kernel(X, Z)
    check_kind(covariance, kind)
    assert shape(covariance) == (4, 4)
    assert list(kernel.named_parameters()) == []
    assert kernel.lengthscale is None and not kernel.is_stationary
    from helpers import to_list

    matrix = to_list(covariance)
    assert_close(kernel(X, Z, diag=True), [matrix[i][i] for i in range(4)])
    diagonal = [
        x[0] ** (2 * order - 1) / (math.factorial(order - 1) ** 2 * (2 * order - 1))
        for x in to_list(X)
    ]
    assert_close(kernel(X, diag=True), diagonal)
    # A parameterless kernel must still broadcast over a kernel batch.
    batched = IntegratedWhiteNoiseKernel(order, batch_shape=(2,))
    assert shape(batched(X, Z)) == (2, 4, 4)
    assert_close(to_list(batched(X, Z))[1], matrix)
    # active_dims and composition execute as a native kernel tree.
    active = IntegratedWhiteNoiseKernel(order, active_dims=[1])
    pairs = as_kind([[10.0, 0.0], [9.0, 0.2], [8.0, 0.6], [7.0, 1.5]], kind)
    assert_close(active(pairs), kernel(X))
    tree = ScaleKernel(IntegratedWhiteNoiseKernel(order)) + RBFKernel()
    assert shape(tree(X)) == (4, 4)


@pytest.mark.parametrize("order", [0, -1, True, 1.5, "2"])
def test_invalid_order(order: object) -> None:
    with pytest.raises(ValueError, match="integer order"):
        IntegratedWhiteNoiseKernel(order)  # type: ignore[arg-type]


def test_invalid_input_dimension() -> None:
    with pytest.raises(ValueError, match="1D inputs"):
        IntegratedWhiteNoiseKernel()([[0.1, 0.2]])


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6, 9])
def test_official_kernel_values_and_known_diagonal_bug(order: int) -> None:
    BtKernel = official_kernel()
    import torch

    generator = torch.Generator().manual_seed(80 + order)
    X = torch.rand(2, 7, 1, generator=generator, dtype=torch.float64) * 2
    Z = torch.rand(2, 7, 1, generator=generator, dtype=torch.float64) * 2
    bt, rs = BtKernel(order), IntegratedWhiteNoiseKernel(order)
    torch.testing.assert_close(rs(X, Z), bt(X, Z).to_dense(), rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(rs(X, diag=True), bt(X, diag=True), rtol=1e-12, atol=1e-14)
    # Official b0dd01be6 ignores x2 in diag=True. Validate against its full
    # cross-covariance, which is the correct mathematical diagonal.
    expected = bt(X, Z).to_dense().diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(rs(X, Z, diag=True), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_single_task_gp_mll_posteriors_and_input_gradients(order: int) -> None:
    BtKernel = official_kernel()
    import torch
    from botorch.acquisition.analytic import PosteriorMean as BtMean
    from botorch.acquisition.analytic import PosteriorStandardDeviation as BtPSD
    from botorch.models import SingleTaskGP as BtGP
    from gpytorch.kernels import ScaleKernel as BtScale
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    X = torch.linspace(0.1, 2.0, 8, dtype=torch.float64).unsqueeze(-1)
    Y = torch.sin(2 * X)
    bt = BtGP(X, Y, covar_module=BtScale(BtKernel(order)), outcome_transform=None)
    rs = SingleTaskGP(
        X, Y, covar_module=ScaleKernel(IntegratedWhiteNoiseKernel(order)), outcome_transform=None
    )
    mb, mr = BtMLL(bt.likelihood, bt), ExactMarginalLogLikelihood(rs.likelihood, rs)
    vb = mb(bt(X), bt.train_targets)
    vb.backward()
    gb = torch.cat([p.grad.reshape(-1) for p in mb.parameters()])
    vr, gr = mr.value_and_grad()
    torch.testing.assert_close(
        torch.tensor(vr, dtype=torch.float64), vb.detach(), rtol=1e-9, atol=1e-10
    )
    torch.testing.assert_close(torch.tensor(gr, dtype=torch.float64), gb, rtol=1e-8, atol=1e-9)
    query = torch.tensor([[[0.15], [0.73]], [[1.1], [1.8]]], dtype=torch.float64)
    for noise in (False, True):
        pb, pr = (
            bt.posterior(query, observation_noise=noise),
            rs.posterior(query, observation_noise=noise),
        )
        torch.testing.assert_close(pr.mean, pb.mean.detach(), rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(
            pr.distribution.covariance_matrix,
            pb.distribution.covariance_matrix.detach(),
            rtol=1e-8,
            atol=1e-10,
        )
    for ab, ar in [(BtMean(bt), PosteriorMean(rs)), (BtPSD(bt), PosteriorStandardDeviation(rs))]:
        candidate = query.reshape(4, 1, 1).clone().requires_grad_(True)
        vb = ab(candidate)
        gb = torch.autograd.grad(vb.sum(), candidate)[0]
        vr, gr = ar.value_and_grad(candidate.detach())
        torch.testing.assert_close(vr, vb.detach(), rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(gr, gb, rtol=1e-8, atol=1e-9)
