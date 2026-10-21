"""SAC shared parameters, native model operations, and official reference parity.

Audited against official BoTorch b0dd01be6. Its shared base lengthscale
prior is registered through M+1 aliases; MLL comparisons remove the M
duplicate terms, leaving one prior on the one shared parameter.
"""

from __future__ import annotations

import copy
import math

import pytest

from botorch_rs.acquisition.analytic import PosteriorMean, PosteriorStandardDeviation
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SACGP
from botorch_rs.models.kernels import SACKernel
from botorch_rs.utils.datasets import SupervisedDataset
from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape

DECOMPOSITION = {"left": [0, 2], "right": [3, 1]}
X = [[0.1, 0.8, 0.3, 0.4], [0.4, 0.2, 0.7, 0.9], [0.8, 0.5, 0.1, 0.6]]
Y = [[0.2], [0.8], [-0.4]]


@pytest.mark.parametrize("kind", KINDS)
def test_shared_kernel_parameters_and_closed_form(kind: str) -> None:
    kernel = SACKernel(DECOMPOSITION, ())
    kernel.base_kernel.lengthscale = as_kind([[0.3, 0.7]], kind)
    kernel.initialize(**{"kernel_dict.left.outputscale": 0.8, "kernel_dict.right.outputscale": 1.3})
    assert kernel.kernel_dict.left.base_kernel is kernel.base_kernel
    assert kernel.kernel_dict.right.base_kernel is kernel.base_kernel
    names = [name for name, _ in kernel.named_parameters()]
    assert names == [
        "base_kernel.raw_lengthscale",
        "kernel_dict.left.raw_outputscale",
        "kernel_dict.right.raw_outputscale",
    ]
    result = kernel(as_kind(X, kind))
    check_kind(result, kind)
    expected = [
        [
            sum(
                scale
                * math.exp(
                    -0.5
                    * sum(((x[d] - z[d]) / ls) ** 2 for d, ls in zip(dims, [0.3, 0.7], strict=True))
                )
                for dims, scale in [(DECOMPOSITION["left"], 0.8), (DECOMPOSITION["right"], 1.3)]
            )
            for z in X
        ]
        for x in X
    ]
    assert_close(result, expected)
    assert_close(kernel(as_kind(X, kind), diag=True), [2.1] * 3)
    clone = copy.deepcopy(kernel)
    assert clone.kernel_dict.left.base_kernel is clone.base_kernel
    assert clone.base_kernel is not kernel.base_kernel
    clone.kernel_dict.right.base_kernel.lengthscale = [[0.5, 0.6]]
    assert_close(clone.base_kernel.lengthscale, [[0.5, 0.6]])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fixed_noise", [False, True])
def test_construct_inputs_native_fit_and_cloning(kind: str, fixed_noise: bool) -> None:
    x, y = as_kind(X, kind), as_kind(Y, kind)
    yvar = as_kind([[0.03]] * 3, kind) if fixed_noise else None
    dataset = SupervisedDataset(
        x,
        y,
        Yvar=yvar,
        feature_names=["a", "b", "c", "d"],
        outcome_names=["y"],
    )
    inputs = SACGP.construct_inputs(dataset, DECOMPOSITION)
    assert "train_Yvar" in inputs
    model = SACGP(**inputs)
    assert model.covar_module.kernel_dict.left.base_kernel is model.covar_module.base_kernel
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    initial = float(mll())
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 25}})
    assert float(mll()) > initial + 0.1
    posterior = model.posterior(x)
    check_kind(posterior.mean, kind)
    assert shape(posterior.mean) == (3, 1)
    clone = copy.deepcopy(model)
    assert clone.covar_module.kernel_dict.right.base_kernel is clone.covar_module.base_kernel
    assert_close(clone.posterior(x).mean, posterior.mean)
    # Model parameter aliases all address the canonical shared vector.
    clone.covar_module.kernel_dict.left.base_kernel.lengthscale = [[0.4, 0.6]]
    assert_close(clone.covar_module.base_kernel.lengthscale, [[0.4, 0.6]])
    conditioned = model.condition_on_observations(
        as_kind(X[:1], kind),
        as_kind(Y[:1], kind),
        **({"noise": as_kind([[0.03]], kind)} if fixed_noise else {}),
    )
    assert shape(conditioned.posterior(x).mean) == (3, 1)


@pytest.mark.parametrize(
    "decomposition",
    [{}, {"a": []}, {"a": [0], "b": [1, 2]}, {"a.b": [0]}, {"a": [-1]}, {"a": [True]}],
)
def test_invalid_context_decomposition(decomposition: dict) -> None:
    with pytest.raises(ValueError):
        SACKernel(decomposition, ())


def test_context_index_bounds() -> None:
    with pytest.raises(RuntimeError, match="out of range"):
        SACKernel({"a": [5]}, ())(X)


@pytest.mark.parametrize("batch,m", [((), 1), ((2,), 1), ((), 2)])
@pytest.mark.parametrize("fixed_noise", [False, True])
def test_official_kernel_corrected_mll_and_posterior_parity(
    batch: tuple, m: int, fixed_noise: bool
) -> None:
    importorskip_botorch()
    import torch

    pytest.importorskip("botorch")
    from botorch.models.contextual import SACGP as BtGP

    pytest.importorskip("gpytorch")
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    generator = torch.Generator().manual_seed(172)
    x = torch.rand(*batch, 7, 4, generator=generator, dtype=torch.float64)
    y = torch.randn(*batch, 7, m, generator=generator, dtype=torch.float64)
    noise = torch.full_like(y, 0.02) if fixed_noise else None
    bt, rs = BtGP(x, y, noise, DECOMPOSITION), SACGP(x, y, noise, DECOMPOSITION)
    # Explicit parameters avoid the upstream default dtype's float32 rounding.
    with torch.no_grad():
        for name, parameter in bt.named_parameters():
            values = torch.full_like(parameter, 0.23)
            if name.endswith("raw_lengthscale"):
                values[..., 1] = 0.57
            parameter.copy_(values)
            rs._set_raw_param(name, values)
    kb, kr = bt.covar_module, rs.covar_module
    torch.testing.assert_close(kr(x), kb(x).to_dense().detach(), rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(kr(x, diag=True), kb(x, diag=True).detach(), rtol=1e-11, atol=1e-12)
    mb, mr = BtMLL(bt.likelihood, bt), ExactMarginalLogLikelihood(rs.likelihood, rs)
    official = mb(bt(*bt.train_inputs), bt.train_targets)
    aliases = [name for name, _, _, _, _ in kb.named_priors() if name.endswith("lengthscale_prior")]
    assert len(aliases) == len(DECOMPOSITION) + 1
    shared_prior = kb.base_kernel.lengthscale_prior.log_prob(kb.base_kernel.lengthscale).sum(
        dim=(-1, -2)
    )
    corrected = official - len(DECOMPOSITION) * shared_prior / x.shape[-2]
    vb, gb = corrected.sum(), None
    vb.backward()
    gb = torch.cat([p.grad.reshape(-1) for p in mb.parameters()])
    vr, gr = mr.value_and_grad()
    torch.testing.assert_close(
        torch.as_tensor(vr, dtype=torch.float64), corrected.detach(), rtol=1e-8, atol=2e-8
    )
    torch.testing.assert_close(torch.tensor(gr, dtype=torch.float64), gb, rtol=2e-7, atol=2e-8)
    # The exact default MLL discrepancy equals the duplicated prior terms.
    torch.testing.assert_close(
        official.sum().detach() - torch.as_tensor(vr, dtype=torch.float64).sum(),
        (len(DECOMPOSITION) * shared_prior / x.shape[-2]).sum().detach(),
        rtol=2e-7,
        atol=2e-8,
    )
    query = torch.rand(3, 4, generator=generator, dtype=torch.float64)
    pb, pr = bt.posterior(query), rs.posterior(query)
    torch.testing.assert_close(pr.mean, pb.mean.detach(), rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix.detach(),
        rtol=1e-9,
        atol=1e-10,
    )
    if not batch and m == 1:
        from botorch.acquisition.analytic import PosteriorMean as BtMean
        from botorch.acquisition.analytic import PosteriorStandardDeviation as BtSD

        for ab, ar in [(BtMean(bt), PosteriorMean(rs)), (BtSD(bt), PosteriorStandardDeviation(rs))]:
            candidate = query.unsqueeze(-2).clone().requires_grad_(True)
            vb = ab(candidate)
            gb = torch.autograd.grad(vb.sum(), candidate)[0]
            vr, gr = ar.value_and_grad(candidate.detach())
            torch.testing.assert_close(vr, vb.detach(), rtol=1e-9, atol=1e-10)
            torch.testing.assert_close(gr, gb, rtol=1e-8, atol=1e-9)
