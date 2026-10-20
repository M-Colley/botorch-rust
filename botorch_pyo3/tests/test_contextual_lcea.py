"""Native LCEA covariance, exact inference, derivatives and reference parity."""

from __future__ import annotations

import copy
import os
import subprocess
import sys

import pytest

from botorch_rs.acquisition.analytic import PosteriorMean, PosteriorStandardDeviation
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import LCEAGP
from botorch_rs.models.kernels import LCEAKernel
from botorch_rs.models.kernels.contextual_lcea import get_order, get_permutation, is_contiguous
from botorch_rs.utils.datasets import SupervisedDataset
from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape

DECOMP = {"a": [0, 1], "b": [2, 3]}
X = [[0.1, 0.7, 0.3, 0.4], [0.4, 0.2, 0.7, 0.9], [0.8, 0.5, 0.1, 0.6]]
Y = [[0.2], [0.8], [-0.4]]


def test_context_order_helpers() -> None:
    assert get_order([4, 5]) == [0, 1]
    assert is_contiguous([2, 1]) and not is_contiguous([1, 3])
    assert get_permutation(DECOMP) is None
    assert get_permutation({"a": [0, 2], "b": [3, 1]}) == [0, 2, 1, 3]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("fixed_noise", [False, True])
def test_array_api_and_native_fit(kind: str, learned: bool, fixed_noise: bool) -> None:
    model = LCEAGP(
        as_kind(X, kind),
        as_kind(Y, kind),
        as_kind([[0.03]] * 3, kind) if fixed_noise else None,
        DECOMP,
        train_embedding=learned,
        embs_feature_dict={"a": [0.1], "b": [0.6]},
    )
    kernel = model.covar_module
    if learned:
        kernel.emb_layers[0].weight = as_kind([[-0.2], [0.4]], kind)
    assert kernel.task_covar_module is not kernel.base_kernel
    matrix = kernel(as_kind(X, kind))
    check_kind(matrix, kind)
    assert shape(matrix) == (3, 3)
    assert shape(kernel._eval_context_covar()) == (2, 2)
    assert shape(kernel._task_embeddings()) == (2, 2 if learned else 1)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = float(mll())
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 20}})
    assert float(mll()) > before + 0.1
    posterior = model.posterior(as_kind(X, kind))
    check_kind(posterior.mean, kind)
    assert shape(posterior.mean) == (3, 1)
    clone = copy.deepcopy(model)
    assert_close(clone.posterior(as_kind(X, kind)).mean, posterior.mean)
    if learned:
        kernel.emb_layers[0].weight = [[2.0], [-3.0]]
        kernel(as_kind(X, kind))
        assert_close(kernel.emb_layers[0].weight, [[2.0 / (2.0 + 1e-7)], [-3.0 / (3.0 + 1e-7)]])


@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("known_weights", [False, True])
@pytest.mark.parametrize("fixed_noise", [False, True])
def test_official_kernel_mll_parameter_and_input_gradients(
    learned: bool, known_weights: bool, fixed_noise: bool
) -> None:
    importorskip_botorch()
    import torch
    from botorch.acquisition.analytic import PosteriorMean as BtMean
    from botorch.acquisition.analytic import PosteriorStandardDeviation as BtSD
    from botorch.models.contextual import LCEAGP as BtGP
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    generator = torch.Generator().manual_seed(367)
    x = torch.rand(8, 4, generator=generator, dtype=torch.float64)
    y = torch.sin(x[:, 0:1] * 4) + 0.2 * x[:, 2:3]
    kwargs = dict(
        train_embedding=learned,
        embs_feature_dict={"a": [0.25], "b": [0.75]},
        context_weight_dict={"a": 0.25, "b": 0.75} if known_weights else None,
    )
    if learned:
        kwargs.update(cat_feature_dict={"a": [0, 0], "b": [1, 0]}, embs_dim_list=[2, 1])
    noise = torch.full_like(y, 0.03) if fixed_noise else None
    bt, rs = BtGP(x, y, noise, DECOMP, **kwargs), LCEAGP(x, y, noise, DECOMP, **kwargs)
    with torch.no_grad():
        for name, parameter in bt.named_parameters():
            values = torch.full_like(parameter, 0.37)
            if name.endswith("weight"):
                values[0] = -0.2
            parameter.copy_(values)
            rs._set_raw_param(name, values)
    kb, kr = bt.covar_module, rs.covar_module
    query = torch.rand(3, 4, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(
        kr(x, query), kb(x, query).to_dense().detach(), rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(kr(x, diag=True), kb(x, diag=True).detach(), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        kr._eval_context_covar(), kb._eval_context_covar().detach(), rtol=1e-11, atol=1e-12
    )
    torch.testing.assert_close(
        kr._eval_base_covar_perm(x, query),
        kb._eval_base_covar_perm(x, query).detach(),
        rtol=1e-11,
        atol=1e-12,
    )
    mb, mr = BtMLL(bt.likelihood, bt), ExactMarginalLogLikelihood(rs.likelihood, rs)
    vb = mb(bt(*bt.train_inputs), bt.train_targets)
    vb.backward()
    gb = torch.cat([p.grad.reshape(-1) for p in mb.parameters()])
    vr, gr = mr.value_and_grad()
    torch.testing.assert_close(
        torch.tensor(vr, dtype=torch.float64), vb.detach(), rtol=1e-8, atol=2e-8
    )
    torch.testing.assert_close(torch.tensor(gr, dtype=torch.float64), gb, rtol=2e-7, atol=2e-8)
    pb, pr = bt.posterior(query), rs.posterior(query)
    torch.testing.assert_close(pr.mean, pb.mean.detach(), rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix.detach(),
        rtol=1e-9,
        atol=1e-10,
    )
    for ab, ar in [(BtMean(bt), PosteriorMean(rs)), (BtSD(bt), PosteriorStandardDeviation(rs))]:
        candidate = query.unsqueeze(-2).clone().requires_grad_(True)
        vb = ab(candidate)
        gb = torch.autograd.grad(vb.sum(), candidate)[0]
        vr, gr = ar.value_and_grad(candidate.detach())
        torch.testing.assert_close(vr, vb.detach(), rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(gr, gb, rtol=1e-8, atol=1e-9)


def test_context_order_alignment_is_preserved() -> None:
    kernel = LCEAKernel(
        {"a": [0, 2], "b": [3, 1]},
        (),
        train_embedding=False,
        embs_feature_dict={"a": [0.25], "b": [0.75]},
    )
    canonical = LCEAKernel(
        DECOMP, (), train_embedding=False, embs_feature_dict={"a": [0.25], "b": [0.75]}
    )
    kernel.base_kernel.lengthscale = canonical.base_kernel.lengthscale = [[0.3, 0.7]]
    permuted = [[row[i] for i in [0, 2, 3, 1]] for row in X]
    assert_close(kernel(X), canonical(permuted))


def test_multiple_categorical_and_fixed_features() -> None:
    kernel = LCEAKernel(
        DECOMP,
        (),
        cat_feature_dict={"a": [10, 4], "b": [20, 4]},
        embs_dim_list=[2, 1],
        embs_feature_dict={"a": [0.2], "b": [0.8]},
    )
    assert shape(kernel._task_embeddings()) == (2, 4)
    assert shape(kernel(X)) == (3, 3)


def test_construct_inputs_without_observed_noise() -> None:
    dataset = SupervisedDataset(X, Y, feature_names=["a", "b", "c", "d"], outcome_names=["y"])
    inputs = LCEAGP.construct_inputs(
        dataset, DECOMP, train_embedding=False, embs_feature_dict={"a": [0.25], "b": [0.75]}
    )
    assert inputs["train_Yvar"] is None
    assert shape(LCEAGP(**inputs).posterior(X).mean) == (3, 1)


def test_fixed_feature_batches_and_explicit_learned_batch_rejection() -> None:
    kwargs = dict(train_embedding=False, embs_feature_dict={"a": [0.25], "b": [0.75]})
    batch = LCEAKernel(DECOMP, (2,), **kwargs)
    batch.outputscale_list = [[0.5, 0.6], [0.8, 1.2]]
    from helpers import to_list

    actual = to_list(batch(X))
    for i, amplitudes in enumerate([[0.5, 0.6], [0.8, 1.2]]):
        single = LCEAKernel(DECOMP, (), **kwargs)
        single.outputscale_list = amplitudes
        assert_close(actual[i], single(X))
    assert shape(batch._eval_context_covar()) == (2, 2, 2)
    with pytest.raises(UnsupportedError, match="Batched learned"):
        LCEAKernel(DECOMP, (2,))


def test_fixed_feature_list_model_remains_torch_free() -> None:
    code = "from botorch_rs.models import LCEAGP; import sys; m=LCEAGP([[0.1,0.2],[0.4,0.6]],[[0.2],[0.8]],None,{'a':[0],'b':[1]},train_embedding=False,embs_feature_dict={'a':[0.25],'b':[0.75]}); assert len(m.posterior([[0.2,0.3]]).mean)==1; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
