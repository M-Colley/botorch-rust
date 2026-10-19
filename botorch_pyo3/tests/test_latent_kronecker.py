"""Native grid inference, missing-data likelihoods and official-source parity."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.means import ConstantMean
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import LatentKroneckerGP
from botorch_rs.models.transforms.input import Normalize, Warp
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from helpers import KINDS, as_kind, assert_close, check_kind, shape, to_list


def data(missing: bool = False) -> tuple[list, list, list]:
    X = [[0.1, 0.2], [0.3, 0.8], [0.6, 0.4], [0.9, 0.1]]
    T = [[0.0], [0.4], [1.0]]
    Y = [[math.sin(3 * x[0] + t[0]) + 2 * x[1] for t in T] for x in X]
    if missing:
        Y[0][1], Y[2][0] = math.nan, math.inf
    return X, T, Y


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("missing", [False, True])
def test_api_missing_grid_fit_and_copy(kind: str, missing: bool) -> None:
    X, T, Y = data(missing)
    model = LatentKroneckerGP(*(as_kind(v, kind) for v in (X, T, Y)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    nobs = 10 if missing else 12
    assert shape(model.train_targets) == (nobs,)
    assert sum(to_list(model.mask_valid)) == nobs
    assert model.num_outputs == 3 and model.batch_shape == ()
    assert model._rs.complete_grid is not missing
    before = float(to_list(mll()))
    result = fit_gpytorch_mll_scipy(mll, options={"maxiter": 8})
    assert math.isfinite(result.fval) and float(to_list(mll())) > before
    raw = model._rs.raw_parameters()
    fit_gpytorch_mll_torch(mll, step_limit=3, stopping_criterion=None)
    assert model._rs.raw_parameters() != raw
    query = as_kind([[0.4, 0.6], [0.7, 0.2]], kind)
    post = model.posterior(query)
    assert shape(post.mean) == shape(post.variance) == (2, 3)
    assert shape(post.distribution.covariance_matrix) == (6, 6)
    assert post.base_sample_shape == (12 + nobs + 6,)
    assert shape(post.rsample((2, 3))) == (2, 3, 2, 3)
    check_kind(post.mean, kind)
    for sampler_cls in (SobolQMCNormalSampler, IIDNormalSampler):
        sampler = sampler_cls((4,), seed=17)
        assert shape(sampler(post)) == (4, 2, 3)
        assert_close(sampler(post), sampler(post))
    cloned = copy.deepcopy(model)
    cloned.likelihood.noise = 0.2
    assert to_list(model.likelihood.noise)[0] != pytest.approx(0.2)
    # A posterior snapshots parameters instead of mutating under cached moments.
    old_mean = to_list(post.mean)
    model.likelihood.noise = 0.8
    assert_close(post.mean, old_mean)


@pytest.mark.parametrize("kind", KINDS)
def test_constant_mean_binding_transforms_and_query_broadcast(kind: str) -> None:
    X, T, Y = data(True)
    model = LatentKroneckerGP(
        *(as_kind(v, kind) for v in (X, T, Y)),
        mean_module_X=ConstantMean(),
        mean_module_T=ConstantMean(),
        input_transform=Normalize(d=2),
    )
    model.mean_module_X.constant = 0.7
    model.mean_module_T.constant = -0.3
    assert float(to_list(model.mean_module_X.constant)) == pytest.approx(0.7)
    assert float(to_list(model.mean_module_T.constant)) == pytest.approx(-0.3)
    q = [[0.4, 0.6], [0.7, 0.2]]
    qt = [[0.1], [0.8]]
    post = model.posterior(as_kind([q, q], kind), as_kind(qt, kind))
    assert shape(post.mean) == (2, 2, 2)
    samples = SobolQMCNormalSampler((4,), seed=11)(post)
    assert shape(samples) == (4, 2, 2, 2)
    assert_close(to_list(samples)[0][0], to_list(samples)[0][1])
    values, gradient = ExactMarginalLogLikelihood(model.likelihood, model).value_and_grad()
    assert math.isfinite(values) and all(math.isfinite(v) for v in gradient)


def test_explicit_invalid_and_unsupported_options() -> None:
    X, T, Y = data()
    with pytest.raises(UnsupportedError, match="unbatched"):
        LatentKroneckerGP([X, X], T, [Y, Y])
    with pytest.raises(BotorchTensorDimensionError):
        LatentKroneckerGP(X, [0.0, 1.0], Y)
    with pytest.raises(ValueError, match="finite observation"):
        LatentKroneckerGP(X, T, [[math.nan] * 3] * 4)
    with pytest.raises(UnsupportedError, match="learned input transforms"):
        LatentKroneckerGP(X, T, Y, input_transform=Warp(d=2, indices=[0, 1]))
    with pytest.raises(UnsupportedError, match="factor means"):
        LatentKroneckerGP(X, T, Y, mean_module_X=object())
    with pytest.raises(UnsupportedError, match="kernels"):
        LatentKroneckerGP(X, T, Y, covar_module_T=object())
    model = LatentKroneckerGP(X, T, Y)
    with pytest.raises(UnsupportedError, match="iterative"):
        model.use_iterative_methods()
    with pytest.raises(NotImplementedError, match="Observation noise"):
        model.posterior(X, observation_noise=True)
    with pytest.raises(NotImplementedError, match="Posterior transforms"):
        model.posterior(X, posterior_transform=object())
    with pytest.raises(NotImplementedError, match="Conditioning"):
        model.condition_on_observations(X, Y)


@pytest.mark.parametrize("kind", KINDS)
def test_construct_inputs_factorizes_observed_rows(kind: str) -> None:
    from botorch_rs.exceptions.warnings import InputDataWarning
    from botorch_rs.utils.datasets import SupervisedDataset

    dataset = SupervisedDataset(
        as_kind([[0.8, 1.0], [0.2, 0.0], [0.8, 0.0]], kind),
        as_kind([[3.0], [1.0], [2.0]], kind),
        feature_names=["feature", "step"],
        outcome_names=["value"],
        Yvar=as_kind([[0.1], [0.2], [0.3]], kind),
    )
    with pytest.warns(InputDataWarning, match="Ignoring Yvar"):
        values = LatentKroneckerGP.construct_inputs(dataset)
    assert_close(values["train_X"], [[0.2], [0.8]])
    assert_close(values["train_T"], [[0.0], [1.0]])
    targets = to_list(values["train_Y"])
    assert targets[0][0] == 1.0 and math.isnan(targets[0][1])
    assert targets[1] == [2.0, 3.0]
    assert LatentKroneckerGP(**values)._rs.num_observed == 3


REFERENCE_SCRIPT = r"""
import json, torch
from botorch.models.latent_kronecker_gp import LatentKroneckerGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import max_cholesky_size, fast_pred_var
from linear_operator.settings import _fast_solves
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
results=[]
for seed in range(8):
    generator=torch.Generator().manual_seed(920+seed)
    X=torch.rand(4+seed%2,2,generator=generator)
    T=torch.tensor([[0.],[0.4],[1.]])
    Y=3+2*torch.randn(X.shape[0],3,generator=generator)
    if seed%2: Y[0,1]=float('nan'); Y[2,0]=float('inf')
    Q=torch.rand(2,2,generator=generator)
    QT=T if seed%3 else torch.tensor([[0.2],[0.8]])
    standardized=seed>=4
    model=LatentKroneckerGP(X,T,Y,mean_module_X=ConstantMean(),mean_module_T=ConstantMean(),outcome_transform=Standardize(1) if standardized else None)
    mll=ExactMarginalLogLikelihood(model.likelihood,model)
    raw=[]
    with torch.no_grad():
        for name,p in mll.named_parameters():
            p.copy_(torch.randn(p.shape,generator=generator)*0.3)
            raw.extend(p.flatten().tolist())
    with _fast_solves(False), max_cholesky_size(10000), fast_pred_var(False):
        value=mll(model(*model.train_inputs),model.train_targets)
        value.backward()
        gradient=torch.cat([p.grad.flatten() for p in mll.parameters()])
        model.eval()
        posterior=model.posterior(Q,QT)
        mean=posterior.mean.reshape(Q.shape[0],QT.shape[0])
        covariance=posterior.distribution.covariance_matrix
        if standardized:
            mean=mean*model.outcome_transform.stdvs.item()+model.outcome_transform.means.item()
            covariance=covariance*model.outcome_transform.stdvs.item()**2
    results.append(dict(X=X.tolist(),T=T.tolist(),Y=Y.nan_to_num(nan=0.,posinf=0.).tolist(),mask=Y.isfinite().tolist(),Q=Q.tolist(),QT=QT.tolist(),standardized=standardized,raw=raw,value=value.item(),gradient=gradient.tolist(),mean=mean.tolist(),covariance=covariance.tolist()))
print(json.dumps(results))
"""


@pytest.fixture(scope="module")
def official_cases() -> list[dict]:
    reference = Path(
        os.environ.get(
            "BOTORCH_RS_REFERENCE_ROOT",
            Path(__file__).resolve().parents[2] / ".reference" / "botorch",
        )
    )
    if not (reference / "botorch/models/latent_kronecker_gp.py").is_file():
        pytest.skip("Requires current official source via BOTORCH_RS_REFERENCE_ROOT (b0dd01be6)")
    environment = dict(os.environ, PYTHONPATH=str(reference))
    result = subprocess.run(
        [sys.executable, "-c", REFERENCE_SCRIPT],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("seed", range(8))
def test_official_likelihood_gradient_and_joint_posterior(
    official_cases: list[dict], seed: int
) -> None:
    case = official_cases[seed]
    Y = [
        [y if valid else math.nan for y, valid in zip(row, mask, strict=True)]
        for row, mask in zip(case["Y"], case["mask"], strict=True)
    ]
    options = {} if case["standardized"] else {"outcome_transform": None}
    model = LatentKroneckerGP(
        case["X"],
        case["T"],
        Y,
        mean_module_X=ConstantMean(),
        mean_module_T=ConstantMean(),
        **options,
    )
    model._rs.set_raw_parameters(case["raw"])
    value, gradient = ExactMarginalLogLikelihood(model.likelihood, model).value_and_grad()
    assert value == pytest.approx(case["value"], rel=2e-8, abs=2e-8)
    assert_close(gradient, case["gradient"], atol=3e-8, rtol=3e-8)
    posterior = model.posterior(case["Q"], case["QT"])
    assert_close(posterior.mean, case["mean"], atol=3e-8, rtol=3e-8)
    assert_close(posterior.distribution.covariance_matrix, case["covariance"], atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("missing", [False, True])
def test_matheron_samples_match_exact_posterior_and_preserve_new_x_variance(missing: bool) -> None:
    np = pytest.importorskip("numpy")
    X, T, Y = data(missing)
    model = LatentKroneckerGP(np.array(X), np.array(T), np.array(Y))
    posterior = model.posterior(np.array([[0.5, 0.5], [8.0, 8.0]]))
    base = np.random.default_rng(874).standard_normal((16384, *posterior.base_sample_shape))
    draws = posterior.rsample_from_base_samples((16384,), base).reshape(16384, -1)
    np.testing.assert_allclose(draws.mean(0), posterior.mean.reshape(-1), atol=0.025, rtol=0.0)
    np.testing.assert_allclose(
        np.cov(draws.T), posterior.distribution.covariance_matrix, atol=0.025, rtol=0.06
    )
    # Official b0dd01be6's product-of-residuals sampler collapses the remote-X
    # variance when query T equals train_T. This is a covariance regression.
    assert draws[:, -1].var() > 0.1
