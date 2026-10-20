"""Native EM empirical-prior parity against official source b0dd01be6."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from botorch_rs import _botorch_rs as _rs
from botorch_rs.kernels import RBFKernel, ScaleKernel
from botorch_rs.means import ConstantMean, ZeroMean
from botorch_rs.models.empirical_gps import EMPriorContainer, pretrain_em_prior
from botorch_rs.models.empirical_gps.utils import ExperimentDataset
from botorch_rs.priors import LogNormalPrior, NormalPrior
from helpers import as_kind, assert_close, check_kind


@pytest.fixture(scope="module")
def official() -> Any:
    pytest.importorskip("botorch")
    root = Path(os.environ.get("BOTORCH_RS_REFERENCE_ROOT", Path(__file__).resolve().parents[2] / ".reference/botorch"))
    folder = root / "botorch/models/empirical_gps"
    if not folder.is_dir():
        pytest.skip("Requires official empirical GP source via BOTORCH_RS_REFERENCE_ROOT")
    package = "botorch.models.empirical_gps"
    if package not in sys.modules:
        pkg = ModuleType(package)
        pkg.__path__ = [str(folder)]
        sys.modules[package] = pkg
    modules = {}
    for name in ("utils", "em_empirical_gp"):
        spec = importlib.util.spec_from_file_location(f"{package}.{name}", folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules)


def datasets(kind: str) -> list[ExperimentDataset]:
    grid = [0.0, 0.2, 0.45, 0.7, 1.0]
    return [
        ExperimentDataset(
            X=as_kind([[grid[i]] for i in indices], kind),
            Y=as_kind([[0.4 * j + math.sin((1.2 + j * 0.2) * grid[i]) - 0.15 * grid[i]] for i in indices], kind),
        )
        for j, indices in enumerate(([0, 1, 2], [0, 2, 3, 4], [1, 3, 4], [0, 1, 2, 3, 4]))
    ]


def modules() -> tuple[ConstantMean, ScaleKernel]:
    mean = ConstantMean().initialize(constant=0.35)
    kernel = ScaleKernel(RBFKernel())
    kernel.base_kernel.lengthscale = 0.42
    kernel.outputscale = 1.3
    return mean, kernel


@pytest.mark.parametrize("inducing", [False, True])
@pytest.mark.parametrize("prior,initial,shrinkage", [(False, "kernel", 0.0), (True, "kernel", 0.23), (True, "naive", 0.1)])
def test_pretrain_matches_official(official: Any, inducing: bool, prior: bool, initial: str, shrinkage: float) -> None:
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    source_data = datasets("torch")
    ref_data = [official.utils.ExperimentDataset(X=d.X, Y=d.Y) for d in source_data]
    mean, kernel = modules()
    ref_mean = gpytorch.means.ConstantMean().double().initialize(constant=torch.tensor(0.35, dtype=torch.double))
    ref_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    ref_kernel.base_kernel.lengthscale = torch.tensor(0.42, dtype=torch.double)
    ref_kernel.outputscale = torch.tensor(1.3, dtype=torch.double)
    locations = torch.tensor([[0.05], [0.4], [0.85]], dtype=torch.double) if inducing else None
    history: list[Any] = []
    ref_history: list[Any] = []
    options = dict(likelihood_noise=0.035, inducing_points=locations, num_em_iterations=5, use_mean_prior=prior, use_covar_prior=prior, em_convergence_tol=None, init_mode=initial, covariance_shrinkage=shrinkage)
    actual = pretrain_em_prior(source_data, mean, kernel, iteration_history=history, **options)
    expected = official.em_empirical_gp.pretrain_em_prior(ref_data, ref_mean, ref_kernel, iteration_history=ref_history, **options)
    for name in ("X_inducing", "mu_inducing", "Sigma_inducing", "L_kernel_inducing", "delta_mu", "likelihood_noise"):
        assert_close(getattr(actual, name), getattr(expected, name), rtol=3e-8, atol=3e-10)
    assert len(history) == len(ref_history) == 5
    for actual_state, expected_state in zip(history, ref_history, strict=True):
        for a, b in zip(actual_state, expected_state, strict=True):
            assert_close(a, b, rtol=3e-8, atol=3e-10)
    assert actual.use_inducing_points is inducing
    assert actual.iw_nu == (actual.X_inducing.shape[0] + 2 if prior else None)
    assert not actual.mu_inducing.requires_grad


def test_pretrain_roundtrip_and_array_kind(kind: str, tmp_path: Path) -> None:
    mean, kernel = modules()
    prior = pretrain_em_prior(datasets(kind), mean, kernel, likelihood_noise=0.03, num_em_iterations=3)
    for name in ("X_inducing", "mu_inducing", "Sigma_inducing", "L_kernel_inducing", "delta_mu", "likelihood_noise"):
        check_kind(getattr(prior, name), kind)
    path = tmp_path / "native_prior.pkl"
    prior.save(str(path))
    restored = EMPriorContainer.load(str(path))
    assert_close(restored.mu_inducing, prior.mu_inducing)
    assert_close(restored.Sigma_inducing, prior.Sigma_inducing)
    assert_close(restored.mean_module.constant, 0.35)
    assert restored.mean_module._model is None
    assert restored.covar_module._model is None
    assert_close(restored.covar_module(restored.X_inducing), kernel(prior.X_inducing))
    restored.covar_module.base_kernel.lengthscale = 0.8
    assert_close(kernel.base_kernel.lengthscale, 0.42)


def test_pretrain_early_stopping_and_validation() -> None:
    data = [ExperimentDataset(X=[[0.0], [1.0]], Y=[[1.0], [-1.0]])] * 2
    history: list[Any] = []
    prior = pretrain_em_prior(data, ZeroMean(), RBFKernel(), likelihood_noise=0.0, num_em_iterations=20, em_convergence_tol=1e-7, iteration_history=history)
    assert len(history) == 2
    assert_close(prior.mu_inducing, [1.0, -1.0])
    assert_close(prior.Sigma_inducing, [[1e-10, 0.0], [0.0, 1e-10]], atol=1e-13)
    for kwargs, pattern in (({"likelihood_noise": -0.1}, "noise"), ({"inducing_points": [[0.0, 0.5]]}, "dimension"), ({"use_covar_prior": True, "iw_nu": 0.5}, "iw_nu"), ({"covariance_shrinkage": 1.5}, "shrinkage")):
        with pytest.raises(ValueError, match=pattern):
            pretrain_em_prior(data, ZeroMean(), RBFKernel(), **kwargs)
    with pytest.raises(ValueError, match="length differs"):
        _rs.em_pretrain([([0.0], [2, 1])], [([1.0, 2.0], [2, 1])], ([0.0, 1.0], [2, 1]), RBFKernel()._node(), [0.0], 0.0, None, 0.1, 1, False, False, None, None, False, 0.0, False)


def test_prior_roundtrip_preserves_native_prior_precision(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    mean = ConstantMean(constant_prior=NormalPrior(0.0, 1.3)).initialize(constant=0.3)
    kernel = RBFKernel(lengthscale_prior=LogNormalPrior(torch.tensor(-0.4), torch.tensor(0.7)))
    prior = pretrain_em_prior(datasets("torch"), mean, kernel, num_em_iterations=2)
    path = tmp_path / "prior_precision.pkl"
    prior.save(str(path))
    loaded = EMPriorContainer.load(str(path))
    assert _rs.em_prior_state(loaded.covar_module.lengthscale_prior._rs) == _rs.em_prior_state(kernel.lengthscale_prior._rs)
    assert_close(loaded.mean_module.mean_prior.log_prob(0.8), mean.mean_prior.log_prob(0.8))
