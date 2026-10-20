"""Native EM empirical-prior parity against official source b0dd01be6."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import GreaterThan
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import RBFKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean, ZeroMean
from botorch_rs.models.empirical_gps import (
    EMEmpiricalGaussianProcess,
    EMEmpiricalMarginalLogLikelihood,
    EMPriorContainer,
    build_shared_gp_model_list,
    pretrain_em_prior,
)
from botorch_rs.models.empirical_gps.utils import ExperimentDataset
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch_rs.priors import LogNormalPrior, NormalPrior
from helpers import as_kind, assert_close, check_kind


@pytest.fixture(scope="module")
def official() -> Any:
    pytest.importorskip("botorch")
    root = Path(
        os.environ.get(
            "BOTORCH_RS_REFERENCE_ROOT", Path(__file__).resolve().parents[2] / ".reference/botorch"
        )
    )
    folder = root / "botorch/models/empirical_gps"
    if not folder.is_dir():
        pytest.skip("Requires official empirical GP source via BOTORCH_RS_REFERENCE_ROOT")
    package = "botorch.models.empirical_gps"
    if package not in sys.modules:
        pkg = ModuleType(package)
        pkg.__path__ = [str(folder)]
        sys.modules[package] = pkg
    modules = {}
    for name in ("utils", "em_empirical_gp", "em_coordinate_ascent"):
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
            Y=as_kind(
                [[0.4 * j + math.sin((1.2 + j * 0.2) * grid[i]) - 0.15 * grid[i]] for i in indices],
                kind,
            ),
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
@pytest.mark.parametrize(
    "prior,initial,shrinkage",
    [(False, "kernel", 0.0), (True, "kernel", 0.23), (True, "naive", 0.1)],
)
def test_pretrain_matches_official(
    official: Any, inducing: bool, prior: bool, initial: str, shrinkage: float
) -> None:
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    source_data = datasets("torch")
    ref_data = [official.utils.ExperimentDataset(X=d.X, Y=d.Y) for d in source_data]
    mean, kernel = modules()
    ref_mean = (
        gpytorch.means.ConstantMean()
        .double()
        .initialize(constant=torch.tensor(0.35, dtype=torch.double))
    )
    ref_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    ref_kernel.base_kernel.lengthscale = torch.tensor(0.42, dtype=torch.double)
    ref_kernel.outputscale = torch.tensor(1.3, dtype=torch.double)
    locations = torch.tensor([[0.05], [0.4], [0.85]], dtype=torch.double) if inducing else None
    history: list[Any] = []
    ref_history: list[Any] = []
    options = dict(
        likelihood_noise=0.035,
        inducing_points=locations,
        num_em_iterations=5,
        use_mean_prior=prior,
        use_covar_prior=prior,
        em_convergence_tol=None,
        init_mode=initial,
        covariance_shrinkage=shrinkage,
    )
    actual = pretrain_em_prior(source_data, mean, kernel, iteration_history=history, **options)
    expected = official.em_empirical_gp.pretrain_em_prior(
        ref_data, ref_mean, ref_kernel, iteration_history=ref_history, **options
    )
    for name in (
        "X_inducing",
        "mu_inducing",
        "Sigma_inducing",
        "L_kernel_inducing",
        "delta_mu",
        "likelihood_noise",
    ):
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
    prior = pretrain_em_prior(
        datasets(kind), mean, kernel, likelihood_noise=0.03, num_em_iterations=3
    )
    for name in (
        "X_inducing",
        "mu_inducing",
        "Sigma_inducing",
        "L_kernel_inducing",
        "delta_mu",
        "likelihood_noise",
    ):
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
    prior = pretrain_em_prior(
        data,
        ZeroMean(),
        RBFKernel(),
        likelihood_noise=0.0,
        num_em_iterations=20,
        em_convergence_tol=1e-7,
        iteration_history=history,
    )
    assert len(history) == 2
    assert_close(prior.mu_inducing, [1.0, -1.0])
    assert_close(prior.Sigma_inducing, [[1e-10, 0.0], [0.0, 1e-10]], atol=1e-13)
    for kwargs, pattern in (
        ({"likelihood_noise": -0.1}, "noise"),
        ({"inducing_points": [[0.0, 0.5]]}, "dimension"),
        ({"use_covar_prior": True, "iw_nu": 0.5}, "iw_nu"),
        ({"covariance_shrinkage": 1.5}, "shrinkage"),
    ):
        with pytest.raises(ValueError, match=pattern):
            pretrain_em_prior(data, ZeroMean(), RBFKernel(), **kwargs)
    with pytest.raises(ValueError, match="length differs"):
        _rs.em_pretrain(
            [([0.0], [2, 1])],
            [([1.0, 2.0], [2, 1])],
            ([0.0, 1.0], [2, 1]),
            RBFKernel()._node(),
            [0.0],
            0.0,
            None,
            0.1,
            1,
            False,
            False,
            None,
            None,
            False,
            0.0,
            False,
        )


def test_prior_roundtrip_preserves_native_prior_precision(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    mean = ConstantMean(constant_prior=NormalPrior(0.0, 1.3)).initialize(constant=0.3)
    kernel = RBFKernel(lengthscale_prior=LogNormalPrior(torch.tensor(-0.4), torch.tensor(0.7)))
    prior = pretrain_em_prior(datasets("torch"), mean, kernel, num_em_iterations=2)
    path = tmp_path / "prior_precision.pkl"
    prior.save(str(path))
    loaded = EMPriorContainer.load(str(path))
    assert _rs.em_prior_state(loaded.covar_module.lengthscale_prior._rs) == _rs.em_prior_state(
        kernel.lengthscale_prior._rs
    )
    assert_close(loaded.mean_module.mean_prior.log_prob(0.8), mean.mean_prior.log_prob(0.8))


def model_pair(official: Any, inducing: bool, base: bool) -> tuple[Any, Any]:
    import gpytorch
    import torch

    mean, kernel = modules()
    points = torch.tensor([[0.05], [0.4], [0.85]], dtype=torch.double) if inducing else None
    native_prior = pretrain_em_prior(
        datasets("torch"),
        mean,
        kernel,
        likelihood_noise=0.035,
        inducing_points=points,
        num_em_iterations=4,
    )
    ref_mean = (
        gpytorch.means.ConstantMean()
        .double()
        .initialize(constant=torch.tensor(0.35, dtype=torch.double))
    )
    ref_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    ref_kernel.base_kernel.lengthscale = torch.tensor(0.42, dtype=torch.double)
    ref_kernel.outputscale = torch.tensor(1.3, dtype=torch.double)
    ref_prior = official.em_empirical_gp.EMPriorContainer(
        **{
            **native_prior.__dict__,
            "mean_module": ref_mean,
            "covar_module": ref_kernel,
            "datasets": [
                official.utils.ExperimentDataset(X=d.X, Y=d.Y) for d in native_prior.datasets
            ],
        }
    )
    X = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)
    Y = torch.tensor([[0.7], [0.5], [0.8]], dtype=torch.double)
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4, initial_value=0.035))
    ref_likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    ref_likelihood.noise = torch.tensor(0.035, dtype=torch.double)
    native_base = ScaleKernel(RBFKernel()) if base else None
    ref_base = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double() if base else None
    if base:
        native_base.outputscale = 0.4
        native_base.base_kernel.lengthscale = 0.25
        ref_base.outputscale = torch.tensor(0.4, dtype=torch.double)
        ref_base.base_kernel.lengthscale = torch.tensor(0.25, dtype=torch.double)
    actual = EMEmpiricalGaussianProcess.from_pretrained(
        native_prior,
        X,
        Y,
        likelihood=likelihood,
        freeze_pretrained=False,
        base_covar_module=native_base,
    )
    actual.likelihood.noise = torch.tensor(0.035, dtype=torch.double)
    expected = official.em_empirical_gp.EMEmpiricalGaussianProcess.from_pretrained(
        ref_prior,
        X,
        Y,
        likelihood=ref_likelihood,
        freeze_pretrained=False,
        base_covar_module=ref_base,
    )
    return actual, expected


@pytest.mark.parametrize("inducing", [False, True])
@pytest.mark.parametrize("base", [False, True])
def test_em_model_posterior_and_historical_mll_parity(
    official: Any, inducing: bool, base: bool
) -> None:
    torch = pytest.importorskip("torch")
    actual, expected = model_pair(official, inducing, base)
    query = torch.tensor([[0.15], [0.6], [0.82]], dtype=torch.double)
    for observation_noise in (False, True):
        a = actual.posterior(query, observation_noise=observation_noise)
        b = expected.posterior(query, observation_noise=observation_noise)
        assert_close(a.mean, b.mean, rtol=2e-8, atol=2e-10)
        assert_close(
            a.distribution.covariance_matrix,
            b.distribution.covariance_matrix,
            rtol=2e-8,
            atol=2e-10,
        )
    a = actual.forward(query)
    b = expected.forward(query)
    assert_close(a.mean, b.mean, rtol=2e-8, atol=2e-10)
    assert_close(a.covariance_matrix, b.covariance_matrix, rtol=2e-8, atol=2e-10)
    actual_mll = EMEmpiricalMarginalLogLikelihood(actual.likelihood, actual)
    expected_mll = official.em_empirical_gp.EMEmpiricalMarginalLogLikelihood(
        expected.likelihood, expected
    )
    ref_value = expected_mll(None, None)
    value, gradient = actual_mll.value_and_grad()
    assert_close(value, ref_value, rtol=2e-8, atol=2e-10)
    ref_value.backward()
    reference = dict(expected.named_parameters())
    offset = 0
    for name, shape in actual._rs.parameter_shapes():
        n = math.prod(shape)
        ref_grad = reference[name].grad
        if ref_grad is None:
            assert_close(gradient[offset : offset + n], [0.0] * n, atol=1e-10)
        else:
            assert_close(gradient[offset : offset + n], ref_grad.reshape(-1), rtol=5e-7, atol=3e-9)
        offset += n
    # The same native Model implementation serves batched posterior requests.
    batched = query.unsqueeze(0).expand(2, -1, -1)
    assert_close(actual.posterior(batched).mean[0], expected.posterior(query).mean)


def test_em_model_fit_freeze_array_kinds_and_snapshot(kind: str) -> None:
    mean, kernel = modules()
    prior = pretrain_em_prior(
        datasets(kind),
        mean,
        kernel,
        likelihood_noise=0.035,
        inducing_points=as_kind([[0.05], [0.4], [0.85]], kind),
        num_em_iterations=4,
    )
    X, Y = as_kind([[0.1], [0.5], [0.9]], kind), as_kind([[0.7], [0.5], [0.8]], kind)
    model = EMEmpiricalGaussianProcess.from_pretrained(
        prior,
        X,
        Y,
        likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-4, initial_value=0.035)),
    )
    mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    initial = float(mll())
    kernel_raw = model.initial_covar_module._raw_flat()
    constant = copy.deepcopy(model.initial_mean_module.constant)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 12}}, max_attempts=1)
    assert float(mll()) >= initial - 1e-10
    assert model.initial_covar_module._raw_flat() == kernel_raw
    assert_close(model.initial_mean_module.constant, constant)
    query = as_kind([[0.15], [0.62]], kind)
    posterior = model.posterior(query)
    check_kind(posterior.mean, kind)
    old_mean, old_cov = (
        copy.deepcopy(posterior.mean),
        copy.deepcopy(posterior.distribution.covariance_matrix),
    )
    model.likelihood.noise = 0.12
    assert_close(posterior.mean, old_mean)
    assert_close(posterior.distribution.covariance_matrix, old_cov)
    model._rs.freeze_pretrained(False)
    before = model._rs.raw_parameters()
    fit_gpytorch_mll_scipy(mll, options={"maxiter": 3})
    fit_gpytorch_mll_torch(mll, step_limit=3, stopping_criterion=None)
    assert before != model._rs.raw_parameters()


def test_em_model_query_autograd_and_validation(official: Any) -> None:
    torch = pytest.importorskip("torch")
    actual, expected = model_pair(official, True, True)
    query = torch.tensor([[0.15], [0.6]], dtype=torch.double, requires_grad=True)
    ref_query = query.detach().clone().requires_grad_()
    a, b = actual.posterior(query), expected.posterior(ref_query)
    loss = a.mean.sum() + 0.7 * a.distribution.covariance_matrix.sum()
    ref_loss = b.mean.sum() + 0.7 * b.distribution.covariance_matrix.sum()
    assert_close(
        torch.autograd.grad(loss, query)[0],
        torch.autograd.grad(ref_loss, ref_query)[0],
        rtol=5e-7,
        atol=3e-9,
    )
    prior, ref_prior = actual.forward(query), expected.forward(ref_query)
    loss = prior.mean.sum() + 0.4 * prior.covariance_matrix.sum()
    ref_loss = ref_prior.mean.sum() + 0.4 * ref_prior.covariance_matrix.sum()
    assert_close(
        torch.autograd.grad(loss, query)[0],
        torch.autograd.grad(ref_loss, ref_query)[0],
        rtol=5e-7,
        atol=3e-9,
    )
    with pytest.raises(ValueError, match="one output"):
        actual.posterior(query, output_indices=[1])
    with pytest.raises(ValueError, match="raw"):
        actual._rs.set_raw_parameters([float("nan")] * len(actual._rs.raw_parameters()))


def test_em_model_zero_mean_index_queries_and_explicit_boundaries() -> None:
    from botorch_rs.exceptions.errors import UnsupportedError

    history = datasets("list")
    prior = pretrain_em_prior(
        history, ZeroMean(), RBFKernel(), enable_interpolation=False, num_em_iterations=3
    )
    model = EMEmpiricalGaussianProcess.from_pretrained(prior, [[0.0], [0.45]], [[0.2], [0.6]])
    assert model.num_outputs == 1
    assert model.initial_mean_module.batch_shape == ()
    assert len(model.posterior([[0.2], [0.7]]).mean) == 2
    with pytest.raises(ValueError, match="not found"):
        _ = model.posterior([[0.3]]).mean
    with pytest.raises(UnsupportedError, match="inducing"):
        EMEmpiricalGaussianProcess.from_pretrained(
            prior, [[0.0]], [[0.2]], learnable_inducing_points=True
        )
    fresh = EMEmpiricalGaussianProcess(
        [[0.1]],
        [[0.4]],
        datasets=history,
        mean_module=ZeroMean(),
        covar_module=RBFKernel(),
        num_em_iterations=2,
    )
    assert math.isfinite(fresh.posterior([[0.4]]).mean[0][0])
    assert math.isfinite(float(EMEmpiricalMarginalLogLikelihood(fresh.likelihood, fresh)()))


@pytest.mark.parametrize("zero_mean", [False, True])
@pytest.mark.parametrize("fixed_noise", [False, True])
def test_shared_gp_list_matches_official(official: Any, zero_mean: bool, fixed_noise: bool) -> None:
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    data = datasets("torch")
    mean = ZeroMean() if zero_mean else ConstantMean().initialize(constant=0.3)
    ref_mean = (
        gpytorch.means.ZeroMean()
        if zero_mean
        else gpytorch.means.ConstantMean()
        .double()
        .initialize(constant=torch.tensor(0.3, dtype=torch.double))
    )
    kernel = RBFKernel()
    kernel.lengthscale = 0.42
    ref_kernel = gpytorch.kernels.RBFKernel().double()
    ref_kernel.lengthscale = torch.tensor(0.42, dtype=torch.double)
    noise = 0.025 if fixed_noise else None
    actual, mll = build_shared_gp_model_list(data, mean, kernel, observation_noise=noise)
    expected, ref_mll = official.em_empirical_gp.build_shared_gp_model_list(
        [official.utils.ExperimentDataset(X=d.X, Y=d.Y) for d in data],
        ref_mean,
        ref_kernel,
        observation_noise=noise,
    )
    assert all(m.mean_module is mean and m.covar_module is kernel for m in actual.models)
    value, gradient = mll.value_and_grad()
    ref_value = ref_mll(expected(*expected.train_inputs), expected.train_targets)
    assert_close(value, ref_value, rtol=3e-7, atol=1e-8)
    ref_value.backward()
    ref_params = dict(ref_mll.named_parameters())
    offset = 0
    for name, shape in mll._rs.parameter_shapes():
        count = math.prod(shape)
        if ref_params[name].grad is not None:
            assert_close(
                gradient[offset : offset + count],
                ref_params[name].grad.reshape(-1),
                rtol=8e-7,
                atol=2e-8,
            )
        offset += count
    query = torch.tensor([[0.15], [0.65]], dtype=torch.double)
    assert_close(actual.posterior(query).mean, expected.posterior(query).mean, rtol=3e-7, atol=2e-8)
    kernel.lengthscale = 0.55
    assert all(
        abs(m._rs.raw_parameters()[-1] - actual.models[0]._rs.raw_parameters()[-1]) < 1e-14
        for m in actual.models
    )


def test_shared_gp_joint_fit_and_array_kinds(kind: str) -> None:
    kernel = ScaleKernel(RBFKernel())
    models, mll = build_shared_gp_model_list(
        datasets(kind), ZeroMean(), kernel, observation_noise=0.025
    )
    before = float(mll())
    noise = [float(m._raw_param("raw_noise")[0]) for m in models.models]
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 12}}, max_attempts=1)
    assert float(mll()) >= before - 1e-10
    assert [float(m._raw_param("raw_noise")[0]) for m in models.models] == noise
    fit_gpytorch_mll_scipy(mll, options={"maxiter": 3})
    fit_gpytorch_mll_torch(mll, step_limit=3, stopping_criterion=None)
    assert [float(m._raw_param("raw_noise")[0]) for m in models.models] == noise
    posterior = models.posterior(as_kind([[0.12], [0.6]], kind))
    check_kind(posterior.mean, kind)
    assert len(posterior.mean) == 2


@pytest.mark.parametrize("adam", [False, True])
def test_coordinate_ascent_updates_kernel_and_keeps_history(adam: bool) -> None:
    from botorch_rs.models.empirical_gps.em_coordinate_ascent import coordinate_ascent_em

    mean, kernel = modules()
    original = kernel._raw_flat()
    history: list[Any] = []
    prior = coordinate_ascent_em(
        datasets("list"),
        mean,
        kernel,
        likelihood_noise=0.035,
        inducing_points=[[0.05], [0.4], [0.85]],
        num_em_iterations=3,
        n_coord_ascent_iterations=2,
        mll_use_adam=adam,
        mll_max_iter=5,
        mll_adam_steps=4,
        container_history=history,
    )
    assert len(history) == 3
    assert history[0].covar_module._raw_flat() == original
    assert kernel._raw_flat() != original
    assert prior.covar_module is kernel
    assert_close(history[-1].Sigma_inducing, prior.Sigma_inducing)
    assert history[-1].covar_module is not kernel
    snapshot = history[0].covar_module._raw_flat()
    kernel.outputscale = 1.8
    assert history[0].covar_module._raw_flat() == snapshot


def test_subsampled_historical_likelihood_uses_correct_indices(official: Any) -> None:
    """Official coordinate subsampling retains stale per-dataset index arrays."""
    torch = pytest.importorskip("torch")
    actual, expected = model_pair(official, False, False)
    actual.datasets = [actual.datasets[2]]
    expected.datasets = [expected.datasets[2]]
    data = expected.datasets[0]
    indices = expected._find_indices_in_historical(data.X)
    mean = expected._mu_inducing[indices]
    covariance = (
        expected._Sigma_inducing[indices][:, indices]
        + torch.eye(len(indices), dtype=torch.double) * expected.likelihood.noise
    )
    correct = torch.distributions.MultivariateNormal(mean, covariance).log_prob(
        data.Y.squeeze(-1)
    ) / len(indices)
    value = EMEmpiricalMarginalLogLikelihood(actual.likelihood, actual)()
    assert_close(value, correct, rtol=2e-8, atol=2e-10)
    wrong = official.em_empirical_gp.EMEmpiricalMarginalLogLikelihood(
        expected.likelihood, expected
    )(None, None)
    assert abs(float((wrong - correct).detach())) > 1e-3


def test_coordinate_default_without_phi_dependence_is_a_valid_noop(official: Any) -> None:
    """An indexed detached prior has zero kernel gradient unless priors add one."""
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    from botorch_rs.models.empirical_gps.em_coordinate_ascent import coordinate_ascent_em

    ref_data = [official.utils.ExperimentDataset(X=d.X, Y=d.Y) for d in datasets("torch")]
    with pytest.raises(RuntimeError, match="does not require grad"):
        official.em_coordinate_ascent.coordinate_ascent_em(
            ref_data,
            gpytorch.means.ZeroMean(),
            gpytorch.kernels.RBFKernel().double(),
            likelihood_noise=0.035,
            num_em_iterations=3,
            mll_max_iter=1,
        )
    kernel = RBFKernel()
    before = kernel._raw_flat()
    prior = coordinate_ascent_em(
        datasets("torch"),
        ZeroMean(),
        kernel,
        likelihood_noise=torch.tensor(0.035, dtype=torch.double),
        num_em_iterations=3,
        mll_max_iter=1,
    )
    assert kernel._raw_flat() == before
    assert prior.Sigma_inducing.isfinite().all()


@pytest.mark.parametrize(
    "inducing,priors,defaults",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),
    ],
)
def test_from_scratch_em_rerun_values_and_gradients_match_official(
    official: Any, inducing: bool, priors: bool, defaults: bool
) -> None:
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    X = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)
    Y = torch.tensor([[0.7], [0.5], [0.8]], dtype=torch.double)
    mean, kernel = modules()
    ref_mean = (
        gpytorch.means.ConstantMean()
        .double()
        .initialize(constant=torch.tensor(0.35, dtype=torch.double))
    )
    ref_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    ref_kernel.base_kernel.lengthscale = torch.tensor(0.42, dtype=torch.double)
    ref_kernel.outputscale = torch.tensor(1.3, dtype=torch.double)
    options = (
        {}
        if defaults
        else dict(
            num_em_iterations=3,
            em_convergence_tol=None,
            use_mean_prior=priors,
            use_covar_prior=priors,
            covariance_shrinkage=0.13 if priors else 0.0,
        )
    )
    if inducing:
        options["inducing_points"] = torch.tensor([[0.05], [0.4], [0.85]], dtype=torch.double)
    actual = EMEmpiricalGaussianProcess(
        X, Y, datasets=datasets("torch"), mean_module=mean, covar_module=kernel, **options
    )
    ref_options = (
        {} if defaults else {"likelihood": gpytorch.likelihoods.GaussianLikelihood().double()}
    )
    expected = official.em_empirical_gp.EMEmpiricalGaussianProcess(
        X,
        Y,
        datasets=[official.utils.ExperimentDataset(X=d.X, Y=d.Y) for d in datasets("torch")],
        mean_module=ref_mean,
        covar_module=ref_kernel,
        **ref_options,
        **options,
    )
    assert actual.initial_mean_module is mean
    assert actual.initial_covar_module is kernel
    a_mll = EMEmpiricalMarginalLogLikelihood(actual.likelihood, actual)
    b_mll = official.em_empirical_gp.EMEmpiricalMarginalLogLikelihood(expected.likelihood, expected)
    value, gradient = a_mll.value_and_grad()
    reference = b_mll(None, None)
    assert_close(value, reference, rtol=2e-7, atol=2e-9)
    reference.backward()
    parameters = dict(expected.named_parameters())
    offset = 0
    for name, shape in actual._rs.parameter_shapes():
        size = math.prod(shape)
        assert_close(
            gradient[offset : offset + size],
            parameters[name].grad.reshape(-1),
            rtol=3e-5,
            atol=3e-7,
        )
        offset += size
    assert_close(actual._mu_inducing, expected._mu_inducing, rtol=2e-7, atol=2e-9)
    assert_close(actual._Sigma_inducing, expected._Sigma_inducing, rtol=2e-7, atol=2e-9)
    query = torch.tensor([[0.15], [0.62]], dtype=torch.double)
    assert_close(actual.posterior(query).mean, expected.posterior(query).mean, rtol=3e-7, atol=3e-9)


def test_default_em_training_native_fit_updates_moments() -> None:
    mean, kernel = modules()
    model = EMEmpiricalGaussianProcess(
        [[0.1], [0.5]],
        [[0.7], [0.6]],
        datasets=datasets("list"),
        mean_module=mean,
        covar_module=kernel,
        num_em_iterations=3,
        em_convergence_tol=None,
    )
    mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    before = float(mll())
    initial = copy.deepcopy(model._mu_inducing)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 5}}, max_attempts=1)
    assert float(mll()) >= before - 1e-9
    assert model._mu_inducing != initial
    assert kernel._model is model
    fit_gpytorch_mll_torch(mll, step_limit=3, stopping_criterion=None)
    assert math.isfinite(float(mll()))


def test_float32_rank_limited_em_covariance_roundtrip() -> None:
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    X = torch.linspace(0, 1, 6, dtype=torch.float32).unsqueeze(-1)
    history = [ExperimentDataset(X=X, Y=torch.sin(X * (i + 1) * 2)) for i in range(3)]
    prior = pretrain_em_prior(history, ZeroMean(), RBFKernel(), num_em_iterations=1)
    # Float32 storage can turn a rank-limited PSD estimate slightly indefinite.
    assert torch.linalg.eigvalsh(prior.Sigma_inducing.double()).min() < 0
    model = EMEmpiricalGaussianProcess.from_pretrained(prior, X[:2], history[0].Y[:2])
    posterior = model.posterior(X[2:])
    assert posterior.mean.dtype is torch.float32
    assert posterior.variance.isfinite().all()
    invalid = replace(prior, Sigma_inducing=-torch.eye(6, dtype=torch.float32))
    with pytest.raises(ValueError, match="indefinite"):
        EMEmpiricalGaussianProcess.from_pretrained(invalid, X[:2], history[0].Y[:2])


@pytest.mark.parametrize("inducing", [False, True])
@pytest.mark.parametrize("priors", [False, True])
def test_warm_start_em_values_gradients_and_state_match_official(
    official: Any, inducing: bool, priors: bool
) -> None:
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    mean, kernel = modules()
    ref_mean = gpytorch.means.ConstantMean().double()
    ref_mean.constant = torch.tensor(0.35, dtype=torch.double)
    ref_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    ref_kernel.base_kernel.lengthscale = torch.tensor(0.42, dtype=torch.double)
    ref_kernel.outputscale = torch.tensor(1.3, dtype=torch.double)
    X = torch.tensor([[0.1], [0.5]], dtype=torch.double)
    Y = torch.tensor([[0.7], [0.6]], dtype=torch.double)
    options = dict(
        warm_start_em=True,
        num_em_iterations=2,
        em_convergence_tol=None,
        use_mean_prior=priors,
        use_covar_prior=priors,
        covariance_shrinkage=0.13,
    )
    if inducing:
        options["inducing_points"] = torch.tensor([[0.05], [0.4], [0.85]], dtype=torch.double)
    actual = EMEmpiricalGaussianProcess(
        X, Y, datasets=datasets("torch"), mean_module=mean, covar_module=kernel, **options
    )
    expected = official.em_empirical_gp.EMEmpiricalGaussianProcess(
        X, Y,
        datasets=[official.utils.ExperimentDataset(X=d.X, Y=d.Y) for d in datasets("torch")],
        mean_module=ref_mean, covar_module=ref_kernel,
        likelihood=gpytorch.likelihoods.GaussianLikelihood().double(), **options,
    )
    a_mll = EMEmpiricalMarginalLogLikelihood(actual.likelihood, actual)
    b_mll = official.em_empirical_gp.EMEmpiricalMarginalLogLikelihood(expected.likelihood, expected)
    for step in range(3):
        value, gradient = a_mll.value_and_grad()
        reference = b_mll(None, None)
        reference.backward()
        assert_close(value, reference, rtol=3e-7, atol=3e-9)
        parameters = dict(expected.named_parameters())
        offset = 0
        for name, shape in actual._rs.parameter_shapes():
            size = math.prod(shape)
            g = parameters[name].grad
            assert_close(gradient[offset:offset+size], [0.0]*size if g is None else g.reshape(-1), rtol=3e-5, atol=3e-7)
            offset += size
        assert_close(actual._mu_inducing, expected._mu_inducing, rtol=3e-7, atol=3e-9)
        assert_close(actual._Sigma_inducing, expected._Sigma_inducing, rtol=3e-7, atol=3e-9)
        expected.zero_grad(set_to_none=True)
        if step == 1:
            previous = actual._mu_inducing.clone()
            kernel.base_kernel.lengthscale = 0.47
            ref_kernel.base_kernel.lengthscale = torch.tensor(0.47, dtype=torch.double)
            torch.testing.assert_close(actual._mu_inducing, previous)


@pytest.mark.parametrize("adam", [False, True])
def test_warm_em_native_fit_updates_moments_and_preserves_posterior(adam: bool) -> None:
    mean, kernel = modules()
    model = EMEmpiricalGaussianProcess(
        [[0.1], [0.5]], [[0.7], [0.6]], datasets=datasets("list"),
        mean_module=mean, covar_module=kernel, warm_start_em=True,
        inducing_points=[[0.05], [0.4], [0.85]], num_em_iterations=2,
        em_convergence_tol=None,
    )
    mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    posterior = model.posterior([[0.2], [0.7]])
    before = copy.deepcopy(posterior.mean)
    initial = copy.deepcopy(model._mu_inducing)
    raw = model._rs.raw_parameters()
    if adam:
        fit_gpytorch_mll_torch(mll, step_limit=3, stopping_criterion=None)
    else:
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 3}}, max_attempts=1)
    assert model._rs.raw_parameters() != raw
    assert model._mu_inducing != initial
    assert_close(posterior.mean, before)
    assert math.isfinite(float(mll()))
