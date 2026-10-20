"""Numerical parity of empirical models against official source b0dd01be6."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any

import pytest

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.empirical_gps.empirical_1d_gp import (
    EmpiricalOneDimensionalGP,
    EmpiricalOneDimensionalKernel,
    EmpiricalOneDimensionalMean,
)
from botorch_rs.models.empirical_gps.multioutput_empirical_1d_gp import (
    MultiOutputEmpiricalOneDimensionalGP,
    MultiOutputEmpiricalOneDimensionalKernel,
    MultiOutputEmpiricalOneDimensionalMean,
)
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from helpers import KINDS, as_kind, assert_close, check_kind, shape, to_list


def test_scalar_empirical_kernel_in_single_task_gp(kind: str) -> None:
    H, curves, X, Y, _ = data(1)
    kernel = EmpiricalOneDimensionalKernel(as_kind(H, kind), as_kind(curves, kind))
    gp = SingleTaskGP(
        as_kind(X, kind), as_kind(Y, kind), covar_module=kernel, outcome_transform=None
    )
    posterior = gp.posterior(as_kind([[0.15], [0.65]], kind))
    check_kind(posterior.mean, kind)
    assert shape(posterior.mean) == (2, 1)
    assert all(math.isfinite(v) for row in to_list(posterior.variance) for v in row)


@pytest.fixture(autouse=True)
def restore_torch_dtype() -> Any:
    if importlib.util.find_spec("torch") is None:
        yield
        return
    import torch

    before = torch.get_default_dtype()
    yield
    torch.set_default_dtype(before)


@pytest.fixture(scope="module")
def official() -> object:
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
    result = {}
    for name in (
        "utils",
        "empirical_1d_gp",
        "multioutput_empirical_1d_gp",
        "multitask_empirical_1d_gp",
    ):
        spec = importlib.util.spec_from_file_location(f"{package}.{name}", folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result[name] = module
    return SimpleNamespace(**result)


def data(outputs: int = 2) -> tuple[list, list, list, list, list]:
    H = [[0.0], [0.2], [0.45], [0.75], [1.0]]
    curves = [
        [[2 + 0.2 * r + math.sin((r + 1) * p[0] * 1.2 + s) for s in range(outputs)] for p in H]
        for r in range(7)
    ]
    X = [[0.1], [0.35], [0.65], [0.9]]
    Y = [[2.2 + math.sin(4 * p[0] + s) for s in range(outputs)] for p in X]
    return H, curves, X, Y, [[0.17], [0.56], [0.82]]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("correlated", [False, True])
def test_empirical_api_fitting_and_snapshot(kind: str, correlated: bool) -> None:
    H, history, X, Y, Q = (as_kind(v, kind) for v in data())
    cls = MultiOutputEmpiricalOneDimensionalGP if correlated else EmpiricalOneDimensionalGP
    model = cls(X, Y, H, history, ard=True)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = sum(to_list(mll())) if not correlated else float(to_list(mll()))
    result = fit_gpytorch_mll_scipy(mll, options={"maxiter": 12})
    after = sum(to_list(mll())) if not correlated else float(to_list(mll()))
    assert math.isfinite(result.fval) and after > before
    raw = model._rs.raw_parameters()
    fit_gpytorch_mll_torch(mll, step_limit=3, stopping_criterion=None)
    assert model._rs.raw_parameters() != raw
    p = model.posterior(Q)
    assert shape(p.mean) == shape(p.variance) == (3, 2)
    assert shape(p.distribution.covariance_matrix) == (6, 6)
    assert shape(p.rsample((4,))) == (4, 3, 2)
    check_kind(p.mean, kind)
    old = to_list(p.mean)
    values = model._rs.raw_parameters()
    values[0] += 1.0
    model._rs.set_raw_parameters(values)
    assert_close(p.mean, old)
    with pytest.raises(ValueError, match="outside"):
        _ = model.posterior(as_kind([[1.1]], kind)).mean


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize(
    "ard,compress", [(False, False), (False, True), (True, False), (True, True)]
)
def test_official_kernel_mean_and_input_derivatives(
    official: object, correlated: bool, ard: bool, compress: bool
) -> None:
    import torch

    H, history, _, _, Q = [torch.tensor(v, dtype=torch.float64) for v in data()]
    module = official.multioutput_empirical_1d_gp if correlated else official.empirical_1d_gp
    prefix = "MultiOutput" if correlated else ""
    k_cls = (
        MultiOutputEmpiricalOneDimensionalKernel if correlated else EmpiricalOneDimensionalKernel
    )
    mean_cls = MultiOutputEmpiricalOneDimensionalMean if correlated else EmpiricalOneDimensionalMean
    a = k_cls(H, history, ard=ard, use_svd=compress)
    b = getattr(module, f"{prefix}EmpiricalOneDimensionalKernel")(
        H, history, ard=ard, use_svd=compress
    )
    ma, mb = (
        mean_cls(H, history),
        getattr(module, f"{prefix}EmpiricalOneDimensionalMean")(H, history),
    )
    q1, q2 = Q.clone().requires_grad_(), Q.clone().requires_grad_()
    z = torch.tensor([[0.26], [0.62], [0.93]], dtype=torch.float64)
    ka, kb = a(q1, z), b.forward(q2, z)
    torch.testing.assert_close(ka, kb, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(ma(q1), mb(q2), atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(ka.sum() + ma(q1).sum(), q1)[0]
    gb = torch.autograd.grad(kb.sum() + mb(q2).sum(), q2)[0]
    torch.testing.assert_close(ga, gb, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(a(Q, z, diag=True), torch.diagonal(kb.detach(), dim1=-2, dim2=-1))
    # Exercise a concrete upstream fix without changing its source checkout.
    if correlated:
        b.num_outputs_per_input = MethodType(lambda self, x1, x2: self.num_outputs, b)
    else:
        b.batch_shape = torch.Size((b.num_outputs,))
    torch.testing.assert_close(b(Q, z).to_dense(), kb.detach(), atol=3e-10, rtol=3e-10)
    # Actual native tree values agree with standalone helper execution.
    augmented = [value for row in Q.tolist() for s in range(2) for value in (row[0], float(s))]
    pair = (augmented, (6, 2))
    dense, _ = a._node().evaluate(a._raw_flat(), pair, None, False)
    dense = torch.tensor(dense, dtype=Q.dtype).reshape(6, 6)
    if correlated:
        torch.testing.assert_close(dense, a(Q), atol=3e-10, rtol=3e-10)
    else:
        expected = a(Q)
        torch.testing.assert_close(dense[0::2, 0::2], expected[0])
        torch.testing.assert_close(dense[1::2, 1::2], expected[1])
        assert dense[0::2, 1::2].abs().max() == 0


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_official_model_mll_gradients_and_joint_posterior(
    official: object, correlated: bool, fixed: bool
) -> None:
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood as OfficialMLL

    torch.set_default_dtype(torch.float64)
    H, history, X, Y, Q = [torch.tensor(v) for v in data()]
    cls = MultiOutputEmpiricalOneDimensionalGP if correlated else EmpiricalOneDimensionalGP
    module = official.multioutput_empirical_1d_gp if correlated else official.empirical_1d_gp
    name = ("MultiOutput" if correlated else "") + "EmpiricalOneDimensionalGP"
    variance = torch.full_like(Y, 0.023) if fixed else None
    a = cls(X, Y, H, history, train_Yvar=variance, ard=True)
    b = getattr(module, name)(X, Y, H, history, train_Yvar=variance, ard=True)
    weights = torch.linspace(0.4, 1.3, 7)
    a.covar_module.curve_weights = weights
    with torch.no_grad():
        (b.covar_module.base_kernel if correlated else b.covar_module).curve_weights.copy_(weights)
    va, ga = ExactMarginalLogLikelihood(a.likelihood, a).value_and_grad()
    vb = OfficialMLL(b.likelihood, b)(b(*b.train_inputs), b.train_targets)
    vb.sum().backward()
    gb = torch.cat([p.grad.reshape(-1) for p in b.parameters() if p.requires_grad])
    torch.testing.assert_close(torch.as_tensor(va), vb, atol=3e-9, rtol=3e-9)
    torch.testing.assert_close(torch.tensor(ga), gb, atol=2e-8, rtol=2e-8)
    b.eval()
    for noise in (False, True):
        pa, pb = a.posterior(Q, observation_noise=noise), b.posterior(Q, observation_noise=noise)
        torch.testing.assert_close(pa.mean, pb.mean, atol=4e-9, rtol=4e-9)
        torch.testing.assert_close(pa.variance, pb.variance, atol=4e-9, rtol=4e-9)
        torch.testing.assert_close(
            pa.distribution.covariance_matrix,
            pb.distribution.covariance_matrix,
            atol=4e-9,
            rtol=4e-9,
        )


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_additive_base_parameter_binding_and_copy(kind: str, correlated: bool) -> None:
    from botorch_rs.kernels import RBFKernel, ScaleKernel

    H, history, X, Y, Q = (as_kind(v, kind) for v in data())
    cls = MultiOutputEmpiricalOneDimensionalGP if correlated else EmpiricalOneDimensionalGP
    base = ScaleKernel(RBFKernel())
    model = cls(X, Y, H, history, base_covar_module=base)
    base.outputscale = 0.37
    base.base_kernel.lengthscale = 0.21
    names = dict(model.named_parameters())
    assert "covar_module.base_kernel.raw_outputscale" in names
    assert "covar_module.base_kernel.base_kernel.raw_lengthscale" in names
    cov = model.covar_module(Q)
    assert shape(cov) == ((6, 6) if correlated else (2, 3, 3))
    cloned = copy.deepcopy(model)
    cloned.covar_module.base_kernel.outputscale = 0.91
    assert to_list(model.covar_module.base_kernel.outputscale) == pytest.approx(0.37)
    assert_close(cloned.train_targets, model.train_targets)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fixed", [False, True])
def test_single_output_noise_metadata_and_configuration_boundaries(kind: str, fixed: bool) -> None:
    from botorch_rs.exceptions.errors import UnsupportedError

    H, history, X, Y, Q = (as_kind(v, kind) for v in data(1))
    variance = as_kind([[0.01], [0.02], [0.03], [0.04]], kind) if fixed else None
    model = EmpiricalOneDimensionalGP(X, Y, H, history, train_Yvar=variance)
    assert shape(model.train_targets) == (4,)
    assert shape(model.posterior(Q).mean) == (3, 1)
    if fixed:
        assert_close(model.likelihood.noise, [0.01, 0.02, 0.03, 0.04])
    else:
        model.likelihood.noise = 0.17
        assert_close(model.likelihood.noise, [0.17])
        with pytest.raises(ValueError, match="constraint"):
            model.likelihood.noise = -0.3
    with pytest.raises(UnsupportedError, match="boolean"):
        model.posterior(Q, observation_noise=as_kind([[0.1]] * 3, kind))
    with pytest.raises(UnsupportedError, match="input_transform"):
        EmpiricalOneDimensionalGP(X, Y, H, history, input_transform=object())
