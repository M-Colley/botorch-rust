"""Public outcome transforms participate in real Torch module graphs."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import io
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from botorch_rs.models._backend_selection import requires_tensor_backend
from botorch_rs.models._torch_backend_modules import build_outcome_transform
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.higher_order_gp import FlattenedStandardize, HigherOrderGP
from botorch_rs.models.latent_kronecker_gp import LatentKroneckerGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP
from botorch_rs.models.transforms.outcome import (
    Bilog,
    ChainedOutcomeTransform,
    Log,
    OutcomeTransform,
    Power,
    Standardize,
    StratifiedStandardize,
)
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent")
    ),
]
FACTORIES = {
    "standardize": lambda: Standardize(1),
    "stratified": lambda: StratifiedStandardize(1, [1, 3]),
    "flattened": lambda: FlattenedStandardize((2, 2)),
    "log": Log,
    "power": lambda: Power(2.3),
    "bilog": Bilog,
    "chain": lambda: ChainedOutcomeTransform(log=Log(), standardize=Standardize(1)),
}


def _values(family, device):
    y = torch.linspace(
        0.2, 1.9, 16 if family == "flattened" else 4, device=device, dtype=torch.double
    )
    y = y.reshape(4, 2, 2) if family == "flattened" else y.unsqueeze(-1)
    x = y.new_tensor([[0.1, 1], [0.3, 3], [0.7, 1], [0.9, 3]])
    return y, x


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FACTORIES)
def test_public_outcomes_are_registered_modules_with_hooks_state_and_migration(device, family):
    transform = FACTORIES[family]()
    assert isinstance(transform, nn.Module)
    if family == "chain":
        assert isinstance(transform, nn.ModuleDict)
    owner = nn.ModuleDict({"outcome": transform}).to(device=device, dtype=torch.double)
    y, x = _values(family, device)
    seen = []
    handle = transform.register_forward_hook(lambda module, args, output: seen.append(module))
    actual, _ = owner["outcome"](y, X=x)
    handle.remove()
    assert seen == [transform]
    transform.eval()
    torch.testing.assert_close(transform.untransform(actual, X=x)[0], y)
    assert owner.get_submodule("outcome") is transform
    state = copy.deepcopy(owner.state_dict())
    assert all("_tensor_standardize" not in name for name in state)
    if family in ("standardize", "stratified", "flattened", "chain"):
        prefix = "outcome.standardize." if family == "chain" else "outcome."
        assert set(state) == {
            prefix + name for name in ("means", "stdvs", "_stdvs_sq", "_is_trained")
        }
    restored = nn.ModuleDict({"outcome": FACTORIES[family]()}).to(y)
    restored.load_state_dict(state, assign=True)
    restored.eval()
    torch.testing.assert_close(restored["outcome"](y, X=x)[0], actual)
    for value in owner.buffers():
        assert value.device.type == device
        if value.is_floating_point():
            assert value.dtype == torch.double
    owner.float().double().cpu()
    cpu_y, cpu_x = y.cpu(), x.cpu()
    torch.testing.assert_close(
        owner["outcome"](cpu_y, X=cpu_x)[0], actual.cpu(), rtol=2e-6, atol=2e-6
    )
    assert all(value.device.type == "cpu" for value in owner.buffers())


@pytest.mark.parametrize("family", FACTORIES)
def test_public_outcome_pickle_deepcopy_and_torch_save_preserve_buffers_and_children(family):
    transform = FACTORIES[family]().double()
    y, x = _values(family, "cpu")
    expected, _ = transform(y, X=x)
    transform.eval()
    stream = io.BytesIO()
    torch.save(transform, stream)
    stream.seek(0)
    for restored in (
        copy.deepcopy(transform),
        pickle.loads(pickle.dumps(transform)),
        torch.load(stream, weights_only=False),
    ):
        assert type(restored) is type(transform)
        assert isinstance(restored, nn.Module)
        torch.testing.assert_close(restored(y, X=x)[0], expected)
        assert restored.state_dict().keys() == transform.state_dict().keys()
        for name, tensor in restored.named_buffers():
            assert tensor is not dict(transform.named_buffers())[name]


@pytest.mark.parametrize("device", DEVICES)
def test_chained_module_dict_edits_and_child_hooks_follow_actual_execution(device):
    chain = ChainedOutcomeTransform(log=Log(), standardize=Standardize(1)).to(device)
    y, x = _values("chain", device)
    calls = []
    hook = chain["log"].register_forward_hook(lambda *args: calls.append("log"))
    chain(y, X=x)
    hook.remove()
    assert calls == ["log"]
    chain["power"] = Power(2)
    assert list(chain.keys()) == ["log", "standardize", "power"]
    assert chain.get_submodule("power") is chain["power"]
    del chain["power"]
    chain.update({"identity": Power(1)})
    assert chain.pop("identity").power == 1
    chain._transforms = chain._transforms
    assert list(chain.keys()) == ["log", "standardize"]
    assert list(chain.named_modules())[1][0] == "log"


def _compare_hessian(actual, expected, leaves):
    torch.testing.assert_close(actual, expected)
    first = torch.autograd.grad(actual, leaves, create_graph=True, retain_graph=True)
    reference = torch.autograd.grad(expected, leaves, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    ref_second = torch.autograd.grad(
        sum(g.square().sum() for g in reference), leaves, retain_graph=True
    )
    for a, b in zip((*first, *second), (*reference, *ref_second), strict=True):
        torch.testing.assert_close(a, b, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_supplied_power_parameter_keeps_module_identity_and_complete_posterior_graphs(device):
    y, _ = _values("power", device)
    y.requires_grad_()
    exponent = nn.Parameter(y.new_tensor(2.3))
    transform = Power(exponent)
    assert dict(transform.named_parameters()) == {"power": exponent}
    assert requires_tensor_backend(transform)
    assert build_outcome_transform(transform, 1, y) is transform
    actual = transform(y)[0]
    _compare_hessian(actual.sin().sum(), y.pow(exponent).sin().sum(), (y, exponent))
    scale = y.new_tensor(0.13, requires_grad=True)
    posterior = TorchGPyTorchPosterior(
        y, torch.eye(4, device=device, dtype=y.dtype) * scale.square()
    )
    bases = y.new_tensor([[-0.8, -0.3, 0.2, 0.7], [0.3, -0.2, 0.1, 0.4]])
    samples = transform.untransform_posterior(posterior).rsample_from_base_samples(
        torch.Size([2]), bases
    )
    expected = (y.squeeze(-1) + scale * bases).pow(1 / exponent).unsqueeze(-1)
    _compare_hessian(samples.sin().sum(), expected.sin().sum(), (y, exponent, scale))


class LearnedOutcome(OutcomeTransform):
    _is_linear = True

    def __init__(self, value):
        self.scale = nn.Parameter(value.new_tensor(1.7))
        self.register_buffer("offset", value.new_tensor(0.4))

    def forward(self, Y, Yvar=None, X=None):
        return (Y - self.offset) / self.scale, None if Yvar is None else Yvar / self.scale.square()

    def untransform(self, Y, Yvar=None, X=None):
        return Y * self.scale + self.offset, None if Yvar is None else Yvar * self.scale.square()

    def untransform_posterior(self, posterior, X=None):
        return TorchGPyTorchPosterior(
            posterior.mean * self.scale + self.offset,
            posterior._covariance_tensor * self.scale.square(),
        )


@pytest.mark.parametrize("device", DEVICES)
def test_custom_public_outcome_parameters_hooks_and_model_gradients(device):
    x = torch.linspace(0.1, 0.9, 4, device=device, dtype=torch.double).unsqueeze(-1)
    y = x.square().requires_grad_()
    outcome = LearnedOutcome(y)
    assert isinstance(outcome, nn.Module)
    assert requires_tensor_backend(outcome)
    model = SingleTaskGP(x, y, outcome_transform=outcome, backend="torch")
    assert model.outcome_transform is outcome
    assert dict(model.named_parameters())["outcome_transform.scale"] is outcome.scale
    assert dict(model.named_buffers())["outcome_transform.offset"] is outcome.offset
    query = x[:2].clone().requires_grad_()
    law = model.posterior(query)
    leaves = (y, outcome.scale, query)
    gradient = torch.autograd.grad(
        law.mean.sin().sum() + law.variance.square().sum(), leaves, create_graph=True
    )
    second = torch.autograd.grad(sum(g.square().sum() for g in gradient), leaves)
    assert all(torch.isfinite(value).all() for value in (*gradient, *second))


def test_plain_builtin_cpu_outcomes_retain_native_automatic_selection():
    for factory in FACTORIES.values():
        assert not requires_tensor_backend(factory())
    x = torch.linspace(0.1, 0.9, 4, dtype=torch.double).unsqueeze(-1)
    transform = Standardize(1)
    model = SingleTaskGP(x, x.square(), outcome_transform=transform)
    assert getattr(model, "_tensor_backend", None) != "torch"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kronecker", "higher", "latent"])
def test_structured_models_attach_actual_public_outcome_module(device, family):
    x = torch.linspace(0.1, 0.9, 4, device=device, dtype=torch.double).unsqueeze(-1)
    y = torch.cat((x.square(), (x + 0.3).sin()), -1).requires_grad_()
    if family == "higher":
        outcome = FlattenedStandardize((2,))
        model = HigherOrderGP(x, y, outcome_transform=outcome, backend="torch")
    elif family == "latent":
        outcome = Standardize(1)
        model = LatentKroneckerGP(
            x, x.new_tensor([[0.0], [1.0]]), y, outcome_transform=outcome, backend="torch"
        )
    else:
        outcome = Standardize(2)
        model = KroneckerMultiTaskGP(x, y, outcome_transform=outcome, backend="torch")
    assert model.outcome_transform is outcome
    calls = []
    hook = outcome.register_forward_hook(lambda *args: calls.append(True))
    law = model.posterior(x[:2])
    hook.remove()
    assert calls
    assert dict(model.named_buffers())["outcome_transform.means"] is outcome.means
    first = torch.autograd.grad(
        law.mean.sin().sum() + law.variance.square().sum(), y, create_graph=True
    )[0]
    second = torch.autograd.grad(first.square().sum(), y)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


@pytest.mark.parametrize("device", DEVICES)
def test_chain_refits_actual_external_standardize_children_without_losing_graphs(device):
    reference = pytest.importorskip("botorch.models.transforms.outcome")
    y, x = _values("chain", device)
    y.requires_grad_()
    child = reference.Standardize(1).to(y)
    chain = ChainedOutcomeTransform(log=Log(), standardize=child).to(y)
    assert chain["standardize"] is child
    build_outcome_transform(chain, 1, y, train_X=x)
    chain.fit(y, X=x)
    expected = (y.log() - y.log().mean(-2, keepdim=True)) / y.log().std(-2, keepdim=True)
    _compare_hessian(chain(y)[0].sin().sum(), expected.sin().sum(), (y,))
    assert not child.training


def test_outcome_native_import_and_pickle_do_not_require_torch():
    from botorch_rs import _botorch_rs

    script = r"""
import importlib.util, pickle, sys
sys.path.insert(0, sys.argv[1])
for name in ('torch', 'botorch', 'gpytorch', 'linear_operator', 'scipy', 'pyro', 'numpyro', 'jax'):
    sys.modules[name] = None
name = 'botorch_rs._botorch_rs'
spec = importlib.util.spec_from_file_location(name, sys.argv[2])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform, Log, Standardize
transform = ChainedOutcomeTransform(log=Log(), standardize=Standardize(1))
expected, _ = transform([[0.2], [0.7], [1.3]])
transform.eval()
restored = pickle.loads(pickle.dumps(transform))
assert restored([[0.2], [0.7], [1.3]])[0] == expected
assert sys.modules['torch'] is None
assert 'botorch_rs.models.transforms._torch_outcome_module' not in sys.modules
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).parents[1] / "python"),
            _botorch_rs.__file__,
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stratified", [False, True])
def test_explicit_native_refit_keeps_torch_module_destination(device, stratified):
    transform = StratifiedStandardize(1, [1, 3]) if stratified else Standardize(1)
    transform.to(device=device, dtype=torch.double)
    y, x = _values("standardize", "cpu")
    transform(y.tolist(), X=x.tolist())
    assert transform.means.device.type == device
    assert transform.means.dtype == torch.double
    expected = StratifiedStandardize(1, [1, 3]) if stratified else Standardize(1)
    expected(y.to(device), X=x.to(device))
    torch.testing.assert_close(transform.means, expected.means)
