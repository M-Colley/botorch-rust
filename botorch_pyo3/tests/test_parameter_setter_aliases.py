"""Public setters update held Torch leaves immediately without replacing them."""

from __future__ import annotations

import pytest

from helpers import as_kind, assert_close, check_kind, make_data

torch = pytest.importorskip("torch")

from botorch_rs.kernels import MaternKernel, RBFKernel, ScaleKernel  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood, VariationalELBO  # noqa: E402
from botorch_rs.models import SingleTaskGP, SingleTaskVariationalGP  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


def make_model(family, outputs, dtype=torch.float64, tree=False, kind="torch"):
    x, y = make_data(n=8, d=2, seed=31)
    y = [[row[0] + 0.2 * index for index in range(outputs)] for row in y]
    if kind == "torch":
        x, y = torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype)
    else:
        x, y = as_kind(x, kind), as_kind(y, kind)
    aug = (outputs,) if outputs > 1 else ()
    kernel = (
        ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=2, batch_shape=aug), batch_shape=aug)
        if tree
        else RBFKernel(ard_num_dims=2, batch_shape=aug)
    )
    kwargs = {"covar_module": kernel, "outcome_transform": None}
    if family == "exact":
        model = SingleTaskGP(x, y, **kwargs)
    else:
        model = SingleTaskVariationalGP(x, y, inducing_points=3, **kwargs)
        model._ensure_initialized()
    return model, x


def inner(model, family):
    return model if family == "exact" else model.model


def strategy(model):
    vs = model.model.variational_strategy
    return getattr(vs, "base_variational_strategy", vs)


@pytest.mark.parametrize("family", ["exact", "variational"])
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_public_setters_immediately_refresh_held_parameter_leaves(
    family, outputs, dtype, tree, frozen
):
    model, x = make_model(family, outputs, dtype, tree)
    leaves = dict(model.named_parameters())
    for leaf in leaves.values():
        leaf.requires_grad_(not frozen)

    def check():
        # Reading the held objects and the native vector must suffice: no
        # property getter / named_parameters call may mask a delayed pull.
        raw = model._rs.raw_parameters()
        offset = 0
        for name, count in model._rs.parameter_names():
            want = torch.tensor(raw[offset : offset + count], dtype=dtype)
            torch.testing.assert_close(leaves[name].detach().reshape(-1), want, rtol=0, atol=0)
            assert leaves[name].requires_grad is not frozen
            offset += count
        assert all(leaf is dict(model.named_parameters())[name] for name, leaf in leaves.items())

    component = inner(model, family)
    kernel = component.covar_module
    base = kernel.base_kernel if tree else kernel
    updates = [
        (component.mean_module, "constant", 0.7),
        (component.mean_module, "raw_constant", -0.2),
        (base, "lengthscale", 0.4),
        (base, "raw_lengthscale", 0.2),
        (model.likelihood, "noise", 0.13),
        (model.likelihood, "raw_noise", 0.21),
    ]
    if tree:
        updates.extend([(kernel, "outputscale", 0.8), (kernel, "raw_outputscale", -0.1)])
    if family == "variational":
        vs = strategy(model)
        updates.extend(
            [
                (vs._variational_distribution, "variational_mean", 0.12),
                (vs._variational_distribution, "chol_variational_covar", 0.2),
                (vs, "inducing_points", x[:3].clone()),
            ]
        )
        if outputs > 1:
            updates.extend(
                [
                    (model.likelihood, "task_noises", 0.19),
                    (model.likelihood, "raw_task_noises", -0.3),
                ]
            )
    for module, name, value in updates:
        setattr(module, name, value)
        check()
    key = "mean_module.constant" if family == "exact" else "model.mean_module.constant"
    model.initialize(**{key: 0.31})
    check()
    if family == "variational":
        model.init_inducing_points(x)
        check()


@pytest.mark.parametrize("family", ["exact", "variational"])
@pytest.mark.parametrize("kind", ["list", "numpy"])
def test_public_setters_preserve_non_torch_array_behavior(family, kind):
    model, _ = make_model(family, 2, kind=kind)
    component = inner(model, family)
    component.mean_module.constant = 0.7
    component.covar_module.lengthscale = 0.4
    model.likelihood.noise = 0.13
    for value in (component.mean_module.constant, component.covar_module.lengthscale):
        check_kind(value, kind)
    assert_close(component.mean_module.constant, [0.7, 0.7], rtol=0, atol=0)
    assert_close(component.covar_module.lengthscale, [[[0.4, 0.4]], [[0.4, 0.4]]])


@pytest.mark.parametrize("family", ["exact", "variational"])
@pytest.mark.parametrize("outputs", [1, 2])
def test_public_setters_leave_saved_native_loss_and_posterior_graphs_immutable(family, outputs):
    model, x = make_model(family, outputs)
    mll = (
        ExactMarginalLogLikelihood(model.likelihood, model)
        if family == "exact"
        else VariationalELBO(model.likelihood, model.model, num_data=len(x))
    )
    leaves = tuple(model.parameters() if family == "exact" else mll.parameters())
    loss = mll()
    if family == "exact":
        posterior = model.posterior(x[:2])
        loss = loss.sum() + posterior.mean.square().sum() + posterior.variance.sum()
    expected = torch.autograd.grad(loss, leaves, retain_graph=True)
    component = inner(model, family)
    component.mean_module.constant = 0.7
    component.covar_module.lengthscale = 0.4
    model.likelihood.noise = 0.13
    if family == "variational":
        strategy(model)._variational_distribution.variational_mean = 0.12
    actual = torch.autograd.grad(loss, leaves)
    for gradient, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, want, rtol=0, atol=0)
