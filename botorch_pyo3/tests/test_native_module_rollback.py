"""Native model rollback with persistent Torch leaves and fixed raw layouts."""

from __future__ import annotations

import pytest

from helpers import KINDS, assert_close

torch = pytest.importorskip("torch")

from test_parameter_setter_aliases import inner, make_model  # noqa: E402

from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood, VariationalELBO  # noqa: E402
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy  # noqa: E402
from botorch_rs.utils.context_managers import TensorCheckpoint, module_rollback_ctx  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


def raw(model):
    return list(model._rs.raw_parameters())


def check_leaves(model, leaves):
    values, offset = raw(model), 0
    for name, count in model._rs.parameter_names():
        torch.testing.assert_close(
            leaves[name].detach().reshape(-1),
            torch.tensor(values[offset : offset + count], dtype=leaves[name].dtype),
            rtol=0,
            atol=0,
        )
        offset += count
    assert all(leaf is dict(model.named_parameters())[name] for name, leaf in leaves.items())


@pytest.mark.parametrize("family", ["exact", "variational"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mutation", ["setter", "fit"])
@pytest.mark.parametrize("exception", [False, True])
def test_fixed_layout_native_rollback_after_public_setters_and_fitting(
    family, kind, mutation, exception
):
    model, _ = make_model(family, 2, kind=kind)
    leaves = dict(model.named_parameters()) if kind == "torch" else None
    before = raw(model)

    def body():
        if mutation == "setter":
            inner(model, family).mean_module.constant = 0.7
            inner(model, family).covar_module.lengthscale = 0.4
        else:
            mll = (
                ExactMarginalLogLikelihood(model.likelihood, model)
                if family == "exact"
                else VariationalELBO(model.likelihood, model.model, num_data=8)
            )
            fit_gpytorch_mll_scipy(mll, options={"maxiter": 2})
        assert raw(model) != before
        if exception:
            raise RuntimeError("body failed")

    if exception:
        with pytest.raises(RuntimeError, match="body failed"), module_rollback_ctx(model):
            body()
    else:
        with module_rollback_ctx(model):
            body()
    assert raw(model) == before
    if leaves is not None:
        check_leaves(model, leaves)


@pytest.mark.parametrize("family", ["exact", "variational"])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("exception", [False, True])
def test_native_rollback_syncs_pending_entry_and_in_context_leaf_edits(family, pending, exception):
    model, _ = make_model(family, 1)
    leaves = dict(model.named_parameters())
    mean = inner(model, family).mean_module.raw_constant
    mean.requires_grad_(False)
    with torch.no_grad():
        mean.fill_(0.45 if pending else 0.0)

    def body():
        with torch.no_grad():
            for leaf in leaves.values():
                leaf.add_(0.2)
        if exception:
            raise RuntimeError("body failed")

    if exception:
        with pytest.raises(RuntimeError, match="body failed"), module_rollback_ctx(model):
            body()
    else:
        with module_rollback_ctx(model):
            body()
    # Inspect the held object without a getter that could hide a delayed pull.
    assert mean.item() == pytest.approx(0.45 if pending else 0.0, abs=0)
    assert not mean.requires_grad
    check_leaves(model, leaves)


@pytest.mark.parametrize("family", ["exact", "variational"])
@pytest.mark.parametrize("kind", KINDS)
def test_native_rollback_filter_and_explicit_checkpoint_values(family, kind):
    model, _ = make_model(family, 1, kind=kind)
    mean_name = next(
        name for name, _ in model._rs.parameter_names() if name.endswith("raw_constant")
    )
    with module_rollback_ctx(model, name_filter=lambda name: name == mean_name) as checkpoint:
        inner(model, family).mean_module.constant = 0.7
        model.likelihood.noise = 0.13
    assert list(checkpoint) == [mean_name]
    assert_close(inner(model, family).mean_module.constant, 0.0)
    assert_close(model.likelihood.noise, [0.13])
    # Caller changes to an explicit checkpoint define the restored state.
    checkpoint[mean_name] = TensorCheckpoint(0.31)
    with module_rollback_ctx(model, checkpoint=checkpoint):
        inner(model, family).mean_module.constant = 0.8
    assert_close(inner(model, family).mean_module.constant, 0.31)


@pytest.mark.parametrize("family", ["exact", "variational"])
@pytest.mark.parametrize("clear", [False, True])
def test_empty_or_cleared_native_checkpoint_commits_pending_leaf_edits(family, clear):
    model, _ = make_model(family, 1)
    leaf = inner(model, family).mean_module.raw_constant
    with module_rollback_ctx(model, checkpoint=None if clear else {}) as checkpoint:
        with torch.no_grad():
            leaf.fill_(0.73)
        checkpoint.clear()
    assert leaf.item() == 0.73
    # Commit flushes the pending edit to native storage, too.
    offset = 0
    for name, size in model._rs.parameter_names():
        if name.endswith("raw_constant"):
            assert raw(model)[offset] == 0.73
            break
        offset += size


@pytest.mark.parametrize("family", ["exact", "variational"])
def test_native_rollback_honors_torch_checkpoint_dtype(family):
    model, _ = make_model(family, 1)
    inner(model, family).mean_module.constant = 0.123456789
    leaf = inner(model, family).mean_module.raw_constant
    with module_rollback_ctx(model, dtype=torch.float32) as checkpoint:
        assert all(entry.values.dtype == torch.float32 for entry in checkpoint.values())
        inner(model, family).mean_module.constant = 0.7
    expected = torch.tensor(0.123456789, dtype=torch.float32).double()
    torch.testing.assert_close(leaf, expected, rtol=0, atol=0)


def test_native_rollback_rejects_changed_svgp_layout_metadata_explicitly():
    model, _ = make_model("variational", 1)
    key = next(name for name in model._param_shapes if name.endswith("variational_mean"))
    original = model._param_shapes[key]
    try:
        with pytest.raises(UnsupportedError, match="unchanged parameter layout"):
            with module_rollback_ctx(model):
                model._param_shapes[key] = (original[0] + 1,)
    finally:
        model._param_shapes[key] = original


@pytest.mark.parametrize("kind", KINDS)
def test_native_rollback_rejects_sparse_support_resizing(kind):
    from botorch_rs.likelihoods import GaussianLikelihood
    from botorch_rs.models import SingleTaskGP
    from botorch_rs.models.likelihoods.sparse_outlier_noise import SparseOutlierGaussianLikelihood
    from helpers import as_kind, make_data

    x, y = make_data(n=8, d=2, seed=31)
    likelihood = SparseOutlierGaussianLikelihood(GaussianLikelihood(), len(x), [])
    model = SingleTaskGP(
        as_kind(x, kind), as_kind(y, kind), likelihood=likelihood, outcome_transform=None
    )
    with pytest.raises(UnsupportedError, match="unchanged parameter layout"):
        with module_rollback_ctx(model):
            likelihood.noise_covar.expand_support([0])


@pytest.mark.parametrize("family", ["exact", "variational"])
def test_rollback_preserves_saved_native_loss_graphs(family):
    model, _ = make_model(family, 1)
    mll = (
        ExactMarginalLogLikelihood(model.likelihood, model)
        if family == "exact"
        else VariationalELBO(model.likelihood, model.model, num_data=8)
    )
    leaves = tuple(model.parameters() if family == "exact" else mll.parameters())
    with module_rollback_ctx(model):
        inner(model, family).mean_module.constant = 0.7
        loss = mll().sum()
        expected = torch.autograd.grad(loss, leaves, retain_graph=True)
    actual = torch.autograd.grad(loss, leaves)
    for gradient, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, want, rtol=0, atol=0)
