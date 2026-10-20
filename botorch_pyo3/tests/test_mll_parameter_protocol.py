"""Native MLL parameter traversal, aliases and basic Torch optimizer protocol."""

from __future__ import annotations

import pytest

from helpers import as_kind, check_kind, importorskip_botorch, make_data

importorskip_botorch()
torch = pytest.importorskip("torch")

from approximate_helpers import CASES, build_pair, elbo_pair, set_random_parameters  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtExactMLL  # noqa: E402
from gpytorch.mlls import SumMarginalLogLikelihood as BtSumMLL  # noqa: E402
from test_posterior_parameter_gradients import model_pair  # noqa: E402

from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood  # noqa: E402
from botorch_rs.models import ModelListGP, SingleTaskGP  # noqa: E402
from botorch_rs.optim.utils.model_utils import get_parameters  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


def paired_mlls(family):
    if family == "variational":
        reference, model, _, _ = build_pair(CASES[5])
        set_random_parameters(reference, model, 31)
        expected, actual = elbo_pair(reference, model, 17)
        return actual, expected
    model, reference = model_pair(outputs=2, batch=(2,), tree=True)
    if family == "exact":
        return ExactMarginalLogLikelihood(model.likelihood, model), BtExactMLL(
            reference.likelihood, reference
        )
    second, expected_second = model_pair(outputs=1, batch=(2,))
    models, expected_models = [model, second], [reference, expected_second]
    if family == "aliased_sum":
        models.append(model)
        expected_models.append(reference)
    model, reference = ModelListGP(*models), BtModelListGP(*expected_models)
    return SumMarginalLogLikelihood(model.likelihood, model), BtSumMLL(
        reference.likelihood, reference
    )


def reference_value(mll):
    mll.train()
    return mll(mll.model(*mll.model.train_inputs), mll.model.train_targets)


@pytest.mark.parametrize("family", ["exact", "sum", "aliased_sum", "variational"])
@pytest.mark.parametrize("prefix", ["", "outer.branch"])
@pytest.mark.parametrize("recurse", [False, True])
@pytest.mark.parametrize("remove_duplicate", [False, True])
def test_mll_parameter_paths_match_reference_registration(
    family, prefix, recurse, remove_duplicate
):
    mll, reference = paired_mlls(family)
    kwargs = dict(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    actual = list(mll.named_parameters(**kwargs))
    expected = list(reference.named_parameters(**kwargs))
    assert [name for name, _ in actual] == [name for name, _ in expected]
    for (_, leaf), (_, want) in zip(actual, expected, strict=True):
        torch.testing.assert_close(leaf, want)
    # Alias equivalence as well as names: every repeated path must reach the
    # same leaf object, while separate model parameters remain independent.
    for i, (_, leaf) in enumerate(actual):
        for j, (_, other) in enumerate(actual):
            assert (leaf is other) == (expected[i][1] is expected[j][1])
    values = list(mll.parameters(recurse=recurse))
    assert len(values) == len(list(reference.parameters(recurse=recurse)))
    assert len(values) == len({id(value) for value in values})


@pytest.mark.parametrize("family", ["exact", "sum", "aliased_sum"])
@pytest.mark.parametrize("optimizer", ["Adam", "SGD"])
def test_standard_torch_optimizers_train_native_exact_mll_parameters(family, optimizer):
    mll, reference = paired_mlls(family)
    constructor = getattr(torch.optim, optimizer)
    lr = 0.0003 if optimizer == "Adam" else 0.00001
    actual_optimizer = constructor(mll.parameters(), lr=lr)
    expected_optimizer = constructor(reference.parameters(), lr=lr)
    leaves, expected_leaves = tuple(mll.parameters()), tuple(reference.parameters())
    losses = []
    for _ in range(3):
        mll.zero_grad()
        reference.zero_grad()
        actual, expected = mll(), reference_value(reference)
        torch.testing.assert_close(actual, expected, rtol=3e-7, atol=3e-8)
        weights = torch.linspace(0.2, 0.8, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        loss, expected_loss = -(actual * weights).sum(), -(expected * weights).sum()
        losses.append(loss.item())
        loss.backward()
        expected_loss.backward()
        for leaf, want in zip(leaves, expected_leaves, strict=True):
            torch.testing.assert_close(leaf.grad, want.grad, rtol=6e-7, atol=6e-8)
        actual_optimizer.step()
        expected_optimizer.step()
    assert losses[-1] < losses[0]
    for leaf, want in zip(mll.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(leaf, want, rtol=6e-7, atol=6e-8)


@pytest.mark.parametrize("family", ["exact", "sum", "aliased_sum", "variational"])
def test_mll_freezing_and_gradient_clearing_keep_parameter_identities(family):
    mll, _ = paired_mlls(family)
    leaves = tuple(mll.parameters())
    (-mll().sum()).backward()
    gradients = tuple(leaf.grad for leaf in leaves)
    assert all(gradient is not None for gradient in gradients)
    assert mll.zero_grad(set_to_none=False) is None
    for leaf, previous in zip(leaves, gradients, strict=True):
        assert leaf.grad is previous and torch.count_nonzero(leaf.grad) == 0
    mll.zero_grad()
    assert all(leaf.grad is None for leaf in leaves)
    assert mll.requires_grad_(False) is mll
    assert all(not leaf.requires_grad for leaf in leaves)
    assert not mll().requires_grad
    assert mll.requires_grad_() is mll
    assert all(leaf.requires_grad for leaf in leaves)
    assert all(leaf is again for leaf, again in zip(leaves, mll.parameters(), strict=True))
    assert mll().requires_grad


@pytest.mark.parametrize("family", ["exact", "sum", "aliased_sum", "variational"])
def test_public_parameter_traversal_does_not_recurse_through_native_optimizer_entries(family):
    mll, _ = paired_mlls(family)
    public, internal = dict(mll.named_parameters()), get_parameters(mll)
    assert public.keys() <= internal.keys()
    assert all(value is internal[name] for name, value in public.items())
    assert {id(value) for value in public.values()} == {id(value) for value in internal.values()}


def test_summed_mll_deduplicates_a_leaf_shared_by_distinct_child_models():
    first, expected_first = model_pair(batch=(2,))
    second, expected_second = model_pair(batch=(2,), tree=True)
    model = ModelListGP(first, second)
    expected_model = BtModelListGP(expected_first, expected_second)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    reference = BtSumMLL(expected_model.likelihood, expected_model)
    # Use the existing leaf registry to represent a shared native mean leaf.
    # All other kernel/noise parameters remain independent.
    shared = first.mean_module.raw_constant
    # Both means need matching shapes for a real shared Torch parameter.
    assert shared.shape == second.mean_module.raw_constant.shape
    state = second._sync_torch_parameters()
    state.entries = [
        (name, shared if name == "mean_module.raw_constant" else leaf)
        for name, leaf in state.entries
    ]
    expected_second.mean_module.raw_constant = expected_first.mean_module.raw_constant
    assert [name for name, _ in mll.named_parameters()] == [
        name for name, _ in reference.named_parameters()
    ]
    assert len(list(mll.parameters())) == len(list(reference.parameters()))


@pytest.mark.parametrize("kind", ["list", "numpy"])
@pytest.mark.parametrize("summed", [False, True])
def test_native_array_mll_parameter_descriptors_and_gradient_flag_boundary(kind, summed):
    x, y = make_data(n=7, d=2, seed=19)
    model = SingleTaskGP(as_kind(x, kind), as_kind(y, kind), outcome_transform=None)
    if summed:
        model = ModelListGP(model, model)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    else:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    parameters = list(mll.named_parameters())
    assert len(parameters) == 3
    assert not list(mll.parameters(recurse=False))
    assert not list(mll.named_parameters(recurse=False))
    for name, value in parameters:
        assert not name.startswith("model.likelihood.")
        check_kind(value, kind)
    assert mll.zero_grad() is None
    with pytest.raises(UnsupportedError, match="persistent CPU Torch"):
        mll.requires_grad_(False)
    assert len(list(mll.named_parameters(remove_duplicate=False))) > len(parameters)
