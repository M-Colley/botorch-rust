"""Native hierarchical kernels, fitting derivatives and exact GP parity."""

from __future__ import annotations

import copy
import re

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from botorch.models.hierarchical.conditional_kernel_gp import (  # noqa: E402
    HierarchicalConditionalKernel as BtKernel,
)
from botorch.models.hierarchical.conditional_kernel_gp import (  # noqa: E402
    HierarchicalConditionalKernelGP as BtGP,
)
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402

from botorch_rs.mlls import ExactMarginalLogLikelihood  # noqa: E402
from botorch_rs.models.hierarchical import (  # noqa: E402
    HierarchicalConditionalKernel,
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
)
from botorch_rs.models.transforms.input import Normalize  # noqa: E402
from botorch_rs.sampling.pathwise.features import KernelEvaluationMap  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
DEPS = {2: {0: [3], 1: [4]}, 4: {0: [5], 1: [6]}}


def inputs():
    x = torch.rand(8, 7, generator=torch.Generator().manual_seed(883), dtype=torch.float64)
    x[:, 2] = torch.tensor([0, 1] * 4)
    x[:, 4] = torch.tensor([0, 0, 1, 1] * 2)
    return x


def as_kind(x, kind):
    return x if kind == "torch" else x.numpy() if kind == "numpy" else x.tolist()


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("keep,separate", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("scaled", [False, True])
def test_hierarchical_kernel_values_and_psd(kind, keep, separate, scaled):
    x = inputs()
    args = dict(
        dim=7,
        hierarchical_dependencies=DEPS,
        eval_hierarchical_features=keep,
        separate_hierarchical_features=separate,
        use_saas_prior=False,
        use_outputscale=scaled,
    )
    actual, reference = HierarchicalConditionalKernel(**args), BtKernel(**args).double()
    for a, b in zip(actual.kernels, reference.kernels, strict=True):
        abase, bbase = (a.base_kernel, b.base_kernel) if scaled else (a, b)
        abase.lengthscale = 0.7
        bbase.lengthscale = torch.tensor(0.7, dtype=x.dtype)
        if scaled:
            a.outputscale = 1.3
            b.outputscale = torch.tensor(1.3, dtype=x.dtype)
    result = actual(as_kind(x, kind))
    np.testing.assert_allclose(result, reference(x).to_dense().detach(), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        actual(as_kind(x, kind), as_kind(x.flip(0), kind), diag=True),
        reference(x, x.flip(0), diag=True).detach(),
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.linalg.eigvalsh(np.asarray(result)).min() > -1e-10
    assert actual.partition == reference.partition


def test_native_hierarchical_input_adjoints_and_inactive_values():
    x = inputs().requires_grad_()
    points = inputs().flip(0).clone().requires_grad_()
    args = dict(dim=7, hierarchical_dependencies=DEPS, use_saas_prior=False, use_outputscale=False)
    actual, reference = HierarchicalConditionalKernel(**args), BtKernel(**args).double()
    values = KernelEvaluationMap(actual, points)(x)
    expected = reference(x, points).to_dense()
    ag = torch.autograd.grad(values.sum(), (x, points))
    eg = torch.autograd.grad(expected.sum(), (x, points))
    for a, e in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
    # C5/C6 are inactive in rows whose root parent chooses C3 instead of P4.
    assert torch.equal(ag[0][x[:, 2] == 0, 5:], torch.zeros_like(ag[0][x[:, 2] == 0, 5:]))


@pytest.mark.parametrize("batch,m", [((), 1)])
@pytest.mark.parametrize("saas", [False, True])
def test_hierarchical_gp_likelihood_gradient_and_posterior(batch, m, saas):
    x = inputs().expand(*batch, 8, 7)
    y = torch.randn(*batch, 8, m, generator=torch.Generator().manual_seed(45), dtype=x.dtype)
    kwargs = dict(
        hierarchical_dependencies=DEPS,
        train_Yvar=torch.full_like(y, 0.02),
        use_saas_prior=saas,
        outcome_transform=None,
    )
    reference, actual = BtGP(x, y, **kwargs), HierarchicalConditionalKernelGP(x, y, **kwargs)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            values = torch.full_like(parameter, 0.23)
            if name.endswith("raw_tau"):
                values.fill_(-2.0)
            parameter.copy_(values)
            name = re.sub(r"covar_module\.kernels\.\d+", r"\g<0>.kernels.1", name)
            actual._set_raw_param(name, values)
    mb, mr = (
        BtMLL(reference.likelihood, reference),
        ExactMarginalLogLikelihood(actual.likelihood, actual),
    )
    value = mb(reference(*reference.train_inputs), reference.train_targets)
    value.sum().backward()
    expected_gradient = torch.cat([p.grad.reshape(-1) for p in mb.parameters()])
    result, gradient = mr.value_and_grad()
    torch.testing.assert_close(
        torch.as_tensor(result, dtype=x.dtype), value.detach(), rtol=3e-7, atol=1e-7
    )
    torch.testing.assert_close(
        torch.as_tensor(gradient, dtype=x.dtype), expected_gradient, rtol=3e-7, atol=2e-7
    )
    query = inputs()[:3]
    pb, pr = reference.posterior(query), actual.posterior(query)
    torch.testing.assert_close(pr.mean, pb.mean.detach(), rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix.detach(),
        rtol=1e-9,
        atol=1e-10,
    )
    clone = copy.deepcopy(actual)
    torch.testing.assert_close(clone.posterior(query).mean, pr.mean)


@pytest.mark.parametrize("training", [False, True])
def test_normalize_learned_branch_values_and_empty_hierarchy(training):
    x = torch.tensor([[2.0, 0.1], [4.0, 0.3], [2.0, 0.4], [4.0, 0.8]], dtype=torch.float64)
    y = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=x.dtype)
    actual = HierarchicalConditionalKernelGP(
        x, y, {0: {2: [1]}}, input_transform=Normalize(2).train(training), use_saas_prior=False
    )
    assert actual.covar_module.hierarchical_dependencies == {0: {0.0: [1]}}
    assert torch.isfinite(actual.posterior(x).mean).all()
    empty = HierarchicalConditionalKernelGP(
        x, y, {}, input_transform=Normalize(2), use_saas_prior=False
    )
    assert empty.covar_module.partition == [[0, 1]]
    assert torch.isfinite(empty.posterior(x).variance).all()


@pytest.mark.parametrize("task_feature", [0, 1, 3])
def test_multitask_feature_mapping_after_normalize(task_feature):
    features = torch.tensor(
        [[0.1, 2.0, 0.3], [0.3, 6.0, 0.5], [0.6, 2.0, 0.4], [0.8, 6.0, 0.9]], dtype=torch.float64
    )
    tasks = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=features.dtype)
    x = torch.cat([features[:, :task_feature], tasks, features[:, task_feature:]], dim=-1)
    y = torch.tensor([[1.0], [2.0], [4.0], [3.0]], dtype=x.dtype)
    model = HierarchicalConditionalKernelMultiTaskGP(
        x,
        y,
        task_feature,
        {1: {2: [0], 6: [2]}},
        input_transform=Normalize(4, indices=[i for i in range(4) if i != task_feature]),
        use_saas_prior=False,
    )
    assert model.covar_module.kernels[0].hierarchical_dependencies == {1: {0.0: [0], 1.0: [2]}}
    assert torch.isfinite(model.posterior(features[:2]).variance).all()


def test_parent_values_validation_is_local_to_each_parent():
    # Distinct parent variables can have nearby category labels without ambiguity.
    kernel = HierarchicalConditionalKernel(
        4, {0: {1.0: [1]}, 2: {1.000001: [3]}}, use_saas_prior=False
    )
    assert len(kernel.paths) == 3
    with pytest.raises(ValueError, match="closer"):
        HierarchicalConditionalKernel(2, {0: {1.0: [1], 1.000001: []}})


@pytest.mark.parametrize("batch,m", [((2,), 1), ((), 2)])
def test_shared_batched_hierarchical_hyperparameters_are_explicitly_unsupported(batch, m):
    from botorch_rs.exceptions.errors import UnsupportedError

    x = inputs().expand(*batch, 8, 7)
    y = torch.ones(*batch, 8, m, dtype=x.dtype)
    with pytest.raises(UnsupportedError, match="batch_shape"):
        HierarchicalConditionalKernelGP(x, y, DEPS)
