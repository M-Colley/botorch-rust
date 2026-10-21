"""Official heterogeneous covariance, GP likelihood, and derivative parity."""

from __future__ import annotations

import copy

import pytest

from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape

importorskip_botorch()
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from botorch.models.heterogeneous_mtgp import HeterogeneousMTGP as BtGP  # noqa: E402
from botorch.models.kernels.heterogeneous_multitask import (  # noqa: E402
    CombinatorialCovarModule as BtCombinatorial,
)
from botorch.models.kernels.heterogeneous_multitask import (  # noqa: E402
    MultiTaskConditionalKernel as BtKernel,
)
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402

from botorch_rs.mlls import ExactMarginalLogLikelihood  # noqa: E402
from botorch_rs.models.heterogeneous_mtgp import HeterogeneousMTGP  # noqa: E402
from botorch_rs.models.kernels.heterogeneous_multitask import (  # noqa: E402
    CombinatorialCovarModule,
    MultiTaskConditionalKernel,
)
from botorch_rs.models.transforms.input import Normalize  # noqa: E402
from botorch_rs.models.transforms.outcome import Standardize  # noqa: E402
from botorch_rs.sampling.pathwise.features import KernelEvaluationMap  # noqa: E402
from botorch_rs.utils.datasets import MultiTaskDataset, SupervisedDataset  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
FEATURES = [[0, 1], [1, 2], [0, 2]]


def full_inputs():
    x = torch.rand(9, 4, generator=torch.Generator().manual_seed(739), dtype=torch.float64)
    x[:, 3] = torch.tensor([0, 1, 2] * 3)
    return x


def training(kind="torch"):
    full = full_inputs()
    xs = [full[i::3, indices] for i, indices in enumerate(FEATURES)]
    ys = [torch.tensor([[0.2], [0.7], [-0.3]], dtype=torch.float64) + i * 0.3 for i in range(3)]
    return [as_kind(x.tolist(), kind) for x in xs], [as_kind(y.tolist(), kind) for y in ys]


def align_kernels(actual, reference):
    assert [name for name, _ in actual.named_parameters()] == [
        name for name, _ in reference.named_parameters()
    ]
    # Exercise noninitial constrained values, including the learned SAAS tau.
    with torch.no_grad():
        for (name, _), (_, b) in zip(
            actual.named_parameters(), reference.named_parameters(), strict=True
        ):
            values = torch.full_like(b, -2.0 if name.endswith("raw_tau") else 0.23)
            b.copy_(values)
            owner, parameter = name.rsplit(".", 1) if "." in name else ("", name)
            obj = actual
            for part in owner.split(".") if owner else []:
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
            obj._set_raw(parameter, values)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("saas", [False, True])
@pytest.mark.parametrize("combinatorial", [False, True])
def test_conditional_kernel_values_diagonals_and_psd(kind, saas, combinatorial):
    args = dict(
        feature_indices=FEATURES, use_saas_prior=saas, use_combinatorial_kernel=combinatorial
    )
    actual, reference = MultiTaskConditionalKernel(**args), BtKernel(**args).double()
    align_kernels(actual, reference)
    x = full_inputs()
    check_kind(actual.map_task_to_binary(as_kind(x[:, 3].tolist(), kind)), kind)
    assert_close(
        actual.map_task_to_binary(as_kind(x[:, 3].tolist(), kind)),
        reference.map_task_to_binary(x[:, 3]),
    )
    actual_value = actual(as_kind(x.tolist(), kind))
    check_kind(actual_value, kind)
    assert_close(actual_value, reference(x).to_dense().detach(), atol=3e-12, rtol=3e-12)
    assert_close(
        actual(as_kind(x.tolist(), kind), as_kind(x.flip(0).tolist(), kind), diag=True),
        reference(x, x.flip(0), diag=True).detach(),
        atol=3e-12,
        rtol=3e-12,
    )
    assert np.linalg.eigvalsh(np.asarray(actual_value)).min() > -1e-10


@pytest.mark.parametrize("ard", [None, 1, 3])
def test_combinatorial_covariance_matches_official(ard):
    actual, reference = CombinatorialCovarModule(ard), BtCombinatorial(ard).double()
    align_kernels(actual, reference)
    x = full_inputs()[:, : ard or 2].round()
    assert_close(actual(x), reference(x).to_dense().detach(), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("combinatorial", [False, True])
def test_conditional_native_input_adjoints_and_inactive_coordinates(combinatorial):
    args = dict(
        feature_indices=FEATURES, use_saas_prior=True, use_combinatorial_kernel=combinatorial
    )
    actual, reference = MultiTaskConditionalKernel(**args), BtKernel(**args).double()
    align_kernels(actual, reference)
    x, points = full_inputs().requires_grad_(), full_inputs().flip(0).clone().requires_grad_()
    values = KernelEvaluationMap(actual, points)(x)
    expected = reference(x, points).to_dense()
    ag = torch.autograd.grad(values.square().sum(), (x, points))
    eg = torch.autograd.grad(expected.square().sum(), (x, points))
    for a, e in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, e, rtol=3e-10, atol=3e-10)
    assert torch.equal(ag[0][:, 3], torch.zeros_like(ag[0][:, 3]))
    for task, active in enumerate(FEATURES):
        inactive = list(set(range(3)) - set(active))
        assert torch.equal(ag[0][task::3, inactive], torch.zeros_like(ag[0][task::3, inactive]))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fixed_noise", [False, True])
@pytest.mark.parametrize(
    "saas,combinatorial", [(False, False), (False, True), (True, False), (True, True)]
)
def test_heterogeneous_mll_parameter_gradient_and_posterior(kind, fixed_noise, saas, combinatorial):
    xs, ys = training(kind)
    bx, by = training()
    noise = [as_kind([[0.03]] * 3, kind) for _ in FEATURES] if fixed_noise else None
    bnoise = [torch.full_like(y, 0.03) for y in by] if fixed_noise else None
    args = dict(
        feature_indices=FEATURES,
        full_feature_dim=3,
        use_saas_prior=saas,
        use_combinatorial_kernel=combinatorial,
    )
    actual, reference = HeterogeneousMTGP(xs, ys, noise, **args), BtGP(bx, by, bnoise, **args)
    with torch.no_grad():
        assert [name for name, _ in actual.named_parameters()] == [
            name for name, _ in reference.named_parameters()
        ]
        for name, parameter in reference.named_parameters():
            values = torch.full_like(parameter, -2.0 if name.endswith("raw_tau") else 0.23)
            parameter.copy_(values)
            actual._set_raw_param(name, values)
    amll = ExactMarginalLogLikelihood(actual.likelihood, actual)
    bmll = BtMLL(reference.likelihood, reference)
    expected = bmll(
        reference(*reference.train_inputs), reference.train_targets, *reference.train_inputs
    )
    expected.backward()
    bg = torch.cat(
        [(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1) for p in bmll.parameters()]
    )
    value, gradient = amll.value_and_grad()
    assert_close(value, expected.detach(), rtol=3e-7, atol=3e-8)
    assert_close(gradient, bg, rtol=5e-7, atol=3e-8)
    query = torch.tensor([[0.25, 0.5, 0.0], [0.7, 0.2, 0.0]], dtype=torch.float64)
    for observed in (False, True):
        bp = reference.posterior(query, observation_noise=observed)
        ap = actual.posterior(as_kind(query.tolist(), kind), observation_noise=observed)
        check_kind(ap.mean, kind)
        assert_close(ap.mean, bp.mean.detach(), rtol=1e-10, atol=1e-10)
        assert_close(
            ap.distribution.covariance_matrix,
            bp.distribution.covariance_matrix.detach(),
            rtol=1e-10,
            atol=1e-10,
        )
    assert_close(actual.feature_imputation_values, reference.feature_imputation_values)
    clone = copy.deepcopy(actual)
    assert_close(
        clone.posterior(as_kind(query.tolist(), kind)).mean,
        actual.posterior(as_kind(query.tolist(), kind)).mean,
    )


def test_imputation_mapping_normalize_and_target_contract():
    xs, ys = training()
    args = dict(
        train_Xs=xs,
        train_Ys=ys,
        train_Yvars=None,
        feature_indices=FEATURES,
        full_feature_dim=3,
        use_saas_prior=False,
    )
    from botorch.models.transforms.input import Normalize as BtNormalize
    from botorch.models.transforms.outcome import Standardize as BtStandardize

    actual = HeterogeneousMTGP(
        **args, input_transform=Normalize(4, indices=[0, 1, 2]), outcome_transform=Standardize(1)
    )
    reference = BtGP(
        **args,
        input_transform=BtNormalize(4, indices=[0, 1, 2]),
        outcome_transform=BtStandardize(1),
    )
    for task, x in enumerate(xs):
        assert_close(actual.map_to_full_tensor(x, task), reference.map_to_full_tensor(x, task))
    query = torch.tensor([[[0.3, 0.4, 0], [0.6, 0.7, 0]]], dtype=torch.float64)
    assert shape(actual.posterior(query).mean) == (1, 2, 1)
    with pytest.raises(Exception, match="target task"):
        actual.posterior(torch.tensor([[0.3, 0.4, 1.0]], dtype=torch.float64))
    with pytest.raises(Exception, match="output_indices"):
        actual.posterior(query, output_indices=[1])
    with pytest.raises(ValueError, match="columns"):
        actual.posterior(query[..., :-1])
    x = xs[0].clone().requires_grad_()
    native = actual.map_to_full_tensor(x, 0)
    native.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_construct_inputs_uses_target_first_dataset_mapping():
    datasets = []
    xs, ys = training()
    for task in (1, 0, 2):
        datasets.append(
            SupervisedDataset(
                X=torch.cat([xs[task], torch.full((3, 1), task, dtype=torch.float64)], dim=-1),
                Y=ys[task],
                feature_names=[*[f"f{i}" for i in FEATURES[task]], "task"],
                outcome_names=[f"y{task}"],
            )
        )
    dataset = MultiTaskDataset(datasets, target_outcome_name="y0", task_feature_index=-1)
    args = HeterogeneousMTGP.construct_inputs(dataset, use_saas_prior=False)
    assert args["feature_indices"] == FEATURES
    assert args["all_tasks"] == [0, 1, 2]
    model = HeterogeneousMTGP(**args)
    assert shape(model.posterior(torch.tensor([[0.3, 0.4, 0.0]], dtype=torch.float64)).mean) == (
        1,
        1,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_native_fit_improves_objective_for_each_array_form(kind):
    from botorch_rs.fit import fit_gpytorch_mll

    xs, ys = training(kind)
    model = HeterogeneousMTGP(
        xs,
        ys,
        [as_kind([[0.03]] * 3, kind) for _ in FEATURES],
        FEATURES,
        3,
        use_saas_prior=False,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = float(mll())
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 8}}, max_attempts=1)
    assert float(mll()) >= before - 1e-10
    check_kind(model.posterior(as_kind([[0.3, 0.4, 0.0]], kind)).mean, kind)


@pytest.mark.parametrize("kind", KINDS)
def test_acquisition_embedding_values_gradients_and_optimization(kind):
    from botorch.acquisition.analytic import ExpectedImprovement as BtEI

    from botorch_rs.acquisition.analytic import ExpectedImprovement
    from botorch_rs.optim import optimize_acqf

    # The target's declared feature order is not its order in full space.
    features = [[2, 0], [0, 1], [1, 2]]
    xs, ys = training(kind)
    bx, by = training()
    noise = [as_kind([[0.03]] * 3, kind) for _ in FEATURES]
    bnoise = [torch.full_like(y, 0.03) for y in by]
    args = dict(feature_indices=features, full_feature_dim=3, use_saas_prior=False)
    actual, reference = HeterogeneousMTGP(xs, ys, noise, **args), BtGP(bx, by, bnoise, **args)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            values = torch.full_like(parameter, -2.0 if name.endswith("raw_tau") else 0.23)
            parameter.copy_(values)
            actual._set_raw_param(name, values)
    acq = ExpectedImprovement(actual, best_f=0.1)
    expected = BtEI(reference, best_f=torch.tensor(0.1, dtype=torch.float64))
    query = torch.tensor(
        [[[0.25, 0.5, 0.0]], [[0.7, 0.2, 0.0]]], dtype=torch.float64, requires_grad=True
    )
    ev = expected(query)
    eg = torch.autograd.grad(ev.sum(), query)[0]
    av, ag = acq.value_and_grad(as_kind(query.detach().tolist(), kind))
    assert_close(av, ev.detach(), rtol=1e-10, atol=1e-11)
    assert_close(ag, eg, rtol=1e-9, atol=1e-10)
    # This must optimize in target-feature coordinates with a fixed task column.
    candidate, value = optimize_acqf(
        acq,
        bounds=as_kind([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], kind),
        q=1,
        num_restarts=3,
        raw_samples=16,
        options={"maxiter": 10, "seed": 24},
    )
    check_kind(candidate, kind)
    assert to_task(candidate) == 0.0
    assert_close(value, acq(candidate), rtol=1e-12)


def to_task(value):
    return float(np.asarray(value)[..., -1].reshape(-1)[0])


def test_target_conditioning_and_fantasies_embed_inputs():
    from botorch_rs.sampling import SobolQMCNormalSampler

    xs, ys = training()
    model = HeterogeneousMTGP(xs, ys, None, FEATURES, 3, use_saas_prior=False)
    x = torch.tensor([[0.3, 0.4, 0.0]], dtype=torch.float64)
    model.posterior(x)
    conditioned = model.condition_on_observations(x, torch.tensor([[0.9]], dtype=torch.float64))
    assert isinstance(conditioned, HeterogeneousMTGP)
    assert shape(conditioned.posterior(x).mean) == (1, 1)
    assert abs(float(conditioned.posterior(x).mean[0, 0]) - 0.9) < abs(
        float(model.posterior(x).mean[0, 0]) - 0.9
    )
    fantasy = model.fantasize(x, SobolQMCNormalSampler((2,), seed=18))
    assert isinstance(fantasy, HeterogeneousMTGP)
    assert shape(fantasy.posterior(x).mean) == (2, 1, 1)


def test_conditioning_matches_official_with_explicit_full_input_bug_workaround():
    xs, ys = training()
    args = dict(feature_indices=FEATURES, full_feature_dim=3, use_saas_prior=False)
    actual, reference = HeterogeneousMTGP(xs, ys, None, **args), BtGP(xs, ys, None, **args)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            values = torch.full_like(parameter, -2.0 if name.endswith("raw_tau") else 0.23)
            parameter.copy_(values)
            actual._set_raw_param(name, values)
    x = torch.tensor([[0.3, 0.4, 0.0]], dtype=torch.float64)
    y = torch.tensor([[0.9]], dtype=torch.float64)
    actual.posterior(x)
    reference.posterior(x)
    # b0dd's inherited conditioning bypasses the heterogeneous input embedding.
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        reference.condition_on_observations(x, y)
    full = reference.map_to_full_tensor(x[:, :-1], 0)
    expected = reference.condition_on_observations(full, y).posterior(x)
    result = actual.condition_on_observations(x, y).posterior(x)
    assert_close(result.mean, expected.mean.detach(), rtol=1e-10, atol=1e-10)
    assert_close(
        result.distribution.covariance_matrix,
        expected.distribution.covariance_matrix.detach(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_unequal_row_counts_shared_ard_group_and_permuted_target_features():
    features = [[2, 0, 1], [0, 1, 3]]
    generator = torch.Generator().manual_seed(183)
    xs = [torch.rand(n, 3, generator=generator, dtype=torch.float64) for n in (5, 3)]
    ys = [torch.rand(n, 1, generator=generator, dtype=torch.float64) for n in (5, 3)]
    args = dict(feature_indices=features, full_feature_dim=4)
    actual, reference = HeterogeneousMTGP(xs, ys, None, **args), BtGP(xs, ys, None, **args)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            values = torch.full_like(parameter, -2.0 if name.endswith("raw_tau") else 0.23)
            parameter.copy_(values)
            actual._set_raw_param(name, values)
    native_mll = ExactMarginalLogLikelihood(actual.likelihood, actual)
    official_mll = BtMLL(reference.likelihood, reference)
    value = official_mll(
        reference(*reference.train_inputs), reference.train_targets, *reference.train_inputs
    )
    value.backward()
    expected = torch.cat(
        [
            (torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
            for p in official_mll.parameters()
        ]
    )
    result, gradient = native_mll.value_and_grad()
    assert_close(result, value.detach(), rtol=5e-7, atol=3e-8)
    assert_close(gradient, expected, rtol=5e-7, atol=3e-8)
    query = torch.tensor([[0.25, 0.5, 0.3, 0.0], [0.7, 0.2, 0.8, 0.0]], dtype=torch.float64)
    assert_close(actual.posterior(query).mean, reference.posterior(query).mean.detach(), rtol=1e-10)


def test_target_task_without_observations_and_explicit_transform_boundary():
    xs, ys = training()
    xs[0], ys[0] = torch.empty(0, 2, dtype=torch.float64), torch.empty(0, 1, dtype=torch.float64)
    model = HeterogeneousMTGP(xs, ys, None, FEATURES, 3, use_saas_prior=False)
    value = model.posterior(torch.tensor([[0.3, 0.4, 0.0]], dtype=torch.float64)).mean
    assert torch.isfinite(value).all()
    assert 0 in model.all_tasks
    with pytest.raises(Exception, match="leave the task column unchanged"):
        HeterogeneousMTGP(xs, ys, None, FEATURES, 3, input_transform=Normalize(4))
