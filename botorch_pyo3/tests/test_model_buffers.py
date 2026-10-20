"""Real registered-buffer parity with official BoTorch (including mixed dtypes)."""

from __future__ import annotations

import subprocess
import sys
from contextlib import nullcontext

import pytest
import torch
from botorch.models.cost import AffineFidelityCostModel as RefAffineCost
from botorch.models.cost import FixedCostModel as RefFixedCost
from botorch.models.deterministic import AffineDeterministicModel as RefAffine
from botorch.models.deterministic import FixedSingleSampleModel as RefFixedSample
from botorch.models.deterministic import GenericDeterministicModel as RefGeneric
from botorch.models.deterministic import PosteriorMeanModel as RefPosteriorMean
from botorch.models.ensemble import EnsembleModel as RefEnsemble
from botorch.models.gp_regression import SingleTaskGP as RefGP
from botorch.models.model import Model as RefModel
from botorch.models.model import ModelList as RefModelList
from botorch.models.pairwise_gp import PairwiseGP as RefPairwise
from botorch.models.transforms import input as ref_input
from botorch.models.transforms import outcome as ref_outcome
from gpytorch import constraints as ref_constraints
from gpytorch import kernels as ref_kernels
from gpytorch import priors as ref_priors

from botorch_rs import constraints, kernels, priors
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.cost import AffineFidelityCostModel, FixedCostModel
from botorch_rs.models.deterministic import (
    AffineDeterministicModel,
    FixedSingleSampleModel,
    GenericDeterministicModel,
    PosteriorMeanModel,
)
from botorch_rs.models.ensemble import EnsembleModel
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model import Model, ModelList
from botorch_rs.models.pairwise_gp import PairwiseGP
from botorch_rs.models.transforms import input as port_input
from botorch_rs.models.transforms import outcome as port_outcome
from botorch_rs.models.utils.priors import BetaPrior


def assert_buffers(actual, expected, *, values=True):
    a, b = dict(actual.named_buffers()), dict(expected.named_buffers())
    assert a.keys() == b.keys()
    assert {key: value.dtype for key, value in a.items()} == {
        key: value.dtype for key, value in b.items()
    }
    if values:
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=2e-7, rtol=2e-6, equal_nan=True)
    if isinstance(actual, Model):
        assert actual.dtypes_of_buffers == expected.dtypes_of_buffers


def data(dtype=torch.double):
    return (
        torch.tensor([[0.0, 0.1], [0.25, 0.4], [0.7, 0.6], [1.0, 0.9]], dtype=dtype),
        torch.tensor([[0.0], [0.5], [1.0], [0.0]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_exact_gp_native_buffers(dtype, fixed, standardize):
    x, y = data(dtype)
    kw = {} if standardize else {"outcome_transform": None}
    if fixed:
        kw["train_Yvar"] = torch.full_like(y, 0.1)
    with pytest.warns(Warning) if dtype == torch.float else nullcontext():
        actual, expected = SingleTaskGP(x, y, **kw), RefGP(x, y, **kw)
    assert_buffers(actual, expected)
    names = dict(actual.named_buffers())
    assert not any("train_inputs" in name or "train_targets" in name for name in names)
    if fixed:
        assert not any(name.startswith("likelihood.") for name in names)
    assert actual.dtypes_of_buffers == ({dtype, torch.bool} if standardize else {dtype})


@pytest.mark.parametrize("kind", ["normalize", "warp", "chain"])
def test_input_transform_native_buffers(kind):
    x, y = data()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    def make(module):
        if kind == "normalize":
            return module.Normalize(2, indices=[0], bounds=bounds)
        if kind == "warp":
            return module.Warp(2, indices=[0, 1], bounds=bounds)
        return module.ChainedInputTransform(
            normalize=module.Normalize(2, indices=[0], bounds=bounds),
            round=module.Round(integer_indices=[1]),
        )

    actual = SingleTaskGP(x, y, input_transform=make(port_input))
    expected = RefGP(x, y, input_transform=make(ref_input))
    assert_buffers(actual, expected)
    assert actual.dtypes_of_buffers == {torch.double, torch.long, torch.bool}


def test_kernel_priors_constraints_and_active_dimensions():
    x, y = data()

    def make(k, p, c):
        return k.ScaleKernel(
            k.RBFKernel(
                ard_num_dims=1,
                active_dims=[1],
                lengthscale_prior=p.NormalPrior(torch.tensor(1.0, dtype=torch.double), 0.5),
                lengthscale_constraint=c.Interval(0.01, 3.0),
            ),
            outputscale_prior=p.GammaPrior(2.0, 1.0),
        )

    actual = SingleTaskGP(x, y, covar_module=make(kernels, priors, constraints))
    expected = RefGP(x, y, covar_module=make(ref_kernels, ref_priors, ref_constraints))
    assert_buffers(actual, expected)
    assert not any("raw_lengthscale" == key for key, _ in actual.covar_module.named_buffers())


def test_shared_prior_tracks_actual_module_casts_not_parent_training_dtype():
    p, ref_p = priors.NormalPrior(1.0, 0.5), ref_priors.NormalPrior(1.0, 0.5)
    x, y = data()
    first = SingleTaskGP(x, y, covar_module=kernels.RBFKernel(lengthscale_prior=p))
    ref_first = RefGP(x, y, covar_module=ref_kernels.RBFKernel(lengthscale_prior=ref_p))
    SingleTaskGP(x.float(), y.float(), covar_module=kernels.RBFKernel(lengthscale_prior=p))
    RefGP(x.float(), y.float(), covar_module=ref_kernels.RBFKernel(lengthscale_prior=ref_p))
    assert_buffers(first, ref_first)
    assert first.dtypes_of_buffers == {torch.float, torch.double, torch.bool}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numpy_gp_logical_buffers(dtype):
    x, y = data()
    xa, ya = x.numpy().astype(dtype), y.numpy().astype(dtype)
    actual = SingleTaskGP(xa, ya)
    expected = RefGP(torch.from_numpy(xa), torch.from_numpy(ya))
    assert_buffers(actual, expected)


def test_native_snapshot_changes_do_not_mutate_rust_state():
    x, y = data()
    model = SingleTaskGP(x, y)
    original = model.outcome_transform.means.clone()
    snapshot = dict(model.named_buffers())["outcome_transform.means"]
    snapshot.fill_(100.0)
    torch.testing.assert_close(model.outcome_transform.means, original)


@pytest.mark.parametrize(
    "name,args",
    [
        ("NormalPrior", (torch.tensor(1.0, dtype=torch.double), torch.tensor(2.0))),
        ("LogNormalPrior", (torch.tensor(1.0, dtype=torch.double), torch.tensor(2.0))),
        ("GammaPrior", (2.0, 3.0)),
        ("HalfCauchyPrior", (2.0,)),
        ("HalfNormalPrior", (2.0,)),
        ("UniformPrior", (1.0, 2.0)),
        ("SmoothedBoxPrior", (1.0, 2.0)),
        ("LKJCholeskyFactorPrior", (2, 1.0)),
    ],
)
def test_standalone_prior_buffers(name, args):
    assert_buffers(getattr(priors, name)(*args), getattr(ref_priors, name)(*args))


def test_beta_expanded_buffer_shape_and_promoted_dtype():
    from botorch.models.utils.priors import BetaPrior as RefBeta

    args = torch.tensor(2.0), torch.tensor(3.0, dtype=torch.double)
    assert_buffers(BetaPrior(*args), RefBeta(*args))
    assert_buffers(BetaPrior(*args).expand((2, 3)), RefBeta(*args).expand((2, 3)))
    assert BetaPrior(*args).concentration1.dtype == RefBeta(*args).concentration1.dtype


def test_standalone_transform_buffers_before_and_after_fitting():
    actual, expected = port_outcome.Standardize(1), ref_outcome.Standardize(1)
    assert_buffers(actual, expected)
    _, y = data()
    actual(y)
    expected(y)
    assert_buffers(actual, expected)
    for name, args in [("Round", {"integer_indices": [0]}), ("Log10", {"indices": [1]})]:
        assert_buffers(getattr(port_input, name)(**args), getattr(ref_input, name)(**args))


@pytest.mark.parametrize("b", [torch.tensor([1.0, 2.0]), torch.tensor([1, 2]), 0.1])
def test_affine_mixed_buffer_dtype_and_subsetting(b):
    a = torch.tensor([[1.0, 2.0], [0.2, 0.3]], dtype=torch.double)
    actual, expected = AffineDeterministicModel(a, b), RefAffine(a, b)
    assert_buffers(actual, expected)
    assert_buffers(actual.subset_output([1]), expected.subset_output([1]))


def test_default_dtype_is_captured_when_buffers_are_created():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.double)
        actual = AffineFidelityCostModel({0: 0.3})
        expected = RefAffineCost({0: 0.3})
        prior, reference_prior = priors.NormalPrior(1.0, 2.0), ref_priors.NormalPrior(1.0, 2.0)
        torch.set_default_dtype(torch.float)
        assert_buffers(actual, expected)
        assert_buffers(prior, reference_prior)
        assert actual.weights.dtype == expected.weights.dtype == torch.double
    finally:
        torch.set_default_dtype(previous)


def test_cost_buffers():
    assert_buffers(AffineFidelityCostModel({0: 0.5}), RefAffineCost({0: 0.5}))
    fixed = torch.tensor([2.0], dtype=torch.double)
    assert_buffers(FixedCostModel(fixed), RefFixedCost(fixed))


def test_custom_registered_and_torch_child_buffers_exclude_parameters_and_data():
    class Child(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("double_value", torch.tensor(1.0, dtype=torch.double))
            self.register_buffer("flag", torch.tensor(True))
            self.p = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x

    def initialize(module):
        module.register_buffer("single", torch.tensor(2.0))
        module.register_buffer("integer", torch.tensor(3))
        module.register_buffer("none", None)
        module.register_buffer("temporary", torch.tensor(False), persistent=False)
        module.child = Child()
        module.train_inputs = (torch.ones(2, dtype=torch.float16),)
        module.train_targets = torch.ones(2, dtype=torch.float16)
        return module

    class ReferenceModel(RefModel):
        def posterior(self, X, **kwargs):
            raise NotImplementedError

    actual, expected = initialize(Model()), initialize(ReferenceModel())
    assert_buffers(actual, expected)
    assert actual.dtypes_of_buffers == {torch.float, torch.double, torch.long, torch.bool}
    assert list(actual.buffers(recurse=False)) == list(expected.buffers(recurse=False))
    assert (
        dict(actual.named_buffers(prefix="root")).keys()
        == dict(expected.named_buffers(prefix="root")).keys()
    )
    actual.unregistered_list = [Child()]
    assert_buffers(actual, expected)


def test_custom_torch_model_uses_real_torch_registration():
    class TorchModel(Model, torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.register_buffer("flag", torch.tensor(True))

    module = TorchModel()
    assert module.dtypes_of_buffers == {torch.bool}
    assert dict(module.named_buffers())["flag"] is module.flag


def test_generic_callable_modules_and_ordinary_tensor_attributes():
    f = torch.nn.BatchNorm1d(2).double()
    assert_buffers(GenericDeterministicModel(f), RefGeneric(f))
    assert GenericDeterministicModel(lambda x: x).dtypes_of_buffers == set()

    class PortEnsemble(EnsembleModel):
        _num_outputs = 1

    class ReferenceEnsemble(RefEnsemble):
        @property
        def num_outputs(self):
            return 1

        def forward(self, X):
            return X

    actual, expected = (
        PortEnsemble(torch.ones(2, dtype=torch.double)),
        ReferenceEnsemble(torch.ones(2, dtype=torch.double)),
    )
    assert_buffers(actual, expected)
    assert actual.dtypes_of_buffers == set()


def test_model_containers_wrappers_and_shared_child_deduplication():
    x, y = data()
    gp, ref_gp = SingleTaskGP(x, y), RefGP(x, y)
    assert_buffers(PosteriorMeanModel(gp), RefPosteriorMean(ref_gp))
    w = torch.tensor([1.0], dtype=torch.float32)
    assert_buffers(FixedSingleSampleModel(gp, w=w), RefFixedSample(ref_gp, w=w))
    actual, expected = ModelList(gp, gp), RefModelList(ref_gp, ref_gp)
    assert_buffers(actual, expected)
    assert list(actual.buffers(recurse=False)) == []
    assert len(list(actual.named_buffers(remove_duplicate=False))) == len(
        list(expected.named_buffers(remove_duplicate=False))
    )
    flag = torch.tensor(True)
    custom = Model()
    custom.register_buffer("a", flag)
    custom.register_buffer("b", flag)
    assert [key for key, _ in custom.named_buffers()] == ["a"]
    assert [key for key, _ in custom.named_buffers(remove_duplicate=False)] == ["a", "b"]


def test_pairwise_registered_data_and_laplace_buffers():
    x, _ = data()
    comparisons = torch.tensor([[1, 0], [2, 1], [2, 3]])
    actual, expected = PairwiseGP(x, comparisons), RefPairwise(x, comparisons)
    assert_buffers(actual, expected, values=False)
    assert actual.dtypes_of_buffers == {torch.double, torch.long}
    buffers = dict(actual.named_buffers())
    torch.testing.assert_close(buffers["consolidated_datapoints"], x)
    torch.testing.assert_close(buffers["consolidated_comparisons"], comparisons)


def test_unsupported_native_caches_are_explicit():
    x, y = data()
    transform = port_input.InputPerturbation(torch.zeros(2, 2))
    model = SingleTaskGP(x, y, input_transform=transform)
    with pytest.raises(UnsupportedError, match="cached perturbations"):
        _ = model.dtypes_of_buffers


def test_import_and_list_construction_do_not_import_torch():
    code = """
import sys
from botorch_rs.models import SingleTaskGP
model = SingleTaskGP([[0.0], [0.5], [1.0]], [[0.0], [1.0], [0.0]])
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
