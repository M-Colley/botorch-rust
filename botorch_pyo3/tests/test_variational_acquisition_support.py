"""SVGP query gradients use the marginal law without claiming parameter VJPs."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from approximate_helpers import (  # noqa: E402
    Case,
    build_pair,
    fixed_allocator,
    set_random_parameters,
)
from botorch.acquisition import analytic as bt_an  # noqa: E402
from botorch.acquisition import logei as bt_log  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtScalarized  # noqa: E402
from kernel_tree_helpers import zero_diagonal_distances  # noqa: E402

from botorch_rs.acquisition import analytic as rs_an  # noqa: E402
from botorch_rs.acquisition import logei as rs_log  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsScalarized,
)
from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from parity_helpers import sampler_pair  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
DT = torch.float64
ANALYTIC = [
    "ExpectedImprovement",
    "LogExpectedImprovement",
    "ProbabilityOfImprovement",
    "LogProbabilityOfImprovement",
    "UpperConfidenceBound",
    "PosteriorMean",
    "PosteriorStandardDeviation",
]
MC = [
    "qExpectedImprovement",
    "qLogExpectedImprovement",
    "qProbabilityOfImprovement",
    "qSimpleRegret",
    "qUpperConfidenceBound",
    "qLowerConfidenceBound",
    "qPosteriorStandardDeviation",
    "qNoisyExpectedImprovement",
    "qLogNoisyExpectedImprovement",
]


class CountedNative:
    def __init__(self, native):
        self.native = native
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def forward(self, *args):
        self.calls.append("forward")
        return self.native.forward(*args)

    def forward_with_grad(self, *args):
        self.calls.append("query")
        return self.native.forward_with_grad(*args)

    def forward_with_adjoint_tape(self, *args):
        self.calls.append("tape")
        return self.native.forward_with_adjoint_tape(*args)


def models(outputs, bnn):
    kernel = (
        {
            "type": "ScaleKernel",
            "args": [
                {
                    "__kernel__": {
                        "type": "InfiniteWidthBNNKernel",
                        "args": [],
                        "kwargs": {"depth": 3},
                    }
                }
            ],
            "kwargs": {},
        }
        if bnn
        else None
    )
    case = Case("query_support", n=8, d=2, m=outputs, inducing=3, kernel=kernel, seed=71)
    with fixed_allocator():
        reference, model, x, _ = build_pair(case)
    set_random_parameters(reference, model, 23)
    reference.eval()
    model.eval()
    return reference, model, x


def acquisitions(name, outputs, bnn):
    reference, model, x = models(outputs, bnn)
    weight = torch.tensor([0.4, 0.6], dtype=DT)
    transforms = (BtScalarized(weight), RsScalarized(weight)) if outputs > 1 else (None, None)
    best = torch.tensor(0.25, dtype=DT)
    q = 2 if name in MC else 1
    samplers = sampler_pair(16, q + (3 if "Noisy" in name else 0), 1, 17)[:2]
    result = []
    for native, current, transform, sampler in zip(
        (False, True), (reference, model), transforms, samplers, strict=True
    ):
        kw = {"model": current, "posterior_transform": transform}
        if "Improvement" in name and "Noisy" not in name:
            kw["best_f"] = best
        if "ConfidenceBound" in name:
            kw["beta"] = torch.tensor(0.7, dtype=DT)
        if name in MC:
            kw["sampler"] = sampler
            if name == "qProbabilityOfImprovement":
                kw["tau"] = torch.tensor(0.15, dtype=DT)
            if "Noisy" in name:
                kw.update(X_baseline=x[:3], prune_baseline=False, cache_root=False)
            module = (
                (rs_log if native else bt_log)
                if name.startswith("qLog")
                else (rs_mc if native else bt_mc)
            )
        else:
            module = rs_an if native else bt_an
        result.append(getattr(module, name)(**kw))
    query = torch.tensor([[[0.25, 0.4], [0.65, 0.8]], [[0.4, 0.3], [0.8, 0.6]]], dtype=DT)[:, :q]
    return *result, model, query


def check_query_and_parameter_boundary(reference, acquisition, model, query):
    counted = CountedNative(acquisition._rs)
    acquisition._rs = counted
    x = query.clone().requires_grad_()
    rx = query.clone().requires_grad_()
    with zero_diagonal_distances():
        expected, actual = reference(rx), acquisition(x)
        want = torch.autograd.grad(expected.sum(), rx)[0]
        derivative = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
        values, gradient = acquisition.value_and_grad(query)
    assert counted.calls == ["query", "query"]
    assert torch.equal(values, actual.detach())
    assert torch.equal(gradient, derivative)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(derivative, want, rtol=1e-8, atol=1e-9)
    # ELBO leaves do not imply that the native posterior has parameter VJPs.
    with pytest.raises(UnsupportedError, match="parameter"):
        torch.autograd.grad(actual.sum(), tuple(model.parameters()))


@pytest.mark.parametrize("name", ANALYTIC + MC)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("bnn", [False, True], ids=["rbf", "bnn"])
def test_svgp_acquisitions_preserve_official_query_law_and_explicit_parameter_boundary(
    name, outputs, bnn
):
    reference, acquisition, model, query = acquisitions(name, outputs, bnn)
    check_query_and_parameter_boundary(reference, acquisition, model, query)


@pytest.mark.parametrize("seed", [1, 19, 23, 24])
def test_bnn_randomized_svgp_regressions_keep_original_values_and_query_gradients(seed):
    from test_equivalence_10c_approximate import test_sweep

    test_sweep(seed)
