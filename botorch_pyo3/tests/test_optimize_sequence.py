"""Sequential acquisition sequences use the actual native optimizer per step."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction as ReferenceAcquisition
from botorch.acquisition.monte_carlo import qExpectedImprovement as RefEI
from botorch.acquisition.monte_carlo import qSimpleRegret as RefSR
from botorch.exceptions.errors import UnsupportedError as RefUnsupported
from botorch.models import SingleTaskGP as RefGP
from botorch.models.deterministic import PosteriorMeanModel as RefPM
from botorch.optim.optimize import optimize_acqf as reference_optimize
from botorch.sampling.stochastic_samplers import StochasticSampler as RefSampler

from botorch_rs.acquisition.monte_carlo import qExpectedImprovement, qSimpleRegret
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.deterministic import PosteriorMeanModel
from botorch_rs.optim.optimize import optimize_acqf
from botorch_rs.sampling.stochastic_samplers import StochasticSampler


def native_sequence(reference=False):
    gp, pm = (RefGP, RefPM) if reference else (SingleTaskGP, PosteriorMeanModel)
    sampler = (RefSampler if reference else StochasticSampler)(torch.Size([1]))
    x = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    models = [pm(gp(x, 1 - 5 * (x - location) ** 2)) for location in (0.25, 0.8)]
    if reference:
        sequence = [
            RefSR(models[0], sampler=sampler),
            RefEI(models[1], best_f=0.0, sampler=sampler),
        ]
    else:
        sequence = [
            qSimpleRegret(models[0], sampler=sampler),
            qExpectedImprovement(models[1], best_f=0.0, sampler=sampler),
        ]
    sequence[1].set_X_pending(torch.tensor([[0.05]], dtype=torch.double))
    return sequence


def conditions(**kwargs):
    assert kwargs["q"] == 1
    return torch.tensor([[[0.15]], [[0.55]], [[0.9]]], dtype=torch.double)


def test_distinct_native_acquisitions_and_per_step_initialization():
    actual, reference = native_sequence(), native_sequence(True)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    histories = [[], []]

    def initializer(index):
        def generate(**kwargs):
            assert kwargs.pop("token") == 17
            acq = kwargs["acq_function"]
            histories[index].append(None if acq.X_pending is None else acq.X_pending.clone())
            return conditions(**kwargs)

        return generate

    results = []
    for index, (optimizer, seq) in enumerate(
        ((optimize_acqf, actual), (reference_optimize, reference))
    ):
        results.append(
            optimizer(
                seq[0],
                bounds,
                q=2,
                num_restarts=3,
                raw_samples=8,
                sequential=True,
                acq_function_sequence=seq,
                ic_generator=initializer(index),
                token=17,
                options={"maxiter": 80, "pgtol": 1e-9},
            )
        )
    torch.testing.assert_close(results[0][0], results[1][0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(results[0][1], results[1][1], atol=1e-8, rtol=1e-8)
    assert results[0][0].shape == (2, 1) and results[0][1].shape == (2,)
    assert histories[0][0] is histories[1][0] is None
    torch.testing.assert_close(histories[0][1], histories[1][1], atol=2e-6, rtol=2e-6)
    assert histories[0][1].shape == (2, 1)
    assert actual[0].X_pending is None
    torch.testing.assert_close(actual[1].X_pending, torch.tensor([[0.05]], dtype=torch.double))


class Quadratic(ReferenceAcquisition):
    def __init__(self, target, pending=None):
        super().__init__(model=None)
        self.target = torch.as_tensor(target, dtype=torch.double)
        self.set_X_pending(pending)

    def forward(self, X):
        target = self.target
        if self.X_pending is not None:
            target = target + 0.1 * self.X_pending.mean(dim=-2)
        return -(X - target).square().sum(dim=(-1, -2))


@pytest.mark.parametrize("kind", ["torch", "numpy", "list"])
@pytest.mark.parametrize("return_values", [True, False])
def test_sequence_constraints_fixed_features_postprocessing_and_array_types(kind, return_values):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    seq = [Quadratic([0.2, 0.5]), Quadratic([0.8, 0.3])]
    constraint = (torch.tensor([0]), torch.tensor([1.0], dtype=torch.double), 0.4)
    if kind == "numpy":
        bounds = bounds.numpy()
    elif kind == "list":
        bounds = bounds.tolist()

    def initialize(**kwargs):
        value = np.array([[[0.5, 0.3]], [[0.8, 0.3]]], dtype=np.float64)
        return (
            torch.from_numpy(value)
            if kind == "torch"
            else value
            if kind == "numpy"
            else value.tolist()
        )

    def postprocess(value):
        value[..., 0] = 0.45 if isinstance(value, torch.Tensor) else 0.45
        return value

    # Lists use the same actual callback semantics without tensor assignment.
    process = (
        (lambda value: [[[0.45, row[0][1]]] for row in value]) if kind == "list" else postprocess
    )
    candidates, values = optimize_acqf(
        seq[0],
        bounds,
        2,
        2,
        raw_samples=4,
        sequential=True,
        acq_function_sequence=seq,
        ic_generator=initialize,
        inequality_constraints=[constraint],
        fixed_features={1: 0.3},
        post_processing_func=process,
        return_acq_values=return_values,
        options={"maxiter": 40, "batch_limit": 1},
    )
    np.testing.assert_allclose(np.asarray(candidates), [[0.45, 0.3], [0.45, 0.3]])
    assert seq[0].X_pending is seq[1].X_pending is None
    if return_values:
        # The second value includes the first postprocessed candidate as pending.
        expected = [
            -((0.45 - 0.2) ** 2) - (0.3 - 0.5) ** 2,
            -((0.45 - 0.845) ** 2) - (0.3 - 0.33) ** 2,
        ]
        np.testing.assert_allclose(np.asarray(values), expected, atol=1e-12)
    else:
        assert values is None


def test_pending_restored_on_error_and_reference_bug_is_reproduced():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    for optimizer in (optimize_acqf, reference_optimize):
        pending = torch.tensor([[0.05]], dtype=torch.double)
        seq = [Quadratic([0.2]), Quadratic([0.8], pending)]
        calls = 0

        def initialize(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("initializer failed on second step")
            return conditions(**kwargs)

        with pytest.raises(RuntimeError, match="initializer failed"):
            optimizer(
                seq[0],
                bounds,
                2,
                3,
                raw_samples=8,
                sequential=True,
                acq_function_sequence=seq,
                ic_generator=initialize,
            )
        if optimizer is optimize_acqf:
            torch.testing.assert_close(seq[1].X_pending, pending)
        else:
            # b0dd01be restores pending only after the entire loop succeeds.
            assert seq[1].X_pending.shape == (2, 1)


def test_none_primary_acquisition_is_supported_and_reference_bug_is_reproduced():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    seq = [Quadratic([0.2]), Quadratic([0.8])]
    candidates, values = optimize_acqf(
        None,
        bounds,
        2,
        3,
        raw_samples=8,
        sequential=True,
        acq_function_sequence=seq,
        ic_generator=conditions,
    )
    torch.testing.assert_close(candidates, torch.tensor([[0.2], [0.82]], dtype=torch.double))
    torch.testing.assert_close(values, torch.zeros(2, dtype=torch.double), atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="Either `acq_function`"):
        reference_optimize(
            None,
            bounds,
            2,
            3,
            raw_samples=8,
            sequential=True,
            acq_function_sequence=seq,
            ic_generator=conditions,
        )


def test_all_fixed_features_evaluate_the_actual_acquisition_of_each_step():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    seq = [Quadratic([0.2]), Quadratic([0.8])]
    candidates, values = optimize_acqf(
        seq[0],
        bounds,
        2,
        3,
        raw_samples=8,
        sequential=True,
        acq_function_sequence=seq,
        fixed_features={0: 0.5},
    )
    torch.testing.assert_close(candidates, torch.full((2, 1), 0.5, dtype=torch.double))
    torch.testing.assert_close(values, torch.tensor([-0.09, -0.1225], dtype=torch.double))
    _, ref_values = reference_optimize(
        seq[0],
        bounds,
        2,
        3,
        raw_samples=8,
        sequential=True,
        acq_function_sequence=seq,
        fixed_features={0: 0.5},
    )
    # Official takes the all-fixed shortcut before sequence handling, evaluates
    # only the primary acquisition on the joint batch, and returns a scalar.
    assert ref_values.ndim == 0
    torch.testing.assert_close(ref_values, torch.tensor(-0.18, dtype=torch.double))


def test_custom_candidate_generator_receives_per_step_acquisition_and_timeout():
    from botorch_rs.generation.gen import gen_candidates_scipy

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    seq = [Quadratic([0.2]), Quadratic([0.8])]
    calls = []

    def generate(initial_conditions, acq, **kwargs):
        calls.append((acq, kwargs["timeout_sec"]))
        return gen_candidates_scipy(initial_conditions, acq, **kwargs)

    candidates, _ = optimize_acqf(
        seq[0],
        bounds,
        2,
        3,
        raw_samples=8,
        sequential=True,
        acq_function_sequence=seq,
        ic_generator=conditions,
        gen_candidates=generate,
        timeout_sec=120.0,
        options={"batch_limit": 3},
    )
    assert [call[0] for call in calls] == seq
    assert [call[1] for call in calls] == [60.0, 60.0]
    torch.testing.assert_close(candidates, torch.tensor([[0.2], [0.82]], dtype=torch.double))


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"sequential": False}, "requires sequential"),
        ({"q": 3}, "length q"),
        ({"q": 1, "acq_function_sequence": [Quadratic([0.2])]}, "q > 1"),
        ({"batch_initial_conditions": torch.zeros(2, 1, 1)}, "batch_initial_conditions"),
        ({"return_best_only": False}, "only supported for joint"),
        ({"inequality_constraints": [(torch.tensor([[0, 0]]), torch.ones(1), 0.0)]}, "Inter-point"),
        (
            {
                "nonlinear_inequality_constraints": [(lambda X: X.sum(), False)],
                "ic_generator": conditions,
            },
            "Inter-point",
        ),
    ],
)
def test_sequence_rejects_unsupported_or_invalid_configurations(kw, match):
    base = dict(
        acq_function=None,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        num_restarts=2,
        raw_samples=4,
        sequential=True,
        acq_function_sequence=[Quadratic([0.2]), Quadratic([0.8])],
    )
    base.update(kw)
    with pytest.raises((ValueError, UnsupportedError, NotImplementedError), match=match):
        optimize_acqf(**base)
    with pytest.raises((ValueError, RefUnsupported, NotImplementedError), match=match):
        reference_optimize(**base)
