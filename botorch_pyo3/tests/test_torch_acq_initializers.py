# ruff: noqa: E402
"""Reference restart laws, device-local RNG, and optimizer option correctness."""

import pytest

pytest.importorskip("torch")


import warnings

import torch
from test_torch_acq_optimization import Quadratic

from botorch_rs.exceptions.errors import CandidateGenerationError, UnsupportedError
from botorch_rs.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch_rs.optim import _torch_acq_initializers as port
from botorch_rs.optim import optimize_acqf

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kind,values,options",
    [
        ("initialize_q_batch", [-2.0, -1.2, -0.6, 0.1, 0.4, 1.7, 0.9, 0.2], {"eta": 1.7}),
        ("initialize_q_batch", [3.0] * 8, {}),
        ("initialize_q_batch", [float("-inf")] * 8, {}),
        (
            "initialize_q_batch_nonneg",
            [-1.0, 0.0, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0],
            {"eta": 2.0, "alpha": 0.95},
        ),
        ("initialize_q_batch_nonneg", [-1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0], {}),
        ("initialize_q_batch_nonneg", [-1.0] * 8, {}),
        (
            "initialize_q_batch_topn",
            [-2.0, -1.2, -0.6, 0.1, 0.4, 1.7, 0.9, 0.2],
            {"largest": False, "sorted": False},
        ),
        ("initialize_q_batch_topn", [3.0] * 8, {}),
    ],
)
def test_restart_selection_matches_unmodified_reference_rng(device, kind, values, options):
    pytest.importorskip("botorch")
    from botorch.optim import initializers as reference

    X = torch.arange(32, dtype=torch.double, device=device).reshape(8, 2, 2)
    scores = X.new_tensor(values)
    torch.manual_seed(13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = getattr(reference, kind)(X, scores, 4, **options)
        generator = torch.Generator(device=device).manual_seed(13)
        actual = getattr(port, kind)(X, scores, 4, generator=generator, **options)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert actual[0].device == X.device


@pytest.mark.parametrize("device", DEVICES)
def test_automatic_initialization_uses_boltzmann_seed_and_chunks_without_host_data(
    device, monkeypatch
):
    pytest.importorskip("botorch")
    from botorch.optim.initializers import initialize_q_batch

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    engine = torch.quasirandom.SobolEngine(4, scramble=True, seed=17)
    raw = engine.draw(32, dtype=bounds.dtype).to(device).reshape(32, 2, 2)
    torch.manual_seed(17)
    expected = initialize_q_batch(raw, Quadratic()(raw), 4, eta=1.7)[0]

    class RecordingQuadratic(Quadratic):
        def __init__(self):
            super().__init__()
            self.sizes = []

        def forward(self, X):
            self.sizes.append(X.shape[0])
            return super().forward(X)

    acq = RecordingQuadratic()

    def forbidden(*args, **kwargs):
        raise AssertionError("Initializer copied floating-point data to the host")

    for name in ("cpu", "numpy", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)

    def retain(initial_conditions, acquisition_function, **kwargs):
        assert kwargs["options"] == {}
        return initial_conditions, acquisition_function(initial_conditions)

    actual, _ = optimize_acqf(
        acq,
        bounds,
        q=2,
        num_restarts=4,
        raw_samples=32,
        gen_candidates=retain,
        return_best_only=False,
        options={"seed": 17, "eta": 1.7, "init_batch_limit": 5},
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert max(acq.sizes) <= 5 and acq.sizes[:7] == [5, 5, 5, 5, 5, 5, 2]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "options",
    [
        {"topn": True, "largest": False, "sorted": True},
        {"nonnegative": True, "alpha": 0.8, "eta": 0.7},
    ],
)
def test_high_level_initializer_options_follow_reference_formula(device, options):
    pytest.importorskip("botorch")
    from botorch.optim import initializers as reference

    class PositiveQuadratic(Quadratic):
        def forward(self, X):
            return super().forward(X) + 2

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    samples = (
        torch.quasirandom.SobolEngine(1, scramble=True, seed=11)
        .draw(24, dtype=bounds.dtype)
        .to(device)
        .reshape(24, 1, 1)
    )
    scores = PositiveQuadratic()(samples)
    torch.manual_seed(11)
    expected = (
        reference.initialize_q_batch_topn(samples, scores, 3, largest=False)
        if options.get("topn")
        else reference.initialize_q_batch_nonneg(samples, scores, 3, alpha=0.8, eta=0.7)
    )[0]

    def retain(initial_conditions, acquisition_function, **kwargs):
        return initial_conditions, acquisition_function(initial_conditions)

    actual, _ = optimize_acqf(
        PositiveQuadratic(),
        bounds,
        1,
        3,
        24,
        options=dict(options, seed=11),
        gen_candidates=retain,
        return_best_only=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_unimplemented_optimizer_options_are_rejected(device):
    pytest.importorskip("scipy")
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    with pytest.raises(UnsupportedError, match="custom ic_generator"):
        optimize_acqf(Quadratic(), bounds, 1, 2, 8, unknown_initializer_argument=True)
    initial = bounds.new_full((2, 1, 1), 0.5)
    with pytest.raises(UnsupportedError, match="serial mode"):
        gen_candidates_scipy(initial, Quadratic(), use_parallel_mode=True)
    for generator in (gen_candidates_torch, gen_candidates_scipy):
        with pytest.raises(ValueError, match="positive integer"):
            generator(
                initial, Quadratic(), options={"max_optimization_problem_aggregation_size": 0}
            )
    with pytest.raises(UnsupportedError, match="inter-point linear"):
        optimize_acqf(
            Quadratic(),
            bounds,
            2,
            2,
            8,
            sequential=True,
            inequality_constraints=[
                (torch.tensor([[0, 0], [1, 0]]), bounds.new_tensor([1.0, 1.0]), 0.0)
            ],
        )


@pytest.mark.parametrize("device", DEVICES)
def test_custom_initializer_receives_fixed_features_constraints_and_extra_keywords(device):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    fixed = {1: 0.4}
    constraints = [(torch.tensor([0]), bounds.new_tensor([1.0]), 0.5)]

    def initialize(
        acq_function,
        bounds,
        q,
        num_restarts,
        raw_samples,
        fixed_features,
        options,
        inequality_constraints,
        equality_constraints,
        sentinel,
    ):
        assert fixed_features == fixed and inequality_constraints is constraints
        assert equality_constraints is None and sentinel == 17
        return bounds.new_tensor([[[0.6, 0.4]]]).expand(num_restarts, q, 2)

    def retain(initial_conditions, acquisition_function, **kwargs):
        return initial_conditions, acquisition_function(initial_conditions)

    candidates, _ = optimize_acqf(
        Quadratic(),
        bounds,
        1,
        2,
        8,
        fixed_features=fixed,
        inequality_constraints=constraints,
        ic_generator=initialize,
        gen_candidates=retain,
        sentinel=17,
        return_best_only=False,
    )
    torch.testing.assert_close(candidates, bounds.new_tensor([[[0.6, 0.4]], [[0.6, 0.4]]]))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["linear", "nonlinear", "bounds", "fixed"])
def test_postprocessing_cannot_return_infeasible_candidates(device, kind):
    pytest.importorskip("scipy")
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    initial = bounds.new_full((2, 1, 1), 0.6)
    kwargs = {}
    if kind == "linear":
        kwargs["inequality_constraints"] = [(torch.tensor([0]), bounds.new_tensor([1.0]), 0.5)]
    elif kind == "nonlinear":
        kwargs["nonlinear_inequality_constraints"] = [(lambda X: X[0] - 0.5, True)]
    elif kind == "fixed":
        kwargs["fixed_features"] = {0: 0.6}
    post = (lambda X: X * 0 + 1.2) if kind == "bounds" else (lambda X: X * 0)
    with pytest.raises(CandidateGenerationError, match="Post-processing"):
        optimize_acqf(
            Quadratic(),
            bounds,
            1,
            2,
            batch_initial_conditions=initial,
            post_processing_func=post,
            options={"maxiter": 15},
            **kwargs,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_lbfgs_uses_fresh_closure_and_leaves_gp_parameter_grads_clean(device):
    from botorch_rs.acquisition.analytic import PosteriorMean
    from botorch_rs.models import SingleTaskGP

    initial = torch.tensor([[[0.1]], [[0.9]]], dtype=torch.double, device=device)
    options = {
        "optimizer_options": {"lr": 1.0, "max_iter": 20, "line_search_fn": "strong_wolfe"},
        "stopping_criterion_options": {"maxiter": 4, "n_window": 4},
    }
    actual, _ = gen_candidates_torch(
        initial, Quadratic(), 0.0, 1.0, optimizer=torch.optim.LBFGS, options=options
    )
    torch.testing.assert_close(actual, torch.full_like(actual, 0.73), atol=1e-5, rtol=0)
    X = torch.linspace(0, 1, 5, dtype=torch.double, device=device).unsqueeze(-1)
    model = SingleTaskGP(X, (5 * X).sin(), backend="torch")
    acq = PosteriorMean(model)
    _, values = gen_candidates_torch(
        initial, acq, 0.0, 1.0, optimizer=torch.optim.LBFGS, options=options
    )
    assert values.max() > acq(initial).max()
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("retry", [False, True])
def test_optimization_warning_retries_only_generated_initial_conditions(device, supplied, retry):
    from botorch_rs.exceptions.warnings import OptimizationWarning

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    seen = []

    def warn_once(initial_conditions, acquisition_function, **kwargs):
        seen.append(initial_conditions.clone())
        if len(seen) == 1:
            warnings.warn("retry this attempt", OptimizationWarning, stacklevel=2)
        return initial_conditions, acquisition_function(initial_conditions)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimize_acqf(
            Quadratic(),
            bounds,
            1,
            2,
            8,
            gen_candidates=warn_once,
            batch_initial_conditions=bounds.new_full((2, 1, 1), 0.5) if supplied else None,
            options={"seed": 11},
            retry_on_optimization_warning=retry,
        )
    attempted = not supplied and retry
    assert len(seen) == (2 if attempted else 1)
    assert any(issubclass(w.category, OptimizationWarning) for w in caught) != attempted
    if attempted:
        assert not torch.equal(seen[0], seen[1])
