"""Analytical and native/Torch checks of the Schmitt decision extensions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from botorch_rs.decision_making import knee_region, select_preferred

FRONT = [[0.0, 1.0], [0.2, 0.4], [0.4, 0.2], [1.0, 0.0]]


def test_native_analytical_knee_and_preferences():
    assert knee_region(FRONT, alpha=0.8, maximize=False) == [False, True, True, False]
    assert select_preferred(FRONT, [80.0, 20.0], alpha=0.8, maximize=False) == 1
    assert select_preferred(FRONT, [20.0, 80.0], alpha=0.8, maximize=False) == 2
    assert select_preferred(FRONT, [1.0, 0.0], alpha=0, maximize=False) == 0


def test_signed_distance_is_not_absolute_or_maximum_sum():
    front = [[1.0, 1.0, 0.0], [0.8, 0.8, 0.8], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
    for alpha in (0.0, 0.5, 1.0):
        assert knee_region(front, alpha=alpha, maximize=False) == [True, False, True, True]


def test_native_duplicate_and_dominated_rows_preserve_original_indices():
    front = [[9.0, 9.0], *FRONT, FRONT[1]]
    assert knee_region(front, alpha=0.8, maximize=False) == [False, False, True, True, False, False]
    assert select_preferred(front, [4.0, 1.0], alpha=0.8, maximize=False) == 2
    assert knee_region([[2.0, 2.0], [2.0, 2.0]], maximize=False) == [True, False]


def test_native_batch_preferences_broadcast_and_integer_outputs():
    front = [[FRONT, FRONT], [FRONT, FRONT]]
    indices = select_preferred(front, [[[4.0, 1.0]], [[1.0, 4.0]]], maximize=False)
    assert indices == [[1, 1], [2, 2]]
    assert all(isinstance(index, int) for row in indices for index in row)
    with pytest.raises(ValueError, match="broadcast"):
        select_preferred(front, [[1.0, 1.0]] * 3, maximize=False)


@pytest.mark.parametrize("alpha", [-1.0, 1.01, float("nan"), float("inf")])
def test_invalid_knee_options(alpha):
    with pytest.raises(ValueError):
        knee_region(FRONT, alpha=alpha)


@pytest.mark.parametrize("preferences", [[0.0, 0.0], [-1.0, 1.0], [float("inf"), 1.0]])
def test_invalid_preferences(preferences):
    with pytest.raises(ValueError):
        select_preferred(FRONT, preferences)


def test_native_decisions_have_no_torch_numpy_or_scipy_dependency():
    source = Path(__file__).resolve().parents[1] / "python"
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(source)!r}); "
        "from botorch_rs.decision_making import knee_region, select_preferred; "
        f"front={FRONT!r}; "
        "assert select_preferred(front,[4,1],maximize=False)==1; "
        "assert knee_region(front,maximize=False)==[False,True,True,False]; "
        "assert not {'torch','numpy','scipy'}.intersection(sys.modules)\n"
        "from botorch_rs.decision_making import sample_fpbi\n"
        "def costs(X):\n"
        "    return [[X[0][0]**2, (1-X[0][0])**2]]\n"
        "front=sample_fpbi(costs,[[0.],[1.]],n_steps=2,maximize=False,num_restarts=1)\n"
        "assert len(front.X)==3 and all(front.ray_succeeded)\n"
        "assert not {'torch','numpy','scipy'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-S", "-c", script], check=True, capture_output=True, text=True)


def test_numpy_native_and_torch_parity():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(231)
    for m in (2, 3, 5):
        values = rng.normal(size=(2, 3, 24, m))
        values[..., -1, :] = values[..., 0, :]
        if m == 3:
            values[..., 2] = 5.0  # constant objective
        prefs = rng.uniform(size=(2, 1, m))
        for maximize in (True, False):
            native_mask = knee_region(values, alpha=0.7, maximize=maximize)
            tensor_mask = knee_region(torch.tensor(values), alpha=0.7, maximize=maximize)
            assert np.array_equal(native_mask, tensor_mask.numpy())
            native_index = select_preferred(values, prefs, alpha=0.7, maximize=maximize)
            tensor_index = select_preferred(
                torch.tensor(values), torch.tensor(prefs), alpha=0.7, maximize=maximize
            )
            assert native_index.dtype == np.int64
            assert tensor_index.dtype == torch.int64
            assert np.array_equal(native_index, tensor_index.numpy())


def test_large_finite_ranges_and_preference_scales():
    torch = pytest.importorskip("torch")
    front = [[-1e308, 1e308], [-5e307, -5e307], [1e308, -1e308]]
    assert knee_region(front, maximize=False) == [False, True, False]
    tensor = torch.tensor(front, dtype=torch.float64)
    assert knee_region(tensor, maximize=False).tolist() == [False, True, False]
    assert select_preferred(front, [1e308, 1e308], maximize=False) == 1
    assert select_preferred(tensor, [1e308, 1e308], maximize=False).item() == 1


def test_affine_scales_and_maximization_orientation():
    torch = pytest.importorskip("torch")
    y = torch.tensor(FRONT, dtype=torch.float64)
    transformed = y * torch.tensor([150.0, 0.001]) + torch.tensor([30.0, -2.0])
    for alpha in (0.0, 0.5, 1.0):
        expected = select_preferred(y, [4.0, 1.0], alpha=alpha, maximize=False)
        assert select_preferred(-y, [4.0, 1.0], alpha=alpha).item() == expected.item()
        assert (
            select_preferred(transformed, [4.0, 1.0], alpha=alpha, maximize=False).item()
            == expected.item()
        )


@pytest.mark.skipif(sys.platform == "emscripten", reason="native device test")
def test_torch_device_and_dtype():
    torch = pytest.importorskip("torch")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for dtype in (torch.float32, torch.float64):
            y = torch.tensor(FRONT, device=device, dtype=dtype)
            mask = knee_region(y, maximize=False)
            index = select_preferred(y, [4.0, 1.0], maximize=False)
            assert mask.device == y.device and mask.dtype == torch.bool
            assert index.device == y.device and index.dtype == torch.int64
            assert index.item() == 1


def test_fpbi_source_grid_and_batched_focus_native_torch_parity():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from botorch_rs.decision_making import FPBIGeometry, fpbi_geometry

    extremes = np.array([[0.0, 1.0], [1.0, 0.0]])
    native = fpbi_geometry(extremes, n_steps=2, maximize=False)
    tensor = fpbi_geometry(torch.tensor(extremes), n_steps=2, maximize=False)
    assert isinstance(native, FPBIGeometry) and isinstance(tensor, FPBIGeometry)
    assert np.allclose(native.anchors, [[0.5, 0.5], [1.0, 0.0]])
    assert np.allclose(native.direction, [-0.5, -0.5])
    assert np.allclose(native.anchors, tensor.anchors.numpy(), atol=1e-12)
    # The paper's r_F=1 anchor is the second extreme, not the midpoint.
    one = fpbi_geometry(extremes.tolist(), n_steps=1, maximize=False)
    assert np.allclose(one.anchors, [[1.0, 0.0]])
    many = np.broadcast_to(extremes, (2, 3, 2, 2)).copy()
    focus = np.array([[[0.0, 0.0]], [[0.1, -0.1]]])
    native = fpbi_geometry(many, focus=focus, n_steps=2, maximize=False)
    tensor = fpbi_geometry(torch.tensor(many), focus=torch.tensor(focus), n_steps=2, maximize=False)
    assert np.allclose(native.anchors, tensor.anchors.numpy())
    assert np.allclose(native.focus, tensor.focus.numpy())
    assert np.allclose(native.direction, tensor.direction.numpy())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_fpbi_continuous_quadratic_front_and_feasibility(dtype):
    np = pytest.importorskip("numpy")
    from botorch_rs.decision_making import FPBIResult, sample_fpbi

    def objective(X):
        return np.column_stack([X[:, 0] ** 2, (1 - X[:, 0]) ** 2])

    result = sample_fpbi(
        objective,
        np.array([[0.0], [1.0]], dtype=dtype),
        n_steps=2,
        maximize=False,
        num_restarts=2,
    )
    assert isinstance(result, FPBIResult)
    assert np.allclose(result.Y, objective(result.X), atol=1e-12)
    assert np.allclose(np.sort(result.X[:, 0]), [0.0, 0.5, 1.0], atol=1e-5)
    assert result.ray_succeeded.tolist() == [True, True]
    assert all(d.max_violation <= 1e-6 for d in result.diagnostics if d.accepted)
    middle = [d for d in result.diagnostics if d.kind == "ray" and d.index == 0]
    assert len(middle) == 1 and middle[0].kappa == pytest.approx(0.5, abs=1e-5)


def test_native_fpbi_constraints_and_declared_failed_solves():
    np = pytest.importorskip("numpy")
    from botorch_rs.decision_making import sample_fpbi

    def objective(X):
        return np.column_stack([X[:, 0] ** 2, (1 - X[:, 0]) ** 2])

    result = sample_fpbi(
        objective,
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        n_steps=2,
        maximize=False,
        num_restarts=2,
        inequality_constraints=lambda X: np.column_stack([X[:, 0] - 0.2, 0.8 - X[:, 0]]),
        equality_constraints=lambda X: X[:, 0] - X[:, 1],
    )
    assert np.all(result.X[:, 0] >= 0.2 - 1e-6)
    assert np.all(result.X[:, 0] <= 0.8 + 1e-6)
    assert np.allclose(result.X[:, 0], result.X[:, 1], atol=1e-6)
    assert result.ray_succeeded.all()
    from botorch_rs.exceptions.errors import CandidateGenerationError

    with pytest.raises(CandidateGenerationError, match="Iteration limit"):
        sample_fpbi(objective, np.array([[0.0, 0.0], [1.0, 1.0]]), maximize=False, maxiter=1)


def test_botorch_and_rust_models_share_posterior_mean_adapter():
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    from botorch.models.deterministic import GenericDeterministicModel as BoTorchModel

    from botorch_rs.decision_making import sample_model_fpbi
    from botorch_rs.models.deterministic import GenericDeterministicModel as RustModel

    def objective(X):
        return torch.stack([X[..., 0].square(), (1 - X[..., 0]).square()], dim=-1)

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    fronts = []
    for model_class in (BoTorchModel, RustModel):
        result = sample_model_fpbi(
            model_class(objective, num_outputs=2),
            bounds,
            n_steps=2,
            maximize=False,
            num_restarts=1,
        )
        assert result.Y.shape[-1] == 2 and result.ray_succeeded.all()
        fronts.append(result.X[:, 0].sort().values)
    assert torch.allclose(fronts[0], fronts[1], atol=1e-6)


@pytest.mark.parametrize("maximize", [False, True])
def test_native_fpbi_nadir_preserves_cancellation_prone_endpoints(maximize):
    from botorch_rs.decision_making import fpbi_geometry

    costs = [[-1e16, 1.0], [1.0, -1e16]]
    values = [[-v for v in row] for row in costs] if maximize else costs
    geometry = fpbi_geometry(values, n_steps=2, maximize=maximize)
    assert geometry.utopia == [-1e16, -1e16]
    assert geometry.nadir == [1.0, 1.0]
    assert geometry.normalized_extremes == [[0.0, 1.0], [1.0, 0.0]]


def test_native_fpbi_grid_guard_precedes_objective_optimization():
    from botorch_rs.decision_making import sample_fpbi

    objective_calls = 0
    constraint_calls = 0

    def objective(X):
        nonlocal objective_calls
        objective_calls += 1
        x = X[0][0]
        return [[x * x, (1 - x) ** 2]]

    def constraints(X):
        nonlocal constraint_calls
        constraint_calls += 1
        return [1.0]

    with pytest.raises(ValueError, match="grid"):
        sample_fpbi(
            objective,
            [[0.0], [1.0]],
            n_steps=2,
            max_rays=1,
            num_restarts=1,
            maximize=False,
            inequality_constraints=constraints,
        )
    assert objective_calls == 1
    assert constraint_calls == 0
