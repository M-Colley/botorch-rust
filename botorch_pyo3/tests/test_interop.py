"""Array interoperability: torch tensors, NumPy arrays and nested lists in,
the same type out, without importing torch or NumPy unless they are used."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import botorch_rs
from botorch_rs._arrays import (
    ArrayKind,
    first_kind,
    from_flat,
    kind_from_dtype,
    kind_of,
    to_flat,
)
from helpers import HAS_NUMPY, HAS_TORCH, as_kind, check_kind, shape


def test_to_flat_nested_lists() -> None:
    assert to_flat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) == ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
    assert to_flat(2.5) == ([2.5], ())
    assert to_flat([]) == ([], (0,))
    assert to_flat(((1, 2),)) == ([1, 2], (1, 2))
    data, shp = to_flat([[[1.0], [2.0]], [[3.0], [4.0]]])
    assert data == [1.0, 2.0, 3.0, 4.0] and shp == (2, 2, 1)


@pytest.mark.parametrize("bad", [[[1.0, 2.0], [3.0]], [[1.0], 2.0], [["a"]]])
def test_to_flat_rejects_ragged_and_non_numeric(bad: list) -> None:
    with pytest.raises((ValueError, TypeError)):
        to_flat(bad)


def test_to_flat_rejects_unknown_types() -> None:
    with pytest.raises(TypeError):
        to_flat({"a": 1.0})


def test_from_flat_roundtrip(kind: str) -> None:
    x = as_kind([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], kind)
    k = kind_of(x)
    data, shp = to_flat(x)
    y = from_flat(data, shp, k)
    check_kind(y, kind)
    assert shape(y) == (2, 3)
    assert to_flat(y) == (data, shp)
    scalar = from_flat([1.5], (), k)
    assert float(scalar) == 1.5
    assert shape(scalar) == ()


def test_kind_detection() -> None:
    assert kind_of([1.0]).lib == "list"
    assert first_kind([1.0], None, 2.0).lib == "list"


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy is not installed")
def test_numpy_dtypes() -> None:
    import numpy as np

    x32 = np.ones((2, 2), dtype=np.float32)
    assert kind_of(x32) == ArrayKind("numpy", np.dtype(np.float32))
    assert from_flat([1.0, 2.0], (2,), kind_of(x32)).dtype == np.float32
    xi = np.ones((2, 2), dtype=np.int64)
    assert from_flat([1.0], (1,), kind_of(xi)).dtype == np.float64
    data, shp = to_flat(xi)
    assert list(data) == [1, 1, 1, 1] and shp == (2, 2)
    assert kind_from_dtype(np.float64).lib == "numpy"


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_torch_dtypes_and_devices() -> None:
    import torch

    x32 = torch.ones(2, 2, dtype=torch.float32, requires_grad=True)
    k = kind_of(x32)
    assert k.lib == "torch" and k.dtype == torch.float32
    assert from_flat([1.0, 2.0], (2,), k).dtype == torch.float32
    data, shp = to_flat(x32)
    assert list(data) == [1.0] * 4 and shp == (2, 2)
    xi = torch.ones(3, dtype=torch.int64)
    assert from_flat([0.5], (1,), kind_of(xi)).dtype == torch.float64
    assert kind_from_dtype(torch.float64) == ArrayKind("torch", torch.float64, None)
    # Nested lists of tensors are flattened too.
    assert to_flat([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]) == (
        [1.0, 2.0, 3.0, 4.0],
        (2, 2),
    )


def test_kind_from_dtype_default_follows_torch_import() -> None:
    expected = "torch" if "torch" in sys.modules else "list"
    assert kind_from_dtype(None).lib == expected
    with pytest.raises(TypeError):
        kind_from_dtype(object())


def test_no_torch_or_numpy_import() -> None:
    """With nested lists, a full fit-and-optimize run imports neither library."""
    code = textwrap.dedent(
        """
        import sys
        import botorch_rs
        from botorch_rs.acquisition import qLogNoisyExpectedImprovement
        from botorch_rs.fit import fit_gpytorch_mll
        from botorch_rs.mlls import ExactMarginalLogLikelihood
        from botorch_rs.models import SingleTaskGP
        from botorch_rs.models.transforms import Normalize
        from botorch_rs.optim import optimize_acqf
        from botorch_rs.test_functions import Branin
        from botorch_rs.utils.sampling import draw_sobol_samples

        problem = Branin(negate=True)
        bounds = problem.bounds
        X = [x[0] for x in draw_sobol_samples(bounds, n=8, q=1, seed=0)]
        Y = [[y] for y in problem(X)]
        model = SingleTaskGP(X, Y, input_transform=Normalize(d=2))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acqf = qLogNoisyExpectedImprovement(model, X_baseline=X)
        cand, value = optimize_acqf(acqf, bounds, q=2, num_restarts=2, raw_samples=16)
        assert isinstance(cand, list) and isinstance(value, float)
        assert "torch" not in sys.modules, "torch was imported"
        assert "numpy" not in sys.modules, "numpy was imported"
        """
    )
    package_root = os.path.dirname(os.path.dirname(botorch_rs.__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
