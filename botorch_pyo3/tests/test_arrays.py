"""Array exchange with the extension: torch tensors and NumPy arrays with more
than a few values cross the boundary as float64 buffers (``array.array('d')``),
nested lists and small arrays as lists. Both routes must give the values of
``tolist()`` exactly, for every dtype and memory layout, and results must be
independent arrays of the caller's type, dtype and device."""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import textwrap
import threading
from array import array
from typing import Any

import pytest

import botorch_rs
from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    _NUMPY_LIST_MAX,
    _TORCH_LIST_MAX,
    as_float,
    bool_array,
    from_flat,
    from_pair,
    kind_of,
    to_flat,
)
from botorch_rs.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch_rs.acquisition.objective import GenericMCObjective, IdentityMCObjective
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.test_functions import Hartmann
from botorch_rs.test_functions.multi_objective import DTLZ2
from botorch_rs.utils.multi_objective.pareto import is_non_dominated
from helpers import HAS_NUMPY, HAS_TORCH, as_kind, make_data

needs_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="NumPy is not installed")
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")

#: Numbers of values below and above the list / buffer thresholds.
SIZES = [1, 6, 40, 600]
#: The extension returns results with more values than this as buffers.
RESULT_LIST_MAX = 16


def bits(values: Any) -> list[bytes]:
    """The exact float64 representations of a sequence of numbers."""
    return [struct.pack("<d", float(v)) for v in values]


def points(n: int, d: int, seed: int = 0) -> list[list[float]]:
    """``n`` deterministic points in [0, 1]^d."""
    alphas = [math.sqrt(p) % 1.0 for p in (2, 3, 5, 7, 11, 13, 17, 19)[:d]]
    return [[(0.1 + seed + i * a) % 1.0 for a in alphas] for i in range(n)]


# -- to_flat: the values of tolist(), for every dtype and layout ---------------


def _numpy_cases() -> list[Any]:
    import numpy as np

    rng = np.random.default_rng(0)
    base = rng.standard_normal((40, 15)) * 10.0 ** rng.integers(-20, 20, (40, 15))
    with np.errstate(over="ignore"):  # float16 overflows to infinity
        cases = [base.astype(t) for t in (np.float16, np.float32, np.float64, np.longdouble)]
    for t in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        info = np.iinfo(t)
        cases.append(rng.integers(info.min, info.max, (40, 15), dtype=t, endpoint=True))
    cases += [
        rng.random((40, 15)) > 0.5,  # bool
        (2**64 - 1 - rng.integers(0, 2**40, 600, dtype=np.uint64)).astype(np.uint64),
        base.astype(">f8"),  # big-endian
        rng.integers(-1000, 1000, (40, 15)).astype(">i4"),
        base[::2, ::-3],  # non-contiguous with a negative stride
        base[::-1],  # negative stride
        np.broadcast_to(base[:1], (40, 15)),  # zero stride, read-only
        np.asfortranarray(base),
        base[:, :1].reshape(40),  # small
        np.float64(3.25),  # NumPy scalar
        np.float32(-1.5),
        np.array(2.5),  # 0-dim
        np.empty((0, 3)),
        np.empty((3, 0, 2)),
    ]
    read_only = base.copy()
    read_only.flags.writeable = False
    cases.append(read_only)
    return cases


@needs_numpy
def test_to_flat_numpy_matches_tolist() -> None:
    import numpy as np

    for x in _numpy_cases():
        data, shape = to_flat(x)
        a = np.asarray(x)
        assert shape == a.shape
        assert bits(data) == bits(a.reshape(-1).tolist()), (a.dtype, a.shape)
        expected = list if a.size <= _NUMPY_LIST_MAX else array
        assert isinstance(data, expected), (type(data), a.size)


def _torch_cases() -> list[Any]:
    import torch

    g = torch.Generator().manual_seed(0)
    base = torch.randn(40, 15, generator=g, dtype=torch.float64) * 1e3
    cases = [base.to(t) for t in (torch.float16, torch.bfloat16, torch.float32, torch.float64)]
    for t in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        info = torch.iinfo(t)
        cases.append(torch.randint(info.min, info.max, (40, 15), generator=g, dtype=t))
    cases += [
        base > 0,  # bool
        torch.full((300,), 2**62 + 12345, dtype=torch.int64),
        base.t(),  # non-contiguous
        base[:, ::3],
        base[:1].expand(40, 15),  # zero stride
        torch._neg_view(base),  # negative bit
        base.clone().requires_grad_(True),
        base[:2, :3],  # small
        torch.tensor(2.5, dtype=torch.float64),  # 0-dim
        torch.tensor(3, dtype=torch.int32),
        torch.empty(0, 3),
        torch.empty(3, 0, 2),
    ]
    return cases


@needs_torch
def test_to_flat_torch_matches_tolist() -> None:
    for x in _torch_cases():
        data, shape = to_flat(x)
        assert shape == tuple(x.shape)
        assert bits(data) == bits(x.detach().reshape(-1).tolist()), (x.dtype, x.shape)
        expected = list if x.numel() <= _TORCH_LIST_MAX else array
        assert isinstance(data, expected), (type(data), x.numel())


def test_to_flat_lists_and_scalars_are_unchanged() -> None:
    assert to_flat([[1, 2.5], [True, -0.0]]) == ([1, 2.5, True, -0.0], (2, 2))
    assert to_flat(1.5) == ([1.5], ())
    assert to_flat([]) == ([], (0,))


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_complex_inputs_are_rejected(lib: str) -> None:
    if lib == "numpy":
        np = pytest.importorskip("numpy")
        for n in (1, 100):
            with pytest.raises(TypeError, match=r"^`Y` must be a real-valued array, got dtype "):
                to_flat(np.ones(n, dtype=np.complex128), "Y")
    else:
        torch = pytest.importorskip("torch")
        for n in (1, 1000):
            with pytest.raises(TypeError, match=r"^`Y` must be real-valued\.$"):
                to_flat(torch.ones(n, dtype=torch.complex64), "Y")


# -- from_flat: independent results of the caller's type ---------------------


def _target_dtypes(lib: str) -> list[Any]:
    if lib == "numpy":
        import numpy as np

        return [np.float16, np.float32, np.float64, np.longdouble, np.int64, np.bool_]
    import torch

    return [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64]


@pytest.mark.parametrize("n", SIZES)
def test_round_trips(kind: str, n: int) -> None:
    values = [math.sin(1.7 * i) * 10.0 ** (i % 7 - 3) for i in range(n)]
    shape = (n,) if n < 6 else (n // 6, 6) if n % 6 == 0 else (n,)
    if kind == "list":
        for data in (values, array("d", values)):
            out = from_flat(data, shape, kind_of([]))
            assert to_flat(out) == (values, shape)
        assert from_flat(array("d", values[:1]), (), kind_of([])) == values[0]
        return
    for dtype in _target_dtypes(kind):
        if kind == "numpy":
            import numpy as np

            k = kind_of(np.zeros(1, dtype=dtype))
        else:
            import torch

            k = kind_of(torch.zeros(1, dtype=dtype))
        for data in (values, array("d", values)):
            out = from_flat(data, shape, k)
            # The values of the list route, i.e. of `torch.tensor` /
            # `np.asarray` on the list, bit for bit.
            if kind == "torch":
                import torch

                ref = torch.tensor(values, dtype=k.dtype or torch.float64).reshape(shape)
                assert out.dtype == ref.dtype and out.shape == ref.shape
                assert torch.equal(out.view(-1).view(torch.uint8), ref.view(-1).view(torch.uint8))
            else:
                import numpy as np

                ref = np.asarray(values, dtype=k.dtype or np.float64).reshape(shape)
                assert out.dtype == ref.dtype and out.shape == ref.shape
                if out.dtype == np.longdouble and out.dtype.itemsize > 8:
                    # x87 extended precision is padded to 12 or 16 bytes, and
                    # the padding is uninitialized: compare the values.
                    assert np.array_equal(out, ref)
                else:
                    assert out.tobytes() == ref.tobytes()
                assert out.flags.writeable and out.flags.c_contiguous
            # Results never share memory with the data.
            if isinstance(data, array):
                data[0] += 1.0
                assert to_flat(out)[0][0] == ref.reshape(-1).tolist()[0]
                data[0] -= 1.0


def test_scalar_results(kind: str) -> None:
    x = as_kind([[0.5]], kind)
    for data in ([1.25], array("d", [1.25])):
        out = from_flat(data, (), kind_of(x))
        assert float(out) == 1.25
        if kind == "numpy":
            import numpy as np

            assert isinstance(out, np.float64)
        elif kind == "torch":
            assert out.dim() == 0
        else:
            assert isinstance(out, float)
    assert as_float(as_kind([2.5], kind), "x") == 2.5


def test_empty_results(kind: str) -> None:
    x = as_kind([[0.5]], kind)
    for data in ([], array("d")):
        out = from_flat(data, (0, 3), kind_of(x))
        if kind == "list":
            assert out == []
        else:
            assert tuple(out.shape) == (0, 3)


# -- the extension: buffers in, buffers out -----------------------------------


def _echo(data: Any, n: int) -> tuple[Any, list[int]]:
    """``(data, (n,))`` through the extension (a reference point expanded to
    no batch dimensions is returned unchanged)."""
    return _rs.expand_ref_point((data, (n,)), [])


def test_array_extraction_accepts_buffers_and_sequences() -> None:
    values = [0.5, -1.25, 3.0e300, -0.0, float("inf")] * 10
    n = len(values)
    for data in (values, tuple(values), array("d", values), memoryview(array("d", values))):
        out, shape = _echo(data, n)
        assert shape == [n] and bits(out) == bits(values)
    # Buffers of other element types are read value by value.
    out, _ = _echo(array("f", [0.5, 2.0]), 2)
    assert list(out) == [0.5, 2.0]
    out, _ = _echo(array("d"), 0)
    assert list(out) == []


@needs_numpy
def test_array_extraction_of_numpy_buffers() -> None:
    import numpy as np

    values = np.linspace(-3.0, 7.0, 50)
    for a in (values, values.astype(">f8"), values.astype(np.float32), values[::-1].copy()):
        out, _ = _echo(a, 50)
        assert bits(out) == bits(a.tolist()), a.dtype
    # Big-endian float64 is not misread as native (pyo3's format check
    # accepts `>d` on little-endian machines).
    out, _ = _echo(np.array([1.0, 2.0], dtype=">f8"), 2)
    assert list(out) == [1.0, 2.0]


def test_large_results_are_buffers() -> None:
    problem = Hartmann(dim=6)
    for n in (RESULT_LIST_MAX, RESULT_LIST_MAX + 1, 500):
        X = points(n, 6)
        data, shape = problem._rs.evaluate_true(to_flat(X))
        assert shape == [n]
        assert isinstance(data, list if n <= RESULT_LIST_MAX else array)
        assert bits(data) == bits(problem.evaluate_true(X))


def test_results_do_not_alias(kind: str) -> None:
    """Mutating a result does not change later results (cached posteriors
    included), and inputs are copied when they are passed."""
    X, Y = make_data(n=12, d=3, seed=1)
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    Xt = as_kind(points(200, 3), kind)
    post = model.posterior(Xt)
    mean = post.mean
    first = to_flat(mean)[0][:]
    if kind == "list":
        mean[0][0] += 1.0
    else:
        mean += 1.0
    assert to_flat(post.mean)[0] == first
    assert to_flat(model.posterior(Xt).mean)[0] == first
    problem = DTLZ2(dim=3, num_objectives=2)
    f = problem(Xt)
    expected = to_flat(f)[0][:]
    if kind == "list":
        f[0][0] = 0.0
    else:
        f *= 0.0
    assert to_flat(problem(Xt))[0] == expected
    # A posterior is computed from a snapshot of its inputs.
    post = model.posterior(Xt)
    if kind == "list":
        Xt[0][0] += 0.5
    else:
        Xt += 0.5
    assert to_flat(post.mean)[0] == first


@needs_torch
def test_torch_results_keep_dtype_and_device() -> None:
    import torch

    problem = DTLZ2(dim=4, num_objectives=2)
    X64 = torch.tensor(points(300, 4), dtype=torch.float64)
    ref = problem(X64)
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        out = problem(X64.to(dtype))
        assert out.dtype == dtype and out.device == X64.device
        # The input values are rounded to `dtype` first.
        assert torch.equal(out, problem(X64.to(dtype).to(torch.float64)).to(dtype))
    out = problem(X64.to(torch.int64))
    assert out.dtype == torch.float64
    assert torch.equal(problem(X64), ref)


@needs_numpy
def test_numpy_results_keep_dtype() -> None:
    import numpy as np

    problem = DTLZ2(dim=4, num_objectives=2)
    X = np.asarray(points(300, 4))
    for dtype in (np.float16, np.float32, np.float64, np.longdouble):
        out = problem(X.astype(dtype))
        assert out.dtype == dtype and out.flags.writeable
    assert problem(X.astype(np.int64)).dtype == np.float64
    # Big-endian inputs give big-endian results (the input's dtype).
    out = problem(X.astype(">f8"))
    assert out.dtype == np.dtype(">f8") and np.array_equal(out, problem(X))


# -- autograd through the bridges ---------------------------------------------


@needs_torch
@pytest.mark.parametrize("b", [1, 3, 64])
def test_acquisition_gradients(b: int) -> None:
    import torch

    X, Y = make_data(n=15, d=4, seed=2)
    model = SingleTaskGP(torch.tensor(X, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64))
    acqf = LogExpectedImprovement(model, best_f=max(y[0] for y in Y))
    for dtype in (torch.float32, torch.float64):
        Xc = torch.tensor(points(b, 4, seed=1), dtype=dtype).unsqueeze(1)
        Xc.requires_grad_(True)
        values = acqf(Xc)
        assert values.dtype == dtype and values.shape == (b,)
        (grad,) = torch.autograd.grad(values.sum(), Xc)
        v_ref, g_ref = acqf.value_and_grad(Xc.detach())
        assert torch.equal(values.detach(), v_ref)
        assert grad.dtype == dtype and torch.equal(grad, g_ref)


@needs_torch
def test_callable_objective_gradients() -> None:
    """A callable objective (evaluated through ``_torch_bridge`` on 256 x q x 1
    samples) gives the values and gradients of the identity objective."""
    import torch

    X, Y = make_data(n=15, d=3, seed=4)
    model = SingleTaskGP(X, Y)

    def acqf(objective: Any) -> Any:
        sampler = SobolQMCNormalSampler(sample_shape=[256], seed=0)
        return qLogExpectedImprovement(
            model, best_f=max(y[0] for y in Y), sampler=sampler, objective=objective
        )

    generic = acqf(GenericMCObjective(lambda samples, X=None: samples[..., 0]))
    identity = acqf(IdentityMCObjective())
    for q in (1, 4):
        Xc = torch.tensor(points(2 * q, 3, seed=2), dtype=torch.float64).reshape(2, q, 3)
        Xg = Xc.clone().requires_grad_(True)
        values = generic(Xg)
        (grad,) = torch.autograd.grad(values.sum(), Xg)
        v_ref, g_ref = identity.value_and_grad(Xc)
        torch.testing.assert_close(values.detach(), v_ref, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(grad, g_ref, rtol=1e-10, atol=1e-12)


@needs_torch
def test_scalarization_gradients() -> None:
    """Gradients of a Chebyshev scalarization of many outcomes (buffers) equal
    those of each outcome on its own (lists)."""
    import torch

    from botorch_rs.utils.multi_objective.scalarization import get_chebyshev_scalarization

    Y = torch.tensor(points(100, 3, seed=5), dtype=torch.float64)
    scalarization = get_chebyshev_scalarization(torch.tensor([0.2, 0.3, 0.5]), Y)
    Yg = Y.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(scalarization(Yg).sum(), Yg)
    for i in (0, 37, 99):
        yi = Y[i : i + 1].clone().requires_grad_(True)
        (gi,) = torch.autograd.grad(scalarization(yi).sum(), yi)
        assert torch.equal(gi[0], grad[i])


# -- threads and environments without NumPy or torch --------------------------


def test_concurrent_calls(kind: str) -> None:
    """Calls from several threads (the GIL is released during the Rust work)
    give the sequential results."""
    problem = DTLZ2(dim=5, num_objectives=3)
    inputs = [as_kind(points(400, 5, seed=s), kind) for s in range(8)]
    expected = [to_flat(problem(x))[0] for x in inputs]
    results: list[Any] = [None] * len(inputs)

    def work(i: int) -> None:
        for _ in range(5):
            results[i] = to_flat(problem(inputs[i]))[0]

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(inputs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == expected


def _run_without(blocked: list[str], code: str) -> None:
    """Runs ``code`` in a fresh interpreter in which importing the modules in
    ``blocked`` fails (CI's jobs without NumPy or torch)."""
    prelude = "import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in blocked)
    package_root = os.path.dirname(os.path.dirname(botorch_rs.__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


@needs_torch
def test_torch_without_numpy() -> None:
    _run_without(
        ["numpy"],
        """
        import warnings
        warnings.simplefilter("ignore")
        import torch
        from array import array
        from botorch_rs._arrays import from_flat, kind_of, to_flat
        from botorch_rs.models import SingleTaskGP
        from botorch_rs.test_functions.multi_objective import DTLZ2

        X = torch.rand(500, 4, dtype=torch.float64)
        for x in (X, X.to(torch.float32), X.t(), (X * 10).to(torch.int64)):
            data, shape = to_flat(x)
            assert isinstance(data, array) and shape == tuple(x.shape)
            assert data.tolist() == x.reshape(-1).tolist()
            y = from_flat(data, shape, kind_of(x))
            assert y.dtype == (x.dtype if x.is_floating_point() else torch.float64)
            assert torch.equal(y, x.to(y.dtype))
        f = DTLZ2(dim=4, num_objectives=2)(X)
        assert f.shape == (500, 2) and f.dtype == torch.float64
        model = SingleTaskGP(X[:20], X[:20].sum(-1, keepdim=True))
        post = model.posterior(X)
        assert post.mean.shape == (500, 1) and bool((post.variance > 0).all())
        assert "numpy" not in sys.modules or sys.modules["numpy"] is None
        """,
    )


@needs_numpy
def test_numpy_without_torch() -> None:
    _run_without(
        ["torch"],
        """
        import numpy as np
        from botorch_rs.models import SingleTaskGP
        from botorch_rs.test_functions.multi_objective import DTLZ2

        X = np.random.default_rng(0).random((500, 4))
        f = DTLZ2(dim=4, num_objectives=2)(X)
        assert isinstance(f, np.ndarray) and f.shape == (500, 2)
        model = SingleTaskGP(X[:20], X[:20].sum(-1, keepdims=True))
        assert model.posterior(X).mean.shape == (500, 1)
        """,
    )


def test_lists_without_numpy_or_torch() -> None:
    _run_without(
        ["numpy", "torch"],
        """
        from botorch_rs.models import SingleTaskGP
        from botorch_rs.test_functions.multi_objective import DTLZ2

        X = [[(0.37 * i + 0.11 * j) % 1.0 for j in range(4)] for i in range(500)]
        f = DTLZ2(dim=4, num_objectives=2)(X)
        assert isinstance(f, list) and len(f) == 500 and isinstance(f[0][0], float)
        model = SingleTaskGP(X[:20], [[sum(x)] for x in X[:20]])
        mean = model.posterior(X).mean
        assert isinstance(mean, list) and len(mean) == 500
        """,
    )


def test_from_pair_accepts_both_forms(kind: str) -> None:
    k = kind_of(as_kind([[0.0]], kind))
    values = [float(i) for i in range(30)]
    a = from_pair((values, [5, 6]), k)
    b = from_pair((array("d", values), [5, 6]), k)
    assert list(to_flat(a)[0]) == list(to_flat(b)[0]) == values
    assert to_flat(a)[1] == (5, 6) == to_flat(b)[1]


# -- boolean masks -------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 17, 300])
def test_bool_arrays_from_bytes_and_lists(kind: str, n: int) -> None:
    k = kind_of(as_kind([[0.0]], kind))
    flags = [(i * 7) % 3 == 0 for i in range(n)]
    a = bool_array(flags, (n,), k)
    b = bool_array(bytes(flags), (n,), k)
    if kind == "list":
        assert a == b == flags and all(isinstance(v, bool) for v in b)
        return
    assert a.dtype == b.dtype and a.tolist() == b.tolist() == flags
    b[0] = not b[0]  # writable, and independent of later results
    assert bool_array(bytes(flags), (n,), k).tolist() == flags


def _non_dominated_reference(Y: list[list[float]]) -> list[bool]:
    return [
        not any(all(u >= v for u, v in zip(z, y, strict=True)) and z != y for z in Y) for y in Y
    ]


@pytest.mark.parametrize("n", [10, 300])
def test_pareto_masks(kind: str, n: int) -> None:
    Y = [[math.sin(3.1 * i + 0.3), math.cos(1.7 * i) * (1 + i % 3)] for i in range(n)]
    data, shape = _rs.is_non_dominated(to_flat(Y))
    assert shape == [n] and isinstance(data, list if n <= RESULT_LIST_MAX else bytes)
    mask = is_non_dominated(as_kind(Y, kind))
    flags = mask if kind == "list" else mask.tolist()
    assert flags == _non_dominated_reference(Y)
