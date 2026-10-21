"""Lazy JAX conversions and LCBench loading from synthetic local Parquet files."""

from __future__ import annotations

import dataclasses
import importlib
import io
import subprocess
import sys
from pathlib import Path

import pytest

from botorch_rs.utils import jax_utils, lcbench


def test_adapters_import_without_optional_libraries() -> None:
    source = """
import sys
import botorch_rs.utils.jax_utils
import botorch_rs.utils.lcbench
import botorch_rs.test_utils.mock
assert not {'torch', 'jax', 'jaxlib', 'numpy', 'pandas', 'pyarrow'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", source], check=True)


@pytest.mark.parametrize("shape", [(), (2, 3), (3, 2)])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_jax_conversions_match_reference_and_copy(shape: tuple[int, ...], dtype_name: str) -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    np = pytest.importorskip("numpy")
    bt = pytest.importorskip("botorch.utils.jax_utils")
    values = torch.arange(6 if shape else 1, dtype=getattr(torch, dtype_name)).reshape(shape)
    if shape == (3, 2):
        values = values.transpose(-1, -2)
    values.requires_grad_()
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        enable_x64 = jax.experimental.enable_x64
    with enable_x64():
        converted = jax_utils.torch_to_jax(values)
        expected = bt.torch_to_jax(values)
        np.testing.assert_array_equal(np.asarray(converted), np.asarray(expected))
        assert converted.dtype == expected.dtype
        converted.block_until_ready()
        saved = np.array(converted, copy=True)
        with torch.no_grad():
            values.fill_(19.0)
        np.testing.assert_array_equal(np.asarray(converted), saved)
        actual = jax_utils.jax_to_torch(converted, device=torch.device("cpu"), dtype=torch.double)
        reference = bt.jax_to_torch(converted, device=torch.device("cpu"), dtype=torch.double)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert not actual.requires_grad
        actual.fill_(-3.0)
        np.testing.assert_array_equal(np.asarray(converted), saved)


def _frames():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    config = pd.DataFrame(
        {
            name: [1.0 + i / 7, 2.0 + i / 5, 3.0 + i / 3]
            for i, name in enumerate(reversed(lcbench.LCBENCH_PARAMETER_NAMES))
        },
        index=pd.RangeIndex(3, name="trial"),
    )
    config["OpenML_task_id"] = [10, 20, 30]
    index = pd.MultiIndex.from_product([range(3), range(52)], names=["trial", "epoch"])
    metrics = pd.DataFrame(
        {
            lcbench.DEFAULT_LCBENCH_METRIC_NAME: [index / 10 for index in range(156)],
            "time": list(range(156)),
        },
        index=index,
    )
    return config, metrics


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_lcbench_local_parquet_matches_reference(
    tmp_path: Path, monkeypatch, dtype_name: str
) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.utils.lcbench")
    config, metrics = _frames()
    dataset = tmp_path / "Fashion-MNIST"
    dataset.mkdir()
    for stem, frame in (("config", config), ("metrics", metrics)):
        frame.to_parquet(dataset / f"{stem}.parquet.gzip", engine="pyarrow", compression="gzip")
    monkeypatch.setattr(
        lcbench.urllib.request,
        "urlopen",
        lambda *a, **kw: pytest.fail("local fixture must not download"),
    )
    dtype = getattr(torch, dtype_name)
    actual = lcbench.load_lcbench_data("Fashion-MNIST", dtype=dtype, cache_dir=tmp_path)
    expected = bt.load_lcbench_data("Fashion-MNIST", dtype=dtype, cache_dir=tmp_path)
    for name in ("parameters", "metrics", "epochs"):
        torch.testing.assert_close(
            getattr(actual, name),
            getattr(expected, name),
            atol=1e-15 if dtype_name == "float64" else 0,
            rtol=1e-15 if dtype_name == "float64" else 0,
        )
    assert actual.metrics.shape == (3, 50)
    assert actual.parameters.shape == (3, 7)
    assert torch.equal(actual.epochs, torch.arange(1, 51, dtype=dtype))
    assert (
        lcbench.load_lcbench_data("Fashion-MNIST", cache_dir=tmp_path).metrics.dtype == torch.double
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        actual.metrics = None
    assert list(config.columns) == list(reversed(lcbench.LCBENCH_PARAMETER_NAMES)) + [
        "OpenML_task_id"
    ]


def test_parquet_atomic_cache_and_parse_failures(tmp_path: Path, monkeypatch) -> None:
    pd = pytest.importorskip("pandas")
    config, _ = _frames()
    payload = config.to_parquet(engine="pyarrow", compression="gzip")
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(lcbench.urllib.request, "urlopen", download)
    actual = lcbench._read_lcbench_parquet("Fashion-MNIST", "config", tmp_path)
    pd.testing.assert_frame_equal(actual, config)
    cached = tmp_path / "Fashion-MNIST" / "config.parquet.gzip"
    assert cached.read_bytes() == payload
    assert len(calls) == 1 and calls[0][1] == 60.0
    pd.testing.assert_frame_equal(
        lcbench._read_lcbench_parquet("Fashion-MNIST", "config", tmp_path), config
    )
    assert len(calls) == 1
    monkeypatch.setattr(
        lcbench.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"truncated parquet")
    )
    with pytest.raises(pytest.importorskip("pyarrow").ArrowInvalid):
        lcbench._read_lcbench_parquet("Fashion-MNIST", "metrics", tmp_path)
    assert not (cached.parent / "metrics.parquet.gzip").exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_cache_publish_failure_cleans_temporary(tmp_path: Path, monkeypatch) -> None:
    config, _ = _frames()
    payload = config.to_parquet(engine="pyarrow", compression="gzip")
    monkeypatch.setattr(lcbench.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    monkeypatch.setattr(
        lcbench.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("cannot publish"))
    )
    with pytest.raises(OSError, match="cannot publish"):
        lcbench._read_lcbench_parquet("Fashion-MNIST", "config", tmp_path)
    assert list(tmp_path.rglob("*.tmp")) == []
    assert list(tmp_path.rglob("*.gzip")) == []


def test_lcbench_plain_log_preserves_nonpositive_and_tiny_values(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.utils.lcbench")
    config, metrics = _frames()
    config.loc[0, "batch_size"] = 0.0
    config.loc[1, "max_units"] = -1.0
    config.loc[2, "learning_rate"] = 1e-200

    def read(dataset_name, stem, cache_dir):
        return (config if stem == "config" else metrics).copy(deep=True)

    monkeypatch.setattr(lcbench, "_read_lcbench_parquet", read)
    monkeypatch.setattr(bt, "_read_lcbench_parquet", read)
    actual = lcbench.load_lcbench_data("Fashion-MNIST")
    expected = bt.load_lcbench_data("Fashion-MNIST")
    torch.testing.assert_close(
        actual.parameters, expected.parameters, atol=1e-14, rtol=1e-15, equal_nan=True
    )
    assert actual.parameters[0, 0].isneginf()
    assert actual.parameters[1, 2].isnan()
    assert actual.parameters[2, 4] < -400


def test_optional_dependency_errors_and_invalid_name_precedence(monkeypatch) -> None:
    original = importlib.import_module

    def blocked(module, *args, **kwargs):
        if module in ("jax.numpy", "pyarrow"):
            raise ImportError(f"no {module}")
        return original(module, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", blocked)
    with pytest.raises(ImportError, match="optional libraries"):
        jax_utils.torch_to_jax(object())
    with pytest.raises(ImportError, match="pandas and pyarrow"):
        lcbench.load_lcbench_data("Fashion-MNIST")
    with pytest.raises(ValueError, match="Invalid dataset"):
        lcbench.load_lcbench_data("../not-a-dataset")
