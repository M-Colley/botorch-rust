"""Load and atomically cache optional LCBench Parquet data."""

from __future__ import annotations

import importlib
import io
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botorch_rs import _ops

LCBENCH_DATASET_NAMES = (
    "APSFailure",
    "Amazon_employee_access",
    "Australian",
    "Fashion-MNIST",
    "KDDCup09_appetency",
    "MiniBooNE",
    "adult",
    "airlines",
    "albert",
    "bank-marketing",
    "blood-transfusion-service-center",
    "car",
    "christine",
    "cnae-9",
    "connect-4",
    "covertype",
    "credit-g",
    "dionis",
    "fabert",
    "helena",
    "higgs",
    "jannis",
    "jasmine",
    "jungle_chess_2pcs_raw_endgame_complete",
    "kc1",
    "kr-vs-kp",
    "mfeat-factors",
    "nomao",
    "numerai28.6",
    "phoneme",
    "segment",
    "shuttle",
    "sylvine",
    "vehicle",
    "volkert",
)
LCBENCH_PARAMETER_NAMES = (
    "batch_size",
    "max_dropout",
    "max_units",
    "num_layers",
    "learning_rate",
    "momentum",
    "weight_decay",
)
LCBENCH_LOG_SCALE_PARAMETER_NAMES = ("batch_size", "max_units", "learning_rate", "momentum")
DEFAULT_LCBENCH_METRIC_NAME = "Train/val_accuracy"
LCBENCH_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/ltiao/LCBenchLite/main/{dataset_name}/{stem}.parquet.gzip"
)
DEFAULT_LCBENCH_CACHE_DIR = "~/.cache/botorch/lcbench"
DEFAULT_LCBENCH_DOWNLOAD_TIMEOUT = 60.0


def _parquet_dependencies() -> Any:
    try:
        pandas = importlib.import_module("pandas")
        importlib.import_module("pyarrow")
    except ImportError as error:
        raise ImportError(
            "The LCBench loader requires optional pandas and pyarrow dependencies. "
            "Install them with `pip install pandas pyarrow`."
        ) from error
    return pandas


def _default_cache_dir() -> Path:
    return Path(DEFAULT_LCBENCH_CACHE_DIR).expanduser()


@dataclass(frozen=True, kw_only=True)
class LCBenchData:
    """Configurations, learning-curve metrics and epoch indices as torch tensors."""

    parameters: Any
    metrics: Any
    epochs: Any


def _read_lcbench_parquet(dataset_name: str, stem: str, cache_dir: Path) -> Any:
    pandas = _parquet_dependencies()
    cache_path = cache_dir / dataset_name / f"{stem}.parquet.gzip"
    if cache_path.exists():
        return pandas.read_parquet(cache_path, engine="pyarrow")
    url = LCBENCH_URL_TEMPLATE.format(dataset_name=dataset_name, stem=stem)
    with urllib.request.urlopen(url, timeout=DEFAULT_LCBENCH_DOWNLOAD_TIMEOUT) as response:
        raw = response.read()
    # Validate the complete Parquet payload before publishing a cache entry.
    frame = pandas.read_parquet(io.BytesIO(raw), engine="pyarrow")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent, suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(raw)
        os.replace(temporary, cache_path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return frame


def load_lcbench_data(
    dataset_name: str,
    metric_name: str = DEFAULT_LCBENCH_METRIC_NAME,
    dtype: Any = None,
    device: Any = None,
    cache_dir: Path | None = None,
) -> LCBenchData:
    """Load a dataset, log-transform its search-space columns in Rust, and cache IO.

    ``dtype=None`` selects torch.float64, matching BoTorch's default without
    importing torch at module import time.
    """
    if dataset_name not in LCBENCH_DATASET_NAMES:
        raise ValueError(
            f"Invalid dataset '{dataset_name}'. Valid datasets: {list(LCBENCH_DATASET_NAMES)}."
        )
    _parquet_dependencies()
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise ImportError(
            "LCBench tensor output requires torch; install it with `pip install torch`."
        ) from error
    if dtype is None:
        dtype = torch.float64
    directory = _default_cache_dir() if cache_dir is None else Path(cache_dir)
    parameter_frame = _read_lcbench_parquet(dataset_name, "config", directory)
    metrics_frame = _read_lcbench_parquet(dataset_name, "metrics", directory)
    parameters = [
        list(row)
        for row in parameter_frame[list(LCBENCH_PARAMETER_NAMES)].itertuples(index=False, name=None)
    ]
    log_columns = [
        LCBENCH_PARAMETER_NAMES.index(name) for name in LCBENCH_LOG_SCALE_PARAMETER_NAMES
    ]
    raw_logs = [float(row[column]) for row in parameters for column in log_columns]
    logs = _ops.apply("log_raw", raw_logs)
    for row, values in enumerate(parameters):
        for index, column in enumerate(log_columns):
            values[column] = logs[row * len(log_columns) + index]
    metric_frame = metrics_frame[metric_name].unstack(level="epoch").iloc[:, 1:-1]
    # Recent pandas exposes read-only NumPy views. Copy so the returned
    # tensors are safely writable instead of aliasing read-only storage.
    metrics = torch.tensor(metric_frame.values, dtype=dtype, device=device)
    parameter_tensor = torch.tensor(parameters, dtype=dtype, device=device)
    if not parameters:
        parameter_tensor = parameter_tensor.reshape(0, len(LCBENCH_PARAMETER_NAMES))
    return LCBenchData(
        parameters=parameter_tensor,
        metrics=metrics,
        epochs=torch.arange(1, metrics.shape[-1] + 1, dtype=dtype, device=device),
    )


__all__ = [
    "LCBenchData",
    "load_lcbench_data",
    "LCBENCH_DATASET_NAMES",
    "LCBENCH_PARAMETER_NAMES",
    "LCBENCH_LOG_SCALE_PARAMETER_NAMES",
    "DEFAULT_LCBENCH_METRIC_NAME",
]
