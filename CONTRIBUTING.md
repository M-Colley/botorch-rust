# Contributing to botorch-rust

Contributions are welcome. botorch-rust aims to reproduce BoTorch's numerics, so
every change to numerical code needs a parity test against BoTorch (see
below), and the Python API should keep BoTorch's names, argument order and
defaults.

## Repository layout

- `botorch_core`: the Rust implementation (models, MLL and fitting, L-BFGS-B,
  acquisition functions, `optimize_acqf`, sampling, test functions).
- `botorch_pyo3`: the Python bindings. `src/` is the PyO3 extension
  `botorch_rs._botorch_rs`; `python/botorch_rs` is the pure-Python package
  that mirrors BoTorch's module layout and handles array conversion;
  `tests/` holds the Python tests.
- `scripts`: fixture generation (`generate_fixtures.py`), the local build
  helper (`develop.py`) and benchmarks.

Run the commands below from the repository root.

## Rust

```console
$ cargo fmt --all --check
$ cargo clippy --workspace --all-targets -- -D warnings
$ cargo test --workspace
```

The parity tests (`botorch_core/tests/*parity*.rs`) compare against JSON
fixtures recorded from BoTorch. After changing what they cover, regenerate
them with BoTorch installed (or a source checkout on `PYTHONPATH`):

```console
$ python scripts/generate_fixtures.py [--phase 1|2|4|4b|5a-utils|5a-tf|5b|all]
```

## Python

Build the extension into the source tree and run the tests:

```console
$ python scripts/develop.py            # cargo build --release, copies the library
$ PYTHONPATH=botorch_pyo3/python python -m pytest botorch_pyo3/tests -q
```

Alternatively, `pip install ./botorch_pyo3` builds and installs a wheel with
maturin. The parity tests (`test_parity.py`, `test_parity_multi_output.py`,
`test_parity_constraints.py`, the unified suite `test_equivalence.py` and the
BoTorch part of `test_bo_loop.py`) run when BoTorch is importable and are
skipped otherwise; set `BOTORCH_RS_REQUIRE_BOTORCH=1` to make a missing
BoTorch an error. `-m equivalence` runs only the unified suite (about three
minutes). The worst error per quantity of the parity tests is printed in the
pytest summary, with the tolerance where one applies.

Lint and type-check (configuration in `botorch_pyo3/pyproject.toml`):

```console
$ ruff check --config botorch_pyo3/pyproject.toml botorch_pyo3/python botorch_pyo3/tests scripts/develop.py scripts/benchmark.py
$ ruff format --check --config botorch_pyo3/pyproject.toml botorch_pyo3/python botorch_pyo3/tests scripts/develop.py scripts/benchmark.py
$ (cd botorch_pyo3 && mypy python/botorch_rs)
```

## Guidelines

- Keep the Rust core free of Python concepts; the bindings only convert
  arguments and results. Release the GIL (`threads::run`) around any work.
- New Python arguments that botorch-rust cannot honor must raise
  `UnsupportedError` rather than being ignored silently (unless BoTorch also
  ignores them).
- Results must have the array type of the input they are computed from; do not
  import torch or NumPy unless the caller passed their arrays.
- Commit messages: an imperative title and a short body explaining why.
