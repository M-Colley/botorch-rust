//! Adds the linker arguments an extension module needs on macOS
//! (`-undefined dynamic_lookup`), so that `cargo build -p botorch_pyo3` produces
//! an importable library without maturin (see `scripts/develop.py`).

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
