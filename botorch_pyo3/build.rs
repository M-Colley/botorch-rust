//! Adds the linker arguments an extension module needs on macOS
//! (`-undefined dynamic_lookup`), so that `cargo build -p botorch_pyo3` produces
//! an importable library without maturin (see `scripts/develop.py`).

fn main() {
    // The linker flags depend on this script and Cargo's target/toolchain,
    // not Python sources or tests. Rust source dependencies are tracked by
    // rustc separately; Python-only wheel updates should not rebuild the DLL.
    println!("cargo:rerun-if-changed=build.rs");
    pyo3_build_config::add_extension_module_link_args();
}
