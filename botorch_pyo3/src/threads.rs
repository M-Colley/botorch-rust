//! Running heavy work without the GIL on a configurable thread pool.
//!
//! Every computational entry point runs through [`run`], which releases the
//! GIL (`Python::detach`) so that other Python threads keep running, and
//! executes the work on the pool chosen with `set_num_threads` (rayon's global
//! pool by default, i.e. one thread per logical CPU unless `RAYON_NUM_THREADS`
//! is set). Multi-start acquisition optimization and batch evaluations are
//! parallelized over this pool.

use std::sync::{Arc, RwLock};

use pyo3::prelude::*;

use crate::errors::value_error;

static POOL: RwLock<Option<Arc<rayon::ThreadPool>>> = RwLock::new(None);

fn current_pool() -> Option<Arc<rayon::ThreadPool>> {
    POOL.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Runs `f` with the GIL released, on the configured thread pool. Python
/// callables invoked by `f` reacquire the GIL; the first exception they raise
/// is kept for [`crate::errors::to_pyerr`].
pub fn run<T, F>(py: Python<'_>, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    crate::callbacks::clear_error();
    let pool = current_pool();
    py.detach(move || match pool {
        Some(pool) => pool.install(f),
        None => f(),
    })
}

/// Sets the number of threads used by botorch_rs (`None` or 0 restores the
/// default: one thread per logical CPU, or `RAYON_NUM_THREADS`).
#[pyfunction]
#[pyo3(signature = (num_threads=None))]
pub fn set_num_threads(num_threads: Option<usize>) -> PyResult<()> {
    let pool = match num_threads {
        None | Some(0) => None,
        Some(n) => Some(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|i| format!("botorch-rs-{i}"))
                .build()
                .map_err(|e| value_error(format!("cannot create a thread pool: {e}")))?,
        )),
    };
    *POOL.write().unwrap_or_else(|e| e.into_inner()) = pool;
    Ok(())
}

/// The number of threads used by botorch_rs.
#[pyfunction]
pub fn get_num_threads() -> usize {
    match current_pool() {
        Some(pool) => pool.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}
