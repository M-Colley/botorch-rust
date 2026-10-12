//! Python bindings of botorch-rust: the `botorch_rs._botorch_rs` extension
//! module.
//!
//! This crate is a thin layer over `botorch_core`. Arrays cross the boundary
//! as `(data, shape)` tuples of plain Python floats ([`arrays`]), so the
//! extension depends on neither NumPy nor torch; the pure-Python
//! `botorch_rs` package (`python/botorch_rs`) converts torch tensors, NumPy
//! arrays and nested lists, and provides BoTorch's module layout, class names
//! and argument handling on top of the classes defined here. Errors map onto
//! the exception hierarchy of `botorch_rs.exceptions` ([`errors`]), and every
//! computation runs with the GIL released ([`threads`]).

use pyo3::prelude::*;

mod acquisition;
mod arrays;
mod callbacks;
mod errors;
mod input_perturbation;
mod mo_acquisition;
mod mo_test_functions;
mod models;
mod momf;
mod multi_objective;
mod optim;
mod polytope;
mod risk_measures;
mod sampling;
mod specs;
mod test_functions;
mod threads;
mod transforms;

/// The `botorch_rs._botorch_rs` extension module.
#[pymodule]
#[pyo3(name = "_botorch_rs")]
fn botorch_rs_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Models and their components.
    m.add_class::<models::PySingleTaskGP>()?;
    m.add_class::<models::PyModelListGP>()?;
    m.add_class::<models::PyModel>()?;
    m.add_class::<transforms::PyNormalize>()?;
    m.add_class::<transforms::PyStandardize>()?;
    m.add_class::<specs::PyPrior>()?;
    m.add_class::<specs::PyGreaterThan>()?;
    m.add_class::<specs::PyKernelSpec>()?;
    m.add_class::<specs::PyLikelihoodSpec>()?;
    // Acquisition functions and their optimization.
    m.add_class::<acquisition::PyAcquisitionFunction>()?;
    m.add_function(wrap_pyfunction!(acquisition::prune_inferior_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(optim::optimize_acqf, m)?)?;
    m.add_function(wrap_pyfunction!(optim::gen_batch_initial_conditions_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        optim::sample_q_batches_from_polytope_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(optim::evaluate_feasibility_py, m)?)?;
    // Polytope sampling.
    m.add_class::<polytope::PyHitAndRunPolytopeSampler>()?;
    m.add_function(wrap_pyfunction!(polytope::get_polytope_samples_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        polytope::sparse_to_dense_constraints_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(polytope::find_interior_point_py, m)?)?;
    m.add_function(wrap_pyfunction!(polytope::sample_polytope_py, m)?)?;
    // Sampling.
    m.add_class::<sampling::PySobolQMCNormalSampler>()?;
    m.add_function(wrap_pyfunction!(sampling::draw_sobol_samples_py, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::draw_sobol_normal_samples_py, m)?)?;
    // Test functions.
    m.add_class::<test_functions::PySyntheticTestFunction>()?;
    m.add_function(wrap_pyfunction!(test_functions::test_function_names, m)?)?;
    // Multi-objective test problems.
    m.add_class::<mo_test_functions::PyMultiObjectiveTestProblem>()?;
    m.add_function(wrap_pyfunction!(
        mo_test_functions::multi_objective_problem_names,
        m
    )?)?;
    // Threads.
    m.add_function(wrap_pyfunction!(threads::set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(threads::get_num_threads, m)?)?;
    // Multi-objective utilities, `sample_simplex` and `sample_hypersphere`.
    multi_objective::register(m)?;
    // Multi-objective acquisition functions.
    mo_acquisition::register(m)?;
    // Robust and multi-fidelity multi-objective optimization: input
    // perturbations, risk measures, MOMF, NSGA-II.
    input_perturbation::register(m)?;
    risk_measures::register(m)?;
    momf::register(m)?;
    Ok(())
}
