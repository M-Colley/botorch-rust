//! Multi-objective multi-fidelity optimization
//! (`botorch.acquisition.multi_objective.multi_fidelity.MOMF`): the
//! cost-weighted qEHVI of `botorch_core`, with BoTorch's default affine cost
//! `1 + s` of the fidelity `s` in the last input or a Python `cost_call` on
//! torch tensors (evaluated through `botorch_rs._robust_bridge`, gradients
//! from `torch.autograd`).

use std::sync::Arc;

use botorch_core::acquisition::multi_objective::multi_fidelity::{CostCall, Momf, MomfOptions};
use botorch_core::{BotorchError, Matrix};
use pyo3::prelude::*;

use crate::acquisition::{constraints_of, model_of, sampler_of, PyAcquisitionFunction};
use crate::arrays::{float_data, float_values, Array};
use crate::callbacks::callback_error;
use crate::errors::IntoPyResult;
use crate::input_perturbation::{row_major, ROBUST_BRIDGE};
use crate::mo_acquisition::{cells_of, mo_objective_of};
use crate::models::PyModel;
use crate::sampling::PySobolQMCNormalSampler;

/// A cost given by a Python callable `cost_call(X)` (`1 x n x d` points to
/// `n` costs).
#[derive(Debug)]
struct PyCostCall {
    callable: Py<PyAny>,
}

impl CostCall for PyCostCall {
    fn cost(
        &self,
        x: &Matrix,
        with_grad: bool,
    ) -> botorch_core::Result<(Vec<f64>, Option<Matrix>)> {
        let (n, d) = x.shape();
        let data = row_major(x);
        let (costs, grad) = Python::attach(|py| -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
            let bridge = py.import(ROBUST_BRIDGE)?;
            let res = bridge.call_method1(
                "costs",
                (
                    self.callable.bind(py),
                    float_values(py, &data)?,
                    n,
                    d,
                    with_grad,
                ),
            )?;
            let (costs, grad): (Bound<'_, PyAny>, Option<Bound<'_, PyAny>>) = res.extract()?;
            Ok((
                float_data(&costs)?,
                grad.map(|g| float_data(&g)).transpose()?,
            ))
        })
        .map_err(callback_error)?;
        let grad = match grad {
            Some(g) if g.len() == n * d => Some(Matrix::from_row_slice(n, d, &g)),
            Some(g) => {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the cost gradient has {} values, expected {}",
                    g.len(),
                    n * d
                )))
            }
            None => None,
        };
        Ok((costs, grad))
    }
}

/// `MOMF(model, ref_point, partitioning, sampler, objective, constraints,
/// eta, X_pending, cost_call)` with the hypercell bounds of the partitioning
/// (`2 x num_cells x (m + 1)`, the trust objective included).
#[pyfunction]
#[pyo3(signature = (
    model,
    ref_point,
    cells,
    sampler=None,
    x_pending=None,
    objective=None,
    constraints=None,
    eta=None,
    cost_call=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn momf(
    model: PyRef<'_, PyModel>,
    ref_point: Vec<f64>,
    cells: Array,
    sampler: Option<PyRef<'_, PySobolQMCNormalSampler>>,
    x_pending: Option<Array>,
    objective: Option<Bound<'_, PyAny>>,
    constraints: Option<Vec<Py<PyAny>>>,
    eta: Option<Vec<f64>>,
    cost_call: Option<Py<PyAny>>,
) -> PyResult<PyAcquisitionFunction> {
    let options = MomfOptions {
        sampler: sampler_of(sampler),
        objective: mo_objective_of(objective.as_ref())?,
        constraints: constraints_of(constraints, eta)?,
        x_pending: x_pending.map(|p| p.matrix("X_pending")).transpose()?,
        cost_call: cost_call.map(|c| Arc::new(PyCostCall { callable: c }) as Arc<dyn CostCall>),
    };
    let a = Momf::from_cell_bounds(model_of(&model), &ref_point, cells_of(&cells)?, options)
        .into_py_result()?;
    Ok(PyAcquisitionFunction::new_plain(Arc::new(a), "MOMF"))
}

/// Registers the functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(momf, m)?)?;
    Ok(())
}
