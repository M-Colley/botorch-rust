//! Run with `cargo run --release -p botorch_core --example decision_making`.

use botorch_core::utils::multi_objective::decision_making::select_preferred;
use botorch_core::utils::multi_objective::fpbi::{generate_fpbi_front, FpbiOptions};
use botorch_core::{Matrix, Result};

fn main() -> Result<()> {
    let outcomes = Matrix::from_row_slice(4, 2, &[0., 1., 0.2, 0.4, 0.4, 0.2, 1., 0.]);
    let index = select_preferred(&outcomes, &[70., 30.], 0.8, false)?;
    println!("Preferred original outcome row: {index}");

    let objective = |x: &[f64]| -> Result<Vec<f64>> { Ok(vec![x[0].powi(2), (1. - x[0]).powi(2)]) };
    let bounds = Matrix::from_row_slice(2, 1, &[0., 1.]);
    let starts = Matrix::from_row_slice(3, 1, &[0.25, 0.5, 0.75]);
    let options = FpbiOptions {
        maximize: false,
        grid_resolution: 5,
        ..Default::default()
    };
    let front = generate_fpbi_front(&objective, &bounds, &starts, None, None, None, &options)?;
    println!("Sampled {} nondominated decisions", front.x.nrows());
    for row in 0..front.x.nrows() {
        let costs: Vec<f64> = front.y.row(row).iter().copied().collect();
        println!("x={} costs={costs:?}", front.x[(row, 0)]);
    }
    Ok(())
}
