//! Pathwise posterior sampling (`botorch.sampling.pathwise`): Matheron
//! paths, prior paths and Gaussian updates of every output of a model, as
//! one [`PySamplePaths`] handle whose values the
//! `botorch_rs.sampling.pathwise` package arranges in BoTorch's shapes.
//!
//! Random numbers arrive as explicit draws (one `(frequencies, weights,
//! noise)` tuple per output, e.g. BoTorch's) or are drawn from a seed
//! (botorch-rust's generators).

use std::sync::Arc;

use botorch_core::sampling::pathwise::{
    draw_update_noise, gaussian_update, kernel_feature_path, matheron_path_draws, MatheronPath,
    MatheronPathDraws, MatheronPathModel, MatheronPathOptions, PathList, SamplePath,
};
use botorch_core::utils::sampling::random_seed;
use botorch_core::{BotorchError, Matrix, Model, Result};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::arrays::{matrix_out, Array, ArrayBuf};
use crate::errors::{value_error, IntoPyResult};
use crate::models::PyModel;
use crate::threads;

/// Explicit draws of one output: `(frequencies, weights, noise)`.
pub type DrawsArg = (Array, Array, Array);

/// The draws of every output from their arrays.
pub fn draws_of(draws: Vec<DrawsArg>) -> PyResult<Vec<MatheronPathDraws>> {
    draws
        .into_iter()
        .map(|(f, w, z)| {
            Ok(MatheronPathDraws {
                frequencies: f.matrix("frequencies")?,
                weights: w.matrix("weights")?,
                noise: z.matrix("noise")?,
            })
        })
        .collect()
}

/// What a [`PySamplePaths`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Matheron paths (`draw_matheron_paths`).
    Matheron,
    /// Prior paths with the outcome transform (`draw_kernel_feature_paths`).
    Prior,
    /// Gaussian updates (`gaussian_update`), in the transformed outcome space.
    Update,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Matheron => "matheron",
            Kind::Prior => "prior",
            Kind::Update => "update",
        }
    }
}

/// The sample paths of every output of a model: Matheron paths, prior paths
/// or Gaussian updates. Values are returned as `(m, S, N)` arrays (output,
/// sample, point) for the `N` points of `X` (all leading dimensions
/// flattened).
#[pyclass(frozen, name = "SamplePaths", module = "botorch_rs._botorch_rs")]
pub struct PySamplePaths {
    /// One path per output.
    pub paths: Arc<PathList>,
    kind: Kind,
    /// The prior and update paths of Matheron paths (`None` for other kinds
    /// and for the paths of `sample_optimal_points`).
    components: Option<(Arc<PathList>, Arc<PathList>)>,
    /// The random numbers of every output (Matheron and prior paths).
    draws: Vec<MatheronPathDraws>,
    /// The weights of the update paths of every output (`S × n`).
    update_weights: Vec<Matrix>,
    /// The noise of the updates of every output (`S × n`).
    noise: Vec<Matrix>,
}

fn gps_of(model: &dyn Model) -> Result<Vec<&botorch_core::SingleTaskGP>> {
    model.output_gps().ok_or_else(|| {
        BotorchError::Unsupported(
            "pathwise sampling requires a botorch_rs SingleTaskGP or ModelListGP".into(),
        )
    })
}

fn path_list(paths: impl Iterator<Item = Arc<dyn SamplePath>>) -> Result<Arc<PathList>> {
    Ok(Arc::new(PathList::new(paths.collect())?))
}

impl PySamplePaths {
    /// The Matheron paths of every output of `model`.
    fn matheron(
        model: &dyn Model,
        num_samples: usize,
        options: &MatheronPathOptions,
    ) -> Result<Self> {
        let draws = matheron_path_draws(model, num_samples, options)?;
        let gps = gps_of(model)?;
        let matheron = gps
            .par_iter()
            .zip(draws.par_iter())
            .map(|(gp, d)| MatheronPath::new(gp, d.clone()))
            .collect::<Result<Vec<_>>>()?;
        let prior = path_list(
            matheron
                .iter()
                .map(|p| Arc::new(p.prior_paths().clone()) as Arc<dyn SamplePath>),
        )?;
        let update = path_list(
            matheron
                .iter()
                .map(|p| Arc::new(p.update_paths().clone()) as Arc<dyn SamplePath>),
        )?;
        let update_weights = matheron
            .iter()
            .map(|p| p.update_paths().weight().clone())
            .collect();
        let paths = path_list(
            matheron
                .into_iter()
                .map(|p| Arc::new(p) as Arc<dyn SamplePath>),
        )?;
        Ok(Self {
            paths,
            kind: Kind::Matheron,
            components: Some((prior, update)),
            noise: draws.iter().map(|d| d.noise.clone()).collect(),
            draws,
            update_weights,
        })
    }

    /// The Matheron paths of a path model (of `sample_optimal_points`).
    pub fn from_path_model(model: MatheronPathModel) -> Self {
        Self {
            paths: Arc::new(model.paths().clone()),
            kind: Kind::Matheron,
            components: None,
            draws: Vec::new(),
            update_weights: Vec::new(),
            noise: Vec::new(),
        }
    }

    /// The prior paths (with the outcome transform) of every output.
    fn prior(model: &dyn Model, num_samples: usize, options: &MatheronPathOptions) -> Result<Self> {
        let draws = matheron_path_draws(model, num_samples, options)?;
        let gps = gps_of(model)?;
        let paths = gps
            .iter()
            .zip(&draws)
            .map(|(gp, d)| {
                kernel_feature_path(gp, d.frequencies.clone(), d.weights.clone(), true)
                    .map(|p| Arc::new(p) as Arc<dyn SamplePath>)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            paths: Arc::new(PathList::new(paths)?),
            kind: Kind::Prior,
            components: None,
            draws,
            update_weights: Vec::new(),
            noise: Vec::new(),
        })
    }

    /// The Gaussian updates of every output.
    fn update(model: &dyn Model, sample_values: &[Matrix], noise: Vec<Matrix>) -> Result<Self> {
        let gps = gps_of(model)?;
        if sample_values.len() != gps.len() || noise.len() != gps.len() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected sample values and noise for {} outputs, got {} and {}",
                gps.len(),
                sample_values.len(),
                noise.len()
            )));
        }
        let updates = gps
            .iter()
            .zip(sample_values.iter().zip(&noise))
            .map(|(gp, (v, z))| gaussian_update(gp, v, z))
            .collect::<Result<Vec<_>>>()?;
        let update_weights = updates.iter().map(|p| p.weight().clone()).collect();
        Ok(Self {
            paths: path_list(
                updates
                    .into_iter()
                    .map(|p| Arc::new(p) as Arc<dyn SamplePath>),
            )?,
            kind: Kind::Update,
            components: None,
            draws: Vec::new(),
            update_weights,
            noise,
        })
    }

    /// The paths of `component` (`None`: the paths themselves).
    fn component(&self, component: Option<&str>) -> PyResult<&Arc<PathList>> {
        let Some(name) = component else {
            return Ok(&self.paths);
        };
        let Some((prior, update)) = &self.components else {
            return Err(value_error(format!(
                "these paths have no {name:?} (only the Matheron paths of draw_matheron_paths do)"
            )));
        };
        match name {
            "prior_paths" => Ok(prior),
            "update_paths" => Ok(update),
            other => Err(value_error(format!(
                "unknown path {other:?} (expected \"prior_paths\" or \"update_paths\")"
            ))),
        }
    }
}

/// Values of every output as `(m, S, N)`.
fn path_values(values: &[Matrix]) -> ArrayBuf {
    let (s, n) = values.first().map_or((0, 0), |v| v.shape());
    let mut data = Vec::with_capacity(values.len() * s * n);
    for v in values {
        for k in 0..s {
            data.extend(v.row(k).iter());
        }
    }
    ArrayBuf(data, vec![values.len(), s, n])
}

fn matrices_out(ms: &[Matrix]) -> Vec<ArrayBuf> {
    ms.iter().map(|m| matrix_out(m).into()).collect()
}

#[pymethods]
impl PySamplePaths {
    /// The number of sample paths.
    #[getter]
    fn num_samples(&self) -> usize {
        self.paths.num_samples()
    }

    /// The number of outputs.
    #[getter]
    fn num_outputs(&self) -> usize {
        self.paths.num_outputs()
    }

    /// The input dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.paths.dim()
    }

    /// `"matheron"`, `"prior"` or `"update"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.name()
    }

    /// Whether the prior and update paths of Matheron paths are available.
    #[getter]
    fn has_components(&self) -> bool {
        self.components.is_some()
    }

    /// The values at the points of `x` (`... x d`, flattened to `N` points),
    /// shape `(m, S, N)`; with `component` (`"prior_paths"` or
    /// `"update_paths"`), the values of that part of Matheron paths, in the
    /// transformed outcome space (like BoTorch's sub-paths).
    #[pyo3(signature = (x, component=None))]
    fn forward(&self, py: Python<'_>, x: Array, component: Option<&str>) -> PyResult<ArrayBuf> {
        let (_, pts) = x.points("X")?;
        let paths = self.component(component)?;
        let values = threads::run(py, || paths.forward(&pts)).into_py_result()?;
        Ok(path_values(&values))
    }

    /// The values (`(m, S, N)`) and their derivatives with respect to the
    /// points (`(m, S, N, d)`; a path value depends only on its point).
    #[pyo3(signature = (x, component=None))]
    fn forward_with_grad(
        &self,
        py: Python<'_>,
        x: Array,
        component: Option<&str>,
    ) -> PyResult<(ArrayBuf, ArrayBuf)> {
        let (_, pts) = x.points("X")?;
        let paths = self.component(component)?;
        let (values, grads) =
            threads::run(py, || paths.forward_with_grad(&pts)).into_py_result()?;
        let (n, d) = pts.shape();
        let s = paths.num_samples();
        let mut g = Vec::with_capacity(values.len() * s * n * d);
        for gt in &grads {
            for gs in gt {
                for i in 0..n {
                    g.extend(gs.row(i).iter());
                }
            }
        }
        Ok((
            path_values(&values),
            ArrayBuf(g, vec![values.len(), s, n, d]),
        ))
    }

    /// The random numbers of every output: `(frequencies, weights, noise)`
    /// (`L/2 x d`, `S x L`, `S x n`); empty for Gaussian updates and the
    /// paths of `sample_optimal_points`.
    fn draws(&self) -> Vec<(ArrayBuf, ArrayBuf, ArrayBuf)> {
        self.draws
            .iter()
            .map(|d| {
                (
                    matrix_out(&d.frequencies).into(),
                    matrix_out(&d.weights).into(),
                    matrix_out(&d.noise).into(),
                )
            })
            .collect()
    }

    /// The weights of the update paths of every output (`S x n`).
    fn update_weights(&self) -> Vec<ArrayBuf> {
        matrices_out(&self.update_weights)
    }

    /// The standard normal noise of the updates of every output (`S x n`).
    fn noise(&self) -> Vec<ArrayBuf> {
        matrices_out(&self.noise)
    }
}

/// The options of the path functions from their arguments.
fn path_options(
    num_features: usize,
    seed: Option<u64>,
    draws: Option<Vec<MatheronPathDraws>>,
) -> MatheronPathOptions {
    MatheronPathOptions {
        num_features,
        seed,
        draws,
    }
}

/// `draw_matheron_paths(model, sample_shape)`: Matheron paths of every
/// output of `model` (`num_samples` samples, `num_features` Fourier
/// features), from `draws` (one `(frequencies, weights, noise)` per output)
/// or from `seed`.
#[pyfunction]
#[pyo3(name = "draw_matheron_paths", signature = (model, num_samples, num_features=1024, seed=None, draws=None))]
pub fn draw_matheron_paths_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    num_samples: usize,
    num_features: usize,
    seed: Option<u64>,
    draws: Option<Vec<DrawsArg>>,
) -> PyResult<PySamplePaths> {
    let m = Arc::clone(&model.inner);
    let options = path_options(num_features, seed, draws.map(draws_of).transpose()?);
    threads::run(py, || {
        PySamplePaths::matheron(m.as_ref(), num_samples, &options)
    })
    .into_py_result()
}

/// `draw_kernel_feature_paths(model, sample_shape)`: prior paths of every
/// output with the outcome transform, from `draws` (one `(frequencies,
/// weights)` per output) or from `seed`.
#[pyfunction]
#[pyo3(name = "draw_kernel_feature_paths", signature = (model, num_samples, num_features=1024, seed=None, draws=None))]
pub fn draw_kernel_feature_paths_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    num_samples: usize,
    num_features: usize,
    seed: Option<u64>,
    draws: Option<Vec<(Array, Array)>>,
) -> PyResult<PySamplePaths> {
    let m = Arc::clone(&model.inner);
    let draws = match draws {
        None => None,
        Some(ds) => {
            let gps = gps_of(m.as_ref()).into_py_result()?;
            if ds.len() != gps.len() {
                return Err(value_error(format!(
                    "expected prior path draws for {} outputs, got {}",
                    gps.len(),
                    ds.len()
                )));
            }
            Some(
                ds.into_iter()
                    .zip(&gps)
                    .map(|((f, w), gp)| {
                        let weights = w.matrix("weights")?;
                        Ok(MatheronPathDraws {
                            frequencies: f.matrix("frequencies")?,
                            noise: Matrix::zeros(weights.nrows(), gp.num_train()),
                            weights,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )
        }
    };
    let options = path_options(num_features, seed, draws);
    threads::run(py, || {
        PySamplePaths::prior(m.as_ref(), num_samples, &options)
    })
    .into_py_result()
}

/// `gaussian_update(model, sample_values)`: the pathwise updates of every
/// output given the values of prior paths at the transformed training
/// inputs (`S x n` per output), with the standard normal noise `noise` (`S x
/// n` per output) or noise drawn from `seed` (the noise of
/// `draw_matheron_paths` with the same seed).
#[pyfunction]
#[pyo3(name = "gaussian_update", signature = (model, sample_values, noise=None, seed=None))]
pub fn gaussian_update_py(
    py: Python<'_>,
    model: PyRef<'_, PyModel>,
    sample_values: Vec<Array>,
    noise: Option<Vec<Array>>,
    seed: Option<u64>,
) -> PyResult<PySamplePaths> {
    let m = Arc::clone(&model.inner);
    let values = sample_values
        .iter()
        .map(|v| v.matrix("sample_values"))
        .collect::<PyResult<Vec<_>>>()?;
    let noise = noise
        .map(|zs| {
            zs.iter()
                .map(|z| z.matrix("noise"))
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;
    threads::run(py, || {
        let noise = match noise {
            Some(z) => z,
            None => draw_update_noise(
                m.as_ref(),
                values.first().map_or(0, |v| v.nrows()),
                seed.unwrap_or_else(random_seed),
            )?,
        };
        PySamplePaths::update(m.as_ref(), &values, noise)
    })
    .into_py_result()
}

/// Registers the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySamplePaths>()?;
    m.add_function(wrap_pyfunction!(draw_matheron_paths_py, m)?)?;
    m.add_function(wrap_pyfunction!(draw_kernel_feature_paths_py, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_update_py, m)?)?;
    Ok(())
}
