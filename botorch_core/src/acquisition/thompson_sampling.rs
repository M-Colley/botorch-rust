//! Pathwise Thompson sampling (`botorch.acquisition.thompson_sampling`).
//!
//! [`PathwiseThompsonSampling`] evaluates `q` independent posterior sample
//! paths (Matheron paths, [`crate::sampling::pathwise`]): the value of a
//! candidate set is `Σ_j f_j(x_j)` for path `j` at point `j` (through the
//! posterior transform or the MC objective), so maximizing it yields `q`
//! Thompson samples. The paths are drawn at the first evaluation with the
//! candidate sets' `q` (BoTorch's `redraw(batch_size)`), from the generator
//! seeded by `seed` or from explicit draws.

use std::sync::{Arc, RwLock};

use crate::acquisition::mc_objective::{McObjective, McSamples};
use crate::acquisition::monte_carlo::{read_lock, write_lock, MULTI_OUTPUT_MESSAGE};
use crate::acquisition::objective::ScalarizedPosteriorTransform;
use crate::acquisition::{check_candidates, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::sampling::pathwise::utils::derive_seed;
use crate::sampling::pathwise::{
    get_matheron_path_model, MatheronPathDraws, MatheronPathModel, MatheronPathOptions, SamplePath,
    DEFAULT_NUM_FEATURES,
};
use crate::utils::sampling::random_seed;
use crate::Matrix;

/// BoTorch's error when the `q` of the candidate sets changes.
fn batch_size_change_error(old: usize, new: usize) -> BotorchError {
    BotorchError::InvalidArgument(format!(
        "The batch size of PathwiseThompsonSampling should not change during a forward pass - \
         was {old}, now {new}. Please re-initialize the acquisition if you want to change the \
         batch size."
    ))
}

/// The adjoint of the path values (`q × m`) from that of the transformed
/// values (`q × 1`).
type ValuePullback = Box<dyn Fn(&Matrix) -> Matrix>;

/// Options of [`PathwiseThompsonSampling`].
#[derive(Clone, Debug)]
pub struct PathwiseThompsonSamplingOptions {
    /// The MC objective applied to the path values (default: the identity;
    /// required for multi-output models without a posterior transform).
    pub objective: Option<Arc<dyn McObjective>>,
    /// A scalarizing posterior transform applied to the path values.
    pub posterior_transform: Option<ScalarizedPosteriorTransform>,
    /// The number of Fourier features of the paths (default 1024).
    pub num_features: usize,
    /// The seed of the paths' random numbers (default: random); every
    /// redraw uses a new stream of it.
    pub seed: Option<u64>,
}

impl Default for PathwiseThompsonSamplingOptions {
    fn default() -> Self {
        Self {
            objective: None,
            posterior_transform: None,
            num_features: DEFAULT_NUM_FEATURES,
            seed: None,
        }
    }
}

/// The drawn paths and their batch size.
#[derive(Debug)]
struct Paths {
    batch_size: usize,
    model: MatheronPathModel,
}

/// BoTorch's `PathwiseThompsonSampling`: see the [module
/// documentation](self).
#[derive(Debug)]
pub struct PathwiseThompsonSampling {
    model: Arc<dyn Model>,
    objective: Option<Arc<dyn McObjective>>,
    transform: Option<ScalarizedPosteriorTransform>,
    num_features: usize,
    seed: u64,
    draws_count: RwLock<u64>,
    paths: RwLock<Option<Paths>>,
}

impl PathwiseThompsonSampling {
    /// `PathwiseThompsonSampling(model, objective, posterior_transform)`.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for multi-output models without an
    /// objective or a posterior transform (BoTorch's message) and batched
    /// models.
    pub fn new(model: Arc<dyn Model>, options: PathwiseThompsonSamplingOptions) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        if options.objective.is_none()
            && model.num_outputs() != 1
            && options.posterior_transform.is_none()
        {
            return Err(BotorchError::Unsupported(MULTI_OUTPUT_MESSAGE.into()));
        }
        if let Some(t) = &options.posterior_transform {
            if t.num_outputs() != model.num_outputs() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Output shape not equal to that of weights. Output shape is {} and weights \
                     are [{}]",
                    model.num_outputs(),
                    t.num_outputs()
                )));
            }
        }
        Ok(Self {
            model,
            objective: options.objective,
            transform: options.posterior_transform,
            num_features: options.num_features,
            seed: options.seed.unwrap_or_else(random_seed),
            draws_count: RwLock::new(0),
            paths: RwLock::new(None),
        })
    }

    /// BoTorch's `redraw(batch_size)`: draws `batch_size` new paths (from
    /// the next stream of the seed, or from explicit `draws`, one per
    /// output with `batch_size` samples each).
    ///
    /// # Errors
    /// Errors of [`get_matheron_path_model`].
    pub fn redraw(&self, batch_size: usize, draws: Option<Vec<MatheronPathDraws>>) -> Result<()> {
        let paths = self.draw(batch_size, draws)?;
        *write_lock(&self.paths) = Some(paths);
        Ok(())
    }

    fn draw(&self, batch_size: usize, draws: Option<Vec<MatheronPathDraws>>) -> Result<Paths> {
        let stream = {
            let mut c = write_lock(&self.draws_count);
            *c += 1;
            *c
        };
        let model = get_matheron_path_model(
            self.model.as_ref(),
            batch_size,
            &MatheronPathOptions {
                num_features: self.num_features,
                seed: Some(derive_seed(self.seed, stream)),
                draws,
            },
        )?;
        Ok(Paths { batch_size, model })
    }

    /// The batch size of the drawn paths (`None` before the first draw).
    pub fn batch_size(&self) -> Option<usize> {
        read_lock(&self.paths).as_ref().map(|p| p.batch_size)
    }

    /// The paths' values at the rows of `x` (one sample per point: path `j`
    /// at point `j`), before the transform: a `q × m` matrix and, with
    /// gradients, the per-output `q × d` derivatives.
    fn path_values(&self, x: &Matrix, with_grad: bool) -> Result<(Matrix, Option<Vec<Matrix>>)> {
        let q = x.nrows();
        {
            // Draw atomically (parallel evaluations of the first batch).
            let mut guard = write_lock(&self.paths);
            if guard.is_none() {
                *guard = Some(self.draw(q, None)?);
            }
        }
        let guard = read_lock(&self.paths);
        let paths = guard.as_ref().expect("drawn");
        if paths.batch_size != q {
            return Err(batch_size_change_error(paths.batch_size, q));
        }
        let pl = paths.model.paths();
        if with_grad {
            let (values, grads) = pl.forward_with_grad(x)?;
            let m = values.len();
            let v = Matrix::from_fn(q, m, |j, t| values[t][(j, j)]);
            let g = (0..m)
                .map(|t| Matrix::from_fn(q, x.ncols(), |j, k| grads[t][j][(j, k)]))
                .collect();
            Ok((v, Some(g)))
        } else {
            let values = pl.forward(x)?;
            let m = values.len();
            Ok((Matrix::from_fn(q, m, |j, t| values[t][(j, j)]), None))
        }
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        check_candidates(x, self.model.dim(), None)?;
        let (q, d) = x.shape();
        let (v, g) = self.path_values(x, with_grad)?;
        let m = v.ncols();
        // The posterior transform (one output) and the objective (identity).
        let (y, y_bar_of): (Matrix, ValuePullback) = match &self.transform {
            Some(t) => {
                let yt = t.evaluate(&v)?;
                let w = t.weights().clone();
                (
                    Matrix::from_column_slice(q, 1, yt.as_slice()),
                    Box::new(move |bar: &Matrix| Matrix::from_fn(q, m, |j, k| bar[(j, 0)] * w[k])),
                )
            }
            None => (v.clone(), Box::new(|bar: &Matrix| bar.clone())),
        };
        let my = y.ncols();
        let data = Matrix::from_fn(my * q, 1, |r, _| y[(r % q, r / q)]);
        let samples = McSamples::new(data, q, my)?;
        let (obj, pullback) = match &self.objective {
            Some(o) => {
                let out = o.evaluate(&samples, None, with_grad)?;
                (out.values, out.pullback)
            }
            None => {
                if my != 1 {
                    return Err(BotorchError::Unsupported(MULTI_OUTPUT_MESSAGE.into()));
                }
                (samples.data().clone(), None)
            }
        };
        if obj.shape() != (q, 1) {
            return Err(BotorchError::DimensionMismatch(format!(
                "the objective returned {} values for {q} points",
                obj.nrows()
            )));
        }
        let value = obj.sum();
        if !with_grad {
            return Ok((value, None));
        }
        let ones = Matrix::from_element(q, 1, 1.0);
        let (sbar, xbar) = match pullback {
            Some(pb) => pb(&ones)?,
            None => (ones, None),
        };
        // Adjoint of the transformed values (q × my) and of the path values.
        let ybar = Matrix::from_fn(q, my, |j, k| sbar[(k * q + j, 0)]);
        let vbar = y_bar_of(&ybar);
        let grads = g.expect("gradient requested");
        let mut grad = Matrix::zeros(q, d);
        for (t, gt) in grads.iter().enumerate() {
            for j in 0..q {
                for k in 0..d {
                    grad[(j, k)] += vbar[(j, t)] * gt[(j, k)];
                }
            }
        }
        if let Some(xb) = xbar {
            grad += xb;
        }
        Ok((value, Some(grad)))
    }
}

impl AcquisitionFunction for PathwiseThompsonSampling {
    fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }
}
