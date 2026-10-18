//! Pathwise Thompson sampling (`botorch.acquisition.thompson_sampling`).
//!
//! [`PathwiseThompsonSampling`] evaluates `q` independent posterior sample
//! paths (Matheron paths, [`crate::sampling::pathwise`]): the value of a
//! candidate set is `Σ_j f_j(x_j)` for path `j` at point `j` (through the
//! posterior transform or the MC objective), so maximizing it yields `q`
//! Thompson samples. The paths are drawn at the first evaluation with the
//! candidate sets' `q` (BoTorch's `redraw(batch_size)`), from the generator
//! seeded by `seed` or from explicit draws.
//!
//! On an ensemble (the samples of a fully Bayesian GP,
//! [`PathwiseThompsonSampling::new_ensemble`]) every member draws `q` paths,
//! and path `j` of output `t` is the path of member `e[j][t]` for ensemble
//! indices drawn uniformly once per redraw (BoTorch's `ensemble_indices`
//! and `select_from_ensemble_models`).

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
use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};

/// The stream of the ensemble indices of a redraw (salted: the members'
/// paths use the streams `0..members`).
const ENSEMBLE_INDEX_STREAM: u64 = 0x656e_7365_6d62_6c65;

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

/// The drawn paths (one path model per member; one for a single model),
/// their batch size and, for ensembles, the member of every path and output
/// (`q × m`, row-major).
#[derive(Debug)]
struct Paths {
    batch_size: usize,
    models: Vec<MatheronPathModel>,
    ensemble_indices: Option<Vec<usize>>,
}

/// BoTorch's `PathwiseThompsonSampling`: see the [module
/// documentation](self).
#[derive(Debug)]
pub struct PathwiseThompsonSampling {
    model: Arc<dyn Model>,
    /// The members of an ensemble (empty for a single model).
    members: Vec<Arc<dyn Model>>,
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
            members: Vec::new(),
            objective: options.objective,
            transform: options.posterior_transform,
            num_features: options.num_features,
            seed: options.seed.unwrap_or_else(random_seed),
            draws_count: RwLock::new(0),
            paths: RwLock::new(None),
        })
    }

    /// `PathwiseThompsonSampling` of an ensemble model (the samples of a
    /// fully Bayesian GP, BoTorch's `is_ensemble`): `members` are its
    /// unbatched members with the same inputs and outputs.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] without members or for members of
    /// different dimensions, and the errors of [`Self::new`].
    pub fn new_ensemble(
        members: Vec<Arc<dyn Model>>,
        options: PathwiseThompsonSamplingOptions,
    ) -> Result<Self> {
        let Some(first) = members.first() else {
            return Err(BotorchError::InvalidArgument(
                "an ensemble needs at least one member".into(),
            ));
        };
        for m in &members {
            check_unbatched(m.as_ref())?;
            if m.dim() != first.dim() || m.num_outputs() != first.num_outputs() {
                return Err(BotorchError::InvalidArgument(
                    "the members of an ensemble must have the same inputs and outputs".into(),
                ));
            }
        }
        let mut a = Self::new(Arc::clone(first), options)?;
        a.members = members;
        Ok(a)
    }

    /// The number of members (1 for a single model).
    fn num_members(&self) -> usize {
        self.members.len().max(1)
    }

    /// BoTorch's `redraw(batch_size)`: draws `batch_size` new paths (from
    /// the next stream of the seed, or from explicit `draws`, one per
    /// output with `batch_size` samples each; for an ensemble one per member
    /// and output, member-major).
    ///
    /// # Errors
    /// Errors of [`get_matheron_path_model`].
    pub fn redraw(&self, batch_size: usize, draws: Option<Vec<MatheronPathDraws>>) -> Result<()> {
        self.redraw_ensemble(batch_size, draws, None)
    }

    /// [`Self::redraw`] with explicit ensemble indices (`q × m`, row-major:
    /// the member of path `j` of output `t`; drawn uniformly from the seed
    /// if `None`). Ignored for a single model.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for ensemble indices of the wrong
    /// size or out of range, draws of the wrong number, and the errors of
    /// [`get_matheron_path_model`].
    pub fn redraw_ensemble(
        &self,
        batch_size: usize,
        draws: Option<Vec<MatheronPathDraws>>,
        ensemble_indices: Option<Vec<usize>>,
    ) -> Result<()> {
        let paths = self.draw(batch_size, draws, ensemble_indices)?;
        *write_lock(&self.paths) = Some(paths);
        Ok(())
    }

    fn draw(
        &self,
        batch_size: usize,
        draws: Option<Vec<MatheronPathDraws>>,
        ensemble_indices: Option<Vec<usize>>,
    ) -> Result<Paths> {
        let stream = {
            let mut c = write_lock(&self.draws_count);
            *c += 1;
            *c
        };
        let seed = derive_seed(self.seed, stream);
        if self.members.is_empty() {
            let model = get_matheron_path_model(
                self.model.as_ref(),
                batch_size,
                &MatheronPathOptions {
                    num_features: self.num_features,
                    seed: Some(seed),
                    draws,
                },
            )?;
            return Ok(Paths {
                batch_size,
                models: vec![model],
                ensemble_indices: None,
            });
        }
        let (s_count, m) = (self.members.len(), self.model.num_outputs());
        let mut per_member: Vec<Option<Vec<MatheronPathDraws>>> = match draws {
            None => vec![None; s_count],
            Some(d) => {
                if d.len() != s_count * m {
                    return Err(BotorchError::InvalidArgument(format!(
                        "an ensemble of {s_count} members with {m} outputs needs {} draws, got {}",
                        s_count * m,
                        d.len()
                    )));
                }
                let mut it = d.into_iter();
                (0..s_count)
                    .map(|_| Some(it.by_ref().take(m).collect()))
                    .collect()
            }
        };
        let models = self
            .members
            .iter()
            .enumerate()
            .map(|(s, member)| {
                get_matheron_path_model(
                    member.as_ref(),
                    batch_size,
                    &MatheronPathOptions {
                        num_features: self.num_features,
                        seed: Some(derive_seed(seed, s as u64)),
                        draws: per_member[s].take(),
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = match ensemble_indices {
            Some(e) => {
                if e.len() != batch_size * m || e.iter().any(|&i| i >= s_count) {
                    return Err(BotorchError::InvalidArgument(format!(
                        "the ensemble indices must be {batch_size} x {m} members in 0..{s_count}"
                    )));
                }
                e
            }
            None => {
                let mut rng =
                    Xoshiro256PlusPlus::seed_from_u64(derive_seed(seed, ENSEMBLE_INDEX_STREAM));
                (0..batch_size * m)
                    .map(|_| rng.random_range(0..s_count))
                    .collect()
            }
        };
        Ok(Paths {
            batch_size,
            models,
            ensemble_indices: Some(indices),
        })
    }

    /// The ensemble indices of the drawn paths (`q × m`, row-major; `None`
    /// for a single model or before the first draw).
    pub fn ensemble_indices(&self) -> Option<Vec<usize>> {
        read_lock(&self.paths)
            .as_ref()
            .and_then(|p| p.ensemble_indices.clone())
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
                *guard = Some(self.draw(q, None, None)?);
            }
        }
        let guard = read_lock(&self.paths);
        let paths = guard.as_ref().expect("drawn");
        if paths.batch_size != q {
            return Err(batch_size_change_error(paths.batch_size, q));
        }
        let m = self.model.num_outputs();
        // The member of path j of output t (0 for a single model).
        let member =
            |j: usize, t: usize| paths.ensemble_indices.as_ref().map_or(0, |e| e[j * m + t]);
        let used: Vec<bool> = (0..self.num_members())
            .map(|s| (0..q).any(|j| (0..m).any(|t| member(j, t) == s)))
            .collect();
        let mut v = Matrix::zeros(q, m);
        let mut g = with_grad.then(|| vec![Matrix::zeros(q, x.ncols()); m]);
        for (s, pm) in paths.models.iter().enumerate() {
            if !used[s] {
                continue;
            }
            let pl = pm.paths();
            if let Some(g) = g.as_mut() {
                let (values, grads) = pl.forward_with_grad(x)?;
                for j in 0..q {
                    for t in (0..m).filter(|&t| member(j, t) == s) {
                        v[(j, t)] = values[t][(j, j)];
                        for k in 0..x.ncols() {
                            g[t][(j, k)] = grads[t][j][(j, k)];
                        }
                    }
                }
            } else {
                let values = pl.forward(x)?;
                for j in 0..q {
                    for t in (0..m).filter(|&t| member(j, t) == s) {
                        v[(j, t)] = values[t][(j, j)];
                    }
                }
            }
        }
        Ok((v, g))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SingleTaskGP, SingleTaskGPOptions};

    fn model(shift: f64) -> Arc<dyn Model> {
        let x = Matrix::from_fn(7, 2, |i, j| ((i * 3 + j * 5) % 7) as f64 / 7.0);
        let y = Matrix::from_fn(7, 1, |i, _| (5.0 * x[(i, 0)]).sin() - x[(i, 1)] + shift);
        Arc::new(SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap())
    }

    /// Deterministic draws of `q` paths with 8 features on 7 points.
    fn draws(q: usize, salt: f64) -> MatheronPathDraws {
        let f = |i: usize, j: usize, k: f64| ((i * 7 + j * 3) as f64 * 0.37 + k + salt).sin();
        MatheronPathDraws {
            frequencies: Matrix::from_fn(4, 2, |i, j| f(i, j, 0.1)),
            weights: Matrix::from_fn(q, 8, |i, j| f(i, j, 0.2)),
            noise: Matrix::from_fn(q, 7, |i, j| f(i, j, 0.3)),
        }
    }

    fn options() -> PathwiseThompsonSamplingOptions {
        PathwiseThompsonSamplingOptions {
            num_features: 8,
            seed: Some(3),
            ..Default::default()
        }
    }

    /// Path `j` of an ensemble is the path `j` of member `e[j]`: the value
    /// at a candidate set is the sum of the members' single-model values
    /// at the points their paths are selected for.
    #[test]
    fn ensemble_paths_are_the_selected_members_paths() {
        let members: Vec<Arc<dyn Model>> = (0..3).map(|s| model(0.4 * s as f64)).collect();
        let q = 3;
        let ens = PathwiseThompsonSampling::new_ensemble(members.clone(), options()).unwrap();
        let all: Vec<MatheronPathDraws> = (0..3).map(|s| draws(q, s as f64)).collect();
        let e = vec![2, 0, 2];
        ens.redraw_ensemble(q, Some(all.clone()), Some(e.clone()))
            .unwrap();
        assert_eq!(ens.ensemble_indices(), Some(e.clone()));
        let x = Matrix::from_fn(q, 2, |i, j| 0.1 + 0.27 * i as f64 + 0.13 * j as f64);
        let (v, g) = ens.forward_with_grad(&x).unwrap();
        let mut expected = 0.0;
        let mut expected_grad = Matrix::zeros(q, 2);
        for (s, member) in members.iter().enumerate() {
            let single = PathwiseThompsonSampling::new(Arc::clone(member), options()).unwrap();
            single.redraw(q, Some(vec![all[s].clone()])).unwrap();
            let (_, gs) = single.forward_with_grad(&x).unwrap();
            let values = single.path_values(&x, false).unwrap().0;
            for j in (0..q).filter(|&j| e[j] == s) {
                expected += values[(j, 0)];
                for k in 0..2 {
                    expected_grad[(j, k)] = gs[(j, k)];
                }
            }
        }
        assert!((v - expected).abs() < 1e-12 * expected.abs().max(1.0));
        assert!((&g - &expected_grad).abs().max() < 1e-12);
        // Random indices are reproducible and in range.
        ens.redraw(q, None).unwrap();
        let drawn = ens.ensemble_indices().unwrap();
        assert_eq!(drawn.len(), q);
        assert!(drawn.iter().all(|&i| i < 3));
        let again = PathwiseThompsonSampling::new_ensemble(members.clone(), options()).unwrap();
        again.redraw(q, None).unwrap();
        again.redraw(q, None).unwrap();
        assert_eq!(again.ensemble_indices().unwrap(), drawn);
        // Errors: indices out of range, wrong number of draws.
        assert!(ens.redraw_ensemble(q, None, Some(vec![0, 3, 1])).is_err());
        assert!(ens
            .redraw_ensemble(q, Some(all[..2].to_vec()), None)
            .is_err());
        assert!(PathwiseThompsonSampling::new_ensemble(Vec::new(), options()).is_err());
    }
}
