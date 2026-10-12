//! Sampled Pareto sets and fronts for the information-theoretic acquisition
//! functions (the sampling utilities of
//! `botorch.acquisition.multi_objective.utils`):
//!
//! * [`sample_optimal_points`]: Pareto sets and fronts of Matheron paths of
//!   the posterior ([`crate::sampling::pathwise`]), optimized by
//!   [`random_search_optimizer`];
//! * [`compute_sample_box_decomposition`]: the box decompositions of the
//!   dominated space of every sampled front (the integration domains of the
//!   entropy estimators).
//!
//! The random numbers (paths and candidate sets) come from botorch_core's
//! generators. For BoTorch's draws (or any explicit ones), pass the path
//! draws and the candidate sets explicitly: given the same random numbers,
//! the results equal BoTorch's.

use rayon::prelude::*;

use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::sampling::pathwise::posterior_samplers::{
    get_matheron_path_model, MatheronPathModel, MatheronPathOptions,
};
use crate::sampling::pathwise::utils::{derive_seed, MatheronPathDraws};
use crate::sampling::pathwise::DEFAULT_NUM_FEATURES;
use crate::utils::multi_objective::box_decompositions::{
    BoxDecomposition, BoxDecompositionList, CellBounds, DominatedPartitioning,
    FastNondominatedPartitioning, NondominatedPartitioning,
};
use crate::utils::multi_objective::pareto::is_non_dominated;
use crate::utils::sampling::{check_bounds, draw_sobol_samples, random_seed};
use crate::Matrix;

/// The lower bound of the integration domains (BoTorch's `NEG_INF = -1e10`
/// of `compute_sample_box_decomposition`, finite so that the normal
/// log-densities of the bounds stay finite).
pub const SAMPLE_BOX_NEG_INF: f64 = -1e10;

/// The box decomposition used by [`compute_sample_box_decomposition`]
/// (BoTorch's `partitioning` class argument).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SamplePartitioning {
    /// `DominatedPartitioning` (BoTorch's default): the dominated space.
    #[default]
    Dominated,
    /// `FastNondominatedPartitioning`: the non-dominated space.
    FastNondominated,
    /// `NondominatedPartitioning` with the given `alpha`.
    Nondominated(f64),
}

/// The hypercell bounds of the box decompositions of sampled Pareto fronts
/// (BoTorch's `compute_sample_box_decomposition`): for each front
/// (`num_points × M`, all of the same size), the cells of a decomposition
/// with reference point `-1e10` (for maximization; the negated fronts and
/// flipped cells for minimization), padded with empty cells to the largest
/// number of cells; with `M = 1`, the single cell `[-1e10, max]` (or
/// `[min, 1e10]`). With `num_constraints = K > 0`, every cell is extended by
/// `[-1e10, 0]` in `K` constraint dimensions and one cell of the infeasible
/// space, `[-1e10, 1e10]^M × [0, 1e10]^K`, is appended.
///
/// The result has batch shape `[num_pareto_samples]`, i.e. BoTorch's
/// `num_pareto_samples x 2 x J x (M + K)` tensor as lower and upper vertices.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for no fronts or no objectives,
/// [`BotorchError::DimensionMismatch`] for fronts of different shapes and
/// errors of the decompositions (e.g. NaN outcomes).
pub fn compute_sample_box_decomposition(
    pareto_fronts: &[Matrix],
    partitioning: SamplePartitioning,
    maximize: bool,
    num_constraints: usize,
) -> Result<CellBounds> {
    let Some(first) = pareto_fronts.first() else {
        return Err(BotorchError::InvalidArgument(
            "at least one Pareto front is required".into(),
        ));
    };
    let (n, m) = first.shape();
    if m == 0 {
        return Err(BotorchError::InvalidArgument(
            "the Pareto fronts must have at least one objective".into(),
        ));
    }
    if pareto_fronts.iter().any(|f| f.shape() != (n, m)) {
        return Err(BotorchError::DimensionMismatch(
            "Currently this only supports Pareto fronts of the shape `num_pareto_samples x \
             num_pareto_points x num_objectives`."
                .into(),
        ));
    }
    let p = pareto_fronts.len();
    let weight = if maximize { 1.0 } else { -1.0 };
    let ref_point = vec![SAMPLE_BOX_NEG_INF; m];
    // Bounds without constraints: `p x J x m` lower and upper vertices.
    let (num_cells, mut lower, mut upper) = if m == 1 {
        let extreme: Vec<f64> = pareto_fronts
            .iter()
            .map(|f| {
                let best = f
                    .iter()
                    .map(|v| weight * v)
                    .fold(f64::NEG_INFINITY, torch_max);
                weight * best
            })
            .collect();
        let far = vec![weight * SAMPLE_BOX_NEG_INF; p];
        if maximize {
            (1, far, extreme)
        } else {
            (1, extreme, far)
        }
    } else {
        let decompositions = pareto_fronts
            .par_iter()
            .map(|f| {
                let y = f.map(|v| weight * v);
                Ok(match partitioning {
                    SamplePartitioning::Dominated => {
                        Box::new(DominatedPartitioning::new(&ref_point, Some(&y))?)
                            as Box<dyn BoxDecomposition>
                    }
                    SamplePartitioning::FastNondominated => {
                        Box::new(FastNondominatedPartitioning::new(&ref_point, Some(&y))?)
                    }
                    SamplePartitioning::Nondominated(alpha) => {
                        Box::new(NondominatedPartitioning::new(&ref_point, Some(&y), alpha)?)
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let cells = BoxDecompositionList::new(decompositions).get_hypercell_bounds()?;
        let (lo, up) = (cells.lower().to_vec(), cells.upper().to_vec());
        if maximize {
            (cells.num_cells(), lo, up)
        } else {
            // `weight * torch.flip(bounds, dims=[1])`.
            (
                cells.num_cells(),
                up.iter().map(|v| weight * v).collect(),
                lo.iter().map(|v| weight * v).collect(),
            )
        }
    };
    if num_constraints == 0 {
        return CellBounds::new(vec![p], num_cells, m, lower, upper);
    }
    let k = num_constraints;
    let width = m + k;
    let cells = num_cells + 1;
    let mut lo = Vec::with_capacity(p * cells * width);
    let mut up = Vec::with_capacity(p * cells * width);
    for b in 0..p {
        for j in 0..num_cells {
            let start = (b * num_cells + j) * m;
            lo.extend_from_slice(&lower[start..start + m]);
            lo.extend(std::iter::repeat_n(SAMPLE_BOX_NEG_INF, k));
            up.extend_from_slice(&upper[start..start + m]);
            up.extend(std::iter::repeat_n(0.0, k));
        }
        // The infeasible box.
        lo.extend(std::iter::repeat_n(SAMPLE_BOX_NEG_INF, m));
        lo.extend(std::iter::repeat_n(0.0, k));
        up.extend(std::iter::repeat_n(-SAMPLE_BOX_NEG_INF, width));
    }
    lower.clear();
    upper.clear();
    CellBounds::new(vec![p], cells, width, lo, up)
}

/// `torch.max` of two values (NaN propagates).
fn torch_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Options of [`random_search_optimizer`].
#[derive(Clone, Debug, PartialEq)]
pub struct RandomSearchOptions {
    /// The number of function evaluations per try (BoTorch's `pop_size`,
    /// default 1024).
    pub pop_size: usize,
    /// The maximum number of tries (BoTorch's `max_tries`, default 10).
    pub max_tries: usize,
    /// The seed of the Sobol candidate sets (try `i` uses stream `i`); `None`
    /// draws a random seed (BoTorch draws from torch's global generator).
    pub seed: Option<u64>,
    /// Explicit candidate sets (`pop_size × d` each), one per try, replacing
    /// the Sobol draws (e.g. BoTorch's); the search fails if it needs more
    /// tries than given.
    pub candidates: Option<Vec<Matrix>>,
}

impl Default for RandomSearchOptions {
    fn default() -> Self {
        Self {
            pop_size: 1024,
            max_tries: 10,
            seed: None,
            candidates: None,
        }
    }
}

/// Optimizes a deterministic function `f: R^d → R^M` (`n × d` points to
/// `n × M` values) by random search (BoTorch's `random_search_optimizer`):
/// each try evaluates `pop_size` scrambled Sobol points in `bounds` (`2 × d`)
/// and keeps the non-dominated points (`is_non_dominated` with
/// deduplication; minimization with `maximize = false`) of the previous
/// optima and the new points, in that order, until at least `num_points`
/// points are non-dominated. Returns the first `num_points` optimal inputs
/// (`num_points × d`) and outputs (`num_points × M`).
///
/// # Errors
/// [`BotorchError::Runtime`] with BoTorch's message if fewer than
/// `num_points` optimal points are found within `max_tries` tries,
/// [`BotorchError::InvalidArgument`] for invalid options or too few explicit
/// candidate sets, [`BotorchError::DimensionMismatch`] for inconsistent
/// shapes, and the errors of `f`.
pub fn random_search_optimizer<F>(
    f: &F,
    bounds: &Matrix,
    num_points: usize,
    maximize: bool,
    options: &RandomSearchOptions,
) -> Result<(Matrix, Matrix)>
where
    F: Fn(&Matrix) -> Result<Matrix> + ?Sized,
{
    let d = check_bounds(bounds)?;
    if num_points == 0 || options.pop_size == 0 {
        return Err(BotorchError::InvalidArgument(
            "num_points and pop_size must be positive".into(),
        ));
    }
    let seed = options.seed.unwrap_or_else(random_seed);
    let weight = if maximize { 1.0 } else { -1.0 };
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut outputs: Vec<Vec<f64>> = Vec::new();
    let mut num_found = 0;
    let mut ratio = 2;
    let mut num_tries = 0;
    while ratio > 1 && num_tries < options.max_tries {
        let x = match &options.candidates {
            Some(c) => {
                let x = c.get(num_tries).ok_or_else(|| {
                    BotorchError::InvalidArgument(format!(
                        "the random search needs more than the {} given candidate sets",
                        c.len()
                    ))
                })?;
                if x.ncols() != d || x.nrows() == 0 {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "candidate sets must be non-empty `n x {d}` matrices, got {}x{}",
                        x.nrows(),
                        x.ncols()
                    )));
                }
                x.clone()
            }
            None => {
                let raw = draw_sobol_samples(
                    bounds,
                    options.pop_size,
                    1,
                    Some(derive_seed(seed, num_tries as u64)),
                )?;
                Matrix::from_fn(raw.len(), d, |i, j| raw[i][(0, j)])
            }
        };
        let y = f(&x)?;
        if y.nrows() != x.nrows() || y.ncols() == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "the function returned {}x{} values for {} points",
                y.nrows(),
                y.ncols(),
                x.nrows()
            )));
        }
        if let Some(first) = outputs.first() {
            if first.len() != y.ncols() {
                return Err(BotorchError::DimensionMismatch(
                    "the number of outputs changed between tries".into(),
                ));
            }
        }
        inputs.extend(x.row_iter().map(|r| r.iter().copied().collect::<Vec<_>>()));
        outputs.extend(y.row_iter().map(|r| r.iter().copied().collect::<Vec<_>>()));
        let m = y.ncols();
        let all = Matrix::from_fn(outputs.len(), m, |i, t| weight * outputs[i][t]);
        let mask = is_non_dominated(&all, true, true);
        let mut keep = mask.iter();
        inputs.retain(|_| *keep.next().unwrap_or(&false));
        let mut keep = mask.iter();
        outputs.retain(|_| *keep.next().unwrap_or(&false));
        num_found = inputs.len();
        ratio = if num_found == 0 {
            usize::MAX
        } else {
            num_points.div_ceil(num_found)
        };
        num_tries += 1;
    }
    if ratio > 1 {
        return Err(BotorchError::Runtime(format!(
            "Only found {num_found} optimal points instead of {num_points}."
        )));
    }
    let m = outputs[0].len();
    Ok((
        Matrix::from_fn(num_points, d, |i, j| inputs[i][j]),
        Matrix::from_fn(num_points, m, |i, t| outputs[i][t]),
    ))
}

/// Options of [`sample_optimal_points`].
#[derive(Clone, Debug, PartialEq)]
pub struct SampleOptimalPointsOptions {
    /// The options of [`random_search_optimizer`] (BoTorch's
    /// `optimizer_kwargs`). Its `seed` and `candidates` are ignored: the
    /// candidate sets come from `seed` and `candidates` below.
    pub random_search: RandomSearchOptions,
    /// The number of Fourier features of the paths (default 1024).
    pub num_features: usize,
    /// The seed: sample `i` draws its path from stream `2i` and its candidate
    /// sets from stream `2i + 1`; `None` draws a random seed.
    pub seed: Option<u64>,
    /// Explicit path draws, one list (one entry per output) per sample.
    pub draws: Option<Vec<Vec<MatheronPathDraws>>>,
    /// Explicit candidate sets of the random search, one list per sample.
    pub candidates: Option<Vec<Vec<Matrix>>>,
}

impl Default for SampleOptimalPointsOptions {
    fn default() -> Self {
        Self {
            random_search: RandomSearchOptions::default(),
            num_features: DEFAULT_NUM_FEATURES,
            seed: None,
            draws: None,
            candidates: None,
        }
    }
}

/// Pareto sets and fronts of `num_samples` posterior sample paths of
/// `model` (BoTorch's `sample_optimal_points` with the default
/// `random_search_optimizer`): every sample is a Matheron path model of all
/// outputs ([`get_matheron_path_model`]) whose `num_points` optimal points
/// are found by [`random_search_optimizer`]. Returns one `num_points × d`
/// Pareto set and one `num_points × M` Pareto front per sample. The samples
/// are drawn and optimized in parallel; the results do not depend on the
/// number of threads.
///
/// # Errors
/// [`BotorchError::Unsupported`] for single-output models with
/// `num_points > 1` (BoTorch's message), [`BotorchError::DimensionMismatch`]
/// for explicit draws or candidates of the wrong length, and errors of the
/// paths and of the random search.
pub fn sample_optimal_points(
    model: &dyn Model,
    bounds: &Matrix,
    num_samples: usize,
    num_points: usize,
    maximize: bool,
    options: &SampleOptimalPointsOptions,
) -> Result<(Vec<Matrix>, Vec<Matrix>)> {
    if let Some(c) = &options.candidates {
        if c.len() != num_samples {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected candidate sets for {num_samples} samples, got {}",
                c.len()
            )));
        }
    }
    let seed = options.seed.unwrap_or_else(random_seed);
    sample_optimal_points_with(
        model,
        bounds,
        num_samples,
        num_points,
        options,
        |path, i| {
            let rs = RandomSearchOptions {
                seed: Some(derive_seed(seed, 2 * i as u64 + 1)),
                candidates: options.candidates.as_ref().map(|c| c[i].clone()),
                ..options.random_search.clone()
            };
            random_search_optimizer(
                &|x: &Matrix| path.forward(x),
                bounds,
                num_points,
                maximize,
                &rs,
            )
        },
    )
}

/// [`sample_optimal_points`] with a custom `optimizer`, called with the path
/// model of sample `i` and `i` (BoTorch's `optimizer` argument; it must
/// return `num_points × d` inputs and `num_points × M` outputs). The paths
/// come from `options.draws` or `options.seed` (`random_search`
/// and `candidates` are not used).
///
/// # Errors
/// See [`sample_optimal_points`]; [`BotorchError::DimensionMismatch`] if the
/// optimizer returns results of the wrong shape.
pub fn sample_optimal_points_with<O>(
    model: &dyn Model,
    bounds: &Matrix,
    num_samples: usize,
    num_points: usize,
    options: &SampleOptimalPointsOptions,
    optimizer: O,
) -> Result<(Vec<Matrix>, Vec<Matrix>)>
where
    O: Fn(&MatheronPathModel, usize) -> Result<(Matrix, Matrix)> + Sync,
{
    let m = model.num_outputs();
    let d = check_bounds(bounds)?;
    if d != model.dim() {
        return Err(BotorchError::DimensionMismatch(format!(
            "the bounds have {d} columns but the model has {} inputs",
            model.dim()
        )));
    }
    if m == 1 && num_points > 1 {
        return Err(BotorchError::Unsupported(
            "For single-objective optimization `num_points` should be 1.".into(),
        ));
    }
    if let Some(dr) = &options.draws {
        if dr.len() != num_samples {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected path draws for {num_samples} samples, got {}",
                dr.len()
            )));
        }
    }
    let seed = options.seed.unwrap_or_else(random_seed);
    let results = (0..num_samples)
        .into_par_iter()
        .map(|i| {
            let path_options = MatheronPathOptions {
                num_features: options.num_features,
                seed: Some(derive_seed(seed, 2 * i as u64)),
                draws: options.draws.as_ref().map(|dr| dr[i].clone()),
            };
            let path = get_matheron_path_model(model, 1, &path_options)?;
            let (ps, pf) = optimizer(&path, i)?;
            if ps.shape() != (num_points, d) || pf.shape() != (num_points, m) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the optimizer must return {num_points}x{d} inputs and {num_points}x{m} \
                     outputs, got {}x{} and {}x{}",
                    ps.nrows(),
                    ps.ncols(),
                    pf.nrows(),
                    pf.ncols()
                )));
            }
            Ok((ps, pf))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(results.into_iter().unzip())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BatchedMultiOutputGP, SingleTaskGPOptions};
    use crate::utils::multi_objective::pareto::is_non_dominated;

    #[test]
    fn sample_box_decompositions() {
        let f0 = Matrix::from_row_slice(3, 2, &[1.0, 0.0, 0.5, 0.5, 0.0, 1.0]);
        let f1 = Matrix::from_row_slice(3, 2, &[2.0, 1.0, 2.0, 1.0, 1.0, 2.0]);
        let max = compute_sample_box_decomposition(
            &[f0.clone(), f1.clone()],
            SamplePartitioning::Dominated,
            true,
            0,
        )
        .unwrap();
        assert_eq!(max.batch_shape(), &[2]);
        // The first front has 3 cells, the second 2 (padded with an empty cell).
        assert_eq!(max.num_cells(), 3);
        let (lo, up) = max.cell(1, 2);
        assert_eq!((lo, up), (&[0.0, 0.0][..], &[0.0, 0.0][..]));
        // The cells of the second front cover the dominated space above -1e10.
        let vol: f64 = (0..3)
            .map(|k| {
                let (l, u) = max.cell(1, k);
                (u[0] - l[0].max(0.0)).max(0.0) * (u[1] - l[1].max(0.0)).max(0.0)
            })
            .sum();
        assert!((vol - 3.0).abs() < 1e-9, "{vol}");
        // Minimization: the negated, flipped cells of the negated fronts.
        let min = compute_sample_box_decomposition(
            &[f0.map(|v| -v), f1.map(|v| -v)],
            SamplePartitioning::Dominated,
            false,
            0,
        )
        .unwrap();
        for (a, b) in min.lower().iter().zip(max.upper()) {
            assert_eq!(*a, -b);
        }
        // One objective and constraints.
        let one = compute_sample_box_decomposition(
            &[Matrix::from_row_slice(1, 1, &[3.0])],
            SamplePartitioning::Dominated,
            true,
            2,
        )
        .unwrap();
        assert_eq!(one.shape(), vec![2, 1, 2, 3]);
        assert_eq!(
            one.cell(0, 0),
            (&[-1e10, -1e10, -1e10][..], &[3.0, 0.0, 0.0][..])
        );
        assert_eq!(
            one.cell(0, 1),
            (&[-1e10, 0.0, 0.0][..], &[1e10, 1e10, 1e10][..])
        );
        assert!(
            compute_sample_box_decomposition(&[], SamplePartitioning::Dominated, true, 0).is_err()
        );
    }

    #[test]
    fn random_search_finds_pareto_points() {
        // f(x) = (x, 1 - x) on [0, 1]: every point is Pareto optimal.
        let f = |x: &Matrix| -> Result<Matrix> {
            Ok(Matrix::from_fn(x.nrows(), 2, |i, t| {
                if t == 0 {
                    x[(i, 0)]
                } else {
                    1.0 - x[(i, 0)]
                }
            }))
        };
        let bounds = Matrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let options = RandomSearchOptions {
            pop_size: 16,
            seed: Some(3),
            ..Default::default()
        };
        let (ps, pf) = random_search_optimizer(&f, &bounds, 5, true, &options).unwrap();
        assert_eq!((ps.shape(), pf.shape()), ((5, 1), (5, 2)));
        assert!(is_non_dominated(&pf, true, true).iter().all(|&b| b));
        // f(x) = (x, x): one optimum per try, so 5 points are never found.
        let g =
            |x: &Matrix| -> Result<Matrix> { Ok(Matrix::from_fn(x.nrows(), 2, |i, _| x[(i, 0)])) };
        let err = random_search_optimizer(&g, &bounds, 5, true, &options).unwrap_err();
        assert_eq!(err.to_string(), "Only found 1 optimal points instead of 5.");
        // Explicit candidates: the minimizer of the second.
        let cand = vec![Matrix::from_row_slice(3, 1, &[0.2, 0.9, 0.5])];
        let explicit = RandomSearchOptions {
            candidates: Some(cand),
            ..Default::default()
        };
        let (ps, pf) = random_search_optimizer(&g, &bounds, 1, false, &explicit).unwrap();
        assert_eq!((ps[(0, 0)], pf[(0, 1)]), (0.2, 0.2));
        assert!(random_search_optimizer(&g, &bounds, 2, false, &explicit).is_err());
    }

    #[test]
    fn optimal_points_of_paths() {
        let n = 10;
        let x = Matrix::from_fn(n, 2, |i, j| ((i * 7 + j * 3) % n) as f64 / n as f64);
        let y = Matrix::from_fn(n, 2, |i, t| {
            let s = x[(i, 0)] + 0.5 * x[(i, 1)];
            if t == 0 {
                (3.0 * s).sin()
            } else {
                (2.0 * s).cos()
            }
        });
        let model =
            BatchedMultiOutputGP::multi_output(x, y, SingleTaskGPOptions::default()).unwrap();
        let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        let options = SampleOptimalPointsOptions {
            num_features: 64,
            seed: Some(5),
            random_search: RandomSearchOptions {
                pop_size: 128,
                ..Default::default()
            },
            ..Default::default()
        };
        let (ps, pf) = sample_optimal_points(&model, &bounds, 3, 4, true, &options).unwrap();
        assert_eq!(ps.len(), 3);
        for (s, f) in ps.iter().zip(&pf) {
            assert_eq!((s.shape(), f.shape()), ((4, 2), (4, 2)));
            assert!(is_non_dominated(f, true, true).iter().all(|&b| b));
            assert!(s.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        // Deterministic for a seed, and independent of the number of threads.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (ps1, pf1) = pool
            .install(|| sample_optimal_points(&model, &bounds, 3, 4, true, &options))
            .unwrap();
        assert_eq!((ps1, pf1), (ps, pf));
        let single = BatchedMultiOutputGP::multi_output(
            Matrix::from_row_slice(2, 1, &[0.0, 1.0]),
            Matrix::from_row_slice(2, 1, &[0.0, 1.0]),
            SingleTaskGPOptions::default(),
        )
        .unwrap();
        let b1 = Matrix::from_row_slice(2, 1, &[0.0, 1.0]);
        assert!(matches!(
            sample_optimal_points(&single, &b1, 2, 2, true, &options),
            Err(BotorchError::Unsupported(_))
        ));
    }
}
