//! Multi-objective optimization of multi-output acquisition functions with
//! NSGA-II (`botorch.utils.multi_objective.optimize.optimize_with_nsgaii`).
//!
//! BoTorch runs pymoo's `NSGA2`; this module implements NSGA-II (Deb, Pratap,
//! Agarwal and Meyarivan, *A fast and elitist multiobjective genetic
//! algorithm: NSGA-II*, IEEE TEC 6(2), 2002) with pymoo's defaults as BoTorch
//! configures them (pymoo 0.6.2: `NSGA2(pop_size, eliminate_duplicates=True,
//! repair)`, `minimize(..., termination, seed)`):
//!
//! * random sampling in the box;
//! * binary tournament selection (`comp_by_dom_and_crowding`: the less
//!   infeasible point, else the dominating point, else the larger crowding
//!   distance, else at random) on concatenated random permutations;
//! * simulated binary crossover (SBX: `eta = 15`, mating probability 0.9,
//!   per-variable probability 0.5, children exchanged with probability 0.5)
//!   and polynomial mutation (`eta = 20`, individual probability 0.9,
//!   per-variable probability `min(0.5, 1/d)`), clamped to the bounds;
//! * elimination of offspring that duplicate the population, earlier
//!   offspring or each other (distance at most `1e-16`), with up to 100
//!   mating rounds per generation;
//! * rank-and-crowding survival of the feasible points (Pareto fronts, then
//!   the crowding distance with random tie-breaks), infeasible points after
//!   them by constraint violation;
//! * pymoo's default multi-objective termination
//!   (`DefaultMultiObjectiveTermination`: design-space tolerance 5e-4 and
//!   objective-space tolerance 5e-3 on the optimum, checked every sixth
//!   generation and held over 50 generations, at most 1000 generations or
//!   100,000 evaluations), or exactly `max_gen` generations;
//! * the result is the first front of the final population (if any point is
//!   feasible).
//!
//! pymoo's random streams cannot be reproduced, so runs agree with BoTorch +
//! pymoo statistically (the hypervolume of the returned fronts), not
//! bitwise. The deterministic building blocks — [`non_dominated_fronts`],
//! [`crowding_distance`], [`sbx`] and [`polynomial_mutation`] given the
//! uniform random numbers — follow pymoo's arithmetic exactly.
//!
//! Objectives are *minimized* by the engine ([`nsga2`]); [`optimize_with_nsgaii`]
//! maximizes, like BoTorch.

use rand::rngs::Xoshiro256PlusPlus;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

use crate::error::{BotorchError, Result};
use crate::utils::multi_objective::hypervolume::get_hypervolume_maximizing_subset;
use crate::utils::multi_objective::pareto::is_non_dominated;
use crate::Matrix;

/// pymoo's `DefaultDuplicateElimination` tolerance.
pub const DUPLICATE_EPSILON: f64 = 1e-16;

/// pymoo's `InfillCriterion.n_max_iterations` (mating rounds per generation).
pub const MAX_MATING_ROUNDS: usize = 100;

/// SBX's tolerance for parents that are too close to cross over.
pub const SBX_EPS: f64 = 1e-14;

// ---------------------------------------------------------------------------
// Deterministic building blocks
// ---------------------------------------------------------------------------

/// `a` dominates `b` (minimization; pymoo's `Dominator.get_relation`: 1 if `a`
/// dominates, -1 if `b` dominates, 0 otherwise).
pub fn dominance_relation(a: &[f64], b: &[f64]) -> i32 {
    let mut val = 0;
    for (x, y) in a.iter().zip(b) {
        if x < y {
            if val == -1 {
                return 0;
            }
            val = 1;
        } else if y < x {
            if val == 1 {
                return 0;
            }
            val = -1;
        }
    }
    val
}

/// The Pareto fronts of `f` (`n × m`, minimized), each in increasing index
/// order (pymoo's `NonDominatedSorting` via Pareto ranks). With
/// `n_stop_if_ranked`, only the fronts up to the one that brings the number of
/// ranked points to at least that many.
pub fn non_dominated_fronts(f: &Matrix, n_stop_if_ranked: Option<usize>) -> Vec<Vec<usize>> {
    let n = f.nrows();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| f.row(i).iter().copied().collect()).collect();
    // Deb's fast non-dominated sort: counts of dominating points and the
    // points each one dominates.
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut count = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            match dominance_relation(&rows[i], &rows[j]) {
                1 => {
                    dominated_by[i].push(j);
                    count[j] += 1;
                }
                -1 => {
                    dominated_by[j].push(i);
                    count[i] += 1;
                }
                _ => {}
            }
        }
    }
    let mut fronts = Vec::new();
    let mut current: Vec<usize> = (0..n).filter(|&i| count[i] == 0).collect();
    let mut ranked = 0;
    while !current.is_empty() {
        ranked += current.len();
        let mut next = Vec::new();
        for &i in &current {
            for &j in &dominated_by[i] {
                count[j] -= 1;
                if count[j] == 0 {
                    next.push(j);
                }
            }
        }
        next.sort_unstable();
        fronts.push(current);
        if n_stop_if_ranked.is_some_and(|s| ranked >= s) {
            break;
        }
        current = next;
    }
    fronts
}

/// pymoo's crowding distance (`calc_crowding_distance`, `crowding_func="cd"`):
/// per objective, the normalized gaps to the neighbors in a stable sort
/// (infinite at the ends, zero for a constant objective), averaged over the
/// objectives; fronts of at most two points are infinite.
pub fn crowding_distance(f: &Matrix) -> Vec<f64> {
    let (n, m) = f.shape();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }
    let mut cd = vec![0.0; n];
    for j in 0..m {
        let mut order: Vec<usize> = (0..n).collect();
        // np.argsort(kind="mergesort"): stable.
        order.sort_by(|&a, &b| {
            f[(a, j)]
                .partial_cmp(&f[(b, j)])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted: Vec<f64> = order.iter().map(|&i| f[(i, j)]).collect();
        let norm = sorted[n - 1] - sorted[0];
        let norm = if norm == 0.0 { f64::NAN } else { norm };
        let scaled = |v: f64| {
            let s = v / norm;
            if s.is_nan() {
                0.0
            } else {
                s
            }
        };
        for (k, &i) in order.iter().enumerate() {
            let last = if k == 0 {
                f64::INFINITY
            } else {
                sorted[k] - sorted[k - 1]
            };
            let next = if k == n - 1 {
                f64::INFINITY
            } else {
                sorted[k + 1] - sorted[k]
            };
            cd[i] += scaled(last) + scaled(next);
        }
    }
    cd.iter().map(|v| v / m as f64).collect()
}

/// Simulated binary crossover of the parent pairs `(a[k], b[k])` (rows of
/// `n_matings × d` matrices) given pymoo's uniform random numbers, in the
/// order pymoo draws them (`cross_sbx`): `u_var` (`n_matings × d`) selects the
/// crossed variables (`< prob_var`; variables whose parents differ by at most
/// [`SBX_EPS`] or whose bounds coincide are not crossed), `u_beta` and
/// `u_exchange` hold one number per crossed variable (row-major order). The
/// children of a crossed variable are exchanged if `(u_exchange < prob_bin)
/// XOR (a > b)`. Both children are clamped to the bounds.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inconsistent shapes or too few
/// random numbers.
#[allow(clippy::too_many_arguments)]
pub fn sbx(
    a: &Matrix,
    b: &Matrix,
    xl: &[f64],
    xu: &[f64],
    eta: f64,
    prob_var: f64,
    prob_bin: f64,
    u_var: &Matrix,
    u_beta: &[f64],
    u_exchange: &[f64],
) -> Result<(Matrix, Matrix)> {
    let (n, d) = a.shape();
    if b.shape() != (n, d) || u_var.shape() != (n, d) || xl.len() != d || xu.len() != d {
        return Err(BotorchError::DimensionMismatch(
            "parents, bounds and random numbers of SBX must have consistent shapes".into(),
        ));
    }
    let crossed: Vec<(usize, usize)> = (0..n)
        .flat_map(|k| (0..d).map(move |j| (k, j)))
        .filter(|&(k, j)| {
            u_var[(k, j)] < prob_var && (a[(k, j)] - b[(k, j)]).abs() > SBX_EPS && xl[j] != xu[j]
        })
        .collect();
    if u_beta.len() < crossed.len() || u_exchange.len() < crossed.len() {
        return Err(BotorchError::DimensionMismatch(format!(
            "SBX needs {} random numbers per kind, got {} and {}",
            crossed.len(),
            u_beta.len(),
            u_exchange.len()
        )));
    }
    let mut c1m = a.clone();
    let mut c2m = b.clone();
    let betaq = |beta: f64, u: f64| -> f64 {
        let alpha = 2.0 - beta.powf(-(eta + 1.0));
        if u <= 1.0 / alpha {
            (u * alpha).powf(1.0 / (eta + 1.0))
        } else {
            (1.0 / (2.0 - u * alpha)).powf(1.0 / (eta + 1.0))
        }
    };
    for (e, &(k, j)) in crossed.iter().enumerate() {
        let (p1, p2) = (a[(k, j)], b[(k, j)]);
        let (y1, y2) = (p1.min(p2), p1.max(p2));
        let delta = y2 - y1;
        let u = u_beta[e];
        let beta = 1.0 + (2.0 * (y1 - xl[j]) / delta);
        let mut c1 = 0.5 * ((y1 + y2) - betaq(beta, u) * delta);
        let beta = 1.0 + (2.0 * (xu[j] - y2) / delta);
        let mut c2 = 0.5 * ((y1 + y2) + betaq(beta, u) * delta);
        if (u_exchange[e] < prob_bin) ^ (p1 > p2) {
            std::mem::swap(&mut c1, &mut c2);
        }
        c1m[(k, j)] = c1;
        c2m[(k, j)] = c2;
    }
    for m in [&mut c1m, &mut c2m] {
        clamp_rows(m, xl, xu);
    }
    Ok((c1m, c2m))
}

/// pymoo's `repair_clamp` (NaN unchanged).
fn clamp_rows(x: &mut Matrix, xl: &[f64], xu: &[f64]) {
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            if x[(i, j)] < xl[j] {
                x[(i, j)] = xl[j];
            } else if x[(i, j)] > xu[j] {
                x[(i, j)] = xu[j];
            }
        }
    }
}

/// Polynomial mutation of the rows of `x` given pymoo's uniform random numbers
/// (`mut_pm`): `u_mask` (`n × d`) selects the mutated variables (`< prob_var`;
/// variables with coinciding bounds are not mutated) and `u` holds one number
/// per mutated variable (row-major order). The result is clamped to the
/// bounds.
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inconsistent shapes or too few
/// random numbers.
pub fn polynomial_mutation(
    x: &Matrix,
    xl: &[f64],
    xu: &[f64],
    eta: f64,
    prob_var: f64,
    u_mask: &Matrix,
    u: &[f64],
) -> Result<Matrix> {
    let (n, d) = x.shape();
    if u_mask.shape() != (n, d) || xl.len() != d || xu.len() != d {
        return Err(BotorchError::DimensionMismatch(
            "points, bounds and random numbers of the mutation must have consistent shapes".into(),
        ));
    }
    let mutated: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .filter(|&(i, j)| u_mask[(i, j)] < prob_var && xl[j] != xu[j])
        .collect();
    if u.len() < mutated.len() {
        return Err(BotorchError::DimensionMismatch(format!(
            "the mutation needs {} random numbers, got {}",
            mutated.len(),
            u.len()
        )));
    }
    let mut out = x.clone();
    let mut_pow = 1.0 / (eta + 1.0);
    for (e, &(i, j)) in mutated.iter().enumerate() {
        let (lo, hi, v, r) = (xl[j], xu[j], x[(i, j)], u[e]);
        let delta1 = (v - lo) / (hi - lo);
        let delta2 = (hi - v) / (hi - lo);
        let deltaq = if r <= 0.5 {
            let xy = 1.0 - delta1;
            let val = 2.0 * r + (1.0 - 2.0 * r) * xy.powf(eta + 1.0);
            val.powf(mut_pow) - 1.0
        } else {
            let xy = 1.0 - delta2;
            let val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * xy.powf(eta + 1.0);
            1.0 - val.powf(mut_pow)
        };
        let mut y = v + deltaq * (hi - lo);
        if y < lo {
            y = lo;
        }
        if y > hi {
            y = hi;
        }
        out[(i, j)] = y;
    }
    clamp_rows(&mut out, xl, xu);
    Ok(out)
}

// ---------------------------------------------------------------------------
// The NSGA-II engine
// ---------------------------------------------------------------------------

/// What [`nsga2`] evaluates: objectives to minimize (`n × m`) and inequality
/// constraints (`n × k`, feasible iff `<= 0`) of a population (`n × d`).
pub type Evaluation = (Matrix, Option<Matrix>);

/// Settings of [`nsga2`] (pymoo's `NSGA2` defaults).
#[derive(Clone, Debug)]
pub struct Nsga2Options {
    /// The population size (also the number of offspring per generation).
    pub pop_size: usize,
    /// Run exactly this many generations (pymoo's `MaximumGenerationTermination`;
    /// the initial population counts as the first); `None`: pymoo's default
    /// multi-objective termination.
    pub max_gen: Option<usize>,
    /// The seed of the random number generator (`None`: random).
    pub seed: Option<u64>,
    /// SBX distribution index (15).
    pub sbx_eta: f64,
    /// Probability that a mating is crossed (0.9).
    pub sbx_prob: f64,
    /// Probability that a variable of a crossed mating is crossed (0.5).
    pub sbx_prob_var: f64,
    /// Probability that the children of a variable are exchanged (0.5).
    pub sbx_prob_bin: f64,
    /// Polynomial-mutation distribution index (20).
    pub pm_eta: f64,
    /// Probability that an offspring is mutated (0.9).
    pub pm_prob: f64,
    /// Probability that a variable is mutated (`None`: `min(0.5, 1/d)`).
    pub pm_prob_var: Option<f64>,
    /// Eliminate duplicates (pymoo's `eliminate_duplicates=True`).
    pub eliminate_duplicates: bool,
    /// Snap the inputs `dim` to the nearest allowed value after sampling and
    /// variation (BoTorch's `DiscreteParameterRepair`).
    pub discrete_choices: Vec<(usize, Vec<f64>)>,
}

impl Default for Nsga2Options {
    fn default() -> Self {
        Self {
            pop_size: 100,
            max_gen: None,
            seed: None,
            sbx_eta: 15.0,
            sbx_prob: 0.9,
            sbx_prob_var: 0.5,
            sbx_prob_bin: 0.5,
            pm_eta: 20.0,
            pm_prob: 0.9,
            pm_prob_var: None,
            eliminate_duplicates: true,
            discrete_choices: Vec::new(),
        }
    }
}

/// The result of [`nsga2`].
#[derive(Clone, Debug)]
pub struct Nsga2Result {
    /// The optimum (pymoo's `res.X`): the first front of the final population
    /// if a point is feasible, else `None`.
    pub x: Option<Matrix>,
    /// The objectives of the optimum (pymoo's `res.F`, minimized).
    pub f: Option<Matrix>,
    /// The final population and its objectives.
    pub pop_x: Matrix,
    /// The objectives of the final population.
    pub pop_f: Matrix,
    /// The number of generations (including the initial population).
    pub n_gen: usize,
    /// The number of evaluated points.
    pub n_evals: usize,
}

/// A population with pymoo's attributes.
#[derive(Clone, Debug)]
struct Pop {
    x: Matrix,
    f: Matrix,
    cv: Vec<f64>,
    rank: Vec<usize>,
    crowding: Vec<f64>,
}

impl Pop {
    fn len(&self) -> usize {
        self.x.nrows()
    }

    fn select(&self, idx: &[usize]) -> Pop {
        let rows = |m: &Matrix| Matrix::from_fn(idx.len(), m.ncols(), |i, j| m[(idx[i], j)]);
        Pop {
            x: rows(&self.x),
            f: rows(&self.f),
            cv: idx.iter().map(|&i| self.cv[i]).collect(),
            rank: idx.iter().map(|&i| self.rank[i]).collect(),
            crowding: idx.iter().map(|&i| self.crowding[i]).collect(),
        }
    }

    fn merge(&self, other: &Pop) -> Pop {
        let d = self.x.ncols();
        let m = self.f.ncols();
        let n = self.len() + other.len();
        let stack = |a: &Matrix, b: &Matrix, c: usize| {
            Matrix::from_fn(n, c, |i, j| {
                if i < a.nrows() {
                    a[(i, j)]
                } else {
                    b[(i - a.nrows(), j)]
                }
            })
        };
        Pop {
            x: stack(&self.x, &other.x, d),
            f: stack(&self.f, &other.f, m),
            cv: self.cv.iter().chain(&other.cv).copied().collect(),
            rank: self.rank.iter().chain(&other.rank).copied().collect(),
            crowding: self
                .crowding
                .iter()
                .chain(&other.crowding)
                .copied()
                .collect(),
        }
    }
}

/// pymoo's constraint violation: the sum of the positive parts.
fn constraint_violation(g: Option<&Matrix>, n: usize) -> Vec<f64> {
    match g {
        None => vec![0.0; n],
        Some(g) => (0..n)
            .map(|i| {
                g.row(i)
                    .iter()
                    .map(|&v| if v > 0.0 { v } else { 0.0 })
                    .sum()
            })
            .collect(),
    }
}

/// The Euclidean distance between rows.
fn row_distance(a: &Matrix, i: usize, b: &Matrix, j: usize) -> f64 {
    let mut s = 0.0;
    for k in 0..a.ncols() {
        let d = a[(i, k)] - b[(j, k)];
        s += d * d;
    }
    s.sqrt()
}

/// pymoo's `DefaultDuplicateElimination.do(x, *others)`: the rows of `x` that
/// duplicate an earlier row or a row of `others` (distance ≤ 1e-16, NaN
/// distances never duplicate) are removed; returns the kept row indices.
fn eliminate_duplicates(x: &Matrix, others: &[&Matrix]) -> Vec<usize> {
    let dup = |d: f64| !d.is_nan() && d <= DUPLICATE_EPSILON;
    let mut keep: Vec<usize> = Vec::new();
    for i in 0..x.nrows() {
        if (0..i).any(|j| dup(row_distance(x, i, x, j))) {
            continue;
        }
        keep.push(i);
    }
    keep.retain(|&i| {
        others
            .iter()
            .all(|o| (0..o.nrows()).all(|j| !dup(row_distance(x, i, o, j))))
    });
    keep
}

/// BoTorch's `DiscreteParameterRepair`: the nearest allowed value (the
/// first on ties).
fn repair(x: &mut Matrix, choices: &[(usize, Vec<f64>)]) {
    for (dim, allowed) in choices {
        if allowed.is_empty() {
            continue;
        }
        for i in 0..x.nrows() {
            let v = x[(i, *dim)];
            let mut best = 0;
            for (k, a) in allowed.iter().enumerate() {
                if (v - a).abs() < (v - allowed[best]).abs() {
                    best = k;
                }
            }
            x[(i, *dim)] = allowed[best];
        }
    }
}

/// pymoo's `randomized_argsort(..., order="descending")`: ascending order of
/// a random permutation (ties at random), reversed.
fn randomized_argsort_desc(values: &[f64], rng: &mut Xoshiro256PlusPlus) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..values.len()).collect();
    perm.shuffle(rng);
    perm.sort_unstable_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    perm.reverse();
    perm
}

/// pymoo's `RankAndCrowding` survival of `n_survive` points (feasible points
/// by fronts and crowding distance, then infeasible points by constraint
/// violation); sets the ranks and crowding distances of the feasible points.
fn survive(pop: &mut Pop, n_survive: usize, rng: &mut Xoshiro256PlusPlus) -> Pop {
    let n_survive = n_survive.min(pop.len());
    let feas: Vec<usize> = (0..pop.len()).filter(|&i| pop.cv[i] <= 0.0).collect();
    let mut infeas: Vec<usize> = (0..pop.len()).filter(|&i| pop.cv[i] > 0.0).collect();
    infeas.sort_by(|&a, &b| {
        pop.cv[a]
            .partial_cmp(&pop.cv[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut survivors: Vec<usize> = Vec::with_capacity(n_survive);
    let n_feas_survive = feas.len().min(n_survive);
    if n_feas_survive > 0 {
        let f = Matrix::from_fn(feas.len(), pop.f.ncols(), |i, j| pop.f[(feas[i], j)]);
        let fronts = non_dominated_fronts(&f, Some(n_feas_survive));
        for (k, front) in fronts.iter().enumerate() {
            let ff = Matrix::from_fn(front.len(), f.ncols(), |i, j| f[(front[i], j)]);
            let cd = crowding_distance(&ff);
            let order: Vec<usize> = if survivors.len() + front.len() > n_feas_survive {
                let n_remove = survivors.len() + front.len() - n_feas_survive;
                let mut idx = randomized_argsort_desc(&cd, rng);
                idx.truncate(front.len() - n_remove);
                idx
            } else {
                (0..front.len()).collect()
            };
            for (j, &i) in front.iter().enumerate() {
                pop.rank[feas[i]] = k;
                pop.crowding[feas[i]] = cd[j];
            }
            survivors.extend(order.iter().map(|&j| feas[front[j]]));
        }
    }
    let n_rest = n_survive - survivors.len();
    survivors.extend(infeas.iter().take(n_rest));
    pop.select(&survivors)
}

/// pymoo's binary tournament (`comp_by_dom_and_crowding`).
fn tournament(pop: &Pop, a: usize, b: usize, rng: &mut Xoshiro256PlusPlus) -> usize {
    let pick = |rng: &mut Xoshiro256PlusPlus| if rng.random::<bool>() { a } else { b };
    let (ca, cb) = (pop.cv[a], pop.cv[b]);
    if ca > 0.0 || cb > 0.0 {
        return if ca < cb {
            a
        } else if cb < ca {
            b
        } else {
            pick(rng)
        };
    }
    let fa: Vec<f64> = pop.f.row(a).iter().copied().collect();
    let fb: Vec<f64> = pop.f.row(b).iter().copied().collect();
    match dominance_relation(&fa, &fb) {
        1 => a,
        -1 => b,
        _ => {
            let (da, db) = (pop.crowding[a], pop.crowding[b]);
            if da > db {
                a
            } else if db > da {
                b
            } else {
                pick(rng)
            }
        }
    }
}

/// One mating round (pymoo's `Mating._do`): `ceil(n / 2)` matings of
/// tournament winners, SBX and polynomial mutation; `2·ceil(n / 2)` offspring
/// (children 0 of all matings, then children 1).
fn mate(
    pop: &Pop,
    n: usize,
    xl: &[f64],
    xu: &[f64],
    o: &Nsga2Options,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Matrix> {
    let d = xl.len();
    let n_matings = n.div_ceil(2);
    // Tournament selection on concatenated random permutations.
    let n_random = n_matings * 2 * 2;
    let len = pop.len();
    let mut p: Vec<usize> = Vec::with_capacity(n_random + len);
    while p.len() < n_random {
        let mut perm: Vec<usize> = (0..len).collect();
        perm.shuffle(rng);
        p.extend(perm);
    }
    let winners: Vec<usize> = (0..n_matings * 2)
        .map(|t| tournament(pop, p[2 * t], p[2 * t + 1], rng))
        .collect();
    let a = Matrix::from_fn(n_matings, d, |k, j| pop.x[(winners[2 * k], j)]);
    let b = Matrix::from_fn(n_matings, d, |k, j| pop.x[(winners[2 * k + 1], j)]);
    // Crossover: which matings cross, then SBX's draws.
    let cross: Vec<bool> = (0..n_matings)
        .map(|_| rng.random::<f64>() < o.sbx_prob)
        .collect();
    let u_var = Matrix::from_fn(n_matings, d, |_, _| rng.random::<f64>());
    let n_crossed = (0..n_matings)
        .flat_map(|k| (0..d).map(move |j| (k, j)))
        .filter(|&(k, j)| {
            u_var[(k, j)] < o.sbx_prob_var
                && (a[(k, j)] - b[(k, j)]).abs() > SBX_EPS
                && xl[j] != xu[j]
        })
        .count();
    let u_beta: Vec<f64> = (0..n_crossed).map(|_| rng.random::<f64>()).collect();
    let u_exchange: Vec<f64> = (0..n_crossed).map(|_| rng.random::<f64>()).collect();
    let (c1, c2) = sbx(
        &a,
        &b,
        xl,
        xu,
        o.sbx_eta,
        o.sbx_prob_var,
        o.sbx_prob_bin,
        &u_var,
        &u_beta,
        &u_exchange,
    )?;
    let mut off = Matrix::zeros(2 * n_matings, d);
    for (k, &crossed) in cross.iter().enumerate() {
        let (r0, r1) = if crossed {
            (c1.row(k).into_owned(), c2.row(k).into_owned())
        } else {
            (a.row(k).into_owned(), b.row(k).into_owned())
        };
        off.set_row(k, &r0);
        off.set_row(n_matings + k, &r1);
    }
    // Polynomial mutation of every offspring, kept with probability pm_prob.
    let prob_var = o.pm_prob_var.unwrap_or_else(|| (1.0 / d as f64).min(0.5));
    let m_off = off.nrows();
    let u_mask = Matrix::from_fn(m_off, d, |_, _| rng.random::<f64>());
    let n_mut = (0..m_off)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .filter(|&(i, j)| u_mask[(i, j)] < prob_var && xl[j] != xu[j])
        .count();
    let u: Vec<f64> = (0..n_mut).map(|_| rng.random::<f64>()).collect();
    let mutated = polynomial_mutation(&off, xl, xu, o.pm_eta, prob_var, &u_mask, &u)?;
    for i in 0..m_off {
        if rng.random::<f64>() <= o.pm_prob {
            off.set_row(i, &mutated.row(i).into_owned());
        }
    }
    Ok(off)
}

/// The termination criteria of pymoo's `DefaultMultiObjectiveTermination`.
struct DefaultTermination {
    x: Robust<DeltaTolerance<Matrix>>,
    cv: Robust<DeltaTolerance<f64>>,
    f: Robust<DeltaTolerance<ObjectiveData>>,
    n_max_gen: usize,
    n_max_evals: usize,
}

/// The optimum's objectives as `MultiObjectiveSpaceTermination` records them.
#[derive(Clone)]
struct ObjectiveData {
    f: Matrix,
    ideal: Vec<f64>,
    nadir: Vec<f64>,
    feas: bool,
}

/// pymoo's `DeltaToleranceTermination` (`tol`, `n_skip`) of data `T`.
struct DeltaTolerance<T> {
    tol: f64,
    n_skip: usize,
    data: Option<T>,
    counter: usize,
    perc: f64,
}

impl<T: Clone> DeltaTolerance<T> {
    fn new(tol: f64, n_skip: usize) -> Self {
        Self {
            tol,
            n_skip,
            data: None,
            counter: 0,
            perc: 0.0,
        }
    }

    fn update(&mut self, current: T, delta: impl Fn(&T, &T) -> f64) -> f64 {
        let perc = match &self.data {
            None => 0.0,
            Some(_) if self.counter > 0 && !self.counter.is_multiple_of(self.n_skip + 1) => {
                self.perc
            }
            Some(prev) => {
                let d = delta(prev, &current);
                if d <= self.tol {
                    self.perc = 1.0;
                    return 1.0;
                }
                1.0 / (1.0 + (d - self.tol))
            }
        };
        self.data = Some(current);
        self.counter += 1;
        self.perc = perc;
        perc
    }
}

/// pymoo's `RobustTermination`: the minimum of the last `period` values.
struct Robust<T> {
    inner: T,
    period: usize,
    history: std::collections::VecDeque<f64>,
}

impl<T> Robust<T> {
    fn new(inner: T, period: usize) -> Self {
        Self {
            inner,
            period,
            history: std::collections::VecDeque::new(),
        }
    }

    fn push(&mut self, perc: f64) -> f64 {
        self.history.push_back(perc);
        while self.history.len() > self.period {
            self.history.pop_front();
        }
        self.history.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// pymoo's IGD of `approx` with respect to `reference`: the mean distance of
/// the reference points to their nearest approximation point.
fn igd(reference: &Matrix, approx: &Matrix) -> f64 {
    if reference.nrows() == 0 || approx.nrows() == 0 {
        return f64::INFINITY;
    }
    let total: f64 = (0..reference.nrows())
        .map(|i| {
            (0..approx.nrows())
                .map(|j| row_distance(reference, i, approx, j))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    total / reference.nrows() as f64
}

/// pymoo's `normalize(X, xl, xu)` (zero ranges: only shifted).
fn normalize(x: &Matrix, lo: &[f64], hi: &[f64]) -> Matrix {
    Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
        let r = hi[j] - lo[j];
        if r == 0.0 || r.is_nan() {
            x[(i, j)] - lo[j]
        } else {
            (x[(i, j)] - lo[j]) / r
        }
    })
}

impl DefaultTermination {
    fn new() -> Self {
        let (n_skip, period) = (5, 50);
        Self {
            x: Robust::new(DeltaTolerance::new(0.0005, n_skip), period),
            cv: Robust::new(DeltaTolerance::new(1e-8, n_skip), period),
            f: Robust::new(DeltaTolerance::new(0.005, n_skip), period),
            n_max_gen: 1000,
            n_max_evals: 100_000,
        }
    }

    /// The progress after a generation (terminate at `>= 1`), from the
    /// optimum `opt` (its inputs normalized to the bounds, objectives,
    /// feasibility) and the counters.
    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        opt_x: &Matrix,
        opt_f: &Matrix,
        opt_cv: &[f64],
        has_constraints: bool,
        xl: &[f64],
        xu: &[f64],
        n_gen: usize,
        n_evals: usize,
    ) -> f64 {
        // DesignSpaceTermination: IGD(current).do(previous) of the normalized
        // optimal inputs.
        let xn = normalize(opt_x, xl, xu);
        let x = {
            let p = self.x.inner.update(xn, |prev, cur| igd(cur, prev));
            self.x.push(p)
        };
        // ConstraintViolationTermination (terminate_when_feasible=False).
        let cv = {
            let p = if !has_constraints || opt_cv.iter().any(|&c| c <= 0.0) {
                0.0
            } else {
                let min_cv = opt_cv.iter().copied().fold(f64::INFINITY, f64::min);
                self.cv
                    .inner
                    .update(min_cv, |prev, cur| (prev - cur).max(0.0))
            };
            self.cv.push(p)
        };
        // MultiObjectiveSpaceTermination(only_feas=True).
        let f = {
            let rows: Vec<usize> = (0..opt_f.nrows()).filter(|&i| opt_cv[i] <= 0.0).collect();
            let ff = Matrix::from_fn(rows.len(), opt_f.ncols(), |i, j| opt_f[(rows[i], j)]);
            let m = opt_f.ncols();
            let ideal: Vec<f64> = (0..m)
                .map(|j| ff.column(j).iter().copied().fold(f64::INFINITY, f64::min))
                .collect();
            let nadir: Vec<f64> = (0..m)
                .map(|j| {
                    ff.column(j)
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .collect();
            let data = ObjectiveData {
                feas: ff.nrows() > 0,
                f: ff,
                ideal,
                nadir,
            };
            let p = self.f.inner.update(data, |prev, cur| {
                if !(prev.feas && cur.feas) {
                    return f64::INFINITY;
                }
                let norm: Vec<f64> = (0..cur.ideal.len())
                    .map(|j| {
                        let v = cur.nadir[j] - cur.ideal[j];
                        if v < 1e-32 {
                            1.0
                        } else {
                            v
                        }
                    })
                    .collect();
                let delta_norm = |a: &[f64], b: &[f64]| {
                    a.iter()
                        .zip(b)
                        .zip(&norm)
                        .map(|((x, y), n)| ((x - y) / n).abs())
                        .fold(f64::NEG_INFINITY, f64::max)
                };
                let delta_ideal = delta_norm(&cur.ideal, &prev.ideal);
                let delta_nadir = delta_norm(&cur.nadir, &prev.nadir);
                let c_n = normalize(&cur.f, &cur.ideal, &cur.nadir);
                let p_n = normalize(&prev.f, &cur.ideal, &cur.nadir);
                let delta_f = igd(&c_n, &p_n);
                delta_ideal.max(delta_nadir).max(delta_f)
            });
            self.f.push(p)
        };
        let gen = n_gen as f64 / self.n_max_gen as f64;
        let evals = n_evals as f64 / self.n_max_evals as f64;
        [x, cv, f, gen, evals].into_iter().fold(0.0, f64::max)
    }
}

/// NSGA-II (see the [module documentation](self)) minimizing the objectives
/// returned by `evaluate` over the box `xl ≤ x ≤ xu`.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for invalid bounds or a population
/// size of zero, [`BotorchError::DimensionMismatch`] for evaluations of the
/// wrong shape, and errors of `evaluate`.
pub fn nsga2<E>(
    mut evaluate: E,
    xl: &[f64],
    xu: &[f64],
    options: &Nsga2Options,
) -> Result<Nsga2Result>
where
    E: FnMut(&Matrix) -> Result<Evaluation>,
{
    let d = xl.len();
    if d == 0
        || xu.len() != d
        || xl
            .iter()
            .zip(xu)
            .any(|(l, u)| l.partial_cmp(u).is_none_or(|o| o.is_gt()))
    {
        return Err(BotorchError::InvalidArgument(
            "NSGA-II needs bounds with xl <= xu in every dimension".into(),
        ));
    }
    if options.pop_size == 0 {
        return Err(BotorchError::InvalidArgument(
            "the population size must be positive".into(),
        ));
    }
    let seed = options
        .seed
        .unwrap_or_else(crate::utils::sampling::random_seed);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut n_evals = 0usize;
    let mut has_constraints = false;
    let mut eval_pop =
        |x: Matrix, n_evals: &mut usize, has_constraints: &mut bool| -> Result<Pop> {
            let n = x.nrows();
            let (f, g) = evaluate(&x)?;
            if f.nrows() != n {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the objectives of {n} points have {} rows",
                    f.nrows()
                )));
            }
            if let Some(g) = &g {
                if g.nrows() != n {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "the constraints of {n} points have {} rows",
                        g.nrows()
                    )));
                }
                *has_constraints |= g.ncols() > 0;
            }
            *n_evals += n;
            Ok(Pop {
                cv: constraint_violation(g.as_ref(), n),
                rank: vec![usize::MAX; n],
                crowding: vec![f64::NAN; n],
                x,
                f,
            })
        };
    // Initialization: random sampling, repair, duplicates eliminated.
    let mut x0 = Matrix::from_fn(options.pop_size, d, |_, _| rng.random::<f64>());
    for i in 0..options.pop_size {
        for j in 0..d {
            x0[(i, j)] = xl[j] + (xu[j] - xl[j]) * x0[(i, j)];
        }
    }
    repair(&mut x0, &options.discrete_choices);
    if options.eliminate_duplicates {
        let keep = eliminate_duplicates(&x0, &[]);
        x0 = Matrix::from_fn(keep.len(), d, |i, j| x0[(keep[i], j)]);
    }
    let mut pop = eval_pop(x0, &mut n_evals, &mut has_constraints)?;
    let n0 = pop.len();
    pop = survive(&mut pop, n0, &mut rng);
    let mut termination = DefaultTermination::new();
    let mut n_gen = 1usize;
    loop {
        // _post_advance: the optimum and the termination.
        let opt = optimum(&pop);
        let progress = match options.max_gen {
            Some(g) => n_gen as f64 / g.max(1) as f64,
            None => termination.update(
                &pop.x.select_rows(&opt),
                &pop.f.select_rows(&opt),
                &opt.iter().map(|&i| pop.cv[i]).collect::<Vec<_>>(),
                has_constraints,
                xl,
                xu,
                n_gen,
                n_evals,
            ),
        };
        if progress >= 1.0 {
            break;
        }
        // Infill: offspring free of duplicates, up to 100 mating rounds.
        let n_off = options.pop_size;
        let mut off = Matrix::zeros(0, d);
        for _ in 0..MAX_MATING_ROUNDS {
            let remaining = n_off - off.nrows();
            let mut cand = mate(&pop, remaining, xl, xu, options, &mut rng)?;
            repair(&mut cand, &options.discrete_choices);
            if options.eliminate_duplicates {
                let keep = eliminate_duplicates(&cand, &[&pop.x, &off]);
                cand = Matrix::from_fn(keep.len(), d, |i, j| cand[(keep[i], j)]);
            }
            let take = cand.nrows().min(remaining);
            let mut merged = Matrix::zeros(off.nrows() + take, d);
            merged.rows_mut(0, off.nrows()).copy_from(&off);
            merged
                .rows_mut(off.nrows(), take)
                .copy_from(&cand.rows(0, take));
            off = merged;
            if off.nrows() >= n_off {
                break;
            }
        }
        if off.nrows() == 0 {
            break;
        }
        let off = eval_pop(off, &mut n_evals, &mut has_constraints)?;
        let mut merged = pop.merge(&off);
        pop = survive(&mut merged, options.pop_size, &mut rng);
        n_gen += 1;
    }
    let opt = optimum(&pop);
    let feasible = opt.iter().any(|&i| pop.cv[i] <= 0.0);
    Ok(Nsga2Result {
        x: feasible.then(|| pop.x.select_rows(&opt)),
        f: feasible.then(|| pop.f.select_rows(&opt)),
        pop_x: pop.x,
        pop_f: pop.f,
        n_gen,
        n_evals,
    })
}

/// NSGA2's `_set_optimum`: the first front if a point is feasible, else the
/// least infeasible point.
fn optimum(pop: &Pop) -> Vec<usize> {
    if pop.cv.iter().any(|&c| c <= 0.0) {
        (0..pop.len())
            .filter(|&i| pop.rank[i] == 0 && pop.cv[i] <= 0.0)
            .collect()
    } else {
        (0..pop.len())
            .min_by(|&a, &b| {
                pop.cv[a]
                    .partial_cmp(&pop.cv[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .into_iter()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// optimize_with_nsgaii
// ---------------------------------------------------------------------------

/// A linear inequality constraint on the inputs: `Σ_i x[indices[i]] ·
/// coefficients[i] >= rhs` (BoTorch's `inequality_constraints`).
pub type InequalityConstraint = (Vec<usize>, Vec<f64>, f64);

/// Options of [`optimize_with_nsgaii`] (BoTorch's keyword arguments).
#[derive(Clone, Debug)]
pub struct OptimizeWithNsgaiiOptions {
    /// The number of candidates (`None`: the whole first front).
    pub q: Option<usize>,
    /// A lower bound on the objectives (constraints `ref_point - obj <= 0`).
    pub ref_point: Option<Vec<f64>>,
    /// Linear inequality constraints on the inputs.
    pub inequality_constraints: Vec<InequalityConstraint>,
    /// The population size (BoTorch's default 250; at least `q`).
    pub population_size: usize,
    /// The number of generations (`None`: pymoo's default termination).
    pub max_gen: Option<usize>,
    /// The seed (`None`: random).
    pub seed: Option<u64>,
    /// Inputs fixed to values (`(index, value)`).
    pub fixed_features: Vec<(usize, f64)>,
    /// The number of runs if NSGA-II finds no feasible point (default 2).
    pub max_attempts: usize,
    /// Allowed values of discrete inputs (`(index, values)`).
    pub discrete_choices: Vec<(usize, Vec<f64>)>,
}

impl Default for OptimizeWithNsgaiiOptions {
    fn default() -> Self {
        Self {
            q: None,
            ref_point: None,
            inequality_constraints: Vec::new(),
            population_size: 250,
            max_gen: None,
            seed: None,
            fixed_features: Vec::new(),
            max_attempts: 2,
            discrete_choices: Vec::new(),
        }
    }
}

/// The result of [`optimize_with_nsgaii`].
#[derive(Clone, Debug)]
pub struct OptimizeWithNsgaiiResult {
    /// The Pareto set (`n × d`).
    pub x: Matrix,
    /// The Pareto front (`n × m`, maximized objectives).
    pub y: Matrix,
    /// BoTorch's warnings (e.g. fewer points than `q`).
    pub warnings: Vec<String>,
    /// The number of NSGA-II runs.
    pub attempts: usize,
}

/// BoTorch's `optimize_with_nsgaii`: maximizes the objectives of `evaluate`
/// (`n × d` points to `(objectives (n × m, maximized), outcome constraints
/// (n × k, feasible iff <= 0))`, the objective and outcome constraints of
/// BoTorch applied to the acquisition values) with NSGA-II and returns the
/// Pareto set and front (a hypervolume-maximizing subset of `q` points
/// relative to the front's nadir if `q` is given).
///
/// # Errors
/// [`BotorchError::Runtime`] if NSGA-II finds no feasible point in
/// `max_attempts` runs (BoTorch's message), and errors of `evaluate`.
pub fn optimize_with_nsgaii<E>(
    mut evaluate: E,
    bounds: &Matrix,
    num_objectives: usize,
    options: &OptimizeWithNsgaiiOptions,
) -> Result<OptimizeWithNsgaiiResult>
where
    E: FnMut(&Matrix) -> Result<Evaluation>,
{
    if bounds.nrows() != 2 || bounds.ncols() == 0 {
        return Err(BotorchError::DimensionMismatch(format!(
            "bounds must be a 2 x d matrix, got {}x{}",
            bounds.nrows(),
            bounds.ncols()
        )));
    }
    let d = bounds.ncols();
    let mut xl: Vec<f64> = bounds.row(0).iter().copied().collect();
    let mut xu: Vec<f64> = bounds.row(1).iter().copied().collect();
    for &(i, v) in &options.fixed_features {
        if i >= d {
            return Err(BotorchError::DimensionMismatch(format!(
                "fixed feature index {i} out of range for d = {d}"
            )));
        }
        xl[i] = v;
        xu[i] = v;
    }
    let discrete: Vec<(usize, Vec<f64>)> = options
        .discrete_choices
        .iter()
        .filter(|(k, _)| !options.fixed_features.iter().any(|(i, _)| i == k))
        .cloned()
        .collect();
    if let Some(r) = &options.ref_point {
        if r.len() != num_objectives {
            return Err(BotorchError::DimensionMismatch(format!(
                "ref_point has {} elements but there are {num_objectives} objectives",
                r.len()
            )));
        }
    }
    let pop_size = options
        .q
        .map_or(options.population_size, |q| options.population_size.max(q));
    let mut warnings = Vec::new();
    let mut attempts = 0;
    let mut res = None;
    for i in 1..=options.max_attempts.max(1) {
        attempts = i;
        let nsga_options = Nsga2Options {
            pop_size,
            max_gen: options.max_gen,
            // Every attempt with its own stream (BoTorch reruns pymoo with
            // the same seed).
            seed: options.seed.map(|s| s.wrapping_add((i - 1) as u64)),
            discrete_choices: discrete.clone(),
            ..Default::default()
        };
        let r = nsga2(
            |x: &Matrix| {
                let (obj, cons) = evaluate(x)?;
                if obj.ncols() != num_objectives || obj.nrows() != x.nrows() {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "the objective returned a {}x{} matrix for {} points and \
                         {num_objectives} objectives",
                        obj.nrows(),
                        obj.ncols(),
                        x.nrows()
                    )));
                }
                // Constraints: outcome constraints, the reference point, the
                // parameter constraints (BoTorch's order).
                let mut cols: Vec<Vec<f64>> = Vec::new();
                if let Some(c) = &cons {
                    for j in 0..c.ncols() {
                        cols.push(c.column(j).iter().copied().collect());
                    }
                }
                if let Some(r) = &options.ref_point {
                    for (t, rt) in r.iter().enumerate() {
                        cols.push((0..x.nrows()).map(|p| rt - obj[(p, t)]).collect());
                    }
                }
                for (idx, coef, rhs) in &options.inequality_constraints {
                    cols.push(
                        (0..x.nrows())
                            .map(|p| {
                                let lhs: f64 =
                                    idx.iter().zip(coef).map(|(&j, &c)| x[(p, j)] * c).sum();
                                rhs - lhs
                            })
                            .collect(),
                    );
                }
                let g = (!cols.is_empty())
                    .then(|| Matrix::from_fn(x.nrows(), cols.len(), |p, j| cols[j][p]));
                Ok((-obj, g))
            },
            &xl,
            &xu,
            &nsga_options,
        )?;
        if r.x.is_some() {
            res = Some(r);
            break;
        }
    }
    let Some(res) = res else {
        return Err(BotorchError::Runtime(format!(
            "NSGA-II failed to find a feasible point after {} attempts. Consider relaxing the \
             constraints or increasing the population size.",
            options.max_attempts.max(1)
        )));
    };
    let x = res.x.expect("feasible result");
    let y = -res.f.expect("feasible result");
    let mask = is_non_dominated(&y, true, true);
    let pareto: Vec<usize> = (0..y.nrows()).filter(|&i| mask[i]).collect();
    let mut x_p = x.select_rows(&pareto);
    let mut y_p = y.select_rows(&pareto);
    if let Some(q) = options.q {
        if y_p.nrows() > q {
            let nadir: Vec<f64> = (0..y_p.ncols())
                .map(|t| y_p.column(t).iter().copied().fold(f64::INFINITY, f64::min))
                .collect();
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                options
                    .seed
                    .unwrap_or_else(crate::utils::sampling::random_seed)
                    ^ 0x6e73_6761,
            );
            let (ys, idx) =
                get_hypervolume_maximizing_subset(q, &y_p, &nadir, |rest: &[usize]| {
                    rest[rng.random_range(0..rest.len())]
                })?;
            y_p = ys;
            x_p = x_p.select_rows(&idx);
        } else if y_p.nrows() < q {
            if y.nrows() >= q {
                // Fill up with random dominated points (in their order in Y).
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                    options
                        .seed
                        .unwrap_or_else(crate::utils::sampling::random_seed)
                        ^ 0x6669_6c6c,
                );
                let mut dominated: Vec<usize> = (0..y.nrows()).filter(|&i| !mask[i]).collect();
                dominated.shuffle(&mut rng);
                let mut keep = mask.clone();
                for &i in dominated.iter().take(q - y_p.nrows()) {
                    keep[i] = true;
                }
                let rows: Vec<usize> = (0..y.nrows()).filter(|&i| keep[i]).collect();
                x_p = x.select_rows(&rows);
                y_p = y.select_rows(&rows);
            } else {
                warnings.push(format!("NSGA-II only returned {} points.", y.nrows()));
                return Ok(OptimizeWithNsgaiiResult {
                    x,
                    y,
                    warnings,
                    attempts,
                });
            }
        }
    }
    Ok(OptimizeWithNsgaiiResult {
        x: x_p,
        y: y_p,
        warnings,
        attempts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fronts_and_crowding() {
        let f = Matrix::from_row_slice(
            6,
            2,
            &[1.0, 4.0, 2.0, 2.0, 4.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0],
        );
        let fronts = non_dominated_fronts(&f, None);
        assert_eq!(fronts, vec![vec![0, 1, 2, 5], vec![3], vec![4]]);
        assert_eq!(non_dominated_fronts(&f, Some(4)).len(), 1);
        let cd = crowding_distance(&f.select_rows(&[0, 1, 2, 5]));
        assert!(cd[0].is_infinite() && cd[2].is_infinite());
        // Points 1 and 3 coincide: the stable sort puts 1 first in both
        // objectives, so 1 has gaps (1/3, 0) and 3 has (0, 2/3).
        assert!((cd[1] - 1.0 / 3.0).abs() < 1e-15 && (cd[3] - 2.0 / 3.0).abs() < 1e-15);
        assert_eq!(
            crowding_distance(&f.select_rows(&[0, 1])),
            vec![f64::INFINITY; 2]
        );
    }

    #[test]
    fn operators_stay_in_bounds() {
        let a = Matrix::from_row_slice(2, 2, &[0.1, 0.9, 0.5, 0.5]);
        let b = Matrix::from_row_slice(2, 2, &[0.8, 0.2, 0.5, 0.6]);
        let u = Matrix::from_row_slice(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let (c1, c2) = sbx(
            &a,
            &b,
            &[0.0, 0.0],
            &[1.0, 1.0],
            15.0,
            0.5,
            0.5,
            &u,
            &[0.3, 0.9, 0.6],
            &[0.1, 0.7, 0.2],
        )
        .unwrap();
        // (1, 0) is too close and stays; the others are crossed.
        assert_eq!(c1[(1, 0)], 0.5);
        for m in [&c1, &c2] {
            assert!(m.iter().all(|&v| (0.0..=1.0).contains(&v)));
        }
        let x = Matrix::from_row_slice(1, 3, &[0.0, 0.5, 1.0]);
        let y = polynomial_mutation(
            &x,
            &[0.0; 3],
            &[1.0; 3],
            20.0,
            0.5,
            &Matrix::from_row_slice(1, 3, &[0.1, 0.9, 0.2]),
            &[0.2, 0.8],
        )
        .unwrap();
        assert_eq!(y[(0, 1)], 0.5);
        assert!(y[(0, 0)] >= 0.0 && y[(0, 2)] <= 1.0);
    }

    #[test]
    fn nsga2_solves_a_convex_problem() {
        // min (x², (x - 2)²) on [-4, 4]: Pareto set [0, 2].
        let res = nsga2(
            |x: &Matrix| {
                let f = Matrix::from_fn(x.nrows(), 2, |i, t| {
                    let v = x[(i, 0)] - 2.0 * t as f64;
                    v * v
                });
                Ok((f, None))
            },
            &[-4.0],
            &[4.0],
            &Nsga2Options {
                pop_size: 40,
                seed: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let x = res.x.unwrap();
        assert!(x.nrows() > 10);
        assert!(x.iter().all(|&v| (-1e-3..=2.0 + 1e-3).contains(&v)));
        assert!(res.n_gen <= 1000 && res.n_evals <= 100_000 + 40);
    }
}
