//! Sampling from polytopes defined by box bounds and linear constraints
//! (`botorch.utils.sampling`): [`find_interior_point`],
//! [`HitAndRunPolytopeSampler`], [`sample_polytope`] and
//! [`get_polytope_samples`].
//!
//! BoTorch's hit-and-run Markov chain: starting from an interior point
//! (found by a linear program that maximizes the slack of all inequality
//! constraints), every step draws a uniformly random direction `r` on the
//! unit sphere of the (equality-constrained) subspace and moves to a
//! uniformly random point of the chord `{x + α r}` inside the polytope. As in
//! BoTorch, constraints are first normalized to the unit cube of the bounds,
//! equality constraints are eliminated by sampling in their null space, and
//! the chain keeps every `n_thinning`-th point after `n_burnin` steps.
//!
//! The random numbers come from seeded Xoshiro256++ generators (BoTorch uses
//! torch's global generator), so the samples differ from BoTorch's but follow
//! the same Markov chain law; the linear program is solved with `microlp`
//! (BoTorch: HiGHS), which may pick a different optimal vertex when the
//! maximal-slack point is not unique.

use microlp::{ComparisonOp, OptimizationDirection, Problem};
use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

use crate::error::{BotorchError, Result};
use crate::optim::parameter_constraints::{ConstraintIndices, LinearConstraint};
use crate::utils::sampling::random_seed;
use crate::{Matrix, Vector};

/// `torch.finfo().max` (float32, torch's default dtype), BoTorch's bound of
/// the step length along unconstrained directions.
const LARGE_CONSTANT: f64 = 3.4028234663852886e38;

/// Dense constraints `A x (≥ | =) b` from sparse intra-point constraints
/// (BoTorch's `sparse_to_dense_constraints`; `A[i, indices] = coefficients`,
/// the last duplicate index wins as in torch's indexed assignment).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for inter-point constraints or indices
/// out of range.
pub fn sparse_to_dense_constraints(
    d: usize,
    constraints: &[LinearConstraint],
) -> Result<(Matrix, Vector)> {
    let mut a = Matrix::zeros(constraints.len(), d);
    let mut b = Vector::zeros(constraints.len());
    for (i, c) in constraints.iter().enumerate() {
        let ConstraintIndices::IntraPoint(idx) = c.indices() else {
            return Err(BotorchError::InvalidArgument(
                "`indices` must be a one-dimensional tensor. This method does not support the \
                 kind of 'inter-point constraints' that are supported by `optimize_acqf()`."
                    .into(),
            ));
        };
        for (&k, &coef) in idx.iter().zip(c.coefficients()) {
            if k >= d {
                return Err(BotorchError::InvalidArgument(format!(
                    "Constraint indices cannot exceed the problem dimension d={d}."
                )));
            }
            a[(i, k)] = coef;
        }
        b[i] = c.rhs();
    }
    Ok((a, b))
}

/// A point of the polytope `{x : A x ≤ b, A_eq x = b_eq}` maximizing the slack
/// `s` of the inequalities (BoTorch's `find_interior_point`): the solution of
/// `min -s` subject to `A x + 2s ≤ b`, `s ≥ 0`, `A_eq x = b_eq` (with
/// `s ≤ 1` added if the problem is unbounded).
///
/// # Errors
/// [`BotorchError::Infeasibility`] with BoTorch's message if the polytope is
/// empty, [`BotorchError::InvalidArgument`] if the linear program fails
/// otherwise.
pub fn find_interior_point(
    a: &Matrix,
    b: &Vector,
    equality: Option<(&Matrix, &Vector)>,
) -> Result<Vector> {
    let d = a.ncols();
    let solve = |bounded: bool| -> std::result::Result<Vector, microlp::Error> {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x: Vec<_> = (0..d)
            .map(|_| problem.add_var(0.0, (f64::NEG_INFINITY, f64::INFINITY)))
            .collect();
        let s = problem.add_var(-1.0, (f64::NEG_INFINITY, f64::INFINITY));
        for i in 0..a.nrows() {
            let mut expr: Vec<(microlp::Variable, f64)> = (0..d)
                .filter(|&k| a[(i, k)] != 0.0)
                .map(|k| (x[k], a[(i, k)]))
                .collect();
            expr.push((s, 2.0));
            problem.add_constraint(expr, ComparisonOp::Le, b[i]);
        }
        problem.add_constraint([(s, -1.0)], ComparisonOp::Le, 0.0);
        if bounded {
            problem.add_constraint([(s, 1.0)], ComparisonOp::Le, 1.0);
        }
        if let Some((aeq, beq)) = equality {
            for i in 0..aeq.nrows() {
                let expr: Vec<(microlp::Variable, f64)> = (0..d)
                    .filter(|&k| aeq[(i, k)] != 0.0)
                    .map(|k| (x[k], aeq[(i, k)]))
                    .collect();
                problem.add_constraint(expr, ComparisonOp::Eq, beq[i]);
            }
        }
        let outcome = problem.solve()?;
        let solution = outcome
            .into_solution()
            .map_err(|_| microlp::Error::InternalError("the solve was interrupted".into()))?;
        Ok(Vector::from_fn(d, |k, _| solution.var_value(x[k])))
    };
    let result = match solve(false) {
        Err(microlp::Error::Unbounded) => solve(true),
        other => other,
    };
    match result {
        Ok(x) => Ok(x),
        Err(microlp::Error::Infeasible) => Err(BotorchError::Infeasibility(
            "No feasible point found. Constraint polytope appears empty. Check your constraints."
                .into(),
        )),
        Err(e) => Err(BotorchError::InvalidArgument(format!(
            "Problem checking constraint specification. linprog status: {e:?}"
        ))),
    }
}

/// Hit-and-run samples from `{x : A x ≤ b}` starting at the feasible point
/// `x0` (BoTorch's `sample_polytope`): `n` samples after `n0` burn-in steps,
/// keeping every `n_thinning`-th point. Step lengths come from `rng_steps`
/// and directions from `rng_directions` (BoTorch seeds them with `seed` and
/// `seed + 1`).
///
/// # Errors
/// [`BotorchError::Infeasibility`] if `x0` violates the constraints.
pub fn sample_polytope(
    a: &Matrix,
    b: &Vector,
    x0: &Vector,
    n: usize,
    n0: usize,
    n_thinning: usize,
    seed: Option<u64>,
) -> Result<Matrix> {
    let slack0 = a * x0 - b;
    if slack0.iter().any(|&v| v.is_nan() || v > 0.0) {
        return Err(BotorchError::Infeasibility(format!(
            "Starting point does not satisfy the constraints: max(A@x0-b) = {}.",
            slack0.max()
        )));
    }
    let n_thinning = n_thinning.max(1);
    // Rows of A that are all zero carry no information (equality constraints
    // absorbed into A); BoTorch removes them.
    let rows: Vec<usize> = (0..a.nrows())
        .filter(|&i| a.row(i).iter().any(|&v| v != 0.0))
        .collect();
    let a = Matrix::from_fn(rows.len(), a.ncols(), |i, k| a[(rows[i], k)]);
    let b = Vector::from_fn(rows.len(), |i, _| b[rows[i]]);
    let dim = x0.len();
    if dim == 0 {
        // The polytope is a single point (e.g. pinned by equality constraints).
        return Ok(Matrix::zeros(n, 0));
    }
    let seed = seed.unwrap_or_else(random_seed);
    let mut rng_steps = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut rng_directions = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(1));
    let n_tot = n0 + n * n_thinning;
    let mut out = Matrix::zeros(n, dim);
    let mut x = x0.clone();
    let mut r = Vector::zeros(dim);
    for i in 0..n_tot {
        let rnd: f64 = rng_steps.random();
        // A uniformly random direction (BoTorch's `sample_hypersphere`).
        if dim == 1 {
            r[0] = if rng_directions.random::<bool>() {
                1.0
            } else {
                -1.0
            };
        } else {
            loop {
                for v in r.iter_mut() {
                    *v = rng_directions.sample(StandardNormal);
                }
                let norm = r.norm();
                if norm > 0.0 {
                    r /= norm;
                    break;
                }
            }
        }
        let ar = &a * &r;
        let slack = &b - &a * &x;
        let mut alpha_max = f64::INFINITY;
        let mut alpha_min = f64::NEG_INFINITY;
        let (mut any_pos, mut any_neg) = (false, false);
        let (mut boundary_up, mut boundary_down) = (false, false);
        for j in 0..a.nrows() {
            let w = slack[j].max(0.0) / ar[j];
            if w > 0.0 {
                any_pos = true;
                alpha_max = alpha_max.min(w);
            } else if w < 0.0 {
                any_neg = true;
                alpha_min = alpha_min.max(w);
            } else if w == 0.0 {
                if ar[j] > 0.0 {
                    boundary_up = true;
                } else if ar[j] < 0.0 {
                    boundary_down = true;
                }
            }
        }
        if !any_pos {
            alpha_max = LARGE_CONSTANT;
        }
        if !any_neg {
            alpha_min = -LARGE_CONSTANT;
        }
        if boundary_up {
            alpha_max = alpha_max.min(0.0);
        }
        if boundary_down {
            alpha_min = alpha_min.max(0.0);
        }
        let alpha = alpha_min + rnd * (alpha_max - alpha_min);
        x += alpha * &r;
        if i >= n0 {
            let k = i - n0;
            if k.is_multiple_of(n_thinning) {
                out.row_mut(k / n_thinning).copy_from(&x.transpose());
            }
        }
    }
    Ok(out)
}

/// A hit-and-run sampler of the polytope `{x : A x ≤ b, C x = e, lower ≤ x ≤
/// upper}` (BoTorch's `HitAndRunPolytopeSampler`).
#[derive(Clone, Debug)]
pub struct HitAndRunPolytopeSampler {
    a: Matrix,
    b: Vector,
    null_c: Matrix,
    new_a: Matrix,
    x0: Vector,
    n_burnin: usize,
    n_thinning: usize,
    seed: Option<u64>,
    num_samples_generated: usize,
    offset: Option<Vector>,
    scale: Option<Vector>,
}

impl HitAndRunPolytopeSampler {
    /// A sampler for inequality constraints `A x ≤ b`, equality constraints
    /// `C x = e` and box `bounds` (`2 × d`; infinite entries are unbounded).
    /// With bounds, the constraints are normalized to the unit cube first.
    /// The chain starts at `interior_point` (which must be feasible) or at the
    /// point found by [`find_interior_point`], discards `n_burnin` steps
    /// before the first sample and keeps every `n_thinning`-th point
    /// (BoTorch's defaults: 200 and 20).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] without inequality constraints and
    /// bounds or for inconsistent shapes, [`BotorchError::Infeasibility`] for
    /// an infeasible `interior_point` or an empty polytope.
    pub fn new(
        inequality: Option<(Matrix, Vector)>,
        equality: Option<(Matrix, Vector)>,
        bounds: Option<&Matrix>,
        interior_point: Option<Vector>,
        n_burnin: usize,
        n_thinning: usize,
        seed: Option<u64>,
    ) -> Result<Self> {
        if inequality.is_none() && bounds.is_none() {
            return Err(BotorchError::InvalidArgument(
                "HitAndRunPolytopeSampler requires either inequality constraints or bounds.".into(),
            ));
        }
        let d = inequality
            .as_ref()
            .map(|(a, _)| a.ncols())
            .or(bounds.map(|bd| bd.ncols()))
            .expect("inequality constraints or bounds");
        for (name, c) in [("inequality", &inequality), ("equality", &equality)] {
            if let Some((a, b)) = c {
                if a.ncols() != d || a.nrows() != b.len() {
                    return Err(BotorchError::InvalidArgument(format!(
                        "{name} constraints must be (A: k x {d}, b: k), got {}x{} and {}",
                        a.nrows(),
                        a.ncols(),
                        b.len()
                    )));
                }
            }
        }
        if let Some(bd) = bounds {
            if bd.shape() != (2, d) {
                return Err(BotorchError::InvalidArgument(format!(
                    "bounds must be 2 x {d}, got {}x{}",
                    bd.nrows(),
                    bd.ncols()
                )));
            }
        }
        let (mut inequality, mut equality, mut interior_point) =
            (inequality, equality, interior_point);
        let mut offset = None;
        let mut scale = None;
        let mut unit_bounds = bounds.cloned();
        if let (Some(bd), true) = (bounds, inequality.is_some() || equality.is_some()) {
            // normalize_dense_linear_constraints: A (u - l), b - A l.
            let lower = bd.row(0).transpose();
            let upper = bd.row(1).transpose();
            let range = &upper - &lower;
            let normalize = |(a, b): (Matrix, Vector)| -> (Matrix, Vector) {
                let al = &a * &lower;
                let mut an = a;
                for k in 0..d {
                    an.column_mut(k).scale_mut(range[k]);
                }
                (an, b - al)
            };
            inequality = inequality.map(normalize);
            equality = equality.map(normalize);
            interior_point = interior_point.map(|p| (p - &lower).component_div(&range));
            offset = Some(lower);
            scale = Some(range);
            let mut ub = Matrix::zeros(2, d);
            ub.row_mut(1).fill(1.0);
            unit_bounds = Some(ub);
        }
        // PolytopeSampler.__init__: append the bounds as inequalities.
        let (mut a, mut b) = inequality.unwrap_or_else(|| (Matrix::zeros(0, d), Vector::zeros(0)));
        if let Some(bd) = &unit_bounds {
            let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
            for k in 0..d {
                if bd[(0, k)].is_finite() {
                    let mut r = vec![0.0; d];
                    r[k] = -1.0;
                    rows.push((r, -bd[(0, k)]));
                }
            }
            for k in 0..d {
                if bd[(1, k)].is_finite() {
                    let mut r = vec![0.0; d];
                    r[k] = 1.0;
                    rows.push((r, bd[(1, k)]));
                }
            }
            let n0 = a.nrows();
            let mut a2 = Matrix::zeros(n0 + rows.len(), d);
            let mut b2 = Vector::zeros(n0 + rows.len());
            a2.rows_mut(0, n0).copy_from(&a);
            b2.rows_mut(0, n0).copy_from(&b);
            for (i, (r, v)) in rows.into_iter().enumerate() {
                a2.row_mut(n0 + i).copy_from_slice(&r);
                b2[n0 + i] = v;
            }
            a = a2;
            b = b2;
        }
        let null_c = match &equality {
            Some((c, _)) => null_space(c),
            None => Matrix::identity(d, d),
        };
        let new_a = &a * &null_c;
        let x0 = match interior_point {
            Some(p) => {
                let ineq_ok = (&a * &p - &b).iter().all(|&v| v <= 0.0);
                let eq_ok = equality
                    .as_ref()
                    .is_none_or(|(c, e)| (c * &p - e).iter().all(|&v| v == 0.0));
                if !(ineq_ok && eq_ok) {
                    return Err(BotorchError::Infeasibility(
                        "The given input point is not feasible.".into(),
                    ));
                }
                p
            }
            None => find_interior_point(&a, &b, equality.as_ref().map(|(c, e)| (c, e)))?,
        };
        Ok(Self {
            a,
            b,
            null_c,
            new_a,
            x0,
            n_burnin,
            n_thinning,
            seed,
            num_samples_generated: 0,
            offset,
            scale,
        })
    }

    /// Draws `n` samples (`n × d`); the chain continues from the last sample
    /// on the next call (without burn-in), and the seed advances by `n` as in
    /// BoTorch.
    ///
    /// # Errors
    /// [`BotorchError::Infeasibility`] if the current point left the polytope
    /// (only possible through round-off).
    pub fn draw(&mut self, n: usize) -> Result<Matrix> {
        let k = self.null_c.ncols();
        let n0 = if self.num_samples_generated == 0 {
            self.n_burnin
        } else {
            0
        };
        let shifted_b = &self.b - &self.a * &self.x0;
        let transformed = sample_polytope(
            &self.new_a,
            &shifted_b,
            &Vector::zeros(k),
            n,
            n0,
            self.n_thinning,
            self.seed,
        )?;
        if let Some(s) = self.seed.as_mut() {
            *s = s.wrapping_add(n as u64);
        }
        let mut samples = transformed * self.null_c.transpose();
        for mut row in samples.row_iter_mut() {
            row += self.x0.transpose();
        }
        if n > 0 {
            self.x0 = samples.row(n - 1).transpose();
        }
        if let (Some(offset), Some(scale)) = (&self.offset, &self.scale) {
            for mut row in samples.row_iter_mut() {
                for k in 0..row.len() {
                    row[k] = offset[k] + scale[k] * row[k];
                }
            }
        }
        self.num_samples_generated += n;
        Ok(samples)
    }
}

/// An orthonormal basis (columns) of the null space of `c` (`k × d`), from the
/// singular value decomposition (BoTorch: `Vh[rank:].T` of `torch.linalg.svd`;
/// the rank counts singular values above `max(k, d) · ε · σ_max` instead of
/// exactly non-zero ones).
fn null_space(c: &Matrix) -> Matrix {
    let (k, d) = c.shape();
    // Pad to a square matrix so that the SVD returns all right singular vectors.
    let mut padded = Matrix::zeros(d.max(k), d);
    padded.rows_mut(0, k).copy_from(c);
    let svd = nalgebra::linalg::SVD::new(padded, false, true);
    let v_t = svd.v_t.expect("right singular vectors requested");
    let s = &svd.singular_values;
    let smax = s.iter().copied().fold(0.0, f64::max);
    let tol = (k.max(d) as f64) * f64::EPSILON * smax;
    let null: Vec<usize> = (0..s.len())
        .filter(|&i| s[i].is_nan() || s[i] <= tol)
        .collect();
    Matrix::from_fn(d, null.len(), |r, j| v_t[(null[j], r)])
}

/// `n` samples from the polytope given by `bounds` (`2 × d`) and intra-point
/// linear constraints (BoTorch's `get_polytope_samples`, with its defaults
/// `n_burnin = 10000`, `n_thinning = 32`).
///
/// # Errors
/// See [`HitAndRunPolytopeSampler::new`] and [`sparse_to_dense_constraints`].
pub fn get_polytope_samples(
    n: usize,
    bounds: &Matrix,
    inequality: &[LinearConstraint],
    equality: &[LinearConstraint],
    seed: Option<u64>,
    n_burnin: usize,
    n_thinning: usize,
) -> Result<Matrix> {
    let d = bounds.ncols();
    let ineq = if inequality.is_empty() {
        None
    } else {
        // `A x >= b` becomes `-A x <= -b`.
        let (a, b) = sparse_to_dense_constraints(d, inequality)?;
        Some((-a, -b))
    };
    let eq = if equality.is_empty() {
        None
    } else {
        Some(sparse_to_dense_constraints(d, equality)?)
    };
    let mut sampler =
        HitAndRunPolytopeSampler::new(ineq, eq, Some(bounds), None, n_burnin, n_thinning, seed)?;
    sampler.draw(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_point_and_samples_satisfy_the_constraints() {
        // [0, 2] x [0, 1] with x0 + x1 >= 0.5 and x0 - x1 <= 1.
        let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 2.0, 1.0]);
        let ineq = vec![
            LinearConstraint::intra_point(vec![0, 1], vec![1.0, 1.0], 0.5).unwrap(),
            LinearConstraint::intra_point(vec![0, 1], vec![-1.0, 1.0], -1.0).unwrap(),
        ];
        let x = get_polytope_samples(2000, &bounds, &ineq, &[], Some(3), 500, 5).unwrap();
        assert_eq!(x.shape(), (2000, 2));
        for i in 0..x.nrows() {
            let (a, b) = (x[(i, 0)], x[(i, 1)]);
            assert!((-1e-12..=2.0 + 1e-12).contains(&a) && (-1e-12..=1.0 + 1e-12).contains(&b));
            assert!(a + b >= 0.5 - 1e-12 && a - b <= 1.0 + 1e-12);
        }
        // Uniform on the polytope: the mean of x1 equals its centroid.
        let mean1 = x.column(1).mean();
        // Area-weighted centroid computed by fine grid integration.
        let (mut area, mut m1) = (0.0, 0.0);
        let h = 1e-3;
        for i in 0..2000 {
            for j in 0..1000 {
                let (a, b) = ((i as f64 + 0.5) * h, (j as f64 + 0.5) * h);
                if a + b >= 0.5 && a - b <= 1.0 {
                    area += 1.0;
                    m1 += b;
                }
            }
        }
        assert!((mean1 - m1 / area).abs() < 0.03, "{mean1} vs {}", m1 / area);
        // Equality constraints: samples on the hyperplane x0 + x1 + x2 = 1.
        let b3 = Matrix::from_row_slice(2, 3, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let eq = vec![LinearConstraint::intra_point(vec![0, 1, 2], vec![1.0; 3], 1.0).unwrap()];
        let x = get_polytope_samples(500, &b3, &[], &eq, Some(1), 100, 3).unwrap();
        for i in 0..x.nrows() {
            assert!((x.row(i).sum() - 1.0).abs() < 1e-12);
            assert!(x.row(i).iter().all(|&v| v >= -1e-12));
        }
        // Empty polytope.
        let bad = vec![LinearConstraint::intra_point(vec![0], vec![1.0], 3.0).unwrap()];
        assert!(matches!(
            get_polytope_samples(10, &bounds, &bad, &[], Some(0), 10, 1),
            Err(BotorchError::Infeasibility(_))
        ));
        // The interior point maximizes the slack (unique here: the center).
        let (a, b) = sparse_to_dense_constraints(2, &[]).unwrap();
        assert_eq!(a.nrows(), 0);
        let box_a = Matrix::from_row_slice(4, 2, &[-1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0]);
        let box_b = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let p = find_interior_point(&box_a, &box_b, None).unwrap();
        assert!(
            (p[0] - 0.5).abs() < 1e-12 && (p[1] - 0.5).abs() < 1e-12,
            "{p}"
        );
        let _ = b;
    }
}
