//! The linear elliptical slice sampler of BoTorch
//! (`botorch.utils.probability.lin_ess.LinearEllipticalSliceSampler`):
//! Markov chains on a multivariate normal `N(mean, Σ)` truncated to a polytope
//! `A x ≤ b`, with fixed features and parallel chains (Gessner, Kanjilal and
//! Hennig, *Integrals over Gaussians under linear domain constraints*,
//! AISTATS 2020; the active intervals of Wu and Gardner, *A fast, robust
//! elliptical slice sampling implementation for linearly truncated
//! multivariate normal distributions*, 2024).
//!
//! The constraints are rewritten for a standard normal `z` (`x = L z + mean`
//! on the free coordinates); every step draws a direction `ν ~ N(0, I)` and a
//! uniform number per chain and moves each chain to a uniformly drawn angle
//! of the feasible arcs of the ellipse `z cos θ + ν sin θ`. The random numbers
//! are passed in (the Python layer draws them with torch's generator in
//! BoTorch's order), so that the chains agree with BoTorch's.

use std::f64::consts::PI;

use crate::error::{BotorchError, Result};
use crate::{Matrix, Vector};

/// The root of the covariance of the free coordinates.
#[derive(Clone, Debug)]
pub enum CovarianceRoot {
    /// A lower-triangular root (`k × k`).
    Lower(Matrix),
    /// A diagonal root (BoTorch's `DiagLinearOperator`).
    Diagonal(Vector),
}

/// A linear elliptical slice sampler (see the module documentation).
#[derive(Clone, Debug)]
pub struct LinearEllipticalSliceSampler {
    a: Matrix,
    b: Vector,
    az: Matrix,
    bz: Vector,
    mean: Option<Vector>,
    root: Option<CovarianceRoot>,
    fixed: Option<(Vec<usize>, Vec<usize>)>,
    z: Matrix,
    x: Matrix,
    single_precision: bool,
    lifetime_samples: usize,
}

impl LinearEllipticalSliceSampler {
    /// A sampler of `N(mean, Σ)` restricted to `A x ≤ b` (`A`: `m × d`),
    /// started at the interior point `x0` (`d`). `mean` and the covariance
    /// (`covariance_matrix`, factorized by Cholesky, or its lower-triangular
    /// or diagonal `root`) are those of all `d` coordinates; with
    /// `fixed_indices` the fixed coordinates keep the values of `x0` and the
    /// root must be diagonal (BoTorch's restriction). `single_precision`
    /// selects float32's trimming of the active intervals.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for both a covariance and a root,
    /// a non-diagonal root with fixed indices, a covariance that is not
    /// positive definite (BoTorch's `ValueError` messages) and inconsistent
    /// shapes.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: Matrix,
        b: Vector,
        x0: Vector,
        fixed_indices: Option<Vec<usize>>,
        mean: Option<Vector>,
        covariance_matrix: Option<Matrix>,
        covariance_root: Option<CovarianceRoot>,
        num_chains: usize,
        single_precision: bool,
    ) -> Result<Self> {
        let d = a.ncols();
        if b.len() != a.nrows() || x0.len() != d || num_chains == 0 {
            return Err(BotorchError::InvalidArgument(
                "inconsistent shapes of the constraints, the interior point or the chains".into(),
            ));
        }
        if covariance_matrix.is_some() && covariance_root.is_some() {
            return Err(BotorchError::InvalidArgument(
                "Provide either covariance_matrix or covariance_root, not both.".into(),
            ));
        }
        let mut az = a.clone();
        let mut bz = b.clone();
        let mut mean = mean;
        let mut cov = covariance_matrix;
        let mut root = covariance_root;
        let mut fixed = None;
        if let Some(fixed_indices) = fixed_indices {
            // A dense root must be diagonal.
            if let Some(CovarianceRoot::Lower(l)) = &root {
                let is_diag = (0..l.nrows())
                    .all(|r| (0..l.ncols()).all(|c| r == c || l[(r, c)] == 0.0));
                if !is_diag {
                    return Err(BotorchError::InvalidArgument(
                        "Provide either covariance_root or fixed_indices, not both.".into(),
                    ));
                }
                root = Some(CovarianceRoot::Diagonal(l.diagonal()));
            }
            let is_fixed: Vec<usize> = fixed_indices;
            let not_fixed: Vec<usize> = (0..d).filter(|i| !is_fixed.contains(i)).collect();
            az = Matrix::from_fn(a.nrows(), not_fixed.len(), |r, c| a[(r, not_fixed[c])]);
            for r in 0..a.nrows() {
                let mut s = 0.0;
                for &k in &is_fixed {
                    s += a[(r, k)] * x0[k];
                }
                bz[r] = b[r] - s;
            }
            mean = mean.map(|m| Vector::from_fn(not_fixed.len(), |i, _| m[not_fixed[i]]));
            cov = cov.map(|c| {
                Matrix::from_fn(not_fixed.len(), not_fixed.len(), |r, k| {
                    c[(not_fixed[r], not_fixed[k])]
                })
            });
            if let Some(CovarianceRoot::Diagonal(dg)) = &root {
                root = Some(CovarianceRoot::Diagonal(Vector::from_fn(
                    not_fixed.len(),
                    |i, _| dg[not_fixed[i]],
                )));
            }
            fixed = Some((is_fixed, not_fixed));
        }
        if let Some(c) = cov {
            let l = c.cholesky().ok_or_else(|| {
                BotorchError::InvalidArgument(
                    "Covariance matrix is not positive definite. Currently only non-degenerate \
                     distributions are supported."
                        .into(),
                )
            })?;
            root = Some(CovarianceRoot::Lower(l.l()));
        }
        let k = az.ncols();
        if mean.as_ref().is_some_and(|m| m.len() != k)
            || root.as_ref().is_some_and(|r| match r {
                CovarianceRoot::Lower(l) => l.shape() != (k, k),
                CovarianceRoot::Diagonal(v) => v.len() != k,
            })
        {
            return Err(BotorchError::InvalidArgument(
                "inconsistent shapes of the mean and the covariance".into(),
            ));
        }
        // Standardization: b - A mean first, then A L.
        if let Some(m) = &mean {
            bz -= &az * m;
        }
        match &root {
            Some(CovarianceRoot::Lower(l)) => az = &az * l,
            Some(CovarianceRoot::Diagonal(v)) => {
                for c in 0..k {
                    az.column_mut(c).scale_mut(v[c]);
                }
            }
            None => {}
        }
        let mut sampler = Self {
            a,
            b,
            az,
            bz,
            mean,
            root,
            fixed,
            z: Matrix::zeros(k, 1),
            x: Matrix::from_column_slice(d, 1, x0.as_slice()),
            single_precision,
            lifetime_samples: 0,
        };
        let z0 = sampler.transform(&x0);
        sampler.z = Matrix::from_fn(k, num_chains, |r, _| z0[r]);
        sampler.x = Matrix::from_fn(d, num_chains, |r, _| x0[r]);
        Ok(sampler)
    }

    /// The number of steps taken.
    pub fn lifetime_samples(&self) -> usize {
        self.lifetime_samples
    }

    /// The standardized constraints `(A_z, b_z)`.
    pub fn standardized_constraints(&self) -> (&Matrix, &Vector) {
        (&self.az, &self.bz)
    }

    /// The current points (`d × num_chains`).
    pub fn current(&self) -> &Matrix {
        &self.x
    }

    /// The current standardized points (`k × num_chains`).
    pub fn current_standardized(&self) -> &Matrix {
        &self.z
    }

    /// The number of chains.
    pub fn num_chains(&self) -> usize {
        self.z.ncols()
    }

    /// The dimension of the free (standardized) coordinates.
    pub fn free_dim(&self) -> usize {
        self.z.nrows()
    }

    fn transform(&self, x: &Vector) -> Vector {
        let mut z = match &self.fixed {
            Some((_, nf)) => Vector::from_fn(nf.len(), |i, _| x[nf[i]]),
            None => x.clone(),
        };
        if let Some(m) = &self.mean {
            z -= m;
        }
        match &self.root {
            Some(CovarianceRoot::Lower(l)) => l
                .solve_lower_triangular(&z)
                .unwrap_or_else(|| Vector::from_element(z.len(), f64::NAN)),
            Some(CovarianceRoot::Diagonal(v)) => z.component_div(v),
            None => z,
        }
    }

    fn untransform(&self, z: &Matrix) -> Matrix {
        let mut xs = match &self.root {
            Some(CovarianceRoot::Lower(l)) => l * z,
            Some(CovarianceRoot::Diagonal(v)) => {
                let mut out = z.clone();
                for r in 0..out.nrows() {
                    out.row_mut(r).scale_mut(v[r]);
                }
                out
            }
            None => z.clone(),
        };
        if let Some(m) = &self.mean {
            for mut col in xs.column_iter_mut() {
                col += m;
            }
        }
        match &self.fixed {
            Some((_, nf)) => {
                let mut x = self.x.clone();
                for (i, &r) in nf.iter().enumerate() {
                    for c in 0..x.ncols() {
                        x[(r, c)] = xs[(i, c)];
                    }
                }
                x
            }
            None => xs,
        }
    }

    /// The intersection angles of the ellipses `z cos θ + ν sin θ` with the
    /// constraint boundaries: `(alpha, beta)`, each `m × num_chains`, the
    /// smaller and larger angle in `[0, 2π)` (`alpha = beta = atan2(q, p)`
    /// for a constraint whose boundary misses the ellipse; BoTorch's
    /// `_find_intersection_angles`, which wraps negative angles with 2π
    /// rounded to float32; the exact 2π here).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for a direction of the wrong shape.
    pub fn intersection_angles(&self, nu: &Matrix) -> Result<(Matrix, Matrix)> {
        if nu.shape() != self.z.shape() {
            return Err(BotorchError::InvalidArgument(format!(
                "expected a direction of shape {:?}, got {:?}",
                self.z.shape(),
                nu.shape()
            )));
        }
        let p = &self.az * &self.z;
        let q = &self.az * nu;
        let (m, nc) = (p.nrows(), p.ncols());
        let mut alpha = Matrix::zeros(m, nc);
        let mut beta = Matrix::zeros(m, nc);
        for c in 0..nc {
            for r in 0..m {
                let radius = (p[(r, c)].powi(2) + q[(r, c)].powi(2)).sqrt();
                let ratio = self.bz[r] / radius;
                let arccos = if ratio < 1.0 { ratio.acos() } else { 0.0 };
                let arctan = q[(r, c)].atan2(p[(r, c)]);
                let mut t1 = arctan + arccos;
                let mut t2 = arctan - arccos;
                if t1 < 0.0 {
                    t1 += 2.0 * PI;
                }
                if t2 < 0.0 {
                    t2 += 2.0 * PI;
                }
                alpha[(r, c)] = t1.min(t2);
                beta[(r, c)] = t1.max(t2);
            }
        }
        Ok((alpha, beta))
    }

    /// The angles of the active arcs of every chain: `(left, right)` with
    /// `num_chains × (m + 1)` entries each (BoTorch's
    /// `_find_active_intersection_angles`).
    fn active_intervals(&self, nu: &Matrix) -> Result<(Matrix, Matrix)> {
        let (alpha, beta) = self.intersection_angles(nu)?;
        let (m, nc) = (alpha.nrows(), alpha.ncols());
        let mut left = Matrix::zeros(nc, m + 1);
        let mut right = Matrix::zeros(nc, m + 1);
        for c in 0..nc {
            let mut pairs: Vec<(f64, f64)> = (0..m).map(|r| (alpha[(r, c)], beta[(r, c)])).collect();
            pairs.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mut cummax = f64::NEG_INFINITY;
            for (j, (a, b)) in pairs.iter().enumerate() {
                cummax = cummax.max(*b);
                left[(c, j + 1)] = cummax;
                right[(c, j)] = *a;
            }
            right[(c, m)] = 2.0 * PI;
        }
        Ok((left, right))
    }

    /// The angle of every chain for the direction `nu` (`k × num_chains`)
    /// and the uniform numbers `u` (one per chain; BoTorch's `_draw_angle`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for draws of the wrong shapes.
    pub fn draw_angles(&self, nu: &Matrix, u: &[f64]) -> Result<Vec<f64>> {
        if u.len() != self.z.ncols() {
            return Err(BotorchError::InvalidArgument(format!(
                "expected {} uniform numbers, got {}",
                self.z.ncols(),
                u.len()
            )));
        }
        let (mut left, mut right) = self.active_intervals(nu)?;
        let cap = if self.single_precision { 1e-6 } else { 1e-12 };
        let nc = left.nrows();
        let w = left.ncols();
        let mut out = Vec::with_capacity(nc);
        for c in 0..nc {
            let mut csum = Vec::with_capacity(w);
            let mut acc = 0.0;
            for j in 0..w {
                let gap = (right[(c, j)] - left[(c, j)]).max(0.0);
                let eps = (gap * 0.25).min(cap);
                left[(c, j)] += eps;
                right[(c, j)] -= eps;
                acc += (right[(c, j)] - left[(c, j)]).max(0.0);
                csum.push(acc);
            }
            let uu = csum[w - 1] * u[c];
            // torch.searchsorted (left): the first index with csum >= uu.
            let idx = csum.partition_point(|&v| v < uu).min(w - 1);
            let prev = if idx == 0 { 0.0 } else { csum[idx - 1] };
            out.push(uu - prev + left[(c, idx)]);
        }
        Ok(out)
    }

    /// One step of every chain with the direction `nu` (`k × num_chains`)
    /// and the uniform numbers `u`; returns the new points (`d ×
    /// num_chains`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for draws of the wrong shapes,
    /// [`BotorchError::Runtime`] with `check_feasibility` if a point violates
    /// the constraints (BoTorch's `RuntimeError`).
    pub fn step(&mut self, nu: &Matrix, u: &[f64], check_feasibility: bool) -> Result<&Matrix> {
        if nu.shape() != self.z.shape() || u.len() != self.z.ncols() {
            return Err(BotorchError::InvalidArgument(format!(
                "expected draws of shapes {:?} and ({},), got {:?} and ({},)",
                self.z.shape(),
                self.z.ncols(),
                nu.shape(),
                u.len()
            )));
        }
        let theta = self.draw_angles(nu, u)?;
        let mut z = self.z.clone();
        for c in 0..z.ncols() {
            let (s, co) = theta[c].sin_cos();
            for r in 0..z.nrows() {
                z[(r, c)] = self.z[(r, c)] * co + nu[(r, c)] * s;
            }
        }
        self.z = z;
        self.x = self.untransform(&self.z);
        self.lifetime_samples += 1;
        if check_feasibility {
            let axmb = &self.a * &self.x;
            let mut count = 0;
            let mut magnitudes = Vec::new();
            for c in 0..axmb.ncols() {
                for r in 0..axmb.nrows() {
                    let v = axmb[(r, c)] - self.b[r];
                    if v > 0.0 {
                        count += 1;
                        magnitudes.push(v);
                    }
                }
            }
            if count > 0 {
                return Err(BotorchError::Runtime(format!(
                    "Sampling resulted in infeasible point. \n\t- Number of violated \
                     constraints: {count}.\n\t- Magnitude of violations: {magnitudes:?}\n\t- If \
                     the error persists, please report this bug on GitHub."
                )));
            }
        }
        Ok(&self.x)
    }

    /// Whether the points (columns of `points`) satisfy `A x ≤ b` (or the
    /// standardized constraints with `transformed`).
    pub fn is_feasible(&self, points: &Matrix, transformed: bool) -> Vec<bool> {
        let (a, b) = if transformed {
            (&self.az, &self.bz)
        } else {
            (&self.a, &self.b)
        };
        let v = a * points;
        (0..v.ncols())
            .map(|c| (0..v.nrows()).all(|r| v[(r, c)] <= b[r]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_distr::{Distribution, StandardNormal};

    #[test]
    fn chains_stay_in_the_polytope_and_match_the_moments() {
        // N(0, I) on the box [-1, 1] x [0, 2]: mean of the second coordinate
        // is phi(0) - phi(2) over Phi(2) - Phi(0).
        let a = Matrix::from_row_slice(4, 2, &[-1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0]);
        let b = Vector::from_vec(vec![1.0, 0.0, 1.0, 2.0]);
        let x0 = Vector::from_vec(vec![0.0, 1.0]);
        let mut s =
            LinearEllipticalSliceSampler::new(a, b, x0, None, None, None, None, 4, false).unwrap();
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(0);
        let mut sum = 0.0;
        let mut count = 0.0;
        for _ in 0..20_000 {
            let nu = Matrix::from_fn(2, 4, |_, _| StandardNormal.sample(&mut rng));
            let u: Vec<f64> = (0..4).map(|_| rng.random::<f64>()).collect();
            let x = s.step(&nu, &u, true).unwrap().clone();
            for c in 0..4 {
                sum += x[(1, c)];
                count += 1.0;
            }
        }
        let p = crate::utils::probability::phi;
        let n = crate::utils::probability::ndtr;
        let expected = (p(0.0) - p(2.0)) / (n(2.0) - n(0.0));
        assert!((sum / count - expected).abs() < 0.02, "{} vs {expected}", sum / count);
        assert_eq!(s.lifetime_samples(), 20_000);
    }

    #[test]
    fn errors() {
        let a = Matrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let b = Vector::from_vec(vec![1.0]);
        let x0 = Vector::zeros(2);
        let cov = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(LinearEllipticalSliceSampler::new(
            a.clone(),
            b.clone(),
            x0.clone(),
            None,
            None,
            Some(cov),
            None,
            1,
            false
        )
        .is_err());
        let root = CovarianceRoot::Lower(Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.5, 1.0]));
        assert!(LinearEllipticalSliceSampler::new(
            a,
            b,
            x0,
            Some(vec![0]),
            None,
            None,
            Some(root),
            1,
            false
        )
        .is_err());
    }
}
