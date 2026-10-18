//! Exact GPs conditioned on fantasized observations at points that depend
//! on the candidates: the nodes of the fantasy trees of
//! `qMultiStepLookahead` (BoTorch's repeated `fantasize`).
//!
//! A [`CondGp`] is a [`SingleTaskGP`] whose training data are extended by
//! *appended* points (raw inputs and targets in the transformed outcome
//! space) with the model's hyperparameters and fitted transforms, like
//! BoTorch's fantasy models. The fantasy models of one node at common new
//! points share their [`CondFactor`] (the enlarged training covariance and
//! its Cholesky factor, updated by a rank update of the parent's factor
//! where possible) and differ only in their targets.
//!
//! The reverse passes give the derivatives of the posteriors at test points
//! with respect to the test points, the appended points and the appended
//! targets ([`CondGp::posterior_backward`], [`CondGp::appended_backward`]),
//! so that the derivative of a value computed at the leaves of a fantasy
//! tree with respect to every point of the tree is exact.

use std::sync::Arc;

use super::{PredictionCache, SingleTaskGP};
use crate::error::Result;
use crate::kernels::Prepared;
use crate::linalg::{psd_safe_cholesky, CholeskyFactor};
use crate::transforms::input::TransformMode;
use crate::{Matrix, Vector};

/// Rows of `a` followed by the rows of `b` (same number of columns).
fn vstack(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.nrows() + b.nrows(), a.ncols());
    out.rows_mut(0, a.nrows()).copy_from(a);
    out.rows_mut(a.nrows(), b.nrows()).copy_from(b);
    out
}

/// The training inputs, noise and factor shared by the fantasy models of
/// one node of a fantasy tree.
#[derive(Debug)]
pub(crate) struct CondFactor {
    /// Transformed training inputs (`N × d`): the model's, then the
    /// appended points.
    z_tf: Matrix,
    /// The training inputs prepared for the kernel.
    full_scaled: Prepared,
    /// Cholesky factor of `K(Z, Z) + diag(noise)` (transformed space).
    chol: CholeskyFactor,
    /// The noise variance (transformed space) of every training point.
    noise: Vec<f64>,
}

/// A single-output exact GP conditioned on appended observations (one
/// output of one node of a fantasy tree).
#[derive(Clone, Debug)]
pub(crate) struct CondGp<'a> {
    gp: &'a SingleTaskGP,
    cache: &'a PredictionCache,
    factor: Arc<CondFactor>,
    /// Transformed targets of all `N` training points.
    targets: Vector,
    /// `K⁻¹ (targets - c)`.
    alpha: Vector,
}

/// The posterior of a [`CondGp`] at test points with what its reverse pass
/// needs.
#[derive(Clone, Debug)]
pub(crate) struct CondPosterior {
    /// The prepared transformed test points.
    xs: Prepared,
    /// `K(Z, x)` (`N × p`).
    k_full: Matrix,
    /// `K⁻¹ K(Z, x)` (`N × p`), with the covariance only.
    w: Option<Matrix>,
    /// Posterior mean (original outcome space, `p`).
    pub mean: Vector,
    /// Posterior covariance (original outcome space, `p × p`), if requested.
    pub covariance: Option<Matrix>,
}

/// The adjoints that the reverse passes of a [`CondGp`]'s posteriors
/// accumulate for its appended training data (linear in the adjoints of the
/// posteriors).
#[derive(Clone, Debug)]
pub(crate) struct CondAdjoint {
    /// Adjoint of the appended targets (transformed space, `a`).
    pub tbar: Vector,
    /// Coefficients of `∂k(z_j, z_b)/∂z_j` of the appended points `z_j`
    /// from the adjoint of the training covariance (`a × N`).
    kbar: Matrix,
    /// Adjoint of the transformed appended points from the cross
    /// covariances with test points (`a × d`).
    xbar_tf: Matrix,
}

impl<'a> CondGp<'a> {
    /// The model itself (no appended points).
    ///
    /// # Errors
    /// [`crate::BotorchError::Unsupported`] for models whose fantasy models
    /// have no input derivatives here (one-to-many or non-affine input
    /// transforms, nonlinear outcome transforms), and errors of the
    /// factorization.
    pub fn root(gp: &'a SingleTaskGP) -> Result<Self> {
        gp.require_no_perturbation("fantasize")?;
        gp.require_plain_outcome("fantasy models with input derivatives")?;
        gp.require_affine_inputs("fantasy models with input derivatives")?;
        let cache = gp.cache()?;
        let likelihood_noise = gp.likelihood_noise();
        let noise = (0..gp.num_train())
            .map(|i| gp.training_noise(&likelihood_noise, i))
            .collect();
        Ok(Self {
            gp,
            cache,
            factor: Arc::new(CondFactor {
                z_tf: gp.train_x_tf.clone(),
                full_scaled: cache.train_scaled.clone(),
                chol: cache.chol.clone(),
                noise,
            }),
            targets: gp.train_y_tf.clone(),
            alpha: cache.alpha.clone(),
        })
    }

    /// The underlying model.
    pub fn gp(&self) -> &'a SingleTaskGP {
        self.gp
    }

    /// The number of training points `N` (the model's and the appended).
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// The number of appended points `a`.
    pub fn num_appended(&self) -> usize {
        self.len() - self.gp.num_train()
    }

    /// The transformed targets of all training points.
    pub fn targets(&self) -> &Vector {
        &self.targets
    }

    /// The noise variance (transformed space) of fantasized observations:
    /// the inferred noise level, or the mean training noise of a fixed-noise
    /// model (BoTorch's `fantasize` default).
    ///
    /// # Errors
    /// Errors of the noise of a fixed-noise model.
    pub fn fantasy_noise(&self) -> Result<f64> {
        let default = self.gp.default_fantasy_noise();
        Ok(self.gp.new_noise(default.map(|v| vec![v]).as_deref(), 1)?[0])
    }

    /// The observation-noise variance (transformed space) of the posterior
    /// from which observations are fantasized: the inferred noise level, or
    /// the mean training noise of a fixed-noise model (BoTorch's
    /// `fantasize`).
    pub fn posterior_noise(&self) -> Result<f64> {
        Ok(match self.gp.default_fantasy_noise() {
            Some(v) => v,
            None => self.gp.likelihood_noise()[0],
        })
    }

    /// The `(mean, stdv)` of the affine outcome transform.
    ///
    /// # Errors
    /// See [`SingleTaskGP::outcome_affine`].
    pub fn outcome_affine(&self) -> Result<(f64, f64)> {
        self.gp.outcome_affine()
    }

    /// The factor of the training data enlarged by the raw points `x_new`
    /// (`k × d`) with the noise variance `noise_tf` (transformed space), by
    /// a rank update of this node's factor where possible.
    ///
    /// # Errors
    /// Shape errors, NaN inputs and errors of the factorization.
    pub fn enlarge(&self, x_new: &Matrix, noise_tf: f64) -> Result<Arc<CondFactor>> {
        let kernel = &self.cache.kernel;
        let x_tf = self.gp.check_new_inputs(x_new, TransformMode::Eval)?;
        let k = x_tf.nrows();
        let z_tf = vstack(&self.factor.z_tf, &x_tf);
        let full_scaled = kernel.prepare(&z_tf)?;
        let new_scaled = kernel.prepare(&x_tf)?;
        let cross = kernel.cross(&new_scaled, &self.factor.full_scaled);
        let mut corner = kernel.symmetric(&new_scaled);
        for j in 0..k {
            corner[(j, j)] += noise_tf;
        }
        let mut noise = self.factor.noise.clone();
        noise.extend(std::iter::repeat_n(noise_tf, k));
        let chol = match self.factor.chol.bordered(&cross, &corner) {
            Some(c) => c,
            None => {
                let mut full = kernel.symmetric(&full_scaled);
                for (i, &v) in noise.iter().enumerate() {
                    full[(i, i)] += v;
                }
                psd_safe_cholesky(&full)?
            }
        };
        Ok(Arc::new(CondFactor {
            z_tf,
            full_scaled,
            chol,
            noise,
        }))
    }

    /// The model conditioned on the transformed targets `y_new_tf` at the
    /// points of `factor` (from [`CondGp::enlarge`] of this model).
    pub fn child(&self, factor: Arc<CondFactor>, y_new_tf: &[f64]) -> Self {
        let n = self.len();
        let mut targets = self
            .targets
            .clone()
            .resize_vertically(n + y_new_tf.len(), 0.0);
        for (j, &y) in y_new_tf.iter().enumerate() {
            targets[n + j] = y;
        }
        let alpha = factor
            .chol
            .solve_vec(&targets.add_scalar(-self.gp.constant()));
        Self {
            gp: self.gp,
            cache: self.cache,
            factor,
            targets,
            alpha,
        }
    }

    /// The posterior at the raw test points `x` (`p × d`), in the original
    /// outcome space, with its covariance if `with_cov`.
    ///
    /// # Errors
    /// [`crate::BotorchError::DimensionMismatch`] for a wrong number of
    /// columns.
    pub fn posterior(&self, x: &Matrix, with_cov: bool) -> Result<CondPosterior> {
        let xt = self.gp.transform_inputs(x)?;
        let kernel = &self.cache.kernel;
        let xs = kernel.prepare(&xt)?;
        let k_full = kernel.cross(&self.factor.full_scaled, &xs);
        let (mu, sd) = self.gp.outcome_affine()?;
        let c = self.gp.constant();
        let mean = k_full.tr_mul(&self.alpha).map(|v| mu + sd * (c + v));
        let (w, covariance) = if with_cov {
            let mut v = k_full.clone();
            self.factor.chol.solve_lower_mut(&mut v);
            let mut cov = kernel.symmetric(&xs);
            cov.gemm_tr(-1.0, &v, &v, 1.0);
            self.factor.chol.solve_lower_transpose_mut(&mut v);
            (Some(v), Some(cov.map(|c| sd * c * sd)))
        } else {
            (None, None)
        };
        Ok(CondPosterior {
            xs,
            k_full,
            w,
            mean,
            covariance,
        })
    }

    /// A zero accumulator for the reverse passes.
    pub fn adjoint(&self) -> CondAdjoint {
        let a = self.num_appended();
        CondAdjoint {
            tbar: Vector::zeros(a),
            kbar: Matrix::zeros(a, self.len()),
            xbar_tf: Matrix::zeros(a, self.gp.dim()),
        }
    }

    /// Reverse pass of [`CondGp::posterior`]: for the adjoints of the mean
    /// (`p`) and, if the covariance was computed, of the covariance (`p ×
    /// p`, only its symmetric part matters), returns the gradient with
    /// respect to the raw test points (`p × d`) and accumulates the adjoints
    /// of the appended training data in `acc`.
    pub fn posterior_backward(
        &self,
        post: &CondPosterior,
        mean_bar: &Vector,
        cov_bar: Option<&Matrix>,
        acc: &mut CondAdjoint,
    ) -> Matrix {
        let kernel = &self.cache.kernel;
        let n = self.gp.num_train();
        let n_full = self.len();
        let a = n_full - n;
        let p = post.mean.len();
        let d = self.gp.dim();
        let sd = self
            .gp
            .outcome_affine()
            .expect("checked when the root was built")
            .1;
        let alpha = &self.alpha;
        let mbar = mean_bar * sd;
        // Adjoint of K(Z, x) (N × p) and of K(x, x) (p × p).
        let mut kfull_bar = alpha * mbar.transpose();
        let abar = match (cov_bar, &post.w) {
            (Some(cb), Some(_)) => Some((cb + cb.transpose()) * (0.5 * sd * sd)),
            _ => None,
        };
        if a > 0 {
            // β = K⁻¹ K(Z, x) mbar: the adjoint of the targets.
            let beta = self.factor.chol.solve_vec(&(&post.k_full * &mbar));
            for j in 0..a {
                acc.tbar[j] += beta[n + j];
                for b in 0..n_full {
                    acc.kbar[(j, b)] -= beta[n + j] * alpha[b] + beta[b] * alpha[n + j];
                }
            }
        }
        if let (Some(am), Some(w)) = (&abar, &post.w) {
            kfull_bar.gemm(-2.0, w, am, 1.0);
            if a > 0 {
                // 2 (W A Wᵀ)[n + j, b] for the appended rows.
                let wa = w.rows(n, a) * am;
                acc.kbar.gemm(2.0, &wa, &w.transpose(), 1.0);
            }
        }
        // Gradient with respect to the test points (transformed space).
        let jac = self
            .gp
            .input_jacobian()
            .expect("checked when the root was built");
        let mut out = Matrix::zeros(p, d);
        for t in 0..p {
            let (_, dk) = kernel.cross_with_grad(&post.xs, t, &self.factor.full_scaled);
            let mut g = dk.tr_mul(&kfull_bar.column(t));
            if let Some(am) = &abar {
                let (_, dkk) = kernel.cross_with_grad(&post.xs, t, &post.xs);
                g += dkk.tr_mul(&am.column(t)) * 2.0;
            }
            for m in 0..d {
                out[(t, m)] = g[m] * jac[m];
            }
        }
        // The appended points' side of K(Z, x).
        for j in 0..a {
            let (_, dk) = kernel.cross_with_grad(&self.factor.full_scaled, n + j, &post.xs);
            let g = dk.tr_mul(&kfull_bar.row(n + j).transpose());
            for m in 0..d {
                acc.xbar_tf[(j, m)] += g[m];
            }
        }
        out
    }

    /// The gradient with respect to the raw appended points (`a × d`) of
    /// everything accumulated in `acc` for them (through the training
    /// covariance and the cross covariances with test points); the adjoint
    /// of the appended targets stays in `acc.tbar`.
    pub fn appended_backward(&self, acc: &CondAdjoint) -> Matrix {
        let kernel = &self.cache.kernel;
        let n = self.gp.num_train();
        let a = self.num_appended();
        let d = self.gp.dim();
        let mut xbar_tf = acc.xbar_tf.clone();
        for j in 0..a {
            let (_, dk) =
                kernel.cross_with_grad(&self.factor.full_scaled, n + j, &self.factor.full_scaled);
            let g = dk.tr_mul(&acc.kbar.row(j).transpose());
            for m in 0..d {
                xbar_tf[(j, m)] += g[m];
            }
        }
        let jac = self
            .gp
            .input_jacobian()
            .expect("checked when the root was built");
        Matrix::from_fn(a, d, |j, m| xbar_tf[(j, m)] * jac[m])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SingleTaskGPOptions;
    use crate::transforms::Normalize;

    fn model(fixed: bool) -> SingleTaskGP {
        let x = Matrix::from_fn(9, 2, |i, j| {
            ((i * 7 + j * 3) % 9) as f64 / 9.0 + 0.05 * j as f64
        });
        let y = Matrix::from_fn(9, 1, |i, _| {
            let s: f64 = x.row(i).iter().sum();
            (3.0 * s).sin() + 0.5 * s
        });
        let yvar = fixed.then(|| Matrix::from_fn(9, 1, |i, _| 0.01 + 0.002 * i as f64));
        let mut gp = SingleTaskGP::new(
            x,
            y,
            SingleTaskGPOptions {
                train_yvar: yvar,
                input_transform: Some(Normalize::new(2)),
                ..Default::default()
            },
        )
        .unwrap();
        let mut h = gp.hyperparameters().clone();
        h.raw_lengthscale = Vector::from_vec(vec![0.4, 0.7]);
        h.raw_constant = 0.2;
        if h.raw_noise.is_some() {
            h.raw_noise = Some(0.02);
        }
        gp.set_hyperparameters(h).unwrap();
        gp
    }

    /// Two levels of conditioning equal one conditioning on all points, and
    /// the derivatives of a function of the posterior of the twice
    /// conditioned model with respect to the test points, the appended
    /// points and the appended targets agree with central differences.
    #[test]
    fn nested_conditioning_and_gradients() {
        for fixed in [false, true] {
            let gp = model(fixed);
            let x1 = Matrix::from_row_slice(2, 2, &[0.3, 0.6, 0.7, 0.2]);
            let x2 = Matrix::from_row_slice(1, 2, &[0.45, 0.85]);
            let y1 = [0.4, -0.3];
            let y2 = [1.1];
            let xt = Matrix::from_row_slice(3, 2, &[0.1, 0.9, 0.55, 0.45, 0.8, 0.7]);
            let mean_bar = Vector::from_vec(vec![0.7, -0.4, 1.3]);
            let cov_bar = Matrix::from_fn(3, 3, |a, b| ((a * 2 + b) % 3) as f64 * 0.3 - 0.2);
            let s = |x1: &Matrix, x2: &Matrix, y1: &[f64], y2: &[f64], xt: &Matrix| -> f64 {
                let r = CondGp::root(&gp).unwrap();
                let noise = r.fantasy_noise().unwrap();
                let c1 = r.child(r.enlarge(x1, noise).unwrap(), y1);
                let c2 = c1.child(c1.enlarge(x2, noise).unwrap(), y2);
                let p = c2.posterior(xt, true).unwrap();
                p.mean.dot(&mean_bar) + p.covariance.unwrap().component_mul(&cov_bar).sum()
            };
            // Reference: the model conditioned on all points at once.
            let r = CondGp::root(&gp).unwrap();
            let noise = r.fantasy_noise().unwrap();
            let c1 = r.child(r.enlarge(&x1, noise).unwrap(), &y1);
            let c2 = c1.child(c1.enlarge(&x2, noise).unwrap(), &y2);
            let (mu, sd) = gp.outcome_affine().unwrap();
            let all_x = vstack(&x1, &x2);
            let all_y = Matrix::from_fn(3, 1, |i, _| mu + sd * [y1[0], y1[1], y2[0]][i]);
            let noise_m = fixed.then(|| Matrix::from_element(3, 1, noise));
            let cond = gp
                .condition_on_observations(&all_x, &all_y, noise_m.as_ref())
                .unwrap();
            let a = c2.posterior(&xt, true).unwrap();
            let b = cond.posterior(&xt, false).unwrap();
            assert!((&a.mean - &b.mean).amax() < 1e-12);
            assert!((a.covariance.as_ref().unwrap() - &b.covariance).amax() < 1e-12);
            // Reverse pass.
            let mut acc = c2.adjoint();
            let gt = c2.posterior_backward(&a, &mean_bar, Some(&cov_bar), &mut acc);
            let ga = c2.appended_backward(&acc);
            let h = 1e-6;
            let check = |fd: f64, an: f64, what: &str| {
                assert!(
                    (fd - an).abs() < 1e-6 * fd.abs().max(1.0),
                    "fixed={fixed} {what}: {an} vs {fd}"
                );
            };
            for r in 0..3 {
                for c in 0..2 {
                    let (mut p, mut m) = (xt.clone(), xt.clone());
                    p[(r, c)] += h;
                    m[(r, c)] -= h;
                    let fd = (s(&x1, &x2, &y1, &y2, &p) - s(&x1, &x2, &y1, &y2, &m)) / (2.0 * h);
                    check(fd, gt[(r, c)], "test points");
                }
            }
            for r in 0..3 {
                for c in 0..2 {
                    let (mut p1, mut m1, mut p2, mut m2) =
                        (x1.clone(), x1.clone(), x2.clone(), x2.clone());
                    if r < 2 {
                        p1[(r, c)] += h;
                        m1[(r, c)] -= h;
                    } else {
                        p2[(0, c)] += h;
                        m2[(0, c)] -= h;
                    }
                    let fd = (s(&p1, &p2, &y1, &y2, &xt) - s(&m1, &m2, &y1, &y2, &xt)) / (2.0 * h);
                    check(fd, ga[(r, c)], "appended points");
                }
            }
            for r in 0..3 {
                let (mut p1, mut m1, mut p2, mut m2) = (y1, y1, y2, y2);
                if r < 2 {
                    p1[r] += h;
                    m1[r] -= h;
                } else {
                    p2[0] += h;
                    m2[0] -= h;
                }
                let fd = (s(&x1, &x2, &p1, &p2, &xt) - s(&x1, &x2, &m1, &m2, &xt)) / (2.0 * h);
                check(fd, acc.tbar[r], "appended targets");
            }
        }
    }
}
