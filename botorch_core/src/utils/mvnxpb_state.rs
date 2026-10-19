//! The stateful API of BoTorch's `MVNXPB` and `PivotedCholesky`
//! (`botorch.utils.probability.{mvnxpb, linalg}`) for batches of problems:
//! partial solves (`solve(num_steps)`), the GGE pivot selection, explicit
//! pivots, the decomposition steps of the partial pivoted Cholesky factor,
//! `augment_cholesky` and `MVNXPB.augment`.
//!
//! The state of a batch of `b` problems in `n` variables is kept in flat
//! row-major arrays ([`MvnxpbState`]): per member the permutation, the
//! standardized bounds (`n × 2`), the factor (`n × n`, its lower triangle),
//! the factor's permutation and standard deviations, the plug-in estimates
//! and the log probabilities. Every member is processed on its own (BoTorch
//! batches the same steps). [`super::mvnxpb::mvnxpb_log_prob`] is the
//! one-shot solve of a fresh problem (with forward-mode derivatives).

#![allow(clippy::needless_range_loop)]

use crate::error::{BotorchError, Result};
use crate::linalg::psd_safe_cholesky_with;
use crate::utils::mvnxpb::{bvn, STANDARDIZED_RANGE};
use crate::utils::probability::{ndtr, phi};
use crate::Matrix;

/// The state of a batch of `MVNXPB` solvers (BoTorch's `mvnxpbState`), each
/// member in `n` variables, in flat row-major arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct MvnxpbState {
    /// Number of variables.
    pub n: usize,
    /// Number of batch members.
    pub batch: usize,
    /// The solver's step (shared by the batch).
    pub step: usize,
    /// The step of the pivoted Cholesky factor (shared by the batch).
    pub pc_step: usize,
    /// The solver's permutation (`batch × n`).
    pub perm: Vec<i64>,
    /// Standardized lower and upper bounds (`batch × n × 2`).
    pub bounds: Vec<f64>,
    /// The partial pivoted Cholesky factor (`batch × n × n`, lower triangle).
    pub tril: Vec<f64>,
    /// The factor's permutation (`batch × n`).
    pub pc_perm: Vec<i64>,
    /// The factor's standard deviations (`batch × n`), if any.
    pub diag: Option<Vec<f64>>,
    /// The plug-in estimates (`batch × n`, NaN before they are computed).
    pub plug_ins: Vec<f64>,
    /// The running log probabilities (`batch`).
    pub log_prob: Vec<f64>,
    /// The univariate log probability of a final unpaired variable (`batch`).
    pub log_prob_extra: Option<Vec<f64>>,
}

/// BoTorch's `safe_log`: `log(max(x, tiny))`.
fn safe_log(x: f64) -> f64 {
    x.max(f64::MIN_POSITIVE).ln()
}

/// BoTorch's `safe_mul`: `0 · ∞ = 0`.
fn safe_mul(a: f64, b: f64) -> f64 {
    if (a.is_infinite() && b == 0.0) || (b.is_infinite() && a == 0.0) {
        0.0
    } else {
        a * b
    }
}

/// One decomposition step of BoTorch's `PivotedCholesky.update_` on the
/// `n × n` factor `tril` (row-major) at `step`.
pub fn pivoted_cholesky_update(tril: &mut [f64], n: usize, step: usize, eps: f64) {
    let i = step;
    let lii = tril[i * n + i].max(0.0).sqrt();
    tril[i * n + i] = lii;
    for c in i + 1..n {
        tril[i * n + c] = 0.0;
    }
    for r in i + 1..n {
        tril[r * n + i] /= lii;
    }
    for r in i + 1..n {
        for c in i + 1..=r {
            let v = tril[r * n + i] * tril[c * n + i];
            tril[r * n + c] -= v;
        }
    }
    if lii <= i as f64 * eps {
        for r in i..n {
            tril[r * n + i] = 0.0;
        }
    }
}

/// BoTorch's `PivotedCholesky.pivot_` for one member: swaps the variables
/// `step` and `piv >= step` in the factor (rows of the computed columns and,
/// symmetrically, the Schur complement in lower-triangular storage), the
/// permutation and the standard deviations.
pub fn pivoted_cholesky_pivot(
    tril: &mut [f64],
    perm: &mut [i64],
    diag: Option<&mut [f64]>,
    n: usize,
    step: usize,
    piv: usize,
) {
    let i = step;
    perm.swap(i, piv);
    if let Some(d) = diag {
        d.swap(i, piv);
    }
    if piv == i {
        return;
    }
    for c in 0..i {
        tril.swap(i * n + c, piv * n + c);
    }
    tril.swap(i * n + i, piv * n + piv);
    for r in i + 1..piv {
        tril.swap(r * n + i, piv * n + r);
    }
    for r in piv + 1..n {
        tril.swap(r * n + i, r * n + piv);
    }
}

impl MvnxpbState {
    /// BoTorch's `MVNXPB(covariance_matrix, bounds)`: standardizes the
    /// covariances (`batch × n × n`) and bounds (`batch × n × 2`, clipped to
    /// `STANDARDIZED_RANGE` standard deviations first).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent lengths.
    pub fn new(cov: &[f64], bounds: &[f64], batch: usize, n: usize) -> Result<Self> {
        if cov.len() != batch * n * n || bounds.len() != batch * n * 2 {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {batch} covariance matrices of {n} x {n} and bounds of {n} x 2"
            )));
        }
        let (lo, hi) = STANDARDIZED_RANGE;
        let mut tril = vec![0.0; batch * n * n];
        let mut sb = vec![0.0; batch * n * 2];
        let mut diag = vec![0.0; batch * n];
        for b in 0..batch {
            let c = &cov[b * n * n..(b + 1) * n * n];
            let std: Vec<f64> = (0..n).map(|i| c[i * n + i].sqrt()).collect();
            let istd: Vec<f64> = (0..n).map(|i| 1.0 / c[i * n + i].sqrt()).collect();
            for r in 0..n {
                for k in 0..=r {
                    tril[b * n * n + r * n + k] = istd[r] * c[r * n + k] * istd[k];
                }
                for j in 0..2 {
                    let v = bounds[b * n * 2 + r * 2 + j];
                    let clipped = v.max(std[r] * lo).min(std[r] * hi);
                    sb[b * n * 2 + r * 2 + j] = istd[r] * clipped;
                }
                diag[b * n + r] = std[r];
            }
        }
        let perm: Vec<i64> = (0..batch).flat_map(|_| 0..n as i64).collect();
        Ok(Self {
            n,
            batch,
            step: 0,
            pc_step: 0,
            perm: perm.clone(),
            bounds: sb,
            tril,
            pc_perm: perm,
            diag: Some(diag),
            plug_ins: vec![f64::NAN; batch * n],
            log_prob: vec![0.0; batch],
            log_prob_extra: None,
        })
    }

    /// Checks the lengths of the arrays.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for inconsistent lengths.
    pub fn validate(&self) -> Result<()> {
        let (b, n) = (self.batch, self.n);
        let ok = self.perm.len() == b * n
            && self.bounds.len() == b * n * 2
            && self.tril.len() == b * n * n
            && self.pc_perm.len() == b * n
            && self.diag.as_ref().is_none_or(|d| d.len() == b * n)
            && self.plug_ins.len() == b * n
            && self.log_prob.len() == b
            && self.log_prob_extra.as_ref().is_none_or(|e| e.len() == b);
        if ok {
            Ok(())
        } else {
            Err(BotorchError::DimensionMismatch(
                "inconsistent shapes of the MVNXPB state".into(),
            ))
        }
    }

    /// BoTorch's `select_pivot`: per member, the index (`>= pc_step`) of the
    /// variable least likely to satisfy its bounds given the plug-in values
    /// of the integrated variables (GGE ordering).
    pub fn select_pivot(&self) -> Vec<usize> {
        let (n, i) = (self.n, self.pc_step);
        (0..self.batch)
            .map(|b| {
                let l = &self.tril[b * n * n..(b + 1) * n * n];
                let y = &self.plug_ins[b * n..(b + 1) * n];
                let bd = &self.bounds[b * n * 2..(b + 1) * n * 2];
                let mut best = i;
                let mut best_p = f64::INFINITY;
                for j in i..n {
                    let mut shift = 0.0;
                    for c in 0..i {
                        shift += l[j * n + c] * y[c];
                    }
                    let inv = 1.0 / l[j * n + j].max(0.0).sqrt();
                    let p = ndtr(inv * (bd[j * 2 + 1] - shift)) - ndtr(inv * (bd[j * 2] - shift));
                    if !best_p.is_nan() && (p.is_nan() || p < best_p) {
                        best = j;
                        best_p = p;
                    }
                }
                best
            })
            .collect()
    }

    /// BoTorch's `MVNXPB.pivot_`: swaps the variables at `step` and
    /// `pivot[b]` of every member (the factor too if it is at the same step).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if the solver ran ahead of the
    /// factor or a pivot is out of range.
    pub fn pivot(&mut self, pivot: &[usize]) -> Result<()> {
        let (n, i) = (self.n, self.step);
        if pivot.len() != self.batch || pivot.iter().any(|&p| p < i || p >= n) {
            return Err(BotorchError::InvalidArgument(format!(
                "invalid pivots {pivot:?} at step {i}"
            )));
        }
        if self.step > self.pc_step {
            return Err(BotorchError::InvalidArgument(
                "the solver ran ahead of the matrix decomposition".into(),
            ));
        }
        let update_chol = self.pc_step == self.step;
        for (b, &p) in pivot.iter().enumerate() {
            if update_chol {
                pivoted_cholesky_pivot(
                    &mut self.tril[b * n * n..(b + 1) * n * n],
                    &mut self.pc_perm[b * n..(b + 1) * n],
                    self.diag.as_mut().map(|d| &mut d[b * n..(b + 1) * n]),
                    n,
                    i,
                    p,
                );
            }
            self.perm.swap(b * n + i, b * n + p);
            self.bounds.swap(b * n * 2 + i * 2, b * n * 2 + p * 2);
            self.bounds
                .swap(b * n * 2 + i * 2 + 1, b * n * 2 + p * 2 + 1);
        }
        Ok(())
    }

    /// BoTorch's `PivotedCholesky.update_` of every member.
    pub fn update_cholesky(&mut self, eps: f64) {
        let n = self.n;
        for b in 0..self.batch {
            pivoted_cholesky_update(
                &mut self.tril[b * n * n..(b + 1) * n * n],
                n,
                self.pc_step,
                eps,
            );
        }
        self.pc_step += 1;
    }

    /// BoTorch's `MVNXPB.solve(num_steps, eps)`: runs `num_steps` steps of
    /// the bivariate conditioning (all remaining ones if `None` or 0) and
    /// returns the log probabilities.
    ///
    /// Unlike BoTorch, solving a complete problem again is a no-op (BoTorch
    /// raises `UnboundLocalError` for an odd number of variables).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if the solver ran ahead of the
    /// decomposition or more steps than variables are requested.
    pub fn solve(&mut self, num_steps: Option<usize>, eps: f64) -> Result<Vec<f64>> {
        self.validate()?;
        if self.step > self.pc_step {
            return Err(BotorchError::InvalidArgument(
                "Invalid state: solver ran ahead of matrix decomposition.".into(),
            ));
        }
        let n = self.n;
        let start = self.step;
        if n > start && start % 2 == 1 {
            if let Some(extra) = self.log_prob_extra.take() {
                for (lp, e) in self.log_prob.iter_mut().zip(extra) {
                    *lp -= e;
                }
            }
        }
        let num_steps = match num_steps {
            Some(k) if k > 0 => k,
            _ => n - start,
        };
        if start + num_steps > n {
            return Err(BotorchError::InvalidArgument(format!(
                "cannot run {num_steps} steps from step {start} of {n}"
            )));
        }
        let mut phi_i = vec![f64::NAN; self.batch];
        for i in start..start + num_steps {
            let update_chol = self.step == self.pc_step;
            let pivot: Vec<usize> = if update_chol {
                self.select_pivot()
            } else {
                (0..self.batch)
                    .map(|b| {
                        let target = self.pc_perm[b * n + i];
                        (i..n)
                            .find(|&j| self.perm[b * n + j] == target)
                            .ok_or_else(|| {
                                BotorchError::InvalidArgument(
                                    "the factor's permutation does not match the solver's".into(),
                                )
                            })
                    })
                    .collect::<Result<_>>()?
            };
            if pivot.iter().any(|&p| p > i) {
                self.pivot(&pivot)?;
            }
            for b in 0..self.batch {
                phi_i[b] = self.step_member(b, i, update_chol, eps);
            }
            self.step += 1;
            if update_chol {
                self.update_cholesky(eps);
            }
        }
        if self.step % 2 == 1 && num_steps > 0 {
            let extra: Vec<f64> = phi_i.iter().map(|&p| safe_log(p)).collect();
            for (lp, e) in self.log_prob.iter_mut().zip(&extra) {
                *lp += e;
            }
            self.log_prob_extra = Some(extra);
        }
        Ok(self.log_prob.clone())
    }

    /// Step `i` of member `b` (after pivoting): the plug-in estimate and, at
    /// odd steps, the bivariate block. Returns the univariate probability.
    fn step_member(&mut self, b: usize, i: usize, update_chol: bool, eps: f64) -> f64 {
        let n = self.n;
        let l = &self.tril[b * n * n..(b + 1) * n * n];
        let bd = &self.bounds[b * n * 2..(b + 1) * n * 2];
        let y = &mut self.plug_ins[b * n..(b + 1) * n];
        let mut lii = l[i * n + i];
        if update_chol {
            lii = lii.max(0.0).sqrt();
        }
        let inv_lii = 1.0 / lii;
        let mut bi = [bd[i * 2], bd[i * 2 + 1]];
        if i != 0 {
            let mut s = 0.0;
            for c in 0..i {
                s += l[i * n + c] * y[c];
            }
            bi = [bi[0] - s, bi[1] - s];
        }
        let (lb, ub) = (inv_lii * bi[0], inv_lii * bi[1]);
        let phi_i = ndtr(ub) - ndtr(lb);
        let small = phi_i <= i as f64 * eps;
        y[i] = if small && lb < -9.0 {
            ub
        } else if small && lb > 9.0 {
            lb
        } else if small {
            0.5 * (lb + ub)
        } else {
            (phi(lb) - phi(ub)) / phi_i
        };
        if i % 2 == 1 {
            let h = i - 1;
            let lhh = l[h * n + h];
            let lih = l[i * n + h];
            let std_i = (lii * lii + lih * lih).sqrt();
            let istds = [1.0 / lhh, 1.0 / std_i];
            let mut blk = [[bd[h * 2], bd[h * 2 + 1]], [bd[i * 2], bd[i * 2 + 1]]];
            if i > 1 {
                for (v, row) in [h, i].into_iter().enumerate() {
                    let mut s = 0.0;
                    for c in 0..i - 1 {
                        s += l[row * n + c] * y[c];
                    }
                    blk[v] = [blk[v][0] - s, blk[v][1] - s];
                }
            }
            let xl = safe_mul(istds[0], blk[0][0]);
            let xu = safe_mul(istds[0], blk[0][1]);
            let yl = safe_mul(istds[1], blk[1][0]);
            let yu = safe_mul(istds[1], blk[1][1]);
            let corr = lhh * lih * (istds[0] * istds[1]);
            let p = bvn(corr, xl, yl, xu, yu);
            let (zh, zi) = crate::utils::mvnxpb::bvnmom_with_p(corr, xl, yl, xu, yu, p);
            if p > 0.0 {
                y[h] = zh;
                y[i] = inv_lii * (std_i * zi - lih * zh);
            } else {
                y[h] = 0.0;
                y[i] = 0.0;
            }
            self.log_prob[b] += safe_log(p);
        }
        phi_i
    }

    /// BoTorch's `MVNXPB.augment`: adds `m` variables with covariance
    /// `cov` (`batch × m × m`), bounds (`batch × m × 2`) and cross-covariance
    /// with the current variables `cross` (`batch × m × n`, in the order of
    /// the current factor) to a solver whose factor is complete. Returns the
    /// augmented state and whether the joint covariance had to be refactorized
    /// (BoTorch's warning "Joint covariance matrix not positive definite,
    /// attempting recovery.").
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] if the factor is incomplete (BoTorch's
    /// `NotImplementedError`), [`BotorchError::NotPsd`] if the recovery
    /// fails, [`BotorchError::DimensionMismatch`] for inconsistent shapes.
    #[allow(clippy::too_many_arguments)]
    pub fn augment(
        &self,
        cov: &[f64],
        bounds: &[f64],
        cross: &[f64],
        m: usize,
        disable_pivoting: bool,
        jitter: Option<f64>,
        max_tries: Option<usize>,
    ) -> Result<(Self, bool)> {
        self.validate()?;
        let (n, nb) = (self.n, self.batch);
        if n != self.pc_step {
            return Err(BotorchError::Unsupported(
                "Augmentation of incomplete solutions not implemented yet.".into(),
            ));
        }
        if cov.len() != nb * m * m || bounds.len() != nb * m * 2 || cross.len() != nb * m * n {
            return Err(BotorchError::DimensionMismatch(
                "inconsistent shapes of the augmenting covariances and bounds".into(),
            ));
        }
        let (lo, hi) = STANDARDIZED_RANGE;
        let nm = n + m;
        let mut out = Self {
            n: nm,
            batch: nb,
            step: self.step,
            pc_step: if disable_pivoting { nm } else { n },
            perm: Vec::with_capacity(nb * nm),
            bounds: Vec::with_capacity(nb * nm * 2),
            tril: vec![0.0; nb * nm * nm],
            pc_perm: Vec::with_capacity(nb * nm),
            diag: Some(Vec::with_capacity(nb * nm)),
            plug_ins: Vec::with_capacity(nb * nm),
            log_prob: self.log_prob.clone(),
            log_prob_extra: self.log_prob_extra.clone(),
        };
        let mut recovered = false;
        for b in 0..nb {
            let c = &cov[b * m * m..(b + 1) * m * m];
            let std: Vec<f64> = (0..m).map(|i| c[i * m + i].sqrt()).collect();
            let istd: Vec<f64> = (0..m).map(|i| 1.0 / c[i * m + i].sqrt()).collect();
            let mut kmn = Matrix::zeros(m, n);
            for r in 0..m {
                for k in 0..n {
                    let mut v = istd[r] * cross[b * m * n + r * n + k];
                    if let Some(d) = &self.diag {
                        v *= 1.0 / d[b * n + k];
                    }
                    kmn[(r, k)] = v;
                }
            }
            let kmm = Matrix::from_fn(m, m, |r, k| istd[r] * c[r * m + k] * istd[k]);
            let lnn = Matrix::from_fn(n, n, |r, k| {
                if k <= r {
                    self.tril[b * n * n + r * n + k]
                } else {
                    0.0
                }
            });
            let mut l = match augment_cholesky(&lnn, &kmm, Some(&kmn), None, jitter) {
                Ok(l) => l,
                Err(BotorchError::NotPsd { .. }) => {
                    recovered = true;
                    let knn = &lnn * lnn.transpose();
                    let mut k = Matrix::zeros(nm, nm);
                    k.view_mut((0, 0), (n, n)).copy_from(&knn);
                    k.view_mut((0, n), (n, m)).copy_from(&kmn.transpose());
                    k.view_mut((n, 0), (m, n)).copy_from(&kmn);
                    k.view_mut((n, n), (m, m)).copy_from(&kmm);
                    psd_safe_cholesky_with(&k, jitter.unwrap_or(1e-8), max_tries.unwrap_or(3))?
                        .l()
                        .clone()
                }
                Err(e) => return Err(e),
            };
            if !disable_pivoting {
                let lmm = l.view((n, n), (m, m)).into_owned();
                let s = &lmm * lmm.transpose();
                for r in 0..m {
                    for k in 0..m {
                        l[(n + r, n + k)] = if k <= r { s[(r, k)] } else { 0.0 };
                    }
                }
            }
            for r in 0..nm {
                for k in 0..nm {
                    out.tril[b * nm * nm + r * nm + k] = l[(r, k)];
                }
            }
            let diag = out.diag.as_mut().expect("set above");
            match &self.diag {
                Some(d) => diag.extend_from_slice(&d[b * n..(b + 1) * n]),
                None => diag.extend(std::iter::repeat_n(1.0, n)),
            }
            diag.extend_from_slice(&std);
            out.perm.extend_from_slice(&self.perm[b * n..(b + 1) * n]);
            out.perm.extend(n as i64..nm as i64);
            out.pc_perm
                .extend_from_slice(&self.pc_perm[b * n..(b + 1) * n]);
            out.pc_perm.extend(n as i64..nm as i64);
            out.bounds
                .extend_from_slice(&self.bounds[b * n * 2..(b + 1) * n * 2]);
            for r in 0..m {
                for j in 0..2 {
                    let v = bounds[b * m * 2 + r * 2 + j];
                    out.bounds
                        .push(istd[r] * v.max(std[r] * lo).min(std[r] * hi));
                }
            }
            out.plug_ins
                .extend_from_slice(&self.plug_ins[b * n..(b + 1) * n]);
            out.plug_ins.extend(std::iter::repeat_n(f64::NAN, m));
        }
        Ok((out, recovered))
    }
}

/// BoTorch's `augment_cholesky`: the Cholesky factor of `[[Kaa, Kab], [Kba,
/// Kbb]]` from `Laa` (`Kaa = Laa Laaᵀ`), `Kbb` and either `Kba` or the
/// precomputed `Lba = Kba Laa⁻ᵀ`, with an optional `jitter` added to the
/// diagonal of `Kbb`.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] unless exactly one of `kba` and `lba`
/// is given (BoTorch's `ValueError`), [`BotorchError::NotPsd`] if the Schur
/// complement is not positive definite, [`BotorchError::DimensionMismatch`]
/// for inconsistent shapes.
pub fn augment_cholesky(
    laa: &Matrix,
    kbb: &Matrix,
    kba: Option<&Matrix>,
    lba: Option<&Matrix>,
    jitter: Option<f64>,
) -> Result<Matrix> {
    let lba = match (kba, lba) {
        (Some(kba), None) => {
            if kba.ncols() != laa.nrows() || kba.nrows() != kbb.nrows() {
                return Err(BotorchError::DimensionMismatch(
                    "inconsistent shapes of Laa, Kba and Kbb".into(),
                ));
            }
            // Lba = Kba Laa^{-T}: solve Laa Lba^T = Kba^T.
            let lt = laa
                .solve_lower_triangular(&kba.transpose())
                .ok_or(BotorchError::NotPsd { jitter: 0.0 })?;
            lt.transpose()
        }
        (None, Some(lba)) => lba.clone(),
        _ => {
            return Err(BotorchError::InvalidArgument(
                "One and only one of `Kba` or `Lba` must be provided.".into(),
            ))
        }
    };
    let mut s = kbb.clone();
    if let Some(j) = jitter {
        for i in 0..s.nrows() {
            s[(i, i)] += j;
        }
    }
    s -= &lba * lba.transpose();
    let lbb = s
        .clone()
        .cholesky()
        .ok_or(BotorchError::NotPsd {
            jitter: jitter.unwrap_or(0.0),
        })?
        .l();
    let (na, nb) = (laa.nrows(), kbb.nrows());
    let mut out = Matrix::zeros(na + nb, na + nb);
    out.view_mut((0, 0), (na, na)).copy_from(laa);
    out.view_mut((na, 0), (nb, na)).copy_from(&lba);
    out.view_mut((na, na), (nb, nb)).copy_from(&lbb);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::mvnxpb::mvnxpb_log_prob;

    fn cov3() -> (Vec<f64>, Vec<f64>) {
        let cov = vec![1.2, 0.5, -0.3, 0.5, 0.9, 0.2, -0.3, 0.2, 1.4];
        let bounds = vec![f64::NEG_INFINITY, 0.3, -0.5, f64::INFINITY, -1.0, 0.8];
        (cov, bounds)
    }

    #[test]
    fn stateful_solve_matches_one_shot() {
        let (cov, bounds) = cov3();
        let mut s = MvnxpbState::new(&cov, &bounds, 1, 3).unwrap();
        let lp = s.solve(None, 1e-10).unwrap();
        let c = Matrix::from_row_slice(3, 3, &cov);
        let b = Matrix::from_row_slice(3, 2, &bounds);
        let e = mvnxpb_log_prob(&c, &b).unwrap();
        assert!((lp[0] - e).abs() < 1e-15, "{} vs {e}", lp[0]);
        // Partial solves give the same result.
        let mut t = MvnxpbState::new(&cov, &bounds, 1, 3).unwrap();
        t.solve(Some(1), 1e-10).unwrap();
        assert!(t.log_prob_extra.is_some());
        t.solve(Some(1), 1e-10).unwrap();
        let lp2 = t.solve(None, 1e-10).unwrap();
        assert!((lp2[0] - e).abs() < 1e-15);
        // Solving again is a no-op.
        assert_eq!(t.solve(None, 1e-10).unwrap(), lp2);
    }

    #[test]
    fn augment_cholesky_reproduces_the_joint_factor() {
        let k = Matrix::from_row_slice(3, 3, &[2.0, 0.3, 0.5, 0.3, 1.0, 0.2, 0.5, 0.2, 1.5]);
        let laa = k.view((0, 0), (2, 2)).into_owned().cholesky().unwrap().l();
        let kba = k.view((2, 0), (1, 2)).into_owned();
        let kbb = k.view((2, 2), (1, 1)).into_owned();
        let l = augment_cholesky(&laa, &kbb, Some(&kba), None, None).unwrap();
        let full = k.clone().cholesky().unwrap().l();
        assert!((l - full).abs().max() < 1e-15);
        assert!(augment_cholesky(&laa, &kbb, None, None, None).is_err());
    }
}
