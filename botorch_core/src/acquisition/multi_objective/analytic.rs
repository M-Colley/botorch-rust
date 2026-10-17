//! Analytic Expected Hypervolume Improvement
//! (`botorch.acquisition.multi_objective.analytic.ExpectedHypervolumeImprovement`;
//! Yang et al., *Efficient computation of expected hypervolume improvement
//! using box decomposition algorithms*, J. Glob. Optim. 2019), with exact
//! gradients (Daulton et al., 2020).
//!
//! For independent Gaussian outcomes `Y_j ~ N(μ_j, σ_j²)` and a box
//! decomposition of the non-dominated space into cells `[l_k, u_k]`,
//!
//! ```text
//! EHVI = Σ_k Π_j (ψ(l_kj, l_kj) - ψ(l_kj, u_kj) + ν(l_kj, u_kj)),
//! ψ(l, u) = σ φ((u - μ)/σ) + (μ - l) (1 - Φ((u - μ)/σ)),
//! ν(l, u) = (u - l) (1 - Φ((u - μ)/σ)),
//! ```
//!
//! with `σ = sqrt(max(σ², 1e-9))` and the upper bounds clamped at `1e10`.
//! BoTorch main sums the `2^m` products of either `ψ` differences or `ν`
//! over all outcomes; the product of the sums used here (from
//! meta-pytorch/botorch#3392) is the same value by the distributive law and
//! agrees to rounding.

use std::sync::Arc;

use errorfunctions::RealErrorFunctions;

use crate::acquisition::multi_objective::check_ref_point;
use crate::acquisition::multi_objective::hypervolume::{HypercellBounds, MAX_CELL_UPPER};
use crate::acquisition::{check_candidates, check_unbatched, AcquisitionFunction};
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::utils::multi_objective::box_decompositions::BoxDecomposition;
use crate::utils::probability::LOG_SQRT_2PI;
use crate::{Matrix, Vector};

/// BoTorch's lower bound of the posterior variance of EHVI.
const MIN_EHVI_VARIANCE: f64 = 1e-9;

/// `torch.distributions.Normal(0, 1).log_prob(u).exp()`.
#[inline]
fn pdf(u: f64) -> f64 {
    (-(u * u) / 2.0 - LOG_SQRT_2PI).exp()
}

/// `torch.distributions.Normal(0, 1).cdf(u)`.
#[inline]
fn cdf(u: f64) -> f64 {
    0.5 * (1.0 + RealErrorFunctions::erf(u / std::f64::consts::SQRT_2))
}

/// `ψ(l, b)` and its derivatives with respect to `μ` and `σ`.
#[inline]
fn psi(l: f64, b: f64, mu: f64, sigma: f64) -> (f64, f64, f64) {
    let w = (b - mu) / sigma;
    let p = pdf(w);
    let tail = 1.0 - cdf(w);
    let v = sigma * p + (mu - l) * tail;
    let dmu = w * p + tail + (mu - l) * p / sigma;
    let dsigma = p + w * w * p + (mu - l) * w * p / sigma;
    (v, dmu, dsigma)
}

/// `ν(l, u)` and its derivatives with respect to `μ` and `σ`.
#[inline]
fn nu(l: f64, u: f64, mu: f64, sigma: f64) -> (f64, f64, f64) {
    let w = (u - mu) / sigma;
    let p = pdf(w);
    let v = (u - l) * (1.0 - cdf(w));
    let dmu = (u - l) * p / sigma;
    let dsigma = (u - l) * w * p / sigma;
    (v, dmu, dsigma)
}

/// Analytic Expected Hypervolume Improvement of a single point (BoTorch's
/// `ExpectedHypervolumeImprovement`; `q = 1`, independent outcomes, no
/// outcome constraints).
#[derive(Debug)]
pub struct ExpectedHypervolumeImprovement {
    model: Arc<dyn Model>,
    ref_point: Vec<f64>,
    cells: HypercellBounds,
}

impl ExpectedHypervolumeImprovement {
    /// `ExpectedHypervolumeImprovement(model, ref_point, partitioning)` with
    /// an unbatched box decomposition of the Pareto front.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if the reference point does not
    /// have one value per outcome or no Pareto point is better than the
    /// reference point (BoTorch's messages), [`BotorchError::Unsupported`]
    /// for batched models or decompositions, and
    /// [`BotorchError::DimensionMismatch`] if the model does not have one
    /// output per outcome.
    pub fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        partitioning: &dyn BoxDecomposition,
    ) -> Result<Self> {
        let cells = crate::acquisition::multi_objective::monte_carlo::cells_of(partitioning)?;
        let pareto = partitioning
            .pareto_y()
            .map(|p| p.matrix(0))
            .unwrap_or_else(|| Matrix::zeros(0, cells.num_outcomes()));
        Self::from_cell_bounds(model, ref_point, cells, &pareto)
    }

    /// Like [`ExpectedHypervolumeImprovement::new`] with the hypercell
    /// bounds and the Pareto front (`n × m`) of the decomposition.
    ///
    /// # Errors
    /// See [`ExpectedHypervolumeImprovement::new`].
    pub fn from_cell_bounds(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        cells: HypercellBounds,
        pareto_y: &Matrix,
    ) -> Result<Self> {
        check_ref_point(ref_point, cells.num_outcomes(), false)?;
        if cells.batches() != 1 {
            return Err(BotorchError::Unsupported(
                "batched box decompositions are not supported".into(),
            ));
        }
        let better = (0..pareto_y.nrows())
            .any(|i| (0..pareto_y.ncols()).all(|j| pareto_y[(i, j)] > ref_point[j]));
        if !better && pareto_y.nrows() > 0 {
            return Err(BotorchError::InvalidArgument(
                "At least one pareto point must be better than the reference point.".into(),
            ));
        }
        check_unbatched(model.as_ref())?;
        if model.num_outputs() != cells.num_outcomes() {
            return Err(BotorchError::DimensionMismatch(format!(
                "The model has {} outputs but the box decomposition has {} outcomes.",
                model.num_outputs(),
                cells.num_outcomes()
            )));
        }
        Ok(Self {
            model,
            ref_point: ref_point.to_vec(),
            cells,
        })
    }

    /// The reference point.
    pub fn ref_point(&self) -> &[f64] {
        &self.ref_point
    }

    /// The hypercell bounds.
    pub fn cell_bounds(&self) -> &HypercellBounds {
        &self.cells
    }

    fn evaluate(&self, x: &Matrix, with_grad: bool) -> Result<(f64, Option<Matrix>)> {
        let d = self.model.dim();
        check_candidates(x, d, Some(1))?;
        let m = self.cells.num_outcomes();
        let nc = self.cells.num_cells();
        let point: Vec<f64> = x.row(0).iter().copied().collect();
        let post = if with_grad {
            self.model.point_posterior_with_grad(&point)?
        } else {
            let pp = self.model.point_posteriors(x)?;
            crate::posterior::JointPosteriorGrad {
                mean: Vector::from_fn(m, |j, _| pp.mean[(0, j)]),
                covariance: pp.covariance_of(0),
                dmean: Vec::new(),
                dcov: Vec::new(),
                num_outputs: m,
            }
        };
        let mu: Vec<f64> = (0..m).map(|j| post.mean[j]).collect();
        let var: Vec<f64> = (0..m).map(|j| post.covariance[(j, j)]).collect();
        // posterior.variance (clamped at 1e-10) .clamp_min(1e-9).sqrt().
        let sigma: Vec<f64> = var
            .iter()
            .map(|&v| {
                if v < MIN_EHVI_VARIANCE {
                    MIN_EHVI_VARIANCE
                } else {
                    v
                }
                .sqrt()
            })
            .collect();
        let (lower, upper) = (self.cells.lower(), self.cells.upper());
        let mut value = 0.0;
        let mut dmu = vec![0.0; m];
        let mut dsigma = vec![0.0; m];
        let mut f = vec![0.0; m];
        let mut fmu = vec![0.0; m];
        let mut fsig = vec![0.0; m];
        for k in 0..nc {
            for j in 0..m {
                let l = lower[k * m + j];
                let u = upper[k * m + j].min(MAX_CELL_UPPER);
                let (lu, lu_mu, lu_sig) = psi(l, u, mu[j], sigma[j]);
                let (ll, ll_mu, ll_sig) = psi(l, l, mu[j], sigma[j]);
                let (nv, nv_mu, nv_sig) = nu(l, u, mu[j], sigma[j]);
                f[j] = ll - lu + nv;
                fmu[j] = ll_mu - lu_mu + nv_mu;
                fsig[j] = ll_sig - lu_sig + nv_sig;
            }
            let prod = f.iter().fold(1.0, |a, v| a * v);
            value += prod;
            if with_grad {
                for j in 0..m {
                    let others = (0..m).filter(|&i| i != j).fold(1.0, |a, i| a * f[i]);
                    dmu[j] += others * fmu[j];
                    dsigma[j] += others * fsig[j];
                }
            }
        }
        if !with_grad {
            return Ok((value, None));
        }
        let mut grad = Matrix::zeros(1, d);
        for k in 0..d {
            let mut g = 0.0;
            for j in 0..m {
                g += dmu[j] * post.dmean[k][j];
                if var[j] > MIN_EHVI_VARIANCE {
                    g += dsigma[j] / (2.0 * sigma[j]) * post.dcov[k][(j, j)];
                }
            }
            grad[(0, k)] = g;
        }
        Ok((value, Some(grad)))
    }
}

impl AcquisitionFunction for ExpectedHypervolumeImprovement {
    fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    fn expected_q(&self) -> Option<usize> {
        Some(1)
    }

    fn forward(&self, x: &Matrix) -> Result<f64> {
        Ok(self.evaluate(x, false)?.0)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)> {
        let (v, g) = self.evaluate(x, true)?;
        Ok((v, g.expect("gradient requested")))
    }

    fn is_nonnegative(&self) -> bool {
        true
    }
}
