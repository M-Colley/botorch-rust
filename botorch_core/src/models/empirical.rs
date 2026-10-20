//! Exact empirical GPs with fixed interpolated means and native kernel trees.

use crate::error::{BotorchError, Result};
use crate::fit::{minimize_with_retries, FitOptions, FitResult, Fittable};
use crate::kernels::empirical::EmpiricalBasisSpec;
use crate::kernels::tree::KernelNode;
use crate::mll::ExactMarginalLogLikelihood;
use crate::models::Likelihood;
use crate::models::{GaussianLikelihoodSpec, Model, SingleTaskGP, SingleTaskGPOptions};
use crate::optim::fit::MllModel;
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::transforms::OutcomeTransformChoice;
use crate::{Matrix, Vector};

/// Exact independent or correlated empirical outputs, sharing curve weights.
#[derive(Clone, Debug)]
pub struct EmpiricalGP {
    gps: Vec<SingleTaskGP>,
    mean: EmpiricalBasisSpec,
    correlated: bool,
    raw: Vec<f64>,
    noise_count: usize,
    covar: KernelNode,
}
impl EmpiricalGP {
    /// Inputs include progression in column zero and integer output in column one.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: Vec<Matrix>,
        y: Vec<Matrix>,
        yvar: Option<Vec<Matrix>>,
        mean: EmpiricalBasisSpec,
        covar: KernelNode,
        noise: GaussianLikelihoodSpec,
        kernel_raw: Vec<f64>,
        noise_raw: Vec<f64>,
        correlated: bool,
    ) -> Result<Self> {
        mean.validate(2)?;
        let outputs = mean.tables.len();
        let count = if correlated { 1 } else { outputs };
        if x.len() != count
            || y.len() != count
            || yvar.as_ref().is_some_and(|v| v.len() != count)
            || noise_raw.len() != count
            || kernel_raw.len() != covar.num_raw()
            || mean.tables.iter().any(|m| m.nrows() != 1)
        {
            return Err(BotorchError::DimensionMismatch(
                "inconsistent empirical model data".into(),
            ));
        }
        let mut gps = Vec::with_capacity(count);
        for i in 0..count {
            if x[i].ncols() != 2 || x[i].nrows() == 0 || y[i].shape() != (x[i].nrows(), 1) {
                return Err(BotorchError::DimensionMismatch(
                    "empirical training inputs require n x 2 and targets n x 1".into(),
                ));
            }
            let mut target = y[i].clone();
            for j in 0..x[i].nrows() {
                let row = [x[i][(j, 0)], x[i][(j, 1)]];
                mean.validate_point(&row)?;
                if !correlated && row[1] != i as f64 {
                    return Err(BotorchError::InvalidArgument(
                        "independent empirical training output index differs".into(),
                    ));
                }
                target[(j, 0)] -= mean.basis(&row).1[0];
            }
            gps.push(SingleTaskGP::new(
                x[i].clone(),
                target,
                SingleTaskGPOptions {
                    train_yvar: yvar.as_ref().map(|v| v[i].clone()),
                    outcome_transform: OutcomeTransformChoice::None,
                    covar_tree: Some(covar.clone()),
                    likelihood: if yvar.is_none() { Some(noise) } else { None },
                    ..Default::default()
                },
            )?);
        }
        let noise_count = if yvar.is_none() { count } else { 0 };
        let mut raw = if noise_count == 0 { vec![] } else { noise_raw };
        raw.extend(kernel_raw);
        let mut model = Self {
            gps,
            mean,
            correlated,
            raw: vec![],
            noise_count,
            covar,
        };
        model.set_raw_parameters(&raw)?;
        Ok(model)
    }
    /// Current parameters in public registration order (noise, then shared kernel).
    pub fn raw_parameters(&self) -> &[f64] {
        &self.raw
    }
    fn local_raw(&self, i: usize, raw: &[f64]) -> Result<Vec<f64>> {
        if raw.len() != self.noise_count + self.covar.num_raw()
            || raw.iter().any(|v| !v.is_finite())
        {
            return Err(BotorchError::InvalidArgument(
                "invalid empirical raw parameter vector".into(),
            ));
        }
        let gp = &self.gps[i];
        let mut local = gp.raw_parameters().as_slice().to_vec();
        if let Some(j) = gp.parameter_layout().noise_index() {
            local[j] = raw[i];
        }
        local[gp.parameter_layout().constant_index()] = 0.0;
        local[gp.parameter_layout().kernel_range()].copy_from_slice(&raw[self.noise_count..]);
        Ok(local)
    }
    /// Replace parameters, keeping every independent output's shared kernel in sync.
    pub fn set_raw_parameters(&mut self, raw: &[f64]) -> Result<()> {
        let local = (0..self.gps.len())
            .map(|i| self.local_raw(i, raw))
            .collect::<Result<Vec<_>>>()?;
        for (gp, p) in self.gps.iter_mut().zip(local) {
            gp.set_raw_parameters(&p)?;
        }
        self.raw = raw.to_vec();
        Ok(())
    }
    /// Registered parameter names and element counts.
    pub fn parameter_names(&self) -> Vec<(String, usize)> {
        let mut names = vec![];
        if self.noise_count > 0 {
            names.push(("likelihood.noise_covar.raw_noise".into(), self.noise_count));
        }
        self.covar.visit_params("covar_module.", &mut |name, spec| {
            names.push((name, spec.numel()))
        });
        names
    }
    /// Raw box constraints, in public registration order.
    pub fn raw_parameter_bounds(&self) -> Vec<(f64, f64)> {
        let first = &self.gps[0];
        let bounds = first.raw_parameter_bounds();
        let mut out = vec![];
        if let Some(i) = first.parameter_layout().noise_index() {
            out.resize(self.noise_count, bounds[i]);
        }
        out.extend_from_slice(&bounds[first.parameter_layout().kernel_range()]);
        out
    }
    /// One homoskedastic noise per independent GP, or a shared correlated noise.
    pub fn noise(&self) -> Vec<f64> {
        self.gps
            .iter()
            .map(SingleTaskGP::observation_noise)
            .collect()
    }
    /// Full fixed-noise metadata, or the learned noise of each output.
    pub fn likelihood_noise(&self) -> Vec<f64> {
        self.gps
            .iter()
            .flat_map(|gp| gp.likelihood_noise().as_slice().to_vec())
            .collect()
    }
    /// Set constrained homoskedastic noise values, broadcasting a scalar.
    pub fn set_noise(&mut self, noise: &[f64]) -> Result<()> {
        if self.noise_count == 0
            || !(noise.len() == 1 || noise.len() == self.noise_count)
            || noise.iter().any(|v| !v.is_finite())
        {
            return Err(BotorchError::InvalidArgument(
                "noise requires finite scalar or one value per independent output".into(),
            ));
        }
        let mut raw = self.raw.clone();
        for (i, gp) in self.gps.iter().enumerate() {
            if let Likelihood::Gaussian(spec) = gp.likelihood() {
                let value = noise[if noise.len() == 1 { 0 } else { i }];
                if value < spec.noise_constraint.lower_bound()
                    || value > spec.noise_constraint.upper_bound()
                {
                    return Err(BotorchError::InvalidArgument(
                        "noise is outside its constraint".into(),
                    ));
                }
                raw[i] = spec.noise_constraint.inverse_transform(value);
            }
        }
        self.set_raw_parameters(&raw)
    }
    /// Sample registered kernel and noise priors without duplicating shared parameters.
    pub fn sample_all_priors(&self, rng: &mut rand::rngs::StdRng) -> Vec<f64> {
        let mut raw = self.raw.clone();
        if self.noise_count > 0 {
            for (i, gp) in self.gps.iter().enumerate() {
                if let Likelihood::Gaussian(spec) = gp.likelihood() {
                    if let Some(prior) = spec.noise_prior {
                        raw[i] = spec.noise_constraint.inverse_transform(prior.sample(rng));
                    }
                }
            }
        }
        crate::kernels::tree::sample_tree_priors(&self.covar, &mut raw[self.noise_count..], rng);
        raw
    }
    /// Independent-output likelihood batch shape.
    pub fn aug_batch_shape(&self) -> Vec<usize> {
        if !self.correlated && self.num_outputs() > 1 {
            vec![self.num_outputs()]
        } else {
            vec![]
        }
    }
    /// Normalized exact MLL per independent output, with summed shared-parameter gradient.
    pub fn mll_value_and_grad(&self, raw: &[f64], with_grad: bool) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut values = vec![];
        let mut grad = vec![0.0; raw.len()];
        for (i, gp) in self.gps.iter().enumerate() {
            let local = self.local_raw(i, raw)?;
            let mll = ExactMarginalLogLikelihood::new(gp);
            if with_grad {
                let (v, g) = mll.value_and_grad(&local)?;
                values.push(v);
                if let Some(j) = gp.parameter_layout().noise_index() {
                    grad[i] = g[j];
                }
                for (j, v) in g.as_slice()[gp.parameter_layout().kernel_range()]
                    .iter()
                    .enumerate()
                {
                    grad[self.noise_count + j] += v;
                }
            } else {
                values.push(mll.value_at(&local)?);
            }
        }
        Ok((values, grad))
    }
    fn query(&self, x: &Matrix, group: usize) -> Result<Matrix> {
        if x.ncols() != 1 || x.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::DimensionMismatch(
                "empirical queries require finite q x 1 progression inputs".into(),
            ));
        }
        let m = self.num_outputs();
        Ok(if self.correlated {
            Matrix::from_fn(x.nrows() * m, 2, |i, j| {
                if j == 0 {
                    x[(i / m, 0)]
                } else {
                    (i % m) as f64
                }
            })
        } else {
            Matrix::from_fn(
                x.nrows(),
                2,
                |i, j| if j == 0 { x[(i, 0)] } else { group as f64 },
            )
        })
    }
    fn mean_at(&self, group: usize, x: f64) -> Result<(f64, f64)> {
        let row = [x, group as f64];
        self.mean.validate_point(&row)?;
        let (_, v, d) = self.mean.basis(&row);
        Ok((v[0], d[0]))
    }
}
impl Model for EmpiricalGP {
    fn dim(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        self.mean.tables.len()
    }
    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let q = x.nrows();
        let m = self.num_outputs();
        let mut mean = Vector::zeros(q * m);
        let mut cov = Matrix::zeros(q * m, q * m);
        if self.correlated {
            let p = self.gps[0].posterior(&self.query(x, 0)?, observation_noise)?;
            for a in 0..q * m {
                let (g, i) = (a / q, a % q);
                let ai = i * m + g;
                mean[a] = p.mean[ai] + self.mean_at(g, x[(i, 0)])?.0;
                for b in 0..q * m {
                    cov[(a, b)] = p.covariance[(ai, (b % q) * m + b / q)];
                }
            }
        } else {
            for g in 0..m {
                let p = self.gps[g].posterior(&self.query(x, g)?, observation_noise)?;
                for i in 0..q {
                    mean[g * q + i] = p.mean[i] + self.mean_at(g, x[(i, 0)])?.0;
                }
                cov.view_mut((g * q, g * q), (q, q))
                    .copy_from(&p.covariance);
            }
        }
        Ok(Posterior::multi_output(mean, cov, m))
    }
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let p = self.posterior(x, false)?;
        let q = x.nrows();
        let m = self.num_outputs();
        let mut dm = vec![Vector::zeros(q * m); q];
        let mut dc = vec![Matrix::zeros(q * m, q * m); q];
        if self.correlated {
            let g = self.gps[0].joint_posterior_with_grad(&self.query(x, 0)?)?;
            for r in 0..q {
                for a in 0..q * m {
                    let ai = (a % q) * m + a / q;
                    dm[r][a] = (0..m).map(|s| g.dmean[(r * m + s) * 2][ai]).sum();
                    for b in 0..q * m {
                        let bi = (b % q) * m + b / q;
                        dc[r][(a, b)] = (0..m).map(|s| g.dcov[(r * m + s) * 2][(ai, bi)]).sum();
                    }
                }
            }
        } else {
            for s in 0..m {
                let g = self.gps[s].joint_posterior_with_grad(&self.query(x, s)?)?;
                for r in 0..q {
                    dm[r].rows_mut(s * q, q).copy_from(&g.dmean[r * 2]);
                    dc[r]
                        .view_mut((s * q, s * q), (q, q))
                        .copy_from(&g.dcov[r * 2]);
                }
            }
        }
        for r in 0..q {
            for s in 0..m {
                dm[r][s * q + r] += self.mean_at(s, x[(r, 0)])?.1;
            }
        }
        Ok(JointPosteriorGrad {
            mean: p.mean,
            covariance: p.covariance,
            dmean: dm,
            dcov: dc,
            num_outputs: m,
        })
    }
}

/// Native exact-MLL optimization, with curve weights shared between outputs.
#[derive(Clone, Debug)]
pub struct EmpiricalFit {
    /// Updated model after optimization.
    pub model: EmpiricalGP,
}
impl MllModel for EmpiricalFit {
    fn raw_parameter_vector(&self) -> Vec<f64> {
        self.model.raw.clone()
    }
    fn set_raw_parameter_vector(&mut self, raw: &[f64]) -> Result<()> {
        self.model.set_raw_parameters(raw)
    }
    fn raw_bounds(&self) -> Vec<(f64, f64)> {
        self.model.raw_parameter_bounds()
    }
    fn mll_values_and_gradient(&self, raw: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        self.model.mll_value_and_grad(raw, true)
    }
}
impl Fittable for EmpiricalFit {
    type Output = FitResult;
    fn fit_mll(&mut self, options: FitOptions) -> Result<FitResult> {
        let (result, raw) = minimize_with_retries(
            &self.model.raw,
            &self.model.raw_parameter_bounds(),
            options,
            |raw| {
                self.model
                    .mll_value_and_grad(raw, true)
                    .map(|(v, g)| (v.iter().sum(), g))
            },
            |rng| self.model.sample_all_priors(rng),
        )?;
        self.model.set_raw_parameters(&raw)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model(correlated: bool) -> EmpiricalGP {
        let knots = vec![vec![0.0, 0.3, 0.7, 1.0]; 2];
        let basis = EmpiricalBasisSpec {
            knots: knots.clone(),
            tables: (0..2)
                .map(|s| Matrix::from_fn(4, 4, |r, j| ((r * 2 + j + s) as f64).sin()))
                .collect(),
            weights: vec![1.0; 4],
            learn_weights: true,
            denominator: 4.0,
            group_column: Some(1),
            independent: !correlated,
        };
        let mean = EmpiricalBasisSpec {
            knots,
            tables: (0..2)
                .map(|s| Matrix::from_fn(1, 4, |_, j| s as f64 + j as f64 * 0.4))
                .collect(),
            weights: vec![1.0],
            learn_weights: false,
            denominator: 1.0,
            group_column: Some(1),
            independent: false,
        };
        let xs = [0.1, 0.4, 0.8];
        let x = if correlated {
            vec![Matrix::from_fn(6, 2, |i, j| {
                if j == 0 {
                    xs[i / 2]
                } else {
                    (i % 2) as f64
                }
            })]
        } else {
            (0..2)
                .map(|s| Matrix::from_fn(3, 2, |i, j| if j == 0 { xs[i] } else { s as f64 }))
                .collect()
        };
        let y = x
            .iter()
            .map(|a| Matrix::from_fn(a.nrows(), 1, |i, _| a[(i, 0)] * a[(i, 0)] + a[(i, 1)] * 0.3))
            .collect();
        EmpiricalGP::new(
            x,
            y,
            None,
            mean,
            KernelNode::empirical_basis(basis).unwrap(),
            crate::models::get_gaussian_likelihood_with_lognormal_prior(),
            vec![0.7, 0.9, 1.2, 1.4],
            vec![0.04; if correlated { 1 } else { 2 }],
            correlated,
        )
        .unwrap()
    }
    #[test]
    fn empirical_shared_parameter_and_query_derivatives() {
        for correlated in [false, true] {
            let model = model(correlated);
            let raw = model.raw_parameters().to_vec();
            let (_, grad) = model.mll_value_and_grad(&raw, true).unwrap();
            for i in 0..raw.len() {
                let mut plus = raw.clone();
                let mut minus = raw.clone();
                let h = 1e-6;
                plus[i] += h;
                minus[i] -= h;
                let f = |p: &[f64]| {
                    model
                        .mll_value_and_grad(p, false)
                        .unwrap()
                        .0
                        .iter()
                        .sum::<f64>()
                };
                assert!(
                    ((f(&plus) - f(&minus)) / (2.0 * h) - grad[i]).abs() < 2e-6,
                    "parameter {i}"
                );
            }
            let x = Matrix::from_column_slice(2, 1, &[0.22, 0.56]);
            let g = model.joint_posterior_with_grad(&x).unwrap();
            for i in 0..2 {
                let mut plus = x.clone();
                let mut minus = x.clone();
                let h = 1e-6;
                plus[(i, 0)] += h;
                minus[(i, 0)] -= h;
                let p = model.posterior(&plus, false).unwrap();
                let n = model.posterior(&minus, false).unwrap();
                assert!(((p.mean - n.mean) / (2.0 * h) - &g.dmean[i]).amax() < 2e-7);
                assert!(((p.covariance - n.covariance) / (2.0 * h) - &g.dcov[i]).amax() < 2e-7);
            }
            let mut changed = model.clone();
            changed.set_noise(&[0.15]).unwrap();
            assert_eq!(changed.noise(), vec![0.15; if correlated { 1 } else { 2 }]);
        }
    }
    #[test]
    fn empirical_invalid_data_and_raw_shapes_are_errors() {
        let mut m = model(false);
        assert!(m.set_raw_parameters(&[0.0]).is_err());
        assert!(m.set_noise(&[f64::NAN]).is_err());
        assert!(m
            .posterior(&Matrix::from_element(2, 2, 0.4), false)
            .is_err());
        assert!(m
            .posterior(&Matrix::from_element(2, 1, 1.1), false)
            .is_err());
        assert!(m
            .posterior(&Matrix::from_element(2, 1, f64::NAN), false)
            .is_err());
    }
}
