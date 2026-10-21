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
use std::sync::Arc;

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
    /// Condition on full-output observations at progression inputs, preserving
    /// the historical mean and shared covariance parameters.
    ///
    /// # Errors
    /// Shape and fixed-noise requirements follow the underlying exact GPs.
    pub fn condition_on_observations(
        &self,
        x: &Matrix,
        y: &Matrix,
        noise: Option<&Matrix>,
    ) -> Result<Self> {
        let m = self.num_outputs();
        if y.shape() != (x.nrows(), m) || noise.is_some_and(|v| v.shape() != y.shape()) {
            return Err(BotorchError::DimensionMismatch(
                "empirical conditioning targets and noise must be n x num_outputs".into(),
            ));
        }
        let mut out = self.clone();
        let mut gps = Vec::with_capacity(self.gps.len());
        for (group, gp) in self.gps.iter().enumerate() {
            let inputs = self.query(x, group)?;
            let count = if self.correlated { m } else { 1 };
            let mut target = Matrix::zeros(x.nrows() * count, 1);
            for i in 0..x.nrows() {
                for output in 0..count {
                    let task = if self.correlated { output } else { group };
                    target[(i * count + output, 0)] =
                        y[(i, task)] - self.mean_at(task, x[(i, 0)])?.0;
                }
            }
            let noise = noise.map(|values| {
                Matrix::from_fn(x.nrows() * count, 1, |i, _| {
                    values[(i / count, if self.correlated { i % count } else { group })]
                })
            });
            gps.push(gp.condition_on_observations(&inputs, &target, noise.as_ref())?);
        }
        out.gps = gps;
        Ok(out)
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

/// A correlated empirical GP viewed at selected tasks or at explicit long inputs.
#[derive(Clone, Debug)]
pub struct EmpiricalTaskView {
    model: Arc<EmpiricalGP>,
    outputs: Vec<usize>,
    task_feature: Option<usize>,
}
impl EmpiricalTaskView {
    /// Select historical tasks, or use a task column in every query row.
    pub fn new(
        model: Arc<EmpiricalGP>,
        outputs: Vec<usize>,
        task_feature: Option<usize>,
    ) -> Result<Self> {
        if !model.correlated
            || outputs.is_empty()
            || outputs.iter().any(|&s| s >= model.mean.tables.len())
            || task_feature.is_some_and(|j| j > 1)
        {
            return Err(BotorchError::InvalidArgument(
                "invalid correlated empirical task view".into(),
            ));
        }
        Ok(Self {
            model,
            outputs,
            task_feature,
        })
    }
    fn inputs(&self, x: &Matrix) -> Result<Matrix> {
        if x.ncols() != self.dim() || x.iter().any(|v| !v.is_finite()) {
            return Err(BotorchError::DimensionMismatch(
                "empirical task queries have invalid dimensions or nonfinite values".into(),
            ));
        }
        let mut out = if let Some(task) = self.task_feature {
            Matrix::from_fn(x.nrows(), 2, |i, j| {
                x[(i, if j == 0 { 1 - task } else { task })]
            })
        } else {
            let m = self.outputs.len();
            Matrix::from_fn(x.nrows() * m, 2, |i, j| {
                if j == 0 {
                    x[(i / m, 0)]
                } else {
                    self.outputs[i % m] as f64
                }
            })
        };
        for i in 0..out.nrows() {
            let task = out[(i, 1)].round();
            if (out[(i, 1)] - task).abs() > 1e-6 {
                return Err(BotorchError::InvalidArgument(
                    "task labels must be integer-valued".into(),
                ));
            }
            out[(i, 1)] = task;
            self.model
                .mean
                .validate_point(&[out[(i, 0)], out[(i, 1)]])?;
        }
        Ok(out)
    }
    fn index(&self, i: usize, q: usize) -> usize {
        if self.task_feature.is_some() {
            i
        } else {
            (i % q) * self.outputs.len() + i / q
        }
    }
}
impl Model for EmpiricalTaskView {
    fn dim(&self) -> usize {
        if self.task_feature.is_some() {
            2
        } else {
            1
        }
    }
    fn num_outputs(&self) -> usize {
        if self.task_feature.is_some() {
            1
        } else {
            self.outputs.len()
        }
    }
    fn posterior(&self, x: &Matrix, noise: bool) -> Result<Posterior> {
        let inputs = self.inputs(x)?;
        let q = x.nrows();
        let m = self.num_outputs();
        let gp = &self.model.gps[0];
        let fixed = matches!(gp.likelihood(), Likelihood::FixedNoise { .. });
        let mut p = gp.posterior(&inputs, noise && !fixed)?;
        if noise && fixed {
            let mut sums = vec![0.0; self.model.mean.tables.len()];
            let mut counts = vec![0usize; sums.len()];
            let variance = gp.likelihood_noise();
            for i in 0..gp.num_train() {
                let group = gp.train_inputs()[(i, 1)] as usize;
                sums[group] += variance[i];
                counts[group] += 1;
            }
            for i in 0..inputs.nrows() {
                let group = inputs[(i, 1)] as usize;
                p.covariance[(i, i)] += if counts[group] > 0 {
                    sums[group] / counts[group] as f64
                } else {
                    // An unseen historical task has no task-specific estimate.
                    // Match SingleTaskGP's observed-noise averaging convention.
                    gp.observation_noise()
                };
            }
        }
        let mean = Vector::from_fn(q * m, |i, _| {
            let j = self.index(i, q);
            p.mean[j] + self.model.mean.basis(&[inputs[(j, 0)], inputs[(j, 1)]]).1[0]
        });
        let cov = Matrix::from_fn(q * m, q * m, |i, j| {
            p.covariance[(self.index(i, q), self.index(j, q))]
        });
        Ok(Posterior::multi_output(mean, cov, m))
    }
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        let inputs = self.inputs(x)?;
        let q = x.nrows();
        let m = self.num_outputs();
        let d = self.dim();
        let source = self.model.gps[0].joint_posterior_with_grad(&inputs)?;
        let p = self.posterior(x, false)?;
        let mut dm = vec![Vector::zeros(q * m); q * d];
        let mut dc = vec![Matrix::zeros(q * m, q * m); q * d];
        let progression = self.task_feature.map_or(0, |j| 1 - j);
        for r in 0..q {
            let slot = r * d + progression;
            let source_rows: Vec<usize> = if self.task_feature.is_some() {
                vec![r]
            } else {
                (0..m).map(|s| r * m + s).collect()
            };
            for a in 0..q * m {
                let ai = self.index(a, q);
                dm[slot][a] = source_rows.iter().map(|&s| source.dmean[s * 2][ai]).sum();
                for b in 0..q * m {
                    let bi = self.index(b, q);
                    dc[slot][(a, b)] = source_rows
                        .iter()
                        .map(|&s| source.dcov[s * 2][(ai, bi)])
                        .sum();
                }
            }
            for s in 0..m {
                let row = if self.task_feature.is_some() {
                    r
                } else {
                    r * m + s
                };
                let slope = self
                    .model
                    .mean
                    .basis(&[inputs[(row, 0)], inputs[(row, 1)]])
                    .2[0];
                dm[slot][s * q + r] += slope;
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
    #[test]
    fn empirical_conditioning_matches_dense_gaussian_conditioning() {
        let x = Matrix::from_column_slice(2, 1, &[0.25, 0.65]);
        let query = Matrix::from_column_slice(2, 1, &[0.32, 0.76]);
        let observed = Matrix::from_row_slice(2, 2, &[0.9, 0.2, 0.4, 1.1]);
        let joint_x = Matrix::from_column_slice(4, 1, &[0.25, 0.65, 0.32, 0.76]);
        for correlated in [false, true] {
            let model = model(correlated);
            for explicit in [false, true] {
                let noise = Matrix::from_fn(2, 2, |i, j| {
                    if explicit {
                        0.02 + (i + j) as f64 * 0.01
                    } else {
                        0.04
                    }
                });
                let conditioned = model
                    .condition_on_observations(&x, &observed, explicit.then_some(&noise))
                    .unwrap();
                let post = model.posterior(&joint_x, false).unwrap();
                let obs_index = [0, 1, 4, 5];
                let query_index = [2, 3, 6, 7];
                let cross =
                    Matrix::from_fn(4, 4, |i, j| post.covariance[(query_index[i], obs_index[j])]);
                let covariance = Matrix::from_fn(4, 4, |i, j| {
                    post.covariance[(obs_index[i], obs_index[j])]
                        + if i == j { noise[(i % 2, i / 2)] } else { 0.0 }
                });
                let factor = crate::linalg::psd_safe_cholesky(&covariance).unwrap();
                let residual =
                    Vector::from_fn(4, |i, _| observed[(i % 2, i / 2)] - post.mean[obs_index[i]]);
                let mean = Vector::from_fn(4, |i, _| post.mean[query_index[i]])
                    + &cross * factor.solve_vec(&residual);
                let covariance = Matrix::from_fn(4, 4, |i, j| {
                    post.covariance[(query_index[i], query_index[j])]
                }) - &cross * factor.solve(&cross.transpose());
                let actual = conditioned.posterior(&query, false).unwrap();
                assert!((actual.mean - mean).amax() < 2e-12);
                assert!((actual.covariance - covariance).amax() < 2e-12);
                assert_eq!(conditioned.raw_parameters(), model.raw_parameters());
                assert_eq!(model.gps[0].num_train(), if correlated { 6 } else { 3 });
            }
        }
    }
    #[test]
    fn empirical_task_views_select_and_differentiate_long_inputs() {
        let model = Arc::new(model(true));
        let view = EmpiricalTaskView::new(model.clone(), vec![1], None).unwrap();
        let x = Matrix::from_column_slice(2, 1, &[0.22, 0.56]);
        let full = model.posterior(&x, false).unwrap();
        let one = view.posterior(&x, false).unwrap();
        assert!((full.mean.rows(2, 2) - one.mean).amax() < 1e-12);
        let long = EmpiricalTaskView::new(model, vec![0, 1], Some(0)).unwrap();
        let x = Matrix::from_row_slice(2, 2, &[0.0, 0.22, 1.0, 0.56]);
        let g = long.joint_posterior_with_grad(&x).unwrap();
        for i in 0..4 {
            let mut p = x.clone();
            let mut n = x.clone();
            let h = 1e-7;
            p[(i / 2, i % 2)] += h;
            n[(i / 2, i % 2)] -= h;
            let p = long.posterior(&p, false).unwrap();
            let n = long.posterior(&n, false).unwrap();
            assert!(((p.mean - n.mean) / (2.0 * h) - &g.dmean[i]).amax() < 2e-7);
            assert!(((p.covariance - n.covariance) / (2.0 * h) - &g.dcov[i]).amax() < 2e-7);
        }
        assert!(long
            .posterior(&Matrix::from_row_slice(1, 2, &[0.4, 0.3]), false)
            .is_err());
    }
}
