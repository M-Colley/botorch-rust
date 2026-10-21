//! Exact component inference conditioned on the full additive GP.

use std::sync::Arc;

use super::SingleTaskGP;
use crate::error::{BotorchError, Result};
use crate::kernels::{BuiltCovar, Prepared};
use crate::models::{BatchedMultiOutputGP, Model};
use crate::posterior::{JointPosteriorGrad, Posterior};
use crate::{Matrix, Vector};

/// A batch of the component marginal processes of an orthogonal additive GP.
/// The leading batch dimension enumerates bias, first order, interactions.
#[derive(Clone, Debug)]
pub struct AdditiveComponentModel {
    shape: Vec<usize>,
    elements: Vec<ComponentGp>,
}

#[derive(Clone, Debug)]
struct ComponentGp {
    model: Arc<SingleTaskGP>,
    component: usize,
}

impl AdditiveComponentModel {
    /// Capture all components of the current immutable fitted model.
    pub fn new(model: &BatchedMultiOutputGP, extra_batch_dims: usize) -> Result<Self> {
        if model.num_outputs() != 1 {
            return Err(BotorchError::Unsupported(
                "additive component inference requires a single output".into(),
            ));
        }
        let mut snapshots = Vec::new();
        let mut count = None;
        for gp in model.models() {
            if gp.has_transformed_posterior() {
                return Err(BotorchError::Unsupported(
                    "additive component inference requires an affine outcome transform".into(),
                ));
            }
            let BuiltCovar::Tree(tree) = &gp.cache()?.kernel else {
                return Err(BotorchError::Unsupported(
                    "an OrthogonalAdditiveKernel is required".into(),
                ));
            };
            let components = tree.additive_component_count()?;
            if count.is_some_and(|c| c != components) {
                return Err(BotorchError::DimensionMismatch(
                    "model batches have different additive components".into(),
                ));
            }
            count = Some(components);
            snapshots.push(Arc::new(gp.clone()));
        }
        let count = count.ok_or_else(|| BotorchError::InvalidArgument("empty GP batch".into()))?;
        let mut elements = Vec::with_capacity(count * snapshots.len());
        for component in 0..count {
            elements.extend(snapshots.iter().map(|model| ComponentGp {
                model: model.clone(),
                component,
            }));
        }
        let mut shape = vec![count];
        shape.resize(1 + extra_batch_dims, 1);
        shape.extend_from_slice(model.batch_shape());
        Ok(Self { shape, elements })
    }
}

impl Model for AdditiveComponentModel {
    fn dim(&self) -> usize {
        self.elements[0].dim()
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn batch_shape(&self) -> &[usize] {
        &self.shape
    }
    fn points_per_input(&self) -> usize {
        self.elements[0].points_per_input()
    }
    fn batch_element(&self, b: usize) -> Option<&dyn Model> {
        self.elements.get(b).map(|g| g as &dyn Model)
    }
    fn posterior(&self, _x: &Matrix, _noise: bool) -> Result<Posterior> {
        Err(BotorchError::Unsupported(
            "select an additive component batch element before inference".into(),
        ))
    }
    fn joint_posterior_with_grad(&self, _x: &Matrix) -> Result<JointPosteriorGrad> {
        Err(BotorchError::Unsupported(
            "select an additive component batch element before differentiation".into(),
        ))
    }
}

impl ComponentGp {
    fn inference(
        &self,
        x: &Matrix,
        observation_noise: bool,
        gradients: bool,
    ) -> Result<JointPosteriorGrad> {
        let gp = &self.model;
        let tape = gp.chain_tape(x)?;
        let xt = match &tape {
            Some((xt, _)) => xt.clone(),
            None => gp.transform_inputs(x)?,
        };
        let cache = gp.cache()?;
        let (BuiltCovar::Tree(tree), Prepared::Tree(train)) = (&cache.kernel, &cache.train_scaled)
        else {
            return Err(BotorchError::Unsupported(
                "an OrthogonalAdditiveKernel is required".into(),
            ));
        };
        let xs = tree.prepare(&xt)?;
        let kxs = tree.additive_component_covariance(&xs, train, self.component)?;
        let mut covariance = tree.additive_component_covariance(&xs, &xs, self.component)?;
        let prior = if self.component == 0 {
            gp.constant()
        } else {
            0.0
        };
        let mean = (&kxs * &cache.alpha).add_scalar(prior);
        let mut w = kxs.transpose();
        cache.chol.solve_lower_mut(&mut w);
        covariance.gemm_tr(-1.0, &w, &w, 1.0);
        if observation_noise {
            let diagonal: Vec<f64> = covariance.diagonal().iter().copied().collect();
            for (i, value) in gp
                .observation_noise_diagonal(&xt, &diagonal)
                .into_iter()
                .enumerate()
            {
                covariance[(i, i)] += value;
            }
        }
        let (aff, _) = gp.outcome_parts(&xt)?;
        let q = xt.nrows();
        let mut result = JointPosteriorGrad {
            mean: Vector::from_fn(q, |i, _| {
                aff.scale(i) * mean[i]
                    + if self.component == 0 {
                        aff.offset(i)
                    } else {
                        0.0
                    }
            }),
            covariance: Matrix::from_fn(q, q, |i, j| {
                aff.scale(i) * covariance[(i, j)] * aff.scale(j)
            }),
            dmean: Vec::new(),
            dcov: Vec::new(),
            num_outputs: 1,
        };
        if !gradients {
            return Ok(result);
        }
        cache.chol.solve_lower_transpose_mut(&mut w);
        let dt = xt.ncols();
        for i in 0..q {
            let (_, dk) = tree.additive_component_cross_grad(&xs, i, train, self.component)?;
            let (_, dkss) = tree.additive_component_cross_grad(&xs, i, &xs, self.component)?;
            let da = dk.tr_mul(&cache.alpha);
            let dw = dk.tr_mul(&w);
            for k in 0..dt {
                let mut dm = Vector::zeros(q);
                dm[i] = aff.scale(i) * da[k];
                let mut dc = Matrix::zeros(q, q);
                for j in 0..q {
                    let v = (dkss[(j, k)] - dw[(k, j)]) * aff.scale(i) * aff.scale(j);
                    dc[(i, j)] += v;
                    dc[(j, i)] += v;
                }
                result.dmean.push(dm);
                result.dcov.push(dc);
            }
        }
        let mut dmean = vec![Vector::zeros(q); x.nrows() * x.ncols()];
        let mut dcov = vec![Matrix::zeros(q, q); x.nrows() * x.ncols()];
        if let Some((_, tape)) = tape {
            let chain = gp.input_chain.as_ref().expect("chain present");
            let jacobians = tape.jacobians_through(chain)?;
            let points = tape.points_per_input();
            for i in 0..x.nrows() {
                for p in 0..points {
                    let row = i * points + p;
                    for k in 0..x.ncols() {
                        for t in 0..dt {
                            let factor = jacobians[row][(t, k)];
                            dmean[i * x.ncols() + k].axpy(factor, &result.dmean[row * dt + t], 1.0);
                            dcov[i * x.ncols() + k] += &result.dcov[row * dt + t] * factor;
                        }
                    }
                }
            }
        } else {
            let jacobian = gp.legacy_jacobian();
            for i in 0..x.nrows() {
                for k in 0..x.ncols() {
                    dmean[i * x.ncols() + k] = &result.dmean[i * dt + k] * jacobian[k];
                    dcov[i * x.ncols() + k] = &result.dcov[i * dt + k] * jacobian[k];
                }
            }
        }
        result.dmean = dmean;
        result.dcov = dcov;
        Ok(result)
    }
}

impl Model for ComponentGp {
    fn dim(&self) -> usize {
        self.model.dim()
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn points_per_input(&self) -> usize {
        self.model.points_per_input()
    }
    fn posterior(&self, x: &Matrix, observation_noise: bool) -> Result<Posterior> {
        let p = self.inference(x, observation_noise, false)?;
        Ok(Posterior::single_output(p.mean, p.covariance))
    }
    fn joint_posterior_with_grad(&self, x: &Matrix) -> Result<JointPosteriorGrad> {
        self.inference(x, false, true)
    }
}
