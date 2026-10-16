//! Posterior paths by Matheron's rule
//! (`botorch.sampling.pathwise.posterior_samplers`): [`draw_matheron_paths`]
//! and the deterministic model of one posterior path,
//! [`MatheronPathModel`] (BoTorch's `botorch.models.deterministic.MatheronPathModel`).

use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{BotorchError, Result};
use crate::models::{Model, SingleTaskGP};
use crate::sampling::pathwise::features::DEFAULT_NUM_FEATURES;
use crate::sampling::pathwise::paths::{
    check_inputs, GeneralizedLinearPath, PathGradients, PathList, PathValues, SamplePath,
};
use crate::sampling::pathwise::prior_samplers::{fourier_kernel, kernel_feature_path, model_gps};
use crate::sampling::pathwise::update_strategies::gaussian_update;
use crate::sampling::pathwise::utils::{derive_seed, MatheronPathDraws, OutcomeUntransform};
use crate::utils::sampling::random_seed;
use crate::Matrix;

/// Options of [`draw_matheron_paths`] and [`get_matheron_path_model`].
#[derive(Clone, Debug, PartialEq)]
pub struct MatheronPathOptions {
    /// The number of Fourier features of the prior paths (BoTorch's default
    /// 1024; must be even).
    pub num_features: usize,
    /// The seed of botorch_core's generators; `None` draws a random seed
    /// (BoTorch draws from torch's global generator). The draws of output
    /// `t` come from the stream `t` of the seed.
    pub seed: Option<u64>,
    /// Explicit random numbers, one entry per output (overriding `seed` and
    /// `num_features`), e.g. BoTorch's draws.
    pub draws: Option<Vec<MatheronPathDraws>>,
}

impl Default for MatheronPathOptions {
    fn default() -> Self {
        Self {
            num_features: DEFAULT_NUM_FEATURES,
            seed: None,
            draws: None,
        }
    }
}

/// The random numbers of the paths of every output of `model`: the
/// explicit draws of `options` (checked) or draws from `options.seed`
/// (output `t` from stream `t`; a random seed if `None`).
///
/// # Errors
/// [`BotorchError::Unsupported`] for batched models or models without
/// independent exact GP outputs, [`BotorchError::DimensionMismatch`] for
/// explicit draws of the wrong shapes or number, and errors of
/// [`MatheronPathDraws::sample`].
pub fn matheron_path_draws(
    model: &dyn Model,
    num_samples: usize,
    options: &MatheronPathOptions,
) -> Result<Vec<MatheronPathDraws>> {
    output_draws(&model_gps(model)?, num_samples, options)
}

/// The draws of every output GP: explicit ones (checked) or drawn from the
/// seed.
pub(crate) fn output_draws(
    gps: &[&SingleTaskGP],
    num_samples: usize,
    options: &MatheronPathOptions,
) -> Result<Vec<MatheronPathDraws>> {
    match &options.draws {
        Some(draws) => {
            if draws.len() != gps.len() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected Matheron path draws for {} outputs, got {}",
                    gps.len(),
                    draws.len()
                )));
            }
            for (gp, d) in gps.iter().zip(draws) {
                d.check(fourier_kernel(gp)?.feature_dim(), gp.num_train())?;
                if d.num_samples() != num_samples {
                    return Err(BotorchError::DimensionMismatch(format!(
                        "expected draws of {num_samples} samples, got {}",
                        d.num_samples()
                    )));
                }
            }
            Ok(draws.clone())
        }
        None => {
            let seed = options.seed.unwrap_or_else(random_seed);
            gps.iter()
                .enumerate()
                .map(|(t, gp)| {
                    let fk = fourier_kernel(gp)?;
                    MatheronPathDraws::sample(
                        fk.kernel.kind(),
                        fk.feature_dim(),
                        gp.num_train(),
                        num_samples,
                        options.num_features,
                        derive_seed(seed, t as u64),
                    )
                })
                .collect()
        }
    }
}

/// Posterior sample paths of a single-output exact GP by Matheron's rule
/// (BoTorch's `MatheronPath`, a `PathDict` of `"prior_paths"` and
/// `"update_paths"` joined by `sum`, with the model's outcome untransform):
///
/// ```text
/// path_s(x) = untransform(prior_s(z) + update_s(z)),   z = input_transform(x).
/// ```
#[derive(Clone, Debug)]
pub struct MatheronPath {
    prior: GeneralizedLinearPath,
    update: GeneralizedLinearPath,
    output_transform: Option<OutcomeUntransform>,
    draws: MatheronPathDraws,
}

impl MatheronPath {
    /// The paths of `gp` for the given random numbers (BoTorch's
    /// `_draw_matheron_paths_ExactGP`): prior paths on the random Fourier
    /// features of the kernel ([`kernel_feature_path`], without the outcome
    /// transform), evaluated at the transformed training inputs, and the
    /// [`gaussian_update`] of these values.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for draws of the wrong shape and
    /// errors of the model's factorization.
    pub fn new(gp: &SingleTaskGP, draws: MatheronPathDraws) -> Result<Self> {
        draws.check(fourier_kernel(gp)?.feature_dim(), gp.num_train())?;
        let prior =
            kernel_feature_path(gp, draws.frequencies.clone(), draws.weights.clone(), false)?;
        let sample_values = prior.forward_transformed(gp.transformed_train_inputs());
        let update = gaussian_update(gp, &sample_values, &draws.noise)?;
        Ok(Self {
            prior,
            update,
            output_transform: OutcomeUntransform::of(gp),
            draws,
        })
    }

    /// The prior paths (`paths["prior_paths"]`; transformed outcome space).
    pub fn prior_paths(&self) -> &GeneralizedLinearPath {
        &self.prior
    }

    /// The update paths (`paths["update_paths"]`; transformed outcome space).
    pub fn update_paths(&self) -> &GeneralizedLinearPath {
        &self.update
    }

    /// The outcome untransform, if the model has an outcome transform.
    pub fn output_transform(&self) -> Option<&OutcomeUntransform> {
        self.output_transform.as_ref()
    }

    /// The random numbers of the paths.
    pub fn draws(&self) -> &MatheronPathDraws {
        &self.draws
    }

    fn untransform(&self, v: f64) -> f64 {
        match &self.output_transform {
            Some(o) => o.apply(v),
            None => v,
        }
    }
}

impl SamplePath for MatheronPath {
    fn dim(&self) -> usize {
        self.prior.dim()
    }

    fn num_samples(&self) -> usize {
        self.prior.num_samples()
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn forward(&self, x: &Matrix) -> Result<PathValues> {
        let z = self.prior.transform(x)?;
        let mut values = self.prior.forward_transformed(&z);
        values += self.update.forward_transformed(&z);
        values.apply(|v| *v = self.untransform(*v));
        Ok(vec![values])
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(PathValues, PathGradients)> {
        let z = self.prior.transform(x)?;
        let (n, d) = z.shape();
        let s = self.num_samples();
        let jac = self.prior.input_jacobian();
        let scale = self.output_transform.map_or(1.0, |o| o.std);
        let mut values = Matrix::zeros(s, n);
        let mut grads = vec![Matrix::zeros(n, d); s];
        for i in 0..n {
            let zi: Vec<f64> = z.row(i).iter().copied().collect();
            let (vp, gp) = self.prior.forward_transformed_with_grad(&zi);
            let (vu, gu) = self.update.forward_transformed_with_grad(&zi);
            for (k, g) in grads.iter_mut().enumerate() {
                values[(k, i)] = self.untransform(vp[k] + vu[k]);
                for j in 0..d {
                    g[(i, j)] = scale * (gp[(k, j)] + gu[(k, j)]) * jac[j];
                }
            }
        }
        Ok((vec![values], vec![grads]))
    }
}

/// Draws `num_samples` functions from the posterior of every output of
/// `model` by Matheron's rule (BoTorch's `draw_matheron_paths(model,
/// sample_shape=[num_samples])`): a [`PathList`] with one [`MatheronPath`]
/// per output (the outputs of a multi-output model and the sub-models of a
/// `ModelListGP` in output order). The outputs are drawn in parallel.
///
/// # Errors
/// [`BotorchError::Unsupported`] for batched models or models without
/// independent exact GP outputs, [`BotorchError::InvalidArgument`] for
/// `num_samples = 0`, and errors of the draws and of the models'
/// factorizations.
pub fn draw_matheron_paths(
    model: &dyn Model,
    num_samples: usize,
    options: &MatheronPathOptions,
) -> Result<PathList> {
    let gps = model_gps(model)?;
    let draws = output_draws(&gps, num_samples, options)?;
    let paths = gps
        .par_iter()
        .zip(draws.into_par_iter())
        .map(|(gp, d)| MatheronPath::new(gp, d).map(|p| Arc::new(p) as Arc<dyn SamplePath>))
        .collect::<Result<Vec<_>>>()?;
    PathList::new(paths)
}

/// A deterministic model whose outputs are Matheron paths of a model's
/// posterior (BoTorch's `MatheronPathModel` from `get_matheron_path_model`):
/// with one sample (BoTorch's default `sample_shape=None`), its "posterior
/// mean" at `x` is the `n × m` matrix of the path values of all outputs.
#[derive(Clone, Debug)]
pub struct MatheronPathModel {
    paths: PathList,
}

impl MatheronPathModel {
    /// The model of `num_samples` paths of `model` (see
    /// [`draw_matheron_paths`]).
    ///
    /// # Errors
    /// See [`draw_matheron_paths`].
    pub fn new(
        model: &dyn Model,
        num_samples: usize,
        options: &MatheronPathOptions,
    ) -> Result<Self> {
        Ok(Self {
            paths: draw_matheron_paths(model, num_samples, options)?,
        })
    }

    /// The model of existing paths.
    pub fn from_paths(paths: PathList) -> Self {
        Self { paths }
    }

    /// The paths (one per output).
    pub fn paths(&self) -> &PathList {
        &self.paths
    }

    /// The input dimension.
    pub fn dim(&self) -> usize {
        self.paths.dim()
    }

    /// The number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.paths.num_outputs()
    }

    /// The number of sample paths per output.
    pub fn num_samples(&self) -> usize {
        self.paths.num_samples()
    }

    /// The values of every sample at the rows of `x`: one `n × m` matrix per
    /// sample (BoTorch's `sample_shape x n x m` output).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a column mismatch.
    pub fn forward_samples(&self, x: &Matrix) -> Result<Vec<Matrix>> {
        let values = self.paths.forward(x)?;
        let (n, m) = (x.nrows(), values.len());
        Ok((0..self.num_samples())
            .map(|s| Matrix::from_fn(n, m, |i, t| values[t][(s, i)]))
            .collect())
    }

    /// The values (`n × m`) of a model of one sample path at the rows of `x`
    /// (BoTorch's `MatheronPathModel(model).posterior(X).mean`).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for a model of several sample paths
    /// and [`BotorchError::DimensionMismatch`] for a column mismatch.
    pub fn forward(&self, x: &Matrix) -> Result<Matrix> {
        if self.num_samples() != 1 {
            return Err(BotorchError::InvalidArgument(format!(
                "forward requires a model of one sample path, this one has {}; use \
                 forward_samples",
                self.num_samples()
            )));
        }
        check_inputs(x, self.dim())?;
        Ok(self.forward_samples(x)?.swap_remove(0))
    }

    /// The values (`n × m`) and their derivatives (one `n × d` matrix per
    /// output) of a model of one sample path.
    ///
    /// # Errors
    /// See [`MatheronPathModel::forward`].
    pub fn forward_with_grad(&self, x: &Matrix) -> Result<(Matrix, Vec<Matrix>)> {
        if self.num_samples() != 1 {
            return Err(BotorchError::InvalidArgument(format!(
                "forward_with_grad requires a model of one sample path, this one has {}",
                self.num_samples()
            )));
        }
        let (values, grads) = self.paths.forward_with_grad(x)?;
        let n = x.nrows();
        let v = Matrix::from_fn(n, values.len(), |i, t| values[t][(0, i)]);
        Ok((v, grads.into_iter().map(|mut g| g.swap_remove(0)).collect()))
    }
}

/// A deterministic model of one Matheron path of `model`'s posterior
/// (BoTorch's `get_matheron_path_model(model)`, `sample_shape=None`); with
/// `num_samples > 1`, a model of that many paths (BoTorch's `sample_shape`).
///
/// # Errors
/// See [`draw_matheron_paths`].
pub fn get_matheron_path_model(
    model: &dyn Model,
    num_samples: usize,
    options: &MatheronPathOptions,
) -> Result<MatheronPathModel> {
    MatheronPathModel::new(model, num_samples, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::KernelKind;
    use crate::models::{BatchedMultiOutputGP, ModelListGP, SingleTaskGPOptions};
    use crate::transforms::Normalize;
    use crate::{Matrix, Vector};

    fn gp(n: usize, kind: KernelKind, normalize: bool, yvar: bool) -> SingleTaskGP {
        let x = Matrix::from_fn(n, 2, |i, j| {
            (((i * 7 + j * 3) % n) as f64 / n as f64) * 3.0 - 1.0 + 0.05 * j as f64
        });
        let y = Matrix::from_fn(n, 1, |i, _| (2.0 * x[(i, 0)]).sin() + 0.3 * x[(i, 1)]);
        let mut gp = SingleTaskGP::new(
            x,
            y,
            SingleTaskGPOptions {
                kernel: kind,
                input_transform: normalize.then(|| Normalize::new(2)),
                train_yvar: yvar.then(|| Matrix::from_element(n, 1, 0.01)),
                ..Default::default()
            },
        )
        .unwrap();
        let mut h = gp.hyperparameters().clone();
        h.raw_lengthscale = Vector::from_vec(vec![0.3, 0.5]);
        h.raw_constant = 0.2;
        if let Some(noise) = h.raw_noise.as_mut() {
            *noise = 0.02;
        }
        gp.set_hyperparameters(h).unwrap();
        gp
    }

    /// Path samples have the posterior's mean and covariance (statistically).
    #[test]
    fn matheron_paths_sample_the_posterior() {
        for (kind, normalize, yvar) in [
            (KernelKind::Rbf, true, false),
            (KernelKind::Matern52, false, true),
        ] {
            let g = gp(9, kind, normalize, yvar);
            let s = 2048;
            let paths = draw_matheron_paths(
                &g,
                s,
                &MatheronPathOptions {
                    num_features: 2048,
                    seed: Some(1),
                    draws: None,
                },
            )
            .unwrap();
            let x = Matrix::from_row_slice(3, 2, &[0.1, 0.2, 1.3, -0.4, 0.95, 1.7]);
            let v = &paths.forward(&x).unwrap()[0];
            let post = g.posterior(&x, false).unwrap();
            let mean = v.row_mean().transpose();
            let centered = Matrix::from_fn(s, 3, |k, i| v[(k, i)] - mean[i]);
            let cov = centered.transpose() * &centered / (s as f64 - 1.0);
            let scale = post.covariance.diagonal().max().sqrt();
            let mean_err = (&mean - &post.mean).amax() / scale;
            let cov_err = (&cov - &post.covariance).amax() / (scale * scale);
            println!("{kind:?}: mean error {mean_err}, covariance error {cov_err}");
            assert!(mean_err < 0.06, "{kind:?}: mean error {mean_err}");
            assert!(cov_err < 0.08, "{kind:?}: covariance error {cov_err}");
        }
    }

    #[test]
    fn path_models_and_gradients() {
        let a = gp(8, KernelKind::Rbf, true, false);
        let b = gp(6, KernelKind::Matern52, false, false);
        let list = ModelListGP::new(vec![
            BatchedMultiOutputGP::from(a.clone()),
            BatchedMultiOutputGP::from(b.clone()),
        ])
        .unwrap();
        let options = MatheronPathOptions {
            num_features: 64,
            seed: Some(7),
            draws: None,
        };
        let pm = get_matheron_path_model(&list, 1, &options).unwrap();
        assert_eq!((pm.num_outputs(), pm.num_samples(), pm.dim()), (2, 1, 2));
        let x = Matrix::from_row_slice(2, 2, &[0.3, 0.1, -0.5, 1.2]);
        let y = pm.forward(&x).unwrap();
        // The paths of the list are those of the sub-models with the seeds of
        // their output index.
        let pa = draw_matheron_paths(
            &a,
            1,
            &MatheronPathOptions {
                seed: Some(7),
                ..options.clone()
            },
        )
        .unwrap();
        assert_eq!(y.column(0).transpose(), pa.forward(&x).unwrap()[0].row(0));
        let (yg, g) = pm.forward_with_grad(&x).unwrap();
        assert!((&yg - &y).amax() < 1e-12);
        for (t, gt) in g.iter().enumerate() {
            for i in 0..2 {
                for k in 0..2 {
                    let h = 1e-6;
                    let mut xp = x.clone();
                    xp[(i, k)] += h;
                    let mut xm = x.clone();
                    xm[(i, k)] -= h;
                    let fd = (pm.forward(&xp).unwrap()[(i, t)] - pm.forward(&xm).unwrap()[(i, t)])
                        / (2.0 * h);
                    assert!(
                        (fd - gt[(i, k)]).abs() < 1e-6 * fd.abs().max(1.0),
                        "{fd} {}",
                        gt[(i, k)]
                    );
                }
            }
        }
        // Explicit draws reproduce the paths.
        let draws: Vec<MatheronPathDraws> = pm
            .paths()
            .paths()
            .iter()
            .enumerate()
            .map(|(t, _)| {
                let gp = if t == 0 { &a } else { &b };
                MatheronPathDraws::sample(
                    gp.covar_module().as_standard().unwrap().kind,
                    2,
                    gp.num_train(),
                    1,
                    64,
                    derive_seed(7, t as u64),
                )
                .unwrap()
            })
            .collect();
        let explicit = get_matheron_path_model(
            &list,
            1,
            &MatheronPathOptions {
                draws: Some(draws),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(explicit.forward(&x).unwrap(), y);
        assert!(pm.forward(&Matrix::zeros(1, 3)).is_err());
        let many = get_matheron_path_model(&a, 3, &options).unwrap();
        assert!(many.forward(&x).is_err());
        assert_eq!(many.forward_samples(&x).unwrap().len(), 3);
    }

    /// The prior paths and the Gaussian updates with the noise of the same
    /// seed are the prior and update paths of the Matheron paths.
    #[test]
    fn matheron_paths_are_prior_plus_update() {
        use crate::sampling::pathwise::{
            draw_kernel_feature_paths, draw_update_noise, gaussian_update,
        };
        let g = gp(7, KernelKind::Matern52, true, false);
        let options = MatheronPathOptions {
            num_features: 32,
            seed: Some(3),
            draws: None,
        };
        let paths = draw_matheron_paths(&g, 4, &options).unwrap();
        let prior = draw_kernel_feature_paths(&g, 4, &options).unwrap();
        let noise = draw_update_noise(&g, 4, 3).unwrap();
        assert_eq!(
            noise[0],
            matheron_path_draws(&g, 4, &options).unwrap()[0].noise
        );
        let draws = matheron_path_draws(&g, 4, &options).unwrap();
        let prior_t = kernel_feature_path(
            &g,
            draws[0].frequencies.clone(),
            draws[0].weights.clone(),
            false,
        )
        .unwrap();
        let train = g.transformed_train_inputs().clone();
        let values = prior_t.forward_transformed(&train);
        let update = gaussian_update(&g, &values, &noise[0]).unwrap();
        let x = Matrix::from_row_slice(2, 2, &[0.3, 0.1, -0.5, 1.2]);
        let untransform = OutcomeUntransform::of(&g).expect("standardized outcomes");
        let joined = &prior_t.forward(&x).unwrap()[0] + &update.forward(&x).unwrap()[0];
        let expected = joined.map(|v| untransform.apply(v));
        assert!((&paths.forward(&x).unwrap()[0] - &expected).amax() < 1e-12);
        // `draw_kernel_feature_paths` has the outcome transform.
        let p = prior.forward(&x).unwrap();
        let pt = prior_t.forward(&x).unwrap()[0].map(|v| untransform.apply(v));
        assert!((&p[0] - &pt).amax() < 1e-12);
    }
}
