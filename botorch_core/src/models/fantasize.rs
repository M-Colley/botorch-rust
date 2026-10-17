//! Conditioning on observations and fantasy models of multi-output, batched
//! and list models (BoTorch's `condition_on_observations` and `fantasize` of
//! `BatchedMultiOutputGPyTorchModel` and `ModelListGP`).
//!
//! Batched arrays (BoTorch's `batch_shape × n × k` tensors) are passed as
//! [`BatchedMatrix`]es. Batch shapes broadcast like tensors: the conditioned
//! model of a [`BatchedMultiOutputGP`] has the broadcast of the model's batch
//! shape and the batch shapes of the new inputs, observations and noise; its
//! element `r` and output `t` is the element of the original model at the
//! broadcast batch index conditioned with [`SingleTaskGP::condition_on_observations`].
//! Elements conditioned at the same inputs (e.g. the fantasies of
//! [`BatchedMultiOutputGP::fantasize`], which have the fantasy dimension in
//! front of the batch shape) share the update of the Cholesky factor.
//!
//! See [`crate::models::gp`] for the semantics of a conditioned model (eval-mode
//! transforms, noise of the new points).

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::models::batched::block_diagonal;
use crate::models::gp::ObservationNoise;
use crate::models::{BatchedInputTransform, BatchedMultiOutputGP, ModelListGP, SingleTaskGP};
use crate::sampling::SobolQMCNormalSampler;
use crate::transforms::input::TransformMode;
use crate::Matrix;

/// A boolean matrix, e.g. BoTorch's `evaluation_mask` (`n' × m`: which
/// outputs are observed at which new point).
pub type BoolMatrix = nalgebra::DMatrix<bool>;

/// A batch of matrices, BoTorch's `batch_shape × r × c` tensor: one `r × c`
/// matrix per batch element, in row-major order of `batch_shape`.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchedMatrix {
    batch_shape: Vec<usize>,
    matrices: Vec<Matrix>,
}

impl BatchedMatrix {
    /// A batch of matrices of equal shape.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the number of matrices is not
    /// the product of `batch_shape` or the shapes differ.
    pub fn new(batch_shape: Vec<usize>, matrices: Vec<Matrix>) -> Result<Self> {
        let n: usize = batch_shape.iter().product();
        if matrices.len() != n || n == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "a batch of shape {batch_shape:?} needs {n} matrices, got {}",
                matrices.len()
            )));
        }
        let shape = matrices[0].shape();
        if matrices.iter().any(|m| m.shape() != shape) {
            return Err(BotorchError::DimensionMismatch(
                "all matrices of a batch must have the same shape".into(),
            ));
        }
        Ok(Self {
            batch_shape,
            matrices,
        })
    }

    /// A single (unbatched) matrix.
    pub fn single(matrix: Matrix) -> Self {
        Self {
            batch_shape: Vec::new(),
            matrices: vec![matrix],
        }
    }

    /// The batch shape.
    pub fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    /// The matrices in row-major order of the batch shape.
    pub fn matrices(&self) -> &[Matrix] {
        &self.matrices
    }

    /// Number of rows of every matrix.
    pub fn nrows(&self) -> usize {
        self.matrices[0].nrows()
    }

    /// Number of columns of every matrix.
    pub fn ncols(&self) -> usize {
        self.matrices[0].ncols()
    }
}

/// The broadcast of batch shapes (`None` if they are incompatible).
pub(crate) fn broadcast_shapes(shapes: &[&[usize]]) -> Option<Vec<usize>> {
    let len = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = vec![1; len];
    for s in shapes {
        let off = len - s.len();
        for (i, &v) in s.iter().enumerate() {
            let o = &mut out[off + i];
            if *o == 1 {
                *o = v;
            } else if v != 1 && v != *o {
                return None;
            }
        }
    }
    Some(out)
}

/// For every element of the batch shape `out`, the index of the element of
/// `shape` (broadcast to `out`) it reads.
pub(crate) fn broadcast_indices(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let total: usize = out.iter().product();
    let off = out.len() - shape.len();
    (0..total)
        .map(|flat| {
            let mut rem = flat;
            let mut idx = 0;
            let mut stride = 1;
            for k in (0..out.len()).rev() {
                let coord = rem % out[k];
                rem /= out[k];
                if k >= off {
                    let s = shape[k - off];
                    if s != 1 {
                        idx += coord * stride;
                    }
                    stride *= s;
                }
            }
            idx
        })
        .collect()
}

/// Where the noise of new observations is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NoiseSpace {
    /// The outcome-transformed space (BoTorch's `SingleTaskGP`).
    Transformed,
    /// The original space (BoTorch's `ModelListGP`, corrected in
    /// meta-pytorch/botorch#3399): divided by the squared standard deviation
    /// of each element's `Standardize`.
    Original,
}

/// Column `t` of `m`, with its rows broadcast to `rows` (from 1).
fn column_rows(m: &Matrix, t: usize, rows: usize) -> Matrix {
    if m.nrows() == rows {
        m.columns(t, 1).into_owned()
    } else {
        Matrix::from_element(rows, 1, m[(0, t)])
    }
}

impl BatchedMultiOutputGP {
    fn conditioned_transform(&self, model_index: &[usize]) -> Option<BatchedInputTransform> {
        match self.input_transform()? {
            BatchedInputTransform::Shared(tf) => Some(BatchedInputTransform::Shared(tf.clone())),
            BatchedInputTransform::PerBatch(tfs) => Some(BatchedInputTransform::PerBatch(
                model_index.iter().map(|&b| tfs[b].clone()).collect(),
            )),
        }
    }

    /// BoTorch's `condition_on_observations(X, Y, noise=noise)`: new inputs
    /// `x` (`batch_x × n' × d`, raw space), observations `y` (`batch_y × n'
    /// × m`, original space) and optional noise variances (`batch × n' × m`,
    /// *transformed* space, as BoTorch assumes for `SingleTaskGP`). The
    /// conditioned model's batch shape is the broadcast of the model's batch
    /// shape and the batch shapes of `x`, `y` and `noise`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for incompatible shapes and the
    /// errors of [`SingleTaskGP::condition_on_observations`].
    pub fn condition_on_observations(
        &self,
        x: &BatchedMatrix,
        y: &BatchedMatrix,
        noise: Option<&BatchedMatrix>,
    ) -> Result<BatchedMultiOutputGP> {
        self.condition_impl(x, y, noise, NoiseSpace::Transformed, TransformMode::Eval)
    }

    fn condition_impl(
        &self,
        x: &BatchedMatrix,
        y: &BatchedMatrix,
        noise: Option<&BatchedMatrix>,
        space: NoiseSpace,
        mode: TransformMode,
    ) -> Result<BatchedMultiOutputGP> {
        let (m, d) = (self.num_outputs(), self.dim());
        let k = x.nrows();
        if x.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X with {d} columns (d), got {}",
                x.ncols()
            )));
        }
        if y.nrows() != k || y.ncols() != m {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected Y with {k} rows (the new points) and {m} columns (outputs), got {}x{}",
                y.nrows(),
                y.ncols()
            )));
        }
        if let Some(v) = noise {
            if (v.nrows() != k && v.nrows() != 1) || v.ncols() != m {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected noise with {k} (or 1) rows and {m} columns, got {}x{}",
                    v.nrows(),
                    v.ncols()
                )));
            }
        }
        let empty: &[usize] = &[];
        let shapes: Vec<&[usize]> = vec![
            self.batch_shape(),
            x.batch_shape(),
            y.batch_shape(),
            noise.map_or(empty, |v| v.batch_shape()),
        ];
        let out = broadcast_shapes(&shapes).ok_or_else(|| {
            BotorchError::DimensionMismatch(format!(
                "The batch shapes of the model ({:?}), X ({:?}), Y ({:?}) and the noise are \
                 not broadcastable.",
                self.batch_shape(),
                x.batch_shape(),
                y.batch_shape()
            ))
        })?;
        let bm = broadcast_indices(self.batch_shape(), &out);
        let bx = broadcast_indices(x.batch_shape(), &out);
        let by = broadcast_indices(y.batch_shape(), &out);
        let bn = noise.map(|v| broadcast_indices(v.batch_shape(), &out));
        let nout = bm.len();
        // Group the result elements conditioned at the same inputs.
        let mut groups: BTreeMap<(usize, usize, usize), Vec<usize>> = BTreeMap::new();
        for r in 0..nout {
            let key = (bm[r], bx[r], bn.as_ref().map_or(0, |b| b[r]));
            groups.entry(key).or_default().push(r);
        }
        let mut slots: Vec<Option<SingleTaskGP>> = vec![None; nout * m];
        for ((b, ix, inn), members) in groups {
            for t in 0..m {
                let gp = self.model(b, t);
                let ys: Vec<Matrix> = members
                    .iter()
                    .map(|&r| y.matrices()[by[r]].columns(t, 1).into_owned())
                    .collect();
                // Noise in the original space under a general outcome
                // transform depends on the observations (e.g. `Log`'s delta
                // method): transform it with each set of observations.
                if let (NoiseSpace::Original, Some(v), Some(chain)) =
                    (space, noise, gp.outcome_chain())
                {
                    let x_tf = gp.transform_inputs_mode(&x.matrices()[ix], mode)?;
                    let col = column_rows(&v.matrices()[inn], t, k);
                    for (&r, yr) in members.iter().zip(&ys) {
                        let (_, nz) = chain.transform(yr, Some(&col), &x_tf)?;
                        let mut c = gp.condition_mode(
                            &x.matrices()[ix],
                            std::slice::from_ref(yr),
                            nz.as_ref(),
                            mode,
                        )?;
                        slots[r * m + t] = c.pop();
                    }
                    continue;
                }
                let nz = noise.map(|v| {
                    let mut col = column_rows(&v.matrices()[inn], t, k);
                    if space == NoiseSpace::Original {
                        if let Some(st) = gp.outcome_transform() {
                            let s = st.stdvs()[0];
                            col = col.map(|v| v / (s * s));
                        }
                    }
                    col
                });
                let conditioned = gp.condition_mode(&x.matrices()[ix], &ys, nz.as_ref(), mode)?;
                for (&r, c) in members.iter().zip(conditioned) {
                    slots[r * m + t] = Some(c);
                }
            }
        }
        let models = slots
            .into_iter()
            .map(|s| s.expect("every element is conditioned"))
            .collect();
        Ok(BatchedMultiOutputGP::from_elements(
            out,
            m,
            self.conditioned_transform(&bm),
            models,
        ))
    }

    /// BoTorch's `fantasize(X, sampler, observation_noise)`: samples
    /// `n_f = sampler.num_samples()` sets of observations at `x`
    /// (`batch_x × n' × d`) from the joint posterior over all outputs with
    /// observation noise (base samples `n_f × n'·m`, shared by all batch
    /// elements, column `p·m + t` for point `p` and output `t`) and
    /// conditions the model on them. The fantasy model has the batch shape
    /// `[n_f] + broadcast(batch_x, model batch shape)`.
    ///
    /// `observation_noise` (`batch × 1 × m` or `batch × n' × m`, transformed
    /// space) defaults to the inferred noise level, or the mean training
    /// noise of each element of a fixed-noise model.
    ///
    /// # Errors
    /// See [`BatchedMultiOutputGP::condition_on_observations`].
    pub fn fantasize(
        &self,
        x: &BatchedMatrix,
        sampler: &SobolQMCNormalSampler,
        observation_noise: Option<&BatchedMatrix>,
    ) -> Result<BatchedMultiOutputGP> {
        let (m, d) = (self.num_outputs(), self.dim());
        let k = x.nrows();
        let mode = TransformMode::Fantasize;
        self.model(0, 0).require_one_to_one(mode, "fantasize")?;
        if x.ncols() != d {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X with {d} columns (d), got {}",
                x.ncols()
            )));
        }
        let empty: &[usize] = &[];
        let post_shape = broadcast_shapes(&[
            self.batch_shape(),
            x.batch_shape(),
            observation_noise.map_or(empty, |v| v.batch_shape()),
        ])
        .ok_or_else(|| {
            BotorchError::DimensionMismatch(format!(
                "The batch shapes of X ({:?}) and of the model ({:?}) are not broadcastable.",
                x.batch_shape(),
                self.batch_shape()
            ))
        })?;
        if let Some(v) = observation_noise {
            if (v.nrows() != k && v.nrows() != 1) || v.ncols() != m {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected observation noise with 1 or {k} rows and {m} columns, got {}x{}",
                    v.nrows(),
                    v.ncols()
                )));
            }
        }
        let bm = broadcast_indices(self.batch_shape(), &post_shape);
        let bx = broadcast_indices(x.batch_shape(), &post_shape);
        let bn = observation_noise.map(|v| broadcast_indices(v.batch_shape(), &post_shape));
        let n_f = sampler.num_samples();
        let np = bm.len();
        // The noise of the fantasy observations, per posterior batch element.
        let noise: Option<Vec<Matrix>> = match observation_noise {
            Some(v) => Some(
                (0..np)
                    .map(|p| {
                        let src = &v.matrices()[bn.as_ref().expect("noise indices")[p]];
                        Matrix::from_fn(k, m, |j, t| src[(if src.nrows() == 1 { 0 } else { j }, t)])
                    })
                    .collect(),
            ),
            None => {
                let defaults: Option<Vec<Vec<f64>>> = (0..np)
                    .map(|p| {
                        (0..m)
                            .map(|t| self.model(bm[p], t).default_fantasy_noise())
                            .collect()
                    })
                    .collect();
                defaults.map(|ds| {
                    ds.into_iter()
                        .map(|dv| Matrix::from_fn(k, m, |_, t| dv[t]))
                        .collect()
                })
            }
        };
        let mut ys = vec![Matrix::zeros(k, m); n_f * np];
        if k > 0 {
            let z = sampler.base_samples(k * m)?;
            for p in 0..np {
                let xp = &x.matrices()[bx[p]];
                let parts = (0..m)
                    .map(|t| {
                        let gp = self.model(bm[p], t);
                        match &noise {
                            Some(nz) => {
                                let col: Vec<f64> = nz[p].column(t).iter().copied().collect();
                                gp.posterior_impl(xp, ObservationNoise::Explicit(&col), mode)
                            }
                            None => gp.posterior_impl(xp, ObservationNoise::Likelihood, mode),
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let post = if m == 1 {
                    parts.into_iter().next().expect("one output")
                } else {
                    block_diagonal(&parts)
                };
                let samples = post.rsample_from_base_samples(&z)?;
                for (i, y) in ys.iter_mut().skip(p).step_by(np).enumerate() {
                    *y = Matrix::from_fn(k, m, |j, t| samples.value(i, j, t));
                }
            }
        }
        let mut y_shape = vec![n_f];
        y_shape.extend_from_slice(&post_shape);
        let y = BatchedMatrix::new(y_shape, ys)?;
        let nz = noise
            .map(|v| BatchedMatrix::new(post_shape.clone(), v))
            .transpose()?;
        self.condition_impl(x, &y, nz.as_ref(), NoiseSpace::Transformed, mode)
    }
}

/// The samplers of [`ModelListGP::fantasize`]: one sampler for all outputs
/// (BoTorch's `MCSampler`) or one per output (a `ListSampler`, required with
/// an evaluation mask).
#[derive(Clone, Copy, Debug)]
pub enum FantasySampler<'a> {
    /// The same sampler (and base samples) for every output.
    Shared(&'a SobolQMCNormalSampler),
    /// One sampler per output.
    PerOutput(&'a [Arc<SobolQMCNormalSampler>]),
}

impl ModelListGP {
    fn require_single_output_models(&self, what: &str) -> Result<()> {
        if self.models().iter().any(|m| m.num_outputs() != 1) {
            return Err(BotorchError::Unsupported(format!(
                "ModelListGP.{what} requires single-output sub-models (BoTorch handles the \
                 outputs of a list one sub-model at a time)"
            )));
        }
        Ok(())
    }

    /// BoTorch's `ModelListGP.condition_on_observations(X, Y, noise=noise)`
    /// with meta-pytorch/botorch#3399: `x` holds the new inputs of every
    /// output (`batch × n' × d` each, raw space), `y` the observations
    /// (`batch × n' × m`, original space) and `noise` the noise variances
    /// in the *original* outcome space (`... × n' × m`). Every sub-model is
    /// conditioned with its own input and outcome transforms (BoTorch main
    /// passes untransformed inputs to GPyTorch, see the module
    /// documentation of [`crate::models::gp`]).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of inputs,
    /// outputs or inconsistent shapes; [`BotorchError::Unsupported`] for
    /// multi-output sub-models; errors of
    /// [`BatchedMultiOutputGP::condition_on_observations`].
    pub fn condition_on_observations(
        &self,
        x: &[BatchedMatrix],
        y: &BatchedMatrix,
        noise: Option<&BatchedMatrix>,
    ) -> Result<ModelListGP> {
        let m = self.num_outputs();
        if y.ncols() != m {
            return Err(BotorchError::DimensionMismatch(format!(
                "Incorrect number of outputs for observations. Received {} observation \
                 outputs, but model has {m} outputs.",
                y.ncols()
            )));
        }
        if x.len() != m {
            return Err(BotorchError::DimensionMismatch(format!(
                "Incorrect number of inputs for observations. Received {} observation inputs, \
                 but model has {m} outputs.",
                x.len()
            )));
        }
        self.require_single_output_models("condition_on_observations")?;
        let pick = |b: &BatchedMatrix, t: usize| -> Result<BatchedMatrix> {
            BatchedMatrix::new(
                b.batch_shape().to_vec(),
                b.matrices()
                    .iter()
                    .map(|mat| mat.columns(t, 1).into_owned())
                    .collect(),
            )
        };
        let models = self
            .models()
            .iter()
            .enumerate()
            .map(|(t, sub)| {
                let yt = pick(y, t)?;
                let nt = noise.map(|v| pick(v, t)).transpose()?;
                sub.condition_impl(
                    &x[t],
                    &yt,
                    nt.as_ref(),
                    NoiseSpace::Original,
                    TransformMode::Eval,
                )
                .map(Arc::new)
            })
            .collect::<Result<Vec<_>>>()?;
        ModelListGP::from_shared(models)
    }

    /// BoTorch's `ModelList.fantasize(X, sampler, observation_noise,
    /// evaluation_mask)`: every (single-output) sub-model is fantasized at
    /// `x` (`batch × n' × d`) with its sampler (or the shared one) and the
    /// fantasy models are collected in a list. With an `evaluation_mask`
    /// (`n' × m`), output `t` is fantasized only at the points `p` with
    /// `mask[(p, t)]` (and conditioned on no observation if there are none),
    /// which requires one sampler per output. `observation_noise`
    /// (`batch × 1 × m` or `batch × n' × m`, transformed space) defaults to
    /// each sub-model's default fantasy noise.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a mask of the wrong shape,
    /// [`BotorchError::InvalidArgument`] for a mask with a shared sampler or
    /// samplers with different numbers of samples,
    /// [`BotorchError::Unsupported`] for multi-output sub-models; errors of
    /// [`BatchedMultiOutputGP::fantasize`].
    pub fn fantasize(
        &self,
        x: &BatchedMatrix,
        sampler: FantasySampler<'_>,
        observation_noise: Option<&BatchedMatrix>,
        evaluation_mask: Option<&BoolMatrix>,
    ) -> Result<ModelListGP> {
        let m = self.num_outputs();
        let k = x.nrows();
        if let Some(mask) = evaluation_mask {
            if mask.shape() != (k, m) {
                return Err(BotorchError::DimensionMismatch(format!(
                    "Expected evaluation_mask of shape `{k} x {m}`, but got {}x{}.",
                    mask.nrows(),
                    mask.ncols()
                )));
            }
            if matches!(sampler, FantasySampler::Shared(_)) {
                return Err(BotorchError::InvalidArgument(
                    "Decoupled fantasization requires a list of samplers.".into(),
                ));
            }
        }
        if let FantasySampler::PerOutput(s) = sampler {
            if s.len() != m {
                return Err(BotorchError::InvalidArgument(format!(
                    "expected one sampler per output ({m}), got {}",
                    s.len()
                )));
            }
            if s.iter().any(|v| v.num_samples() != s[0].num_samples()) {
                return Err(BotorchError::InvalidArgument(
                    "the samplers of the outputs must draw the same number of fantasies".into(),
                ));
            }
        }
        self.require_single_output_models("fantasize")?;
        let models = self
            .models()
            .iter()
            .enumerate()
            .map(|(t, sub)| {
                let rows: Vec<usize> = match evaluation_mask {
                    Some(mask) => (0..k).filter(|&p| mask[(p, t)]).collect(),
                    None => (0..k).collect(),
                };
                let select = |b: &BatchedMatrix, cols: Option<usize>| -> Result<BatchedMatrix> {
                    BatchedMatrix::new(
                        b.batch_shape().to_vec(),
                        b.matrices()
                            .iter()
                            .map(|mat| {
                                let r: Vec<usize> = if mat.nrows() == 1 && k != 1 {
                                    vec![0]
                                } else {
                                    rows.clone()
                                };
                                let c: Vec<usize> =
                                    cols.map_or((0..mat.ncols()).collect(), |c| vec![c]);
                                Matrix::from_fn(r.len(), c.len(), |i, j| mat[(r[i], c[j])])
                            })
                            .collect(),
                    )
                };
                let xt = select(x, None)?;
                let nt = observation_noise.map(|v| select(v, Some(t))).transpose()?;
                let s: &SobolQMCNormalSampler = match sampler {
                    FantasySampler::Shared(s) => s,
                    FantasySampler::PerOutput(s) => &s[t],
                };
                sub.fantasize(&xt, s, nt.as_ref()).map(Arc::new)
            })
            .collect::<Result<Vec<_>>>()?;
        ModelListGP::from_shared(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Model, SingleTaskGPOptions};
    use crate::transforms::Normalize;

    fn data(n: usize, d: usize, m: usize, shift: f64) -> (Matrix, Matrix) {
        let x = Matrix::from_fn(n, d, |i, j| {
            ((i * 7 + j * 3) % n) as f64 / n as f64 + 0.03 * j as f64
        });
        let y = Matrix::from_fn(n, m, |i, t| {
            let s: f64 = x.row(i).iter().sum();
            (s * (1.0 + t as f64)).sin() + shift * t as f64
        });
        (x, y)
    }

    #[test]
    fn broadcasting() {
        assert_eq!(broadcast_shapes(&[&[3], &[2, 1], &[]]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[&[3], &[2]]), None);
        assert_eq!(broadcast_indices(&[3], &[2, 3]), vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(broadcast_indices(&[2, 1], &[2, 3]), vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(broadcast_indices(&[], &[2]), vec![0, 0]);
    }

    #[test]
    fn fantasy_models_of_multi_output_and_list_models() {
        let (x, y) = data(8, 2, 2, 0.3);
        let options = SingleTaskGPOptions {
            input_transform: Some(Normalize::new(2)),
            ..Default::default()
        };
        let gp = BatchedMultiOutputGP::multi_output(x.clone(), y.clone(), options.clone()).unwrap();
        let xn = BatchedMatrix::new(
            vec![3],
            (0..3)
                .map(|b| Matrix::from_fn(2, 2, |i, j| 0.1 + 0.3 * (b + i + j) as f64 / 3.0))
                .collect(),
        )
        .unwrap();
        let sampler = SobolQMCNormalSampler::new(4, Some(3)).unwrap();
        let fm = gp.fantasize(&xn, &sampler, None).unwrap();
        assert_eq!(fm.batch_shape(), &[4, 3]);
        assert_eq!(fm.num_train(), 10);
        // Element (i, b) equals the single-output models conditioned on the
        // samples drawn from the posterior at xn[b].
        let z = sampler.base_samples(4).unwrap();
        let xt = Matrix::from_row_slice(2, 2, &[0.4, 0.4, 0.9, 0.2]);
        for b in 0..3 {
            let post = gp.posterior_at(0, &xn.matrices()[b], true).unwrap();
            let s = post.rsample_from_base_samples(&z).unwrap();
            for i in 0..4 {
                for t in 0..2 {
                    let yt = Matrix::from_fn(2, 1, |j, _| s.value(i, j, t));
                    let c = gp
                        .model(0, t)
                        .condition_on_observations(&xn.matrices()[b], &yt, None)
                        .unwrap();
                    let a = fm.model(i * 3 + b, t).posterior(&xt, false).unwrap();
                    let e = c.posterior(&xt, false).unwrap();
                    assert!((&a.mean - &e.mean).amax() < 1e-13);
                }
            }
        }
        // A model list of the two outputs with an evaluation mask.
        let subs: Vec<BatchedMultiOutputGP> = (0..2)
            .map(|t| {
                SingleTaskGP::new(x.clone(), y.columns(t, 1).into_owned(), options.clone())
                    .unwrap()
                    .into()
            })
            .collect();
        let list = ModelListGP::new(subs).unwrap();
        let samplers = vec![
            Arc::new(SobolQMCNormalSampler::new(4, Some(1)).unwrap()),
            Arc::new(SobolQMCNormalSampler::new(4, Some(2)).unwrap()),
        ];
        let mask = BoolMatrix::from_row_slice(2, 2, &[true, false, true, false]);
        let x1 = BatchedMatrix::single(Matrix::from_row_slice(2, 2, &[0.2, 0.3, 0.6, 0.7]));
        let fl = list
            .fantasize(&x1, FantasySampler::PerOutput(&samplers), None, Some(&mask))
            .unwrap();
        assert_eq!(fl.batch_shape(), &[4]);
        assert_eq!(fl.models()[0].num_train(), 10);
        // Output 1 is not observed: its fantasy models equal the model.
        assert_eq!(fl.models()[1].num_train(), 8);
        let a = fl.posterior_at(2, &xt, false).unwrap();
        let e = Model::posterior(&list, &xt, false).unwrap();
        assert_eq!(
            a.select_outputs(&[1]).unwrap(),
            e.select_outputs(&[1]).unwrap()
        );
        assert!(list
            .fantasize(&x1, FantasySampler::Shared(&samplers[0]), None, Some(&mask))
            .is_err());
        // Conditioning a list with noise in the original space.
        let yn = BatchedMatrix::single(Matrix::from_row_slice(2, 2, &[0.1, 0.2, 0.3, 0.4]));
        let noise = BatchedMatrix::single(Matrix::from_element(2, 2, 0.04));
        let cl = list
            .condition_on_observations(&[x1.clone(), x1.clone()], &yn, Some(&noise))
            .unwrap();
        let sd = list.models()[0].outcome_transform(0, 0).unwrap().stdvs()[0];
        let direct = list.models()[0]
            .model(0, 0)
            .condition_on_observations(
                &x1.matrices()[0],
                &yn.matrices()[0].columns(0, 1).into_owned(),
                Some(&Matrix::from_element(2, 1, 0.04 / (sd * sd))),
            )
            .unwrap();
        let a = cl.models()[0].posterior_at(0, &xt, false).unwrap();
        let e = direct.posterior(&xt, false).unwrap();
        assert!((&a.mean - &e.mean).amax() < 1e-14);
    }
}
