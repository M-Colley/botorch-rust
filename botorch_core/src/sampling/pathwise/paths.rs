//! Sample paths (`botorch.sampling.pathwise.paths`): the [`SamplePath`]
//! trait, generalized linear paths ([`GeneralizedLinearPath`]) and their
//! compositions ([`PathList`], [`PathDict`]).

use std::fmt::Debug;
use std::sync::Arc;

use crate::error::{BotorchError, Result};
use crate::sampling::pathwise::features::FeatureMap;
use crate::sampling::pathwise::utils::OutcomeUntransform;
use crate::transforms::Normalize;
use crate::Matrix;

/// Values of sample paths at `n` points: one `num_samples × n` matrix per
/// output.
pub type PathValues = Vec<Matrix>;

/// Derivatives of sample-path values with respect to their points:
/// `gradients[t][s]` is the `n × d` matrix of `∂f_{s,t}(x_i)/∂x_i` for output
/// `t` and sample `s`.
pub type PathGradients = Vec<Vec<Matrix>>;

/// A batch of `num_samples` deterministic functions `R^d → R^m` (BoTorch's
/// `SamplePath`), evaluated on raw inputs.
pub trait SamplePath: Send + Sync + Debug {
    /// The input dimension `d`.
    fn dim(&self) -> usize;

    /// The number of sample paths.
    fn num_samples(&self) -> usize;

    /// The number of outputs `m`.
    fn num_outputs(&self) -> usize;

    /// The values at the rows of `x` (`n × d`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if `x` does not have `d` columns.
    fn forward(&self, x: &Matrix) -> Result<PathValues>;

    /// The values at the rows of `x` and their derivatives with respect to
    /// the points.
    ///
    /// # Errors
    /// See [`SamplePath::forward`].
    fn forward_with_grad(&self, x: &Matrix) -> Result<(PathValues, PathGradients)>;
}

/// Checks the number of columns of path inputs.
pub(crate) fn check_inputs(x: &Matrix, d: usize) -> Result<()> {
    if x.ncols() != d {
        return Err(BotorchError::DimensionMismatch(format!(
            "Expected inputs with {d} columns, got {}",
            x.ncols()
        )));
    }
    Ok(())
}

/// A sample path in the form of a generalized linear model (BoTorch's
/// `GeneralizedLinearPath`, single output):
///
/// ```text
/// path_s(x) = output_transform(bias + φ(z)ᵀ w_s),   z = input_transform(x),
/// ```
///
/// with a feature map `φ`, the weights `w_s` of sample `s` (the rows of
/// `weight`), an optional constant `bias` (BoTorch's `bias_module`, the
/// model's `ConstantMean`), an optional input transform (eval mode) and an
/// optional outcome untransform.
#[derive(Clone, Debug)]
pub struct GeneralizedLinearPath {
    feature_map: Arc<dyn FeatureMap>,
    weight: Matrix,
    bias: Option<f64>,
    input_transform: Option<Normalize>,
    output_transform: Option<OutcomeUntransform>,
}

impl GeneralizedLinearPath {
    /// A path with `num_samples × L` weights for a feature map with `L`
    /// features.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] if the weights do not have one
    /// column per feature, no rows, or the input transform has a different
    /// dimension.
    pub fn new(
        feature_map: Arc<dyn FeatureMap>,
        weight: Matrix,
        bias: Option<f64>,
        input_transform: Option<Normalize>,
        output_transform: Option<OutcomeUntransform>,
    ) -> Result<Self> {
        if weight.ncols() != feature_map.num_features() || weight.nrows() == 0 {
            return Err(BotorchError::DimensionMismatch(format!(
                "the weights of a path with {} features must be `num_samples x {}`, got {}x{}",
                feature_map.num_features(),
                feature_map.num_features(),
                weight.nrows(),
                weight.ncols()
            )));
        }
        if let Some(tf) = &input_transform {
            if tf.dim() != feature_map.dim() {
                return Err(BotorchError::DimensionMismatch(format!(
                    "the input transform has dimension {} but the feature map {}",
                    tf.dim(),
                    feature_map.dim()
                )));
            }
        }
        Ok(Self {
            feature_map,
            weight,
            bias,
            input_transform,
            output_transform,
        })
    }

    /// The feature map.
    pub fn feature_map(&self) -> &Arc<dyn FeatureMap> {
        &self.feature_map
    }

    /// The weights (`num_samples × L`).
    pub fn weight(&self) -> &Matrix {
        &self.weight
    }

    /// The constant bias, if any.
    pub fn bias(&self) -> Option<f64> {
        self.bias
    }

    /// The input transform, if any.
    pub fn input_transform(&self) -> Option<&Normalize> {
        self.input_transform.as_ref()
    }

    /// The outcome untransform, if any.
    pub fn output_transform(&self) -> Option<&OutcomeUntransform> {
        self.output_transform.as_ref()
    }

    /// The inputs after the input transform.
    pub(crate) fn transform(&self, x: &Matrix) -> Result<Matrix> {
        check_inputs(x, self.feature_map.dim())?;
        let z = match &self.input_transform {
            Some(tf) => tf.transform(x)?,
            None => x.clone(),
        };
        self.feature_map.validate(&z)?;
        Ok(z)
    }

    /// `∂z/∂x` of the input transform (per coordinate).
    pub(crate) fn input_jacobian(&self) -> Vec<f64> {
        match &self.input_transform {
            Some(tf) => tf.coefficient().iter().map(|c| 1.0 / c).collect(),
            None => vec![1.0; self.feature_map.dim()],
        }
    }

    /// `bias + φ(z)ᵀ w_s` at the rows of the transformed inputs `z`
    /// (`num_samples × n`), without the output transform (BoTorch's
    /// `forward`, which bypasses the module's transforms).
    pub fn forward_transformed(&self, z: &Matrix) -> Matrix {
        let features = self.feature_map.features(z);
        let mut out = &self.weight * features.transpose();
        if let Some(b) = self.bias {
            out.add_scalar_mut(b);
        }
        out
    }

    /// Values (`num_samples`) and derivatives (`num_samples × d`, with
    /// respect to `z`) at one transformed point, without the output
    /// transform.
    pub(crate) fn forward_transformed_with_grad(&self, z: &[f64]) -> (Vec<f64>, Matrix) {
        let (f, df) = self.feature_map.features_with_grad(z);
        let mut values: Vec<f64> = (&self.weight * f).iter().copied().collect();
        if let Some(b) = self.bias {
            values.iter_mut().for_each(|v| *v += b);
        }
        (values, &self.weight * df)
    }

    fn untransform(&self, values: &mut Matrix) {
        if let Some(ot) = &self.output_transform {
            values.apply(|v| *v = ot.apply(*v));
        }
    }
}

impl SamplePath for GeneralizedLinearPath {
    fn dim(&self) -> usize {
        self.feature_map.dim()
    }

    fn num_samples(&self) -> usize {
        self.weight.nrows()
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn forward(&self, x: &Matrix) -> Result<PathValues> {
        let z = self.transform(x)?;
        let mut values = self.forward_transformed(&z);
        self.untransform(&mut values);
        Ok(vec![values])
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(PathValues, PathGradients)> {
        let z = self.transform(x)?;
        let (n, d) = z.shape();
        let s = self.num_samples();
        let jac = self.input_jacobian();
        let scale = self.output_transform.map_or(1.0, |o| o.std);
        let mut values = Matrix::zeros(s, n);
        let mut grads = vec![Matrix::zeros(n, d); s];
        for i in 0..n {
            let zi: Vec<f64> = z.row(i).iter().copied().collect();
            let (v, dv) = self.forward_transformed_with_grad(&zi);
            for (k, g) in grads.iter_mut().enumerate() {
                values[(k, i)] = v[k];
                for j in 0..d {
                    g[(i, j)] = scale * dv[(k, j)] * jac[j];
                }
            }
        }
        self.untransform(&mut values);
        Ok((vec![values], vec![grads]))
    }
}

/// Checks that paths can be combined: same input dimension and number of
/// samples.
fn check_compatible(paths: &[&Arc<dyn SamplePath>]) -> Result<(usize, usize)> {
    let Some(first) = paths.first() else {
        return Err(BotorchError::InvalidArgument(
            "a composition of paths needs at least one path".into(),
        ));
    };
    let (d, s) = (first.dim(), first.num_samples());
    if paths.iter().any(|p| p.dim() != d || p.num_samples() != s) {
        return Err(BotorchError::DimensionMismatch(
            "the paths of a composition must have the same input dimension and number of samples"
                .into(),
        ));
    }
    Ok((d, s))
}

/// A list of sample paths (BoTorch's `PathList` without a join rule): the
/// outputs of the paths in order, e.g. the outputs of the sub-models of a
/// `ModelListGP` or of a multi-output model.
#[derive(Clone, Debug)]
pub struct PathList {
    paths: Vec<Arc<dyn SamplePath>>,
    dim: usize,
    num_samples: usize,
    num_outputs: usize,
}

impl PathList {
    /// A list of paths with the same input dimension and number of samples.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an empty list and
    /// [`BotorchError::DimensionMismatch`] for incompatible paths.
    pub fn new(paths: Vec<Arc<dyn SamplePath>>) -> Result<Self> {
        let (dim, num_samples) = check_compatible(&paths.iter().collect::<Vec<_>>())?;
        let num_outputs = paths.iter().map(|p| p.num_outputs()).sum();
        Ok(Self {
            paths,
            dim,
            num_samples,
            num_outputs,
        })
    }

    /// The paths.
    pub fn paths(&self) -> &[Arc<dyn SamplePath>] {
        &self.paths
    }

    /// The number of paths.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Whether the list is empty (never true for a constructed list).
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

impl SamplePath for PathList {
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn forward(&self, x: &Matrix) -> Result<PathValues> {
        check_inputs(x, self.dim)?;
        let mut out = Vec::with_capacity(self.num_outputs);
        for p in &self.paths {
            out.extend(p.forward(x)?);
        }
        Ok(out)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(PathValues, PathGradients)> {
        check_inputs(x, self.dim)?;
        let mut values = Vec::with_capacity(self.num_outputs);
        let mut grads = Vec::with_capacity(self.num_outputs);
        for p in &self.paths {
            let (v, g) = p.forward_with_grad(x)?;
            values.extend(v);
            grads.extend(g);
        }
        Ok((values, grads))
    }
}

/// How a [`PathDict`] combines the outputs of its paths (BoTorch's `join`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathJoin {
    /// The elementwise sum of the paths' values (BoTorch's `join=sum`, used
    /// by `MatheronPath`); all paths must have the same number of outputs.
    Sum,
}

/// A dictionary of named sample paths (BoTorch's `PathDict`). Without a join
/// rule the outputs of the paths are listed in insertion order; with
/// [`PathJoin::Sum`] they are summed, and the sum may be untransformed
/// (one [`OutcomeUntransform`] per output).
#[derive(Clone, Debug)]
pub struct PathDict {
    paths: Vec<(String, Arc<dyn SamplePath>)>,
    join: Option<PathJoin>,
    output_transform: Option<Vec<OutcomeUntransform>>,
    dim: usize,
    num_samples: usize,
    num_outputs: usize,
}

impl PathDict {
    /// A dictionary of paths with an optional join rule and output transform.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for an output transform without a join
    /// rule (BoTorch's message), [`BotorchError::DimensionMismatch`] for
    /// incompatible paths or transforms, [`BotorchError::InvalidArgument`]
    /// for an empty dictionary or duplicate names.
    pub fn new(
        paths: Vec<(String, Arc<dyn SamplePath>)>,
        join: Option<PathJoin>,
        output_transform: Option<Vec<OutcomeUntransform>>,
    ) -> Result<Self> {
        if join.is_none() && output_transform.is_some() {
            return Err(BotorchError::Unsupported(
                "Output transforms must be preceded by a join rule.".into(),
            ));
        }
        let (dim, num_samples) =
            check_compatible(&paths.iter().map(|(_, p)| p).collect::<Vec<_>>())?;
        for (i, (name, _)) in paths.iter().enumerate() {
            if paths[..i].iter().any(|(other, _)| other == name) {
                return Err(BotorchError::InvalidArgument(format!(
                    "duplicate path name {name:?}"
                )));
            }
        }
        let num_outputs = match join {
            None => paths.iter().map(|(_, p)| p.num_outputs()).sum(),
            Some(PathJoin::Sum) => {
                let m = paths[0].1.num_outputs();
                if paths.iter().any(|(_, p)| p.num_outputs() != m) {
                    return Err(BotorchError::DimensionMismatch(
                        "the paths of a sum must have the same number of outputs".into(),
                    ));
                }
                m
            }
        };
        if let Some(ot) = &output_transform {
            if ot.len() != num_outputs {
                return Err(BotorchError::DimensionMismatch(format!(
                    "expected {num_outputs} output transforms, got {}",
                    ot.len()
                )));
            }
        }
        Ok(Self {
            paths,
            join,
            output_transform,
            dim,
            num_samples,
            num_outputs,
        })
    }

    /// The path named `key`, if any.
    pub fn get(&self, key: &str) -> Option<&Arc<dyn SamplePath>> {
        self.paths.iter().find(|(k, _)| k == key).map(|(_, p)| p)
    }

    /// The names of the paths, in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.paths.iter().map(|(k, _)| k.as_str())
    }

    /// The number of paths.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Whether the dictionary is empty (never true for a constructed one).
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    fn finish(&self, values: &mut [Matrix], grads: Option<&mut [Vec<Matrix>]>) {
        if let Some(ot) = &self.output_transform {
            for (v, t) in values.iter_mut().zip(ot) {
                v.apply(|x| *x = t.apply(*x));
            }
            if let Some(grads) = grads {
                for (g, t) in grads.iter_mut().zip(ot) {
                    for m in g.iter_mut() {
                        *m *= t.std;
                    }
                }
            }
        }
    }
}

impl SamplePath for PathDict {
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn forward(&self, x: &Matrix) -> Result<PathValues> {
        check_inputs(x, self.dim)?;
        let mut out: PathValues = Vec::new();
        for (i, (_, p)) in self.paths.iter().enumerate() {
            let v = p.forward(x)?;
            match self.join {
                None => out.extend(v),
                Some(PathJoin::Sum) if i == 0 => out = v,
                Some(PathJoin::Sum) => {
                    for (o, vi) in out.iter_mut().zip(v) {
                        *o += vi;
                    }
                }
            }
        }
        self.finish(&mut out, None);
        Ok(out)
    }

    fn forward_with_grad(&self, x: &Matrix) -> Result<(PathValues, PathGradients)> {
        check_inputs(x, self.dim)?;
        let mut values: PathValues = Vec::new();
        let mut grads: PathGradients = Vec::new();
        for (i, (_, p)) in self.paths.iter().enumerate() {
            let (v, g) = p.forward_with_grad(x)?;
            match self.join {
                None => {
                    values.extend(v);
                    grads.extend(g);
                }
                Some(PathJoin::Sum) if i == 0 => {
                    values = v;
                    grads = g;
                }
                Some(PathJoin::Sum) => {
                    for (o, vi) in values.iter_mut().zip(v) {
                        *o += vi;
                    }
                    for (o, gi) in grads.iter_mut().zip(g) {
                        for (os, gs) in o.iter_mut().zip(gi) {
                            *os += gs;
                        }
                    }
                }
            }
        }
        self.finish(&mut values, Some(&mut grads));
        Ok((values, grads))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{Kernel, KernelKind};
    use crate::sampling::pathwise::features::{gen_kernel_features, KernelEvaluationMap};
    use crate::Vector;

    fn path(seed: u64, bias: Option<f64>) -> GeneralizedLinearPath {
        let kernel =
            Kernel::new(KernelKind::Rbf, Vector::from_vec(vec![0.3, 0.6]), Some(1.2)).unwrap();
        let fm = gen_kernel_features(&kernel, 2, 32, seed).unwrap();
        let w = Matrix::from_fn(3, 32, |i, j| ((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.5);
        let tf =
            Normalize::with_bounds(&Matrix::from_row_slice(2, 2, &[-1.0, 0.0, 3.0, 2.0])).unwrap();
        GeneralizedLinearPath::new(
            Arc::new(fm),
            w,
            bias,
            Some(tf),
            Some(OutcomeUntransform {
                mean: 0.7,
                std: 2.5,
            }),
        )
        .unwrap()
    }

    #[test]
    fn linear_path_values_and_gradients() {
        let p = path(1, Some(0.3));
        let x = Matrix::from_row_slice(4, 2, &[0.0, 0.5, 1.2, 1.9, -0.7, 0.1, 2.5, 1.0]);
        let v = p.forward(&x).unwrap();
        assert_eq!((v.len(), v[0].shape()), (1, (3, 4)));
        let z = p.transform(&x).unwrap();
        let raw = p.forward_transformed(&z);
        assert!((raw.map(|r| 0.7 + 2.5 * r) - &v[0]).amax() < 1e-14);
        let (vg, g) = p.forward_with_grad(&x).unwrap();
        assert!((&vg[0] - &v[0]).amax() < 1e-13);
        for i in 0..4 {
            for k in 0..2 {
                let h = 1e-6;
                let mut xp = x.clone();
                xp[(i, k)] += h;
                let mut xm = x.clone();
                xm[(i, k)] -= h;
                let fd = (p.forward(&xp).unwrap()[0].column(i)
                    - p.forward(&xm).unwrap()[0].column(i))
                    / (2.0 * h);
                for s in 0..3 {
                    assert!(
                        (fd[s] - g[0][s][(i, k)]).abs() < 1e-6,
                        "{fd} {}",
                        g[0][s][(i, k)]
                    );
                }
            }
        }
        assert!(p.forward(&Matrix::zeros(2, 3)).is_err());
    }

    #[test]
    fn compositions() {
        let a: Arc<dyn SamplePath> = Arc::new(path(1, Some(0.3)));
        let b: Arc<dyn SamplePath> = Arc::new(path(2, None));
        let x = Matrix::from_row_slice(2, 2, &[0.2, 0.4, 1.0, 1.5]);
        let (va, vb) = (a.forward(&x).unwrap(), b.forward(&x).unwrap());
        let list = PathList::new(vec![a.clone(), b.clone()]).unwrap();
        assert_eq!(list.num_outputs(), 2);
        let vl = list.forward(&x).unwrap();
        assert_eq!(vl[0], va[0]);
        assert_eq!(vl[1], vb[0]);
        let sum = PathDict::new(
            vec![("a".into(), a.clone()), ("b".into(), b.clone())],
            Some(PathJoin::Sum),
            Some(vec![OutcomeUntransform {
                mean: 1.0,
                std: 2.0,
            }]),
        )
        .unwrap();
        let vs = sum.forward(&x).unwrap();
        assert!(((&va[0] + &vb[0]).map(|v| 1.0 + 2.0 * v) - &vs[0]).amax() < 1e-13);
        let (_, gs) = sum.forward_with_grad(&x).unwrap();
        let (_, ga) = a.forward_with_grad(&x).unwrap();
        let (_, gb) = b.forward_with_grad(&x).unwrap();
        assert!(((&ga[0][1] + &gb[0][1]) * 2.0 - &gs[0][1]).amax() < 1e-12);
        assert!(sum.get("b").is_some() && sum.keys().count() == 2);
        assert!(PathDict::new(
            vec![("a".into(), a.clone())],
            None,
            Some(vec![OutcomeUntransform {
                mean: 0.0,
                std: 1.0
            }])
        )
        .is_err());
        assert!(PathDict::new(vec![("a".into(), a.clone()), ("a".into(), b)], None, None).is_err());
        // Kernel evaluation features with a path.
        let kernel = Kernel::new(KernelKind::Rbf, Vector::from_vec(vec![0.3, 0.6]), None).unwrap();
        let map =
            KernelEvaluationMap::new(kernel, Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]))
                .unwrap();
        let p = GeneralizedLinearPath::new(
            Arc::new(map),
            Matrix::from_row_slice(1, 2, &[1.0, -1.0]),
            None,
            None,
            None,
        )
        .unwrap();
        let v = p
            .forward(&Matrix::from_row_slice(1, 2, &[0.0, 0.0]))
            .unwrap();
        let expected = 1.0 - (-0.5_f64 * (1.0 / 0.09 + 1.0 / 0.36)).exp();
        assert!((v[0][(0, 0)] - expected).abs() < 1e-15);
    }
}
