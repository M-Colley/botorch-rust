//! One-dimensional native kernel views for GPyTorch's feature batch mode.

use super::{KernelNode, KernelType};
use crate::error::{BotorchError, Result};

impl KernelNode {
    /// Separate this kernel's final feature dimension into covariance batches.
    ///
    /// Returned trees select their actual input columns and retain constraints,
    /// scalar parameters and the corresponding ARD entry. Composite kernels
    /// broadcast their children's feature batches as GPyTorch does.
    ///
    /// # Errors
    /// Invalid input/parameter dimensions and specialized kernels without a
    /// featurewise interpretation are rejected.
    pub fn feature_batch_nodes(
        &self,
        raw: &[f64],
        input_dim: usize,
    ) -> Result<Vec<(KernelNode, Vec<f64>)>> {
        self.validate(input_dim)?;
        if raw.len() != self.num_raw() {
            return Err(BotorchError::DimensionMismatch(
                "feature-batch raw parameters do not match the kernel".into(),
            ));
        }
        let dimensions: Vec<usize> = self
            .active_dims
            .clone()
            .unwrap_or_else(|| (0..input_dim).collect());
        let local_dim = dimensions.len();
        match &self.kind {
            KernelType::Scale | KernelType::Additive | KernelType::Product => {
                let own: usize = self.params.iter().map(|p| p.numel()).sum();
                let mut offset = own;
                let mut children = Vec::with_capacity(self.children.len());
                for child in &self.children {
                    let end = offset + child.num_raw();
                    children.push(child.feature_batch_nodes(&raw[offset..end], local_dim)?);
                    offset = end;
                }
                let mut width = children.iter().map(Vec::len).max().unwrap_or(local_dim);
                if children.iter().any(Vec::is_empty) && width <= 1 {
                    width = 0;
                }
                if children.iter().any(|c| c.len() != 1 && c.len() != width) {
                    return Err(BotorchError::DimensionMismatch(
                        "child kernel feature-batch dimensions do not broadcast".into(),
                    ));
                }
                let mut result = Vec::with_capacity(width);
                for index in 0..width {
                    let mut node = self.clone();
                    node.children.clear();
                    let mut values = raw[..own].to_vec();
                    for plans in &children {
                        let (child, child_raw) = &plans[if plans.len() == 1 { 0 } else { index }];
                        node.children.push(child.clone());
                        values.extend_from_slice(child_raw);
                    }
                    result.push((node, values));
                }
                Ok(result)
            }
            KernelType::Stationary(_)
            | KernelType::RationalQuadratic
            | KernelType::Periodic
            | KernelType::Linear
            | KernelType::Polynomial { .. }
            | KernelType::Cosine
            | KernelType::Categorical => {
                let mut result = Vec::with_capacity(local_dim);
                for (index, &column) in dimensions.iter().enumerate() {
                    let mut node = self.clone();
                    node.active_dims = Some(vec![column]);
                    let mut values = Vec::new();
                    let mut offset = 0;
                    for parameter in &mut node.params {
                        let size = parameter.numel();
                        if size > 1
                            && matches!(
                                parameter.name.as_str(),
                                "raw_lengthscale" | "raw_period_length" | "raw_variance"
                            )
                        {
                            if size != local_dim {
                                return Err(BotorchError::DimensionMismatch(
                                    "ARD parameters must match the selected feature count".into(),
                                ));
                            }
                            values.push(raw[offset + index]);
                            parameter.shape = vec![1, 1];
                        } else {
                            values.extend_from_slice(&raw[offset..offset + size]);
                        }
                        offset += size;
                    }
                    result.push((node, values));
                }
                Ok(result)
            }
            _ => Err(BotorchError::Unsupported(
                "last_dim_is_batch for this specialized kernel".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{KernelKind, TreeKernel};
    use crate::Matrix;

    fn positive(raw: f64) -> f64 {
        raw.exp().ln_1p()
    }

    fn evaluate(node: &KernelNode, raw: &[f64], x: &Matrix, y: &Matrix) -> Vec<Matrix> {
        node.feature_batch_nodes(raw, x.ncols())
            .unwrap()
            .into_iter()
            .map(|(node, parameters)| {
                let kernel = TreeKernel::build(&node, &parameters, x.ncols()).unwrap();
                kernel.cross(&kernel.prepare(x).unwrap(), &kernel.prepare(y).unwrap())
            })
            .collect()
    }

    #[test]
    fn feature_batch_trees_preserve_ard_values_and_active_dimension_order() {
        let x = Matrix::from_row_slice(2, 3, &[0.1, 0.8, 0.3, 0.6, 0.2, 0.7]);
        let y = Matrix::from_row_slice(3, 3, &[0.4, 0.1, 0.9, 0.3, 0.5, 0.7, 0.8, 0.2, 0.1]);
        let node = KernelNode::additive(vec![
            KernelNode::stationary(KernelKind::Rbf, Some(3)).unwrap(),
            KernelNode::scale(KernelNode::stationary(KernelKind::Matern52, Some(3)).unwrap()),
        ])
        .unwrap();
        let raw = [-0.3, -0.2, 0.1, 0.4, 0.5, 0.7, 0.9];
        let matrices = evaluate(&node, &raw, &x, &y);
        assert_eq!(matrices.len(), 3);
        for feature in 0..3 {
            for i in 0..2 {
                for j in 0..3 {
                    let distance = (x[(i, feature)] - y[(j, feature)]).abs();
                    let rbf = (-0.5 * (distance / positive(raw[feature])).powi(2)).exp();
                    let r = 5.0_f64.sqrt() * distance / positive(raw[4 + feature]);
                    let expected = rbf + positive(raw[3]) * (1.0 + r + r * r / 3.0) * (-r).exp();
                    assert!((matrices[feature][(i, j)] - expected).abs() < 2e-14);
                }
            }
        }
        let node = KernelNode::scale(
            KernelNode::stationary(KernelKind::Rbf, Some(2))
                .unwrap()
                .with_active_dims(vec![2, 0])
                .unwrap(),
        );
        let raw = [0.2, -0.1, 0.4];
        let matrices = evaluate(&node, &raw, &x, &y);
        assert_eq!(matrices.len(), 2);
        for (feature, physical) in [2, 0].into_iter().enumerate() {
            let expected = positive(raw[0])
                * (-0.5
                    * ((x[(0, physical)] - y[(1, physical)]) / positive(raw[1 + feature])).powi(2))
                .exp();
            assert!((matrices[feature][(0, 1)] - expected).abs() < 2e-14);
        }
    }

    #[test]
    fn feature_batch_children_broadcast_single_feature() {
        let x = Matrix::from_row_slice(2, 3, &[0.1, 0.8, 0.3, 0.6, 0.2, 0.7]);
        let node = KernelNode::product(vec![
            KernelNode::stationary(KernelKind::Rbf, Some(3))
                .unwrap()
                .with_active_dims(vec![2, 0, 1])
                .unwrap(),
            KernelNode::linear(None)
                .unwrap()
                .with_active_dims(vec![1])
                .unwrap(),
        ])
        .unwrap();
        let raw = [-0.3, -0.2, 0.1, 0.4];
        let matrices = evaluate(&node, &raw, &x, &x);
        for (feature, physical) in [2, 0, 1].into_iter().enumerate() {
            let expected = (-0.5
                * ((x[(0, physical)] - x[(1, physical)]) / positive(raw[feature])).powi(2))
            .exp()
                * positive(raw[3])
                * x[(0, 1)]
                * x[(1, 1)];
            assert!((matrices[feature][(0, 1)] - expected).abs() < 2e-14);
        }
    }
}
