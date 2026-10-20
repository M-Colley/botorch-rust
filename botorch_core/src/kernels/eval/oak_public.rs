//! Individual orthogonal additive covariance components.

use super::{NodeKind, TreeKernel, TreePrep};
use crate::error::{BotorchError, Result};
use crate::{Matrix, Vector};

impl TreeKernel {
    /// Number of additive components, including the constant bias.
    pub fn additive_component_count(&self) -> Result<usize> {
        let NodeKind::Oak(st) = &self.root.kind else {
            return Err(BotorchError::Unsupported(
                "an OrthogonalAdditiveKernel is required".into(),
            ));
        };
        Ok(1 + st.dim + st.c2.as_ref().map_or(0, Vec::len))
    }

    /// Covariance matrices for the bias, first order and interaction components.
    /// Their sum is the full covariance. Inputs must be prepared by this kernel.
    pub fn additive_components(&self, a: &TreePrep, b: &TreePrep) -> Result<Vec<Matrix>> {
        let count = self.additive_component_count()?;
        let NodeKind::Oak(st) = &self.root.kind else {
            unreachable!()
        };
        let mut out = vec![Matrix::zeros(a.n, b.n); count];
        for i in 0..a.n {
            for j in 0..b.n {
                let c = st.components(&a.root, i, &b.root, j, false);
                out[0][(i, j)] = st.offset;
                for k in 0..st.dim {
                    out[1 + k][(i, j)] = st.c1[k] * c[k];
                }
                if let Some(c2) = &st.c2 {
                    let mut p = 0;
                    for k in 0..st.dim {
                        for l in k + 1..st.dim {
                            out[1 + st.dim + p][(i, j)] = c2[p] * c[k] * c[l];
                            p += 1;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// A single component, without constructing the remaining covariance matrices.
    pub fn additive_component_covariance(
        &self,
        a: &TreePrep,
        b: &TreePrep,
        component: usize,
    ) -> Result<Matrix> {
        if component >= self.additive_component_count()? {
            return Err(BotorchError::InvalidArgument(
                "additive component index out of range".into(),
            ));
        }
        let NodeKind::Oak(st) = &self.root.kind else {
            unreachable!()
        };
        if component == 0 {
            return Ok(Matrix::from_element(a.n, b.n, st.offset));
        }
        let first = component <= st.dim;
        let mut pair = (component.saturating_sub(1), 0);
        let mut coefficient = if first { st.c1[pair.0] } else { 0.0 };
        if !first {
            let p = component - 1 - st.dim;
            let mut index = 0;
            for k in 0..st.dim {
                for l in k + 1..st.dim {
                    if index == p {
                        pair = (k, l);
                    }
                    index += 1;
                }
            }
            coefficient = st.c2.as_ref().expect("validated component")[p];
        }
        let value = |k, i, j| {
            st.base(k, a.root.x[i * st.dim + k], b.root.x[j * st.dim + k])
                .0
                - st.s(&a.root, k, i) * st.s(&b.root, k, j)
        };
        Ok(Matrix::from_fn(a.n, b.n, |i, j| {
            let first_value = value(pair.0, i, j);
            coefficient
                * if first {
                    first_value
                } else {
                    first_value * value(pair.1, i, j)
                }
        }))
    }

    /// One component's cross covariance and derivatives in the first input.
    pub fn additive_component_cross_grad(
        &self,
        a: &TreePrep,
        ia: usize,
        b: &TreePrep,
        component: usize,
    ) -> Result<(Vector, Matrix)> {
        if component >= self.additive_component_count()? || ia >= a.n {
            return Err(BotorchError::InvalidArgument(
                "additive component or point index out of range".into(),
            ));
        }
        let NodeKind::Oak(st) = &self.root.kind else {
            unreachable!()
        };
        let mut values = Vector::zeros(b.n);
        let mut gradient = Matrix::zeros(b.n, self.input_dim);
        for j in 0..b.n {
            let c = st.components(&a.root, ia, &b.root, j, false);
            let mut weights = vec![0.0; st.dim];
            if component == 0 {
                values[j] = st.offset;
            } else if component <= st.dim {
                let k = component - 1;
                values[j] = st.c1[k] * c[k];
                weights[k] = st.c1[k];
            } else {
                let p = component - 1 - st.dim;
                let mut pair = (0, 0);
                let mut index = 0;
                for k in 0..st.dim {
                    for l in k + 1..st.dim {
                        if index == p {
                            pair = (k, l);
                        }
                        index += 1;
                    }
                }
                let coefficient = st.c2.as_ref().expect("validated component")[p];
                let (k, l) = pair;
                values[j] = coefficient * c[k] * c[l];
                weights[k] = coefficient * c[l];
                weights[l] = coefficient * c[k];
            }
            for (k, weight) in weights.into_iter().enumerate() {
                let (_, g, u) = st.base(k, a.root.x[ia * st.dim + k], b.root.x[j * st.dim + k]);
                let derivative = -g * u / st.ls[k] - st.ds(&a.root, k, ia) * st.s(&b.root, k, j);
                let column = self.root.dims.as_ref().map_or(k, |dims| dims[k]);
                gradient[(j, column)] = weight * derivative;
            }
        }
        Ok((values, gradient))
    }

    /// Unorthogonalized one-dimensional base covariances, one per input.
    pub fn additive_base_covariances(&self, x1: &Matrix, x2: &Matrix) -> Result<Vec<Matrix>> {
        self.prepare(x1)?;
        self.prepare(x2)?;
        let NodeKind::Oak(st) = &self.root.kind else {
            return Err(BotorchError::Unsupported(
                "an OrthogonalAdditiveKernel is required".into(),
            ));
        };
        Ok((0..st.dim)
            .map(|k| {
                let column = self.root.dims.as_ref().map_or(k, |dims| dims[k]);
                Matrix::from_fn(x1.nrows(), x2.nrows(), |i, j| {
                    st.base(k, x1[(i, column)], x2[(j, column)]).0
                })
            })
            .collect())
    }

    /// Quadrature normalizers with a caller-specified lower bound.
    pub fn additive_normalizers(&self, eps: f64) -> Result<Vec<f64>> {
        if !eps.is_finite() || eps <= 0.0 {
            return Err(BotorchError::InvalidArgument(
                "eps must be finite and positive".into(),
            ));
        }
        let NodeKind::Oak(st) = &self.root.kind else {
            return Err(BotorchError::Unsupported(
                "an OrthogonalAdditiveKernel is required".into(),
            ));
        };
        Ok((0..st.dim)
            .map(|k| {
                let mut total = 0.0;
                for (i, &zi) in st.z.iter().enumerate() {
                    for (j, &zj) in st.z.iter().enumerate() {
                        total += st.w[i] * st.base(k, zi, zj).0 * st.w[j];
                    }
                }
                total.max(eps)
            })
            .collect())
    }
}
