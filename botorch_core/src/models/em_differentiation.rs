//! Analytic directional differentiation through closed-form EM updates.

use crate::error::{BotorchError, Result};
use crate::kernels::TreeKernel;
use crate::linalg::psd_safe_cholesky;
use crate::models::em_gp::{EMDataset, EMOptions};
use crate::models::em_inducing::kernel_input_direction;
use crate::{Matrix, Vector};

/// Estimated moments and one directional derivative of each.
pub struct EMDirectional {
    /// Converged mean.
    pub mean: Vector,
    /// Converged covariance.
    pub covariance: Matrix,
    /// Mean tangent.
    pub mean_tangent: Vector,
    /// Covariance tangent.
    pub covariance_tangent: Matrix,
}

#[derive(Clone)]
struct Dual {
    value: Matrix,
    tangent: Matrix,
}
impl Dual {
    fn fixed(value: Matrix) -> Self {
        let tangent = Matrix::zeros(value.nrows(), value.ncols());
        Self { value, tangent }
    }
    fn add(&self, other: &Self) -> Self {
        Self {
            value: &self.value + &other.value,
            tangent: &self.tangent + &other.tangent,
        }
    }
    fn sub(&self, other: &Self) -> Self {
        Self {
            value: &self.value - &other.value,
            tangent: &self.tangent - &other.tangent,
        }
    }
    fn scale(&self, value: f64) -> Self {
        Self {
            value: &self.value * value,
            tangent: &self.tangent * value,
        }
    }
    fn mul(&self, other: &Self) -> Self {
        Self {
            value: &self.value * &other.value,
            tangent: &self.tangent * &other.value + &self.value * &other.tangent,
        }
    }
    fn transpose(&self) -> Self {
        Self {
            value: self.value.transpose(),
            tangent: self.tangent.transpose(),
        }
    }
    fn inverse(&self) -> Result<Self> {
        let inverse = psd_safe_cholesky(&self.value)?.inverse();
        let tangent = -&inverse * &self.tangent * &inverse;
        Ok(Self {
            value: inverse,
            tangent,
        })
    }
    fn project(&self) -> Self {
        let symmetric = (&self.value + self.value.transpose()) * 0.5;
        let tangent = (&self.tangent + self.tangent.transpose()) * 0.5;
        let eigen = symmetric.symmetric_eigen();
        let values = eigen.eigenvalues;
        let clipped = values.map(|v| v.max(1e-10));
        let rotated = eigen.eigenvectors.transpose() * tangent * &eigen.eigenvectors;
        let derivative = Matrix::from_fn(values.len(), values.len(), |i, j| {
            let coefficient = if values[i] >= 1e-10 && values[j] >= 1e-10 {
                1.0
            } else if values[i] < 1e-10 && values[j] < 1e-10 {
                0.0
            } else {
                (clipped[i] - clipped[j]) / (values[i] - values[j])
            };
            rotated[(i, j)] * coefficient
        });
        Self {
            value: &eigen.eigenvectors
                * Matrix::from_diagonal(&clipped)
                * eigen.eigenvectors.transpose(),
            tangent: &eigen.eigenvectors * derivative * eigen.eigenvectors.transpose(),
        }
    }
    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Self {
        Self {
            value: Matrix::from_fn(rows.len(), cols.len(), |i, j| {
                self.value[(rows[i], cols[j])]
            }),
            tangent: Matrix::from_fn(rows.len(), cols.len(), |i, j| {
                self.tangent[(rows[i], cols[j])]
            }),
        }
    }
}

/// Exact kernel matrix derivative along one raw-parameter direction.
///
/// Two-point contractions use the existing analytic kernel-tree derivatives,
/// including its active dimensions and composite kernels, without dense tapes.
fn kernel_direction(
    kernel: &TreeKernel,
    x: &Matrix,
    y: &Matrix,
    parameter: Option<(usize, f64)>,
    left_input: Option<(usize, usize)>,
    right_input: Option<(usize, usize)>,
) -> Result<Dual> {
    let value = kernel.covariance(x, y)?;
    let mut tangent = Matrix::zeros(x.nrows(), y.nrows());
    if let Some((parameter, transform)) = parameter {
        if transform != 0.0 {
            let weights = Matrix::from_row_slice(2, 2, &[0.0, 0.5, 0.5, 0.0]);
            for i in 0..x.nrows() {
                for j in 0..y.nrows() {
                    let points =
                        Matrix::from_fn(
                            2,
                            x.ncols(),
                            |r, c| if r == 0 { x[(i, c)] } else { y[(j, c)] },
                        );
                    let gradient = kernel.contract(&kernel.prepare(&points)?, &weights);
                    tangent[(i, j)] = *gradient.get(parameter).ok_or_else(|| {
                        BotorchError::InvalidArgument(
                            "EM kernel derivative coordinate out of range".into(),
                        )
                    })? * transform;
                }
            }
        }
    }
    if left_input.is_some() || right_input.is_some() {
        tangent += kernel_input_direction(kernel, x, y, left_input, right_input)?;
    }
    Ok(Dual { value, tangent })
}

enum Observation {
    Indexed(Vec<usize>),
    Interpolated { weights: Dual, residual: Dual },
}

/// Differentiate exact EM with fixed historical inputs and optional moving inducing locations.
///
/// `parameter` selects an actual kernel coordinate and its raw transform
/// derivative. Mean/noise tangents are already with respect to the selected
/// raw coordinate. Spectral PSD stabilization uses its analytic Fréchet map.
#[allow(clippy::too_many_arguments)]
pub fn em_directional(
    datasets: &[EMDataset],
    kernel: &TreeKernel,
    inducing: &Matrix,
    constant: f64,
    inducing_observations: bool,
    options: &EMOptions,
    parameter: Option<(usize, f64)>,
    constant_tangent: f64,
    noise_tangent: f64,
    inducing_direction: Option<(usize, usize)>,
    initial: Option<(&Vector, &Matrix)>,
) -> Result<EMDirectional> {
    let n = inducing.nrows();
    let count = datasets.len();
    if n == 0
        || inducing.ncols() == 0
        || count == 0
        || !constant.is_finite()
        || !constant_tangent.is_finite()
        || !noise_tangent.is_finite()
        || !options.noise.is_finite()
        || options.noise < 0.0
        || !(0.0..=1.0).contains(&options.shrinkage)
        || options.tolerance.is_some_and(|v| !v.is_finite() || v < 0.0)
        || parameter.is_some_and(|(_, d)| !d.is_finite())
        || datasets.iter().any(|d| {
            d.x.nrows() == 0
                || d.x.ncols() != inducing.ncols()
                || d.y.len() != d.x.nrows()
                || d.x.iter().chain(d.y.iter()).any(|v| !v.is_finite())
        })
        || inducing.iter().any(|v| !v.is_finite())
        || options.covariance_prior
            && options
                .iw_nu
                .is_some_and(|v| !v.is_finite() || v <= n as f64 - 1.0)
    {
        return Err(BotorchError::InvalidArgument(
            "EM derivatives require data and inducing points".into(),
        ));
    }
    let kernel_covariance = kernel_direction(
        kernel,
        inducing,
        inducing,
        parameter,
        inducing_direction,
        inducing_direction,
    )?;
    let inverse = kernel_covariance.inverse()?;
    let initial_covariance = if options.naive_init {
        Dual::fixed(Matrix::identity(n, n))
    } else {
        kernel_covariance.clone()
    };
    let mut mean = Dual {
        value: Matrix::from_element(n, 1, if options.naive_init { 0.0 } else { constant }),
        tangent: Matrix::from_element(
            n,
            1,
            if options.naive_init {
                0.0
            } else {
                constant_tangent
            },
        ),
    };
    let mut covariance = initial_covariance.clone();
    if let Some((seed_mean, seed_covariance)) = initial {
        if seed_mean.len() != n
            || seed_covariance.shape() != (n, n)
            || seed_mean
                .iter()
                .chain(seed_covariance.iter())
                .any(|v| !v.is_finite())
            || (seed_covariance - seed_covariance.transpose()).amax() > 1e-9
            || seed_covariance.clone().symmetric_eigen().eigenvalues.min() < -1e-9
        {
            return Err(BotorchError::InvalidArgument(
                "invalid EM warm-start moments".into(),
            ));
        }
        mean = Dual::fixed(Matrix::from_column_slice(n, 1, seed_mean.as_slice()));
        covariance = Dual::fixed(seed_covariance.clone());
    }
    let all: Vec<_> = (0..n).collect();
    let prior_mean = Dual {
        value: Matrix::from_element(n, 1, constant),
        tangent: Matrix::from_element(n, 1, constant_tangent),
    };
    let mut observations = Vec::with_capacity(count);
    for data in datasets {
        observations.push(if inducing_observations {
            let cross = kernel_direction(
                kernel,
                &data.x,
                inducing,
                parameter,
                None,
                inducing_direction,
            )?;
            let weights = cross.mul(&inverse);
            let residual = kernel_direction(kernel, &data.x, &data.x, parameter, None, None)?
                .sub(&weights.mul(&cross.transpose()));
            Observation::Interpolated { weights, residual }
        } else {
            let indices = (0..data.x.nrows())
                .map(|r| {
                    (0..n)
                        .find(|&i| {
                            (0..inducing.ncols()).all(|j| data.x[(r, j)] == inducing[(i, j)])
                        })
                        .ok_or_else(|| {
                            BotorchError::InvalidArgument(
                                "EM observation absent from inducing union".into(),
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            Observation::Indexed(indices)
        });
    }
    let mean_prior_inverse = if options.mean_prior {
        Some(initial_covariance.inverse()?)
    } else {
        None
    };
    for _ in 0..options.iterations {
        let mut conditional_means = Vec::with_capacity(count);
        let mut conditional_covariances = Vec::with_capacity(count);
        for (data, observation) in datasets.iter().zip(&observations) {
            let k = data.x.nrows();
            let (observed_mean, mut observed_covariance, cross) = match observation {
                Observation::Indexed(indices) => (
                    mean.submatrix(indices, &[0]),
                    covariance.submatrix(indices, indices),
                    covariance.submatrix(&all, indices),
                ),
                Observation::Interpolated { weights, residual } => {
                    let cross = covariance.mul(&weights.transpose());
                    let base = Dual {
                        value: Matrix::from_element(k, 1, constant),
                        tangent: Matrix::from_element(k, 1, constant_tangent),
                    };
                    (
                        base.add(&weights.mul(&mean.sub(&prior_mean))),
                        residual.add(&weights.mul(&cross)),
                        cross,
                    )
                }
            };
            for i in 0..k {
                observed_covariance.value[(i, i)] += options.noise;
                observed_covariance.tangent[(i, i)] += noise_tangent;
            }
            let precision = observed_covariance.inverse()?;
            let y = Dual::fixed(Matrix::from_column_slice(k, 1, data.y.as_slice()));
            let alpha = precision.mul(&y.sub(&observed_mean));
            conditional_means.push(mean.add(&cross.mul(&alpha)));
            conditional_covariances
                .push(covariance.sub(&cross.mul(&precision).mul(&cross.transpose())));
        }
        let sum = conditional_means
            .iter()
            .fold(Dual::fixed(Matrix::zeros(n, 1)), |sum, value| {
                sum.add(value)
            });
        let updated_mean = if let Some(prior_inverse) = &mean_prior_inverse {
            let current_inverse = covariance.inverse()?;
            let precision = current_inverse.scale(count as f64).add(prior_inverse);
            precision.inverse()?.mul(&current_inverse.mul(&sum))
        } else {
            sum.scale(1.0 / count as f64)
        };
        let mut scatter = Dual::fixed(Matrix::zeros(n, n));
        for (mean, covariance) in conditional_means.iter().zip(&conditional_covariances) {
            let diff = mean.sub(&updated_mean);
            scatter = scatter.add(covariance).add(&diff.mul(&diff.transpose()));
        }
        let mut updated_covariance = if options.covariance_prior {
            let scale = options.iw_nu.unwrap_or(n as f64 + 2.0) + n as f64 + 1.0;
            scatter
                .add(&initial_covariance.scale(scale))
                .scale(1.0 / (count as f64 + scale))
        } else {
            scatter.scale(1.0 / count as f64)
        };
        if options.shrinkage > 0.0 {
            let trace = updated_covariance.value.trace();
            let trace_tangent = updated_covariance.tangent.trace();
            let target_trace = kernel_covariance.value.trace();
            let target_tangent = kernel_covariance.tangent.trace();
            let denominator = target_trace.max(1e-12);
            let denominator_tangent = if target_trace >= 1e-12 {
                target_tangent
            } else {
                0.0
            };
            let ratio = trace / denominator;
            let ratio_tangent = (trace_tangent * denominator - trace * denominator_tangent)
                / (denominator * denominator);
            let target = Dual {
                value: &kernel_covariance.value * ratio,
                tangent: &kernel_covariance.tangent * ratio
                    + &kernel_covariance.value * ratio_tangent,
            };
            updated_covariance = updated_covariance
                .scale(1.0 - options.shrinkage)
                .add(&target.scale(options.shrinkage));
        }
        updated_covariance = updated_covariance.project();
        let mean_change = (&updated_mean.value - &mean.value).amax();
        let covariance_change = (&updated_covariance.value - &covariance.value).amax();
        mean = updated_mean;
        covariance = updated_covariance;
        if options
            .tolerance
            .is_some_and(|t| mean_change < t && covariance_change < t)
        {
            break;
        }
    }
    Ok(EMDirectional {
        mean: Vector::from_column_slice(mean.value.as_slice()),
        covariance: covariance.value,
        mean_tangent: Vector::from_column_slice(mean.tangent.as_slice()),
        covariance_tangent: covariance.tangent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{KernelKind, KernelNode};
    use crate::models::em_gp::pretrain_em_prior;

    #[test]
    fn psd_tangents_at_close_branches_and_tiny_shrinkage_target_are_correct() {
        let value = Matrix::from_diagonal(&Vector::from_vec(vec![1e-10 + 5e-14, 1e-10 - 5e-14]));
        let tangent = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let result = Dual { value, tangent }.project();
        assert!(
            (result.tangent - Matrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 0.0])).amax() < 1e-12
        );
        let x = Matrix::from_column_slice(2, 1, &[0.0, 1.0]);
        let data: Vec<_> = (0..3)
            .map(|i| EMDataset {
                x: x.clone(),
                y: Vector::from_vec(vec![i as f64, 0.2 - i as f64 * 0.4]),
            })
            .collect();
        let node = KernelNode::scale(KernelNode::stationary(KernelKind::Rbf, Some(1)).unwrap());
        let raw = [-35.0, -0.3];
        let kernel = TreeKernel::build(&node, &raw, 1).unwrap();
        let options = EMOptions {
            noise: 0.1,
            iterations: 2,
            naive_init: true,
            shrinkage: 0.5,
            tolerance: None,
            ..Default::default()
        };
        let indices = vec![vec![0, 1]; 3];
        let directional = em_directional(
            &data,
            &kernel,
            &x,
            0.25,
            false,
            &options,
            Some((0, node.transform_derivatives(&raw)[0])),
            0.0,
            0.0,
            None,
            None,
        )
        .unwrap();
        let run = |scale: f64| {
            pretrain_em_prior(
                &data,
                TreeKernel::build(&node, &[scale, raw[1]], 1).unwrap(),
                x.clone(),
                0.25,
                Some(&indices),
                &options,
            )
            .unwrap()
        };
        let expected = run(raw[0]);
        assert!((&directional.covariance - expected.covariance).amax() < 1e-10);
        let h = 1e-5;
        let numerical = (run(raw[0] + h).covariance - run(raw[0] - h).covariance) / (2.0 * h);
        assert!((numerical - directional.covariance_tangent).amax() < 1e-8);
    }

    #[test]
    fn complete_em_chain_derivatives_match_finite_differences() {
        let x = Matrix::from_column_slice(3, 1, &[0.0, 0.4, 1.0]);
        let data: Vec<_> = (0..4)
            .map(|i| EMDataset {
                x: x.rows(0, 2 + i % 2).into_owned(),
                y: Vector::from_fn(2 + i % 2, |j, _| {
                    ((i + j * 2) as f64).sin() + i as f64 * 0.2
                }),
            })
            .collect();
        let node = KernelNode::scale(KernelNode::stationary(KernelKind::Rbf, Some(1)).unwrap());
        let raw = [0.1, -0.3];
        let indices: Vec<Vec<usize>> = data.iter().map(|d| (0..d.x.nrows()).collect()).collect();
        for inducing in [false, true] {
            for priors in [false, true] {
                let z = if inducing {
                    Matrix::from_column_slice(3, 1, &[0.05, 0.5, 0.9])
                } else {
                    x.clone()
                };
                let options = EMOptions {
                    noise: 0.1,
                    iterations: 3,
                    mean_prior: priors,
                    covariance_prior: priors,
                    tolerance: None,
                    shrinkage: 0.13,
                    ..Default::default()
                };
                let kernel = TreeKernel::build(&node, &raw, 1).unwrap();
                let transforms = node.transform_derivatives(&raw);
                let run = |p: &[f64]| {
                    pretrain_em_prior(
                        &data,
                        TreeKernel::build(&node, &p[2..], 1).unwrap(),
                        z.clone(),
                        p[1],
                        if inducing { None } else { Some(&indices) },
                        &EMOptions {
                            noise: p[0],
                            ..options.clone()
                        },
                    )
                    .unwrap()
                };
                let mut state = vec![0.1, 0.25];
                state.extend_from_slice(&raw);
                let expected = run(&state);
                for parameter in 0..state.len() {
                    let directional = em_directional(
                        &data,
                        &kernel,
                        &z,
                        0.25,
                        inducing,
                        &options,
                        if parameter >= 2 {
                            Some((parameter - 2, transforms[parameter - 2]))
                        } else {
                            None
                        },
                        if parameter == 1 { 1.0 } else { 0.0 },
                        if parameter == 0 { 1.0 } else { 0.0 },
                        None,
                        None,
                    )
                    .unwrap();
                    assert!((&directional.mean - &expected.mean).amax() < 1e-9);
                    assert!((&directional.covariance - &expected.covariance).amax() < 1e-9);
                    let mut plus = state.clone();
                    let mut minus = state.clone();
                    let h = 1e-6;
                    plus[parameter] += h;
                    minus[parameter] -= h;
                    let p = run(&plus);
                    let m = run(&minus);
                    assert!(
                        ((p.mean - m.mean) / (2.0 * h) - directional.mean_tangent).amax() < 2e-5,
                        "mean inducing={inducing} prior={priors} parameter={parameter}"
                    );
                    assert!(
                        ((p.covariance - m.covariance) / (2.0 * h)
                            - directional.covariance_tangent)
                            .amax()
                            < 2e-5,
                        "cov inducing={inducing} prior={priors} parameter={parameter}"
                    );
                }
            }
        }
    }
}
