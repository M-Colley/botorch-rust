//! Sequential optimization of a list of acquisition functions
//! (`botorch.optim.optimize.optimize_acqf_list`), e.g. one qLogNParEGO per
//! random scalarization: the acquisition functions are optimized one after
//! the other for `q = 1`, each with the previous candidates as pending
//! points.

use std::collections::BTreeMap;

use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::optim::optimize::{optimize_acqf_detailed, OptimizeAcqfOptions};
use crate::Matrix;

/// The result of [`optimize_acqf_list`].
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizeAcqfListResult {
    /// The candidates (`len × d`), one per acquisition function.
    pub candidates: Matrix,
    /// The acquisition value of every candidate (conditional on the previous
    /// candidates as pending points).
    pub acq_values: Vec<f64>,
    /// Messages corresponding to BoTorch's warnings.
    pub warnings: Vec<String>,
}

/// BoTorch's `optimize_acqf_mixed` for `q = 1`: optimizes `acq_function` with
/// each of the `fixed_features_list` and returns the best restart over all
/// of them (`CandidateGenerationError`s of single fixed features are
/// skipped with a warning).
fn optimize_mixed_q1(
    acq_function: &dyn AcquisitionFunction,
    bounds: &Matrix,
    num_restarts: usize,
    raw_samples: usize,
    options: &OptimizeAcqfOptions,
    fixed_features_list: &[BTreeMap<usize, f64>],
    warnings: &mut Vec<String>,
) -> Result<(Matrix, f64)> {
    let mut best: Option<(Matrix, f64)> = None;
    let mut failures = 0;
    let mut succeeded = 0;
    for fixed in fixed_features_list {
        let opts = OptimizeAcqfOptions {
            fixed_features: fixed.clone(),
            sequential: false,
            ..options.clone()
        };
        let r =
            match optimize_acqf_detailed(acq_function, bounds, 1, num_restarts, raw_samples, &opts)
            {
                Ok(r) => r,
                Err(BotorchError::CandidateGeneration(_)) => {
                    failures += 1;
                    continue;
                }
                Err(e) => return Err(e),
            };
        succeeded += 1;
        warnings.extend(r.warnings.iter().cloned());
        // The best restart of these fixed features (`torch.max(dim=-1)`),
        // then the best fixed features (`torch.argmax`, first on ties).
        let mut restarts: Vec<(Matrix, f64)> = if r.restarts.is_empty() {
            vec![(r.candidates.clone(), r.acq_value)]
        } else {
            r.restarts
                .iter()
                .map(|x| (x.candidates.clone(), x.acq_value))
                .collect()
        };
        let mut top = 0;
        for (i, (_, v)) in restarts.iter().enumerate() {
            if *v > restarts[top].1 || (v.is_nan() && !restarts[top].1.is_nan()) {
                top = i;
            }
        }
        let cand = restarts.swap_remove(top);
        match &best {
            Some((_, v)) if !(cand.1 > *v || (cand.1.is_nan() && !v.is_nan())) => {}
            _ => best = Some(cand),
        }
    }
    if succeeded == 0 {
        return Err(BotorchError::CandidateGeneration(
            "Candidate generation failed for all `fixed_features`.".into(),
        ));
    }
    if failures > 0 {
        warnings.push(format!(
            "Candidate generation failed for {failures} combinations of `fixed_features`. To \
             suppress this warning, make sure all equality/inequality constraints can be \
             satisfied by all `fixed_features` in `fixed_features_list`."
        ));
    }
    Ok(best.expect("at least one fixed feature set succeeded"))
}

/// Generates one candidate per acquisition function (BoTorch's
/// `optimize_acqf_list`): the acquisition functions are optimized in
/// sequence for `q = 1` with [`optimize_acqf_detailed`] (`options` apply to
/// every optimization, with `sequential = false`), each after setting the
/// previous candidates (appended to the first acquisition function's
/// pending points) as its pending points. With `fixed_features_list`, every
/// optimization is BoTorch's `optimize_acqf_mixed` over these fixed
/// features. The pending points are left as set (as in BoTorch).
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for an empty list or for both
/// `options.fixed_features` and a `fixed_features_list` (BoTorch's
/// messages), [`BotorchError::CandidateGeneration`] if the optimization
/// failed for all fixed features, and the errors of [`optimize_acqf_detailed`]
/// and of the acquisition functions.
pub fn optimize_acqf_list(
    acq_function_list: &[&dyn AcquisitionFunction],
    bounds: &Matrix,
    num_restarts: usize,
    raw_samples: usize,
    options: &OptimizeAcqfOptions,
    fixed_features_list: Option<&[BTreeMap<usize, f64>]>,
) -> Result<OptimizeAcqfListResult> {
    let fixed_features_list = fixed_features_list.filter(|l| !l.is_empty());
    if !options.fixed_features.is_empty() && fixed_features_list.is_some() {
        return Err(BotorchError::InvalidArgument(
            "Èither `fixed_feature` or `fixed_features_list` can be provided, not both.".into(),
        ));
    }
    let Some(first) = acq_function_list.first() else {
        return Err(BotorchError::InvalidArgument(
            "acq_function_list must be non-empty.".into(),
        ));
    };
    let d = bounds.ncols();
    let base_pending = first.x_pending();
    let mut candidates: Vec<Matrix> = Vec::with_capacity(acq_function_list.len());
    let mut values = Vec::with_capacity(acq_function_list.len());
    let mut warnings = Vec::new();
    let joint = OptimizeAcqfOptions {
        sequential: false,
        ..options.clone()
    };
    for acq in acq_function_list {
        if !candidates.is_empty() {
            let mut parts: Vec<&Matrix> = Vec::new();
            if let Some(p) = &base_pending {
                parts.push(p);
            }
            parts.extend(candidates.iter());
            let pending = crate::acquisition::monte_carlo::vstack(&parts, d);
            acq.set_x_pending(Some(&pending))?;
        }
        let (candidate, value) = match fixed_features_list {
            Some(list) => optimize_mixed_q1(
                *acq,
                bounds,
                num_restarts,
                raw_samples,
                &joint,
                list,
                &mut warnings,
            )?,
            None => {
                let r = optimize_acqf_detailed(*acq, bounds, 1, num_restarts, raw_samples, &joint)?;
                warnings.extend(r.warnings);
                (r.candidates, r.acq_value)
            }
        };
        candidates.push(candidate);
        values.push(value);
    }
    let parts: Vec<&Matrix> = candidates.iter().collect();
    Ok(OptimizeAcqfListResult {
        candidates: crate::acquisition::monte_carlo::vstack(&parts, d),
        acq_values: values,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::acquisition::{
        LogExpectedImprovement, QLogExpectedImprovement, QLogExpectedImprovementOptions,
    };
    use crate::models::{Model, SingleTaskGP, SingleTaskGPOptions};
    use crate::sampling::SobolQMCNormalSampler;

    fn model() -> Arc<dyn Model> {
        let x = Matrix::from_fn(8, 2, |i, j| ((i * 5 + j * 3) % 8) as f64 / 7.0);
        let y = Matrix::from_fn(8, 1, |i, _| (4.0 * x[(i, 0)]).sin() + x[(i, 1)]);
        let mut gp = SingleTaskGP::new(x, y, SingleTaskGPOptions::default()).unwrap();
        gp.set_raw_parameters(&[0.01, 0.0, 0.4, 0.5]).unwrap();
        Arc::new(gp)
    }

    #[test]
    fn sequential_candidates_with_pending_points() {
        let m = model();
        let bounds = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        let make = || {
            QLogExpectedImprovement::new(
                m.clone(),
                1.5,
                QLogExpectedImprovementOptions {
                    sampler: Some(Arc::new(SobolQMCNormalSampler::new(64, Some(1)).unwrap())),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let (a, b) = (make(), make());
        let options = OptimizeAcqfOptions {
            seed: Some(3),
            ..Default::default()
        };
        let r = optimize_acqf_list(&[&a, &b], &bounds, 4, 64, &options, None).unwrap();
        assert_eq!(r.candidates.shape(), (2, 2));
        assert_eq!(r.acq_values.len(), 2);
        // The second acquisition function has the first candidate pending.
        assert_eq!(b.x_pending().unwrap(), r.candidates.rows(0, 1).into_owned());
        assert!(a.x_pending().is_none());
        // Fixed features: the best over the list.
        let ei = LogExpectedImprovement::new(m.clone(), 1.5, true).unwrap();
        let list = vec![
            BTreeMap::from([(0usize, 0.1)]),
            BTreeMap::from([(0usize, 0.9)]),
        ];
        let r = optimize_acqf_list(&[&ei], &bounds, 4, 64, &options, Some(&list)).unwrap();
        let v0 = ei.forward(&Matrix::from_row_slice(1, 2, &[0.1, r.candidates[(0, 1)]]));
        assert!(r.candidates[(0, 0)] == 0.1 || r.candidates[(0, 0)] == 0.9);
        assert!(v0.is_ok());
        // Errors with BoTorch's messages.
        let e = optimize_acqf_list(&[], &bounds, 4, 64, &options, None).unwrap_err();
        assert!(e.to_string().contains("must be non-empty"));
        let both = OptimizeAcqfOptions {
            fixed_features: BTreeMap::from([(1usize, 0.5)]),
            ..options.clone()
        };
        let e = optimize_acqf_list(&[&ei], &bounds, 4, 64, &both, Some(&list)).unwrap_err();
        assert!(e.to_string().contains("not both"));
    }
}
