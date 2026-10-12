//! The task handling of BoTorch's `MultiTaskGP` (`botorch.models.multitask`),
//! shared by its subclass [`crate::models::LCEMGP`]: the task feature (a
//! column of the training inputs), the task values inferred from the data or
//! given as `all_tasks` (which may include tasks without observations), the
//! output tasks, and the mapping of raw task values to task indices with
//! BoTorch's validation (`_map_tasks` with `validate_task_values=True`).
//!
//! Task values are integers; like torch's `.long()`, values in the task
//! column are truncated towards zero. Task indices are the positions of the
//! task values in the sorted `all_tasks`.

use crate::error::{BotorchError, Result};
use crate::Matrix;

/// The task structure of a multi-task model (see the [module
/// documentation](self)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskLayout {
    task_feature: usize,
    num_non_task_features: usize,
    all_tasks: Vec<i64>,
    output_tasks: Vec<i64>,
    observed: Vec<bool>,
}

/// torch's `.long()` of a task value (truncation towards zero).
fn task_value(v: f64) -> Result<i64> {
    if !v.is_finite() {
        return Err(BotorchError::InputData(format!(
            "task features must be finite, got {v}"
        )));
    }
    Ok(v.trunc() as i64)
}

impl TaskLayout {
    /// BoTorch's `MultiTaskGP.get_all_tasks` and the task checks of
    /// `MultiTaskGP.__init__` for training inputs `train_x` (`n × (d + 1)`)
    /// with the task values in column `task_feature` (negative values count
    /// from the end, `-d <= task_feature <= d` after BoTorch's shift).
    ///
    /// `all_tasks` (BoTorch's `all_tasks`; `None` or empty: the tasks of the
    /// data) must contain every task of the data; `output_tasks` (`None`: all
    /// tasks, sorted) must be a subset of the tasks.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] with BoTorch's messages for an
    /// invalid `task_feature`, [`BotorchError::Unsupported`] if `all_tasks`
    /// misses a task of the data, [`BotorchError::Runtime`] for output tasks
    /// that are not tasks, and [`BotorchError::InputData`] for non-finite
    /// task values or duplicates in `all_tasks`.
    pub fn new(
        train_x: &Matrix,
        task_feature: isize,
        output_tasks: Option<&[i64]>,
        all_tasks: Option<&[i64]>,
    ) -> Result<Self> {
        let cols = train_x.ncols();
        if cols < 2 {
            return Err(BotorchError::DimensionMismatch(format!(
                "train_X must have the task feature and at least one other column, got {cols} \
                 columns"
            )));
        }
        let d = cols - 1;
        // MultiTaskGP shifts negative indices by the number of columns, then
        // get_all_tasks checks the range and takes the index modulo d + 1.
        let mut tf = task_feature;
        if tf < 0 {
            tf += cols as isize;
        }
        let di = d as isize;
        if !(-di <= tf && tf <= di) {
            return Err(BotorchError::InvalidArgument(format!(
                "Must have that -{d} <= task_feature <= {d}"
            )));
        }
        let tf = tf.rem_euclid(cols as isize) as usize;
        let mut inferred = train_x
            .column(tf)
            .iter()
            .map(|&v| task_value(v))
            .collect::<Result<Vec<_>>>()?;
        inferred.sort_unstable();
        inferred.dedup();
        let all = match all_tasks {
            Some(given) if !given.is_empty() => {
                let mut sorted = given.to_vec();
                sorted.sort_unstable();
                if sorted.windows(2).any(|w| w[0] == w[1]) {
                    return Err(BotorchError::InputData(format!(
                        "all_tasks contains duplicate task values: {given:?}"
                    )));
                }
                if inferred.iter().any(|t| sorted.binary_search(t).is_err()) {
                    return Err(BotorchError::Unsupported(format!(
                        "The provided all_tasks={given:?} does not contain all the task \
                         features inferred from the training data \
                         all_tasks_inferred={inferred:?}. This is not allowed as it will lead \
                         to errors during model training."
                    )));
                }
                sorted
            }
            _ => inferred.clone(),
        };
        let outputs = match output_tasks {
            None => all.clone(),
            Some(o) => {
                if o.iter().any(|t| all.binary_search(t).is_err()) {
                    return Err(BotorchError::Runtime(
                        "All output tasks must be present in input data.".into(),
                    ));
                }
                o.to_vec()
            }
        };
        let observed = all
            .iter()
            .map(|t| inferred.binary_search(t).is_ok())
            .collect();
        Ok(Self {
            task_feature: tf,
            num_non_task_features: d,
            all_tasks: all,
            output_tasks: outputs,
            observed,
        })
    }

    /// The column of the task feature (non-negative).
    pub fn task_feature(&self) -> usize {
        self.task_feature
    }

    /// The number of non-task features `d`.
    pub fn num_non_task_features(&self) -> usize {
        self.num_non_task_features
    }

    /// The sorted task values.
    pub fn all_tasks(&self) -> &[i64] {
        &self.all_tasks
    }

    /// The number of tasks.
    pub fn num_tasks(&self) -> usize {
        self.all_tasks.len()
    }

    /// The output tasks (task values, in the given order).
    pub fn output_tasks(&self) -> &[i64] {
        &self.output_tasks
    }

    /// Whether every task value is its own index (BoTorch's task mapper is
    /// `None`).
    pub fn is_identity(&self) -> bool {
        self.all_tasks
            .iter()
            .enumerate()
            .all(|(i, &t)| t == i as i64)
    }

    /// Whether task index `k` has training data.
    pub fn is_observed(&self, k: usize) -> bool {
        self.observed[k]
    }

    /// A copy with other output tasks (task values).
    ///
    /// # Errors
    /// [`BotorchError::Runtime`] if a value is not a task.
    pub fn with_output_tasks(&self, output_tasks: &[i64]) -> Result<Self> {
        if output_tasks.is_empty() {
            return Err(BotorchError::InvalidArgument(
                "at least one output task is required".into(),
            ));
        }
        if output_tasks
            .iter()
            .any(|t| self.all_tasks.binary_search(t).is_err())
        {
            return Err(BotorchError::Runtime(
                "All output tasks must be present in input data.".into(),
            ));
        }
        Ok(Self {
            output_tasks: output_tasks.to_vec(),
            ..self.clone()
        })
    }

    /// The task index of a raw task value, validated as BoTorch's
    /// `_map_tasks` with `validate_task_values=True` (with BoTorch's error
    /// messages).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for values that are not tasks.
    pub fn task_index(&self, value: f64) -> Result<usize> {
        let v = task_value(value)?;
        if self.is_identity() {
            if !(value >= 0.0 && value < self.num_tasks() as f64) {
                return Err(BotorchError::InvalidArgument(format!(
                    "Expected all task features in `X` to be between 0 and self.num_tasks - 1. \
                     Got {value}."
                )));
            }
            return Ok(v as usize);
        }
        self.all_tasks.binary_search(&v).map_err(|_| {
            BotorchError::InvalidArgument(format!(
                "Received invalid raw task values. Expected raw value to be in {:?}, but got \
                 unexpected task values: {{{v}}}.",
                self.all_tasks
            ))
        })
    }

    /// The task indices of the output tasks.
    pub fn output_task_indices(&self) -> Vec<usize> {
        self.output_tasks
            .iter()
            .map(|t| {
                self.all_tasks
                    .binary_search(t)
                    .expect("output tasks are tasks")
            })
            .collect()
    }

    /// Splits full inputs (`n × (d + 1)`) into the non-task features
    /// (`n × d`, BoTorch's `x_before` and `x_after` concatenated) and the task
    /// indices.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns and
    /// the errors of [`TaskLayout::task_index`].
    pub fn split_inputs(&self, x: &Matrix) -> Result<(Matrix, Vec<usize>)> {
        let d = self.num_non_task_features;
        if x.ncols() != d + 1 {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected inputs with {} columns (including the task feature), got {}",
                d + 1,
                x.ncols()
            )));
        }
        let tf = self.task_feature;
        let tasks = x
            .column(tf)
            .iter()
            .map(|&v| self.task_index(v))
            .collect::<Result<Vec<_>>>()?;
        let data = Matrix::from_fn(x.nrows(), d, |i, j| x[(i, if j < tf { j } else { j + 1 })]);
        Ok((data, tasks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(tasks: &[f64]) -> Matrix {
        Matrix::from_fn(tasks.len(), 3, |i, j| {
            if j == 1 {
                tasks[i]
            } else {
                0.1 * (i + j) as f64
            }
        })
    }

    #[test]
    fn infers_and_extends_tasks() {
        let x = data(&[0.0, 2.0, 2.0, 0.0]);
        let t = TaskLayout::new(&x, 1, None, None).unwrap();
        assert_eq!(t.all_tasks(), &[0, 2]);
        assert!(!t.is_identity());
        assert_eq!(t.output_tasks(), &[0, 2]);
        let t = TaskLayout::new(&x, -2, Some(&[1]), Some(&[2, 1, 0])).unwrap();
        assert_eq!(t.task_feature(), 1);
        assert_eq!(t.all_tasks(), &[0, 1, 2]);
        assert!(t.is_identity());
        assert_eq!(t.output_task_indices(), vec![1]);
        assert!(!t.is_observed(1) && t.is_observed(2));
        let (xd, idx) = t.split_inputs(&x).unwrap();
        assert_eq!(idx, vec![0, 2, 2, 0]);
        assert_eq!(xd[(1, 1)], x[(1, 2)]);
        assert!(t.task_index(3.0).is_err());
    }

    #[test]
    fn rejects_invalid_task_configurations() {
        let x = data(&[0.0, 1.0]);
        assert!(matches!(
            TaskLayout::new(&x, 5, None, None),
            Err(BotorchError::InvalidArgument(_))
        ));
        assert!(matches!(
            TaskLayout::new(&x, 1, None, Some(&[0])),
            Err(BotorchError::Unsupported(_))
        ));
        assert!(matches!(
            TaskLayout::new(&x, 1, Some(&[4]), None),
            Err(BotorchError::Runtime(_))
        ));
        let t = TaskLayout::new(&data(&[3.0, 7.0]), 1, None, None).unwrap();
        assert_eq!(t.task_index(7.0).unwrap(), 1);
        assert!(t.task_index(5.0).is_err());
    }
}
