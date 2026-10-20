//! Candidate-member selection scoped to a native acquisition worker.

use std::cell::RefCell;

type BatchContext = (Vec<usize>, Option<usize>);

thread_local! {
    static CURRENT: RefCell<Option<BatchContext>> = const { RefCell::new(None) };
}

struct Restore(Option<BatchContext>);

impl Drop for Restore {
    fn drop(&mut self) {
        CURRENT.with(|context| {
            context.replace(self.0.take());
        });
    }
}

pub(crate) fn current() -> Option<BatchContext> {
    CURRENT.with(|context| context.borrow().clone())
}

/// `index=None` evaluates all preference members for a common pruning set.
/// A candidate worker supplies its row-major batch index instead.
pub(crate) fn with_context<T>(batch: &[usize], index: Option<usize>, f: impl FnOnce() -> T) -> T {
    let previous = CURRENT.with(|context| context.replace(Some((batch.to_vec(), index))));
    let _restore = Restore(previous);
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn member_context_restores_after_nested_errors_and_unwinding() {
        assert!(current().is_none());
        with_context(&[2, 3], Some(4), || {
            assert_eq!(current(), Some((vec![2, 3], Some(4))));
            let error: Result<(), ()> = with_context(&[], None, || Err(()));
            assert!(error.is_err());
            assert_eq!(current(), Some((vec![2, 3], Some(4))));
            let panic = std::panic::catch_unwind(|| {
                with_context(&[5], Some(2), || panic!("restore test"));
            });
            assert!(panic.is_err());
            assert_eq!(current(), Some((vec![2, 3], Some(4))));
        });
        assert!(current().is_none());
    }
}
