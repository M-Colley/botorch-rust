//! Compile-time checks that the types shared across threads (and, in phase 3,
//! with Python) are `Send + Sync`.

use std::sync::Arc;

use botorch_core::acquisition::{
    AcquisitionFunction, ExpectedImprovement, LogExpectedImprovement, LogProbabilityOfImprovement,
    PosteriorMean, PosteriorStandardDeviation, ProbabilityOfImprovement, QExpectedImprovement,
    QLogExpectedImprovement, QLogNoisyExpectedImprovement, QProbabilityOfImprovement,
    QSimpleRegret, QUpperConfidenceBound, UpperConfidenceBound,
};
use botorch_core::optim::OptimizeAcqfOptions;
use botorch_core::sampling::SobolQMCNormalSampler;
use botorch_core::test_functions::SyntheticTestFunction;
use botorch_core::{Model, SingleTaskGP};

fn assert_send_sync<T: Send + Sync + ?Sized>() {}

#[test]
fn public_types_are_send_and_sync() {
    assert_send_sync::<SingleTaskGP>();
    assert_send_sync::<dyn Model>();
    assert_send_sync::<Arc<dyn Model>>();
    assert_send_sync::<dyn AcquisitionFunction>();
    assert_send_sync::<ExpectedImprovement>();
    assert_send_sync::<LogExpectedImprovement>();
    assert_send_sync::<ProbabilityOfImprovement>();
    assert_send_sync::<LogProbabilityOfImprovement>();
    assert_send_sync::<UpperConfidenceBound>();
    assert_send_sync::<PosteriorMean>();
    assert_send_sync::<PosteriorStandardDeviation>();
    assert_send_sync::<QExpectedImprovement>();
    assert_send_sync::<QProbabilityOfImprovement>();
    assert_send_sync::<QUpperConfidenceBound>();
    assert_send_sync::<QSimpleRegret>();
    assert_send_sync::<QLogExpectedImprovement>();
    assert_send_sync::<QLogNoisyExpectedImprovement>();
    assert_send_sync::<SobolQMCNormalSampler>();
    assert_send_sync::<SyntheticTestFunction>();
    assert_send_sync::<OptimizeAcqfOptions>();
}
