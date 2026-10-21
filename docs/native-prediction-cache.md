# Native prediction cache reuse

The native exact-GP backend now shares an immutable lazy prediction cache
between a model and its value snapshots. This repairs repeated training
factorization during ordinary `model.posterior(X)` calls. It does not change
the posterior law, factorization algorithm, or triangular-solve arithmetic.

Previously, each public posterior copied a model whose cache was empty. The
posterior initialized only its own copy, leaving the original model empty;
the next public call therefore factored the training covariance again.
Copying an explicitly warmed original also copied its dense factor. The cache
is now an `Arc<OnceLock<Result<PredictionCache, BotorchError>>>`, so snapshots
reuse both lazy initialization and its immutable result. Parameter, sparse
noise, target/noise replacement, and SAAS-prior setters install a fresh cache
on the modified model. Conditioned and noiseless fantasy models own separate
enlarged caches. Existing snapshots keep their original data, parameters, and
factor after later mutations.

## Controlled production-build comparison

These measurements use an Intel Core Ultra 7 155H on Windows, CPU float64,
Torch 2.13, and one Torch/OMP/MKL/Rayon thread. Both implementations use the
same training data, query data, mean `0.1`, learned noise `0.02`, ARD RBF
lengthscales `0.31`, and no outcome transform. Queries contain 16 batches of
two points in four dimensions. Parameters are frozen and prediction runs
under `torch.no_grad()`. The reference is official BoTorch commit
`b0dd01be61d517c387e234903a6e2cc38c057ea2`.

Both native artifacts were built with the production release profile. The
preserved previous artifact has SHA256
`e8adc00da79eed53dd1260aa3c54574369ffb979b6d55b7ff81c6e1597e083cf`;
the candidate has SHA256
`3b7c95aa07214afaf759badf111dec3b8a781e476fedd1006c65e5782709f04f`.
Each artifact ran in a separate process using the same committed benchmark
script and current Python bridge. Native/reference public calls were shuffled
and interleaved after five warmups, with 21 timings per implementation. Other
agents held tests and builds during both runs. The benchmark verifies mean
and variance at `atol=rtol=1e-9` before timing; the largest observed absolute
errors were `6.33e-15` and `2.66e-15` respectively.

| Training points | Previous native median | Candidate native median | BoTorch median in candidate run | Previous / candidate |
| --- | ---: | ---: | ---: | ---: |
| 32 | 0.1754 ms | 0.1674 ms | 0.8204 ms | 1.05x |
| 256 | 1.0754 ms | 0.3916 ms | 1.0975 ms | 2.75x |
| 1024 | 50.5029 ms | 4.3047 ms | 3.6698 ms | 11.73x |

At 1024 training points the candidate remains **17.3% slower than BoTorch**
for this public prediction workload. It is faster for the two smaller cases.
The first native prediction is unchanged: 48.36 ms before and 48.56 ms after
at 1024 points. A populated native model copy dropped from 1.4544 ms to
0.0023 ms. Raw samples and component probes are preserved in
[the previous-build report](benchmarks/native-prediction-cache-before-2026-10-21.json)
and [the candidate report](benchmarks/native-prediction-cache-after-2026-10-21.json).

The separate first-call observations include one-time library initialization;
the first BoTorch case is particularly affected. They are not repeated cold
training benchmarks and do not establish a cold-prediction speedup. Earlier
[October 20 results](benchmarks/tensor-main-prediction-2026-10-20-float64.json)
remain historical observations: their description says the first training
factor is excluded, but this cache defect meant repeated native public calls
still paid for factorization. Their measured timings should not be reinterpreted
as timings from a correctly warmed native cache.

## Validation and limits

Five new Rust tests check shared initialization, concurrent snapshots,
mutation invalidation, fixed-noise/target replacement, and conditioned-model
isolation. Eighteen new Python cases cover list, NumPy, and Torch callers;
single and multiple outputs; model batches; learned and fixed noise; eager
and lazy posteriors; and native copied handles after public parameter changes.
The actual candidate extension passed 161 focused Python cases including
existing query/parameter gradient and projection tests. The complete native
workspace passed 546 tests across 46 result groups, with no failures or ignored
tests. Workspace Clippy with warnings denied and `cargo fmt --check` passed.
[The validation receipt](benchmarks/native-prediction-cache-validation-2026-10-21.json)
records the artifact, source, build, and log hashes.

This repair retains an `O(n^2)` cache while its model or snapshots remain
alive. Initial factorization is still `O(n^3)`; marginal query solves remain
`O(n^2 q)`. It does not introduce packed right-hand sides or change the
native CPU backend into a GPU implementation. The existing Tensor backend
provides device-preserving GPU execution separately. These results apply to
the measured kernel, shapes, hardware, and release builds; they do not establish
a universal performance advantage.

Reproduce with `scripts/benchmark_native_prediction.py`, supplying
`--native-extension`, `--reference-root`, and `--output` for each artifact.
The script records both artifact hashes, source/reference revisions, exact
hyperparameters, numerical errors, and every timing sample.
