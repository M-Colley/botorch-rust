# IID sampler equivalence fixture

The IID sampler sweep now draws candidates with seed `1000 + seed`, separately
from the training-data generator. It compares every acquisition value directly
at relative tolerance `1e-9` and its query gradient at `1e-8`.

The previous seed-5 fixture reset `Generator(5)` for both training points and
candidates. All six candidate points therefore repeated baseline points
exactly. Its joint latent covariance had three zero eigenvalues per batch;
float64 factorization rounded the smallest eigenvalues to about `-6.8e-18`.
For qNEI these duplicated candidates offer zero improvement over the baseline
before numerical stabilization. Cholesky jitter instead produced small,
platform-dependent positive values.

Merged-main CI run
[39686129040](https://github.com/M-Colley/botorch-rust/actions/runs/39686129040)
exposed this in `test_equivalence_11a_utils.py::test_iid_sampler_sweep[5]`:
the training-perturbation twin envelope reported scaled excess `1.4588857161`.
Its broad tests passed; its equivalence step also had a separate preference
fixture failure. This failed run remains separate from earlier successful
verification and publication receipts.

A bounded check used the published Windows 0.3.0 native binary, pinned BoTorch
`b0dd01be61d517c387e234903a6e2cc38c057ea2`, GPyTorch 1.15.2, and Torch
2.14.1+cpu. The actual native and source IID normal draws agreed exactly.
Posterior covariances differed by at most `9.54e-17`. Source-only training
perturbations at relative scales `1e-14` through `1e-11` changed the seed-5
qNEI value from `5.67e-10` to `6.08e-5`. These observations identify the
singular fixture's stabilization sensitivity; no sampler runtime change is
required by this failure.

With independent candidates, all twelve IID seeds passed direct comparisons
on the same CPU environment. Seed-5 qNEI relative errors were `1.49e-12` for
values and `1.13e-13` for gradients; seed-6 qLogNEI errors were `2.28e-15`
and `4.03e-14`. The sweep's noisy acquisitions now require disjoint joint
query points and a smallest covariance eigenvalue greater than
`64 * eps * largest_eigenvalue`, without adding jitter in the guard.

The focused regressions reject duplicate and nearly duplicate joint queries,
verify exact normal draws, and detect deliberately corrupted values,
gradients, and sampler seeds. Shared twin-envelope settings are unchanged.
All twelve sweep cases and both guard cases passed with Torch 2.14.1+cpu
and with the base Torch 2.13.0 environment using the published package.

Raw bounded evidence is retained locally:

- `D:/botorch-rs-build/iid-seed5-cpu214-unchanged.log`
- `D:/botorch-rs-build/iid-seed5-cpu214-probe.json`
- `D:/botorch-rs-build/iid-seed5-cpu214-probe.log`
- `D:/botorch-rs-build/iid-seed5-independent-cpu214.json`
- `D:/botorch-rs-build/iid-seed5-independent-cpu214.log`
- `D:/botorch-rs-build/iid-seed5-cpu214-fixed.log`
- `D:/botorch-rs-build/iid-seed5-base213-fixed.log`
- `D:/botorch-release-030/merged-main-ci-evidence/job-logs/job-119520517608.log`

The failed CI log's SHA256 is
`cd8deab0b026d4f5b6fbb53a46134eaa7f4733dcd1cb14b9f4865b9724c94b2b`.
