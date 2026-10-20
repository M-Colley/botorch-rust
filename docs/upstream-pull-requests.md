# Upstream correctness submissions

The user authorized upstream PR submission on 2026-10-20. Existing PRs from the
earlier audit are reused when they already address a finding. New patches are
tested against unchanged upstream before publication. An open PR is a proposed
repair, not an upstream release containing that repair.

BoTorch main remains `b0dd01be61d517c387e234903a6e2cc38c057ea2`.
New GPyTorch fixes target `e1302d24cbd334649ada70608cbe98e903a47a83`.

| Finding | Submission / remaining work |
| --- | --- |
| Batched LOO and fully Bayesian target layout | Existing [BoTorch #3386](https://github.com/meta-pytorch/botorch/pull/3386) |
| Shared SAC prior counted repeatedly | New [GPyTorch #2812](https://github.com/cornellius-gp/gpytorch/pull/2812) |
| SAC/LCEA dataset constructors omit optional noise | New [BoTorch #3472](https://github.com/meta-pytorch/botorch/pull/3472) |
| Latent Kronecker conditional sample covariance and outcome-space moments | Patch and upstream regressions pending |
| Batched Boltzmann normalization | Existing [BoTorch #3435](https://github.com/meta-pytorch/botorch/pull/3435) |
| Integrated white-noise cross diagonal | Existing [BoTorch #3387](https://github.com/meta-pytorch/botorch/pull/3387) |
| Invalid expectation weights | New [BoTorch #3467](https://github.com/meta-pytorch/botorch/pull/3467) |
| LCEA embeddings | Batch-table repair in existing [BoTorch #3387](https://github.com/meta-pytorch/botorch/pull/3387); declared feature order and categorical label mapping pending |
| Hierarchical category validation, transform fitting, task-column mapping and cycles | Patches and upstream regressions pending |
| Batched L-BFGS-B restored objective | New [BoTorch #3471](https://github.com/meta-pytorch/botorch/pull/3471) |
| SciPy L-BFGS-B restored objective | Tested [branch](https://github.com/M-Colley/scipy/compare/main...codex/lbfgsb-restored-objective); human submission required, see below |
| Heterogeneous multitask conditioning input layout | Patch and upstream regressions pending |
| Empirical multitask unseen-task noise | Existing [BoTorch #3389](https://github.com/meta-pytorch/botorch/pull/3389) |
| Seven sparse-outlier / relevance-pursuit defects | Patches and upstream regressions pending |
| Selecting one output from a multi-output GP | Existing [BoTorch #3396](https://github.com/meta-pytorch/botorch/pull/3396) |
| Additive component bias broadcasting and repeated outcome location | Patches and upstream regressions pending |
| Validated empty Beta expansion | New [BoTorch #3473](https://github.com/meta-pytorch/botorch/pull/3473) |
| Multi-axis fantasy observation noise | New [BoTorch #3476](https://github.com/meta-pytorch/botorch/pull/3476) |
| EM dataset subset indices | Existing [BoTorch #3388](https://github.com/meta-pytorch/botorch/pull/3388) |
| Constant EM fitting objective | New [BoTorch #3475](https://github.com/meta-pytorch/botorch/pull/3475), including scalar and batched closure regressions |
| Fully Bayesian known-noise prior trace | New [BoTorch #3477](https://github.com/meta-pytorch/botorch/pull/3477) |
| Fully Bayesian default multitask warp columns | Existing [BoTorch #3465](https://github.com/meta-pytorch/botorch/pull/3465) |
| Higher-order fantasy covariance wrappers | Patch and upstream regressions pending |
| Sequential acquisition dispatch and leaked pending state | New [BoTorch #3469](https://github.com/meta-pytorch/botorch/pull/3469) |
| All-fixed candidate feasibility | New [BoTorch #3470](https://github.com/meta-pytorch/botorch/pull/3470) |
| Correlated empirical fantasies | Existing [BoTorch #3388](https://github.com/meta-pytorch/botorch/pull/3388) |
| Empty data-loader closure hang | New [BoTorch #3466](https://github.com/meta-pytorch/botorch/pull/3466) |
| Learned-preference noisy-improvement cache law | Patch and upstream regressions pending |
| Categorical feature axis | Existing [BoTorch #3387](https://github.com/meta-pytorch/botorch/pull/3387) |
| Generic kernel diagonal ranks | New [GPyTorch #2815](https://github.com/cornellius-gp/gpytorch/pull/2815) |
| Composite lazy feature counts | New [GPyTorch #2818](https://github.com/cornellius-gp/gpytorch/pull/2818), including composite feature indexing |
| Model-list joint covariance blocks | Existing [BoTorch #3409](https://github.com/meta-pytorch/botorch/pull/3409) and [#3412](https://github.com/meta-pytorch/botorch/pull/3412); exact workaround coverage under audit |
| Variational unbatched zero-mean conversion | New [GPyTorch #2813](https://github.com/cornellius-gp/gpytorch/pull/2813) |
| Neural-kernel self-diagonal clipping convention | Paired-input repair exists in [BoTorch #3387](https://github.com/meta-pytorch/botorch/pull/3387); self-covariance consistency pending |
| Constant-target transform first/second derivatives | New [BoTorch #3468](https://github.com/meta-pytorch/botorch/pull/3468) |
| Periodic/cosine second derivatives | New [GPyTorch #2816](https://github.com/cornellius-gp/gpytorch/pull/2816) |
| Polynomial feature-batched offset | New [GPyTorch #2814](https://github.com/cornellius-gp/gpytorch/pull/2814) |
| Ensemble posterior standard deviation | Existing [BoTorch #3444](https://github.com/meta-pytorch/botorch/pull/3444) |
| Fully Bayesian explicit prediction-noise batch axis | New [BoTorch #3474](https://github.com/meta-pytorch/botorch/pull/3474) |
| Multitask kernel repeats existing task covariance batches | New [GPyTorch #2817](https://github.com/cornellius-gp/gpytorch/pull/2817) |
| GIBBON pending covariance units | Existing [BoTorch #3391](https://github.com/meta-pytorch/botorch/pull/3391) |
| GIBBON multiple query batch axes | Upstream patch and regressions pending |
| MES zero-variance control variate | Newly verified; upstream patch and regression pending |
| Scalar JES conditional normal standardization | New [BoTorch #3479](https://github.com/meta-pytorch/botorch/pull/3479); native CPU and GPU repaired with independent Gaussian regression |
| PES damping accepts failed first-pivot Cholesky | New [BoTorch #3480](https://github.com/meta-pytorch/botorch/pull/3480); native CPU and GPU regressions cover the repair |
| Additive MAP SAAS factory dtype/device | New [BoTorch #3478](https://github.com/meta-pytorch/botorch/pull/3478) |
| ZeroLinearOperator device/dtype and batched logdet | New [linear_operator #151](https://github.com/cornellius-gp/linear_operator/pull/151) |
| Multiobjective entropy at infinite integration bounds | GPU repaired with Gaussian-limit and derivative regressions; native and upstream repairs in progress |
| Port cosine float32 large-phase accuracy | Fixed in `4f1ad62`; CPU/CUDA regressions fail against old frozen binary and all 132 focused tensor cases pass with repair |

## Retracted report

The earlier rectangular-linear-root report was invalid: a direct
`RootLinearOperator(q, r)` advertises `r` base normals and samples correctly
when that advertised shape is supplied. It is not an upstream defect. The
port's square covariance root remains a valid independent law for numerical
comparisons. The incorrect finding and reproduction have been removed.

## SciPy human submission step

SciPy's [AI contribution policy](https://scipy.github.io/devdocs/dev/conduct/ai_policy.html)
requires a human to check generated code and submit the PR; it also requires
human-authored communication and an AI-generation disclosure. No autonomous
SciPy PR has been submitted.

The prepared branch targets main `bbc137a014ae67572c79b0ffd9ef493a278b6c24`.
Its Python-wrapper fix and eight parameterized regressions are committed as
`6a6995b` on `M-Colley/scipy:codex/lbfgsb-restored-objective`. The ten tests in
the affected test module pass when loading the latest Python wrapper against
the installed SciPy 1.18 LP64 extension. Five new cases fail with the original
latest wrapper. The harness supplies the new main-branch `HAS_ILP64=False`
configuration flag for that older LP64 build. No native C code was changed or
rebuilt. Ruff with SciPy's own lint configuration and `git diff --check` pass.
