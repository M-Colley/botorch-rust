# Upstream correctness submissions

The user authorized upstream PR submission on 2026-10-20. Existing PRs from the
earlier audit are reused when they already address a finding. New patches are
tested against unchanged upstream before publication. An open PR is a proposed
repair, not an upstream release containing that repair.

BoTorch main remains `b0dd01be61d517c387e234903a6e2cc38c057ea2`.
New GPyTorch fixes target `e1302d24cbd334649ada70608cbe98e903a47a83`.

| Finding | Submission / remaining work |
| --- | --- |
| Batched inducing allocation reuses the first model covariance | New [BoTorch #3511](https://github.com/meta-pytorch/botorch/pull/3511); allocator/approximate suites pass 27 tests and 14 subtests on Torch 2.13 CPU/CUDA and 2.14.1 CPU |
| Batched LOO and fully Bayesian target layout | Existing [BoTorch #3386](https://github.com/meta-pytorch/botorch/pull/3386) |
| Shared SAC prior counted repeatedly | New [GPyTorch #2812](https://github.com/cornellius-gp/gpytorch/pull/2812) |
| SAC structural indices and device metadata after module migration | New [BoTorch #3493](https://github.com/meta-pytorch/botorch/pull/3493); 10 source tests and two migration subtests pass |
| SAC/LCEA dataset constructors omit optional noise | New [BoTorch #3472](https://github.com/meta-pytorch/botorch/pull/3472) |
| Latent Kronecker sample covariance, outcome-space moments, time batches and composite factors | New [BoTorch #3488](https://github.com/meta-pytorch/botorch/pull/3488) |
| Batched Boltzmann normalization | Existing [BoTorch #3435](https://github.com/meta-pytorch/botorch/pull/3435) |
| Integrated white-noise cross diagonal | Existing [BoTorch #3387](https://github.com/meta-pytorch/botorch/pull/3387) |
| Integrated Wiener coincident smooth Hessians | New [BoTorch #3496](https://github.com/meta-pytorch/botorch/pull/3496); four source tests and 18 CPU/CUDA subtests pass |
| Invalid expectation weights | New [BoTorch #3467](https://github.com/meta-pytorch/botorch/pull/3467) |
| LCEA embeddings | Batch-table repair in existing [BoTorch #3387](https://github.com/meta-pytorch/botorch/pull/3387); declared feature order and label mapping in new [#3487](https://github.com/meta-pytorch/botorch/pull/3487) |
| Hierarchical category validation, transform fitting, task-column mapping and cycles | New [BoTorch #3490](https://github.com/meta-pytorch/botorch/pull/3490); 19 tests and 22 subtests pass |
| Batched L-BFGS-B restored objective | New [BoTorch #3471](https://github.com/meta-pytorch/botorch/pull/3471) |
| SciPy L-BFGS-B restored objective | Tested [branch](https://github.com/M-Colley/scipy/compare/main...codex/lbfgsb-restored-objective); human submission required, see below |
| Heterogeneous multitask conditioning input layout | New [BoTorch #3484](https://github.com/meta-pytorch/botorch/pull/3484) |
| Empirical multitask unseen-task noise | Existing [BoTorch #3389](https://github.com/meta-pytorch/botorch/pull/3389) |
| Sparse-outlier / relevance-pursuit numerical and state defects | New [BoTorch #3482](https://github.com/meta-pytorch/botorch/pull/3482), including gradient-probe rollback; 43 tests and 120 subtests pass |
| Selecting one output from a multi-output GP | Existing [BoTorch #3396](https://github.com/meta-pytorch/botorch/pull/3396) |
| Additive component bias broadcasting and repeated outcome location | New [BoTorch #3486](https://github.com/meta-pytorch/botorch/pull/3486) |
| Orthogonal additive quadrature/interaction buffers after module migration | New [BoTorch #3495](https://github.com/meta-pytorch/botorch/pull/3495); two source tests and ten subtests pass |
| Validated empty Beta expansion | New [BoTorch #3473](https://github.com/meta-pytorch/botorch/pull/3473) |
| Multi-axis fantasy observation noise | New [BoTorch #3476](https://github.com/meta-pytorch/botorch/pull/3476) |
| EM dataset subset indices | Existing [BoTorch #3388](https://github.com/meta-pytorch/botorch/pull/3388) |
| EM Gaussian mean-prior location and shared-model target units | Existing [BoTorch #3388](https://github.com/meta-pytorch/botorch/pull/3388); both port backends now use the corrected mean prior and raw shared target law |
| Constant EM fitting objective | New [BoTorch #3475](https://github.com/meta-pytorch/botorch/pull/3475), including scalar and batched closure regressions |
| Fully Bayesian known-noise prior trace | New [BoTorch #3477](https://github.com/meta-pytorch/botorch/pull/3477) |
| Fully Bayesian default multitask warp columns | Existing [BoTorch #3465](https://github.com/meta-pytorch/botorch/pull/3465) |
| Higher-order fantasy covariance wrappers and sample axes | New [BoTorch #3491](https://github.com/meta-pytorch/botorch/pull/3491); 19 tests and 12 subtests pass |
| Sequential acquisition dispatch and leaked pending state | New [BoTorch #3469](https://github.com/meta-pytorch/botorch/pull/3469) |
| Graph-attached Torch candidate restarts are not optimizer leaves | New [BoTorch #3502](https://github.com/meta-pytorch/botorch/pull/3502); six source tests and 46 subtests pass |
| Homotopy failures leave schedule/pending state behind | New [BoTorch #3503](https://github.com/meta-pytorch/botorch/pull/3503); 14 source tests and six subtests pass |
| Acquisition-list own pending baseline and list/mixed-alternating exception cleanup | Existing [BoTorch #3395](https://github.com/meta-pytorch/botorch/pull/3395) extended with `finally` rollback; nine integrated list/mixed tests and six exception subtests pass |
| Fixed-feature values follow unsorted caller columns | Existing [BoTorch #3391](https://github.com/meta-pytorch/botorch/pull/3391); both port backends corrected and 14 wrapper tests pass |
| All-fixed candidate feasibility | New [BoTorch #3470](https://github.com/meta-pytorch/botorch/pull/3470) |
| Correlated empirical fantasies | Existing [BoTorch #3388](https://github.com/meta-pytorch/botorch/pull/3388) |
| Empty data-loader closure hang | New [BoTorch #3466](https://github.com/meta-pytorch/botorch/pull/3466) |
| Learned-preference noisy-improvement cache law | New [BoTorch #3497](https://github.com/meta-pytorch/botorch/pull/3497); seven source tests and ten subtests pass |
| Constrained MC objectives drop `X` and fail with augmented preference samples | New [BoTorch #3498](https://github.com/meta-pytorch/botorch/pull/3498); 17 source tests and 18 subtests pass |
| Fused CPU log-EHVI drops cell-bound gradients and detaches second derivatives | New [BoTorch #3510](https://github.com/meta-pytorch/botorch/pull/3510); actual compiled C++ kernel tests pass 14 methods and 33 subtests, including dependent arguments and numerical gradgradcheck; unchanged source fails 13 checks |
| Bilog and inverse have zero origin gradients and lose tiny forward values | New [BoTorch #3509](https://github.com/meta-pytorch/botorch/pull/3509); 19 tests pass on Torch 2.13 CPU/CUDA and 2.14 CPU, with 63/35 subtests respectively |
| Categorical feature axis | Existing [BoTorch #3387](https://github.com/meta-pytorch/botorch/pull/3387) |
| Generic kernel diagonal ranks | New [GPyTorch #2815](https://github.com/cornellius-gp/gpytorch/pull/2815) |
| Composite lazy feature counts | New [GPyTorch #2818](https://github.com/cornellius-gp/gpytorch/pull/2818), including composite feature indexing |
| Model-list joint covariance blocks | Existing [BoTorch #3409](https://github.com/meta-pytorch/botorch/pull/3409) extracts noninterleaved blocks, including interpolated kernels; [#3412](https://github.com/meta-pytorch/botorch/pull/3412) extracts interleaved variational blocks. These cover the documented dense-block workaround without a duplicate PR |
| Exact predictive cache squeezes a singleton model batch axis | Existing [GPyTorch #2804](https://github.com/cornellius-gp/gpytorch/pull/2804) |
| Variational unbatched zero-mean conversion | New [GPyTorch #2813](https://github.com/cornellius-gp/gpytorch/pull/2813) |
| Neural-kernel self-diagonal consistency, zero-clipping derivatives and large-variance overflow | New [BoTorch #3492](https://github.com/meta-pytorch/botorch/pull/3492); 19 tests and 31 subtests pass. Separate paired-input repair remains in [#3387](https://github.com/meta-pytorch/botorch/pull/3387) |
| Constant-target transform first/second derivatives | New [BoTorch #3468](https://github.com/meta-pytorch/botorch/pull/3468) |
| Periodic/cosine second derivatives | New [GPyTorch #2816](https://github.com/cornellius-gp/gpytorch/pull/2816) |
| Polynomial feature-batched offset | New [GPyTorch #2814](https://github.com/cornellius-gp/gpytorch/pull/2814) |
| Ensemble posterior standard deviation | Existing [BoTorch #3444](https://github.com/meta-pytorch/botorch/pull/3444) |
| Fully Bayesian explicit prediction-noise batch axis | New [BoTorch #3474](https://github.com/meta-pytorch/botorch/pull/3474) |
| Multitask kernel repeats existing task covariance batches | New [GPyTorch #2817](https://github.com/cornellius-gp/gpytorch/pull/2817) |
| GIBBON pending covariance units | Existing [BoTorch #3391](https://github.com/meta-pytorch/botorch/pull/3391) |
| GIBBON multiple query batch axes | New [BoTorch #3483](https://github.com/meta-pytorch/botorch/pull/3483) |
| MES zero-variance and one-draw control variates | New [BoTorch #3481](https://github.com/meta-pytorch/botorch/pull/3481); native regressions pass in `35884d0` |
| MES/GIBBON long-format multitask training baselines | New [BoTorch #3504](https://github.com/meta-pytorch/botorch/pull/3504); ten source tests and 14 subtests pass |
| Scalar JES conditional normal standardization | New [BoTorch #3479](https://github.com/meta-pytorch/botorch/pull/3479); native CPU and GPU repaired with independent Gaussian regression |
| PES damping accepts failed first-pivot Cholesky | New [BoTorch #3480](https://github.com/meta-pytorch/botorch/pull/3480); native CPU and GPU regressions cover the repair |
| Additive MAP SAAS factory dtype/device | New [BoTorch #3478](https://github.com/meta-pytorch/botorch/pull/3478) |
| ZeroLinearOperator device/dtype and batched logdet | New [linear_operator #151](https://github.com/cornellius-gp/linear_operator/pull/151) |
| Multiobjective entropy at infinite integration bounds | New [BoTorch #3485](https://github.com/meta-pytorch/botorch/pull/3485); GPU and native Gaussian-limit and derivative regressions pass |
| Linear latent-task MCMC loader bypasses mixin dispatch | Existing [BoTorch #3465](https://github.com/meta-pytorch/botorch/pull/3465) extended with mean/covariance hooks and continuous-only centering; 16 source tests and 30 subtests pass |
| Pairwise model broadcasting to singleton query batches | New [BoTorch #3489](https://github.com/meta-pytorch/botorch/pull/3489); 17 model tests and 32 subtests pass |
| Pairwise self-comparisons after duplicate consolidation and stable logit derivatives | New [BoTorch #3494](https://github.com/meta-pytorch/botorch/pull/3494); 19 source tests and 46 subtests pass |
| Correlated empirical kernel lazy output-count metadata | New [BoTorch #3499](https://github.com/meta-pytorch/botorch/pull/3499); four source tests and 31 subtests pass |
| Independent empirical lazy kernel output batches and rectangular query broadcasting | New [BoTorch #3500](https://github.com/meta-pytorch/botorch/pull/3500); four source tests and 34 subtests pass |
| Cached normal sampler and low-rank draws with multidimensional sample shapes | New [BoTorch #3501](https://github.com/meta-pytorch/botorch/pull/3501); eight source tests and 64 subtests pass |
| Multitask data-kernel active dimensions | Existing [GPyTorch #2809](https://github.com/cornellius-gp/gpytorch/pull/2809) |
| Repeated equality coefficients overwrite terms and get incorrect derivatives | New [BoTorch #3505](https://github.com/meta-pytorch/botorch/pull/3505); 26 source tests and four subtests pass |
| PSD projection derivatives at repeated eigenspaces and tiny floors | New [BoTorch #3506](https://github.com/meta-pytorch/botorch/pull/3506); 57 source tests and 14 subtests pass, including independent near-floor first/second derivative checks |
| Wide empirical orthogonal bases introduce repeated zero eigenspaces and NaN derivatives | New [BoTorch #3507](https://github.com/meta-pytorch/botorch/pull/3507); 19 source tests and 24 subtests pass |
| Empirical interpolation reuses constructor sort/gather graphs across backward calls | New [BoTorch #3508](https://github.com/meta-pytorch/botorch/pull/3508); 20 source tests and 12 subtests pass |
| Multistep incumbent units and batched analytic thresholds | Existing [BoTorch #3391](https://github.com/meta-pytorch/botorch/pull/3391) and [#3453](https://github.com/meta-pytorch/botorch/pull/3453) |
| Multistep fantasy base ranges and query-batch tree layout | Existing [BoTorch #3452](https://github.com/meta-pytorch/botorch/pull/3452) |
| TriangularLinearOperator second derivatives through matrix parameters | Narrow analytic-VJP enhancement in new [linear_operator #152](https://github.com/cornellius-gp/linear_operator/pull/152); 130 tests and 12 subtests pass |
| Port cosine float32 large-phase accuracy | Fixed in `4f1ad62`; CPU/CUDA regressions fail against old frozen binary and all 132 focused tensor cases pass with repair |
| Port bounded Torch L-BFGS false stationary trial | Fixed in `4062e86`, included in port PR #1; isolated CPU/CUDA interior and boundary regressions pass |
| Port rank-zero Index/PositiveIndex rejection | Fixed in `9a74f3f`, included in port PR #1; all 18 focused native/Tensor cases pass with the rebuilt library |
| Port CUDA MAP SAAS factory converts metadata twice | Fixed in `5b32fd2`, included in [port PR #1](https://github.com/M-Colley/botorch-rust/pull/1); 106 focused MAP SAAS / core-kernel cases pass, including CUDA active columns and coupled prior derivatives |
| Port Tensor validation and constant-batch restart diagnostics | Fixed in `6963a5c`, `01845f1` and `5867b1c`, included in [port PR #1](https://github.com/M-Colley/botorch-rust/pull/1); source-compatible constraint/fantasy width errors and warning categories/messages retain the existing numerical behavior |
| Port PyTorch 2.14 boundary derivatives | Fixed in the Tensor Bilog and shrinkage paths; CPU/CUDA Torch 2.13 and isolated CPU Torch 2.14 regressions pass. The PSD boundary oracle now explicitly selects its declared subgradient instead of inheriting version-dependent clamp semantics; included in [port PR #1](https://github.com/M-Colley/botorch-rust/pull/1) |
| Port type checking without Torch and cross-platform normal fixtures | Fixed in `e48230e` and `1bf2990`, included in [port PR #1](https://github.com/M-Colley/botorch-rust/pull/1); Mypy 2.4 passes with and without optional packages, and normal fixtures allow eight ULPs of libm roundoff while uniform draws remain bit exact |

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
