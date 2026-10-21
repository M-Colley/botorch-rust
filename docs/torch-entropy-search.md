# Device-resident entropy search

The Torch backend implements `qMaxValueEntropy`, `qLowerBoundMaxValueEntropy`
(GIBBON), `qMultiFidelityMaxValueEntropy` and
`qMultiFidelityLowerBoundMaxValueEntropy`. GP inference calls the port's
`model.posterior`; acquisition formulas and conditional Gaussian algebra use
Torch tensors on the model device. No runtime calculation delegates to BoTorch,
SciPy or NumPy. Native CPU models retain their Rust entropy implementations.

MES supports scalar Gaussian models, including supplied Gaussian-distribution
adapters and model batches. For batched ExactGP models the source requires an
explicit two-dimensional `train_inputs` candidate baseline; its default
three-dimensional training baseline is rejected. Queries have `batch_shape x 1 x d`
shape, including multiple query batch dimensions. Pending-point fantasies
support greedy batch filling. Multifidelity MES supports projection, trace
observations and cost weighting; multifidelity GIBBON supports projection and
costs, and rejects trace observations. Initially batched GIBBON remains an
explicit source boundary. MES supports loaded fully Bayesian mixtures without
pending observations, averaging their actual component estimates. The source
models do not expose a fantasy method for pending observations. MES observation/fantasy
samplers currently require one sample dimension. A single observation sample
uses the finite zero-control-coefficient fallback described below.

Custom public models may return a Gaussian posterior backed by an actual Torch
or caller-supplied GPyTorch multivariate normal. Scalar and multi-objective
entropy formulas read that posterior's public mean and full covariance without
requiring the port's private posterior class. Multitask interleaved and
noninterleaved event layouts retain their correct point/output ordering.
CPU/CUDA controls compare all four multi-objective estimators, joint and
nonjoint, with the unmodified source; separate dense conditioning and sampling
controls check full covariance, explicit draws and all-leaf mixed Hessians.

Thompson maxima and Gumbel fitting run on device. The Gumbel sampler solves the
independent Gaussian maximum's three quantiles with batched log-CDF bisection.
It retains posterior precision, consistent with the existing native CPU sampler;
the pinned BoTorch helper rounds its SciPy roots through the default dtype.
Independent float64 SciPy roots provide the stricter numerical oracle, with the
unmodified reference compared separately. Standalone Tensor Gumbel sampling
retains implicit first and second training-data derivatives through Newton
refinements at the solved root. Acquisition constructors deliberately sample
their maxima under `no_grad`, matching the fixed-maximum-sample estimator.

The default Gumbel constructors preserve the reference's random-draw order,
including observation/fantasy sampler seeds and the multifidelity redraw.
Explicit `seed`, `sampler`, `fantasies_sampler`, `uniforms` and `base_samples`
extensions allow controlled sampling. Pending observations are detached when
fantasized as in the reference; kernel, mean, noise, training-data and query
operations otherwise retain their Torch graphs. Cost samplers have independent
state. Raw training inputs are appended to the maximum candidate set, avoiding
the existing reference double-input-transform issue.

Validation uses official revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`. The focused CPU/CUDA suite has 84
tests, including float32 and float64, synchronized maximum/observation draws,
minimization/maximization, pending fantasies, traces, costs, query batch shapes,
first/mixed second derivatives and complete training/query `gradcheck` and
`gradgradcheck`. Tests forbid `.cpu()`, `.numpy()` and `.tolist()` during
construction, inference and double backward. Nine existing native max-entropy
constructor/sampling regressions also pass. No speed claim is established for
these entropy acquisitions yet.

## Model-batch MES layouts

The Tensor MES implementation keeps model, pending-fantasy and query axes
separate, then averages the model/fantasy estimates. Max-value samples retain
their model-batch dimensions and can share singleton dimensions. Its Gaussian
calculations support multiple model-batch axes without folding a model axis
into a query or fantasy axis. Queries and max samples retain their full
training/query/parameter graphs.

Multi-fidelity MES follows the same layout for a supplied Gaussian model
protocol that does not expose a batched training baseline, as allowed by the
source constructor. Projection, the default deterministic cost weighting
and pending observations remain supported. The original ExactGP MF
constructor has no `train_inputs` override and rejects a batched default
baseline; a generic Gaussian model can instead supply the candidate set
directly. This does not claim support for initially batched GIBBON or JES.

The batched regression module checks unmodified source values and query
derivatives, and complete initial data/parameter derivatives with its
`propagate_grads` setting enabled. The reference's fantasy method requires
shared pending locations to be explicitly expanded over the initial model
batch to satisfy its input-rank check; the port accepts shared raw locations.
The distinct source rank-check repair is proposed in
[GPyTorch #2830](https://github.com/cornellius-gp/gpytorch/pull/2830), with
independent dense conditioning controls for fully and partially shared locations.
Source comparisons use one pending fantasy. Independent component averaging
checks multiple model axes and three fantasies, including every active
data/query/max-sample/parameter gradient and Hessian-vector product. It also
forbids inference-data host conversions during construction, conditioning
and double backward. No batched MES speed claim is established.

Fully Bayesian MES and multifidelity MES retain the implicit MCMC component
axis separately from observation and maximum-sample axes, then reduce both
the component and singleton fantasy dimensions. The 30-case CPU/CUDA suite
checks all three scalar fully Bayesian families, component-wise values and
complete data/query/max-sample/parameter gradients and mixed HVPs. Unmodified
source acquisitions evaluated separately for each component provide an
additional independent value/query-gradient oracle. All 15 available cases
pass on Torch 2.14.1 CPU, and 14 independent cases pass with reference packages
blocked. No mixture MES speed claim is established.

The source's direct mixture acquisition incorrectly derives the broadcast
rank from the input rather than the posterior mean: unequal observation and
maximum-sample counts raise, and equal counts silently combine the wrong axes.
Both source defects are asserted separately from the independent component
oracle. [BoTorch #3529](https://github.com/meta-pytorch/botorch/pull/3529)
proposes deriving the shape from the actual posterior batch dimensions;
the repaired upstream family passes 10 tests and 16 CPU/CUDA subcases.

## Verified reference defects and concrete repairs

**GIBBON pending covariance uses inconsistent output units.** With Standardize,
the reference obtains cross-covariance from `model(X_batches)` in transformed
units, but its pending block and marginal variance from `model.posterior` in
physical output units. The Schur complement then changes with the arbitrary
output scale. The new test multiplies the training outcomes by four and compares
against the complete noisy joint posterior. The repair is to obtain cross and
pending blocks from that same posterior, applying any posterior transform
consistently. The port uses this corrected covariance space. The unmodified
reference discrepancy is asserted separately from the independent corrected
Gaussian-law comparison.
This output-unit repair is already proposed in
[BoTorch PR #3391](https://github.com/meta-pytorch/botorch/pull/3391);
it is not a newly discovered independent upstream defect.

**GIBBON assumes one query batch dimension.** `acq.mean(dim=1)` averages the
wrong axis for a `3 x 2 x 1 x d` query, and pending expansion/indexing also fixes
the number of batch axes. The unmodified pinned implementation raises for that
query shape. Average the final maximum-sample axis and use ellipsis indexing
and broadcasting across `X.shape[:-2]`. The port's batched values equal separate
row evaluations and retain all parameter derivatives.

**MES control-variate correlation is undefined for constant observation
draws.** An explicit all-zero normal base-sample tensor yields zero variance for
the log-density control variate. The pinned implementation divides by a zero
variance product and returns NaN. Set the control coefficient to zero in this
case; select a finite positive product before the square root to keep inactive
branches safe under double backward. Tests assert the reference NaN and the
port's finite value and second derivative.
With a single observation, skip variance evaluation entirely: the unbiased
variance has undefined derivatives even when its NaN value is masked later.
The one/eight-observation regressions run on both CPU and CUDA. The native
control mean is centered before summation, so an exactly constant control
retains exactly zero variance instead of a floating-point mean residual.

These are local implementations and proposed upstream repairs. Root-level
publication work tracks their issues and pull requests; this chapter does not
claim an upstream submission.

## Joint entropy search

`qJointEntropySearch` now provides device-native `LB` and `MC` estimates for
initially unbatched Gaussian models with differentiable conditioning. Candidate
batches, pending points, raw optimal locations and scalarizing posterior
transforms retain the same semantics as the corrected native CPU estimator.
The initial model must support conditioning; fully Bayesian mixtures and
initial model batches remain explicitly unsupported for JES. The focused
50-case CPU/CUDA suite includes synchronized official Gaussian sampling,
independent Schur-complement conditioning for explicit fantasy noise,
Normalize, `q > 1`, multiple query-batch axes, training/optimal/query first and
second derivatives, scalarized multi-output and ModelListGP models, float32
and float64, all parameter Hessian products, and checks forbidding
inference-data host conversion. This does not establish a speed claim.

Conditioned models own independent copies of their module parameters, matching
the source ExactGP protocol. JES uses the initial covariance and the conditional
mean/covariance, so the initial model's mean parameter is mathematically unused.
The parameter regressions assert that specific unused leaf against the
unmodified source, then check every active initial and conditional parameter
strictly. Independent source GP priors rebuilt from the augmented observations,
with dense Gaussian conditioning and sampling, provide the child-parameter
partials and all-leaf mixed Hessian-vector products. These bypass the retained
incremental fantasy-cache graph and the separately reported source RBF and
operator higher-backward defects (GPyTorch #2821, linear_operator #153/#155). Scalar and
multi-objective JES both use these checks on CPU and CUDA. The scalar MC oracle
applies the explicitly documented conditional-CDF correction below.

The scalar JES Monte Carlo CDF has a further verified scaling defect in the
official pinned revision. At
`botorch/acquisition/joint_entropy_search.py:346`, it evaluates
`(normalized_mvs - rho * normalized_samples) / (1 - rho**2)`. For standardized
jointly Gaussian latent and noisy observations, the conditional variance is
`1 - rho**2`; standardizing a normal CDF requires its square root. For example,
`F ~ N(0, 0.3)` and `Y = F + E`, with independent `E ~ N(0, 0.7)`, give
`F | Y=1 ~ N(0.3, 0.21)`. Therefore
`P(F <= 0 | Y=1) = Phi(-sqrt(3/7)) = 0.25634538013096175`.
The current reference instead evaluates `Phi(-sqrt(0.3)/0.7)`, approximately
`0.2169719081489336`. Replace the denominator by `(1 - rho**2).clamp_min(CLAMP_LB).sqrt()`;
the clamp follows the existing multi-objective entropy estimator's numerical
boundary convention. The Rust and Tensor implementations apply this repair.

Tests separately assert the unmodified reference discrepancy and the corrected
Gaussian-law result. Existing live Monte Carlo comparisons explicitly install
the corrected reference oracle. Four static Rust MC fixture entries were
regenerated from the official revision using their identical recorded model
parameters, query locations and normal draws; other fixture entries remain
unchanged. Earlier conditioning/input-transform/marginal-density corrections
are described in [the native acquisition chapter](single-objective-acquisition.md).
The new Rust scalar regression tests the closed Gaussian example directly.

## Predictive entropy search

`qPredictiveEntropySearch` and `qMultiObjectivePredictiveEntropySearch` now
run their expectation-propagation algebra on device. Both support pending
points, `q > 1`, several query-batch axes, maximization/minimization and scalar
or independent multi-output Gaussian models, including ModelListGP. Ordinary
initial model batches are rejected in the same way as the official family;
fully Bayesian EP mixtures are also outside this implementation's supported
scope. Raw training locations are used when building factors, consistent with
the native port's existing input-transform correction. The reference's
deliberately noisy candidate EP covariance convention is preserved.

PES damping and fallback select discrete branches. The implementation retains
exact Torch derivatives within the selected branch; it does not claim
differentiability at a damping/convergence/fallback boundary. It recomputes the
EP factor graph for differentiable evaluations, retaining training-data,
optimal-location and hyperparameter dependence and permitting repeated backward
calls. Functional factor assembly avoids in-place modification of tensors saved
by autograd. Cholesky branch choices are made without recording invalid factors;
only the final positive matrix is factored on the differentiable tape.
`log_cdf_robust` preserves the reference's CDF rounding/clamps and tail
approximation while masking logarithm-of-zero branches before evaluation.

The 30-case CPU/CUDA suite checks reference values/query derivatives, complete
training/optimal/query `gradcheck` and `gradgradcheck`, all parameter Hessian
products, repeated backward calls, Normalize, pending points, scalar/MO/list
models, float32/float64, minimization, arbitrary query batches and forbidden
inference-data `.cpu()`, `.numpy()` and `.tolist()` conversions. Scalar device
convergence/PSD decisions synchronize metadata with Python, as in the reference.
Two additional JES regressions check real device candidate optimization and
ensure model parameter `.grad` fields remain untouched. No speed improvement
has been measured for this EP implementation.

**PES can accept a failed first Cholesky pivot.** The official
`multi_objective/predictive_entropy_search.py:1032,1046` checks whether the sum
of Cholesky `info` codes exceeds one. Every nonzero code is a failure, including
`info=1`. On CPU, calling `_update_damping(I2, diag(-1,1), tensor(1.), 1e-4)`
returns damping one and a finite invalid factor. The additional NaN test does
not detect it. CUDA's failed factor can contain NaNs and trigger that separate
fallback, so the exact unmodified symptom is backend dependent. Reject any
nonzero code; in this example damping one half produces the valid matrix
`diag(1e-4, 1.0001)` after jitter. Both native Rust tests and CPU/CUDA Tensor
tests cover the repair, with the unmodified CPU failure asserted separately.
The Rust change also rejects a previous precision whose first pivot fails.

## Multiobjective lower-bound MES and JES

`qLowerBoundMultiObjectiveMaxValueEntropySearch` and
`qLowerBoundMultiObjectiveJointEntropySearch` now support all four reference
estimators (`0`, `LB`, `LB2`, `MC`) with device-resident posterior statistics,
Gaussian draws/densities and truncated moments. They preserve the existing
native/reference Gaussian-constant convention for batches, including its
`q > 1` behavior. This checkpoint does not change that numerical definition.
JES conditions on raw Pareto locations, retaining gradients through the
conditioning observations. The public entropy helper functions also retain
Tensor graphs when a Tensor is on CUDA or requires gradients.

The 80-case CPU/CUDA suite checks scalar and multiple objectives, ModelListGP,
Normalize, pending points, arbitrary query batches, synchronized Monte Carlo
draws, float32/float64, all model-parameter Hessian products and complete
training/Pareto/bound/query `gradcheck` and `gradgradcheck`. Bound Hessian checks
use disjoint boxes in the smooth interior. Independently perturbing duplicate
shared partition faces can make boxes overlap and cross the reference's
probability-mass clamp, where a classical Hessian need not exist. No inference
arrays are converted to host Python/NumPy data. Initial model batches are
rejected as in the official family; JES conditioning on fixed-noise models
requires unspecified Pareto observation noise and is explicitly rejected,
matching the source's unsupported configuration.

**Infinite integration bounds need normal-tail limits.** In the pinned
multiobjective entropy helpers, `g * phi(g)` is evaluated directly at
`g = +/-inf`. A single box spanning the full Gaussian space therefore yields
non-finite noiseless/moment entropy (`NaN`, or `-inf` for CUDA log-determinant
paths), although its exact entropy is finite. The Monte Carlo forward value
can remain finite while its variance derivative becomes NaN because
standardizing an infinite bound records an infinite variance derivative.
The port masks infinite bounds *before* standardization, substitutes exact
CDF limits `0/1`, and uses the tail limit `g * phi(g) = 0`.
Tests compare the resulting values with the closed Gaussian entropy and
assert finite first/second derivatives; the original non-finite values or
variance gradients are reproduced separately. The usual finite-bound
estimators remain compared directly with the unmodified pinned reference.

The native CPU box implementation applies the same limits: infinite endpoints
have zero `g*phi(g)` moment and zero posterior-parameter adjoints, including
the rescaled endpoints in the MC estimator. New Rust regressions verify the
full-space Gaussian entropy/adjoints for all four estimators and a half-line's
closed entropy with finite-difference mean/variance/noise derivatives. MES
has a separate exact-law regression for one/eight zero-normal observation
draws, checking its finite value and Dual derivatives.

## Hypervolume knowledge gradient

`qHypervolumeKnowledgeGradient` and its multi-fidelity variant now evaluate
their complete fantasy trees on the Tensor backend. The inner value is the
port's device qEHVI or qLogEHVI over an empty front, using either posterior
means or synchronized Gaussian draws. Training inputs/targets, model
parameters, reference points, objective weights and every actual/fantasy
design retain their graphs. Pending observations, decoupled evaluation masks,
projection and deterministic/stochastic costs are supported. Cost sampling
uses independent clones of the fantasy streams, retaining Gaussian cost
correlations. A completely unobserved component retains its original model
instead of invoking an invalid zero-dimensional QMC sampler.

The 34-case CPU/CUDA suite checks pinned-reference values for mean/sample,
standard/log and pending/decoupled configurations, two/three objectives,
live weights, full training/query `gradcheck` and `gradgradcheck`, all
parameter Hessian products, multiple query batches, multi-fidelity projection
and costs, and absence of inference-data host conversions. Combined with
the scalar KG regression suite, 56 cases pass. Reference value comparisons
use a single query batch because the pinned reference's fantasy reshape
mixes designs across different query batches. The port retains its existing
correct per-batch fantasy layout and tests batched evaluations against
independent calls.

HVKG currently requires a ModelListGP of single-output components, as in the
native implementation. Trace observations remain unsupported, matching
BoTorch. The reserved `valfunc_cls` and `valfunc_argfac` arguments have no
effect, matching the pinned official value factory. No HVKG speedup has been
measured.

## Multi-step lookahead

`qMultiStepLookahead` now recursively conditions device models and evaluates
port analytic or MC acquisition classes at every stage. User argument
factories receive each actual fantasy model and that stage's design Tensor.
The complete training, fantasy-observation, kernel/noise-parameter and query
graphs are retained, including the initial fantasy distribution. Nonuniform
`base_weights` remain differentiable. Pending observations, scalarizing
posterior transforms, nonlinear MC objectives, tree extraction and the
induced fantasy model are supported. Initial/later analytic stages still
require `q=1`, matching the corresponding acquisition; pending points can
make an initial analytic stage invalid.

The 38-case CPU/CUDA suite covers one/two stages, mean/EI/qEI values against
the pinned reference, pending points, multiple query batches, nonlinear
objectives and scalarized ModelLists, all-parameter Hessian products and
complete training/query `gradcheck` and `gradgradcheck`, explicit fantasy
layouts, live integration weights and forbidden host data conversions.
Combined HVKG/multistep validation passes 72 cases. Multioutput reference
tests use ModelListGP: the pinned GPyTorch SingleTaskGP multioutput fantasy
path can fail its prediction reshape before the stage value is evaluated.
Stage classes must be port classes (or explicitly supplied custom classes),
and their device-supported configurations determine the supported stage
values. Nested lookahead value functions are rejected, as in BoTorch.

Two incumbent defects are already covered by existing upstream proposals;
they are not new unique reports:

- **Outcome scale:** pinned `multi_step_lookahead.py:679` returns
  `model.train_targets.max(-1).values`. With standardized targets this is a
  standardized incumbent, although EI compares it with an original-scale
  posterior. For `Y=12*sin(5.3*X)+7`, `X=[[.04],[.27],[.56],[.81]]`, it
  differs from `Y.max()` by construction. The Tensor branch of the port's
  `make_best_f` reads the live raw targets, including fantasy observations,
  retaining shape `fantasy_shape x batch_shape` and gradients. Existing
  [PR 3391](https://github.com/meta-pytorch/botorch/pull/3391) proposes
  untransforming source targets before taking the maximum.
- **Batched analytic layout:** with two fantasies and two query batches,
  `make_best_f` has shape `2 x 2`, while analytic EI's mean retains shape
  `2 x 2 x 1`. The pinned EI broadcasts these to `2 x 2 x 2` and fails its
  output shape assertion. Construct a lookahead with `batch_sizes=[1]`,
  `num_fantasies=[2]`, `valfunc_cls=[None, ExpectedImprovement]`,
  `valfunc_argfacs=[None, make_best_f]`, and evaluate a `2 x 3 x d` tree to
  reproduce it. Existing [PR 3453](https://github.com/meta-pytorch/botorch/pull/3453)
  adds the missing output dimension to batched analytic `best_f`, as the
  port already does. Unmodified-reference analytic-stage parity therefore
  uses a scalar incumbent; a separate regression exercises the dynamic
  fantasy-batch incumbent with the port's corrected layout/scale.

When fantasy base samples are not collapsed, the Tensor implementation
derives the collapsed query-batch range from the posterior's actual event
rank (`q` for scalar, `q,m` for multioutput). This preserves separate normal
streams for earlier fantasy nodes. The pinned implementation hard-codes
two event axes, which collapses a scalar fantasy axis and retains a query
axis instead. A regression checks the independent scalar streams' complete
shape. This is a deliberate correction, not a claim of identical draws for
that noncollapsed reference configuration. Tree partitioning also preserves
the existing native port's per-query fantasy grouping for `q_i>1`.

Both independence fixes are also already proposed in
[PR 3452](https://github.com/meta-pytorch/botorch/pull/3452). A minimal scalar
base-layout reproduction uses `batch_sizes=[2,1]`, fantasy counts `[2,3]`,
`collapse_fantasy_base_samples=False` and a `2 x 11 x d` tree. The pinned
source produces second-stage normals of shape `3 x 1 x 2 x 2`, sharing the
earlier fantasy axis and retaining the query axis. The correct shape is
`3 x 2 x 1 x 2`: each earlier fantasy has its own stream and every query
batch shares the stream. The PR independently verifies that adding other
query batches does not change an existing query's value. The upstream
multioutput fantasy reshape failure is already tracked as
[GPyTorch issue 2804](https://github.com/cornellius-gp/gpytorch/issues/2804).

## Multitask entropy baselines

MES and GIBBON also support single-output `MultiTaskGP` designs without the
categorical task column. Their training baseline is aligned to the continuous
design coordinates, while posterior sampling still targets `output_tasks`.
Task columns at the beginning, interior and end, external labels, full-labelled
designs and explicitly supplied full/reduced baselines are covered. Full task
columns are retained when the candidate set supplies them. This corrects a
constructor failure in both pinned BoTorch b0dd01be6 and the initial Tensor
port. The native default already used continuous training coordinates; explicit
native baselines now follow the same rule. Task-free pending fantasies remain
the model's separate unsupported configuration. The combined max-entropy and
multitask suite passes 161 CPU/CUDA tests, including 34 focused cases with
first/second derivatives and no host conversion. The independently tested
source patch is `D:/botorch-pr-multitask-entropy`, commit `92d08a63f`; see
[multitask bug 10](multitask.md) for the historical report and source test counts.

## Device pathwise sampling

`draw_kernel_feature_paths`, `draw_matheron_paths`, `gaussian_update` and
`gen_kernel_features` support device exact SingleTaskGP models with
RBF/Matérn kernels, optional ScaleKernel/active dimensions, model/output
batches and ModelLists. Fourier frequencies, Gaussian weights and update
noise can be supplied explicitly, preserving their gradients. The exact
Gaussian update uses the port's native Tensor covariance and PSD factor
bindings; feature algebra, linear solves and outcome transformations retain
the full Torch graph. Scalar/multioutput path layouts match the reference's
sample/model batch broadcasting. Explicit Tensor feature maps, generalized
linear paths and Matheron compositions also preserve higher derivatives.

`get_matheron_path_model` and the public `MatheronPathModel` factory expose
the device paths as deterministic models with exact point-mass posteriors.
They can be used directly by device MC acquisitions. Explicit device/dtype
conversion rebuilds the exact Gaussian correction for the same stored
random draw. A path remains a draw with its construction-time update;
redraw it after fitting/changing the source model, as with the reference.

The 48-case CPU/CUDA suite checks synchronized pinned-reference path values,
query derivatives and update weights for one/two outputs, RBF/Matérn/scaled
kernels and Normalize/Standardize. It verifies fixed noise, model/list
batches, multidimensional or empty sample shapes, reproducible/explicit
draws, active dimensions, float32/float64, all train/query/parameter/draw
Hessian products, complete data/query/frequency/weight/noise `gradcheck`
and `gradgradcheck`, public model conversion and an end-to-end scalar MC
acquisition. No inference data is converted to host arrays. Together with
the existing feature, path-constructor and utility regressions, 142 cases
pass; the no-Torch/no-NumPy subprocess import check also passes.

Device `SingleTaskVariationalGP` prior paths and scalar whitened Matheron
updates are also supported. This follows the pinned official BoTorch
`b0dd01be61d517c387e234903a6e2cc38c057ea2` registrations for
`ApproximateGPyTorchModel` and `VariationalStrategy`: draw the current
whitened variational Gaussian, multiply by the Cholesky factor of `K_ZZ`,
add the mean at the inducing points, then solve the latent Gaussian
correction. The inducing points are already in transformed coordinates;
only subsequent raw queries pass through the input transform. This update
adds neither observation noise nor `VariationalStrategy.jitter_val` to
`K_ZZ`; it retains the factorization's PSD fallback behavior. The stored
third draw is the whitened standard normal, so explicit draws and seeds
reproduce paths without changing the variational parameters or initializing
an untouched variational distribution.

The additional 28-case CPU/CUDA suite checks synchronized unmodified
reference update formulas for RBF/Matérn, model batches and
Normalize/Standardize; all training-data, query, parameter and draw Hessian
products against an independent RBF formula; complete data/query/draw
`gradcheck` and `gradgradcheck`; seeded and explicit targets; path-model
MC acquisition use; and dtype conversion. No inference tensor is converted
to a host array. Training-data gradients arise through fitted transforms;
the already fitted variational Gaussian is a separate model state, and
observation-noise parameters intentionally do not enter this latent update.

Independent multi-output variational prior paths match the reference.
Its Matheron sampler fails on `IndependentMultitaskVariationalStrategy`
because that wrapper has no direct `inducing_points` attribute; the port
reports this same unsupported strategy explicitly. Structured/fully Bayesian
device path updates and arbitrary custom prior/update strategies remain
explicit unsupported configurations. As with the native path model and reference,
MatheronPathModel requires single-output components in a ModelList.
Random Sobol uniforms originate from Torch's CPU Sobol generator; only new
random numbers are transferred, and all normal transforms/inference run
on the model device. This checkpoint does not claim a pathwise speedup.
