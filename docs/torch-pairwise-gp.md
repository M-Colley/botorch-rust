# Device-resident preference GPs

`PairwiseGP(..., backend="torch")` evaluates Laplace inference, Gaussian
prediction, and the Laplace marginal log likelihood on the training Tensor's
CPU or CUDA device. CUDA training inputs, inputs requiring gradients, or Torch
kernel/transform modules select this backend automatically. The existing
native CPU model remains available through `backend="native"`.

The model supports the built-in probit and logit pairwise likelihoods, scaled
Tensor kernels, input transforms, training batches, prior-only prediction,
duplicate consolidation, and conditioning on new pairwise observations.
Raw kernel and transform parameters are actual Torch Parameters. The public
`PairwiseLaplaceMarginalLogLikelihood` factory returns a Torch module for this
model, so direct Torch optimizers, loss closures, and the public SciPy/Torch
fitters optimize those same leaves. SciPy's optimizer explicitly transfers
parameter vectors; covariance calculations and the utility solve remain on
device. Kernel parameter batches must broadcast to the training batch.
Arbitrary custom likelihood implementations are rejected explicitly.

The latent utility is the converged root of
`K^-1 (f - mean) + negative_log_likelihood_gradient(f) = 0`.
A damped Newton solver uses device tensors throughout and synchronizes only
scalar stopping decisions. It does not call SciPy, NumPy, GPyTorch models,
BoTorch models, or the flattened native CPU preference solver. Failed
convergence raises rather than returning an unchecked approximation.

An implicit autograd Function differentiates the converged root. Its backward
solves the transposed root Jacobian system and retains the returned root in
the adjoint graph. Differentiating that graph therefore includes the mode's
dependence again, supporting second and higher derivatives. The tests apply
both gradcheck and gradgradcheck to independent covariance and mean inputs,
and evaluate third derivatives. Training data, queries, and parameters retain
their complete functional graphs, including learned Normalize statistics.
No derivative is claimed through discrete comparison indices or changes of
duplicate membership. The usual Torch version checks apply when a Parameter
needed by an old graph is changed before backward.

Each training output retains its numerical state for evidence evaluation.
An older output's loss does not mix its utility with a newer model covariance
or Hessian. Consolidated input selections are reconstructed from the original
training data for independent calls, so repeated backward calls do not reuse
a freed index-selection graph. Saved posterior moments retain their values
after later model updates. State loading into a compatible model and actual
Adam/SciPy fitting are covered by the focused regression suite.

The reference probit implementation clamps standardized differences to
`[-3,3]`; the logit likelihood clamps its reported log probabilities to
`[-8,8]`. The Tensor path retains these public formulas. The probit root
gradient uses the clipped value, so its actual Jacobian includes the clamp's
derivative. The supplied reference likelihood Hessian omits that derivative.
The implicit mode adjoint uses the actual root Jacobian; the Laplace covariance
continues to use the reference's declared likelihood-Hessian formula.
At a clipping boundary, higher derivatives have the same mathematical
nondifferentiability as the underlying clamp.

## Verified reference defects and repairs

These reproductions use official source
`b0dd01be61d517c387e234903a6e2cc38c057ea2`.

1. **Singleton query batch against a larger training batch.** Construct a
   PairwiseGP with datapoints `[2,4,2]` and comparisons `[2,5,2]`, then call
   `posterior` with query `[1,2,2]`. `_transform_batch_shape` compares batch
   ranks and attempts to expand the training batch from 2 down to 1, raising
   `RuntimeError`. Compute `batch=torch.broadcast_shapes(X.shape[:-2],
   X_new.shape[:-2])`, then expand both operands to that batch. The Tensor
   implementation broadcasts correctly; parity tests compare with explicitly
   expanded reference queries and reproduce the original exception.
2. **Self-comparison created by duplicate consolidation.** Datapoints
   `[[.2,.3],[.2,.3]]` and comparison `[[0,1]]` consolidate to one item.
   The reference's winner scatter writes `+1`, then its loser scatter
   overwrites the same location with `-1`. It produces `D=[[-1]]`, mode
   approximately `-.4338`, and comparison probability approximately `.6205`.
   A self-comparison has utility difference zero and probability `.5` for
   every utility; it must leave the zero-mean prior mode unchanged. Accumulate
   both signs with `scatter_add_` so the row cancels to zero. Both the native
   CPU and Tensor implementations use that correct law. The regression
   reproduces the reference defect before checking the repaired law.

The 40-case CPU/CUDA focused suite compares utilities, evidence, full covariance, query
gradients, and all evidence data/parameter gradients with the independent
reference. Complete mixed evidence Hessian-vector products are compared with
differences of reference first gradients; prediction products are checked
against independent finite perturbations of complete functional gradients.

## Preference acquisitions

Analytic EUBO, qEUBO, and pairwise BALD dispatch to device-preserving Tensor
implementations for Torch utility models. They support deterministic outcome
maps, analytic previous winners, MC objectives/posterior transforms, qEUBO
pending points, candidate optimization, and first/second derivatives through
training data, queries, utility-model parameters, and outcome-map parameters.
Keyword `pref_model=...` and positional construction select the same backend.
BALD draws device-local normal samples on each evaluation; qEUBO uses its
supplied sampler or the default Sobol sampler. Ensemble components use the
reference's final-axis average.

The 26-case CPU/CUDA acquisition suite compares values and query gradients
against independent official PairwiseGP models. It also compares complete
first and mixed second derivatives against the unmodified official acquisition
formulas fed the same port Gaussian law, so model-law and acquisition-formula
checks remain separate. Candidate optimization and `value_and_grad` are covered.

For qEUBO, the installed reference sampling dependency drops a higher-order
graph: `TriangularLinearOperator._bilinear_derivative` inherits a generic
fallback that calls `autograd.grad` without `create_graph`. For example,
`grad((TriangularLinearOperator(L) @ z).square().sum(), L,
create_graph=True)` returns a gradient without a graph for a leaf dense
triangular `L`, while ordinary Tensor multiplication retains it. The first
derivative values agree. Consequently qEUBO mixed-Hessian validation uses
differences of independent reference first derivatives. Delegating the
triangular wrapper's bilinear derivative to its wrapped operator preserves
the dense analytic VJP and its graph. This is a demonstrated mathematical
limitation of that path, not a claim of a library-wide higher-order contract.
