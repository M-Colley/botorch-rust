# PairwiseGP input-transform correctness

Reference: official BoTorch revision `b0dd01be6`. Port audit based on
`ef7412f` and the associated phase-10 worktree.

## Shared normalization across training batches

The port previously fitted a shared `Normalize(batch_shape=())` separately
inside every native PairwiseGP batch member. Official BoTorch fits that
transform across all training batches. This changes training covariances,
the Laplace approximation, predictions, and gradients whenever the member
input ranges differ. The Python transform also reported only member zero's
bounds, hiding the discrepancy.

For example, use these two input batches, repeated comparisons
`[[0, 1], [2, 1], [3, 0]]`, and a shared `Normalize(2)`:

```python
X = torch.tensor([
    [[.1, .2], [.3, .8], [.7, .4], [.9, .7]],
    [[2.1, 2.2], [2.3, 2.8], [2.7, 2.4], [2.9, 2.7]],
], dtype=torch.float64)
```

Official bounds are `[[.1, .2], [2.9, 2.8]]`; the affected port reported
`[[.1, .2], [.9, .8]]`. With the default kernels the maximum absolute
training-covariance discrepancy was approximately `0.7592449`.

The native fix fits the shared chain against pooled training rows before
the member Laplace solves. Those member views freeze coefficient fitting
and reuse the common coefficients. The batch retains the original chain's
learning settings and fitted state. A subsequent training forward or
training-mode conditioning therefore refits the pooled coefficients;
evaluation-mode conditioning preserves the original bounds. Python receives
the retained shared state, including its original learning flags.

Regression tests compare shared and chained Normalize bounds, training
covariances and roots, utilities, likelihood Hessians, marginal likelihoods
and their parameter gradients, and posterior query derivatives against
the official implementation. Both one- and two-dimensional training batches
and list, NumPy, and Torch inputs are covered. Conditioning verifies both
training and evaluation behavior.

This is a port bug; the audited official implementation has the correct
shared normalization behavior.

## Independent default outputscales in training batches

The same regression exposed a separate port error: its default PairwiseGP
kernel forcibly treated the outputscale as shared across batch members.
The official model's GPyTorch `ScaleKernel` inherits the batched RBF kernel's
batch shape and therefore has an independent outputscale per member.
The port now retains that inherited batch shape. Its existing native
parameter mapping already supports the independent scales.

Equal initial scales hid this discrepancy in kernel-matrix comparisons, but
the parameter shapes, prior count, and gradients were incorrect. In a
two-member example, official MLL was `-16.88436555` and the affected port's
MLL was `-12.27904472`; the difference was one missing outputscale prior
term (`4.60532084`). Regression tests now compare complete parameter shapes,
the MLL, and every raw-parameter gradient, in addition to Laplace state.

## Learned input transforms

PairwiseGP now supports learned Warp, including feature subsets,
nondefault epsilon, concentration priors, and supported native input chains
such as Normalize followed by Warp. The warp, Laplace approximation,
parameter derivatives, optimizer, prediction, and query derivatives execute
in Rust; no BoTorch inference is called.

Every optimizer trial transforms the training inputs using that trial's
concentrations and records the native transform tape. The Laplace reverse
pass supplies the training-covariance adjoint through the same two Newton
updates as official BoTorch, holding the numerical fsolve solution fixed.
Kernel input derivatives contract that symmetric adjoint into transformed
training-input adjoints, which the tape propagates into the concentrations.
Concentration priors and their derivatives are included once per parameter.
The final fit installs both the concentrations and fitted transform state;
Laplace cache identity includes the transform parameters.

The public transform is bound to the native parameter state, so setters,
`named_parameters`, state dictionaries, copies, and loading include learned
concentrations. Public names follow BoTorch's module order. The extension
`PairwiseLaplaceMarginalLogLikelihood.value_and_grad()` uses the native flat
vector order returned by `model._param_names()`: kernel parameters followed
by input-transform parameters. Warp parameters are `concentration0` then
`concentration1` in the supplied feature-index order. The fixed mean is not
part of this optimizer vector.

Validation includes official Laplace values, every parameter gradient,
posteriors and query VJPs, native fits, state restoration, training/evaluation
conditioning, and fantasy predictions. It covers list/NumPy/Torch inputs,
probit and logit likelihoods, both default and composite kernels, concentration
priors, and changed trial concentrations. Native finite differences check
the Laplace gradients with and without priors for both likelihoods and kernel
paths. The fantasy comparison uses the previously audited official pairwise
sampler orientation correction only inside the reference test.

This checkpoint exposes explicit native MLL gradients and query autograd;
persistent Torch parameter graphs through PairwiseGP inference are not
claimed. One-to-many transforms,
`Warp(reverse=True)`, and chains that learn new normalization coefficients
after a learned transform retain explicit errors. A fixed transform after
Warp does not have that coefficient-dependence restriction.

## Batched Warp and global parameter ownership

Training batches support shared Warp concentrations or independent
concentrations with exactly the model's batch shape. A chain may mix these
two ownership patterns. The Warp's internal normalization and a preceding
shared Normalize learn their bounds over all training batches; the member
Laplace solves use frozen copies of those coefficients with their own trial
concentrations. Native data gradients accumulate through each member's map
into the global raw vector. Every global concentration prior contributes
once, including when several members or fantasy dimensions share it.

This works with the default independent batched RBF parameters, shared
standard kernels, and shared composite kernel trees. Independent batched
parameters inside a composite kernel tree retain an explicit error.
Per-member coefficient-learning chains, such as a Normalize with a batch
shape, remain unsupported; the constructor rejects them explicitly. Warp
batch shapes broadcast from a smaller nonempty shape are not accepted at
initial construction. Conditioned models retain their parent's parameter
batch shape when adding fantasy dimensions, with shared native maps and
folded gradients.

The same work fixes two additional port defects. Conditioning previously
kept each member's parameter values for prediction, but rebuilt the global
raw vector and ownership map using only member zero. Updating parameters,
evaluating priors, or fitting that conditioned model could therefore lose
independent member values. Conditioning now preserves the parent's raw
vector and repeats its existing member maps over new fantasy axes.
Replacing batched training data also used to discard learned kernel values;
the wrapper now materializes every bound kernel and Warp parameter before
rebuilding its native data state.

Regression tests compare named parameter values/shapes, Laplace state,
MLLs and every global gradient, posterior query adjoints, changed trial
concentrations, fitting, conditioning in both modes, state restoration,
data replacement, and extra fantasy dimensions against the official pinned
reference. They include list/NumPy/Torch inputs, probit/logit likelihoods,
one- and two-dimensional model batches, shared/per-member concentrations,
and shared composite kernels. Native finite differences independently
check all four shared/per-member kernel and Warp ownership combinations.
