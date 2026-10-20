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

PairwiseGP currently rejects input chains with learned parameters, including
Warp. Its native chain/tape machinery supports Warp application and its
derivatives, but the PairwiseGP optimizer parameter layout and Laplace
covariance-adjoint contraction do not yet include those parameters.
Removing the rejection alone would be incorrect: transformed training inputs
are currently prepared outside the optimizer closure and would not change
with trial concentrations. End-to-end learned Warp support requires
trial-dependent training transforms, concentration priors and bounds,
updated fit state, and correct batched parameter ownership.
