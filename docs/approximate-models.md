# Approximate and higher-order GP models

`SingleTaskVariationalGP`, `ApproximateGPyTorchModel`, `VariationalELBO`,
the whitened Cholesky variational strategy, and the greedy variance and
improvement inducing-point allocators run natively in Rust. They support
independent output tasks, kernel trees, constant and zero means, Gaussian
likelihoods, static input transforms, `Standardize`, and native L-BFGS-B and
Adam fitting. The 132 Python API, parity and randomized equivalence tests
compare against installed BoTorch 0.18.1; Rust fixtures test the same numerical
formulas independently of Python.

Variational [external-data loss closures](variational-data-loader-closures.md)
evaluate actual DataLoader minibatches and selected parameter gradients in
Rust, including epoch cycling and multiple outputs. This does not change
the full-data closures used by the default fitting convenience routines.

CPU Torch variational models expose persistent `torch.nn.Parameter` leaves.
The same objects are returned from `parameters()`, `named_parameters()`, the
variational mean / Cholesky / learned inducing-point properties, and raw
kernel, constant-mean and likelihood properties. Torch optimizer updates are
synchronized into Rust before evaluation; native fits update those held leaves
without changing their identities. Public mean, kernel, noise, variational and
inducing-point setters update held leaf references immediately, retaining their
trainability flags and saved native graph snapshots. Frozen leaves remain fixed during native
Adam and L-BFGS-B fits, including prior restarts. Copies preserve their flags
and have independent leaves.

`VariationalELBO` connects its native analytic parameter gradients to Torch
autograd for both stored training data and caller-supplied minibatches. For
`combine_terms=False`, the expected log likelihood, scaled KL divergence and
scaled log prior each have their own native gradients, so arbitrary weighted
combinations work. Saved ELBO graphs retain immutable native parameter and
minibatch snapshots. The latent output of `model.model(X)` also retains its
native source and input values before any lazy moment evaluation. An ELBO
called with that saved output uses the same source for its expected likelihood,
KL and prior terms, and connects to the original parameter leaves. Calling
`mll()` without an output evaluates the current model. Multitask logical noise
properties synchronize raw edits and expose native constraint derivatives.
These are first-order parameter derivatives; this does not
add higher-order autograd, batched training models, learned input transforms,
or automatic parameter derivatives of the variational posterior distribution.
Non-Torch arrays retain the native explicit-gradient API. Native numerical
calculation remains CPU float64, with results converted to the caller's dtype.

Acquisition values and query derivatives remain available with persistent
variational leaves, including via `torch.autograd.grad(value, X)`. Acquisition
parameter backward is explicitly unsupported for these models; persistent
ELBO leaves do not imply posterior or acquisition parameter pullbacks.
Analytic acquisitions use the model's marginal variance, preserving the
BNN kernel's [lazy diagonal convention](infinite-width-bnn-diagonals.md).

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
mll = VariationalELBO(model.likelihood, model.model, num_data=len(train_X))
for X_batch, Y_batch in loader:
    optimizer.zero_grad()
    loss = -mll(model.model(X_batch), Y_batch.squeeze(-1))
    loss.backward()
    optimizer.step()
```

Inputs to the inner `model.model` and targets passed to the ELBO must already
be in the model's transformed training spaces, as in GPyTorch. The parity
suite checks full-data and minibatch gradients against BoTorch 0.18.1, including
independent outputs, fixed inducing points, kernel and mean priors, transforms,
separate terms and external Torch Adam steps. Rust finite differences also
check separately weighted terms independently of the Python reference.

The inner `model.model.forward(X)` evaluates the GP prior mean and kernel,
with independent output tasks retained as a batch dimension. Calling
`model.model(X, prior=True)` also bypasses variational initialization and
converts the output-task batch into interleaved multitask events. Both paths
use the supplied inputs directly, accept broadcastable query batches, and
retain immutable native parameters and query values. The outer `model(X)`
continues to apply its training input transform and evaluate `q(f)`.
Prior moments currently expose values without automatic Torch parameter or
input derivatives, like the variational output's direct moment properties.
`VariationalELBO` explicitly rejects prior outputs: its native ELBO and
derivatives require `q(f)` from `model.model(X)`. The port does not silently
replace a supplied prior with the variational posterior inside the loss.

One reference edge case needs a workaround: with several outputs and
`ZeroMean(batch_shape=())`, BoTorch 0.18.1 / GPyTorch 1.15.2 raises
`RuntimeError: mean should be a matrix or a batch matrix (batch mode)` for
`model.model(X, prior=True)` at unbatched `X`. The zero mean has shape `(n,)`,
while its covariance has batch shape `(m,)`; GPyTorch's
`MultitaskMultivariateNormal.from_batch_mvn` permutes the unexpanded mean.
The port returns the valid zero-mean multitask prior. Using
`ZeroMean(batch_shape=torch.Size([m]))` fixes the reference configuration
and provides the parity comparison. A general upstream solution is to expand
`batch_mvn.mean` to `batch_mvn.batch_shape + batch_mvn.event_shape` before
computing its dimensions and permuting it in `from_batch_mvn`. Official
BoTorch b0dd01be's inner `forward` uses the same mean/kernel prior construction;
the conversion defect is in the installed GPyTorch dependency.

`HigherOrderGP` uses Kronecker factor eigendecompositions for its exact
likelihood, analytic likelihood gradient, posterior means and variances,
joint posterior derivatives and Matheron samples. Its Python API supports
tensor-shaped outputs, `FlattenedStandardize`, static input transforms,
Gaussian noise, learned or fixed latent coordinates, GP latent initialization,
conditioning and fantasy samples. The default L-BFGS-B and Adam paths run in
Rust. Seeded random draws come from the caller's library; factorization and
sampling calculations run in Rust.

Higher-order numerical tests compare likelihood values and gradients, latent
posterior moments and identical-base-sample Matheron draws against the
installed BoTorch 0.18.1 distribution. Rust tests also compare against dense
Kronecker reference matrices and finite differences. These references must
be distinguished from the separate local BoTorch audit fork: its
`fix/audit-correctness-fixes` branch includes additional fixes, including
`0629df4bc` for missing higher-order observation noise.

The port retains that observation-noise correction: noisy predictive variance
equals latent variance plus likelihood noise, scaled by the outcome transform.
Tests verify this formula directly because official BoTorch 0.18.1 omits the
noise in this path. For several coordinates per latent output dimension,
the port sizes the default kernel's ARD parameter to the coordinate count.
Official BoTorch 0.18.1 instead creates an ARD=1 kernel and raises during
training; explicitly supplying dimension-correct kernels is its workaround.
The regression test uses that workaround for a valid numerical comparison.

Current limitations are explicit errors: batched training data, learned input
transforms such as `Warp`, non-Gaussian variational likelihoods, and variational
strategies other than the whitened Cholesky family. `ZeroMean` accepts an empty
batch shape or the independent-output augmentation shape; extra variational
model batch dimensions are explicitly rejected. Higher-order GP latent
initialization with a GP prior currently requires one coordinate per output
dimension. Higher-order fantasy models support unbatched query inputs; repeated
conditioning of an existing fantasy batch and posterior output selection are
not supported. These restrictions are not evidence of full BoTorch coverage.

Run the relevant checks with:

```text
cargo test -p botorch_core --release --lib models::higher_order_gp::tests
cargo test -p botorch_core --release --test approximate_10c_parity
python -m pytest botorch_pyo3/tests/test_10c_approximate.py botorch_pyo3/tests/test_parity_10c_approximate.py botorch_pyo3/tests/test_equivalence_10c_approximate.py botorch_pyo3/tests/test_10c_higher_order.py
```
