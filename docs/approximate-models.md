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
strategies other than the whitened Cholesky family. Higher-order GP latent
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
