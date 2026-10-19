# Approximate and higher-order GP models

`SingleTaskVariationalGP`, `ApproximateGPyTorchModel`, `VariationalELBO`,
the whitened Cholesky variational strategy, and the greedy variance and
improvement inducing-point allocators run natively in Rust. They support
independent output tasks, kernel trees, constant and zero means, Gaussian
likelihoods, static input transforms, `Standardize`, and native L-BFGS-B and
Adam fitting. The 132 Python API, parity and randomized equivalence tests
compare against installed BoTorch 0.18.1; Rust fixtures test the same numerical
formulas independently of Python.

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
