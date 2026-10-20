# Latent context embedding additive covariance

`LCEAKernel` and `LCEAGP` implement native latent-context additive exact
Gaussian processes. The covariance sums over every ordered context pair:
`sum_cd amplitude[c] * amplitude[d] * RBF_embedding(E[c], E[d]) *
RBF_spatial(x[context[c]], y[context[d]])`.

The spatial ARD vector is shared by all context pairs. A separate shared
ARD vector acts on context embeddings. Both use BoTorch's dimension-scaled
lengthscale priors and constraints. Context amplitudes use the positive
constraint and `GammaPrior(2, 15)`, with one amplitude per context or one
amplitude multiplied by known context weights. Each learned categorical
feature has its own embedding table; fixed features may be appended or
used alone. No parameter or prior is duplicated.

Values, context covariance, input derivatives, hyperparameter derivatives,
exact inference, and fitting execute in Rust. Embedding max-norm updates
reuse LCEMGP's native `embedding_renorm`. As in torch and native LCEMGP,
the optimizer evaluates the objective and gradient at renormalized table
values while treating that normalization as a forward-pass update, rather
than differentiating the normalization map. Exposed weights are normalized
by kernel/model forward calls.

`LCEAGP` accepts unbatched `n x d` inputs and `n x 1` outcomes, matching the
official contextual model's documented training scope. It supports inferred
or observed noise and leaves outcomes on their original scale. Lists,
NumPy arrays, and torch tensors are accepted. Dataset construction includes
`train_Yvar=None` when observed noise is absent, fixing the same required-
argument omission as `SACGP.construct_inputs`.

## Official reference and corrected behavior

The audited reference is official BoTorch **b0dd01be6**.

* Official LCEA's permutation helper sorts each context's indices and may
  leave contiguous contexts in physical input order. This loses the feature
  pairings and context ordering declared by the decomposition. For example,
  `{"a": [0, 2], "b": [3, 1]}` should pair columns `(0, 3)` and `(2, 1)`;
  sorting instead pairs `(0, 1)` and `(2, 3)`. Native evaluation uses each
  context's listed indices directly. The four public ordering helpers retain
  their reference outputs for compatibility; kernel mathematics uses the
  actual decomposition. An upstream fix is to use the flattened listed
  indices as the permutation and skip permutation only when that exact
  sequence equals the original column order.
* Official categorical tables are sized by the number of unique values but
  indexed by the original values. Labels such as 10 and 20 then index outside
  a two-row table. This implementation maps each feature's unique labels to
  contiguous row indices. An upstream fix is the same unique-value mapping.
* Official learned embedding batches use categorical feature column zero
  for every table and concatenate different tables along the batch axis.
  Batched learned kernel embeddings are explicitly unsupported here until
  their complete parameter interface is implemented. Use independent
  unbatched models or fixed context features. This implementation does not
  claim parity with the broken multi-table batch path.

The public diagonal kernel call is correct in the reference: GPyTorch
extracts the vector from LCEA's diagonal operator. Native diagonal values
are the diagonal of the full native covariance.
