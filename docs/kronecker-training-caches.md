# Kronecker GP training caches

Compared against official BoTorch revision `b0dd01be6`.

`KroneckerMultiTaskGP.train_full_covar` returns the actual noiseless
`Kxx ⊗ Ktt` on input-transformed training data. Its dense array is ordered
by training point then task and uses the model's array kind, dtype, and
device. As with this port's kernel calls, this is a dense array rather than
GPyTorch's `LinearOperator`. Native inference retains its existing cached
factors; the extra full noiseless matrix is materialized lazily only when
this public property is requested, then reused.

`predictive_mean_cache` returns the actual existing native solve
`(Kxx ⊗ Ktt + I ⊗ N)⁻¹ (train_targets - train_mean)`, flattened in the same
interleaved order. Targets and means are in the transformed training
outcome space. Access does not refactor or re-solve an already cached model.
Changing model parameters invalidates the native caches. Existing model and
posterior snapshots retain their original quantities, and mutating a returned
array cannot modify the native cache.

Validation includes direct official covariance and solve comparisons for
default, rank-deficient task covariance, correlated task noise, standardized
outcomes, normalized inputs, Matérn/scale and active-dimension kernels;
independent dense solve residuals; all three array kinds; float32/float64;
cache reuse and invalidation; and snapshot independence. Existing active-dims
reference normalization in `multitask_helpers.build_kron_pair` is retained.

The model's existing unbatched-training boundary applies. Returned properties
are detached snapshots, matching BoTorch's default detached test caches;
GPyTorch setting contexts that request hyperparameter graphs are not emulated.
