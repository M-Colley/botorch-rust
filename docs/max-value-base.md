# Custom max-value entropy acquisition functions

`MaxValueBase` accepts the full base constructor: model, candidate set,
maximum-sample count, posterior transform, Gumbel/Thompson selection,
maximization direction, pending points, and explicit training inputs.
Subclasses implement `_compute_information_gain(X, mean_M, variance_M,
covar_mM)`; the base evaluates native posterior moments, applies the variance
floor, invokes that hook, and averages its fantasy dimension.

Maximum values are sampled by the existing Rust Gumbel/Thompson algorithms
over the candidate and training points, with pending points included for the
initial sampling heuristic. `posterior_max_values` can be replaced explicitly.
`seed` is an additional port-specific option. Custom hooks use the caller's
array operations; native posterior query adjoints preserve their Torch graph.
The existing callable bridge supplies those actual hook values and gradients
to native acquisition optimization. It keeps a weak reference to its owning
Python acquisition object to avoid an ownership cycle.

The four concrete MES/GIBBON classes retain their specialized native
implementations. Direct construction of the abstract base, or a subclass
without an information-gain implementation, raises `TypeError`.
Current base limits follow the acquisition/model interface: unbatched,
single-output native models and unbatched candidate/training sets.

Six focused tests compare a real custom hook's values and query gradients
against official BoTorch in both sampling and optimization directions, check
native optimizer execution against a dense grid, and enforce the subclass
requirement. Another 37 existing MES/GIBBON tests pass; three optional cases
remain skipped by their established conditions.
