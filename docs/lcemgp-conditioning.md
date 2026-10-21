# Native LCEM conditioning and fantasies

The native `LCEMGP` and `ProvidedEmbeddingLCEMGP` implement
`condition_on_observations(X, Y, noise=None)` and `fantasize(X, sampler)`.
Inputs include the task column and targets are in the original outcome space.
Each update keeps the fitted outcome-transform statistics, context tables,
learned embeddings, task kernel and current hyperparameters. It does not fit
new standardization statistics to fantasy observations.

Explicit observation-noise variances are in the outcome-transformed space,
matching the multitask conditioning protocol. They are retained per row in
the Rust model and affect both predictive means and covariances, including
after later hyperparameter replacements. Learned task noise remains the
prediction noise for future observations. Explicitly fixed rows do not
contribute derivatives to learned-noise parameters in the native MLL.

Input and target batches broadcast into a native model batch. Fantasy sample
dimensions become model batch dimensions, and an already conditioned model
can be conditioned again. Multiple target sets at common inputs and noise
share one Rust covariance factor. Batches preserve shared hyperparameters;
public replacements validate copied native states and then update all members
together. Previously returned posteriors retain their original native snapshots.
These operations support nested lists, NumPy arrays and CPU Torch tensors,
without changing the numerical backend or requiring optional array packages.

Fantasy sampling evaluates the noisy posterior, calls the supplied sampler,
and conditions on its observations. As in the source multitask posterior,
an explicit `observation_noise` tensor is unsupported. Empty input arrays keep
the training data and expand the fantasy batch; nested empty lists cannot
express the required trailing feature dimension. Conditioning requires
evaluation mode. Invalid task labels, nonfinite observations, negative noise
and incompatible shapes are rejected before installing a new state.

The Tensor LCEM backend already inherits device-resident conditioning and
fantasies. Native updates explicitly reject graph-attached observation data
and `propagate_grads=True`; use `backend="torch"` for derivatives through new
training data. A single conditioned native model remains fittable. Refitting
a shared-parameter native fantasy batch is an explicit remaining boundary;
MLL values and gradients of its summed members are available, and public
hyperparameter setters remain supported.
The existing native acquisition constructors require unbatched models;
single conditioned LCEM models work with them, while fantasy-batch acquisition
evaluation remains available through the Tensor backend.

Rust tests compare updates with the independent dense Gaussian conditioning
identity and check all MLL parameter gradients with finite differences,
including explicit row noise. Python tests additionally exercise source
conditional parity, fitted transforms, learned and supplied embeddings,
batched/nested updates, actual normal-sampler replay, shared parameter updates,
immutable snapshots, native acquisitions and optional-package-free execution.
