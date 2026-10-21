# Tensor costs and inverse-cost utilities

Affine and fixed cost models evaluate Tensor inputs with device-resident Torch
operations and preserve full autograd. When Torch is already imported, an
AffineFidelityCostModel owns a persistent `weights` buffer in the construction
default dtype, matching the pinned official model. FixedCostModel preserves the
identity and graph of an input Tensor as its `fixed_cost` buffer. They are
ordinary Torch modules: `.to(...)`, buffer registration, state dictionaries,
explicit buffer gradients, and replacing a buffer with a Parameter work normally.
Default-dtype rounding occurs before a later cast, including native array calls.
Tensor-valued affine weights or offsets also retain their original graphs.

Fixed Tensor costs retain their buffer's dtype and device, like upstream; move
the model to the desired device with `.to(X)` before using it for GPU cost
weighting. Affine weights follow the input's dtype/device during evaluation.
Deterministic posterior samples are exact copies of the cost values, with zero
variance and no artificial jitter. Output subsets keep their Tensor graph.

InverseCostWeightedUtility constructs no native model handle for a Tensor
forward call. It uses the cost model's actual posterior mean, or actual posterior
samples from the supplied sampler when `use_mean=False`. A Torch GP cost model's
training data, hyperparameters and queries therefore remain connected through
the Gaussian posterior and sample root, including higher derivatives. The
utility registers a Torch cost model and Torch objective as real child modules.

The default objective selects a single cost output or sums multiple cost
outputs. Evaluation masks must have identical rows, as in the pinned reference
`b0dd01be61d517c387e234903a6e2cc38c057ea2`. Selected output indices stay on device;
no `.tolist`, `.cpu`, NumPy conversion, or detachment is used in the Tensor cost
path. Cost-objective values are checked for strict positivity before summing over
candidate points. Positive utility differences are divided by cost; nonpositive
differences are multiplied by cost. Log utilities subtract the log total cost.
The supplied cost objective receives costs, with its optional X argument left
at None, matching the reference utility.

List/NumPy inputs keep the Rust implementation. Native handles are created only
when those inputs or a native acquisition require them. Mutable deterministic
Tensor buffers are read when constructing those handles; native acquisitions
retain their ordinary construction-time snapshot semantics. Native array
stochastic cost weighting retains its previous deterministic-cost scope. A
Tensor GP cost model requires a Tensor-aware acquisition rather than a native
CPU callback. A CPU/native GP cannot silently serve as a GPU cost model.
List-only construction/import does not import Torch.

Tests cover CPU/CUDA, float32/float64, values and first/second gradients versus
the reference, finite-difference gradcheck/gradgradcheck for inputs, weights,
offsets and deltas, shared-normal GP cost parity, live GP parameter/data
gradients, deterministic samples, evaluation masks and errors, native array
compatibility, and end-to-end multi-fidelity KG with deterministic or stochastic
costs and full second derivatives.
