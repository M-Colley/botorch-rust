# Device categorical, fidelity and task kernels

`CategoricalKernel`, `DownsamplingKernel`, `ExponentialDecayKernel`,
`LinearTruncatedFidelityKernel`, and `PositiveIndexKernel` accept
`backend="torch"`. A CUDA `device` selects this backend automatically. Native
kernel descriptors supplied to a Torch GP are converted once at construction.
Their covariance, priors, parameter optimization, and first and higher
derivatives remain on the Tensor device. Categorical and task indices are
discrete; continuous derivatives with respect to their labels are undefined.

The factories retain the public kernel class, actual persistent Torch parameters,
constraints, arbitrary kernel/input batch broadcasting, active dimensions, and
aligned diagonal evaluation. Categorical, decay, downsampling and task kernels
support feature batches. Linear truncated fidelity retains its physical fidelity
axis contract, both one- and two-fidelity formulas, shared child identities, and
user-supplied differentiable Torch subkernels. Dotted `initialize` names reach
the actual child without replacing its parameter. User modules retain their own
autograd capabilities; a GPyTorch module using `cdist` may still reject higher
derivatives independently of the port.

Positive task covariance supports positive low-rank factors, diagonal variance,
optional target-unit normalization, scalar correlation priors, and matrix LKJ
priors. LKJ log density is evaluated once per correlation matrix, plus the sum
of marginal-scale log densities, matching the corrected native law. Density,
MLL fitting, Cholesky, and matrix-prior draws use device Tensor operations with
first and higher parameter derivatives. No runtime numerical path calls BoTorch
or transfers a covariance matrix to the CPU.

Prior sampling is parameter initialization. A general LKJ covariance draw cannot
usually be represented by an elementwise-positive fixed-rank factor, especially
with target-unit normalization. `sample_all_priors` therefore projects such a
draw through its Cholesky factor, positive factor floor and positive residual
diagonal, before the kernel's normalization. The resulting kernel is positive
definite, but this projected initialization is not an exact pushforward sample
from the unrestricted LKJ covariance distribution. Scalar correlation draws
likewise use the existing positive-correlation projection convention.

`test_torch_kernel_catalogue.py` covers CPU and CUDA official-source values,
parameter gradients, full Hessian-vector graphs, feature batches, safe invalid
task indices, shared-tree conversion, initialization, actual Adam updates and
custom child priors. Matrix density is independently compared with the native
Rust scalar law; no comparison relies on GPyTorch's previously reported
vector-valued LKJ covariance-density defect. Source reference is official BoTorch
`b0dd01be61d517c387e234903a6e2cc38c057ea2`.
