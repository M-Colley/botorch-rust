# Torch backend parameter modules

The optional Torch model backend uses port-owned `torch.nn.Module` components
from `models/_torch_backend_modules.py`. It does not bind parameters to the
CPU/f64 parameter mirror. Parameters and buffers move with `.to(...)`, and
ordinary Torch optimizers consume the same parameter objects used in inference.
Covariance evaluation dispatches through Rust `tensor_kernel` to the installed
Torch runtime, retaining the input device, dtype, and autograd graph.

The current converter supports RBF, Matérn (0.5, 1.5, 2.5), RQ, Periodic,
Cosine, Linear, Polynomial, Scale, Additive and Product kernels. ARD, active
dimensions, model batches, feature batches, shared kernel objects and configured
constraints/priors retain their respective parameter layouts. Custom Torch
kernel and mean modules can also be supplied directly. Other port kernel
families raise `UnsupportedError` rather than using a CPU numerical fallback.

ConstantMean, ZeroMean, Gaussian likelihoods, fixed observation noise,
Normalize, Warp and ChainedInputTransform are implemented as Torch components.
Warp concentrations are actual Parameters with optimizer bounds. Normalize and
Standardize expose differentiable `fit` methods; a model can rebuild fitting
statistics from its original training tensors before each differentiable
evaluation. Query evaluation reuses fitted statistics. Standardize converts
independent output covariance blocks with shape `(..., m, q, q)`.

The Gaussian likelihood registers its learned parameter at
`noise_covar.raw_noise`; its convenience `raw_noise` accessor refers to that same
leaf. Calling the likelihood on function samples returns conditional Normal
distributions. Calling it on a latent Gaussian adds noise before any required
Cholesky factorization. Fixed-noise inspection buffers are detached initial
snapshots; the model recomputes its transformed observation noise from the
original training Tensor for inference and gradients. Noise floors are 1e-4 for
float32 and 1e-6 for float64, matching GPyTorch defaults.

Prior terms reduce parameter event dimensions and preserve model/output batch
axes. Shared kernel/prior objects are visited once. The default LogNormal prior
retains its construction-precision variance and log-scale constants, matching
GPyTorch's cached Normal base distribution even when parameters use float64.
The buffers reside on the selected device; parameter log densities retain first
and higher derivatives.

## Verification and reference limitations

`test_torch_backend_modules.py` passes 218 cases on the available CPU and CUDA
device. Tests compare values and first derivatives to the installed GPyTorch
1.15.2 / official BoTorch reference, compare Hessian-vector products to central
differences of reference first derivatives, check full versus diagonal
covariance and finite self-covariance Hessians in float32/float64, and exercise
optimizer updates, held leaf identities, shared parameters and transform
statistics gradients. These checks establish the listed components; they do
not establish GPU/autograd coverage for every model or acquisition family.

Three reference details require care when constructing an independent oracle:

* GPyTorch Matérn evaluations use `torch.cdist`, whose backward does not implement
  a second derivative in the tested Torch runtime. First derivatives are valid;
  central differences of those derivatives check the port's differentiable
  pairwise-distance calculation. A differentiable broadcast difference and
  squared-distance implementation repairs the missing higher-order path.
* GPyTorch's composite lazy covariance wrapper can predict the original feature
  dimension after child active-dimension selection. Eager reference evaluation
  avoids that incorrect shape assertion while retaining the kernel formula.
* GPyTorch PolynomialKernel's offset broadcasting omits the feature axis for
  simultaneous model and feature batches. Distinct per-batch offsets expose
  either a shape error or an incorrect broadcast when dimensions happen to
  agree. The correct offset has an additional singleton feature axis. The
  regression oracle evaluates each selected feature with the unchanged
  non-feature reference kernel and stacks those results.

Prior constructors taking Python floats can also retain cached CPU/float32
transformed-distribution state after GPyTorch `.to(...)`; tests of a general
device-local prior construct the reference from Tensor hyperparameters on the
intended device and dtype. A separate default-prior regression checks the
historical cached-constant behavior to 2e-14 instead of relaxing MLL tolerance.
