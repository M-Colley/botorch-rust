# Candidate-generation protocols

The Torch candidate driver supports independent native L-BFGS-B states with
batched acquisition evaluation. Each restart keeps its own line-search and
curvature history. Candidate vectors, scores and gradients cross the solver
boundary; acquisition inference remains on its original device. Eligible
problems select this path by default, and `use_parallel_mode=True` requests it
explicitly. Constraints, timeouts, unsupported options, extra batch dimensions
or unequal free-variable bounds require the serial driver.

The serial driver supports `with_grad=False` and SciPy's numerical-gradient
methods, including acquisition callables without a differentiation graph.
Callback signatures are preserved for SciPy's `intermediate_result` protocol.
SciPy itself validates method-specific options and capabilities. These options
also work through `optimize_acqf`, including native-model acquisitions.

Native candidate generation retains its Rust fast paths for their usual
configurations. Shaped fixed features, custom SciPy callbacks, numerical
gradients and other SciPy methods use the generic candidate loop. Lists,
NumPy arrays and Tensors retain their result type. Native models still compute
their posterior on the CPU; select a Torch model for device inference. Custom
raw-sample generators are invoked during native initialization and retain the
requested output array type.

The focused CPU/CUDA regression run passes 136 cases. New tests compare parallel candidate values and
locations to the official batched driver, optimize nondifferentiable scores
with numerical gradients, exercise callbacks, and check per-restart/per-point
fixed features through both generator interfaces. Closed-form quadratic optima
also verify high-level optimization through lists, NumPy arrays and Tensors.
