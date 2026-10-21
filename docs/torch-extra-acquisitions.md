# Additional acquisition device paths

The Torch registry now includes analytic `ExpectedHypervolumeImprovement`,
`MOMF`, `qLogNParEGO`, and fantasy-based `NoisyExpectedImprovement` and
`LogNoisyExpectedImprovement`. They use the port's posterior calculations and
retain their Torch graphs. The separate active-learning and regularization
notes describe the other newly registered families.

Analytic EHVI integrates independent Gaussian marginals over hypercells.
Distributing the sum of its two factors before multiplication avoids an
explicit enumeration of all `2**m` products. MOMF combines the actual MC
hypervolume calculation with the configured cost model, including cost
derivatives. ParEGO composes Chebyshev scalarization with actual noisy LogEI;
its baseline normalization is fixed without gradients, matching the reference.
The public scalarization helper itself retains weight, observation-bound and
query derivatives when called directly with Tensors.

The scalarization/hypervolume regression run passes 153 cases, with two skips
for the reference's unsupported negative weights without observations. New
CPU/CUDA checks compare values, all first derivatives and Hessian-vector
products with the independent reference acquisition formulas. They cover
two/three outcomes, positive/negative scalarization weights, actual cost
gradients, and all model parameters.

Analytic noisy improvement constructs actual noiseless fantasy models and
averages their EI or LogEI. Their copied hyperparameters and fixed fitted
transform statistics follow the reference factory's semantics. The fantasy
noise is `1e-7` in transformed coordinates, as in the source assignment after
construction. Eighteen CPU/CUDA cases compare fantasy laws, values, query
gradients and Hessians against BoTorch, and exercise observed-input and
fantasy/original-parameter graphs with host-buffer conversions forbidden.
