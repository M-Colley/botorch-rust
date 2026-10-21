# Analytic batch probability of improvement

`qAnalyticProbabilityOfImprovement` uses the port's device-resident MVNXPB
solver for Torch models. The Gaussian joint covariance, best-value thresholds,
bound construction, pivoted conditioning and ensemble reduction retain their
Tensor graphs. No Gaussian probability computation is delegated to BoTorch.

Both optimization directions, batched thresholds, scalarized multi-output
posteriors and loaded fully Bayesian ensembles are supported. As in BoTorch,
the probability is the MVNXPB approximation to a multivariate Gaussian tail;
an analytic acquisition requires a Gaussian single-output posterior after any
posterior transform. Pending points are unsupported by the source interface.

The 24 CPU/CUDA cases compare values and complete query, training-data,
parameter and threshold gradients with the pinned official source for q=1–4.
Directional Hessian products use central differences of the source's first
derivatives because its in-place solver has known higher-derivative defects.
Independent gradcheck/gradgradcheck, scalarization-weight graphs, ensemble
reduction and guards against host data transfers provide further checks.
These are focused development results, not a replacement release certificate.
