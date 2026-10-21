# Regularized acquisition graphs

Torch models support `PenalizedAcquisitionFunction`,
`PriorGuidedAcquisitionFunction`, and `ProximalAcquisitionFunction`, including
construction with `raw_acqf=`. Wrapping an acquisition keeps candidate,
model-parameter, penalty, and prior graphs on the original device.

Prior weights apply to each MC utility before the q and sample reductions.
Log weights, prior exponents, and pending points follow the reference protocol.
Proximity weighting supports raw and transformed coordinates and a softplus
for negative acquisition values. The last training point remains connected to
its data graph. The implementation accounts for the port storing raw training
inputs, whereas the reference exposes transformed inputs after evaluation.

Sixteen new CPU/CUDA cases compare values, all first derivatives, and
Hessian-vector products with the pinned reference using a separately specified
exact GP law. They also exercise actual port models and keyword dispatch.
Together with the existing feature/projection wrapper suite, 30 cases pass.
