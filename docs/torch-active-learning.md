# Feasibility and active learning on the Tensor backend

`qLogProbabilityOfFeasibility`, `qNegIntegratedPosteriorVariance` and
`PairwiseMCPosteriorVariance` have device-resident implementations. They call
the port's model and posterior protocols and retain the complete Torch graph;
they do not delegate to BoTorch acquisitions or copy numeric tensors to CPU.

The public `get_sampler` dispatcher recognizes the port's Tensor Gaussian
distribution, including transformed and list posteriors. `ListSampler` and
public `PosteriorList` concatenate all-Tensor components with `torch.cat`,
preserving their device and graphs while retaining the existing equal-batch
validation. Default feasibility and pairwise samplers are selected lazily from
the actual posterior and reused.

* Log feasibility evaluates every supplied constraint on joint posterior
  samples, sums log indicators, reduces over candidate points with the selected
  smooth/fat maximum, and averages in log space over samples. Tensor constraint
  temperatures retain mixed derivatives. The objective argument is unused as
  in the reference, including for multioutput models.
* Pairwise variance subtracts adjacent sampled points, evaluates the supplied
  link objective, computes unbiased variance over the first sample axis and
  averages over pairs. Positive even candidate counts are required. The first
  sample-axis convention follows the reference; a usual one-dimensional sample
  shape gives one value per model/query batch.
* Integrated variance calls the actual `fantasize` and `posterior` methods of
  the port model. This preserves supplied sampler, fixed-noise, pending-point
  and posterior-transform behavior. For an exact Gaussian model its value is
  independent of fantasy targets; tests also compare an independently formed
  joint Gaussian Schur complement. Multioutput models require a scalarizing
  posterior transform. Integration-point batch dimensions are flattened into
  the common integration pool, matching the pinned reference.

Integrated variance requires a model with the corresponding fantasy protocol.
The dense Tensor posterior currently forms exact covariance matrices; this
implementation does not claim the reference's lazy-covariance performance.

`test_torch_active_learning.py` checks CPU/CUDA values, signed batch-weighted
first derivatives, Hessian pullbacks through training data, parameters and
integration points, repeated backward calls, real PairwiseGP comparisons and
host-transfer guards. Independent SingleTaskGP comparisons include both
float32 and float64.

Two existing reference limitations require explicit distinctions in those
checks. The default qNIPV fantasy covariance cache gives wrong query gradients;
the supported `fast_pred_var(False)` setting restores agreement, including
second derivatives. See [the original reproduction](single-objective-acquisition.md).
The sampled-posterior reference can also have incorrect second derivatives
while values and first derivatives agree. These tests compare the acquisition
Hessian on a shared posterior graph, and independently compare the port's full
Hessian with centered differences of the reference's first gradient. See
[sampled-posterior evidence](torch-backend-acquisitions.md#reference-sampled-posterior-second-derivatives).
