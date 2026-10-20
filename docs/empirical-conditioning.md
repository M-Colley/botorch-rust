# Empirical conditioning and fantasies

Empirical one-dimensional GPs condition their native residual GPs on new
observations after subtracting the fixed historical mean at the new inputs.
Both independent and correlated outputs retain the historical basis and
shared covariance parameters. Fixed-noise updates require new observation
variances, and inferred-noise models also accept explicit conditioning
variances. The original model and existing posterior snapshots are preserved.

Fantasy sampling uses the actual noisy native posterior. Fixed-noise models
use average observed noise (per output for independent outputs), matching
their native predictive likelihood. Native model snapshots hold the full
fantasy batch, including multidimensional sample shapes. Subsequent queries
broadcast against that batch; native query derivatives and immutable
posterior snapshots remain available. Training-input and target properties
expose the real expanded data and the independent-output batch dimension.

Shared-parameter replacements validate and update every fantasy leaf together.
Refitting a fantasy batch is explicitly unsupported; its optimizer needs a
joint shared-parameter objective. New training-data derivatives and fantasy
sampling with caller-specified observation noise are also not implemented.
The supported default fantasy path handles both inferred and fixed noise.

Native conditional moments are checked against independent dense Gaussian
conditioning, including the historical-mean contribution, correlated outputs
and explicit observation noise. Python tests compare official conditioning,
fantasy batches, fixed-noise layouts, snapshots, empty queries and query
gradients. Multiple sample dimensions are flattened for the official ExactGP
reference, which accepts only one new fantasy batch dimension.
