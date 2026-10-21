# Native fully Bayesian Gaussian outcome compositions

Scalar native fully Bayesian models now use the existing native outcome
pipeline for single-output `Standardize` transforms, behavior-preserving
subclasses, and chains of these transforms. Selected output `[0]` and the
empty selection are also supported. The latter are explicit port extensions:
the pinned reference's `Standardize.untransform_posterior` rejects any
non-`None` output selection. Nonlinear transforms cannot produce the required
Gaussian mixture and remain outside this family contract.

For native fully Bayesian multitask models, the fitted composition is folded
into its scalar inverse map `raw = offset + scale * transformed`. A native
setter installs this map after the GP has been constructed from transformed
targets and clamped transformed noise. It preserves the native solve, restores
raw target/noise metadata, and applies the same inverse map to posterior
covariances and native query derivatives. New conditioning targets are mapped
back through that fitted transform; their explicit noise remains in
transformed units, as in BoTorch. The existing native Standardize machinery
requires a finite **positive** scale. This change does not accept arbitrary
custom signed affine maps or point-dependent stratified transforms.

Scalar acquisitions obtain native member handles with the same outcome
pipeline. This avoids accepting an outcome chain whose posterior values look
correct while its native acquisition constructor still rejects the transform.
The public original transform and its fitted unbatched statistics are retained;
MCMC members share those statistics without refitting on new observations.

Tensor conversion accepts canonical inherited Standardize behavior. Structural
attributes alone no longer cause an explicit caller-owned Torch module to be
silently replaced with canonical standardization. Canonical reference modules
are recognized by their actual inherited method definitions; user overrides
with the device transform `fit(Y)` protocol remain the original module and
retain its parameters. Other overridden Standardize-like modules fail with an
explicit protocol error. This is a documented interoperability boundary, not
support for arbitrary source outcome subclasses.

Twenty-five focused tests compare native and Tensor posterior laws, covariance,
fixed/inferred noise, raw metadata and conditioning for scalar and multitask
models. Separate native-versus-Tensor acquisition values/query gradients and
official scalar/multitask chain comparisons use unchanged strict tolerances.
Custom fitted Tensor-module identity, constructor fitting, numerical behavior,
and first/second query/parameter graphs are checked on CPU/CUDA. An independent
native regression checks the affine law, conditioning, metadata and query
adjoints. The 24-case initial test set plus existing prior/scalar tests passed
152 cases; the additional official multitask chain control passes separately.
