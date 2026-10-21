# Custom Torch model protocol

Caller-defined subclasses of `botorch_rs.models.model.Model` support ordinary
Torch module construction, parameters, buffers, child modules, recursive
training/evaluation modes, migration, state dictionaries, deepcopy and pickle.
Import Torch and define a `posterior` method, as with the reference `Model`.
The resulting instance retains the caller's class identity and automatically
selects device acquisitions through the Tensor model protocol. Built-in native
model families retain their separate allocation paths.

The compatibility adapter also preserves existing custom buffer-only classes
and older port constructors that omitted `super().__init__`. No native model or
upstream BoTorch implementation is substituted for the caller's posterior.

`test_custom_model_protocol.py` exercises a custom differentiable Gaussian
posterior on CPU/CUDA. Analytic and MC acquisition values and complete
query/data/parameter first derivatives match the pinned official acquisition
protocols. Analytic higher derivatives match directly. For MC covariance-root
sampling, the known `TriangularLinearOperator` double-backward defect
([linear_operator #152](https://github.com/cornellius-gp/linear_operator/pull/152))
requires an independent oracle: two convergent central differences of the
unmodified source's complete first derivatives verify the port's mixed higher
derivatives. No source patch or tolerance relaxation hides that discrepancy.
The new cases and existing model-buffer/fantasy regressions pass together:
66 CPU/CUDA cases.
