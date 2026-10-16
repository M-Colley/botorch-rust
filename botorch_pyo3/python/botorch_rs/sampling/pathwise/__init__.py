"""Pathwise posterior sampling (``botorch.sampling.pathwise``): functions drawn
from (an approximation of) a GP posterior that can be evaluated anywhere,
cheaply and with exact gradients (Wilson et al., 2020, 2021).

Supported: ``draw_matheron_paths`` and ``get_matheron_path_model`` for
botorch_rs ``SingleTaskGP`` (one or several outputs) and ``ModelListGP``
models, ``draw_kernel_feature_paths`` (random Fourier features of RBF and
Matérn kernels, with ``active_dims``, with or without ``ScaleKernel``) and
``gaussian_update``.
The random numbers come from botorch_rs's generators (keyword-only ``seed``)
or are passed explicitly (keyword-only ``draws``); given BoTorch's draws, the
paths equal BoTorch's to rounding. ``gen_kernel_features`` and
``KernelEvaluationMap`` are part of the paths computed in Rust and not
exposed in Python.
"""

from botorch_rs.sampling.pathwise.features import FeatureMap, KernelFeatureMap
from botorch_rs.sampling.pathwise.paths import (
    GeneralizedLinearPath,
    PathDict,
    PathList,
    SamplePath,
)
from botorch_rs.sampling.pathwise.posterior_samplers import (
    MatheronPath,
    draw_matheron_paths,
    get_matheron_path_model,
)
from botorch_rs.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
from botorch_rs.sampling.pathwise.update_strategies import gaussian_update

__all__ = [
    "FeatureMap",
    "GeneralizedLinearPath",
    "KernelFeatureMap",
    "MatheronPath",
    "PathDict",
    "PathList",
    "SamplePath",
    "draw_kernel_feature_paths",
    "draw_matheron_paths",
    "gaussian_update",
    "get_matheron_path_model",
]
