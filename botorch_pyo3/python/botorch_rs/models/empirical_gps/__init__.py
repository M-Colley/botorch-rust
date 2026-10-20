"""Empirical Gaussian-process models with native interpolation and inference."""

from botorch_rs.models.empirical_gps.em_empirical_gp import (
    EMEmpiricalGaussianProcess,
    EMEmpiricalMarginalLogLikelihood,
    EMPriorContainer,
    build_shared_gp_model_list,
    pretrain_em_prior,
)
from botorch_rs.models.empirical_gps.empirical_1d_gp import (
    BaseAugmentedEmpiricalKernel,
    EmpiricalOneDimensionalGP,
    EmpiricalOneDimensionalKernel,
    EmpiricalOneDimensionalMean,
)
from botorch_rs.models.empirical_gps.multioutput_empirical_1d_gp import (
    MultiOutputEmpiricalOneDimensionalGP,
    MultiOutputEmpiricalOneDimensionalKernel,
    MultiOutputEmpiricalOneDimensionalMean,
)
from botorch_rs.models.empirical_gps.multitask_empirical_1d_gp import (
    MultiTaskEmpiricalOneDimensionalGP,
    MultiTaskEmpiricalOneDimensionalKernel,
    MultiTaskEmpiricalOneDimensionalMean,
)
from botorch_rs.models.empirical_gps.utils import trace_matched_shrinkage

__all__ = [
    "build_shared_gp_model_list",
    "EMEmpiricalGaussianProcess",
    "EMEmpiricalMarginalLogLikelihood",
    "EMPriorContainer",
    "BaseAugmentedEmpiricalKernel",
    "EmpiricalOneDimensionalGP",
    "EmpiricalOneDimensionalKernel",
    "EmpiricalOneDimensionalMean",
    "MultiOutputEmpiricalOneDimensionalGP",
    "MultiOutputEmpiricalOneDimensionalKernel",
    "MultiOutputEmpiricalOneDimensionalMean",
    "MultiTaskEmpiricalOneDimensionalGP",
    "MultiTaskEmpiricalOneDimensionalKernel",
    "MultiTaskEmpiricalOneDimensionalMean",
    "pretrain_em_prior",
    "trace_matched_shrinkage",
]
