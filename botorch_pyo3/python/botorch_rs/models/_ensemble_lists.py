"""Lists of ensemble models (``ModelListGP`` / ``ModelList`` of fully
Bayesian or MAP SAAS ensemble GPs): their members are the lists of the
members of the same index (BoTorch's ensemble dimension ``MCMC_DIM``,
broadcast over the list), which acquisition functions average over."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from botorch_rs.exceptions.errors import UnsupportedError


def _ensemble_parts(model_list: Any) -> list[list[Any] | None] | None:
    """The members of every sub-model (``None`` for a sub-model that is not
    an ensemble), or ``None`` if no sub-model is an ensemble."""
    parts: list[list[Any] | None] = []
    sizes = set()
    for m in model_list.models:
        if not getattr(m, "_is_ensemble", False):
            parts.append(None)
            continue
        members_of = getattr(m, "_ensemble_members", None)
        if members_of is None:
            raise UnsupportedError(
                "botorch_rs supports lists of ensembles of GPs (fully Bayesian or MAP SAAS "
                f"ensembles), not of {type(m).__name__}."
            )
        members = members_of()
        sizes.add(len(members))
        parts.append(members)
    if not sizes:
        return None
    if len(sizes - {1}) > 1:
        raise UnsupportedError(
            f"The ensembles of a model list have different sizes {sorted(sizes)}: BoTorch "
            "broadcasts their ensemble dimensions, which requires equal or singleton sizes."
        )
    return parts


def members_method(model_list: Any) -> Callable[[], list[Any]]:
    """``model_list._ensemble_members`` (``AttributeError`` if no sub-model
    is an ensemble)."""
    if not any(getattr(m, "_is_ensemble", False) for m in model_list.models):
        raise AttributeError("_ensemble_members")

    def members() -> list[Any]:
        parts = _ensemble_parts(model_list)
        assert parts is not None
        size = max(len(p) for p in parts if p is not None)
        cls = type(model_list)
        return [
            cls(
                *[
                    p[0 if len(p) == 1 else s] if p is not None else m
                    for m, p in zip(model_list.models, parts, strict=True)
                ]
            )
            for s in range(size)
        ]

    return members


def acquisition_method(model_list: Any) -> Callable[[Any], tuple[list[Any], None, Any]]:
    """``model_list._ensemble_acquisition`` (``AttributeError`` if no
    sub-model is an ensemble)."""
    members_of = members_method(model_list)

    def acquisition(sampler: Any) -> tuple[list[Any], None, Any]:
        from botorch_rs.acquisition.monte_carlo import DEFAULT_MC_SAMPLES
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(DEFAULT_MC_SAMPLES,))
        return members_of(), None, sampler

    return acquisition
