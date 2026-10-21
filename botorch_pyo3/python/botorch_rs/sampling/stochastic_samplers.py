"""Samplers that are not base-sample driven
(``botorch.sampling.stochastic_samplers``): ``ForkedRNGSampler`` and
``StochasticSampler``.

BoTorch's ``StochasticSampler`` calls ``posterior.rsample`` at every
evaluation, for stochastic optimization of acquisition functions. Its main use
with ``optimize_acqf`` is with deterministic models (``PosteriorMeanModel``,
``GenericDeterministicModel``), whose samples are the mean whatever the random
numbers: there, botorch_rs's acquisition functions accept it and are exact.
With any other model they raise ``UnsupportedError``, because botorch_rs's
Monte-Carlo acquisition functions are deterministic sample-average
approximations with fixed base samples.
"""

from __future__ import annotations

import math
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.normal import (
    IIDNormalSampler,
    MCSampler,
    NormalMCSampler,
    SobolQMCNormalSampler,
    _sample_shape,
    _sampler_seed,
)


class ForkedRNGSampler(MCSampler):
    r"""A sampler using ``torch.fork_rng`` to enable replicable sampling
    from a posterior that does not support base samples
    (``botorch.sampling.stochastic_samplers.ForkedRNGSampler``).

    NOTE: This approach is not a one-to-one replacement for base sample
    driven sampling. The main missing piece in this approach is that its
    outputs are not replicable across the batch dimensions. As a result,
    when an acquisition function is batch evaluated with repeated candidates,
    each candidate will produce a different acquisition value, which is not
    compatible with Sample Average Approximation.

    ``forward`` calls ``posterior.rsample(sample_shape)`` with torch's
    generator seeded with ``seed`` (restored afterwards), as in BoTorch, so
    the samples are replicable. Like ``StochasticSampler``, it is accepted by
    botorch_rs's Monte-Carlo acquisition functions for deterministic models
    only.
    """

    def __init__(self, sample_shape: Any, seed: int | None = None) -> None:
        self.sample_shape = _sample_shape(sample_shape)
        if seed is None:
            import importlib
            import random

            try:
                torch = importlib.import_module("torch")
            except ImportError:
                seed = random.randint(0, 999999)
            else:
                seed = int(torch.randint(0, 1000000, (1,)).item())
        self.seed = int(seed)
        self.base_samples: Any = None
        self._rs = _rs.SobolQMCNormalSampler(math.prod(self.sample_shape), self.seed)

    @property
    def num_samples(self) -> int:
        return math.prod(self.sample_shape)

    def forward(self, posterior: Any) -> Any:
        r"""Draws MC samples from the posterior in a ``fork_rng`` context.

        Args:
            posterior: The posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        import importlib

        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return posterior.rsample(sample_shape=self.sample_shape)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            return posterior.rsample(sample_shape=self.sample_shape)

    def __repr__(self) -> str:
        return f"ForkedRNGSampler(sample_shape={self.sample_shape}, seed={self.seed})"


class StochasticSampler(MCSampler):
    r"""A sampler that simply calls ``posterior.rsample`` to generate the
    samples (BoTorch's ``StochasticSampler``). It ignores the ``seed``.

    In botorch_rs acquisition functions it is supported for deterministic
    models only (see the module documentation).

    Example:
        >>> sampler = StochasticSampler(sample_shape=torch.Size([1]))
        >>> acqf = qLogExpectedHypervolumeImprovement(
        ...     PosteriorMeanModel(model), ref_point, partitioning, sampler=sampler
        ... )
    """

    def __init__(self, sample_shape: Any, seed: int | None = None) -> None:
        self.sample_shape = _sample_shape(sample_shape)
        # Sized base samples for the acquisition functions; a deterministic
        # model's samples do not depend on them.
        self._rs = _rs.SobolQMCNormalSampler(math.prod(self.sample_shape), _sampler_seed(seed))

    @property
    def seed(self) -> int:
        """The seed (drawn at random if not given, as in BoTorch; not used for
        sampling)."""
        return self._rs.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._rs = _rs.SobolQMCNormalSampler(math.prod(self.sample_shape), int(value))

    @property
    def num_samples(self) -> int:
        return self._rs.num_samples

    def forward(self, posterior: Any) -> Any:
        """Draws MC samples from the posterior (``posterior.rsample``)."""
        return posterior.rsample(sample_shape=self.sample_shape)

    def __repr__(self) -> str:
        return f"StochasticSampler(sample_shape={self.sample_shape})"


def _is_botorch_stochastic_sampler(sampler: Any) -> bool:
    cls = type(sampler)
    return cls.__name__ == "StochasticSampler" and cls.__module__.startswith("botorch.")


def as_stochastic_sampler(sampler: Any) -> StochasticSampler | ForkedRNGSampler | None:
    """``sampler`` as a botorch_rs ``StochasticSampler`` (or
    ``ForkedRNGSampler``) if it is one of botorch_rs or BoTorch (``None``
    otherwise)."""
    if isinstance(sampler, (StochasticSampler, ForkedRNGSampler)):
        return sampler
    if _is_botorch_stochastic_sampler(sampler):
        return StochasticSampler(sampler.sample_shape, seed=getattr(sampler, "seed", None))
    cls = type(sampler)
    if cls.__name__ == "ForkedRNGSampler" and cls.__module__.startswith("botorch."):
        return ForkedRNGSampler(sampler.sample_shape, seed=sampler.seed)
    return None


def as_iid_sampler(sampler: Any) -> IIDNormalSampler | None:
    """A botorch_rs ``IIDNormalSampler`` with the sample shape and seed of
    BoTorch's ``IIDNormalSampler`` ``sampler`` (whose base samples it
    reproduces), or ``None``."""
    cls = type(sampler)
    if cls.__name__ == "IIDNormalSampler" and cls.__module__.startswith("botorch."):
        result = IIDNormalSampler(tuple(sampler.sample_shape), seed=int(sampler.seed))
        result.base_samples = sampler.base_samples
        return result
    return None


def _require_native_sampler(sampler: Any) -> None:
    """Do not erase caller numerical methods, parameters or module hooks."""
    cls = type(sampler)
    public = getattr(cls, "_sampler_public_type", cls)
    canonical = public in (
        IIDNormalSampler,
        SobolQMCNormalSampler,
        ForkedRNGSampler,
        StochasticSampler,
    ) or (
        cls.__module__ in ("botorch.sampling.normal", "botorch.sampling.stochastic_samplers")
        and cls.__name__
        in ("IIDNormalSampler", "SobolQMCNormalSampler", "ForkedRNGSampler", "StochasticSampler")
    )
    message = (
        "Native acquisitions require canonical sampler descriptors without caller methods, "
        "Parameters or hooks. Construct the model with backend='torch' to execute a custom "
        "sampler and retain its module state and autograd graph."
    )
    if not canonical:
        raise UnsupportedError(message)
    for name in (
        "forward",
        "__call__",
        "_construct_base_samples",
        "_get_collapsed_shape",
        "_get_extended_base_sample_shape",
        "_get_batch_range",
        "_update_base_samples",
    ):
        method = getattr(sampler, name, None)
        if name in sampler.__dict__ or (
            method is not None
            and (
                getattr(method, "__qualname__", "").split(".")[0]
                not in (
                    "IIDNormalSampler",
                    "SobolQMCNormalSampler",
                    "NormalMCSampler",
                    "MCSampler",
                    "ForkedRNGSampler",
                    "StochasticSampler",
                    "Module",
                )
                or getattr(method, "__wrapped__", None) is not None
            )
        ):
            raise UnsupportedError(message)
    if (
        getattr(sampler, "batch_range_override", None) is not None
        or getattr(sampler, "_compiled_call_impl", None) is not None
    ):
        raise UnsupportedError(message)
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(sampler, torch.nn.Module):
        if any(True for _ in sampler.parameters()) or any(True for _ in sampler.children()):
            raise UnsupportedError(message)
        if any(
            getattr(sampler, name, None)
            for name in (
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
            )
        ) or any(
            getattr(torch.nn.modules.module, name, None)
            for name in (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
            )
        ):
            raise UnsupportedError(message)
    base = getattr(sampler, "base_samples", None)
    if (
        getattr(base, "requires_grad", False)
        or getattr(getattr(base, "device", None), "type", "cpu") != "cpu"
    ):
        raise UnsupportedError(message)


def _native_normals(sampler: NormalMCSampler) -> NormalMCSampler:
    if sampler.base_samples is not None:
        values, _ = to_flat(sampler.base_samples)
        sampler._rs.set_base_samples(
            (values, (sampler.num_samples, len(values) // sampler.num_samples))
        )
    return sampler


def mc_sampler_arg(
    sampler: Any, rs_model: Any, default_shape: tuple[int, ...]
) -> NormalMCSampler | StochasticSampler | ForkedRNGSampler:
    """The sampler of a Monte-Carlo acquisition function on the model handle
    ``rs_model``: a botorch_rs ``SobolQMCNormalSampler`` (default: one with
    ``default_shape`` samples) or ``IIDNormalSampler`` (BoTorch's
    ``IIDNormalSampler`` is converted: same draws), or a
    ``StochasticSampler`` / ``ForkedRNGSampler`` (of botorch_rs or BoTorch)
    for a deterministic model."""
    if sampler is None:
        return SobolQMCNormalSampler(sample_shape=default_shape)
    _require_native_sampler(sampler)
    if isinstance(sampler, NormalMCSampler):
        return _native_normals(sampler)
    iid = as_iid_sampler(sampler)
    if iid is not None:
        return _native_normals(iid)
    if type(sampler).__name__ == "SobolQMCNormalSampler":
        normal = SobolQMCNormalSampler(sampler.sample_shape, seed=int(sampler.seed))
        normal.base_samples = sampler.base_samples
        return _native_normals(normal)
    stochastic = as_stochastic_sampler(sampler)
    if stochastic is not None:
        if not rs_model.is_deterministic:
            raise UnsupportedError(
                "A StochasticSampler draws new samples at every evaluation; botorch_rs "
                "supports it only for deterministic models (e.g. PosteriorMeanModel), whose "
                "samples equal the mean. Use a SobolQMCNormalSampler."
            )
        return stochastic
    raise UnsupportedError(
        "`sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler (or a "
        "StochasticSampler with a deterministic model)."
    )


__all__ = ["ForkedRNGSampler", "StochasticSampler"]
