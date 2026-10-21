"""``ListSampler`` (``botorch.sampling.list_sampler``): one sampler per
output of a list of models."""

from __future__ import annotations

import sys
from typing import Any

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.normal import MCSampler


class ListSampler(MCSampler):
    r"""A list of samplers, one for each sub-model of a ``ModelListGP``
    (e.g. the fantasy samplers of ``ModelListGP.fantasize`` and of the
    hypervolume knowledge gradient). All samplers must have the same sample
    shape."""

    def __init__(self, *samplers: MCSampler) -> None:
        for s in samplers:
            torch = sys.modules.get("torch")
            if not isinstance(s, MCSampler) and not (
                torch is not None and isinstance(s, torch.nn.Module)
            ):
                raise UnsupportedError(
                    "The samplers of a ListSampler must be MCSamplers or Torch modules."
                )
        self.samplers = list(samplers)
        self._validate_samplers()

    def _validate_samplers(self) -> None:
        shapes = [s.sample_shape for s in self.samplers]
        if not all(shapes[0] == s for s in shapes):
            raise UnsupportedError(
                "ListSampler requires all samplers to have the same sample shape."
            )

    @property
    def sample_shape(self) -> tuple[int, ...]:  # type: ignore[override]
        """The sample shape of the underlying samplers."""
        self._validate_samplers()
        return self.samplers[0].sample_shape

    def forward(self, posterior: Any) -> Any:
        """Sample each component posterior and concatenate along its output axis."""
        from botorch_rs.posteriors.posterior_list import PosteriorList, _cat_last

        reshape = getattr(posterior, "_reshape_and_cat", None)
        if not isinstance(posterior, PosteriorList) and not callable(reshape):
            raise UnsupportedError("ListSampler requires the PosteriorList protocol.")
        if len(self.samplers) != len(posterior.posteriors):
            raise ValueError("ListSampler requires one sampler per component posterior.")
        samples = [
            sampler(posterior=component)
            for sampler, component in zip(self.samplers, posterior.posteriors, strict=True)
        ]
        if isinstance(posterior, PosteriorList):
            return _cat_last(samples)
        assert callable(reshape)
        return reshape(samples)

    def _update_base_samples(self, posterior: Any, base_sampler: Any) -> None:
        self._instance_check(base_sampler)
        if len(self.samplers) != len(posterior.posteriors) or len(self.samplers) != len(
            base_sampler.samplers
        ):
            raise ValueError("ListSampler updates require matching sampler and posterior lists.")
        for sampler, component, baseline in zip(
            self.samplers, posterior.posteriors, base_sampler.samplers, strict=True
        ):
            sampler._update_base_samples(component, baseline)

    def __repr__(self) -> str:
        return f"ListSampler({', '.join(repr(s) for s in self.samplers)})"


__all__ = ["ListSampler"]
