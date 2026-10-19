"""``ListSampler`` (``botorch.sampling.list_sampler``): one sampler per
output of a list of models."""

from __future__ import annotations

from typing import Any

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.normal import MCSampler, NormalMCSampler


class ListSampler(MCSampler):
    r"""A list of samplers, one for each sub-model of a ``ModelListGP``
    (e.g. the fantasy samplers of ``ModelListGP.fantasize`` and of the
    hypervolume knowledge gradient). All samplers must have the same sample
    shape."""

    def __init__(self, *samplers: NormalMCSampler) -> None:
        for s in samplers:
            if not isinstance(s, NormalMCSampler):
                raise UnsupportedError(
                    "The samplers of a ListSampler must be botorch_rs SobolQMCNormalSamplers or IIDNormalSamplers."
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
        raise UnsupportedError(
            "A botorch_rs ListSampler is used by ModelListGP.fantasize and the hypervolume "
            "knowledge gradient; sample posteriors with SobolQMCNormalSampler."
        )

    __call__ = forward

    def __repr__(self) -> str:
        return f"ListSampler({', '.join(repr(s) for s in self.samplers)})"


__all__ = ["ListSampler"]
