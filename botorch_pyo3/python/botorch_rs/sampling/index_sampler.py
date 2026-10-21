"""Sampler to be used with ``EnsemblePosterior``\\ s to enable deterministic
optimization of acquisition functions with ensemble models
(``botorch.sampling.index_sampler``)."""

from __future__ import annotations

import importlib
import math
import random
from typing import Any, cast

from botorch_rs._arrays import LIST, from_flat, to_flat
from botorch_rs.posteriors.ensemble import EnsemblePosterior
from botorch_rs.sampling.normal import MCSampler, _sample_shape, _uses_tensor_protocol


class IndexSampler(MCSampler):
    r"""A sampler that calls ``posterior.rsample_from_base_samples`` to
    generate the samples via index base samples.

    The base samples are the indices of the ensemble members of every sample
    (``sample_shape x batch_shape``), drawn from the posterior's mixture
    weights with torch's generator seeded with ``seed`` (as BoTorch draws
    them; Python's ``random`` without torch), or assigned to
    ``base_samples``.
    """

    def __init__(self, sample_shape: Any, seed: int | None = None) -> None:
        self.sample_shape = _sample_shape(sample_shape)
        if seed is None:
            try:
                torch = importlib.import_module("torch")
            except ImportError:
                seed = random.randint(0, 999999)
            else:
                seed = int(torch.randint(0, 1000000, (1,)).item())
        self.seed = int(seed)
        self.base_samples: Any = None

    def _construct_base_samples(self, posterior: EnsemblePosterior) -> None:
        r"""Constructs base samples as indices to sample with them from the
        Posterior."""
        if _uses_tensor_protocol(posterior):
            import torch

            from botorch_rs.posteriors._torch_ensemble import TorchEnsemblePosterior, draw_indices

            expected = (*self.sample_shape, *posterior.batch_shape)
            current = (
                None
                if self.base_samples is None
                else torch.as_tensor(self.base_samples, device=posterior.device)
            )
            if current is None or tuple(current.shape) != expected:
                generator = torch.Generator(device=posterior.device).manual_seed(self.seed)
                self.base_samples = draw_indices(
                    cast(TorchEnsemblePosterior, posterior), self.sample_shape, generator
                )
            else:
                self.base_samples = current
            return
        expected = (*self.sample_shape, *posterior.batch_shape)
        if self.base_samples is not None and tuple(to_flat(self.base_samples)[1]) == expected:
            return
        probs = posterior._normalized_mixture_weights()
        mix_shape = posterior._shape[:-2]
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            s = posterior.ensemble_size
            rng = random.Random(self.seed)
            nb = math.prod(posterior.batch_shape)
            idx = [
                float(rng.choices(range(s), weights=probs[b * s : (b + 1) * s])[0])
                for _ in range(math.prod(self.sample_shape))
                for b in range(nb)
            ]
            self.base_samples = from_flat(idx, expected, LIST)
            return
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            p = torch.tensor(probs, dtype=torch.float64).reshape(*mix_shape)
            self.base_samples = (
                torch.distributions.Multinomial(probs=p)
                .sample(sample_shape=torch.Size(self.sample_shape))
                .argmax(dim=-1)
            )

    def forward(self, posterior: EnsemblePosterior) -> Any:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The ensemble posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        self._construct_base_samples(posterior=posterior)
        return posterior.rsample_from_base_samples(
            sample_shape=self.sample_shape, base_samples=self.base_samples
        )

    def _update_base_samples(self, posterior: Any, base_sampler: Any) -> None:
        """Ensemble member coordinates do not grow with the number of query points."""
        # This is intentionally a null operation, matching the public reference.

    def _member_counts(self, ensemble_size: int, weights: Any = None) -> list[float]:
        """The number of samples that select each ensemble member for a
        single t-batch (base samples of shape ``sample_shape x 1``)."""
        values = [0.0] * ensemble_size
        p = EnsemblePosterior(
            from_flat([0.0] * ensemble_size, (1, ensemble_size, 1, 1), LIST), weights=weights
        )
        self._construct_base_samples(p)
        data, _ = to_flat(self.base_samples, "base_samples")
        for v in data:
            values[int(v)] += 1.0
        return values


__all__ = ["IndexSampler"]
