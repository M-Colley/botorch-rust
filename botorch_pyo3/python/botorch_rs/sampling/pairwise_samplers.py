"""Samplers of pairwise comparisons (``botorch.sampling.pairwise_samplers``),
for the fantasies of a ``PairwiseGP`` (e.g. in the knowledge gradient).

The samplers draw posterior samples like their base samplers and compare
``max_num_comparisons`` pairs of the ``q`` points of every sample. The
pairs are chosen as BoTorch chooses them: ``np.random.seed(seed)`` (which
resets numpy's global generator, as in BoTorch) and ``np.random.choice``
without replacement (Python's ``random.Random(seed)`` without numpy).

Deviation from BoTorch (a bug): a comparison lists the item with the larger
sampled utility first, ``[preferred, other]``, the convention of
``PairwiseGP``. BoTorch's samplers list the item with the smaller sampled
utility first (``np.choose(prefs, pairs.T)`` picks the second item of a pair
when the first one is preferred), so its fantasies condition on reversed
preferences; see ``docs/drafts/10b-preference.md``.
"""

from __future__ import annotations

import importlib
import math
import random
from itertools import combinations
from typing import Any

from botorch_rs._arrays import to_flat
from botorch_rs.sampling.normal import IIDNormalSampler, MCSampler, SobolQMCNormalSampler


def _choose_pairs(seed: int, num_pairs: int, k: int) -> list[int]:
    """BoTorch's choice of ``k`` of ``num_pairs`` pairs without replacement."""
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        return random.Random(seed).sample(range(num_pairs), k)
    np.random.seed(seed)
    return [int(i) for i in np.random.choice(range(num_pairs), k, replace=False)]


class PairwiseMCSampler(MCSampler):
    r"""
    Abstract class for Pairwise MC Sampler.

    This sampler will sample pairwise comparisons. It is to be used together
    with PairwiseGP and BoTorch acquisition functions (e.g., qKnowledgeGradient)

    """

    def __init__(self, max_num_comparisons: int | None = None, seed: int | None = None) -> None:
        r"""
        Args:
            max_num_comparisons: Max number of comparisons drawn within samples.
                If None, use all possible pairwise comparisons
            seed: The seed for np.random.seed. If omitted, use a random seed.
                May be overwritten by sibling classes or subclasses.
        """
        self.max_num_comparisons = max_num_comparisons
        if seed is None:
            try:
                torch = importlib.import_module("torch")
            except ImportError:
                seed = random.randint(0, 999999)
            else:
                seed = int(torch.randint(0, 1000000, (1,)).item())
        self.seed = int(seed)

    def _base_forward(self, posterior: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, posterior: Any) -> Any:
        r"""Draws MC samples from the posterior and make comparisons

        Args:
            posterior: The Posterior to sample from.
                The returned samples are expected to have output dimension of 1.

        Returns:
            Posterior sample pairwise comparisons (``sample_shape x
            batch_shape x num_comparisons x 2`` indices, the preferred item
            first).
        """
        from botorch_rs.models.pairwise_gp import _long_array

        samples = self._base_forward(posterior)
        data, shape = to_flat(samples, "samples")
        s_n = shape[-2]  # candidate number per batch
        if s_n < 2:
            raise RuntimeError("Number of samples < 2, cannot make comparisons")
        all_pairs = list(combinations(range(s_n), 2))
        if self.max_num_comparisons is None:
            comp_n = len(all_pairs)
        else:
            comp_n = min(self.max_num_comparisons, len(all_pairs))
        chosen = [all_pairs[i] for i in _choose_pairs(self.seed, len(all_pairs), comp_n)]
        rows = math.prod(shape[:-2])
        out: list[int] = []
        for r in range(rows):
            v = data[r * s_n : (r + 1) * s_n]
            for i1, i2 in chosen:
                if v[i1] > v[i2]:
                    out.extend((i1, i2))
                else:
                    out.extend((i2, i1))
        kind = posterior._kind
        return _long_array(out, (*shape[:-2], comp_n, 2), kind)

    __call__ = forward


class PairwiseIIDNormalSampler(PairwiseMCSampler, IIDNormalSampler):
    def __init__(
        self,
        sample_shape: Any,
        seed: int | None = None,
        max_num_comparisons: int | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            sample_shape: The ``sample_shape`` of the samples to generate.
            seed: The seed for the RNG. If omitted, use a random seed.
            max_num_comparisons:  Max number of comparisons drawn within samples.
                If None, use all possible pairwise comparisons.
            kwargs: Catch-all for deprecated arguments.
        """
        PairwiseMCSampler.__init__(self, max_num_comparisons=max_num_comparisons, seed=seed)
        IIDNormalSampler.__init__(self, sample_shape=sample_shape, seed=self.seed, **kwargs)

    def _base_forward(self, posterior: Any) -> Any:
        return IIDNormalSampler.forward(self, posterior)

    forward = PairwiseMCSampler.forward
    __call__ = PairwiseMCSampler.forward


class PairwiseSobolQMCNormalSampler(PairwiseMCSampler, SobolQMCNormalSampler):
    def __init__(
        self,
        sample_shape: Any,
        seed: int | None = None,
        max_num_comparisons: int | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            sample_shape: The ``sample_shape`` of the samples to generate.
            seed: The seed for the RNG. If omitted, use a random seed.
            max_num_comparisons:  Max number of comparisons drawn within samples.
                If None, use all possible pairwise comparisons.
            kwargs: Catch-all for deprecated arguments.
        """
        if kwargs:
            from botorch_rs.exceptions.errors import UnsupportedError

            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
        PairwiseMCSampler.__init__(self, max_num_comparisons=max_num_comparisons, seed=seed)
        SobolQMCNormalSampler.__init__(self, sample_shape=sample_shape, seed=self.seed)

    @property  # type: ignore[override]
    def seed(self) -> int:
        return self._pairwise_seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._pairwise_seed = int(value)

    def _base_forward(self, posterior: Any) -> Any:
        return SobolQMCNormalSampler.forward(self, posterior)

    forward = PairwiseMCSampler.forward
    __call__ = PairwiseMCSampler.forward


__all__ = ["PairwiseIIDNormalSampler", "PairwiseMCSampler", "PairwiseSobolQMCNormalSampler"]
