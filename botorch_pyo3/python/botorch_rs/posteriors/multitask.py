"""The posterior of a ``KroneckerMultiTaskGP`` (``botorch.posteriors.multitask``)."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import ArrayKind, Flat, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.posteriors.gpytorch import (
    GPyTorchPosterior,
    MultitaskMultivariateNormal,
    _as_shape,
    _standard_normal,
)


class _InterleavedMarginalMTMVN(MultitaskMultivariateNormal):
    """BoTorch's ``MultitaskMultivariateNormal`` of a Kronecker multi-task
    posterior: interleaved, with the marginal variances on the diagonal of
    its covariance (and, with observation noise, the task noise covariance
    of every point)."""

    _interleaved = True

    @property
    def covariance_matrix(self) -> Any:
        p: MultitaskGPPosterior = self._p  # type: ignore[assignment]
        _, var = p._get_marginals()
        qm = p._q * p._m
        nb = math.prod(p._batch)
        noise = p._task_noise_offdiag()
        out: list[float] = []
        for b in range(nb):
            block = var[b * qm : (b + 1) * qm]
            for i in range(qm):
                for j in range(qm):
                    if i == j:
                        out.append(block[i])
                    elif noise is not None and i // p._m == j // p._m:
                        out.append(noise[(i % p._m) * p._m + (j % p._m)])
                    else:
                        out.append(0.0)
        return p._out(out, (*p._batch, qm, qm))


class MultitaskGPPosterior(GPyTorchPosterior):
    r"""The posterior of a ``KroneckerMultiTaskGP`` (BoTorch's
    ``MultitaskGPPosterior``): ``mean`` and ``variance`` (``batch_shape x q x
    m``) of the exact posterior, a ``distribution`` with the marginal
    variances (interleaved, as BoTorch), and samples drawn with Matheron's
    rule from base samples of shape ``base_sample_shape`` (``batch_shape x
    ((n + q) m + n m)``, plus ``q m`` with observation noise): the joint
    prior of the training and test outputs, the training noise and the test
    noise.

    Deviation: the roots of the joint prior and noise covariances are
    Cholesky factors (BoTorch: eigendecomposition-based roots), so given base
    samples give other (equally distributed) samples than BoTorch's.
    """

    def __init__(
        self,
        source: Any,
        model_rs: Any,
        X: Flat,
        kind: ArrayKind,
        observation_noise: bool,
        num_outputs: int,
        num_train: int,
        scales: list[float] | None = None,
    ) -> None:
        super().__init__(source, X, kind, observation_noise, num_outputs=num_outputs)
        self._model_rs = model_rs
        self._num_train = num_train
        self._scales = scales

    @property
    def _is_mt(self) -> bool:
        return True

    @property
    def distribution(self) -> _InterleavedMarginalMTMVN:  # type: ignore[override]
        return _InterleavedMarginalMTMVN(self)

    def _task_noise_offdiag(self) -> list[float] | None:
        """The task noise covariance (row-major ``m x m``, transformed space)
        added with observation noise, if it has off-diagonal entries."""
        if not self._observation_noise:
            return None
        data, _ = self._model_rs.task_noise_covariance()
        m = self._m
        if all(data[i * m + j] == 0.0 for i in range(m) for j in range(m) if i != j):
            return None
        s = self._scales or [1.0] * m
        return [data[i * m + j] * s[i] * s[j] for i in range(m) for j in range(m)]

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -1)

    def _total(self) -> int:
        n, q, m = self._num_train, self._q, self._m
        return (n + q) * m + n * m + (q * m if self._observation_noise else 0)

    @property
    def base_sample_shape(self) -> tuple[int, ...]:  # type: ignore[override]
        """``batch_shape x ((n + q) m + n m [+ q m])``."""
        return (*self._batch, self._total())

    def _matheron(self, z: Any) -> list[float]:
        out = self._model_rs.matheron_samples(self._x, z, self._observation_noise)
        return list(out[0])

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        """Posterior samples (``sample_shape x batch_shape x q x m``) by
        Matheron's rule from base samples of shape ``sample_shape x
        base_sample_shape`` (a trailing dimension of size 1 is allowed). As in
        BoTorch, base samples of the shape of the outputs (``sample_shape x
        batch_shape x q x m``) drive the first ``q m`` entries of the joint
        prior, the others being drawn with the caller's array library."""
        sample_shape = _as_shape(sample_shape)
        data, zshape = to_flat(base_samples, "base_samples")
        k = len(sample_shape)
        if tuple(zshape[:k]) != sample_shape:
            raise RuntimeError(
                "sample_shape disagrees with shape of base_samples."
                f"provided base sample shape is {tuple(zshape)} while"
                f"the expected shape is {sample_shape}."
            )
        rest = tuple(zshape[k:])
        if rest and rest[-1] == 1 and len(rest) == len(self._batch) + 2:
            rest = rest[:-1]
        n = math.prod(sample_shape)
        nb = math.prod(self._batch)
        total = self._total()
        qm = self._q * self._m
        if rest and rest[-1] == total:
            per_batch = math.prod(rest[:-1]) == nb and nb > 1
            if math.prod(rest[:-1]) not in (1, nb):
                raise BotorchTensorDimensionError(
                    f"base samples of shape {tuple(zshape)} do not match the batch shape "
                    f"{self._batch}."
                )
            z = (list(data), [n, nb, total] if per_batch else [n, total])
        elif len(data) == n * nb * qm:
            # BoTorch: the given values first, fresh draws for the rest.
            extra = _standard_normal(n * nb * (total - qm), self._kind)
            vals: list[float] = []
            for i in range(n * nb):
                vals.extend(data[i * qm : (i + 1) * qm])
                vals.extend(extra[i * (total - qm) : (i + 1) * (total - qm)])
            z = (vals, [n, nb, total] if nb > 1 else [n, total])
        else:
            raise BotorchTensorDimensionError(
                "The base samples are not compatible with base sample shape. "
                f"Received base samples of shape {tuple(zshape)}, expected "
                f"{(*sample_shape, *self.base_sample_shape)}."
            )
        return self._out(self._matheron(z), (*sample_shape, *self._batch, self._q, self._m))

    def rsample(self, sample_shape: Any = None) -> Any:
        """Samples (``sample_shape x batch_shape x q x m``) from base samples
        drawn with the caller's array library, in BoTorch's order (joint
        prior, training noise, test noise)."""
        sample_shape = _as_shape(sample_shape)
        n = math.prod(sample_shape)
        nb = math.prod(self._batch)
        n_tr, q, m = self._num_train, self._q, self._m
        joint = _standard_normal(n * nb * (n_tr + q) * m, self._kind)
        noise = _standard_normal(n * nb * n_tr * m, self._kind)
        test = _standard_normal(n * nb * q * m, self._kind) if self._observation_noise else []
        total = self._total()
        vals: list[float] = []
        for i in range(n * nb):
            vals.extend(joint[i * (n_tr + q) * m : (i + 1) * (n_tr + q) * m])
            vals.extend(noise[i * n_tr * m : (i + 1) * n_tr * m])
            if test:
                vals.extend(test[i * q * m : (i + 1) * q * m])
        z = (vals, [n, nb, total] if nb > 1 else [n, total])
        return self._out(self._matheron(z), (*sample_shape, *self._batch, self._q, self._m))

    def __repr__(self) -> str:
        return f"MultitaskGPPosterior(batch_shape={self._batch}, q={self._q}, m={self._m})"


__all__ = ["MultitaskGPPosterior"]
