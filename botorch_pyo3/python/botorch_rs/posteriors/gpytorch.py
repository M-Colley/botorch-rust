"""Gaussian posteriors (``botorch.posteriors.gpytorch``).

Posteriors are evaluated lazily: ``mean`` and ``variance`` only need the
marginal distribution of each point (linear in the number of points), while
``distribution.covariance_matrix`` and sampling compute the joint posterior of
every ``q``-batch.

A posterior over ``m > 1`` outputs is BoTorch's ``GPyTorchPosterior`` of a
``MultitaskMultivariateNormal`` with ``interleaved=False``: ``mean`` and
``variance`` have shape ``batch_shape x q x m`` and the covariance matrix is
``batch_shape x (q m) x (q m)`` over the output-major flattening (index
``t * q + p`` for output ``t`` at point ``p``); outputs are independent.

With a one-to-many input transform (``InputPerturbation`` with ``n_w``
perturbations), ``q`` counts the ``q_in * n_w`` posterior points of ``q_in``
inputs, as in BoTorch.
"""

from __future__ import annotations

import math
import random
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    ArrayKind,
    Flat,
    FloatData,
    Pair,
    broadcast_shapes,
    from_flat,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError


def _as_shape(sample_shape: Any) -> tuple[int, ...]:
    if sample_shape is None:
        return (1,)
    if isinstance(sample_shape, int):
        return (sample_shape,)
    return tuple(int(s) for s in sample_shape)


def _standard_normal(n: int, kind: ArrayKind) -> list[float]:
    """``n`` i.i.d. standard normal draws from the random number generator of
    the caller's array library (so that ``torch.manual_seed`` and
    ``numpy.random.seed`` apply)."""
    if kind.lib == "torch":
        import torch

        return torch.randn(n, dtype=torch.float64).tolist()
    if kind.lib == "numpy":
        import numpy as np

        return np.random.standard_normal(n).tolist()
    return [random.gauss(0.0, 1.0) for _ in range(n)]


class Posterior:
    """Base class of the posteriors."""

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `rsample_from_base_samples`."
        )

    @property
    def device(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement `device`.")

    @property
    def dtype(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement `dtype`.")

    @property
    def base_sample_shape(self) -> tuple[int, ...]:
        raise NotImplementedError(f"{type(self).__name__} does not implement `base_sample_shape`.")

    @property
    def batch_range(self) -> tuple[int, int]:
        raise NotImplementedError(f"{type(self).__name__} does not implement `batch_range`.")

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        raise NotImplementedError(f"{type(self).__name__} does not implement `_extended_shape`.")

    def rsample(self, sample_shape: Any = None) -> Any:
        raise NotImplementedError

    def sample(self, sample_shape: Any = None) -> Any:
        """Draw samples without tracking gradients, preserving the array type."""
        torch = sys.modules.get("torch")
        if torch is not None:
            with torch.no_grad():
                return self.rsample(sample_shape=sample_shape)
        samples = self.rsample(sample_shape=sample_shape)
        # A caller's posterior can itself import torch lazily while sampling.
        torch = sys.modules.get("torch")
        return (
            samples.detach() if torch is not None and isinstance(samples, torch.Tensor) else samples
        )

    def quantile(self, value: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement a `quantile` method.")

    def density(self, value: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement a `density` method.")


class GPyTorchPosterior(Posterior):
    """The Gaussian posterior of a model at ``X`` (``batch_shape x q x d``),
    in the original outcome space.

    ``source`` is the extension object that evaluates the posterior (a
    ``SingleTaskGP`` or ``ModelListGP``); ``output_indices`` selects outputs
    and ``transform`` is a ``(weights, offset)`` scalarization (BoTorch's
    ``ScalarizedPosteriorTransform``). Mutable sources are snapshotted so
    later fitting or parameter updates do not change this posterior's law.
    """

    def __init__(
        self,
        source: Any = None,
        X: Flat | None = None,
        kind: ArrayKind | None = None,
        observation_noise: bool = False,
        num_outputs: int = 1,
        model_batch_shape: tuple[int, ...] = (),
        output_indices: list[int] | None = None,
        transform: tuple[list[float], float] | None = None,
        points_per_input: int = 1,
        base: bool = False,
        interleaved: bool = False,
        query_inputs: Any = None,
        parameter_owner: Any = None,
        *,
        distribution: Any = None,
    ) -> None:
        self._rs: Any
        self._parameter_inputs: tuple[Any, ...] = ()
        self._parameter_gradient_error: str | None = None
        self._marginals: tuple[list[float], list[float]] | None = None
        self._joint: tuple[Flat, Flat] | None = None
        if distribution is not None or (source is not None and X is None and kind is None):
            if distribution is not None and source is not None:
                raise TypeError("Supply either distribution or a native model source.")
            from botorch_rs.posteriors._distribution import GaussianDistributionAdapter

            adapter = GaussianDistributionAdapter(source if distribution is None else distribution)
            self._distribution_adapter = adapter
            self._rs = None
            self._query_inputs = None
            self._kind = kind_of(adapter.mean)
            self._batch = tuple(adapter.mean.shape[:-2])
            self._q, self._m = adapter.mean.shape[-2:]
            self._interleaved_mt = bool(getattr(adapter.distribution, "_interleaved", False))
            self._base = False
            self._transform = None
            return
        if X is None or kind is None:
            raise TypeError("Provide distribution, or a native source, X, and kind.")
        sync = getattr(parameter_owner, "_sync_torch_parameters", None)
        if kind.lib == "torch" and sync is not None:
            state = sync()
            if state is not None:
                self._parameter_inputs = tuple(p for _, p in state.entries)
                self._parameter_gradient_error = state.gradient_error
        snapshot = getattr(source, "copy", None)
        self._rs = snapshot() if callable(snapshot) else source
        self._interleaved_mt = interleaved
        self._base = base
        self._points_per_input = points_per_input
        self._x = X
        # Keep the query's graph and value snapshot separately from the flat
        # native buffer. Cloning preserves first-order input connectivity.
        self._query_inputs = (
            query_inputs.clone() if kind.lib == "torch" and query_inputs is not None else None
        )
        self._kind = kind
        self._observation_noise = observation_noise
        self._output_indices = output_indices
        self._transform = transform
        shape = X[1]
        batch = broadcast_shapes(tuple(shape[:-2]), tuple(model_batch_shape))
        if batch is None:
            raise BotorchTensorDimensionError(
                "The trailing batch dimensions of X must match the trailing batch dimensions "
                f"of the training inputs. Got X.shape={tuple(shape)} and "
                f"original_batch_shape={tuple(model_batch_shape)}."
            )
        self._batch = batch
        self._q = shape[-2] * points_per_input
        selected = len(output_indices) if output_indices is not None else num_outputs
        if transform is not None and len(transform[0]) != selected:
            raise RuntimeError(
                "Output shape not equal to that of weights. Output shape is "
                f"{selected} and weights are {len(transform[0])}"
            )
        self._m = 1 if transform is not None else selected

    @property
    def _is_mt(self) -> bool:
        """Whether the distribution is a ``MultitaskMultivariateNormal``."""
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return bool(adapter.is_multitask)
        return self._m > 1

    def _kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "output_indices": self._output_indices,
            "transform": self._transform,
        }
        if self._base:
            kwargs["base"] = True
        return kwargs

    # -- lazily computed quantities -------------------------------------

    def _get_marginals(self) -> tuple[list[float], list[float]]:
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return list(to_flat(adapter.mean)[0]), list(to_flat(adapter.variance)[0])
        if self._marginals is None:
            (mean, _), (var, _) = self._rs.posterior_marginals(
                self._x, self._observation_noise, **self._kwargs()
            )
            self._marginals = (mean, var)
        return self._marginals

    def _get_joint(self) -> tuple[Flat, Flat]:
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            covariance = adapter.distribution.covariance_matrix
            if adapter.is_multitask and adapter.distribution._interleaved:
                k = len(self._batch)
                covariance = covariance.reshape(*self._batch, self._q, self._m, self._q, self._m)
                covariance = covariance.permute(*range(k), k + 1, k, k + 3, k + 2)
                covariance = covariance.reshape(*self._batch, self._q * self._m, self._q * self._m)
            return to_flat(adapter.mean), to_flat(covariance)
        if self._joint is None:
            self._joint = self._rs.posterior_joint(
                self._x, self._observation_noise, **self._kwargs()
            )
        return self._joint

    def _out(self, data: FloatData, shape: tuple[int, ...]) -> Any:
        return from_flat(data, shape, self._kind)

    def _query_out(self, mode: str, data: Any, shape: tuple[int, ...], base: Any = None) -> Any:
        query = getattr(self, "_query_inputs", None)
        if query is not None and query.requires_grad and sys.modules["torch"].is_grad_enabled():
            from botorch_rs.posteriors._query_autograd import query_output

            result = query_output(self, mode, data, shape, base)
        else:
            result = self._out(data, shape)
        if self.__dict__.get("_parameter_inputs"):
            from botorch_rs.posteriors._parameter_autograd import parameter_output

            result = parameter_output(self, mode, result, shape, base)
        return result

    def _moment_out(self, mode: str, shape: tuple[int, ...]) -> Any:
        return self._query_out(mode, self._get_marginals()[0 if mode == "mean" else 1], shape)

    def _samples_out(self, z: Pair, sample_shape: tuple[int, ...]) -> Any:
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            data, shape = z
            batch = (1,) * len(self._batch) if len(shape) == 2 else self._batch
            event = (self._q, self._m) if self._is_mt else (self._q,)
            normals = from_flat(data, (*sample_shape, *batch, *event), self._kind)
            return adapter.rsample(sample_shape, normals)
        return self._query_out(
            "samples", self._samples(z), (*sample_shape, *self._batch, self._q, self._m), z
        )

    def _samples(self, z: Pair) -> list[float]:
        out = self._rs.posterior_samples(self._x, z, self._observation_noise, **self._kwargs())
        return out[0]

    # -- BoTorch's Posterior API -----------------------------------------

    @property
    def mean(self) -> Any:
        """The posterior mean, shape ``batch_shape x q x m``."""
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return adapter.mean
        return self._moment_out("mean", (*self._batch, self._q, self._m))

    @property
    def variance(self) -> Any:
        """The posterior variance (clamped at 1e-10), shape ``batch_shape x q x m``."""
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return adapter.variance
        return self._moment_out("variance", (*self._batch, self._q, self._m))

    @property
    def distribution(self) -> MultivariateNormal:
        """The joint normal distribution of each ``q``-batch (a
        ``MultitaskMultivariateNormal`` for several outputs)."""
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return adapter.distribution
        if self._is_mt:
            mvn = MultitaskMultivariateNormal(self)
            if getattr(self, "_interleaved_mt", False):
                mvn._interleaved = True
            return mvn
        return MultivariateNormal(self)

    @property
    def mvn(self) -> MultivariateNormal:
        return self.distribution

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range of the base samples (BoTorch's ``batch_range``)."""
        return (0, -2) if self._is_mt else (0, -1)

    @property
    def device(self) -> Any:
        if self._kind.lib == "torch":
            import importlib

            return self._kind.device or importlib.import_module("torch").device("cpu")
        return None

    @property
    def dtype(self) -> Any:
        if self._kind.lib == "torch":
            import importlib

            return self._kind.dtype or importlib.import_module("torch").float64
        return None

    @property
    def base_sample_shape(self) -> tuple[int, ...]:
        """The shape of the base samples of one posterior sample."""
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return adapter.base_sample_shape
        if self._is_mt:
            return (*self._batch, self._q, self._m)
        return (*self._batch, self._q)

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        return (*_as_shape(sample_shape or ()), *self._batch, self._q, self._m)

    def rsample(self, sample_shape: Any = None) -> Any:
        """Posterior samples, shape ``sample_shape x batch_shape x q x m``.

        The base samples are drawn with the caller's array library (torch,
        NumPy or Python's ``random``)."""
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return adapter.rsample(sample_shape)
        sample_shape = _as_shape(sample_shape)
        n = math.prod(sample_shape)
        nb = math.prod(self._batch)
        qm = self._q * self._m
        z = _standard_normal(n * nb * qm, self._kind)
        return self._samples_out((z, [n, nb, qm]), sample_shape)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        """Posterior samples ``mean + L z`` from explicit base samples ``z`` of
        shape ``sample_shape x base_sample_shape`` (batch dimensions of size 1
        are broadcast; for a single output a trailing output dimension of size
        1 is allowed). For several outputs the base sample of point ``p`` and
        output ``t`` drives that point and output, as in BoTorch."""
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return adapter.rsample(sample_shape, base_samples)
        sample_shape = _as_shape(sample_shape)
        data, zshape = to_flat(base_samples, "base_samples")
        k = len(sample_shape)
        if tuple(zshape[:k]) != sample_shape:
            raise RuntimeError(
                "`sample_shape` disagrees with shape of `base_samples`. "
                f"Got sample_shape={sample_shape} and base_samples.shape={zshape}."
            )
        rest = tuple(zshape[k:])
        event = (self._q, self._m) if self._is_mt else (self._q,)
        if not self._is_mt and len(rest) == len(self._batch) + 2 and rest[-1] == 1:
            rest = rest[:-1]
        n = math.prod(sample_shape)
        if rest[len(rest) - len(event) :] != event:
            raise BotorchTensorDimensionError(
                f"base samples of shape {zshape} do not match the event shape {event}."
            )
        batch = rest[: len(rest) - len(event)]
        qm = self._q * self._m
        if math.prod(batch) == 1:
            z: Pair = (data, (n, qm))
        elif batch == self._batch:
            z = (data, (n, math.prod(self._batch), qm))
        else:
            raise BotorchTensorDimensionError(
                f"base samples of shape {zshape} do not match the batch shape {self._batch}."
            )
        return self._samples_out(z, sample_shape)

    def _marginal_value(self, value: Any) -> Any:
        """Arrange marginal evaluation values before the posterior dimensions."""
        data, shape = to_flat(value, "value")
        target = (*shape, *(1 for _ in range(len(self._batch) + 2))) if len(data) > 1 else ()
        if hasattr(value, "reshape"):
            return value.reshape(target)
        return from_flat(data, target, kind_of(value))

    def quantile(self, value: Any) -> Any:
        """Quantiles of the Gaussian marginals, computed natively in Rust."""
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return adapter.marginal(value, density=False)
        from botorch_rs import _ops

        z = _ops.apply("ndtri", self._marginal_value(value))
        sd = _ops.apply("sqrt", self.variance)
        return _ops.apply("add", self.mean, _ops.apply("mul", sd, z))

    def density(self, value: Any) -> Any:
        """Probability density of each Gaussian marginal at ``value``."""
        if (adapter := self.__dict__.get("_distribution_adapter")) is not None:
            return adapter.marginal(value, density=True)
        from botorch_rs import _ops

        sd = _ops.apply("sqrt", self.variance)
        z = _ops.apply("div", _ops.apply("sub", self._marginal_value(value), self.mean), sd)
        log_density = _ops.apply("sub", _ops.apply("log_phi", z), _ops.apply("log", sd))
        return _ops.apply("exp", log_density)

    def __repr__(self) -> str:
        return f"GPyTorchPosterior(batch_shape={self._batch}, q={self._q}, m={self._m})"

    def __getattr__(self, name: str) -> Any:
        adapter = self.__dict__.get("_distribution_adapter")
        if adapter is None:
            raise AttributeError(name)
        return getattr(adapter.distribution, name)


class MultivariateNormal:
    """The joint normal distribution of a single-output posterior's
    ``q``-batches (``posterior.distribution``; no output dimension)."""

    _interleaved = True

    def __init__(self, posterior: GPyTorchPosterior) -> None:
        self._p = posterior

    def _event(self) -> tuple[int, ...]:
        return (self._p._q,)

    @property
    def mean(self) -> Any:
        """Shape ``batch_shape x q`` (``batch_shape x q x m`` for several outputs)."""
        p = self._p
        return p._moment_out("mean", (*p._batch, *self._event()))

    @property
    def variance(self) -> Any:
        """Marginal variances (clamped at 1e-10), shape of ``mean``."""
        p = self._p
        return p._moment_out("variance", (*p._batch, *self._event()))

    @property
    def stddev(self) -> Any:
        from botorch_rs import _ops

        return _ops.apply("sqrt", self.variance)

    @property
    def covariance_matrix(self) -> Any:
        """Joint covariances, shape ``batch_shape x q x q`` (``batch_shape x
        (q m) x (q m)`` over the output-major flattening for several outputs)."""
        p = self._p
        _, cov = p._get_joint()
        return p._query_out("covariance", cov[0], tuple(cov[1]))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._p._batch

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._event()

    def rsample(self, sample_shape: Any = None, base_samples: Any = None) -> Any:
        """Samples of shape ``sample_shape x batch_shape x event_shape``."""
        p = self._p
        if base_samples is None:
            out = p.rsample(sample_shape)
        else:
            out = p.rsample_from_base_samples(_as_shape(sample_shape), base_samples)
        if p._is_mt:
            return out
        if hasattr(out, "reshape"):
            return out.reshape(*out.shape[:-1])
        data, shape = to_flat(out, "samples")
        return p._out(data, shape[:-1])


class MultitaskMultivariateNormal(MultivariateNormal):
    """The joint normal distribution of a multi-output posterior's
    ``q``-batches (GPyTorch's ``MultitaskMultivariateNormal`` with
    ``interleaved=False``)."""

    _interleaved = False

    def _event(self) -> tuple[int, ...]:
        return (self._p._q, self._p._m)

    @property
    def covariance_matrix(self) -> Any:
        """Joint covariances, ``batch_shape x (q m) x (q m)``: over the
        output-major flattening, or interleaved (index ``p * m + t``) for an
        interleaved distribution (e.g. that of a ``SingleTaskVariationalGP``)."""
        p = self._p
        _, (data, shape) = p._get_joint()
        if p._kind.lib == "torch":
            covariance = p._query_out("covariance", data, tuple(shape))
            if self._interleaved:
                k = len(p._batch)
                covariance = covariance.reshape(*p._batch, p._m, p._q, p._m, p._q)
                covariance = covariance.permute(*range(k), k + 1, k, k + 3, k + 2)
                covariance = covariance.reshape(*shape)
            return covariance
        if not self._interleaved:
            return p._out(data, tuple(shape))
        q, m = p._q, p._m
        n = q * m
        out = [0.0] * len(data)
        for b in range(len(data) // max(n * n, 1)):
            off = b * n * n
            for t in range(m):
                for i in range(q):
                    for u in range(m):
                        for j in range(q):
                            out[off + (i * m + t) * n + j * m + u] = data[
                                off + (t * q + i) * n + u * q + j
                            ]
        return p._out(out, tuple(shape))

    @property
    def num_tasks(self) -> int:
        return self._p._m


def _validate_scalarize_inputs(weights: Any, m: int) -> list[float]:
    values, shape = to_flat(weights, "weights")
    if len(shape) != 1:
        raise BotorchTensorDimensionError("`weights` must be one-dimensional")
    if len(values) != m:
        raise RuntimeError(
            f"Output shape not equal to that of weights. Output shape is {m} and weights are {shape}"
        )
    return [float(v) for v in values]


class _MomentPosterior(GPyTorchPosterior):
    """A Gaussian posterior backed by explicit native-computed moments."""

    def __init__(self, mean: Pair, covariance: Pair, kind: ArrayKind, num_outputs: int = 1) -> None:
        m = num_outputs
        batch, q = (
            (tuple(mean[1][:-1]), mean[1][-1]) if m == 1 else (tuple(mean[1][:-2]), mean[1][-2])
        )
        super().__init__(None, ([], (*batch, q, 0)), kind, False, num_outputs=m)
        qm = q * m
        self._joint = ((list(mean[0]), (*batch, q, m)), (list(covariance[0]), tuple(covariance[1])))
        diagonal = [
            max(float(covariance[0][b * qm * qm + (t * q + i) * qm + t * q + i]), 1e-10)
            for b in range(math.prod(batch))
            for i in range(q)
            for t in range(m)
        ]
        self._marginals = (list(mean[0]), diagonal)

    def _samples(self, z: Pair) -> list[float]:
        data, zshape = z
        samples, q, m, batches = zshape[0], self._q, self._m, math.prod(self._batch)
        qm = q * m
        collapsed = len(zshape) == 2
        base = [
            data[s * qm + i * m + t] if collapsed else data[(s * batches + b) * qm + i * m + t]
            for b in range(batches)
            for t in range(m)
            for i in range(q)
            for s in range(samples)
        ]
        (mean, _), (covariance, _) = self._get_joint()
        loc = [
            mean[(b * q + i) * m + t] for b in range(batches) for t in range(m) for i in range(q)
        ]
        out, _ = _rs.sample_cached_cholesky(covariance, [], loc, base, batches, 0, qm, samples, 3)
        return [
            out[(b * qm + t * q + i) * samples + s]
            for s in range(samples)
            for b in range(batches)
            for i in range(q)
            for t in range(m)
        ]


def scalarize_posterior_gpytorch(
    posterior: GPyTorchPosterior, weights: Any, offset: float = 0.0
) -> tuple[Any, Any]:
    """Mean and covariance of the affine scalarization of a Gaussian posterior."""
    if not isinstance(posterior, GPyTorchPosterior):
        raise UnsupportedError("scalarize_posterior_gpytorch requires a GPyTorchPosterior.")
    if posterior.__dict__.get("_distribution_adapter") is not None:
        from botorch_rs.posteriors._distribution import scalarize_distribution

        return scalarize_distribution(posterior, weights, offset)
    w = _validate_scalarize_inputs(weights, posterior._m)
    mean, covariance = posterior._get_joint()
    mu, cov = _rs.scalarize_posterior_moments(mean, covariance, w, float(offset))
    return posterior._out(mu[0], tuple(mu[1])), posterior._out(cov[0], tuple(cov[1]))


def scalarize_posterior(posterior: Any, weights: Any, offset: float = 0.0) -> GPyTorchPosterior:
    """Affine scalarization of Gaussian posteriors, or single-point PosteriorLists."""
    from botorch_rs.posteriors.posterior_list import PosteriorList

    if isinstance(posterior, GPyTorchPosterior):
        if posterior.__dict__.get("_distribution_adapter") is not None:
            import torch

            mean, covariance = scalarize_posterior_gpytorch(posterior, weights, offset)
            return GPyTorchPosterior(
                distribution=torch.distributions.MultivariateNormal(mean, covariance)
            )
        w = _validate_scalarize_inputs(weights, posterior._m)
        # Keep source-backed models lazy and use their native posterior sampler.
        if posterior._rs is not None and not posterior._base:
            original_weights = w
            new_offset = float(offset)
            if posterior._transform is not None:
                old_weights, old_offset = posterior._transform
                original_weights = [w[0] * v for v in old_weights]
                new_offset += w[0] * old_offset
            return GPyTorchPosterior(
                posterior._rs,
                posterior._x,
                posterior._kind,
                posterior._observation_noise,
                num_outputs=len(original_weights),
                model_batch_shape=posterior._batch,
                output_indices=posterior._output_indices,
                transform=(original_weights, new_offset),
                points_per_input=posterior._points_per_input,
                query_inputs=posterior._query_inputs,
            )
        mean_moments, covariance_moments = posterior._get_joint()
        mu, cov = _rs.scalarize_posterior_moments(
            mean_moments, covariance_moments, w, float(offset)
        )
        return _MomentPosterior(mu, cov, posterior._kind)
    if not isinstance(posterior, PosteriorList):
        raise NotImplementedError(
            "scalarize_posterior only works with a posterior that has an attribute "
            "`distribution`, such as a GPyTorchPosterior, or a posterior that contains "
            "sub-posteriors in an attribute `posteriors`, as in a PosteriorList."
        )
    list_mean, shape = to_flat(posterior.mean, "mean")
    q, m = shape[-2:]
    w = _validate_scalarize_inputs(weights, m)
    if q != 1:
        raise NotImplementedError(
            "scalarize_posterior only works with a PosteriorList if each sub-posterior has q=1."
        )
    if any(to_flat(p.mean)[1][-1] != 1 for p in posterior.posteriors):
        raise NotImplementedError(
            "scalarize_posterior only works with a PosteriorList if each sub-posterior has one outcome."
        )
    variance, _ = to_flat(posterior.variance, "variance")
    batch = tuple(shape[:-2])
    list_covariance = [0.0] * (math.prod(batch) * m * m)
    for b in range(math.prod(batch)):
        for t in range(m):
            list_covariance[(b * m + t) * m + t] = variance[b * m + t]
    mu, cov = _rs.scalarize_posterior_moments(
        (list_mean, shape), (list_covariance, (*batch, m, m)), w, float(offset)
    )
    return _MomentPosterior(mu, cov, kind_of(posterior.mean))


__all__ = [
    "GPyTorchPosterior",
    "MultitaskMultivariateNormal",
    "MultivariateNormal",
    "Posterior",
    "scalarize_posterior",
    "scalarize_posterior_gpytorch",
]
