r"""The truncated multivariate normal distribution
(``botorch.utils.probability.truncated_multivariate_normal``).

``TruncatedMultivariateNormal`` is, as in BoTorch, a subclass of torch's
``MultivariateNormal`` (the class is created when it is first accessed, which
requires torch; without torch, instantiating it raises ``UnsupportedError``).
Its log partition comes from botorch_rs's ``MVNXPB`` (in Rust, with exact
gradients) and its samples from botorch_rs's ``LinearEllipticalSliceSampler``
(Markov chains in Rust, random numbers from torch's generator). The interior
point of the sampler's polytope comes from botorch_rs's linear program; pass
a ``sampler`` with an explicit interior point to reproduce BoTorch's chains.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from botorch_rs.exceptions.errors import UnsupportedError


def _build() -> type:
    torch = importlib.import_module("torch")
    from torch.distributions.multivariate_normal import MultivariateNormal

    from botorch_rs.utils.probability.lin_ess import LinearEllipticalSliceSampler
    from botorch_rs.utils.probability.mvnxpb import MVNXPB

    class TruncatedMultivariateNormal(MultivariateNormal):
        def __init__(
            self,
            loc: Any,
            covariance_matrix: Any = None,
            precision_matrix: Any = None,
            scale_tril: Any = None,
            bounds: Any = None,
            solver: Any = None,
            sampler: Any = None,
            validate_args: bool | None = None,
        ) -> None:
            r"""Initializes an instance of a TruncatedMultivariateNormal distribution.

            Let ``x ~ N(0, K)`` be an ``n``-dimensional Gaussian random vector. This class
            represents the distribution of the truncated Multivariate normal random vector
            ``x | a <= x <= b``.

            Args:
                loc: A mean vector for the distribution, ``batch_shape x event_shape``.
                covariance_matrix: Covariance matrix distribution parameter.
                precision_matrix: Inverse covariance matrix distribution parameter.
                scale_tril: Lower triangular, square-root covariance matrix distribution
                    parameter.
                bounds: A ``batch_shape x event_shape x 2`` tensor of strictly increasing
                    bounds for ``x`` so that ``bounds[..., 0] < bounds[..., 1]`` everywhere.
                solver: A pre-solved MVNXPB instance used to approximate the log partition.
                sampler: A LinearEllipticalSliceSampler instance used for sample generation.
                validate_args: Optional argument to super().__init__.
            """
            if bounds is None:
                raise SyntaxError("Missing required argument `bounds`.")
            elif bounds.shape[-1] != 2:
                raise ValueError(
                    f"Expected bounds.shape[-1] to be 2 but bounds shape is {bounds.shape}"
                )
            elif torch.gt(*bounds.unbind(dim=-1)).any():
                raise ValueError("`bounds` must be strictly increasing along dim=-1.")

            super().__init__(
                loc=loc,
                covariance_matrix=covariance_matrix,
                precision_matrix=precision_matrix,
                scale_tril=scale_tril,
                validate_args=validate_args,
            )
            if sampler is not None and self.batch_shape:
                raise ValueError(
                    "A custom `sampler` cannot be used with a batched "
                    "TruncatedMultivariateNormal."
                )
            self.bounds = bounds
            self._solver = solver
            self._sampler = sampler

        def log_prob(self, value: Any) -> Any:
            r"""Approximates the true log probability."""
            neg_inf = float("-inf")
            inbounds = torch.logical_and(
                (self.bounds[..., 0] < value).all(-1),
                (self.bounds[..., 1] > value).all(-1),
            )
            if inbounds.any():
                return torch.where(
                    inbounds,
                    super().log_prob(value) - self.log_partition,
                    neg_inf,
                )
            return torch.full(value.shape[: -len(self.event_shape)], neg_inf)

        def rsample(self, sample_shape: Any = torch.Size()) -> Any:  # noqa: B008
            r"""Draw samples from the Truncated Multivariate Normal.

            Args:
                sample_shape: The shape of the samples.

            Returns:
                The (sample_shape x batch_shape x event_shape) tensor of samples.
            """
            sample_shape = torch.Size(sample_shape)
            num_samples = sample_shape.numel() if sample_shape else 1
            sampler = self.sampler
            if isinstance(sampler, LinearEllipticalSliceSampler):
                samples = sampler.draw(n=num_samples).view(*sample_shape, -1)
            else:
                samples = torch.stack(
                    [batch_sampler.draw(n=num_samples) for batch_sampler in sampler],
                    dim=1,
                ).reshape(sample_shape + self.batch_shape + self.event_shape)
            return self.loc + samples

        @property
        def log_partition(self) -> Any:
            return self.solver.log_prob

        @property
        def solver(self) -> Any:
            if self._solver is None:
                self._solver = MVNXPB(
                    covariance_matrix=self.covariance_matrix,
                    bounds=self.bounds - self.loc.unsqueeze(-1),
                )
                self._solver.solve()
            return self._solver

        @property
        def sampler(self) -> Any:
            if self._sampler is None:
                event_size = self.scale_tril.shape[-1]
                eye = torch.eye(
                    event_size,
                    dtype=self.scale_tril.dtype,
                    device=self.scale_tril.device,
                )

                A = torch.concat([-eye, eye])
                b = torch.concat(
                    [
                        self.loc - self.bounds[..., 0],
                        self.bounds[..., 1] - self.loc,
                    ],
                    dim=-1,
                ).unsqueeze(-1)

                if self.batch_shape:
                    num_batches = self.batch_shape.numel()
                    roots = self.scale_tril.expand(
                        self.batch_shape + torch.Size([event_size, event_size])
                    ).reshape(num_batches, event_size, event_size)
                    self._sampler = [
                        LinearEllipticalSliceSampler(
                            inequality_constraints=(A, batch_b),
                            covariance_root=batch_root,
                        )
                        for batch_b, batch_root in zip(
                            b.reshape(num_batches, 2 * event_size, 1), roots, strict=True
                        )
                    ]
                else:
                    self._sampler = LinearEllipticalSliceSampler(
                        inequality_constraints=(A, b),
                        covariance_root=self.scale_tril,
                    )
            return self._sampler

        def expand(self, batch_shape: Sequence[int], _instance: Any = None) -> Any:
            new = self._get_checked_instance(TruncatedMultivariateNormal, _instance)
            super().expand(batch_shape=batch_shape, _instance=new)

            new.bounds = self.bounds.expand(*new.batch_shape, *self.event_shape, 2)
            new._sampler = None  # does not implement ``expand``
            new._solver = None if self._solver is None else self._solver.expand(*batch_shape)
            return new

        def __repr__(self) -> str:
            return super().__repr__()[:-1] + f", bounds: {self.bounds.shape})"

    TruncatedMultivariateNormal.__module__ = __name__
    TruncatedMultivariateNormal.__qualname__ = "TruncatedMultivariateNormal"
    return TruncatedMultivariateNormal


class _TorchRequired:
    """Placeholder of a torch distribution class without torch."""

    _name = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise UnsupportedError(
            f"{self._name} is a torch distribution (a subclass of torch's "
            "MultivariateNormal, as in BoTorch) and requires torch."
        )


def __getattr__(name: str) -> Any:
    if name == "TruncatedMultivariateNormal":
        try:
            cls: Any = _build()
        except ImportError:
            cls = type(name, (_TorchRequired,), {"_name": name, "__module__": __name__})
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TruncatedMultivariateNormal"]
