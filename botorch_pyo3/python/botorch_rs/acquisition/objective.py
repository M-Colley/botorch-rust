"""Posterior transforms and MC objectives (``botorch.acquisition.objective``).

``ScalarizedPosteriorTransform`` scalarizes multi-output posteriors. The MC
objectives map posterior samples (``sample_shape x batch_shape x q x m``) to
objective values (``sample_shape x batch_shape x q``): ``IdentityMCObjective``,
``LinearMCObjective`` (evaluated in Rust), and ``GenericMCObjective``,
``ConstrainedMCObjective`` and other ``MCAcquisitionObjective`` subclasses,
whose callables operate on torch tensors as in BoTorch. Monte-Carlo
acquisition functions evaluate these callables, and their vector-Jacobian
products (via ``torch.autograd``), from within the Rust computation, so that
acquisition values and gradients stay exact. BoTorch's own objective objects
are accepted as well.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from botorch_rs._arrays import as_float, from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError


class PosteriorTransform:
    """Base class of the posterior transforms."""

    #: Whether the transform maps a multi-output posterior to a single output.
    scalarize: bool = False

    def evaluate(self, Y: Any, X: Any = None) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, posterior: Any, X: Any = None) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, posterior: Any, X: Any = None) -> Any:
        return self.forward(posterior, X)


class ScalarizedPosteriorTransform(PosteriorTransform):
    r"""An affine posterior transform for scalarizing multi-output posteriors.

    For a Gaussian posterior with mean ``mu`` and covariance ``Sigma`` over
    ``m`` outputs, this yields the single-output posterior of
    ``offset + weights^T f`` (mean ``offset + weights^T mu``, covariance
    ``weights^T Sigma weights`` for every pair of points).

    Example:
        >>> weights = torch.tensor([0.5, 0.25], dtype=torch.float64)
        >>> posterior_transform = ScalarizedPosteriorTransform(weights)
        >>> EI = ExpectedImprovement(model, best_f=0.1, posterior_transform=posterior_transform)
    """

    scalarize = True

    def __init__(self, weights: Any, offset: float = 0.0) -> None:
        data, shape = to_flat(weights, "weights")
        if len(shape) != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.weights = weights
        self.offset = offset
        self._weights = [float(w) for w in data]
        self._offset = as_float(offset, "offset")

    def evaluate(self, Y: Any, X: Any = None) -> Any:
        """The transform applied to outcomes ``Y`` (``batch_shape x q x m``):
        ``offset + Y @ weights`` (shape ``batch_shape x q``)."""
        data, shape = to_flat(Y, "Y")
        m = len(self._weights)
        if not shape or shape[-1] != m:
            raise BotorchTensorDimensionError(
                f"Expected outcomes with {m} outputs (the number of weights), got shape {shape}."
            )
        out = [
            self._offset + sum(data[i * m + t] * self._weights[t] for t in range(m))
            for i in range(len(data) // m)
        ]
        return from_flat(out, tuple(shape[:-1]), kind_of(Y))

    def forward(self, posterior: Any, X: Any = None) -> Any:
        """The scalarized (single-output) posterior of a botorch_rs posterior."""
        from botorch_rs.posteriors.gpytorch import GPyTorchPosterior

        if not isinstance(posterior, GPyTorchPosterior):
            raise UnsupportedError("ScalarizedPosteriorTransform requires a botorch_rs posterior.")
        if posterior._transform is not None:
            raise UnsupportedError("The posterior has already been transformed.")
        if posterior._m != len(self._weights):
            raise RuntimeError(
                "Output shape not equal to that of weights. Output shape is "
                f"{posterior._m} and weights are {len(self._weights)}"
            )
        return GPyTorchPosterior(
            posterior._rs,
            posterior._x,
            posterior._kind,
            posterior._observation_noise,
            num_outputs=posterior._m,
            model_batch_shape=posterior._batch,
            output_indices=posterior._output_indices,
            transform=(self._weights, self._offset),
            points_per_input=posterior._points_per_input,
        )

    def __repr__(self) -> str:
        return f"ScalarizedPosteriorTransform(weights={self._weights}, offset={self._offset})"


def transform_arg(
    posterior_transform: Any, num_outputs: int | None = None
) -> tuple[list[float], float] | None:
    """The ``(weights, offset)`` of a posterior transform for the extension
    (``None`` without a transform), checked against ``num_outputs``."""
    if posterior_transform is None:
        return None
    if not isinstance(posterior_transform, ScalarizedPosteriorTransform):
        raise UnsupportedError(
            "botorch_rs supports posterior transforms of type "
            "botorch_rs.acquisition.objective.ScalarizedPosteriorTransform, got "
            f"{type(posterior_transform).__name__}."
        )
    weights = posterior_transform._weights
    if num_outputs is not None and len(weights) != num_outputs:
        raise RuntimeError(
            "Output shape not equal to that of weights. Output shape is "
            f"{num_outputs} and weights are {len(weights)}"
        )
    return weights, posterior_transform._offset


# -- MC objectives -------------------------------------------------------------


def _q_batch_check(output: Any, X: Any, is_mo: bool) -> None:
    q_batch_idx = -2 if is_mo else -1
    if X is not None and tuple(output.shape)[q_batch_idx] != tuple(X.shape)[-2]:
        raise RuntimeError(
            "The q-batch shape of the objective values does not agree with "
            f"the q-batch shape of X. Got {tuple(output.shape)[q_batch_idx]} and "
            f"{tuple(X.shape)[-2]}. This may happen if you used a one-to-many input "
            "transform but forgot to use a corresponding objective."
        )


class MCAcquisitionObjective:
    r"""Base class of the MC-based objectives.

    Subclasses implement ``forward(samples, X=None)``, mapping posterior
    samples (``sample_shape x batch_shape x q x m``) and optionally the inputs
    ``X`` (``batch_shape x q x d``) to objective values (``sample_shape x
    batch_shape x q``), on torch tensors. Acquisition functions call the
    objective from Rust through torch (so ``forward`` must be differentiable
    with torch.autograd for gradient-based optimization).

    Attributes:
        _verify_output_shape: If True and ``X`` is given, check that the q-batch
            shape of the objectives agrees with that of X.
        _is_mo: Whether the objectives are multi-output (not supported by the
            botorch_rs acquisition functions).
    """

    _verify_output_shape: bool = True
    _is_mo: bool = False

    def forward(self, samples: Any, X: Any = None) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, samples: Any, X: Any = None) -> Any:
        output = self.forward(samples, X=X)
        if self._verify_output_shape:
            _q_batch_check(output, X, self._is_mo)
        return output

    def _spec(self) -> tuple[Any, ...]:
        """The objective as passed to the extension: evaluated by calling it."""
        return ("callable", self)


class IdentityMCObjective(MCAcquisitionObjective):
    r"""Trivial objective extracting the last dimension (single-output samples).

    Example:
        >>> identity_objective = IdentityMCObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def forward(self, samples: Any, X: Any = None) -> Any:
        return samples.squeeze(-1)

    def _spec(self) -> tuple[Any, ...]:
        return ("identity",)

    def __repr__(self) -> str:
        return "IdentityMCObjective()"


class LinearMCObjective(MCAcquisitionObjective):
    r"""Linear objective ``sum_i weights[i] * samples[..., i]`` (evaluated in
    Rust by the acquisition functions).

    Example:
        >>> weights = torch.tensor([0.75, 0.25], dtype=torch.float64)
        >>> linear_objective = LinearMCObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = linear_objective(samples)
    """

    def __init__(self, weights: Any) -> None:
        data, shape = to_flat(weights, "weights")
        if len(shape) != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.weights = weights
        self._weights = [float(w) for w in data]

    def forward(self, samples: Any, X: Any = None) -> Any:
        if tuple(samples.shape)[-1] != len(self._weights):
            raise RuntimeError("Output shape of samples not equal to that of weights")
        if is_torch_tensor(samples):
            import torch

            w = torch.as_tensor(self._weights, dtype=samples.dtype, device=samples.device)
            return torch.einsum("...m, m", [samples, w])
        import numpy as np

        return np.asarray(samples) @ np.asarray(self._weights)

    def _spec(self) -> tuple[Any, ...]:
        return ("linear", list(self._weights))

    def __repr__(self) -> str:
        return f"LinearMCObjective(weights={self._weights})"


class GenericMCObjective(MCAcquisitionObjective):
    r"""Objective given by a callable ``f(samples, X)`` on torch tensors.

    Gradients of acquisition functions are propagated through the callable
    with torch.autograd.

    Example:
        >>> generic_objective = GenericMCObjective(
        ...     lambda Y, X: torch.sqrt(Y).sum(dim=-1),
        ... )
        >>> samples = sampler(posterior)
        >>> objective = generic_objective(samples)
    """

    def __init__(self, objective: Callable[..., Any]) -> None:
        self.objective = objective

    def forward(self, samples: Any, X: Any = None) -> Any:
        return self.objective(samples, X=X)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(objective={self.objective!r})"


def _eta_list(eta: Any, k: int) -> list[float]:
    """One temperature per constraint (a float is used for every constraint)."""
    if isinstance(eta, (int, float)):
        return [float(eta)] * k
    data, shape = to_flat(eta, "eta")
    if not shape:
        return [float(data[0])] * k
    return [float(e) for e in data]


class ConstrainedMCObjective(GenericMCObjective):
    r"""Feasibility-weighted objective (the legacy way of handling outcome
    constraints in MC acquisition functions):

    ``max(objective(samples) + infeasible_cost, 0) * prod_i sigmoid(-constraint_i(samples) / eta_i)
    - infeasible_cost``.

    Example:
        >>> bound = 0.0
        >>> objective = lambda Y, X=None: Y[..., 0]
        >>> # apply non-negativity constraint on f(x)[1]
        >>> constraint = lambda Y: bound - Y[..., 1]
        >>> constrained_objective = ConstrainedMCObjective(objective, [constraint])
    """

    def __init__(
        self,
        objective: Callable[..., Any],
        constraints: list[Callable[[Any], Any]],
        infeasible_cost: Any = 0.0,
        eta: Any = 1e-3,
    ) -> None:
        super().__init__(objective=objective)
        self.constraints = constraints
        self.eta = eta
        self.infeasible_cost = infeasible_cost
        self._eta = _eta_list(eta, len(constraints))
        self._infeasible_cost = as_float(infeasible_cost, "infeasible_cost")

    def forward(self, samples: Any, X: Any = None) -> Any:
        import torch

        obj = self.objective(samples, X=None)
        if len(self._eta) != len(self.constraints):
            raise ValueError(
                "Number of provided constraints and number of provided etas do not match."
            )
        if not all(e > 0 for e in self._eta):
            raise ValueError("eta must be positive.")
        log_ind = torch.zeros_like(samples[..., 0])
        for constraint, eta in zip(self.constraints, self._eta, strict=True):
            log_ind = log_ind + torch.nn.functional.logsigmoid(-constraint(samples) / eta)
        cost = self._infeasible_cost
        return obj.add(cost).clamp_min(0).mul(log_ind.exp()).add(-cost)

    def _spec(self) -> tuple[Any, ...]:
        return (
            "constrained",
            self.objective,
            list(self.constraints),
            self._infeasible_cost,
            list(self._eta),
        )


def _botorch_class_names(obj: Any) -> list[str]:
    """Names of the BoTorch classes in the MRO of ``obj``."""
    return [
        c.__name__ for c in type(obj).__mro__ if getattr(c, "__module__", "").startswith("botorch.")
    ]


def _floats(x: Any, name: str) -> list[float]:
    data, _ = to_flat(x, name)
    return [float(v) for v in data]


def objective_arg(objective: Any) -> tuple[Any, ...] | None:
    """The specification of an MC objective for the extension: ``None``,
    ``("identity",)``, ``("linear", weights)``, ``("callable", f)`` with
    ``f(samples, X=X)``, or ``("constrained", f, constraints,
    infeasible_cost, etas)``. BoTorch's own objectives are converted."""
    if objective is None:
        return None
    spec: tuple[Any, ...]
    if isinstance(objective, MCAcquisitionObjective):
        if objective._is_mo:
            raise UnsupportedError("Multi-output MC objectives are not supported by botorch_rs.")
        spec = objective._spec()
    else:
        names = _botorch_class_names(objective)
        if "MCAcquisitionObjective" not in names:
            raise UnsupportedError(
                "`objective` must be an MCAcquisitionObjective (of botorch_rs or BoTorch), "
                f"got {type(objective).__name__}."
            )
        if getattr(objective, "_is_mo", False):
            raise UnsupportedError("Multi-output MC objectives are not supported by botorch_rs.")
        name = type(objective).__name__ if type(objective).__name__ in names else None
        if name == "ConstrainedMCObjective":
            spec = (
                "constrained",
                objective.objective,
                list(objective.constraints),
                float(objective.infeasible_cost),
                _floats(objective.eta, "eta"),
            )
        elif name == "LinearMCObjective":
            spec = ("linear", _floats(objective.weights, "weights"))
        elif name == "IdentityMCObjective":
            spec = ("identity",)
        else:
            from botorch_rs.acquisition.risk_measures import risk_measure_from_botorch

            # BoTorch's risk measures are computed in Rust.
            native = risk_measure_from_botorch(objective)
            spec = ("callable", objective) if native is None else native._spec()
    if spec[0] in ("callable", "constrained"):
        from botorch_rs._torch_bridge import require_torch

        require_torch()
    return spec


def is_constrained_objective(objective: Any) -> bool:
    """Whether ``objective`` is a ``ConstrainedMCObjective`` (of botorch_rs or
    BoTorch)."""
    return isinstance(objective, ConstrainedMCObjective) or (
        objective is not None
        and not isinstance(objective, MCAcquisitionObjective)
        and "ConstrainedMCObjective" in _botorch_class_names(objective)
    )


def constraints_arg(
    constraints: Sequence[Callable[[Any], Any]] | None, eta: Any
) -> tuple[list[Any] | None, list[float] | None]:
    """Outcome constraints (callables on torch tensors) and one temperature
    per constraint for the extension."""
    if constraints is None:
        return None, None
    if callable(constraints) or not isinstance(constraints, Sequence):
        raise TypeError("`constraints` must be a list of callables.")
    cons = list(constraints)
    for c in cons:
        if not callable(c):
            raise TypeError(f"Constraints must be callables, got {type(c).__name__}.")
    if cons:
        from botorch_rs._torch_bridge import require_torch

        require_torch()
    return cons, _eta_list(eta, len(cons))


def fat_arg(fat: Any) -> tuple[bool, list[bool | None] | None]:
    """BoTorch's ``fat`` (a bool, or one entry per constraint: ``True``
    (fatmoid), ``False`` (sigmoid) or ``None`` (the constraint returns
    probabilities)) as ``(bool(fat), per-constraint list or None)``."""
    if isinstance(fat, (list, tuple)):
        return bool(fat), [None if f is None else bool(f) for f in fat]
    return bool(fat), None


__all__ = [
    "ConstrainedMCObjective",
    "GenericMCObjective",
    "IdentityMCObjective",
    "LinearMCObjective",
    "MCAcquisitionObjective",
    "PosteriorTransform",
    "ScalarizedPosteriorTransform",
]
