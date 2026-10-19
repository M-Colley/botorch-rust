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


def _clamp_min_zero(x: Any) -> Any:
    """``x.clamp_min(0)`` of a torch tensor with PyTorch ≥ 2.14's subgradient
    at the bound under any torch version: the gradient passes where ``x > 0``
    only (torch ≤ 2.13 also passed it at ``x == 0``), as in the Rust
    computations. The value is the native one."""
    import torch

    clamped = x.clamp_min(0)
    if not (torch.is_grad_enabled() and x.requires_grad):
        return clamped
    return torch.where(x > 0, x, clamped.detach())


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
        m = len(self._weights)
        if is_torch_tensor(Y) and Y.ndim > 0 and Y.shape[-1] == m:
            import torch

            # BoTorch's ``offset + Y @ weights`` (the same rounding).
            w = torch.as_tensor(self._weights, dtype=Y.dtype, device=Y.device)
            return self._offset + Y @ w
        data, shape = to_flat(Y, "Y")
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
        return _clamp_min_zero(obj.add(cost)).mul(log_ind.exp()).add(-cost)

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
        elif name == "PenalizedMCObjective" and getattr(objective, "expand_dim", None) is None:
            from botorch_rs.acquisition.penalized import penalized_objective_spec

            spec = penalized_objective_spec(
                objective.objective,
                objective.penalty_objective,
                objective.regularization_parameter,
            )
        else:
            from botorch_rs.acquisition.risk_measures import risk_measure_from_botorch

            # BoTorch's risk measures are computed in Rust.
            native = risk_measure_from_botorch(objective)
            spec = ("callable", objective) if native is None else native._spec()
    if spec[0] in ("callable", "constrained", "penalized"):
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


DEFAULT_NUM_PREF_SAMPLES = 16

LEARNED_OBJECTIVE_PREF_MODEL_MIXED_DTYPE_WARN = (
    "pref_model has double-precision data, but single-precision data "
    "was passed to the LearnedObjective. Upcasting to double."
)


def _is_deterministic_model(model: Any) -> bool:
    from botorch_rs.models.deterministic import DeterministicModel

    return isinstance(model, DeterministicModel)


def _learned_samples_autograd(handle: Any, Y: Any, z: Any, out_shape: tuple[int, ...]) -> Any:
    """``LearnedObjective``'s samples connected to torch's autograd (the
    exact vector-Jacobian product of the preference posterior's samples
    with respect to the outcome samples, from Rust)."""
    import torch

    from botorch_rs import _botorch_rs as _rs

    class _LearnedSamples(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, Yt: Any) -> Any:  # type: ignore[override]
            ctx.save_for_backward(Yt)
            (vals, _), _ = _rs.learned_objective_samples(handle, to_flat(Yt), z)
            return torch.tensor(vals, dtype=Yt.dtype).reshape(out_shape)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Any:  # type: ignore[override]
            (Yt,) = ctx.saved_tensors
            adj = to_flat(grad_out.reshape(out_shape))
            _, grads = _rs.learned_objective_samples(handle, to_flat(Yt), z, adj)
            assert grads is not None
            return torch.tensor(list(grads[0]), dtype=Yt.dtype).reshape(Yt.shape)

    return _LearnedSamples.apply(Y)


class LearnedObjective(MCAcquisitionObjective):
    r"""Learned preference objective constructed from a preference model.

    For input ``samples``, it samples each individual sample again from the
    latent preference posterior distribution using ``pref_model`` and return the
    posterior mean.

    The preference posterior is sampled in Rust with the base samples of an
    ``IIDNormalSampler`` (torch's draws for ``seed``, as in BoTorch), and its
    exact vector-Jacobian product connects the result to torch's autograd.
    botorch_rs's Monte-Carlo acquisition functions do not accept it (its
    values have ``sample_shape`` times more samples than its input).

    Example:
        >>> train_X = torch.rand(2, 2)
        >>> train_comps = torch.LongTensor([[0, 1]])
        >>> pref_model = PairwiseGP(train_X, train_comps)
        >>> learned_pref_obj = LearnedObjective(pref_model)
        >>> samples = sampler(posterior)
        >>> objective = learned_pref_obj(samples)
    """

    def __init__(
        self,
        pref_model: Any,
        sample_shape: Any = None,
        seed: int | None = None,
    ) -> None:
        r"""
        Args:
            pref_model: A BoTorch model, which models the latent preference/utility
                function. Given an input tensor of size
                ``sample_size x batch_shape x q x d``, its ``posterior`` method should
                return a ``Posterior`` object with single outcome representing the
                utility values of the input.
            sample_shape: Determines the number of preference-model samples drawn
                *per outcome-model sample* when the ``LearnedObjective`` is called.
                Note that this is an additional layer of sampling relative to what
                is needed when evaluating most MC acquisition functions in order to
                account for uncertainty in the preference model. If ``None``, it will
                default to ``torch.Size([16])``, so that 16 samples will be drawn
                from the preference model at each outcome sample. This number is
                relatively high because sampling from the preference model is general
                cheap relative to generating the outcome model posterior.
            seed: The seed of the sampler.
        """
        from botorch_rs.sampling.normal import IIDNormalSampler

        self.pref_model = pref_model
        self.sampler: Any
        if _is_deterministic_model(pref_model):
            if sample_shape is not None:
                raise ValueError("sample_shape must be None for DeterministicModel.")
            self.sampler = None
        else:
            if sample_shape is None:
                sample_shape = (DEFAULT_NUM_PREF_SAMPLES,)
            # using an IIDNormalSampler instead of a SobolQMCNormalSampler by default
            # because SobolQMCNormalSampler can support up to 21201 total samples and
            # becomes noticeably slower than uniform sampling when the sample size is
            # large.
            self.sampler = IIDNormalSampler(sample_shape=sample_shape, seed=seed)
            self.sampler.batch_range_override = (1, -1)

    def forward(self, samples: Any, X: Any = None) -> Any:
        r"""Sample each element of samples.

        Args:
            samples: A ``sample_size x batch_shape x q x d``-dim Tensors of
                samples from a model posterior.

        Returns:
            A ``(sample_size * num_samples) x batch_shape x q``-dim Tensor of
            objective values sampled from utility posterior using ``pref_model``.
        """
        import math
        import warnings

        from botorch_rs import _botorch_rs as _rs
        from botorch_rs.exceptions.warnings import InputDataWarning

        if is_torch_tensor(samples):
            import torch

            if samples.dtype != torch.float64:
                warnings.warn(
                    LEARNED_OBJECTIVE_PREF_MODEL_MIXED_DTYPE_WARN,
                    InputDataWarning,
                    stacklevel=2,
                )
                samples = samples.to(torch.float64)
        shape = tuple(to_flat(samples, "samples")[1])
        if len(shape) < 3:
            raise ValueError("samples should have at least 3 dimensions.")
        posterior = self.pref_model.posterior(samples)
        if self.sampler is None:
            data, mshape = to_flat(posterior.mean, "mean")
            return from_flat(data, mshape[:-1], kind_of(samples))
        # The sampler's base samples (sample_shape x S x 1 ... x q, shared
        # by the t-batches), broadcast to every set of outcome samples.
        self.sampler(posterior)
        z_col, _ = to_flat(self.sampler.base_samples, "base_samples")
        n = math.prod(self.sampler.sample_shape)
        s0 = shape[0]
        nb = math.prod(shape[1:-2])
        q = shape[-2]
        z: list[float] = []
        for i in range(n):
            for s in range(s0):
                block = list(z_col[(i * s0 + s) * q : (i * s0 + s + 1) * q])
                for _ in range(nb):
                    z.extend(block)
        zpair = (z, (n, s0 * nb, q))
        handle = self.pref_model._acquisition_model(None)
        out_shape = (n * s0, *shape[1:-2], q)
        if is_torch_tensor(samples) and samples.requires_grad:
            import torch

            if torch.is_grad_enabled():
                Y = samples.reshape(s0 * nb, q, shape[-1])
                return _learned_samples_autograd(handle, Y, zpair, out_shape)
        flat = to_flat(samples, "samples")
        y = (flat[0], (s0 * nb, q, shape[-1]))
        (vals, _), _ = _rs.learned_objective_samples(handle, y, zpair)
        return from_flat(vals, out_shape, kind_of(samples))

    def _spec(self) -> tuple[Any, ...]:
        raise UnsupportedError(
            "LearnedObjective is not supported in botorch_rs's Monte-Carlo acquisition "
            "functions: its values have sample_shape times more samples than the posterior "
            "samples. Use it directly, or qExpectedUtilityOfBestOption with the preference "
            "model."
        )


__all__ = [
    "ConstrainedMCObjective",
    "GenericMCObjective",
    "IdentityMCObjective",
    "LearnedObjective",
    "LinearMCObjective",
    "MCAcquisitionObjective",
    "PosteriorTransform",
    "ScalarizedPosteriorTransform",
]
