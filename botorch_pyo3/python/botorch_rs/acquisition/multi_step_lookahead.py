r"""
A general implementation of multi-step look-ahead acquisition function with
configurable value functions (``botorch.acquisition.multi_step_lookahead``).
See [Jiang2020multistep]_.

The fantasy tree is evaluated in Rust: the fantasy models are exact GPs
conditioned on the fantasized observations of all levels, and the values
and gradients with respect to every point of the tree are exact.

.. [Jiang2020multistep]
    S. Jiang, D. R. Jiang, M. Balandat, B. Karrer, J. Gardner, and R. Garnett.
    Efficient Nonmyopic Bayesian Optimization via One-Shot Multi-Step Trees.
    In Advances in Neural Information Processing Systems 33, 2020.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
    pending_arg,
)
from botorch_rs.acquisition.knowledge_gradient import (
    NATIVE_VALUE_FUNCTIONS,
    _is_mc_objective,
    _scalar,
)
from botorch_rs.acquisition.monte_carlo import MCAcquisitionFunction
from botorch_rs.acquisition.objective import objective_arg, transform_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.sampling.normal import NormalMCSampler, SobolQMCNormalSampler

#: ``(model, X) -> kwargs`` of a stage value function.
TAcqfArgConstructor = Callable[..., dict[str, Any]]


def _class_names(cls: Any) -> list[str]:
    """The names of the classes in the MRO of the class ``cls`` (of
    botorch_rs or BoTorch)."""
    return [c.__name__ for c in getattr(cls, "__mro__", ())]


def _is_lookahead_class(cls: Any) -> bool:
    return cls is qMultiStepLookahead or (
        getattr(cls, "__name__", None) == "qMultiStepLookahead"
        and getattr(cls, "__module__", "").startswith("botorch.")
    )


def _is_analytic_class(cls: Any) -> bool:
    return "AnalyticAcquisitionFunction" in _class_names(cls)


def _is_make_best_f(argfac: Any) -> bool:
    """Whether ``argfac`` is ``make_best_f`` (of botorch_rs or BoTorch)."""
    return argfac is make_best_f or (
        getattr(argfac, "__name__", None) == "make_best_f"
        and getattr(argfac, "__module__", "") == "botorch.acquisition.multi_step_lookahead"
    )


def _construct_inner_samplers(
    batch_sizes: list[int],
    valfunc_cls: list[Any],
    inner_mc_samples: list[int | None],
    objective: Any = None,
) -> list[SobolQMCNormalSampler | None]:
    r"""Check validity of inputs and construct inner samplers.

    Helper function to be used internally for constructing inner samplers.

    Args:
        batch_sizes: A list ``[q_1, ..., q_k]`` containing the batch sizes for the
            ``k`` look-ahead steps.
        valfunc_cls: A list of ``k + 1`` acquisition function classes to be used as the
            (stage + terminal) value functions. Each element (except for the last one)
            can be ``None``, in which case a zero stage value is assumed for the
            respective stage.
        inner_mc_samples: A list ``[n_0, ..., n_k]`` containing the number of MC
            samples to be used for evaluating the stage value function. Ignored if
            the objective is ``None``.
        objective: The objective under which the output is evaluated. If ``None``, use
            the model output (requires a single-output model or a posterior transform).
            Otherwise the objective is MC-evaluated (using ``inner_sampler``).

    Returns:
        A list with ``k + 1`` elements that are either ``MCSampler``s or ``None``.
    """
    inner_samplers: list[SobolQMCNormalSampler | None] = []
    for q, vfc, mcs in zip([None, *batch_sizes], valfunc_cls, inner_mc_samples, strict=False):
        if vfc is None:
            inner_samplers.append(None)
        elif _is_lookahead_class(vfc):
            raise UnsupportedError(
                "qMultiStepLookahead not supported as a value function "
                "(I see what you did there, nice try...)."
            )
        elif _is_analytic_class(vfc):
            if objective is not None:
                raise UnsupportedError(
                    "Only PosteriorTransforms are supported for analytic value "
                    f"functions. Received a {objective.__class__.__name__}."
                )
            # At this point, we don't know the initial q-batch size here
            if q is not None and q > 1:
                raise UnsupportedError(
                    "Only batch sizes of q=1 are supported for analytic value functions."
                )
            if q is not None and mcs is not None:
                warnings.warn(
                    "inner_mc_samples is ignored for analytic acquisition functions",
                    BotorchWarning,
                    stacklevel=3,
                )
            inner_samplers.append(None)
        else:
            inner_sampler = SobolQMCNormalSampler(sample_shape=(32 if mcs is None else mcs,))
            inner_samplers.append(inner_sampler)
    return inner_samplers


def _sample_weights(sampler: Any) -> list[float] | None:
    """The ``base_weights`` of a sampler (e.g. of Gauss-Hermite quadrature),
    if it has them."""
    weights = getattr(sampler, "base_weights", None)
    if weights is None:
        return None
    return [float(w) for w in to_flat(weights, "base_weights")[0]]


class qMultiStepLookahead(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""MC-based batch Multi-Step Look-Ahead (one-shot optimization)."""

    def __init__(
        self,
        model: Any,
        batch_sizes: list[int],
        num_fantasies: list[int] | None = None,
        samplers: list[SobolQMCNormalSampler] | None = None,
        valfunc_cls: list[Any] | None = None,
        valfunc_argfacs: list[TAcqfArgConstructor | None] | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        inner_mc_samples: list[int | None] | None = None,
        X_pending: Any = None,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""q-Multi-Step Look-Ahead (one-shot optimization).

        Performs a ``k``-step lookahead by means of repeated fantasizing.

        Allows to specify the stage value functions by passing the respective class
        objects via the ``valfunc_cls`` list. Optionally,
        ``valfunc_argfacs`` takes a list of callables that generate
        additional kwargs for these constructors. By default,
        ``valfunc_cls`` will be chosen as ``[None, ..., None, PosteriorMean]``,
        which corresponds to the (parallel) multi-step KnowledgeGradient.
        If, in addition,
        ``k=1`` and ``q_1 = 1``, this reduces to the classic Knowledge Gradient.

        WARNING: The complexity of evaluating this function is exponential in the number
        of lookahead steps!

        botorch_rs supports the value functions of the knowledge gradient
        (``NATIVE_VALUE_FUNCTIONS``, of botorch_rs or BoTorch) on models of
        exact GPs (``SingleTaskGP``, ``ModelListGP``). ``make_best_f`` as an
        argument factory gives every fantasy model the best value of its
        (observed and fantasized) data in the outcome space of its posterior;
        other argument factories are called once, with the current model and
        ``X=None`` (BoTorch calls them with every fantasy model), so their
        arguments must not depend on the fantasized observations. The
        samplers' ``base_weights`` weight the fantasies, as in BoTorch.

        Args:
            model: A fitted model.
            batch_sizes: A list ``[q_1, ..., q_k]`` containing the batch sizes for the
                ``k`` look-ahead steps.
            num_fantasies: A list ``[f_1, ..., f_k]`` containing the number of fantasy
                points to use for the ``k`` look-ahead steps.
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            valfunc_cls: A list of ``k + 1`` acquisition function classes
                to be used as the (stage + terminal) value functions. Each
                element (except for the last one) can be ``None``, in which case
                a zero stage value is assumed for the respective stage. If
                ``None``, this defaults to ``[None, ..., None, PosteriorMean]``
            valfunc_argfacs: A list of ``k + 1`` "argument factories",
                i.e. callables that map a ``Model`` and input tensor ``X`` to a
                dictionary of kwargs for the respective stage value function
                constructor (e.g. ``best_f`` for ``ExpectedImprovement``). If
                None, only the standard (``model``, ``sampler`` and ``objective``)
                kwargs will be used.
            objective: The objective under which the output is evaluated.
                If ``None``, use the model output (requires a single-output
                model or a posterior transform). Otherwise the objective is
                MC-evaluated (using ``inner_sampler``).
            posterior_transform: An optional PosteriorTransform. If given,
                this transforms the posterior before evaluation. If ``objective
                is None``, then the output of the transformed posterior is used.
                If ``objective`` is given, the ``inner_sampler`` is used to draw
                samples from the transformed posterior, which are then evaluated
                under the ``objective``.
            inner_mc_samples: A list ``[n_0, ..., n_k]`` containing the
                number of MC samples to be used for evaluating the stage value
                function. Ignored if the objective is ``None``.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have
                points that have been submitted for function evaluation but have
                not yet been evaluated. Concatenated into ``X`` upon forward
                call. Copied and set to have no gradient.
            collapse_fantasy_base_samples: If True, collapse_batch_dims of the Samplers
                will be applied on fantasy batch dimensions as well, meaning that base
                samples are the same in all subtrees starting from the same level.
        """
        from botorch_rs.acquisition.analytic import PosteriorMean

        if objective is not None and not _is_mc_objective(objective):
            raise UnsupportedError(
                "`qMultiStepLookahead` got a non-MC `objective`. This is not supported."
                " Use `posterior_transform` and `objective=None` instead."
            )
        from botorch_rs.models.pairwise_gp import PairwiseGP

        if any(isinstance(m, PairwiseGP) for m in getattr(model, "models", [model])):
            # BoTorch fails when it fantasizes (``PairwiseGP`` conditions on
            # comparisons only, not on sampled outcomes).
            raise RuntimeError("Conditioning on non-pairwise comparison observations.")
        AcquisitionFunction.__init__(self, model)
        self.batch_sizes = list(batch_sizes)
        if not ((num_fantasies is None) ^ (samplers is None)):
            raise UnsupportedError(
                "qMultiStepLookahead requires exactly one of `num_fantasies` or "
                "`samplers` as arguments."
            )
        if samplers is None:
            assert num_fantasies is not None
            samplers = [SobolQMCNormalSampler(sample_shape=(nf,)) for nf in num_fantasies]
        else:
            for s in samplers:
                if not isinstance(s, NormalMCSampler):
                    raise UnsupportedError(
                        "The samplers of qMultiStepLookahead must be botorch_rs "
                        "SobolQMCNormalSamplers."
                    )
            num_fantasies = [s.sample_shape[0] for s in samplers]
        self.num_fantasies = list(num_fantasies)
        # By default do not use stage values and use PosteriorMean as terminal value
        # function (= multi-step KG)
        if valfunc_cls is None:
            valfunc_cls = [None for _ in self.num_fantasies] + [PosteriorMean]
        if inner_mc_samples is None:
            inner_mc_samples = [None] * (1 + len(self.num_fantasies))
        inner_samplers = _construct_inner_samplers(
            batch_sizes=self.batch_sizes,
            valfunc_cls=list(valfunc_cls),
            objective=objective,
            inner_mc_samples=list(inner_mc_samples),
        )
        if valfunc_argfacs is None:
            valfunc_argfacs = [None] * (1 + len(self.batch_sizes))
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.samplers = list(samplers)
        self.inner_samplers = inner_samplers
        self._valfunc_cls = list(valfunc_cls)
        self._valfunc_argfacs = list(valfunc_argfacs)
        self._collapse_fantasy_base_samples = bool(collapse_fantasy_base_samples)
        self._constraints = None
        rs_model = model._acquisition_model(None)
        pending = pending_arg(X_pending, rs_model.dim)
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        stages = [
            self._stage_arg(vfc, argfac, sampler)
            for vfc, argfac, sampler in zip(
                self._valfunc_cls, self._valfunc_argfacs, inner_samplers, strict=False
            )
        ]
        self._rs, self._state = _rs.q_multi_step_lookahead(
            rs_model,
            [int(q) for q in self.batch_sizes],
            samplers=[s._rs for s in self.samplers],
            sample_weights=[_sample_weights(s) for s in self.samplers],
            value_functions=stages,
            objective=objective_arg(objective),
            transform=transform_arg(posterior_transform, model.num_outputs),
            x_pending=pending,
            collapse_fantasy_base_samples=self._collapse_fantasy_base_samples,
        )

    def _stage_arg(self, valfunc_cls: Any, argfac: Any, sampler: Any) -> Any:
        """``(name, kwargs, best_f_from_model, sampler)`` of a stage value
        function for the extension (``None`` without one)."""
        if valfunc_cls is None:
            return None
        name = getattr(valfunc_cls, "__name__", str(valfunc_cls))
        if name not in NATIVE_VALUE_FUNCTIONS:
            raise UnsupportedError(
                f"`valfunc_cls` {name} is not supported by botorch_rs (supported: "
                f"{', '.join(NATIVE_VALUE_FUNCTIONS)})."
            )
        from_model = False
        kwargs: dict[str, Any] = {}
        if argfac is not None:
            if _is_make_best_f(argfac):
                from_model = True
                kwargs = {"best_f": 0.0}
            else:
                kwargs = dict(argfac(model=self.model, X=None))
        allowed = {"best_f", "beta", "tau", "maximize", "weights"}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise UnsupportedError(
                f"Unsupported arguments of the value function {name}: {unknown}."
            )
        return (
            name,
            {k: _scalar(v) for k, v in kwargs.items()},
            from_model,
            None if sampler is None else sampler._rs,
        )

    @property
    def _num_auxiliary(self) -> int:
        r"""Number of auxiliary variables in the q-batch dimension.

        Returns:
             ``q_aux`` s.t. ``q + q_aux = augmented_q_batch_size``
        """
        total, prod = 0, 1
        for q, f in zip(self.batch_sizes, self.num_fantasies, strict=True):
            prod *= f
            total += q * prod
        return total

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions): ``q_0 + f_1 q_1 + f_2 f_1 q_2 + ...``
        """
        return q + self._num_auxiliary

    def get_split_shapes(self, X: Any) -> tuple[tuple[int, ...], list[tuple[int, ...]], list[int]]:
        r"""Get the split shapes from X.

        Args:
            X: A ``batch_shape x q_aug x d``-dim tensor including fantasy points.

        Returns:
            A 3-tuple ``(batch_shape, shapes, sizes)``, where
            ``shape[i] = f_i x .... x f_1 x batch_shape x q_i x d`` and
            ``size[i] = f_i * ... f_1 * q_i``.
        """
        shape = tuple(to_flat(X, "X")[1]) if not hasattr(X, "shape") else tuple(X.shape)
        batch_shape, (q_aug, d) = shape[:-2], shape[-2:]
        q = q_aug - self._num_auxiliary
        batch_sizes = [q, *self.batch_sizes]
        # X_i needs to have shape f_i x .... x f_1 x batch_shape x q_i x d
        shapes = [
            (*self.num_fantasies[:i][::-1], *batch_shape, q_i, d)
            for i, q_i in enumerate(batch_sizes)
        ]
        # Each X_i in the split X has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        sizes = [math.prod(s[: (-2 - len(batch_shape))]) * s[-2] for s in shapes]
        return batch_shape, shapes, sizes

    def get_multi_step_tree_input_representation(self, X: Any) -> list[Any]:
        r"""Get the multi-step tree representation of X.

        Args:
            X: A ``batch_shape x q' x d``-dim Tensor with ``q'`` design points for each
                batch, where ``q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...``. Here ``q_i``
                is the number of candidates jointly considered in look-ahead step
                ``i``, and ``f_i`` is respective number of fantasies.

        Returns:
            A list ``[X_j, ..., X_k]`` of tensors, where ``X_i`` has shape
            ``f_i x .... x f_1 x batch_shape x q_i x d``: ``X_i[i_i, ..., i_1, b]``
            are the points of rows ``ψ q_i .. (ψ + 1) q_i`` of the ``i``-th block of
            ``X[b]``, ``ψ = i_1 + f_1 i_2 + ... + f_1 ⋯ f_{i-1} i_i`` (for ``q_i = 1``
            or a single t-batch this is BoTorch's reshape, which otherwise mixes the
            blocks of different t-batches).
        """
        batch_shape, shapes, sizes = self.get_split_shapes(X=X)
        nb = len(batch_shape)
        if is_torch_tensor(X):
            import torch

            Xsplit = torch.split(X, sizes, dim=-2)
            X0 = Xsplit[0].reshape(shapes[0])
            Xother = []
            for Xi, shape in zip(Xsplit[1:], shapes[1:], strict=True):
                nf = len(shape) - nb - 2
                Xi = Xi.reshape(*batch_shape, *shape[:nf], shape[-2], shape[-1])
                perm = [nb + i for i in range(nf)] + list(range(nb)) + [nb + nf, nb + nf + 1]
                Xother.append(Xi.permute(*perm))
            pending = self.X_pending
            if pending is not None:
                pending = torch.as_tensor(pending, dtype=X0.dtype, device=X0.device)
                X0 = torch.cat([X0, pending.expand(*batch_shape, *pending.shape)], dim=-2)
            return [X0, *Xother]
        data, _ = to_flat(X, "X")
        kind = kind_of(X)
        d = shapes[0][-1]
        q_aug = sum(sizes)
        nbatch = math.prod(batch_shape)
        out = []
        offset = 0
        for i, (shape, size) in enumerate(zip(shapes, sizes, strict=True)):
            q_i = shape[-2]
            vals: list[float] = []
            # Node `psi` (row-major over the fantasy indices, newest first)
            # of every t-batch.
            for psi in range(size // q_i):
                for b in range(nbatch):
                    start = (b * q_aug + offset + psi * q_i) * d
                    vals.extend(data[start : start + q_i * d])
            if i == 0 and self.X_pending is not None:
                pdata, pshape = to_flat(self.X_pending, "X_pending")
                merged: list[float] = []
                for b in range(nbatch):
                    merged.extend(vals[b * q_i * d : (b + 1) * q_i * d])
                    merged.extend(pdata)
                vals = merged
                shape = (*batch_shape, q_i + pshape[0], d)
            out.append(from_flat(vals, shape, kind))
            offset += size
        return out

    def extract_candidates(self, X_full: Any) -> Any:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A ``batch_shape x q' x d``-dim Tensor with ``q'`` design points for
                each batch, where ``q' = q + f_1 q_1 + f_2 f_1 q_2 + ...``.

        Returns:
            A ``batch_shape x q x d``-dim Tensor with ``q`` design points for
            each batch.
        """
        if hasattr(X_full, "shape") and (is_torch_tensor(X_full) or hasattr(X_full, "ndim")):
            return X_full[..., : -self._num_auxiliary, :]
        data, shape = to_flat(X_full, "X_full")
        rows, d = shape[-2], shape[-1]
        q = rows - self._num_auxiliary
        out: list[float] = []
        for b in range(len(data) // (rows * d)):
            out.extend(data[b * rows * d : b * rows * d + q * d])
        return from_flat(out, (*shape[:-2], q, d), kind_of(X_full))

    def get_induced_fantasy_model(self, X: Any) -> Any:
        r"""Fantasy model induced by X.

        Args:
            X: A ``batch_shape x q' x d``-dim Tensor with ``q'`` design points for each
                batch, where ``q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...``. Here ``q_i``
                is the number of candidates jointly considered in look-ahead step
                ``i``, and ``f_i`` is respective number of fantasies.

        Returns:
            The fantasy model induced by X (a batched fantasy model of
            botorch_rs, built with the samplers' base samples shared by all
            nodes of a level).
        """
        if not self._collapse_fantasy_base_samples:
            raise UnsupportedError(
                "get_induced_fantasy_model requires collapse_fantasy_base_samples=True "
                "in botorch_rs."
            )
        Xs = self.get_multi_step_tree_input_representation(X)
        return _get_induced_fantasy_model(model=self.model, Xs=Xs, samplers=self.samplers)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(batch_sizes={self.batch_sizes}, "
            f"num_fantasies={self.num_fantasies})"
        )


def _get_induced_fantasy_model(model: Any, Xs: list[Any], samplers: list[Any]) -> Any:
    r"""Recursive computation of the fantasy model induced by an input tree.

    Args:
        model: A Model of appropriate batch size. Specifically, it must be
            possible to evaluate the model's posterior at ``Xs[0]``.
        Xs: A list ``[X_j, ..., X_k]`` of tensors, where ``X_i`` has shape
            ``f_i x .... x f_1 x batch_shape x q_i x d``.
        samplers: A list of ``k - j`` samplers, such that the number of
            samples of sampler ``i`` is ``f_i``. The last element of this list
            is considered the "inner sampler", which is used for evaluating
            the objective in case it is an MCAcquisitionObjective.

    Returns:
        A Model obtained by iteratively fantasizing over the input tree ``Xs``.
    """
    if len(Xs) == 1:
        return model
    fantasy_model = model.fantasize(X=Xs[0], sampler=samplers[0])
    return _get_induced_fantasy_model(model=fantasy_model, Xs=Xs[1:], samplers=samplers[1:])


def warmstart_multistep(
    acq_function: qMultiStepLookahead,
    bounds: Any,
    num_restarts: int,
    raw_samples: int,
    full_optimizer: Any,
) -> Any:
    r"""Warm-start initialization for multi-step look-ahead acquisition functions.

    For now uses the same q' as in ``full_optimizer``. TODO: allow different ``q``.

    The random perturbations use the random number generator of the array
    library of ``full_optimizer`` (torch's for tensors, so that
    ``torch.manual_seed`` reproduces BoTorch's draws), else Python's
    ``random``.

    Args:
        acq_function: A qMultiStepLookahead acquisition function.
        bounds: A ``2 x d`` tensor of lower and upper bounds for each column
            of features.
        num_restarts: The number of starting points for multistart
            acquisition function optimization.
        raw_samples: The number of raw samples to consider in the
            initialization heuristic.
        full_optimizer: The full tree of optimizers of the previous iteration
            of shape ``batch_shape x q' x d``. Typically obtained by passing
            ``return_best_only=False`` and ``return_full_tree=True`` into
            ``optimize_acqf``.

    Returns:
        A ``num_restarts x q' x d`` tensor for initial points for optimization.

    This is a very simple initialization heuristic.
    TODO: Use the observed values to identify the fantasy sub-tree that is closest to
    the observed value.
    """
    from botorch_rs.optim.initializers import initialize_q_batch

    batch_shape, shapes, sizes = acq_function.get_split_shapes(full_optimizer)
    if is_torch_tensor(full_optimizer):
        X_full = _warmstart_tree_torch(full_optimizer, bounds, raw_samples, batch_shape, sizes)
    else:
        X_full = _warmstart_tree_python(full_optimizer, bounds, raw_samples, batch_shape, sizes)
    acq_vals = acq_function(X_full)
    if is_torch_tensor(acq_vals):
        acq_vals = acq_vals.detach()
    X_init, _ = initialize_q_batch(X=X_full, acq_vals=acq_vals, n=num_restarts, eta=1.0)
    if is_torch_tensor(X_init):
        return X_init[:raw_samples]
    data, shape = to_flat(X_init, "X_init")
    n = min(shape[0], raw_samples)
    per = math.prod(shape[1:])
    return from_flat(data[: n * per], (n, *shape[1:]), kind_of(X_init))


def _warmstart_tree_torch(
    full_optimizer: Any, bounds: Any, raw_samples: int, batch_shape: Sequence[int], sizes: list[int]
) -> Any:
    """BoTorch's perturbed copies of the full tree (torch's generator)."""
    import torch
    from torch.distributions import Beta

    Xopts = torch.split(full_optimizer, sizes, dim=-2)
    tkwargs: dict[str, Any] = {"device": Xopts[0].device, "dtype": Xopts[0].dtype}
    bounds = torch.as_tensor(bounds, **tkwargs)

    B = Beta(torch.ones(1, **tkwargs), 3 * torch.ones(1, **tkwargs))

    def unnormalize(X: Any) -> Any:
        return X * (bounds[1] - bounds[0]) + bounds[0]

    def mixin_layer(X: Any, eta: Any) -> Any:
        perturbations = unnormalize(B.sample(X.shape).squeeze(-1))
        return (1 - eta) * X + eta * perturbations

    def make_init_tree(Xopts: Sequence[Any], etas: Any) -> Any:
        Xtrs = [mixin_layer(X=X, eta=eta) for eta, X in zip(etas, Xopts, strict=False)]
        return torch.cat(Xtrs, dim=-2)

    def mixin_tree(T: Any, alpha: Any) -> Any:
        return (1 - alpha) * T + alpha * unnormalize(torch.rand_like(T))

    n_repeat = math.ceil(raw_samples / batch_shape[0])
    alphas = torch.linspace(0, 0.75, n_repeat, **tkwargs)
    etas = torch.linspace(0.1, 1.0, len(Xopts), **tkwargs)
    return torch.cat(
        [mixin_tree(T=make_init_tree(Xopts=Xopts, etas=etas), alpha=alpha) for alpha in alphas],
        dim=0,
    )


def _warmstart_tree_python(
    full_optimizer: Any, bounds: Any, raw_samples: int, batch_shape: Sequence[int], sizes: list[int]
) -> Any:
    """The perturbed copies of the full tree with Python's generator."""
    import random

    data, shape = to_flat(full_optimizer, "full_optimizer")
    bdata, bshape = to_flat(bounds, "bounds")
    q_aug, d = shape[-2], shape[-1]
    nb = math.prod(batch_shape)
    lo = list(bdata[:d])
    hi = list(bdata[d : 2 * d])
    n_repeat = math.ceil(raw_samples / batch_shape[0])
    alphas = [0.75 * i / (n_repeat - 1) if n_repeat > 1 else 0.0 for i in range(n_repeat)]
    nl = len(sizes)
    etas = [0.1 + 0.9 * i / (nl - 1) if nl > 1 else 0.1 for i in range(nl)]
    out: list[float] = []
    for alpha in alphas:
        tree = []
        for b in range(nb):
            row = 0
            for level, size in enumerate(sizes):
                eta = etas[level]
                for r in range(row, row + size):
                    for c in range(d):
                        x = data[(b * q_aug + r) * d + c]
                        p = lo[c] + (hi[c] - lo[c]) * random.betavariate(1.0, 3.0)
                        tree.append((1 - eta) * x + eta * p)
                row += size
        for i, t in enumerate(tree):
            c = i % d
            out.append((1 - alpha) * t + alpha * (lo[c] + (hi[c] - lo[c]) * random.random()))
    return from_flat(out, (n_repeat * nb, q_aug, d), kind_of(full_optimizer))


def make_best_f(model: Any, X: Any) -> dict[str, Any]:
    r"""Extract the best observed training input from the model.

    botorch_rs returns the best training target in the original outcome
    space (the space of the model's posterior; shape ``batch_shape``, or
    ``batch_shape x m`` for ``m`` outputs). BoTorch returns the maximum of
    ``model.train_targets``, which are outcome-transformed (e.g.
    standardized, so that ``best_f`` is on the wrong scale for the value
    functions); both agree for models without an outcome transform. As the
    argument factory of ``qMultiStepLookahead``, every fantasy model gets the
    best value of its observed and fantasized data.
    """
    if getattr(model, "_tensor_backend", None) == "torch" and hasattr(model, "_train_Y"):
        best = model._train_Y.amax(dim=-2)
        return {"best_f": best.squeeze(-1) if model.num_outputs == 1 else best}
    rs = getattr(model, "_rs", None)
    if rs is not None and hasattr(rs, "train_targets"):
        data, shape = rs.train_targets()
        *batch, n, m = shape
        nb = math.prod(batch)
        best = [
            max(data[(b * n + i) * m + t] for i in range(n)) for b in range(nb) for t in range(m)
        ]
        out_shape = (*batch, m) if m > 1 else tuple(batch)
        return {"best_f": from_flat(best, out_shape, model._kind)}
    return {"best_f": model.train_targets.max(dim=-1).values}


__all__ = ["make_best_f", "qMultiStepLookahead", "warmstart_multistep"]
