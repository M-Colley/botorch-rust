r"""Max-value entropy search (``botorch.acquisition.max_value_entropy_search``),
computed in Rust with exact gradients.

The acquisition functions estimate the mutual information between the
observation at a candidate and the maximum of the objective, for samples of
the maximum drawn from a discretization of the design space (the candidate
set with the training inputs): ``qMaxValueEntropy`` (MES, Wang & Jegelka,
ICML 2017), its multi-fidelity version ``qMultiFidelityMaxValueEntropy``
(Takeno et al., ICML 2020), and the general-purpose information-based lower
bound GIBBON (``qLowerBoundMaxValueEntropy``, Moss et al., JMLR 2021) with
its multi-fidelity version.

Random numbers: the max values are sampled with botorch_rs's generator
(``seed``, a botorch_rs extension) and can be replaced through
``posterior_max_values``; the observation samples of MES and the fantasies of
pending points use the samplers ``sampler`` and ``fantasies_sampler``.

Differences to BoTorch main: the training inputs appended to the candidate
set are the model's raw inputs (BoTorch appends ``model.train_inputs``, which
a model with an input transform holds transformed in eval mode and
transforms again); GIBBON's repulsion uses the posterior cross-covariance of
the candidate with the pending points; the multi-fidelity GIBBON uses the
``X_pending`` passed to its constructor (BoTorch drops it).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, pending_arg
from botorch_rs.acquisition.cost_aware import CostAwareUtility
from botorch_rs.acquisition.knowledge_gradient import _expand_arg
from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _identity,
    _project_arg,
)
from botorch_rs.acquisition.objective import transform_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.normal import SobolQMCNormalSampler
from botorch_rs.sampling.pathwise.prior_samplers import seed_arg

#: BoTorch's lower bound of the variances (``CLAMP_LB``).
CLAMP_LB = 1.0e-8


def _candidates(candidate_set: Any, d: int, name: str = "candidate_set") -> Any:
    flat = to_flat(candidate_set, name)
    if len(flat[1]) != 2:
        raise UnsupportedError(
            f"`{name}` must be an `n x d` tensor (batched candidate sets are not supported)."
        )
    if flat[1][1] != d:
        raise ValueError(f"`{name}` must have {d} columns, got shape {tuple(flat[1])}.")
    return flat


class MaxValueBase(AcquisitionFunction):
    r"""Base class of the max-value entropy search acquisition functions: the
    max values are sampled at construction from the posterior over the
    candidate set, the training inputs and the pending points (Gumbel
    approximation or discrete Thompson sampling)."""

    _name = "MaxValueBase"

    def _setup(
        self,
        model: Any,
        candidate_set: Any,
        num_mv_samples: int,
        posterior_transform: Any,
        use_gumbel: bool,
        maximize: bool,
        X_pending: Any,
        train_inputs: Any,
        seed: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(model)
        if model.num_outputs > 1:
            raise UnsupportedError(f"Multi-output models are not supported by {self._name}.")
        rs_model = model._acquisition_model(None)
        d = int(rs_model.dim)
        self.candidate_set = candidate_set
        self.num_mv_samples = int(num_mv_samples)
        self.posterior_transform = posterior_transform
        self.use_gumbel = bool(use_gumbel)
        self.maximize = bool(maximize)
        self.weight = 1.0 if maximize else -1.0
        pending = pending_arg(X_pending, d)
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        self._rs, self._state = _rs.q_max_value_entropy(
            rs_model,
            _candidates(candidate_set, d),
            self._name,
            num_mv_samples=self.num_mv_samples,
            transform=transform_arg(posterior_transform, model.num_outputs),
            use_gumbel=self.use_gumbel,
            maximize=self.maximize,
            x_pending=pending,
            train_inputs=None
            if train_inputs is None
            else _candidates(train_inputs, d, "train_inputs"),
            seed=seed_arg(seed),
            **kwargs,
        )

    @property
    def posterior_max_values(self) -> Any:
        """The max-value samples (``num_mv_samples x 1``)."""
        values = self._state.posterior_max_values()
        return from_flat(values, (len(values), 1), self.model._kind)

    @posterior_max_values.setter
    def posterior_max_values(self, values: Any) -> None:
        data, _ = to_flat(values, "posterior_max_values")
        self._state.set_posterior_max_values([float(v) for v in data])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_mv_samples={self.num_mv_samples})"


class qMaxValueEntropy(MaxValueBase):
    r"""The acquisition function for Max-value Entropy Search.

    This acquisition function computes the mutual information of max values
    and a candidate point ``X``. See [Wang2017mves]_ for a detailed discussion.

    The model must be single-outcome. The batch case ``q > 1`` is supported
    through cyclic optimization and fantasies (``X_pending``).

    Example:
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> MES = qMaxValueEntropy(model, candidate_set)
        >>> mes = MES(test_X)

    botorch_rs extensions (keyword-only): ``seed`` of the max values,
    ``sampler`` (``num_y_samples``) and ``fantasies_sampler``
    (``num_fantasies``) of the observation samples and fantasies.
    """

    _name = "qMaxValueEntropy"

    def __init__(
        self,
        model: Any,
        candidate_set: Any,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        posterior_transform: Any = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Any = None,
        train_inputs: Any = None,
        *,
        seed: int | None = None,
        sampler: SobolQMCNormalSampler | None = None,
        fantasies_sampler: SobolQMCNormalSampler | None = None,
    ) -> None:
        self._init_mes(
            model,
            candidate_set,
            num_fantasies,
            num_mv_samples,
            num_y_samples,
            posterior_transform,
            use_gumbel,
            maximize,
            X_pending,
            train_inputs,
            seed,
            sampler,
            fantasies_sampler,
        )

    def _init_mes(
        self,
        model: Any,
        candidate_set: Any,
        num_fantasies: int,
        num_mv_samples: int,
        num_y_samples: int,
        posterior_transform: Any,
        use_gumbel: bool,
        maximize: bool,
        X_pending: Any,
        train_inputs: Any,
        seed: Any,
        sampler: Any,
        fantasies_sampler: Any,
        **kwargs: Any,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(num_y_samples,))
        if fantasies_sampler is None:
            fantasies_sampler = SobolQMCNormalSampler(sample_shape=(num_fantasies,))
        for s, name in ((sampler, "sampler"), (fantasies_sampler, "fantasies_sampler")):
            if not isinstance(s, SobolQMCNormalSampler):
                raise UnsupportedError(f"`{name}` must be a botorch_rs SobolQMCNormalSampler.")
        self.sampler = sampler
        self.fantasies_sampler = fantasies_sampler
        self.num_fantasies = int(fantasies_sampler.num_samples)
        self.num_y_samples = int(sampler.num_samples)
        self._setup(
            model,
            candidate_set,
            num_mv_samples,
            posterior_transform,
            use_gumbel,
            maximize,
            X_pending,
            train_inputs,
            seed,
            num_fantasies=self.num_fantasies,
            num_y_samples=self.num_y_samples,
            sampler=sampler._rs,
            fantasies_sampler=fantasies_sampler._rs,
            **kwargs,
        )


class qLowerBoundMaxValueEntropy(MaxValueBase):
    r"""The acquisition function for General-purpose Information-Based
    Bayesian Optimisation (GIBBON).

    This acquisition function provides a computationally cheap approximation
    of the mutual information between max values and a batch of candidate
    points ``X``. See [Moss2021gibbon]_ for a detailed discussion.

    The model must be single-outcome; ``q > 1`` is supported through cyclic
    optimization (the pending points repulse the candidate).

    Example:
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> qGIBBON = qLowerBoundMaxValueEntropy(model, candidate_set)
        >>> candidates, _ = optimize_acqf(qGIBBON, bounds, q=5)
    """

    _name = "qLowerBoundMaxValueEntropy"

    def __init__(
        self,
        model: Any,
        candidate_set: Any,
        num_mv_samples: int = 10,
        posterior_transform: Any = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Any = None,
        train_inputs: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._setup(
            model,
            candidate_set,
            num_mv_samples,
            posterior_transform,
            use_gumbel,
            maximize,
            X_pending,
            train_inputs,
            seed,
        )


def _cost_utility_arg(cost_aware_utility: Any) -> Any:
    if cost_aware_utility is None:
        return None
    if not isinstance(cost_aware_utility, CostAwareUtility):
        raise UnsupportedError(
            "`cost_aware_utility` must be a botorch_rs InverseCostWeightedUtility or "
            "GenericCostAwareUtility."
        )
    return cost_aware_utility._rs_utility


class qMultiFidelityMaxValueEntropy(qMaxValueEntropy):
    r"""Multi-fidelity max-value entropy.

    The acquisition function for multi-fidelity max-value entropy search with
    support for trace observations. See [Takeno2020mfmves]_ for a detailed
    discussion of the basic ideas on multi-fidelity MES (note that this
    implementation is somewhat different).

    The model must be single-outcome. The default cost-aware utility is an
    ``InverseCostWeightedUtility`` of ``AffineFidelityCostModel({-1: 1.0})``.
    ``project`` and ``expand`` are evaluated in Rust for ``functools.partial``
    objects of ``project_to_target_fidelity`` / ``expand_trace_observations``
    and otherwise as callables of torch tensors.

    Example:
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> MF_MES = qMultiFidelityMaxValueEntropy(model, candidate_set)
        >>> mf_mes = MF_MES(test_X)
    """

    _name = "qMultiFidelityMaxValueEntropy"

    def __init__(
        self,
        model: Any,
        candidate_set: Any,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        posterior_transform: Any = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Any = None,
        cost_aware_utility: CostAwareUtility | None = None,
        project: Callable[[Any], Any] = _identity,
        expand: Callable[[Any], Any] = _identity,
        *,
        seed: int | None = None,
        sampler: SobolQMCNormalSampler | None = None,
        fantasies_sampler: SobolQMCNormalSampler | None = None,
    ) -> None:
        self.cost_aware_utility = cost_aware_utility
        self.project = project
        self.expand = expand
        self._init_mes(
            model,
            candidate_set,
            num_fantasies,
            num_mv_samples,
            num_y_samples,
            posterior_transform,
            use_gumbel,
            maximize,
            X_pending,
            None,
            seed,
            sampler,
            fantasies_sampler,
            cost_aware_utility=_cost_utility_arg(cost_aware_utility),
            project=_project_arg(project),
            expand=_expand_arg(expand),
        )


class qMultiFidelityLowerBoundMaxValueEntropy(qMultiFidelityMaxValueEntropy):
    r"""Lower bound multi-fidelity max-value entropy search (GIBBON for
    multi-fidelity problems); trace observations are not supported.

    Example:
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> MF_qGIBBON = qMultiFidelityLowerBoundMaxValueEntropy(model, candidate_set)
        >>> mf_gibbon = MF_qGIBBON(test_X)
    """

    _name = "qMultiFidelityLowerBoundMaxValueEntropy"

    def __init__(
        self,
        model: Any,
        candidate_set: Any,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        X_pending: Any = None,
        posterior_transform: Any = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        cost_aware_utility: CostAwareUtility | None = None,
        project: Callable[[Any], Any] = _identity,
        expand: Callable[[Any], Any] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        if expand is not None:
            raise UnsupportedError(
                f"{self.__class__.__name__} does not support trace observations. "
            )
        super().__init__(
            model,
            candidate_set,
            num_fantasies=num_fantasies,
            num_mv_samples=num_mv_samples,
            num_y_samples=num_y_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            X_pending=X_pending,
            cost_aware_utility=cost_aware_utility,
            project=project,
            seed=seed,
        )
        self.expand: Callable[[Any], Any] | None = None  # type: ignore[assignment]


def _candidate_batches(candidate_set: Any, d: int) -> tuple[tuple[int, ...], list[Any]]:
    """The batch shape of a ``batch_shape x n x d`` candidate set and its
    ``n x d`` candidate sets."""
    data, shape = to_flat(candidate_set, "candidate_set")
    if len(shape) < 2:
        raise UnsupportedError("`candidate_set` must be a `batch_shape x n x d` tensor.")
    if shape[-1] != d:
        raise ValueError(f"`candidate_set` must have {d} columns, got shape {tuple(shape)}.")
    n = shape[-2]
    batch = tuple(shape[:-2])
    nb = math.prod(batch)
    per = n * d
    return batch, [(data[b * per : (b + 1) * per], (n, d)) for b in range(nb)]


def _sample_max_value_Gumbel(
    model: Any,
    candidate_set: Any,
    num_samples: int,
    posterior_transform: Any = None,
    maximize: bool = True,
    *,
    uniforms: Any = None,
    seed: int | None = None,
) -> Any:
    """Samples the max values by Gumbel approximation.

    Args:
        model: A fitted single-outcome model.
        candidate_set: A ``batch_shape x n x d`` Tensor including ``n`` candidate
            points per batch to discretize the design space.
        num_samples: Number of max value samples.
        posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A ``num_samples x batch_shape`` Tensor of posterior max value samples,
        where ``batch_shape`` is ``(1,)`` for an unbatched posterior.

    The Gumbel distribution of every candidate set is fitted to its
    quantiles; the same uniform numbers (BoTorch's ``torch.rand(num_samples,
    1)``) give the samples of every batch. botorch_rs extensions
    (keyword-only): the uniform numbers ``uniforms`` or their ``seed``."""
    import random

    if uniforms is None:
        rng = random.Random(seed)
        u = [rng.random() for _ in range(num_samples)]
    else:
        u = [float(v) for v in to_flat(uniforms, "uniforms")[0]]
        if len(u) != num_samples:
            raise ValueError(f"Expected {num_samples} uniform numbers, got {len(u)}.")
    rs_model = model._acquisition_model(None)
    batch, sets = _candidate_batches(candidate_set, int(rs_model.dim))
    transform = transform_arg(posterior_transform, model.num_outputs)
    columns = [
        _rs.sample_max_value_gumbel(rs_model, cs, u, bool(maximize), transform) for cs in sets
    ]
    values = [columns[b][i] for i in range(num_samples) for b in range(len(sets))]
    return from_flat(values, (num_samples, *(batch or (1,))), kind_of(candidate_set))


def _sample_max_value_Thompson(
    model: Any,
    candidate_set: Any,
    num_samples: int,
    posterior_transform: Any = None,
    maximize: bool = True,
    *,
    base_samples: Any = None,
    seed: int | None = None,
) -> Any:
    """Samples the max values by discrete Thompson sampling.

    Args:
        model: A fitted single-outcome model.
        candidate_set: A ``batch_shape x n x d`` Tensor including ``n`` candidate
            points per batch to discretize the design space.
        num_samples: Number of max value samples.
        posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A ``num_samples x batch_shape`` Tensor of posterior max value samples,
        where ``batch_shape`` is ``(1,)`` for an unbatched posterior.

    botorch_rs extensions (keyword-only): the standard normal
    ``base_samples`` (``num_samples x batch_shape x n``, independent between
    the batches as BoTorch's draws) or their ``seed``."""
    import random

    rs_model = model._acquisition_model(None)
    batch, sets = _candidate_batches(candidate_set, int(rs_model.dim))
    nb = len(sets)
    n = sets[0][1][0] if sets else 0
    total = num_samples * nb * n
    if base_samples is None:
        rng = random.Random(seed)
        z = [rng.gauss(0.0, 1.0) for _ in range(total)]
    else:
        z = list(to_flat(base_samples, "base_samples")[0])
        if len(z) != total:
            raise ValueError(
                f"Expected base samples of shape {num_samples} x {tuple(batch)} x {n}."
            )
    transform = transform_arg(posterior_transform, model.num_outputs)
    columns = []
    for b, cs in enumerate(sets):
        zb = [z[(i * nb + b) * n + j] for i in range(num_samples) for j in range(n)]
        columns.append(
            _rs.sample_max_value_thompson(
                rs_model, cs, (zb, (num_samples, n)), bool(maximize), transform
            )
        )
    values = [columns[b][i] for i in range(num_samples) for b in range(nb)]
    return from_flat(values, (num_samples, *(batch or (1,))), kind_of(candidate_set))


__all__ = [
    "CLAMP_LB",
    "MaxValueBase",
    "qLowerBoundMaxValueEntropy",
    "qMaxValueEntropy",
    "qMultiFidelityLowerBoundMaxValueEntropy",
    "qMultiFidelityMaxValueEntropy",
]
