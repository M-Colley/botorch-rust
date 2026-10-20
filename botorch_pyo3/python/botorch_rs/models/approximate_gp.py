r"""
Approximate (variational) GP models (``botorch.models.approximate_gp``).

References

.. [burt2020svgp]
    David R. Burt and Carl Edward Rasmussen and Mark van der Wilk,
    Convergence of Sparse Variational Inference in Gaussian Process Regression,
    Journal of Machine Learning Research, 2020,
    http://jmlr.org/papers/v21/19-1015.html.

.. [hensman2013svgp]
    James Hensman and Nicolo Fusi and Neil D. Lawrence, Gaussian Processes
    for Big Data, Proceedings of the 29th Conference on Uncertainty in
    Artificial Intelligence, 2013, https://arxiv.org/abs/1309.6835.

.. [moss2023ipa]
    Henry B. Moss and Sebastian W. Ober and Victor Picheny,
    Inducing Point Allocation for Sparse Gaussian Processes
    in High-Throughput Bayesian Optimization, Proceedings of
    the 25th International Conference on Artificial Intelligence
    and Statistics, 2023, https://arxiv.org/pdf/2301.10123.pdf.

The variational strategy (GPyTorch's whitened ``VariationalStrategy`` with a
``CholeskyVariationalDistribution``), the variational ELBO with its exact
gradient, posteriors with input derivatives and the fits run in Rust. The
random initialization of the variational mean at the model's first
evaluation draws from torch's global generator (as GPyTorch does) when torch
is installed.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Iterator
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    LIST,
    ArrayKind,
    broadcast_shapes,
    broadcast_to,
    from_flat,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import (
    BotorchTensorDimensionError,
    InputDataError,
    UnsupportedError,
)
from botorch_rs.exceptions.warnings import UserInputWarning
from botorch_rs.kernels import Kernel, _batch_matrix, _indices
from botorch_rs.likelihoods import GaussianLikelihood, Likelihood, MultitaskGaussianLikelihood
from botorch_rs.means import ConstantMean, Mean, ZeroMean
from botorch_rs.models.gpytorch import GPyTorchModel
from botorch_rs.models.transforms.input import InputTransform, chain_spec
from botorch_rs.models.transforms.outcome import OutcomeTransform, Standardize
from botorch_rs.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch_rs.models.utils.inducing_point_allocators import (
    GreedyVarianceReduction,
    InducingPointAllocator,
)
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior
from botorch_rs.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
    _VariationalDistribution,
)

TRANSFORM_WARNING = (
    "Using an {ttype} transform with `SingleTaskVariationalGP`. If this "
    "model is trained in minibatches, a {ttype} transform with learnable "
    "parameters would update its parameters for each minibatch, which is "
    "undesirable. If you do intend to train in minibatches, we recommend "
    "you not use a {ttype} transform and instead pre-transform your whole "
    "data set before fitting the model."
)


def _squeeze_last(pair: tuple[Any, tuple[int, ...]]) -> tuple[Any, tuple[int, ...]]:
    data, shape = pair
    if shape and shape[-1] == 1:
        return data, tuple(shape[:-1])
    return data, tuple(shape)


class _ApproximateOutput:
    r"""The approximate posterior ``q(f)`` at inputs ``X`` (GPyTorch's
    output of ``model.model(X)``, a ``MultivariateNormal``, or a
    ``MultitaskMultivariateNormal`` for several outputs): ``mean``,
    ``variance``, ``covariance_matrix`` (in the transformed outcome space)."""

    _is_prior = False

    def __init__(self, owner: SingleTaskVariationalGP, X: Any) -> None:
        self._owner = owner
        state = owner._sync_torch_parameters()
        self._parameter_leaves = () if state is None else tuple(p for _, p in state.entries)
        self._native = owner._rs.copy()
        self._kind = kind_of(X)
        flat = to_flat(X, "X")
        if len(flat[1]) == 1:
            flat = (flat[0], (flat[1][0], 1))
        if len(flat[1]) != 2:
            raise UnsupportedError(
                "botorch_rs evaluates the variational GP at unbatched `n x d` inputs."
            )
        self._x = (list(flat[0]), tuple(flat[1]))
        self._cache: tuple[Any, Any, Any] | None = None
        # GPyTorch's training mode with more points than inducing points
        # updates only the diagonal of the prior covariance (the marginal
        # variances, which is all the ELBO needs); its multitask strategy
        # computes the full covariance.
        self._diag_only = (
            owner.model.training
            and owner.num_outputs == 1
            and self._x[1][0] > int(self._native.num_inducing)
        )

    def _parts(self) -> tuple[Any, Any, Any]:
        if self._cache is None:
            self._cache = self._native.latent(self._x)
        return self._cache

    @property
    def event_shape(self) -> tuple[int, ...]:
        m = self._owner.num_outputs
        n = self._x[1][0]
        return (n, m) if m > 1 else (n,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def mean(self) -> Any:
        (data, shape), _, _ = self._parts()
        if self._owner.num_outputs == 1:
            return from_flat(data, (shape[0],), self._kind)
        return from_flat(data, tuple(shape), self._kind)

    @property
    def loc(self) -> Any:
        """The scalar-event location, flattened across interleaved tasks."""
        (data, shape), _, _ = self._parts()
        return from_flat(data, (math.prod(shape),), self._kind)

    @property
    def variance(self) -> Any:
        """The marginal variances (clamped at 1e-10, GPyTorch's lazy
        ``diag=True`` kernel values)."""
        _, _, (var, shape) = self._parts()
        out = [max(v, 1e-10) for v in var]
        return from_flat(out, (shape[0],) if shape[1] == 1 else tuple(shape), self._kind)

    @property
    def covariance_matrix(self) -> Any:
        """The covariance (``n x n``, or ``nm x nm`` interleaved for several
        outputs)."""
        _, (cov, shape), _ = self._parts()
        m, q = shape[0], shape[1]
        if self._diag_only:
            # Off the diagonal: the prior covariance K(X, X).
            cov = list(cov)
            for t in range(m):
                k, _ = self._native.kernel_matrix(self._x, t)
                for i in range(q):
                    for j in range(q):
                        if i != j:
                            cov[t * q * q + i * q + j] = k[i * q + j]
        if m == 1:
            return from_flat(cov, (q, q), self._kind)
        out = [0.0] * (q * m * q * m)
        for t in range(m):
            for i in range(q):
                for j in range(q):
                    out[(i * m + t) * q * m + j * m + t] = cov[t * q * q + i * q + j]
        return from_flat(out, (q * m, q * m), self._kind)


class _VariationalPriorOutput(_ApproximateOutput):
    """An immutable prior law from the native mean and kernel, without q(u).

    ``forward`` retains the independent output batch; the variational
    strategy's ``prior=True`` call turns that batch into interleaved tasks.
    """

    _is_prior = True

    def __init__(self, owner: SingleTaskVariationalGP, X: Any, multitask: bool) -> None:
        self._owner = owner
        owner._sync_torch_parameters()
        self._native = owner._rs.copy()
        self._kind = kind_of(X)
        data, shape = to_flat(X, "X")
        if len(shape) == 1:
            shape = (shape[0], 1)
        if len(shape) < 2:
            raise BotorchTensorDimensionError("Expected prior inputs with shape `... x n x d`.")
        self._x = (list(data), tuple(shape))
        self._num_outputs = owner.num_outputs
        self._multitask = multitask and self._num_outputs > 1
        batch = broadcast_shapes(
            shape[:-2],
            owner.model._aug_batch_shape,
            owner.model.mean_module.batch_shape,
        )
        if batch is None:
            raise RuntimeError("The prior input, mean and kernel batch shapes do not broadcast.")
        self._batch_shape = batch[:-1] if self._multitask else batch
        # MVN.mean/loc retain the mean module's own broadcast shape, even
        # when a ZeroMean lacks an output batch present in the covariance.
        mean_batch = broadcast_shapes(shape[:-2], owner.model.mean_module.batch_shape)
        assert mean_batch is not None  # Already checked by the joint broadcast.
        self._mean_shape = (
            (*self._batch_shape, *self.event_shape) if self._multitask else (*mean_batch, shape[-2])
        )
        self._prior_cache: tuple[list[float], list[float], list[float]] | None = None

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def event_shape(self) -> tuple[int, ...]:
        n = self._x[1][-2]
        return (n, self._num_outputs) if self._multitask else (n,)

    def _prior_parts(self) -> tuple[list[float], list[float], list[float]]:
        if self._prior_cache is None:
            means: list[float] = []
            covariances: list[float] = []
            variances: list[float] = []
            constants = self._native.constant
            n, m = self._x[1][-2], self._num_outputs
            if not self._multitask:
                for idx in _indices(self._mean_shape[:-1]):
                    t = idx[-1] if m > 1 and constants else 0
                    means.extend([constants[t] if constants else 0.0] * n)
            for idx in _indices(self.batch_shape):
                if self._multitask:
                    mean, var = [0.0] * (n * m), [0.0] * (n * m)
                    cov = [0.0] * (n * m * n * m)
                    for t in range(m):
                        x = _batch_matrix(self._x, (*idx, t))
                        k, _ = self._native.kernel_matrix(x, t)
                        for i in range(n):
                            mean[i * m + t] = constants[t] if constants else 0.0
                            var[i * m + t] = max(k[i * n + i], 1e-10)
                            for j in range(n):
                                cov[(i * m + t) * n * m + j * m + t] = k[i * n + j]
                    means.extend(mean)
                    variances.extend(var)
                    covariances.extend(cov)
                else:
                    t = idx[-1] if m > 1 else 0
                    k, _ = self._native.kernel_matrix(_batch_matrix(self._x, idx), t)
                    variances.extend(max(k[i * n + i], 1e-10) for i in range(n))
                    covariances.extend(k)
            self._prior_cache = means, covariances, variances
        return self._prior_cache

    @property
    def mean(self) -> Any:
        data, _, _ = self._prior_parts()
        return from_flat(data, self._mean_shape, self._kind)

    @property
    def loc(self) -> Any:
        data, _, _ = self._prior_parts()
        shape = (
            (*self.batch_shape, math.prod(self.event_shape))
            if self._multitask
            else self._mean_shape
        )
        return from_flat(data, shape, self._kind)

    @property
    def variance(self) -> Any:
        _, _, data = self._prior_parts()
        return from_flat(data, (*self.batch_shape, *self.event_shape), self._kind)

    @property
    def covariance_matrix(self) -> Any:
        _, data, _ = self._prior_parts()
        n = math.prod(self.event_shape)
        return from_flat(data, (*self.batch_shape, n, n), self._kind)


class _SingleTaskVariationalGP:
    """
    Base class wrapper for a stochastic variational Gaussian Process (SVGP)
    model [hensman2013svgp]_.

    Uses by default pivoted Cholesky initialization for allocating inducing points,
    however, custom inducing point allocators can be provided.
    """

    _builds_variational_strategy = True

    def __init__(
        self,
        train_X: Any,
        train_Y: Any = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type = VariationalStrategy,
        inducing_points: Any = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
    ) -> None:
        r"""
        Args:
            train_X: Training inputs (due to the ability of the SVGP to sub-sample
                this does not have to be all of the training inputs).
            train_Y: Not used.
            num_outputs: Number of output responses per input.
            covar_module: Kernel function. If omitted, uses an ``RBFKernel``.
            mean_module: Mean of GP model. If omitted, uses a ``ConstantMean``.
            variational_distribution: Type of variational distribution to use
                (default: CholeskyVariationalDistribution), the properties of the
                variational distribution will encourage scalability or ease of
                optimization.
            variational_strategy: Type of variational strategy to use (default:
                VariationalStrategy). The default setting uses "whitening" of the
                variational distribution to make training easier.
            inducing_points: The number or specific locations of the inducing points.
            inducing_point_allocator: The ``InducingPointAllocator`` used to
                initialize the inducing point locations. If omitted,
                uses ``GreedyVarianceReduction``.
        """
        num_outputs = int(num_outputs)
        x = to_flat(train_X, "train_X")
        if len(x[1]) < 2:
            raise BotorchTensorDimensionError(
                f"Expected train_X to be a `batch_shape x n x d` tensor, got shape {x[1]}."
            )
        input_batch_shape = tuple(x[1][:-2])
        if input_batch_shape:
            raise UnsupportedError(
                "botorch_rs's SingleTaskVariationalGP supports unbatched training inputs "
                "(`n x d`) only."
            )
        d = int(x[1][-1])
        aug: tuple[int, ...] = (num_outputs,) if num_outputs > 1 else ()
        self._aug_batch_shape = aug
        from botorch_rs.models.gp_regression import _check_module_batch_shape

        if covar_module is None:
            covar_module = get_covar_module_with_dim_scaled_prior(ard_num_dims=d, batch_shape=aug)
        elif not isinstance(covar_module, Kernel):
            raise UnsupportedError(
                "`covar_module` must be a botorch_rs kernel (from botorch_rs.kernels, "
                "botorch_rs.models.kernels or botorch_rs.models.utils.gpytorch_modules)."
            )
        elif covar_module._model is not None:
            raise UnsupportedError(
                "This kernel is already part of another model; create a new one for each model."
            )
        else:
            _check_module_batch_shape(covar_module, aug, "covar_module")
        if inducing_point_allocator is None:
            inducing_point_allocator = GreedyVarianceReduction()
        z: Any
        if inducing_points is None or isinstance(inducing_points, int):
            num = int(0.25 * x[1][-2]) if inducing_points is None else int(inducing_points)
            z = inducing_point_allocator.allocate_inducing_points(
                inputs=train_X,
                covar_module=covar_module,
                num_inducing=num,
                input_batch_shape=input_batch_shape,
            )
        else:
            z = inducing_points
        zf = to_flat(z, "inducing_points")
        if len(zf[1]) == 1:
            zf = (zf[0], (zf[1][0], 1))
        if len(zf[1]) != 2 or zf[1][-1] != d:
            raise BotorchTensorDimensionError(
                f"Expected `M x {d}` inducing points, got shape {tuple(zf[1])}."
            )
        num_inducing = int(zf[1][0])
        if variational_distribution is None:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing, batch_shape=aug
            )
        elif not isinstance(variational_distribution, CholeskyVariationalDistribution):
            raise UnsupportedError(
                "botorch_rs supports the CholeskyVariationalDistribution of botorch_rs.variational."
            )
        elif variational_distribution._owner is not None:
            raise UnsupportedError(
                "This variational distribution is already part of another model; create a "
                "new one for each model."
            )
        if variational_distribution.num_inducing_points != num_inducing:
            raise RuntimeError(
                f"The variational distribution has {variational_distribution.num_inducing_points}"
                f" inducing points, but {num_inducing} inducing points were given."
            )
        if variational_distribution.batch_shape != aug:
            raise BotorchTensorDimensionError(
                f"The variational distribution has batch_shape="
                f"{variational_distribution.batch_shape}, expected {aug}."
            )
        if variational_strategy is not VariationalStrategy:
            raise UnsupportedError(
                "botorch_rs supports the (whitened) VariationalStrategy of botorch_rs.variational."
            )
        strategy = VariationalStrategy(
            self,
            inducing_points=from_flat(zf[0], tuple(zf[1]), LIST),
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points,
        )
        self.variational_strategy: Any = (
            IndependentMultitaskVariationalStrategy(
                base_variational_strategy=strategy, num_tasks=num_outputs, task_dim=-1
            )
            if num_outputs > 1
            else strategy
        )
        if mean_module is None:
            mean_module = ConstantMean(batch_shape=aug)
        elif not isinstance(mean_module, (ConstantMean, ZeroMean)):
            raise UnsupportedError("Only a ConstantMean or ZeroMean `mean_module` is supported.")
        elif isinstance(mean_module, ConstantMean):
            _check_module_batch_shape(mean_module, aug, "mean_module")
        elif tuple(mean_module.batch_shape) not in ((), aug):
            raise UnsupportedError(
                f"ZeroMean batch_shape must be () or the independent-output shape {aug}; "
                "additional variational model batch dimensions are not supported."
            )
        self.mean_module: Mean = mean_module
        self.covar_module: Kernel = covar_module
        self._num_outputs = num_outputs
        self._train_X = train_X
        self._inducing = (list(zf[0]), tuple(zf[1]))
        self._learn_inducing_points = bool(learn_inducing_points)
        self._outer: Any = None
        self.training = True

    # -- binding -----------------------------------------------------------

    @property
    def _strategy(self) -> VariationalStrategy:
        vs = self.variational_strategy
        return vs.base_variational_strategy if self._num_outputs > 1 else vs

    @property
    def _rs(self) -> Any:
        if self._outer is None:
            raise RuntimeError("The variational GP is not part of a model.")
        return self._outer._rs

    @property
    def _kind(self) -> ArrayKind:
        return self._outer._kind if self._outer is not None else LIST

    # -- GPyTorch's module API ----------------------------------------------

    @property
    def train_inputs(self) -> tuple[Any]:
        """``[transformed train_X]`` (set by ``SingleTaskVariationalGP``)."""
        if self._outer is None or not self._outer._has_train_inputs:
            raise AttributeError(
                "'_SingleTaskVariationalGP' object has no attribute 'train_inputs'"
            )
        data, shape = self._rs.train_inputs()
        return (from_flat(data, tuple(shape), self._kind),)

    @property
    def train_targets(self) -> Any:
        """The (outcome-transformed) training targets, ``train_Y.squeeze(-1)``."""
        targets = self._rs.train_targets() if self._outer is not None else None
        if targets is None or not self._outer._has_train_inputs:
            raise AttributeError(
                "'_SingleTaskVariationalGP' object has no attribute 'train_targets'"
            )
        data, shape = targets
        return from_flat(*_squeeze_last((data, tuple(shape))), self._kind)

    def __call__(self, X: Any, prior: bool = False) -> _ApproximateOutput:
        """``q(f)`` at the (already transformed) inputs ``X``; the variational
        parameters are initialized at the first call (GPyTorch). ``prior=True``
        bypasses q(u) and combines independent outputs into multitask events."""
        if prior:
            return _VariationalPriorOutput(self._outer, X, multitask=True)
        self._outer._ensure_initialized()
        return _ApproximateOutput(self._outer, X)

    def forward(self, X: Any) -> _ApproximateOutput:
        """The GP prior at already transformed ``X``, with independent outputs
        in the batch dimensions, before the variational strategy is applied."""
        return _VariationalPriorOutput(self._outer, X, multitask=False)

    def train(self, mode: bool = True) -> _SingleTaskVariationalGP:
        self.training = mode
        return self

    def eval(self) -> _SingleTaskVariationalGP:
        return self.train(False)

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of the model's parameters (GPyTorch's names,
        without the likelihood)."""
        for name, value in self._outer._named_raw():
            if name.startswith("model."):
                yield name[len("model.") :], value

    def parameters(self) -> Iterator[Any]:
        for _, value in self.named_parameters():
            yield value

    def __repr__(self) -> str:
        return f"_SingleTaskVariationalGP(num_outputs={self._num_outputs})"


class ApproximateGPyTorchModel(GPyTorchModel):
    r"""
    Botorch wrapper class for various (variational) approximate GP models in
    GPyTorch.

    This can either include stochastic variational GPs (SVGPs) or
    variational implementations of weight space approximate GPs.
    """

    _has_train_inputs = False

    def __init__(
        self,
        model: _SingleTaskVariationalGP | None = None,
        likelihood: Likelihood | None = None,
        num_outputs: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            model: Instance of gpytorch.approximate GP models. If omitted,
                constructs a ``_SingleTaskVariationalGP``.
            likelihood: Instance of a GPyTorch likelihood. If omitted, uses a
                either a ``GaussianLikelihood`` (if ``num_outputs=1``) or a
                ``MultitaskGaussianLikelihood``(if ``num_outputs>1``).
            num_outputs: Number of outputs expected for the GP model.
            args: Optional positional arguments passed to the
                ``_SingleTaskVariationalGP`` constructor if no model is provided.
            kwargs: Optional keyword arguments passed to the
                ``_SingleTaskVariationalGP`` constructor if no model is provided.
        """
        if model is None:
            options = dict(kwargs, num_outputs=num_outputs)
            model = _SingleTaskVariationalGP(*args, **options)
        elif not isinstance(model, _SingleTaskVariationalGP):
            raise UnsupportedError(
                "botorch_rs's ApproximateGPyTorchModel wraps a botorch_rs _SingleTaskVariationalGP."
            )
        if likelihood is None:
            likelihood = (
                GaussianLikelihood()
                if num_outputs == 1
                else MultitaskGaussianLikelihood(num_tasks=num_outputs)
            )
        self.model = model
        self.likelihood: Likelihood = likelihood
        self._desired_num_outputs = int(num_outputs)
        if not isinstance(self, SingleTaskVariationalGP):
            self._kind = kind_of(model._train_X)
            self._build(model._train_X, None, int(to_flat(model._train_X, "X")[1][-1]))

    # -- construction --------------------------------------------------------

    def _build(
        self,
        transformed_X: Any,
        transformed_Y: Any,
        dim: int,
        input_steps: list[Any] | None = None,
        input_prefixes: list[str] | None = None,
        outcome: Standardize | None = None,
        input_training: list[bool] | None = None,
        raw_train_X: Any = None,
    ) -> None:
        inner = self.model
        m = inner._num_outputs
        if m != self._desired_num_outputs:
            raise BotorchTensorDimensionError(
                f"The model has {inner._num_outputs} outputs, expected {self._desired_num_outputs}."
            )
        lik = self.likelihood
        if m == 1:
            if type(lik) is not GaussianLikelihood:
                raise UnsupportedError(
                    "botorch_rs's SingleTaskVariationalGP supports a GaussianLikelihood with one "
                    "output (non-Gaussian likelihoods such as BetaLikelihood are not ported)."
                )
            if tuple(lik.batch_shape):
                raise UnsupportedError("A likelihood with a batch shape is not supported.")
            lik_kind = "gaussian"
        else:
            if not isinstance(lik, MultitaskGaussianLikelihood) or lik.num_tasks != m:
                raise UnsupportedError(
                    "botorch_rs's SingleTaskVariationalGP with several outputs supports a "
                    f"MultitaskGaussianLikelihood(num_tasks={m}) (GPyTorch's variational ELBO "
                    "fails with a GaussianLikelihood there)."
                )
            if lik.rank != 0:
                raise UnsupportedError("A MultitaskGaussianLikelihood with rank > 0.")
            lik_kind = "multitask"
        if lik._model is not None:
            raise UnsupportedError(
                "This likelihood is already part of another model; create a new one for each model."
            )
        x = to_flat(transformed_X, "train_X")
        y = None
        if transformed_Y is not None:
            ydata, yshape = to_flat(transformed_Y, "train_Y")
            y = (ydata, tuple(yshape))
        mean = inner.mean_module
        mean_kind = "zero" if isinstance(mean, ZeroMean) else "constant"
        mean_prior = getattr(mean, "mean_prior", None)
        mean_constraint = getattr(mean, "raw_constant_constraint", None)
        vd = inner._strategy._variational_distribution
        covar = inner.covar_module
        aug = inner._aug_batch_shape
        self._rs = _rs.SingleTaskVariationalGP(
            x,
            y,
            inner._inducing,
            m,
            dim,
            covar._node(),
            bool(aug),
            lik._spec,
            likelihood=lik_kind,
            mean=mean_kind,
            mean_prior=None if mean_prior is None else mean_prior._rs,
            mean_constraint=None if mean_constraint is None else mean_constraint._rs,
            mean_batched=bool(aug) and mean_kind == "constant",
            learn_inducing_points=inner._learn_inducing_points,
            input_chain=input_steps,
            input_chain_training=input_training,
            input_chain_prefixes=input_prefixes,
            raw_train_x=None if raw_train_X is None else to_flat(raw_train_X, "train_X"),
            outcome_means=None if outcome is None else list(outcome._rs.means[0]),
            outcome_stdvs=None if outcome is None else list(outcome._rs.stdvs[0]),
            outcome_min_stdv=1e-8 if outcome is None else outcome.min_stdv,
        )
        inner._outer = self
        # Initial values of the modules (e.g. set before the model was built).
        raw = list(self._rs.raw_parameters())
        offsets = self._offsets()
        local_mean = getattr(vd, "_local_mean", None)
        local_chol = getattr(vd, "_local_chol", None)
        for key, values in (
            ("variational_mean", local_mean),
            ("chol_variational_covar", local_chol),
        ):
            name = next(n for n in offsets if n.endswith(key))
            off, size = offsets[name]
            if values is not None:
                raw[off : off + size] = values
        self._rs.set_raw_parameters(raw)
        # Bind the modules to the model's parameters.
        lik._bind(self)
        covar._bind(self, "covar_module.")
        if isinstance(mean, ConstantMean):
            mean._bind(self)
        prefix = "variational_strategy."
        if m > 1:
            prefix += "base_variational_strategy."
        inner._strategy._bind(self, prefix)
        self._param_shapes = self._compute_shapes()
        self._torch_parameters_ready = True

    def _compute_shapes(self) -> dict[str, tuple[int, ...]]:
        inner = self.model
        aug = inner._aug_batch_shape
        m = inner._num_outputs
        mi = int(self._rs.num_inducing)
        dt = len(self._rs.inducing_points()[0]) // max(mi, 1) if mi else 0
        kernel_shapes = {
            f"model.covar_module.{name}": (*aug, *tuple(shape))
            for name, shape in inner.covar_module._node().param_shapes("")
        }
        shapes: dict[str, tuple[int, ...]] = {}
        for name, _ in self._rs.parameter_names():
            if name in kernel_shapes:
                shapes[name] = kernel_shapes[name]
            elif name.endswith("raw_task_noises"):
                shapes[name] = (m,)
            elif name.endswith("raw_noise"):
                shapes[name] = (1,)
            elif name.endswith("inducing_points"):
                shapes[name] = (mi, dt)
            elif name.endswith("variational_mean"):
                shapes[name] = (*aug, mi)
            elif name.endswith("chol_variational_covar"):
                shapes[name] = (*aug, mi, mi)
            elif name.endswith("raw_constant"):
                shapes[name] = aug
            else:  # pragma: no cover - all names are covered above
                raise RuntimeError(f"Unknown parameter {name}.")
        return shapes

    def _offsets(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        off = 0
        for name, size in self._rs.parameter_names():
            out[name] = (off, size)
            off += size
        return out

    # -- parameters ------------------------------------------------------------

    def _sync_torch_parameters(self) -> Any:
        """Synchronize persistent Torch leaves with the native variational GP."""
        if (
            self._kind.lib != "torch"
            or getattr(self._kind.device, "type", "cpu") != "cpu"
            or not self.__dict__.get("_torch_parameters_ready", False)
        ):
            return None
        state = self.__dict__.get("_torch_parameter_state")
        if state is None:
            from botorch_rs.models._torch_parameters import TorchParameterState

            shapes = [(name, self._param_shapes[name]) for name, _ in self._rs.parameter_names()]
            state = self._torch_parameter_state = TorchParameterState(self, shapes)
        else:
            state.sync(self)
        return state

    def _param_index(self, suffix: str) -> tuple[str, int, int]:
        offsets = self._offsets()
        if suffix in offsets:
            return suffix, *offsets[suffix]
        for name, (off, size) in offsets.items():
            if name.endswith(suffix):
                return name, off, size
        raise AttributeError(f"The model has no parameter {suffix!r}.")

    def _raw_param(self, suffix: str) -> list[float]:
        self._sync_torch_parameters()
        _, off, size = self._param_index(suffix)
        return list(self._rs.raw_parameters()[off : off + size])

    def _raw_param_array(self, suffix: str) -> Any:
        state = self._sync_torch_parameters()
        name, _, _ = self._param_index(suffix)
        if state is not None:
            return state.parameter(name)
        return from_flat(self._raw_param(suffix), self._param_shapes[name], self._kind)

    def _raw_param_shape(self, suffix: str) -> tuple[int, ...]:
        name, _, _ = self._param_index(suffix)
        return self._param_shapes[name]

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        self._sync_torch_parameters()
        name, off, size = self._param_index(suffix)
        data, shape = to_flat(value, suffix)
        target = self._param_shapes[name]
        try:
            new = broadcast_to(data, tuple(shape), target)
        except ValueError:
            if len(data) != size:
                raise BotorchTensorDimensionError(
                    f"{name} has shape {target}, got a value of shape {tuple(shape)}."
                ) from None
            new = [float(v) for v in data]
        values = list(self._rs.raw_parameters())
        values[off : off + size] = new
        self._rs.set_raw_parameters(values)
        self._sync_torch_parameters()

    def _named_raw(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` in the ELBO's order (likelihood first)."""
        state = self._sync_torch_parameters()
        if state is not None:
            yield from state.entries
            return
        values = self._rs.raw_parameters()
        off = 0
        for name, size in self._rs.parameter_names():
            yield name, from_flat(values[off : off + size], self._param_shapes[name], self._kind)
            off += size

    def _param_entries(self) -> list[tuple[str, Any, int, int]]:
        """``(name, raw value, offset, size)`` in ``named_parameters()`` order."""
        offsets = self._offsets()
        return [(n, v, *offsets[n]) for n, v in self.named_parameters()]

    def named_parameters(self) -> Iterator[tuple[str, Any]]:
        """``(name, raw value)`` of every parameter (BoTorch's names and order:
        the model's, then the likelihood's)."""
        items = list(self._named_raw())
        for name, value in items:
            if name.startswith("model."):
                yield name, value
        for name, value in items:
            if name.startswith("likelihood."):
                yield name, value

    def parameters(self) -> Iterator[Any]:
        for _, value in self.named_parameters():
            yield value

    def initialize(self, **kwargs: Any) -> ApproximateGPyTorchModel:
        """Sets parameters by (dotted) name, like GPyTorch's ``Module.initialize``."""
        for name, value in kwargs.items():
            obj: Any = self
            *path, attr = name.split(".")
            for part in path:
                obj = getattr(obj, part)
            if not isinstance(getattr(type(obj), attr, None), property):
                raise AttributeError(f"Unknown parameter {name!r}.")
            setattr(obj, attr, value)
        return self

    def _ensure_initialized(self) -> None:
        self._sync_torch_parameters()
        self.model._strategy._ensure_initialized()
        self._sync_torch_parameters()

    def _kl_divergence(self, summed: bool) -> Any:
        """``KL(q(u) ‖ N(0, I))`` of every output (summed over the outputs)."""
        mi = int(self._rs.num_inducing)
        m = self.model._num_outputs
        mean = self._raw_param("variational_mean")
        chol = self._raw_param("chol_variational_covar")
        out = []
        for t in range(m):
            kl = -float(mi)
            for i in range(mi):
                lii = chol[t * mi * mi + i * mi + i]
                kl -= math.log(lii * lii)
                for j in range(i + 1):
                    v = chol[t * mi * mi + i * mi + j]
                    kl += v * v
            kl += sum(v * v for v in mean[t * mi : (t + 1) * mi])
            out.append(0.5 * kl)
        if summed or m == 1:
            return from_flat([sum(out)], (), self._kind)
        return from_flat(out, (m,), self._kind)

    # -- Model API -------------------------------------------------------------

    @property
    def num_outputs(self) -> int:
        return self._desired_num_outputs

    def eval(self) -> ApproximateGPyTorchModel:
        r"""Puts the model in ``eval`` mode."""
        return self.train(False)

    def train(self, mode: bool = True) -> ApproximateGPyTorchModel:
        r"""Put the model in ``train`` mode.

        Args:
            mode: A boolean denoting whether to put in ``train`` or ``eval`` mode.
                If ``False``, model is put in ``eval`` mode.
        """
        self.training = mode
        self.model.train(mode)
        for name in ("input_transform", "outcome_transform"):
            tf = getattr(self, name, None)
            if tf is not None:
                tf.train(mode)
        return self

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""The batch shape of the model (unbatched models only in botorch_rs)."""
        return ()

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """The posterior at ``X`` (``batch_shape x q x d``), in the original
        outcome space; ``observation_noise=True`` adds the likelihood noise."""
        from botorch_rs.acquisition.objective import transform_arg

        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output indices."
            )
        if not isinstance(observation_noise, bool):
            raise UnsupportedError("`observation_noise` must be a bool in botorch_rs.")
        self.eval()
        self._ensure_initialized()
        kind = kind_of(X)
        flat = to_flat(X, "X")
        if len(flat[1]) < 2:
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d`, got shape {flat[1]}."
            )
        if flat[1][-1] != self._rs.dim:
            raise BotorchTensorDimensionError(
                f"Expected X with {self._rs.dim} columns (d), got {flat[1][-1]}."
            )
        m = self.num_outputs
        return GPyTorchPosterior(
            self._rs.model_handle(),
            flat,
            kind,
            observation_noise,
            num_outputs=m,
            model_batch_shape=(),
            transform=transform_arg(posterior_transform, m),
            interleaved=True,
            query_inputs=X,
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> Any:
        """The model handle that acquisition functions are built on."""
        from botorch_rs.acquisition.objective import transform_arg

        self._ensure_initialized()
        return self._rs.model_handle(transform_arg(posterior_transform, self.num_outputs))

    def forward(self, X: Any) -> _ApproximateOutput:
        """``q(f)`` at ``X`` (transformed by the input transform in train mode,
        as BoTorch's ``forward``)."""
        if self.training and getattr(self, "input_transform", None) is not None:
            X = self.transform_inputs(X)
        return self.model(X)

    __call__ = forward

    def transform_inputs(self, X: Any, input_transform: Any = None) -> Any:
        """``X`` transformed by ``input_transform`` (default: the model's)."""
        tf = (
            input_transform
            if input_transform is not None
            else getattr(self, "input_transform", None)
        )
        return X if tf is None else tf(X)

    def condition_on_observations(self, X: Any, Y: Any, **kwargs: Any) -> Any:
        """Not supported (BoTorch raises ``AttributeError``: approximate models
        have no ``get_fantasy_model``)."""
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute 'get_fantasy_model'"
        )

    # -- copying -----------------------------------------------------------

    def __deepcopy__(self, memo: dict[int, Any]) -> ApproximateGPyTorchModel:
        self._sync_torch_parameters()
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if (
                key == "_torch_parameter_requires_grad"
                and "_torch_parameter_state" in self.__dict__
            ):
                continue
            if key == "_torch_parameter_state":
                new.__dict__["_torch_parameter_requires_grad"] = {
                    name: parameter.requires_grad for name, parameter in value.entries
                }
                continue
            if key == "_rs":
                new.__dict__[key] = value.copy()
            else:
                new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_outputs={self.num_outputs}, "
            f"num_inducing={self._rs.num_inducing})"
        )


class SingleTaskVariationalGP(ApproximateGPyTorchModel):
    r"""A single-task variational GP model following [hensman2013svgp]_.

    By default, the inducing points are initialized though the
    ``GreedyVarianceReduction`` of [burt2020svgp]_, which is known to be
    effective for building globally accurate models. However, custom
    inducing point allocators designed for specific down-stream tasks can also be
    provided (see [moss2023ipa]_ for details), e.g. ``GreedyImprovementReduction``
    when the goal is to build a model suitable for standard BO.

    A single-task variational GP using relatively strong priors on the Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    When the training observations include multiple outputs, this model will
    use batching to model outputs independently (botorch_rs: unbatched
    training inputs only).

    Use this model if you have a lot of data or if your responses are non-Gaussian.

    To train this model, you should use gpytorch.mlls.VariationalELBO and not
    the exact marginal log likelihood.

    Example:
        >>> import torch
        >>> from botorch_rs.models import SingleTaskVariationalGP
        >>> from botorch_rs.mlls import VariationalELBO
        >>>
        >>> train_X = torch.rand(20, 2)
        >>> model = SingleTaskVariationalGP(train_X)
        >>> mll = VariationalELBO(
        >>>     model.likelihood, model.model, num_data=train_X.shape[-2]
        >>> )
    """

    _has_train_inputs = True

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        if cls is SingleTaskVariationalGP:
            if backend not in ("auto", "native", "torch"):
                raise ValueError("backend must be 'auto', 'native', or 'torch'.")
            tensor_backend = backend == "torch" or (
                backend == "auto"
                and any(
                    getattr(value, "requires_grad", False)
                    or getattr(getattr(value, "device", None), "type", "cpu") != "cpu"
                    for value in (train_X, train_Y, kwargs.get("inducing_points"))
                )
            )
            if tensor_backend:
                from botorch_rs.models._torch_backend_variational import (
                    TorchSingleTaskVariationalGP,
                )

                return object.__new__(TorchSingleTaskVariationalGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any = None,
        likelihood: Likelihood | None = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type = VariationalStrategy,
        inducing_points: Any = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        *,
        backend: str = "auto",
    ) -> None:
        r"""
        Args:
            train_X: Training inputs (due to the ability of the SVGP to sub-sample
                this does not have to be all of the training inputs).
            train_Y: Training targets (optional).
            likelihood: Instance of a GPyTorch likelihood. If omitted, uses a
                either a ``GaussianLikelihood`` (if ``num_outputs=1``) or a
                ``MultitaskGaussianLikelihood``(if ``num_outputs>1``).
            num_outputs: Number of output responses per input (default: 1).
            learn_inducing_points: If True, the inducing point locations are learned
                jointly with the other model parameters.
            covar_module: Kernel function. If omitted, uses an ``RBFKernel``.
            mean_module: Mean of GP model. If omitted, uses a ``ConstantMean``.
            variational_distribution: Type of variational distribution to use
                (default: CholeskyVariationalDistribution), the properties of the
                variational distribution will encourage scalability or ease of
                optimization.
            variational_strategy: Type of variational strategy to use (default:
                VariationalStrategy). The default setting uses "whitening" of the
                variational distribution to make training easier.
            inducing_points: The number or specific locations of the inducing points.
            inducing_point_allocator: The ``InducingPointAllocator`` used to
                initialize the inducing point locations. If omitted,
                uses ``GreedyVarianceReduction``.
            outcome_transform: An outcome transform that is applied to the training
                data during instantiation and to the posterior during inference
                (botorch_rs: ``Standardize``).
                NOTE: If this model is trained in minibatches, an outcome transform
                with learnable parameters (such as ``Standardize``) would update its
                parameters for each minibatch, which is undesirable. If you do intend
                to train in minibatches, we recommend you not use an outcome transform
                and instead pre-transform your whole data set before fitting the model.
            input_transform: An input transform that is applied in the model's
                forward pass.
                NOTE: If this model is trained in minibatches, an input transform
                with learnable parameters (such as ``Normalize``) would update its
                parameters for each minibatch, which is undesirable. If you do intend
                to train in minibatches, we recommend you not use an input transform
                and instead pre-transform your whole data set before fitting the model.
        """
        kind = kind_of(train_X)
        x = to_flat(train_X, "train_X")
        if len(x[1]) < 2:
            raise BotorchTensorDimensionError(
                f"Expected train_X to be an `n x d` tensor, got shape {x[1]}."
            )
        d = int(x[1][-1])
        if input_transform is not None and not isinstance(input_transform, InputTransform):
            raise UnsupportedError(
                "`input_transform` must be a botorch_rs input transform (from "
                "botorch_rs.models.transforms.input)."
            )
        # The training inputs transformed by the input transform in train mode
        # (BoTorch's `self.transform_inputs(X=train_X, input_transform=...)`).
        transformed_X = train_X if input_transform is None else input_transform(train_X)
        tx = to_flat(transformed_X, "train_X")
        if any(math.isnan(v) for v in tx[0]):
            raise InputDataError("Input data contains NaN values.")
        if train_Y is not None:
            if outcome_transform is not None:
                warnings.warn(
                    TRANSFORM_WARNING.format(ttype="outcome"),
                    UserInputWarning,
                    stacklevel=2,
                )
                if not isinstance(outcome_transform, Standardize):
                    raise UnsupportedError(
                        "botorch_rs's SingleTaskVariationalGP supports a Standardize outcome "
                        "transform (others, e.g. Log, are not ported for this model)."
                    )
                outcome_transform.train()
                train_Y, _ = outcome_transform(train_Y, X=transformed_X)
            from botorch_rs.models.gp_regression import (
                _validate_input_scaling,
                _validate_tensor_args,
            )

            y = to_flat(train_Y, "train_Y")
            _validate_tensor_args(tuple(tx[1]), tuple(y[1]), None)
            if any(math.isnan(v) for v in y[0]):
                raise InputDataError("Input data contains NaN values.")
            if covar_module is None:
                _validate_input_scaling(tx, y, int(y[1][-2]))
            if y[1][-1] != num_outputs:
                num_outputs = int(y[1][-1])
        elif outcome_transform is not None:
            raise UnsupportedError(
                "An outcome transform needs training targets (`train_Y`) in botorch_rs."
            )
        self._num_outputs = int(num_outputs)
        self._input_batch_shape: tuple[int, ...] = tuple(x[1][:-2])
        aug = (num_outputs,) if num_outputs > 1 else ()
        if likelihood is None:
            likelihood = (
                get_gaussian_likelihood_with_lognormal_prior(batch_shape=aug)
                if num_outputs == 1
                else MultitaskGaussianLikelihood(num_tasks=num_outputs)
            )
        else:
            self._is_custom_likelihood = True
        if learn_inducing_points and (inducing_point_allocator is not None):
            warnings.warn(
                "After all the effort of specifying an inducing point allocator, "
                "you probably want to stop the inducing point locations "
                "being further optimized during the model fit. If so "
                "then set `learn_inducing_points` to False.",
                UserWarning,
                stacklevel=2,
            )
        self._inducing_point_allocator: InducingPointAllocator = (
            GreedyVarianceReduction()
            if inducing_point_allocator is None
            else inducing_point_allocator
        )
        model = _SingleTaskVariationalGP(
            train_X=transformed_X,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=self._inducing_point_allocator,
        )
        super().__init__(model=model, likelihood=likelihood, num_outputs=num_outputs)
        self._kind = kind
        input_steps = prefixes = training = None
        if input_transform is not None:
            spec = chain_spec(input_transform, d, ())
            if any(list(getattr(st.rs, "params", []) or []) for st in spec.chains[0]):
                raise UnsupportedError(
                    "SingleTaskVariationalGP does not yet optimize learned input transforms "
                    "(such as Warp); pre-transform the inputs instead."
                )
            input_steps = [st.rs for st in spec.chains[0]]
            prefixes = [st.prefix for st in spec.chains[0]]
            training = [st.training for st in spec.chains[0]]
        self._build(
            transformed_X,
            train_Y,
            d,
            input_steps,
            prefixes,
            outcome_transform if isinstance(outcome_transform, Standardize) else None,
            training,
            train_X,
        )
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            warnings.warn(
                TRANSFORM_WARNING.format(ttype="input"),
                UserInputWarning,
                stacklevel=2,
            )
            self.input_transform = input_transform
        self._train_Y_kind = kind_of(train_Y) if train_Y is not None else kind

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective. For a model with ``m``
        outputs, a ``test_batch_shape x q x d``-shaped input ``X`` to the ``posterior``
        method returns a Posterior object over an output of shape
        ``broadcast(test_batch_shape, model.batch_shape) x q x m``.
        """
        return self._input_batch_shape

    def init_inducing_points(
        self,
        inputs: Any,
    ) -> Any:
        r"""
        Reinitialize the inducing point locations in-place with the current kernel
        applied to ``inputs`` through the model's inducing point allocation strategy.
        The variational distribution and variational strategy caches are reset.

        Args:
            inputs: (\*batch_shape, n, d)-dim input data tensor.

        Returns:
            (\*batch_shape, m, d)-dim tensor of selected inducing point locations.
        """
        strategy = self.model._strategy
        num_inducing = int(self._rs.num_inducing)
        inducing_points = self._inducing_point_allocator.allocate_inducing_points(
            inputs=inputs,
            covar_module=self.model.covar_module,
            num_inducing=num_inducing,
            input_batch_shape=self._input_batch_shape,
        )
        shape = tuple(to_flat(inducing_points, "inducing_points")[1])
        current = tuple(to_flat(strategy.inducing_points, "inducing_points")[1])
        if shape != current:
            raise RuntimeError(
                f"The size of tensor a ({current[-2]}) must match the size of tensor b "
                f"({shape[-2] if shape else 0}) at non-singleton dimension 0"
            )
        strategy.inducing_points = inducing_points
        strategy.variational_params_initialized.fill_(0)
        return inducing_points


__all__ = ["ApproximateGPyTorchModel", "SingleTaskVariationalGP"]
