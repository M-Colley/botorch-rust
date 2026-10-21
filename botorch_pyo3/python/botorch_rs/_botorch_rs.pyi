"""Type stubs of the ``botorch_rs._botorch_rs`` extension module.

This is the low-level layer used by the ``botorch_rs`` package. Arrays are
exchanged as ``(data, shape)`` tuples: ``data`` holds the values in row-major
order and ``shape`` the dimensions. Arguments accept a float64
``array.array`` (read with one memcpy) or any sequence of numbers as
``data``. Results whose size grows with the inputs (``ArrayBuf``,
``BoolBuf``) return more than 16 values as an ``array.array('d')`` (or
``bytes`` of 0/1 for boolean masks), fewer as a list; small results of fixed
size (``ArrayOut``) return a list. Use the BoTorch-style classes of the
``botorch_rs`` package instead of this module. The ``tensor_*`` functions accept
Torch tensors directly and preserve their device, dtype and autograd graph.
"""

from array import array
from collections.abc import Callable, Mapping, Sequence
from typing import Any

Array = tuple[Sequence[float], Sequence[int]]
ArrayOut = tuple[list[float], list[int]]
ArrayBuf = tuple[list[float] | array[float], list[int]]
BoolBuf = tuple[list[bool] | bytes, list[int]]
#: An MC objective: ``("identity",)``, ``("linear", weights)``,
#: ``("callable", f)`` with ``f(samples, X=X)`` on torch tensors,
#: ``("constrained", f, constraints, infeasible_cost, etas)`` or
#: ``("native", RiskMeasure)``.
ObjectiveSpec = tuple[Any, ...]
#: A linear parameter constraint ``(indices, coefficients, rhs)``: 1-d indices
#: (intra-point) or ``(point, feature)`` pairs (inter-point).
LinearConstraint = tuple[Sequence[int] | Sequence[Sequence[int]], Sequence[float], float]
#: A nonlinear inequality constraint ``(callable, is_intrapoint)`` with
#: ``callable(x) >= 0`` on torch tensors.
NonlinearConstraint = tuple[Callable[..., Any], bool]

__version__: str

def set_num_threads(num_threads: int | None = None) -> None:
    """Sets the number of threads (``None`` or 0: one per logical CPU)."""

def get_num_threads() -> int:
    """The number of threads used for parallel work."""

class Normalize:
    def __init__(
        self,
        d: int,
        bounds: Array | None = None,
        learn_bounds: bool = True,
        min_range: float = 1e-8,
        batch_shape: Sequence[int] = ...,
    ) -> None: ...
    @property
    def d(self) -> int: ...
    @property
    def batch_shape(self) -> list[int]: ...
    @property
    def learn_bounds(self) -> bool: ...
    @property
    def min_range(self) -> float: ...
    @property
    def offset(self) -> ArrayOut: ...
    @property
    def coefficient(self) -> ArrayOut: ...
    @property
    def bounds(self) -> ArrayOut: ...
    def fit(self, x: Array) -> Normalize: ...
    def transform(self, x: Array) -> ArrayBuf: ...
    def untransform(self, x: Array) -> ArrayBuf: ...

class Standardize:
    def __init__(self, min_stdv: float = 1e-8, batch_shape: Sequence[int] = ...) -> None: ...
    @staticmethod
    def fitted(
        means: Sequence[Sequence[float]],
        stdvs: Sequence[Sequence[float]],
        min_stdv: float,
        batch_shape: Sequence[int],
    ) -> Standardize: ...
    @property
    def min_stdv(self) -> float: ...
    @property
    def batch_shape(self) -> list[int]: ...
    @property
    def is_trained(self) -> bool: ...
    @property
    def means(self) -> ArrayOut: ...
    @property
    def stdvs(self) -> ArrayOut: ...
    def fit_transform(
        self, y: Array, yvar: Array | None = None
    ) -> tuple[Standardize, ArrayBuf, ArrayBuf | None]: ...
    def transform(
        self, y: Array, yvar: Array | None = None
    ) -> tuple[ArrayBuf, ArrayBuf | None]: ...
    def untransform(
        self, y: Array, yvar: Array | None = None
    ) -> tuple[ArrayBuf, ArrayBuf | None]: ...

class Prior:
    @staticmethod
    def log_normal(loc: float, scale: float, float32: bool = True) -> Prior: ...
    @staticmethod
    def gamma(concentration: float, rate: float, float32: bool = True) -> Prior: ...
    @staticmethod
    def normal(loc: float, scale: float, float32: bool = True) -> Prior: ...
    @staticmethod
    def uniform(a: float, b: float, float32: bool = True) -> Prior: ...
    @staticmethod
    def half_cauchy(scale: float, float32: bool = True) -> Prior: ...
    @staticmethod
    def half_normal(scale: float, float32: bool = True) -> Prior: ...
    @staticmethod
    def beta(concentration1: float, concentration0: float, float32: bool = True) -> Prior: ...
    @staticmethod
    def smoothed_box(a: float, b: float, sigma: float, float32: bool = True) -> Prior: ...
    @property
    def kind(self) -> str: ...
    @property
    def parameters(self) -> list[float]: ...
    @property
    def mode(self) -> float: ...
    def log_prob(self, x: float) -> float: ...
    def grad_log_prob(self, x: float) -> float: ...
    def in_support(self, x: float) -> bool: ...
    def with_torch_log(self, value: float) -> Prior: ...

class Constraint:
    @staticmethod
    def greater_than(
        lower_bound: float,
        transform: str | None = "softplus",
        initial_value: float | None = None,
        float32: bool = True,
    ) -> Constraint: ...
    @staticmethod
    def interval(
        lower_bound: float,
        upper_bound: float,
        transform: str | None = "sigmoid",
        inv_transform: str | None = None,
        initial_value: float | None = None,
        float32: bool = True,
    ) -> Constraint: ...
    @staticmethod
    def less_than(
        upper_bound: float,
        transform: str | None = "softplus",
        initial_value: float | None = None,
        float32: bool = True,
    ) -> Constraint: ...
    @staticmethod
    def log_transformed_interval(
        lower_bound: float,
        upper_bound: float,
        initial_value: float | None = None,
        float32: bool = True,
        initial_raw: float | None = None,
    ) -> Constraint: ...
    @property
    def kind(self) -> str: ...
    @property
    def lower_bound(self) -> float: ...
    @property
    def upper_bound(self) -> float: ...
    @property
    def enforced(self) -> bool: ...
    @property
    def transform_name(self) -> str | None: ...
    @property
    def initial_value(self) -> float | None: ...
    @property
    def initial_raw_value(self) -> float | None: ...
    @property
    def raw_bounds(self) -> tuple[float, float]: ...
    def transform(self, raw: float) -> float: ...
    def inverse_transform(self, value: float) -> float: ...
    def transform_derivative(self, raw: float) -> float: ...

class KernelSpec:
    def __init__(
        self,
        kind: str,
        ard_num_dims: int,
        lengthscale_prior: Prior | None = None,
        lengthscale_constraint: Constraint | None = None,
    ) -> None: ...
    @staticmethod
    def dim_scaled(ard_num_dims: int, use_rbf_kernel: bool = True) -> KernelSpec: ...
    @staticmethod
    def matern_gamma(ard_num_dims: int) -> KernelSpec: ...
    def with_scale_kernel(
        self,
        outputscale_prior: Prior | None = None,
        outputscale_constraint: Constraint | None = None,
    ) -> KernelSpec: ...
    @property
    def kind(self) -> str: ...
    @property
    def ard_num_dims(self) -> int: ...
    @property
    def has_scale_kernel(self) -> bool: ...
    @property
    def lengthscale_prior(self) -> Prior | None: ...
    @property
    def lengthscale_constraint(self) -> Constraint: ...
    @property
    def outputscale_prior(self) -> Prior | None: ...
    @property
    def outputscale_constraint(self) -> Constraint | None: ...

def posterior_query_vjp(
    source: Any,
    x: Array,
    adjoint: Array,
    mode: str,
    observation_noise: bool = False,
    output_indices: Sequence[int] | None = None,
    transform: Transform | None = None,
    base: bool = False,
) -> ArrayBuf: ...
def posterior_query_samples_vjp(
    source: Any,
    x: Array,
    base_samples: Array,
    adjoint: Array,
    observation_noise: bool = False,
    output_indices: Sequence[int] | None = None,
    transform: Transform | None = None,
    base: bool = False,
) -> ArrayBuf: ...
def target_task_model(
    model: Model, feature_indices: Sequence[int], imputation: Sequence[float]
) -> Model: ...
def additive_component_model(model: SingleTaskGP, extra_batch_dims: int = 0) -> Model: ...
def gaussian_kl(
    mu_p: Array,
    sigma_p: Array,
    mu_q: Array,
    sigma_q: Array,
    gradients: bool = False,
) -> tuple[float, tuple[ArrayBuf, ArrayBuf, ArrayBuf, ArrayBuf] | None]: ...

class KernelPriorKl:
    def __init__(
        self,
        node: KernelNode,
        points: Array,
        mean: Array,
        covariance: Array,
        learn_mean: bool = True,
        mean_constraint: Constraint | None = None,
    ) -> None: ...
    def value_and_grad(self, raw: Sequence[float]) -> tuple[float, list[float]]: ...
    def fit(
        self,
        raw: Sequence[float],
        maxcor: int | None = None,
        pgtol: float | None = None,
        ftol: float | None = None,
        maxiter: int | None = None,
        maxfun: int | None = None,
        maxls: int | None = None,
        max_attempts: int | None = None,
        seed: int | None = None,
    ) -> tuple[list[float], float, int, int, str, str, int]: ...

def additive_kernel_parts(
    node: KernelNode,
    raw: Sequence[float],
    input_dim: int,
    x1: Array | None = None,
    x2: Array | None = None,
    mode: str = "base",
    diag: bool = False,
    eps: float = 1e-6,
) -> ArrayBuf: ...

class KernelNode:
    @staticmethod
    def heterogeneous(
        task_feature: int,
        binary_map: Sequence[Sequence[bool]],
        spatial: Sequence[KernelNode],
        combinatorial: KernelNode,
        use_combinatorial: bool,
    ) -> KernelNode: ...
    @staticmethod
    def empirical_basis(
        knots: Sequence[Sequence[float]],
        tables: Sequence[Array],
        weights: Sequence[float],
        learn_weights: bool,
        denominator: float,
        group_column: int | None = None,
        independent: bool = False,
    ) -> KernelNode: ...
    @staticmethod
    def contextual_lcea(
        contexts: list[list[int]],
        categorical: list[list[int]],
        embedding_shapes: list[tuple[int, int]],
        fixed_embeddings: Array,
        context_weights: list[float] | None,
        task: KernelNode,
        base: KernelNode,
    ) -> KernelNode: ...
    def contextual_lcea_normalize(self, raw: list[float]) -> list[float]: ...
    def contextual_lcea_embeddings(self, raw: list[float]) -> ArrayOut: ...
    def contextual_lcea_covariance(self, raw: list[float]) -> ArrayOut: ...
    @staticmethod
    def activation(conditions: Sequence[tuple[int, float]]) -> KernelNode: ...
    def with_parent_active_dims(self, dims: Sequence[int]) -> KernelNode: ...
    @staticmethod
    def contextual_sac(
        decomposition: list[tuple[str, list[int]]], base: KernelNode
    ) -> KernelNode: ...
    @staticmethod
    def integrated_white_noise(order: int = 1) -> KernelNode: ...
    @staticmethod
    def stationary(kind: str, ard_num_dims: int | None = None) -> KernelNode: ...
    @staticmethod
    def rational_quadratic(ard_num_dims: int | None = None) -> KernelNode: ...
    @staticmethod
    def periodic(ard_num_dims: int | None = None) -> KernelNode: ...
    @staticmethod
    def linear(ard_num_dims: int | None = None) -> KernelNode: ...
    @staticmethod
    def polynomial(power: float) -> KernelNode: ...
    @staticmethod
    def cosine() -> KernelNode: ...
    @staticmethod
    def index(num_tasks: int, rank: int) -> KernelNode: ...
    @staticmethod
    def positive_index(
        num_tasks: int, rank: int, target_task_index: int, unit_scale_for_target: bool
    ) -> KernelNode: ...
    @staticmethod
    def categorical(ard_num_dims: int | None = None) -> KernelNode: ...
    @staticmethod
    def downsampling() -> KernelNode: ...
    @staticmethod
    def exponential_decay(ard_num_dims: int | None = None) -> KernelNode: ...
    @staticmethod
    def infinite_width_bnn(depth: int, acos_eps: float) -> KernelNode: ...
    @staticmethod
    def linear_truncated_fidelity(
        fidelity_dims: Sequence[int], unbiased: KernelNode, biased: KernelNode
    ) -> KernelNode: ...
    @staticmethod
    def orthogonal_additive(
        dim: int,
        base: KernelNode,
        per_dim_lengthscales: bool,
        quad_deg: int,
        second_order: bool,
        check_hypercube: bool,
        coeff_constraint: Constraint,
    ) -> KernelNode: ...
    @staticmethod
    def scale(base: KernelNode) -> KernelNode: ...
    @staticmethod
    def additive(kernels: Sequence[KernelNode]) -> KernelNode: ...
    @staticmethod
    def product(kernels: Sequence[KernelNode]) -> KernelNode: ...
    def with_active_dims(self, active_dims: Sequence[int]) -> KernelNode: ...
    def with_param(
        self,
        name: str,
        constraint: Constraint | None = None,
        prior: Prior | None = None,
        default_raw: float | None = None,
    ) -> KernelNode: ...
    def with_task_prior(self, prior: Prior) -> KernelNode: ...
    def with_task_prior_lkj(self, n: int, eta: float, sd_prior: Prior) -> KernelNode: ...
    def with_saas_prior(
        self,
        prior: Prior,
        tau: float | None = None,
        tau_constraint: Constraint | None = None,
        tau_prior: Prior | None = None,
    ) -> KernelNode: ...
    def saas_taus(self) -> list[float]: ...
    def own_params(self) -> list[tuple[str, list[int], float]]: ...
    def param_shapes(self, prefix: str = "") -> list[tuple[str, list[int]]]: ...
    @property
    def num_raw(self) -> int: ...
    def initial_raw(self) -> list[float]: ...
    def validate(self, input_dim: int) -> None: ...
    def as_kernel_spec(self, input_dim: int) -> KernelSpec | None: ...
    @property
    def kind(self) -> str: ...
    def feature_batch_size(self, input_dim: int) -> int: ...
    def evaluate(
        self,
        raw: Sequence[float],
        x1: Array,
        x2: Array | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> ArrayBuf: ...
    def log_prior(self, raw: Sequence[float], input_dim: int) -> tuple[float, list[float]]: ...
    def evaluate_vjp(
        self, raw: Sequence[float], x1: Array, x2: Array, adjoint: Array
    ) -> tuple[ArrayBuf, ArrayBuf]: ...

def leggauss(deg: int) -> tuple[list[float], list[float]]: ...
def lkj_covariance_log_prob(matrix: Array, eta: float, sd_prior: Prior) -> float: ...

class GaussianLikelihoodSpec:
    def __init__(
        self, noise_prior: Prior | None = None, noise_constraint: Constraint | None = None
    ) -> None: ...
    @staticmethod
    def lognormal() -> GaussianLikelihoodSpec: ...
    @staticmethod
    def gamma() -> GaussianLikelihoodSpec: ...
    @property
    def noise_prior(self) -> Prior | None: ...
    @property
    def noise_constraint(self) -> Constraint: ...

Transform = tuple[Sequence[float], float]
ElementFit = tuple[float, int, int, str, str, int]

class SingleTaskGP:
    def __init__(
        self,
        train_x: Array,
        train_y: Array,
        train_yvar: Array | None = None,
        input_transform: Normalize | None = None,
        outcome_transform: Standardize | None = None,
        standardize_by_default: bool = True,
        covar_module: KernelSpec | None = None,
        likelihood: GaussianLikelihoodSpec | None = None,
        input_chain: Sequence[Normalize | InputPerturbation | InputStep] | None = None,
        covar_tree: KernelNode | None = None,
        input_chain_training: Sequence[bool] | None = None,
        input_chain_prefixes: Sequence[str] | None = None,
        input_chains: Sequence[Sequence[Normalize | InputPerturbation | InputStep]] | None = None,
        outcome_chains: Sequence[Sequence[OutcomeStep]] | None = None,
        input_chain_params: Sequence[Sequence[float]] | None = None,
        mean_prior: Prior | None = None,
        mean_constraint: Constraint | None = None,
    ) -> None: ...
    def copy(self) -> SingleTaskGP: ...
    def set_saas_taus(self, taus: list[list[float]]) -> None: ...
    def subset_output(self, idcs: list[int]) -> SingleTaskGP: ...
    def batch_element(self, batch: int) -> SingleTaskGP: ...
    @property
    def points_per_input(self) -> int: ...
    def input_chain_state(self) -> list[Normalize | int | None] | None: ...
    def input_chain_states(self) -> list[list[Any]] | None: ...
    def outcome_chain_states(self) -> list[list[OutcomeStep]] | None: ...
    def construction_train_inputs(self) -> ArrayBuf: ...
    @property
    def transformed_dim(self) -> int: ...
    @property
    def transformed_outputs(self) -> list[bool]: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_train(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def batch_shape(self) -> list[int]: ...
    @property
    def aug_batch_shape(self) -> list[int]: ...
    @property
    def is_fixed_noise(self) -> bool: ...
    def train_inputs(self) -> ArrayBuf: ...
    def train_targets(self) -> ArrayBuf: ...
    def train_yvar(self) -> ArrayBuf | None: ...
    def transformed_train_inputs(self) -> ArrayBuf: ...
    def transformed_train_targets(self) -> ArrayBuf: ...
    def loo_predictions(
        self, observation_noise: bool = True
    ) -> tuple[ArrayBuf, ArrayBuf, ArrayBuf, ArrayBuf | None]: ...
    @staticmethod
    def diagonal_samples(
        mean: Sequence[float], variance: Sequence[float], base: Array, qm: int
    ) -> list[float]: ...
    def input_transform(self) -> Normalize | None: ...
    def outcome_transform(self) -> Standardize | None: ...
    def covar_module(self) -> KernelSpec | None: ...
    def covar_tree(self) -> KernelNode | None: ...
    def likelihood_spec(self) -> GaussianLikelihoodSpec | None: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def parameter_shapes(self) -> list[tuple[str, list[int]]]: ...
    def raw_parameters(self) -> list[float]: ...
    @property
    def parameter_state_version(self) -> int: ...
    def posterior_samples_parameter_vjp(
        self,
        x: Array,
        base_samples: Array,
        adjoint: Array,
        observation_noise: bool = False,
        output_indices: list[int] | None = None,
        transform: tuple[list[float], float] | None = None,
        base: bool = False,
    ) -> ArrayBuf: ...
    def mll_parameter_vjp(self, adjoint: Array) -> ArrayBuf: ...
    def posterior_moment_parameter_vjp(
        self,
        x: Array,
        adjoint: Array,
        mode: str,
        observation_noise: bool = False,
        output_indices: list[int] | None = None,
        transform: tuple[list[float], float] | None = None,
        base: bool = False,
    ) -> ArrayBuf: ...
    def set_raw_parameters(self, values: Sequence[float]) -> None: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    @property
    def lengthscale(self) -> list[float]: ...
    @property
    def outputscale(self) -> list[float] | None: ...
    @property
    def noise(self) -> list[float]: ...
    @property
    def constant(self) -> list[float]: ...
    def set_lengthscale(self, value: Array) -> None: ...
    def set_outputscale(self, value: Array) -> None: ...
    def set_noise(self, value: Array) -> None: ...
    def set_constant(self, value: Array) -> None: ...
    def observation_noise(self) -> list[float]: ...
    def posterior_marginals(
        self,
        x: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
        base: bool = False,
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def posterior_joint(
        self,
        x: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
        base: bool = False,
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def posterior_samples(
        self,
        x: Array,
        base_samples: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
        base: bool = False,
    ) -> ArrayBuf: ...
    def replace_transformed_train_data(
        self, targets: Array, fixed_noise: Array | None = None, strict: bool = True
    ) -> None: ...
    def set_sparse_outlier_noise(
        self,
        support: list[int],
        raw: list[float],
        convex: bool = True,
        lower: float = 0.0,
        upper: float = 0.999,
        prior: Prior | None = None,
    ) -> None: ...
    def sparse_outlier_state(self) -> tuple[list[int], list[float]]: ...
    def sparse_outlier_expansion(self, loo: bool) -> list[float]: ...
    def mll(self, raw_parameters: Sequence[float] | None = None) -> ArrayOut: ...
    def posterior_parameter_vjp(
        self,
        x: Array,
        mean_adjoint: Array,
        covariance_adjoint: Array,
        observation_noise: bool = False,
    ) -> ArrayBuf: ...
    def mll_value_and_grad(
        self, raw_parameters: Sequence[float] | None = None
    ) -> tuple[ArrayOut, list[float]]: ...
    def fit(
        self,
        maxcor: int | None = None,
        pgtol: float | None = None,
        ftol: float | None = None,
        maxiter: int | None = None,
        maxfun: int | None = None,
        maxls: int | None = None,
        max_attempts: int | None = None,
        seed: int | None = None,
    ) -> tuple[float, int, int, str, str, int, list[ElementFit]]: ...

class ModelListGP:
    def posterior_moment_parameter_vjp(
        self,
        x: Array,
        adjoint: Array,
        mode: str,
        observation_noise: bool = False,
        output_indices: list[int] | None = None,
        transform: tuple[list[float], float] | None = None,
        base: bool = False,
    ) -> ArrayBuf: ...
    def posterior_samples_parameter_vjp(
        self,
        x: Array,
        base_samples: Array,
        adjoint: Array,
        observation_noise: bool = False,
        output_indices: list[int] | None = None,
        transform: tuple[list[float], float] | None = None,
        base: bool = False,
    ) -> ArrayBuf: ...
    def __init__(self, models: Sequence[SingleTaskGP]) -> None: ...
    @property
    def transformed_outputs(self) -> list[bool]: ...
    @property
    def points_per_input(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def batch_shape(self) -> list[int]: ...
    def posterior_marginals(
        self,
        x: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
        base: bool = False,
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def posterior_joint(
        self,
        x: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
        base: bool = False,
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def posterior_samples(
        self,
        x: Array,
        base_samples: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
        base: bool = False,
    ) -> ArrayBuf: ...

class LCEMGP:
    def __init__(
        self,
        train_x: Array,
        train_y: Array,
        task_feature: int,
        output_tasks: Sequence[int] | None = None,
        all_tasks: Sequence[int] | None = None,
        context_cat_feature: Array | None = None,
        context_emb_feature: Array | None = None,
        embs_dim_list: Sequence[int] | None = None,
        outcome_transform: Standardize | None = None,
        standardize_by_default: bool = True,
        covar_module: KernelSpec | None = None,
        embedding_weights: Sequence[Array] | None = None,
        seed: int | None = None,
        provided_embedding: bool = False,
    ) -> None: ...
    def copy(self) -> LCEMGP: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_train(self) -> int: ...
    @property
    def num_tasks(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def task_feature(self) -> int: ...
    @property
    def all_tasks(self) -> list[int]: ...
    @property
    def output_tasks(self) -> list[int]: ...
    @property
    def observed_task_indices(self) -> list[int]: ...
    @property
    def tasks_are_indices(self) -> bool: ...
    @property
    def embedding_shapes(self) -> list[tuple[int, int]]: ...
    @property
    def embedding_dim(self) -> int: ...
    def context_cat_feature(self) -> list[list[int]]: ...
    def context_emb_feature(self) -> ArrayOut | None: ...
    def train_inputs(self) -> ArrayBuf: ...
    def train_targets(self) -> ArrayBuf: ...
    def transformed_train_targets(self) -> ArrayBuf: ...
    def outcome_transform(self) -> Standardize | None: ...
    def covar_module(self) -> KernelSpec: ...
    def task_covar_module(self) -> KernelSpec: ...
    def set_task_covar_module(self, spec: KernelSpec) -> None: ...
    def likelihood_spec(self) -> GaussianLikelihoodSpec: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def parameter_shapes(self) -> list[tuple[str, list[int]]]: ...
    def raw_parameters(self) -> list[float]: ...
    def set_raw_parameters(self, values: Sequence[float]) -> None: ...
    def set_raw_parameter(self, name: str, values: Sequence[float]) -> None: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    @property
    def is_provided_embedding(self) -> bool: ...
    @property
    def context_specific_weight(self) -> float | None: ...
    def eval_unobserved_means(self) -> None: ...
    @property
    def noise(self) -> list[float]: ...
    @property
    def constant(self) -> list[float]: ...
    def lengthscale_of(self, which: str) -> list[float]: ...
    def outputscale_of(self, which: str) -> float | None: ...
    def set_lengthscale_of(self, which: str, values: Sequence[float]) -> None: ...
    def set_outputscale_of(self, which: str, value: float) -> None: ...
    def set_noise(self, values: Sequence[float]) -> None: ...
    def embedding_weights(self) -> list[ArrayOut]: ...
    def renorm_embeddings(self) -> None: ...
    def task_embeddings(self) -> ArrayOut: ...
    def task_covariance(self) -> ArrayOut: ...
    def mll(self, raw_parameters: Sequence[float] | None = None) -> ArrayOut: ...
    def mll_value_and_grad(
        self, raw_parameters: Sequence[float] | None = None
    ) -> tuple[ArrayOut, list[float]]: ...
    def fit(
        self,
        maxcor: int | None = None,
        pgtol: float | None = None,
        ftol: float | None = None,
        maxiter: int | None = None,
        maxfun: int | None = None,
        maxls: int | None = None,
        max_attempts: int | None = None,
        seed: int | None = None,
    ) -> tuple[float, int, int, str, str, int, list[ElementFit]]: ...

class MultiTaskGP:
    def _set_fitted_outcome_affine(self, offset: float, scale: float) -> None: ...
    def set_input_feature_offsets(
        self, indices: Sequence[int], offsets: Sequence[float]
    ) -> None: ...
    def set_input_warp(
        self, indices: Sequence[int], c0: Sequence[float], c1: Sequence[float], eps: float
    ) -> None: ...
    def __init__(
        self,
        train_x: Array,
        train_y: Array,
        task_feature: int,
        train_yvar: Array | None = None,
        covar_module: KernelNode | None = None,
        covar_module_active_dims: Sequence[int] | None = None,
        task_covar_module: KernelNode | None = None,
        default_task_prior: bool = True,
        likelihood: GaussianLikelihoodSpec | None = None,
        likelihood_kind: str = "hadamard",
        mean_kind: str = "multitask",
        output_tasks: Sequence[int] | None = None,
        rank: int | None = None,
        all_tasks: Sequence[int] | None = None,
        outcome_transform: Standardize | None = None,
        standardize_by_default: bool = True,
        input_transform: Normalize | None = None,
        input_transform_indices: Sequence[int] | None = None,
        validate_task_values: bool = True,
    ) -> None: ...
    def copy(self) -> MultiTaskGP: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_train(self) -> int: ...
    @property
    def num_tasks(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def task_feature(self) -> int: ...
    @property
    def all_tasks(self) -> list[int]: ...
    @property
    def output_tasks(self) -> list[int]: ...
    @property
    def observed_task_indices(self) -> list[int]: ...
    @property
    def tasks_are_indices(self) -> bool: ...
    @property
    def batch_shape(self) -> list[int]: ...
    @property
    def aug_batch_shape(self) -> list[int]: ...
    @property
    def rank(self) -> int: ...
    @property
    def task_kernel_index(self) -> int: ...
    @property
    def is_fixed_noise(self) -> bool: ...
    def covar_tree(self) -> KernelNode: ...
    def train_inputs(self) -> ArrayBuf: ...
    def train_targets(self) -> ArrayBuf: ...
    def transformed_train_targets(self) -> ArrayBuf: ...
    def transformed_train_inputs(self) -> ArrayBuf: ...
    def outcome_transform(self) -> Standardize | None: ...
    def input_transform(self, learn_bounds: bool) -> Normalize | None: ...
    def likelihood_spec(self) -> GaussianLikelihoodSpec | None: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def parameter_shapes(self) -> list[tuple[str, list[int]]]: ...
    def raw_parameters(self) -> list[float]: ...
    def set_raw_parameters(self, values: Sequence[float]) -> None: ...
    def set_raw_parameter(self, name: str, values: Sequence[float]) -> None: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    @property
    def noise(self) -> list[float]: ...
    def set_noise(self, values: Sequence[float] | Array) -> None: ...
    @property
    def constant(self) -> list[float]: ...
    def set_constant(self, value: Array) -> None: ...
    def eval_unobserved(self) -> None: ...
    def task_covariance(self) -> ArrayOut: ...
    def mll(self, raw_parameters: Sequence[float] | None = None) -> ArrayOut: ...
    def mll_value_and_grad(
        self, raw_parameters: Sequence[float] | None = None
    ) -> tuple[ArrayOut, list[float]]: ...
    def fit(
        self,
        maxcor: int | None = None,
        pgtol: float | None = None,
        ftol: float | None = None,
        maxiter: int | None = None,
        maxfun: int | None = None,
        maxls: int | None = None,
        max_attempts: int | None = None,
        seed: int | None = None,
    ) -> tuple[float, int, int, str, str, int, list[ElementFit]]: ...
    def condition_on_observations(
        self, x: Array, y: Array, noise: Array | None = None
    ) -> MultiTaskGP: ...
    def fantasize(
        self, x: Array, sampler: SobolQMCNormalSampler, observation_noise: Array | None = None
    ) -> MultiTaskGP: ...

def multitask_model(
    model: MultiTaskGP,
    output_tasks: Sequence[int] | None = None,
    transform: Transform | None = None,
) -> Model: ...
def multitask_task_inputs_model(model: MultiTaskGP) -> Model: ...

class KroneckerMultiTaskGP:
    def __init__(
        self,
        train_x: Array,
        train_y: Array,
        data_covar_module: KernelNode | None = None,
        task_covar_module: KernelNode | None = None,
        likelihood: GaussianLikelihoodSpec | None = None,
        likelihood_rank: int = 0,
        rank: int | None = None,
        outcome_transform: Standardize | None = None,
        input_transform: Normalize | None = None,
    ) -> None: ...
    def copy(self) -> KroneckerMultiTaskGP: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_train(self) -> int: ...
    @property
    def num_tasks(self) -> int: ...
    @property
    def aug_batch_shape(self) -> list[int]: ...
    def train_inputs(self) -> ArrayBuf: ...
    def transformed_train_targets(self) -> ArrayBuf: ...
    def train_full_covar(self) -> ArrayBuf: ...
    def predictive_mean_cache(self) -> ArrayBuf: ...
    def outcome_transform(self) -> Standardize | None: ...
    def input_transform(self, learn_bounds: bool) -> Normalize | None: ...
    def likelihood_spec(self) -> GaussianLikelihoodSpec: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def parameter_shapes(self) -> list[tuple[str, list[int]]]: ...
    def raw_parameters(self) -> list[float]: ...
    def set_raw_parameters(self, values: Sequence[float]) -> None: ...
    def set_raw_parameter(self, name: str, values: Sequence[float]) -> None: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    def noises(self) -> tuple[list[float] | None, float]: ...
    def task_noise_covariance(self) -> ArrayOut: ...
    @property
    def constant(self) -> list[float]: ...
    def task_covariance(self) -> ArrayOut: ...
    def mll(self, raw_parameters: Sequence[float] | None = None) -> ArrayOut: ...
    def mll_value_and_grad(
        self, raw_parameters: Sequence[float] | None = None
    ) -> tuple[ArrayOut, list[float]]: ...
    def fit(
        self,
        maxcor: int | None = None,
        pgtol: float | None = None,
        ftol: float | None = None,
        maxiter: int | None = None,
        maxfun: int | None = None,
        maxls: int | None = None,
        max_attempts: int | None = None,
        seed: int | None = None,
    ) -> tuple[float, int, int, str, str, int, list[ElementFit]]: ...
    def matheron_samples(self, x: Array, z: Array, observation_noise: bool = False) -> ArrayBuf: ...
    def marginal_vjp(
        self,
        x: Array,
        mean_adjoint: Array,
        variance_adjoint: Array,
        observation_noise: bool = False,
    ) -> ArrayBuf: ...
    def matheron_samples_vjp(
        self, x: Array, z: Array, adjoint: Array, observation_noise: bool = False
    ) -> ArrayBuf: ...

def kronecker_model(model: KroneckerMultiTaskGP) -> Model: ...

class Model:
    @staticmethod
    def from_batch_models(models: Sequence[Model], batch_shape: Sequence[int]) -> Model: ...
    @property
    def points_per_input(self) -> int: ...
    @staticmethod
    def from_single_task_gp(model: SingleTaskGP, transform: Transform | None = None) -> Model: ...
    @staticmethod
    def from_lcemgp(
        model: LCEMGP,
        output_tasks: Sequence[int] | None = None,
        transform: Transform | None = None,
    ) -> Model: ...
    @staticmethod
    def from_lcemgp_task_inputs(model: LCEMGP) -> Model: ...
    @staticmethod
    def from_models(models: Sequence[Model], transform: Transform | None = None) -> Model: ...
    def posterior_marginals(
        self,
        x: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
        base: bool = False,
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def posterior_joint(
        self,
        x: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
        base: bool = False,
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def posterior_samples(
        self,
        x: Array,
        base_samples: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
        base: bool = False,
    ) -> ArrayBuf: ...
    @staticmethod
    def from_model_list(model: ModelListGP, transform: Transform | None = None) -> Model: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def batch_shape(self) -> list[int]: ...
    @property
    def is_deterministic(self) -> bool: ...

class SobolQMCNormalSampler:
    def __init__(self, num_samples: int, seed: int | None = None) -> None: ...
    @staticmethod
    def iid(num_samples: int, seed: int | None = None) -> SobolQMCNormalSampler:
        """A sampler with i.i.d. standard-normal base samples (torch's
        ``randn`` after ``manual_seed(seed)``, reproduced natively)."""
    @property
    def is_iid(self) -> bool: ...
    @property
    def num_samples(self) -> int: ...
    @property
    def seed(self) -> int: ...
    def base_samples(self, dim: int) -> ArrayBuf: ...
    def set_base_samples(self, base_samples: Array) -> None: ...
    def __deepcopy__(self, memo: Any) -> SobolQMCNormalSampler: ...
    def __copy__(self) -> SobolQMCNormalSampler: ...

class AcquisitionParameterTape:
    def parameter_vjp(self, weights: Array, parameter_count: int) -> ArrayBuf: ...

class AcquisitionFunction:
    @property
    def supports_parameter_gradients(self) -> bool: ...
    def forward_with_adjoint_tape(
        self, x: Array, query_gradient: bool = False
    ) -> tuple[ArrayBuf, ArrayBuf | None, AcquisitionParameterTape]: ...
    @staticmethod
    def analytic(
        name: str,
        model: Model,
        best_f: float | None = None,
        beta: float | None = None,
        maximize: bool = True,
    ) -> AcquisitionFunction: ...
    @staticmethod
    def monte_carlo(
        name: str,
        model: Model,
        sampler: SobolQMCNormalSampler | None = None,
        x_pending: Array | None = None,
        best_f: float | None = None,
        beta: float | None = None,
        tau: float = 1e-3,
        fat: bool = True,
        tau_max: float = 1e-2,
        tau_relu: float = 1e-6,
        objective: ObjectiveSpec | None = None,
        constraints: Sequence[Callable[[Any], Any]] | None = None,
        eta: Sequence[float] | None = None,
        fat_list: Sequence[bool | None] | None = None,
        objective_batch_shape: Sequence[int] | None = None,
    ) -> AcquisitionFunction: ...
    @staticmethod
    def q_log_probability_of_feasibility(
        model: Model,
        constraints: Sequence[Callable[[Any], Any]],
        eta: Sequence[float],
        sampler: SobolQMCNormalSampler | None = None,
        x_pending: Array | None = None,
        fat: bool = True,
        tau_max: float = 1e-2,
        fat_list: Sequence[bool | None] | None = None,
    ) -> AcquisitionFunction: ...
    @staticmethod
    def constrained(
        name: str,
        model: Model,
        best_f: float,
        objective_index: int,
        constraints: Sequence[tuple[int, float | None, float | None]],
        maximize: bool = True,
    ) -> AcquisitionFunction: ...
    @staticmethod
    def q_log_noisy_expected_improvement(
        model: Model,
        x_baseline: Array,
        sampler: SobolQMCNormalSampler | None = None,
        x_pending: Array | None = None,
        fat: bool = True,
        prune_baseline: bool = True,
        cache_root: bool = True,
        prune_num_samples: int = 2048,
        prune_max_frac: float = 1.0,
        prune_seed: int | None = None,
        tau_max: float = 1e-2,
        tau_relu: float = 1e-6,
        incremental: bool = True,
        objective: ObjectiveSpec | None = None,
        constraints: Sequence[Callable[[Any], Any]] | None = None,
        eta: Sequence[float] | None = None,
        fat_list: Sequence[bool | None] | None = None,
        lower_bound_points: Array | None = None,
        objective_batch_shape: Sequence[int] | None = None,
    ) -> AcquisitionFunction: ...
    @property
    def name(self) -> str: ...
    @property
    def dim(self) -> int: ...
    @property
    def expected_q(self) -> int | None: ...
    @property
    def is_nonnegative(self) -> bool: ...
    @property
    def is_log(self) -> bool: ...
    @property
    def cache_root(self) -> bool: ...
    def baseline_best_f(self) -> list[float] | None: ...
    def last_baseline_best_f(self) -> ArrayBuf | None: ...
    def forward(self, x: Array) -> ArrayBuf: ...
    def forward_with_grad(self, x: Array) -> tuple[ArrayBuf, ArrayBuf]: ...
    def x_pending(self) -> ArrayOut | None: ...
    def set_x_pending(self, x_pending: Array | None = None) -> None: ...
    def x_baseline(self) -> ArrayOut: ...
    def full_x_baseline(self) -> ArrayOut: ...
    def lower_bound_points(self) -> ArrayOut: ...
    def infeasible_lower_bound(self) -> float | None: ...

def prune_inferior_points(
    model: Model,
    x: Array,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    seed: int | None = None,
    base_samples: Array | None = None,
    objective: ObjectiveSpec | None = None,
    constraints: Sequence[Callable[[Any], Any]] | None = None,
) -> ArrayOut: ...
def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Array,
    q: int,
    num_restarts: int,
    raw_samples: int | None = None,
    *,
    seed: int | None = None,
    maxiter: int | None = None,
    maxfun: int | None = None,
    ftol: float | None = None,
    gtol: float | None = None,
    maxcor: int | None = None,
    maxls: int | None = None,
    batch_limit: int | None = None,
    init_batch_limit: int | None = None,
    eta: float | None = None,
    alpha: float | None = None,
    nonnegative: bool = False,
    fixed_features: dict[int, float] | None = None,
    batch_initial_conditions: Array | None = None,
    sequential: bool = False,
    retry_on_optimization_warning: bool = True,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    nonlinear_inequality_constraints: Sequence[NonlinearConstraint] | None = None,
    ic_generator: Callable[[int, int, int | None], ArrayOut] | None = None,
    native_ic_generator: bool = False,
    method: str | None = None,
    n_burnin: int | None = None,
    n_thinning: int | None = None,
    frac_random: float | None = None,
    num_inner_restarts: int | None = None,
    raw_inner_samples: int | None = None,
    one_shot_eta: float | None = None,
    post_processing: Callable[[list[float], list[int]], Any] | None = None,
    timeout_sec: float | None = None,
    lbfgsb_aggregation: int | None = None,
    gen_candidates: str | Callable[..., Any] | None = None,
    lr: float | None = None,
    betas: tuple[float, float] | None = None,
    eps: float | None = None,
    weight_decay: float | None = None,
    amsgrad: bool | None = None,
    maximize: bool | None = None,
    stop_maxiter: int | None = None,
    stop_minimize: bool | None = None,
    n_window: int | None = None,
    stop_eta: float | None = None,
    rel_tol: float | None = None,
    topn: bool = False,
    largest: bool = True,
    sorted: bool = True,
    sample_around_best: Array | None = None,
    sample_around_best_sigma: float | None = None,
    sample_around_best_subset_sigma: float | None = None,
    sample_around_best_prob_perturb: float | None = None,
) -> tuple[ArrayOut, list[float], ArrayOut, list[float], list[tuple[str, str]]]:
    """Returns ``(candidates, acq_values, restart_candidates, restart_values,
    warnings)``. ``post_processing(data, shape)`` returns the processed
    ``(data, shape)``; ``gen_candidates`` is ``"scipy"``, ``"torch"`` (Adam
    with the ``lr`` ... ``rel_tol`` settings) or
    ``generate(data, shape, timeout_sec, fixed_features)`` returning
    ``(candidates, acq_values)``."""

def gen_batch_initial_conditions(
    acq_function: AcquisitionFunction,
    bounds: Array,
    q: int,
    num_restarts: int,
    raw_samples: int,
    *,
    fixed_features: dict[int, float] | None = None,
    seed: int | None = None,
    eta: float | None = None,
    alpha: float | None = None,
    nonnegative: bool = False,
    batch_limit: int | None = None,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    n_burnin: int | None = None,
    n_thinning: int | None = None,
    fixed_x_fantasies: Array | None = None,
    topn: bool = False,
    largest: bool = True,
    sorted: bool = True,
    sample_around_best: Array | None = None,
    sample_around_best_sigma: float | None = None,
    sample_around_best_subset_sigma: float | None = None,
    sample_around_best_prob_perturb: float | None = None,
) -> tuple[ArrayOut, bool]:
    """Returns ``(initial_conditions, selected_randomly)``."""

def sample_q_batches_from_polytope(
    n: int,
    q: int,
    bounds: Array,
    n_burnin: int,
    n_thinning: int,
    seed: int | None = None,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
) -> ArrayBuf: ...
def evaluate_feasibility(
    x: Array,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    nonlinear_inequality_constraints: Sequence[NonlinearConstraint] | None = None,
    tolerance: float | None = None,
) -> BoolBuf: ...
def project_linear_equalities(x: Array, equality: Sequence[LinearConstraint]) -> ArrayBuf: ...
def equality_projection_vjp(
    x: Array, equality: Sequence[LinearConstraint], output_adjoint: Array
) -> tuple[ArrayBuf, list[list[float]], list[float]]: ...
def project_feasible_space(
    x: Array,
    bounds: Array,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    nonlinear_inequality_constraints: Sequence[NonlinearConstraint] | None = None,
    fixed_features: dict[int, Sequence[float]] | None = None,
    tolerance: float = 1e-8,
) -> ArrayBuf: ...
def get_polytope_samples(
    n: int,
    bounds: Array,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    seed: int | None = None,
    n_burnin: int = 10000,
    n_thinning: int = 32,
) -> ArrayBuf: ...
def sparse_to_dense_constraints(
    d: int, constraints: Sequence[LinearConstraint]
) -> tuple[ArrayOut, ArrayOut]: ...
def find_interior_point(
    a: Array, b: Array, a_eq: Array | None = None, b_eq: Array | None = None
) -> list[float]: ...
def sample_polytope(
    a: Array,
    b: Array,
    x0: Array,
    n: int = 10000,
    n0: int = 100,
    n_thinning: int = 1,
    seed: int | None = None,
) -> ArrayBuf: ...

class HitAndRunPolytopeSampler:
    def __init__(
        self,
        inequality_constraints: tuple[Array, Array] | None = None,
        equality_constraints: tuple[Array, Array] | None = None,
        bounds: Array | None = None,
        interior_point: Array | None = None,
        n_burnin: int = 200,
        n_thinning: int = 20,
        seed: int | None = None,
    ) -> None: ...
    def draw(self, n: int = 1) -> ArrayBuf: ...

def draw_sobol_samples(bounds: Array, n: int, q: int, seed: int | None = None) -> ArrayBuf: ...
def draw_sobol_normal_samples(d: int, n: int, seed: int | None = None) -> ArrayBuf: ...
def feature_matmul(a: Array, b: Array) -> ArrayBuf: ...
def pathwise_cholesky_solve(scale_tril: Array, errors: Array) -> ArrayBuf: ...
def feature_frequencies(
    kind: str, num_inputs: int, num_features: int, seed: int | None = None
) -> ArrayBuf: ...

class SyntheticTestFunction:
    def __init__(
        self,
        name: str,
        dim: int | None = None,
        m: int | None = None,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: Array | None = None,
        seed: int | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def dim(self) -> int: ...
    @property
    def m(self) -> int | None: ...
    @property
    def bounds(self) -> ArrayOut: ...
    @property
    def negate(self) -> bool: ...
    @property
    def noise_std(self) -> float | None: ...
    @property
    def optimal_value(self) -> float | None: ...
    @property
    def optimizers(self) -> ArrayOut | None: ...
    @property
    def is_minimization_problem(self) -> bool: ...
    def evaluate_true(self, x: Array) -> ArrayBuf: ...
    def forward(self, x: Array, noise: bool = True) -> ArrayBuf: ...

def test_function_names() -> list[str]: ...

# Multi-objective utilities (botorch.utils.multi_objective) and the simplex /
# hypersphere samplers of botorch.utils.sampling.

def is_non_dominated(y: Array, maximize: bool = True, deduplicate: bool = True) -> BoolBuf: ...
def hypervolume(ref_point: Sequence[float], pareto_y: Array) -> float: ...
def infer_reference_point(
    pareto_y: Array,
    max_ref_point: Sequence[float] | None = None,
    scale: float = 0.1,
    scale_max_ref_point: bool = False,
) -> list[float]: ...
def get_hypervolume_maximizing_subset(
    n: int, y: Array, ref_point: Sequence[float], choose: Callable[[list[int]], int]
) -> tuple[ArrayOut, list[int]]: ...

class BoxDecomposition:
    def __init__(
        self, kind: str, ref_point: Array, y: Array | None = None, alpha: float = 0.0
    ) -> None: ...
    def update(self, y: Array) -> None: ...
    def reset(self) -> None: ...
    def partition_space(self) -> None: ...
    def compute_hypervolume(self) -> ArrayBuf | None: ...
    def get_hypercell_bounds(self) -> ArrayBuf | None: ...
    def pareto_y(self) -> ArrayBuf | None: ...
    def y(self) -> ArrayBuf | None: ...
    @property
    def has_outcomes(self) -> bool: ...
    @property
    def ref_point(self) -> ArrayOut: ...
    @property
    def num_outcomes(self) -> int: ...
    @property
    def batch_shape(self) -> list[int] | None: ...
    @property
    def class_name(self) -> str: ...
    def local_upper_bounds(self) -> tuple[ArrayBuf, ArrayBuf] | None: ...

def expand_ref_point(ref_point: Array, batch_shape: Sequence[int]) -> ArrayOut: ...
def pad_batch_pareto_frontier(
    y: Array,
    ref_point: Array,
    is_pareto: bool = False,
    feasibility_mask: Sequence[bool] | None = None,
) -> ArrayBuf: ...
def compute_local_upper_bounds(
    u: Array, z: Array, point: Sequence[float]
) -> tuple[ArrayBuf, ArrayBuf]: ...
def update_local_upper_bounds_incremental(
    new_pareto_y: Array, u: Array, z: Array
) -> tuple[ArrayBuf, ArrayBuf]: ...
def get_partition_bounds(z: Array, u: Array, ref_point: Sequence[float]) -> ArrayBuf: ...
def compute_non_dominated_hypercell_bounds_2d(
    pareto_y_sorted: Array, ref_point: Array
) -> ArrayBuf: ...
def compute_dominated_hypercell_bounds_2d(pareto_y_sorted: Array, ref_point: Array) -> ArrayBuf: ...

class ChebyshevScalarization:
    def __init__(self, weights: Sequence[float], y: Array, alpha: float) -> None: ...
    def evaluate(self, y: Array) -> ArrayBuf: ...
    def evaluate_with_grad(self, y: Array) -> tuple[ArrayBuf, ArrayBuf]: ...
    @property
    def weights(self) -> list[float]: ...
    @property
    def alpha(self) -> float: ...
    @property
    def bounds(self) -> ArrayOut | None: ...

def sample_simplex(d: int, n: int = 1, qmc: bool = False, seed: int | None = None) -> ArrayBuf: ...
def sample_hypersphere(
    d: int, n: int = 1, qmc: bool = False, seed: int | None = None
) -> ArrayBuf: ...
def simplex_from_uniform(u: Array) -> ArrayBuf: ...
def hypersphere_from_normal(z: Array) -> ArrayBuf: ...

class MultiObjectiveTestProblem:
    def __init__(
        self,
        name: str,
        dim: int | None = None,
        num_objectives: int | None = None,
        noise_std: float | Sequence[float] | None = None,
        constraint_noise_std: float | Sequence[float] | None = None,
        negate: bool = False,
        seed: int | None = None,
        bounds: Array | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_objectives(self) -> int: ...
    @property
    def num_constraints(self) -> int: ...
    @property
    def bounds(self) -> ArrayOut: ...
    @property
    def negate(self) -> bool: ...
    @property
    def ref_point(self) -> list[float]: ...
    @property
    def raw_ref_point(self) -> list[float]: ...
    @property
    def max_hv(self) -> float | None: ...
    @property
    def has_pareto_front(self) -> bool: ...
    @property
    def deviation(self) -> str | None: ...
    def evaluate_true(self, x: Array) -> ArrayBuf: ...
    def forward(self, x: Array, noise: bool = True) -> ArrayBuf: ...
    def evaluate_slack_true(self, x: Array) -> ArrayBuf: ...
    def evaluate_slack(self, x: Array, noise: bool = True) -> ArrayBuf: ...
    def is_feasible(self, x: Array, noise: bool = True) -> BoolBuf: ...
    def gen_pareto_front(self, n: int, seed: int | None = None) -> ArrayBuf: ...

def multi_objective_problem_names() -> list[str]: ...

# -- multi-objective acquisition functions -------------------------------------

#: A multi-output objective: ``("identity", outcomes)``, ``("weighted",
#: weights, outcomes)``, ``("callable", f)`` with ``f(samples, X=X)`` on torch
#: tensors, or ``("native", MultiOutputObjective)``.
MultiOutputObjectiveSpec = tuple[Any, ...]

class MultiOutputObjective:
    @staticmethod
    def feasibility_weighted(
        model: Model,
        x_baseline: Array,
        constraint_idcs: Sequence[int],
        objective: MultiOutputObjectiveSpec | None = None,
    ) -> MultiOutputObjective: ...
    @property
    def infeasible_cost(self) -> list[float]: ...
    @property
    def objective_idcs(self) -> list[int]: ...
    @property
    def constraint_idcs(self) -> list[int]: ...

class NehviState:
    def x_baseline(self) -> ArrayBuf: ...
    def x_baseline_and_pending(self) -> ArrayBuf: ...
    def cell_bounds(self) -> ArrayBuf: ...
    def prev_nehvi(self) -> float: ...
    def hypervolumes(self) -> ArrayBuf: ...
    def baseline_hvs(self) -> ArrayBuf | None: ...
    def base_samples(self) -> ArrayBuf | None: ...

def q_expected_hypervolume_improvement(
    log: bool,
    model: Model,
    ref_point: Sequence[float],
    cells: Array,
    sampler: SobolQMCNormalSampler | None = None,
    x_pending: Array | None = None,
    objective: MultiOutputObjectiveSpec | None = None,
    constraints: Sequence[Callable[[Any], Any]] | None = None,
    eta: Sequence[float] | None = None,
    fat: bool = False,
    tau_relu: float = 1e-6,
    tau_max: float = 1e-2,
) -> AcquisitionFunction: ...
def q_noisy_expected_hypervolume_improvement(
    log: bool,
    model: Model,
    ref_point: Sequence[float],
    x_baseline: Array,
    sampler: SobolQMCNormalSampler | None = None,
    x_pending: Array | None = None,
    objective: MultiOutputObjectiveSpec | None = None,
    constraints: Sequence[Callable[[Any], Any]] | None = None,
    eta: Sequence[float] | None = None,
    fat: bool = False,
    prune_baseline: bool = False,
    prune_num_samples: int = 2048,
    prune_seed: int | None = None,
    alpha: float = 0.0,
    cache_pending: bool = True,
    max_iep: int = 0,
    incremental_nehvi: bool = True,
    cache_root: bool = True,
    tau_relu: float = 1e-6,
    tau_max: float = 1e-3,
) -> tuple[AcquisitionFunction, NehviState]: ...
def expected_hypervolume_improvement(
    model: Model, ref_point: Sequence[float], cells: Array, pareto_y: Array
) -> AcquisitionFunction: ...
def q_log_nparego(
    model: Model,
    x_baseline: Array,
    scalarization_weights: Sequence[float] | None = None,
    weights_seed: int | None = None,
    sampler: SobolQMCNormalSampler | None = None,
    objective: MultiOutputObjectiveSpec | None = None,
    constraints: Sequence[Callable[[Any], Any]] | None = None,
    eta: Sequence[float] | None = None,
    x_pending: Array | None = None,
    fat: bool = True,
    prune_baseline: bool = False,
    cache_root: bool = True,
    prune_num_samples: int = 2048,
    prune_seed: int | None = None,
    tau_relu: float = 1e-6,
    tau_max: float = 1e-2,
    incremental: bool = True,
    y_baseline: Array | None = None,
) -> tuple[AcquisitionFunction, list[float], ArrayBuf]: ...
def prune_inferior_points_multi_objective(
    model: Model,
    x: Array,
    ref_point: Sequence[float],
    objective: MultiOutputObjectiveSpec | None = None,
    constraints: Sequence[Callable[[Any], Any]] | None = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    seed: int | None = None,
    base_samples: Array | None = None,
) -> ArrayBuf: ...

# -- robust and multi-fidelity multi-objective optimization ----------------------

class InputPerturbation:
    def __init__(
        self,
        perturbation_set: Array | None = None,
        perturbation_function: Callable[[Any], Any] | None = None,
        bounds: Array | None = None,
        indices: Sequence[int] | None = None,
        multiplicative: bool = False,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None: ...
    @property
    def n_w(self) -> int | None: ...
    @property
    def perturbation_set(self) -> ArrayOut | None: ...
    @property
    def bounds(self) -> ArrayOut | None: ...
    @property
    def indices(self) -> list[int] | None: ...
    @property
    def multiplicative(self) -> bool: ...
    @property
    def transform_on_train(self) -> bool: ...
    @property
    def transform_on_eval(self) -> bool: ...
    @property
    def transform_on_fantasize(self) -> bool: ...
    def transform(self, x: Array) -> ArrayBuf: ...

class RiskMeasure:
    @staticmethod
    def cvar(alpha: float, n_w: int, preprocessing: ObjectiveSpec | None = None) -> RiskMeasure: ...
    @staticmethod
    def var(alpha: float, n_w: int, preprocessing: ObjectiveSpec | None = None) -> RiskMeasure: ...
    @staticmethod
    def worst_case(n_w: int, preprocessing: ObjectiveSpec | None = None) -> RiskMeasure: ...
    @staticmethod
    def expectation(n_w: int, preprocessing: ObjectiveSpec | None = None) -> RiskMeasure: ...
    @staticmethod
    def mars(
        alpha: float,
        n_w: int,
        chebyshev_weights: Sequence[float],
        baseline_y: Array | None = None,
        ref_point: Sequence[float] | None = None,
        preprocessing: MultiOutputObjectiveSpec | None = None,
    ) -> RiskMeasure: ...
    def with_chebyshev_weights(self, weights: Sequence[float]) -> RiskMeasure: ...
    def with_baseline_y(self, baseline_y: Array) -> RiskMeasure: ...
    def with_baseline_from_samples(self, y_samples: Array) -> RiskMeasure: ...
    def with_baseline_from_model(self, model: Model, x_baseline: Array) -> RiskMeasure: ...
    @property
    def chebyshev_weights(self) -> list[float]: ...
    @property
    def baseline_y(self) -> ArrayOut | None: ...
    @property
    def normalization_bounds(self) -> ArrayOut | None: ...
    def forward(self, samples: Array) -> ArrayBuf: ...
    def vjp(self, samples: Array, adjoint: Array) -> ArrayBuf: ...

class MultiOutputRiskMeasure:
    @staticmethod
    def mvar(
        n_w: int,
        alpha: float,
        expectation: bool = False,
        preprocessing: MultiOutputObjectiveSpec | None = None,
        pad_to_n_w: bool = False,
        filter_dominated: bool = True,
        use_counting: bool = False,
    ) -> MultiOutputRiskMeasure: ...
    @staticmethod
    def independent_cvar(
        alpha: float, n_w: int, preprocessing: MultiOutputObjectiveSpec | None = None
    ) -> MultiOutputRiskMeasure: ...
    @staticmethod
    def independent_var(
        alpha: float, n_w: int, preprocessing: MultiOutputObjectiveSpec | None = None
    ) -> MultiOutputRiskMeasure: ...
    @staticmethod
    def worst_case(
        n_w: int, preprocessing: MultiOutputObjectiveSpec | None = None
    ) -> MultiOutputRiskMeasure: ...
    @staticmethod
    def expectation(
        n_w: int, preprocessing: MultiOutputObjectiveSpec | None = None
    ) -> MultiOutputRiskMeasure: ...
    def objective(self) -> MultiOutputObjective: ...
    def mvar_set(self, y: Array, use_counting: bool) -> list[ArrayOut]: ...
    def forward(self, samples: Array) -> ArrayBuf: ...
    def vjp(self, samples: Array, adjoint: Array) -> ArrayBuf: ...

def mvar_make_differentiable(
    prepared_samples: Array, mvars: Array, adjoint: Array | None = None
) -> ArrayBuf: ...
def mars_y_normalization_bounds(y: Array, ref_point: Sequence[float] | None = None) -> ArrayOut: ...
def momf(
    model: Model,
    ref_point: Sequence[float],
    cells: Array,
    sampler: SobolQMCNormalSampler | None = None,
    x_pending: Array | None = None,
    objective: MultiOutputObjectiveSpec | None = None,
    constraints: Sequence[Callable[[Any], Any]] | None = None,
    eta: Sequence[float] | None = None,
    cost_call: Callable[[Any], Any] | None = None,
) -> AcquisitionFunction: ...

class MultiOutputAcquisitionFunction:
    @staticmethod
    def posterior_mean(
        model: Model, weights: Sequence[float] | None = None
    ) -> MultiOutputAcquisitionFunction: ...
    @staticmethod
    def wrapper(acqfs: Sequence[AcquisitionFunction]) -> MultiOutputAcquisitionFunction: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def dim(self) -> int: ...
    def forward(self, x: Array) -> ArrayBuf: ...

def optimize_with_nsgaii(
    bounds: Array,
    num_objectives: int,
    acqf: MultiOutputAcquisitionFunction | None = None,
    evaluate: Callable[[Array], tuple[Array, Array | None]] | None = None,
    q: int | None = None,
    ref_point: Sequence[float] | None = None,
    inequality_constraints: Sequence[tuple[Sequence[int], Sequence[float], float]] = ...,
    population_size: int = 250,
    max_gen: int | None = None,
    seed: int | None = None,
    fixed_features: Sequence[tuple[int, float]] = ...,
    max_attempts: int = 2,
    discrete_choices: Sequence[tuple[int, Sequence[float]]] = ...,
) -> tuple[ArrayBuf, ArrayBuf, list[str], int]: ...
def nsga2_non_dominated_fronts(
    f: Array, n_stop_if_ranked: int | None = None
) -> list[list[int]]: ...
def nsga2_crowding_distance(f: Array) -> list[float]: ...

# Fantasy models, cost models and the hypervolume knowledge gradient.

#: An evaluation mask ``(data, shape)`` (``n x m`` booleans).
Mask = tuple[Sequence[bool], Sequence[int]]
#: A projection: ``None`` (identity), ``("target", [(column, value), ...])``
#: or ``("callable", f)`` with ``f(X)`` on torch tensors.
ProjectionSpec = tuple[Any, ...]

def condition_on_observations(
    model: SingleTaskGP, x: Array, y: Array, noise: Array | None = None
) -> SingleTaskGP: ...
def fantasize(
    model: SingleTaskGP,
    x: Array,
    sampler: SobolQMCNormalSampler,
    observation_noise: Array | None = None,
) -> SingleTaskGP: ...
def model_list_condition_on_observations(
    models: Sequence[SingleTaskGP], xs: Sequence[Array], y: Array, noise: Array | None = None
) -> list[SingleTaskGP]: ...
def model_list_fantasize(
    models: Sequence[SingleTaskGP],
    x: Array,
    samplers: Sequence[SobolQMCNormalSampler],
    shared: bool,
    observation_noise: Array | None = None,
    evaluation_mask: Mask | None = None,
) -> list[SingleTaskGP]: ...

class CostModel:
    @staticmethod
    def affine(
        fidelity_weights: Sequence[tuple[int, float]] | None = None, fixed_cost: float = 0.01
    ) -> CostModel: ...
    @staticmethod
    def fixed(fixed_cost: Sequence[float]) -> CostModel: ...
    @staticmethod
    def posterior_mean(model: Model) -> CostModel: ...
    @staticmethod
    def callable(f: Callable[..., Any], num_outputs: int) -> CostModel: ...
    @property
    def num_outputs(self) -> int: ...
    def forward(self, x: Array) -> ArrayBuf: ...
    def backward(self, x: Array, cost_bar: Array) -> ArrayBuf: ...

class CostAwareUtility:
    @staticmethod
    def inverse_cost_weighted(
        cost_model: CostModel,
        use_mean: bool = True,
        cost_objective: Callable[..., Any] | None = None,
        log: bool = False,
    ) -> CostAwareUtility: ...
    @staticmethod
    def generic(cost: Callable[..., Any]) -> CostAwareUtility: ...
    @property
    def log(self) -> bool | None: ...
    def forward(
        self,
        x: Array,
        deltas: Array,
        evaluation_mask: Mask | None = None,
        adjoint: Array | None = None,
    ) -> tuple[ArrayBuf, tuple[ArrayBuf, ArrayBuf] | None]: ...

class HvkgState:
    def set_x_pending(self, x_pending: Array | None = None, mask: Mask | None = None) -> None: ...
    def x_pending_evaluation_mask(self) -> BoolBuf | None: ...
    def construct_evaluation_mask(self, q: int) -> BoolBuf | None: ...
    def gen_one_shot_initial_conditions(
        self,
        bounds: Array,
        q: int,
        num_restarts: int,
        raw_samples: int,
        *,
        fixed_features: dict[int, float] | None = None,
        inequality_constraints: Sequence[LinearConstraint] | None = None,
        equality_constraints: Sequence[LinearConstraint] | None = None,
        seed: int | None = None,
        eta: float | None = None,
        alpha: float | None = None,
        batch_limit: int | None = None,
        init_batch_limit: int | None = None,
        maxiter: int | None = None,
        frac_random: float | None = None,
        num_inner_restarts: int | None = None,
        raw_inner_samples: int | None = None,
        one_shot_eta: float | None = None,
        nonnegative: bool = False,
        n_burnin: int | None = None,
        n_thinning: int | None = None,
        topn: bool = False,
        largest: bool = True,
        sorted: bool = True,
        sample_around_best: Array | None = None,
        sample_around_best_sigma: float | None = None,
        sample_around_best_subset_sigma: float | None = None,
        sample_around_best_prob_perturb: float | None = None,
    ) -> tuple[ArrayBuf, list[str]]: ...

def q_hypervolume_knowledge_gradient(
    model: Model,
    ref_point: Sequence[float],
    num_fantasies: int = 8,
    num_pareto: int = 10,
    samplers: Sequence[SobolQMCNormalSampler] | None = None,
    objective: MultiOutputObjectiveSpec | None = None,
    inner_sampler: SobolQMCNormalSampler | None = None,
    x_evaluation_mask: Mask | None = None,
    x_pending: Array | None = None,
    x_pending_evaluation_mask: Mask | None = None,
    current_value: float | None = None,
    use_posterior_mean: bool = True,
    cost_aware_utility: CostAwareUtility | None = None,
    log: bool = False,
    target_fidelities: Sequence[tuple[int, float]] | None = None,
    project: ProjectionSpec | None = None,
) -> tuple[AcquisitionFunction, HvkgState]: ...
def hv_value_function(
    model: Model,
    ref_point: Sequence[float],
    objective: MultiOutputObjectiveSpec | None = None,
    sampler: SobolQMCNormalSampler | None = None,
    use_posterior_mean: bool = False,
    log: bool = False,
    project: ProjectionSpec | None = None,
) -> AcquisitionFunction: ...
def project_to_target_fidelity(
    x: Array, target_fidelities: Sequence[tuple[int, float]], d: int
) -> ArrayBuf: ...
def expand_trace_observations(
    x: Array, fidelity_dims: Sequence[int] | None = None, num_trace_obs: int = 0
) -> ArrayBuf: ...

# Pathwise sampling (``botorch.sampling.pathwise``).

PathDraws = tuple[Array, Array, Array]

class SamplePaths:
    @property
    def num_samples(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def has_components(self) -> bool: ...
    def forward(self, x: Array, component: str | None = None) -> ArrayBuf: ...
    def forward_with_grad(
        self, x: Array, component: str | None = None
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def draws(self) -> list[tuple[ArrayBuf, ArrayBuf, ArrayBuf]]: ...
    def update_weights(self) -> list[ArrayBuf]: ...
    def noise(self) -> list[ArrayBuf]: ...

def draw_matheron_paths(
    model: Model,
    num_samples: int,
    num_features: int = 1024,
    seed: int | None = None,
    draws: Sequence[PathDraws] | None = None,
) -> SamplePaths: ...
def draw_kernel_feature_paths(
    model: Model,
    num_samples: int,
    num_features: int = 1024,
    seed: int | None = None,
    draws: Sequence[tuple[Array, Array]] | None = None,
) -> SamplePaths: ...
def gaussian_update(
    model: Model,
    sample_values: Sequence[Array],
    noise: Sequence[Array] | None = None,
    seed: int | None = None,
) -> SamplePaths: ...

# Information-theoretic multi-objective acquisition functions.

class PesState:
    @property
    def ep_iterations(self) -> int: ...
    def ep_damping(self) -> ArrayBuf: ...
    def cached_factors(self) -> tuple[ArrayBuf, ArrayBuf]: ...

def compute_sample_box_decomposition(
    pareto_fronts: Array,
    partitioning: str = "dominated",
    alpha: float = 0.0,
    maximize: bool = True,
    num_constraints: int = 0,
) -> ArrayBuf: ...
def random_search_optimizer(
    model: SamplePaths | Callable[[Array], Array],
    bounds: Array,
    num_points: int,
    maximize: bool,
    pop_size: int = 1024,
    max_tries: int = 10,
    seed: int | None = None,
    candidates: Sequence[Array] | None = None,
    sample: int = 0,
) -> tuple[ArrayBuf, ArrayBuf]: ...
def sample_optimal_points(
    model: Model,
    bounds: Array,
    num_samples: int,
    num_points: int,
    maximize: bool = True,
    pop_size: int = 1024,
    max_tries: int = 10,
    num_features: int = 1024,
    seed: int | None = None,
    draws: Sequence[Sequence[PathDraws]] | None = None,
    candidates: Sequence[Sequence[Array]] | None = None,
) -> tuple[ArrayBuf, ArrayBuf]: ...
def sample_optimal_point_paths(
    model: Model,
    num_samples: int,
    num_features: int = 1024,
    seed: int | None = None,
    draws: Sequence[Sequence[PathDraws]] | None = None,
) -> list[SamplePaths]: ...
def q_lower_bound_mo_max_value_entropy_search(
    model: Model,
    hypercell_bounds: Array,
    x_pending: Array | None = None,
    estimation_type: str = "LB",
    sampler: SobolQMCNormalSampler | None = None,
) -> AcquisitionFunction: ...
def q_lower_bound_mo_joint_entropy_search(
    model: Model,
    pareto_sets: Array,
    pareto_fronts: Array,
    hypercell_bounds: Array,
    x_pending: Array | None = None,
    estimation_type: str = "LB",
    sampler: SobolQMCNormalSampler | None = None,
) -> AcquisitionFunction: ...
def q_mo_predictive_entropy_search(
    model: Model,
    pareto_sets: Array,
    maximize: bool = True,
    x_pending: Array | None = None,
    max_ep_iterations: int = 250,
    ep_jitter: float = 1e-4,
    test_jitter: float = 1e-4,
    threshold: float = 1e-2,
) -> tuple[AcquisitionFunction, PesState]: ...
def compute_entropy(
    estimation_type: str,
    hypercell_bounds: Array,
    mean: Array,
    variance: Array,
    observation_noise: Array,
    samples: Array | None = None,
    samples_log_prob: Array | None = None,
) -> ArrayBuf: ...
def log_cdf_robust(x: Array) -> ArrayBuf: ...

# Deterministic models, composite and callable acquisition functions.

def posterior_mean_model(model: Model) -> Model: ...
def generic_deterministic_model(callable: Any, dim: int, num_outputs: int) -> Model: ...
def weighted_log_mean_exp(
    terms: Sequence[AcquisitionFunction], weights: Sequence[float]
) -> AcquisitionFunction: ...
def callable_acquisition_function(callable: Any, dim: int) -> AcquisitionFunction: ...

# Single-objective acquisition functions (phase 8).

Draws = tuple[Array, Array, Array]

class KnowledgeGradientState:
    def gen_one_shot_initial_conditions(
        self,
        bounds: Array,
        q: int,
        num_restarts: int,
        raw_samples: int,
        *,
        fixed_features: dict[int, float] | None = None,
        inequality_constraints: Sequence[Any] | None = None,
        equality_constraints: Sequence[Any] | None = None,
        seed: int | None = None,
        eta: float | None = None,
        alpha: float | None = None,
        batch_limit: int | None = None,
        init_batch_limit: int | None = None,
        frac_random: float | None = None,
        num_inner_restarts: int | None = None,
        raw_inner_samples: int | None = None,
        one_shot_eta: float | None = None,
        nonnegative: bool = False,
        n_burnin: int | None = None,
        n_thinning: int | None = None,
        topn: bool = False,
        largest: bool = True,
        sorted: bool = True,
        sample_around_best: Array | None = None,
        sample_around_best_sigma: float | None = None,
        sample_around_best_subset_sigma: float | None = None,
        sample_around_best_prob_perturb: float | None = None,
    ) -> tuple[ArrayOut, list[str]]: ...
    def evaluate(
        self,
        x: Array,
        bounds: Array,
        num_restarts: int = 20,
        raw_samples: int = 1024,
        *,
        frac_random: float | None = None,
        num_inner_restarts: int | None = None,
        raw_inner_samples: int | None = None,
        eta: float | None = None,
        seed: int | None = None,
        maxiter: int | None = None,
        maxfun: int | None = None,
        ftol: float | None = None,
        gtol: float | None = None,
    ) -> tuple[list[float], list[int]]: ...
    def current_value_function(self) -> AcquisitionFunction: ...
    @property
    def num_fantasies(self) -> int: ...
    @property
    def value_function_name(self) -> str: ...
    @property
    def inner_sampler_seed(self) -> int | None: ...

def q_knowledge_gradient(
    model: Model,
    num_fantasies: int | None = 64,
    sampler: SobolQMCNormalSampler | None = None,
    objective: Any = None,
    transform: tuple[Sequence[float], float] | None = None,
    inner_sampler: SobolQMCNormalSampler | None = None,
    x_pending: Array | None = None,
    current_value: float | None = None,
    multi_fidelity: bool = False,
    cost_aware_utility: CostAwareUtility | None = None,
    project: Any = None,
    expand: Any = None,
    value_function: tuple[str, dict[str, Any]] | None = None,
) -> tuple[AcquisitionFunction, KnowledgeGradientState]: ...
def projected_acquisition_function(
    base: AcquisitionFunction, project: Any
) -> AcquisitionFunction: ...

class NoisyExpectedImprovementState:
    def x_baseline(self) -> ArrayOut: ...
    def baseline_best_f(self) -> list[float] | None: ...
    def lower_bound_points(self) -> ArrayOut: ...
    def infeasible_lower_bound(self) -> float | None: ...
    @property
    def cache_root(self) -> bool: ...

def q_noisy_expected_improvement(
    model: Model,
    x_baseline: Array,
    sampler: SobolQMCNormalSampler | None = None,
    x_pending: Array | None = None,
    objective: Any = None,
    constraints: Sequence[Callable[[Any], Any]] | None = None,
    eta: Sequence[float] | None = None,
    prune_baseline: bool = True,
    prune_num_samples: int = 2048,
    prune_max_frac: float = 1.0,
    prune_seed: int | None = None,
    cache_root: bool = True,
    lower_bound_points: Array | None = None,
    objective_batch_shape: Sequence[int] | None = None,
) -> tuple[AcquisitionFunction, NoisyExpectedImprovementState]: ...
def log_probability_of_feasibility(
    model: Model, constraints: Sequence[tuple[int, float | None, float | None]]
) -> AcquisitionFunction: ...
def scalarized_posterior_mean(model: Model, weights: Sequence[float]) -> AcquisitionFunction: ...
def q_analytic_probability_of_improvement(
    model: Model, best_f: float, maximize: bool = True
) -> AcquisitionFunction: ...
def noisy_expected_improvement(
    model: Model,
    x_observed: Array,
    num_fantasies: int = 20,
    maximize: bool = True,
    sampler: SobolQMCNormalSampler | None = None,
    log: bool = False,
    transform: tuple[Sequence[float], float] | None = None,
) -> tuple[AcquisitionFunction, list[float]]: ...

class MaxValueEntropyState:
    def posterior_max_values(self) -> list[float]: ...
    def set_posterior_max_values(self, values: Sequence[float]) -> None: ...
    def candidate_set(self) -> ArrayOut: ...

def q_max_value_entropy(
    model: Model,
    candidate_set: Array,
    kind: str,
    num_fantasies: int = 16,
    num_mv_samples: int = 10,
    num_y_samples: int = 128,
    transform: tuple[Sequence[float], float] | None = None,
    use_gumbel: bool = True,
    maximize: bool = True,
    x_pending: Array | None = None,
    train_inputs: Array | None = None,
    sampler: SobolQMCNormalSampler | None = None,
    fantasies_sampler: SobolQMCNormalSampler | None = None,
    seed: int | None = None,
    cost_aware_utility: CostAwareUtility | None = None,
    project: Any = None,
    expand: Any = None,
) -> tuple[AcquisitionFunction, MaxValueEntropyState]: ...
def sample_max_value_gumbel(
    model: Model,
    candidate_set: Array,
    uniforms: Sequence[float],
    maximize: bool = True,
    transform: tuple[Sequence[float], float] | None = None,
) -> list[float]: ...
def sample_max_value_thompson(
    model: Model,
    candidate_set: Array,
    base_samples: Array,
    maximize: bool = True,
    transform: tuple[Sequence[float], float] | None = None,
) -> list[float]: ...
def q_joint_entropy_search(
    model: Model,
    optimal_inputs: Array,
    optimal_outputs: Array,
    condition_noiseless: bool = True,
    condition_noise: Sequence[float] | None = None,
    transform: tuple[Sequence[float], float] | None = None,
    x_pending: Array | None = None,
    estimation_type: str = "LB",
    sampler: SobolQMCNormalSampler | None = None,
) -> AcquisitionFunction: ...
def get_optimal_samples(
    model: Model,
    bounds: Array,
    num_optima: int,
    raw_samples: int = 1024,
    num_restarts: int = 20,
    transform: tuple[Sequence[float], float] | None = None,
    objective: Any = None,
    return_transformed: bool = False,
    num_features: int = 1024,
    seed: int | None = None,
    draws: Sequence[Draws] | None = None,
    candidates: Array | None = None,
    maxiter: int | None = None,
) -> tuple[ArrayOut, ArrayOut]: ...

class ThompsonSamplingState:
    def redraw(self, batch_size: int, draws: Sequence[Draws] | None = None) -> None: ...
    @property
    def batch_size(self) -> int | None: ...

def pathwise_thompson_sampling(
    model: Model,
    objective: Any = None,
    transform: tuple[Sequence[float], float] | None = None,
    num_features: int = 1024,
    seed: int | None = None,
) -> tuple[AcquisitionFunction, ThompsonSamplingState]: ...
def q_neg_integrated_posterior_variance(
    model: Model,
    mc_points: Array,
    transform: tuple[Sequence[float], float] | None = None,
    x_pending: Array | None = None,
) -> AcquisitionFunction: ...
def penalized_acquisition_function(
    raw_acqf: AcquisitionFunction, penalty: Any, regularization_parameter: float
) -> AcquisitionFunction: ...
def proximal_acquisition_function(
    acq_function: AcquisitionFunction,
    model: Model,
    proximal_weights: Sequence[float],
    transformed_weighting: bool = True,
    beta: float | None = None,
) -> AcquisitionFunction: ...
def fixed_feature_acquisition_function(
    acq_function: AcquisitionFunction, d: int, columns: Sequence[int], values: Array
) -> AcquisitionFunction: ...
def prior_guided_acquisition_function(
    acq_function: AcquisitionFunction,
    prior_module: Callable[[Any], Any],
    log: bool = False,
    prior_exponent: float = 1.0,
    x_pending: Array | None = None,
) -> AcquisitionFunction: ...

# Phase 9: optimization and candidate generation.

#: ``(maxiter, minimize, n_window, eta, rel_tol)`` of an
#: ``ExpMAStoppingCriterion``.
StoppingSpec = tuple[int, bool, int, float, float]
#: ``(step, fval, status, runtime, message)`` of an ``OptimizationResult``
#: (``status``: 1 running, 2 success, 3 failure, 4 stopped).
ResultOut = tuple[int, float, int, float | None, str | None]

def optimize_acqf_mixed_py(
    acq_function: AcquisitionFunction,
    bounds: Array,
    q: int,
    num_restarts: int,
    fixed_features_list: Sequence[dict[int, float]],
    raw_samples: int | None = None,
    return_best_only: bool = True,
    **kwargs: Any,
) -> tuple[ArrayOut, float, ArrayOut, list[float], list[str]]:
    """Returns ``(candidates, value, restart_candidates, restart_values,
    warnings)``; ``kwargs`` are those of ``optimize_acqf``."""

def optimize_acqf_discrete_py(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Array,
    unique: bool = True,
    x_avoid: Array | None = None,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
) -> tuple[ArrayOut, list[float], list[str]]: ...
def optimize_acqf_discrete_local_search_py(
    acq_function: AcquisitionFunction,
    discrete_choices: Sequence[Sequence[float]],
    q: int,
    num_restarts: int = 20,
    raw_samples: int = 4096,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    x_avoid: Array | None = None,
    batch_initial_conditions: Array | None = None,
    max_tries: int = 100,
    unique: bool = True,
    return_acq_values: bool = True,
    lattice_points: Callable[..., Any] | None = None,
    seed: int | None = None,
) -> tuple[ArrayOut, list[float]]: ...
def gen_lattice_points_py(
    discrete_choices: Sequence[Sequence[float]],
    raw_samples: int,
    x_avoid: Array,
    inequality_constraints: Sequence[LinearConstraint] | None,
    min_points: int,
    max_tries: int = 100,
    seed: int | None = None,
) -> ArrayOut: ...
def optimize_acqf_cyclic_py(
    acq_function: AcquisitionFunction,
    bounds: Array,
    q: int,
    num_restarts: int,
    raw_samples: int | None,
    cyclic: StoppingSpec,
    **kwargs: Any,
) -> tuple[ArrayOut, list[float], list[str]]: ...

class MixedProblem:
    """The problem of ``optimize_acqf_mixed_alternating`` after BoTorch's
    validation and relaxation of large discrete dimensions."""

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        bounds: Array,
        discrete_dims: Mapping[int, Sequence[float]],
        cat_dims: Mapping[int, Sequence[float]],
        num_restarts: int,
        raw_samples: int,
        fixed_features: dict[int, float] | None = None,
        inequality_constraints: Sequence[LinearConstraint] | None = None,
        equality_constraints: Sequence[LinearConstraint] | None = None,
        post_processing: Callable[[list[float], list[int]], Any] | None = None,
        retry_on_optimization_warning: bool = True,
        x_baseline: Array | None = None,
        **kwargs: Any,
    ) -> None: ...
    @property
    def discrete_dims(self) -> dict[int, list[float]]: ...
    @property
    def cat_dims(self) -> dict[int, list[float]]: ...
    @property
    def cont_dims(self) -> list[int]: ...
    @property
    def relaxed_dims(self) -> dict[int, list[float]]: ...
    def generate_starting_points(
        self, seed: int | None = None
    ) -> tuple[ArrayOut, list[float], list[str]]: ...
    def discrete_step(self, x: Array, seed: int | None = None) -> tuple[ArrayOut, list[float]]: ...
    def continuous_step(self, x: Array) -> tuple[ArrayOut, list[float], list[str]]: ...
    def sample_feasible_points(self, num_points: int, seed: int | None = None) -> ArrayOut: ...
    def post_process(self, x: Array) -> ArrayOut: ...
    def optimize(
        self,
        q: int,
        seed: int | None = None,
        starting_points: Callable[..., Any] | None = None,
        return_acq_values: bool = True,
    ) -> tuple[ArrayOut, float, list[str]]: ...

def round_discrete_dims_py(x: Array, discrete_dims: Mapping[int, Sequence[float]]) -> ArrayOut: ...
def get_nearest_neighbors_py(
    current_x: Sequence[float], bounds: Array, discrete_dims: Mapping[int, Sequence[float]]
) -> list[list[float]]: ...
def get_categorical_neighbors_py(
    current_x: Sequence[float],
    cat_dims: Mapping[int, Sequence[float]],
    max_num_cat_values: int = 20,
    seed: int | None = None,
) -> list[list[float]]: ...
def get_spray_points_py(
    x_baseline: Array,
    cont_dims: Sequence[int],
    discrete_dims: Mapping[int, Sequence[float]],
    cat_dims: Mapping[int, Sequence[float]],
    bounds: Array,
    num_spray_points: int,
    std_cont_perturbation: float = 0.1,
    seed: int | None = None,
) -> ArrayOut: ...
def prune_candidates_py(
    candidates: Array, acq_values: Sequence[float], prune_tolerance: float
) -> ArrayOut: ...
def gen_candidates_torch_py(
    acq_function: AcquisitionFunction,
    initial_conditions: Array,
    bounds: Array,
    fixed_features: dict[int, float] | None = None,
    callback: Callable[[int, float, list[float] | array[float], list[int]], Any] | None = None,
    **kwargs: Any,
) -> tuple[ArrayOut, list[float], int]:
    """Returns ``(candidates, acq_values, steps)``."""

def gen_candidates_scipy_py(
    acq_function: AcquisitionFunction,
    initial_conditions: Array,
    bounds: Array,
    fixed_features: dict[int, float] | None = None,
    aggregation: int | None = None,
    use_slsqp: bool = False,
    timeout_sec: float | None = None,
    **kwargs: Any,
) -> tuple[ArrayOut, list[float], list[str]]: ...
def sample_truncated_normal_perturbations_py(
    x: Array,
    n_discrete_points: int,
    sigma: float,
    bounds: Array,
    qmc: bool = True,
    seed: int | None = None,
) -> ArrayOut: ...
def truncated_normal_perturbations_py(
    centers: Array, u: Array, sigma: float, bounds: Array
) -> ArrayOut: ...
def sample_perturbed_subset_dims_py(
    x: Array,
    bounds: Array,
    n_discrete_points: int,
    sigma: float = 0.1,
    qmc: bool = True,
    prob_perturb: float | None = None,
    seed: int | None = None,
) -> ArrayOut: ...
def sample_points_around_py(
    best_x: Array,
    n_discrete_points: int,
    sigma: float,
    bounds: Array,
    subset_sigma: float = 0.1,
    prob_perturb: float | None = None,
    seed: int | None = None,
) -> ArrayOut: ...
def maximize_samples_py(
    obj: Sequence[float], num_samples: int, batch: int, n: int, replacement: bool
) -> list[int]: ...
def constrained_scores_py(
    y: Sequence[float], c: Sequence[float], rows: int, n: int, k: int
) -> list[float]: ...
def boltzmann_weights_py(
    values: Sequence[float], eta: float, temp_decrease: float = 0.5
) -> tuple[list[float], float]: ...
def expma_weights_py(eta: float, n_window: int) -> list[float]: ...
def fit_gpytorch_mll_torch_py(
    model: SingleTaskGP,
    mask: Sequence[bool] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    step_limit: int | None = None,
    stopping: StoppingSpec | None = None,
    timeout_sec: float | None = None,
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    maximize: bool = False,
) -> ResultOut: ...
def fit_gpytorch_mll_scipy_py(
    model: SingleTaskGP,
    mask: Sequence[bool] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    timeout_sec: float | None = None,
    maxiter: int | None = None,
    maxfun: int | None = None,
    ftol: float | None = None,
    gtol: float | None = None,
    maxcor: int | None = None,
    maxls: int | None = None,
) -> ResultOut: ...
def sample_all_priors(model: SingleTaskGP, seed: int | None = None) -> None: ...

# -- Input and outcome transform steps (phase 6b) ----------------------------

Flags = tuple[bool, bool, bool]

class InputStep:
    @staticmethod
    def affine(
        d: int,
        indices: Sequence[int] | None,
        offset: Sequence[float],
        coefficient: Sequence[float],
        kind: str,
        learn: bool,
        reverse: bool,
        flags: Flags,
        min_range: float = 1e-8,
        center: float = 0.5,
        min_std: float = 1e-8,
    ) -> InputStep: ...
    @staticmethod
    def warp(
        d: int,
        indices: Sequence[int],
        eps: float,
        bounds: tuple[Sequence[float], Sequence[float]] | None,
        c0_prior: Prior | None,
        c1_prior: Prior | None,
        reverse: bool,
        flags: Flags,
        batched: bool = False,
        concentrations: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> InputStep: ...
    @staticmethod
    def round(
        integer_indices: Sequence[int],
        categorical: Sequence[tuple[int, int]],
        approximate: bool,
        tau: float,
        flags: Flags,
    ) -> InputStep: ...
    @staticmethod
    def log10(indices: Sequence[int], reverse: bool, flags: Flags) -> InputStep: ...
    @staticmethod
    def append(
        feature_set: Array | None,
        f: Callable[..., Any] | None,
        indices: Sequence[int] | None,
        fkwargs: Mapping[str, Any] | None,
        skip_expand: bool,
        flags: Flags,
    ) -> InputStep: ...
    @staticmethod
    def interaction(indices: Sequence[int] | None = None) -> InputStep: ...
    @staticmethod
    def filter(feature_indices: Sequence[int], flags: Flags) -> InputStep: ...
    @staticmethod
    def one_hot_to_numeric(
        dim: int, categorical: Sequence[tuple[int, int]], flags: Flags
    ) -> InputStep: ...
    @staticmethod
    def numeric_to_categorical(
        dim: int,
        categorical: Sequence[tuple[int, int]],
        encoders: Sequence[Callable[..., Any] | None],
        flags: Flags,
    ) -> InputStep: ...
    @staticmethod
    def imputation(
        feature_indices: Sequence[tuple[int, Sequence[int]]],
        d: int,
        target_task: int | None,
        bounds: tuple[Sequence[float], Sequence[float]] | None,
        flags: Flags,
        raw: Sequence[float] | None = None,
    ) -> InputStep: ...
    @property
    def name(self) -> str: ...
    @property
    def flags(self) -> Flags: ...
    @property
    def params(self) -> list[float]: ...
    @property
    def param_shapes(self) -> list[tuple[str, list[int]]]: ...
    def with_params(self, values: Sequence[float]) -> InputStep: ...
    @property
    def affine_state(self) -> tuple[list[float], list[float]] | None: ...
    @property
    def n_f(self) -> int | None: ...
    @property
    def imputation_values(self) -> list[float] | None: ...
    def frozen(self) -> InputStep: ...
    def out_dim(self, d: int) -> int: ...
    def fit(self, x: Array) -> InputStep: ...
    def transform(self, x: Array) -> ArrayBuf: ...
    def untransform(self, x: Array) -> ArrayBuf: ...

class OutcomeStep:
    @staticmethod
    def standardize(min_stdv: float) -> OutcomeStep: ...
    @staticmethod
    def standardize_fitted(mean: float, stdv: float, min_stdv: float) -> OutcomeStep: ...
    @staticmethod
    def log() -> OutcomeStep: ...
    @staticmethod
    def power(power: float) -> OutcomeStep: ...
    @staticmethod
    def bilog() -> OutcomeStep: ...
    @staticmethod
    def stratified(
        stratification_idx: int,
        task_values: Sequence[int],
        min_stdv: float,
        stats: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> OutcomeStep: ...
    @property
    def name(self) -> str: ...
    @property
    def stats(self) -> tuple[list[float], list[float]] | None: ...
    @property
    def is_trained(self) -> bool: ...
    def fit_transform(
        self, y: Sequence[float], yvar: Sequence[float] | None = None, x: Array | None = None
    ) -> tuple[OutcomeStep, list[float], list[float] | None]: ...
    def apply(
        self,
        y: Sequence[float],
        yvar: Sequence[float] | None = None,
        x: Array | None = None,
        inverse: bool = False,
    ) -> tuple[list[float], list[float] | None]: ...

def norm_to_lognorm_mean_py(mu: Array, var: Array) -> ArrayOut: ...
def norm_to_lognorm_variance_py(mu: Array, var: Array) -> ArrayOut: ...
def lognormal_moments_py(mu: Array, cov: Array, inverse: bool) -> tuple[ArrayOut, ArrayOut]: ...
def kumaraswamy_warp_py(
    x: Array, c0: Sequence[float], c1: Sequence[float], eps: float, inverse: bool
) -> ArrayOut: ...
def interaction_features_py(x: Array) -> ArrayOut: ...
def nanstd_py(x: Array, dim: int, keepdim: bool) -> ArrayOut: ...

# Phase 7b: deterministic models and ensembles.
def affine_deterministic_model(a: Array, b: Sequence[float]) -> Model: ...
def fixed_single_sample_model(model: Model, w: Sequence[float], jitter: float) -> Model: ...
def ensemble_moments_model(
    members: Model, size: int, weights: Sequence[float] | None = None
) -> Model: ...
def ensemble_average(
    members: Sequence[AcquisitionFunction], weights: Sequence[float] | None = None
) -> AcquisitionFunction: ...

# Phase 8b: the multi-step look-ahead and the posterior-sample VJPs of
# Python subclasses of SampleReducingMCAcquisitionFunction.
class MultiStepLookaheadState:
    def num_auxiliary(self) -> int: ...
    def num_fantasies(self) -> list[int]: ...
    def inner_sampler_seeds(self) -> list[int | None]: ...

def q_multi_step_lookahead(
    model: Model,
    batch_sizes: Sequence[int],
    num_fantasies: Sequence[int] | None = None,
    samplers: Sequence[SobolQMCNormalSampler] | None = None,
    sample_weights: Sequence[Sequence[float] | None] | None = None,
    value_functions: Sequence[tuple[str, dict[str, Any], bool, SobolQMCNormalSampler | None] | None]
    | None = None,
    objective: Any = None,
    transform: tuple[Sequence[float], float] | None = None,
    x_pending: Array | None = None,
    collapse_fantasy_base_samples: bool = True,
) -> tuple[AcquisitionFunction, MultiStepLookaheadState]: ...
def posterior_samples_vjp(model: Model, x: Array, base: Array, samples_bar: Array) -> ArrayBuf: ...

# Fully Bayesian GPs (phase 10a).
def fully_bayesian_kernel(kernel: str, x: Array, scale: Array) -> ArrayBuf: ...

class PyroModel:
    def __init__(
        self,
        kernel: str,
        train_x: Array,
        train_y: Sequence[float],
        train_yvar: Sequence[float] | None = None,
        outputscale_prior: tuple[float, float] | None = None,
        warp_indices: Sequence[int] | None = None,
        warp_eps: float = 1e-7,
        task: tuple[int, int, int] | None = None,
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    def sites(self) -> list[tuple[str, list[int], bool]]: ...
    def potential(self, z: Sequence[float]) -> tuple[float, list[float]]: ...
    def postprocess(
        self, samples: Sequence[Sequence[float]]
    ) -> list[tuple[str, list[int], list[float]]]: ...
    def sample_nuts(
        self,
        num_warmup: int,
        num_samples: int,
        max_tree_depth: tuple[int, int] = (6, 6),
        seed: int = 0,
        z0: Sequence[float] | None = None,
        draws: Sequence[Any] | None = None,
        target_accept_prob: float = 0.8,
        dense_mass: bool = True,
        step_size: float = 1.0,
        stop_when_draws_fail: bool = False,
        progress: Callable[[int, int, bool, float, int, bool], None] | None = None,
    ) -> dict[str, Any]: ...

def q_bayesian_active_learning_by_disagreement(
    members: Sequence[Model],
    sampler: SobolQMCNormalSampler,
    x_pending: Array | None = None,
) -> AcquisitionFunction: ...

class EnsembleThompsonSamplingState:
    def redraw(
        self,
        batch_size: int,
        draws: Sequence[Draws] | None = None,
        ensemble_indices: Sequence[int] | None = None,
    ) -> None: ...
    @property
    def batch_size(self) -> int | None: ...
    @property
    def ensemble_indices(self) -> list[int] | None: ...

def ensemble_pathwise_thompson_sampling(
    members: Sequence[Model],
    objective: Any = None,
    transform: tuple[Sequence[float], float] | None = None,
    num_features: int = 1024,
    seed: int | None = None,
) -> tuple[AcquisitionFunction, EnsembleThompsonSamplingState]: ...

# Phase 10b: preference learning.
class PairwiseGP:
    def __init__(
        self,
        datapoints: Array | None,
        comparisons: Array | None,
        dim: int,
        likelihood: str = "probit",
        covar_spec: KernelSpec | None = None,
        covar_tree: KernelNode | None = None,
        raw: Sequence[float] | None = None,
        param_map: Sequence[Sequence[int]] | None = None,
        input_chain: Sequence[Any] | None = None,
        input_chain_training: Sequence[bool] | None = None,
        input_chain_prefixes: Sequence[str] | None = None,
        jitter: float = 1e-6,
        xtol: float | None = None,
        consolidate_rtol: float = 0.0,
        consolidate_atol: float = 1e-4,
        maxfev: int | None = None,
        mean_constant: float = 0.0,
        float32: bool = True,
        draws: Sequence[float] | None = None,
        seed: int = 0,
        update: bool = True,
    ) -> None: ...
    def copy(self) -> PairwiseGP: ...
    @property
    def batch_shape(self) -> list[int]: ...
    @property
    def dim(self) -> int: ...
    @property
    def has_data(self) -> bool: ...
    @property
    def n(self) -> int: ...
    @property
    def m(self) -> int: ...
    @property
    def likelihood(self) -> str: ...
    @property
    def needs_draws(self) -> int: ...
    def raw_parameters(self) -> list[float]: ...
    def set_raw_parameters(self, raw: Sequence[float]) -> None: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    @staticmethod
    def default_kernel_spec(d: int, float32: bool = True) -> KernelSpec: ...
    @staticmethod
    def initial_raw(
        covar_spec: KernelSpec | None = None, covar_tree: KernelNode | None = None
    ) -> list[float]: ...
    mean_constant: float
    training: bool
    def update(self, draws: Sequence[float] | None = None) -> None: ...
    def set_train_data(
        self,
        datapoints: Array,
        comparisons: Array,
        update_model: bool = True,
        draws: Sequence[float] | None = None,
    ) -> None: ...
    def datapoints(self) -> ArrayOut | None: ...
    def unconsolidated_datapoints(self) -> ArrayOut | None: ...
    def comparisons(self, unconsolidated: bool = False) -> ArrayOut | None: ...
    def consolidated_indices(self) -> list[int] | None: ...
    def D(self) -> ArrayOut | None: ...
    def utility(self) -> ArrayOut: ...
    def x0(self) -> ArrayOut | None: ...
    def set_x0(self, x0: Sequence[float] | None) -> None: ...
    def likelihood_hess(self) -> ArrayOut: ...
    def covar(self) -> ArrayOut: ...
    def covar_chol(self) -> ArrayOut: ...
    def covar_inv(self) -> ArrayOut: ...
    def hlcov_eye(self) -> ArrayOut: ...
    def mll(self) -> float: ...
    def forward_mll(self, raw: Sequence[float] | None = None) -> tuple[float, list[float]]: ...
    def fit(
        self,
        maxcor: int | None = None,
        pgtol: float | None = None,
        ftol: float | None = None,
        maxiter: int | None = None,
        maxfun: int | None = None,
        maxls: int | None = None,
        max_attempts: int | None = None,
        seed: int | None = None,
    ) -> tuple[float, int, int, str, str, int, list[tuple[float, int, int, str, str, int]]]: ...
    def condition_on_observations(
        self, x: Array, y: Array, update: bool = True, draws: Sequence[float] | None = None
    ) -> PairwiseGP: ...
    @property
    def constant(self) -> list[float]: ...
    def set_constant(self, value: Array) -> None: ...
    @property
    def aug_batch_shape(self) -> list[int]: ...
    def model_handle(self) -> Model: ...
    def input_chain_states(self) -> list[Any] | None: ...
    def settings(self) -> tuple[float, float | None, float, float, int | None]: ...

def pairwise_likelihood_eval(likelihood: str, what: str, utility: Array, d: Array) -> ArrayBuf: ...
def compose_models(pref: Model, outcome: Model | None = None) -> Model: ...
def analytic_eubo(
    pref: Model, outcome: Model | None = None, previous_winner: Sequence[float] | None = None
) -> AcquisitionFunction: ...
def pairwise_mc_posterior_variance(
    model: Model, objective: ObjectiveSpec | None, sampler: SobolQMCNormalSampler
) -> AcquisitionFunction: ...
def learned_objective_samples(
    model: Model, y: Array, z: Array, adjoint: Array | None = None, model_index: int | None = None
) -> tuple[ArrayBuf, ArrayBuf | None]: ...

class PairwiseBALD:
    def __init__(
        self,
        pref: Model,
        outcome: Model | None = None,
        num_samples: int = 1024,
        std_noise: float = 0.0,
        seed: int = 0,
    ) -> None: ...
    def acquisition_function(self) -> AcquisitionFunction: ...
    def forward(self, x: Array, draws: Array) -> ArrayBuf: ...
    def forward_with_grad(self, x: Array, draws: Array) -> tuple[ArrayBuf, ArrayBuf]: ...
    @property
    def num_samples(self) -> int: ...

# -- Approximate GP models (phase 10c) ----------------------------------------

class SingleTaskVariationalGP:
    def __init__(
        self,
        train_x: Array,
        train_y: Array | None,
        inducing: Array,
        num_outputs: int,
        dim: int,
        covar_tree: KernelNode,
        covar_batched: bool,
        noise: GaussianLikelihoodSpec,
        likelihood: str = "gaussian",
        mean: str = "constant",
        mean_prior: Prior | None = None,
        mean_constraint: Constraint | None = None,
        mean_batched: bool = False,
        learn_inducing_points: bool = True,
        jitter: float = 1e-6,
        input_chain: Sequence[Any] | None = None,
        input_chain_training: Sequence[bool] | None = None,
        input_chain_prefixes: Sequence[str] | None = None,
        raw_train_x: Array | None = None,
        outcome_means: Sequence[float] | None = None,
        outcome_stdvs: Sequence[float] | None = None,
        outcome_min_stdv: float = 1e-8,
        raw: Sequence[float] | None = None,
    ) -> None: ...
    def model_handle(self, transform: tuple[Sequence[float], float] | None = None) -> Model: ...
    def raw_parameters(self) -> list[float]: ...
    @property
    def parameter_state_version(self) -> int: ...
    def set_raw_parameters(self, values: Sequence[float]) -> None: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    def inducing_points(self) -> ArrayOut: ...
    def set_inducing_points(self, z: Array) -> None: ...
    def likelihood_values(self) -> list[float]: ...
    def kernel_matrix(self, x: Array, output: int) -> ArrayOut: ...
    def latent(self, x: Array) -> tuple[ArrayOut, ArrayOut, ArrayOut]: ...
    @property
    def noise(self) -> list[float]: ...
    def set_noise(self, value: Array) -> None: ...
    def noises(self) -> tuple[list[float], float]: ...
    @property
    def constant(self) -> list[float]: ...
    def set_constant(self, value: Array) -> None: ...
    @property
    def aug_batch_shape(self) -> list[int]: ...
    def elbo(
        self,
        x: Array,
        y: Array,
        num_data: float,
        beta: float = 1.0,
        raw: Sequence[float] | None = None,
    ) -> tuple[float, float, float]: ...
    def elbo_value_and_grad(
        self, num_data: float, beta: float = 1.0, raw: Sequence[float] | None = None
    ) -> tuple[float, list[float]]: ...
    def elbo_value_and_grad_on_data(
        self,
        x: Array,
        y: Array,
        num_data: float,
        beta: float = 1.0,
        raw: Sequence[float] | None = None,
    ) -> tuple[float, list[float]]: ...
    def elbo_terms_parameter_vjp(
        self,
        x: Array,
        y: Array,
        num_data: float,
        beta: float,
        adjoints: Sequence[float],
    ) -> list[float]: ...
    def fit(
        self,
        num_data: float,
        beta: float = 1.0,
        maxcor: int | None = None,
        pgtol: float | None = None,
        ftol: float | None = None,
        maxiter: int | None = None,
        maxfun: int | None = None,
        maxls: int | None = None,
        max_attempts: int | None = None,
        seed: int | None = None,
        mask: Sequence[bool] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
    ) -> tuple[float, int, int, str, str, int]: ...
    def fit_torch(
        self,
        num_data: float,
        beta: float = 1.0,
        mask: Sequence[bool] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        step_limit: int | None = None,
        stopping: tuple[int, bool, int, float, float] | None = None,
        timeout_sec: float | None = None,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> tuple[int, float, int, float | None, str | None]: ...
    def sample_all_priors(self, seed: int | None = None) -> None: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_inducing(self) -> int: ...
    def train_inputs(self) -> ArrayOut: ...
    def train_targets(self) -> ArrayOut | None: ...
    def copy(self) -> SingleTaskVariationalGP: ...

def pivoted_cholesky_init_py(
    kernel_matrix: Array, max_length: int, quality_scores: Sequence[float], epsilon: float = 1e-6
) -> list[int]: ...
def expected_improvement_quality_py(
    mean: Sequence[float], variance: Sequence[float], maximize: bool
) -> list[float]: ...

# ---------------------------------------------------------------------------
# Phase 11a: safe_math / probability kernels, MVNXPB, LinESS, QMC, torch's
# generator, test functions and helpers
# ---------------------------------------------------------------------------

Values = list[float] | array[float]

def elementwise(
    name: str,
    inputs: Sequence[Sequence[float]],
    params: Sequence[float] = ...,
    single_precision: bool = False,
    with_grad: bool = False,
) -> tuple[list[Values], list[list[Values]] | None]:
    """The element-wise function ``name`` of equally long inputs: the values
    of every output and (with ``with_grad``) the derivatives of every output
    with respect to every input (``grads[o][k]``)."""

def reduce(
    name: str, data: Sequence[float], n: int, params: Sequence[float] = ..., with_grad: bool = False
) -> tuple[Values, Values | None]:
    """The reduction ``name`` over consecutive rows of ``n`` values."""

def log_prob_feas_from_bounds(
    lower: Sequence[tuple[int, float]],
    upper: Sequence[tuple[int, float]],
    both: Sequence[tuple[int, float, float]],
    means: Sequence[float],
    sigmas: Sequence[float],
    m: int,
    with_grad: bool = False,
) -> tuple[Values, tuple[Values, Values] | None]: ...
def percentile_of_score(
    data: Sequence[float], scores: Sequence[float], n: int, single_precision: bool
) -> Values: ...

class MvnxpbState:
    def __init__(
        self,
        n: int,
        batch: int,
        step: int,
        pc_step: int,
        perm: Sequence[int],
        bounds: Sequence[float],
        tril: Sequence[float],
        pc_perm: Sequence[int],
        diag: Sequence[float] | None,
        plug_ins: Sequence[float],
        log_prob: Sequence[float],
        log_prob_extra: Sequence[float] | None,
    ) -> None: ...
    @staticmethod
    def standardized(
        cov: Sequence[float], bounds: Sequence[float], batch: int, n: int
    ) -> MvnxpbState: ...
    @property
    def n(self) -> int: ...
    @property
    def batch(self) -> int: ...
    @property
    def step(self) -> int: ...
    @property
    def pc_step(self) -> int: ...
    @property
    def perm(self) -> list[int]: ...
    @property
    def pc_perm(self) -> list[int]: ...
    @property
    def bounds(self) -> Values: ...
    @property
    def tril(self) -> Values: ...
    @property
    def diag(self) -> Values | None: ...
    @property
    def plug_ins(self) -> Values: ...
    @property
    def log_prob(self) -> Values: ...
    @property
    def log_prob_extra(self) -> Values | None: ...
    def solve(self, num_steps: int | None = None, eps: float = 1e-10) -> Values: ...
    def select_pivot(self) -> list[int]: ...
    def pivot(self, pivot: Sequence[int]) -> None: ...
    def augment(
        self,
        cov: Sequence[float],
        bounds: Sequence[float],
        cross: Sequence[float],
        m: int,
        disable_pivoting: bool = False,
        jitter: float | None = None,
        max_tries: int | None = None,
    ) -> tuple[MvnxpbState, bool]: ...

def mvnxpb_log_prob_with_grad(
    cov: Sequence[float], bounds: Sequence[float], batch: int, n: int
) -> tuple[Values, Values, Values]: ...
def pivoted_cholesky_update(
    tril: Sequence[float], batch: int, n: int, step: int, eps: float
) -> ArrayBuf: ...
def pivoted_cholesky_pivot(
    tril: Sequence[float],
    perm: Sequence[int],
    diag: Sequence[float] | None,
    batch: int,
    n: int,
    step: int,
    pivots: Sequence[int],
) -> tuple[Values, list[int], Values | None]: ...
def augment_cholesky(
    laa: Sequence[float],
    kbb: Sequence[float],
    kba: Sequence[float] | None,
    lba: Sequence[float] | None,
    jitter: float | None,
    batch: int,
    na: int,
    nb: int,
) -> ArrayBuf: ...

class LinearEllipticalSliceSampler:
    def __init__(
        self,
        a: Array,
        b: Sequence[float],
        x0: Sequence[float],
        fixed_indices: Sequence[int] | None = None,
        mean: Sequence[float] | None = None,
        covariance_matrix: Array | None = None,
        root: Array | None = None,
        root_diag: Sequence[float] | None = None,
        num_chains: int = 1,
        single_precision: bool = False,
    ) -> None: ...
    @property
    def draw_shape(self) -> tuple[int, int]: ...
    @property
    def lifetime_samples(self) -> int: ...
    @property
    def x(self) -> ArrayBuf: ...
    @property
    def z(self) -> ArrayBuf: ...
    @property
    def standardized(self) -> tuple[ArrayBuf, ArrayBuf]: ...
    def run(
        self,
        nu: Sequence[float],
        u: Sequence[float],
        steps: int,
        keep: int,
        check_feasibility: bool = False,
    ) -> ArrayBuf: ...
    def arc_ends(self, nu: Sequence[float]) -> ArrayBuf: ...
    def is_feasible(self, points: Array, transformed: bool = False) -> list[bool]: ...

def normal_qmc_transform(
    uniform: Sequence[float],
    n: int,
    sobol_dim: int,
    d: int,
    inv_transform: bool,
    single_precision: bool,
) -> ArrayBuf: ...
def mvn_qmc_root(cov: Array, single_precision: bool) -> ArrayBuf: ...
def detect_duplicates(x: Array, rtol: float, atol: float) -> list[tuple[int, int]]: ...
def scalarize_posterior_moments(
    mean: Array, covariance: Array, weights: Sequence[float], offset: float
) -> tuple[ArrayBuf, ArrayBuf]: ...
def mvn_qmc_project(
    samples: Array, root: Array, mean: Sequence[float], single_precision: bool
) -> ArrayBuf: ...

class SobolEngine:
    def __init__(self, dimension: int, seed: int | None = None) -> None: ...
    @property
    def seed(self) -> int: ...
    @property
    def num_generated(self) -> int: ...
    def draw(self, n: int) -> ArrayBuf: ...

def torch_randn(seed: int, n: int) -> ArrayBuf:
    """``torch.randn(n, dtype=float64)`` after ``torch.manual_seed(seed)``."""

def torch_rand(seed: int, n: int, single_precision: bool = False) -> ArrayBuf:
    """``torch.rand(n)`` (float64 or float32) after ``torch.manual_seed(seed)``."""

def null_space(c: Array) -> ArrayBuf: ...

class SingleObjectiveProblem:
    def __init__(
        self,
        name: str,
        dim: int,
        bounds: Array,
        discrete: Sequence[int] = ...,
        params: Sequence[float] = ...,
        use_smooth_interp: bool | None = None,
        max_step_size: float | None = None,
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def bounds(self) -> ArrayBuf: ...
    @property
    def num_constraints(self) -> int: ...
    def evaluate_true(self, x: Array) -> ArrayBuf: ...
    def evaluate_slack_true(self, x: Array) -> ArrayBuf: ...
    def cost(self, x: Array) -> ArrayBuf: ...

def round_nearest(
    x: Sequence[float], increment: float, bounds: tuple[float, float] | None = None
) -> list[float]: ...
def data_log_likelihoods(model: SingleTaskGP) -> ArrayBuf: ...
def sample_cached_cholesky(
    rows: Sequence[float],
    baseline_l: Sequence[float],
    mean_new: Sequence[float],
    base: Sequence[float],
    batch: int,
    n: int,
    q: int,
    s: int,
    max_tries: int,
) -> ArrayBuf: ...
def psd_safe_cholesky(
    a: Sequence[float],
    batch: int,
    n: int,
    jitter: float | None = None,
    max_tries: int | None = None,
) -> ArrayBuf: ...

# -- Higher-order GP models (phase 10c) ---------------------------------------

class HigherOrderGP:
    def __init__(
        self,
        train_x: Array,
        train_y: Array,
        target_shape: Sequence[int],
        latent_dims: Sequence[int],
        dim: int,
        covar_trees: Sequence[KernelNode],
        noise: GaussianLikelihoodSpec,
        raw: Sequence[float],
        latent_priors: Sequence[Array | None] | None = None,
        learn_latent_pars: bool = True,
        input_chain: Sequence[Any] | None = None,
        input_chain_training: Sequence[bool] | None = None,
        input_chain_prefixes: Sequence[str] | None = None,
        raw_train_x: Array | None = None,
        outcome: tuple[float, float] | None = None,
    ) -> None: ...
    def model_handle(self) -> Model: ...
    def raw_parameters(self) -> list[float]: ...
    def set_raw_parameters(self, values: Sequence[float]) -> None: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    @property
    def aug_batch_shape(self) -> list[int]: ...
    @property
    def noise(self) -> list[float]: ...
    def set_noise(self, value: Array) -> None: ...
    def mll(self, raw: Sequence[float] | None = None) -> ArrayOut: ...
    def mll_value_and_grad(
        self, raw: Sequence[float] | None = None
    ) -> tuple[ArrayOut, list[float]]: ...
    def fit(
        self,
        maxcor: int | None = None,
        pgtol: float | None = None,
        ftol: float | None = None,
        maxiter: int | None = None,
        maxfun: int | None = None,
        maxls: int | None = None,
        max_attempts: int | None = None,
        seed: int | None = None,
    ) -> tuple[float, int, int, str, str, int]: ...
    def fit_torch(
        self,
        mask: Sequence[bool] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        step_limit: int | None = None,
        stopping: tuple[int, bool, int, float, float] | None = None,
        timeout_sec: float | None = None,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> tuple[int, float, int, float | None, str | None]: ...
    def sample_all_priors(self, seed: int | None = None) -> None: ...
    def mean_variance(
        self, x: Array, observation_noise: bool = False, original: bool = True
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def samples(self, x: Array, base_samples: Array, original: bool = True) -> ArrayBuf: ...
    def moment_vjp(
        self, x: Array, mean_adjoint: Array, variance_adjoint: Array, original: bool = True
    ) -> ArrayBuf: ...
    def samples_vjp(
        self, x: Array, base_samples: Array, adjoint: Array, original: bool = True
    ) -> ArrayBuf: ...
    def condition_on_observations(self, x_tf: Array, y_tf: Array) -> HigherOrderGP: ...
    def train_inputs(self) -> ArrayOut: ...
    def train_targets(self) -> ArrayOut: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_train(self) -> int: ...
    def copy(self) -> HigherOrderGP: ...

def higher_order_latent_sample_py(covariance: Array, normal: Array) -> ArrayOut: ...
def higher_order_variances(factors: Sequence[Array], num_train: int, noise: Array) -> ArrayBuf: ...
def higher_order_variances_vjp(
    factors: Sequence[Array], num_train: int, noise: Array, adjoint: Array
) -> tuple[list[ArrayBuf], ArrayBuf]: ...

class EmpiricalGP:
    def __init__(
        self,
        x: Sequence[Array],
        y: Sequence[Array],
        yvar: Sequence[Array] | None,
        mean: KernelNode,
        covar: KernelNode,
        noise: GaussianLikelihoodSpec,
        kernel_raw: list[float],
        noise_raw: list[float],
        correlated: bool = False,
    ) -> None: ...
    def raw_parameters(self) -> list[float]: ...
    def set_raw_parameters(self, raw: list[float]) -> None: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def condition_on_observations(
        self, x: Array, y: Array, noise: Array | None = None
    ) -> EmpiricalGP: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    @property
    def noise(self) -> list[float]: ...
    def set_noise(self, noise: Array) -> None: ...
    @property
    def aug_batch_shape(self) -> list[int]: ...
    @property
    def dim(self) -> int: ...
    def model_handle(self) -> Model: ...
    def task_model_handle(
        self, outputs: Sequence[int], task_feature: int | None = None
    ) -> Model: ...
    def mll(self, raw: list[float] | None = None) -> ArrayOut: ...
    def mll_value_and_grad(
        self, raw: list[float] | None = None
    ) -> tuple[ArrayOut, list[float]]: ...
    def copy(self) -> EmpiricalGP: ...
    def sample_all_priors(self, seed: int | None = None) -> None: ...
    def fit(self, **kwargs: Any) -> tuple[float, int, int, str, str, int]: ...
    def fit_torch(self, **kwargs: Any) -> tuple[int, float, int, float | None, str | None]: ...

class LatentKroneckerGP:
    def __init__(
        self,
        train_x: Array,
        train_t: Array,
        train_y: Array,
        dim: int,
        covar_x: KernelNode,
        covar_t: KernelNode,
        noise: GaussianLikelihoodSpec,
        raw: Sequence[float],
        mean_x: str = "zero",
        mean_t: str = "zero",
        mean_x_prior: Prior | None = None,
        mean_t_prior: Prior | None = None,
        mean_x_constraint: Constraint | None = None,
        mean_t_constraint: Constraint | None = None,
        input_chain: Sequence[Any] | None = None,
        input_chain_training: Sequence[bool] | None = None,
        input_chain_prefixes: Sequence[str] | None = None,
        raw_train_x: Array | None = None,
        outcome: tuple[float, float] | None = None,
    ) -> None: ...
    def raw_parameters(self) -> list[float]: ...
    def set_raw_parameters(self, raw: Sequence[float]) -> None: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    @property
    def noise(self) -> list[float]: ...
    def set_noise(self, value: Array) -> None: ...
    @property
    def aug_batch_shape(self) -> list[int]: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_train(self) -> int: ...
    @property
    def num_observed(self) -> int: ...
    @property
    def complete_grid(self) -> bool: ...
    def observed_indices(self) -> list[int]: ...
    def train_inputs(self) -> ArrayOut: ...
    def train_t(self) -> ArrayOut: ...
    def train_targets(self) -> ArrayOut: ...
    def model_handle(self) -> Model: ...
    def mll(self, raw: Sequence[float] | None = None) -> ArrayOut: ...
    def mll_value_and_grad(
        self, raw: Sequence[float] | None = None
    ) -> tuple[ArrayOut, list[float]]: ...
    def posterior_grid(self, x: Array, t: Array) -> tuple[ArrayBuf, ArrayBuf]: ...
    def posterior_grid_vjp(
        self, x: Array, t: Array, mean_adjoint: Array, covariance_adjoint: Array
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def samples_vjp(
        self, x: Array, t: Array, base: Array, adjoint: Array
    ) -> tuple[ArrayBuf, ArrayBuf]: ...
    def samples(self, x: Array, t: Array, base: Array) -> ArrayBuf: ...
    def fit(self, **kwargs: Any) -> tuple[float, int, int, str, str, int]: ...
    def fit_torch(self, **kwargs: Any) -> tuple[int, float, int, float | None, str | None]: ...
    def sample_all_priors(self, seed: int | None = None) -> None: ...
    def copy(self) -> LatentKroneckerGP: ...

def empirical_orthogonal_basis(a: Array, method: str = "svd") -> ArrayBuf: ...
def empirical_covariance(
    u1: Array, u2: Array, num_curves: int, correction: int = 0, diag: bool = False
) -> ArrayBuf: ...
def empirical_covariance_vjp(
    u1: Array, u2: Array, gradient: Array, num_curves: int, correction: int = 0, diag: bool = False
) -> tuple[ArrayBuf, ArrayBuf]: ...
def empirical_project_psd(a: Array, min_eigval: float = 0.0) -> ArrayBuf: ...
def empirical_project_psd_vjp(a: Array, gradient: Array, min_eigval: float = 0.0) -> ArrayBuf: ...
def empirical_shrinkage(cov: Array, target: Array, alpha: float) -> ArrayBuf: ...
def empirical_shrinkage_vjp(
    cov: Array, target: Array, alpha: float, gradient: Array
) -> tuple[ArrayBuf, ArrayBuf, float]: ...
def empirical_em_basis(
    mu: Array,
    sigma: Array,
    num_modes: int | None = None,
    correction: int = 0,
    eigenvalue_tol: float = 0.0,
) -> ArrayBuf: ...
def empirical_interp(
    x: Array, y: Array, x_new: Array, bounds_error: bool = True, fill_value: float = ...
) -> ArrayBuf: ...
def empirical_interp_vjp(
    x: Array, y: Array, x_new: Array, gradient: Array, bounds_error: bool = True
) -> tuple[ArrayBuf, ArrayBuf, ArrayBuf]: ...

class _LbfgsbState:
    def __init__(
        self,
        x0: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
        maxcor: int = 10,
        ftol: float = ...,
        gtol: float = 1e-5,
        maxiter: int = 15000,
        maxfun: int = ...,
        maxls: int = 20,
    ) -> None: ...
    def copy(self) -> _LbfgsbState: ...
    @property
    def x(self) -> list[float]: ...
    @property
    def f(self) -> float: ...
    @property
    def g(self) -> list[float]: ...
    @property
    def iterations(self) -> int: ...
    @property
    def evaluations(self) -> int: ...
    @property
    def updates(self) -> int: ...
    @property
    def task(self) -> tuple[int, int]: ...
    def provide_evaluation(self, f: float, gradient: Sequence[float]) -> None: ...
    def resume(self, stop: bool) -> None: ...
    def corrections(self) -> tuple[list[list[float]], list[list[float]]]: ...
    def inverse_hessian_product(self, vector: Sequence[float]) -> list[float]: ...

def em_pretrain(
    xs: Sequence[Array],
    ys: Sequence[Array],
    inducing: Array,
    kernel: KernelNode,
    raw: Sequence[float],
    constant: float,
    indices: Sequence[Sequence[int]] | None,
    noise: float,
    iterations: int,
    mean_prior: bool,
    covariance_prior: bool,
    iw_nu: float | None,
    tolerance: float | None,
    naive_init: bool,
    shrinkage: float,
    record_history: bool,
) -> tuple[ArrayOut, ArrayBuf, ArrayBuf, ArrayOut, list[tuple[ArrayOut, ArrayBuf]], int]: ...
def em_prior_state(prior: Prior) -> tuple[str, list[float], bool, float | None]: ...

class EMGaussianProcess:
    def __init__(
        self,
        x: Array,
        y: Array,
        history_x: Sequence[Array],
        history_y: Sequence[Array],
        inducing: Array,
        mean: Array,
        covariance: Array,
        kernel: KernelNode,
        kernel_raw: Sequence[float],
        likelihood: GaussianLikelihoodSpec,
        noise_raw: float,
        constant: float = 0.0,
        learn_mean: bool = False,
        mean_prior: Prior | None = None,
        mean_constraint: Constraint | None = None,
        base: KernelNode | None = None,
        base_raw: Sequence[float] = (),
        interpolate: bool = True,
        inducing_observations: bool = False,
        freeze_pretrained: bool = False,
        em_iterations: int | None = None,
        em_mean_prior: bool = False,
        em_covar_prior: bool = False,
        em_iw_nu: float | None = None,
        em_tolerance: float | None = 1e-6,
        em_shrinkage: float = 0.0,
        warm_start_em: bool = False,
        learnable_inducing: bool = False,
    ) -> None: ...
    def raw_parameters(self) -> list[float]: ...
    def set_raw_parameters(self, raw: Sequence[float]) -> None: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def parameter_shapes(self) -> list[tuple[str, list[int]]]: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    @property
    def noise(self) -> list[float]: ...
    def set_noise(self, value: Array) -> None: ...
    @property
    def constant(self) -> list[float]: ...
    def set_constant(self, value: Array) -> None: ...
    @property
    def aug_batch_shape(self) -> list[int]: ...
    @property
    def dim(self) -> int: ...
    def model_handle(self) -> Model: ...
    def prior_model_handle(self) -> Model: ...
    def em_state(self) -> tuple[ArrayOut, ArrayBuf, ArrayBuf, ArrayOut]: ...
    def inducing_points(self) -> ArrayBuf: ...
    def refresh_em(self) -> None: ...
    def advance_mll(self, raw: Sequence[float] | None = None) -> tuple[ArrayOut, list[float]]: ...
    def freeze_pretrained(self, freeze: bool) -> None: ...
    def copy(self) -> EMGaussianProcess: ...
    def mll(self, raw: Sequence[float] | None = None) -> ArrayOut: ...
    def mll_value_and_grad(
        self, raw: Sequence[float] | None = None
    ) -> tuple[ArrayOut, list[float]]: ...
    def sample_all_priors(self, seed: int | None = None) -> None: ...
    def fit(self, **kwargs: Any) -> tuple[float, int, int, str, str, int]: ...
    def fit_torch(self, **kwargs: Any) -> tuple[int, float, int, float | None, str | None]: ...

class SharedGPFit:
    def __init__(
        self, models: Sequence[SingleTaskGP], learn_mean: bool, freeze_noise: bool
    ) -> None: ...
    def raw_parameters(self) -> list[float]: ...
    def set_raw_parameters(self, raw: Sequence[float]) -> None: ...
    def local_parameters(self) -> list[list[float]]: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def parameter_shapes(self) -> list[tuple[str, list[int]]]: ...
    def raw_parameter_bounds(self) -> list[tuple[float, float]]: ...
    def mll(self, raw: Sequence[float] | None = None) -> ArrayOut: ...
    def mll_value_and_grad(
        self, raw: Sequence[float] | None = None
    ) -> tuple[ArrayOut, list[float]]: ...
    def fit(self, **kwargs: Any) -> tuple[float, int, int, str, str, int]: ...
    def fit_torch(self, **kwargs: Any) -> tuple[int, float, int, float | None, str | None]: ...

def tensor_kernel(
    kind: str,
    x1: Any,
    x2: Any,
    lengthscale: Any = None,
    nu: float = 2.5,
    diag: bool = False,
    last_dim_is_batch: bool = False,
    parameters: dict[str, Any] | None = None,
) -> Any: ...
def tensor_exact_gp_posterior(
    train_y: Any,
    k_train: Any,
    k_train_query: Any,
    k_query: Any,
    train_mean: Any,
    query_mean: Any,
    train_noise: Any,
    jitter: float = 0.0,
    observation_noise: Any = None,
) -> tuple[Any, Any]: ...
def tensor_exact_gp_factor(
    train_y: Any, k_train: Any, train_mean: Any, train_noise: Any, jitter: float = 0.0
) -> tuple[Any, Any]: ...
def tensor_exact_gp_predict(
    chol: Any,
    alpha: Any,
    k_train_query: Any,
    k_query: Any,
    query_mean: Any,
    observation_noise: Any = None,
) -> tuple[Any, Any]: ...
def tensor_exact_mll(
    train_y: Any,
    k_train: Any,
    train_mean: Any,
    train_noise: Any,
    jitter: float = 0.0,
    normalize: bool = False,
) -> Any: ...
def tensor_psd_safe_cholesky(
    matrix: Any, jitter: float | None = None, max_tries: int = 3
) -> Any: ...
