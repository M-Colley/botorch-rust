"""Type stubs of the ``botorch_rs._botorch_rs`` extension module.

This is the low-level layer used by the ``botorch_rs`` package. Arrays are
exchanged as ``(data, shape)`` tuples: ``data`` holds the values in row-major
order and ``shape`` the dimensions. Use the BoTorch-style classes of the
``botorch_rs`` package instead of this module.
"""

from collections.abc import Callable, Sequence
from typing import Any

Array = tuple[Sequence[float], Sequence[int]]
ArrayOut = tuple[list[float], list[int]]
#: An MC objective: ``("identity",)``, ``("linear", weights)``,
#: ``("callable", f)`` with ``f(samples, X=X)`` on torch tensors, or
#: ``("constrained", f, constraints, infeasible_cost, etas)``.
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
    def transform(self, x: Array) -> ArrayOut: ...
    def untransform(self, x: Array) -> ArrayOut: ...

class Standardize:
    def __init__(self, min_stdv: float = 1e-8, batch_shape: Sequence[int] = ...) -> None: ...
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
    ) -> tuple[Standardize, ArrayOut, ArrayOut | None]: ...
    def transform(
        self, y: Array, yvar: Array | None = None
    ) -> tuple[ArrayOut, ArrayOut | None]: ...
    def untransform(
        self, y: Array, yvar: Array | None = None
    ) -> tuple[ArrayOut, ArrayOut | None]: ...

class Prior:
    @staticmethod
    def log_normal(loc: float, scale: float, float32: bool = True) -> Prior: ...
    @staticmethod
    def gamma(concentration: float, rate: float, float32: bool = True) -> Prior: ...
    @property
    def kind(self) -> str: ...
    @property
    def parameters(self) -> tuple[float, float]: ...
    @property
    def mode(self) -> float: ...
    def log_prob(self, x: float) -> float: ...

class GreaterThan:
    def __init__(
        self,
        lower_bound: float,
        transform: str | None = "softplus",
        initial_value: float | None = None,
        float32: bool = True,
    ) -> None: ...
    @property
    def lower_bound(self) -> float: ...
    @property
    def enforced(self) -> bool: ...
    @property
    def initial_value(self) -> float | None: ...
    def transform(self, raw: float) -> float: ...
    def inverse_transform(self, value: float) -> float: ...

class KernelSpec:
    def __init__(
        self,
        kind: str,
        ard_num_dims: int,
        lengthscale_prior: Prior | None = None,
        lengthscale_constraint: GreaterThan | None = None,
    ) -> None: ...
    @staticmethod
    def dim_scaled(ard_num_dims: int, use_rbf_kernel: bool = True) -> KernelSpec: ...
    @staticmethod
    def matern_gamma(ard_num_dims: int) -> KernelSpec: ...
    def with_scale_kernel(
        self,
        outputscale_prior: Prior | None = None,
        outputscale_constraint: GreaterThan | None = None,
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
    def lengthscale_constraint(self) -> GreaterThan: ...
    @property
    def outputscale_prior(self) -> Prior | None: ...
    @property
    def outputscale_constraint(self) -> GreaterThan | None: ...

class GaussianLikelihoodSpec:
    def __init__(
        self, noise_prior: Prior | None = None, noise_constraint: GreaterThan | None = None
    ) -> None: ...
    @staticmethod
    def lognormal() -> GaussianLikelihoodSpec: ...
    @staticmethod
    def gamma() -> GaussianLikelihoodSpec: ...
    @property
    def noise_prior(self) -> Prior | None: ...
    @property
    def noise_constraint(self) -> GreaterThan: ...

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
    ) -> None: ...
    def copy(self) -> SingleTaskGP: ...
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
    def train_inputs(self) -> ArrayOut: ...
    def train_targets(self) -> ArrayOut: ...
    def train_yvar(self) -> ArrayOut | None: ...
    def transformed_train_inputs(self) -> ArrayOut: ...
    def transformed_train_targets(self) -> ArrayOut: ...
    def input_transform(self) -> Normalize | None: ...
    def outcome_transform(self) -> Standardize | None: ...
    def covar_module(self) -> KernelSpec: ...
    def likelihood_spec(self) -> GaussianLikelihoodSpec | None: ...
    def parameter_names(self) -> list[tuple[str, int]]: ...
    def parameter_shapes(self) -> list[tuple[str, list[int]]]: ...
    def raw_parameters(self) -> list[float]: ...
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
    ) -> tuple[ArrayOut, ArrayOut]: ...
    def posterior_joint(
        self,
        x: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
    ) -> tuple[ArrayOut, ArrayOut]: ...
    def posterior_samples(
        self,
        x: Array,
        base_samples: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
    ) -> ArrayOut: ...
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

class ModelListGP:
    def __init__(self, models: Sequence[SingleTaskGP]) -> None: ...
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
    ) -> tuple[ArrayOut, ArrayOut]: ...
    def posterior_joint(
        self,
        x: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
    ) -> tuple[ArrayOut, ArrayOut]: ...
    def posterior_samples(
        self,
        x: Array,
        base_samples: Array,
        observation_noise: bool = False,
        output_indices: Sequence[int] | None = None,
        transform: Transform | None = None,
    ) -> ArrayOut: ...

class Model:
    @staticmethod
    def from_single_task_gp(model: SingleTaskGP, transform: Transform | None = None) -> Model: ...
    @staticmethod
    def from_model_list(model: ModelListGP, transform: Transform | None = None) -> Model: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def batch_shape(self) -> list[int]: ...

class SobolQMCNormalSampler:
    def __init__(self, num_samples: int, seed: int | None = None) -> None: ...
    @property
    def num_samples(self) -> int: ...
    @property
    def seed(self) -> int: ...
    def base_samples(self, dim: int) -> ArrayOut: ...
    def set_base_samples(self, base_samples: Array) -> None: ...

class AcquisitionFunction:
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
    def forward(self, x: Array) -> ArrayOut: ...
    def forward_with_grad(self, x: Array) -> tuple[ArrayOut, ArrayOut]: ...
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
) -> tuple[ArrayOut, list[float], ArrayOut, list[float], list[tuple[str, str]]]:
    """Returns ``(candidates, acq_values, restart_candidates, restart_values,
    warnings)``."""

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
) -> ArrayOut: ...
def evaluate_feasibility(
    x: Array,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    nonlinear_inequality_constraints: Sequence[NonlinearConstraint] | None = None,
    tolerance: float | None = None,
) -> tuple[list[bool], list[int]]: ...
def get_polytope_samples(
    n: int,
    bounds: Array,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    seed: int | None = None,
    n_burnin: int = 10000,
    n_thinning: int = 32,
) -> ArrayOut: ...
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
) -> ArrayOut: ...

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
    def draw(self, n: int = 1) -> ArrayOut: ...

def draw_sobol_samples(bounds: Array, n: int, q: int, seed: int | None = None) -> ArrayOut: ...
def draw_sobol_normal_samples(d: int, n: int, seed: int | None = None) -> ArrayOut: ...

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
    def evaluate_true(self, x: Array) -> ArrayOut: ...
    def forward(self, x: Array, noise: bool = True) -> ArrayOut: ...

def test_function_names() -> list[str]: ...

# Multi-objective utilities (botorch.utils.multi_objective) and the simplex /
# hypersphere samplers of botorch.utils.sampling.

def is_non_dominated(
    y: Array, maximize: bool = True, deduplicate: bool = True
) -> tuple[list[bool], list[int]]: ...
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
    def compute_hypervolume(self) -> ArrayOut | None: ...
    def get_hypercell_bounds(self) -> ArrayOut | None: ...
    def pareto_y(self) -> ArrayOut | None: ...
    def y(self) -> ArrayOut | None: ...
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
    def local_upper_bounds(self) -> tuple[ArrayOut, ArrayOut] | None: ...

def expand_ref_point(ref_point: Array, batch_shape: Sequence[int]) -> ArrayOut: ...
def pad_batch_pareto_frontier(
    y: Array,
    ref_point: Array,
    is_pareto: bool = False,
    feasibility_mask: Sequence[bool] | None = None,
) -> ArrayOut: ...
def compute_local_upper_bounds(
    u: Array, z: Array, point: Sequence[float]
) -> tuple[ArrayOut, ArrayOut]: ...
def update_local_upper_bounds_incremental(
    new_pareto_y: Array, u: Array, z: Array
) -> tuple[ArrayOut, ArrayOut]: ...
def get_partition_bounds(z: Array, u: Array, ref_point: Sequence[float]) -> ArrayOut: ...
def compute_non_dominated_hypercell_bounds_2d(
    pareto_y_sorted: Array, ref_point: Array
) -> ArrayOut: ...
def compute_dominated_hypercell_bounds_2d(pareto_y_sorted: Array, ref_point: Array) -> ArrayOut: ...

class ChebyshevScalarization:
    def __init__(self, weights: Sequence[float], y: Array, alpha: float) -> None: ...
    def evaluate(self, y: Array) -> ArrayOut: ...
    def evaluate_with_grad(self, y: Array) -> tuple[ArrayOut, ArrayOut]: ...
    @property
    def weights(self) -> list[float]: ...
    @property
    def alpha(self) -> float: ...
    @property
    def bounds(self) -> ArrayOut | None: ...

def sample_simplex(d: int, n: int = 1, qmc: bool = False, seed: int | None = None) -> ArrayOut: ...
def sample_hypersphere(
    d: int, n: int = 1, qmc: bool = False, seed: int | None = None
) -> ArrayOut: ...
def simplex_from_uniform(u: Array) -> ArrayOut: ...
def hypersphere_from_normal(z: Array) -> ArrayOut: ...

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
    def evaluate_true(self, x: Array) -> ArrayOut: ...
    def forward(self, x: Array, noise: bool = True) -> ArrayOut: ...
    def evaluate_slack_true(self, x: Array) -> ArrayOut: ...
    def evaluate_slack(self, x: Array, noise: bool = True) -> ArrayOut: ...
    def is_feasible(self, x: Array, noise: bool = True) -> tuple[list[bool], list[int]]: ...
    def gen_pareto_front(self, n: int, seed: int | None = None) -> ArrayOut: ...

def multi_objective_problem_names() -> list[str]: ...
