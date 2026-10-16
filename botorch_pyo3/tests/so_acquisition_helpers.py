"""Shared helpers of the live comparisons of the single-objective
acquisition functions of phase 8 with BoTorch (``test_parity_so_acquisition.py``,
``test_equivalence_so_acquisition.py``).

Import only after ``importorskip_botorch()``: this module requires torch and
BoTorch.

Random numbers: explicit base samples (``sampler_pair``), BoTorch's Sobol
draws replaced by seeded normal numbers per dimension (``seeded_draws``,
matched by ``rs_sampler``) or recorded (``recorded_draws``), the path draws
and Sobol points of ``get_optimal_samples`` (``recorded_optimal_samples``),
and BoTorch's max values and samplers of the max-value entropy search
(``mes_pair``).
"""

from __future__ import annotations

import contextlib
import statistics
from collections.abc import Callable, Iterator
from typing import Any

import botorch.acquisition.utils as bt_autils
import botorch.sampling.normal as bt_normal
import botorch.utils.sampling as bt_usampling
import torch
from botorch.acquisition import joint_entropy_search as bt_jes
from botorch.acquisition import knowledge_gradient as bt_kg
from botorch.acquisition import max_value_entropy_search as bt_mes
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
from mo_entropy_helpers import bt_path_draws, recorded_randn

from botorch_rs.acquisition import max_value_entropy_search as rs_mes
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler
from parity_helpers import Checker, perturbed_copy, rel_err

DT = torch.float64


NEI_SAMPLES = 64


def base_samples(n: int, shape: tuple[int, ...], seed: int) -> Any:
    return torch.randn(n, *shape, generator=torch.Generator().manual_seed(seed), dtype=DT)


def sampler_pair(n: int, bt_shape: tuple[int, ...], seed: int) -> tuple[Any, Any]:
    """Samplers with the same base samples (BoTorch's shape ``n x
    bt_shape``)."""
    Z = base_samples(n, bt_shape, seed)
    s_bt = BtSampler(sample_shape=torch.Size([n]), seed=0)
    s_bt.base_samples = Z.clone()
    s_rs = RsSampler(sample_shape=torch.Size([n]), seed=0)
    s_rs.base_samples = Z.reshape(n, -1).clone()
    return s_bt, s_rs


def seeded_draw(d: int, n: int, device: Any = None, dtype: Any = None, seed: Any = None) -> Any:
    """Deterministic standard normal base samples by dimension (installed in
    BoTorch's samplers by :func:`seeded_draws`)."""
    return torch.randn(n, d, generator=torch.Generator().manual_seed(1000 + d), dtype=DT)


@contextlib.contextmanager
def seeded_draws() -> Iterator[None]:
    """BoTorch's Sobol base samples replaced by :func:`seeded_draw` (qNEI draws
    base samples for the baseline and for the baseline with the candidates)."""
    orig = bt_normal.draw_sobol_normal_samples
    bt_normal.draw_sobol_normal_samples = seeded_draw
    try:
        yield
    finally:
        bt_normal.draw_sobol_normal_samples = orig


def rs_sampler(n: int, dims: list[int]) -> Any:
    """A botorch_rs sampler with :func:`seeded_draw`'s base samples for ``dims``."""
    s = RsSampler(sample_shape=torch.Size([n]), seed=0)
    for d in dims:
        z = seeded_draw(d, n)
        s._rs.set_base_samples((z.reshape(-1).tolist(), (n, d)))
    return s


@contextlib.contextmanager
def recorded_draws(record: list[Any]) -> Iterator[None]:
    """Records BoTorch's Sobol base-sample draws."""
    orig = bt_normal.draw_sobol_normal_samples

    def draw(d: int, n: int, device: Any = None, dtype: Any = None, seed: Any = None) -> Any:
        z = orig(d=d, n=n, device=device, dtype=dtype, seed=seed)
        record.append(z.detach().clone())
        return z

    bt_normal.draw_sobol_normal_samples = draw
    try:
        yield
    finally:
        bt_normal.draw_sobol_normal_samples = orig


def finite_differences(acqf: Any, X: Any, h: float = 1e-6) -> Any:
    """Central differences of the sum of ``acqf``'s values."""
    G = torch.zeros_like(X)
    flat = G.view(-1)
    for k in range(X.numel()):
        e = torch.zeros_like(X).view(-1)
        e[k] = h
        e = e.view_as(X)
        flat[k] = (acqf(X + e).sum() - acqf(X - e).sum()).item() / (2 * h)
    return G


def f32(x: float) -> float:
    """A float that BoTorch stores exactly (it keeps Python floats as
    float32 buffers)."""
    return float(torch.tensor(x, dtype=torch.float32))


def botorch_has_kg_fix() -> bool:
    """Whether BoTorch includes meta-pytorch/botorch#3427 (exact gradients of
    qKG with MC value functions)."""
    return hasattr(bt_kg, "_value_function_settings")


def jes_has_3399() -> bool:
    """Whether BoTorch's qJointEntropySearch compares the posterior with the
    transformed optimal values in its MC estimate (meta-pytorch/botorch#3399)."""
    import inspect

    src = inspect.getsource(bt_jes.qJointEntropySearch._compute_monte_carlo_information_gain)
    return "self.optimal_output_values - mean_m" in src


def jes_optimal_inputs(model: Any, optimal_inputs: Any) -> Any:
    """The optimal inputs to pass to BoTorch's qJointEntropySearch so that it
    conditions on the raw inputs (#3399): without the fix, the untransformed
    inputs, which BoTorch's double transform maps back."""
    import inspect

    src = inspect.getsource(bt_jes.qJointEntropySearch.__init__)
    tf = getattr(getattr(model, "models", [model])[0], "input_transform", None)
    if "transform_inputs(self.optimal_inputs)" not in src or tf is None:
        return optimal_inputs
    return tf.untransform(optimal_inputs)


@contextlib.contextmanager
def no_fast_pred_var() -> Iterator[None]:
    """BoTorch's posteriors without ``fast_pred_var`` (exact gradients of the
    fantasy covariances)."""
    import gpytorch.settings as gs
    from botorch.models import gpytorch as bg

    orig = bg.gpt_posterior_settings

    @contextlib.contextmanager
    def settings() -> Iterator[None]:
        with orig(), gs.fast_pred_var(False):
            yield

    bg.gpt_posterior_settings = settings
    try:
        yield
    finally:
        bg.gpt_posterior_settings = orig


@contextlib.contextmanager
def recorded_optimal_samples(draws: list[Any], candidates: list[Any]) -> Iterator[None]:
    """Records BoTorch's random numbers in ``get_optimal_samples``: the path
    draws and the Sobol points."""
    noises: list[Any] = []
    orig_get = bt_autils.get_matheron_path_model
    orig_sobol = bt_usampling.SobolEngine

    def get_path_model(*args: Any, **kwargs: Any) -> Any:
        start = len(noises)
        pm = orig_get(*args, **kwargs)
        draws.extend(bt_path_draws(pm._path, noises[start:]))
        return pm

    class Sobol(orig_sobol):  # type: ignore[misc, valid-type]
        def draw(self, *args: Any, **kwargs: Any) -> Any:
            x = super().draw(*args, **kwargs)
            candidates.append(x.clone())
            return x

    bt_autils.get_matheron_path_model = get_path_model
    bt_usampling.SobolEngine = Sobol
    try:
        with recorded_randn(noises):
            yield
    finally:
        bt_autils.get_matheron_path_model = orig_get
        bt_usampling.SobolEngine = orig_sobol


def mes_pair(cls: str, bt: Any, rs: Any, cs: Any, pending: Any, kb: dict, kr: dict) -> tuple:
    """A max-value entropy search in both libraries with BoTorch's max values
    and samplers."""
    mc = cls in ("qMaxValueEntropy", "qMultiFidelityMaxValueEntropy")
    if mc:
        kb = {"num_fantasies": 4, "num_y_samples": 16, **kb}
    torch.manual_seed(1)
    a_bt = getattr(bt_mes, cls)(bt, cs, X_pending=pending, **kb)
    if mc:
        # BoTorch draws the observation base samples at the first evaluation.
        a_bt(cs[:1].unsqueeze(-2))
        s = RsSampler(sample_shape=torch.Size([16]), seed=0)
        s.base_samples = a_bt.sampler.base_samples.reshape(16, -1).clone()
        f = RsSampler(sample_shape=torch.Size([4]), seed=0)
        if pending is not None:
            f.base_samples = a_bt.fantasies_sampler.base_samples.reshape(4, -1).clone()
        kr = {"num_fantasies": 4, "num_y_samples": 16, "sampler": s, "fantasies_sampler": f, **kr}
    a_rs = getattr(rs_mes, cls)(rs, cs, X_pending=pending, seed=0, **kr)
    a_rs.posterior_max_values = a_bt.posterior_max_values.detach()
    return a_bt, a_rs


def gumbel_reference(model: Any, cs: Any, u: Any, maximize: bool) -> Any:
    """BoTorch's ``_sample_max_value_Gumbel`` in float64 (SciPy's ``brentq``)."""
    import math

    import numpy as np
    from scipy.optimize import brentq
    from scipy.stats import norm

    post = model.posterior(cs)
    w = 1.0 if maximize else -1.0
    mu = (w * post.mean).detach().reshape(-1).numpy()
    sigma = post.variance.clamp_min(1e-8).sqrt().detach().reshape(-1).numpy()
    lo, hi = (mu - 3 * sigma).min(), (mu + 5 * sigma).max()
    dist = norm(mu, sigma)
    q25, q50, q75 = (
        brentq(lambda y, p=p: np.exp(np.sum(dist.logcdf(y))) - p, lo, hi) for p in (0.25, 0.5, 0.75)
    )
    b = (q25 - q75) / (math.log(math.log(4.0 / 3.0)) - math.log(math.log(4.0)))
    a = q50 + b * math.log(math.log(2.0))
    return a - b * u.log().mul(-1.0).log()


class Prior(torch.nn.Module):
    """A Gaussian (log) density of the inputs."""

    def __init__(self, log: bool) -> None:
        super().__init__()
        self.log = log

    def forward(self, X: Any) -> Any:
        lp = -0.5 * ((X - 0.5) / 0.3).pow(2).sum(-1)
        return lp if self.log else lp.exp()


#: BoTorch's own outcomes under rounding-level perturbations of the training
#: inputs: twins at these relative scales, more of them (up to
#: ``N_TWINS_MAX``) when botorch_rs's outcome is a rare one of BoTorch's.
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
N_TWINS = 16
N_TWINS_MAX = 128
#: BoTorch's outcomes whose values lie within this relative distance (or 100
#: x BoTorch's median change, if larger) of one another belong to the same
#: regime.
REGIME_GAP = 1e-6


def twin(model: Any, k: int) -> Any:
    """BoTorch's model (``k = 0``) or its twin ``k`` with training inputs
    perturbed at ``TWIN_SCALES[(k - 1) % 4]`` (relative)."""
    if k == 0:
        return model
    return perturbed_copy(model, seed=k, rel=TWIN_SCALES[(k - 1) % len(TWIN_SCALES)])


def _set_excess(rs_vg: tuple, outcomes: list[tuple], i: int, vtol: float, gtol: float) -> tuple:
    """Candidate set ``i`` against BoTorch's outcomes: botorch_rs's value and
    gradient against the nearest outcome of the regime of BoTorch's outcomes
    that botorch_rs's value falls into, scaled by 10 x the median spread of
    that regime (at least the nominal tolerances)."""
    rv, rg = rs_vg[0][i : i + 1], rs_vg[1][i]
    vals = [v[i : i + 1] for v, _ in outcomes]
    grads = [g[i] for _, g in outcomes]
    gap = max(REGIME_GAP, 100 * statistics.median(rel_err(v, vals[0]) for v in vals[1:]))
    nearest = min(range(len(vals)), key=lambda j: rel_err(rv, vals[j]))
    regime = [j for j in range(len(vals)) if rel_err(vals[j], vals[nearest]) <= gap]
    others = [j for j in regime if j != nearest]
    floor_v = statistics.median(rel_err(vals[j], vals[nearest]) for j in others) if others else 0.0
    floor_g = (
        statistics.median(rel_err(grads[j], grads[nearest]) for j in others) if others else 0.0
    )
    tol_v, tol_g = max(vtol, 10 * floor_v), max(gtol, 10 * floor_g)
    excess, ev, eg = min(
        (max(ev / tol_v, eg / tol_g), ev, eg)
        for ev, eg in ((rel_err(rv, vals[j]), rel_err(rg, grads[j])) for j in regime)
    )
    jumps = sum(rel_err(v, vals[0]) > gap for v in vals[1:])
    return excess, ev, eg, len(regime), jumps


def check_against_twins(
    check: Checker,
    name: str,
    rs_vg: tuple,
    outcome: Callable[[int], tuple],
    vtol: float = 1e-9,
    gtol: float = 1e-8,
) -> None:
    """botorch_rs's values and gradients against BoTorch's outcomes
    ``outcome(k)`` on twin ``k`` (0: the model itself).

    On nearly singular posteriors (e.g. the joint posterior of qNEI over a
    baseline of training points) BoTorch's own result jumps between discrete
    regimes (whether a Cholesky factorization needs jitter) under
    rounding-level perturbations. So each candidate set is compared with the
    regime of BoTorch's outcomes that botorch_rs's value falls into, within
    10 x that regime's own spread; more twins are drawn while a candidate set
    exceeds its tolerance."""
    outcomes = [outcome(k) for k in range(N_TWINS + 1)]
    n_sets = rs_vg[0].shape[0]
    sets = [_set_excess(rs_vg, outcomes, i, vtol, gtol) for i in range(n_sets)]
    while max(s[0] for s in sets) > 1.0 and len(outcomes) <= N_TWINS_MAX:
        n = len(outcomes)
        outcomes += [outcome(k) for k in range(n, n + N_TWINS)]
        sets = [_set_excess(rs_vg, outcomes, i, vtol, gtol) for i in range(n_sets)]
    excess, ev, eg, size, _ = max(sets)
    jumps = max(s[4] / (len(outcomes) - 1) for s in sets)
    check.record(f"{name} values (unscaled, nearest BoTorch outcome)", ev, None)
    check.record(f"{name} gradients (unscaled, nearest BoTorch outcome)", eg, None)
    check.record(f"{name}: share of BoTorch's twins in another regime", jumps, None)
    check.record(f"{name}: size of the regime compared with", float(size), None)
    check.check_bound(f"{name} values and gradients (scaled)", excess, 1.0)


def qpi_nan_gradients(check: Checker, a_rs: Any, X: Any, gr: Any, gb: Any) -> Any:
    """BoTorch's qAnalyticProbabilityOfImprovement gradients (one candidate
    set per row of ``gb``) with the candidate sets where they are NaN
    replaced by botorch_rs's, after checking those against central
    differences of botorch_rs's values (BoTorch's MVNXPB backward produces
    NaN where its ``safe_exp`` clips or the probability underflows; see the
    draft)."""
    nan = torch.isnan(gb).flatten(1).any(-1)
    if not bool(nan.any()):
        return gb
    for k in torch.nonzero(nan).flatten().tolist():
        fd = finite_differences(a_rs, X[k : k + 1])
        # Absolute agreement for the tiny gradients at value 1.
        check.check(
            "qAnalyticPI where BoTorch's gradient is NaN: vs finite differences",
            gr[k : k + 1],
            fd,
            1e-7,
            atol=1e-12,
        )
    out = gb.clone()
    out[nan] = gr[nan]
    return out
