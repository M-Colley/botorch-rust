"""Randomized equivalence sweep of phase 10c (approximate GPs) against live
BoTorch (``-m equivalence``): random ``SingleTaskVariationalGP`` cases
(``n``, ``d``, ``m``, the number of inducing points or BoTorch's default,
learned or fixed inducing points, BoTorch's default kernel or a random
kernel tree of ``kernel_tree_helpers.random_spec``, constant / prior / zero
means, Gaussian likelihoods with and without priors, ``Normalize`` and
``Standardize``). Every case compares the greedy inducing point allocation,
the ELBO terms and gradient at random parameters (two ``beta`` values), the
latent ``q(f)``, posteriors with and without observation noise, and
LogEI / qEI values and input gradients (identical base samples;
scalarized for several outputs).

Tolerances: values 1e-9, gradients 1e-8 (relative). BoTorch's numerical
quirks of kernels (``HalfCauchyPrior`` in float32, the diagonal of
``sq_dist`` with inputs that require gradients, here the learned inducing
points) are fixed for the comparison as in the 6a tests.
"""

from __future__ import annotations

import random

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from approximate_helpers import (  # noqa: E402
    DT,
    GRAD_TOL,
    VALUE_TOL,
    Case,
    bt_elbo_and_grad,
    build_pair,
    copy_parameters,
    elbo_pair,
    fixed_allocator,
    make_data,
    strategy,
)
from botorch.acquisition import analytic as bt_an  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as BtScalarized,
)
from kernel_tree_helpers import (  # noqa: E402
    double_precision_half_cauchy,
    random_spec,
    zero_diagonal_distances,
)
from parity_helpers import Checker, sampler_pair  # noqa: E402

import botorch_rs.acquisition.analytic as rs_an  # noqa: E402
from botorch_rs.exceptions.errors import NumericalError  # noqa: E402
import botorch_rs.acquisition.monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsScalarized,
)

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

check = Checker("10c approximate sweep")
SEEDS = range(40)


def random_case(seed: int) -> Case:
    r = random.Random(seed)
    d = r.randint(1, 4)
    m = r.choice([1, 1, 2, 3])
    n = r.randint(6, 22)
    inducing = r.choice([None, 2, 3, 5, 7])
    kernel = random_spec(r, d) if r.random() < 0.6 else None
    case = Case(name="probe", n=n, d=d, scale=1.0, seed=1000 + seed)
    X, _ = make_data(case)
    while kernel is not None and not _valid_kernel(kernel, X):
        # Kernels whose initial values are degenerate (e.g. a lengthscale of
        # zero from a constraint without transform or initial value: NaN in
        # GPyTorch) are drawn again.
        kernel = random_spec(r, d)
    return Case(
        name=f"sweep{seed}",
        n=n,
        d=d,
        m=m,
        inducing=inducing,
        kernel=kernel,
        mean=r.choice(["default", "prior", "zero"]),
        likelihood=r.choice(["default", "gamma"] + (["gaussian"] if m == 1 else [])),
        learn=r.random() < 0.6,
        input_tf=r.random() < 0.25,
        outcome_tf=r.random() < 0.3,
        scale=r.choice([1.0, 2.0]),
        seed=1000 + seed,
    )


def values_and_grads(acqf: object, X: object) -> tuple[object, object]:
    """Acquisition values and the gradient of their sum (zero where the
    values do not depend on ``X``, e.g. a categorical kernel only)."""
    Xg = X.detach().clone().requires_grad_(True)  # type: ignore[attr-defined]
    values = acqf(Xg)  # type: ignore[operator]
    (grad,) = torch.autograd.grad(values.sum(), Xg, allow_unused=True)
    return values.detach(), torch.zeros_like(Xg) if grad is None else grad


def _valid_kernel(spec: dict, X: object) -> bool:
    from kernel_tree_helpers import build_kernel

    try:
        with torch.no_grad():
            K = build_kernel(spec, "bt").to(DT)(X).to_dense()
        if not bool(torch.isfinite(K).all()):
            return False
        eye = torch.eye(K.shape[-1], dtype=DT)
        return float(torch.linalg.eigvalsh(K + 1e-6 * eye).min()) > 0
    except Exception:  # torch's LinAlgError: ill-conditioned
        return False


def _perturb(bt: object, rs: object, seed: int) -> None:
    from gpytorch.mlls import VariationalELBO

    strategy(bt).variational_params_initialized.fill_(1)
    strategy(rs).variational_params_initialized.fill_(1)
    g = torch.Generator().manual_seed(seed)
    mll = VariationalELBO(bt.likelihood, bt.model, num_data=1)  # type: ignore[attr-defined]
    with torch.no_grad():
        for name, p in mll.named_parameters():
            noise = 0.25 * torch.randn(p.shape, generator=g, dtype=DT)
            if any(k in name for k in ("variational", "inducing", "raw_constant")):
                p.add_(noise)
            elif bool((p >= 0).all()):
                # Non-negative values (e.g. untransformed constraints) stay so.
                p.copy_(p * torch.exp(noise) + noise.abs() * (p == 0))
            else:
                p.add_(noise)
    copy_parameters(bt, rs)


@pytest.mark.parametrize("seed", SEEDS)
def test_sweep(seed: int) -> None:
    case = random_case(seed)
    X, Y = make_data(case)
    with fixed_allocator():
        bt, rs, _, _ = build_pair(case, X, Y)
    check.check(
        "inducing points (allocation)",
        strategy(rs).inducing_points,
        strategy(bt).inducing_points.detach(),
        0.0,
    )
    _perturb(bt, rs, seed)
    with zero_diagonal_distances(), double_precision_half_cauchy():
        for beta in (1.0, 0.5):
            mb, mr = elbo_pair(bt, rs, case.n + 3, beta=beta)
            vb, gb = bt_elbo_and_grad(mb)
            vr, gr = mr.value_and_grad()
            check.check("ELBO", vr, vb, VALUE_TOL)
            check.check("ELBO gradient", gr, gb, GRAD_TOL)
        g = torch.Generator().manual_seed(seed)
        Xt = case.scale * torch.rand(4, case.d, generator=g, dtype=DT) + (1.0 if case.input_tf else 0.0)
        bt.eval()
        rs.eval()
        Xm = Xt if not case.input_tf else bt.input_transform(Xt)
        ob, orr = bt.model(Xm), rs.model(Xm)
        check.check("q(f) mean", orr.mean, ob.mean.detach(), VALUE_TOL)
        check.check("q(f) variance", orr.variance, ob.variance.detach(), VALUE_TOL)
        check.check("q(f) covariance", orr.covariance_matrix, ob.covariance_matrix.detach(), VALUE_TOL)
        for noise in (False, True):
            pb = bt.posterior(Xt, observation_noise=noise)
            pr = rs.posterior(Xt, observation_noise=noise)
            check.check("posterior mean", pr.mean, pb.mean.detach(), VALUE_TOL)
            check.check("posterior variance", pr.variance, pb.variance.detach(), VALUE_TOL)
            check.check(
                "posterior covariance",
                pr.distribution.covariance_matrix,
                pb.distribution.covariance_matrix.detach(),
                VALUE_TOL,
            )
        w = torch.linspace(0.6, 1.4, case.m, dtype=DT)
        tb = BtScalarized(w) if case.m > 1 else None
        tr = RsScalarized(w) if case.m > 1 else None
        best = torch.tensor(float((Y * w).sum(-1).mean()), dtype=DT)
        Xq = Xt.unsqueeze(1)
        vb, gb = values_and_grads(bt_an.LogExpectedImprovement(bt, best_f=best, posterior_transform=tb), Xq)
        vr, gr = rs_an.LogExpectedImprovement(rs, best_f=best, posterior_transform=tr).value_and_grad(Xq)
        check.check("LogEI values", vr, vb, VALUE_TOL)
        check.check("LogEI input gradients", gr, gb, GRAD_TOL)
        sb, sr, _ = sampler_pair(32, 2, 1, seed)
        Xq2 = Xt.reshape(2, 2, case.d)
        cov = bt.posterior(Xq2, posterior_transform=tb).distribution.covariance_matrix.detach()
        eye = torch.eye(cov.shape[-1], dtype=DT)
        if bool(torch.linalg.cholesky_ex(cov + 1e-3 * eye)[1].any()):
            # Not positive definite even with BoTorch's largest jitter (a
            # kernel outside its domain, e.g. ExponentialDecayKernel at
            # negative transformed inputs): both libraries sample through
            # eigendecompositions, whose eigenvectors are not unique.
            return
        ab = bt_mc.qExpectedImprovement(bt, best_f=best, sampler=sb, posterior_transform=tb)
        ar = rs_mc.qExpectedImprovement(rs, best_f=best, sampler=sr, posterior_transform=tr)
        vb, gb = values_and_grads(ab, Xq2)
        try:
            vr, gr = ar.value_and_grad(Xq2)
        except NumericalError:
            # A joint posterior that is not positive definite even with
            # jitter: botorch_rs samples it through its eigendecomposition,
            # whose derivative it does not compute (values only).
            vr, gr = ar(Xq2), None
        check.check("qEI values", vr, vb, VALUE_TOL)
        if gr is not None:
            check.check("qEI input gradients", gr, gb, GRAD_TOL)
