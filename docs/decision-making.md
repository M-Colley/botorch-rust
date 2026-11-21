# Human preferences and focus point boundary intersection

These extensions implement the two main algorithms from Schmitt, Hoffmann,
Rodemann and Adamy (2022), *Incorporating Human Preferences in Decision Making
for Dynamic Multi-Objective Optimization in Model Predictive Control*,
[DOI 10.3390/inventions7030046](https://doi.org/10.3390/inventions7030046).

`botorch_rs.decision_making` and `openbo.decision_making` share the Python API.
The native Rust API is in
`botorch_core::utils::multi_objective::{decision_making, fpbi}`. These are
extensions inspired by the paper, not existing upstream BoTorch API names.
For the Torch/SciPy sampler, install from the checkout with
`pip install "./botorch_pyo3[decision]"`. Native list-based use needs no extras.

## Select a decision from observed outcomes

```python
import torch
from botorch_rs.decision_making import knee_region, select_preferred

# Costs to minimize; BoTorch-style maximization is the default otherwise.
Y = torch.tensor([[0., 1.], [.2, .4], [.4, .2], [1., 0.]], dtype=torch.double)
mask = knee_region(Y, alpha=.8, maximize=False)
index = select_preferred(Y, [70., 30.], alpha=.8, maximize=False)
# mask = [False, True, True, False], index = 1
# For actual candidate designs X, deploy X[index], not the objective row Y[index].
```

`Y` may have shape `(..., n, m)`. Preferences have shape `(..., m)` and broadcast
over outcome batch dimensions. Indices refer to the original rows. Dominated
rows and duplicates after the first are removed before normalization. Ties
select the first remaining original row. A zero observed range contributes
zero; it is not evidence that the objective is globally constant.

The implementation follows equations 8--13: normalize each objective on the
retained front; find the normalized point with largest Euclidean norm; compute
the **signed** distance from its plane; retain distances at least
`alpha * max_distance`; minimize the preference-weighted normalized costs on
that region. `alpha` lies in `[0,1]`. Because distances are signed, `alpha=0`
can exclude some points. Nonnegative preference weights need not sum to one,
but must not all be zero. Objectives are maximized unless `maximize=False`.
For mixed objective directions, orient the columns before calling.

Lists and NumPy arrays use native Rust without importing Torch or SciPy. Torch
selection keeps its boolean masks and int64 indices on the original device.
Discrete selection is not an acquisition function or a differentiable utility.

## Generate a front using FPBI

```python
import torch
from botorch_rs.decision_making import sample_fpbi, select_preferred

def costs(X):
    return torch.stack([X[..., 0].square(), (1-X[..., 0]).square()], dim=-1)

front = sample_fpbi(
    costs, torch.tensor([[0.], [1.]], dtype=torch.double),
    n_steps=5, maximize=False, num_restarts=3, seed=0,
)
index = select_preferred(front.Y, [70., 30.], maximize=False)
chosen_X = front.X[index]
print(chosen_X, front.ray_succeeded, front.diagnostics)
```

Bounds have shape `(2,d)`. Objectives accept `(n,d)` and return `(n,m)` with
`m >= 2`. Optional `inequality_constraints(X)` must be nonnegative;
`equality_constraints(X)` must be zero. They constrain individual-extreme
optimization and the FPBI ray problems. `initial_X` supplies additional raw
decision-space starting rows. `num_restarts`, `seed`, `maxiter`, and
`feasibility_tol` control the local solver. `max_rays=10000` guards the
exponential `n_steps ** (m-1)` grid.

FPBI minimizes every individual objective, normalizes using the resulting
extrema, constructs the farthest-pair sampling plane, and solves the actual
continuous constrained subproblems in equation 27. It uses the source's
asymmetric grid (`k=1,...,n_steps`) and unrestricted kappa. For two extreme
costs `[[0,1],[1,0]]`, `n_steps=2` produces anchors `[[.5,.5],[1,0]]`.
`n_steps=1` produces the second extreme, not the midpoint.

`focus` is independently specified in **normalized cost coordinates**, including
when maximizing; its default is the zero/Utopia point. The paper does not
specify a conversion from preference weights to a focus point. Some focus
choices make maximization of kappa unbounded and are rejected. Use
`fpbi_geometry(extreme_Y, ...)` to inspect the plane before optimization;
extreme outcomes have shape `(...,m,m)` with row i coming from optimizing i.
Focus vectors broadcast across geometry batch dimensions.

Only converged solves passing independent feasibility checks are accepted.
Failure to solve an individual extreme raises. Failed rays are exposed in
`ray_succeeded` and `diagnostics`; retained samples include successful individual
extrema and are deduplicated and filtered for domination. Native diagnostics
report the best restart per subproblem (`restart=-1`); the Torch/SciPy adapter
reports every attempt. Local solvers, starting designs and finite differences
can yield different numerical front approximations while implementing the
same equations. Neither path certifies a global Pareto front.

NumPy/list FPBI uses native SLSQP without Torch or SciPy. Torch FPBI uses CPU
SciPy optimization and calls objectives on the original Tensor device; it needs
Torch, NumPy, and SciPy, supports float32/64, and does not propagate autograd
through the optimizer. The core Rust interface has no Python runtime dependency.
Zero observed extreme ranges are padded out of the geometry, repeated rays
reuse their first result, and rank-deficient frames use a reported deterministic
orthogonal completion. Completely collapsed extrema return their unique
nondominated samples and skip rays. Rescale units when ranges or bound widths
overflow. Observed-extrema normalization can miss interior variation in an
objective; it is a geometric approximation, not a proof of constancy.

## BoTorch and botorch_rs surrogate models

```python
from botorch_rs.decision_making import sample_model_fpbi

# fitted_model can be a BoTorch or botorch_rs model.
predicted_front = sample_model_fpbi(fitted_model, bounds, n_steps=5)
```

This adapter uses `model.posterior(X).mean`. These are predicted outcomes, not
new observations or uncertainty-aware safety guarantees. Select an explicit
`model_batch_index=(...)` for batched posterior means. Keep acquisition-driven
exploration and any deployment safeguards separate. The MPC feasibility and
stability results discussed in the source are not guarantees for BO or noisy
human preferences.

## Native Rust example

Run `cargo run --release -p botorch_core --example decision_making`. The example uses
`Matrix` rows for outcomes, `select_preferred`, and `generate_fpbi_front` with a
vector-valued callback and box bounds. Native constraints follow the same
nonnegative-inequality convention as the Python adapter.

## Measured evidence and current limits

See [what worked and what did not](decision-making-evidence.md) for functional
validation, negative final-utility and FPBI comparisons, runtime regressions,
applicable uses and the upstream draft. The methods remain optional research
utilities; the measurements do not establish a general robustness or speed
improvement.
