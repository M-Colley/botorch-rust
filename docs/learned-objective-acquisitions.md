# Learned objectives in native Monte Carlo acquisitions

Audited against official BoTorch revision `b0dd01be6`.

`LearnedObjective` is accepted by native single-objective MC acquisitions.
The preference model can be a native `PairwiseGP` or a supported native
deterministic utility model. Multi-output outcome models supply vectors to
that scalar preference model. Preference sample counts and seeds retain the
standalone API; each outcome sample receives its own IID preference draws,
shared across candidate t-batches as in the official sampler.

The Rust MC core retains the expanded `P * S` utility sample axis. Constraint
probabilities repeat over the preference draws, and their adjoints sum back
to the original `S` outcome samples. Native preference posterior/root VJPs
and native outcome posterior VJPs compute candidate-input gradients without
finite differences. The existing optional Torch objective callback carries
the adjoints between these native computations; it does not invoke BoTorch
inference. List, NumPy, and Torch candidate inputs work through that bridge.
The bridge therefore requires Torch even when candidate arrays are lists.

Deterministic preference objectives use the native utility mean and now
preserve their standalone input-autograd graph. The mixed-dtype warning
checks actual preference-model buffer dtypes, matching the official rule.
These are first-order query derivatives; implicit differentiation through
preference fitting and arbitrary preference-parameter graphs are not claimed.

Validation covers EI, LogEI, PI, simple regret, UCB, posterior standard
deviation, LCB, NEI, and LogNEI; stochastic PairwiseGP and affine deterministic
preferences; multiple candidate batches; constraints and pending points;
list/NumPy/Torch finite differences; expanded `compute_best_f` shapes; and
fully Bayesian outcome ensembles sharing preference normals. The scalar
objective is not a multi-objective hypervolume objective. Existing ensemble
baseline pruning with a custom objective remains explicitly unsupported.
An explicit `GenericDeterministicModel` utility executes its caller-supplied
Torch function and preserves that graph, without requiring the native model
input-dimension hint; this is the existing callable interoperability path.

## Correcting the noisy-improvement cache law

With a stochastic learned preference, NEI and LogNEI use full-joint sampling
even if `cache_root=True` was requested. An explicit request emits a warning;
the default selects the correct path quietly. Pointwise deterministic
preferences retain native baseline-root caching.

The pinned official cached path applies `LearnedObjective` separately to
baseline outcomes and candidate outcomes. Those separate preference GP
samples do not have the required joint baseline/candidate covariance, and
changing their point counts changes the layout of their preference normals.
This can assign substantial improvement to a candidate already in the
baseline. This is a probability-law error, not ordinary numerical variation.

The following unmodified official example gives approximately `6.4e-5` with
`cache_root=False` and `0.45037` with `cache_root=True` in the audited
environment. The exact repeated point should not improve its baseline;
the small nonzero joint value comes from covariance-root jitter.

```python
import numpy as np
import torch
from botorch.models import PairwiseGP, SingleTaskGP
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.objective import LearnedObjective
from botorch.sampling.normal import SobolQMCNormalSampler

X = torch.tensor([[.1, .2], [.3, .8], [.7, .4], [.9, .7]], dtype=torch.float64)
Y = torch.stack((X[:, 0].sin(), X[:, 1].square()), -1)
outcome = SingleTaskGP(X, Y, torch.full_like(Y, .03))
P = torch.rand(7, 2, generator=torch.Generator().manual_seed(82), dtype=torch.float64)
C = torch.tensor([[0, 1], [2, 3], [4, 5], [5, 6], [0, 4], [3, 6]])
np.random.seed(83)
preference = PairwiseGP(P, C)
for cached in [False, True]:
    objective = LearnedObjective(preference, sample_shape=torch.Size([32]), seed=84)
    acquisition = qNoisyExpectedImprovement(
        outcome, X[:2], objective=objective, cache_root=cached,
        sampler=SobolQMCNormalSampler(torch.Size([64]), seed=85),
        prune_baseline=False,
    )
    print(cached, acquisition(X[:1]).detach().item())
```

The immediate upstream solution is `cache_root=False` for stochastic learned
objectives. A correct optimized implementation could cache the outcome
baseline solve while evaluating the preference posterior jointly on the
concatenated baseline and candidate outcome samples. Alternatively it must
condition a genuine learned-preference cache and retain consistent common
preference normals. Merely caching baseline preference utilities and drawing
candidate utilities separately does not preserve the law.

## Objective composition

Native legacy `ConstrainedMCObjective` also repeats its feasibility values
when wrapping an augmented objective, and folds their derivatives back.
Its public Torch forward uses the same repetition. The official legacy
wrapper does not repeat these samples and can fail with incompatible leading
dimensions. The comparison test evaluates the explicit repeated feasibility
formula with the official learned samples. Direct acquisition `constraints=`
remain the preferred interface; official LogEI rejects the legacy wrapper.

The port's existing multi-dimensional preference `sample_shape` convention
flattens all preference draws into the leading utility axis. The official
standalone method leaves extra sample dimensions in its output; this known
shape deviation remains documented in the preference audit.
