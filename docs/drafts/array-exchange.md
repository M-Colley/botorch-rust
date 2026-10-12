# Array exchange between Python and Rust: README note (draft)

Draft text for `README.md`, to be integrated by the maintainer (branch
`perf/array-exchange`). Measured on Windows 11 (Intel Core Ultra 7 155H) with
Python 3.13.12, PyTorch 2.13.0 (CPU), NumPy 2.5.2 and BoTorch main at
da00e04.

## What changed

Arrays used to cross the boundary between Python and the extension as lists
of Python floats (`tolist()` on the way in, `torch.tensor(list)` /
`np.asarray(list)` on the way out). That cost about 50 ns per value in each
direction, which dominated cheap functions evaluated on many points: DTLZ2 on
10,000 points took 2–7 ms through the Python API with NumPy or torch inputs,
against 1.2 ms in BoTorch and about 0.35 ms for the evaluation in Rust.

Arrays now cross as float64 buffers of the standard library
(`array.array('d')`), with no new dependency and no `unsafe` code:

- **Arguments.** Tensors and NumPy arrays with more than 128 (torch) or 32
  (NumPy) values are copied once by their library into a buffer (torch
  converts the dtype, follows strides and copies from the GPU; NumPy also
  converts the byte order), which the extension reads with one memcpy
  through PyO3's buffer support. torch does not need NumPy for this.
- **Results** whose size grows with the inputs (test functions, posterior
  means, variances, covariances and samples, acquisition values and
  gradients, transforms, samplers, polytope samples and the multi-objective
  utilities) come back as buffers above 16 values, boolean masks as bytes.
  `torch.frombuffer` / `np.array` turn them into the caller's array type with
  one copy, so results never share memory with anything else.
- **Callables and autograd.** Samples passed to callable objectives and
  constraints, their gradients, and the autograd bridges of acquisition
  functions and the Chebyshev scalarization use the same buffers.
- Nested lists, small arrays and small results of fixed size (bounds,
  hyperparameters, candidates) are still exchanged as lists, which are
  cheaper for a few values.

Behaviour is unchanged: results have the array type, floating dtype and
device of the input, with the same values bit for bit (every dtype, strided,
big-endian and read-only inputs included). One corner case differs: under
`torch.set_default_device(...)`, results with more than 16 values computed
from CPU tensors are now CPU tensors like their input, whereas `torch.tensor`
used to put them on the default device.

## Speedups

Wall time through the Python API (`scripts/benchmark_arrays.py`, each
library's default number of threads; best of two interleaved runs of the
median of 7):

| Task | NumPy before | NumPy after | torch before | torch after | BoTorch |
|---|---|---|---|---|---|
| `model.posterior(X)` mean and variance, 10,000 x 6 | 4.5 ms | 2.5 ms | 5.6 ms | 2.4 ms | 227 ms |
| LogEI, 10,000 x 1 x 6 | 4.4 ms | 2.8 ms | 5.0 ms | 2.7 ms | 25.8 ms |
| DTLZ2, 10,000 x 6 | 2.2 ms | 0.50 ms | 3.2 ms | 0.62 ms | 1.24 ms |

DTLZ2 on 10,000 points is 4–5× faster, now twice as fast as BoTorch instead
of 2.5× slower: converting its arguments and results takes about 0.06 ms
(`to_flat` 45 µs, the result 12 µs) instead of 2.3 ms. Other calls (best of
three interleaved runs):

| Task | Before | After |
|---|---|---|
| `is_non_dominated`, 128 x 200 x 3 (torch) | 6.2 ms | 0.9 ms |
| qLogEI with a callable objective, 64 x 2 x 6, 512 samples (torch) | 18.8 ms | 13.9 ms |

Calls at a single point (`q = 1`) are unchanged: posterior, LogEI and DTLZ2
at 1 x 6 differ by at most 0.3 µs (2%) for torch, NumPy and lists (runs
pinned to one core, interleaved, e.g. 33 µs / 19 µs / 15 µs with torch), and
nested lists of any size take the same time as before.

## Text for the README

"Arrays" in the migration notes, after the first sentence:

> Arrays with more than a few values cross the boundary as float64 buffers
> (one copy by torch or NumPy, one memcpy in Rust), so converting arrays costs
> microseconds even for 10,000 points; results never share memory with the
> inputs or with each other.

"Development": `python scripts/benchmark_arrays.py [--botorch]` measures the
exchange (posteriors, LogEI and DTLZ2 on 10,000 points and at one point).
