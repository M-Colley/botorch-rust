//! Stopping criteria of iterative optimizers (`botorch.optim.stopping`):
//! [`ExpMaStoppingCriterion`], the exponential-moving-average criterion that
//! BoTorch uses for torch optimizers (`gen_candidates_torch`,
//! `fit_gpytorch_mll_torch`) and for the outer loop of
//! `optimize_acqf_cyclic`.

/// A stopping criterion evaluated once per iteration (BoTorch's
/// `StoppingCriterion` protocol).
pub trait StoppingCriterion: Send {
    /// Records the function values of the current iteration (one value per
    /// element; with several elements the criterion holds when it holds for
    /// all of them) and returns whether to stop.
    fn evaluate(&mut self, fvals: &[f64]) -> bool;

    /// Resets the criterion to its initial state.
    fn reset(&mut self);
}

/// The weights `exp(linspace(-1, 0, 10)) / sum` of BoTorch's default
/// criterion as torch computes them (float32 values of torch 2.x on CPU,
/// whose vectorized `exp` and `sum` are not correctly rounded).
const DEFAULT_WEIGHT_BITS: [u32; 10] = [
    0x3d6c_38f6,
    0x3d83_fdd8,
    0x3d93_80cb,
    0x3da4_d668,
    0x3db8_3586,
    0x3dcd_db6f,
    0x3de6_0c9e,
    0x3e00_8ad0,
    0x3e0f_a5fe,
    0x3e20_87a3,
];

/// The float32 points of `torch.linspace(start, end, steps)` (torch's
/// symmetric formula).
fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                start + step * i as f32
            } else {
                end - step * (steps - 1 - i) as f32
            }
        })
        .collect()
}

/// The normalized float32 weights `exp(linspace(-eta, 0, n_window))`; for
/// the defaults (`eta = 1`, `n_window = 10`) torch's exact values.
pub fn expma_weights(eta: f64, n_window: usize) -> Vec<f64> {
    if eta == 1.0 && n_window == 10 {
        return DEFAULT_WEIGHT_BITS
            .iter()
            .map(|&w| f64::from(f32::from_bits(w)))
            .collect();
    }
    let w: Vec<f32> = linspace_f32(-(eta as f32), 0.0, n_window)
        .into_iter()
        .map(|v| f64::from(v).exp() as f32)
        .collect();
    let total: f32 = w.iter().sum();
    w.iter().map(|&v| f64::from(v / total)).collect()
}

/// Exponential moving average stopping criterion (BoTorch's
/// `ExpMAStoppingCriterion`).
///
/// Computes an exponentially weighted moving average over a window of
/// `n_window` function values and stops when the relative decrease of this
/// average between iterations (the increase if `minimize` is false) falls
/// below `rel_tol`, or after `maxiter` evaluations.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpMaStoppingCriterion {
    /// Maximum number of iterations.
    pub maxiter: usize,
    /// Whether the function is minimized.
    pub minimize: bool,
    /// Size of the moving-average window.
    pub n_window: usize,
    /// Relative tolerance.
    pub rel_tol: f64,
    weights: Vec<f64>,
    iter: usize,
    prev: Vec<Vec<f64>>,
}

impl Default for ExpMaStoppingCriterion {
    fn default() -> Self {
        Self::new(10_000, true, 10, 1.0, 1e-5)
    }
}

impl ExpMaStoppingCriterion {
    /// A criterion with BoTorch's arguments (defaults `maxiter = 10000`,
    /// `minimize = true`, `n_window = 10`, `eta = 1`, `rel_tol = 1e-5`).
    pub fn new(maxiter: usize, minimize: bool, n_window: usize, eta: f64, rel_tol: f64) -> Self {
        Self {
            maxiter,
            minimize,
            n_window,
            rel_tol,
            weights: expma_weights(eta, n_window),
            iter: 0,
            prev: Vec::new(),
        }
    }

    /// The (normalized) weights of the moving average.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The number of evaluations since the last reset.
    pub fn iterations(&self) -> usize {
        self.iter
    }
}

impl StoppingCriterion for ExpMaStoppingCriterion {
    fn evaluate(&mut self, fvals: &[f64]) -> bool {
        self.iter += 1;
        if self.iter == self.maxiter {
            return true;
        }
        // Keep the last `n_window + 1` values.
        if self.prev.len() > self.n_window {
            let excess = self.prev.len() - self.n_window;
            self.prev.drain(..excess);
        }
        self.prev.push(fvals.to_vec());
        if self.prev.len() < self.n_window + 1 {
            return false;
        }
        let k = fvals.len();
        let mut max_delta = f64::NEG_INFINITY;
        for e in 0..k {
            let mut prev_ma = 0.0;
            let mut ma = 0.0;
            for (j, w) in self.weights.iter().enumerate() {
                prev_ma += self.prev[j][e] * w;
                ma += self.prev[j + 1][e] * w;
            }
            let mut rel_delta = (prev_ma - ma) / prev_ma.abs();
            if !self.minimize {
                rel_delta = -rel_delta;
            }
            // torch.max propagates NaN.
            if rel_delta.is_nan() || max_delta.is_nan() {
                max_delta = f64::NAN;
            } else if rel_delta > max_delta {
                max_delta = rel_delta;
            }
        }
        max_delta < self.rel_tol
    }

    fn reset(&mut self) {
        self.iter = 0;
        self.prev.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_torch_formula() {
        let v = linspace_f32(-1.0, 0.0, 10);
        assert_eq!(v[0], -1.0);
        assert_eq!(v[9], 0.0);
        assert_eq!(v[1], -0.888_888_9_f32);
        assert_eq!(linspace_f32(-2.0, 0.0, 1), vec![-2.0]);
    }

    #[test]
    fn stops_on_converged_sequence() {
        let mut c = ExpMaStoppingCriterion::default();
        let mut stopped_at = None;
        for i in 0..100 {
            let f = 1.0 + (-(i as f64)).exp();
            if c.evaluate(&[f]) {
                stopped_at = Some(i + 1);
                break;
            }
        }
        // The window must fill before the criterion can hold.
        let s = stopped_at.unwrap();
        assert!(s >= 11, "{s}");
        // maxiter.
        let mut c = ExpMaStoppingCriterion::new(3, true, 10, 1.0, 1e-5);
        assert!(!c.evaluate(&[1.0]));
        assert!(!c.evaluate(&[0.5]));
        assert!(c.evaluate(&[0.25]));
        c.reset();
        assert_eq!(c.iterations(), 0);
        // NaN never stops.
        let mut c = ExpMaStoppingCriterion::new(1000, true, 2, 1.0, 1e-5);
        for _ in 0..10 {
            assert!(!c.evaluate(&[f64::NAN, 1.0]));
        }
    }
}
