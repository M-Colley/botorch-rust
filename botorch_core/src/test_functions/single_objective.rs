//! The remaining single-objective test problems of BoTorch: the
//! multi-fidelity functions (`botorch.test_functions.multi_fidelity`), the
//! sensitivity-analysis functions (`botorch.test_functions.sensitivity_analysis`)
//! and the mixed, binary, trajectory and constrained synthetic functions of
//! `botorch.test_functions.synthetic` (`AckleyMixed`, `Labs`,
//! `TrajectoryPlanning`, `ConstrainedGramacy`, `ConstrainedHartmann`,
//! `ConstrainedHartmannSmooth`, `PressureVessel`, `WeldedBeamSO`,
//! `TensionCompressionString`, `SpeedReducer`, `KeaneBumpFunction`).
//!
//! [`SingleObjectiveProblem::evaluate_true`] and
//! [`SingleObjectiveProblem::evaluate_slack_true`] validate the inputs like
//! BoTorch's `validate_inputs` (number of columns, bounds, integer values of
//! the discrete inputs) and follow BoTorch's floating-point operations
//! (constants of float32 buffers rounded to float32, `pow` with the special
//! exponents of PyTorch), so that values agree to round-off. Observation
//! noise and negation are applied by the Python layer.

use std::f64::consts::PI;

use crate::error::{BotorchError, Result};
use crate::test_functions::SyntheticTestFunction;
use crate::Matrix;

/// The problems of this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SingleObjectiveKind {
    /// `AugmentedBranin` (3 inputs, the last a fidelity).
    AugmentedBranin,
    /// `AugmentedHartmann` (7 inputs, the last a fidelity).
    AugmentedHartmann,
    /// `AugmentedRosenbrock` (`d >= 3`, the last two fidelities).
    AugmentedRosenbrock,
    /// `BoreholeMultiFidelity` (8 inputs and a fidelity index in 0..=4).
    BoreholeMultiFidelity,
    /// `WingWeightMultiFidelity` (10 inputs and a fidelity index in 0..=3).
    WingWeightMultiFidelity,
    /// `Gsobol` (parameters `a`).
    Gsobol,
    /// `Ishigami` (parameter `b`).
    Ishigami,
    /// `Morris` (20 inputs).
    Morris,
    /// `AckleyMixed` (binary inputs and 3 continuous ones; parameter `x_opt`).
    AckleyMixed,
    /// `Labs` (binary inputs, a maximization problem).
    Labs,
    /// `TrajectoryPlanning` (parameters: smooth interpolation, max step size).
    TrajectoryPlanning,
    /// `ConstrainedGramacy` (2 constraints).
    ConstrainedGramacy,
    /// `ConstrainedHartmann` (`||x||_2 <= 1`).
    ConstrainedHartmann,
    /// `ConstrainedHartmannSmooth` (`||x||_2^2 <= 1`).
    ConstrainedHartmannSmooth,
    /// `PressureVessel` (4 constraints).
    PressureVessel,
    /// `WeldedBeamSO` (6 constraints).
    WeldedBeamSO,
    /// `TensionCompressionString` (4 constraints).
    TensionCompressionString,
    /// `SpeedReducer` (11 constraints).
    SpeedReducer,
    /// `KeaneBumpFunction` (2 constraints).
    KeaneBumpFunction,
}

impl SingleObjectiveKind {
    /// All problems.
    pub const ALL: [SingleObjectiveKind; 19] = [
        Self::AugmentedBranin,
        Self::AugmentedHartmann,
        Self::AugmentedRosenbrock,
        Self::BoreholeMultiFidelity,
        Self::WingWeightMultiFidelity,
        Self::Gsobol,
        Self::Ishigami,
        Self::Morris,
        Self::AckleyMixed,
        Self::Labs,
        Self::TrajectoryPlanning,
        Self::ConstrainedGramacy,
        Self::ConstrainedHartmann,
        Self::ConstrainedHartmannSmooth,
        Self::PressureVessel,
        Self::WeldedBeamSO,
        Self::TensionCompressionString,
        Self::SpeedReducer,
        Self::KeaneBumpFunction,
    ];

    /// The BoTorch class name.
    pub fn name(self) -> &'static str {
        match self {
            Self::AugmentedBranin => "AugmentedBranin",
            Self::AugmentedHartmann => "AugmentedHartmann",
            Self::AugmentedRosenbrock => "AugmentedRosenbrock",
            Self::BoreholeMultiFidelity => "BoreholeMultiFidelity",
            Self::WingWeightMultiFidelity => "WingWeightMultiFidelity",
            Self::Gsobol => "Gsobol",
            Self::Ishigami => "Ishigami",
            Self::Morris => "Morris",
            Self::AckleyMixed => "AckleyMixed",
            Self::Labs => "Labs",
            Self::TrajectoryPlanning => "TrajectoryPlanning",
            Self::ConstrainedGramacy => "ConstrainedGramacy",
            Self::ConstrainedHartmann => "ConstrainedHartmann",
            Self::ConstrainedHartmannSmooth => "ConstrainedHartmannSmooth",
            Self::PressureVessel => "PressureVessel",
            Self::WeldedBeamSO => "WeldedBeamSO",
            Self::TensionCompressionString => "TensionCompressionString",
            Self::SpeedReducer => "SpeedReducer",
            Self::KeaneBumpFunction => "KeaneBumpFunction",
        }
    }

    /// The problem with the BoTorch class name `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// The number of outcome constraints.
    pub fn num_constraints(self) -> usize {
        match self {
            Self::ConstrainedGramacy | Self::KeaneBumpFunction => 2,
            Self::ConstrainedHartmann | Self::ConstrainedHartmannSmooth => 1,
            Self::PressureVessel | Self::TensionCompressionString => 4,
            Self::WeldedBeamSO => 6,
            Self::SpeedReducer => 11,
            _ => 0,
        }
    }
}

/// `f64` rounded to `f32` (a float32 buffer converted to float64).
const fn f32r(v: f64) -> f64 {
    v as f32 as f64
}

/// PyTorch's `pow(x, e)` for a scalar exponent (special cases 2, 3, 0.5,
/// -0.5, -1, -2; `std::pow` otherwise).
#[inline]
fn tpow(x: f64, e: f64) -> f64 {
    if e == 2.0 {
        x * x
    } else if e == 3.0 {
        x * x * x
    } else if e == 0.5 {
        x.sqrt()
    } else if e == -0.5 {
        1.0 / x.sqrt()
    } else if e == -1.0 {
        1.0 / x
    } else if e == -2.0 {
        1.0 / (x * x)
    } else {
        x.powf(e)
    }
}

const AUG_HARTMANN_ALPHA: [f64; 4] = [f32r(1.0), f32r(1.2), f32r(3.0), f32r(3.2)];
const AUG_HARTMANN_A: [[f64; 6]; 4] = [
    [f32r(10.0), f32r(3.0), f32r(17.0), f32r(3.5), f32r(1.7), f32r(8.0)],
    [f32r(0.05), f32r(10.0), f32r(17.0), f32r(0.1), f32r(8.0), f32r(14.0)],
    [f32r(3.0), f32r(3.5), f32r(1.7), f32r(10.0), f32r(17.0), f32r(8.0)],
    [f32r(17.0), f32r(8.0), f32r(0.05), f32r(10.0), f32r(0.1), f32r(14.0)],
];
const AUG_HARTMANN_P: [[f64; 6]; 4] = [
    [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
    [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
    [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
    [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
];

/// The configuration of `TrajectoryPlanning`.
#[derive(Clone, Debug)]
pub struct TrajectoryConfig {
    /// Cubic-spline (`true`) or piecewise-linear interpolation.
    pub use_smooth_interp: bool,
    /// The maximum step size.
    pub max_step_size: f64,
}

/// A single-objective test problem of this module (see the module
/// documentation).
#[derive(Clone, Debug)]
pub struct SingleObjectiveProblem {
    kind: SingleObjectiveKind,
    dim: usize,
    bounds: Matrix,
    discrete: Vec<usize>,
    params: Vec<f64>,
    trajectory: Option<TrajectoryConfig>,
    inner: Option<SyntheticTestFunction>,
}

impl SingleObjectiveProblem {
    /// The problem `kind` with dimension `dim` (for the problems of variable
    /// dimension), bounds (`2 × d`), discrete (integer-valued) inputs,
    /// parameters (`Ishigami`: `[b]`, `Gsobol`: `a`, `AckleyMixed`: `x_opt`)
    /// and the `TrajectoryPlanning` configuration.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for inconsistent dimensions, bounds
    /// or parameters.
    pub fn new(
        kind: SingleObjectiveKind,
        dim: usize,
        bounds: Matrix,
        discrete: Vec<usize>,
        params: Vec<f64>,
        trajectory: Option<TrajectoryConfig>,
    ) -> Result<Self> {
        use SingleObjectiveKind as K;
        if bounds.shape() != (2, dim) || dim == 0 {
            return Err(BotorchError::InvalidArgument(format!(
                "expected 2 x {dim} bounds, got {}x{}",
                bounds.nrows(),
                bounds.ncols()
            )));
        }
        let fixed = match kind {
            K::AugmentedBranin => Some(3),
            K::AugmentedHartmann => Some(7),
            K::BoreholeMultiFidelity => Some(9),
            K::WingWeightMultiFidelity => Some(11),
            K::Ishigami => Some(3),
            K::Morris => Some(20),
            K::ConstrainedGramacy => Some(2),
            K::PressureVessel | K::WeldedBeamSO => Some(4),
            K::TensionCompressionString => Some(3),
            K::SpeedReducer => Some(7),
            _ => None,
        };
        if fixed.is_some_and(|f| f != dim) {
            return Err(BotorchError::InvalidArgument(format!(
                "{} is {}-dimensional",
                kind.name(),
                fixed.unwrap_or(0)
            )));
        }
        let inner = match kind {
            K::AckleyMixed => Some(SyntheticTestFunction::ackley(dim)?),
            K::ConstrainedHartmann | K::ConstrainedHartmannSmooth => {
                Some(SyntheticTestFunction::hartmann(dim)?)
            }
            _ => None,
        };
        let ok_params = match kind {
            K::Ishigami => params.len() == 1,
            K::Gsobol => params.len() == dim,
            K::AckleyMixed => params.len() == dim,
            _ => params.is_empty(),
        };
        if !ok_params {
            return Err(BotorchError::InvalidArgument(format!(
                "invalid parameters for {}",
                kind.name()
            )));
        }
        if kind == K::AugmentedRosenbrock && dim < 3 {
            return Err(BotorchError::InvalidArgument(
                "AugmentedRosenbrock must be defined in at least 3 dimensions".into(),
            ));
        }
        if kind == K::TrajectoryPlanning && (dim % 2 != 0 || trajectory.is_none()) {
            return Err(BotorchError::InvalidArgument(format!(
                "dim must be even, got {dim}"
            )));
        }
        if discrete.iter().any(|&i| i >= dim) {
            return Err(BotorchError::InvalidArgument(
                "discrete index out of range".into(),
            ));
        }
        Ok(Self {
            kind,
            dim,
            bounds,
            discrete,
            params,
            trajectory,
            inner,
        })
    }

    /// The kind of problem.
    pub fn kind(&self) -> SingleObjectiveKind {
        self.kind
    }

    /// The input dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The `2 × d` bounds.
    pub fn bounds(&self) -> &Matrix {
        &self.bounds
    }

    /// The number of outcome constraints.
    pub fn num_constraints(&self) -> usize {
        self.kind.num_constraints()
    }

    /// BoTorch's `validate_inputs`.
    fn validate(&self, x: &Matrix) -> Result<()> {
        if x.ncols() != self.dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected `X` to have shape `(batch_shape) x n x d`. Got {} columns and dim={}",
                x.ncols(),
                self.dim
            )));
        }
        for i in 0..x.nrows() {
            for k in 0..self.dim {
                let v = x[(i, k)];
                if !(v >= self.bounds[(0, k)] && v <= self.bounds[(1, k)]) {
                    return Err(BotorchError::InvalidArgument(
                        "Expected `X` to be within the bounds of the test problem.".into(),
                    ));
                }
            }
            for &k in &self.discrete {
                let v = x[(i, k)];
                if v != v.round_ties_even() {
                    return Err(BotorchError::InvalidArgument(
                        "Expected `X` to have integer values for the discrete and categorical \
                         parameters."
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Noiseless, un-negated values at the rows of `x`.
    ///
    /// # Errors
    /// BoTorch's `validate_inputs` errors.
    pub fn evaluate_true(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.validate(x)?;
        Ok((0..x.nrows())
            .map(|i| {
                let p: Vec<f64> = x.row(i).iter().copied().collect();
                self.value(&p)
            })
            .collect())
    }

    /// Noiseless constraint slacks at the rows of `x` (`n × n_c`, feasible
    /// iff non-negative).
    ///
    /// # Errors
    /// BoTorch's `validate_inputs` errors; [`BotorchError::Unsupported`] for
    /// an unconstrained problem.
    pub fn evaluate_slack_true(&self, x: &Matrix) -> Result<Matrix> {
        let nc = self.num_constraints();
        if nc == 0 {
            return Err(BotorchError::Unsupported(format!(
                "{} has no outcome constraints",
                self.kind.name()
            )));
        }
        self.validate(x)?;
        let mut out = Matrix::zeros(x.nrows(), nc);
        for i in 0..x.nrows() {
            let p: Vec<f64> = x.row(i).iter().copied().collect();
            let s = self.slack(&p);
            for (c, v) in s.into_iter().enumerate() {
                out[(i, c)] = v;
            }
        }
        Ok(out)
    }

    /// The evaluation costs of the multi-fidelity problems with discrete
    /// fidelities (`WingWeightMultiFidelity`, `BoreholeMultiFidelity`).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for other problems.
    pub fn cost(&self, x: &Matrix) -> Result<Vec<f64>> {
        let (col, costs): (usize, &[f64]) = match self.kind {
            SingleObjectiveKind::WingWeightMultiFidelity => (10, &[1000.0, 100.0, 10.0, 1.0]),
            SingleObjectiveKind::BoreholeMultiFidelity => {
                (8, &[1000.0, 100.0, 10.0, 100.0, 10.0])
            }
            _ => {
                return Err(BotorchError::Unsupported(format!(
                    "{} has no cost model",
                    self.kind.name()
                )))
            }
        };
        if x.ncols() <= col {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected at least {} columns",
                col + 1
            )));
        }
        Ok((0..x.nrows())
            .map(|i| {
                let f = x[(i, col)];
                costs
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(k, _)| f == *k as f64)
                    .map_or(0.0, |(_, c)| *c)
            })
            .collect())
    }

    fn value(&self, x: &[f64]) -> f64 {
        use SingleObjectiveKind as K;
        match self.kind {
            K::AugmentedBranin => {
                let t1 = x[1] - (5.1 / (4.0 * tpow(PI, 2.0)) - 0.1 * (1.0 - x[2])) * tpow(x[0], 2.0)
                    + 5.0 / PI * x[0]
                    - 6.0;
                let t2 = 10.0 * (1.0 - 1.0 / (8.0 * PI)) * x[0].cos();
                tpow(t1, 2.0) + t2 + 10.0
            }
            K::AugmentedHartmann => {
                let inner: Vec<f64> = (0..4)
                    .map(|i| {
                        let mut s = 0.0;
                        for j in 0..6 {
                            s += AUG_HARTMANN_A[i][j]
                                * tpow(x[j] - 0.0001 * AUG_HARTMANN_P[i][j], 2.0);
                        }
                        s
                    })
                    .collect();
                let alpha1 = AUG_HARTMANN_ALPHA[0] - 0.1 * (1.0 - x[6]);
                let mut rest = 0.0;
                for i in 1..4 {
                    rest += AUG_HARTMANN_ALPHA[i] * (-inner[i]).exp();
                }
                -rest - alpha1 * (-inner[0]).exp()
            }
            K::AugmentedRosenbrock => {
                let d = x.len();
                let mut s = 0.0;
                for i in 0..d - 3 {
                    let t1 =
                        100.0 * tpow(x[i + 1] - tpow(x[i], 2.0) + 0.1 * (1.0 - x[d - 2]), 2.0);
                    let t2 = tpow(x[i] - 1.0 + 0.1 * tpow(1.0 - x[d - 1], 2.0), 2.0);
                    s += t1 + t2;
                }
                s
            }
            K::WingWeightMultiFidelity => wing_weight(x),
            K::BoreholeMultiFidelity => borehole(x),
            K::Ishigami => {
                let b = self.params[0];
                x[0].sin() + 7.0 * tpow(x[1].sin(), 2.0) + b * tpow(x[2], 4.0) * x[0].sin()
            }
            K::Gsobol => {
                let mut t = 1.0;
                for (i, a) in self.params.iter().enumerate() {
                    t = t * ((4.0 * x[i] - 2.0).abs() + a) / (1.0 + a);
                }
                t
            }
            K::Morris => morris(x),
            K::AckleyMixed => {
                let shifted: Vec<f64> = x
                    .iter()
                    .zip(&self.params)
                    .map(|(v, o)| (v - o).abs())
                    .collect();
                self.inner
                    .as_ref()
                    .expect("Ackley")
                    .value_unchecked(&shifted)
            }
            K::Labs => {
                let d = x.len();
                let s: Vec<f64> = x.iter().map(|v| 2.0 * v - 1.0).collect();
                let mut energy = 0.0;
                for k in 1..d {
                    let mut c = 0.0;
                    for i in 0..d - k {
                        c += s[i] * s[i + k];
                    }
                    energy += tpow(c, 2.0);
                }
                (d * d) as f64 / (2.0 * energy)
            }
            K::TrajectoryPlanning => {
                trajectory_cost(x, self.trajectory.as_ref().expect("configuration"))
            }
            K::ConstrainedGramacy => x[0] + x[1],
            K::ConstrainedHartmann | K::ConstrainedHartmannSmooth => self
                .inner
                .as_ref()
                .expect("Hartmann")
                .value_unchecked(x),
            K::PressureVessel => {
                let x1 = round_nearest(x[0], 0.0625, Some((0.0, 10.0)));
                let x2 = round_nearest(x[1], 0.0625, Some((0.0, 10.0)));
                let (x3, x4) = (x[2], x[3]);
                0.6224 * x1 * x3 * x4
                    + 1.7781 * x2 * tpow(x3, 2.0)
                    + 3.1661 * tpow(x1, 2.0) * x4
                    + 19.84 * tpow(x1, 2.0) * x3
            }
            K::WeldedBeamSO => {
                let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
                1.10471 * tpow(x1, 2.0) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
            }
            K::TensionCompressionString => {
                let (x1, x2, x3) = (x[0], x[1], x[2]);
                tpow(x1, 2.0) * x2 * (x3 + 2.0)
            }
            K::SpeedReducer => {
                let (x1, x2, x3, x4, x5, x6, x7) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
                0.7854 * x1 * tpow(x2, 2.0) * (3.3333 * tpow(x3, 2.0) + 14.9334 * x3 - 43.0934)
                    + -1.508 * x1 * (tpow(x6, 2.0) + tpow(x7, 2.0))
                    + 7.4777 * (tpow(x6, 3.0) + tpow(x7, 3.0))
                    + 0.7854 * (x4 * tpow(x6, 2.0) + x5 * tpow(x7, 2.0))
            }
            K::KeaneBumpFunction => {
                let mut s4 = 0.0;
                let mut p2 = 1.0;
                let mut den = 0.0;
                for (i, v) in x.iter().enumerate() {
                    let c = v.cos();
                    s4 += tpow(c, 4.0);
                    p2 *= tpow(c, 2.0);
                    den += (i + 1) as f64 * tpow(*v, 2.0);
                }
                let num = s4 - 2.0 * p2;
                let den = den.sqrt().max(1e-3);
                -(num / den).abs()
            }
        }
    }

    fn slack(&self, x: &[f64]) -> Vec<f64> {
        use SingleObjectiveKind as K;
        match self.kind {
            K::ConstrainedGramacy => {
                let (x1, x2) = (x[0], x[1]);
                let c1 = 1.5 - x1 - 2.0 * x2 - 0.5 * (2.0 * PI * (tpow(x1, 2.0) - 2.0 * x2)).sin();
                let c2 = tpow(x1, 2.0) + tpow(x2, 2.0) - 1.5;
                vec![-c1, -c2]
            }
            K::ConstrainedHartmann => {
                let mut s = 0.0;
                for v in x {
                    s += v * v;
                }
                vec![-s.sqrt() + 1.0]
            }
            K::ConstrainedHartmannSmooth => {
                let mut s = 0.0;
                for v in x {
                    s += tpow(*v, 2.0);
                }
                vec![-s + 1.0]
            }
            K::PressureVessel => {
                let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
                vec![
                    -(-x1 + 0.0193 * x3),
                    -(-x2 + 0.00954 * x3),
                    -(-PI * tpow(x3, 2.0) * x4 - (4.0 / 3.0) * PI * tpow(x3, 3.0) + 1296000.0),
                    -(x4 - 240.0),
                ]
            }
            K::WeldedBeamSO => welded_beam_slack(x),
            K::TensionCompressionString => {
                let (x1, x2, x3) = (x[0], x[1], x[2]);
                let c = [
                    1.0 - tpow(x2, 3.0) * x3 / (71785.0 * tpow(x1, 4.0)),
                    (4.0 * tpow(x2, 2.0) - x1 * x2) / (12566.0 * tpow(x1, 3.0) * (x2 - x1))
                        + 1.0 / (5108.0 * tpow(x1, 2.0))
                        - 1.0,
                    1.0 - 140.45 * x1 / (x3 * tpow(x2, 2.0)),
                    (x1 + x2) / 1.5 - 1.0,
                ];
                c.iter()
                    .map(|v| -(if *v > 100.0 { 100.0 } else { *v }))
                    .collect()
            }
            K::SpeedReducer => {
                let (x1, x2, x3, x4, x5, x6, x7) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
                let c = [
                    27.0 * (1.0 / x1) * (1.0 / tpow(x2, 2.0)) * (1.0 / x3) - 1.0,
                    397.5 * (1.0 / x1) * (1.0 / tpow(x2, 2.0)) * (1.0 / tpow(x3, 2.0)) - 1.0,
                    1.93 * (1.0 / x2) * (1.0 / x3) * tpow(x4, 3.0) * (1.0 / tpow(x6, 4.0)) - 1.0,
                    1.93 * (1.0 / x2) * (1.0 / x3) * tpow(x5, 3.0) * (1.0 / tpow(x7, 4.0)) - 1.0,
                    1.0 / (0.1 * tpow(x6, 3.0))
                        * (tpow(745.0 * x4 / (x2 * x3), 2.0) + 16.9 * 1e6).sqrt()
                        - 1100.0,
                    1.0 / (0.1 * tpow(x7, 3.0))
                        * (tpow(745.0 * x5 / (x2 * x3), 2.0) + 157.5 * 1e6).sqrt()
                        - 850.0,
                    x2 * x3 - 40.0,
                    5.0 - x1 / x2,
                    x1 / x2 - 12.0,
                    (1.5 * x6 + 1.9) / x4 - 1.0,
                    (1.1 * x7 + 1.9) / x5 - 1.0,
                ];
                c.iter().map(|v| -v).collect()
            }
            K::KeaneBumpFunction => {
                let mut prod = 1.0;
                let mut sum = 0.0;
                for v in x {
                    prod *= v;
                    sum += v;
                }
                vec![prod - 0.75, 7.5 * self.dim as f64 - sum]
            }
            _ => Vec::new(),
        }
    }
}

/// BoTorch's `round_nearest(X, increment, bounds)`.
pub fn round_nearest(x: f64, increment: f64, bounds: Option<(f64, f64)>) -> f64 {
    let mut r = (x / increment).round_ties_even() * increment;
    if let Some((lo, hi)) = bounds {
        if r < lo {
            r += increment;
        }
        if r > hi {
            r -= increment;
        }
    }
    r
}

fn wing_weight(x: &[f64]) -> f64 {
    let (s_w, w_fw, a, lambda_deg, q, lam, t_c, n_z, w_dg, w_pp, fidelity) = (
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10],
    );
    let lambda_rad = lambda_deg * PI / 180.0;
    let cos_val = lambda_rad.cos();
    let shared = 0.036
        * tpow(q, 0.006)
        * tpow(lam, 0.04)
        * tpow(a / tpow(cos_val, 2.0), 0.6)
        * tpow(100.0 * t_c / cos_val, -0.3)
        * tpow(n_z * w_dg, 0.49)
        * tpow(w_fw, 0.0035);
    if fidelity == 0.0 {
        tpow(s_w, 0.758) * shared + s_w * w_pp
    } else if fidelity == 1.0 {
        tpow(s_w, 0.758) * shared + w_pp
    } else if fidelity == 2.0 {
        tpow(s_w, 0.8) * shared + w_pp
    } else if fidelity == 3.0 {
        tpow(s_w, 0.9) * shared
    } else {
        0.0
    }
}

fn borehole(x: &[f64]) -> f64 {
    let (r_w, r, t_u, t_l, h_u, h_l, l, k_w, fidelity) =
        (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]);
    let ltu = l * t_u;
    let two_pi_t_u = 2.0 * PI * t_u;
    let log_term = (r / r_w).ln();
    let denom = log_term * tpow(r_w, 2.0) * k_w;
    let numer = two_pi_t_u * (h_u - h_l);
    let t_ratio = t_u / t_l;
    if fidelity == 0.0 {
        numer / (log_term * (1.0 + (2.0 * ltu) / denom + t_ratio))
    } else if fidelity == 1.0 {
        let n1 = two_pi_t_u * (h_u - 0.8 * h_l);
        n1 / (log_term * (1.0 + ltu / denom + t_ratio))
    } else if fidelity == 2.0 {
        numer / (log_term * (1.0 + (8.0 * ltu) / denom + 0.75 * t_ratio))
    } else if fidelity == 3.0 {
        let lt = (4.0 * r / r_w).ln();
        let n3 = two_pi_t_u * (1.09 * h_u - h_l);
        n3 / (lt * (1.0 + (3.0 * ltu) / denom + t_ratio))
    } else if fidelity == 4.0 {
        let lt = (2.0 * r / r_w).ln();
        let n4 = two_pi_t_u * (1.05 * h_u - h_l);
        n4 / (lt * (1.0 + (3.0 * ltu) / denom + t_ratio))
    } else {
        0.0
    }
}

fn morris(x: &[f64]) -> f64 {
    let d = 20;
    let w: Vec<f64> = (0..d)
        .map(|i| {
            if [2, 4, 6].contains(&i) {
                2.0 * (1.1 * x[i] / (x[i] + 0.1) - 0.5)
            } else {
                2.0 * (x[i] - 0.5)
            }
        })
        .collect();
    let mut t1 = 0.0;
    for (i, wi) in w.iter().enumerate() {
        let beta = if i < 10 {
            20.0
        } else if (i + 1) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        t1 += beta * wi;
    }
    let mut t2 = 0.0;
    let mut t3 = 0.0;
    for i in 0..d {
        for j in i + 1..d {
            let beta_ij = if i < 6 || j < 6 {
                -15.0
            } else if (i + j + 2) % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            t2 += beta_ij * w[i] * w[j];
            for k in j + 1..d {
                let beta_ijk = if i < 5 || j < 5 || k < 5 { -10.0 } else { 0.0 };
                t3 += beta_ijk * w[i] * w[j] * w[k];
            }
        }
    }
    let t4 = 5.0 * w[0] * w[1] * w[2] * w[3];
    t1 + t2 + t3 + t4
}

fn welded_beam_slack(x: &[f64]) -> Vec<f64> {
    let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
    let p = 6000.0;
    let l = 14.0;
    let e = 30e6;
    let g = 12e6;
    let t_max = 13600.0;
    let s_max = 30000.0;
    let d_max = 0.25;
    let m = p * (l + x2 / 2.0);
    let r = (0.25 * (tpow(x2, 2.0) + tpow(x1 + x3, 2.0))).sqrt();
    let j = 2.0 * 2f64.sqrt() * x1 * x2 * (tpow(x2, 2.0) / 12.0 + 0.25 * tpow(x1 + x3, 2.0));
    let p_c = 4.013 * e * x3 * tpow(x4, 3.0) * 6.0 / tpow(l, 2.0)
        * (1.0 - 0.25 * x3 * (e / g).sqrt() / l);
    let t1 = p / (2f64.sqrt() * x1 * x2);
    let t2 = m * r / j;
    let t = (tpow(t1, 2.0) + t1 * t2 * x2 / r + tpow(t2, 2.0)).sqrt();
    let s = 6.0 * p * l / (x4 * tpow(x3, 2.0));
    let d = 4.0 * p * tpow(l, 3.0) / (e * tpow(x3, 3.0) * x4);
    vec![
        -(t - t_max),
        -(s - s_max),
        -(x1 - x4),
        -(0.10471 * tpow(x1, 2.0) + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0),
        -(d - d_max),
        -(p - p_c),
    ]
}

/// `torch.linspace(start, end, steps)` in single precision (as `f64`).
fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f64> {
    if steps == 1 {
        return vec![f64::from(start)];
    }
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            f64::from(if i < half {
                start + step * i as f32
            } else {
                end - step * (steps - i - 1) as f32
            })
        })
        .collect()
}

/// `torch.linspace(start, end, steps)` in double precision.
fn linspace_f64(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f64;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                start + step * i as f64
            } else {
                end - step * (steps - i - 1) as f64
            }
        })
        .collect()
}

fn norm2(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// The obstacles of `TrajectoryPlanning`: `(low, high)` corners of the
/// boxes, from torch's float32 grid as in BoTorch.
fn obstacles() -> Vec<([f64; 2], [f64; 2])> {
    let grid = linspace_f32(0.0, 1.0, 6);
    let mut centers = Vec::new();
    for gi in &grid {
        for gj in &grid {
            centers.push([*gi as f32, *gj as f32]);
        }
    }
    let n = centers.len();
    centers[1..n - 1]
        .iter()
        .map(|c| {
            let lo = [f64::from(c[0] - 0.05f32), f64::from(c[1] - 0.05f32)];
            let hi = [f64::from(c[0] + 0.05f32), f64::from(c[1] + 0.05f32)];
            (lo, hi)
        })
        .collect()
}

fn trajectory_cost(params: &[f64], cfg: &TrajectoryConfig) -> f64 {
    let nw = params.len() / 2;
    let start = [0.05, 0.05];
    let goal = [0.95, 0.95];
    let mut waypoints = vec![start];
    let mut cur = start;
    for i in 0..nw {
        let to_goal = [goal[0] - cur[0], goal[1] - cur[1]];
        let dist = norm2(to_goal);
        if dist <= 1e-6 {
            break;
        }
        let pert = [
            (params[2 * i] - 0.5) * 2.0 * cfg.max_step_size,
            (params[2 * i + 1] - 0.5) * 2.0 * cfg.max_step_size,
        ];
        let ideal = (dist / (nw - i) as f64).min(cfg.max_step_size);
        let mut step = [
            to_goal[0] / dist * ideal + pert[0],
            to_goal[1] / dist * ideal + pert[1],
        ];
        let sn = norm2(step);
        if sn > cfg.max_step_size {
            let f = cfg.max_step_size / sn;
            step = [step[0] * f, step[1] * f];
        }
        cur = [
            (cur[0] + step[0]).clamp(0.0, 1.0),
            (cur[1] + step[1]).clamp(0.0, 1.0),
        ];
        waypoints.push(cur);
    }
    waypoints.push(goal);
    let points = interpolate_trajectory(&waypoints, cfg.use_smooth_interp, 1000);
    let obs = obstacles();
    let collides: Vec<bool> = points
        .iter()
        .map(|p| {
            obs.iter()
                .any(|(lo, hi)| p[0] >= lo[0] && p[1] >= lo[1] && p[0] <= hi[0] && p[1] <= hi[1])
        })
        .collect();
    let mut total = 0.0;
    for k in 0..points.len() - 1 {
        let seg = norm2([points[k + 1][0] - points[k][0], points[k + 1][1] - points[k][1]]);
        let w = 0.5 * (f64::from(u8::from(collides[k])) + f64::from(u8::from(collides[k + 1])));
        total += seg * w;
    }
    total * 20.0
}

fn interpolate_trajectory(waypoints: &[[f64; 2]], smooth: bool, n_samples: usize) -> Vec<[f64; 2]> {
    if smooth {
        let mut unique = vec![waypoints[0]];
        for k in 1..waypoints.len() {
            let d = norm2([
                waypoints[k][0] - waypoints[k - 1][0],
                waypoints[k][1] - waypoints[k - 1][1],
            ]);
            if d > 1e-6 {
                unique.push(waypoints[k]);
            }
        }
        if unique.len() >= 4 {
            if let Some(points) = spline_points(&unique, n_samples) {
                return points;
            }
        }
    }
    let n_seg = waypoints.len() - 1;
    let t_all = linspace_f64(0.0, 1.0, n_samples / n_seg);
    let t = &t_all[..t_all.len().saturating_sub(1)];
    let mut out = Vec::with_capacity(n_seg * t.len());
    for s in 0..n_seg {
        for &ti in t {
            out.push([
                (1.0 - ti) * waypoints[s][0] + ti * waypoints[s + 1][0],
                (1.0 - ti) * waypoints[s][1] + ti * waypoints[s + 1][1],
            ]);
        }
    }
    out
}

/// The B-spline basis functions of degree 3 that are non-zero at `x` in
/// the knot span `span` (de Boor's recursion).
fn basis(knots: &[f64], span: usize, x: f64) -> [f64; 4] {
    let mut n = [0.0; 4];
    let mut left = [0.0; 4];
    let mut right = [0.0; 4];
    n[0] = 1.0;
    for j in 1..=3 {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// The knot span of `x` (FITPACK's `splev` interval search: the last span
/// with `t[l] <= x`, at most `n - k - 2`).
fn find_span(knots: &[f64], x: f64) -> usize {
    let k = 3;
    let n = knots.len();
    let mut l = k;
    while l < n - k - 2 && x >= knots[l + 1] {
        l += 1;
    }
    l
}

/// SciPy's `splev(linspace(0, 1, n_samples), splprep(points.T, k=3, s=0))`:
/// the interpolating parametric cubic spline through `points` with
/// chord-length parameters and FITPACK's knots for interpolation, evaluated
/// at torch's float32 `linspace(0, 1, n_samples)` and clamped to `[0, 1]`.
fn spline_points(points: &[[f64; 2]], n_samples: usize) -> Option<Vec<[f64; 2]>> {
    let m = points.len();
    let mut v = vec![0.0; m];
    for i in 1..m {
        v[i] = v[i - 1]
            + ((points[i][0] - points[i - 1][0]).powi(2) + (points[i][1] - points[i - 1][1]).powi(2))
                .sqrt();
    }
    let total = v[m - 1];
    let u: Vec<f64> = v.iter().map(|x| x / total).collect();
    let mut knots = vec![u[0]; 4];
    knots.extend_from_slice(&u[2..m - 2]);
    knots.extend(std::iter::repeat_n(u[m - 1], 4));
    // Collocation: B(u_i) c = x_i.
    let mut a = Matrix::zeros(m, m);
    for (i, &ui) in u.iter().enumerate() {
        let span = find_span(&knots, ui);
        let b = basis(&knots, span, ui);
        for (r, bv) in b.iter().enumerate() {
            a[(i, span - 3 + r)] = *bv;
        }
    }
    let rhs = Matrix::from_fn(m, 2, |i, c| points[i][c]);
    let coef = a.lu().solve(&rhs)?;
    let ts = linspace_f32(0.0, 1.0, n_samples);
    Some(
        ts.into_iter()
            .map(|t| {
                let span = find_span(&knots, t);
                let b = basis(&knots, span, t);
                let mut p = [0.0; 2];
                for (r, bv) in b.iter().enumerate() {
                    p[0] += bv * coef[(span - 3 + r, 0)];
                    p[1] += bv * coef[(span - 3 + r, 1)];
                }
                [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)]
            })
            .collect(),
    )
}

/// The default maximum step size of `TrajectoryPlanning`:
/// `1.5 * ||goal - start|| / num_waypoints`.
pub fn default_max_step_size(dim: usize) -> f64 {
    1.5 * norm2([0.9, 0.9]) / (dim / 2) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem(kind: SingleObjectiveKind, dim: usize, lo: f64, hi: f64) -> SingleObjectiveProblem {
        let bounds = Matrix::from_fn(2, dim, |r, _| if r == 0 { lo } else { hi });
        let params = match kind {
            SingleObjectiveKind::Ishigami => vec![0.1],
            SingleObjectiveKind::Gsobol | SingleObjectiveKind::AckleyMixed => vec![0.0; dim],
            _ => vec![],
        };
        let traj = (kind == SingleObjectiveKind::TrajectoryPlanning).then(|| TrajectoryConfig {
            use_smooth_interp: true,
            max_step_size: default_max_step_size(dim),
        });
        SingleObjectiveProblem::new(kind, dim, bounds, vec![], params, traj).unwrap()
    }

    #[test]
    fn known_optima() {
        let p = problem(SingleObjectiveKind::AugmentedBranin, 3, -5.0, 15.0);
        let x = Matrix::from_row_slice(1, 3, &[PI, 2.275, 1.0]);
        assert!((p.evaluate_true(&x).unwrap()[0] - 0.397887).abs() < 1e-5);
        let p = problem(SingleObjectiveKind::AugmentedRosenbrock, 4, -5.0, 10.0);
        let x = Matrix::from_element(1, 4, 1.0);
        assert_eq!(p.evaluate_true(&x).unwrap()[0], 0.0);
        let p = problem(SingleObjectiveKind::AugmentedHartmann, 7, 0.0, 1.0);
        let x = Matrix::from_row_slice(
            1,
            7,
            &[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0],
        );
        assert!((p.evaluate_true(&x).unwrap()[0] + 3.32237).abs() < 1e-4);
    }

    #[test]
    fn validation_and_slacks() {
        let p = problem(SingleObjectiveKind::ConstrainedGramacy, 2, 0.0, 1.0);
        assert!(p.evaluate_true(&Matrix::from_element(1, 2, 2.0)).is_err());
        let s = p
            .evaluate_slack_true(&Matrix::from_row_slice(1, 2, &[0.1954, 0.4044]))
            .unwrap();
        assert_eq!(s.ncols(), 2);
        assert_eq!(round_nearest(0.03, 0.0625, Some((0.0625, 1.0))), 0.0625);
        assert_eq!(round_nearest(0.09, 0.0625, None), 0.0625);
    }

    #[test]
    fn trajectory_is_finite() {
        let p = problem(SingleObjectiveKind::TrajectoryPlanning, 10, 0.0, 1.0);
        let x = Matrix::from_element(1, 10, 0.5);
        let v = p.evaluate_true(&x).unwrap()[0];
        assert!(v.is_finite() && v >= 0.0);
    }
}
