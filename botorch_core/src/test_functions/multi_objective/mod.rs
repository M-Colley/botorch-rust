//! Multi-objective test problems (`botorch.test_functions.multi_objective` and
//! `botorch.test_functions.multi_objective_multi_fidelity`).
//!
//! A [`MultiObjectiveTestProblem`] has the semantics of BoTorch's
//! `MultiObjectiveTestProblem` (and, for constrained problems, of
//! `ConstrainedBaseTestProblem`):
//!
//! * [`MultiObjectiveTestProblem::evaluate_true`] returns the noiseless,
//!   un-negated objectives (`n × M`) and an error for points outside the
//!   bounds;
//! * [`MultiObjectiveTestProblem::forward`] adds Gaussian observation noise
//!   (one standard deviation, or one per objective) and then negates the
//!   objectives if `negate` is set;
//! * [`MultiObjectiveTestProblem::evaluate_slack_true`] returns the constraint
//!   slacks (`n × n_c`, feasible iff non-negative), which are never negated;
//!   [`MultiObjectiveTestProblem::evaluate_slack`] adds noise with
//!   `constraint_noise_std` and [`MultiObjectiveTestProblem::is_feasible`]
//!   checks that all slacks are non-negative;
//! * [`MultiObjectiveTestProblem::ref_point`] is negated with `negate`,
//!   [`MultiObjectiveTestProblem::max_hv`] is not.
//!
//! The evaluations follow BoTorch's floating-point operations, so that values
//! agree to a few units in the last place. Where BoTorch's definitions are
//! wrong, the corrected definitions are implemented (see
//! [`MultiObjectiveKind::deviation`]).

mod branin_currin;
mod dh;
mod dtlz;
mod engineering;
mod gmm;
mod multi_fidelity;
mod penicillin;
mod toy_robust;
mod zdt;

use std::sync::Mutex;

use rand::rngs::Xoshiro256PlusPlus;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::error::{BotorchError, Result};
use crate::test_functions::SyntheticTestFunction;
use crate::utils::sampling::random_seed;
use crate::Matrix;

/// The multi-objective test problems of BoTorch (concrete classes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MultiObjectiveKind {
    /// `BraninCurrin`: the rescaled Branin and the Currin function on `[0, 1]^2`.
    BraninCurrin,
    /// `DH1`, `d ≥ 2` inputs (Deb & Gupta, 2005).
    Dh1,
    /// `DH2`: DH1 with `beta = 10`.
    Dh2,
    /// `DH3`, `d ≥ 3` inputs.
    Dh3,
    /// `DH4`, `d ≥ 3` inputs.
    Dh4,
    /// `DTLZ1`, `M` objectives, `d > M` inputs (Deb et al., 2005).
    Dtlz1,
    /// `DTLZ2`, `M` objectives, `d > M` inputs.
    Dtlz2,
    /// `DTLZ3`, `M` objectives, `d > M` inputs.
    Dtlz3,
    /// `DTLZ4`, `M` objectives, `d > M` inputs.
    Dtlz4,
    /// `DTLZ5`, `M` objectives, `d > M` inputs.
    Dtlz5,
    /// `DTLZ7`, `M` objectives, `d > M` inputs.
    Dtlz7,
    /// `GMM`: `M ∈ {2, 3, 4}` Gaussian mixtures on `[0, 1]^2`.
    Gmm,
    /// `Penicillin`: a fermentation simulator, 7 inputs and 3 objectives.
    Penicillin,
    /// `ToyRobust`: a one-dimensional problem with two objectives.
    ToyRobust,
    /// `VehicleSafety`: 5 inputs, 3 objectives (Tanabe & Ishibuchi, 2020).
    VehicleSafety,
    /// `ZDT1`, two objectives, `d ≥ 2` inputs (Zitzler et al., 2000).
    Zdt1,
    /// `ZDT2`, two objectives, `d ≥ 2` inputs.
    Zdt2,
    /// `ZDT3`, two objectives, `d ≥ 2` inputs.
    Zdt3,
    /// `CarSideImpact`: 7 inputs, 4 objectives (Tanabe & Ishibuchi, 2020).
    CarSideImpact,
    /// `MOMFBraninCurrin` (`multi_objective_multi_fidelity`): 2 inputs and a fidelity.
    MomfBraninCurrin,
    /// `MOMFPark` (`multi_objective_multi_fidelity`): 4 inputs and a fidelity.
    MomfPark,
}

impl MultiObjectiveKind {
    /// All problems, in the order of BoTorch's modules.
    pub const ALL: [MultiObjectiveKind; 21] = [
        Self::BraninCurrin,
        Self::Dh1,
        Self::Dh2,
        Self::Dh3,
        Self::Dh4,
        Self::Dtlz1,
        Self::Dtlz2,
        Self::Dtlz3,
        Self::Dtlz4,
        Self::Dtlz5,
        Self::Dtlz7,
        Self::Gmm,
        Self::Penicillin,
        Self::ToyRobust,
        Self::VehicleSafety,
        Self::Zdt1,
        Self::Zdt2,
        Self::Zdt3,
        Self::CarSideImpact,
        Self::MomfBraninCurrin,
        Self::MomfPark,
    ];

    /// The BoTorch class name.
    pub fn name(self) -> &'static str {
        match self {
            Self::BraninCurrin => "BraninCurrin",
            Self::Dh1 => "DH1",
            Self::Dh2 => "DH2",
            Self::Dh3 => "DH3",
            Self::Dh4 => "DH4",
            Self::Dtlz1 => "DTLZ1",
            Self::Dtlz2 => "DTLZ2",
            Self::Dtlz3 => "DTLZ3",
            Self::Dtlz4 => "DTLZ4",
            Self::Dtlz5 => "DTLZ5",
            Self::Dtlz7 => "DTLZ7",
            Self::Gmm => "GMM",
            Self::Penicillin => "Penicillin",
            Self::ToyRobust => "ToyRobust",
            Self::VehicleSafety => "VehicleSafety",
            Self::Zdt1 => "ZDT1",
            Self::Zdt2 => "ZDT2",
            Self::Zdt3 => "ZDT3",
            Self::CarSideImpact => "CarSideImpact",
            Self::MomfBraninCurrin => "MOMFBraninCurrin",
            Self::MomfPark => "MOMFPark",
        }
    }

    /// The problem with the BoTorch class name `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// Whether the problem is defined in
    /// `botorch.test_functions.multi_objective_multi_fidelity`.
    pub fn is_multi_fidelity(self) -> bool {
        matches!(self, Self::MomfBraninCurrin | Self::MomfPark)
    }

    /// The number of outcome constraints (0 for unconstrained problems).
    pub fn num_constraints(self) -> usize {
        0
    }

    /// Whether the problem has outcome constraints (BoTorch's
    /// `ConstrainedBaseTestProblem`).
    pub fn is_constrained(self) -> bool {
        self.num_constraints() > 0
    }

    /// The input dimension of problems with a fixed dimension.
    pub fn fixed_dim(self) -> Option<usize> {
        match self {
            Self::BraninCurrin | Self::Gmm => Some(2),
            Self::ToyRobust => Some(1),
            Self::VehicleSafety => Some(5),
            Self::CarSideImpact | Self::Penicillin => Some(7),
            Self::MomfBraninCurrin => Some(3),
            Self::MomfPark => Some(5),
            _ => None,
        }
    }

    /// Whether the number of objectives is an argument (`num_objectives`).
    fn has_num_objectives_argument(self) -> bool {
        self.is_dtlz() || self.is_zdt() || self == Self::Gmm
    }

    fn is_dtlz(self) -> bool {
        matches!(
            self,
            Self::Dtlz1 | Self::Dtlz2 | Self::Dtlz3 | Self::Dtlz4 | Self::Dtlz5 | Self::Dtlz7
        )
    }

    fn is_zdt(self) -> bool {
        matches!(self, Self::Zdt1 | Self::Zdt2 | Self::Zdt3)
    }

    /// The number of objectives of problems with a fixed number.
    pub fn fixed_num_objectives(self) -> Option<usize> {
        match self {
            _ if self.has_num_objectives_argument() => None,
            Self::VehicleSafety | Self::Penicillin => Some(3),
            Self::CarSideImpact => Some(4),
            _ => Some(2),
        }
    }

    /// BoTorch's default number of objectives (the fixed one, or the default
    /// of the `num_objectives` argument).
    pub fn default_num_objectives(self) -> usize {
        self.fixed_num_objectives().unwrap_or(2)
    }

    /// A short description of how botorch-rust deviates from BoTorch's main
    /// branch for this problem, with the BoTorch pull request that corrects
    /// it (`None` if it does not).
    pub fn deviation(self) -> Option<&'static str> {
        match self {
            Self::BraninCurrin => Some(
                "meta-pytorch/botorch#3402: max_hv is 59.4066 (BoTorch's NSGA-II estimate \
                 59.3601 is exceeded by evaluated points)",
            ),
            Self::Dh4 => Some(
                "meta-pytorch/botorch#3402: max_hv depends on the dimension (h < 0 for \
                 x_0 > 0.98534, where f_1 decreases with g)",
            ),
            Self::Dtlz1 => Some(
                "meta-pytorch/botorch#3402: max_hv subtracts the volume 0.5^M / M! of the \
                 simplex below the Pareto front (BoTorch: 1 / 2^M)",
            ),
            Self::Dtlz4 => Some(
                "meta-pytorch/botorch#3402: the position variables are mapped to x_i^100 \
                 (BoTorch's DTLZ4 equals DTLZ2)",
            ),
            Self::Zdt3 => Some(
                "meta-pytorch/botorch#3402: the second objective is multiplied by g, as in \
                 Zitzler et al. (2000)",
            ),
            Self::Gmm => Some(
                "meta-pytorch/botorch#3405: float64 parameters (BoTorch creates them in \
                 float32, so it evaluates float64 inputs in float32 and returns float32)",
            ),
            _ => None,
        }
    }

    /// Validates and resolves the constructor arguments `dim` and
    /// `num_objectives` like BoTorch's constructors (`None`: not given), with
    /// BoTorch's errors. The arguments are signed so that invalid negative
    /// values are reported like BoTorch does.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] (BoTorch's `ValueError`) or
    /// [`BotorchError::Unsupported`] (BoTorch's `NotImplementedError` /
    /// `UnsupportedError`) for invalid sizes.
    pub fn resolve_sizes(
        self,
        dim: Option<i64>,
        num_objectives: Option<i64>,
    ) -> Result<(usize, usize)> {
        let name = self.name();
        let m = match (self.fixed_num_objectives(), num_objectives) {
            (Some(f), Some(m)) if m != f as i64 => {
                return Err(BotorchError::InvalidArgument(format!(
                    "{name} has {f} objectives, got num_objectives = {m}"
                )))
            }
            (Some(f), _) => f as i64,
            (None, Some(m)) => m,
            (None, None) => self.default_num_objectives() as i64,
        };
        let d = match (self.fixed_dim(), dim) {
            (Some(f), Some(d)) if d != f as i64 => {
                return Err(BotorchError::InvalidArgument(format!(
                    "{name} is {f}-dimensional, got dim = {d}"
                )))
            }
            (Some(f), _) => f as i64,
            (None, Some(d)) => d,
            (None, None) => {
                return Err(BotorchError::InvalidArgument(format!(
                    "{name} requires the input dimension `dim`"
                )))
            }
        };
        let min_dim = match self {
            Self::Dh1 | Self::Dh2 => 2,
            Self::Dh3 | Self::Dh4 => 3,
            _ => 1,
        };
        if matches!(self, Self::Dh1 | Self::Dh2 | Self::Dh3 | Self::Dh4) && d < min_dim {
            return Err(BotorchError::InvalidArgument(format!(
                "dim must be >= {min_dim}, but got dim={d}!"
            )));
        }
        if self.is_dtlz() && d <= m {
            return Err(BotorchError::InvalidArgument(format!(
                "dim must be > num_objectives, but got {d} and {m}."
            )));
        }
        if self == Self::Gmm && !(2..=4).contains(&m) {
            return Err(BotorchError::Unsupported(
                "GMM only currently supports 2 to 4 objectives.".into(),
            ));
        }
        if self.is_zdt() {
            if m != 2 {
                return Err(BotorchError::Unsupported(format!(
                    "{name} currently only supports 2 objectives."
                )));
            }
            if d < m {
                return Err(BotorchError::InvalidArgument(format!(
                    "dim must be >= num_objectives, but got {d} and {m}"
                )));
            }
        }
        if d < 1 || m < 1 {
            return Err(BotorchError::InvalidArgument(format!(
                "dim and num_objectives must be positive, got {d} and {m}"
            )));
        }
        Ok((d as usize, m as usize))
    }
}

/// The observation-noise standard deviation of the objectives or of the
/// constraint slacks (BoTorch's `noise_std` / `constraint_noise_std`).
#[derive(Clone, Debug, PartialEq)]
pub enum NoiseStd {
    /// One standard deviation for all outputs.
    Scalar(f64),
    /// One standard deviation per output (BoTorch's list argument).
    PerOutput(Vec<f64>),
}

impl NoiseStd {
    fn expand(&self, n: usize) -> Vec<f64> {
        match self {
            Self::Scalar(s) => vec![*s; n],
            Self::PerOutput(v) => v.clone(),
        }
    }
}

/// Points per evaluation from which evaluations run in parallel (rayon).
const PARALLEL_MIN_POINTS: usize = 4096;
/// The same for the (much more expensive) Penicillin simulation.
const PARALLEL_MIN_POINTS_SIMULATION: usize = 32;

/// A multi-objective test problem (BoTorch's `MultiObjectiveTestProblem`,
/// constrained or not) with bounds, optional negation and observation noise.
#[derive(Debug)]
pub struct MultiObjectiveTestProblem {
    kind: MultiObjectiveKind,
    dim: usize,
    num_objectives: usize,
    bounds: Matrix,
    negate: bool,
    noise_std: Option<NoiseStd>,
    constraint_noise_std: Option<NoiseStd>,
    /// Branin for `BraninCurrin`, Levy for `ToyRobust`.
    component: Option<SyntheticTestFunction>,
    rng: Mutex<Xoshiro256PlusPlus>,
}

impl Clone for MultiObjectiveTestProblem {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind,
            dim: self.dim,
            num_objectives: self.num_objectives,
            bounds: self.bounds.clone(),
            negate: self.negate,
            noise_std: self.noise_std.clone(),
            constraint_noise_std: self.constraint_noise_std.clone(),
            component: self.component.clone(),
            rng: Mutex::new(self.rng.lock().unwrap_or_else(|e| e.into_inner()).clone()),
        }
    }
}

impl MultiObjectiveTestProblem {
    /// Creates the problem `kind` with input dimension `dim` and
    /// `num_objectives` objectives (`None`: the fixed value or BoTorch's
    /// default; problems without a default dimension require `dim`).
    ///
    /// # Errors
    /// See [`MultiObjectiveKind::resolve_sizes`].
    pub fn new(
        kind: MultiObjectiveKind,
        dim: Option<usize>,
        num_objectives: Option<usize>,
    ) -> Result<Self> {
        let signed = |v: usize| i64::try_from(v).unwrap_or(i64::MAX);
        let (dim, num_objectives) =
            kind.resolve_sizes(dim.map(signed), num_objectives.map(signed))?;
        let pairs = default_bounds(kind, dim);
        let bounds = Matrix::from_fn(2, dim, |i, k| if i == 0 { pairs[k].0 } else { pairs[k].1 });
        let component = match kind {
            MultiObjectiveKind::BraninCurrin => Some(SyntheticTestFunction::branin()),
            MultiObjectiveKind::ToyRobust => Some(SyntheticTestFunction::levy(2)?),
            _ => None,
        };
        Ok(Self {
            kind,
            dim,
            num_objectives,
            bounds,
            negate: false,
            noise_std: None,
            constraint_noise_std: None,
            component,
            rng: Mutex::new(Xoshiro256PlusPlus::seed_from_u64(random_seed())),
        })
    }

    /// Sets `negate` (BoTorch's `negate=True` negates the objectives and the
    /// reference point, turning minimization into maximization problems).
    pub fn with_negate(mut self, negate: bool) -> Self {
        self.negate = negate;
        self
    }

    /// Sets the standard deviation of the observation noise of the objectives
    /// in [`MultiObjectiveTestProblem::forward`].
    ///
    /// # Errors
    /// [`BotorchError::InputData`] (BoTorch's message) if a per-objective
    /// list does not have one entry per objective.
    pub fn with_noise_std(mut self, noise_std: NoiseStd) -> Result<Self> {
        if let NoiseStd::PerOutput(v) = &noise_std {
            if v.len() != self.num_objectives {
                return Err(BotorchError::InputData(format!(
                    "If specified as a list, length of noise_std ({}) must match the number \
                     of objectives ({})",
                    v.len(),
                    self.num_objectives
                )));
            }
        }
        self.noise_std = Some(noise_std);
        Ok(self)
    }

    /// Sets the standard deviation of the observation noise of the constraint
    /// slacks in [`MultiObjectiveTestProblem::evaluate_slack`].
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for unconstrained problems and
    /// [`BotorchError::InputData`] if a list does not have one entry per
    /// constraint (BoTorch fails when evaluating the slack).
    pub fn with_constraint_noise_std(mut self, noise_std: NoiseStd) -> Result<Self> {
        let n_c = self.num_constraints();
        if n_c == 0 {
            return Err(BotorchError::Unsupported(format!(
                "{} has no constraints.",
                self.name()
            )));
        }
        if let NoiseStd::PerOutput(v) = &noise_std {
            if v.len() != n_c {
                return Err(BotorchError::InputData(format!(
                    "If specified as a list, length of constraint_noise_std ({}) must match \
                     the number of constraints ({n_c})",
                    v.len()
                )));
            }
        }
        self.constraint_noise_std = Some(noise_std);
        Ok(self)
    }

    /// Seeds the random number generator of the observation noise (by
    /// default it is seeded randomly, like torch's global RNG).
    pub fn with_seed(self, seed: u64) -> Self {
        *self.rng.lock().unwrap_or_else(|e| e.into_inner()) =
            Xoshiro256PlusPlus::seed_from_u64(seed);
        self
    }

    /// The kind of problem.
    pub fn kind(&self) -> MultiObjectiveKind {
        self.kind
    }

    /// The BoTorch class name.
    pub fn name(&self) -> &'static str {
        self.kind.name()
    }

    /// The input dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of objectives `M`.
    pub fn num_objectives(&self) -> usize {
        self.num_objectives
    }

    /// The number of outcome constraints (0 for unconstrained problems).
    pub fn num_constraints(&self) -> usize {
        self.kind.num_constraints()
    }

    /// The `2 × d` bounds (lower bounds in row 0).
    pub fn bounds(&self) -> &Matrix {
        &self.bounds
    }

    /// Whether the objectives are negated.
    pub fn negate(&self) -> bool {
        self.negate
    }

    /// The observation-noise standard deviation of the objectives, if any.
    pub fn noise_std(&self) -> Option<&NoiseStd> {
        self.noise_std.as_ref()
    }

    /// The observation-noise standard deviation of the constraint slacks.
    pub fn constraint_noise_std(&self) -> Option<&NoiseStd> {
        self.constraint_noise_std.as_ref()
    }

    /// BoTorch's `_ref_point`: the reference point of the un-negated problem.
    pub fn raw_ref_point(&self) -> Vec<f64> {
        raw_ref_point(self.kind, self.num_objectives)
    }

    /// The reference point for hypervolume computations (negated if `negate`).
    pub fn ref_point(&self) -> Vec<f64> {
        let mut r = self.raw_ref_point();
        if self.negate {
            r.iter_mut().for_each(|v| *v = -*v);
        }
        r
    }

    /// The maximum hypervolume with respect to the reference point, if known.
    pub fn max_hv(&self) -> Option<f64> {
        max_hv(self.kind, self.dim, self.num_objectives)
    }

    /// Validates row-major points with `ncols` columns and returns their number.
    fn validate(&self, x: &[f64], ncols: usize) -> Result<usize> {
        if ncols != self.dim || !x.len().is_multiple_of(self.dim) {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected `X` to have shape `(batch_shape) x n x d`. Got {ncols} columns and \
                 dim={}",
                self.dim
            )));
        }
        for row in x.chunks(self.dim) {
            for (k, &v) in row.iter().enumerate() {
                if !(v >= self.bounds[(0, k)] && v <= self.bounds[(1, k)]) {
                    return Err(BotorchError::InvalidArgument(
                        "Expected `X` to be within the bounds of the test problem.".into(),
                    ));
                }
            }
        }
        Ok(x.len() / self.dim)
    }

    /// Applies `f` to every point (row) of `x`, writing `width` outputs per
    /// point (in parallel for many points).
    fn map_points<F>(&self, x: &[f64], n: usize, width: usize, f: F) -> Vec<f64>
    where
        F: Fn(&[f64], &mut [f64]) + Sync + Send,
    {
        let mut out = vec![0.0; n * width];
        let min_points = match self.kind {
            MultiObjectiveKind::Penicillin => PARALLEL_MIN_POINTS_SIMULATION,
            _ => PARALLEL_MIN_POINTS,
        };
        if n >= min_points {
            out.par_chunks_mut(width)
                .zip(x.par_chunks(self.dim))
                .for_each(|(o, xi)| f(xi, o));
        } else {
            out.chunks_mut(width)
                .zip(x.chunks(self.dim))
                .for_each(|(o, xi)| f(xi, o));
        }
        out
    }

    /// The objectives at one point (no bounds check).
    fn objectives_at(&self, x: &[f64], out: &mut [f64]) {
        use MultiObjectiveKind as K;
        match self.kind {
            K::BraninCurrin => {
                branin_currin::evaluate(self.component.as_ref().expect("Branin"), x, out)
            }
            K::Dh1 => dh::dh1(x, 1.0, out),
            K::Dh2 => dh::dh1(x, 10.0, out),
            K::Dh3 => dh::dh3(x, false, out),
            K::Dh4 => dh::dh3(x, true, out),
            K::Dtlz1 => dtlz::dtlz1(x, self.num_objectives, out),
            K::Dtlz2 => dtlz::dtlz2(x, self.num_objectives, out),
            K::Dtlz3 => dtlz::dtlz3(x, self.num_objectives, out),
            K::Dtlz4 => dtlz::dtlz4(x, self.num_objectives, out),
            K::Dtlz5 => dtlz::dtlz5(x, self.num_objectives, out),
            K::Dtlz7 => dtlz::dtlz7(x, self.num_objectives, out),
            K::Zdt1 => zdt::zdt1(x, out),
            K::Zdt2 => zdt::zdt2(x, out),
            K::Zdt3 => zdt::zdt3(x, out),
            K::Gmm => gmm::evaluate(x, self.num_objectives, out),
            K::Penicillin => penicillin::evaluate(x, out),
            K::ToyRobust => toy_robust::evaluate(self.component.as_ref().expect("Levy"), x, out),
            K::VehicleSafety => engineering::vehicle_safety(x, out),
            K::CarSideImpact => engineering::car_side_impact(x, out),
            K::MomfBraninCurrin => multi_fidelity::branin_currin(x, out),
            K::MomfPark => multi_fidelity::park(x, out),
        }
    }

    /// The constraint slacks at one point (no bounds check).
    fn slack_at(&self, _x: &[f64], _out: &mut [f64]) {
        unreachable!("{} has no constraints", self.name())
    }

    /// Noiseless, un-negated objectives (row-major `n × M`) at the row-major
    /// points `x` with `ncols` columns (BoTorch's `evaluate_true`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns and
    /// [`BotorchError::InvalidArgument`] for points outside the bounds.
    pub fn evaluate_true_rows(&self, x: &[f64], ncols: usize) -> Result<Vec<f64>> {
        let n = self.validate(x, ncols)?;
        Ok(self.map_points(x, n, self.num_objectives, |xi, o| self.objectives_at(xi, o)))
    }

    /// Objectives with observation noise (if `noise` and a `noise_std` is
    /// set, drawn from `rng`) and negation (BoTorch's `forward(X, noise)`),
    /// row-major `n × M`.
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_true_rows`].
    pub fn forward_rows_with_rng<R: Rng + ?Sized>(
        &self,
        x: &[f64],
        ncols: usize,
        noise: bool,
        rng: &mut R,
    ) -> Result<Vec<f64>> {
        let mut f = self.evaluate_true_rows(x, ncols)?;
        if let (true, Some(s)) = (noise, &self.noise_std) {
            add_noise(&mut f, &s.expand(self.num_objectives), rng);
        }
        if self.negate {
            f.iter_mut().for_each(|v| *v = -*v);
        }
        Ok(f)
    }

    /// [`MultiObjectiveTestProblem::forward_rows_with_rng`] with the problem's
    /// own (seeded) random number generator.
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_true_rows`].
    pub fn forward_rows(&self, x: &[f64], ncols: usize, noise: bool) -> Result<Vec<f64>> {
        let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
        self.forward_rows_with_rng(x, ncols, noise, &mut *rng)
    }

    fn require_constraints(&self) -> Result<usize> {
        match self.num_constraints() {
            0 => Err(BotorchError::Unsupported(format!(
                "{} has no constraints.",
                self.name()
            ))),
            n_c => Ok(n_c),
        }
    }

    /// Noiseless constraint slacks (row-major `n × n_c`; feasible iff
    /// non-negative) at the row-major points `x` (BoTorch's
    /// `evaluate_slack_true`).
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for unconstrained problems, otherwise see
    /// [`MultiObjectiveTestProblem::evaluate_true_rows`].
    pub fn evaluate_slack_true_rows(&self, x: &[f64], ncols: usize) -> Result<Vec<f64>> {
        let n_c = self.require_constraints()?;
        let n = self.validate(x, ncols)?;
        Ok(self.map_points(x, n, n_c, |xi, o| self.slack_at(xi, o)))
    }

    /// Constraint slacks with observation noise (if `noise` and a
    /// `constraint_noise_std` is set, drawn from `rng`; BoTorch's
    /// `evaluate_slack(X, noise)`), row-major `n × n_c`. Slacks are never
    /// negated.
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_slack_true_rows`].
    pub fn evaluate_slack_rows_with_rng<R: Rng + ?Sized>(
        &self,
        x: &[f64],
        ncols: usize,
        noise: bool,
        rng: &mut R,
    ) -> Result<Vec<f64>> {
        let mut c = self.evaluate_slack_true_rows(x, ncols)?;
        if let (true, Some(s)) = (noise, &self.constraint_noise_std) {
            add_noise(&mut c, &s.expand(self.num_constraints()), rng);
        }
        Ok(c)
    }

    /// [`MultiObjectiveTestProblem::evaluate_slack_rows_with_rng`] with the
    /// problem's own random number generator.
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_slack_true_rows`].
    pub fn evaluate_slack_rows(&self, x: &[f64], ncols: usize, noise: bool) -> Result<Vec<f64>> {
        let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
        self.evaluate_slack_rows_with_rng(x, ncols, noise, &mut *rng)
    }

    /// Whether all (possibly noisy) constraint slacks of each point are
    /// non-negative (BoTorch's `is_feasible(X, noise)`).
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_slack_true_rows`].
    pub fn is_feasible_rows(&self, x: &[f64], ncols: usize, noise: bool) -> Result<Vec<bool>> {
        let c = self.evaluate_slack_rows(x, ncols, noise)?;
        Ok(c.chunks(self.num_constraints())
            .map(|row| row.iter().all(|&v| v >= 0.0))
            .collect())
    }

    /// Noiseless, un-negated objectives at the rows of `x` (`n × d`), as an
    /// `n × M` matrix (BoTorch's `evaluate_true`).
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_true_rows`].
    pub fn evaluate_true(&self, x: &Matrix) -> Result<Matrix> {
        let f = self.evaluate_true_rows(&row_major(x), x.ncols())?;
        Ok(Matrix::from_row_slice(x.nrows(), self.num_objectives, &f))
    }

    /// Objectives with observation noise and negation (BoTorch's
    /// `forward(X, noise)`), using the problem's own random number generator.
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_true_rows`].
    pub fn forward(&self, x: &Matrix, noise: bool) -> Result<Matrix> {
        let f = self.forward_rows(&row_major(x), x.ncols(), noise)?;
        Ok(Matrix::from_row_slice(x.nrows(), self.num_objectives, &f))
    }

    /// [`MultiObjectiveTestProblem::forward`] with an external random number
    /// generator.
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_true_rows`].
    pub fn forward_with_rng<R: Rng + ?Sized>(
        &self,
        x: &Matrix,
        noise: bool,
        rng: &mut R,
    ) -> Result<Matrix> {
        let f = self.forward_rows_with_rng(&row_major(x), x.ncols(), noise, rng)?;
        Ok(Matrix::from_row_slice(x.nrows(), self.num_objectives, &f))
    }

    /// Noiseless constraint slacks at the rows of `x`, as an `n × n_c` matrix
    /// (BoTorch's `evaluate_slack_true`).
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_slack_true_rows`].
    pub fn evaluate_slack_true(&self, x: &Matrix) -> Result<Matrix> {
        let c = self.evaluate_slack_true_rows(&row_major(x), x.ncols())?;
        Ok(Matrix::from_row_slice(
            x.nrows(),
            self.num_constraints(),
            &c,
        ))
    }

    /// Constraint slacks with observation noise (BoTorch's
    /// `evaluate_slack(X, noise)`), using the problem's own random number
    /// generator.
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_slack_true_rows`].
    pub fn evaluate_slack(&self, x: &Matrix, noise: bool) -> Result<Matrix> {
        let c = self.evaluate_slack_rows(&row_major(x), x.ncols(), noise)?;
        Ok(Matrix::from_row_slice(
            x.nrows(),
            self.num_constraints(),
            &c,
        ))
    }

    /// Whether the rows of `x` are feasible (BoTorch's `is_feasible(X, noise)`).
    ///
    /// # Errors
    /// See [`MultiObjectiveTestProblem::evaluate_slack_true_rows`].
    pub fn is_feasible(&self, x: &Matrix, noise: bool) -> Result<Vec<bool>> {
        self.is_feasible_rows(&row_major(x), x.ncols(), noise)
    }
}

/// The entries of `m` in row-major order.
fn row_major(m: &Matrix) -> Vec<f64> {
    m.transpose().as_slice().to_vec()
}

/// Adds `std[j] * z` (standard normal `z`) to the entries of the row-major
/// `values` with `std.len()` columns.
fn add_noise<R: Rng + ?Sized>(values: &mut [f64], std: &[f64], rng: &mut R) {
    for row in values.chunks_mut(std.len()) {
        for (v, s) in row.iter_mut().zip(std) {
            let z: f64 = StandardNormal.sample(rng);
            *v += s * z;
        }
    }
}

/// BoTorch's `_bounds` (one `(lower, upper)` pair per input).
fn default_bounds(kind: MultiObjectiveKind, dim: usize) -> Vec<(f64, f64)> {
    use MultiObjectiveKind as K;
    match kind {
        K::BraninCurrin | K::MomfBraninCurrin | K::MomfPark => vec![(0.0, 1.0); dim],
        K::Dtlz1 | K::Dtlz2 | K::Dtlz3 | K::Dtlz4 | K::Dtlz5 | K::Dtlz7 => vec![(0.0, 1.0); dim],
        K::Zdt1 | K::Zdt2 | K::Zdt3 | K::Gmm => vec![(0.0, 1.0); dim],
        K::ToyRobust => vec![(0.0, 0.7)],
        K::Penicillin => vec![
            (60.0, 120.0),
            (0.05, 18.0),
            (293.0, 303.0),
            (0.05, 18.0),
            (0.01, 0.5),
            (500.0, 700.0),
            (5.0, 6.5),
        ],
        K::VehicleSafety => vec![(1.0, 3.0); 5],
        K::CarSideImpact => vec![
            (0.5, 1.5),
            (0.45, 1.35),
            (0.5, 1.5),
            (0.5, 1.5),
            (0.875, 2.625),
            (0.4, 1.2),
            (0.4, 1.2),
        ],
        K::Dh1 | K::Dh2 | K::Dh3 | K::Dh4 => {
            let x1_lower = match kind {
                K::Dh1 | K::Dh2 => -1.0,
                K::Dh3 => 0.0,
                _ => -0.15,
            };
            let mut b = vec![(0.0, 1.0), (x1_lower, 1.0)];
            b.resize(dim, (-1.0, 1.0));
            b
        }
    }
}

/// BoTorch's `_ref_point` (of the un-negated problem).
fn raw_ref_point(kind: MultiObjectiveKind, m: usize) -> Vec<f64> {
    use MultiObjectiveKind as K;
    match kind {
        K::BraninCurrin => vec![18.0, 6.0],
        K::Dh1 | K::Dh2 | K::Dh3 | K::Dh4 => vec![1.1, 1.1],
        K::Dtlz1 => vec![400.0; m],
        K::Dtlz2 | K::Dtlz4 => vec![1.1; m],
        K::Dtlz3 => vec![10000.0; m],
        K::Dtlz5 => vec![10.0; m],
        K::Dtlz7 => vec![15.0; m],
        K::Zdt1 | K::Zdt2 | K::Zdt3 => vec![11.0, 11.0],
        K::Gmm => gmm::REF_POINT[..m].to_vec(),
        K::ToyRobust => vec![-6.1397, -8.1942],
        K::Penicillin => vec![25.935, 57.612, 935.5],
        K::VehicleSafety => vec![1864.72022, 11.81993945, 0.2903999384],
        K::CarSideImpact => vec![45.4872, 4.5114, 13.3394, 10.3942],
        K::MomfBraninCurrin | K::MomfPark => vec![0.0, 0.0],
    }
}

/// The maximum hypervolume (BoTorch's `_max_hv`), with the corrections of
/// [`MultiObjectiveKind::deviation`].
fn max_hv(kind: MultiObjectiveKind, dim: usize, m: usize) -> Option<f64> {
    use MultiObjectiveKind as K;
    // DH: the area of the box spanned by the reference point minus the area
    // under the Pareto front.
    let dh_box = 1.1 * 1.1;
    match kind {
        K::BraninCurrin => Some(59.4066),
        K::Dh1 | K::Dh2 => Some(dh_box - 2.0 / 3.0),
        K::Dh3 => Some(dh_box - 0.328449169794718),
        // 0.228445... is the area under the part of the front with g = 0 and
        // -0.001572... the change per unit of max(g) = 50 (d - 2) where h < 0.
        K::Dh4 => Some(dh_box - (0.228445034673 - 0.001572780262 * 50.0 * (dim - 2) as f64)),
        // The box spanned by the reference point minus the region below the
        // Pareto front (a simplex for DTLZ1, a ball's orthant for DTLZ2-4).
        K::Dtlz1 => Some(400f64.powf(m as f64) - dtlz::dtlz1_simplex_volume(m)),
        K::Dtlz2 | K::Dtlz4 => Some(1.1f64.powf(m as f64) - dtlz::positive_ball_volume(m)),
        K::Dtlz3 => Some(10000f64.powf(m as f64) - dtlz::positive_ball_volume(m)),
        K::Dtlz5 | K::Dtlz7 => None,
        K::Zdt1 => Some(120.0 + 2.0 / 3.0),
        K::Zdt2 => Some(120.0 + 1.0 / 3.0),
        K::Zdt3 => Some(128.778_116_130_690_76),
        K::Gmm | K::ToyRobust => None,
        K::Penicillin => Some(2183455.909507436),
        K::VehicleSafety => Some(246.81607081187002),
        K::CarSideImpact => Some(484.72654347642793),
        K::MomfBraninCurrin => Some(0.5235514158034145),
        K::MomfPark => Some(0.08551927363087991),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_validated_like_botorch() {
        use MultiObjectiveKind as K;
        let err = |k: K, d: Option<i64>, m: Option<i64>| k.resolve_sizes(d, m).unwrap_err();
        assert_eq!(
            err(K::Dh1, Some(1), None),
            BotorchError::InvalidArgument("dim must be >= 2, but got dim=1!".into())
        );
        assert_eq!(
            err(K::Dh3, Some(-2), None),
            BotorchError::InvalidArgument("dim must be >= 3, but got dim=-2!".into())
        );
        assert!(K::Dh1.resolve_sizes(None, None).is_err());
        assert!(K::BraninCurrin.resolve_sizes(Some(3), None).is_err());
        assert!(K::BraninCurrin.resolve_sizes(None, Some(3)).is_err());
        assert_eq!(K::BraninCurrin.resolve_sizes(None, None).unwrap(), (2, 2));
        assert_eq!(K::MomfPark.resolve_sizes(Some(5), Some(2)).unwrap(), (5, 2));
        assert_eq!(
            err(K::Dtlz2, Some(3), Some(3)),
            BotorchError::InvalidArgument("dim must be > num_objectives, but got 3 and 3.".into())
        );
        assert_eq!(K::Dtlz2.resolve_sizes(Some(4), None).unwrap(), (4, 2));
        assert_eq!(
            err(K::Zdt1, Some(4), Some(3)),
            BotorchError::Unsupported("ZDT1 currently only supports 2 objectives.".into())
        );
        assert_eq!(
            err(K::Zdt3, Some(1), None),
            BotorchError::InvalidArgument("dim must be >= num_objectives, but got 1 and 2".into())
        );
        assert_eq!(
            err(K::Gmm, None, Some(5)),
            BotorchError::Unsupported("GMM only currently supports 2 to 4 objectives.".into())
        );
        assert_eq!(K::Gmm.resolve_sizes(None, Some(4)).unwrap(), (2, 4));
        assert_eq!(K::CarSideImpact.resolve_sizes(None, None).unwrap(), (7, 4));
        for kind in MultiObjectiveKind::ALL {
            assert_eq!(MultiObjectiveKind::from_name(kind.name()), Some(kind));
        }
    }

    #[test]
    fn noise_negation_and_validation() {
        let p = MultiObjectiveTestProblem::new(MultiObjectiveKind::Dh1, Some(3), None)
            .unwrap()
            .with_negate(true)
            .with_noise_std(NoiseStd::PerOutput(vec![0.0, 0.5]))
            .unwrap()
            .with_seed(7);
        let x = Matrix::from_row_slice(2, 3, &[0.5, 0.0, 0.0, 0.25, 0.5, -0.5]);
        let truth = p.evaluate_true(&x).unwrap();
        assert_eq!(p.forward(&x, false).unwrap(), -truth.clone());
        let noisy = p.forward(&x, true).unwrap();
        // No noise on the first objective, noise on the second.
        assert_eq!(noisy.column(0), -truth.column(0));
        assert!(noisy[(0, 1)] != -truth[(0, 1)]);
        let a = p.clone().with_seed(3).forward(&x, true).unwrap();
        let b = p.clone().with_seed(3).forward(&x, true).unwrap();
        assert_eq!(a, b);
        assert_eq!(p.ref_point(), vec![-1.1, -1.1]);
        assert_eq!(p.raw_ref_point(), vec![1.1, 1.1]);
        // Outside the bounds, wrong dimension, invalid noise, no constraints.
        assert!(p
            .evaluate_true(&Matrix::from_row_slice(1, 3, &[1.5, 0.0, 0.0]))
            .is_err());
        assert!(p
            .evaluate_true(&Matrix::from_row_slice(1, 3, &[f64::NAN, 0.0, 0.0]))
            .is_err());
        assert!(matches!(
            p.evaluate_true(&Matrix::zeros(1, 2)),
            Err(BotorchError::DimensionMismatch(_))
        ));
        assert!(matches!(
            p.clone().with_noise_std(NoiseStd::PerOutput(vec![0.1])),
            Err(BotorchError::InputData(_))
        ));
        assert!(p.evaluate_slack_true(&x).is_err());
        assert!(p
            .clone()
            .with_constraint_noise_std(NoiseStd::Scalar(0.1))
            .is_err());
        // Empty inputs.
        assert_eq!(
            p.evaluate_true(&Matrix::zeros(0, 3)).unwrap().shape(),
            (0, 2)
        );
    }

    #[test]
    fn parallel_and_serial_evaluations_agree() {
        let p = MultiObjectiveTestProblem::new(MultiObjectiveKind::Dh3, Some(4), None).unwrap();
        let n = PARALLEL_MIN_POINTS + 3;
        let x: Vec<f64> = (0..n * 4)
            .map(|i| ((i * 7919) % 1000) as f64 / 1000.0)
            .collect();
        let all = p.evaluate_true_rows(&x, 4).unwrap();
        let some = p.evaluate_true_rows(&x[..40], 4).unwrap();
        assert_eq!(&all[..20], &some[..]);
    }
}
