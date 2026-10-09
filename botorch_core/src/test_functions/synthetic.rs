//! Implementation of the synthetic test functions.

use std::f64::consts::{E, PI};
use std::sync::Mutex;

use rand::rngs::Xoshiro256PlusPlus;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::error::{BotorchError, Result};
use crate::utils::sampling::random_seed;
use crate::{Matrix, Vector};

/// The synthetic test functions of `botorch.test_functions.synthetic` that
/// are available (the unconstrained, continuous ones).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TestFunctionKind {
    /// Ackley, `d`-dimensional on `[-32.768, 32.768]^d`.
    Ackley,
    /// Beale, 2-dimensional on `[-4.5, 4.5]^2`.
    Beale,
    /// Branin, 2-dimensional on `[-5, 10] × [0, 15]`.
    Branin,
    /// Bukin N.6, 2-dimensional on `[-15, -5] × [-3, 3]`.
    Bukin,
    /// Cosine mixture, 8-dimensional on `[-1, 1]^8` (a maximization problem).
    Cosine8,
    /// Dixon–Price, `d`-dimensional on `[-10, 10]^d`.
    DixonPrice,
    /// Drop-wave, 2-dimensional on `[-5.12, 5.12]^2`.
    DropWave,
    /// Eggholder, 2-dimensional on `[-512, 512]^2`.
    EggHolder,
    /// Griewank, `d`-dimensional on `[-600, 600]^d`.
    Griewank,
    /// Hartmann, `d ∈ {3, 4, 6}` on `[0, 1]^d`.
    Hartmann,
    /// Hölder table, 2-dimensional on `[-10, 10]^2`.
    HolderTable,
    /// Levy, `d`-dimensional on `[-10, 10]^d`.
    Levy,
    /// Michalewicz (`m = 10`), `d`-dimensional on `[0, π]^d`.
    Michalewicz,
    /// Powell, `d`-dimensional on `[-4, 5]^d` (uses the first `4⌊d/4⌋` inputs).
    Powell,
    /// Rastrigin, `d`-dimensional on `[-5.12, 5.12]^d`.
    Rastrigin,
    /// Rosenbrock, `d`-dimensional on `[-5, 10]^d`.
    Rosenbrock,
    /// Shekel with `m ∈ {5, 7, 10}` terms, 4-dimensional on `[0, 10]^4`.
    Shekel,
    /// Six-hump camel, 2-dimensional on `[-3, 3] × [-2, 2]`.
    SixHumpCamel,
    /// Styblinski–Tang, `d`-dimensional on `[-5, 5]^d`.
    StyblinskiTang,
    /// Three-hump camel, 2-dimensional on `[-5, 5]^2`.
    ThreeHumpCamel,
}

impl TestFunctionKind {
    /// All available test functions.
    pub const ALL: [TestFunctionKind; 20] = [
        Self::Ackley,
        Self::Beale,
        Self::Branin,
        Self::Bukin,
        Self::Cosine8,
        Self::DixonPrice,
        Self::DropWave,
        Self::EggHolder,
        Self::Griewank,
        Self::Hartmann,
        Self::HolderTable,
        Self::Levy,
        Self::Michalewicz,
        Self::Powell,
        Self::Rastrigin,
        Self::Rosenbrock,
        Self::Shekel,
        Self::SixHumpCamel,
        Self::StyblinskiTang,
        Self::ThreeHumpCamel,
    ];

    /// The BoTorch class name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ackley => "Ackley",
            Self::Beale => "Beale",
            Self::Branin => "Branin",
            Self::Bukin => "Bukin",
            Self::Cosine8 => "Cosine8",
            Self::DixonPrice => "DixonPrice",
            Self::DropWave => "DropWave",
            Self::EggHolder => "EggHolder",
            Self::Griewank => "Griewank",
            Self::Hartmann => "Hartmann",
            Self::HolderTable => "HolderTable",
            Self::Levy => "Levy",
            Self::Michalewicz => "Michalewicz",
            Self::Powell => "Powell",
            Self::Rastrigin => "Rastrigin",
            Self::Rosenbrock => "Rosenbrock",
            Self::Shekel => "Shekel",
            Self::SixHumpCamel => "SixHumpCamel",
            Self::StyblinskiTang => "StyblinskiTang",
            Self::ThreeHumpCamel => "ThreeHumpCamel",
        }
    }

    /// The dimension of fixed-dimensional functions.
    pub fn fixed_dim(self) -> Option<usize> {
        match self {
            Self::Beale
            | Self::Branin
            | Self::Bukin
            | Self::DropWave
            | Self::EggHolder
            | Self::HolderTable
            | Self::SixHumpCamel
            | Self::ThreeHumpCamel => Some(2),
            Self::Cosine8 => Some(8),
            Self::Shekel => Some(4),
            _ => None,
        }
    }

    /// BoTorch's default dimension.
    pub fn default_dim(self) -> usize {
        match self {
            Self::Hartmann => 6,
            Self::Powell => 4,
            other => other.fixed_dim().unwrap_or(2),
        }
    }

    /// Default bounds of one coordinate (`(lower, upper)` for dimension `k`).
    fn default_bounds(self, k: usize) -> (f64, f64) {
        match self {
            Self::Ackley => (-32.768, 32.768),
            Self::Beale => (-4.5, 4.5),
            Self::Branin => [(-5.0, 10.0), (0.0, 15.0)][k],
            Self::Bukin => [(-15.0, -5.0), (-3.0, 3.0)][k],
            Self::Cosine8 => (-1.0, 1.0),
            Self::DixonPrice | Self::Levy | Self::HolderTable => (-10.0, 10.0),
            Self::DropWave | Self::Rastrigin => (-5.12, 5.12),
            Self::EggHolder => (-512.0, 512.0),
            Self::Griewank => (-600.0, 600.0),
            Self::Hartmann => (0.0, 1.0),
            Self::Michalewicz => (0.0, PI),
            Self::Powell => (-4.0, 5.0),
            Self::Rosenbrock => (-5.0, 10.0),
            Self::Shekel => (0.0, 10.0),
            Self::SixHumpCamel => [(-3.0, 3.0), (-2.0, 2.0)][k],
            Self::StyblinskiTang | Self::ThreeHumpCamel => (-5.0, 5.0),
        }
    }
}

/// `x as f32 as f64`: a constant stored by BoTorch in a float32 buffer.
const fn f32r(x: f32) -> f64 {
    x as f64
}

const HARTMANN_ALPHA: [f64; 4] = [f32r(1.0), f32r(1.2), f32r(3.0), f32r(3.2)];

const HARTMANN3_A: [[f64; 3]; 4] = [
    [f32r(3.0), f32r(10.0), f32r(30.0)],
    [f32r(0.1), f32r(10.0), f32r(35.0)],
    [f32r(3.0), f32r(10.0), f32r(30.0)],
    [f32r(0.1), f32r(10.0), f32r(35.0)],
];
const HARTMANN3_P: [[f64; 3]; 4] = [
    [3689.0, 1170.0, 2673.0],
    [4699.0, 4387.0, 7470.0],
    [1091.0, 8732.0, 5547.0],
    [381.0, 5743.0, 8828.0],
];
const HARTMANN4_A: [[f64; 4]; 4] = [
    [f32r(10.0), f32r(3.0), f32r(17.0), f32r(3.5)],
    [f32r(0.05), f32r(10.0), f32r(17.0), f32r(0.1)],
    [f32r(3.0), f32r(3.5), f32r(1.7), f32r(10.0)],
    [f32r(17.0), f32r(8.0), f32r(0.05), f32r(10.0)],
];
const HARTMANN4_P: [[f64; 4]; 4] = [
    [1312.0, 1696.0, 5569.0, 124.0],
    [2329.0, 4135.0, 8307.0, 3736.0],
    [2348.0, 1451.0, 3522.0, 2883.0],
    [4047.0, 8828.0, 8732.0, 5743.0],
];
const HARTMANN6_A: [[f64; 6]; 4] = [
    [
        f32r(10.0),
        f32r(3.0),
        f32r(17.0),
        f32r(3.5),
        f32r(1.7),
        f32r(8.0),
    ],
    [
        f32r(0.05),
        f32r(10.0),
        f32r(17.0),
        f32r(0.1),
        f32r(8.0),
        f32r(14.0),
    ],
    [
        f32r(3.0),
        f32r(3.5),
        f32r(1.7),
        f32r(10.0),
        f32r(17.0),
        f32r(8.0),
    ],
    [
        f32r(17.0),
        f32r(8.0),
        f32r(0.05),
        f32r(10.0),
        f32r(0.1),
        f32r(14.0),
    ],
];
const HARTMANN6_P: [[f64; 6]; 4] = [
    [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
    [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
    [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
    [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
];

const SHEKEL_BETA: [f64; 10] = [1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 3.0, 7.0, 5.0, 5.0];
/// Shekel's `C` (BoTorch's `C_t.T`; `3.6` is stored in float32).
const SHEKEL_C: [[f64; 4]; 10] = [
    [4.0, 4.0, 4.0, 4.0],
    [1.0, 1.0, 1.0, 1.0],
    [8.0, 8.0, 8.0, 8.0],
    [6.0, 6.0, 6.0, 6.0],
    [3.0, 7.0, 3.0, 7.0],
    [2.0, 9.0, 2.0, 9.0],
    [5.0, 3.0, 5.0, 3.0],
    [8.0, 1.0, 8.0, 1.0],
    [6.0, 2.0, 6.0, 2.0],
    [7.0, f32r(3.6), 7.0, f32r(3.6)],
];

/// A synthetic test problem (BoTorch's `SyntheticTestFunction`) with bounds,
/// optional negation and observation noise.
#[derive(Debug)]
pub struct SyntheticTestFunction {
    kind: TestFunctionKind,
    dim: usize,
    shekel_m: usize,
    bounds: Matrix,
    negate: bool,
    noise_std: Option<f64>,
    rng: Mutex<Xoshiro256PlusPlus>,
}

impl Clone for SyntheticTestFunction {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind,
            dim: self.dim,
            shekel_m: self.shekel_m,
            bounds: self.bounds.clone(),
            negate: self.negate,
            noise_std: self.noise_std,
            rng: Mutex::new(self.rng.lock().unwrap_or_else(|e| e.into_inner()).clone()),
        }
    }
}

impl SyntheticTestFunction {
    /// Creates a test function of the given kind; `dim` must be `None` or
    /// equal to the fixed dimension for fixed-dimensional functions, and
    /// defaults to BoTorch's default otherwise. Shekel uses `m = 10`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for an unsupported dimension.
    pub fn new(kind: TestFunctionKind, dim: Option<usize>) -> Result<Self> {
        let d = match (kind.fixed_dim(), dim) {
            (Some(f), Some(d)) if d != f => {
                return Err(BotorchError::InvalidArgument(format!(
                    "{} is {f}-dimensional, got dim = {d}",
                    kind.name()
                )))
            }
            (Some(f), _) => f,
            (None, Some(d)) => d,
            (None, None) => kind.default_dim(),
        };
        if d == 0 {
            return Err(BotorchError::InvalidArgument(
                "the dimension must be positive".into(),
            ));
        }
        if kind == TestFunctionKind::Hartmann && ![3, 4, 6].contains(&d) {
            return Err(BotorchError::InvalidArgument(format!(
                "Hartmann with dim {d} not defined"
            )));
        }
        let bounds = Matrix::from_fn(2, d, |i, k| {
            let (lo, hi) = kind.default_bounds(k);
            if i == 0 {
                lo
            } else {
                hi
            }
        });
        Ok(Self {
            kind,
            dim: d,
            shekel_m: if kind == TestFunctionKind::Shekel {
                10
            } else {
                0
            },
            bounds,
            negate: false,
            noise_std: None,
            rng: Mutex::new(Xoshiro256PlusPlus::seed_from_u64(random_seed())),
        })
    }

    fn fixed(kind: TestFunctionKind) -> Self {
        Self::new(kind, None).expect("fixed-dimensional test function")
    }

    /// `Ackley(dim)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn ackley(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::Ackley, Some(dim))
    }

    /// `Beale()`.
    pub fn beale() -> Self {
        Self::fixed(TestFunctionKind::Beale)
    }

    /// `Branin()`.
    pub fn branin() -> Self {
        Self::fixed(TestFunctionKind::Branin)
    }

    /// `Bukin()`.
    pub fn bukin() -> Self {
        Self::fixed(TestFunctionKind::Bukin)
    }

    /// `Cosine8()`.
    pub fn cosine8() -> Self {
        Self::fixed(TestFunctionKind::Cosine8)
    }

    /// `DixonPrice(dim)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn dixon_price(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::DixonPrice, Some(dim))
    }

    /// `DropWave()`.
    pub fn drop_wave() -> Self {
        Self::fixed(TestFunctionKind::DropWave)
    }

    /// `EggHolder()`.
    pub fn egg_holder() -> Self {
        Self::fixed(TestFunctionKind::EggHolder)
    }

    /// `Griewank(dim)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn griewank(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::Griewank, Some(dim))
    }

    /// `Hartmann(dim)` for `dim ∈ {3, 4, 6}`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for other dimensions.
    pub fn hartmann(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::Hartmann, Some(dim))
    }

    /// `HolderTable()`.
    pub fn holder_table() -> Self {
        Self::fixed(TestFunctionKind::HolderTable)
    }

    /// `Levy(dim)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn levy(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::Levy, Some(dim))
    }

    /// `Michalewicz(dim)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn michalewicz(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::Michalewicz, Some(dim))
    }

    /// `Powell(dim)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn powell(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::Powell, Some(dim))
    }

    /// `Rastrigin(dim)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn rastrigin(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::Rastrigin, Some(dim))
    }

    /// `Rosenbrock(dim)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn rosenbrock(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::Rosenbrock, Some(dim))
    }

    /// `Shekel(m)` for `m ∈ {5, 7, 10}`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for other `m`.
    pub fn shekel(m: usize) -> Result<Self> {
        if ![5, 7, 10].contains(&m) {
            return Err(BotorchError::InvalidArgument(format!(
                "Shekel is defined for m in {{5, 7, 10}}, got {m}"
            )));
        }
        let mut f = Self::fixed(TestFunctionKind::Shekel);
        f.shekel_m = m;
        Ok(f)
    }

    /// `SixHumpCamel()`.
    pub fn six_hump_camel() -> Self {
        Self::fixed(TestFunctionKind::SixHumpCamel)
    }

    /// `StyblinskiTang(dim)`.
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] if `dim == 0`.
    pub fn styblinski_tang(dim: usize) -> Result<Self> {
        Self::new(TestFunctionKind::StyblinskiTang, Some(dim))
    }

    /// `ThreeHumpCamel()`.
    pub fn three_hump_camel() -> Self {
        Self::fixed(TestFunctionKind::ThreeHumpCamel)
    }

    /// Sets `negate` (BoTorch's `negate=True` turns minimization problems
    /// into maximization problems).
    pub fn with_negate(mut self, negate: bool) -> Self {
        self.negate = negate;
        self
    }

    /// Sets the standard deviation of the observation noise of
    /// [`SyntheticTestFunction::forward`].
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] unless `noise_std` is finite and
    /// non-negative.
    pub fn with_noise_std(mut self, noise_std: f64) -> Result<Self> {
        if !(noise_std >= 0.0 && noise_std.is_finite()) {
            return Err(BotorchError::InvalidArgument(format!(
                "noise_std must be finite and non-negative, got {noise_std}"
            )));
        }
        self.noise_std = Some(noise_std);
        Ok(self)
    }

    /// Seeds the random number generator of the observation noise (by
    /// default it is seeded randomly, like torch's global RNG).
    pub fn with_seed(self, seed: u64) -> Self {
        *self.rng.lock().unwrap_or_else(|e| e.into_inner()) =
            Xoshiro256PlusPlus::seed_from_u64(seed);
        self
    }

    /// Replaces the bounds (BoTorch's `bounds=` argument; `2 × d`). As in
    /// BoTorch, at least one known optimizer must lie within custom bounds.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] / [`BotorchError::InvalidArgument`]
    /// for invalid bounds or bounds that exclude every optimizer.
    pub fn with_bounds(mut self, bounds: Matrix) -> Result<Self> {
        let d = crate::utils::sampling::check_bounds(&bounds)?;
        if d != self.dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected the bounds to match the dimensionality of the domain. Got {} and {d}.",
                self.dim
            )));
        }
        if let Some(opt) = self.optimizers() {
            let inside = (0..opt.nrows()).any(|i| {
                (0..d).all(|k| opt[(i, k)] >= bounds[(0, k)] && opt[(i, k)] <= bounds[(1, k)])
            });
            if !inside {
                return Err(BotorchError::InvalidArgument(format!(
                    "No global optimum found within custom bounds. Please specify bounds which \
                     include at least one point in `{}._optimizers`.",
                    self.kind.name()
                )));
            }
        }
        self.bounds = bounds;
        Ok(self)
    }

    /// The kind of function.
    pub fn kind(&self) -> TestFunctionKind {
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

    /// Shekel's number of terms `m` (`None` for other functions).
    pub fn shekel_m(&self) -> Option<usize> {
        (self.kind == TestFunctionKind::Shekel).then_some(self.shekel_m)
    }

    /// The `2 × d` bounds (lower bounds in row 0).
    pub fn bounds(&self) -> &Matrix {
        &self.bounds
    }

    /// Whether [`SyntheticTestFunction::forward`] negates the function.
    pub fn negate(&self) -> bool {
        self.negate
    }

    /// The observation noise standard deviation, if any.
    pub fn noise_std(&self) -> Option<f64> {
        self.noise_std
    }

    /// Whether the problem is a minimization problem after accounting for
    /// `negate` (all functions are minimization problems by default except
    /// Cosine8).
    pub fn is_minimization_problem(&self) -> bool {
        let default_min = self.kind != TestFunctionKind::Cosine8;
        default_min != self.negate
    }

    /// The global optimum (negated if `negate`), if known.
    pub fn optimal_value(&self) -> Option<f64> {
        use TestFunctionKind as K;
        let v = match self.kind {
            K::Ackley
            | K::Beale
            | K::Bukin
            | K::DixonPrice
            | K::Griewank
            | K::Levy
            | K::Powell
            | K::Rastrigin
            | K::Rosenbrock
            | K::ThreeHumpCamel => Some(0.0),
            K::Branin => Some(0.397887),
            K::Cosine8 => Some(0.8),
            K::DropWave => Some(-1.0),
            K::EggHolder => Some(-959.6407),
            K::Hartmann => match self.dim {
                3 => Some(-3.86278),
                6 => Some(-3.32237),
                _ => None,
            },
            K::HolderTable => Some(-19.2085),
            K::Michalewicz => match self.dim {
                2 => Some(-1.80130341),
                5 => Some(-4.687658),
                10 => Some(-9.66015),
                _ => None,
            },
            K::Shekel => match self.shekel_m {
                5 => Some(-10.1532),
                7 => Some(-10.4029),
                _ => Some(-10.536443),
            },
            K::SixHumpCamel => Some(-1.0316),
            K::StyblinskiTang => Some(-39.166166 * self.dim as f64),
        };
        v.map(|v| if self.negate { -v } else { v })
    }

    /// The known global optimizers (one per row), if any.
    // BoTorch's literal optimizer values (e.g. Michalewicz's 1.57079633), not constants.
    #[allow(clippy::approx_constant)]
    pub fn optimizers(&self) -> Option<Matrix> {
        use TestFunctionKind as K;
        let d = self.dim;
        let rows: Vec<Vec<f64>> = match self.kind {
            K::Ackley | K::Cosine8 | K::Griewank | K::Powell | K::Rastrigin => vec![vec![0.0; d]],
            K::Levy | K::Rosenbrock => vec![vec![1.0; d]],
            K::Beale => vec![vec![3.0, 0.5]],
            K::Branin => vec![vec![-PI, 12.275], vec![PI, 2.275], vec![9.42478, 2.475]],
            K::Bukin => vec![vec![-10.0, 1.0]],
            K::DixonPrice => vec![(1..=d)
                .map(|i| 2f64.powf(-(1.0 - 2f64.powf(-((i as f64) - 1.0)))))
                .collect()],
            K::DropWave | K::ThreeHumpCamel => vec![vec![0.0, 0.0]],
            K::EggHolder => vec![vec![512.0, 404.2319]],
            K::Hartmann => match d {
                3 => vec![vec![0.114614, 0.555649, 0.852547]],
                6 => vec![vec![
                    0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573,
                ]],
                _ => return None,
            },
            K::HolderTable => vec![
                vec![8.05502, 9.66459],
                vec![-8.05502, -9.66459],
                vec![-8.05502, 9.66459],
                vec![8.05502, -9.66459],
            ],
            K::Michalewicz => match d {
                2 => vec![vec![2.20290552, 1.57079633]],
                _ => return None,
            },
            K::Shekel => vec![vec![4.000747, 3.99951, 4.00075, 3.99951]],
            K::SixHumpCamel => vec![vec![0.0898, -0.7126], vec![-0.0898, 0.7126]],
            K::StyblinskiTang => vec![vec![-2.903534; d]],
        };
        Some(Matrix::from_fn(rows.len(), d, |i, k| rows[i][k]))
    }

    /// Noiseless, un-negated function values at the rows of `x` (BoTorch's
    /// `evaluate_true`).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns and
    /// [`BotorchError::InvalidArgument`] for points outside the bounds.
    pub fn evaluate_true(&self, x: &Matrix) -> Result<Vector> {
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
        }
        Ok(Vector::from_iterator(
            x.nrows(),
            (0..x.nrows()).map(|i| {
                let p: Vec<f64> = x.row(i).iter().copied().collect();
                self.value(&p)
            }),
        ))
    }

    /// Function values with observation noise (if `noise` and a `noise_std`
    /// is set) and negation, using the function's own seeded random number
    /// generator (BoTorch's `forward(X, noise)`).
    ///
    /// # Errors
    /// See [`SyntheticTestFunction::evaluate_true`].
    pub fn forward(&self, x: &Matrix, noise: bool) -> Result<Vector> {
        let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
        self.forward_with_rng(x, noise, &mut *rng)
    }

    /// [`SyntheticTestFunction::forward`] with an external random number
    /// generator.
    ///
    /// # Errors
    /// See [`SyntheticTestFunction::evaluate_true`].
    pub fn forward_with_rng<R: Rng + ?Sized>(
        &self,
        x: &Matrix,
        noise: bool,
        rng: &mut R,
    ) -> Result<Vector> {
        let mut f = self.evaluate_true(x)?;
        if let (true, Some(s)) = (noise, self.noise_std) {
            for v in f.iter_mut() {
                let z: f64 = StandardNormal.sample(rng);
                *v += s * z;
            }
        }
        if self.negate {
            f.neg_mut();
        }
        Ok(f)
    }

    /// The function value at a single point (no bounds check).
    fn value(&self, x: &[f64]) -> f64 {
        use TestFunctionKind as K;
        let d = self.dim as f64;
        match self.kind {
            K::Ackley => {
                let (a, b, c) = (20.0, 0.2, 2.0 * PI);
                let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                let part1 = -a * (-norm * b / d.sqrt()).exp();
                let part2 = -(x.iter().map(|v| (c * v).cos()).sum::<f64>() / d).exp();
                part1 + part2 + a + E
            }
            K::Beale => {
                let (x1, x2) = (x[0], x[1]);
                let p1 = (1.5 - x1 + x1 * x2).powi(2);
                let p2 = (2.25 - x1 + x1 * x2.powi(2)).powi(2);
                let p3 = (2.625 - x1 + x1 * x2.powi(3)).powi(2);
                p1 + p2 + p3
            }
            K::Branin => {
                let t1 = x[1] - 5.1 / (4.0 * PI.powi(2)) * x[0].powi(2) + 5.0 / PI * x[0] - 6.0;
                let t2 = 10.0 * (1.0 - 1.0 / (8.0 * PI)) * x[0].cos();
                t1.powi(2) + t2 + 10.0
            }
            K::Bukin => {
                let p1 = 100.0 * (x[1] - 0.01 * x[0].powi(2)).abs().sqrt();
                let p2 = 0.01 * (x[0] + 10.0).abs();
                p1 + p2
            }
            K::Cosine8 => x
                .iter()
                .map(|v| 0.1 * (5.0 * PI * v).cos() - v.powi(2))
                .sum(),
            K::DixonPrice => {
                let p1 = (x[0] - 1.0).powi(2);
                let p2: f64 = (1..x.len())
                    .map(|j| ((j + 1) as f64) * (2.0 * x[j].powi(2) - x[j - 1]).powi(2))
                    .sum();
                p1 + p2
            }
            K::DropWave => {
                let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                let p1 = 1.0 + (12.0 * norm).cos();
                let p2 = 0.5 * norm.powi(2) + 2.0;
                -p1 / p2
            }
            K::EggHolder => {
                let (x1, x2) = (x[0], x[1]);
                let p1 = -(x2 + 47.0) * (x2 + x1 / 2.0 + 47.0).abs().sqrt().sin();
                let p2 = -x1 * (x1 - (x2 + 47.0)).abs().sqrt().sin();
                p1 + p2
            }
            K::Griewank => {
                let p1: f64 = x.iter().map(|v| v.powi(2) / 4000.0).sum();
                let p2 = -x
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v / ((i + 1) as f64).sqrt()).cos())
                    .product::<f64>();
                p1 + p2 + 1.0
            }
            K::Hartmann => {
                let h = match self.dim {
                    3 => hartmann(x, &HARTMANN3_A, &HARTMANN3_P),
                    4 => hartmann(x, &HARTMANN4_A, &HARTMANN4_P),
                    _ => hartmann(x, &HARTMANN6_A, &HARTMANN6_P),
                };
                if self.dim == 4 {
                    (1.1 + h) / 0.839
                } else {
                    h
                }
            }
            K::HolderTable => {
                let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                let term = (1.0 - norm / PI).abs();
                -(x[0].sin() * x[1].cos() * term.exp()).abs()
            }
            K::Levy => {
                let w: Vec<f64> = x.iter().map(|v| 1.0 + (v - 1.0) / 4.0).collect();
                let n = w.len();
                let p1 = (PI * w[0]).sin().powi(2);
                let p2: f64 = w[..n - 1]
                    .iter()
                    .map(|wi| (wi - 1.0).powi(2) * (1.0 + 10.0 * (PI * wi + 1.0).sin().powi(2)))
                    .sum();
                let wd = w[n - 1];
                let p3 = (wd - 1.0).powi(2) * (1.0 + (2.0 * PI * wd).sin().powi(2));
                p1 + p2 + p3
            }
            K::Michalewicz => -x
                .iter()
                .enumerate()
                .map(|(i, v)| v.sin() * (((i + 1) as f64) * v.powi(2) / PI).sin().powi(20))
                .sum::<f64>(),
            K::Powell => {
                let mut result = 0.0;
                for i in 0..self.dim / 4 {
                    let b = 4 * i;
                    let p1 = (x[b] + 10.0 * x[b + 1]).powi(2);
                    let p2 = 5.0 * (x[b + 2] - x[b + 3]).powi(2);
                    let p3 = (x[b + 1] - 2.0 * x[b + 2]).powi(4);
                    let p4 = 10.0 * (x[b] - x[b + 3]).powi(4);
                    result += p1 + p2 + p3 + p4;
                }
                result
            }
            K::Rastrigin => {
                10.0 * d
                    + x.iter()
                        .map(|v| v.powi(2) - 10.0 * (2.0 * PI * v).cos())
                        .sum::<f64>()
            }
            K::Rosenbrock => (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (x[i] - 1.0).powi(2))
                .sum(),
            K::Shekel => {
                let mut s = 0.0;
                for i in 0..self.shekel_m {
                    let beta = SHEKEL_BETA[i] / 10.0;
                    let dist: f64 = (0..4).map(|k| (x[k] - SHEKEL_C[i][k]).powi(2)).sum();
                    s += 1.0 / (dist + beta);
                }
                -s
            }
            K::SixHumpCamel => {
                let (x1, x2) = (x[0], x[1]);
                (4.0 - 2.1 * x1.powi(2) + x1.powi(4) / 3.0) * x1.powi(2)
                    + x1 * x2
                    + (4.0 * x2.powi(2) - 4.0) * x2.powi(2)
            }
            K::StyblinskiTang => {
                0.5 * x
                    .iter()
                    .map(|v| v.powi(4) - 16.0 * v.powi(2) + 5.0 * v)
                    .sum::<f64>()
            }
            K::ThreeHumpCamel => {
                let (x1, x2) = (x[0], x[1]);
                2.0 * x1.powi(2) - 1.05 * x1.powi(4) + x1.powi(6) / 6.0 + x1 * x2 + x2.powi(2)
            }
        }
    }
}

/// `-Σ_i α_i exp(-Σ_j A_ij (x_j - 1e-4 P_ij)²)`.
fn hartmann<const D: usize>(x: &[f64], a: &[[f64; D]; 4], p: &[[f64; D]; 4]) -> f64 {
    let mut h = 0.0;
    for i in 0..4 {
        let inner: f64 = (0..D)
            .map(|j| a[i][j] * (x[j] - 0.0001 * p[i][j]).powi(2))
            .sum();
        h += HARTMANN_ALPHA[i] * (-inner).exp();
    }
    -h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optima_are_attained() {
        for kind in TestFunctionKind::ALL {
            let dims: Vec<Option<usize>> = match kind {
                TestFunctionKind::Hartmann => vec![Some(3), Some(6)],
                k if k.fixed_dim().is_some() => vec![None],
                TestFunctionKind::Michalewicz => vec![Some(2)],
                _ => vec![Some(2), Some(4)],
            };
            for dim in dims {
                let f = SyntheticTestFunction::new(kind, dim).unwrap();
                let (Some(opt), Some(x)) = (f.optimal_value(), f.optimizers()) else {
                    continue;
                };
                let v = f.evaluate_true(&x).unwrap();
                for &vi in v.iter() {
                    assert!(
                        (vi - opt).abs() <= 1e-3 * opt.abs().max(1.0),
                        "{} (dim {}): f(x*) = {vi}, optimal value {opt}",
                        f.name(),
                        f.dim()
                    );
                }
            }
        }
    }

    #[test]
    fn negate_noise_and_bounds() {
        let f = SyntheticTestFunction::branin()
            .with_negate(true)
            .with_noise_std(0.1)
            .unwrap()
            .with_seed(3);
        assert!(!f.is_minimization_problem());
        assert_eq!(f.optimal_value(), Some(-0.397887));
        let x = f.optimizers().unwrap();
        let clean = f.forward(&x, false).unwrap();
        let truth = f.evaluate_true(&x).unwrap();
        assert_eq!(clean, -truth.clone());
        let noisy = f.forward(&x, true).unwrap();
        assert!(noisy
            .iter()
            .zip(clean.iter())
            .all(|(a, b)| a != b && (a - b).abs() < 1.0));
        let g = f.clone().with_seed(3);
        let h = f.clone().with_seed(3);
        assert_eq!(g.forward(&x, true).unwrap(), h.forward(&x, true).unwrap());
        assert!(f
            .evaluate_true(&Matrix::from_row_slice(1, 2, &[11.0, 0.0]))
            .is_err());
        assert!(f.evaluate_true(&Matrix::zeros(1, 3)).is_err());
        assert!(!SyntheticTestFunction::cosine8().is_minimization_problem());
        // Custom bounds must contain an optimizer.
        let b = Matrix::from_row_slice(2, 2, &[0.0, 0.0, 4.0, 4.0]);
        assert!(SyntheticTestFunction::branin().with_bounds(b).is_ok());
        let b = Matrix::from_row_slice(2, 2, &[5.0, 5.0, 6.0, 6.0]);
        assert!(SyntheticTestFunction::branin().with_bounds(b).is_err());
        assert!(SyntheticTestFunction::hartmann(5).is_err());
        assert!(SyntheticTestFunction::shekel(6).is_err());
        assert!(SyntheticTestFunction::new(TestFunctionKind::Beale, Some(3)).is_err());
        assert_eq!(
            SyntheticTestFunction::hartmann(4).unwrap().optimal_value(),
            None
        );
        assert!(SyntheticTestFunction::hartmann(4)
            .unwrap()
            .optimizers()
            .is_none());
    }
}
