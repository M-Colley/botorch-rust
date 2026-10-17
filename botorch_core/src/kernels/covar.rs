//! The covariance module of a [`crate::SingleTaskGP`]: either a
//! [`KernelSpec`] (an RBF or Matérn kernel with ARD lengthscales and an
//! optional `ScaleKernel`, BoTorch's default modules, evaluated by the
//! specialized [`Kernel`]) or a general kernel tree ([`KernelNode`],
//! evaluated by [`TreeKernel`]), with one evaluation interface.

use crate::error::Result;
use crate::kernels::eval::{TreeKernel, TreePrep};
use crate::kernels::nonfinite::TrainingNan;
use crate::kernels::tree::KernelNode;
use crate::kernels::{Kernel, KernelSpec, ScaledPoints};
use crate::{Matrix, Vector};

/// A covariance-module configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum CovarModule {
    /// An RBF or Matérn kernel with ARD lengthscales, optionally in a
    /// `ScaleKernel` (raw parameters `raw_outputscale`, `raw_lengthscale`).
    Standard(KernelSpec),
    /// A general kernel tree (raw parameters in `named_parameters()` order).
    Tree(KernelNode),
}

impl From<KernelSpec> for CovarModule {
    fn from(spec: KernelSpec) -> Self {
        CovarModule::Standard(spec)
    }
}

impl From<KernelNode> for CovarModule {
    fn from(node: KernelNode) -> Self {
        CovarModule::Tree(node)
    }
}

impl CovarModule {
    /// The standard configuration, if this is one.
    pub fn as_standard(&self) -> Option<&KernelSpec> {
        match self {
            CovarModule::Standard(s) => Some(s),
            CovarModule::Tree(_) => None,
        }
    }

    /// The kernel tree, if this is one.
    pub fn as_tree(&self) -> Option<&KernelNode> {
        match self {
            CovarModule::Standard(_) => None,
            CovarModule::Tree(t) => Some(t),
        }
    }

    /// Whether the module has hyperparameter priors.
    pub fn has_priors(&self) -> bool {
        match self {
            CovarModule::Standard(s) => {
                s.lengthscale_prior.is_some()
                    || s.scale.and_then(|sc| sc.outputscale_prior).is_some()
            }
            CovarModule::Tree(t) => t.has_priors(),
        }
    }
}

/// A covariance module with concrete hyperparameter values.
#[derive(Clone, Debug)]
pub enum BuiltCovar {
    /// See [`CovarModule::Standard`].
    Standard(Kernel),
    /// See [`CovarModule::Tree`].
    Tree(TreeKernel),
}

/// Points prepared for a [`BuiltCovar`].
#[derive(Clone, Debug)]
pub enum Prepared {
    /// Points scaled by the lengthscales of a standard kernel.
    Standard(ScaledPoints),
    /// Points prepared for a kernel tree.
    Tree(TreePrep),
}

impl Prepared {
    /// The number of points.
    pub fn len(&self) -> usize {
        match self {
            Prepared::Standard(p) => p.len(),
            Prepared::Tree(p) => p.len(),
        }
    }

    /// Whether there are no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

macro_rules! pair {
    ($a:expr, $b:expr) => {
        match ($a, $b) {
            (Prepared::Standard(a), Prepared::Standard(b)) => (Some((a, b)), None),
            (Prepared::Tree(a), Prepared::Tree(b)) => (None, Some((a, b))),
            _ => unreachable!("points prepared for another kernel"),
        }
    };
}

impl BuiltCovar {
    /// Prepares the rows of `x` for evaluation.
    ///
    /// # Errors
    /// See [`TreeKernel::prepare`] (a standard kernel never fails).
    pub fn prepare(&self, x: &Matrix) -> Result<Prepared> {
        Ok(match self {
            BuiltCovar::Standard(k) => Prepared::Standard(k.scale_points(x)),
            BuiltCovar::Tree(t) => Prepared::Tree(t.prepare(x)?),
        })
    }

    /// GPyTorch's NaN entries of the training covariance `K(X, X)` of the
    /// raw training inputs `x` and the hyperparameters whose gradient is NaN
    /// (lengthscale indices of a standard kernel, flat offsets in a tree);
    /// see [`crate::kernels::nonfinite`]. `inputs_require_grad`: the inputs
    /// depend on learned parameters (an input transform fitted with the
    /// GP), which selects GPyTorch's autograd path for every kernel.
    pub fn training_nan(&self, x: &Matrix, inputs_require_grad: bool) -> TrainingNan {
        match self {
            BuiltCovar::Standard(k) => k.training_nan(x, inputs_require_grad),
            BuiltCovar::Tree(t) => t.training_nan(x, inputs_require_grad),
        }
    }

    /// `K(A, B)`.
    pub fn cross(&self, a: &Prepared, b: &Prepared) -> Matrix {
        match (self, pair!(a, b)) {
            (BuiltCovar::Standard(k), (Some((a, b)), _)) => k.cross_scaled(a, b),
            (BuiltCovar::Tree(t), (_, Some((a, b)))) => t.cross(a, b),
            _ => unreachable!("points prepared for another kernel"),
        }
    }

    /// `K(A, A)` (full matrix).
    pub fn symmetric(&self, a: &Prepared) -> Matrix {
        match (self, a) {
            (BuiltCovar::Standard(k), Prepared::Standard(a)) => k.symmetric_scaled(a),
            (BuiltCovar::Tree(t), Prepared::Tree(a)) => t.symmetric(a),
            _ => unreachable!("points prepared for another kernel"),
        }
    }

    /// `k(a_i, a_i)` (GPyTorch's `diag=True` mode, the prior variance).
    pub fn diag(&self, a: &Prepared, i: usize) -> f64 {
        match (self, a) {
            (BuiltCovar::Standard(k), _) => k.prior_variance(),
            (BuiltCovar::Tree(t), Prepared::Tree(a)) => t.diag(a, i),
            _ => unreachable!("points prepared for another kernel"),
        }
    }

    /// `k(a_i, a_i)` and its derivative with respect to `a_i`, or `None` for
    /// the derivative if it vanishes everywhere (stationary kernels).
    pub fn diag_with_grad(&self, a: &Prepared, i: usize) -> (f64, Option<Vector>) {
        match (self, a) {
            (BuiltCovar::Standard(k), _) => (k.prior_variance(), None),
            (BuiltCovar::Tree(t), Prepared::Tree(a)) => {
                if t.has_constant_diag() {
                    (t.diag(a, i), None)
                } else {
                    let (v, g) = t.diag_with_grad(a, i);
                    (v, Some(g))
                }
            }
            _ => unreachable!("points prepared for another kernel"),
        }
    }

    /// `k(a_i, b_j)` for all `j` and the derivatives with respect to the
    /// (unprepared) coordinates of `a_i` (`n_b × d`).
    pub fn cross_with_grad(&self, a: &Prepared, i: usize, b: &Prepared) -> (Vector, Matrix) {
        match (self, pair!(a, b)) {
            (BuiltCovar::Standard(k), (Some((a, b)), _)) => k.cross_with_grad_scaled(a.row(i), b),
            (BuiltCovar::Tree(t), (_, Some((a, b)))) => t.cross_with_grad(a, i, b),
            _ => unreachable!("points prepared for another kernel"),
        }
    }

    /// The standard kernel, if this is one.
    pub fn as_standard(&self) -> Option<&Kernel> {
        match self {
            BuiltCovar::Standard(k) => Some(k),
            BuiltCovar::Tree(_) => None,
        }
    }

    /// The kernel tree, if this is one.
    pub fn as_tree(&self) -> Option<&TreeKernel> {
        match self {
            BuiltCovar::Standard(_) => None,
            BuiltCovar::Tree(t) => Some(t),
        }
    }
}
