//! Evaluation of kernel trees ([`KernelNode`]) with concrete hyperparameter
//! values: kernel matrices, GPyTorch's `diag=True` values, derivatives with
//! respect to the inputs and contractions of the derivatives with respect to
//! the hyperparameters (what the gradient of the marginal log likelihood
//! needs), and the log densities of the priors.
//!
//! The arithmetic follows GPyTorch's `forward` methods (e.g. inputs divided
//! by the lengthscales before distances are taken, the Matérn and cosine
//! distances clamped at `1e-15`). Prior variances `k(x, x)`
//! ([`TreeKernel::diag`]) are the diagonal of the full kernel matrix, which
//! is what BoTorch's posterior variances use: GPyTorch evaluates the test
//! covariances eagerly (`settings.max_eager_kernel_size`, 2048 points). Its
//! lazy `diag=True` mode ([`TreeKernel::diag_mode`]) differs where kernels
//! clamp (the Matérn-1/2 distance by `1e-15`, the arc-cosine argument of
//! `InfiniteWidthBNNKernel` by `acos_eps`).
//!
//! Points are first *prepared* for a tree ([`TreePrep`]: every kernel's
//! inputs selected by its `active_dims` and scaled by its lengthscales,
//! index kernels' task indices validated, ...), so that repeated
//! evaluations against the same points (e.g. the training inputs) do not
//! repeat that work.

use std::f64::consts::PI;

use crate::error::{BotorchError, Result};
use crate::kernels::nonfinite::{self, TrainingNan};
use crate::kernels::oak::{gauss_legendre_01, OAK_NORMALIZER_EPS};
use crate::kernels::tree::{KernelNode, KernelType, MatrixPrior, ParamSpec};
use crate::kernels::{KernelKind, MATERN_MIN_DISTANCE};
use crate::priors::Prior;
use crate::{Matrix, Vector};

/// Clamp of the distances of `torch.cdist`-based kernels (`gpytorch.kernels.kernel.dist`).
const MIN_DIST: f64 = MATERN_MIN_DISTANCE;

/// A kernel tree with concrete hyperparameter values.
#[derive(Clone, Debug)]
pub struct TreeKernel {
    root: Node,
    input_dim: usize,
    /// The actual (constrained) values of all parameters (flat).
    values: Vec<f64>,
    /// Scalar priors: `(offset, length, prior)` in `named_priors()` order.
    scalar_priors: Vec<(usize, usize, Prior)>,
    /// SAAS priors on `tau / ℓⱼ²` (see [`crate::kernels::SaasPrior`]).
    saas_priors: Vec<SaasTerm>,
}

/// A SAAS prior of a [`TreeKernel`]: the prior on `tau / ℓⱼ²` for the
/// lengthscales at `ls_off .. ls_off + ls_len` (actual values), with a
/// fixed `tau` or the one at offset `tau_off`.
#[derive(Clone, Debug)]
struct SaasTerm {
    ls_off: usize,
    ls_len: usize,
    tau: Option<f64>,
    tau_off: usize,
    prior: Prior,
}

/// The SAAS priors of a kernel tree (pre-order, offsets in the flat
/// parameter vector).
fn collect_saas(node: &KernelNode, off: &mut usize, out: &mut Vec<SaasTerm>) {
    let start = *off;
    let mut tau_off = start;
    let mut pos = start;
    for p in node.params() {
        if p.name == "raw_tau" {
            tau_off = pos;
        }
        pos += p.numel();
    }
    if let Some(saas) = node.saas() {
        out.push(SaasTerm {
            ls_off: start,
            ls_len: node.params()[0].numel(),
            tau: saas.tau,
            tau_off,
            prior: saas.prior,
        });
    }
    *off = pos;
    for child in node.children() {
        collect_saas(child, off, out);
    }
}

/// The inputs of a [`TreeKernel`] prepared for evaluation.
#[derive(Clone, Debug)]
pub struct TreePrep {
    n: usize,
    root: NodePrep,
}

impl TreePrep {
    /// The number of points.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether there are no points.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

#[derive(Clone, Debug)]
struct NodePrep {
    /// Number of points.
    n: usize,
    /// Columns of `x`.
    k: usize,
    /// Row-major `n × k` data (selected, and scaled where the kernel scales).
    x: Vec<f64>,
    /// Kernel-specific per-point data.
    aux: Vec<f64>,
    kids: Vec<NodePrep>,
}

impl NodePrep {
    #[inline]
    fn row(&self, i: usize) -> &[f64] {
        &self.x[i * self.k..(i + 1) * self.k]
    }
}

#[derive(Clone, Debug)]
struct Node {
    /// `active_dims` (columns of the parent's inputs).
    dims: Option<Vec<usize>>,
    /// Number of inputs of the kernel (after `active_dims`).
    k: usize,
    /// Offset of the node's first parameter in the flat vector.
    off: usize,
    kind: NodeKind,
}

#[derive(Clone, Debug)]
enum NodeKind {
    Stationary {
        kind: KernelKind,
        ls: Vec<f64>,
        shared: bool,
    },
    Rq {
        ls: Vec<f64>,
        shared: bool,
        alpha: f64,
    },
    Periodic {
        ls: Vec<f64>,
        ls_shared: bool,
        period: Vec<f64>,
        period_shared: bool,
    },
    Linear {
        var: Vec<f64>,
        shared: bool,
    },
    Polynomial {
        offset: f64,
        power: f64,
    },
    Cosine {
        period: f64,
    },
    Index(Box<IndexState>),
    Categorical {
        ls: Vec<f64>,
        shared: bool,
    },
    Downsampling {
        power: f64,
        offset: f64,
    },
    ExpDecay {
        ls: Vec<f64>,
        shared: bool,
        power: f64,
        offset: f64,
    },
    Bnn {
        w: f64,
        b: f64,
        depth: usize,
        eps: f64,
    },
    Ltf {
        fid: Vec<usize>,
        nonfid: Vec<usize>,
        power: f64,
        unbiased: Box<Node>,
        biased: Box<Node>,
    },
    Oak(Box<OakState>),
    Scale {
        s: f64,
        base: Box<Node>,
    },
    Additive(Vec<Node>),
    Product(Vec<Node>),
}

/// The task covariance of an `IndexKernel` / `PositiveIndexKernel`.
#[derive(Clone, Debug)]
struct IndexState {
    num_tasks: usize,
    /// `F` (`T × r`, actual values).
    f: Matrix,
    /// `C = F Fᵀ + diag(v)`.
    c: Matrix,
    /// The task covariance `B` (`C`, normalized by `C[t*, t*]` for a
    /// `PositiveIndexKernel` with `unit_scale_for_target`).
    b: Matrix,
    /// `Some(t*)` for a normalized `PositiveIndexKernel`.
    target: Option<usize>,
    positive: bool,
    prior: Option<MatrixPrior>,
    /// Offsets of `F` and `v` in the flat vector.
    f_off: usize,
    v_off: usize,
}

/// An `OrthogonalAdditiveKernel`.
#[derive(Clone, Debug)]
struct OakState {
    dim: usize,
    kind: KernelKind,
    /// Base lengthscale of every input dimension.
    ls: Vec<f64>,
    shared: bool,
    offset: f64,
    c1: Vec<f64>,
    /// Upper-triangle second-order coefficients (row-major), if any.
    c2: Option<Vec<f64>>,
    /// Quadrature nodes and weights on `[0, 1]`.
    z: Vec<f64>,
    w: Vec<f64>,
    /// Normalizers `max(wᵀ K_i(z, z) w, eps)` and whether they were clamped.
    norm: Vec<f64>,
    clamped: Vec<bool>,
    check_hypercube: bool,
    /// Offsets of the base lengthscales, the offset, `c1` and `c2`.
    ls_off: usize,
}

// ----------------------------------------------------------------------
// Building
// ----------------------------------------------------------------------

struct Cursor<'a> {
    raw: &'a [f64],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, p: &ParamSpec) -> (usize, Vec<f64>) {
        let off = self.pos;
        let n = p.numel();
        let v = self.raw[off..off + n]
            .iter()
            .map(|&r| p.transform(r))
            .collect();
        self.pos += n;
        (off, v)
    }
}

fn expand(v: Vec<f64>, k: usize) -> (Vec<f64>, bool) {
    if v.len() == 1 && k != 1 {
        (vec![v[0]; k], true)
    } else {
        let shared = v.len() == 1;
        (v, shared)
    }
}

fn finite(values: &[f64], what: &str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(BotorchError::InvalidArgument(format!(
            "{what} must be finite, got {values:?}"
        )));
    }
    Ok(())
}

impl TreeKernel {
    /// The kernel `node` for `input_dim` inputs with the raw parameter
    /// values `raw` (flat, in `named_parameters()` order).
    ///
    /// Like GPyTorch, any finite values are accepted (constraints that are
    /// not enforced may give e.g. negative lengthscales, which enter the
    /// kernel as in GPyTorch; zero lengthscales give non-finite values).
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of values or
    /// inconsistent dimensions (see [`KernelNode::validate`]) and
    /// [`BotorchError::InvalidArgument`] for non-finite raw values.
    pub fn build(node: &KernelNode, raw: &[f64], input_dim: usize) -> Result<Self> {
        if raw.len() != node.num_raw() {
            return Err(BotorchError::DimensionMismatch(format!(
                "expected {} raw kernel parameters, got {}",
                node.num_raw(),
                raw.len()
            )));
        }
        node.validate(input_dim)?;
        finite(raw, "raw kernel parameters")?;
        let mut cur = Cursor { raw, pos: 0 };
        let root = build_node(node, &mut cur, input_dim)?;
        let values = node.transform_all(raw);
        let mut scalar_priors = Vec::new();
        let mut off = 0;
        for p in node.all_params() {
            if let Some(prior) = p.prior {
                scalar_priors.push((off, p.numel(), prior));
            }
            off += p.numel();
        }
        let mut saas_priors = Vec::new();
        collect_saas(node, &mut 0, &mut saas_priors);
        Ok(Self {
            root,
            input_dim,
            values,
            scalar_priors,
            saas_priors,
        })
    }

    /// The number of inputs.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// The actual (constrained) values of all parameters (flat).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Whether `k(x, x)` (in `diag=True` mode) is the same for every `x`
    /// (then its derivative with respect to `x` vanishes).
    pub fn has_constant_diag(&self) -> bool {
        self.root.constant_diag()
    }

    /// Prepares the rows of `x` (`n × input_dim`) for evaluation.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns,
    /// [`BotorchError::Runtime`] for task indices out of range (index
    /// kernels, torch's `RuntimeError`), [`BotorchError::InvalidArgument`]
    /// for inputs outside the unit cube
    /// (`OrthogonalAdditiveKernel`, BoTorch's `ValueError`).
    pub fn prepare(&self, x: &Matrix) -> Result<TreePrep> {
        if x.ncols() != self.input_dim {
            return Err(BotorchError::DimensionMismatch(format!(
                "the kernel has {} inputs, got points with {} columns",
                self.input_dim,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let data: Vec<f64> = (0..n)
            .flat_map(|i| (0..x.ncols()).map(move |j| x[(i, j)]))
            .collect();
        Ok(TreePrep {
            n,
            root: self.root.prepare(&data, n, x.ncols())?,
        })
    }

    /// GPyTorch's NaN entries of the training covariance `K(X, X)` (`X`
    /// the `n × input_dim` training inputs) and the parameters (flat
    /// offsets) whose gradient is NaN, for the RBF, Matérn, rational
    /// quadratic, cosine and periodic kernels of the tree (see
    /// [`nonfinite`]; other kernels never contribute). `inputs_require_grad`:
    /// the inputs depend on learned parameters, which selects GPyTorch's
    /// autograd path also for kernels with a single lengthscale.
    pub fn training_nan(&self, x: &Matrix, inputs_require_grad: bool) -> TrainingNan {
        let mut out = TrainingNan::default();
        let (n, d) = x.shape();
        if n == 0 || d != self.input_dim {
            return out;
        }
        let data: Vec<f64> = (0..n)
            .flat_map(|i| (0..d).map(move |j| x[(i, j)]))
            .collect();
        self.root
            .training_nan(&data, n, d, inputs_require_grad, &mut out);
        out
    }

    /// `K(A, B)` (`n_a × n_b`).
    pub fn cross(&self, a: &TreePrep, b: &TreePrep) -> Matrix {
        self.root.cross(&a.root, &b.root)
    }

    /// `K(A, A)`, the full (symmetric) kernel matrix.
    pub fn symmetric(&self, a: &TreePrep) -> Matrix {
        self.root.symmetric(&a.root)
    }

    /// `k(a_i, a_i)`, the diagonal of the kernel matrix `K(A, A)`.
    pub fn diag(&self, a: &TreePrep, i: usize) -> f64 {
        self.root.value(&a.root, i, &a.root, i, true)
    }

    /// `k(a_i, a_i)` in GPyTorch's lazy `diag=True` mode (`kernel(x,
    /// diag=True)`).
    pub fn diag_mode(&self, a: &TreePrep, i: usize) -> f64 {
        self.root.diag_mode(&a.root, i)
    }

    /// `k(a_i, b_j)` for all `j` and the derivatives with respect to the
    /// coordinates of `a_i` (`n_b × input_dim`).
    pub fn cross_with_grad(&self, a: &TreePrep, i: usize, b: &TreePrep) -> (Vector, Matrix) {
        let (k, g) = self.root.cross_grad(&a.root, i, &b.root);
        (k, scatter_rows(&self.root, g, self.input_dim))
    }

    /// `k(a_i, a_i)` (see [`TreeKernel::diag`]) and its derivative with
    /// respect to the coordinates of `a_i` (`2 ∂₁k(a_i, a_i)`, the kernel
    /// being symmetric).
    pub fn diag_with_grad(&self, a: &TreePrep, i: usize) -> (f64, Vector) {
        let row = select_rows(&a.root, i);
        let (_, g) = self.root.cross_grad(&row, 0, &row);
        let g = scatter_rows(&self.root, g, self.input_dim);
        (self.diag(a, i), g.row(0).transpose() * 2.0)
    }

    /// `Σᵢⱼ Wᵢⱼ ∂K(A, A)ᵢⱼ/∂θ` for every parameter `θ` (actual values, flat
    /// order), for a symmetric `W`.
    pub fn contract(&self, a: &TreePrep, w: &Matrix) -> Vec<f64> {
        let mut out = vec![0.0; self.values.len()];
        self.root.contract(&a.root, w, &mut out);
        out
    }

    /// The covariance matrix `K(x1, x2)` of raw points.
    ///
    /// # Errors
    /// See [`TreeKernel::prepare`].
    pub fn covariance(&self, x1: &Matrix, x2: &Matrix) -> Result<Matrix> {
        Ok(self.cross(&self.prepare(x1)?, &self.prepare(x2)?))
    }

    /// The covariance matrix `K(x, x)` of raw points.
    ///
    /// # Errors
    /// See [`TreeKernel::prepare`].
    pub fn covariance_symmetric(&self, x: &Matrix) -> Result<Matrix> {
        Ok(self.symmetric(&self.prepare(x)?))
    }

    // ------------------------------------------------------------------
    // Priors
    // ------------------------------------------------------------------

    /// The sum of the log densities of all priors (GPyTorch's
    /// `named_priors()` evaluated on the actual values).
    ///
    /// # Errors
    /// [`BotorchError::InvalidArgument`] for a value outside the support
    /// that torch validates (BoTorch's `ValueError`), and for an LKJ prior
    /// on a matrix that is not positive definite.
    pub fn log_prior(&self) -> Result<f64> {
        let mut grad = vec![0.0; self.values.len()];
        self.log_prior_impl(&mut grad, false, None)
    }

    /// [`TreeKernel::log_prior`] and its gradient with respect to the
    /// actual values (flat).
    ///
    /// # Errors
    /// See [`TreeKernel::log_prior`].
    pub fn log_prior_and_grad(&self) -> Result<(f64, Vec<f64>)> {
        let mut grad = vec![0.0; self.values.len()];
        let v = self.log_prior_impl(&mut grad, true, None)?;
        Ok((v, grad))
    }

    /// `scale` times the gradient of [`TreeKernel::log_prior`], with the
    /// scale applied in the order of torch's backward pass (see
    /// [`Prior::scaled_grad_log_prob`]): finite wherever GPyTorch's
    /// gradient is, also for subnormal parameter values.
    ///
    /// # Errors
    /// See [`TreeKernel::log_prior`].
    pub fn log_prior_scaled_grad(&self, scale: f64) -> Result<Vec<f64>> {
        let mut grad = vec![0.0; self.values.len()];
        self.log_prior_impl(&mut grad, true, Some(scale))?;
        Ok(grad)
    }

    fn log_prior_impl(&self, grad: &mut [f64], with_grad: bool, scale: Option<f64>) -> Result<f64> {
        let mut total = 0.0;
        for &(off, len, prior) in &self.scalar_priors {
            let mut s = 0.0;
            for (j, &v) in self.values[off..off + len].iter().enumerate() {
                check_support(&prior, v)?;
                s += prior.log_prob(v);
                if with_grad {
                    grad[off + j] += match scale {
                        None => prior.grad_log_prob(v),
                        Some(c) => prior.scaled_grad_log_prob(v, c),
                    };
                }
            }
            total += s;
        }
        // SAAS priors: `prior(tau / l_j^2)` for every lengthscale.
        for t in &self.saas_priors {
            let tau = t.tau.unwrap_or(self.values[t.tau_off]);
            for j in 0..t.ls_len {
                let l = self.values[t.ls_off + j];
                let u = tau / (l * l);
                check_support(&t.prior, u)?;
                total += t.prior.log_prob(u);
                if with_grad {
                    let g = t.prior.grad_log_prob(u) * scale.unwrap_or(1.0);
                    grad[t.ls_off + j] += g * (-2.0 * u / l);
                    if t.tau.is_none() {
                        grad[t.tau_off] += g / (l * l);
                    }
                }
            }
        }
        let mut index_grad = vec![0.0; if scale.is_some() { grad.len() } else { 0 }];
        self.root.visit_index(&mut |st| -> Result<()> {
            if let Some(prior) = &st.prior {
                let g = if scale.is_some() {
                    &mut index_grad[..]
                } else {
                    &mut grad[..]
                };
                total += index_prior(st, prior, g, with_grad)?;
            }
            Ok(())
        })?;
        if let Some(c) = scale {
            for (a, b) in grad.iter_mut().zip(index_grad) {
                *a += b * c;
            }
        }
        Ok(total)
    }
}

fn check_support(prior: &Prior, v: f64) -> Result<()> {
    if !prior.in_validated_support(v) {
        return Err(BotorchError::InvalidArgument(format!(
            "Expected value argument ({v}) to be within the support of the distribution \
             {}Prior{:?}, but found invalid values",
            prior.name(),
            prior.parameters()
        )));
    }
    Ok(())
}

fn build_node(spec: &KernelNode, cur: &mut Cursor<'_>, parent_dim: usize) -> Result<Node> {
    let dims = spec.active_dims().map(<[usize]>::to_vec);
    let k = dims.as_ref().map_or(parent_dim, Vec::len);
    let off = cur.pos;
    let params = spec.params();
    let kind = match spec.kind() {
        KernelType::Stationary(kind) => {
            let (_, ls) = cur.take(&params[0]);
            // (A SAAS prior's `raw_tau` enters the priors only.)
            for p in &params[1..] {
                cur.take(p);
            }
            let (ls, shared) = expand(ls, k);
            NodeKind::Stationary {
                kind: *kind,
                ls,
                shared,
            }
        }
        KernelType::RationalQuadratic => {
            let (_, ls) = cur.take(&params[0]);
            let (ls, shared) = expand(ls, k);
            let (_, alpha) = cur.take(&params[1]);
            NodeKind::Rq {
                ls,
                shared,
                alpha: alpha[0],
            }
        }
        KernelType::Periodic => {
            let (_, ls) = cur.take(&params[0]);
            let (ls, ls_shared) = expand(ls, k);
            let (_, period) = cur.take(&params[1]);
            let (period, period_shared) = expand(period, k);
            NodeKind::Periodic {
                ls,
                ls_shared,
                period,
                period_shared,
            }
        }
        KernelType::Linear => {
            let (_, var) = cur.take(&params[0]);
            let (var, shared) = expand(var, k);
            NodeKind::Linear { var, shared }
        }
        KernelType::Polynomial { power } => {
            let (_, offset) = cur.take(&params[0]);
            finite(&offset, "offset")?;
            NodeKind::Polynomial {
                offset: offset[0],
                power: *power,
            }
        }
        KernelType::Cosine => {
            let (_, period) = cur.take(&params[0]);
            NodeKind::Cosine { period: period[0] }
        }
        KernelType::Index { num_tasks, rank } => {
            let (f_off, f) = cur.take(&params[0]);
            let (v_off, var) = cur.take(&params[1]);
            finite(&var, "task variances")?;
            let f = Matrix::from_row_slice(*num_tasks, *rank, &f);
            NodeKind::Index(Box::new(index_state(
                *num_tasks,
                f,
                var,
                None,
                false,
                spec.task_prior().cloned(),
                f_off,
                v_off,
            )?))
        }
        KernelType::PositiveIndex {
            num_tasks,
            rank,
            target_task_index,
            unit_scale_for_target,
        } => {
            let (v_off, var) = cur.take(&params[0]);
            let (f_off, f) = cur.take(&params[1]);
            finite(&var, "task variances")?;
            let f = Matrix::from_row_slice(*num_tasks, *rank, &f);
            let target = unit_scale_for_target.then_some(*target_task_index);
            NodeKind::Index(Box::new(index_state(
                *num_tasks,
                f,
                var,
                target,
                true,
                spec.task_prior().cloned(),
                f_off,
                v_off,
            )?))
        }
        KernelType::Categorical => {
            let (_, ls) = cur.take(&params[0]);
            let (ls, shared) = expand(ls, k);
            NodeKind::Categorical { ls, shared }
        }
        KernelType::Downsampling => {
            let (_, power) = cur.take(&params[0]);
            let (_, offset) = cur.take(&params[1]);
            finite(&[power[0], offset[0]], "power and offset")?;
            NodeKind::Downsampling {
                power: power[0],
                offset: offset[0],
            }
        }
        KernelType::ExponentialDecay => {
            let (_, ls) = cur.take(&params[0]);
            let (ls, shared) = expand(ls, k);
            let (_, power) = cur.take(&params[1]);
            let (_, offset) = cur.take(&params[2]);
            finite(&[power[0], offset[0]], "power and offset")?;
            NodeKind::ExpDecay {
                ls,
                shared,
                power: power[0],
                offset: offset[0],
            }
        }
        KernelType::InfiniteWidthBnn { depth, acos_eps } => {
            let (_, w) = cur.take(&params[0]);
            let (_, b) = cur.take(&params[1]);
            finite(&[w[0], b[0]], "weight and bias variances")?;
            NodeKind::Bnn {
                w: w[0],
                b: b[0],
                depth: *depth,
                eps: *acos_eps,
            }
        }
        KernelType::LinearTruncatedFidelity { fidelity_dims } => {
            let (_, power) = cur.take(&params[0]);
            finite(&power, "power")?;
            let nonfid: Vec<usize> = (0..k).filter(|j| !fidelity_dims.contains(j)).collect();
            let children = spec.children();
            let unbiased = build_node(&children[0], cur, nonfid.len())?;
            let biased = build_node(&children[1], cur, nonfid.len())?;
            NodeKind::Ltf {
                fid: fidelity_dims.clone(),
                nonfid,
                power: power[0],
                unbiased: Box::new(unbiased),
                biased: Box::new(biased),
            }
        }
        KernelType::OrthogonalAdditive {
            dim,
            quad_deg,
            second_order,
            check_hypercube,
            per_dim_lengthscales: _,
        } => {
            let (_, offset) = cur.take(&params[0]);
            let (_, c1) = cur.take(&params[1]);
            let c2 = if *second_order {
                Some(cur.take(&params[2]).1)
            } else {
                None
            };
            finite(&offset, "offset")?;
            finite(&c1, "first-order coefficients")?;
            let base = &spec.children()[0];
            let KernelType::Stationary(kind) = base.kind() else {
                return Err(BotorchError::Unsupported(
                    "the base kernel of an OrthogonalAdditiveKernel must be an RBFKernel or a \
                     MaternKernel"
                        .into(),
                ));
            };
            let ls_off = cur.pos;
            let (_, ls) = cur.take(&base.params()[0]);
            let (ls, shared) = expand(ls, *dim);
            let (z, w) = gauss_legendre_01(*quad_deg);
            let mut st = OakState {
                dim: *dim,
                kind: *kind,
                ls,
                shared,
                offset: offset[0],
                c1,
                c2,
                z,
                w,
                norm: Vec::new(),
                clamped: Vec::new(),
                check_hypercube: *check_hypercube,
                ls_off,
            };
            st.compute_normalizers();
            NodeKind::Oak(Box::new(st))
        }
        KernelType::Scale => {
            let (_, s) = cur.take(&params[0]);
            let base = build_node(&spec.children()[0], cur, k)?;
            NodeKind::Scale {
                s: s[0],
                base: Box::new(base),
            }
        }
        KernelType::Additive => NodeKind::Additive(
            spec.children()
                .iter()
                .map(|c| build_node(c, cur, k))
                .collect::<Result<_>>()?,
        ),
        KernelType::Product => NodeKind::Product(
            spec.children()
                .iter()
                .map(|c| build_node(c, cur, k))
                .collect::<Result<_>>()?,
        ),
    };
    Ok(Node { dims, k, off, kind })
}

#[allow(clippy::too_many_arguments)]
fn index_state(
    num_tasks: usize,
    f: Matrix,
    var: Vec<f64>,
    target: Option<usize>,
    positive: bool,
    prior: Option<MatrixPrior>,
    f_off: usize,
    v_off: usize,
) -> Result<IndexState> {
    let mut c = &f * f.transpose();
    for (t, v) in var.iter().enumerate() {
        c[(t, t)] += v;
    }
    let b = match target {
        Some(t) => {
            // A target variance that underflowed to zero gives NaN entries,
            // as in torch, which BoTorch's fit turns into a NaN loss
            // (linear_operator's `NanError`).
            let s = c[(t, t)];
            let b = c.map(|v| v / s);
            let count = b.iter().filter(|v| v.is_nan()).count();
            if count > 0 {
                return Err(BotorchError::Nan {
                    count,
                    total: b.len(),
                });
            }
            b
        }
        None => c.clone(),
    };
    Ok(IndexState {
        num_tasks,
        f,
        c,
        b,
        target,
        positive,
        prior,
        f_off,
        v_off,
    })
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `base^power` as torch's `pow` with a Python number exponent.
#[inline]
fn pow(base: f64, power: f64) -> f64 {
    if power == power.trunc() && power.abs() <= 64.0 {
        base.powi(power as i32)
    } else {
        base.powf(power)
    }
}

/// `x^e` and its derivative with respect to `e` as torch computes it
/// (`x^e · log(x)`, zero for `x = 0` and `e ≥ 0`).
#[inline]
fn pow_and_dexp(x: f64, e: f64) -> (f64, f64) {
    let v = x.powf(e);
    let d = if x == 0.0 && e >= 0.0 {
        0.0
    } else {
        v * x.ln()
    };
    (v, d)
}

fn symmetric_from<F: FnMut(usize, usize) -> f64>(n: usize, mut f: F) -> Matrix {
    let mut m = Matrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let v = f(i, j);
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    m
}

/// Maps the derivatives with respect to a node's inputs (`n × node.k`) to
/// those with respect to its parent's inputs (`n × parent_k`).
fn scatter_rows(node: &Node, g: Matrix, parent_k: usize) -> Matrix {
    match &node.dims {
        None => g,
        Some(dims) => {
            let mut out = Matrix::zeros(g.nrows(), parent_k);
            for (c, &j) in dims.iter().enumerate() {
                for r in 0..g.nrows() {
                    out[(r, j)] += g[(r, c)];
                }
            }
            out
        }
    }
}

/// The prepared point `i` of `a` alone (a one-point set).
fn select_rows(a: &NodePrep, i: usize) -> NodePrep {
    let k = a.k;
    let x = if a.x.is_empty() {
        Vec::new()
    } else {
        a.x[i * k..(i + 1) * k].to_vec()
    };
    // Per-point data is stored point-major.
    let per = a.aux.len().checked_div(a.n).unwrap_or(0);
    let aux = a.aux[i * per..(i + 1) * per].to_vec();
    NodePrep {
        n: 1,
        k,
        x,
        aux,
        kids: a.kids.iter().map(|c| select_rows(c, i)).collect(),
    }
}

/// `x[:, cols]` of a row-major `n × k` buffer.
fn select(x: &[f64], n: usize, k: usize, cols: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(n * cols.len());
    for i in 0..n {
        for &c in cols {
            out.push(x[i * k + c]);
        }
    }
    out
}

/// Accumulates `Σ_{i<j} (W_ij + W_ji) f(i, j) + Σ_i W_ii f(i, i)` style
/// contractions: calls `f(i, j, w)` for every pair `i ≤ j` with the weight
/// of the pair (`W_ii` on the diagonal, `W_ij + W_ji` off it).
fn for_pairs<F: FnMut(usize, usize, f64)>(w: &Matrix, diagonal: bool, mut f: F) {
    let n = w.nrows();
    for i in 0..n {
        if diagonal {
            f(i, i, w[(i, i)]);
        }
        for j in (i + 1)..n {
            f(i, j, w[(i, j)] + w[(j, i)]);
        }
    }
}

// ----------------------------------------------------------------------
// Evaluation
// ----------------------------------------------------------------------

impl Node {
    /// See [`TreeKernel::training_nan`]: `data` holds the parent's `n ×
    /// parent_k` inputs (row-major). Unless the inputs require gradients, a
    /// single lengthscale selects GPyTorch's `RBFCovariance` /
    /// `MaternCovariance` functions.
    fn training_nan(
        &self,
        data: &[f64],
        n: usize,
        parent_k: usize,
        grad: bool,
        out: &mut TrainingNan,
    ) {
        let k = self.k;
        let local: Vec<f64> = match &self.dims {
            Some(dims) => select(data, n, parent_k, dims),
            None => data.to_vec(),
        };
        match &self.kind {
            NodeKind::Stationary { kind, ls, shared } => {
                let r = nonfinite::stationary(*kind, &local, n, k, ls, *shared && !grad);
                if r.any() {
                    out.merge(r);
                } else if !r.nan_gradients().is_empty() {
                    out.add_grad(self.off);
                }
            }
            NodeKind::Rq { ls, alpha, .. } => {
                out.merge(nonfinite::rational_quadratic(&local, n, k, ls, *alpha));
            }
            NodeKind::Periodic { ls, period, .. } => {
                out.merge(nonfinite::periodic(&local, n, k, ls, period));
            }
            NodeKind::Cosine { period } => out.merge(nonfinite::cosine(&local, n, k, *period)),
            NodeKind::Scale { base, .. } => base.training_nan(&local, n, k, grad, out),
            NodeKind::Additive(kids) | NodeKind::Product(kids) => {
                for kid in kids {
                    kid.training_nan(&local, n, k, grad, out);
                }
            }
            NodeKind::Ltf {
                nonfid,
                unbiased,
                biased,
                ..
            } => {
                let sub = select(&local, n, k, nonfid);
                unbiased.training_nan(&sub, n, nonfid.len(), grad, out);
                biased.training_nan(&sub, n, nonfid.len(), grad, out);
            }
            _ => {}
        }
    }

    fn constant_diag(&self) -> bool {
        match &self.kind {
            NodeKind::Stationary { .. }
            | NodeKind::Rq { .. }
            | NodeKind::Periodic { .. }
            | NodeKind::Cosine { .. }
            | NodeKind::Categorical { .. } => true,
            NodeKind::Scale { base, .. } => base.constant_diag(),
            NodeKind::Additive(kids) | NodeKind::Product(kids) => {
                kids.iter().all(Node::constant_diag)
            }
            _ => false,
        }
    }

    fn visit_index<F: FnMut(&IndexState) -> Result<()>>(&self, f: &mut F) -> Result<()> {
        match &self.kind {
            NodeKind::Index(st) => f(st),
            NodeKind::Ltf {
                unbiased, biased, ..
            } => {
                unbiased.visit_index(f)?;
                biased.visit_index(f)
            }
            NodeKind::Scale { base, .. } => base.visit_index(f),
            NodeKind::Additive(kids) | NodeKind::Product(kids) => {
                for kid in kids {
                    kid.visit_index(f)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Prepares `n` points given as a row-major `n × parent_k` buffer.
    fn prepare(&self, data: &[f64], n: usize, parent_k: usize) -> Result<NodePrep> {
        let k = self.k;
        let local: Vec<f64> = match &self.dims {
            Some(dims) => select(data, n, parent_k, dims),
            None => data.to_vec(),
        };
        let scaled = |div: &[f64]| -> Vec<f64> {
            local
                .iter()
                .enumerate()
                .map(|(i, v)| v / div[i % k])
                .collect()
        };
        let leaf = |x: Vec<f64>, aux: Vec<f64>| NodePrep {
            n,
            k,
            x,
            aux,
            kids: Vec::new(),
        };
        Ok(match &self.kind {
            NodeKind::Stationary { ls, .. }
            | NodeKind::Rq { ls, .. }
            | NodeKind::ExpDecay { ls, .. } => leaf(scaled(ls), Vec::new()),
            NodeKind::Periodic { period, .. } => {
                let div: Vec<f64> = period.iter().map(|p| p / PI).collect();
                leaf(scaled(&div), Vec::new())
            }
            NodeKind::Linear { var, .. } => {
                let sq: Vec<f64> = var.iter().map(|v| v.sqrt()).collect();
                let x = local
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * sq[i % k])
                    .collect();
                leaf(x, local)
            }
            NodeKind::Cosine { period } => leaf(local.iter().map(|v| v / period).collect(), vec![]),
            NodeKind::Index(st) => {
                let mut tasks = Vec::with_capacity(n);
                for &v in &local {
                    let t = v.trunc();
                    if !(t >= 0.0 && (t as usize) < st.num_tasks) || v.is_nan() {
                        return Err(BotorchError::Runtime(format!(
                            "task index {v} is out of range for {} tasks (index out of range)",
                            st.num_tasks
                        )));
                    }
                    tasks.push(t);
                }
                leaf(Vec::new(), tasks)
            }
            NodeKind::Downsampling { power, .. } => {
                let e = 1.0 + power;
                let a: Vec<f64> = local.iter().map(|v| 1.0 - v).collect();
                let x = a.iter().map(|v| v.powf(e)).collect();
                leaf(x, a)
            }
            NodeKind::Bnn { .. } => {
                let xx = (0..n)
                    .map(|i| local[i * k..(i + 1) * k].iter().map(|v| v * v).sum())
                    .collect();
                leaf(local, xx)
            }
            NodeKind::Polynomial { .. } | NodeKind::Categorical { .. } => leaf(local, Vec::new()),
            NodeKind::Ltf {
                nonfid,
                unbiased,
                biased,
                ..
            } => {
                let sub = select(&local, n, k, nonfid);
                let m = nonfid.len();
                let kids = vec![unbiased.prepare(&sub, n, m)?, biased.prepare(&sub, n, m)?];
                NodePrep {
                    n,
                    k,
                    x: local,
                    aux: Vec::new(),
                    kids,
                }
            }
            NodeKind::Oak(st) => {
                if st.check_hypercube && local.iter().any(|&v| !(-1e-6..=1.0 + 1e-6).contains(&v)) {
                    return Err(BotorchError::InvalidArgument(
                        "x1 is not in hypercube [0, 1]^d.".into(),
                    ));
                }
                let aux = st.projections(&local, n);
                leaf(local, aux)
            }
            NodeKind::Scale { base, .. } => NodePrep {
                n,
                k,
                x: Vec::new(),
                aux: Vec::new(),
                kids: vec![base.prepare(&local, n, k)?],
            },
            NodeKind::Additive(kids) | NodeKind::Product(kids) => NodePrep {
                n,
                k,
                x: Vec::new(),
                aux: Vec::new(),
                kids: kids
                    .iter()
                    .map(|c| c.prepare(&local, n, k))
                    .collect::<Result<_>>()?,
            },
        })
    }

    /// The kernel value of points `i` of `a` and `j` of `b` (`same`: the
    /// same point of the same set, as on the diagonal of `K(A, A)`).
    fn value(&self, a: &NodePrep, i: usize, b: &NodePrep, j: usize, same: bool) -> f64 {
        match &self.kind {
            NodeKind::Stationary { kind, .. } => {
                if same {
                    kind.value(0.0)
                } else {
                    kind.value(sq_dist(a.row(i), b.row(j)))
                }
            }
            NodeKind::Rq { alpha, .. } => {
                let r2 = if same {
                    0.0
                } else {
                    sq_dist(a.row(i), b.row(j))
                };
                (1.0 + r2 / (2.0 * alpha)).powf(-alpha)
            }
            NodeKind::Periodic { ls, .. } => {
                // GPyTorch clamps every per-dimension distance at 1e-15, also
                // on the diagonal: `exp(-2 sin²(1e-15) / ℓ)` is 1 unless the
                // lengthscale is below ~1e-14, and vanishes below ~1e-30.
                let mut e = 0.0;
                for ((x, y), l) in a.row(i).iter().zip(b.row(j)).zip(ls) {
                    let s = (x - y).abs().max(MIN_DIST).sin();
                    e += s * s / l * -2.0;
                }
                e.exp()
            }
            NodeKind::Linear { .. } => dot(a.row(i), b.row(j)),
            NodeKind::Polynomial { offset, power } => pow(offset + dot(a.row(i), b.row(j)), *power),
            NodeKind::Cosine { .. } => {
                let r = if same {
                    0.0
                } else {
                    sq_dist(a.row(i), b.row(j)).sqrt()
                };
                (r.max(MIN_DIST) * PI).cos()
            }
            NodeKind::Index(st) => st.b[(a.aux[i] as usize, b.aux[j] as usize)],
            NodeKind::Categorical { ls, .. } => {
                let mut s = 0.0;
                for ((x, y), l) in a.row(i).iter().zip(b.row(j)).zip(ls) {
                    if x != y {
                        s += 1.0 / l;
                    }
                }
                (-(s / self.k as f64)).exp()
            }
            NodeKind::Downsampling { offset, .. } => offset + dot(a.row(i), b.row(j)),
            NodeKind::ExpDecay { power, offset, .. } => {
                let s: f64 = a
                    .row(i)
                    .iter()
                    .zip(b.row(j))
                    .map(|(x, y)| (x + y) * (x + y))
                    .sum();
                offset + pow(s.sqrt().max(MIN_DIST) + 1.0, -power)
            }
            NodeKind::Bnn {
                w,
                b: bias,
                depth,
                eps,
            } => {
                let xy = dot(a.row(i), b.row(j));
                bnn_pair(
                    xy,
                    a.aux[i],
                    b.aux[j],
                    self.k as f64,
                    *w,
                    *bias,
                    *depth,
                    *eps,
                    false,
                )
                .k
            }
            NodeKind::Ltf {
                fid,
                power,
                unbiased,
                biased,
                ..
            } => {
                let ku = unbiased.value(&a.kids[0], i, &b.kids[0], j, same);
                let kb = biased.value(&a.kids[1], i, &b.kids[1], j, same);
                let bias = ltf_bias(a.row(i), b.row(j), fid, *power).0;
                ku + bias * kb
            }
            NodeKind::Oak(st) => st.value(a, i, b, j, same),
            NodeKind::Scale { s, base } => base.value(&a.kids[0], i, &b.kids[0], j, same) * s,
            NodeKind::Additive(kids) => {
                let mut v = 0.0;
                for (c, kid) in kids.iter().enumerate() {
                    v += kid.value(&a.kids[c], i, &b.kids[c], j, same);
                }
                v
            }
            NodeKind::Product(kids) => {
                let mut v = kids[0].value(&a.kids[0], i, &b.kids[0], j, same);
                for (c, kid) in kids.iter().enumerate().skip(1) {
                    v *= kid.value(&a.kids[c], i, &b.kids[c], j, same);
                }
                v
            }
        }
    }

    fn cross(&self, a: &NodePrep, b: &NodePrep) -> Matrix {
        match &self.kind {
            NodeKind::Scale { s, base } => base.cross(&a.kids[0], &b.kids[0]).map(|v| v * s),
            NodeKind::Additive(kids) => {
                let mut m = kids[0].cross(&a.kids[0], &b.kids[0]);
                for (c, kid) in kids.iter().enumerate().skip(1) {
                    m += kid.cross(&a.kids[c], &b.kids[c]);
                }
                m
            }
            NodeKind::Product(kids) => {
                let mut m = kids[0].cross(&a.kids[0], &b.kids[0]);
                for (c, kid) in kids.iter().enumerate().skip(1) {
                    m.component_mul_assign(&kid.cross(&a.kids[c], &b.kids[c]));
                }
                m
            }
            NodeKind::Oak(st) => st.cross(a, b),
            _ => Matrix::from_fn(a.n, b.n, |i, j| self.value(a, i, b, j, false)),
        }
    }

    fn symmetric(&self, a: &NodePrep) -> Matrix {
        match &self.kind {
            NodeKind::Scale { s, base } => base.symmetric(&a.kids[0]).map(|v| v * s),
            NodeKind::Additive(kids) => {
                let mut m = kids[0].symmetric(&a.kids[0]);
                for (c, kid) in kids.iter().enumerate().skip(1) {
                    m += kid.symmetric(&a.kids[c]);
                }
                m
            }
            NodeKind::Product(kids) => {
                let mut m = kids[0].symmetric(&a.kids[0]);
                for (c, kid) in kids.iter().enumerate().skip(1) {
                    m.component_mul_assign(&kid.symmetric(&a.kids[c]));
                }
                m
            }
            NodeKind::Oak(st) => st.symmetric(a),
            NodeKind::Ltf {
                fid,
                power,
                unbiased,
                biased,
                ..
            } => {
                let ku = unbiased.symmetric(&a.kids[0]);
                let kb = biased.symmetric(&a.kids[1]);
                let bias = symmetric_from(a.n, |i, j| ltf_bias(a.row(i), a.row(j), fid, *power).0);
                ku + bias.component_mul(&kb)
            }
            _ => symmetric_from(a.n, |i, j| self.value(a, i, a, j, i == j)),
        }
    }

    fn diag_mode(&self, a: &NodePrep, i: usize) -> f64 {
        match &self.kind {
            NodeKind::Stationary { kind, .. } => kind.diag_value(),
            // Unclamped zero distances: `exp(-2 Σ sin²(0) / ℓ)`, which is NaN
            // for a lengthscale that underflowed to zero, as in GPyTorch.
            NodeKind::Periodic { ls, .. } => ls.iter().map(|l| 0.0 / l * -2.0).sum::<f64>().exp(),
            NodeKind::Rq { .. } | NodeKind::Cosine { .. } | NodeKind::Categorical { .. } => 1.0,
            NodeKind::ExpDecay { power, offset, .. } => {
                // `torch.linalg.norm(x1 - (-x2))` without clamping.
                let s: f64 = a.row(i).iter().map(|x| (x + x) * (x + x)).sum();
                offset + pow(s.sqrt() + 1.0, -power)
            }
            NodeKind::Bnn { w, b, depth, .. } => {
                let mut v = w * a.aux[i] / self.k as f64 + b;
                for _ in 0..*depth {
                    v = w * v / 2.0 + b;
                }
                v
            }
            NodeKind::Ltf {
                fid,
                power,
                unbiased,
                biased,
                ..
            } => {
                let ku = unbiased.diag_mode(&a.kids[0], i);
                let kb = biased.diag_mode(&a.kids[1], i);
                ku + ltf_bias(a.row(i), a.row(i), fid, *power).0 * kb
            }
            NodeKind::Oak(st) => st.diag_mode(a, i),
            NodeKind::Scale { s, base } => base.diag_mode(&a.kids[0], i) * s,
            NodeKind::Additive(kids) => {
                let mut v = 0.0;
                for (c, kid) in kids.iter().enumerate() {
                    v += kid.diag_mode(&a.kids[c], i);
                }
                v
            }
            NodeKind::Product(kids) => {
                let mut v = kids[0].diag_mode(&a.kids[0], i);
                for (c, kid) in kids.iter().enumerate().skip(1) {
                    v *= kid.diag_mode(&a.kids[c], i);
                }
                v
            }
            _ => self.value(a, i, a, i, true),
        }
    }

    /// `k(a_i, b_j)` and `∂k/∂a_i` (`n_b × self.k`, local inputs).
    fn cross_grad(&self, a: &NodePrep, i: usize, b: &NodePrep) -> (Vector, Matrix) {
        let nb = b.n;
        let k = self.k;
        let mut val = Vector::zeros(nb);
        let mut grad = Matrix::zeros(nb, k);
        match &self.kind {
            NodeKind::Stationary { kind, ls, .. } => {
                let x = a.row(i);
                for j in 0..nb {
                    let y = b.row(j);
                    let (v, g) = kind.value_and_factor(sq_dist(x, y));
                    val[j] = v;
                    for m in 0..k {
                        grad[(j, m)] = -g * (x[m] - y[m]) / ls[m];
                    }
                }
            }
            NodeKind::Rq { ls, alpha, .. } => {
                let x = a.row(i);
                for j in 0..nb {
                    let y = b.row(j);
                    let base = 1.0 + sq_dist(x, y) / (2.0 * alpha);
                    let v = base.powf(-alpha);
                    val[j] = v;
                    let g = v / base;
                    for m in 0..k {
                        grad[(j, m)] = -g * (x[m] - y[m]) / ls[m];
                    }
                }
            }
            NodeKind::Periodic { ls, period, .. } => {
                let x = a.row(i);
                for j in 0..nb {
                    let y = b.row(j);
                    let v = self.value(a, i, b, j, false);
                    val[j] = v;
                    for m in 0..k {
                        let d = x[m] - y[m];
                        if d.abs() > MIN_DIST {
                            grad[(j, m)] = v * (-2.0 / ls[m]) * (2.0 * d).sin() * (PI / period[m]);
                        }
                    }
                }
            }
            NodeKind::Linear { var, .. } => {
                let x = a.row(i);
                let sq: Vec<f64> = var.iter().map(|v| v.sqrt()).collect();
                for j in 0..nb {
                    let y = b.row(j);
                    val[j] = dot(x, y);
                    for m in 0..k {
                        grad[(j, m)] = sq[m] * y[m];
                    }
                }
            }
            NodeKind::Polynomial { offset, power } => {
                let x = a.row(i);
                for j in 0..nb {
                    let y = b.row(j);
                    let base = offset + dot(x, y);
                    val[j] = pow(base, *power);
                    let d = power * pow(base, power - 1.0);
                    for m in 0..k {
                        grad[(j, m)] = d * y[m];
                    }
                }
            }
            NodeKind::Cosine { period } => {
                let x = a.row(i);
                for j in 0..nb {
                    let y = b.row(j);
                    let r = sq_dist(x, y).sqrt();
                    val[j] = (r.max(MIN_DIST) * PI).cos();
                    if r > MIN_DIST {
                        let f = -(r * PI).sin() * PI / r / period;
                        for m in 0..k {
                            grad[(j, m)] = f * (x[m] - y[m]);
                        }
                    }
                }
            }
            NodeKind::Index(_) | NodeKind::Categorical { .. } => {
                for j in 0..nb {
                    val[j] = self.value(a, i, b, j, false);
                }
            }
            NodeKind::Downsampling { power, .. } => {
                let e = 1.0 + power;
                for j in 0..nb {
                    val[j] = self.value(a, i, b, j, false);
                    for m in 0..k {
                        let am = a.aux[i * k + m];
                        grad[(j, m)] = -(e * am.powf(e - 1.0)) * b.x[j * k + m];
                    }
                }
            }
            NodeKind::ExpDecay {
                ls, power, offset, ..
            } => {
                let x = a.row(i);
                for j in 0..nb {
                    let y = b.row(j);
                    let s: f64 = x.iter().zip(y).map(|(p, q)| (p + q) * (p + q)).sum();
                    let dist = s.sqrt();
                    let base = dist.max(MIN_DIST) + 1.0;
                    val[j] = offset + pow(base, -power);
                    if dist > MIN_DIST {
                        let dd = -power * pow(base, -power - 1.0) / dist;
                        for m in 0..k {
                            grad[(j, m)] = dd * (x[m] + y[m]) / ls[m];
                        }
                    }
                }
            }
            NodeKind::Bnn {
                w,
                b: bias,
                depth,
                eps,
            } => {
                let x = a.row(i);
                for j in 0..nb {
                    let y = b.row(j);
                    let out = bnn_pair(
                        dot(x, y),
                        a.aux[i],
                        b.aux[j],
                        k as f64,
                        *w,
                        *bias,
                        *depth,
                        *eps,
                        true,
                    );
                    val[j] = out.k;
                    for m in 0..k {
                        grad[(j, m)] = out.dxy * y[m] + out.dxx * 2.0 * x[m];
                    }
                }
            }
            NodeKind::Ltf {
                fid,
                nonfid,
                power,
                unbiased,
                biased,
            } => {
                let (ku, du) = unbiased.cross_grad(&a.kids[0], i, &b.kids[0]);
                let (kb, db) = biased.cross_grad(&a.kids[1], i, &b.kids[1]);
                let du = scatter_rows(unbiased, du, nonfid.len());
                let db = scatter_rows(biased, db, nonfid.len());
                let x = a.row(i);
                for j in 0..nb {
                    let (bias, dbias, _) = ltf_bias(x, b.row(j), fid, *power);
                    val[j] = ku[j] + bias * kb[j];
                    for (c, &m) in nonfid.iter().enumerate() {
                        grad[(j, m)] = du[(j, c)] + bias * db[(j, c)];
                    }
                    for (f, &m) in fid.iter().enumerate() {
                        grad[(j, m)] = dbias[f] * kb[j];
                    }
                }
            }
            NodeKind::Oak(st) => return st.cross_grad(a, i, b),
            NodeKind::Scale { s, base } => {
                let (v, g) = base.cross_grad(&a.kids[0], i, &b.kids[0]);
                let g = scatter_rows(base, g, k);
                return (v * *s, g * *s);
            }
            NodeKind::Additive(kids) => {
                for (c, kid) in kids.iter().enumerate() {
                    let (v, g) = kid.cross_grad(&a.kids[c], i, &b.kids[c]);
                    val += v;
                    grad += scatter_rows(kid, g, k);
                }
            }
            NodeKind::Product(kids) => {
                let parts: Vec<(Vector, Matrix)> = kids
                    .iter()
                    .enumerate()
                    .map(|(c, kid)| {
                        let (v, g) = kid.cross_grad(&a.kids[c], i, &b.kids[c]);
                        (v, scatter_rows(kid, g, k))
                    })
                    .collect();
                for j in 0..nb {
                    let mut v = parts[0].0[j];
                    for p in &parts[1..] {
                        v *= p.0[j];
                    }
                    val[j] = v;
                    for (c, p) in parts.iter().enumerate() {
                        let mut others = 1.0;
                        for (d, q) in parts.iter().enumerate() {
                            if d != c {
                                others *= q.0[j];
                            }
                        }
                        for m in 0..k {
                            grad[(j, m)] += p.1[(j, m)] * others;
                        }
                    }
                }
            }
        }
        (val, grad)
    }

    /// Adds `Σᵢⱼ Wᵢⱼ ∂K(A, A)ᵢⱼ/∂θ` to `out` for the node's parameters.
    fn contract(&self, a: &NodePrep, w: &Matrix, out: &mut [f64]) {
        let k = self.k;
        let off = self.off;
        let n = a.n;
        match &self.kind {
            NodeKind::Stationary { kind, ls, shared } => {
                let mut gl = vec![0.0; k];
                for_pairs(w, false, |i, j, wij| {
                    let (x, y) = (a.row(i), a.row(j));
                    let (_, g) = kind.value_and_factor(sq_dist(x, y));
                    let c = wij * g;
                    for m in 0..k {
                        let u = x[m] - y[m];
                        gl[m] += c * u * u;
                    }
                });
                write_ls(out, off, &gl, ls, *shared);
            }
            NodeKind::Rq { ls, shared, alpha } => {
                let mut gl = vec![0.0; k];
                let mut ga = 0.0;
                for_pairs(w, false, |i, j, wij| {
                    let (x, y) = (a.row(i), a.row(j));
                    let r2 = sq_dist(x, y);
                    let z = r2 / (2.0 * alpha);
                    let base = 1.0 + z;
                    let v = base.powf(-alpha);
                    let c = wij * v / base;
                    for m in 0..k {
                        let u = x[m] - y[m];
                        gl[m] += c * u * u;
                    }
                    ga += wij * v * (z / base - base.ln());
                });
                write_ls(out, off, &gl, ls, *shared);
                out[off + if *shared { 1 } else { k }] += ga;
            }
            NodeKind::Periodic {
                ls,
                ls_shared,
                period,
                period_shared,
            } => {
                let mut gl = vec![0.0; k];
                let mut gp = vec![0.0; k];
                // Including the diagonal and clamped distances (whose sines
                // enter the lengthscale gradient, not the period gradient).
                for_pairs(w, true, |i, j, wij| {
                    let v = self.value(a, i, a, j, i == j);
                    let (x, y) = (a.row(i), a.row(j));
                    for m in 0..k {
                        let d = (x[m] - y[m]).abs();
                        let s = d.max(MIN_DIST).sin();
                        gl[m] += wij * v * 2.0 * s * s / (ls[m] * ls[m]);
                        if d > MIN_DIST {
                            gp[m] += wij * v * 2.0 * (2.0 * d).sin() * d / (ls[m] * period[m]);
                        }
                    }
                });
                let nl = if *ls_shared { 1 } else { k };
                if *ls_shared {
                    out[off] += gl.iter().sum::<f64>();
                } else {
                    for m in 0..k {
                        out[off + m] += gl[m];
                    }
                }
                if *period_shared {
                    out[off + nl] += gp.iter().sum::<f64>();
                } else {
                    for m in 0..k {
                        out[off + nl + m] += gp[m];
                    }
                }
            }
            NodeKind::Linear { shared, .. } => {
                let x = &a.aux;
                let mut gv = vec![0.0; k];
                for_pairs(w, true, |i, j, wij| {
                    for m in 0..k {
                        gv[m] += wij * x[i * k + m] * x[j * k + m];
                    }
                });
                if *shared {
                    out[off] += gv.iter().sum::<f64>();
                } else {
                    for m in 0..k {
                        out[off + m] += gv[m];
                    }
                }
            }
            NodeKind::Polynomial { offset, power } => {
                let mut g = 0.0;
                for_pairs(w, true, |i, j, wij| {
                    let base = offset + dot(a.row(i), a.row(j));
                    g += wij * power * pow(base, power - 1.0);
                });
                out[off] += g;
            }
            NodeKind::Cosine { period } => {
                let mut g = 0.0;
                for_pairs(w, false, |i, j, wij| {
                    let r = sq_dist(a.row(i), a.row(j)).sqrt();
                    if r > MIN_DIST {
                        g += wij * (r * PI).sin() * PI * r / period;
                    }
                });
                out[off] += g;
            }
            NodeKind::Index(st) => {
                let t = st.num_tasks;
                let mut m = Matrix::zeros(t, t);
                for i in 0..n {
                    for j in 0..n {
                        m[(a.aux[i] as usize, a.aux[j] as usize)] += w[(i, j)];
                    }
                }
                st.backprop_b(&m, out);
            }
            NodeKind::Categorical { ls, shared } => {
                let mut gl = vec![0.0; k];
                let kf = k as f64;
                for_pairs(w, false, |i, j, wij| {
                    let v = self.value(a, i, a, j, false);
                    let (x, y) = (a.row(i), a.row(j));
                    for m in 0..k {
                        if x[m] != y[m] {
                            gl[m] += wij * v / (kf * ls[m] * ls[m]);
                        }
                    }
                });
                if *shared {
                    out[off] += gl.iter().sum::<f64>();
                } else {
                    for m in 0..k {
                        out[off + m] += gl[m];
                    }
                }
            }
            NodeKind::Downsampling { power, .. } => {
                let e = 1.0 + power;
                let mut gp = 0.0;
                let mut go = 0.0;
                for_pairs(w, true, |i, j, wij| {
                    go += wij;
                    for m in 0..k {
                        let (pa, da) = pow_and_dexp(a.aux[i * k + m], e);
                        let (pb, db) = pow_and_dexp(a.aux[j * k + m], e);
                        gp += wij * (da * pb + pa * db);
                    }
                });
                out[off] += gp;
                out[off + 1] += go;
            }
            NodeKind::ExpDecay {
                ls, shared, power, ..
            } => {
                let mut gl = vec![0.0; k];
                let mut gp = 0.0;
                let mut go = 0.0;
                for_pairs(w, true, |i, j, wij| {
                    let (x, y) = (a.row(i), a.row(j));
                    let s: f64 = x.iter().zip(y).map(|(p, q)| (p + q) * (p + q)).sum();
                    let dist = s.sqrt();
                    let base = dist.max(MIN_DIST) + 1.0;
                    let v = pow(base, -power);
                    go += wij;
                    gp += -wij * v * base.ln();
                    if dist > MIN_DIST {
                        let dd = -power * pow(base, -power - 1.0);
                        for m in 0..k {
                            let u = x[m] + y[m];
                            gl[m] += wij * dd * (-(u * u) / (dist * ls[m]));
                        }
                    }
                });
                let nl = if *shared { 1 } else { k };
                if *shared {
                    out[off] += gl.iter().sum::<f64>();
                } else {
                    for m in 0..k {
                        out[off + m] += gl[m];
                    }
                }
                out[off + nl] += gp;
                out[off + nl + 1] += go;
            }
            NodeKind::Bnn {
                w: wv,
                b: bias,
                depth,
                eps,
            } => {
                let mut gw = 0.0;
                let mut gb = 0.0;
                for_pairs(w, true, |i, j, wij| {
                    let o = bnn_pair(
                        dot(a.row(i), a.row(j)),
                        a.aux[i],
                        a.aux[j],
                        k as f64,
                        *wv,
                        *bias,
                        *depth,
                        *eps,
                        true,
                    );
                    gw += wij * o.dw;
                    gb += wij * o.db;
                });
                out[off] += gw;
                out[off + 1] += gb;
            }
            NodeKind::Ltf {
                fid,
                power,
                unbiased,
                biased,
                ..
            } => {
                let kb = biased.symmetric(&a.kids[1]);
                let mut bias = Matrix::zeros(n, n);
                let mut gp = 0.0;
                for i in 0..n {
                    for j in i..n {
                        let (v, _, dp) = ltf_bias(a.row(i), a.row(j), fid, *power);
                        bias[(i, j)] = v;
                        bias[(j, i)] = v;
                        let wij = if i == j {
                            w[(i, i)]
                        } else {
                            w[(i, j)] + w[(j, i)]
                        };
                        gp += wij * dp * kb[(i, j)];
                    }
                }
                out[off] += gp;
                unbiased.contract(&a.kids[0], w, out);
                biased.contract(&a.kids[1], &w.component_mul(&bias), out);
            }
            NodeKind::Oak(st) => st.contract(a, w, off, out),
            NodeKind::Scale { s, base } => {
                let kbase = base.symmetric(&a.kids[0]);
                out[off] += w.component_mul(&kbase).sum();
                base.contract(&a.kids[0], &(w * *s), out);
            }
            NodeKind::Additive(kids) => {
                for (c, kid) in kids.iter().enumerate() {
                    kid.contract(&a.kids[c], w, out);
                }
            }
            NodeKind::Product(kids) => {
                let mats: Vec<Matrix> = kids
                    .iter()
                    .enumerate()
                    .map(|(c, kid)| kid.symmetric(&a.kids[c]))
                    .collect();
                for (c, kid) in kids.iter().enumerate() {
                    let mut wc = w.clone();
                    for (d, m) in mats.iter().enumerate() {
                        if d != c {
                            wc.component_mul_assign(m);
                        }
                    }
                    kid.contract(&a.kids[c], &wc, out);
                }
            }
        }
    }
}

/// Writes lengthscale derivatives `gl[m] / ℓ_m` (summed for a shared one).
fn write_ls(out: &mut [f64], off: usize, gl: &[f64], ls: &[f64], shared: bool) {
    if shared {
        out[off] += gl.iter().zip(ls).map(|(g, l)| g / l).sum::<f64>();
    } else {
        for (m, (g, l)) in gl.iter().zip(ls).enumerate() {
            out[off + m] += g / l;
        }
    }
}

// ----------------------------------------------------------------------
// Index kernels
// ----------------------------------------------------------------------

impl IndexState {
    /// Adds the gradient of `Σ M ∘ B` (with respect to `F` and `v`) to `out`
    /// for an arbitrary `M` (`T × T`).
    fn backprop_b(&self, m: &Matrix, out: &mut [f64]) {
        // dB -> dC.
        let g = match self.target {
            Some(t) => {
                let s = self.c[(t, t)];
                let mut g = m / s;
                let total = m.component_mul(&self.c).sum();
                g[(t, t)] -= total / (s * s);
                g
            }
            None => m.clone(),
        };
        let gs = &g + g.transpose();
        let gf = &gs * &self.f;
        let (t, r) = self.f.shape();
        for a in 0..t {
            for c in 0..r {
                out[self.f_off + a * r + c] += gf[(a, c)];
            }
            out[self.v_off + a] += g[(a, a)];
        }
    }
}

/// The log density of an index kernel's task prior (and its gradient).
fn index_prior(
    st: &IndexState,
    prior: &MatrixPrior,
    grad: &mut [f64],
    with_grad: bool,
) -> Result<f64> {
    let t = st.num_tasks;
    if let (true, MatrixPrior::Elementwise(p)) = (st.positive, prior) {
        // The lower-triangle correlations of the (normalized) task covariance
        // (an LKJ covariance prior acts on the covariance matrix itself, as
        // on an `IndexKernel`).
        let b = &st.b;
        let mut total = 0.0;
        let mut gb = Matrix::zeros(t, t);
        for r in 1..t {
            for c in 0..r {
                let norm = b[(r, r)].sqrt() * b[(c, c)].sqrt();
                let corr = b[(r, c)] / norm;
                check_support(p, corr)?;
                total += p.log_prob(corr);
                if with_grad {
                    let g = p.grad_log_prob(corr);
                    gb[(r, c)] += g / norm;
                    gb[(r, r)] -= 0.5 * g * corr / b[(r, r)];
                    gb[(c, c)] -= 0.5 * g * corr / b[(c, c)];
                }
            }
        }
        if with_grad {
            st.backprop_b(&gb, grad);
        }
        return Ok(total);
    }
    match prior {
        MatrixPrior::Elementwise(p) => {
            let mut total = 0.0;
            let mut gb = Matrix::zeros(t, t);
            for r in 0..t {
                for c in 0..t {
                    let v = st.b[(r, c)];
                    check_support(p, v)?;
                    total += p.log_prob(v);
                    gb[(r, c)] = p.grad_log_prob(v);
                }
            }
            if with_grad {
                st.backprop_b(&gb, grad);
            }
            Ok(total)
        }
        MatrixPrior::LkjCovariance { eta, sd_prior, .. } => {
            let (v, gb) = lkj_covariance_log_prob(&st.b, *eta, sd_prior)?;
            if with_grad {
                st.backprop_b(&gb, grad);
            }
            Ok(v)
        }
    }
}

/// The log density of GPyTorch's `LKJCovariancePrior(n, eta, sd_prior)` at
/// the covariance matrix `B`, `log p_LKJ(R) + Σᵢ log p_sd(σᵢ)` (correlations
/// `R`, standard deviations `σ`), and its gradient with respect to the
/// entries of `B`.
///
/// Deviation: with a univariate `sd_prior` (all priors of botorch_rs),
/// GPyTorch returns the `n` values `log p_LKJ(R) + log p_sd(σᵢ)` (the
/// correlation term broadcast over the standard deviations), so its MLL,
/// which adds their sum, counts the correlation term `n` times.
///
/// # Errors
/// [`BotorchError::InvalidArgument`] for negative variances or a
/// correlation matrix that is not positive definite.
pub fn lkj_covariance_log_prob(b: &Matrix, eta: f64, sd_prior: &Prior) -> Result<(f64, Matrix)> {
    let n = b.nrows();
    let mut sd = vec![0.0; n];
    for (s, v) in sd.iter_mut().enumerate() {
        let var = b[(s, s)];
        if var.is_nan() || var < 0.0 {
            return Err(BotorchError::InvalidArgument(
                "Variance(s) cannot be negative".into(),
            ));
        }
        *v = var.sqrt();
    }
    let r = Matrix::from_fn(n, n, |i, j| (1.0 / sd[i]) * b[(i, j)] * (1.0 / sd[j]));
    let chol = r.clone().cholesky().ok_or_else(|| {
        BotorchError::InvalidArgument("Input is not a valid correlation matrix".into())
    })?;
    let l = chol.l();
    // torch.distributions.LKJCholesky.log_prob.
    let dm1 = (n - 1) as f64;
    let mut unnorm = 0.0;
    let mut lbar = Matrix::zeros(n, n);
    for i in 1..n {
        let order = 2.0 * (eta - 1.0) + n as f64 - (i + 1) as f64;
        unnorm += order * l[(i, i)].ln();
        lbar[(i, i)] = order / l[(i, i)];
    }
    let alpha = eta + 0.5 * dm1;
    let denominator = statrs::function::gamma::ln_gamma(alpha) * dm1;
    let p = n - 1;
    let mut numerator = (p as f64) * (p as f64 - 1.0) / 4.0 * std::f64::consts::PI.ln();
    for i in 0..p {
        numerator += statrs::function::gamma::ln_gamma(alpha - 0.5 - i as f64 / 2.0);
    }
    let pi_constant = 0.5 * dm1 * std::f64::consts::PI.ln();
    let corr_lp = unnorm - (pi_constant + numerator - denominator);
    let mut sd_lp = 0.0;
    for &s in &sd {
        check_support(sd_prior, s)?;
        sd_lp += sd_prior.log_prob(s);
    }
    // Reverse mode: L -> R (Cholesky backward, symmetrized like torch).
    let lt_lbar = l.transpose() * &lbar;
    let mut phi = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            phi[(i, j)] = if i == j {
                0.5 * lt_lbar[(i, j)]
            } else {
                lt_lbar[(i, j)]
            };
        }
    }
    let linv = l
        .clone()
        .try_inverse()
        .ok_or_else(|| BotorchError::InvalidArgument("singular correlation matrix".into()))?;
    let s = linv.transpose() * phi * &linv;
    let rbar = (&s + s.transpose()) * 0.5;
    // R_ij = B_ij / (σ_i σ_j), σ_i = sqrt(B_ii).
    let mut gb = Matrix::zeros(n, n);
    let mut gsd = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            gb[(i, j)] += rbar[(i, j)] / (sd[i] * sd[j]);
            let rij = r[(i, j)];
            gsd[i] -= rbar[(i, j)] * rij / sd[i];
            gsd[j] -= rbar[(i, j)] * rij / sd[j];
        }
    }
    for i in 0..n {
        gsd[i] += sd_prior.grad_log_prob(sd[i]);
        gb[(i, i)] += gsd[i] / (2.0 * sd[i]);
    }
    Ok((corr_lp + sd_lp, gb))
}

// ----------------------------------------------------------------------
// InfiniteWidthBNNKernel
// ----------------------------------------------------------------------

struct BnnOut {
    k: f64,
    dw: f64,
    db: f64,
    dxy: f64,
    dxx: f64,
}

struct BnnLayer {
    k11: f64,
    k22: f64,
    s: f64,
    fr: f64,
    f: f64,
    theta: f64,
    tt: f64,
}

/// One entry of BoTorch's arc-cosine recursion for inputs with inner
/// product `xy` and squared norms `xx`, `yy`, with reverse-mode derivatives
/// (with respect to the weight and bias variances, `xy` and `xx`).
#[allow(clippy::too_many_arguments)]
fn bnn_pair(
    xy: f64,
    xx: f64,
    yy: f64,
    d: f64,
    w: f64,
    b: f64,
    depth: usize,
    eps: f64,
    grad: bool,
) -> BnnOut {
    let (lo, hi) = (-1.0 + eps, 1.0 - eps);
    let mut k12 = w * (xy / d) + b;
    let mut k11 = w * xx / d + b;
    let mut k22 = w * yy / d + b;
    let mut layers = Vec::with_capacity(depth);
    for l in 0..depth {
        if l > 0 {
            k11 = w * k11 / 2.0 + b;
            k22 = w * k22 / 2.0 + b;
        }
        let s = (k11 * k22).sqrt();
        let fr = k12 / s;
        let f = fr.max(lo).min(hi);
        let theta = f.acos();
        let tt = theta.sin() + (PI - theta) * f;
        layers.push(BnnLayer {
            k11,
            k22,
            s,
            fr,
            f,
            theta,
            tt,
        });
        k12 = w / (2.0 * PI) * s * tt + b;
    }
    let mut out = BnnOut {
        k: k12,
        dw: 0.0,
        db: 0.0,
        dxy: 0.0,
        dxx: 0.0,
    };
    if !grad {
        return out;
    }
    let (mut g12, mut g11, mut g22) = (1.0, 0.0, 0.0);
    let (mut gw, mut gb) = (0.0, 0.0);
    for l in (0..depth).rev() {
        let lay = &layers[l];
        gb += g12;
        gw += g12 * lay.s * lay.tt / (2.0 * PI);
        let c = w / (2.0 * PI);
        let mut gs = g12 * c * lay.tt;
        let gtt = g12 * c * lay.s;
        let mut gf = gtt * (PI - lay.theta);
        let gtheta = gtt * (lay.theta.cos() - lay.f);
        gf += gtheta * (-1.0 / (1.0 - lay.f * lay.f).sqrt());
        let gfr = if lay.fr > lo && lay.fr < hi { gf } else { 0.0 };
        let g12_in = gfr / lay.s;
        gs -= gfr * lay.fr / lay.s;
        g11 += gs * lay.k22 / (2.0 * lay.s);
        g22 += gs * lay.k11 / (2.0 * lay.s);
        g12 = g12_in;
        if l > 0 {
            let prev = &layers[l - 1];
            gw += (g11 * prev.k11 + g22 * prev.k22) / 2.0;
            gb += g11 + g22;
            g11 *= w / 2.0;
            g22 *= w / 2.0;
        }
    }
    gw += g12 * xy / d + g11 * xx / d + g22 * yy / d;
    gb += g12 + g11 + g22;
    out.dw = gw;
    out.db = gb;
    out.dxy = g12 * w / d;
    out.dxx = g11 * w / d;
    out
}

// ----------------------------------------------------------------------
// LinearTruncatedFidelityKernel
// ----------------------------------------------------------------------

/// The bias factor of BoTorch's `LinearTruncatedFidelityKernel` for inputs
/// `x`, `y`: its value, its derivatives with respect to the fidelities of
/// `x` (zero where the clamp to `[0, 1]` is active, including at the bounds
/// 0 and 1: scalar bounds have a zero subgradient at equality in torch
/// ≥ 2.14) and with respect to the power.
fn ltf_bias(x: &[f64], y: &[f64], fid: &[usize], p: f64) -> (f64, [f64; 2], f64) {
    let clamp = |v: f64| (v.clamp(0.0, 1.0), 0.0 < v && v < 1.0);
    let (a1, in1) = clamp(x[fid[0]]);
    let (b1, _) = clamp(y[fid[0]]);
    let c1 = (1.0 - a1) * (1.0 - b1);
    let q1 = 1.0 + a1 * b1;
    let p1 = q1.powf(p);
    let mut bias = c1 * p1;
    let mut d1 = -(1.0 - b1) * p1 + c1 * p * q1.powf(p - 1.0) * b1;
    let mut d2 = 0.0;
    let mut dp = c1 * p1 * q1.ln();
    if fid.len() > 1 {
        let (a2, in2) = clamp(x[fid[1]]);
        let (b2, _) = clamp(y[fid[1]]);
        let q = 1.0 + (a1 * b1 + a2 * b2);
        let k2 = q.powf(p);
        let c2 = (1.0 - a2) * (1.0 - b2);
        let q2 = 1.0 + a2 * b2;
        let p2 = q2.powf(p);
        bias += c2 * p2;
        bias += c2 * c1 * k2;
        let dk2 = p * q.powf(p - 1.0);
        d1 += c2 * (-(1.0 - b1)) * k2 + c2 * c1 * dk2 * b1;
        d2 = -(1.0 - b2) * p2
            + c2 * p * q2.powf(p - 1.0) * b2
            + (-(1.0 - b2)) * c1 * k2
            + c2 * c1 * dk2 * b2;
        dp += c2 * p2 * q2.ln() + c2 * c1 * k2 * q.ln();
        if !in2 {
            d2 = 0.0;
        }
    }
    if !in1 {
        d1 = 0.0;
    }
    (bias, [d1, d2], dp)
}

// ----------------------------------------------------------------------
// OrthogonalAdditiveKernel
// ----------------------------------------------------------------------

impl OakState {
    /// `k_i(x, y)` and the factor `g` (`∂k/∂u = -g u`) for scaled inputs.
    #[inline]
    fn base(&self, i: usize, x: f64, y: f64) -> (f64, f64, f64) {
        let l = self.ls[i];
        let u = x / l - y / l;
        let (v, g) = self.kind.value_and_factor(u * u);
        (v, g, u)
    }

    fn compute_normalizers(&mut self) {
        let q = self.z.len();
        self.norm = Vec::with_capacity(self.dim);
        self.clamped = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let mut total = 0.0;
            for a in 0..q {
                let mut row = 0.0;
                for b in 0..q {
                    let k = if a == b {
                        self.kind.value(0.0)
                    } else {
                        self.base(i, self.z[a], self.z[b]).0
                    };
                    row += k * self.w[b];
                }
                total += self.w[a] * row;
            }
            self.clamped.push(total <= OAK_NORMALIZER_EPS);
            self.norm.push(total.max(OAK_NORMALIZER_EPS));
        }
    }

    /// `S_i(x_a)` and `S_i'(x_a)` for all dimensions and points: a buffer of
    /// `2 · dim` values per point (`S` first).
    fn projections(&self, x: &[f64], n: usize) -> Vec<f64> {
        let d = self.dim;
        let mut out = vec![0.0; 2 * d * n];
        for i in 0..d {
            let wn: Vec<f64> = self.w.iter().map(|w| w / self.norm[i].sqrt()).collect();
            for a in 0..n {
                let xa = x[a * d + i];
                let (mut s, mut ds) = (0.0, 0.0);
                for (q, &zq) in self.z.iter().enumerate() {
                    let (v, g, u) = self.base(i, xa, zq);
                    s += v * wn[q];
                    ds += -g * u / self.ls[i] * wn[q];
                }
                out[a * 2 * d + i] = s;
                out[a * 2 * d + d + i] = ds;
            }
        }
        out
    }

    #[inline]
    fn s(&self, p: &NodePrep, i: usize, a: usize) -> f64 {
        p.aux[a * 2 * self.dim + i]
    }

    #[inline]
    fn ds(&self, p: &NodePrep, i: usize, a: usize) -> f64 {
        p.aux[a * 2 * self.dim + self.dim + i]
    }

    /// `K_i(x_a, y_b)` for every dimension.
    fn components(&self, a: &NodePrep, ia: usize, b: &NodePrep, jb: usize, same: bool) -> Vec<f64> {
        let d = self.dim;
        (0..d)
            .map(|i| {
                let k = if same {
                    self.kind.value(0.0)
                } else {
                    self.base(i, a.x[ia * d + i], b.x[jb * d + i]).0
                };
                k - self.s(a, i, ia) * self.s(b, i, jb)
            })
            .collect()
    }

    fn combine(&self, comps: &[f64]) -> f64 {
        let d = self.dim;
        let mut k1 = 0.0;
        for (c, k) in self.c1.iter().zip(comps) {
            k1 += c * k;
        }
        let mut v = k1 + self.offset;
        if let Some(c2) = &self.c2 {
            let mut k2 = 0.0;
            let mut idx = 0;
            for i in 0..d {
                for j in (i + 1)..d {
                    k2 += comps[i] * comps[j] * c2[idx];
                    idx += 1;
                }
            }
            v += k2;
        }
        v
    }

    /// `c1_i + Σ_{j≠i} c2_ij K_j` (the derivative of the combination with
    /// respect to `K_i`).
    fn weights(&self, comps: &[f64]) -> Vec<f64> {
        let d = self.dim;
        let mut out = self.c1.clone();
        if let Some(c2) = &self.c2 {
            let mut idx = 0;
            for i in 0..d {
                for j in (i + 1)..d {
                    out[i] += c2[idx] * comps[j];
                    out[j] += c2[idx] * comps[i];
                    idx += 1;
                }
            }
        }
        out
    }

    fn value(&self, a: &NodePrep, i: usize, b: &NodePrep, j: usize, same: bool) -> f64 {
        self.combine(&self.components(a, i, b, j, same))
    }

    fn cross(&self, a: &NodePrep, b: &NodePrep) -> Matrix {
        Matrix::from_fn(a.n, b.n, |i, j| self.value(a, i, b, j, false))
    }

    fn symmetric(&self, a: &NodePrep) -> Matrix {
        symmetric_from(a.n, |i, j| self.value(a, i, a, j, i == j))
    }

    fn diag_mode(&self, a: &NodePrep, ia: usize) -> f64 {
        let comps: Vec<f64> = (0..self.dim)
            .map(|i| self.kind.diag_value() - self.s(a, i, ia) * self.s(a, i, ia))
            .collect();
        self.combine(&comps)
    }

    fn cross_grad(&self, a: &NodePrep, ia: usize, b: &NodePrep) -> (Vector, Matrix) {
        let d = self.dim;
        let mut val = Vector::zeros(b.n);
        let mut grad = Matrix::zeros(b.n, d);
        for jb in 0..b.n {
            let comps = self.components(a, ia, b, jb, false);
            val[jb] = self.combine(&comps);
            let wts = self.weights(&comps);
            for i in 0..d {
                let (_, g, u) = self.base(i, a.x[ia * d + i], b.x[jb * d + i]);
                let dk = -g * u / self.ls[i] - self.ds(a, i, ia) * self.s(b, i, jb);
                grad[(jb, i)] = wts[i] * dk;
            }
        }
        (val, grad)
    }

    fn contract(&self, a: &NodePrep, w: &Matrix, off: usize, out: &mut [f64]) {
        let d = self.dim;
        let n = a.n;
        let q = self.z.len();
        // Components K_i(A, A).
        let mut comps = vec![Matrix::zeros(n, n); d];
        for ia in 0..n {
            for jb in ia..n {
                let c = self.components(a, ia, a, jb, ia == jb);
                for i in 0..d {
                    comps[i][(ia, jb)] = c[i];
                    comps[i][(jb, ia)] = c[i];
                }
            }
        }
        // offset, c1, c2.
        out[off] += w.sum();
        for i in 0..d {
            out[off + 1 + i] += w.component_mul(&comps[i]).sum();
        }
        if self.c2.is_some() {
            let mut idx = 0;
            for i in 0..d {
                for j in (i + 1)..d {
                    out[off + 1 + d + idx] +=
                        w.component_mul(&comps[i]).component_mul(&comps[j]).sum();
                    idx += 1;
                }
            }
        }
        // Base lengthscales: adjoint A_i = W ∘ (c1_i + Σ_j c2_ij K_j).
        let mut gl = vec![0.0; d];
        for (i, gli) in gl.iter_mut().enumerate() {
            let mut ai = w * self.c1[i];
            if let Some(c2) = &self.c2 {
                let mut idx = 0;
                for r in 0..d {
                    for c in (r + 1)..d {
                        if r == i {
                            ai += w.component_mul(&comps[c]) * c2[idx];
                        } else if c == i {
                            ai += w.component_mul(&comps[r]) * c2[idx];
                        }
                        idx += 1;
                    }
                }
            }
            let l = self.ls[i];
            // Σ_ab A ∂k(x_a, x_b)/∂ℓ.
            let mut g = 0.0;
            for ia in 0..n {
                for jb in (ia + 1)..n {
                    let (_, gg, u) = self.base(i, a.x[ia * d + i], a.x[jb * d + i]);
                    g += (ai[(ia, jb)] + ai[(jb, ia)]) * gg * u * u / l;
                }
            }
            // ∂S(x_a)/∂ℓ.
            let sq = self.norm[i].sqrt();
            let dnorm = if self.clamped[i] {
                0.0
            } else {
                let mut t = 0.0;
                for qa in 0..q {
                    for qb in 0..q {
                        if qa != qb {
                            let (_, gg, u) = self.base(i, self.z[qa], self.z[qb]);
                            t += self.w[qa] * self.w[qb] * gg * u * u / l;
                        }
                    }
                }
                t
            };
            let a_s: Vec<f64> = (0..n)
                .map(|ia| (0..n).map(|jb| ai[(ia, jb)] * self.s(a, i, jb)).sum())
                .collect();
            for (ia, a_s_ia) in a_s.iter().enumerate() {
                let xa = a.x[ia * d + i];
                let mut t = 0.0;
                for (qq, &zq) in self.z.iter().enumerate() {
                    let (_, gg, u) = self.base(i, xa, zq);
                    t += gg * u * u / l * self.w[qq];
                }
                let s = self.s(a, i, ia);
                let ds = t / sq - s / (2.0 * self.norm[i]) * dnorm;
                g -= 2.0 * ds * a_s_ia;
            }
            *gli = g;
        }
        if self.shared {
            out[self.ls_off] += gl.iter().sum::<f64>();
        } else {
            for i in 0..d {
                out[self.ls_off + i] += gl[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{Constraint, GreaterThan, Interval};
    use crate::ParameterTransform;

    fn points(n: usize, d: usize, seed: u64) -> Matrix {
        let mut s = seed;
        Matrix::from_fn(n, d, |_, _| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        })
    }

    fn rand_raw(node: &KernelNode, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..node.num_raw())
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                -0.6 + 1.2 * ((s >> 11) as f64) / ((1u64 << 53) as f64)
            })
            .collect()
    }

    fn trees(d: usize) -> Vec<KernelNode> {
        let st = |k: KernelKind, ard: Option<usize>| KernelNode::stationary(k, ard).unwrap();
        let ad = |n: KernelNode, dims: Vec<usize>| n.with_active_dims(dims).unwrap();
        let mut v = vec![
            st(KernelKind::Rbf, Some(d)),
            st(KernelKind::Matern12, None),
            st(KernelKind::Matern32, Some(d)),
            KernelNode::scale(st(KernelKind::Matern52, Some(d))),
            KernelNode::rational_quadratic(Some(d)).unwrap(),
            KernelNode::periodic(Some(d)).unwrap(),
            KernelNode::periodic(None).unwrap(),
            KernelNode::linear(Some(d)).unwrap(),
            KernelNode::polynomial(2.0).unwrap(),
            KernelNode::polynomial(3.0).unwrap(),
            KernelNode::cosine(),
            KernelNode::categorical(Some(d)).unwrap(),
            KernelNode::exponential_decay(None).unwrap(),
            KernelNode::infinite_width_bnn(3, 1e-7).unwrap(),
            KernelNode::scale(
                KernelNode::additive(vec![
                    ad(st(KernelKind::Rbf, Some(1)), vec![0]),
                    KernelNode::product(vec![
                        ad(st(KernelKind::Matern52, Some(2)), vec![1, 2]),
                        ad(KernelNode::linear(None).unwrap(), vec![0, 2]),
                    ])
                    .unwrap(),
                ])
                .unwrap(),
            ),
            KernelNode::linear_truncated_fidelity(
                vec![2],
                st(KernelKind::Matern52, Some(2)),
                st(KernelKind::Matern52, Some(2)),
            )
            .unwrap(),
            KernelNode::linear_truncated_fidelity(
                vec![1, 2],
                st(KernelKind::Matern32, Some(1)),
                st(KernelKind::Rbf, Some(1)),
            )
            .unwrap(),
            ad(KernelNode::downsampling(), vec![2]),
            KernelNode::orthogonal_additive(
                d,
                st(KernelKind::Rbf, None),
                true,
                8,
                true,
                true,
                Constraint::positive(),
            )
            .unwrap(),
            KernelNode::orthogonal_additive(
                d,
                st(KernelKind::Matern52, None),
                false,
                6,
                false,
                true,
                Constraint::positive(),
            )
            .unwrap(),
        ];
        let idx = KernelNode::product(vec![
            ad(st(KernelKind::Rbf, Some(2)), vec![0, 1]),
            ad(KernelNode::index(3, 2).unwrap(), vec![2]),
        ])
        .unwrap();
        v.push(idx);
        let pidx = KernelNode::product(vec![
            ad(st(KernelKind::Rbf, Some(2)), vec![0, 1]),
            ad(KernelNode::positive_index(3, 2, 1, true).unwrap(), vec![2]),
        ])
        .unwrap();
        v.push(pidx);
        v
    }

    /// Inputs with the last column holding task indices 0..3 for the index
    /// kernels (fractional values truncate).
    fn inputs(n: usize, d: usize, seed: u64, tasks: bool) -> Matrix {
        let mut x = points(n, d, seed);
        if tasks {
            for i in 0..n {
                x[(i, d - 1)] = (i % 3) as f64 + 0.3;
            }
        }
        x
    }

    fn uses_tasks(t: &KernelNode) -> bool {
        format!("{t:?}").contains("Index")
    }

    #[test]
    fn hyperparameter_gradients_match_finite_differences() {
        let d = 3;
        for (case, tree) in trees(d).into_iter().enumerate() {
            let raw = rand_raw(&tree, 11 + case as u64);
            let x = inputs(6, d, 3 + case as u64, uses_tasks(&tree));
            let w = {
                let m = points(6, 6, 99);
                &m + m.transpose()
            };
            let kern = TreeKernel::build(&tree, &raw, d).unwrap();
            let prep = kern.prepare(&x).unwrap();
            let g = kern.contract(&prep, &w);
            let vals = kern.values().to_vec();
            // Finite differences in the actual values (via raw values).
            let derivs = tree.transform_derivatives(&raw);
            for p in 0..raw.len() {
                let h = 1e-6;
                let mut rp = raw.clone();
                rp[p] += h;
                let mut rm = raw.clone();
                rm[p] -= h;
                let f = |r: &[f64]| {
                    let k = TreeKernel::build(&tree, r, d).unwrap();
                    let pr = k.prepare(&x).unwrap();
                    w.component_mul(&k.symmetric(&pr)).sum()
                };
                let fd = (f(&rp) - f(&rm)) / (2.0 * h) / derivs[p];
                assert!(
                    (fd - g[p]).abs() < 2e-6 * (1.0 + fd.abs()),
                    "case {case} param {p}: fd {fd} analytic {} (value {})",
                    g[p],
                    vals[p]
                );
            }
        }
    }

    #[test]
    fn input_gradients_match_finite_differences() {
        let d = 3;
        for (case, tree) in trees(d).into_iter().enumerate() {
            let raw = rand_raw(&tree, 21 + case as u64);
            let tasks = uses_tasks(&tree);
            let kern = TreeKernel::build(&tree, &raw, d).unwrap();
            let xb = inputs(5, d, 7 + case as u64, tasks);
            let pb = kern.prepare(&xb).unwrap();
            let x = inputs(1, d, 50 + case as u64, tasks);
            let pa = kern.prepare(&x).unwrap();
            let (k, dk) = kern.cross_with_grad(&pa, 0, &pb);
            let cross = kern.cross(&pa, &pb);
            assert!(
                (cross.row(0).transpose() - &k).amax() < 1e-14,
                "case {case}"
            );
            let (dv, ddv) = kern.diag_with_grad(&pa, 0);
            for m in 0..d {
                if tasks && m == d - 1 {
                    continue;
                }
                let h = 1e-6;
                let mut xp = x.clone();
                xp[(0, m)] += h;
                let mut xm = x.clone();
                xm[(0, m)] -= h;
                let kp = kern.cross(&kern.prepare(&xp).unwrap(), &pb);
                let km = kern.cross(&kern.prepare(&xm).unwrap(), &pb);
                let fd = (kp - km) / (2.0 * h);
                for j in 0..5 {
                    assert!(
                        (fd[(0, j)] - dk[(j, m)]).abs() < 1e-6 * (1.0 + fd[(0, j)].abs()),
                        "case {case} dim {m} point {j}: {} vs {}",
                        fd[(0, j)],
                        dk[(j, m)]
                    );
                }
                let dp = kern.diag(&kern.prepare(&xp).unwrap(), 0);
                let dm = kern.diag(&kern.prepare(&xm).unwrap(), 0);
                let fd = (dp - dm) / (2.0 * h);
                assert!(
                    (fd - ddv[m]).abs() < 1e-6 * (1.0 + fd.abs()),
                    "case {case} diag dim {m}: {fd} vs {}",
                    ddv[m]
                );
            }
            // The diagonal is that of the full matrix; GPyTorch's diag=True
            // mode agrees up to the documented clamping differences.
            assert_eq!(kern.symmetric(&pa)[(0, 0)], dv, "case {case}");
            let mode = kern.diag_mode(&pa, 0);
            assert!((mode - dv).abs() < 1e-6 * (1.0 + dv.abs()), "case {case}");
            if kern.has_constant_diag() {
                assert!(ddv.iter().all(|v| *v == 0.0), "case {case}");
            }
        }
    }

    #[test]
    fn symmetric_matches_cross_and_is_symmetric() {
        let d = 3;
        for (case, tree) in trees(d).into_iter().enumerate() {
            let raw = rand_raw(&tree, 31 + case as u64);
            let kern = TreeKernel::build(&tree, &raw, d).unwrap();
            let x = inputs(5, d, 9 + case as u64, uses_tasks(&tree));
            let p = kern.prepare(&x).unwrap();
            let s = kern.symmetric(&p);
            let c = kern.cross(&p, &p);
            assert!((&s - s.transpose()).amax() == 0.0, "case {case}");
            for i in 0..5 {
                for j in 0..5 {
                    if i != j {
                        assert!((s[(i, j)] - c[(i, j)]).abs() < 1e-13, "case {case}");
                    }
                }
            }
        }
    }

    #[test]
    fn prior_gradients_match_finite_differences() {
        let lkj = MatrixPrior::LkjCovariance {
            n: 3,
            eta: 1.5,
            sd_prior: Prior::gamma(2.0, 1.5).unwrap(),
        };
        let idx = KernelNode::index(3, 2)
            .unwrap()
            .with_task_prior(lkj)
            .unwrap();
        let pidx = KernelNode::positive_index(3, 2, 0, true)
            .unwrap()
            .with_task_prior(MatrixPrior::Elementwise(Prior::beta(2.5, 1.5).unwrap()))
            .unwrap();
        let st = KernelNode::stationary(KernelKind::Rbf, Some(1))
            .unwrap()
            .with_prior("raw_lengthscale", Prior::half_cauchy(0.5).unwrap())
            .unwrap();
        let oak = KernelNode::orthogonal_additive(
            2,
            KernelNode::stationary(KernelKind::Rbf, None).unwrap(),
            true,
            5,
            true,
            true,
            Constraint::positive(),
        )
        .unwrap()
        .with_prior("raw_coeffs_2", Prior::gamma(2.0, 3.0).unwrap())
        .unwrap();
        for (tree, dim) in [(idx, 1), (pidx, 1), (st, 1), (oak, 2)] {
            let raw = rand_raw(&tree, 5);
            let k = TreeKernel::build(&tree, &raw, dim).unwrap();
            let (_, g) = k.log_prior_and_grad().unwrap();
            let derivs = tree.transform_derivatives(&raw);
            for p in 0..raw.len() {
                let h = 1e-6;
                let mut rp = raw.clone();
                rp[p] += h;
                let mut rm = raw.clone();
                rm[p] -= h;
                let f = |r: &[f64]| {
                    TreeKernel::build(&tree, r, dim)
                        .unwrap()
                        .log_prior()
                        .unwrap()
                };
                let fd = (f(&rp) - f(&rm)) / (2.0 * h) / derivs[p];
                assert!(
                    (fd - g[p]).abs() < 1e-6 * (1.0 + fd.abs()),
                    "{p}: {fd} vs {}",
                    g[p]
                );
            }
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let idx = KernelNode::index(3, 1).unwrap();
        let k = TreeKernel::build(&idx, &[0.1, 0.2, 0.3, 0.0, 0.0, 0.0], 1).unwrap();
        assert!(k.prepare(&Matrix::from_element(1, 1, 3.0)).is_err());
        assert!(k.prepare(&Matrix::from_element(1, 1, -1.0)).is_err());
        assert!(k.prepare(&Matrix::from_element(1, 1, 2.9)).is_ok());
        let oak = KernelNode::orthogonal_additive(
            1,
            KernelNode::stationary(KernelKind::Rbf, None).unwrap(),
            true,
            4,
            false,
            true,
            Constraint::positive(),
        )
        .unwrap();
        let k = TreeKernel::build(&oak, &oak.initial_raw(), 1).unwrap();
        assert!(k.prepare(&Matrix::from_element(1, 1, 1.1)).is_err());
        // Like GPyTorch, non-enforced constraints may give negative
        // lengthscales, which enter the kernel squared; non-finite raw values
        // are rejected.
        let rbf = KernelNode::stationary(KernelKind::Rbf, None)
            .unwrap()
            .with_constraint("raw_lengthscale", GreaterThan::new(0.0, None, None).into())
            .unwrap();
        let x = points(3, 2, 1);
        let neg = TreeKernel::build(&rbf, &[-0.1], 2).unwrap();
        let pos = TreeKernel::build(&rbf, &[0.1], 2).unwrap();
        assert_eq!(
            neg.covariance_symmetric(&x).unwrap(),
            pos.covariance_symmetric(&x).unwrap()
        );
        assert!(TreeKernel::build(&rbf, &[f64::NAN], 2).is_err());
        let iv: Constraint = Interval::new(0.1, 2.0, Some(ParameterTransform::Sigmoid), None, None)
            .unwrap()
            .into();
        let rbf = rbf.with_constraint("raw_lengthscale", iv).unwrap();
        assert!(TreeKernel::build(&rbf, &[-0.1], 2).is_ok());
    }

    #[test]
    fn training_nan_follows_the_kernel_nodes() {
        // ScaleKernel(AdditiveKernel(RBF on column 0, Matérn-5/2 on columns
        // 1 and 2)): raw outputscale, RBF lengthscale, Matérn lengthscales.
        let st = |k: KernelKind, ard: Option<usize>| KernelNode::stationary(k, ard).unwrap();
        let tree = KernelNode::scale(
            KernelNode::additive(vec![
                st(KernelKind::Rbf, Some(1))
                    .with_active_dims(vec![0])
                    .unwrap(),
                st(KernelKind::Matern52, Some(2))
                    .with_active_dims(vec![1, 2])
                    .unwrap(),
            ])
            .unwrap(),
        );
        let x = points(6, 3, 7);
        let k = TreeKernel::build(&tree, &[0.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert!(!k.training_nan(&x, false).any());
        // A Matérn lengthscale that underflowed to zero: NaN (GPyTorch's
        // x / 0 before the centering of `sq_dist`).
        let k = TreeKernel::build(&tree, &[0.0, 0.0, -800.0, 0.0], 3).unwrap();
        let nan = k.training_nan(&x, false);
        assert!(nan.any());
        let mut m = k.covariance_symmetric(&x).unwrap();
        nan.inject(&mut m);
        assert!(m.iter().all(|v| v.is_nan()));
        // A single RBF lengthscale of 1e-160 on two points either side of
        // their mean: `RBFCovariance` zeroes the diagonal, the off-diagonal
        // squared distance overflows to +inf, K is finite and its
        // lengthscale derivative `inf · 0` is NaN.
        let rbf = st(KernelKind::Rbf, None);
        let k = TreeKernel::build(&rbf, &[1e-160f64.ln()], 1).unwrap();
        let nan = k.training_nan(&Matrix::from_row_slice(2, 1, &[-0.5, 0.5]), false);
        assert!(!nan.any());
        assert_eq!(nan.nan_gradients(), &[0]);
        // Inputs from a learned transform require gradients: GPyTorch's
        // autograd path computes the diagonal too, `-2c² + c² + c²` with
        // `c² = inf`, which is NaN.
        let two = Matrix::from_row_slice(2, 1, &[-0.5, 0.5]);
        assert!(k.training_nan(&two, true).any());
    }
}
