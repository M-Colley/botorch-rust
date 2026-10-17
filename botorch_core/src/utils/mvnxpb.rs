//! Bivariate normal probabilities and the bivariate conditioning algorithm
//! MVNXPB for multivariate normal probabilities (BoTorch's
//! `botorch.utils.probability.{bvn, mvnxpb}`).
//!
//! [`bvn`] computes `P(xl < x < xu, yl < y < yu)` of a standard bivariate
//! normal with correlation `r` by Genz's method (A. Genz, *Numerical
//! computation of rectangular bivariate and trivariate normal and t
//! probabilities*, Statistics and Computing 2004): Gauss–Legendre quadrature
//! in polar coordinates for `|r| < 0.925` (6, 12 or 20 points), a Taylor
//! expansion for `|r| >= 0.925`. [`bvnmom`] gives the means of the truncated
//! bivariate normal (Muthén 1990). [`mvnxpb_log_prob`] approximates `log P(X
//! ∈ bounds)` for `X ~ N(0, Σ)` by bivariate conditioning with the GGE
//! variable ordering (Trinh & Genz, *Bivariate conditioning approximations for
//! multivariate normal probabilities*, Statistics and Computing 2015; Genz &
//! Trinh 2016; Gibson et al. 1994), step by step as BoTorch's `MVNXPB.solve`.
//!
//! Every function is generic over the crate's [`Scalar`] type, so that dual
//! numbers give exact (forward-mode) derivatives with respect to the
//! covariances and bounds. The pivots depend on the values only.
//!
//! One difference to BoTorch: the number of quadrature points of the polar
//! integration is chosen from the correlation of the pair at hand; BoTorch
//! chooses it from the largest correlation of all pairs evaluated together
//! (e.g. the t-batch of candidate sets), so its value for one pair can depend
//! on the others (at the level of the quadrature error, 5e-16).

// The step-by-step translation of BoTorch's tensor code indexes rows and
// columns of the factor explicitly.
#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::acquisition::dual::Scalar;
use crate::error::{BotorchError, Result};
use crate::Matrix;

/// BoTorch's `STANDARDIZED_RANGE`: standardized bounds are clipped to it.
pub const STANDARDIZED_RANGE: (f64, f64) = (-1e6, 1e6);

const INV_2PI: f64 = 1.0 / (2.0 * PI);
const SQRT_2PI: f64 = 2.506_628_274_631_000_7;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// The Gauss–Legendre nodes and weights of degree `n` on `[-1, 1]`
/// (ascending nodes, numpy's `leggauss`), by Newton's method on the Legendre
/// polynomial.
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        // Initial guess (Tricomi), then Newton.
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            // P_n(z) = p1, P_{n-1}(z) = p0.
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() <= 1e-16 {
                break;
            }
        }
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

/// The cached Gauss–Legendre rules of degree 6, 12 and 20.
fn rule(n: usize) -> &'static (Vec<f64>, Vec<f64>) {
    static R6: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    static R12: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    static R20: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    match n {
        6 => R6.get_or_init(|| gauss_legendre(6)),
        12 => R12.get_or_init(|| gauss_legendre(12)),
        _ => R20.get_or_init(|| gauss_legendre(20)),
    }
}

fn phi<S: Scalar>(x: S) -> S {
    (x * x * -0.5).exp() * INV_SQRT_2PI
}

/// BoTorch's `safe_mul`: `0 · ∞ = 0`.
fn safe_mul<S: Scalar>(a: S, b: S) -> S {
    if (a.value().is_infinite() && b.value() == 0.0)
        || (b.value().is_infinite() && a.value() == 0.0)
    {
        a.lift(0.0)
    } else {
        a * b
    }
}

/// BoTorch's `safe_sub`: `∞ - ∞ = 0` (for infinities of the same sign).
fn safe_sub<S: Scalar>(a: S, b: S) -> S {
    if a.value().is_infinite() && b.value().is_infinite() && a.value() == b.value() {
        a.lift(0.0)
    } else {
        a - b
    }
}

/// BoTorch's `safe_div`: `0 / 0` and `∞ / ∞` are `±1`.
fn safe_div<S: Scalar>(a: S, b: S) -> S {
    let (av, bv) = (a.value(), b.value());
    if (av == 0.0 && bv == 0.0) || (av.is_infinite() && bv.is_infinite()) {
        a.lift(if av != bv { -1.0 } else { 1.0 })
    } else {
        a / b
    }
}

/// `log(max(f64::MAX))`, BoTorch's clip of `safe_exp`'s argument.
const MAX_EXP_ARG: f64 = 709.782_712_893_384 - 1e-4;

/// BoTorch's `safe_exp`: `exp(min(x, log(max) - 1e-4))` (zero gradient
/// where clipped). The bound is a tensor (`get_constants_like`), so torch
/// ≥ 2.14 passes half of the gradient to `x` at a tie.
fn safe_exp<S: Scalar>(x: S) -> S {
    if x.value() > MAX_EXP_ARG {
        x.lift(MAX_EXP_ARG).exp()
    } else if x.value() == MAX_EXP_ARG {
        x.tie(x.lift(MAX_EXP_ARG)).exp()
    } else {
        x.exp()
    }
}

/// BoTorch's `safe_log`: `log(max(x, tiny))` (zero gradient where clipped,
/// including at `x == tiny`: a scalar bound, torch ≥ 2.14).
fn safe_log<S: Scalar>(x: S) -> S {
    if x.value() <= f64::MIN_POSITIVE {
        x.lift(x.value().max(f64::MIN_POSITIVE)).ln()
    } else {
        x.ln()
    }
}

/// `x.clip(min=0, max=1)` (scalar bounds: zero gradient where clipped and at
/// the bounds, torch ≥ 2.14).
fn clip01<S: Scalar>(x: S) -> S {
    if x.value() < 0.0 {
        x.lift(0.0)
    } else if x.value() > 1.0 {
        x.lift(1.0)
    } else if x.value() == 0.0 || x.value() == 1.0 {
        x.lift(x.value())
    } else {
        x
    }
}

/// BoTorch's `_bvnu_polar`: `P(x > h, y > k)` for `|r| < 0.925` (the
/// operations in BoTorch's order).
fn bvnu_polar<S: Scalar>(r: S, h: S, k: S) -> S {
    let mar = r.value().abs();
    let n = if mar < 0.3 {
        6
    } else if mar < 0.75 {
        12
    } else {
        20
    };
    let (x, w) = rule(n);
    let asin_r = r.asin() * 0.5;
    let hk = h * k;
    let hh = (h * h + k * k) * 0.5;
    let mut acc = r.lift(0.0);
    for (xi, wi) in x.iter().zip(w) {
        let s = (asin_r * (xi + 1.0)).sin();
        let v = safe_exp(safe_sub(safe_mul(s, hk), hh) / (-(s * s) + 1.0));
        acc = acc + v * *wi;
    }
    clip01((-h).ndtr() * (-k).ndtr() + asin_r * INV_2PI * acc)
}

/// BoTorch's `_bvnu_taylor`: `P(x > h, y > k)` for `|r| >= 0.925` (the
/// operations in BoTorch's order).
fn bvnu_taylor<S: Scalar>(r: S, h: S, k: S) -> S {
    let (x, w) = rule(20);
    let s = if r.value() > 0.0 { 1.0 } else { -1.0 };
    let sk = k * s;
    let skh = sk * h;
    let comp_r2 = -(r * r) + 1.0;
    let a = comp_r2.clamp_min(0.0).sqrt();
    let b = safe_sub(h, sk);
    let b2 = b * b;
    let c = (-skh + 4.0) / 8.0;
    let d = (-skh + 12.0) / 80.0;
    let int_from_0_to_s = if r.value() > 0.0 {
        // torch.maximum(h, k): an even split at a tie.
        let m = if h.value() > k.value() {
            h
        } else if h.value() == k.value() {
            h.tie(k)
        } else {
            k
        };
        (-m).ndtr()
    } else {
        (sk.ndtr() - h.ndtr()).clamp_min(0.0)
    };
    let a0 = (safe_div(b2, comp_r2) + skh) * -0.5;
    let a1 = c * (-(d * b2) + 1.0) / 3.0;
    let a2 = -(b2 * a1) + 1.0;
    let abs_b = b.abs();
    let analytic = a * (a2 + comp_r2 * a1 + c * d * (comp_r2 * comp_r2)) * safe_exp(a0)
        - safe_div(-abs_b, a).ndtr() * SQRT_2PI * abs_b * a2 * safe_exp(skh * -0.5);
    let mut quad = r.lift(0.0);
    for (xi, wi) in x.iter().zip(w) {
        let xx = xi + 1.0;
        let q0 = comp_r2 * 0.25 * (xx * xx);
        let q1 = (-q0 + 1.0).sqrt();
        let q2 = (b2 / q0 + skh) * -0.5;
        if q2.value() <= -100.0 {
            continue;
        }
        let t1 = c * q0 * (d * q0 * 5.0 + 1.0) + 1.0;
        let t2 = safe_exp(q0 * -0.5 / ((q1 + 1.0) * (q1 + 1.0)) * skh) / q1;
        quad = quad + safe_exp(q2) * (t1 - t2) * *wi;
    }
    let quadrature = a * -0.5 * quad;
    let int_from_0_to_a = (analytic + quadrature) * (INV_2PI * s);
    clip01(int_from_0_to_s - int_from_0_to_a)
}

/// BoTorch's `bvnu`: `P(x > h, y > k)` of a standard bivariate normal with
/// correlation `r`.
pub(crate) fn bvnu<S: Scalar>(r: S, h: S, k: S) -> S {
    let (lower, upper) = STANDARDIZED_RANGE;
    let x_free = h.value() < lower;
    let y_free = k.value() < lower;
    if h.value() > upper || k.value() > upper {
        r.lift(0.0)
    } else if x_free && y_free {
        r.lift(1.0)
    } else if x_free {
        (-k).ndtr()
    } else if y_free {
        (-h).ndtr()
    } else if r.value() == 0.0 {
        (-h).ndtr() * (-k).ndtr()
    } else if r.value().abs() >= 0.925 {
        bvnu_taylor(r, h, k)
    } else {
        bvnu_polar(r, h, k)
    }
}

/// BoTorch's `bvn`: `P(xl < x < xu, yl < y < yu)` of a standard bivariate
/// normal with correlation `r`.
pub(crate) fn bvn_generic<S: Scalar>(r: S, xl: S, yl: S, xu: S, yu: S) -> S {
    let flip_x = xl.value().abs() > xu.value();
    let flip_y = yl.value().abs() > yu.value();
    let flip = (flip_x && (!flip_y || yu.value().is_infinite()))
        || (flip_y && (!flip_x || xu.value().is_infinite()));
    let (xl, xu, yl, yu) = if flip {
        (-xl, -xu, -yl, -yu)
    } else {
        (xl, xu, yl, yu)
    };
    let terms = [
        bvnu(r, xl, yl),
        -bvnu(r, xl, yu),
        -bvnu(r, xu, yl),
        bvnu(r, xu, yu),
    ];
    let mut res = r.lift(0.0);
    for t in terms {
        if t.value() != 0.0 {
            res = res + t;
        }
    }
    clip01(res)
}

/// `P(xl < x < xu, yl < y < yu)` of a standard bivariate normal with
/// correlation `r` (BoTorch's `bvn`).
pub fn bvn(r: f64, xl: f64, yl: f64, xu: f64, yu: f64) -> f64 {
    bvn_generic(r, xl, yl, xu, yu)
}

/// BoTorch's `bvnmom`: `E([x, y] | [xl, yl] < [x, y] < [xu, yu])` of a
/// standard bivariate normal with correlation `r`, given the probability
/// `p` of the rectangle.
pub(crate) fn bvnmom_generic<S: Scalar>(r: S, xl: S, yl: S, xu: S, yu: S, p: S) -> (S, S) {
    let istd = (-(r * r) + 1.0).sqrt();
    let istd = istd.lift(1.0) / istd;
    // Rows: x (bounds [xl, xu], other [yl, yu]) and y.
    let rows = [([xl, xu], (yl, yu)), ([yl, yu], (xl, xu))];
    let mut pd = [r.lift(0.0), r.lift(0.0)];
    for (v, (b, (ol, ou))) in rows.iter().enumerate() {
        let mut terms = [r.lift(0.0), r.lift(0.0)];
        for (j, bj) in b.iter().enumerate() {
            let delta = safe_mul(r, *bj);
            let up = safe_mul(istd, safe_sub(*ou, delta));
            let lo = safe_mul(istd, safe_sub(*ol, delta));
            let prob = up.ndtr() - lo.ndtr();
            terms[j] = phi(*bj) * prob;
        }
        pd[v] = terms[0] - terms[1];
    }
    let inv_p = p.lift(1.0) / p;
    ((pd[0] + r * pd[1]) * inv_p, (pd[1] + r * pd[0]) * inv_p)
}

/// The means of the truncated standard bivariate normal (BoTorch's
/// `bvnmom`).
pub fn bvnmom(r: f64, xl: f64, yl: f64, xu: f64, yu: f64) -> (f64, f64) {
    let p = bvn(r, xl, yl, xu, yu);
    bvnmom_generic(r, xl, yl, xu, yu, p)
}

/// The state of BoTorch's `PivotedCholesky` (lower-triangular storage of
/// the factor's first `step` columns and of the remaining Schur complement).
struct PivotedCholesky<S> {
    step: usize,
    tril: Vec<Vec<S>>,
}

impl<S: Scalar> PivotedCholesky<S> {
    /// BoTorch's `update_`: one step of the decomposition.
    fn update(&mut self, eps: f64) {
        let i = self.step;
        let n = self.tril.len();
        let lii = self.tril[i][i].clamp_min(0.0).sqrt();
        self.tril[i][i] = lii;
        for c in i + 1..n {
            self.tril[i][c] = lii.lift(0.0);
        }
        for r in i + 1..n {
            self.tril[r][i] = self.tril[r][i] / lii;
        }
        for r in i + 1..n {
            for c in i + 1..=r {
                let v = self.tril[r][i] * self.tril[c][i];
                self.tril[r][c] = self.tril[r][c] - v;
            }
        }
        if lii.value() <= i as f64 * eps {
            for r in i..n {
                self.tril[r][i] = lii.lift(0.0);
            }
        }
        self.step += 1;
    }

    /// BoTorch's `pivot_`: swaps the variables `step` and `piv > step`
    /// (rows of the factor's columns, and symmetrically in the Schur
    /// complement).
    fn pivot(&mut self, piv: usize) {
        let i = self.step;
        let t = &mut self.tril;
        // Rows `i` and `piv` of the factor's columns `0..i`.
        let (head, tail) = t.split_at_mut(piv);
        head[i][..i].swap_with_slice(&mut tail[0][..i]);
        // The Schur complement, symmetrically (lower-triangular storage).
        let a = t[i][i];
        t[i][i] = t[piv][piv];
        t[piv][piv] = a;
        for r in i + 1..piv {
            let a = t[r][i];
            t[r][i] = t[piv][r];
            t[piv][r] = a;
        }
        for row in t.iter_mut().skip(piv + 1) {
            row.swap(i, piv);
        }
    }
}

/// BoTorch's `MVNXPB(covariance_matrix, bounds).solve()`: an approximation
/// of `log P(X ∈ bounds)` for `X ~ N(0, Σ)`, with `bounds` one `[lower,
/// upper]` pair per variable (clipped to `STANDARDIZED_RANGE` standard
/// deviations).
pub(crate) fn mvnxpb_generic<S: Scalar>(cov: &[Vec<S>], bounds: &[[S; 2]]) -> S {
    let n = cov.len();
    let eps = 1e-10;
    let zero = cov[0][0].lift(0.0);
    let std: Vec<S> = (0..n).map(|i| cov[i][i].sqrt()).collect();
    let istd: Vec<S> = std.iter().map(|s| s.lift(1.0) / *s).collect();
    let mut tril = vec![vec![zero; n]; n];
    for r in 0..n {
        for c in 0..=r {
            tril[r][c] = istd[r] * cov[r][c] * istd[c];
        }
    }
    let (lo, hi) = STANDARDIZED_RANGE;
    let mut b: Vec<[S; 2]> = (0..n)
        .map(|i| {
            // `bounds.clip(std * lo, std * hi)`: tensor bounds, so torch
            // ≥ 2.14 splits the gradient evenly at a tie.
            let clip = |v: S| {
                let l = std[i] * lo;
                let u = std[i] * hi;
                if v.value() < l.value() {
                    l
                } else if v.value() > u.value() {
                    u
                } else if v.value() == l.value() {
                    v.tie(l)
                } else if v.value() == u.value() {
                    v.tie(u)
                } else {
                    v
                }
            };
            [istd[i] * clip(bounds[i][0]), istd[i] * clip(bounds[i][1])]
        })
        .collect();
    let mut pc = PivotedCholesky { step: 0, tril };
    let mut y = vec![zero.lift(f64::NAN); n];
    let mut log_prob = zero;
    let mut phi_i = zero;
    for i in 0..n {
        // GGE pivot: the least likely variable given the plug-in values.
        let mut best = i;
        let mut best_p = f64::INFINITY;
        for j in i..n {
            let mut shift = 0.0;
            for (c, yc) in y.iter().enumerate().take(i) {
                shift += pc.tril[j][c].value() * yc.value();
            }
            let inv = 1.0 / pc.tril[j][j].value().max(0.0).sqrt();
            let p = crate::utils::probability::ndtr(inv * (b[j][1].value() - shift))
                - crate::utils::probability::ndtr(inv * (b[j][0].value() - shift));
            if !best_p.is_nan() && (p.is_nan() || p < best_p) {
                best = j;
                best_p = p;
            }
        }
        if best > i {
            pc.pivot(best);
            b.swap(i, best);
        }
        let lii = pc.tril[i][i].clamp_min(0.0).sqrt();
        let inv_lii = lii.lift(1.0) / lii;
        let mut bi = b[i];
        if i != 0 {
            let mut s = zero;
            for c in 0..i {
                s = s + pc.tril[i][c] * y[c];
            }
            bi = [bi[0] - s, bi[1] - s];
        }
        let lb = inv_lii * bi[0];
        let ub = inv_lii * bi[1];
        phi_i = ub.ndtr() - lb.ndtr();
        let small = phi_i.value() <= i as f64 * eps;
        y[i] = if small && lb.value() < -9.0 {
            ub
        } else if small && lb.value() > 9.0 {
            lb
        } else if small {
            (lb + ub) * 0.5
        } else {
            (phi(lb) - phi(ub)) / phi_i
        };
        if i % 2 == 1 {
            let h = i - 1;
            let lhh = pc.tril[h][h];
            let lih = pc.tril[i][h];
            let std_i = (lii * lii + lih * lih).sqrt();
            let istds = [lhh.lift(1.0) / lhh, std_i.lift(1.0) / std_i];
            let mut blk = [b[h], b[i]];
            if i > 1 {
                for (v, row) in [h, i].into_iter().enumerate() {
                    let mut s = zero;
                    for c in 0..i - 1 {
                        s = s + pc.tril[row][c] * y[c];
                    }
                    blk[v] = [blk[v][0] - s, blk[v][1] - s];
                }
            }
            let xl = safe_mul(istds[0], blk[0][0]);
            let xu = safe_mul(istds[0], blk[0][1]);
            let yl = safe_mul(istds[1], blk[1][0]);
            let yu = safe_mul(istds[1], blk[1][1]);
            let corr = lhh * lih * (istds[0] * istds[1]);
            let p = bvn_generic(corr, xl, yl, xu, yu);
            if p.value() > 0.0 {
                let (zh, zi) = bvnmom_generic(corr, xl, yl, xu, yu, p);
                y[h] = zh;
                y[i] = inv_lii * (std_i * zi - lih * zh);
            } else {
                y[h] = zero;
                y[i] = zero;
            }
            log_prob = log_prob + safe_log(p);
        }
        pc.update(eps);
    }
    if n % 2 == 1 {
        log_prob = log_prob + safe_log(phi_i);
    }
    log_prob
}

/// `log P(X ∈ bounds)` for `X ~ N(0, Σ)` by BoTorch's MVNXPB, for the
/// covariance `cov` (`n × n`) and the bounds `bounds` (`n × 2`, lower and
/// upper; infinite bounds allowed).
///
/// # Errors
/// [`BotorchError::DimensionMismatch`] for inputs of the wrong shapes.
pub fn mvnxpb_log_prob(cov: &Matrix, bounds: &Matrix) -> Result<f64> {
    let n = cov.nrows();
    if n == 0 || cov.ncols() != n || bounds.shape() != (n, 2) {
        return Err(BotorchError::DimensionMismatch(format!(
            "expected an n x n covariance and n x 2 bounds, got {}x{} and {}x{}",
            cov.nrows(),
            cov.ncols(),
            bounds.nrows(),
            bounds.ncols()
        )));
    }
    let c: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| cov[(i, j)]).collect())
        .collect();
    let b: Vec<[f64; 2]> = (0..n).map(|i| [bounds[(i, 0)], bounds[(i, 1)]]).collect();
    Ok(mvnxpb_generic(&c, &b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition::dual::Dual;

    #[test]
    fn gauss_legendre_rules() {
        for n in [6, 12, 20] {
            let (x, w) = gauss_legendre(n);
            assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-14);
            // Exact for x^(2n-2).
            let k = 2 * n - 2;
            let integral: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(k as i32)).sum();
            assert!((integral - 2.0 / (k as f64 + 1.0)).abs() < 1e-14, "{n}");
            assert!(x.windows(2).all(|p| p[0] < p[1]));
        }
    }

    #[test]
    fn subgradients_at_ties_follow_torch() {
        // `torch.maximum(h, k)` in the Taylor branch splits the gradient at
        // h == k: bvnu is symmetric in (h, k), so are its subgradients.
        let v = |x: f64, i: usize, n: usize| Dual::<3>::variable(x, i, n);
        let p = bvnu(v(0.95, 0, 3), v(0.3, 1, 3), v(0.3, 2, 3));
        assert!(p.g[1] != 0.0 && (p.g[1] - p.g[2]).abs() <= 1e-15 * p.g[1].abs());
        // clip(0, 1), safe_log at tiny (scalar bounds): zero tangent at the
        // bound, value kept; safe_exp's tensor bound halves it at a tie.
        assert_eq!(clip01(v(1.0, 0, 1)).g[0], 0.0);
        assert_eq!(clip01(v(0.5, 0, 1)).g[0], 1.0);
        let t = safe_log(v(f64::MIN_POSITIVE, 0, 1));
        assert_eq!((t.v, t.g[0]), (f64::MIN_POSITIVE.ln(), 0.0));
        let e = safe_exp(v(MAX_EXP_ARG, 0, 1));
        assert_eq!(e.g[0], 0.5 * e.v);
    }

    #[test]
    fn bvn_closed_forms() {
        // Independent: the product of the marginals.
        let p = bvn(0.0, -0.3, 0.2, 1.1, 1.5);
        let e = (crate::utils::probability::ndtr(1.1) - crate::utils::probability::ndtr(-0.3))
            * (crate::utils::probability::ndtr(1.5) - crate::utils::probability::ndtr(0.2));
        assert!((p - e).abs() < 1e-15);
        // P(x < 0, y < 0) = 1/4 + asin(r) / (2π) (polar and Taylor branches).
        for r in [-0.95, -0.5, 0.2, 0.6, 0.8, 0.93, 0.99] {
            let p = bvn(r, -1e7, -1e7, 0.0, 0.0);
            let e = 0.25 + r_asin(r) / (2.0 * PI);
            assert!((p - e).abs() < 1e-14, "{r}: {p} vs {e}");
        }
    }

    fn r_asin(r: f64) -> f64 {
        r.asin()
    }

    #[test]
    fn mvnxpb_is_exact_for_two_variables_and_independent_ones() {
        let cov = Matrix::from_row_slice(2, 2, &[1.3, 0.6, 0.6, 0.8]);
        let b = Matrix::from_row_slice(2, 2, &[f64::NEG_INFINITY, 0.4, f64::NEG_INFINITY, -0.2]);
        let lp = mvnxpb_log_prob(&cov, &b).unwrap();
        let (s0, s1) = (1.3f64.sqrt(), 0.8f64.sqrt());
        let e = bvn(0.6 / (s0 * s1), -1e6, -1e6, 0.4 / s0, -0.2 / s1);
        assert!((lp - e.ln()).abs() < 1e-14);
        let cov = Matrix::from_diagonal(&crate::Vector::from_vec(vec![0.5, 1.0, 2.0]));
        let b = Matrix::from_row_slice(3, 2, &[-1.0, 0.3, f64::NEG_INFINITY, 0.1, -0.2, 2.0]);
        let lp = mvnxpb_log_prob(&cov, &b).unwrap();
        let nd = crate::utils::probability::ndtr;
        let e = (nd(0.3 / 0.5f64.sqrt()) - nd(-1.0 / 0.5f64.sqrt())).ln()
            + nd(0.1).ln()
            + (nd(2.0 / 2f64.sqrt()) - nd(-0.2 / 2f64.sqrt())).ln();
        assert!((lp - e).abs() < 1e-13, "{lp} vs {e}");
    }

    #[test]
    fn dual_derivatives_match_finite_differences() {
        let base = [[1.2, 0.5, -0.3], [0.5, 0.9, 0.2], [-0.3, 0.2, 1.4]];
        let ub = [0.3, -0.1, 0.8];
        let f = |cov: [[f64; 3]; 3], ub: [f64; 3]| {
            let c: Vec<Vec<f64>> = cov.iter().map(|r| r.to_vec()).collect();
            let b: Vec<[f64; 2]> = ub.iter().map(|u| [f64::NEG_INFINITY, *u]).collect();
            mvnxpb_generic(&c, &b)
        };
        // Tangents: the three upper bounds and the covariance (0, 1).
        let c: Vec<Vec<Dual<4>>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        if (i, j) == (0, 1) || (i, j) == (1, 0) {
                            Dual::variable(base[i][j], 3, 4)
                        } else {
                            Dual::constant(base[i][j], 4)
                        }
                    })
                    .collect()
            })
            .collect();
        let b: Vec<[Dual<4>; 2]> = (0..3)
            .map(|i| {
                [
                    Dual::constant(f64::NEG_INFINITY, 4),
                    Dual::variable(ub[i], i, 4),
                ]
            })
            .collect();
        let d = mvnxpb_generic(&c, &b);
        assert!((d.v - f(base, ub)).abs() < 1e-15);
        let h = 1e-6;
        for k in 0..3 {
            let (mut up, mut dn) = (ub, ub);
            up[k] += h;
            dn[k] -= h;
            let fd = (f(base, up) - f(base, dn)) / (2.0 * h);
            assert!((d.g[k] - fd).abs() < 1e-7, "{k}: {} vs {fd}", d.g[k]);
        }
        let (mut up, mut dn) = (base, base);
        up[0][1] += h;
        up[1][0] += h;
        dn[0][1] -= h;
        dn[1][0] -= h;
        let fd = (f(up, ub) - f(dn, ub)) / (2.0 * h);
        assert!((d.g[3] - fd).abs() < 1e-7, "{} vs {fd}", d.g[3]);
    }
}
