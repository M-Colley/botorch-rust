//! Resumable L-BFGS-B state for batching independent objective requests.
//!
//! The subproblem and More-Thuente line search use the same routines and
//! operation order as the synchronous driver. No objective callback or worker
//! thread is retained: callers own scheduling and can clone a suspended state.
use super::*;

/// A reverse-communication event from an independent optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LbfgsbTask {
    /// Supply a value and gradient at [`LbfgsbState::x`].
    Evaluate,
    /// A complete step was accepted. Call `resume` after running callbacks.
    NewIterate,
    /// Optimization has terminated.
    Finished(LbfgsbStatus),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Initial,
    LineSearch,
    NewIterate,
    Finished(LbfgsbStatus),
}

/// Independent, clonable reverse-communication L-BFGS-B state.
#[derive(Clone, Debug)]
pub struct LbfgsbState {
    n: usize,
    m: usize,
    options: LbfgsbOptions,
    phase: Phase,
    nbd: I1,
    l: V1,
    u: V1,
    x: V1,
    g: V1,
    ws: Mat,
    wy: Mat,
    sy: Mat,
    ss: Mat,
    wt: Mat,
    wn: Mat,
    snd: Mat,
    z: V1,
    r: V1,
    d: V1,
    t: V1,
    xp: V1,
    wa: V1,
    index: I1,
    iwhere: I1,
    indx2: I1,
    cnstnd: bool,
    boxed: bool,
    col: usize,
    head: usize,
    theta: f64,
    iupdat: usize,
    updatd: bool,
    iback: usize,
    itail: usize,
    ileave: usize,
    nenter: usize,
    fold: f64,
    dnorm: f64,
    gd: f64,
    stpmx: f64,
    stp: f64,
    gdold: f64,
    dtd: f64,
    iter: usize,
    nfree: usize,
    ifun: usize,
    info: i32,
    f: f64,
    sbgnrm: f64,
    evaluations: usize,
    ls: LineSearch,
}

impl LbfgsbState {
    /// Validate inputs, project `x0` into its box, and request its evaluation.
    pub fn new(
        x0: &[f64],
        lower: &[f64],
        upper: &[f64],
        options: LbfgsbOptions,
    ) -> Result<Self, String> {
        let n = x0.len();
        if n == 0 {
            return Err("x0 must not be empty".into());
        }
        if lower.len() != n || upper.len() != n {
            return Err("bounds and x0 must have matching lengths".into());
        }
        if options.maxcor == 0 {
            return Err("maxcor must be positive".into());
        }
        if options.maxls == 0 {
            return Err("maxls must be positive".into());
        }
        if !options.ftol.is_finite()
            || options.ftol < 0.0
            || !options.gtol.is_finite()
            || options.gtol < 0.0
        {
            return Err("ftol and gtol must be finite and non-negative".into());
        }
        if x0.iter().any(|v| !v.is_finite()) || lower.iter().chain(upper).any(|v| v.is_nan()) {
            return Err("x0 must be finite and bounds must not contain NaN".into());
        }
        let mut nbd = I1::new(n);
        let mut l = V1::new(n);
        let mut u = V1::new(n);
        let mut x = V1(x0.to_vec());
        for i in 1..=n {
            let (lo, hi) = (lower[i - 1], upper[i - 1]);
            if lo > hi || lo == f64::INFINITY || hi == f64::NEG_INFINITY {
                return Err(format!("infeasible bounds for variable {}", i - 1));
            }
            let (has_lo, has_hi) = (lo.is_finite(), hi.is_finite());
            nbd[i] = match (has_lo, has_hi) {
                (false, false) => 0,
                (true, false) => 1,
                (true, true) => 2,
                (false, true) => 3,
            };
            l[i] = if has_lo { lo } else { 0.0 };
            u[i] = if has_hi { hi } else { 0.0 };
            x[i] = x[i].max(lo).min(hi);
        }
        let mut iwhere = I1::new(n);
        let (_, cnstnd, boxed) = active(n, &l, &u, &nbd, &mut x, &mut iwhere);
        let m = options.maxcor;
        Ok(Self {
            n,
            m,
            options,
            phase: Phase::Initial,
            nbd,
            l,
            u,
            x,
            g: V1::new(n),
            ws: Mat::new(n, m),
            wy: Mat::new(n, m),
            sy: Mat::new(m, m),
            ss: Mat::new(m, m),
            wt: Mat::new(m, m),
            wn: Mat::new(2 * m, 2 * m),
            snd: Mat::new(2 * m, 2 * m),
            z: V1::new(n),
            r: V1::new(n),
            d: V1::new(n),
            t: V1::new(n),
            xp: V1::new(n),
            wa: V1::new(8 * m),
            index: I1::new(n),
            iwhere,
            indx2: I1::new(n),
            cnstnd,
            boxed,
            col: 0,
            head: 1,
            theta: 1.0,
            iupdat: 0,
            updatd: false,
            iback: 0,
            itail: 0,
            ileave: 0,
            nenter: 0,
            fold: 0.0,
            dnorm: 0.0,
            gd: 0.0,
            stpmx: 0.0,
            stp: 0.0,
            gdold: 0.0,
            dtd: 0.0,
            iter: 0,
            nfree: n,
            ifun: 0,
            info: 0,
            f: 0.0,
            sbgnrm: 0.0,
            evaluations: 0,
            ls: LineSearch::new(),
        })
    }

    /// Current event. Repeated reads do not advance the optimizer.
    pub fn task(&self) -> LbfgsbTask {
        match self.phase {
            Phase::Initial | Phase::LineSearch => LbfgsbTask::Evaluate,
            Phase::NewIterate => LbfgsbTask::NewIterate,
            Phase::Finished(status) => LbfgsbTask::Finished(status),
        }
    }
    /// Current trial (or accepted/final) position.
    pub fn x(&self) -> &[f64] {
        &self.x.0
    }
    /// Most recently supplied value.
    pub fn value(&self) -> f64 {
        self.f
    }
    /// Most recently supplied gradient (restored on a failed line search).
    pub fn gradient(&self) -> &[f64] {
        &self.g.0
    }
    /// Number of accepted steps.
    pub fn iterations(&self) -> usize {
        self.iter
    }
    /// Number of value/gradient submissions.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }
    /// Number of curvature updates since the last memory restart.
    pub fn updates(&self) -> usize {
        self.iupdat
    }
    /// Stored correction vectors in the physical workspace order used by SciPy.
    pub fn corrections(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let k = self.iupdat.min(self.m);
        let rows = |a: &Mat| {
            (1..=k)
                .map(|j| (1..=self.n).map(|i| a[(i, j)]).collect())
                .collect()
        };
        (rows(&self.ws), rows(&self.wy))
    }
    /// Apply the reported limited-memory inverse Hessian (identity initial
    /// metric, as in SciPy's `LbfgsInvHessProduct`) to one vector.
    pub fn inverse_hessian_product(&self, vector: &[f64]) -> Result<Vec<f64>, String> {
        if vector.len() != self.n {
            return Err("inverse Hessian vector has the wrong length".into());
        }
        let (s, y) = self.corrections();
        let k = s.len();
        let mut rho = vec![0.0; k];
        let mut alpha = vec![0.0; k];
        let mut out = vector.to_vec();
        for i in (0..k).rev() {
            rho[i] = 1.0 / ddot(self.n, &s[i], &y[i]);
            alpha[i] = rho[i] * ddot(self.n, &s[i], &out);
            for (v, yi) in out.iter_mut().zip(&y[i]) {
                *v -= alpha[i] * yi;
            }
        }
        for i in 0..k {
            let beta = rho[i] * ddot(self.n, &y[i], &out);
            for (v, si) in out.iter_mut().zip(&s[i]) {
                *v += (alpha[i] - beta) * si;
            }
        }
        Ok(out)
    }
    /// Supply an objective result at the current requested position.
    /// Invalid shapes leave the pending request unchanged.
    pub fn provide_evaluation(&mut self, f: f64, gradient: &[f64]) -> Result<LbfgsbTask, String> {
        let initial = match self.phase {
            Phase::Initial => true,
            Phase::LineSearch => false,
            _ => return Err("the optimizer is not requesting an evaluation".into()),
        };
        if gradient.len() != self.n {
            return Err(format!(
                "gradient has length {} but x has length {}",
                gradient.len(),
                self.n
            ));
        }
        if !f.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err("objective value and gradient must be finite".into());
        }
        self.f = f;
        self.g.0.copy_from_slice(gradient);
        self.evaluations += 1;
        if initial {
            self.sbgnrm = projgr(self.n, &self.l, &self.u, &self.nbd, &self.x, &self.g);
            if self.sbgnrm <= self.options.gtol {
                self.phase = Phase::Finished(LbfgsbStatus::ConvergedProjectedGradient);
                return Ok(self.task());
            }
            self.prepare_direction();
        }
        self.advance_line();
        Ok(self.task())
    }
    /// Resume an accepted step, optionally stopping after the callback.
    pub fn resume(&mut self, stop: bool) -> Result<LbfgsbTask, String> {
        if self.phase != Phase::NewIterate {
            return Err("resume requires a newly accepted iterate".into());
        }
        let status = if stop {
            Some(LbfgsbStatus::CallbackStop)
        } else if self.iter >= self.options.maxiter {
            Some(LbfgsbStatus::MaxIterations)
        } else if self.evaluations > self.options.maxfun {
            Some(LbfgsbStatus::MaxEvaluations)
        } else if self.sbgnrm <= self.options.gtol {
            Some(LbfgsbStatus::ConvergedProjectedGradient)
        } else if self.fold - self.f
            <= self.options.ftol * self.fold.abs().max(self.f.abs()).max(1.0)
        {
            Some(LbfgsbStatus::ConvergedRelativeReduction)
        } else {
            None
        };
        if let Some(status) = status {
            self.phase = Phase::Finished(status);
            return Ok(self.task());
        }
        for i in 1..=self.n {
            self.r[i] = self.g[i] - self.r[i];
        }
        let rr = ddot(self.n, &self.r.0, &self.r.0);
        let (dr, ddum) = if self.stp == 1.0 {
            (self.gd - self.gdold, -self.gdold)
        } else {
            let dr = (self.gd - self.gdold) * self.stp;
            for i in 1..=self.n {
                self.d[i] *= self.stp;
            }
            (dr, -self.gdold * self.stp)
        };
        if dr <= f64::EPSILON * ddum {
            self.updatd = false;
        } else {
            self.updatd = true;
            self.iupdat += 1;
            matupd(
                self.n,
                self.m,
                &mut self.ws,
                &mut self.wy,
                &mut self.sy,
                &mut self.ss,
                &self.d,
                &self.r,
                &mut self.itail,
                self.iupdat,
                &mut self.col,
                &mut self.head,
                &mut self.theta,
                rr,
                dr,
                self.stp,
                self.dtd,
            );
            if formt(
                self.m,
                &mut self.wt,
                &self.sy,
                &self.ss,
                self.col,
                self.theta,
            ) != 0
            {
                self.reset_memory();
            }
        }
        self.prepare_direction();
        self.advance_line();
        Ok(self.task())
    }
    /// Final result, available only after termination.
    pub fn result(&self) -> Option<LbfgsbResult> {
        if let Phase::Finished(status) = self.phase {
            Some(LbfgsbResult {
                x: self.x.0.clone(),
                f: self.f,
                g: self.g.0.clone(),
                iterations: self.iter,
                evaluations: self.evaluations,
                status,
            })
        } else {
            None
        }
    }
    fn reset_memory(&mut self) {
        self.col = 0;
        self.head = 1;
        self.theta = 1.0;
        self.iupdat = 0;
        self.updatd = false;
    }
    fn prepare_direction(&mut self) {
        loop {
            let wrk = if !self.cnstnd && self.col > 0 {
                self.z.0.copy_from_slice(&self.x.0);
                self.updatd
            } else {
                self.info = cauchy(
                    self.n,
                    &self.x,
                    &self.l,
                    &self.u,
                    &self.nbd,
                    &self.g,
                    &mut self.indx2,
                    &mut self.iwhere,
                    &mut self.t,
                    &mut self.d,
                    &mut self.z,
                    self.m,
                    &self.wy,
                    &self.ws,
                    &self.sy,
                    &self.wt,
                    self.theta,
                    self.col,
                    self.head,
                    &mut self.wa,
                    self.sbgnrm,
                    f64::EPSILON,
                );
                if self.info != 0 {
                    // Singular triangular system in bmv: refresh the memory.
                    self.col = 0;
                    self.head = 1;
                    self.theta = 1.0;
                    self.iupdat = 0;
                    self.updatd = false;
                    continue;
                }
                freev(
                    self.n,
                    &mut self.nfree,
                    &mut self.index,
                    &mut self.nenter,
                    &mut self.ileave,
                    &mut self.indx2,
                    &self.iwhere,
                    self.updatd,
                    self.cnstnd,
                    self.iter,
                )
            };
            // ---- L333 ----------------------------------------------------------
            if self.nfree != 0 && self.col != 0 {
                self.info = 0;
                if wrk {
                    self.info = formk(
                        self.n,
                        self.nfree,
                        &self.index,
                        self.nenter,
                        self.ileave,
                        &self.indx2,
                        self.iupdat,
                        self.updatd,
                        &mut self.wn,
                        &mut self.snd,
                        self.m,
                        &self.ws,
                        &self.wy,
                        &self.sy,
                        self.theta,
                        self.col,
                        self.head,
                    );
                }
                if self.info == 0 {
                    self.info = cmprlb(
                        self.n,
                        self.m,
                        &self.x,
                        &self.g,
                        &self.ws,
                        &self.wy,
                        &self.sy,
                        &self.wt,
                        &self.z,
                        &mut self.r,
                        &mut self.wa,
                        &self.index,
                        self.theta,
                        self.col,
                        self.head,
                        self.nfree,
                        self.cnstnd,
                    );
                }
                if self.info == 0 {
                    self.info = subsm(
                        self.n,
                        self.m,
                        self.nfree,
                        &self.index,
                        &self.l,
                        &self.u,
                        &self.nbd,
                        &mut self.z,
                        &mut self.r,
                        &mut self.xp,
                        &self.ws,
                        &self.wy,
                        self.theta,
                        &self.x,
                        &self.g,
                        self.col,
                        self.head,
                        &mut self.wa,
                        &self.wn,
                    );
                }
                if self.info != 0 {
                    self.col = 0;
                    self.head = 1;
                    self.theta = 1.0;
                    self.iupdat = 0;
                    self.updatd = false;
                    continue;
                }
            }
            // ---- L555: search direction self.d = self.z - self.x -------------------------------
            for i in 1..=self.n {
                self.d[i] = self.z[i] - self.x[i];
            }
            // ---- L666: line search (lnsrlb) -----------------------------------
            self.dtd = ddot(self.n, &self.d.0, &self.d.0);
            self.dnorm = self.dtd.sqrt();
            self.stpmx = 1e10;
            if self.cnstnd {
                if self.iter == 0 {
                    self.stpmx = 1.0;
                } else {
                    for i in 1..=self.n {
                        let a1 = self.d[i];
                        if self.nbd[i] != 0 {
                            if a1 < 0.0 && self.nbd[i] <= 2 {
                                let a2 = self.l[i] - self.x[i];
                                if a2 >= 0.0 {
                                    self.stpmx = 0.0;
                                } else if a1 * self.stpmx < a2 {
                                    self.stpmx = a2 / a1;
                                }
                            } else if a1 > 0.0 && self.nbd[i] >= 2 {
                                let a2 = self.u[i] - self.x[i];
                                if a2 <= 0.0 {
                                    self.stpmx = 0.0;
                                } else if a1 * self.stpmx > a2 {
                                    self.stpmx = a2 / a1;
                                }
                            }
                        }
                    }
                }
            }
            self.stp = if self.iter == 0 && !self.boxed {
                (1.0 / self.dnorm).min(self.stpmx)
            } else {
                1.0
            };
            self.t.0.copy_from_slice(&self.x.0);
            self.r.0.copy_from_slice(&self.g.0);
            self.fold = self.f;
            self.ifun = 0usize;
            self.iback = 0usize;
            self.info = 0;
            self.ls = LineSearch::new();
            return;
        }
    }
    fn advance_line(&mut self) {
        loop {
            self.gd = ddot(self.n, &self.g.0, &self.d.0);
            if self.ifun == 0 {
                self.gdold = self.gd;
                if self.gd >= 0.0 {
                    self.info = -4;
                }
            }
            if self.info == 0 {
                self.ls.dcsrch(
                    self.f,
                    self.gd,
                    &mut self.stp,
                    FTOL,
                    GTOL,
                    XTOL,
                    STPMIN,
                    self.stpmx,
                );
                if !self.ls.task.is_warning() && !self.ls.task.is_converged() {
                    self.ifun += 1;
                    self.iback = self.ifun - 1;
                    if self.stp == 1.0 {
                        self.x.0.copy_from_slice(&self.z.0);
                    } else {
                        for i in 1..=self.n {
                            self.x[i] = self.stp * self.d[i] + self.t[i];
                        }
                    }
                    if self.iback < self.options.maxls {
                        self.phase = Phase::LineSearch;
                        return;
                    }
                }
            }
            if self.info != 0 || self.iback >= self.options.maxls {
                self.x.0.copy_from_slice(&self.t.0);
                self.g.0.copy_from_slice(&self.r.0);
                self.f = self.fold;
                if self.col == 0 {
                    self.phase = Phase::Finished(LbfgsbStatus::AbnormalLineSearch);
                    return;
                }
                self.reset_memory();
                self.prepare_direction();
                continue;
            }
            self.iter += 1;
            self.sbgnrm = projgr(self.n, &self.l, &self.u, &self.nbd, &self.x, &self.g);
            self.phase = Phase::NewIterate;
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> Result<(f64, Vec<f64>), ()> {
        let a = x[1] - x[0] * x[0];
        Ok((
            100.0 * a * a + (1.0 - x[0]).powi(2),
            vec![-400.0 * x[0] * a + 2.0 * (x[0] - 1.0), 200.0 * a],
        ))
    }

    fn drive(state: &mut LbfgsbState, stop_at: usize) -> (LbfgsbResult, Vec<Vec<f64>>) {
        let mut accepted = Vec::new();
        for _ in 0..10_000 {
            match state.task() {
                LbfgsbTask::Evaluate => {
                    let (f, g) = rosenbrock(state.x()).unwrap();
                    state.provide_evaluation(f, &g).unwrap();
                }
                LbfgsbTask::NewIterate => {
                    accepted.push(state.x().to_vec());
                    state.resume(state.iterations() == stop_at).unwrap();
                }
                LbfgsbTask::Finished(_) => return (state.result().unwrap(), accepted),
            }
        }
        panic!("optimizer failed to terminate");
    }

    #[test]
    fn reverse_matches_synchronous_every_accepted_step() {
        for bounds in [
            (vec![-2.0, -1.0], vec![2.0, 3.0]),
            (vec![f64::NEG_INFINITY; 2], vec![f64::INFINITY; 2]),
            (vec![0.0, 0.0], vec![0.8, 0.7]),
        ] {
            for (maxiter, maxfun, maxls, stop_at) in [
                (15000, 15000, 20, usize::MAX),
                (3, 15000, 20, usize::MAX),
                (15000, 2, 20, usize::MAX),
                (15000, 15000, 1, usize::MAX),
                (15000, 15000, 20, 4),
            ] {
                let options = LbfgsbOptions {
                    maxiter,
                    maxfun,
                    maxls,
                    ..Default::default()
                };
                let mut expected_steps = Vec::new();
                let expected = minimize_with_callback(
                    rosenbrock,
                    &[-1.2, 1.0],
                    &bounds.0,
                    &bounds.1,
                    &options,
                    |iteration, x, _| {
                        expected_steps.push(x.to_vec());
                        iteration == stop_at
                    },
                )
                .unwrap();
                let mut state =
                    LbfgsbState::new(&[-1.2, 1.0], &bounds.0, &bounds.1, options).unwrap();
                let (actual, steps) = drive(&mut state, stop_at);
                assert_eq!(steps, expected_steps);
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn suspended_clone_preserves_line_search_and_curvature() {
        let mut state = LbfgsbState::new(
            &[-1.2, 1.0],
            &[-2.0, -1.0],
            &[2.0, 3.0],
            LbfgsbOptions::default(),
        )
        .unwrap();
        while state.iterations() < 4 || state.task() != LbfgsbTask::Evaluate {
            match state.task() {
                LbfgsbTask::Evaluate => {
                    let (f, g) = rosenbrock(state.x()).unwrap();
                    state.provide_evaluation(f, &g).unwrap();
                }
                LbfgsbTask::NewIterate => {
                    state.resume(false).unwrap();
                }
                _ => panic!("premature termination"),
            }
        }
        let mut fork = state.clone();
        assert!(state.updates() > 0);
        assert_eq!(drive(&mut state, usize::MAX), drive(&mut fork, usize::MAX));
        assert_eq!(state.corrections(), fork.corrections());
    }

    #[test]
    fn invalid_submission_does_not_consume_pending_request() {
        let mut state =
            LbfgsbState::new(&[0.0, 0.0], &[-1.0; 2], &[1.0; 2], LbfgsbOptions::default()).unwrap();
        assert!(state.resume(false).is_err());
        assert!(state.provide_evaluation(1.0, &[1.0]).is_err());
        assert_eq!(state.evaluations(), 0);
        assert_eq!(state.task(), LbfgsbTask::Evaluate);
        assert!(matches!(
            state.provide_evaluation(0.0, &[0.0, 0.0]).unwrap(),
            LbfgsbTask::Finished(LbfgsbStatus::ConvergedProjectedGradient)
        ));
        assert!(state.provide_evaluation(0.0, &[0.0, 0.0]).is_err());
    }
}
