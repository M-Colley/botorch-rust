//! Kernel trees described as JSON *specs* (GPyTorch / BoTorch class names
//! and keyword arguments, see `botorch_pyo3/tests/kernel_tree_helpers.py`),
//! built as [`KernelNode`]s mirroring botorch_rs's kernel classes (the
//! fixtures of `kernels_6a_parity.rs` and `models_7b_parity.rs`).
//!
//! Extensions of phase 7b: the pseudo-kernel `DimScaledRBF` (BoTorch's
//! `get_covar_module_with_dim_scaled_prior`), the `LogTransformedInterval`
//! constraint, double-precision priors (`[name, args, {"double": true}]`) and
//! the SAAS prior of a `MaternKernel` (`"saas": {"tau": ...}`).

#![allow(dead_code)]

use botorch_core::constraints::{
    Constraint, GreaterThan, Interval, LessThan, LogTransformedInterval, ParameterTransform,
};
use botorch_core::kernels::{
    get_covar_module_with_dim_scaled_prior, KernelKind, KernelNode, MatrixPrior,
};
use botorch_core::Prior;
use serde_json::Value;

pub fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) => match s.as_str() {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            other => panic!("not a number: {other}"),
        },
        other => panic!("not a number: {other}"),
    }
}

// ----------------------------------------------------------------------
// Building the kernel tree of a spec (mirrors botorch_rs's kernel classes)
// ----------------------------------------------------------------------

pub fn prior_of(v: &Value) -> Prior {
    let name = v[0].as_str().unwrap();
    let a: Vec<f64> = v[1].as_array().unwrap().iter().map(num).collect();
    // `[name, args, {"double": true}]`: parameters given as float64 tensors.
    let double = v
        .get(2)
        .and_then(|o| o.get("double"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let p = match name {
        "GammaPrior" => Prior::gamma(a[0], a[1]),
        "LogNormalPrior" => Prior::log_normal(a[0], a[1]),
        "NormalPrior" => Prior::normal(a[0], a[1]),
        "UniformPrior" => Prior::uniform(a[0], a[1]),
        "HalfCauchyPrior" => Prior::half_cauchy(a[0]),
        "HalfNormalPrior" => Prior::half_normal(a[0]),
        "BetaPrior" => Prior::beta(a[0], a[1]),
        other => panic!("unknown prior {other}"),
    };
    if double {
        p.unwrap()
    } else {
        p.unwrap().rounded_to_f32()
    }
}

pub fn transform_of(kw: &Value, default: Option<ParameterTransform>) -> Option<ParameterTransform> {
    match kw.get("transform") {
        None => default,
        Some(Value::Null) => None,
        Some(Value::String(s)) if s == "softplus" => Some(ParameterTransform::Softplus),
        Some(Value::String(s)) if s == "sigmoid" => Some(ParameterTransform::Sigmoid),
        Some(other) => panic!("unknown transform {other}"),
    }
}

pub fn constraint_of(v: &Value) -> Constraint {
    let name = v[0].as_str().unwrap();
    let a: Vec<f64> = v[1].as_array().unwrap().iter().map(num).collect();
    let kw = &v[2];
    let init = kw.get("initial_value").map(num);
    match name {
        "Positive" => GreaterThan::new(
            0.0,
            transform_of(kw, Some(ParameterTransform::Softplus)),
            init,
        )
        .rounded_to_f32()
        .into(),
        "GreaterThan" => GreaterThan::new(
            a[0],
            transform_of(kw, Some(ParameterTransform::Softplus)),
            init,
        )
        .rounded_to_f32()
        .into(),
        "LessThan" => LessThan::new(
            a[0],
            transform_of(kw, Some(ParameterTransform::Softplus)),
            init,
        )
        .rounded_to_f32()
        .into(),
        "Interval" => {
            let t = transform_of(kw, Some(ParameterTransform::Sigmoid));
            Interval::new(a[0], a[1], t, None, init)
                .unwrap()
                .rounded_to_f32()
                .into()
        }
        "LogTransformedInterval" => LogTransformedInterval::new(a[0], a[1], init)
            .unwrap()
            .rounded_to_f32()
            .into(),
        other => panic!("unknown constraint {other}"),
    }
}

pub fn kw_prior(kw: &Value, key: &str) -> Option<Prior> {
    kw.get(key).map(|v| prior_of(&v["__prior__"]))
}

pub fn kw_constraint(kw: &Value, key: &str) -> Option<Constraint> {
    kw.get(key).map(|v| constraint_of(&v["__constraint__"]))
}

pub fn kw_usize(kw: &Value, key: &str) -> Option<usize> {
    kw.get(key).and_then(Value::as_u64).map(|v| v as usize)
}

pub fn with(
    mut node: KernelNode,
    kw: &Value,
    param: &str,
    prior: &str,
    constraint: &str,
) -> KernelNode {
    if let Some(c) = kw_constraint(kw, constraint) {
        node = node.with_constraint(param, c).unwrap();
    }
    if let Some(p) = kw_prior(kw, prior) {
        node = node.with_prior(param, p).unwrap();
    }
    node
}

pub fn lengthscale(node: KernelNode, kw: &Value) -> KernelNode {
    with(
        node,
        kw,
        "raw_lengthscale",
        "lengthscale_prior",
        "lengthscale_constraint",
    )
}

pub fn matern(
    nu: f64,
    ard: Option<usize>,
    prior: Option<Prior>,
    c: Option<Constraint>,
) -> KernelNode {
    let mut n = KernelNode::stationary(KernelKind::matern(nu).unwrap(), ard).unwrap();
    if let Some(p) = prior {
        n = n.with_prior("raw_lengthscale", p).unwrap();
    }
    if let Some(c) = c {
        n = n.with_constraint("raw_lengthscale", c).unwrap();
    }
    n
}

pub fn node(spec: &Value) -> KernelNode {
    let ty = spec["type"].as_str().unwrap();
    let kw = &spec["kwargs"];
    let args: Vec<&Value> = spec["args"].as_array().unwrap().iter().collect();
    let ard = kw_usize(kw, "ard_num_dims");
    let child = |v: &Value| node(&v["__kernel__"]);
    let mut n = match ty {
        "RBFKernel" => lengthscale(KernelNode::stationary(KernelKind::Rbf, ard).unwrap(), kw),
        "MaternKernel" => {
            let nu = kw.get("nu").map_or(2.5, num);
            let n = lengthscale(
                KernelNode::stationary(KernelKind::matern(nu).unwrap(), ard).unwrap(),
                kw,
            );
            match kw.get("saas") {
                Some(saas) => saas_prior(n, saas),
                None => n,
            }
        }
        "DimScaledRBF" => {
            // BoTorch's `get_covar_module_with_dim_scaled_prior`.
            let d = kw_usize(kw, "ard_num_dims").unwrap();
            let spec = get_covar_module_with_dim_scaled_prior(d, true);
            KernelNode::stationary(KernelKind::Rbf, Some(d))
                .unwrap()
                .with_prior("raw_lengthscale", spec.lengthscale_prior.unwrap())
                .unwrap()
                .with_constraint("raw_lengthscale", spec.lengthscale_constraint)
                .unwrap()
        }
        "RQKernel" => {
            let n = lengthscale(KernelNode::rational_quadratic(ard).unwrap(), kw);
            with(n, kw, "raw_alpha", "-", "alpha_constraint")
        }
        "PeriodicKernel" => {
            let n = lengthscale(KernelNode::periodic(ard).unwrap(), kw);
            with(
                n,
                kw,
                "raw_period_length",
                "period_length_prior",
                "period_length_constraint",
            )
        }
        "LinearKernel" => with(
            KernelNode::linear(ard).unwrap(),
            kw,
            "raw_variance",
            "variance_prior",
            "variance_constraint",
        ),
        "PolynomialKernel" => with(
            KernelNode::polynomial(num(&kw["power"])).unwrap(),
            kw,
            "raw_offset",
            "offset_prior",
            "offset_constraint",
        ),
        "CosineKernel" => with(
            KernelNode::cosine(),
            kw,
            "raw_period_length",
            "period_length_prior",
            "period_length_constraint",
        ),
        "IndexKernel" => {
            let mut n = KernelNode::index(
                kw_usize(kw, "num_tasks").unwrap(),
                kw_usize(kw, "rank").unwrap_or(1),
            )
            .unwrap();
            n = with(n, kw, "raw_var", "-", "var_constraint");
            if let Some(p) = kw.get("prior") {
                let p = &p["__prior__"];
                n = if p[0] == "LKJCovariancePrior" {
                    let a = p[1].as_array().unwrap();
                    n.with_task_prior(MatrixPrior::LkjCovariance {
                        n: a[0].as_u64().unwrap() as usize,
                        eta: num(&a[1]) as f32 as f64,
                        sd_prior: prior_of(&a[2]["__prior__"]),
                    })
                    .unwrap()
                } else {
                    n.with_task_prior(MatrixPrior::Elementwise(prior_of(p)))
                        .unwrap()
                };
            }
            n
        }
        "PositiveIndexKernel" => {
            let mut n = KernelNode::positive_index(
                kw_usize(kw, "num_tasks").unwrap(),
                kw_usize(kw, "rank").unwrap_or(1),
                kw_usize(kw, "target_task_index").unwrap_or(0),
                kw.get("unit_scale_for_target")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            )
            .unwrap();
            n = with(n, kw, "raw_var", "-", "var_constraint");
            if let Some(p) = kw_prior(kw, "task_prior") {
                n = n.with_task_prior(MatrixPrior::Elementwise(p)).unwrap();
            }
            n
        }
        "CategoricalKernel" => lengthscale(KernelNode::categorical(ard).unwrap(), kw),
        "DownsamplingKernel" => {
            let n = with(
                KernelNode::downsampling(),
                kw,
                "raw_power",
                "power_prior",
                "power_constraint",
            );
            with(n, kw, "raw_offset", "offset_prior", "offset_constraint")
        }
        "ExponentialDecayKernel" => {
            let n = lengthscale(KernelNode::exponential_decay(ard).unwrap(), kw);
            let n = with(n, kw, "raw_power", "power_prior", "power_constraint");
            with(n, kw, "raw_offset", "offset_prior", "offset_constraint")
        }
        "InfiniteWidthBNNKernel" => KernelNode::infinite_width_bnn(
            kw_usize(kw, "depth").unwrap_or(3),
            kw.get("acos_eps").map_or(1e-7, num),
        )
        .unwrap(),
        "LinearTruncatedFidelityKernel" => {
            let fid: Vec<usize> = kw["fidelity_dims"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let dimension = match kw.get("active_dims") {
                Some(a) => a.as_array().unwrap().len(),
                None => kw_usize(kw, "dimension").unwrap(),
            };
            let nu = kw.get("nu").map_or(2.5, num);
            let k = dimension - fid.len();
            let unbiased = match kw.get("covar_module_unbiased") {
                Some(v) => child(v),
                None => matern(
                    nu,
                    Some(k),
                    Some(
                        kw_prior(kw, "lengthscale_prior_unbiased")
                            .unwrap_or_else(|| Prior::gamma(3.0, 6.0).unwrap().rounded_to_f32()),
                    ),
                    kw_constraint(kw, "lengthscale_constraint_unbiased"),
                ),
            };
            let biased = match kw.get("covar_module_biased") {
                Some(v) => child(v),
                None => matern(
                    nu,
                    Some(k),
                    Some(
                        kw_prior(kw, "lengthscale_prior_biased")
                            .unwrap_or_else(|| Prior::gamma(6.0, 2.0).unwrap().rounded_to_f32()),
                    ),
                    kw_constraint(kw, "lengthscale_constraint_biased"),
                ),
            };
            let n = KernelNode::linear_truncated_fidelity(fid, unbiased, biased).unwrap();
            with(n, kw, "raw_power", "power_prior", "power_constraint")
        }
        "OrthogonalAdditiveKernel" => {
            let dim = kw_usize(kw, "dim").unwrap();
            let per_dim = kw
                .get("per_dim_lengthscales")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let base = match kw.get("base_kernel") {
                Some(v) => child(v),
                None => KernelNode::stationary(KernelKind::Rbf, None).unwrap(),
            };
            let coeff = kw_constraint(kw, "coeff_constraint").unwrap_or_else(Constraint::positive);
            let second = kw
                .get("second_order")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let mut n = KernelNode::orthogonal_additive(
                dim,
                base,
                per_dim,
                kw_usize(kw, "quad_deg").unwrap_or(32),
                second,
                kw.get("check_hypercube")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                coeff,
            )
            .unwrap();
            let exact = kw.get("dtype").is_some();
            let round = |v: f64| if exact { v } else { v as f32 as f64 };
            let log_d = (dim as f64).ln();
            n = n.with_default_raw("raw_coeffs_1", round(-log_d)).unwrap();
            if second {
                n = n
                    .with_default_raw("raw_coeffs_2", round(-2.0 * log_d))
                    .unwrap();
            }
            for (param, key) in [
                ("raw_offset", "offset_prior"),
                ("raw_coeffs_1", "coeffs_1_prior"),
                ("raw_coeffs_2", "coeffs_2_prior"),
            ] {
                if let Some(p) = kw_prior(kw, key) {
                    n = n.with_prior(param, p).unwrap();
                }
            }
            n
        }
        "ScaleKernel" => with(
            KernelNode::scale(child(args[0])),
            kw,
            "raw_outputscale",
            "outputscale_prior",
            "outputscale_constraint",
        ),
        "AdditiveKernel" => KernelNode::additive(args.iter().map(|a| child(a)).collect()).unwrap(),
        "ProductKernel" => KernelNode::product(args.iter().map(|a| child(a)).collect()).unwrap(),
        other => panic!("unknown kernel {other}"),
    };
    if let Some(dims) = kw.get("active_dims") {
        let dims: Vec<usize> = dims
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        n = n.with_active_dims(dims).unwrap();
    }
    n
}

/// BoTorch's `add_saas_prior` (`{"tau": null | value}`): `HalfCauchyPrior(1)`
/// on `tau / l^2`, with a fixed `tau` or a learned one
/// (`LogTransformedInterval(1e-3, 10, 0.1)`, `HalfCauchyPrior(0.1)`); the
/// priors in double precision (BoTorch's MAP SAAS models create them in the
/// kernel's dtype).
pub fn saas_prior(node: KernelNode, saas: &Value) -> KernelNode {
    let hc1 = Prior::half_cauchy(1.0).unwrap();
    match saas.get("tau").filter(|t| !t.is_null()) {
        Some(t) => node.with_saas_prior(hc1, Some(num(t)), None).unwrap(),
        None => {
            let c: Constraint = LogTransformedInterval::new(1e-3, 10.0, Some(0.1))
                .unwrap()
                .rounded_to_f32()
                .into();
            node.with_saas_prior(hc1, None, Some((c, Prior::half_cauchy(0.1).unwrap())))
                .unwrap()
        }
    }
}

pub fn flatten_value(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flatten_value).collect(),
        other => vec![num(other)],
    }
}
