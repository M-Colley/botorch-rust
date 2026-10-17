//! Fits through extreme hyperparameters (see `botorch_core::kernels::nonfinite`).
//!
//! L-BFGS-B's line search can try raw hyperparameters whose softplus
//! underflows or whose scaled inputs overflow. There GPyTorch's covariance
//! is NaN (BoTorch's closure returns NaN) or a prior density vanishes (loss
//! `+inf`), and BoTorch's fit continues from there; botorch_rs must see the
//! same values. The expected classes were measured with GPyTorch 1.15
//! (`test_parity_fit_nonfinite.py` compares them live), the fitted losses
//! with BoTorch (`fit_gpytorch_mll`, single-threaded torch) on the data of
//! `test_parity.make_data`.

use botorch_core::kernels::{get_matern_kernel_with_gamma_prior, KernelKind, KernelSpec};
use botorch_core::lbfgsb::{self, LbfgsbOptions, LbfgsbStatus};
use botorch_core::{
    fit_gpytorch_mll, get_gaussian_likelihood_with_gamma_prior, BotorchError,
    ExactMarginalLogLikelihood, FitOptions, Kernel, Matrix, SingleTaskGP, SingleTaskGPOptions,
    Vector,
};

/// `test_parity.make_data(12, 2, seed=104)`.
const X104: [f64; 24] = [
    0.9414392110545223,
    0.5550354874937108,
    0.7464058062949064,
    0.2759396528026299,
    0.32739686919533684,
    0.33721165329123115,
    0.34014491818203674,
    0.6124941805301982,
    0.9705828898596494,
    0.10840234570916762,
    0.9177527755760683,
    0.9704448017577858,
    0.6827089407642225,
    0.910189614930042,
    0.7911491419316273,
    0.7281035095698288,
    0.17509726668674497,
    0.13275539476769782,
    0.2264281661369768,
    0.6507108275119756,
    0.31805089676876686,
    0.8127432707556308,
    0.45190722289921026,
    0.39197369354417133,
];
const Y104: [f64; 12] = [
    -0.1423740040601822,
    -0.9188388350112173,
    0.9665611432022273,
    1.403051513342557,
    -0.41030878750773514,
    0.18495462848976868,
    0.10950400980002352,
    -0.3347603232666424,
    0.956836410217399,
    1.4228964455432431,
    1.5285795203412984,
    0.5662688920404767,
];
/// `test_parity.make_data(12, 2, seed=667)`.
const X667: [f64; 24] = [
    0.5782269641284062,
    0.46000373368381675,
    0.05346297252424426,
    0.08830007377940685,
    0.3278211996855944,
    0.8723755921377756,
    0.014817778737975407,
    0.6602400152233107,
    0.988444637221682,
    0.3638749281410536,
    0.854865493663122,
    0.352252902314089,
    0.23721455137799574,
    0.07338824078514772,
    0.51969867657366,
    0.7440483752214944,
    0.8496402767161237,
    0.9229708655896978,
    0.5596683947054086,
    0.2648218677032168,
    0.8580620901956287,
    0.1675536801222851,
    0.8665089406151023,
    0.6943399820962239,
];
const Y667: [f64; 12] = [
    -0.08746410304208192,
    0.4257416578719076,
    1.6816811813330852,
    0.48045520218374665,
    -0.24788297041071253,
    -0.80177327911087,
    0.8729197046887409,
    0.5296565861023209,
    -0.1448145034742344,
    -0.07563944885055766,
    -0.8869115844878375,
    -0.4324335794605003,
];
/// `test_parity.make_data(12, 3, seed=514)`.
const X514: [f64; 36] = [
    0.23160087034180699,
    0.4981016602335121,
    0.1785031614636372,
    0.672128042987133,
    0.7712367790400513,
    0.7114547393924395,
    0.6090506986727371,
    0.8086490999934551,
    0.9520692564089279,
    0.02495108445624361,
    0.5823102013904291,
    0.24130887629276243,
    0.2784795624833142,
    0.860239013526252,
    0.7824253877808544,
    0.5993326997267994,
    0.3544406867796125,
    0.10100372152668491,
    0.003417736087832135,
    0.8796446124621916,
    0.10663025717165542,
    0.6621080372182443,
    0.587420602179096,
    0.5759533370943112,
    0.35320791432822785,
    0.7241695827473346,
    0.30740823587487576,
    0.9514192357866373,
    0.5988142629879881,
    0.35986422749820846,
    0.11931017728248405,
    0.4173984928414407,
    0.45482081518486617,
    0.86207695527774,
    0.8255195935838385,
    0.6295433429504589,
];
const Y514: [f64; 12] = [
    1.464980197098555,
    1.452340808127272,
    2.716897499387261,
    0.8911392859322422,
    3.7124941676081047,
    -0.22125159647264003,
    1.0166686159589025,
    0.5902752446392037,
    1.8927380186494394,
    0.42564555864595743,
    1.4096064354840692,
    1.2056647881927425,
];

/// test_parity's `legacy_gamma` model: `ScaleKernel(MaternKernel(2.5))` with
/// Gamma priors and softplus constraints, the Gamma-prior likelihood.
fn legacy_gamma(x: &[f64], y: &[f64]) -> SingleTaskGP {
    SingleTaskGP::new(
        Matrix::from_row_slice(12, 2, x),
        Matrix::from_column_slice(12, 1, y),
        SingleTaskGPOptions {
            covar_module: Some(get_matern_kernel_with_gamma_prior(2)),
            likelihood: Some(get_gaussian_likelihood_with_gamma_prior()),
            ..Default::default()
        },
    )
    .unwrap()
}

/// test_parity's `custom_scale_rbf` model: the same priors and constraints
/// on an RBF kernel, BoTorch's default likelihood.
fn custom_scale_rbf() -> SingleTaskGP {
    let spec = KernelSpec {
        kind: KernelKind::Rbf,
        ..get_matern_kernel_with_gamma_prior(3)
    };
    SingleTaskGP::new(
        Matrix::from_row_slice(12, 3, &X514),
        Matrix::from_column_slice(12, 1, &Y514),
        SingleTaskGPOptions {
            covar_module: Some(spec),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Raw parameters: noise, constant, outputscale, lengthscales.
fn raw(model: &SingleTaskGP, outputscale: f64, lengthscales: &[f64]) -> Vec<f64> {
    let mut r = model.raw_parameters().as_slice().to_vec();
    r[2] = outputscale;
    r[3..].copy_from_slice(lengthscales);
    r
}

#[test]
fn mll_is_nan_where_gpytorch_sq_dist_overflows() {
    let model = legacy_gamma(&X104, &Y104);
    let mll = ExactMarginalLogLikelihood::new(&model);
    // A lengthscale that underflowed to zero (x / 0), one of 1e-293
    // (squares of the centered scaled inputs overflow) and one of 1e-109
    // (squares of about 1e216: finite in GPyTorch too).
    for (ls, nan) in [(-800.0, true), (-674.0, true), (-251.0, false)] {
        let r = raw(&model, 0.5, &[ls, 0.3]);
        match mll.value_and_grad(&r) {
            Err(BotorchError::Nan { count, total }) => {
                assert!(nan, "{ls}: unexpected NaN");
                assert!(count > 0 && total == 144);
            }
            Ok((v, g)) => {
                assert!(!nan, "{ls}: GPyTorch's covariance is NaN, got {v}");
                assert!(v.is_finite() && g.iter().all(|x| x.is_finite()));
            }
            Err(e) => panic!("{ls}: {e}"),
        }
    }
}

#[test]
fn zero_outputscale_gives_infinite_loss_and_nan_gradient() {
    // GPyTorch: GammaPrior(2, 0.15).log_prob(0) = -inf, and its derivative
    // inf times softplus'(-3074) = 0 is NaN; all other gradients are finite.
    let model = legacy_gamma(&X667, &Y667);
    let mll = ExactMarginalLogLikelihood::new(&model);
    let r = raw(&model, -3074.0, &[0.2, 0.4]);
    let (v, g) = mll.value_and_grad(&r).unwrap();
    assert_eq!(v, f64::NEG_INFINITY);
    for (i, gi) in g.iter().enumerate() {
        assert_eq!(gi.is_nan(), i == 2, "gradient {i}: {gi}");
    }
}

#[test]
fn user_specified_zero_hyperparameters_still_raise() {
    let mut model = legacy_gamma(&X104, &Y104);
    let r = raw(&model, 0.5, &[-800.0, 0.3]);
    assert!(matches!(
        model.set_raw_parameters(&r),
        Err(BotorchError::InvalidArgument(_))
    ));
    let r = raw(&model, -800.0, &[0.2, 0.3]);
    assert!(matches!(
        model.set_raw_parameters(&r),
        Err(BotorchError::InvalidArgument(_))
    ));
    assert!(Kernel::new(KernelKind::Rbf, Vector::from_vec(vec![0.0, 1.0]), None).is_err());
    assert!(Kernel::new(KernelKind::Rbf, Vector::from_vec(vec![1.0]), Some(0.0)).is_err());
}

#[test]
fn lbfgsb_handles_nan_trial_points_like_scipy() {
    // f(x) = x - ln(x), NaN for x <= 0, from x0 = 3: the second step lands
    // at x = -1, the line search extrapolates on NaN values until it hits
    // its maximum step, L-BFGS-B restarts from steepest descent and
    // converges. scipy 1.18 evaluates f at these 27 points (`nfev`; its
    // `ScalarFunction` does not call f again when the line search returns
    // to the point evaluated last, and neither does the port) and returns
    // x = 0.9999999603665313, f = 1.0000000000000009 after 7 iterations.
    let scipy: [f64; 27] = [
        3.0,
        2.0,
        -0.9999999999999978,
        -12.99999999999999,
        -60.99999999999995,
        -252.9999999999998,
        -1020.9999999999992,
        -4092.999999999997,
        -16380.999999999987,
        -65532.99999999995,
        -262140.9999999998,
        -1048572.9999999992,
        -4194300.9999999967,
        -16777212.999999987,
        -67108860.99999995,
        -268435452.9999998,
        -1073741820.9999992,
        -4294967292.9999967,
        -17179869180.999987,
        -29999999997.999977,
        1.5,
        0.49999999999999933,
        1.050145320811719,
        0.9749273395941405,
        1.001257276599655,
        1.0000315232692194,
        0.9999999603665313,
    ];
    let points = std::cell::RefCell::new(Vec::new());
    let f = |x: &[f64]| -> Result<(f64, Vec<f64>), BotorchError> {
        points.borrow_mut().push(x[0]);
        if x[0] > 0.0 {
            Ok((x[0] - x[0].ln(), vec![1.0 - 1.0 / x[0]]))
        } else {
            Ok((f64::NAN, vec![f64::NAN]))
        }
    };
    let r = lbfgsb::minimize(
        f,
        &[3.0],
        &[f64::NEG_INFINITY],
        &[f64::INFINITY],
        &LbfgsbOptions::default(),
    )
    .unwrap();
    assert_eq!(points.into_inner(), scipy);
    assert_eq!(r.evaluations, scipy.len());
    assert_eq!(r.x[0], 0.9999999603665313);
    assert_eq!(r.f, 1.0000000000000009);
    assert_eq!(r.iterations, 7);
    assert_eq!(r.status, LbfgsbStatus::ConvergedProjectedGradient);
}

/// BoTorch's final losses (twins agree to 2e-11 or better).
#[test]
fn fits_through_extreme_points_reach_botorch() {
    for (name, mut model, bt) in [
        (
            "legacy_gamma 104",
            legacy_gamma(&X104, &Y104),
            1.141575140486,
        ),
        (
            "legacy_gamma 667",
            legacy_gamma(&X667, &Y667),
            1.309324078517,
        ),
        ("custom_scale_rbf 514", custom_scale_rbf(), 0.867802027435),
    ] {
        let r = fit_gpytorch_mll(&mut model, FitOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(r.attempts, 1, "{name}");
        assert!((r.loss - bt).abs() <= 1e-6, "{name}: {} vs {bt}", r.loss);
    }
}
