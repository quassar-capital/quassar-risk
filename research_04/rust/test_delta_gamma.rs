#[path = "delta_gamma.rs"]
mod delta_gamma;

use delta_gamma::autodiff::DirectionalJet;
use delta_gamma::delta_gamma as calculate;

fn check(actual: f64, expected: f64, label: &str) {
    assert!(
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs()
                <= 1e-11 * (1.0 + expected.abs()),
        "{}: actual={}, expected={}",
        label,
        actual,
        expected
    );
}

fn quadratic(x: &[DirectionalJet]) -> DirectionalJet {
    let a = [
        [4.0, 1.0, -2.0],
        [1.0, 6.0, 0.5],
        [-2.0, 0.5, 8.0],
    ];
    let b = [1.5, -0.8, 2.0];

    let mut result = DirectionalJet::constant(0.0, x.len());

    for i in 0..x.len() {
        result = result
            + DirectionalJet::constant(b[i], x.len())
                * x[i].clone();

        for j in 0..x.len() {
            result = result
                + DirectionalJet::constant(
                    0.5 * a[i][j], x.len()
                ) * x[i].clone() * x[j].clone();
        }
    }

    result
}

fn nonlinear(x: &[DirectionalJet]) -> DirectionalJet {
    let two = DirectionalJet::constant(2.0, x.len());

    (x[0].clone() * x[1].clone()).exp()
        + x[0].sin()
        + (x[1].clone() * x[1].clone() + two).ln()
}

fn main() {
    // Test 1: Independent quadratic reference.
    let a = [
        [4.0, 1.0, -2.0],
        [1.0, 6.0, 0.5],
        [-2.0, 0.5, 8.0],
    ];
    let b = [1.5, -0.8, 2.0];
    let point = [0.3, -0.7, 1.2];
    let shock = [0.12, -0.09, 0.05];

    let result = calculate(quadratic, &point, &shock);

    let mut expected_delta = 0.0;
    let mut expected_gamma = 0.0;

    for i in 0..3 {
        let mut gradient = b[i];

        for j in 0..3 {
            gradient += a[i][j] * point[j];
            expected_gamma +=
                0.5 * shock[i] * a[i][j] * shock[j];
        }

        expected_delta += gradient * shock[i];
    }

    check(result.delta, expected_delta, "Quadratic delta");
    check(result.gamma, expected_gamma, "Quadratic gamma");
    check(
        result.total,
        expected_delta + expected_gamma,
        "Quadratic total",
    );

    // Exact change in value for a quadratic function.
    let shifted = [
        point[0] + shock[0],
        point[1] + shock[1],
        point[2] + shock[2],
    ];

    let evaluate = |p: &[f64; 3]| -> f64 {
        let mut value = 0.0;

        for i in 0..3 {
            value += b[i] * p[i];

            for j in 0..3 {
                value += 0.5 * a[i][j] * p[i] * p[j];
            }
        }

        value
    };

    check(
        result.total,
        evaluate(&shifted) - evaluate(&point),
        "Exact quadratic valuation change",
    );

    println!("Quadratic analytical reference: PASSED");

    // Test 2: Independent nonlinear derivatives.
    // f(x,y) = exp(x*y) + sin(x) + log(y*y+2)
    let p: [f64; 2] = [0.4, -0.2];
    let dx: [f64; 2] = [0.07, -0.05];

    let r = calculate(nonlinear, &p, &dx);

    let x = p[0];
    let y = p[1];
    let e = (x * y).exp();
    let d = y * y + 2.0;

    let gx = y * e + x.cos();
    let gy = x * e + 2.0 * y / d;

    let hxx = y * y * e - x.sin();
    let hxy = (1.0 + x * y) * e;
    let hyy = x * x * e
        + 2.0 * (2.0 - y * y) / (d * d);

    let analytical_delta = gx * dx[0] + gy * dx[1];

    let analytical_gamma = 0.5 * (
        hxx * dx[0] * dx[0]
        + 2.0 * hxy * dx[0] * dx[1]
        + hyy * dx[1] * dx[1]
    );

    check(r.delta, analytical_delta, "Nonlinear delta");
    check(r.gamma, analytical_gamma, "Nonlinear gamma");
    check(
        r.total,
        analytical_delta + analytical_gamma,
        "Nonlinear total",
    );

    println!("Nonlinear analytical reference: PASSED");

    // Test 3: Zero shock.
    let zero = calculate(quadratic, &point, &[0.0, 0.0, 0.0]);

    check(zero.delta, 0.0, "Zero-shock delta");
    check(zero.gamma, 0.0, "Zero-shock gamma");
    check(zero.total, 0.0, "Zero-shock total");

    // Test 4: Invalid dimensions.
    let rejected = std::panic::catch_unwind(|| {
        calculate(quadratic, &point, &[0.1, 0.2]);
    });

    assert!(
        rejected.is_err(),
        "Invalid shock dimensions were accepted"
    );

    println!("Zero shock and input validation: PASSED");
    println!(
        "RESEARCH 04: RUST MATRIX-FREE DELTA-GAMMA VALIDATION PASSED"
    );
}
