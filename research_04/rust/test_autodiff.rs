#[path = "autodiff.rs"]
mod autodiff;

use autodiff::DirectionalJet;

fn check(actual: f64, expected: f64, label: &str) {
    assert!(
        actual.is_finite()
            && (actual - expected).abs()
                <= 1e-11 * (1.0 + expected.abs()),
        "{}: actual={}, expected={}",
        label,
        actual,
        expected
    );
}

fn model(x: DirectionalJet, y: DirectionalJet) -> DirectionalJet {
    x.clone() * x.clone() * y.clone()
        + x.exp()
        + y.sin()
}

fn main() {
    let a: f64 = 1.2;
    let b: f64 = 0.7;

    let expected_value = a * a * b + a.exp() + b.sin();

    let gradient = [
        2.0 * a * b + a.exp(),
        a * a + b.cos(),
    ];

    let hessian = [
        [2.0 * b + a.exp(), 2.0 * a],
        [2.0 * a, -b.sin()],
    ];

    let directions = [
        [1.0, 0.0],
        [0.4, -0.8],
        [0.0, 0.0],
    ];

    for direction in directions {
        let x = DirectionalJet::variable(a, 0, &direction);
        let y = DirectionalJet::variable(b, 1, &direction);

        let result = model(x, y);

        check(result.value, expected_value, "Value");

        for i in 0..2 {
            check(
                result.gradient[i],
                gradient[i],
                "Gradient",
            );

            let expected_hvp =
                hessian[i][0] * direction[0]
                + hessian[i][1] * direction[1];

            check(
                result.hvp[i],
                expected_hvp,
                "Hessian-vector product",
            );
        }

        let expected_tangent =
            gradient[0] * direction[0]
            + gradient[1] * direction[1];

        check(
            result.tangent,
            expected_tangent,
            "Directional derivative",
        );
    }

    // Independent logarithmic derivative reference.
    let direction = [1.0];
    let z = DirectionalJet::variable(2.0, 0, &direction);

    let g = (
        z.clone() * z.clone()
        + DirectionalJet::constant(1.0, 1)
    ).ln();

    check(g.value, 5.0_f64.ln(), "Logarithmic value");
    check(g.gradient[0], 4.0 / 5.0, "Logarithmic gradient");
    check(g.hvp[0], -6.0 / 25.0, "Logarithmic HVP");

    // Reciprocal derivative reference.
    let reciprocal = z.reciprocal();

    check(reciprocal.value, 0.5, "Reciprocal value");
    check(reciprocal.gradient[0], -0.25, "Reciprocal gradient");
    check(reciprocal.hvp[0], 0.25, "Reciprocal HVP");

    // Input validation.
    let invalid = std::panic::catch_unwind(|| {
        DirectionalJet::variable(1.0, 2, &[1.0, 0.0]);
    });

    assert!(invalid.is_err(), "Invalid index was accepted");

    println!("RESEARCH 04: RUST AUTODIFF VALIDATION PASSED");
    println!(
        "Analytical gradients, matrix-free HVPs, \
         nonlinear derivatives and input checks verified."
    );
}
