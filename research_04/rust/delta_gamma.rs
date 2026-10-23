#[path = "autodiff.rs"]
pub mod autodiff;

use autodiff::DirectionalJet;

#[derive(Debug, Clone, Copy)]
pub struct DeltaGammaResult {
    pub delta: f64,
    pub gamma: f64,
    pub total: f64,
}

fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;

    for value in values {
        assert!(value.is_finite(), "Nonfinite contribution");

        let next = sum + value;

        if sum.abs() >= value.abs() {
            correction += (sum - next) + value;
        } else {
            correction += (value - next) + sum;
        }

        sum = next;
    }

    let result = sum + correction;
    assert!(result.is_finite(), "Nonfinite accumulated result");
    result
}

pub fn delta_gamma<F>(
    model: F,
    point: &[f64],
    shock: &[f64],
) -> DeltaGammaResult
where
    F: FnOnce(&[DirectionalJet]) -> DirectionalJet,
{
    let n = point.len();

    assert!(n > 0, "Dimension must be positive");
    assert_eq!(n, shock.len(), "Point/shock dimension mismatch");

    assert!(
        point.iter().chain(shock.iter()).all(|x| x.is_finite()),
        "Nonfinite point or shock"
    );

    let variables: Vec<DirectionalJet> = point
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            DirectionalJet::variable(value, i, shock)
        })
        .collect();

    let result = model(&variables);

    assert_eq!(
        result.gradient.len(),
        n,
        "Model gradient dimension mismatch"
    );

    assert_eq!(
        result.hvp.len(),
        n,
        "Model HVP dimension mismatch"
    );

    let delta = compensated_sum(
        result.gradient
            .iter()
            .zip(shock.iter())
            .map(|(g, s)| g * s),
    );

    let gamma = compensated_sum(
        result.hvp
            .iter()
            .zip(shock.iter())
            .map(|(h, s)| 0.5 * h * s),
    );

    let total = delta + gamma;

    assert!(total.is_finite(), "Nonfinite total approximation");

    DeltaGammaResult {
        delta,
        gamma,
        total,
    }
}
