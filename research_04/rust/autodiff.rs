use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Debug)]
pub struct DirectionalJet {
    pub value: f64,
    pub gradient: Vec<f64>,
    pub hvp: Vec<f64>,
    pub tangent: f64,
}

impl DirectionalJet {
    pub fn new(
        value: f64,
        gradient: Vec<f64>,
        hvp: Vec<f64>,
        tangent: f64,
    ) -> Self {
        assert!(
            !gradient.is_empty() && gradient.len() == hvp.len(),
            "Invalid derivative dimensions"
        );

        assert!(
            value.is_finite()
                && tangent.is_finite()
                && gradient.iter().all(|x| x.is_finite())
                && hvp.iter().all(|x| x.is_finite()),
            "Nonfinite value or derivative"
        );

        Self {
            value,
            gradient,
            hvp,
            tangent,
        }
    }

    pub fn constant(value: f64, n: usize) -> Self {
        assert!(n > 0, "Dimension must be positive");

        Self::new(value, vec![0.0; n], vec![0.0; n], 0.0)
    }

    pub fn variable(
        value: f64,
        index: usize,
        direction: &[f64],
    ) -> Self {
        let n = direction.len();

        assert!(n > 0 && index < n, "Invalid variable index");
        assert!(
            direction.iter().all(|x| x.is_finite()),
            "Nonfinite direction"
        );

        let mut gradient = vec![0.0; n];
        gradient[index] = 1.0;

        Self::new(value, gradient, vec![0.0; n], direction[index])
    }

    fn check_dimension(&self, other: &Self) {
        assert_eq!(
            self.gradient.len(),
            other.gradient.len(),
            "Derivative dimension mismatch"
        );
    }

    pub fn compose(
        &self,
        value: f64,
        first: f64,
        second: f64,
    ) -> Self {
        let gradient = self.gradient
            .iter()
            .map(|g| first * g)
            .collect();

        let hvp = self.hvp
            .iter()
            .zip(self.gradient.iter())
            .map(|(h, g)| first * h + second * self.tangent * g)
            .collect();

        Self::new(value, gradient, hvp, first * self.tangent)
    }

    pub fn reciprocal(&self) -> Self {
        assert!(self.value != 0.0, "Division by zero");

        let x = self.value;

        self.compose(
            1.0 / x,
            -1.0 / (x * x),
            2.0 / (x * x * x),
        )
    }

    pub fn exp(&self) -> Self {
        let y = self.value.exp();
        self.compose(y, y, y)
    }

    pub fn ln(&self) -> Self {
        assert!(self.value > 0.0, "Invalid logarithm argument");

        let x = self.value;

        self.compose(
            x.ln(),
            1.0 / x,
            -1.0 / (x * x),
        )
    }

    pub fn sin(&self) -> Self {
        self.compose(
            self.value.sin(),
            self.value.cos(),
            -self.value.sin(),
        )
    }

    pub fn cos(&self) -> Self {
        self.compose(
            self.value.cos(),
            -self.value.sin(),
            -self.value.cos(),
        )
    }
}

impl Add for DirectionalJet {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.check_dimension(&other);

        let gradient = self.gradient
            .iter()
            .zip(other.gradient.iter())
            .map(|(a, b)| a + b)
            .collect();

        let hvp = self.hvp
            .iter()
            .zip(other.hvp.iter())
            .map(|(a, b)| a + b)
            .collect();

        Self::new(
            self.value + other.value,
            gradient,
            hvp,
            self.tangent + other.tangent,
        )
    }
}

impl Neg for DirectionalJet {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(
            -self.value,
            self.gradient.iter().map(|x| -x).collect(),
            self.hvp.iter().map(|x| -x).collect(),
            -self.tangent,
        )
    }
}

impl Sub for DirectionalJet {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl Mul for DirectionalJet {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.check_dimension(&other);

        let n = self.gradient.len();
        let mut gradient = vec![0.0; n];
        let mut hvp = vec![0.0; n];

        for i in 0..n {
            gradient[i] =
                self.gradient[i] * other.value
                + self.value * other.gradient[i];

            hvp[i] =
                self.hvp[i] * other.value
                + self.value * other.hvp[i]
                + self.gradient[i] * other.tangent
                + self.tangent * other.gradient[i];
        }

        Self::new(
            self.value * other.value,
            gradient,
            hvp,
            self.tangent * other.value
                + self.value * other.tangent,
        )
    }
}
