use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn polar(radius: f32, phase: f32) -> Self {
        Self {
            re: radius * phase.cos(),
            im: radius * phase.sin(),
        }
    }

    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn norm(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    pub fn mul_conj_rhs(self, rhs: Self) -> Self {
        Self {
            re: rhs.re * self.re + rhs.im * self.im,
            im: rhs.re * self.im - rhs.im * self.re,
        }
    }
}

impl From<f32> for Complex {
    fn from(value: f32) -> Self {
        Self::new(value, 0.0)
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Mul<f32> for Complex {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Mul<Complex> for f32 {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Self::Output {
        rhs * self
    }
}

impl Div<f32> for Complex {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}
