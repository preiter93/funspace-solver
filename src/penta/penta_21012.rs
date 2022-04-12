//! Penta-diagonal matrix solver
//!     Ax = b
//! where A is banded with diagonals in offsets -2, 1, 0, 1, 2
#![allow(clippy::many_single_char_names)]
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Penta21012<T> {
    /// Lower diagonal (-2)
    pub(crate) l2: Vec<T>,
    /// Lower diagonal (-1)
    pub(crate) l1: Vec<T>,
    /// Main diagonal (0)
    pub(crate) d0: Vec<T>,
    /// Upper diagonal (+1)
    pub(crate) u1: Vec<T>,
    /// Upper diagonal (+2)
    pub(crate) u2: Vec<T>,
}

impl<T> Penta21012<T> {
    /// Constructor
    ///
    /// ## Panics
    /// Shape mismatch of diagonals
    #[must_use]
    pub fn new(l2: Vec<T>, l1: Vec<T>, d0: Vec<T>, u1: Vec<T>, u2: Vec<T>) -> Self {
        assert!(d0.len() > 3, "Problem size too small.");
        assert!(l2.len() == d0.len() - 2);
        assert!(l1.len() == d0.len() - 1);
        assert!(u1.len() == d0.len() - 1);
        assert!(u2.len() == d0.len() - 2);
        Self { l2, l1, d0, u1, u2 }
    }

    /// Construct from matrix
    ///
    /// # Panics
    /// - Matrix is non-square
    /// - Matrix is non-zero on some diagonal other than -2, 1, 0, 1, 2
    #[must_use]
    pub fn from_array2(array: &ndarray::Array2<T>) -> Self
    where
        T: Zero + Copy + PartialEq,
    {
        let diagonals = crate::utils::extract_diagonals(array, [-2, 1, 0, 1, 2]);
        let diagonals = match diagonals {
            Ok(diagonals) => diagonals,
            Err(error) => panic!("{}", error.0),
        };
        let l2 = diagonals[0].clone();
        let l1 = diagonals[1].clone();
        let d0 = diagonals[2].clone();
        let u1 = diagonals[3].clone();
        let u2 = diagonals[4].clone();
        Self::new(l2, l1, d0, u1, u2)
    }

    /// Returns banded matrix
    #[must_use]
    pub fn as_array2(&self) -> ndarray::Array2<T>
    where
        T: Zero + Copy,
    {
        match crate::utils::array_from_diags(
            [&self.l2, &self.l1, &self.d0, &self.u1, &self.u2],
            [-2, -1, 0, 1, 2],
            self.d0.len(),
        ) {
            Ok(array) => array,
            Err(error) => panic!("{}", error.0),
        }
    }

    /// # Dot product
    /// ```text
    /// A x = b
    ///```
    /// Returns b
    ///
    /// ## Panics
    /// - ``x.len() /= d0.len()``
    pub fn dot<U>(&self, x: &[U]) -> Vec<U>
    where
        T: Copy,
        U: Zero + Copy + Add<Output = U> + Mul<T, Output = U>,
    {
        let n = x.len();
        assert!(
            n == self.d0.len(),
            "Shape mismatch. {} /= {}",
            n,
            self.d0.len()
        );
        let mut b = vec![U::zero(); n];
        let (l2, l1, d0, u1, u2) = (&self.l2, &self.l1, &self.d0, &self.u1, &self.u2);
        b[0] = x[0] * d0[0] + x[1] * u1[0] + x[2] * u2[0];
        b[1] = x[1] * d0[1] + x[2] * u1[1] + x[3] * u2[1] + x[0] * l1[0];
        for i in 2..n - 2 {
            b[i] = x[i] * d0[i]
                + x[i + 1] * u1[i]
                + x[i + 2] * u2[i]
                + x[i - 1] * l1[i - 1]
                + x[i - 2] * l2[i - 2];
        }
        b[n - 2] = x[n - 2] * d0[n - 2]
            + x[n - 1] * u1[n - 2]
            + x[n - 3] * l1[n - 3]
            + x[n - 4] * l2[n - 4];
        b[n - 1] = x[n - 1] * d0[n - 1] + x[n - 2] * l1[n - 2] + x[n - 3] * l2[n - 3];
        b
    }

    /// # Dot product
    /// ```text
    /// A x = b
    ///```
    /// Returns b
    ///
    /// ## Panics
    /// - ``x.len() /= d0.len()``
    ///
    /// ## Note
    /// ** Unsafe ** due to unchecked bounds
    pub fn dot_unchecked<U>(&self, x: &[U]) -> Vec<U>
    where
        T: Copy,
        U: Zero + Copy + Add<Output = U> + Mul<T, Output = U>,
    {
        let n = x.len();
        assert!(
            n == self.d0.len(),
            "Shape mismatch. {} /= {}",
            n,
            self.d0.len()
        );
        let mut b = vec![U::zero(); n];
        let (l2, l1, d0, u1, u2) = (&self.l2, &self.l1, &self.d0, &self.u1, &self.u2);
        unsafe {
            *b.get_unchecked_mut(0) = *x.get_unchecked(0) * *d0.get_unchecked(0)
                + *x.get_unchecked(1) * *u1.get_unchecked(0)
                + *x.get_unchecked(2) * *u2.get_unchecked(0);
            *b.get_unchecked_mut(1) = *x.get_unchecked(1) * *d0.get_unchecked(1)
                + *x.get_unchecked(2) * *u1.get_unchecked(1)
                + *x.get_unchecked(3) * *u2.get_unchecked(1)
                + *x.get_unchecked(0) * *l1.get_unchecked(0);
            for i in 2..n - 2 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 1) * *u1.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i - 1) * *l1.get_unchecked(i - 1)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2);
            }
            *b.get_unchecked_mut(n - 2) = *x.get_unchecked(n - 2) * *d0.get_unchecked(n - 2)
                + *x.get_unchecked(n - 1) * *u1.get_unchecked(n - 2)
                + *x.get_unchecked(n - 3) * *l1.get_unchecked(n - 3)
                + *x.get_unchecked(n - 4) * *l2.get_unchecked(n - 4);
            *b.get_unchecked_mut(n - 1) = *x.get_unchecked(n - 1) * *d0.get_unchecked(n - 1)
                + *x.get_unchecked(n - 2) * *l1.get_unchecked(n - 2)
                + *x.get_unchecked(n - 3) * *l2.get_unchecked(n - 3);
        }
        b
    }

    /// # Solve
    /// ```text
    /// A x = b
    ///```
    /// Returns x
    ///
    /// ## Panics
    /// - ``b.len() /= d0.len()``
    pub fn solve<U>(&self, b: &[U]) -> Vec<U>
    where
        T: Zero + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
        U: Zero
            + Copy
            + Add<Output = U>
            + Sub<Output = U>
            + Mul<T, Output = U>
            + Div<T, Output = U>,
    {
        let n = b.len();
        assert!(
            n == self.d0.len(),
            "Shape mismatch. {} /= {}",
            n,
            self.d0.len()
        );

        let (l2, l1, d0, u1, u2) = (&self.l2, &self.l1, &self.d0, &self.u1, &self.u2);
        let mut x = vec![U::zero(); n];

        let mut al = vec![T::zero(); n - 1];
        let mut be = vec![T::zero(); n - 2];
        let mut ze = vec![U::zero(); n];
        let mut ga = vec![T::zero(); n];
        let mut mu = vec![T::zero(); n];

        mu[0] = d0[0];
        al[0] = u1[0] / mu[0];
        be[0] = u2[0] / mu[0];
        ze[0] = b[0] / mu[0];

        ga[1] = l1[0];
        mu[1] = d0[1] - al[0] * ga[1];
        al[1] = (u1[1] - be[0] * ga[1]) / mu[1];
        be[1] = u2[1] / mu[1];
        ze[1] = (b[1] - ze[0] * ga[1]) / mu[1];

        for i in 2..n - 2 {
            ga[i] = l1[i - 1] - al[i - 2] * l2[i - 2];
            mu[i] = d0[i] - be[i - 2] * l2[i - 2] - al[i - 1] * ga[i];
            al[i] = (u1[i] - be[i - 1] * ga[i]) / mu[i];
            be[i] = u2[i] / mu[i];
            ze[i] = (b[i] - ze[i - 2] * l2[i - 2] - ze[i - 1] * ga[i]) / mu[i];
        }

        ga[n - 2] = l1[n - 3] - al[n - 4] * l2[n - 4];
        mu[n - 2] = d0[n - 2] - be[n - 4] * l2[n - 4] - al[n - 3] * ga[n - 2];
        al[n - 2] = (u1[n - 2] - be[n - 3] * ga[n - 2]) / mu[n - 2];
        ze[n - 2] = (b[n - 2] - ze[n - 4] * l2[n - 4] - ze[n - 3] * ga[n - 2]) / mu[n - 2];

        ga[n - 1] = l1[n - 2] - al[n - 3] * l2[n - 3];
        mu[n - 1] = d0[n - 1] - be[n - 3] * l2[n - 3] - al[n - 2] * ga[n - 1];
        ze[n - 1] = (b[n - 1] - ze[n - 3] * l2[n - 3] - ze[n - 2] * ga[n - 1]) / mu[n - 1];

        // Backward substitution
        x[n - 1] = ze[n - 1];
        x[n - 2] = ze[n - 2] - x[n - 1] * al[n - 2];

        for i in (0..n - 2).rev() {
            x[i] = ze[i] - x[i + 1] * al[i] - x[i + 2] * be[i];
        }
        x
    }

    /// # Solve
    /// ```text
    /// A x = b
    ///```
    /// Returns x
    ///
    /// ## Panics
    /// - ``b.len() /= d0.len()``
    ///
    /// ## Note
    /// ** Unsafe ** due to unchecked bounds
    pub fn solve_unchecked<U>(&self, b: &[U]) -> Vec<U>
    where
        T: Zero + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
        U: Zero
            + Copy
            + Add<Output = U>
            + Sub<Output = U>
            + Mul<T, Output = U>
            + Div<T, Output = U>,
    {
        let n = b.len();
        assert!(
            n == self.d0.len(),
            "Shape mismatch. {} /= {}",
            n,
            self.d0.len()
        );

        let (l2, l1, d0, u1, u2) = (&self.l2, &self.l1, &self.d0, &self.u1, &self.u2);
        let mut x = vec![U::zero(); n];

        let mut al = vec![T::zero(); n - 1];
        let mut be = vec![T::zero(); n - 2];
        let mut ze = vec![U::zero(); n];
        let mut ga = vec![T::zero(); n];
        let mut mu = vec![T::zero(); n];

        unsafe {
            *mu.get_unchecked_mut(0) = *d0.get_unchecked(0);
            *al.get_unchecked_mut(0) = *u1.get_unchecked(0) / *mu.get_unchecked(0);
            *be.get_unchecked_mut(0) = *u2.get_unchecked(0) / *mu.get_unchecked(0);
            *ze.get_unchecked_mut(0) = *b.get_unchecked(0) / *mu.get_unchecked(0);

            *ga.get_unchecked_mut(1) = *l1.get_unchecked(0);
            *mu.get_unchecked_mut(1) =
                *d0.get_unchecked(1) - *al.get_unchecked(0) * *ga.get_unchecked(1);
            *al.get_unchecked_mut(1) = (*u1.get_unchecked(1)
                - *be.get_unchecked(0) * *ga.get_unchecked(1))
                / *mu.get_unchecked(1);
            *be.get_unchecked_mut(1) = *u2.get_unchecked(1) / *mu.get_unchecked(1);
            *ze.get_unchecked_mut(1) = (*b.get_unchecked(1)
                - *ze.get_unchecked(0) * *ga.get_unchecked(1))
                / *mu.get_unchecked(1);

            for i in 2..n - 2 {
                *ga.get_unchecked_mut(i) =
                    *l1.get_unchecked(i - 1) - *al.get_unchecked(i - 2) * *l2.get_unchecked(i - 2);
                *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                    - *be.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    - *al.get_unchecked(i - 1) * *ga.get_unchecked(i);
                *al.get_unchecked_mut(i) = (*u1.get_unchecked(i)
                    - *be.get_unchecked(i - 1) * *ga.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *be.get_unchecked_mut(i) = *u2.get_unchecked(i) / *mu.get_unchecked(i);
                *ze.get_unchecked_mut(i) = (*b.get_unchecked(i)
                    - *ze.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    - *ze.get_unchecked(i - 1) * *ga.get_unchecked(i))
                    / *mu.get_unchecked(i);
            }

            *ga.get_unchecked_mut(n - 2) =
                *l1.get_unchecked(n - 3) - *al.get_unchecked(n - 4) * *l2.get_unchecked(n - 4);
            *mu.get_unchecked_mut(n - 2) = *d0.get_unchecked(n - 2)
                - *be.get_unchecked(n - 4) * *l2.get_unchecked(n - 4)
                - *al.get_unchecked(n - 3) * *ga.get_unchecked(n - 2);
            *al.get_unchecked_mut(n - 2) = (*u1.get_unchecked(n - 2)
                - *be.get_unchecked(n - 3) * *ga.get_unchecked(n - 2))
                / *mu.get_unchecked(n - 2);
            *ze.get_unchecked_mut(n - 2) = (*b.get_unchecked(n - 2)
                - *ze.get_unchecked(n - 4) * *l2.get_unchecked(n - 4)
                - *ze.get_unchecked(n - 3) * *ga.get_unchecked(n - 2))
                / *mu.get_unchecked(n - 2);

            *ga.get_unchecked_mut(n - 1) =
                *l1.get_unchecked(n - 2) - *al.get_unchecked(n - 3) * *l2.get_unchecked(n - 3);
            *mu.get_unchecked_mut(n - 1) = *d0.get_unchecked(n - 1)
                - *be.get_unchecked(n - 3) * *l2.get_unchecked(n - 3)
                - *al.get_unchecked(n - 2) * *ga.get_unchecked(n - 1);
            *ze.get_unchecked_mut(n - 1) = (*b.get_unchecked(n - 1)
                - *ze.get_unchecked(n - 3) * *l2.get_unchecked(n - 3)
                - *ze.get_unchecked(n - 2) * *ga.get_unchecked(n - 1))
                / *mu.get_unchecked(n - 1);

            // Backward substitution
            *x.get_unchecked_mut(n - 1) = *ze.get_unchecked(n - 1);
            *x.get_unchecked_mut(n - 2) =
                *ze.get_unchecked(n - 2) - *x.get_unchecked(n - 1) * *al.get_unchecked(n - 2);

            for i in (0..n - 2).rev() {
                *x.get_unchecked_mut(i) = *ze.get_unchecked(i)
                    - *x.get_unchecked(i + 1) * *al.get_unchecked(i)
                    - *x.get_unchecked(i + 2) * *be.get_unchecked(i);
            }
        }
        x
    }
}

/// Elementwise multiplication with scalar
impl<'a, T> std::ops::Mul<T> for &'a Penta21012<T>
where
    T: std::ops::MulAssign + Copy,
{
    type Output = Penta21012<T>;

    fn mul(self, other: T) -> Self::Output {
        let mut new = self.clone();
        for x in &mut new.l2 {
            *x *= other;
        }
        for x in &mut new.l1 {
            *x *= other;
        }
        for x in &mut new.d0 {
            *x *= other;
        }
        for x in &mut new.u1 {
            *x *= other;
        }
        for x in &mut new.u2 {
            *x *= other;
        }
        new
    }
}

/// Addition : &Self + &Self
impl<'a, 'b, T> Add<&'b Penta21012<T>> for &'a Penta21012<T>
where
    T: std::ops::AddAssign + Copy,
{
    type Output = Penta21012<T>;

    fn add(self, other: &'b Penta21012<T>) -> Self::Output {
        assert!(self.d0.len() == other.d0.len(), "Size mismatch");
        let mut new = self.clone();
        for (x, y) in new.l2.iter_mut().zip(other.l2.iter()) {
            *x += *y;
        }
        for (x, y) in new.l1.iter_mut().zip(other.l1.iter()) {
            *x += *y;
        }
        for (x, y) in new.d0.iter_mut().zip(other.d0.iter()) {
            *x += *y;
        }
        for (x, y) in new.u1.iter_mut().zip(other.u1.iter()) {
            *x += *y;
        }
        for (x, y) in new.u2.iter_mut().zip(other.u2.iter()) {
            *x += *y;
        }
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    fn assert_approx_tol(x: &[f64], y: &[f64], tol: f64) {
        for (i, (xi, yi)) in x.iter().zip(y.iter()).enumerate() {
            assert!((xi - yi).abs() < tol, "{}: {} /= {}", i, xi, yi);
        }
    }

    #[test]
    fn test_penta_21012_dot() {
        let n = 21;
        let mut rng = thread_rng();
        // init
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let l1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let pdma = Penta21012::new(l2.clone(), l1.clone(), d0.clone(), u1.clone(), u2.clone());
        // Vector
        let x: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        // Solve
        let b_my = pdma.dot(&x);
        let b_nd = pdma.as_array2().dot(&ndarray::Array1::from_vec(x.clone()));
        assert_approx_tol(&b_my, &b_nd.to_vec(), 1e-6);
        let b_my = pdma.dot_unchecked(&x);
        assert_approx_tol(&b_my, &b_nd.to_vec(), 1e-6);
    }

    #[test]
    fn test_penta_21012_solve() {
        use ndarray_linalg::Solve;
        let n = 29;
        let mut rng = thread_rng();
        // init
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let l1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let pdma = Penta21012::new(l2.clone(), l1.clone(), d0.clone(), u1.clone(), u2.clone());
        // Vector
        let b: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        // Solve
        let x_my = pdma.solve(&b);
        let b_nd = ndarray::Array1::from_vec(b.clone());
        let x_nd = pdma.as_array2().solve_into(b_nd).unwrap();
        assert_approx_tol(&x_my, &x_nd.to_vec(), 1e-6);
        let x_my = pdma.solve_unchecked(&b);
        assert_approx_tol(&x_my, &x_nd.to_vec(), 1e-6);
    }
}
