//! Tetra-diagonal matrix solver
//!     Ax = b
//! where A is banded with diagonals in offsets -1, 0, 1, 2
#![allow(clippy::many_single_char_names)]
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Tetra1012<T> {
    /// Lower diagonal (-1)
    pub(crate) l1: Vec<T>,
    /// Main diagonal (0)
    pub(crate) d0: Vec<T>,
    /// Upper diagonal (+1)
    pub(crate) u1: Vec<T>,
    /// Upper diagonal (+2)
    pub(crate) u2: Vec<T>,
}

impl<T> Tetra1012<T> {
    /// Constructor
    ///
    /// ## Panics
    /// Shape mismatch of diagonals
    #[must_use]
    pub fn new(l1: Vec<T>, d0: Vec<T>, u1: Vec<T>, u2: Vec<T>) -> Self {
        assert!(d0.len() > 2, "Problem size too small.");
        assert!(l1.len() == d0.len() - 1);
        assert!(u1.len() == d0.len() - 1);
        assert!(u2.len() == d0.len() - 2);
        Self { l1, d0, u1, u2 }
    }

    /// Construct from matrix
    ///
    /// # Panics
    /// - Matrix is non-square
    /// - Matrix is non-zero on some diagonal other than -1, 0, 1, 2
    #[must_use]
    pub fn from_array2(array: &ndarray::Array2<T>) -> Self
    where
        T: Zero + Copy + PartialEq,
    {
        let diagonals = crate::utils::extract_diagonals(array, [-1, 0, 1, 2]);
        let diagonals = match diagonals {
            Ok(diagonals) => diagonals,
            Err(error) => panic!("{}", error.0),
        };
        let l1 = diagonals[0].clone();
        let d0 = diagonals[1].clone();
        let u1 = diagonals[2].clone();
        let u2 = diagonals[3].clone();
        Self::new(l1, d0, u1, u2)
    }

    /// Returns banded matrix
    #[must_use]
    pub fn as_array2(&self) -> ndarray::Array2<T>
    where
        T: Zero + Copy,
    {
        match crate::utils::array_from_diags(
            [&self.l1, &self.d0, &self.u1, &self.u2],
            [-1, 0, 1, 2],
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
        let (l1, d0, u1, u2) = (&self.l1, &self.d0, &self.u1, &self.u2);
        b[0] = x[0] * d0[0] + x[1] * u1[0] + x[2] * u2[0];
        for i in 1..n - 2 {
            b[i] = x[i] * d0[i] + x[i + 1] * u1[i] + x[i + 2] * u2[i] + x[i - 1] * l1[i - 1];
        }
        b[n - 2] = x[n - 2] * d0[n - 2] + x[n - 1] * u1[n - 2] + x[n - 3] * l1[n - 3];
        b[n - 1] = x[n - 1] * d0[n - 1] + x[n - 2] * l1[n - 2];
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
        let (l1, d0, u1, u2) = (&self.l1, &self.d0, &self.u1, &self.u2);
        unsafe {
            *b.get_unchecked_mut(0) = *x.get_unchecked(0) * *d0.get_unchecked(0)
                + *x.get_unchecked(1) * *u1.get_unchecked(0)
                + *x.get_unchecked(2) * *u2.get_unchecked(0);
            for i in 1..n - 2 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 1) * *u1.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i - 1) * *l1.get_unchecked(i - 1);
            }
            *b.get_unchecked_mut(n - 2) = *x.get_unchecked(n - 2) * *d0.get_unchecked(n - 2)
                + *x.get_unchecked(n - 1) * *u1.get_unchecked(n - 2)
                + *x.get_unchecked(n - 3) * *l1.get_unchecked(n - 3);
            *b.get_unchecked_mut(n - 1) = *x.get_unchecked(n - 1) * *d0.get_unchecked(n - 1)
                + *x.get_unchecked(n - 2) * *l1.get_unchecked(n - 2);
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

        let (l1, d0, u1, u2) = (&self.l1, &self.d0, &self.u1, &self.u2);
        let mut x = vec![U::zero(); n];

        let mut c = vec![T::zero(); n - 1];
        let mut d = vec![T::zero(); n - 2];

        // Forward sweep
        c[0] = u1[0] / d0[0];
        d[0] = u2[0] / d0[0];
        x[0] = b[0] / d0[0];

        for i in 1..n - 2 {
            d[i] = u2[i] / (d0[i] - l1[i - 1] * c[i - 1]);
            c[i] = (u1[i] - d[i - 1] * l1[i - 1]) / (d0[i] - l1[i - 1] * c[i - 1]);
        }
        c[n - 2] = (u1[n - 2] - d[n - 3] * l1[n - 3]) / (d0[n - 2] - l1[n - 3] * c[n - 3]);
        for i in 1..n {
            x[i] = (b[i] - x[i - 1] * l1[i - 1]) / (d0[i] - l1[i - 1] * c[i - 1]);
        }

        // Back substitution
        x[n - 2] = x[n - 2] - x[n - 1] * c[n - 2];
        for i in (1..n - 1).rev() {
            x[i - 1] = x[i - 1] - x[i] * c[i - 1] - x[i + 1] * d[i - 1];
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

        let (l1, d0, u1, u2) = (&self.l1, &self.d0, &self.u1, &self.u2);
        let mut x = vec![U::zero(); n];
        let mut c = vec![T::zero(); n - 1];
        let mut d = vec![T::zero(); n - 2];
        unsafe {
            // Forward sweep
            *c.get_unchecked_mut(0) = *u1.get_unchecked(0) / *d0.get_unchecked(0);
            *d.get_unchecked_mut(0) = *u2.get_unchecked(0) / *d0.get_unchecked(0);
            *x.get_unchecked_mut(0) = *b.get_unchecked(0) / *d0.get_unchecked(0);

            for i in 1..n - 2 {
                *d.get_unchecked_mut(i) = *u2.get_unchecked(i)
                    / (*d0.get_unchecked(i) - *l1.get_unchecked(i - 1) * *c.get_unchecked(i - 1));
                *c.get_unchecked_mut(i) = (*u1.get_unchecked(i)
                    - *d.get_unchecked(i - 1) * *l1.get_unchecked(i - 1))
                    / (*d0.get_unchecked(i) - *l1.get_unchecked(i - 1) * *c.get_unchecked(i - 1));
            }
            *c.get_unchecked_mut(n - 2) = (*u1.get_unchecked(n - 2)
                - *d.get_unchecked(n - 3) * *l1.get_unchecked(n - 3))
                / (*d0.get_unchecked(n - 2) - *l1.get_unchecked(n - 3) * *c.get_unchecked(n - 3));
            for i in 1..n {
                *x.get_unchecked_mut(i) = (*b.get_unchecked(i)
                    - *x.get_unchecked(i - 1) * *l1.get_unchecked(i - 1))
                    / (*d0.get_unchecked(i) - *l1.get_unchecked(i - 1) * *c.get_unchecked(i - 1));
            }
            // Back substitution
            *x.get_unchecked_mut(n - 2) =
                *x.get_unchecked(n - 2) - *x.get_unchecked(n - 1) * *c.get_unchecked(n - 2);
            for i in (1..n - 1).rev() {
                *x.get_unchecked_mut(i - 1) = *x.get_unchecked(i - 1)
                    - *x.get_unchecked(i) * *c.get_unchecked(i - 1)
                    - *x.get_unchecked(i + 1) * *d.get_unchecked(i - 1);
            }
        }
        x
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
    fn test_tetra_1012_dot() {
        let n = 21;
        let mut rng = thread_rng();
        // tdma
        let l1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let tdma = Tetra1012::new(l1.clone(), d0.clone(), u1.clone(), u2.clone());
        // Vector
        let x: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        // Solve
        let b_my = tdma.dot(&x);
        let b_nd = tdma.as_array2().dot(&ndarray::Array1::from_vec(x.clone()));
        assert_approx_tol(&b_my, &b_nd.to_vec(), 1e-6);
        let b_my = tdma.dot_unchecked(&x);
        assert_approx_tol(&b_my, &b_nd.to_vec(), 1e-6);
    }

    #[test]
    fn test_tetra_1012_solve() {
        use ndarray_linalg::Solve;
        let n = 22;
        let mut rng = thread_rng();
        // tdma
        let l1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let tdma = Tetra1012::new(l1.clone(), d0.clone(), u1.clone(), u2.clone());
        // Vector
        let b: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        // Solve
        let x_my = tdma.solve(&b);
        let b_nd = ndarray::Array1::from_vec(b.clone());
        let x_nd = tdma.as_array2().solve_into(b_nd).unwrap();
        assert_approx_tol(&x_my, &x_nd.to_vec(), 1e-6);
        let x_my = tdma.solve_unchecked(&b);
        assert_approx_tol(&x_my, &x_nd.to_vec(), 1e-6);
    }
}
