//! Tridiagonal matrix solver
//!     Ax = b
//! where A is banded with diagonals in offsets -2, 0, 2
#![allow(clippy::many_single_char_names)]
use num_traits::Zero;
use std::clone::Clone;
use std::cmp::PartialEq;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Tri202<T> {
    /// Lower diagonal (-2)
    pub(crate) l2: Vec<T>,
    /// Main diagonal (0)
    pub(crate) d0: Vec<T>,
    /// Upper diagonal (+2)
    pub(crate) u2: Vec<T>,
}

impl<T> Tri202<T> {
    /// Sizes must be:
    /// ``l2.len() == u2.len() == d0.len() - 2``
    ///
    /// ## Panics
    /// Shape mismatch of diagonals
    #[must_use]
    pub fn new(l2: Vec<T>, d0: Vec<T>, u2: Vec<T>) -> Self {
        assert!(d0.len() > 3, "Problem size too small.");
        assert!(l2.len() == u2.len());
        assert!(l2.len() == d0.len() - 2);
        Self { l2, d0, u2 }
    }

    /// Construct from matrix
    ///
    /// # Panics
    /// - Matrix is non-square
    /// - Matrix is non-zero on some diagonal other than -2, 0, 2
    #[must_use]
    pub fn from_array2(array: &ndarray::Array2<T>) -> Self
    where
        T: Zero + Copy + PartialEq,
    {
        let diagonals = crate::utils::extract_diagonals(array, [-2, 0, 2]);
        let diagonals = match diagonals {
            Ok(diagonals) => diagonals,
            Err(error) => panic!("{}", error.0),
        };
        let l2 = diagonals[0].clone();
        let d0 = diagonals[1].clone();
        let u2 = diagonals[2].clone();
        Self::new(l2, d0, u2)
    }

    /// Returns tridiagonal banded matrix
    #[must_use]
    pub fn as_array2(&self) -> ndarray::Array2<T>
    where
        T: Zero + Copy,
    {
        match crate::utils::array_from_diags(
            [&self.l2, &self.d0, &self.u2],
            [-2, 0, 2],
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
    /// ``x.len() /= d0.len()``
    pub fn dot<U>(&self, x: &[U]) -> Vec<U>
    where
        T: Copy,
        U: Zero + Copy + Add<Output = U> + Mul<T, Output = U>,
    {
        assert!(x.len() == self.d0.len(), "Shape mismatch.");
        let n = x.len();
        let mut b = vec![U::zero(); n];
        let (l2, d0, u2) = (&self.l2, &self.d0, &self.u2);
        b[0] = x[0] * d0[0] + x[2] * u2[0];
        b[1] = x[1] * d0[1] + x[3] * u2[1];
        for i in 2..n - 2 {
            b[i] = x[i] * d0[i] + x[i + 2] * u2[i] + x[i - 2] * l2[i - 2];
        }
        b[n - 2] = x[n - 2] * d0[n - 2] + x[n - 4] * l2[n - 4];
        b[n - 1] = x[n - 1] * d0[n - 1] + x[n - 3] * l2[n - 3];
        b
    }

    /// # Dot product
    /// ```text
    /// A x = b
    ///```
    /// Returns b
    ///
    /// ## Panics
    /// ``x.len() /= d0.len()``
    ///
    /// ## Note
    /// ** Unsafe ** due to unchecked bounds
    pub fn dot_unchecked<U>(&self, x: &[U]) -> Vec<U>
    where
        T: Copy,
        U: Zero + Copy + Add<Output = U> + Mul<T, Output = U>,
    {
        assert!(x.len() == self.d0.len(), "Shape mismatch.");
        let n = x.len();
        let mut b = vec![U::zero(); n];
        let (l2, d0, u2) = (&self.l2, &self.d0, &self.u2);
        unsafe {
            *b.get_unchecked_mut(0) = *x.get_unchecked(0) * *d0.get_unchecked(0)
                + *x.get_unchecked(2) * *u2.get_unchecked(0);
            *b.get_unchecked_mut(1) = *x.get_unchecked(1) * *d0.get_unchecked(1)
                + *x.get_unchecked(3) * *u2.get_unchecked(1);
            for i in 2..n - 2 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2);
            }
            *b.get_unchecked_mut(n - 2) = *x.get_unchecked(n - 2) * *d0.get_unchecked(n - 2)
                + *x.get_unchecked(n - 4) * *l2.get_unchecked(n - 4);
            *b.get_unchecked_mut(n - 1) = *x.get_unchecked(n - 1) * *d0.get_unchecked(n - 1)
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
        assert!(n == self.d0.len(), "Shape mismatch.");

        let (l2, d0, u2) = (&self.l2, &self.d0, &self.u2);
        let mut x = vec![U::zero(); n];
        let mut w = vec![T::zero(); n - 2];
        // Forward sweep
        w[0] = u2[0] / d0[0];
        x[0] = b[0] / d0[0];

        w[1] = u2[1] / d0[1];
        x[1] = b[1] / d0[1];

        for i in 2..n - 2 {
            w[i] = u2[i] / (d0[i] - l2[i - 2] * w[i - 2]);
        }
        for i in 2..n {
            x[i] = (b[i] - x[i - 2] * l2[i - 2]) / (d0[i] - l2[i - 2] * w[i - 2]);
        }

        // Back substitution
        for i in (1..n - 1).rev() {
            x[i - 1] = x[i - 1] - x[i + 1] * w[i - 1];
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
        assert!(n == self.d0.len(), "Shape mismatch.");

        let (u2, d0, l2) = (&self.u2, &self.d0, &self.l2);
        let mut x = vec![U::zero(); n];
        let mut w = vec![T::zero(); n - 2];
        unsafe {
            // Forward sweep
            *w.get_unchecked_mut(0) = *u2.get_unchecked(0) / *d0.get_unchecked(0);
            *x.get_unchecked_mut(0) = *b.get_unchecked(0) / *d0.get_unchecked(0);

            *w.get_unchecked_mut(1) = *u2.get_unchecked(1) / *d0.get_unchecked(1);
            *x.get_unchecked_mut(1) = *b.get_unchecked(1) / *d0.get_unchecked(1);

            for i in 2..n - 2 {
                *w.get_unchecked_mut(i) = *u2.get_unchecked(i)
                    / (*d0.get_unchecked(i) - *l2.get_unchecked(i - 2) * *w.get_unchecked(i - 2));
            }
            for i in 2..n {
                *x.get_unchecked_mut(i) = (*b.get_unchecked(i)
                    - *x.get_unchecked_mut(i - 2) * *l2.get_unchecked(i - 2))
                    / (*d0.get_unchecked(i) - *l2.get_unchecked(i - 2) * *w.get_unchecked(i - 2));
            }

            // Back substitution
            for i in (1..n - 1).rev() {
                *x.get_unchecked_mut(i - 1) = *x.get_unchecked_mut(i - 1)
                    - *x.get_unchecked_mut(i + 1) * *w.get_unchecked(i - 1);
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
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert!((xi - yi).abs() < tol);
        }
    }

    #[test]
    fn test_tri_202_dot() {
        let n = 20;
        let mut rng = thread_rng();
        // Tdma
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let tdma = Tri202::new(l2.clone(), d0.clone(), u2.clone());
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
    fn test_tri_202_solve() {
        use ndarray_linalg::Solve;

        let n = 20;
        let mut rng = thread_rng();
        // Tdma
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let tdma = Tri202::new(l2.clone(), d0.clone(), u2.clone());
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
