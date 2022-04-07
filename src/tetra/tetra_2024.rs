//! Tetra-diagonal matrix solver
//!     Ax = b
//! where A is banded with diagonals in offsets -2, 0, 2, 4
#![allow(clippy::many_single_char_names)]
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Tetra2024<T> {
    /// Lower diagonal (-2)
    pub(crate) l2: Vec<T>,
    /// Main diagonal (0)
    pub(crate) d0: Vec<T>,
    /// Upper diagonal (+2)
    pub(crate) u2: Vec<T>,
    /// Upper diagonal (+4)
    pub(crate) u4: Vec<T>,
}

impl<T> Tetra2024<T> {
    /// Constructor
    ///
    /// ## Panics
    /// Shape mismatch of diagonals
    #[must_use]
    pub fn new(l2: Vec<T>, d0: Vec<T>, u2: Vec<T>, u4: Vec<T>) -> Self {
        assert!(d0.len() > 5, "Problem size too small.");
        assert!(l2.len() == u2.len());
        assert!(u2.len() == d0.len() - 2);
        assert!(u4.len() == d0.len() - 4);
        Self { l2, d0, u2, u4 }
    }

    /// Construct from matrix
    ///
    /// # Panics
    /// - Matrix is non-square
    /// - Matrix is non-zero on some diagonal other than -2, 0, 2, 4
    #[must_use]
    pub fn from_array2(array: &ndarray::Array2<T>) -> Self
    where
        T: Zero + Copy + PartialEq,
    {
        let diagonals = crate::utils::extract_diagonals(array, [-2, 0, 2, 4]);
        let diagonals = match diagonals {
            Ok(diagonals) => diagonals,
            Err(error) => panic!("{}", error.0),
        };
        let l2 = diagonals[0].clone();
        let d0 = diagonals[1].clone();
        let u2 = diagonals[2].clone();
        let u4 = diagonals[3].clone();
        Self::new(l2, d0, u2, u4)
    }

    /// Returns banded matrix
    #[must_use]
    pub fn as_array2(&self) -> ndarray::Array2<T>
    where
        T: Zero + Copy,
    {
        match crate::utils::array_from_diags(
            [&self.l2, &self.d0, &self.u2, &self.u4],
            [-2, 0, 2, 4],
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
        let (l2, d0, u2, u4) = (&self.l2, &self.d0, &self.u2, &self.u4);
        b[0] = x[0] * d0[0] + x[2] * u2[0] + x[4] * u4[0];
        b[1] = x[1] * d0[1] + x[3] * u2[1] + x[5] * u4[1];
        for i in 2..n - 4 {
            b[i] = x[i] * d0[i] + x[i + 2] * u2[i] + x[i + 4] * u4[i] + x[i - 2] * l2[i - 2];
        }
        b[n - 4] = x[n - 4] * d0[n - 4] + x[n - 2] * u2[n - 4] + x[n - 6] * l2[n - 6];
        b[n - 3] = x[n - 3] * d0[n - 3] + x[n - 1] * u2[n - 3] + x[n - 5] * l2[n - 5];
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
        let (l2, d0, u2, u4) = (&self.l2, &self.d0, &self.u2, &self.u4);
        unsafe {
            *b.get_unchecked_mut(0) = *x.get_unchecked(0) * *d0.get_unchecked(0)
                + *x.get_unchecked(2) * *u2.get_unchecked(0)
                + *x.get_unchecked(4) * *u4.get_unchecked(0);
            *b.get_unchecked_mut(1) = *x.get_unchecked(1) * *d0.get_unchecked(1)
                + *x.get_unchecked(3) * *u2.get_unchecked(1)
                + *x.get_unchecked(5) * *u4.get_unchecked(1);
            for i in 2..n - 4 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i + 4) * *u4.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2);
            }
            *b.get_unchecked_mut(n - 4) = *x.get_unchecked(n - 4) * *d0.get_unchecked(n - 4)
                + *x.get_unchecked(n - 2) * *u2.get_unchecked(n - 4)
                + *x.get_unchecked(n - 6) * *l2.get_unchecked(n - 6);
            *b.get_unchecked_mut(n - 3) = *x.get_unchecked(n - 3) * *d0.get_unchecked(n - 3)
                + *x.get_unchecked(n - 1) * *u2.get_unchecked(n - 3)
                + *x.get_unchecked(n - 5) * *l2.get_unchecked(n - 5);
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
        assert!(
            n == self.d0.len(),
            "Shape mismatch. {} /= {}",
            n,
            self.d0.len()
        );

        let (l2, d0, u2, u4) = (&self.l2, &self.d0, &self.u2, &self.u4);
        let mut x = vec![U::zero(); n];

        let mut c = vec![T::zero(); n - 2];
        let mut d = vec![T::zero(); n - 4];

        // Forward sweep
        c[0] = u2[0] / d0[0];
        d[0] = u4[0] / d0[0];
        x[0] = b[0] / d0[0];

        c[1] = u2[1] / d0[1];
        d[1] = u4[1] / d0[1];
        x[1] = b[1] / d0[1];

        for i in 2..n - 4 {
            d[i] = u4[i] / (d0[i] - l2[i - 2] * c[i - 2]);
            c[i] = (u2[i] - d[i - 2] * l2[i - 2]) / (d0[i] - l2[i - 2] * c[i - 2]);
        }
        for i in n - 4..n - 2 {
            c[i] = (u2[i] - d[i - 2] * l2[i - 2]) / (d0[i] - l2[i - 2] * c[i - 2]);
        }
        for i in 2..n {
            x[i] = (b[i] - x[i - 2] * l2[i - 2]) / (d0[i] - l2[i - 2] * c[i - 2]);
        }

        // Back substitution
        x[n - 3] = x[n - 3] - x[n - 1] * c[n - 3];
        x[n - 4] = x[n - 4] - x[n - 2] * c[n - 4];
        for i in (1..n - 3).rev() {
            x[i - 1] = x[i - 1] - x[i + 1] * c[i - 1] - x[i + 3] * d[i - 1];
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

        let (l2, d0, u2, u4) = (&self.l2, &self.d0, &self.u2, &self.u4);
        let mut x = vec![U::zero(); n];
        let mut c = vec![T::zero(); n - 2];
        let mut d = vec![T::zero(); n - 4];
        unsafe {
            // Forward sweep
            *c.get_unchecked_mut(0) = *u2.get_unchecked(0) / *d0.get_unchecked(0);
            *d.get_unchecked_mut(0) = *u4.get_unchecked(0) / *d0.get_unchecked(0);
            *x.get_unchecked_mut(0) = *b.get_unchecked(0) / *d0.get_unchecked(0);

            *c.get_unchecked_mut(1) = *u2.get_unchecked(1) / *d0.get_unchecked(1);
            *d.get_unchecked_mut(1) = *u4.get_unchecked(1) / *d0.get_unchecked(1);
            *x.get_unchecked_mut(1) = *b.get_unchecked(1) / *d0.get_unchecked(1);

            for i in 2..n - 4 {
                *d.get_unchecked_mut(i) = *u4.get_unchecked(i)
                    / (*d0.get_unchecked(i) - *l2.get_unchecked(i - 2) * *c.get_unchecked(i - 2));
                *c.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                    - *d.get_unchecked(i - 2) * *l2.get_unchecked(i - 2))
                    / (*d0.get_unchecked(i) - *l2.get_unchecked(i - 2) * *c.get_unchecked(i - 2));
            }
            for i in n - 4..n - 2 {
                *c.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                    - *d.get_unchecked(i - 2) * *l2.get_unchecked(i - 2))
                    / (*d0.get_unchecked(i) - *l2.get_unchecked(i - 2) * *c.get_unchecked(i - 2));
            }
            for i in 2..n {
                *x.get_unchecked_mut(i) = (*b.get_unchecked(i)
                    - *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2))
                    / (*d0.get_unchecked(i) - *l2.get_unchecked(i - 2) * *c.get_unchecked(i - 2));
            }

            // Back substitution
            *x.get_unchecked_mut(n - 3) =
                *x.get_unchecked(n - 3) - *x.get_unchecked(n - 1) * *c.get_unchecked(n - 3);
            *x.get_unchecked_mut(n - 4) =
                *x.get_unchecked(n - 4) - *x.get_unchecked(n - 2) * *c.get_unchecked(n - 4);
            for i in (1..n - 3).rev() {
                *x.get_unchecked_mut(i - 1) = *x.get_unchecked(i - 1)
                    - *x.get_unchecked(i + 1) * *c.get_unchecked(i - 1)
                    - *x.get_unchecked(i + 3) * *d.get_unchecked(i - 1);
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
    fn test_tetra_2024_dot() {
        let n = 23;
        let mut rng = thread_rng();
        // tdma
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let tdma = Tetra2024::new(l2.clone(), d0.clone(), u2.clone(), u4.clone());
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
    fn test_tetra_2024_solve() {
        use ndarray_linalg::Solve;
        let n = 22;
        let mut rng = thread_rng();
        // tdma
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        // let u4: Vec<f64> = vec![0.; n-4];
        let tdma = Tetra2024::new(l2.clone(), d0.clone(), u2.clone(), u4.clone());
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
