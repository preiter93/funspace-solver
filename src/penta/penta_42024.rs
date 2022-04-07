//! Penta-diagonal matrix solver
//!     Ax = b
//! where A is banded with diagonals in offsets -4, -2, 0, 2, 4
#![allow(clippy::many_single_char_names)]
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Penta42024<T> {
    /// Lower diagonal (-4)
    pub(crate) l4: Vec<T>,
    /// Lower diagonal (-2)
    pub(crate) l2: Vec<T>,
    /// Main diagonal (0)
    pub(crate) d0: Vec<T>,
    /// Upper diagonal (+2)
    pub(crate) u2: Vec<T>,
    /// Upper diagonal (+4)
    pub(crate) u4: Vec<T>,
}

impl<T> Penta42024<T> {
    /// Constructor
    ///
    /// ## Panics
    /// Shape mismatch of diagonals
    #[must_use]
    pub fn new(l4: Vec<T>, l2: Vec<T>, d0: Vec<T>, u2: Vec<T>, u4: Vec<T>) -> Self {
        assert!(d0.len() > 5, "Problem size too small.");
        assert!(l2.len() == d0.len() - 2);
        assert!(u2.len() == d0.len() - 2);
        assert!(l4.len() == d0.len() - 4);
        assert!(u4.len() == d0.len() - 4);
        Self { l4, l2, d0, u2, u4 }
    }

    /// Construct from matrix
    ///
    /// # Panics
    /// - Matrix is non-square
    /// - Matrix is non-zero on some diagonal other than -4, -2, 0, 2, 4
    #[must_use]
    pub fn from_array2(array: &ndarray::Array2<T>) -> Self
    where
        T: Zero + Copy + PartialEq,
    {
        let diagonals = crate::utils::extract_diagonals(array, [-4, -2, 0, 2, 4]);
        let diagonals = match diagonals {
            Ok(diagonals) => diagonals,
            Err(error) => panic!("{}", error.0),
        };
        let l4 = diagonals[0].clone();
        let l2 = diagonals[1].clone();
        let d0 = diagonals[2].clone();
        let u2 = diagonals[3].clone();
        let u4 = diagonals[4].clone();
        Self::new(l4, l2, d0, u2, u4)
    }

    /// Returns banded matrix
    #[must_use]
    pub fn as_array2(&self) -> ndarray::Array2<T>
    where
        T: Zero + Copy,
    {
        match crate::utils::array_from_diags(
            [&self.l4, &self.l2, &self.d0, &self.u2, &self.u4],
            [-4, -2, 0, 2, 4],
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
        let (l4, l2, d0, u2, u4) = (&self.l4, &self.l2, &self.d0, &self.u2, &self.u4);
        b[0] = x[0] * d0[0] + x[2] * u2[0] + x[4] * u4[0];
        b[1] = x[1] * d0[1] + x[3] * u2[1] + x[5] * u4[1];
        b[2] = x[2] * d0[2] + x[4] * u2[2] + x[6] * u4[2] + x[0] * l2[0];
        b[3] = x[3] * d0[3] + x[5] * u2[3] + x[7] * u4[3] + x[1] * l2[1];
        for i in 4..n - 4 {
            b[i] = x[i] * d0[i]
                + x[i + 2] * u2[i]
                + x[i + 4] * u4[i]
                + x[i - 2] * l2[i - 2]
                + x[i - 4] * l4[i - 4];
        }
        b[n - 4] = x[n - 4] * d0[n - 4]
            + x[n - 2] * u2[n - 4]
            + x[n - 6] * l2[n - 6]
            + x[n - 8] * l4[n - 8];
        b[n - 3] = x[n - 3] * d0[n - 3]
            + x[n - 1] * u2[n - 3]
            + x[n - 5] * l2[n - 5]
            + x[n - 7] * l4[n - 7];
        b[n - 2] = x[n - 2] * d0[n - 2] + x[n - 4] * l2[n - 4] + x[n - 6] * l4[n - 6];
        b[n - 1] = x[n - 1] * d0[n - 1] + x[n - 3] * l2[n - 3] + x[n - 5] * l4[n - 5];
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
        let (l4, l2, d0, u2, u4) = (&self.l4, &self.l2, &self.d0, &self.u2, &self.u4);
        unsafe {
            *b.get_unchecked_mut(0) = *x.get_unchecked(0) * *d0.get_unchecked(0)
                + *x.get_unchecked(2) * *u2.get_unchecked(0)
                + *x.get_unchecked(4) * *u4.get_unchecked(0);
            *b.get_unchecked_mut(1) = *x.get_unchecked(1) * *d0.get_unchecked(1)
                + *x.get_unchecked(3) * *u2.get_unchecked(1)
                + *x.get_unchecked(5) * *u4.get_unchecked(1);

            *b.get_unchecked_mut(2) = *x.get_unchecked(2) * *d0.get_unchecked(2)
                + *x.get_unchecked(4) * *u2.get_unchecked(2)
                + *x.get_unchecked(6) * *u4.get_unchecked(2)
                + *x.get_unchecked(0) * *l2.get_unchecked(0);

            *b.get_unchecked_mut(3) = *x.get_unchecked(3) * *d0.get_unchecked(3)
                + *x.get_unchecked(5) * *u2.get_unchecked(3)
                + *x.get_unchecked(7) * *u4.get_unchecked(3)
                + *x.get_unchecked(1) * *l2.get_unchecked(1);

            for i in 4..n - 4 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i + 4) * *u4.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    + *x.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
            }
            *b.get_unchecked_mut(n - 4) = *x.get_unchecked(n - 4) * *d0.get_unchecked(n - 4)
                + *x.get_unchecked(n - 2) * *u2.get_unchecked(n - 4)
                + *x.get_unchecked(n - 6) * *l2.get_unchecked(n - 6)
                + *x.get_unchecked(n - 8) * *l4.get_unchecked(n - 8);
            *b.get_unchecked_mut(n - 3) = *x.get_unchecked(n - 3) * *d0.get_unchecked(n - 3)
                + *x.get_unchecked(n - 1) * *u2.get_unchecked(n - 3)
                + *x.get_unchecked(n - 5) * *l2.get_unchecked(n - 5)
                + *x.get_unchecked(n - 7) * *l4.get_unchecked(n - 7);
            *b.get_unchecked_mut(n - 2) = *x.get_unchecked(n - 2) * *d0.get_unchecked(n - 2)
                + *x.get_unchecked(n - 4) * *l2.get_unchecked(n - 4)
                + *x.get_unchecked(n - 6) * *l4.get_unchecked(n - 6);
            *b.get_unchecked_mut(n - 1) = *x.get_unchecked(n - 1) * *d0.get_unchecked(n - 1)
                + *x.get_unchecked(n - 3) * *l2.get_unchecked(n - 3)
                + *x.get_unchecked(n - 5) * *l4.get_unchecked(n - 5);
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
    #[allow(clippy::too_many_lines)]
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

        let (l4, l2, d0, u2, u4) = (&self.l4, &self.l2, &self.d0, &self.u2, &self.u4);
        let mut x = vec![U::zero(); n];
        let mut al = vec![T::zero(); n - 2];
        let mut be = vec![T::zero(); n - 4];
        let mut ze = vec![U::zero(); n];
        let mut ga = vec![T::zero(); n];
        let mut mu = vec![T::zero(); n];

        mu[0] = d0[0];
        al[0] = u2[0] / mu[0];
        be[0] = u4[0] / mu[0];
        ze[0] = b[0] / mu[0];

        mu[1] = d0[1];
        al[1] = u2[1] / mu[1];
        be[1] = u4[1] / mu[1];
        ze[1] = b[1] / mu[1];

        ga[2] = l2[0];
        mu[2] = d0[2] - al[0] * ga[2];
        al[2] = (u2[2] - be[0] * ga[2]) / mu[2];
        if u4.len() > 2 {
            be[2] = u4[2] / mu[2];
        }
        ze[2] = (b[2] - ze[0] * ga[2]) / mu[2];

        ga[3] = l2[1];
        mu[3] = d0[3] - al[1] * ga[3];
        al[3] = (u2[3] - be[1] * ga[3]) / mu[3];
        if u4.len() > 3 {
            be[3] = u4[3] / mu[3];
        }
        ze[3] = (b[3] - ze[1] * ga[3]) / mu[3];

        for i in 4..n - 4 {
            ga[i] = l2[i - 2] - al[i - 4] * l4[i - 4];
            mu[i] = d0[i] - be[i - 4] * l4[i - 4] - al[i - 2] * ga[i];
            al[i] = (u2[i] - be[i - 2] * ga[i]) / mu[i];
            be[i] = u4[i] / mu[i];
            ze[i] = (b[i] - ze[i - 4] * l4[i - 4] - ze[i - 2] * ga[i]) / mu[i];
        }
        if l4.len() > 3 {
            ga[n - 4] = l2[n - 6] - al[n - 8] * l4[n - 8];
            mu[n - 4] = d0[n - 4] - be[n - 8] * l4[n - 8] - al[n - 6] * ga[n - 4];
            al[n - 4] = (u2[n - 4] - be[n - 6] * ga[n - 4]) / mu[n - 4];
            ze[n - 4] = (b[n - 4] - ze[n - 8] * l4[n - 8] - ze[n - 6] * ga[n - 4]) / mu[n - 4];
        }
        if l4.len() > 2 {
            ga[n - 3] = l2[n - 5] - al[n - 7] * l4[n - 7];
            mu[n - 3] = d0[n - 3] - be[n - 7] * l4[n - 7] - al[n - 5] * ga[n - 3];
            al[n - 3] = (u2[n - 3] - be[n - 5] * ga[n - 3]) / mu[n - 3];
            ze[n - 3] = (b[n - 3] - ze[n - 7] * l4[n - 7] - ze[n - 5] * ga[n - 3]) / mu[n - 3];
        }

        ga[n - 2] = l2[n - 4] - al[n - 6] * l4[n - 6];
        mu[n - 2] = d0[n - 2] - be[n - 6] * l4[n - 6] - al[n - 4] * ga[n - 2];
        ze[n - 2] = (b[n - 2] - ze[n - 6] * l4[n - 6] - ze[n - 4] * ga[n - 2]) / mu[n - 2];

        ga[n - 1] = l2[n - 3] - al[n - 5] * l4[n - 5];
        mu[n - 1] = d0[n - 1] - be[n - 5] * l4[n - 5] - al[n - 3] * ga[n - 1];
        ze[n - 1] = (b[n - 1] - ze[n - 5] * l4[n - 5] - ze[n - 3] * ga[n - 1]) / mu[n - 1];

        // Backward substitution
        x[n - 1] = ze[n - 1];
        x[n - 2] = ze[n - 2];
        x[n - 3] = ze[n - 3] - x[n - 1] * al[n - 3];
        x[n - 4] = ze[n - 4] - x[n - 2] * al[n - 4];

        for i in (0..n - 4).rev() {
            x[i] = ze[i] - x[i + 2] * al[i] - x[i + 4] * be[i];
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
    #[allow(clippy::too_many_lines)]
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

        let (l4, l2, d0, u2, u4) = (&self.l4, &self.l2, &self.d0, &self.u2, &self.u4);
        let mut x = vec![U::zero(); n];
        let mut al = vec![T::zero(); n - 2];
        let mut be = vec![T::zero(); n - 4];
        let mut ze = vec![U::zero(); n];
        let mut ga = vec![T::zero(); n];
        let mut mu = vec![T::zero(); n];
        unsafe {
            *mu.get_unchecked_mut(0) = *d0.get_unchecked(0);
            *al.get_unchecked_mut(0) = *u2.get_unchecked(0) / *mu.get_unchecked(0);
            *be.get_unchecked_mut(0) = *u4.get_unchecked(0) / *mu.get_unchecked(0);
            *ze.get_unchecked_mut(0) = *b.get_unchecked(0) / *mu.get_unchecked(0);

            *mu.get_unchecked_mut(1) = *d0.get_unchecked(1);
            *al.get_unchecked_mut(1) = *u2.get_unchecked(1) / *mu.get_unchecked(1);
            *be.get_unchecked_mut(1) = *u4.get_unchecked(1) / *mu.get_unchecked(1);
            *ze.get_unchecked_mut(1) = *b.get_unchecked(1) / *mu.get_unchecked(1);

            *ga.get_unchecked_mut(2) = *l2.get_unchecked(0);
            *mu.get_unchecked_mut(2) =
                *d0.get_unchecked(2) - *al.get_unchecked(0) * *ga.get_unchecked(2);
            *al.get_unchecked_mut(2) = (*u2.get_unchecked(2)
                - *be.get_unchecked(0) * *ga.get_unchecked(2))
                / *mu.get_unchecked(2);
            if u4.len() > 2 {
                *be.get_unchecked_mut(2) = *u4.get_unchecked(2) / *mu.get_unchecked(2);
            }
            *ze.get_unchecked_mut(2) = (*b.get_unchecked(2)
                - *ze.get_unchecked(0) * *ga.get_unchecked(2))
                / *mu.get_unchecked(2);

            *ga.get_unchecked_mut(3) = *l2.get_unchecked(1);
            *mu.get_unchecked_mut(3) =
                *d0.get_unchecked(3) - *al.get_unchecked(1) * *ga.get_unchecked(3);
            *al.get_unchecked_mut(3) = (*u2.get_unchecked(3)
                - *be.get_unchecked(1) * *ga.get_unchecked(3))
                / *mu.get_unchecked(3);
            if u4.len() > 3 {
                *be.get_unchecked_mut(3) = *u4.get_unchecked(3) / *mu.get_unchecked(3);
            }
            *ze.get_unchecked_mut(3) = (*b.get_unchecked(3)
                - *ze.get_unchecked(1) * *ga.get_unchecked(3))
                / *mu.get_unchecked(3);

            for i in 4..n - 4 {
                *ga.get_unchecked_mut(i) =
                    *l2.get_unchecked(i - 2) - *al.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
                *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                    - *be.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *al.get_unchecked(i - 2) * *ga.get_unchecked(i);
                *al.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                    - *be.get_unchecked(i - 2) * *ga.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *be.get_unchecked_mut(i) = *u4.get_unchecked(i) / *mu.get_unchecked(i);
                *ze.get_unchecked_mut(i) = (*b.get_unchecked(i)
                    - *ze.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *ze.get_unchecked(i - 2) * *ga.get_unchecked(i))
                    / *mu.get_unchecked(i);
            }
            if l4.len() > 3 {
                *ga.get_unchecked_mut(n - 4) =
                    *l2.get_unchecked(n - 6) - *al.get_unchecked(n - 8) * *l4.get_unchecked(n - 8);
                *mu.get_unchecked_mut(n - 4) = *d0.get_unchecked(n - 4)
                    - *be.get_unchecked(n - 8) * *l4.get_unchecked(n - 8)
                    - *al.get_unchecked(n - 6) * *ga.get_unchecked(n - 4);
                *al.get_unchecked_mut(n - 4) = (*u2.get_unchecked(n - 4)
                    - *be.get_unchecked(n - 6) * *ga.get_unchecked(n - 4))
                    / *mu.get_unchecked(n - 4);
                *ze.get_unchecked_mut(n - 4) = (*b.get_unchecked(n - 4)
                    - *ze.get_unchecked(n - 8) * *l4.get_unchecked(n - 8)
                    - *ze.get_unchecked(n - 6) * *ga.get_unchecked(n - 4))
                    / *mu.get_unchecked(n - 4);
            }
            if l4.len() > 2 {
                *ga.get_unchecked_mut(n - 3) =
                    *l2.get_unchecked(n - 5) - *al.get_unchecked(n - 7) * *l4.get_unchecked(n - 7);
                *mu.get_unchecked_mut(n - 3) = *d0.get_unchecked(n - 3)
                    - *be.get_unchecked(n - 7) * *l4.get_unchecked(n - 7)
                    - *al.get_unchecked(n - 5) * *ga.get_unchecked(n - 3);
                *al.get_unchecked_mut(n - 3) = (*u2.get_unchecked(n - 3)
                    - *be.get_unchecked(n - 5) * *ga.get_unchecked(n - 3))
                    / *mu.get_unchecked(n - 3);
                *ze.get_unchecked_mut(n - 3) = (*b.get_unchecked(n - 3)
                    - *ze.get_unchecked(n - 7) * *l4.get_unchecked(n - 7)
                    - *ze.get_unchecked(n - 5) * *ga.get_unchecked(n - 3))
                    / *mu.get_unchecked(n - 3);
            }
            *ga.get_unchecked_mut(n - 2) =
                *l2.get_unchecked(n - 4) - *al.get_unchecked(n - 6) * *l4.get_unchecked(n - 6);
            *mu.get_unchecked_mut(n - 2) = *d0.get_unchecked(n - 2)
                - *be.get_unchecked(n - 6) * *l4.get_unchecked(n - 6)
                - *al.get_unchecked(n - 4) * *ga.get_unchecked(n - 2);
            *ze.get_unchecked_mut(n - 2) = (*b.get_unchecked(n - 2)
                - *ze.get_unchecked(n - 6) * *l4.get_unchecked(n - 6)
                - *ze.get_unchecked(n - 4) * *ga.get_unchecked(n - 2))
                / *mu.get_unchecked(n - 2);

            *ga.get_unchecked_mut(n - 1) =
                *l2.get_unchecked(n - 3) - *al.get_unchecked(n - 5) * *l4.get_unchecked(n - 5);
            *mu.get_unchecked_mut(n - 1) = *d0.get_unchecked(n - 1)
                - *be.get_unchecked(n - 5) * *l4.get_unchecked(n - 5)
                - *al.get_unchecked(n - 3) * *ga.get_unchecked(n - 1);

            *ze.get_unchecked_mut(n - 1) = (*b.get_unchecked(n - 1)
                - *ze.get_unchecked(n - 5) * *l4.get_unchecked(n - 5)
                - *ze.get_unchecked(n - 3) * *ga.get_unchecked(n - 1))
                / *mu.get_unchecked(n - 1);

            // Backward substitution
            *x.get_unchecked_mut(n - 1) = *ze.get_unchecked(n - 1);
            *x.get_unchecked_mut(n - 2) = *ze.get_unchecked(n - 2);
            *x.get_unchecked_mut(n - 3) =
                *ze.get_unchecked(n - 3) - *x.get_unchecked(n - 1) * *al.get_unchecked(n - 3);
            *x.get_unchecked_mut(n - 4) =
                *ze.get_unchecked(n - 4) - *x.get_unchecked(n - 2) * *al.get_unchecked(n - 4);

            for i in (0..n - 4).rev() {
                *x.get_unchecked_mut(i) = *ze.get_unchecked(i)
                    - *x.get_unchecked(i + 2) * *al.get_unchecked(i)
                    - *x.get_unchecked(i + 4) * *be.get_unchecked(i);
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
    fn test_penta_42024_dot() {
        let n = 23;
        let mut rng = thread_rng();
        // init
        let l4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let dma = Penta42024::new(l4, l2, d0, u2, u4);
        // Vector
        let x: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        // Solve
        let b_my = dma.dot(&x);
        let b_nd = dma.as_array2().dot(&ndarray::Array1::from_vec(x.clone()));
        assert_approx_tol(&b_my, &b_nd.to_vec(), 1e-6);
        let b_my = dma.dot_unchecked(&x);
        assert_approx_tol(&b_my, &b_nd.to_vec(), 1e-6);
    }

    #[test]
    fn test_penta_42024_solve() {
        use ndarray_linalg::Solve;
        let n = 6;
        let mut rng = thread_rng();
        // init
        let l4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let dma = Penta42024::new(l4, l2, d0, u2, u4);
        // Vector
        let b: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        // Solve
        let x_my = dma.solve(&b);
        let b_nd = ndarray::Array1::from_vec(b.clone());
        let x_nd = dma.as_array2().solve_into(b_nd).unwrap();
        assert_approx_tol(&x_my, &x_nd.to_vec(), 1e-6);
        let x_my = dma.solve_unchecked(&b);
        assert_approx_tol(&x_my, &x_nd.to_vec(), 1e-6);
    }
}
