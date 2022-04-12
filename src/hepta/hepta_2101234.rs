//! Hepta-diagonal matrix solver
//!     Ax = b
//! where A is banded with diagonals in offsets -2, -1, 0, 1, 2, 3, 4
#![allow(clippy::many_single_char_names)]
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Hepta2101234<T> {
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
    /// Upper diagonal (+3)
    pub(crate) u3: Vec<T>,
    /// Upper diagonal (+4)
    pub(crate) u4: Vec<T>,
}

impl<T> Hepta2101234<T> {
    /// Constructor
    ///
    /// ## Panics
    /// Shape mismatch of diagonals
    #[must_use]
    pub fn new(
        l2: Vec<T>,
        l1: Vec<T>,
        d0: Vec<T>,
        u1: Vec<T>,
        u2: Vec<T>,
        u3: Vec<T>,
        u4: Vec<T>,
    ) -> Self {
        assert!(d0.len() > 5, "Problem size too small.");
        assert!(l2.len() == d0.len() - 2);
        assert!(l1.len() == d0.len() - 1);
        assert!(u1.len() == d0.len() - 1);
        assert!(u2.len() == d0.len() - 2);
        assert!(u3.len() == d0.len() - 3);
        assert!(u4.len() == d0.len() - 4);
        Self {
            l2,
            l1,
            d0,
            u1,
            u2,
            u3,
            u4,
        }
    }

    /// Construct from matrix
    ///
    /// # Panics
    /// - Matrix is non-square
    /// - Matrix is non-zero on some diagonal other than -2, -1, 0, 1, 2, 3, 4
    #[must_use]
    pub fn from_array2(array: &ndarray::Array2<T>) -> Self
    where
        T: Zero + Copy + PartialEq,
    {
        let diagonals = crate::utils::extract_diagonals(array, [-2, -1, 0, 1, 2, 3, 4]);
        let diagonals = match diagonals {
            Ok(diagonals) => diagonals,
            Err(error) => panic!("{}", error.0),
        };
        let l2 = diagonals[0].clone();
        let l1 = diagonals[1].clone();
        let d0 = diagonals[2].clone();
        let u1 = diagonals[3].clone();
        let u2 = diagonals[4].clone();
        let u3 = diagonals[5].clone();
        let u4 = diagonals[6].clone();
        Self::new(l2, l1, d0, u1, u2, u3, u4)
    }

    /// Returns banded matrix
    #[must_use]
    pub fn as_array2(&self) -> ndarray::Array2<T>
    where
        T: Zero + Copy,
    {
        match crate::utils::array_from_diags(
            [
                &self.l2, &self.l1, &self.d0, &self.u1, &self.u2, &self.u3, &self.u4,
            ],
            [-2, -1, 0, 1, 2, 3, 4],
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
        let (l2, l1, d0, u1, u2, u3, u4) = (
            &self.l2, &self.l1, &self.d0, &self.u1, &self.u2, &self.u3, &self.u4,
        );
        b[0] = x[0] * d0[0] + x[1] * u1[0] + x[2] * u2[0] + x[3] * u3[0] + x[4] * u4[0];
        b[1] =
            x[1] * d0[1] + x[2] * u1[1] + x[3] * u2[1] + x[4] * u3[1] + x[5] * u4[1] + x[0] * l1[0];
        for i in 2..n - 4 {
            b[i] = x[i] * d0[i]
                + x[i + 1] * u1[i]
                + x[i + 2] * u2[i]
                + x[i + 3] * u3[i]
                + x[i + 4] * u4[i]
                + x[i - 1] * l1[i - 1]
                + x[i - 2] * l2[i - 2];
        }
        b[n - 4] = x[n - 4] * d0[n - 4]
            + x[n - 3] * u1[n - 4]
            + x[n - 2] * u2[n - 4]
            + x[n - 1] * u3[n - 4]
            + x[n - 5] * l1[n - 5]
            + x[n - 6] * l2[n - 6];
        b[n - 3] = x[n - 3] * d0[n - 3]
            + x[n - 2] * u1[n - 3]
            + x[n - 1] * u2[n - 3]
            + x[n - 4] * l1[n - 4]
            + x[n - 5] * l2[n - 5];
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
        let (l2, l1, d0, u1, u2, u3, u4) = (
            &self.l2, &self.l1, &self.d0, &self.u1, &self.u2, &self.u3, &self.u4,
        );
        unsafe {
            *b.get_unchecked_mut(0) = *x.get_unchecked(0) * *d0.get_unchecked(0)
                + *x.get_unchecked(1) * *u1.get_unchecked(0)
                + *x.get_unchecked(2) * *u2.get_unchecked(0)
                + *x.get_unchecked(3) * *u3.get_unchecked(0)
                + *x.get_unchecked(4) * *u4.get_unchecked(0);
            *b.get_unchecked_mut(1) = *x.get_unchecked(1) * *d0.get_unchecked(1)
                + *x.get_unchecked(2) * *u1.get_unchecked(1)
                + *x.get_unchecked(3) * *u2.get_unchecked(1)
                + *x.get_unchecked(4) * *u3.get_unchecked(1)
                + *x.get_unchecked(5) * *u4.get_unchecked(1)
                + *x.get_unchecked(0) * *l1.get_unchecked(0);
            for i in 2..n - 4 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 1) * *u1.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i + 3) * *u3.get_unchecked(i)
                    + *x.get_unchecked(i + 4) * *u4.get_unchecked(i)
                    + *x.get_unchecked(i - 1) * *l1.get_unchecked(i - 1)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2);
            }
            *b.get_unchecked_mut(n - 4) = *x.get_unchecked(n - 4) * *d0.get_unchecked(n - 4)
                + *x.get_unchecked(n - 3) * *u1.get_unchecked(n - 4)
                + *x.get_unchecked(n - 2) * *u2.get_unchecked(n - 4)
                + *x.get_unchecked(n - 1) * *u3.get_unchecked(n - 4)
                + *x.get_unchecked(n - 5) * *l1.get_unchecked(n - 5)
                + *x.get_unchecked(n - 6) * *l2.get_unchecked(n - 6);
            *b.get_unchecked_mut(n - 3) = *x.get_unchecked(n - 3) * *d0.get_unchecked(n - 3)
                + *x.get_unchecked(n - 2) * *u1.get_unchecked(n - 3)
                + *x.get_unchecked(n - 1) * *u2.get_unchecked(n - 3)
                + *x.get_unchecked(n - 4) * *l1.get_unchecked(n - 4)
                + *x.get_unchecked(n - 5) * *l2.get_unchecked(n - 5);
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

        let (l2, l1, d0, u1, u2, u3, u4) = (
            &self.l2, &self.l1, &self.d0, &self.u1, &self.u2, &self.u3, &self.u4,
        );
        // Allocate arrays
        let mut al = vec![T::zero(); n - 1];
        let mut be = vec![T::zero(); n - 2];
        let mut ga = vec![T::zero(); n - 3];
        let mut de = vec![T::zero(); n - 4];
        let mut ka = vec![T::zero(); n];
        let mut mu = vec![T::zero(); n];
        let mut x = vec![U::zero(); n];

        // Precompute sweep
        mu[0] = d0[0];
        al[0] = u1[0] / mu[0];
        be[0] = u2[0] / mu[0];
        ga[0] = u3[0] / mu[0];
        de[0] = u4[0] / mu[0];

        ka[1] = l1[0];
        mu[1] = d0[1] - al[0] * ka[1];
        al[1] = (u1[1] - be[0] * ka[1]) / mu[1];
        be[1] = (u2[1] - ga[0] * ka[1]) / mu[1];
        ga[1] = (u3[1] - de[0] * ka[1]) / mu[1];
        de[1] = u4[1] / mu[1];

        for i in 2..n - 4 {
            ka[i] = l1[i - 1] - al[i - 2] * l2[i - 2];
            mu[i] = d0[i] - be[i - 2] * l2[i - 2] - al[i - 1] * ka[i];
            al[i] = (u1[i] - ga[i - 2] * l2[i - 2] - be[i - 1] * ka[i]) / mu[i];
            be[i] = (u2[i] - de[i - 2] * l2[i - 2] - ga[i - 1] * ka[i]) / mu[i];
            ga[i] = (u3[i] - de[i - 1] * ka[i]) / mu[i];
            de[i] = u4[i] / mu[i];
        }

        ka[n - 4] = l1[n - 5] - al[n - 6] * l2[n - 6];
        mu[n - 4] = d0[n - 4] - be[n - 6] * l2[n - 6] - al[n - 5] * ka[n - 4];
        al[n - 4] = (u1[n - 4] - ga[n - 6] * l2[n - 6] - be[n - 5] * ka[n - 4]) / mu[n - 4];
        be[n - 4] = (u2[n - 4] - de[n - 6] * l2[n - 6] - ga[n - 5] * ka[n - 4]) / mu[n - 4];
        ga[n - 4] = (u3[n - 4] - de[n - 5] * ka[n - 4]) / mu[n - 4];

        ka[n - 3] = l1[n - 4] - al[n - 5] * l2[n - 5];
        mu[n - 3] = d0[n - 3] - be[n - 5] * l2[n - 5] - al[n - 4] * ka[n - 3];
        al[n - 3] = (u1[n - 3] - ga[n - 5] * l2[n - 5] - be[n - 4] * ka[n - 3]) / mu[n - 3];
        be[n - 3] = (u2[n - 3] - de[n - 5] * l2[n - 5] - ga[n - 4] * ka[n - 3]) / mu[n - 3];

        ka[n - 2] = l1[n - 3] - al[n - 4] * l2[n - 4];
        mu[n - 2] = d0[n - 2] - be[n - 4] * l2[n - 4] - al[n - 3] * ka[n - 2];
        al[n - 2] = (u1[n - 2] - ga[n - 4] * l2[n - 4] - be[n - 3] * ka[n - 2]) / mu[n - 2];

        ka[n - 1] = l1[n - 2] - al[n - 3] * l2[n - 3];
        mu[n - 1] = d0[n - 1] - be[n - 3] * l2[n - 3] - al[n - 2] * ka[n - 1];

        // Forward step
        x[0] = b[0] / mu[0];
        x[1] = (b[1] - x[0] * ka[1]) / mu[1];
        for i in 2..n {
            x[i] = (b[i] - x[i - 2] * l2[i - 2] - x[i - 1] * ka[i]) / mu[i];
        }

        // Backward substitution
        x[n - 2] = x[n - 2] - x[n - 1] * al[n - 2];
        x[n - 3] = x[n - 3] - x[n - 2] * al[n - 3] - x[n - 1] * be[n - 3];
        x[n - 4] = x[n - 4] - x[n - 3] * al[n - 4] - x[n - 2] * be[n - 4] - x[n - 1] * ga[n - 4];

        for i in (0..n - 4).rev() {
            x[i] = x[i] - x[i + 1] * al[i] - x[i + 2] * be[i] - x[i + 3] * ga[i] - x[i + 4] * de[i];
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

        let (l2, l1, d0, u1, u2, u3, u4) = (
            &self.l2, &self.l1, &self.d0, &self.u1, &self.u2, &self.u3, &self.u4,
        );
        // Allocate arrays
        let mut al = vec![T::zero(); n - 1];
        let mut be = vec![T::zero(); n - 2];
        let mut ga = vec![T::zero(); n - 3];
        let mut de = vec![T::zero(); n - 4];
        let mut ka = vec![T::zero(); n];
        let mut mu = vec![T::zero(); n];
        let mut x = vec![U::zero(); n];
        unsafe {
            // Precompute sweep
            *mu.get_unchecked_mut(0) = *d0.get_unchecked(0);
            *al.get_unchecked_mut(0) = *u1.get_unchecked(0) / *mu.get_unchecked(0);
            *be.get_unchecked_mut(0) = *u2.get_unchecked(0) / *mu.get_unchecked(0);
            *ga.get_unchecked_mut(0) = *u3.get_unchecked(0) / *mu.get_unchecked(0);
            *de.get_unchecked_mut(0) = *u4.get_unchecked(0) / *mu.get_unchecked(0);

            *ka.get_unchecked_mut(1) = *l1.get_unchecked(0);
            *mu.get_unchecked_mut(1) =
                *d0.get_unchecked(1) - *al.get_unchecked(0) * *ka.get_unchecked(1);
            *al.get_unchecked_mut(1) = (*u1.get_unchecked(1)
                - *be.get_unchecked(0) * *ka.get_unchecked(1))
                / *mu.get_unchecked(1);
            *be.get_unchecked_mut(1) = (*u2.get_unchecked(1)
                - *ga.get_unchecked(0) * *ka.get_unchecked(1))
                / *mu.get_unchecked(1);
            *ga.get_unchecked_mut(1) = (*u3.get_unchecked(1)
                - *de.get_unchecked(0) * *ka.get_unchecked(1))
                / *mu.get_unchecked(1);
            *de.get_unchecked_mut(1) = *u4.get_unchecked(1) / *mu.get_unchecked(1);

            for i in 2..n - 4 {
                *ka.get_unchecked_mut(i) =
                    *l1.get_unchecked(i - 1) - *al.get_unchecked(i - 2) * *l2.get_unchecked(i - 2);
                *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                    - *be.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    - *al.get_unchecked(i - 1) * *ka.get_unchecked(i);
                *al.get_unchecked_mut(i) = (*u1.get_unchecked(i)
                    - *ga.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    - *be.get_unchecked(i - 1) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *be.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                    - *de.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    - *ga.get_unchecked(i - 1) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *ga.get_unchecked_mut(i) = (*u3.get_unchecked(i)
                    - *de.get_unchecked(i - 1) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *de.get_unchecked_mut(i) = *u4.get_unchecked(i) / *mu.get_unchecked(i);
            }

            *ka.get_unchecked_mut(n - 4) =
                *l1.get_unchecked(n - 5) - *al.get_unchecked(n - 6) * *l2.get_unchecked(n - 6);
            *mu.get_unchecked_mut(n - 4) = *d0.get_unchecked(n - 4)
                - *be.get_unchecked(n - 6) * *l2.get_unchecked(n - 6)
                - *al.get_unchecked(n - 5) * *ka.get_unchecked(n - 4);
            *al.get_unchecked_mut(n - 4) = (*u1.get_unchecked(n - 4)
                - *ga.get_unchecked(n - 6) * *l2.get_unchecked(n - 6)
                - *be.get_unchecked(n - 5) * *ka.get_unchecked(n - 4))
                / *mu.get_unchecked(n - 4);
            *be.get_unchecked_mut(n - 4) = (*u2.get_unchecked(n - 4)
                - *de.get_unchecked(n - 6) * *l2.get_unchecked(n - 6)
                - *ga.get_unchecked(n - 5) * *ka.get_unchecked(n - 4))
                / *mu.get_unchecked(n - 4);
            *ga.get_unchecked_mut(n - 4) = (*u3.get_unchecked(n - 4)
                - *de.get_unchecked(n - 5) * *ka.get_unchecked(n - 4))
                / *mu.get_unchecked(n - 4);

            *ka.get_unchecked_mut(n - 3) =
                *l1.get_unchecked(n - 4) - *al.get_unchecked(n - 5) * *l2.get_unchecked(n - 5);
            *mu.get_unchecked_mut(n - 3) = *d0.get_unchecked(n - 3)
                - *be.get_unchecked(n - 5) * *l2.get_unchecked(n - 5)
                - *al.get_unchecked(n - 4) * *ka.get_unchecked(n - 3);
            *al.get_unchecked_mut(n - 3) = (*u1.get_unchecked(n - 3)
                - *ga.get_unchecked(n - 5) * *l2.get_unchecked(n - 5)
                - *be.get_unchecked(n - 4) * *ka.get_unchecked(n - 3))
                / *mu.get_unchecked(n - 3);
            *be.get_unchecked_mut(n - 3) = (*u2.get_unchecked(n - 3)
                - *de.get_unchecked(n - 5) * *l2.get_unchecked(n - 5)
                - *ga.get_unchecked(n - 4) * *ka.get_unchecked(n - 3))
                / *mu.get_unchecked(n - 3);

            *ka.get_unchecked_mut(n - 2) =
                *l1.get_unchecked(n - 3) - *al.get_unchecked(n - 4) * *l2.get_unchecked(n - 4);
            *mu.get_unchecked_mut(n - 2) = *d0.get_unchecked(n - 2)
                - *be.get_unchecked(n - 4) * *l2.get_unchecked(n - 4)
                - *al.get_unchecked(n - 3) * *ka.get_unchecked(n - 2);
            *al.get_unchecked_mut(n - 2) = (*u1.get_unchecked(n - 2)
                - *ga.get_unchecked(n - 4) * *l2.get_unchecked(n - 4)
                - *be.get_unchecked(n - 3) * *ka.get_unchecked(n - 2))
                / *mu.get_unchecked(n - 2);

            *ka.get_unchecked_mut(n - 1) =
                *l1.get_unchecked(n - 2) - *al.get_unchecked(n - 3) * *l2.get_unchecked(n - 3);
            *mu.get_unchecked_mut(n - 1) = *d0.get_unchecked(n - 1)
                - *be.get_unchecked(n - 3) * *l2.get_unchecked(n - 3)
                - *al.get_unchecked(n - 2) * *ka.get_unchecked(n - 1);

            // Forward step
            *x.get_unchecked_mut(0) = *b.get_unchecked(0) / *mu.get_unchecked(0);
            *x.get_unchecked_mut(1) = (*b.get_unchecked(1)
                - *x.get_unchecked(0) * *ka.get_unchecked(1))
                / *mu.get_unchecked(1);
            for i in 2..n {
                *x.get_unchecked_mut(i) = (*b.get_unchecked(i)
                    - *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    - *x.get_unchecked(i - 1) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
            }

            // Backward substitution
            *x.get_unchecked_mut(n - 2) =
                *x.get_unchecked(n - 2) - *x.get_unchecked(n - 1) * *al.get_unchecked(n - 2);
            *x.get_unchecked_mut(n - 3) = *x.get_unchecked(n - 3)
                - *x.get_unchecked(n - 2) * *al.get_unchecked(n - 3)
                - *x.get_unchecked(n - 1) * *be.get_unchecked(n - 3);
            *x.get_unchecked_mut(n - 4) = *x.get_unchecked(n - 4)
                - *x.get_unchecked(n - 3) * *al.get_unchecked(n - 4)
                - *x.get_unchecked(n - 2) * *be.get_unchecked(n - 4)
                - *x.get_unchecked(n - 1) * *ga.get_unchecked(n - 4);

            for i in (0..n - 4).rev() {
                *x.get_unchecked_mut(i) = *x.get_unchecked(i)
                    - *x.get_unchecked(i + 1) * *al.get_unchecked(i)
                    - *x.get_unchecked(i + 2) * *be.get_unchecked(i)
                    - *x.get_unchecked(i + 3) * *ga.get_unchecked(i)
                    - *x.get_unchecked(i + 4) * *de.get_unchecked(i);
            }
        }
        x
    }
}

/// Elementwise multiplication with scalar
impl<'a, T> std::ops::Mul<T> for &'a Hepta2101234<T>
where
    T: std::ops::MulAssign + Copy,
{
    type Output = Hepta2101234<T>;

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
        for x in &mut new.u3 {
            *x *= other;
        }
        for x in &mut new.u4 {
            *x *= other;
        }
        new
    }
}

/// Addition : &Self + &Self
impl<'a, 'b, T> Add<&'b Hepta2101234<T>> for &'a Hepta2101234<T>
where
    T: std::ops::AddAssign + Copy,
{
    type Output = Hepta2101234<T>;

    fn add(self, other: &'b Hepta2101234<T>) -> Self::Output {
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
        for (x, y) in new.u3.iter_mut().zip(other.u3.iter()) {
            *x += *y;
        }
        for (x, y) in new.u4.iter_mut().zip(other.u4.iter()) {
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
    fn test_hepta_2101234_dot() {
        let n = 23;
        let mut rng = thread_rng();
        // init
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let l1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u3: Vec<f64> = (0..n - 3).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let dma = Hepta2101234::new(l2, l1, d0, u1, u2, u3, u4);
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
    fn test_hepta_2101234_solve() {
        use ndarray_linalg::Solve;
        let n = 6;
        let mut rng = thread_rng();
        // init
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let l1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u3: Vec<f64> = (0..n - 3).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let dma = Hepta2101234::new(l2, l1, d0, u1, u2, u3, u4);
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
