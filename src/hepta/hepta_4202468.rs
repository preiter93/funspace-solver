//! Hepta-diagonal matrix solver
//!     Ax = b
//! where A is banded with diagonals in offsets -4, -2, 0, 2, 4, 6, 8
#![allow(clippy::many_single_char_names)]
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Hepta4202468<T> {
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
    /// Upper diagonal (+6)
    pub(crate) u6: Vec<T>,
    /// Upper diagonal (+8)
    pub(crate) u8: Vec<T>,
}

impl<T> Hepta4202468<T> {
    /// Constructor
    ///
    /// ## Panics
    /// Shape mismatch of diagonals
    #[must_use]
    pub fn new(
        l4: Vec<T>,
        l2: Vec<T>,
        d0: Vec<T>,
        u2: Vec<T>,
        u4: Vec<T>,
        u6: Vec<T>,
        u8: Vec<T>,
    ) -> Self {
        assert!(d0.len() > 12, "Problem size too small.");
        assert!(l4.len() == d0.len() - 4);
        assert!(l2.len() == d0.len() - 2);
        assert!(u2.len() == d0.len() - 2);
        assert!(u4.len() == d0.len() - 4);
        assert!(u6.len() == d0.len() - 6);
        assert!(u8.len() == d0.len() - 8);
        Self {
            l4,
            l2,
            d0,
            u2,
            u4,
            u6,
            u8,
        }
    }

    /// Construct from matrix
    ///
    /// # Panics
    /// - Matrix is non-square
    /// - Matrix is non-zero on some diagonal other than -4, -2, 0, 2, 4, 6, 8
    #[must_use]
    pub fn from_array2(array: &ndarray::Array2<T>) -> Self
    where
        T: Zero + Copy + PartialEq,
    {
        let diagonals = crate::utils::extract_diagonals(array, [-4, -2, 0, 2, 4, 6, 8]);
        let diagonals = match diagonals {
            Ok(diagonals) => diagonals,
            Err(error) => panic!("{}", error.0),
        };
        let l4 = diagonals[0].clone();
        let l2 = diagonals[1].clone();
        let d0 = diagonals[2].clone();
        let u2 = diagonals[3].clone();
        let u4 = diagonals[4].clone();
        let u6 = diagonals[5].clone();
        let u8 = diagonals[6].clone();
        Self::new(l4, l2, d0, u2, u4, u6, u8)
    }

    /// Returns banded matrix
    #[must_use]
    pub fn as_array2(&self) -> ndarray::Array2<T>
    where
        T: Zero + Copy,
    {
        match crate::utils::array_from_diags(
            [
                &self.l4, &self.l2, &self.d0, &self.u2, &self.u4, &self.u6, &self.u8,
            ],
            [-4, -2, 0, 2, 4, 6, 8],
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
        let (l4, l2, d0, u2, u4, u6, u8) = (
            &self.l4, &self.l2, &self.d0, &self.u2, &self.u4, &self.u6, &self.u8,
        );
        for i in 0..2 {
            b[i] = x[i] * d0[i]
                + x[i + 2] * u2[i]
                + x[i + 4] * u4[i]
                + x[i + 6] * u6[i]
                + x[i + 8] * u8[i];
        }
        for i in 2..4 {
            b[i] = x[i] * d0[i]
                + x[i + 2] * u2[i]
                + x[i + 4] * u4[i]
                + x[i + 6] * u6[i]
                + x[i + 8] * u8[i]
                + x[i - 2] * l2[i - 2];
        }
        for i in 4..n - 8 {
            b[i] = x[i] * d0[i]
                + x[i + 2] * u2[i]
                + x[i + 4] * u4[i]
                + x[i + 6] * u6[i]
                + x[i + 8] * u8[i]
                + x[i - 2] * l2[i - 2]
                + x[i - 4] * l4[i - 4];
        }
        for i in n - 8..n - 6 {
            b[i] = x[i] * d0[i]
                + x[i + 2] * u2[i]
                + x[i + 4] * u4[i]
                + x[i + 6] * u6[i]
                + x[i - 2] * l2[i - 2]
                + x[i - 4] * l4[i - 4];
        }
        for i in n - 6..n - 4 {
            b[i] = x[i] * d0[i]
                + x[i + 2] * u2[i]
                + x[i + 4] * u4[i]
                + x[i - 2] * l2[i - 2]
                + x[i - 4] * l4[i - 4];
        }
        for i in n - 4..n - 2 {
            b[i] = x[i] * d0[i] + x[i + 2] * u2[i] + x[i - 2] * l2[i - 2] + x[i - 4] * l4[i - 4];
        }
        for i in n - 2..n {
            b[i] = x[i] * d0[i] + x[i - 2] * l2[i - 2] + x[i - 4] * l4[i - 4];
        }
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
        let (l4, l2, d0, u2, u4, u6, u8) = (
            &self.l4, &self.l2, &self.d0, &self.u2, &self.u4, &self.u6, &self.u8,
        );
        unsafe {
            for i in 0..2 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i + 4) * *u4.get_unchecked(i)
                    + *x.get_unchecked(i + 6) * *u6.get_unchecked(i)
                    + *x.get_unchecked(i + 8) * *u8.get_unchecked(i);
            }
            for i in 2..4 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i + 4) * *u4.get_unchecked(i)
                    + *x.get_unchecked(i + 6) * *u6.get_unchecked(i)
                    + *x.get_unchecked(i + 8) * *u8.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2);
            }
            for i in 4..n - 8 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i + 4) * *u4.get_unchecked(i)
                    + *x.get_unchecked(i + 6) * *u6.get_unchecked(i)
                    + *x.get_unchecked(i + 8) * *u8.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    + *x.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
            }
            for i in n - 8..n - 6 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i + 4) * *u4.get_unchecked(i)
                    + *x.get_unchecked(i + 6) * *u6.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    + *x.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
            }
            for i in n - 6..n - 4 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i + 4) * *u4.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    + *x.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
            }
            for i in n - 4..n - 2 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i + 2) * *u2.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    + *x.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
            }
            for i in n - 2..n {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *d0.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                    + *x.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
            }
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

        let (l4, l2, d0, u2, u4, u6, u8) = (
            &self.l4, &self.l2, &self.d0, &self.u2, &self.u4, &self.u6, &self.u8,
        );
        // Allocate arrays
        let mut al = vec![T::zero(); n - 2];
        let mut be = vec![T::zero(); n - 4];
        let mut ga = vec![T::zero(); n - 6];
        let mut de = vec![T::zero(); n - 8];
        let mut ka = vec![T::zero(); n];
        let mut mu = vec![T::zero(); n];
        let mut x = vec![U::zero(); n];

        // Precompute sweep
        for i in 0..2 {
            mu[i] = d0[i];
            al[i] = u2[i] / mu[i];
            be[i] = u4[i] / mu[i];
            ga[i] = u6[i] / mu[i];
            de[i] = u8[i] / mu[i];
        }
        for i in 2..4 {
            ka[i] = l2[i - 2];
            mu[i] = d0[i] - al[i - 2] * ka[i];
            al[i] = (u2[i] - be[i - 2] * ka[i]) / mu[i];
            be[i] = (u4[i] - ga[i - 2] * ka[i]) / mu[i];
            ga[i] = (u6[i] - de[i - 2] * ka[i]) / mu[i];
            de[i] = u8[i] / mu[i];
        }

        for i in 4..n - 8 {
            ka[i] = l2[i - 2] - al[i - 4] * l4[i - 4];
            mu[i] = d0[i] - be[i - 4] * l4[i - 4] - al[i - 2] * ka[i];
            al[i] = (u2[i] - ga[i - 4] * l4[i - 4] - be[i - 2] * ka[i]) / mu[i];
            be[i] = (u4[i] - de[i - 4] * l4[i - 4] - ga[i - 2] * ka[i]) / mu[i];
            ga[i] = (u6[i] - de[i - 2] * ka[i]) / mu[i];
            de[i] = u8[i] / mu[i];
        }
        for i in n - 8..n - 6 {
            ka[i] = l2[i - 2] - al[i - 4] * l4[i - 4];
            mu[i] = d0[i] - be[i - 4] * l4[i - 4] - al[i - 2] * ka[i];
            al[i] = (u2[i] - ga[i - 4] * l4[i - 4] - be[i - 2] * ka[i]) / mu[i];
            be[i] = (u4[i] - de[i - 4] * l4[i - 4] - ga[i - 2] * ka[i]) / mu[i];
            ga[i] = (u6[i] - de[i - 2] * ka[i]) / mu[i];
        }
        for i in n - 6..n - 4 {
            ka[i] = l2[i - 2] - al[i - 4] * l4[i - 4];
            mu[i] = d0[i] - be[i - 4] * l4[i - 4] - al[i - 2] * ka[i];
            al[i] = (u2[i] - ga[i - 4] * l4[i - 4] - be[i - 2] * ka[i]) / mu[i];
            be[i] = (u4[i] - de[i - 4] * l4[i - 4] - ga[i - 2] * ka[i]) / mu[i];
        }
        for i in n - 4..n - 2 {
            ka[i] = l2[i - 2] - al[i - 4] * l4[i - 4];
            mu[i] = d0[i] - be[i - 4] * l4[i - 4] - al[i - 2] * ka[i];
            al[i] = (u2[i] - ga[i - 4] * l4[i - 4] - be[i - 2] * ka[i]) / mu[i];
        }
        for i in n - 2..n {
            ka[i] = l2[i - 2] - al[i - 4] * l4[i - 4];
            mu[i] = d0[i] - be[i - 4] * l4[i - 4] - al[i - 2] * ka[i];
        }
        // Forward step
        x[0] = b[0] / mu[0];
        x[1] = b[1] / mu[1];
        x[2] = (b[2] - x[0] * ka[2]) / mu[2];
        x[3] = (b[3] - x[1] * ka[3]) / mu[3];
        for i in 4..n {
            x[i] = (b[i] - x[i - 4] * l4[i - 4] - x[i - 2] * ka[i]) / mu[i];
        }

        // Backward substitution
        x[n - 3] = x[n - 3] - x[n - 1] * al[n - 3];
        x[n - 4] = x[n - 4] - x[n - 2] * al[n - 4];
        x[n - 5] = x[n - 5] - x[n - 3] * al[n - 5] - x[n - 1] * be[n - 5];
        x[n - 6] = x[n - 6] - x[n - 4] * al[n - 6] - x[n - 2] * be[n - 6];
        x[n - 7] = x[n - 7] - x[n - 5] * al[n - 7] - x[n - 3] * be[n - 7] - x[n - 1] * ga[n - 7];
        x[n - 8] = x[n - 8] - x[n - 6] * al[n - 8] - x[n - 4] * be[n - 8] - x[n - 2] * ga[n - 8];

        for i in (0..n - 8).rev() {
            x[i] = x[i] - x[i + 2] * al[i] - x[i + 4] * be[i] - x[i + 6] * ga[i] - x[i + 8] * de[i];
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

        let (l4, l2, d0, u2, u4, u6, u8) = (
            &self.l4, &self.l2, &self.d0, &self.u2, &self.u4, &self.u6, &self.u8,
        );
        // Allocate arrays
        let mut al = vec![T::zero(); n - 2];
        let mut be = vec![T::zero(); n - 4];
        let mut ga = vec![T::zero(); n - 6];
        let mut de = vec![T::zero(); n - 8];
        let mut ka = vec![T::zero(); n];
        let mut mu = vec![T::zero(); n];
        let mut x = vec![U::zero(); n];
        unsafe {
            // Precompute sweep
            for i in 0..2 {
                *mu.get_unchecked_mut(i) = *d0.get_unchecked(i);
                *al.get_unchecked_mut(i) = *u2.get_unchecked(i) / *mu.get_unchecked(i);
                *be.get_unchecked_mut(i) = *u4.get_unchecked(i) / *mu.get_unchecked(i);
                *ga.get_unchecked_mut(i) = *u6.get_unchecked(i) / *mu.get_unchecked(i);
                *de.get_unchecked_mut(i) = *u8.get_unchecked(i) / *mu.get_unchecked(i);
            }
            for i in 2..4 {
                *ka.get_unchecked_mut(i) = *l2.get_unchecked(i - 2);
                *mu.get_unchecked_mut(i) =
                    *d0.get_unchecked(i) - *al.get_unchecked(i - 2) * *ka.get_unchecked(i);
                *al.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                    - *be.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *be.get_unchecked_mut(i) = (*u4.get_unchecked(i)
                    - *ga.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *ga.get_unchecked_mut(i) = (*u6.get_unchecked(i)
                    - *de.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *de.get_unchecked_mut(i) = *u8.get_unchecked(i) / *mu.get_unchecked(i);
            }

            for i in 4..n - 8 {
                *ka.get_unchecked_mut(i) =
                    *l2.get_unchecked(i - 2) - *al.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
                *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                    - *be.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *al.get_unchecked(i - 2) * *ka.get_unchecked(i);
                *al.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                    - *ga.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *be.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *be.get_unchecked_mut(i) = (*u4.get_unchecked(i)
                    - *de.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *ga.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *ga.get_unchecked_mut(i) = (*u6.get_unchecked(i)
                    - *de.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *de.get_unchecked_mut(i) = *u8.get_unchecked(i) / *mu.get_unchecked(i);
            }
            for i in n - 8..n - 6 {
                *ka.get_unchecked_mut(i) =
                    *l2.get_unchecked(i - 2) - *al.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
                *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                    - *be.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *al.get_unchecked(i - 2) * *ka.get_unchecked(i);
                *al.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                    - *ga.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *be.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *be.get_unchecked_mut(i) = (*u4.get_unchecked(i)
                    - *de.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *ga.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *ga.get_unchecked_mut(i) = (*u6.get_unchecked(i)
                    - *de.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
            }
            for i in n - 6..n - 4 {
                *ka.get_unchecked_mut(i) =
                    *l2.get_unchecked(i - 2) - *al.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
                *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                    - *be.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *al.get_unchecked(i - 2) * *ka.get_unchecked(i);
                *al.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                    - *ga.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *be.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
                *be.get_unchecked_mut(i) = (*u4.get_unchecked(i)
                    - *de.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *ga.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
            }
            for i in n - 4..n - 2 {
                *ka.get_unchecked_mut(i) =
                    *l2.get_unchecked(i - 2) - *al.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
                *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                    - *be.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *al.get_unchecked(i - 2) * *ka.get_unchecked(i);
                *al.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                    - *ga.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *be.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
            }
            for i in n - 2..n {
                *ka.get_unchecked_mut(i) =
                    *l2.get_unchecked(i - 2) - *al.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
                *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                    - *be.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *al.get_unchecked(i - 2) * *ka.get_unchecked(i);
            }
            // Forward step
            *x.get_unchecked_mut(0) = *b.get_unchecked(0) / *mu.get_unchecked(0);
            *x.get_unchecked_mut(1) = *b.get_unchecked(1) / *mu.get_unchecked(1);
            *x.get_unchecked_mut(2) = (*b.get_unchecked(2)
                - *x.get_unchecked(0) * *ka.get_unchecked(2))
                / *mu.get_unchecked(2);
            *x.get_unchecked_mut(3) = (*b.get_unchecked(3)
                - *x.get_unchecked(1) * *ka.get_unchecked(3))
                / *mu.get_unchecked(3);
            for i in 4..n {
                *x.get_unchecked_mut(i) = (*b.get_unchecked(i)
                    - *x.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                    - *x.get_unchecked(i - 2) * *ka.get_unchecked(i))
                    / *mu.get_unchecked(i);
            }

            // Backward substitution
            *x.get_unchecked_mut(n - 3) =
                *x.get_unchecked(n - 3) - *x.get_unchecked(n - 1) * *al.get_unchecked(n - 3);
            *x.get_unchecked_mut(n - 4) =
                *x.get_unchecked(n - 4) - *x.get_unchecked(n - 2) * *al.get_unchecked(n - 4);
            *x.get_unchecked_mut(n - 5) = *x.get_unchecked(n - 5)
                - *x.get_unchecked(n - 3) * *al.get_unchecked(n - 5)
                - *x.get_unchecked(n - 1) * *be.get_unchecked(n - 5);
            *x.get_unchecked_mut(n - 6) = *x.get_unchecked(n - 6)
                - *x.get_unchecked(n - 4) * *al.get_unchecked(n - 6)
                - *x.get_unchecked(n - 2) * *be.get_unchecked(n - 6);
            *x.get_unchecked_mut(n - 7) = *x.get_unchecked(n - 7)
                - *x.get_unchecked(n - 5) * *al.get_unchecked(n - 7)
                - *x.get_unchecked(n - 3) * *be.get_unchecked(n - 7)
                - *x.get_unchecked(n - 1) * *ga.get_unchecked(n - 7);
            *x.get_unchecked_mut(n - 8) = *x.get_unchecked(n - 8)
                - *x.get_unchecked(n - 6) * *al.get_unchecked(n - 8)
                - *x.get_unchecked(n - 4) * *be.get_unchecked(n - 8)
                - *x.get_unchecked(n - 2) * *ga.get_unchecked(n - 8);

            for i in (0..n - 8).rev() {
                *x.get_unchecked_mut(i) = *x.get_unchecked(i)
                    - *x.get_unchecked(i + 2) * *al.get_unchecked(i)
                    - *x.get_unchecked(i + 4) * *be.get_unchecked(i)
                    - *x.get_unchecked(i + 6) * *ga.get_unchecked(i)
                    - *x.get_unchecked(i + 8) * *de.get_unchecked(i);
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
    fn test_hepta_4202468_dot() {
        let n = 23;
        let mut rng = thread_rng();
        // init
        let l4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u6: Vec<f64> = (0..n - 6).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u8: Vec<f64> = (0..n - 8).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let dma = Hepta4202468::new(l4, l2, d0, u2, u4, u6, u8);
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
    fn test_hepta_4202468_solve() {
        use ndarray_linalg::Solve;
        let n = 16;
        let mut rng = thread_rng();
        // init
        let l4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u6: Vec<f64> = (0..n - 6).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u8: Vec<f64> = (0..n - 8).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let dma = Hepta4202468::new(l4, l2, d0, u2, u4, u6, u8);
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
