//! # Utils collection

#[derive(Debug)]
pub struct CustomError(pub String);

/// Returns requested diagonals
///
/// # Errors
/// - Matrix is non-square
/// - Requested diagonal is larger than array size
/// - Matrix is non-zero on some diagonal other than the requested
///
/// # Example
///```
/// use funspace_solver::utils::extract_diagonals;
/// use ndarray::Array2;
///
/// let n = 10;
///
/// // Construct some test matrix
/// let l2: Vec<f64> = vec![1.; n - 2];
/// let l1: Vec<f64> = vec![1.; n - 1];
/// let u1: Vec<f64> = vec![1.; n - 1];
/// let u2: Vec<f64> = vec![1.; n - 2];
/// let d0: Vec<f64> = vec![1.; n];
/// let mut array = ndarray::Array2::zeros((n, n));
/// for i in 0..n {
///     array[[i, i]] = d0[i];
/// }
/// for i in 0..n - 1 {
///     array[[i, i + 1]] = u1[i];
///     array[[i + 1, i]] = l1[i];
/// }
/// for i in 0..n - 2 {
///     array[[i, i + 2]] = u2[i];
///     array[[i + 2, i]] = l2[i];
/// }
/// // Test
/// let diagonals = extract_diagonals(&array, [-2, -1, 0, 1, 2]).unwrap();
/// for (x, y) in diagonals.iter().zip(vec![l2, l1, d0, u1, u2].iter()) {
///     assert_eq!(x, y);
/// }
/// // Test should fail (diagonal +2 is non zero but not requested)
/// let diagonals = extract_diagonals(&array, [-2, -1, 0, 1]);
/// assert!(diagonals.is_err());
///```
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn extract_diagonals<T, const N: usize>(
    array: &ndarray::Array2<T>,
    requested_diagonals: [i32; N],
) -> Result<Vec<Vec<T>>, CustomError>
where
    T: num_traits::Zero + std::cmp::PartialEq + Copy,
{
    // Check squareness
    if !array.is_square() {
        return Err(CustomError("Expected square matrix".to_string()));
    }
    let n = array.shape()[0];
    // Check diagonal is not larger than matrix itself
    for i in requested_diagonals {
        if i.abs() as usize >= n {
            return Err(CustomError(format!(
                "Requested diagonal {} larger than size of matrix {}",
                i, n
            )));
        }
    }
    // Check for duplicates
    for i in 1..requested_diagonals.len() {
        if requested_diagonals[i..].contains(&requested_diagonals[i - 1]) {
            return Err(CustomError(format!(
                "Requested diagonal {} appears multiple times",
                requested_diagonals[i - 1]
            )));
        }
    }
    // Check only requested diagonals are non-zero
    for i in 0..n {
        for j in 0..n {
            if !requested_diagonals.contains(&(j as i32 - i as i32)) && array[[i, j]] != T::zero() {
                return Err(CustomError(format!(
                    "Non-zero diagonal detected ({}) outside requested diagonals {:?}!",
                    j - i,
                    requested_diagonals
                )));
            }
        }
    }
    let mut diagonals: Vec<Vec<T>> = Vec::new();
    for d in requested_diagonals {
        let mut values: Vec<T> = Vec::new();
        if d >= 0 {
            // Upper diagonals
            for i in 0..n - d.abs() as usize {
                values.push(array[[i, i + d.abs() as usize]]);
            }
        } else {
            // Lower diagonals
            for i in 0..n - d.abs() as usize {
                values.push(array[[i + d.abs() as usize, i]]);
            }
        }
        diagonals.push(values);
    }
    Ok(diagonals)
}

/// Construct matrix from diagonals
///
/// # Errors
/// - Diagonal is outside of array
/// - Diagonal is specified multiple times
/// - Diagonal has wrong length
///
/// # Example
///```
/// use funspace_solver::utils::array_from_diags;
/// use ndarray::{Array2, array};
///
/// // Test diagonals
/// let l2: Vec<f64> = vec![1.; 1];
/// let d0: Vec<f64> = vec![2.; 3];
/// let u1: Vec<f64> = vec![3.; 2];
///
/// // Construct array
/// let array = array_from_diags([&l2, &d0, &u1], [-2, 0, 1], 3).unwrap();
/// assert_eq!(array, array![[2., 3., 0.], [0., 2., 3.], [1., 0., 2.]]);
///```
pub fn array_from_diags<T, const N: usize>(
    values: [&Vec<T>; N],
    diagonals: [i32; N],
    size: usize,
) -> Result<ndarray::Array2<T>, CustomError>
where
    T: num_traits::Zero + Copy,
{
    // Check diagonals are not larger than size
    for d in diagonals {
        if d.abs() as usize >= size {
            return Err(CustomError(format!(
                "Diagonal {} larger than size of matrix {}",
                d, size
            )));
        }
    }
    // Check for duplicates
    for i in 1..diagonals.len() {
        if diagonals[i..].contains(&diagonals[i - 1]) {
            return Err(CustomError(format!(
                "Diagonal {} appears multiple times",
                diagonals[i - 1]
            )));
        }
    }
    let mut array = ndarray::Array2::<T>::zeros((size, size));
    for (vec, &d) in values.iter().zip(diagonals.iter()) {
        // Check size
        if vec.len() != size - d.abs() as usize {
            return Err(CustomError(format!(
                "Diagonal has wrong length. Got {} expected {}",
                vec.len(),
                size - d.abs() as usize
            )));
        }
        if d >= 0 {
            // Upper diagonals
            for i in 0..size - d.abs() as usize {
                array[[i, i + d.abs() as usize]] = vec[i];
            }
        } else {
            // Lower diagonals
            for i in 0..size - d.abs() as usize {
                array[[i + d.abs() as usize, i]] = vec[i];
            }
        }
    }
    Ok(array)
}

/// Determine bandedness of a matrix
///
/// # Panics
/// Not a square matrix
///
/// # Example
///```
/// use funspace_solver::utils::determine_bandedness;
/// use ndarray::Array2;
///
/// let n = 10;
///
/// // Construct some test matrix
/// let l1: Vec<f64> = vec![1.; n - 1];
/// let u1: Vec<f64> = vec![1.; n - 1];
/// let l2: Vec<f64> = vec![1.; n - 2];
/// let u2: Vec<f64> = vec![1.; n - 2];
/// let d0: Vec<f64> = vec![1.; n];
/// let mut array = ndarray::Array2::zeros((n, n));
/// for i in 0..n {
///     array[[i, i]] = d0[i];
/// }
/// for i in 0..n - 1 {
///     array[[i, i + 1]] = u1[i];
///     array[[i + 1, i]] = l1[i];
/// }
/// for i in 0..n - 2 {
///     array[[i, i + 2]] = u2[i];
///     array[[i + 2, i]] = l2[i];
/// }
///
/// // Test
/// let non_zero_bands = determine_bandedness(&array);
/// assert_eq!(non_zero_bands, vec![-2, -1, 0, 1, 2]);
///```
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
pub fn determine_bandedness<T: num_traits::Zero + std::cmp::PartialEq>(
    array: &ndarray::Array2<T>,
) -> Vec<i32> {
    assert!(array.is_square(), "Expected square matrix");
    let n = array.shape()[0];
    // Loop over all possible diagonals
    let mut non_zero_bands: Vec<i32> = vec![];
    for diag in 0..n {
        // Check upper diagonal
        // Loop over all elements
        for i in 0..n - diag {
            if array[[i, i + diag]] != T::zero() {
                non_zero_bands.push(diag as i32);
                break;
            }
        }
        // Check lower diagonal
        if diag != 0 {
            for i in 0..n - diag {
                if array[[i + diag, i]] != T::zero() {
                    non_zero_bands.push(-(diag as i32));
                    break;
                }
            }
        }
    }
    non_zero_bands.sort_unstable();
    non_zero_bands
}
