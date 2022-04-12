//! # Two-dimensional Poisson equation in Chebyshev space
//! BC: u(-1) = u(1) = 0
//!
//! # Documentation
//! see ``examples/docs/doc_poisson2d.pdf``
use funspace::traits::*;
use funspace::{cheb_dirichlet, chebyshev};
use funspace_solver::tetra::Tetra2024;
use funspace_solver::tri::Tri202;
use ndarray::{Array1, Array2, Zip};
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    let now = Instant::now();

    let n = 26;
    let ch = chebyshev::<f64>(n);
    let cd = cheb_dirichlet::<f64>(n);
    let xc = cd.coords();

    // Initialize right hand side
    let k = PI / 2.;
    let mut f = Array2::zeros((n, n));
    for (i, xi) in xc.iter().enumerate() {
        for (j, yi) in xc.iter().enumerate() {
            f[[i, j]] = (k * (xi - xc[0])).sin() * (k * (yi - xc[0])).sin();
        }
    }
    let mut fhat = Array2::zeros((ch.len_spec(), ch.len_spec()));
    {
        let scratch = ch.forward(&f, 1);
        ch.forward_inplace(&scratch, &mut fhat, 0);
    }

    // Initialize solution
    let mut s = Array2::zeros((n, n));
    for (si, fi) in s.iter_mut().zip(f.iter()) {
        *si = -1. / (k * k * 2.) * fi;
    }
    let mut shat = Array2::zeros((cd.len_spec(), cd.len_spec()));
    {
        let scratch = cd.forward(&s, 1);
        cd.forward_inplace(&scratch, &mut shat, 0);
    }

    // Ingredients
    let (tx, bx, ay, cy, lam, q) = {
        let st = cd.stencil();
        let st_inv = cd.stencil_inv();
        let d2 = cd.laplacian();
        let (l_pinv, i_pinv) = cd.laplacian_pinv();
        let ay: Array2<f64> = i_pinv.dot(&l_pinv).dot(&st);
        let cy: Array2<f64> = i_pinv.dot(&st);
        // Eigendecomposition
        let g = st_inv.dot(&(d2.dot(&st)));
        let (lam, q, q_inv) = eig(&g);
        // // Test eigendecomposition
        // let lam2: Array2<f64> = Array2::from_diag(&lam);
        // let g2 = q.dot(&(lam2.dot(&q_inv)));
        // for (g1, g2) in g.iter().zip(g2.iter()) {
        //     assert!((g1 - g2).abs() < 1e-3);
        // }
        // Transforms
        let tx = q_inv.dot(&st_inv);
        // Preconditioners
        let bx = l_pinv;
        (tx, bx, ay, cy, lam, q)
    };

    // Banded solvers
    // let bands_bx = funspace_solver::utils::determine_bandedness(&bx);
    // println!("{:?}", bands_bx);
    let rdma_bx = Tri202::from_array2(&bx);

    // let bands_ay = funspace_solver::utils::determine_bandedness(&ay);
    // println!("{:?}", bands_ay);
    // let bands_cy = funspace_solver::utils::determine_bandedness(&cy);
    // println!("{:?}", bands_cy);
    let ldma_ay = Tetra2024::from_array2(&ay);
    let ldma_cy = Tetra2024::from_array2(&cy);

    let elapsed = now.elapsed();
    println!("Initialize: {:.2?}", elapsed);
    let now = Instant::now();

    // Step 1: Forward Transform rhs along x
    let fhat_star = tx.dot(&fhat);

    // Step 2: Solve along y (but iterate over all lanes in x)
    let mut vhat_star: Array2<f64> = Array2::zeros((cd.len_spec(), cd.len_spec()));
    Zip::from(vhat_star.outer_iter_mut())
        .and(fhat_star.outer_iter())
        .and(lam.outer_iter())
        .par_for_each(|mut v, f, lam| {
            let mut rhs = rdma_bx.dot(&f.as_slice().unwrap());
            assert!(rhs[0] == 0.);
            assert!(rhs[1] == 0.);
            rhs.drain(0..2);
            // Unpack lambda
            let l = lam.as_slice().unwrap()[0];
            // lhs solver
            let ldma = &ldma_cy + &(&ldma_ay * l);
            let vhat_vec = ldma.solve(&rhs);
            for (x, y) in v.iter_mut().zip(vhat_vec.iter()) {
                *x = *y
            }
        });

    // Step 3: Backward Transform solution along x
    let vhat = q.dot(&vhat_star);

    let elapsed = now.elapsed();
    println!("Solve     :  {:.2?}", elapsed);

    // Transform back
    let mut v = Array2::zeros((cd.len_phys(), cd.len_phys()));
    {
        let scratch = cd.backward(&vhat, 0);
        cd.backward_inplace(&scratch, &mut v, 1);
    }

    // Print
    let mut norm = 0.;
    let mut rel = 0.;
    for (xi, yi) in v.iter().zip(s.iter()) {
        norm += (xi - yi).powi(2);
        rel += yi.powi(2);
    }
    norm = norm.sqrt() / rel.sqrt();
    println!(" |err| = {:10.5e}", norm);
}

/// Returns real-valued eigendecomposition A = Q lam Qi,
/// where A is a square matrix.
/// The output is already sorted with respect to the
/// eigenvalues, i.e. largest -> smallest.
///
/// # Example
/// ```
/// use ndarray::array;
/// let test = array![
///         [1., 2., 3., 4., 5.],
///         [1., 2., 3., 4., 5.],
///         [1., 2., 3., 4., 5.],
///         [1., 2., 3., 4., 5.],
///         [1., 2., 3., 4., 5.]
///     ];
/// let (e, evec, evec_inv) = eig(&test);
/// ```
///
/// ## Panics
/// Panics if eigendecomposition or inverse fails.
fn eig(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    use ndarray::Axis;
    use ndarray_linalg::Eig;

    // use old ndarray version, which supports linalg
    let (n, m) = (a.shape()[0], a.shape()[1]);
    let mut m = Array2::<f64>::zeros((n, m));
    for (oldv, newv) in m.iter_mut().zip(a.iter()) {
        *oldv = *newv;
    }
    let (eval_c, evec_c) = m.eig().unwrap();
    // let eval_c = ndarray_vec_to_new(&eval_c);
    // let evec_c = ndarray_to_new(&evec_c);
    // Convert complex -> f64
    let mut eval = Array1::zeros(eval_c.raw_dim());
    let mut evec = Array2::zeros(evec_c.raw_dim());
    for (e, ec) in eval.iter_mut().zip(eval_c.iter()) {
        *e = ec.re;
    }
    for (e, ec) in evec.iter_mut().zip(evec_c.iter()) {
        *e = ec.re;
    }
    // Order Eigenvalues, largest first
    let permut: Vec<usize> = argsort(eval.as_slice().unwrap())
        .into_iter()
        .rev()
        .collect();
    let eval = eval.select(Axis(0), &permut).to_owned();
    let evec = evec.select(Axis(1), &permut).to_owned();
    // Inverse of evec
    let evec_inv = inv(&evec);
    (eval, evec, evec_inv)
}

/// Return inverse of square matrix
/// ## Panics
/// Panics when computation of inverse fails.
fn inv(a: &Array2<f64>) -> Array2<f64> {
    use ndarray_linalg::Inverse;
    a.inv().unwrap()
}

/// Argsort Vector ( smallest -> largest ).
/// Returns permutation vector.
///
/// ```
/// use ndarray::{array,Axis};
/// let vec = array![3., 1., 2., 9., 7.];
/// let permut: Vec<usize> = argsort(vec.as_slice().unwrap());
/// let vec = vec.select(Axis(0), &permut).to_owned();
/// assert_eq!(vec,array![1.0, 2.0, 3.0, 7.0, 9.0]);
/// ```
fn argsort(vec: &[f64]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..vec.len()).collect();
    perm.sort_by(|i, j| {
        if vec[*i] < vec[*j] {
            Ordering::Less
        } else if vec[*i] > vec[*j] {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    perm
}
