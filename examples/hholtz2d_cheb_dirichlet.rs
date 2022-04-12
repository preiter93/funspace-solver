//! # Two-dimensional Helmholtz equation in Chebyshev space
//! BC: u(-1) = u(1) = 0
//!
//! $$$
//! (1 - c \nabla^2) x = b
//! $$$
//! where c is a constant.
//!
//! # Documentation
//! see ``examples/docs/doc_hholtz_adi2d.pdf``
extern crate ndarray_linalg;
use funspace::traits::*;
use funspace::{cheb_dirichlet, chebyshev};
use funspace_solver::tetra::Tetra2024;
use ndarray::{Array2, Axis, Zip};
use std::f64::consts::PI;

fn main() {
    let n = 80;
    let ch = chebyshev::<f64>(n);
    let cd = cheb_dirichlet::<f64>(n);
    let xc = cd.coords();
    let length = xc[xc.len() - 1] - xc[0];
    let c = 1e-4;

    // Initialize right hand side
    let k = 4. * PI / length;
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
        *si = 1. / (1. + c * k * k + c * k * k) * fi;
    }
    let mut shat = Array2::zeros((cd.len_spec(), cd.len_spec()));
    {
        let scratch = cd.forward(&s, 1);
        cd.forward_inplace(&scratch, &mut shat, 0);
    }

    // Initialize left- and right-hand side matrices
    let (l_mat, r_mat) = {
        let stencil = cd.stencil();
        let (l_pinv, i_pinv) = cd.laplacian_pinv();
        let l: Array2<f64> = (i_pinv.dot(&l_pinv) - c * i_pinv).dot(&stencil);
        let r: Array2<f64> = l_pinv;
        (l, r)
    };

    // Extract diagonals and initialize solver of lhs
    let ldma = Tetra2024::from_array2(&l_mat);

    // Extract diagonals and initialize solver of rhs
    let rdma = Tetra2024::from_array2(&r_mat);

    let bands_ay = funspace_solver::utils::determine_bandedness(&l_mat);
    println!("{:?}", bands_ay);
    let bands_cy = funspace_solver::utils::determine_bandedness(&r_mat);
    println!("{:?}", bands_cy);

    // Step 1: Solve along y
    let mut ghat: Array2<f64> = Array2::zeros((ch.len_spec(), cd.len_spec()));
    Zip::from(ghat.lanes_mut(Axis(1)))
        .and(fhat.lanes(Axis(1)))
        .for_each(|mut v, f| {
            // Solve rhs
            let mut rhs = rdma.dot(&f.as_slice().unwrap());
            assert!(rhs[0] == 0.);
            assert!(rhs[1] == 0.);
            rhs.drain(0..2);
            // Solve lhs
            let vhat_vec = ldma.solve(&rhs);
            for (x, y) in v.iter_mut().zip(vhat_vec.iter()) {
                *x = *y
            }
        });

    // Step 2: Solve along x
    let mut vhat: Array2<f64> = Array2::zeros((cd.len_spec(), cd.len_spec()));
    Zip::from(vhat.lanes_mut(Axis(0)))
        .and(ghat.lanes(Axis(0)))
        .for_each(|mut v, f| {
            // Solve rhs
            let mut rhs = rdma.dot(&f.to_vec());
            assert!(rhs[0] == 0.);
            assert!(rhs[1] == 0.);
            rhs.drain(0..2);
            // Solve lhs
            let vhat_vec = ldma.solve(&rhs);
            for (x, y) in v.iter_mut().zip(vhat_vec.iter()) {
                *x = *y
            }
        });

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
        // println!("{:?} {:?}", xi, yi);
        norm += (xi - yi).powi(2);
        rel += yi.powi(2);
    }
    norm = norm.sqrt() / rel.sqrt();
    println!(" |err| = {:10.5e}", norm);
}

// fn assert_approx_tol(x: &[f64], y: &[f64], tol: f64) {
//     for (xi, yi) in x.iter().zip(y.iter()) {
//         assert!((xi - yi).abs() < tol);
//     }
// }
