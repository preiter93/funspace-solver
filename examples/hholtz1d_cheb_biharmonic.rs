//! # One-dimensional Helmholtz equation in Chebyshev space
//! BC: u(-1) = u''(-1) = u(1) = u''(1) = 0
//!
//! $$$
//! (c - \nabla^4) x = b
//! $$$
//! where c is a constant.
#![allow(unused_imports, dead_code, unused_variables)]
extern crate ndarray_linalg;
use funspace::traits::*;
use funspace::{cheb_biharmonic_b, chebyshev};
use funspace_solver::hepta::Hepta4202468;
use funspace_solver::penta::Penta42024;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use std::f64::consts::PI;
//use funspace_solver::utils::determine_bandedness;

fn main() {
    let n = 20;
    let ch = chebyshev::<f64>(n);
    let cd = cheb_biharmonic_b::<f64>(n);
    let xc = cd.coords();
    let length = xc[xc.len() - 1] - xc[0];
    let c = 1e-1;

    // Initialize right hand side vector
    let mut b: Vec<f64> = vec![0.; cd.len_phys()];
    let k = 4. * PI / length;
    for (xi, bi) in xc.iter().zip(b.iter_mut()) {
        *bi = (c - k * k * k * k) * (k * (xi - xc[0])).sin();
    }
    let mut bhat: Vec<f64> = vec![0.; ch.len_spec()];
    ch.forward_slice(&b, &mut bhat);

    // Initialize solution
    let mut s: Vec<f64> = vec![0.; cd.len_phys()];
    for (xi, si) in xc.iter().zip(s.iter_mut()) {
        *si = (k * (xi - xc[0])).sin();
    }

    // Initialize left- and right-hand side matrices
    let (l_mat, r_mat) = {
        let stencil = cd.stencil();
        let (l_pinv, i_pinv) = cd.diffmat_pinv(4);
        let l: Array2<f64> = (c * i_pinv.dot(&l_pinv) - i_pinv).dot(&stencil);
        let r: Array2<f64> = l_pinv;
        (l, r)
    };

    // let b = determine_bandedness(&l_mat);
    // println!("{:?}", b);
    // let b = determine_bandedness(&r_mat);
    // println!("{:?}", b);

    // Extract diagonals and initialize solver of lhs
    let ldma = Hepta4202468::from_array2(&l_mat);

    // Extract diagonals and initialize solver of rhs
    let rdma = Penta42024::from_array2(&r_mat);

    // Solve rhs
    let mut rhs = rdma.dot(&bhat);
    assert!(rhs[0] == 0.);
    assert!(rhs[1] == 0.);
    assert!(rhs[2] == 0.);
    assert!(rhs[3] == 0.);
    rhs.drain(0..4);

    // Solve lhs
    let xhat = ldma.solve(&rhs);
    // Solve lhs with ndarray-linalg
    let xhat_nd = l_mat.solve_into(Array1::from_vec(rhs)).unwrap();
    assert_approx_tol(&xhat, &xhat_nd.to_vec(), 1e-6);

    // Transform back
    let mut x: Vec<f64> = vec![0.; cd.len_phys()];
    cd.backward_slice(&xhat, &mut x);
    println!("{:10} | {:10}", " solution", "theoretical");
    println!("{:10} | {:10}", "----------", "----------");
    let mut norm = 0.;
    for (xi, si) in x.iter().zip(s.iter()) {
        println!("{:10.5} | {:10.5}", xi, si);
        norm += (xi - si).powi(2);
    }
    println!("{:10} | {:10}", "----------", "----------");
    norm = norm.sqrt();
    println!(" |err| = {:10.5e}", norm);
}

fn assert_approx_tol(x: &[f64], y: &[f64], tol: f64) {
    for (xi, yi) in x.iter().zip(y.iter()) {
        assert!((xi - yi).abs() < tol);
    }
}
