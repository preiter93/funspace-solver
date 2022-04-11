//! # One-dimensional Poisson equation in Chebyshev space
//! BC: u(-1) = u'(-1) = u(1) = u'(1) = 0
//!
//! $$$
//! \nabla^4 x = b
//! $$$
extern crate ndarray_linalg;
use funspace::traits::*;
use funspace::{cheb_biharmonic_b, chebyshev};
use funspace_solver::penta::Penta42024;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use std::f64::consts::PI;

fn main() {
    let n = 20;
    let ch = chebyshev::<f64>(n);
    let cd = cheb_biharmonic_b::<f64>(n);
    let xc = cd.coords();
    let length = xc[xc.len() - 1] - xc[0];
    // Initialize left- and right-hand side matrices
    let (l_mat, r_mat) = {
        let stencil = cd.stencil();
        let (l_pinv, i_pinv) = cd.diffmat_pinv(4);
        let l: Array2<f64> = i_pinv.dot(&stencil);
        let r: Array2<f64> = l_pinv;
        (l, r)
    };

    // Initialize right hand side vector
    let mut b: Vec<f64> = vec![0.; cd.len_phys()];
    let k = 4. * PI / length;
    for (xi, bi) in xc.iter().zip(b.iter_mut()) {
        *bi = (k * (xi - xc[0])).sin();
    }
    let mut bhat: Vec<f64> = vec![0.; ch.len_spec()];
    ch.forward_slice(&b, &mut bhat);

    // Extract diagonals and initialize solver of lhs
    let ldma = Penta42024::from_array2(&l_mat);

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

    // Solve inefficiently without preconditioning
    let (l_mat, r_mat) = {
        let stencil = cd.stencil();
        let stencil_inv = cd.stencil_inv();
        let l = stencil_inv.dot(&(cd.diffmat(4).dot(&stencil)));
        let r = stencil_inv.to_owned();
        (l, r)
    };
    let rhs = r_mat.dot(&Array1::from_vec(bhat));
    let xhat_nd = l_mat.solve_into(rhs).unwrap();
    assert_approx_tol(&xhat, &xhat_nd.to_vec(), 1e-6);
    //
    // Transform back
    let mut x: Vec<f64> = vec![0.; cd.len_phys()];
    cd.backward_slice(&xhat, &mut x);
    println!("{:10} | {:10}", " solution", "theoretical");
    println!("{:10} | {:10}", "----------", "----------");
    let mut norm = 0.;
    for (xi, bi) in x.iter().zip(b.iter()) {
        println!("{:10.5} | {:10.5}", xi, 1. * bi / k.powi(4 as i32));
        norm += (xi - 1. * bi / k.powi(4 as i32)).powi(2);
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
