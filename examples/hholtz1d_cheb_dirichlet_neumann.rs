//! # One-dimensional Helmholtz equation in Chebyshev space
//! BC: u(-1) = u'(1) = 0
//!
//! $$$
//! (c - \nabla^2) x = b
//! $$$
//! where c is a constant.
extern crate ndarray_linalg;
use funspace::traits::*;
use funspace::{cheb_dirichlet_neumann, chebyshev};
use funspace_solver::hepta::Hepta2101234;
use funspace_solver::tri::Tri202;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use std::f64::consts::PI;

fn main() {
    let n = 10;
    let ch = chebyshev::<f64>(n);
    let cd = cheb_dirichlet_neumann::<f64>(n);
    let xc = cd.coords();
    let length = xc[xc.len() - 1] - xc[0];
    let c = 1e-1;

    // Initialize right hand side vector
    let mut b: Vec<f64> = vec![0.; cd.len_phys()];
    let k = 6. / 4. * PI / length;
    for (xi, bi) in xc.iter().zip(b.iter_mut()) {
        *bi = (c + k * k) * (k * (xi - xc[0])).sin();
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
        let (l_pinv, i_pinv) = cd.laplacian_pinv();
        let l: Array2<f64> = (c * i_pinv.dot(&l_pinv) - i_pinv).dot(&stencil);
        let r: Array2<f64> = l_pinv;
        (l, r)
    };

    // Extract diagonals and initialize solver of lhs
    let ldma = Hepta2101234::from_array2(&l_mat);

    // Extract diagonals and initialize solver of rhs
    let rdma = Tri202::from_array2(&r_mat);

    // Solve rhs
    let mut rhs = rdma.dot(&bhat);
    assert!(rhs[0] == 0.);
    assert!(rhs[1] == 0.);
    rhs.drain(0..2);

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
