use std::ops::{AddAssign, Mul};

use rayon;
use rayon::prelude::*;

fn main() {
    let _mat_d: usize = 100;
    let _mat_n: usize = 20;
    let _matrix_count: usize = 10;

    let inputs_w_x: Vec<(Vec<f32>, Vec<f32>)> = vec![(vec![0., 1., 2., 3.], vec![10., 20.])];
    let mut outputs_xout: Vec<Vec<f32>> = vec![vec![0.; 2]];

    outputs_xout
        .iter_mut()
        .zip(inputs_w_x.iter())
        .for_each(|(xout, (w, x))| {
            matmul(xout, x, w, 2, 2);
        });

    println!("{outputs_xout:?}");
}

/// W (d,n) @ x (n,) -> xout (d,)
fn matmul<T>(xout: &mut [T], x: &[T], w: &[T], n: usize, d: usize)
where
    T: Default + AddAssign + Mul<Output = T> + Send + Sync + Copy,
{
    assert_eq!(
        xout.len(),
        d,
        "expected different size of the slice. xout.len()={}, expected d={}",
        xout.len(),
        d
    );
    xout.par_iter_mut().enumerate().for_each(|(k, xout_val)| {
        //xout.iter_mut().enumerate().for_each(|(k, xout_val)| { // serial
        let mut val: T = T::default();
        for i in 0..n {
            val += w[k * n + i] * x[i];
        }
        *xout_val = val;
    });
}
