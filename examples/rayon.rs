use std::ops::{AddAssign, Mul};
use std::time::SystemTime;

use rayon;
use rayon::prelude::*;

fn elapsed(start: SystemTime) -> f32 {
    let now = SystemTime::now();
    now.duration_since(start).unwrap().as_secs_f32()
}

fn main() {
    let start = SystemTime::now();
    let m_d: usize = 1000;
    let m_n: usize = 1000;
    let m_count: usize = 1000;
    let n_repeat: usize = 100;

    println!("{t:.1} init arrays", t = elapsed(start));

    let inputs_w_x: Vec<(Vec<f32>, Vec<f32>)> = (0..m_count)
        .map(|_| (init_vec_f32(m_d * m_n), init_vec_f32(m_n)))
        .collect();

    let mut outputs_xout: Vec<Vec<f32>> = (0..m_count).map(|_| vec![0.; m_d]).collect();

    println!("{t:.1} start matmul", t = elapsed(start));

    for i in 0..n_repeat {
        outputs_xout
            .iter_mut()
            .zip(inputs_w_x.iter())
            .for_each(|(xout, (w, x))| {
                matmul(xout, x, w, m_n, m_d);
            });
        println!(
            "{t:.1} done matmuls iter {} ({}...)",
            i + 1,
            outputs_xout.len(),
            t = elapsed(start)
        );
    }
    println!("{t:.1} end", t = elapsed(start))
}

//// This could be generic.
//fn init_random_vec_f32(n: usize) -> Vec<f32> {
//    let mut rng = rand::rng();
//    (0..n).map(|_| rng.random()).collect()
//}

fn init_vec_f32(n: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..n).map(|i| (i as f32).into()).collect();
    v
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
