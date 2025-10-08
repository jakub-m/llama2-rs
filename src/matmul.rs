use crate::trace;

use rayon;
use rayon::prelude::*;

/// W (d,n) @ x (n,) -> xout (d,)
/// by far the most amount of time is spent inside this little function
/// Here d is used only for extra boundary checks, xout slice should have the correct length.
pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    trace!("matmul n={n} d={d}");
    assert_eq!(
        xout.len(),
        d,
        "expected different size of the slice. xout.len()={}, expected d={}",
        xout.len(),
        d
    );
    // Purely serial option
    //   xout.iter_mut().enumerate().for_each(|(k, xout_val)| {
    // Rayon par_iter_mut
    //   xout.par_iter_mut().enumerate().for_each(|(k, xout_val)| {
    // Par Bridge
    //   xout.iter_mut().enumerate().par_bridge().for_each(|(k, xout_val)| {
    xout.par_iter_mut()
        .with_min_len(15)
        .enumerate()
        .for_each(|(k, xout_val)| {
            let mut val: f32 = 0.0;
            for i in 0..n {
                val += w[k * n + i] * x[i];
            }
            *xout_val = val;
        });
}

#[cfg(test)]
mod tests {
    use crate::matmul::matmul;

    #[test]
    fn test_matmul() {
        let w: Vec<f32> = vec![
            0.1, 0.2, 0.3, // r0
            0.4, 0.5, 0.6, // r1
        ];
        let x: Vec<f32> = vec![
            1.0, //r0
            2.0, //r1
            3.0, //r2
        ];

        let mut xout: Vec<f32> = vec![0.0; 2];

        matmul(&mut xout, &x, &w, 3, 2);
        assert_eq!(
            xout,
            vec![
                0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0, //r0
                0.4 * 1.0 + 0.5 * 2.0 + 0.6 * 3.0, //r1
            ]
        );
        assert_eq!(xout.len(), 2);
    }

    #[test]
    fn tests_matmul_w_4_3() {
        let input_w: Vec<f32> = vec![
            1., 2., 3., //
            4., 5., 6., //
            7., 8., 9., //
            10., 11., 12., //
        ];
        let input_x: Vec<f32> = vec![
            1., //
            2., //
            3., //
        ];
        let mut output: Vec<f32> = vec![0.; 4];
        matmul(&mut output, &input_x, &input_w, 3, 4);

        assert_eq!(
            output,
            vec![
                1. * 1. + 2. * 2. + 3. * 3.,    //
                4. * 1. + 5. * 2. + 6. * 3.,    //
                7. * 1. + 8. * 2. + 9. * 3.,    //
                10. * 1. + 11. * 2. + 12. * 3., //
            ]
        );
    }
}
