//! An example of SIMD instructions.

use std::arch::aarch64::{float32x4_t, vfmaq_f32, vld1q_f32};

fn main() {
    let a: Vec<f32> = vec![1., 2., 3., 4.];
    let b: Vec<f32> = vec![10., 20., 30., 40.];
    let c: Vec<f32> = vec![100., 200., 300., 400.];

    let va: float32x4_t = unsafe { vld1q_f32(a.as_ptr()) };
    let vb = unsafe { vld1q_f32(b.as_ptr()) };
    let vc = unsafe { vld1q_f32(c.as_ptr()) };

    let z = unsafe { vfmaq_f32(va, vb, vc) }; // (b * c + a)
    println!("a={a:?}");
    println!("b={b:?}");
    println!("c={c:?}");
    println!("z={z:?}");
}
