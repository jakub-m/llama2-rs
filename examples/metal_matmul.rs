//! Useful references:
//! - https://github.com/ryan-tobin/rustframes/blob/490be685dcc5252b6eacbf45df65b2f03476ed18/src/array/gpu.rs#L518
//!
//!
//! For Metal, the benchamrk is:
//!
//! ms per multiplication for dims n=3000 k=3000 m=3000 and 30 repeats:
//! - 14.5 ms - all allocations done in the loop
//! - 11.8 ms - memory allocations (even with shared memory) done outside loop
//! - 11.9 ms - initialising matrices done outside loop
//! - 11.0 ms - comitting at the end of the loop

use llama2_rs::matmul::matmul;
use objc2::AnyThread;
use objc2_foundation::NSUInteger;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLResourceOptions,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::{ffi::c_void, ptr::NonNull, time::SystemTime};

fn main() {
    let n_repeats = 30;

    for (i, d) in MTLCopyAllDevices().iter().enumerate() {
        eprintln!("(Device {j}: {d:?})", j = i + 1);
    }

    //   kkkkk        nnn         nnnnn
    // m 1 2 3      k 1 2       m 22 28
    // m 4 5 6  x   k 3 4  =    m 49 64
    // m 7 8 9      k 5 6       m 76 100

    let dim_m: usize = 3_000;
    let dim_k: usize = 3_000;
    let dim_n: usize = 3_000;

    let input_w: Vec<f32> = init_vec(dim_m * dim_k);
    let input_x: Vec<f32> = init_vec(dim_k * dim_n);
    // initial output values matter if beta MPSMatrixMultiplication is != 0 (it's GEMM)
    let mut output: Vec<f32> = vec![0.; dim_m * dim_n];

    let start = SystemTime::now();
    eprintln!("{t} start matmul ", t = elapsed(start));

    run_matmul_metal(
        n_repeats,
        &mut output,
        &input_w,
        &input_x,
        dim_m,
        dim_k,
        dim_n,
    );

    //---

    eprint!("\r");
    eprintln!("{} finished", elapsed(start));
    let dt = elapsed(start);
    eprintln!(
        "{p} ms per multiplication for dims n={dim_n} k={dim_k} m={dim_m} and {n_repeats} repeats",
        p = 1000. * dt / n_repeats as f32
    );
}

fn run_matmul_metal(
    n_repeats: usize,
    output: &mut [f32],
    input_w: &[f32],
    input_x: &[f32],
    dim_m: usize,
    dim_k: usize,
    dim_n: usize,
) {
    let device = MTLCreateSystemDefaultDevice().unwrap();
    eprintln!("{device:?}");
    let command_queue = device.newCommandQueue().unwrap();

    let buf_input_w = unsafe {
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                input_w.as_c_void(),
                input_w.len() * size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .unwrap()
    };

    let buf_input_x = unsafe {
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                input_x.as_c_void(),
                input_x.len() * size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .unwrap()
    };

    let buf_out = unsafe {
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                output.as_c_void(),
                output.len() * size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .unwrap()
    };

    let mat_w;
    unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_m as NSUInteger,
            dim_k as NSUInteger,
            dim_k * size_of::<f32>() as NSUInteger,
            MPSDataType::Float32,
        );

        mat_w = MPSMatrix::initWithBuffer_descriptor(mat, &buf_input_w, &desc);
    }
    //dbg!(&mat_w);

    let mat_x;
    unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_k as NSUInteger,
            dim_n as NSUInteger,
            dim_n * size_of::<f32>() as NSUInteger,
            MPSDataType::Float32,
        );

        mat_x = MPSMatrix::initWithBuffer_descriptor(mat, &buf_input_x, &desc);
    }
    //dbg!(&mat_x);

    let mat_out;
    unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_m as NSUInteger,
            dim_n as NSUInteger,
            dim_n * size_of::<f32>() as NSUInteger,
            MPSDataType::Float32,
        );

        mat_out = MPSMatrix::initWithBuffer_descriptor(mat, &buf_out, &desc);
    }
    //dbg!(&mat_out);

    let matmul;
    unsafe {
        let matmul_alloc = MPSMatrixMultiplication::alloc();
        matmul = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            matmul_alloc,
            &device,
            false, // transpose
            false,
            dim_m, // rows and cols
            dim_n,
            dim_k,
            1.0, // alpha, beta
            0.0,
        );
    };
    //dbg!(&matmul);

    let command_buffer = command_queue.commandBuffer().unwrap();
    for i in 0..n_repeats {
        eprint!("\r{i}  ");

        //dbg!(&command_buffer);
        // no compute encoder, because we don't have our library functions.

        unsafe {
            matmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                &command_buffer,
                &mat_w,
                &mat_x,
                &mat_out,
            );
        }
        //dbg!(command_buffer.status());

        //dbg!(input_w);
        //dbg!(input_x);
        //dbg!(output);
    }

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
}

trait AsNonNull {
    fn as_c_void(&self) -> NonNull<c_void>;
}

impl<T> AsNonNull for Vec<T> {
    fn as_c_void(&self) -> NonNull<c_void> {
        self.as_slice().as_c_void()
    }
}

impl<T> AsNonNull for &[T] {
    fn as_c_void(&self) -> NonNull<c_void> {
        let ptr = self.as_ptr() as *mut c_void;
        NonNull::new(ptr).unwrap()
    }
}

impl<T> AsNonNull for &mut [T] {
    fn as_c_void(&self) -> NonNull<c_void> {
        let ptr = self.as_ptr() as *mut c_void;
        NonNull::new(ptr).unwrap()
    }
}

fn init_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.01_f32 * i as f32).collect()
}

fn elapsed(start: SystemTime) -> f32 {
    SystemTime::now()
        .duration_since(start)
        .unwrap()
        .as_secs_f32()
}
