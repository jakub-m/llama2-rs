//! Useful references:
//! - https://github.com/ryan-tobin/rustframes/blob/490be685dcc5252b6eacbf45df65b2f03476ed18/src/array/gpu.rs#L518
//!
use objc2::AnyThread;
use objc2_foundation::NSUInteger;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::{ffi::c_void, ptr::NonNull};

fn main() {
    let device = MTLCreateSystemDefaultDevice().unwrap();
    let command_queue = device.newCommandQueue().unwrap();

    //   kkkkk        nnn         nnnnn
    // m 1 2 3      k 1 2       m 22 28
    // m 4 5 6  x   k 3 4  =    m 49 64
    // m 7 8 9      k 5 6       m 76 100

    let input_w: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let input_x: Vec<f32> = vec![1., 2., 3., 4., 5., 6.];
    let output: Vec<f32> = vec![0.; 6]; // input values matter if beta MPSMatrixMultiplication is != 0 (it's GEMM)
    // to the output
    let dim_m = 3_usize;
    let dim_k = 3_usize;
    let dim_n = 2_usize;

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
    dbg!(&mat_w);

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
    dbg!(&mat_x);

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
    dbg!(&mat_out);

    let command_buffer = command_queue.commandBuffer().unwrap();
    dbg!(&command_buffer);
    // no compute encoder, because we don't have our library functions.

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
    dbg!(&matmul);
    unsafe {
        matmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            &command_buffer,
            &mat_w,
            &mat_x,
            &mat_out,
        );
    }
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    dbg!(command_buffer.status());

    dbg!(input_w);
    dbg!(input_x);
    dbg!(output);
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

// TODO: use non-square matrix.
// TODO transpose
