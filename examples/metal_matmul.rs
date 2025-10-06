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

    let input_a: Vec<f32> = vec![1., 2., 3., 4.];
    let input_b: Vec<f32> = vec![10., 20., 30., 40.];
    let output: Vec<f32> = vec![0.; 4]; // input values matter since MPSMatrixMultiplication adds
    // to the output
    let h = 2_usize;
    let w = 2_usize;
    assert_eq!(input_a.len(), h * w);
    assert_eq!(input_b.len(), h * w);
    assert_eq!(output.len(), h * w);

    let buf_a = unsafe {
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                input_a.as_c_void(),
                input_a.len() * size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .unwrap()
    };

    let buf_b = unsafe {
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                input_b.as_c_void(),
                input_b.len() * size_of::<f32>(),
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
    dbg!(&buf_out);

    let m_a;
    unsafe {
        let m_input_a = MPSMatrix::alloc();
        let m_a_desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            2 as NSUInteger,
            2 as NSUInteger,
            2 * size_of::<f32>() as NSUInteger,
            MPSDataType::Float32,
        );

        m_a = MPSMatrix::initWithBuffer_descriptor(m_input_a, &buf_a, &m_a_desc);
    }
    dbg!(&m_a);

    let m_b;
    unsafe {
        let m_input_b = MPSMatrix::alloc();
        let m_b_desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            2 as NSUInteger,
            2 as NSUInteger,
            2 * size_of::<f32>() as NSUInteger,
            MPSDataType::Float32,
        );

        m_b = MPSMatrix::initWithBuffer_descriptor(m_input_b, &buf_b, &m_b_desc);
    }
    dbg!(&m_b);

    let m_out;
    unsafe {
        let m_output = MPSMatrix::alloc();
        let m_out_desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            2 as NSUInteger,
            2 as NSUInteger,
            2 * size_of::<f32>() as NSUInteger,
            MPSDataType::Float32,
        );

        m_out = MPSMatrix::initWithBuffer_descriptor(m_output, &buf_out, &m_out_desc);
    }
    dbg!(&m_out);

    let command_buffer = command_queue.commandBuffer().unwrap();
    dbg!(&command_buffer);
    // no compute encoder, because we don't have our library functions.
    //
    let matmul;
    unsafe {
        let matmul_alloc = MPSMatrixMultiplication::alloc();
        matmul = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            matmul_alloc,
            &device,
            false, // transpose
            false,
            2, // rows and cols
            2,
            2,
            1.0, // alpha, beta
            1.0,
        );
    };
    dbg!(&matmul);
    unsafe {
        matmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            &command_buffer,
            &m_a,
            &m_b,
            &m_out,
        );
    }
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    dbg!(command_buffer.status());

    dbg!(input_a);
    dbg!(input_b);
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
