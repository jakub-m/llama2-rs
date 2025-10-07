use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::{ffi::c_void, ptr::NonNull};

pub struct MetalState {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalState {
    pub fn new() -> MetalState {
        let device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>> =
            MTLCreateSystemDefaultDevice().unwrap();
        let command_queue = device.newCommandQueue().unwrap();
        MetalState {
            device,
            command_queue,
        }
    }
}

/// Implementation of matmul with Metal, in a ways it fits the ovaral Llama code.
/// ```text
/// W (d,n) @ x (n,) -> xout (d,)
///
///   --n--       1     1(=u)
/// | wwwww     | x     | o
/// d wwwww  @  | x  =  d o
/// | wwwww     n x     | o
///             | x
///             | x
///
/// n - n. rows
/// d - internal
/// 1 - n cols
/// ```text
pub fn matmul(
    metal_state: &MetalState,
    xout: &mut [f32],
    x: &[f32],
    w: &[f32],
    dim_n: usize,
    dim_d: usize,
) {
    let dim_u = 1;

    // Declare shared buffers. The memory is already allocaded in the main code, here we say to
    // share the allocated memory with GPU.
    assert_eq!(
        w.len(),
        dim_n * dim_d,
        "w.len() ({w_len}) != n * d ({dim_n} * {dim_d})",
        w_len = w.len(),
    );
    let buf_w = unsafe {
        metal_state
            .device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                w.as_c_void(),
                dim_n * dim_d * size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .unwrap()
    };

    assert_eq!(
        x.len(),
        dim_n,
        "x.len() ({x_len}) != dim_n ({dim_n})",
        x_len = x.len()
    );
    let buf_x = unsafe {
        metal_state
            .device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                x.as_c_void(),
                dim_n * size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .unwrap()
    };

    assert_eq!(
        xout.len(),
        dim_d,
        "xout.len() ({xout_len}) != dim_d ({dim_d})",
        xout_len = xout.len()
    );
    let buf_xout = unsafe {
        metal_state
            .device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                xout.as_c_void(),
                dim_d * size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .unwrap()
    };

    // Now describe the input buffers as matrices of appropriate dimensions.
    // W matrix is an array of values with rows packed one after another (no padding etc).
    let mat_w;
    unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_d as NSUInteger,
            dim_n as NSUInteger,
            dim_n * size_of::<f32>() as NSUInteger, // stride
            MPSDataType::Float32,
        );
        mat_w = MPSMatrix::initWithBuffer_descriptor(mat, &buf_w, &desc);
    };

    // X matric (vector) is just a flat array.
    let mat_x;
    unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_n as NSUInteger,
            dim_u as NSUInteger,
            dim_u * size_of::<f32>() as NSUInteger,
            MPSDataType::Float32,
        );

        mat_x = MPSMatrix::initWithBuffer_descriptor(mat, &buf_x, &desc);
    }

    // The output matrix (vector) is just a flat array.
    let mat_xout;
    unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_d as NSUInteger,
            dim_u as NSUInteger,
            dim_u * size_of::<f32>() as NSUInteger,
            MPSDataType::Float32,
        );

        mat_xout = MPSMatrix::initWithBuffer_descriptor(mat, &buf_xout, &desc);
    }
    // TODO prepare command buffer only once?
    // TODO do not prepare commad  buffer if the last operation was the same op. with the same
    // dims?
    let command_buffer = metal_state.command_queue.commandBuffer().unwrap();

    let matmul = unsafe {
        let matmul_alloc = MPSMatrixMultiplication::alloc();
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                matmul_alloc,
                &metal_state.device,
                false,
                false,
                dim_d,
                dim_u,
                dim_n,
                1.0,
                0.0,
            )
    };

    unsafe {
        matmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            &command_buffer,
            &mat_w,
            &mat_x,
            &mat_xout,
        );
    }
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    //dbg!(command_buffer.status());
    //dbg!(&w);
    //dbg!(&x);
    //dbg!(&xout);
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

#[cfg(test)]
mod tests {
    use super::*;
    /// There should be a similar test in main.rs but for the regular matmul.
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
        let ms = MetalState::new();
        matmul(&ms, &mut output, &input_x, &input_w, 3, 4);

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

    //#[test]
    //fn test_matmul_large_vector() {
    //    //matmul xout.len=1376 x.len=512 w.len=704512
    //    // W (d,n) @ x (n,) -> xout (d,)
    //    let dim_n: usize = 512;
    //    let dim_d: usize = 1376;

    //    let mut input_w = vec![0_f32; dim_n * dim_d];

    //    for i_d in 0..dim_d {
    //        for i_n in 0..dim_n {
    //            input_w[i_n + dim_n * i_d] = 0.01 * i_n as f32 + 1. * i_d as f32;
    //        }
    //    }

    //    let mut input_x = vec![0_f32; dim_n];
    //    for i in 0..dim_n {
    //        input_x[i] = 0.001 * i as f32;
    //    }

    //    let mut output = vec![0_f32; dim_d];

    //    let ms = MetalState::new();
    //    matmul(&ms, &mut output, &input_x, &input_w, dim_n, dim_d);
    //}
}
