use crate::sliceutil::{Offset, SliceFromOffset};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLResourceOptions,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::{ffi::c_void, ptr::NonNull};

pub struct MetalState {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    // whole mtl buffer
    pub mtl_buffer_wq: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MetalState {
    pub fn new(wq: &[f32]) -> MetalState {
        let device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>> =
            MTLCreateSystemDefaultDevice().unwrap();
        let command_queue = device.newCommandQueue().unwrap();

        let mtl_buffer_wq = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    wq.as_c_void(),
                    wq.len() * size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared, // TODO here initialize private
                    None,
                )
                .unwrap()
        };

        MetalState {
            device,
            command_queue,
            mtl_buffer_wq,
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
    panic!("bla")
}

/// Return direct reference to the particular buffer with weights.
pub trait WithBufferRef<B> {
    fn buffer_ref(&self, b_sel: B) -> &[f32];
}

/// Return already pre-initialized buffer suitable for GPU usage. The purpose is to initialise the
/// buffers exactly once (in shared or in GPU-private memory).
pub trait WithMetalBuf<B> {
    fn metal_buffer(&self, b_sel: B) -> Option<&Retained<ProtocolObject<dyn MTLBuffer>>>;
}

/// Return state of Metal, which is initialised at the very beginning. This state can also
/// contain per-buffer data.
pub trait WithMetalState {
    fn metal_state(&self) -> &MetalState;
}

/// State is the state that is needed for efficient operation. This includes: Metal state, shared weight
/// buffer state. Buffer indicates which buffer (from the state) is used for multiplication.
/// `w_sel` - selects the buffer from the state.
pub fn matmul_s<S: WithBufferRef<B> + WithMetalBuf<B> + WithMetalState, B: Copy>(
    state: &S,
    xout: &mut [f32],
    x: &[f32],
    w_sel: B,
    w_offset: Offset,
    dim_n: usize,
    dim_d: usize,
) {
    let dim_u = 1;

    let w_ref_full: &[f32] = state.buffer_ref(w_sel);

    // Declare shared buffers. The memory is already allocaded in the main code, here we say to
    // share the allocated memory with GPU.
    //assert_eq!(
    //    w_ref_with_offset.len(),
    //    dim_n * dim_d,
    //    "w.len() ({w_len}) != n * d ({dim_n} * {dim_d})",
    //    w_len = w_ref_with_offset.len(),
    //);
    let metal_state = state.metal_state();

    let buf_w_holder;
    let buf_w = if let Some(mb) = state.metal_buffer(w_sel) {
        // TODO add some check if dim_n * dim_d is the actual size of the whole w?
        panic!("has buffer");
        mb
    } else {
        //assert_eq!(dim_n * dim_d, w_ref_with_offset.len());
        let o = unsafe {
            metal_state
                .device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    w_ref_full.as_c_void(),
                    w_ref_full.len() * size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .unwrap()
        };
        buf_w_holder = Some(o);
        buf_w_holder.as_ref().unwrap()
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
        //mat_w = MPSMatrix::initWithBuffer_descriptor(mat, &buf_w, &desc);
        mat_w = MPSMatrix::initWithBuffer_offset_descriptor(
            mat,
            &buf_w,
            w_offset.start * size_of::<f32>(),
            &desc,
        );
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

    // X matrics (vector) is just a flat array.
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
                dim_d,
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
}

//struct MatmulState<'a> {
//    // The weights in memory (mmaped possibly) for multiplications. Those weights might be copied
//    // to GPU-private memory.
//    w: TransformerWeights<'a>,
//}

///// One of the predefined buffers with weights. The buffer values correspond to particular buffers
///// in [TransformerWeights].
//enum MatmulBuffer {
//    Wq,
//}

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
}
