use crate::{metal, sliceutil::Offset};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSUInteger, ns_string};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::{ffi::c_void, fmt::Debug, ptr::NonNull};

const CONVERT_F16_SOURCE: &str = include_str!("./convert_f16.metal");

type RetainedMTLBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type RetainedMTLCommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type RetainedMTLCommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type RetainedMTLComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type RetainedMTLDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type RetainedMTLFunction = Retained<ProtocolObject<dyn MTLFunction>>;
type RetainedMTLLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;

/// A placeholder for f16 that I can use around the code here. I don't handle f16 in Rust anyway,
/// it's only to allocate proper buffer sizes for GPU.
#[derive(Clone)]
struct F16(u16);

impl Default for F16 {
    fn default() -> Self {
        Self(0)
    }
}

pub struct MetalState {
    device: RetainedMTLDevice,
    command_queue: RetainedMTLCommandQueue,
    /// Library with the compiled functions.
    library: RetainedMTLLibrary,
    /// A function witin the library.
    func_pso_convert_f32_to_f16: RetainedMTLComputePipelineState,
    /// A function witin the library.
    func_pso_convert_f16_to_f32: RetainedMTLComputePipelineState,

    // whole mtl buffer
    pub mtl_buffer_wq_f16: RetainedMTLBuffer,
    pub mtl_buffer_wk_f16: RetainedMTLBuffer,
    pub mtl_buffer_wv: RetainedMTLBuffer,
    pub mtl_buffer_wo: RetainedMTLBuffer,
    pub mtl_buffer_w1: RetainedMTLBuffer,
    pub mtl_buffer_w2: RetainedMTLBuffer,
    pub mtl_buffer_w3: RetainedMTLBuffer,
    pub mtl_buffer_wcls: RetainedMTLBuffer,
}

impl MetalState {
    pub fn new(
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        wcls: &[f32],
    ) -> Self {
        let device: RetainedMTLDevice = MTLCreateSystemDefaultDevice().unwrap();
        let command_queue = device.newCommandQueue().unwrap();

        let convert_f16_source = ns_string!(CONVERT_F16_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(
                convert_f16_source,
                None, /*compilation options*/
            )
            .unwrap();

        let convert_f32_to_f16_func = library
            .newFunctionWithName(ns_string!("convert_f32_to_f16"))
            .unwrap();

        let convert_f16_to_f32_func = library
            .newFunctionWithName(ns_string!("convert_f16_to_f32"))
            .unwrap();

        let func_pso_convert_f32_to_f16 = device
            .newComputePipelineStateWithFunction_error(&convert_f32_to_f16_func)
            .unwrap();

        let func_pso_convert_f16_to_f32 = device
            .newComputePipelineStateWithFunction_error(&convert_f16_to_f32_func)
            .unwrap();

        let new_f16_buffer = |source_f32: &[f32]| unsafe {
            let (b, cb) = Self::new_private_f16_buffer_from_f32_slice_nowait(
                &device,
                &command_queue,
                &func_pso_convert_f32_to_f16,
                source_f32,
            );
            cb.waitUntilCompleted();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            (b, cb)
        };

        let (mtl_buffer_wq_f16, _) = new_f16_buffer(wq);
        let (mtl_buffer_wk_f16, _) = new_f16_buffer(wk);
        let mtl_buffer_wv = unsafe { Self::new_shared_mtl_buffer_priv(&device, wv) };
        let mtl_buffer_wo = unsafe { Self::new_shared_mtl_buffer_priv(&device, wo) };
        let mtl_buffer_w1 = unsafe { Self::new_shared_mtl_buffer_priv(&device, w1) };
        let mtl_buffer_w2 = unsafe { Self::new_shared_mtl_buffer_priv(&device, w2) };
        let mtl_buffer_w3 = unsafe { Self::new_shared_mtl_buffer_priv(&device, w3) };
        let mtl_buffer_wcls = unsafe { Self::new_shared_mtl_buffer_priv(&device, wcls) };

        MetalState {
            device,
            command_queue,
            library,
            func_pso_convert_f32_to_f16,
            func_pso_convert_f16_to_f32,
            mtl_buffer_wq_f16,
            mtl_buffer_wk_f16,
            mtl_buffer_wv,
            mtl_buffer_wo,
            mtl_buffer_w1,
            mtl_buffer_w2,
            mtl_buffer_w3,
            mtl_buffer_wcls,
        }
    }

    unsafe fn new_private_f16_buffer_from_f32_slice_nowait(
        device: &RetainedMTLDevice,
        command_queue: &RetainedMTLCommandQueue,
        func_pso_convert_f32_to_f16: &RetainedMTLComputePipelineState,
        source_f32: &[f32],
    ) -> (RetainedMTLBuffer, RetainedMTLCommandBuffer) {
        let mtl_buf_f32 = unsafe { Self::new_shared_mtl_buffer_priv(&device, source_f32) };
        let mtl_buf_f16 = unsafe {
            Self::new_private_mtl_buffer_priv(&device, source_f32.len() * size_of::<F16>())
        };
        let command_buffer = Self::execute_func_over_array_no_wait_priv(
            &command_queue,
            &func_pso_convert_f32_to_f16,
            source_f32.len(),
            &mtl_buf_f32,
            &mtl_buf_f16,
        );
        (mtl_buf_f16, command_buffer)
    }

    unsafe fn new_private_mtl_buffer_from_slice(
        device: &RetainedMTLDevice,
        command_queue: &RetainedMTLCommandQueue,
        buf: &[f32],
    ) -> RetainedMTLBuffer {
        let shared_buf = unsafe { Self::new_shared_mtl_buffer_priv(&device, buf) };
        let buf_size = buf.len() * size_of::<f32>();
        let private_buf = device
            .newBufferWithLength_options(buf_size, MTLResourceOptions::StorageModePrivate)
            .unwrap();

        let command_buffer = command_queue.commandBuffer().unwrap();
        let blit_command_encoder = command_buffer.blitCommandEncoder().unwrap();
        unsafe {
            blit_command_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &shared_buf,
                0,
                &private_buf,
                0,
                buf_size,
            );
        };
        blit_command_encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        private_buf
    }

    pub unsafe fn new_private_mtl_buffer(&self, buf_size_bytes: usize) -> RetainedMTLBuffer {
        unsafe { Self::new_private_mtl_buffer_priv(&self.device, buf_size_bytes) }
    }

    unsafe fn new_private_mtl_buffer_priv(
        device: &RetainedMTLDevice,
        buf_size_bytes: usize,
    ) -> RetainedMTLBuffer {
        device
            .newBufferWithLength_options(buf_size_bytes, MTLResourceOptions::StorageModePrivate)
            .unwrap()
    }

    unsafe fn new_shared_mtl_buffer(&self, buf: &[f32]) -> RetainedMTLBuffer {
        Self::new_shared_mtl_buffer_priv(&self.device, buf)
    }

    unsafe fn new_shared_mtl_buffer_priv(
        device: &RetainedMTLDevice,
        buf: &[f32],
    ) -> RetainedMTLBuffer {
        unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    buf.as_c_void(),
                    buf.len() * size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .unwrap()
        }
    }

    fn execute_func_over_array_wait(
        &self,
        convert_func: &RetainedMTLComputePipelineState,
        n_elem: usize,
        source: &RetainedMTLBuffer,
        target: &RetainedMTLBuffer,
    ) {
        Self::execute_func_over_array_wait_priv(
            &self.command_queue,
            convert_func,
            n_elem,
            source,
            target,
        );
    }

    // TODO waitUntilCompleted after all the passes are done
    /// Execute the function over 1d array, and wait for the result.
    fn execute_func_over_array_wait_priv(
        command_queue: &RetainedMTLCommandQueue,
        convert_func: &RetainedMTLComputePipelineState,
        n_elem: usize,
        source: &RetainedMTLBuffer,
        target: &RetainedMTLBuffer,
    ) {
        let command_buffer = Self::execute_func_over_array_no_wait_priv(
            command_queue,
            convert_func,
            n_elem,
            source,
            target,
        );
        command_buffer.waitUntilCompleted();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
    }

    pub fn execute_func_over_array_no_wait(
        &self,
        convert_func: &RetainedMTLComputePipelineState,
        n_elem: usize,
        source: &RetainedMTLBuffer,
        target: &RetainedMTLBuffer,
    ) -> RetainedMTLCommandBuffer {
        Self::execute_func_over_array_no_wait_priv(
            &self.command_queue,
            convert_func,
            n_elem,
            source,
            target,
        )
    }

    /// Commit the function over 1d array, but do not wait for the result.
    fn execute_func_over_array_no_wait_priv(
        command_queue: &RetainedMTLCommandQueue,
        convert_func: &RetainedMTLComputePipelineState,
        n_elem: usize,
        source: &RetainedMTLBuffer,
        target: &RetainedMTLBuffer,
    ) -> RetainedMTLCommandBuffer {
        let command_buffer = command_queue.commandBuffer().unwrap();
        let compute_encoder = command_buffer.computeCommandEncoder().unwrap();

        compute_encoder.setComputePipelineState(&convert_func);
        unsafe {
            compute_encoder.setBuffer_offset_atIndex(Some(&source), 0, 0);
            compute_encoder.setBuffer_offset_atIndex(Some(&target), 0, 1);
        }
        let (grid_size, thread_group_size) =
            Self::grid_and_thread_group_size_for_linear_op(n_elem, convert_func);

        compute_encoder.dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);
        compute_encoder.endEncoding();
        command_buffer.commit();
        command_buffer
    }

    /// Return parameters needed to run function over 1d array.
    fn grid_and_thread_group_size_for_linear_op(
        array_elem_count: usize,
        func: &RetainedMTLComputePipelineState,
    ) -> (MTLSize, MTLSize) {
        let grid_size = MTLSize {
            width: array_elem_count,
            height: 1,
            depth: 1,
        };

        let max_threads_per_group = func.maxTotalThreadsPerThreadgroup();
        let width = if max_threads_per_group > array_elem_count {
            array_elem_count
        } else {
            max_threads_per_group
        };
        let thread_group_size = MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        (grid_size, thread_group_size)
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
    fn metal_buffer(&self, b_sel: B) -> Option<&RetainedMTLBuffer>;
}

/// Return state of Metal, which is initialised at the very beginning. This state can also
/// contain per-buffer data.
pub trait WithMetalState {
    fn metal_state(&self) -> &MetalState;
}

/// State is the state that is needed for efficient operation. This includes: Metal state, shared weight
/// buffer state. Buffer indicates which buffer (from the state) is used for multiplication.
/// `w_sel` - selects the buffer from the state.
pub fn matmul_s<S: WithMetalBuf<B> + WithMetalState, B: Copy + Debug>(
    state: &S,
    xout: &mut [f32],
    x: &[f32],
    w_sel: B,
    w_offset: Offset,
    dim_n: usize,
    dim_d: usize,
) {
    let dim_u = 1;

    // Declare shared buffers. The memory is already allocaded in the main code, here we say to
    // share the allocated memory with GPU.
    //assert_eq!(
    //    w_ref_with_offset.len(),
    //    dim_n * dim_d,
    //    "w.len() ({w_len}) != n * d ({dim_n} * {dim_d})",
    //    w_len = w_ref_with_offset.len(),
    //);
    assert_eq!(w_offset.end - w_offset.start, dim_n * dim_d);
    let metal_state = state.metal_state();

    let buf_w = state.metal_buffer(w_sel).unwrap();

    // Now describe the input buffers as matrices of appropriate dimensions.
    // W matrix is an array of values with rows packed one after another (no padding etc).
    let mat_w = unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_d as NSUInteger,
            dim_n as NSUInteger,
            dim_n * size_of::<f32>() as NSUInteger, // stride
            MPSDataType::Float32,
        );
        //mat_w = MPSMatrix::initWithBuffer_descriptor(mat, &buf_w, &desc);
        MPSMatrix::initWithBuffer_offset_descriptor(
            mat,
            &buf_w,
            w_offset.start * size_of::<f32>(),
            &desc,
        )
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
                dim_d, // result_col
                dim_u, // result_row
                dim_n, // interior_col
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

/// Run matrix multiplication on f16 matrices.
///
/// Internally, during matrix multiplication, W matrix, X and XOUT vectors are f16.  W should be
/// already transformed to f16 in advance, but x and xout are transformed on the fly here. Those
/// buffers are relatively small, and it does not seem that allocating those buffers is that much
/// overhead (although, it is definitely, and it would be an optimization opportunity to reuse
/// those interim buffers).
///
/// # Parameters
/// - w_sel - selector of the buffer with W weights. The weights are f16.
pub fn matmul_s_f16<S: WithMetalBuf<B> + WithMetalState, B: Copy + Debug>(
    state: &S,
    xout: &mut [f32],
    x: &[f32],
    w_sel: B,
    w_offset: Offset,
    dim_n: usize,
    dim_d: usize,
) {
    let dim_u = 1;

    assert_eq!(w_offset.end - w_offset.start, dim_n * dim_d);
    let metal_state = state.metal_state();
    let buf_w = state.metal_buffer(w_sel).unwrap();

    // Now describe the input buffers as matrices of appropriate dimensions.
    // W matrix is an array of values with rows packed one after another (no padding etc).
    let mat_w = unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_d as NSUInteger,
            dim_n as NSUInteger,
            dim_n * size_of::<F16>() as NSUInteger, // stride
            MPSDataType::Float16,
        );

        MPSMatrix::initWithBuffer_offset_descriptor(
            mat,
            &buf_w,
            w_offset.start * size_of::<F16>(),
            &desc,
        )
    };

    // Now for the x f32 buffer, we need to convert the buffer to f16.
    assert_eq!(
        x.len(),
        dim_n,
        "x.len() ({x_len}) != dim_n ({dim_n})",
        x_len = x.len()
    );

    let buf_x_f32 = unsafe { metal_state.new_shared_mtl_buffer(&x) };
    // This is an interim f16 x buffer. The optimization opportunity is to not allocate this buffer
    // in each matmul, but reuse.
    let buf_x_f16 = unsafe { metal_state.new_private_mtl_buffer(x.len() * size_of::<F16>()) };
    // TODO Do not wait for the result yet here.
    metal_state.execute_func_over_array_wait(
        &metal_state.func_pso_convert_f32_to_f16,
        x.len(),
        &buf_x_f32,
        &buf_x_f16,
    );

    // Now do the same with the output buffer.
    // TODO: Do not allocate output, but write to xout, and then pass over xout and convert f16 to
    // f32 in place. Mind the stride!
    assert_eq!(
        xout.len(),
        dim_d,
        "xout.len() ({xout_len}) != dim_d ({dim_d})",
        xout_len = xout.len()
    );
    let buf_xout_f32 = unsafe { metal_state.new_shared_mtl_buffer(&xout) };
    let buf_xout_f16 = unsafe { metal_state.new_private_mtl_buffer(xout.len() * size_of::<F16>()) };

    // Now define the matrices.
    // X matrics (vector) is just a flat array.
    let mat_x = unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_n as NSUInteger,
            dim_u as NSUInteger,
            dim_u * size_of::<F16>() as NSUInteger,
            MPSDataType::Float16,
        );

        MPSMatrix::initWithBuffer_descriptor(mat, &buf_x_f16, &desc)
    };

    // The output matrix (vector) is just a flat array.
    let mat_xout = unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_d as NSUInteger,
            dim_u as NSUInteger,
            dim_u * size_of::<F16>() as NSUInteger,
            MPSDataType::Float16,
        );

        MPSMatrix::initWithBuffer_descriptor(mat, &buf_xout_f16, &desc)
    };

    let command_buffer = metal_state.command_queue.commandBuffer().unwrap();
    let matmul = unsafe {
        let matmul_alloc = MPSMatrixMultiplication::alloc();
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    matmul_alloc,
                    &metal_state.device,
                    false,
                    false,
                    dim_d, // result_col
                    dim_u, // result_row
                    dim_n, // interior_col
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
    // TODO Do not wait for the reslt yet!
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Now convert f16 to f32 into the shared buffer.
    // TODO do not wait for the result yet
    metal_state.execute_func_over_array_wait(
        &metal_state.func_pso_convert_f16_to_f32,
        x.len(),
        &buf_xout_f16,
        &buf_xout_f32,
    );

    // TODO convert f16 to f32 after matmul.
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
}
