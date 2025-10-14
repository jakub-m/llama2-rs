//! Useful references:
//! - https://github.com/ryan-tobin/rustframes/blob/490be685dcc5252b6eacbf45df65b2f03476ed18/src/array/gpu.rs#L518
//!
//!
//! For Metal, the benchamrk is:
//!
//! ms per multiplication for dims n=1 k=1000 m=1000000 and 30 repeats.
//! Note that those dimensions mean 1GB W matrix.
//!
//! - 80.2 -- CPU (rayon)
//! - 179.6 -- GPU, configure shared memory buffers in each iteration
//! - 71.5 -- GPU, allocate buf. once, set matrix in each iteration
//! - 73.1 -- GPU, allocate buf. once, commit after each command
//! - 73.1 -- GPU, allocate buf. once, commit after all the loops  (frees CPU from waiting)
//!
//!
//! For a square W matrix, and larger output vector k=31623 m=31623 and 30 repeats
//! - 111.0156 ms GPU
//! - 88.0205 ms CPU (rayon)
//!
//! Allocating buffers and matrices (even if no-copy in shared mem) between each run, makes the
//! computation run at 126ms per matmul, vs. 5ms per matmul when all the buffers and matrices are
//! allocated only once.
//!
//! Initializing matrices (but not buffers) inside the loop seems to be of a small overhead.
//!
//! Initialising the small shared buffers inside the loop and the large W matrix outside loop seems
//! still be very fast. It's large W allocation that seems very slow.
//!
//! Changing offset in the matrices in each iteration (offset to layer) does not affect
//! performance, or affects it insignivicantly (unsurprisingly).

use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSUInteger, ns_string};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::{ffi::c_void, ptr::NonNull, time::SystemTime};

const N_LAYERS: usize = 32;
const N_REPEATS: usize = 1000;

const DIM: usize = 4096;
const HIDDEN_DIM: usize = 11008;

const CONVERT_F16_SOURCE: &str = include_str!("./convert_f16.metal");

const SIZE_OF_F32: usize = size_of::<f32>();
const SIZE_OF_F16: usize = SIZE_OF_F32 / 2;

type RetainedMTLBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type RetainedMTLCommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type RetainedMTLDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type RetainedMTLFunction = Retained<ProtocolObject<dyn MTLFunction>>;
type RetainedMTLComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

fn main() {
    for (i, d) in MTLCopyAllDevices().iter().enumerate() {
        eprintln!("(Device {j}: {d:?})", j = i + 1);
    }

    //   kkkkk        nnn         nnnnn
    // m 1 2 3      k 1 2       m 22 28
    // m 4 5 6  x   k 3 4  =    m 49 64
    // m 7 8 9      k 5 6       m 76 100

    let dim_m: usize = HIDDEN_DIM;
    let dim_k: usize = DIM;
    let dim_n: usize = 1;

    let input_w: Vec<f32> = init_vec(dim_m * dim_k * N_LAYERS);
    let input_x: Vec<f32> = init_vec(dim_k * dim_n);
    // initial output values matter if beta MPSMatrixMultiplication is != 0 (it's GEMM)
    let mut output: Vec<f32> = vec![0.; dim_m * dim_n];

    let start = SystemTime::now();
    eprintln!("{t} start matmul ", t = elapsed(start));

    run_matmul_metal(&mut output, &input_w, &input_x, dim_m, dim_k, dim_n);

    eprint!("\r");
    eprintln!("{} finished", elapsed(start));
    let dt = elapsed(start);
    eprintln!(
        "{p} ms per multiplication for dims n={dim_n} k={dim_k} m={dim_m} and {N_REPEATS} repeats",
        p = 1000. * dt / N_REPEATS as f32
    );
}

fn run_matmul_metal(
    output: &mut [f32],
    input_w: &[f32],
    input_x: &[f32],
    dim_d: usize,
    dim_n: usize,
    dim_u: usize,
) {
    let device = MTLCreateSystemDefaultDevice().unwrap();
    dbg!(&device, dim_d, dim_n, dim_u);
    dbg!(dim_d * dim_n * dim_u);

    // Library with the functions
    let convert_f16_source = ns_string!(CONVERT_F16_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(convert_f16_source, None /*compilation options*/)
        .unwrap();

    let convert_f32_to_f16_func = library
        .newFunctionWithName(ns_string!("convert_f32_to_f16"))
        .unwrap();

    let convert_f16_to_f32_func = library
        .newFunctionWithName(ns_string!("convert_f16_to_f32"))
        .unwrap();

    let convert_f32_to_f16_pso = device
        .newComputePipelineStateWithFunction_error(&convert_f32_to_f16_func)
        .unwrap();

    let convert_f16_to_f32_pso = device
        .newComputePipelineStateWithFunction_error(&convert_f16_to_f32_func)
        .unwrap();
    // End library with the functions

    let command_queue = device.newCommandQueue().unwrap();

    let input_w_elem = input_w.len(); // len_layer_bytes = dim_d * dim_n * size_of::<f32>();
    //
    // Prepare W input matrix.
    let buf_input_f32_w = unsafe { new_shared_mtl_buffer(&device, &input_w) };
    let target_buf_f16 = unsafe { new_private_mtl_buffer(&device, input_w.len() * SIZE_OF_F16) };

    // Convert weights to f16, once, into a private buffer. In the "real" code, the original f32
    // input should be deallocated.
    execute_func_over_array(
        &command_queue,
        &convert_f32_to_f16_pso,
        input_w.len(),
        &buf_input_f32_w,
        &target_buf_f16,
    );

    // Prepare X input vector.
    let buf_input_x_f32 = unsafe { new_shared_mtl_buffer(&device, &input_x) };
    let buf_input_x_f16 = unsafe { new_private_mtl_buffer(&device, &input_x.len() * SIZE_OF_F16) };
    execute_func_over_array(
        &command_queue,
        &convert_f32_to_f16_pso,
        input_x.len(),
        &buf_input_x_f32,
        &buf_input_x_f16,
    );

    // Prepare output vector. We should have such a f16 vector for each f32 vector. Maybe we can
    // use
    let buf_output_f16 = unsafe { new_private_mtl_buffer(&device, output.len() * SIZE_OF_F16) };

    // TODO output to the xout (mind the stride) as f16
    // TODO convert to f32 output in place (mind the stride)

    //for i in 0..N_REPEATS {
    //TODO convert X and OUT in each iteration.

    assert_eq!(input_w_elem, dim_d * dim_n * N_LAYERS);
    let mat_w = unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_d as NSUInteger,
            dim_n as NSUInteger,
            dim_n * SIZE_OF_F16 as NSUInteger,
            MPSDataType::Float16,
        );
        MPSMatrix::initWithBuffer_offset_descriptor(
            mat,
            &buf_input_x_f16,
            0, // input_w_len_layer_bytes * (i % N_LAYERS),
            &desc,
        )
    };

    let mat_x = unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_n as NSUInteger,
            dim_u as NSUInteger,
            dim_u * SIZE_OF_F16 as NSUInteger,
            MPSDataType::Float16,
        );
        MPSMatrix::initWithBuffer_descriptor(mat, &buf_input_x_f16, &desc)
    };

    let mat_out = unsafe {
        let mat = MPSMatrix::alloc();
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            dim_d as NSUInteger,
            dim_u as NSUInteger,
            dim_u * SIZE_OF_F16 as NSUInteger,
            MPSDataType::Float16,
        );

        MPSMatrix::initWithBuffer_descriptor(mat, &buf_output_f16, &desc)
    };

    //    eprint!("\rmetal {i}  ");
    let command_buffer = command_queue.commandBuffer().unwrap();
    // encoder is needed to encode the function invocations.
    let compute_encoder = command_buffer.computeCommandEncoder().unwrap();

    let matmul = unsafe {
        let matmul_alloc = MPSMatrixMultiplication::alloc();
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            matmul_alloc,
            &device,
            false, // transpose
            false,
            dim_d, // rows and cols
            dim_u,
            dim_n,
            1.0, // alpha, beta
            0.0,
        )
    };
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
    //}
}

fn execute_func_over_array(
    command_queue: &RetainedMTLCommandQueue,
    convert_func: &RetainedMTLComputePipelineState,
    n_elem: usize,
    source: &RetainedMTLBuffer,
    target: &RetainedMTLBuffer,
) {
    let command_buffer = command_queue.commandBuffer().unwrap();
    let compute_encoder = command_buffer.computeCommandEncoder().unwrap();

    compute_encoder.setComputePipelineState(&convert_func);
    unsafe {
        compute_encoder.setBuffer_offset_atIndex(Some(&source), 0, 0);
        compute_encoder.setBuffer_offset_atIndex(Some(&target), 0, 1);
    }
    let (grid_size, thread_group_size) =
        grid_and_thread_group_size_for_linear_op(n_elem, convert_func);

    compute_encoder.dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);
    compute_encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
}

fn grid_and_thread_group_size_for_linear_op(
    array_len_elem: usize,
    func: &RetainedMTLComputePipelineState,
) -> (MTLSize, MTLSize) {
    let grid_size = MTLSize {
        width: array_len_elem,
        height: 1,
        depth: 1,
    };

    let max_threads_per_group = func.maxTotalThreadsPerThreadgroup();
    let width = if max_threads_per_group > array_len_elem {
        array_len_elem
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

unsafe fn new_private_mtl_buffer_from_slice(
    device: &RetainedMTLDevice,
    command_queue: &RetainedMTLCommandQueue,
    buf: &[f32],
) -> RetainedMTLBuffer {
    let shared_buf = unsafe { new_shared_mtl_buffer(&device, buf) };
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

unsafe fn new_shared_mtl_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    buf: &[f32],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
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

unsafe fn new_private_mtl_buffer(
    device: &RetainedMTLDevice,
    buf_size_bytes: usize,
) -> RetainedMTLBuffer {
    device
        .newBufferWithLength_options(buf_size_bytes, MTLResourceOptions::StorageModePrivate)
        .unwrap()
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
    //(0..n).map(|i| 0.01_f32 * i as f32).collect()
    (0..n).map(|i| 1_f32).collect()
}

fn elapsed(start: SystemTime) -> f32 {
    SystemTime::now()
        .duration_since(start)
        .unwrap()
        .as_secs_f32()
}
