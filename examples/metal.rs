//! A rewrite of the following example into Rust:
//! - https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu
//!
//! Some other useful links:
//! - https://docs.rs/objc2-metal/latest/objc2_metal/
//! - https://github.com/madsmtm/objc2/blob/main/examples/metal/triangle/main.rs

use std::{ffi::c_void, ptr::NonNull};

use objc2::{
    self,
    rc::Retained,
    runtime::{Protocol, ProtocolObject},
};
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

const ADD_ARRAYS_SOURCE: &str = include_str!("./add.metal");

fn main() {
    // Get a GPU
    let device = MTLCreateSystemDefaultDevice().unwrap();
    dbg!(&device);

    let add_arrays_source = ns_string!(ADD_ARRAYS_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(add_arrays_source, None /*compilation options*/)
        .unwrap();
    dbg!(&library);

    let add_arrays_func = library
        .newFunctionWithName(ns_string!("add_arrays"))
        .unwrap();
    println!("add_arrays_func: {add_arrays_func:?}");
    dbg!(&add_arrays_func);

    let add_arrays_pso = device
        .newComputePipelineStateWithFunction_error(&add_arrays_func)
        .unwrap();
    dbg!(&add_arrays_pso);

    let command_queue = device.newCommandQueue().unwrap();
    dbg!(&command_queue);

    let input_a: Vec<f32> = vec![1., 2., 3.];
    let input_b: Vec<f32> = vec![10., 20., 30.];
    let output: Vec<f32> = vec![99.; 3];
    let array_len = input_a.len();
    assert_eq!(input_a.len(), input_b.len());
    assert_eq!(input_b.len(), output.len());

    // This would allocate a new buffer.
    //let buf_a = device
    //    .newBufferWithLength_options(
    //        input_a.len() * size_of::<f32>(),
    //        MTLResourceOptions::StorageModeShared,
    //    )
    //    .unwrap();

    // This uses an existing buffer in a shared memory.
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
    dbg!(&buf_a);
    //unsafe { copy_slice_into_mtlbuffer(input_a.as_slice(), &buf_a) };

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
    dbg!(&buf_b);
    //unsafe { copy_slice_into_mtlbuffer(input_b.as_slice(), &buf_b) };

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

    let command_buffer = command_queue.commandBuffer().unwrap();
    dbg!(&command_buffer);

    let compute_encoder = command_buffer.computeCommandEncoder().unwrap();
    dbg!(&compute_encoder);

    compute_encoder.setComputePipelineState(&add_arrays_pso);
    unsafe { compute_encoder.setBuffer_offset_atIndex(Some(&buf_a), 0, 0) };
    unsafe { compute_encoder.setBuffer_offset_atIndex(Some(&buf_b), 0, 1) };
    unsafe { compute_encoder.setBuffer_offset_atIndex(Some(&buf_out), 0, 2) };

    let grid_size = MTLSize {
        width: array_len,
        height: 1,
        depth: 1,
    };

    let thread_group_size: MTLSize;
    {
        let max_threads_per_group = add_arrays_pso.maxTotalThreadsPerThreadgroup();
        dbg!(max_threads_per_group);
        let width = if max_threads_per_group > array_len {
            array_len
        } else {
            max_threads_per_group
        };
        thread_group_size = MTLSize {
            width,
            height: 1,
            depth: 1,
        };
    }
    dbg!(thread_group_size);

    dbg!("before execution", &output);

    compute_encoder.dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);
    compute_encoder.endEncoding();
    command_buffer.commit();
    dbg!(command_buffer.status());
    command_buffer.waitUntilCompleted();
    dbg!(command_buffer.status());

    //dbg!(input_a);
    unsafe {
        let ptr = buf_a.contents();
        let ptr = ptr.as_ptr() as *const f32;
        let raw_a = std::slice::from_raw_parts(ptr, array_len);
        dbg!(raw_a);
    }

    //dbg!(input_b);
    unsafe {
        let ptr = buf_b.contents();
        let ptr = ptr.as_ptr() as *const f32;
        let raw_b = std::slice::from_raw_parts(ptr, array_len);
        dbg!(raw_b);
    }

    dbg!("after execution", &output);
    unsafe {
        let ptr = buf_out.contents();
        let ptr = ptr.as_ptr() as *const f32;
        let raw_out = std::slice::from_raw_parts(ptr, array_len);
        dbg!(raw_out);
    }
}

// unsafe fn copy_slice_into_mtlbuffer(source: &[f32], target: &ProtocolObject<dyn MTLBuffer>) {
//     let target_ptr = target.contents();
//     let target_ptr = target_ptr.as_ptr() as *mut f32;
//     let target_slice: &mut [f32] =
//         unsafe { std::slice::from_raw_parts_mut(target_ptr, source.len()) };
//     for i in 0..(source.len()) {
//         // This probably adds unneeded bound checks.
//         target_slice[i] = source[i];
//     }
// }

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

// todo add N argument
// todo matmul
// todo output not needed - reserved?
