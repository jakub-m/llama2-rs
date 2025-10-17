# llama2-rs

This is a rewrite of [llama2.c][llama2_c] to Rust and Apple M2 Metal GPU, with
a bit of [Rayon][rayon]. This code is just for learning purposes, it is not
maintained. Below are the learnings.

## Prerequisites

You need `tokenizer.bin` from the original llama2.c repo, also you need the
[TinyStories][TinyStories] model, or a full llama2 7B model. See
[llama2.c][llama2_c] on how to download and convert the model.

[rayon]: https://github.com/rayon-rs/rayon
[TinyStories]: https://huggingface.co/datasets/roneneldan/TinyStories
[llama2_c]: https://github.com/karpathy/llama2.c



# Diagrams

Here are some diagrams that I draw along the implementation to understand what
the code actually does. Those diagrams later helped me to figure which matrix
multiplications can be run independently.


## High level

```mermaid
$diagram_highlevel
```

##  Detailed

The detailed flow diagram based directly on the code:


```mermaid
$diagram_details
```


# Learnings

Rust:

- Using Rust type (like `TokenId` instead of `usize`) helps to understand the
  code.

- Use `assert!` a lot. When you have matrices or vectors that assume some
  sizing, add assertions to find the bugs early. Most of this code deals with
  buffers of different sizes, and the sizing is often the only thing that you
  can assume. An assertion firing was several times the only thing that hinted
  me that I confused dimensions.

- If you don't keep [Mmap][https://docs.rs/memmap2/latest/memmap2/] object
  alive, it will be dropped and accessing the pointed data will result in a
  segfault.

- Use `cargo objdump --bin llama2-rs -- -d -S ./target/debug/llama2-rs` to see
  the assembly output, e.g. to see if the code is vectorised.

- You can check if the generated code uses [SIMD and vectorization][vfma_rust]
  by inspecting disassembled code, and looking for [SIMD
  instructions][vfma_arm] (that [start with "f" like `fmul`][fmul]):

  ```bash
  make objdump-llama | egrep '^000| \tf' | grep llama2_rs -A1 
  
  0000000100005494 <_llama2_rs::main::h9ba2e6463cb6eab5>:
  100005710: 1e202008    	fcmp	s0, #0.0
  ```

[vfma_rust]: https://doc.rust-lang.org/core/arch/aarch64/fn.vfma_n_f32.html
[vfma_arm]: https://developer.arm.com/architectures/instruction-sets/intrinsics/vfma_n_f32
[fmul]: https://developer.arm.com/documentation/ddi0602/2025-06/SIMD-FP-Instructions/FMUL--by-element---Floating-point-multiply--by-element--


[Rayon][rayon]:

- Just slapping `rayon` and  [`par_iter`][par_iter] tremendously speeds up the
  code.

- Use `RAYON_NUM_THREADS=1` for sequential execution, good for debugging.

[par_iter]: https://docs.rs/rayon/latest/rayon/iter/index.html


Metal:

- Just slapping GPU at the problem (this problem at least) will not make it
  magically faster.

- I didn't see much difference in performance when using shared or private GPU
  memory buffers. Maybe it's because of specific access patterns of the
  program.


Float16:

- 65520 is already an infinity in f16.

- Apple M2 Metal does not support matrix multiplication on BFloat16:

  ```text
  MPSMatrixMultiplication.mm:3260: failed assertion `Input data type must be one
  of MPSDataTypeFloat32, MPSDataTypeFloat16, MPSDataTypeInt8, or
  MPSDataTypeInt16.'
  ```

- When doing matmul on f16, you need the output to be f32, otherwise the output
  will corrupt. Fortunately, Metal supports matmul for f16 at input and f32 at output.

- Using f16 was _slower_ for TinyStories than f32.


Other:

- ChatGPT was _very useful_ in learning `lldb` commands.

- [mermaid](https://mermaid.live) is an absolutely fantastic tool for diagrams.

- Monitor if you memory does not start to swap , with `sysctl vm.swapusage` or
  Activity Monitor. If mem swaps, the computation will instantly become dog
  slow.


# Benchmarking Rayon

I didn't run very rigorous benchmarks, rather run inference on TinyStories 42M
model, and also on llama2 7B model.

- 34 tok/s - TinyStories, no parallelization, no rayon, sequential as it can
  be. [commit](https://github.com/jakub-m/llama2-rs/commit/44fce5a)

- 27 tok/s - using naively [par_bridge][par_bridge]
  [commit](https://github.com/jakub-m/llama2-rs/commit/f4d9041). Slapping
  naively `par_bridge` is slower that sequential execution on a single code. I
  suppose it's the overhead of coordination of those small work chunks.

- 132 tok/s - with naive use of rayon and [par_iter][par_iter].
  [commit](https://github.com/jakub-m/llama2-rs/commit/8eda5d5). Rayon is
  awesome!

We can further tweak Rayon to take larger chunks of work, to spend less on
coordination. Here are the results of modifying [`with_min_len`][with_min_len]
parameter:

- 1 - 132 tok/s
- 5 - 146 tok/s
- 10 - 153 tok/s
- 15 - 154 tok/s - best `with_min_len` value. [commit](https://github.com/jakub-m/llama2-rs/commit/b596ff4) 
- 20 - 150 tok/s
- 40 - 142 tok/s
- 70 - 115 tok/s 
- 150 - 93 tok/s

FTR, the machine I use has [available parallelism][available_parallelism] is 12
(8 performance and 4 efficiency cores).

[par_bridge]: https://docs.rs/rayon/latest/rayon/iter/trait.ParallelBridge.html
[par_iter]: https://docs.rs/rayon/1.11.0/rayon/iter/index.html
[with_min_len]: https://docs.rs/rayon/1.11.0/rayon/iter/trait.IndexedParallelIterator.html#method.with_min_len
[available_parallelism]: https://doc.rust-lang.org/std/thread/fn.available_parallelism.html


# Metal

- [metal_add.rs](examples/metal_add.rs) implements a simple addition in GPU using a shared memory buffer.
- [metal_matmul.rs](examples/metal_matmul.rs) runs matrix multiplication on GPU.

Running llama2 (`make run-napalm`) with matmul naively computed in GPU (shared
memory buffers, Metal native matmul) yields ~20% GPU utilization, and ~60
seconds per token. For CPU with Rayon, it's ~20 sec per token.

# Next steps

- GPU: Use private GPU memory for W matrices that don't change. Check first
  with benchamrking if this gives better yield.
    - https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
    - https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/ResourceOptions.html

- Pipelining of GPU work, do not wait until finished, carry on in parallel when
  possible.

- Run comput. in GPU and CPU at once (2x yield!)



# Materials

- [From Multi-Head to Latent Attention: The Evolution of Attention Mechanisms](https://vinithavn.medium.com/from-multi-head-to-latent-attention-the-evolution-of-attention-mechanisms-64e3c0505f24)
- [Positional Embeddings in Transformers: A Math Guide to RoPE & ALiBi](https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/)
- [LLM Embeddings Explained: A Visual and Intuitive Guide](https://huggingface.co/spaces/hesamation/primer-llm-embedding)
- [SwiGLU](https://medium.com/@s_boudefel/exploring-swiglu-the-activation-function-powering-modern-llms-9697f88221e7)
- [Metal: Setting up a command structure](https://developer.apple.com/documentation/metal/setting-up-a-command-structure) 
