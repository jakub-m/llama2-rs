# llama2-rs

This is a rewrite of [llama2.c][1] to Rust. This code is just for learning
purposes, it is not maintained.

To run tests you need `tokenizer.bin` from the [original repo][1].

[1]: https://github.com/karpathy/llama2.c

# Learnings


Rust:

- Using Rust-ish type (like `TokenId` instead of `usize`) helps understanding the code.

- If you don't keep Mmap object alive, it will be dropped and accessing the pointed data will result in a segfault.

- Use `cargo objdump --bin llama2-rs -- -d -S ./target/debug/llama2-rs` to see
  the assembly output, e.g. to see if the code is vectoried.

- Using a single array of values for multi-dimensional structures can be
  tricky. Adding extra asserts here and there to see if we didn't pass a slice
  that has unexpected size (where we know the size), saves a lot of trouble.


Rayon:

- Use `RAYON_NUM_THREADS=1` for sequential execution (good for debugging).

- Just slapping `rayon` and  [`par_iter`][par_iter] speeds up the code.

Metal:

- Just slapping GPU at the problem (this problem at least) will not make it
  magically faster.

- I didn't see much difference in performance when using shared or private GPU
  memory buffers. Maybe it's because of specific access patterns of the
  program.


Other:

- ChatGPT was _very useful_ in learning `lldb` commands.

- [mermaid](https://mermaid.live) is an absolutely fantastic tool for diagrams.

- Monitor if you memory does not start to swap , with `sysctl vm.swapusage` or
  -Activity Monitor. Your computation will instantly become dog slow.

- 65520 is already an infinity in `f16`.


[par_iter]: https://docs.rs/rayon/latest/rayon/iter/index.html

# Materials
- [From Multi-Head to Latent Attention: The Evolution of Attention Mechanisms](https://vinithavn.medium.com/from-multi-head-to-latent-attention-the-evolution-of-attention-mechanisms-64e3c0505f24)
- [Positional Embeddings in Transformers: A Math Guide to RoPE & ALiBi](https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/)
- [LLM Embeddings Explained: A Visual and Intuitive Guide](https://huggingface.co/spaces/hesamation/primer-llm-embedding)
- [SwiGLU](https://medium.com/@s_boudefel/exploring-swiglu-the-activation-function-powering-modern-llms-9697f88221e7)

# Diagrams

## High level

```mermaid
$diagram_highlevel
```

##  Detailed

The detailed flow diagram based directly on the code:


```mermaid
$diagram_details
```

# Benchmarking

Run on "tiny stories" 42M model

- 34 tok/s  - no parallelization, no rayon, sequential as it can be. [commit](https://github.com/jakub-m/llama2-rs/commit/44fce5a)

- 27 tok/s - using naively [par_bridge][par_bridge] [commit](https://github.com/jakub-m/llama2-rs/commit/f4d9041)

Slapping naively `par_bridge` is slower that sequential execution on a single
code. I suppose it's the overhead of coordination of those small work chunks.

- 132 tok/s - with naive use of rayon and [par_iter][par_iter]. [commit](https://github.com/jakub-m/llama2-rs/commit/8eda5d5)

Perventing rayon allocating to small work chunks with
[`with_min_len`][with_min_len] yields some benefits:
- 1 - 132 tok/s
- 5 - 146 tok/s
- 10 - 153 tok/s
- 15 - 154 tok/s - best `with_min_len` value. [commit](https://github.com/jakub-m/llama2-rs/commit/b596ff4) 
- 20 - 150 tok/s
- 40 - 142 tok/s
- 70 - 115 tok/s 
- 150 - 93 tok/s

For my machine [available parallelism][available_parallelism] is 12 (8 performance and 4 efficiency cores).

[par_bridge]: https://docs.rs/rayon/latest/rayon/iter/trait.ParallelBridge.html
[par_iter]: https://docs.rs/rayon/1.11.0/rayon/iter/index.html
[with_min_len]: https://docs.rs/rayon/1.11.0/rayon/iter/trait.IndexedParallelIterator.html#method.with_min_len
[available_parallelism]: https://doc.rust-lang.org/std/thread/fn.available_parallelism.html

To speed the inference up I thought about [SIMD and vectorization][vfma_rust],
but the code seems to be vectorised already: It seems that the code is
auto-vectorised already. In the disasembled output I see those [SIMD
instructions][vfma_arm] that [start with "f"][fmul]:

```bash
make objdump-llama | egrep '^000| \tf' | grep llama2_rs -A1 

0000000100005494 <_llama2_rs::main::h9ba2e6463cb6eab5>:
100005710: 1e202008    	fcmp	s0, #0.0
````

[vfma_rust]: https://doc.rust-lang.org/core/arch/aarch64/fn.vfma_n_f32.html
[vfma_arm]: https://developer.arm.com/architectures/instruction-sets/intrinsics/vfma_n_f32
[fmul]: https://developer.arm.com/documentation/ddi0602/2025-06/SIMD-FP-Instructions/FMUL--by-element---Floating-point-multiply--by-element--

# Metal

- [metal_add.rs](examples/metal_add.rs) implements a simple addition in GPU using a shared memory buffer.
- [metal_matmul.rs](examples/metal_matmul.rs) runs matrix multiplication on GPU.
- https://developer.apple.com/documentation/metal/setting-up-a-command-structure

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

