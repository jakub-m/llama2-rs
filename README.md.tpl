This is a rewrite of [llama2.c][1] to Rust. This code is just for learning purposes, it is not maintained.

To run tests you need `tokenizer.bin` from the [original repo][1].

[1]: https://github.com/karpathy/llama2.c

# Learnings
#
- Using Rust-ish type (like `TokenId` instead of `usize`) helps understanding the code.

- ChatGPT was _very useful_ in learning `lldb` commands.

- [mermaid](https://mermaid.live) is an absolutely fantastic tool for diagrams.

- If you don't keep Mmap object alive, it will be dropped and accessing the pointed data will result in a segfault.

- Use `RAYON_NUM_THREADS=1` for sequential execution (good for debugging).

- Just slapping `rayon` and  [`par_iter`][par_iter] will not magicly speed up
  the processing. It turned out that the processors underutilised. I suppose
  it's because the program does the computation quite sequentially, and if I
  use `par_iter` in matmul for small matrices, the overhead is significant.

- Use `cargo objdump --bin llama2-rs -- -d -S ./target/debug/llama2-rs` to see
  the assembly output, e.g. to see if the code is vectoried.

- Using a single array of values for multi-dimensional structures can be
  tricky. Adding extra asserts here and there to see if we didn't pass a slice
  that has unexpected size (where we know the size), saves a lot of trouble.


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

- 34.5 tok/s  - no parallelization, no rayon, sequential as it can be. 44fce5a
- 132.2 tok/s - with naive use of rayon and par_iter. 8eda5d5
