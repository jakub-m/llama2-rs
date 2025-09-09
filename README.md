This is a rewrite of [llama2.c][1] to Rust. This code is just for learning purposes, it is not maintained.

To run tests you need `tokenizer.bin` from the [original repo][1].

[1]: https://github.com/karpathy/llama2.c


# Learnings

- Using Rust-ish type (like `TokenId` instead of `usize`) helps understanding the code.
- ChatGPT was _very useful_ in learning `lldb` commands.

# Materials
- [From Multi-Head to Latent Attention: The Evolution of Attention Mechanisms](https://vinithavn.medium.com/from-multi-head-to-latent-attention-the-evolution-of-attention-mechanisms-64e3c0505f24)
- [Positional Embeddings in Transformers: A Math Guide to RoPE & ALiBi](https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/)
- [LLM Embeddings Explained: A Visual and Intuitive Guide](https://huggingface.co/spaces/hesamation/primer-llm-embedding)

