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
flowchart TD

add_fnn_swiglu(("+"))
add_residual(("+"))
attention("Attention <br> N heads <br> (parallel)")
embedding("Embedding <br> for input token")
fnn_swiglu("FNN w. SwiGLU <br> W1, W2, W3, Wfnn")
layer_2("Next layer...")
layer_3("Final layer")
logits_sampler_decoder("RMS norm, Wcls <br> Logits <br> Sampler <br> Decoder")
query_key("Query <br> Key cache <br> Wq, Wk")
rms_norm_emb("RMS norm")
rope(["RoPE"])


embedding --> rms_norm_emb
embedding--"Residual <br> connection"-->add_residual

%%%% layer internals %%%%
subgraph "Layer"

rms_norm_emb --> query_key
query_key --> rope
rope --> attention
rms_norm_emb --"Value cache <br> Wv"--> attention

attention=="Wo"==>add_residual

add_residual --> fnn_swiglu
fnn_swiglu --> add_fnn_swiglu
add_residual --> add_fnn_swiglu

end
%%%% end layer internals %%%%

add_fnn_swiglu --> layer_2
layer_2 --"..."--> layer_3
layer_3 --> logits_sampler_decoder



%% styles
classDef dashed fill:#fff,stroke:#000,stroke-dasharray: 5 5;
classDef fat stroke-width: 3px;
class attention fat

```

##  Detailed

The detailed flow diagram based directly on the code:


```mermaid
flowchart TD

embedding("embedding <br/> [seq_len x dim] <br/> (tok)")
embedding --> x_1
x_1("x <br/> [dim]")

%% subgraph For each layer

rmsnorm_1(("RMSNorm <br/> [dim]"))
rms_att_weight("RMS att. weight <br/> [L x dim] <br/> (layer)")
xb_1("xb <br/> [dim]")

%% This is for every layer    


x_1 -->rmsnorm_1
rms_att_weight --> rmsnorm_1
rmsnorm_1 --> xb_1

w_q("Wq <br/> [L x dim x dim] <br/> (layer)")
query("query <br/> [dim]")
w_q --> matmul_q(("@"))
xb_1 ---> matmul_q
matmul_q --> query

key("key cache <br/> [L x seq_len x kv_dim] (layer, pos)")
w_k("Wk <br/> [L x dim x kv_dim] <br/> (layer)")
w_k --> matmul_k(("@"))
xb_1 ---> matmul_k
matmul_k --> key

w_v("Wv <br/> [L x dim x kv_dim] <br/> (layer)")
value("value cache <br/> [L x seq_len x kv_dim] <br/> (layer, pos)
")
w_v --> matmul_v(("@"))
xb_1 ---> matmul_v
matmul_v --> value

rope_q(["RoPE (pos)"])
query --> rope_q

rope_k(["RoPE (pos)"])
key --> rope_k

%% This is for every attention head
rope_q --[n_heads x head_size] (head)--> att_q
rope_k --[???]--> att_k
value --[???]--> att_v
subgraph Each attention head
    att_q("query slice") --> att_dot_1
    att_k("key slice") --> att_dot_1
    att_dot_1(["dot"]) --> att
    att("attention scores") --> att_softmax
    att_softmax(["softmax"])
    att_v("value slice")
    att_softmax-->att_dot_2
    att_v-->att_dot_2
    att_dot_2(["dot"]) --> att_xb
    att_xb["xb <br/> [dim]"]
end

att_xb-->matmul_xb2(("@"))
w_o("Wo </br> [L x dim x dim] </br> (layer)")-->matmul_xb2
matmul_xb2-->xb2
xb2("xb2 <br/> [dim]")-->add_x_xb2
x_1-->add_x_xb2
x_2("x' <br/> [dim]")
add_x_xb2(("+"))-->x_2

fnn_rmsnorm(("RMSNorm <br/> [dim]"))
rms_fnn_w("Wfnn <br/> [L x dim] <br/> (layer)")-->fnn_rmsnorm
x_2-->fnn_rmsnorm
xb_2("xb' <br/> [dim]")
fnn_rmsnorm-->xb_2
xb_2-->fnn_matmul_hb
xb_2-->fnn_matmul_hb2

subgraph FNN with SwiGLU
    fnn_matmul_hb(("@"))
    fnn_hb("hb <br/> [hidden_dim]")
    w1("W1 <br/> [L x dim x hidden_dim] <br/> (layer)")
    
    w1 --> fnn_matmul_hb
    fnn_matmul_hb-->fnn_hb

    fnn_matmul_hb2(("@"))
    fnn_hb2("hb2 <br/> [hidden_dim]")
    w3("W3 <br/> [L x dim x hidden_dim] <br/> (layer)")
    
    w3 --> fnn_matmul_hb2
    fnn_matmul_hb2-->fnn_hb2

    fnn_hb_2("hb' <br> [hidden_dim]")
    swiglu_hb((SwiGLU))
    fnn_hb --> swiglu_hb
    fnn_hb2 --> swiglu_hb
    swiglu_hb --> fnn_hb_2
    fnn_hb_2
end 

xb_3("xb'' <br> [dim]")
w2("W2 <br> [L x dim x hidden_dim] <br> (layer)")
fnn_matmul_final(("@"))
w2 --> fnn_matmul_final
fnn_hb_2 --> fnn_matmul_final
fnn_matmul_final-->xb_3

x_2 --> add_x_xb_3
xb_3 --> add_x_xb_3
x_3("x'' <br> [dim]")
add_x_xb_3(("+"))
add_x_xb_3 --> x_3

x_3 == "LOOP FOR EACH LAYER <br> x'' is x in next iteration" ==> x_1

%% end for each layer

%% outside loop
rms_final_weight("RMS final <br> weight <br> [dim]") --> rmsnorm_final
rmsnorm_final(("RMSNorm"))
x_4("x''' <br> [dim]")
x_3 =="After final layer"==> rmsnorm_final
rmsnorm_final --> x_4

wcls("Wcls <br> [dim x vocab_size]")
logits("logits <br> [vocab_size]")
matmul_logit(("@"))
x_4 --> matmul_logit
wcls --> matmul_logit
matmul_logit --> logits
logits --> sampler
sampler --> decoder

```


