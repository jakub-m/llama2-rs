use clap::Parser;
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use std::{
    fs::{File, metadata},
    io::{BufReader, Read},
    rc::Rc,
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "debug")]
#[macro_export]
macro_rules! debug_eprintln {
    ($($arg:tt)*) => {
        {
            eprintln!($($arg)*);
        }
    };
}

#[cfg(feature = "debug")]
#[macro_export]
macro_rules! debug_eprint {
    ($($arg:tt)*) => {
        {
            eprint!($($arg)*);
        }
    };
}

#[cfg(not(feature = "debug"))]
#[macro_export]
macro_rules! debug_eprintln {
    ($($arg:tt)*) => {
        /* nop */
    };
}

#[cfg(not(feature = "debug"))]
#[macro_export]
macro_rules! debug_eprint {
    ($($arg:tt)*) => {
        /* nop */
    };
}

#[derive(Clone)]
struct TokenIndex {
    /// Refernce to string in [Tokenizer::vocab] vector.
    token_str: Rc<String>,
    /// position in [Tokenizer::vocab] vector.
    id: usize,
}

struct Tokenizer {
    // NOTE: use ascii string (char) instead of unicode str.
    vocab: Vec<Rc<String>>,
    vocab_scores: Vec<f32>,
    /// sorted_vocab refers to vocab strings.
    sorted_vocab: Vec<TokenIndex>,
    vocab_size: usize,
    max_token_length: usize,
    // NOTE: use memory-continuous representation of those byte pieces, with simpler strings.
    byte_pieces: [Box<String>; 256],
}

const TOK_UNK: usize = 0;
const TOK_BOS: usize = 1;
const TOK_EOS: usize = 2;

impl Tokenizer {
    fn build(path: &str, vocab_size: usize) -> Tokenizer {
        let byte_pieces: [Box<String>; 256] =
            std::array::from_fn(|i| Box::new((i as u8 as char).to_string()));
        let mut vocab: Vec<Rc<String>> = (0..vocab_size).map(|_| Rc::new("".to_string())).collect();
        let mut vocab_scores = vec![0f32; vocab_size];
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);

        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).unwrap();
        let max_token_length = i32::from_le_bytes(buf) as usize;

        for i in 0..vocab_size {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).unwrap();
            let score = f32::from_le_bytes(buf);
            vocab_scores[i] = score;

            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).unwrap();
            let len: i32 = i32::from_le_bytes(buf);

            let mut buf: Vec<u8> = vec![0u8; len as usize];
            reader.read_exact(&mut buf).unwrap();
            let s = String::from_utf8(buf).unwrap();
            #[cfg(feature = "debug-encoder")]
            debug_eprintln!("{i}: score={score} len={len} s={s:?}");
            vocab[i] = Rc::new(s);
        }

        assert_eq!(
            vocab.get(TOK_UNK).unwrap().as_str(),
            "<unk>",
            "expected UNK token"
        );
        assert_eq!(
            vocab.get(TOK_BOS).unwrap().as_str(),
            "\n<s>\n",
            "expected BOS token"
        );
        assert_eq!(
            vocab.get(TOK_EOS).unwrap().as_str(),
            "\n</s>\n",
            "expected EOS token"
        );

        let mut sorted_vocab: Vec<TokenIndex> = (0..vocab_size)
            .map(|id| TokenIndex {
                token_str: Rc::clone(&vocab[id]),
                id,
            })
            .collect();
        sorted_vocab.sort_by(|a, b| a.token_str.partial_cmp(&b.token_str).unwrap());

        //for s in &sorted_vocab {
        //    debug_eprintln!("sorted_vocab {}, {}", s.token_str, s.id);
        //}

        Tokenizer {
            vocab_size,
            vocab,
            vocab_scores,
            sorted_vocab,
            byte_pieces,
            max_token_length,
        }
    }

    /// encode the string text (input) into an upper-bound preallocated tokens[] array
    /// bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    fn encode(&self, text: &str, add_bos: AddBos, add_eos: AddEos) -> Vec<usize> {
        // void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)

        // First, encode codepoint by codepoint, later merge succesive encoded tokens using the "best"
        // match.
        let mut tokens: Vec<usize> = Vec::with_capacity(text.len() + 3); // max. possible capacity, +3 for '\0', ?BOS, ?EOS

        if let AddBos::Yes = add_bos {
            tokens.push(TOK_BOS)
        }

        if let AddEos::Yes = add_eos {
            tokens.push(TOK_EOS)
        }

        // Original comment:
        //   add_dummy_prefix is true by default
        //   so prepend a dummy prefix token to the input string, but only if text != ""
        //   pretty sure this isn't correct in the general case but I don't have the
        //   energy to read more of the sentencepiece code to figure out what it's doing
        if !text.is_empty() {
            let dummy_prefix = self.str_lookup(" ").expect("could not find \" \"");
            tokens.push(dummy_prefix.id);
        }
        {
            // Decode codepoint by codepoint.
            let mut tmp = [0u8; 4];
            for c in text.chars() {
                let s = c.encode_utf8(&mut tmp);
                if let Some(tok) = self.str_lookup(&s) {
                    tokens.push(tok.id);
                    #[cfg(feature = "debug-encoder")]
                    debug_eprintln!("encode {} {}", tok.token_str, tok.id);
                } else {
                    // byte_fallback encoding: just encode each byte as a token
                    // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                    // so the individual bytes only start at index 3
                    #[cfg(feature = "debug-encoder")]
                    debug_eprintln!("encode fallback for {s:?}");
                    for b in s.as_bytes() {
                        let i = ((*b + 3) as usize);
                        debug_eprint!(" {b:#x} {i}");
                        tokens.push(i)
                    }
                    #[cfg(feature = "debug-encoder")]
                    debug_eprintln!("");
                }
            }
        }

        #[derive(Debug)]
        struct Best {
            s: String,
            /// Token at this idx will be replaced with tok_id.
            tok_idx: usize,
            tok_id: usize,
            score: f32,
        }

        let mut curr_best: Option<Best> = None;
        loop {
            for tok_idx in (0..tokens.len() - 1) {
                let str0 = self.vocab.get(*tokens.get(tok_idx + 0).unwrap()).unwrap();
                let str1 = self.vocab.get(*tokens.get(tok_idx + 1).unwrap()).unwrap();
                let candidate_str = format!("{}{}", str0, str1);
                if let Some(found) = self.str_lookup(&candidate_str) {
                    let candidate_score = *(self.vocab_scores.get(found.id).unwrap());
                    if let Some(best) = &curr_best {
                        if candidate_score > best.score {
                            curr_best = Some(Best {
                                s: candidate_str,
                                tok_idx,
                                tok_id: found.id,
                                score: candidate_score,
                            });
                        } else {
                            // The candidate is not better so just carry on.
                        }
                    } else {
                        // Just use the candidate as there is nothing to compare with.
                        curr_best = Some(Best {
                            s: candidate_str,
                            tok_idx,
                            tok_id: found.id,
                            score: candidate_score,
                        });
                    }
                } else {
                    // Candidate not found in vocab, so carry on with the next token pair.
                }
            }
            if let Some(best) = curr_best {
                // Replace the token with the best candidate
                *(tokens.get_mut(best.tok_idx).unwrap()) = best.tok_id;
                // Shift all the tokens after the best one to the left by 1, because we merged two
                // tokens into a pair.
                for i in best.tok_idx + 1..tokens.len() - 1 {
                    let t = *(tokens.get(i + 1).unwrap());
                    *(tokens.get_mut(i).unwrap()) = t;
                }
                tokens.pop();
                curr_best = None;
            } else {
                // Failed to merge a pair, time to end.
                break;
            }
        }
        assert!(curr_best.is_none(), "{curr_best:?}");

        // Compare the decoded text and check if it is the same as the original.
        // This test will fail for input containing non-ascii entries (like, new lines), because
        // the encoder uses for \n tokens that are literal <0x0A> (6 characters in angle brackts), instad
        // of a byte of value 0x0a (same applies for other non-ascii characters).
        #[cfg(feature = "debug-encoder")]
        {
            let mut decoded = String::new();
            let mut original = String::new();
            if let AddBos::Yes = add_bos {
                original.push_str("\n<s>\n")
            }
            if !text.is_empty() {
                original.push_str(" ");
            }
            original.push_str(text);
            if let AddEos::Yes = add_eos {
                original.push_str("\n</s>\n")
            }
            debug_eprint!("|");
            for tok_id in &tokens {
                let s = self.vocab.get(*tok_id).unwrap();
                decoded.push_str(s.as_str());
                debug_eprint!("{}|", s);
            }
            debug_eprintln!("");
            assert_eq!(
                &decoded, &original,
                "decoded (left) and original (right) text do not match"
            );
            debug_eprintln!("decoded and original match");
        }

        tokens
    }

    /// find the perfect match for str in vocab
    fn str_lookup(&self, needle: &str) -> Option<TokenIndex> {
        // int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
        let i_found = self
            .sorted_vocab
            .binary_search_by(|probe| probe.token_str.as_str().cmp(needle))
            .ok()?;
        Some(self.sorted_vocab.get(i_found).unwrap().clone())
    }
}

struct Transformer<'a> {
    /// the hyperparameters of the architecture (the blueprint)
    config: Config,
    /// the weights of the model
    weights: TransformerWeights<'a>,
    /// buffers for the "wave" of activations in the forward pass
    // state: RunState,
    //    // some more state needed to properly clean up the memory mapping (sigh)
    //    int fd; // file descriptor for memory mapping
    //    float* data; // memory mapped data pointer
    //    ssize_t file_size; // size of the checkpoint file in bytes
    /// the mmap-ed file with weights.
    file: File,
}

struct TransformerWeights<'a> {
    /// token embedding table
    token_embedding_table: &'a [f32], // (vocab_size, dim)
    /// weights for rmsnorms
    rms_att_weight: &'a [f32], // (layer, dim) rmsnorm weights
    rms_ffn_weight: &'a [f32], // (layer, dim)
    /// weights for matmuls. note dim == n_heads * head_size
    wq: &'a [f32], // (layer, dim, n_heads * head_size)
    wk: &'a [f32],             // (layer, dim, n_kv_heads * head_size)
    wv: &'a [f32],             // (layer, dim, n_kv_heads * head_size)
    wo: &'a [f32],             // (layer, n_heads * head_size, dim)
    /// weights for ffn
    w1: &'a [f32], // (layer, hidden_dim, dim)
    w2: &'a [f32],             // (layer, dim, hidden_dim)
    w3: &'a [f32],             // (layer, hidden_dim, dim)
    /// final rmsnorm
    rms_final_weight: &'a [f32], // (dim,)
    /// (optional) classifier weights for the logits, on the last layer
    wcls: &'a [f32],
}

impl<'a> Transformer<'a> {
    fn build(checkpoint_path: &str) -> Self {
        // void build_transformer(Transformer *t, char* checkpoint_path) {
        let (config, checkpoint) = Self::read_checkpoint(checkpoint_path);
        //let state = RunState::new(&config);
        Transformer {
            config,
            weights: checkpoint.transformer_weights,
            file: checkpoint.file,
        }
    }

    fn new_run_state(&self) -> RunState {
        RunState::new(&self.config)
    }

    fn read_checkpoint<'c>(checkpoint_path: &str) -> (Config, Checkpoint<'c>) {
        let config: Config;
        let shared_weights: bool;
        {
            let file = File::open(checkpoint_path).unwrap();
            let mut file = BufReader::new(file);
            // read in the config header
            let mut buf = [0u8; size_of::<ConfigDeser>()];
            file.read_exact(&mut buf).unwrap();
            let config_deser: ConfigDeser = unsafe { std::mem::transmute(buf) };
            debug_eprintln!("{config_deser:?}");
            shared_weights = config_deser.is_shared_weights();
            config = config_deser.into();
        }

        // memory map the Transformer weights into the data pointer
        let file = File::open(checkpoint_path).unwrap();
        let data = unsafe {
            MmapOptions::new().map_copy_read_only(&file).unwrap() // open in read only mode, equivalent of mmap PROT_READ, MAP_PRIVATE
        };

        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / size_of::<f32>())
        };

        let data = &data[size_of::<ConfigDeser>() / size_of::<f32>()..];
        let transformer_weights = memory_map_weights(&config, data, shared_weights);
        (
            config,
            Checkpoint {
                file,
                transformer_weights,
            },
        )
    }
}

/// current wave of activations
struct RunState {
    /// activation at current time stamp (dim,)
    x: Vec<f32>, //float *x;
    /// activation at current time stamp
    /// but inside a residual branch (dim,)
    xb: Vec<f32>, // float *xb;
    /// an additional buffer just for convenience (dim,)
    xb2: Vec<f32>, //float *xb2;
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb: Vec<f32>, //float *hb;
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>, //float *hb2;
    /// query (dim,)
    q: Vec<f32>, //float *q;
    /// key (dim,)
    // k: Vec<f32>, //float *k; // TODO add k and v to RunState::new()
    /// value (dim,)
    // v: Vec<f32>, //float *v;
    /// buffer for scores/attention values (n_heads, seq_len)
    att: Vec<f32>, //float *att;
    /// output logits
    logits: Vec<f32>, //float *logits;
    /// kv cache (layer, seq_len, dim)
    key_cache: Vec<f32>, //float* key_cache;
    /// kv cache (layer, seq_len, dim)
    value_cache: Vec<f32>, //float* value_cache;
}
impl RunState {
    fn new(p: &Config) -> RunState {
        // void malloc_run_state(RunState* s, Config* p) {
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let x = vec![0_f32; p.dim]; //s->x = calloc(p->dim, sizeof(float));
        let xb = vec![0_f32; p.dim]; //s->xb = calloc(p->dim, sizeof(float));
        let xb2 = vec![0_f32; p.dim]; //s->xb2 = calloc(p->dim, sizeof(float));
        let hb = vec![0_f32; p.hidden_dim];
        let hb2 = vec![0_f32; p.hidden_dim]; //s->hb2 = calloc(p->hidden_dim, sizeof(float));
        let q = vec![0_f32; p.dim]; //s->q = calloc(p->dim, sizeof(float));
        let key_cache = vec![0_f32; p.n_layers * p.seq_len * kv_dim]; //s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
        let value_cache = vec![0_f32; p.n_layers * p.seq_len * kv_dim]; //s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
        let att = vec![0_f32; p.n_heads * p.seq_len]; //s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
        let logits = vec![0_f32; p.vocab_size]; //s->logits = calloc(p->vocab_size, sizeof(float));
        RunState {
            x,
            xb,
            xb2,
            hb,
            hb2,
            q,
            //k,
            //v,
            att,
            logits,
            key_cache,
            value_cache,
        }
    }
}

/// A helper to return mmap-ed memory and the mmap-ed file.
struct Checkpoint<'a> {
    file: File,
    transformer_weights: TransformerWeights<'a>,
}

fn memory_map_weights<'a>(
    p: &Config,
    ptr: &'a [f32],
    shared_weights: bool,
) -> TransformerWeights<'a> {
    let head_size = p.dim / p.n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    let token_embedding_table = ptr; // (vocab_size, dim)
    let ptr: &[f32] = &ptr[(p.vocab_size * p.dim)..];

    // weights for rmsnorms
    let rms_att_weight = ptr; // (layer, dim) rmsnorm weights
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim)..];

    // weights for matmuls. note dim == n_heads * head_size
    let wq = ptr; // (layer, dim, n_heads * head_size)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * (p.n_heads * head_size))..];

    let wk = ptr; // (layer, dim, n_kv_heads * head_size)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * (p.n_kv_heads * head_size))..];

    let wv = ptr; //  (layer, dim, n_kv_heads * head_size)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * (p.n_kv_heads * head_size))..];

    let wo = ptr; // (layer, n_heads * head_size, dim)
    let ptr: &[f32] = &ptr[(p.n_layers * (p.n_heads * head_size) * p.dim)..];

    let rms_ffn_weight = ptr;
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim)..];

    // weights for ffn
    let w1 = ptr; // (layer, hidden_dim, dim)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * p.hidden_dim)..];

    let w2 = ptr; // (layer, dim, hidden_dim);
    let ptr: &[f32] = &ptr[(p.n_layers * p.hidden_dim * p.dim)..];

    let w3 = ptr; // (layer, hidden_dim, dim)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * p.hidden_dim)..];

    // final rmsnorm
    let rms_final_weight = ptr; // (dim,)
    let ptr: &[f32] = &ptr[(p.dim)..];
    let ptr: &[f32] = &ptr[(p.seq_len * head_size / 2)..]; // skip what used to be freq_cis_real (for RoPE)
    let ptr: &[f32] = &ptr[(p.seq_len * head_size / 2)..]; // skip what used to be freq_cis_imag (for RoPE)

    // (optional) classifier weights for the logits, on the last layer
    let wcls: &[f32] = if shared_weights {
        token_embedding_table
    } else {
        ptr
    };

    TransformerWeights {
        token_embedding_table,
        rms_att_weight,
        rms_ffn_weight,
        wq,
        wk,
        wv,
        wo,
        w1,
        w2,
        w3,
        rms_final_weight,
        wcls,
    }
}

/// The Sampler, which takes logits and returns a sampled token
/// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
struct Sampler {
    vocab_size: usize,
    /// buffer used in top-p sampling
    probindex: Vec<ProbIndex>,
    temperature: f32,
    topp: f32,
    rng_state: u128,
}

/// struct used when sorting probabilities during top-p sampling
#[derive(Clone, Default)]
struct ProbIndex {
    prob: f32,
    index: usize,
}

impl Sampler {
    fn new(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u128) -> Self {
        // void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed)
        Sampler {
            vocab_size,
            temperature,
            topp,
            rng_state: rng_seed,
            // buffer only used with nucleus sampling; may not need but it's ~small
            probindex: vec![ProbIndex::default(); vocab_size],
        }
    }
}

/// Definition of Config for deserialization (with i32)
#[repr(C)]
#[derive(Debug)]
struct ConfigDeser {
    /// transformer dimension
    dim: i32,
    /// for ffn layers
    hidden_dim: i32,
    /// number of layers
    n_layers: i32,
    /// number of query heads
    n_heads: i32,
    /// number of key/value heads (can be < query heads because of multiquery)
    n_kv_heads: i32,
    /// vocabulary size, usually 256 (byte-level)
    vocab_size: i32,
    /// max sequence length
    seq_len: i32,
}

impl ConfigDeser {
    fn is_shared_weights(&self) -> bool {
        self.vocab_size > 0
    }
}

#[derive(Debug)]
struct Config {
    /// transformer dimension
    dim: usize,
    /// for ffn layers
    hidden_dim: usize,
    /// number of layers
    n_layers: usize,
    /// number of query heads
    n_heads: usize,
    /// number of key/value heads (can be < query heads because of multiquery)
    n_kv_heads: usize,
    /// vocabulary size, usually 256 (byte-level)
    vocab_size: usize,
    /// max sequence length
    seq_len: usize,
}

impl From<ConfigDeser> for Config {
    fn from(c: ConfigDeser) -> Self {
        let ConfigDeser {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        } = c;
        Config {
            dim: dim as usize,
            hidden_dim: hidden_dim as usize,
            n_layers: n_layers as usize,
            n_heads: n_heads as usize,
            n_kv_heads: n_kv_heads as usize,
            vocab_size: vocab_size.abs() as usize,
            seq_len: seq_len as usize,
        }
    }
}

/// steps - number of steps to run.
fn generate(
    transformer: &Transformer,
    tokenizer: &Tokenizer,
    sampler: &Sampler,
    prompt: String,
    steps: usize,
) {
    let prompt_tokens = tokenizer.encode(&prompt, AddBos::Yes, AddEos::No);
    let mut prompt_tokens = prompt_tokens.iter();
    // start the main loop
    // let mut next = 0_usize;        // will store the next token in the sequence
    let token = *(prompt_tokens.next().unwrap());
    let mut pos: usize = 0; // position in the sequence
    while pos < steps {
        /*
           // forward the transformer to get logits for the next token
           float* logits = forward(transformer, token, pos);

           // advance the state machine
           if (pos < num_prompt_tokens - 1) {
               // if we are still processing the input prompt, force the next prompt token
               next = prompt_tokens[pos + 1];
           } else {
               // otherwise sample the next token from the logits
               next = sample(sampler, logits);
           }
           pos++;

           // data-dependent terminating condition: the BOS (=1) token delimits sequences
           if (next == 1) { break; }

           // print the token as string, decode it with the Tokenizer object
           char* piece = decode(tokenizer, token, next);
           safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
           fflush(stdout);
           token = next;
        */
        todo!("HERE")
    }
    // TODO report achieved tok/s (pos-1 because the timer starts after first iteration)
}

fn forward(transformer: &Transformer, token: usize, pos: usize) {
    let p = &transformer.config;
    let w = &transformer.weights;
    let mut s = transformer.new_run_state();
    let mut x = s.x;
    let dim = p.dim;
    let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    //int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    //int hidden_dim =  p->hidden_dim;
    //int head_size = dim / p->n_heads;

    // copy the token embedding into x
    //float* content_row = w->token_embedding_table + token * dim;
    //memcpy(x, content_row, dim*sizeof(*x));
    //slicecpy(&mut x, &w.token_embedding_table[token * dim..], dim);
    slicecpy(&mut x[..], &w.token_embedding_table[token * dim..], dim);

    for l in 0..p.n_layers {
        // attention rmsnorm
        //rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        rmsnorm(&mut s.xb, &mut x, &w.rms_att_weight[l * dim..], dim);
        // key and value point to the kv cache
        let loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
        //s->k = s->key_cache + loff + pos * kv_dim;
        let s_k = &mut s.key_cache[loff + pos * kv_dim..];
        //s->v = s->value_cache + loff + pos * kv_dim;
        let s_v = &mut s.value_cache[loff + pos * kv_dim..];
        //
        // qkv matmuls for this position
        matmul(&mut s.q, &s.xb, &w.wq[l * dim * dim..], dim, dim);
        // matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s_k, &s.xb, &w.wk[l * dim * kv_dim..], dim, kv_dim);
        // matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        matmul(s_v, &s.xb, &w.wv[l * dim * kv_dim..], dim, kv_dim);
        todo!("HERE");
    }

    // forward all the layers
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
    // calculate sum of squares
    let mut ss: f32 = 0_f32;
    for j in 0..size {
        ss += x[j] * x[j]
    }
    ss /= size as f32;
    ss += 1e-5;
    ss = 1.0_f32 / ss.sqrt();
    // normalize and scale
    for j in 0..size {
        o[j] = weight[j] * (ss * x[j]);
    }
}

/// W (d,n) @ x (n,) -> xout (d,)
/// by far the most amount of time is spent inside this little function
fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    xout.par_iter_mut().enumerate().for_each(|(i, xout_val)| {
        let mut val: f32 = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        *xout_val = val;
    });
}

//fn slicecpy<T: Clone>(target: &mut Vec<T>, source: &[T], n_t: usize) {
//    let capacity_init = target.capacity();
//    target.clear();
//    target.extend_from_slice(&source[..n_t]);
//    assert_eq!(
//        capacity_init,
//        target.capacity(),
//        "Error! the vector was reallocated but it should not have"
//    );
//}

fn slicecpy<T: Clone>(target: &mut [T], source: &[T], n_t: usize) {
    target[..n_t].clone_from_slice(&source[..n_t]);
}

/// Should mark beginnig of string.
#[derive(Debug)]
enum AddBos {
    No,
    Yes,
}

/// Should mark end of string.
#[derive(Debug)]
enum AddEos {
    No,
    Yes,
}

fn main() {
    let args = Args::parse();
    assert!(args.temperature >= 0.0 && args.temperature <= 1.0);
    assert!(args.topp >= 0.0 && args.topp <= 1.0);
    debug_eprintln!("{args:?}");

    let transformer = Transformer::build(&args.checkpoint_path);
    assert!(args.steps <= transformer.config.seq_len);
    let tokenizer = Tokenizer::build(&args.tokenizer_path, transformer.config.vocab_size);
    let sampler = Sampler::new(
        transformer.config.vocab_size,
        args.temperature,
        args.topp,
        args.rng_seed,
    );

    generate(&transformer, &tokenizer, &sampler, args.prompt, args.steps);
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 'z')]
    tokenizer_path: String,
    /// 0.0 = greedy deterministic. 1.0 = original. don't set higher
    #[arg(short = 't', default_value_t = 1.0)]
    temperature: f32,
    /// top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    #[arg(short = 'p', default_value_t = 0.9)]
    topp: f32,
    /// number of steps to run for
    #[arg(short = 'n', default_value_t = 256)]
    steps: usize,
    /// prompt string
    #[arg(short = 'i', default_value = "")]
    prompt: String,
    /// seed rng with time by default
    #[arg(short='s', default_value_t=default_seed())]
    rng_seed: u128,

    checkpoint_path: String,
}

fn default_seed() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::debug_eprintln;
    use crate::{AddBos, Tokenizer, matmul, rmsnorm, slicecpy};
    use std::sync::{LazyLock, Mutex};

    // TOKENIZER tries to share the Tokenizer across test cases.
    thread_local! { // Tokenizer is not Sync, works with thread_local.
        static TOKENIZER: LazyLock<Mutex<Tokenizer>> =
            LazyLock::new(|| Mutex::new({
                let p = "tokenizer.bin";
                debug_eprintln!("Loading {p}");
                Tokenizer::build(p, 32000)}));
    }

    // The tests are 1:1 with the original C tests.
    #[test]
    fn test_tokenizer_0() {
        assert_encoding("", vec![1]);
    }

    #[test]
    fn test_tokenizer_1() {
        assert_encoding(
            "I believe the meaning of life is",
            vec![1, 306, 4658, 278, 6593, 310, 2834, 338],
        );
    }

    #[test]
    fn test_tokenizer_2() {
        assert_encoding(
            "Simply put, the theory of relativity states that ",
            vec![
                1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871,
            ],
        );
    }

    #[test]
    fn test_tokenizer_3() {
        assert_encoding(
            "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ",
            vec![
                1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13,
                13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871,
            ],
        );
    }

    #[test]
    fn test_tokenizer_4() {
        assert_encoding(
            "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>",
            vec![
                1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301,
                449, 276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440,
                29878, 1318, 13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639,
                28754, 13, 4706, 923, 968, 1149,
            ],
        );
    }

    #[test]
    fn test_rms() {
        let mut o: Vec<f32> = vec![0., 0., 0., 0.];
        let x: Vec<f32> = vec![1., 2., 3., 4.];
        let weight: Vec<f32> = vec![0.1, 1.0, 10.0, 100.0];
        let size = 4_usize;

        rmsnorm(&mut o, &x, &weight, size);
        let rms = ((((1 * 1) + (2 * 2) + (3 * 3) + (4 * 4)) as f32) / 4.0).sqrt();
        let expected = [0.1 / rms, 2.0 / rms, 30.0 / rms, 400.0 / rms];

        assert_eq!(expected.len(), o.len());
        for i in 0..o.len() {
            assert!(
                (o[i] - expected[i]).abs() < 1e-4,
                "actual {o:?}\nexpected {expected:?}"
            );
        }
    }

    #[test]
    fn test_matmul() {
        let w: Vec<f32> = vec![
            0.1, 0.2, 0.3, // r0
            0.4, 0.5, 0.6, // r1
        ];
        let x: Vec<f32> = vec![
            1.0, //r0
            2.0, //r1
            3.0, //r2
        ];

        let mut xout: Vec<f32> = vec![0.0; 2];

        matmul(&mut xout, &x, &w, 3, 2);
        assert_eq!(
            xout,
            vec![
                0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0, //r0
                0.4 * 1.0 + 0.5 * 2.0 + 0.6 * 3.0, //r1
            ]
        );
        assert_eq!(xout.len(), 2);
    }

    #[test]
    fn test_slicecpy() {
        let mut t: Vec<f32> = vec![0.0; 3];
        let s: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        slicecpy(&mut t[..], &s[2..], 3);
        assert_eq!(t, vec![3.0, 4.0, 5.0]);
    }

    fn assert_encoding(text: &str, expected_tokens: Vec<usize>) {
        TOKENIZER.with(|tokenizer| {
            let tokenizer = tokenizer.lock().unwrap();
            let actual = tokenizer.encode(text, AddBos::Yes, crate::AddEos::No);
            assert_eq!(
                actual,
                expected_tokens,
                "actual tokenization:\n{}\n\nexpected tokenization: \n{}",
                debug_tokenization(&tokenizer, &actual),
                debug_tokenization(&tokenizer, &expected_tokens),
            );
        });
    }

    fn debug_tokenization(tokenizer: &Tokenizer, token_ids: &Vec<usize>) -> String {
        let mut out = String::new();
        for (i, tok_id) in token_ids.iter().enumerate() {
            let tok_id = *tok_id;
            let vocab = tokenizer.vocab.get(tok_id).unwrap();
            let vocab_score = tokenizer.vocab_scores.get(tok_id).unwrap();
            let s = format!("{i}\t{vocab:?}\t{vocab_score}\n");
            out.push_str(&s);
        }
        out
    }
}
