use clap::Parser;
use memmap2::{Mmap, MmapOptions};
use std::{
    fs::{File, metadata},
    io::{BufReader, Read},
    rc::Rc,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone)]
struct TokenIndex {
    token_str: Rc<String>,
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

impl Tokenizer {
    fn build(path: &str, vocab_size: usize) -> Tokenizer {
        let byte_pieces: [Box<String>; 256] =
            std::array::from_fn(|i| Box::new((i as u8 as char).to_string()));
        //dbg!(&byte_pieces);
        let mut vocab: Vec<Rc<String>> = (0..vocab_size).map(|_| Rc::new("".to_string())).collect();
        let mut vocab_scores = vec![0f32; vocab_size];
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);

        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).unwrap();
        let max_token_length = i32::from_le_bytes(buf) as usize;
        //dbg!(max_token_length);

        for i in 0..vocab_size {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).unwrap();
            let score = f32::from_le_bytes(buf);
            //dbg!(i, score);
            vocab_scores[i] = score;

            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).unwrap();
            let len: i32 = i32::from_le_bytes(buf);
            //dbg!(i, &len);

            let mut buf: Vec<u8> = vec![0u8; len as usize];
            reader.read_exact(&mut buf).unwrap();
            let s = String::from_utf8(buf).unwrap();
            // dbg!(i, score, &len, &s);
            // dbg!(score);
            // eprintln!("{i} {score} {len} {s}");
            vocab[i] = Rc::new(s);
        }

        let mut sorted_vocab: Vec<TokenIndex> = (0..vocab_size)
            .map(|id| TokenIndex {
                token_str: Rc::clone(&vocab[id]),
                id,
            })
            .collect();
        sorted_vocab.sort_by(|a, b| a.token_str.partial_cmp(&b.token_str).unwrap());

        for s in &sorted_vocab {
            eprintln!("{}, {}", s.token_str, s.id)
        }

        Tokenizer {
            vocab_size,
            vocab,
            vocab_scores,
            sorted_vocab,
            byte_pieces,
            max_token_length,
        }
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
    state: RunState,
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
        let state = RunState::new(&config);
        Transformer {
            config,
            state,
            weights: checkpoint.transformer_weights,
            file: checkpoint.file,
        }
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
            eprintln!("{config_deser:?}");
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
    /// same, but inside a residual branch (dim,)
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

fn generate(
    transformer: &Transformer,
    tokenizer: &Tokenizer,
    sampler: &Sampler,
    prompt: String,
    steps: usize,
) {
    // encode the (string) prompt into tokens sequence
    //num_prompt_tokens: usize = 0;
    //int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    //encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    //if (num_prompt_tokens < 1) {
    //    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    //    exit(EXIT_FAILURE);
    //}
    todo!("HERE")
    /*

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

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

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
     */
}

enum BosEos {
    /// Beginning of string
    BOS,
    // Neither BOS nor EOS
    MID,
    /// End of string
    EOS,
}

/// encode the string text (input) into an upper-bound preallocated tokens[] array
/// bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
fn encode(t: &Tokenizer, text: &str, bos_eos: BosEos) {
    // void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
    // int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

    // TODO return tokens, return n_tokens
    /*
        /// // create a temporary buffer that will store merge candidates of always two consecutive tokens
        /// // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
        /// char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
        /// size_t str_len = 0;

        /// // start at 0 tokens
        /// *n_tokens = 0;

        // TODO add BOS
        // add optional BOS (=1) token, if desired
        if (bos) tokens[(*n_tokens)++] = 1;

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing

        // TODO Add dummy prefix
        if (text[0] != '\0') {
            int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
            tokens[(*n_tokens)++] = dummy_prefix;
        }

        /// // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
        /// // Code point ↔ UTF-8 conversion
        /// // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
        /// // U+0000	U+007F	    0xxxxxxx
        /// // U+0080	U+07FF	    110xxxxx	10xxxxxx
        /// // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
        /// // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

        // process the raw (UTF-8) byte sequence of the input string
        for (char *c = text; *c != '\0'; c++) {

            /// // reset buffer if the current byte is ASCII or a leading byte
            /// // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
            /// // 0x80 is 10000000
            /// // in UTF-8, all continuation bytes start with "10" in first two bits
            /// // so in English this is: "if this byte is not a continuation byte"
            /// if ((*c & 0xC0) != 0x80) {
            ///     // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            ///     // => reset our location, as we're starting a new UTF-8 codepoint
            ///     str_len = 0;
            /// }

            /// // append the current byte to the buffer
            /// str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
            /// str_buffer[str_len] = '\0';

            /// // while the next character is a continuation byte, continue appending
            /// // but if there are too many of them, just stop to avoid overruning str_buffer size.
            /// if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            ///     continue;
            /// }

            // ok c+1 is not a continuation byte, so we've read in a full codepoint
            // decode codepoint by codepoint
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens[(*n_tokens)++] = id;
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for (int i=0; i < str_len; i++) {
                    tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
                }
            }
            str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (1) {
            float best_score = -1e10;
            int best_id = -1;
            int best_idx = -1;

            for (int i=0; i < (*n_tokens-1); i++) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
                int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
                if (id != -1 && t->vocab_scores[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = t->vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx+1; i < (*n_tokens-1); i++) {
                tokens[i] = tokens[i+1];
            }
            (*n_tokens)--; // token length decreased
        }

        // add optional EOS (=2) token, if desired
        if (eos) tokens[(*n_tokens)++] = 2;

        free(str_buffer);
    }

         */
}

fn main() {
    let args = Args::parse();
    assert!(args.temperature >= 0.0 && args.temperature <= 1.0);
    assert!(args.topp >= 0.0 && args.topp <= 1.0);
    eprintln!("{args:?}");

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
